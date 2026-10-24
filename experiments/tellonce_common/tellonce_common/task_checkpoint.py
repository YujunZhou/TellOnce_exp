"""Atomic task/round checkpoints; mutable memory and progress commit together.

Checkpoint at completed round/task boundaries with background learning quiescent. Immutable
completed task evidence and text-attempt receipts stay at their original paths.
Restoration requires the caller's exclusive run/worker ownership.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import uuid

from .atomic_io import atomic_write_bytes, fsync_parent
from .provenance import canonical_json_bytes, content_hash, file_hash


class CheckpointError(RuntimeError):
    pass


class TaskCheckpoint:
    def __init__(self, run, *, namespace, binding, phase, task_directories, test_scope=None):
        self.run=Path(run)
        if self.run.resolve()!=self.run or not self.run.is_dir() or phase not in {'training','test'}:
            raise CheckpointError('canonical existing run and phase required')
        if not isinstance(namespace,str) or not namespace or not isinstance(binding,str) or len(binding)!=64:
            raise CheckpointError('checkpoint identity required')
        self.phase=phase
        self.tasks=dict(task_directories)
        if (not self.tasks or any(not isinstance(k,str) or not k for k in self.tasks) or
                len(set(self.tasks.values()))!=len(self.tasks)):
            raise CheckpointError('unique ordered task directories required')
        for name in self.tasks.values():
            if not isinstance(name,str) or Path(name).name!=name or not name.startswith('task-'):
                raise CheckpointError('task must be a direct owned run directory')
        self.identity={'schema':1,'namespace':namespace,'binding':binding,'phase':phase,'tasks':self.tasks}
        self.root=self.run/'recovery'/phase
        self.abandoned=self.run/'recovery'/'abandoned'
        if self.root.resolve()!=self.root:raise CheckpointError('checkpoint root redirected')
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.mutable=[phase+'-phase',phase+'-queue','roles/roles.json']
        self.mutable += ['roles/'+role+'/journal' for role in
                         ('feedback','memory','preference','appeal','execution_check')]
        self.mutable += ['roles/'+role+'/direct-luna-journal' for role in ('feedback','memory')]
        self.mutable.append('roles/execution_check/runtime-journal')
        self.mutable += ['roles/'+role+'/api-luna-low-v1-journal' for role in
                         ('feedback','memory','preference','appeal','execution_check')]
        if phase=='training':self.mutable.append('method')
        if test_scope is not None:
            if phase != 'test' or list(self.tasks) != [test_scope]:
                raise CheckpointError('parallel checkpoint owns exactly one test task')
            scope = 'parallel-test/'+content_hash(test_scope)
            self.root = self._path('recovery/test-tasks/'+content_hash(test_scope))
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.abandoned=self.root/'abandoned'
            self.mutable = [scope+'/phase', scope+'/queue', scope+'/roles/roles.json']
            self.mutable += [scope+'/roles/'+role+'/journal' for role in
                            ('feedback','memory','preference','appeal','execution_check')]
            self.mutable += [scope+'/roles/'+role+'/direct-luna-journal' for role in ('feedback','memory')]
            self.mutable.append(scope+'/roles/execution_check/runtime-journal')
            self.mutable += [scope+'/roles/'+role+'/api-luna-low-v1-journal' for role in
                            ('feedback','memory','preference','appeal','execution_check')]
            # A test checkpoint never snapshots or restores the shared method.
        self.restored=False
        self.value=None

    def _path(self, relative):
        path=self.run/relative
        if path.resolve()!=path or not path.is_relative_to(self.run):
            raise CheckpointError('checkpoint path redirected')
        return path

    def current(self):
        pointer=self.root/'CURRENT.json'
        if not pointer.exists():return None
        if pointer.is_symlink() or pointer.stat().st_size>64*1024**2:
            raise CheckpointError('invalid checkpoint pointer')
        value=json.loads(pointer.read_text())
        if (not isinstance(value,dict) or not isinstance(value.get('completed'),list) or
                not isinstance(value.get('receipts'),dict) or set(value['receipts'])!=set(value['completed']) or
                not isinstance(value.get('inventory'),dict)):
            raise CheckpointError('invalid checkpoint metadata')
        if value.get('identity')!=self.identity or value.get('completed')!=list(self.tasks)[:len(value.get('completed',[]))]:
            raise CheckpointError('checkpoint identity or task prefix differs')
        name=value.get('archive','')
        if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.tar.gz'):
            raise CheckpointError('invalid checkpoint archive')
        path=self.root/name
        if path.is_symlink() or file_hash(path)!=value.get('sha256'):
            raise CheckpointError('checkpoint archive corrupted')
        return value

    @contextmanager
    def _quiet(self, memory):
        active=memory is not None and getattr(memory,'_thread',None) is not None
        if active:memory._stop_background()
        if memory is not None:memory.assert_healthy()
        try:yield
        finally:
            if active and not memory._closed:
                memory._thread=None;memory._stop.clear();memory.start_background();memory._wake.set()

    def _partial_paths(self, partial):
        if partial is None:
            return []
        name = self.tasks[partial['task_id']]
        return [name+'/'+p for p in ('work', 'captures', 'workspace-state', 'memory',
            'native/cli/sessions', 'native/cli/projects', 'native/execution-evidence', 'unit-resume.json')]

    def save(self, completed, *, memory=None, partial=None):
        completed=list(completed)
        if completed!=list(self.tasks)[:len(completed)]:
            raise CheckpointError('only a complete ordered task prefix can commit')
        if partial is not None and (len(completed)>=len(self.tasks) or
                partial.get('task_id')!=list(self.tasks)[len(completed)] or
                type(partial.get('completed_units')) is not int or partial['completed_units']<1):
            raise CheckpointError('round checkpoint must belong to the next uncommitted task')
        previous=self.current()
        if previous is not None and len(previous['completed'])>len(completed):
            raise CheckpointError('checkpoint cannot move backwards')
        if previous is not None and previous.get('partial') and len(previous['completed'])==len(completed):
            if partial is None or partial['completed_units']<previous['partial']['completed_units']:
                raise CheckpointError('round checkpoint cannot move backwards')
        if shutil.disk_usage(self.run).free<5*1024**3:
            raise CheckpointError('insufficient disk space for checkpoint')
        name=uuid.uuid4().hex+'.tar.gz';path=self.root/name
        try:
            with self._quiet(memory):
                inventory={};present=[];total_bytes=0
                with tarfile.open(path,'w:gz',compresslevel=1,dereference=True) as archive:
                    for relative in self.mutable+self._partial_paths(partial):
                        source=self._path(relative)
                        if not source.exists():continue
                        present.append(relative)
                        paths=[source]+sorted(source.rglob('*')) if source.is_dir() else [source]
                        for item in paths:
                            if item.is_symlink() or not (item.is_file() or item.is_dir()):
                                raise CheckpointError('mutable checkpoint contains unsupported file')
                            key=item.relative_to(self.run).as_posix()
                            if item.is_file():
                                inventory[key]=file_hash(item);total_bytes+=item.stat().st_size
                                if total_bytes>8*1024**3 or len(inventory)>100000:
                                    raise CheckpointError('checkpoint exceeds bounded state size')
                            archive.add(item,arcname=key,recursive=False)
                with path.open('rb') as f:os.fsync(f.fileno())
                receipts={}
                for task in completed:
                    result=self._path(self.tasks[task])/'task-result.json'
                    if not result.is_file():raise CheckpointError('completed task has no receipt')
                    receipts[task]=file_hash(result)
                value={'identity':self.identity,'completed':completed,'receipts':receipts,'present':present,
                    'inventory':inventory,'total_bytes':total_bytes,'archive':name,'sha256':file_hash(path),'created_at':time.time(),
                    'memory_event_ids':sorted(getattr(memory,'_event_ids',())) if memory is not None else []}
                if partial is not None:
                    value['partial'] = partial
                atomic_write_bytes(self.root/'CURRENT.json',canonical_json_bytes(value))
            # Keep the newest and its predecessor, never a growing checkpoint history.
            keep={name}
            if previous:keep.add(previous['archive'])
            for old in self.root.glob('*.tar.gz'):
                if old.name not in keep and not old.is_symlink():old.unlink()
            self.value=value
            return content_hash(value)
        except BaseException:
            # A crash after pointer publication must not remove its archive.
            try:published=json.loads((self.root/'CURRENT.json').read_text()).get('archive')==name
            except (OSError,ValueError):published=False
            if not published:path.unlink(missing_ok=True)
            raise

    def restore(self):
        value=self.current()
        if value is None:return None
        partial=value.get('partial')
        if partial is not None and (len(value['completed'])>=len(self.tasks) or
                partial.get('task_id')!=list(self.tasks)[len(value['completed'])] or
                type(partial.get('completed_units')) is not int or partial['completed_units']<1):
            raise CheckpointError('invalid partial-task checkpoint')
        mutable=self.mutable+self._partial_paths(partial)
        for task,digest in value['receipts'].items():
            if file_hash(self._path(self.tasks[task])/'task-result.json')!=digest:
                raise CheckpointError('committed task evidence changed')
        total=value.get('total_bytes')
        if type(total) is not int or total<0 or total>8*1024**3:
            raise CheckpointError('invalid checkpoint size')
        if shutil.disk_usage(self.run).free<total+5*1024**3:
            raise CheckpointError('insufficient disk space to restore checkpoint')
        stage=self.root/('restore-'+uuid.uuid4().hex);stage.mkdir(mode=0o700)
        try:
            with tarfile.open(self.root/value['archive'],'r:gz') as archive:
                names=set();total_bytes=0
                for item in archive:
                    parts=Path(item.name).parts
                    if (item.name in names or Path(item.name).is_absolute() or '..' in parts or
                            not any(item.name==p or item.name.startswith(p+'/') for p in mutable) or
                            not (item.isfile() or item.isdir())):
                        raise CheckpointError('unsafe checkpoint archive member')
                    total_bytes+=item.size
                    if total_bytes>8*1024**3 or len(names)>=150000:
                        raise CheckpointError('checkpoint extraction exceeds bound')
                    names.add(item.name);dest=stage/item.name
                    if item.isdir():dest.mkdir(parents=True,exist_ok=True,mode=0o700)
                    else:
                        dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                        with archive.extractfile(item) as src,dest.open('xb') as out:shutil.copyfileobj(src,out)
                        dest.chmod((item.mode & 0o777) | 0o600)
            actual={p.relative_to(stage).as_posix():file_hash(p) for p in stage.rglob('*') if p.is_file()}
            if actual!=value['inventory']:raise CheckpointError('restored checkpoint content differs')
            # All contents verified before changing live state. Repeating after a
            # crash during installation uses the same immutable checkpoint.
            atomic_write_bytes(self.root/'RESTORING.json',canonical_json_bytes({'archive':value['archive']}))
            # Move unfinished runtime directories away before restoring their
            # durable subset. Never retain files produced after the saved round.
            for task,name in self.tasks.items():
                path=self._path(name)
                if task not in value['completed'] and path.exists():
                    abandoned=self._path(self.abandoned.relative_to(self.run));abandoned.mkdir(exist_ok=True,mode=0o700)
                    os.replace(path,abandoned/(name+'-'+uuid.uuid4().hex))
            for relative in mutable:
                target=self._path(relative);source=stage/relative
                if target.is_dir():shutil.rmtree(target)
                elif target.exists():target.unlink()
                if source.exists():
                    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700);os.replace(source,target)
            for relative in mutable:
                if (self.run/relative).parent.exists():fsync_parent(self.run/relative)
            (self.root/'RESTORING.json').unlink()
            from .runtime_retention import compact_abandoned
            compact_abandoned(self.abandoned)
            self.restored=True;self.value=value
            return value
        finally:shutil.rmtree(stage,ignore_errors=True)

    def completed_receipt(self, task):
        if self.value and task in self.value['completed']:
            # Receipt validators use canonical JSON content hashes, rather than
            # raw file hashes used to protect the checkpoint's immutable input.
            return content_hash(json.loads((self._path(self.tasks[task])/'task-result.json').read_text()))
        return None
