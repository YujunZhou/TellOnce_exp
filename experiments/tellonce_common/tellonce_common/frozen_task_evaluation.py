"""Reopen immutable evaluation evidence without a solver, learner or live files.

A task archive only means that task's native session has closed. The phase
owner must still finish all interaction and learning before scoring admission.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

from .codex_task_session import CodexTaskSession
from .claude_task_session import ClaudeTaskSession
from .deferred_scoring import _write
from .native_delivery_evidence import NativeDeliveryEvidence
from .provenance import content_hash
from .task_artifacts import TaskArtifacts
from .task_evaluation_context import TaskEvaluationContext


def _directory(path):
    if (not isinstance(path,Path) or not path.is_absolute() or path!=path.resolve() or
            any((parent/'.git/HEAD').is_file() or (parent/'.git').is_file() for parent in (path,*path.parents))):
        raise ValueError('evaluation archive must be a canonical private directory outside source repositories')


def _digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _read(path,limit):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('immutable evaluation evidence must be a regular file')
        data=stream.read(limit+1)
    if len(data)>limit:
        raise ValueError('immutable evaluation evidence exceeded its bound')
    return json.loads(data)


class FrozenTaskArtifacts(TaskArtifacts):
    def __init__(self, *, directory, workspace, metadata, baseline):
        _directory(directory)
        if not isinstance(workspace,Path) or not workspace.is_absolute() or workspace!=workspace.resolve():
            raise ValueError('original canonical task workspace identity required')
        self.directory,self.workspace=directory,workspace
        self.file_budget,self.total_budget,self.entry_budget=(metadata[k] for k in
            ('file_budget','total_budget','entry_budget'))
        if (any(type(value) is not int or value<=0 for value in
                (self.file_budget,self.total_budget,self.entry_budget)) or not _digest(metadata['baseline_hash'])):
            raise ValueError('invalid archived artifact budgets or original baseline binding')
        self._baseline_hash=metadata['baseline_hash']
        self._root,self._last,self._public_updates=None,None,{}
        self.validate_baseline(baseline)


class FrozenOperationEvidence(NativeDeliveryEvidence):
    def __init__(self, *, directory, task_id, session_id, native_thread_id):
        _directory(directory)
        self.directory,self.task_id,self.session_id,self.native_thread_id=directory,task_id,session_id,native_thread_id

    def observe(self,*args,**kwargs):
        raise ValueError('frozen operation evidence cannot record new operations')

    def capture(self,*args,**kwargs):
        raise ValueError('frozen operation evidence cannot create new snapshots')

    def load(self,digest,*,artifact_hash,user_event_id):
        if not _digest(digest):
            raise ValueError('operation evidence requires its captured content identity')
        value=_read(self.directory/(digest+'.json'),2*1024*1024)
        if (content_hash(value)!=digest or value.get('schema')!=1 or
                value.get('task_id')!=self.task_id or value.get('session_id')!=self.session_id or
                value.get('native_thread_id')!=self.native_thread_id or value.get('artifact_hash')!=artifact_hash or
                value.get('user_event_id')!=user_event_id):
            raise ValueError('frozen operations do not match the captured task delivery')
        if 'request_scope' in value:
            from .operation_scope import validate_request_scope
            validate_request_scope(value['request_scope'], event_id=user_event_id, events=value.get('events'))
        return value


def _deliveries(events):
    for event in events:
        if event['type']=='answer_delivered':
            yield event['artifact_hash'],event['execution_evidence_hash'],event['event_id']
            if event['before_repair_artifact_hash']:
                yield event['before_repair_artifact_hash'],event['before_repair_execution_evidence_hash'],event['event_id']


def seal_task_evaluation(owner, *, directory, binding):
    _directory(directory)
    if (not isinstance(owner,TaskEvaluationContext) or owner._frozen or not _digest(binding) or
            not isinstance(owner.runtime,(CodexTaskSession,ClaudeTaskSession)) or not getattr(owner.runtime,'_closed',False) or
            owner.runtime.artifacts._root is not None or directory.exists()):
        raise ValueError('archive only a closed native task into fresh private evidence storage')
    artifacts,operations=owner.runtime.artifacts,owner.runtime.execution_evidence
    directory.mkdir(mode=0o700)
    for name in ('artifacts','operations'):
        (directory/name).mkdir(mode=0o700)
    captured,executed=set(),set()
    for artifact,operation,event in _deliveries(owner._events):
        if not _digest(artifact) or not _digest(operation):
            raise ValueError('archive requires actual file and operation snapshots for every delivery')
        snapshot=artifacts.load(artifact)
        proof=operations.load(operation,artifact_hash=artifact,user_event_id=event)
        if artifact not in captured:
            _write(directory/'artifacts'/(artifact+'.json'),snapshot)
            captured.add(artifact)
        if operation not in executed:
            _write(directory/'operations'/(operation+'.json'),proof)
            executed.add(operation)
    value={'schema':1,'namespace':owner.identity.namespace,'binding':binding,
        'unit_hashes':{key:content_hash(asdict(unit)) for key,unit in owner.units.items()},
        'task_id':owner.task_id,'session_id':owner.session_id,'native_thread_id':operations.native_thread_id,
        'original_workspace':str(artifacts.workspace),
        'artifacts':{'file_budget':artifacts.file_budget,'total_budget':artifacts.total_budget,
                     'entry_budget':artifacts.entry_budget,'baseline_hash':artifacts._baseline_hash},
        'events':owner._events,'artifact_hashes':sorted(captured),'operation_hashes':sorted(executed)}
    _write(directory/'task-evaluation.json',value)
    owner._frozen=True
    return content_hash(value)


def open_task_evaluation(*,directory,expected_digest,identity,units,binding,baseline):
    _directory(directory)
    if not _digest(expected_digest) or not _digest(binding):
        raise ValueError('evaluation recovery requires the externally recorded task seal and source binding')
    value=_read(directory/'task-evaluation.json',256*1024*1024)
    owner=TaskEvaluationContext(identity=identity,units=units)
    if (content_hash(value)!=expected_digest or value.get('schema')!=1 or
            value.get('namespace')!=identity.namespace or value.get('binding')!=binding or
            value.get('task_id')!=owner.task_id or
            value.get('unit_hashes')!={key:content_hash(asdict(unit)) for key,unit in owner.units.items()}):
        raise ValueError('frozen evaluation transcript or source identity changed')
    for key,subdirectory in (('artifact_hashes','artifacts'),('operation_hashes','operations')):
        hashes=value[key]
        if (not isinstance(hashes,list) or any(not _digest(digest) for digest in hashes) or
                len(set(hashes))!=len(hashes) or
                {path.name for path in (directory/subdirectory).iterdir()}!={digest+'.json' for digest in hashes}):
            raise ValueError('frozen evaluation evidence was added or removed')
    for event in value['events']:
        owner.observe(event)
    if owner.session_id!=value['session_id']:
        raise ValueError('frozen transcript does not match its task session')
    artifacts=FrozenTaskArtifacts(directory=directory/'artifacts',workspace=Path(value['original_workspace']),
                                  metadata=value['artifacts'],baseline=baseline)
    operations=FrozenOperationEvidence(directory=directory/'operations',task_id=owner.task_id,
        session_id=owner.session_id,native_thread_id=value['native_thread_id'])
    used_artifacts,used_operations=set(),set()
    for artifact,operation,event in _deliveries(owner._events):
        artifacts.load(artifact)
        operations.load(operation,artifact_hash=artifact,user_event_id=event)
        used_artifacts.add(artifact)
        used_operations.add(operation)
    if used_artifacts!=set(value['artifact_hashes']) or used_operations!=set(value['operation_hashes']):
        raise ValueError('frozen evidence does not belong exactly to its recorded deliveries')
    owner.runtime=SimpleNamespace(artifacts=artifacts,execution_evidence=operations)
    owner._frozen=True
    return owner
