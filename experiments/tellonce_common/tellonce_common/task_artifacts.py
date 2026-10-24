"""Capture changed public task files without following links outside the task.

This is a bounded text-task adapter, not a disk image of arbitrary environments.
Binary/large/unreadable outputs remain explicit omissions. Native graders must
reject unsupported or incomplete required artifacts rather than invent scores.
"""
from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import unquote

from .provenance import content_hash


@dataclass(frozen=True)
class ArtifactCapture:
    texts: dict[str,str]
    omissions: tuple[dict, ...]
    deleted: tuple[str, ...]

    @property
    def complete(self):
        return not self.omissions


def _signature(value):
    return (value.st_dev,value.st_ino,value.st_mode,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def _artifact_path(value):
    return (isinstance(value, str) and bool(value) and not value.startswith('/') and '\\' not in value and
            '\x00' not in value and all(part not in {'', '.', '..'} for part in value.split('/')))


def _file_description(files):
    return {path: {'mode': item['mode'], 'sha256': hashlib.sha256(item['data']).hexdigest()}
            for path, item in files.items()}


def referenced_output_files(text, files):
    """Resolve explicit references against captured names, never read a path.

    Preserve mention order. A bare basename is usable only if unique; a relative
    path never falls back to a different suffix. Absolute links may match one
    unique captured suffix because native agents often include the workspace.
    """
    names = tuple(files)
    mentions = []
    structured = []
    for match in re.finditer(r'\[[^\]\n]*\]\((<[^>]+>|[^)]+)\)|(?<!`)`([^`\n]+)`(?!`)', text):
        structured.append(match.span())
        target = unquote((match[1] or match[2]).strip().strip('<>'))
        if '://' in target or '\\' in target:
            continue
        target = re.sub(r'(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)$', '', target)
        target = target.removeprefix('./')
        if '..' in target.split('/'):
            continue
        if target in files:
            matches = [target]
        elif target.startswith('/'):
            matches = [path for path in names if target.endswith('/' + path)]
        elif '/' not in target:
            matches = [path for path in names if path.rsplit('/', 1)[-1] == target]
        else:
            matches = []
        if len(matches) == 1:
            mentions.append((match.start(), matches[0]))
    # Plain exact paths also count, while path prefixes and .bak suffixes do
    # not. Only known relative names are searched, so this cannot expose IO.
    for path in names:
        for match in re.finditer(r'(?<![\w./-])' + re.escape(path) + r'(?![\w/-]|\.(?=[\w/.-]))', text):
            if not any(start <= match.start() < end for start, end in structured):
                mentions.append((match.start(), path))
                break
    return list(dict.fromkeys(path for _, path in sorted(mentions)))


class TaskArtifacts:
    def __init__(self, *, workspace: Path, directory: Path, file_budget: int = 256*1024,
                 total_budget: int = 4*1024*1024, entry_budget: int = 5000, resume=None):
        if (any(not p.is_absolute() or p != p.resolve() for p in (workspace,directory)) or
                directory == workspace or workspace in directory.parents or
                min(file_budget,total_budget,entry_budget) <= 0):
            raise ValueError('canonical public workspace, separate private storage and positive budgets required')
        self.workspace, self.directory = workspace,directory
        self.file_budget,self.total_budget,self.entry_budget = file_budget,total_budget,entry_budget
        self._root = os.open(workspace,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        self._last = None
        self._public_updates = {}
        try:
            directory.mkdir(mode=0o700,exist_ok=resume is not None)
            if resume is not None:
                self._baseline={k:tuple(v) for k,v in resume['baseline'].items()}
                self._baseline_hash=resume['baseline_hash']
                self._public_updates={k:tuple(v) for k,v in resume['public_updates'].items()}
                self._last=resume['last']
                return
            self._baseline, _, omissions, _ = self._scan(baseline=True)
            if omissions:
                raise ValueError('initial task inventory is incomplete; prepare a supported workspace')
            self._baseline_hash = content_hash({path: {'mode': row[-2], 'sha256': row[-1]}
                                               for path, row in self._baseline.items()})
        except Exception:
            self.close()
            raise

    def _scan(self, *, baseline=False):
        if self._root is None:
            raise ValueError('task artifact capture is closed')
        inventory, files, omissions = {},{},[]
        directories, directory_signatures = {}, {}
        entries, used = 0,0
        def error(exc):
            # Do not expose exception paths outside the declared workspace.
            omissions.append({'path':'.','reason':'directory_unreadable'})
        for directory, dirs, names, dirfd in os.fwalk('.',dir_fd=self._root,follow_symlinks=False,onerror=error):
            relative_directory = str(Path(directory))
            try:
                before = os.fstat(dirfd)
                ordered_entries = os.listdir(dirfd)
                if (_signature(os.fstat(dirfd)) != _signature(before) or
                        set(ordered_entries) != set(dirs + names)):
                    raise ValueError('directory changed during enumeration')
            except (OSError, ValueError):
                omissions.append({'path': relative_directory, 'reason': 'unreadable_or_unstable'})
                dirs[:] = []
                continue
            dirs.sort()
            names.sort()
            entries += len(dirs)+len(names)
            if entries > self.entry_budget:
                omissions.append({'path':'.','reason':'entry_budget_exceeded'})
                break
            directories[relative_directory] = {
                'mode': stat.S_IMODE(before.st_mode) & 0o777,
                'mtime_ns': before.st_mtime_ns, 'entries': ordered_entries}
            directory_signatures[relative_directory] = _signature(before)
            # fwalk will not descend through directory symlinks, but they must
            # still appear as uninspectable output when created by the solver.
            linked_dirs = []
            for name in list(dirs):
                try:
                    if stat.S_ISLNK(os.stat(name,dir_fd=dirfd,follow_symlinks=False).st_mode):
                        linked_dirs.append(name)
                        dirs.remove(name)
                except OSError:
                    omissions.append({'path':str(Path(directory)/name),'reason':'entry_unreadable'})
                    dirs.remove(name)
            for name in sorted(names+linked_dirs):
                relative = str(Path(directory)/name)
                try:
                    metadata = os.stat(name,dir_fd=dirfd,follow_symlinks=False)
                    signature = _signature(metadata)
                    inventory[relative] = signature
                    if not stat.S_ISREG(metadata.st_mode):
                        if baseline:
                            omissions.append({'path':relative,'reason':'unsupported_initial_entry'})
                        else:
                            omissions.append({'path':relative,'reason':'non_regular_or_symlink'})
                        continue
                    if metadata.st_size > self.file_budget:
                        omissions.append({'path':relative,'reason':'file_budget_exceeded'})
                        continue
                    if used+metadata.st_size > self.total_budget:
                        omissions.append({'path':relative,'reason':'total_budget_exceeded'})
                        continue
                    fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=dirfd)
                    try:
                        if _signature(os.fstat(fd)) != signature:
                            raise ValueError('file changed before read')
                        with os.fdopen(fd,'rb',closefd=False) as stream:
                            data = stream.read(self.file_budget+1)
                        if (_signature(os.fstat(fd)) != signature or
                                _signature(os.stat(name,dir_fd=dirfd,follow_symlinks=False)) != signature or
                                len(data) != metadata.st_size):
                            raise ValueError('file changed during read')
                    finally:
                        os.close(fd)
                    used += len(data)
                    mode = stat.S_IMODE(metadata.st_mode) & 0o777
                    inventory[relative] = (*signature,mode,hashlib.sha256(data).hexdigest())
                    # Metadata timestamps can remain identical across quick,
                    # same-size writes. Only actual content can prove equality.
                    if baseline:
                        continue
                    initial_unchanged = (relative in self._baseline and
                                         self._baseline[relative][-2:] == inventory[relative][-2:])
                    source_unchanged = (relative not in self._public_updates or
                                        self._public_updates[relative] == inventory[relative][-2:])
                    if initial_unchanged and source_unchanged:
                        continue
                    try:
                        text = data.decode('utf-8')
                        if '\x00' in text:
                            raise UnicodeError('binary content')
                        files[relative] = {'text':text, 'mode':mode}
                    except UnicodeError:
                        # Preserve the actual small binary for native graders,
                        # but never present decoded guesses to rule checkers.
                        files[relative] = {'base64':base64.b64encode(data).decode('ascii'), 'mode':mode}
                        omissions.append({'path':relative,'reason':'binary_not_text_checkable'})
                except (OSError,ValueError):
                    omissions.append({'path':relative,'reason':'unreadable_or_unstable'})
        # Recheck every directory and file after the final content read. This
        # detects background writes, additions and replacements during capture,
        # including mutations to files read earlier in the traversal.
        for path, signature in directory_signatures.items():
            fd = None
            try:
                fd = self._open_directory(path)
                if (_signature(os.fstat(fd)) != signature or
                        os.listdir(fd) != directories[path]['entries']):
                    raise ValueError('directory changed during capture')
            except (OSError, ValueError):
                omissions.append({'path': path, 'reason': 'unreadable_or_unstable'})
            finally:
                if fd is not None:
                    os.close(fd)
        for path, signature in inventory.items():
            fd = None
            try:
                fd = self._open_directory(str(Path(path).parent))
                if _signature(os.stat(Path(path).name, dir_fd=fd, follow_symlinks=False)) != signature[:6]:
                    raise ValueError('file changed after content read')
                if len(signature) == 8:
                    # Some filesystems expose coarse timestamps. Recheck the
                    # bytes too: a same-size write can preserve every stat field.
                    file_fd = os.open(Path(path).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                    with os.fdopen(file_fd, 'rb') as stream:
                        if _signature(os.fstat(stream.fileno())) != signature[:6]:
                            raise ValueError('file replaced before content verification')
                        data = stream.read(self.file_budget + 1)
                        if (_signature(os.fstat(stream.fileno())) != signature[:6] or
                                hashlib.sha256(data).hexdigest() != signature[-1] or
                                _signature(os.stat(Path(path).name, dir_fd=fd, follow_symlinks=False)) != signature[:6]):
                            raise ValueError('file content changed after capture')
            except (OSError, ValueError):
                omissions.append({'path': path, 'reason': 'unreadable_or_unstable'})
            finally:
                if fd is not None:
                    os.close(fd)
        metadata = {'file_mtimes_ns': {path: row[4] for path, row in inventory.items()
                                      if len(row) == 8},
                    'directories': directories, 'directory_signatures': directory_signatures}
        return inventory,files,omissions,metadata

    def _open_directory(self, path):
        """Resolve each directory component without following intermediate links."""
        fd = os.dup(self._root)
        try:
            for component in (() if path == '.' else Path(path).parts):
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd
        except Exception:
            os.close(fd)
            raise

    def _snapshot(self):
        public_updates = self._public_updates
        inventory,files,omissions,metadata = self._scan()
        unknown = [row['path'] for row in omissions if row['reason'] in
                   {'directory_unreadable','entry_unreadable','entry_budget_exceeded','unreadable_or_unstable'}]
        deleted = [path for path in (set(self._baseline) | set(public_updates))-set(inventory)
                   if not any(prefix=='.' or path==prefix or path.startswith(prefix+'/') for prefix in unknown)]
        inputs = sorted(path for path in files if path in public_updates and
                        inventory[path][-2:] == public_updates[path])
        return {'files':files,'deleted':sorted(deleted),'deletions_complete':not unknown,
                'omissions':omissions, 'input_files':inputs, 'baseline_hash':self._baseline_hash,
                'file_mtimes_ns': metadata['file_mtimes_ns'], 'directories': metadata['directories']}

    def record_public_update(self, files: dict):
        """Host-only registration of exact injected bytes, before the next turn.

        Files remain in the frozen reconstruction delta. They are excluded
        from agent-output checks only while matching the trusted source bytes
        and modes. Later edits/deletions are observable agent changes again.
        """
        self._validate_file_map(files)
        inventory, _, omissions, _ = self._scan()
        failed = {row['path'] for row in omissions if row['reason'] != 'binary_not_text_checkable'}
        verified = {}
        for path, item in files.items():
            expected = (item['mode'], hashlib.sha256(item['data']).hexdigest())
            if path not in inventory or path in failed or inventory[path][-2:] != expected:
                raise ValueError('injected public source does not match the actual task file')
            verified[path] = expected
        self._public_updates = {**self._public_updates, **verified}

    @staticmethod
    def _agent_outputs(snapshot):
        inputs = set(snapshot['input_files'])
        return {**snapshot, 'files': {p: v for p, v in snapshot['files'].items() if p not in inputs},
                'omissions': [r for r in snapshot['omissions'] if r['path'] not in inputs]}

    def appeal_state(self) -> str | None:
        """Bind a temporary appeal to complete current public file contents.

        This does not replace the actual delivery snapshot or change _last.
        Incomplete or unreadable captures cannot prove unchanged state. A
        bounded binary file still has an exact content hash and file identity.
        """
        inventory,_,omissions,metadata=self._scan()
        if any(row['reason'] != 'binary_not_text_checkable' for row in omissions):
            return None
        # Include identity/mode/timestamps as well as bytes: replacing a file,
        # chmod, or changing it and later restoring its content is a new state.
        return content_hash({'files': inventory, 'metadata': metadata})

    def capture_delivery(self, text: str) -> str:
        if not isinstance(text,str) or len(text.encode()) > 4*1024*1024:
            raise ValueError('actual bounded delivery text is required')
        snapshot = {'schema':3,'answer':text,**self._snapshot()}
        digest = content_hash(snapshot)
        path = self.directory/(digest+'.json')
        encoded = json.dumps(snapshot,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()
        try:
            fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        except FileExistsError:
            if self.load(digest) != snapshot:
                raise ValueError('existing snapshot differs from the captured artifact')
        else:
            with os.fdopen(fd,'wb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            parent = os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        self._last = snapshot
        return digest

    def capture_artifacts(self, payload: dict) -> ArtifactCapture:
        if payload.get('hook_event_name') == 'Stop':
            if self._last is None:
                raise ValueError('capture the actual Stop draft before checking it')
            snapshot = self._last
        else:
            snapshot = self._snapshot()
        snapshot = self._agent_outputs(snapshot)
        return ArtifactCapture({path:body['text'] for path,body in snapshot['files'].items() if 'text' in body},
            tuple(dict(row) for row in snapshot['omissions']),tuple(snapshot['deleted']))

    def load(self, digest: str) -> dict:
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
            raise ValueError('snapshot identity must be its content SHA256')
        path = self.directory/(digest+'.json')
        fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('snapshot must be a regular private file')
            data = stream.read(8*(self.total_budget+4*1024*1024)+1)
        if len(data) > 8*(self.total_budget+4*1024*1024):
            raise ValueError('stored snapshot exceeded its budget')
        snapshot = json.loads(data)
        if content_hash(snapshot) != digest:
            raise ValueError('captured snapshot content hash mismatch')
        self._validate_snapshot(snapshot)
        return snapshot

    def _validate_file_map(self, files):
        if (not isinstance(files, dict) or len(files) > self.entry_budget or
                any(not _artifact_path(path) or not isinstance(body, dict) or set(body) != {'data', 'mode'} or
                    not isinstance(body['data'], bytes) or len(body['data']) > self.file_budget or
                    type(body['mode']) is not int or not 0 <= body['mode'] <= 0o777 for path, body in files.items()) or
                sum(len(body['data']) for body in files.values()) > self.total_budget):
            raise ValueError('bounded regular artifact files with explicit modes required')

    def validate_baseline(self, baseline: dict):
        self._validate_file_map(baseline)
        if content_hash(_file_description(baseline)) != self._baseline_hash:
            raise ValueError('original baseline does not match the captured task files')

    def _validate_snapshot(self, snapshot):
        fields = {'schema', 'answer', 'files', 'deleted', 'deletions_complete', 'omissions', 'input_files',
                  'baseline_hash', 'file_mtimes_ns', 'directories'}
        if (not isinstance(snapshot, dict) or set(snapshot) != fields or snapshot['schema'] != 3 or
                snapshot['baseline_hash'] != self._baseline_hash or not isinstance(snapshot['answer'], str) or
                len(snapshot['answer'].encode()) > 4 * 1024 * 1024 or not isinstance(snapshot['files'], dict) or
                type(snapshot['deletions_complete']) is not bool or
                any(not isinstance(snapshot[k], list) or any(not _artifact_path(p) for p in snapshot[k]) or
                    len(set(snapshot[k])) != len(snapshot[k]) for k in ('deleted', 'input_files')) or
                not set(snapshot['input_files']) <= set(snapshot['files']) or
                set(snapshot['deleted']) & set(snapshot['files']) or not isinstance(snapshot['omissions'], list) or
                any(not isinstance(row, dict) or set(row) != {'path', 'reason'} or
                    (row['path'] != '.' and not _artifact_path(row['path'])) or not isinstance(row['reason'], str)
                    for row in snapshot['omissions'])):
            raise ValueError('snapshot schema or baseline binding is invalid')
        files = self._decode_files(snapshot['files'])
        self._validate_file_map(files)
        mtimes, directories = snapshot['file_mtimes_ns'], snapshot['directories']
        if (not isinstance(mtimes, dict) or len(mtimes) > self.entry_budget or
                any(not _artifact_path(p) or type(t) is not int or not -(2**63) <= t < 2**63
                    for p, t in mtimes.items()) or
                not isinstance(directories, dict) or len(directories) > self.entry_budget + 1):
            raise ValueError('bounded captured file and directory metadata required')
        total_entries = 0
        for path, row in directories.items():
            if ((path != '.' and not _artifact_path(path)) or not isinstance(row, dict) or
                    set(row) != {'mode', 'mtime_ns', 'entries'} or
                    type(row['mode']) is not int or not 0 <= row['mode'] <= 0o777 or
                    type(row['mtime_ns']) is not int or not -(2**63) <= row['mtime_ns'] < 2**63 or
                    not isinstance(row['entries'], list) or
                    any(not _artifact_path(name) or '/' in name for name in row['entries']) or
                    len(set(row['entries'])) != len(row['entries'])):
                raise ValueError('captured directory metadata is invalid')
            total_entries += len(row['entries'])
        if total_entries > self.entry_budget:
            raise ValueError('captured directory entry budget exceeded')

    @staticmethod
    def _decode_files(files):
        decoded = {}
        for path, body in files.items():
            if not isinstance(body, dict) or set(body) not in ({'text', 'mode'}, {'base64', 'mode'}):
                raise ValueError('snapshot must contain exact text or binary files with modes')
            if 'text' in body:
                if not isinstance(body['text'], str):
                    raise ValueError('snapshot text is invalid')
                data = body['text'].encode('utf-8')
            else:
                if not isinstance(body['base64'], str):
                    raise ValueError('snapshot binary encoding is invalid')
                data = base64.b64decode(body['base64'], validate=True)
            decoded[path] = {'data': data, 'mode': body['mode']}
        return decoded

    def restore(self, digest: str, *, baseline: dict, destination: Path):
        """Rebuild the captured delivery for a private native grader.

        Never reads the later live workspace. The caller then runs the grader
        in an OS sandbox that can access this copy and its private eval assets.
        """
        snapshot = self.load(digest)
        self.validate_baseline(baseline)
        if (not snapshot['deletions_complete'] or
                any(r['reason'] != 'binary_not_text_checkable' for r in snapshot['omissions']) or
                not isinstance(destination, Path) or not destination.is_absolute() or destination != destination.resolve() or
                destination.exists() or destination == self.workspace or self.workspace in destination.parents or
                destination == self.directory or self.directory in destination.parents):
            raise ValueError('complete captured files, exact original baseline and fresh private grade directory required')
        files = {p: dict(body) for p, body in baseline.items() if p not in snapshot['deleted']}
        files.update(self._decode_files(snapshot['files']))
        self._validate_file_map(files)
        for path in files:
            if any(str(parent) in files for parent in Path(path).parents if str(parent) != '.'):
                raise ValueError('snapshot has conflicting file and directory paths')
        directories, mtimes = snapshot['directories'], snapshot['file_mtimes_ns']
        if '.' not in directories or set(mtimes) != set(files) or set(files) & set(directories):
            raise ValueError('captured metadata does not describe the reconstructed files')
        listed = {str(Path(parent) / name) for parent, row in directories.items() for name in row['entries']}
        if listed != set(files) | (set(directories) - {'.'}):
            raise ValueError('captured directory topology does not match the reconstructed files')
        destination.mkdir(mode=0o700)
        pending = ['.']
        while pending:
            parent = pending.pop()
            for name in directories[parent]['entries']:
                path = str(Path(parent) / name)
                target = destination / path
                if path in directories:
                    target.mkdir(mode=0o700)
                    pending.append(path)
                    continue
                body = files[path]
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(body['data'])
                    stream.flush()
                    os.fchmod(stream.fileno(), body['mode'])
                    os.utime(stream.fileno(), ns=(mtimes[path], mtimes[path]))
                    if os.fstat(stream.fileno()).st_mtime_ns != mtimes[path]:
                        raise ValueError('grading filesystem cannot preserve captured file times')
                    os.fsync(stream.fileno())
            if os.listdir(destination / parent) != directories[parent]['entries']:
                raise ValueError('grading filesystem cannot preserve captured enumeration order')
        # Set directories last: creating children changes their parent mtime.
        for path in sorted(directories, key=lambda p: len(Path(p).parts), reverse=True):
            target, row = destination / path, directories[path]
            fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fchmod(fd, row['mode'])
                os.utime(fd, ns=(row['mtime_ns'], row['mtime_ns']))
                if os.fstat(fd).st_mtime_ns != row['mtime_ns']:
                    raise ValueError('grading filesystem cannot preserve captured directory times')
                os.fsync(fd)
            finally:
                os.close(fd)

    def feedback_view(self, digest: str, *, max_chars: int = 32000, current_request: str = '') -> dict:
        """Project frozen public outputs, never the private native grade.

        The adapter must keep hidden evaluation resources outside workspace.
        Whole files are omitted when necessary; coverage is part of the input.
        """
        if type(max_chars) is not int or not 1024<=max_chars<=64000:
            raise ValueError('bounded public artifact projection required')
        if not isinstance(current_request, str):
            raise ValueError('current public user request must be text')
        snapshot=self._agent_outputs(self.load(digest))
        texts={path:body['text'] for path,body in snapshot['files'].items() if 'text' in body}
        coverage={'complete':False,'text_files_total':len(texts),'text_files_omitted':len(texts),
                  'captured_omissions':len(snapshot['omissions']),
                  'omission_details_omitted':len(snapshot['omissions']),
                  'deleted_total':len(snapshot['deleted']),'deleted_details_omitted':len(snapshot['deleted']),
                  'deletions_complete':snapshot['deletions_complete']}
        view={'snapshot_hash':digest,'files':{},'deleted':[],'omissions':[],'coverage':coverage}
        def fits(limit):
            return len(json.dumps(view,ensure_ascii=False,allow_nan=False))<=limit
        # Reserve most of the budget for actual file content. Extremely long
        # paths or huge inventories remain counted without flooding the role.
        for row in snapshot['omissions']:
            view['omissions'].append(dict(row))
            if not fits(max_chars//2):
                view['omissions'].pop()
            else:
                coverage['omission_details_omitted']-=1
        for path in snapshot['deleted']:
            view['deleted'].append(path)
            if not fits(max_chars//2):
                view['deleted'].pop()
            else:
                coverage['deleted_details_omitted']-=1
        # The current delivery must not lose its budget to historical outputs.
        # Use the frozen answer, not a live workspace or later model response.
        delivered = {path: index for index, path in enumerate(
            referenced_output_files(snapshot['answer'], texts))}
        requested = {path: index for index, path in enumerate(
            referenced_output_files(current_request, texts))}
        def priority(item):
            path = item[0]
            if path in delivered:
                return (0, delivered[path], path)
            if path in requested:
                return (1, requested[path], path)
            return (2, 0, path)
        for path,text in sorted(texts.items(), key=priority):
            view['files'][path]=text
            if not fits(max_chars):
                del view['files'][path]
            else:
                coverage['text_files_omitted']-=1
        coverage['complete']=(not coverage['captured_omissions'] and not coverage['text_files_omitted']
                              and not coverage['deleted_details_omitted'] and coverage['deletions_complete'])
        return view

    def close(self):
        if self._root is not None:
            os.close(self._root)
            self._root = None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
