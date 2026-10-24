"""Source-exact public ClawArena files and between-round updates.

The pinned Codex workspace is supplemented with the original public main
conversation, which that projection omits. No evaluator, answer key, native
runtime metadata, private reasoning, or inferred preference is copied. A
partially applied update aborts the task; this is not a resume implementation.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import uuid

from tellonce_common.atomic_io import atomic_write_bytes
from tellonce_common.interaction_loop import RecoverableCallError
from tellonce_common.provenance import canonical_json_bytes, content_hash
from tellonce_common.role_inputs import validate_public_materials
from tellonce_common.task_artifacts import TaskArtifacts


def _relative(value):
    if (not isinstance(value, str) or not value or '\x00' in value or '\\' in value or
            PurePosixPath(value).is_absolute() or any(p in {'', '.', '..'} for p in value.split('/'))):
        raise ValueError('asset paths must be normalized workspace-relative paths')
    return value


@dataclass(frozen=True)
class AssetFile:
    path: str
    data: bytes
    mode: int


@dataclass(frozen=True)
class AssetUpdate:
    update_id: str
    files: tuple[AssetFile, ...]
    append: bool = False


@dataclass(frozen=True)
class ClawAssets:
    scenario_id: str
    initial: tuple[AssetFile, ...]
    updates: tuple[AssetUpdate, ...]

    @property
    def digest(self):
        def describe(files):
            return [{'path': f.path, 'sha256': hashlib.sha256(f.data).hexdigest(), 'mode': f.mode}
                    for f in files]
        return content_hash({'scenario_id': self.scenario_id, 'initial': describe(self.initial),
            'updates': [{'id': u.update_id, 'files': describe(u.files),
                         **({'append': True} if u.append else {})} for u in self.updates]})

    def expand(self, identifiers):
        available = {u.update_id: u for u in self.updates}
        expanded = []
        if not isinstance(identifiers, list) or any(not isinstance(i, str) for i in identifiers):
            raise ValueError('public update identifiers must be an ordered list of strings')
        for identifier in identifiers:
            if identifier in available:
                expanded.append(identifier)
                continue
            number = re.fullmatch(r'(?:U|u|upd|UPD)?([0-9]+)', identifier)
            aliases = ([f'upd{number[1]}_sessions', f'upd{number[1]}_workspace'] if number else [])
            found = [alias for alias in aliases if alias in available]
            if not found:
                raise ValueError('a referenced native update is unavailable')
            expanded.extend(found)
        return tuple(dict.fromkeys(expanded))


def _apply_public_update(files, update):
    """Materialize only the introduced public prefix, never live agent edits."""
    introduced = {}
    for asset in update.files:
        if update.append and asset.path in files:
            data = files[asset.path].data + b'\n\n' + asset.data
            if len(data) > 8 * 1024 * 1024:
                raise ValueError('accumulated public conversation exceeds file budget')
            asset = AssetFile(asset.path, data, asset.mode)
        files[asset.path] = asset
        introduced[asset.path] = asset
    return introduced


def public_materials_for_unit(assets, units, unit_id):
    """Original public inputs plus updates first introduced through this unit.

    Never inspect the live solver workspace or private evaluator. Applying an
    already introduced update again would incorrectly undo a later overwrite.
    Text is kept whole; model capacity checks may reject it, never truncate it.
    """
    if (not isinstance(assets, ClawAssets) or not units or
            any(u.independent_task_id != assets.scenario_id or u.instance.benchmark != 'clawarena'
                for u in units) or
            len({u.instance.common_instance_id for u in units}) != len(units)):
        raise ValueError('public materials require one ordered native task')
    by_id = {u.instance.common_instance_id: u for u in units}
    if unit_id not in by_id:
        raise ValueError('public materials requested for a foreign scoring unit')
    files = {f.path: f for f in assets.initial}
    updates = {u.update_id: u for u in assets.updates}
    applied = []
    for unit in units:
        for update_id in assets.expand(json.loads(unit.public_context_json).get('update_ids', [])):
            if update_id not in applied:
                _apply_public_update(files, updates[update_id])
                applied.append(update_id)
        if unit.instance.common_instance_id == unit_id:
            break
    value = dict(schema=1, task_id=assets.scenario_id, unit_id=unit_id, asset_digest=assets.digest,
                 applied_updates=applied, files={}, omissions=[], complete=True)
    for path, asset in sorted(files.items()):
        metadata = {'sha256': hashlib.sha256(asset.data).hexdigest(), 'bytes': len(asset.data)}
        try:
            text = asset.data.decode('utf-8')
            if '\x00' in text:
                raise UnicodeError('binary content')
        except UnicodeError:
            value['omissions'].append({'path': path, **metadata, 'reason': 'non_text'})
        else:
            value['files'][path] = {'text': text, **metadata}
    value['complete'] = not value['omissions']
    validate_public_materials(value, by_id[unit_id])
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load_claw_assets(dataset_root: Path, scenario_id: str, *, expected_digest=None) -> ClawAssets:
    if not isinstance(scenario_id, str) or not re.fullmatch(r'[a-z0-9_]{1,80}', scenario_id):
        raise ValueError('explicit native scenario identity required')
    base = dataset_root / 'data/clawarena/codex'
    manifest = json.loads((base / 'manifest.json').read_text())
    if (manifest.get('framework') != 'codex' or
            manifest.get('agents', {}).get(scenario_id, {}).get('workspace') != 'workspaces/' + scenario_id):
        raise ValueError('native scenario workspace mapping is unavailable')
    total = 0
    def read(source, target, source_base=base):
        nonlocal total
        _relative(target)
        relative = source.relative_to(source_base)
        current = source_base
        if current.is_symlink():
            raise ValueError('native source assets cannot contain symbolic links')
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('native source assets cannot contain symbolic links')
        metadata = source.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 8 * 1024 * 1024:
            raise ValueError('native source file is unsupported or exceeds its preparation budget')
        data = source.read_bytes()
        total += len(data)
        if len(data) != metadata.st_size or total > 32 * 1024 * 1024:
            raise ValueError('native source changed or exceeds the preparation budget')
        return AssetFile(target, data, stat.S_IMODE(metadata.st_mode) & 0o777)
    workspace = base / 'workspaces' / scenario_id
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError('native public workspace is unavailable')
    initial = []
    for path in sorted(workspace.rglob('*')):
        if path.is_symlink():
            raise ValueError('native workspace contains a symbolic link')
        if not path.is_dir():
            initial.append(read(path, path.relative_to(workspace).as_posix()))
    if not initial:
        raise ValueError('native public workspace is empty')
    legacy_initial = tuple(initial)
    # Only the seed named in the original manifest is public at task start.
    # Never enumerate the sessions directory or import future update sessions.
    original = dataset_root / 'data/clawarena/openclaw'
    manifest_path = original / 'manifest.json'
    if manifest_path.exists() or manifest_path.is_symlink():
        original_manifest = json.loads(read(manifest_path, 'manifest.json', original).data)
        agent = original_manifest.get('agents', {}).get(scenario_id, {})
        session = agent.get('session')
        if (original_manifest.get('framework') != 'openclaw' or
                agent.get('agent_dir') != 'state/agents/' + scenario_id or
                not isinstance(session, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', session)):
            raise ValueError('original public main conversation mapping is unavailable')
        target = 'message_logs/main_session.md'
        if any(asset.path == target for asset in initial):
            raise ValueError('original main conversation would overwrite a workspace asset')
        source = original / agent['agent_dir'] / 'sessions' / (session + '.jsonl')
        data = read(source, target, original).data
        initial.append(AssetFile(target, _public_main_conversation(data), 0o600))
    updates = []
    groups = manifest.get('updates', {}).get(scenario_id, {})
    for identifier, specification in groups.items():
        if (not isinstance(identifier, str) or not isinstance(specification, dict) or
                specification.get('type') not in {'workspace', 'workspace_md'} or
                not isinstance(specification.get('files'), list)):
            raise ValueError('unsupported native update type')
        source_dir = base / _relative(specification.get('dir'))
        files = []
        for item in specification['files']:
            if isinstance(item, str):
                item = {'name': item, 'target': item}
            if not isinstance(item, dict) or item.get('action', 'new') != 'new':
                raise ValueError('this pinned data adapter only supports native new/overwrite updates')
            name = _relative(item.get('name'))
            target = _relative(item.get('target', name))
            files.append(read(source_dir / name, target))
        # workspace_md is the Codex projection of a session append, not a new
        # snapshot of the whole conversation. Ordinary workspace files remain
        # replacements. Preserve both old and newly introduced public messages.
        append = specification['type'] == 'workspace_md'
        if append and any(not f.path.startswith('message_logs/') or not f.path.endswith('.md')
                          for f in files):
            raise ValueError('session projections must target public Markdown conversation logs')
        updates.append(AssetUpdate(identifier, tuple(files), append=append))
    assets = ClawAssets(scenario_id, tuple(initial), tuple(updates))
    if expected_digest is None or assets.digest == expected_digest:
        return assets
    # Receipts before session-append repair bind replacement semantics, with or
    # without the public main conversation. Accept only these exact historical
    # projections; never silently change a frozen training/test source.
    legacy_updates = tuple(AssetUpdate(u.update_id, u.files) for u in updates)
    for historical_initial in (tuple(initial), legacy_initial):
        legacy = ClawAssets(scenario_id, historical_initial, legacy_updates)
        if legacy.digest == expected_digest:
            return legacy
    raise ValueError('public asset digest differs from all supported projections')


def _public_main_conversation(data):
    """Project visible messages from the pinned seed, preserving their text."""
    parts = ['# Original public main conversation\n']
    for line in data.decode('utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('type') != 'message':
            continue
        message = row['message']
        role = message.get('role')
        if role not in {'user', 'assistant', 'toolResult'}:
            raise ValueError('unsupported public seed message role')
        content = message.get('content', [])
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        visible = []
        for block in content:
            kind = block.get('type')
            if kind == 'text':
                visible.append(block['text'])
            elif kind == 'toolCall':
                visible.append(json.dumps({'tool': block['name'], 'arguments': block['arguments']},
                                          ensure_ascii=False, sort_keys=True))
            elif kind != 'thinking':
                raise ValueError('unsupported content in public seed conversation')
        if visible:
            timestamp = row.get('timestamp', '')
            parts.append(f'\n---\n\n**{role}** ({timestamp}):\n\n' + '\n\n'.join(visible) + '\n')
    if len(parts) == 1:
        raise ValueError('original public main conversation is empty')
    return ''.join(parts).encode('utf-8')


def _write_file(root_fd, asset):
    # Resolve directories relative to the already-owned task root, never
    # through an agent-created symlink. Replacing a leaf symlink is safe.
    parts = _relative(asset.path).split('/')
    parent_fd = os.dup(root_fd)
    temporary = '.native-update-' + uuid.uuid4().hex
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(asset.data)
            os.fchmod(stream.fileno(), asset.mode | stat.S_IRUSR | stat.S_IWUSR)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, parts[-1], src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


class NativeClawWorkspace:
    def __init__(self, *, assets: ClawAssets, expected_digest: str, workspace: Path,
                 state_dir: Path, start: dict, units: tuple, resume=False):
        if (not isinstance(assets, ClawAssets) or assets.digest != expected_digest or
                any(not p.is_absolute() or p != p.resolve() for p in (workspace, state_dir)) or
                workspace == state_dir or workspace in state_dir.parents or state_dir in workspace.parents or
                ((workspace.exists() or state_dir.exists()) and not resume) or not isinstance(units, tuple) or not units or
                start.get('task_id') != assets.scenario_id or
                any(not isinstance(start.get(k), str) or not start[k] for k in ('namespace', 'session_id')) or
                any(u.independent_task_id != assets.scenario_id or u.instance.benchmark != 'clawarena' for u in units)):
            raise ValueError('fresh task-owned workspace, private state and pinned assets required')
        self.assets, self.workspace, self.state_dir = assets, workspace, state_dir
        self.start, self.units = dict(start), units
        self.plan = tuple((u.instance.common_instance_id,
                           assets.expand(json.loads(u.public_context_json)['update_ids'])) for u in units)
        if len({row[0] for row in self.plan}) != len(self.plan):
            raise ValueError('duplicate native scoring unit in workspace plan')
        self._lock = threading.RLock()
        self._root = None
        self._artifacts = None
        self._public_files = {asset.path: asset for asset in assets.initial}
        self._state = {'schema': 1, 'namespace': start['namespace'], 'session_id': start['session_id'],
            'task_id': assets.scenario_id, 'asset_digest': assets.digest, 'plan_digest': content_hash(self.plan),
            'status': 'preparing', 'next_index': 0, 'applied_updates': [], 'last_unit': None}
        if resume:
            state=json.loads((state_dir/'workspace-state.json').read_text())
            if any(state[k]!=self._state[k] for k in ('namespace','session_id','task_id','asset_digest','plan_digest')) or state['status']!='ready':
                raise ValueError('saved workspace differs from this round checkpoint')
            self._state=state
            updates = {u.update_id: u for u in assets.updates}
            for identifier in state['applied_updates']:
                _apply_public_update(self._public_files, updates[identifier])
            self._root=os.open(workspace,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            return
        state_dir.mkdir(mode=0o700)
        self._save()
        workspace.mkdir(mode=0o700)
        try:
            self._root = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            for asset in assets.initial:
                _write_file(self._root, asset)
            self._state['status'] = 'ready'
            self._save()
        except Exception:
            self.close()
            raise

    def _save(self):
        atomic_write_bytes(self.state_dir / 'workspace-state.json', canonical_json_bytes(self._state))

    def initial_files(self):
        return {asset.path: {'data': asset.data, 'mode': asset.mode | stat.S_IRUSR | stat.S_IWUSR}
                for asset in self.assets.initial}

    def create_artifacts(self, directory: Path, resume=None):
        with self._lock:
            if self._root is None or self._artifacts is not None or (self._state['next_index'] != 0 and resume is None):
                raise ValueError('capture the initial native baseline once, before preparing any round')
            artifacts = TaskArtifacts(workspace=self.workspace, directory=directory,
                                     file_budget=8*1024*1024, total_budget=16*1024*1024, resume=resume)
            try:
                artifacts.validate_baseline(self.initial_files())
            except Exception:
                artifacts.close()
                raise
            self._artifacts = artifacts
            return artifacts

    def prepare_unit(self, unit, *, start):
        with self._lock:
            if self._root is None or start != self.start:
                raise ValueError('native workspace belongs to another task session')
            if self._state['status'] != 'ready':
                raise RecoverableCallError('workspace_update_interrupted', retryable=False)
            unit_id = unit.instance.common_instance_id
            if self._state['last_unit'] == unit_id:
                if unit != self.units[self._state['next_index'] - 1]:
                    raise ValueError('native unit identity was reused with different public input')
                return
            index = self._state['next_index']
            if index >= len(self.plan) or self.units[index] != unit:
                raise ValueError('native rounds must prepare in their fixed task order')
            updates = {u.update_id: u for u in self.assets.updates}
            self._state['status'] = 'applying'
            self._save()
            try:
                introduced = {}
                for update_id in self.plan[index][1]:
                    # Repeated references mean evidence already introduced in
                    # this task. Do not overwrite subsequent agent work again.
                    if update_id in self._state['applied_updates']:
                        continue
                    materialized = _apply_public_update(self._public_files, updates[update_id])
                    for asset in materialized.values():
                        _write_file(self._root, asset)
                        introduced[asset.path] = {'data': asset.data, 'mode': asset.mode | stat.S_IRUSR | stat.S_IWUSR}
                    self._state['applied_updates'].append(update_id)
                if self._artifacts is not None and introduced:
                    self._artifacts.record_public_update(introduced)
                self._state.update(status='ready', next_index=index + 1, last_unit=unit_id)
                self._save()
            except Exception:
                self._state['status'] = 'interrupted'
                self._save()
                raise RecoverableCallError('workspace_update_interrupted', retryable=False) from None

    def close(self):
        if self._artifacts is not None:
            self._artifacts.close()
        if self._root is not None:
            os.close(self._root)
            self._root = None
