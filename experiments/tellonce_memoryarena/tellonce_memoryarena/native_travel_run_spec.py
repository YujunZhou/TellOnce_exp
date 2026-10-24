"""Frozen whole-row travel plans, resources and model roles; no model calls.

Only the public tool tree and databases enter the tool sandbox. Original rows
and native evaluation sources remain host-side, bound to the run plan.
"""
from dataclasses import asdict
from pathlib import Path

from tellonce_common.codex_text import configured_context_window
from tellonce_common.claude_native import ClaudeConfig
from tellonce_common.contracts import Split
from tellonce_common.frozen_task_evaluation import _directory, _digest, _read
from tellonce_common.provenance import content_hash, file_hash
from tellonce_common.role_inputs import RunIdentity, load_user_profile
from tellonce_common.text_roles import validate_model_config
from .memoryarena_loader import load_rows, project_native_row
from .native_scoring import TRAVEL_EVAL_SHA256
from .travel_submission import PINNED, NativeTravelSubmission
from .travel_tool_service import verify_public_tree


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('an explicit canonical absolute path is required')
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError('run paths must be canonical and absolute')
    return path


def _reference(value):
    """A run input file: a path, or {'path', 'sha256'} to also pin its content."""
    if isinstance(value, str):
        value = {'path': value}
    if (not isinstance(value, dict) or not set(value) <= {'path', 'sha256'} or 'path' not in value or
            ('sha256' in value and not _digest(value['sha256']))):
        raise ValueError('a run input file reference is required')
    path = _path(value['path'])
    if not path.is_file() or ('sha256' in value and file_hash(path) != value['sha256']):
        raise ValueError('a travel run input changed or is missing')
    return path


def _ref_path(value):
    return _path(value if isinstance(value, str) else value['path'])


def _tree_state(root):
    """After full startup hashing, detect any resource replacement or edit."""
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError('public resources contain a link or special file')
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(root))] = (stat.st_dev, stat.st_ino, stat.st_size,
                                                  stat.st_mtime_ns, stat.st_ctime_ns)
    return result


class TravelRunSpec:
    @classmethod
    def load(cls, *, path, expected_digest=None):
        path = _path(str(path))
        value = _read(path, 8*1024*1024)
        if expected_digest is None:
            expected_digest = content_hash(value)
        fields = {'schema', 'run_id', 'user_id', 'method', 'models', 'directory',
            'dataset_revision', 'rows', 'tasks', 'runtime',
            'binary', 'catalog', 'tokenizer', 'auth_home', 'api_key_file',
            'compiler_id', 'task_timeout_s', 'role_deadline_s', 'phase_timeout_s', 'execution_enabled',
            'score_before_repair', 'python', 'official_source', 'public_tool_source',
            'database', 'tool_python', 'tool_runtime_roots'}
        uses_claude = (isinstance(value, dict) and isinstance(value.get('models'), dict) and
            isinstance(value['models'].get('solver'), dict) and
            str(value['models']['solver'].get('model', '')).startswith('claude-'))
        if uses_claude:
            fields.update({'claude_binary', 'claude_auth_file', 'claude_helpers'})
        if (not _digest(expected_digest) or content_hash(value) != expected_digest or
                not isinstance(value, dict) or set(value) != fields or value['schema'] != 1 or
                value['method'] != 'trace' or not isinstance(value['runtime'], dict) or
                type(value['execution_enabled']) is not bool or type(value['score_before_repair']) is not bool or
                (value['score_before_repair'] and not value['execution_enabled']) or
                any(not isinstance(value[k], str) or not value[k] for k in ('compiler_id', 'dataset_revision'))):
            raise ValueError('a complete travel run specification is required')
        for field, maximum in (('task_timeout_s', 3600), ('role_deadline_s', 86400), ('phase_timeout_s', 86400)):
            if type(value[field]) not in (int, float) or not 1 <= value[field] <= maximum:
                raise ValueError('explicit finite task, role and phase deadlines required')
        result = cls()
        result.path, result.value, result.binding = path, value, expected_digest
        result.identity = RunIdentity(value['run_id'], 'memoryarena', value['user_id'], value['method'],
                                      content_hash(value['models']))
        result.solver_config, result.role_configs = validate_model_config(value['models'])
        if uses_claude:
            if (any(value['models']['roles'][role]['kind'] not in {'claude_native','openrouter'}
                    for role in ('memory', 'appeal')) or
                    value['models']['roles']['feedback']['kind'] not in {'claude_native', 'openrouter'} or
                    value['models']['roles']['preference']['kind'] not in {'claude_native', 'native', 'openrouter'} or
                    any(config.model != result.solver_config.model for role, config in result.role_configs.items()
                        if value['models']['roles'][role]['kind'] == 'claude_native')):
                raise ValueError('Claude travel requires matching native organizers and explicit judges')
        elif any(row['kind'] == 'claude_native' for row in value['models']['roles'].values()):
            raise ValueError('native Claude roles require a Claude solver')
        for key in ('directory', 'tokenizer', 'auth_home', 'api_key_file',
                    'official_source', 'public_tool_source', 'database'):
            setattr(result, key, _path(value[key]))
        _directory(result.directory)
        result.references = [value[key] for key in ('rows', 'binary', 'catalog', 'python', 'tool_python')]
        for key in ('binary', 'catalog', 'python', 'tool_python'):
            setattr(result, key, _reference(value[key]))
        result.solver_binary = result.binary
        if uses_claude:
            result.claude_binary = _reference(value['claude_binary'])
            result.solver_binary = result.claude_binary
            result.claude_auth_file = _path(value['claude_auth_file'])
            helpers = value['claude_helpers']
            if not isinstance(helpers, list) or len(helpers) != 1 or _reference(helpers[0]) != result.claude_binary.parent/'socat':
                raise ValueError('Claude sandbox requires a local socat helper next to the Claude binary')
            result.references.extend([value['claude_binary'], *helpers])
        if not isinstance(value['tool_runtime_roots'], list) or not value['tool_runtime_roots']:
            raise ValueError('explicit isolated tool runtime roots required')
        result.tool_runtime_roots = tuple(_path(root) for root in value['tool_runtime_roots'])
        if not any(result.tool_python.is_relative_to(root) for root in result.tool_runtime_roots):
            raise ValueError('tool interpreter must belong to its runtime roots')
        catalog = _read(result.catalog, 8*1024*1024)
        for config in (result.solver_config, *[result.role_configs[k] for k, row in value['models']['roles'].items()
                                               if row['kind'] == 'native']):
            if isinstance(config, ClaudeConfig):
                continue
            metadata = [m for m in catalog.get('models', []) if m.get('slug') == config.model]
            if len(metadata) != 1:
                raise ValueError('pinned catalog must describe each selected native model exactly once')
            configured_context_window(metadata[0], config.context_window)
        from tellonce_common.runtime_revision import load_runtime_revision
        load_runtime_revision(result)
        result.profile = load_user_profile(value['user_id'])
        result.profile_hash = content_hash(asdict(result.profile))
        row_path = _reference(value['rows'])
        if row_path.stat().st_size > 64*1024*1024:
            raise ValueError('travel row source exceeds the explicit read budget')
        original = load_rows(row_path)
        by_id = {}
        for row in original:
            if (not isinstance(row, dict) or type(row.get('id')) not in (int, str) or
                    not str(row['id']).strip() or str(row['id']) in by_id):
                raise ValueError('travel source requires distinct native row identities')
            by_id[str(row['id'])] = row
        result.rows, result.units = {}, {}
        if not isinstance(value['tasks'], list) or not value['tasks']:
            raise ValueError('complete whole-row train/test plan required')
        splits = set()
        for planned in value['tasks']:
            if (not isinstance(planned, dict) or set(planned) != {'row_id', 'split', 'seed', 'row_hash', 'units_hash'} or
                    not isinstance(planned['row_id'], str) or planned['row_id'] not in by_id or
                    planned['split'] not in {'acquisition', 'id_test'} or type(planned['seed']) is not int):
                raise ValueError('one explicit whole travel row per declared split required')
            row = by_id[planned['row_id']]
            units, _ = project_native_row(row, subset='group_travel_planner', dataset_revision=value['dataset_revision'],
                split=Split(planned['split']), user_id=value['user_id'], seed=planned['seed'])
            task_id = units[0].independent_task_id
            if (task_id in result.units or content_hash(row) != planned['row_hash'] or
                    content_hash([asdict(unit) for unit in units]) != planned['units_hash']):
                raise ValueError('travel row is duplicated, incomplete or differs from the run plan')
            NativeTravelSubmission(result.official_source, units=units)
            result.rows[task_id], result.units[task_id] = row, units
            splits.add(planned['split'])
        if splits != {'acquisition', 'id_test'}:
            raise ValueError('travel development needs separate complete training and test rows')
        result.official_references = {
            result.official_source/relative: digest for relative, (digest, _) in PINNED.items()}
        result.official_references[result.official_source/'env/env_systems/travel_planner_env/eval.py'] = TRAVEL_EVAL_SHA256
        pins = _read(Path(__file__).with_name('travel_tool_pins.json'), 1024*1024)
        result.resource_states = {root: _tree_state(root) for root in (result.public_tool_source, result.database)}
        verify_public_tree(result.public_tool_source, pins['sources'])
        verify_public_tree(result.database, pins['database'])
        if any(_tree_state(root) != state for root, state in result.resource_states.items()):
            raise ValueError('public resources changed while verifying their content')
        result.verify()
        return result

    def verify(self):
        if content_hash(self.value) != self.binding or content_hash(_read(self.path, 8*1024*1024)) != self.binding:
            raise ValueError('frozen travel specification changed during execution')
        if self.identity != RunIdentity(self.value['run_id'], 'memoryarena', self.value['user_id'],
                self.value['method'], content_hash(self.value['models'])):
            raise ValueError('experiment identity differs from the frozen plan')
        for key in ('directory', 'tokenizer', 'auth_home', 'api_key_file',
                    'official_source', 'public_tool_source', 'database'):
            if getattr(self, key) != _path(self.value[key]):
                raise ValueError('derived travel path differs from the frozen plan')
        for key in ('binary', 'catalog', 'python', 'tool_python'):
            if getattr(self, key) != _ref_path(self.value[key]):
                raise ValueError('derived runtime differs from the frozen plan')
        expected_solver = self.binary
        if isinstance(self.solver_config, ClaudeConfig):
            expected_solver = _ref_path(self.value['claude_binary'])
            if (self.claude_binary != expected_solver or self.claude_auth_file != _path(self.value['claude_auth_file'])):
                raise ValueError('derived Claude runtime differs from the frozen plan')
        if self.solver_binary != expected_solver:
            raise ValueError('solver binary differs from the run plan')
        if (content_hash(asdict(self.profile)) != self.profile_hash or
                self.tool_runtime_roots != tuple(_path(root) for root in self.value['tool_runtime_roots']) or
                (self.solver_config, self.role_configs) != validate_model_config(self.value['models'])):
            raise ValueError('profile or runtime role configuration changed in memory')
        from tellonce_common.runtime_revision import verify_runtime_revision
        verify_runtime_revision(self)
        for reference in self.references:
            _reference(reference)
        for path, digest in self.official_references.items():
            _reference({'path': str(path), 'sha256': digest})
        for root, expected in self.resource_states.items():
            if root != root.resolve() or _tree_state(root) != expected:
                raise ValueError('public resource changed during execution')
        for planned in self.value['tasks']:
            task_id = 'group_travel_planner:'+planned['row_id']
            if (content_hash(self.rows[task_id]) != planned['row_hash'] or
                    content_hash([asdict(unit) for unit in self.units[task_id]]) != planned['units_hash']):
                raise ValueError('in-memory native travel plan changed')
        return self.binding
