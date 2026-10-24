"""Run inputs for the ClawArena phase worker.

Reading a specification never reads credential contents or starts a model.
Whole independent tasks and model roles are fixed before creating writable
experiment state.
"""
from dataclasses import asdict
from pathlib import Path
import re

from tellonce_common.codex_text import configured_context_window
from tellonce_common.claude_native import ClaudeConfig
from tellonce_common.contracts import Split
from tellonce_common.frozen_task_evaluation import _directory, _digest, _read
from tellonce_common.provenance import content_hash, file_hash, tree_hash
from tellonce_common.role_inputs import load_user_profile
from tellonce_common.role_inputs import RunIdentity
from tellonce_common.text_roles import validate_model_config
from .clawarena_loader import load_scenario, project_native_scenario
from .native_workspace import load_claw_assets


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('an explicit absolute path is required')
    result = Path(value)
    if not result.is_absolute() or result != result.resolve():
        raise ValueError('run paths must be canonical and absolute')
    return result


def _reference(value):
    """A run input file: a path, or {'path', 'sha256'} to also pin its content."""
    if isinstance(value, str):
        value = {'path': value}
    if (not isinstance(value, dict) or not set(value) <= {'path', 'sha256'} or 'path' not in value or
            ('sha256' in value and not _digest(value['sha256']))):
        raise ValueError('a run input file reference is required')
    path = _path(value['path'])
    if not path.is_file() or ('sha256' in value and file_hash(path) != value['sha256']):
        raise ValueError('a run input file changed or is missing')
    return path


class ClawRunSpec:
    @classmethod
    def load(cls, *, path, expected_digest=None):
        path = _path(str(path))
        value = _read(path, 8*1024*1024)
        if expected_digest is None:
            expected_digest = content_hash(value)
        if not _digest(expected_digest) or content_hash(value) != expected_digest:
            raise ValueError('run specification differs from its expected digest')
        required = {'schema', 'run_id', 'user_id', 'method', 'models', 'directory',
                    'dataset_root', 'dataset_revision', 'tasks',
                    'binary', 'catalog', 'tokenizer', 'runtime',
                    'auth_home', 'api_key_file', 'compiler_id', 'task_timeout_s', 'role_deadline_s',
                    'execution_enabled', 'score_before_repair', 'python', 'python_roots', 'phase_timeout_s'}
        uses_claude=(isinstance(value,dict) and isinstance(value.get('models'),dict) and
                     isinstance(value['models'].get('solver'),dict) and
                     str(value['models']['solver'].get('model','')).startswith('claude-'))
        if uses_claude:required.update({'claude_binary','claude_auth_file','claude_helpers'})
        if (not isinstance(value, dict) or set(value) != required or value['schema'] != 1 or
                value['method'] != 'trace' or not isinstance(value['runtime'], dict) or
                type(value['execution_enabled']) is not bool or type(value['score_before_repair']) is not bool or
                (value['score_before_repair'] and not value['execution_enabled']) or
                not isinstance(value['compiler_id'], str) or not value['compiler_id'] or
                not isinstance(value['dataset_revision'], str) or not value['dataset_revision']):
            raise ValueError('a complete supported frozen run specification is required')
        for field, maximum in (('task_timeout_s', 3600), ('role_deadline_s', 86400), ('phase_timeout_s', 86400)):
            if type(value[field]) not in (int, float) or not 1 <= value[field] <= maximum:
                raise ValueError('each model/task budget must be finite and explicit')
        result = cls()
        result.solver_config, result.role_configs = validate_model_config(value['models'])
        if uses_claude:
            if (any(value['models']['roles'][role]['kind'] not in {'claude_native','openrouter'}
                    for role in ('memory','appeal')) or
                    value['models']['roles']['feedback']['kind'] not in {'claude_native','openrouter'} or
                    value['models']['roles']['preference']['kind'] not in {'claude_native','native','openrouter'} or
                    any(config.model!=result.solver_config.model for role,config in result.role_configs.items()
                        if value['models']['roles'][role]['kind']=='claude_native')):
                raise ValueError('Claude tasks require matching native organizers and explicit scoring roles')
        elif any(row['kind']=='claude_native' for row in value['models']['roles'].values()):
            raise ValueError('native Claude roles require a Claude solver')
        result.path, result.value, result.binding = path, value, expected_digest
        result.identity = RunIdentity(value['run_id'], 'clawarena', value['user_id'], value['method'],
                                      content_hash(value['models']))
        result.directory = _path(value['directory'])
        _directory(result.directory)
        result.dataset_root = _path(value['dataset_root'])
        result.binary, result.catalog = _reference(value['binary']), _reference(value['catalog'])
        result.solver_binary=result.binary
        if uses_claude:
            result.claude_binary=_reference(value['claude_binary']);result.solver_binary=result.claude_binary
            result.claude_auth_file=_path(value['claude_auth_file'])
            helpers=value['claude_helpers']
            if not isinstance(helpers,list) or len(helpers)!=1 or _reference(helpers[0])!=result.claude_binary.parent/'socat':
                raise ValueError('Claude sandbox requires a local socat helper next to the Claude binary')
        result.python = _reference(value['python'])
        if not isinstance(value['python_roots'], list) or not value['python_roots']:
            raise ValueError('explicit native grader interpreter roots are required')
        result.python_roots = tuple(_path(root) for root in value['python_roots'])
        if not any(result.python.is_relative_to(root) for root in result.python_roots):
            raise ValueError('the native grader interpreter must belong to its pinned runtime')
        catalog = _read(result.catalog, 8*1024*1024)
        for config in (result.solver_config, *[result.role_configs[role] for role, row in
                       value['models']['roles'].items() if row['kind'] == 'native']):
            if isinstance(config, ClaudeConfig):
                continue
            metadata = [row for row in catalog.get('models', []) if row.get('slug') == config.model]
            if len(metadata) != 1:
                raise ValueError('pinned catalog must describe each selected native model exactly once')
            configured_context_window(metadata[0], config.context_window)
        result.tokenizer, result.auth_home = _path(value['tokenizer']), _path(value['auth_home'])
        result.api_key_file = _path(value['api_key_file'])  # Never read here.
        from tellonce_common.runtime_revision import load_runtime_revision
        load_runtime_revision(result)
        result.profile = load_user_profile(value['user_id'])
        result.units, result.assets = {}, {}
        tasks = value['tasks']
        if not isinstance(tasks, list) or not tasks:
            raise ValueError('a fixed whole-task train/test plan is required')
        train_families, test_rows = set(), []
        for row in tasks:
            if (not isinstance(row, dict) or set(row) != {'scenario_id', 'family', 'split', 'seed',
                                                        'units_hash', 'asset_digest', 'evaluation_digest'} or
                    row['split'] not in {'acquisition', 'id_test', 'task_ood'} or type(row['seed']) is not int or
                    not isinstance(row['family'], str) or not row['family'] or
                    not isinstance(row['scenario_id'], str) or
                    not re.fullmatch(r'[a-z0-9_]{1,80}', row['scenario_id']) or
                    row['scenario_id'] in result.units):
                raise ValueError('every independent task occurs in exactly one declared split')
            sid = row['scenario_id']
            units, _ = project_native_scenario(load_scenario(result.dataset_root, sid),
                dataset_revision=value['dataset_revision'], task_family=row['family'], split=Split(row['split']),
                user_id=value['user_id'], seed=row['seed'])
            assets = load_claw_assets(result.dataset_root, sid, expected_digest=row['asset_digest'])
            if (content_hash([asdict(unit) for unit in units]) != row['units_hash'] or
                    assets.digest != row['asset_digest'] or
                    tree_hash(result.dataset_root/'data/clawarena/eval'/sid) != row['evaluation_digest']):
                raise ValueError('task definition, public assets or hidden evaluation differ from the run plan')
            result.units[sid], result.assets[sid] = units, assets
            if row['split'] == 'acquisition':
                train_families.add(row['family'])
            else:
                test_rows.append(row)
        if (not train_families or not test_rows or any(
                (row['split'] == 'id_test' and row['family'] not in train_families) or
                (row['split'] == 'task_ood' and row['family'] in train_families) for row in test_rows)):
            raise ValueError('ID uses trained families and OOD uses untrained families, with independent tasks')
        result.verify()
        return result

    def verify(self):
        if content_hash(self.value) != self.binding or content_hash(_read(self.path, 8*1024*1024)) != self.binding:
            raise ValueError('run configuration changed during execution')
        from tellonce_common.runtime_revision import verify_runtime_revision
        verify_runtime_revision(self)
        for reference in (self.value['binary'], self.value['catalog'], self.value['python']):
            _reference(reference)
        if isinstance(self.solver_config,ClaudeConfig):
            for reference in (self.value['claude_binary'],*self.value['claude_helpers']):_reference(reference)
