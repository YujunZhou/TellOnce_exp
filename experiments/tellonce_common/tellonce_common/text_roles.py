"""Explicit model roles with persistent attempts and mandatory capacity checks.

Construction prepares no credentials or model processes. The admitted worker
provides one native account owner and a provider-bound HTTP key source.
"""
from pathlib import Path
import threading
import uuid

from .codex_text import CodexTextConfig
from .codex_text_transport import CodexTextTransport
from .claude_native import ClaudeConfig
from .claude_text_transport import ClaudeTextTransport
from .deferred_scoring import _write
from .durable_text_transport import DurableTextTransport
from .frozen_task_evaluation import _directory, _digest, _read
from .http_text_contract import HTTPTextConfig, GEMMA_MODEL
from .http_text_transport import HTTPTextTransport
from .managed_model_requests import USER_ROLES, PUBLICATION_ROLES
from .provenance import content_hash
from .role_inputs import RunIdentity
from .text_capacity import TextCapacity, bundled_gemma_tokenizer_root
from .text_request_journal import TextRequestJournal


ROLES = frozenset({'feedback', 'memory', 'preference', 'appeal'})
OPTIONAL_ROLES = frozenset({'execution_check'})
NATIVE_MODELS = frozenset({'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol', 'gpt-5.6-terra', 'gpt-5.6-sol', 'gpt-6-astra'})
CALL_ROLES = {'feedback': {'feedback'}, 'memory': USER_ROLES | PUBLICATION_ROLES | {'baseline_memory', 'baseline_plan', 'baseline_retrieval'},
              'preference': {'preference_score', 'preference_score_before_repair'}, 'appeal': {'rule_appeal'},
              'execution_check': {'rule_check', 'compile_case'}}


def validate_model_config(value):
    """Validate all roles without creating journals or consulting credentials."""
    if (not isinstance(value, dict) or set(value) != {'solver', 'roles'} or
            not isinstance(value['solver'], dict) or not isinstance(value['roles'], dict) or
            not ROLES <= set(value['roles']) <= ROLES | OPTIONAL_ROLES):
        raise ValueError('complete explicit solver and text-role model configuration required')
    try:
        if not isinstance(value['solver'].get('model'),str):raise ValueError('explicit solver model required')
        solver_type = ClaudeConfig if value['solver'].get('model', '').startswith('claude-') else CodexTextConfig
        solver = solver_type(**value['solver'])
        if isinstance(solver, CodexTextConfig) and (solver.model not in NATIVE_MODELS or solver.context_window is None):
            raise ValueError('solver requires an admitted native model with explicit context capacity')
        configs = {}
        for role, row in value['roles'].items():
            if (not isinstance(row, dict) or set(row) != {'kind', 'config', 'timeout_s'} or
                    not isinstance(row['config'], dict) or type(row['timeout_s']) not in (int, float) or
                    not 0 < row['timeout_s'] <= 900):
                raise ValueError('each role requires explicit provider, configuration and bounded timeout')
            if role == 'execution_check' and row['kind'] != 'openrouter':
                raise ValueError('separate execution checks require an explicitly admitted OpenRouter model')
            if row['kind'] == 'native':
                config = CodexTextConfig(**row['config'])
                if config.model not in NATIVE_MODELS or config.context_window is None:
                    raise ValueError('native text roles require an admitted model and explicit capacity')
            elif row['kind'] == 'claude_native':
                config = ClaudeConfig(**row['config'])
            elif row['kind'] == 'deepinfra':
                config = HTTPTextConfig(**row['config'])
                if config.model != 'google/gemma-4-31B-it' or config.provider!='deepinfra':
                    raise ValueError('HTTP text roles require the admitted Gemma model')
            elif row['kind']=='openrouter':
                config=HTTPTextConfig(**row['config'])
                allowed = {'feedback': {GEMMA_MODEL, 'deepseek/deepseek-v4.1-flash', 'google/gemini-3.8-flash'},
                           'preference': {GEMMA_MODEL, 'deepseek/deepseek-v4.1-flash', 'google/gemini-3.8-flash'},
                           'memory': {GEMMA_MODEL}, 'appeal': {GEMMA_MODEL},
                           'execution_check': {GEMMA_MODEL, 'deepseek/deepseek-v4.1-flash'}}
                if config.provider!='openrouter' or config.model not in allowed[role]:
                    raise ValueError('OpenRouter model is not admitted for this text role')
            else:
                raise ValueError('unreviewed text-role provider')
            configs[role] = config
        if len({row['kind'] for row in value['roles'].values() if row['kind'] not in {'native','claude_native'}})>1:
            raise ValueError('one provider-bound HTTP credential source per experiment required')
    except TypeError as exc:
        raise ValueError('invalid model configuration fields') from exc
    return solver, configs


class TextRoles:
    def _setup(self, *, directory, identity, binding, model_config, binary, catalog, tokenizer,
               authorize, record, check_health, provision_auth, api_key_source,
               claude_binary=None, claude_credentials=None):
        _directory(directory)
        if (not isinstance(identity, RunIdentity) or not _digest(binding) or
                not isinstance(model_config, dict) or set(model_config) != {'solver', 'roles'} or
                content_hash(model_config) != identity.model_config_hash or
                any(not callable(fn) for fn in (authorize, record, check_health)) or
                any(not isinstance(path, Path) or not path.is_absolute() or path != path.resolve()
                    for path in (binary, catalog, tokenizer))):
            raise ValueError('complete admitted model configuration and private role ownership required')
        # Validate every role before publishing configuration or acquiring auth.
        _, self.configs = validate_model_config(model_config)
        for role, value in model_config['roles'].items():
            if value['kind'] == 'native':
                if not callable(provision_auth):
                    raise ValueError('native text roles require the single auth owner')
            elif value['kind'] == 'claude_native':
                if (not callable(claude_credentials) or not isinstance(claude_binary,Path) or
                        not claude_binary.is_absolute() or claude_binary!=claude_binary.resolve()):
                    raise ValueError('Claude text roles require a pinned binary and access-lease owner')
            elif value['kind'] in {'deepinfra','openrouter'}:
                if not callable(api_key_source):
                    raise ValueError('HTTP text roles require the provider key source')
                if getattr(api_key_source,'provider',value['kind'])!=value['kind']:
                    raise ValueError('HTTP text role and credential provider differ')
        self.directory, self.identity, self.binding = directory, identity, binding
        self.binary, self.catalog, self.tokenizer = binary, catalog, tokenizer
        self.authorize, self.record, self.check_health = authorize, record, check_health
        self.provision_auth, self.api_key_source = provision_auth, api_key_source
        self.claude_binary,self.claude_credentials=claude_binary,claude_credentials
        # Store a fresh canonical copy rather than retaining mutable caller maps.
        import json
        self.model_config = json.loads(json.dumps(model_config))
        self._manifest = {'schema': 1, 'namespace': identity.namespace, 'binding': binding,
                          'model_config': self.model_config, 'binary': str(binary), 'catalog': str(catalog),
                          'tokenizer': str(tokenizer)}
        if any(row['kind']=='claude_native' for row in self.model_config['roles'].values()):
            self._manifest['claude_binary']=str(claude_binary)
        self._transports, self._journals = {}, {}
        self._prepare_lock = threading.Lock()

    @classmethod
    def create(cls, **kwargs):
        owner = cls.__new__(cls)
        owner._setup(**kwargs)
        if owner.directory.exists():
            raise ValueError('new model-role ownership requires a fresh directory')
        owner.authorize('prepare_text_roles', dict(owner._manifest))
        owner.directory.mkdir(mode=0o700)
        _write(owner.directory/'roles.json', owner._manifest)
        for role in sorted(owner.configs):
            root = owner.directory/role
            root.mkdir(mode=0o700)
            (root/'attempts').mkdir(mode=0o700)
            owner._journals[role] = TextRequestJournal.create(root/'journal', owner.identity, binding=owner.binding)
        return owner

    @classmethod
    def open(cls, **kwargs):
        owner = cls.__new__(cls)
        owner._setup(**kwargs)
        owner.verify()
        owner.authorize('open_text_roles', dict(owner._manifest))
        # Do not silently recreate a missing attempt ledger on recovery.
        for role in sorted(owner.configs):
            owner._journals[role] = TextRequestJournal(owner.directory/role/'journal', owner.identity,
                                                      binding=owner.binding)
        return owner

    def verify(self):
        self.check_health()
        if _read(self.directory/'roles.json', 1024*1024) != self._manifest:
            raise ValueError('the model-role configuration changed after admission')

    def get(self, role):
        with self._prepare_lock:
            return self._get_owned(role)

    def _get_owned(self, role):
        self.verify()
        if role not in self.configs:
            raise ValueError('unknown model role')
        if role in self._transports:
            return self._transports[role]
        value, config = self.model_config['roles'][role], self.configs[role]
        def authorize(action, detail):
            self.verify()
            if 'role' in detail and detail['role'] not in CALL_ROLES[role]:
                raise ValueError('model request was routed to a different configured role')
            self.authorize(action, {'text_role': role, **detail})
        private = self.directory/role/'attempts'/str(uuid.uuid4())
        shared = dict(identity=self.identity, private_root=private, config=config,
                      authorize=authorize, record=self.record, timeout_s=value['timeout_s'])
        if value['kind'] == 'native':
            transport = CodexTextTransport(**shared, binary=self.binary, source_catalog=self.catalog,
                                           provision_auth=self.provision_auth)
            capacity = TextCapacity.native_text(model=config.model, context_window=transport.context_window,
                                                record=self.record)
        elif value['kind']=='claude_native':
            transport=ClaudeTextTransport(**shared,binary=self.claude_binary,credential_source=self.claude_credentials)
            capacity=TextCapacity.native_text(model=config.model,context_window=transport.context_window,record=self.record)
        else:
            transport = HTTPTextTransport(**shared, api_key_source=self.api_key_source)
            capacity_tokenizer = self.tokenizer
            if config.model == 'google/gemma-4-31b-it' and not capacity_tokenizer.is_dir():
                bundled = bundled_gemma_tokenizer_root()
                if bundled.is_dir():
                    capacity_tokenizer = bundled
            capacity = (TextCapacity.gemma(tokenizer_root=capacity_tokenizer,output_reserve=config.max_tokens,record=self.record,model=config.model)
                        if value['kind']=='deepinfra' or (config.model=='google/gemma-4-31b-it' and capacity_tokenizer.is_dir()) else
                        TextCapacity.openrouter_text(model=config.model,output_reserve=config.max_tokens,record=self.record))
        result = DurableTextTransport(transport, journal=self._journals[role], authorize=authorize,
                                      record=self.record, capacity=capacity)
        self._transports[role] = result
        return result
