"""Reviewed helper-model updates while retaining committed run identities.

The immutable runtime revision admits the extra model configuration. Original
model roles, task IDs and memories retain their bindings; the new checker owns
its own configuration and durable request journal, including across recovery.
"""
from copy import deepcopy
import uuid

from .deferred_scoring import _write
from .durable_text_transport import DurableTextTransport
from .frozen_task_evaluation import _read
from .http_text_transport import HTTPTextTransport
from .provenance import content_hash
from .text_capacity import TextCapacity
from .text_request_journal import TextRequestJournal
from .text_roles import validate_model_config


def task_timeout_s(spec):
    """Read an admitted wall-time extension without changing task/model bindings."""
    base = spec.value['task_timeout_s']
    reference = getattr(spec, 'runtime_bundle_reference', {})
    override = reference.get('task_timeout_s_override') if isinstance(reference, dict) else None
    if override is None:
        return base
    if (type(override) is not int or override < base or
            override > spec.value['phase_timeout_s']):
        raise ValueError('runtime task timeout override is outside the admitted phase budget')
    return override


def execution_check_transport(spec, roles):
    if getattr(roles,'api_luna_roles',False):
        if spec.identity!=roles.identity or spec.binding!=roles.binding:
            raise ValueError('API checker belongs to a different experiment')
        return roles.get('execution_check') if spec.identity.method=='trace' else None
    update = getattr(spec, 'runtime_bundle_reference', {}).get('execution_check')
    if update is None:
        return roles.get('execution_check') if 'execution_check' in spec.role_configs else None
    if spec.identity.method != 'trace' or spec.identity != roles.identity or spec.binding != roles.binding:
        raise ValueError('runtime checker update must belong to this TRACE experiment')
    if not getattr(spec, 'runtime_revision_hash', None):
        raise ValueError('runtime checker update requires a reviewed revision receipt')
    config_row = deepcopy(update)
    pinned = 'execution_check' in spec.role_configs
    if pinned:
        from .plan_a_spec import PlanARunSpec
        if (not isinstance(spec, PlanARunSpec) or
                spec.extension.get('purpose') != 'frozen-memory-plan-a-test'):
            raise ValueError('a pinned checker can change settings only in a fresh admitted frozen test')
        spec.verify()
        # The source model configuration owns the trained memory namespace.
        # A separately reviewed test can tune only reasoning/output space for
        # the same checker, never rewrite that source or change its recipient.
        expected = deepcopy(spec.value['models']['roles']['execution_check'])
        if not isinstance(config_row, dict) or not isinstance(config_row.get('config'), dict):
            raise ValueError('explicit frozen-test checker configuration required')
        for field in ('reasoning_effort', 'max_tokens'):
            if field in config_row['config']:
                expected['config'][field] = config_row['config'][field]
        if config_row != expected:
            raise ValueError('frozen-test checker can change only reasoning_effort and max_tokens')
    effective = deepcopy(spec.value['models'])
    effective['roles']['execution_check'] = config_row
    _, configs = validate_model_config(effective)
    config = configs['execution_check']
    if config.model != 'deepseek/deepseek-v4.1-flash' or config_row['kind'] != 'openrouter':
        raise ValueError('this recovery update admits only the reviewed DeepSeek checker')
    if getattr(roles.api_key_source, 'provider', None) != 'openrouter':
        raise ValueError('runtime checker requires the existing OpenRouter credential owner')
    roles.verify()
    root = roles.directory / 'execution_check'
    # A pinned original checker already has a journal. Never reuse its cached
    # verdicts for a different reasoning configuration, or overwrite it.
    journal_root = root / ('runtime-journal' if pinned else 'journal')
    manifest = {'schema': 1, 'namespace': roles.identity.namespace, 'binding': roles.binding,
                'role': 'execution_check', 'configuration': config_row}
    manifest_path = journal_root / 'runtime-role.json'
    if journal_root.exists():
        # A changed configuration or missing checkpointed ownership is an error,
        # never permission to reuse another model's cached response.
        if _read(manifest_path, 1024*1024) != manifest:
            raise ValueError('checkpointed runtime checker configuration changed')
        journal = TextRequestJournal(journal_root, roles.identity, binding=roles.binding)
    else:
        root.mkdir(mode=0o700, exist_ok=True)
        journal = TextRequestJournal.create(journal_root, roles.identity, binding=roles.binding)
        _write(manifest_path, manifest)
    (root / 'attempts').mkdir(mode=0o700, exist_ok=True)

    def authorize(action, detail):
        roles.verify()
        if _read(manifest_path, 1024*1024) != manifest:
            raise ValueError('runtime checker configuration changed during execution')
        if 'role' in detail and detail['role'] not in {'rule_check', 'compile_case'}:
            raise ValueError('runtime checker cannot perform learning, selection or feedback')
        roles.authorize(action, {'text_role': 'execution_check', **detail})

    transport = HTTPTextTransport(identity=roles.identity,
        private_root=root/'attempts'/str(uuid.uuid4()), config=config,
        authorize=authorize, record=roles.record, timeout_s=config_row['timeout_s'],
        api_key_source=roles.api_key_source)
    capacity = TextCapacity.openrouter_text(model=config.model,
        output_reserve=config.max_tokens, record=roles.record)
    roles.record({'type': 'runtime_execution_check_configured', 'namespace': roles.identity.namespace,
                  'model': config.model, 'configuration_hash': content_hash(config_row),
                  'runtime_revision_hash': spec.runtime_revision_hash})
    return DurableTextTransport(transport, journal=journal, authorize=authorize,
                                record=roles.record, capacity=capacity)
