"""Explicit role projections for the reviewed interaction protocol.

Adapters must construct PublicUnit from the benchmark's public fields. Hashing
proves identity, not that arbitrary benchmark metadata is safe to disclose.
Private payloads are deliberately absent from solver-facing objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .contracts import CommonInstance, Split
from .provenance import content_hash, canonical_json_bytes
from .users import USERS
from .compact_profiles import compact_profile, validate_profile_payload
from .native_delivery_evidence import FIELDS as OPERATION_FIELDS


def _identity(value: str):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('explicit nonempty identity required')


def _object_json(value: str):
    if not isinstance(value, str):
        raise ValueError('payload must be an immutable JSON string')
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError('payload must be a JSON object')
    canonical_json_bytes(decoded)  # Reject nonfinite values.
    return decoded


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    benchmark: str
    user_id: str
    method: str
    model_config_hash: str

    def __post_init__(self):
        for value in (self.run_id, self.benchmark, self.user_id, self.method, self.model_config_hash):
            _identity(value)
        if (len(self.model_config_hash) != 64 or
                any(c not in '0123456789abcdef' for c in self.model_config_hash)):
            raise ValueError('model configuration must be bound by SHA256')
        if self.benchmark in {'clawarena', 'memoryarena'}:
            if self.user_id not in USERS:
                raise ValueError('personalized benchmarks require a registered simulated user')
        elif self.user_id != 'general':
            raise ValueError('performance feedback uses general, not a personal profile')

    @property
    def namespace(self):
        # Do not concatenate unescaped names: run='a/b', dataset='c' must
        # differ from run='a', dataset='b/c'.
        return 'experiment_' + content_hash([self.run_id, self.benchmark, self.user_id,
                                           self.method, self.model_config_hash])


@dataclass(frozen=True)
class PublicUnit:
    instance: CommonInstance
    independent_task_id: str
    instruction: str
    public_context_json: str = '{}'

    def __post_init__(self):
        _identity(self.independent_task_id)
        _identity(self.instruction)
        context = _object_json(self.public_context_json)
        if self.instance.public_input_hash != content_hash({'instruction': self.instruction, 'context': context}):
            raise ValueError('public input does not match the declared instance hash')

    def solver_payload(self):
        # Never serialize CommonInstance: its metadata can contain native
        # evaluation specs and hidden preferences.
        return {'task_id': self.independent_task_id, 'unit_id': self.instance.common_instance_id,
                'instruction': self.instruction, 'context': _object_json(self.public_context_json)}


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    source_sha256: str
    profile_json: str
    source_example_ids: tuple[str, ...]

    def __post_init__(self):
        if self.user_id not in USERS:
            raise ValueError('unknown simulated user profile')
        if (not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in self.source_sha256)):
            raise ValueError('profile source requires a SHA256 binding')
        evidence = _object_json(self.profile_json)
        if (not isinstance(self.source_example_ids, tuple) or not self.source_example_ids or
                any(not isinstance(i, str) or len(i) != 64 or any(c not in '0123456789abcdef' for c in i)
                    for i in self.source_example_ids) or
                len(set(self.source_example_ids)) != len(self.source_example_ids)):
            raise ValueError('profile requires distinct source example identities')
        validate_profile_payload(evidence, user_id=self.user_id,
                                 encoded_bytes=len(self.profile_json.encode('utf-8')))


def load_user_profile(user_id: str) -> UserProfile:
    """Build the profile from its registered simulator persona; the hash binds its content."""
    text = json.dumps(compact_profile(user_id), ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return UserProfile(user_id, digest, text, (digest,))


@dataclass(frozen=True)
class PrivateEvaluation:
    unit_id: str
    evaluation_json: str

    def __post_init__(self):
        _identity(self.unit_id)
        _object_json(self.evaluation_json)


def feedback_payload(unit: PublicUnit, identity: RunIdentity, *, delivery: str,
                     visible_history: tuple[dict, ...] = (), profile: UserProfile | None = None,
                     training_reference: PrivateEvaluation | None = None, visible_artifacts_json: str = '{}',
                     public_materials_json: str = '{}', visible_operations_json: str = '{}',
                     delivery_context: dict | None = None):
    if unit.instance.benchmark != identity.benchmark or unit.instance.profile_id != identity.user_id:
        raise ValueError('feedback identity differs from the task manifest')
    if not isinstance(delivery, str):
        raise ValueError('delivery must be visible text')
    history = []
    for turn in visible_history:
        if set(turn) != {'role', 'text'} or turn['role'] not in {'user', 'assistant'} or not isinstance(turn['text'], str):
            raise ValueError('feedback history contains non-visible fields or invalid roles')
        history.append(dict(turn))
    payload = {**unit.solver_payload(), 'delivery': delivery, 'visible_history': history}
    materials = _object_json(public_materials_json)
    if materials:
        validate_public_materials(materials, unit)
        payload['public_task_materials'] = materials
    artifacts=_object_json(visible_artifacts_json)
    if artifacts:
        validate_visible_artifacts(artifacts)
        payload['visible_artifacts']=artifacts
    operations = _object_json(visible_operations_json)
    if operations:
        validate_visible_operations(operations, unit=unit, delivery_context=delivery_context)
        payload['observed_operations']=operations
    if identity.benchmark in {'clawarena', 'memoryarena'}:
        if profile is None or profile.user_id != identity.user_id:
            raise ValueError('feedback requires the matching real-user profile')
        if training_reference is not None:
            raise ValueError('personalized benchmarks never give extra gold to feedback')
        payload['user_profile'] = _object_json(profile.profile_json)
    else:
        if profile is not None:
            raise ValueError('performance feedback cannot use a personal profile')
        if training_reference is not None:
            if unit.instance.split != Split.ACQUISITION:
                raise ValueError('test feedback cannot receive a reference or hidden verdict')
            if training_reference.unit_id != unit.instance.common_instance_id:
                raise ValueError('reference belongs to another scoring unit')
            payload['training_reference'] = _object_json(training_reference.evaluation_json)
    return payload


def validate_visible_operations(value, *, unit, delivery_context):
    """Check ownership and the native recorder's public operation projection.

    The recorder, not this schema, proves origin. Its source digest and actual
    delivery boundary must be supplied separately by the task owner.
    """
    context_fields={'session_id','event_id','artifact_hash','execution_evidence_hash'}
    fields={'schema','task_id','session_id','event_id','artifact_hash','source_hash','events','coverage'}
    if isinstance(value, dict) and 'public_dialogue' in value:
        fields.add('public_dialogue')
    if isinstance(value, dict) and 'request_scope' in value:
        fields.add('request_scope')
    if (not isinstance(delivery_context,dict) or set(delivery_context)!=context_fields or
            any(not isinstance(v,str) or not v for v in delivery_context.values()) or
            not isinstance(value,dict) or set(value)!=fields or type(value['schema']) is not int or
            value['schema']!=1 or value['task_id']!=unit.independent_task_id or
            any(value[k]!=delivery_context[k] for k in ('session_id','event_id','artifact_hash')) or
            value['source_hash']!=delivery_context['execution_evidence_hash'] or
            len(value['source_hash'])!=64 or any(c not in '0123456789abcdef' for c in value['source_hash'])):
        raise ValueError('feedback operations require their actual task and delivery boundary')
    if 'public_dialogue' in value:
        from .public_dialogue import validate_public_dialogue
        validate_public_dialogue(value['public_dialogue'], value['event_id'])
    coverage=value['coverage']
    if (not isinstance(coverage,dict) or set(coverage)!={'complete','omitted_events','pending_operations'} or
            type(coverage['complete']) is not bool or
            any(type(coverage[k]) is not int or coverage[k]<0 for k in ('omitted_events','pending_operations')) or
            coverage['complete']!=(coverage['omitted_events']==coverage['pending_operations']==0) or
            not isinstance(value['events'],list)):
        raise ValueError('feedback operation coverage must be explicit and consistent')
    last=0
    for row in value['events']:
        if (not isinstance(row,dict) or set(row)!={'sequence','native_turn_id','event','item'} or
                type(row['sequence']) is not int or row['sequence']<=last or
                not isinstance(row['native_turn_id'],str) or not row['native_turn_id'] or
                row['event'] not in ('started','completed') or not isinstance(row['item'],dict)):
            raise ValueError('feedback requires an ordered actual native operation sequence')
        item=row['item']
        kind=item.get('type')
        if (not isinstance(kind,str) or kind not in OPERATION_FIELDS or set(item)-OPERATION_FIELDS[kind] or
                not isinstance(item.get('id'),str) or not item['id'] or
                (kind=='dynamicToolCall' and (not isinstance(item.get('tool'),str) or not item['tool'] or
                                             item['tool']=='tellonce_review' or item.get('namespace') is not None))):
            raise ValueError('feedback operations contain private or unadmitted fields')
        last=row['sequence']
    if 'request_scope' in value:
        from .operation_scope import validate_request_scope
        validate_request_scope(value['request_scope'], event_id=value['event_id'], events=value['events'])


def validate_public_materials(value, unit):
    """Validate the adapter's immutable current-round public source projection.

    Hashes detect text corruption; the reviewed benchmark adapter, not this
    schema, establishes that files are public and available by this round.
    """
    def digest_ok(v):
        return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
    def path_ok(v):
        return (isinstance(v, str) and bool(v) and '\x00' not in v and '\\' not in v
                and not v.startswith('/') and all(p not in {'', '.', '..'} for p in v.split('/')))
    fields = {'schema', 'task_id', 'unit_id', 'asset_digest', 'applied_updates', 'files', 'omissions', 'complete'}
    if (not isinstance(value, dict) or set(value) != fields or type(value['schema']) is not int
            or value['schema'] != 1 or value['task_id'] != unit.independent_task_id
            or value['unit_id'] != unit.instance.common_instance_id or not digest_ok(value['asset_digest'])
            or not isinstance(value['applied_updates'], list)
            or any(not isinstance(v, str) or not v for v in value['applied_updates'])
            or len(set(value['applied_updates'])) != len(value['applied_updates'])
            or not isinstance(value['files'], dict) or not isinstance(value['omissions'], list)
            or type(value['complete']) is not bool or value['complete'] != (not value['omissions'])):
        raise ValueError('public materials must bind the current task and scoring unit')
    seen = set()
    for path, row in value['files'].items():
        if (not path_ok(path) or not isinstance(row, dict) or set(row) != {'text', 'sha256', 'bytes'}
                or not isinstance(row['text'], str) or not digest_ok(row['sha256'])
                or type(row['bytes']) is not int or row['bytes'] != len(row['text'].encode('utf-8'))
                or row['sha256'] != hashlib.sha256(row['text'].encode('utf-8')).hexdigest()):
            raise ValueError('public material text must be whole and hash-bound')
        seen.add(path)
    for row in value['omissions']:
        if (not isinstance(row, dict) or set(row) != {'path', 'sha256', 'bytes', 'reason'}
                or not path_ok(row['path']) or row['path'] in seen or not digest_ok(row['sha256'])
                or type(row['bytes']) is not int or row['bytes'] < 0 or row['reason'] != 'non_text'):
            raise ValueError('unavailable public material must have explicit coverage')
        seen.add(row['path'])


def validate_visible_artifacts(value, *, expected_hash=None):
    """The host passes only TaskArtifacts' public output projection."""
    if (not isinstance(value,dict) or set(value)!={'snapshot_hash','files','deleted','omissions','coverage'} or
            not isinstance(value['snapshot_hash'],str) or len(value['snapshot_hash'])!=64 or
            any(c not in '0123456789abcdef' for c in value['snapshot_hash']) or
            (expected_hash is not None and value['snapshot_hash']!=expected_hash) or
            not isinstance(value['files'],dict) or not isinstance(value['deleted'],list) or
            not isinstance(value['omissions'],list)):
        raise ValueError('public artifacts must match the captured delivery')
    def path_ok(path):
        return isinstance(path,str) and path and not path.startswith('/') and '..' not in path.split('/')
    if (any(not path_ok(path) or not isinstance(text,str) for path,text in value['files'].items()) or
            any(not path_ok(path) for path in value['deleted']) or
            any(not isinstance(row,dict) or set(row)!={'path','reason'} or not path_ok(row['path']) or
                not isinstance(row['reason'],str) for row in value['omissions'])):
        raise ValueError('public artifact projection contains invalid output fields')
    coverage=value['coverage']
    counts={'text_files_total','text_files_omitted','captured_omissions','omission_details_omitted',
            'deleted_total','deleted_details_omitted'}
    if (not isinstance(coverage,dict) or set(coverage)!=counts|{'complete','deletions_complete'} or
            any(type(coverage[key]) is not int or coverage[key]<0 for key in counts) or
            type(coverage['complete']) is not bool or type(coverage['deletions_complete']) is not bool or
            coverage['text_files_total']!=len(value['files'])+coverage['text_files_omitted'] or
            coverage['captured_omissions']!=len(value['omissions'])+coverage['omission_details_omitted'] or
            coverage['deleted_total']!=len(value['deleted'])+coverage['deleted_details_omitted']):
        raise ValueError('public artifact coverage is inconsistent')
    complete=(not coverage['captured_omissions'] and not coverage['text_files_omitted']
              and not coverage['deleted_details_omitted'] and coverage['deletions_complete'])
    if coverage['complete']!=complete or len(json.dumps(value,ensure_ascii=False,allow_nan=False))>64000:
        raise ValueError('public artifact completeness or size is invalid')


def validate_task_split(units: tuple[PublicUnit, ...]):
    if not units:
        raise ValueError('task manifest is empty')
    task_splits, unit_ids, tasks_by_family, native_units = {}, set(), {}, set()
    for unit in units:
        instance = unit.instance
        if instance.split not in {Split.ACQUISITION, Split.ID_TEST, Split.TASK_OOD}:
            raise ValueError('unsupported split in the current protocol')
        if instance.common_instance_id in unit_ids:
            raise ValueError('duplicate scoring unit')
        unit_ids.add(instance.common_instance_id)
        native_unit = (instance.benchmark, instance.dataset_revision, unit.independent_task_id,
                       instance.subtask_id, instance.profile_id)
        if native_unit in native_units:
            raise ValueError('one native scoring unit has multiple manifest identities')
        native_units.add(native_unit)
        # A task cannot migrate across splits for another user or method.
        key = (instance.benchmark, instance.dataset_revision, unit.independent_task_id)
        if key in task_splits and task_splits[key] != instance.split:
            raise ValueError('rounds of one independent task cross train/test splits')
        task_splits[key] = instance.split
        tasks_by_family.setdefault((instance.benchmark, instance.dataset_revision,
                                    instance.profile_id, instance.task_family), set()).add(instance.split)
    for splits in tasks_by_family.values():
        if Split.ID_TEST in splits and Split.ACQUISITION not in splits:
            raise ValueError('ID test family has no independent training task')
        if Split.TASK_OOD in splits and Split.ACQUISITION in splits:
            raise ValueError('OOD family was already used in training')
