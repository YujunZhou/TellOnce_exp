"""One supervised ClawArena phase in a fresh process.

The CLI has no live-provider fallback or synthetic flag. Unit tests substitute
leaves in-process; real launches execute the code in this repository.
"""
from tellonce_common.runtime_updates import execution_check_transport, task_timeout_s
import json
import os
from pathlib import Path
import sys
import threading
import traceback
import uuid

from tellonce_common.task_checkpoint import TaskCheckpoint
from tellonce_common.deferred_scoring import DeferredScoring
from tellonce_common.feedback_review_factory import make_feedback_agent
from tellonce_common.claude_native import ClaudeConfig
from tellonce_common.claude_auth import ClaudeCredentialSource
from tellonce_common.managed_model_requests import ManagedModelRequests
from tellonce_common.native_auth import NativeAuthLease
from tellonce_common.native_auth_owner import NativeAccountProcess, NativeAuthOwner
from tellonce_common.preference_transport import PreferenceJudge
from tellonce_common.provenance import canonical_json_bytes, content_hash
from tellonce_common.provider_credentials import ProviderKeyFile
from tellonce_common.rule_appeal import AppealReviewer
from tellonce_common.tellonce_method import TellonceAPI, TellonceMethod
from tellonce_common.text_roles import TextRoles
from tellonce_common.worker_signals import WorkerSignals
from .native_phase import ClawPhase, open_claw_phase, evaluate_claw_phase
from .native_run_spec import ClawRunSpec
from .native_task import ClawTask, open_claw_scorer


ACTIONS = frozenset({'training_interaction', 'test_interaction', 'training_evaluation', 'test_evaluation'})


def verify_execution(spec):
    """Reject project code imported from outside this repository's experiments directory."""
    root = Path(spec.bundle['roots']['project'])/'experiments'
    if Path(sys.executable).resolve() != spec.python:
        raise ValueError('worker is using a different interpreter from its run specification')
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in {'tellonce_common', 'tellonce_clawarena'}:
            continue
        origin = Path(getattr(module, '__file__', '')).resolve()
        if not origin.is_relative_to(root):
            raise ValueError('worker imported project code outside this repository')


class NativeCredentials:
    """Lazily start one native account owner for this serial worker."""
    def __init__(self, spec, *, directory, authorize, record):
        self.spec, self.directory, self.authorize, self.record = spec, directory, authorize, record
        self._lock = threading.RLock()
        self.owner = None

    def provision(self, home, server):
        with self._lock:
            if self.owner is None:
                process = NativeAccountProcess(binary=self.spec.binary, auth_home=self.spec.auth_home,
                    private_root=self.directory/'native-account', model_catalog=self.spec.catalog,
                    authorize=lambda action: self.authorize('native_account_'+action, {}), record=self.record)
                try:
                    self.owner = NativeAuthOwner(server=process, auth_home=self.spec.auth_home,
                        authorize=lambda action: self.authorize('native_credentials_'+action, {}), record=self.record)
                except BaseException:
                    process.close()
                    raise
            return NativeAuthLease(server=server, source=self.owner, record=self.record)

    def close(self):
        with self._lock:
            if self.owner is not None:
                self.owner.close()


def _previous_phase(spec, phase, receipt, authorize, record):
    manifest, closed, _, _ = open_claw_phase(directory=spec.directory/(phase+'-phase'),
        expected_receipt=receipt, identity=spec.identity, binding=spec.binding, authorize=authorize, record=record)
    planned = [row['scenario_id'] for row in spec.value['tasks']
               if (row['split'] == 'acquisition') == (phase == 'training')]
    if manifest['phase'] != phase or [row['task_id'] for row in manifest['tasks']] != planned:
        raise ValueError('prior completed phase differs from the admitted complete task plan')
    return closed


def run_phase(spec, *, action, prior_receipt, signals, record, attempt_directory, test_task_id=None):
    from tellonce_common.test_protocol import PLAN_A, INTERACTIVE, NoTestFeedback
    protocol = getattr(spec, 'test_protocol', INTERACTIVE)
    source = getattr(spec, 'training_source', spec)
    if protocol == PLAN_A and not action.startswith('test_'):
        raise ValueError('plan A extension cannot create or resume training')
    direct_test = spec.identity.method == 'no_memory' and action == 'test_interaction' and prior_receipt is None
    if action not in ACTIONS or (not direct_test and (action == 'training_interaction') != (prior_receipt is None)):
        raise ValueError('phase action requires its exact predecessor receipt')
    spec.verify()
    verify_execution(spec)
    phase = 'training' if action.startswith('training_') else 'test'
    interaction = action.endswith('_interaction')
    admitted_identity, admitted_binding = spec.identity, spec.binding
    def authorize(action, detail):
        signals.assert_healthy()
        # The immutable runtime was admitted at phase entry. Rewalking all
        # sources/binaries for every tool, feedback and memory action adds no
        # task information. Keep the cheap run-ownership check here instead.
        if spec.identity != admitted_identity or spec.binding != admitted_binding:
            raise ValueError('worker identity changed after phase admission')
        # Raw public task inputs stay in their private task records, not this audit.
        record({'type': 'worker_admission', 'action': action, 'phase': phase,
                'namespace': spec.identity.namespace, 'binding': spec.binding,
                'runtime_bundle_hash':getattr(spec,'runtime_bundle_reference',spec.value['source_bundle'])['bundle_hash']})
    prior = None
    if prior_receipt is not None:
        if protocol == PLAN_A and interaction:
            if prior_receipt != spec.extension['training_receipt']:
                raise ValueError('plan A requires its admitted original training receipt')
            prior = spec.prior
        else:
            prior = _previous_phase(spec, 'training' if interaction else phase, prior_receipt, authorize, record)
    if not spec.directory.is_dir() or attempt_directory.parent != spec.directory/'worker-attempts':
        raise ValueError('worker must use its prepared run and a fresh private attempt directory')
    if interaction and test_task_id is None:
        closed_result=spec.directory/(phase+'-phase')/'phase-result.json'
        if closed_result.is_file():
            receipt=content_hash(json.loads(closed_result.read_text()))
            _previous_phase(spec,phase,receipt,authorize,record)
            return {'action':action,'phase_receipt':receipt}
    from tellonce_common.parallel_test import (worker_count, has_parallel_state, run_parallel_test,
                                            scope_paths, task_recovery)
    if test_task_id is not None and action != 'test_interaction':
        raise ValueError('parallel child can only execute test interaction')
    if action == 'test_interaction' and test_task_id is None and (worker_count(spec) > 1 or has_parallel_state(spec)):
        authorize('start_parallel_test', {'workers': worker_count(spec)})
        return run_parallel_test(spec, prior=prior, prior_receipt=prior_receipt,
                                 signals=signals, record=record, authorize=authorize)
    phase_directory = spec.directory/(phase+'-phase')
    queue_directory = spec.directory/(phase+'-queue')
    roles_directory = spec.directory/'roles'
    if test_task_id is not None:
        phase_directory, queue_directory, roles_directory = scope_paths(spec, test_task_id)
    recovery=None
    resume_training=False
    if interaction:
        selected=[row for row in spec.value['tasks']
                  if (row['split']=='acquisition')==(phase=='training')]
        task_directories={row['scenario_id']:'task-'+row['scenario_id'] for row in selected}
        recovery=TaskCheckpoint(spec.directory,namespace=spec.identity.namespace,binding=spec.binding,
                                phase=phase,task_directories=task_directories)
        if test_task_id is not None:
            recovery = task_recovery(spec, test_task_id)
        recovery.restore()
        resume_training=phase=='training' and recovery.restored
    attempt_directory.mkdir(mode=0o700)
    credentials = NativeCredentials(spec, directory=attempt_directory, authorize=authorize, record=record)
    memory = None
    try:
        # The frozen model configuration admits at most one HTTP provider.
        # Native-only runs must not acquire an unrelated provider credential.
        providers = {row['kind'] for row in spec.value['models']['roles'].values()
                     if row['kind'] not in {'native','claude_native'}}
        if len(providers) > 1:
            raise ValueError('one provider-bound HTTP credential source per experiment required')
        key = (ProviderKeyFile(spec.api_key_file, provider=next(iter(providers)), authorize=authorize)
               if providers else None)
        role_arguments = dict(directory=roles_directory, identity=spec.identity, binding=spec.binding,
            model_config=spec.value['models'], binary=spec.binary, catalog=spec.catalog, tokenizer=spec.tokenizer,
            authorize=authorize, record=record, check_health=signals.assert_healthy,
            provision_auth=credentials.provision, api_key_source=key)
        claude_credentials=None
        if isinstance(spec.solver_config,ClaudeConfig):
            claude_credentials=ClaudeCredentialSource(path=spec.claude_auth_file,authorize=authorize)
            role_arguments.update(claude_binary=spec.claude_binary,claude_credentials=claude_credentials)
        if test_task_id is not None and roles_directory.exists() and not (roles_directory/'roles.json').exists():
            # Keep uncommitted native/HTTP attempt evidence, but don't mistake
            # a leftover directory for a valid role manifest after rollback.
            roles_directory.rename(roles_directory.with_name('roles-uncommitted-'+uuid.uuid4().hex))
        create_roles = (action == 'training_interaction' and not resume_training) or (
            (direct_test or test_task_id is not None or protocol == PLAN_A) and not roles_directory.exists())
        roles = (TextRoles.create if create_roles else TextRoles.open)(**role_arguments)
        from tellonce_common.api_luna_roles import configure_api_luna_roles
        roles = configure_api_luna_roles(spec, roles, phase=phase)
        if test_task_id is not None and create_roles:
            recovery.save(recovery.value['completed'] if recovery.value else [],
                          partial=recovery.value.get('partial') if recovery.value else None)
        if not interaction:
            judge = PreferenceJudge(roles.get('preference'))
            def scorer(result):
                sid = result.outcome.task_id
                return open_claw_scorer(result=result, identity=spec.identity, units=spec.units[sid],
                    profile=spec.profile, source_root=spec.dataset_root, expected_asset_digest=spec.assets[sid].digest,
                    binding=spec.binding, binary=spec.binary, python=spec.python, python_roots=spec.python_roots,
                    directory=attempt_directory/('grader-'+sid), preference_judge=judge,
                    authorize=authorize, record=record, wait_retry=signals.wait_retry,
                    role_deadline_s=spec.value['role_deadline_s'])
            value = evaluate_claw_phase(directory=spec.directory/(phase+'-phase'), expected_receipt=prior_receipt,
                identity=spec.identity, binding=spec.binding, scorer_factory=scorer, authorize=authorize,
                record=record, check_health=signals.assert_healthy)
            return {'action': action, 'phase_receipt': prior_receipt, 'evaluation_digest': content_hash(value)}
        api = None
        if spec.identity.method == 'trace':
            api = TellonceAPI(library=getattr(spec, 'trace_tellonce_library', Path(spec.bundle['roots']['tellonce'])/'lib'), authorize=authorize)
            invoke = ManagedModelRequests(roles.get('memory'), request_error=api.ModelRequestError,
                check_transport=execution_check_transport(spec, roles),
                wait_retry=signals.wait_retry, deadline_s=spec.value['role_deadline_s'])
            method_arguments = dict(directory=source.directory/'method', identity=spec.identity, api=api, invoke=invoke,
                binding=spec.binding, execution_enabled=spec.value['execution_enabled'], authorize=authorize,
                record=record, check_health=signals.assert_healthy)
            if phase == 'training':
                memory = (TellonceMethod.open_training if resume_training else TellonceMethod.create)(**method_arguments, compiler_id=spec.value['compiler_id'])
            else:
                # Runtime settings may accept bounded learner failures recorded in the frozen audit
                # (learning_admission.require_trace_learning); by default incomplete learning fails closed.
                runtime_reference = getattr(spec, 'runtime_bundle_reference', None) or {}
                memory = TellonceMethod.open_test(**method_arguments, expected_receipt=prior['memory_receipt'],
                    admitted_learning_audit=runtime_reference.get('admitted_learning_audit')
                                            if isinstance(runtime_reference, dict) else None)
        if resume_training and spec.identity.method=='trace':
            memory._event_ids={tuple(row) for row in recovery.value['memory_event_ids']}
        feedback = (NoTestFeedback(spec.identity) if protocol == PLAN_A else
                    make_feedback_agent(spec, roles, phase=phase, record=record))
        hooks = spec.value['execution_enabled']
        appeal = AppealReviewer(roles.get('appeal')) if hooks else None
        queue = (DeferredScoring.open_unsealed if recovery.restored else DeferredScoring)(directory=queue_directory, identity=spec.identity,
            phase=phase, binding=spec.binding, authorize=authorize, record=record)
        tasks = tuple(ClawTask(identity=spec.identity, units=spec.units[row['scenario_id']], profile=spec.profile,
            source_root=spec.dataset_root, expected_asset_digest=spec.assets[row['scenario_id']].digest,
            directory=spec.directory/('task-'+row['scenario_id']), queue=queue, feedback=feedback,
            test_protocol=protocol, completed_receipt=recovery.completed_receipt(row['scenario_id']), recovery=recovery,
            binding=spec.binding, binary=getattr(spec,'solver_binary',spec.binary), config=spec.solver_config, source_catalog=spec.catalog,
            authorize=authorize, record=record, hooks_enabled=hooks, memory_factory=memory,
            score_before_repair=spec.value['score_before_repair'], observation_type=api.Observation if api else None,
            event_type=api.Event if api else None, appeal_reviewer=appeal, provision_auth=claude_credentials or credentials.provision,
            timeout_s=task_timeout_s(spec), wait_retry=signals.wait_retry,
            role_deadline_s=spec.value['role_deadline_s']) for row in spec.value['tasks']
            if (row['split'] == 'acquisition') == (phase == 'training')
            and (test_task_id is None or row['scenario_id'] == test_task_id))
        owner = ClawPhase(directory=phase_directory, queue=queue, tasks=tasks, memory=memory,
            authorize=authorize, record=record, check_health=signals.assert_healthy, checkpoint=signals.checkpoint, recovery=recovery)
        receipt = owner.run()
        return {'action': action, 'phase_receipt': receipt}
    finally:
        try:
            if memory is not None:
                memory.close()
        finally:
            credentials.close()


def main():
    # All model/library diagnostics go to the private stderr capture. Stdout is
    # reserved for bounded heartbeats, checkpoints and the final result.
    control = sys.stdout
    sys.stdout = sys.stderr
    try:
        raw = sys.stdin.buffer.read(1024*1024+1)
        if len(raw) > 1024*1024:
            raise ValueError('worker request exceeded its bound')
        request = json.loads(raw)
        fields = {'namespace', 'binding', 'job_id', 'supervisor_attempt', 'payload', 'heartbeat_interval_s'}
        payload = request.get('payload', {})
        if (set(request) != fields or not isinstance(payload, dict) or
                set(payload) not in ({'spec_path', 'spec_hash', 'action', 'prior_receipt'},
                                     {'spec_path', 'spec_hash', 'action', 'prior_receipt', 'test_task_id'}) or
                payload['action'] not in ACTIONS or type(request['supervisor_attempt']) is not int or
                request['supervisor_attempt'] not in {1, 2} or
                (payload['action'].endswith('_interaction') and request['supervisor_attempt'] != 1)):
            raise ValueError('invalid admitted supervisor request')
        with WorkerSignals(stream=control, interval_s=request['heartbeat_interval_s']) as signals:
            spec = ClawRunSpec.load(path=payload['spec_path'], expected_digest=payload['spec_hash'])
            if request['namespace'] != spec.identity.namespace or request['binding'] != spec.binding:
                raise ValueError('worker belongs to a different run namespace')
            verify_execution(spec)
            # Each attempt has separate events and native process homes. Shared
            # text journals and sealed scoring results retain consumed retries.
            events = spec.directory/'worker-attempts'/(str(uuid.uuid4())+'.jsonl')
            lock = threading.Lock()
            descriptor = os.open(events, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as stream:
                def record(event):
                    data = canonical_json_bytes(event)+b'\n'
                    with lock:
                        stream.write(data)
                        stream.flush()
                result = run_phase(spec, action=payload['action'], prior_receipt=payload['prior_receipt'],
                    signals=signals, record=record, attempt_directory=events.with_suffix(''),
                    test_task_id=payload.get('test_task_id'))
                stream.flush()
                os.fsync(stream.fileno())
            signals.result(result)
        return 0
    except Exception as exc:
        # Keep useful locations in the supervisor's private diagnostics without
        # serializing exception bodies, source lines, locals or credential data.
        frames = [{'file': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name}
                  for frame in traceback.extract_tb(exc.__traceback__)[-16:]]
        print(json.dumps({'type': 'native_worker_failed', 'error_type': type(exc).__name__, 'frames': frames}),
              file=sys.stderr, flush=True)
        return 1
    finally:
        sys.stdout = control


if __name__ == '__main__':
    raise SystemExit(main())
