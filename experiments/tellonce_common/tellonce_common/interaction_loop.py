"""One native session per complete task and bounded, visible user corrections.

The supervisor supplies isolated native sessions, experiment-owned memory,
single-attempt text/scoring transports, and a durable audit sink. This module
never launches models itself. Source admission, native tool interception,
artifact capture and phase-wide memory freeze remain supervisor obligations.
"""
from dataclasses import dataclass, asdict
import json
import math
import re
import time
import uuid

from .contracts import Split
from .feedback_agent import FeedbackDecision
from .interaction_metrics import Delivery, PreferenceVerdict, UnitTrace
from .provenance import content_hash
from .role_inputs import PublicUnit, RunIdentity, UserProfile, validate_visible_artifacts
from .public_task_contracts import runtime_public_payload, runtime_feedback_payload as feedback_payload
from .memory_input import validate_memory_kind
from .request_budget import bounded_deadline, request_budget, RequestDeadline, supervised_retry_wait
from .text_request_journal import TextRequestDeferred
from .test_protocol import INTERACTIVE, PLAN_A, validate_protocol


class RecoverableCallError(RuntimeError):
    """A transport failure with a safe code, never raw provider stderr/secrets."""
    def __init__(self, code: str, *, retryable: bool = True, retry_after_s: float | None = None):
        if not isinstance(code, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,79}', code):
            raise ValueError('call failures require a bounded diagnostic code')
        self.code = code
        if (type(retryable) is not bool or (retry_after_s is not None and
                (type(retry_after_s) not in {int,float} or not math.isfinite(retry_after_s) or retry_after_s<0))):
            raise ValueError('retry policy must have a finite nonnegative delay')
        self.retryable,self.retry_after_s=retryable,retry_after_s
        super().__init__(code)


@dataclass(frozen=True)
class MemoryContext:
    text: str
    version: str
    kind: str = 'selected_requirements'

    def __post_init__(self):
        if not isinstance(self.text, str) or not isinstance(self.version, str) or not self.version:
            raise ValueError('memory text must be bound to an explicit version')
        validate_memory_kind(self.kind, self.text)


@dataclass(frozen=True)
class SolverAnswer:
    text: str
    artifact_hash: str
    internal_repairs: int = 0
    before_repair_artifact_hash: str = ''
    before_repair_text: str | None = None
    visible_artifacts_json: str = '{}'
    execution_evidence_hash: str = ''
    before_repair_execution_evidence_hash: str = ''

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise ValueError('solver delivery requires the actual visible text')
        for value in (self.execution_evidence_hash,self.before_repair_execution_evidence_hash):
            if not isinstance(value,str) or (value and not re.fullmatch('[a-f0-9]{64}',value)):
                raise ValueError('native operation evidence requires a content hash')
        if self.before_repair_execution_evidence_hash and not self.before_repair_artifact_hash:
            raise ValueError('pre-repair operation evidence requires its captured draft')
        if not isinstance(self.visible_artifacts_json,str):
            raise ValueError('visible artifact projection must be immutable JSON')
        artifacts=json.loads(self.visible_artifacts_json)
        if artifacts != {}:
            validate_visible_artifacts(artifacts,expected_hash=self.artifact_hash)
        Delivery(self.artifact_hash, None, None, (), self.internal_repairs,
                 self.before_repair_artifact_hash)
        if self.before_repair_text is not None:
            if not isinstance(self.before_repair_text,str) or not self.before_repair_artifact_hash:
                raise ValueError('captured pre-repair text requires its artifact identity')
            if self.before_repair_artifact_hash == self.artifact_hash and self.before_repair_text != self.text:
                raise ValueError('the same captured artifact cannot have different answer texts')


@dataclass(frozen=True)
class Grade:
    native_score: float | None
    native_pass: bool | None
    preferences: tuple[PreferenceVerdict, ...]
    technical_missing: bool = False
    scoring_pending: bool = False

    def __post_init__(self):
        Delivery('grade-validation', self.native_score, self.native_pass, self.preferences,
                 technical_missing=self.technical_missing,scoring_pending=self.scoring_pending)


@dataclass(frozen=True)
class UnitOutcome:
    unit_id: str
    trace: UnitTrace | None
    feedback_status: str
    error_code: str = ''


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    session_id: str
    units: tuple[UnitOutcome, ...]
    aborted: bool


class InteractionLoop:
    def __init__(self, *, identity: RunIdentity, profile: UserProfile | None,
                 start_solver, start_memory, feedback, score, audit, score_before_repair: bool = False,
                 prepare_unit=None, public_materials=None, feedback_operations=None,
                 wait_retry=None, read_deadline_s=60, resume=None, commit_unit=None,
                 test_protocol=INTERACTIVE, training_correction_limit=1):
        if type(score_before_repair) is not bool:
            raise ValueError('pre-delivery repair evaluation must be explicit')
        self.score_before_repair = score_before_repair
        if prepare_unit is not None and not callable(prepare_unit):
            raise ValueError('native unit preparation must be an explicit callback')
        self.prepare_unit = prepare_unit
        if public_materials is not None and not callable(public_materials):
            raise ValueError('public materials require a benchmark projection callback')
        self.public_materials = public_materials
        if feedback_operations is not None and not callable(feedback_operations):
            raise ValueError('feedback operations require a captured native evidence callback')
        self.feedback_operations = feedback_operations
        bounded_deadline(read_deadline_s)
        if wait_retry is not None and not callable(wait_retry):
            raise ValueError('read-only retry waits require an explicit callback')
        self.read_deadline_s = read_deadline_s
        self.wait_retry = wait_retry or (lambda seconds, **kwargs: time.sleep(seconds))
        self.identity, self.profile = identity, profile
        if type(training_correction_limit) is not int or training_correction_limit not in (1, 2):
            raise ValueError('training correction limit must be one, or explicit legacy two')
        self.training_correction_limit = training_correction_limit
        self.test_protocol = validate_protocol(test_protocol, benchmark=identity.benchmark)
        self.start_solver, self.start_memory = start_solver, start_memory
        self.feedback, self.score, self.audit = feedback, score, audit
        self._opened_tasks = set()
        self.resume, self.commit_unit = resume, commit_unit

    def _event(self, kind, start, **data):
        self.audit({'type': kind, 'namespace': self.identity.namespace,
                    'task_id': start['task_id'], 'session_id': start['session_id'], **data})

    def _read_call(self, role, invoke, start, unit_id, event_id):
        # Only read-only feedback/scoring calls get a controller retry. Memory
        # owns its own retry budget; a solver may already have changed files.
        call_id=content_hash([self.identity.namespace,start['session_id'],unit_id,event_id,role])
        deadline = bounded_deadline(self.read_deadline_s)
        def wait(delay, context, reason):
            self._event('call_retry_wait', start, role=role, unit_id=unit_id,
                        call_id=call_id, attempt=context['attempt'], retry_after_s=delay, error_code=reason)
            try:
                supervised_retry_wait(delay, deadline=deadline, wait_retry=self.wait_retry,
                                      context=context, reason=reason)
            except RequestDeadline:
                self._event('call_wait_deadline', start, role=role, unit_id=unit_id, call_id=call_id)
                return False
            return True
        for attempt in (1,2):
            context={'phase':'training' if start['training'] else 'test','task_id':start['task_id'],
                     'session_id':start['session_id'],'event_id':event_id,'role':role,
                     'call_id':call_id,'attempt':attempt}
            while True:
                if time.monotonic() >= deadline:
                    self._event('call_wait_deadline', start, role=role, unit_id=unit_id, call_id=call_id)
                    return None
                self._event('call_started', start, role=role, unit_id=unit_id, event_id=event_id,
                            call_id=call_id, attempt=attempt)
                try:
                    with request_budget(max(.000001, deadline-time.monotonic())):
                        value = invoke(context)
                except TextRequestDeferred as exc:
                    if not wait(exc.retry_after_s, context, exc.code):
                        return None
                    continue  # Ownership/cooldown waits consume no model attempt.
                except RequestDeadline:
                    self._event('call_wait_deadline', start, role=role, unit_id=unit_id, call_id=call_id)
                    return None
                except RecoverableCallError as exc:
                    self._event('call_failed', start, role=role, unit_id=unit_id,
                                event_id=event_id,call_id=call_id,attempt=attempt,error_code=exc.code,
                                retryable=exc.retryable,retry_after_s=exc.retry_after_s)
                    if not exc.retryable or attempt==2:
                        return None
                    if exc.retry_after_s and not wait(exc.retry_after_s, context, exc.code):
                        return None
                    break
                else:
                    self._event('call_completed', start, role=role, unit_id=unit_id,
                                event_id=event_id,call_id=call_id,attempt=attempt)
                    return value
        return None

    def run_task(self, units: tuple[PublicUnit, ...], *, training_references=None) -> TaskOutcome:
        if getattr(self.feedback,'identity',self.identity)!=self.identity:
            raise ValueError('feedback transport belongs to another experiment, user or method')
        if not isinstance(units, tuple) or not units:
            raise ValueError('one complete ordered task manifest is required')
        first = units[0]
        task_id = first.independent_task_id
        instance = first.instance
        original_only = self.test_protocol == PLAN_A and instance.split != Split.ACQUISITION
        correction_limit = (0 if original_only else
                            self.training_correction_limit if instance.split == Split.ACQUISITION else 2)
        if instance.split not in {Split.ACQUISITION, Split.ID_TEST, Split.TASK_OOD}:
            raise ValueError('unsupported task phase')
        seen = set()
        references = dict(training_references or {})
        for unit in units:
            item = unit.instance
            if (unit.independent_task_id != task_id or item.split != instance.split or
                    item.dataset_revision != instance.dataset_revision or item.task_family != instance.task_family or
                    item.benchmark != self.identity.benchmark or item.profile_id != self.identity.user_id or
                    item.common_instance_id in seen):
                raise ValueError('mixed, duplicated or differently owned task units')
            seen.add(item.common_instance_id)
            # Validate reference/profile authority before opening any runtime.
            feedback_payload(unit, self.identity, delivery='', profile=self.profile,
                             training_reference=references.get(item.common_instance_id))
        if set(references) - seen:
            raise ValueError('training references contain a unit outside this task')
        task_key = (instance.dataset_revision, task_id)
        if task_key in self._opened_tasks:
            raise ValueError('task already opened; resume requires a validated checkpoint')
        self._opened_tasks.add(task_key)
        start = {'task_id': task_id, 'session_id': str(uuid.uuid4()),
                 'namespace': self.identity.namespace, 'training': instance.split == Split.ACQUISITION,
                 'dataset_revision': instance.dataset_revision}
        solver = memory = None
        history, outcomes = [], []
        output_summary, previous_feedback, unreviewed_outputs = '', None, []
        if self.resume is not None:
            from .task_outcomes import outcome_from_dict
            saved = self.resume
            # Old checkpoints did not record this setting and allowed two.
            # Even a prefix with no corrections belongs to that old policy.
            if start['training'] and (type(saved.get('training_correction_limit', 2)) is not int or
                    saved.get('training_correction_limit', 2) != self.training_correction_limit):
                raise ValueError('cannot resume a different training correction limit')
            restored = outcome_from_dict(saved['outcome'])
            if (saved['start'] != {**start, 'session_id': restored.session_id} or
                    restored.task_id != task_id or restored.aborted or
                    [v.unit_id for v in restored.units] != [u.instance.common_instance_id for u in units[:len(restored.units)]] or
                    any(v.error_code or v.trace is None for v in restored.units)):
                raise ValueError('round recovery differs from the completed task prefix')
            start = dict(saved['start'])
            history, outcomes = list(saved['history']), list(restored.units)
            output_summary, previous_feedback = saved['output_summary'], saved['previous_feedback']
            unreviewed_outputs = list(saved['unreviewed_outputs'])
            if saved.get('test_protocol', INTERACTIVE) != self.test_protocol:
                raise ValueError('cannot resume another test protocol or reuse interactive answers as plan A')
            if original_only and any(u.trace is None or u.trace.corrections or
                    len(u.trace.deliveries) != 1 for u in restored.units):
                raise ValueError('plan A checkpoint contains extra user interaction')
            if start['training'] and any(len(u.trace.corrections) > correction_limit or
                    len(u.trace.deliveries) > correction_limit + 1 for u in restored.units):
                raise ValueError('checkpoint exceeds the training correction limit')
        self._event('task_resumed' if self.resume is not None else 'task_started', start,
                    unit_ids=[u.instance.common_instance_id for u in units], completed_units=len(outcomes),
                    test_protocol=self.test_protocol if not start['training'] else INTERACTIVE,
                    correction_limit=correction_limit)
        aborted = False
        try:
            memory = self.start_memory(dict(start))
            solver = self.start_solver(dict(start))
            if solver.session_id != start['session_id']:
                raise ValueError('native session did not honor the fresh task identity')
            for unit in units[len(outcomes):]:
                unit_id = unit.instance.common_instance_id
                if aborted:
                    outcomes.append(UnitOutcome(unit_id, None, 'not_attempted', 'prior_unit_aborted'))
                    self._event('unit_not_attempted', start, unit_id=unit_id, error_code='prior_unit_aborted')
                    continue
                self._event('unit_started', start, unit_id=unit_id)
                if self.prepare_unit is not None:
                    try:
                        self.prepare_unit(unit, start=dict(start))
                    except RecoverableCallError as exc:
                        aborted = True
                        outcomes.append(UnitOutcome(unit_id, None, 'not_prepared', exc.code))
                        self._event('unit_preparation_failed', start, unit_id=unit_id, error_code=exc.code)
                        continue
                    self._event('unit_prepared', start, unit_id=unit_id)
                materials_json = (self.public_materials(unit)
                    if self.public_materials is not None and not original_only else '{}')
                feedback_payload(unit, self.identity, delivery='', profile=self.profile,
                    training_reference=references.get(unit_id), public_materials_json=materials_json)
                trace = UnitTrace(unit_id, task_id)
                user_text = unit.instruction
                learnable_input = True
                learning_text = ''
                status, error = 'budget_exhausted', ''
                for delivery_index in range(correction_limit + 1):
                    event_id = content_hash([self.identity.namespace, start['session_id'], unit_id, delivery_index])
                    self._event('user_input', start, unit_id=unit_id, event_id=event_id,
                                delivery_index=delivery_index, text=user_text)
                    history.append({'role': 'user', 'text': user_text})
                    try:
                        selected = memory.user_boundary(unit_payload=runtime_public_payload(unit), user_text=user_text, event_id=event_id)
                    except RecoverableCallError as exc:
                        selected = MemoryContext('', 'unavailable')
                        self._event('memory_unavailable', start, unit_id=unit_id, error_code=exc.code)
                    if not isinstance(selected, MemoryContext):
                        raise TypeError('memory port must return an explicit context')
                    self._event('memory_selected', start, unit_id=unit_id, event_id=event_id,
                                version=selected.version, text_hash=content_hash(selected.text),
                                memory_kind=selected.kind, input_protocol=2)
                    # Pin before admission: even an immediately finishing
                    # worker cannot make this input's new rules active mid-turn.
                    # Benchmark-provided questions remain task context, not
                    # personal learning signals. Only actual simulated-user
                    # feedback enters experiment memory, for every method.
                    # Retrieval still runs on all inputs, including the first.
                    # This adapter boundary does not change Tellonce's product
                    # behavior for real user prompts outside these experiments.
                    # A correction that only restates the task's own requirement (learnable False) is
                    # still the solver's next input but is no learning signal for any method.
                    if start['training'] and delivery_index > 0 and not learnable_input:
                        self._event('learning_skipped_task_correction', start, unit_id=unit_id, event_id=event_id)
                    if start['training'] and delivery_index > 0 and learnable_input:
                        if learning_text and learning_text != user_text:
                            self._event('learning_input_restricted_to_preference', start, unit_id=unit_id,
                                        event_id=event_id, text_hash=content_hash(learning_text))
                        try:
                            memory.enqueue_user(unit_payload=runtime_public_payload(unit), user_text=user_text,
                                                event_id=event_id, learning_text=learning_text or None)
                        except RecoverableCallError as exc:
                            self._event('learning_admission_failed', start, unit_id=unit_id, error_code=exc.code)
                    request = {**runtime_public_payload(unit), 'session_id': start['session_id'],
                               'event_id': event_id, 'user_text': user_text,
                               'memory_text': selected.text, 'memory_version': selected.version,
                               'memory_kind': selected.kind}
                    self._event('solver_started', start, unit_id=unit_id, event_id=event_id)
                    try:
                        answer = solver.respond(request)
                    except RecoverableCallError as exc:
                        self._event('solver_failed', start, unit_id=unit_id, event_id=event_id, error_code=exc.code)
                        aborted, status, error = True, 'not_delivered', exc.code
                        break
                    if not isinstance(answer, SolverAnswer):
                        raise TypeError('native runtime must return a captured delivery')
                    if self.score_before_repair and answer.before_repair_text is None:
                        raise ValueError('compiler ablation requires the actual captured pre-repair text and files')
                    self._event('answer_delivered', start, unit_id=unit_id, event_id=event_id, **asdict(answer))
                    try:
                        memory.observe_assistant(answer)
                    except RecoverableCallError as exc:
                        self._event('memory_context_unavailable', start, unit_id=unit_id, error_code=exc.code)
                    grade = self._read_call('score', lambda c:self.score(unit_id,answer,context=c), start,unit_id,event_id)
                    if grade is None:
                        grade = Grade(None, None, (PreferenceVerdict('__evaluation_unavailable__', 'unknown'),), True)
                    if not isinstance(grade, Grade):
                        raise TypeError('private scorer must return a grade')
                    before = None
                    if self.score_before_repair:
                        draft = SolverAnswer(answer.before_repair_text, answer.before_repair_artifact_hash,
                            execution_evidence_hash=answer.before_repair_execution_evidence_hash)
                        # Equal files do not prove equal operation histories.
                        # Reuse only identical captured evidence, or a legacy
                        # test fixture with no repair and no operation evidence.
                        same_operations = (draft.execution_evidence_hash == answer.execution_evidence_hash and
                                           (bool(answer.execution_evidence_hash) or not answer.internal_repairs))
                        before_grade = grade if draft.artifact_hash == answer.artifact_hash and same_operations else self._read_call(
                            'score_before_repair', lambda c:self.score(unit_id,draft,context=c), start,unit_id,event_id)
                        if before_grade is None:
                            before_grade = Grade(None,None,(PreferenceVerdict('__evaluation_unavailable__','unknown'),),True)
                        if not isinstance(before_grade,Grade):
                            raise TypeError('private draft scorer must return a grade')
                        before = Delivery(draft.artifact_hash,before_grade.native_score,before_grade.native_pass,
                            before_grade.preferences,technical_missing=before_grade.technical_missing,
                            scoring_pending=before_grade.scoring_pending)
                        self._event('pre_repair_draft_queued_for_scoring' if before_grade.scoring_pending else 'pre_repair_draft_graded',
                            start,unit_id=unit_id,delivery_index=delivery_index,
                            artifact_hash=draft.artifact_hash,**asdict(before_grade))
                    trace.add_delivery(Delivery(answer.artifact_hash, grade.native_score, grade.native_pass,
                        grade.preferences, answer.internal_repairs, answer.before_repair_artifact_hash, grade.technical_missing,
                        scoring_pending=grade.scoring_pending),
                        before_repair=before)
                    self._event('delivery_queued_for_scoring' if grade.scoring_pending else 'delivery_graded',
                                start, unit_id=unit_id, delivery_index=delivery_index, **asdict(grade))
                    if original_only:
                        # Private grades never determine another solver turn.
                        # A question, refusal or empty answer is this unit's
                        # actual delivery; only the next original input follows.
                        history.append({'role': 'assistant', 'text': answer.text})
                        status = 'not_requested'
                        break
                    # Current answer has its own field; the preceding history
                    # includes earlier units of this task but no private scores.
                    operations_json = (self.feedback_operations(unit,answer,start=dict(start),event_id=event_id)
                        if self.feedback_operations is not None and delivery_index < correction_limit else '{}')
                    payload = feedback_payload(unit, self.identity, delivery=answer.text,
                        visible_history=tuple(dict(t) for t in history), profile=self.profile,
                        training_reference=references.get(unit_id),visible_artifacts_json=answer.visible_artifacts_json,
                        public_materials_json=materials_json,visible_operations_json=operations_json,
                        delivery_context={'session_id':start['session_id'],'event_id':event_id,
                            'artifact_hash':answer.artifact_hash,'execution_evidence_hash':answer.execution_evidence_hash})
                    payload.update(previous_output_summary=output_summary,
                        previous_feedback=dict(previous_feedback) if previous_feedback is not None else None,
                        unreviewed_outputs=[dict(row) for row in unreviewed_outputs])
                    history.append({'role': 'assistant', 'text': answer.text})
                    if delivery_index == correction_limit:
                        # No extra call just to summarize the final budget-limited
                        # answer: include it in the next ordinary feedback call.
                        unreviewed_outputs.append({'instruction': unit.instruction,
                            'user_request': user_text, 'delivery': answer.text})
                        break
                    decision = self._read_call('feedback', lambda c:self.feedback(payload,context=c),start,unit_id,event_id)
                    if decision is None:
                        unreviewed_outputs.append({'instruction': unit.instruction,
                            'user_request': user_text, 'delivery': answer.text})
                        status = 'error'
                        break
                    if not isinstance(decision, FeedbackDecision):
                        raise TypeError('feedback role must return a parsed decision')
                    if decision.summary:
                        output_summary = decision.summary
                        unreviewed_outputs = []
                    else:
                        # Legacy/local test doubles may have no summary. Never
                        # pretend their unsummarized answers were remembered.
                        unreviewed_outputs.append({'instruction': unit.instruction,
                            'user_request': user_text, 'delivery': answer.text})
                    previous_feedback = {'status': decision.status, 'message': decision.message}
                    status = decision.status
                    self._event('feedback_decided', start, unit_id=unit_id, delivery_index=delivery_index, **asdict(decision))
                    if status != 'correction':
                        break
                    trace.add_correction(decision.message)
                    user_text = decision.message
                    learnable_input = decision.learnable
                    learning_text = decision.learning_message
                    status = 'budget_exhausted'
                outcome = UnitOutcome(unit_id, trace if trace.deliveries else None, status, error)
                outcomes.append(outcome)
                self._event('unit_completed', start, unit_id=unit_id, feedback_status=status, error_code=error,
                            metrics=trace.metrics() if trace.deliveries and not trace.scoring_pending else None,
                            scoring_pending=trace.scoring_pending)
                if self.commit_unit is not None and not aborted and trace.deliveries and not error:
                    from .task_outcomes import outcome_to_dict
                    self.commit_unit({'start': dict(start), 'test_protocol': self.test_protocol,
                        'training_correction_limit': self.training_correction_limit,
                        'outcome': outcome_to_dict(TaskOutcome(task_id,start['session_id'],tuple(outcomes),False)),
                        'history': history, 'output_summary': output_summary,
                        'previous_feedback': previous_feedback, 'unreviewed_outputs': unreviewed_outputs})
        finally:
            # Close both even if one cleanup fails; freeze violations must
            # propagate and prevent admission of results by the supervisor.
            try:
                if solver is not None:
                    solver.close()
            finally:
                if memory is not None:
                    memory.close()
        self._event('task_completed', start, aborted=aborted)
        return TaskOutcome(task_id, start['session_id'], tuple(outcomes), aborted)
