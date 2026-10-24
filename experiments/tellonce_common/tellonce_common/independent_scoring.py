"""Independent task/preference scoring failures and bounded component retries.

Use a task-owned instance. The interaction controller gets a complete Grade,
including unknown components, and therefore does not retry both scorers when
only one is unavailable. This cache holds private evaluation receipts only;
it is not agent memory and does not resume a task after process loss.
"""
from dataclasses import asdict
import threading
import time

from .interaction_loop import Grade, RecoverableCallError, SolverAnswer
from .interaction_metrics import PreferenceVerdict, preference_state
from .provenance import content_hash
from .request_budget import RequestDeadline, bounded_deadline, request_budget, supervised_retry_wait
from .text_request_journal import TextRequestDeferred


class IndependentScorer:
    def __init__(self, *, native, preferences, record, wait_retry, deadline_s=7200):
        if any(not callable(callback) for callback in (native, preferences, record, wait_retry)):
            raise ValueError('independent scorers, a private audit sink and supervised retry waits are required')
        self.native, self.preferences, self.record, self.wait_retry = native, preferences, record, wait_retry
        bounded_deadline(deadline_s)
        self.deadline_s = deadline_s
        self._done, self._active = {}, set()
        self._bindings,self._components={},{}
        self._lock = threading.Lock()

    def _component(self, kind, invoke, unit_id, answer, context):
        key=(context['call_id'],kind)
        with self._lock:
            if key in self._components:
                return self._components[key]
        def terminal(value):
            with self._lock:
                self._components[key]=value
            return value
        deadline = bounded_deadline(self.deadline_s)
        def wait(delay, child, reason, event):
            self.record({'type': 'score_component_wait', **event, 'reason': reason, 'retry_after_s': delay})
            try:
                supervised_retry_wait(delay, deadline=deadline, wait_retry=self.wait_retry,
                                      context=child, reason=reason)
            except RequestDeadline:
                self.record({'type': 'score_retry_deadline', **event})
                return False
            return True
        for attempt in (1,2):
            child = {**context, 'attempt':attempt}
            if kind == 'preferences':
                child.update(role='preference_'+context['role'],
                             call_id=content_hash([context['call_id'], 'personal_preferences']))
            event = dict(component=kind, unit_id=unit_id, artifact_hash=answer.artifact_hash, context=child)
            while True:
                if time.monotonic() >= deadline:
                    self.record({'type': 'score_retry_deadline', **event})
                    return terminal(None)
                self.record({'type':'score_component_started', **event})
                started = time.monotonic()
                try:
                    with request_budget(max(.000001, deadline-time.monotonic())):
                        value = invoke(unit_id, answer, context=child)
                except TextRequestDeferred as exc:
                    if not wait(exc.retry_after_s, child, exc.code, event):
                        return terminal(None)
                    continue
                except RequestDeadline:
                    self.record({'type': 'score_retry_deadline', **event})
                    return terminal(None)
                except RecoverableCallError as exc:
                    self.record({'type':'score_component_unavailable', **event, 'code':exc.code,
                        'retryable':exc.retryable, 'retry_after_s':exc.retry_after_s,
                        'wall_s':time.monotonic()-started})
                    if attempt == 2 or not exc.retryable:
                        return terminal(None)
                    if exc.retry_after_s and not wait(exc.retry_after_s, child, exc.code, event):
                        return terminal(None)
                    break
                else:
                    if kind == 'native':
                        if not isinstance(value, Grade) or value.preferences or value.scoring_pending:
                            raise ValueError('native scoring cannot contain personal-preference verdicts')
                    elif not isinstance(value, tuple) or any(not isinstance(v, PreferenceVerdict) for v in value):
                        raise ValueError('personal preference scoring requires attributed verdicts')
                    else:
                        preference_state(value)
                    self.record({'type':'score_component_completed', **event, 'wall_s':time.monotonic()-started})
                    return terminal(value)

    def __call__(self, unit_id, answer, *, context, previous_grade=None):
        keys = {'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (not isinstance(unit_id,str) or not unit_id or not isinstance(answer,SolverAnswer) or
                not isinstance(context,dict) or set(context) != keys or context.get('attempt') != 1 or
                type(context['attempt']) is not int or context['role'] not in {'score','score_before_repair'} or
                context['phase'] not in {'training','test'} or
                any(not isinstance(context[key],str) or not 1 <= len(context[key]) <= 480 for key in keys-{'attempt'})):
            raise ValueError('independent scoring must own the component retry budget and actual delivery context')
        if previous_grade is not None and (not isinstance(previous_grade, Grade) or previous_grade.scoring_pending):
            raise ValueError('preference-only repair requires an existing completed native grade')
        context = dict(context)
        key = context['call_id']
        binding = content_hash({'unit_id':unit_id, 'answer':asdict(answer), 'context':context,
                               'previous_grade':asdict(previous_grade) if previous_grade is not None else None})
        with self._lock:
            if key in self._bindings and self._bindings[key]!=binding:
                raise ValueError('a scoring request cannot change its delivery while waiting')
            self._bindings[key]=binding
            if key in self._done:
                prior, result = self._done[key]
                if prior != binding:
                    raise ValueError('a scoring receipt cannot be reused for a different delivery')
                return result
            if key in self._active:
                raise ValueError('the same delivery is already being scored')
            self._active.add(key)
        try:
            native = (Grade(previous_grade.native_score, previous_grade.native_pass, (), previous_grade.technical_missing)
                      if previous_grade is not None else self._component('native', self.native, unit_id, answer, context))
            preferences = self._component('preferences', self.preferences, unit_id, answer, context)
            if native is None:
                native = Grade(None,None,(),True)
            if preferences is None:
                preferences = (PreferenceVerdict('__preference_evaluation_unavailable__','unknown'),)
            result = Grade(native.native_score, native.native_pass, preferences, native.technical_missing)
            with self._lock:
                self._done[key] = binding, result
            return result
        finally:
            with self._lock:
                self._active.remove(key)
