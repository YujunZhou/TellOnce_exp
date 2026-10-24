"""Connect the reviewed tellonce APIs to one experiment task session.

The owning process loads one admitted tellonce source bundle and passes its
TaskMemory, TrainingMemory, publication loader and transcript capture function.
No installed plugin, operator memory, environment default or model is selected
here. The solver must never be given this object or its library paths.
"""
import json
import os
from pathlib import Path

from .interaction_loop import MemoryContext
from .contracts import learning_input
from .provenance import content_hash
from .request_budget import request_budget


class _ScopeSelectionCache:
    """Reuse validated identical selector batches within one user boundary.

    The complete prompt includes the pinned rules and actual scope. Checker
    calls are never cached. No failed request or unvalidated response is saved.
    """
    def __init__(self, invoke):
        self.invoke = invoke
        self._selected = {}

    def __call__(self, prompt):
        return self.invoke(prompt)

    def request_validated(self, prompt, *, role, validate, max_attempts=2, request_key=None):
        key = content_hash(prompt)
        if role == 'rule_select' and key in self._selected:
            return validate(self._selected[key]), 0
        keyed = {'request_key': request_key} if request_key is not None else {}
        value, attempts = self.invoke.request_validated(
            prompt, role=role, validate=validate, max_attempts=max_attempts, **keyed)
        if role == 'rule_select' and isinstance(value, list) and all(isinstance(x, str) for x in value):
            if len(self._selected) >= 256:
                self._selected.pop(next(iter(self._selected)))
            self._selected[key] = json.dumps(value, ensure_ascii=False)
        return value, attempts


class TellonceSession:
    def __init__(self, *, start: dict, runtime, learner, ready_publication,
                 transcript: Path, capture_reference, record, verify_frozen):
        if (runtime.namespace != start['namespace'] or runtime.task_id != start['task_id'] or
                runtime.training is not start['training']):
            raise ValueError('tellonce runtime belongs to another task, experiment or phase')
        if start['training']:
            if learner is None or learner.namespace != start['namespace'] or not callable(ready_publication):
                raise ValueError('training needs its matching experiment-owned learner and publication loader')
        elif learner is not None or ready_publication is not None or not callable(verify_frozen):
            raise ValueError('test memory cannot own a writer and needs an explicit freeze verifier')
        if (not isinstance(transcript, Path) or not transcript.is_absolute() or
                transcript != transcript.resolve() or transcript.exists()):
            raise ValueError('each task needs a fresh canonical host-private transcript path')
        if any(not callable(f) for f in (capture_reference, record)):
            raise ValueError('explicit transcript capture and memory audit callbacks required')
        if not start['training']:
            verify_frozen()
        self.start, self.runtime, self.learner = dict(start), runtime, learner
        self.ready_publication = ready_publication
        self.transcript, self.capture_reference = transcript, capture_reference
        self.record, self.verify_frozen = record, verify_frozen
        self.initial_digest = runtime.publication.digest
        self._event = None
        self._seen = set()
        self._closed = False
        self._context_ok = True
        transcript.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(transcript, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)

    def _open(self):
        if self._closed:
            raise ValueError('task session is already closed')

    def _append(self, role, text):
        line = json.dumps({'type': role, 'sessionId': self.start['session_id'],
                           'message': {'content': text}}, ensure_ascii=False) + '\n'
        try:
            with os.fdopen(os.open(self.transcript, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW), 'a') as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            # Future requests cannot mistake an incomplete conversation for a
            # complete context. Prompt-only learning can still proceed.
            self._context_ok = False
            self.record({'type': 'tellonce_context_unavailable', 'session_id': self.start['session_id'],
                         'error_code': 'task_transcript_write_failed'})

    def _validate_input(self, unit_payload, user_text, event_id):
        self._open()
        if (not isinstance(unit_payload, dict) or set(unit_payload) != {'task_id', 'unit_id', 'instruction', 'context'} or
                unit_payload['task_id'] != self.start['task_id'] or not isinstance(user_text, str) or not user_text.strip() or
                not isinstance(event_id, str) or not event_id):
            raise ValueError('memory boundary needs the matching public task and actual user prompt')
        return content_hash([unit_payload, user_text, event_id])

    def user_boundary(self, *, unit_payload, user_text, event_id):
        signature = self._validate_input(unit_payload, user_text, event_id)
        if event_id in self._seen or (self._event is not None and not self._event['answered']):
            raise ValueError('duplicate or unfinished task user boundary')
        self._seen.add(event_id)
        reference = self.capture_reference({'session_id': self.start['session_id'],
            'transcript_path': str(self.transcript)}) if self._context_ok else None
        self._event = {'signature': signature, 'reference': reference, 'admitted': False, 'answered': False}
        public_task = json.dumps(unit_payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self.runtime.public_task = public_task
        self._append('user', user_text)
        ready = self.ready_publication() if self.start['training'] else None
        if isinstance(self.runtime.invoke, _ScopeSelectionCache):
            self.runtime.invoke = self.runtime.invoke.invoke
        with request_budget(240):
            boundary = self.runtime.user_boundary(user_text, ready_publication=ready,
                                                  session_id=self.start['session_id'],event_id=event_id)
        # Install after binding the actual user-event request owner. The next
        # user boundary discards this cache, even if the words are identical.
        if callable(getattr(type(self.runtime.invoke), 'request_validated', None)):
            self.runtime.invoke = _ScopeSelectionCache(self.runtime.invoke)
        result = boundary.result
        self.record({'type': 'tellonce_selection', 'session_id': self.start['session_id'], 'event_id': event_id,
                     'publication_digest': boundary.publication_digest, 'status': result.status,
                     'selected_ids': list(result.selected_ids), 'errors': list(result.errors),
                     'omitted_ids': list(result.omitted_ids), 'unexamined_ids': list(result.unexamined_ids)})
        return MemoryContext(result.injection, boundary.publication_digest, kind='selected_requirements')

    def enqueue_user(self, *, unit_payload, user_text, event_id, learning_text=None):
        signature = self._validate_input(unit_payload, user_text, event_id)
        learned = learning_input(user_text, learning_text)
        if not self.start['training'] or self.learner is None:
            raise ValueError('test feedback cannot enter the learning pipeline')
        if (self._event is None or signature != self._event['signature'] or
                self._event['admitted'] or self._event['answered']):
            raise ValueError('learning admission must match the current unconsumed user boundary')
        result = self.learner.enqueue_user(task_id=self.start['task_id'], session_id=self.start['session_id'],
            event_id=event_id, source_text=learned, public_task=self.runtime.public_task,
            context_reference=self._event['reference'] if self._context_ok else None)
        self._event['admitted'] = True
        self.record({'type': 'tellonce_enqueued', 'event_id': event_id, 'result': result})

    def observe_assistant(self, answer):
        self._open()
        if self._event is None or self._event['answered']:
            raise ValueError('assistant delivery has no pending user input')
        self._event['answered'] = True
        self._append('assistant', answer.text)

    def expand_scope(self, operation: str):
        self._open()
        if self._event is None or self._event['answered']:
            raise ValueError('scope expansion needs an active user turn')
        boundary = self.runtime.expand_scope(operation)
        self.record({'type': 'tellonce_scope_selection', 'session_id': self.start['session_id'],
                     'publication_digest': boundary.publication_digest, 'status': boundary.result.status,
                     'attempts': boundary.result.attempts,
                     'selected_ids': list(boundary.result.selected_ids), 'errors': list(boundary.result.errors),
                     'omitted_ids': list(boundary.result.omitted_ids),
                     'unexamined_ids': list(boundary.result.unexamined_ids)})
        return MemoryContext(boundary.result.injection, boundary.publication_digest, kind='selected_requirements')

    def check(self, observation):
        self._open()
        if self._event is None or self._event['answered']:
            raise ValueError('execution checks need an active user turn')
        # Scoped HTTP transports, journals and event writers support concurrent
        # independent requests. The core propagates the existing total deadline.
        # 2026-10-03: a delivery round of 5–8 checks ran in two waves at four workers (~6 s vs ~4 s per round).
        return self.runtime.check(observation, max_workers=12)

    def describe_rules(self, rule_digests):
        self._open()
        if self._event is None or self._event['answered']:
            raise ValueError('rule explanations need an active user turn')
        return self.runtime.describe_rules(rule_digests)

    def close(self):
        if self._closed:
            return
        if not self.start['training']:
            if self.runtime.publication.digest != self.initial_digest:
                raise ValueError('test task changed its frozen publication')
            self.verify_frozen()
        self._closed = True
