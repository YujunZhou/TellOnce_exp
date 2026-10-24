"""Bind Claude tool hooks to actual proposals and results in one native task.

This is an adapter component, not TRACE admission. The owner must forward all
native tool proposals/results and call assert_complete before accepting a
delivery. Hook callbacks may run checks; native stream callbacks must not.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import threading

from .provenance import content_hash


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class ClaudeToolHooks:
    def __init__(self, *, native_session_id, workspace, failed_tool, wait_s=3):
        if (not isinstance(native_session_id, str) or not native_session_id or
                not isinstance(workspace, Path) or not workspace.is_absolute() or workspace != workspace.resolve() or
                not callable(failed_tool) or type(wait_s) not in (int, float) or not 0 < wait_s <= 3):
            raise ValueError('explicit native task identity and failure callback required')
        self.native_session_id, self.workspace = native_session_id, workspace
        self.failed_tool, self.wait_s = failed_tool, wait_s
        self._condition = threading.Condition()
        self._pending, self._seen, self._failed = {}, set(), False

    @contextmanager
    def _state(self):
        with self._condition:
            if self._failed:
                raise ValueError('Claude hook observation previously failed')
            try:
                yield
            except Exception:
                self._failed = True
                raise
            finally:
                self._condition.notify_all()

    def observe_proposal(self, part, *, native_session_id):
        with self._state():
            if (native_session_id != self.native_session_id or not isinstance(part, dict) or
                    part.get('type') != 'tool_use' or
                    any(not isinstance(part.get(k), str) or not part[k] for k in ('id', 'name')) or
                    not isinstance(part.get('input'), dict)):
                raise ValueError('actual native tool proposal required')
            ident = part['id']
            if ident in self._seen or len(self._seen) >= 4096:
                raise ValueError('duplicate native tool identity or task tool budget exhausted')
            action = _copy({'tool': part['name'], 'arguments': part['input']})
            if len(json.dumps(action, ensure_ascii=False).encode()) > 65536:
                raise ValueError('native tool operation exceeds identity budget')
            self._seen.add(ident)
            self._pending[ident] = {'action': action, 'pre': None, 'post': None}

    def handle(self, payload, *, check):
        """Bind before checking; never hold the stream lock during model work."""
        with self._state():
            if (not isinstance(payload, dict) or payload.get('session_id') != self.native_session_id or
                    payload.get('cwd') != str(self.workspace) or not callable(check)):
                raise ValueError('foreign Claude tool hook')
            kind, ident = payload.get('hook_event_name'), payload.get('tool_use_id')
            if kind not in ('PreToolUse', 'PostToolUse') or not isinstance(ident, str) or not ident:
                raise ValueError('actual native tool hook identity required')
            # Native stdout and hook socket use different threads. Only the
            # hook side may wait for the already emitted assistant proposal.
            if not self._condition.wait_for(lambda: ident in self._seen or self._failed, self.wait_s):
                raise ValueError('tool hook has no observed native proposal')
            if self._failed or ident not in self._pending:
                raise ValueError('failed or already completed native tool hook')
            operation = self._pending[ident]
            actual = {'tool': payload.get('tool_name'), 'arguments': payload.get('tool_input')}
            if content_hash(actual) != content_hash(operation['action']):
                raise ValueError('tool hook changed native tool or arguments')
            if kind == 'PreToolUse':
                if operation['pre'] is not None:
                    raise ValueError('duplicate native pre hook')
                operation['pre'] = 'checking'
            else:
                if operation['pre'] != 'allowed' or operation['post'] is not None or 'tool_response' not in payload:
                    raise ValueError('post hook lacks a permitted pre hook or actual response')
                operation['post'] = 'checking'
                operation['post_hash'] = content_hash(payload['tool_response'])
            captured = _copy(payload)
        try:
            output = check(captured)
            with self._state():
                if not isinstance(output, dict):
                    raise ValueError('native checker must return a hook object')
                details = output.get('hookSpecificOutput', {})
                if (not isinstance(details, dict) or details.get('hookEventName', kind) != kind or
                        details.get('permissionDecision') not in (None, 'deny') or
                        'updatedInput' in details or output.get('continue') is False):
                    raise ValueError('checker cannot change native arguments or grant permissions')
                if kind == 'PreToolUse':
                    operation['pre'] = 'denied' if details.get('permissionDecision') == 'deny' else 'allowed'
                else:
                    if 'permissionDecision' in details:
                        raise ValueError('post hook cannot decide pre-execution permissions')
                    operation['post'] = 'checked'
            return output
        except Exception:
            with self._condition:
                self._failed = True
                self._condition.notify_all()
            raise

    def observe_result(self, part, *, native_session_id, native_result):
        """Close only actual results; queue unmatched errors to the guard."""
        failure = None
        with self._state():
            if (native_session_id != self.native_session_id or not isinstance(part, dict) or
                    part.get('type') != 'tool_result' or not isinstance(part.get('tool_use_id'), str) or
                    type(part.get('is_error', False)) is not bool or part['tool_use_id'] not in self._pending):
                raise ValueError('unknown or invalid native tool result')
            ident = part['tool_use_id']
            operation = self._pending[ident]
            if operation.get('closing'):
                raise ValueError('native result arrived after its hook was closed')
            # Native stdout and hook callbacks are independent. A tool result
            # can therefore arrive before the post hook has even entered its
            # checking state, or while either checker is still running. For a
            # successful native result, wait for an allowed pre hook and the
            # post hook's decision; failing immediately when post is still
            # None turns this valid ordering race into a failed native turn.
            # Error results intentionally need no post hook: they close the
            # operation as an observed native failure after the pre hook settles.
            def ready():
                if self._failed or operation['pre'] == 'checking':
                    return self._failed
                if part.get('is_error', False) or operation['pre'] != 'allowed':
                    return True
                return operation['post'] is not None and operation['post'] != 'checking'
            if not self._condition.wait_for(ready, self.wait_s):
                raise ValueError('native hook did not complete before its result')
            if self._failed:
                raise ValueError('Claude hook observation previously failed')
            pre, post = operation['pre'], operation['post']
            if part.get('is_error', False):
                if post is not None:
                    raise ValueError('native error contradicts successful post hook')
                if pre == 'allowed':
                    failure = {'native_session_id': self.native_session_id,
                               'tool_name': operation['action']['tool'],
                               'tool_input': _copy(operation['action']['arguments']), 'result': _copy(part)}
                # No pre: native argument validation may reject before hooks.
                # Denied pre: guard has no allowed pending action to close.
            elif (pre != 'allowed' or post != 'checked' or
                    content_hash(native_result) != operation['post_hash']):
                raise ValueError('native success lacks matching pre/post evidence')
            operation['closing'] = True
        try:
            if failure is not None:
                self.failed_tool(**failure)
            with self._state():
                del self._pending[ident]
        except Exception:
            with self._condition:
                self._failed = True
                self._condition.notify_all()
            raise

    def assert_complete(self):
        with self._state():
            # Stop arrives on a separate socket from native stdout. It can
            # reach this thread just before the stream finishes queuing an
            # already emitted tool result. Waiting here releases the lock;
            # the stream never waits for hook checks in the opposite direction.
            if (not self._condition.wait_for(lambda: not self._pending or self._failed, self.wait_s) or
                    self._failed):
                raise ValueError('native delivery has unfinished tool operations')
