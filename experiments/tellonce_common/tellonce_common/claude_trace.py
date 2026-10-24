"""Task-owned Claude event binding for the shared TRACE execution guard."""
import time
from collections import deque

from .claude_tool_hooks import ClaudeToolHooks
from .native_rule_guard import NativeRuleGuard
from .request_budget import request_budget
from .rule_appeal import AppealInputError


REVIEW_NAME = 'mcp__benchmark__tellonce_review'


class ClaudeTrace:
    def __init__(self, *, memory, native_session_id, workspace, observation_type, event_type,
                 capture_delivery, artifacts, record, reviewer):
        self.native_session_id = native_session_id
        self.deadline = None
        self.responses = set()
        self.stop_feedback = deque()
        self.stop_notifications = 0
        self.guard = NativeRuleGuard(memory=memory, native_session_id=native_session_id,
            workspace=workspace, observation_type=observation_type, event_type=event_type,
            capture_delivery=capture_delivery, capture_artifacts=artifacts.capture_artifacts, record=record,
            require_native_delivery_observation=True, native_response_tracking=True,
            appeal_review=reviewer.review if reviewer is not None else None,
            capture_appeal_state=artifacts.appeal_state if reviewer is not None else None)
        self.tools = ClaudeToolHooks(native_session_id=native_session_id, workspace=workspace,
                                    failed_tool=self._failed_tool)

    def _failed_tool(self, **value):
        if value['tool_name'] == REVIEW_NAME:
            value['tool_name'] = 'tellonce_review'
        self.guard.observe_failed_tool(**value)

    def begin_delivery(self, event_id, *, user_text, deadline):
        if self.stop_feedback:
            raise ValueError('previous native repair feedback was not observed')
        self.stop_notifications = 0
        self.deadline = deadline
        self.guard.begin_delivery(event_id, user_text=user_text)

    def observe_assistant(self, message):
        ident = message.get('id')
        if not isinstance(ident, str) or not ident:
            raise ValueError('TRACE requires an actual Claude response identity')
        if ident not in self.responses:
            self.responses.add(ident)
            self.guard.observe_model_response(ident, messages=(), native_session_id=self.native_session_id)
        for part in message.get('content', []):
            if part.get('type') == 'text':
                self.guard.observe_response_text(ident, part['text'], native_session_id=self.native_session_id)
            elif part.get('type') == 'tool_use':
                self.tools.observe_proposal(part, native_session_id=self.native_session_id)

    def dispatch(self, payload):
        remaining = self.deadline - time.monotonic() - 1 if self.deadline is not None else 0
        if remaining <= 0:
            raise TimeoutError('Claude TRACE has no remaining turn budget')
        with request_budget(min(240, remaining)):
            if payload.get('hook_event_name') == 'Stop':
                self.tools.assert_complete()
                result = self.guard.handle(payload)
                if result.get('decision') == 'block':
                    self.stop_feedback.append('Stop hook feedback:\n' + result['reason'])
                    self.stop_notifications += 1
                return result
            return self.tools.handle(payload, check=self._check)

    def observe_control_message(self, row):
        content = row.get('message', {}).get('content')
        if (row.get('isSynthetic') is not True or row.get('parent_tool_use_id') is not None or
                not self.stop_feedback or content != [{'type':'text','text':self.stop_feedback[0]}]):
            raise ValueError('native control message does not match actual TRACE feedback')
        self.stop_feedback.popleft()

    def observe_notification(self, row):
        if (not self.stop_notifications or row.get('key') != 'stop-hook-error' or
                row.get('text') != 'Stop hook error occurred · ctrl+o to see' or
                row.get('priority') != 'immediate'):
            raise ValueError('notification lacks an actual TRACE Stop block')
        self.stop_notifications -= 1

    def _check(self, payload):
        # Only the exact host-defined read-only review tool is exempt from
        # its disputed check. The ledger binds its real MCP name unchanged.
        if payload.get('tool_name') == REVIEW_NAME:
            payload = {**payload, 'tool_name': 'tellonce_review'}
        return self.guard.handle(payload)

    def review(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) != {'disputes'}:
            return {'status': 'invalid', 'reason': 'Supply one disputes array.'}
        try:
            with request_budget(max(.01, min(240, self.deadline - time.monotonic() - 1))):
                return self.guard.submit_appeal(arguments['disputes'])
        except AppealInputError as error:
            return error.response()
        except ValueError:
            return {'status': 'invalid', 'reason': 'Use current check references and captured counterevidence.'}
