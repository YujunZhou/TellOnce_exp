"""One private preference-judge attempt, with retries owned by IndependentScorer."""
import json

from .http_text_transport import HTTPTextFailure
from .interaction_loop import RecoverableCallError
from .native_availability import NativeUnavailable
from .interaction_metrics import PreferenceVerdict
from .preference_judge import PreferenceInput
from .preference_spans import (index_preferences, parse_indexed_preferences,
                               parse_may_delivery_preferences, MAY_DELIVERY_PREFERENCE_INSTRUCTIONS,
                               INDEXED_PREFERENCE_INSTRUCTIONS, DELIVERY_PREFERENCE_INSTRUCTIONS)
from .request_budget import RequestDeadline
from .role_inputs import RunIdentity
from .text_input_budget import TextInputBudgetExceeded
from .text_request_journal import TextAttemptError, TextRequestDeferred
from .feedback_transport import FeedbackServiceUnavailable
from .provenance import content_hash


class PreferenceJudge:
    def __init__(self, transport, *, may_protocol='legacy-v1', workspace_roots=()):
        if not isinstance(getattr(transport,'identity',None),RunIdentity):
            raise ValueError('preference judge requires an experiment-owned text transport')
        self.transport,self.identity=transport,transport.identity
        if may_protocol not in {'legacy-v1', 'coverage-v2'}:
            raise ValueError('explicit supported May preference protocol required')
        self.may_protocol = may_protocol
        if workspace_roots:
            from .preference_path_blinding import blind_workspace_paths
            blind_workspace_paths('{}', workspace_roots)
        elif workspace_roots != ():
            raise ValueError('workspace roots must be an explicit immutable tuple')
        self.workspace_roots = workspace_roots

    def __call__(self, value: PreferenceInput, *, context: dict):
        if (not isinstance(value,PreferenceInput) or value.namespace!=self.identity.namespace or
                context!=json.loads(value.context_json)):
            raise ValueError('private preference input and request context differ')
        may_delivery = (value.delivery_only and
                        json.loads(value.payload_json)['user_profile'].get('kind', '').startswith('may-'))
        if self.may_protocol == 'coverage-v2' and not may_delivery:
            raise ValueError('coverage-v2 requires May delivery input')
        if value.delivery_only:
            public=json.loads(value.payload_json)
            if not public['delivery'].strip() and not public.get('visible_artifacts',{}).get('files'):
                # No result exists to judge. Avoid a model call that can only
                # invent evidence; native task correctness remains separate.
                return (PreferenceVerdict('__no_visible_delivery__','unknown'),)
        parser = parse_may_delivery_preferences if may_delivery else parse_indexed_preferences
        instructions=(MAY_DELIVERY_PREFERENCE_INSTRUCTIONS if may_delivery else
                      DELIVERY_PREFERENCE_INSTRUCTIONS if value.delivery_only else INDEXED_PREFERENCE_INSTRUCTIONS)
        request_context = context
        if self.may_protocol == 'coverage-v2':
            from .may_preference_coverage import (
                PROTOCOL, INSTRUCTIONS, coverage_preferences, parse_coverage_preferences)
            indexed = coverage_preferences(value)
            parser = parse_coverage_preferences
            instructions = INSTRUCTIONS
            request_context = {**context, 'call_id': content_hash([context['call_id'], PROTOCOL])}
        else:
            indexed = index_preferences(value)
        prompt = indexed.payload_json
        if self.workspace_roots:
            from .preference_path_blinding import blind_workspace_paths
            projection = blind_workspace_paths(prompt, self.workspace_roots)
            prompt = projection.payload_json
            request_context = {**request_context, 'call_id': content_hash([
                request_context['call_id'], projection.audit])}
            record = getattr(self.transport, 'record', None)
            if callable(record):
                record({'type': 'private_preference_path_projection', 'context': request_context,
                        'audit': projection.audit})
        # Display paths may be aliased, but evidence IDs resolve through the
        # original full texts. Never reinterpret masked text as an observation.
        parse=lambda text:parser(text,indexed)
        try:
            reply=self.transport.call(prompt,base_instructions=instructions,
                                      context=request_context,validate=parse)
        except (TextRequestDeferred, NativeUnavailable):
            raise
        except RequestDeadline:
            raise RecoverableCallError('preference_deadline',retryable=False) from None
        except TextInputBudgetExceeded:
            raise RecoverableCallError('preference_input_budget',retryable=False) from None
        except (HTTPTextFailure,TextAttemptError) as exc:
            credit_outage = (exc.code == 'provider_credit_exhausted' if isinstance(exc, TextAttemptError)
                             else exc.metrics.get('http_status') == 402)
            if credit_outage:
                # Missing provider credit is not an unknown preference verdict.
                # Leave this scoring item open for a later funded phase owner.
                raise FeedbackServiceUnavailable('provider_credit_exhausted') from None
            raise RecoverableCallError('preference_transport_failed',retryable=exc.retryable,
                                       retry_after_s=exc.retry_after_s) from None
        except RuntimeError:
            raise RecoverableCallError('preference_native_failed') from None
        return parse(reply.text)
