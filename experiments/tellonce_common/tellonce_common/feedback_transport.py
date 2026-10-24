"""One feedback attempt; the interaction controller owns the retry budget."""
from .feedback_agent import feedback_prompt, parse_rolling_feedback, USER_IDENTITY_INSTRUCTIONS
from .http_text_transport import HTTPTextFailure
from .spending_budget import SpendingLimitReached
from .interaction_loop import RecoverableCallError
from .role_inputs import RunIdentity
from .text_request_journal import TextAttemptError, TextRequestDeferred
from .request_budget import RequestDeadline
from .text_input_budget import TextInputBudgetExceeded
from .feedback_evidence import EVIDENCE_INSTRUCTIONS, RELATIONAL_CONSTRAINT_INSTRUCTIONS
from .feedback_context import VERSION
from .provenance import content_hash
from .compact_profiles import PROFILE_VERSION


class FeedbackServiceUnavailable(RuntimeError):
    """Preserve the unfinished unit when service recovery needs a new phase."""
    def __init__(self, code, retry_after_s=None):
        self.code = code
        self.retry_after_s = retry_after_s
        super().__init__(code)


class FeedbackAgent:
    def __init__(self, transport, *, review=None, may_protocol='legacy-v4'):
        if not isinstance(getattr(transport,'identity',None),RunIdentity):
            raise ValueError('feedback transport must belong to an explicit experiment')
        self.transport=transport
        self.identity=transport.identity
        if review is not None and getattr(review, 'identity', None) != self.identity:
            raise ValueError('feedback review must belong to this experiment owner')
        self.review = review
        if may_protocol not in {'legacy-v4', 'reviewed-v5'}:
            raise ValueError('unknown May feedback protocol')
        # Reviewed v5 is explicitly admitted by a new run, never silently
        # substituted into an existing durable feedback/review binding.
        self.may_protocol = may_protocol

    def _context(self, payload, context):
        binding = [context['call_id'], VERSION]
        if ((payload.get('user_profile') or {}).get('kind') == PROFILE_VERSION
                and self.may_protocol != 'legacy-v4'):
            binding.append(self.may_protocol)
        return {**context, 'call_id': content_hash(binding)}

    def __call__(self, payload, *, context):
        decision = self._generate(payload, context=context)
        if self.review is None:
            return decision
        # Deliberately outside model failure conversion: awaiting the human
        # neither spends another model attempt nor delivers an unreviewed correction.
        review_context = self._context(payload, context)
        return self.review.review(payload, decision, context=review_context)

    def _generate(self, payload, *, context):
        if not isinstance(context,dict) or context.get('role')!='feedback':
            raise ValueError('feedback requires an explicit controller feedback context')
        # A resumed old-version unit may already have a journal entry. Keep it
        # intact and give the reviewed new feedback protocol its own stable ID.
        context = self._context(payload, context)
        try:
            if (payload.get('user_profile') or {}).get('kind') == PROFILE_VERSION:
                if self.may_protocol == 'reviewed-v5':
                    from .may_feedback_reviewed import run_feedback
                else:
                    from .may_feedback import run_feedback
                from .feedback_context import compact_feedback_view
                return run_feedback(compact_feedback_view(payload), self.transport, context)
            rendered=feedback_prompt(payload)
            instructions,prompt=rendered.split('\nINPUT_JSON\n',1)
            instructions += '\n' + EVIDENCE_INSTRUCTIONS
            instructions = USER_IDENTITY_INSTRUCTIONS + '\n' + instructions
            # Set the actual text-role instructions on native and HTTP providers.
            instructions+='\nINPUT_JSON is the JSON object in the user message.'
            instructions+=RELATIONAL_CONSTRAINT_INSTRUCTIONS
            reply=self.transport.call(prompt,context=context,base_instructions=instructions,validate=parse_rolling_feedback)
        except TextRequestDeferred:
            # Waiting/ownership is not a failed model attempt. A supervisor
            # must preserve this distinction rather than spending another try.
            raise
        except SpendingLimitReached:
            raise FeedbackServiceUnavailable('experiment_spending_limit') from None
        except RequestDeadline:
            raise RecoverableCallError('feedback_deadline',retryable=False) from None
        except TextInputBudgetExceeded:
            raise RecoverableCallError('feedback_input_budget',retryable=False) from None
        except (HTTPTextFailure,TextAttemptError) as exc:
            code = (exc.code if isinstance(exc, TextAttemptError) else
                    {402: 'provider_credit_exhausted', 401: 'provider_authorization_failed',
                     403: 'provider_authorization_failed', 429: 'provider_rate_limited',
                     500: 'provider_service_unavailable', 502: 'provider_service_unavailable',
                     503: 'provider_service_unavailable', 504: 'provider_service_unavailable'}.get(exc.metrics.get('http_status')))
            if code in {'provider_credit_exhausted', 'provider_authorization_failed', 'experiment_spending_limit',
                        'provider_rate_limited', 'provider_service_unavailable'}:
                if (code in {'provider_rate_limited', 'provider_service_unavailable'} and
                        exc.retryable and context['attempt'] == 1 and
                        (exc.retry_after_s is None or exc.retry_after_s <= 5)):
                    # The interaction owner already has one unused read-only
                    # retry. Use it in this task, independently of other groups.
                    # A second rejection still preserves the unfinished unit.
                    raise RecoverableCallError(code, retry_after_s=5) from None
                raise FeedbackServiceUnavailable(code, exc.retry_after_s) from None
            raise RecoverableCallError('feedback_http_failed',retryable=exc.retryable,
                                       retry_after_s=exc.retry_after_s) from None
        except RuntimeError:
            raise RecoverableCallError('feedback_native_failed') from None
        return parse_rolling_feedback(reply.text)
