"""Recoverable logical text attempts over admitted single-attempt transports.

Each physical child is explicitly linked to its logical request and attempt.
The journal, rather than a fresh provider object's RAM, enforces the total cap.
Validators must be deterministic format parsers, never answer-quality graders.
"""
from dataclasses import dataclass
import json
import re

from .http_text_transport import HTTPTextFailure, conversation_config
from .spending_budget import SpendingLimitReached
from .provenance import canonical_json_bytes, content_hash
from .role_inputs import RunIdentity
from .text_request_journal import TextAttemptError, TextRequestJournal
from .request_budget import RequestDeadline
from .native_availability import NativeUnavailable
from .text_input_budget import MAX_TEXT_INPUT_BYTES, TextInputBudgetExceeded, check_text_input
from .text_capacity import TextCapacity, TextTokenCapacityExceeded


@dataclass(frozen=True)
class DurableTextResult:
    text: str
    response_id: str | None
    thread_id: str | None
    usage_json: str
    original_usage_json: str
    replayed: bool
    attempt: int
    child_call_id: str


class DurableTextTransport:
    def __init__(self,transport,*,journal: TextRequestJournal,authorize,record,capacity=None):
        identity=getattr(transport,'identity',None)
        if (not isinstance(identity,RunIdentity) or not isinstance(journal,TextRequestJournal) or
                identity.namespace!=journal.namespace or not callable(authorize) or not callable(record)):
            raise ValueError('durable text owner requires matching namespace and source admission')
        self.transport,self.journal=transport,journal
        self.identity,self.authorize,self.record=identity,authorize,record
        if capacity is not None and (not isinstance(capacity, TextCapacity) or
                capacity.model != getattr(getattr(transport, 'config', None), 'model', None) or
                capacity.output_reserve < getattr(getattr(transport, 'config', None), 'max_tokens', 0) or
                capacity.context_window > getattr(transport, 'context_window', capacity.context_window)):
            raise ValueError('capacity measurement must belong to this exact text model')
        self.capacity = capacity

    def call(self,prompt,*,context,base_instructions=None,validate=None):
        return self._call(prompt, context=context, base_instructions=base_instructions, validate=validate)

    def call_messages(self, messages, *, context, generation, validate=None):
        if (not isinstance(context, dict) or context.get('role') != 'feedback' or
                not callable(getattr(self.transport, 'call_messages', None))):
            raise ValueError('this provider does not support an admitted feedback conversation')
        config = conversation_config(getattr(self.transport, 'config', None), generation)
        config.payload(messages)
        messages = [dict(message) for message in messages]
        # JSON role framing is included in the capacity guard; the actual HTTP
        # call keeps the original message boundaries rather than this rendering.
        rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return self._call(rendered, context=context, base_instructions=None, validate=validate,
                          conversation=messages, generation=dict(generation))

    def _call(self, prompt, *, context, base_instructions, validate, conversation=None, generation=None):
        self._check_request(prompt,context,base_instructions,validate)
        context=dict(context)
        if self.capacity is not None:
            try:
                self.capacity(prompt, base_instructions=base_instructions, context=context)
            except TextTokenCapacityExceeded:
                raise TextAttemptError('text_token_capacity', retryable=False) from None
        # Admission applies to replay as well as a new physical call. The
        # supplied gate verifies the journal's full source/configuration binding.
        self.authorize('call',{'namespace':self.identity.namespace,'binding':self.journal.binding,**context})
        request_value={'binding':self.journal.binding,'prompt':prompt,
            'base_instructions':base_instructions,'context':{k:v for k,v in context.items() if k!='attempt'}}
        if conversation is not None:
            request_value['conversation_generation'] = generation
        request_hash=content_hash(request_value)
        def invoke():
            child_id=content_hash({'namespace':self.identity.namespace,'call_id':context['call_id'],
                                  'attempt':context['attempt'],'request_hash':request_hash})
            child_context={**context,'call_id':child_id,'attempt':1}
            self.record({'type':'text_attempt_link','namespace':self.identity.namespace,'context':context,
                'child_context':child_context,'request_hash':request_hash})
            try:
                if conversation is not None:
                    reply=self.transport.call_messages(conversation, context=child_context,
                        generation=generation, validate=validate)
                else:
                    reply=self.transport.call(prompt,context=child_context,base_instructions=base_instructions,validate=validate)
            except NativeUnavailable:
                raise
            except SpendingLimitReached:
                raise TextAttemptError('experiment_spending_limit',retryable=False) from None
            except HTTPTextFailure as exc:
                code = {402: 'provider_credit_exhausted', 401: 'provider_authorization_failed',
                        403: 'provider_authorization_failed', 429: 'provider_rate_limited',
                        500: 'provider_service_unavailable', 502: 'provider_service_unavailable',
                        503: 'provider_service_unavailable', 504: 'provider_service_unavailable'}.get(exc.metrics.get('http_status'), 'http_text_failed')
                code=exc.metrics.get('failure_code',code)
                if (self.identity.method=='trace' and context['phase']=='training' and
                        context['role'] in {'user_signal','correction_interpretation','lifecycle_select',
                                            'lifecycle_plan','context_lookup'} and
                        exc.metrics.get('transport_status')=='timeout' and exc.metrics.get('http_status') is None):
                    code='provider_timeout'
                raise TextAttemptError(code,retryable=exc.retryable,retry_after_s=exc.retry_after_s) from None
            except RequestDeadline:
                raise TextAttemptError('request_deadline',retryable=False) from None
            except TextInputBudgetExceeded:
                raise TextAttemptError('text_input_budget',retryable=False) from None
            except Exception:
                raise TextAttemptError('provider_text_failed') from None
            value={'text':reply.text,'response_id':getattr(reply,'response_id',None),
                   'thread_id':getattr(reply,'thread_id',None),'usage_json':reply.usage_json,'child_call_id':child_id}
            self._validate(value,validate)
            return value
        receipt=self.journal.perform(call_id=context['call_id'],request_hash=request_hash,
                                     attempt=context['attempt'],invoke=invoke)
        # Corrupt or no-longer-valid receipts are rejected without sampling a
        # fresh answer. Changed parsers also require a new admitted source binding.
        self._validate(receipt.value,validate)
        value=receipt.value
        self.record({'type':'text_receipt_delivered','namespace':self.identity.namespace,'context':context,
            'receipt_attempt':receipt.attempt,'child_call_id':value['child_call_id'],'replayed':receipt.replayed,
            'request_hash':request_hash})
        return DurableTextResult(value['text'],value['response_id'],value['thread_id'],
            'null' if receipt.replayed else value['usage_json'],value['usage_json'],
            receipt.replayed,receipt.attempt,value['child_call_id'])

    def _check_request(self,prompt,context,base_instructions,validate):
        keys={'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (not isinstance(context,dict) or set(context)!=keys or
                context.get('phase') not in {'preparation','training','test'} or
                any(not isinstance(context[k],str) or not context[k].strip() or len(context[k])>480 for k in keys-{'attempt'}) or
                not re.fullmatch('[a-z][a-z0-9_]{0,63}',context['role']) or
                type(context['attempt']) is not int or context['attempt'] not in {1,2} or
                not isinstance(prompt,str) or not prompt.strip() or
                (base_instructions is not None and not isinstance(base_instructions,str)) or
                (validate is not None and not callable(validate))):
            raise ValueError('complete bounded durable text request required')
        try:
            check_text_input(prompt, base_instructions or '',
                limit=getattr(getattr(self.transport, 'config', None), 'input_max_bytes', MAX_TEXT_INPUT_BYTES))
        except TextInputBudgetExceeded:
            raise TextAttemptError('text_input_budget', retryable=False) from None

    @staticmethod
    def _validate(value,validate):
        if (not isinstance(value,dict) or set(value)!={'text','response_id','thread_id','usage_json','child_call_id'} or
                not isinstance(value['text'],str) or not value['text'].strip() or
                not isinstance(value['child_call_id'],str) or not re.fullmatch('[0-9a-f]{64}',value['child_call_id']) or
                not isinstance(value['usage_json'],str)):
            raise ValueError('invalid durable text receipt')
        identifiers=[value['response_id'],value['thread_id']]
        if (sum(x is not None for x in identifiers)!=1 or
                any(x is not None and (not isinstance(x,str) or not 1<=len(x)<=512) for x in identifiers)):
            raise ValueError('one captured response or thread identity is required')
        usage=json.loads(value['usage_json'])
        if usage is not None and not isinstance(usage,dict):
            raise ValueError('captured usage must be an object or unknown')
        canonical_json_bytes(usage)
        if validate is not None:
            validate(value['text'])
