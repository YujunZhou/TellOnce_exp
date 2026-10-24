"""Explicit Tellonce call scopes with one durable retry owner.

The host supplies the admitted Tellonce error class and its supervised wait
callback. This module does not import an installed plugin or personal memory.
"""
import re
import time
from types import MappingProxyType

from .durable_text_transport import DurableTextTransport
from .provenance import content_hash
from .text_request_journal import TextAttemptError, TextRequestDeferred
from .request_budget import bounded_deadline,request_budget


USER_ROLES=frozenset({'user_signal','correction_interpretation','lifecycle_select','lifecycle_plan','context_lookup','rule_select','rule_check'})
PUBLICATION_ROLES=frozenset({'compile_support','compile_review','compile_case'})
TEST_ROLES=frozenset({'rule_select','rule_check'})


class ManagedModelRequests:
    def __init__(self,transport,*,request_error,wait_retry,deadline_s=7200,_context=None,
                 check_transport=None):
        if (not isinstance(transport,DurableTextTransport) or not isinstance(request_error,type) or
                not issubclass(request_error,RuntimeError) or not getattr(request_error,'retry_exhausted',False) or
                not callable(wait_retry) or type(deadline_s) not in {int,float} or not 1<=deadline_s<=86400):
            raise ValueError('admitted durable transport, Tellonce failure class and supervised finite waits required')
        if check_transport is not None and (
                not isinstance(check_transport, DurableTextTransport) or
                check_transport.identity != transport.identity or
                check_transport.journal.binding != transport.journal.binding):
            raise ValueError('execution checker must belong to the same admitted experiment')
        self.transport,self.request_error,self.wait_retry=transport,request_error,wait_retry
        self.check_transport = check_transport
        self.identity,self.deadline_s=transport.identity,deadline_s
        self.context=MappingProxyType(dict(_context)) if _context is not None else None

    def __call__(self,prompt):
        raise ValueError('Tellonce requests require an explicit context and model role')

    def for_context(self,context):
        fields={'namespace','phase','scope','task_id','session_id','event_id'}
        if (not isinstance(context,dict) or set(context)!=fields or
                any(not isinstance(v,str) or not v.strip() or len(v)>480 for v in context.values()) or
                context['namespace']!=self.identity.namespace or context['phase'] not in {'training','test'} or
                context['scope'] not in {'user_event','publication'} or
                (context['scope']=='publication' and (context['phase']!='training' or
                 context['task_id']!='__library__' or context['session_id']!='__publication__'))):
            raise ValueError('Tellonce context must belong to its actual event or training publication')
        return type(self)(self.transport,request_error=self.request_error,wait_retry=self.wait_retry,
                          deadline_s=self.deadline_s,_context=context,check_transport=self.check_transport)

    def request_validated_with_repair(self,prompt,*,role,validate,repair,max_attempts=2,request_key=None):
        with request_budget(self.deadline_s):
            return self._request_validated_with_repair(prompt,role=role,validate=validate,
                repair=repair,max_attempts=max_attempts,request_key=request_key)

    def _request_validated_with_repair(self,prompt,*,role,validate,repair,max_attempts,request_key):
        """One learning decision, with at most two physical text attempts.

        Raw replies are durable evidence, not accepted learning plans. Validation
        stays outside those receipts so a rejected reply survives a restart and
        can be included in the one remaining repair attempt. Distinct immutable
        stage requests share an explicit parent identity and total attempt cap.
        """
        if (self.context is None or self.context['phase']!='training' or
                self.context['scope']!='user_event' or role!='lifecycle_plan' or
                not callable(validate) or not callable(repair) or
                not isinstance(prompt,str) or not prompt.strip() or
                type(max_attempts) is not int or max_attempts not in {1,2} or
                (request_key is not None and (not isinstance(request_key,str) or
                 not request_key.strip() or len(request_key)>480))):
            raise ValueError('format repair is restricted to a bounded training lifecycle decision')
        parent=content_hash({'context':dict(self.context),'role':role,
                             'key':request_key if request_key is not None else content_hash(prompt),
                             'protocol':'lifecycle_format_repair_v1'})
        raw,used=self.request_validated(prompt,role=role,validate=lambda text:text,
            max_attempts=max_attempts,request_key=parent+':initial')
        try:
            return validate(raw),used
        except Exception as error:
            self.transport.record({'type':'lifecycle_validation_rejected','namespace':self.identity.namespace,
                'context':dict(self.context),'parent_call_id':parent,'stage':'initial',
                'attempts_used':used,'response_hash':content_hash(raw),
                'validation_error':str(error)[:2048]})
            if used>=max_attempts:
                raise self.request_error(role,used,['response_validation_failed']) from None
            repaired_prompt=repair(prompt,raw,error)
        # A transport retry in the initial stage already consumes the full cap.
        # The repair stage has exactly one attempt, including transport failure.
        self.transport.record({'type':'lifecycle_format_repair','namespace':self.identity.namespace,
            'context':dict(self.context),'parent_call_id':parent,
            'initial_key':parent+':initial','repair_key':parent+':repair',
            'initial_response_hash':content_hash(raw),'repair_prompt_hash':content_hash(repaired_prompt),
            'remaining_attempts':max_attempts-used})
        try:
            repaired,repair_used=self.request_validated(repaired_prompt,role=role,validate=lambda text:text,
                max_attempts=1,request_key=parent+':repair')
        except self.request_error as error:
            raise self.request_error(role,used+error.attempts,
                ['response_validation_failed',*error.error_types],retry_after_s=error.retry_after_s) from None
        try:
            return validate(repaired),used+repair_used
        except Exception as error:
            self.transport.record({'type':'lifecycle_validation_rejected','namespace':self.identity.namespace,
                'context':dict(self.context),'parent_call_id':parent,'stage':'repair',
                'attempts_used':used+repair_used,'response_hash':content_hash(repaired),
                'validation_error':str(error)[:2048]})
            raise self.request_error(role,used+repair_used,
                ['response_validation_failed','response_validation_failed']) from None

    def request_validated(self,prompt,*,role,validate,max_attempts=2,request_key=None):
        if (self.context is None or not isinstance(prompt,str) or not prompt.strip() or not callable(validate) or
                not isinstance(role,str) or not re.fullmatch('[a-z][a-z0-9_]{0,63}',role) or
                type(max_attempts) is not int or max_attempts not in {1,2} or
                (request_key is not None and (not isinstance(request_key,str) or not request_key.strip() or len(request_key)>480))):
            raise ValueError('scoped Tellonce request, bounded attempts and admitted parser required')
        allowed=PUBLICATION_ROLES if self.context['scope']=='publication' else USER_ROLES
        if role not in allowed or (self.context['phase']=='test' and role not in TEST_ROLES):
            raise ValueError('this Tellonce model role is not allowed in the current phase and scope')
        if role=='rule_select' and request_key is None:
            raise ValueError('runtime selection needs its actual occurrence and batch identity')
        # Learning, compilation and checks already bind a source event, rule or
        # captured observation in their complete input. Repeated runtime scope
        # selection uses an explicit occurrence key, not a text-query cache.
        key=request_key if request_key is not None else content_hash(prompt)
        call_id=content_hash({'context':dict(self.context),'role':role,'key':key})
        context={k:self.context[k] for k in ('phase','task_id','session_id','event_id')}
        context.update(role=role,call_id=call_id)
        errors=[]
        deadline=bounded_deadline(self.deadline_s)
        transport = (self.check_transport if self.check_transport is not None and
                     role in {'rule_check', 'compile_case'} else self.transport)

        def wait(delay,reason,attempt):
            remaining=deadline-time.monotonic()
            if remaining<=0 or delay>remaining:
                raise self.request_error(role,attempt,errors+['request_wait_deadline'],retry_after_s=delay) from None
            # The host keeps heartbeat/cancellation alive during each short
            # wait. The journal rechecks not-before before reserving a retry.
            self.wait_retry(min(delay,30),context={**context,'attempt':attempt},reason=reason)

        for attempt in range(1,max_attempts+1):
            while True:
                if time.monotonic()>=deadline:
                    raise self.request_error(role,attempt-1,errors+['request_deadline']) from None
                try:
                    with request_budget(max(.000001,deadline-time.monotonic())):
                        reply=transport.call(prompt,context={**context,'attempt':attempt},validate=validate)
                except TextRequestDeferred as exc:
                    wait(exc.retry_after_s,exc.code,attempt)
                    continue
                except TextAttemptError as exc:
                    errors.append(exc.code)
                    if not exc.retryable or attempt==max_attempts:
                        raise self.request_error(role,attempt,errors,retry_after_s=exc.retry_after_s) from None
                    if exc.retry_after_s:
                        wait(exc.retry_after_s,exc.code,attempt)
                    break
                else:
                    if reply.attempt>max_attempts:
                        raise ValueError('saved receipt exceeds this admitted request attempt budget')
                    return validate(reply.text),reply.attempt
