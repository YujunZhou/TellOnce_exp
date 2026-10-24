"""Task-labelled, stateless HTTP text calls through a private one-shot worker."""
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid

from .http_text_contract import HTTPTextConfig, ENDPOINTS, SYNTHETIC_KEY, GEMMA_MODEL, clean_worker_metrics, endpoint
from .native_execution import run_native_command
from .provenance import content_hash, file_hash
from .request_budget import RequestBudget
from .spending_budget import SpendingBudget
from .validation_diagnostics import validation_failure


@dataclass(frozen=True)
class HTTPTextResult:
    text: str
    response_id: str
    usage_json: str


class HTTPTextFailure(RuntimeError):
    def __init__(self,code,*,metrics=None):
        super().__init__(code)
        self.metrics=metrics or {}
        self.retry_after_s=self.metrics.get('retry_after_s')
        status=self.metrics.get('http_status')
        self.retryable=status not in {400,401,402,403,404,405,413,422}
        if self.metrics.get('failure_code') in {'model_refusal','provider_credit_exhausted','provider_authorization_failed','provider_context_exceeded'}:
            self.retryable=False


def conversation_config(config, generation):
    """Only bounded generation settings vary; provider/model/route stay pinned."""
    if (not isinstance(config, HTTPTextConfig) or not isinstance(generation, dict) or
            set(generation) != {'max_tokens', 'temperature'} or
            type(generation['max_tokens']) is not int or not 1 <= generation['max_tokens'] <= config.max_tokens):
        raise ValueError('conversation generation must fit the pinned HTTP role budget')
    maximum=generation['max_tokens']
    if config.provider=='openai':
        maximum=max(8192,maximum)  # Includes reasoning; retain requested stage limits in the journal.
        if maximum>config.max_tokens:
            raise ValueError('official feedback requires its admitted reasoning output reserve')
    return replace(config, max_tokens=maximum, temperature=generation['temperature'])


class HTTPTextTransport:
    def __init__(self, *, identity, config, private_root: Path, authorize, record,
                 api_key_source=None, offline_endpoint=None, timeout_s=180,
                 selector_reasoning_effort=None):
        if (not isinstance(private_root,Path) or not private_root.is_absolute() or private_root!=private_root.resolve() or
                not 0<timeout_s<=900 or (offline_endpoint is None and not callable(api_key_source)) or
                (offline_endpoint is not None and api_key_source is not None)):
            raise ValueError('explicit private HTTP runtime and exactly one credential mode required')
        if (offline_endpoint is None and
                getattr(api_key_source,'provider',None)!=config.provider and
                (config.provider=='openrouter' or getattr(api_key_source,'provider',None) is not None)):
            raise ValueError('HTTP credential source belongs to another provider')
        self.endpoint=offline_endpoint or ENDPOINTS[config.provider]
        self.synthetic=offline_endpoint is not None
        endpoint(self.endpoint,synthetic=self.synthetic,provider=config.provider)
        authorize('prepare',{'namespace':identity.namespace,'model':config.model})
        self.identity,self.config,self.private_root=identity,config,private_root
        # Pin a separate immutable config; other roles never inherit this override.
        self._selector_config=(config if selector_reasoning_effort is None else
                               replace(config,reasoning_effort=selector_reasoning_effort))
        self.authorize,self.record,self.api_key_source=authorize,record,api_key_source
        self.timeout_s=timeout_s
        self.worker=Path(__file__).with_name('http_text_worker.py').resolve()
        self._sources=[Path(__file__).resolve(),self.worker,*[self.worker.with_name(name) for name in
            ('http_text_contract.py','native_execution.py','process_control.py','__init__.py','contracts.py','provenance.py',
             'request_budget.py','text_input_budget.py','spending_budget.py')]]
        self._binding={str(path):file_hash(path) for path in self._sources}
        self._lock=threading.Lock()
        self._requests,self._attempts,self._not_before={}, {}, {}
        private_root.mkdir(mode=0o700,exist_ok=False)

    # tellonce marks where the shared part of a prompt (instructions, observation, rule library) ends. Sent as the
    # system message that part is served from the provider's prompt cache (measured 99.7 % cached on repeats);
    # sent inside the user text it never was (0 %). The model sees the same content either way.
    PROMPT_PREFIX_END='\n[[tellonce-prefix-end]]\n'

    def call(self,prompt: str,*,context: dict,base_instructions=None,validate=None):
        if (not isinstance(prompt, str) or not prompt.strip() or
                (base_instructions is not None and not isinstance(base_instructions, str))):
            raise ValueError('explicit text prompt and optional instructions required')
        if base_instructions is None and self.PROMPT_PREFIX_END in prompt:
            base_instructions,prompt=prompt.split(self.PROMPT_PREFIX_END,1)
            if not prompt.strip():
                base_instructions,prompt=None,base_instructions
        messages=([{'role':'system','content':base_instructions}] if base_instructions is not None else [])
        messages.append({'role':'user','content':prompt})
        config=(self._selector_config if isinstance(context,dict) and context.get('role')=='rule_select'
                else self.config)
        return self._call_messages(messages, context=context, config=config, validate=validate)

    def call_messages(self, messages, *, context, generation, validate=None):
        if not isinstance(context, dict) or context.get('role') != 'feedback':
            raise ValueError('explicit conversation calls are admitted only for feedback')
        config = conversation_config(self.config, generation)
        return self._call_messages(messages, context=context, config=config, validate=validate)

    def _call_messages(self, messages, *, context, config, validate):
        budget=RequestBudget(self.timeout_s)
        keys={'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (not isinstance(context,dict) or set(context)!=keys or
                context.get('phase') not in {'preparation','training','test'} or
                any(not isinstance(context[key],str) or not 1<=len(context[key])<=480 for key in keys-{'attempt'}) or
                not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',context['role']) or
                type(context['attempt']) is not int or context['attempt'] not in {1,2} or
                (validate is not None and not callable(validate))):
            raise ValueError('explicit text role input and logical attempt identity required')
        context=dict(context)
        config.payload(messages)
        messages = [dict(message) for message in messages]
        request_hash=content_hash({'messages':messages,'config':config.__dict__,
            'context':{k:v for k,v in context.items() if k!='attempt'}})
        call_id=context['call_id']
        attempt=(call_id,context['attempt'])
        if {str(path):file_hash(path) for path in self._sources}!=self._binding:
            raise ValueError('HTTP worker source changed after preparation')
        self.authorize('call',{'namespace':self.identity.namespace,'model':config.model,**context})
        with self._lock:
            if (attempt in self._attempts or (call_id in self._requests and self._requests[call_id]!=request_hash) or
                    (context['attempt']==2 and self._attempts.get((call_id,1))!='failed') or
                    time.monotonic()<self._not_before.get(call_id,0)):
                raise ValueError('HTTP text retry is duplicated, changed, premature or lacks a retryable failure')
            self._requests[call_id]=request_hash
            self._attempts[attempt]='running'
        spending = None if self.synthetic else SpendingBudget.for_request(
            self.api_key_source, self.identity, config.model)
        reservation = spending.reserve(self.identity.namespace) if spending is not None else None
        runtime=self.private_root/str(uuid.uuid4())
        started=time.monotonic()
        metrics={}
        worker_error_type=None
        stage='prepare'
        try:
            runtime.mkdir(mode=0o700)
            home=runtime/'home'
            home.mkdir(mode=0o700)
            manifest={'namespace':self.identity.namespace,'context':context,'messages':messages,
                'config':config.__dict__,'request_hash':request_hash,'endpoint':self.endpoint}
            stream_selection=(context['role']=='rule_select' and validate is not None and
                              config.provider=='openrouter' and config.model==GEMMA_MODEL)
            manifest['stream_json_selection']=stream_selection
            (runtime/'request.json').write_text(json.dumps(manifest,ensure_ascii=False,allow_nan=False))
            stage='credentials'
            key=SYNTHETIC_KEY if self.synthetic else self.api_key_source()
            if not isinstance(key,str) or not key:
                raise HTTPTextFailure('HTTP text credential is unavailable')
            request={'config':config.__dict__,'endpoint':self.endpoint,'synthetic':self.synthetic,
                     'messages':messages,'api_key':key,'timeout_s':budget.remaining(300),
                     'stream_json_selection':stream_selection}
            replies=[]
            stage='http'
            capture=run_native_command([str(Path(sys.executable).resolve()),'-S','-B',str(self.worker),
                '--owner-pid',str(os.getpid())],cwd=runtime,
                env={'PATH':'/usr/bin:/bin','HOME':str(home),'LANG':'C.UTF-8','PYTHONNOUSERSITE':'1'},
                prompt=json.dumps(request,ensure_ascii=False,allow_nan=False),timeout_s=budget.remaining(),
                on_event=replies.append,stdout_budget=4*1024*1024,stderr_budget=32768,input_budget=4*1024*1024)
            if capture.status!='completed' or len(replies)!=1:
                metrics={'transport_status':capture.status,'http_attempts_started':None,'usage':None}
                raise HTTPTextFailure('HTTP worker did not produce one completed record')
            envelope=replies[0]
            if not isinstance(envelope,dict) or set(envelope)!={'result','metrics'}:
                raise HTTPTextFailure('HTTP worker result schema changed')
            result=envelope['result']
            metrics=clean_worker_metrics(envelope['metrics'])
            if spending is not None:
                cost = (metrics.get('usage') or {}).get('cost')
                # Explicit pre-completion rejections have no generated output.
                # Network loss and 5xx remain uncertain and keep their reserve.
                if (config.provider == 'openrouter' and
                        metrics.get('http_status') in {400, 401, 402, 403, 404, 422, 429} and
                        result.get('status') == 'failed' and metrics.get('usage') is None):
                    cost = 0.0
                spending.settle(reservation, cost)
            if not isinstance(result,dict):
                raise HTTPTextFailure('HTTP worker result schema changed',metrics=metrics)
            if capture.exit_code!=0 or result.get('status')!='completed':
                if result.get('error_type')=='RepeatedInvalidJSON':
                    worker_error_type='RepeatedInvalidJSON'
                raise HTTPTextFailure('HTTP text request failed',metrics=metrics)
            if (result.get('model')!=config.model or not isinstance(result.get('text'),str) or not result['text'].strip() or
                    not isinstance(result.get('response_id'),str) or not 1<=len(result['response_id'])<=512 or
                    metrics['http_status']!=200 or metrics['http_attempts_started']!=1):
                raise HTTPTextFailure('HTTP text result does not match this model request',metrics=metrics)
            # Retain the bounded, identity-checked response before rejecting a
            # truncated generation. It is diagnostic evidence, never a receipt.
            candidate_fd=os.open(runtime/'response_candidate.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(candidate_fd,'w') as stream:
                json.dump({'result':result,'metrics':metrics},stream,ensure_ascii=False,allow_nan=False)
            if metrics['finish_reason']!='stop':
                raise HTTPTextFailure('HTTP text result does not match this model request',metrics=metrics)
            stage='validation'
            if validate is not None:
                from .http_text_contract import unwrap_json_fence
                result['text']=unwrap_json_fence(result['text'])
                validate(result['text'])
            answer=HTTPTextResult(result['text'],result['response_id'],json.dumps(metrics.get('usage'),sort_keys=True))
            self.record({'type':'http_text_completed','namespace':self.identity.namespace,'context':context,
                'request_hash':request_hash,'metrics':metrics,'runtime':str(runtime),'wall_s':time.monotonic()-started})
            with self._lock:
                self._attempts[attempt]='completed'
            return answer
        except Exception as exc:
            # Credential providers and validators are external callbacks. Their
            # exception messages may contain secrets; only our stage is public.
            failure=HTTPTextFailure('HTTP text role failed at '+stage,metrics=metrics)
            with self._lock:
                self._attempts[attempt]='failed' if failure.retryable else 'nonretryable'
                if failure.retry_after_s is not None:
                    self._not_before[call_id]=time.monotonic()+failure.retry_after_s
            self.record({'type':'http_text_failed','namespace':self.identity.namespace,'context':context,
                'request_hash':request_hash,'stage':stage,'metrics':metrics,'runtime':str(runtime),
                'wall_s':time.monotonic()-started,
                'worker_error_type':worker_error_type,
                'failure_code':validation_failure(exc) if stage == 'validation' else type(exc).__name__[:80]})
            raise HTTPTextFailure(str(failure),metrics=metrics) from None
