"""One isolated Claude Code text response per journal-owned attempt."""
import json
from pathlib import Path
import re
import threading
import time
import uuid
from .validation_diagnostics import validation_failure

from .claude_native import ClaudeConfig,TextEvents,command,environment
from .native_execution import run_native_command
from .native_availability import NativeUnavailable
from .provenance import content_hash,file_hash
from .request_budget import RequestBudget
from .text_input_budget import check_text_input
from .http_text_contract import strict_json
from dataclasses import replace


def text_role_instructions(instructions):
    """Preserve role semantics while requiring its requested wire format."""
    return ((instructions or 'Complete the supplied text role without tools.') +
        '\nThis is a machine-consumed text role. Follow the requested output format exactly. '
        'When the task requests JSON, return only the raw JSON value: no Markdown code fences, '
        'preamble, explanation, or text after it. Do not add fields beyond the requested schema.')


def unwrap_json_fence(text):
    """Decode only one complete JSON fence, never extract from commentary."""
    stripped = text.strip()
    for opening in ('```json\n', '```\n'):
        if stripped.startswith(opening) and stripped.endswith('\n```'):
            body = stripped[len(opening):-4]
            strict_json(body)  # Reject duplicate fields, partial JSON and extra blocks.
            return body
    return text


class ClaudeTextTransport:
    def __init__(self,*,identity,binary,private_root,config,authorize,record,credential_source=None,
                 offline_base_url=None,timeout_s=180):
        if (not isinstance(config,ClaudeConfig) or type(timeout_s) not in (int,float) or not 0<timeout_s<=900 or
                any(not isinstance(p,Path) or not p.is_absolute() or p!=p.resolve() for p in (binary,private_root)) or
                (credential_source is None)==(offline_base_url is None) or not callable(authorize) or not callable(record)):
            raise ValueError('admitted Claude model, isolated paths and one auth mode required')
        if any((p/'.git').is_file() or (p/'.git/HEAD').is_file() for p in (private_root,*private_root.parents)):
            raise ValueError('text role runtime must be outside source repositories')
        authorize('prepare',{'namespace':identity.namespace,'model':config.model})
        self.identity,self.binary,self.private_root,self.config=identity,binary,private_root,config
        self.context_window=config.context_window
        self.authorize,self.record,self.credential_source=authorize,record,credential_source
        self.offline_base_url,self.timeout_s=offline_base_url,timeout_s
        self._binary_hash=file_hash(binary);self._requests={};self._attempts={};self._lock=threading.Lock()
        private_root.mkdir(mode=0o700,exist_ok=False)

    def call(self,prompt,*,context,base_instructions=None,validate=None):
        budget=RequestBudget(self.timeout_s)
        keys={'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (not isinstance(context,dict) or set(context)!=keys or context.get('phase') not in {'preparation','training','test'} or
                any(not isinstance(context[k],str) or not 1<=len(context[k])<=480 for k in keys-{'attempt'}) or
                not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',context['role']) or
                type(context['attempt']) is not int or context['attempt'] not in (1,2) or
                not isinstance(prompt,str) or not prompt.strip() or
                (base_instructions is not None and not isinstance(base_instructions,str)) or
                (validate is not None and not callable(validate))):raise ValueError('explicit owned text request required')
        base_instructions=text_role_instructions(base_instructions)
        check_text_input(prompt,base_instructions,limit=self.config.input_max_bytes)
        if file_hash(self.binary)!=self._binary_hash:raise ValueError('pinned Claude binary changed')
        context=dict(context);call_id=context['call_id'];attempt=(call_id,context['attempt'])
        request_hash=content_hash({'prompt':prompt,'base_instructions':base_instructions,
                                  'context':{k:v for k,v in context.items() if k!='attempt'}})
        self.authorize('call',{'namespace':self.identity.namespace,'model':self.config.model,**context})
        with self._lock:
            if (attempt in self._attempts or (call_id in self._requests and self._requests[call_id]!=request_hash) or
                    (context['attempt']==2 and self._attempts.get((call_id,1))!='failed')):
                raise ValueError('attempt duplicated, changed or lacking failed predecessor')
            self._requests[call_id]=request_hash;self._attempts[attempt]='running'
        runtime=self.private_root/str(uuid.uuid4());sid=str(uuid.uuid4());parser=TextEvents(config=self.config,session_id=sid)
        lease=None;result=None;capture=None;stage='prepare';failure=None;started=time.monotonic();stderr='';availability=None
        try:
            runtime.mkdir(mode=0o700)
            work,home,host=runtime/'work',runtime/'cli',runtime/'home'
            for p in (work,home,host):p.mkdir(mode=0o700)
            (runtime/'request.json').write_text(json.dumps({'context':context,'prompt':prompt,
                'base_instructions':base_instructions,'model':self.config.__dict__,'request_hash':request_hash},ensure_ascii=False))
            if self.credential_source is not None:
                stage='auth';lease=self.credential_source(minimum_lifetime_s=budget.remaining())
                lease.__enter__()
            env=environment(cli_home=home,host_home=host,access_token_fd=lease.fd if lease else None,
                            offline_base_url=self.offline_base_url)
            env['PATH']=str(self.binary.parent)+':/usr/bin:/bin'
            args=command(binary=self.binary,workspace=work,cli_home=home,config=self.config,session_id=sid,
                         system_prompt=base_instructions)
            stage='native'
            capture=run_native_command(args,cwd=work,env=env,prompt=prompt,timeout_s=budget.remaining(),
                on_event=parser.observe,input_budget=self.config.input_max_bytes,
                pass_fds=(lease.fd,) if lease else ())
            stderr=lease.redact(capture.stderr) if lease else capture.stderr
            (runtime/'native-stream.jsonl').write_text(lease.redact(capture.stdout) if lease else capture.stdout)
            (runtime/'native-capture.json').write_text(json.dumps({'status':capture.status,'exit_code':capture.exit_code}))
            if capture.status!='completed':raise ValueError('incomplete native event stream')
            result=parser.finish(exit_code=capture.exit_code)
            (runtime/'response_candidate.json').write_text(json.dumps({'text':result.text,'usage':parser.usage},ensure_ascii=False))
            stage='validation'
            if validate is not None:
                decoded=unwrap_json_fence(result.text)
                validate(decoded)
                if decoded!=result.text:
                    self.record({'type':'text_role_format_decoded','namespace':self.identity.namespace,
                        'context':context,'encoding':'single_outer_json_fence',
                        'raw_text_hash':content_hash(result.text),'decoded_text_hash':content_hash(decoded)})
                    result=replace(result,text=decoded)
        except Exception as exc:
            failure=stage
            availability = (exc if isinstance(exc,NativeUnavailable) else
                            NativeUnavailable('native_auth_unavailable') if stage=='auth' else parser.availability)
            diagnostic = validation_failure(exc) if stage == 'validation' else type(exc).__name__[:80]
        finally:
            if lease is not None:lease.__exit__(None,None,None)
        usage=parser.usage
        if failure is not None:
            with self._lock:self._attempts[attempt]='failed'
            self.record({'type':'text_role_failed','namespace':self.identity.namespace,'context':context,
                'request_hash':request_hash,'stage':failure,'usage':usage,'wall_s':time.monotonic()-started,
                'stderr':stderr,'failure_code':diagnostic,
                'availability_code':availability.code if availability else None,
                'reset_at':availability.reset_at if availability else None})
            if availability is not None:
                raise availability from None
            raise RuntimeError('Claude text role failed; see private stage record') from None
        try:
            (runtime/'response.json').write_text(json.dumps({'text':result.text,'thread_id':sid,'usage':usage},ensure_ascii=False))
            self.record({'type':'text_role_completed','namespace':self.identity.namespace,'context':context,
                'thread_id':sid,'request_hash':request_hash,'usage':usage,'wall_s':time.monotonic()-started,
                'runtime':str(runtime),'response_ids':list(parser.messages),'stderr':stderr})
        except Exception:
            with self._lock:self._attempts[attempt]='failed'
            raise RuntimeError('Claude result could not be recorded') from None
        with self._lock:self._attempts[attempt]='completed'
        return result
