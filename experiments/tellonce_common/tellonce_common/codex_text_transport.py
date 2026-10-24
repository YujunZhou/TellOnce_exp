"""Fresh, tool-free native calls for independently projected text roles.

No internal retry and no memory access. The role caller owns its prompt and
one permitted technical retry; the supervisor admits all dependencies before
this transport can obtain credentials or send a native turn.
"""
import json
from pathlib import Path
import re
import threading
import time
import uuid

from .codex_text import TOOL_FREE_FEATURES, TextResult, text_app_server_command, text_environment, text_role_catalog, configured_context_window
from .native_app_server import NativeAppServer
from .native_availability import NativeUnavailable, codex_failure
from .provenance import content_hash
from .request_budget import RequestBudget
from .text_input_budget import check_text_input
from .validation_diagnostics import validation_failure


class CodexTextTransport:
    def __init__(self, *, identity, binary: Path, source_catalog: Path, private_root: Path,
                 config, authorize, record, provision_auth=None, offline_base_url=None, timeout_s=180):
        if (any(not isinstance(p,Path) or not p.is_absolute() or p!=p.resolve()
                for p in (binary,source_catalog,private_root)) or not 0<timeout_s<=900 or
                (offline_base_url is None and provision_auth is None) or
                (offline_base_url is not None and provision_auth is not None)):
            raise ValueError('explicit isolated text runtime and one admitted auth mode required')
        # The managed workspace can expose empty .git placeholders. A Git
        # directory needs HEAD; a gitdir file also marks a real worktree.
        if any((parent/'.git').is_file() or (parent/'.git/HEAD').is_file() or
               ((parent/'HEAD').is_file() and (parent/'objects').is_dir())
               for parent in (private_root,*private_root.parents)):
            raise ValueError('text role runtime must be outside source repositories')
        authorize('prepare',{'namespace':identity.namespace,'model':config.model})
        catalog=source_catalog.read_text()
        self._catalog=text_role_catalog(catalog,model=config.model)
        metadata=json.loads(self._catalog)['models'][0]
        self.context_window=configured_context_window(metadata, config.context_window)
        # UTF-8 bytes and model tokens are different units. The run's input
        # capacity preflight checks tokens separately against this catalog.
        self._input_budget=config.input_max_bytes
        self._catalog_digest=content_hash(json.loads(catalog))
        self._source_catalog=source_catalog
        self.identity,self.binary,self.private_root,self.config=identity,binary,private_root,config
        self.authorize,self.record,self.provision_auth=authorize,record,provision_auth
        self.offline_base_url,self.timeout_s=offline_base_url,timeout_s
        self._requests,self._attempts={},{}
        self._reservation=threading.Lock()
        private_root.mkdir(mode=0o700,exist_ok=False)

    def call(self, prompt: str, *, context: dict, base_instructions: str | None = None,
             validate=None) -> TextResult:
        """Optionally validate the role's output format before accepting it.

        A validator is the admitted role parser (for example parse_feedback),
        never a task-quality scorer used to sample a better answer. Validation
        failure permits the caller's one retry with the same request content.
        """
        budget=RequestBudget(self.timeout_s)
        keys={'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (not isinstance(context,dict) or set(context)!=keys or
                context.get('phase') not in {'preparation','training','test'} or
                any(not isinstance(context[key],str) or not 1<=len(context[key])<=480 for key in keys-{'attempt'}) or
                not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',context['role']) or
                type(context['attempt']) is not int or context['attempt'] not in {1,2} or
                not isinstance(prompt,str) or not prompt.strip() or
                (validate is not None and not callable(validate)) or
                (base_instructions is not None and not isinstance(base_instructions,str))):
            raise ValueError('bounded text role input and explicit call/phase/attempt identity required')
        check_text_input(prompt, base_instructions or '', limit=self._input_budget)
        context=dict(context)
        request_hash=content_hash({'prompt':prompt,'base_instructions':base_instructions,
            'context':{k:v for k,v in context.items() if k!='attempt'}})
        call_id=context['call_id']
        attempt=(call_id,context['attempt'])
        if content_hash(json.loads(self._source_catalog.read_text()))!=self._catalog_digest:
            raise ValueError('pinned native text model catalog changed')
        self.authorize('call',{'namespace':self.identity.namespace,'model':self.config.model,**context})
        with self._reservation:
            if (attempt in self._attempts or (call_id in self._requests and self._requests[call_id]!=request_hash) or
                    (context['attempt']==2 and self._attempts.get((call_id,1))!='failed')):
                raise ValueError('text retry is duplicated, changed or lacks one failed first attempt')
            self._requests[call_id]=request_hash
            self._attempts[attempt]='running'
        runtime=self.private_root/str(uuid.uuid4())
        work,home=runtime/'work',runtime/'cli'
        server=lease=None
        started=time.monotonic()
        stage='prepare'
        usage=None
        result=failure_stage=None
        diagnostic = None
        safe_stderr='';availability=None
        try:
            runtime.mkdir(mode=0o700)
            work.mkdir(mode=0o700)
            home.mkdir(mode=0o700)
            (home/'text-role-model.json').write_text(self._catalog)
            (runtime/'request.json').write_text(json.dumps({'namespace':self.identity.namespace,
                'context':context,'prompt':prompt,'base_instructions':base_instructions,
                'model_config':self.config.__dict__,'request_hash':request_hash},ensure_ascii=False,allow_nan=False))
            command=text_app_server_command(binary=self.binary,workspace=work,cli_home=home,
                config=self.config,offline_base_url=self.offline_base_url)
            server=NativeAppServer(command,cwd=work,env=text_environment(cli_home=home,host_home=runtime/'empty-home'))
            stage='initialize'
            server.request('initialize',{'clientInfo':{'name':'reviewed-experiment-text','version':'1'},
                'capabilities':{'experimentalApi':True}},timeout_s=budget.remaining(30))
            server.notify('initialized',{})
            stage='configuration'
            effective=server.request('config/read',{'includeLayers':False},timeout_s=budget.remaining(10))['config']
            mcp=effective.get('mcp_servers',{})
            features=effective.get('features',{})
            if ((self.config.context_window is not None and effective.get('model_context_window') != self.config.context_window) or
                    not isinstance(mcp,dict) or any(not isinstance(value,dict) or value.get('enabled') is not False
                    for value in mcp.values()) or
                    any(features.get(feature) is not False for feature in TOOL_FREE_FEATURES) or
                    effective.get('cli_auth_credentials_store')!='ephemeral' or
                    effective.get('model_catalog_json')!=str(home/'text-role-model.json')):
                raise ValueError('native text role did not honor its tool/auth/catalog isolation')
            if self.provision_auth is not None:
                stage='auth'
                if budget.remaining()<10:
                    raise TimeoutError('insufficient time for bounded account provisioning')
                lease=self.provision_auth(home,server)
                if any(not callable(getattr(lease,name,None)) for name in ('__enter__','__exit__','begin_turn','redact')):
                    lease=None
                    raise ValueError('text authentication requires an admitted managed lease')
                lease.__enter__()
            stage='thread'
            params={'cwd':str(work),'model':self.config.model,'modelProvider':'experiment_text',
                'approvalPolicy':'never','ephemeral':True,'experimentalRawEvents':True,
                'allowProviderModelFallback':False}
            if base_instructions is not None:
                params['baseInstructions']=base_instructions
            thread=server.request('thread/start',params,timeout_s=budget.remaining(10))
            if (thread.get('model')!=self.config.model or thread.get('modelProvider')!='experiment_text' or
                    thread.get('cwd')!=str(work) or thread.get('approvalPolicy')!='never' or
                    thread.get('runtimeWorkspaceRoots')!=[str(work)] or
                    thread.get('sandbox',{}).get('networkAccess') is not False or
                    thread.get('sandbox',{}).get('type')!='workspaceWrite' or
                    thread.get('activePermissionProfile')!={'id':'experiment','extends':None}):
                raise ValueError('native text thread configuration differs from the admitted role')
            thread_id=thread['thread']['id']
            if str(uuid.UUID(thread_id))!=thread_id:
                raise ValueError('native text thread must have a canonical identity')
            stage='turn'
            if lease is not None:
                lease.begin_turn(str(runtime.name))
            response=server.request('turn/start',{'threadId':thread_id,
                'input':[{'type':'text','text':prompt}],'effort':self.config.reasoning_effort},timeout_s=budget.remaining(10))
            turn_id=response['turn']['id']
            deadline=budget.deadline
            messages,responses=[],set()
            while time.monotonic()<deadline:
                event=server.next_event(timeout_s=min(.25,max(.001,deadline-time.monotonic())))
                if event is None:
                    continue
                if 'id' in event:
                    raise ValueError('a text role cannot issue an unadmitted host request')
                method,params=event.get('method'),event.get('params',{})
                if params.get('threadId') is not None and params['threadId']!=thread_id:
                    raise ValueError('foreign native text thread event')
                if params.get('turnId') is not None and params['turnId']!=turn_id:
                    raise ValueError('foreign native text turn event')
                if method in {'item/started','item/updated','item/completed'}:
                    item=params['item']
                    if item.get('type') not in {'userMessage','agentMessage','reasoning'}:
                        raise ValueError('unexpected tool or context compaction in text-only role')
                    if method=='item/completed' and item.get('type')=='agentMessage':
                        messages.append(item['text'])
                elif method=='rawResponse/completed':
                    if not isinstance(params.get('responseId'),str) or not params['responseId']:
                        raise ValueError('native text response identity is missing')
                    responses.add(params['responseId'])
                elif method=='thread/tokenUsage/updated':
                    usage=params['tokenUsage']['last']
                    if (not isinstance(usage,dict) or any(type(usage.get(key)) is not int or usage[key]<0
                            for key in ('inputTokens','outputTokens'))):
                        raise ValueError('native text usage schema changed')
                elif method=='turn/completed':
                    if params['turn']['id']!=turn_id or params['turn']['status']!='completed':
                        unavailable=codex_failure(params['turn'].get('error'))
                        if unavailable is not None:raise unavailable
                        raise ValueError('native text turn did not complete successfully')
                    break
            else:
                raise TimeoutError('native text role exceeded its time budget')
            if len(responses)!=1 or not messages or not isinstance(messages[-1],str) or not messages[-1].strip():
                raise ValueError('text role must produce one actual model response and nonempty answer')
            result=TextResult(messages[-1],thread_id,json.dumps(usage,sort_keys=True,allow_nan=False))
            stage='validation'
            (runtime/'response_candidate.json').write_text(json.dumps({'text':result.text,'usage':usage,
                'request_hash':request_hash},ensure_ascii=False,allow_nan=False))
            if validate is not None:
                validate(result.text)
        except Exception as exc:
            availability=(exc if isinstance(exc,NativeUnavailable) else
                          NativeUnavailable('native_auth_unavailable') if stage=='auth' else None)
            # A local process that cannot initialize has not received a model
            # request. Preserve that distinction from malformed model output.
            # Protocol/configuration rejection is deliberately not retryable
            # through this route; only observed process/pipe/time failures are.
            if (stage=='initialize' and isinstance(exc,(RuntimeError,TimeoutError,OSError)) and
                    getattr(server,'_failure',None) in
                    {'request_timeout','native_process_exited','write_timeout','write_failed'}):
                availability=NativeUnavailable('native_startup_unavailable')
            failure_stage=stage
            diagnostic = validation_failure(exc) if stage == 'validation' else type(exc).__name__[:80]
        finally:
            try:
                if server is not None:
                    server.close()
                    raw_stderr=server.stderr.decode(errors='replace')
                    if lease is not None:
                        raw_stderr=lease.redact(raw_stderr)
                    safe_stderr=raw_stderr
            except Exception:
                failure_stage=failure_stage or 'cleanup'
            finally:
                try:
                    if lease is not None:
                        lease.__exit__(None,None,None)
                except Exception:
                    failure_stage=failure_stage or 'auth_cleanup'
        if failure_stage is not None:
            with self._reservation:
                self._attempts[attempt]='failed'
            self.record({'type':'text_role_failed','namespace':self.identity.namespace,'context':context,
                'request_hash':request_hash,'stage':failure_stage,'usage':usage,'wall_s':time.monotonic()-started,
                'stderr':safe_stderr,'failure_code':diagnostic,
                'availability_code':availability.code if availability else None,
                'reset_at':availability.reset_at if availability else None})
            if availability is not None:
                raise availability
            raise RuntimeError('native text role failed; see its private stage record') from None
        try:
            (runtime/'response.json').write_text(json.dumps({'text':result.text,'thread_id':thread_id,'usage':usage},
                                                           ensure_ascii=False,allow_nan=False))
            self.record({'type':'text_role_completed','namespace':self.identity.namespace,'context':context,
                'thread_id':thread_id,'request_hash':request_hash,'usage':usage,'wall_s':time.monotonic()-started,
                'runtime':str(runtime),'response_ids':sorted(responses),'stderr':safe_stderr})
        except Exception:
            with self._reservation:
                self._attempts[attempt]='failed'
            raise RuntimeError('native text role result could not be recorded') from None
        with self._reservation:
            self._attempts[attempt]='completed'
        # The process and credential lease are closed. Receipts, prompts,
        # responses and usage remain; the ephemeral CLI databases are not memory.
        try:
            import shutil
            shutil.rmtree(home)
        except OSError:
            self.record({'type':'native_cache_cleanup_deferred','namespace':self.identity.namespace,
                         'runtime':str(runtime)})
        return result
