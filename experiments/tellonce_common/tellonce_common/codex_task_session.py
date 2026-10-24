"""Task-owned native Codex process implementing the interaction solver port.

The supervisor supplies reviewed sources, fresh task assets, role admission,
an isolated memory port and a durable private audit sink. Real-model admission
is mandatory before construction and each turn; offline fixtures use no auth.
"""
import traceback
import json
from .public_task_contracts import SELECTED_MEMORY_CONTRACT
from .memory_input import native_input_payload
from .native_availability import NativeUnavailable, codex_failure
from pathlib import Path
import shlex
import sys
import time
import uuid

from . import hook_bridge
from .codex_session import app_server_command, prepare_native_catalog
from .codex_text import text_environment, TOOL_FREE_FEATURES
from .hook_bridge import HookBridge
from .interaction_loop import RecoverableCallError, SolverAnswer
from .native_app_server import NativeAppServer
from .native_hooks import discover_hooks, hook_trust_arguments
from .native_rule_guard import NativeRuleGuard
from .native_delivery_evidence import NativeDeliveryEvidence
from .public_dialogue import validate_public_dialogue
from .native_benchmark_tools import NativeBenchmarkTools
from .provenance import file_hash
from .provenance import content_hash
from .rule_appeal import APPEAL_TOOL,AppealReviewer,AppealInputError
from .request_budget import request_budget


ASYNC_TOOL_WAIT_GUIDANCE = (
    'A tool result such as "Script running with cell ID ..." means the operation is still pending, '
    'not that it failed. Tools may wait for pre-execution validation for several minutes before '
    'producing output. Use the available wait mechanism with the SAME cell or session ID until '
    'completion or an explicit error/timeout. Do not cancel and relaunch an operation merely '
    'because one or more yields returned no output. Do not submit a final answer while your '
    'tools remain pending. This does not require unlimited waiting: the host enforces execution '
    'deadlines. An explicit user request to stop, or an explicit tool error requiring termination, '
    'still permits stopping the operation.'
)


class CodexTaskSession:
    def __init__(self, *, start: dict, binary: Path, workspace: Path, private_root: Path,
                 config, source_catalog: Path, memory, artifacts, observation_type, event_type, hooks_enabled: bool,
                 authorize, record, provision_auth=None, offline_base_url=None, timeout_s=600,appeal_reviewer=None,
                 benchmark_tools=None, resume=None):
        if (not isinstance(start,dict) or any(not isinstance(start.get(key),str) or not start[key]
                for key in ('task_id','session_id','namespace')) or type(hooks_enabled) is not bool or
                not 0 < timeout_s <= 86400 or any(not p.is_absolute() or p != p.resolve()
                for p in (binary,workspace,private_root,source_catalog)) or private_root==workspace or workspace in private_root.parents):
            raise ValueError('explicit task identity, canonical private paths and bounded native runtime required')
        if offline_base_url is None and provision_auth is None:
            raise ValueError('live native auth must be provisioned explicitly after source admission')
        if hooks_enabled and (memory is None or memory.start['session_id']!=start['session_id'] or
                memory.start['task_id']!=start['task_id'] or memory.start['namespace']!=start['namespace']):
            raise ValueError('execution hooks require this exact task-owned memory port')
        if appeal_reviewer is not None and (not hooks_enabled or not isinstance(appeal_reviewer,AppealReviewer) or
                appeal_reviewer.identity.namespace!=start['namespace']):
            raise ValueError('appeal requires execution hooks and this experiment-owned reviewer')
        if benchmark_tools is not None and (not isinstance(benchmark_tools, NativeBenchmarkTools)
                or benchmark_tools.owner != {key:start[key] for key in ('namespace','task_id','session_id')}):
            raise ValueError('benchmark tools must belong to this exact task session')
        authorize('prepare',dict(start))
        self.start,self.session_id=dict(start),start['session_id']
        self.workspace,self.private_root,self.artifacts=workspace,private_root,artifacts
        self.memory,self.authorize,self.record=memory,authorize,record
        self.timeout_s,self.config=timeout_s,config
        self._seen,self._closed,self._failed=set(),False,False
        if resume is not None:self._seen=set(resume['seen'])
        self._server,self._bridge,self._guard,self._auth_lease=None,None,None,None
        self._raw_messages=[]
        # The pinned thread/resume protocol has no experimentalRawEvents flag.
        # Native tool hooks may precede the response's usage update. Open a
        # response group at its first observed message/tool hook and close it
        # on the usage update; parallel tools remain in the same group.
        self._resume_response_tracking=resume is not None and hooks_enabled
        self._resume_messages=[]
        self._resume_response_ids=set()
        self._resume_response_group=None
        self._resume_response_open=False
        self._resume_response_sequence=0
        self._appeal_reviewer,self._appeal_calls=appeal_reviewer,{}
        self._benchmark_tools = benchmark_tools
        self._turn_deadline=None
        self._started_native_items=set()
        self._capture_event=None
        self._delivery_captures=[]
        home=private_root/'cli'
        try:
            private_root.mkdir(mode=0o700,exist_ok=resume is not None)
            home.mkdir(mode=0o700,exist_ok=resume is not None)
            self._source_catalog=source_catalog
            self._catalog=home/'native-models.json'
            self._catalog_digest=prepare_native_catalog(source=source_catalog,cli_home=home,model=config.model,
                                                       context_window=config.context_window)
            if file_hash(source_catalog)!=self._catalog_digest:
                raise ValueError('native model catalog changed during preparation')
            if provision_auth is not None and offline_base_url is not None:
                raise ValueError('offline native fixtures must not receive credentials')
            trust=[]
            if hooks_enabled:
                def dispatch(payload):
                    if self._guard is None:
                        raise ValueError('native task identity is not bound')
                    remaining=self._turn_deadline-time.monotonic()-5 if self._turn_deadline is not None else 240
                    if remaining<=0:
                        raise TimeoutError('native task has no remaining hook budget')
                    with request_budget(min(240,remaining)):
                        return self._guard.handle(payload)
                self._bridge=HookBridge(directory=private_root/'bridge',handle=dispatch,timeout_s=300).start()
                command=shlex.join([sys.executable,str(Path(hook_bridge.__file__).resolve()),
                    '--socket',str(self._bridge.path),'--timeout','300'])
                declared=[('PreToolUse','preToolUse','.*'),('PostToolUse','postToolUse','.*'),('Stop','stop',None)]
                hook_file=home/'hooks.json'
                hook_file.write_text(json.dumps({'hooks':{name:[{**({'matcher':matcher} if matcher else {}),
                    'hooks':[{'type':'command','command':command,'timeout':300}]}] for name,_,matcher in declared}}))
                # Local hook discovery must use the already verified task
                # catalog, just like the solver server. Otherwise native
                # startup can refresh its default catalog before hooks/list.
                probe_config=('-c','model_catalog_json='+json.dumps(str(self._catalog)))
                discovered=discover_hooks(binary=binary,cli_home=home,workspace=workspace,
                    config_arguments=probe_config,record=record)
                expected_hooks=tuple({'eventName':event,'command':command,'matcher':matcher,'timeoutSec':300}
                                     for _,event,matcher in declared)
                trust=hook_trust_arguments(discovered,hooks_file=hook_file,workspace=workspace,
                    expected=expected_hooks)
            native_command=app_server_command(binary=binary,workspace=workspace,cli_home=home,
                config=config,hooks_enabled=hooks_enabled,offline_base_url=offline_base_url)+trust
            self._server=NativeAppServer(native_command,cwd=workspace,
                env=text_environment(cli_home=home,host_home=home/'host-home'))
            self._server.request('initialize',{'clientInfo':{'name':'reviewed-experiment-task','version':'1'},
                'capabilities':{'experimentalApi':True}},timeout_s=60)
            self._server.notify('initialized',{})
            if hooks_enabled:
                # Verify the actual solver process before provisioning auth or
                # starting/resuming a thread. A second throwaway app-server
                # repeats SQLite startup without proving this process's trust.
                verified=self._server.request('hooks/list',{'cwds':[str(workspace)]},timeout_s=60)
                if (hook_trust_arguments(verified,hooks_file=hook_file,workspace=workspace,
                        expected=expected_hooks)!=trust or
                        any(h.get('trustStatus')!='trusted' for entry in verified['data'] for h in entry['hooks'])):
                    raise ValueError('declared native hooks are not trusted')
                self.record({'type':'native_hooks_verified','session_id':self.session_id,
                             'basis':'actual_solver_process','hook_count':len(expected_hooks)})
            effective=self._server.request('config/read',{'includeLayers':False},timeout_s=60).get('config',{})
            features=effective.get('features',{})
            disabled=set(TOOL_FREE_FEATURES)-{'shell_tool','code_mode','code_mode_only','hooks'}
            mcp=effective.get('mcp_servers',{})
            if ((config.context_window is not None and effective.get('model_context_window') != config.context_window) or
                    effective.get('model_catalog_json')!=str(self._catalog) or
                    effective.get('cli_auth_credentials_store')!='ephemeral' or
                    effective.get('analytics',{}).get('enabled') is not False or effective.get('notify')!=[] or
                    not isinstance(mcp,dict) or any(not isinstance(v,dict) or v.get('enabled',True) for v in mcp.values()) or
                    any(features.get(name) is not False for name in disabled) or
                    features.get('shell_tool') is not True or features.get('hooks') is not hooks_enabled or
                    features.get('skip_host_skill_discovery') is not True):
                raise ValueError('native task effective configuration differs before authentication')
            self.record({'type':'native_task_configuration_verified','session_id':self.session_id,
                         'catalog_sha256':self._catalog_digest})
            if provision_auth is not None:
                # The admitted broker owns credential refresh and cleanup. In
                # particular, do not independently copy rotating refresh tokens
                # into many task workers or inherit the operator's entire home.
                lease=provision_auth(home,self._server)
                if any(not callable(getattr(lease,name,None)) for name in ('__enter__','__exit__','begin_turn','redact')):
                    raise ValueError('native authentication requires an explicit managed lease')
                self._auth_lease=lease
                lease.__enter__()
            dynamic_tools = ([APPEAL_TOOL] if appeal_reviewer is not None else [])
            if benchmark_tools is not None:
                dynamic_tools += benchmark_tools.definitions
            tool_args = {'dynamicTools': dynamic_tools} if dynamic_tools else {}
            thread_arguments={'cwd':str(workspace),'model':config.model,
                'modelProvider':'experiment_text','approvalPolicy':'never','ephemeral':False,
                'experimentalRawEvents':True,'allowProviderModelFallback':False,**tool_args}
            operation='thread/start'
            if resume is not None:
                rollout=home/resume['rollout']
                if not rollout.is_file() or rollout.resolve()!=rollout or not rollout.is_relative_to(home/'sessions'):
                    raise ValueError('resume requires this task private native rollout')
                operation='thread/resume'
                thread_arguments={k:thread_arguments[k] for k in ('cwd','model','modelProvider','approvalPolicy')}
                thread_arguments.update(threadId=resume['native_thread_id'],path=str(rollout))
            started=self._server.request(operation,thread_arguments,timeout_s=60)
            if (started.get('model')!=config.model or started.get('modelProvider')!='experiment_text' or
                    started.get('cwd')!=str(workspace) or started.get('approvalPolicy')!='never' or
                    started.get('runtimeWorkspaceRoots')!=[str(workspace)] or
                    started.get('sandbox',{}).get('networkAccess') is not False or
                    started.get('sandbox',{}).get('type')!='workspaceWrite' or
                    started.get('activePermissionProfile')!={'id':'experiment','extends':None}):
                raise ValueError('native thread did not honor the declared task configuration')
            self.native_thread_id=started['thread']['id']
            if resume is not None and self.native_thread_id!=resume['native_thread_id']:
                raise ValueError('native resume opened another thread')
            if str(uuid.UUID(self.native_thread_id))!=self.native_thread_id:
                raise ValueError('native task thread identity must be a canonical UUID')
            self.record({'type':'native_task_started','session_id':self.session_id,'native_thread_id':self.native_thread_id,
                         'pid':self._server.pid,'configuration':started,'command':native_command})
            self.execution_evidence=NativeDeliveryEvidence(directory=private_root/'execution-evidence',
                task_id=start['task_id'],session_id=self.session_id,native_thread_id=self.native_thread_id,
                benchmark_tools=tuple(sorted(benchmark_tools.names)) if benchmark_tools is not None else (),
                resume=resume['evidence'] if resume else None)
            if hooks_enabled:
                self._guard=NativeRuleGuard(memory=memory,native_session_id=self.native_thread_id,workspace=workspace,
                    observation_type=observation_type,event_type=event_type,capture_delivery=self._capture_delivery,
                    capture_artifacts=artifacts.capture_artifacts,record=record,
                    require_native_delivery_observation=True,native_response_tracking=True,
                    appeal_review=appeal_reviewer.review if appeal_reviewer is not None else None,
                    capture_appeal_state=artifacts.appeal_state if appeal_reviewer is not None else None,
                    cancel_unstarted_tool=self._cancel_unstarted_native_tool,
                    capture_delivery_context=self._delivery_public_context,
                    capture_public_context=self._public_context_so_far)
                if resume is not None:
                    from .round_recovery import restore_guard
                    restore_guard(self._guard,resume['guard'])
        except Exception:
            self.close()
            raise

    def _capture_delivery(self, text):
        if self._capture_event is None:
            raise ValueError('a delivery capture must belong to an actual user input')
        artifact = self.artifacts.capture_delivery(text)
        evidence = self.execution_evidence.capture(artifact,user_event_id=self._capture_event)
        self._delivery_captures.append((artifact,evidence))
        self.record({'type':'delivery_operation_evidence','session_id':self.session_id,
            'event_id':self._capture_event,'artifact_hash':artifact,'execution_evidence_hash':evidence})
        return artifact

    def _public_context_so_far(self, *, user_event_id):
        """Public dialogue of the current request as observed so far (pre-delivery stages)."""
        if user_event_id != self._capture_event:
            raise ValueError('public dialogue projection requires the current user input')
        owner = self.execution_evidence
        if (owner.task_id != self.start['task_id'] or owner.session_id != self.session_id
                or owner.native_thread_id != self.native_thread_id):
            raise ValueError('public dialogue evidence belongs to another native task session')
        dialogue = owner.public_dialogue_so_far(user_event_id)
        context = {'source': 'captured_native_public_dialogue_so_far', 'task_id': owner.task_id,
                   'session_id': owner.session_id, 'native_thread_id': owner.native_thread_id,
                   'user_event_id': user_event_id, 'public_dialogue': dialogue,
                   'coverage_scope': {'describes': 'Public messages of the current user event observed before this '
                                      'check; a provisional message precedes a tool call that has not started yet.',
                                      'history_scope': 'Current user event only.'}}
        while len(json.dumps(context, ensure_ascii=False)) > 16000 and dialogue['messages']:
            dialogue['messages'].pop(); dialogue['coverage']['complete'] = False
            dialogue['coverage']['omitted_messages'] += 1
        return context

    def _delivery_public_context(self, *, delivery, user_event_id, artifact_hash):
        """Read the public dialogue already frozen with this exact candidate."""
        if (not isinstance(delivery, str) or user_event_id != self._capture_event
                or not self._delivery_captures or self._delivery_captures[-1][0] != artifact_hash):
            raise ValueError('public dialogue requires the latest current delivery capture')
        owner = self.execution_evidence
        if (owner.task_id != self.start['task_id'] or owner.session_id != self.session_id
                or owner.native_thread_id != self.native_thread_id):
            raise ValueError('public dialogue evidence belongs to another native task session')
        if self.artifacts.load(artifact_hash).get('answer') != delivery:
            raise ValueError('public dialogue candidate differs from the captured delivery text')
        evidence_hash = self._delivery_captures[-1][1]
        captured = owner.load(evidence_hash, artifact_hash=artifact_hash, user_event_id=user_event_id)
        dialogue = validate_public_dialogue(captured.get('public_dialogue'), user_event_id)
        context = {'source': 'captured_native_public_dialogue',
            'task_id': owner.task_id, 'session_id': owner.session_id,
            'native_thread_id': owner.native_thread_id, 'user_event_id': user_event_id,
            'artifact_hash': artifact_hash, 'evidence_hash': evidence_hash,
            'public_dialogue': json.loads(json.dumps(dialogue, ensure_ascii=False)),
            'coverage_scope': {
                'complete_describes': 'The bounded source projection, not all prior communication.',
                'unknown_phase_visibility': 'Unphased text without a later tool may be absent; visibility is not established.',
                'history_scope': 'Current user event only; prior delivered messages are not included.',
                'identity_limit': 'The source projection does not deduplicate message notifications by native item ID.'}}

        # The source's raw-text byte bound does not bound JSON escaping or
        # per-message metadata. Keep full messages and disclose extra omissions.
        def encoded_size():
            return len(json.dumps(context, ensure_ascii=False))
        if encoded_size() <= 16000:
            return context
        projected = context['public_dialogue']
        messages = projected['messages']
        source_omitted = projected['coverage']['omitted_messages']
        context['projection_omissions'] = {'reason': 'delivery_context_json_budget', 'omitted_messages': 0}
        def fits(count):
            omitted = len(messages) - count
            projected['messages'] = messages[:count]
            projected['coverage'] = {'complete': source_omitted + omitted == 0,
                'omitted_messages': source_omitted + omitted}
            context['projection_omissions']['omitted_messages'] = omitted
            return encoded_size() <= 16000
        if not fits(0):
            raise ValueError('public dialogue identity metadata exceeds the context budget')
        lower, upper = 0, len(messages)
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if fits(middle):
                lower = middle
            else:
                upper = middle - 1
        if not fits(lower):
            raise ValueError('public dialogue projection exceeded the context budget')
        return context

    def _handle_appeal_call(self,event,turn_id):
        params=event.get('params')
        if (not isinstance(params,dict) or params.get('threadId')!=self.native_thread_id or
                params.get('turnId')!=turn_id or params.get('namespace') is not None or
                params.get('tool')!='tellonce_review' or not isinstance(params.get('callId'),str) or
                not 1<=len(params['callId'])<=480 or self._guard is None):
            self._server.respond(event['id'],error={'code':-32602,'message':'Unadmitted review tool identity.'})
            raise ValueError('foreign native review tool request')
        call_id=params['callId']
        fingerprint=content_hash(params)
        if call_id in self._appeal_calls:
            previous,result=self._appeal_calls[call_id]
            if previous!=fingerprint:
                self._server.respond(event['id'],error={'code':-32602,'message':'Review tool call changed its input.'})
                raise ValueError('native review tool identity was reused with different input')
        else:
            args=params.get('arguments')
            try:
                if not isinstance(args,dict) or set(args)!={'disputes'}:
                    raise ValueError('invalid review arguments')
                deadline=getattr(self,'_turn_deadline',None)
                remaining=deadline-time.monotonic()-5 if deadline is not None else 240
                if remaining<=0:
                    outcome={'status':'unavailable','reason':'No remaining task time for this review.'}
                else:
                    with request_budget(min(240,remaining)):
                        outcome=self._guard.submit_appeal(args['disputes'])
            except AppealInputError as error:
                outcome=error.response()
            except ValueError:
                outcome={'status':'invalid','reason':'Use current blocked-check references and exact captured counterevidence; '
                         'wait for pending tools to finish. Changed or foreign evidence is not eligible.'}
            result={'success':outcome['status']=='completed','contentItems':[
                {'type':'inputText','text':json.dumps(outcome,ensure_ascii=False,allow_nan=False)}]}
            self._appeal_calls[call_id]=(fingerprint,result)
        self._server.respond(event['id'],result=result)

    def _begin_resumed_response(self, turn_id):
        if not self._resume_response_open:
            self._resume_response_sequence+=1
            self._resume_response_group='native-resumed-'+content_hash([
                self.native_thread_id,turn_id,self._resume_response_sequence])
            self._resume_response_open=True
            self._guard.observe_model_response(self._resume_response_group,messages=(),native_session_id=self.native_thread_id)

    def _event(self, event, turn_id, messages):
        audit_event=event
        if self._auth_lease is not None:
            audit_event=json.loads(self._auth_lease.redact(json.dumps(event,ensure_ascii=False)))
        self.record({'type':'native_protocol_event','session_id':self.session_id,'event':audit_event})
        if 'id' in event:
            if event.get('method')=='item/tool/call':
                tool = event.get('params', {}).get('tool')
                if self._benchmark_tools is not None and tool in self._benchmark_tools.names:
                    try:
                        self.authorize('benchmark_tool', dict(self.start))
                        remaining = self._turn_deadline-time.monotonic()-5
                        reply = self._benchmark_tools.call_native(event['params'], native_thread_id=self.native_thread_id,
                            turn_id=turn_id, unit_id=self._benchmark_unit, event_id=self._capture_event,
                            timeout_s=max(0, min(120, remaining)))
                    except Exception:
                        self._server.respond(event['id'], error={'code':-32602,'message':'Invalid or unavailable benchmark tool request.'})
                        raise
                    self._server.respond(event['id'], result=reply)
                    return False
                if tool == 'tellonce_review' and self._appeal_reviewer is not None:
                    self._handle_appeal_call(event,turn_id)
                    return False
            self._server.respond(event['id'],error={'code':-32601,'message':'This experiment has not admitted this host request.'})
            raise ValueError('unadmitted native host request')
        method,params=event.get('method'),event.get('params',{})
        if params.get('threadId') is not None and params['threadId']!=self.native_thread_id:
            raise ValueError('native event belongs to another task thread')
        if (method=='hook/completed' and self._bridge and params.get('turnId')==turn_id and
                params.get('threadId')==self.native_thread_id):
            run=params.get('run',{})
            # Callback/client failures leave the listener alive. Native stop
            # hooks surface them as blocked feedback and may otherwise ask the
            # model to respond forever. Only our task-owned transport marker
            # is fatal; ordinary rule corrections still follow the guard flow.
            if (run.get('source')=='user' and run.get('handlerType')=='command' and
                    run.get('sourcePath')==str(self.private_root/'cli/hooks.json') and
                    run.get('eventName') in {'preToolUse','postToolUse','stop'} and
                    run.get('status')=='blocked' and any(
                        entry.get('kind')=='feedback' and entry.get('text')==hook_bridge.FAILURE_MESSAGE
                        for entry in run.get('entries',[]))):
                raise RuntimeError('native task hook transport failed; reject this turn')
        # Tool history is task-wide. Public opening text, however, belongs only
        # to this user response; a late message from a prior native turn must
        # not become the new request's first paragraph.
        current_turn = isinstance(turn_id, str) and bool(turn_id) and params.get('turnId') == turn_id
        if not (params.get('item', {}).get('type') == 'agentMessage' and not current_turn):
            self.execution_evidence.observe(audit_event, public_dialogue=current_turn)
        if getattr(self,'_resume_response_tracking',False) and params.get('turnId')==turn_id and self._guard:
            if method=='hook/started' and params.get('run',{}).get('eventName')=='preToolUse':
                self._begin_resumed_response(turn_id)
            elif method=='thread/tokenUsage/updated':
                usage=params.get('tokenUsage',{}).get('total',{})
                if not usage or any(type(v) is not int or v<0 for v in usage.values()):
                    raise ValueError('resumed response requires actual native usage evidence')
                boundary=content_hash([self.native_thread_id,turn_id,usage])
                if boundary not in self._resume_response_ids:
                    self._resume_response_ids.add(boundary)
                    self._begin_resumed_response(turn_id)
                    self._resume_response_open=False
                    self.record({'type':'resumed_native_response_boundary','session_id':self.session_id,
                                 'native_turn_id':turn_id,'response_group':self._resume_response_group,
                                 'basis':'native_message_or_tool_through_usage_update'})
        if method=='rawResponseItem/completed' and params.get('turnId')==turn_id:
            item=params['item']
            if item.get('type')=='message' and item.get('role')=='assistant':
                self._raw_messages.append(''.join(part['text'] for part in item.get('content',[])
                                                 if part.get('type') in {'output_text','input_text'}))
        elif method=='rawResponse/completed' and params.get('turnId')==turn_id:
            if self._guard:
                self._guard.observe_model_response(params['responseId'],messages=tuple(self._raw_messages),
                    native_session_id=self.native_thread_id)
            self._raw_messages.clear()
        elif method in {'item/started','item/updated','item/completed'}:
            item=params.get('item',{})
            kind=item.get('type')
            if isinstance(item.get('id'),str) and item['id']:
                # Tests build sessions without __init__; the set is created on first use.
                self.__dict__.setdefault('_started_native_items',set()).add(item['id'])
            if method=='item/completed' and kind=='agentMessage' and current_turn:
                messages.append(item['text'])
                if getattr(self,'_resume_response_tracking',False) and self._guard:
                    self._begin_resumed_response(turn_id)
                    self._guard.observe_response_text(self._resume_response_group,item['text'],native_session_id=self.native_thread_id)
            benchmark_event = (kind == 'dynamicToolCall' and self._benchmark_tools is not None
                               and item.get('tool') in self._benchmark_tools.names)
            if self._guard and (kind in {'agentMessage','commandExecution','fileChange','mcpToolCall'} or benchmark_event):
                mapping={'agentMessage':'agent_message','commandExecution':'command_execution',
                         'fileChange':'file_change','mcpToolCall':'mcp_tool_call','dynamicToolCall':'dynamic_tool_call'}
                normalized={**item,'type':mapping[kind],'native_turn_id':params.get('turnId')}
                for old,new in [('exitCode','exit_code'),('aggregatedOutput','aggregated_output')]:
                    if old in normalized:
                        normalized[new]=normalized.pop(old)
                self._guard.observe_native_event({'type':method.replace('/','.'),'item':normalized},
                    native_session_id=self.native_thread_id)
        return method=='turn/completed' and params.get('turn',{}).get('id')==turn_id

    def feedback_operations(self, answer, *, event_id):
        if (not isinstance(answer,SolverAnswer) or not self._delivery_captures or
                event_id!=self._capture_event or
                self._delivery_captures[-1]!=(answer.artifact_hash,answer.execution_evidence_hash)):
            raise ValueError('feedback must use this native session’s actual latest delivery')
        return json.dumps(self.execution_evidence.feedback_view(answer.execution_evidence_hash,
            artifact_hash=answer.artifact_hash,user_event_id=event_id),ensure_ascii=False,allow_nan=False)

    def respond(self, request):
        required={'task_id','unit_id','instruction','context','session_id','event_id','user_text','memory_text','memory_version'}
        if (self._closed or self._failed or not isinstance(request,dict) or set(request) not in (required,required|{'memory_kind'}) or
                request['session_id']!=self.session_id or request['task_id']!=self.start['task_id'] or
                not isinstance(request['event_id'],str) or not request['event_id'] or request['event_id'] in self._seen or
                any(not isinstance(request[key],str) for key in ('user_text','memory_text','memory_version','instruction'))):
            raise ValueError('native solver input is duplicated, foreign or not the public protocol')
        payload=native_input_payload(request)
        self.authorize('turn',dict(self.start))
        if any(file_hash(path)!=self._catalog_digest for path in (self._catalog,self._source_catalog)):
            raise ValueError('native model catalog changed after task preparation')
        self._seen.add(request['event_id'])
        self._capture_event=request['event_id']
        self.execution_evidence.begin_request(request['event_id'])
        self._benchmark_unit=request['unit_id']
        self._delivery_captures=[]
        if self._auth_lease is not None:
            self._auth_lease.begin_turn(request['event_id'])
        if self._guard:
            if self.memory.runtime.publication.digest != request['memory_version']:
                raise ValueError('selected text and execution support use different memory versions')
            self._guard.begin_delivery(request['event_id'], user_text=request['user_text'])
        self._raw_messages=[]
        # Preserve this boundary's memory type; raw retrieval is not rule selection.
        prompt=json.dumps({**payload,'tool_execution_guidance':ASYNC_TOOL_WAIT_GUIDANCE},ensure_ascii=False)
        messages=[]
        self._resume_response_open=False
        try:
            self._turn_deadline=time.monotonic()+self.timeout_s
            reply=self._server.request('turn/start',{'threadId':self.native_thread_id,
                'clientUserMessageId':request['event_id'],'input':[{'type':'text','text':prompt}],
                'effort':self.config.reasoning_effort},timeout_s=60)
            turn_id=reply['turn']['id']
            self.execution_evidence.bind_request_turn(request['event_id'], turn_id,
                origin='codex_turn_start_reply')
            deadline=self._turn_deadline
            while time.monotonic() < deadline:
                if self._bridge:
                    self._bridge.assert_healthy()
                event=self._server.next_event(timeout_s=min(.25,max(.001,deadline-time.monotonic())))
                if event is None:
                    continue
                if self._event(event,turn_id,messages):
                    if event['params']['turn']['status']!='completed':
                        unavailable=codex_failure(event['params']['turn'].get('error'))
                        if unavailable is not None:raise unavailable
                        raise RuntimeError('native task turn failed')
                    break
            else:
                raise TimeoutError('native task turn exceeded its time budget')
            if not messages or not isinstance(messages[-1],str) or not messages[-1].strip():
                raise ValueError('native task produced no actual delivery')
            text=messages[-1]
            if self._bridge:
                self._bridge.assert_healthy()
                summary=self._guard.delivery_summary()
                if not summary['completed']:
                    raise ValueError('native turn ended before its delivery checks')
                final=self.artifacts.load(summary['final_artifact_hash'])
                before=self.artifacts.load(summary['before_repair_artifact_hash'])
                if final['answer']!=text:
                    raise ValueError('native delivered text differs from the captured checked draft')
                if (not self._delivery_captures or self._delivery_captures[0][0]!=summary['before_repair_artifact_hash'] or
                        self._delivery_captures[-1][0]!=summary['final_artifact_hash']):
                    raise ValueError('native operation evidence differs from actual draft capture order')
                self.record({'type':'native_repair_summary','session_id':self.session_id,'event_id':request['event_id'],
                             'summary':summary})
                return SolverAnswer(text,summary['final_artifact_hash'],summary['following_candidates'],
                    summary['before_repair_artifact_hash'],before['answer'],
                    json.dumps(self.artifacts.feedback_view(summary['final_artifact_hash'],current_request=request['user_text']),ensure_ascii=False),
                    execution_evidence_hash=self._delivery_captures[-1][1],
                    before_repair_execution_evidence_hash=self._delivery_captures[0][1])
            final_hash=self._capture_delivery(text)
            return SolverAnswer(text,final_hash,
                visible_artifacts_json=json.dumps(self.artifacts.feedback_view(final_hash,current_request=request['user_text']),ensure_ascii=False),
                execution_evidence_hash=self._delivery_captures[-1][1])
        except Exception as exc:
            self._failed=True
            self.record({'type':'native_turn_failed','session_id':self.session_id,'event_id':request['event_id'],
                         'error_type':type(exc).__name__,
                         'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)[-6:]],
                         'availability_code':exc.code if isinstance(exc,NativeUnavailable) else None,
                         'reset_at':exc.reset_at if isinstance(exc,NativeUnavailable) else None})
            self.close()
            raise RecoverableCallError(exc.code if isinstance(exc,NativeUnavailable) else 'native_task_turn_failed') from None

    def _cancel_unstarted_native_tool(self, *, tool_use_id, tool_name, tool_input):
        """Prove at the delivery boundary that a proposed operation never ran.

        Host-owned benchmark tools keep their own atomic fence. A native operation
        (command, patch, MCP call) is executed by Codex only as an item whose id is
        the hook's tool_use_id; Codex can also answer a call without starting any
        item (2026-10-02 ClawArena hil_g1: "apply_patch verification failed: Failed
        to read file to update …" returned the call output with no fileChange item
        and no PostToolUse hook). Stop then arrived with that operation still
        pending and the whole turn was rejected, discarding a finished unit three
        times. By Stop every item of the turn has started or never will, so an id
        that never started is a proven non-execution."""
        if (self._benchmark_tools is not None and isinstance(tool_name,str)
                and tool_name in self._benchmark_tools.names):
            return self._benchmark_tools.cancel_if_unstarted(tool_use_id=tool_use_id,tool_name=tool_name,tool_input=tool_input)
        if not isinstance(tool_use_id,str) or not tool_use_id:
            raise ValueError('exact native tool identity required for cancellation')
        started = tool_use_id in getattr(self,'_started_native_items',set())
        if not started:
            self.record({'type':'native_tool_never_started','session_id':self.session_id,'tool_use_id':tool_use_id,
                         'tool_name':tool_name if isinstance(tool_name,str) else None})
        return not started

    def close(self):
        if self._closed:
            return
        self._closed=True
        try:
            if self._server is not None:
                self._server.close()
                stderr=self._server.stderr.decode(errors='replace')
                if self._auth_lease is not None:
                    stderr=self._auth_lease.redact(stderr)
                self.record({'type':'native_task_stderr','session_id':self.session_id,'stderr':stderr})
        finally:
            try:
                if self._bridge is not None:
                    self._bridge.close()
            finally:
                try:
                    if self._auth_lease is not None:
                        self._auth_lease.__exit__(None,None,None)
                finally:
                    try:
                        if self._benchmark_tools is not None:
                            self._benchmark_tools.close()
                    finally:
                        self.artifacts.close()
