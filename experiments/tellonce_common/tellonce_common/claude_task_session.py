"""Task-owned Claude Code session with explicit native events and private homes.

File and benchmark tasks retain their admitted tool sets. TRACE uses the shared
guard with native Claude proposal/result binding and a private hook channel.
"""
import traceback
import json
from .public_task_contracts import SELECTED_MEMORY_CONTRACT
from .memory_input import native_input_payload
import math
import shlex
import sys
from .native_availability import NativeUnavailable
from pathlib import Path
import time
import uuid

from .claude_native import ClaudeConfig,command,environment,task_settings,thinking_progress
from .claude_benchmark_mcp import ClaudeBenchmarkMCP
from .native_benchmark_tools import NativeBenchmarkTools
from .interaction_loop import RecoverableCallError,SolverAnswer
from .native_execution import run_native_command
from .native_delivery_evidence import NativeDeliveryEvidence
from .provenance import file_hash
from .claude_trace import ClaudeTrace, REVIEW_NAME
from .hook_bridge import HookBridge
from . import hook_bridge
from .rule_appeal import AppealReviewer


_NATIVE_CONTINUATION_MESSAGES = {
    '[Your previous response had no visible output. Please continue and produce a user-visible response.]':
        'empty_visible_output',
    'Your response above was cut off mid-stream. Resume directly from where it stops — no apology, no recap. '
        'If none of it survived, answer the request from the start.':
        'truncated_visible_output',
}


class ClaudeTaskSession:
    def __init__(self,*,start,binary,workspace,private_root,config,source_catalog,memory,artifacts,
                 observation_type,event_type,hooks_enabled,authorize,record,provision_auth=None,
                 offline_base_url=None,timeout_s=600,appeal_reviewer=None,benchmark_tools=None,resume=None):
        if (not isinstance(config,ClaudeConfig) or any(not isinstance(start.get(k),str) or not start[k]
                for k in ('task_id','session_id','namespace')) or
                any(not isinstance(p,Path) or not p.is_absolute() or p!=p.resolve() for p in (binary,workspace,private_root)) or
                private_root==workspace or workspace in private_root.parents or private_root in workspace.parents or
                type(timeout_s) not in (int,float) or not 0<timeout_s<=3600):
            raise ValueError('explicit admitted Claude task and isolated private runtime required')
        if type(hooks_enabled) is not bool or (hooks_enabled and
                (memory is None or any(memory.start.get(k) != start[k] for k in ('session_id','task_id','namespace')) or
                 observation_type is None or event_type is None)):
            raise ValueError('execution hooks require this exact task-owned memory port')
        if appeal_reviewer is not None and (not hooks_enabled or not isinstance(appeal_reviewer, AppealReviewer) or
                appeal_reviewer.identity.namespace != start['namespace']):
            raise ValueError('appeal requires execution hooks and this experiment-owned reviewer')
        if benchmark_tools is not None and (not isinstance(benchmark_tools, NativeBenchmarkTools) or
                benchmark_tools.owner != {key: start[key] for key in ('namespace', 'task_id', 'session_id')}):
            raise ValueError('Claude benchmark tools must belong to this task')
        if (provision_auth is None)==(offline_base_url is None):
            raise ValueError('one explicit native login or synthetic provider required')
        authorize('prepare',dict(start))
        self.start,self.session_id=dict(start),start['session_id'];self.native_thread_id=str(uuid.uuid4())
        if resume is not None:self.native_thread_id=resume['native_thread_id']
        self.binary,self.workspace,self.private_root,self.config=binary,workspace,private_root,config
        self.memory,self.artifacts,self.authorize,self.record=memory,artifacts,authorize,record
        self.credential_source,self.offline_base_url,self.timeout_s=provision_auth,offline_base_url,timeout_s
        self._seen=set();self._failed=False;self._closed=False;self._completed_turns=0
        if resume is not None:
            self._seen=set(resume['seen']);self._completed_turns=resume['completed_turns']
        self._last_delivery=None
        self._trace = self._bridge = None
        self._appeal_reviewer = appeal_reviewer
        self._capture_event = None
        self._delivery_captures = []
        self.benchmark_tools=benchmark_tools
        self._mcp_names = {'mcp__benchmark__'+name for name in benchmark_tools.names} if benchmark_tools else set()
        if appeal_reviewer is not None:
            self._mcp_names.add(REVIEW_NAME)
        self._usage={'inputTokens':0,'outputTokens':0,'cachedInputTokens':0};self._pending_tools={}
        self._background_tasks={}
        if resume is not None:self._usage=dict(resume['usage'])
        self._binary_hash=file_hash(binary)
        private_root.mkdir(mode=0o700,exist_ok=resume is not None)
        self.home=private_root/'cli';self.host=private_root/'home'
        for p in (self.home,self.host):p.mkdir(mode=0o700,exist_ok=resume is not None)
        self.settings=private_root/'settings.json'
        self._host_roots={p.name for p in Path('/').iterdir()}
        self.execution_evidence=NativeDeliveryEvidence(directory=private_root/'execution-evidence',task_id=start['task_id'],
            session_id=self.session_id,native_thread_id=self.native_thread_id,resume=resume['evidence'] if resume else None)
        try:
            hooks = None
            if hooks_enabled:
                self._trace = ClaudeTrace(memory=memory, native_session_id=self.native_thread_id, workspace=workspace,
                    observation_type=observation_type, event_type=event_type, capture_delivery=self._capture_delivery,
                    artifacts=artifacts, record=record, reviewer=appeal_reviewer)
                if resume is not None:
                    from .round_recovery import restore_guard
                    restore_guard(self._trace.guard,resume['guard'])
                self._bridge = HookBridge(directory=private_root/'bridge', handle=self._trace.dispatch, timeout_s=300).start()
                hook_command = shlex.join([sys.executable, '-B', str(Path(hook_bridge.__file__).resolve()),
                    '--socket', str(self._bridge.path), '--timeout', '300'])
                hooks = {name:[{'hooks':[{'type':'command','command':hook_command,'timeout':300}]}]
                         for name in ('PreToolUse','PostToolUse','Stop')}
            self.settings.write_text(json.dumps(task_settings(workspace=workspace,cli_home=self.home,binary=binary,hooks=hooks)))
            self._settings_hash=file_hash(self.settings)
        except Exception:
            self.close()
            raise
        self.record({'type':'native_task_started','session_id':self.session_id,'native_thread_id':self.native_thread_id,
                     'runtime':'claude','model':config.model,'binary_sha256':self._binary_hash})

    def _capture_delivery(self, text):
        if self._capture_event is None:
            raise ValueError('delivery capture requires an actual user input')
        artifact = self.artifacts.capture_delivery(text)
        evidence = self.execution_evidence.capture(artifact, user_event_id=self._capture_event)
        self._delivery_captures.append((artifact,evidence))
        self.record({'type':'delivery_operation_evidence','session_id':self.session_id,
            'event_id':self._capture_event,'artifact_hash':artifact,'execution_evidence_hash':evidence})
        return artifact

    def _normalized(self,method,**params):
        event={'method':method,'params':{'threadId':self.native_thread_id,**params},
               'emittedAtMs':int(time.time()*1000),'adapter_normalized_from':'claude_stream_json'}
        self.record({'type':'native_protocol_event','session_id':self.session_id,'event':event})
        return event

    def _tool(self,state,ident,**fields):
        if fields.get('tool') == REVIEW_NAME:
            return  # Internal check disputes are not benchmark operation evidence.
        event={'method':'item/'+state,'params':{'threadId':self.native_thread_id,'turnId':self._turn_id,
            'item':{'type':'claudeToolCall','id':ident,**fields}}}
        self.execution_evidence.observe(event)

    def _observe(self,row):
        if not isinstance(row,dict) or row.get('session_id') not in (None,self.native_thread_id):
            raise ValueError('foreign native Claude session event')
        self.record({'type':'claude_protocol_event','session_id':self.session_id,'event':row})
        kind=row.get('type')
        if kind=='system':
            subtype=row.get('subtype')
            if subtype=='init':
                allowed={'Read','Write','Edit','Bash','Glob','Grep'} | self._mcp_names
                servers=row.get('mcp_servers')
                valid_servers = (isinstance(servers,list) and len(servers)==1 and
                    servers[0].get('name')=='benchmark' and servers[0].get('status')=='connected') if self._mcp_names else servers==[]
                if (self._initialized or row.get('cwd')!=str(self.workspace) or row.get('model')!=self.config.model or
                        set(row.get('tools',[]))!=allowed or not valid_servers or
                        row.get('skills')!=[] or row.get('plugins')!=[] or row.get('permissionMode')!='dontAsk'):
                    raise ValueError('Claude effective task configuration differs')
                self._initialized=True
            elif subtype=='status' and row.get('status') in {'requesting','compacting',None}:pass
            elif subtype=='thinking_tokens':thinking_progress(row)
            elif subtype=='task_started':
                pending=self._pending_tools.get(row.get('tool_use_id'))
                if (pending is None or pending[0]!='Bash' or
                        not isinstance(row.get('task_id'),str) or not row['task_id'] or
                        row.get('task_type')!='local_bash' or
                        type(row.get('is_backgrounded')) is not bool or
                        not isinstance(row.get('description'),str) or not row['description'].strip()):
                    raise ValueError('local Bash task start does not match a pending operation')
                task_id=row['task_id']
                background_tasks=getattr(self,'_background_tasks',{})
                if task_id in background_tasks:
                    raise ValueError('duplicate local Bash task start')
                background_tasks[task_id]=row['tool_use_id']
                self._background_tasks=background_tasks
            elif subtype=='background_tasks_changed':
                tasks=row.get('tasks')
                if not isinstance(tasks,list):
                    raise ValueError('background task update is not a list')
                background_tasks=getattr(self,'_background_tasks',{})
                seen=set()
                for task in tasks:
                    if (not isinstance(task,dict) or
                            not isinstance(task.get('task_id'),str) or not task['task_id'] or
                            task['task_id'] in seen or task['task_id'] not in background_tasks or
                            task.get('task_type')!='local_bash' or
                            not isinstance(task.get('description'),str) or not task['description'].strip()):
                        raise ValueError('background task update does not match an admitted local Bash task')
                    seen.add(task['task_id'])
            elif subtype=='task_notification':
                task_id=row.get('task_id')
                tool_use_id=row.get('tool_use_id')
                background_tasks=getattr(self,'_background_tasks',{})
                if (not isinstance(task_id,str) or not task_id or
                        background_tasks.get(task_id)!=tool_use_id or
                        row.get('status') not in {'completed','failed','stopped'} or
                        not isinstance(row.get('summary'),str) or
                        not isinstance(row.get('output_file'),str)):
                    raise ValueError('background task notification does not match an admitted local Bash task')
            elif subtype=='compact_boundary':
                from .claude_native import compact_boundary
                compact_boundary(row)
                metadata = row['compact_metadata']
                anchor = metadata.get('preserved_segment', {}).get('anchor_uuid')
                if anchor is None:
                    anchor = metadata.get('preserved_messages', {}).get('anchor_uuid')
                self._compact_summary_anchor = anchor if isinstance(anchor, str) and anchor else None
            elif subtype=='notification' and self._trace is not None:
                self._bridge.assert_healthy()
                self._trace.observe_notification(row)
            elif subtype=='permission_denied':
                # A refused operation is observable task behaviour, not a
                # transport failure. Its following tool_result closes the call.
                pending=self._pending_tools.get(row.get('tool_use_id'))
                if pending is None or row.get('tool_name')!=pending[0]:
                    raise ValueError('permission refusal does not match a pending tool')
            else:raise ValueError('unadmitted native system event or implicit retry/compaction')
        elif kind=='assistant':
            message=row.get('message',{})
            if message.get('model')!=self.config.model or row.get('parent_tool_use_id') is not None:
                raise ValueError('model changed or subagent invoked')
            for part in message.get('content',[]):
                if part.get('type')=='text':
                    self._messages.append(part['text'])
                    self.execution_evidence.observe_public_text(part['text'])
                elif part.get('type')=='tool_use':
                    ident=part['id'];name=part['name'];args=part['input']
                    if ident in self._pending_tools:
                        raise ValueError('unadmitted or repeated tool operation')
                    self._pending_tools[ident]=(name,args)
                    self._tool('started',ident,tool=name,arguments=args,status='requested')
                elif part.get('type') not in {'thinking','redacted_thinking'}:
                    raise ValueError('unadmitted assistant content')
            if self._trace is not None:
                # Publishing the native response wakes the Stop-hook thread.
                # Queue its public candidate first so an immediate capture
                # clears rejected final text before any later tool can flush it
                # as if it were a public preamble. Preserve content order above.
                self._trace.observe_assistant(message)
        elif kind=='user':
            anchor = getattr(self, '_compact_summary_anchor', None)
            if anchor is not None and row.get('uuid') == anchor:
                message = row.get('message', {})
                parts = message.get('content')
                if (row.get('isSynthetic') is not True or row.get('parent_tool_use_id') is not None or
                        row.get('session_id') != self.native_thread_id or message.get('role') != 'user' or
                        not isinstance(parts, list) or not parts or any(
                            not isinstance(part, dict) or part.get('type') != 'text' or
                            not isinstance(part.get('text'), str) or not part['text'].strip() for part in parts)):
                    raise ValueError('invalid native compaction summary')
                self._compact_summary_anchor = None
                self.record({'type': 'claude_native_compaction_summary', 'session_id': self.session_id,
                             'summary_uuid': anchor})
                return
            # Claude may request a visible answer after an empty or truncated
            # native output. These are native runtime controls, not user
            # corrections or feedback from our Stop hook. Match only the
            # observed exact messages, for every experimental method.
            if row.get('isSynthetic') is True and row.get('parent_tool_use_id') is None:
                message = row.get('message')
                content = message.get('content') if isinstance(message, dict) else None
                text = content[0].get('text') if (isinstance(message, dict) and message.get('role') == 'user' and
                    isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict) and
                    content[0].get('type') == 'text' and isinstance(content[0].get('text'), str)) else None
                reason = _NATIVE_CONTINUATION_MESSAGES.get(text)
                if reason is not None:
                    self.record({'type':'claude_native_continuation','session_id':self.session_id,
                                 'reason':reason})
                    return
            if row.get('isSynthetic') is True and self._trace is not None:
                self._trace.observe_control_message(row)
                return
            for part in row.get('message',{}).get('content',[]):
                if part.get('type')!='tool_result':raise ValueError('unexpected user content from native runtime')
                ident=part['tool_use_id']
                if ident not in self._pending_tools:raise ValueError('unknown native tool result')
                name,args=self._pending_tools.pop(ident)
                if name not in {'Read','Write','Edit','Bash','Glob','Grep'} | self._mcp_names and not part.get('is_error'):
                    raise ValueError('unadmitted tool must be rejected by the native runtime')
                if name in self._mcp_names:
                    self._mcp_observed.append({'tool':name.removeprefix('mcp__benchmark__'), 'arguments':args,
                        'content':part.get('content'), 'isError':bool(part.get('is_error',False))})
                self._tool('completed',ident,tool=name,arguments=args,status='completed',result=part.get('content'),
                           isError=part.get('is_error'),nativeToolResult=row.get('tool_use_result'))
                if self._trace is not None:
                    # Publish immutable operation evidence before allowing a
                    # concurrent Stop hook to capture the completed delivery.
                    self._trace.tools.observe_result(part,native_session_id=self.native_thread_id,
                        native_result=row.get('tool_use_result'))
        elif kind=='result':
            if self._result is not None or row.get('subtype')!='success' or row.get('is_error') is not False:
                raise ValueError('native Claude turn failed or duplicated result')
            self._result=row
        elif kind=='rate_limit_event':
            if row.get('rate_limit_info',{}).get('status')=='rejected':
                self._availability=NativeUnavailable('native_quota_exhausted',row['rate_limit_info'].get('resetsAt'))
                raise self._availability
        elif kind=='tool_progress':
            ident=row.get('parent_tool_use_id')
            pending=self._pending_tools.get(ident)
            elapsed=row.get('elapsed_time_seconds')
            if (pending is None or row.get('tool_name')!=pending[0] or row.get('heartbeat') is not True or
                    not isinstance(row.get('tool_use_id'),str) or not row['tool_use_id'].startswith(ident+'-heartbeat-') or
                    type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0):
                raise ValueError('tool heartbeat does not belong to a pending native operation')
            # Progress is not completion. Preserve pending tool/check/evidence
            # bindings until the real tool_result and post hook arrive.
        elif kind!='stream_event':raise ValueError('unknown native Claude event')

    def _observe_recorded(self, row):
        try:
            self._observe(row)
        except Exception as exc:
            self.record({'type': 'claude_observer_failed', 'session_id': self.session_id,
                         'error_type': type(exc).__name__,
                         'frames': [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                                    for f in traceback.extract_tb(exc.__traceback__)[-6:]]})
            raise

    def respond(self,request):
        required={'task_id','unit_id','instruction','context','session_id','event_id','user_text','memory_text','memory_version'}
        if (self._closed or self._failed or not isinstance(request,dict) or set(request) not in (required,required|{'memory_kind'}) or
                request.get('task_id')!=self.start['task_id'] or request.get('session_id')!=self.session_id or
                not isinstance(request.get('event_id'),str) or not request['event_id'] or request['event_id'] in self._seen or
                any(not isinstance(request[k],str) for k in ('user_text','memory_text','memory_version','instruction'))):
            raise ValueError('duplicate or foreign Claude solver request')
        payload=native_input_payload(request)
        self.authorize('turn',dict(self.start))
        if (file_hash(self.binary)!=self._binary_hash or file_hash(self.settings)!=self._settings_hash or
                {p.name for p in Path('/').iterdir()}!=self._host_roots):
            raise ValueError('Claude binary or settings changed after admission')
        self._seen.add(request['event_id']);self._turn_id=str(uuid.uuid4());self._messages=[]
        self._result=None;self._initialized=False;lease=None;mcp=None;started=time.monotonic()
        self._mcp_observed=[]
        self._compact_summary_anchor=None
        self._capture_event=request['event_id'];self._delivery_captures=[]
        self.execution_evidence.begin_request(request['event_id'])
        self.execution_evidence.bind_request_turn(request['event_id'], self._turn_id,
            origin='claude_adapter_request_turn')
        if self._trace is not None:
            if self.memory.runtime.publication.digest != request['memory_version']:
                raise ValueError('selected text and execution support use different memory versions')
            self._trace.begin_delivery(request['event_id'], user_text=request['user_text'],deadline=started+self.timeout_s)
        self._normalized('turn/started',turn={'id':self._turn_id})
        prompt=json.dumps(payload,ensure_ascii=False)
        try:
            if self.credential_source is not None:
                lease=self.credential_source(minimum_lifetime_s=self.timeout_s);lease.__enter__()
            env=environment(cli_home=self.home,host_home=self.host,access_token_fd=lease.fd if lease else None,
                            offline_base_url=self.offline_base_url)
            env['PATH']=str(self.binary.parent)+':/usr/bin:/bin'
            if self.benchmark_tools is not None or self._appeal_reviewer is not None:
                # Raise the MCP token limit to the bridge's text bound. The CLI
                # has a separate persistence threshold: validate_observed must
                # still reject substituted previews. Real database output
                # capacity is checked separately before run admission.
                env['MAX_MCP_OUTPUT_TOKENS']=str(2*1024*1024)
                mcp=ClaudeBenchmarkMCP(tools=self.benchmark_tools,native_thread_id=self.native_thread_id,
                    turn_id=self._turn_id,unit_id=request['unit_id'],event_id=request['event_id'],
                    deadline=started+self.timeout_s,
                    review=self._trace.review if self._appeal_reviewer is not None else None).start()
            self._availability=None
            captured=run_native_command(command(binary=self.binary,workspace=self.workspace,cli_home=self.home,
                config=self.config,session_id=self.native_thread_id,text_only=False,resume=bool(self._completed_turns),
                settings_path=self.settings,benchmark_mcp=mcp,hooks_enabled=self._trace is not None),cwd=self.workspace,env=env,prompt=prompt,
                timeout_s=max(.01,self.timeout_s-(time.monotonic()-started)),on_event=self._observe_recorded,
                pass_fds=(lease.fd,) if lease else ())
            if self._availability is not None:raise self._availability
            self.record({'type':'claude_turn_stderr','session_id':self.session_id,
                         'stderr':lease.redact(captured.stderr) if lease else captured.stderr})
            self.record({'type':'claude_turn_capture','session_id':self.session_id,'event_id':request['event_id'],
                         'status':captured.status,'exit_code':captured.exit_code,'wall_s':time.monotonic()-started})
            result=self._result
            if (captured.status!='completed' or captured.exit_code or not self._initialized or result is None or
                    self._pending_tools or not self._messages or not self._messages[-1].strip() or
                    result.get('result')!=self._messages[-1]):
                raise ValueError('Claude turn lacks an actual complete delivery')
            if mcp is not None:
                mcp.assert_healthy()
                missing=mcp.validate_observed(self._mcp_observed)
                if missing:
                    self.record({'type':'native_backend_result_not_delivered','session_id':self.session_id,
                                 'event_id':request['event_id'],'count':missing,'reason':'native_tool_reported_error'})
            if self._bridge is not None:
                self._bridge.assert_healthy()
                self._trace.tools.assert_complete()
            usage=result.get('usage');model_usage=result.get('modelUsage')
            if not isinstance(usage,dict) or not isinstance(model_usage,dict) or set(model_usage)!={self.config.model}:
                raise ValueError('actual model or usage missing')
            counts=[usage.get('input_tokens'),usage.get('output_tokens'),usage.get('cache_read_input_tokens',0),usage.get('cache_creation_input_tokens',0)]
            if any(type(v) is not int or v<0 for v in counts):raise ValueError('invalid native token counts')
            self._usage['inputTokens']+=counts[0]+counts[2]+counts[3]
            self._usage['outputTokens']+=counts[1];self._usage['cachedInputTokens']+=counts[2]
            self._normalized('thread/tokenUsage/updated',tokenUsage={'total':dict(self._usage),'basis':'sum_of_CLI_invocation_usage'})
            self._normalized('turn/completed',turn={'id':self._turn_id,'status':'completed','durationMs':(time.monotonic()-started)*1000})
            text=result['result']
            summary=None
            if self._trace is not None:
                summary=self._trace.guard.delivery_summary()
                if (not summary['completed'] or not self._delivery_captures or
                        self._delivery_captures[0][0]!=summary['before_repair_artifact_hash'] or
                        self._delivery_captures[-1][0]!=summary['final_artifact_hash'] or
                        self.artifacts.load(summary['final_artifact_hash'])['answer']!=text):
                    raise ValueError('Claude delivery lacks matching completed TRACE checks')
                artifact,evidence=self._delivery_captures[-1]
                self.record({'type':'native_repair_summary','session_id':self.session_id,
                    'event_id':request['event_id'],'summary':summary})
            else:
                artifact=self._capture_delivery(text);evidence=self._delivery_captures[-1][1]
            self._last_delivery=(request['event_id'],artifact,evidence)
            self.record({'type':'native_delivery_capture','session_id':self.session_id,
                         'event_id':request['event_id'],'artifact_hash':artifact,'execution_evidence_hash':evidence})
            self._completed_turns+=1
            if summary is not None:
                before=self.artifacts.load(summary['before_repair_artifact_hash'])
                return SolverAnswer(text,artifact,summary['following_candidates'],summary['before_repair_artifact_hash'],
                    before['answer'],json.dumps(self.artifacts.feedback_view(artifact,current_request=request['user_text']),ensure_ascii=False),
                    execution_evidence_hash=evidence,before_repair_execution_evidence_hash=self._delivery_captures[0][1])
            return SolverAnswer(text,artifact,visible_artifacts_json=json.dumps(self.artifacts.feedback_view(artifact,current_request=request['user_text']),ensure_ascii=False),
                                execution_evidence_hash=evidence)
        except Exception as exc:
            self._failed=True
            self.record({'type':'native_turn_failed','session_id':self.session_id,'event_id':request['event_id'],'error_type':type(exc).__name__,
                         'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)[-6:]],
                         'availability_code':exc.code if isinstance(exc,NativeUnavailable) else None,
                         'reset_at':exc.reset_at if isinstance(exc,NativeUnavailable) else None})
            self.close();raise RecoverableCallError(exc.code if isinstance(exc,NativeUnavailable) else 'native_claude_task_turn_failed') from None
        finally:
            if mcp is not None:mcp.close()
            if lease is not None:lease.__exit__(None,None,None)

    def feedback_operations(self,answer,*,event_id):
        if (not isinstance(answer,SolverAnswer) or self._last_delivery!=
                (event_id,answer.artifact_hash,answer.execution_evidence_hash)):
            raise ValueError('feedback requires this task runtime latest actual delivery')
        return json.dumps(self.execution_evidence.feedback_view(answer.execution_evidence_hash,
            artifact_hash=answer.artifact_hash,user_event_id=event_id),ensure_ascii=False)

    def close(self):
        if not self._closed:
            self._closed=True
            try:self.record({'type':'native_task_stderr','session_id':self.session_id,'stderr':''})
            finally:
                try:
                    if self._bridge is not None:self._bridge.close()
                finally:
                    if callable(getattr(self.artifacts,'close',None)):self.artifacts.close()
