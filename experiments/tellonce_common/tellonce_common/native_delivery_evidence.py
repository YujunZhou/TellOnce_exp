"""Bounded native operation evidence frozen at each actual delivery candidate.

This is private evaluation evidence, not Tellonce admission or learned memory.
Tool operations and an optional public opening projection are captured; model
reasoning, memory and rule checks are not included. Missing operation details
are explicit, never fabricated as pass. Openings seal before internal repairs.
"""
import json
from pathlib import Path
import threading

from .provenance import canonical_json_bytes, content_hash
from .operation_scope import RequestScopeState, validate_request_scope


FIELDS = {
    'claudeToolCall': {'id','type','tool','arguments','result','isError','nativeToolResult','status'},
    'commandExecution': {'id','type','command','cwd','status','aggregatedOutput','exitCode'},
    'fileChange': {'id','type','changes','status'},
    'mcpToolCall': {'id','type','server','tool','arguments','result','error','status'},
    'dynamicToolCall': {'id','type','namespace','tool','arguments','contentItems','status','success','durationMs'},
}


def internal_memory_tool(item):
    name = str(item.get('tool', '')).split('__')[-1].casefold()
    return name == 'tellonce_review' or name.startswith('tellonce_')


class NativeDeliveryEvidence:
    def __init__(self, *, directory: Path, task_id: str, session_id: str, native_thread_id: str,
                 total_bytes=96*1024, item_bytes=16*1024, benchmark_tools=(), resume=None):
        if (not isinstance(directory,Path) or not directory.is_absolute() or directory!=directory.resolve() or
                (directory.exists() and resume is None) or any(not isinstance(value,str) or not value for value in
                                         (task_id,session_id,native_thread_id)) or
                type(total_bytes) is not int or type(item_bytes) is not int or
                not 1024 <= item_bytes <= total_bytes <= 512*1024):
            raise ValueError('fresh private operation evidence and explicit bounded task identities required')
        if (not isinstance(benchmark_tools, tuple) or len(set(benchmark_tools)) != len(benchmark_tools)
                or any(not isinstance(name, str) or not name or name == 'tellonce_review' for name in benchmark_tools)):
            raise ValueError('explicit benchmark tools required; internal memory tools are excluded')
        directory.mkdir(mode=0o700,exist_ok=resume is not None)
        self.directory,self.task_id,self.session_id,self.native_thread_id = directory,task_id,session_id,native_thread_id
        self.total_bytes,self.item_bytes = total_bytes,item_bytes
        self._events,self._pending,self._seen,self._completed = [],set(),set(),set()
        self._used,self._omitted,self._sequence = 0,0,0
        self._lock = threading.Lock()
        self._public_dialogue = None
        self._request_scope = RequestScopeState()
        self._benchmark_tools = frozenset(benchmark_tools)
        if resume is not None:
            from .round_recovery import restore_evidence
            restore_evidence(self,resume)

    def begin_request(self, user_event_id):
        from .public_dialogue import PublicDialogue
        dialogue = PublicDialogue(user_event_id)
        with self._lock:
            self._request_scope.begin(user_event_id)
            self._public_dialogue = dialogue

    def bind_request_turn(self, user_event_id, native_turn_id, *, origin):
        """Bind only the actual owner response; never derive a turn from tools."""
        with self._lock:
            self._request_scope.bind(user_event_id, native_turn_id, origin=origin)

    def observe_public_text(self, text, *, commentary=False):
        with self._lock:
            if self._public_dialogue is not None:
                self._public_dialogue.observe_text(text, commentary=commentary)

    def public_dialogue_so_far(self, user_event_id):
        """Bounded projection of the current request's public dialogue BEFORE the delivery capture, for checks
        that run at pre_action/artifact stages (2026-10-01: an opening-lines rule was judged violated at every first
        lookup because the opening was invisible to the pre_action observation). Pending preamble text (not yet
        confirmed by a following tool call) is included and marked provisional; nothing is sealed or mutated."""
        with self._lock:
            dialogue = self._public_dialogue
            if dialogue is None or dialogue.event_id != user_event_id:
                raise ValueError('public dialogue projection requires the current user event')
            messages = [dict(m) for m in dialogue.messages]
            provisional = [{'sequence': len(messages) + i + 1, 'text': t, 'before_first_tool': not dialogue.tool_seen,
                            'provisional': True} for i, t in enumerate(dialogue.pending)]
            return {'schema': 1, 'event_id': dialogue.event_id, 'messages': messages + provisional,
                    'coverage': {'complete': dialogue.omitted == 0 and dialogue.pending_omitted == 0,
                                 'omitted_messages': dialogue.omitted + dialogue.pending_omitted,
                                 'provisional_messages': len(provisional), 'sealed': dialogue.sealed}}

    def observe(self, event, *, public_dialogue=True):
        if type(public_dialogue) is not bool:
            raise ValueError('public dialogue observation requires an explicit boolean')
        method, params = event.get('method'), event.get('params',{})
        if method not in {'item/started','item/completed'}:
            return
        item = params.get('item',{})
        if (method == 'item/completed' and item.get('type') == 'agentMessage'
                and params.get('threadId') == self.native_thread_id):
            # Only explicit public text, never rawResponseItem or reasoning.
            if public_dialogue and item.get('phase') != 'final_answer':
                self.observe_public_text(item.get('text'), commentary=item.get('phase') == 'commentary')
            return
        if item.get('type') not in FIELDS:
            return
        if internal_memory_tool(item):
            return
        if item.get('type') == 'dynamicToolCall':
            if item.get('tool') not in self._benchmark_tools:
                return
            if item.get('namespace') is not None:
                raise ValueError('benchmark operation used an unadmitted native namespace')
        if (params.get('threadId') != self.native_thread_id or not isinstance(params.get('turnId'),str) or
                not params['turnId'] or not isinstance(item.get('id'),str) or not item['id']):
            raise ValueError('operation evidence belongs to a foreign or incomplete native event')
        key = (params['turnId'],item['id'])
        state = method.split('/')[1]
        projected = {'native_turn_id':params['turnId'], 'event':state,
                     'item':{field:item[field] for field in FIELDS[item['type']] if field in item}}
        if item['type'] == 'claudeToolCall' and 'result' in item and 'nativeToolResult' in item:
            # The original provider event remains in claude_protocol_event.
            # This model-facing projection need not pay for the same bytes twice.
            native, result = item['nativeToolResult'], item['result']
            if canonical_json_bytes(native) == canonical_json_bytes(result):
                del projected['item']['nativeToolResult']
            elif (isinstance(native, dict) and isinstance(result, str) and
                  isinstance(native.get('stdout'), str) and native['stdout'] == result):
                projected['item']['nativeToolResult'] = {k: v for k, v in native.items() if k != 'stdout'}
        encoded = canonical_json_bytes(projected)
        with self._lock:
            if (public_dialogue and self._public_dialogue is not None and not self._public_dialogue.sealed
                    and (state == 'started' or key not in self._seen)):
                self._public_dialogue.observe_tool()
                if state != 'started':
                    self._public_dialogue.omitted += 1
            if state == 'started':
                if key in self._seen:
                    raise ValueError('native operation start was duplicated')
                self._seen.add(key)
                self._pending.add(key)
            else:
                if key in self._completed:
                    raise ValueError('native operation completion was duplicated')
                if key not in self._seen:
                    # Some native providers may omit start notifications. The
                    # completed operation is still evidence, but ordering of its
                    # start cannot be claimed as observed.
                    self._omitted += 1
                    self._seen.add(key)
                self._pending.discard(key)
                self._completed.add(key)
            self._sequence += 1
            if len(encoded)>self.item_bytes:
                self._omitted += 1
                return
            # A task may contain many rounds. Keep the latest actual operations
            # instead of letting early results permanently exhaust the budget.
            # Previously captured receipts are immutable and remain loadable.
            while self._events and self._used+len(encoded)>self.total_bytes:
                old=self._events.pop(0)
                self._used-=len(canonical_json_bytes({k:v for k,v in old.items() if k!='sequence'}))
                self._omitted+=1
            self._events.append({'sequence':self._sequence, **json.loads(encoded)})
            self._used += len(encoded)

    def capture(self, artifact_hash, *, user_event_id):
        if any(not isinstance(value,str) or not value for value in (artifact_hash,user_event_id)):
            raise ValueError('operation evidence must bind the actual artifact and user input')
        with self._lock:
            value = {'schema':1, 'task_id':self.task_id, 'session_id':self.session_id,
                'native_thread_id':self.native_thread_id, 'user_event_id':user_event_id,
                'artifact_hash':artifact_hash, 'events':list(self._events),
                'coverage':{'complete':not self._omitted and not self._pending,
                            'omitted_events':self._omitted,'pending_operations':len(self._pending)}}
            scope = self._request_scope.freeze(user_event_id, self._events)
            if scope is not None:
                value['request_scope'] = scope
            if self._public_dialogue is not None:
                if self._public_dialogue.event_id != user_event_id:
                    raise ValueError('public opening belongs to another user request')
                value['public_dialogue'] = self._public_dialogue.capture()
            encoded = canonical_json_bytes(value)
        digest = content_hash(value)
        target = self.directory/(digest+'.json')
        if target.exists():
            if target.read_bytes()!=encoded:
                raise ValueError('private operation evidence was changed')
        else:
            with target.open('xb') as stream:
                stream.write(encoded)
        return digest

    def load(self, digest, *, artifact_hash, user_event_id):
        if (not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('operation evidence requires a content hash')
        value = json.loads((self.directory/(digest+'.json')).read_text())
        if (content_hash(value)!=digest or value.get('schema')!=1 or
                value.get('task_id')!=self.task_id or value.get('session_id')!=self.session_id or
                value.get('native_thread_id')!=self.native_thread_id or
                value.get('artifact_hash')!=artifact_hash or value.get('user_event_id')!=user_event_id):
            raise ValueError('operation evidence differs from its captured delivery')
        if 'request_scope' in value:
            validate_request_scope(value['request_scope'], event_id=user_event_id, events=value.get('events'))
        return value

    def feedback_view(self, digest, *, artifact_hash, user_event_id):
        """Only the operations already captured with this actual delivery."""
        value = self.load(digest, artifact_hash=artifact_hash, user_event_id=user_event_id)
        result = {'schema':1, 'task_id':self.task_id, 'session_id':self.session_id,
                'event_id':user_event_id, 'artifact_hash':artifact_hash, 'source_hash':digest,
                'events':value['events'], 'coverage':value['coverage']}
        if 'request_scope' in value:
            result['request_scope'] = value['request_scope']
        if 'public_dialogue' in value:
            result['public_dialogue'] = value['public_dialogue']
        return result
