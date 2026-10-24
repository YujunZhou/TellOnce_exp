"""Task-owned execution checks for trusted native hook observations.

The native supervisor owns this object across a complete task, serializes hook
requests, and calls begin_delivery at real user-input boundaries. It supplies
the admitted tellonce Observation/Event classes and actual artifact capturers.
The hook transport, false-positive appeal and crash recovery are separate.
"""
from dataclasses import asdict, is_dataclass, replace
from bisect import bisect_right
import json
from pathlib import Path
import queue
import threading

from .provenance import content_hash
from .task_artifacts import ArtifactCapture
from .rule_appeal import AppealInputError, parse_review, validate_cases
from .public_task_contracts import SELECTED_MEMORY_CONTRACT
from .request_budget import request_budget


class NativeRuleGuard:
    # Tellonce's semantic-support transport has its own 180k-character prompt
    # ceiling. Leave room for the published rule, rubric and fixed instructions;
    # the full event history remains in the task journal and is only compacted
    # for this check.
    _SEMANTIC_OBSERVATION_BUDGET = 80_000

    def __init__(self, *, memory, native_session_id: str, workspace: Path,
                 observation_type, event_type, capture_delivery, capture_artifacts, record,
                 require_native_delivery_observation: bool = False, native_response_tracking: bool = False,
                 appeal_review=None, capture_appeal_state=None, cancel_unstarted_tool=None,
                 capture_delivery_context=None, capture_public_context=None):
        if (not isinstance(native_session_id,str) or not native_session_id or not workspace.is_absolute() or
                workspace != workspace.resolve()):
            raise ValueError('an explicit native session and canonical public workspace are required')
        self.memory, self.native_session_id, self.workspace = memory,native_session_id,workspace
        self.observation_type, self.event_type = observation_type,event_type
        self.capture_delivery, self.capture_artifacts, self.record = capture_delivery,capture_artifacts,record
        if (appeal_review is None)!=(capture_appeal_state is None) or (appeal_review is not None and
                (not callable(appeal_review) or not callable(capture_appeal_state))):
            raise ValueError('appeal needs both an admitted reviewer and complete current-state capture')
        self.appeal_review,self.capture_appeal_state=appeal_review,capture_appeal_state
        if cancel_unstarted_tool is not None and not callable(cancel_unstarted_tool):
            raise ValueError('host-owned cancellation must be an explicit callback')
        self.cancel_unstarted_tool = cancel_unstarted_tool
        if capture_delivery_context is not None and not callable(capture_delivery_context):
            raise ValueError('delivery context requires a source-owned callback')
        self.capture_delivery_context = capture_delivery_context
        if capture_public_context is not None and not callable(capture_public_context):
            raise ValueError('public context requires a source-owned callback')
        self.capture_public_context = capture_public_context
        self._handler_lock=threading.RLock()
        self.events, self._tool_ids, self._pending_tools, self._user_events = [],set(),{},set()
        self._pending_tool_actions = {}
        self._sequence, self._open = 0,False
        self._user_event = None
        self._action_scope_selected = False
        self._artifact_scope_selected = False
        self._artifact_verdicts = {}
        self._native_events = queue.Queue(maxsize=128)
        if type(require_native_delivery_observation) is not bool or type(native_response_tracking) is not bool:
            raise ValueError('native delivery synchronization must be explicit')
        self._require_native_delivery = require_native_delivery_observation
        self._native_condition = threading.Condition()
        self._native_messages = {}
        self._track_responses = native_response_tracking
        self._pre_action_denials, self._waived_blocks = {}, []
        self._native_response_ids = set()
        self._current_response_id = None
        self._response_messages = {}

    def observe_model_response(self, response_id, *, messages, native_session_id):
        """Record an actual upstream completion from the native raw event API."""
        if (native_session_id!=self.native_session_id or not isinstance(response_id,str) or not response_id or
                not isinstance(messages,tuple) or any(not isinstance(text,str) for text in messages)):
            raise ValueError('model response requires its native identity and actual message texts')
        with self._native_condition:
            if response_id in self._native_response_ids:
                raise ValueError('duplicate native model response identity')
            self._native_response_ids.add(response_id)
            self._current_response_id=response_id
            for text in messages:
                self._response_messages.setdefault(content_hash(text),[]).append(response_id)
            self._native_condition.notify_all()

    def observe_native_event(self, event, *, native_session_id):
        """Nonblocking stream callback; only the hook worker mutates history."""
        if native_session_id != self.native_session_id or not isinstance(event,dict):
            raise ValueError('native event belongs to another task session')
        if event.get('type') not in {'item.started','item.updated','item.completed'}:
            return
        item = event.get('item')
        if not isinstance(item,dict):
            raise ValueError('native item event lacks its actual item')
        if event.get('type')=='item.completed' and item.get('type')=='agent_message':
            text=item.get('text')
            if not isinstance(text,str):
                raise ValueError('native assistant message lacks its actual text')
            with self._native_condition:
                key=content_hash(text)
                self._native_messages[key]=self._native_messages.get(key,0)+1
                self._native_condition.notify_all()
            return
        if item.get('type') not in {'command_execution','file_change','mcp_tool_call','dynamic_tool_call'}:
            return
        # Keep actual command/exit status even when stdout is too large. The
        # omission is explicit; never treat clipped output as complete evidence.
        captured = json.loads(json.dumps(event,ensure_ascii=False,allow_nan=False))
        output = captured['item'].get('aggregated_output')
        if isinstance(output,str) and len(output.encode()) > 32768:
            captured['item']['aggregated_output'] = {'omitted':True,'sha256':content_hash(output),
                                                     'characters':len(output)}
        if len(json.dumps(captured,ensure_ascii=False).encode()) > 65536:
            captured = {'type':'native_event_omitted','sha256':content_hash(event),'reason':'event_budget'}
        self._native_events.put_nowait(captured)

    def observe_response_text(self, response_id, text, *, native_session_id):
        """Bind a completed streamed text block to its already observed response."""
        with self._native_condition:
            if (native_session_id != self.native_session_id or response_id not in self._native_response_ids or
                    response_id != self._current_response_id or not isinstance(text, str)):
                raise ValueError('text must belong to the current actual native response')
            self._response_messages.setdefault(content_hash(text), []).append(response_id)
            key = content_hash(text)
            self._native_messages[key] = self._native_messages.get(key, 0) + 1
            self._native_condition.notify_all()

    def _drain_native_events(self):
        while True:
            try:
                event = self._native_events.get_nowait()
            except queue.Empty:
                break
            if event.get('type') == 'native_tool_failed':
                tool_id = event['result']['tool_use_id']
                action = {'tool': event['tool_name'], 'arguments': event['tool_input']}
                if event['tool_name'] == 'tellonce_review' and self.appeal_review is not None:
                    if self._appeal_tool_ids.pop(tool_id, None) != content_hash(action):
                        raise ValueError('native review failure does not match its prechecked operation')
                    self._event('native_review_failed', event)
                    continue
                if self._pending_tools.get(tool_id) != content_hash(action):
                    raise ValueError('native failure does not match a pending prechecked operation')
                del self._pending_tools[tool_id]
                self._pending_tool_actions.pop(tool_id, None)
                self._appeal_overrides.clear()
                self._appeal_failures.clear()
                # A failed result is not proof of zero effects. Stop still
                # captures and checks all actual output artifacts.
                self._event('native_tool_failed', {**event, 'partial_effects_possible': True})
            else:
                self._event('native_runtime_observation',event)

    def observe_failed_tool(self, *, native_session_id, tool_name, tool_input, result):
        """Queue an actual native error when the runtime emits no post hook.

        This callback performs no model work and acquires no handler lock: the
        hook worker may be waiting for native stream observations. The adapter
        must only forward failures for tools whose precheck was permitted and
        whose post hook has not already been processed.
        """
        if (native_session_id != self.native_session_id or not isinstance(tool_name, str) or not tool_name or
                not isinstance(tool_input, dict) or not isinstance(result, dict) or
                result.get('type') != 'tool_result' or result.get('is_error') is not True or
                not isinstance(result.get('tool_use_id'), str) or not result['tool_use_id']):
            raise ValueError('actual failed native tool result and matching session required')
        captured = json.loads(json.dumps({'type': 'native_tool_failed', 'tool_name': tool_name,
                                         'tool_input': tool_input, 'result': result}, allow_nan=False))
        if len(json.dumps(captured, ensure_ascii=False).encode()) > 65536:
            # Keep the exact operation binding, but bound the potentially large
            # native error payload without inventing an execution outcome.
            captured['result'] = {'type': 'tool_result', 'tool_use_id': result['tool_use_id'],
                                  'is_error': True, 'content': {'omitted': True, 'sha256': content_hash(result)}}
        if len(json.dumps(captured, ensure_ascii=False).encode()) > 65536:
            raise ValueError('failed native operation identity exceeds observation budget')
        self._native_events.put_nowait(captured)

    def begin_delivery(self, user_event_id: str, *, user_text: str):
        if (self._open or self._pending_tools or not isinstance(user_event_id,str) or not user_event_id or
                user_event_id in self._user_events or not isinstance(user_text,str) or not user_text.strip()):
            raise ValueError('delivery boundary is duplicated or the previous turn is unfinished')
        self._open, self._user_event = True,user_event_id
        self._current_user_text = user_text
        self._user_events.add(user_event_id)
        self._event('user_input', {'event_id': user_event_id, 'text': user_text})
        self._repair_requests, self._observed_repairs, self._awaiting_candidate = 0,0,False
        self._repair_generations,self._generation_failures = set(),{}
        self._candidate_response,self._awaiting_response = None,None
        self._first_draft, self._last_draft = '', ''
        self._unresolved = set()
        self._blocked_actions, self._blocked_deliveries = {},{}
        self._pre_action_denials, self._waived_blocks = {}, []
        self._artifact_omissions = []
        self._uncheckable = set()
        self._selection_incomplete_seen = False
        self._action_scope_selected = False
        self._artifact_scope_selected = False
        self._appeal_attempts,self._appeal_failures,self._appeal_overrides=0,{},{}
        self._appeal_released_checks=set()
        self._appeal_tool_ids={}
        with self._native_condition:
            self._native_messages.clear()
            self._response_messages.clear()
            self._current_response_id=None
        self._scope_texts = {}
        current = getattr(self.memory.runtime,'selection',None)
        self._injected_scope_hashes = {content_hash(current.result.injection)} if current else set()

    def _event(self, kind, value):
        self._sequence += 1
        text = json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)
        captured = self.event_type(self._sequence,kind,text,self.memory.start['task_id'])
        self.events.append(captured)
        self.record({'type':'execution_event','sequence':self._sequence,'kind':kind,
                     'user_event_id':self._user_event,'content':value})

    def _bounded_check_observation(self, observation):
        """Keep semantic checks within the admitted support prompt budget.

        The native journal is authoritative and is not truncated.  When its
        serialized observation is too large, retain a deterministic sample
        from both ends and add an explicit omission event.  The checker can
        therefore return unknown for missing evidence, but can never mistake
        omitted history for a clean observation.
        """
        if not is_dataclass(observation) or not getattr(observation, 'events', ()):
            return observation

        def size(candidate):
            return len(json.dumps(asdict(candidate), ensure_ascii=False,
                                  sort_keys=True, separators=(',', ':')))

        if size(observation) <= self._SEMANTIC_OBSERVATION_BUDGET:
            return observation

        events = tuple(observation.events)
        task_id = observation.task_id
        # The greedy choice below is unchanged. Serialize each real event only
        # once instead of copying/encoding the whole history for every trial.
        event_sizes = {id(event): len(json.dumps(asdict(event), ensure_ascii=False,
                       sort_keys=True, separators=(',', ':'))) for event in events}
        empty_size = size(replace(observation, events=()))

        def omission_ranges(indices):
            sequences = [events[index].sequence for index in indices]
            ranges = []
            for sequence in sequences:
                if ranges and sequence == ranges[-1][1] + 1:
                    ranges[-1][1] = sequence
                else:
                    ranges.append([sequence, sequence])
            return ranges

        def compact(indices):
            selected_events = tuple(events[index] for index in sorted(indices))
            omitted = [index for index in range(len(events)) if index not in indices]
            if not omitted:
                return replace(observation, events=selected_events), omitted
            marker = self.event_type(
                events[omitted[0]].sequence,
                'native_event_history_omitted',
                json.dumps({'omitted': True, 'count': len(omitted),
                            'sequence_ranges': omission_ranges(omitted)},
                           ensure_ascii=False, sort_keys=True, separators=(',', ':')),
                task_id)
            return replace(observation, events=tuple(
                event for event in selected_events if event.sequence < marker.sequence
            ) + (marker,) + tuple(
                event for event in selected_events if event.sequence > marker.sequence
            )), omitted

        selected = set()
        selected_size = 0
        remaining_ranges = omission_ranges(range(len(events)))
        def range_size(pair):
            return 3 + len(str(pair[0])) + len(str(pair[1]))
        remaining_size = sum(map(range_size, remaining_ranges))
        empty_marker = self.event_type(0, 'native_event_history_omitted',
            json.dumps({'omitted': True, 'count': 0, 'sequence_ranges': []},
                       ensure_ascii=False, sort_keys=True, separators=(',', ':')), task_id)
        marker_base_size = len(json.dumps(asdict(empty_marker), ensure_ascii=False,
                                         sort_keys=True, separators=(',', ':')))
        # Prefer both the earliest context and the most recent evidence.  The
        # order is deterministic, while the final tuple remains chronological.
        candidate_order = []
        for left in range((len(events) + 1) // 2):
            candidate_order.append(left)
            candidate_order.append(len(events) - 1 - left)
        for index in candidate_order:
            if index < 0 or index >= len(events) or index in selected:
                continue
            sequence = events[index].sequence
            position = bisect_right([pair[0] for pair in remaining_ranges], sequence) - 1
            left, right = remaining_ranges[position]
            replacement = []
            if left < sequence:
                replacement.append([left, sequence - 1])
            if sequence < right:
                replacement.append([sequence + 1, right])
            candidate_ranges = (remaining_ranges[:position] + replacement +
                                remaining_ranges[position + 1:])
            ranges_size = (remaining_size - range_size([left, right]) +
                           sum(map(range_size, replacement)))
            count = len(selected) + 1
            omitted_count = len(events) - count
            candidate_size = empty_size + selected_size + event_sizes[id(events[index])]
            if omitted_count:
                # Only decimal integers and JSON punctuation vary in the marker.
                # Their lengths remain identical when its content is JSON-escaped.
                candidate_size += (marker_base_size + len(str(candidate_ranges[0][0])) - 1 +
                                   len(str(omitted_count)) - 1 + ranges_size +
                                   max(0, len(candidate_ranges) - 1))
                count += 1
            candidate_size += max(0, count - 1)
            if candidate_size <= self._SEMANTIC_OBSERVATION_BUDGET:
                selected.add(index)
                selected_size += event_sizes[id(events[index])]
                remaining_ranges, remaining_size = candidate_ranges, ranges_size

        compacted, omitted = compact(selected)
        if size(compacted) > self._SEMANTIC_OBSERVATION_BUDGET:
            # A single event may itself exceed the budget. Keep only the
            # explicit omission marker in that case.
            compacted, omitted = compact(set())
        self.record({'type': 'semantic_observation_compacted',
                     'original_event_count': len(events),
                     'retained_event_count': len(compacted.events),
                     'omitted_event_count': len(omitted),
                     'original_observation_digest': getattr(observation, 'digest', None),
                     'check_observation_digest': getattr(compacted, 'digest', None)})
        return compacted

    def _check(self, stage, **evidence):
        self._drain_native_events()
        coverage = self._selection_coverage()
        self._selection_incomplete_seen |= not coverage['complete']
        self._sequence += 1
        task_context = {'public_task': self.memory.runtime.public_task,
                        'current_user_input': self._current_user_text}
        project = getattr(self.memory, 'public_check_context', None)
        if callable(project):
            # Only the task adapter may supply a bounded projection of already
            # visible task history. No private grader or cross-task memory.
            context = project(user_text=self._current_user_text, delivery=evidence.get('delivery'))
            if not isinstance(context,dict) or len(json.dumps(context,ensure_ascii=False)) > 16000:
                raise ValueError('execution context requires a bounded public task projection')
            task_context['public_history_lookup'] = context
        if stage == 'delivery' and self.capture_delivery_context is not None:
            context = self.capture_delivery_context(delivery=evidence.get('delivery'),
                user_event_id=self._user_event, artifact_hash=self._last_draft)
            if not isinstance(context, dict) or len(json.dumps(context, ensure_ascii=False)) > 16000:
                raise ValueError('delivery context requires a bounded public source projection')
            task_context['native_public_dialogue'] = context
        elif stage != 'delivery' and self.capture_public_context is not None:
            # The opening and other public messages already emitted for this user event are evidence for
            # pre_action/artifact checks too (an opening-lines rule is decidable before the first lookup).
            context = self.capture_public_context(user_event_id=self._user_event)
            if not isinstance(context, dict) or len(json.dumps(context, ensure_ascii=False)) > 16000:
                raise ValueError('public context requires a bounded public source projection')
            task_context['native_public_dialogue'] = context
        task = json.dumps(task_context, ensure_ascii=False)
        observation = self.observation_type(task_id=self.memory.start['task_id'],sequence=self._sequence,
            stage=stage,task=task,events=tuple(self.events),**evidence)
        check_observation = self._bounded_check_observation(observation)
        results = tuple(self.memory.check(check_observation))
        if self.appeal_review is not None:
            results=self._apply_appeal(check_observation,results)
        self._last_results = results
        failures, uncertain = [],[]
        for result in results:
            if result.status not in {'pass','fail','not_applicable','unknown','error'}:
                raise ValueError('invalid admitted checker status')
            self.record({'type':'rule_checked','user_event_id':self._user_event,'stage':stage,
                'status':result.status,'rule_digest':result.rule_digest,'support_digest':result.support_digest,
                'observation_digest':result.observation_digest,'reason':result.reason,'attempts':result.attempts,
                'released_by_appeal':(result.rule_digest,result.observation_digest) in self._appeal_released_checks})
            if result.status=='fail' and (result.rule_digest,result.observation_digest) not in self._appeal_released_checks:
                failures.append(result)
            elif result.status in {'unknown','error'}:
                uncertain.append(result)
        return failures,uncertain

    def _appeal_state(self):
        try:
            value=self.capture_appeal_state()
        except OSError:
            self.record({'type':'rule_appeal_unavailable','reason':'state_capture_io_failed',
                         'user_event_id':self._user_event})
            return None
        if value is not None and (not isinstance(value,str) or not value):
            raise ValueError('appeal state must be a complete captured digest or unavailable')
        return value

    @staticmethod
    def _appeal_target(observation):
        return content_hash({key:getattr(observation,key) for key in ('stage','task','action','artifact','delivery')})

    def _apply_appeal(self,observation,results):
        failed=[r for r in results if r.status=='fail']
        if not failed:
            return results
        if not is_dataclass(observation) or any(not is_dataclass(r) for r in failed):
            raise ValueError('appeals require actual immutable observation and checker records')
        state=self._appeal_state()
        target=self._appeal_target(observation)
        for result in results:
            if result.status!='fail':
                continue
            key=(result.rule_digest,result.support_digest,target,state)
            override=self._appeal_overrides.pop(key,None) if state is not None else None
            if override is not None:
                self._appeal_released_checks.add((result.rule_digest,result.observation_digest))
                self.record({'type':'rule_appeal_release','user_event_id':self._user_event,
                    'rule_digest':result.rule_digest,'original_observation_digest':override['observation_digest'],
                    'current_observation_digest':result.observation_digest,'original_status':result.status,
                    'state_digest':state,'candidate_digest':target})
                continue
            if state is None:
                self.record({'type':'rule_appeal_unavailable','reason':'incomplete_state_capture',
                             'rule_digest':result.rule_digest,'user_event_id':self._user_event})
                continue
            publication=self.memory.runtime.publication
            supports=[support for support in publication.supports if support.digest==result.support_digest]
            if len(supports)!=1:
                raise ValueError('disputed check must refer to the pinned published execution support')
            rule=self.memory.describe_rules((result.rule_digest,))[result.rule_digest]
            self._appeal_failures[(result.rule_digest,result.observation_digest)]={
                'case':{'rule_digest':result.rule_digest,'support_digest':result.support_digest,
                    'observation_digest':result.observation_digest,'rule':rule,'support':{
                        'tier':supports[0].tier,'stages':list(supports[0].stages),'check':json.loads(supports[0].check_json)},
                    'observation':asdict(observation),'original_reason':result.reason},
                'state':state,'target':target}
        return results

    def submit_appeal(self,disputes):
        """Review one batch, then authorize only unchanged candidates once."""
        with self._handler_lock:
            if not self._open or self.appeal_review is None:
                raise ValueError('no open delivery with an admitted appeal reviewer')
            if self._appeal_attempts:
                return {'status':'unavailable','reason':'One model appeal for this delivery has already been used.'}
            if self._pending_tools:
                raise AppealInputError('Wait for actual pending tool outcomes before requesting review.')
            if not isinstance(disputes,list) or not 1<=len(disputes)<=32:
                raise AppealInputError('Supply one batch of 1–32 actual disputed checks.')
            state=self._appeal_state()
            cases,records=[],[]
            for dispute in disputes:
                if (not isinstance(dispute,dict) or set(dispute)!={'rule_digest','observation_digest','argument','evidence'} or
                        not isinstance(dispute['rule_digest'],str) or not isinstance(dispute['observation_digest'],str)):
                    raise AppealInputError('Each dispute needs rule_digest, observation_digest, argument and evidence.')
                saved=self._appeal_failures.get((dispute['rule_digest'],dispute['observation_digest']))
                if saved is None:
                    raise AppealInputError('Unknown blocked-check reference. Copy both complete digests exactly '
                        'from the current block message; do not shorten or reconstruct them.')
                if state is None or saved['state']!=state:
                    raise ValueError('disputed check is foreign or its captured task state has changed')
                cases.append({**saved['case'],'argument':dispute['argument'],'evidence':dispute['evidence']})
                records.append(saved)
            validate_cases(cases)
            self._appeal_attempts=1
            context={'phase':'training' if self.memory.start['training'] else 'test',
                'task_id':self.memory.start['task_id'],'session_id':self.memory.start['session_id'],
                'event_id':self._user_event,'role':'rule_appeal',
                'call_id':content_hash([self.memory.start['namespace'],self.memory.start['session_id'],self._user_event,'rule_appeal']),
                'attempt':1}
            self.record({'type':'rule_appeal_started','user_event_id':self._user_event,'context':context,
                         'disputed_checks':[(c['rule_digest'],c['observation_digest']) for c in cases]})
            try:
                reviews=self.appeal_review(cases,context=context)
                reviews=parse_review(json.dumps({'reviews':reviews},allow_nan=False),cases)
            except Exception as exc:
                self.record({'type':'rule_appeal_failed','user_event_id':self._user_event,'error_type':type(exc).__name__})
                return {'status':'unavailable','reason':'Review did not establish a false positive. Existing blocks remain.'}
            if self._appeal_state()!=state:
                self.record({'type':'rule_appeal_stale','user_event_id':self._user_event})
                return {'status':'unavailable','reason':'Task state changed during review. Existing blocks remain.'}
            by_identity={(r['rule_digest'],r['observation_digest']):r for r in reviews}
            for saved in records:
                case=saved['case']
                result=by_identity[(case['rule_digest'],case['observation_digest'])]
                if result['status']=='overturn':
                    self._appeal_overrides[(case['rule_digest'],case['support_digest'],saved['target'],state)]=result
            self.record({'type':'rule_appeal_completed','user_event_id':self._user_event,'reviews':reviews,
                         'state_digest':state,'long_term_memory_updated':False})
            return {'status':'completed','reviews':reviews,
                    'scope':'Confirmed false positives apply once to the unchanged candidate in this delivery.'}

    def _retain_known_blocks(self, cache, identity, failures, results):
        retained=dict(cache.get(identity,{}))
        failed_ids={r.rule_digest for r in failures}
        for result in results:
            if (result.status in {'pass','not_applicable'} or
                    (result.rule_digest,result.observation_digest) in self._appeal_released_checks) and result.rule_digest not in failed_ids:
                retained.pop(result.rule_digest,None)
        # A rule can fail on several artifacts in the same check batch. Keep
        # every observation so the solver receives all actionable reasons.
        newly_failed={}
        for result in failures:
            newly_failed.setdefault(result.rule_digest, {})[result.observation_digest] = result
        retained.update(newly_failed)
        if set(retained)-set(newly_failed):
            self.record({'type':'previous_block_retained','user_event_id':self._user_event,
                         'candidate_identity':identity,'rule_digests':sorted(set(retained)-set(newly_failed))})
        cache[identity]=retained
        return [result for observations in retained.values() for result in observations.values()]

    PRE_ACTION_BLOCK_BUDGET = 2   # distinct denied proposals per rule per user event

    def _waive_stalled_blocks(self, failures, identity):
        """A rule that has already denied PRE_ACTION_BLOCK_BUDGET distinct proposals in this user event is
        not about the operation: the solver changed the operation and the verdict did not, so the unmet
        obligation lies elsewhere (a message already sent, a value only a later step can supply). Denying
        again stalls the task instead of enforcing the rule (2026-10-02 MemoryArena round 9: 182 blocked
        operations, four empty plans). The failure stays recorded and is relayed as a reminder; the delivery
        check judges the response. An identical re-proposal stays blocked, so a prohibited operation is never
        admitted by repetition, and parallel proposals of one response count as one denial."""
        # Parallel proposals of one model response share one denial: the solver has not yet seen the
        # verdict, so they are not a changed operation (budget counts distinct responses as well).
        response = self._candidate_response if self._track_responses and self._candidate_response is not None else identity
        kept, waived = [], []
        for result in failures:
            seen = self._pre_action_denials.setdefault(result.rule_digest, {'identities': [], 'responses': []})
            if (identity in seen['identities'] or len(seen['identities']) < self.PRE_ACTION_BLOCK_BUDGET
                    or len(seen['responses']) < self.PRE_ACTION_BLOCK_BUDGET):
                if identity not in seen['identities']:
                    seen['identities'].append(identity)
                if response not in seen['responses']:
                    seen['responses'].append(response)
                kept.append(result)
            else:
                waived.append(result)
        if waived:
            self.record({'type':'action_block_waived','user_event_id':self._user_event,'candidate_identity':identity,
                         'rule_digests':sorted({r.rule_digest for r in waived}),'reasons':[r.reason for r in waived]})
        self._waived_blocks = waived
        return kept

    @staticmethod
    def _waived_reminder(waived):
        return ('Learned requirements remain unmet but do not concern this operation, which proceeds. '
                'Attend to them in your messages and the final delivery.\n'
                +'\n'.join(dict.fromkeys(r.reason for r in waived))) if waived else ''

    def _candidate(self):
        # This counts a following observable candidate, not proof that a model
        # completed a successful repair. Keep it distinct from repair requests.
        if self._awaiting_candidate:
            if self._track_responses and self._candidate_response==self._awaiting_response:
                return
            self._observed_repairs += 1
            self._awaiting_candidate = False

    def _repair_message(self, failures):
        if self._track_responses:
            merged=self._generation_failures.setdefault(self._candidate_response,{})
            merged.update({(r.rule_digest,r.observation_digest):r for r in failures})
            failures=list(merged.values())
        self._unresolved = {r.rule_digest for r in failures}
        requirements = self.memory.describe_rules(tuple(sorted(self._unresolved)))
        can_review = self.appeal_review is not None and not self._appeal_attempts
        sections, has_reference = [], False
        for digest in sorted(self._unresolved):
            issues = []
            for result in failures:
                if result.rule_digest != digest:
                    continue
                issue = 'Observed issue: ' + result.reason
                if can_review and (digest, result.observation_digest) in self._appeal_failures:
                    # Keep the exact reference next to its own finding. Two
                    # independently ordered lists force the solver to guess
                    # which opaque digest belongs to the issue it disputes.
                    issue += '\nReview reference for this issue: ' + json.dumps({
                        'rule_digest': digest, 'observation_digest': result.observation_digest})
                    has_reference = True
                if issue not in issues:
                    issues.append(issue)
            sections.append(requirements[digest] + '\n' + '\n'.join(issues))
        reasons = '\n\n'.join(sections)
        if has_reference:
            reasons += ('\nIf you have concrete captured counterevidence, you may request the one '
                'tellonce_review for this delivery. Use the reference paired with the specific '
                'issue you dispute; do not pair an earlier issue with a different current check. '
                'Copy both complete digests exactly. Use a short verbatim quotation from the '
                'indicated evidence field and put any paraphrase in argument.')
        same_round=self._track_responses and self._candidate_response in self._repair_generations
        if same_round or self._repair_requests < 3:
            if not same_round:
                self._repair_requests += 1
                self._awaiting_candidate = True
                self._awaiting_response=self._candidate_response
                self._repair_generations.add(self._candidate_response)
            message = (
                'Please address these applicable learned requirements together. '
                'Make the smallest changes that resolve the observed issues and their necessary '
                'consequences. Preserve unrelated accepted content and already satisfied task '
                'requirements. Explicit current instructions and supported corrections still '
                'take precedence; do not preserve a value known to be wrong. '
                'Captured task facts are available evidence, not missing information. '
                'A failed lookup alone does not prove impossibility or authorize erasing them. '
                'Follow the task\'s existing delivery format. This is an internal check, not a new user '
                'request for commentary about the check. After correction or review, resubmit the '
                'requested deliverable, including all parts required by that format. If the candidate '
                'is supported and should remain unchanged, deliver it again; do not replace it with '
                'only a defense of the candidate or a summary of edits.\n'+reasons)
            self.record({'type':'internal_repair_merged' if same_round else 'internal_repair_requested',
                         'user_event_id':self._user_event,'number':self._repair_requests,
                         'native_response_at_feedback':self._candidate_response,'rule_digests':sorted(self._unresolved)})
            return message
        return ('The three requested repair rounds are exhausted. This prohibited action remains blocked. '
                'Complete the safe parts and explain unresolved work.\n'+reasons)

    @staticmethod
    def _reminder(uncertain):
        return ('Some learned requirements could not be checked reliably; attend to them while continuing the task.\n'
                +'\n'.join(dict.fromkeys(r.reason for r in uncertain))) if uncertain else ''

    def _validate(self, payload):
        if (not self._open or not isinstance(payload,dict) or payload.get('session_id')!=self.native_session_id or
                payload.get('cwd')!=str(self.workspace)):
            raise ValueError('hook observation belongs to another or closed task session')
        kind=payload.get('hook_event_name')
        if kind not in {'PreToolUse','PostToolUse','Stop'}:
            raise ValueError('unsupported native hook observation')
        return kind

    def _expand_scope(self, operation):
        selected = self.memory.expand_scope(operation)
        if selected is None:  # Minimal synthetic test adapters do not inject text.
            return
        text = selected.text if hasattr(selected,'text') else selected.result.injection
        if not isinstance(text,str):
            raise ValueError('scope selection must return the actual complete rule text')
        identity = content_hash(text)
        if not text:
            # Nothing to inject: the selection is trivially visible, so later reuse
            # decisions must not treat an empty selection as never injected.
            self._injected_scope_hashes.add(identity)
        if text and identity not in self._injected_scope_hashes and identity not in self._scope_texts:
            if sum(len(value) for value in self._scope_texts.values())+len(text) <= 48000:
                self._scope_texts[identity] = text
            else:
                self.record({'type':'scope_context_omitted','user_event_id':self._user_event,
                             'injection_hash':identity,'reason':'whole_context_budget'})

    def _scope_context(self):
        text = '\n\n'.join(self._scope_texts.values())
        self._injected_scope_hashes.update(self._scope_texts)
        self._scope_texts.clear()
        return ('Applicable learned requirements for the current operation (preserve all conditions and exceptions):\n'
                +SELECTED_MEMORY_CONTRACT+'\n'+text
                if text else '')

    def handle(self, payload):
        with self._handler_lock,request_budget(240):
            return self._handle(payload)

    def _boundary_selection_reusable(self):
        """The user-boundary selection is complete, current and already visible to the solver."""
        runtime = self.memory.runtime
        publication = getattr(runtime, 'publication', None)
        boundary = getattr(runtime, 'selection', None)
        if publication is None or boundary is None:
            return None, None
        result = getattr(boundary, 'result', None)
        digest = getattr(publication, 'digest', None)
        if (result is None or digest is None or getattr(boundary, 'publication_digest', None) != digest
                or getattr(result, 'status', None) != 'ok' or getattr(result, 'omitted_ids', ())
                or getattr(result, 'unexamined_ids', ()) or getattr(result, 'errors', ())
                or not isinstance(getattr(result, 'injection', None), str)
                or content_hash(result.injection) not in self._injected_scope_hashes
                or any(getattr(support, 'stages', None) is None for support in getattr(publication, 'supports', ()))):
            return None, None  # Unknown or incomplete boundary state: select, never reuse.
        return publication, result

    def _pre_action_selection_needed(self):
        """Actions reuse the user-boundary selection; action-stage predicates still run on every action.

        Returns the reuse reason (falsy when a selection is needed). Until 2026-10-03 the first action of
        a user event re-screened the library against the actual operation (one medium-effort selector
        call, ~12.7 s serial per event). Rounds 10/12 showed that screening narrows the boundary
        selection in most events (drops in 68 of 89, adds in 11) and adds only reminder text the delivery
        re-selection recovers anyway, so the boundary selection is reused whenever it is complete and
        already injected; an incomplete selection (omitted/unexamined/errors/not injected/new publication)
        keeps screening against the action."""
        publication, result = self._boundary_selection_reusable()
        if publication is None:
            return ''
        if self._action_scope_selected:
            return 'first_action_of_user_event_already_screened'
        return 'boundary_selection_reused_for_actions'

    def _artifact_selection_needed(self):
        """Screen the first public artifact of a user event; later artifacts reuse that screening.

        Artifact predicates still run on every captured artifact against the screened
        selection, and the delivery boundary re-selects over the whole library. The
        2026-10-02 ClawArena pilot (v26c) spent 630 artifact-scope selections on 120 user
        events (median 5, max 54 per event; 48% of all auxiliary wall time): 586 chose a
        subset of the boundary selection, 207 chose nothing, and no compiled rule had an
        artifact stage, so each re-selection could only repeat already visible text.
        An incomplete selection (omitted/unexamined/errors/not injected) keeps re-screening.

        Returns the reuse reason (falsy when a selection is needed)."""
        publication, _ = self._boundary_selection_reusable()
        if publication is None:
            return ''
        if self._artifact_scope_selected:
            return 'first_artifact_of_user_event_already_screened'
        return ''

    def _delivery_selection_needed(self):
        """Delivery re-selection only matters when a delivery/artifact predicate can run."""
        publication, _ = self._boundary_selection_reusable()
        if publication is None:
            return ''
        if any(stage in support.stages for support in publication.supports for stage in ('delivery', 'artifact')):
            return ''
        # Nothing can be checked at delivery, and text injected at Stop is never
        # delivered unless a repair round follows (scope_context_not_delivered).
        return 'no_delivery_or_artifact_stage_checks'

    def _handle(self,payload):
        kind=self._validate(payload)
        self._drain_native_events()
        if kind=='Stop':
            return self._stop(payload)
        if self._track_responses:
            with self._native_condition:
                if not self._native_condition.wait_for(lambda:self._current_response_id is not None,timeout=3):
                    raise RuntimeError('native model response has not been observed before its tool feedback')
                self._candidate_response=self._current_response_id
        tool_id,tool,arguments=payload.get('tool_use_id'),payload.get('tool_name'),payload.get('tool_input')
        if any(not isinstance(v,str) or not v for v in (tool_id,tool)) or not isinstance(arguments,dict):
            raise ValueError('native tool identity and actual arguments are required')
        action={'tool':tool,'arguments':arguments}
        serialized=json.dumps(action,ensure_ascii=False,sort_keys=True,allow_nan=False)
        if tool=='tellonce_review' and self.appeal_review is not None:
            # This exact host-defined read-only tool must not be blocked by the
            # check whose verdict it contests. No executable command is exempt.
            if kind=='PreToolUse':
                if tool_id in self._tool_ids:
                    raise ValueError('duplicate native review tool operation')
                self._tool_ids.add(tool_id)
                self._appeal_tool_ids[tool_id]=content_hash(action)
            elif self._appeal_tool_ids.pop(tool_id,None)!=content_hash(action):
                raise ValueError('unknown or changed native review tool completion')
            return {}
        if kind=='PreToolUse':
            if tool_id in self._tool_ids:
                raise ValueError('duplicate proposed native operation')
            self._tool_ids.add(tool_id)
            self._candidate()
            reuse_reason = self._pre_action_selection_needed()
            if not reuse_reason:
                self._expand_scope('Proposed native operation:\n'+serialized)
                self._action_scope_selected = True
            else:
                self.record({'type': 'scope_selection_reused', 'user_event_id': self._user_event,
                             'reason': reuse_reason, 'stage': 'pre_action',
                             'publication_digest': self.memory.runtime.publication.digest})
            failures,uncertain=self._check('pre_action',action=serialized)
            failures=self._retain_known_blocks(self._blocked_actions,content_hash(action),failures,self._last_results)
            failures=self._waive_stalled_blocks(failures,content_hash(action))
            self._event('action_blocked' if failures else 'action_proposed',{'tool_use_id':tool_id,'action':action})
            if failures:
                return {'hookSpecificOutput':{'hookEventName':kind,'permissionDecision':'deny',
                    'permissionDecisionReason':self._repair_message(failures)+'\n'+self._scope_context()}}
            self._pending_tools[tool_id]=content_hash(action)
            self._pending_tool_actions[tool_id]=json.loads(serialized)
            reminder='\n'.join(text for text in (self._scope_context(),self._reminder(uncertain),self._waived_reminder(self._waived_blocks)) if text)
            # Compliance does not grant permissions or override the sandbox.
            return {'hookSpecificOutput':{'hookEventName':kind,'additionalContext':reminder}} if reminder else {}
        if self._pending_tools.get(tool_id)!=content_hash(action):
            raise ValueError('cannot invent completion of a blocked, unknown or changed operation')
        del self._pending_tools[tool_id]
        self._pending_tool_actions.pop(tool_id, None)
        self._appeal_overrides.clear()
        self._appeal_failures.clear()
        response=payload.get('tool_response')
        response_text=json.dumps(response,ensure_ascii=False,sort_keys=True,allow_nan=False)
        if len(response_text)>65536:
            response={'omitted':True,'sha256':content_hash(response),'characters':len(response_text)}
        self._event('tool_completed',{'tool_use_id':tool_id,'action':action,'tool_response':response,
                                     'success_established_by_hook':False})
        failures,uncertain=self._artifacts(payload)
        message=self._repair_message(failures) if failures else self._reminder(uncertain)
        message='\n'.join(text for text in (self._scope_context(),message) if text)
        if self._artifact_omissions:
            message += '\nSome task artifacts could not be inspected; their compliance remains unknown.'
        return {'hookSpecificOutput':{'hookEventName':kind,'additionalContext':message}} if message else {}

    def _artifacts(self,payload):
        artifacts=self.capture_artifacts(payload)
        self._artifact_omissions = []
        deleted=()
        if isinstance(artifacts,ArtifactCapture):
            self._artifact_omissions = list(artifacts.omissions)
            deleted=artifacts.deleted
            self.record({'type':'artifact_capture_coverage','user_event_id':self._user_event,
                         'complete':artifacts.complete,'omissions':self._artifact_omissions,
                         'deleted':list(artifacts.deleted)})
            artifacts=artifacts.texts
        if not isinstance(artifacts,dict):
            raise ValueError('artifact capture must return actual named text contents')
        failures,uncertain=[],[]
        self._artifact_results=[]
        if len(set(deleted))!=len(deleted) or set(deleted).intersection(artifacts):
            raise ValueError('artifact cannot be both deleted and present')
        captures=[(relative,text,'text') for relative,text in artifacts.items()]
        captures.extend((relative,None,'deleted') for relative in deleted)
        for relative,text,kind in captures:
            path=Path(relative)
            if ((kind=='text' and not isinstance(text,str)) or not relative or path.is_absolute() or '..' in path.parts):
                raise ValueError('artifact evidence must belong to the public task workspace')
            if kind=='deleted':
                self._event('artifact_deleted',{'path':relative})
            else:
                self._event('artifact_captured',{'path':relative,'content_hash':content_hash(text)})
            # An artifact whose content has not changed since it was checked under this publication is not a product
            # of the operation that just completed. 2026-10-05 ClawArena round 25 (v46c): a language card compiled with
            # an artifact stage ran 1,970 artifact checks on 2,256 captures of 50 distinct (path, content) pairs —
            # every earlier file was re-read after every tool call (18.6 M prompt tokens against 4.1 M the round
            # before). Within the same user event the stored verdicts keep counting (the repair message and the
            # delivery boundary still see an unrepaired failure); a file left untouched since an earlier user event
            # is not rechecked against the new request.
            verdict_key=(getattr(getattr(self.memory.runtime,'publication',None),'digest',None),relative,kind,
                         content_hash(text) if kind=='text' else None)
            cached=self._artifact_verdicts.get(verdict_key)
            if cached is not None:
                owner,failed,unknown,results=cached
                self.record({'type':'artifact_check_reused','user_event_id':self._user_event,'path':relative,
                             'checked_in_user_event':owner,'results':len(results),
                             'publication_digest':verdict_key[0]})
                if owner==self._user_event:
                    self._artifact_results.extend(results)
                    failures.extend(failed)
                    uncertain.extend(unknown)
                continue
            reuse_reason = self._artifact_selection_needed()
            if not reuse_reason:
                self._expand_scope('Inspect the '+('deleted' if kind=='deleted' else 'updated')+' public artifact '+relative+'.')
                self._artifact_scope_selected = True
            else:
                self.record({'type': 'scope_selection_reused', 'user_event_id': self._user_event,
                             'reason': reuse_reason, 'stage': 'artifact', 'path': relative,
                             'publication_digest': self.memory.runtime.publication.digest})
            # This is the artifact schema promised to the compiler. Supplying
            # only content would make even a valid file-name predicate unknown.
            captured=json.dumps({'path':relative,'content':text,'kind':kind},
                                ensure_ascii=False,sort_keys=True,allow_nan=False)
            failed,unknown=self._check('artifact',artifact=captured)
            self._artifact_verdicts[verdict_key]=(self._user_event,tuple(failed),tuple(unknown),tuple(self._last_results))
            self._artifact_results.extend(self._last_results)
            failures.extend(failed)
            uncertain.extend(unknown)
        return failures,uncertain

    def _stop(self,payload):
        if self.cancel_unstarted_tool is not None:
            for tool_id, action in list(self._pending_tool_actions.items()):
                if self.cancel_unstarted_tool(tool_use_id=tool_id, tool_name=action['tool'],
                                              tool_input=action['arguments']) is True:
                    del self._pending_tools[tool_id]
                    del self._pending_tool_actions[tool_id]
                    self._event('native_tool_cancelled_before_dispatch',{
                        'tool_use_id':tool_id, 'action':action,
                        'reason':'Host fenced unstarted operation at delivery boundary.',
                        'executed':False})
        if self._pending_tools:
            raise ValueError('cannot deliver while observed operations have no recorded result')
        text=payload.get('last_assistant_message')
        if not isinstance(text,str):
            raise ValueError('delivery checks require actual native assistant text')
        if self._track_responses:
            with self._native_condition:
                identity=content_hash(text)
                if not self._native_condition.wait_for(lambda:bool(self._response_messages.get(identity)),timeout=3):
                    raise RuntimeError('native completion identity for this actual draft is missing')
                self._candidate_response=self._response_messages[identity].pop(0)
        if self._require_native_delivery:
            with self._native_condition:
                identity=content_hash(text)
                if not self._native_condition.wait_for(lambda:self._native_messages.get(identity,0)>0,timeout=3):
                    raise RuntimeError('Stop arrived without its matching native delivery observation')
                self._native_messages[identity]-=1
        self._candidate()
        artifact_hash=self.capture_delivery(text)
        if not isinstance(artifact_hash,str) or not artifact_hash:
            raise ValueError('capture the draft before checking or requesting repair')
        self._last_draft=artifact_hash
        if not self._first_draft:
            self._first_draft=artifact_hash
        self._event('delivery_prepared',{'artifact_hash':artifact_hash})
        artifact_failures,artifact_uncertain=self._artifacts(payload)
        reuse_reason=self._delivery_selection_needed()
        if not reuse_reason:
            self._expand_scope('Deliver the answer and the completed task artifacts.')
        else:
            self.record({'type':'scope_selection_reused','user_event_id':self._user_event,
                         'reason':reuse_reason,'stage':'delivery',
                         'publication_digest':self.memory.runtime.publication.digest})
        failures,uncertain=self._check('delivery',delivery=text)
        failures=self._retain_known_blocks(self._blocked_deliveries,artifact_hash,
            [*artifact_failures,*failures],[*self._artifact_results,*self._last_results])
        uncertain=[*artifact_uncertain,*uncertain]
        self._uncheckable = {r.rule_digest for r in uncertain}
        self._event('delivery_candidate',{'artifact_hash':artifact_hash})
        self._unresolved={r.rule_digest for r in failures}
        pending_same_response = (self._track_responses and self._awaiting_candidate and
                                 self._candidate_response in self._repair_generations)
        if failures and (self._repair_requests < 3 or pending_same_response):
            return {'decision':'block','reason':self._repair_message(failures)+'\n'+self._scope_context()}
        if self._scope_texts:
            self.record({'type':'scope_context_not_delivered','user_event_id':self._user_event,
                         'injection_hashes':list(self._scope_texts),'reason':'no_further_generation_before_delivery'})
        self._open=False
        self.record({'type':'delivery_checks_completed','user_event_id':self._user_event,
                     'summary':self.delivery_summary(),'uncheckable_rule_digests':[r.rule_digest for r in uncertain]})
        # Budget exhaustion retains the actual last answer and its violations.
        # It never permits a separately prohibited operation to execute.
        return {}

    def _selection_coverage(self):
        boundary = getattr(self.memory.runtime, 'selection', None)
        if boundary is None:
            return {'status': 'unavailable', 'complete': False}
        result = boundary.result
        missing = {name: list(getattr(result, name, ()))
                   for name in ('failed_ids', 'omitted_ids', 'unexamined_ids')}
        return {'status': result.status,
                'complete': result.status == 'ok' and not any(missing.values()) and not result.errors,
                **missing}

    def delivery_summary(self):
        if self._user_event is None:
            raise ValueError('no delivery boundary exists')
        coverage = self._selection_coverage()
        boundary = getattr(self.memory.runtime, 'selection', None)
        selected = set(boundary.result.selected_ids) if boundary is not None else set()
        publication = getattr(self.memory.runtime, 'publication', None)
        support_tiers = ({r.atomic_id: s.tier for r,s in zip(publication.snapshot.rules, publication.supports)
                          if r.atomic_id in selected} if publication is not None else {})
        non_executable = sorted(rule for rule in selected
                                if support_tiers.get(rule) not in {'semantic', 'deterministic'})
        return {'repair_requests':self._repair_requests,'following_candidates':self._observed_repairs,
                'appeal_available':self.appeal_review is not None,'appeal_attempts':self._appeal_attempts,
                'repair_round_tracking':'native_response' if self._track_responses else 'hook_candidate_only',
                'before_repair_artifact_hash':self._first_draft,'final_artifact_hash':self._last_draft,
                'artifact_capture_complete':not self._artifact_omissions,
                'artifact_omissions':list(self._artifact_omissions),
                'selection_coverage': coverage,
                'selection_incomplete_seen': self._selection_incomplete_seen,
                'selected_support_tiers': support_tiers,
                'non_executable_selected_ids': non_executable,
                'uncheckable_rule_digests': sorted(self._uncheckable),
                'check_coverage_complete': (not self._open and coverage['complete']
                    and not self._selection_incomplete_seen and not self._artifact_omissions
                    and not self._uncheckable and not non_executable),
                # completed denotes delivery lifecycle completion, not a pass.
                'unresolved_rule_digests':sorted(self._unresolved),'completed':not self._open}
