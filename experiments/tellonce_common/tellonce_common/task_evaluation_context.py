"""Private, delivery-time transcript projection for independent evaluation."""
from dataclasses import replace
import json

from .interaction_loop import SolverAnswer
from .preference_judge import prepare_preference_input
from .provenance import content_hash
from .role_inputs import PublicUnit, RunIdentity
from .test_protocol import INTERACTIVE, PLAN_A, validate_protocol


class TaskEvaluationContext:
    def __init__(self, *, identity: RunIdentity, units: tuple[PublicUnit,...]):
        if (not isinstance(identity,RunIdentity) or not isinstance(units,tuple) or not units or
                any(not isinstance(unit,PublicUnit) or unit.instance.benchmark!=identity.benchmark or
                    unit.instance.profile_id!=identity.user_id or
                    unit.independent_task_id!=units[0].independent_task_id for unit in units) or
                len({unit.instance.common_instance_id for unit in units})!=len(units)):
            raise ValueError('one experiment-owned complete task is required for private evaluation')
        self.identity,self.units=identity,{unit.instance.common_instance_id:unit for unit in units}
        self.task_id=units[0].independent_task_id
        self.session_id=None
        self.runtime=None
        self._history,self._corrections=[],[]
        self._inputs,self._deliveries,self._indices={},{},{}
        self._pending=None
        self._position=-1
        self._events=[]
        self._frozen=False
        self.test_protocol=INTERACTIVE

    def observe(self, event):
        if self._frozen:
            raise ValueError('a frozen evaluation transcript cannot observe new interactions')
        if event.get('type') not in {'task_started','user_input','answer_delivered'}:
            return
        if event.get('namespace')!=self.identity.namespace or event.get('task_id')!=self.task_id:
            raise ValueError('evaluation transcript crossed an experiment or task boundary')
        if event['type']=='task_started':
            if (self.session_id is not None or event.get('unit_ids')!=list(self.units) or
                    not isinstance(event.get('session_id'),str) or not event['session_id']):
                raise ValueError('evaluation transcript requires one fresh complete task session')
            self.session_id=event['session_id']
            self.test_protocol=validate_protocol(event.get('test_protocol', INTERACTIVE),
                                                 benchmark=self.identity.benchmark)
            self._events.append({key:json.loads(json.dumps(event[key])) for key in
                                 ('type','namespace','task_id','session_id','unit_ids')})
            if 'test_protocol' in event:
                self._events[-1]['test_protocol']=self.test_protocol
            return
        if self.session_id is None or event.get('session_id')!=self.session_id:
            raise ValueError('evaluation transcript belongs to another native session')
        unit_id,event_id=event.get('unit_id'),event.get('event_id')
        if unit_id not in self.units or not isinstance(event_id,str) or not event_id:
            raise ValueError('evaluation transcript lacks its actual scoring unit and user event')
        if event['type']=='user_input':
            index=event.get('delivery_index')
            if self.test_protocol==PLAN_A and index!=0:
                raise ValueError('plan A evaluation cannot contain added user corrections or clarification')
            if (type(index) is not int or index not in {0,1,2} or index!=self._indices.get(unit_id,-1)+1 or
                    self._pending is not None or
                    event_id in self._inputs or not isinstance(event.get('text'),str) or not event['text'].strip() or
                    (index==0 and event['text']!=self.units[unit_id].instruction) or
                    event_id!=content_hash([self.identity.namespace,self.session_id,unit_id,index])):
                raise ValueError('evaluation user message differs from the real delivery boundary')
            if index==0:
                if self._position+1>=len(self.units) or list(self.units)[self._position+1]!=unit_id:
                    raise ValueError('evaluation tasks must follow the complete native round order')
                self._position+=1
            elif list(self.units)[self._position]!=unit_id:
                raise ValueError('a prior round cannot receive a correction after the next round starts')
            self._indices[unit_id]=index
            self._inputs[event_id]=unit_id
            self._pending=event_id
            self._history.append({'role':'user','text':event['text']})
            if index:
                self._corrections.append({'event_id':event_id,'text':event['text']})
            self._events.append({key:event[key] for key in
                ('type','namespace','task_id','session_id','unit_id','event_id','delivery_index','text')})
            return
        if self._inputs.get(event_id)!=unit_id or event_id in self._deliveries or self._pending!=event_id:
            raise ValueError('delivery lacks its preceding user message or was duplicated')
        fields=SolverAnswer.__dataclass_fields__
        answer=SolverAnswer(**{key:event[key] for key in fields})
        # Freeze before appending this answer. Earlier candidates must never
        # see the final answer or a correction that came after their delivery.
        self._deliveries[event_id]=(unit_id,answer,
            tuple(dict(row) for row in self._history),tuple(dict(row) for row in self._corrections))
        self._history.append({'role':'assistant','text':answer.text})
        self._pending=None
        self._events.append({key:event[key] for key in
            ('type','namespace','task_id','session_id','unit_id','event_id',*fields)})

    def bind_runtime(self, runtime):
        from .codex_task_session import CodexTaskSession
        from .claude_task_session import ClaudeTaskSession
        if (not isinstance(runtime,(CodexTaskSession,ClaudeTaskSession)) or self.runtime is not None or self.session_id is None or
                runtime.start['namespace']!=self.identity.namespace or runtime.start['task_id']!=self.task_id or
                runtime.start['session_id']!=self.session_id):
            raise ValueError('evaluation evidence must come from this exact experiment-owned native task')
        self.runtime=runtime

    def preference_input(self, unit_id, answer, *, profile, context, public_materials_json='{}'):
        if (self.runtime is None or not isinstance(context,dict) or context.get('session_id')!=self.session_id or
                context.get('task_id')!=self.task_id or context.get('event_id') not in self._deliveries):
            raise ValueError('private scoring requires a captured delivery from the bound native runtime')
        actual_unit,actual,history,corrections=self._deliveries[context['event_id']]
        if unit_id!=actual_unit:
            raise ValueError('captured delivery belongs to another scoring unit')
        if context.get('role')=='preference_score_before_repair':
            expected=SolverAnswer(actual.before_repair_text,actual.before_repair_artifact_hash,
                                 execution_evidence_hash=actual.before_repair_execution_evidence_hash)
        elif context.get('role')=='preference_score':
            expected=actual
        else:
            raise ValueError('private preference evidence cannot be requested as another role')
        if answer!=expected:
            raise ValueError('requested answer differs from the actual delivery or pre-repair candidate')
        artifacts=self.runtime.artifacts
        if artifacts.load(answer.artifact_hash)['answer']!=answer.text:
            raise ValueError('captured public artifacts do not match the delivered text')
        delivery_only=self.test_protocol==PLAN_A
        may_profile = json.loads(profile.profile_json).get('kind', '').startswith('may-')
        captured = ({} if delivery_only and not may_profile else
            self.runtime.execution_evidence.load(answer.execution_evidence_hash,
                artifact_hash=answer.artifact_hash,user_event_id=context['event_id']))
        operations = None if delivery_only else captured
        # In particular, a pre-repair SolverAnswer only carries a file identity;
        # reconstruct its public output view from that earlier frozen snapshot.
        view=(artifacts.feedback_view(answer.artifact_hash,current_request='\n'.join(
            (self.units[unit_id].instruction,answer.text))) if delivery_only else
            artifacts.feedback_view(answer.artifact_hash))
        projected=replace(answer,visible_artifacts_json=json.dumps(view,ensure_ascii=False,allow_nan=False))
        return prepare_preference_input(self.units[unit_id],self.identity,profile,projected,context=context,
                                        visible_history=history,corrections=corrections,operations=operations,
                                        public_materials_json=public_materials_json,delivery_only=delivery_only,
                                        public_dialogue=captured.get('public_dialogue'))
