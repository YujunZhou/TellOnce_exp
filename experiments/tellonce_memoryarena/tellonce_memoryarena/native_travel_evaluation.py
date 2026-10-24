"""Read-only travel scoring from receipt-bound, delivery-time evidence.

Per-question grades describe the current traveler. Native group metrics use
every traveler's latest plan at the last question's delivery boundary; they
must not be assembled from earlier per-question grades. The phase owner must
admit evaluation only after interaction and required learning have finished.
"""
from dataclasses import asdict

from tellonce_common.contracts import Split
from tellonce_common.frozen_task_evaluation import _directory, _read, open_task_evaluation
from tellonce_common.independent_scoring import IndependentScorer
from tellonce_common.interaction_loop import Grade, SolverAnswer
from tellonce_common.preference_transport import PreferenceJudge
from tellonce_common.provenance import content_hash
from tellonce_common.role_inputs import RunIdentity, feedback_payload
from tellonce_common.task_outcomes import outcome_from_dict
from .memoryarena_loader import project_native_row
from .native_scoring import NativeTravelScorer, TravelScore
from .native_travel_task import TravelTaskResult
from .travel_submission import NativeTravelSubmission


def read_travel_task_result(*,directory,expected_digest,identity,binding):
    _directory(directory)
    value=_read(directory/'task-result.json',256*1024*1024)
    fields={'schema','namespace','binding','row_hash','profile_hash','evidence_digest','outcome'}
    if (content_hash(value)!=expected_digest or set(value)!=fields or value['schema']!=1 or
            value['namespace']!=identity.namespace or value['binding']!=binding):
        raise ValueError('travel task receipt changed or belongs to another experiment')
    return TravelTaskResult(outcome_from_dict(value['outcome']),directory/'evaluation',
                            value['evidence_digest'],expected_digest)


class TravelTaskEvaluation:
    def __init__(self, *, result,identity,row,expected_row_hash,units,profile,binding,official_source,record):
        if (not isinstance(result,TravelTaskResult) or not isinstance(identity,RunIdentity) or
                identity.benchmark!='memoryarena' or not isinstance(units,tuple) or not units or
                not isinstance(row,dict) or content_hash(row)!=expected_row_hash or not callable(record)):
            raise ValueError('one complete receipt-bound travel task and private scoring sources required')
        first=units[0].instance
        expected,references=project_native_row(row,subset='group_travel_planner',dataset_revision=first.dataset_revision,
            split=first.split,user_id=identity.user_id,seed=first.seed)
        if units!=expected or first.split not in {Split.ACQUISITION,Split.ID_TEST,Split.TASK_OOD}:
            raise ValueError('private travel scoring requires every original question in order')
        for unit in units:feedback_payload(unit,identity,delivery='',profile=profile)
        receipt=read_travel_task_result(directory=result.evidence_directory.parent,
            expected_digest=result.receipt_digest,identity=identity,binding=binding)
        raw=_read(result.evidence_directory.parent/'task-result.json',256*1024*1024)
        if (content_hash(raw)!=result.receipt_digest or receipt!=result or
                raw['row_hash']!=expected_row_hash or raw['profile_hash']!=content_hash(asdict(profile)) or
                tuple(u.unit_id for u in result.outcome.units)!=tuple(u.instance.common_instance_id for u in units)):
            raise ValueError('travel result, row, profile or archive differs from the captured task')
        self.evaluation=open_task_evaluation(directory=result.evidence_directory,expected_digest=result.evidence_digest,
            identity=identity,units=units,binding=binding,baseline={})
        if (self.evaluation.task_id!=result.outcome.task_id or self.evaluation.session_id!=result.outcome.session_id):
            raise ValueError('travel result and frozen evidence describe different sessions')
        self.identity,self.units,self.references,self.profile=identity,units,references,profile
        self.result,self.record=result,record
        self.phase='training' if first.split==Split.ACQUISITION else 'test'
        self.parser=NativeTravelSubmission(official_source,units=units)
        self.native=NativeTravelScorer(official_source/'env/env_systems/travel_planner_env/eval.py')
        self._unit_ids=tuple(u.instance.common_instance_id for u in units)
        self._events={unit_id:[] for unit_id in self._unit_ids}
        for event_id,(unit_id,answer,_,_) in self.evaluation._deliveries.items():
            self._events[unit_id].append(event_id)
        for outcome in result.outcome.units:
            events=self._events[outcome.unit_id]
            trace=outcome.trace
            if (len(events)!=(len(trace.deliveries) if trace else 0) or
                    trace and any(self.evaluation._deliveries[e][1].artifact_hash!=d.artifact_hash
                                  for e,d in zip(events,trace.deliveries))):
                raise ValueError('travel outcome deliveries differ from the frozen transcript')
            corrections=[e['text'] for e in self.evaluation._events if e['type']=='user_input' and
                         e['unit_id']==outcome.unit_id and e['delivery_index']>0]
            if corrections!=(trace.corrections if trace else []):
                raise ValueError('travel correction boundary differs from the frozen transcript')

    def _candidate(self,unit_id,answer,context):
        fields={'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (unit_id not in self._unit_ids or not isinstance(answer,SolverAnswer) or
                not isinstance(context,dict) or set(context)!=fields or context['phase']!=self.phase or
                context['task_id']!=self.evaluation.task_id or context['session_id']!=self.evaluation.session_id or
                context['role'] not in {'score','score_before_repair'} or type(context['attempt']) is not int or
                context['attempt'] not in {1,2} or context['event_id'] not in self.evaluation._deliveries or
                context['call_id']!=content_hash([self.identity.namespace,self.evaluation.session_id,
                                                unit_id,context['event_id'],context['role']])):
            raise ValueError('travel scoring context is not an actual captured delivery')
        actual_unit,actual,history,_=self.evaluation._deliveries[context['event_id']]
        if context['role']=='score_before_repair':
            if actual.before_repair_text is None:
                raise ValueError('no pre-repair candidate was captured')
            expected=SolverAnswer(actual.before_repair_text,actual.before_repair_artifact_hash,
                                 execution_evidence_hash=actual.before_repair_execution_evidence_hash)
        else:expected=actual
        if actual_unit!=unit_id or answer!=expected:
            raise ValueError('travel candidate differs from the captured delivery or pre-repair draft')
        artifacts=self.evaluation.runtime.artifacts
        if artifacts.load(answer.artifact_hash)['answer']!=answer.text:
            raise ValueError('travel candidate differs from its frozen artifact')
        self.evaluation.runtime.execution_evidence.load(answer.execution_evidence_hash,
            artifact_hash=answer.artifact_hash,user_event_id=context['event_id'])
        return tuple(row['text'] for row in history if row['role']=='assistant')+(answer.text,)

    def score_snapshot(self,unit_id,answer,*,context):
        texts=self._candidate(unit_id,answer,context)
        plans=self.parser.plans_at(unit_id,delivered_texts=texts)
        scores=tuple(self.native.score(unit,reference,plan['plan'])
                     for unit,reference,plan in zip(self.units,self.references,plans))
        self.record({'type':'native_travel_score','namespace':self.identity.namespace,'unit_id':unit_id,
            'context':dict(context),'artifact_hash':answer.artifact_hash,
            'execution_evidence_hash':answer.execution_evidence_hash,
            'traveler_scores':[{'unit_id':unit.instance.common_instance_id,**asdict(score)}
                               for unit,score in zip(self.units,scores)],
            'definition':'latest_plan_per_introduced_traveler_at_captured_delivery'})
        return scores

    def native_grade(self,unit_id,answer,*,context):
        score=self.score_snapshot(unit_id,answer,context=context)[-1]
        return Grade(float(score.full_pass),score.full_pass,())

    def independent(self,*,preference_judge,wait_retry,role_deadline_s=7200):
        if not isinstance(preference_judge,PreferenceJudge) or preference_judge.identity!=self.identity:
            raise ValueError('travel preferences require their own experiment-owned judge')
        def preferences(unit_id,answer,*,context):
            value=self.evaluation.preference_input(unit_id,answer,profile=self.profile,context=context)
            return preference_judge(value,context=context)
        return IndependentScorer(native=self.native_grade,preferences=preferences,record=self.record,
                                 wait_retry=wait_retry,deadline_s=role_deadline_s)

    def endpoint_scores(self,stage):
        """Whole-group state at the last question's first/final delivery.

        Earlier questions keep their real correction history. This is not a
        counterfactual run without corrections. An unanswered final correction
        has no final state; last_usable is an explicitly separate diagnostic.
        """
        if stage not in {'first','final','last_usable'}:
            raise ValueError('select an explicit whole-task travel endpoint')
        last=self.result.outcome.units[-1];events=self._events[last.unit_id]
        trace=last.trace
        if not events or stage=='final' and len(trace.corrections)==len(trace.deliveries):
            return tuple(TravelScore(None,None,self.native.score(u,r,None).constraint_total)
                         for u,r in zip(self.units,self.references))
        event_id=events[0 if stage=='first' else -1]
        context={'phase':self.phase,'task_id':self.evaluation.task_id,'session_id':self.evaluation.session_id,
            'event_id':event_id,'role':'score','attempt':1,
            'call_id':content_hash([self.identity.namespace,self.evaluation.session_id,last.unit_id,event_id,'score'])}
        return self.score_snapshot(last.unit_id,self.evaluation._deliveries[event_id][1],context=context)
