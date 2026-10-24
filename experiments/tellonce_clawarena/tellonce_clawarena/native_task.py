"""A complete ClawArena task using native tools, isolated memory and deferred grades.

This is a task owner, not a daemon or a review-gate bypass. The phase supervisor
supplies admission, role transports and memory factories; it seals the shared
grading queue only after all interactions and required learning have finished.
"""
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import stat

from tellonce_common.codex_task_session import CodexTaskSession
from tellonce_common.claude_task_session import ClaudeTaskSession
from tellonce_common.claude_native import ClaudeConfig
from tellonce_common.codex_text import CodexTextConfig
from tellonce_common.contracts import Split
from tellonce_common.deferred_scoring import DeferredScoring,_write
from tellonce_common.frozen_task_evaluation import seal_task_evaluation,open_task_evaluation,_directory,_read
from tellonce_common.independent_scoring import IndependentScorer
from tellonce_common.interaction_loop import Grade,InteractionLoop,TaskOutcome
from tellonce_common.no_memory_session import NoMemorySession
from tellonce_common.preference_transport import PreferenceJudge
from tellonce_common.provenance import content_hash,tree_hash
from tellonce_common.role_inputs import RunIdentity,feedback_payload
from tellonce_common.request_budget import bounded_deadline
from tellonce_common.task_evaluation_context import TaskEvaluationContext
from tellonce_common.task_outcomes import outcome_to_dict,outcome_from_dict
from tellonce_common.round_recovery import load_round, commit_round, open_memory
from .clawarena_loader import load_scenario,project_native_scenario
from .native_exec_scoring import CommandSpec,NativeCommandGrader
from .native_mc_scoring import score_multiple_choice
from .native_scoring import NativeScenarioScorer
from .native_workspace import load_claw_assets,NativeClawWorkspace,public_materials_for_unit


def _task_sources(identity,units,profile,source_root,expected_asset_digest):
    if (not isinstance(identity,RunIdentity) or identity.benchmark!='clawarena' or
            not isinstance(units,tuple) or not units or not isinstance(source_root,Path) or
            not source_root.is_absolute() or source_root!=source_root.resolve()):
        raise ValueError('one complete ClawArena task and pinned source directory are required')
    first=units[0].instance
    scenario=load_scenario(source_root,units[0].independent_task_id)
    expected,references=project_native_scenario(scenario,dataset_revision=first.dataset_revision,
        task_family=first.task_family,split=first.split,user_id=identity.user_id,seed=first.seed)
    if units!=expected or first.split not in {Split.ACQUISITION,Split.ID_TEST,Split.TASK_OOD}:
        raise ValueError('the task owner must receive every original round in its native order')
    assets=load_claw_assets(source_root,scenario.scenario_id,expected_digest=expected_asset_digest)
    if assets.digest!=expected_asset_digest:
        raise ValueError('public task assets changed after admission')
    for unit,reference in zip(units,references):
        feedback_payload(unit,identity,delivery='',profile=profile)
        assets.expand(json.loads(unit.public_context_json)['update_ids'])
        spec=json.loads(reference.evaluation_json)
        if spec['round_type']=='multi_choice':
            score_multiple_choice('',expected=spec['eval']['answer'],options=spec['eval']['options'])
        else:
            CommandSpec.from_native(spec['eval'])
        if spec['native_pref']:
            CommandSpec.from_native(spec['native_pref'])
    return assets


@dataclass(frozen=True)
class ClawTaskResult:
    outcome: TaskOutcome
    evidence_directory: Path
    evidence_digest: str
    asset_digest: str
    evaluation_digest: str
    profile_digest: str
    receipt_digest: str


class ClawTask:
    def __init__(self, *, identity,units,profile,source_root,expected_asset_digest,directory,
                 queue,feedback,binding,binary,config,source_catalog,authorize,record,hooks_enabled,
                 memory_factory=None,score_before_repair=False,
                 observation_type=None,event_type=None,appeal_reviewer=None,provision_auth=None,
                 offline_base_url=None,timeout_s=600,wait_retry=None,role_deadline_s=60, completed_receipt=None, recovery=None, test_protocol='interactive-v1'):
        _directory(directory)
        assets=_task_sources(identity,units,profile,source_root,expected_asset_digest)
        phase='training' if units[0].instance.split==Split.ACQUISITION else 'test'
        partial=(recovery.value or {}).get('partial',{}) if recovery is not None and recovery.restored else {}
        if ((directory.exists() and completed_receipt is None and partial.get('task_id')!=units[0].independent_task_id) or not isinstance(queue,DeferredScoring) or queue.identity!=identity or
                queue.phase!=phase or queue._manifest['binding']!=binding or queue._sealed is not None or
                not isinstance(config,(CodexTextConfig,ClaudeConfig)) or any(not callable(f) for f in (feedback,authorize,record)) or
                getattr(feedback,'identity',None)!=identity or type(hooks_enabled) is not bool or
                type(score_before_repair) is not bool or (score_before_repair and not hooks_enabled) or
                (identity.method=='no_memory' and (memory_factory is not None or hooks_enabled)) or
                (identity.method!='no_memory' and not callable(memory_factory)) or
                (hooks_enabled and (observation_type is None or event_type is None)) or
                any(not isinstance(path,Path) or not path.is_absolute() or path!=path.resolve()
                    for path in (binary,source_catalog)) or
                (provision_auth is None)==(offline_base_url is None)):
            raise ValueError('explicit matching phase queue, roles, native runtime and method-owned memory required')
        self.identity,self.units,self.profile=identity,units,profile
        self.profile_digest=content_hash(asdict(profile))
        self.assets,self.source_root,self.directory=assets,source_root,directory
        self.evaluation_digest=tree_hash(source_root/'data/clawarena/eval'/assets.scenario_id)
        self.queue,self.feedback,self.binding=queue,feedback,binding
        self.authorize,self.record,self.memory_factory=authorize,record,memory_factory
        self.native_arguments=dict(binary=binary,config=config,source_catalog=source_catalog,
            hooks_enabled=hooks_enabled,observation_type=observation_type,event_type=event_type,
            appeal_reviewer=appeal_reviewer,provision_auth=provision_auth,offline_base_url=offline_base_url,
            timeout_s=timeout_s)
        self.score_before_repair=score_before_repair
        bounded_deadline(role_deadline_s)
        if wait_retry is not None and not callable(wait_retry):
            raise ValueError('task feedback requires a callable supervised wait')
        self.wait_retry,self.role_deadline_s=wait_retry,role_deadline_s
        self.evaluation=TaskEvaluationContext(identity=identity,units=units)
        self.workspace_owner,self.memory,self.native=None,None,None
        self.completed_receipt=completed_receipt
        if completed_receipt is not None:
            result=read_claw_task_result(directory=directory,expected_digest=completed_receipt,
                          identity=identity,binding=binding)
            if result.outcome.aborted or result.outcome.task_id!=units[0].independent_task_id:
                raise ValueError('only a committed complete task can be resumed')
        from tellonce_common.test_protocol import validate_protocol
        self.test_protocol=validate_protocol(test_protocol, benchmark=identity.benchmark)
        self._ran=False
        load_round(self,recovery)

    def _audit(self,event):
        self.evaluation.observe(event)
        self.record(event)

    def _start_memory(self,start):
        self.workspace_owner=NativeClawWorkspace(assets=self.assets,expected_digest=self.assets.digest,
            workspace=self.directory/'work',state_dir=self.directory/'workspace-state',start=start,units=self.units,
            resume=self.round_state is not None)
        self.artifacts=self.workspace_owner.create_artifacts(self.directory/'captures',
            resume=self.round_state['artifacts'] if self.round_state else None)
        if self.identity.method=='no_memory':
            self.memory=NoMemorySession(identity=self.identity,start=start)
        else:
            self.memory=open_memory(self,start,lambda:self.memory_factory(start=dict(start),directory=self.directory/'memory',record=self.record))
        if getattr(self.memory,'start',None)!=start:
            raise ValueError('the method returned memory belonging to another task or namespace')
        if (self.identity.method=='trace' and
                getattr(getattr(self.memory,'runtime',None),'execution_enabled',None) is not self.native_arguments['hooks_enabled']):
            raise ValueError('Tellonce runtime execution and native hook configuration must agree')
        return self.memory

    def _start_solver(self,start):
        runtime=ClaudeTaskSession if isinstance(self.native_arguments['config'],ClaudeConfig) else CodexTaskSession
        additional = {}
        if self.round_state is not None:
            additional['resume']=self.round_state['native']
        self.native=runtime(start=start,workspace=self.workspace_owner.workspace,
            private_root=self.directory/'native',memory=self.memory,artifacts=self.artifacts,
            authorize=self.authorize,record=self.record,**additional,**self.native_arguments)
        self.evaluation.bind_runtime(self.native)
        return self.native

    def run(self):
        if self._ran or self.queue._sealed is not None:
            raise ValueError('a complete task runs once; solver effects cannot be blindly replayed')
        # Revalidate assets and full task definitions immediately before opening
        # the session, not just when the supervisor built its plan.
        _task_sources(self.identity,self.units,self.profile,self.source_root,self.assets.digest)
        if tree_hash(self.source_root/'data/clawarena/eval'/self.assets.scenario_id)!=self.evaluation_digest:
            raise ValueError('private native answers or evaluation scripts changed before the task')
        self.authorize('prepare_task',{'namespace':self.identity.namespace,'binding':self.binding,
            'asset_digest':self.assets.digest,'evaluation_digest':self.evaluation_digest,'profile_digest':self.profile_digest,
            'units':[asdict(unit) for unit in self.units]})
        self._ran=True
        self.directory.mkdir(mode=0o700,exist_ok=self.round_state is not None)
        loop=InteractionLoop(identity=self.identity,profile=self.profile,start_solver=self._start_solver,
            test_protocol=self.test_protocol,
            start_memory=self._start_memory,feedback=self.feedback,score=self.queue,audit=self._audit,
            score_before_repair=self.score_before_repair,
            wait_retry=self.wait_retry,read_deadline_s=self.role_deadline_s,
            prepare_unit=lambda unit,**kwargs:self.workspace_owner.prepare_unit(unit,**kwargs),
            feedback_operations=lambda unit,answer,**kwargs:self.native.feedback_operations(
                answer,event_id=kwargs['event_id']),
            public_materials=lambda unit:public_materials_for_unit(self.assets,self.units,unit.instance.common_instance_id),
            resume=self.round_state['loop'] if self.round_state else None,
            commit_unit=(lambda state:commit_round(self,state)) if self.recovery is not None else None)
        try:
            outcome=loop.run_task(self.units)
        finally:
            try:
                if self.native is not None:
                    self.native.close()
            finally:
                try:
                    if self.memory is not None:
                        self.memory.close()
                finally:
                    if self.workspace_owner is not None:
                        self.workspace_owner.close()
        evidence=self.directory/'evaluation'
        digest=seal_task_evaluation(self.evaluation,directory=evidence,binding=self.binding)
        receipt={'schema':2,'namespace':self.identity.namespace,'binding':self.binding,
                 'asset_digest':self.assets.digest,'evaluation_digest':self.evaluation_digest,
                 'profile_digest':self.profile_digest,'evidence_digest':digest,'outcome':outcome_to_dict(outcome)}
        _write(self.directory/'task-result.json',receipt)
        return ClawTaskResult(outcome,evidence,digest,self.assets.digest,self.evaluation_digest,
                              self.profile_digest,content_hash(receipt))


def read_claw_task_result(*,directory,expected_digest,identity,binding):
    _directory(directory)
    value=_read(directory/'task-result.json',256*1024*1024)
    if (content_hash(value)!=expected_digest or set(value)!={'schema','namespace','binding','asset_digest',
                                                           'evaluation_digest','profile_digest','evidence_digest','outcome'} or
            value['schema']!=2 or value['namespace']!=identity.namespace or value['binding']!=binding):
        raise ValueError('task outcome receipt changed or belongs to another experiment')
    outcome=outcome_from_dict(value['outcome'])
    return ClawTaskResult(outcome,directory/'evaluation',value['evidence_digest'],value['asset_digest'],
                          value['evaluation_digest'],value['profile_digest'],expected_digest)


def open_claw_scorer(*,result,identity,units,profile,source_root,expected_asset_digest,binding,
                     binary,python,python_roots,directory,preference_judge,authorize,record,wait_retry,role_deadline_s=7200):
    assets=_task_sources(identity,units,profile,source_root,expected_asset_digest)
    if (not isinstance(result,ClawTaskResult) or result.asset_digest!=assets.digest or
            result.profile_digest!=content_hash(asdict(profile)) or
            result.evaluation_digest!=tree_hash(source_root/'data/clawarena/eval'/assets.scenario_id) or
            not isinstance(preference_judge,PreferenceJudge) or preference_judge.identity!=identity):
        raise ValueError('private scoring requires this task archive and its own preference judge')
    receipt=read_claw_task_result(directory=result.evidence_directory.parent,expected_digest=result.receipt_digest,
                                  identity=identity,binding=binding)
    if (receipt!=result or tuple(unit.unit_id for unit in result.outcome.units)!=
            tuple(unit.instance.common_instance_id for unit in units)):
        raise ValueError('scoring must use the complete original task result and its bound archive')
    evaluation=open_task_evaluation(directory=result.evidence_directory,expected_digest=result.evidence_digest,
        identity=identity,units=units,binding=binding,
        baseline={asset.path:{'data':asset.data,'mode':asset.mode|stat.S_IRUSR|stat.S_IWUSR} for asset in assets.initial})
    if evaluation.task_id!=result.outcome.task_id or evaluation.session_id!=result.outcome.session_id:
        raise ValueError('task result and immutable evaluation archive describe different sessions')
    start=dict(namespace=identity.namespace,task_id=evaluation.task_id,session_id=evaluation.session_id,
               training=units[0].instance.split==Split.ACQUISITION)
    baseline={asset.path:{'data':asset.data,'mode':asset.mode|stat.S_IRUSR|stat.S_IWUSR} for asset in assets.initial}
    grader=NativeCommandGrader(identity=identity,start=start,binary=binary,python=python,python_roots=python_roots,
        eval_root=source_root/'data/clawarena/eval',private_root=directory,
        artifacts=evaluation.runtime.artifacts,baseline=baseline,authorize=authorize,record=record)
    native=NativeScenarioScorer(units=units,grader=grader,source_root=source_root,record=record)
    by_id={unit.instance.common_instance_id:unit for unit in units}
    def native_score(unit_id,answer,*,context):
        value=native.score(by_id[unit_id],answer,context=context)
        return Grade(value.score,value.passed,())
    def preferences(unit_id,answer,*,context):
        value=evaluation.preference_input(unit_id,answer,profile=profile,context=context,
            public_materials_json=public_materials_for_unit(assets,units,unit_id))
        return preference_judge(value,context=context)
    return IndependentScorer(native=native_score,preferences=preferences,record=record,wait_retry=wait_retry,
                             deadline_s=role_deadline_s)
