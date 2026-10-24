"""One complete travel row with isolated native tools and deferred evaluation.

The phase supervisor must admit the source, data, models and tool environment
before constructing this owner. No benchmark reset or reference answer is
ever passed to the native solver, feedback agent or public tool process.
"""
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from tellonce_common.codex_task_session import CodexTaskSession
from tellonce_common.claude_native import ClaudeConfig
from tellonce_common.claude_task_session import ClaudeTaskSession
from tellonce_common.contracts import Split
from tellonce_common.deferred_scoring import DeferredScoring, _write
from tellonce_common.frozen_task_evaluation import _directory, seal_task_evaluation
from tellonce_common.interaction_loop import InteractionLoop, TaskOutcome
from tellonce_common.native_benchmark_tools import NativeBenchmarkTools
from tellonce_common.native_isolation import codex_permission_arguments
from tellonce_common.no_memory_session import NoMemorySession
from tellonce_common.provenance import content_hash
from tellonce_common.role_inputs import RunIdentity, feedback_payload
from tellonce_common.task_artifacts import TaskArtifacts
from tellonce_common.task_evaluation_context import TaskEvaluationContext
from tellonce_common.task_outcomes import outcome_to_dict
from tellonce_common.round_recovery import load_round, commit_round, open_memory
from .memoryarena_loader import project_native_row
from .travel_submission import NativeTravelSubmission
from .travel_tool_process import TravelToolProcess
from .travel_tool_service import native_tool_definitions


@dataclass(frozen=True)
class TravelTaskResult:
    outcome: TaskOutcome
    evidence_directory: Path
    evidence_digest: str
    receipt_digest: str


class NativeTravelTask:
    def __init__(self, *, identity, row, expected_row_hash, units, profile, directory, binding, queue,
                 binary, config, catalog, official_source, tool_python, tool_runtime_roots,
                 public_tool_source, database, feedback, authorize, record, memory_factory=None,
                 hooks_enabled=False, score_before_repair=False, observation_type=None, event_type=None,
                 appeal_reviewer=None, provision_auth=None, offline_base_url=None,
                 task_timeout_s=600, role_deadline_s=60, wait_retry=None, solver_binary=None, completed_receipt=None, recovery=None, test_protocol='interactive-v1'):
        _directory(directory)
        partial=(recovery.value or {}).get('partial',{}) if recovery is not None and recovery.restored else {}
        if (not isinstance(identity,RunIdentity) or identity.benchmark!='memoryarena' or
                not isinstance(units,tuple) or not units or (directory.exists() and completed_receipt is None and partial.get('task_id')!=units[0].independent_task_id) or
                not isinstance(row,dict) or content_hash(row)!=expected_row_hash or
                not isinstance(queue,DeferredScoring) or queue.identity!=identity or queue._sealed is not None or
                queue._manifest['binding']!=binding or any(not callable(f) for f in (feedback,authorize,record)) or
                getattr(feedback,'identity',None)!=identity or type(hooks_enabled) is not bool or
                type(score_before_repair) is not bool or (score_before_repair and not hooks_enabled) or
                (identity.method=='no_memory' and (memory_factory is not None or hooks_enabled)) or
                (identity.method!='no_memory' and not callable(memory_factory)) or
                (hooks_enabled and (observation_type is None or event_type is None)) or
                (provision_auth is None)==(offline_base_url is None) or
                not isinstance(tool_runtime_roots,tuple) or not tool_runtime_roots):
            raise ValueError('one admitted complete travel task and matching native roles required')
        first=units[0].instance
        expected,_=project_native_row(row,subset='group_travel_planner',dataset_revision=first.dataset_revision,
            split=first.split,user_id=identity.user_id,seed=first.seed)
        phase='training' if first.split==Split.ACQUISITION else 'test'
        if units!=expected or queue.phase!=phase or first.split not in {Split.ACQUISITION,Split.ID_TEST,Split.TASK_OOD}:
            raise ValueError('travel owner must receive every original question in order and the right phase')
        for path in (binary,catalog,official_source,tool_python,public_tool_source,database,*tool_runtime_roots):
            if not isinstance(path,Path) or not path.is_absolute() or path!=path.resolve():
                raise ValueError('travel runtime paths must be explicit and canonical')
        if isinstance(config, ClaudeConfig):
            if (not isinstance(solver_binary, Path) or not solver_binary.is_absolute() or
                    solver_binary != solver_binary.resolve() or solver_binary == binary or
                    identity.method not in {'no_memory', 'trace'} or
                    (hooks_enabled and identity.method != 'trace')):
                raise ValueError('Claude travel requires a separate solver binary and an admitted memory adapter')
        elif solver_binary is not None and solver_binary != binary:
            raise ValueError('Codex solver must use its admitted tool sandbox binary')
        if not any(tool_python.is_relative_to(path) for path in tool_runtime_roots):
            raise ValueError('tool interpreter must belong to its declared runtime roots')
        for unit in units:feedback_payload(unit,identity,delivery='',profile=profile)
        # Verify all parser/prompt sources now, without loading a model or data.
        self.submission=NativeTravelSubmission(official_source,units=units)
        self.identity,self.units,self.profile=identity,units,profile
        self.directory,self.binding,self.queue=directory,binding,queue
        self.row_hash,self.profile_hash=expected_row_hash,content_hash(asdict(profile))
        self.feedback,self.authorize,self.record=feedback,authorize,record
        self.memory_factory,self.hooks_enabled=memory_factory,hooks_enabled
        self.score_before_repair,self.wait_retry,self.role_deadline_s=score_before_repair,wait_retry,role_deadline_s
        self.tool_python,self.tool_runtime_roots=tool_python,tool_runtime_roots
        self.public_tool_source,self.database=public_tool_source,database
        self.tool_sandbox_binary = binary
        self.native_arguments=dict(binary=solver_binary or binary,config=config,source_catalog=catalog,
            hooks_enabled=hooks_enabled,observation_type=observation_type,event_type=event_type,
            appeal_reviewer=appeal_reviewer,provision_auth=provision_auth,offline_base_url=offline_base_url,
            timeout_s=task_timeout_s)
        self.evaluation=TaskEvaluationContext(identity=identity,units=units)
        self.native,self.memory,self.artifacts,self.tools=None,None,None,None
        self.completed_receipt=completed_receipt
        if completed_receipt is not None:
            from .native_travel_evaluation import read_travel_task_result
            result=read_travel_task_result(directory=directory,expected_digest=completed_receipt,
                                          identity=identity,binding=binding)
            if result.outcome.aborted or result.outcome.task_id!=units[0].independent_task_id:
                raise ValueError('only a committed complete task can be resumed')
        from tellonce_common.test_protocol import validate_protocol
        self.test_protocol=validate_protocol(test_protocol, benchmark=identity.benchmark)
        self._cleanup,self._ran=ExitStack(),False
        load_round(self,recovery)

    def _feedback_with_evidence(self, payload, *, context):
        from .travel_feedback_evidence import venue_evidence
        enriched=dict(payload)
        enriched['public_lookup']=venue_evidence(self.database,payload)
        return self.feedback(enriched,context=context)

    def _audit(self,event):
        self.evaluation.observe(event)
        self.record(event)

    def _public_check_context(self, *, user_text, delivery=None):
        from .travel_feedback_evidence import execution_history_context
        # _history contains only observed public user inputs and delivered
        # replies. It is restored by the existing round checkpoint, with no
        # reference answers, grading results or future turns.
        if (self.evaluation.task_id != self.memory.start['task_id'] or
                self.evaluation.session_id != self.memory.start['session_id']):
            raise ValueError('execution history belongs to another task session')
        return execution_history_context(self.memory.runtime.public_task,
            self.evaluation._history, user_text=user_text, delivery=delivery)

    def _start_memory(self,start):
        work=self.directory/'work'
        work.mkdir(mode=0o700,exist_ok=self.round_state is not None)
        self.artifacts=TaskArtifacts(workspace=work,directory=self.directory/'captures',
            resume=self.round_state['artifacts'] if self.round_state else None)
        self._cleanup.callback(self.artifacts.close)
        self.memory=(NoMemorySession(identity=self.identity,start=start) if self.identity.method=='no_memory'
                     else open_memory(self,start,lambda:self.memory_factory(start=dict(start),directory=self.directory/'memory',record=self.record)))
        self._cleanup.callback(self.memory.close)
        if getattr(self.memory,'start',None)!=start:
            raise ValueError('travel task received memory owned by another task')
        if (self.identity.method=='trace' and
                getattr(getattr(self.memory,'runtime',None),'execution_enabled',None) is not self.hooks_enabled):
            raise ValueError('Tellonce execution setting differs from travel native hooks')
        if self.hooks_enabled:
            self.memory.public_check_context = self._public_check_context
        return self.memory

    def _start_solver(self,start):
        from .travel_flight_index import INDEX_ROOT
        work,home=self.directory/'tool-work',self.directory/'tool-home'
        work.mkdir(mode=0o700);home.mkdir(mode=0o700)
        binary=self.tool_sandbox_binary
        service=Path(__file__).with_name('travel_tool_service.py')
        permissions=codex_permission_arguments(workspace=work,cli_home=home,
            runtime_roots=(binary.parent,*self.tool_runtime_roots),
            readonly_assets=(self.public_tool_source,self.database,service,service.with_name('travel_tool_pins.json'),
                             service.with_name('travel_flight_index.py'),INDEX_ROOT))
        owner={key:start[key] for key in ('namespace','task_id','session_id')}
        self.authorize('prepare_benchmark_tools',owner)
        client=TravelToolProcess(command=[str(binary),'sandbox','-C',str(work),'-P','experiment',*permissions,
            str(self.tool_python),'-I','-B',str(service)],cwd=work,
            env={'HOME':str(work),'CODEX_HOME':str(home),'PATH':'/usr/bin:/bin','LANG':'C.UTF-8',
                 'TMPDIR':str(work),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
            setup={'owner':owner,'source_root':str(self.public_tool_source),'database_root':str(self.database)},
            record=self.record)
        self._cleanup.callback(client.close)
        self.tools=NativeBenchmarkTools(owner=owner,definitions=native_tool_definitions(),invoke=client.call,
            close=client.close,record=self.record,directory=self.directory/'tool-receipts')
        self._cleanup.callback(self.tools.close)
        runtime = ClaudeTaskSession if isinstance(self.native_arguments['config'], ClaudeConfig) else CodexTaskSession
        extra={'resume':self.round_state['native']} if self.round_state else {}
        self.native=runtime(start=start,workspace=self.directory/'work',private_root=self.directory/'native',
            memory=self.memory,artifacts=self.artifacts,authorize=self.authorize,record=self.record,
            benchmark_tools=self.tools,**extra,**self.native_arguments)
        self._cleanup.callback(self.native.close)
        self.evaluation.bind_runtime(self.native)
        return self.native

    def run(self):
        if self._ran or self.queue._sealed is not None:
            raise ValueError('native travel tasks cannot be replayed as fresh solver interactions')
        self.authorize('prepare_task',{'namespace':self.identity.namespace,'binding':self.binding,
            'row_hash':self.row_hash,'profile_hash':self.profile_hash,'units':[asdict(u) for u in self.units]})
        self._ran=True
        self.directory.mkdir(mode=0o700,exist_ok=self.round_state is not None)
        loop=InteractionLoop(identity=self.identity,profile=self.profile,start_solver=self._start_solver,
            test_protocol=self.test_protocol,
            start_memory=self._start_memory,feedback=self._feedback_with_evidence,score=self.queue,audit=self._audit,
            score_before_repair=self.score_before_repair,read_deadline_s=self.role_deadline_s,wait_retry=self.wait_retry,
            feedback_operations=lambda unit,answer,**kwargs:self.native.feedback_operations(answer,event_id=kwargs['event_id']),
            resume=self.round_state['loop'] if self.round_state else None,
            commit_unit=(lambda state:commit_round(self,state)) if self.recovery is not None else None)
        try:
            outcome=loop.run_task(self.units)
        finally:
            self._cleanup.close()
        evidence=self.directory/'evaluation'
        digest=seal_task_evaluation(self.evaluation,directory=evidence,binding=self.binding)
        receipt={'schema':1,'namespace':self.identity.namespace,'binding':self.binding,'row_hash':self.row_hash,
            'profile_hash':self.profile_hash,'evidence_digest':digest,'outcome':outcome_to_dict(outcome)}
        _write(self.directory/'task-result.json',receipt)
        return TravelTaskResult(outcome,evidence,digest,content_hash(receipt))
