"""Seal private grading jobs after interaction, then evaluate frozen deliveries.

Enqueue does not call a judge. Pending grades cannot be aggregated as results.
The phase owner seals only after its interactions and required learning drain;
the supplied phase receipt binds that boundary. Source admission stays mandatory
for evaluation and sealed replay. This does not resume solver/learning effects.
"""
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import tempfile

from .interaction_loop import Grade, SolverAnswer, TaskOutcome
from .interaction_metrics import PreferenceVerdict
from .provenance import canonical_json_bytes, content_hash, file_hash
from .role_inputs import RunIdentity


def _hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _write(path, value):
    descriptor,temporary=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary,path)  # Atomic publication, without overwriting a receipt.
    finally:
        os.unlink(temporary)
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DeferredScoring:
    def __init__(self, *, directory: Path, identity: RunIdentity, phase: str, binding: str, authorize, record):
        self._setup(directory,identity,phase,binding,authorize,record)
        if directory.exists():
            raise ValueError('grading collection requires a fresh private directory')
        authorize('prepare',dict(self._manifest))
        directory.mkdir(mode=0o700)
        for name in ('jobs','results'):
            (directory/name).mkdir(mode=0o700)
        _write(directory/'manifest.json',self._manifest)

    def _setup(self,directory,identity,phase,binding,authorize,record):
        if (not isinstance(directory,Path) or not directory.is_absolute() or directory!=directory.resolve() or
                not isinstance(identity,RunIdentity) or phase not in {'training','test'} or not _hash(binding) or
                not callable(authorize) or not callable(record)):
            raise ValueError('private namespace, phase, source binding and evaluation admission are required')
        if any((path/'.git/HEAD').is_file() or (path/'.git').is_file() for path in (directory,*directory.parents)):
            raise ValueError('private grading jobs must be outside source repositories')
        self.directory,self.identity,self.phase,self.authorize,self.record=directory,identity,phase,authorize,record
        self._manifest={'schema':1,'namespace':identity.namespace,'phase':phase,'binding':binding}
        self._jobs={}
        self._sealed=None

    def _validate(self,unit_id,answer,context):
        fields={'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (not isinstance(unit_id,str) or not unit_id or not isinstance(answer,SolverAnswer) or
                not isinstance(context,dict) or set(context)!=fields or context['phase']!=self.phase or
                context['role'] not in {'score','score_before_repair'} or
                type(context['attempt']) is not int or context['attempt']!=1 or
                any(not isinstance(context[key],str) or not 1<=len(context[key])<=480 for key in fields-{'attempt'})):
            raise ValueError('only an actual delivery scoring request may be queued')
        expected=content_hash([self.identity.namespace,context['session_id'],unit_id,context['event_id'],context['role']])
        if context['call_id']!=expected:
            raise ValueError('queued scoring request belongs to another experiment or delivery')
        body={'unit_id':unit_id,'answer':asdict(answer),'context':dict(context)}
        return {**body,'request_hash':content_hash(body)}

    def __call__(self,unit_id,answer,*,context):
        if self._sealed is not None:
            raise ValueError('a sealed interaction phase cannot add grading jobs')
        row=self._validate(unit_id,answer,context)
        key=context['call_id']
        if key in self._jobs:
            if self._jobs[key]!=row:
                raise ValueError('a queued scoring request changed its delivery')
        else:
            _write(self.directory/'jobs'/(key+'.json'),row)
            self._jobs[key]=row
            self.record({'type':'private_score_enqueued','namespace':self.identity.namespace,
                         'context':dict(context),'unit_id':unit_id,'request_hash':row['request_hash']})
        return Grade(None,None,(),scoring_pending=True)

    def seal(self, *, phase_receipt: str):
        if self._sealed is not None or not _hash(phase_receipt):
            raise ValueError('seal once using the completed interaction/learning phase receipt')
        jobs={}
        for key,row in self._jobs.items():
            path=self.directory/'jobs'/(key+'.json')
            if path.is_symlink() or path.read_bytes()!=canonical_json_bytes(row):
                raise ValueError('a collected grading job changed before the phase was sealed')
            jobs[key]=file_hash(path)
        seal={'schema':1,'manifest_hash':content_hash(self._manifest),'phase_receipt':phase_receipt,'jobs':jobs}
        _write(self.directory/'seal.json',seal)
        self._sealed=content_hash(seal)
        return self._sealed

    @classmethod
    def open_unsealed(cls, *, directory, identity, phase, binding, authorize, record):
        result=cls.__new__(cls)
        result._setup(directory,identity,phase,binding,authorize,record)
        if (json.loads((directory/'manifest.json').read_text())!=result._manifest or
                (directory/'seal.json').exists() or any((directory/'results').iterdir())):
            raise ValueError('resume requires the matching unsealed interaction queue')
        for path in sorted((directory/'jobs').iterdir()):
            if path.is_symlink() or not path.is_file() or not _hash(path.stem) or path.suffix!='.json':
                raise ValueError('invalid checkpoint grading job')
            row=json.loads(path.read_text())
            if (row!=result._validate(row['unit_id'],SolverAnswer(**row['answer']),row['context']) or
                    row['context']['call_id']!=path.stem):
                raise ValueError('checkpoint grading job differs from its request')
            result._jobs[path.stem]=row
        authorize('resume_unsealed',dict(result._manifest))
        return result

    @classmethod
    def open_sealed(cls, *, directory, identity, phase, binding, expected_seal, authorize, record):
        result=cls.__new__(cls)
        result._setup(directory,identity,phase,binding,authorize,record)
        if json.loads((directory/'manifest.json').read_text())!=result._manifest or not _hash(expected_seal):
            raise ValueError('sealed scoring directory has another source or experiment identity')
        seal=json.loads((directory/'seal.json').read_text())
        if (content_hash(seal)!=expected_seal or seal.get('schema')!=1 or
                seal.get('manifest_hash')!=content_hash(result._manifest) or not _hash(seal.get('phase_receipt')) or
                not isinstance(seal.get('jobs'),dict)):
            raise ValueError('the grading seal differs from the admitted phase receipt')
        expected_files={key+'.json' for key in seal['jobs'] if _hash(key)}
        if len(expected_files)!=len(seal['jobs']) or {p.name for p in (directory/'jobs').iterdir()}!=expected_files:
            raise ValueError('sealed grading jobs were added or removed')
        for key,digest in seal['jobs'].items():
            path=directory/'jobs'/(key+'.json')
            if path.is_symlink() or file_hash(path)!=digest:
                raise ValueError('sealed grading job was changed')
            row=json.loads(path.read_text())
            validated=result._validate(row['unit_id'],SolverAnswer(**row['answer']),row['context'])
            if row!=validated or row['context']['call_id']!=key:
                raise ValueError('malformed sealed grading job')
            result._jobs[key]=row
        result._sealed=expected_seal
        authorize('open_sealed',{**result._manifest,'seal':expected_seal})
        return result

    def _result(self,key):
        path=self.directory/'results'/(key+'.json')
        if not path.exists():
            return None
        if path.is_symlink():
            raise ValueError('private grading receipt cannot be a symlink')
        receipt=json.loads(path.read_text())
        if (set(receipt)!={'request_hash','grade','digest'} or
                receipt['request_hash']!=self._jobs[key]['request_hash'] or
                receipt['digest']!=content_hash({'request_hash':receipt['request_hash'],'grade':receipt['grade']})):
            raise ValueError('grading receipt no longer matches its frozen request')
        value=dict(receipt['grade'])
        value['preferences']=tuple(PreferenceVerdict(**row) for row in value['preferences'])
        grade=Grade(**value)
        if grade.scoring_pending:
            raise ValueError('a pending grade is not an evaluation receipt')
        return grade

    def evaluate(self, score):
        if self._sealed is None or not callable(score):
            raise ValueError('evaluate only after the interaction phase has been sealed')
        # Recheck the seal and all immutable job files before invoking any judge.
        verified=self.open_sealed(directory=self.directory,identity=self.identity,phase=self.phase,
            binding=self._manifest['binding'],expected_seal=self._sealed,authorize=self.authorize,record=self.record)
        self._jobs=verified._jobs
        descriptor=os.open(self.directory/'evaluation.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            try:
                fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('another worker already owns evaluation of this sealed queue') from None
            self._evaluate_owned(score)
        finally:
            os.close(descriptor)

    def _evaluate_owned(self,score):
        for key,row in self._jobs.items():
            self.authorize('evaluate',{**self._manifest,'seal':self._sealed,'call_id':key})
            if self._result(key) is not None:
                continue
            grade=score(row['unit_id'],SolverAnswer(**row['answer']),context=dict(row['context']))
            if not isinstance(grade,Grade) or grade.scoring_pending:
                raise ValueError('deferred evaluation requires an actual completed independent grade')
            body={'request_hash':row['request_hash'],'grade':asdict(grade)}
            _write(self.directory/'results'/(key+'.json'),{**body,'digest':content_hash(body)})
            self.record({'type':'private_score_completed','namespace':self.identity.namespace,
                         'context':row['context'],'unit_id':row['unit_id'],'grade':asdict(grade)})

    def apply(self,outcome: TaskOutcome):
        if self._sealed is None or not isinstance(outcome,TaskOutcome):
            raise ValueError('apply only completed private grades to their captured task')
        verified=self.open_sealed(directory=self.directory,identity=self.identity,phase=self.phase,
            binding=self._manifest['binding'],expected_seal=self._sealed,authorize=self.authorize,record=self.record)
        self._jobs=verified._jobs
        replacements=[]
        for unit in outcome.units:
            trace=unit.trace
            if trace is None:
                continue
            if trace.unit_id!=unit.unit_id or trace.task_id!=outcome.task_id:
                raise ValueError('pending trace belongs to another scoring unit or task')
            for index,delivery in enumerate(trace.deliveries):
                event=content_hash([self.identity.namespace,outcome.session_id,unit.unit_id,index])
                def key(role):
                    return content_hash([self.identity.namespace,outcome.session_id,unit.unit_id,event,role])
                normal=key('score')
                if normal not in self._jobs or self._jobs[normal]['context']['task_id']!=outcome.task_id:
                    raise ValueError('evaluation receipt belongs to another task')
                job=SolverAnswer(**self._jobs[normal]['answer'])
                if (job.artifact_hash!=delivery.artifact_hash or job.internal_repairs!=delivery.internal_repairs or
                        job.before_repair_artifact_hash!=delivery.before_repair_artifact_hash):
                    raise ValueError('evaluation receipt belongs to another delivery artifact')
                grade=self._result(normal)
                if grade is None:
                    raise ValueError('task still has an unevaluated delivery')
                replacements.append((trace,index,False,grade))
                draft=trace.before_repair_drafts[index]
                if draft is not None:
                    draft_key=key('score_before_repair')
                    if draft_key in self._jobs:
                        if self._jobs[draft_key]['answer']['artifact_hash']!=draft.artifact_hash:
                            raise ValueError('evaluation receipt belongs to another pre-repair artifact')
                        before=self._result(draft_key)
                    elif (draft.artifact_hash==delivery.artifact_hash and job.before_repair_text==job.text and
                          job.before_repair_execution_evidence_hash==job.execution_evidence_hash and
                          (job.execution_evidence_hash or not job.internal_repairs)):
                        before=grade
                    else:
                        raise ValueError('missing evaluation of a distinct pre-repair candidate')
                    if before is None:
                        raise ValueError('task still has an unevaluated pre-repair candidate')
                    replacements.append((trace,index,True,before))
        # Validate every receipt before changing any of the result objects.
        updates=[]
        for trace,index,before,grade in replacements:
            target=trace.before_repair_drafts if before else trace.deliveries
            updated=replace(target[index],native_score=grade.native_score,native_pass=grade.native_pass,
                preferences=grade.preferences,technical_missing=grade.technical_missing,scoring_pending=False)
            if not target[index].scoring_pending and target[index]!=updated:
                raise ValueError('an already evaluated delivery cannot be assigned a different grade')
            updates.append((trace,target,index,before,updated))
        for trace,target,index,before,updated in updates:
            target[index]=updated
            if not before:
                trace.irreversible_rule_ids.update(v.rule_id for v in updated.preferences if v.irreversible)
        return outcome
