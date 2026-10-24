"""Revisioned private scores: recover proven native outages, retain judgments."""
from dataclasses import asdict
import json

from .deferred_scoring import DeferredScoring, _write
from .interaction_loop import Grade, SolverAnswer
from .interaction_metrics import PreferenceVerdict
from .provenance import content_hash

REVISION = 'delivery-v6'


class PreferenceRepairQueue(DeferredScoring):
    """Keep the original sealed jobs/lock; store new grades in a separate owner.

    Reassess pre-v3 preferences with the fixed rubric; retain v3/v4/v5 judgments
    regardless of their label. Only a matched native-outage proof reopens
    a missing v3/v4 judgment. New receipts, including unknowns, are never
    resampled. Native grades and native missingness are retained.
    """
    def __init__(self, queue):
        if not isinstance(queue, DeferredScoring) or queue._sealed is None:
            raise ValueError('preference repair needs a sealed original scoring queue')
        self.__dict__.update(queue.__dict__)
        from .preference_outage_recovery import load
        self.outage_recovery={revision:load(self,revision) for revision in ('delivery-v3','delivery-v4','delivery-v5')}
        self.repair_directory=self.directory/('preference-'+REVISION)
        self.repair_directory.mkdir(mode=0o700,exist_ok=True)
        if self.repair_directory.is_symlink():
            raise ValueError('preference repair cannot follow a redirected directory')
        owner={'schema':1,'revision':REVISION,'seal':self._sealed,'manifest':self._manifest}
        path=self.repair_directory/'owner.json'
        if path.exists():
            if json.loads(path.read_text())!=owner:
                raise ValueError('preference repair owner changed')
        else:
            _write(path,owner)

    @staticmethod
    def needs_repair(grade):
        return grade is not None

    def _prior_result(self,key):
        old=super()._result(key)
        for revision in ('delivery-v2','delivery-v3','delivery-v4','delivery-v5'):
            old=self._prior_revision(key,old,revision)
        return old

    def _prior_revision(self,key,old,revision):
        directory=self.directory/('preference-'+revision)
        path=directory/(key+'.json')
        if not path.exists():
            return old
        owner={'schema':1,'revision':revision,'seal':self._sealed,'manifest':self._manifest}
        if directory.is_symlink() or path.is_symlink() or json.loads((directory/'owner.json').read_text())!=owner:
            raise ValueError('prior preference owner changed')
        value=json.loads(path.read_text())
        body={k:v for k,v in value.items() if k!='digest'}
        if (set(value)!={'request_hash','original_grade_hash','grade','digest'} or
                value['digest']!=content_hash(body) or value['request_hash']!=self._jobs[key]['request_hash'] or
                value['original_grade_hash']!=content_hash(asdict(old) if old is not None else None)):
            raise ValueError('prior preference receipt changed')
        prior=Grade(**{**value['grade'],'preferences':tuple(PreferenceVerdict(**v) for v in value['grade']['preferences'])})
        self._validate_repair(old,prior)
        return prior

    def _result(self,key):
        old=self._prior_result(key)
        path=self.repair_directory/(key+'.json')
        if not path.exists():
            for revision in ('delivery-v5','delivery-v4','delivery-v3'):
                prior=self.directory/('preference-'+revision)/(key+'.json')
                if not prior.exists():continue
                recovery=self.outage_recovery[revision].get(key)
                if recovery is None:
                    # Same rubric: retain all valid judgments, actual unknowns,
                    # and exhausted format failures. Only proven outages retry.
                    return old
                value=json.loads(prior.read_text())
                if (value['digest']!=recovery['receipt_digest'] or
                        not any(v.rule_id=='__preference_evaluation_unavailable__' for v in old.preferences)):
                    raise ValueError('native-outage recovery does not match the failed preference receipt')
                break
            return None if self.needs_repair(old) else old
        if path.is_symlink():
            raise ValueError('preference receipt cannot follow a symlink')
        value=json.loads(path.read_text())
        body={k:v for k,v in value.items() if k!='digest'}
        if (set(value)!={'request_hash','original_grade_hash','grade','digest'} or
                value['digest']!=content_hash(body) or
                value['request_hash']!=self._jobs[key]['request_hash'] or
                value['original_grade_hash']!=content_hash(asdict(old) if old is not None else None)):
            raise ValueError('preference repair receipt changed or lost its original grade')
        grade=Grade(**{**value['grade'],'preferences':tuple(PreferenceVerdict(**v) for v in value['grade']['preferences'])})
        self._validate_repair(old,grade)
        return grade

    @staticmethod
    def _validate_repair(old,grade):
        if not isinstance(grade,Grade) or grade.scoring_pending:
            raise ValueError('completed independent grade required')
        if old is not None and (grade.native_score,grade.native_pass,grade.technical_missing)!=(
                old.native_score,old.native_pass,old.technical_missing):
            raise ValueError('preference repair cannot alter a native grade')

    def _evaluate_owned(self,score):
        for key,row in self._jobs.items():
            self.authorize('evaluate',{**self._manifest,'seal':self._sealed,'call_id':key})
            if self._result(key) is not None:
                continue
            old=self._prior_result(key)
            grade=score(row['unit_id'],SolverAnswer(**row['answer']),context=dict(row['context']),previous_grade=old)
            self._validate_repair(old,grade)
            body={'request_hash':row['request_hash'],'original_grade_hash':content_hash(asdict(old) if old is not None else None),
                  'grade':asdict(grade)}
            _write(self.repair_directory/(key+'.json'),{**body,'digest':content_hash(body)})
            self.record({'type':'private_score_completed','namespace':self.identity.namespace,'revision':REVISION,
                         'context':row['context'],'unit_id':row['unit_id'],'grade':asdict(grade)})
