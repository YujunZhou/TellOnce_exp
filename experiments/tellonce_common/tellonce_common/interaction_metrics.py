"""Delivery-level metrics kept separate from task correctness and repair drafts."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics

from .users import USERS as REAL_USERS

PREFERENCE_STATES = {'pass', 'fail', 'not_applicable', 'unknown', 'error'}


@dataclass(frozen=True)
class PreferenceVerdict:
    rule_id: str
    status: str
    irreversible: bool = False

    def __post_init__(self):
        if not self.rule_id or self.status not in PREFERENCE_STATES or type(self.irreversible) is not bool:
            raise ValueError('invalid attributed preference verdict')
        if self.irreversible and self.status != 'fail':
            raise ValueError('irreversible flag requires an observed violation')


def preference_state(verdicts: tuple[PreferenceVerdict, ...], *, irreversible_rule_ids=()) -> str:
    if len({v.rule_id for v in verdicts}) != len(verdicts):
        raise ValueError('one rule has conflicting verdicts at the same delivery')
    states = {v.status for v in verdicts}
    if irreversible_rule_ids or 'fail' in states:
        return 'fail'
    if states & {'unknown', 'error'}:
        return 'unknown'
    if 'pass' in states:
        return 'pass'
    return 'not_applicable'


@dataclass(frozen=True)
class Delivery:
    artifact_hash: str
    native_score: float | None
    native_pass: bool | None
    preferences: tuple[PreferenceVerdict, ...]
    internal_repairs: int = 0
    before_repair_artifact_hash: str = ''
    technical_missing: bool = False
    scoring_pending: bool = False

    def __post_init__(self):
        if not isinstance(self.artifact_hash, str) or not self.artifact_hash:
            raise ValueError('delivery must reference an actual captured artifact')
        if type(self.internal_repairs) is not int or not 0 <= self.internal_repairs <= 3:
            raise ValueError('each delivery permits at most three internal repair rounds')
        if self.internal_repairs and not self.before_repair_artifact_hash:
            raise ValueError('repairs require the captured pre-repair draft identity')
        if self.native_pass is not None and type(self.native_pass) is not bool:
            raise ValueError('native pass is a native boolean judgment, not a thresholded score')
        if self.native_score is not None and (type(self.native_score) not in {int, float} or not math.isfinite(self.native_score)):
            raise ValueError('native score must be finite or explicitly unavailable')
        if type(self.technical_missing) is not bool:
            raise ValueError('technical missing must be an explicit boolean')
        if (type(self.scoring_pending) is not bool or self.scoring_pending and
                (self.technical_missing or self.native_score is not None or self.native_pass is not None or self.preferences)):
            raise ValueError('pending scoring cannot pretend to have an observed result')
        if self.technical_missing and (self.native_pass is not None or self.native_score is not None):
            raise ValueError('technical missing cannot pretend to be an observed task score')
        if not isinstance(self.preferences, tuple):
            raise ValueError('preference verdicts must be an immutable tuple')
        preference_state(self.preferences)


@dataclass
class UnitTrace:
    unit_id: str
    task_id: str
    deliveries: list[Delivery] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    irreversible_rule_ids: set[str] = field(default_factory=set)
    before_repair_drafts: list[Delivery | None] = field(default_factory=list)

    def add_delivery(self, delivery: Delivery, *, before_repair: Delivery | None = None):
        if not self.unit_id or not self.task_id:
            raise ValueError('scoring unit and independent task identities are required')
        if len(self.deliveries) != len(self.corrections) or len(self.deliveries) >= 3:
            raise ValueError('delivery needs the preceding user turn; repairs are not extra deliveries')
        if before_repair is not None and (before_repair.artifact_hash != delivery.before_repair_artifact_hash or
                before_repair.internal_repairs or before_repair.before_repair_artifact_hash):
            raise ValueError('the draft must be the captured pre-repair artifact from this same delivery')
        self.deliveries.append(delivery)
        self.before_repair_drafts.append(before_repair)
        self.irreversible_rule_ids.update(v.rule_id for v in delivery.preferences if v.irreversible)

    def add_correction(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('count only an actual nonempty user correction')
        if len(self.corrections) >= 2 or len(self.deliveries) != len(self.corrections)+1:
            raise ValueError('each scoring unit allows two corrections after delivered answers')
        self.corrections.append(text)

    @property
    def scoring_pending(self):
        return any(delivery.scoring_pending for delivery in [*self.deliveries,*self.before_repair_drafts]
                   if delivery is not None)

    def metrics(self):
        if self.scoring_pending:
            raise ValueError('private scoring is pending; no final metrics are available')
        if not self.deliveries:
            raise ValueError('no delivery exists; record infrastructure failure separately')
        first, final = self.deliveries[0], self.deliveries[-1]
        pending_response = len(self.corrections) == len(self.deliveries)
        return {'unit_id': self.unit_id, 'task_id': self.task_id,
                'first_native_score': first.native_score, 'first_native_pass': first.native_pass,
                'final_native_score': final.native_score, 'final_native_pass': final.native_pass,
                'first_preference': preference_state(first.preferences),
                'final_preference': preference_state(final.preferences, irreversible_rule_ids=self.irreversible_rule_ids),
                'correction_count': len(self.corrections),
                'internal_repair_count': sum(d.internal_repairs for d in self.deliveries),
                'delivery_count': len(self.deliveries),
                'pre_delivery_repair':[
                    {'delivery_index':index,'before_artifact_hash':draft.artifact_hash,
                     'after_artifact_hash':actual.artifact_hash,
                     'before_native_score':draft.native_score,'before_native_pass':draft.native_pass,
                     'before_preference':preference_state(draft.preferences),
                     'after_native_score':actual.native_score,'after_native_pass':actual.native_pass,
                     'after_preference':preference_state(actual.preferences),
                     'before_technical_missing':draft.technical_missing,'after_technical_missing':actual.technical_missing}
                    for index,(draft,actual) in enumerate(zip(self.before_repair_drafts,self.deliveries))
                    if draft is not None],
                'correction_response_missing': pending_response,
                'final_is_last_usable_delivery': pending_response,
                'technical_missing': final.technical_missing or pending_response}


def summarize_preferences(states: list[str]) -> dict:
    if any(s not in PREFERENCE_STATES for s in states):
        raise ValueError('invalid preference state')
    counts = {s: states.count(s) for s in ('pass', 'fail', 'not_applicable', 'unknown')}
    counts['unknown'] += states.count('error')
    eligible = counts['pass'] + counts['fail'] + counts['unknown']
    resolved = counts['pass'] + counts['fail']
    # Unknowns are never silently counted as compliant. Main tables must show
    # the denominator and coverage alongside the rate; bounds retain unknowns.
    return {**counts, 'total': len(states), 'eligible_or_unknown': eligible,
            'resolved_violation_rate': counts['fail']/resolved if resolved else None,
            'determination_coverage': resolved/eligible if eligible else None,
            'violation_rate_lower_bound': counts['fail']/eligible if eligible else None,
            'violation_rate_upper_bound': (counts['fail']+counts['unknown'])/eligible if eligible else None}


def summarize_task_any_violation(unit_metrics: list[dict], *, delivery: str) -> dict:
    if delivery not in {'first', 'final'}:
        raise ValueError('select first or final delivered answers explicitly')
    grouped, seen = {}, set()
    for unit in unit_metrics:
        if unit['unit_id'] in seen:
            raise ValueError('duplicate scoring unit would distort the task denominator')
        seen.add(unit['unit_id'])
        state = unit[delivery+'_preference']
        if state not in PREFERENCE_STATES:
            raise ValueError('invalid preference state in a native scoring unit')
        grouped.setdefault(unit['task_id'], []).append(state)
    task_states = []
    for states in grouped.values():
        task_states.append('fail' if 'fail' in states else 'unknown' if set(states) & {'unknown', 'error'}
                           else 'pass' if 'pass' in states else 'not_applicable')
    return summarize_preferences(task_states)


def summarize_three_users(values: dict[str, float | None]) -> dict:
    if set(values) != set(REAL_USERS):
        raise ValueError('report exactly the three real users; generic performance has no three-user error bar')
    if any(v is not None and (type(v) not in {int, float} or not math.isfinite(v)) for v in values.values()):
        raise ValueError('per-user metrics must be finite or explicitly missing')
    complete = all(v is not None for v in values.values())
    numbers = list(values.values())
    return {'users': dict(values), 'complete': complete,
            'equal_user_mean': statistics.mean(numbers) if complete else None,
            'sample_sd_across_users': statistics.stdev(numbers) if complete else None,
            'interpretation': 'Three observed users, not repeated seeds or a confidence interval.'}


def interval_wall_seconds(intervals: list[tuple[float, float]]) -> float:
    """Union intervals on one clock for one phase/component, not a CPU sum."""
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError('timing intervals require a common clock and ordered finite endpoints')
    total, previous_end = 0., 0.
    for start, end in sorted(intervals):
        total += max(0., end-max(start, previous_end))
        previous_end = max(previous_end, end)
    return total
