"""Durable task outcomes, preserving pending grades and unanswered corrections."""
from dataclasses import asdict

from .interaction_loop import TaskOutcome,UnitOutcome
from .interaction_metrics import Delivery,PreferenceVerdict,UnitTrace


def outcome_to_dict(outcome):
    if not isinstance(outcome,TaskOutcome):
        raise ValueError('an actual task outcome is required')
    value=asdict(outcome)
    value['units']=list(value['units'])
    for unit in value['units']:
        trace=unit['trace']
        if trace is not None:
            trace['irreversible_rule_ids']=sorted(trace['irreversible_rule_ids'])
            for delivery in [*trace['deliveries'],*trace['before_repair_drafts']]:
                if delivery is not None:
                    delivery['preferences']=list(delivery['preferences'])
    # Reuse the same sequence validation before publishing and after reading.
    if outcome_from_dict(value)!=outcome:
        raise ValueError('task result cannot be represented without loss')
    return value


def outcome_from_dict(value):
    if (not isinstance(value,dict) or set(value)!={'task_id','session_id','units','aborted'} or
            any(not isinstance(value[k],str) or not value[k] for k in ('task_id','session_id')) or
            type(value['aborted']) is not bool or not isinstance(value['units'],list) or not value['units']):
        raise ValueError('invalid task outcome receipt')
    units,seen=[],set()
    def delivery(row):
        if not isinstance(row,dict) or set(row)!=set(Delivery.__dataclass_fields__) or not isinstance(row['preferences'],list):
            raise ValueError('invalid captured delivery receipt')
        return Delivery(**{**row,'preferences':tuple(PreferenceVerdict(**v) for v in row['preferences'])})
    for row in value['units']:
        if (not isinstance(row,dict) or set(row)!=set(UnitOutcome.__dataclass_fields__) or
                not isinstance(row['unit_id'],str) or not row['unit_id'] or row['unit_id'] in seen or
                any(not isinstance(row[k],str) for k in ('feedback_status','error_code'))):
            raise ValueError('invalid or duplicated scoring-unit outcome')
        seen.add(row['unit_id'])
        raw,trace=row['trace'],None
        if raw is not None:
            if (not isinstance(raw,dict) or set(raw)!=set(UnitTrace.__dataclass_fields__) or
                    raw['unit_id']!=row['unit_id'] or raw['task_id']!=value['task_id'] or
                    any(not isinstance(raw[k],list) for k in
                        ('deliveries','corrections','before_repair_drafts','irreversible_rule_ids')) or
                    not 1<=len(raw['deliveries'])<=3 or
                    len(raw['before_repair_drafts'])!=len(raw['deliveries']) or
                    len(raw['corrections']) not in {len(raw['deliveries'])-1,len(raw['deliveries'])}):
                raise ValueError('invalid task trace sequence or ownership')
            trace=UnitTrace(raw['unit_id'],raw['task_id'])
            for index,actual in enumerate(raw['deliveries']):
                draft=raw['before_repair_drafts'][index]
                trace.add_delivery(delivery(actual),before_repair=delivery(draft) if draft is not None else None)
                if index<len(raw['corrections']):
                    trace.add_correction(raw['corrections'][index])
            if raw['irreversible_rule_ids']!=sorted(trace.irreversible_rule_ids):
                raise ValueError('irreversible events must derive from actual delivered verdicts')
        units.append(UnitOutcome(row['unit_id'],trace,row['feedback_status'],row['error_code']))
    return TaskOutcome(value['task_id'],value['session_id'],tuple(units),value['aborted'])
