"""Summarize a fixed phase without dropping missing or unfinished samples.

This is read-only postprocessing. Callers verify run/phase receipts before
passing outcomes from exactly one namespace, split and method. It never reads
live model state, changes memory, or chooses the better of multiple attempts.
"""
import statistics

from .interaction_metrics import summarize_preferences, summarize_task_any_violation
from .task_outcomes import outcome_to_dict


def summarize_plan_a_outcomes(planned_tasks,outcomes,*,native_pass_supported=True):
    """One external delivery, with task-any-violation and missingness retained."""
    from .test_protocol import PLAN_A
    for task in outcomes:
        for unit in task.units:
            if unit.trace is not None and (unit.trace.corrections or len(unit.trace.deliveries)!=1 or
                                          unit.trace.before_repair_drafts!=[None]):
                raise ValueError('interactive results or privately scored drafts cannot enter a plan A table')
    summary=summarize_outcomes(planned_tasks,outcomes,native_pass_supported=native_pass_supported)
    return {'schema':1,'test_protocol':PLAN_A,'coverage':summary['coverage'],
            'delivery':summary['final'],'internal_repairs_observed':summary['internal_repairs_observed'],
            'units':[{**{k:v for k,v in row.items() if not k.startswith(('first_','final_','last_usable_'))
                         and k not in {'correction_count','feedback_status'}},
                      **{k.removeprefix('final_'):v for k,v in row.items() if k.startswith('final_')}}
                     for row in summary['units']],
            'interpretation':'One external delivery per original question. Native scores are unchanged; '
                'task-any-violation excludes inapplicable tasks and reports unresolved cases separately. '
                'Internal repairs are not user corrections or extra external deliveries.'}


def _delivery_summary(rows, prefix, native_pass_supported):
    total = len(rows)
    scores = [row[prefix+'_native_score'] for row in rows if row[prefix+'_native_score'] is not None]
    passes = [row[prefix+'_native_pass'] for row in rows if row[prefix+'_native_pass'] is not None]
    known, passed = len(passes), sum(passes)
    states = [row[prefix+'_preference'] for row in rows]
    task_rows = [{'unit_id': row['unit_id'], 'task_id': row['task_id'],
                  'first_preference': row[prefix+'_preference']} for row in rows]
    return {
        'native_score': {'known': len(scores), 'missing': total-len(scores),
                         'coverage': len(scores)/total,
                         'observed_mean': statistics.mean(scores) if scores else None},
        'native_pass': {'known': known, 'passed': passed, 'failed': known-passed,
                        'missing': total-known, 'coverage': known/total,
                        'observed_rate': passed/known if known else None,
                        'all_planned_lower_bound': passed/total,
                        'all_planned_upper_bound': (passed+total-known)/total}
                        if native_pass_supported else None,
        'preference': summarize_preferences(states),
        'task_any_violation': summarize_task_any_violation(task_rows, delivery='first'),
    }


def summarize_outcomes(planned_tasks, outcomes, *, native_pass_supported=True):
    """One phase's fixed whole-task IDs and validated, possibly partial results.

    Missing outcomes are retained as unknown, including entire unstarted tasks.
    A correction awaiting a response has no final result; the earlier usable
    delivery remains a separately labeled observation. Known irreversible
    violations still carry forward. Continuous native scores are not converted
    into pass/fail and are not imputed onto an unknown score scale.
    """
    if (not isinstance(planned_tasks, dict) or not planned_tasks or
            not isinstance(outcomes, (list, tuple)) or type(native_pass_supported) is not bool):
        raise ValueError('a fixed complete phase plan and explicit native scoring semantics are required')
    ownership = {}
    for task, units in planned_tasks.items():
        if (not isinstance(task, str) or not task or not isinstance(units, (list, tuple)) or not units or
                any(not isinstance(unit, str) or not unit for unit in units)):
            raise ValueError('every planned task needs its complete nonempty scoring-unit list')
        for unit in units:
            if unit in ownership:
                raise ValueError('duplicate planned unit changes the denominator')
            ownership[unit] = task
    observed, task_ids, aborted = {}, set(), 0
    for task in outcomes:
        outcome_to_dict(task)  # Reuse validated sequence and irreversible-evidence reconstruction.
        if task.task_id not in planned_tasks or task.task_id in task_ids:
            raise ValueError('foreign or duplicate task outcome')
        task_ids.add(task.task_id)
        aborted += int(task.aborted)
        for unit in task.units:
            if ownership.get(unit.unit_id) != task.task_id or unit.unit_id in observed:
                raise ValueError('foreign or duplicate scoring-unit outcome')
            if unit.trace is not None and unit.trace.scoring_pending:
                raise ValueError('pending scoring cannot enter a final experiment summary')
            observed[unit.unit_id] = unit
    rows, drafts = [], []
    corrections, repairs, unanswered, delivered = [], 0, 0, 0
    transitions = {key: 0 for key in ('improved', 'degraded', 'unchanged_pass', 'unchanged_fail', 'unknown_or_unanswered')}
    for unit_id, task_id in ownership.items():
        row = {'unit_id': unit_id, 'task_id': task_id}
        for prefix in ('first', 'final', 'last_usable'):
            row.update({prefix+'_native_score': None, prefix+'_native_pass': None,
                        prefix+'_preference': 'unknown'})
        unit = observed.get(unit_id)
        trace = unit.trace if unit else None
        row['status'] = 'missing_outcome' if unit is None else 'no_delivery'
        if trace is not None:
            metrics = trace.metrics()
            delivered += 1
            corrections.append(metrics['correction_count'])
            for index in range(len(trace.corrections)):
                before = trace.deliveries[index].native_pass
                after = trace.deliveries[index+1].native_pass if index+1 < len(trace.deliveries) else None
                if before is None or after is None:
                    transition = 'unknown_or_unanswered'
                elif before != after:
                    transition = 'improved' if after else 'degraded'
                else:
                    transition = 'unchanged_pass' if after else 'unchanged_fail'
                transitions[transition] += 1
            repairs += metrics['internal_repair_count']
            unanswered += int(metrics['correction_response_missing'])
            row['status'] = 'unanswered_correction' if metrics['correction_response_missing'] else 'delivered'
            for suffix in ('native_score', 'native_pass', 'preference'):
                row['first_'+suffix] = metrics['first_'+suffix]
                row['last_usable_'+suffix] = metrics['final_'+suffix]
                if not metrics['correction_response_missing']:
                    row['final_'+suffix] = metrics['final_'+suffix]
            if metrics['correction_response_missing'] and trace.irreversible_rule_ids:
                row['final_preference'] = 'fail'
            row.update(correction_count=metrics['correction_count'],
                       delivery_count=metrics['delivery_count'], internal_repair_count=metrics['internal_repair_count'],
                       feedback_status=unit.feedback_status, error_code=unit.error_code)
            drafts.extend({'unit_id': unit_id, 'task_id': task_id, **draft}
                          for draft in metrics['pre_delivery_repair'])
        rows.append(row)
    total = len(rows)
    return {'schema': 1, 'coverage': {
                'tasks_planned': len(planned_tasks), 'tasks_with_outcome': len(task_ids),
                'tasks_aborted': aborted, 'scoring_units_planned': total,
                'scoring_units_with_outcome': len(observed), 'scoring_units_with_delivery': delivered,
                'delivery_coverage': delivered/total, 'unanswered_corrections': unanswered},
            **{prefix: _delivery_summary(rows, prefix, native_pass_supported)
               for prefix in ('first', 'final', 'last_usable')},
            'corrections': {'observed_total': sum(corrections), 'units_with_observation': len(corrections),
                            'mean_per_observed_unit': statistics.mean(corrections) if corrections else None,
                            'missing_units': total-len(corrections)},
            'internal_repairs_observed': repairs,
            'correction_response_transitions': {**transitions,
                'interpretation': 'Observed native pass changes after corrections, not a causal or correctness judgment of feedback.'},
            'pre_delivery_repair_pairs': drafts, 'units': rows,
            'interpretation': 'Fixed phase denominator; last_usable is not a replacement for a missing final answer. '
                              'Pass bounds include missing planned units. Corrections exclude unobserved units.'}
