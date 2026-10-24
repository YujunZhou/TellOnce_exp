"""Fail closed when preserved training failures reach a new test comparison.

Freezing evidence and completing learning are separate operations. This check
does not retry, repair, change a denominator, or judge semantic memory quality.
"""
from collections import Counter


def require_baseline_learning(audit, *, namespace, binding):
    if (not isinstance(audit,dict) or audit.get('namespace')!=namespace or
            audit.get('binding')!=binding or not isinstance(audit.get('counts'),dict) or
            not isinstance(audit.get('events'),list)):
        raise ValueError('missing or foreign learning completion evidence')
    counts,events=audit['counts'],audit['events']
    states={'done','failed','pending','running'}
    if any(k not in states or type(v) is not int or v<0 for k,v in counts.items()):
        raise ValueError('invalid learning completion counts')
    seen,actual=set(),Counter()
    for event in events:
        if (not isinstance(event,dict) or not isinstance(event.get('event_id'),str) or
                not event['event_id'] or event['event_id'] in seen or event.get('status') not in states or
                (event['status']=='done' and event.get('error_code') is not None)):
            raise ValueError('invalid learning completion events')
        seen.add(event['event_id']);actual[event['status']]+=1
    if actual!=Counter(counts):
        raise ValueError('learning completion counts differ from events')
    if any(actual[k] for k in ('failed','pending','running')):
        raise ValueError('incomplete learning cannot enter a new test comparison')


def require_trace_learning(audit, *, namespace, publication_digest, admitted=None):
    """Fail closed by default. `admitted` is an explicit, reviewed copy of the exact frozen audit
    (namespace, publication digest, turn outcomes, pending rule ids) that a runtime revision admits
    as evidence into the comparison: only bounded learner failures ('failed' turns, whose model
    requests the library already refused to retry) and compiler-pending rules (excluded from the
    active snapshot) can be admitted this way; unprocessed learning never can."""
    if (not isinstance(audit,dict) or audit.get('namespace')!=namespace or
            audit.get('publication_digest')!=publication_digest or
            not isinstance(audit.get('turn_outcomes'),dict) or
            not isinstance(audit.get('pending_rule_ids'),list)):
        raise ValueError('missing or foreign TRACE learning completion evidence')
    completed={'projected','noop','clarified','dismissed','rejected'}
    counts=audit['turn_outcomes']
    known=completed|{'pending','resolving','committed','needs_user','failed'}
    if any(k not in known or type(v) is not int or v<0 for k,v in counts.items()):
        raise ValueError('invalid TRACE learning completion counts')
    incomplete={k:v for k,v in counts.items() if v and k not in completed}
    if not incomplete and not audit['pending_rule_ids']:
        return
    if admitted is None:
        raise ValueError('incomplete learning cannot enter a new test comparison')
    expected={'namespace':namespace,'publication_digest':publication_digest,
              'turn_outcomes':{k:v for k,v in counts.items() if v},'pending_rule_ids':list(audit['pending_rule_ids'])}
    if (not isinstance(admitted,dict) or set(admitted)!=set(expected) or
            any(admitted[k]!=expected[k] for k in expected)):
        raise ValueError('admitted learning evidence differs from the frozen audit')
    if any(k!='failed' for k in incomplete):
        raise ValueError('unprocessed learning cannot be admitted as evidence')
