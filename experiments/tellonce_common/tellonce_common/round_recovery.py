"""Task-local round state, committed atomically with the phase memory checkpoint.

Only explicit JSON fields are serialized; never pickle a live client, lock,
credential, subprocess or model. Native conversation files stay host-private.
"""
import json
import shutil
from dataclasses import asdict
import importlib

from .atomic_io import atomic_write_bytes
from .provenance import canonical_json_bytes
from .operation_scope import RequestScopeState


def export_evidence(evidence):
    with evidence._lock:
        if evidence._pending:
            raise ValueError('cannot checkpoint pending native operations')
        return {**{key: (sorted(getattr(evidence,key)) if key in {'_seen','_completed'} else getattr(evidence,key))
                for key in ('_events','_seen','_completed','_used','_omitted','_sequence')},
                'request_scope_state': evidence._request_scope.export()}


def restore_evidence(evidence, state):
    scope = RequestScopeState(state.get('request_scope_state'))
    if 'request_scope_state' in state and state['request_scope_state'] is None:
        raise ValueError('present operation scope checkpoint cannot be null')
    for key in ('_events','_seen','_completed','_used','_omitted','_sequence'):
        value=state[key]
        setattr(evidence,key,set(map(tuple,value)) if key in {'_seen','_completed'} else value)
    evidence._request_scope = scope


def export_native(native):
    value={'native_thread_id':native.native_thread_id,'seen':sorted(native._seen),
           'evidence':export_evidence(native.execution_evidence)}
    guard=getattr(native,'_guard',None) or getattr(getattr(native,'_trace',None),'guard',None)
    if guard is not None:
        if guard._open or guard._pending_tools:
            raise ValueError('cannot checkpoint unfinished execution checks')
        value['guard']={'events':[asdict(e) for e in guard.events], 'sequence':guard._sequence,
                        'tool_ids':sorted(guard._tool_ids),'user_events':sorted(guard._user_events)}
    if hasattr(native,'_completed_turns'):
        value.update(completed_turns=native._completed_turns,usage=native._usage)
    else:
        paths=list((native.private_root/'cli/sessions').rglob('*'+native.native_thread_id+'*.jsonl'))
        if len(paths)!=1:
            raise ValueError('native round checkpoint requires its unique persisted rollout')
        value['rollout']=str(paths[0].relative_to(native.private_root/'cli'))
    return value


def restore_guard(guard, state):
    guard.events=[guard.event_type(**row) for row in state['events']]
    guard._sequence=state['sequence']
    guard._tool_ids=set(state['tool_ids']);guard._user_events=set(state['user_events'])


def export_memory(port):
    value={'seen':sorted(port._seen)}
    if hasattr(port,'_previous_answer'):
        value['previous_answer']=port._previous_answer
    if hasattr(port,'_context_ok'):
        value['context_ok']=port._context_ok
        value['publication']=port.runtime.publication.to_json()
        value['selection']=asdict(port.runtime.selection) if port.runtime.selection else None
    return value


def restore_memory(port, state):
    port._seen=set(state['seen'])
    # Recreate the bridge, preserving old receipts as immutable evidence.
    if 'previous_answer' in state:port._previous_answer=state['previous_answer']
    if 'context_ok' in state:port._context_ok=state['context_ok']
    if 'publication' in state:
        runtime=port.runtime
        runtime._publication=type(runtime.publication).from_json(state['publication'],expected_namespace=runtime.namespace)
        if state['selection'] is not None:
            module=importlib.import_module(type(runtime).__module__)
            raw=state['selection']['result']
            result=module.SelectionResult(**{k:tuple(v) if isinstance(v,list) else v for k,v in raw.items()})
            runtime._selection=module.BoundarySelection(state['selection']['publication_digest'],result)


def open_memory(owner, start, create):
    """Recreate a method port without replaying learning or model calls."""
    directory=owner.directory/'memory'
    prior=owner.directory/'memory-resuming'
    if owner.round_state is not None and directory.exists():
        directory.rename(prior)
    port=create()
    if prior.exists():
        shutil.copytree(prior,directory,dirs_exist_ok=True)
        shutil.rmtree(prior)
    if owner.round_state is not None:
        restore_memory(port,owner.round_state['memory'])
    return port


def load_round(owner, recovery):
    owner.recovery=recovery
    owner.round_state=None
    if recovery is None or not recovery.restored or not recovery.value.get('partial'):
        return
    partial=recovery.value['partial']
    if partial['task_id']!=owner.units[0].independent_task_id:
        return
    state=json.loads((owner.directory/'unit-resume.json').read_text())
    if (state['binding']!=owner.binding or state['namespace']!=owner.identity.namespace or
            state['loop']['start']['task_id']!=partial['task_id'] or
            len(state['loop']['outcome']['units'])!=partial['completed_units']):
        raise ValueError('round state does not match the admitted checkpoint')
    owner.round_state=state
    for event in state['evaluation_events']:
        owner.evaluation.observe(event)


def commit_round(owner, loop_state):
    if owner.recovery is None:return
    artifacts=owner.artifacts
    value={'binding':owner.binding,'namespace':owner.identity.namespace,'loop':loop_state,
           'native':export_native(owner.native),'memory':export_memory(owner.memory),
           'artifacts':{'baseline':artifacts._baseline,'baseline_hash':artifacts._baseline_hash,
                        'public_updates':artifacts._public_updates,'last':artifacts._last},
           'evaluation_events':owner.evaluation._events}
    atomic_write_bytes(owner.directory/'unit-resume.json',canonical_json_bytes(value))
    cp=owner.recovery
    cp.save(cp.value['completed'] if cp.value else [], memory=owner.memory_factory,
            partial={'task_id':loop_state['start']['task_id'],
                     'completed_units':len(loop_state['outcome']['units'])})
    owner.record({'type':'round_checkpoint_committed','namespace':owner.identity.namespace,
                  'task_id':loop_state['start']['task_id'],
                  'completed_units':len(loop_state['outcome']['units'])})
