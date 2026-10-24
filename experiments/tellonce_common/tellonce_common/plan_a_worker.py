"""Test-only entry point; reuse the native task owners and round recovery."""
from contextlib import nullcontext
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

from .plan_a_spec import PlanARunSpec
from .native_availability import availability_fields
from .provenance import canonical_json_bytes, content_hash
from .phase_signals import PhaseSignals
from .run_lock import run_lock


def execute(spec, *, action, prior_receipt, task_id, signals, record, attempt):
    if action not in {'test_interaction','test_evaluation'}:
        raise ValueError('plan A cannot train or alter a prior experiment')
    if action=='test_interaction' and prior_receipt!=spec.extension['training_receipt']:
        raise ValueError('plan A request has another training receipt')
    if action=='test_evaluation':
        from .plan_a_evaluation import evaluate_extension
        return evaluate_extension(spec,prior_receipt=prior_receipt,signals=signals,record=record,attempt=attempt)
    if spec.identity.benchmark=='clawarena':
        from tellonce_clawarena.native_worker import run_phase
    else:
        from tellonce_memoryarena.native_travel_worker import run_phase
    return run_phase(spec,action=action,prior_receipt=prior_receipt,test_task_id=task_id,
                     signals=signals,record=record,attempt_directory=attempt)


def main():
    worker_started = (time.monotonic(), time.time())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['worker','preflight','test','evaluate'])
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--digest')
    args=parser.parse_args()
    try:
        if args.action=='worker':
            raw=sys.stdin.buffer.read(1024*1024+1)
            if len(raw)>1024*1024:
                raise ValueError('bounded plan A worker request required')
            request=json.loads(raw);payload=request['payload']
            if (request.get('job_id')!='test_interaction' or request.get('supervisor_attempt')!=1 or
                    payload.get('action')!='test_interaction' or
                    set(payload) not in ({'spec_path','spec_hash','action','prior_receipt'},
                                         {'spec_path','spec_hash','action','prior_receipt','test_task_id'})):
                raise ValueError('plan A worker can only receive a test interaction request')
            spec=PlanARunSpec.load(path=payload['spec_path'],expected_digest=payload['spec_hash'])
            if request.get('namespace')!=spec.identity.namespace or request.get('binding')!=spec.binding:
                raise ValueError('worker request belongs to another frozen memory owner')
            action='test_interaction';prior=payload['prior_receipt'];task_id=payload.get('test_task_id')
        else:
            if args.spec is None or args.digest is None:
                raise ValueError('an explicit pinned plan A specification is required')
            spec=PlanARunSpec.load(path=args.spec,expected_digest=args.digest)
            if args.action=='preflight':
                from tellonce_clawarena.native_worker import verify_execution
                from tellonce_memoryarena.native_travel_worker import verify_execution as verify_memory_execution
                verify_execution(spec);verify_memory_execution(spec)
                print(json.dumps({'status':'passed','protocol':spec.test_protocol,
                    'tasks':len(spec.value['tasks']),'method':spec.identity.method,
                    'source_memory_receipt':spec.extension['memory_receipt'],
                    'output':str(spec.directory)}))
                return 0
            action='test_interaction' if args.action=='test' else 'test_evaluation';task_id=None
            prior=(spec.extension['training_receipt'] if args.action=='test' else
                   content_hash(json.loads((spec.directory/'test-phase/phase-result.json').read_text())))
        spec.verify()
        spec.directory.mkdir(mode=0o700,parents=True,exist_ok=True)
        manifest=spec.directory/'plan-a-owner.json'
        ownership={'spec_digest':spec.spec_digest,'source_binding':spec.binding,
                   'test_protocol':spec.test_protocol,'profile_hash':spec.extension['profile_hash']}
        from .deferred_scoring import _write
        if manifest.exists():
            if json.loads(manifest.read_text())!=ownership:
                raise ValueError('result directory belongs to another protocol or test plan')
        elif task_id is not None or any(spec.directory.iterdir()):
            raise ValueError('new plan A result directory must be empty before parent initialization')
        else:
            _write(manifest,ownership)
        (spec.directory/'worker-attempts').mkdir(mode=0o700,exist_ok=True)
        with run_lock(spec) if task_id is None else nullcontext():
            events=spec.directory/'worker-attempts'/(uuid.uuid4().hex+'.jsonl')
            lock=threading.Lock()
            signals=PhaseSignals()
            from .test_timing import AttemptTiming, termination_as_exception
            with termination_as_exception(), AttemptTiming(spec.directory/'timing.jsonl',
                    namespace=spec.identity.namespace, spec_digest=spec.spec_digest,
                    phase=action, task_id=task_id, started=worker_started) as timing, events.open('xb') as stream:
                def record(event):
                    with lock:
                        signals.observe(event)
                        timing.observe(event)
                        stream.write(canonical_json_bytes(event)+b'\n');stream.flush()
                result=execute(spec,action=action,prior_receipt=prior,task_id=task_id,
                               signals=signals,record=record,attempt=events.with_suffix(''))
                signals.assert_healthy()
            print(json.dumps({'type':'result','value':result}),flush=True)
        return 0
    except Exception as exc:
        frames=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}
                for f in traceback.extract_tb(exc.__traceback__)[-12:]]
        print(json.dumps({'type':'plan_a_worker_failed','error_type':type(exc).__name__,
            **availability_fields(exc),'frames':frames}),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
