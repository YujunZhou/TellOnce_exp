"""Independent test processes with task-local recovery and original result receipts.

The parent owns the run lock. Children share only frozen method data; task
directories keep their original paths so native round recovery remains valid.
No training work is dispatched here.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from .atomic_io import atomic_write_bytes
from .deferred_scoring import DeferredScoring, _write
from .provenance import canonical_json_bytes, content_hash, file_hash
from .task_checkpoint import TaskCheckpoint
from .process_owner import owned_process_alive, process_identity
from . import host_launch_gate


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    atomic_write_bytes(Path(path), canonical_json_bytes(value))


def task_map(spec):
    claw = spec.identity.benchmark == 'clawarena'
    subset = 'group_travel_planner'
    return {(r['scenario_id'] if claw else subset+':'+r['row_id']):
            ('task-'+r['scenario_id'] if claw else 'task-'+content_hash(subset+':'+r['row_id']))
            for r in spec.value['tasks'] if r['split'] != 'acquisition'}


def scope_paths(spec, task_id):
    if task_id not in task_map(spec):
        raise ValueError('parallel child must be an admitted complete test task')
    root = spec.directory/'parallel-test'/content_hash(task_id)
    if root.resolve() != root:
        raise ValueError('parallel task directory redirected')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root/'phase', root/'queue', root/'roles'


def task_recovery(spec, task_id):
    checkpoint = TaskCheckpoint(spec.directory, namespace=spec.identity.namespace, binding=spec.binding,
                                phase='test', task_directories={task_id: task_map(spec)[task_id]}, test_scope=task_id)
    return checkpoint


def worker_count(spec):
    reference = getattr(spec, 'runtime_bundle_reference', spec.value['source_bundle'])
    value = reference.get('test_parallelism', reference.get('test_task_workers', 1))
    if type(value) is not int or not 1 <= value <= 16:
        raise ValueError('test task concurrency must be an explicit integer between 1 and 16')
    return value


def has_parallel_state(spec):
    return (spec.directory/'parallel-test/manifest.json').is_file()


def phase_plan(spec):
    claw = spec.identity.benchmark == 'clawarena'
    rows = []
    for row in spec.value['tasks']:
        if row['split'] == 'acquisition':
            continue
        tid = row['scenario_id'] if claw else 'group_travel_planner:'+row['row_id']
        units = spec.units[tid]
        planned = {'task_id': tid, 'directory': str(spec.directory/task_map(spec)[tid]),
                   'units_hash': content_hash([asdict(u) for u in units]),
                   'unit_ids': [u.instance.common_instance_id for u in units]}
        if claw:
            public = getattr(spec, 'test_public_asset_digests', None) or getattr(spec, 'extension', {}).get('public_asset_digests', {})
            planned.update(asset_digest=public.get(tid, row['asset_digest']), evaluation_digest=row['evaluation_digest'],
                           profile_digest=content_hash(asdict(spec.profile)))
        else:
            planned.update(row_hash=content_hash(spec.rows[tid]), profile_hash=spec.profile_hash,
                           actual_profile_hash=spec.profile_hash)
        rows.append(planned)
    return {'schema': 1, 'namespace': spec.identity.namespace, 'phase': 'test', 'binding': spec.binding,
            'queue_directory': str(spec.directory/'test-queue'), 'tasks': rows}


def _publish(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('parallel completion receipt changed')
    else:
        _write(path, value)


def finish_sealed_phase(spec, phase, queue_dir, authorize, record):
    """Recover death between queue sealing and final receipt publication."""
    if (phase/'phase-result.json').exists() or not (queue_dir/'seal.json').exists():
        return
    manifest, closed, seal = read(phase/'phase.json'), read(phase/'closed-phase.json'), read(queue_dir/'seal.json')
    if closed['manifest_hash'] != content_hash(manifest) or seal['phase_receipt'] != content_hash(closed):
        raise ValueError('incomplete sealed phase has inconsistent receipts')
    DeferredScoring.open_sealed(directory=queue_dir, identity=spec.identity, phase='test',
        binding=spec.binding, expected_seal=content_hash(seal), authorize=authorize, record=record)
    _publish(phase/'phase-result.json', {'schema': 1, 'manifest_hash': content_hash(manifest),
             'closed_hash': content_hash(closed), 'queue_seal': content_hash(seal)})


def _initialize(spec, prior, authorize, record):
    root = spec.directory/'parallel-test'
    root.mkdir(mode=0o700, exist_ok=True)
    cp = TaskCheckpoint(spec.directory, namespace=spec.identity.namespace, binding=spec.binding,
                        phase='test', task_directories=task_map(spec))
    manifest = root/'manifest.json'
    successor_receipt = (getattr(spec, 'memory_successor_receipt', {}).get('frozen_method_receipt')
                         if 'memory_successor' in getattr(spec, 'extension', {}) else None)
    expected_memory = None if prior is None else prior.get('memory_receipt')
    if successor_receipt is not None and (expected_memory != successor_receipt or
            getattr(spec, 'prior', {}).get('memory_receipt') != successor_receipt):
        raise ValueError('Plan A curated successor must be the exact frozen memory passed to the worker')
    if manifest.exists():
        state = read(manifest)
        if (state['binding'] != spec.binding or
                state.get('memory_successor_receipt') != successor_receipt):
            raise ValueError('parallel test plan changed during recovery')
        if state['plan'] != phase_plan(spec):
            state = _recover_public_asset_plan(spec, state, manifest, record)
        return cp, state
    # Exactly once: no children have been admitted until this marker publishes.
    cp.restore()
    plan = phase_plan(spec)
    phase = spec.directory/'test-phase'
    phase.mkdir(mode=0o700, exist_ok=True)
    _publish(phase/'phase.json', plan)
    if not (spec.directory/'test-queue').exists():
        DeferredScoring(directory=spec.directory/'test-queue', identity=spec.identity, phase='test',
                        binding=spec.binding, authorize=authorize, record=record)
    if cp.current() is None:
        cp.save([])
    completed = list(cp.value['completed'])
    state = {'schema': 1, 'binding': spec.binding, 'plan': plan, 'completed_before_parallel': completed,
             'partial_before_parallel': cp.value.get('partial'),
             'memory_receipt': None if prior is None else prior['memory_receipt'],
             'started_at': time.time()}
    if successor_receipt is not None:
        state['memory_successor_receipt'] = successor_receipt
    # The immutable migration marker prevents a later whole-phase rollback.
    _write(manifest, state)
    return cp, state


def _recover_public_asset_plan(spec, state, manifest, record):
    """Repair the old parent-only digest bug without changing child evidence."""
    current = phase_plan(spec)
    legacy = deepcopy(current)
    admitted = getattr(spec, 'test_public_asset_digests', None) or getattr(spec, 'extension', {}).get('public_asset_digests', {})
    phase = spec.directory/'test-phase'
    if (spec.identity.benchmark != 'clawarena' or not admitted or
            getattr(spec, 'test_protocol', None) != 'original-input-only-v1' or
            state['completed_before_parallel'] or
            any((phase/name).exists() for name in ('closed-phase.json', 'phase-result.json')) or
            (spec.directory/'test-queue/seal.json').exists()):
        raise ValueError('parallel test plan changed during recovery')
    original = {row['scenario_id']: row['asset_digest'] for row in spec.value['tasks']}
    for row in legacy['tasks']:
        row['asset_digest'] = original[row['task_id']]
    if state['plan'] != legacy or read(phase/'phase.json') not in (legacy, current):
        raise ValueError('parallel recovery differs beyond admitted public asset digests')
    for row in current['tasks']:
        child_phase = spec.directory/'parallel-test'/content_hash(row['task_id'])/'phase/phase.json'
        if child_phase.exists():
            expected = {**current, 'tasks': [row],
                        'queue_directory': str(child_phase.parent.parent/'queue')}
            if read(child_phase) != expected:
                raise ValueError('parallel child did not use the admitted public materials')
    evidence = {'schema': 1, 'original_state': state, 'repaired_plan': current}
    _publish(manifest.with_name('public-asset-plan-recovery.json'), evidence)
    # Each replacement is atomic. If interrupted between them, the exact old
    # marker plus the already repaired parent plan are accepted on retry.
    save(phase/'phase.json', current)
    repaired = {**state, 'plan': current}
    save(manifest, repaired)
    record({'type': 'parallel_parent_public_assets_recovered', 'binding': spec.binding,
            'before': content_hash(legacy), 'after': content_hash(current)})
    return repaired


def _prepare_child(spec, task_id, state, authorize, record):
    phase, queue_dir, roles = scope_paths(spec, task_id)
    cp = task_recovery(spec, task_id)
    if cp.current() is not None or (phase/'phase-result.json').exists():
        return
    source = spec.directory/'roles'
    partial = state['partial_before_parallel']
    # Only the one resumed legacy task needs its old text-request receipts.
    # Fresh tasks create empty task-owned role journals in the child; copying
    # all training journals per test task wastes disk and is unnecessary.
    if partial and partial['task_id'] == task_id and source.exists() and not roles.exists():
        staging = roles.with_name('roles-preparing')
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(mode=0o700)
        shutil.copy2(source/'roles.json', staging/'roles.json')
        for role in spec.value['models']['roles']:
            dest = staging/role
            dest.mkdir(mode=0o700)
            (dest/'attempts').mkdir(mode=0o700)
            shutil.copytree(source/role/'journal', dest/'journal')
            if role == 'execution_check' and (source/role/'runtime-journal').exists():
                shutil.copytree(source/role/'runtime-journal', dest/'runtime-journal')
        staging.rename(roles)
    factory = DeferredScoring.open_unsealed if queue_dir.exists() else DeferredScoring
    factory(directory=queue_dir, identity=spec.identity, phase='test',
            binding=spec.binding, authorize=authorize, record=record)
    base = DeferredScoring.open_unsealed(directory=spec.directory/'test-queue', identity=spec.identity,
                     phase='test', binding=spec.binding, authorize=authorize, record=record)
    for key, row in base._jobs.items():
        if row['context']['task_id'] == task_id:
            _publish(queue_dir/'jobs'/(key+'.json'), row)
    if partial and partial['task_id'] == task_id:
        cp.save([], partial=partial)
    else:
        cp.save([])


def _launch_child(spec, task_id, prior_receipt, signals, record):
    phase, _, _ = scope_paths(spec, task_id)
    root = phase.parent
    old_process = root/'process.json'
    if old_process.exists():
        value = read(old_process)
        if owned_process_alive(value['pid'], value['start']):
            raise RuntimeError('prior task worker still alive; refuse duplicate dispatch')
    project = Path(spec.bundle['roots']['project'])
    if getattr(spec, 'test_protocol', None) != 'original-input-only-v1':
        raise ValueError('parallel test workers run only the frozen-memory test protocol')
    launcher = project/'experiments/tellonce_common/run_plan_a.py'
    wrapper = Path(__file__).with_name('native_exec_worker.py')
    command = [str(spec.python), '-I', '-B', str(wrapper), '--owner-pid', str(os.getpid()), '--',
               str(spec.python), '-I', '-B', str(launcher), 'worker']
    request = child_request(spec, task_id, prior_receipt)
    for attempt in range(3):
        signals.assert_healthy()
        finish_sealed_phase(spec, phase, root/'queue', lambda *a: signals.assert_healthy(), record)
        if (phase/'phase-result.json').exists():
            return True
        token = uuid.uuid4().hex
        with (root/(token+'.stdout')).open('xb') as out, (root/(token+'.stderr')).open('xb') as err:
            _wait_for_memory_admission(signals, record, task_id, attempt+1)
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, start_new_session=True)
            identity = process_identity(proc.pid)
            save(old_process, {'pid': proc.pid, 'start': identity['start'] if identity else '',
                               'task_id': task_id, 'started_at': time.time(), 'attempt': attempt+1})
            try:
                proc.stdin.write(canonical_json_bytes(request)); proc.stdin.close()
                deadline = time.monotonic()+spec.value['phase_timeout_s']
                while proc.poll() is None:
                    signals.assert_healthy()
                    if time.monotonic() >= deadline:
                        raise TimeoutError('parallel task worker exceeded its phase budget')
                    time.sleep(.5)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        record({'type': 'parallel_test_child_exit', 'task_id': task_id, 'exit_code': proc.returncode,
                'attempt': attempt+1, 'namespace': spec.identity.namespace})
        if (phase/'phase-result.json').exists():
            return True
        if attempt < 2:
            signals.wait_retry(5, context={'task_id': task_id}, reason='independent_test_retry')
    return False


def child_request(spec, task_id, prior_receipt):
    if task_id not in task_map(spec):
        raise ValueError('child request is outside the admitted test split')
    return {'namespace': spec.identity.namespace, 'binding': spec.binding, 'job_id': 'test_interaction',
               'supervisor_attempt': 1, 'heartbeat_interval_s': 10,
               'payload': {'spec_path': str(spec.path), 'spec_hash': getattr(spec, 'spec_digest', spec.binding), 'action': 'test_interaction',
                           'prior_receipt': prior_receipt, 'test_task_id': task_id}}


def _wait_for_memory_admission(signals, record, task_id, attempt):
    result = host_launch_gate.wait_for_admission(
        continue_waiting=signals.assert_healthy,
        sleep=lambda seconds: (signals.assert_healthy(), time.sleep(seconds))[1])
    if result['waited']:
        record({'type': 'memory_launch_gate_waited', 'task_id': task_id,
                'attempt': attempt, 'available_bytes': result['available_bytes']})


def _open_child(spec, task_id, authorize, record):
    phase, _, _ = scope_paths(spec, task_id)
    if spec.identity.benchmark == 'clawarena':
        from tellonce_clawarena.native_phase import open_claw_phase as open_phase
    else:
        from tellonce_memoryarena.native_travel_phase import open_travel_phase as open_phase
    receipt = content_hash(read(phase/'phase-result.json'))
    manifest, closed, queue, tasks = open_phase(directory=phase, expected_receipt=receipt,
        identity=spec.identity, binding=spec.binding, authorize=authorize, record=record)
    expected = next(row for row in phase_plan(spec)['tasks'] if row['task_id'] == task_id)
    if manifest['tasks'] != [expected] or closed['memory_receipt'] != read(spec.directory/'parallel-test/manifest.json')['memory_receipt']:
        raise ValueError('parallel child used a different task or frozen memory')
    return closed['tasks'][0], queue


def run_parallel_test(spec, *, prior, prior_receipt, signals, record, authorize):
    workers = worker_count(spec)
    finish_sealed_phase(spec, spec.directory/'test-phase', spec.directory/'test-queue', authorize, record)
    if (spec.directory/'test-phase/phase-result.json').exists():
        return {'action': 'test_interaction', 'phase_receipt': content_hash(read(spec.directory/'test-phase/phase-result.json'))}
    cp, state = _initialize(spec, prior, authorize, record)
    completed = set(state['completed_before_parallel'])
    pending = [tid for tid in task_map(spec) if tid not in completed]
    failures = []
    def execute(tid):
        # Start the first tasks immediately, without preparing every pending
        # task's journals and snapshots before any solver can run.
        _prepare_child(spec, tid, state, authorize, record)
        return _launch_child(spec, tid, prior_receipt, signals, record)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, tid): tid for tid in pending}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                if not future.result():
                    failures.append(tid)
                    continue
                _open_child(spec, tid, authorize, record)
                completed.add(tid)
            except Exception as exc:
                failures.append(tid)
                record({'type': 'parallel_test_child_failed', 'task_id': tid, 'error_type': type(exc).__name__})
            finally:
                save(spec.directory/'parallel-test/progress.json', {'completed': sorted(completed),
                      'total': len(task_map(spec)), 'failed': failures, 'workers': workers, 'at': time.time()})
                signals.checkpoint({'phase': 'test', 'completed_tasks': len(completed),
                                    'total_tasks': len(task_map(spec)), 'parallel': True, 'last_task_id': tid})
    if failures:
        raise RuntimeError('some independent test tasks require recovery; successful tasks retained')
    queue = DeferredScoring.open_unsealed(directory=spec.directory/'test-queue', identity=spec.identity,
                  phase='test', binding=spec.binding, authorize=authorize, record=record)
    done = []
    phase = spec.directory/'test-phase'
    for tid in task_map(spec):
        receipt_path = phase/(content_hash(tid)+'.task.json')
        if tid in state['completed_before_parallel']:
            done.append(read(receipt_path))
            continue
        value, child_queue = _open_child(spec, tid, authorize, record)
        for key, row in child_queue._jobs.items():
            _publish(queue.directory/'jobs'/(key+'.json'), row)
            if key in queue._jobs and queue._jobs[key] != row:
                raise ValueError('task grading request changed during merge')
            queue._jobs[key] = row
        _publish(receipt_path, value)
        done.append(value)
    plan = state['plan']
    closed = {'schema': 1, 'manifest_hash': content_hash(plan), 'tasks': done,
              'memory_receipt': state['memory_receipt'], 'interaction_wall_s': time.time()-state['started_at'],
              'drain_wall_s': 0.0}
    closed_path = phase/'closed-phase.json'
    if closed_path.exists():
        previous = read(closed_path)
        if {k:v for k,v in previous.items() if k != 'interaction_wall_s'} != {k:v for k,v in closed.items() if k != 'interaction_wall_s'}:
            raise ValueError('completed parallel phase changed on recovery')
        closed = previous
    _publish(closed_path, closed)
    cp.save(list(task_map(spec)))
    seal = queue.seal(phase_receipt=content_hash(closed))
    result = {'schema': 1, 'manifest_hash': content_hash(plan), 'closed_hash': content_hash(closed), 'queue_seal': seal}
    _publish(phase/'phase-result.json', result)
    return {'action': 'test_interaction', 'phase_receipt': content_hash(result)}
