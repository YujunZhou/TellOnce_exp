"""Close a complete native task phase before admitting independent evaluation.

The enclosing process supervisor owns credentials, heartbeat and hard limits.
Task-boundary checkpoints resume only after restoring coherent owned state.
"""
from dataclasses import asdict
from pathlib import Path
import time

from tellonce_common.deferred_scoring import DeferredScoring, _write
from tellonce_common.frozen_task_evaluation import _directory, _digest, _read
from tellonce_common.provenance import content_hash
from tellonce_common.tellonce_method import TellonceMethod
from tellonce_common.task_outcomes import outcome_to_dict
from .native_task import ClawTask, read_claw_task_result
from tellonce_common.native_availability import task_interruption, availability_fields


class ClawPhase:
    def __init__(self, *, directory, queue, tasks, memory, authorize, record, check_health, checkpoint, recovery=None):
        _directory(directory)
        if ((directory.exists() and recovery is None) or not isinstance(queue, DeferredScoring) or queue._sealed is not None or (queue._jobs and recovery is None) or
                not isinstance(tasks, tuple) or not tasks or any(not isinstance(task, ClawTask) for task in tasks) or
                any(not callable(fn) for fn in (authorize, record, check_health, checkpoint))):
            raise ValueError('a fresh complete task phase and explicit supervision are required')
        identity, binding = queue.identity, queue._manifest['binding']
        expected = {'trace': TellonceMethod}
        if identity.method == 'no_memory':
            if memory is not None:
                raise ValueError('no-memory phase cannot own a learned state')
        elif (identity.method not in expected or not isinstance(memory, expected[identity.method]) or
              memory.identity != identity or memory.binding != binding):
            raise ValueError('phase requires its namespace-owned memory method')
        if (any(task.queue is not queue or task.identity != identity or task.binding != binding or
                task.memory_factory is not memory or task._ran for task in tasks) or
                len({task.units[0].independent_task_id for task in tasks}) != len(tasks) or
                len({task.directory for task in tasks}) != len(tasks)):
            raise ValueError('every independent task must occur once with this exact queue and memory')
        self.directory, self.queue, self.tasks, self.memory = directory, queue, tasks, memory
        self.identity, self.binding = identity, binding
        self.authorize, self.record, self.check_health, self.checkpoint = authorize, record, check_health, checkpoint
        self.recovery = recovery
        self._ran = False
        self._manifest = {'schema': 1, 'namespace': identity.namespace, 'phase': queue.phase,
            'binding': binding, 'queue_directory': str(queue.directory), 'tasks': [
                {'task_id': task.units[0].independent_task_id, 'directory': str(task.directory),
                 'units_hash': content_hash([asdict(unit) for unit in task.units]),
                 'unit_ids': [unit.instance.common_instance_id for unit in task.units],
                 'asset_digest': task.assets.digest, 'evaluation_digest': task.evaluation_digest,
                 'profile_digest': task.profile_digest} for task in tasks]}

    def run(self):
        if self._ran:
            raise ValueError('a native phase runs once; interrupted solver effects cannot be replayed')
        self.check_health()
        if (self.queue._sealed is not None or (self.queue._jobs and self.recovery is None) or any(task._ran for task in self.tasks) or
                any(content_hash([asdict(unit) for unit in task.units]) != planned['units_hash']
                    for task, planned in zip(self.tasks, self._manifest['tasks']))):
            raise ValueError('phase tasks or grading queue changed after preparation')
        self.authorize('start_phase', dict(self._manifest))
        self._ran = True
        if self.directory.exists():
            if self.recovery is None or not self.recovery.restored or _read(self.directory/'phase.json', 8*1024*1024)!=self._manifest:
                raise ValueError('restored phase plan differs from the admitted complete task plan')
        else:
            self.directory.mkdir(mode=0o700)
            _write(self.directory/'phase.json', self._manifest)
        if self.recovery is not None and self.recovery.current() is None:
            self.recovery.save([], memory=self.memory)
        started = time.monotonic()
        completed = []
        restored_ids=set(self.recovery.value['completed']) if self.recovery and self.recovery.restored else set()
        try:
            if self.memory is not None:
                if self.queue.phase == 'training':
                    self.memory.start_background()
                else:
                    self.memory.verify_frozen()
            for task in self.tasks:
                self.check_health()
                task_id = task.units[0].independent_task_id
                if self.recovery is not None and self.recovery.completed_receipt(task_id) is not None:
                    value = _read(self.directory/(content_hash(task_id)+'.task.json'), 1024*1024)
                    if (value['task_id']!=task_id or value['aborted'] is not False or
                            value['receipt_digest']!=self.recovery.completed_receipt(task_id) or
                            value['directory']!=str(task.directory)):
                        raise ValueError('checkpoint completed task differs from its receipt')
                    completed.append(value)
                    continue
                task_started = time.monotonic()
                result = task.run()
                value = {'task_id': result.outcome.task_id, 'receipt_digest': result.receipt_digest,
                         'directory': str(task.directory), 'wall_s': time.monotonic()-task_started,
                         'aborted': result.outcome.aborted}
                _write(self.directory/(content_hash(value['task_id'])+'.task.json'), value)
                completed.append(value)
                if result.outcome.aborted:
                    self.checkpoint({'phase': self.queue.phase, 'completed_tasks': len(completed)-1,
                                     'total_tasks': len(self.tasks), 'last_task': value, 'committed': False})
                    raise task_interruption(result.outcome)
                if self.recovery is not None:
                    self.recovery.save([row['task_id'] for row in completed], memory=self.memory)
                self.checkpoint({'phase': self.queue.phase, 'completed_tasks': len(completed),
                                 'total_tasks': len(self.tasks), 'last_task': value})
            interaction_wall_s = time.monotonic()-started
            if self.recovery is not None and self.recovery.restored:
                interaction_wall_s += sum(row['wall_s'] for row in completed if row['task_id'] in restored_ids)
            drain_started = time.monotonic()
            memory_receipt = None
            if self.memory is not None:
                if self.queue.phase == 'training':
                    memory_receipt = self.memory.finish_training()
                else:
                    self.memory.verify_frozen()
                    memory_receipt = content_hash(self.memory._receipt)
                self.memory.close()
            self.check_health()
            closed = {'schema': 1, 'manifest_hash': content_hash(self._manifest), 'tasks': completed,
                      'memory_receipt': memory_receipt, 'interaction_wall_s': interaction_wall_s,
                      'drain_wall_s': time.monotonic()-drain_started}
            _write(self.directory/'closed-phase.json', closed)
            closed_hash = content_hash(closed)
            seal = self.queue.seal(phase_receipt=closed_hash)
            result = {'schema': 1, 'manifest_hash': content_hash(self._manifest),
                      'closed_hash': closed_hash, 'queue_seal': seal}
            _write(self.directory/'phase-result.json', result)
            digest = content_hash(result)
            self.record({'type': 'native_phase_closed', 'namespace': self.identity.namespace,
                         'phase': self.queue.phase, 'receipt': digest, 'memory_receipt': memory_receipt,
                         'interaction_wall_s': interaction_wall_s, 'drain_wall_s': closed['drain_wall_s']})
            return digest
        except BaseException as exc:
            availability = availability_fields(exc)
            if type(exc).__name__ == 'FeedbackServiceUnavailable':
                availability['availability_code'] = getattr(exc, 'code', None)
            self.record({'type': 'native_phase_interrupted', 'namespace': self.identity.namespace,
                         'phase': self.queue.phase, 'completed_tasks': (len(self.recovery.value['completed']) if self.recovery and self.recovery.value
                             else sum(not row['aborted'] for row in completed)),
                         'total_tasks': len(self.tasks), 'error_type': type(exc).__name__,
                         **availability,
                         'retry_after_s': getattr(exc, 'retry_after_s', None) if type(exc).__name__ == 'FeedbackServiceUnavailable' else None})
            raise
        finally:
            if self.memory is not None:
                self.memory.close()


def open_claw_phase(*, directory, expected_receipt, identity, binding, authorize, record):
    """Recover only closed evidence and evaluation work, without memory or solver."""
    _directory(directory)
    manifest = _read(directory/'phase.json', 8*1024*1024)
    closed = _read(directory/'closed-phase.json', 8*1024*1024)
    result = _read(directory/'phase-result.json', 1024*1024)
    if (not _digest(expected_receipt) or content_hash(result) != expected_receipt or
            result.get('schema') != 1 or manifest.get('schema') != 1 or closed.get('schema') != 1 or
            manifest.get('namespace') != identity.namespace or manifest.get('binding') != binding or
            result.get('manifest_hash') != content_hash(manifest) or
            closed.get('manifest_hash') != content_hash(manifest) or result.get('closed_hash') != content_hash(closed) or
            len(manifest.get('tasks', [])) != len(closed.get('tasks', [])) or not manifest.get('tasks')):
        raise ValueError('native phase does not match its completed external receipt')
    queue = DeferredScoring.open_sealed(directory=Path(manifest['queue_directory']), identity=identity,
        phase=manifest['phase'], binding=binding, expected_seal=result['queue_seal'], authorize=authorize, record=record)
    seal_value = _read(queue.directory/'seal.json', 8*1024*1024)
    if seal_value.get('phase_receipt') != result['closed_hash']:
        raise ValueError('scoring queue was not sealed by this completed phase')
    tasks = {}
    for planned, done in zip(manifest['tasks'], closed['tasks']):
        if (done.get('task_id') != planned['task_id'] or done.get('directory') != planned['directory'] or
                done.get('aborted') is not False or planned['task_id'] in tasks):
            raise ValueError('completed tasks differ from their full phase plan')
        task = read_claw_task_result(directory=Path(done['directory']), expected_digest=done['receipt_digest'],
                                     identity=identity, binding=binding)
        if (task.outcome.task_id != planned['task_id'] or task.outcome.aborted or
                [unit.unit_id for unit in task.outcome.units] != planned['unit_ids'] or
                task.asset_digest != planned['asset_digest'] or task.evaluation_digest != planned['evaluation_digest'] or
                task.profile_digest != planned['profile_digest']):
            raise ValueError('task receipt differs from its completed phase')
        tasks[planned['task_id']] = task
    for job in queue._jobs.values():
        task = tasks.get(job['context']['task_id'])
        if (task is None or job['context']['session_id'] != task.outcome.session_id or
                job['unit_id'] not in {unit.unit_id for unit in task.outcome.units}):
            raise ValueError('sealed scoring job belongs to a task outside this phase')
    return manifest, closed, queue, tasks


def evaluate_claw_phase(*, directory, expected_receipt, identity, binding, scorer_factory,
                        authorize, record, check_health, repair_preferences=False):
    check_health()
    manifest, closed, queue, tasks = open_claw_phase(directory=directory, expected_receipt=expected_receipt,
        identity=identity, binding=binding, authorize=authorize, record=record)
    scorers = {}
    if repair_preferences:
        from tellonce_common.preference_repair import PreferenceRepairQueue, REVISION
        queue = PreferenceRepairQueue(queue)
    started = time.monotonic()
    def score(unit_id, answer, *, context, previous_grade=None):
        check_health()
        task_id = context['task_id']
        if task_id not in scorers:
            scorers[task_id] = scorer_factory(tasks[task_id])
        return scorers[task_id](unit_id, answer, context=context, **(
            {'previous_grade': previous_grade} if previous_grade is not None else {}))
    queue.evaluate(score)
    outcomes = [outcome_to_dict(queue.apply(task.outcome)) for task in tasks.values()]
    check_health()
    value = {'schema': 1, 'phase_receipt': expected_receipt, 'outcomes': outcomes}
    path = directory/('evaluated-phase-'+REVISION+'.json' if repair_preferences else 'evaluated-phase.json')
    if path.exists():
        if _read(path, 256*1024*1024) != value:
            raise ValueError('completed phase evaluation changed on recovery')
    else:
        _write(path, value)
    record({'type': 'native_phase_evaluated', 'namespace': identity.namespace, 'phase': manifest['phase'],
            'wall_s': time.monotonic()-started, 'receipt': content_hash(value)})
    return value
