"""Boundary-only timing, retained outside checkpoint rollback and model inputs.

One small append-only file per run; no polling, transcripts, token accounting or
periodic writes. Worker/round attempts remain distinct so retries are observable.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
import uuid


@contextmanager
def termination_as_exception():
    """Allow normal SIGTERM shutdown to persist timing through finally blocks."""
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class AttemptTiming:
    def __init__(self, path, *, namespace, spec_digest, phase, task_id,
                 monotonic=time.monotonic, utc=time.time, started=None):
        self.path = Path(path)
        self.monotonic, self.utc = monotonic, utc
        self.started, self.started_at = started or (monotonic(), utc())
        self.base = {'schema': 1, 'namespace': namespace, 'spec_digest': spec_digest,
            'phase': phase, 'task_scope': task_id, 'attempt_id': uuid.uuid4().hex}
        self.lock = threading.RLock()
        self.fd = None
        self.unit = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT |
            os.O_CLOEXEC | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        self.emit('attempt_started', started_at=self.started_at)
        return self

    def emit(self, event, **fields):
        row = {**self.base, 'event': event, 'at': self.utc(), **fields}
        data = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
        if len(data) > 4096:
            raise ValueError('timing boundary exceeded its small metadata bound')
        with self.lock:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            try:
                while data:
                    written = os.write(self.fd, data)
                    if written <= 0:
                        raise OSError('timing append made no progress')
                    data = data[written:]
            finally:
                fcntl.flock(self.fd, fcntl.LOCK_UN)

    def _finish_unit(self, event, **fields):
        unit, self.unit = self.unit, None
        if unit is None:
            return
        elapsed = self.monotonic()-unit['started']
        self.emit(event, task_id=unit['task_id'], unit_id=unit['unit_id'], wall_s=elapsed,
            work_wall_s=unit['work_wall_s'],
            checkpoint_wall_s=None if unit['work_wall_s'] is None else elapsed-unit['work_wall_s'], **fields)

    def observe(self, row):
        kind = row.get('type')
        if kind not in {'task_started', 'task_resumed', 'unit_started', 'unit_completed',
                        'unit_preparation_failed', 'round_checkpoint_committed'}:
            return
        with self.lock:
            if kind in {'task_started', 'task_resumed'}:
                if self.base['task_scope'] is not None and row['task_id'] != self.base['task_scope']:
                    raise ValueError('timing task belongs to another worker scope')
                self.emit(kind, task_id=row['task_id'], previously_completed_units=row['completed_units'])
            elif kind == 'unit_started':
                self._finish_unit('unit_interrupted', reason='new_round_before_checkpoint')
                self.unit = {'task_id': row['task_id'], 'unit_id': row['unit_id'],
                    'started': self.monotonic(), 'work_wall_s': None}
                self.emit('unit_started', task_id=row['task_id'], unit_id=row['unit_id'])
            elif kind in {'unit_completed', 'unit_preparation_failed'}:
                if self.unit is None:
                    return
                if (self.unit['task_id'], self.unit['unit_id']) != (row['task_id'], row['unit_id']):
                    raise ValueError('timing completion belongs to another round')
                self.unit['work_wall_s'] = self.monotonic()-self.unit['started']
                if row.get('error_code') or kind == 'unit_preparation_failed':
                    self._finish_unit('unit_failed', error_code=row.get('error_code'))
                else:
                    self.emit('unit_delivered', task_id=row['task_id'], unit_id=row['unit_id'],
                              work_wall_s=self.unit['work_wall_s'])
            elif self.unit is not None:
                if self.unit['task_id'] != row['task_id'] or self.unit['work_wall_s'] is None:
                    raise ValueError('timing checkpoint lacks its completed round')
                self._finish_unit('unit_committed', completed_units=row['completed_units'])

    def __exit__(self, typ, exc, tb):
        try:
            with self.lock:
                self._finish_unit('unit_interrupted', reason='attempt_ended_before_checkpoint')
                self.emit('attempt_finished', wall_s=self.monotonic()-self.started,
                    status=('completed' if typ is None else 'interrupted'
                            if issubclass(typ, (SystemExit, KeyboardInterrupt)) else 'failed'),
                    error_type=None if typ is None else typ.__name__)
        finally:
            os.close(self.fd)
            self.fd = None
        return False
