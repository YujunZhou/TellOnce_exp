"""Small structured worker messages and interruptible provider waits.

Keep this stream separate from library diagnostics. Checkpoints describe
progress; they do not authorize replay of side-effecting task operations.
"""
import math
import sys
import threading
import time

from .provenance import canonical_json_bytes


class WorkerSignals:
    def __init__(self,*,stream=None,interval_s=10):
        if type(interval_s) not in {int,float} or not math.isfinite(interval_s) or not .01<=interval_s<=30:
            raise ValueError('finite bounded worker heartbeat interval required')
        self.stream=sys.stdout if stream is None else stream
        self.interval_s=interval_s
        self._closing=threading.Event()
        self._failed=threading.Event()
        self._lock=threading.Lock()
        self._control=threading.RLock()
        self._thread=None
        self._result=False

    def _emit(self,event):
        raw=canonical_json_bytes(event)
        if len(raw)>4*1024*1024:
            raise ValueError('worker event exceeds the private stream budget')
        with self._lock:
            self.stream.write(raw.decode()+'\n')
            self.stream.flush()

    def _heartbeat(self):
        while not self._closing.wait(self.interval_s):
            try:
                self._emit({'type':'heartbeat'})
            except Exception:
                self._failed.set()
                return

    def __enter__(self):
        if self._thread is not None or self._closing.is_set():
            raise ValueError('worker heartbeat cannot be reopened')
        self._emit({'type':'heartbeat'})
        self._thread=threading.Thread(target=self._heartbeat,daemon=True,name='worker-heartbeat')
        self._thread.start()
        return self

    def assert_healthy(self):
        if self._failed.is_set() or self._closing.is_set() or self._thread is None or not self._thread.is_alive():
            raise RuntimeError('worker_supervision_unavailable')

    def checkpoint(self,value):
        with self._control:
            self.assert_healthy()
            if not isinstance(value,dict) or self._result or len(canonical_json_bytes(value))>1024*1024:
                raise ValueError('one bounded nonterminal worker checkpoint required')
            self._emit({'type':'checkpoint','value':value})

    def result(self,value):
        with self._control:
            self.assert_healthy()
            if not isinstance(value,dict) or self._result:
                raise ValueError('one terminal worker result object required')
            self._emit({'type':'result','value':value})
            self._result=True

    def wait_retry(self,seconds,*,context,reason):
        """Compatible with ManagedModelRequests; heartbeat continues while waiting."""
        self.assert_healthy()
        if type(seconds) not in {int,float} or not math.isfinite(seconds) or not 0<=seconds<=30:
            raise ValueError('provider waits must be supervised in short segments')
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if self._closing.wait(min(1,max(0,deadline-time.monotonic()))):
                raise RuntimeError('worker_stopping')
            self.assert_healthy()

    def close(self):
        self._closing.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError('worker_heartbeat_did_not_stop')

    def __exit__(self,*args):
        self.close()
