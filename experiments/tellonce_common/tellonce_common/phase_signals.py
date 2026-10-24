"""Phase signals: latch provider outages observed in worker events and emit checkpoints."""
import json
import math
import threading
import time

from .feedback_transport import FeedbackServiceUnavailable


class PhaseSignals:
    """No heartbeat files or polling thread; the process parent owns liveness."""
    def __init__(self):
        self._credit_outage = threading.Event()

    def observe(self, event):
        # Selection/checking SDKs may intentionally tolerate malformed model
        # output. An unfunded provider is an experiment interruption instead:
        # latch it even if the inner SDK catches the transport exception.
        if (event.get('type') == 'http_text_failed' and
                event.get('metrics', {}).get('http_status') == 402):
            self._credit_outage.set()

    def assert_healthy(self):
        if self._credit_outage.is_set():
            raise FeedbackServiceUnavailable('provider_credit_exhausted')

    def checkpoint(self, value):
        self.assert_healthy()
        print(json.dumps({'type': 'checkpoint', 'value': value}), flush=True)

    def wait_retry(self, seconds, *, context, reason):
        if type(seconds) not in {int, float} or not math.isfinite(seconds) or not 0 <= seconds <= 30:
            raise ValueError('bounded retry interval required')
        self.assert_healthy()
        time.sleep(seconds)
        self.assert_healthy()
