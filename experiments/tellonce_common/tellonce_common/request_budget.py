"""An explicit host operation deadline shared by its nested text requests.

Scope is local to the executing context. Worker threads must establish their
own scope; unrelated task/model calls never inherit another thread's deadline.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

_deadline=ContextVar('experiment_request_deadline',default=None)


class RequestDeadline(RuntimeError):
    retryable=False


def _duration(seconds):
    if type(seconds) not in {int,float} or not math.isfinite(seconds) or not 0<seconds<=86400:
        raise ValueError('finite positive request operation duration required')


def bounded_deadline(seconds):
    _duration(seconds)
    own=time.monotonic()+seconds
    parent=_deadline.get()
    return min(own,parent) if parent is not None else own


def supervised_retry_wait(seconds, *, deadline, wait_retry, context, reason):
    """Honor a provider delay within one deadline, keeping host waits short."""
    if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('finite nonnegative provider wait required')
    remaining = max(.01, seconds)
    if remaining > deadline-time.monotonic():
        raise RequestDeadline('provider_wait_exceeds_operation_deadline')
    while remaining > 0:
        chunk = min(30, remaining)
        if chunk > deadline-time.monotonic():
            raise RequestDeadline('provider_wait_exceeds_operation_deadline')
        wait_retry(chunk, context=context, reason=reason)
        remaining -= chunk


@contextmanager
def request_budget(seconds):
    token=_deadline.set(bounded_deadline(seconds))
    try:
        yield
    finally:
        _deadline.reset(token)


class RequestBudget:
    def __init__(self,seconds):
        self.deadline=bounded_deadline(seconds)
        self.remaining()

    def remaining(self,maximum=None):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:
            raise RequestDeadline('request_operation_deadline')
        if maximum is not None:
            _duration(maximum)
            remaining=min(remaining,maximum)
        return remaining
