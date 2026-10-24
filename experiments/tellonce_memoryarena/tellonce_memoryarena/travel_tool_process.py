"""Bounded lifetime and result ownership for one travel task's tool process.

The trusted host supplies the admitted native sandbox command and public-only
resource setup. Agent arguments can choose only the six validated queries;
they cannot change this command, resource paths, or experiment ownership.
"""
import hashlib
import json
import math
import threading
import time

from tellonce_common.native_app_server import NativeAppServer

from .travel_tool_service import MAX_REQUEST_CHARACTERS, validate_owner, validate_request


def _timeout(value):
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 120:
        raise ValueError('explicit tool deadline of at most 120 seconds required')
    return value


class TravelToolProcess:
    def __init__(self, *, command, cwd, env, setup, record, initialize_timeout_s=120):
        _timeout(initialize_timeout_s)
        if (not isinstance(setup, dict) or set(setup) != {'owner', 'source_root', 'database_root'}
                or not callable(record)):
            raise ValueError('public tool setup and audit callback required')
        validate_owner(setup['owner'])
        self.owner = dict(setup['owner'])
        self.record, self._closed, self._seen = record, False, set()
        self._lock = threading.Lock()
        self._server = NativeAppServer(command, cwd=cwd, env=env)
        try:
            reply = self._server.request('initialize', setup, timeout_s=initialize_timeout_s)
            if reply != {'type': 'ready', 'owner': self.owner}:
                raise ValueError('tool initialization belongs to another owner')
        except BaseException:
            self.close()
            raise

    @property
    def pid(self):
        return self._server.pid

    def call(self, *, context, tool, args, timeout_s):
        timeout_s = _timeout(timeout_s)
        # Defensive copy prevents a host caller from changing an in-flight call.
        request = json.loads(json.dumps({'context': context, 'tool': tool, 'args': args}, allow_nan=False))
        validate_request(request, self.owner)
        if len(json.dumps({'id': 999999999, 'method': 'tool/call', 'params': request}, ensure_ascii=False))+1 > MAX_REQUEST_CHARACTERS:
            raise ValueError('tool request exceeds its service input budget')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('one travel tool operation at a time per task')
        try:
            if self._closed:
                raise RuntimeError('travel task tool process is closed')
            identity = request['context']['call_id']
            if identity in self._seen:
                raise ValueError('duplicate native tool call identity')
            self._seen.add(identity)
            started = time.monotonic()
            try:
                reply = self._server.request('tool/call', request, timeout_s=timeout_s)
                self._validate_result(reply, request)
                self.record({'type': 'benchmark_tool_completed', **request, 'reply': reply,
                             'wall_s': time.monotonic()-started})
                return reply
            except BaseException as exc:
                self.close()
                self.record({'type': 'benchmark_tool_failed', **request, 'error_type': type(exc).__name__,
                             'wall_s': time.monotonic()-started})
                if not isinstance(exc, Exception):
                    raise
                raise RuntimeError('travel tool call failed; no result can be accepted') from None
        finally:
            self._lock.release()

    def _validate_result(self, reply, request):
        common = {'type', 'owner', 'context', 'status', 'wall_s'}
        if (not isinstance(reply, dict) or not common <= set(reply) or reply['type'] != 'tool_result'
                or reply['owner'] != self.owner or reply['context'] != request['context']
                or type(reply['wall_s']) not in {int, float} or not math.isfinite(reply['wall_s'])
                or reply['wall_s'] < 0):
            raise ValueError('tool result differs from its actual owned request')
        if reply['status'] == 'completed':
            if (set(reply) != common|{'result', 'result_sha256'} or not isinstance(reply['result'], str)
                    or hashlib.sha256(reply['result'].encode()).hexdigest() != reply['result_sha256']):
                raise ValueError('original tool result is incomplete or changed')
        elif reply['status'] == 'unavailable':
            if (not set(reply) <= common|{'error_code', 'error_type'}
                    or reply.get('error_code') not in {'output_budget', 'wire_output_budget', 'native_tool_exception'}):
                raise ValueError('unknown or malformed tool failure')
        else:
            raise ValueError('service did not accept the validated tool request')

    def close(self):
        if not self._closed:
            self._closed = True
            self._server.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
