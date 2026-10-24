"""Task-owned bridge from native dynamic tool requests to admitted benchmarks.

The caller owns tool policy and sandboxed backends. This bridge binds native
identities and saves exact responses privately so delivery retries never issue
the same physical query twice. No model, memory or task answer is loaded here.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from .atomic_io import atomic_write_bytes
from .provenance import canonical_json_bytes, content_hash


class NativeBenchmarkTools:
    def __init__(self, *, owner, definitions, invoke, close, record, directory):
        if (not isinstance(owner, dict) or set(owner) != {'namespace', 'task_id', 'session_id'}
                or any(not isinstance(v, str) or not 1 <= len(v) <= 480 for v in owner.values())
                or not isinstance(directory, Path) or not directory.is_absolute()
                or directory != directory.resolve() or directory.exists()
                or any(not callable(f) for f in (invoke, close, record))
                or not isinstance(definitions, list) or not 1 <= len(definitions) <= 32):
            raise ValueError('explicit benchmark owner, fresh private receipts and tool contract required')
        serialized = canonical_json_bytes(definitions)
        if len(serialized) > 65536:
            raise ValueError('benchmark tool definitions exceed their budget')
        names = set()
        for row in definitions:
            if (not isinstance(row, dict) or set(row) != {'type', 'name', 'description', 'inputSchema'}
                    or row['type'] != 'function' or not isinstance(row['name'], str)
                    or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}', row['name'])
                    or row['name'] in names|{'tellonce_review','exec','exec_command','write_stdin','apply_patch','view_image','wait'}
                    or not isinstance(row['description'], str) or not row['description'].strip()
                    or not isinstance(row['inputSchema'], dict) or row['inputSchema'].get('type') != 'object'):
                raise ValueError('distinct admitted benchmark function definitions required')
            names.add(row['name'])
        directory.mkdir(mode=0o700)
        self.directory, self._owner, self._definitions = directory, dict(owner), serialized
        self.names = frozenset(names)
        self._invoke, self._close_backend, self.record = invoke, close, record
        self._seen, self._closed, self._lock = {}, False, threading.Lock()
        self._cancelled_before_dispatch = {}

    @property
    def owner(self):
        return dict(self._owner)

    @property
    def definitions(self):
        return json.loads(self._definitions)

    def cancel_if_unstarted(self, *, tool_use_id, tool_name, tool_input):
        """Atomically fence a host-owned call that has never reached dispatch.

        A native code cell can be cancelled while its pre-action hook is still
        running. Stop may then arrive without a native tool result. Only this
        host bridge can prove that the operation never ran and prevent a late
        request from executing after the delivery was accepted.
        """
        if (not isinstance(tool_use_id, str) or not 1 <= len(tool_use_id) <= 480
                or not isinstance(tool_name, str) or not isinstance(tool_input, dict)):
            raise ValueError('exact native tool identity required for cancellation')
        if tool_name not in self.names or not self._lock.acquire(blocking=False):
            return False
        try:
            if self._closed or tool_use_id in self._seen:
                return False  # Never infer completion or zero effects after dispatch.
            signature = content_hash({'tool':tool_name, 'arguments':tool_input})
            previous = self._cancelled_before_dispatch.get(tool_use_id)
            if previous is not None and previous != signature:
                raise ValueError('cancelled native tool identity changed its action')
            self._cancelled_before_dispatch[tool_use_id] = signature
            return True
        finally:
            self._lock.release()

    def call_native(self, params, *, native_thread_id, turn_id, unit_id, event_id, timeout_s):
        required = {'threadId', 'turnId', 'tool', 'callId', 'arguments'}
        if (not isinstance(params, dict) or not required <= set(params)
                or set(params)-required-{'namespace'} or params.get('namespace') is not None
                or params['threadId'] != native_thread_id or params['turnId'] != turn_id
                or not isinstance(params['tool'], str) or params['tool'] not in self.names
                or not isinstance(params['arguments'], dict)
                or any(not isinstance(v, str) or not 1 <= len(v) <= 480
                       for v in (native_thread_id, turn_id, unit_id, event_id, params['callId']))
                or type(timeout_s) not in {int, float} or not math.isfinite(timeout_s) or not 0 <= timeout_s <= 120):
            raise ValueError('foreign, malformed or expired native benchmark tool request')
        params = json.loads(canonical_json_bytes(params))
        params.setdefault('namespace', None)
        context = {**self._owner, 'unit_id': unit_id, 'event_id': event_id, 'call_id': params['callId']}
        signature = content_hash({'params': params, 'context': context})
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('one benchmark dispatch at a time per native task')
        try:
            if self._closed:
                raise RuntimeError('native benchmark tool bridge is closed')
            cancelled = self._cancelled_before_dispatch.get(params['callId'])
            if cancelled is not None:
                if cancelled != content_hash({'tool':params['tool'], 'arguments':params['arguments']}):
                    raise ValueError('cancelled native tool identity changed its action')
                return self._unavailable()
            previous = self._seen.get(params['callId'])
            if previous is not None:
                if previous[0] != signature:
                    raise ValueError('native benchmark call changed its input or owner')
                with (self.directory/(signature+'.json')).open('rb') as stream:
                    raw = stream.read(4*1024*1024+1)
                if len(raw) > 4*1024*1024 or hashlib.sha256(raw).hexdigest() != previous[1]:
                    raise ValueError('recorded native tool response changed')
                return json.loads(raw)
            # Reserve identity before invoking: an unexpected write/audit failure
            # must not make the already executed tool safe to replay.
            self._seen[params['callId']] = (signature, None)
            started = time.monotonic()
            try:
                reply = (self._invoke(context=context, tool=params['tool'], args=params['arguments'], timeout_s=timeout_s)
                         if timeout_s else {'status': 'unavailable'})
                if not isinstance(reply, dict) or reply.get('status') not in {'completed', 'unavailable'}:
                    raise ValueError('invalid benchmark tool result')
                if reply['status'] == 'completed':
                    text = reply.get('result')
                    if (not isinstance(text, str) or len(text.encode()) > 2*1024*1024
                            or hashlib.sha256(text.encode()).hexdigest() != reply.get('result_sha256')):
                        raise ValueError('benchmark tool did not return its complete original text')
                    result = {'success': True, 'contentItems': [{'type': 'inputText', 'text': text}]}
                else:
                    result = self._unavailable()
                kind, error = (('native_benchmark_tool_completed', None) if timeout_s else
                               ('native_benchmark_tool_skipped', 'NoTimeRemaining'))
            except Exception as exc:
                result, kind, error = self._unavailable(), 'native_benchmark_tool_failed', type(exc).__name__
            raw = canonical_json_bytes(result)
            if len(raw) > 3*1024*1024:
                result, kind, error = self._unavailable(), 'native_benchmark_tool_failed', 'OutputBudget'
                raw = canonical_json_bytes(result)
            atomic_write_bytes(self.directory/(signature+'.json'), raw)
            self._seen[params['callId']] = (signature, hashlib.sha256(raw).hexdigest())
            self.record({'type': kind, 'context': context, 'tool': params['tool'], 'arguments': params['arguments'],
                'success': result['success'], 'error_type': error, 'wall_s': time.monotonic()-started,
                'response_digest': hashlib.sha256(raw).hexdigest()})
            return result
        finally:
            self._lock.release()

    @staticmethod
    def _unavailable():
        return {'success': False, 'contentItems': [{'type': 'inputText', 'text':
            'The benchmark tool could not provide a usable result for this request. Do not treat it as evidence.'}]}

    def close(self):
        if not self._closed:
            self._closed = True
            self._close_backend()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
