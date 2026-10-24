"""One-turn authenticated loopback MCP facade for task-owned benchmark tools.

Implements the JSON response subset of MCP Streamable HTTP (2025-06-18).
No remote endpoints, model calls, reference answers or memory are loaded here.
The native CLI invokes this facade; the underlying benchmark keeps its existing
sandbox and exact-result receipts. The solver shell retains network denial.
"""
import hmac
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import secrets
import threading
import time

from .feedback_agent import _unique_object
from .native_benchmark_tools import NativeBenchmarkTools
from .provenance import content_hash
from .rule_appeal import APPEAL_TOOL

VERSION = '2025-06-18'
MAX_REQUEST = 256 * 1024
MAX_RESPONSE = 4 * 1024 * 1024


class ClaudeBenchmarkMCP:
    def __init__(self, *, tools, native_thread_id, turn_id, unit_id, event_id, deadline, review=None):
        if ((tools is not None and not isinstance(tools, NativeBenchmarkTools)) or
                (tools is None and review is None) or (review is not None and not callable(review)) or
                (tools is not None and 'tellonce_review' in tools.names) or
                any(not isinstance(v, str) or not 1 <= len(v) <= 480 for v in
                    (native_thread_id, turn_id, unit_id, event_id)) or
                type(deadline) not in {int, float} or not math.isfinite(deadline)):
            raise ValueError('explicit task-owned benchmark tools and bounded native turn required')
        self.tools = tools
        self.review = review
        self.names = (set(tools.names) if tools is not None else set()) | ({'tellonce_review'} if review is not None else set())
        self.definitions = (tools.definitions if tools is not None else []) + ([APPEAL_TOOL] if review is not None else [])
        self._reviews = {}
        self.native_thread_id, self.turn_id = native_thread_id, turn_id
        self.unit_id, self.event_id, self.deadline = unit_id, event_id, deadline
        self._token = secrets.token_urlsafe(32)
        self._server = self._thread = None
        self._initialized = self._ready = self._closed = False
        self._failure = None
        self.calls = []

    @property
    def port(self):
        return self._server.server_port

    @property
    def configuration(self):
        if self._server is None or self._closed:
            raise ValueError('MCP facade must be running before configuring its native client')
        return {'mcpServers': {'benchmark': {'type': 'http', 'url': f'http://127.0.0.1:{self.port}/mcp',
                'headers': {'Authorization': 'Bearer ' + self._token}}}}

    def _rpc(self, obj):
        if (not isinstance(obj, dict) or obj.get('jsonrpc') != '2.0' or
                not isinstance(obj.get('method'), str) or set(obj)-{'jsonrpc', 'id', 'method', 'params'}):
            raise ValueError('invalid MCP request')
        method, params = obj['method'], obj.get('params', {})
        ident = obj.get('id')
        if not isinstance(params, dict) or (ident is not None and
                (type(ident) not in {str, int} or len(str(ident)) > 128)):
            raise ValueError('invalid MCP request identity or parameters')
        if 'id' not in obj:
            if method == 'notifications/initialized' and self._initialized:
                self._ready = True
            elif method not in {'notifications/cancelled', 'notifications/progress'}:
                raise ValueError('unadmitted MCP notification')
            return None
        if ident is None:
            raise ValueError('null MCP request identity')

        def error(code, message):
            return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': code, 'message': message}}

        if method == 'initialize':
            if self._initialized or not isinstance(params.get('protocolVersion'), str):
                return error(-32600, 'Invalid initialization')
            self._initialized = True
            value = {'protocolVersion': VERSION, 'capabilities': {'tools': {}},
                     'serverInfo': {'name': 'experiment-benchmark', 'version': '1'}}
        elif method == 'ping':
            value = {}
        elif not self._ready:
            return error(-32600, 'Initialization required')
        elif method == 'tools/list':
            value = {'tools': [{k: v for k, v in row.items() if k != 'type'} for row in self.definitions]}
        elif method == 'tools/call':
            if (set(params)-{'name', 'arguments', '_meta'} or params.get('name') not in self.names or
                    not isinstance(params.get('arguments'), dict)):
                return error(-32602, 'Unknown tool or invalid arguments')
            if self._failure is not None:
                return error(-32603, 'Benchmark channel unavailable')
            # RPC identifiers restart with each native CLI invocation. Bind the
            # receipt identity to this turn, while keeping duplicate deliveries
            # of the same request idempotent in NativeBenchmarkTools.
            call_id = content_hash([self.turn_id, type(ident).__name__, ident])
            remaining = max(0, min(120, self.deadline-time.monotonic()))
            if params['name'] == 'tellonce_review':
                if remaining <= 0:
                    return error(-32603, 'Native turn deadline exhausted')
                fingerprint = content_hash(params['arguments'])
                if call_id in self._reviews:
                    previous, value = self._reviews[call_id]
                    if previous != fingerprint:
                        self._failure = 'changed_review_request'
                        return error(-32602, 'Review replay changed arguments')
                else:
                    try:
                        outcome = self.review(params['arguments'])
                        value = {'content': [{'type': 'text', 'text': json.dumps(outcome, ensure_ascii=False, allow_nan=False)}], 'isError': False}
                    except Exception:
                        self._failure = 'review_dispatch_failed'
                        return error(-32603, 'Review channel unavailable')
                    self._reviews[call_id] = (fingerprint, value)
                self.calls.append({'call_id':call_id, 'tool':params['name'], 'arguments':params['arguments'], 'result':value})
                return {'jsonrpc':'2.0', 'id':ident, 'result':value}
            try:
                result = self.tools.call_native({'threadId': self.native_thread_id, 'turnId': self.turn_id,
                    'callId': call_id, 'tool': params['name'], 'arguments': params['arguments']},
                    native_thread_id=self.native_thread_id, turn_id=self.turn_id, unit_id=self.unit_id,
                    event_id=self.event_id, timeout_s=remaining)
            except Exception:
                self._failure = 'benchmark_dispatch_failed'
                return error(-32603, 'Benchmark channel unavailable')
            value = {'content': [{'type': 'text', 'text': item['text']} for item in result['contentItems']],
                     'isError': not result['success']}
            self.calls.append({'call_id': call_id, 'tool': params['name'], 'arguments': params['arguments'],
                               'result': value})
        else:
            return error(-32601, 'Method not supported')
        return {'jsonrpc': '2.0', 'id': ident, 'result': value}

    def start(self):
        if self._server is not None or self._closed:
            raise ValueError('MCP facade cannot be restarted')
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, *args):
                pass

            def send(self, status, value=None):
                body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if value is not None else b''
                if len(body) > MAX_RESPONSE:
                    owner._failure = 'response_budget_exceeded'
                    status, body = 500, b''
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)

            def allowed(self):
                authority = f'127.0.0.1:{owner.port}'
                token = self.headers.get('Authorization', '')
                if (self.path != '/mcp' or self.headers.get('Host') != authority or
                        self.headers.get('Origin') not in (None, 'http://' + authority) or
                        not hmac.compare_digest(token.encode('utf-8'), ('Bearer ' + owner._token).encode())):
                    self.send(403)
                    return False
                if self.headers.get('MCP-Protocol-Version') not in (None, VERSION):
                    self.send(400)
                    return False
                return True

            def do_POST(self):
                if not self.allowed():
                    return
                try:
                    lengths = self.headers.get_all('Content-Length', [])
                    if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                        raise ValueError('one bounded content length required')
                    size = int(lengths[0])
                    if not 0 < size <= MAX_REQUEST:
                        raise ValueError('MCP request exceeds bound')
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise ValueError('incomplete MCP request')
                    obj = json.loads(raw, object_pairs_hook=_unique_object,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                    value = owner._rpc(obj)
                except (ValueError, TypeError, OSError):
                    self.send(400)
                    return
                self.send(202 if value is None else 200, value)

            def do_GET(self):
                if self.allowed():
                    self.send(405)

            do_DELETE = do_GET

        self._server = HTTPServer(('127.0.0.1', 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={'poll_interval': .1},
                                        name='claude-benchmark-mcp', daemon=True)
        self._thread.start()
        return self

    def assert_healthy(self):
        if self._failure is not None or self._thread is None or not self._thread.is_alive():
            raise RuntimeError('Claude benchmark MCP channel is unhealthy')

    def validate_observed(self, observed):
        """Bind facade replies to real native tool completions, including repeats.

        Failed native validation/permission attempts may have no backend call.
        Every successful native result and every backend reply must match.
        """
        def text(content):
            if isinstance(content, str):
                return content
            if isinstance(content, list) and all(isinstance(c, dict) and c.get('type') == 'text' and
                                                isinstance(c.get('text'), str) for c in content):
                return ''.join(c['text'] for c in content)
            raise ValueError('benchmark result is not the original textual tool output')
        def key(tool, args, content, error):
            return content_hash([tool, args, text(content), error])
        unique = {c['call_id']: c for c in self.calls}
        failures = Counter()
        expected = Counter(key(c['tool'], c['arguments'], c['result']['content'], c['result']['isError'])
                           for c in unique.values())
        for row in observed:
            identity = key(row['tool'], row['arguments'], row['content'], row['isError'])
            if expected[identity]:
                expected[identity] -= 1
            elif not row['isError']:
                raise ValueError('native benchmark completion lacks matching original backend result')
            else:
                failures[content_hash([row['tool'],row['arguments']])] += 1
        # A backend may finish after Claude's tool timeout. The model saw an
        # error, not the backend result; preserve that fact instead of rejecting
        # the whole answer. Never accept altered successful tool output.
        missing = 0
        for call in unique.values():
            identity=key(call['tool'],call['arguments'],call['result']['content'],call['result']['isError'])
            if not expected[identity]:continue
            failed=content_hash([call['tool'],call['arguments']])
            if not failures[failed]:
                raise ValueError('benchmark backend result lacks an actual native completion')
            failures[failed]-=1;expected[identity]-=1;missing+=1
        return missing

    def close(self):
        if not self._closed:
            self._closed = True
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
                self._thread.join(timeout=1)

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()
