"""One bounded stdio native app-server process for a complete task.

This transport does not choose models, permissions or tool policy. Its caller
must admit the source/configuration and answer native requests explicitly.
Closing the task, or failing it, owns cleanup of the whole process group.
"""
from collections import deque
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from .process_control import terminate_process_tree

MAX_MESSAGE = 4*1024*1024
MAX_QUEUED = 8*1024*1024


class NativeAppServer:
    def __init__(self, command: list[str], *, cwd: Path, env: dict[str,str]):
        self._condition,self._write_lock = threading.Condition(),threading.Lock()
        self._events,self._responses,self._waiting = deque(),{},set()
        self._server_requests,self._counter,self._queued = set(),0,0
        self._request_handlers = {}
        self._failure,self._closed = None,False
        self.stderr = bytearray()
        self.last_server_error = None
        if not isinstance(command,list) or not command or any(not isinstance(arg,str) or not arg for arg in command):
            raise ValueError('native process requires an explicit command argument list')
        # A separate stdlib-only exec wrapper avoids preexec_fn in this
        # multithreaded host. It preserves PID and the original native argv.
        self.spawn_command=[str(Path(sys.executable).resolve()),'-S','-B',
            str(Path(__file__).with_name('native_exec_worker.py').resolve()),
            '--owner-pid',str(os.getpid()),'--',*command]
        self._process = subprocess.Popen(self.spawn_command,cwd=cwd,env=env,stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        os.set_blocking(self._process.stdin.fileno(),False)
        self._reader = threading.Thread(target=self._read,daemon=True,name='native-app-server-events')
        self._reader.start()

    @property
    def pid(self):
        return self._process.pid

    def _fail(self, code):
        with self._condition:
            if self._failure is None:
                self._failure = code
            self._condition.notify_all()

    def assert_healthy(self):
        if self._closed or self._failure is not None or self._process.poll() is not None:
            raise RuntimeError('native task process is unavailable; do not accept this turn')

    def _send(self, message, *, timeout_s):
        data = json.dumps(message,ensure_ascii=False,allow_nan=False).encode()+b'\n'
        if len(data) > MAX_MESSAGE or not 0 < timeout_s <= 3600:
            raise ValueError('native request exceeds its input or time budget')
        deadline = time.monotonic()+timeout_s
        if not self._write_lock.acquire(timeout=timeout_s):
            self._fail('write_timeout')
            raise RuntimeError('native task request write timed out')
        try:
            self.assert_healthy()
            offset = 0
            while offset < len(data):
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('native write timeout')
                _,ready,_ = select.select([], [self._process.stdin], [], min(.1,remaining))
                if not ready:
                    self.assert_healthy()
                    continue
                try:
                    size = os.write(self._process.stdin.fileno(),data[offset:])
                except BlockingIOError:
                    continue
                if size <= 0:
                    raise OSError('native pipe write failed')
                offset += size
        except Exception:
            self._fail('write_failed')
            raise RuntimeError('native task request could not be written') from None
        finally:
            self._write_lock.release()

    def request(self, method: str, params: dict, *, timeout_s: float = 30) -> dict:
        if not isinstance(method,str) or not method or not isinstance(params,dict) or not 0 < timeout_s <= 3600:
            raise ValueError('an explicit bounded native request is required')
        deadline = time.monotonic()+timeout_s
        with self._condition:
            self.assert_healthy()
            self._counter += 1
            identity = self._counter
            self._waiting.add(identity)
        try:
            self._send({'id':identity,'method':method,'params':params},timeout_s=timeout_s)
            while True:
                with self._condition:
                    self.assert_healthy()
                    if identity in self._responses:
                        response = self._responses.pop(identity)
                        self._queued -= len(json.dumps(response,ensure_ascii=False).encode())
                        break
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        self._fail('request_timeout')
                        self.assert_healthy()
                    handled = self._take_handled_request()
                    if handled is None:
                        self._condition.wait(timeout=remaining)
                        continue
                # Native auth can request a refresh before completing our RPC.
                # Do not wait for that RPC before serving an admitted request.
                self._serve_request(handled,deadline=deadline)
            if 'error' in response or not isinstance(response.get('result'),dict):
                self.last_server_error = response.get('error')
                self._fail('server_error')
                raise RuntimeError('native task server rejected a request; see private audit')
            return response['result']
        finally:
            with self._condition:
                self._waiting.discard(identity)

    def notify(self, method: str, params: dict):
        if not isinstance(method,str) or not method or not isinstance(params,dict):
            raise ValueError('an explicit native notification is required')
        self._send({'method':method,'params':params},timeout_s=10)

    def respond(self, identity, *, result: dict | None = None, error: dict | None = None):
        if (result is None)==(error is None):
            raise ValueError('answer a native request with one explicit result or error')
        with self._condition:
            if identity not in self._server_requests:
                raise ValueError('cannot answer an unobserved native request')
        self._send({'id':identity,**({'result':result} if result is not None else {'error':error})},timeout_s=10)
        with self._condition:
            self._server_requests.remove(identity)

    def register_request_handler(self, method, handler):
        """Register one trusted host operation, never an unrestricted tool.

        Handlers receive a deadline budget and must enforce it on their own
        I/O. They must not synchronously issue another RPC on this server.
        Unregistered requests remain visible to the caller for explicit denial.
        """
        with self._condition:
            self.assert_healthy()
            if not isinstance(method,str) or not method or not callable(handler) or method in self._request_handlers:
                raise ValueError('one explicit handler per admitted native request required')
            self._request_handlers[method]=handler
            self._condition.notify_all()

    def _take_handled_request(self):
        # Called under _condition; leave all ordinary events in original order.
        for index,(message,size) in enumerate(self._events):
            if 'id' in message and message['method'] in self._request_handlers:
                del self._events[index]
                self._queued -= size
                return message,self._request_handlers[message['method']]
        return None

    def _serve_request(self, handled, *, deadline):
        message,handler=handled
        try:
            remaining=min(8,deadline-time.monotonic())
            if remaining <= 0 or not isinstance(message.get('params'),dict):
                raise ValueError('invalid or expired host request')
            limit=time.monotonic()+remaining
            result=handler(message['params'],timeout_s=remaining)
            if time.monotonic() >= limit or not isinstance(result,dict):
                raise ValueError('host request failed its result or time contract')
            self.respond(message['id'],result=result)
        except Exception:
            self._fail('host_request_failed')
            raise RuntimeError('admitted native host request failed') from None

    def next_event(self, *, timeout_s: float = 1) -> dict | None:
        if not 0 < timeout_s <= 3600:
            raise ValueError('bounded native event wait required')
        deadline=time.monotonic()+timeout_s
        while True:
            with self._condition:
                self.assert_healthy()
                if not self._events:
                    remaining=deadline-time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(timeout=remaining)
                    continue
                message,size = self._events[0]
                if 'id' not in message or message['method'] not in self._request_handlers:
                    self._events.popleft()
                    self._queued -= size
                    return message
                self._events.popleft()
                self._queued -= size
                handled=(message,self._request_handlers[message['method']])
            # Polling duration limits waiting for an event, not the auth owner.
            self._serve_request(handled,deadline=time.monotonic()+8)

    def _dispatch(self, line):
        def pairs(items):
            value={}
            for key,item in items:
                if key in value:
                    raise ValueError('duplicate native protocol field')
                value[key]=item
            return value
        message = json.loads(line,object_pairs_hook=pairs,
            parse_constant=lambda value:(_ for _ in ()).throw(ValueError('non-finite native JSON')))
        if not isinstance(message,dict):
            raise ValueError('native protocol message is not an object')
        size = len(json.dumps(message,ensure_ascii=False).encode())
        with self._condition:
            if self._queued+size > MAX_QUEUED:
                raise ValueError('native pending event budget exceeded')
            if 'method' in message:
                if not isinstance(message['method'],str):
                    raise ValueError('native protocol method is invalid')
                if 'id' in message:
                    identity = message['id']
                    if type(identity) not in {str,int} or identity in self._server_requests:
                        raise ValueError('native request identity is duplicated or invalid')
                    self._server_requests.add(identity)
                self._events.append((message,size))
            else:
                identity = message.get('id')
                if type(identity) is not int or identity not in self._waiting or identity in self._responses:
                    raise ValueError('native reply does not match one pending request')
                self._responses[identity] = message
            self._queued += size
            self._condition.notify_all()

    def _read(self):
        streams = {self._process.stdout,self._process.stderr}
        pending = bytearray()
        try:
            while streams and not self._closed:
                ready,_,_ = select.select(list(streams),[],[],.1)
                if not ready and self._process.poll() is not None:
                    self._fail('native_process_exited')
                    break
                for stream in ready:
                    part = os.read(stream.fileno(),65536)
                    if not part:
                        streams.remove(stream)
                        continue
                    if stream is self._process.stderr:
                        available = 2*1024*1024-len(self.stderr)
                        self.stderr.extend(part[:available])
                        if len(part) > available:
                            raise ValueError('native stderr budget exceeded')
                        continue
                    pending.extend(part)
                    while b'\n' in pending:
                        line,_,rest = pending.partition(b'\n')
                        pending = bytearray(rest)
                        if len(line) > MAX_MESSAGE:
                            raise ValueError('native event line budget exceeded')
                        if line.strip():
                            self._dispatch(line)
                    if len(pending) > MAX_MESSAGE:
                        raise ValueError('native event line budget exceeded')
            if not self._closed:
                self._fail('native_process_exited')
        except Exception:
            if not self._closed:
                self._fail('native_event_stream_failed')

    def close(self):
        if self._closed:
            return
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        terminate_process_tree(self._process,own_process_group=True)
        self._process.wait(timeout=5)
        self._reader.join(timeout=2)
        self._process.stdin.close()
        self._process.stdout.close()
        self._process.stderr.close()
        if self._reader.is_alive():
            raise RuntimeError('native event reader failed to stop')

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
