"""Host-private, task-owned channel for short-lived native hook processes.

The supervisor owns the handler and must call assert_healthy before accepting
any delivery. A broken channel is an infrastructure failure, not a rule verdict.
Keep its directory outside the solver's allowed filesystem. No model runs here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import struct
import sys
import threading
import tempfile
import time
import uuid

if __package__:
    from .request_budget import request_budget
else:  # The reviewed native hook command executes this file directly.
    from request_budget import request_budget

MAX_MESSAGE = 1024 * 1024
FAILURE_MESSAGE = 'Task rule-check channel unavailable; supervisor must stop and record infrastructure failure.'


def _encode(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()+b'\n'
    if len(encoded) > MAX_MESSAGE:
        raise ValueError('hook message exceeds the explicit transport budget')
    return encoded


def _read(connection):
    data = bytearray()
    while len(data) < MAX_MESSAGE:
        part = connection.recv(min(65536, MAX_MESSAGE-len(data)))
        if not part:
            raise ValueError('incomplete hook message')
        data.extend(part)
        if b'\n' in part:
            line, tail = bytes(data).split(b'\n', 1)
            if tail:
                raise ValueError('one request per hook connection is required')
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('duplicate hook JSON field')
                    result[key] = value
                return result
            value = json.loads(line, object_pairs_hook=pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite hook JSON')))
            if not isinstance(value, dict):
                raise ValueError('hook message must be an object')
            return value
    raise ValueError('hook message exceeds the explicit transport budget')


def request_hook(path: Path, payload: dict, *, request_id: str | None = None, timeout_s: float = 30) -> dict:
    if not isinstance(payload, dict) or not 0 < timeout_s <= 300:
        raise ValueError('a bounded hook request is required')
    envelope = _encode({'request_id':request_id or str(uuid.uuid4()), 'payload':payload,
                       'deadline':time.monotonic()+timeout_s})
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_s)
        connection.connect(str(path))
        connection.sendall(envelope)
        reply = _read(connection)
    if reply.get('ok') is not True or not isinstance(reply.get('result'), dict):
        raise RuntimeError('task hook bridge failed; supervisor must reject this turn')
    return reply['result']


class HookBridge:
    def __init__(self, *, directory: Path, handle, timeout_s: float = 30):
        if not directory.is_absolute() or directory != directory.resolve() or not 0 < timeout_s <= 300:
            raise ValueError('a canonical private directory and bounded timeout are required')
        self.directory, self.path = directory, directory/'hook.sock'
        self._socket_directory = None
        self.handle, self.timeout_s = handle, timeout_s
        self._stop, self._failure, self._cache = threading.Event(), None, {}
        self._thread, self._socket = None, None
        self._owned, self._cache_bytes = False, 0

    def start(self):
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._owned = True
        try:
            if len(os.fsencode(self.path)) > 100:
                # The public workspace remains the only solver-visible task
                # tree. Put only the transient socket in a fresh sibling under
                # an existing private ancestor; never expose /tmp globally or
                # abbreviate the persistent task/evaluation paths.
                for parent in self.directory.parents:
                    info = parent.stat()
                    candidate = parent/('.hook-'+'x'*8)/'hook.sock'
                    if (len(os.fsencode(candidate)) <= 100 and stat.S_ISDIR(info.st_mode) and
                            info.st_uid == os.getuid() and not info.st_mode & 0o077):
                        self._socket_directory = Path(tempfile.mkdtemp(prefix='.hook-', dir=parent))
                        self.path = self._socket_directory/'hook.sock'
                        break
                else:
                    raise ValueError('no sufficiently short private ancestor for the task socket')
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self._socket.listen(8)
            self._socket.settimeout(.1)
            self._thread = threading.Thread(target=self._serve, daemon=True, name='task-hook-bridge')
            self._thread.start()
        except Exception:
            if self._socket is not None:
                self._socket.close()
            self.path.unlink(missing_ok=True)
            if self._socket_directory is not None:
                self._socket_directory.rmdir()
                self._socket_directory = None
            raise
        return self

    def _serve(self):
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    self._failure = 'accept_failed'
                break
            with connection:
                connection.settimeout(self.timeout_s)
                try:
                    if self._failure is not None:
                        raise RuntimeError('previous hook failure')
                    # Linux native CLI hooks run as the supervisor's user.
                    _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.getuid():
                        raise ValueError('foreign hook process owner')
                    request = _read(connection)
                    identity, payload = request.get('request_id'), request.get('payload')
                    deadline=request.get('deadline')
                    if (set(request) != {'request_id','payload','deadline'} or not isinstance(identity,str) or
                            not 0 < len(identity) <= 128 or not isinstance(payload,dict) or
                            type(deadline) not in {int,float} or not math.isfinite(deadline)):
                        raise ValueError('invalid hook request')
                    fingerprint = hashlib.sha256(_encode(payload)).hexdigest()
                    if identity in self._cache:
                        previous, response = self._cache[identity]
                        if previous != fingerprint:
                            raise ValueError('replayed hook request changed its payload')
                    else:
                        if len(self._cache) >= 4096:
                            raise ValueError('task hook request budget exhausted')
                        remaining=min(self.timeout_s,deadline-time.monotonic())
                        if remaining<=0:
                            raise TimeoutError('native hook arrived after its client deadline')
                        # Queued hooks retain the client's original deadline;
                        # they cannot each restart a full model timeout after
                        # waiting behind another handler.
                        try:
                            with request_budget(remaining-min(10,remaining*.2)):
                                result = self.handle(payload)
                        except Exception as exc:
                            self._record_handler_failure(exc, payload)
                            # A handler failure belongs to this turn.  The
                            # caller receives a blocked hook result and must
                            # abort that turn, but a transient callback
                            # timeout must not poison the bridge for later
                            # turns.  Request parsing, peer validation, and
                            # listener failures still take the outer failure
                            # path below.
                            # The client may have timed out and closed its
                            # turn-local connection while the callback was
                            # unwinding.  That is not a listener failure and
                            # must not poison later turns.
                            try:
                                connection.sendall(_encode({'ok':False, 'error':'hook_bridge_failed'}))
                            except OSError:
                                pass
                            continue
                        if not isinstance(result, dict):
                            raise ValueError('hook handler must return an object')
                        response = _encode({'ok':True, 'result':result})
                        if self._cache_bytes+len(response) > 16*1024*1024:
                            raise ValueError('task hook replay storage budget exhausted')
                        self._cache[identity] = fingerprint, response
                        self._cache_bytes += len(response)
                    # A reply is scoped to this one hook connection.  A
                    # timed-out native client may already have gone away;
                    # dropping that reply must not mark the shared bridge
                    # unhealthy for subsequent turns.
                    try:
                        connection.sendall(response)
                    except OSError:
                        pass
                except Exception as exc:
                    # Never serialize exception text: callbacks may contain private evidence.
                    self._failure = type(exc).__name__
                    try:
                        connection.sendall(_encode({'ok':False, 'error':'hook_bridge_failed'}))
                    except OSError:
                        pass

    def _record_handler_failure(self, exc, payload):
        # Keep one host-private diagnostic, without exception messages,
        # source lines, locals, request contents, or absolute source paths.
        frames = []
        tb = exc.__traceback__
        while tb is not None:
            frames.append({'file': Path(tb.tb_frame.f_code.co_filename).name,
                           'function': tb.tb_frame.f_code.co_name,
                           'line': tb.tb_lineno})
            frames = frames[-12:]
            tb = tb.tb_next
        kind = payload.get('hook_event_name')
        value = {'error_type': type(exc).__name__, 'frames': frames,
                 'hook_event_name': kind if kind in {'PreToolUse', 'PostToolUse', 'Stop'} else None,
                 'at': time.time()}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.directory,
                                             prefix='.failure-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, allow_nan=False)
                stream.write('\n')
            temporary.replace(self.directory/'latest-handler-failure.json')
        except OSError:
            # Diagnostic storage failure must not change turn rejection or
            # poison a healthy listener; the client still receives failure.
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def assert_healthy(self):
        if self._failure is not None or self._thread is None or not self._thread.is_alive():
            raise RuntimeError('task hook bridge is unhealthy; do not accept or score the turn')

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_s+1)
            if self._thread.is_alive():
                raise RuntimeError('hook handler did not stop; task supervisor must terminate its worker')
        if self._socket is not None:
            self._socket.close()
        if self._owned:
            self.path.unlink(missing_ok=True)
            if self._socket_directory is not None:
                self._socket_directory.rmdir()
                self._socket_directory = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    try:
        data = sys.stdin.buffer.read(MAX_MESSAGE+1)
        if len(data) > MAX_MESSAGE:
            raise ValueError('native hook input exceeded budget')
        payload = json.loads(data)
        result = request_hook(args.socket, payload, timeout_s=args.timeout)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        print(FAILURE_MESSAGE, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
