"""Discover and trust only explicitly reviewed hooks in one native CLI home."""
import json
import hashlib
import os
from pathlib import Path
import re
import select
import subprocess
import time

from .codex_text import text_environment
from .process_control import terminate_process_tree


def discover_hooks(*, binary: Path, cli_home: Path, workspace: Path, timeout_s: float = 60,
                   config_arguments: tuple[str, ...] = (), record=None) -> dict:
    # hooks/list is a local startup probe, but it launches a fresh native
    # app-server. Under the full 30-group schedule its startup can be delayed
    # by local process and I/O contention; keep the probe bounded while leaving
    # enough room for the admitted server to initialize.
    if any(not p.is_absolute() or p != p.resolve() for p in (binary, cli_home, workspace)):
        raise ValueError('hook discovery requires canonical explicit runtime paths')
    if not 0 < timeout_s <= 60:
        raise ValueError('bounded hook discovery timeout required')
    if len(config_arguments)%2 or any(config_arguments[i]!='-c' for i in range(0,len(config_arguments),2)):
        raise ValueError('hook probe accepts only explicit configuration overrides')
    if record is not None and not callable(record):
        raise ValueError('hook diagnostic recorder must be callable')
    process = subprocess.Popen([str(binary),'app-server','-c','features.hooks=true',
                                '-c','features.plugins=false','-c','features.apps=false',*config_arguments],
        cwd=workspace, env=text_environment(cli_home=cli_home,host_home=cli_home/'host-home'),
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    started = time.monotonic()
    deadline, buffer = started+timeout_s, b''
    streams = [process.stdout, process.stderr]
    error_tail, error_bytes, error_hash = b'', 0, hashlib.sha256()
    failure = 'startup_timeout'
    try:
        for message in (
            {'jsonrpc':'2.0','id':1,'method':'initialize','params':{'clientInfo':{'name':'experiment-hook-check','version':'1'}}},
            {'jsonrpc':'2.0','method':'initialized','params':{}},
            {'jsonrpc':'2.0','id':2,'method':'hooks/list','params':{'cwds':[str(workspace)]}}):
            process.stdin.write((json.dumps(message)+'\n').encode())
            process.stdin.flush()
        while time.monotonic() < deadline:
            ready,_,_ = select.select(streams,[],[],min(.2,max(0,deadline-time.monotonic())))
            if not ready:
                if process.poll() is not None:
                    failure = 'process_exited'
                    break
                continue
            if process.stderr in ready:
                part = os.read(process.stderr.fileno(),65536)
                if not part:
                    streams.remove(process.stderr)
                else:
                    error_bytes += len(part)
                    error_hash.update(part)
                    error_tail = (error_tail + part)[-16384:]
            if process.stdout not in ready:
                continue
            part = os.read(process.stdout.fileno(),65536)
            if not part:
                failure = 'stdout_closed'
                break
            buffer += part
            if len(buffer) > 2*1024*1024:
                failure = 'response_budget'
                raise RuntimeError('native hook response exceeded its budget')
            while b'\n' in buffer:
                line,buffer = buffer.split(b'\n',1)
                failure = 'invalid_protocol'
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise RuntimeError('native hook discovery response is not an object')
                if message.get('id') == 2:
                    if 'error' in message or not isinstance(message.get('result'),dict):
                        raise RuntimeError('native hook discovery failed')
                    return message['result']
                failure = 'startup_timeout'
        raise RuntimeError('native hook discovery timed out or exited')
    except Exception as exc:
        if isinstance(exc, BrokenPipeError):
            failure = 'stdin_closed'
        if record is not None:
            # Raw stderr and exception messages can contain private paths or
            # credentials. Retain only bounded, allowlisted diagnostic facts.
            categories = [(b'Resource temporarily unavailable', 'resource_unavailable'),
                          (b'Too many open files', 'file_descriptor_limit'),
                          (b'No space left on device', 'storage_unavailable'),
                          (b'Permission denied', 'permission_denied'),
                          (b'panicked at', 'native_panic')]
            record({'type': 'native_hook_discovery_failed', 'status': failure,
                    'exit_code': process.poll(), 'wall_s': time.monotonic()-started,
                    'stderr_observed_bytes': error_bytes, 'stderr_observed_sha256': error_hash.hexdigest(),
                    'stderr_complete': process.stderr not in streams,
                    'stderr_categories': [label for marker, label in categories if marker in error_tail]})
        raise
    finally:
        terminate_process_tree(process,own_process_group=True)
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def hook_trust_arguments(discovery: dict, *, hooks_file: Path, workspace: Path,
                         expected: tuple[dict, ...]) -> list[str]:
    """Bind native normalized handler hashes to the exact declared handlers.

    This never approves project/plugin/managed additions. The supervisor must
    separately verify hook executable/dependency file hashes against its source
    bundle before each launch: a native handler hash does not hash Python code.
    """
    entries = discovery.get('data')
    if (not isinstance(entries,list) or len(entries)!=1 or entries[0].get('cwd')!=str(workspace) or
            entries[0].get('errors') or entries[0].get('warnings')):
        raise ValueError('hook discovery contains errors or unexpected workspace entries')
    actual = entries[0].get('hooks')
    if not isinstance(actual,list) or len(actual)!=len(expected):
        raise ValueError('native hooks differ from the declared experiment hooks')
    wanted = {(e['eventName'],e['command'],e['matcher'],e['timeoutSec']) for e in expected}
    if len(wanted)!=len(expected):
        raise ValueError('duplicate expected hook handler')
    states, seen, keys = [], set(), set()
    for hook in actual:
        identity = (hook.get('eventName'),hook.get('command'),hook.get('matcher'),hook.get('timeoutSec'))
        if (identity not in wanted or identity in seen or hook.get('handlerType')!='command' or
                hook.get('sourcePath')!=str(hooks_file) or hook.get('isManaged') is not False or
                hook.get('enabled') is not True or hook.get('async') is True):
            raise ValueError('unreviewed or incorrectly configured native hook')
        key, digest = hook.get('key'),hook.get('currentHash')
        if (not isinstance(key,str) or not key.startswith(str(hooks_file)+':') or key in keys or
                not isinstance(digest,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',digest)):
            raise ValueError('native hook has invalid trust identity')
        seen.add(identity)
        keys.add(key)
        states.append(json.dumps(key)+'={trusted_hash='+json.dumps(digest)+'}')
    # A dotted -c key splits periods inside paths (including hooks.json).
    # Supply the entire literal table so native handler IDs remain exact keys.
    return ['-c','hooks={state={'+','.join(states)+'}}']
