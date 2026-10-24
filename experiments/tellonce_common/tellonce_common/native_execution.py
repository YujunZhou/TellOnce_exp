"""Bounded native event streaming; no implicit model retry or session resume."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time

from .process_control import terminate_process_tree


@dataclass(frozen=True)
class CommandCapture:
    stdout: str
    stderr: str
    exit_code: int
    status: str


def run_native_command(command: list[str], *, cwd: Path, env: dict[str,str], prompt: str,
                       timeout_s: float, on_event, stdout_budget: int = 16*1024*1024,
                       stderr_budget: int = 2*1024*1024,on_started=None,cancelled=None,
                       input_budget: int = 1024*1024, pass_fds: tuple[int, ...] = ()) -> CommandCapture:
    """Observe thread.started before servicing native hook requests.

    Callbacks must be short, nonblocking identity/audit operations. Model checks
    run in the separate host hook worker. Non-completed status invalidates the
    turn even if the child happened to exit zero. Captured streams are private
    audit material, never automatically forwarded to users or feedback agents.
    Pass on_event=None only for bounded raw CLI configuration output; that
    caller must validate the complete output format itself.
    """
    if (type(input_budget) is not int or not 1 <= input_budget <= 4*1024*1024 or
            not isinstance(prompt,str) or len(prompt.encode()) > input_budget or
            not 0 < timeout_s <= 86400 or min(stdout_budget,stderr_budget) <= 0 or
            (on_started is not None and not callable(on_started)) or
            (cancelled is not None and not isinstance(cancelled,threading.Event))):
        raise ValueError('native execution requires bounded input, time and output')
    if (not isinstance(pass_fds,tuple) or len(set(pass_fds))!=len(pass_fds) or
            any(type(fd) is not int or fd<3 for fd in pass_fds)):
        raise ValueError('only explicit owned credential descriptors may be inherited')
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, pass_fds=pass_fds)
    output, errors, pending = bytearray(), bytearray(), bytearray()
    write_failed = threading.Event()
    def send_prompt():
        try:
            process.stdin.write(prompt.encode())
            process.stdin.close()
        except (OSError, ValueError):
            write_failed.set()
    writer = threading.Thread(target=send_prompt, daemon=True)
    writer.start()
    status, deadline = 'completed', time.monotonic()+timeout_s
    streams = {process.stdout: (output, stdout_budget), process.stderr: (errors, stderr_budget)}
    try:
        if on_started is not None:
            try:
                on_started(process)
            except Exception:
                status='start_observer_failed'
        while status=='completed' and (streams or process.poll() is None):
            if cancelled is not None and cancelled.is_set():
                status='cancelled'
                break
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                status = 'timeout'
                break
            ready, _, _ = select.select(list(streams), [], [], min(.1, remaining))
            for stream in ready:
                part = os.read(stream.fileno(), 65536)
                if not part:
                    del streams[stream]
                    continue
                destination, budget = streams[stream]
                available = budget-len(destination)
                destination.extend(part[:available])
                if len(part) > available:
                    status = 'output_budget_exceeded'
                    break
                if stream is process.stdout and on_event is not None:
                    pending.extend(part)
                    while b'\n' in pending:
                        line, _, rest = pending.partition(b'\n')
                        pending = bytearray(rest)
                        if not line.strip():
                            continue
                        try:
                            value = json.loads(line)
                            if not isinstance(value,dict):
                                raise ValueError('non-object native event')
                        except (ValueError, UnicodeError):
                            status = 'invalid_event'
                            break
                        try:
                            on_event(value)
                        except Exception:
                            # Tool evidence pairing may finish its bounded wait
                            # after the native turn's deadline. Preserve that
                            # local timeout instead of blaming a helper service
                            # whose request shared the same expired budget.
                            # Both outcomes reject the turn; callers retain
                            # their private observer diagnostic separately.
                            status = ('timeout' if time.monotonic() >= deadline
                                      else 'event_observer_failed')
                            break
                if status != 'completed':
                    break
            if status != 'completed':
                break
        if status == 'completed' and pending.strip():
            status = 'incomplete_event'
        if status == 'completed' and write_failed.is_set():
            status = 'input_write_failed'
    finally:
        # Own the process group from creation, including a departed leader's children.
        terminate_process_tree(process, own_process_group=True)
        process.wait(timeout=5)
        writer.join(timeout=1)
        process.stdout.close()
        process.stderr.close()
        if not writer.is_alive():
            process.stdin.close()
    return CommandCapture(output.decode(errors='replace'), errors.decode(errors='replace'),
                          process.returncode, status)
