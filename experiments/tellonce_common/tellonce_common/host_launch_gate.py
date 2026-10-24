"""Space out experiment process launches and wait for free host memory."""
import fcntl
import json
import os
from pathlib import Path
import time


STATE = Path(os.environ.get('TELLONCE_EXP_LAUNCH_STATE', Path.home()/'.cache/tellonce-experiments/launch'))
HEADROOM = 20 * 1024**3
SPACING = 2.0


def available_bytes(text):
    for line in text.splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('MemAvailable missing')


def reason(available, now, previous):
    if available < HEADROOM:
        return 'host_memory'
    if previous is not None and 0 <= now - previous < SPACING:
        return 'startup_spacing'
    return None


def claim(directory=STATE):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous_path = directory/'last-launch.json'
        previous = json.loads(previous_path.read_text())['monotonic'] if previous_path.exists() else None
        now = time.monotonic()
        available = available_bytes(Path('/proc/meminfo').read_text())
        waiting = reason(available, now, previous)
        if waiting is None:
            temporary = directory/'last-launch.tmp'
            temporary.write_text(json.dumps({'monotonic': now, 'pid': os.getpid(),
                                             'available_bytes': available}))
            temporary.replace(previous_path)
        return waiting, available


def wait_for_admission(*, continue_waiting=None, sleep=time.sleep, directory=STATE):
    """Wait until the host has memory headroom and launches are spaced out."""
    waited = False
    while True:
        if continue_waiting is not None:
            continue_waiting()
        waiting, available = claim(directory)
        if waiting is None:
            return {'waited': waited, 'available_bytes': available}
        waited = True
        sleep(SPACING)
