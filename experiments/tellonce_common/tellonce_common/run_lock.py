"""Prevent two processes from advancing the same run directory concurrently."""
from contextlib import contextmanager
import fcntl
import os
import stat


@contextmanager
def run_lock(spec):
    path = spec.directory/'scheduler.lock'
    fd = os.open(path, os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError('invalid scheduler ownership file')
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)
