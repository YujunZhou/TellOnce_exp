"""Identify processes this experiment started, by PID and kernel start tick."""
import os
from pathlib import Path


def process_identity(pid):
    """Only used for a PID obtained from our own Popen and saved start tick."""
    if type(pid) is not int or pid<=0:
        raise ValueError('positive owned process PID required')
    path=Path('/proc')/str(pid)
    try:
        if path.stat().st_uid!=os.getuid():
            return None
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return {'start':fields[19],'state':fields[0]}
    except (FileNotFoundError,ProcessLookupError):
        return None


def owned_process_alive(pid,start):
    identity=process_identity(pid)
    return identity is not None and identity['start']==start and identity['state'] not in {'Z','X'}
