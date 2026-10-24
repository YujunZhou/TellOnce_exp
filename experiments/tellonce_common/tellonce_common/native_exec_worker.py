"""Bind a native stdio process to its direct host before exec, retaining its PID."""
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tellonce_common.process_control import bind_parent_lifetime


def main():
    try:
        if len(sys.argv)<5 or sys.argv[1]!='--owner-pid' or sys.argv[3]!='--':
            raise ValueError('explicit native owner and command required')
        bind_parent_lifetime(int(sys.argv[2]))
        command=sys.argv[4:]
        os.execvpe(command[0],command,dict(os.environ))
    except Exception as exc:
        # No arguments, environment values, or raw errors leave this wrapper.
        print('Native process lifetime initialization failed: '+type(exc).__name__,file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
