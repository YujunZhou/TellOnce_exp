"""Frozen-memory test worker: test, evaluate, or a single parallel test task (worker)."""
from pathlib import Path
import sys

experiment=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(experiment/name) for name in ('tellonce_common','tellonce_clawarena','tellonce_memoryarena')]

from tellonce_common.plan_a_worker import main

if __name__=='__main__':
    raise SystemExit(main())
