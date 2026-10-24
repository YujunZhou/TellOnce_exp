"""Locate the code that phase workers launch.

Workers start their subprocesses from this repository (`experiments/`) and load
the TellOnce library from a checkout of https://github.com/YujunZhou/tellonce,
named by $TELLONCE_SKILL_DIR (run_trace.py sets it from --skill).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def tellonce_root():
    value = os.environ.get('TELLONCE_SKILL_DIR')
    if not value:
        raise ValueError('set TELLONCE_SKILL_DIR (or pass --skill) to a tellonce checkout')
    root = Path(value).expanduser().resolve()
    if not (root/'lib'/'memory_upsert.py').is_file():
        raise ValueError(f'{root} is not a tellonce checkout (lib/memory_upsert.py missing)')
    return root


def source_bundle():
    return {'roots': {'project': str(REPO_ROOT), 'tellonce': str(tellonce_root())}}
