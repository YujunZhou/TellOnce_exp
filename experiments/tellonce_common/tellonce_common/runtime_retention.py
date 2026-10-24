"""Bounded debug samples from discarded, uncommitted task attempts only."""
import json
from pathlib import Path
import shutil
import time

from .atomic_io import atomic_write_bytes
from .provenance import canonical_json_bytes


def compact_abandoned(root, *, keep=2, max_case_bytes=8*1024**2):
    root=Path(root)
    if root.resolve()!=root or root.is_symlink():raise ValueError('owned real abandoned directory required')
    if not root.exists():return
    cases=root/'cases';cases.mkdir(exist_ok=True,mode=0o700)
    for attempt in sorted(root.iterdir()):
        if attempt==cases:continue
        if attempt.is_symlink() or not attempt.is_dir() or not attempt.name.startswith('task-'):
            raise ValueError('unexpected file in abandoned task storage')
        # These records suffice for debugging the interrupted attempt. Native
        # account homes and arbitrary workspace files are never copied here.
        candidates=[attempt/'task-result.json',*sorted((attempt/'captures').glob('*.json'))]
        saved={};used=0
        for path in candidates:
            if not path.is_file() or path.is_symlink():continue
            size=path.stat().st_size
            if size>max_case_bytes-used:continue
            saved[str(path.relative_to(attempt))]=path.read_text();used+=size
        case={'schema':1,'attempt':attempt.name,'retained_at':time.time(),'records':saved,
              'note':'Uncommitted task: excluded from metrics; only bounded debug records retained.'}
        atomic_write_bytes(cases/(attempt.name+'.json'),canonical_json_bytes(case))
        shutil.rmtree(attempt)
    paths=sorted(cases.glob('task-*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in paths[keep:]:
        if path.is_symlink():raise ValueError('debug case path redirected')
        path.unlink()
