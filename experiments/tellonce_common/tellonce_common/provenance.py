from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"tree hash path is not a directory: {root}")
    digest = hashlib.sha256()
    children = (item for item in root.rglob("*") if item.is_file())
    for child in sorted(children, key=lambda item: item.relative_to(root).as_posix()):
        relative = child.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
