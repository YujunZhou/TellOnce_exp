"""Workspace and update handling for the experiments ClawArena harness."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


FRAMEWORK_BY_RUNTIME = {
    "oracle_dry_run": "claude-code",
    "claude_cli": "claude-code",
    "codex_cli": "codex",
    # codex_cli_trace is the TRACE-native variant of codex_cli (no
    # --ignore-user-config / --ignore-rules). Same framework as codex_cli
    # for workspace setup (AGENTS.md, no .claude/settings.json).
    "codex_cli_trace": "codex",
}


def framework_for_runtime(runtime: str) -> str:
    try:
        return FRAMEWORK_BY_RUNTIME[runtime]
    except KeyError as exc:
        raise KeyError(f"unknown runtime {runtime!r}") from exc


def manifest_path(clawarena_root: Path, framework: str) -> Path:
    return clawarena_root / "data" / "clawarena" / framework / "manifest.json"


def load_manifest(clawarena_root: Path, framework: str) -> dict[str, Any]:
    path = manifest_path(clawarena_root, framework)
    return json.loads(path.read_text(encoding="utf-8"))


def eval_dir(clawarena_root: Path) -> Path:
    return clawarena_root / "data" / "clawarena" / "eval"


def prepare_workspace(
    *,
    clawarena_root: Path,
    runtime: str,
    model: str,
    scenario_id: str,
    condition: str,
    profile_id: str,
    out_dir: Path,
    resume: bool,
) -> Path:
    framework = framework_for_runtime(runtime)
    src = (
        clawarena_root
        / "data"
        / "clawarena"
        / framework
        / "workspaces"
        / scenario_id
    )
    if not src.is_dir():
        raise FileNotFoundError(f"ClawArena workspace not found: {src}")
    dst = (
        out_dir
        / "workspaces"
        / runtime
        / _safe_segment(model)
        / condition
        / profile_id
        / scenario_id
    )
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    if framework == "claude-code":
        settings_src = clawarena_root / "data" / "clawarena" / framework / "config" / "settings.json"
        if settings_src.exists():
            settings_dst = dst / ".claude" / "settings.json"
            settings_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(settings_src, settings_dst)
    elif framework == "codex":
        (dst / "AGENTS.md").write_text(
            "Use targeted file inspection only. Do not print entire files, large "
            "CSV files, or full message logs. Prefer `rg -n` with small context, "
            "`sed -n` windows of at most 80 lines, and `head -5` for CSV headers. "
            "For multi-choice rounds, run at most four shell commands before "
            "answering. If evidence is incomplete, answer from the best available "
            "small excerpts and state remaining uncertainty in the JSON fields. "
            "Return the final answer as exactly one JSON object requested by the "
            "task prompt.\n",
            encoding="utf-8",
        )
        _write_codex_output_limiter(dst)
    _write_applied_updates(dst, [])
    return dst


def _write_codex_output_limiter(workspace: Path) -> None:
    bin_dir = workspace / ".eval_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    scripts = {
        "cat": "#!/usr/bin/env bash\n/usr/bin/cat \"$@\" | /usr/bin/head -c \"${EVAL_MAX_TOOL_BYTES:-2500}\"\n",
        "sed": "#!/usr/bin/env bash\n/usr/bin/sed \"$@\" | /usr/bin/head -c \"${EVAL_MAX_TOOL_BYTES:-2500}\"\n",
        "rg": "#!/usr/bin/env bash\n/usr/bin/rg \"$@\" | /usr/bin/head -c \"${EVAL_MAX_TOOL_BYTES:-2500}\"\n",
        "grep": "#!/usr/bin/env bash\n/usr/bin/grep \"$@\" | /usr/bin/head -c \"${EVAL_MAX_TOOL_BYTES:-2500}\"\n",
    }
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)


def apply_updates(
    *,
    clawarena_root: Path,
    runtime: str,
    scenario_id: str,
    workspace: Path,
    update_ids: list[str],
) -> list[dict[str, Any]]:
    if not update_ids:
        return []
    framework = framework_for_runtime(runtime)
    manifest = load_manifest(clawarena_root, framework)
    manifest_dir = manifest_path(clawarena_root, framework).parent
    updates = manifest.get("updates", {}).get(scenario_id, {})
    applied = set(_read_applied_updates(workspace))
    events: list[dict[str, Any]] = []
    for update_id in _expand_update_ids(update_ids, updates):
        if update_id in applied:
            events.append({"update_id": update_id, "status": "already_applied", "files": []})
            continue
        update_meta = updates.get(update_id)
        if not isinstance(update_meta, dict):
            events.append(
                {
                    "update_id": update_id,
                    "status": "missing",
                    "type": None,
                    "files": [],
                }
            )
            continue
        source_dir = manifest_dir / str(update_meta.get("dir") or "")
        copied = _apply_update_files(source_dir, workspace, update_meta.get("files") or [])
        applied.add(update_id)
        _write_applied_updates(workspace, sorted(applied))
        events.append(
            {
                "update_id": update_id,
                "status": "applied",
                "type": update_meta.get("type"),
                "files": copied,
            }
        )
    return events


def _expand_update_ids(update_ids: list[str], updates: dict[str, Any]) -> list[str]:
    expanded: list[str] = []
    for update_id in update_ids:
        raw = str(update_id)
        if raw in updates:
            expanded.append(raw)
            continue
        update_number = _update_number_alias(raw)
        if update_number is not None:
            aliases = [f"upd{update_number}_sessions", f"upd{update_number}_workspace"]
            matched = [alias for alias in aliases if alias in updates]
            if matched:
                expanded.extend(matched)
                continue
        expanded.append(raw)
    return expanded


def _update_number_alias(update_id: str) -> str | None:
    raw = update_id.strip()
    if raw.isdigit():
        return raw
    for prefix in ("U", "u", "upd", "UPD"):
        if raw.startswith(prefix) and raw[len(prefix) :].isdigit():
            return raw[len(prefix) :]
    return None


def current_applied_updates(workspace: Path) -> list[str]:
    return _read_applied_updates(workspace)


def dataset_hash(clawarena_root: Path, runtime: str, scenario_id: str) -> str:
    framework = framework_for_runtime(runtime)
    paths = [
        clawarena_root / "data" / "clawarena" / "eval" / scenario_id / "questions.json",
        manifest_path(clawarena_root, framework),
        clawarena_root / "data" / "clawarena" / framework / "config" / "settings.json",
    ]
    h = hashlib.sha256()
    for path in paths:
        _hash_path(h, path, clawarena_root)
    for rel in [
        ("data", "clawarena", "eval", scenario_id, "scripts"),
        ("data", "clawarena", framework, "workspaces", scenario_id),
        ("data", "clawarena", framework, "updates", scenario_id),
    ]:
        path = clawarena_root.joinpath(*rel)
        if path.exists():
            _hash_path(h, path, clawarena_root)
    return h.hexdigest()


def harness_hash(package_dir: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        _hash_path(h, path, package_dir.parent)
    return h.hexdigest()


def snapshot_workspace(*, out_dir: Path, workspace: Path, record: dict[str, Any]) -> Path:
    dst = _snapshot_path(out_dir, record)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    marker = _applied_updates_path(workspace)
    if marker.exists():
        marker_dst = _snapshot_marker_path(dst)
        marker_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(marker, marker_dst)
    fsync_parent(dst)
    return dst


def restore_workspace_snapshot(*, out_dir: Path, workspace: Path, record: dict[str, Any]) -> None:
    src = _snapshot_path(out_dir, record)
    if not src.is_dir():
        raise FileNotFoundError(f"missing workspace snapshot for resume: {src}")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, workspace)
    marker_src = _snapshot_marker_path(src)
    if marker_src.exists():
        shutil.copy2(marker_src, _applied_updates_path(workspace))


def _snapshot_path(out_dir: Path, record: dict[str, Any]) -> Path:
    return (
        out_dir
        / "workspace_snapshots"
        / str(record["runtime"])
        / _safe_segment(str(record["model"]))
        / str(record["condition"])
        / str(record["profile_id"])
        / f"{int(record['round_index']):04d}_{record['round_id']}"
    )


def _snapshot_marker_path(snapshot_path: Path) -> Path:
    return snapshot_path.parent / f"{snapshot_path.name}.applied_updates.json"


def fsync_parent(path: Path) -> None:
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _hash_path(h: "hashlib._Hash", path: Path, root: Path) -> None:
    if path.is_file():
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            _hash_path(h, child, root)


def _apply_update_files(
    source_dir: Path, workspace: Path, files: list[dict[str, Any] | str]
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for item in files:
        if isinstance(item, str):
            filename = item
            action = "new"
            target = filename
        else:
            filename = str(item.get("name") or "")
            action = str(item.get("action") or "new")
            target = str(item.get("target") or filename)
        if not filename or not target:
            raise ValueError(f"malformed update file spec: {item!r}")
        src = source_dir / filename
        dst = workspace / target
        if action == "new":
            if not src.exists():
                raise FileNotFoundError(f"update source not found: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif action == "append":
            if not src.exists():
                raise FileNotFoundError(f"update source not found: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("a", encoding="utf-8") as f:
                f.write(src.read_text(encoding="utf-8"))
        elif action == "insert":
            if not src.exists():
                raise FileNotFoundError(f"update source not found: {src}")
            param = item.get("param", {}) if isinstance(item, dict) else {}
            after = int(param.get("after", 0)) if isinstance(param, dict) else 0
            existing = dst.read_text(encoding="utf-8").splitlines(keepends=True) if dst.exists() else []
            new_content = "".join(existing[:after]) + src.read_text(encoding="utf-8") + "".join(existing[after:])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(new_content, encoding="utf-8")
        elif action == "delete":
            if dst.exists():
                param = item.get("param", {}) if isinstance(item, dict) else {}
                line_specs = param.get("lines", []) if isinstance(param, dict) else []
                existing = dst.read_text(encoding="utf-8").splitlines(keepends=True)
                delete_indexes: set[int] = set()
                for spec in line_specs:
                    if isinstance(spec, int):
                        delete_indexes.add(spec - 1)
                    elif isinstance(spec, str) and ":" in spec:
                        start_s, end_s = spec.split(":", 1)
                        start = int(start_s) if start_s else 0
                        end = int(end_s) if end_s else len(existing)
                        delete_indexes.update(range(start, end))
                dst.write_text(
                    "".join(line for idx, line in enumerate(existing) if idx not in delete_indexes),
                    encoding="utf-8",
                )
        else:
            raise ValueError(f"unsupported update action {action!r}")
        copied.append({"source": str(src), "target": target, "action": action})
    return copied


def _applied_updates_path(workspace: Path) -> Path:
    return workspace.parent / f".{workspace.name}.eval_applied_updates.json"


def _read_applied_updates(workspace: Path) -> list[str]:
    path = _applied_updates_path(workspace)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"malformed applied update state: {path}")
    return [str(x) for x in data]


def _write_applied_updates(workspace: Path, update_ids: list[str]) -> None:
    path = _applied_updates_path(workspace)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(update_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    fsync_parent(path)


def _safe_segment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:120] or "model"
