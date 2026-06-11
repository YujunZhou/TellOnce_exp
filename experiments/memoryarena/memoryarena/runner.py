from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import re
import json
import os
import shutil
import uuid
from dataclasses import replace
from typing import Any

from .memoryarena_loader import load_rows, mock_group_travel_rows, validate_memoryarena_row
from .profiles import DEFAULT_PROFILES, profile_by_id
from .schemas import ConversationTurn, append_jsonl, to_jsonable, write_json
from .scoring import build_score
from .skill_manifest import compile_corrections_to_manifest
from .user_simulator import simulate_correction
from .wrapper import artifact_path_for_task, build_task_from_row, write_workspace
from .claude_cli_runtime import (
    generate_artifact_with_anthropic_api,
    generate_artifact_with_claude,
    generate_artifact_with_codex,
    generate_artifact_with_openai_api,
)


# This public release ships only TRACE (our method) and the No-Memory
# control. The memory baselines compared in the paper (Mem0, Hindsight,
# ReMe-Light) are intentionally not included; see the repository README.
CONDITIONS = ["no_memory", "trace_native_cc"]
TRACE_NATIVE_CONDITIONS = ["trace_native_cc", "trace_native_codex"]
VALID_CONDITIONS = ["no_memory", "compiled_enforcement", *TRACE_NATIVE_CONDITIONS]

MEMORY_RUN_VERSION = 2


def _oracle_dry_agent_artifact(task, condition: str, include_constraints: bool) -> dict:
    """Oracle artifact for schema dry-run only.

    This deliberately uses expected answers so the wrapper/scorer can be tested
    without a model. It must never be used for paper results.
    """
    artifact = {
        "session_id": task.session_id,
        "condition": condition,
        "oracle_dry_run": True,
        "answer": task.expected_answer,
        "constraint_notes": [],
    }
    if include_constraints:
        artifact["constraint_notes"] = [c["text"] for c in task.hidden_constraints]
    return artifact


def _pt_skill_dir() -> Path:
    return Path.home() / ".claude" / "skills" / "tellonce"


def _trace_workspace_settings_payload(
    skill_dir: Path | None = None,
    hooks: dict[str, list[tuple[str, int]]] | None = None,
) -> dict[str, Any]:
    skill_dir = skill_dir or _pt_skill_dir()
    hooks_dir = skill_dir / "hooks"
    hooks_block: dict[str, list[dict[str, Any]]] = {}
    for event, hook_list in (hooks or _PT_HOOKS_TRACE).items():
        entry: dict[str, Any] = {"matcher": "", "hooks": []}
        for basename, timeout in hook_list:
            entry["hooks"].append({
                "type": "command",
                "command": str(hooks_dir / basename),
                "timeout": timeout,
            })
        hooks_block[event] = [entry]
    return {"hooks": hooks_block}


def _write_trace_workspace_settings(
    workspace: Path,
    skill_dir: Path | None = None,
    hooks: dict[str, list[tuple[str, int]]] | None = None,
) -> Path:
    settings_dir = workspace / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.json"
    payload = _trace_workspace_settings_payload(skill_dir, hooks=hooks)
    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return settings_path


def _write_trace_heartbeat_hook(workspace: Path) -> Path:
    """Mirror ClawArena's per-workspace heartbeat hook so retrieve hook
    delta can be observed even when memory-retrieve-inject.sh is absent
    (e.g. unit tests on hosts that don't ship tellonce)."""
    script = (workspace / ".claude" / "trace_heartbeat.py").resolve()
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, time\n"
        "from pathlib import Path\n"
        "state = os.environ.get('B5_STATE_DIR')\n"
        "if state:\n"
        "    path = Path(state) / 'trace_hook_heartbeat.jsonl'\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    rec = {\n"
        "        'ts': time.time(),\n"
        "        'event': 'UserPromptSubmit',\n"
        "        'env_b5_memory_dir': os.environ.get('B5_MEMORY_DIR'),\n"
        "        'env_b5_state_dir': os.environ.get('B5_STATE_DIR'),\n"
        "        'env_b5_obs_log_dir': os.environ.get('B5_OBS_LOG_DIR'),\n"
        "    }\n"
        "    mem_dir = rec['env_b5_memory_dir']\n"
        "    if mem_dir and os.path.isdir(mem_dir):\n"
        "        rec['memory_dir_md_count'] = len([f for f in os.listdir(mem_dir) if f.endswith('.md')])\n"
        "    with path.open('a', encoding='utf-8') as f:\n"
        "        f.write(json.dumps(rec) + '\\n')\n"
        "    try:\n"
        "        os.chmod(path, 0o600)\n"
        "    except OSError:\n"
        "        pass\n",
        encoding="utf-8",
    )
    try:
        os.chmod(script, 0o755)
    except OSError:
        pass
    return script


def _trace_memory_state_hash(memory_dir: Path | None) -> str:
    h = hashlib.sha256()
    if memory_dir is None or not memory_dir.is_dir():
        h.update(b"<no_memory_dir>")
        return h.hexdigest()[:16]
    for path in sorted(memory_dir.rglob("*.md")):
        try:
            h.update(path.name.encode("utf-8"))
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
        except OSError:
            continue
    return h.hexdigest()[:16]


def _count_obs_log_events(
    obs_log_dir: Path | None,
    state_dir: Path | None,
) -> int:
    candidates: list[Path] = []
    if obs_log_dir and Path(obs_log_dir).is_dir():
        candidates.append(Path(obs_log_dir) / "pending_queue.jsonl")
        candidates.append(Path(obs_log_dir) / "compliance_log.jsonl")
    if state_dir and Path(state_dir).is_dir():
        candidates.append(Path(state_dir) / "trace_hook_heartbeat.jsonl")
        candidates.append(Path(state_dir) / "retrieve_debug.jsonl")
        candidates.append(Path(state_dir) / "runtime" / "retrieve_debug.jsonl")
    total = 0
    for log in candidates:
        if not log.is_file():
            continue
        try:
            with log.open(encoding="utf-8") as f:
                total += sum(1 for _ in f)
        except OSError:
            continue
    return total


def _compile_trace_requirement(correction_text: str) -> str:
    base = correction_text.strip()
    rule_groups = sorted(set(re.findall(r"\b(P[1-5]|O[1-4])(?:\b|_)", base)))
    if rule_groups:
        return f"Enforce rule groups {','.join(rule_groups)} in subsequent rounds: {base}"
    return f"Enforce in subsequent rounds: {base}"


class _MemoryArenaTraceState:
    """TRACE-native memory adapter for MemoryArena.

    Mirrors the design of ClawArena's TraceNativeCcCondition /
    TraceNativeCodexCondition: per (profile, condition) we allocate
    isolated B5_MEMORY_DIR / B5_STATE_DIR / B5_OBS_LOG_DIR. Phase A
    add_correction writes a real tellonce memory .md so the
    next round's UPS retrieve hook surfaces it. Phase B sets _frozen=True
    so add_correction is a no-op (proves the eval split was frozen).
    """

    def __init__(
        self,
        *,
        profile_id: str,
        condition: str,
        run_dir: Path,
        skill_dir: Path | None = None,
    ):
        self._profile_id = profile_id
        self._condition = condition
        self._is_codex = condition == "trace_native_codex"
        base = (run_dir / profile_id / condition).resolve()
        self._memory_dir = base / "trace_memory"
        self._state_dir = base / "trace_state"
        self._obs_log_dir = base / "trace_obs_log"
        for d in (self._memory_dir, self._state_dir, self._obs_log_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._skill_dir = skill_dir or _pt_skill_dir()
        self._frozen = False
        self._frozen_memory_hash: str | None = None
        # 2026-05-03 codex review fix: initialize _write_count from disk so
        # crash/orphan recovery doesn't reuse atomic_id `trace-pref-000_...`.
        # If a previous run wrote N .md files (whether resume-eligible or
        # orphan from crash mid-write), start counter at N to avoid filename
        # collision on the next add_correction.
        self._write_count = len(list(self._memory_dir.glob("trace-pref-*.md")))
        # 2026-05-04 orphan reconciliation: opt-in via env var
        # EVAL_TRACE_RECONCILE_ORPHANS=1. Scans records.jsonl (cell-level)
        # and quarantines .md files whose session_id key doesn't match any
        # successful record with corrections_from_this_round non-empty.
        # Default OFF to avoid touching existing data.
        if os.environ.get("EVAL_TRACE_RECONCILE_ORPHANS") == "1":
            self._reconcile_orphan_md(run_dir)

    def _reconcile_orphan_md(self, run_dir: Path) -> None:
        """Quarantine orphan trace-pref-*.md files in MemoryArena.

        records.jsonl lives at run_dir / 'records.jsonl' (cell-level, all
        profiles and conditions). Filter to records for THIS profile +
        condition; successful ones with non-empty corrections_from_this_round
        define the legitimate session_id keys. Orphans = .md whose filename
        suffix doesn't match any such session_id.
        """
        if not self._memory_dir.is_dir():
            return
        records_path = Path(run_dir) / "records.jsonl"
        if not records_path.is_file():
            return
        legit = set()
        try:
            with open(records_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("runtime_error"):
                        continue
                    if r.get("profile_id") != self._profile_id:
                        continue
                    if r.get("condition") != self._condition:
                        continue
                    corrs = r.get("corrections_from_this_round") or []
                    if not corrs:
                        continue
                    sid = r.get("session_id")
                    if sid:
                        legit.add(sid)
        except Exception:
            return
        orphan_dir = self._memory_dir / "_orphan_quarantine"
        moved = 0
        for md in list(self._memory_dir.glob("trace-pref-*.md")):
            stem = md.stem
            try:
                # filename: trace-pref-NNN_<session_id>
                key = "_".join(stem.split("_")[2:])
            except Exception:
                continue
            if key not in legit:
                orphan_dir.mkdir(parents=True, exist_ok=True)
                md.rename(orphan_dir / md.name)
                moved += 1
        if moved:
            print(
                f"[trace_reconcile] {self._profile_id}/{self._condition}: "
                f"quarantined {moved} orphan .md to {orphan_dir}",
                flush=True,
            )
        # Round 3 review fix: surface heartbeat hook registration failures
        # instead of swallowing in setup_workspace.
        self._setup_warning: str | None = None

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    @property
    def obs_log_dir(self) -> Path:
        return self._obs_log_dir

    @property
    def is_codex(self) -> bool:
        return self._is_codex

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def trace_mode(self) -> str:
        return "native_codex" if self._is_codex else "native_cc"

    def freeze(self) -> None:
        self._frozen = True
        self._frozen_memory_hash = self.memory_hash_now()

    def setup_workspace(self, workspace: Path) -> Path | None:
        """CC: drop workspace .claude/settings.json registering the
        non-blocking TRACE hook subset (UPS retrieve + Stop pending-promote)
        and the heartbeat hook so we can observe hook firings even on
        hosts without tellonce installed.

        Codex: returns None — codex reads ~/.codex/hooks.json globally.
        """
        if self._is_codex:
            return None
        settings_path = _write_trace_workspace_settings(
            workspace, skill_dir=self._skill_dir, hooks=_PT_HOOKS_TRACE
        )
        heartbeat = _write_trace_heartbeat_hook(workspace)
        try:
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
            ups = payload.setdefault("hooks", {}).setdefault("UserPromptSubmit", [])
            if not ups:
                ups.append({"matcher": "", "hooks": []})
            hooks = ups[0].setdefault("hooks", [])
            heartbeat_cmd = str(heartbeat)
            if not any(h.get("command") == heartbeat_cmd for h in hooks):
                hooks.insert(
                    0,
                    {
                        "type": "command",
                        "command": heartbeat_cmd,
                        "timeout": 5,
                    },
                )
            settings_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            # Round 3 review fix: previously a bare `except: pass` here would
            # silently mask any failure to register the heartbeat hook into
            # workspace settings.json, so trace_runtime_loaded could falsely
            # stay True even when the heartbeat path was broken. Surface it as
            # diagnostic state on self instead of swallowing.
            self._setup_warning = (
                f"heartbeat hook registration failed: {type(exc).__name__}: "
                f"{str(exc)[:240]}"
            )
            print(
                f"[trace_setup_workspace] WARNING: {self._setup_warning} "
                f"(workspace={workspace})",
                flush=True,
            )
        return settings_path

    def runtime_env(self) -> dict[str, str]:
        return {
            "B5_MEMORY_DIR": str(self._memory_dir),
            "B5_STATE_DIR": str(self._state_dir),
            "B5_OBS_LOG_DIR": str(self._obs_log_dir),
            "B5_RETRIEVE_DEBUG": "1",
        }

    def add_correction(self, session_id: str, text: str) -> None:
        if self._frozen:
            return
        if not text.strip():
            return
        from datetime import datetime as _dt, timezone as _tz

        ts = _dt.now(_tz.utc).isoformat()
        atomic_id = f"trace-pref-{self._write_count:03d}_{session_id}".replace(":", "_")
        atomic_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", atomic_id)
        requirement = _compile_trace_requirement(text)
        md = (
            "---\n"
            f"name: TRACE corrected preference {atomic_id}\n"
            f"description: MemoryArena harness-derived preference promoted from session {session_id}\n"
            "type: preference\n"
            "domain: workflow\n"
            "scope: project\n"
            "confidence: high\n"
            "priority_tier: 2\n"
            f"atomic_id: {atomic_id}\n"
            "applies_when: future sessions in this scenario context\n"
            "status: active\n"
            "lifecycle_state: confirmed\n"
            "memory_kind: procedural\n"
            f"created: {ts[:10]}\n"
            f"updated: {ts[:10]}\n"
            f"originSessionId: trace-{self.trace_mode().replace('_', '-')}-{session_id}\n"
            "---\n\n"
            "# Correction text\n\n"
            f"{text.strip()}\n\n"
            "# Compiled requirement\n\n"
            f"{requirement}\n"
        )
        try:
            self._memory_dir.mkdir(parents=True, exist_ok=True)
            (self._memory_dir / f"{atomic_id}.md").write_text(md, encoding="utf-8")
            self._write_count += 1
        except OSError:
            pass

    def retrieve(self, query: str, limit: int = 10) -> list[str]:
        del query  # MemoryArena trace retrieval is order-sensitive, not query-aware here
        notes: list[str] = []
        if not self._memory_dir.is_dir():
            return notes
        for path in sorted(self._memory_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            # Strip the YAML frontmatter so retrieved note is just the
            # correction body + compiled requirement.
            if text.startswith("---"):
                end = text.find("\n---", 3)
                if end < 0:
                    # Malformed frontmatter (no closing delimiter).
                    # Skip rather than leak `---` into the agent prompt.
                    continue
                text = text[end + 4 :].lstrip()
            stripped = text.strip()
            if not stripped:
                continue
            notes.append(stripped)
            if len(notes) >= limit:
                break
        return notes

    def memory_hash_now(self) -> str:
        return _trace_memory_state_hash(self._memory_dir)

    def memory_hash_frozen(self) -> str | None:
        return self._frozen_memory_hash

    def hook_event_count(self) -> int:
        return _count_obs_log_events(self._obs_log_dir, self._state_dir)

    def diagnostics(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "backend": self._condition,
            "backend_mode": "trace_native",
            "trace_mode": self.trace_mode(),
            "trace_skill_dir": str(self._skill_dir),
            "trace_skill_present": self._skill_dir.is_dir(),
            "trace_frozen": self._frozen,
            "trace_frozen_memory_hash": self._frozen_memory_hash,
            "trace_memory_dir": str(self._memory_dir),
            "trace_state_dir": str(self._state_dir),
            "trace_obs_log_dir": str(self._obs_log_dir),
        }
        if self._is_codex:
            codex_hooks = Path.home() / ".codex" / "hooks.json"
            d["trace_codex_hooks_path"] = str(codex_hooks)
            d["trace_codex_hooks_present"] = codex_hooks.is_file()
        return d


def _call_generate_artifact(
    *,
    runtime: str,
    session_root: Path,
    memory_text: str,
    model: str,
    timeout_s: int,
    max_budget_usd: float,
    runtime_retries: int,
    extra_env: dict[str, str] | None,
    trace_state: "_MemoryArenaTraceState | None",
    task: Any,
    condition: str,
) -> tuple[dict[str, Any], str | None]:
    """Single artifact-generation call dispatching on runtime. Factored out
    of the main loop so the phase-B retry block can call it identically to
    the initial pass. Returns (artifact, runtime_error). Mirrors the
    runner's pre-existing dispatch (runtime in {claude_cli, anthropic_api,
    openai_api, codex_cli, oracle_dry_run}). On exception we synthesise a
    null-answer artifact with runtime_error set, matching the prior inline
    behavior so the rest of the pipeline (scoring, record building) is
    unchanged.

    Anthropic-API path is preserved here ONLY because the existing runner
    already supports it (smoke harness use). Production memoryarena
    launches never set runtime=anthropic_api per
    [[feedback-no-anthropic-api-only-subscription]]; this path is dead in
    paper runs and is not touched by the self-check retry hook (the judge
    itself goes through DeepInfra, never Anthropic).
    """
    runtime_error: str | None = None
    include_constraints = condition == "compiled_enforcement"  # phase B caller asserts
    if runtime == "claude_cli":
        try:
            artifact = generate_artifact_with_claude(
                session_root=session_root,
                memory_text=memory_text,
                model=model,
                timeout_s=timeout_s,
                max_budget_usd=max_budget_usd,
                runtime_retries=runtime_retries,
                extra_env=extra_env,
            )
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            artifact = {
                "session_id": task.session_id,
                "model": model,
                "oracle_dry_run": False,
                "answer": None,
                "constraint_notes": [],
                "unresolved_questions": [],
                "decision_log": "",
                "runtime_error": runtime_error,
            }
    elif runtime == "anthropic_api":
        try:
            artifact = generate_artifact_with_anthropic_api(
                session_root=session_root,
                memory_text=memory_text,
                model=model,
                timeout_s=timeout_s,
                runtime_retries=runtime_retries,
            )
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            artifact = {
                "session_id": task.session_id,
                "model": model,
                "oracle_dry_run": False,
                "answer": None,
                "constraint_notes": [],
                "unresolved_questions": [],
                "decision_log": "",
                "runtime_error": runtime_error,
            }
    elif runtime == "openai_api":
        try:
            artifact = generate_artifact_with_openai_api(
                session_root=session_root,
                memory_text=memory_text,
                model=model,
                timeout_s=timeout_s,
                runtime_retries=runtime_retries,
            )
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            artifact = {
                "session_id": task.session_id,
                "model": model,
                "oracle_dry_run": False,
                "answer": None,
                "constraint_notes": [],
                "unresolved_questions": [],
                "decision_log": "",
                "runtime_error": runtime_error,
            }
    elif runtime == "codex_cli":
        try:
            artifact = generate_artifact_with_codex(
                session_root=session_root,
                memory_text=memory_text,
                model=model,
                timeout_s=timeout_s,
                runtime_retries=runtime_retries,
                extra_env=extra_env,
                preserve_user_config=(
                    trace_state is not None and trace_state.is_codex
                ),
            )
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            artifact = {
                "session_id": task.session_id,
                "model": model,
                "oracle_dry_run": False,
                "answer": None,
                "constraint_notes": [],
                "unresolved_questions": [],
                "decision_log": "",
                "runtime_error": runtime_error,
            }
    else:
        artifact = _oracle_dry_agent_artifact(task, condition, include_constraints)
    return artifact, runtime_error


def _memory_text_for_condition(
    *,
    condition: str,
    corrections: list[ConversationTurn],
    trace_state: "_MemoryArenaTraceState | None",
    task_question: str,
    phase: str = "A",
) -> tuple[str, list[str], str | None, dict[str, Any]]:
    if condition == "compiled_enforcement":
        notes = [t.text for t in corrections]
        return "\n".join(notes), notes, None, {}
    if condition in TRACE_NATIVE_CONDITIONS and trace_state is not None:
        notes = trace_state.retrieve(task_question, limit=10)
        diagnostics = trace_state.diagnostics()
        return "\n".join(notes), notes, diagnostics.get("backend_mode"), diagnostics
    return "", [], None, {}


def _correction_turn_to_record(turn: ConversationTurn) -> dict[str, Any]:
    return to_jsonable(turn)


def _correction_turns_from_record(record: dict[str, Any]) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    for idx, item in enumerate(record.get("corrections_from_this_round") or []):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        turns.append(
            ConversationTurn(
                turn_id=int(item.get("turn_id") or idx + 1),
                speaker="user_simulator",
                turn_type=item.get("turn_type") or "project_correction",
                visible_to_agent=bool(item.get("visible_to_agent", True)),
                text=text,
                linked_hidden_constraint_ids=list(item.get("linked_hidden_constraint_ids") or []),
                reveals_gold_answer=bool(item.get("reveals_gold_answer", False)),
                allowed_information_level=str(item.get("allowed_information_level") or "constraint_only"),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return turns


def _restore_learning_from_record(
    *,
    record: dict[str, Any],
    corrections: list[ConversationTurn],
    trace_state: "_MemoryArenaTraceState | None" = None,
) -> None:
    if record.get("phase") != "A":
        return
    for correction in _correction_turns_from_record(record):
        corrections.append(correction)
        if trace_state is not None:
            trace_state.add_correction(str(record.get("session_id") or "resume"), correction.text)


def _record_eval_domain(record: dict[str, Any]) -> str:
    existing = record.get("eval_domain")
    if existing:
        return str(existing)
    return "train" if str(record["phase"]) == "A" else "test"


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record["user_id"]),
        str(record["condition"]),
        _record_eval_domain(record),
        str(record["phase"]),
        str(record["session_id"]),
    )


def _record_is_resume_success(record: dict[str, Any]) -> bool:
    if record.get("runtime_error"):
        return False
    if record.get("memory_run_version") != MEMORY_RUN_VERSION:
        return False
    if not isinstance(record.get("memory_run_id"), str) or not record.get("memory_run_id"):
        return False
    if record.get("runtime") == "codex_cli":
        provenance = record.get("runtime_provenance")
        if not isinstance(provenance, dict):
            return False
        if provenance.get("codex_hook_isolation_version") != 2:
            return False
        if provenance.get("codex_runtime_variant") == "codex_cli_trace":
            # TRACE-native codex condition deliberately preserves user
            # CODEX_HOME so ~/.codex/hooks.json is visible to the run.
            return record.get("condition") == "trace_native_codex"
        return provenance.get("codex_home_isolated") is True
    if record.get("runtime") == "openai_api":
        provenance = record.get("runtime_provenance")
        if not isinstance(provenance, dict):
            return False
        return (
            provenance.get("openai_structured_outputs_version") == 3
            and provenance.get("openai_max_output_tokens") == 4096
        )
    return True


def _namespace_part(value: str) -> str:
    part = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return part.strip("._-") or "default"


def _memory_run_id_from_records(records: list[dict]) -> str | None:
    run_ids = {
        str(record.get("memory_run_id"))
        for record in records
        if record.get("memory_run_version") == MEMORY_RUN_VERSION
        and isinstance(record.get("memory_run_id"), str)
        and record.get("memory_run_id")
    }
    if len(run_ids) == 1:
        return next(iter(run_ids))
    return None


def _ensure_memory_run_id(*, out_dir: Path, resume: bool, records: list[dict]) -> str:
    run_id = _memory_run_id_from_records(records) if resume and records else None
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]
    write_json(
        out_dir / "memory_run_meta.json",
        {"memory_run_version": MEMORY_RUN_VERSION, "memory_run_id": run_id},
    )
    return run_id


def _phase_plan(
    *,
    rows: list[dict],
    train_rows: list[dict] | None,
    test_rows: list[dict] | None,
    id_test_rows: list[dict] | None,
    ood_test_rows: list[dict] | None,
    phase_a: int,
    phase_b: int,
) -> list[tuple[dict, str, str, int, int]]:
    if any(x is not None for x in (train_rows, test_rows, id_test_rows, ood_test_rows)):
        plan: list[tuple[str, list[dict] | None, str, int, int]] = [
            ("A", train_rows, "train", phase_a, 0),
            ("B", test_rows, "test", phase_b, 0),
            ("B", id_test_rows, "id_test", phase_b, 0),
            ("B", ood_test_rows, "ood_test", phase_b, 0),
        ]
        return [
            (row, phase, eval_domain, n_sessions, offset)
            for phase, phase_rows, eval_domain, n_sessions, offset in plan
            for row in (phase_rows or [])
        ]
    return [
        (row, phase, "train" if phase == "A" else "test", n_sessions, offset)
        for row in rows
        for phase, n_sessions, offset in (("A", phase_a, 0), ("B", phase_b, phase_a))
    ]


def _rewrite_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    with tmp.open("r", encoding="utf-8") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _dedupe_latest_records(records: list[dict]) -> list[dict]:
    latest_by_key: dict[tuple[str, str, str, str, str], dict] = {}
    key_order: list[tuple[str, str, str, str, str]] = []
    for record in records:
        record_key = _record_key(record)
        if record_key not in latest_by_key:
            key_order.append(record_key)
        latest_by_key[record_key] = record
    return [latest_by_key[key] for key in key_order]


def _load_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _load_row_files(paths: list[Path] | None, limit: int | None) -> list[dict] | None:
    if not paths:
        return None
    rows: list[dict] = []
    for path in paths:
        rows.extend(load_rows(path, limit=limit))
    return rows


def run_dry_matrix(
    *,
    rows: list[dict],
    out_dir: Path,
    profile_ids: list[str],
    phase_a: int,
    phase_b: int,
    seed: int | None = None,
    runtime: str = "oracle_dry_run",
    model: str = "haiku",
    timeout_s: int = 120,
    max_budget_usd: float = 0.5,
    runtime_retries: int = 1,
    max_records: int | None = None,
    conditions: list[str] | None = None,
    train_rows: list[dict] | None = None,
    test_rows: list[dict] | None = None,
    id_test_rows: list[dict] | None = None,
    ood_test_rows: list[dict] | None = None,
    resume: bool = False,
) -> list[dict]:
    records: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    records_jsonl = out_dir / "records.jsonl"
    completed: set[tuple[str, str, str, str, str]] = set()
    existing_by_key: dict[tuple[str, str, str, str, str], dict] = {}
    if resume and records_jsonl.exists():
        loaded_records = _load_jsonl_records(records_jsonl)
        deduped = _dedupe_latest_records(loaded_records)
        records = [
            record
            for record in deduped
            if _record_is_resume_success(record)
        ]
        stale_dropped = len(deduped) - len(records)
        if stale_dropped > 0:
            print(
                f"[resume] WARNING: dropped {stale_dropped}/{len(deduped)} stale/v1 records "
                f"(missing memory_run_version=2 or runtime_provenance v3); "
                f"keeping {len(records)} resume-eligible records",
                flush=True,
            )
        _rewrite_jsonl(records_jsonl, records)
        for record in records:
            record_key = _record_key(record)
            completed.add(record_key)
            existing_by_key[record_key] = record
    else:
        records_jsonl.write_text("", encoding="utf-8")
    memory_run_id = _ensure_memory_run_id(out_dir=out_dir, resume=resume, records=records)
    all_rows = list(rows)
    if any(x is not None for x in (train_rows, test_rows, id_test_rows, ood_test_rows)):
        all_rows = (
            list(train_rows or [])
            + list(test_rows or [])
            + list(id_test_rows or [])
            + list(ood_test_rows or [])
        )
    for row in all_rows:
        errors = validate_memoryarena_row(row)
        if errors:
            raise ValueError(f"invalid MemoryArena row {row.get('id')}: {errors}")

    run_conditions = conditions or CONDITIONS
    row_phase_iter = _phase_plan(
        rows=rows,
        train_rows=train_rows,
        test_rows=test_rows,
        id_test_rows=id_test_rows,
        ood_test_rows=ood_test_rows,
        phase_a=phase_a,
        phase_b=phase_b,
    )
    for profile_id in profile_ids:
        profile = profile_by_id(profile_id)
        for condition in run_conditions:
            if condition not in VALID_CONDITIONS:
                raise ValueError(f"unknown condition: {condition}")
            phase_a_corrections: list[ConversationTurn] = []
            trace_state = (
                _MemoryArenaTraceState(
                    profile_id=profile_id,
                    condition=condition,
                    run_dir=out_dir,
                )
                if condition in TRACE_NATIVE_CONDITIONS
                else None
            )
            for row, phase, eval_domain, n_sessions, offset in row_phase_iter:
                n_available = max(0, len(row.get("questions") or []) - offset)
                n_effective = min(n_sessions, n_available)
                for local_idx in range(n_effective):
                    session_index = offset + local_idx
                    task = build_task_from_row(
                        row,
                        user_profile=profile,
                        phase=phase,
                        session_index=session_index,
                    )
                    record_key = (profile.user_id, condition, eval_domain, phase, task.session_id)
                    if record_key in completed:
                        existing_record = existing_by_key.get(record_key)
                        if existing_record is not None:
                            _restore_learning_from_record(
                                record=existing_record,
                                corrections=phase_a_corrections,
                                trace_state=trace_state,
                            )
                        continue
                    session_root = out_dir / profile.user_id / condition / eval_domain / task.session_id
                    # Round 3 review fix (D3): when session_index>0 in same row,
                    # carry over accepted_constraints / accepted_decisions / prior
                    # artifact from the prior session so q1->q2->q3 share state.
                    prior_session_root: Path | None = None
                    if session_index > 0:
                        prior_task = build_task_from_row(
                            row,
                            user_profile=profile,
                            phase=phase,
                            session_index=session_index - 1,
                        )
                        candidate = (
                            out_dir / profile.user_id / condition / eval_domain / prior_task.session_id
                        )
                        if (candidate / "input" / "project_state.json").is_file():
                            prior_session_root = candidate
                    write_workspace(session_root, task, profile, prior_session_root=prior_session_root)
                    workspace_settings_path: Path | None = None
                    if trace_state is not None:
                        workspace_settings_path = trace_state.setup_workspace(session_root)
                        if phase == "B" and not trace_state.is_frozen:
                            trace_state.freeze()

                    include_constraints = condition == "compiled_enforcement" and phase == "B"
                    runtime_error = None
                    (
                        memory_text,
                        retrieved_memory,
                        memory_backend_mode,
                        memory_backend_diagnostics,
                    ) = _memory_text_for_condition(
                        condition=condition,
                        corrections=phase_a_corrections,
                        trace_state=trace_state,
                        task_question=task.question,
                        phase=phase,
                    )
                    trace_extra_env = (
                        trace_state.runtime_env() if trace_state is not None else None
                    )
                    trace_hook_events_before = (
                        trace_state.hook_event_count() if trace_state is not None else 0
                    )
                    trace_memory_hash_before = (
                        trace_state.memory_hash_now() if trace_state is not None else None
                    )
                    artifact, runtime_error = _call_generate_artifact(
                        runtime=runtime,
                        session_root=session_root,
                        memory_text=memory_text,
                        model=model,
                        timeout_s=timeout_s,
                        max_budget_usd=max_budget_usd,
                        runtime_retries=runtime_retries,
                        extra_env=trace_extra_env,
                        trace_state=trace_state,
                        task=task,
                        condition=condition,
                    )
                    agent_turn = ConversationTurn(
                        turn_id=0,
                        speaker="agent",
                        turn_type="agent_response",
                        visible_to_agent=True,
                        text=json.dumps(artifact, ensure_ascii=False),
                    )
                    user_turns = simulate_correction(
                        user_profile=profile,
                        current_artifact=artifact,
                        expected_answer=task.expected_answer,
                        turn_id=1,
                    )
                    conversation = [agent_turn, *user_turns]
                    for turn in conversation:
                        append_jsonl(session_root / "logs" / "conversation.jsonl", turn)
                    round_corrections: list[ConversationTurn] = []
                    if phase == "A":
                        for t in user_turns:
                            if t.turn_type in {"preference_correction", "project_correction"}:
                                correction = replace(
                                    t,
                                    metadata={
                                        **t.metadata,
                                        "session_id": task.session_id,
                                        "phase": phase,
                                        "correction_id": (
                                            f"{profile.user_id}:{task.session_id}:"
                                            f"turn_{t.turn_id}"
                                        )
                                    },
                                )
                                phase_a_corrections.append(correction)
                                round_corrections.append(correction)
                                if trace_state is not None:
                                    trace_state.add_correction(task.session_id, correction.text)
                    if condition == "compiled_enforcement":
                        manifest = compile_corrections_to_manifest(
                            user_profile=profile,
                            corrections=phase_a_corrections,
                            seed=seed,
                            subset=str(task.metadata.get("subset") or "group_travel_planner"),
                            artifact_path=str(task.metadata.get("artifact_path") or "workspace/trip_plan.json"),
                        )
                    else:
                        manifest = {"version": "memoryarena_v1", "skills": [], "by_domain": {}}
                    write_json(session_root / "skills" / "manifest.json", manifest)
                    write_json(artifact_path_for_task(session_root, task), artifact)
                    score = build_score(
                        session_id=task.session_id,
                        condition=condition,
                        phase=phase,
                        predicted_answer=artifact["answer"],
                        expected_answer=task.expected_answer,
                        hard_constraints=task.hidden_constraints,
                        conversation=conversation,
                        correction_opportunities=task.correction_opportunities,
                        retry_count=1 if include_constraints else 0,
                        constraint_artifact=artifact,
                    )
                    write_json(session_root / "logs" / "score.json", score)
                    trace_hook_events_after = (
                        trace_state.hook_event_count() if trace_state is not None else 0
                    )
                    trace_memory_hash_after = (
                        trace_state.memory_hash_now() if trace_state is not None else None
                    )
                    if trace_state is not None:
                        memory_write_count_this_round = (
                            0
                            if trace_state.is_frozen
                            else int(
                                trace_memory_hash_before is not None
                                and trace_memory_hash_after is not None
                                and trace_memory_hash_before != trace_memory_hash_after
                            )
                        )
                    else:
                        memory_write_count_this_round = 0
                    record = {
                        "profile_id": profile.user_id,
                        "user_id": profile.user_id,
                        "profile_type": profile.profile_type,
                        "condition": condition,
                        "phase": phase,
                        "eval_domain": eval_domain,
                        "session_id": task.session_id,
                        "runtime": runtime,
                        "model": model,
                        "memory_run_version": MEMORY_RUN_VERSION,
                        "memory_run_id": memory_run_id,
                        "subset": task.metadata.get("subset"),
                        "artifact_path": task.metadata.get("artifact_path"),
                        "runtime_error": runtime_error,
                        "runtime_provenance": artifact.get("_runtime_provenance", {}),
                        "memory_notes_before": retrieved_memory,
                        "memory_note_count": len(retrieved_memory),
                        "memory_backend_mode": memory_backend_mode,
                        "memory_backend_diagnostics": memory_backend_diagnostics,
                        "corrections_from_this_round": [
                            _correction_turn_to_record(t) for t in round_corrections
                        ],
                        # ---- TRACE-native fields (mirror ClawArena multi_runner) ----
                        "trace_mode": (
                            trace_state.trace_mode() if trace_state is not None else "n/a"
                        ),
                        "trace_runtime_loaded": (
                            trace_state is not None
                            and trace_hook_events_after > trace_hook_events_before
                        ),
                        "trace_frozen": (
                            trace_state.is_frozen if trace_state is not None else False
                        ),
                        "memory_state_hash_before": trace_memory_hash_before,
                        "memory_state_hash_after": trace_memory_hash_after,
                        "memory_write_count_this_round": memory_write_count_this_round,
                        "hook_events_before": trace_hook_events_before,
                        "hook_events_after": trace_hook_events_after,
                        "hook_events_delta": (
                            trace_hook_events_after - trace_hook_events_before
                        ),
                        "workspace_settings_path": (
                            str(workspace_settings_path)
                            if workspace_settings_path and workspace_settings_path.is_file()
                            else None
                        ),
                        "score": to_jsonable(score),
                    }
                    records.append(record)
                    with records_jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
                    if max_records is not None and len(records) >= max_records:
                        records = _dedupe_latest_records(records)
                        _rewrite_jsonl(records_jsonl, records)
                        write_json(out_dir / "summary_records.json", records)
                        return records
    records = _dedupe_latest_records(records)
    _rewrite_jsonl(records_jsonl, records)
    write_json(out_dir / "summary_records.json", records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=None, help="Local MemoryArena JSON/JSONL export")
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--train-rows", type=Path, nargs="+", default=None)
    parser.add_argument("--train-row-limit", type=int, default=None, help="Per-file row limit for --train-rows")
    parser.add_argument("--test-rows", type=Path, nargs="+", default=None)
    parser.add_argument("--test-row-limit", type=int, default=None)
    parser.add_argument("--id-test-rows", type=Path, nargs="+", default=None)
    parser.add_argument("--id-test-row-limit", type=int, default=None, help="Per-file row limit for --id-test-rows")
    parser.add_argument("--ood-test-rows", type=Path, nargs="+", default=None)
    parser.add_argument("--ood-test-row-limit", type=int, default=None, help="Per-file row limit for --ood-test-rows")
    parser.add_argument("--mock-data", action="store_true", help="Use deterministic local mock rows")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", default=[p.user_id for p in DEFAULT_PROFILES])
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--phase-a", type=int, default=2)
    parser.add_argument("--phase-b", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runtime", choices=["oracle_dry_run", "claude_cli", "anthropic_api", "openai_api", "codex_cli"], default="oracle_dry_run")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-budget-usd", type=float, default=0.5)
    parser.add_argument("--runtime-retries", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    model = args.model or ("gpt-5.4-mini" if args.runtime in {"codex_cli", "openai_api"} else "haiku")

    train_rows = None
    test_rows = None
    id_test_rows = None
    ood_test_rows = None
    if any((args.train_rows, args.test_rows, args.id_test_rows, args.ood_test_rows)):
        if args.rows or args.mock_data:
            parser.error("do not combine --rows/--mock-data with split-row inputs")
        if not args.train_rows:
            parser.error("provide --train-rows when using split-row inputs")
        if args.test_rows and (args.id_test_rows or args.ood_test_rows):
            parser.error("use either --test-rows or --id-test-rows/--ood-test-rows, not both")
        if not args.test_rows and not (args.id_test_rows or args.ood_test_rows):
            parser.error("provide --test-rows or at least one of --id-test-rows/--ood-test-rows")
        train_rows = _load_row_files(args.train_rows, limit=args.train_row_limit)
        if args.test_rows:
            test_rows = _load_row_files(args.test_rows, limit=args.test_row_limit)
        if args.id_test_rows:
            id_test_rows = _load_row_files(args.id_test_rows, limit=args.id_test_row_limit)
        if args.ood_test_rows:
            ood_test_rows = _load_row_files(args.ood_test_rows, limit=args.ood_test_row_limit)
        rows = []
    elif args.mock_data:
        rows = mock_group_travel_rows(n_rows=1, n_sessions=args.phase_a + args.phase_b)
    elif args.rows:
        rows = load_rows(args.rows, limit=args.row_limit)
    else:
        parser.error("provide --rows or --mock-data")

    records = run_dry_matrix(
        rows=rows,
        out_dir=args.out_dir,
        profile_ids=args.profiles,
        phase_a=args.phase_a,
        phase_b=args.phase_b,
        seed=args.seed,
        runtime=args.runtime,
        model=model,
        timeout_s=args.timeout_s,
        max_budget_usd=args.max_budget_usd,
        runtime_retries=args.runtime_retries,
        max_records=args.max_records,
        conditions=args.conditions,
        train_rows=train_rows,
        test_rows=test_rows,
        id_test_rows=id_test_rows,
        ood_test_rows=ood_test_rows,
        resume=args.resume,
    )
    print(f"wrote {len(records)} records to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
