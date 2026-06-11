"""Condition state for the ClawArena experiment pilot."""
from __future__ import annotations

import hashlib
import json
import os
import re
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompiledRule:
    rule_id: str
    trigger: str
    requirement: str
    source_round_id: str
    violation_type: str

    def as_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "trigger": self.trigger,
            "requirement": self.requirement,
            "source_round_id": self.source_round_id,
            "violation_type": self.violation_type,
        }


@dataclass
class ConditionState:
    name: str
    profile: dict[str, Any]
    corrections: list[str] = field(default_factory=list)
    compiled_rules: list[CompiledRule] = field(default_factory=list)

    def memory_notes(self) -> list[str]:
        return []

    def enforcement_rules(self) -> list[dict[str, str]]:
        return []

    def diagnostics(self) -> dict[str, Any]:
        return {}

    def prepare_round(
        self,
        *,
        scenario_id: str,
        round_id: str,
        question: str,
        update_events: list[dict[str, Any]],
    ) -> None:
        del scenario_id, round_id, question, update_events

    def observe_record(self, record: dict[str, Any]) -> None:
        correction = record.get("correction_text")
        if isinstance(correction, str) and correction.strip():
            self.observe_correction(
                round_id=str(record.get("global_round_id") or record.get("round_id") or "unknown"),
                correction_text=correction,
                violation_type=str(record.get("violation_type") or "task_error"),
            )

    def observe_correction(
        self, round_id: str, correction_text: str, violation_type: str
    ) -> None:
        return None


class NoMemoryCondition(ConditionState):
    def __init__(self, profile: dict[str, Any]):
        super().__init__(name="no_memory", profile=profile)


class CompiledEnforcementCondition(ConditionState):
    def __init__(self, profile: dict[str, Any]):
        super().__init__(name="compiled_enforcement", profile=profile)

    def memory_notes(self) -> list[str]:
        return []

    def enforcement_rules(self) -> list[dict[str, str]]:
        return [rule.as_dict() for rule in self.compiled_rules]

    def observe_correction(
        self, round_id: str, correction_text: str, violation_type: str
    ) -> None:
        if violation_type != "preference_violation":
            return
        self.corrections.append(correction_text)
        rule_id = f"live_{len(self.compiled_rules) + 1:03d}_{round_id}"
        requirement = _compile_requirement(correction_text, violation_type)
        self.compiled_rules.append(
            CompiledRule(
                rule_id=rule_id,
                trigger="future rounds with conflicting or updated project evidence",
                requirement=requirement,
                source_round_id=round_id,
                violation_type=violation_type,
            )
        )


def build_condition(name: str, profile: dict[str, Any]) -> ConditionState:
    # This public release ships only TRACE (our method) and the No-Memory
    # control. The memory baselines compared in the paper (Mem0, Hindsight,
    # ReMe-Light) are intentionally not included here; see the repository
    # README for the rationale.
    if name == "no_memory":
        return NoMemoryCondition(profile)
    if name == "compiled_enforcement":
        return CompiledEnforcementCondition(profile)
    if name == "trace_native_cc":
        return TraceNativeCcCondition(profile)
    if name == "trace_native_codex":
        return TraceNativeCodexCondition(profile)
    raise KeyError(
        f"unknown condition {name!r}; expected one of "
        "no_memory/compiled_enforcement/trace_native_cc/trace_native_codex"
    )


# ---------------------------------------------------------------------------
# TRACE-native-CC condition (gap-review M1 + M2 + M3 + M4, 2026-05-02)
# ---------------------------------------------------------------------------

# Hooks we expect preference-tracker to register, all under
# ~/.claude/skills/preference-tracker/hooks/. Keep in sync with
# lib/_install_merge_settings.py PT_HOOKS dict.
_PT_HOOKS = {
    "Stop": [
        ("check-observation-log.sh", 5),
        ("memory-deterministic-block.sh", 10),
        ("memory-verify-compliance.sh", 20),
        ("memory-shadow-judge.sh", 90),
        ("memory-pending-promote.sh", 10),
    ],
    "UserPromptSubmit": [
        ("memory-retrieve-inject.sh", 10),
        ("memory-pending-inject.sh", 5),
        ("memory-shadow-alert-inject.sh", 5),
    ],
}


# Round-7b (TRACE_NATIVE_CC_PATCH_REVIEW C0 fix, 2026-05-02): the experiments
# experiments register a CURATED non-blocking subset. The full _PT_HOOKS
# above includes Stop hooks (check-observation-log,
# memory-deterministic-block, memory-verify-compliance) that can return
# decision=block, which leaves Claude with terminal_reason=stop_hook_prevented
# and an empty answer that the ClawArena harness can't parse. For TRACE
# evidence we only need:
#   - UserPromptSubmit retrieve-inject: proves runtime loaded + injects rules
#   - Stop pending-promote: writes observed corrections into per-run memory
# All blocking gates are skipped; final-answer enforcement is the harness's
# job (verify_compliance equivalent runs in score_response/overlay), not
# the inner Claude session's hook chain.
# 2026-05-03 fix (B5_MEMORY_DIR isolation): user-global
# memory-retrieve-inject.sh ALSO loads ~/.claude/skills/preference-tracker/
# lib/fingerprints.yaml and the user's personal memory dir, leaking ~19
# external rules into the experiment regardless of B5_MEMORY_DIR. Drop it.
# Runner builds memory_text from condition.memory_notes() (which reads
# B5_MEMORY_DIR/*.md from disk) and feeds it directly into the agent prompt;
# UPS hook retrieve was a redundant + leaky second injection path. Stop
# pending-promote stays as diagnostic evidence the Stop chain loaded.
_PT_HOOKS_TRACE = {
    "Stop": [
        ("memory-pending-promote.sh", 10),
    ],
    "UserPromptSubmit": [],
}


def _pt_skill_dir() -> Path:
    return Path.home() / ".claude" / "skills" / "preference-tracker"


def workspace_settings_payload(
    skill_dir: Path | None = None,
    hooks: dict[str, list[tuple[str, int]]] | None = None,
) -> dict[str, Any]:
    """Build the .claude/settings.json contents that load preference-tracker
    hooks. Default `hooks` is the non-blocking TRACE subset (`_PT_HOOKS_TRACE`):
    only UPS retrieve + Stop pending-promote. Pass `hooks=_PT_HOOKS` for the
    full chain when blocking is acceptable."""
    skill_dir = skill_dir or _pt_skill_dir()
    hooks_dir = skill_dir / "hooks"
    hooks_block: dict[str, list[dict[str, Any]]] = {}
    for event, hook_list in (hooks or _PT_HOOKS_TRACE).items():
        entry = {"matcher": "", "hooks": []}
        for basename, timeout in hook_list:
            entry["hooks"].append({
                "type": "command",
                "command": str(hooks_dir / basename),
                "timeout": timeout,
            })
        hooks_block[event] = [entry]
    return {"hooks": hooks_block}


def write_workspace_settings(
    workspace: Path,
    skill_dir: Path | None = None,
    hooks: dict[str, list[tuple[str, int]]] | None = None,
) -> Path:
    """Drop a workspace-local .claude/settings.json. Default registers the
    non-blocking TRACE hook subset; pass `hooks=_PT_HOOKS` for the full
    blocking chain. Idempotent."""
    settings_dir = workspace / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.json"
    payload = workspace_settings_payload(skill_dir, hooks=hooks)
    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return settings_path


def memory_state_hash(memory_dir: Path | None = None) -> str:
    """Hash a preference-tracker memory directory contents so we can prove
    the eval split was frozen (hash before == hash after).

    TRACE_NATIVE_CC_PATCH_REVIEW C2 fix: when memory_dir is None, prefer the
    explicit B5_MEMORY_DIR env (the per-run isolated dir we set on the inner
    claude subprocess), then ~/.preference-tracker.config.json memory_dir
    (matches what hooks see), then cwd-derived fallback last. Without this
    we used to hash a path that didn't exist while hooks wrote elsewhere.
    """
    if memory_dir is None:
        env_dir = os.environ.get("B5_MEMORY_DIR")
        if env_dir:
            memory_dir = Path(env_dir)
        else:
            cfg_path = Path.home() / ".preference-tracker.config.json"
            try:
                cfg_memory = json.loads(cfg_path.read_text())["memory_dir"]
                memory_dir = Path(cfg_memory)
            except Exception:
                cwd = Path.cwd().resolve()
                cwd_escaped = str(cwd).replace("/", "-")
                memory_dir = Path.home() / ".claude" / "projects" / cwd_escaped / "memory"
    h = hashlib.sha256()
    if not memory_dir.is_dir():
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


def count_obs_log_events(
    obs_log_dir: Path | None = None,
    state_dir: Path | None = None,
) -> int:
    """Count hook-side log lines (proxy for hook firing). Used by
    multi_runner to compute trace_runtime_loaded as event-delta, not just
    settings-file presence.

    Round-7c (TRACE_NATIVE_CC_PREFLIGHT_RERUN 2026-05-02 fix): the TRACE
    non-blocking hook subset (UPS retrieve + Stop pending-promote) does
    NOT write to compliance_log.jsonl. Earlier C3 patch only counted
    that file, so trace_runtime_loaded stayed false even when hooks
    fired correctly. Now we count any of the JSONL artifacts these
    hooks actually produce:
      <state_dir>/runtime/retrieve_debug.jsonl   (UPS retrieve, when B5_RETRIEVE_DEBUG=1)
      <obs_log_dir>/pending_queue.jsonl          (Stop pending-promote)
      <obs_log_dir>/compliance_log.jsonl         (legacy / blocking-subset hooks)
    """
    # Round-7c review fix: empty env value -> Path("") resolves to cwd,
    # which is a dir and could false-count any unrelated pending_queue.jsonl
    # there. Treat empty env strings as unset.
    if obs_log_dir is None:
        env_v = os.environ.get("B5_OBS_LOG_DIR", "")
        obs_log_dir = Path(env_v) if env_v else None
    if state_dir is None:
        env_v = os.environ.get("B5_STATE_DIR", "")
        state_dir = Path(env_v) if env_v else None
    candidates: list[Path] = []
    if obs_log_dir and Path(obs_log_dir).is_dir():
        candidates.append(Path(obs_log_dir) / "pending_queue.jsonl")
        candidates.append(Path(obs_log_dir) / "compliance_log.jsonl")
    if state_dir and Path(state_dir).is_dir():
        candidates.append(Path(state_dir) / "trace_hook_heartbeat.jsonl")
        candidates.append(Path(state_dir) / "retrieve_debug.jsonl")
        # state_dir may itself nest a runtime/ subdir under some configs.
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


class TraceNativeCcCondition(CompiledEnforcementCondition):
    """TRACE-native condition where retrieval, correction, and memory write
    all happen via real preference-tracker runtime hooks plus per-run
    isolated memory/state/obs_log dirs.

    Round-7b (TRACE_NATIVE_CC_PATCH_REVIEW 2026-05-02 fixes):
    - C0/C2: per-run isolated B5_MEMORY_DIR / B5_STATE_DIR / B5_OBS_LOG_DIR
      via set_run_paths(out_dir). Inner claude subprocess gets these via
      runtime_env() merged into _call_claude_cli env. Hook subprocess writes
      go to OUR dirs, hash compares OUR dirs.
    - C0: workspace .claude/settings.json registers ONLY non-blocking TRACE
      hook subset (UPS retrieve, Stop pending-promote). The full PT chain
      including check-observation-log / deterministic-block /
      verify-compliance would otherwise return decision=block and leave
      claude with terminal_reason=stop_hook_prevented + empty answer.
    - C1: observe_correction now writes a real memory .md file in the
      per-run B5_MEMORY_DIR (not just compiled_rules in Python). Future
      rounds' memory-retrieve-inject.sh hook picks it up.
    """

    def __init__(self, profile: dict[str, Any]):
        super().__init__(profile)
        self.name = "trace_native_cc"
        self._frozen = False
        self._frozen_memory_hash: str | None = None
        # Per-run paths set by set_run_paths(); None until then.
        self._memory_dir: Path | None = None
        self._state_dir: Path | None = None
        self._obs_log_dir: Path | None = None
        # Round 4 review fix #1: monotonic counter for memory file atomic_id.
        # Previously used len(self.compiled_rules) which collides on resume:
        # observe_record(existing) replays corrections from a fresh-state
        # condition, so first replay's len()==0 produces the same atomic_id
        # as the original training run -> .md file overwritten. Use a
        # dedicated counter that survives __deepcopy__ but resets per fresh
        # condition init (matches MemoryArena _MemoryArenaTraceState pattern).
        self._write_count = 0

    def trace_mode(self) -> str:
        return "native_cc"

    def set_run_paths(self, out_dir: Path, condition_name: str | None = None) -> None:
        """Allocate per-run isolated dirs under `out_dir/trace_state/<name>/`.
        Called once per condition, before any rounds run."""
        cn = condition_name or self.name
        base = (Path(out_dir) / "trace_state" / cn).resolve()
        self._memory_dir = base / "memory"
        self._state_dir = base / "runtime"
        self._obs_log_dir = base / "obs_log"
        for d in (self._memory_dir, self._state_dir, self._obs_log_dir):
            d.mkdir(parents=True, exist_ok=True)
        # 2026-05-03 codex review fix: initialize _write_count from disk so
        # crash/orphan recovery doesn't reuse atomic_id `trace-pref-000_...`.
        # If previous run wrote N .md files, start counter at N to avoid
        # filename collision on the next observe_correction.
        self._write_count = len(list(self._memory_dir.glob("trace-pref-*.md")))
        # 2026-05-04 orphan reconciliation: opt-in via env var. When
        # enabled, scan records.jsonl (sibling to out_dir/trace_state/) and
        # quarantine .md files whose (scenario, round_id) don't match any
        # successful record with n_corrections > 0. Default OFF to avoid
        # touching existing data; future fresh runs can enable.
        if os.environ.get("EVAL_TRACE_RECONCILE_ORPHANS") == "1":
            self._reconcile_orphan_md(out_dir)

    def _reconcile_orphan_md(self, out_dir: Path) -> None:
        """Quarantine orphan trace-pref-*.md files (no matching record).

        Records.jsonl lives at out_dir.parent.parent / 'records.jsonl'
        when called via multi_runner.run_multi_experiment. Successful
        record keys = (scenario_id, round_id) pairs with n_corrections > 0.
        Orphan = .md whose filename embeds (scenario, round) NOT in the
        successful set. Move orphans to <memory_dir>/_orphan_quarantine/.
        """
        if self._memory_dir is None:
            return
        records_path = Path(out_dir) / "records.jsonl"
        if not records_path.is_file():
            return
        successful = set()
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
                    if (r.get("n_corrections") or 0) <= 0:
                        continue
                    sid = r.get("scenario_id")
                    rid = r.get("round_id")
                    if sid and rid:
                        successful.add(f"{sid}_{rid}")
                        # Also accept global_round_id form
                        successful.add(f"{sid}:{rid}".replace(":", "_"))
        except Exception:
            return
        orphan_dir = self._memory_dir / "_orphan_quarantine"
        moved = 0
        for md in list(self._memory_dir.glob("trace-pref-*.md")):
            stem = md.stem  # trace-pref-000_hil_c1_q1
            # Strip trace-pref-NNN_ prefix
            try:
                key = "_".join(stem.split("_")[2:])
            except Exception:
                continue
            if key not in successful:
                orphan_dir.mkdir(parents=True, exist_ok=True)
                md.rename(orphan_dir / md.name)
                moved += 1
        if moved:
            print(f"[trace_reconcile] {self.name}: quarantined {moved} orphan .md to {orphan_dir}", flush=True)

    def runtime_env(self) -> dict[str, str]:
        """Env vars to inject into the claude/codex subprocess so its hook
        subprocess writes to OUR per-run dirs, not the user-global ones.

        Round-7c also exports B5_RETRIEVE_DEBUG=1 so the UPS retrieve hook
        ALWAYS appends to retrieve_debug.jsonl (not just when the user
        sets it manually). That gives count_obs_log_events a reliable
        artifact to count for trace_runtime_loaded — without it the
        TRACE subset would silently produce no observable log events
        despite hooks firing.
        """
        if not self._memory_dir:
            return {}
        return {
            "B5_MEMORY_DIR": str(self._memory_dir),
            "B5_STATE_DIR": str(self._state_dir),
            "B5_OBS_LOG_DIR": str(self._obs_log_dir),
            "B5_RETRIEVE_DEBUG": "1",
        }

    def setup_workspace(self, workspace: Path) -> Path:
        """Workspace-local .claude/settings.json registering ONLY the
        non-blocking TRACE hook subset."""
        settings_path = write_workspace_settings(workspace, hooks=_PT_HOOKS_TRACE)
        heartbeat = self._write_trace_heartbeat_hook(workspace)
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
            settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        return settings_path

    def _write_trace_heartbeat_hook(self, workspace: Path) -> Path:
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
            "    with path.open('a', encoding='utf-8') as f:\n"
            "        f.write(json.dumps({'ts': time.time(), 'event': 'UserPromptSubmit'}) + '\\n')\n"
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

    def freeze(self) -> None:
        self._frozen = True
        self._frozen_memory_hash = self.memory_hash_now()

    def is_frozen(self) -> bool:
        return self._frozen

    def memory_hash_now(self) -> str:
        return memory_state_hash(self._memory_dir) if self._memory_dir else memory_state_hash()

    def memory_hash_frozen(self) -> str | None:
        return self._frozen_memory_hash

    def hook_event_count(self) -> int:
        """Real hook-fired evidence used by multi_runner to compute
        trace_runtime_loaded as an event delta. Counts lines across the
        files our TRACE hook subset actually writes (Round-7c fix:
        retrieve_debug.jsonl + pending_queue.jsonl, not just
        compliance_log.jsonl)."""
        return count_obs_log_events(self._obs_log_dir, self._state_dir)

    def enforcement_rules(self) -> list[dict[str, str]]:
        """Pure-TRACE: we DO NOT inject compiled rules into the prompt.
        Memory promotion happens through observe_correction writing real
        .md files in B5_MEMORY_DIR; the next round's UPS retrieve hook
        injects them via runtime, not via prompt. Returning [] here lets
        record fields like compiled_rule_count stay 0 so the run cannot
        be confounded with prompt-level compiled_enforcement.

        TRACE_NATIVE_CC_PREFLIGHT_RERUN minimal-fix-#4: previously this
        condition inherited CompiledEnforcementCondition.enforcement_rules,
        so compiled_rules leaked back into prompt and the observed
        improvement could be explained by prompt-level compiled rules,
        not native hook retrieval.
        """
        return []

    def observe_correction(
        self, round_id: str, correction_text: str, violation_type: str
    ) -> None:
        # Keep CompiledEnforcement parent behavior (compiled_rules) for
        # backwards compat with existing scoring/summarize code.
        super().observe_correction(round_id, correction_text, violation_type)
        if self._frozen:
            return
        if violation_type != "preference_violation":
            return
        if not self._memory_dir:
            return
        # Round-7b C1 fix: actually write a real preference-tracker memory
        # entry (markdown with frontmatter) under the per-run B5_MEMORY_DIR.
        # The next round's memory-retrieve-inject.sh hook will see it via
        # the standard _build_index() scan.
        try:
            from datetime import datetime as _dt, timezone as _tz
            ts = _dt.now(_tz.utc).isoformat()
            # Round 4 fix #1: use _write_count not len(self.compiled_rules) to
            # avoid atomic_id collision on resume (see __init__ comment).
            atomic_id = f"trace-pref-{self._write_count:03d}_{round_id}".replace(":", "_")
            requirement = _compile_requirement(correction_text, violation_type)
            md = (
                "---\n"
                f"name: TRACE corrected preference {atomic_id}\n"
                f"description: Harness-derived preference promoted from round {round_id}\n"
                "type: preference\n"
                "domain: workflow\n"
                "scope: project\n"
                "confidence: high\n"
                "priority_tier: 2\n"
                f"atomic_id: {atomic_id}\n"
                "applies_when: future rounds in this scenario context\n"
                "status: active\n"
                "lifecycle_state: confirmed\n"
                "memory_kind: procedural\n"
                f"created: {ts[:10]}\n"
                f"updated: {ts[:10]}\n"
                f"originSessionId: trace-{self.trace_mode().replace('_', '-')}-{round_id}\n"
                "---\n\n"
                "# Correction text\n\n"
                f"{correction_text.strip()}\n\n"
                "# Compiled requirement\n\n"
                f"{requirement}\n"
            )
            self._memory_dir.mkdir(parents=True, exist_ok=True)
            (self._memory_dir / f"{atomic_id}.md").write_text(md, encoding="utf-8")
            # Increment AFTER successful write (Round 4 fix #1).
            self._write_count += 1
        except OSError:
            pass

    def memory_notes(self) -> list[str]:
        """Return the persisted TRACE memory entries from disk so the next
        round's prompt builder can inject them.

        2026-05-03 fixes:
        - Critical #1 (memory_notes inherited empty list): override to read
          self._memory_dir/*.md files. Without this, TRACE was effectively
          no_memory because train-phase memory was never re-read in eval.
        - Round 4 #2 (frontmatter pollution): strip YAML frontmatter (`---\n
          ... \n---\n`) from each markdown body so the agent sees clean
          actionable text, not internal preference-tracker metadata. Matches
          MemoryArena's `_MemoryArenaTraceState.retrieve()` behavior so cross-
          dataset prompts are consistent.
        """
        if self._memory_dir is None or not self._memory_dir.is_dir():
            return []
        notes: list[str] = []
        for md_path in sorted(self._memory_dir.glob("*.md")):
            try:
                text = md_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            # Strip YAML frontmatter so agent sees clean body only.
            if text.startswith("---"):
                end = text.find("\n---", 3)
                if end < 0:
                    # Malformed frontmatter (no closing delimiter).
                    # Skip rather than leak `---` into the agent prompt.
                    continue
                text = text[end + 4:].lstrip()
            if text:
                notes.append(text)
        return notes

    def diagnostics(self) -> dict[str, Any]:
        return {
            "trace_mode": self.trace_mode(),
            "trace_skill_dir": str(_pt_skill_dir()),
            "trace_skill_present": _pt_skill_dir().is_dir(),
            "trace_frozen": self._frozen,
            "trace_frozen_memory_hash": self._frozen_memory_hash,
            "trace_memory_dir": str(self._memory_dir) if self._memory_dir else None,
            "trace_state_dir": str(self._state_dir) if self._state_dir else None,
            "trace_obs_log_dir": str(self._obs_log_dir) if self._obs_log_dir else None,
            "trace_memory_files": (
                len(list(self._memory_dir.glob("*.md")))
                if self._memory_dir and self._memory_dir.is_dir()
                else 0
            ),
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> "TraceNativeCcCondition":
        clone = TraceNativeCcCondition(copy.deepcopy(self.profile, memo))
        clone.corrections = list(self.corrections)
        clone.compiled_rules = list(self.compiled_rules)
        # Eval clone shares per-run paths with parent (we want eval reads
        # to see the train-written memory files) but it's frozen.
        clone._memory_dir = self._memory_dir
        clone._state_dir = self._state_dir
        clone._obs_log_dir = self._obs_log_dir
        clone._frozen = True
        clone._frozen_memory_hash = clone.memory_hash_now()
        # Round 4 fix #1: copy _write_count so any further (non-frozen) writes
        # in the parent advance from a consistent counter base. Eval clone
        # itself is frozen so won't write, but inheriting the counter keeps
        # invariants tidy.
        clone._write_count = self._write_count
        return clone


class TraceNativeCodexCondition(TraceNativeCcCondition):
    """TRACE-native condition for the codex_cli / codex_cli_trace runtime.

    Reuses TraceNativeCcCondition's per-run isolation + observe_correction
    machinery. Differences vs CC:

    - Hook source: ~/.codex/hooks.json (user-global). Empirically verified
      2026-05-02 with codex 0.128.0 that the hooks load even with codex
      exec --ignore-user-config --ignore-rules. The codex_cli_trace runtime
      variant drops those flags so the registration is unambiguous in
      record provenance.
    - No workspace settings file: codex framework writes AGENTS.md, not
      .claude/settings.json. setup_workspace returns None.
    - Codex has no Stop event so the "Stop hook prevents final answer"
      issue that motivated _PT_HOOKS_TRACE on CC does not apply. Codex's
      PostToolUse deterministic-block hook defaults to mode=audit_only
      (exits 0 with advisory log), keeping the agent loop unblocked.
    """

    def __init__(self, profile: dict[str, Any]):
        super().__init__(profile)
        self.name = "trace_native_codex"

    def trace_mode(self) -> str:
        return "native_codex"

    def setup_workspace(self, workspace: Path) -> Path | None:
        # Codex loads hooks from ~/.codex/hooks.json globally; no
        # workspace-local settings file is honored. Return None so
        # multi_runner records workspace_settings_path correctly.
        return None

    def diagnostics(self) -> dict[str, Any]:
        d = super().diagnostics()
        codex_hooks = Path.home() / ".codex" / "hooks.json"
        d["trace_codex_hooks_path"] = str(codex_hooks)
        d["trace_codex_hooks_present"] = codex_hooks.is_file()
        return d

    def __deepcopy__(self, memo: dict[int, Any]) -> "TraceNativeCodexCondition":
        clone = TraceNativeCodexCondition(copy.deepcopy(self.profile, memo))
        clone.corrections = list(self.corrections)
        clone.compiled_rules = list(self.compiled_rules)
        clone._memory_dir = self._memory_dir
        clone._state_dir = self._state_dir
        clone._obs_log_dir = self._obs_log_dir
        clone._frozen = True
        clone._frozen_memory_hash = clone.memory_hash_now()
        clone._write_count = self._write_count
        return clone


def _compile_requirement(correction_text: str, violation_type: str) -> str:
    base = correction_text.strip()
    if violation_type != "preference_violation":
        return "Ignore non-preference failures for durable enforcement compilation."
    rule_groups = set(re.findall(r"\b(P[1-5]|O[1-4])(?:\b|_)", base))
    requirements: list[str] = []
    if "P1" in rule_groups:
        requirements.append(
            "Apply P1 formatting preferences before finalizing artifacts: use ISO-style "
            "timestamps and thousands separators for numbers of 10,000 or more."
        )
    if "P2" in rule_groups:
        requirements.append(
            "Apply P2 artifact naming preferences before finalizing files: use "
            "YYYY-MM-DD_<topic>_v<N>.<ext> filenames and keep versioned copies when "
            "revising existing artifacts."
        )
    if "P3" in rule_groups:
        requirements.append(
            "Apply P3 report-structure preferences before finalizing user-facing reports: "
            "include Summary, Details, and Action Items sections."
        )
    if "P4" in rule_groups:
        requirements.append(
            "Apply P4 code-quality preferences before finalizing code or tests: concise "
            "docstrings, type hints, and logging instead of print statements."
        )
    if "P5" in rule_groups:
        requirements.append(
            "Apply P5 source-grounding preferences before finalizing reports or JSON: "
            "cite sources, tag unverified claims, and include source fields where required."
        )
    if "O1" in rule_groups:
        requirements.append("Return one well-formed JSON object with the required keys.")
    if "O2" in rule_groups:
        requirements.append(
            "Structure rationale so evidence and conclusion are separable by using explicit "
            "'Evidence:' and 'Decision:' labels."
        )
    if "O3" in rule_groups:
        requirements.append(
            "State which staged updates were considered in the decision log using an explicit "
            "'updates:' status, such as 'updates:none' or 'updates:upd1,upd2'."
        )
    if "O4" in rule_groups:
        requirements.append("Represent unresolved items as a list, even when empty.")
    if requirements:
        return " ".join(requirements)
    return f"Apply the durable user/project correction: {base}"
