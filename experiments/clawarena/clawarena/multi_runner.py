"""Run ClawArena the experiments across multiple scenarios with frozen ID/OOD eval."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .clawarena_loader import load_scenario
from .conditions import build_condition
from .exec_checks import run_exec_checks
from .preference_overlay import evaluate_overlay, overlay_correction_text, overlay_rule_prompt
from .profiles import get_profile
from .prompts import build_enforcement_retry_prompt, build_round_prompt
from .runtimes import call_runtime
from .schema import EXEC_CHECK, Round, Scenario
from .scoring import correction_has_answer_leak, pref_rule_ids, safe_pref_feedback, score_response
from .workspace import (
    apply_updates,
    current_applied_updates,
    dataset_hash,
    fsync_parent,
    harness_hash,
    prepare_workspace,
)

# This public release ships only TRACE (our method) and the No-Memory control.
DEFAULT_CONDITIONS = ["no_memory", "trace_native_cc"]


def run_multi_experiment(
    *,
    clawarena_root: Path,
    train_scenarios: list[str],
    id_test_scenarios: list[str],
    ood_test_scenarios: list[str],
    profile_id: str,
    conditions: list[str],
    runtime: str,
    model: str,
    out_dir: Path,
    train_round_limit: int | None,
    eval_round_limit: int | None,
    enforcement_retries: int,
    timeout_s: int,
    runtime_retries: int,
    max_budget_usd: float,
    include_profile_preferences: bool = False,
    seed_corrections: list[dict[str, str]] | None = None,
    skip_train: bool = False,
    skip_id_test: bool = False,
    skip_ood_test: bool = False,
    resume: bool = False,
    allow_harness_hash_mismatch: bool = False,
    allow_resume_config_shrink: bool = False,
) -> list[dict[str, Any]]:
    profile = get_profile(profile_id)
    if not include_profile_preferences:
        profile = {**profile, "durable_preferences": []}
    scenario_cache = {
        scenario_id: load_scenario(clawarena_root, scenario_id)
        for scenario_id in sorted(set(train_scenarios + id_test_scenarios + ood_test_scenarios))
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    run_config = {
        "runner": "multi_runner",
        "harness_version": 1,
        "train_scenarios": train_scenarios,
        "id_test_scenarios": id_test_scenarios,
        "ood_test_scenarios": ood_test_scenarios,
        "profile_id": profile_id,
        "conditions": conditions,
        "runtime": runtime,
        "model": model,
        "train_round_limit": train_round_limit,
        "eval_round_limit": eval_round_limit,
        "enforcement_retries": enforcement_retries,
        "timeout_s": timeout_s,
        "runtime_retries": runtime_retries,
        "max_budget_usd": max_budget_usd,
        "include_profile_preferences": include_profile_preferences,
        "seed_corrections": seed_corrections or [],
        "skip_train": skip_train,
        "dataset_hashes": {
            scenario_id: dataset_hash(clawarena_root, runtime, scenario_id)
            for scenario_id in scenario_cache
        },
        "harness_hash": harness_hash(Path(__file__).parent),
        "profile_hash": _stable_hash(profile),
        "overlay_rules": overlay_rule_prompt(),
    }
    current_run_hash = _stable_hash(run_config)
    run_hash = current_run_hash
    config_path = out_dir / "run_config.json"
    if resume and config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        existing_run_hash = existing_config.get("run_hash")
        if existing_run_hash and existing_run_hash != current_run_hash:
            existing_semantic = {
                k: existing_config.get(k)
                for k in run_config
                if k != "harness_hash"
            }
            if "seed_corrections" not in existing_config:
                existing_semantic["seed_corrections"] = []
            if "skip_train" not in existing_config:
                existing_semantic["skip_train"] = False
            current_semantic = {
                k: run_config.get(k)
                for k in run_config
                if k != "harness_hash"
            }
            # 2026-05-03: allow_resume_config_shrink permits scope-shrink
            # changes (smaller train_round_limit / eval_round_limit) on resume
            # without forcing fresh re-run. Use case: user decides mid-run to
            # cap rounds at 3 instead of 5; existing q4/q5 records stay (just
            # ignored by paper analysis), new scenarios produce only q1-q3.
            # 2026-05-06: also ignore skip_train / skip_id_test / skip_ood_test
            # in hash check — these flags toggle which split phases run, not the
            # underlying experiment definition. Allows resume after switching
            # from full pipeline to OOD-only continuation.
            shrink_only = (
                allow_resume_config_shrink
                and {k: existing_semantic.get(k) for k in existing_semantic if k not in ("train_round_limit", "eval_round_limit", "skip_train", "skip_id_test", "skip_ood_test")}
                == {k: current_semantic.get(k) for k in current_semantic if k not in ("train_round_limit", "eval_round_limit", "skip_train", "skip_id_test", "skip_ood_test")}
                and (current_semantic.get("train_round_limit") or 99) <= (existing_semantic.get("train_round_limit") or 99)
                and (current_semantic.get("eval_round_limit") or 99) <= (existing_semantic.get("eval_round_limit") or 99)
            )
            # also: allow_harness_hash_mismatch should ignore skip_* flags (resume after switching skip_* args)
            harness_match_with_skip_excluded = (
                allow_harness_hash_mismatch
                and {k: existing_semantic.get(k) for k in existing_semantic if k not in ("skip_train", "skip_id_test", "skip_ood_test")}
                == {k: current_semantic.get(k) for k in current_semantic if k not in ("skip_train", "skip_id_test", "skip_ood_test")}
            )
            if not (allow_harness_hash_mismatch and existing_semantic == current_semantic) and not shrink_only and not harness_match_with_skip_excluded:
                raise RuntimeError(
                    "cannot resume multi_runner: run_config mismatch "
                    f"existing={existing_run_hash} current={current_run_hash}"
                )
            run_hash = str(existing_run_hash)
            prior_hashes = existing_config.get("resume_harness_hashes") or []
            if run_config["harness_hash"] not in prior_hashes:
                prior_hashes.append(run_config["harness_hash"])
            merged_config = {
                **existing_config,
                "resume_harness_hashes": prior_hashes,
                "last_resume_harness_hash": run_config["harness_hash"],
            }
            _write_json_atomic(config_path, merged_config)
        else:
            _write_json_atomic(config_path, {"run_hash": run_hash, **run_config})
    else:
        _write_json_atomic(config_path, {"run_hash": run_hash, **run_config})

    existing_records = _load_existing_records(records_path) if resume else []
    if existing_records:
        mismatched = [r for r in existing_records if r.get("run_hash") != run_hash]
        if mismatched:
            if not allow_resume_config_shrink:
                raise RuntimeError(
                    "cannot resume multi_runner: existing records have a different run_hash"
                )
            # 2026-05-03: under shrink-mode, rewrite each existing record's
            # run_hash to the current one so resume scan accepts them. Records
            # are still valid (same scenarios + conditions, just smaller round
            # window now). Save original under `original_run_hash` for audit.
            for r in existing_records:
                if r.get("run_hash") != run_hash:
                    r["original_run_hash"] = r.get("run_hash")
                    r["run_hash"] = run_hash
        existing_records = _dedupe_latest_records(existing_records)
        _rewrite_jsonl(records_path, existing_records)
    else:
        records_path.write_text("", encoding="utf-8")
    existing_by_key = _successful_records_by_key(existing_records)

    records: list[dict[str, Any]] = list(existing_records)
    for condition_name in conditions:
        condition = build_condition(condition_name, profile)
        # Round-7b C2 fix (TRACE_NATIVE_CC_PATCH_REVIEW 2026-05-02): TRACE-
        # native conditions need per-run isolated memory/state/obs_log dirs
        # so multiple parallel runs don't trample each other and so the
        # condition's memory_state_hash points at the EXACT directory the
        # inner claude subprocess writes to via its B5_MEMORY_DIR env.
        if hasattr(condition, "set_run_paths"):
            try:
                condition.set_run_paths(out_dir, condition_name)
            except Exception as exc:
                raise RuntimeError(
                    f"condition {condition_name} set_run_paths failed: {exc}"
                ) from exc
        for item in seed_corrections or []:
            correction_text = item.get("correction_text") or item.get("text") or ""
            if not correction_text.strip():
                continue
            condition.observe_correction(
                round_id=item.get("round_id") or item.get("global_round_id") or "seeded",
                correction_text=correction_text,
                violation_type=item.get("violation_type") or "preference_violation",
            )
        learning_step = 0
        split_specs = []
        if not skip_train:
            split_specs.append(("train", train_scenarios, False, train_round_limit))
        if not skip_id_test:
            split_specs.append(("id_test", id_test_scenarios, True, eval_round_limit))
        if not skip_ood_test:
            split_specs.append(("ood_test", ood_test_scenarios, True, eval_round_limit))
        for split_name, scenario_ids, frozen, round_limit in split_specs:
            # M4 freeze: TRACE-native conditions snapshot memory state hash
            # at the train→eval boundary so frozen splits can later prove no
            # writes occurred. The deepcopy clone in eval splits also gets
            # _frozen=True via __deepcopy__.
            if frozen and hasattr(condition, "freeze"):
                try:
                    condition.freeze()
                except Exception:
                    pass
            eval_condition = copy.deepcopy(condition) if frozen else condition
            for scenario_id in scenario_ids:
                scenario = scenario_cache[scenario_id]
                rounds = _select_rounds_for_split(
                    scenario=scenario,
                    scenario_id=scenario_id,
                    split_name=split_name,
                    train_scenarios=train_scenarios,
                    train_round_limit=train_round_limit,
                    round_limit=round_limit,
                )
                workspace = prepare_workspace(
                    clawarena_root=clawarena_root,
                    runtime=runtime,
                    model=model,
                    scenario_id=scenario_id,
                    condition=f"{condition_name}_{split_name}",
                    profile_id=profile_id,
                    out_dir=out_dir,
                    resume=False,
                )
                last_skipped_record: dict[str, Any] | None = None
                restored_from_snapshot = False
                for round_obj in rounds:
                    if not frozen:
                        learning_step += 1
                    planned_key = (
                        condition_name,
                        split_name,
                        scenario.scenario_id,
                        round_obj.round_id,
                    )
                    existing = existing_by_key.get(planned_key)
                    if existing is not None:
                        if (
                            not frozen
                            and existing.get("collected_from_this_round")
                            and existing.get("process_ok")
                        ):
                            condition.observe_record(existing)
                        last_skipped_record = existing
                        continue
                    if last_skipped_record is not None and not restored_from_snapshot:
                        _restore_workspace_snapshot_if_available(
                            out_dir=out_dir,
                            workspace=workspace,
                            record=last_skipped_record,
                        )
                        restored_from_snapshot = True
                    record = _run_one_round(
                        clawarena_root=clawarena_root,
                        scenario=scenario,
                        round_obj=round_obj,
                        split_name=split_name,
                        frozen=frozen,
                        condition_name=condition_name,
                        condition=eval_condition,
                        profile_id=profile_id,
                        runtime=runtime,
                        model=model,
                        out_dir=out_dir,
                        workspace=workspace,
                        learning_step=learning_step if not frozen else None,
                        enforcement_retries=enforcement_retries,
                        timeout_s=timeout_s,
                        runtime_retries=runtime_retries,
                        max_budget_usd=max_budget_usd,
                        run_hash=run_hash,
                    )
                    correction = record.get("correction_text")
                    collected = bool(correction and not frozen and record.get("process_ok"))
                    if collected:
                        before_memory_n = len(condition.memory_notes())
                        before_rule_n = len(condition.enforcement_rules())
                        before_correction_n = len(getattr(condition, "corrections", []))
                        condition.observe_correction(
                            round_id=str(record["global_round_id"]),
                            correction_text=str(correction),
                            violation_type=str(record.get("violation_type") or "preference_violation"),
                        )
                        record["memory_notes_after"] = condition.memory_notes()
                        record["compiled_rules_after"] = condition.enforcement_rules()
                        record["collected_from_this_round"] = True
                        record["learned_from_this_round"] = (
                            len(record["memory_notes_after"]) > before_memory_n
                            or len(record["compiled_rules_after"]) > before_rule_n
                            or len(getattr(condition, "corrections", [])) > before_correction_n
                        )
                        if hasattr(condition, "memory_hash_now"):
                            try:
                                memory_hash_after = condition.memory_hash_now()
                            except Exception:
                                memory_hash_after = None
                            record["memory_state_hash_after"] = memory_hash_after
                            before_hash = record.get("memory_state_hash_before")
                            if before_hash is not None and memory_hash_after is not None:
                                record["memory_write_count_this_round"] = (
                                    1 if before_hash != memory_hash_after else 0
                                )
                        record["condition_diagnostics"] = condition.diagnostics()
                    else:
                        record["collected_from_this_round"] = False
                        record["learned_from_this_round"] = False
                        record["memory_notes_after"] = condition.memory_notes()
                        record["compiled_rules_after"] = condition.enforcement_rules()
                    # Always capture condition diagnostics (train AND test). Gate
                    # 1 I4 fix — previously only set inside `if collected`, so
                    # eval-phase records lost selfcheck_judge_call_count /
                    # selfcheck_frozen / other condition-specific introspection
                    # needed for Gate 2 post-hoc audit. Pre-existing limitation
                    # surfaced by mem0_self_check work 2026-05-15.
                    if "condition_diagnostics" not in record:
                        try:
                            record["condition_diagnostics"] = condition.diagnostics()
                        except Exception as diag_exc:  # noqa: BLE001
                            record["condition_diagnostics"] = {
                                "error": f"{type(diag_exc).__name__}: {str(diag_exc)[:200]}"
                            }
                    # NOTE: selfcheck_audit_this_round is set INSIDE
                    # _run_one_round (line ~730) where the anchor variable
                    # is in scope. Previous attempt to compute the slice
                    # here used locals().get() which always returned None
                    # because _audit_len_before_this_round lives in
                    # _run_one_round's scope, not this scope. Re-review N2
                    # post-fix verification 2026-05-15.
                    records.append(record)
                    _write_round_record(out_dir, record)
                    _append_jsonl(records_path, record)
                    _snapshot_workspace(out_dir=out_dir, workspace=workspace, record=record)
    records = _dedupe_latest_records(records)
    _rewrite_jsonl(records_path, records)
    _write_json_atomic(out_dir / "summary_records.json", {"records": records})
    return records


def _run_one_round(
    *,
    clawarena_root: Path,
    scenario: Scenario,
    round_obj: Round,
    split_name: str,
    frozen: bool,
    condition_name: str,
    condition: Any,
    profile_id: str,
    runtime: str,
    model: str,
    out_dir: Path,
    workspace: Path,
    learning_step: int | None,
    enforcement_retries: int,
    timeout_s: int,
    runtime_retries: int,
    max_budget_usd: float,
    run_hash: str,
) -> dict[str, Any]:
    started = time.time()
    update_events = apply_updates(
        clawarena_root=clawarena_root,
        runtime=runtime,
        scenario_id=scenario.scenario_id,
        workspace=workspace,
        update_ids=round_obj.update_ids,
    )
    # M2: TRACE-native-CC condition writes per-workspace .claude/settings.json
    # registering tellonce hooks. claude CLI then loads them via
    # --settings flag (see runtimes._call_claude_cli).
    if hasattr(condition, "setup_workspace"):
        try:
            condition.setup_workspace(workspace)
        except Exception as exc:
            raise RuntimeError(
                f"condition {condition_name} setup_workspace failed: {exc}"
            ) from exc

    # M4: snapshot memory hash before this round (for eval-frozen comparison).
    memory_hash_before = None
    if hasattr(condition, "memory_hash_now"):
        try:
            memory_hash_before = condition.memory_hash_now()
        except Exception:
            memory_hash_before = None

    condition.prepare_round(
        scenario_id=scenario.scenario_id,
        round_id=round_obj.round_id,
        question=round_obj.question,
        update_events=update_events,
    )
    memory_notes_before = condition.memory_notes()
    compiled_rules_before = condition.enforcement_rules()
    session_dir = (
        out_dir
        / runtime
        / _safe_segment(model)
        / condition_name
        / split_name
        / scenario.scenario_id
        / round_obj.round_id
    )
    prompt = build_round_prompt(
        scenario=scenario,
        round_obj=round_obj,
        condition=condition,
        workspace=workspace,
        update_events=update_events,
    )
    # Round-7b C2 fix: pass per-run B5_MEMORY_DIR/B5_STATE_DIR/B5_OBS_LOG_DIR
    # into the inner claude subprocess so its hook subprocess writes there.
    extra_env = (
        condition.runtime_env() if hasattr(condition, "runtime_env") else {}
    ) or {}

    # Round-7b C3 fix: hook event count BEFORE the call so we can compare
    # AFTER and only mark trace_runtime_loaded=true when hooks actually
    # fired (not just because settings.json exists on disk).
    hook_events_before = (
        condition.hook_event_count() if hasattr(condition, "hook_event_count") else 0
    )

    response = call_runtime(
        runtime,
        prompt,
        model=model,
        session_dir=session_dir,
        working_dir=workspace,
        allow_workspace_write=round_obj.round_type == EXEC_CHECK,
        timeout_s=timeout_s,
        runtime_retries=runtime_retries,
        max_budget_usd=max_budget_usd,
        round_obj=round_obj,
        extra_env=extra_env,
    )

    hook_events_after = (
        condition.hook_event_count() if hasattr(condition, "hook_event_count") else 0
    )
    exec_check = (
        run_exec_checks(
            clawarena_root=clawarena_root,
            scenario_id=scenario.scenario_id,
            workspace=workspace,
            round_obj=round_obj,
        )
        if round_obj.round_type == EXEC_CHECK and response.process_ok
        else None
    )
    initial_exec_check = exec_check
    enforcement_attempts: list[dict[str, Any]] = []
    if (
        condition_name == "compiled_enforcement"
        and round_obj.round_type == EXEC_CHECK
        and exec_check
        and exec_check.get("pref_pass") is False
        and response.process_ok
    ):
        for attempt_idx in range(enforcement_retries):
            retry_response = call_runtime(
                runtime,
                build_enforcement_retry_prompt(
                    round_obj=round_obj,
                    condition=condition,
                    workspace=workspace,
                    verifier_feedback=safe_pref_feedback(round_obj),
                ),
                model=model,
                session_dir=session_dir / f"enforcement_retry_{attempt_idx + 1}",
                working_dir=workspace,
                allow_workspace_write=True,
                timeout_s=timeout_s,
                runtime_retries=runtime_retries,
                max_budget_usd=max_budget_usd,
                round_obj=round_obj,
            )
            retry_exec_check = (
                run_exec_checks(
                    clawarena_root=clawarena_root,
                    scenario_id=scenario.scenario_id,
                    workspace=workspace,
                    round_obj=round_obj,
                )
                if retry_response.process_ok
                else None
            )
            enforcement_attempts.append(
                {
                    "attempt": attempt_idx + 1,
                    "runtime_error": retry_response.runtime_error,
                    "process_ok": retry_response.process_ok,
                    "parse_error": retry_response.parse_error,
                    "elapsed_s": retry_response.elapsed_s,
                    "exec_check": retry_exec_check,
                }
            )
            if retry_exec_check is not None:
                response = retry_response
                exec_check = retry_exec_check
            if retry_exec_check and retry_exec_check.get("pref_pass") is not False:
                break
    score = score_response(round_obj, response.parsed, condition, exec_check=exec_check)
    overlay = evaluate_overlay(
        round_obj=round_obj,
        parsed=response.parsed,
        workspace=workspace,
        update_events=update_events,
    )
    if overlay["overlay_pref_pass"] is False:
        overlay_text = overlay_correction_text(round_obj.round_id, overlay["overlay_pref_failures"])
        if not score.get("correction_text"):
            score["correction_text"] = overlay_text
            score["n_corrections"] = 1
            score["violation_type"] = "preference_violation"
        score["overlay_correction_text"] = overlay_text
    else:
        score["overlay_correction_text"] = None

    if correction_has_answer_leak(round_obj, score.get("correction_text")):
        score["correction_leak_blocked"] = True
        score["correction_text"] = None
        score["n_corrections"] = 0
    else:
        score["correction_leak_blocked"] = False

    # M1 + M4 trace fields. We ALWAYS emit trace_mode (even for non-TRACE
    # conditions, where it's "n/a") so summarize/dashboard code can filter
    # records uniformly.
    trace_mode = (
        condition.trace_mode() if hasattr(condition, "trace_mode") else "n/a"
    )
    workspace_settings_path = workspace / ".claude" / "settings.json"
    # Round-7b C3 + 2026-05-04 codex isolation fix: trace_runtime_loaded
    # is an EVENT-DELTA assertion, not a settings-file check. Registration
    # check by trace_mode:
    #   native_cc + claude_cli: workspace .claude/settings.json must exist
    #                           (claude --setting-sources project,local).
    #   native_codex + codex_cli_trace: hooks.json lives in per-call
    #                           isolated CODEX_HOME (created by
    #                           runtimes._prepare_trace_codex_home, torn
    #                           down after call), so post-hoc file check
    #                           is impossible. Trust hook_events_after >
    #                           hook_events_before delta below — that's
    #                           the real authority. (We always create the
    #                           isolated CODEX_HOME with hooks.json before
    #                           the call, so registration is by construction.)
    if trace_mode == "native_cc" and runtime == "claude_cli":
        runtime_registered = workspace_settings_path.is_file()
    elif trace_mode == "native_codex" and runtime in ("codex_cli", "codex_cli_trace"):
        # Isolated CODEX_HOME is per-call. Registration is by-construction,
        # delta gate (hook_events_after > before) is the real authority.
        runtime_registered = True
    else:
        runtime_registered = False
    trace_runtime_loaded = (
        runtime_registered and hook_events_after > hook_events_before
    )
    memory_hash_after = None
    if hasattr(condition, "memory_hash_now"):
        try:
            memory_hash_after = condition.memory_hash_now()
        except Exception:
            memory_hash_after = None
    memory_write_count_this_round = 0
    if memory_hash_before is not None and memory_hash_after is not None:
        if memory_hash_before != memory_hash_after:
            # Hash changed -> something wrote to memory dir during this round.
            # We can't easily count exact writes without diffing dirs; report 1+
            # so the assertion below catches it.
            memory_write_count_this_round = 1

    return {
        "timestamp": time.time(),
        "run_hash": run_hash,
        "scenario_id": scenario.scenario_id,
        "profile_id": profile_id,
        "condition": condition_name,
        "runtime": runtime,
        "model": model,
        "split": split_name,
        "frozen_preferences": frozen,
        # ---- TRACE fields (gap-review M1 + M4, 2026-05-02) ----
        "trace_mode": trace_mode,
        "trace_runtime_loaded": trace_runtime_loaded,
        "trace_frozen": frozen,
        "memory_state_hash_before": memory_hash_before,
        "memory_state_hash_after": memory_hash_after,
        "memory_write_count_this_round": memory_write_count_this_round,
        "workspace_settings_path": str(workspace_settings_path) if workspace_settings_path.is_file() else None,
        "hook_events_before": hook_events_before,
        "hook_events_after": hook_events_after,
        "hook_events_delta": hook_events_after - hook_events_before,
        "learning_step": learning_step,
        "round_index": round_obj.round_index,
        "round_id": round_obj.round_id,
        "global_round_id": f"{scenario.scenario_id}:{round_obj.round_id}",
        "round_type": round_obj.round_type,
        "pref_rule_ids": pref_rule_ids(round_obj),
        "native_pref_rule_ids": pref_rule_ids(round_obj),
        "overlay_pref_rule_ids": overlay["overlay_pref_rule_ids"],
        "collected_from_this_round": False,
        "learned_from_this_round": False,
        "update_ids": round_obj.update_ids,
        "applied_updates": current_applied_updates(workspace),
        "update_events": update_events,
        "workspace": str(workspace),
        "memory_note_count": len(memory_notes_before),
        "compiled_rule_count": len(compiled_rules_before),
        "condition_diagnostics": condition.diagnostics(),
        "parsed_response": response.parsed,
        "raw_response_excerpt": response.raw_text[:20000],
        "runtime_error": response.runtime_error,
        "process_ok": response.process_ok,
        "parse_error": response.parse_error,
        "runtime_provenance": response.runtime_provenance,
        "runtime_stdout_excerpt": response.stdout[:20000],
        "runtime_stderr_excerpt": response.stderr[:20000],
        "elapsed_s": time.time() - started,
        "initial_exec_check": initial_exec_check,
        "enforcement_attempts": enforcement_attempts,
        **score,
        **overlay,
        "compiled_rules_before": compiled_rules_before,
        "memory_notes_before": memory_notes_before,
        # Placeholders identical to *_before. The caller (run_multi_experiment)
        # ALWAYS overwrites these post-collection (see lines ~284-308) with
        # condition.memory_notes()/enforcement_rules() AFTER observe_correction.
        # If you ever change the caller, update this placeholder semantics.
        "compiled_rules_after": compiled_rules_before,
        "memory_notes_after": memory_notes_before,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    with tmp.open("r", encoding="utf-8") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_parent(path)


def _successful_records_by_key(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str, str, str], dict[str, Any]]:
    return {
        _record_key(record): record
        for record in records
        if _record_is_resume_success(record)
    }


def _record_is_resume_success(record: dict[str, Any]) -> bool:
    if record.get("runtime_error"):
        return False
    if (
        record.get("runtime") == "codex_cli"
        and record.get("trace_mode") in {None, "n/a"}
        and not _record_has_codex_hook_isolation(record)
    ):
        return False
    trace_mode = record.get("trace_mode")
    if trace_mode and trace_mode != "n/a":
        if not record.get("trace_runtime_loaded"):
            return False
        if record.get("trace_frozen") and record.get("memory_write_count_this_round", 0) > 0:
            return False
    return True


def _record_has_codex_hook_isolation(record: dict[str, Any]) -> bool:
    provenance = record.get("runtime_provenance")
    if not isinstance(provenance, dict):
        return False
    return (
        provenance.get("codex_home_isolated") is True
        and provenance.get("codex_hook_isolation_version") == 2
    )


def _dedupe_latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
    key_order: list[tuple[str, str, str, str, str, str, str]] = []
    for record in records:
        record_key = _record_key(record)
        if record_key not in latest_by_key:
            key_order.append(record_key)
        latest_by_key[record_key] = record
    return [latest_by_key[key] for key in key_order]


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(record.get("profile_id")),
        str(record.get("model")),
        str(record.get("runtime")),
        str(record.get("condition")),
        str(record.get("split")),
        str(record.get("scenario_id")),
        str(record.get("round_id")),
    )


def _write_round_record(out_dir: Path, record: dict[str, Any]) -> None:
    rel = (
        Path("round_records")
        / str(record["runtime"])
        / _safe_segment(str(record["model"]))
        / str(record["condition"])
        / str(record["profile_id"])
        / str(record["split"])
        / str(record["scenario_id"])
        / f"{int(record['round_index']):04d}_{record['round_id']}.json"
    )
    _write_json_atomic(out_dir / rel, record)


def _snapshot_workspace(*, out_dir: Path, workspace: Path, record: dict[str, Any]) -> Path:
    dst = _workspace_snapshot_path(out_dir, record)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    fsync_parent(dst)
    return dst


def _workspace_snapshot_path(out_dir: Path, record: dict[str, Any]) -> Path:
    return (
        out_dir
        / "workspace_snapshots"
        / str(record["runtime"])
        / _safe_segment(str(record["model"]))
        / str(record["condition"])
        / str(record["profile_id"])
        / str(record["split"])
        / str(record["scenario_id"])
        / f"{int(record['round_index']):04d}_{record['round_id']}"
    )


def _restore_workspace_snapshot_if_available(
    *, out_dir: Path, workspace: Path, record: dict[str, Any]
) -> None:
    src = _workspace_snapshot_path(out_dir, record)
    if not src.is_dir():
        return
    tmp = workspace.with_name(workspace.name + ".resume_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if workspace.exists():
        shutil.rmtree(workspace)
    os.replace(tmp, workspace)
    fsync_parent(workspace)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with tmp.open("r", encoding="utf-8") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_parent(path)


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _safe_segment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:120] or "model"


def _parse_scenarios(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    if not out:
        raise ValueError("scenario list cannot be empty")
    return out


def _select_rounds_for_split(
    *,
    scenario: Scenario,
    scenario_id: str,
    split_name: str,
    train_scenarios: list[str],
    train_round_limit: int | None,
    round_limit: int | None,
) -> list[Round]:
    if split_name == "id_test" and scenario_id in set(train_scenarios):
        start = train_round_limit or 0
    else:
        start = 0
    selected = scenario.rounds[start:]
    return selected[:round_limit] if round_limit else selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clawarena-root", type=Path, default=Path("/tmp/eval_candidate_repos/ClawArena"))
    parser.add_argument("--train-scenarios", nargs="+", required=True)
    parser.add_argument("--id-test-scenarios", nargs="+", required=True)
    parser.add_argument("--ood-test-scenarios", nargs="+", required=True)
    parser.add_argument("--profile", default="p1")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument(
        "--runtime",
        choices=["oracle_dry_run", "claude_cli", "codex_cli", "codex_cli_trace"],
        default="oracle_dry_run",
    )
    parser.add_argument("--model", default="oracle")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-round-limit", type=int, default=None)
    parser.add_argument("--eval-round-limit", type=int, default=None)
    parser.add_argument("--enforcement-retries", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--runtime-retries", type=int, default=1)
    parser.add_argument("--max-budget-usd", type=float, default=0.5)
    parser.add_argument("--include-profile-preferences", action="store_true")
    parser.add_argument("--seed-corrections-json", type=Path, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-id-test", action="store_true")
    parser.add_argument("--skip-ood-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-harness-hash-mismatch", action="store_true")
    parser.add_argument(
        "--allow-resume-config-shrink",
        action="store_true",
        help="Permit resume when train_round_limit/eval_round_limit shrunk (existing q4/q5 records preserved but ignored by paper analysis; rewrites records' run_hash to current).",
    )
    args = parser.parse_args()
    seed_corrections = None
    if args.seed_corrections_json:
        seed_corrections = json.loads(args.seed_corrections_json.read_text(encoding="utf-8"))
        if not isinstance(seed_corrections, list):
            raise ValueError("--seed-corrections-json must contain a JSON list")
    records = run_multi_experiment(
        clawarena_root=args.clawarena_root,
        train_scenarios=_parse_scenarios(args.train_scenarios),
        id_test_scenarios=_parse_scenarios(args.id_test_scenarios),
        ood_test_scenarios=_parse_scenarios(args.ood_test_scenarios),
        profile_id=args.profile,
        conditions=args.conditions,
        runtime=args.runtime,
        model=args.model,
        out_dir=args.out_dir,
        train_round_limit=args.train_round_limit,
        eval_round_limit=args.eval_round_limit,
        enforcement_retries=args.enforcement_retries,
        timeout_s=args.timeout_s,
        runtime_retries=args.runtime_retries,
        max_budget_usd=args.max_budget_usd,
        include_profile_preferences=args.include_profile_preferences,
        seed_corrections=seed_corrections,
        skip_train=args.skip_train,
        skip_id_test=args.skip_id_test,
        skip_ood_test=args.skip_ood_test,
        resume=args.resume,
        allow_harness_hash_mismatch=args.allow_harness_hash_mismatch,
        allow_resume_config_shrink=args.allow_resume_config_shrink,
    )
    print(f"wrote {len(records)} records to {args.out_dir}")


if __name__ == "__main__":
    main()
