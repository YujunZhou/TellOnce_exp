"""Summarize multi-scenario ClawArena the experiments records."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _is_valid_trace_record(rec: dict[str, Any]) -> tuple[bool, str | None]:
    """Gap-review M6 (2026-05-02): a record claiming TRACE-native must
    actually have the runtime loaded, and frozen splits must show no
    memory writes. Return (valid, reason_if_invalid).

    Non-TRACE records (trace_mode == 'n/a' or missing) always pass — we
    only gate the trace_native_* conditions.
    """
    trace_mode = rec.get("trace_mode")
    if not trace_mode or trace_mode == "n/a":
        return True, None
    if not rec.get("trace_runtime_loaded"):
        return False, "trace_runtime_loaded=false (workspace settings.json missing or runtime mismatch)"
    if rec.get("trace_frozen") and rec.get("memory_write_count_this_round", 0) > 0:
        return False, "frozen split wrote memory (memory_write_count_this_round>0)"
    return True, None


def filter_trace_invalid(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into (valid, invalid). Invalid records carry a
    `_trace_invalid_reason` annotation."""
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for rec in records:
        ok, reason = _is_valid_trace_record(rec)
        if ok:
            valid.append(rec)
        else:
            annotated = dict(rec)
            annotated["_trace_invalid_reason"] = reason
            invalid.append(annotated)
    return valid, invalid


def summarize_records(records: list[dict[str, Any]], *, curve_window: int = 10) -> dict[str, Any]:
    records = _dedupe_latest_records(records)
    valid, invalid = filter_trace_invalid(records)
    summary = _summary_rows(valid)
    return {
        "summary": summary,
        "summary_split_pivot": _summary_split_pivot(summary),
        "training_curve": _training_curve(valid, window=curve_window),
        "trace_invalid_count": len(invalid),
        "trace_invalid_reasons": [
            {
                "condition": r.get("condition"),
                "split": r.get("split"),
                "scenario_id": r.get("scenario_id"),
                "round_id": r.get("round_id"),
                "reason": r.get("_trace_invalid_reason"),
            }
            for r in invalid
        ],
    }


def _summary_split_pivot(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pivot per-(runtime, model, profile, condition, split) summary rows into
    one row per (runtime, model, profile, condition) with train / id_test /
    ood_test elapsed_s and other split-aware metrics side-by-side.

    Recording both train and test timings here makes the train-vs-frozen-test
    cost separation directly readable from the summary JSON, instead of
    requiring downstream split-grouping.
    """
    pivot: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in summary_rows:
        key = (
            str(row.get("runtime")),
            str(row.get("model")),
            str(row.get("profile_id")),
            str(row.get("condition")),
        )
        split = str(row.get("split"))
        slot = pivot.setdefault(
            key,
            {
                "runtime": row.get("runtime"),
                "model": row.get("model"),
                "profile_id": row.get("profile_id"),
                "condition": row.get("condition"),
            },
        )
        prefix = f"{split}_"
        for k, v in row.items():
            if k in ("runtime", "model", "profile_id", "condition", "split"):
                continue
            slot[prefix + k] = v
    return sorted(pivot.values(), key=lambda r: (r["runtime"], r["model"], r["profile_id"], r["condition"]))


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (
            str(rec.get("runtime")),
            str(rec.get("model")),
            str(rec.get("profile_id")),
            str(rec.get("condition")),
            str(rec.get("split")),
        )
        groups.setdefault(key, []).append(rec)
    rows = []
    for (runtime, model, profile_id, condition, split), rs in sorted(groups.items()):
        scored = [r for r in rs if r.get("scored")]
        task_passed = [r for r in scored if r.get("task_pass")]
        overlay_checked = [r for r in rs if r.get("overlay_pref_checked")]
        overlay_failed = [r for r in overlay_checked if r.get("overlay_pref_pass") is False]
        native_checked = [r for r in rs if r.get("pref_pass") is not None]
        native_failed = [r for r in native_checked if r.get("pref_pass") is False]
        corrections = [int(r.get("n_corrections") or 0) for r in rs]
        retry_seconds = sum(
            float(attempt.get("elapsed_s") or 0.0)
            for r in rs
            for attempt in (r.get("enforcement_attempts") or [])
        )
        retry_count = sum(len(r.get("enforcement_attempts") or []) for r in rs)
        elapsed_total = sum(float(r.get("elapsed_s") or 0.0) for r in rs)
        first_pass_total = sum(
            max(
                0.0,
                float(r.get("elapsed_s") or 0.0)
                - sum(
                    float(attempt.get("elapsed_s") or 0.0)
                    for attempt in (r.get("enforcement_attempts") or [])
                ),
            )
            for r in rs
        )
        rows.append(
            {
                "runtime": runtime,
                "model": model,
                "profile_id": profile_id,
                "condition": condition,
                "split": split,
                "n_records": len(rs),
                "n_scenarios": len({str(r.get("scenario_id")) for r in rs}),
                "n_runtime_errors": sum(1 for r in rs if r.get("runtime_error")),
                "objective_task_pass_rate": len(task_passed) / len(scored) if scored else None,
                "objective_task_pass_rate_higher_is_better": True,
                "overlay_pref_violation_rate": len(overlay_failed) / len(overlay_checked) if overlay_checked else None,
                "overlay_pref_violation_rate_lower_is_better": True,
                "native_pref_violation_rate": len(native_failed) / len(native_checked) if native_checked else None,
                "native_pref_violation_rate_lower_is_better": True,
                "total_user_corrections": sum(corrections),
                "total_user_corrections_lower_is_better": True,
                "mean_user_corrections": statistics.mean(corrections) if corrections else 0.0,
                "mean_user_corrections_lower_is_better": True,
                "mean_elapsed_s": statistics.mean(float(r.get("elapsed_s") or 0.0) for r in rs) if rs else None,
                "mean_elapsed_s_lower_is_better": True,
                "total_elapsed_s": elapsed_total,
                "total_elapsed_s_lower_is_better": True,
                "first_pass_total_elapsed_s": first_pass_total,
                "first_pass_total_elapsed_s_lower_is_better": True,
                "enforcement_retry_total_elapsed_s": retry_seconds,
                "enforcement_retry_total_elapsed_s_lower_is_better": True,
                "n_enforcement_retries": retry_count,
                "n_enforcement_retries_lower_is_better": True,
                "mean_compiled_rule_count_before": statistics.mean(int(r.get("compiled_rule_count") or 0) for r in rs) if rs else None,
                "frozen_preferences": all(bool(r.get("frozen_preferences")) for r in rs),
            }
        )
    return rows


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("condition") or ""),
        str(record.get("split") or ""),
        str(record.get("scenario_id") or ""),
        str(record.get("round_id") or ""),
    )


def _dedupe_latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    key_order: list[tuple[str, str, str, str]] = []
    for record in records:
        record_key = _record_key(record)
        if record_key not in latest_by_key:
            key_order.append(record_key)
        latest_by_key[record_key] = record
    return [latest_by_key[key] for key in key_order]


def _training_curve(records: list[dict[str, Any]], *, window: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        if rec.get("split") != "train" or rec.get("learning_step") is None:
            continue
        key = (
            str(rec.get("runtime")),
            str(rec.get("model")),
            str(rec.get("profile_id")),
            str(rec.get("condition")),
        )
        groups.setdefault(key, []).append(rec)
    rows = []
    for (runtime, model, profile_id, condition), rs in sorted(groups.items()):
        ordered = sorted(rs, key=lambda r: int(r.get("learning_step") or 0))
        for idx, rec in enumerate(ordered):
            window_records = ordered[max(0, idx - window + 1) : idx + 1]
            scored = [r for r in window_records if r.get("scored")]
            overlay_checked = [r for r in window_records if r.get("overlay_pref_checked")]
            pref_eligible = [
                r
                for r in window_records
                if r.get("overlay_pref_checked") or r.get("pref_pass") is not None
            ]
            rows.append(
                {
                    "runtime": runtime,
                    "model": model,
                    "profile_id": profile_id,
                    "condition": condition,
                    "learning_step": int(rec.get("learning_step") or 0),
                    "window_n": len(window_records),
                    "rolling_objective_task_pass_rate": (
                        sum(1 for r in scored if r.get("task_pass")) / len(scored)
                        if scored else None
                    ),
                    "rolling_overlay_pref_violation_rate": (
                        sum(1 for r in overlay_checked if r.get("overlay_pref_pass") is False) / len(overlay_checked)
                        if overlay_checked else None
                    ),
                    "rolling_mean_user_corrections": statistics.mean(
                        int(r.get("n_corrections") or 0) for r in window_records
                    ),
                    "rolling_mean_user_corrections_all": statistics.mean(
                        int(r.get("n_corrections") or 0) for r in window_records
                    ),
                    "rolling_mean_user_corrections_pref_eligible": (
                        statistics.mean(int(r.get("n_corrections") or 0) for r in pref_eligible)
                        if pref_eligible else None
                    ),
                    "compiled_rule_count_after": len(rec.get("compiled_rules_after") or []),
                    "memory_note_count_after": len(rec.get("memory_notes_after") or []),
                }
            )
    return rows


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    records_jsonl = run_dir / "records.jsonl"
    if records_jsonl.is_file():
        return _dedupe_latest_records(
            [
                json.loads(line)
                for line in records_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
    summary = run_dir / "summary_records.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return _dedupe_latest_records(data["records"])
    raise FileNotFoundError(f"no records found in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--curve-window", type=int, default=10)
    args = parser.parse_args()
    payload = summarize_records(load_records(args.run_dir), curve_window=args.curve_window)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
