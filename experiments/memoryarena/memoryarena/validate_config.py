from __future__ import annotations

from pathlib import Path
import json
import sys


REQUIRED_TOP_LEVEL = {
    "dataset",
    "runtime_plan",
    "profiles",
    "conditions",
    "phase_split",
    "primary_metrics",
}


def validate_config(path: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_TOP_LEVEL - set(data)
    if missing:
        errors.append(f"missing top-level keys: {sorted(missing)}")
    if data.get("dataset", {}).get("subset") != "group_travel_planner":
        errors.append("dataset.subset must be group_travel_planner for the main run")
    conditions = data.get("conditions", [])
    expected_conditions = [
        "no_memory",
        "compiled_enforcement",
        "trace_native_cc",
        "trace_native_codex",
    ]
    if conditions != expected_conditions:
        errors.append(f"conditions must be {expected_conditions}")
    profiles = data.get("profiles", {})
    if profiles.get("n_total") != 8:
        errors.append("profiles.n_total must be 8")
    if profiles.get("n_real_derived") != 4 or profiles.get("n_fictional") != 4:
        errors.append("profiles split must be 4 real-derived + 4 fictional")
    primary = set(data.get("primary_metrics", []))
    for metric in ("avoidable_correction_rate", "objective_task_quality"):
        if metric not in primary:
            errors.append(f"primary metric missing: {metric}")
    split = data.get("phase_split", {})
    if split.get("phase_b_sessions_per_user_condition", 0) < 10:
        errors.append("Phase B should have at least 10 sessions per user-condition")
    if split.get("phase_a_min_usable_corrections", 0) < 3:
        errors.append("Phase A should require at least 3 usable corrections")
    if split.get("phase_a_max_sessions_per_user_condition", 0) < split.get("phase_a_sessions_per_user_condition", 0):
        errors.append("Phase A max sessions must be >= default Phase A sessions")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: validate_config.py CONFIG.json", file=sys.stderr)
        return 2
    errors = validate_config(Path(argv[1]))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
