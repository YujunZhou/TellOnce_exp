"""Cross-scenario preference overlay checks for ClawArena the experiments."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schema import EXEC_CHECK, Round


OVERLAY_RULES = [
    {
        "rule_id": "O1_JSON_CONTRACT",
        "requirement": "Return exactly one JSON object with answer, rationale, decision_log, and unresolved_questions.",
    },
    {
        "rule_id": "O2_RATIONALE_LABELS",
        "requirement": "Format rationale with explicit 'Evidence:' and 'Decision:' labels.",
    },
    {
        "rule_id": "O3_UPDATE_STATUS",
        "requirement": "In decision_log, include an explicit updates status using 'updates:' such as 'updates:none' or 'updates:upd1,upd2'.",
    },
    {
        "rule_id": "O4_UNRESOLVED_LIST",
        "requirement": "Represent unresolved_questions as a JSON list, using [] when there are none.",
    },
]


def overlay_rule_prompt() -> list[dict[str, str]]:
    return [dict(rule) for rule in OVERLAY_RULES]


def evaluate_overlay(
    *,
    round_obj: Round,
    parsed: dict[str, Any] | None,
    workspace: Path,
    update_events: list[dict[str, Any]],
) -> dict[str, Any]:
    del workspace
    failures: list[dict[str, str]] = []
    if not isinstance(parsed, dict):
        failures.append(
            {
                "rule_id": "O1_JSON_CONTRACT",
                "message": "response was not a parsed JSON object",
            }
        )
        return _result(failures)

    required = {"answer", "rationale", "decision_log", "unresolved_questions"}
    missing = sorted(required - set(parsed))
    if missing:
        failures.append(
            {
                "rule_id": "O1_JSON_CONTRACT",
                "message": f"missing required JSON keys: {', '.join(missing)}",
            }
        )

    decision_log = str(parsed.get("decision_log") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    if not decision_log:
        failures.append(
            {
                "rule_id": "O3_UPDATE_STATUS",
                "message": "decision_log is empty",
            }
        )
    elif "updates:" not in decision_log.lower():
        failures.append(
            {
                "rule_id": "O3_UPDATE_STATUS",
                "message": "decision_log did not include an explicit 'updates:' status",
            }
        )
    else:
        update_status_error = _validate_update_status(decision_log, update_events)
        if update_status_error:
            failures.append(
                {
                    "rule_id": "O3_UPDATE_STATUS",
                    "message": update_status_error,
                }
            )

    rationale_lower = rationale.lower()
    if "evidence:" not in rationale_lower or "decision:" not in rationale_lower:
        failures.append(
            {
                "rule_id": "O2_RATIONALE_LABELS",
                "message": "rationale did not include both 'Evidence:' and 'Decision:' labels",
            }
        )

    if not isinstance(parsed.get("unresolved_questions"), list):
        failures.append(
            {
                "rule_id": "O4_UNRESOLVED_LIST",
                "message": "unresolved_questions is not a JSON list",
            }
        )

    return _result(failures)


def overlay_correction_text(round_id: str, failures: list[dict[str, str]]) -> str:
    rule_ids = sorted({str(row.get("rule_id")) for row in failures if row.get("rule_id")})
    detail = "; ".join(str(row.get("message") or "") for row in failures if row.get("message"))
    requirements = " ".join(
        rule["requirement"] for rule in OVERLAY_RULES if rule["rule_id"] in set(rule_ids)
    )
    return (
        f"In round {round_id}, you violated cross-task user preference overlay rule(s): "
        f"{', '.join(rule_ids)}. {detail}. For future rounds, apply these user preferences "
        f"before finalizing the task response or workspace artifact: {requirements}"
    )


def _result(failures: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "overlay_pref_checked": True,
        "overlay_pref_pass": not failures,
        "overlay_pref_failures": failures,
        "overlay_pref_rule_ids": [rule["rule_id"] for rule in OVERLAY_RULES],
    }


def _validate_update_status(
    decision_log: str, update_events: list[dict[str, Any]]
) -> str | None:
    expected = _expected_update_roots(update_events)
    status = _extract_update_status(decision_log)
    if not expected:
        if re.search(r"\b(none|no[-_\s]?updates?|n/a|null|empty)\b", status):
            return None
        return "decision_log update status should be 'updates:none' when no staged updates applied"
    mentioned = set(re.findall(r"\bupd\d+\b", status.lower()))
    missing = sorted(expected - mentioned)
    if missing:
        return "decision_log update status missed applied update id(s): " + ", ".join(missing)
    return None


def _expected_update_roots(update_events: list[dict[str, Any]]) -> set[str]:
    roots: set[str] = set()
    for event in update_events:
        if not isinstance(event, dict):
            continue
        if str(event.get("status") or "").lower() not in {"applied", "already_applied"}:
            continue
        update_id = str(event.get("update_id") or "").lower()
        match = re.match(r"(upd\d+)", update_id)
        roots.add(match.group(1) if match else update_id)
    return {root for root in roots if root}


def _extract_update_status(decision_log: str) -> str:
    match = re.search(r"updates\s*:\s*([^\n.;]*)", decision_log, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip().lower()
