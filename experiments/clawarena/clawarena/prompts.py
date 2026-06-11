"""Prompt builders for ClawArena the experiments runs."""
from __future__ import annotations

import json
from typing import Any
from pathlib import Path

from .conditions import ConditionState
from .schema import EXEC_CHECK, MULTI_CHOICE, Round, Scenario


def build_round_prompt(
    *,
    scenario: Scenario,
    round_obj: Round,
    condition: ConditionState,
    workspace: Path,
    update_events: list[dict[str, Any]],
) -> str:
    profile = condition.profile
    payload: dict[str, Any] = {
        "profile": {
            "display_name": profile.get("display_name"),
            "mainline": profile.get("mainline"),
            "durable_preferences": profile.get("durable_preferences", []),
        },
        "condition": condition.name,
        "scenario_id": scenario.scenario_id,
        "scenario_description": scenario.description,
        "round": {
            "id": round_obj.round_id,
            "index": round_obj.round_index,
            "type": round_obj.round_type,
            "question": round_obj.question,
            "options": round_obj.options,
            "update_ids": round_obj.update_ids,
        },
        "workspace": {
            "cwd": str(workspace),
            "applied_update_events": update_events,
        },
        "memory_notes": condition.memory_notes(),
        "compiled_enforcement_rules": condition.enforcement_rules(),
    }
    if round_obj.round_type == MULTI_CHOICE:
        answer_instruction = (
            "This is a multi-choice round. Inspect the workspace files if needed, "
            "but do not edit files. The JSON answer value must be a list of option "
            'letters, for example ["A", "C"].'
        )
    elif round_obj.round_type == EXEC_CHECK:
        answer_instruction = (
            "This is an exec-check round. The current working directory is the "
            "benchmark workspace. Inspect and edit files there as needed, then "
            "return the final JSON object. The JSON answer value should briefly "
            "describe the concrete files/actions completed."
        )
    else:
        answer_instruction = "Return the requested JSON answer."
    return (
        "You are solving a the experiments ClawArena round for one persistent user profile.\n"
        "Use only the public round question, options, workspace files, applied "
        "updates, and the user/profile rules below. Do not invent hidden ground truth.\n"
        "Do not reveal or assume any gold answer. Follow compiled enforcement "
        "rules when present.\n\n"
        f"{answer_instruction}\n\n"
        "Return exactly one JSON object with keys:\n"
        "- answer\n"
        "- rationale\n"
        "- decision_log\n"
        "- unresolved_questions\n\n"
        "Do not write Markdown, headings, tables, code fences, or prose outside "
        "that JSON object. If you need an evidence table, put it inside the "
        "`rationale` string.\n\n"
        "When inspecting files, avoid dumping entire files. Use targeted searches "
        "such as `rg -n` with small context, `sed -n` windows of at most 80 lines, "
        "and `head -5` for CSV headers. Never print large CSVs or full message logs.\n\n"
        "For multi-choice rounds, keep workspace inspection compact: at most four "
        "shell commands, each returning only short excerpts. Do not keep searching "
        "after you have enough evidence to choose.\n\n"
        "The value of decision_log must explicitly mention source conflicts or "
        "state updates considered. Keep rationale concise.\n\n"
        "Input payload:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def build_enforcement_retry_prompt(
    *,
    round_obj: Round,
    condition: ConditionState,
    workspace: Path,
    verifier_feedback: str,
) -> str:
    payload = {
        "round": {
            "id": round_obj.round_id,
            "type": round_obj.round_type,
            "question": round_obj.question,
        },
        "workspace": str(workspace),
        "compiled_enforcement_rules": condition.enforcement_rules(),
        "verifier_feedback": verifier_feedback,
    }
    return (
        "A runtime enforcement check found that the current workspace artifact "
        "violates a durable user/project preference. Repair the workspace artifact "
        "without changing unrelated files, then return exactly one JSON object with "
        "keys: answer, rationale, decision_log, unresolved_questions.\n\n"
        "Do not infer or reveal any hidden gold answer. Use only the verifier "
        "feedback below as preference guidance.\n\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )
