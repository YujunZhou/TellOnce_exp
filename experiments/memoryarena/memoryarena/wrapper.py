from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import MemoryArenaTask, UserProfile, write_json


ARTIFACT_BY_SUBSET = {
    "group_travel_planner": ("trip_plan.json", "trip_plan"),
    "bundled_shopping": ("shopping_plan.json", "shopping_plan"),
    "progressive_search": ("search_report.json", "search_report"),
    "formal_reasoning_math": ("reasoning_solution.json", "reasoning_solution"),
    "formal_reasoning_phys": ("reasoning_solution.json", "reasoning_solution"),
}


def subset_from_row(row: dict[str, Any]) -> str:
    return str(row.get("subset") or row.get("config") or "group_travel_planner")


def artifact_spec_for_subset(subset: str) -> dict[str, str]:
    file_name, artifact_kind = ARTIFACT_BY_SUBSET.get(
        subset,
        ("task_artifact.json", "task_artifact"),
    )
    return {
        "subset": subset,
        "artifact": file_name,
        "artifact_kind": artifact_kind,
        "artifact_path": f"workspace/{file_name}",
    }


def artifact_path_for_task(root: Path, task: MemoryArenaTask) -> Path:
    artifact = str(task.metadata.get("artifact") or "trip_plan.json")
    return root / "workspace" / artifact


def safe_session_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return cleaned or "unknown"


def constraint_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in text.lower().replace("/", " ").replace("-", " ").split():
        token = token.strip(".,;:()[]{}")
        if len(token) >= 7 and token not in terms:
            terms.append(token)
    return terms[:4]


def _row_context(row: dict[str, Any]) -> str:
    parts = []
    for key in ("backgrounds", "background", "base_person", "context"):
        value = row.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def build_task_from_row(
    row: dict[str, Any],
    *,
    user_profile: UserProfile,
    phase: str,
    session_index: int,
) -> MemoryArenaTask:
    questions = row.get("questions") or []
    answers = row.get("answers") or []
    if session_index >= len(questions):
        raise IndexError(f"session_index {session_index} outside row questions")
    expected_answer = answers[session_index] if session_index < len(answers) else None
    row_id = str(row.get("id") or row.get("row_id") or "unknown_row")
    subset = subset_from_row(row)
    artifact_spec = artifact_spec_for_subset(subset)
    session_id = (
        f"{safe_session_component(subset)}_{safe_session_component(row_id)}_"
        f"{safe_session_component(user_profile.user_id)}_{phase}{session_index:02d}"
    )
    hidden_constraints = [
        {
            "constraint_id": f"{user_profile.user_id}_hard_{idx}",
            "source": "user_profile",
            "text": text,
            "required_terms": constraint_terms(text),
            "forbidden_terms": [],
        }
        for idx, text in enumerate(user_profile.hard_constraints)
    ]
    correction_opportunities = [c["constraint_id"] for c in hidden_constraints]
    return MemoryArenaTask(
        row_id=row_id,
        session_id=session_id,
        phase=phase,  # type: ignore[arg-type]
        session_index=session_index,
        question=str(questions[session_index]),
        expected_answer=expected_answer,
        visible_context=_row_context(row),
        hidden_constraints=hidden_constraints,
        correction_opportunities=correction_opportunities,
        metadata={
            "source": "memoryarena",
            **artifact_spec,
            "user_id": user_profile.user_id,
            "profile_type": user_profile.profile_type,
        },
    )


def _read_prior_session_state(prior_session_root: Path | None) -> dict:
    """Load accepted_constraints / accepted_decisions / prior artifact from the
    previous session in the same row, so q2 / q3 can build on q1 instead of
    seeing a blank workspace.

    Round 3 review fix (D3): MemoryArena interdependent subtask assumption
    requires q1->q2->q3 to share state. Previously each write_workspace reset
    accepted_* to empty, breaking the assumption. Now caller passes prior
    session_root; we extract:
      - accepted_constraints from prior input/project_state.json (if non-empty)
      - accepted_decisions from prior input/project_state.json (if non-empty)
      - prior_artifact_summary from prior workspace/<artifact>.json (the
        agent-produced answer that subsequent sessions should preserve)
      - prior_decision_log_text from prior workspace/decision_log.md
    Returns empty dict if prior_session_root is None or unreadable.
    """
    if prior_session_root is None:
        return {}
    state: dict = {}
    prior_state_path = prior_session_root / "input" / "project_state.json"
    if prior_state_path.is_file():
        try:
            prev = json.loads(prior_state_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                state["accepted_constraints"] = list(prev.get("accepted_constraints") or [])
                state["accepted_decisions"] = list(prev.get("accepted_decisions") or [])
        except Exception:
            pass
    prior_log_path = prior_session_root / "workspace" / "decision_log.md"
    if prior_log_path.is_file():
        try:
            text = prior_log_path.read_text(encoding="utf-8").strip()
            if text:
                state["prior_decision_log"] = text[:6000]
        except Exception:
            pass
    # Look for any artifact JSON under workspace/.
    prior_workspace = prior_session_root / "workspace"
    if prior_workspace.is_dir():
        for candidate in prior_workspace.glob("*.json"):
            if candidate.name == "artifact_schema.json":
                continue
            try:
                content = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(content, dict) and content.get("answer") is not None:
                    state.setdefault("prior_artifacts", []).append(
                        {"file": candidate.name, "content": content}
                    )
            except Exception:
                continue
    return state


def write_workspace(
    root: Path,
    task: MemoryArenaTask,
    user_profile: UserProfile,
    *,
    prior_session_root: Path | None = None,
) -> None:
    input_dir = root / "input"
    hidden_dir = root / "hidden"
    workspace_dir = root / "workspace"
    skills_dir = root / "skills"
    logs_dir = root / "logs"
    runtime_dir = root / "runtime"

    write_json(
        input_dir / "task_spec.json",
        {
            "row_id": task.row_id,
            "session_id": task.session_id,
            "phase": task.phase,
            "session_index": task.session_index,
            "question": task.question,
            "visible_context": task.visible_context,
            "metadata": task.metadata,
        },
    )
    prior_state = _read_prior_session_state(prior_session_root)
    write_json(
        input_dir / "project_state.json",
        {
            "session_id": task.session_id,
            "row_id": task.row_id,
            "phase": task.phase,
            "session_index": task.session_index,
            "visible_context": task.visible_context,
            "current_request": task.question,
            "accepted_constraints": prior_state.get("accepted_constraints", []),
            "accepted_decisions": prior_state.get("accepted_decisions", []),
            "prior_decision_log": prior_state.get("prior_decision_log"),
            "prior_artifacts": prior_state.get("prior_artifacts", []),
            "prior_session_root": str(prior_session_root) if prior_session_root else None,
        },
    )
    (input_dir / "visible_user_context.md").write_text(
        "\n".join(
            [
                f"# User {user_profile.user_id}",
                "",
                user_profile.mainline_goal,
                "",
                "Soft preferences:",
                *[f"- {pref}" for pref in user_profile.soft_preferences],
            ]
        ),
        encoding="utf-8",
    )
    write_json(hidden_dir / "user_profile.json", user_profile)
    write_json(hidden_dir / "gold_constraints.json", task.hidden_constraints)
    write_json(hidden_dir / "expected_answer.json", task.expected_answer)
    artifact = str(task.metadata["artifact"])
    artifact_kind = str(task.metadata["artifact_kind"])
    write_json(workspace_dir / artifact, {"session_id": task.session_id, "answer": None})
    write_json(
        workspace_dir / "artifact_schema.json",
        {
            "artifact": artifact,
            "artifact_kind": artifact_kind,
            "subset": task.metadata.get("subset"),
            "required_top_level_fields": ["session_id", "answer", "constraint_notes"],
            "notes": "Agent-visible schema only; no gold answer or hidden constraints.",
        },
    )
    (workspace_dir / "decision_log.md").write_text("", encoding="utf-8")
    write_json(workspace_dir / "unresolved_questions.json", [])
    write_json(skills_dir / "manifest.json", {"skills": []})
    (skills_dir / "compiled_rules.md").write_text("", encoding="utf-8")
    (skills_dir / "prompt_injection.md").write_text("", encoding="utf-8")
    write_json(
        skills_dir / "verifier_spec.json",
        {
            "artifact": str(task.metadata["artifact_path"]),
            "gold_access_level": "hidden_constraint",
            "uses_expected_answer": False,
        },
    )
    write_json(
        runtime_dir / "runtime_capabilities.json",
        {
            "runtime": "unset",
            "supports_hooks": None,
            "supports_file_write": True,
            "supports_retry": None,
            "supports_session_resume": None,
            "supports_structured_logs": True,
        },
    )
    write_json(
        runtime_dir / "adapter_config.json",
        {
            "skill_load_path": "skills/manifest.json",
            "artifact_path": str(task.metadata["artifact_path"]),
            "conversation_log": "logs/conversation.jsonl",
        },
    )
    for log_name in ("conversation.jsonl", "rule_events.jsonl", "verifier_events.jsonl"):
        (logs_dir / log_name).parent.mkdir(parents=True, exist_ok=True)
        (logs_dir / log_name).write_text("", encoding="utf-8")
