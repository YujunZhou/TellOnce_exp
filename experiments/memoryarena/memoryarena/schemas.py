from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import json


ConditionName = Literal[
    "no_memory",
    "compiled_enforcement",
    "trace_native_cc",
    "trace_native_codex",
]
PhaseName = Literal["A", "B"]
ProfileType = Literal["real_derived", "fictional"]
TurnType = Literal[
    "task_info",
    "clarification_answer",
    "preference_correction",
    "project_correction",
    "accept_step",
]

FRICTION_TURN_TYPES = {"preference_correction", "project_correction"}


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    profile_type: ProfileType
    mainline_goal: str
    hard_constraints: list[str]
    soft_preferences: list[str]
    correction_style: str
    leak_guardrails: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryArenaTask:
    row_id: str
    session_id: str
    phase: PhaseName
    session_index: int
    question: str
    expected_answer: Any
    visible_context: str
    hidden_constraints: list[dict[str, Any]]
    correction_opportunities: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: int
    speaker: Literal["agent", "user_simulator", "system"]
    turn_type: TurnType | Literal["agent_response", "system_event"]
    visible_to_agent: bool
    text: str
    linked_hidden_constraint_ids: list[str] = field(default_factory=list)
    reveals_gold_answer: bool = False
    allowed_information_level: str = "unspecified"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleEvent:
    event_id: str
    run_id: str
    runtime: str
    model: str
    condition: ConditionName
    user_id: str
    phase: PhaseName
    session_id: str
    task_id: str
    turn_id: int
    skill_id: str
    skill_version: str
    source_correction_ids: list[str]
    opportunity_id: str
    retrieved: bool
    retrieval_rank: int | None
    retrieval_score: float | None
    applicable: bool
    applied: bool
    enforcement_mode: str
    decision: str
    artifact_path: str
    artifact_hash_before: str = ""
    artifact_hash_after: str = ""
    reason: str = ""


@dataclass(frozen=True)
class VerifierEvent:
    event_id: str
    run_id: str
    verifier_id: str
    verifier_type: str
    check_family: str
    phase: PhaseName
    session_id: str
    condition: ConditionName
    skill_id: str
    check_name: str
    target_path: str
    json_pointer: str
    expected_predicate: str
    observed_value: str
    status: Literal["pass", "fail", "fail_then_pass", "not_applicable"]
    status_before: str
    status_after: str
    retry_count: int
    max_retries: int
    repair_action: str
    gold_access_level: str
    leak_sensitive: bool
    before_violation: str = ""
    after_fix: str = ""


@dataclass(frozen=True)
class Score:
    session_id: str
    condition: ConditionName
    phase: PhaseName
    quality_score: float
    answer_field_f1: float
    hard_constraint_pass_rate: float
    final_success: bool
    avoidable_corrections: int
    correction_opportunities: int
    avoidable_correction_rate: float
    user_turns: int
    agent_turns: int
    retry_count: int
    gold_answer_leak_detected: bool


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(value), ensure_ascii=False) + "\n")
