from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import re

from .schemas import ConversationTurn, FRICTION_TURN_TYPES, Score


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def normalize_text(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(value: Any) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_text(value)))


def token_f1(predicted: Any, gold: Any) -> float:
    pred_tokens = tokens(predicted)
    gold_tokens = tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def flatten_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_fields(child, child_prefix))
        return out
    if isinstance(value, list):
        out = {}
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            out.update(flatten_fields(child, child_prefix))
        if not out:
            out[prefix] = []
        return out
    return {prefix: value}


def answer_field_f1(predicted: Any, gold: Any) -> float:
    gold_fields = flatten_fields(gold)
    if not gold_fields:
        return token_f1(predicted, gold)
    pred_fields = flatten_fields(predicted)
    scores = []
    for path, gold_value in gold_fields.items():
        pred_value = pred_fields.get(path, "")
        scores.append(token_f1(pred_value, gold_value))
    return sum(scores) / len(scores)


def hard_constraint_pass_rate(artifact: Any, constraints: Iterable[dict[str, Any]]) -> float:
    constraints = list(constraints)
    if not constraints:
        return 1.0
    artifact_text = normalize_text(artifact)
    passed = 0
    for constraint in constraints:
        required_terms = [normalize_text(x) for x in constraint.get("required_terms", [])]
        forbidden_terms = [normalize_text(x) for x in constraint.get("forbidden_terms", [])]
        if not required_terms and not forbidden_terms and constraint.get("text"):
            required_terms = [normalize_text(constraint["text"])]
        required_ok = all(term in artifact_text for term in required_terms if term)
        forbidden_ok = all(term not in artifact_text for term in forbidden_terms if term)
        if required_ok and forbidden_ok:
            passed += 1
    return passed / len(constraints)


def count_avoidable_corrections(turns: Iterable[ConversationTurn | dict[str, Any]]) -> int:
    count = 0
    for turn in turns:
        turn_type = turn.turn_type if isinstance(turn, ConversationTurn) else turn.get("turn_type")
        if turn_type in FRICTION_TURN_TYPES:
            count += 1
    return count


def count_gold_leaks(turns: Iterable[ConversationTurn | dict[str, Any]]) -> int:
    leaks = 0
    for turn in turns:
        reveals = turn.reveals_gold_answer if isinstance(turn, ConversationTurn) else turn.get("reveals_gold_answer")
        if reveals:
            leaks += 1
    return leaks


def build_score(
    *,
    session_id: str,
    condition: str,
    phase: str,
    predicted_answer: Any,
    expected_answer: Any,
    hard_constraints: list[dict[str, Any]],
    conversation: list[ConversationTurn | dict[str, Any]],
    correction_opportunities: int | list[Any],
    retry_count: int,
    constraint_artifact: Any | None = None,
    success_threshold: float = 0.8,
) -> Score:
    field_score = answer_field_f1(predicted_answer, expected_answer)
    constraint_score = hard_constraint_pass_rate(
        predicted_answer if constraint_artifact is None else constraint_artifact,
        hard_constraints,
    )
    quality = 0.7 * field_score + 0.3 * constraint_score
    avoidable = count_avoidable_corrections(conversation)
    n_opportunities = (
        len(correction_opportunities)
        if isinstance(correction_opportunities, list)
        else correction_opportunities
    )
    opps = max(n_opportunities, 1)
    user_turns = sum(
        1 for t in conversation
        if (t.speaker if isinstance(t, ConversationTurn) else t.get("speaker")) == "user_simulator"
    )
    agent_turns = sum(
        1 for t in conversation
        if (t.speaker if isinstance(t, ConversationTurn) else t.get("speaker")) == "agent"
    )
    return Score(
        session_id=session_id,
        condition=condition,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        quality_score=round(quality, 4),
        answer_field_f1=round(field_score, 4),
        hard_constraint_pass_rate=round(constraint_score, 4),
        final_success=(
            quality >= success_threshold
            and constraint_score >= 1.0
            and count_gold_leaks(conversation) == 0
        ),
        avoidable_corrections=avoidable,
        correction_opportunities=n_opportunities,
        avoidable_correction_rate=round(avoidable / opps, 4),
        user_turns=user_turns,
        agent_turns=agent_turns,
        retry_count=retry_count,
        gold_answer_leak_detected=count_gold_leaks(conversation) > 0,
    )
