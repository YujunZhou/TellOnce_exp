from __future__ import annotations

from typing import Any
import json

from .schemas import ConversationTurn, UserProfile


def artifact_text(artifact: Any) -> str:
    if isinstance(artifact, str):
        return artifact
    return json.dumps(artifact, ensure_ascii=False, sort_keys=True)


def detects_gold_leak(text: str, expected_answer: Any) -> bool:
    """Conservative leak detector for simulator turns.

    The simulator should never mention exact expected answer strings. This is
    intentionally simple; the real run can add stricter judge-based auditing.
    """
    gold = artifact_text(expected_answer)
    if not gold or gold in ("null", "None"):
        return False
    if len(gold) <= 12:
        return gold in text
    return gold[:120] in text


def simulate_correction(
    *,
    user_profile: UserProfile,
    current_artifact: Any,
    expected_answer: Any,
    turn_id: int,
    max_corrections: int = 1,
) -> list[ConversationTurn]:
    """Deterministic guardrail simulator for the first dry-run harness.

    It only checks whether user hard constraints are represented in the
    artifact text. It does not use the gold answer to construct corrections.
    """
    text = artifact_text(current_artifact).lower()
    turns: list[ConversationTurn] = []
    for idx, constraint in enumerate(user_profile.hard_constraints):
        if len(turns) >= max_corrections:
            break
        keywords = [tok for tok in constraint.lower().replace("/", " ").split() if len(tok) >= 6]
        if keywords and not any(keyword in text for keyword in keywords):
            correction_text = (
                f"You violated an already stated project constraint: {constraint}. "
                "Restore the constraint before optimizing the trip plan."
            )
            turns.append(
                ConversationTurn(
                    turn_id=turn_id + len(turns),
                    speaker="user_simulator",
                    turn_type="project_correction",
                    visible_to_agent=True,
                    text=correction_text,
                    linked_hidden_constraint_ids=[f"{user_profile.user_id}_hard_{idx}"],
                    reveals_gold_answer=detects_gold_leak(correction_text, expected_answer),
                    allowed_information_level="constraint_only",
                    metadata={"simulator_policy": "deterministic_constraint_smoke"},
                )
            )
    if not turns:
        turns.append(
            ConversationTurn(
                turn_id=turn_id,
                speaker="user_simulator",
                turn_type="accept_step",
                visible_to_agent=True,
                text="This step is acceptable; continue.",
                reveals_gold_answer=False,
                allowed_information_level="no_gold",
                metadata={"simulator_policy": "deterministic_constraint_smoke"},
            )
        )
    return turns
