from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load MemoryArena rows from a local JSON or JSONL export."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("rows"), list):
            rows = data["rows"]
        else:
            rows = [data]
    if limit is not None:
        return rows[:limit]
    return rows


def mock_group_travel_rows(n_rows: int = 1, n_sessions: int = 16) -> list[dict[str, Any]]:
    """Deterministic local rows for dry-run and tests.

    This does not replace MemoryArena. It only lets the harness validate schema,
    workspace writing, and scoring before the real dataset export is present.
    """
    rows = []
    for row_idx in range(n_rows):
        questions = []
        answers = []
        for session_idx in range(n_sessions):
            person = f"member_{session_idx}"
            questions.append(
                f"Update the trip plan for {person}. Preserve prior confirmed constraints "
                f"and add the new requirement for session {session_idx}."
            )
            answers.append(
                {
                    "row": row_idx,
                    "member": person,
                    "preserve_prior_constraints": True,
                    "new_requirement_recorded": f"session_{session_idx}",
                }
            )
        rows.append(
            {
                "id": f"mock_travel_{row_idx}",
                "subset": "group_travel_planner",
                "background": "A group trip plan grows over many sessions.",
                "questions": questions,
                "answers": answers,
            }
        )
    return rows


def validate_memoryarena_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    questions = row.get("questions")
    answers = row.get("answers")
    if not isinstance(questions, list) or not questions:
        errors.append("row.questions must be a non-empty list")
    if not isinstance(answers, list) or not answers:
        errors.append("row.answers must be a non-empty list")
    if isinstance(questions, list) and isinstance(answers, list) and len(answers) < len(questions):
        errors.append("row.answers should cover every question used by the split")
    return errors
