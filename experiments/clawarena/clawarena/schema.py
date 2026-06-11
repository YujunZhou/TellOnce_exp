"""Shared schemas for the experiments ClawArena harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MULTI_CHOICE = "multi_choice"
EXEC_CHECK = "exec_check"


@dataclass(frozen=True)
class Round:
    round_id: str
    round_index: int
    round_type: str
    question: str
    eval_spec: dict[str, Any] = field(default_factory=dict)
    pref_spec: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    update_ids: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_answer(self) -> list[str]:
        answer = self.eval_spec.get("answer", [])
        if isinstance(answer, list):
            return [str(x).strip().upper() for x in answer if str(x).strip()]
        if isinstance(answer, str) and answer.strip():
            return [answer.strip().upper()]
        return []

    @property
    def options(self) -> dict[str, str]:
        opts = self.eval_spec.get("options", {})
        if not isinstance(opts, dict):
            return {}
        return {str(k).strip().upper(): str(v) for k, v in opts.items()}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    rounds: list[Round]


@dataclass
class RuntimeResponse:
    parsed: dict[str, Any] | None
    raw_text: str
    runtime_error: str | None = None
    process_ok: bool = True
    parse_error: str | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    runtime_provenance: dict[str, Any] = field(default_factory=dict)
