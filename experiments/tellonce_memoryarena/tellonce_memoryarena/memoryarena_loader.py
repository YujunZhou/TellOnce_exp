from __future__ import annotations

import os
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


def memoryarena_source():
    """The official MemoryArena checkout; run_trace.py sets it from the run configuration."""
    return Path(os.environ.get('TELLONCE_MEMORYARENA_SOURCE',
                               Path(__file__).resolve().parents[2]/'vendor/MemoryArena'))


def project_native_row(row: dict[str, Any], *, subset: str, dataset_revision: str,
                       split, user_id: str, seed: int):
    """Separate public native questions from scorer-only reference material.

    The common module must be installed/on PYTHONPATH by the reviewed runner.
    This projection does not initialize a tool environment or call a model.
    Shopping/search require their own reviewed environment projections.
    """
    from tellonce_common.contracts import CommonInstance
    from tellonce_common.provenance import content_hash, canonical_json_bytes
    from tellonce_common.role_inputs import PublicUnit, PrivateEvaluation
    from tellonce_common.users import USERS as REAL_USERS

    if subset != 'group_travel_planner':
        raise ValueError('only the travel-planning subset has a native environment adapter here')
    if user_id not in REAL_USERS or not dataset_revision:
        raise ValueError('native rows require a registered simulated user and pinned data revision')
    if 'subset' in row and row['subset'] != subset:
        raise ValueError('row belongs to another dataset subset')
    if type(row.get('id')) not in {int, str} or not str(row['id']).strip():
        raise ValueError('native row identity is required')
    questions, answers = row.get('questions'), row.get('answers')
    if (not isinstance(questions, list) or not questions or not isinstance(answers, list)
            or len(questions) != len(answers) or any(not isinstance(q, str) or not q.strip() for q in questions)):
        raise ValueError('native questions and reference answers must align without truncation')
    task_id = subset + ':' + str(row['id'])
    from .travel_submission import (adapted_output_instructions, GROUP_PLAN_CONTRACT,
                                    TRAVEL_PUBLIC_INPUT_VERSION)
    base = row.get('base_person')
    if (not isinstance(base, dict) or not isinstance(base.get('name'), str)
            or not isinstance(base.get('query'), str) or not isinstance(base.get('daily_plans'), list)):
        raise ValueError('travel requires the official initial base-person query and plan')
    common_context = {'base_person': {k: base[k] for k in ('name', 'query', 'daily_plans')},
                      'output_instructions':adapted_output_instructions(memoryarena_source()),
                      'output_instructions_version': TRAVEL_PUBLIC_INPUT_VERSION,
                      'travel_plan_contract': GROUP_PLAN_CONTRACT,
                      'available_tools': ['FlightSearch', 'RestaurantSearch', 'AccommodationSearch',
                                          'AttractionSearch', 'DistanceMatrix', 'CitySearch']}
    if any(not isinstance(answer, list) or not answer for answer in answers):
        raise ValueError('travel references must contain native daily plans')
    units, evaluations = [], []
    for index, question in enumerate(questions):
        context = dict(common_context)
        public_hash = content_hash({'instruction': question, 'context': context})
        instance = CommonInstance.build(benchmark='memoryarena', dataset_revision=dataset_revision,
            split=split, task_family=subset, instance_id=task_id, subtask_id=str(index+1),
            public_input_hash=public_hash, seed=seed, cluster_id=task_id, profile_id=user_id)
        units.append(PublicUnit(instance, task_id, question, canonical_json_bytes(context).decode()))
        evaluations.append(PrivateEvaluation(instance.common_instance_id, canonical_json_bytes(
            {'subset': subset, 'question': question, 'answer': answers[index], 'question_index': index,
             'native_row_id': row['id']}).decode()))
    return tuple(units), tuple(evaluations)
