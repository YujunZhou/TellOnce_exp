"""Load ClawArena scenario data from a checked-out repository."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import MULTI_CHOICE, EXEC_CHECK, Round, Scenario


def load_scenario(clawarena_root: str | Path, scenario_id: str) -> Scenario:
    root = Path(clawarena_root)
    q_path = root / "data" / "clawarena" / "eval" / scenario_id / "questions.json"
    if not q_path.is_file():
        raise FileNotFoundError(f"ClawArena questions.json not found: {q_path}")
    data = json.loads(q_path.read_text(encoding="utf-8"))
    raw_rounds = data.get("rounds")
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise ValueError(f"scenario {scenario_id} has no rounds")

    rounds: list[Round] = []
    for idx, item in enumerate(raw_rounds):
        if not isinstance(item, dict):
            raise ValueError(f"round {idx} is not an object")
        round_id = str(item.get("id") or "").strip()
        round_type = str(item.get("type") or item.get("question_type") or "").strip()
        question = str(item.get("question") or "").strip()
        if not round_id:
            raise ValueError(f"round {idx} missing id")
        if round_type not in {MULTI_CHOICE, EXEC_CHECK}:
            raise ValueError(f"round {round_id} has unsupported type {round_type!r}")
        if not question:
            raise ValueError(f"round {round_id} missing question")
        eval_spec = item.get("eval") or {}
        if round_type == MULTI_CHOICE:
            if not isinstance(eval_spec, dict) or not eval_spec.get("answer"):
                raise ValueError(f"multi_choice round {round_id} missing eval.answer")
        rounds.append(
            Round(
                round_id=round_id,
                round_index=idx,
                round_type=round_type,
                question=question,
                eval_spec=eval_spec if isinstance(eval_spec, dict) else {},
                pref_spec=item.get("pref") if isinstance(item.get("pref"), dict) else {},
                feedback=item.get("feedback") if isinstance(item.get("feedback"), dict) else {},
                update_ids=[
                    str(x) for x in item.get("update_ids", []) if str(x).strip()
                ],
                raw=item,
            )
        )
    return Scenario(
        # The framework manifest and requested evaluation directory define the
        # task identity. One pinned file still carries its old trace_c5 label.
        scenario_id=scenario_id,
        description=str(data.get("desc") or ""),
        rounds=rounds,
    )


def project_native_scenario(scenario: Scenario, *, dataset_revision: str, task_family: str,
                            split, user_id: str, seed: int):
    """Project original public questions/options; keep all native checks private.

    This preserves the benchmark's explicit task requirements. It does not
    preinstall a real-user profile, a hidden overlay, or native correction
    templates. The runner separately prepares official public workspace assets.
    """
    from tellonce_common.contracts import CommonInstance
    from tellonce_common.provenance import content_hash, canonical_json_bytes
    from tellonce_common.users import USERS as REAL_USERS
    from tellonce_common.role_inputs import PublicUnit, PrivateEvaluation

    if user_id not in REAL_USERS or not dataset_revision or not task_family:
        raise ValueError('explicit real user, revision and task family required')
    if not scenario.scenario_id or not scenario.rounds:
        raise ValueError('native scenario is empty')
    units, evaluations, seen = [], [], set()
    for index, round_obj in enumerate(scenario.rounds):
        if round_obj.round_id in seen or round_obj.round_index != index:
            raise ValueError('native round identities/order must be unique and complete')
        seen.add(round_obj.round_id)
        context = {'round_type': round_obj.round_type, 'update_ids': list(round_obj.update_ids)}
        if round_obj.round_type == MULTI_CHOICE:
            options = round_obj.eval_spec.get('options')
            if (not isinstance(options, dict) or not options or
                    any(not isinstance(k, str) or not isinstance(v, str) for k, v in options.items())):
                raise ValueError('native multiple-choice task requires public text options')
            # Never serialize the entire eval spec: `answer` is in the same dict.
            context['options'] = dict(options)
            context['answer_format'] = (
                r'End with your selected option letters inside \boxed{A,B,...} '
                r'(\bbox{A,B,...} is also accepted). This is the benchmark answer marker; '
                'do not put unrelated letters inside it.')
        elif round_obj.round_type != EXEC_CHECK:
            raise ValueError('unsupported native round type')
        instance = CommonInstance.build(benchmark='clawarena', dataset_revision=dataset_revision,
            split=split, task_family=task_family, instance_id=scenario.scenario_id,
            subtask_id=round_obj.round_id,
            public_input_hash=content_hash({'instruction': round_obj.question, 'context': context}),
            seed=seed, cluster_id=scenario.scenario_id, profile_id=user_id)
        units.append(PublicUnit(instance, scenario.scenario_id, round_obj.question,
                                canonical_json_bytes(context).decode()))
        evaluations.append(PrivateEvaluation(instance.common_instance_id, canonical_json_bytes({
            'scenario_id': scenario.scenario_id, 'round_id': round_obj.round_id,
            'round_index': index, 'round_type': round_obj.round_type,
            'eval': round_obj.eval_spec, 'native_pref': round_obj.pref_spec}).decode()))
    return tuple(units), tuple(evaluations)
