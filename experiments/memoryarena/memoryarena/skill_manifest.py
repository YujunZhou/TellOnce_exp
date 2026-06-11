from __future__ import annotations

from .schemas import ConversationTurn, UserProfile


def compile_corrections_to_manifest(
    *,
    user_profile: UserProfile,
    corrections: list[ConversationTurn],
    seed: int | None = None,
    subset: str = "group_travel_planner",
    artifact_path: str = "workspace/trip_plan.json",
) -> dict:
    """Build a per-user manifest compatible with skill_library_v2 readers.

    This is a lightweight wrapper for MemoryArena. It avoids editing the shared
    `live_compile.py` while still emitting required schema fields.
    """
    skills = []
    artifact_name = artifact_path.rsplit("/", 1)[-1]
    for idx, correction in enumerate(corrections, start=1):
        if correction.turn_type not in {"preference_correction", "project_correction"}:
            continue
        atomic_id = f"MA_{user_profile.user_id}_{seed or 0}_{idx}"
        correction_id = correction.metadata.get(
            "correction_id",
            f"{user_profile.user_id}:unknown_session:turn_{correction.turn_id}",
        )
        skills.append(
            {
                "atomic_id": atomic_id,
                "rule_text": correction.text[:400],
                "scope": "global",
                "user_scope": "same_user",
                "user_id": user_profile.user_id,
                "domain": "workflow",
                "task_domain": "user_correction",
                "enforcement_phase": "execution",
                "source_correction_ids": [correction_id],
                "created_from_phase": "A",
                "compiler_version": "memoryarena_v1",
                "priority": "high",
                "severity": "blocking",
                "activation": "always",
                "hierarchy_role": "atomic",
                "children_ids": [],
                "applies_when": f"MemoryArena {subset} same-user held-out session",
                "enforcement": "verify-retry",
                "enforcement_config": {
                    "mode": "verify_retry",
                    "artifact": artifact_path,
                    "max_retry": 2,
                },
                "applicability": {
                    "task_family": subset,
                    "artifact_types": [artifact_name],
                    "same_user_only": True,
                },
                "trigger": {
                    "artifact_paths": [artifact_path],
                    "linked_hidden_constraint_ids": correction.linked_hidden_constraint_ids,
                },
                "verifier": {
                    "check_name": "hard_constraint_preserved",
                    "target_path": artifact_path,
                    "gold_access_level": "hidden_constraint",
                    "repair_hint": "Restore the missing user constraint before optimizing preferences.",
                },
                "runtime_requirements": {
                    "needs_file_access": True,
                    "needs_post_generation_verify": True,
                    "prompt_only_fallback": True,
                },
                "skill_dir": None,
                "hook_events": [],
                "n_hook_buckets": 0,
                "_provenance": {
                    "source": "memoryarena_phase_a_correction",
                    "correction_id": correction_id,
                    "user_id": user_profile.user_id,
                    "turn_id": correction.turn_id,
                    "linked_hidden_constraint_ids": correction.linked_hidden_constraint_ids,
                    "seed": seed,
                },
            }
        )
    return {
        "version": "memoryarena_v1",
        "skills": skills,
        "by_domain": {"workflow": [s["atomic_id"] for s in skills]},
    }
