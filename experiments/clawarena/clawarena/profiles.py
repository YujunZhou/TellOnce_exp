"""User profiles for the ClawArena experiment pilot."""
from __future__ import annotations

from copy import deepcopy


P1 = {
    "profile_id": "p1",
    "display_name": "Primary Researcher Profile",
    "mainline": (
        "Keep the experiment aligned with the paper RQ: show that "
        "compiled runtime enforcement reduces repeated user corrections "
        "without lowering objective task quality."
    ),
    "durable_preferences": [
        {
            "rule_id": "self_evidence_first",
            "rule_text": (
                "When sources conflict, build an explicit evidence table before "
                "choosing an answer; do not rely on the most recent or most public "
                "source by default."
            ),
        },
        {
            "rule_id": "self_preserve_mainline",
            "rule_text": (
                "Do not drift into generic benchmark chasing; every decision must "
                "serve the experiment RQ and the current project mainline."
            ),
        },
        {
            "rule_id": "self_decision_log",
            "rule_text": (
                "Keep a concise decision log that states which constraints were "
                "used, which conflicts remain, and why the final answer follows."
            ),
        },
        {
            "rule_id": "self_no_hidden_answer_leak",
            "rule_text": (
                "Corrections may describe violated process rules, but must not "
                "reveal the gold answer or tell the agent which option to pick."
            ),
        },
        {
            "rule_id": "self_latest_state",
            "rule_text": (
                "Before finalizing, account for staged updates and silent state "
                "changes instead of assuming earlier notes are still complete."
            ),
        },
    ],
}


REAL_DERIVED_01 = {
    "profile_id": "real_derived_01",
    "display_name": "Real-Derived Profile 01",
    "mainline": (
        "Maintain a long-running project plan without deleting already accepted "
        "constraints; surface conflicts before changing the plan."
    ),
    "durable_preferences": [
        {
            "rule_id": "rd01_preserve_constraints",
            "rule_text": (
                "Previously accepted hard constraints must be preserved in later "
                "revisions; if a new step would violate a prior constraint, surface "
                "the conflict explicitly instead of silently dropping the constraint."
            ),
        },
        {
            "rule_id": "rd01_decision_log",
            "rule_text": (
                "Keep a concise decision log per revision, listing constraints used, "
                "conflicts unresolved, and the rationale for the chosen path."
            ),
        },
        {
            "rule_id": "rd01_cite_constraints",
            "rule_text": (
                "When justifying a final choice, cite the specific input constraint "
                "(by id or text) it satisfies, not just generic best-practice reasons."
            ),
        },
        {
            "rule_id": "rd01_no_silent_drop",
            "rule_text": (
                "Do not silently delete or alter accepted decisions from prior "
                "rounds; if a change is required, request explicit user approval."
            ),
        },
    ],
}


REAL_DERIVED_02 = {
    "profile_id": "real_derived_02",
    "display_name": "Real-Derived Profile 02",
    "mainline": (
        "Optimize project plans while preserving accessibility, timing, and risk "
        "commitments; ask only necessary clarifying questions."
    ),
    "durable_preferences": [
        {
            "rule_id": "rd02_accessibility_priority",
            "rule_text": (
                "Accessibility / inclusion constraints outrank performance or cost "
                "optimization; do not trade them away for marginal speedups."
            ),
        },
        {
            "rule_id": "rd02_explicit_assumptions",
            "rule_text": (
                "State assumptions explicitly in the final artifact; do not silently "
                "fill in missing data with plausible defaults."
            ),
        },
        {
            "rule_id": "rd02_minimal_questions",
            "rule_text": (
                "Ask only the minimum set of clarifying questions required to make "
                "a defensible decision; avoid scope expansion."
            ),
        },
        {
            "rule_id": "rd02_protect_commitments",
            "rule_text": (
                "Confirmed deadlines, meeting windows, and shared commitments cannot "
                "be moved silently; surface scheduling impact before changes."
            ),
        },
    ],
}


FICTIONAL_01 = {
    "profile_id": "fictional_01",
    "display_name": "Fictional Profile 01",
    "mainline": (
        "Coordinate a group plan where comfort and explicit reasoning outrank "
        "density of suggestions; prefer fewer, higher-confidence recommendations."
    ),
    "durable_preferences": [
        {
            "rule_id": "fc01_fewer_higher_conf",
            "rule_text": (
                "Prefer fewer recommendations with higher confidence over many "
                "lower-confidence options; rank by how well they satisfy hard "
                "constraints, not by completeness of coverage."
            ),
        },
        {
            "rule_id": "fc01_explain_rejection",
            "rule_text": (
                "When recommending an option, briefly explain why rejected "
                "alternatives failed (which constraint they violated)."
            ),
        },
        {
            "rule_id": "fc01_no_backtracking",
            "rule_text": (
                "Do not propose plans that require revisiting an already-completed "
                "step more than once; flag if the user request implies it."
            ),
        },
        {
            "rule_id": "fc01_comfort_outranks_optimization",
            "rule_text": (
                "Group comfort constraints (e.g., accessibility, fatigue, dietary) "
                "outrank logistical optimization."
            ),
        },
    ],
}


FICTIONAL_02 = {
    "profile_id": "fictional_02",
    "display_name": "Fictional Profile 02",
    "mainline": (
        "Keep budget-aware plans without dropping member-specific constraints; "
        "if no option satisfies all constraints, mark infeasible explicitly."
    ),
    "durable_preferences": [
        {
            "rule_id": "fc02_budget_separate",
            "rule_text": (
                "Show budget impact separately from constraint satisfaction; do not "
                "blend cost optimization into constraint reasoning."
            ),
        },
        {
            "rule_id": "fc02_no_drop_member_constraint",
            "rule_text": (
                "Budget savings or convenience cannot remove a member-specific hard "
                "constraint; if such a trade-off is forced, surface infeasibility."
            ),
        },
        {
            "rule_id": "fc02_terse_after_accept",
            "rule_text": (
                "After the user accepts a recommendation, do not re-explain the "
                "reasoning; move on or wait for the next request."
            ),
        },
        {
            "rule_id": "fc02_mark_infeasible",
            "rule_text": (
                "If no option satisfies all hard constraints, explicitly mark the "
                "task infeasible rather than silently relaxing a constraint."
            ),
        },
    ],
}


PROFILES = {
    "p1": P1,
    "real_derived_01": REAL_DERIVED_01,
    "real_derived_02": REAL_DERIVED_02,
    "fictional_01": FICTIONAL_01,
    "fictional_02": FICTIONAL_02,
}


def get_profile(profile_id: str) -> dict:
    if profile_id not in PROFILES:
        raise KeyError(f"unknown profile_id {profile_id!r}; options={sorted(PROFILES)}")
    return deepcopy(PROFILES[profile_id])

