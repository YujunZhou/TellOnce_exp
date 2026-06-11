from __future__ import annotations

from .schemas import UserProfile


DEFAULT_PROFILES: list[UserProfile] = [
    UserProfile(
        user_id="real_derived_01",
        profile_type="real_derived",
        mainline_goal="Maintain a long-running plan without deleting already accepted constraints.",
        hard_constraints=[
            "Previously accepted hard constraints must be preserved in later revisions.",
            "When constraints conflict, surface the conflict before changing the plan.",
        ],
        soft_preferences=[
            "Start with hard constraints, then list trade-offs.",
            "Keep a decision log for every revision.",
        ],
        correction_style="Directly identify the violated previously stated constraint.",
        leak_guardrails=["Do not reveal the final answer or exact gold item."],
    ),
    UserProfile(
        user_id="real_derived_02",
        profile_type="real_derived",
        mainline_goal="Optimize travel plans while preserving accessibility and timing commitments.",
        hard_constraints=[
            "Accessibility constraints outrank price optimization.",
            "Confirmed meeting or event windows cannot be moved silently.",
        ],
        soft_preferences=[
            "Ask only necessary clarifying questions.",
            "State assumptions explicitly in the artifact.",
        ],
        correction_style="Point out the missing accessibility or timing constraint without giving the solution.",
        leak_guardrails=["Do not name the expected hotel, flight, or itinerary item."],
    ),
    UserProfile(
        user_id="real_derived_03",
        profile_type="real_derived",
        mainline_goal="Keep a stable project plan across sessions and avoid undoing earlier user decisions.",
        hard_constraints=[
            "Do not replace a prior accepted choice unless the user explicitly allows it.",
            "If a new request is impossible, keep the old plan and report the blocker.",
        ],
        soft_preferences=[
            "Prefer compact tables for constraints.",
            "Separate facts from recommendations.",
        ],
        correction_style="Emphasize that the agent overwrote a prior accepted decision.",
        leak_guardrails=["Do not expose hidden benchmark answers."],
    ),
    UserProfile(
        user_id="real_derived_04",
        profile_type="real_derived",
        mainline_goal="Build plans that are reproducible and auditable from saved project state.",
        hard_constraints=[
            "Every final choice must be traceable to an input constraint.",
            "Do not invent missing facts; record unresolved questions instead.",
        ],
        soft_preferences=[
            "Use stable IDs for people, locations, and constraints.",
            "Prefer deterministic JSON output over prose.",
        ],
        correction_style="Ask the agent to restore traceability or move unknowns to unresolved questions.",
        leak_guardrails=["Do not provide the exact expected JSON answer."],
    ),
    UserProfile(
        user_id="fictional_01",
        profile_type="fictional",
        mainline_goal="Coordinate a group trip where family comfort constraints are non-negotiable.",
        hard_constraints=[
            "Family comfort constraints outrank sightseeing density.",
            "No itinerary may require backtracking more than once per day.",
        ],
        soft_preferences=[
            "Prefer fewer, higher-confidence recommendations.",
            "Explain why rejected options failed.",
        ],
        correction_style="Call out the ignored family comfort rule.",
        leak_guardrails=["Do not disclose the benchmark answer."],
    ),
    UserProfile(
        user_id="fictional_02",
        profile_type="fictional",
        mainline_goal="Keep a budget-aware plan without dropping members' individual constraints.",
        hard_constraints=[
            "Budget savings cannot remove a member-specific hard constraint.",
            "If no option satisfies all constraints, mark the task infeasible.",
        ],
        soft_preferences=[
            "Show budget impact separately from constraint satisfaction.",
            "Avoid unnecessary explanations after acceptance.",
        ],
        correction_style="State that budget optimization overrode a member constraint.",
        leak_guardrails=["Do not reveal the exact gold answer."],
    ),
    UserProfile(
        user_id="fictional_03",
        profile_type="fictional",
        mainline_goal="Preserve a multi-person planning history and update only the requested part.",
        hard_constraints=[
            "Unrelated itinerary sections should not change between sessions.",
            "A new member's needs must be merged, not used to restart the plan.",
        ],
        soft_preferences=[
            "Summarize only changed fields.",
            "Keep old and new constraints in the same table.",
        ],
        correction_style="Tell the agent it restarted instead of merging.",
        leak_guardrails=["Do not reveal final field values from the gold answer."],
    ),
    UserProfile(
        user_id="fictional_04",
        profile_type="fictional",
        mainline_goal="Maintain a plan that prioritizes risk reduction and explicit uncertainty.",
        hard_constraints=[
            "Do not make a recommendation when required information is unknown.",
            "Risk constraints outrank convenience.",
        ],
        soft_preferences=[
            "List unresolved questions before final recommendations.",
            "Keep risk notes in the final artifact.",
        ],
        correction_style="Flag missing uncertainty handling or risk notes.",
        leak_guardrails=["Do not give the evaluator-only answer."],
    ),
]


def profile_by_id(user_id: str) -> UserProfile:
    for profile in DEFAULT_PROFILES:
        if profile.user_id == user_id:
            return profile
    raise KeyError(f"unknown profile: {user_id}")
