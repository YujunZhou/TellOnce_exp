from memoryarena.scoring import (
    answer_field_f1,
    build_score,
    count_avoidable_corrections,
    hard_constraint_pass_rate,
)


def test_answer_field_f1_scores_matching_fields():
    pred = {"hotel": "Alpha Inn", "day": {"morning": "museum"}}
    gold = {"hotel": "Alpha Inn", "day": {"morning": "museum"}}
    assert answer_field_f1(pred, gold) == 1.0


def test_answer_field_f1_penalizes_missing_field():
    pred = {"hotel": "Alpha Inn"}
    gold = {"hotel": "Alpha Inn", "flight": "F9"}
    assert 0 < answer_field_f1(pred, gold) < 1


def test_count_avoidable_corrections_only_counts_correction_turns():
    turns = [
        {"speaker": "user_simulator", "turn_type": "task_info"},
        {"speaker": "user_simulator", "turn_type": "preference_correction"},
        {"speaker": "user_simulator", "turn_type": "project_correction"},
        {"speaker": "agent", "turn_type": "agent_response"},
    ]
    assert count_avoidable_corrections(turns) == 2


def test_build_score_tracks_quality_and_gold_leaks():
    turns = [
        {"speaker": "agent", "turn_type": "agent_response", "reveals_gold_answer": False},
        {"speaker": "user_simulator", "turn_type": "accept_step", "reveals_gold_answer": False},
    ]
    score = build_score(
        session_id="s1",
        condition="compiled_enforcement",
        phase="B",
        predicted_answer={"hotel": "Alpha Inn"},
        expected_answer={"hotel": "Alpha Inn"},
        hard_constraints=[],
        conversation=turns,
        correction_opportunities=2,
        retry_count=1,
    )
    assert score.final_success is True
    assert score.avoidable_correction_rate == 0


def test_empty_constraint_terms_do_not_auto_pass():
    constraints = [{"text": "must preserve walking distance constraint"}]
    assert hard_constraint_pass_rate({"notes": []}, constraints) == 0.0
    assert hard_constraint_pass_rate(
        {"notes": ["must preserve walking distance constraint"]},
        constraints,
    ) == 1.0
