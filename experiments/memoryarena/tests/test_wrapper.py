from pathlib import Path
import json

from memoryarena.profiles import profile_by_id
from memoryarena.wrapper import build_task_from_row, write_workspace


def test_build_task_from_row_uses_question_answer_pair():
    profile = profile_by_id("fictional_01")
    row = {
        "id": "row1",
        "questions": ["q0", "q1"],
        "answers": [{"a": 0}, {"a": 1}],
        "background": "base trip",
    }
    task = build_task_from_row(row, user_profile=profile, phase="B", session_index=1)
    assert task.session_id == "group_travel_planner_row1_fictional_01_B01"
    assert task.question == "q1"
    assert task.expected_answer == {"a": 1}
    assert task.correction_opportunities


def test_write_workspace_creates_visible_and_hidden_files(tmp_path: Path):
    profile = profile_by_id("fictional_02")
    row = {"id": "row2", "questions": ["q0"], "answers": ["a0"]}
    task = build_task_from_row(row, user_profile=profile, phase="A", session_index=0)
    write_workspace(tmp_path, task, profile)
    assert (tmp_path / "input" / "task_spec.json").is_file()
    assert (tmp_path / "hidden" / "expected_answer.json").is_file()
    assert (tmp_path / "workspace" / "trip_plan.json").is_file()
    assert (tmp_path / "logs" / "conversation.jsonl").is_file()


def test_write_workspace_uses_subset_specific_artifact(tmp_path: Path):
    profile = profile_by_id("fictional_02")
    row = {
        "id": "row_shop",
        "subset": "bundled_shopping",
        "questions": ["q0"],
        "answers": [{"items": []}],
    }
    task = build_task_from_row(row, user_profile=profile, phase="A", session_index=0)
    write_workspace(tmp_path, task, profile)
    schema = json.loads((tmp_path / "workspace" / "artifact_schema.json").read_text(encoding="utf-8"))
    assert schema["artifact"] == "shopping_plan.json"
    assert (tmp_path / "workspace" / "shopping_plan.json").is_file()
    adapter = json.loads((tmp_path / "runtime" / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter["artifact_path"] == "workspace/shopping_plan.json"


def test_session_id_includes_subset_to_avoid_cross_subset_collisions():
    profile = profile_by_id("fictional_02")
    base = {"id": "shared_row", "questions": ["q0"], "answers": ["a0"]}
    travel = build_task_from_row(
        {**base, "subset": "group_travel_planner"},
        user_profile=profile,
        phase="B",
        session_index=0,
    )
    shopping = build_task_from_row(
        {**base, "subset": "bundled_shopping"},
        user_profile=profile,
        phase="B",
        session_index=0,
    )
    assert travel.session_id != shopping.session_id
    assert travel.session_id.startswith("group_travel_planner_")
    assert shopping.session_id.startswith("bundled_shopping_")


def test_public_task_spec_does_not_contain_gold_or_hidden_constraints(tmp_path: Path):
    profile = profile_by_id("fictional_02")
    row = {"id": "row3", "questions": ["q0"], "answers": [{"secret": "gold"}]}
    task = build_task_from_row(row, user_profile=profile, phase="B", session_index=0)
    write_workspace(tmp_path, task, profile)
    public_spec = json.loads((tmp_path / "input" / "task_spec.json").read_text(encoding="utf-8"))
    assert "expected_answer" not in public_spec
    assert "hidden_constraints" not in public_spec
    assert "correction_opportunities" not in public_spec
    assert "gold" not in json.dumps(public_spec, ensure_ascii=False)
