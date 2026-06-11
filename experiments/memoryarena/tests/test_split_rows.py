from pathlib import Path
import json

from memoryarena import split_rows as split_rows_mod


def test_subset_default_reads_matching_dataset(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    travel_row = {"id": "travel_only", "questions": ["travel"], "answers": ["travel"]}
    shopping_rows = [
        {"id": "shop_1", "questions": ["shop1"], "answers": ["shop1"]},
        {"id": "shop_2", "questions": ["shop2"], "answers": ["shop2"]},
    ]
    (data_dir / "group_travel_planner.jsonl").write_text(
        json.dumps(travel_row) + "\n",
        encoding="utf-8",
    )
    (data_dir / "bundled_shopping.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in shopping_rows),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["split_rows", "--subset", "bundled_shopping", "--train-frac", "0.5"],
    )

    split_rows_mod.main()

    train_path = tmp_path / "data/splits/bundled_shopping_train_rows.jsonl"
    test_path = tmp_path / "data/splits/bundled_shopping_test_rows.jsonl"
    split_ids = {
        json.loads(line)["id"]
        for path in (train_path, test_path)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert split_ids == {"shop_1", "shop_2"}
