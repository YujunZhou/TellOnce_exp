from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

from .memoryarena_loader import load_rows


def row_key(row: dict) -> str:
    return str(row.get("id") or row.get("row_id") or json.dumps(row, sort_keys=True)[:200])


def stable_bucket(row: dict, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{row_key(row)}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def split_rows(rows: list[dict], *, seed: int, train_frac: float) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: stable_bucket(row, seed))
    n_train = int(round(len(ordered) * train_frac))
    n_train = min(max(n_train, 1), len(ordered) - 1)
    return ordered[:n_train], ordered[n_train:]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--subset", default=None)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    subset = args.subset or (args.input.stem if args.input is not None else "group_travel_planner")
    input_path = args.input or Path(f"data/{subset}.jsonl")
    rows = load_rows(input_path)
    for row in rows:
        row.setdefault("subset", subset)
    train_rows, test_rows = split_rows(rows, seed=args.seed, train_frac=args.train_frac)
    train_path = args.out_dir / f"{subset}_train_rows.jsonl"
    test_path = args.out_dir / f"{subset}_test_rows.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)
    meta = {
        "source": str(input_path),
        "subset": subset,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "n_total": len(rows),
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "sessions_per_row": 6,
        "phase_a_subtasks_per_row": 3,
        "phase_b_subtasks_per_row": 3,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    (args.out_dir / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
