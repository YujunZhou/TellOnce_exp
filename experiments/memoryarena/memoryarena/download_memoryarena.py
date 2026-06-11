from __future__ import annotations

from pathlib import Path
import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--subset", default="group_travel_planner")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    out_path = args.out or Path(f"data/{args.subset}.jsonl")

    from datasets import load_dataset

    ds = load_dataset("ZexueHe/memoryarena", args.subset, split=args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            item = dict(row)
            item.setdefault("subset", args.subset)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(ds)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
