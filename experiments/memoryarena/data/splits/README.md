# MemoryArena-derived splits

The MemoryArena experiments wrap the **MemoryArena** agent-memory
benchmark with a user-in-the-loop interaction layer
(see `memoryarena/wrapper.py` and `memoryarena/user_simulator.py`).
We do **not** redistribute MemoryArena rows here; regenerate them locally.

Only `split_meta.json` (the split recipe) is committed. The `*_train_rows.jsonl`
/ `*_test_rows.jsonl` files are produced by the steps below and are gitignored.

## Regenerate the splits

For each subset, download the source rows then create the seeded train/test split:

```bash
# 1) Download a MemoryArena subset (HuggingFace: ZexueHe/memoryarena)
python -m memoryarena.download_memoryarena --subset group_travel_planner --split test --out data/group_travel_planner.jsonl

# 2) Deterministic, seeded train/test split -> data/splits/<subset>_{train,test}_rows.jsonl
python -m memoryarena.split_rows --input data/group_travel_planner.jsonl --subset group_travel_planner --out-dir data/splits --train-frac 0.8
```

Repeat for every subset. `split_rows.py` also (re)writes `split_meta.json`
recording the seed, fractions, and row counts actually used.

## Subsets used in the paper

- **In-distribution**: `group_travel_planner`, `bundled_shopping`, `progressive_search`
- **Out-of-distribution**: `formal_reasoning_phys`, `formal_reasoning_math`

The exact filenames consumed by `scripts/run_trace_memoryarena.sh` are
`<subset>_train_rows.jsonl` and `<subset>_test_rows.jsonl`.
