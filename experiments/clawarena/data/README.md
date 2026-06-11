# ClawArena data

The ClawArena experiments run against the **ClawArena** human-in-the-loop
coding-agent benchmark. We do **not** redistribute the benchmark in this repository.
Download it from its official release and point the `CLAWARENA_ROOT` environment
variable (used by `scripts/run_trace_clawarena.sh`) at the resulting directory.

## Expected layout

The loader (`clawarena/clawarena_loader.py`) reads each scenario's
`questions.json` from:

```
<CLAWARENA_ROOT>/data/clawarena/eval/<scenario_id>/questions.json
```

## Splits used in the paper

- **Train** (32 scenarios, 4 families):
  `hil_c1..c8`, `hil_d1..d8`, `hil_e1..e8`, `hil_f1..f8`
- **In-distribution (ID) eval**: held-out rounds of the same 32 scenarios
  (the rounds after the training round limit, `--train-round-limit 5`).
- **Out-of-distribution (OOD) eval** (30 scenarios, unseen families):
  `hil_g1..g7`, `hil_h1/h2/h3/h5/h7/h8`, `hil_i1..i8`, `hil_j1..j8`, `hil_s1`.

These lists are encoded directly in `scripts/run_trace_clawarena.sh`.
