# Experiments — Reproducing TRACE on ClawArena and MemoryArena

This directory contains the experiment harness used to evaluate
**TRACE** under a simulated user-in-the-loop protocol with frozen
in-distribution (ID) and out-of-distribution (OOD) evaluation.

## Conditions shipped here

| Condition            | What it is                                                            |
|----------------------|-----------------------------------------------------------------------|
| `trace_native_cc`    | TRACE running natively as a **Claude Code** skill (runtime hooks).    |
| `trace_native_codex` | TRACE running natively as a **Codex CLI** skill (runtime hooks).      |
| `no_memory`          | Trivial control: the agent with no memory and no rules.               |
| `compiled_enforcement` | Prompt-side variant that injects the compiled rules into the prompt. |

The memory-system baselines from the paper (Mem0, Hindsight, ReMe-Light) are
**not** included in this release (see the top-level `README.md`).

## Layout

```
experiments/
├── clawarena/        # ClawArena coding-agent tasks
│   ├── clawarena/    #   Python package (runner, conditions, scoring, ...)
│   ├── scripts/run_trace_clawarena.sh
│   └── data/README.md     #   how to obtain the ClawArena benchmark
└── memoryarena/      # MemoryArena agent-memory tasks (+ user-in-the-loop wrapper)
    ├── memoryarena/  #   Python package (runner, wrapper, user_simulator, ...)
    ├── scripts/run_trace_memoryarena.sh
    ├── configs/           #   experiment-design config
    ├── tests/             #   unit tests for the wrapper / scoring / splits
    └── data/splits/README.md  #   how to regenerate the MemoryArena row splits
```

## Prerequisites

1. **Skill install** (for `trace_native_*`): install the preference-tracker skill
   at `~/.claude/skills/preference-tracker` — see
   [YujunZhou/preference-tracker](https://github.com/YujunZhou/preference-tracker). The
   harness wires the skill's hooks into each per-run isolated workspace.
2. **Benchmark data**: not redistributed. See
   `clawarena/data/README.md` (ClawArena) and
   `memoryarena/data/splits/README.md` (MemoryArena).
3. **Python 3.10+**. The harness uses only the standard library plus, optionally,
   `datasets` for the MemoryArena download helper.

## Running

ClawArena (one condition per invocation):

```bash
cd clawarena
CLAWARENA_ROOT=/path/to/ClawArena \
  scripts/run_trace_clawarena.sh trace_haiku claude_cli haiku trace_native_cc
# writes records.jsonl + summary_metrics.json under state/runtime/trace_haiku/
```

MemoryArena:

```bash
cd memoryarena
scripts/run_trace_memoryarena.sh trace_native_cc claude_cli haiku
# writes records.jsonl under state/runtime/.../
```

Use `--runtime oracle_dry_run` (ClawArena) or runtime `oracle_dry_run`
(MemoryArena) to dry-run the harness without any model call — useful to verify
the pipeline end-to-end before spending budget.

## Metrics

Each run writes per-round `records.jsonl`. The reported metrics are **task pass**
(benchmark objective checker / final task-completion score), **violation rate**
(whether the final output violates the checked user/project preference), and
**mean corrections** (average simulated-user corrections per held-out task);
ClawArena additionally reports test-time efficiency (user turns, wall-clock).
For ClawArena, `multi_summarize.py` aggregates `records.jsonl` into
`summary_metrics.json`.
