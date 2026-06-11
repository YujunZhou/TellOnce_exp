#!/usr/bin/env bash
# Run TRACE (and the No-Memory control) on the MemoryArena-derived tasks with
# our user-in-the-loop wrapper, using frozen ID/OOD eval.
#
# This public release ships only our method (TRACE) and the No-Memory control.
# The memory baselines compared in the paper (Mem0, Hindsight, ReMe-Light) are
# intentionally not included here; see ../../README.md for the rationale.
#
# Usage:
#   ./run_trace_memoryarena.sh <condition> <runtime> <model>
#     condition : trace_native_cc | trace_native_codex | no_memory | compiled_enforcement
#     runtime   : claude_cli | codex_cli | codex_cli_trace | oracle_dry_run
#     model     : haiku | sonnet (claude_cli); gpt-5.4-mini | gpt-5.5 (codex_cli)
#
# Prerequisites:
#   - For trace_native_*: install the preference-tracker skill at
#     ~/.claude/skills/preference-tracker (see https://github.com/YujunZhou/preference-tracker).
#   - Regenerate the MemoryArena row splits under data/splits/
#     (see ../data/splits/README.md).
set -euo pipefail

COND="${1:-trace_native_cc}"
RT="${2:-claude_cli}"
MODEL="${3:-haiku}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

[[ -f .env ]] && { set -a; source .env; set +a; }
unset ANTHROPIC_API_KEY Anthropic_API_KEY anthropic_api_key 2>/dev/null || true

TRAIN_LIMIT="${TRAIN_LIMIT:-5}"
ID_LIMIT="${ID_LIMIT:-3}"
OOD_LIMIT="${OOD_LIMIT:-3}"
PROFILES="${PROFILES:-real_derived_01 real_derived_02 real_derived_03 real_derived_04 fictional_01 fictional_02 fictional_03 fictional_04}"
TIMEOUT="${TIMEOUT:-600}"

S=data/splits
OUT="state/runtime/${MODEL//[^a-zA-Z0-9]/_}_memoryarena_${COND}"

# Train: 3 in-distribution domains. ID eval: held-out rows of the same domains.
# OOD eval: 2 unseen formal-reasoning domains.
PYTHONPATH=. python3 -u -m memoryarena.runner \
  --train-rows   "$S/group_travel_planner_train_rows.jsonl" "$S/bundled_shopping_train_rows.jsonl" "$S/progressive_search_train_rows.jsonl" \
  --train-row-limit "$TRAIN_LIMIT" \
  --id-test-rows "$S/group_travel_planner_test_rows.jsonl" "$S/bundled_shopping_test_rows.jsonl" "$S/progressive_search_test_rows.jsonl" \
  --id-test-row-limit "$ID_LIMIT" \
  --ood-test-rows "$S/formal_reasoning_phys_test_rows.jsonl" "$S/formal_reasoning_math_test_rows.jsonl" \
  --ood-test-row-limit "$OOD_LIMIT" \
  --conditions "$COND" \
  --out-dir "$OUT" \
  --profiles $PROFILES \
  --phase-a 2 --phase-b 2 \
  --runtime "$RT" --model "$MODEL" \
  --timeout-s "$TIMEOUT" --runtime-retries 1 --max-budget-usd 20.0 \
  --resume
