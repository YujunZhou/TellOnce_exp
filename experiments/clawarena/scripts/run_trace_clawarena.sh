#!/usr/bin/env bash
# Run TRACE (and the No-Memory control) on ClawArena with frozen ID/OOD eval.
#
# This public release ships only our method (TRACE) and the No-Memory control.
# The memory baselines compared in the paper (Mem0, Hindsight, ReMe-Light) are
# intentionally not included here; see ../../README.md for the rationale.
#
# Usage:
#   CLAWARENA_ROOT=/path/to/ClawArena \
#   ./run_trace_clawarena.sh <out_name> <runtime> <model> <condition>
#
#     runtime    : claude_cli | codex_cli | codex_cli_trace | oracle_dry_run
#     model      : haiku | sonnet (claude_cli); gpt-5.4-mini | gpt-5.5 (codex_cli)
#     condition  : trace_native_cc | trace_native_codex | no_memory | compiled_enforcement
#
# Prerequisites:
#   - For trace_native_cc / trace_native_codex: install the preference-tracker
#     skill at ~/.claude/skills/preference-tracker (see https://github.com/YujunZhou/preference-tracker).
#   - Download the ClawArena benchmark and point CLAWARENA_ROOT at it
#     (see ../data/README.md).
set -euo pipefail

OUT_NAME="${1:?usage: run_trace_clawarena.sh <out_name> <runtime> <model> <condition>}"
RUNTIME="${2:-claude_cli}"
MODEL="${3:-haiku}"
CONDITION="${4:-trace_native_cc}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# Optional local secrets (e.g. DEEPINFRA_API_KEY for the skill's lightweight LLM).
[[ -f .env ]] && { set -a; source .env; set +a; }
# Force subscription auth for claude_cli / codex_cli; never use an API key.
unset ANTHROPIC_API_KEY Anthropic_API_KEY anthropic_api_key 2>/dev/null || true

: "${CLAWARENA_ROOT:?set CLAWARENA_ROOT to your downloaded ClawArena directory}"

# 32 training scenarios across 4 families (c/d/e/f). In-distribution eval reuses
# held-out rounds of these same scenarios; out-of-distribution eval uses 30
# scenarios from unseen families (g/h/i/j/s).
TRAIN_SCENARIOS=(
  hil_c1 hil_c2 hil_c3 hil_c4 hil_c5 hil_c6 hil_c7 hil_c8
  hil_d1 hil_d2 hil_d3 hil_d4 hil_d5 hil_d6 hil_d7 hil_d8
  hil_e1 hil_e2 hil_e3 hil_e4 hil_e5 hil_e6 hil_e7 hil_e8
  hil_f1 hil_f2 hil_f3 hil_f4 hil_f5 hil_f6 hil_f7 hil_f8
)
OOD_SCENARIOS=(
  hil_g1 hil_g2 hil_g3 hil_g4 hil_g5 hil_g6 hil_g7
  hil_h1 hil_h2 hil_h3 hil_h5 hil_h7 hil_h8
  hil_i1 hil_i2 hil_i3 hil_i4 hil_i5 hil_i6 hil_i7 hil_i8
  hil_j1 hil_j2 hil_j3 hil_j4 hil_j5 hil_j6 hil_j7 hil_j8
  hil_s1
)

OUT_DIR="state/runtime/${OUT_NAME}"
mkdir -p "$OUT_DIR"

PYTHONPATH=. python3 -B -m clawarena.multi_runner \
  --clawarena-root "$CLAWARENA_ROOT" \
  --train-scenarios "${TRAIN_SCENARIOS[@]}" \
  --id-test-scenarios "${TRAIN_SCENARIOS[@]}" \
  --ood-test-scenarios "${OOD_SCENARIOS[@]}" \
  --profile p1 \
  --conditions "$CONDITION" \
  --runtime "$RUNTIME" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --train-round-limit 5 \
  --eval-round-limit 5 \
  --enforcement-retries 1 \
  --timeout-s 240 \
  --runtime-retries 1 \
  --max-budget-usd 5.0 \
  --resume

PYTHONPATH=. python3 -B -m clawarena.multi_summarize \
  "$OUT_DIR" \
  --out "$OUT_DIR/summary_metrics.json" \
  --curve-window 10
