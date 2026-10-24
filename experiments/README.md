# Experiments

This harness runs TellOnce on ClawArena (8 learning and 16 test scenarios) and on the travel-planning tasks of MemoryArena (30 learning and 60 test tasks). Each run has three phases:

1. **train**: the solver works through the learning tasks. A simulated user corrects it from a fixed persona, and TellOnce learns rules, compiles checks and freezes the library.
2. **test**: the held-out tasks run with the frozen library. The original task inputs are used, and there is no feedback or learning.
3. **evaluate**: native task scoring, plus preference scoring of each test delivery.

## Run

```bash
git clone https://github.com/YujunZhou/tellonce.git
pip install -r experiments/requirements.txt
python experiments/run_trace.py all --skill tellonce \
    --config experiments/configs/clawarena-luna.json --output /tmp/tellonce-runs/clawarena-luna
```

- `--skill` (or `$TELLONCE_SKILL_DIR`) points to a checkout of [tellonce](https://github.com/YujunZhou/tellonce). The paper's runs used tag `v1.8.0`.

- Run the command with the Python interpreter named in the config (`spec.python`).
- `train`, `test` and `evaluate` can also be run one at a time.
- Rerunning the same command resumes an interrupted phase.
- The output directory must be outside any git repository.

Configs: `configs/{clawarena,memoryarena}-{luna,sol,haiku}.json`. The solvers are GPT-5.6 Luna (medium reasoning) and GPT-5.6 Sol (low reasoning) through the Codex CLI, and Claude Haiku 4.5 through Claude Code. The auxiliary roles all use `gpt-6-luna` through the OpenAI API, with low reasoning and medium for rule selection (`runtime.api_luna_roles`). These roles are the simulated user, correction interpretation, learning, compilation, selection, checking, appeal and preference scoring.

Results:

- `OUTPUT/test/test-phase/`: per-task outcomes.
- `OUTPUT/test/plan-a-summary-*.json` (written by `evaluate`): native task score and preference results per task and unit.
- `OUTPUT/train/method/`: the learned rule library.

## Setup

Replace every `/ABS/PATH/TO/...` entry in a config:

| Entry | What it is |
|---|---|
| `binary`, `auth_home` | Codex CLI 0.153.4 and its logged-in home directory. Haiku runs also need it for sandboxing. |
| `catalog` | `experiments/configs/codex-models-0.153.4.json` (included) |
| `claude_binary`, `claude_auth_file`, `claude_helpers` | Haiku configs only: the Claude Code CLI, its credentials file, and `socat` next to the binary |
| `api_key_file` | a `.env` file with `OPENAI_API_KEY` for the auxiliary roles |
| `runtime.api_luna_roles.slot_directory` | any writable directory; it limits concurrent API calls to 16 |
| `python`, `python_roots` | Python 3.10 environment with `requirements.txt` and the benchmark's grader dependencies |
| `tokenizer` | optional Gemma 4 tokenizer directory, used only to size requests for Gemma-routed roles |

ClawArena (`dataset_root`): clone `https://huggingface.co/datasets/Haonian/ClawArena` at revision `146bf9a952a9833fa949061aa93ec7eaa3a05ce0`.

MemoryArena travel planning:

- `official_source`: clone `https://github.com/ZexueHe/MemoryArena` at revision `6cd9de14b71915e39ac742a20dc33785e14b6aab`.
- `rows`: the `group_travel_planner` subset of the MemoryArena data as JSONL, one task per line. Its SHA-256 is pinned in the config.
- `database`: the TravelPlanner database, including `flights/clean_Flights_2022.csv` (see `setup_travel.md` in MemoryArena).
- `public_tool_source`: a directory that contains only the `travel_planner_env` tool files listed in `tellonce_memoryarena/tellonce_memoryarena/travel_tool_pins.json`, copied from MemoryArena.
- `tool_python`, `tool_runtime_roots`: a Python environment with `pandas` for the sandboxed travel tools.
- Build the flight index once:

  ```bash
  python experiments/tellonce_memoryarena/tellonce_memoryarena/travel_flight_index.py build --database /ABS/PATH/TO/travel-database
  ```

## Code map

| Part | Code |
|---|---|
| Entry point | `run_trace.py` (train), `tellonce_common/run_plan_a.py` (test and evaluate workers) |
| TellOnce inside a task: rule selection, injection, hook bridge, checks and repair requests | `tellonce_common/tellonce_method.py`, `tellonce_session.py`, `native_rule_guard.py`, `hook_bridge.py`, `claude_tool_hooks.py`, `rule_appeal.py` |
| Solver sessions | `claude_task_session.py`, `codex_task_session.py`, `interaction_loop.py` |
| Simulated user (persona, corrections, evidence rules) | `profile_data_clawarena.py`, `profile_data_memoryarena.py`, `may_feedback.py`, `feedback_*.py`, `preference_spans.py` |
| Preference scoring | `preference_judge.py`, `preference_transport.py` |
| Auxiliary model routing | `api_luna_roles.py`, `text_roles.py`, `http_text_*.py` |
| Benchmarks | `tellonce_clawarena/` (task loading, workspace, native grading), `tellonce_memoryarena/` (travel tasks, tool service, submission and grading) |

The simulated user is `user_a`. Its persona has four habits with conditions and exceptions: K1 to K4 for ClawArena and H1, H2, H3 and H6 for travel planning.
