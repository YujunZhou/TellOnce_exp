# TellOnce experiments

Experiment code for the paper *TellOnce: Correcting Coding Agents Once, Improving Their Future Actions*.

TellOnce turns a user's corrections into scoped rules and runtime checks. A background learner decides whether a correction is a lasting rule and records it with its scope, conditions and exceptions. It then reconciles the rule with the existing library and compiles it into a check specification. In later tasks, relevant rules are selected for the agent. The execution layer checks the agent's actions and deliveries against them and asks for a revision when a check fails.

![TellOnce overview](assets/overview.png)

**The method itself lives in [YujunZhou/tellonce](https://github.com/YujunZhou/tellonce)**: an installable skill for Claude Code, Codex and Copilot CLI. This repository holds only the experiment harness. The harness loads the learning, compilation, selection and checking code from a tellonce checkout.

## Reproduce

```bash
git clone https://github.com/YujunZhou/tellonce.git
pip install -r experiments/requirements.txt
python experiments/run_trace.py all --skill tellonce \
    --config experiments/configs/clawarena-luna.json --output /tmp/tellonce-runs/clawarena-luna
```

- Fill in the `/ABS/PATH/TO/...` entries of a config first.
- Benchmark data, credentials and the code map are in [experiments/README.md](experiments/README.md).
- The paper's runs used tellonce v1.8.0 (`git -C tellonce checkout v1.8.0` reproduces them exactly). Versions 1.8.1 and 1.8.2 change only the installed skill's defaults: the background model call, the per-turn injection cap and the skill description. They do not change the library interfaces this harness uses.

## License

MIT. See [LICENSE](LICENSE).
