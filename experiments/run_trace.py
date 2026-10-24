#!/usr/bin/env python3
"""Run TellOnce on ClawArena or MemoryArena travel planning.

    python experiments/run_trace.py all --config CONFIG.json --output OUTPUT_DIR --skill TELLONCE_DIR

Phases (each can be run on its own and resumed after an interruption):

  train     the simulated user corrects the agent on the learning tasks; TellOnce
            learns rules, compiles checks, and freezes the library at the end
  test      held-out tasks run with the frozen library; no feedback, no learning
  evaluate  task scoring and preference scoring of the test deliveries

TELLONCE_DIR is a checkout of https://github.com/YujunZhou/tellonce (the method).
OUTPUT_DIR must be outside any git repository. The command must be run with the
Python interpreter named in the config (`spec.python`).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

EXPERIMENTS = Path(__file__).resolve().parent
PACKAGES = ('tellonce_common', 'tellonce_clawarena', 'tellonce_memoryarena')
SOURCE_KIND = {'clawarena': 'clawarena', 'memoryarena': 'memoryarena_travel'}


def setup(benchmark):
    if benchmark not in SOURCE_KIND:
        raise SystemExit('config "benchmark" must be clawarena or memoryarena')
    os.environ['TELLONCE_EXP_BENCHMARK'] = benchmark
    sys.path[:0] = [str(EXPERIMENTS/name) for name in PACKAGES]


def loader(benchmark):
    if benchmark == 'clawarena':
        from tellonce_clawarena.native_run_spec import ClawRunSpec
        from tellonce_clawarena.native_worker import run_phase, verify_execution
        return ClawRunSpec, run_phase, verify_execution
    from tellonce_memoryarena.native_travel_run_spec import TravelRunSpec
    from tellonce_memoryarena.native_travel_worker import run_phase, verify_execution
    return TravelRunSpec, run_phase, verify_execution


def write_once(path, value):
    from tellonce_common.deferred_scoring import _write
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise SystemExit(f'{path} already exists with different content; use a new output directory')
    else:
        _write(path, value)


def training_spec(config, output):
    value = dict(config['spec'])
    value['directory'] = str(output/'train')
    path = output/'train-spec.json'
    write_once(path, value)
    return path


def train(config, output):
    from tellonce_common.phase_signals import PhaseSignals
    from tellonce_common.plan_a_spec import frozen_training
    from tellonce_common.provenance import canonical_json_bytes
    from tellonce_common.run_lock import run_lock
    Spec, run_phase, verify_execution = loader(config['benchmark'])
    spec = Spec.load(path=training_spec(config, output))
    verify_execution(spec)
    spec.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (spec.directory/'training-phase/phase-result.json').exists():
        print(json.dumps({'phase': 'train', 'status': 'already_complete'}))
        return
    expected = {'schema': 1, 'namespace': spec.identity.namespace, 'binding': spec.binding}
    with run_lock(spec):
        owner = spec.directory/'run.json'
        if owner.exists():
            if json.loads(owner.read_text()) != expected:
                raise SystemExit('training directory belongs to another run')
        elif any(p.name != 'scheduler.lock' for p in spec.directory.iterdir()):
            raise SystemExit('new training must start with an empty directory')
        else:
            from tellonce_common.deferred_scoring import _write
            _write(owner, expected)
        attempts = spec.directory/'worker-attempts'
        attempts.mkdir(mode=0o700, exist_ok=True)
        events = attempts/(uuid.uuid4().hex+'.jsonl')
        lock = threading.Lock()
        signals = PhaseSignals()
        with events.open('xb') as stream:
            def record(event):
                with lock:
                    signals.observe(event)
                    stream.write(canonical_json_bytes(event)+b'\n')
                    stream.flush()
            result = run_phase(spec, action='training_interaction', prior_receipt=None,
                               signals=signals, record=record, attempt_directory=events.with_suffix(''))
        signals.assert_healthy()
        frozen_training(spec, result['phase_receipt'])
    print(json.dumps({'phase': 'train', 'status': 'complete', 'directory': str(spec.directory)}))


def test_spec(config, output):
    from tellonce_common.plan_a_prepare import prepare_extension
    return prepare_extension(source_kind=SOURCE_KIND[config['benchmark']], source_path=output/'train-spec.json',
                             destination=output, result_directory=output/'test',
                             test_task_workers=config.get('test_task_workers', 4))


def run_test_worker(config, output, action):
    spec = test_spec(config, output)
    command = [sys.executable, str(EXPERIMENTS/'tellonce_common/run_plan_a.py'), action,
               '--spec', str(spec.path), '--digest', spec.spec_digest]
    code = subprocess.run(command, env=os.environ).returncode
    if code:
        raise SystemExit(f'{action} stopped with exit code {code}; rerun the same command to resume')
    print(json.dumps({'phase': action, 'status': 'complete', 'directory': str(spec.directory)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('phase', choices=('train', 'test', 'evaluate', 'all'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skill', type=Path, default=os.environ.get('TELLONCE_SKILL_DIR'),
                        help='tellonce checkout (default: $TELLONCE_SKILL_DIR)')
    args = parser.parse_args()
    if args.skill is None:
        parser.error('--skill (or $TELLONCE_SKILL_DIR) is required')
    os.environ['TELLONCE_SKILL_DIR'] = str(args.skill.expanduser().resolve())
    config = json.loads(args.config.read_text())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    setup(config['benchmark'])
    if config['benchmark'] == 'memoryarena':
        os.environ['TELLONCE_MEMORYARENA_SOURCE'] = config['spec']['official_source']
    if args.phase in ('train', 'all'):
        train(config, output)
    if args.phase in ('test', 'all'):
        run_test_worker(config, output, 'test')
    if args.phase in ('evaluate', 'all'):
        run_test_worker(config, output, 'evaluate')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
