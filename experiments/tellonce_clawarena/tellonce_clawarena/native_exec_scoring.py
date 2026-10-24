"""Private ClawArena command checks on frozen files in a native OS sandbox.

No model, implicit retry, user feedback, or changes to the live task workspace.
The interaction controller owns any technical retry. All captured output is
private evaluation evidence, not material for the simulated user or learner.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
import uuid

from tellonce_common.interaction_loop import RecoverableCallError, SolverAnswer
from tellonce_common.native_execution import CommandCapture, run_native_command
from tellonce_common.native_isolation import codex_permission_arguments
from tellonce_common.provenance import content_hash, file_hash, tree_hash
from tellonce_common.request_budget import RequestBudget, RequestDeadline
from tellonce_common.role_inputs import RunIdentity
from tellonce_common.task_artifacts import TaskArtifacts


@dataclass(frozen=True)
class CommandSpec:
    command: str
    expect_exit: int = 0
    expect_stdout: str | None = None
    expect_stdout_regex: bool = False
    timeout: float = 30

    def __post_init__(self):
        if (not isinstance(self.command, str) or not self.command.strip() or '\x00' in self.command or
                len(self.command.encode()) > 65536 or type(self.expect_exit) is not int or
                not 0 <= self.expect_exit <= 255 or type(self.expect_stdout_regex) is not bool or
                (self.expect_stdout is not None and not isinstance(self.expect_stdout, str)) or
                type(self.timeout) not in {int, float} or not math.isfinite(self.timeout) or not 0 < self.timeout <= 300):
            raise ValueError('bounded original native command and score conditions required')
        if self.expect_stdout_regex and self.expect_stdout is not None:
            re.compile(self.expect_stdout)

    @classmethod
    def from_native(cls, value):
        if not isinstance(value, dict) or 'command' not in value:
            raise ValueError('missing native evaluation command')
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})

    def expand(self, values):
        # The native specification only expands known ${...} placeholders.
        # Shell variables and command substitutions remain original shell code.
        return re.sub(r'\$\{([^}]+)\}', lambda m: shlex.quote(str(values[m[1]]))
                      if m[1] in values else m[0], self.command)

    def passed(self, exit_code, stdout):
        if exit_code != self.expect_exit:
            return False
        if self.expect_stdout is None:
            return True
        if self.expect_stdout_regex:
            return re.search(self.expect_stdout, stdout) is not None
        return self.expect_stdout in stdout


@dataclass(frozen=True)
class CommandGrade:
    passed: bool
    exit_code: int
    artifact_hash: str
    command_hash: str
    wall_s: float


class NativeCommandGrader:
    def __init__(self, *, identity: RunIdentity, start: dict, binary: Path, python: Path,
                 python_roots: tuple[Path, ...], eval_root: Path, private_root: Path,
                 artifacts: TaskArtifacts, baseline: dict, authorize, record):
        paths = (binary, python, eval_root, private_root, *python_roots)
        if (not isinstance(identity, RunIdentity) or identity.benchmark != 'clawarena' or
                not isinstance(start, dict) or start.get('namespace') != identity.namespace or
                any(not isinstance(start.get(k), str) or not start[k] for k in ('task_id', 'session_id')) or
                not re.fullmatch('[a-z0-9_]{1,80}', start['task_id']) or type(start.get('training')) is not bool or
                not python_roots or any(not isinstance(p, Path) or not p.is_absolute() or p != p.resolve() for p in paths) or
                private_root.exists() or not isinstance(artifacts, TaskArtifacts) or
                private_root == artifacts.workspace or artifacts.workspace in private_root.parents or
                not callable(authorize) or not callable(record)):
            raise ValueError('task-owned native grader, pinned runtime and isolated private paths required')
        artifacts.validate_baseline(baseline)
        if not any(python.is_relative_to(root) for root in python_roots):
            raise ValueError('the admitted interpreter must be inside its declared read-only runtime')
        if any((p/'.git/HEAD').is_file() or (p/'.git').is_file() for p in (private_root, *private_root.parents)):
            raise ValueError('private native grading files must be outside source repositories')
        self.identity, self.start = identity, dict(start)
        self.binary, self.python, self.python_roots = binary, python, python_roots
        self.eval_root, self.private_root = eval_root, private_root
        self.artifacts = artifacts
        self.baseline = {p: dict(v) for p, v in baseline.items()}
        self.authorize, self.record = authorize, record
        self._scenario = eval_root / start['task_id']
        self._executor = Path(__file__).resolve().parents[2] / 'tellonce_common/tellonce_common/native_exec_worker.py'
        self._sources = [Path(__file__).resolve(), self._executor, *[
            self._executor.with_name(n) for n in ('native_execution.py', 'native_isolation.py', 'process_control.py',
                                                'request_budget.py', 'task_artifacts.py', 'provenance.py')]]
        self._binding = {'binary': file_hash(binary), 'python': file_hash(python),
            'scenario': tree_hash(self._scenario), 'sources': {str(p): file_hash(p) for p in self._sources}}
        authorize('prepare', {'namespace': identity.namespace, 'binding': self._binding})
        private_root.mkdir(mode=0o700)

    def check(self, spec: CommandSpec, answer: SolverAnswer, *, context: dict, kind='task') -> CommandGrade:
        fields = {'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (not isinstance(spec, CommandSpec) or not isinstance(answer, SolverAnswer) or kind not in {'task', 'native_pref'} or
                not isinstance(context, dict) or set(context) != fields or context.get('task_id') != self.start['task_id'] or
                context.get('session_id') != self.start['session_id'] or context.get('phase') !=
                    ('training' if self.start['training'] else 'test') or
                context.get('role') not in {'score', 'score_before_repair'} or
                type(context.get('attempt')) is not int or context['attempt'] not in {1,2} or
                any(not isinstance(context.get(k), str) or not 1 <= len(context[k]) <= 480 for k in fields-{'attempt'})):
            raise ValueError('private native scoring must match the actual delivery and controller context')
        context = dict(context)
        if (file_hash(self.binary) != self._binding['binary'] or file_hash(self.python) != self._binding['python'] or
                tree_hash(self._scenario) != self._binding['scenario'] or
                any(file_hash(p) != self._binding['sources'][str(p)] for p in self._sources)):
            raise ValueError('native grader source changed after preparation')
        snapshot = self.artifacts.load(answer.artifact_hash)
        if snapshot['answer'] != answer.text:
            raise ValueError('native score must use the actual text and artifact from one delivery')
        self.authorize('check', {'namespace': self.identity.namespace, 'binding': self._binding,
                               'artifact_hash': answer.artifact_hash, 'kind': kind, **context})
        runtime = self.private_root / uuid.uuid4().hex
        runtime.mkdir(mode=0o700)
        workspace, home, state = runtime/'work', runtime/'cli', runtime/'state'
        for directory in (home, state, runtime/'empty-home'):
            directory.mkdir(mode=0o700)
        started = time.monotonic()
        try:
            self.artifacts.restore(answer.artifact_hash, baseline=self.baseline, destination=workspace)
        except (ValueError, OSError) as exc:
            incomplete = isinstance(exc, ValueError)
            self.record({'namespace': self.identity.namespace, 'context': context, 'kind': kind,
                'artifact_hash': answer.artifact_hash, 'type': 'native_grade_unavailable',
                'status': 'incomplete_artifact' if incomplete else 'restore_io_failed',
                'wall_s': time.monotonic() - started})
            raise RecoverableCallError('native_grade_incomplete_artifact' if incomplete else
                                       'native_grade_restore_failed', retryable=not incomplete) from None
        command = spec.expand({'workspace': workspace, 'eval_dir': self.eval_root,
            'agent_id': self.start['task_id'], 'test_id': self.start['task_id'], 'state_dir': state})
        # The original data uses `python`. Bind that name to the exact admitted
        # interpreter without importing the operator's shell or Python config.
        shim = runtime/'bin'
        shim.mkdir(mode=0o700)
        for name in ('python', 'python3'):
            (shim/name).symlink_to(self.python)
        permissions = codex_permission_arguments(workspace=workspace, cli_home=home,
            runtime_roots=(self.binary.parent, *self.python_roots, shim), readonly_assets=(self.eval_root,),
            writable_state=(state,))
        marker = 'NATIVE_GRADER_STARTED_' + uuid.uuid4().hex
        execution_script = "printf '%s\\n' " + shlex.quote(marker) + ' >&2\n' + command
        native = [str(self.binary), 'sandbox', '-C', str(workspace), '-P', 'experiment', *permissions,
                  '/bin/bash', '--noprofile', '--norc', '-c', execution_script]
        wrapped = [str(Path(sys.executable).resolve()), '-S', '-B', str(self._executor),
                   '--owner-pid', str(os.getpid()), '--', *native]
        environment = {'PATH': str(shim) + ':/usr/bin:/bin', 'HOME': str(runtime/'empty-home'),
            'CODEX_HOME': str(home), 'LANG': 'C.UTF-8', 'TZ': 'UTC', 'PYTHONUTF8': '1',
            'PYTHONIOENCODING': 'utf-8', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0'}
        # The original 30/60s command budget includes native sandbox startup.
        try:
            budget = RequestBudget(spec.timeout)
            result = run_native_command(wrapped, cwd=workspace, env=environment, prompt='',
                timeout_s=budget.remaining(), on_event=None, stdout_budget=1024*1024, stderr_budget=1024*1024)
        except RequestDeadline:
            result = CommandCapture('', '', -1, 'request_deadline')
        except OSError:
            # Failure to launch a process is unavailable evidence, never a
            # native incorrect-answer verdict. Do not expose operator paths.
            result = CommandCapture('', '', -1, 'launch_failed')
        wall_s = time.monotonic() - started
        command_started = marker in result.stderr.splitlines()
        stderr = '\n'.join(line for line in result.stderr.splitlines() if line != marker)
        private = {'namespace': self.identity.namespace, 'context': context, 'kind': kind,
            'artifact_hash': answer.artifact_hash, 'spec': spec.__dict__, 'expanded_command': command,
            'status': result.status, 'exit_code': result.exit_code, 'stdout': result.stdout,
            'stderr': stderr, 'command_started': command_started, 'wall_s': wall_s}
        (runtime/'grade.json').write_text(json.dumps(private, ensure_ascii=False, allow_nan=False))
        if (not command_started or result.status != 'completed' or result.exit_code < 0 or
                (result.exit_code in {126,127} and result.exit_code != spec.expect_exit)):
            self.record({**private, 'type': 'native_grade_unavailable'})
            raise RecoverableCallError('native_grade_execution_unavailable', retryable=result.status!='request_deadline')
        passed = spec.passed(result.exit_code, result.stdout)
        self.record({**private, 'type': 'native_grade_completed', 'passed': passed})
        return CommandGrade(passed, result.exit_code, answer.artifact_hash, content_hash(spec.__dict__), wall_s)
