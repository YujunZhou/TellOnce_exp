"""Run ClawArena exec_check commands against a condition workspace."""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .schema import Round
from .workspace import eval_dir


def run_exec_checks(
    *,
    clawarena_root: Path,
    scenario_id: str,
    workspace: Path,
    round_obj: Round,
) -> dict[str, Any] | None:
    if not round_obj.eval_spec.get("command"):
        return None
    workspace_abs = workspace.resolve()
    ctx = {
        "workspace": workspace_abs,
        "eval_dir": eval_dir(clawarena_root).resolve(),
        "test_id": scenario_id,
        "agent_id": scenario_id,
        "state_dir": (workspace_abs / ".eval_state").resolve(),
    }
    task = _run_command(round_obj.eval_spec, ctx)
    pref = _run_preference_checks(round_obj, ctx)
    return {
        "task": task,
        "pref": pref,
        "task_pass": bool(task.get("passed")),
        "pref_pass": None if pref is None else bool(pref.get("passed")),
        "combined_pass": bool(task.get("passed")) and (pref is None or bool(pref.get("passed"))),
    }


def _run_preference_checks(round_obj: Round, ctx: dict[str, Any]) -> dict[str, Any] | None:
    if round_obj.pref_spec.get("command"):
        result = _run_command(round_obj.pref_spec, ctx)
        result["source"] = "pref_command"
        return result
    commands = _extract_embedded_preference_commands(str(round_obj.eval_spec.get("command") or ""))
    checks = []
    for command in commands:
        checks.append(_run_command({"command": command, "timeout": round_obj.eval_spec.get("timeout", 30)}, ctx))
    if not checks:
        return None
    return {
        "source": "embedded_eval_check_preferences",
        "passed": all(check.get("passed") for check in checks),
        "checks": checks,
    }


def _run_command(spec: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    raw_command = str(spec.get("command") or "")
    if not raw_command:
        return {"passed": False, "exit_code": -1, "stdout": "", "stderr": "no command"}
    command = _expand_placeholders(raw_command, ctx)
    timeout = float(spec.get("timeout", 30))
    expect_exit = int(spec.get("expect_exit", 0))
    expect_stdout = spec.get("expect_stdout")
    expect_stdout_regex = bool(spec.get("expect_stdout_regex", False))
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx["workspace"]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        passed = proc.returncode == expect_exit
        if passed and expect_stdout is not None:
            if expect_stdout_regex:
                passed = bool(re.search(str(expect_stdout), proc.stdout))
            else:
                passed = str(expect_stdout) in proc.stdout
        return {
            "passed": passed,
            "exit_code": proc.returncode,
            "stdout_excerpt": proc.stdout[:2000],
            "stderr_excerpt": proc.stderr[:2000],
            "command_template": raw_command,
            "expanded_command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "exit_code": -1,
            "stdout_excerpt": (exc.stdout or "")[:2000] if isinstance(exc.stdout, str) else "",
            "stderr_excerpt": f"timeout after {timeout}s",
            "command_template": raw_command,
            "expanded_command": command,
        }
    except Exception as exc:
        return {
            "passed": False,
            "exit_code": -1,
            "stdout_excerpt": "",
            "stderr_excerpt": str(exc)[:2000],
            "command_template": raw_command,
            "expanded_command": command,
        }


def _expand_placeholders(template: str, ctx: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in ctx:
            return match.group(0)
        return shlex.quote(str(ctx[key]))

    return re.sub(r"\$\{([^}]+)\}", repl, template)


def _extract_embedded_preference_commands(command: str) -> list[str]:
    pattern = re.compile(
        r"python\s+\$\{eval_dir\}/\$\{agent_id\}/scripts/check_preferences\.py\b.*?(?=\s+&&\s+|$)"
    )
    commands = []
    for match in pattern.finditer(command):
        candidate = match.group(0).strip()
        # Some official eval commands compute shell variables or command
        # substitutions before invoking check_preferences. Those are not safe to
        # replay standalone; only our ${...} placeholders may remain.
        if re.search(r"\$(?!\{)|`", candidate):
            continue
        commands.append(candidate)
    return commands
