"""Runtime adapters for the experiments ClawArena."""
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .schema import MULTI_CHOICE, Round, RuntimeResponse
from .scoring import extract_json_object


def call_runtime(
    runtime: str,
    prompt: str,
    *,
    model: str,
    session_dir: Path,
    working_dir: Path,
    allow_workspace_write: bool,
    timeout_s: int,
    runtime_retries: int,
    max_budget_usd: float,
    round_obj: Round | None = None,
    extra_env: dict[str, str] | None = None,
) -> RuntimeResponse:
    session_dir = session_dir.resolve()
    working_dir = working_dir.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    if runtime == "oracle_dry_run":
        return _oracle_response(round_obj)
    if runtime == "claude_cli":
        return _call_with_retries(
            lambda: _call_claude_cli(
                prompt,
                model=model,
                cwd=working_dir,
                allow_workspace_write=allow_workspace_write,
                timeout_s=timeout_s,
                max_budget_usd=max_budget_usd,
                extra_env=extra_env,
            ),
            runtime_retries=runtime_retries,
        )
    if runtime == "codex_cli":
        return _call_with_retries(
            lambda: _call_codex_cli(
                prompt,
                model=model,
                cwd=working_dir,
                session_dir=session_dir,
                allow_workspace_write=allow_workspace_write,
                timeout_s=timeout_s,
                extra_env=extra_env,
            ),
            runtime_retries=runtime_retries,
        )
    if runtime == "codex_cli_trace":
        # TRACE-native variant for trace_native_codex experiments.
        # Drops --ignore-user-config / --ignore-rules so codex loads
        # ~/.codex/hooks.json + ~/.codex/config.toml (preference-tracker
        # skill registration). Empirically codex 0.128.0 loads hooks.json
        # even with --ignore-user-config, but this variant removes the
        # ambiguity so the TRACE-native condition's runtime registration
        # is unambiguous in record provenance.
        return _call_with_retries(
            lambda: _call_codex_cli(
                prompt,
                model=model,
                cwd=working_dir,
                session_dir=session_dir,
                allow_workspace_write=allow_workspace_write,
                timeout_s=timeout_s,
                extra_env=extra_env,
                preserve_user_config=True,
            ),
            runtime_retries=runtime_retries,
        )
    raise KeyError(f"unknown runtime {runtime!r}")


def _oracle_response(round_obj: Round | None) -> RuntimeResponse:
    if round_obj and round_obj.round_type == MULTI_CHOICE:
        payload: dict[str, Any] = {
            "answer": round_obj.expected_answer,
            "rationale": "oracle_dry_run uses the known answer only for local harness validation.",
            "decision_log": "oracle_dry_run validated JSON and scoring plumbing.",
            "unresolved_questions": [],
        }
    else:
        payload = {
            "answer": "Task Completed",
            "rationale": "oracle_dry_run placeholder for unscored exec-check rounds.",
            "decision_log": "oracle_dry_run did not execute workspace actions.",
            "unresolved_questions": [],
        }
    return RuntimeResponse(parsed=payload, raw_text=json.dumps(payload), elapsed_s=0.0)


def _call_with_retries(fn, *, runtime_retries: int) -> RuntimeResponse:
    last: RuntimeResponse | None = None
    for attempt in range(runtime_retries + 1):
        last = fn()
        if not last.runtime_error:
            return last
        err = last.runtime_error
        if "error_max_budget_usd" in err or "budget" in err.lower():
            return last
        if attempt < runtime_retries:
            time.sleep(2)
    assert last is not None
    return last


def _diagnostic_excerpt(text: str, *, head: int = 2000, tail: int = 4000) -> str:
    if len(text) <= head + tail + 80:
        return text
    return text[:head] + "\n...[truncated middle]...\n" + text[-tail:]


def _call_claude_cli(
    prompt: str,
    *,
    model: str,
    cwd: Path,
    allow_workspace_write: bool,
    timeout_s: int,
    max_budget_usd: float,
    extra_env: dict[str, str] | None = None,
) -> RuntimeResponse:
    disallowed = "Skill,WebFetch,WebSearch"
    if not allow_workspace_write:
        disallowed = "Edit,Write,Bash,Skill,WebFetch,WebSearch"
    cmd = [
        "claude",
        "-p",
        "--no-session-persistence",
        "--setting-sources",
        "project,local",
        "--disable-slash-commands",
        "--disallowedTools",
        disallowed,
        "--model",
        model,
        "--output-format",
        "json",
        "--max-budget-usd",
        str(max_budget_usd),
        prompt,
    ]
    if allow_workspace_write:
        cmd[cmd.index("--model"):cmd.index("--model")] = ["--permission-mode", "bypassPermissions"]
    settings = (cwd / ".claude" / "settings.json").resolve()
    if settings.exists():
        cmd[cmd.index("--model"):cmd.index("--model")] = ["--settings", str(settings)]
    # strip_anthropic_api=True: force subscription auth in `claude -p`
    # subprocess; do NOT inherit ANTHROPIC_API_KEY etc from parent shell.
    env = _runtime_env(strip_anthropic_api=True)
    if extra_env:
        env.update(extra_env)
    return _run_process(cmd, cwd=cwd, timeout_s=timeout_s, parser=_parse_claude_stdout, env=env)


def _call_codex_cli(
    prompt: str,
    *,
    model: str,
    cwd: Path,
    session_dir: Path,
    allow_workspace_write: bool,
    timeout_s: int,
    extra_env: dict[str, str] | None = None,
    preserve_user_config: bool = False,
) -> RuntimeResponse:
    session_dir.mkdir(parents=True, exist_ok=True)
    out_file = session_dir / "codex_last_message.txt"
    # Default codex_cli runtime adds --ignore-user-config + --ignore-rules
    # and points CODEX_HOME at an isolated auth-only directory. Codex 0.128.0
    # still loads hooks.json from CODEX_HOME even with --ignore-user-config,
    # so clean baselines must not inherit the user's real ~/.codex directory.
    # The TRACE-native variant (preserve_user_config=True, runtime
    # "codex_cli_trace") keeps the user's CODEX_HOME so preference-tracker
    # hooks can fire by design.
    cmd = [
        "codex",
        "exec",
        "--ephemeral",
    ]
    if not preserve_user_config:
        cmd += ["--ignore-user-config", "--ignore-rules"]
    cmd += [
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write" if allow_workspace_write else "read-only",
        "--cd",
        str(cwd),
        "-m",
        model,
        "-c",
        'model_reasoning_effort="low"',
        "--output-last-message",
        str(out_file),
    ]
    started = time.time()
    # codex_cli_trace intentionally preserves the user's hook configuration.
    # Do not prepend .eval_bin in that mode: the preference-tracker hook reads
    # its JSON stdin with `cat`, and our output-limited cat wrapper truncates
    # large ClawArena prompts before the hook can parse them.
    codex_env = _runtime_env(None if preserve_user_config else cwd / ".eval_bin")
    if extra_env:
        codex_env.update(extra_env)
    # 2026-05-03 codex review fix (BLOCK): even trace_native_codex now uses
    # an isolated CODEX_HOME — but with a custom hooks.json registering ONLY
    # a heartbeat hook. This stops the user's global retrieve-inject hook
    # from leaking ~/.codex/skills/preference-tracker fingerprints + memory.
    isolated_codex_home: Path | None = None
    if preserve_user_config:
        isolated_codex_home = _prepare_trace_codex_home()
    else:
        isolated_codex_home = _prepare_isolated_codex_home()
    codex_env["CODEX_HOME"] = str(isolated_codex_home)
    try:
        returncode, stdout, stderr, process_error = _run_command(
            cmd,
            cwd=cwd,
            timeout_s=timeout_s,
            env=codex_env,
            input_text=prompt,
        )
    finally:
        if isolated_codex_home is not None:
            _cleanup_isolated_codex_home(isolated_codex_home)
    runtime_provenance = _codex_runtime_provenance(
        isolated=True,
        trace_variant=preserve_user_config,
    )
    if process_error:
        return RuntimeResponse(
            parsed=None,
            raw_text="",
            runtime_error=f"codex {process_error}",
            process_ok=False,
            parse_error=None,
            stdout=stdout[:20000],
            stderr=stderr[:20000],
            elapsed_s=time.time() - started,
            runtime_provenance=runtime_provenance,
        )
    raw_text = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else stdout
    parsed, parse_error = _parse_raw_json(raw_text)
    process_ok = returncode == 0
    err = None
    if returncode != 0:
        err = (
            f"codex rc={returncode} "
            f"stdout={_diagnostic_excerpt(stdout)} "
            f"stderr={_diagnostic_excerpt(stderr)}"
        )
    if parse_error:
        err = f"{err}; parse_error={parse_error}" if err else f"parse_error={parse_error}"
    return RuntimeResponse(
        parsed=parsed,
        raw_text=raw_text,
        runtime_error=err,
        process_ok=process_ok,
        parse_error=parse_error,
        stdout=stdout[:20000],
        stderr=stderr[:20000],
        elapsed_s=time.time() - started,
        runtime_provenance=runtime_provenance,
    )


def _run_process(cmd: list[str], *, cwd: Path, timeout_s: int, parser, env: dict[str, str] | None = None) -> RuntimeResponse:
    started = time.time()
    if env is None:
        env = _runtime_env()
    returncode, stdout, stderr, process_error = _run_command(
        cmd,
        cwd=cwd,
        timeout_s=timeout_s,
        env=env,
    )
    if process_error:
        return RuntimeResponse(
            parsed=None,
            raw_text="",
            runtime_error=process_error,
            process_ok=False,
            parse_error=None,
            stdout=stdout[:20000],
            stderr=stderr[:20000],
            elapsed_s=time.time() - started,
        )
    parsed, raw_text, parse_error = parser(stdout)
    process_ok = returncode == 0
    err = None
    if returncode != 0:
        err = (
            f"rc={returncode} "
            f"stdout={_diagnostic_excerpt(stdout)} "
            f"stderr={_diagnostic_excerpt(stderr)}"
        )
    if parse_error:
        err = f"{err}; parse_error={parse_error}" if err else f"parse_error={parse_error}"
    return RuntimeResponse(
        parsed=parsed,
        raw_text=raw_text,
        runtime_error=err,
        process_ok=process_ok,
        parse_error=parse_error,
        stdout=stdout[:20000],
        stderr=stderr[:20000],
        elapsed_s=time.time() - started,
    )


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_s: int,
    env: dict[str, str],
    input_text: str | None = None,
) -> tuple[int | None, str, str, str | None]:
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            text=True,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        return None, "", "", f"exec_error {type(exc).__name__}: {str(exc)[:500]}"

    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout_s)
        return proc.returncode, stdout or "", stderr or "", None
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = _coerce_timeout_text(exc.stdout), _coerce_timeout_text(exc.stderr)
        return None, stdout or "", stderr or "", f"timeout after {timeout_s}s"


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _coerce_timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _parse_claude_stdout(stdout: str) -> tuple[dict[str, Any] | None, str, str | None]:
    if not stdout.strip():
        return None, "", "empty stdout"
    raw_text = stdout
    try:
        envelope = json.loads(stdout)
        if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
            raw_text = envelope["result"]
        elif isinstance(envelope, dict) and "answer" in envelope:
            return envelope, stdout, None
    except json.JSONDecodeError:
        raw_text = stdout
    parsed, parse_error = _parse_raw_json(raw_text)
    return parsed, raw_text, parse_error


def _parse_raw_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return extract_json_object(text), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


def _runtime_env(
    prepend_path: Path | None = None,
    *,
    strip_anthropic_api: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    if prepend_path and prepend_path.is_dir():
        env["PATH"] = str(prepend_path) + os.pathsep + env.get("PATH", "")
        env.setdefault("EVAL_MAX_TOOL_BYTES", "2500")
    if strip_anthropic_api:
        # Defense-in-depth (2026-05-16 incident fix per
        # feedback-no-anthropic-api-only-subscription memory):
        # `claude -p` will prefer API-key auth over subscription auth if
        # ANTHROPIC_API_KEY is in env. The launch scripts unset these, but
        # systemd / cron / shell rc files could re-introduce them. This
        # belt-and-suspenders scrub ensures the subprocess NEVER sees an
        # Anthropic API key.
        for key in (
            "ANTHROPIC_API_KEY",
            "Anthropic_API_KEY",
            "anthropic_api_key",
            "CLAUDE_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
        ):
            env.pop(key, None)
    return env


def _codex_runtime_provenance(*, isolated: bool, trace_variant: bool = False) -> dict[str, Any]:
    """Return provenance metadata for codex subprocess.

    2026-05-04 fix: trace variant now uses isolated CODEX_HOME with
    heartbeat-only hooks (see _prepare_trace_codex_home), so its provenance
    must reflect that. Previously trace_variant returned isolated=False /
    user_codex_home which polluted paper records.
    """
    if trace_variant:
        return {
            "codex_home_isolated": True,
            "codex_hook_isolation_version": 2,
            "codex_auth_strategy": "temp_symlink_trace_heartbeat_only",
            "codex_runtime_variant": "codex_cli_trace",
        }
    return {
        "codex_home_isolated": isolated,
        "codex_hook_isolation_version": 2 if isolated else 0,
        "codex_auth_strategy": "temp_symlink" if isolated else "user_codex_home",
    }


def _prepare_isolated_codex_home() -> Path:
    """Create a temporary auth-only CODEX_HOME for hook-free baseline calls."""
    real_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    source_auth = real_home / "auth.json"
    parent = _codex_temp_parent()
    isolated = Path(tempfile.mkdtemp(prefix="eval_codex_home_", dir=str(parent)))
    os.chmod(isolated, 0o700)
    if not source_auth.is_file():
        return isolated
    dest_auth = isolated / "auth.json"
    os.symlink(source_auth, dest_auth)
    return isolated


def _prepare_trace_codex_home() -> Path:
    """Create a TRACE-isolated CODEX_HOME with auth + heartbeat-only hooks.

    2026-05-03 codex review fix (BLOCK): the previous trace_native_codex
    setup kept user CODEX_HOME so preference-tracker hooks could fire,
    but `~/.codex/hooks.json` registers `userpromptsubmit-retrieve-inject.sh`
    which loads `~/.codex/skills/preference-tracker/shared_lib/fingerprints.yaml`
    and the user's personal memory dir — leaking ~19 external rules into
    every trace_native_codex turn regardless of B5_MEMORY_DIR. Mirrors
    the claude-side Fix 3 (drop memory-retrieve-inject.sh): runner already
    injects memory_text via prompt body.

    This isolated CODEX_HOME contains:
      - auth.json (symlink to user's, preserves login)
      - hooks.json registering ONLY a heartbeat hook for diagnostic proof
        the codex hook chain loaded inside the isolated session
      - NO preference-tracker skill dir (so no fingerprints.yaml exists
        even if a hook script tried to load it)
    """
    real_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    source_auth = real_home / "auth.json"
    parent = _codex_temp_parent()
    isolated = Path(tempfile.mkdtemp(prefix="eval_codex_trace_home_", dir=str(parent)))
    os.chmod(isolated, 0o700)
    if source_auth.is_file():
        os.symlink(source_auth, isolated / "auth.json")
    # Write a minimal heartbeat hook script.
    heartbeat = isolated / "eval_trace_codex_heartbeat.py"
    heartbeat.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, time, sys\n"
        "from pathlib import Path\n"
        "# Drain stdin so codex hook protocol doesn't deadlock.\n"
        "try: _ = sys.stdin.read()\n"
        "except Exception: pass\n"
        "state = os.environ.get('B5_STATE_DIR')\n"
        "if state:\n"
        "    p = Path(state) / 'trace_hook_heartbeat.jsonl'\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    rec = {\n"
        "        'ts': time.time(),\n"
        "        'event': 'codex_UserPromptSubmit',\n"
        "        'env_b5_memory_dir': os.environ.get('B5_MEMORY_DIR'),\n"
        "    }\n"
        "    mem = rec['env_b5_memory_dir']\n"
        "    if mem and os.path.isdir(mem):\n"
        "        rec['memory_dir_md_count'] = len([f for f in os.listdir(mem) if f.endswith('.md')])\n"
        "    with p.open('a', encoding='utf-8') as f:\n"
        "        f.write(json.dumps(rec) + '\\n')\n",
        encoding="utf-8",
    )
    os.chmod(heartbeat, 0o755)
    hooks_payload = {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "",
                    "hooks": [
                        {"type": "command", "command": str(heartbeat), "timeout": 5},
                    ],
                }
            ],
        }
    }
    (isolated / "hooks.json").write_text(json.dumps(hooks_payload, indent=2), encoding="utf-8")
    return isolated


def _codex_temp_parent() -> Path:
    """Return a writable parent dir for codex isolated home temp dirs.

    2026-05-03 codex review fix: previously hard-required XDG_RUNTIME_DIR
    when set, but `/run/user/<uid>` can be a read-only filesystem in some
    container/sandboxed environments (e.g. when chmod or mkdtemp fails).
    Probe XDG first; if it can't be made 0o700-writable, fall back to
    `~/.cache/eval_codex_runtime`.
    """
    fallback = Path.home() / ".cache" / "eval_codex_runtime"
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        try:
            path = Path(xdg)
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
            # Verify writable by attempting to create a probe subdir.
            probe = path / ".eval_codex_probe"
            probe.mkdir(exist_ok=True)
            probe.rmdir()
            return path
        except (OSError, PermissionError):
            pass
    fallback.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(fallback, 0o700)
    return fallback


def _cleanup_isolated_codex_home(path: Path) -> None:
    auth = path / "auth.json"
    if auth.exists() or auth.is_symlink():
        auth.unlink()
    shutil.rmtree(path, ignore_errors=True)
