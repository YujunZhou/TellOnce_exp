from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import subprocess
import re
import shutil
import tempfile
import time


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        return _restore_latex_control_escapes(json.loads(text))
    except json.JSONDecodeError:
        pass
    candidates = list(_json_object_candidates(text))
    for candidate in candidates:
        if _is_artifact_object(candidate):
            return _restore_latex_control_escapes(candidate)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in output: {text[:200]}")
    blob = text[start : end + 1]
    try:
        return _restore_latex_control_escapes(json.loads(blob))
    except json.JSONDecodeError:
        blob = _replace_braced_unicode_escapes(blob)
        # Models sometimes emit single LaTeX/path backslashes such as \(,
        # while already escaped JSON sequences like \\[ must stay untouched.
        repaired = re.sub(r"(?<!\\)\\(?![\"\\/u])", r"\\\\", blob)
        repaired = _escape_control_chars_inside_strings(repaired)
        try:
            return _restore_latex_control_escapes(json.loads(repaired))
        except json.JSONDecodeError:
            from json_repair import repair_json

            repaired_obj = repair_json(repaired, return_objects=True)
            if not isinstance(repaired_obj, dict):
                raise ValueError("repaired model output was not a JSON object")
            return _restore_latex_control_escapes(repaired_obj)


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def _is_artifact_object(obj: dict[str, Any]) -> bool:
    required = {"answer", "constraint_notes", "unresolved_questions", "decision_log"}
    return required.issubset(obj)


def _parse_claude_stdout_envelope(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        raise ValueError("empty claude stdout")
    try:
        candidates = [json.loads(text)]
    except json.JSONDecodeError:
        candidates = _json_object_candidates(text)
    for envelope in candidates:
        if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
            return _extract_json_object(envelope["result"])
    for envelope in candidates:
        if isinstance(envelope, dict) and _is_artifact_object(envelope):
            return _restore_latex_control_escapes(envelope)
    return _extract_json_object(text)


def _replace_braced_unicode_escapes(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)

    return re.sub(r"\\u\{([0-9a-fA-F]{1,6})\}", repl, text)


def _escape_control_chars_inside_strings(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            continue
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = False
            continue
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def _restore_latex_control_escapes(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace("\b", "\\b")
            .replace("\f", "\\f")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
    if isinstance(value, list):
        return [_restore_latex_control_escapes(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_latex_control_escapes(item) for key, item in value.items()}
    return value


def build_public_prompt(
    task_spec: dict[str, Any],
    project_state: dict[str, Any],
    memory_text: str = "",
) -> str:
    memory_block = f"\nLearned user memory:\n{memory_text}\n" if memory_text.strip() else ""
    subset = task_spec.get("metadata", {}).get("subset", "group_travel_planner")
    artifact_kind = task_spec.get("metadata", {}).get("artifact_kind", "task_artifact")
    return (
        f"You are updating a structured MemoryArena artifact for subset `{subset}` "
        f"and artifact kind `{artifact_kind}`.\n"
        "Use only the public task spec and public project state below. Do not invent hidden facts.\n"
        "Do not quote poems, lyrics, articles, or other copyrighted passages; report only concise factual identifiers and decisions.\n"
        "Return exactly one JSON object with keys: answer, constraint_notes, unresolved_questions, decision_log.\n"
        "Do not wrap the JSON in markdown.\n\n"
        f"Task spec:\n{json.dumps(task_spec, ensure_ascii=False, indent=2)}\n\n"
        f"Project state:\n{json.dumps(project_state, ensure_ascii=False, indent=2)}\n"
        f"{memory_block}"
    )


def resolve_anthropic_model_alias(model: str) -> str:
    aliases = {
        "haiku": "claude-haiku-4-5-20251001",
        "claude-haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-6",
        "claude-sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-7",
        "claude-opus": "claude-opus-4-7",
    }
    return aliases.get(model, model)


def extract_anthropic_artifact(message: Any) -> dict[str, Any]:
    text_parts: list[str] = []
    for block in getattr(message, "content", []):
        block_type = getattr(block, "type", None)
        if block_type == "tool_use" and getattr(block, "name", None) == "submit_artifact":
            artifact = getattr(block, "input", None)
            if isinstance(artifact, dict):
                return artifact
        if block_type == "text" and getattr(block, "text", None):
            text_parts.append(block.text)
    if not text_parts:
        raise ValueError("anthropic response contained no artifact tool call or text content")
    return _extract_json_object("\n".join(text_parts))


def extract_openai_artifact(response: Any) -> dict[str, Any]:
    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        text_parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if isinstance(value, str):
                    text_parts.append(value)
        text = "\n".join(text_parts)
    artifact = _extract_json_object(text)
    if not _is_artifact_object(artifact):
        missing = sorted(
            {"answer", "constraint_notes", "unresolved_questions", "decision_log"}
            - set(artifact)
        )
        raise ValueError(f"openai response missing artifact keys: {missing}")
    return artifact


def openai_artifact_text_config() -> dict[str, Any]:
    trip_day_schema = {
        "type": "object",
        "properties": {
            "days": {"type": ["integer", "number", "string"]},
            "current_city": {"type": "string"},
            "transportation": {"type": "string"},
            "breakfast": {"type": "string"},
            "attraction": {"type": "string"},
            "lunch": {"type": "string"},
            "dinner": {"type": "string"},
            "accommodation": {"type": "string"},
        },
        "required": [
            "days",
            "current_city",
            "transportation",
            "breakfast",
            "attraction",
            "lunch",
            "dinner",
            "accommodation",
        ],
        "additionalProperties": False,
    }
    return {
        "format": {
            "type": "json_schema",
            "name": "memoryarena_artifact",
            "description": "Structured MemoryArena artifact returned by the agent.",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "description": "The task answer artifact. Shape may vary by MemoryArena subset.",
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "target_asin": {"type": "string"},
                                    "attributes": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["target_asin", "attributes"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "array",
                                "items": trip_day_schema,
                            },
                            {"type": "null"},
                        ],
                    },
                    "constraint_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "unresolved_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "decision_log": {
                        "type": "string",
                        "description": "Concise explanation of the decision process.",
                    },
                },
                "required": [
                    "answer",
                    "constraint_notes",
                    "unresolved_questions",
                    "decision_log",
                ],
                "additionalProperties": False,
            },
        }
    }


def _is_openai_transient_error(exc: Exception) -> bool:
    if "Request too large" in str(exc):
        return False
    name = type(exc).__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "try again in",
            "connection reset",
            "temporarily unavailable",
            "timeout",
            "timed out",
        )
    )


def openai_retry_delay_s(exc: Exception, attempt: int = 0) -> float | None:
    message = str(exc)
    if "Request too large" in message:
        return None
    ms_match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)ms", message, re.I)
    if ms_match:
        return max(1.0, float(ms_match.group(1)) / 1000.0 + 0.5)
    sec_match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", message, re.I)
    if sec_match:
        return max(1.0, float(sec_match.group(1)) + 0.5)
    if not _is_openai_transient_error(exc):
        return None
    return min(60.0, 10.0 * (attempt + 1))


def call_anthropic_api(
    prompt: str,
    *,
    timeout_s: int = 120,
    model: str = "haiku",
    runtime_retries: int = 1,
) -> dict[str, Any]:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is not installed") from exc
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("Anthropic_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    model_id = resolve_anthropic_model_alias(model)
    client = Anthropic(api_key=api_key, timeout=timeout_s)
    last_error: Exception | None = None
    for attempt in range(runtime_retries + 1):
        try:
            message = client.messages.create(
                model=model_id,
                max_tokens=2048,
                # User 2026-05-03 directive: when this Anthropic SDK API path
                # is ever used (currently disabled in favor of claude_cli
                # subscription path), keep agent rollout temperature=1.0 to
                # match Claude CLI default sampling. Cross-runtime parity for
                # paper main matrix: Claude CLI (default ~1.0), Codex CLI
                # reasoning_effort=low, Anthropic SDK 1.0 (this line).
                temperature=1.0,
                tools=[
                    {
                        "name": "submit_artifact",
                        "description": "Submit the MemoryArena artifact as structured data.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "answer": {},
                                "constraint_notes": {
                                    "type": "array",
                                    "items": {},
                                },
                                "unresolved_questions": {
                                    "type": "array",
                                    "items": {},
                                },
                                "decision_log": {},
                            },
                            "required": [
                                "answer",
                                "constraint_notes",
                                "unresolved_questions",
                                "decision_log",
                            ],
                            "additionalProperties": True,
                        },
                    }
                ],
                tool_choice={"type": "tool", "name": "submit_artifact"},
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_anthropic_artifact(message)
        except Exception as exc:
            last_error = exc
            if attempt < runtime_retries:
                time.sleep(2)
                continue
            raise
    raise RuntimeError(f"anthropic failed without response: {last_error}")


def call_openai_api(
    prompt: str,
    *,
    timeout_s: int = 120,
    model: str = "gpt-5.4-mini",
    runtime_retries: int = 1,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed") from exc
    client = OpenAI(timeout=timeout_s)
    return call_openai_api_with_client(
        client,
        prompt,
        model=model,
        runtime_retries=runtime_retries,
        sleep_fn=time.sleep,
    )


def call_openai_api_with_client(
    client: Any,
    prompt: str,
    *,
    model: str = "gpt-5.4-mini",
    runtime_retries: int = 1,
    sleep_fn: Any = time.sleep,
) -> dict[str, Any]:
    try:
        max_output_tokens = int(os.environ.get("EVAL_OPENAI_MAX_OUTPUT_TOKENS", "4096"))
    except ValueError as exc:
        raise ValueError("EVAL_OPENAI_MAX_OUTPUT_TOKENS must be an integer") from exc
    last_error: Exception | None = None
    for attempt in range(runtime_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": "low"},
                text=openai_artifact_text_config(),
            )
            artifact = extract_openai_artifact(response)
            artifact["_runtime_provenance"] = {
                "api_model": model,
                "openai_max_output_tokens": max_output_tokens,
                "openai_structured_outputs_version": 3,
            }
            return artifact
        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
            delay_s = openai_retry_delay_s(exc, attempt)
            if delay_s is None:
                raise
            if attempt < runtime_retries:
                sleep_fn(delay_s)
                continue
            raise
    raise RuntimeError(f"openai failed without response: {last_error}")


def call_claude_haiku(
    prompt: str,
    *,
    cwd: Path,
    timeout_s: int = 120,
    model: str = "haiku",
    max_budget_usd: float = 0.5,
    runtime_retries: int = 1,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    cmd = [
        "claude",
        "-p",
        "--no-session-persistence",
        "--setting-sources",
        "project,local",
        "--disable-slash-commands",
        "--disallowedTools",
        "Edit,Write,Bash,Skill,WebFetch,WebSearch",
        "--model",
        model,
        "--output-format",
        "json",
        "--max-budget-usd",
        str(max_budget_usd),
        prompt,
    ]
    settings = (cwd / ".claude" / "settings.json").resolve()
    if settings.is_file():
        cmd[cmd.index("--model"):cmd.index("--model")] = ["--settings", str(settings)]
    proc_env = os.environ.copy()
    if extra_env:
        proc_env.update(extra_env)
    last_error: Exception | None = None
    for attempt in range(runtime_retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                env=proc_env,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            if attempt < runtime_retries:
                time.sleep(2)
                continue
            raise RuntimeError(
                f"claude timeout after {timeout_s}s on attempt {attempt + 1}/"
                f"{runtime_retries + 1}"
            ) from exc
        if proc.returncode == 0:
            break
        stdout = proc.stdout[:500]
        stderr = proc.stderr[:500]
        if "error_max_budget_usd" in stdout or "error_max_budget_usd" in stderr:
            raise RuntimeError(f"claude rc={proc.returncode} stdout={stdout} stderr={stderr}")
        last_error = RuntimeError(f"claude rc={proc.returncode} stdout={stdout} stderr={stderr}")
        if attempt < runtime_retries:
            time.sleep(2)
            continue
        raise last_error
    else:
        raise RuntimeError(f"claude failed without process result: {last_error}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude rc={proc.returncode} stdout={proc.stdout[:500]} stderr={proc.stderr[:500]}"
        )
    return _parse_claude_stdout_envelope(proc.stdout)


def call_codex_cli(
    prompt: str,
    *,
    cwd: Path,
    timeout_s: int = 120,
    model: str = "gpt-5.4-mini",
    runtime_retries: int = 1,
    extra_env: dict[str, str] | None = None,
    preserve_user_config: bool = False,
) -> dict[str, Any]:
    """Call codex_cli.

    Default mode (preserve_user_config=False): hook-isolated CODEX_HOME with
    only auth.json symlinked, plus --ignore-user-config / --ignore-rules. This
    is the clean baseline used by the no_memory control where TRACE hooks
    must NOT fire.

    TRACE-native mode (preserve_user_config=True): keep the user's real
    ~/.codex/hooks.json visible by passing the user's CODEX_HOME and dropping
    --ignore-user-config / --ignore-rules. Mirrors ClawArena's
    codex_cli_trace runtime variant. Use only with the trace_native_codex
    condition.
    """
    cwd = cwd.resolve()
    out_file = cwd / "runtime" / "codex_last_message.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    codex_env = os.environ.copy()
    codex_env.setdefault("NO_COLOR", "1")
    # 2026-05-03 codex review fix (BLOCK): trace_native_codex now uses an
    # isolated CODEX_HOME with custom heartbeat-only hooks.json instead of
    # user CODEX_HOME, to prevent the global retrieve-inject hook from
    # leaking ~/.codex/skills/tellonce fingerprints + memory.
    codex_home: Path | None = None
    if preserve_user_config:
        codex_home = _prepare_trace_codex_home()
    else:
        codex_home = _prepare_isolated_codex_home()
    codex_env["CODEX_HOME"] = str(codex_home)
    if extra_env:
        codex_env.update(extra_env)
    cmd: list[str] = [
        "codex",
        "exec",
        "--ephemeral",
    ]
    if not preserve_user_config:
        cmd += ["--ignore-user-config", "--ignore-rules"]
    cmd += [
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(cwd),
        "-m",
        model,
        "-c",
        'model_reasoning_effort="low"',
        "--output-last-message",
        str(out_file),
    ]
    last_error: Exception | None = None
    try:
        for attempt in range(runtime_retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    input=prompt,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=codex_env,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                last_error = exc
                if attempt < runtime_retries:
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"codex timeout after {timeout_s}s on attempt {attempt + 1}/"
                    f"{runtime_retries + 1}"
                ) from exc
            if proc.returncode == 0:
                raw_text = (
                    out_file.read_text(encoding="utf-8", errors="replace")
                    if out_file.exists()
                    else proc.stdout
                )
                artifact = _extract_json_object(raw_text)
                artifact["_runtime_provenance"] = _codex_runtime_provenance(
                    preserve_user_config=preserve_user_config
                )
                return artifact
            stdout = proc.stdout[:1000]
            stderr = proc.stderr[:1000]
            last_error = RuntimeError(f"codex rc={proc.returncode} stdout={stdout} stderr={stderr}")
            if attempt < runtime_retries:
                time.sleep(2)
                continue
            raise last_error
        raise RuntimeError(f"codex failed without process result: {last_error}")
    finally:
        if codex_home is not None:
            _cleanup_isolated_codex_home(codex_home)


def _codex_runtime_provenance(*, preserve_user_config: bool = False) -> dict[str, Any]:
    """Codex provenance metadata.

    2026-05-04 fix: trace variant (preserve_user_config=True) now uses
    isolated CODEX_HOME with heartbeat-only hooks (see
    _prepare_trace_codex_home), so its provenance must reflect that.
    Previously returned isolated=False / user_codex_home which polluted
    paper records.
    """
    if preserve_user_config:
        return {
            "codex_home_isolated": True,
            "codex_hook_isolation_version": 2,
            "codex_auth_strategy": "temp_symlink_trace_heartbeat_only",
            "codex_runtime_variant": "codex_cli_trace",
        }
    return {
        "codex_home_isolated": True,
        "codex_hook_isolation_version": 2,
        "codex_auth_strategy": "temp_symlink",
    }


def _prepare_isolated_codex_home() -> Path:
    """Create a temporary auth-only CODEX_HOME so baselines ignore hooks."""
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
    """Create TRACE-isolated CODEX_HOME with auth + heartbeat-only hooks.

    2026-05-03 codex review fix: trace_native_codex previously kept user
    CODEX_HOME so PT hooks could fire, but `~/.codex/hooks.json` registers
    `userpromptsubmit-retrieve-inject.sh` which loads
    `~/.codex/skills/tellonce/shared_lib/fingerprints.yaml`
    leaking ~19 external rules into every turn regardless of B5_MEMORY_DIR.
    Mirrors claude-side Fix 3. Runner injects memory_text via prompt body.
    """
    real_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    source_auth = real_home / "auth.json"
    parent = _codex_temp_parent()
    isolated = Path(tempfile.mkdtemp(prefix="eval_codex_trace_home_", dir=str(parent)))
    os.chmod(isolated, 0o700)
    if source_auth.is_file():
        os.symlink(source_auth, isolated / "auth.json")
    heartbeat = isolated / "eval_trace_codex_heartbeat.py"
    heartbeat.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, time, sys\n"
        "from pathlib import Path\n"
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
    """Probe XDG_RUNTIME_DIR; fall back to ~/.cache if it's read-only.

    2026-05-03 codex review fix mirrors clawarena/runtimes.py."""
    fallback = Path.home() / ".cache" / "eval_codex_runtime"
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        try:
            path = Path(xdg)
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
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


def load_public_workspace(session_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task_spec = json.loads((session_root / "input" / "task_spec.json").read_text(encoding="utf-8"))
    project_state = json.loads((session_root / "input" / "project_state.json").read_text(encoding="utf-8"))
    forbidden = {"expected_answer", "hidden_constraints", "correction_opportunities"}
    leaked = forbidden & set(task_spec)
    if leaked:
        raise RuntimeError(f"public task_spec leaked hidden fields: {sorted(leaked)}")
    return task_spec, project_state


def generate_artifact_with_claude(
    *,
    session_root: Path,
    memory_text: str = "",
    model: str = "haiku",
    timeout_s: int = 120,
    max_budget_usd: float = 0.5,
    runtime_retries: int = 1,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    task_spec, project_state = load_public_workspace(session_root)
    prompt = build_public_prompt(task_spec, project_state, memory_text=memory_text)
    response = call_claude_haiku(
        prompt,
        cwd=session_root,
        timeout_s=timeout_s,
        model=model,
        max_budget_usd=max_budget_usd,
        runtime_retries=runtime_retries,
        extra_env=extra_env,
    )
    return {
        "session_id": task_spec["session_id"],
        "model": model,
        "oracle_dry_run": False,
        "answer": response.get("answer"),
        "constraint_notes": response.get("constraint_notes", []),
        "unresolved_questions": response.get("unresolved_questions", []),
        "decision_log": response.get("decision_log", ""),
        "_runtime_provenance": response.get("_runtime_provenance", {}),
    }


def generate_artifact_with_anthropic_api(
    *,
    session_root: Path,
    memory_text: str = "",
    model: str = "haiku",
    timeout_s: int = 120,
    runtime_retries: int = 1,
) -> dict[str, Any]:
    task_spec, project_state = load_public_workspace(session_root)
    prompt = build_public_prompt(task_spec, project_state, memory_text=memory_text)
    response = call_anthropic_api(
        prompt,
        timeout_s=timeout_s,
        model=model,
        runtime_retries=runtime_retries,
    )
    return {
        "session_id": task_spec["session_id"],
        "model": model,
        "api_model": resolve_anthropic_model_alias(model),
        "oracle_dry_run": False,
        "answer": response.get("answer"),
        "constraint_notes": response.get("constraint_notes", []),
        "unresolved_questions": response.get("unresolved_questions", []),
        "decision_log": response.get("decision_log", ""),
        "_runtime_provenance": response.get("_runtime_provenance", {}),
    }


def generate_artifact_with_openai_api(
    *,
    session_root: Path,
    memory_text: str = "",
    model: str = "gpt-5.4-mini",
    timeout_s: int = 120,
    runtime_retries: int = 1,
) -> dict[str, Any]:
    task_spec, project_state = load_public_workspace(session_root)
    prompt = build_public_prompt(task_spec, project_state, memory_text=memory_text)
    response = call_openai_api(
        prompt,
        timeout_s=timeout_s,
        model=model,
        runtime_retries=runtime_retries,
    )
    return {
        "session_id": task_spec["session_id"],
        "model": model,
        "api_model": model,
        "oracle_dry_run": False,
        "answer": response.get("answer"),
        "constraint_notes": response.get("constraint_notes", []),
        "unresolved_questions": response.get("unresolved_questions", []),
        "decision_log": response.get("decision_log", ""),
        "_runtime_provenance": {
            **response.get("_runtime_provenance", {}),
            "api_model": model,
        },
    }


def generate_artifact_with_codex(
    *,
    session_root: Path,
    memory_text: str = "",
    model: str = "gpt-5.4-mini",
    timeout_s: int = 120,
    runtime_retries: int = 1,
    extra_env: dict[str, str] | None = None,
    preserve_user_config: bool = False,
) -> dict[str, Any]:
    task_spec, project_state = load_public_workspace(session_root)
    prompt = build_public_prompt(task_spec, project_state, memory_text=memory_text)
    response = call_codex_cli(
        prompt,
        cwd=session_root,
        timeout_s=timeout_s,
        model=model,
        runtime_retries=runtime_retries,
        extra_env=extra_env,
        preserve_user_config=preserve_user_config,
    )
    return {
        "session_id": task_spec["session_id"],
        "model": model,
        "oracle_dry_run": False,
        "answer": response.get("answer"),
        "constraint_notes": response.get("constraint_notes", []),
        "unresolved_questions": response.get("unresolved_questions", []),
        "decision_log": response.get("decision_log", ""),
        "_runtime_provenance": response.get("_runtime_provenance", {}),
    }
