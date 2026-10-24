"""Native persistent Codex task sessions, with explicit per-task CLI homes.

Command construction and event decoding only. The admitted host supervisor
owns subprocesses, credentials, hook trust, artifact capture and timeouts.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import uuid

from .codex_text import CodexTextConfig, text_command, configured_context_window
from .native_isolation import validate_codex_command
from .provenance import file_hash


def prepare_native_catalog(*, source: Path, cli_home: Path, model: str, context_window=None) -> str:
    """Copy the pinned official catalog byte for byte, including native tools."""
    if any(not isinstance(p,Path) or not p.is_absolute() or p!=p.resolve() for p in (source,cli_home)):
        raise ValueError('explicit canonical native model catalog and CLI home required')
    fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('native catalog must be a regular source file')
        raw=stream.read(16*1024*1024+1)
    if len(raw)>16*1024*1024:
        raise ValueError('native catalog exceeds its preparation budget')
    def unique(items):
        result={}
        for key,value in items:
            if key in result:
                raise ValueError('duplicate native model catalog field')
            result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite native catalog field')))
    models=value.get('models') if isinstance(value,dict) else None
    if (not isinstance(models,list) or any(not isinstance(item,dict) for item in models) or
            len([item for item in models if item.get('slug')==model])!=1):
        raise ValueError('native catalog must contain exactly one requested model')
    if context_window is not None:
        configured_context_window(next(item for item in models if item.get('slug') == model), context_window)
    target=cli_home/'native-models.json'
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return file_hash(target)


def session_command(*, binary: Path, workspace: Path, cli_home: Path,
                    config: CodexTextConfig, hooks_enabled: bool,
                    native_thread_id: str | None = None,
                    offline_base_url: str | None = None) -> list[str]:
    if type(hooks_enabled) is not bool:
        raise ValueError('execution hooks must be configured explicitly')
    if native_thread_id is not None:
        if not isinstance(native_thread_id, str) or str(uuid.UUID(native_thread_id)) != native_thread_id:
            raise ValueError('resume the exact recorded native UUID, never the last session')
    base = text_command(binary=binary, workspace=workspace, cli_home=cli_home,
                        config=config, offline_base_url=offline_base_url)
    # Share the checked provider/retry/isolation configuration. Persistence and
    # tools are intentional solver differences from a tool-free scoring role.
    args = []
    index = 2
    while index < len(base)-1:
        token = base[index]
        if token == '--ephemeral':
            index += 1
        elif token == '--cd':
            index += 2
        elif token == '-c' and base[index+1].startswith('model_catalog_json='):
            args.extend(['-c','model_catalog_json='+json.dumps(str(cli_home/'native-models.json'))])
            index += 2  # Exact official catalog, retaining native tool metadata.
        elif token == '--disable' and base[index+1] == 'shell_tool':
            args.extend(['--enable', 'shell_tool'])
            index += 2
        elif token == '--disable' and base[index+1] in {'code_mode', 'code_mode_only', 'hooks'}:
            index += 2  # Native model tool mode; hooks are explicitly chosen below.
        else:
            args.append(token)
            index += 1
    args.extend(['--enable' if hooks_enabled else '--disable', 'hooks'])
    args.extend(['-c','notify=[]'])
    command = [str(binary), '--cd', str(workspace), 'exec']
    if native_thread_id is not None:
        command.append('resume')
    command.extend(args)
    if native_thread_id is not None:
        command.append(native_thread_id)
    command.append('-')
    validate_codex_command(command)
    return command


def app_server_command(*, binary: Path, workspace: Path, cli_home: Path,
                       config: CodexTextConfig, hooks_enabled: bool,
                       offline_base_url: str | None = None) -> list[str]:
    """Use the same explicit provider/tool/permission settings in a task server.

    App-server has no exec-only ignore-config flags. Its caller must create a
    fresh private CODEX_HOME and HOME, audit task-local config/AGENTS assets,
    and validate the resulting native thread configuration before any turn.
    """
    source = session_command(binary=binary,workspace=workspace,cli_home=cli_home,
        config=config,hooks_enabled=hooks_enabled,offline_base_url=offline_base_url)
    command = [str(binary),'app-server','--stdio','--strict-config']
    index = 4  # binary, --cd, workspace, exec
    while index < len(source)-1:
        token = source[index]
        if token in {'-c','--enable','--disable'}:
            command.extend(source[index:index+2])
            index += 2
        elif token == '--model':
            command.extend(['-c','model='+json.dumps(source[index+1])])
            index += 2
        elif token in {'--ignore-user-config','--ignore-rules','--skip-git-repo-check','--json'}:
            index += 1
        else:
            raise ValueError('review the app-server adapter for the changed native command')
    validate_codex_command(command)
    return command


@dataclass(frozen=True)
class NativeTurn:
    text: str
    thread_id: str
    usage_json: str
    events_json: str


def parse_native_turn(stdout: str, *, exit_code: int, expected_thread_id: str | None = None) -> NativeTurn:
    if exit_code != 0 or not isinstance(stdout, str) or len(stdout.encode()) > 16*1024*1024:
        raise RuntimeError('native solver request failed or exceeded its event budget')
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    threads, completed, messages = [], [], []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('invalid native solver event')
        kind = event.get('type')
        if kind == 'thread.started':
            threads.append(event.get('thread_id'))
        elif kind == 'turn.completed':
            completed.append(event.get('usage'))
        elif kind == 'turn.started':
            pass
        elif kind in {'turn.failed', 'error'}:
            raise RuntimeError('native solver reported a failed turn')
        elif kind in {'item.started', 'item.updated', 'item.completed'}:
            item = event.get('item')
            if not isinstance(item, dict):
                raise ValueError('invalid native solver item')
            if kind == 'item.completed' and item.get('type') == 'agent_message':
                messages.append(item.get('text'))
        else:
            raise ValueError('native event schema changed; review the runtime adapter')
    if (len(threads) != 1 or not isinstance(threads[0], str) or
            len(completed) != 1 or not isinstance(completed[0], dict) or
            not messages or not isinstance(messages[-1], str) or not messages[-1].strip()):
        raise RuntimeError('native solver lacks one completed response')
    if str(uuid.UUID(threads[0])) != threads[0]:
        raise ValueError('invalid native thread identity')
    if expected_thread_id is not None and threads[0] != expected_thread_id:
        raise RuntimeError('native response belongs to another task session')
    return NativeTurn(messages[-1], threads[0], json.dumps(completed[0], sort_keys=True, allow_nan=False),
                      json.dumps(events, ensure_ascii=False, sort_keys=True, allow_nan=False))
