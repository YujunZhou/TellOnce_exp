"""Tool-free native Codex requests for feedback, selection, and judging.

This is not the task-solving runtime. A reviewed experiment supervisor must
admit the run before using this transport. Construction and parsing are pure;
the offline integration fixture uses a loopback fake provider and no login.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .native_isolation import codex_permission_arguments, validate_codex_command
from .provenance import content_hash
from .text_input_budget import MAX_TEXT_INPUT_BYTES, validate_input_limit


TOOL_FREE_FEATURES = ('shell_tool', 'multi_agent', 'multi_agent_v2', 'plugins', 'apps', 'shell_snapshot',
                     'sleep_tool', 'tool_suggest', 'skill_search', 'code_mode', 'code_mode_only',
                     'view_image', 'goals', 'image_generation', 'browser_use', 'computer_use',
                     'request_permissions_tool', 'memories', 'hooks')


@dataclass(frozen=True)
class CodexTextConfig:
    model: str
    reasoning_effort: str = 'medium'
    stream_idle_timeout_ms: int = 120000
    input_max_bytes: int = MAX_TEXT_INPUT_BYTES
    context_window: int | None = None

    def __post_init__(self):
        validate_input_limit(self.input_max_bytes)
        if not isinstance(self.model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', self.model):
            raise ValueError('explicit model identifier required')
        if 'sonnet' in self.model.lower():
            raise ValueError('Sonnet is excluded from this experiment')
        if self.reasoning_effort not in {'minimal', 'low', 'medium', 'high', 'xhigh'}:
            raise ValueError('unsupported explicit reasoning effort')
        if type(self.stream_idle_timeout_ms) is not int or not 1000 <= self.stream_idle_timeout_ms <= 300000:
            raise ValueError('invalid stream timeout')
        if self.context_window is not None and (type(self.context_window) is not int or self.context_window < 4096):
            raise ValueError('explicit native context window must be a positive token capacity')

    @property
    def digest(self):
        return content_hash(self.__dict__)


def text_command(*, binary: Path, workspace: Path, cli_home: Path,
                 config: CodexTextConfig, offline_base_url: str | None = None) -> list[str]:
    if not binary.is_absolute() or binary != binary.resolve():
        raise ValueError('pin the canonical native binary')
    provider = {'name': 'Reviewed experiment text transport', 'wire_api': 'responses',
                'requires_openai_auth': offline_base_url is None,
                'request_max_retries': 0, 'stream_max_retries': 0,
                'stream_idle_timeout_ms': config.stream_idle_timeout_ms}
    if offline_base_url is not None:
        # This test seam can never redirect a fixture to an external provider.
        from urllib.parse import urlsplit
        url = urlsplit(offline_base_url)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.port is None or
                url.username or url.password or url.query or url.fragment or url.path != '/v1'):
            raise ValueError('offline provider must be a local loopback fixture')
        provider['base_url'] = offline_base_url
    literal = '{' + ','.join(k+'='+json.dumps(v) for k, v in provider.items()) + '}'
    command = [str(binary), 'exec', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
               '--ephemeral', '--json', '--cd', str(workspace), '--model', config.model,
               '-c', 'model_provider="experiment_text"',
               '-c', 'model_providers.experiment_text='+literal,
               '-c', 'model_reasoning_effort='+json.dumps(config.reasoning_effort),
               '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
               '-c', 'agents.enabled=false', '-c', 'tools.experimental_request_user_input.enabled=false',
               '-c', 'tools.update_plan.enabled=false',
               '-c', 'approval_policy="never"',
               '-c', 'cli_auth_credentials_store="ephemeral"',
               '-c', 'analytics.enabled=false',
               '-c', 'model_catalog_json='+json.dumps(str(cli_home/'text-role-model.json')),
               *codex_permission_arguments(workspace=workspace, cli_home=cli_home,
                                            runtime_roots=(binary.parent,))]
    for feature in TOOL_FREE_FEATURES:
        command += ['--disable', feature]
    if config.context_window is not None:
        command += ['-c', 'model_context_window='+str(config.context_window)]
    command += ['--enable', 'skip_host_skill_discovery',
                '-c', 'suppress_unstable_features_warning=true', '-']
    validate_codex_command(command)
    return command


def configured_context_window(metadata, requested=None):
    default = metadata.get('context_window')
    maximum = metadata.get('max_context_window', default)
    if (type(default) is not int or default < 4096 or type(maximum) is not int or maximum < default or
            (requested is not None and (type(requested) is not int or not 4096 <= requested <= maximum))):
        raise ValueError('context capacity must fit the pinned native model catalog')
    return default if requested is None else requested


def text_role_catalog(source_json: str, *, model: str) -> str:
    """Disable model-declared tools for a text role, retaining model settings.

    Codex 0.153.4 model metadata takes precedence over code-mode feature flags.
    The supervisor must pin the official input catalog and this derived file,
    then place it at cli_home/text-role-model.json before launching text_command.
    A missing catalog is a native startup error, not a tool-enabled fallback.
    """
    source = json.loads(source_json)
    candidates = [m for m in source['models'] if m.get('slug') == model]
    if len(candidates) != 1:
        raise ValueError('the pinned catalog must contain exactly one requested model')
    selected = dict(candidates[0])
    selected['tool_mode'] = 'direct'
    selected['shell_type'] = 'disabled'
    selected['apply_patch_tool_type'] = None
    selected['experimental_supported_tools'] = []
    return json.dumps({'models':[selected]}, ensure_ascii=False, sort_keys=True, allow_nan=False)


def text_app_server_command(*, binary: Path, workspace: Path, cli_home: Path,
                            config: CodexTextConfig, offline_base_url=None) -> list[str]:
    """The text-only command settings, with app-server's explicit auth lease.

    App-server has no ignore-user-config flag. The caller must create fresh
    private homes and the derived tool-free model catalog before starting it.
    """
    source=text_command(binary=binary,workspace=workspace,cli_home=cli_home,
        config=config,offline_base_url=offline_base_url)
    command=[str(binary),'app-server','--stdio','--strict-config']
    index=2
    while index<len(source)-1:
        token=source[index]
        if token in {'-c','--enable','--disable'}:
            command.extend(source[index:index+2])
            index+=2
        elif token=='--model':
            command.extend(['-c','model='+json.dumps(source[index+1])])
            index+=2
        elif token=='--cd':
            index+=2
        elif token in {'--ignore-user-config','--ignore-rules','--skip-git-repo-check','--json','--ephemeral'}:
            index+=1
        else:
            raise ValueError('review the native text app-server command for a changed option')
    validate_codex_command(command)
    return command


def request_tool_surface(request: dict) -> list[dict]:
    """Inspect both ordinary Responses and native Responses Lite tool fields."""
    containers = [request.get('tools', [])]
    containers.extend(item.get('tools', []) for item in request.get('input', [])
                      if isinstance(item, dict) and item.get('type') == 'additional_tools')
    result = []
    def collect(tool, namespace=''):
        if not isinstance(tool, dict):
            raise ValueError('invalid tool declaration in native request')
        name = tool.get('name') or tool.get('function', {}).get('name') or ''
        qualified = '.'.join(p for p in (namespace, name) if p)
        if tool.get('type') == 'namespace':
            for nested in tool.get('tools', []):
                collect(nested, qualified)
        else:
            result.append({'type': tool.get('type'), 'name': qualified})
    for container in containers:
        for tool in container:
            collect(tool)
    return result


def text_environment(*, cli_home: Path, host_home: Path) -> dict[str, str]:
    if any(not p.is_absolute() or p != p.resolve() for p in (cli_home, host_home)):
        raise ValueError('explicit canonical role and host homes required')
    # No operator project settings, provider variables, MCP configuration,
    # Python import path, or API keys are inherited. Native auth, if admitted,
    # must be provisioned separately in this role's private CLI directory.
    return {'PATH': '/usr/bin:/bin', 'HOME': str(host_home), 'LANG': 'C.UTF-8',
            'CODEX_HOME': str(cli_home)}


@dataclass(frozen=True)
class TextResult:
    text: str
    thread_id: str
    usage_json: str


def parse_text_result(stdout: str, *, exit_code: int) -> TextResult:
    if exit_code != 0 or len(stdout.encode('utf-8')) > 4 * 1024 * 1024:
        raise RuntimeError('Codex text request failed or exceeded its output budget')
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    messages, threads, completions = [], [], []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('invalid native event')
        kind = event.get('type')
        if kind == 'thread.started':
            threads.append(event.get('thread_id'))
        elif kind == 'turn.completed':
            completions.append(event.get('usage'))
        elif kind in {'turn.failed', 'error'}:
            raise RuntimeError('native text request reported a failure')
        elif kind=='turn.started':
            pass
        elif kind in {'item.started', 'item.updated', 'item.completed'}:
            item = event.get('item')
            if not isinstance(item, dict):
                raise ValueError('invalid native item')
            if item.get('type') not in {'agent_message', 'reasoning'}:
                raise RuntimeError('unexpected tool or error in a tool-free text request')
            if kind == 'item.completed' and item.get('type') == 'agent_message':
                messages.append(item.get('text'))
        else:
            raise ValueError('native text event schema changed; review the adapter')
    if (len(threads) != 1 or not isinstance(threads[0], str) or not threads[0] or
            len(completions) != 1 or not isinstance(completions[0], dict) or
            not messages or not isinstance(messages[-1], str) or not messages[-1].strip()):
        raise RuntimeError('native text request lacks one completed response')
    return TextResult(messages[-1], threads[0], json.dumps(completions[0], sort_keys=True, allow_nan=False))
