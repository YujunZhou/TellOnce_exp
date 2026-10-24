"""Explicit Claude Code commands and validated tool-free native responses.

Real runs require a reviewed owner to supply one access token to a fresh home.
This module never reads operator settings, memories, or credential stores.
"""
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
import uuid
from urllib.parse import urlsplit

from .codex_text import TextResult
from .claude_benchmark_mcp import ClaudeBenchmarkMCP
from .native_isolation import claude_settings
from .text_input_budget import MAX_TEXT_INPUT_BYTES, validate_input_limit


MODELS = frozenset({'claude-haiku-4-5-20251001', 'claude-sonnet-5'})


def thinking_progress(row):
    """Validate native progress estimates; only final usage is billable evidence."""
    if any(type(row.get(k)) is not int or row[k]<0 for k in ('estimated_tokens','estimated_tokens_delta')):
        raise ValueError('invalid native thinking progress estimate')


@dataclass(frozen=True)
class ClaudeConfig:
    model: str
    reasoning_effort: str | None = None
    context_window: int = 200000
    input_max_bytes: int = MAX_TEXT_INPUT_BYTES

    def __post_init__(self):
        validate_input_limit(self.input_max_bytes)
        capacities={'claude-haiku-4-5-20251001':{200000},'claude-sonnet-5':{200000,1000000}}
        if self.model not in MODELS or self.context_window not in capacities[self.model]:
            raise ValueError('explicit admitted Claude model and capacity required')
        if self.model.startswith('claude-haiku'):
            if self.reasoning_effort is not None:
                raise ValueError('Haiku has no admitted effort control')
        elif self.reasoning_effort not in {'low','medium','high'}:
            raise ValueError('Sonnet requires explicit reasoning effort')


def _canonical(path):
    if not isinstance(path,Path) or not path.is_absolute() or path != path.resolve():
        raise ValueError('canonical absolute native paths required')
    return str(path)


def environment(*, cli_home, host_home, access_token_fd=None, offline_base_url=None):
    _canonical(cli_home); _canonical(host_home)
    if (access_token_fd is None)==(offline_base_url is None):
        raise ValueError('exactly one real login or local synthetic endpoint required')
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(host_home),
         'CLAUDE_CONFIG_DIR':str(cli_home),'DISABLE_AUTOUPDATER':'1',
         'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
         'CLAUDE_CODE_DISABLE_AUTO_MEMORY':'1','CLAUDE_CODE_DISABLE_ORG_MEMORY':'1',
         'DISABLE_PLUGIN_AUTOLOAD':'1','CLAUDE_CODE_DISABLE_PLUGIN_FORWARDING':'1',
         'CLAUDE_CODE_DISABLE_CLAUDE_MDS':'1','CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS':'1',
         'CLAUDE_CODE_DISABLE_BUNDLED_SKILLS':'1','ENABLE_TOOL_SEARCH':'false',
         'CLAUDE_CODE_SKIP_PLUGIN_MCP_SERVERS':'1','CLAUDE_CODE_MAX_RETRIES':'0',
         'CLAUDE_CODE_OAUTH_401_WAIT_MS':'0'}
    if offline_base_url is not None:
        url=urlsplit(offline_base_url)
        if (url.scheme!='http' or url.hostname!='127.0.0.1' or not url.port or
                url.username or url.password or url.query or url.fragment or url.path not in {'','/'}):
            raise ValueError('synthetic Claude endpoint must be loopback only')
        env.update(ANTHROPIC_BASE_URL=offline_base_url,ANTHROPIC_API_KEY='SYNTHETIC_INVALID_KEY')
    else:
        if type(access_token_fd) is not int or access_token_fd < 3:
            raise ValueError('one explicitly inherited access-token descriptor required')
        env['CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR']=str(access_token_fd)
    return env


def command(*, binary, workspace, cli_home, config, session_id, text_only=True,
            resume=False, system_prompt=None, settings_path=None, benchmark_mcp=None, hooks_enabled=False):
    for path in (binary,workspace,cli_home):_canonical(path)
    if (workspace==cli_home or workspace in cli_home.parents or cli_home in workspace.parents or
            str(uuid.UUID(session_id))!=session_id or not isinstance(config,ClaudeConfig) or
            type(text_only) is not bool or type(resume) is not bool):
        raise ValueError('one isolated native task session required')
    if benchmark_mcp is not None and not isinstance(benchmark_mcp, ClaudeBenchmarkMCP):
        raise ValueError('only the task-owned benchmark MCP facade is admitted')
    if type(hooks_enabled) is not bool:
        raise ValueError('explicit hook mode required')
    if text_only and (resume or settings_path is not None or benchmark_mcp is not None or hooks_enabled):
        raise ValueError('text roles use fresh sessions without executable hooks')
    mcp = benchmark_mcp.configuration if benchmark_mcp is not None else {'mcpServers': {}}
    args=[str(binary),'--print','--output-format','stream-json','--verbose',
          '--include-partial-messages','--model',config.model,'--setting-sources','',
          '--restricted' if benchmark_mcp is not None or hooks_enabled else '--safe-mode',
          '--strict-mcp-config','--mcp-config',json.dumps(mcp, separators=(',', ':')),
          '--disable-slash-commands','--no-chrome','--permission-prompts','none',
          '--permission-mode','dontAsk', '--resume' if resume else '--session-id',session_id]
    if config.reasoning_effort is not None:args+=['--effort',config.reasoning_effort]
    if text_only:
        args+=['--tools','','--no-session-persistence','--settings',
               '{"disableAllHooks":true,"autoMemoryEnabled":false,"enabledPlugins":{}}',
               '--system-prompt',system_prompt or 'Complete the supplied text role without tools.']
    else:
        if settings_path is None:raise ValueError('solver requires its reviewed settings file')
        _canonical(settings_path)
        args+=['--tools','Read,Write,Edit,Bash,Glob,Grep','--settings',str(settings_path)]
        if benchmark_mcp is not None:
            args+=['--allowedTools', ','.join('mcp__benchmark__'+name for name in sorted(benchmark_mcp.names))]
        if system_prompt is not None:args+=['--append-system-prompt',system_prompt]
    # The wrapper preserves PID and binds the native process to this task host.
    return [str(Path(sys.executable).resolve()),'-S','-B',
            str(Path(__file__).with_name('native_exec_worker.py').resolve()),
            '--owner-pid',str(os.getpid()),'--',*args]


def task_settings(*,workspace,cli_home,binary,hooks=None):
    settings=claude_settings(workspace=workspace,cli_home=cli_home,runtime_roots=(binary.parent,))
    # Claude's root deny expansion masks an allowed top-level directory before
    # rebinding nested exceptions, breaking Ubuntu's /lib -> /usr/lib links.
    # Enumerate the same denied host roots while retaining system runtimes;
    # nested work/runtime exceptions remain explicit. /proc is PID-isolated by
    # the native sandbox, /dev is reconstructed, and private roots stay masked.
    runtime_names={'usr','bin','sbin','lib','lib32','lib64','libx32','proc','sys','dev'}
    settings['sandbox']['filesystem']['denyRead']=[str(p) for p in sorted(Path('/').iterdir())
                                                   if p.name not in runtime_names]
    settings.update(autoMemoryEnabled=False,enabledPlugins={})
    if hooks is None:settings['disableAllHooks']=True
    else:settings.update(hooks=hooks,disableAllHooks=False)
    return settings


class TextEvents:
    """Reject hidden retries, tool use, model changes and incomplete outputs."""
    def __init__(self, *, config, session_id):
        self.config,self.session_id=config,session_id
        self.initialized=False;self.results=[];self.messages={};self.usage=None;self.availability=None

    def observe(self,row):
        if not isinstance(row,dict):raise ValueError('invalid Claude event')
        if row.get('session_id') not in (None,self.session_id):
            raise ValueError('foreign Claude session event')
        kind=row.get('type')
        if kind=='system':
            subtype=row.get('subtype')
            if subtype=='init':
                if (self.initialized or row.get('model')!=self.config.model or row.get('tools')!=[] or
                        row.get('mcp_servers')!=[] or row.get('skills')!=[] or row.get('plugins')!=[]):
                    raise ValueError('Claude text configuration differs from admitted model/tools')
                self.initialized=True
            elif subtype=='status' and row.get('status') in {'requesting',None}:
                pass
            elif subtype=='thinking_tokens':thinking_progress(row)
            elif subtype in {'api_retry','compact_boundary','local_command'}:
                raise ValueError('unadmitted retry, compaction or local command')
            elif subtype not in {'hook_started','hook_progress','hook_response'}:
                raise ValueError('unknown Claude system event requires review')
            else:raise ValueError('text roles cannot run hooks')
        elif kind=='assistant':
            message=row.get('message',{})
            if message.get('model')!=self.config.model or row.get('parent_tool_use_id') is not None:
                raise ValueError('Claude response changed model or used a subagent')
            blocks=message.get('content',[])
            if any(b.get('type') not in {'text','thinking','redacted_thinking'} for b in blocks):
                raise ValueError('text role attempted a tool or unsupported content')
            identity=message.get('id')
            if not isinstance(identity,str) or not identity:raise ValueError('missing response identity')
            self.messages[identity]=''.join(b['text'] for b in blocks if b.get('type')=='text')
        elif kind=='stream_event':
            event=row.get('event',{})
            if event.get('type')=='content_block_start' and event.get('content_block',{}).get('type')=='tool_use':
                raise ValueError('text role attempted a streamed tool')
        elif kind=='result':
            if row.get('subtype')!='success' or row.get('is_error') is not False:
                raise ValueError('Claude text request did not succeed')
            self.results.append(row)
        elif kind=='rate_limit_event':
            if row.get('rate_limit_info',{}).get('status')=='rejected':
                from .native_availability import NativeUnavailable
                self.availability=NativeUnavailable('native_quota_exhausted',row['rate_limit_info'].get('resetsAt'))
                raise self.availability
        else:raise ValueError('unknown Claude text event requires review')

    def finish(self, *, exit_code):
        if exit_code or not self.initialized or len(self.messages)!=1 or len(self.results)!=1:
            raise ValueError('text role requires one complete actual response')
        result=self.results[0];text=result.get('result')
        if not isinstance(text,str) or not text.strip() or text!=next(iter(self.messages.values())):
            raise ValueError('result differs from actual Claude assistant response')
        usage=result.get('usage');models=result.get('modelUsage')
        if (not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0
                for k in ('input_tokens','output_tokens')) or
                not isinstance(models,dict) or set(models)!={self.config.model}):
            raise ValueError('Claude usage or actual model missing')
        cached=usage.get('cache_read_input_tokens',0);created=usage.get('cache_creation_input_tokens',0)
        if any(type(v) is not int or v<0 for v in (cached,created)):
            raise ValueError('invalid Claude cache usage')
        self.usage={'inputTokens':usage['input_tokens']+cached+created,'cachedInputTokens':cached,
                    'outputTokens':usage['output_tokens'],
                    'native_usage':usage,'model_usage':models,'native_cost_estimate_usd':result.get('total_cost_usd'),
                    'subscription_charge_usd':None}
        return TextResult(text,self.session_id,json.dumps(self.usage,sort_keys=True,allow_nan=False))


def compact_boundary(row):
    """Validate native context maintenance without treating it as a delivery."""
    metadata=row.get('compact_metadata')
    if (not isinstance(metadata,dict) or metadata.get('trigger') not in {'auto','manual'} or
            type(metadata.get('pre_tokens')) is not int or metadata['pre_tokens']<0):
        raise ValueError('malformed native compaction boundary')
