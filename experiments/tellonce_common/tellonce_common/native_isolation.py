"""Native CLI filesystem policies; no model calls or personal-config loading.

Linux Codex 0.153.4 was checked with synthetic public/private files. These
policies supplement role projections; initial workspaces still need a public
asset manifest, and managed/remote tools need separate adapter review.
"""
from __future__ import annotations

import json
from pathlib import Path


def _path(path: Path) -> str:
    if not isinstance(path, Path) or not path.is_absolute() or path != path.resolve():
        raise ValueError('isolation paths must be absolute, canonical, and free of symlink aliases')
    if path == Path('/'):
        raise ValueError('the filesystem root cannot be an experiment asset')
    return str(path)


def codex_permission_arguments(*, workspace: Path, cli_home: Path,
                               runtime_roots: tuple[Path, ...], readonly_assets: tuple[Path, ...] = (),
                               writable_state: tuple[Path, ...] = ()) -> list[str]:
    work, home = _path(workspace), _path(cli_home)
    if cli_home == workspace or cli_home.is_relative_to(workspace) or workspace.is_relative_to(cli_home):
        raise ValueError('CLI state and public workspace must be separate')
    if not runtime_roots:
        raise ValueError('declare the actual native runtime/helper binaries')
    filesystem = {':root': 'deny', ':minimal': 'read', work: 'write'}
    for path in (*runtime_roots, *readonly_assets):
        key = _path(path)
        if workspace == path or workspace.is_relative_to(path) or cli_home == path or cli_home.is_relative_to(path):
            raise ValueError('read-only dependency must not reopen the workspace or private CLI state')
        filesystem[key] = 'read'
    protected = (workspace, cli_home, *runtime_roots, *readonly_assets)
    for index, path in enumerate(writable_state):
        key = _path(path)
        if any(path == other or path.is_relative_to(other) or other.is_relative_to(path)
               for other in (*protected, *writable_state[:index])):
            raise ValueError('writable grader state must be separate from task, CLI and read-only assets')
        filesystem[key] = 'write'
    # Codex 0.153.4 synthesizes argv[0] helper links under tmp/arg0, not
    # tmp/path. This permits the helper links but not auth/session/config files.
    filesystem[str(cli_home/'tmp/arg0')] = 'read'
    literal = '{' + ', '.join(json.dumps(k)+' = '+json.dumps(v) for k, v in filesystem.items()) + '}'
    return ['-c', 'default_permissions="experiment"', '-c', 'permissions.experiment.filesystem='+literal,
            '-c', 'permissions.experiment.network={enabled=false}']


def claude_settings(*, workspace: Path, cli_home: Path,
                    runtime_roots: tuple[Path, ...], readonly_assets: tuple[Path, ...] = ()) -> dict:
    # Validate the same separation assumptions for both native backends.
    codex_permission_arguments(workspace=workspace, cli_home=cli_home,
                               runtime_roots=runtime_roots, readonly_assets=readonly_assets)
    return {
        'permissions': {'defaultMode': 'dontAsk', 'blockReadsOutsideWorkingDirectories': True,
                        'allow': ['Read', 'Edit', 'Write', 'Bash'],
                        'deny': ['Agent', 'WebSearch', 'WebFetch', 'Skill',
                                 'Read(/'+str(cli_home)+'/**)', 'Edit(/'+str(cli_home)+'/**)']},
        'sandbox': {'enabled': True, 'failIfUnavailable': True, 'allowUnsandboxedCommands': False,
                    'excludedCommands': [],
                    'filesystem': {'disabled': False, 'denyRead': ['/'],
                                   'allowRead': [str(workspace), '/usr', '/bin', '/lib', '/lib64',
                                                 *map(str, runtime_roots), *map(str, readonly_assets)],
                                   'denyWrite': list(map(str, readonly_assets))},
                    'network': {'allowedDomains': [], 'allowAllUnixSockets': False}},
        'enableAllProjectMcpServers': False,
    }


def validate_codex_command(command: list[str]):
    if any(arg == '--sandbox' or arg.startswith('--sandbox=') or
           'sandbox_mode=' in arg or 'sandbox_workspace_write' in arg or
           arg == '--dangerously-bypass-approvals-and-sandbox' for arg in command):
        raise ValueError('legacy sandbox options override the explicit read-isolation profile')
