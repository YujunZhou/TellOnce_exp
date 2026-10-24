"""Read access tokens from one explicitly admitted native managed-auth owner.

The supplied account-only NativeAppServer owns normal OAuth refresh and its
credential store. This reader never writes that store or sends refresh tokens
to task processes. Preparing that server with operator features disabled is a
separate required integration; do not pass a task-solving server here.
"""
import json
import base64
import os
from pathlib import Path
import stat
import threading
import time

from .native_auth import AccessTokens
from .native_app_server import NativeAppServer
from .codex_text import TOOL_FREE_FEATURES, text_environment
from .native_execution import run_native_command


def _assert_synthetic_store(home):
    """A local test endpoint must never receive real cached credentials."""
    fd=os.open(home/'auth.json',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size>65536:
            raise ValueError('synthetic auth fixture must be a bounded regular file')
        data=os.read(fd,65537)
        value=json.loads(data)
        tokens=value['tokens']
        parts=tokens['access_token'].split('.')
        claims=json.loads(base64.urlsafe_b64decode(parts[1]+'='*(-len(parts[1])%4)))
        if (value.get('OPENAI_API_KEY') is not None or tokens['refresh_token']!='SYNTHETIC_REFRESH_TOKEN' or
                tokens['id_token']!=tokens['access_token'] or tokens['account_id']!='synthetic-account' or
                len(parts)!=3 or parts[2]!='SYNTHETIC_INVALID_SIGNATURE' or
                claims.get('email')!='synthetic@example.invalid' or
                claims.get('https://api.openai.com/auth',{}).get('chatgpt_account_id')!='synthetic-account'):
            raise ValueError('local auth services require invalid synthetic credentials')
    except Exception:
        raise ValueError('local auth services require an explicit synthetic credential store') from None
    finally:
        os.close(fd)


def owner_command(*, binary: Path, model_catalog: Path, auxiliary_base_url=None, mcp_names=()):
    if any(not p.is_absolute() or p!=p.resolve() for p in (binary,model_catalog)):
        raise ValueError('canonical native owner binary and pinned catalog required')
    # This owner cannot create model threads through its public interface. A
    # loopback sink also prevents accidental model requests if that regresses.
    provider='{name="Native auth owner",wire_api="responses",requires_openai_auth=true,'+\
        'base_url="http://127.0.0.1:9/v1",request_max_retries=0,stream_max_retries=0}'
    command=[str(binary),'app-server','--stdio','--strict-config',
        '-c','model_provider="experiment_auth_owner"','-c','model_providers.experiment_auth_owner='+provider,
        '-c','model_catalog_json='+json.dumps(str(model_catalog)),
        '-c','cli_auth_credentials_store="file"','-c','analytics.enabled=false',
        '-c','mcp_servers={}','-c','notify=[]','-c','agents.enabled=false',
        '-c','web_search="disabled"','-c','project_doc_max_bytes=0',
        '-c','default_permissions=":read-only"']
    for feature in TOOL_FREE_FEATURES:
        command+=['--disable',feature]
    command+=['--enable','skip_host_skill_discovery','-c','suppress_unstable_features_warning=true']
    if mcp_names:
        if len(mcp_names)>1024 or any(not isinstance(name,str) or not 1<=len(name)<=512 for name in mcp_names):
            raise ValueError('bounded native MCP configuration names required')
        disabled='{'+','.join(json.dumps(name)+'={enabled=false}' for name in mcp_names)+'}'
        if len(disabled.encode())>65536:
            raise ValueError('native MCP disable overrides exceed their argument budget')
        command+=['-c','mcp_servers='+disabled]
    if auxiliary_base_url is not None:
        from urllib.parse import urlsplit
        url=urlsplit(auxiliary_base_url)
        if (url.scheme!='http' or url.hostname!='127.0.0.1' or url.port is None or
                url.username or url.password or url.query or url.fragment or url.path not in {'','/'}):
            raise ValueError('synthetic native auth services must be loopback-only')
        command+=['-c','chatgpt_base_url='+json.dumps(auxiliary_base_url)]
    return command


class NativeAccountProcess:
    """One account-only server; never exposes a model/thread RPC to its owner.

    CODEX_HOME is deliberately the existing native credential home. Only this
    account process uses it. Task workers keep fresh homes and access tokens.
    Native config is read, but its MCP, hooks, tools and analytics are disabled;
    effective configuration is checked before the first credential read.
    """
    def __init__(self, *, binary, auth_home, private_root, model_catalog, authorize,
                 auxiliary_base_url=None, synthetic_refresh_url=None, record=lambda event:None):
        if (any(not isinstance(p,Path) or not p.is_absolute() or p!=p.resolve()
                for p in (binary,auth_home,private_root,model_catalog)) or
                private_root==auth_home or private_root.is_relative_to(auth_home) or
                auth_home.is_relative_to(private_root)):
            raise ValueError('separate canonical native credential home and private owner runtime required')
        authorize('process_prepare')
        self._server=None
        self._closed=False
        private_root.mkdir(mode=0o700,exist_ok=False)
        stage='list_mcp'
        try:
            environment=text_environment(cli_home=auth_home,host_home=private_root/'empty-home')
            if auxiliary_base_url is not None or synthetic_refresh_url is not None:
                _assert_synthetic_store(auth_home)
            if synthetic_refresh_url is not None:
                from urllib.parse import urlsplit
                url=urlsplit(synthetic_refresh_url)
                if (auxiliary_base_url is None or url.scheme!='http' or url.hostname!='127.0.0.1' or
                        url.port is None or url.username or url.password or url.query or url.fragment or
                        url.path!='/oauth/token'):
                    raise ValueError('synthetic refresh must use the local fixture endpoint')
                # A pinned-binary test seam, never inherited in live mode.
                environment['CODEX_REFRESH_TOKEN_URL_OVERRIDE']=synthetic_refresh_url
            # Empty tables merge with native config rather than erasing it.
            # List configured names using the read-only CLI command before
            # starting app-server, then explicitly disable every named server.
            listing=run_native_command([str(binary),'mcp','list','--json','-c','analytics.enabled=false'],
                cwd=private_root,env=environment,prompt='',timeout_s=60,on_event=None,
                stdout_budget=1024*1024,stderr_budget=32768)
            if listing.status!='completed' or listing.exit_code!=0:
                raise ValueError('native MCP configuration could not be listed')
            declared=json.loads(listing.stdout)
            if not isinstance(declared,list) or any(not isinstance(item,dict) for item in declared):
                raise ValueError('native MCP configuration schema changed')
            names=tuple(item.get('name') for item in declared)
            command=owner_command(binary=binary,model_catalog=model_catalog,
                auxiliary_base_url=auxiliary_base_url,mcp_names=names)
            stage='start'
            self._server=NativeAppServer(command,cwd=private_root,
                env=environment)
            stage='initialize'
            self._server.request('initialize',{'clientInfo':{'name':'reviewed-account-only-owner','version':'1'},
                'capabilities':{'experimentalApi':True}},timeout_s=60)
            self._server.notify('initialized',{})
            stage='config_read'
            config=self._server.request('config/read',{'includeLayers':False},timeout_s=60)['config']
            features=config.get('features',{})
            servers=config.get('mcp_servers',{})
            checks={'mcp_disabled':isinstance(servers,dict) and all(isinstance(value,dict) and
                        value.get('enabled') is False for value in servers.values()),
                    'native_file_auth':config.get('cli_auth_credentials_store')=='file',
                    'analytics_disabled':config.get('analytics',{}).get('enabled') is False,
                    'model_sink':config.get('model_provider')=='experiment_auth_owner',
                    'notifications_disabled':config.get('notify',[])==[],
                    'host_skill_discovery_disabled':features.get('skip_host_skill_discovery') is True,
                    **{'feature_disabled_'+name:features.get(name) is False for name in TOOL_FREE_FEATURES}}
            stage='config_check'
            record({'type':'native_auth_owner_config','checks':checks})
            if not all(checks.values()):
                raise ValueError('native account owner did not honor feature isolation')
        except Exception as exc:
            try:
                self.close()
            finally:
                try:
                    record({'type':'native_auth_owner_prepare_failed','stage':stage,'error_type':type(exc).__name__})
                except Exception:
                    pass  # Preparation still fails; audit failure cannot skip cleanup.
            raise RuntimeError('native account-only process could not be prepared') from None

    def request(self, method, params, *, timeout_s):
        if (self._closed or method!='account/read' or not isinstance(params,dict) or
                set(params)!={'refreshToken'} or type(params['refreshToken']) is not bool):
            raise ValueError('account owner allows only explicit native account reads/refreshes')
        return self._server.request(method,params,timeout_s=timeout_s)

    def close(self):
        if not self._closed:
            self._closed=True
            if self._server is not None:
                self._server.close()


class NativeAuthOwner:
    def __init__(self, *, server, auth_home: Path, authorize, record):
        if (not isinstance(auth_home,Path) or not auth_home.is_absolute() or auth_home!=auth_home.resolve() or
                not callable(authorize) or not callable(record)):
            raise ValueError('explicit native auth owner, canonical credential home and admission required')
        authorize('prepare')
        self.server,self.auth_home,self.authorize,self.record=server,auth_home,authorize,record
        self._lock=threading.Lock()
        self._account=None
        self._closed,self._failed=False,False

    def _read_access(self, plan_type):
        # Atomic replacement by another native login process is allowed. A
        # bounded re-read handles that race without triggering OAuth again.
        for attempt in range(2):
            directory=os.open(self.auth_home,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                home_stat=os.fstat(directory)
                if home_stat.st_uid!=os.geteuid():
                    raise ValueError('foreign auth owner directory')
                fd=os.open('auth.json',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
                try:
                    before=os.fstat(fd)
                    if (not stat.S_ISREG(before.st_mode) or before.st_uid!=os.geteuid() or
                            before.st_nlink!=1 or before.st_size>65536):
                        raise ValueError('native auth store is not a bounded owned regular file')
                    parts=[]
                    size=0
                    while True:
                        part=os.read(fd, min(8192,65537-size))
                        if not part:
                            break
                        parts.append(part)
                        size+=len(part)
                        if size>65536:
                            raise ValueError('native auth store exceeded its read budget')
                    after=os.fstat(fd)
                    current=os.stat('auth.json',dir_fd=directory,follow_symlinks=False)
                    stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                    if stamp(before)!=stamp(after) or stamp(before)!=stamp(current):
                        if attempt==0:
                            continue
                        raise ValueError('native auth store changed during the bounded read')
                    def pairs(items):
                        value={}
                        for key,item in items:
                            if key in value:
                                raise ValueError('duplicate auth store field')
                            value[key]=item
                        return value
                    value=json.loads(b''.join(parts),object_pairs_hook=pairs,
                        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('non-finite auth store field')))
                    if (not isinstance(value,dict) or value.get('auth_mode') not in {None,'chatgpt'} or
                            value.get('OPENAI_API_KEY') is not None or not isinstance(value.get('tokens'),dict)):
                        raise ValueError('only native managed ChatGPT authentication is admitted')
                    tokens=value['tokens']
                    return AccessTokens(tokens.get('access_token'),tokens.get('account_id'),plan_type)
                finally:
                    os.close(fd)
            finally:
                os.close(directory)
        raise RuntimeError('native auth store could not be read')

    def __call__(self, *, refresh, previous_account_id, timeout_s):
        if (type(refresh) is not bool or not isinstance(timeout_s,(int,float)) or
                isinstance(timeout_s,bool) or not 0 < timeout_s <= 8 or
                (previous_account_id is not None and not isinstance(previous_account_id,str))):
            raise ValueError('bounded native auth acquisition required')
        started=time.monotonic()
        if not self._lock.acquire(timeout=timeout_s):
            raise RuntimeError('native auth owner is busy')
        try:
            if self._closed or self._failed:
                raise RuntimeError('native auth owner is unavailable')
            self.authorize('refresh' if refresh else 'read')
            if ((previous_account_id is not None and previous_account_id!=self._account) or
                    (refresh and (self._account is None or previous_account_id is None))):
                raise ValueError('native auth acquisition belongs to another account')
            remaining=timeout_s-(time.monotonic()-started)
            if remaining<=0:
                raise TimeoutError('native auth owner wait exhausted the budget')
            # Only the native owner refreshes. Do not duplicate its OAuth flow,
            # write auth.json ourselves, or fan out rotating refresh tokens.
            reply=self.server.request('account/read',{'refreshToken':refresh},timeout_s=remaining)
            account=reply.get('account')
            if (not isinstance(account,dict) or account.get('type')!='chatgpt' or
                    reply.get('requiresOpenaiAuth') is not True):
                raise ValueError('native owner is not using managed ChatGPT authentication')
            tokens=self._read_access(account.get('planType'))
            if self._account is not None and tokens.account_id!=self._account:
                raise ValueError('native logged-in account changed')
            if time.monotonic()-started>=timeout_s:
                raise TimeoutError('native auth owner exceeded the acquisition budget')
            self._account=tokens.account_id
            self.record({'type':'native_auth_owner_read','refresh':refresh,'wall_s':time.monotonic()-started})
            return tokens
        except Exception:
            self._failed=True
            # Never expose a credential JSON parse error, native error body or
            # a chained exception through the task's authentication callback.
            raise RuntimeError('native auth owner could not supply access credentials') from None
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed=True
            self._account=None
            self.server.close()
