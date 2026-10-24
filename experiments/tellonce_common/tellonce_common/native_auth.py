"""Ephemeral native auth lease supplied by an admitted, single auth owner.

This module never reads an operator credential file, copies refresh tokens,
implements OAuth, or starts a browser login. The owner must enforce the time
budget when acquiring/refreshing tokens. Its implementation is a separate
review dependency. The documented external-token mode is experimental and
must be validated against the pinned Codex binary before live admission.
"""
from dataclasses import dataclass, field
import re
import threading
import time


@dataclass(frozen=True)
class AccessTokens:
    access_token: str = field(repr=False)
    account_id: str = field(repr=False)
    plan_type: str | None = None

    def __post_init__(self):
        if (not isinstance(self.access_token,str) or not 16 <= len(self.access_token) <= 32768 or
                not re.fullmatch(r'[A-Za-z0-9._~-]+',self.access_token) or
                not isinstance(self.account_id,str) or not 1 <= len(self.account_id) <= 512 or
                not re.fullmatch(r'[A-Za-z0-9_-]+',self.account_id) or
                (self.plan_type is not None and (not isinstance(self.plan_type,str) or
                    not re.fullmatch(r'[a-z0-9_]{1,64}',self.plan_type)))):
            raise ValueError('explicit bounded access-token material required')

    def payload(self):
        result={'accessToken':self.access_token,'chatgptAccountId':self.account_id}
        if self.plan_type is not None:
            result['chatgptPlanType']=self.plan_type
        return result


class NativeAuthLease:
    """One task's short-lived access tokens; the host retains refresh ownership.

    source(refresh=bool, previous_account_id=str|None, timeout_s=float) returns
    AccessTokens. It must be bound to the admitted account and fail closed on
    account changes. One native 401 refresh is allowed per logical turn.
    No token or account identifier is passed to the durable audit callback.
    """
    def __init__(self, *, server, source, record):
        if not callable(source) or not callable(record):
            raise ValueError('explicit admitted auth source and audit sink required')
        self.server,self.source,self.record=server,source,record
        self._lock=threading.RLock()
        self._secrets=set()
        self._account=None
        self._entered,self._closed=False,False
        self._turn=None
        self._seen=set()
        self._refreshes=0

    def _acquire(self, *, refresh, timeout_s):
        started=time.monotonic()
        try:
            tokens=self.source(refresh=refresh,previous_account_id=self._account,timeout_s=timeout_s)
            if not isinstance(tokens,AccessTokens):
                raise ValueError('auth source returned an unsupported value')
            # Retain redaction material even when a returned account is rejected.
            self._secrets.add(tokens.access_token)
            if time.monotonic()-started >= timeout_s or (self._account is not None and tokens.account_id!=self._account):
                raise ValueError('auth source exceeded its budget or changed account')
            self._account=tokens.account_id
            self.record({'type':'native_auth_acquired','refresh':refresh,'event_id':self._turn,
                         'wall_s':time.monotonic()-started})
            return tokens
        except Exception:
            raise RuntimeError('native authentication material is unavailable') from None

    def __enter__(self):
        with self._lock:
            if self._entered or self._closed:
                raise ValueError('native authentication lease cannot be reused')
            self._entered=True
            self.server.register_request_handler('account/chatgptAuthTokens/refresh',self._refresh)
            try:
                tokens=self._acquire(refresh=False,timeout_s=8)
                result=self.server.request('account/login/start',{'type':'chatgptAuthTokens',**tokens.payload()},timeout_s=10)
                if result!={'type':'chatgptAuthTokens'}:
                    raise ValueError('native external token mode was not honored')
                account=self.server.request('account/read',{'refreshToken':False},timeout_s=10)
                if not isinstance(account.get('account'),dict) or account['account'].get('type')!='chatgpt':
                    raise ValueError('native account did not retain external ChatGPT authentication')
                return self
            except Exception:
                self._closed=True
                raise RuntimeError('native authentication lease could not be established') from None

    def begin_turn(self, event_id):
        with self._lock:
            if (not self._entered or self._closed or not isinstance(event_id,str) or
                    not event_id or len(event_id)>512 or event_id in self._seen):
                raise ValueError('one live authentication boundary per native turn required')
            self._turn=event_id
            self._seen.add(event_id)
            self._refreshes=0

    def _refresh(self, params, *, timeout_s):
        with self._lock:
            if (not self._entered or self._closed or self._turn is None or self._refreshes>=1 or
                    not isinstance(params,dict) or set(params)-{'reason','previousAccountId'} or
                    params.get('reason')!='unauthorized' or params.get('previousAccountId')!=self._account):
                raise ValueError('unbound, foreign or repeated native auth refresh')
            self._refreshes+=1
            return self._acquire(refresh=True,timeout_s=min(8,timeout_s)).payload()

    def redact(self, text):
        with self._lock:
            for secret in sorted(self._secrets,key=len,reverse=True):
                text=text.replace(secret,'[REDACTED_ACCESS_TOKEN]')
            return text

    def __exit__(self, *args):
        # The task owner closes the native process first. External tokens are
        # in memory; account/logout must never touch the operator's auth owner.
        self._closed=True
        self.source=None
        self._account=None
