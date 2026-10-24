"""Read-only Claude access leases; refresh tokens never enter experiment homes."""
import json
import math
import os
from pathlib import Path
import stat
import time
from .native_availability import NativeUnavailable


class ClaudeAuthUnavailable(NativeUnavailable):
    """Safe local auth reason; never preserve credential contents in messages."""
    def __init__(self,reason,*,remaining_lifetime_s=None,required_lifetime_s=None):
        if reason not in {'credential_unreadable','credential_invalid_store',
                          'credential_changed_during_read','credential_malformed',
                          'access_token_expired','access_token_insufficient_lifetime'}:
            raise ValueError('unknown local credential failure')
        super().__init__('native_auth_unavailable')
        self.auth_reason=reason
        self.remaining_lifetime_s=remaining_lifetime_s
        self.required_lifetime_s=required_lifetime_s
        self.args=('Claude native access lease unavailable: '+reason,)


class AccessTokenPipe:
    def __init__(self,token):
        if not isinstance(token,str) or not 8<=len(token.encode())<=4096 or '\n' in token:
            raise ValueError('bounded access token required')
        self._token=token;self.fd=None

    def __enter__(self):
        if self.fd is not None:raise ValueError('credential pipe is already open')
        read,write=os.pipe()
        try:
            data=self._token.encode();written=os.write(write,data)
            if written!=len(data):raise OSError('incomplete credential pipe write')
        except BaseException:
            os.close(read);raise
        finally:os.close(write)
        self.fd=read
        return self

    def redact(self,text):return text.replace(self._token,'[REDACTED]')

    def __exit__(self,*args):
        if self.fd is not None:os.close(self.fd);self.fd=None


class ClaudeCredentialSource:
    def __init__(self,*,path,authorize):
        if not isinstance(path,Path) or not path.is_absolute() or path!=path.resolve() or not callable(authorize):
            raise ValueError('one explicitly authorized native credential path required')
        self.path,self.authorize=path,authorize

    def __call__(self,*,minimum_lifetime_s):
        if type(minimum_lifetime_s) not in (int,float) or not 0<minimum_lifetime_s<=86400:
            raise ValueError('bounded requested credential lifetime required')
        self.authorize('read_claude_access_token',{})
        reason='credential_unreadable'
        try:
            fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as stream:
                before=os.fstat(stream.fileno())
                if (not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or
                        before.st_nlink!=1 or before.st_size>65536):
                    raise ClaudeAuthUnavailable('credential_invalid_store')
                data=stream.read(65537);after=os.fstat(stream.fileno());current=self.path.lstat()
            stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            if len(data)>65536 or stamp(before)!=stamp(after) or stamp(after)!=stamp(current):
                raise ClaudeAuthUnavailable('credential_changed_during_read')
            reason='credential_malformed'
            value=json.loads(data)['claudeAiOauth'];expires=value['expiresAt']
            if type(expires) not in (int,float) or not math.isfinite(expires):
                raise ClaudeAuthUnavailable('credential_malformed')
            remaining=expires/1000-time.time();required=minimum_lifetime_s+60
            if remaining<required:
                raise ClaudeAuthUnavailable('access_token_expired' if remaining<=0 else
                    'access_token_insufficient_lifetime',remaining_lifetime_s=remaining,
                    required_lifetime_s=required)
            return AccessTokenPipe(value['accessToken'])
        except ClaudeAuthUnavailable as exc:
            raise exc from None
        except Exception:
            raise ClaudeAuthUnavailable(reason) from None
