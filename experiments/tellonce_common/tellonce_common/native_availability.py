"""Typed service failures; keep capacity, account quotas and task errors distinct."""
import math


NATIVE_AVAILABILITY_CODES = frozenset({
    'native_server_overloaded', 'native_quota_exhausted', 'native_network_unavailable',
    'native_auth_unavailable', 'native_startup_unavailable',
})


class NativeUnavailable(RuntimeError):
    def __init__(self, code, reset_at=None):
        if code not in NATIVE_AVAILABILITY_CODES:
            raise ValueError('unknown native availability category')
        self.code=code
        self.reset_at=reset_at if type(reset_at) in (int,float) and math.isfinite(reset_at) and reset_at>0 else None
        super().__init__(code)


def task_interruption(outcome):
    """Recover the first technical failure after the failed task receipt is saved."""
    if not outcome.aborted:
        raise ValueError('only an aborted task has an interruption')
    for unit in outcome.units:
        if unit.error_code and unit.error_code != 'prior_unit_aborted':
            if unit.error_code in NATIVE_AVAILABILITY_CODES:
                # Task outcomes store only the category, not a reset timestamp.
                return NativeUnavailable(unit.error_code)
            break
    return RuntimeError('native task was technically interrupted; retain evidence and stop this phase')


def availability_fields(error):
    """Expose safe categories and existing structured resets, never message guesses."""
    code = getattr(error, 'code', None)
    known = isinstance(code, str) and code in NATIVE_AVAILABILITY_CODES | {'provider_credit_exhausted'}
    fields = {'availability_code': code if known else None}
    if isinstance(error, NativeUnavailable) and error.reset_at is not None:
        fields['reset_at'] = error.reset_at
    return fields


def codex_failure(error):
    if not isinstance(error,dict):return None
    info=error.get('codexErrorInfo')
    name=info if isinstance(info,str) else next(iter(info),None) if isinstance(info,dict) else None
    if name=='serverOverloaded':return NativeUnavailable('native_server_overloaded')
    if name in {'usageLimitExceeded','rateLimitExceeded'}:return NativeUnavailable('native_quota_exhausted')
    if name in {'httpConnectionFailed','responseStreamConnectionFailed','responseStreamDisconnected'}:
        return NativeUnavailable('native_network_unavailable')
    # Some app-server versions wrap a transport disconnect as generic `other`.
    # Match the observed transport prefix, never arbitrary task/user error text.
    message = error.get('message')
    if (name == 'other' and isinstance(message, str) and message.startswith(
            'stream disconnected before completion: Transport error: network error:')):
        return NativeUnavailable('native_network_unavailable')
    # Codex app-server versions may expose an upstream 503 as generic `other`.
    # Match only the observed transport-level prefix; task text must not affect
    # availability classification.
    if (name == 'other' and isinstance(message, str) and message.startswith(
            'unexpected status 503 Service Unavailable: upstream connect error or disconnect/reset before headers.')):
        return NativeUnavailable('native_network_unavailable')
    return None
