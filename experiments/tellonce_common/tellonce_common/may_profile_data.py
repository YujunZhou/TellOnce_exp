"""Select the simulated-user persona for the benchmark of this run.

Each benchmark has its own persona (profile_data_clawarena / profile_data_memoryarena).
The run driver sets TELLONCE_EXP_BENCHMARK before any worker starts.
"""
import os

_BENCHMARK = os.environ.get('TELLONCE_EXP_BENCHMARK')
if _BENCHMARK == 'clawarena':
    from . import profile_data_clawarena as _data
elif _BENCHMARK == 'memoryarena':
    from . import profile_data_memoryarena as _data
else:
    raise RuntimeError('set TELLONCE_EXP_BENCHMARK to clawarena or memoryarena')

MAY_PROFILE_VERSION = _data.PROFILE_VERSION
MAY_PREFIX_VOICE = _data.PREFIX_VOICE
MAY_PROFILES = _data.PROFILES
MAY_COOCCURRENCE = _data.COOCCURRENCE
