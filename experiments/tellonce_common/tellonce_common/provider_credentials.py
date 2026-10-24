"""Read an explicitly chosen provider key without changing process environment.

The experiment launcher pins the path/provider pairing, never the secret in a
public run manifest. Native Codex authentication uses NativeAuthOwner instead.
"""
import io
import os
from pathlib import Path
import re
import stat


class ProviderKeyFile:
    def __init__(self, path: Path, *, provider: str, authorize):
        if (not isinstance(path, Path) or not path.is_absolute() or path != path.resolve() or
                provider not in {'deepinfra', 'openai', 'openrouter'} or not callable(authorize)):
            raise ValueError('explicit canonical key file, provider and credential admission required')
        self.path, self.provider, self.authorize = path, provider, authorize

    def __call__(self):
        self.authorize('read_provider_key', {'provider': self.provider})
        try:
            from dotenv import dotenv_values
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                before = os.fstat(stream.fileno())
                if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or
                        before.st_nlink != 1 or before.st_size > 256*1024):
                    raise ValueError('invalid credential source')
                raw = stream.read(256*1024+1)
                after = os.fstat(stream.fileno())
                current = self.path.lstat()
            stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            if len(raw) > 256*1024 or stamp(before) != stamp(after) or stamp(before) != stamp(current):
                raise ValueError('credential source changed while reading')
            # No load_dotenv(), environment fallback, or variable interpolation.
            values = dotenv_values(stream=io.StringIO(raw.decode('utf-8')), interpolate=False)
            names = {'deepinfra':('DEEPINFRA_API_KEY','DeepInfra_API_KEY'),
                     'openai':('OPENAI_API_KEY',),'openrouter':('OPENROUTER_API_KEY',)}[self.provider]
            value = next((values[name] for name in names if values.get(name)), None)
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9._:/+=-]{8,4096}', value):
                raise ValueError('provider key missing or malformed')
            return value
        except (OSError, UnicodeError, ValueError, TypeError):
            raise RuntimeError('the explicitly selected provider credential is unavailable') from None
