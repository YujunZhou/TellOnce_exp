"""Local full-input capacity checks, separate from transport byte limits.

Gemma uses its pinned official chat template and loads the tokenizer only when
a conservative normalized-byte bound cannot establish that the input fits. Native Codex does
not expose the server tokenizer: its text-only admission uses a conservative
UTF-8 byte bound plus explicit framing/output reserves, not a token estimate.
If that bound cannot establish a fit, native admission is left to the provider;
an upper bound alone cannot prove token overflow. Transport byte limits remain.
Neither path summarizes, truncates or sends the input to a counting service.
"""
import json
from pathlib import Path
from threading import Lock

from .provenance import file_hash


GEMMA_TOKENIZER_FILES = {
    'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'tokenizer_config.json': '9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633',
    'chat_template.jinja': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
}

BUNDLED_GEMMA_TOKENIZER = Path(__file__).resolve().parents[2] / 'vendor' / 'gemma4-tokenizer-842da3794eaa'
_GEMMA_COUNTER_LOCK = Lock()
_GEMMA_COUNTER_CACHE = None


def bundled_gemma_tokenizer_root():
    """Return the reviewed tokenizer shipped with the source bundle."""
    return BUNDLED_GEMMA_TOKENIZER


def _gemma_signature(root):
    if (not isinstance(root, Path) or not root.is_absolute() or root != root.resolve()):
        raise ValueError('capacity counter requires the pinned official local Gemma tokenizer')
    signature = [str(root)]
    for name, digest in GEMMA_TOKENIZER_FILES.items():
        path = root/name
        if path.is_symlink():
            raise ValueError('capacity counter requires the pinned official local Gemma tokenizer')
        stat = path.stat()
        # These two files total about 22 KiB. Check their bytes even on file
        # systems with coarse timestamps; avoid rereading the 32 MB vocabulary.
        if name != 'tokenizer.json' and file_hash(path) != digest:
            raise ValueError('capacity counter requires the pinned official local Gemma tokenizer')
        signature.append((name, digest, stat.st_dev, stat.st_ino, stat.st_size,
                          stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(signature)


class _GemmaCounter:
    """Small shared template state; the large vocabulary is loaded on demand."""

    def __init__(self, root, signature):
        self.root, self.signature = root, signature
        self.config = json.loads((root/'tokenizer_config.json').read_text())
        self.template = (root/'chat_template.jinja').read_text()
        self.tokenizer = None
        self.lock = Lock()

    def render(self, prompt, instructions):
        from transformers.utils.chat_template_utils import render_jinja_template
        messages = ([] if instructions is None else [{'role': 'system', 'content': instructions}])
        messages.append({'role': 'user', 'content': prompt})
        for thinking in (False, True):
            rendered, _ = render_jinja_template([messages], chat_template=self.template,
                add_generation_prompt=True, enable_thinking=thinking, bos_token=self.config['bos_token'])
            if any(message['content'].strip() not in rendered[0] for message in messages):
                raise ValueError('official template did not preserve complete role input')
            yield rendered[0]

    def upper_bound(self, prompt, instructions):
        # This pinned tokenizer uses space -> U+2581 normalization and byte-
        # fallback BPE, with no prefix/suffix insertion. Each normalized byte
        # needs at most one token; merges and special-token matches reduce it.
        # A space expands from one to three UTF-8 bytes. Bound the complete
        # rendered template, not just user text, and retain both thinking forms.
        bound = max(len(text.encode('utf-8')) + 2*text.count(' ')
                    for text in self.render(prompt, instructions))
        return bound, 'official_gemma_template_normalized_byte_upper_bound'

    def __call__(self, prompt, instructions):
        with self.lock:
            if self.tokenizer is None:
                if (_gemma_signature(self.root) != self.signature or
                        file_hash(self.root/'tokenizer.json') != GEMMA_TOKENIZER_FILES['tokenizer.json']):
                    raise ValueError('Gemma tokenizer files changed before exact counting')
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(self.root/'tokenizer.json'))
                tokenizer.no_truncation()
                tokenizer.no_padding()
                if _gemma_signature(self.root) != self.signature:
                    raise ValueError('Gemma tokenizer files changed while loading')
                self.tokenizer = tokenizer
        return (max(len(self.tokenizer.encode(text, add_special_tokens=False).ids)
                    for text in self.render(prompt, instructions)), 'official_gemma_chat_template_tokens')


def _gemma_counter(root):
    """Share one validated counter across roles, without retaining user input."""
    global _GEMMA_COUNTER_CACHE
    with _GEMMA_COUNTER_LOCK:
        signature = _gemma_signature(root)
        if _GEMMA_COUNTER_CACHE is not None and _GEMMA_COUNTER_CACHE[0] == signature:
            return _GEMMA_COUNTER_CACHE[1]
        if any(file_hash(root/name) != digest for name, digest in GEMMA_TOKENIZER_FILES.items()):
            raise ValueError('capacity counter requires the pinned official local Gemma tokenizer')
        count = _GemmaCounter(root, signature)
        if _gemma_signature(root) != signature:
            raise ValueError('Gemma tokenizer files changed while loading')

        # Bound the cache to the last tokenizer, not every path ever encountered.
        _GEMMA_COUNTER_CACHE = (signature, count)
        return count


class TextTokenCapacityExceeded(ValueError):
    retryable = False

    def __init__(self):
        super().__init__('complete_text_input_exceeds_model_capacity')


class TextCapacity:
    def __init__(self, *, model, context_window, output_reserve, framing_reserve, counter, record,
                 upper_counter=None, provider_checks_upper_bound=False):
        if (not isinstance(model, str) or not model or
                any(type(value) is not int or value <= 0 for value in (context_window, output_reserve, framing_reserve)) or
                output_reserve+framing_reserve >= context_window or not callable(counter) or not callable(record) or
                (upper_counter is not None and not callable(upper_counter))):
            raise ValueError('explicit model capacity, output/framing reserves and local counter required')
        self.model, self.context_window = model, context_window
        self.output_reserve, self.framing_reserve = output_reserve, framing_reserve
        self.counter, self.record = counter, record
        self.upper_counter = upper_counter
        self.provider_checks_upper_bound = provider_checks_upper_bound

    def __call__(self, prompt, *, base_instructions=None, context):
        if not isinstance(prompt, str) or (base_instructions is not None and not isinstance(base_instructions, str)):
            raise ValueError('full original text input required for capacity checking')
        counted, basis = ((self.upper_counter or self.counter)(prompt, base_instructions))
        if type(counted) is not int or counted < 0 or not isinstance(basis, str):
            raise ValueError('local counter returned an invalid capacity measurement')
        if self.upper_counter is not None and counted+self.output_reserve+self.framing_reserve > self.context_window:
            # An upper bound cannot reject an input. Preserve exact admission
            # for large inputs that fit after tokenization.
            counted, basis = self.counter(prompt, base_instructions)
            if type(counted) is not int or counted < 0 or not isinstance(basis, str):
                raise ValueError('local counter returned an invalid capacity measurement')
        required = counted+self.output_reserve+self.framing_reserve
        deferred = required > self.context_window and self.provider_checks_upper_bound
        self.record({'type': 'text_capacity', 'context': dict(context), 'model': self.model,
                     'basis': basis, 'input_count': counted, 'output_reserve': self.output_reserve,
                     'framing_reserve': self.framing_reserve, 'context_window': self.context_window,
                     'status': ('provider_check_required' if deferred else
                                'fits' if required <= self.context_window else 'unavailable')})
        if required > self.context_window and not deferred:
            raise TextTokenCapacityExceeded()
        return required

    @classmethod
    def gemma(cls, *, tokenizer_root: Path, output_reserve, record, model="google/gemma-4-31B-it"):
        if model not in {"google/gemma-4-31B-it", "google/gemma-4-31b-it"}:
            raise ValueError("official Gemma tokenizer requires the matching Gemma model")
        count = _gemma_counter(tokenizer_root)
        # Provider's published serving window, verified separately from the
        # tokenizer's unbounded model_max_length sentinel.
        return cls(model=model, context_window=262144,
                   output_reserve=output_reserve, framing_reserve=2048, counter=count, record=record,
                   upper_counter=count.upper_bound)

    @classmethod
    def native_text(cls, *, model, context_window, record):
        def count(prompt, instructions):
            return len(prompt.encode('utf-8'))+len((instructions or '').encode('utf-8')), 'utf8_byte_upper_bound'
        return cls(model=model, context_window=context_window, output_reserve=65536,
                   framing_reserve=16384, counter=count, record=record,
                   provider_checks_upper_bound=True)

    @classmethod
    def openai_text(cls, *, model, output_reserve, record):
        from .http_text_contract import LUNA_API_MODEL
        if model!=LUNA_API_MODEL:
            raise ValueError('unreviewed official API model capacity')
        def count(prompt,instructions):
            return len(prompt.encode('utf-8'))+len((instructions or '').encode('utf-8')), 'utf8_byte_upper_bound'
        return cls(model=model,context_window=1050000,output_reserve=output_reserve,
                   framing_reserve=16384,counter=count,record=record,provider_checks_upper_bound=True)

    @classmethod
    def openrouter_text(cls, *, model, output_reserve, record):
        from .http_text_contract import OPENROUTER_MODELS
        if model not in OPENROUTER_MODELS:
            raise ValueError('OpenRouter text capacity requires a reviewed model')
        # Reviewed serving windows, not the Gemma tokenizer.
        # UTF-8 byte count is a conservative text bound, not an exact token count.
        def count(prompt,instructions):
            return len(prompt.encode('utf-8'))+len((instructions or '').encode('utf-8')), 'utf8_byte_upper_bound'
        return cls(model=model,context_window=OPENROUTER_MODELS[model][2],
                   output_reserve=output_reserve,framing_reserve=16384,counter=count,record=record)
