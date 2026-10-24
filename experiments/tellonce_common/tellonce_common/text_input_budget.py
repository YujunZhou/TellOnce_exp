"""A byte limit is a transport bound, not a model token-window estimate.

Full role input is retained. Exceeding this bound never silently truncates a
profile, a task, or its history. Model token capacity is a separate admission
check; provider context errors remain failures, never satisfied feedback.
"""
import json

MAX_TEXT_INPUT_BYTES = 768 * 1024
MAX_SERIALIZED_TEXT_BYTES = 3 * 1024 * 1024


class TextInputBudgetExceeded(ValueError):
    retryable = False

    def __init__(self, actual_bytes, limit_bytes):
        super().__init__('text_input_budget_exceeded')
        self.actual_bytes = actual_bytes
        self.limit_bytes = limit_bytes


def validate_input_limit(limit):
    if type(limit) is not int or not 1024 <= limit <= MAX_TEXT_INPUT_BYTES:
        raise ValueError('text input requires an explicit bounded byte limit')


def check_text_input(*texts, limit=MAX_TEXT_INPUT_BYTES):
    validate_input_limit(limit)
    if any(not isinstance(text, str) for text in texts):
        raise ValueError('text input must contain text strings')
    actual = sum(len(text.encode('utf-8')) for text in texts)
    if actual > limit:
        raise TextInputBudgetExceeded(actual, limit)
    # The HTTP worker and native RPC each have a 4 MiB envelope bound. Escape
    # expansion of control characters must fail before reserving a request,
    # with room left for the surrounding fixed request metadata.
    serialized = sum(len(json.dumps(text, ensure_ascii=False).encode('utf-8')) for text in texts)
    if serialized > MAX_SERIALIZED_TEXT_BYTES:
        raise TextInputBudgetExceeded(serialized, MAX_SERIALIZED_TEXT_BYTES)
    return actual
