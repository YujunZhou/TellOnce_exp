"""Explicit interaction protocol; changing it requires a new test result owner."""

INTERACTIVE = 'interactive-v1'
PLAN_A = 'original-input-only-v1'
DELIVERY_RUBRIC = 'delivery-preferences-v1'


def validate_protocol(value, *, benchmark):
    if value not in {INTERACTIVE, PLAN_A}:
        raise ValueError('unknown test interaction protocol')
    if value == PLAN_A and benchmark not in {'clawarena', 'memoryarena'}:
        raise ValueError('plan A is limited to the two personalized benchmarks')
    return value


class NoTestFeedback:
    """An owned guard, not a model or a clarification-generating fallback."""
    def __init__(self, identity):
        self.identity=identity

    def __call__(self, *args, **kwargs):
        raise RuntimeError('plan A forbids simulated-user feedback and clarification')
