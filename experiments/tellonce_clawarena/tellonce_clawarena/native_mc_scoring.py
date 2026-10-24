"""ClawArena's documented boxed-answer exact-match score and diagnostics.

The pinned public data release includes the scoring specification, but not
the original qtype package. The extraction convention uses the last answer
marker, so an explicit final revision supersedes earlier reasoning candidates.
No old JSON overlay, preference checker or feedback template is involved.
"""
from dataclasses import dataclass
import re


ANSWER_MARKER = re.compile(r'\\(?:bbox|boxed)\s*\{')


@dataclass(frozen=True)
class MCScore:
    score: float
    passed: bool
    predicted: tuple[str, ...]
    iou: float
    precision: float
    recall: float
    f1: float
    extraction: str


def extract_multiple_choice(text: str) -> tuple[tuple[str, ...], str]:
    """Extract public delivery syntax without loading or fabricating an answer key.

    A parsed answer is not necessarily correct. This is also usable by task
    diagnostics to distinguish a delivered answer from a plan awaiting approval.
    """
    if not isinstance(text, str):
        raise ValueError('public answer extraction requires text')
    markers = list(ANSWER_MARKER.finditer(text))
    predicted = set()
    extraction = 'missing_marker'
    if markers:
        contents = re.match(r'([^{}]*)\}', text[markers[-1].end():])
        answer = contents[1].strip().upper() if contents else ''
        if re.fullmatch(r'[A-Z](?:\s*[,，]\s*[A-Z])*', answer):
            predicted = set(re.findall('[A-Z]', answer))
            extraction = 'parsed'
        else:
            extraction = 'invalid_marker'
    return tuple(sorted(predicted)), extraction


def score_multiple_choice(text: str, *, expected: list[str], options: dict[str, str]) -> MCScore:
    if (not isinstance(text, str) or not isinstance(expected, list) or not expected or
            any(not isinstance(x, str) or not re.fullmatch('[A-Z]', x) for x in expected) or
            not isinstance(options, dict) or not options or
            any(not isinstance(k, str) or not re.fullmatch('[A-Z]', k) or not isinstance(v, str)
                for k, v in options.items()) or not set(expected) <= set(options)):
        raise ValueError('native multiple-choice scoring requires valid private answers and public options')
    labels, extraction = extract_multiple_choice(text)
    predicted = set(labels)
    gold = set(expected)
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    passed = predicted == gold
    return MCScore(float(passed), passed, tuple(sorted(predicted)), overlap / len(predicted | gold),
                   precision, recall, f1, extraction)
