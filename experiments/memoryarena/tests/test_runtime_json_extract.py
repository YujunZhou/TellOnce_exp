import json

from memoryarena.claude_cli_runtime import _extract_json_object


def test_extract_json_preserves_already_escaped_latex_backslashes():
    raw = json.dumps(
        {
            "answer": "distribution\n\\[\n\\frac{1}{\\sigma}\n\\]",
            "constraint_notes": [],
            "unresolved_questions": [],
            "decision_log": "",
        }
    )

    parsed = _extract_json_object(raw)

    assert "\\[" in parsed["answer"]
    assert "\\sigma" in parsed["answer"]


def test_extract_json_repairs_single_latex_backslashes():
    raw = (
        '{"answer":"By \\(L\\)-smoothness, use '
        '\\\\[\\\\mathbb{E}[f(x)]\\\\]","constraint_notes":[],"unresolved_questions":[],'
        '"decision_log":""}'
    )

    parsed = _extract_json_object(raw)

    assert "\\(L\\)" in parsed["answer"]
    assert "\\mathbb" in parsed["answer"]


def test_extract_json_repairs_literal_newlines_inside_strings():
    raw = '{"answer":"Line one\nLine two","constraint_notes":[],"unresolved_questions":[],"decision_log":""}'

    parsed = _extract_json_object(raw)

    assert parsed["answer"] == "Line one\nLine two"


def test_extract_json_restores_valid_latex_escapes_that_json_treats_as_control_chars():
    raw = (
        '{"answer":"Use \\big and \\right in the derivation",'
        '"constraint_notes":[],"unresolved_questions":[],"decision_log":""}'
    )

    parsed = _extract_json_object(raw)

    assert "\\big" in parsed["answer"]
    assert "\\right" in parsed["answer"]


def test_extract_json_repairs_braced_unicode_escapes():
    raw = (
        '{"answer":"Use '
        + "\\u"
        + '{1d49c}={1} with \\sigma=4",'
        '"constraint_notes":[],"unresolved_questions":[],"decision_log":""}'
    )

    parsed = _extract_json_object(raw)

    assert "𝒜={1}" in parsed["answer"]
    assert "\\sigma=4" in parsed["answer"]
