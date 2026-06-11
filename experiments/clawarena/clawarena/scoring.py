"""Response parsing, scoring, and correction generation."""
from __future__ import annotations

import json
import re
from typing import Any

from .conditions import ConditionState
from .schema import EXEC_CHECK, MULTI_CHOICE, Round


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    candidates = _json_object_candidates(text)
    required = {"answer", "rationale", "decision_log", "unresolved_questions"}
    for obj in candidates:
        if required.issubset(obj):
            return obj
    if candidates:
        return candidates[-1]

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    blob = text[start : end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", blob)
        obj = json.loads(repaired)
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    return obj


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        idx = match.start()
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text[idx:])
            try:
                obj, _ = decoder.raw_decode(repaired)
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            candidates.append(obj)
    return candidates


def score_response(
    round_obj: Round,
    parsed: dict[str, Any] | None,
    condition: ConditionState,
    *,
    exec_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if round_obj.round_type == EXEC_CHECK and exec_check is not None:
        return _score_exec_response(round_obj, parsed, condition, exec_check)

    if parsed is None:
        return _score_error(round_obj, "schema_error", "The response was not parseable JSON.")

    schema_errors = _schema_errors(round_obj, parsed)
    compliance_errors = _compliance_errors(parsed, condition)
    if schema_errors:
        return _score_error(round_obj, "schema_error", "; ".join(schema_errors))

    if round_obj.round_type == MULTI_CHOICE:
        pred = normalize_answer_letters(parsed.get("answer"))
        expected = sorted(set(round_obj.expected_answer))
        task_pass = sorted(set(pred)) == expected
        violation_type = None if task_pass and not compliance_errors else (
            "profile_or_enforcement_violation" if compliance_errors else "wrong_answer"
        )
        return {
            "scored": True,
            "score_kind": "multi_choice_exact",
            "task_pass": task_pass,
            "pref_pass": None,
            "combined_pass": task_pass,
            "predicted_answer": pred,
            "expected_answer": None,
            "expected_answer_hash": _answer_hash(expected),
            "n_corrections": 0,
            "correction_text": None,
            "violation_type": violation_type,
            "compliance_errors": compliance_errors,
        }

    if round_obj.round_type == EXEC_CHECK:
        return _score_error(round_obj, "exec_not_checked", "exec_check command was not run")

    return _score_error(round_obj, "unsupported_round_type", round_obj.round_type)


def normalize_answer_letters(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = re.findall(r"(?<![A-Z])[A-Z](?![A-Z])", value.upper())
    else:
        raw = []
    letters = []
    for item in raw:
        s = str(item).strip().upper()
        if re.fullmatch(r"[A-Z]", s):
            letters.append(s)
    return sorted(set(letters))


def build_correction_text(round_obj: Round, violation_type: str | None, detail: str) -> str:
    if violation_type == "wrong_answer":
        return (
            f"In round {round_obj.round_id}, you did not resolve the evidence "
            "carefully enough before choosing options. For future rounds, make "
            "an explicit evidence/conflict table and only select claims supported "
            "after resolving source reliability and staged updates."
        )
    if violation_type == "schema_error":
        return (
            f"In round {round_obj.round_id}, the output format broke the required "
            "JSON contract. For future rounds, return exactly one JSON object and "
            "use a list of option letters for multi-choice answers."
        )
    if violation_type == "preference_violation":
        return (
            f"In round {round_obj.round_id}, the workspace artifact violated an "
            f"already exposed durable user/project preference: {detail}. For future "
            "rounds, run the preference check mentally before finalizing artifacts."
        )
    if violation_type == "exec_artifact_failed":
        return (
            f"In round {round_obj.round_id}, the workspace artifact did not satisfy "
            "the public task requirements. For future exec-check rounds, inspect the "
            "target file paths, run or reason through the relevant validation command, "
            "and confirm latest workspace/session updates before finalizing."
        )
    if violation_type == "exec_not_checked":
        return (
            f"In round {round_obj.round_id}, the execution checker did not run. "
            "Treat this as invalid evidence and rerun the round before learning from it."
        )
    return (
        f"In round {round_obj.round_id}, you violated a durable project/user rule: "
        f"{detail}. For future rounds, apply the compiled rule before finalizing."
    )


def _score_error(round_obj: Round, violation_type: str, detail: str) -> dict[str, Any]:
    return {
        "scored": round_obj.round_type == MULTI_CHOICE,
        "score_kind": "multi_choice_exact" if round_obj.round_type == MULTI_CHOICE else "unscored_error",
        "task_pass": False if round_obj.round_type == MULTI_CHOICE else None,
        "pref_pass": None,
        "combined_pass": False if round_obj.round_type == MULTI_CHOICE else None,
        "predicted_answer": None,
        "expected_answer": None,
        "expected_answer_hash": _answer_hash(round_obj.expected_answer) if round_obj.round_type == MULTI_CHOICE else None,
        "n_corrections": 0,
        "correction_text": None,
        "violation_type": violation_type,
        "compliance_errors": [detail],
    }


def _schema_errors(round_obj: Round, parsed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["answer", "rationale", "decision_log", "unresolved_questions"]:
        if key not in parsed:
            errors.append(f"missing key {key}")
    if round_obj.round_type == MULTI_CHOICE:
        if not normalize_answer_letters(parsed.get("answer")):
            errors.append("answer must contain option letters")
    return errors


def _compliance_errors(parsed: dict[str, Any], condition: ConditionState) -> list[str]:
    errors: list[str] = []
    decision_log = str(parsed.get("decision_log") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    combined = f"{decision_log}\n{rationale}".lower()
    if not decision_log:
        errors.append("decision_log is empty")
    return errors


def _score_exec_response(
    round_obj: Round,
    parsed: dict[str, Any] | None,
    condition: ConditionState,
    exec_check: dict[str, Any],
) -> dict[str, Any]:
    compliance_errors = _compliance_errors(parsed, condition) if parsed is not None else []
    task_pass = bool(exec_check.get("task_pass"))
    pref_pass = exec_check.get("pref_pass")
    combined_pass = bool(exec_check.get("combined_pass"))
    violation_type = None
    correction_text = None
    if pref_pass is False:
        violation_type = "preference_violation"
        correction_text = build_correction_text(
            round_obj=round_obj,
            violation_type=violation_type,
            detail=safe_pref_feedback(round_obj),
        )
    elif compliance_errors:
        violation_type = "profile_or_enforcement_violation"
    elif not task_pass:
        violation_type = "exec_artifact_failed"
    return {
        "scored": True,
        "score_kind": "exec_check",
        "task_pass": task_pass,
        "pref_pass": pref_pass,
        "combined_pass": combined_pass,
        "predicted_answer": parsed.get("answer") if parsed else None,
        "expected_answer": None,
        "expected_answer_hash": None,
        "exec_check": exec_check,
        "n_corrections": 1 if correction_text else 0,
        "correction_text": correction_text,
        "violation_type": violation_type,
        "compliance_errors": compliance_errors,
        "response_schema_ok": parsed is not None and not _schema_errors(round_obj, parsed),
    }


def _answer_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=True)
    import hashlib

    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def correction_has_answer_leak(round_obj: Round, correction_text: str | None) -> bool:
    if not correction_text:
        return False
    text = correction_text.lower()
    if round_obj.round_type == MULTI_CHOICE:
        for answer in round_obj.expected_answer:
            letter = re.escape(str(answer).lower())
            patterns = [
                rf"\boption\s+{letter}\b",
                rf"\banswer\s+{letter}\b",
                rf"\bselect\s+{letter}\b",
                rf"\bchoose\s+{letter}\b",
                rf"\b{letter}\s+is\s+correct\b",
            ]
            if any(re.search(pattern, text) for pattern in patterns):
                return True
        feedback = json.dumps(round_obj.feedback, ensure_ascii=False).lower()
        if feedback and feedback in text:
            return True
    return False


def safe_pref_feedback(round_obj: Round) -> str:
    rule_ids = pref_rule_ids(round_obj)
    pref_feedback = (
        round_obj.pref_spec.get("feedback", {})
        if isinstance(round_obj.pref_spec, dict)
        else {}
    )
    if isinstance(pref_feedback, dict):
        text = str(pref_feedback.get("incorrect") or "").strip()
        if text:
            return _redact_pref_feedback(text, rule_ids)
    if rule_ids:
        return f"preference checker failed for rule group(s): {', '.join(rule_ids)}"
    return "the artifact violated a preference checker attached to this round"


def pref_rule_ids(round_obj: Round) -> list[str]:
    commands = [
        str(round_obj.pref_spec.get("command") or ""),
        str(round_obj.eval_spec.get("command") or ""),
    ]
    rules: list[str] = []
    for command in commands:
        for match in re.finditer(r"--rules\s+([A-Za-z0-9_, -]+)", command):
            raw = match.group(1).split()[0]
            rules.extend(part.strip().upper() for part in raw.split(",") if part.strip())
    return sorted(set(rules))


def _redact_pref_feedback(text: str, rule_ids: list[str]) -> str:
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"`[^`]+`", "`[PATH_OR_LITERAL]`", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?\b", "[DATE]", text)
    text = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", "[NUMBER]", text)
    prefix = f"preference checker failed for rule group(s): {', '.join(rule_ids)}. " if rule_ids else ""
    return prefix + text
