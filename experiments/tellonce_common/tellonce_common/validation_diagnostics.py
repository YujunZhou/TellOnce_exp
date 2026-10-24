"""Allowlisted failure descriptions, without logging arbitrary provider content."""
import json


MESSAGES = (
    'delivery requirement must quote a cited user expectation',
    'incomplete current-user source coverage',
    'invalid attributed snippet reference', 'preference citation budget exceeded',
    'invalid indexed preference envelope', 'explicit preference eligibility required',
    'invalid reusable preference eligibility', 'invalid attributed span reference',
    'preference span outside observed document or exceeds excerpt bound',
    'empty preference evidence span', 'invalid excluded task-specific signal',
    'invalid preference evaluation envelope', 'invalid attributed preference assessment',
    'preference requirement cites an unknown or fabricated user source',
    'preference judgment cites unobserved evidence', 'pass/fail requires actual delivery evidence',
    'irreversible violation requires an observed completed operation', 'duplicate preference assessment',
)


def validation_failure(exc):
    if isinstance(exc, json.JSONDecodeError):
        return 'invalid_json'
    # Unknown validators/providers may include input data in exception strings.
    # Only these constant descriptions are admitted to the diagnostic record.
    message = str(exc)
    return message if type(exc) is ValueError and message in MESSAGES else 'validator_rejected'
