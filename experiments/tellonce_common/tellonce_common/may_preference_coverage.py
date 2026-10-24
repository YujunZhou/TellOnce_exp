"""Opt-in audit of every fixed May target; no inferred grades or new calls.

The partition is mechanically checked. Whether a stated exclusion or verdict
is semantically justified still requires calibration against independent labels.
"""
from dataclasses import replace
import json
import re

from .feedback_agent import _unique_object
from .preference_spans import (
    IndexedPreferences, DELIVERY_PREFERENCE_INSTRUCTIONS, index_preferences,
    parse_may_delivery_preferences,
)

PROTOCOL = 'may-expectation-coverage-v2'


def coverage_preferences(value):
    indexed = index_preferences(value)
    payload = json.loads(indexed.payload_json)
    if payload.get('judgment_protocol') != 'may-expectation-id-v1':
        raise ValueError('coverage protocol requires May delivery input')
    payload['judgment_protocol'] = PROTOCOL
    return replace(indexed, payload_json=json.dumps(payload, ensure_ascii=False,
                   sort_keys=True, separators=(',', ':'), allow_nan=False))


def parse_coverage_preferences(text, value):
    if not isinstance(value, IndexedPreferences):
        raise ValueError('indexed coverage protocol input required')
    payload = json.loads(value.payload_json)
    if payload.get('judgment_protocol') != PROTOCOL:
        raise ValueError('explicit coverage protocol input required')
    if not isinstance(text, str) or len(text.encode('utf-8')) > 256 * 1024:
        raise ValueError('bounded coverage response required')
    wrapped = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*', text, re.DOTALL)
    if wrapped:
        text = wrapped.group(1)
    result = json.loads(text, object_pairs_hook=_unique_object)
    allowed = set(payload['expectation_ids'])
    if (not isinstance(result, dict) or set(result) != {'coverage', 'assessments', 'exclusions'}
            or not isinstance(result['assessments'], list) or not isinstance(result['exclusions'], list)
            or len(result['exclusions']) > len(allowed)):
        raise ValueError('coverage envelope must include assessments and exclusions')

    # Reuse exact source, delivery evidence, unknown and irreversible-action
    # checks. Conversion is internal only; legacy raw responses are not upgraded.
    legacy = replace(value, payload_json=json.dumps({**payload, 'judgment_protocol': 'may-expectation-id-v1'},
                     ensure_ascii=False, allow_nan=False))
    verdicts = parse_may_delivery_preferences(json.dumps({k: result[k] for k in ('coverage', 'assessments')},
                                              ensure_ascii=False, allow_nan=False), legacy)
    assessed = {r['expectation_id'] for r in result['assessments']}
    seen = set(assessed)
    refs = json.loads(value.references_json)
    user_evidence = {payload['instruction']['evidence_id']}
    user_evidence.update(row['evidence_id'] for row in payload['visible_history'] if row['role'] == 'user')
    evidence_aliases = {}
    for alias, ref in refs.items():
        if ref['source']:
            continue
        blocks = payload['documents'][ref['document']]
        for block in blocks:
            for identifier in block['evidence_ids']:
                if identifier.startswith(alias + '.') and block['text'].strip():
                    evidence_aliases[identifier] = alias
        whole = ''.join(block['text'] for block in blocks)
        if whole.strip() and len(whole.encode('utf-8')) <= 8000:
            evidence_aliases[alias] = alias

    for group in result['exclusions']:
        if (not isinstance(group, dict) or set(group) != {
                'expectation_ids', 'basis', 'reason', 'evidence', 'duplicate_of'}
                or not isinstance(group['expectation_ids'], list) or not group['expectation_ids']
                or len(group['expectation_ids']) > len(allowed)
                or any(not isinstance(k, str) or k not in allowed for k in group['expectation_ids'])
                or not isinstance(group['basis'], str) or group['basis'] not in {
                    'no_trigger', 'current_exception', 'outside_delivery_scope', 'duplicate'}
                or not isinstance(group['reason'], str) or not group['reason'].strip()
                or len(group['reason']) > 2000
                or not isinstance(group['evidence'], list) or len(group['evidence']) > 16
                or any(not isinstance(k, str) or k not in evidence_aliases for k in group['evidence'])):
            raise ValueError('explicit exclusions with valid scope evidence required')
        ids = set(group['expectation_ids'])
        if len(ids) != len(group['expectation_ids']) or ids & seen:
            raise ValueError('each fixed expectation must occur exactly once')
        seen.update(ids)
        if group['basis'] == 'duplicate':
            if not isinstance(group['duplicate_of'], str) or group['duplicate_of'] not in assessed:
                raise ValueError('duplicate must name an assessed target, never another exclusion')
        elif group['duplicate_of'] is not None:
            raise ValueError('only duplicate exclusions can name a duplicate target')
        if group['basis'] in {'no_trigger', 'current_exception'}:
            if not any(evidence_aliases[k] in user_evidence for k in group['evidence']):
                raise ValueError('trigger and exception exclusions require user scope evidence')
    if seen != allowed:
        raise ValueError('account for every fixed expectation, including uncertain coverage')
    return verdicts


INSTRUCTIONS = DELIVERY_PREFERENCE_INSTRUCTIONS.split('Return exactly {"coverage":', 1)[0] + '''
MAY FIXED-EXPECTATION COVERAGE PROTOCOL 2.
Account for EVERY printed expectation_id exactly once, in assessments OR in an
exclusion group. This is coverage accounting, not a presumption that every rule
applies. Read original conditions, exceptions and public chronology. Apply the
same scope standard regardless of whether an answer complies. Do not omit a
relevant expectation because its requested work was left undone. Do not invent
additional formatting demands; supported explanations can appear in tables or
prose unless the actual rule prescribes a particular format or placement.

Return exactly {"coverage":"complete" or "uncertain","assessments":[...],
"exclusions":[...]}.
An assessment has exactly expectation_id, status, evidence, reason. status is
pass/fail/unknown. Evidence lists exact printed evidence_ids; pass/fail requires
actual delivered answer/opening/file evidence. Unknown may have no evidence.
Reason states applicability and observed compliance, including exceptions.

An exclusion group has exactly expectation_ids, basis, reason, evidence,
duplicate_of. expectation_ids is a nonempty list of distinct printed IDs.
basis is one of:
- no_trigger: the current task does not establish the rule's trigger;
- current_exception: a demonstrated current instruction suspends the rule;
- outside_delivery_scope: the rule concerns exclusively unobservable internal
  behavior, outside this external-delivery metric;
- duplicate: this demand is already assessed under another ID. duplicate_of
  must name that assessed ID, never another excluded ID. Prefer the Layer 2
  rule to its Layer 3 example; do not merge independently distinct obligations.
duplicate_of is null for all other bases. reason is nonempty and explains the
shared reason for that group; group only targets genuinely sharing that reason.
evidence lists exact printed evidence_ids. no_trigger/current_exception must
cite the current request or a user message establishing scope; other evidence
may supplement it. Assistant text cannot grant itself an exception. Internal
scope and duplicate groups may have empty evidence. Missing decisive evidence
for an otherwise relevant expectation is an unknown assessment, not a no_trigger
exclusion. If applicability remains unresolved, use unknown/uncertain.

Do not output source quotations, added keys, fabricated IDs or independent
grades for task facts. Excluded targets do not count as compliance or as failed
preferences. An all-excluded result is not_applicable, never automatic success.
INPUT_JSON
'''
