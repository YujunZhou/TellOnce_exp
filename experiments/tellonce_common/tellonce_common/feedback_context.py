"""Public task projection for May feedback and explicit legacy rolling callers."""
from .compact_profiles import PROFILE_VERSION, feedback_profile_projection
from copy import deepcopy
import json
from .feedback_evidence import public_context, select_evidence, prior_user_requests
from .task_artifacts import referenced_output_files

VERSION = 'may-full-public-context-evidence-scope-v3'
SUMMARY_MAX_CHARS = 2000
SUMMARY_MAX_UTF8_BYTES = 2400


def validate_summary(text, *, allow_empty=True):
    if (not isinstance(text, str) or len(text) > SUMMARY_MAX_CHARS or
            len(text.encode('utf-8')) > SUMMARY_MAX_UTF8_BYTES or
            (not allow_empty and not text.strip())):
        raise ValueError('feedback summary must be short plain text within the character/byte guard')
    return text


def compact_feedback_view(payload):
    if not isinstance(payload, dict):
        raise ValueError('reviewed feedback payload required')
    instruction, delivery = payload.get('instruction', ''), payload.get('delivery', '')
    if not isinstance(instruction, str) or not isinstance(delivery, str):
        raise ValueError('full feedback question and delivery must be text')
    history = payload.get('visible_history', [])
    if not isinstance(history, list) or any(not isinstance(row, dict) or
            row.get('role') not in {'user', 'assistant'} or not isinstance(row.get('text'), str) for row in history):
        raise ValueError('feedback history requires attributed visible messages')
    previous = payload.get('previous_feedback')
    if previous is not None and (not isinstance(previous, dict) or set(previous) != {'status', 'message'} or
            not isinstance(previous['status'], str) or previous['status'] not in {'correction', 'satisfied', 'cannot_judge'} or not isinstance(previous['message'], str)):
        raise ValueError('previous feedback must contain only its full status and message')
    pending = payload.get('unreviewed_outputs', [])
    if not isinstance(pending, list) or any(not isinstance(row, dict) or
            set(row) != {'instruction', 'user_request', 'delivery'} or
            any(not isinstance(text, str) for text in row.values()) for row in pending):
        raise ValueError('unsummarized deliveries must contain only full public questions and outputs')
    latest = next((row['text'] for row in reversed(history) if row['role'] == 'user'), instruction)
    result = {'feedback_view': VERSION, 'instruction': instruction, 'delivery': delivery,
        'previous_output_summary': validate_summary(payload.get('previous_output_summary', '')),
        'previous_feedback': payload.get('previous_feedback'),
        'unreviewed_outputs': payload.get('unreviewed_outputs', [])}
    if latest != instruction:
        result['latest_user_request'] = latest
    for key in ('task_id', 'unit_id'):
        if key in payload:
            result[key] = payload[key]
    profile = payload.get('user_profile')
    is_may = isinstance(profile, dict) and profile.get('kind') == PROFILE_VERSION
    if profile is not None:
        result['user_profile'] = feedback_profile_projection(profile)
    if is_may:
        # The restored May simulator reads actual same-task messages. A previous
        # model summary can be wrong and is not an alternative evidence source.
        result.pop('previous_output_summary')
        result['visible_history'] = deepcopy(history)
        dialogue = payload.get('observed_operations', {}).get('public_dialogue')
        if dialogue is not None:
            if (not isinstance(dialogue, dict) or dialogue.get('schema') != 1 or
                    not isinstance(dialogue.get('messages'), list) or
                    not isinstance(dialogue.get('coverage'), dict) or
                    any(not isinstance(row, dict) or not isinstance(row.get('text'), str)
                        for row in dialogue['messages'])):
                raise ValueError('captured public dialogue requires explicit messages and coverage')
            result['visible_commentary'] = deepcopy(dialogue)
        else:
            result['visible_commentary'] = {'messages': [], 'coverage': {'complete': False},
                                             'reason': 'public opening not captured'}
        result['visible_commentary_scope'] = {
            'version': 1,
            'complete_describes': 'No recorded size or tool-ordering omissions within the bounded opening projection; not complete public-communication visibility.',
            'before_first_tool_describes': 'Before the first observed tool event, not necessarily before actual execution when the start event is missing.',
            'unknown_phase_visibility': 'Unphased text without a later tool may be absent; visibility is not established.',
            'history_scope': 'This projection covers the current user event only; earlier delivered messages are in visible_history when provided.',
            'identity_limit': 'Native message notifications are not deduplicated by item ID.'}
    if 'training_reference' in payload:
        # Only the caller may authorize training references. No new access.
        if is_may:
            raise ValueError('personal May feedback must never receive a reference answer')
        result['training_reference'] = payload['training_reference']
    context = public_context(payload)
    if context: result['public_context'] = context
    # Workspace questions can refer to earlier instructions that a rolling
    # answer summary dropped. This is literal context, not learned memory.
    if 'options' in context:
        prior = prior_user_requests(history, instruction + '\n' + latest + '\n' +
                                    json.dumps(context['options'], ensure_ascii=False))
        if prior['total_messages']:
            result['prior_user_requests'] = prior
    evidence_payload={**payload, 'latest_user_request': latest}
    evidence = select_evidence(evidence_payload)
    if is_may:
        artifacts = payload.get('visible_artifacts', {}).get('files', {})
        delivered = referenced_output_files(delivery, artifacts)
        if delivered:
            # Current explicitly delivered files are the answer, not optional
            # supporting excerpts. The transport rejects excess input instead
            # of silently treating a truncated answer as the complete output.
            result['current_delivered_files'] = {path: artifacts[path] for path in delivered}
            evidence['excerpts'] = [row for row in evidence['excerpts']
                if not (row['kind'] == 'delivered_file' and row['source'] in delivered)]
            evidence['selected_chunks'] = len(evidence['excerpts'])
            evidence['delivery_selection'].update(referenced_files=delivered,
                fully_included=delivered, incomplete=[], referenced_paths_omitted=0,
                note='Full captured current delivery is in current_delivered_files; capture omissions remain unknown.')
    if (evidence['candidate_chunks'] or evidence['source_inventory']['available_paths'] or
            evidence['delivery_selection']['referenced_files'] or 'observed_operations' in payload):
        result['public_evidence'] = evidence
    if 'public_lookup' in payload: result['public_lookup'] = payload['public_lookup']
    return result
