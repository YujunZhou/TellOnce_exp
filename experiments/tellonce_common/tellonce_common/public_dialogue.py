"""Small public opening-message projection, captured with an existing delivery.

Never records reasoning or final candidates rejected by internal checks. It has
no files, timer or network client; the existing evidence snapshot owns storage.
"""
from copy import deepcopy

MAX_BYTES = 12000


def validate_public_dialogue(value, event_id):
    if (not isinstance(value, dict) or set(value) != {'schema', 'event_id', 'messages', 'coverage'}
            or type(value['schema']) is not int or value['schema'] != 1 or value['event_id'] != event_id
            or not isinstance(value['messages'], list)):
        raise ValueError('public dialogue must belong to the current delivery')
    coverage = value['coverage']
    if (not isinstance(coverage, dict) or set(coverage) != {'complete', 'omitted_messages'}
            or type(coverage['complete']) is not bool or type(coverage['omitted_messages']) is not int
            or coverage['omitted_messages'] < 0
            or coverage['complete'] != (coverage['omitted_messages'] == 0)):
        raise ValueError('public dialogue coverage must be explicit')
    used = 0
    for i, row in enumerate(value['messages']):
        if (not isinstance(row, dict) or set(row) != {'sequence', 'text', 'before_first_tool'}
                or type(row['sequence']) is not int or row['sequence'] != i + 1
                or not isinstance(row['text'], str) or not row['text'].strip()
                or type(row['before_first_tool']) is not bool):
            raise ValueError('public dialogue contains non-public or unordered fields')
        used += len(row['text'].encode('utf-8'))
    if used > MAX_BYTES:
        raise ValueError('public dialogue exceeded its small fixed bound')
    return value


class PublicDialogue:
    def __init__(self, event_id):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('public dialogue requires a user event')
        self.event_id = event_id
        self.messages, self.pending = [], []
        self.used = self.pending_bytes = self.omitted = self.pending_omitted = 0
        self.tool_seen = False
        self.sealed = False

    def _append(self, text):
        size = len(text.encode('utf-8'))
        if self.used + size > MAX_BYTES:
            self.omitted += 1
            return
        self.messages.append({'sequence': len(self.messages) + 1, 'text': text,
                              'before_first_tool': not self.tool_seen})
        self.used += size

    def observe_text(self, text, *, commentary=False):
        if self.sealed or not isinstance(text, str) or not text.strip():
            return
        if commentary:
            self._append(text)
        elif self.used + self.pending_bytes + len(text.encode('utf-8')) <= MAX_BYTES:
            # Providers without explicit phases identify a public preamble by
            # its following real tool call. A stop candidate is discarded below.
            self.pending.append(text)
            self.pending_bytes += len(text.encode('utf-8'))
        else:
            self.pending_omitted += 1

    def observe_tool(self):
        if self.sealed:
            return
        for text in self.pending:
            self._append(text)
        self.omitted += self.pending_omitted
        self.pending.clear()
        self.pending_bytes = self.pending_omitted = 0
        self.tool_seen = True

    def capture(self):
        # Final text is already provided as the actual SolverAnswer. A blocked
        # earlier final candidate must never become public preference evidence.
        self.pending.clear()
        self.pending_bytes = self.pending_omitted = 0
        # Internal checks may request repairs after this candidate. Their
        # explanations must not become a different user-visible opening.
        # A new actual user request constructs a fresh PublicDialogue instead.
        self.sealed = True
        return deepcopy({'schema': 1, 'event_id': self.event_id, 'messages': self.messages,
                         'coverage': {'complete': self.omitted == 0, 'omitted_messages': self.omitted}})
