"""Owner-bound request/turn attribution and bounded public operation indexes.

This module never discovers identities from events or evaluates operation bodies.
The task owner binds native turns; snapshots and checkpoints preserve that proof.
"""
from copy import deepcopy
import json

from .provenance import canonical_json_bytes

IDENTITY_MAX_BYTES = 256
STATE_MAX_BYTES = 65536
SNAPSHOT_MAX_BYTES = 68 * 1024
INDEX_MAX_BYTES = 4096
ORIGINS = frozenset({'codex_turn_start_reply', 'claude_adapter_request_turn'})
RELATIONS = ('current', 'unknown', 'historical')
OPERATION_TYPES = frozenset({'claudeToolCall', 'commandExecution', 'fileChange',
                             'mcpToolCall', 'dynamicToolCall'})
SCOPE_INSTRUCTIONS = """Operation scope: current/historical/unknown describes the captured
request/turn binding, not operation success. Historical operations remain usable
as prior evidence; do not attribute them to this request. Unknown is not current.
operation_coverage describes bounded task history, not the current turn. An empty
current index does not prove zero current tools. Index omissions are explicit;
an index without an excerpt does not disclose the operation's unseen contents.
"""


def _identity(value):
    if (not isinstance(value, str) or not value or
            len(value.encode('utf-8')) > IDENTITY_MAX_BYTES):
        raise ValueError('operation scope requires a bounded explicit identity')


def _positive(value):
    return type(value) is int and value > 0


def _bindings(value, counter):
    if not isinstance(value, list):
        raise ValueError('operation scope bindings must be an ordered list')
    turns, events, last = set(), set(), 0
    for row in value:
        if not isinstance(row, dict) or set(row) != {'turn_id', 'event_id', 'ordinal', 'origin'}:
            raise ValueError('operation scope binding fields differ from the protocol')
        _identity(row['turn_id']); _identity(row['event_id'])
        if (not _positive(row['ordinal']) or not last < row['ordinal'] <= counter or
                row['turn_id'] in turns or row['event_id'] in events or
                not isinstance(row['origin'], str) or row['origin'] not in ORIGINS):
            raise ValueError('operation scope contains conflicting or unordered bindings')
        turns.add(row['turn_id']); events.add(row['event_id']); last = row['ordinal']


class RequestScopeState:
    """Mutable task-owner ledger; the recorder provides its lock."""
    def __init__(self, saved=None):
        self.counter, self.bindings, self.active = 0, [], None
        if saved is None:
            return
        if (not isinstance(saved, dict) or set(saved) != {'version', 'counter', 'bindings'} or
                type(saved['version']) is not int or saved['version'] != 1 or
                type(saved['counter']) is not int or saved['counter'] < 0):
            raise ValueError('invalid operation scope checkpoint')
        _bindings(saved['bindings'], saved['counter'])
        self._budget(saved)
        self.counter, self.bindings = saved['counter'], deepcopy(saved['bindings'])

    @staticmethod
    def _budget(value):
        if len(canonical_json_bytes(value)) > STATE_MAX_BYTES:
            raise ValueError('operation scope checkpoint budget exceeded')

    def export(self):
        return {'version': 1, 'counter': self.counter, 'bindings': deepcopy(self.bindings)}

    def begin(self, event_id):
        _identity(event_id)
        if ((self.active is not None and self.active['event_id'] == event_id) or
                any(row['event_id'] == event_id for row in self.bindings)):
            raise ValueError('operation scope request was duplicated')
        future = self.export(); future['counter'] += 1
        self._budget(future)
        self.counter = future['counter']
        self.active = {'event_id': event_id, 'ordinal': self.counter, 'status': 'unavailable',
                       'turn_id': None, 'origin': None}

    def bind(self, event_id, turn_id, *, origin):
        _identity(event_id); _identity(turn_id)
        if (self.active is None or self.active['event_id'] != event_id or
                self.active['status'] != 'unavailable' or not isinstance(origin, str) or
                origin not in ORIGINS or any(row['turn_id'] == turn_id for row in self.bindings)):
            raise ValueError('operation scope binding is foreign, repeated or conflicting')
        row = {'event_id': event_id, 'turn_id': turn_id,
               'ordinal': self.active['ordinal'], 'origin': origin}
        future = self.export(); future['bindings'].append(row)
        self._budget(future)
        self.bindings.append(row)
        self.active.update(status='bound', turn_id=turn_id, origin=origin)

    def freeze(self, event_id, events):
        # Older recorder-only callers may capture without begin_request. Their
        # legacy snapshots remain unbound; no identity is inferred from events.
        if self.active is None:
            return None
        if self.active['event_id'] != event_id:
            raise ValueError('operation scope capture belongs to another request')
        turns = {row['native_turn_id'] for row in events}
        value = {'version': 1, 'current_request': deepcopy(self.active),
                 'turn_bindings': [deepcopy(row) for row in self.bindings
                    if row['turn_id'] in turns or row['turn_id'] == self.active['turn_id']]}
        validate_request_scope(value, event_id=event_id, events=events)
        return value


def validate_request_scope(value, *, event_id, events):
    if (not isinstance(value, dict) or set(value) != {'version', 'current_request', 'turn_bindings'} or
            type(value['version']) is not int or value['version'] != 1 or
            len(canonical_json_bytes(value)) > SNAPSHOT_MAX_BYTES):
        raise ValueError('invalid bounded operation scope snapshot')
    current = value['current_request']
    if (not isinstance(current, dict) or
            set(current) != {'event_id', 'ordinal', 'status', 'turn_id', 'origin'} or
            current['event_id'] != event_id or not _positive(current['ordinal']) or
            not isinstance(current['status'], str) or current['status'] not in {'bound', 'unavailable'}):
        raise ValueError('operation scope current request differs from its capture')
    _identity(current['event_id'])
    _bindings(value['turn_bindings'], current['ordinal'])
    if (not isinstance(events, list) or any(not isinstance(row, dict) or
            not isinstance(row.get('native_turn_id'), str) or not row['native_turn_id'] for row in events)):
        raise ValueError('operation scope requires attributed native events')
    turns = {row['native_turn_id'] for row in events}
    current_rows = []
    for row in value['turn_bindings']:
        if ((row['event_id'] == event_id) != (row['ordinal'] == current['ordinal']) or
                (row['turn_id'] not in turns and row['turn_id'] != current['turn_id'])):
            raise ValueError('operation scope binding is unrelated to its capture')
        if row['ordinal'] == current['ordinal']:
            current_rows.append(row)
    if current['status'] == 'unavailable':
        if current['turn_id'] is not None or current['origin'] is not None or current_rows:
            raise ValueError('unavailable operation scope cannot assert a current identity')
    else:
        _identity(current['turn_id'])
        expected = {k: current[k] for k in ('event_id', 'turn_id', 'ordinal', 'origin')}
        if current_rows != [expected]:
            raise ValueError('operation scope current binding is missing or conflicting')


def _relation(scope, turn_id):
    if scope is None or scope['current_request']['status'] != 'bound':
        return 'unknown'
    current = scope['current_request']
    if turn_id == current['turn_id']:
        return 'current'
    prior = next((row for row in scope['turn_bindings'] if row['turn_id'] == turn_id), None)
    return 'historical' if prior and prior['ordinal'] < current['ordinal'] else 'unknown'


def operation_index(operations, *, excluded):
    """Return bounded metadata and sequence -> included operation reference.

    No invocation/result text or unbounded native/item IDs enter the index.
    The source digest plus exact sequence resolves every reference losslessly.
    """
    if not isinstance(operations, dict) or not isinstance(operations.get('events'), list):
        raise ValueError('operation index requires captured native events')
    digest = operations.get('source_hash')
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('operation index requires the immutable source hash')
    scope = operations.get('request_scope')
    if 'request_scope' in operations:
        validate_request_scope(scope, event_id=operations.get('event_id'), events=operations['events'])
    groups, last = {}, 0
    for event in operations['events']:
        if (not isinstance(event, dict) or set(event) != {'sequence', 'native_turn_id', 'event', 'item'} or
                not _positive(event['sequence']) or event['sequence'] <= last or
                not isinstance(event['native_turn_id'], str) or not event['native_turn_id'] or
                not isinstance(event['event'], str) or event['event'] not in {'started', 'completed'} or
                not isinstance(event['item'], dict)):
            raise ValueError('operation index requires an ordered native event sequence')
        last = event['sequence']; item = event['item']
        if (not isinstance(item.get('id'), str) or not item['id'] or
                not isinstance(item.get('type'), str) or item['type'] not in OPERATION_TYPES):
            raise ValueError('operation index requires native operation identities')
        if excluded(item):
            continue
        key = event['native_turn_id'], item['id']
        if key not in groups:
            groups[key] = {'ref': 'op:' + str(event['sequence']),
                           'relation': _relation(scope, event['native_turn_id']),
                           'type': item['type'], 'events': []}
        row = groups[key]
        if row['type'] != item['type'] or any(e['event'] == event['event'] for e in row['events']):
            raise ValueError('operation index contains conflicting operation events')
        if row['events'] and event['event'] == 'started':
            raise ValueError('operation start follows its completion')
        row['events'].append({'sequence': event['sequence'], 'event': event['event']})
    rows = list(groups.values())
    counts = {relation: sum(row['relation'] == relation for row in rows) for relation in RELATIONS}
    result = {'version': 1, 'capture_ref': digest, 'capture_scope': 'bounded_task_history_at_delivery',
              'current_binding_status': scope['current_request']['status'] if scope is not None else 'legacy',
              'current_turn_completeness': 'not_established', 'captured_counts': counts,
              'represented_counts': dict.fromkeys(RELATIONS, 0), 'omitted_counts': dict(counts),
              'index_complete': not rows, 'operations': []}
    def size(value):
        return len(json.dumps(value, ensure_ascii=False).encode('utf-8'))
    if size(result) > INDEX_MAX_BYTES:
        raise ValueError('mandatory operation scope metadata exceeds its budget')
    for row in sorted(rows, key=lambda row: (RELATIONS.index(row['relation']), -row['events'][-1]['sequence'])):
        trial = deepcopy(result)
        trial['operations'].append(row)
        trial['represented_counts'][row['relation']] += 1
        trial['omitted_counts'][row['relation']] -= 1
        trial['index_complete'] = len(trial['operations']) == len(rows)
        if size(trial) <= INDEX_MAX_BYTES:
            result = trial
    references = {event['sequence']: row['ref'] for row in result['operations'] for event in row['events']}
    return result, references
