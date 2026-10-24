"""Reversible display aliases for explicitly owned workspace path prefixes.

This hides path metadata only. Ordinary mentions of methods in task data are
preserved. Validators continue to use the original indexed evidence, not this
presentation, so IDs still resolve to exact captured strings.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re

from .feedback_agent import _unique_object

VERSION = 'owned-workspace-path-alias-v1'


@dataclass(frozen=True)
class BlindedPreferences:
    payload_json: str
    audit: dict


def _pattern(prefix):
    return re.compile(r'(?<![A-Za-z0-9_./%~\-])' + re.escape(prefix) + r'(?=/|$|[\s:<>\)\]\}\"\x27,;])')


def _map_strings(value, transform):
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, list):
        return [_map_strings(v, transform) for v in value]
    if isinstance(value, dict):
        pairs = [(transform(k), _map_strings(v, transform)) for k, v in value.items()]
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError('workspace alias key collision')
        return dict(pairs)
    return value


def restore_workspace_paths(projection):
    value = json.loads(projection.payload_json, object_pairs_hook=_unique_object)
    mappings = projection.audit['mappings']
    def restore(text):
        for entry in mappings:
            text = _pattern(entry['alias']).sub(lambda _: entry['source_root'], text)
        return text
    lengths = projection.audit.get('document_segment_lengths', {})
    documents = value.pop('documents') if lengths else None
    restored = _map_strings(value, restore)
    if documents is not None:
        restored['documents'] = {}
        for name, blocks in documents.items():
            text = restore(''.join(block['text'] for block in blocks))
            if sum(lengths[name]) != len(text):
                raise ValueError('restored document lengths differ')
            rebuilt, offset = [], 0
            for block, size in zip(blocks, lengths[name], strict=True):
                fields = _map_strings({k: v for k, v in block.items() if k != 'text'}, restore)
                rebuilt.append({**fields, 'text': text[offset:offset+size]})
                offset += size
            restored['documents'][name] = rebuilt
    return restored


def blind_workspace_paths(payload_json, roots):
    if (not isinstance(roots, tuple) or not roots or len(roots) > 16
            or any(not isinstance(r, str) or not r.startswith('/')
                   or str(PurePosixPath(r)) != r or len(PurePosixPath(r).parts) < 4
                   or '..' in PurePosixPath(r).parts for r in roots)
            or len(set(roots)) != len(roots)
            or any(a != b and b.startswith(a + '/') for a in roots for b in roots)):
        raise ValueError('distinct bounded nonoverlapping absolute workspace roots required')
    value = json.loads(payload_json, object_pairs_hook=_unique_object)
    mappings = [{'source_root': root, 'alias': '/__audit_workspace_' + str(i) + '__'}
                for i, root in enumerate(roots, 1)]
    count = 0
    def hide(text, *, boundaries=None):
        nonlocal count
        if any(entry['alias'] in text for entry in mappings):
            raise ValueError('workspace alias collision in original input')
        changes = []
        for entry in mappings:
            changes.extend((m.start(), m.end(), entry['alias'])
                           for m in _pattern(entry['source_root']).finditer(text))
        changes.sort()
        pieces, offset = [], 0
        for start, end, alias in changes:
            if start < offset:
                raise ValueError('overlapping workspace path matches')
            pieces.extend((text[offset:start], alias)); offset = end
        pieces.append(text[offset:])
        count += len(changes)
        displayed = ''.join(pieces)
        if boundaries is None:
            return displayed
        def position(old):
            delta = 0
            for start, end, alias in changes:
                if old <= start:
                    break
                if old < end:
                    return start + delta + len(alias)
                delta += len(alias) - (end-start)
            return old + delta
        return displayed, [position(b) for b in boundaries]

    lengths = {}
    # The lossless index can split an absolute path between arbitrary text
    # segments. Redact its joined document, then map segment boundaries; hiding
    # each segment independently leaves method names visible in split prefixes.
    documents = value.get('documents')
    base = {k: v for k, v in value.items() if k != 'documents'} if isinstance(documents, dict) else value
    displayed_value = _map_strings(base, hide)
    if isinstance(documents, dict):
        displayed_value['documents'] = {}
        for name, blocks in documents.items():
            lengths[name] = [len(block['text']) for block in blocks]
            boundaries = [0]
            for size in lengths[name]:
                boundaries.append(boundaries[-1]+size)
            displayed, mapped = hide(''.join(block['text'] for block in blocks), boundaries=boundaries)
            displayed_value['documents'][name] = [
                {**_map_strings({k: v for k, v in block.items() if k != 'text'}, hide),
                 'text': displayed[mapped[i]:mapped[i+1]]} for i, block in enumerate(blocks)]
    rendered = json.dumps(displayed_value, ensure_ascii=False,
                          sort_keys=True, separators=(',', ':'), allow_nan=False)
    audit = {'version': VERSION, 'mappings': mappings, 'replacement_count': count,
             'document_segment_lengths': lengths,
             'source_payload_sha256': hashlib.sha256(payload_json.encode()).hexdigest(),
             'projected_payload_sha256': hashlib.sha256(rendered.encode()).hexdigest()}
    projection = BlindedPreferences(rendered, audit)
    if restore_workspace_paths(projection) != value:
        raise ValueError('workspace display projection must be losslessly reversible')
    return projection
