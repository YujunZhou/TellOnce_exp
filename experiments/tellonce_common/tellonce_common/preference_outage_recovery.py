"""One-time evidence-bound recovery of native outages mislabeled by old workers."""
import json

from .deferred_scoring import _write
from .provenance import content_hash
from .text_request_journal import NATIVE_OUTAGES

NAME = 'native-outage-recovery-v1.json'
NAMES = {'delivery-v3': NAME, 'delivery-v4': 'native-outage-recovery-v2.json',
         'delivery-v5': 'native-startup-recovery-v3.json'}


def capture(directory, *, runtime_bundle_hash, revision='delivery-v3'):
    """After the old worker exits, read its metadata logs, never saved answers."""
    path=directory/NAMES[revision]
    if path.exists():
        return json.loads(path.read_text())
    prior=directory/('preference-'+revision)
    if not (prior/'owner.json').exists():
        return None
    owner=json.loads((prior/'owner.json').read_text())
    namespace=owner['manifest']['namespace'];by_parent={};keys_by_parent={}
    for log in (directory.parent/'worker-attempts').glob('*.jsonl'):
        with log.open() as stream:
            # Evaluators do not emit worker_admission headers. Bind the
            # generation using their actual revisioned completion events.
            head=stream.read(8192)
            if '"score_component_' not in head and '"text_attempt_link"' not in head:
                continue
            stream.seek(0);links={};failures={};completed=set()
            for line in stream:
                if not line.endswith('\n'):
                    continue  # Interrupted final metadata write is not evidence.
                if not any(kind in line for kind in ('"text_attempt_link"','"text_role_failed"','"private_score_completed"')):
                    continue
                event=json.loads(line)
                if event.get('namespace')!=namespace:
                    continue
                if event['type']=='private_score_completed':
                    if event.get('revision')==revision:
                        key=event['context']['call_id'];parent=content_hash([key,'personal_preferences'])
                        completed.add(parent);keys_by_parent[parent]=key
                    continue
                if event.get('context',{}).get('role')!='preference_score':continue
                if event['type']=='text_attempt_link':
                    links[event['child_context']['call_id']]=event['context']['call_id']
                elif event.get('stage')=='native' and event.get('availability_code') in NATIVE_OUTAGES:
                    failures[event['context']['call_id']]=event['availability_code']
                elif (revision=='delivery-v4' and event.get('stage')=='auth'
                      and event.get('failure_code')=='RuntimeError' and event.get('availability_code') is None):
                    # Old v4 workers recorded local lease refusal before any
                    # provider request; this is not a model's unknown judgment.
                    failures[event['context']['call_id']]='native_auth_unavailable'
                elif revision=='delivery-v5':
                    if event.get('availability_code') in NATIVE_OUTAGES:
                        failures[event['context']['call_id']]=event['availability_code']
                    elif (event.get('stage')=='initialize' and event.get('failure_code')=='RuntimeError'
                          and event.get('usage') is None and event.get('availability_code') is None):
                        # initialize precedes authentication/thread/turn. These
                        # old records attest to no model request, not a judgment.
                        failures[event['context']['call_id']]='native_startup_unavailable'
            for child,code in failures.items():
                if child in links and links[child] in completed:
                    if revision=='delivery-v5' and any(c not in failures for c,p in links.items() if p==links[child]):
                        # Never reopen mixed output-format/semantic failures on
                        # the strength of just one earlier startup failure.
                        continue
                    by_parent.setdefault(links[child],[]).append({'child_call_id':child,'code':code})
    receipts={}
    # No reason to touch successful/recently-written receipts. Read only the
    # score files explicitly named by already-proven failed calls.
    for parent in by_parent:
        file=prior/(keys_by_parent[parent]+'.json')
        if not file.exists():continue
        value=json.loads(file.read_text());parent=content_hash([file.stem,'personal_preferences'])
        if (parent in by_parent and any(v['rule_id']=='__preference_evaluation_unavailable__'
                                       for v in value['grade']['preferences'])):
            receipts[file.stem]={'receipt_digest':value['digest'],'parent_call_id':parent,
                                'native_failures':sorted(by_parent[parent],key=lambda r:r['child_call_id'])}
    body={'schema':1,'owner':owner,'runtime_bundle_hash':runtime_bundle_hash,'receipts':receipts}
    value={**body,'digest':content_hash(body)}
    _write(path,value)
    return value


def load(queue, revision='delivery-v3'):
    path=queue.directory/NAMES[revision]
    if not path.exists():return {}
    if path.is_symlink() or path.stat().st_size>16*1024*1024:
        raise ValueError('bounded owned native-outage recovery proof required')
    value=json.loads(path.read_text());body={k:v for k,v in value.items() if k!='digest'}
    owner={'schema':1,'revision':revision,'seal':queue._sealed,'manifest':queue._manifest}
    if (set(value)!={'schema','owner','runtime_bundle_hash','receipts','digest'} or value['schema']!=1 or
            value['digest']!=content_hash(body) or value['owner']!=owner or not isinstance(value['receipts'],dict)):
        raise ValueError('native-outage recovery proof changed or belongs to another queue')
    for key,row in value['receipts'].items():
        if (key not in queue._jobs or row['parent_call_id']!=content_hash([key,'personal_preferences']) or
                not row['native_failures'] or any(r['code'] not in NATIVE_OUTAGES for r in row['native_failures'])):
            raise ValueError('native-outage recovery requires matched native failure evidence')
    return value['receipts']
