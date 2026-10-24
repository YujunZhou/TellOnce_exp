"""Route auxiliary text roles to the official OpenAI API (gpt-6-luna, low reasoning).

The run's `runtime.api_luna_roles` policy selects this route. The solver,
namespace and roles.json stay unchanged; every auxiliary call (simulated user,
learning, compilation, selection, checking, scoring) goes through a shared slot
pool and a durable request journal.
"""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
import uuid
from urllib.parse import quote

from .deferred_scoring import _write
from .durable_text_transport import DurableTextTransport, DurableTextResult
from .frozen_task_evaluation import _read
from .http_text_contract import HTTPTextConfig, LUNA_API_MODEL
from .http_text_transport import HTTPTextTransport
from .provenance import content_hash
from .provider_credentials import ProviderKeyFile
from .request_budget import request_budget
from .text_capacity import TextCapacity
from .text_request_journal import TextRequestJournal
from .text_roles import CALL_ROLES

POLICY='api-luna-low-v1'
ROLES=('feedback','memory','preference','appeal','execution_check')


class _SharedSlots:
    def __init__(self, directory, count=16):
        self.directory = Path(directory)
        if (not self.directory.is_absolute() or self.directory != self.directory.resolve() or
                type(count) is not int or count != 16):
            raise ValueError('the shared API slot pool must be canonical and exactly 16 slots')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.fds = []
        self._guard = threading.Lock()
        self._busy = set()
        for index in range(count):
            path = self.directory / ('slot-%02d.lock' % index)
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1):
                os.close(fd)
                raise ValueError('API slot must be an owned regular lock file')
            self.fds.append(fd)

    @contextmanager
    def acquire(self, deadline):
        selected = None
        while selected is None:
            if time.monotonic() >= deadline:
                from .native_availability import NativeUnavailable
                raise NativeUnavailable('native_server_overloaded', reset_at=time.time() + 1)
            for index, fd in enumerate(self.fds):
                with self._guard:
                    if index in self._busy:
                        continue
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if index not in self._busy:
                            self._busy.add(index)
                            selected = (index, fd)
                            break
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except BlockingIOError:
                        continue
            if selected is None:
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
        try:
            yield
        finally:
            index, fd = selected
            fcntl.flock(fd, fcntl.LOCK_UN)
            with self._guard:
                self._busy.discard(index)

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()


class _LogicalModelTransport:
    """Expose the old, pinned SDK model while delegating only to native Luna."""
    def __init__(self, effective, logical_config):
        self._effective = effective
        self.config = logical_config
        self.context_window = effective.context_window
        self.identity = effective.identity

    def call(self, *args, **kwargs):
        return self._effective.call(*args, **kwargs)


class _SlottedAPI:
    def __init__(self, transport, pool, record, role):
        self.inner,self.pool,self.record,self.role=transport,pool,record,role
        self.identity,self.config=transport.identity,transport.config
        self.context_window=1050000

    def _invoke(self, method, *args, **kwargs):
        started=time.monotonic()
        with self.pool.acquire(started+self.inner.timeout_s):
            self.record({'type':'api_luna_call','role':self.role,'policy':POLICY,
                         'context':dict(kwargs['context']),'effective_config':asdict(self.config)})
            with request_budget(max(.001,self.inner.timeout_s-(time.monotonic()-started))):
                return getattr(self.inner,method)(*args,**kwargs)

    def call(self,*args,**kwargs):
        return self._invoke('call',*args,**kwargs)

    def call_messages(self,*args,**kwargs):
        return self._invoke('call_messages',*args,**kwargs)


def historical_receipt(root, identity, binding, call_id, request_hash, *, successor=None, successor_hash=None):
    """Look up only this request, read-only; never invoke a former provider."""
    path=root/identity.namespace/'requests.db'
    if not path.exists():
        return None
    if path.is_symlink() or path.resolve()!=path:
        raise ValueError('historical journal path changed')
    db=sqlite3.connect('file:'+quote(str(path),safe='/')+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        owner=db.execute('SELECT * FROM owner').fetchall()
        if len(owner)!=1 or owner[0]['namespace']!=identity.namespace or owner[0]['binding']!=binding:
            raise ValueError('historical journal owner changed')
        request=db.execute('SELECT request_hash FROM requests WHERE call_id=?',(call_id,)).fetchone()
        if request is None:
            return None
        completed=db.execute("SELECT * FROM attempts WHERE call_id=? AND status='completed' ORDER BY number",(call_id,)).fetchall()
        if request['request_hash']!=request_hash:
            raise ValueError('historical request input changed')
        if not completed:
            rows=[dict(row) for row in db.execute('SELECT * FROM attempts WHERE call_id=? ORDER BY number',(call_id,))]
            if not rows:
                return None
            if successor is None:
                raise ValueError('failed historical attempts require explicit successor retention')
            # Preserve the original attempt cap and terminal refusals. Existing
            # journal rules alone decide whether a failed request may resume.
            with successor._lock(call_id), successor._db() as target:
                target.execute('BEGIN IMMEDIATE')
                current=target.execute('SELECT request_hash FROM requests WHERE call_id=?',(call_id,)).fetchone()
                if current is None:
                    target.execute('INSERT INTO requests VALUES (?,?)',(call_id,successor_hash))
                    for row in rows:
                        if row['status']=='failed':
                            TextRequestJournal._failure_value(row)
                        if row['status'] not in {'failed','running'}:
                            raise ValueError('unexpected historical attempt state')
                        target.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?)',tuple(row[k] for k in
                            ('call_id','number','status','started','finished','result_json','result_hash','failure_json')))
                    target.execute('CREATE TABLE IF NOT EXISTS migration_history (call_id TEXT PRIMARY KEY, prior_journal TEXT NOT NULL, prior_request_hash TEXT NOT NULL)')
                    target.execute('INSERT INTO migration_history VALUES (?,?,?)',(call_id,str(root),request_hash))
                elif current[0]!=successor_hash:
                    raise ValueError('successor request changed after failure migration')
            return False
        if len(completed)!=1:
            raise ValueError('historical completed request input changed')
        return TextRequestJournal._receipt(dict(completed[0]),replayed=True)
    finally:
        db.close()


class _MigratingDurable(DurableTextTransport):
    def __init__(self,*args,prior_roots,role,**kwargs):
        super().__init__(*args,**kwargs)
        self.prior_roots,self.role=prior_roots,role

    def _call(self,prompt,*,context,base_instructions,validate,conversation=None,generation=None):
        self._check_request(prompt,context,base_instructions,validate)
        self.authorize('call',{'namespace':self.identity.namespace,'binding':self.journal.binding,**context})
        request={'binding':self.journal.binding,'prompt':prompt,'base_instructions':base_instructions,
                 'context':{k:v for k,v in context.items() if k!='attempt'}}
        if conversation is not None:
            request['conversation_generation']=generation
        for root in self.prior_roots:
            old_request=request
            if self.role=='feedback' and root.name=='direct-luna-journal' and conversation is not None:
                serialized=json.dumps(conversation,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                old_prompt=('May stage request metadata: '+json.dumps(generation,sort_keys=True)+
                            '\nReturn only the requested stage response.\nConversation JSON:\n'+serialized)
                old_request={k:v for k,v in request.items() if k!='conversation_generation'}
                old_request.update(prompt=old_prompt,base_instructions='Follow the role/content order in Conversation JSON exactly. Preserve the original May stage task and return its strict JSON response.')
            receipt=historical_receipt(root,self.identity,self.journal.binding,context['call_id'],content_hash(old_request),
                successor=self.journal,successor_hash=content_hash(request))
            if receipt is None:
                continue
            if receipt is False:
                break
            if context['attempt'] not in {1,receipt.attempt}:
                raise ValueError('a historical completion cannot be resampled')
            self._validate(receipt.value,validate)
            value=receipt.value
            self.record({'type':'historical_text_receipt_delivered','policy':POLICY,'role':self.role,
                         'prior_journal':str(root),'call_id':context['call_id'],'replayed':True})
            return DurableTextResult(value['text'],value['response_id'],value['thread_id'],'null',
                value['usage_json'],True,receipt.attempt,value['child_call_id'])
        return super()._call(prompt,context=context,base_instructions=base_instructions,
                            validate=validate,conversation=conversation,generation=generation)


class _APIRoles:
    api_luna_roles=True
    def __init__(self,original,overrides,phase,pool):
        self._original,self._overrides=original,overrides
        self.direct_luna_feedback=phase=='training'
        self.api_luna_policy=POLICY
        self._api_pool=pool

    def __getattr__(self,name):
        return getattr(self._original,name)

    def get(self,role):
        if role not in self._overrides:
            raise ValueError('role is not admitted in this API successor phase')
        return self._overrides[role]


def configure_api_luna_roles(spec, roles, *, phase):
    ref=getattr(spec,'runtime_bundle_reference',{})
    policy=ref.get('api_luna_roles')
    if policy is None:
        return roles
    if spec.identity!=roles.identity or spec.binding!=roles.binding:
        raise ValueError('auxiliary API roles must belong to this run')
    revision_hash=spec.runtime_revision_hash
    fields={'schema','policy','provider','model','reasoning_effort','max_tokens','slot_directory','slot_count'}
    # Optional reviewed extension (2026-09-30): rule selection alone may use a
    # different reasoning tier; every other auxiliary role keeps the low policy.
    selector_effort=policy.get('selector_reasoning_effort') if isinstance(policy,dict) else None
    if selector_effort is not None and (selector_effort not in {'none','low','medium'} or
            not isinstance(policy,dict) or set(policy)!=fields|{'selector_reasoning_effort'}):
        raise ValueError('selector reasoning override must be none, low or medium inside the low policy')
    policy_fields=set(policy) - {'selector_reasoning_effort'} if isinstance(policy,dict) else set()
    if (not isinstance(policy,dict) or policy_fields!=fields or policy['schema']!=1 or policy['policy']!=POLICY or
            policy['provider']!='openai' or policy['model']!=LUNA_API_MODEL or policy['reasoning_effort']!='low' or
            policy['max_tokens']!=16384 or policy['slot_count']!=16 or
            phase not in {'training','test','preparation'} or getattr(roles,'api_luna_roles',False)):
        raise ValueError('explicit bounded official gpt-6-luna low role policy required')
    slots=Path(policy['slot_directory'])
    if not slots.is_absolute() or slots.resolve()!=slots:
        raise ValueError('canonical shared API slot directory required')
    roles.verify()
    pool=_SharedSlots(slots,policy['slot_count'])
    overrides={}
    key=ProviderKeyFile(spec.api_key_file,provider='openai',authorize=roles.authorize)
    try:
        for role in ROLES:
            if role=='feedback' and phase!='training':
                continue
            if role=='execution_check' and roles.identity.method!='trace':
                continue
            root=roles.directory/role
            root.mkdir(mode=0o700,parents=True,exist_ok=True)
            journal_root=root/(POLICY+'-journal')
            config=HTTPTextConfig(provider='openai',model=LUNA_API_MODEL,reasoning_effort='low',max_tokens=policy['max_tokens'])
            manifest={'schema':1,'namespace':roles.identity.namespace,'binding':roles.binding,'phase':phase,
                      'role':role,'policy':policy,'effective_config':asdict(config)}
            manifest_path=journal_root/'api-role.json'
            if journal_root.exists():
                if _read(manifest_path,1024*1024)!=manifest:
                    raise ValueError('API role journal configuration changed')
                journal=TextRequestJournal(journal_root,roles.identity,binding=roles.binding)
            else:
                journal=TextRequestJournal.create(journal_root,roles.identity,binding=roles.binding)
                _write(manifest_path,manifest)
            def authorize(action,detail,*,role=role,expected=manifest,path=manifest_path):
                roles.verify()
                if _read(path,1024*1024)!=expected:
                    raise ValueError('API role ownership changed')
                if action=='prepare':
                    if detail.get('model')!=LUNA_API_MODEL:
                        raise ValueError('API role model changed')
                elif detail.get('role') not in CALL_ROLES[role] or detail.get('phase')!=phase:
                    raise ValueError('API call crossed the admitted phase or role')
                roles.authorize(action,{'text_role':role,'api_luna_policy':POLICY,**detail})
            attempt_root=root/(POLICY+'-attempts')
            attempt_root.mkdir(mode=0o700,exist_ok=True)
            transport=HTTPTextTransport(identity=roles.identity,config=config,private_root=attempt_root/str(uuid.uuid4()),
                authorize=authorize,record=roles.record,api_key_source=key,timeout_s=900,
                selector_reasoning_effort=selector_effort if role=='memory' else None)
            slotted=_SlottedAPI(transport,pool,roles.record,role)
            prior=[root/name for name in ('direct-luna-journal','runtime-journal','journal')]
            durable=_MigratingDurable(slotted,journal=journal,authorize=authorize,record=roles.record,
                capacity=TextCapacity.openai_text(model=config.model,output_reserve=config.max_tokens,record=roles.record),
                prior_roots=prior,role=role)
            durable.effective_transport=slotted
            if role=='memory':
                # Preserve SDK storage/model bindings. Effective provider and
                # every new HTTP request are separately recorded as API Luna.
                logical=roles.configs[role]
                durable.transport=_LogicalModelTransport(slotted,logical)
            overrides[role]=durable
        roles.record({'type':'api_luna_roles_configured','policy':policy,'phase':phase,
                      'namespace':roles.identity.namespace,'runtime_revision_hash':revision_hash})
        return _APIRoles(roles,overrides,phase,pool)
    except Exception:
        pool.close()
        raise
