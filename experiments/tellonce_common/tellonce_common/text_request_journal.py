"""Durable receipts for at most two attempts of one read-only text request.

The caller owns source admission and parent-bound native/HTTP processes. This
journal neither chooses a model nor calls it implicitly, and never owns memory
learning. A missing completion after owner death is unknown, not a free call.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
from urllib.parse import quote

from .provenance import canonical_json_bytes, content_hash
from .role_inputs import RunIdentity
from .native_availability import NativeUnavailable

NATIVE_OUTAGES = {'native_quota_exhausted', 'native_server_overloaded', 'native_network_unavailable',
                 'native_auth_unavailable', 'native_startup_unavailable'}


class TextAttemptError(RuntimeError):
    def __init__(self,code,*,retryable=True,retry_after_s=None,replayed=False):
        if (not isinstance(code,str) or not re.fullmatch('[a-z][a-z0-9_]{0,79}',code) or
                type(retryable) is not bool or type(replayed) is not bool or
                (retry_after_s is not None and (type(retry_after_s) not in {int,float} or
                 not math.isfinite(retry_after_s) or retry_after_s<0))):
            raise ValueError('bounded safe text failure and retry metadata required')
        self.code,self.retryable,self.retry_after_s,self.replayed=code,retryable,retry_after_s,replayed
        super().__init__(code)


class TextRequestDeferred(RuntimeError):
    def __init__(self,code,*,retry_after_s):
        TextAttemptError(code,retry_after_s=retry_after_s)
        if retry_after_s is None:
            raise ValueError('deferred request requires a finite wait')
        self.code,self.retry_after_s=code,retry_after_s
        super().__init__(code)


@dataclass(frozen=True)
class TextReceipt:
    value: dict
    attempt: int
    replayed: bool


class TextRequestJournal:
    def __init__(self,private_root: Path,identity: RunIdentity,*,binding: str,_create=False):
        if (not isinstance(private_root,Path) or not private_root.is_absolute() or
                private_root!=private_root.resolve() or not isinstance(identity,RunIdentity) or
                not isinstance(binding,str) or not re.fullmatch('[0-9a-f]{64}',binding)):
            raise ValueError('explicit private request journal, namespace and source binding required')
        self.root=private_root/identity.namespace
        self.path=self.root/'requests.db'
        self.namespace,self.binding=identity.namespace,binding
        self.opened_at = time.time()
        if _create:
            private_root.mkdir(parents=True,exist_ok=True,mode=0o700)
            self.root.mkdir(mode=0o700)
            (self.root/'locks').mkdir(mode=0o700)
            connection=sqlite3.connect(self.path)
            try:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=FULL')
                connection.executescript('''
                    CREATE TABLE owner (singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                        schema_version INTEGER NOT NULL, namespace TEXT NOT NULL, binding TEXT NOT NULL);
                    CREATE TABLE requests (call_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL);
                    CREATE TABLE attempts (call_id TEXT NOT NULL, number INTEGER NOT NULL CHECK(number IN (1,2)),
                        status TEXT NOT NULL CHECK(status IN ('running','failed','completed')),
                        started REAL NOT NULL, finished REAL,
                        result_json TEXT, result_hash TEXT, failure_json TEXT,
                        PRIMARY KEY(call_id,number), FOREIGN KEY(call_id) REFERENCES requests(call_id));
                ''')
                connection.execute('INSERT INTO owner VALUES (1,1,?,?)',(self.namespace,binding))
                connection.commit()
            finally:
                connection.close()
            self.path.chmod(0o600)
        with self._db() as connection:
            self._owner(connection)

    def _owner(self,connection):
        owner=connection.execute('SELECT * FROM owner').fetchall()
        if (len(owner)!=1 or owner[0]['singleton']!=1 or owner[0]['schema_version']!=1 or
                owner[0]['namespace']!=self.namespace or owner[0]['binding']!=self.binding):
            raise ValueError('request journal ownership or schema changed')

    @classmethod
    def create(cls,private_root,identity,*,binding):
        return cls(private_root,identity,binding=binding,_create=True)

    @contextmanager
    def _db(self):
        if self.root!=self.root.resolve() or self.path.is_symlink():
            raise ValueError('private request journal cannot follow links')
        info=self.path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid!=os.getuid():
            raise ValueError('request journal must be an owned regular file')
        connection=sqlite3.connect('file:'+quote(str(self.path),safe='/')+'?mode=rw',uri=True,timeout=5)
        connection.row_factory=sqlite3.Row
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _lock(self,call_id):
        directory=self.root/'locks'
        if directory!=directory.resolve() or not directory.is_dir():
            raise ValueError('owned request-lock directory is missing or redirected')
        fd=os.open(directory/(content_hash(call_id)+'.lock'),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
        locked=False
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid!=os.getuid():
                raise ValueError('request lock must be an owned regular file')
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked=True
            except BlockingIOError:
                raise TextRequestDeferred('request_still_running',retry_after_s=.25) from None
            yield
        finally:
            if locked:
                fcntl.flock(fd,fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _failure_value(row):
        encoded=row['failure_json']
        if not isinstance(encoded,str) or len(encoded)>2048:
            raise ValueError('saved text failure is corrupt')
        value=json.loads(encoded)
        if (not isinstance(value,dict) or set(value)!={'code','retryable','not_before'} or
                type(value['not_before']) not in {int,float} or not math.isfinite(value['not_before']) or
                value['not_before']<0):
            raise ValueError('saved text failure is corrupt')
        TextAttemptError(value['code'],retryable=value['retryable'])
        return value

    @classmethod
    def _failure(cls,row,*,replayed):
        value=cls._failure_value(row)
        delay=max(0,value['not_before']-time.time()) if value['not_before'] else None
        return TextAttemptError(value['code'],retryable=value['retryable'],retry_after_s=delay,replayed=replayed)

    @staticmethod
    def _receipt(row,*,replayed):
        if not isinstance(row['result_json'],str) or len(row['result_json'])>4*1024*1024:
            raise ValueError('completed text receipt exceeds its result budget')
        value=json.loads(row['result_json'])
        if not isinstance(value,dict) or content_hash(value)!=row['result_hash']:
            raise ValueError('completed text receipt content is corrupt')
        return TextReceipt(value,row['number'],replayed)

    def _reserve(self,call_id,request_hash,attempt):
        with self._db() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._owner(connection)
            request=connection.execute('SELECT * FROM requests WHERE call_id=?',(call_id,)).fetchone()
            if request is None:
                if attempt!=1:
                    raise ValueError('second text attempt has no first request')
                connection.execute('INSERT INTO requests VALUES (?,?)',(call_id,request_hash))
            elif request['request_hash']!=request_hash:
                raise ValueError('one logical text request changed its input or configuration')
            rows={row['number']:dict(row) for row in connection.execute(
                'SELECT * FROM attempts WHERE call_id=? ORDER BY number',(call_id,))}
            completed=next((row for row in rows.values() if row['status']=='completed'),None)
            if completed is not None:
                # A recovered caller starts at attempt 1 and can recover a
                # completion from attempt 2. Success never authorizes sampling
                # a second answer by explicitly requesting a later attempt.
                if attempt not in {1,completed['number']}:
                    raise ValueError('a completed text request cannot be retried for another answer')
                return 'completed',completed
            for number,row in rows.items():
                if row['status']=='running':
                    # The kernel request lock is ours: no previous owning
                    # callback is running. Its completion was never committed.
                    failure=json.dumps({'code':'completion_unobserved','retryable':True,'not_before':0})
                    connection.execute('UPDATE attempts SET status=\'failed\',finished=?,failure_json=? WHERE call_id=? AND number=?',
                        (time.time(),failure,call_id,number))
                    row.update(status='failed',failure_json=failure)
            if attempt in rows:
                previous = rows[attempt]
                failure = self._failure_value(previous)
                if failure['code'] == 'provider_credit_exhausted':
                    # An HTTP 402 produced no model judgment. A fresh phase
                    # owner may resume the same request after funding recovery;
                    # the failed owner must yield rather than spin in place.
                    if (previous['finished'] is None or previous['finished'] >= self.opened_at or
                            failure['not_before'] > time.time()):
                        return 'failed', previous
                    connection.execute('CREATE TABLE IF NOT EXISTS availability_history '
                        '(id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, number INTEGER NOT NULL, '
                        'attempt_json TEXT NOT NULL, recovered_at REAL NOT NULL)')
                    connection.execute('INSERT INTO availability_history(call_id,number,attempt_json,recovered_at) '
                        'VALUES (?,?,?,?)', (call_id, attempt, json.dumps(previous), time.time()))
                    connection.execute('UPDATE attempts SET status=\'running\',started=?,finished=NULL,'
                        'failure_json=NULL WHERE call_id=? AND number=?', (time.time(),call_id,attempt))
                    return 'invoke', None
                if failure['code'] in NATIVE_OUTAGES:
                    # No judgment was delivered. Yield the whole scoring phase,
                    # preserving the request for a fresh owner after recovery.
                    if (previous['finished'] is None or previous['finished'] >= self.opened_at or
                            failure['not_before'] > time.time()):
                        return 'unavailable', failure
                    connection.execute('CREATE TABLE IF NOT EXISTS availability_history '
                        '(id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, number INTEGER NOT NULL, '
                        'attempt_json TEXT NOT NULL, recovered_at REAL NOT NULL)')
                    connection.execute('INSERT INTO availability_history(call_id,number,attempt_json,recovered_at) '
                        'VALUES (?,?,?,?)', (call_id, attempt, json.dumps(previous), time.time()))
                    connection.execute('UPDATE attempts SET status=\'running\',started=?,finished=NULL,'
                        'failure_json=NULL WHERE call_id=? AND number=?', (time.time(),call_id,attempt))
                    return 'invoke', None
                # A local operation deadline expires with its owner. A fresh
                # process has a new operation budget; replaying that timeout
                # forever would make checkpoint recovery impossible. Preserve
                # the failed reservation before reusing its slot. Completed
                # receipts and all other failure policies remain unchanged.
                if (previous['status'] == 'failed' and
                        self._failure_value(previous)['code'] == 'request_deadline' and
                        previous['finished'] is not None and previous['finished'] < self.opened_at):
                    connection.execute('CREATE TABLE IF NOT EXISTS deadline_history '
                        '(id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, number INTEGER NOT NULL, '
                        'attempt_json TEXT NOT NULL, recovered_at REAL NOT NULL)')
                    connection.execute('INSERT INTO deadline_history(call_id,number,attempt_json,recovered_at) '
                        'VALUES (?,?,?,?)', (call_id, attempt, json.dumps(previous), time.time()))
                    connection.execute('UPDATE attempts SET status=\'running\',started=?,finished=NULL,'
                        'failure_json=NULL WHERE call_id=? AND number=?', (time.time(), call_id, attempt))
                    return 'invoke', None
                return 'failed',rows[attempt]
            if attempt==2:
                first=rows.get(1)
                if first is None or first['status']!='failed':
                    raise ValueError('second text attempt needs a failed first attempt')
                failure=self._failure_value(first)
                if not failure['retryable']:
                    raise ValueError('first text failure does not permit retry')
                delay=failure['not_before']-time.time()
                if delay>0:
                    return 'deferred',delay
            connection.execute('INSERT INTO attempts(call_id,number,status,started) VALUES (?,?,\'running\',?)',
                               (call_id,attempt,time.time()))
        return 'invoke',None

    def _finish(self,call_id,request_hash,attempt,*,failure=None,result=None):
        with self._db() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._owner(connection)
            request=connection.execute('SELECT request_hash FROM requests WHERE call_id=?',(call_id,)).fetchone()
            if request is None or request['request_hash']!=request_hash:
                raise ValueError('request binding changed during text callback')
            if failure is not None:
                cursor=connection.execute("UPDATE attempts SET status='failed',finished=?,failure_json=? "
                    "WHERE call_id=? AND number=? AND status='running'",
                    (time.time(),json.dumps(failure,allow_nan=False),call_id,attempt))
            else:
                cursor=connection.execute("UPDATE attempts SET status='completed',finished=?,result_json=?,result_hash=? "
                    "WHERE call_id=? AND number=? AND status='running'",
                    (time.time(),result,content_hash(json.loads(result)),call_id,attempt))
            if cursor.rowcount!=1:
                raise ValueError('text completion lost its running reservation')

    def perform(self,*,call_id,request_hash,attempt,invoke):
        if (not isinstance(call_id,str) or not call_id.strip() or len(call_id)>480 or
                not isinstance(request_hash,str) or not re.fullmatch('[0-9a-f]{64}',request_hash) or
                type(attempt) is not int or attempt not in {1,2} or not callable(invoke)):
            raise ValueError('one bounded identified text attempt and request hash required')
        with self._lock(call_id):
            action,record=self._reserve(call_id,request_hash,attempt)
            if action=='completed':
                return self._receipt(record,replayed=True)
            if action=='failed':
                raise self._failure(record,replayed=True)
            if action=='deferred':
                raise TextRequestDeferred('provider_not_before',retry_after_s=record)
            if action=='unavailable':
                raise NativeUnavailable(record['code'],record['not_before'] or None)
            failure=None
            try:
                value=invoke()
                if not isinstance(value,dict):
                    raise ValueError('text callback must return its captured result object')
                encoded=canonical_json_bytes(value)
                if len(encoded)>4*1024*1024:
                    raise ValueError('text receipt exceeds its result budget')
            except NativeUnavailable as exc:
                self._finish(call_id,request_hash,attempt,failure={
                    'code':exc.code,'retryable':True,'not_before':exc.reset_at or 0})
                raise
            except TextAttemptError as exc:
                failure=exc
            except Exception:
                failure=TextAttemptError('text_callback_failed')
            if failure is not None:
                failed={'code':failure.code,'retryable':failure.retryable,
                        'not_before':time.time()+failure.retry_after_s if failure.retry_after_s else 0}
                self._finish(call_id,request_hash,attempt,failure=failed)
                raise TextAttemptError(failure.code,retryable=failure.retryable,
                                       retry_after_s=failure.retry_after_s,replayed=False) from None
            self._finish(call_id,request_hash,attempt,result=encoded.decode())
            return TextReceipt(json.loads(encoded),attempt,False)
