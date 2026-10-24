"""Phase-owned Tellonce background learning and frozen per-task memory ports.

Import only an explicitly admitted source bundle in a fresh worker process.
This owner never reads an installed plugin or imports an operator memory store.
The phase supervisor still owns real-model admission, budgets and process death.
"""
import importlib
import json
from pathlib import Path
import sys
import threading
import time

from .deferred_scoring import _write
from .frozen_task_evaluation import _directory,_digest,_read
from .interaction_loop import RecoverableCallError
from .managed_model_requests import ManagedModelRequests
from .provenance import content_hash,file_hash,tree_hash
from .role_inputs import RunIdentity
from .tellonce_session import TellonceSession

# Exact, directional source inventory transition reviewed for the 2026-09-18
# correction-learning/checker repair. Original training receipts stay intact;
# the worker admission records the actual successor runtime bundle. Any other
# source change still requires its own review and explicit transition.
# The 2026-09-19 successor changes execution support only, preserving the
# stored rule/publication schema and every original training receipt.
_REVIEWED_SOURCE_TRANSITIONS = frozenset({(
    '1a954e1e19a2ec3d643dbba08b683d9ce71f08032921e08c85084fc066f65c81',
    '899c0accd7d08785b340a20c3692a26d8d3f08670b3c131f21a812b9cd4c5b36',
), (
    '1a954e1e19a2ec3d643dbba08b683d9ce71f08032921e08c85084fc066f65c81',
    'b7b54cc338f2d27a878452d3f95ad9ba5a84f4712fc6ba809d13e4e05e3e2202',
), (
    '899c0accd7d08785b340a20c3692a26d8d3f08670b3c131f21a812b9cd4c5b36',
    'b7b54cc338f2d27a878452d3f95ad9ba5a84f4712fc6ba809d13e4e05e3e2202',
), (
    # 2026-09-21: bounded transient learning recovery. Rule/publication schemas
    # and original receipts remain unchanged; no legacy failed turn is reopened.
    'aed21179cba80041afbb55cdbc51db34821298068e5257818cfa4a2cf6ea6198',
    'bd96163f141a2611fcaaa9424bc285a60d8bf5dda5c634074a7d4013b17a8951',
)})


# 2026-09-22 diagnostic successor: only execution_support and memory_runtime
# differ. Stored schemas and learning sources are unchanged. This transition
# admits frozen reads only; it cannot resume training with different code.
_REVIEWED_TEST_SOURCE_TRANSITIONS = frozenset({(
    'bd96163f141a2611fcaaa9424bc285a60d8bf5dda5c634074a7d4013b17a8951',
    '164a9a8a88838281b8a00e73598bf93497dfec0f37077cee623ffb89b34a7bd5',
), (
    # Current-candidate binding: immutable rules/support schemas; checked on
    # eight current observations and the prior fifteen-observation regression.
    '647a8d45d2665e1a36e22213e98cb11d7b96a95968951d487b4884ff6a2f83f9',
    '92fb0f9229ee2f76b23901632b66979e0ee45464533f2da841dd1f9743873cf5',
), (
    'bd96163f141a2611fcaaa9424bc285a60d8bf5dda5c634074a7d4013b17a8951',
    'c9e978662791587c03285b15862b6234c89e02dcbda12dab40dfcf86428e8837',
), (
    # 2026-09-23: bounded checks inherit the host deadline. Selector/checker
    # activity scope changes do not alter saved schemas or learning sources.
    '647a8d45d2665e1a36e22213e98cb11d7b96a95968951d487b4884ff6a2f83f9',
    'd28333c21b9183abfb87a07b83525f67919e8744b9ccaf29112824b3aed9cfb6',
), (
    'bd96163f141a2611fcaaa9424bc285a60d8bf5dda5c634074a7d4013b17a8951',
    '86a23063ffabc4a4128758f121ae4ef0e64abade55536b89acf2c9f1d7361d85',
), (
    # Literal comparison triggers: same frozen schemas, one checker-prompt
    # addition verified on the 32-observation regression (2026-09-23).
    '647a8d45d2665e1a36e22213e98cb11d7b96a95968951d487b4884ff6a2f83f9',
    '6c04a16aa8bd52d69c22f2fa56edfdeb6639f93556475d83993a0f662dd90d1b',
), (
    'bd96163f141a2611fcaaa9424bc285a60d8bf5dda5c634074a7d4013b17a8951',
    '1fca4aeaa5cdf9d824140759b26ae2344e502e1cfc8cefca235e2ad47f8e3597',
), (
    '647a8d45d2665e1a36e22213e98cb11d7b96a95968951d487b4884ff6a2f83f9',
    '7c4898c60c55e7048237d0ea3fce6d2917bd2bd50a48d265ef5ec2f5f381caca',
), (
    'bd96163f141a2611fcaaa9424bc285a60d8bf5dda5c634074a7d4013b17a8951',
    '6f3bd77bcf4926617c22f4f30dbf11377f01dd88673a518b588d45ddf7e78321',
), (
    # Horizontal table padding may be restored to an exact observed quote.
    # No rule, support schema, learning behavior, or frozen memory changes.
    '647a8d45d2665e1a36e22213e98cb11d7b96a95968951d487b4884ff6a2f83f9',
    'ebbeb7d304ea00ccf4da46a6897f1cfd3df146c67fc2e77e4466abfbaddd3186',
), (
    'bd96163f141a2611fcaaa9424bc285a60d8bf5dda5c634074a7d4013b17a8951',
    '0c7017c1b4a113660760700b09d8137b77a28cac195347c4bc36779ed0bc522b',
)})


def _sources_compatible(training, runtime, *, frozen_test=False):
    if not isinstance(training, dict) or not isinstance(runtime, dict):
        return False
    transition = (content_hash(training), content_hash(runtime))
    return (training == runtime or transition in _REVIEWED_SOURCE_TRANSITIONS or
            (frozen_test and transition in _REVIEWED_TEST_SOURCE_TRANSITIONS))


# Freeze admission of this experiment (2026-09-30). Tellonce v1.7.0 refuses to
# freeze while any learner turn is 'failed' or any rule support is 'pending'.
# Both are bounded outcomes the library itself already refused to retry: a
# validator-rejected judge response after its request budget, or a compiler
# that explicitly left a rule unresolved. The earlier reviewed bundles froze
# with them recorded, and this experiment keeps that semantics: the counts and
# pending rule ids stay in freeze-audit.json and the frozen-method receipt as
# evidence, pending rules never enter the active snapshot, and only genuinely
# unprocessed work (pending/resolving turns, unprojected commits) blocks.
FREEZE_EVIDENCE_STATES=frozenset({'failed'})
_LEARNING_COMPLETED_STATES=frozenset({'projected','noop','clarified','dismissed','rejected'})


def _evidence_preserving_freeze(cls,error):
    if getattr(cls,'_require_completed_learning',None) is None or not isinstance(error,type):
        return cls
    class TrainingMemory(cls):
        @staticmethod
        def _require_completed_learning(counts,publication):
            if not isinstance(counts,dict):
                raise error('freeze audit requires turn outcome counts')
            incomplete={key:count for key,count in counts.items()
                        if count and key not in _LEARNING_COMPLETED_STATES|FREEZE_EVIDENCE_STATES}
            if incomplete:
                raise error('learning or projection incomplete at freeze: '+json.dumps(incomplete,sort_keys=True))
    TrainingMemory.__name__=cls.__name__
    TrainingMemory.__qualname__=cls.__qualname__
    return TrainingMemory


class TellonceAPI:
    def __init__(self, *, library: Path,authorize):
        if (not isinstance(library,Path) or not library.is_absolute() or library!=library.resolve() or
                not library.is_dir() or not callable(authorize)):
            raise ValueError('explicit admitted Tellonce source library required')
        self.library=library
        self.sources=self._sources()
        authorize('load_tellonce',{'library':str(library),'sources':dict(self.sources)})
        self._verify_imports()
        # Tellonce uses top-level sibling imports. A worker may own one source
        # bundle only; reject cached modules from another bundle, never reload
        # mixed classes into an already running experiment.
        sys.path.insert(0,str(library))
        names={'TrainingMemory':('training_memory','TrainingMemory'),
               'TaskMemory':('memory_runtime','TaskMemory'),
               'Publication':('memory_publication','Publication'),
               'load_publication':('memory_publication','load_publication'),
               'capture_reference':('transcript_adapter','capture_context_reference'),
               'ModelRequestError':('model_requests','ModelRequestError'),
               'Observation':('execution_support','Observation'),'Event':('execution_support','Event')}
        for name,(module,attribute) in names.items():
            setattr(self,name,getattr(importlib.import_module(module),attribute))
        self.TrainingMemory=_evidence_preserving_freeze(
            self.TrainingMemory,importlib.import_module('execution_support').SupportError)
        self.verify()

    def _sources(self):
        paths=sorted([*self.library.glob('*.py'),*self.library.glob('*.yaml')])
        if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
            raise ValueError('Tellonce source inventory must consist of explicit regular files')
        return {path.name:file_hash(path) for path in paths}

    def _verify_imports(self):
        for name in self.sources:
            if not name.endswith('.py'):
                continue
            module=sys.modules.get(Path(name).stem)
            if module is not None and Path(getattr(module,'__file__','')).resolve()!=self.library/name:
                raise ValueError('a different Tellonce source is already imported; use a fresh worker')

    def verify(self):
        if self._sources()!=self.sources:
            raise ValueError('Tellonce source changed after admission')
        self._verify_imports()


class _Enqueuer:
    def __init__(self,owner):
        self.owner,self.namespace=owner,owner.identity.namespace

    def enqueue_user(self,**kwargs):
        owner=self.owner
        if not owner._accepting or owner._closed or owner._thread is None:
            raise ValueError('Tellonce training admission requires its active phase worker')
        if owner._failure is not None:
            raise RecoverableCallError('tellonce_background_unavailable',retryable=False)
        result=owner.learner.enqueue_user(**kwargs)
        owner._event_ids.add((kwargs['task_id'],kwargs['session_id'],kwargs['event_id']))
        owner._wake.set()
        return result


class TellonceMethod:
    def _setup(self,*,directory,identity,api,invoke,binding,execution_enabled,authorize,record,check_health):
        _directory(directory)
        if (not isinstance(identity,RunIdentity) or identity.method!='trace' or not isinstance(api,TellonceAPI) or
                not isinstance(invoke,ManagedModelRequests) or invoke.identity!=identity or not _digest(binding) or
                invoke.transport.journal.binding!=binding or type(execution_enabled) is not bool or
                any(not callable(f) for f in (authorize,record,check_health))):
            raise ValueError('matching admitted Tellonce phase, managed roles and supervision required')
        api.verify()
        self.directory,self.identity,self.api,self.invoke=directory,identity,api,invoke
        self.binding,self.execution_enabled=binding,execution_enabled
        self.authorize,self.record,self.check_health=authorize,record,check_health
        self.learner=None
        self._thread=None
        self._wake,self._stop=threading.Event(),threading.Event()
        self._failure,self._closed,self._accepting=None,False,False
        self._sessions,self._event_ids={},set()
        self._receipt=None

    @classmethod
    def create(cls,*,directory,identity,api,invoke,binding,compiler_id,execution_enabled,authorize,record,check_health):
        result=cls.__new__(cls)
        result._setup(directory=directory,identity=identity,api=api,invoke=invoke,binding=binding,
            execution_enabled=execution_enabled,authorize=authorize,record=record,check_health=check_health)
        if directory.exists() or not isinstance(compiler_id,str) or not compiler_id:
            raise ValueError('Tellonce training starts in a new method directory with a fixed compiler')
        result._manifest={'schema':1,'namespace':identity.namespace,'binding':binding,'compiler_id':compiler_id,
                          'sources':dict(api.sources),'training_execution_enabled':execution_enabled}
        authorize('prepare_training_method',dict(result._manifest))
        directory.mkdir(mode=0o700)
        _write(directory/'method.json',result._manifest)
        result.learner=api.TrainingMemory.create(directory/'library',namespace=identity.namespace,
                                               compiler_id=compiler_id,invoke=invoke)
        result._accepting=True
        return result

    @classmethod
    def open_training(cls,*,directory,identity,api,invoke,binding,compiler_id,execution_enabled,authorize,record,check_health):
        result=cls.__new__(cls)
        result._setup(directory=directory,identity=identity,api=api,invoke=invoke,binding=binding,
            execution_enabled=execution_enabled,authorize=authorize,record=record,check_health=check_health)
        if not directory.is_dir() or not isinstance(compiler_id,str) or not compiler_id:
            raise ValueError('Tellonce training starts in a new method directory with a fixed compiler')
        result._manifest={'schema':1,'namespace':identity.namespace,'binding':binding,'compiler_id':compiler_id,
                          'sources':dict(api.sources),'training_execution_enabled':execution_enabled}
        original = _read(directory/'method.json', 1024*1024)
        if (not _sources_compatible(original.get('sources'), api.sources) or
                {**original, 'sources': dict(api.sources)} != result._manifest or
                (directory/'frozen-method.json').exists()):
            raise ValueError('Tellonce checkpoint has another configuration or is frozen')
        result._manifest = original
        result.learner=api.TrainingMemory(directory/'library',namespace=identity.namespace,
                                        compiler_id=compiler_id,invoke=invoke)
        result._accepting=True
        authorize('resume_training_method',{**result._manifest,
            'runtime_source_inventory': content_hash(api.sources)})
        return result

    @classmethod
    def open_test(cls,*,directory,identity,api,invoke,binding,expected_receipt,execution_enabled,authorize,record,check_health, admitted_learning_audit=None):
        result=cls.__new__(cls)
        result._setup(directory=directory,identity=identity,api=api,invoke=invoke,binding=binding,
            execution_enabled=execution_enabled,authorize=authorize,record=record,check_health=check_health)
        result._manifest=_read(directory/'method.json',1024*1024)
        value=_read(directory/'frozen-method.json',1024*1024)
        if (not _digest(expected_receipt) or content_hash(value)!=expected_receipt or value.get('schema')!=1 or
                value.get('manifest_hash')!=content_hash(result._manifest) or
                result._manifest.get('namespace')!=identity.namespace or result._manifest.get('binding')!=binding or
                not _sources_compatible(result._manifest.get('sources'), api.sources, frozen_test=True)):
            raise ValueError('frozen Tellonce method differs from its admitted training receipt')
        from .learning_admission import require_trace_learning
        require_trace_learning(value.get('freeze_audit'), namespace=identity.namespace,
                               publication_digest=value.get('publication_digest'),
                               admitted=admitted_learning_audit)
        result._receipt=value
        result.verify_frozen()
        authorize('open_frozen_method',{'namespace':identity.namespace,'binding':binding,
                  'receipt':expected_receipt,'execution_enabled':execution_enabled,
                  'admitted_learning_audit':admitted_learning_audit,
                  'training_source_inventory': content_hash(result._manifest['sources']),
                  'runtime_source_inventory': content_hash(api.sources)})
        return result

    @property
    def background_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def assert_healthy(self):
        if self._closed:
            raise ValueError('Tellonce phase owner is closed')
        self.check_health()
        self.api.verify()
        if _read(self.directory/'method.json',1024*1024)!=self._manifest:
            raise ValueError('Tellonce method configuration changed during its phase')
        if self._failure is not None or (self._thread is not None and not self._stop.is_set() and not self.background_alive):
            raise RuntimeError('Tellonce background processing stopped; audit the phase before continuing')

    def ready_publication(self):
        # Background failure is recorded and stops new tasks/freeze. A current
        # task can still consume its last complete, coherent publication.
        self.check_health()
        self.api.verify()
        if self._closed:
            raise ValueError('Tellonce phase owner is closed')
        if _read(self.directory/'method.json',1024*1024)!=self._manifest:
            raise ValueError('Tellonce method configuration changed during its phase')
        if self.learner is None:
            if self._receipt is None:
                raise ValueError('no frozen Tellonce publication is available')
            return self.api.Publication.from_json((self.directory/'library/frozen.json').read_text(),
                                                  expected_namespace=self.identity.namespace)
        return self.api.load_publication(self.learner.publications,expected_namespace=self.identity.namespace)

    def __call__(self,*,start,directory,record):
        self.assert_healthy()
        _directory(directory)
        training=self.learner is not None
        if (not isinstance(start,dict) or start.get('namespace')!=self.identity.namespace or
                start.get('training') is not training or (training and not self._accepting) or
                start.get('session_id') in self._sessions or directory.exists()):
            raise ValueError('a Tellonce port requires a fresh task in this exact open phase')
        if not training:
            self.verify_frozen()
        publication=self.ready_publication()
        runtime=self.api.TaskMemory(publication,expected_namespace=self.identity.namespace,task_id=start['task_id'],
            public_task='',invoke=self.invoke,training=training,execution_enabled=self.execution_enabled)
        port=TellonceSession(start=start,runtime=runtime,learner=_Enqueuer(self) if training else None,
            ready_publication=self.ready_publication if training else None,transcript=directory/'transcript.jsonl',
            capture_reference=self.api.capture_reference,record=record,verify_frozen=self.verify_frozen if not training else None)
        self._sessions[start['session_id']]=port
        return port

    def start_background(self):
        self.assert_healthy()
        if not self._accepting or self.learner is None or self._thread is not None:
            raise ValueError('start one background processor for an accepting training phase')
        def work():
            try:
                while not self._stop.is_set():
                    self._wake.wait(.25)
                    if self._stop.is_set():
                        return
                    if not self._wake.is_set():
                        continue
                    self._wake.clear()
                    while not self._stop.is_set():
                        self.check_health();self.api.verify()
                        started=time.monotonic()
                        result=self.learner.process_pending(limit=1)
                        self.record({'type':'tellonce_background_batch','namespace':self.identity.namespace,
                            'phase':'training','wall_s':time.monotonic()-started,'result':result})
                        if not result['remaining']:
                            break
            except Exception as exc:
                self._failure=type(exc).__name__
                self.record({'type':'tellonce_background_failed','namespace':self.identity.namespace,
                             'phase':'training','error_type':self._failure})
        self._thread=threading.Thread(target=work,daemon=True,name='tellonce-phase-learning')
        self._thread.start()

    def _stop_background(self):
        self._stop.set();self._wake.set()
        deadline=time.monotonic()+max(30,self.invoke.deadline_s+10)
        while self.background_alive:
            self.check_health()
            if time.monotonic()>=deadline:
                raise RuntimeError('Tellonce background processor did not finish; phase must remain interrupted')
            self._thread.join(timeout=.1)

    def finish_training(self):
        self.assert_healthy()
        if self.learner is None or not self._accepting or any(not port._closed for port in self._sessions.values()):
            raise ValueError('close all training task sessions before finishing and freezing Tellonce')
        self._accepting=False
        self._stop_background()
        self.assert_healthy()
        started=time.monotonic()
        publication=self.learner.finish_training(max_passes=max(10,len(self._event_ids)+1),batch_limit=20)
        audit=_read(self.directory/'library/freeze-audit.json',1024*1024)
        value={'schema':1,'manifest_hash':content_hash(self._manifest),'publication_digest':publication.digest,
               'library_hash':tree_hash(self.directory/'library'),'freeze_audit':audit}
        self.authorize('freeze_training_method',{'namespace':self.identity.namespace,'binding':self.binding,
                                               'receipt':content_hash(value),'audit':audit})
        _write(self.directory/'frozen-method.json',value)
        self._receipt=value
        self.record({'type':'tellonce_training_frozen','namespace':self.identity.namespace,'phase':'training',
                     'wall_s':time.monotonic()-started,'receipt':content_hash(value),'audit':audit})
        return content_hash(value)

    def verify_frozen(self):
        self.api.verify()
        if (self._receipt is None or self._closed or
                _read(self.directory/'method.json',1024*1024)!=self._manifest or
                _read(self.directory/'frozen-method.json',1024*1024)!=self._receipt or
                tree_hash(self.directory/'library')!=self._receipt['library_hash']):
            raise ValueError('frozen Tellonce library changed or has no completed training receipt')
        value=self.api.Publication.from_json((self.directory/'library/frozen.json').read_text(),
                                            expected_namespace=self.identity.namespace)
        if value.digest!=self._receipt['publication_digest']:
            raise ValueError('frozen Tellonce publication differs from its training audit')

    def close(self):
        if self._closed:
            return
        self._accepting=False
        self._stop_background()
        for port in self._sessions.values():
            port.close()
        if self.learner is None:
            self.verify_frozen()
        self._closed=True
