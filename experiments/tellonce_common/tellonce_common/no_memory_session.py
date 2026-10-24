"""No long-term memory; ordinary task conversation stays in the native session."""
from .interaction_loop import MemoryContext, SolverAnswer
from .contracts import learning_input
from .provenance import content_hash
from .role_inputs import RunIdentity


class NoMemorySession:
    def __init__(self, *, identity: RunIdentity, start: dict):
        if (not isinstance(identity,RunIdentity) or identity.method!='no_memory' or
                not isinstance(start,dict) or start.get('namespace')!=identity.namespace or
                type(start.get('training')) is not bool or
                any(not isinstance(start.get(k),str) or not start[k] for k in ('task_id','session_id'))):
            raise ValueError('no-memory control requires its own explicit task identity')
        self.start=dict(start)
        self._version=content_hash(['no_memory',identity.namespace])
        self._seen,self._current,self._closed=set(),None,False

    def _input(self,unit_payload,user_text,event_id):
        if (self._closed or not isinstance(unit_payload,dict) or
                set(unit_payload)!={'task_id','unit_id','instruction','context'} or
                unit_payload['task_id']!=self.start['task_id'] or
                not isinstance(user_text,str) or not user_text.strip() or
                not isinstance(event_id,str) or not event_id):
            raise ValueError('no-memory boundary requires the actual public task and user input')
        return content_hash([unit_payload,user_text,event_id])

    def user_boundary(self, *, unit_payload,user_text,event_id):
        signature=self._input(unit_payload,user_text,event_id)
        if event_id in self._seen or self._current is not None:
            raise ValueError('duplicate or unanswered no-memory task boundary')
        self._seen.add(event_id)
        self._current=(signature,False)
        return MemoryContext('',self._version,kind='none')

    def enqueue_user(self, *, unit_payload,user_text,event_id,learning_text=None):
        signature=self._input(unit_payload,user_text,event_id)
        learning_input(user_text,learning_text)
        if not self.start['training'] or self._current!=(signature,False):
            raise ValueError('learning admission must match this training input exactly once')
        # Deliberately no learner, transcript, rule store or model request.
        self._current=(signature,True)

    def observe_assistant(self,answer):
        if self._closed or self._current is None or not isinstance(answer,SolverAnswer):
            raise ValueError('assistant delivery requires an open no-memory user boundary')
        self._current=None

    def close(self):
        self._current=None
        self._closed=True
