"""Bind private native scores to a complete, pinned ClawArena task manifest.

Real-user preference judgments are separate. Original native `pref` commands
are optional private diagnostics and never become user feedback or task scores.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from tellonce_common.contracts import Split
from tellonce_common.interaction_loop import SolverAnswer
from tellonce_common.provenance import content_hash, file_hash
from tellonce_common.role_inputs import PublicUnit
from .clawarena_loader import load_scenario, project_native_scenario
from .native_exec_scoring import CommandSpec, NativeCommandGrader
from .native_mc_scoring import score_multiple_choice
from .schema import MULTI_CHOICE, EXEC_CHECK


@dataclass(frozen=True)
class NativeScore:
    score: float
    passed: bool
    artifact_hash: str


class NativeScenarioScorer:
    def __init__(self, *, units: tuple[PublicUnit, ...], grader: NativeCommandGrader,
                 source_root: Path, record):
        if (not isinstance(units, tuple) or not units or
                any(not isinstance(unit, PublicUnit) for unit in units) or
                not isinstance(grader, NativeCommandGrader) or not callable(record) or
                not isinstance(source_root, Path) or not source_root.is_absolute() or
                source_root != source_root.resolve() or
                grader.eval_root != source_root/'data/clawarena/eval'):
            raise ValueError('a complete native task and its matching private scorer are required')
        first = units[0].instance
        scenario = load_scenario(source_root, grader.start['task_id'])
        expected, references = project_native_scenario(scenario, dataset_revision=first.dataset_revision,
            task_family=first.task_family, split=first.split, user_id=grader.identity.user_id, seed=first.seed)
        if (units != expected or first.split not in {Split.ACQUISITION, Split.ID_TEST, Split.TASK_OOD} or
                (first.split == Split.ACQUISITION) != grader.start['training']):
            raise ValueError('scoring units must match all original task rounds and the actual phase')
        # Freeze strings rather than retain mutable nested native dictionaries.
        self._unit_hashes = {unit.instance.common_instance_id: content_hash(asdict(unit)) for unit in units}
        self._references = {reference.unit_id: reference.evaluation_json for reference in references}
        self._questions = grader.eval_root/grader.start['task_id']/'questions.json'
        self._question_hash = file_hash(self._questions)
        self._sources = tuple(Path(__file__).with_name(name).resolve() for name in
                              ('native_scoring.py', 'native_mc_scoring.py', 'clawarena_loader.py', 'schema.py'))
        self._binding = {str(path):file_hash(path) for path in self._sources}
        self.grader, self.record = grader, record
        # Reject an unsupported score specification before any task/model work.
        for reference in self._references.values():
            value = json.loads(reference)
            if value['round_type'] == MULTI_CHOICE:
                score_multiple_choice('', expected=value['eval']['answer'], options=value['eval']['options'])
            elif value['round_type'] == EXEC_CHECK:
                CommandSpec.from_native(value['eval'])
            if value['native_pref']:
                CommandSpec.from_native(value['native_pref'])
        grader.authorize('prepare', {'namespace':grader.identity.namespace, 'kind':'native_scenario',
            'questions_hash':self._question_hash, 'unit_hashes':dict(self._unit_hashes), 'sources':dict(self._binding)})

    def _check(self, unit, answer, context):
        fields = {'phase','task_id','session_id','event_id','role','call_id','attempt'}
        if (not isinstance(unit, PublicUnit) or not isinstance(answer, SolverAnswer) or
                self._unit_hashes.get(unit.instance.common_instance_id) != content_hash(asdict(unit)) or
                not isinstance(context, dict) or set(context) != fields or
                any(not isinstance(context.get(k),str) or not 1 <= len(context[k]) <= 480 for k in fields-{'attempt'}) or
                context.get('task_id') != self.grader.start['task_id'] or
                context.get('session_id') != self.grader.start['session_id'] or
                context.get('phase') != ('training' if self.grader.start['training'] else 'test') or
                context.get('role') not in {'score', 'score_before_repair'} or
                not isinstance(context.get('event_id'), str) or not context['event_id'] or
                type(context.get('attempt')) is not int or context['attempt'] not in {1,2}):
            raise ValueError('native score belongs to another task, unit, session or delivery')
        expected_call = content_hash([self.grader.identity.namespace, context['session_id'],
                                      unit.instance.common_instance_id, context['event_id'], context['role']])
        if context.get('call_id') != expected_call:
            raise ValueError('native scoring call must bind its actual unit and delivery')
        if (file_hash(self._questions) != self._question_hash or
                any(file_hash(path) != self._binding[str(path)] for path in self._sources)):
            raise ValueError('native questions or scoring sources changed after score preparation')
        if self.grader.artifacts.load(answer.artifact_hash)['answer'] != answer.text:
            raise ValueError('native scoring requires the captured delivery text')
        return json.loads(self._references[unit.instance.common_instance_id])

    def score(self, unit: PublicUnit, answer: SolverAnswer, *, context: dict) -> NativeScore:
        value = self._check(unit, answer, context)
        if value['round_type'] == MULTI_CHOICE:
            self.grader.authorize('check', {'namespace': self.grader.identity.namespace,
                'kind': 'native_mc', 'unit_id': unit.instance.common_instance_id,
                'artifact_hash': answer.artifact_hash, 'questions_hash': self._question_hash,
                'sources':dict(self._binding), **context})
            result = score_multiple_choice(answer.text, expected=value['eval']['answer'], options=value['eval']['options'])
            self.record({'type':'native_mc_completed', 'unit_id':unit.instance.common_instance_id,
                         'artifact_hash':answer.artifact_hash, 'context':dict(context), **asdict(result)})
            return NativeScore(result.score, result.passed, answer.artifact_hash)
        result = self.grader.check(CommandSpec.from_native(value['eval']), answer, context=context)
        return NativeScore(float(result.passed), result.passed, answer.artifact_hash)

    def native_preference_diagnostic(self, unit: PublicUnit, answer: SolverAnswer, *, context: dict) -> bool | None:
        value = self._check(unit, answer, context)
        if not value['native_pref']:
            return None
        result = self.grader.check(CommandSpec.from_native(value['native_pref']), answer,
                                   context=context, kind='native_pref')
        return result.passed
