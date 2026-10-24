"""Score a sealed plan A phase without opening a solver or a memory learner."""
from pathlib import Path

from .claude_auth import ClaudeCredentialSource
from .claude_native import ClaudeConfig
from .deferred_scoring import _write
from .experiment_summary import summarize_plan_a_outcomes
from .preference_transport import PreferenceJudge
from .provenance import content_hash, file_hash, canonical_json_bytes
from .provider_credentials import ProviderKeyFile
from .task_outcomes import outcome_from_dict
from .text_roles import TextRoles
from .preference_repair import REVISION


def evaluate_extension(spec,*,prior_receipt,signals,record,attempt):
    from tellonce_clawarena.native_worker import NativeCredentials,verify_execution
    from tellonce_memoryarena.native_travel_worker import verify_execution as verify_memory_execution
    spec.verify();verify_execution(spec);verify_memory_execution(spec)
    if attempt.parent!=spec.directory/'worker-attempts':
        raise ValueError('plan A evaluator must use its private result owner')
    attempt.mkdir(mode=0o700)
    authorize=lambda *a:signals.assert_healthy()
    credentials=NativeCredentials(spec,directory=attempt,authorize=authorize,record=record)
    try:
        providers={row['kind'] for row in spec.value['models']['roles'].values()
                   if row['kind'] not in {'native','claude_native'}}
        if len(providers)>1:
            raise ValueError('one explicitly admitted auxiliary provider required')
        key=ProviderKeyFile(spec.api_key_file,provider=next(iter(providers)),authorize=authorize) if providers else None
        claude=ClaudeCredentialSource(path=spec.claude_auth_file,authorize=authorize) if isinstance(spec.solver_config,ClaudeConfig) else None
        role_dir=spec.directory/('evaluation-roles-'+REVISION)
        args=dict(directory=role_dir,identity=spec.identity,binding=spec.binding,
            model_config=spec.value['models'],binary=spec.binary,catalog=spec.catalog,tokenizer=spec.tokenizer,
            authorize=authorize,record=record,check_health=signals.assert_healthy,
            provision_auth=credentials.provision,api_key_source=key)
        if claude is not None:
            args.update(claude_binary=spec.claude_binary,claude_credentials=claude)
        roles=(TextRoles.open if (role_dir/'roles.json').exists() else TextRoles.create)(**args)
        from .api_luna_roles import configure_api_luna_roles
        roles=configure_api_luna_roles(spec,roles,phase='test')
        judge=PreferenceJudge(roles.get('preference'))
        shared=dict(directory=spec.directory/'test-phase',expected_receipt=prior_receipt,
                    identity=spec.identity,binding=spec.binding,authorize=authorize,record=record,
                    check_health=signals.assert_healthy,repair_preferences=True)
        if spec.identity.benchmark=='clawarena':
            from tellonce_clawarena.native_task import open_claw_scorer
            from tellonce_clawarena.native_phase import evaluate_claw_phase
            def scorer(result):
                sid=result.outcome.task_id
                return open_claw_scorer(result=result,identity=spec.identity,units=spec.units[sid],profile=spec.profile,
                    source_root=spec.dataset_root,expected_asset_digest=spec.assets[sid].digest,binding=spec.binding,
                    binary=spec.binary,python=spec.python,python_roots=spec.python_roots,
                    directory=attempt/('grader-'+sid),preference_judge=judge,authorize=authorize,record=record,
                    wait_retry=signals.wait_retry,role_deadline_s=spec.value['role_deadline_s'])
            value=evaluate_claw_phase(**shared,scorer_factory=scorer)
        else:
            from tellonce_memoryarena.native_travel_evaluation import TravelTaskEvaluation
            from tellonce_memoryarena.native_travel_phase import evaluate_travel_phase as evaluate
            def evaluator(result):
                sid=result.outcome.task_id
                return TravelTaskEvaluation(result=result,identity=spec.identity,units=spec.units[sid],row=spec.rows[sid],
                    expected_row_hash=content_hash(spec.rows[sid]),profile=spec.profile,binding=spec.binding,
                    official_source=spec.official_source,record=record)
            value=evaluate(**shared,evaluator_factory=evaluator,preference_judge=judge,
                           wait_retry=signals.wait_retry,role_deadline_s=spec.value['role_deadline_s'])
        outcomes=[outcome_from_dict(row) for row in value['outcomes']]
        from .parallel_test import task_map
        planned={tid:[u.instance.common_instance_id for u in spec.units[tid]] for tid in task_map(spec)}
        summary=summarize_plan_a_outcomes(planned,outcomes)
        summary.update(spec_digest=spec.spec_digest,profile_version=spec.extension['profile_version'],
            preference_rubric=spec.extension['preference_rubric'],source_memory_receipt=spec.extension['memory_receipt'])
        if 'native_endpoints' in value:
            summary['native_task_endpoint']=value['native_endpoints']['final']
        summary['preference_implementation']=REVISION
        if getattr(roles,'api_luna_roles',False):
            summary['auxiliary_model_policy']=roles.api_luna_policy
            summary['preference_model_provenance']='Historical completed judgments retained; new calls use OpenAI gpt-6-luna low. Per-request provenance is in the role journals.'
        summary['preference_recovery']='retain_v3_v4_v5_judgments_retry_proven_native_outages'
        suffix='-'+roles.api_luna_policy if getattr(roles,'api_luna_roles',False) else ''
        path=spec.directory/('plan-a-summary-'+REVISION+suffix+'.json')
        if not path.exists():
            _write(path,summary)
        else:
            import json
            if json.loads(path.read_text())!=summary:
                raise ValueError('completed plan A summary changed during replay')
        from .atomic_io import atomic_write_bytes
        pointer={'schema':1,'summary_file':path.name,'sha256':file_hash(path),
                 'preference_implementation':REVISION}
        atomic_write_bytes(spec.directory/'plan-a-summary-current.json',canonical_json_bytes(pointer))
        return {'action':'test_evaluation','phase_receipt':prior_receipt,'evaluation_digest':content_hash(value)}
    finally:
        credentials.close()
