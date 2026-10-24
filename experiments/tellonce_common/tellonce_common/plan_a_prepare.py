"""Prepare the frozen-memory test specification after training, without model calls."""
from dataclasses import asdict

from .compact_profiles import PROFILE_VERSION
from .deferred_scoring import _write
from .frozen_task_evaluation import _read
from .plan_a_spec import PlanARunSpec, frozen_training
from .provenance import content_hash
from .test_protocol import PLAN_A, DELIVERY_RUBRIC


def prepare_extension(*,source_kind,source_path,destination,result_directory,test_task_workers=4,
                      repair_public_assets=False):
    from tellonce_clawarena.native_run_spec import ClawRunSpec
    from tellonce_memoryarena.native_travel_run_spec import TravelRunSpec
    loaders={'clawarena':ClawRunSpec,'memoryarena_travel':TravelRunSpec}
    if source_kind not in loaders:
        raise ValueError('unsupported benchmark for a frozen-memory test')
    if result_directory.exists() and not (destination/'test-spec.json').exists():
        raise ValueError('the test output directory exists without its test specification')
    if type(test_task_workers) is not int or not 1 <= test_task_workers <= 4:
        raise ValueError('one to four independent test tasks may run at once')
    if type(repair_public_assets) is not bool or (repair_public_assets and source_kind != 'clawarena'):
        raise ValueError('public asset repair is an explicit ClawArena test-only choice')
    if destination!=destination.resolve() or result_directory!=result_directory.resolve():
        raise ValueError('canonical preparation and output paths required')
    source=loaders[source_kind].load(path=source_path,expected_digest=None)
    if destination==source.directory or destination.is_relative_to(source.directory) or \
            result_directory.is_relative_to(source.directory):
        raise ValueError('preparation cannot write into the training directory')
    training_receipt=content_hash(_read(source.directory/'training-phase/phase-result.json',1024*1024))
    prior=frozen_training(source,training_receipt)
    value={'schema':1,'purpose':'frozen-memory-plan-a-test','test_protocol':PLAN_A,
        'preference_rubric':DELIVERY_RUBRIC,'profile_version':PROFILE_VERSION,
        'profile_hash':content_hash(asdict(source.profile)),'source_kind':source_kind,
        'source_spec':str(source_path),'source_binding':source.binding,
        'source_runtime_hash':content_hash(source.runtime_bundle_reference),
        'training_receipt':training_receipt,'memory_receipt':prior['memory_receipt'],
        'directory':str(result_directory),'tasks':[r for r in source.value['tasks'] if r['split']!='acquisition'],
        'test_task_workers':test_task_workers}
    if repair_public_assets:
        from tellonce_clawarena.native_workspace import load_claw_assets
        value['public_asset_digests'] = {
            row['scenario_id']: load_claw_assets(source.dataset_root, row['scenario_id']).digest
            for row in value['tasks']}
    destination.mkdir(parents=True,mode=0o700,exist_ok=True)
    path=destination/'test-spec.json'
    if path.exists():
        if _read(path,2*1024*1024)!=value:
            raise ValueError('a different test specification already exists')
    else:
        _write(path,value)
    return PlanARunSpec.load(path=path,expected_digest=content_hash(value))
