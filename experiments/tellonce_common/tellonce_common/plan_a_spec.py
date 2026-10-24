"""A fresh test owner using an intact, completed training run.

The source run's identity/binding remains authoritative for frozen memories.
The extension's own digest binds protocol, current profile and new output
paths. Neither old test answers nor training files are copied/rewritten.
"""
from copy import deepcopy
from dataclasses import asdict

from .compact_profiles import PROFILE_VERSION
from .frozen_task_evaluation import _read, _directory
from .provenance import content_hash
from .test_protocol import PLAN_A, DELIVERY_RUBRIC


def training_profile_hashes(source):
    return {content_hash(asdict(source.profile))}


def frozen_training(source, expected_receipt):
    """Validate small phase closures and original training ownership, without transcripts."""
    if source.identity.method=='no_memory':
        if expected_receipt is not None:
            raise ValueError('no-memory test has no training receipt')
        return None
    root=source.directory/'training-phase'
    result=_read(root/'phase-result.json',1024*1024)
    manifest=_read(root/'phase.json',8*1024*1024)
    closed=_read(root/'closed-phase.json',8*1024*1024)
    claw=source.identity.benchmark=='clawarena'
    planned=[]
    profile_hashes=training_profile_hashes(source)
    tasks=manifest.get('tasks',[])
    expected_rows=[r for r in source.value['tasks'] if r['split']=='acquisition']
    if len(tasks)!=len(expected_rows) or not tasks:
        raise ValueError('training task count differs from the admitted source')
    for row,actual in zip(expected_rows,tasks):
        tid=row['scenario_id'] if claw else 'group_travel_planner:'+row['row_id']
        expected={'task_id':tid,'directory':str(source.directory/('task-'+tid if claw else 'task-'+content_hash(tid))),
                  'units_hash':content_hash([asdict(u) for u in source.units[tid]]),
                  'unit_ids':[u.instance.common_instance_id for u in source.units[tid]]}
        if claw:
            expected.update(asset_digest=row['asset_digest'],evaluation_digest=row['evaluation_digest'],
                            profile_digest=actual.get('profile_digest'))
            profiles=[actual.get('profile_digest')]
        else:
            expected.update(row_hash=content_hash(source.rows[tid]),profile_hash=actual.get('profile_hash'),
                            actual_profile_hash=actual.get('actual_profile_hash'))
            profiles=[actual.get('profile_hash'),actual.get('actual_profile_hash')]
        if actual!=expected or any(p not in profile_hashes for p in profiles) or len(set(profiles))!=1:
            raise ValueError('training plan or pinned training profile differs from the original receipt')
        planned.append(tid)
    if (content_hash(result)!=expected_receipt or manifest.get('namespace')!=source.identity.namespace or
            manifest.get('binding')!=source.binding or manifest.get('phase')!='training' or
            result.get('manifest_hash')!=content_hash(manifest) or closed.get('manifest_hash')!=content_hash(manifest) or
            result.get('closed_hash')!=content_hash(closed) or
            [r.get('task_id') for r in closed.get('tasks',[])]!=planned or
            any(r.get('aborted') is not False for r in closed['tasks']) or
            not isinstance(closed.get('memory_receipt'),str) or len(closed['memory_receipt'])!=64):
        raise ValueError('test requires the exact completed training and frozen memory receipts')
    return closed


class PlanARunSpec:
    @classmethod
    def load(cls, *, path, expected_digest=None):
        from tellonce_clawarena.native_run_spec import ClawRunSpec, _path
        from tellonce_memoryarena.native_travel_run_spec import TravelRunSpec
        result=cls();result.path=_path(str(path));value=_read(result.path,2*1024*1024)
        if expected_digest is None:
            expected_digest=content_hash(value)
        fields={'schema','purpose','test_protocol','preference_rubric','profile_version','profile_hash',
                'source_kind','source_spec','source_binding','source_runtime_hash','training_receipt',
                'memory_receipt','directory','tasks','test_task_workers'}
        if isinstance(value, dict) and 'public_asset_digests' in value:
            fields.add('public_asset_digests')
        loaders={'clawarena':ClawRunSpec,'memoryarena_travel':TravelRunSpec}
        if (not isinstance(value,dict) or set(value)!=fields or value['schema']!=1 or
                value['purpose']!='frozen-memory-plan-a-test' or value['test_protocol']!=PLAN_A or
                value['preference_rubric']!=DELIVERY_RUBRIC or value['profile_version']!=PROFILE_VERSION or
                value['source_kind'] not in loaders or content_hash(value)!=expected_digest or
                type(value['test_task_workers']) is not int or not 1<=value['test_task_workers']<=4):
            raise ValueError('a complete frozen-memory test specification is required')
        source=loaders[value['source_kind']].load(path=_path(value['source_spec']),expected_digest=value['source_binding'])
        if (content_hash(source.runtime_bundle_reference)!=value['source_runtime_hash'] or
                content_hash(asdict(source.profile))!=value['profile_hash']):
            raise ValueError('source runtime settings or test profile differ from the test specification')
        result.source=source
        result._training_owner=source
        result.training_source=source
        result.identity,result.binding=source.identity,source.binding
        result.extension,result.spec_digest=value,expected_digest
        result.test_protocol=PLAN_A
        result.directory=_path(value['directory']);_directory(result.directory)
        if (result.directory==source.directory or result.directory.is_relative_to(source.directory) or
                source.directory.is_relative_to(result.directory)):
            raise ValueError('new tests must not overwrite or reuse the training directory')
        expected=[r for r in source.value['tasks'] if r['split']!='acquisition']
        if value['tasks']!=expected or not expected:
            raise ValueError('the test retains the complete test plan of its training run')
        if 'public_asset_digests' in value:
            from tellonce_clawarena.native_workspace import load_claw_assets
            digests = value['public_asset_digests']
            if (value['source_kind'] != 'clawarena' or not isinstance(digests, dict) or
                    set(digests) != {row['scenario_id'] for row in expected}):
                raise ValueError('test public assets must cover exactly the original test tasks')
            result.assets = dict(source.assets)
            for sid, digest in digests.items():
                assets = load_claw_assets(source.dataset_root, sid)
                if assets.digest != digest:
                    raise ValueError('test public assets differ from the test specification')
                result.assets[sid] = assets
        result.bundle=source.bundle
        result.prior=frozen_training(source,value['training_receipt'])
        if value['memory_receipt']!=(None if result.prior is None else result.prior['memory_receipt']):
            raise ValueError('the test must load the frozen memory of its training run')
        result.value={**deepcopy(source.value),'directory':str(result.directory),'tasks':deepcopy(expected),
                      'score_before_repair':False}
        # Runtime settings (task timeouts, auxiliary roles) carry over from
        # training; only the test concurrency is added.
        result.runtime_bundle_reference={**deepcopy(source.runtime_bundle_reference),
                                         'test_task_workers':value['test_task_workers']}
        result.runtime_revision_hash=expected_digest
        result._derived_hash=content_hash([result.value,result.runtime_bundle_reference])
        result.verify()
        return result

    def __getattr__(self,name):
        return getattr(self.source,name)

    def verify(self):
        from tellonce_clawarena.native_run_spec import _path
        if (content_hash(_read(self.path,2*1024*1024))!=self.spec_digest or
                content_hash(self.extension)!=self.spec_digest or self.identity!=self.source.identity or
                self.binding!=self.source.binding or self.directory!=_path(self.extension['directory']) or
                self.value['tasks']!=self.extension['tasks'] or
                content_hash([self.value,self.runtime_bundle_reference])!=self._derived_hash or
                content_hash(asdict(self.profile))!=self.extension['profile_hash']):
            raise ValueError('test owner or profile changed while running')
        self.source.verify()
        for sid, digest in self.extension.get('public_asset_digests', {}).items():
            if sid not in self.assets or self.assets[sid].digest != digest:
                raise ValueError('test public assets changed while running')
        frozen_training(self._training_owner,self.extension['training_receipt'])
