"""Runtime settings stored next to a run specification.

`runtime` in a run specification holds execution settings that are not part of
the task plan: the auxiliary-role policy (`api_luna_roles`) and an optional
`task_timeout_s_override`. Code always runs from this repository.
"""
from .source_tree import source_bundle


def resolve_runtime_bundle(spec_path, value, binding):
    return source_bundle(), dict(value.get('runtime') or {}), binding


def load_runtime_revision(spec):
    spec.bundle, spec.runtime_bundle_reference, spec.runtime_revision_hash = resolve_runtime_bundle(
        spec.path, spec.value, spec.binding)


def verify_runtime_revision(spec):
    if resolve_runtime_bundle(spec.path, spec.value, spec.binding) != (
            spec.bundle, spec.runtime_bundle_reference, spec.runtime_revision_hash):
        raise ValueError('runtime settings changed while the worker was running')
