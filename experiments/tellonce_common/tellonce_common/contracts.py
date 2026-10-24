from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping

from .provenance import content_hash


SCHEMA_VERSION = "tellonce.v2"
CONDITION_ALIAS_VERSION = "tellonce.v2.aliases.1"


class Split(str, Enum):
    ACQUISITION = "acquisition"
    ID_TEST = "id_test"
    TASK_OOD = "task_ood"
    PREFERENCE_OOD = "preference_ood"


class RuleDelivery(str, Enum):
    NONE = "none"
    RAW_RETRIEVAL = "raw_retrieval"
    MEMORY_RETRIEVAL = "memory_retrieval"
    COMPILED_PROMPT = "compiled_prompt"


class EnforcementMode(str, Enum):
    NONE = "none"
    VERIFIER_RETRY = "verifier_retry"
    NATIVE_HOOK = "native_hook"


class EnforcementDecision(str, Enum):
    PASS = "pass"
    RETRY_SUCCESS = "retry_success"
    RETRY_FAILED_BLOCKED = "retry_failed_blocked"
    RETRY_INVALID_BLOCKED = "retry_invalid_blocked"
    BLOCKED_WITHOUT_RETRY = "blocked_without_retry"
    ERROR = "error"


class InformationLevel(IntEnum):
    PUBLIC_INSTANCE = 0
    LEARNED_CORRECTIONS = 1
    HIDDEN_PREFERENCES = 2
    GOLD_ANSWER = 3


class MetricKind(str, Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    COUNT = "count"


@dataclass(frozen=True)
class MetricDescriptor:
    name: str
    kind: MetricKind
    higher_is_better: bool
    semantics: str
    primary: bool = False
    minimum: float | None = None
    maximum: float | None = None
    runtime_failure_value: float | int | bool | None = None
    blocked_failure_value: float | int | bool | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.semantics:
            raise ValueError("metric name and semantics are required")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("metric minimum cannot exceed maximum")


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    rule_delivery: RuleDelivery
    enforcement: EnforcementMode
    initial_generation_group: str
    max_repairs: int = 0
    real_backend_required: bool = False
    allowed_backend_modes: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    legacy_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.initial_generation_group:
            raise ValueError("condition name and generation group are required")
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")
        if self.enforcement == EnforcementMode.NONE and self.max_repairs:
            raise ValueError("non-enforcement conditions cannot request repairs")
        if self.enforcement != EnforcementMode.NONE and self.max_repairs < 1:
            raise ValueError("active enforcement requires at least one repair")
        if self.real_backend_required and self.rule_delivery != RuleDelivery.MEMORY_RETRIEVAL:
            raise ValueError("real backend requirement is only valid for memory retrieval")
        if self.real_backend_required and not self.allowed_backend_modes:
            raise ValueError("real backend condition requires an explicit mode allowlist")
        if not self.real_backend_required and self.allowed_backend_modes:
            raise ValueError("backend mode allowlist requires a real backend condition")


@dataclass(frozen=True)
class CommonInstance:
    common_instance_id: str
    benchmark: str
    dataset_revision: str
    split: Split
    task_family: str
    instance_id: str
    subtask_id: str
    public_input_hash: str
    seed: int
    cluster_id: str
    profile_id: str = ""
    applicable_preference_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("common_instance_id", self.common_instance_id),
            ("benchmark", self.benchmark),
            ("dataset_revision", self.dataset_revision),
            ("task_family", self.task_family),
            ("instance_id", self.instance_id),
            ("subtask_id", self.subtask_id),
            ("public_input_hash", self.public_input_hash),
            ("cluster_id", self.cluster_id),
        ):
            if not value:
                raise ValueError(f"{field_name} is required")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @classmethod
    def build(
        cls,
        *,
        benchmark: str,
        dataset_revision: str,
        split: Split,
        task_family: str,
        instance_id: str,
        subtask_id: str,
        public_input_hash: str,
        seed: int,
        cluster_id: str,
        profile_id: str = "",
        applicable_preference_ids: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "CommonInstance":
        identity = {
            "benchmark": benchmark,
            "dataset_revision": dataset_revision,
            "split": split.value,
            "task_family": task_family,
            "instance_id": instance_id,
            "subtask_id": subtask_id,
            "public_input_hash": public_input_hash,
            "seed": seed,
            "cluster_id": cluster_id,
            "profile_id": profile_id,
        }
        return cls(
            common_instance_id=content_hash(identity),
            benchmark=benchmark,
            dataset_revision=dataset_revision,
            split=split,
            task_family=task_family,
            instance_id=instance_id,
            subtask_id=subtask_id,
            public_input_hash=public_input_hash,
            seed=seed,
            cluster_id=cluster_id,
            profile_id=profile_id,
            applicable_preference_ids=tuple(sorted(applicable_preference_ids)),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["split"] = self.split.value
        return value


@dataclass(frozen=True)
class ExperimentContract:
    protocol_id: str
    benchmark: str
    dataset_revision: str
    dataset_source_hash: str
    runtime_protocol_id: str
    scorer_id: str
    simulator_id: str
    evaluator_id: str
    conditions: tuple[ConditionSpec, ...]
    metrics: tuple[MetricDescriptor, ...]
    cluster_fields: tuple[str, ...]
    role_information_limits: Mapping[str, InformationLevel]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, value in (
            ("protocol_id", self.protocol_id),
            ("benchmark", self.benchmark),
            ("dataset_revision", self.dataset_revision),
            ("dataset_source_hash", self.dataset_source_hash),
            ("runtime_protocol_id", self.runtime_protocol_id),
            ("scorer_id", self.scorer_id),
            ("simulator_id", self.simulator_id),
            ("evaluator_id", self.evaluator_id),
        ):
            if not value:
                raise ValueError(f"{field_name} is required")
        condition_names = [condition.name for condition in self.conditions]
        if len(condition_names) != len(set(condition_names)):
            raise ValueError("condition names must be unique")
        aliases = [
            alias for condition in self.conditions for alias in condition.legacy_aliases
        ]
        if len(aliases) != len(set(aliases)):
            raise ValueError("condition aliases must be unique")
        if set(aliases) & set(condition_names):
            raise ValueError("condition aliases cannot shadow canonical names")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("metric names must be unique")
        if not any(metric.primary for metric in self.metrics):
            raise ValueError("at least one primary metric is required")
        delivery_by_generation_group: dict[str, RuleDelivery] = {}
        for condition in self.conditions:
            prior = delivery_by_generation_group.setdefault(
                condition.initial_generation_group,
                condition.rule_delivery,
            )
            if prior != condition.rule_delivery:
                raise ValueError(
                    "conditions sharing an initial generation must use the same "
                    "rule delivery"
                )
        for role in ("agent", "simulator", "runtime_verifier", "evaluator"):
            if role not in self.role_information_limits:
                raise ValueError(f"missing information limit for role {role}")
        if self.role_information_limits["agent"] > InformationLevel.LEARNED_CORRECTIONS:
            raise ValueError("agent information limit cannot include hidden preferences")
        if (
            self.role_information_limits["runtime_verifier"]
            > InformationLevel.LEARNED_CORRECTIONS
        ):
            raise ValueError(
                "runtime verifier information limit cannot include hidden preferences"
            )
        if self.role_information_limits["simulator"] > InformationLevel.HIDDEN_PREFERENCES:
            raise ValueError("simulator information limit cannot include gold answers")

    @property
    def contract_hash(self) -> str:
        value = asdict(self)
        value["conditions"] = [
            {
                **asdict(condition),
                "rule_delivery": condition.rule_delivery.value,
                "enforcement": condition.enforcement.value,
            }
            for condition in self.conditions
        ]
        value["metrics"] = [
            {**asdict(metric), "kind": metric.kind.value} for metric in self.metrics
        ]
        value["role_information_limits"] = {
            role: int(level) for role, level in self.role_information_limits.items()
        }
        return content_hash(value)

    def condition(self, name: str) -> ConditionSpec:
        for condition in self.conditions:
            if condition.name == name:
                return condition
        raise KeyError(f"unknown condition: {name}")


@dataclass(frozen=True)
class EnforcementEvidence:
    mode: EnforcementMode
    verifier_id: str
    verifier_called: bool
    verifier_information_level: InformationLevel
    verifier_input_hash: str
    verifier_input_payload: str
    verifier_output_hash: str
    verifier_input_schema: str
    verifier_input_sources: tuple[str, ...]
    decision: EnforcementDecision
    pre_artifact_hash: str
    final_artifact_hash: str
    repair_count: int
    reverified: bool
    reverified_artifact_hash: str = ""
    reverifier_input_hash: str = ""
    reverifier_input_payload: str = ""
    reverifier_output_hash: str = ""
    hook_event_id: str = ""
    hook_config_hash: str = ""
    rollback_verified: bool | None = None
    verifier_output_payload: str = ""
    reverifier_output_payload: str = ""
    correction_ledger_id: str = ""
    correction_ledger_hash: str = ""
    learned_corrections_hash: str = ""
    verifier_backend_mode: str = ""


def default_v2_conditions() -> tuple[ConditionSpec, ...]:
    return (
        ConditionSpec(
            name="no_persistence",
            rule_delivery=RuleDelivery.NONE,
            enforcement=EnforcementMode.NONE,
            initial_generation_group="no_rules",
            legacy_aliases=("no_memory",),
        ),
        ConditionSpec(
            name="runtime_only_verifier",
            rule_delivery=RuleDelivery.NONE,
            enforcement=EnforcementMode.VERIFIER_RETRY,
            initial_generation_group="no_rules",
            max_repairs=1,
            required_capabilities=("verifier_call", "artifact_fork"),
        ),
        ConditionSpec(
            name="compiled_prompt_rules",
            rule_delivery=RuleDelivery.COMPILED_PROMPT,
            enforcement=EnforcementMode.NONE,
            initial_generation_group="compiled_rules",
            required_capabilities=("compiled_rule_delivery",),
            legacy_aliases=("compiled_enforcement",),
        ),
        ConditionSpec(
            name="compiled_runtime_verifier",
            rule_delivery=RuleDelivery.COMPILED_PROMPT,
            enforcement=EnforcementMode.VERIFIER_RETRY,
            initial_generation_group="compiled_rules",
            max_repairs=1,
            required_capabilities=(
                "compiled_rule_delivery",
                "verifier_call",
                "artifact_fork",
            ),
        ),
        ConditionSpec(
            name="raw_correction_retrieval",
            rule_delivery=RuleDelivery.RAW_RETRIEVAL,
            enforcement=EnforcementMode.NONE,
            initial_generation_group="raw_retrieval",
            required_capabilities=("raw_correction_delivery",),
            legacy_aliases=("prompt_memory",),
        ),
    )


def learning_input(user_text, learning_text=None):
    """The text a memory method learns from at a training user boundary: the whole user message, or the verbatim
    lines of it that the simulated user marked as its own preference (2026-10-06, item 50: a mixed correction's
    task-error part is the task's own requirement and is learned by no method). The boundary signature still binds
    the whole message; an excerpt that is not made of its lines is refused."""
    if learning_text is None or learning_text == user_text:
        return user_text
    if not isinstance(learning_text, str) or not learning_text.strip():
        raise ValueError('learning text must be a non-empty excerpt of the user message')
    if any(line.strip() and line.strip() not in user_text for line in learning_text.splitlines()):
        raise ValueError('learning text must consist of verbatim lines of the user message')
    return learning_text
