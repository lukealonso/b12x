"""Contracts shared by the top-level profiler and component generators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from b12x.policy.types import DeviceIdentity, FrozenMapping

JsonObject = Mapping[str, object]


@dataclass(frozen=True, kw_only=True)
class GenerationSettings:
    """Measurement settings applied consistently across all components."""

    warmup: int = 2
    repetitions: int = 5
    groups: int = 5
    seed: int = 20260828
    minimum_cosine: float = 0.998
    cold_l2: bool = True
    timing_clock: str = "cuda_event"
    full_corpus: bool = False
    max_candidate_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in ("warmup", "repetitions", "groups"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not -1.0 <= self.minimum_cosine <= 1.0:
            raise ValueError("minimum_cosine must be in [-1, 1]")
        if self.timing_clock not in ("cuda_event", "globaltimer"):
            raise ValueError("timing_clock must be cuda_event or globaltimer")
        if type(self.full_corpus) is not bool:
            raise TypeError("full_corpus must be a boolean")
        if self.max_candidate_seconds <= 0:
            raise ValueError("max_candidate_seconds must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "warmup": self.warmup,
            "repetitions": self.repetitions,
            "groups": self.groups,
            "seed": self.seed,
            "minimum_cosine": self.minimum_cosine,
            "cold_l2": self.cold_l2,
            "timing_clock": self.timing_clock,
            "full_corpus": self.full_corpus,
            "max_candidate_seconds": self.max_candidate_seconds,
        }


@dataclass(frozen=True, kw_only=True)
class GenerationContext:
    """Stable device, workspace, and measurement inputs for one run."""

    device: DeviceIdentity
    device_ordinal: int
    work_dir: Path
    source_revision: str
    settings: GenerationSettings
    provenance: FrozenMapping = FrozenMapping()
    accepted_physical_devices: tuple[str, ...] = ()
    measurement_cohort: str = "initial"

    def __post_init__(self) -> None:
        if not self.measurement_cohort:
            raise ValueError("generation requires a nonempty measurement cohort")

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            **self.provenance.to_dict(),
            "measurement_cohort": self.measurement_cohort,
            "source_revision": self.source_revision,
            "device": {
                "vendor": self.device.vendor,
                "compute_capability": list(self.device.compute_capability),
                "sm_count": self.device.sm_count,
                "product_name": self.device.product_name,
            },
            "settings": self.settings.to_dict(),
        }

    def checkpoint_metadata_matches(self, value: object) -> bool:
        """Require identical source, toolchain, protocol, and an assigned GPU.

        A coordinator may reduce records from its explicitly assigned physical
        GPUs. Raw observation identities always retain the measuring GPU UUID.
        """

        if not isinstance(value, Mapping):
            return False
        expected = self.checkpoint_metadata()
        if set(value) != set(expected):
            return False
        for name, requested in expected.items():
            if name == "physical_device" and self.accepted_physical_devices:
                if value[name] not in self.accepted_physical_devices:
                    return False
            elif value[name] != requested:
                return False
        return True


@dataclass(frozen=True, kw_only=True)
class WorkEstimate:
    """Preflight estimate used for progress and user-visible scope."""

    component_id: str
    work_units: int
    case_count: int
    description: str
    dimensions: JsonObject

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        if self.work_units < 0 or self.case_count < 0:
            raise ValueError("work estimates cannot be negative")


@dataclass(frozen=True, kw_only=True)
class MeasurementPartition:
    """An independently measurable, checkpoint-disjoint unit of GPU work."""

    component_id: str
    partition_id: str
    work_units: int
    case_count: int
    description: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.partition_id:
            raise ValueError("measurement partition identifiers must be non-empty")
        if self.work_units <= 0 or self.case_count <= 0:
            raise ValueError("measurement partitions must contain positive work")


@dataclass(frozen=True, kw_only=True)
class ComponentGenerationResult:
    """One generated component planner and its reproducibility evidence."""

    component: JsonObject
    evidence: JsonObject
    completed_work_units: int
    completion_reason: str = "exhaustive"
    qualification: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.completed_work_units < 0:
            raise ValueError("completed_work_units cannot be negative")
        if self.completion_reason not in {"exhaustive", "qualified", "budget_exhausted"}:
            raise ValueError("unknown component completion reason")
        if self.completion_reason == "qualified" and (
            self.qualification is None or self.qualification.get("status") != "qualified"
        ):
            raise ValueError("qualified completion requires a passing independent qualification report")


@runtime_checkable
class ProgressReporter(Protocol):
    """Progress surface owned by the top-level tool."""

    def start_component(self, estimate: WorkEstimate) -> None: ...

    def start_stage(
        self,
        component_id: str,
        *,
        stage: str,
        total: int,
    ) -> None: ...

    def advance(
        self,
        component_id: str,
        *,
        units: int = 1,
        detail: str | None = None,
    ) -> None: ...

    def finish_component(self, component_id: str) -> None: ...


@runtime_checkable
class ComponentGenerator(Protocol):
    """Offline provider for one independently plannable runtime component."""

    component_id: str
    query_schema_version: int
    config_schema_version: int

    def estimate(self, context: GenerationContext) -> WorkEstimate: ...

    def generate(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: "CheckpointStore",
    ) -> ComponentGenerationResult: ...


@runtime_checkable
class PartitionableComponentGenerator(Protocol):
    """A generator whose independent measurement work can run concurrently."""

    component_id: str

    def measurement_partitions(
        self,
        context: GenerationContext,
    ) -> tuple[MeasurementPartition, ...]: ...

    def select_measurement_partitions(
        self,
        partition_ids: tuple[str, ...],
    ) -> ComponentGenerator: ...


from .store import CheckpointStore  # noqa: E402

__all__ = [
    "ComponentGenerationResult",
    "ComponentGenerator",
    "GenerationContext",
    "GenerationSettings",
    "JsonObject",
    "MeasurementPartition",
    "PartitionableComponentGenerator",
    "ProgressReporter",
    "WorkEstimate",
]
