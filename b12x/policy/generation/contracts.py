"""Contracts shared by the top-level profiler and component generators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from b12x.policy.types import DeviceIdentity

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
    max_candidate_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in ("warmup", "repetitions", "groups"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not -1.0 <= self.minimum_cosine <= 1.0:
            raise ValueError("minimum_cosine must be in [-1, 1]")
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

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
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
        """Return whether measurements can be resumed in this context.

        Source revision is retained for provenance, but is not a measurement
        input. Case IDs, candidate IDs, checkpoint schema, device identity, and
        timing settings independently invalidate incompatible measurements.
        A cached sampling protocol may exceed the requested sampling strength.
        """

        if not isinstance(value, Mapping):
            return False
        expected = self.checkpoint_metadata()
        if value.get("device") != expected["device"]:
            return False
        cached_settings = value.get("settings")
        requested_settings = expected["settings"]
        if not isinstance(cached_settings, Mapping):
            return False
        if set(cached_settings) != set(requested_settings):
            return False
        if any(
            cached_settings[field] != requested_settings[field]
            for field in ("seed", "cold_l2")
        ):
            return False
        return all(
            cached_settings[field] >= requested_settings[field]
            for field in (
                "warmup",
                "repetitions",
                "groups",
                "minimum_cosine",
                "max_candidate_seconds",
            )
        )


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

    def __post_init__(self) -> None:
        if self.completed_work_units < 0:
            raise ValueError("completed_work_units cannot be negative")


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
