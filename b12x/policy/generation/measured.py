"""Measured generation for components with one production implementation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from b12x.policy.context import ComponentPolicy
from b12x.policy.types import FrozenMapping

from .contracts import (
    ComponentGenerationResult,
    GenerationContext,
    ProgressReporter,
    WorkEstimate,
)
from .store import CheckpointStore

QueryT = TypeVar("QueryT")
ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True, kw_only=True)
class QualificationOutcome:
    """Qualified single config of a component and its measurement evidence."""

    encoded_config: FrozenMapping
    evidence: dict[str, object]
    completed_work_units: int


@dataclass(frozen=True, kw_only=True)
class GpuProbeMeasurement:
    """One correctness-gated production-path GPU timing."""

    label: str
    latency_us: float
    correct: bool
    metrics: FrozenMapping = FrozenMapping()

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("GPU probe labels must be non-empty")
        if not math.isfinite(self.latency_us) or self.latency_us <= 0:
            raise ValueError("GPU probe latency must be finite and positive")
        if not isinstance(self.correct, bool):
            raise TypeError("GPU probe correctness must be a boolean")
        if not isinstance(self.metrics, FrozenMapping):
            object.__setattr__(self, "metrics", FrozenMapping(self.metrics))

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "latency_us": self.latency_us,
            "correct": self.correct,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GpuProbeMeasurement":
        metrics = value.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise TypeError("GPU probe metrics must be an object")
        correct = value.get("correct")
        if not isinstance(correct, bool):
            raise TypeError("GPU probe correct field must be a boolean")
        return cls(
            label=str(value["label"]),
            latency_us=float(value["latency_us"]),
            correct=correct,
            metrics=FrozenMapping(metrics),
        )


class GpuQualificationProbe(Protocol):
    """Run a production-path benchmark suite for one fixed implementation."""

    @property
    def case_count(self) -> int: ...

    @property
    def case_ids(self) -> tuple[str, ...]: ...

    @property
    def description(self) -> str: ...

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]: ...


class MeasuredPolicyGenerator(Generic[QueryT, ConfigT]):
    """Emit a fixed config only after real GPU qualification succeeds."""

    def __init__(
        self,
        *,
        policy: ComponentPolicy[QueryT, ConfigT],
        queries: Sequence[QueryT],
        encode_config: Callable[[ConfigT], Mapping[str, object]],
        probe: GpuQualificationProbe,
    ) -> None:
        self.component_id = policy.component_id
        self.query_schema_version = policy.query_schema_version
        self.config_schema_version = policy.config_schema_version
        self._policy = policy
        self._queries = tuple(queries)
        self._encode_config = encode_config
        self._probe = probe
        self._case_ids = tuple(probe.case_ids)
        if not self._queries:
            raise ValueError(f"{self.component_id} requires validation queries")
        if self._probe.case_count <= 0:
            raise ValueError(f"{self.component_id} requires GPU probe cases")
        if len(self._case_ids) != self._probe.case_count:
            raise ValueError(
                f"{self.component_id} probe case IDs do not match its case count"
            )
        if any(not case_id for case_id in self._case_ids):
            raise ValueError(f"{self.component_id} probe case IDs must be non-empty")
        if len(self._case_ids) != len(set(self._case_ids)):
            raise ValueError(f"{self.component_id} probe case IDs must be unique")

    def estimate(self, context: GenerationContext) -> WorkEstimate:
        del context
        return WorkEstimate(
            component_id=self.component_id,
            work_units=self._probe.case_count + len(self._queries),
            case_count=self._probe.case_count,
            description=self._probe.description,
            dimensions={
                "gpu_measurement_cases": self._probe.case_count,
                "validated_runtime_queries": len(self._queries),
            },
        )

    def reviewed_queries(self) -> tuple[QueryT, ...]:
        return self._queries

    def _config(self, context: GenerationContext) -> tuple[ConfigT, FrozenMapping]:
        configs: dict[FrozenMapping, ConfigT] = {}
        for query in self._queries:
            config = self._policy.heuristic(query, context.device)
            self._policy.validate_config(query, config, context.device)
            encoded = FrozenMapping(self._encode_config(config))
            configs.setdefault(encoded, config)
        if len(configs) != 1:
            raise ValueError(
                f"{self.component_id} has multiple configurations and requires "
                "a candidate-race generator"
            )
        encoded, config = next(iter(configs.items()))
        return config, encoded

    def _measure(
        self,
        context: GenerationContext,
        checkpoints: CheckpointStore,
        encoded_config: FrozenMapping,
    ) -> tuple[GpuProbeMeasurement, ...]:
        checkpoint_id = "production-qualification"
        cached = checkpoints.load(self.component_id, checkpoint_id)
        if (
            cached is not None
            and cached.get("schema_version") in (1, 2)
            and context.checkpoint_metadata_matches(cached.get("generation"))
            and cached.get("case_count") == self._probe.case_count
        ):
            raw = cached.get("measurements")
            if not isinstance(raw, list):
                raise TypeError("GPU qualification checkpoint must contain an array")
            measurements = tuple(
                GpuProbeMeasurement.from_dict(item) for item in raw
            )
            measured_case_ids = tuple(item.label for item in measurements)
            raw_case_ids = cached.get("case_ids")
            raw_config = cached.get("config")
            case_ids_match = (
                measured_case_ids == self._case_ids
                and (
                    raw_case_ids is None
                    or raw_case_ids == list(self._case_ids)
                )
            )
            config_matches = (
                raw_config is None or raw_config == encoded_config.to_dict()
            )
            if case_ids_match and config_matches:
                raw_generation = cached.get("generation")
                if not isinstance(raw_generation, Mapping):
                    raise TypeError(
                        "GPU qualification generation must be an object"
                    )
                if (
                    cached.get("schema_version") != 2
                    or raw_case_ids != list(self._case_ids)
                    or raw_config != encoded_config.to_dict()
                ):
                    checkpoints.save(
                        self.component_id,
                        checkpoint_id,
                        self._checkpoint_payload(
                            generation=raw_generation,
                            encoded_config=encoded_config,
                            measurements=measurements,
                        ),
                    )
                return measurements

        measurements = self._probe(context)
        if len(measurements) != self._probe.case_count:
            raise ValueError(
                f"{self.component_id} GPU probe returned {len(measurements)} "
                f"measurements, expected {self._probe.case_count}"
            )
        measured_case_ids = tuple(item.label for item in measurements)
        if measured_case_ids != self._case_ids:
            raise ValueError(
                f"{self.component_id} GPU probe returned unexpected case IDs"
            )
        checkpoints.save(
            self.component_id,
            checkpoint_id,
            self._checkpoint_payload(
                generation=context.checkpoint_metadata(),
                encoded_config=encoded_config,
                measurements=measurements,
            ),
        )
        return measurements

    def _checkpoint_payload(
        self,
        *,
        generation: Mapping[str, object],
        encoded_config: FrozenMapping,
        measurements: Sequence[GpuProbeMeasurement],
    ) -> dict[str, object]:
        return {
            "schema_version": 2,
            "generation": dict(generation),
            "case_count": self._probe.case_count,
            "case_ids": list(self._case_ids),
            "config": encoded_config.to_dict(),
            "measurements": [item.to_dict() for item in measurements],
        }

    def qualify(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> QualificationOutcome:
        """Measure the production paths and validate the single config.

        Composite generators combine the qualified config with decisions from
        other sources; :meth:`generate` emits it as a single-leaf planner.
        """
        config, encoded_config = self._config(context)
        del config
        progress.start_stage(
            self.component_id,
            stage="production-path GPU qualification",
            total=self._probe.case_count,
        )
        measurements = self._measure(context, checkpoints, encoded_config)
        for measurement in measurements:
            if not measurement.correct:
                raise RuntimeError(
                    f"{self.component_id} failed GPU qualification "
                    f"{measurement.label}"
                )
            progress.advance(self.component_id, detail=measurement.label)

        progress.start_stage(
            self.component_id,
            stage="validate profiled runtime envelope",
            total=len(self._queries),
        )
        for index, query in enumerate(self._queries):
            decoded = self._policy.decode_profile(encoded_config)
            self._policy.validate_config(query, decoded, context.device)
            progress.advance(self.component_id, detail=f"query-{index + 1}")

        latencies = [measurement.latency_us for measurement in measurements]
        return QualificationOutcome(
            encoded_config=encoded_config,
            evidence={
                "selection": "single_candidate_gpu_qualification",
                "gpu_measurement_cases": len(measurements),
                "latency_us": {
                    "minimum": min(latencies),
                    "median": statistics.median(latencies),
                    "maximum": max(latencies),
                },
            },
            completed_work_units=self.estimate(context).work_units,
        )

    def generate(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> ComponentGenerationResult:
        outcome = self.qualify(context, progress=progress, checkpoints=checkpoints)
        return ComponentGenerationResult(
            component={
                "component_id": self.component_id,
                "query_schema_version": self.query_schema_version,
                "config_schema_version": self.config_schema_version,
                "planner": {
                    "kind": "leaf",
                    "name": "measured-production-implementation",
                    "config": outcome.encoded_config.to_dict(),
                },
            },
            evidence=outcome.evidence,
            completed_work_units=outcome.completed_work_units,
        )


__all__ = [
    "GpuProbeMeasurement",
    "GpuQualificationProbe",
    "MeasuredPolicyGenerator",
]
