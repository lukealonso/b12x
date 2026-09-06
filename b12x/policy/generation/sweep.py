"""Reusable discrete-sweep generator for component-owned GPU races."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import ContextManager, Protocol, cast

from b12x.policy.types import FrozenMapping

from .contracts import (
    ComponentGenerationResult,
    GenerationContext,
    MeasurementPartition,
    ProgressReporter,
    WorkEstimate,
)
from .reducer import DecisionRecord, build_axis_tree, decision_node_to_dict
from .store import CheckpointStore
from .observations import ObservationStore, measure_observation
from .selection import reduce_scenarios


def _stable_id(value: Mapping[str, object], *, length: int = 16) -> str:
    payload = value.to_dict() if isinstance(value, FrozenMapping) else dict(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True, kw_only=True)
class SweepCase:
    """One measured scenario for one runtime query point."""

    case_id: str
    group_id: str
    query: FrozenMapping
    scenario: str = "default"
    metadata: FrozenMapping = FrozenMapping()

    @classmethod
    def create(
        cls,
        *,
        group_id: str,
        query: Mapping[str, object],
        scenario: str = "default",
        metadata: Mapping[str, object] | None = None,
        label: str | None = None,
    ) -> "SweepCase":
        identity = {
            "group_id": group_id,
            "query": dict(query),
            "scenario": scenario,
            "metadata": dict(metadata or {}),
        }
        prefix = label or group_id
        case_id = f"{prefix}-{_stable_id(identity, length=12)}"
        return cls(
            case_id=case_id,
            group_id=group_id,
            query=FrozenMapping(query),
            scenario=scenario,
            metadata=FrozenMapping(metadata),
        )

    def __post_init__(self) -> None:
        if not self.case_id or not self.group_id or not self.scenario:
            raise ValueError("sweep case identifiers must be non-empty")
        if not self.query:
            raise ValueError("sweep cases require a non-empty runtime query")


@dataclass(frozen=True, kw_only=True)
class SweepCandidate:
    """One component config eligible for a measured case."""

    config: FrozenMapping
    candidate_id: str
    decision: FrozenMapping | None = None
    equivalent_decisions: tuple[FrozenMapping, ...] = ()

    @classmethod
    def create(cls, config: Mapping[str, object], *, decision: Mapping[str, object] | None = None) -> "SweepCandidate":
        frozen = FrozenMapping(config)
        return cls(config=frozen, candidate_id=_stable_id(frozen),
                   decision=None if decision is None else FrozenMapping(decision))

    def __post_init__(self) -> None:
        if not self.config:
            raise ValueError("sweep candidates require a non-empty config")
        if self.candidate_id != _stable_id(self.config):
            raise ValueError("sweep candidate ID does not match its config")
        if self.equivalent_decisions:
            if self.decision is None:
                raise ValueError("execution-equivalent aliases require an independent decision")
            decisions = (self.decision, *self.equivalent_decisions)
            if len({ _stable_id(item) for item in decisions }) != len(decisions):
                raise ValueError("execution-equivalent decisions must be unique")


@dataclass(frozen=True, kw_only=True)
class SweepMeasurement:
    """Correctness and timing result for one candidate."""

    candidate: SweepCandidate
    latency_us: float | None
    correct: bool
    metrics: FrozenMapping = FrozenMapping()
    error: str | None = None
    selection_eligible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, FrozenMapping):
            if not isinstance(self.metrics, Mapping):
                raise TypeError("metrics must be an object")
            object.__setattr__(self, "metrics", FrozenMapping(self.metrics))
        if self.latency_us is not None and (
            not math.isfinite(self.latency_us) or self.latency_us <= 0
        ):
            raise ValueError("latency_us must be finite and positive")
        if not isinstance(self.correct, bool):
            raise TypeError("correct must be a boolean")
        if type(self.selection_eligible) is not bool:
            raise TypeError("selection_eligible must be a boolean")
        if self.error is not None and not self.error:
            raise ValueError("measurement errors must be non-empty")

    def qualified(self) -> bool:
        return self.error is None and self.latency_us is not None and self.correct

    def passes(self) -> bool:
        return self.qualified() and self.selection_eligible

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "config": self.candidate.config.to_dict(),
            "latency_us": self.latency_us,
            "correct": self.correct,
            "metrics": self.metrics.to_dict(),
            "error": self.error,
            "selection_eligible": self.selection_eligible,
            "kernel_decision": None if self.candidate.decision is None else self.candidate.decision.to_dict(),
            "equivalent_decisions": [item.to_dict() for item in self.candidate.equivalent_decisions],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SweepMeasurement":
        raw_config = value.get("config")
        if not isinstance(raw_config, Mapping):
            raise TypeError("measurement config must be an object")
        candidate = SweepCandidate.create(raw_config, decision=value.get("kernel_decision"))
        from dataclasses import replace

        candidate = replace(candidate, equivalent_decisions=tuple(
            FrozenMapping(item) for item in value.get("equivalent_decisions", ())
        ))
        if value.get("candidate_id") != candidate.candidate_id:
            raise ValueError("checkpoint candidate ID does not match its config")
        latency = value.get("latency_us")
        correct = value.get("correct")
        if not isinstance(correct, bool):
            raise TypeError("measurement correct field must be a boolean")
        raw_metrics = value.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("measurement metrics must be an object")
        error = value.get("error")
        return cls(
            candidate=candidate,
            latency_us=None if latency is None else float(latency),
            correct=correct,
            metrics=FrozenMapping(raw_metrics),
            error=None if error is None else str(error),
            selection_eligible=value.get("selection_eligible", True),
        )


@dataclass(frozen=True, kw_only=True)
class _CachedSweepMeasurements:
    generation: Mapping[str, object]
    candidate_ids: tuple[str, ...]
    measurements: tuple[SweepMeasurement, ...]
    checkpoint_schema_version: int
    candidate_contract_version: int | None
    observation_key: str | None = None


@dataclass(frozen=True, kw_only=True)
class SweepRaceOutcome:
    """Reduced result of a sweep: one winning config per runtime query point."""

    records: tuple[DecisionRecord, ...]
    coverage: dict[str, object]
    evidence: dict[str, object]
    completed_work_units: int


class SweepSession(Protocol):
    """Stable-allocation measurement session for one case group."""

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]: ...

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]: ...


class SweepBenchmarkFactory(Protocol):
    """Create and fully release one geometry-scoped measurement session."""

    def __call__(
        self,
        group_id: str,
        cases: tuple[SweepCase, ...],
        context: GenerationContext,
    ) -> ContextManager[SweepSession]: ...


def _query_key(case: SweepCase, fields: tuple[str, ...]) -> tuple[object, ...]:
    return tuple(case.query[field] for field in fields)


class DiscreteSweepGenerator:
    """Generate one planner from correctness-gated component measurements.

    Providers must bump ``candidate_contract_version`` when candidate
    enumeration or eligibility changes. Case IDs independently version the
    measured corpus. A provider may explicitly reuse a subset from earlier
    contracts when the retained candidate IDs and measurement inputs are
    unchanged; migration enumerates and verifies every retained candidate.
    """

    def __init__(
        self,
        *,
        component_id: str,
        query_schema_version: int,
        config_schema_version: int,
        query_fields: tuple[str, ...],
        range_fields: frozenset[str],
        cases: Sequence[SweepCase],
        benchmark_factory: SweepBenchmarkFactory,
        coverage: Mapping[str, object],
        candidate_contract_version: int = 1,
        subset_reuse_contract_versions: Sequence[int] = (),
        nearest_range_bounds: Mapping[str, tuple[int, int]] | None = None,
        candidate_tie_breaker: Callable[[SweepCandidate], int | str] | None = None,
    ) -> None:
        self.component_id = component_id
        self.query_schema_version = int(query_schema_version)
        self.config_schema_version = int(config_schema_version)
        self._query_fields = tuple(query_fields)
        self._range_fields = frozenset(range_fields)
        self._cases = tuple(cases)
        self._benchmark_factory = benchmark_factory
        self._coverage = FrozenMapping(coverage)
        self._candidate_contract_version = candidate_contract_version
        self._subset_reuse_contract_versions = tuple(subset_reuse_contract_versions)
        self._nearest_range_bounds = dict(nearest_range_bounds or {})
        self._candidate_tie_breaker = candidate_tie_breaker
        self._fresh_cases = 0
        self._measurement_seconds = 0.0
        self._storage_seconds = 0.0
        if not self._cases:
            raise ValueError(f"{component_id} requires at least one sweep case")
        if not self._query_fields or len(self._query_fields) != len(
            set(self._query_fields)
        ):
            raise ValueError("query_fields must be non-empty and unique")
        if not self._range_fields <= frozenset(self._query_fields):
            raise ValueError("range_fields must be present in query_fields")
        expected = frozenset(self._query_fields)
        if any(frozenset(case.query) != expected for case in self._cases):
            raise ValueError("sweep case query fields differ from the component schema")
        if len({case.case_id for case in self._cases}) != len(self._cases):
            raise ValueError("sweep case IDs must be unique")
        if type(candidate_contract_version) is not int or candidate_contract_version <= 0:
            raise ValueError("candidate_contract_version must be a positive integer")
        if any(
            type(version) is not int or not 0 < version < self._candidate_contract_version
            for version in self._subset_reuse_contract_versions
        ):
            raise ValueError("subset reuse requires positive, earlier candidate contracts")
        if not frozenset(self._nearest_range_bounds) <= self._range_fields:
            raise ValueError("nearest range fields must also be range_fields")

    def measurement_context(self, context: GenerationContext) -> GenerationContext:
        """Bind component-specific execution dependencies before checkpoint lookup."""
        return context

    def estimate(self, context: GenerationContext) -> WorkEstimate:
        del context
        query_count = len(
            {_query_key(case, self._query_fields) for case in self._cases}
        )
        group_count = len({case.group_id for case in self._cases})
        return WorkEstimate(
            component_id=self.component_id,
            work_units=len(self._cases) + query_count,
            case_count=len(self._cases),
            description=(
                f"{group_count} allocation groups; correctness-gated GPU race "
                "and decision-tree reduction"
            ),
            dimensions={
                "allocation_groups": group_count,
                "measurement_cases": len(self._cases),
                "runtime_queries": query_count,
            },
        )

    def measurement_partitions(
        self,
        context: GenerationContext,
    ) -> tuple[MeasurementPartition, ...]:
        del context
        cases_by_group: dict[str, list[SweepCase]] = defaultdict(list)
        for case in self._cases:
            cases_by_group[case.group_id].append(case)
        partitions = []
        for group_id in sorted(cases_by_group):
            cases = tuple(cases_by_group[group_id])
            query_count = len({_query_key(case, self._query_fields) for case in cases})
            partitions.append(
                MeasurementPartition(
                    component_id=self.component_id,
                    partition_id=group_id,
                    work_units=len(cases) + query_count,
                    case_count=len(cases),
                    description=f"allocation group {group_id}",
                )
            )
        return tuple(partitions)

    def select_measurement_partitions(
        self,
        partition_ids: tuple[str, ...],
    ) -> "DiscreteSweepGenerator":
        selected = frozenset(partition_ids)
        available = frozenset(case.group_id for case in self._cases)
        unknown = selected - available
        if not selected or unknown:
            raise ValueError(
                f"invalid {self.component_id} measurement partitions: "
                f"{sorted(unknown) if unknown else 'empty selection'}"
            )
        restricted = copy(self)
        restricted._cases = tuple(
            case for case in self._cases if case.group_id in selected
        )
        return restricted

    def _measure_case(
        self,
        *,
        case: SweepCase,
        session: SweepSession,
        context: GenerationContext,
        checkpoints: CheckpointStore,
        cached: _CachedSweepMeasurements | None = None,
    ) -> tuple[SweepMeasurement, ...]:
        if cached is None:
            cached = self._load_checkpoint(
                case=case,
                context=context,
                checkpoints=checkpoints,
            )
        if cached is not None and self._checkpoint_is_current(cached):
            return cached.measurements

        candidates = session.candidates(case)
        if not candidates:
            raise RuntimeError(f"no candidates were produced for {case.case_id}")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"candidate IDs are not unique for {case.case_id}")
        can_reuse = cached is not None and (
            cached.candidate_ids == tuple(candidate_ids)
            or (
                cached.candidate_contract_version in self._subset_reuse_contract_versions
                and set(candidate_ids) <= set(cached.candidate_ids)
            )
        )
        if can_reuse and cached.observation_key is None:
            by_id = {item.candidate.candidate_id: item for item in cached.measurements}
            reused = tuple(by_id[candidate_id] for candidate_id in candidate_ids)
            checkpoints.save(
                self.component_id,
                case.case_id,
                self._checkpoint_payload(
                    case=case,
                    generation=cached.generation,
                    candidate_ids=candidate_ids,
                    measurements=reused,
                ),
            )
            return reused

        def measure():
            measurements = session.measure(case, candidates)
            if [item.candidate.candidate_id for item in measurements] != candidate_ids:
                raise ValueError("measurement sessions must preserve the requested candidate order")
            return {"measurements": [item.to_dict() for item in measurements]}

        observed = measure_observation(
            context=context, component_id=self.component_id,
            inputs=FrozenMapping({"group_id": case.group_id, "query": case.query.to_dict(),
                                  "scenario": case.scenario, "metadata": case.metadata.to_dict()}),
            candidates=tuple(candidate.config for candidate in candidates),
            oracle_contract=f"{self.component_id}:{self._candidate_contract_version}",
            store=ObservationStore(checkpoints.observations_path),
            measure=measure,
        )
        measurements = tuple(SweepMeasurement.from_dict(item) for item in observed.result["measurements"])
        self._measurement_seconds += observed.measurement_seconds
        self._storage_seconds += observed.storage_seconds
        self._fresh_cases += observed.fresh
        measured_ids = [item.candidate.candidate_id for item in measurements]
        if measured_ids != candidate_ids:
            raise ValueError(
                "measurement sessions must preserve the requested candidate order"
            )
        payload = self._checkpoint_payload(
            case=case, generation=context.checkpoint_metadata(),
            candidate_ids=candidate_ids, measurements=measurements,
        )
        started = time.monotonic()
        if observed.identity is not None:
            payload.pop("measurements")
            payload["schema_version"] = 3
            payload["observation_key"] = observed.identity.key
        checkpoints.save(
            self.component_id,
            case.case_id,
            payload,
        )
        self._storage_seconds += time.monotonic() - started
        return measurements

    def _load_checkpoint(
        self,
        *,
        case: SweepCase,
        context: GenerationContext,
        checkpoints: CheckpointStore,
    ) -> _CachedSweepMeasurements | None:
        cached = checkpoints.load(self.component_id, case.case_id)
        schema_version = None if cached is None else cached.get("schema_version")
        contract_version = (
            None if cached is None else cached.get("candidate_contract_version")
        )
        if (
            cached is None
            or type(schema_version) is not int
            or schema_version not in (1, 2, 3)
            or not context.checkpoint_metadata_matches(cached.get("generation"))
            or cached.get("case_id") != case.case_id
            or (
                schema_version in (2, 3)
                and (
                    type(contract_version) is not int
                    or contract_version not in (
                        self._candidate_contract_version,
                        *self._subset_reuse_contract_versions,
                    )
                )
            )
        ):
            return None
        raw_candidate_ids = cached.get("candidate_ids")
        raw_measurements = cached.get("measurements")
        raw_generation = cached.get("generation")
        observation_key = cached.get("observation_key")
        if schema_version == 3:
            if not isinstance(observation_key, str):
                raise ValueError("sweep checkpoint requires an observation reference")
            stored = ObservationStore(checkpoints.observations_path).load_key(observation_key)
            if stored is None:
                return None
            identity = stored["identity"]
            inputs = {"group_id": case.group_id, "query": case.query.to_dict(),
                      "scenario": case.scenario, "metadata": case.metadata.to_dict()}
            if (identity.get("generation") != raw_generation or identity.get("inputs") != inputs
                    or identity.get("component_id") != self.component_id
                    or identity.get("cohort") != context.measurement_cohort
                    or identity.get("oracle_contract") != f"{self.component_id}:{contract_version}"):
                raise ValueError("sweep checkpoint references a different measurement identity")
            raw_measurements = stored["result"].get("measurements")
        if not isinstance(raw_candidate_ids, list) or not all(
            isinstance(candidate_id, str) for candidate_id in raw_candidate_ids
        ):
            raise TypeError("sweep checkpoint candidate IDs must be an array")
        if not isinstance(raw_measurements, list):
            raise TypeError("sweep checkpoint measurements must be an array")
        if not isinstance(raw_generation, Mapping):
            raise TypeError("sweep checkpoint generation must be an object")
        measurements = tuple(
            SweepMeasurement.from_dict(item) for item in raw_measurements
        )
        candidate_ids = tuple(raw_candidate_ids)
        measured_ids = tuple(item.candidate.candidate_id for item in measurements)
        if schema_version == 3 and identity.get("candidates") != [item.candidate.config.to_dict() for item in measurements]:
            raise ValueError("sweep observations differ from their production candidate cohort")
        if measured_ids != candidate_ids:
            raise ValueError("sweep checkpoint measurements do not match candidate IDs")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("sweep checkpoint candidate IDs are not unique")
        return _CachedSweepMeasurements(
            generation=raw_generation,
            candidate_ids=candidate_ids,
            measurements=measurements,
            checkpoint_schema_version=int(schema_version),
            candidate_contract_version=(
                cast(int, contract_version) if schema_version in (2, 3) else None
            ),
            observation_key=observation_key,
        )

    def _checkpoint_is_current(
        self,
        cached: _CachedSweepMeasurements | None,
    ) -> bool:
        return (
            cached is not None and cached.checkpoint_schema_version in (2, 3)
            and cached.candidate_contract_version == self._candidate_contract_version
        )

    def _checkpoint_payload(
        self,
        *,
        case: SweepCase,
        generation: Mapping[str, object],
        candidate_ids: Sequence[str],
        measurements: Sequence[SweepMeasurement],
    ) -> dict[str, object]:
        return {
            "schema_version": 2,
            "candidate_contract_version": self._candidate_contract_version,
            "generation": dict(generation),
            "case_id": case.case_id,
            "group_id": case.group_id,
            "query": case.query.to_dict(),
            "scenario": case.scenario,
            "metadata": case.metadata.to_dict(),
            "candidate_ids": list(candidate_ids),
            "measurements": [item.to_dict() for item in measurements],
        }

    def race(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> SweepRaceOutcome:
        """Measure every case and reduce the races to one winner per query point.

        Composite generators combine the returned records with decisions from
        other sources before building a planner; :meth:`generate` builds the
        planner from these records alone.
        """
        context = self.measurement_context(context)
        self._fresh_cases = 0
        self._measurement_seconds = 0.0
        self._storage_seconds = 0.0
        cases_by_group: dict[str, list[SweepCase]] = defaultdict(list)
        for case in self._cases:
            cases_by_group[case.group_id].append(case)
        measured: list[tuple[SweepCase, tuple[SweepMeasurement, ...]]] = []
        qualification_cases = 0
        race_cases = 0
        progress.start_stage(
            self.component_id,
            stage="correctness and candidate races",
            total=len(self._cases),
        )
        for group_id in sorted(cases_by_group):
            group_cases = tuple(cases_by_group[group_id])
            progress.advance(
                self.component_id,
                units=0,
                detail=f"prepare {group_id}",
            )
            cached_by_case = {
                case.case_id: self._load_checkpoint(
                    case=case,
                    context=context,
                    checkpoints=checkpoints,
                )
                for case in group_cases
            }
            if all(
                self._checkpoint_is_current(
                    cached_by_case[case.case_id],
                )
                for case in group_cases
            ):
                group_measurements = []
                for case in group_cases:
                    cached = cast(
                        _CachedSweepMeasurements,
                        cached_by_case[case.case_id],
                    )
                    progress.advance(
                        self.component_id,
                        units=0,
                        detail=f"race {case.case_id}",
                    )
                    group_measurements.append((case, cached.measurements))
            else:
                group_measurements = []
                with self._benchmark_factory(
                    group_id,
                    group_cases,
                    context,
                ) as session:
                    for case in group_cases:
                        progress.advance(
                            self.component_id,
                            units=0,
                            detail=f"race {case.case_id}",
                        )
                        group_measurements.append(
                            (
                                case,
                                self._measure_case(
                                    case=case,
                                    session=session,
                                    context=context,
                                    checkpoints=checkpoints,
                                    cached=cached_by_case[case.case_id],
                                ),
                            )
                        )
            for case, measurements in group_measurements:
                if not any(item.passes() for item in measurements):
                    raise RuntimeError(
                        f"all candidates failed correctness for {case.case_id}"
                    )
                measured.append((case, measurements))
                if len(measurements) == 1:
                    qualification_cases += 1
                else:
                    race_cases += 1
                progress.advance(
                    self.component_id,
                    detail=f"race {case.case_id}",
                )

        grouped_results: dict[
            tuple[object, ...],
            list[tuple[SweepCase, tuple[SweepMeasurement, ...]]],
        ] = defaultdict(list)
        for case, measurements in measured:
            grouped_results[_query_key(case, self._query_fields)].append(
                (case, measurements)
            )
        progress.start_stage(
            self.component_id,
            stage="scenario-robust reduction",
            total=len(grouped_results),
        )
        records: list[DecisionRecord] = []
        winner_counts: dict[str, int] = defaultdict(int)
        for grouped in grouped_results.values():
            scores = self.reduce_measurements(tuple(measurements for _, measurements in grouped))
            if not scores.eligible_candidates:
                raise RuntimeError(
                    "no candidate passed every scenario for query "
                    f"{grouped[0][0].query.to_dict()}"
                )
            winner = scores.select(self._candidate_tie_breaker)
            records.append(
                DecisionRecord(
                    query=grouped[0][0].query,
                    config=self.profile_config(grouped[0][0].query, winner),
                )
            )
            winner_counts[winner.candidate_id] += 1
            progress.advance(
                self.component_id,
                detail=f"reduce {grouped[0][0].case_id}",
            )

        coverage = self._coverage.to_dict()
        coverage.update(
            {
                "allocation_groups": len(cases_by_group),
                "measurement_cases": len(self._cases),
                "runtime_query_points": len(records),
            }
        )
        return SweepRaceOutcome(
            records=tuple(records),
            coverage=coverage,
            evidence={
                "winner_query_counts": dict(sorted(winner_counts.items())),
                "gpu_measurement_cases": len(measured),
                "fresh_measurement_cases": self._fresh_cases,
                "reused_measurement_cases": len(measured) - self._fresh_cases,
                "measurement_seconds": self._measurement_seconds,
                "storage_seconds": self._storage_seconds,
                "profile_cases": len(measured),
                "candidate_race_cases": race_cases,
                "single_candidate_qualification_cases": qualification_cases,
                **({"external_dependencies": context.provenance.to_dict()["external_dependencies"]}
                   if "external_dependencies" in context.provenance else {}),
            },
            completed_work_units=self.estimate(context).work_units,
        )

    def reduce_measurements(self, scenarios):
        return reduce_scenarios(scenarios, qualified=lambda item: item.qualified(),
                                selectable=lambda item: item.selection_eligible)

    def tuning_inputs(self, query):
        return query

    def decision_for_candidate(self, query, candidate):
        return candidate.config if candidate.decision is None else candidate.decision

    def profile_config(self, query, candidate):
        return candidate.config

    def build_planner(self, records: Sequence[DecisionRecord], *, device=None):
        """Reduce decision records to this component's axis-tree planner."""
        return build_axis_tree(
            tuple(records),
            field_order=self._query_fields,
            range_fields=self._range_fields,
            nearest_range_bounds=self._nearest_range_bounds,
        )

    def generate(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> ComponentGenerationResult:
        outcome = self.race(context, progress=progress, checkpoints=checkpoints)
        return ComponentGenerationResult(
            component={
                "component_id": self.component_id,
                "query_schema_version": self.query_schema_version,
                "config_schema_version": self.config_schema_version,
                "coverage": outcome.coverage,
                "planner": decision_node_to_dict(self.build_planner(outcome.records, device=context.device), compact=True),
            },
            evidence=outcome.evidence,
            completed_work_units=outcome.completed_work_units,
        )


__all__ = [
    "DiscreteSweepGenerator",
    "SweepBenchmarkFactory",
    "SweepRaceOutcome",
    "SweepCandidate",
    "SweepCase",
    "SweepMeasurement",
    "SweepSession",
]
