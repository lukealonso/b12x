"""Shared search orchestration over component-owned production race sessions."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, replace

from b12x.policy.problem import FieldRole
from b12x.policy.types import FrozenMapping

from .search import QueryMeasurement, SearchBudget, SearchPoint, SearchStrategy, run_search
from .store import CheckpointStore


def _search_points(problem, queries):
    points = []
    derived = {field.name for field in problem.inputs if field.role is FieldRole.DERIVED}
    fields = {field.name: field for field in (*problem.inputs, *problem.sampled_inputs)}
    for query in queries:
        if set(query) - fields.keys():
            raise ValueError("provider query contains undeclared tuning inputs")
        axes = tuple(field for field in problem.axes if field.name in query)
        missing = {field.name for field in (*problem.inputs, *problem.sampled_inputs)
                   if field.role is not FieldRole.ENVIRONMENT and field.name not in query}
        if missing:
            raise ValueError(f"provider must supply its complete tuning inputs: {sorted(missing)}")
        for field in axes:
            field.validate(query[field.name])
        if problem.derive_inputs is not None:
            for name, value in problem.derive_inputs(query).items():
                if type(query[name]) is not type(value) or query[name] != value:
                    raise ValueError(f"provider has inconsistent derived input {name!r}")
        names = {field.name for field in axes}
        points.append(SearchPoint(query=query,
            family=FrozenMapping({name: value for name, value in query.items() if name not in names | derived}),
            coordinates=tuple(query[field.name] for field in axes)))
    return tuple(points)


def _decision_scores(scores, provider, problem, query, configs):
    from .selection import ScenarioScores
    from .sweep import SweepCandidate

    candidates, identities = [], {}
    project = getattr(provider, "decision_for_candidate", lambda query, candidate: candidate.config)
    for candidate in scores.candidates:
        decisions = (project(query, candidate), *getattr(candidate, "equivalent_decisions", ()))
        keys = []
        for decision in decisions:
            problem.validate_decision(decision)
            normalized = SweepCandidate.create(decision)
            keys.append(normalized.candidate_id)
            candidates.append(normalized)
            configs[normalized.candidate_id] = normalized.config
        identities[candidate.candidate_id] = tuple(keys)
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("distinct production candidates lower to the same independent decision")
    return ScenarioScores(candidates=tuple(candidates),
                          latencies_us=FrozenMapping({alias: value for key, value in scores.latencies_us.items()
                                                       for alias in identities[key]}),
                          eligible_candidates=tuple(alias for key in scores.eligible_candidates
                                                    for alias in identities[key])), identities


class DiscreteSearch:
    """Adapt measured query scenarios without changing component timing math.

    Allocation groups retain their complete corpus when preparing a session.
    Search chooses queries; every requested query races all production
    candidates across every declared scenario. Component-specific selection
    contracts must provide their own reducer before using this adapter.
    """

    def __init__(self, generator, problem, context, checkpoints: CheckpointStore):
        if problem.component_id != generator.component_id:
            raise ValueError("search adapter and tuning problem differ")
        context = generator.measurement_context(context)
        self.generator, self.problem, self.context, self.checkpoints = generator, problem, context, checkpoints
        self._by_query = defaultdict(list)
        self._by_group = defaultdict(list)
        self.configs = {}
        for case in generator._cases:
            self._by_query[generator.tuning_inputs(case.query)].append(case)
            self._by_group[case.group_id].append(case)
        self.points = _search_points(problem, self._by_query)
        self._stack = ExitStack()
        self._group = None
        self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stack.__exit__(*exc)
        self._group = self._session = None

    def measure(self, point: SearchPoint) -> QueryMeasurement:
        cases = self._by_query[point.query]
        results = []
        fresh_before = self.generator._fresh_cases
        started = time.monotonic()
        preparation = 0.
        for case in cases:
            cached = self.generator._load_checkpoint(case=case, context=self.context, checkpoints=self.checkpoints)
            if self.generator._checkpoint_is_current(cached):
                results.append(cached.measurements)
                continue
            if self._group != case.group_id:
                self._stack.close()
                preparing = time.monotonic()
                self._session = self._stack.enter_context(self.generator._benchmark_factory(
                    case.group_id, tuple(self._by_group[case.group_id]), self.context))
                preparation += time.monotonic() - preparing
                self._group = case.group_id
            results.append(self.generator._measure_case(case=case, session=self._session,
                           context=self.context, checkpoints=self.checkpoints, cached=cached))
        scores = self.generator.reduce_measurements(results)
        ordered = sorted(scores.candidates, key=lambda candidate: (self.generator._candidate_tie_breaker(candidate), candidate.candidate_id)
                         if self.generator._candidate_tie_breaker is not None else candidate.candidate_id)
        scores, identities = _decision_scores(scores, self.generator, self.problem, point.query, self.configs)
        return QueryMeasurement(point=point, latencies_us=scores.latencies_us,
                                candidate_ids=scores.candidate_ids,
                                eligible_candidates=scores.eligible_candidates,
                                tie_break_order=tuple(alias for candidate in ordered for alias in identities[candidate.candidate_id]),
                                fresh=self.generator._fresh_cases != fresh_before,
                                cohort=self.context.measurement_cohort,
                                costs_seconds=FrozenMapping({"allocation_preparation": preparation,
                                                             "race_and_storage": time.monotonic() - started - preparation}))

    def search(self, *, strategy: SearchStrategy, budget: SearchBudget):
        from .progression import audit_progressions

        result = run_search(self.points, strategy=strategy, budget=budget, measure=self.measure)
        started = time.monotonic()
        progressions = audit_progressions(
            result.measurements, self.configs, axes=tuple(field.name for field in self.problem.axes),
            ordered_knobs=tuple(knob.name for knob in self.problem.decisions if knob.ordered),
        )
        elapsed = time.monotonic() - started
        return replace(result, progressions=progressions, wall_seconds=result.wall_seconds + elapsed,
                       selection_seconds=result.selection_seconds + elapsed)


class MoeSearch(DiscreteSearch):
    """Search full production MoE races using the component's precision reducer.

    Every sampled query retains all route scenarios and legal candidates.
    Screening and coarse-stage pruning cannot substitute for the exhaustive
    candidate evidence required at independent qualification points.
    """

    def __init__(self, generator, problem, context, checkpoints: CheckpointStore):
        if problem.component_id != generator.component_id:
            raise ValueError("search adapter and tuning problem differ")
        self.generator, self.problem, self.context, self.checkpoints = generator, problem, context, checkpoints
        self._by_query = defaultdict(list)
        for case in generator._cases:
            self._by_query[FrozenMapping(case.query())].append(case)
        self.points = _search_points(problem, self._by_query)
        self.configs = {}
        self._stack = ExitStack()
        self._group = self._session = None

    def measure(self, point: SearchPoint) -> QueryMeasurement:
        cases = self._by_query[point.query]
        geometry = cases[0].geometry
        started = time.monotonic()
        preparation = 0.
        fresh_before = self.generator._fresh_cases
        if self._group != geometry.key:
            self._stack.close()
            preparing = time.monotonic()
            self._session = self._stack.enter_context(self.generator._benchmark_factory(geometry, self.context))
            preparation = time.monotonic() - preparing
            self._group = geometry.key
        grouped = []
        for case in cases:
            candidates = self._session.eligible_candidates(case, self._session.candidates)
            if not candidates:
                raise RuntimeError(f"no legal production candidates for {case.case_id}")
            measurements = self.generator._race(
                stage="search", case=case, candidates=candidates, session=self._session,
                context=self.context, checkpoints=self.checkpoints,
            )
            grouped.append((case, measurements))
        scores = self.generator.reduce_measurements(grouped, self.context)
        ordered = sorted(scores.candidates, key=lambda candidate: (candidate.config["backend"] != "w4a16", candidate.candidate_id))
        scores, identities = _decision_scores(scores, self.generator, self.problem, point.query, self.configs)
        return QueryMeasurement(point=point, latencies_us=scores.latencies_us,
                                candidate_ids=scores.candidate_ids,
                                eligible_candidates=scores.eligible_candidates,
                                tie_break_order=tuple(alias for candidate in ordered for alias in identities[candidate.candidate_id]),
                                fresh=self.generator._fresh_cases != fresh_before,
                                cohort=self.context.measurement_cohort,
                                costs_seconds=FrozenMapping({"allocation_preparation": preparation,
                                                             "race_and_storage": time.monotonic() - started - preparation}))


@dataclass(frozen=True, kw_only=True)
class MeasurementTask:
    """A production race or fixed-implementation probe in a component program."""

    name: str
    provider: object
    problem: object

    @property
    def kind(self):
        from .measured import MeasuredPolicyGenerator
        from .providers.moe import MoeDecodeGenerator
        from .sweep import DiscreteSweepGenerator

        if isinstance(self.provider, MeasuredPolicyGenerator):
            return "fixed_backend_probe"
        if isinstance(self.provider, (DiscreteSweepGenerator, MoeDecodeGenerator)):
            if getattr(self.provider, "measurement_kind", None) == "fixed_backend_probe":
                return "fixed_backend_probe"
            return "candidate_race"
        raise TypeError("measurement tasks require a production race or fixed-backend probe adapter")

    def open_search(self, context, checkpoints):
        from .providers.moe import MoeDecodeGenerator

        if self.kind != "candidate_race":
            raise ValueError("fixed-backend probes have no kernel-choice search; qualify their production path")
        adapter = MoeSearch if isinstance(self.provider, MoeDecodeGenerator) else DiscreteSearch
        return adapter(self.provider, self.problem, context, checkpoints)

    def describe(self):
        from .measured import MeasuredPolicyGenerator

        provider = self.provider
        if isinstance(provider, MeasuredPolicyGenerator):
            queries = tuple(self.problem.canonical_inputs(query) for query in provider.reviewed_queries())
            cases = len(provider._case_ids)
        else:
            cases = len(provider._cases)
            queries = {FrozenMapping(case.query()) if callable(case.query)
                       else provider.tuning_inputs(case.query) for case in provider._cases}
            _search_points(self.problem, queries)
        return {"name": self.name, "kind": self.kind, "measurement_cases": cases,
                "query_points": len(queries),
                "arbitrary_query_preparation": callable(getattr(provider, "cases_for_tuning_queries", None)),
                "unmeasured_region_legality": callable(getattr(provider, "validate_region_decision", None)),
                "sampled_input_confirmation_required": bool(self.problem.sampled_inputs),
                "provider": f"{type(provider).__module__}.{type(provider).__name__}"}


def measurement_program(generator, problem) -> tuple[MeasurementTask, ...]:
    children = getattr(generator, "measurement_children", None)
    providers = children() if children is not None else (("production", generator),)
    if not providers or len({name for name, _ in providers}) != len(providers):
        raise ValueError("component measurement programs require unique nonempty tasks")
    tasks = tuple(MeasurementTask(name=name, provider=provider, problem=problem) for name, provider in providers)
    for task in tasks:
        if not task.name or task.provider.component_id != problem.component_id:
            raise ValueError("measurement task ownership differs from its component")
        _ = task.kind
    return tasks


__all__ = ["DiscreteSearch", "MoeSearch", "MeasurementTask", "measurement_program"]
