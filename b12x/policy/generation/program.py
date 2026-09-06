"""Qualification-gated generation from explicit shape domains and GPU races."""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

from b12x.policy.problem import AxisInterval, BindingTime, FieldRole, SearchDomain, stable_identity
from b12x.policy.types import ExactDecisionNode, FrozenMapping, ProfileLeaf

from .contracts import ComponentGenerationResult, WorkEstimate
from .engine import _search_points
from .qualification import QualificationCase, qualify_policy
from .reducer import DecisionRecord, decision_node_to_dict
from .regions import fit_regret_regions
from .search import SearchBudget, SearchStrategy
from .store import CheckpointStore


@dataclass(frozen=True, kw_only=True)
class ProfileSearchPlan:
    """Freeze training candidates, independent holdouts, and claimed coverage.

    Domains are integer lattices. Every covered query undergoes CPU lowering
    validation; GPU search samples only the declared training pool. Holdout
    measurements cannot modify the fitted policy within this plan.
    """

    domains: tuple[SearchDomain, ...]
    training: tuple[FrozenMapping, ...]
    holdouts: tuple[FrozenMapping, ...]
    strategy: SearchStrategy
    budget: SearchBudget
    legality_query_budget: int

    @classmethod
    def from_dict(cls, value):
        required = {"domains", "training", "holdouts", "strategy", "query_budget", "legality_query_budget"}
        if not isinstance(value, Mapping) or not required <= set(value) or set(value) - required - {"seconds"}:
            raise ValueError("search plan fields differ from the generation contract")
        return cls(domains=tuple(SearchDomain(fixed=FrozenMapping(item["fixed"]), axes=tuple(
                       AxisInterval(**axis) for axis in item["axes"])) for item in value["domains"]),
                   training=tuple(FrozenMapping(query) for query in value["training"]),
                   holdouts=tuple(FrozenMapping(query) for query in value["holdouts"]),
                   strategy=SearchStrategy(value["strategy"]),
                   budget=SearchBudget(queries=value["query_budget"], seconds=value.get("seconds")),
                   legality_query_budget=value["legality_query_budget"])

    def __post_init__(self):
        if not self.domains or not self.training or not self.holdouts:
            raise ValueError("search requires explicit domains, training points, and independent holdouts")
        if type(self.legality_query_budget) is not int or self.legality_query_budget <= 0:
            raise ValueError("legality query budget must be a positive integer")
        if sum(domain.size for domain in self.domains) > self.legality_query_budget:
            raise ValueError("declared coverage exceeds the CPU legality validation budget")
        identities = [stable_identity(query) for query in (*self.training, *self.holdouts)]
        if len(identities) != len(set(identities)):
            raise ValueError("training and holdout points must be unique and disjoint")

    def to_dict(self):
        return {"domains": [domain.to_dict() for domain in self.domains],
                "training": [query.to_dict() for query in self.training],
                "holdouts": [query.to_dict() for query in self.holdouts],
                "strategy": self.strategy.value, "query_budget": self.budget.queries,
                "seconds": self.budget.seconds, "legality_query_budget": self.legality_query_budget}


def _complete_inputs(problem, values):
    query = problem.query_from_inputs(values)
    canonical = problem.canonical_inputs(query)
    return FrozenMapping({**canonical, **{field.name: values[field.name] for field in problem.sampled_inputs}})


def _scope_task(task, queries):
    provider = copy.copy(task.provider)
    by_query = defaultdict(list)
    for case in provider._cases:
        query = FrozenMapping(case.query()) if callable(case.query) else provider.tuning_inputs(case.query)
        by_query[query].append(case)
    missing = tuple(query for query in queries if query not in by_query)
    if missing:
        factory = getattr(provider, "cases_for_tuning_queries", None)
        if factory is None:
            raise ValueError(f"{provider.component_id} has no production fixture factory for {len(missing)} requested queries")
        for case in factory(missing):
            query = FrozenMapping(case.query()) if callable(case.query) else provider.tuning_inputs(case.query)
            if query not in missing:
                raise ValueError("query fixture factory changed a requested tuning input")
            by_query[query].append(case)
    if any(not by_query[query] for query in queries):
        raise ValueError("query fixture factory did not cover every requested query")
    provider._cases = tuple(case for query in queries for case in by_query[query])
    if len({case.case_id for case in provider._cases}) != len(provider._cases):
        raise ValueError("production fixture factory returned duplicate scenarios")
    if hasattr(provider, "_geometries"):
        provider._geometries = tuple({case.geometry.key: case.geometry for case in provider._cases}.values())
    return replace(task, provider=provider)


def _project_runtime_fields(node, fields, memo=None):
    memo = {} if memo is None else memo
    if id(node) in memo:
        return memo[id(node)]
    if isinstance(node, ProfileLeaf):
        return node
    if node.field not in fields:
        if not isinstance(node, ExactDecisionNode) or len(node.branches) != 1 or node.default is not None:
            raise ValueError("runtime policy cannot dispatch on an omitted tuning input")
        result = _project_runtime_fields(node.branches[0][1], fields, memo)
    else:
        result = replace(node, branches=tuple((value, _project_runtime_fields(child, fields, memo))
                                              for value, child in node.branches),
                         default=None if node.default is None else _project_runtime_fields(node.default, fields, memo))
    memo[id(node)] = result
    return result


def _compile_and_validate(fit, problem, domains, device, validate_decision=None, compile_constraints=None):
    """Prove lowering legality at every query claimed by the bounded planner."""
    if problem.materialize_collection is None:
        planner = _project_runtime_fields(fit.planner, problem.policy.query_fields)
        count = 0
        records, expected = [], []

        def validate_runtime(query, lowered):
            runtime_leaf = planner.lookup(problem.policy.encode_query(query))
            if runtime_leaf is None:
                raise ValueError("runtime query encoding lost declared search coverage")
            runtime = problem.policy.decode_profile(query, device, runtime_leaf.config)
            problem.policy.validate_config(query, runtime, device)
            if runtime != lowered:
                raise ValueError("runtime policy differs from the qualified independent decision")

        for domain in domains:
            for values in domain.queries():
                inputs = _complete_inputs(problem, values)
                query = problem.query_from_inputs(inputs)
                selected = fit.planner.lookup(inputs)
                if selected is None:
                    raise ValueError("fitted policy has an uncovered query inside its declared domain")
                lowered = problem.lower(query, device, selected.config)
                if validate_decision is not None:
                    validate_decision(inputs, device, selected.config)
                if compile_constraints is None:
                    validate_runtime(query, lowered)
                else:
                    records.append(DecisionRecord.create(query=problem.policy.encode_query(query), config=selected.config))
                    expected.append((query, lowered))
                count += 1
        if compile_constraints is not None:
            planner = compile_constraints(records, device=device)
            for query, lowered in expected:
                validate_runtime(query, lowered)
        return planner, count

    raise ValueError("sampled-input collection regions require per-query precision confirmation")


def _holdout_partitions(point, training, problem, domain, fit):
    names = ["all_holdouts"]
    peers = [item.point for item in training if item.point.family == point.family]
    for name, binding in (("geometry", BindingTime.MODEL), ("capacity", BindingTime.PLAN)):
        axes = [index for index, field in enumerate(problem.axes) if field.binding is binding]
        varying = [index for index in axes if domain.axes[index].count > 1]
        key = tuple(point.coordinates[index] for index in varying)
        if varying and all(key != tuple(peer.coordinates[index] for index in varying) for peer in peers):
            names.append(name)
    if any(axis.count > 2 for axis in domain.axes) and all(
        axis.count == 1 or axis.minimum < coordinate < axis.maximum
        for axis, coordinate in zip(domain.axes, point.coordinates, strict=True)
    ):
        names.append("interior")
    for axis in domain.axes:
        for step in (-axis.alignment, axis.alignment):
            neighbor = dict(point.query)
            neighbor[axis.name] += step
            if domain.contains(neighbor):
                for field in problem.inputs:
                    if field.role is FieldRole.DERIVED:
                        neighbor.pop(field.name, None)
                neighbor = _complete_inputs(problem, neighbor)
                if fit.select(neighbor) != fit.select(point.query):
                    return (*names, "decision_boundary")
    return tuple(names)


def _required_partitions(problem, domain, fit):
    required = {"all_holdouts"}
    for name, binding in (("geometry", BindingTime.MODEL), ("capacity", BindingTime.PLAN)):
        if any(axis.count > 1 and field.binding is binding
               for axis, field in zip(domain.axes, problem.axes, strict=True)):
            required.add(name)
    if all(axis.count == 1 or axis.count > 2 for axis in domain.axes) and domain.size > 1:
        required.add("interior")
    first = None
    for query in domain.queries():
        selected = fit.select(_complete_inputs(problem, query))
        if first is not None and selected != first:
            required.add("decision_boundary")
            break
        first = selected
    return required


class QualifiedSearchGenerator:
    """Execute component measurement programs with a frozen search contract."""

    def __init__(self, generator, plan: ProfileSearchPlan):
        if getattr(generator, "artifact_kind", "runtime_profile") != "runtime_profile":
            raise ValueError("API qualification providers do not emit runtime dispatch regions")
        self.component_id = generator.component_id
        self.query_schema_version = generator.query_schema_version
        self.config_schema_version = generator.config_schema_version
        self.problem = generator.problem
        self.plan = plan
        if self.problem.sampled_inputs:
            raise ValueError("sampled-input collection regions require per-query precision confirmation; use the component production sweep")
        tasks = generator.measurement_program
        races = [task for task in tasks if task.kind == "candidate_race"]
        self.probes = tuple(task for task in tasks if task.kind == "fixed_backend_probe")
        if len(races) != 1 or len(self.probes) > 1:
            raise ValueError("a searched component requires exactly one decision race and at most one fixed-path probe")
        self.training = _search_points(self.problem, tuple(_complete_inputs(self.problem, query) for query in plan.training))
        self.holdouts = _search_points(self.problem, tuple(_complete_inputs(self.problem, query) for query in plan.holdouts))
        if len({point.key for point in (*self.training, *self.holdouts)}) != len(self.training) + len(self.holdouts):
            raise ValueError("canonical training and holdout queries overlap")
        self.domains = {domain.fixed: domain for domain in plan.domains}
        if len(self.domains) != len(plan.domains):
            raise ValueError("a search family requires one unambiguous coverage domain")
        axes = tuple(field.name for field in self.problem.axes)
        if any(tuple(axis.name for axis in domain.axes) != axes for domain in plan.domains):
            raise ValueError("search domain axes differ from the component's ordered coordinates")
        for domain in plan.domains:
            for axis, field in zip(domain.axes, self.problem.axes, strict=True):
                field.validate(axis.minimum)
                field.validate(axis.maximum)
                if axis.alignment % field.alignment:
                    raise ValueError("search domain alignment violates the production axis contract")
        for point in (*self.training, *self.holdouts):
            domain = self.domains.get(point.family)
            if domain is None or not domain.contains(point.query):
                raise ValueError("a training or holdout query lies outside the declared family domain")
        if {point.family for point in self.training} != set(self.domains):
            raise ValueError("every covered family requires training support")
        if {point.family for point in self.holdouts} != set(self.domains):
            raise ValueError("every covered family requires independent holdouts")
        if plan.budget.queries < len(self.domains):
            raise ValueError("search budget cannot seed every declared family")
        self.race = _scope_task(races[0], tuple(point.query for point in (*self.training, *self.holdouts)))
        self._validate_region_decision = getattr(self.race.provider, "validate_region_decision", None)
        self._compile_constraints = getattr(self.race.provider, "compile_constraint_coverage", None)
        if (sum(domain.size for domain in plan.domains) > len(self.training) + len(self.holdouts)
                and self._validate_region_decision is None):
            raise ValueError(f"{self.component_id} must declare production candidate legality before emitting unmeasured regions")
        self.measurement_program = (*self.probes, self.race)

    def estimate(self, context):
        probe_units = sum(task.provider.estimate(context).work_units for task in self.probes)
        queries = min(self.plan.budget.queries, len(self.training)) + len(self.holdouts)
        legality = sum(domain.size for domain in self.plan.domains)
        return WorkEstimate(component_id=self.component_id, work_units=queries + legality + probe_units,
                            case_count=len(self.race.provider._cases) + sum(task.provider._probe.case_count for task in self.probes),
                            description="bounded shape search, exhaustive CPU lowering, and independent production holdouts",
                            dimensions={"training_pool_queries": len(self.training),
                                        "maximum_training_queries": min(self.plan.budget.queries, len(self.training)),
                                        "independent_holdout_queries": len(self.holdouts),
                                        "legality_queries": legality, "strategy": self.plan.strategy.value,
                                        "measurement_program": [task.describe() for task in self.measurement_program]})

    def generate(self, context, *, progress, checkpoints):
        started = time.monotonic()
        contract_id = stable_identity(self.plan.to_dict())
        search_context = replace(context, measurement_cohort=f"{context.measurement_cohort}:search:production")
        holdout_context = replace(context, measurement_cohort=f"{context.measurement_cohort}:holdout:independent")
        # Stage checkpoint namespaces retain both cohorts across repeated resumes.
        stores = {name: CheckpointStore(checkpoints.root / name / contract_id,
                                       observations_path=checkpoints.observations_path)
                  for name in ("search", "holdout")}
        progress.start_stage(self.component_id, stage="shape-space production races", total=min(self.plan.budget.queries, len(self.training)))
        with self.race.open_search(search_context, stores["search"]) as search:
            search.points = self.training
            measure = search.measure
            def measured(point):
                result = measure(point)
                progress.advance(self.component_id, detail=point.key[:12])
                return result
            search.measure = measured
            outcome = search.search(strategy=self.plan.strategy, budget=self.plan.budget)
            decisions = dict(search.configs)
        if {item.point.family for item in outcome.measurements} != set(self.domains):
            raise RuntimeError("search budget exhausted before every coverage family was measured")
        if any(set(item.candidate_ids) != set(item.latencies_us) for item in outcome.measurements):
            raise RuntimeError("production candidate correctness failure blocks profile generation")
        fit_started = time.monotonic()
        fit = fit_regret_regions(outcome.measurements, decisions,
                                 axes=tuple(field.name for field in self.problem.axes), domains=self.plan.domains)
        fit_seconds = time.monotonic() - fit_started
        progress.start_stage(self.component_id, stage="validate every covered query", total=sum(domain.size for domain in self.plan.domains))
        lowering_started = time.monotonic()
        planner, legality_count = _compile_and_validate(fit, self.problem, self.plan.domains, context.device,
                                                       self._validate_region_decision, self._compile_constraints)
        progress.advance(self.component_id, units=legality_count)
        lowering_seconds = time.monotonic() - lowering_started
        required = set()
        for family, domain in self.domains.items():
            names = _required_partitions(self.problem, domain, fit)
            required.update(names)
            required.update(f"family.{stable_identity(family)}.{name}" for name in names)
        progress.start_stage(self.component_id, stage="independent production holdouts", total=len(self.holdouts))
        cases = []
        with self.race.open_search(holdout_context, stores["holdout"]) as search:
            for point in self.holdouts:
                measurement = search.measure(point)
                partitions = _holdout_partitions(point, outcome.measurements, self.problem, self.domains[point.family], fit)
                partitions = (*partitions, *(f"family.{stable_identity(point.family)}.{name}" for name in partitions))
                cases.append(QualificationCase(measurement=measurement, selected_candidate=fit.select(point.query),
                                             partition=partitions[0], additional_partitions=partitions[1:],
                                             cohort=holdout_context.measurement_cohort))
                progress.advance(self.component_id, detail=point.key[:12])
        report = qualify_policy(cases, training_queries=frozenset(point.key for point in self.training),
                                required_partitions=frozenset(required))
        evidence = {"search_contract": self.plan.to_dict(), "contract_id": contract_id,
                    "generation": context.checkpoint_metadata(), "sampling": outcome.accounting(),
                    "fit": fit.describe(), "fit_seconds": fit_seconds, "lowering_seconds": lowering_seconds,
                    "legality_queries": legality_count, "qualification": report.to_dict(),
                    "required_partitions": sorted(required),
                    "holdouts": [{"query": case.measurement.point.query.to_dict(),
                                  "selected_candidate": case.selected_candidate,
                                  "candidate_ids": list(case.measurement.candidate_ids),
                                  "latencies_us": case.measurement.latencies_us.to_dict(),
                                  "partitions": [case.partition, *case.additional_partitions]} for case in cases],
                    "gpu_measurement_cases": sum(len(search_cases) for point, search_cases in self._measured_cases(outcome)),
                    "generation_seconds": time.monotonic() - started}
        checkpoints.save(self.component_id, f"search-report-{contract_id}", evidence)
        if not report.passed:
            raise RuntimeError(f"{self.component_id} failed independent policy qualification; report search-report-{contract_id}")
        probe_units = 0
        for task in self.probes:
            probe = task.provider.qualify(context, progress=progress, checkpoints=checkpoints)
            from .providers.tunable import _with_default_leaf
            planner = _with_default_leaf(planner, ProfileLeaf.create(name="measured-production-implementation", config=probe.encoded_config))
            evidence["production_probe"] = probe.evidence
            evidence["gpu_measurement_cases"] += probe.evidence["gpu_measurement_cases"]
            probe_units += probe.completed_work_units
        evidence["generation_seconds"] = time.monotonic() - started
        return ComponentGenerationResult(component={"component_id": self.component_id,
                    "query_schema_version": self.query_schema_version, "config_schema_version": self.config_schema_version,
                    "planner": decision_node_to_dict(planner, compact=True),
                    "coverage": {"domains": [domain.to_dict() for domain in self.plan.domains],
                                 "legality_queries": legality_count}}, evidence=evidence,
                    completed_work_units=len(outcome.measurements) + len(self.holdouts) + legality_count + probe_units,
                    completion_reason="qualified", qualification=report.to_dict())

    def _measured_cases(self, outcome):
        measured = {item.point.query for item in outcome.measurements} | {point.query for point in self.holdouts}
        grouped = defaultdict(list)
        for case in self.race.provider._cases:
            query = FrozenMapping(case.query()) if callable(case.query) else self.race.provider.tuning_inputs(case.query)
            if query in measured:
                grouped[query].append(case)
        return grouped.items()


__all__ = ["ProfileSearchPlan", "QualifiedSearchGenerator"]
