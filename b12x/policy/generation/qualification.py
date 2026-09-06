"""Empirical regret gates for independently measured policy holdouts."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .search import QueryMeasurement, SearchStrategy


@dataclass(frozen=True, kw_only=True)
class QualificationLimits:
    geometric_mean_regret: float = 0.005
    worst_regret: float = 0.02

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in (
            self.geometric_mean_regret, self.worst_regret,
        )):
            raise ValueError("regret limits must be finite and nonnegative")


@dataclass(frozen=True, kw_only=True)
class QualificationCase:
    measurement: QueryMeasurement
    selected_candidate: str | None
    partition: str
    cohort: str
    additional_partitions: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.partition or not self.cohort or self.cohort == "initial":
            raise ValueError("qualification requires a named independent cohort and partition")
        if self.cohort != self.measurement.cohort:
            raise ValueError("qualification cannot relabel a measurement cohort")
        names = (self.partition, *self.additional_partitions)
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("qualification partitions must be nonempty and unique")

    @property
    def ratio(self) -> float:
        latencies = self.measurement.latencies_us
        if (self.selected_candidate not in latencies or
                (self.measurement.eligible_candidates is not None
                 and self.selected_candidate not in self.measurement.eligible_candidates)):
            return math.inf
        return latencies[self.selected_candidate] / latencies[self.measurement.winner]


@dataclass(frozen=True, kw_only=True)
class QualificationReport:
    passed: bool
    cases: int
    geometric_mean_regret: float | None
    worst_regret: float | None
    invalid_selections: int
    failed_candidate_cases: int
    partitions: Mapping[str, Mapping[str, object]]
    missing_partitions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"status": "qualified" if self.passed else "research-only", "cases": self.cases,
                "geometric_mean_regret": self.geometric_mean_regret,
                "observed_worst_regret": self.worst_regret,
                "invalid_selections": self.invalid_selections,
                "failed_candidate_cases": self.failed_candidate_cases,
                "partitions": dict(self.partitions), "missing_partitions": list(self.missing_partitions),
                "guarantee": "empirical_on_independent_holdouts"}


def _statistics(cases: Sequence[QualificationCase], limits: QualificationLimits) -> dict[str, object]:
    ratios = [case.ratio for case in cases]
    invalid = sum(not math.isfinite(value) for value in ratios)
    failed = sum(set(case.measurement.candidate_ids) != set(case.measurement.latencies_us) for case in cases)
    if invalid or not ratios:
        return {"cases": len(cases), "passed": False, "geometric_mean_regret": None,
                "worst_regret": None, "invalid_selections": invalid, "failed_candidate_cases": failed}
    mean_log = math.fsum(math.log(value) for value in ratios) / len(ratios)
    geometric = math.expm1(mean_log)
    worst = max(ratios) - 1.
    return {"cases": len(cases), "passed": not failed and mean_log <= math.log1p(limits.geometric_mean_regret) and max(ratios) <= 1. + limits.worst_regret,
            "geometric_mean_regret": geometric, "worst_regret": worst, "invalid_selections": 0,
            "failed_candidate_cases": failed}


_DEFAULT_LIMITS = QualificationLimits()


def qualify_policy(cases: Sequence[QualificationCase], *, training_queries: frozenset[str],
                   required_partitions: frozenset[str],
                   limits: QualificationLimits = _DEFAULT_LIMITS) -> QualificationReport:
    """Gate the aggregate and every declared independent holdout partition.

    The caller must exhaust the production candidate set at each holdout query.
    A failed selection cannot be dropped from the denominator or replaced by
    the heuristic. These observations certify empirical regret, not a uniform
    guarantee over unmeasured queries in a continuous region.
    """
    if not required_partitions:
        raise ValueError("qualification must declare its required holdout partitions")
    identities = [case.measurement.point.key for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("qualification cannot count a paired observation twice")
    if any(case.measurement.point.key in training_queries for case in cases):
        raise ValueError("qualification queries overlap training data")
    grouped = defaultdict(list)
    for case in cases:
        for partition in (case.partition, *case.additional_partitions):
            grouped[partition].append(case)
    partitions = {name: _statistics(values, limits) for name, values in sorted(grouped.items())}
    missing = tuple(sorted(required_partitions - partitions.keys()))
    aggregate = _statistics(cases, limits)
    return QualificationReport(
        passed=bool(aggregate["passed"] and not missing and all(value["passed"] for value in partitions.values())),
        cases=len(cases), geometric_mean_regret=aggregate["geometric_mean_regret"],
        worst_regret=aggregate["worst_regret"], invalid_selections=aggregate["invalid_selections"],
        failed_candidate_cases=aggregate["failed_candidate_cases"],
        partitions=partitions, missing_partitions=missing,
    )


@dataclass(frozen=True, kw_only=True)
class QualifiedStrategy:
    strategy: SearchStrategy
    qualification: QualificationReport
    generation_seconds: float

    def __post_init__(self):
        if not math.isfinite(self.generation_seconds) or self.generation_seconds < 0:
            raise ValueError("strategy cost must be finite and nonnegative")


def select_strategy(results: Sequence[QualifiedStrategy], *, simplicity_tolerance: float = 0.05) -> QualifiedStrategy:
    if not 0 <= simplicity_tolerance < 1:
        raise ValueError("simplicity tolerance must be in [0, 1)")
    qualified = [result for result in results if result.qualification.passed]
    if not qualified:
        raise ValueError("no search strategy passed independent policy qualification")
    fastest = min(result.generation_seconds for result in qualified)
    simplicity = {strategy: i for i, strategy in enumerate(SearchStrategy)}
    return min((result for result in qualified if result.generation_seconds <= fastest * (1. + simplicity_tolerance)),
               key=lambda result: (simplicity[result.strategy], result.generation_seconds))


__all__ = ["QualificationLimits", "QualificationCase", "QualificationReport", "qualify_policy",
           "QualifiedStrategy", "select_strategy"]
