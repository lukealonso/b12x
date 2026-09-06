"""Scenario-robust latency reduction with independent selection eligibility."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from b12x.policy.types import FrozenMapping


@dataclass(frozen=True, kw_only=True)
class ScenarioScores:
    candidates: tuple
    latencies_us: FrozenMapping
    eligible_candidates: tuple[str, ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    def select(self, tie_breaker: Callable | None = None):
        if not self.eligible_candidates:
            raise RuntimeError("no eligible candidate passed every required scenario")
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        return min((by_id[key] for key in self.eligible_candidates), key=lambda candidate: (
            self.latencies_us[candidate.candidate_id],
            tie_breaker(candidate) if tie_breaker is not None else candidate.candidate_id,
            candidate.candidate_id,
        ))


def reduce_scenarios(scenarios: Sequence[Sequence], *, qualified: Callable,
                     selectable: Callable = lambda item: True,
                     latency: Callable = lambda item: item.latency_us) -> ScenarioScores:
    """Keep correctness failures visible and require eligibility in every scenario.

    Candidates unavailable in any required scenario cannot cover the query.
    Correct, timed candidates remain in the latency table even if a selection
    constraint prevents choosing them. This distinction is required when
    independently confirming a change in activation precision.
    """
    if not scenarios or any(not scenario for scenario in scenarios):
        raise ValueError("scenario reduction requires nonempty candidate races")
    tables = []
    for scenario in scenarios:
        table = {item.candidate.candidate_id: item for item in scenario}
        if len(table) != len(scenario):
            raise ValueError("a scenario contains duplicate candidate measurements")
        tables.append(table)
    common = sorted(set.intersection(*(set(table) for table in tables)))
    candidates, latencies, eligible = [], {}, []
    for key in common:
        measurements = [table[key] for table in tables]
        candidate = measurements[0].candidate
        if any(item.candidate.config != candidate.config for item in measurements):
            raise ValueError("a candidate identity has inconsistent configurations")
        candidates.append(candidate)
        if not all(qualified(item) for item in measurements):
            continue
        times = [latency(item) for item in measurements]
        if any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
               for value in times):
            raise ValueError("qualified candidate timings must be finite and positive")
        latencies[key] = math.exp(math.fsum(math.log(value) for value in times) / len(times))
        if all(selectable(item) for item in measurements):
            eligible.append(key)
    return ScenarioScores(candidates=tuple(candidates), latencies_us=FrozenMapping(latencies),
                          eligible_candidates=tuple(eligible))


__all__ = ["ScenarioScores", "reduce_scenarios"]
