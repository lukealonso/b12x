"""Offline sampling of conditional shape spaces and decision boundaries."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from b12x.policy.problem import stable_identity
from b12x.policy.types import FrozenMapping


class SearchStrategy(str, Enum):
    EXHAUSTIVE = "exhaustive"
    SPACE_FILLING = "space_filling"
    ADAPTIVE = "adaptive"
    BAYESIAN = "bayesian"


@dataclass(frozen=True, kw_only=True)
class SearchPoint:
    query: FrozenMapping
    family: FrozenMapping
    coordinates: tuple[int, ...]

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in self.coordinates):
            raise ValueError("search coordinates must be nonnegative integers")

    @property
    def key(self) -> str:
        return stable_identity(self.query)


@dataclass(frozen=True, kw_only=True)
class QueryMeasurement:
    """Scenario-reduced candidate latencies from production measurement.

    Failed or ineligible candidates are absent. Relative log latency removes
    each query's common scale without imposing monotonicity on kernel choices.
    """

    point: SearchPoint
    latencies_us: FrozenMapping
    candidate_ids: tuple[str, ...]
    fresh: bool
    costs_seconds: FrozenMapping = FrozenMapping()
    cohort: str = "initial"
    eligible_candidates: tuple[str, ...] | None = None
    tie_break_order: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.latencies_us or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in self.latencies_us.values()
        ):
            raise ValueError("query measurements require positive qualified latencies")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in self.costs_seconds.values()):
            raise ValueError("measurement costs must be finite and nonnegative")
        if (not self.cohort or not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or not set(self.latencies_us) <= set(self.candidate_ids)):
            raise ValueError("measurement candidates must match their production enumeration")
        if self.eligible_candidates is not None and (
            not self.eligible_candidates or not set(self.eligible_candidates) <= set(self.latencies_us)
        ):
            raise ValueError("eligible decisions must have qualified latency measurements")
        if self.tie_break_order and (len(set(self.tie_break_order)) != len(self.candidate_ids)
                                     or set(self.tie_break_order) != set(self.candidate_ids)):
            raise ValueError("tie-break order must cover the production candidate cohort exactly")

    @property
    def winner(self) -> str:
        order = {key: index for index, key in enumerate(self.tie_break_order)}
        return min(self.eligible_candidates or self.latencies_us,
                   key=lambda key: (self.latencies_us[key], order.get(key, 0), key))

    def relative_log_latency(self, candidate: str) -> float:
        return math.log(self.latencies_us[candidate] / self.latencies_us[self.winner])


@dataclass(frozen=True, kw_only=True)
class SearchBudget:
    queries: int
    seconds: float | None = None

    def __post_init__(self):
        if type(self.queries) is not int or self.queries <= 0:
            raise ValueError("query budget must be a positive integer")
        if self.seconds is not None and (not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("time budget must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class SearchOutcome:
    strategy: SearchStrategy
    measurements: tuple[QueryMeasurement, ...]
    available_queries: int
    stop_reason: str
    wall_seconds: float
    selection_seconds: float
    progressions: tuple[dict[str, object], ...] = ()

    @property
    def exhausted_domain(self) -> bool:
        return len(self.measurements) == self.available_queries

    def accounting(self) -> dict[str, object]:
        costs: dict[str, float] = {}
        for measurement in self.measurements:
            for name, value in measurement.costs_seconds.items():
                costs[name] = costs.get(name, 0.) + value
        fresh = sum(item.fresh for item in self.measurements)
        return {"strategy": self.strategy.value, "available_queries": self.available_queries,
                "requested_queries": len(self.measurements), "fresh_queries": fresh,
                "reused_queries": len(self.measurements) - fresh, "stop_reason": self.stop_reason,
                "wall_seconds": self.wall_seconds, "selection_seconds": self.selection_seconds,
                "measurement_costs_seconds": costs, "progressions": list(self.progressions), "qualified": False}


class SpatialSampler:
    """Farthest-point coverage, boundary refinement, or local GP uncertainty.

    Exploration steps remain mandatory even when neighboring winners agree.
    Equal labels at sampled corners are not a certificate for the interior.
    """

    def __init__(self, points: Sequence[SearchPoint], strategy: SearchStrategy):
        import numpy as np

        self.points = tuple(points)
        self.strategy = SearchStrategy(strategy)
        if not self.points or len({point.key for point in self.points}) != len(self.points):
            raise ValueError("search points must be nonempty and unique")
        self.families = tuple(stable_identity(point.family) for point in self.points)
        dimensions = {len(point.coordinates) for point in self.points}
        if len(dimensions) != 1:
            raise ValueError("one sampler requires a consistent coordinate declaration")
        self.features = np.log2(1. + np.array([point.coordinates for point in self.points], dtype=float))
        for family in set(self.families):
            indices = [i for i, value in enumerate(self.families) if value == family]
            values = self.features[indices]
            low, high = values.min(axis=0), values.max(axis=0)
            self.features[indices] = (values - low) / np.where(high > low, high - low, 1.)
        self.measured: dict[int, QueryMeasurement] = {}
        self.distances = np.full((len(self.points), 2), np.inf)
        self.neighbors = np.full((len(self.points), 2), -1, dtype=int)
        self.axis_lines = defaultdict(list)
        self.point_lines = []
        for index, point in enumerate(self.points):
            keys = tuple((self.families[index], axis, point.coordinates[:axis], point.coordinates[axis + 1:])
                         for axis in range(len(point.coordinates)))
            self.point_lines.append(keys)
            for key in keys:
                self.axis_lines[key].append(index)
        for key, indices in self.axis_lines.items():
            indices.sort(key=lambda index: self.points[index].coordinates[key[1]])
        self.axis_boundaries = {}

    def observe(self, index: int, measurement: QueryMeasurement) -> None:
        import numpy as np

        if index in self.measured or measurement.point != self.points[index]:
            raise ValueError("observation does not match an unmeasured search point")
        self.measured[index] = measurement
        indices = np.array([i for i, family in enumerate(self.families) if family == self.families[index]])
        distances = np.linalg.norm(self.features[indices] - self.features[index], axis=1)
        first = distances < self.distances[indices, 0]
        shifted = indices[first]
        self.distances[shifted, 1] = self.distances[shifted, 0]
        self.neighbors[shifted, 1] = self.neighbors[shifted, 0]
        self.distances[shifted, 0] = distances[first]
        self.neighbors[shifted, 0] = index
        second = (~first) & (distances < self.distances[indices, 1])
        self.distances[indices[second], 1] = distances[second]
        self.neighbors[indices[second], 1] = index
        for key in self.point_lines[index]:
            line = self.axis_lines[key]
            observed = [i for i in line if i in self.measured]
            proposals = {}
            for left_index, right_index in zip(observed, observed[1:], strict=False):
                left, right = self.measured[left_index], self.measured[right_index]
                if (left.winner == right.winner and
                        set(left.eligible_candidates or left.latencies_us) == set(right.eligible_candidates or right.latencies_us)):
                    continue
                axis = key[1]
                low, high = self.features[left_index, axis], self.features[right_index, axis]
                interior = [i for i in line if i not in self.measured and low < self.features[i, axis] < high]
                if interior:
                    midpoint = (low + high) / 2.
                    proposal = min(interior, key=lambda i: (abs(self.features[i, axis] - midpoint), i))
                    proposals[proposal] = float(high - low)
            self.axis_boundaries[key] = proposals

    def _boundary_score(self, index: int) -> float:
        first, second = self.neighbors[index]
        radius = float(self.distances[index, 0])
        if second < 0:
            return radius
        left, right = self.measured[int(first)], self.measured[int(second)]
        disagreement = left.winner != right.winner or set(left.latencies_us) != set(right.latencies_us)
        gap = abs(float(self.distances[index, 1]) - radius)
        axis_boundary = max((self.axis_boundaries.get(key, {}).get(index, 0.)
                             for key in self.point_lines[index]), default=0.)
        return radius + (2. / (1. + gap) if disagreement else 0.) + (2. + axis_boundary if axis_boundary else 0.)

    def _bayesian_score(self, index: int) -> float:
        import numpy as np

        indices = [i for i in self.measured if self.families[i] == self.families[index]]
        indices.sort(key=lambda i: (float(np.linalg.norm(self.features[i] - self.features[index])), i))
        indices = indices[:32]
        candidates = sorted({candidate for i in indices for candidate in self.measured[i].latencies_us})
        predictions = []
        for candidate in candidates:
            support = [i for i in indices if candidate in self.measured[i].latencies_us]
            if len(support) < 2:
                predictions.append((0., 1.))
                continue
            x = self.features[support]
            y = np.array([self.measured[i].relative_log_latency(candidate) for i in support])
            distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
            nonzero = distances[distances > 0]
            length = max(float(np.median(nonzero)) if nonzero.size else 1., 0.05)
            amplitude = max(float(np.std(y)), 0.05)

            def matern(distance):
                scaled = math.sqrt(3.) * distance / length
                return amplitude**2 * (1. + scaled) * np.exp(-scaled)

            covariance = matern(distances) + np.eye(len(support)) * 0.005**2
            cross = matern(np.linalg.norm(x - self.features[index], axis=1))
            factor = np.linalg.cholesky(covariance)
            centered = y - y.mean()
            mean = float(y.mean() + cross @ np.linalg.solve(factor.T, np.linalg.solve(factor, centered)))
            solved = np.linalg.solve(factor, cross)
            sigma = math.sqrt(max(0., amplitude**2 - float(solved @ solved)))
            predictions.append((mean, sigma))
        if not predictions:
            return self._boundary_score(index)
        best = min(predictions, key=lambda value: value[0])
        optimistic = min(mean - 2. * sigma for mean, sigma in predictions)
        return best[0] + 2. * best[1] - optimistic + 0.05 * float(self.distances[index, 0])

    def choose(self) -> int | None:
        import numpy as np

        remaining = [i for i in range(len(self.points)) if i not in self.measured]
        if not remaining:
            return None
        if self.strategy is SearchStrategy.EXHAUSTIVE:
            return remaining[0]
        unseeded = [i for i in remaining if self.neighbors[i, 0] < 0]
        if unseeded:
            family = self.families[unseeded[0]]
            return min((i for i in unseeded if self.families[i] == family),
                       key=lambda i: (float(np.linalg.norm(self.features[i] - 0.5)), i))
        if self.strategy is SearchStrategy.SPACE_FILLING or len(self.measured) % 4 == 0:
            return max(remaining, key=lambda i: (self.distances[i, 0], -i))
        if self.strategy is SearchStrategy.ADAPTIVE:
            return max(remaining, key=lambda i: (self._boundary_score(i), -i))
        proposals = sorted(remaining, key=lambda i: (-self._boundary_score(i), i))[:64]
        return max(proposals, key=lambda i: (self._bayesian_score(i), -i))


def run_search(points: Sequence[SearchPoint], *, strategy: SearchStrategy, budget: SearchBudget,
               measure: Callable[[SearchPoint], QueryMeasurement]) -> SearchOutcome:
    started = time.monotonic()
    sampler = SpatialSampler(points, strategy)
    selection_seconds = 0.
    stop_reason = "domain_exhausted"
    while len(sampler.measured) < len(sampler.points):
        if len(sampler.measured) >= budget.queries:
            stop_reason = "query_budget"
            break
        if budget.seconds is not None and time.monotonic() - started >= budget.seconds:
            stop_reason = "time_budget"
            break
        selecting = time.monotonic()
        index = sampler.choose()
        selection_seconds += time.monotonic() - selecting
        sampler.observe(index, measure(sampler.points[index]))
    return SearchOutcome(strategy=SearchStrategy(strategy), measurements=tuple(sampler.measured.values()),
                         available_queries=len(sampler.points), stop_reason=stop_reason,
                         wall_seconds=time.monotonic() - started, selection_seconds=selection_seconds)


__all__ = ["SearchStrategy", "SearchPoint", "QueryMeasurement", "SearchBudget", "SearchOutcome",
           "SpatialSampler", "run_search"]
