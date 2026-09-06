"""Fit bounded decision regions by measured latency regret, with no runtime model."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from b12x.policy.problem import stable_identity
from b12x.policy.types import ExactDecisionNode, FrozenMapping, MatchRange, ProfileLeaf, RangeDecisionNode


@dataclass(frozen=True, kw_only=True)
class RegionFit:
    planner: object
    training_queries: frozenset[str]
    leaf_count: int
    geometric_mean_regret: float
    worst_regret: float

    def select(self, query):
        leaf = self.planner.lookup(query)
        return None if leaf is None else leaf.name

    def describe(self):
        return {"status": "research-only", "training_queries": len(self.training_queries),
                "leaves": self.leaf_count, "training_geometric_mean_regret": self.geometric_mean_regret,
                "training_worst_regret": self.worst_regret,
                "qualification": "requires independent production-candidate holdouts"}


def _compact_ranges(node, memo=None):
    if memo is None:
        memo = {}
    key = id(node)
    if key in memo:
        return memo[key]
    if isinstance(node, ProfileLeaf):
        return node
    default = None if node.default is None else _compact_ranges(node.default, memo)
    if isinstance(node, ExactDecisionNode):
        result = ExactDecisionNode(field=node.field, branches=tuple(
            (value, _compact_ranges(child, memo)) for value, child in node.branches
        ), default=default)
        memo[key] = result
        return result
    branches = []
    for interval, child in node.branches:
        child = _compact_ranges(child, memo)
        expanded = (child.branches if isinstance(child, RangeDecisionNode)
                    and child.field == node.field and child.default is None
                    and node.default is None else ((interval, child),))
        for child_interval, descendant in expanded:
            low, high = max(interval.minimum, child_interval.minimum), min(interval.maximum, child_interval.maximum)
            if low > high:
                continue
            if branches and branches[-1][0].maximum + 1 == low and branches[-1][1] == descendant:
                previous, _ = branches.pop()
                low = previous.minimum
            branches.append((MatchRange(low, high), descendant))
    result = RangeDecisionNode(field=node.field, branches=tuple(branches), default=default)
    memo[key] = result
    return result


def fit_regret_regions(measurements, decisions, *, axes, domains,
                       geometric_mean_limit=.0025, worst_limit=.01,
                       max_leaves=4096, max_depth=32, split_proposals=32):
    """Greedily partition explicit domains using scenario-reduced log regret.

    A leaf may select only a decision eligible at every training point it owns.
    Prefix sums evaluate a bounded number of thresholds per coordinate. The
    domains define proposed dispatch coverage; their interiors require separate
    legality and regret qualification before the tree may become a profile.
    """
    import numpy as np

    if not measurements or len({item.point.key for item in measurements}) != len(measurements):
        raise ValueError("region fitting requires unique nonempty training queries")
    if any(type(value) is not int or value <= 0 for value in (max_leaves, max_depth, split_proposals)):
        raise ValueError("region fitting budgets must be positive integers")
    if any(not math.isfinite(value) or value < 0 for value in (geometric_mean_limit, worst_limit)):
        raise ValueError("training regret limits must be finite and nonnegative")
    axes = tuple(axes)
    grouped = defaultdict(list)
    for item in measurements:
        if len(item.point.coordinates) != len(axes):
            raise ValueError("training coordinates differ from the declared axes")
        grouped[item.point.family].append(item)
    domain_map = {stable_identity(domain.fixed): domain for domain in domains}
    if len(domain_map) != len(domains):
        raise ValueError("each family requires exactly one explicit search domain")
    trees = {}
    leaves = 0
    for family, samples in grouped.items():
        domain = domain_map.get(stable_identity(family))
        if domain is None or tuple(axis.name for axis in domain.axes) != axes:
            raise ValueError("training family lacks its exact declared coordinate domain")
        if any(not domain.contains(item.point.query) for item in samples):
            raise ValueError("training queries lie outside their declared search domain")
        candidates = sorted({key for item in samples for key in item.latencies_us})
        if not set(candidates) <= set(decisions):
            raise ValueError("training decisions lack their independent kernel parameters")
        coordinates = np.array([item.point.coordinates for item in samples], dtype=np.int64)
        costs = np.full((len(samples), len(candidates)), np.inf)
        for row, item in enumerate(samples):
            eligible = set(item.eligible_candidates or item.latencies_us)
            for column, candidate in enumerate(candidates):
                if candidate in eligible:
                    costs[row, column] = item.relative_log_latency(candidate)
        finite = np.isfinite(costs)
        numerical = np.where(finite, costs, 0.)

        def best(indices):
            valid = finite[indices].all(axis=0)
            sums = numerical[indices].sum(axis=0)
            sums[~valid] = np.inf
            column = int(np.argmin(sums))
            return (column if valid[column] else None), sums

        def build(indices, bounds, depth):
            nonlocal leaves
            column, sums = best(indices)
            if column is not None:
                within = (sums <= len(indices) * math.log1p(geometric_mean_limit)) & (
                    costs[indices].max(axis=0) <= math.log1p(worst_limit))
                if within.any():
                    column = int(np.argmin(np.where(within, sums, np.inf)))
                if within.any() or len(indices) == 1:
                    leaves += 1
                    if leaves > max_leaves:
                        raise ValueError("region fitting exhausted its leaf budget")
                    candidate = candidates[column]
                    return ProfileLeaf(name=candidate, config=FrozenMapping(decisions[candidate]))
            if depth >= max_depth:
                raise ValueError("region fitting exhausted its depth budget")
            best_split = None
            for axis in range(len(axes)):
                ordered = indices[np.argsort(coordinates[indices, axis], kind="stable")]
                values = coordinates[ordered, axis]
                cuts = np.flatnonzero(values[:-1] != values[1:])
                if not len(cuts):
                    continue
                cuts = cuts[np.unique(np.linspace(0, len(cuts) - 1, min(split_proposals, len(cuts)), dtype=int))]
                prefix = np.cumsum(numerical[ordered], axis=0)
                invalid = np.cumsum(~finite[ordered], axis=0)
                for cut in cuts:
                    left = np.where(invalid[cut] == 0, prefix[cut], np.inf)
                    right = np.where(invalid[-1] == invalid[cut], prefix[-1] - prefix[cut], np.inf)
                    left_best, right_best = float(left.min()), float(right.min())
                    unsupported = (cut + 1 if not math.isfinite(left_best) else 0) + (
                        len(indices) - cut - 1 if not math.isfinite(right_best) else 0)
                    score = (int(unsupported),
                             sum(value for value in (left_best, right_best) if math.isfinite(value)),
                             abs(len(indices) - 2 * (cut + 1)), axis, int(values[cut]))
                    if best_split is None or score < best_split[0]:
                        best_split = (score, axis, ordered, int(cut))
            if best_split is None:
                raise ValueError("coordinates cannot separate incompatible training decisions")
            _, axis, ordered, cut = best_split
            interval = domain.axes[axis]
            low, high = bounds[axis]
            left_value, right_value = (int(coordinates[ordered[i], axis]) for i in (cut, cut + 1))
            threshold = ((left_value + right_value) // (2 * interval.alignment)) * interval.alignment
            left_bounds, right_bounds = list(bounds), list(bounds)
            left_bounds[axis] = (low, threshold)
            right_bounds[axis] = (threshold + 1, high)
            return RangeDecisionNode(field=axes[axis], branches=(
                (MatchRange(low, threshold), build(ordered[:cut + 1], tuple(left_bounds), depth + 1)),
                (MatchRange(threshold + 1, high), build(ordered[cut + 1:], tuple(right_bounds), depth + 1)),
            ))

        bounds = tuple((axis.minimum, axis.maximum) for axis in domain.axes)
        tree = build(np.arange(len(samples)), bounds, 0)
        for axis in reversed(domain.axes):
            if axis.alignment == 1:
                tree = RangeDecisionNode(field=axis.name, branches=((MatchRange(axis.minimum, axis.maximum), tree),))
            else:
                if axis.count > 65536:
                    raise ValueError("aligned region coverage exceeds the exact-branch representation budget")
                tree = ExactDecisionNode(field=axis.name, branches=tuple(
                    (value, tree) for value in range(axis.minimum, axis.maximum + 1, axis.alignment)
                ))
        trees[family] = tree
    fields = tuple(sorted(next(iter(grouped))))
    if any(tuple(sorted(family)) != fields for family in grouped):
        raise ValueError("families must share their declared fixed fields")

    def combine(families, depth=0):
        if depth == len(fields):
            if len(families) != 1:
                raise ValueError("family fields do not distinguish region trees")
            return trees[families[0]]
        field = fields[depth]
        branches = defaultdict(list)
        for family in families:
            value = family[field]
            branches[(type(value).__name__, value)].append(family)
        return ExactDecisionNode(field=field, branches=tuple(
            (key[1], combine(values, depth + 1)) for key, values in sorted(branches.items(), key=lambda item: repr(item[0]))
        ))

    planner = _compact_ranges(combine(tuple(grouped)))
    regret = [item.relative_log_latency(planner.lookup(item.point.query).name) for item in measurements]
    return RegionFit(planner=planner, training_queries=frozenset(item.point.key for item in measurements),
                     leaf_count=len(tuple(planner.iter_leaves())), geometric_mean_regret=math.expm1(math.fsum(regret) / len(regret)),
                     worst_regret=math.expm1(max(regret)))


__all__ = ["RegionFit", "fit_regret_regions"]
