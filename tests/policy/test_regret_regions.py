import pytest

from b12x.policy.generation.qualification import QualificationCase, qualify_policy
from b12x.policy.generation.regions import fit_regret_regions
from b12x.policy.generation.search import QueryMeasurement, SearchPoint
from b12x.policy.problem import AxisInterval, SearchDomain
from b12x.policy.types import FrozenMapping


DECISIONS = {"a": {"tile": 16}, "b": {"tile": 64}}


def _measurement(x, *, family="f", a=1., b=2., cohort="initial", eligible=None):
    return QueryMeasurement(point=SearchPoint(query=FrozenMapping({"family": family, "x": x}),
                            family=FrozenMapping({"family": family}), coordinates=(x,)),
                            latencies_us=FrozenMapping({"a": a, "b": b}), candidate_ids=("a", "b"),
                            fresh=True, cohort=cohort, eligible_candidates=eligible)


def _domain(minimum=1, maximum=17, alignment=1):
    return SearchDomain(fixed=FrozenMapping({"family": "f"}), axes=(
        AxisInterval(name="x", minimum=minimum, maximum=maximum, alignment=alignment),
    ))


def test_regret_fit_keeps_one_region_for_near_ties_with_alternating_winners():
    measurements = [_measurement(x, a=1. if x % 2 else 1.001, b=1.001 if x % 2 else 1.)
                    for x in range(1, 18)]
    fit = fit_regret_regions(measurements, DECISIONS, axes=("x",), domains=(_domain(),))
    assert fit.leaf_count == 1
    assert fit.worst_regret < .002
    assert fit.describe()["status"] == "research-only"
    assert fit.select({"family": "unknown", "x": 5}) is None
    assert fit.select({"family": "f", "x": 0}) is None
    assert fit.select({"family": "f", "x": 18}) is None


def test_interior_island_fits_but_unseen_boundary_error_still_fails_qualification():
    def measure(x, cohort="initial"):
        return _measurement(x, a=2. if 7 <= x <= 11 else 1., b=1.5, cohort=cohort)
    fit = fit_regret_regions([measure(x) for x in range(1, 18, 2)], DECISIONS,
                             axes=("x",), domains=(_domain(),))
    assert fit.leaf_count == 3
    assert fit.worst_regret == 0.
    assert fit.select({"family": "f", "x": 9}) == "b"
    holdout = measure(12, "independent-boundary")
    qualification = qualify_policy([QualificationCase(measurement=holdout,
                                   selected_candidate=fit.select(holdout.point.query),
                                   partition="boundary", cohort="independent-boundary")],
                                   training_queries=fit.training_queries, required_partitions=frozenset({"boundary"}))
    assert not qualification.passed
    assert qualification.worst_regret == .5
    with pytest.raises(ValueError, match="leaf budget"):
        fit_regret_regions([measure(x) for x in range(1, 18, 2)], DECISIONS,
                            axes=("x",), domains=(_domain(),), max_leaves=2)


def test_candidate_legality_and_domain_alignment_remain_dispatch_holes():
    measurements = [_measurement(4, a=1., b=3.), _measurement(8, a=1., b=3., eligible=("b",)),
                    _measurement(12, a=1., b=3.)]
    fit = fit_regret_regions(measurements, DECISIONS, axes=("x",), domains=(_domain(4, 12, 4),))
    assert fit.select({"family": "f", "x": 4}) == "a"
    assert fit.select({"family": "f", "x": 8}) == "b"
    assert fit.select({"family": "f", "x": 12}) == "a"
    assert fit.select({"family": "f", "x": 6}) is None
    assert fit.worst_regret == 0.


def test_aligned_multiaxis_coverage_preserves_shared_subtrees():
    family = FrozenMapping({"family": "f"})
    measurements = [QueryMeasurement(
        point=SearchPoint(query=FrozenMapping({"family": "f", "x": x, "y": y}),
                          family=family, coordinates=(x, y)),
        latencies_us=FrozenMapping({"a": 1., "b": 2.}), candidate_ids=("a", "b"), fresh=True,
    ) for x, y in [(4, 4), (256, 256)]]
    domain = SearchDomain(fixed=family, axes=tuple(
        AxisInterval(name=name, minimum=4, maximum=256, alignment=4) for name in ("x", "y")))
    fit = fit_regret_regions(measurements, DECISIONS, axes=("x", "y"), domains=(domain,))
    x_node = fit.planner.branches[0][1]
    assert len({id(child) for _, child in x_node.branches}) == 1
    assert fit.leaf_count == 1
    assert fit.select({"family": "f", "x": 128, "y": 64}) == "a"
    assert fit.select({"family": "f", "x": 128, "y": 65}) is None
