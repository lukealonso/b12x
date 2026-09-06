from dataclasses import replace

import pytest

from b12x.policy.generation.search import SearchPoint, SearchStrategy, SearchBudget, QueryMeasurement, run_search
from b12x.policy.generation.qualification import QualificationCase, QualifiedStrategy, qualify_policy, select_strategy
from b12x.policy.types import FrozenMapping


def _point(x, family='a'):
    return SearchPoint(query=FrozenMapping({'x': x, 'family': family}),
                       family=FrozenMapping({'family': family}), coordinates=(x,))


def _measure(point, *, cohort='initial', a=1., b=2.):
    return QueryMeasurement(point=point, latencies_us=FrozenMapping({'a': a, 'b': b}),
                            candidate_ids=('a', 'b'), fresh=True, cohort=cohort)


@pytest.mark.parametrize('strategy', list(SearchStrategy))
def test_every_strategy_exhausts_without_repeating_or_crossing_families(strategy):
    points = tuple(_point(x, family) for family in ('a', 'b') for x in range(1, 13))
    calls = []
    def measure(point):
        calls.append(point.key)
        x = point.coordinates[0]
        return _measure(point, a=1. if x < 5 or x > 9 else 2., b=1.3)
    result = run_search(points, strategy=strategy, budget=SearchBudget(queries=len(points)), measure=measure)
    assert result.exhausted_domain
    assert result.stop_reason == 'domain_exhausted'
    assert len(set(calls)) == len(points)
    assert result.accounting()['qualified'] is False
    assert result.accounting()['fresh_queries'] == len(points)


def test_budget_exhaustion_is_not_profile_completion():
    points = tuple(_point(x) for x in range(32))
    result = run_search(points, strategy=SearchStrategy.ADAPTIVE,
                        budget=SearchBudget(queries=5), measure=_measure)
    assert result.stop_reason == 'query_budget'
    assert not result.exhausted_domain
    assert len(result.measurements) == 5
    assert result.accounting()['qualified'] is False


def test_boundary_sampler_keeps_exploring_and_detects_an_interior_island():
    points = tuple(_point(x) for x in range(1, 34))
    def measure(point):
        x = point.coordinates[0]
        return _measure(point, a=2. if 7 <= x <= 12 else 1., b=1.5)
    result = run_search(points, strategy=SearchStrategy.ADAPTIVE,
                        budget=SearchBudget(queries=10), measure=measure)
    observed = sorted(result.measurements, key=lambda item: item.point.coordinates)
    labels = [item.winner for item in observed]
    assert 'b' in labels
    assert labels[0] == labels[-1] == 'a'


def test_axis_refinement_uses_a_bracket_with_other_coordinates_fixed():
    from b12x.policy.generation.search import SpatialSampler

    points = tuple(SearchPoint(query=FrozenMapping({'x': x, 'y': y}), family=FrozenMapping({'family': 'a'}),
                               coordinates=(x, y)) for y in (10, 20) for x in range(1, 18))
    sampler = SpatialSampler(points, SearchStrategy.ADAPTIVE)
    sampler.observe(0, _measure(points[0], a=1., b=2.))
    sampler.observe(16, _measure(points[16], a=2., b=1.))
    selected = points[sampler.choose()]
    assert selected.coordinates[1] == 10
    assert 1 < selected.coordinates[0] < 17


def test_progression_audit_reports_reversals_and_separates_inapplicable_knobs():
    from b12x.policy.generation.progression import audit_progressions

    decisions = {name: {'tile': value} for name, value in [('small', 16), ('large', 64), ('inactive', None)]}
    observations = []
    for family, winners in [('reversal', ('small', 'large', 'small')),
                            ('gap', ('small', 'inactive', 'large'))]:
        for x, winner in enumerate(winners, 1):
            observations.append(QueryMeasurement(point=_point(x, family),
                                latencies_us=FrozenMapping({winner: 1.}), candidate_ids=(winner,), fresh=True))
    report, = audit_progressions(observations, decisions, axes=('x',), ordered_knobs=('tile',))
    assert report['observed_segments'] == report['nonmonotone'] == 1
    assert report['reversal_examples'][0]['values'] == [16, 64, 16]
    assert report['guarantee'] == 'observed_samples_only'


def _holdout(x, *, ratio=1., partition='capacity', selected='a'):
    measurement = _measure(_point(x), cohort='qualification', a=ratio, b=1.)
    return QualificationCase(measurement=measurement, selected_candidate=selected,
                             partition=partition, cohort='qualification')


def _qualify(cases, **kwargs):
    return qualify_policy(cases, training_queries=frozenset(),
                          required_partitions=frozenset({'capacity'}), **kwargs)


def test_qualification_rejects_bad_worst_case_even_with_small_mean_regret():
    cases = [_holdout(i) for i in range(100)] + [_holdout(101, ratio=1.021)]
    result = _qualify(cases)
    assert result.geometric_mean_regret < .005
    assert result.worst_regret > .02
    assert not result.passed


def test_qualification_cannot_hide_a_bad_partition_in_aggregate():
    cases = [_holdout(i) for i in range(100)] + [_holdout(101, ratio=1.01, partition='geometry')]
    result = _qualify(cases)
    assert result.geometric_mean_regret < .005
    assert result.worst_regret < .02
    assert not result.passed
    assert result.partitions['geometry']['passed'] is False


def test_qualification_requires_independent_queries_cohorts_and_complete_candidates():
    case = _holdout(5)
    with pytest.raises(ValueError, match='overlap'):
        qualify_policy([case], training_queries=frozenset({case.measurement.point.key}),
                       required_partitions=frozenset({'capacity'}))
    with pytest.raises(ValueError, match='twice'):
        _qualify([case, case])
    with pytest.raises(ValueError, match='relabel'):
        replace(case, measurement=replace(case.measurement, cohort='initial'))
    failed = replace(case, measurement=replace(case.measurement, candidate_ids=('a', 'b', 'failed')))
    report = _qualify([failed])
    assert report.failed_candidate_cases == 1
    assert not report.passed
    missing = _qualify([replace(case, selected_candidate=None)])
    assert missing.invalid_selections == 1
    assert not missing.passed
    assert not _qualify([]).passed


def test_precision_eligibility_is_a_hard_selection_constraint():
    case = _holdout(3, ratio=.8)
    restricted = replace(case.measurement, eligible_candidates=('b',))
    report = _qualify([replace(case, measurement=restricted)])
    assert report.invalid_selections == 1
    assert not report.passed


def test_strategy_selection_uses_total_cost_and_prefers_simplicity_within_five_percent():
    passed = _qualify([_holdout(1)])
    failed = _qualify([_holdout(1, ratio=1.1)])
    results = [QualifiedStrategy(strategy=SearchStrategy.BAYESIAN, qualification=passed, generation_seconds=100.),
               QualifiedStrategy(strategy=SearchStrategy.ADAPTIVE, qualification=passed, generation_seconds=104.),
               QualifiedStrategy(strategy=SearchStrategy.SPACE_FILLING, qualification=failed, generation_seconds=1.)]
    assert select_strategy(results).strategy is SearchStrategy.ADAPTIVE
    with pytest.raises(ValueError, match='no search strategy'):
        select_strategy([results[-1]])
