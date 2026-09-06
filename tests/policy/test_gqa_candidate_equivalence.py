from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.policy import FrozenMapping
from b12x.policy.generation.attention_corpus import gqa_cases
from b12x.policy.generation.providers.attention import GqaAttentionGenerator
from b12x.policy.generation.providers.gpu_workers import _GqaSession, _gqa_execution_key
from b12x.policy.generation.sweep import SweepCandidate


def test_gqa_races_distinct_schedules_and_keeps_the_first_representative():
    case = gqa_cases()[0]
    session = _GqaSession((case,), SimpleNamespace())

    def capacity(_case, *, graph_ctas_per_sm, force_split_kv):
        ctas = graph_ctas_per_sm or 2
        split = force_split_kv is not False
        return SimpleNamespace(
            graph_ctas_per_sm=ctas, cta_tile_q=16, query_tiles_per_request=1,
            architecture_max_chunks_per_request=ctas * 94,
            max_chunks_per_request=2 if split else 1,
            max_work_items=2 if split else 1,
            max_partial_rows=2 if split else 0,
            max_effective_kv_pages=2, worst_page_count=2 if split else 1,
            chunk_pages_lut=(1, 1) if split else (1, 2),
        )

    session._capacity = capacity
    candidates = session.candidates(case)

    assert len(candidates) == 2
    assert {item.config["max_chunks_per_request"] for item in candidates} == {1, 2}
    assert {item.config["graph_ctas_per_sm"] for item in candidates} == {2}
    assert GqaAttentionGenerator(cases=(case,))._candidate_contract_version == 5
    decisions = [decision for candidate in candidates for decision in (candidate.decision, *candidate.equivalent_decisions)]
    assert len(decisions) == len(set(decisions)) == 18
    assert {decision["graph_ctas_per_sm"] for decision in decisions} == {None, 1, 2, 3, 4, 6}
    assert {decision["force_split_kv"] for decision in decisions} == {None, False, True}


def test_gqa_allocation_groups_cannot_replace_query_capacity_with_group_maximum():
    case = gqa_cases()[0]
    larger = replace(case, query=FrozenMapping({**case.query, "cache_tokens": 2 * case.query["cache_tokens"]}))
    with pytest.raises(ValueError, match="planned cache capacity"):
        _GqaSession((case, larger), SimpleNamespace())


def test_gqa_workspace_limits_are_profile_coordinates_and_prune_illegal_splits():
    import torch
    from b12x.policy import EMBEDDED_REGISTRY, PolicyContext, PolicySource
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
    from b12x.attention.paged.planner import _plan_decode_graph_capacity_heuristic

    device = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm").targets[0]
    case = next(case for case in gqa_cases() if case.query["cache_tokens"] == 16384 and case.query["batch_size"] == 1)
    query = GqaQuery(device=None, **case.query)
    context = PolicyContext.for_identity(device)
    assert context.resolve(GQA_POLICY, query).source is PolicySource.PREPLANNED
    restricted = replace(query, requested_max_partial_rows=0)
    assert GQA_POLICY.encode_query(query) != GQA_POLICY.encode_query(restricted)
    resolution = context.resolve(GQA_POLICY, restricted)
    assert resolution.source is PolicySource.HEURISTIC
    assert resolution.config.max_partial_rows == 0
    assert resolution.config.max_chunks_per_request == 1

    cases = tuple(GqaAttentionGenerator.cases_for_tuning_queries((restricted.profile_fields(),)))
    session = _GqaSession(cases, SimpleNamespace())
    def capacity(case, *, graph_ctas_per_sm, force_split_kv):
        values = case.query
        return _plan_decode_graph_capacity_heuristic(
            device=device, q_dtype=getattr(torch, values["q_dtype"]), kv_dtype=getattr(torch, values["kv_dtype"]),
            num_q_heads=values["q_heads"], num_kv_heads=values["kv_heads"], head_dim_qk=values["head_dim_qk"],
            head_dim_vo=values["head_dim_vo"], page_size=values["page_size"], batch=values["batch_size"],
            max_cache_page_count=values["cache_tokens"] // values["page_size"],
            graph_ctas_per_sm=graph_ctas_per_sm, force_split_kv=force_split_kv,
            max_partial_rows=values["requested_max_partial_rows"])
    session._capacity = capacity
    candidates = session.candidates(cases[0])
    assert candidates
    assert all(candidate.config["max_partial_rows"] == 0 for candidate in candidates)
    decisions = [decision for candidate in candidates for decision in (candidate.decision, *candidate.equivalent_decisions)]
    assert len(decisions) == 12
    assert {decision["force_split_kv"] for decision in decisions} == {None, False}


def test_gqa_profiles_reuse_only_unchanged_schedules_under_nonbinding_limits():
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
    from b12x.policy import EMBEDDED_REGISTRY, ProfileRegistry, PolicyContext, PolicyMode, PolicySource
    from b12x.policy.generation.reducer import DecisionRecord
    from b12x.policy.serialization import profile_from_dict
    from dataclasses import asdict

    device = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm").targets[0]
    query = GqaQuery(device=None, **gqa_cases()[0].query)
    query = replace(query, q_heads=24, kv_heads=4, head_dim_qk=128, head_dim_vo=128,
                    batch_size=3, cache_tokens=2048, page_size=128)
    decision = FrozenMapping({"graph_ctas_per_sm": 4, "force_split_kv": None})
    selected = GQA_POLICY.decode_profile(query, device, decision)
    generator = GqaAttentionGenerator()
    planner = generator.build_planner((DecisionRecord.create(query=query.profile_fields(), config=decision),), device=device)
    from b12x.policy.generation.reducer import decision_node_to_dict
    registry = ProfileRegistry()
    registry.register(profile_from_dict(dict(profile_id="test.constraints", targets=[asdict(device)], components=[dict(
        component_id=GQA_POLICY.component_id, query_schema_version=3, config_schema_version=3,
        planner=decision_node_to_dict(planner, compact=True))])))
    context = PolicyContext.for_identity(device, registry=registry, mode=PolicyMode.PREPLANNED_ONLY)
    for work in (None, selected.max_work_items, selected.max_work_items * 4):
        for partial in (None, selected.max_partial_rows, selected.max_partial_rows * 4):
            constrained = replace(query, requested_max_work_items=work, requested_max_partial_rows=partial)
            resolution = context.resolve(GQA_POLICY, constrained)
            assert resolution.source is PolicySource.PREPLANNED
            assert resolution.config == selected
    assert planner.lookup(replace(query, requested_max_partial_rows=0).profile_fields()) is None
    assert planner.lookup(replace(query, cache_tokens=4096).profile_fields()) is None


@pytest.mark.parametrize("work,partial", ((None, 0), (None, 9), (9, None)))
def test_gqa_partially_bounded_coverage_preserves_the_measured_launch(work, partial):
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
    from b12x.policy import EMBEDDED_REGISTRY
    from b12x.policy.generation.reducer import DecisionRecord

    device = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm").targets[0]
    query = replace(GqaQuery(device=None, **gqa_cases()[0].query), q_heads=24, kv_heads=4,
                    head_dim_qk=128, head_dim_vo=128, page_size=128, batch_size=3,
                    cache_tokens=16384, requested_max_work_items=work, requested_max_partial_rows=partial)
    decision = FrozenMapping({"graph_ctas_per_sm": 4, "force_split_kv": None})
    config = GQA_POLICY.decode_profile(query, device, decision)
    planner = GqaAttentionGenerator().build_planner(
        (DecisionRecord.create(query=query.profile_fields(), config=decision),), device=device)
    field, requirement = (("requested_max_work_items", config.max_work_items) if work is None
                          else ("requested_max_partial_rows", config.max_partial_rows))
    for value in (None, requirement, requirement + 1, requirement * 4):
        bounded = replace(query, **{field: value})
        leaf = planner.lookup(bounded.profile_fields())
        assert leaf is not None
        assert GQA_POLICY.decode_profile(bounded, device, leaf.config) == config
    if requirement:
        assert planner.lookup(replace(query, **{field: requirement - 1}).profile_fields()) is None
    exact_field = "requested_max_partial_rows" if work is None else "requested_max_work_items"
    assert planner.lookup(replace(query, **{exact_field: None}).profile_fields()) is None


def test_gqa_explicit_constraint_records_precede_nonbinding_coverage():
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
    from b12x.policy import EMBEDDED_REGISTRY
    from b12x.policy.generation.reducer import DecisionRecord, decision_node_to_dict

    device = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm").targets[0]
    query = GqaQuery(device=None, **gqa_cases()[0].query)
    original = FrozenMapping({"graph_ctas_per_sm": 4, "force_split_kv": None})
    explicit = FrozenMapping({"graph_ctas_per_sm": 1, "force_split_kv": None})
    bound = GQA_POLICY.decode_profile(query, device, original).max_work_items * 4
    bounded = replace(query, requested_max_work_items=bound)
    records = (DecisionRecord.create(query=query.profile_fields(), config=original),
               DecisionRecord.create(query=bounded.profile_fields(), config=explicit))
    generator = GqaAttentionGenerator()
    planner = generator.build_planner(records, device=device)
    assert planner.lookup(query.profile_fields()).config == original
    assert planner.lookup(bounded.profile_fields()).config == explicit
    assert planner.lookup(replace(query, requested_max_work_items=bound + 1).profile_fields()).config == original
    assert decision_node_to_dict(planner, compact=True) == decision_node_to_dict(
        generator.build_planner(tuple(reversed(records)), device=device), compact=True)


@pytest.mark.parametrize("field,value", (
    ("base_chunk_pages_runs", [[2, 2]]),
    ("max_work_items", 4),
    ("max_partial_rows", 4),
    ("cta_tile_q", 64),
))
def test_gqa_equivalence_retains_execution_and_workspace_fields(field, value):
    case = gqa_cases()[0]
    config = dict(
        graph_ctas_per_sm=2, architecture_max_chunks_per_request=188,
        base_chunk_pages_runs=[[2, 1]], max_work_items=2, max_partial_rows=2,
        cta_tile_q=16,
    )
    original = SweepCandidate.create(config)
    metadata_only = SweepCandidate.create({**config, "graph_ctas_per_sm": 4,
                                          "architecture_max_chunks_per_request": 376})
    changed = SweepCandidate.create({**config, field: value})

    assert _gqa_execution_key(case, original) == _gqa_execution_key(case, metadata_only)
    assert _gqa_execution_key(case, original) != _gqa_execution_key(case, changed)
    verify = replace(case, query=FrozenMapping({**case.query.to_dict(), "mode": "verify"}))
    assert _gqa_execution_key(verify, original) != _gqa_execution_key(verify, metadata_only)
