from dataclasses import replace

import pytest

from b12x.policy.catalog import list_generation_components
from b12x.policy.problem import AxisInterval, FieldRole, SearchDomain
from b12x.policy.types import DeviceIdentity, FrozenMapping


def test_every_registered_component_owns_complete_field_accounting():
    for registration in list_generation_components():
        problem = registration.load_problem()
        assert problem.component_id == registration.component_id
        independent = {field.name for field in problem.inputs
                       if field.role not in (FieldRole.ENVIRONMENT, FieldRole.DERIVED)}
        derived = {field.name for field in problem.inputs if field.role is FieldRole.DERIVED}
        assert independent | derived >= problem.policy.query_fields
        assert problem.describe()["component_id"] == registration.component_id


def test_every_registered_component_has_an_executable_measurement_program():
    kinds = set()
    for registration in list_generation_components():
        generator = registration.create_generator()
        assert generator.measurement_program
        for task in generator.measurement_program:
            description = task.describe()
            assert description["measurement_cases"] > 0
            assert description["query_points"] > 0
            kinds.add(description["kind"])
    assert kinds == {"fixed_backend_probe", "candidate_race"}


def test_moe_derived_rows_are_not_an_independent_axis():
    from b12x.moe.fused_moe._policy import MoeDecodeQuery, TUNING_PROBLEM

    query = MoeDecodeQuery("nvfp4", "modelopt_nvfp4", "silu", 256, 2048, 1024, 8, 16, 128)
    assert {field.name for field in TUNING_PROBLEM.axes} == {
        "hidden_size", "intermediate_size", "num_tokens", "num_experts", "top_k",
    }
    assert TUNING_PROBLEM.canonical_inputs(query)["routed_rows"] == 128
    assert TUNING_PROBLEM.family_key(query) == TUNING_PROBLEM.family_key(replace(
        query, num_tokens=32, routed_rows=256,
    ))
    with pytest.raises(ValueError, match="inconsistent derived"):
        TUNING_PROBLEM.canonical_inputs(replace(query, routed_rows=129))


def test_missing_and_cyclic_input_declarations_fail():
    from b12x.moe.fused_moe._policy import TUNING_PROBLEM

    with pytest.raises(ValueError, match="input accounting"):
        replace(TUNING_PROBLEM, inputs=TUNING_PROBLEM.inputs[:-1])
    with pytest.raises(ValueError, match="cyclic or missing"):
        replace(TUNING_PROBLEM, inputs=tuple(
            replace(field, dependencies=("routed_rows",))
            if field.role is FieldRole.DERIVED else field
            for field in TUNING_PROBLEM.inputs
        ))


def test_blockscaled_live_rows_are_separate_from_policy_queries():
    from b12x.gemm.blockscaled._policy import TUNING_PROBLEM

    assert "measured_m" not in TUNING_PROBLEM.policy.query_fields
    assert TUNING_PROBLEM.sampled_inputs[0].name == "measured_m"
    assert TUNING_PROBLEM.sampled_inputs[0].binding.value == "runtime"
    assert TUNING_PROBLEM.derived_config_fields == ("a16_rows",)


def test_precision_collection_lowers_only_declared_exact_live_counts():
    from b12x.gemm.blockscaled._policy import BlockscaledQuery, TUNING_PROBLEM

    device = DeviceIdentity(vendor="nvidia", product_name="test", compute_capability=(12, 1), sm_count=48)
    query = BlockscaledQuery(recipe="mxfp8", in_features=128, out_features=256)
    a16 = {"precision": "a16", "tile_n": 128, "tile_k": 64, "split_k": 4}
    decisions = [({"measured_m": 8}, a16), ({"measured_m": 2}, {"precision": "quantized"})]
    config = TUNING_PROBLEM.lower_collection(query, device, decisions)
    assert config.a16_rows == ((8, 128, 64, 4),)
    assert config.select(8) == (128, 64, 4)
    assert config.select(7) is None
    with pytest.raises(ValueError, match="unique"):
        TUNING_PROBLEM.lower_collection(query, device, decisions + decisions)
    with pytest.raises(ValueError, match="active kernel"):
        TUNING_PROBLEM.validate_decision({**a16, "precision": "quantized"})
    with pytest.raises(ValueError, match="domain"):
        TUNING_PROBLEM.validate_decision({**a16, "split_k": True})


def test_domains_preserve_alignment_and_fixed_value_types():
    domain = SearchDomain(fixed=FrozenMapping({"mode": True}), axes=(
        AxisInterval(name="width", minimum=64, maximum=512, alignment=64),
        AxisInterval(name="rows", minimum=1, maximum=8),
    ))
    assert domain.size == 64
    assert domain.contains({"mode": True, "width": 128, "rows": 3})
    assert not domain.contains({"mode": 1, "width": 128, "rows": 3})
    assert not domain.contains({"mode": True, "width": 129, "rows": 3})
    assert not domain.contains({"mode": True, "width": 128, "rows": 9})
    with pytest.raises(ValueError, match="endpoints"):
        AxisInterval(name="width", minimum=65, maximum=512, alignment=64)


def test_gqa_decision_materializes_each_shape_without_cuda_discovery(monkeypatch):
    import torch
    from b12x.attention.paged._policy import GqaDecision, GqaQuery

    def forbidden(*args, **kwargs):
        raise AssertionError("pure schedule lowering must use the supplied device identity")

    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    device = DeviceIdentity(vendor="nvidia", product_name="nvidia gb10",
                            compute_capability=(12, 1), sm_count=48)
    query = GqaQuery(device=None, mode="decode", q_dtype="bfloat16", kv_dtype="bfloat16",
                     q_heads=16, kv_heads=2, head_dim_qk=128, head_dim_vo=128, page_size=64,
                     kv_cache_layout="separate", batch_size=1, query_len=1, cache_tokens=128,
                     window_left=-1, requested_graph_ctas_per_sm=None,
                     requested_max_work_items=None, requested_max_partial_rows=None,
                     force_split_kv=None)
    decision = GqaDecision(graph_ctas_per_sm=2, force_split_kv=False)
    first = decision.materialize(query, device)
    second = decision.materialize(replace(query, batch_size=8, cache_tokens=4096), device)
    assert first.max_effective_kv_pages == 2
    assert second.max_effective_kv_pages == 64
    assert second.max_work_items == 8 * first.max_work_items
    assert first.max_partial_rows == second.max_partial_rows == 0
    with pytest.raises(ValueError, match="work items"):
        decision.materialize(replace(query, batch_size=8, requested_max_work_items=1), device)


def test_gqa_cache_identity_retains_workspace_constraints():
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery

    kwargs = dict(device=None, mode="decode", q_dtype="bfloat16", kv_dtype="bfloat16",
                  q_heads=16, kv_heads=2, head_dim_qk=128, head_dim_vo=128, page_size=64,
                  kv_cache_layout="separate", batch_size=1, query_len=1, cache_tokens=128,
                  window_left=-1, requested_graph_ctas_per_sm=None,
                  requested_max_work_items=None, requested_max_partial_rows=None,
                  force_split_kv=None)
    query = GqaQuery(**kwargs)
    assert GQA_POLICY.encode_cache_query(query) != GQA_POLICY.encode_cache_query(
        replace(query, requested_max_work_items=8))
    assert GQA_POLICY.encode_cache_query(query) != GQA_POLICY.encode_cache_query(
        replace(query, requested_max_partial_rows=0))
