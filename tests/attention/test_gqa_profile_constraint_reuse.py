from dataclasses import asdict

import pytest
import torch

import b12x
from b12x.attention import paged
from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
from b12x.attention.paged.reference import paged_attention_reference
from b12x.policy import PolicyContext, PolicyMode, PolicySource, ProfileRegistry
from b12x.policy.generation.providers.attention import GqaAttentionGenerator
from b12x.policy.generation.reducer import DecisionRecord, decision_node_to_dict
from b12x.policy.serialization import profile_from_dict
from tests.conftest import require_b12x


@pytest.mark.parametrize("window,partial", ((-1, None), (127, None), (-1, 0)))
def test_profile_choice_survives_workspace_sizing_and_high_page_graph_replay(window, partial):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    identity = PolicyContext.for_device(device).device
    query = GqaQuery(device=device, mode="decode", q_dtype="bfloat16", kv_dtype="bfloat16",
                     q_heads=8, kv_heads=1, head_dim_qk=128, head_dim_vo=128,
                     page_size=128, batch_size=1, query_len=1, cache_tokens=256,
                     window_left=window, kv_cache_layout="combined", requested_graph_ctas_per_sm=None,
                     requested_max_work_items=None, requested_max_partial_rows=partial, force_split_kv=None)
    record = DecisionRecord.create(query=query.profile_fields(), config={"graph_ctas_per_sm": 4, "force_split_kv": None})
    tree = GqaAttentionGenerator().build_planner((record,), device=identity)
    registry = ProfileRegistry()
    registry.register(profile_from_dict(dict(profile_id="test.gqa-constraints", targets=[asdict(identity)], components=[dict(
        component_id=GQA_POLICY.component_id, query_schema_version=3, config_schema_version=3,
        planner=decision_node_to_dict(tree, compact=True))])))
    policy = PolicyContext.for_identity(identity, registry=registry, mode=PolicyMode.PREPLANNED_ONLY)
    capacity = paged.decode_graph_capacity(device=device, q_dtype=torch.bfloat16, kv_dtype=torch.bfloat16,
        num_q_heads=8, num_kv_heads=1, head_dim_qk=128, head_dim_vo=128, page_size=128,
        batch=1, max_cache_page_count=2, window_left=window, max_partial_rows=partial,
        kv_cache_layout="combined", policy=policy)
    stride = 2 * 128 * 128 * 2
    high_page = (2**31 - 1) // stride + 2
    storage = torch.empty((high_page + 2, 2, 128, 1, 128), device=device, dtype=torch.bfloat16)
    assert high_page * storage.stride(0) * storage.element_size() > 2**31
    storage[high_page:].normal_(std=.25)
    k, v = storage[:, 0], storage[:, 1]
    q = torch.randn((1, 8, 128), device=device, dtype=torch.bfloat16) * .25
    output = torch.empty_like(q)
    page_table = torch.tensor([[high_page, high_page + 1]], device=device, dtype=torch.int32)
    lengths = torch.tensor([193], device=device, dtype=torch.int32)
    cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    plan = paged.plan(paged.Caps(device=device, mode="decode", dtype=torch.bfloat16, kv_dtype=torch.bfloat16,
        num_q_heads=8, num_kv_heads=1, head_dim_qk=128, head_dim_vo=128, page_size=128,
        max_total_q=1, max_batch=1, max_page_table_width=2, max_work_items=capacity.max_work_items,
        max_partial_rows=capacity.max_partial_rows, num_cache_pages=high_page + 2, use_cuda_graph=True,
        copy_runtime_metadata=False, kv_cache_layout="combined"), policy=policy)
    plan.prepare_decode_graph_replay_state(batch=1, total_q_capacity=1, max_page_table_width=2,
                                          max_cache_page_count=2, window_left=window)
    assert plan.policy_resolution.source is PolicySource.PREPLANNED
    assert plan.policy_resolution.config == capacity.policy_resolution.config
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    binding = paged.bind(plan, scratch=scratch, q=q, k_cache=k, v_cache=v, output=output,
                         page_table=page_table, cache_seqlens=lengths, cu_seqlens_q=cu_q,
                         active_total_q=1, window_left=window)
    paged.run(binding=binding)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    b12x.freeze_kernel_resolution("GQA profile workspace and paged-pool qualification")
    try:
        with torch.cuda.graph(graph):
            paged.run(binding=binding)
        addresses = tuple(tensor.data_ptr() for tensor in (scratch, storage, output))
        for live in (193, 65, 256):
            lengths.fill_(live)
            expected, _ = paged_attention_reference(q, k, v, page_table, lengths, cu_q, window_left=window)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            assert addresses == tuple(tensor.data_ptr() for tensor in (scratch, storage, output))
            assert torch.isfinite(output).all() and torch.count_nonzero(output)
            torch.testing.assert_close(output.float(), expected.float(), rtol=.05, atol=.02)
    finally:
        b12x.unfreeze_kernel_resolution()
        graph.reset()
