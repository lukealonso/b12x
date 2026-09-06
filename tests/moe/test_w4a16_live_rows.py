"""One W4A16 execution capacity must accept changing live rows after prewarm."""

import torch
import pytest

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.moe import fused_moe
from tests.conftest import require_b12x
from tests.moe.test_w4a16_e2e import _make_weights, _reference_w4a16, _assert_matches_oracle


@pytest.mark.parametrize("capacity", (1, 8))
def test_w4a16_small_m_plan_reuses_prepared_callable_across_live_counts(capacity):
    from b12x.moe._shared.kernels.w4a16 import kernel

    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate, topk = 8, 128, 128, 2
    torch.manual_seed(9045)
    weights = _make_weights(experts=experts, hidden_size=hidden, intermediate_size=intermediate, activation="silu")
    w13, s13, g13, w2, s2, g2 = weights
    plan_weights = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat.MODELOPT_NVFP4,
                                      w13_layout=fused_moe.W13Layout.W13),
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A16, nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=experts, hidden_size=hidden, intermediate_size=intermediate))
    prepared = fused_moe.prepare_weights(plan=plan_weights, weights=fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2,
        w13_global_scales=g13, w2_global_scales=g2))
    plan = fused_moe.plan_execution(experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=topk, warmup_token_counts=(capacity,)))
    fused_moe.prewarm(plan)
    compiled = {key: id(value.compiled) for key, value in kernel._SMALL_M_DIRECT_CACHE.items()}
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    x = torch.randn(capacity, hidden, device=device, dtype=torch.bfloat16) * .25
    ids = torch.randint(experts, (capacity, topk), device=device, dtype=torch.int32)
    routes = torch.softmax(torch.randn(capacity, topk, device=device), dim=-1)
    output = torch.empty_like(x)
    addresses = (x.data_ptr(), output.data_ptr(), scratch.data_ptr())
    for rows in ((1,) if capacity == 1 else (8, 1, 3, 7)):
        expected = _reference_w4a16(x[:rows], *weights, ids[:rows], routes[:rows], activation="silu")
        freeze_kernel_resolution("W4A16 live counts must reuse a prewarmed execution capacity")
        graph = torch.cuda.CUDAGraph()
        try:
            binding = fused_moe.bind(plan, scratch=scratch, a=x[:rows], experts=prepared,
                topk_weights=routes[:rows], topk_ids=ids[:rows], output=output[:rows], input_scales_static=True)
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            allocated = torch.cuda.memory_allocated(device)
            for _ in range(3):
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                _assert_matches_oracle(output[:rows], expected, activation="silu")
                assert torch.isnan(output[rows:]).all()
                assert (x.data_ptr(), output.data_ptr(), scratch.data_ptr()) == addresses
                assert compiled == {key: id(value.compiled) for key, value in kernel._SMALL_M_DIRECT_CACHE.items()}
        finally:
            unfreeze_kernel_resolution()
            graph.reset()
