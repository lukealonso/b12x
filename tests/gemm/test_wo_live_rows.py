"""WO capacity specializations reuse compiled kernels and caller-owned scratch."""

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import wo_projection as wo
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    quantize_mxfp8_rows_torch,
    quantize_wo_projection_weights_mxfp8_torch,
)
from tests.conftest import require_b12x


def _quantized_values(x):
    quantized = quantize_mxfp8_rows_torch(x.contiguous())
    return dequantize_mxfp8_rows_torch(quantized.values, quantized.scale_rows)


@pytest.mark.parametrize("inverse_rope", [False, True])
@pytest.mark.parametrize("capacity,groups,width,rank,hidden,hint", [
    (1, 1, 512, 128, 128, None),
    (8, 2, 512, 64, 128, None),
    (17, 1, 512, 128, 128, None),
    (17, 2, 512, 64, 128, 2),
    (16, 4, 4096, 1024, 4096, None),
])
def test_wo_plan_reuses_compiled_kernels_for_partial_live_rows(
    capacity, groups, width, rank, hidden, hint, inverse_rope,
):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(4589)
    head_dim, rope_dim = 512, 64
    heads_per_group = width // head_dim
    x = torch.randn(capacity, groups, width, device=device, dtype=torch.bfloat16) * .25
    positions = torch.arange(capacity, device=device)
    angles = torch.randn(capacity, rope_dim // 2, device=device)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    a = torch.randn(groups, rank, width, device=device, dtype=torch.bfloat16) / width**.5
    b = torch.randn(hidden, groups * rank, device=device, dtype=torch.bfloat16) / (groups * rank)**.5
    weights = quantize_wo_projection_weights_mxfp8_torch(a, b)
    plan = wo.plan(wo.Caps(device=device, max_tokens=capacity, groups=groups, group_width=width, rank=rank, hidden=hidden))
    scratch = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in plan.shapes_and_dtypes())

    def bind(rows):
        common = dict(scratch=scratch, weights=weights, expected_m=hint)
        if inverse_rope:
            return wo.bind_inv_rope(
                plan, o=x[:rows].view(rows, groups * heads_per_group, head_dim),
                positions=positions[:rows], cos_sin_cache=cache,
                heads_per_group=heads_per_group, nope_dim=head_dim-rope_dim,
                rope_dim=rope_dim, **common,
            )
        return wo.bind(plan, source_tgd=x[:rows], **common)

    run = wo.run_inv_rope if inverse_rope else wo.run
    binding = bind(capacity)
    assert binding.expected_m == (capacity if hint is None else hint)
    for _ in range(2):
        run(binding=binding)
    torch.cuda.synchronize(device)
    aq = dequantize_mxfp8_rows_torch(weights.wo_a.values, weights.wo_a.scale_rows)
    if groups == 1:
        aq = aq.unsqueeze(-1)
    bq = dequantize_mxfp8_rows_torch(weights.wo_b.values, weights.wo_b.scale_rows)
    for rows in dict.fromkeys((capacity, 1, min(3, capacity), max(1, capacity-1))):
        source = x[:rows].float()
        if inverse_rope:
            source = source.view(rows, groups * heads_per_group, head_dim).clone()
            pairs = source[..., -rope_dim:].reshape(rows, groups * heads_per_group, rope_dim // 2, 2).clone()
            cos = cache[positions[:rows], :rope_dim//2, None].transpose(1, 2)
            sin = cache[positions[:rows], rope_dim//2:, None].transpose(1, 2)
            source[..., -rope_dim::2] = pairs[..., 0] * cos + pairs[..., 1] * sin
            source[..., -rope_dim+1::2] = pairs[..., 1] * cos - pairs[..., 0] * sin
            source = source.reshape(rows, groups, width)
        intermediate = torch.cat([
            (_quantized_values(source[:, group]) @ aq[..., group].T).to(torch.bfloat16)
            for group in range(groups)
        ], dim=1)
        expected = (_quantized_values(intermediate) @ bq.T).to(torch.bfloat16)
        freeze_kernel_resolution("WO live rows must reuse the planned capacity kernels")
        graph = torch.cuda.CUDAGraph()
        try:
            binding = bind(rows)
            with torch.cuda.graph(graph):
                output = run(binding=binding)
            assert output.data_ptr() == binding.output.data_ptr()
            assert output.untyped_storage().data_ptr() == scratch[0].untyped_storage().data_ptr()
            address = output.data_ptr()
            allocated = torch.cuda.memory_allocated(device)
            for _ in range(3):
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                assert output.data_ptr() == address and torch.cuda.memory_allocated(device) == allocated
                assert torch.isfinite(output).all() and torch.count_nonzero(output)
                torch.testing.assert_close(output, expected, rtol=.02, atol=.02)
        finally:
            unfreeze_kernel_resolution()
            graph.reset()
