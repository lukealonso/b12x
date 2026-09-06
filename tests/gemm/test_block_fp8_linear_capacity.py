from __future__ import annotations

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import block_fp8_linear as bfl
from tests._reference.helpers import require_b12x
from tests.gemm.test_gemm_block_fp8_linear import (
    _make_block_fp8_weight,
    _reference_from_quantized_operands,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_fp8_linear_scratch_reuse_across_row_tiles(dtype: torch.dtype) -> None:
    require_b12x()
    capacity, in_features, out_features = 129, 256, 384
    source = torch.randn((capacity, in_features), device="cuda", dtype=dtype).mul_(0.25)
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = bfl.pack_weight(weight, scale)
    plan = bfl.plan(bfl.Caps(device=source.device, max_tokens=capacity,
                           in_features=in_features, out_features=out_features,
                           output_dtype=dtype))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty((capacity, out_features, 1), dtype=dtype, device=source.device)
    pointers = (source.data_ptr(), scratch.data_ptr(), output.data_ptr())

    def bind(tokens: int):
        return bfl.bind(plan, scratch=scratch, source=source[:tokens],
                        packed_weight=packed, output=output[:tokens])

    for tokens in (capacity, 8):
        bfl.run(binding=bind(tokens))
    freeze_kernel_resolution("block FP8 scratch reuse across row-tile boundaries")
    try:
        for tokens in (129, 9, 128, 8, 127):
            binding = bind(tokens)
            bfl.run(binding=binding)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bfl.run(binding=binding)
            source.normal_().mul_(0.25)
            expected = _reference_from_quantized_operands(source[:tokens], weight, scale)
            binding.x_q.scale_rows.view(torch.uint8).fill_(127)
            binding.x_q.scale_mma.view(torch.uint8).fill_(127)
            initialized = bfl.run(binding=binding).clone()
            scratch.fill_(255)
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                assert before[key] == after[key]
            assert pointers == (source.data_ptr(), scratch.data_ptr(), output.data_ptr())
            actual = output[:tokens, :, 0]
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            torch.testing.assert_close(actual, initialized, rtol=0, atol=0)
            # Distinct accumulation orders may round to adjacent output values.
            lower = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
            upper = torch.nextafter(expected, torch.full_like(expected, float("inf")))
            assert torch.all((actual >= lower) & (actual <= upper))
    finally:
        unfreeze_kernel_resolution()
