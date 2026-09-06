"""Tiny-decode phases reuse their compiled callables across live row counts."""

import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl
from tests.moe.test_cute_migration_moe_standard_corpus import (
    _assert_oracle,
    _Inputs,
    _make_inputs,
    _make_mxfp4_weights,
    _mxfp4_oracle,
    _prepare_and_bind,
    _reset_dispatch_environment,
    require_b12x,
)


def test_tiny_decode_reuses_capacity_callables_for_live_rows(monkeypatch):
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    capacity = 4
    weights = _make_mxfp4_weights(device, seed=301)
    inputs = _make_inputs(device, m=capacity, seed=303, route_shift=2)
    inputs.topk_ids[1, 1] = -1
    references = {
        rows: _mxfp4_oracle(weights, _Inputs(
            inputs.a[:rows], inputs.topk_ids[:rows], inputs.topk_weights[:rows],
        ))
        for rows in (capacity, 1, 3, 2)
    }
    case = _prepare_and_bind(weights, inputs, quant_mode="w4a8_mx", source_format="fp4_e8m0_k32")
    assert case.binding.implementation == "micro"
    fused_moe.run(binding=case.binding)
    torch.cuda.synchronize(device)
    compiled = {key: id(value) for key, value in _impl._TINY_DECODE_KERNEL_CACHE.items()}
    assert compiled
    output = case.binding.output
    addresses = (inputs.a.data_ptr(), output.data_ptr(), *(t.data_ptr() for t in case.scratch))
    for rows, expected in references.items():
        freeze_kernel_resolution("tiny decode must reuse capacity callables across live rows")
        graph = torch.cuda.CUDAGraph()
        try:
            binding = case.scratch_plan.bind(
                scratch=case.scratch, a=inputs.a[:rows], experts=case.experts,
                topk_weights=inputs.topk_weights[:rows], topk_ids=inputs.topk_ids[:rows],
                output=output[:rows], input_scales_static=True, fast_math=False,
            )
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            allocated = torch.cuda.memory_allocated(device)
            for _ in range(3):
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                _assert_oracle(output[:rows], expected, context=f"tiny-live-rows-{rows}",
                               min_cos=0.998, max_normalized_rmse=0.05)
                assert torch.isnan(output[rows:]).all()
                assert addresses == (inputs.a.data_ptr(), output.data_ptr(), *(t.data_ptr() for t in case.scratch))
                assert compiled == {key: id(value) for key, value in _impl._TINY_DECODE_KERNEL_CACHE.items()}
        finally:
            unfreeze_kernel_resolution()
            graph.reset()
