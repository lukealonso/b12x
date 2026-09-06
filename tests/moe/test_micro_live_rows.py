"""NVFP4 micro launch decisions depend on planned capacity, not live rows."""

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl
from tests.moe.test_cute_migration_moe_standard_corpus import (
    _assert_oracle,
    _Inputs,
    _make_inputs,
    _make_nvfp4_weights,
    _nvfp4_oracle,
    _prepare_and_bind,
    _reset_dispatch_environment,
    require_b12x,
)


@pytest.mark.parametrize("capacity", [2, 4, 8])
def test_nvfp4_micro_reuses_capacity_callable_for_live_rows(monkeypatch, capacity):
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=101)
    inputs = _make_inputs(device, m=capacity, seed=103, route_shift=2)
    references = {
        rows: _nvfp4_oracle(weights, _Inputs(
            inputs.a[:rows], inputs.topk_ids[:rows], inputs.topk_weights[:rows],
        ), quant_scale_math="reciprocal_multiply")
        for rows in (capacity, 1, capacity - 1, 2)
    }
    case = _prepare_and_bind(weights, inputs, quant_mode="nvfp4", source_format="modelopt_nvfp4")
    assert case.binding.implementation == "micro"
    fused_moe.run(binding=case.binding)
    torch.cuda.synchronize(device)
    compiled = {key: id(value) for key, value in _impl._MICRO_KERNEL_CACHE.items()}
    assert compiled
    output = case.binding.output
    addresses = (inputs.a.data_ptr(), output.data_ptr(), *(t.data_ptr() for t in case.scratch))
    for rows, expected in references.items():
        freeze_kernel_resolution("NVFP4 micro must reuse capacity callables across live rows")
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
                _assert_oracle(output[:rows], expected, context=f"micro-capacity-{capacity}-rows-{rows}",
                               min_cos=0.999, max_normalized_rmse=0.03)
                assert torch.isnan(output[rows:]).all()
                assert addresses == (inputs.a.data_ptr(), output.data_ptr(), *(t.data_ptr() for t in case.scratch))
                assert compiled == {key: id(value) for key, value in _impl._MICRO_KERNEL_CACHE.items()}
        finally:
            unfreeze_kernel_resolution()
            graph.reset()
