"""Full-rotation MoE keeps one fused callable per declared row capacity."""

import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.w4a16 import kernel
from tests.conftest import require_b12x
from tests.moe.test_fused_moe_trellis import _prepare_weights, _plan, _reference_full_rotation


def test_full_rotation_reuses_capacity_callable_for_live_rows(monkeypatch):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(57321)
    experts, hidden, intermediate, bits, topk, capacity = 2, 128, 128, 3, 2, 8
    w13 = torch.randint(-32768, 32767, (2, experts, hidden//16, intermediate//16, 16*bits), dtype=torch.int16, device=device)
    w2 = torch.randint(-32768, 32767, (experts, intermediate//16, hidden//16, 16*bits), dtype=torch.int16, device=device)
    def scales(*shape):
        return (.875 + .25*torch.rand(shape, device=device)).half()
    gate, up = scales(experts, hidden), scales(experts, hidden)
    inter, down = scales(experts, 3*intermediate), scales(experts, hidden)
    weights = _prepare_weights(w13, w2, gate_suh=gate, up_suh=up, intermediate_rotations=inter,
        down_svh=down, codebook="mcg", tile_config=(64, 128, 64, 128))
    plan = _plan(weights, max_tokens=capacity, num_topk=topk, route_num_experts=experts, block_size_m=8, device=device)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    x = (torch.randn(capacity, hidden, device=device)*1e-3).bfloat16()
    ids = torch.tensor([[0, 1]], device=device, dtype=torch.int32).expand(capacity, -1).contiguous()
    routes = torch.softmax(torch.randn(capacity, topk, device=device), dim=-1)
    output = torch.empty(capacity, hidden, dtype=torch.float32, device=device)
    compiled = kernel.compile_w4a16_fused_moe
    resolved = []

    def observe(**kwargs):
        result = compiled(**kwargs)
        resolved.append((kwargs["size_m"], id(result.compiled)))
        return result

    monkeypatch.setattr(kernel, "compile_w4a16_fused_moe", observe)
    for rows in (capacity, 1, 3, 7):
        expected = _reference_full_rotation(x[:rows], ids[:rows], routes[:rows], w13, w2, gate, up, inter, down)
        freeze_kernel_resolution("full-rotation live rows must reuse their capacity callable")
        graph = torch.cuda.CUDAGraph()
        try:
            binding = fused_moe.bind(plan, scratch=scratch, a=x[:rows], experts=weights,
                topk_weights=routes[:rows], topk_ids=ids[:rows], output=output[:rows])
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            allocated = torch.cuda.memory_allocated(device)
            for _ in range(3):
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                assert torch.isnan(output[rows:]).all()
                assert torch.isfinite(output[:rows]).all() and torch.count_nonzero(output[:rows])
                relative = float((output[:rows]-expected).norm() / expected.norm())
                assert relative < .02
            assert {count for count, _ in resolved} == {capacity}
            assert len({callable_id for _, callable_id in resolved}) == 1
        finally:
            unfreeze_kernel_resolution()
            graph.reset()
