"""Child-graph timing preserves the production graph and reset ordering."""

import math

import pytest
import torch

from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from b12x.policy.generation.providers.gpu_workers import _l2_flush_fn
from b12x.policy.generation.timing import CapturedGraphTimer
from b12x.policy.generation.timestamps import GlobaltimerGraphRaceTimer, warm_globaltimer
from b12x.quantization import nvfp4
from tests.conftest import require_b12x


@pytest.mark.parametrize("clock", ["cuda_event", "globaltimer"])
def test_child_graph_replays_production_nvfp4_with_fixed_storage(clock):
    import b12x

    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(1435)
    source = torch.randn((128, 4096), device=device, dtype=torch.bfloat16, generator=generator) * .25
    scale = torch.tensor([.5], device=device)
    counts = torch.tensor([128], device=device, dtype=torch.int32)
    expected, scale_view = quantize_grouped_nvfp4_torch(source.unsqueeze(0), counts, scale)
    expected_scale = scale_view.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8).reshape(-1)
    plan = nvfp4.plan(128, 4096)
    outputs = nvfp4.allocate_outputs(plan, device=device)

    def run():
        nvfp4.run(plan=plan, x=source, global_scale=scale, outputs=outputs)

    def poison():
        outputs.packed_a_storage.zero_()
        outputs.scale_flat.zero_()

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    flush = _l2_flush_fn(device, enabled=True)
    if clock == "globaltimer":
        warm_globaltimer(device)
    b12x.freeze_kernel_resolution("NVFP4 child-graph timer qualification")
    try:
        with torch.cuda.graph(graph):
            run()
        graph.instantiate()
        addresses = (outputs.packed_a_storage.data_ptr(), outputs.scale_flat.data_ptr())
        timer = (CapturedGraphTimer(graph, count=7, device=device, flush=flush, before_each=poison)
                 if clock == "cuda_event" else GlobaltimerGraphRaceTimer(
                     {"first": graph, "second": graph}, count={"first": 7, "second": 11},
                     device=device, flush=flush, before_each=lambda _name: poison()))
        with timer:
            allocated = torch.cuda.memory_allocated(device)
            for _ in range(3):
                timer.replay()
                samples = timer.samples_us()
                if clock == "globaltimer":
                    assert {name: len(values) for name, values in samples.items()} == {"first": 7, "second": 11}
                    samples = (*samples["first"], *samples["second"])
                else:
                    assert len(samples) == 7
                assert all(math.isfinite(value) and value > 0 for value in samples)
                assert torch.cuda.memory_allocated(device) == allocated
                assert torch.equal(outputs.packed_a_storage.permute(1, 2, 0), expected)
                assert torch.equal(outputs.scale_flat, expected_scale)
                assert addresses == (outputs.packed_a_storage.data_ptr(), outputs.scale_flat.data_ptr())
        poison()
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(outputs.packed_a_storage.permute(1, 2, 0), expected)
        assert torch.equal(outputs.scale_flat, expected_scale)
    finally:
        b12x.unfreeze_kernel_resolution()
        graph.reset()
