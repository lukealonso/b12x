"""Nine-rank arithmetic with pre-arrived peers; no physical link/barrier test.

The collective's inputs are resident before each launch and peer-arrival slots
are seeded to that state. This isolates rank-indexed loads and graph arithmetic
without requiring nine mutually waiting kernels to co-reside on one GPU.
"""

import os

import pytest
import torch

from b12x.comm.pcie.pcie_oneshot import _CuTeOneshotBackend
from b12x.comm.pcie._oneshot_cute import _SELF_COUNTER_BYTES


@pytest.mark.skipif(
    os.getenv("B12X_PCIE_TEST_NINE_RANK_MATH") != "1",
    reason="set B12X_PCIE_TEST_NINE_RANK_MATH=1 for the single-GPU kernel test",
)
def test_nine_rank_oneshot_eager_and_graph_include_every_peer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.cuda.set_device(0)
    backend = _CuTeOneshotBackend()
    signals = [
        torch.zeros(backend.meta_size(), dtype=torch.uint8, device="cuda")
        for _ in range(9)
    ]
    tables = [torch.empty(4096, dtype=torch.uint8, device="cuda") for _ in range(9)]
    inputs = [
        (torch.arange(1024, device="cuda") % 7 + rank - 7).to(torch.bfloat16)
        for rank in range(9)
    ]
    outputs = [torch.empty_like(item) for item in inputs]
    handles = []
    for rank in range(9):
        handle = backend.init_custom_ar(
            [item.data_ptr() for item in signals], tables[rank], rank
        )
        backend.register_buffer(handle, [item.data_ptr() for item in inputs])
        backend.prepare_all_reduce(handle, inputs[rank])
        handles.append(handle)
    torch.cuda.synchronize()

    def seed_arrived_peers():
        for signal in signals:
            words = signal.view(torch.int32)
            words.zero_()
            words[_SELF_COUNTER_BYTES // 4 :].fill_(1)

    def ordered_reference():
        result = inputs[0].float()
        for value in inputs[1:]:
            result.add_(value.float())
        return result.to(torch.bfloat16)

    for rank in range(9):
        seed_arrived_peers()
        backend.all_reduce(handles[rank], inputs[rank], outputs[rank], 0, 0)
        torch.cuda.synchronize()
        for owner, signal in enumerate(signals):
            expected_count = 1 if owner == rank else 0
            assert torch.equal(
                signal.view(torch.int32)[:9].cpu(),
                torch.full((9,), expected_count, dtype=torch.int32),
            )
        expected = ordered_reference()
        torch.testing.assert_close(outputs[rank], expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            backend.all_reduce(handles[rank], inputs[rank], outputs[rank], 0, 0)
        for iteration in range(4):
            inputs[8].fill_(iteration + 1)
            if iteration == 3:
                inputs[0].fill_(65536)
                inputs[1].fill_(-65536)
                for value in inputs[2:]:
                    value.fill_(2**-13)
            seed_arrived_peers()
            graph.replay()
            torch.cuda.synchronize()
            expected = ordered_reference()
            torch.testing.assert_close(outputs[rank], expected, atol=0, rtol=0)
