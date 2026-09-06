"""Nine-rank kernel arithmetic on one GPU; this does not qualify PCIe links."""

import os

import pytest
import torch

from b12x.comm.pcie.pcie_oneshot import _CuTeOneshotBackend


@pytest.mark.skipif(
    os.getenv("B12X_PCIE_TEST_NINE_RANK_MATH") != "1",
    reason="set B12X_PCIE_TEST_NINE_RANK_MATH=1 for the single-GPU kernel test",
)
def test_nine_rank_registered_protocol_and_graphs():
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
    streams = [torch.cuda.Stream() for _ in range(9)]
    handles = []
    for rank in range(9):
        handle = backend.init_custom_ar(
            [item.data_ptr() for item in signals], tables[rank], rank
        )
        backend.register_buffer(handle, [item.data_ptr() for item in inputs])
        backend.prepare_all_reduce(handle, inputs[rank])
        handles.append(handle)
    torch.cuda.synchronize()

    from b12x.comm.pcie._oneshot_cute import _SELF_COUNTER_BYTES

    def reset_signals(arrived=False):
        for signal in signals:
            signal.zero_()
            if arrived:
                signal.view(torch.int32)[_SELF_COUNTER_BYTES // 4 :].fill_(1)

    # Warm execution as well as compilation before any rank waits for peers.
    for rank in range(9):
        reset_signals(arrived=True)
        backend.all_reduce(handles[rank], inputs[rank], outputs[rank], 0, 0)
        torch.cuda.synchronize()
    reset_signals()
    torch.cuda.synchronize()

    def launch(rank):
        with torch.cuda.stream(streams[rank]):
            backend.all_reduce(handles[rank], inputs[rank], outputs[rank], 0, 0)

    for rank in range(9):
        launch(rank)
    torch.cuda.synchronize()
    expected = torch.stack(inputs).float().sum(0).to(torch.bfloat16)
    for output in outputs:
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    graphs = [torch.cuda.CUDAGraph() for _ in range(9)]
    for rank in range(9):
        with torch.cuda.graph(graphs[rank], stream=streams[rank]):
            backend.all_reduce(handles[rank], inputs[rank], outputs[rank], 0, 0)
        reset_signals(arrived=True)
        torch.cuda.synchronize()
        graphs[rank].replay()
        torch.cuda.synchronize()
    reset_signals()
    torch.cuda.synchronize()
    for iteration in range(4):
        inputs[8].fill_(iteration + 1)
        torch.cuda.synchronize()
        for rank in range(9):
            with torch.cuda.stream(streams[rank]):
                graphs[rank].replay()
        torch.cuda.synchronize()
        expected = torch.stack(inputs).float().sum(0).to(torch.bfloat16)
        for output in outputs:
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
