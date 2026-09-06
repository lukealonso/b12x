from __future__ import annotations

import os
import socket
from math import gcd

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce, _load_kernels


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DMA_TEST") != "1",
    reason="set B12X_RUN_PCIE_DMA_TEST=1 to run PCIe ring allreduce GPU tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_input(
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    iteration: int,
) -> torch.Tensor:
    values = torch.arange(rows * hidden, device=device, dtype=torch.float32)
    values = values.reshape(rows, hidden)
    return (torch.sin(values * 0.001 + rank * 0.31 + iteration * 0.17) * 0.5).to(dtype)


def _reference(inp: torch.Tensor) -> torch.Tensor:
    # clone: .float() aliases the input for fp32 and all_reduce is in-place.
    ref = inp.detach().clone().float()
    dist.all_reduce(ref)
    return ref


def _assert_close(actual: torch.Tensor, ref: torch.Tensor, world_size: int) -> None:
    # Stepwise low-precision ring adds; allow world_size half-ulps around the
    # fp32 reference. Compressed wire modes need a wider band because the ring
    # can requantize partial sums at each reduce-scatter hop.
    if os.getenv("B12X_PCIE_DMA_FP8", "0") not in ("", "0"):
        torch.testing.assert_close(
            actual.float(), ref, rtol=1.5e-1, atol=6e-2 * world_size
        )
    elif actual.dtype == torch.float32:
        torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-5 * world_size)
    else:
        torch.testing.assert_close(
            actual.float(), ref, rtol=3e-2, atol=3e-2 * world_size
        )


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    hidden = 6144
    max_rows = 512
    row_multiple = (world_size * 8) // gcd(hidden, world_size * 8)

    def valid_rows(rows: int) -> int:
        return ((rows + row_multiple - 1) // row_multiple) * row_multiple

    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_bytes=max_rows * hidden * 4,
    )
    try:
        explicit_inp = _make_input(
            valid_rows(8), hidden, torch.bfloat16, device, rank, 0
        )
        explicit_ref = _reference(explicit_inp)
        explicit_out = torch.empty_like(explicit_inp)
        with pytest.raises(ValueError, match="device"):
            ring.all_reduce(explicit_inp, out=torch.empty_like(explicit_inp, device="cpu"))
        with pytest.raises(ValueError, match="device"):
            ring.all_reduce(
                explicit_inp,
                out=torch.empty_like(explicit_inp, device=(rank + 1) % world_size),
            )
        wrong_device = (rank + 1) % world_size
        torch.cuda.set_device(wrong_device)
        ring.all_reduce(explicit_inp, out=explicit_out)
        assert torch.cuda.current_device() == wrong_device
        torch.cuda.set_device(rank)
        torch.cuda.synchronize(device)
        _assert_close(explicit_out, explicit_ref, world_size)

        retained_outputs: list[torch.Tensor] = []
        retained_refs: list[torch.Tensor] = []
        for dtype in (torch.bfloat16,):
            for requested_rows in (8, 64, 256):
                rows = valid_rows(requested_rows)
                inp = _make_input(rows, hidden, dtype, device, rank, 0)
                ref = _reference(inp)
                out = ring.all_reduce(inp)
                torch.cuda.synchronize(device)
                _assert_close(out, ref, world_size)
                assert all(
                    out.data_ptr() != prior.data_ptr() for prior in retained_outputs
                )
                retained_outputs.append(out)
                retained_refs.append(ref)
                for retained, retained_ref in zip(
                    retained_outputs, retained_refs, strict=True
                ):
                    _assert_close(retained, retained_ref, world_size)

        # Captured callers provide stable output storage explicitly.
        rows = valid_rows(256)
        dtype = torch.bfloat16
        inp = _make_input(rows, hidden, dtype, device, rank, 0)
        out = torch.empty_like(inp)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ring.all_reduce(inp, out=out)
        for iteration in range(1, 4):
            inp.copy_(_make_input(rows, hidden, dtype, device, rank, iteration))
            ref = _reference(inp)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_close(out, ref, world_size)

        dist.barrier()
    finally:
        ring.close()
        dist.destroy_process_group()


def test_pcie_dma_all_reduce_eager_and_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("B12X_PCIE_DMA_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    _load_kernels()
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


def _replay_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY"] = "1"
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY_MIN_BYTES"] = "0"
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY_MAX_ENTRIES"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    hidden = 6144
    max_rows = 512
    row_multiple = (world_size * 8) // gcd(hidden, world_size * 8)
    rows_a = ((256 + row_multiple - 1) // row_multiple) * row_multiple
    rows_b = ((64 + row_multiple - 1) // row_multiple) * row_multiple
    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_bytes=max_rows * hidden * 4,
    )
    try:
        assert ring._graph_replay

        def eager(inp: torch.Tensor) -> torch.Tensor:
            ring._graph_replay = False
            try:
                out = ring.all_reduce(inp)
            finally:
                ring._graph_replay = True
            torch.cuda.synchronize(device)
            return out

        inputs = [
            _make_input(rows_a, hidden, torch.bfloat16, device, rank, it)
            for it in range(4)
        ]
        expected = [eager(inp) for inp in inputs]

        # First call of a shape stays eager, the second captures and replays,
        # later ones replay; all bit-identical to the eager sequence.
        assert not ring._replay_entries
        out0 = ring.all_reduce(inputs[0])
        torch.cuda.synchronize(device)
        assert not ring._replay_entries
        assert torch.equal(out0, expected[0])
        allocated_before_capture = torch.cuda.memory_allocated(device)
        out1 = ring.all_reduce(inputs[1])
        torch.cuda.synchronize(device)
        # The capture carves its static buffers out of the arena reserved at
        # construction, so the allocator grows by the returned output plus at
        # most the small launch bookkeeping of the captured kernels, never by
        # another pair of tensor-sized buffers.
        capture_growth = torch.cuda.memory_allocated(device) - allocated_before_capture
        assert capture_growth <= out1.numel() * out1.element_size() + (4 << 20), (
            f"capture allocated {capture_growth} bytes beyond the output"
        )
        assert len(ring._replay_entries) == 1
        entry = next(iter(ring._replay_entries.values()))
        arena_start = ring._replay_arena.data_ptr()
        arena_end = arena_start + ring._replay_arena.numel()
        assert arena_start <= entry.inp.data_ptr() < arena_end
        assert arena_start <= entry.out.data_ptr() < arena_end
        assert torch.equal(out1, expected[1])
        assert out1.data_ptr() not in (entry.inp.data_ptr(), entry.out.data_ptr())
        out2 = ring.all_reduce(inputs[2])
        out3 = ring.all_reduce(inputs[3])
        torch.cuda.synchronize(device)
        assert torch.equal(out2, expected[2])
        assert torch.equal(out3, expected[3])
        # Retained outputs are independent of later replays.
        assert torch.equal(out1, expected[1])
        _assert_close(out3, _reference(inputs[3]), world_size)

        # A tensor of another shape but the same element count reuses the
        # entry (the ring is a flat all-reduce) and matches the eager result.
        reshaped = inputs[2].view(rows_a * 2, hidden // 2)
        out_reshaped = ring.all_reduce(reshaped)
        torch.cuda.synchronize(device)
        assert len(ring._replay_entries) == 1
        assert out_reshaped.shape == reshaped.shape
        assert torch.equal(out_reshaped.view(rows_a, hidden), expected[2])

        # A second shape evicts the first at max_entries=1 and still matches.
        small = [
            _make_input(rows_b, hidden, torch.bfloat16, device, rank, it)
            for it in range(2)
        ]
        small_expected = [eager(inp) for inp in small]
        ring.all_reduce(small[0])
        out_small = ring.all_reduce(small[1])
        torch.cuda.synchronize(device)
        assert list(ring._replay_entries) == [(small[1].numel(), torch.bfloat16)]
        assert torch.equal(out_small, small_expected[1])

        # Inside an enclosing capture the eager sequence is recorded as before.
        static_in = inputs[0].clone()
        static_out = torch.empty_like(static_in)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ring.all_reduce(static_in, out=static_out)
        for it in range(1, 3):
            static_in.copy_(inputs[it])
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(static_out, expected[it])
        dist.barrier()
    finally:
        ring.close()
        dist.destroy_process_group()


def test_pcie_dma_all_reduce_graph_replay_matches_eager() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("B12X_PCIE_DMA_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    _load_kernels()
    mp.spawn(
        _replay_worker, args=(world_size, _free_port()), nprocs=world_size, join=True
    )


def _fp8_worker(rank: int, world_size: int, port: int, mode: str) -> None:
    os.environ["B12X_PCIE_DMA_FP8"] = mode
    _worker(rank, world_size, port)


@pytest.mark.parametrize(
    "mode",
    [
        "ag",
        "a2a",
        "ring",
        "i8",
        "i8_a2a",
        "i8_ring",
        "mx",
        "mx_a2a",
        "mx_ring",
    ],
)
def test_pcie_dma_all_reduce_compressed_wire(mode: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("B12X_PCIE_DMA_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    _load_kernels()
    mp.spawn(
        _fp8_worker,
        args=(world_size, _free_port(), mode),
        nprocs=world_size,
        join=True,
    )


def test_pcie_dma_mxfp8_codec_matches_cpu_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda:0")
    elems = 4 * 128
    source = (
        torch.sin(torch.arange(elems, device=device, dtype=torch.float32) * 0.07)
        * torch.linspace(0.01, 900.0, elems, device=device)
    ).to(torch.bfloat16)
    source[17] = 1200.0
    source[131] = -0.0002
    storage = torch.empty(elems + elems // 32, dtype=torch.uint8, device=device)
    output = torch.empty_like(source)

    kernels = _load_kernels()
    kernels.dma_quant_mx(
        source.data_ptr(), storage.data_ptr(), storage.data_ptr() + elems, elems
    )
    kernels.dma_dequant_store_mx(
        output.data_ptr(), storage.data_ptr(), storage.data_ptr() + elems, elems
    )
    torch.cuda.synchronize(device)

    groups = source.float().cpu().reshape(-1, 32)
    amax = groups.abs().amax(dim=1)
    exponents = torch.where(
        amax > 0,
        torch.ceil(torch.log2(amax / 448.0)),
        torch.zeros_like(amax),
    ).clamp(-127, 127)
    expected_scales = (exponents + 127).to(torch.uint8)
    scale_values = torch.pow(2.0, exponents)
    expected_payload = (groups / scale_values[:, None]).to(torch.float8_e4m3fn)
    expected_output = (expected_payload.float() * scale_values[:, None]).reshape(-1)

    actual_payload = storage[:elems].cpu()
    actual_scales = storage[elems:].cpu()
    assert torch.equal(actual_scales, expected_scales)
    assert torch.equal(actual_payload, expected_payload.reshape(-1).view(torch.uint8))
    assert torch.equal(output.cpu(), expected_output.to(torch.bfloat16))


def test_pcie_dma_mxfp8_quantize_is_exact_for_every_bf16_bit_pattern() -> None:
    """Lock the CUDA E8M0/SATFINITE edge semantics without native code."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda:0")
    bit_patterns = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16)
    source_values = bit_patterns.view(torch.bfloat16)
    source = source_values.to(device).repeat_interleave(32)
    payload = torch.empty(source.numel(), dtype=torch.uint8, device=device)
    scales = torch.empty(1 << 16, dtype=torch.uint8, device=device)

    kernels = _load_kernels()
    kernels.dma_quant_mx(
        source.data_ptr(), payload.data_ptr(), scales.data_ptr(), source.numel()
    )
    torch.cuda.synchronize(device)

    # __nv_cvt_float_to_e8m0(amax / 448, SATFINITE, round-positive-inf).
    # For a finite normal BF16 value, division by 448 (= 1.75 * 2**8)
    # increments the resulting exponent exactly when its significand exceeds
    # 1.75. E8M0 reserves byte 0 for 2**-127 and saturates infinity to 254.
    absolute_bits = bit_patterns.to(torch.int32) & 0x7FFF
    exponent = absolute_bits >> 7
    fraction = absolute_bits & 0x7F
    expected_scales = torch.clamp(
        exponent - 8 + (fraction > 96).to(torch.int32), min=0
    )
    expected_scales = torch.where(
        absolute_bits == 0, torch.full_like(expected_scales, 127), expected_scales
    )
    expected_scales = torch.where(
        exponent == 255,
        torch.where(
            fraction == 0,
            torch.full_like(expected_scales, 254),
            torch.full_like(expected_scales, 255),
        ),
        expected_scales,
    ).to(torch.uint8)

    scale_exponents = expected_scales.to(torch.int32) - 127
    scale_values = torch.pow(2.0, scale_exponents.to(torch.float32))
    scale_values = torch.where(
        expected_scales == 0, torch.tensor(2.0**-127), scale_values
    )
    scale_values = torch.where(
        expected_scales == 255, torch.tensor(float("nan")), scale_values
    )
    expected_payload = (
        source_values.float() / scale_values
    ).to(torch.float8_e4m3fn).view(torch.uint8)
    is_inf = (exponent == 255) & (fraction == 0)
    is_nan = (exponent == 255) & (fraction != 0)
    expected_payload = torch.where(
        is_inf,
        torch.where(
            (bit_patterns.to(torch.int32) & 0x8000) != 0,
            torch.full_like(expected_payload, 0xFE),
            torch.full_like(expected_payload, 0x7E),
        ),
        expected_payload,
    )
    expected_payload = torch.where(
        is_nan, torch.full_like(expected_payload, 0x7F), expected_payload
    )

    assert torch.equal(scales.cpu(), expected_scales)
    assert torch.equal(
        payload.cpu(), expected_payload.repeat_interleave(32)
    )
