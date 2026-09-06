"""Reference-backend-free helpers shared by offline dense GEMM autotuners."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from b12x._lib.utils import convert_sf_to_mma_layout


def bench_events(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    l2_flush: Callable[[], None] | None = None,
) -> list[float]:
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        if l2_flush is not None:
            l2_flush()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def capture_graph_replay(fn: Callable[[], None]) -> Callable[[], None]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(graph: torch.cuda.CUDAGraph = graph) -> None:
        graph.replay()

    return replay


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.to(torch.float32).reshape(-1),
        b.to(torch.float32).reshape(-1),
        dim=0,
    ).item()


def make_nvfp4_operand(rows: int, k: int) -> tuple[torch.Tensor, ...]:
    source = torch.randn(1, rows, k, device="cuda", dtype=torch.bfloat16) / 4
    row_counts = torch.full((1,), rows, dtype=torch.int32, device="cuda")
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device="cuda",
    )
    packed, scales = quantize_grouped_nvfp4_torch(
        source, row_counts, global_scale
    )
    return packed, scales, global_scale


def make_mxfp8_operand(rows: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build valid prequantized MXFP8 values/scales without a reference backend."""
    values = (
        torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) / 4
    ).to(torch.float8_e4m3fn)
    m_tiles = (rows + 127) // 128
    k_tiles = ((k + 31) // 32 + 3) // 4
    scale_storage = torch.full(
        (m_tiles * k_tiles * 32 * 4 * 4,),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)
    scale_mma = convert_sf_to_mma_layout(
        scale_storage,
        m=rows,
        k=k,
        num_groups=1,
        sf_vec_size=32,
    )
    return values.contiguous(), scale_mma
