"""Qualify TP9 collectives on nine physical GPUs with Kimi-K3 tensor shapes."""

from __future__ import annotations

import json
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, port: int) -> None:
    from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A
    from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce
    from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool
    from b12x.comm.pcie.pcie_island9 import PCIeIsland9AllReduce
    from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=9,
        timeout=timedelta(seconds=240),
        device_id=device,
    )
    group = dist.group.WORLD

    def record(stage: str) -> None:
        torch.cuda.synchronize(device)
        dist.barrier()
        if rank == 0:
            print(json.dumps({"stage": stage, "world_size": 9}), flush=True)

    def check(tensor: torch.Tensor, expected: float) -> None:
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            tensor, torch.full_like(tensor, expected), rtol=0, atol=0
        )

    record("process_group_ready")
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=group,
        device=device,
        max_input_bytes=1 << 20,
        max_concurrent_channels=2,
    )
    pool.prepare_channels(("eager", "graph"))
    inp = torch.full((4, 7168), rank + 1, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(inp)
    pool.all_reduce(inp, out=out, channel_id="eager")
    check(out, 45)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        pool.all_reduce(inp, out=out, channel_id="graph")
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        pool.capture(stream, channel_id="graph"),
        torch.cuda.graph(graph, stream=stream),
    ):
        pool.all_reduce(inp, out=out, channel_id="graph")
    for iteration in range(3):
        inp.fill_(rank + 1 + (iteration if rank == 8 else 0))
        graph.replay()
        check(out, 45 + iteration)
    del graph
    pool.close()
    record("oneshot_eager_graph_passed")

    def position_pattern(rows: int, width: int) -> torch.Tensor:
        # Position-dependent integers keep every partial sum exact in bf16
        # (at most 5 * 45 = 225) and expose misplaced or dropped shards,
        # which uniform fills cannot.
        return (
            torch.arange(rows * width, device=device).remainder(5) + 1
        ).to(torch.bfloat16).view(rows, width)

    def check_pattern(tensor: torch.Tensor, scale: float) -> None:
        torch.cuda.synchronize(device)
        expected = position_pattern(*tensor.shape) * scale
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

    # Kimi-K3 decode payloads: 7168 wide rows (T tokens) and the 3584 wide
    # latent projection. Their pack counts are multiples of nine only when
    # 9 | T, so most shapes exercise the balanced (uneven) shard partition.
    # The pull (remote read) and push (posted write) kernels share it.
    for mode in ("pull", "push"):
        twoshot = PCIeTwoShotBF16.from_exchange_group(
            exchange_group=group,
            device=device,
            max_rows=49149,
            row_elems=8,
        )
        twoshot.all_reduce_mode = mode
        for rows, width in ((4, 7168), (8, 7168), (9, 7168), (16, 7168), (32, 7168),
                            (4, 3584), (16, 3584)):
            inp = position_pattern(rows, width) * (rank + 1)
            out = torch.empty_like(inp)
            assert twoshot.accepts(inp)
            twoshot.all_reduce(inp, out=out)
            check_pattern(out, 45)
        inp = position_pattern(16, 7168) * (rank + 1)
        out = torch.empty_like(inp)
        graph = torch.cuda.CUDAGraph()
        with twoshot.capture(), torch.cuda.graph(graph):
            twoshot.all_reduce(inp, out=out)
        for iteration in range(3):
            inp.copy_(
                position_pattern(16, 7168) * (rank + 1 + (iteration if rank == 8 else 0))
            )
            graph.replay()
            check_pattern(out, 45 + iteration)
        del graph
        twoshot.close()
        record(f"twoshot_{mode}_balanced_partition_eager_graph_passed")

    # Two-island push all-reduce with rank 8 contributing through island 0.
    # Quarter boundaries are not divisible by the launch stride for most of
    # these shapes, and the odd widths leave short last quarters.
    island9 = PCIeIsland9AllReduce.from_exchange_group(
        exchange_group=group,
        device=device,
        max_rows=49149,
        row_elems=8,
    )
    assert island9.mapped_peers == (
        (0, 1, 2, 3) if rank == 8 else tuple(
            sorted(({(rank // 4) * 4 + p for p in range(4)} - {rank})
                   | {((1 - rank // 4) * 4 + rank % 4)}
                   | ({8} if rank < 4 else set()))
        )
    )
    for rows, width in ((1, 7168), (2, 7168), (4, 7168), (8, 7168), (9, 7168),
                        (16, 7168), (32, 7168), (4, 3584), (16, 3584), (1, 1030)):
        inp = position_pattern(rows, width) * (rank + 1)
        out = torch.empty_like(inp)
        assert island9.accepts(inp)
        island9.all_reduce(inp, out=out)
        check_pattern(out, 45)
    # The island-0 partial (4 x 64 + rank 8's 1 = 257) is not representable
    # in bf16; only fp32 partials give the representable total 258.
    value = {8: 1.0, 4: 1.0}.get(rank, 64.0 if rank < 4 else 0.0)
    inp = torch.full((4, 7168), value, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(inp)
    island9.all_reduce(inp, out=out)
    check(out, 258.0)
    inp = position_pattern(16, 7168) * (rank + 1)
    out = torch.empty_like(inp)
    graph = torch.cuda.CUDAGraph()
    with island9.capture(), torch.cuda.graph(graph):
        island9.all_reduce(inp, out=out)
    for iteration in range(3):
        inp.copy_(
            position_pattern(16, 7168) * (rank + 1 + (iteration if rank == 8 else 0))
        )
        graph.replay()
        check_pattern(out, 45 + iteration)
    del graph
    island9.close()
    record("island9_push_eager_graph_passed")

    dma = PCIeDmaAllReduce(
        exchange_group=group,
        device=device,
        max_bytes=32 << 20,
        fp8="0",
    )
    inp = torch.full((1536, 7168), rank + 1, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(inp)
    assert dma.should_allreduce(inp)
    retained = dma.all_reduce(inp)
    check(retained, 45)
    for iteration in range(3):
        inp.fill_(rank + 1 + (iteration if rank == 8 else 0))
        dma.all_reduce(inp, out=out)
        check(out, 45 + iteration)
        check(retained, 45)
    if hasattr(dma, "_replay_entries"):
        # The serving lineage caches wire-tail copies for graph replay; master
        # replays the ring without host patching and keeps no such table.
        assert dma._replay_entries, "DMA graph replay must include the wire-tail copies"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dma.all_reduce(inp, out=out)
    for iteration in range(3):
        inp.fill_(rank + 1 + (iteration if rank == 8 else 0))
        graph.replay()
        check(out, 45 + iteration)
    del graph
    dma.close()
    record("dma_wire_tail_eager_cached_graph_external_graph_passed")

    dcp = PCIeDCPA2A.from_exchange_group(
        exchange_group=group,
        device=device,
        max_batch_size=4,
        total_heads=99,
        head_dim=512,
        query_head_dim=576,
        stream_affine=False,
    )
    q = torch.full((2, 11, 576), rank + 1, dtype=torch.bfloat16, device=device)
    gathered = dcp.all_gather_heads(q)
    for source in range(9):
        check(gathered[:, source * 11 : (source + 1) * 11], source + 1)
    padded = torch.full((2, 104, 512), rank + 1, dtype=torch.bfloat16, device=device)
    partial = padded[:, :99]
    lse = torch.zeros((2, 99), dtype=torch.float32, device=device)
    reduced = torch.empty((2, 11, 512), dtype=torch.bfloat16, device=device)
    dcp.lse_reduce_scatter(partial, lse, out=reduced)
    check(reduced, 5)
    dcp.prepare_graph_lse_reduce_scatter()
    dcp.prepare_graph_all_gather_heads()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dcp.all_gather_heads(q, out=gathered)
        dcp.lse_reduce_scatter(partial, lse, out=reduced)
    for iteration in range(3):
        partial.fill_(rank + 1 + iteration)
        graph.replay()
        check(reduced, 5 + iteration)
    if rank == 8:
        partial.fill_(float("nan"))
        lse.fill_(float("-inf"))
    else:
        partial.fill_(rank + 1)
    graph.replay()
    check(reduced, 4.5)
    del graph
    dcp.close()
    record("dcp_gather_lse_head_tail_and_empty_rank_graph_passed")
    dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_TP9_TEST") != "1",
    reason="requires nine idle GPUs and B12X_RUN_PCIE_TP9_TEST=1",
)
def test_tp9_physical_collectives() -> None:
    assert torch.cuda.device_count() == 9
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY"] = "1"
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY_MAX_ENTRIES"] = "1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=9, join=True)


if __name__ == "__main__":
    test_tp9_physical_collectives()
