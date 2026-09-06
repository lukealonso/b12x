"""Per-call latency of the PCIe all-reduce paths on nine physical GPUs.

Times the B12X one-shot pool, the BF16 two-shot runtime and the PyNCCL
all-reduce for Kimi-K3 decode payloads (``T`` rows of 7168 or 3584 bf16
elements) on the full nine-rank exchange group, eager and inside a replayed
CUDA graph. Every path is checked bit-exactly against an fp32 reference before
it is timed. Rank 0 prints one JSON line per (path, shape, mode) with the
median and p90 microseconds over ``--iters`` synchronized calls.

Run with nine idle GPUs::

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,0 python benchmarks/benchmark_pcie_tp9_allreduce.py
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEFAULT_SHAPES = "1x1024,1x4096,1x7168,2x7168,4x7168,8x7168,16x7168,32x7168,4x3584,8x3584,16x3584"


def _shapes(raw: str) -> tuple[tuple[int, int], ...]:
    shapes = []
    for item in raw.split(","):
        rows, width = item.lower().split("x", 1)
        shapes.append((int(rows), int(width)))
    return tuple(shapes)


def _pattern(rows: int, width: int, rank: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(rows * width, device=device).remainder(5) + 1
    return (base * (rank + 1)).to(torch.bfloat16).view(rows, width)


def _expected(rows: int, width: int, world: int, device: torch.device) -> torch.Tensor:
    total = world * (world + 1) // 2
    return _pattern(rows, width, 0, device) * total


def _time(fn, device: torch.device, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1, int(len(samples) * 0.9))]


def _worker(rank: int, port: int, args: argparse.Namespace) -> None:
    from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool
    from b12x.comm.pcie.pcie_island9 import PCIeIsland9AllReduce
    from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16

    world = args.world_size
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    group = dist.group.WORLD
    shapes = _shapes(args.shapes)
    results = []

    def emit(path: str, rows: int, width: int, mode: str, median: float, p90: float):
        record = {
            "path": path, "rows": rows, "width": width,
            "bytes": rows * width * 2, "mode": mode,
            "median_us": round(median, 2), "p90_us": round(p90, 2),
            "world_size": world,
        }
        results.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)

    def verify(out: torch.Tensor, rows: int, width: int, path: str) -> None:
        torch.cuda.synchronize(device)
        expected = _expected(rows, width, world, device)
        if not torch.equal(out, expected):
            raise AssertionError(f"{path} {rows}x{width} mismatch on rank {rank}")

    # PyNCCL baseline.
    for rows, width in shapes:
        inp = _pattern(rows, width, rank, device)
        work = inp.clone()

        def nccl_call():
            work.copy_(inp)
            dist.all_reduce(work, group=group)

        nccl_call()
        verify(work, rows, width, "nccl")
        median, p90 = _time(nccl_call, device, args.iters, args.warmup)
        emit("nccl", rows, width, "eager", median, p90)

    # B12X one-shot pool (all-peer pull). Every independently replayable
    # graph needs its own logical channel and the pool's resident-CTA budget
    # bounds concurrent channels, so each shape gets a fresh two-channel pool.
    for rows, width in shapes:
        inp = _pattern(rows, width, rank, device)
        out = torch.empty_like(inp)
        pool = PCIeOneshotAllReducePool.from_process_group(
            process_group=group,
            device=device,
            max_input_bytes=rows * width * 2,
            max_concurrent_channels=2,
        )
        pool.prepare_channels(("eager", "graph"))
        # Acceptance is a deterministic function of the shape, so a rejection
        # happens on every rank before any peer traffic.
        try:
            pool.all_reduce(inp, out=out, channel_id="eager")
        except Exception as error:
            if rank == 0:
                print(json.dumps({"path": "oneshot", "rows": rows, "width": width,
                                  "skipped": repr(error)[:200]}), flush=True)
            pool.close()
            continue
        verify(out, rows, width, "oneshot")
        median, p90 = _time(
            lambda: pool.all_reduce(inp, out=out, channel_id="eager"),
            device, args.iters, args.warmup,
        )
        emit("oneshot", rows, width, "eager", median, p90)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            pool.all_reduce(inp, out=out, channel_id="graph")
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with pool.capture(stream, channel_id="graph"), torch.cuda.graph(graph, stream=stream):
            pool.all_reduce(inp, out=out, channel_id="graph")
        graph.replay()
        verify(out, rows, width, "oneshot-graph")
        median, p90 = _time(graph.replay, device, args.iters, args.warmup)
        emit("oneshot", rows, width, "graph", median, p90)
        del graph
        pool.close()

    # B12X BF16 two-shot (pull reduce-scatter + pull all-gather).
    twoshot = PCIeTwoShotBF16.from_exchange_group(
        exchange_group=group,
        device=device,
        max_rows=args.twoshot_max_rows,
        row_elems=args.twoshot_row_elems,
    )
    for rows, width in shapes:
        inp = _pattern(rows, width, rank, device)
        out = torch.empty_like(inp)
        if not twoshot.accepts(inp):
            if rank == 0:
                print(json.dumps({"path": "twoshot", "rows": rows, "width": width,
                                  "skipped": "not accepted"}), flush=True)
            continue
        twoshot.all_reduce(inp, out=out)
        verify(out, rows, width, "twoshot")
        median, p90 = _time(
            lambda: twoshot.all_reduce(inp, out=out), device, args.iters, args.warmup
        )
        emit("twoshot", rows, width, "eager", median, p90)
        graph = torch.cuda.CUDAGraph()
        with twoshot.capture(), torch.cuda.graph(graph):
            twoshot.all_reduce(inp, out=out)
        graph.replay()
        verify(out, rows, width, "twoshot-graph")
        median, p90 = _time(graph.replay, device, args.iters, args.warmup)
        emit("twoshot", rows, width, "graph", median, p90)
        del graph
    # Fused push all-reduce: one launch built on posted PCIe writes.
    pushed = PCIeTwoShotBF16.from_exchange_group(
        exchange_group=group,
        device=device,
        max_rows=args.twoshot_max_rows,
        row_elems=args.twoshot_row_elems,
    )
    pushed.all_reduce_mode = "push"
    for rows, width in shapes:
        inp = _pattern(rows, width, rank, device)
        out = torch.empty_like(inp)
        if not pushed.accepts(inp):
            continue
        pushed.all_reduce(inp, out=out)
        verify(out, rows, width, "twoshot-push-fused")
        median, p90 = _time(
            lambda: pushed.all_reduce(inp, out=out), device, args.iters, args.warmup
        )
        emit("twoshot-push-fused", rows, width, "eager", median, p90)
        graph = torch.cuda.CUDAGraph()
        with pushed.capture(), torch.cuda.graph(graph):
            pushed.all_reduce(inp, out=out)
        graph.replay()
        verify(out, rows, width, "twoshot-push-fused-graph")
        median, p90 = _time(graph.replay, device, args.iters, args.warmup)
        emit("twoshot-push-fused", rows, width, "graph", median, p90)
        del graph
    pushed.close()

    # Two-island push all-reduce (ranks 0-3 / 4-7 reduce quarters inside
    # their switch clusters; rank 8 contributes through island 0).
    island9 = PCIeIsland9AllReduce.from_exchange_group(
        exchange_group=group,
        device=device,
        max_rows=args.twoshot_max_rows,
        row_elems=args.twoshot_row_elems,
    )
    for rows, width in shapes:
        inp = _pattern(rows, width, rank, device)
        out = torch.empty_like(inp)
        if not island9.accepts(inp):
            continue
        island9.all_reduce(inp, out=out)
        verify(out, rows, width, "island9-push")
        median, p90 = _time(
            lambda: island9.all_reduce(inp, out=out), device, args.iters, args.warmup
        )
        emit("island9-push", rows, width, "eager", median, p90)
        graph = torch.cuda.CUDAGraph()
        with island9.capture(), torch.cuda.graph(graph):
            island9.all_reduce(inp, out=out)
        graph.replay()
        verify(out, rows, width, "island9-push-graph")
        median, p90 = _time(graph.replay, device, args.iters, args.warmup)
        emit("island9-push", rows, width, "graph", median, p90)
        del graph
    island9.close()

    # Push-based reduce-scatter followed by push-based all-gather (two
    # launches, posted PCIe writes instead of remote reads). Rows are padded
    # to a multiple of the world size, which is the row contract of the
    # push kernels; the reference only covers the unpadded prefix.
    for rows, width in shapes:
        packs = rows * width // 8
        padded_rows = -(-packs // world) * world
        payload = torch.zeros(padded_rows, 8, dtype=torch.bfloat16, device=device)
        payload.view(-1)[: rows * width].copy_(_pattern(rows, width, rank, device).view(-1))
        shard = torch.empty(padded_rows // world, 8, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(payload)

        def push_call():
            twoshot.reduce_scatter(payload, out=shard)
            twoshot.all_gather(shard, out=out)

        try:
            push_call()
        except Exception as error:
            if rank == 0:
                print(json.dumps({"path": "twoshot-push", "rows": rows, "width": width,
                                  "skipped": repr(error)[:200]}), flush=True)
            continue
        verify(out.view(-1)[: rows * width].view(rows, width), rows, width, "twoshot-push")
        median, p90 = _time(push_call, device, args.iters, args.warmup)
        emit("twoshot-push", rows, width, "eager", median, p90)
        graph = torch.cuda.CUDAGraph()
        with twoshot.capture(), torch.cuda.graph(graph):
            push_call()
        graph.replay()
        verify(out.view(-1)[: rows * width].view(rows, width), rows, width, "twoshot-push-graph")
        median, p90 = _time(graph.replay, device, args.iters, args.warmup)
        emit("twoshot-push", rows, width, "graph", median, p90)
        del graph
    twoshot.close()

    if rank == 0 and args.output:
        with open(args.output, "w") as stream_out:
            json.dump({"shapes": shapes, "iters": args.iters, "results": results},
                      stream_out, indent=2)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--twoshot-max-rows", type=int, default=49149)
    parser.add_argument("--twoshot-row-elems", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"need {args.world_size} visible GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
