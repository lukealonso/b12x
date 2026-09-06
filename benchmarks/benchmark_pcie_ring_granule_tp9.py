#!/usr/bin/env python3
"""Graph-replay latency of the nine-rank PCIe DMA ring per chunk mapping and
per fp32 reduce-scatter hop count.

Each variant is one ``PCIeDmaAllReduce`` channel: ``dma-ring`` is the served
ring (contiguous chunks, bf16 on every hop), ``dma-ring-granule`` its
row-count-invariant chunk mapping (``--granule-rows``), and
``dma-ring-granule-fp32hP`` that mapping with ``P`` trailing fp32
reduce-scatter hops. The mapping does not change wire bytes; an fp32 hop
doubles the bytes of one of the ``2 * (world - 1)`` hops, so the ring moves
``(2 * (world - 1) + P) / world`` of the tensor per rank instead of
``2 * (world - 1) / world``: +1/16 of the served bytes per hop at nine ranks.
The measured column to compare against that model is ``median_us``.

Every variant is captured into its own CUDA graph over its own buffers and
replayed ``--iters`` times per sample, alternating variant order between
samples; the reported number is the maximum over ranks of the per-replay
median. Correctness is checked once per shape against the correctly rounded
fp64 sum of the nine bf16 inputs, and the two row halves of each shape are
reduced separately to report the elements on which they differ from the whole
all-reduce (0 is the property the granule mapping exists for).

    B12X_RUN_PCIE_TP9_TEST=1 python benchmarks/benchmark_pcie_ring_granule_tp9.py \
        --shapes 4608x7168,2304x7168 --output r1-timing.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import timedelta
from statistics import median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 9


def _shapes(raw: str) -> tuple[tuple[int, int], ...]:
    shapes = []
    for item in raw.split(","):
        rows, _, width = item.strip().partition("x")
        shapes.append((int(rows), int(width)))
    return tuple(shapes)


def _activations(
    rows: int, width: int, rank: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(9000 + rank)
    values = torch.randn(rows, width, generator=generator, dtype=torch.float32)
    return (values * float(2.0 ** (rank - 4))).to(torch.bfloat16).to(device)


def _variants(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    """(name, granule rows, fp32 hops) of every channel to time."""
    out = [("dma-ring", 0, 0), ("dma-ring-granule", args.granule_rows, 0)]
    for hops in args.fp32_hops:
        out.append((f"dma-ring-granule-fp32h{hops}", args.granule_rows, hops))
    return out


def _wire_bytes_per_rank(numel: int, hops: int) -> int:
    """Bytes one rank sends for one all-reduce: ``2 * (world - 1)`` hops of one
    shard, with ``hops`` of them widened to fp32."""
    shard = numel * 2 // WORLD
    return (2 * (WORLD - 1) + hops) * shard


def _time_graphs(
    graphs: dict[str, torch.cuda.CUDAGraph],
    device: torch.device,
    *,
    warmup: int,
    iters: int,
    samples: int,
) -> dict[str, list[float]]:
    for graph in graphs.values():
        for _ in range(warmup):
            graph.replay()
    torch.cuda.synchronize(device)
    timings: dict[str, list[float]] = {name: [] for name in graphs}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    rank_max = torch.empty((), dtype=torch.float64, device=device)
    names = list(graphs)
    for sample in range(samples):
        order = names if sample % 2 == 0 else list(reversed(names))
        for name in order:
            dist.barrier(device_ids=[device.index])
            start.record()
            for _ in range(iters):
                graphs[name].replay()
            end.record()
            end.synchronize()
            rank_max.fill_(start.elapsed_time(end) * 1e3 / iters)
            dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
            timings[name].append(float(rank_max.item()))
    return timings


def _worker(rank: int, port: int, args: argparse.Namespace) -> None:
    from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=600),
        device_id=device,
    )
    group = dist.group.WORLD
    shapes = _shapes(args.shapes)
    max_bytes = max(rows * width for rows, width in shapes) * 2
    rings = {}
    for name, granule_rows, hops in _variants(args):
        rings[name] = PCIeDmaAllReduce(
            exchange_group=group,
            device=device,
            max_bytes=max_bytes,
            granule_rows=granule_rows,
            fp32_hops=hops,
        )
    records = []
    for rows, width in shapes:
        inp = _activations(rows, width, rank, device)
        gathered = [torch.empty_like(inp) for _ in range(WORLD)]
        dist.all_gather(gathered, inp, group=group)
        reference64 = sum(tensor.double() for tensor in gathered)
        outputs = {}
        graphs = {}
        for name, ring in rings.items():
            out = torch.empty_like(inp)
            ring.all_reduce(inp, out=out)
            torch.cuda.synchronize(device)
            outputs[name] = out.clone()
            graph_in = inp.clone()
            graph_out = torch.empty_like(inp)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ring.all_reduce(graph_in, out=graph_out)
            graphs[name] = graph
        timings = _time_graphs(
            graphs,
            device,
            warmup=args.warmup,
            iters=args.iters,
            samples=args.samples,
        )
        for name, ring in rings.items():
            out = outputs[name]
            error = float(
                (
                    (out.double() - reference64).norm()
                    / reference64.norm().clamp_min(1e-30)
                ).item()
            )
            split_mismatch = -1
            if rows % 2 == 0:
                halves = torch.empty_like(out)
                half = rows // 2
                ring.all_reduce(inp[:half].contiguous(), out=halves[:half])
                ring.all_reduce(inp[half:].contiguous(), out=halves[half:])
                torch.cuda.synchronize(device)
                split_mismatch = int((halves != out).sum().item())
            hops = int(name.rpartition("fp32h")[2] or 0)
            records.append(
                {
                    "rows": rows,
                    "width": width,
                    "kernel": name,
                    "fp32_hops": hops,
                    "granule_rows": 0 if name == "dma-ring" else args.granule_rows,
                    "median_us": float(median(timings[name])),
                    "min_us": float(min(timings[name])),
                    "max_us": float(max(timings[name])),
                    "wire_bytes_per_rank": _wire_bytes_per_rank(inp.numel(), hops),
                    "rel_l2_vs_fp64": error,
                    "split_mismatch_elements": split_mismatch,
                }
            )
        del graphs
        torch.cuda.synchronize(device)
    for ring in rings.values():
        ring.close()
    if rank == 0:
        print(
            "rows,width,kernel,median_us,min_us,max_us,wire_MB_per_rank,"
            "rel_l2_vs_fp64,split_mismatch",
            flush=True,
        )
        for record in records:
            print(
                f"{record['rows']},{record['width']},{record['kernel']},"
                f"{record['median_us']:.1f},{record['min_us']:.1f},"
                f"{record['max_us']:.1f},"
                f"{record['wire_bytes_per_rank'] / 1e6:.2f},"
                f"{record['rel_l2_vs_fp64']:.3e},"
                f"{record['split_mismatch_elements']}",
                flush=True,
            )
        if args.output:
            with open(args.output, "w") as stream:
                json.dump({"world": WORLD, "records": records}, stream, indent=2)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="4608x7168,2304x7168,4608x3584")
    parser.add_argument("--granule-rows", type=int, default=128)
    parser.add_argument("--fp32-hops", default="1,2,3,7")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    args.fp32_hops = [int(v) for v in args.fp32_hops.split(",") if v.strip()]
    if os.getenv("B12X_RUN_PCIE_TP9_TEST") != "1":
        raise SystemExit("requires nine idle GPUs and B12X_RUN_PCIE_TP9_TEST=1")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_worker, args=(port, args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
