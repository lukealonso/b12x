#!/usr/bin/env python3
"""Compare TP2 B12X plain all-reduce with NCCL under CUDA graph replay.

The benchmark records the per-sample slowest-rank latency because that is the
latency observed by a distributed caller. Inputs are reset before every replay,
and each backend must produce the exact two-rank sum before timings are emitted.
Run it with ``torchrun --standalone --nproc-per-node=2``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--rows",
        default="1,2,4,6,8,16,24,32,48,64,96,128",
        help="Comma-separated row counts.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument(
        "--topology-description",
        required=True,
        help=(
            "Semantic description of the physical path between the selected "
            "GPUs, for example 'separate CPU root ports without a PCIe switch'"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def _source_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return {
        "worktree": str(root),
        "revision": _run_text(["git", "rev-parse", "HEAD"], cwd=root),
        "tree": _run_text(["git", "rev-parse", "HEAD^{tree}"], cwd=root),
        "dirty_paths": _run_text(["git", "status", "--short"], cwd=root).splitlines(),
    }


def _topology() -> str:
    return _run_text(["nvidia-smi", "topo", "-m"])


def _nvidia_smi_inventory() -> str:
    return _run_text(
        [
            "nvidia-smi",
            (
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version,"
                "persistence_mode,compute_mode,clocks.sm,clocks.max.sm,power.limit"
            ),
            "--format=csv,noheader",
        ]
    )


def _group_max_samples(samples_us: list[float], device: torch.device) -> list[float]:
    values = torch.tensor(samples_us, dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return [float(value) for value in values.cpu().tolist()]


def _measure_graph(
    graph: torch.cuda.CUDAGraph,
    reset: Callable[[], None],
    *,
    warmup: int,
    samples: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        reset()
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()

    timings = []
    for _ in range(samples):
        reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1e3)
    return _group_max_samples(timings, device)


def _distribution(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "minimum_us": ordered[0],
        "median_us": ordered[(len(ordered) - 1) // 2],
        "p95_us": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "maximum_us": ordered[-1],
    }


def _correct(output: torch.Tensor, expected: float, device: torch.device) -> bool:
    local = torch.tensor(
        int(bool(torch.all(output == expected).item())),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    return bool(local.item())


def _capture_b12x(
    pool: object,
    inp: torch.Tensor,
    out: torch.Tensor,
) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream(device=inp.device)
    stream.wait_stream(torch.cuda.current_stream(inp.device))
    graph = torch.cuda.CUDAGraph()
    with pool.capture(stream) as channel:
        with torch.cuda.stream(stream):
            channel.prepare_graph_all_reduce(inp)
        with torch.cuda.graph(graph, stream=stream):
            channel.all_reduce(inp, out=out)
    torch.cuda.current_stream(inp.device).wait_stream(stream)
    torch.cuda.synchronize(inp.device)
    return graph


def _b12x_graph_plan(pool: object, inp: torch.Tensor) -> dict[str, object]:
    """Return the immutable launch plan retained during graph preparation."""

    channel = pool.for_stream()
    backend = channel._ext
    state = backend._state(channel._ptr)
    plan = state.plain_graph_plans[backend._plain_graph_plan_key(inp)]
    return {
        "transport": plan.transport,
        "threads_per_cta": plan.threads,
        "ctas": plan.blocks,
    }


def _capture_nccl(inp: torch.Tensor) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream(device=inp.device)
    stream.wait_stream(torch.cuda.current_stream(inp.device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        dist.all_reduce(inp)
    torch.cuda.current_stream(inp.device).wait_stream(stream)
    torch.cuda.synchronize(inp.device)
    return graph


def _benchmark_b12x(
    pool: object,
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    rank: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    value = float(rank + 1)
    inp = torch.full((rows, hidden_size), value, dtype=dtype, device=device)
    out = torch.empty_like(inp)
    graph = _capture_b12x(pool, inp, out)
    launch_plan = _b12x_graph_plan(pool, inp)
    samples = _measure_graph(
        graph,
        lambda: inp.fill_(value),
        warmup=args.warmup,
        samples=args.samples,
        device=device,
    )
    graph.replay()
    torch.cuda.synchronize(device)
    return {
        "backend": "b12x",
        "launch_plan": launch_plan,
        "correct": _correct(out, 3.0, device),
        "samples_slowest_rank_us": samples,
        **_distribution(samples),
    }


def _benchmark_nccl(
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    rank: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    value = float(rank + 1)
    inp = torch.full((rows, hidden_size), value, dtype=dtype, device=device)
    graph = _capture_nccl(inp)
    samples = _measure_graph(
        graph,
        lambda: inp.fill_(value),
        warmup=args.warmup,
        samples=args.samples,
        device=device,
    )
    inp.fill_(value)
    graph.replay()
    torch.cuda.synchronize(device)
    return {
        "backend": "nccl",
        "correct": _correct(inp, 3.0, device),
        "samples_slowest_rank_us": samples,
        **_distribution(samples),
    }


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"TP2 benchmark requires world_size=2, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    dtype = getattr(torch, args.dtype)
    row_counts = [int(value) for value in args.rows.split(",")]

    from b12x.comm.pcie import OneshotAllReducePool

    maximum_bytes = max(row_counts) * args.hidden_size * dtype.itemsize
    pool = OneshotAllReducePool(
        rank=rank,
        world_size=world_size,
        device=device,
        exchange_group=dist.group.WORLD,
        eager_buffer_bytes=maximum_bytes,
        max_size=maximum_bytes,
        rank_data_bytes=maximum_bytes,
        single_channel=True,
    )

    rank_device = {
        "rank": rank,
        "logical_device": local_rank,
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }
    devices: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(devices, rank_device)

    results = []
    try:
        for rows in row_counts:
            b12x = _benchmark_b12x(
                pool, rows, args.hidden_size, dtype, rank, args, device
            )
            dist.barrier()
            nccl = _benchmark_nccl(rows, args.hidden_size, dtype, rank, args, device)
            dist.barrier()
            if not b12x["correct"] or not nccl["correct"]:
                raise RuntimeError(f"all-reduce oracle failed for rows={rows}")
            results.append(
                {
                    "rows": rows,
                    "hidden_size": args.hidden_size,
                    "bytes": rows * args.hidden_size * dtype.itemsize,
                    "b12x": b12x,
                    "nccl": nccl,
                    "median_ratio_b12x_over_nccl": (
                        float(b12x["median_us"]) / float(nccl["median_us"])
                    ),
                }
            )
    finally:
        pool.close()

    if rank == 0:
        report = {
            "contract": "TP2 CUDA-graph plain all-reduce",
            "ratio_definition": (
                "b12x median / NCCL median; values below 1 mean B12X is faster"
            ),
            "base_revision": args.base_revision,
            "source": _source_identity(),
            "host": socket.gethostname(),
            "topology_description": args.topology_description,
            "worker_command": [sys.executable, *sys.argv],
            "distributed_environment": {
                name: os.environ.get(name)
                for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
            },
            "python_version": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "collective_environment": {
                name: os.environ.get(name)
                for name in (
                    "B12X_PCIE_ONESHOT_BLOCK_LIMIT",
                    "B12X_PCIE_ONESHOT_THREADS",
                    "B12X_PCIE_TP2_PLAIN_REMOTE_PUSH",
                    "B12X_PCIE_TP2_REMOTE_PUSH",
                    "NCCL_ALLOC_P2P_NET_LL_BUFFERS",
                    "NCCL_BUFFSIZE",
                    "NCCL_DMABUF_ENABLE",
                    "NCCL_IB_DISABLE",
                    "NCCL_IGNORE_CPU_AFFINITY",
                    "NCCL_MIN_NCHANNELS",
                    "NCCL_P2P_LEVEL",
                    "NCCL_PROTO",
                )
            },
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "dtype": args.dtype,
            "warmup_replays": args.warmup,
            "timed_replays": args.samples,
            "devices": devices,
            "nvidia_smi_inventory": _nvidia_smi_inventory(),
            "nvidia_smi_topology": _topology(),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
        for result in results:
            print(
                f"M={result['rows']:>3}: "
                f"B12X={result['b12x']['median_us']:.2f} us "
                f"NCCL={result['nccl']['median_us']:.2f} us "
                f"ratio={result['median_ratio_b12x_over_nccl']:.3f}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
