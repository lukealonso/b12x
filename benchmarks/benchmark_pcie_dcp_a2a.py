from __future__ import annotations

import json
import os
import pathlib
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2APool,
    lse_reduce_scatter_reference,
)


TOTAL_HEADS = int(os.getenv("B12X_PCIE_DCP_A2A_TOTAL_HEADS", "32"))
HEAD_DIM = int(os.getenv("B12X_PCIE_DCP_A2A_HEAD_DIM", "512"))
QUERY_HEAD_DIM = int(os.getenv("B12X_PCIE_DCP_A2A_QUERY_HEAD_DIM", "576"))
MAX_BATCH = int(os.getenv("B12X_PCIE_DCP_A2A_MAX_BATCH", "8"))
GEOMETRY_OVERRIDE_ENVS = (
    "B12X_PCIE_DCP_THREADS",
    "B12X_PCIE_DCP_BLOCK_LIMIT",
)


def _csv_ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(value) for value in os.getenv(name, default).split(","))


def _launches(world_size: int) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    for raw in os.getenv("B12X_PCIE_DCP_A2A_LAUNCHES", "256x16").split(","):
        try:
            raw_threads, raw_blocks = raw.lower().split("x", 1)
            threads = int(raw_threads)
            blocks = int(raw_blocks)
        except ValueError as exc:
            raise ValueError(
                "B12X_PCIE_DCP_A2A_LAUNCHES must contain "
                "comma-separated THREADSxBLOCKS tuples"
            ) from exc
        if not world_size <= threads <= 1024 or threads % 32 != 0:
            raise ValueError(
                f"invalid launch {raw!r}: threads must be a multiple of 32 "
                f"in [{world_size}, 1024]"
            )
        if not 1 <= blocks <= 64:
            raise ValueError(f"invalid launch {raw!r}: blocks must be in [1, 64]")
        values.append((threads, blocks))
    return tuple(values)


def _reject_production_geometry_overrides() -> None:
    active = tuple(
        name for name in GEOMETRY_OVERRIDE_ENVS if os.getenv(name, "").strip()
    )
    if active:
        raise ValueError(
            "benchmark launch sweeps require production geometry overrides to be "
            f"unset; active overrides: {', '.join(active)}"
        )


def _batches() -> tuple[int, ...]:
    batches = _csv_ints("B12X_PCIE_DCP_A2A_BATCHES", "1,2,4,8")
    invalid = tuple(batch for batch in batches if not 1 <= batch <= MAX_BATCH)
    if invalid:
        raise ValueError(
            "B12X_PCIE_DCP_A2A_BATCHES values must be in "
            f"[1, {MAX_BATCH}], got {invalid}"
        )
    return batches


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _capture(
    fn: Callable[[], None],
    pool: PCIeDCPA2APool,
    channel_id: str,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with pool.capture(channel_id=channel_id), torch.cuda.graph(graph):
        fn()
    return graph


def _measure(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    *,
    warmup: int = 200,
    iterations: int = 2_000,
) -> float:
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize(device)
    latency = (time.perf_counter() - start) * 1e6 / iterations
    rank_latency = torch.tensor(latency, dtype=torch.float64, device=device)
    dist.all_reduce(rank_latency, op=dist.ReduceOp.MAX)
    return float(rank_latency.item())


def _latency_samples(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
) -> tuple[float, list[float]]:
    samples = [_measure(graph, device) for _ in range(3)]
    return statistics.median(samples), samples


def _rank_inputs(
    source_rank: int,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(10_000 * batch + source_rank)
    output = torch.randn(
        batch,
        TOTAL_HEADS,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    lse = torch.randn(
        batch,
        TOTAL_HEADS,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    return output, lse


def _rank_query(
    source_rank: int,
    world_size: int,
    batch: int,
    query_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20_000 * batch + source_rank)
    return torch.randn(
        batch,
        TOTAL_HEADS // world_size,
        QUERY_HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=query_dtype)


def _assert_gather_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float8_e4m3fn:
        if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            raise AssertionError("raw-byte FP8 gather differs from its reference")
        return
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _validate_graphs(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    query_dtype: torch.dtype,
    batch: int,
    reduce_result: torch.Tensor,
    gather_out: torch.Tensor,
    reduce_graph: torch.cuda.CUDAGraph,
    gather_graph: torch.cuda.CUDAGraph,
    pair_graph: torch.cuda.CUDAGraph,
) -> None:
    stacked = [_rank_inputs(source, batch, device) for source in range(world_size)]
    expected_reduce = lse_reduce_scatter_reference(
        torch.stack([item[0] for item in stacked]),
        torch.stack([item[1] for item in stacked]),
        rank,
    )
    expected_gather = torch.cat(
        [
            _rank_query(source, world_size, batch, query_dtype, device)
            for source in range(world_size)
        ],
        dim=1,
    )

    reduce_graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(reduce_result, expected_reduce, rtol=2e-2, atol=2e-2)
    gather_graph.replay()
    torch.cuda.synchronize(device)
    _assert_gather_exact(gather_out, expected_gather)
    pair_graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(reduce_result, expected_reduce, rtol=2e-2, atol=2e-2)
    _assert_gather_exact(gather_out, expected_gather)


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()


def _metadata(world_size: int, query_dtype_name: str) -> dict[str, object]:
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    status = _git_value("status", "--short")
    return {
        "schema": "b12x.pcie_dcp_a2a.graph_benchmark.v1",
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "worktree": _git_value("rev-parse", "--show-toplevel"),
        "dirty": None if status == "unavailable" else bool(status),
        "world_size": world_size,
        "query_dtype": query_dtype_name,
        "transport": os.getenv("B12X_PCIE_DCP_A2A_TRANSPORT", "pull"),
        "gpu": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_index": device_index,
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "uuid": str(getattr(props, "uuid", "")),
        },
        "correctness": "graph_oracle_passed_before_timing",
        "ratio_direction": "lower_latency_is_better",
    }


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool: PCIeDCPA2APool | None = None
    try:
        dtype = torch.bfloat16
        query_dtype_name = (
            os.getenv("B12X_PCIE_DCP_A2A_QUERY_DTYPE", "bf16").strip().lower()
        )
        query_dtypes = {
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
        }
        try:
            query_dtype = query_dtypes[query_dtype_name]
        except KeyError:
            raise ValueError(
                "B12X_PCIE_DCP_A2A_QUERY_DTYPE must be 'bf16' or 'fp8'"
            ) from None
        batches = _batches()
        launches = _launches(world_size)
        heads_per_rank = TOTAL_HEADS // world_size
        pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=MAX_BATCH,
            total_heads=TOTAL_HEADS,
            head_dim=HEAD_DIM,
            query_head_dim=QUERY_HEAD_DIM,
        )
        channel_ids = tuple(
            f"benchmark:{batch}:{threads}:{blocks}:{operation}"
            for batch in batches
            for threads, blocks in launches
            for operation in ("reduce", "gather", "pair")
        )
        pool.prepare_channels(channel_ids)
        preparation_channel_id = channel_ids[0]
        for threads in sorted({threads for threads, _ in launches}):
            pool.prepare_graph_lse_reduce_scatter(
                dtype=dtype,
                threads=threads,
                channel_id=preparation_channel_id,
            )
            pool.prepare_graph_all_gather_heads(
                threads=threads,
                channel_id=preparation_channel_id,
            )
        if rank == 0:
            print(f"# metadata={json.dumps(_metadata(world_size, query_dtype_name))}")
            print(
                "world_size,query_dtype,batch,threads,block_limit,"
                "reduce_in_bytes,gather_in_bytes,correctness,"
                "reduce_us,gather_us,pair_us,reduce_samples_us,"
                "gather_samples_us,pair_samples_us,ratio_direction"
            )
        for batch in batches:
            reduce_out, reduce_lse = _rank_inputs(rank, batch, device)
            gather_in = _rank_query(
                rank,
                world_size,
                batch,
                query_dtype,
                device,
            )
            reduce_result = torch.empty(
                batch, heads_per_rank, HEAD_DIM, dtype=dtype, device=device
            )
            gather_out = torch.empty(
                batch,
                TOTAL_HEADS,
                QUERY_HEAD_DIM,
                dtype=query_dtype,
                device=device,
            )

            for threads, block_limit in launches:
                channel_prefix = f"benchmark:{batch}:{threads}:{block_limit}"
                reduce_graph = _capture(
                    lambda: pool.lse_reduce_scatter(
                        reduce_out,
                        reduce_lse,
                        reduce_result,
                        threads=threads,
                        block_limit=block_limit,
                    ),
                    pool,
                    f"{channel_prefix}:reduce",
                )
                gather_graph = _capture(
                    lambda: pool.all_gather_heads(
                        gather_in,
                        gather_out,
                        threads=threads,
                        block_limit=block_limit,
                    ),
                    pool,
                    f"{channel_prefix}:gather",
                )

                def pair() -> None:
                    pool.all_gather_heads(
                        gather_in,
                        gather_out,
                        threads=threads,
                        block_limit=block_limit,
                    )
                    pool.lse_reduce_scatter(
                        reduce_out,
                        reduce_lse,
                        reduce_result,
                        threads=threads,
                        block_limit=block_limit,
                    )

                pair_graph = _capture(pair, pool, f"{channel_prefix}:pair")
                _validate_graphs(
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    query_dtype=query_dtype,
                    batch=batch,
                    reduce_result=reduce_result,
                    gather_out=gather_out,
                    reduce_graph=reduce_graph,
                    gather_graph=gather_graph,
                    pair_graph=pair_graph,
                )
                reduce_us, reduce_samples = _latency_samples(reduce_graph, device)
                gather_us, gather_samples = _latency_samples(gather_graph, device)
                pair_us, pair_samples = _latency_samples(pair_graph, device)
                if rank == 0:
                    reduce_bytes = (
                        reduce_out.numel() * dtype.itemsize
                        + reduce_lse.numel() * reduce_lse.itemsize
                    )
                    gather_bytes = gather_in.numel() * query_dtype.itemsize
                    reduce_raw = "|".join(f"{value:.3f}" for value in reduce_samples)
                    gather_raw = "|".join(f"{value:.3f}" for value in gather_samples)
                    pair_raw = "|".join(f"{value:.3f}" for value in pair_samples)
                    print(
                        f"{world_size},{query_dtype_name},{batch},{threads},"
                        f"{block_limit},{reduce_bytes},{gather_bytes},pass,"
                        f"{reduce_us:.3f},{gather_us:.3f},{pair_us:.3f},"
                        f"{reduce_raw},{gather_raw},{pair_raw},"
                        "lower_latency_is_better",
                        flush=True,
                    )
                del reduce_graph, gather_graph, pair_graph
                torch.cuda.synchronize(device)
    finally:
        if pool is not None:
            pool.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    _reject_production_geometry_overrides()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    world_size = int(os.getenv("B12X_PCIE_DCP_A2A_WORLD_SIZE", "8"))
    if torch.cuda.device_count() < world_size:
        raise SystemExit(f"need {world_size} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(
        _worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
