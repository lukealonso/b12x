"""Latency of the RoCEnante collectives versus torch.distributed (NCCL), with a receipt.

Launch with torchrun on every node (one GPU per node)::

    torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29651 \\
        benchmarks/benchmark_roce_oneshot.py --output docs/evidence/rocenante/<receipt>.json

Correctness gates run before any timing: the RoCEnante all-reduce must match
NCCL within the dtype tolerance and the all-gather must be bit-exact, or the
benchmark raises.  Both are checked again after timing.  Timing alternates the
NCCL eager/graph and RoCEnante eager/graph arms in blocks so clock or thermal
drift cannot bias one direction, and the executed order is recorded.

Rank 0 prints a table and writes one JSON receipt (schema
``b12x.comm.roce.oneshot.benchmark`` version 3) with the command, source
revision and worktree state, per-rank hostname and GPU identity, correctness
results, unrounded raw samples from rank 0, per-rank medians, the cross-rank
median-of-slowest summary, and ratios labelled with their direction.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    """Run ``git`` in the repository; ``None`` when git is unavailable."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def _source_state() -> dict[str, object]:
    """Source revision and worktree state for the receipt.

    Uses git when present.  A container without git can receive the launcher's
    values through ``B12X_BENCH_GIT_REV`` and ``B12X_BENCH_GIT_STATUS`` (the
    output of ``git status --porcelain`` on the host); failing both, the
    revision is read from ``.git/HEAD`` and the worktree state is unknown.
    """
    rev = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if rev is not None and status is not None:
        return {
            "source_revision": rev,
            "worktree_dirty": bool(status),
            "git_status": status.splitlines()[:40],
            "source_state_from": "git",
        }
    env_rev = os.environ.get("B12X_BENCH_GIT_REV", "")
    if env_rev and "B12X_BENCH_GIT_STATUS" in os.environ:
        status = os.environ["B12X_BENCH_GIT_STATUS"]
        return {
            "source_revision": env_rev,
            "worktree_dirty": bool(status.strip()),
            "git_status": status.splitlines()[:40],
            "source_state_from": "launcher environment (B12X_BENCH_GIT_REV, B12X_BENCH_GIT_STATUS)",
        }
    head = ""
    try:
        head = (REPO_ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            head = (REPO_ROOT / ".git" / head[5:]).read_text().strip()
    except OSError:
        pass
    return {
        "source_revision": head,
        "worktree_dirty": None,
        "git_status": ["unknown: git unavailable"],
        "source_state_from": ".git/HEAD",
    }


def dump_compact_json(doc: object, path: Path) -> None:
    """Write ``doc`` indented, with lists of numbers on one line each.

    The receipt keeps every raw sample; this layout keeps it reviewable (one
    line per sample array instead of one per number).
    """
    markers: dict[str, list[float]] = {}

    def mark(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: mark(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if obj and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj
            ):
                key = f"__compact_{len(markers)}__"
                markers[key] = obj
                return key
            return [mark(v) for v in obj]
        return obj

    text = json.dumps(mark(doc), indent=2)
    for key, values in markers.items():
        text = text.replace(
            f'"{key}"', "[" + ", ".join(json.dumps(v) for v in values) + "]"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def _gpu_identity() -> dict[str, object]:
    """Physical GPU identity and mode for the receipt (torch plus best-effort nvidia-smi)."""
    props = torch.cuda.get_device_properties(0)
    info: dict[str, object] = {
        "name": props.name,
        "uuid": str(getattr(props, "uuid", "")),
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    fields = "driver_version,pstate,clocks.sm,clocks.mem,power.limit,persistence_mode,compute_mode"
    try:
        smi = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        if smi:
            info["nvidia_smi"] = dict(
                zip(
                    fields.split(","), [v.strip() for v in smi.split(",")], strict=False
                )
            )
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _tolerance(dtype: torch.dtype, world: int) -> tuple[float, float]:
    """(rtol, atol) for a ``world``-way sum in ``dtype`` with float32 accumulation."""
    if dtype == torch.float32:
        return 1e-5, 1e-5 * world
    if dtype == torch.bfloat16:
        return 1e-2, 2e-2 * world
    return 2e-3, 4e-3 * world


def _time_eager(
    fn: Callable[[], object],
    warmups: int,
    samples: int,
    prep: Callable[[], object] | None = None,
) -> list[float]:
    """Time ``samples`` calls of ``fn`` in microseconds with CUDA events; ``prep`` runs untimed before each."""
    for _ in range(warmups):
        if prep is not None:
            prep()
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    out = []
    for _ in range(samples):
        if prep is not None:
            prep()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0)
    return out


def _capture_graph(
    fn: Callable[[], object], warmups: int, per_graph: int
) -> torch.cuda.CUDAGraph:
    """Warm ``fn`` on a side stream, then capture ``per_graph`` calls into one graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(per_graph):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    return graph


Arm = tuple[Callable[[], object], Callable[[], object] | None, int]


def _time_arms(
    arms: dict[str, Arm], warmups: int, samples: int, blocks: int
) -> tuple[dict[str, list[float]], list[str]]:
    """Time every arm in ``blocks`` interleaved blocks, reversing the arm order on odd blocks.

    An arm is ``(fn, prep, ops_per_call)``; samples are divided by ``ops_per_call``
    so a graph that replays several collectives reports per-collective time.
    Returns the raw samples per arm and the executed order.
    """
    if samples < 1 or blocks < 1:
        raise ValueError("samples and blocks must be positive")
    blocks = min(blocks, samples)
    # exactly ``samples`` per arm: the remainder goes one each to the first blocks
    per_block = [
        samples // blocks + (1 if b < samples % blocks else 0) for b in range(blocks)
    ]
    raw: dict[str, list[float]] = {name: [] for name in arms}
    order: list[str] = []
    names = list(arms)
    for block in range(blocks):
        for name in names if block % 2 == 0 else names[::-1]:
            fn, prep, ops = arms[name]
            block_warmups = warmups if block == 0 else min(warmups, 5)
            raw[name].extend(
                v / ops for v in _time_eager(fn, block_warmups, per_block[block], prep)
            )
            order.append(name)
    assert all(len(v) == samples for v in raw.values())
    return raw, order


def _summarize(raw: dict[str, list[float]], world: int) -> dict[str, object]:
    """Per-rank medians and the median of paired cross-rank maxima."""
    per_rank_raw: list[dict[str, list[float]]] = [{} for _ in range(world)]
    dist.all_gather_object(per_rank_raw, raw)
    summary: dict[str, object] = {}
    for name in raw:
        medians = [statistics.median(r[name]) for r in per_rank_raw]
        paired_maxima = [
            max(samples)
            for samples in zip(*(r[name] for r in per_rank_raw), strict=True)
        ]
        summary[f"{name}_us"] = round(statistics.median(paired_maxima), 1)
        summary[f"{name}_per_rank_median_us"] = [round(m, 2) for m in medians]
    summary["raw_samples_rank0_us"] = raw
    return summary


def main() -> None:
    """Entry point: correctness gates, interleaved timing, table, receipt."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="8192,32768,49152,262144,786432,1048576")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--warmups", type=int, default=50)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument(
        "--blocks", type=int, default=4, help="interleaved timing blocks per size"
    )
    ap.add_argument("--graph-ops", type=int, default=20)
    ap.add_argument("--max-size", type=int, default=2 << 20)
    ap.add_argument("--runtime-threads", type=int, default=512)
    ap.add_argument("--runtime-blocks", type=int, default=8)
    ap.add_argument(
        "--gather-rows",
        default="6,16,96",
        help="rows of a [rows, gather_cols] logits shard to all-gather",
    )
    ap.add_argument("--gather-cols", type=int, default=38720)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    rtol, atol = _tolerance(dtype, world)

    from b12x.comm import roce

    runtime = roce.AllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=args.max_size,
        max_gather_bytes=16 << 20,
        threads=args.runtime_threads,
        blocks=args.runtime_blocks,
    )
    runtime.prepare((dtype,))
    sizes = [int(s) for s in args.sizes.split(",") if int(s) <= args.max_size]

    def progress(msg: str) -> None:
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    identities: list[dict[str, object]] = [{} for _ in range(world)]
    dist.all_gather_object(
        identities,
        {"rank": rank, "hostname": os.uname().nodename, "gpu": _gpu_identity()},
    )
    progress(f"runtime ready: {runtime.stats()}")

    rows = []
    for nbytes in sizes:
        progress(f"size {nbytes}: correctness")
        numel = nbytes // torch.tensor([], dtype=dtype).element_size()
        torch.manual_seed(rank + 17)
        inp = torch.randn(numel, dtype=dtype, device=device)
        out = torch.empty_like(inp)
        expected = inp.clone()
        dist.all_reduce(expected)
        runtime.all_reduce(inp, out=out)
        torch.cuda.synchronize()
        # Oracle before timing: a mismatch stops the run.
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
        err = (out.float() - expected.float()).abs().max().item()
        scale = expected.float().abs().max().item() or 1.0
        nccl_in = inp.clone()
        progress(f"size {nbytes}: timing (interleaved)")
        roce_graph = _capture_graph(
            lambda: runtime.all_reduce(inp, out=out), args.warmups, args.graph_ops
        )
        nccl_graph_in = torch.zeros_like(inp)
        nccl_graph = _capture_graph(
            lambda: dist.all_reduce(nccl_graph_in), args.warmups, args.graph_ops
        )
        arms: dict[str, Arm] = {
            # NCCL reduces in place; reset the input untimed before every call
            # so the operands stay finite and equal to the RoCEnante operands.
            "nccl": (lambda: dist.all_reduce(nccl_in), lambda: nccl_in.copy_(inp), 1),
            # The zero input remains zero across graph replays, avoiding
            # overflow from graph_ops repeated in-place reductions.
            "nccl_graph": (nccl_graph.replay, None, args.graph_ops),
            "roce_eager": (lambda: runtime.all_reduce(inp, out=out), None, 1),
            "roce_graph": (roce_graph.replay, None, args.graph_ops),
        }
        raw, order = _time_arms(arms, args.warmups, args.samples, args.blocks)
        runtime.check_health()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
        row: dict[str, object] = {"bytes": nbytes, "order": order}
        row.update(_summarize(raw, world))
        row["ratio_nccl_over_roce_eager"] = round(
            row["nccl_us"] / row["roce_eager_us"], 3
        )
        row["ratio_nccl_over_roce_graph"] = round(
            row["nccl_us"] / row["roce_graph_us"], 3
        )
        row["ratio_nccl_graph_over_roce_graph"] = round(
            row["nccl_graph_us"] / row["roce_graph_us"], 3
        )
        row["correctness"] = {
            "max_abs_err": err,
            "rel_err": err / scale,
            "rtol": rtol,
            "atol": atol,
            "passed_before_timing": True,
            "passed_after_timing": True,
        }
        rows.append(row)
        if rank == 0:
            print(
                f"{nbytes:>9} B  nccl {row['nccl_us']:>8.1f} us  roce eager {row['roce_eager_us']:>8.1f} us"
                f"  roce graph {row['roce_graph_us']:>8.1f} us  rel_err {err / scale:.2e}",
                flush=True,
            )
        del arms, nccl_graph, roce_graph

    # all-gather of a logits shard along the last dim vs NCCL + layout copy
    gather_rows = []
    for gather_row_count in (int(r) for r in args.gather_rows.split(",") if r.strip()):
        shard = torch.randn(
            gather_row_count, args.gather_cols, dtype=dtype, device=device
        )
        if not runtime.should_all_gather(shard, -1):
            continue
        shape = [gather_row_count, args.gather_cols]
        progress(f"all-gather {shape}: correctness")
        parts = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(parts, shard)
        expected = torch.cat(parts, dim=-1)
        got = torch.empty_like(expected)
        runtime.all_gather(shard, dim=-1, out=got)
        torch.cuda.synchronize()
        if not torch.equal(got, expected):
            raise RuntimeError(f"RoCE all-gather {shape} is not bit-exact against NCCL")
        stacked = torch.empty(
            (world * gather_row_count, args.gather_cols),
            dtype=dtype,
            device=device,
        )

        def nccl_gather(
            stacked: torch.Tensor = stacked,
            shard: torch.Tensor = shard,
            n: int = gather_row_count,
            cols: int = args.gather_cols,
        ) -> torch.Tensor:
            dist.all_gather_into_tensor(stacked, shard)
            return (
                stacked.reshape(world, n, cols).movedim(0, 1).reshape(n, world * cols)
            )

        progress(f"all-gather {shape}: timing (interleaved)")
        roce_graph = _capture_graph(
            lambda: runtime.all_gather(shard, dim=-1, out=got),
            args.warmups,
            args.graph_ops,
        )
        nccl_graph = _capture_graph(nccl_gather, args.warmups, args.graph_ops)
        arms = {
            "nccl": (nccl_gather, None, 1),
            "nccl_graph": (nccl_graph.replay, None, args.graph_ops),
            "roce_eager": (lambda: runtime.all_gather(shard, dim=-1, out=got), None, 1),
            "roce_graph": (roce_graph.replay, None, args.graph_ops),
        }
        raw, order = _time_arms(arms, args.warmups, args.samples, args.blocks)
        runtime.check_health()
        torch.cuda.synchronize()
        exact_after = bool(torch.equal(got, expected))
        if not exact_after:
            raise RuntimeError(
                f"RoCE all-gather {shape} diverged from NCCL during timing"
            )
        row = {
            "rows": gather_row_count,
            "cols": args.gather_cols,
            "shard_bytes": shard.numel() * shard.element_size(),
            "order": order,
        }
        row.update(_summarize(raw, world))
        row["ratio_nccl_over_roce_eager"] = round(
            row["nccl_us"] / row["roce_eager_us"], 3
        )
        row["ratio_nccl_over_roce_graph"] = round(
            row["nccl_us"] / row["roce_graph_us"], 3
        )
        row["ratio_nccl_graph_over_roce_graph"] = round(
            row["nccl_graph_us"] / row["roce_graph_us"], 3
        )
        row["correctness"] = {
            "exact_before_timing": True,
            "exact_after_timing": exact_after,
        }
        gather_rows.append(row)
        if rank == 0:
            print(
                f"all-gather {shape}  nccl+copy {row['nccl_us']:>8.1f} us"
                f"  roce eager {row['roce_eager_us']:>8.1f} us  roce graph {row['roce_graph_us']:>8.1f} us  exact=True",
                flush=True,
            )
        del arms, nccl_graph, roce_graph

    stats = runtime.stats()
    if rank == 0:
        doc = {
            "schema": "b12x.comm.roce.oneshot.benchmark",
            "version": 3,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": " ".join([sys.executable, *sys.argv]),
            "launcher_env": {
                k: os.environ.get(k, "")
                for k in (
                    "WORLD_SIZE",
                    "MASTER_ADDR",
                    "NCCL_IB_HCA",
                    "NCCL_IB_GID_INDEX",
                    "B12X_ROCE_HCA",
                    "B12X_ROCE_GID_INDEX",
                )
            },
            "benchmark_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            **_source_state(),
            "worktree": str(REPO_ROOT),
            "world_size": world,
            "dtype": args.dtype,
            "ranks": identities,
            "runtime": stats,
            "measurement": {
                "warmups": args.warmups,
                "samples": args.samples,
                "blocks": args.blocks,
                "graph_ops": args.graph_ops,
                "runtime_threads": args.runtime_threads,
                "runtime_blocks": args.runtime_blocks,
                "timing": "CUDA events around one call on the caller's stream; the graph arm replays graph_ops collectives per call and reports per-collective time",
                "summary": "*_us is the median of paired per-sample maxima across ranks",
                "ratio_direction": "ratio_nccl_over_roce_* = nccl_us / roce_*_us and ratio_nccl_graph_over_roce_graph = nccl_graph_us / roce_graph_us; above 1 means RoCEnante is faster",
                "ordering": "arms run in interleaved blocks, order reversed on odd blocks; the executed order is recorded per row",
                "target_path": "RoceOneshotAllReduce.all_reduce / all_gather, the entry points the vLLM adapter dispatches to; graph replay is the decode path",
            },
            "rows": rows,
            "all_gather": gather_rows,
        }
        if args.output:
            dump_compact_json(doc, Path(args.output))
            print(f"wrote {args.output}")
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
