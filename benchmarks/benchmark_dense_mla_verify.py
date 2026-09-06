#!/usr/bin/env python3
"""Compare deployed, row-specialized, and fused-q4 dense MLA plans."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from b12x.attention import dense_mla

FP8 = torch.float8_e4m3fn


@dataclass
class Arm:
    name: str
    plan: dense_mla.Plan
    binding: dense_mla.Binding
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    lse: torch.Tensor
    oracle_cosine: float


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cache-tokens", type=int, default=131072)
    parser.add_argument("--planned-cache-tokens", type=int, default=131072)
    parser.add_argument("--page-size", type=int, default=1536)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--query-len", type=int, default=4)
    parser.add_argument("--deployed-max-rows", type=int, default=28)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sparse-stride", type=int, default=1)
    parser.add_argument("--sparse-min-tokens", type=int, default=32768)
    parser.add_argument("--sparse-sink-chunks", type=int, default=8)
    parser.add_argument("--sparse-recent-chunks", type=int, default=64)
    parser.add_argument("--sparse-refresh-interval", type=int, default=0)
    parser.add_argument("--split-candidates", default="")
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _scratch(plan: dense_mla.Plan) -> torch.Tensor:
    (spec,) = plan.scratch_specs()
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def _make_plan(
    *,
    name: str,
    device: torch.device,
    batch: int,
    heads: int,
    query_len: int,
    page_size: int,
    planned_cache_tokens: int,
    deployed_max_rows: int,
    sparse_kwargs: dict[str, int],
    max_splits: int | None,
) -> dense_mla.Plan:
    total_q = batch * query_len
    planned_pages = (planned_cache_tokens + page_size - 1) // page_size
    if name == "deployed":
        mode = "decode"
        max_total_q = deployed_max_rows
        max_batch = deployed_max_rows
        per_query_lens = False
    elif name == "row_specialized":
        mode = "decode"
        max_total_q = total_q
        max_batch = total_q
        per_query_lens = False
    elif name in ("tiled", "tiled_sparse") or name.startswith("tiled_s"):
        mode = "verify"
        max_total_q = total_q
        max_batch = batch
        per_query_lens = True
    else:
        raise ValueError(f"unknown benchmark arm: {name}")
    return dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode=mode,
            kv_dtype=FP8,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_cache_tokens=planned_cache_tokens,
            max_page_table_width=planned_pages,
            num_cache_pages=torch.iinfo(torch.int32).max,
            use_cuda_graph=True,
            uses_query_cache_seqlens=per_query_lens,
            budget=(
                dense_mla.Budget(max_splits=max_splits)
                if max_splits is not None
                else None
            ),
            **(sparse_kwargs if name == "tiled_sparse" else {}),
        )
    )


def _make_arm(
    *,
    name: str,
    q: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    query_cache_lens: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    planned_cache_tokens: int,
    deployed_max_rows: int,
    sparse_kwargs: dict[str, int],
    max_splits: int | None = None,
) -> Arm:
    total_q, heads = int(q.shape[0]), int(q.shape[1])
    batch, query_len = int(page_table.shape[0]), total_q // int(page_table.shape[0])
    plan = _make_plan(
        name=name,
        device=q.device,
        batch=batch,
        heads=heads,
        query_len=query_len,
        page_size=int(cache.shape[1]),
        planned_cache_tokens=planned_cache_tokens,
        deployed_max_rows=deployed_max_rows,
        sparse_kwargs=sparse_kwargs,
        max_splits=max_splits,
    )
    output = torch.empty(
        total_q,
        heads,
        512,
        dtype=torch.bfloat16,
        device=q.device,
    )
    tiled = name in ("tiled", "tiled_sparse") or name.startswith("tiled_s")
    if tiled:
        arm_table = page_table
        cache_lens = query_cache_lens.view(batch, query_len)[:, -1].contiguous()
        cu_seqlens_q = torch.arange(
            0,
            total_q + 1,
            query_len,
            dtype=torch.int32,
            device=q.device,
        )
        per_query_lens = query_cache_lens
    else:
        arm_table = (
            page_table[:, None, :]
            .expand(-1, query_len, -1)
            .reshape(total_q, -1)
            .contiguous()
        )
        cache_lens = query_cache_lens
        cu_seqlens_q = torch.arange(
            total_q + 1,
            dtype=torch.int32,
            device=q.device,
        )
        per_query_lens = None
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=arm_table,
        cache_seqlens=cache_lens,
        query_cache_seqlens=per_query_lens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    dense_mla.compile(binding=binding)
    actual, lse = dense_mla.run(binding=binding)
    expected, expected_lse = dense_mla.reference(
        q,
        cache,
        arm_table,
        cache_lens,
        cu_seqlens_q,
        query_cache_seqlens=per_query_lens,
        q_scale=q_scale,
        kv_scale=kv_scale,
        **(sparse_kwargs if name == "tiled_sparse" else {}),
    )
    torch.cuda.synchronize(q.device)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    if float(cosine) <= 0.999 or not torch.isfinite(lse).all():
        raise RuntimeError(f"{name} correctness failed: cosine={float(cosine):.8f}")
    torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dense_mla.run(binding=binding)
    return Arm(name, plan, binding, graph, actual, lse, float(cosine))


def main() -> None:
    args = _arguments()
    if args.query_len != 4:
        raise SystemExit("the fused verification benchmark requires --query-len 4")
    if args.batch <= 0 or args.batch * args.query_len > args.deployed_max_rows:
        raise SystemExit("batch/query rows exceed the deployed plan capacity")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((12, 0), (12, 1)):
        raise SystemExit("SM120/SM121 is required")
    torch.manual_seed(args.seed)
    pages_per_request = (args.cache_tokens + args.page_size - 1) // args.page_size
    total_pages = args.batch * pages_per_request
    total_q = args.batch * args.query_len
    q_float = torch.randn(total_q, args.heads, 576, device=device) * 0.1
    cache_float = torch.randn(total_pages, args.page_size, 576, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.arange(
        total_pages,
        dtype=torch.int32,
        device=device,
    ).view(args.batch, pages_per_request)
    one_request_lens = torch.arange(
        args.cache_tokens - args.query_len + 1,
        args.cache_tokens + 1,
        dtype=torch.int32,
        device=device,
    )
    query_cache_lens = one_request_lens.repeat(args.batch)
    sparse_kwargs = {
        "sparse_stride": args.sparse_stride,
        "sparse_min_tokens": args.sparse_min_tokens,
        "sparse_sink_chunks": args.sparse_sink_chunks,
        "sparse_recent_chunks": args.sparse_recent_chunks,
        "sparse_refresh_interval": args.sparse_refresh_interval,
    }
    arm_names = ["deployed", "row_specialized", "tiled"]
    split_candidates = [
        int(value) for value in args.split_candidates.split(",") if value.strip()
    ]
    arm_names.extend(f"tiled_s{value}" for value in split_candidates)
    if args.sparse_stride > 1:
        arm_names.append("tiled_sparse")
    arms = [
        _make_arm(
            name=name,
            q=q,
            cache=cache,
            page_table=page_table,
            query_cache_lens=query_cache_lens,
            q_scale=q_scale,
            kv_scale=kv_scale,
            planned_cache_tokens=args.planned_cache_tokens,
            deployed_max_rows=args.deployed_max_rows,
            sparse_kwargs=sparse_kwargs,
            max_splits=(
                int(name.removeprefix("tiled_s"))
                if name.startswith("tiled_s") and name != "tiled_sparse"
                else None
            ),
        )
        for name in arm_names
    ]
    for _ in range(args.warmup):
        for arm in arms:
            arm.graph.replay()
    torch.cuda.synchronize(device)

    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    timings: dict[str, list[float]] = {arm.name: [] for arm in arms}
    for sample in range(args.samples):
        ordered = arms[sample % len(arms) :] + arms[: sample % len(arms)]
        for arm in ordered:
            flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            arm.graph.replay()
            end.record()
            end.synchronize()
            timings[arm.name].append(float(start.elapsed_time(end)))

    medians = {name: statistics.median(values) for name, values in timings.items()}
    baseline = medians["deployed"]
    tiled_output = next(arm.output for arm in arms if arm.name == "tiled")
    props = torch.cuda.get_device_properties(device)
    result = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "source_commit": os.environ.get(
            "B12X_BENCHMARK_SOURCE_REVISION", _git("rev-parse", "HEAD")
        ),
        "source_tree": os.environ.get("B12X_BENCHMARK_SOURCE_TREE", _git("write-tree")),
        "source_status": os.environ.get(
            "B12X_BENCHMARK_SOURCE_STATUS", _git("status", "--short")
        ),
        "device": props.name,
        "device_uuid": f"GPU-{props.uuid}",
        "capability": list(capability),
        "multiprocessor_count": int(props.multi_processor_count),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch": args.batch,
        "cache_tokens": args.cache_tokens,
        "planned_cache_tokens": args.planned_cache_tokens,
        "page_size": args.page_size,
        "heads": args.heads,
        "query_len": args.query_len,
        "samples": args.samples,
        "cold_l2_flush_bytes": int(flush.numel()),
        "arms": {
            arm.name: {
                "query_tile": arm.plan.query_tile,
                "num_splits": arm.plan.num_splits,
                "median_ms": medians[arm.name],
                "speedup_vs_deployed": baseline / medians[arm.name],
                "oracle_cosine": arm.oracle_cosine,
                "cosine_vs_tiled": float(
                    torch.nn.functional.cosine_similarity(
                        arm.output.float().flatten(),
                        tiled_output.float().flatten(),
                        dim=0,
                    )
                ),
                "raw_ms": timings[arm.name],
            }
            for arm in arms
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
