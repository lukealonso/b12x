#!/usr/bin/env python3
"""Digest and time the packed MXFP8 linear at the served Kimi-K3 shapes.

Runs ``b12x.gemm.mxfp8_linear.mm`` (activation quantization + dense GEMM,
``expected_m`` = live rows, as the serving path calls it) on one GPU for the
per-rank TP9 projections of Kimi-K3 and records, per shape and row count, the
SHA-256 of the BF16 output bytes and the CUDA-graph replay time. Inputs and
weights are generated on the host from a fixed seed, so two trees run on the
same GPU produce comparable digests: equal digests prove bit-identical
outputs across a planner change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
)

# Per-rank (K, N) of the served Kimi-K3 TP9 MXFP8 dense linears (hidden 7,168).
SHAPES = {
    "kda_in_proj_qkv": (7168, 4608),
    "kda_o_proj": (1536, 7168),
    "mla_fused_qkv_a": (7168, 2112),
    "mla_q_b": (1536, 2112),
    "mla_o_proj": (1408, 7168),
    "shared_gate_up": (7168, 1536),
    "shared_down": (768, 7168),
    "dense0_gate_up": (7168, 7680),
    "dense0_down": (3840, 7168),
}


def _quantize_modelopt_mxfp8_rows(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ModelOpt MXFP8 rows: E4M3 values with one UE8M0 scale per 32 columns."""
    rows, width = map(int, source.shape)
    chunks = width // 32
    blocked = source.to(torch.float32).reshape(rows, chunks, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(max_abs > 0.0, max_abs / 448.0, torch.ones_like(max_abs))
    scale_exp = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale_u8 = (scale_exp + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    values = (
        (blocked / scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, width)
        .contiguous()
    )
    return values, scale_u8.contiguous()


def _host_inputs(
    k: int, n: int, m: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = (
        torch.randn((m, k), generator=generator, dtype=torch.float32) / 4
    ).to(torch.bfloat16)
    weight_bf16 = (
        torch.randn((n, k), generator=generator, dtype=torch.float32) / 8
    ).to(torch.bfloat16)
    weight, weight_scale = _quantize_modelopt_mxfp8_rows(weight_bf16)
    return source, weight, weight_scale


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--shapes",
        default="all",
        help="comma-separated shape names (default: all of " + ",".join(SHAPES) + ")",
    )
    parser.add_argument("--m-values", default="1,8,830,4608")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evict L2 before every timed replay (default: enabled)",
    )
    args = parser.parse_args()

    from b12x.gemm import mxfp8_linear

    device = require_sm120()
    names = list(SHAPES) if args.shapes == "all" else args.shapes.split(",")
    m_values = [int(value) for value in args.m_values.split(",")]
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2)
    records = []
    for name in names:
        k, n = SHAPES[name]
        for m in m_values:
            source, weight, weight_scale = _host_inputs(k, n, m, args.seed)
            source = source.to(device)
            packed = mxfp8_linear.pack_weight(weight.to(device), weight_scale.to(device))

            def run() -> torch.Tensor:
                return mxfp8_linear.mm(source, packed, expected_m=m)

            eager = run()
            torch.cuda.synchronize()
            graph = capture_cuda_graph(run, warmup=args.warmup)
            replay_us = bench_cuda_graph(
                graph, replays=args.iters, l2_flush=l2_flush
            )["replay_us"]
            again = run()
            torch.cuda.synchronize()
            record = {
                "shape": name,
                "k": k,
                "n": n,
                "m": m,
                "output_sha256": _digest(eager),
                "repeat_identical": bool(torch.equal(eager, again)),
                "graph_replay_us": {
                    "median": round(statistics.median(replay_us), 2),
                    "min": round(min(replay_us), 2),
                    "p90": round(sorted(replay_us)[int(0.9 * (len(replay_us) - 1))], 2),
                },
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            del graph
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": args.seed,
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
