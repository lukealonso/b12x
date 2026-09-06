#!/usr/bin/env python3
"""Offline graph-replay autotuner for K128 block-scaled FP8 dense plans.

This calibration is deliberately separate from MXFP8: values use ordinary
E4M3 MMA, each K128 block is accumulated before compact FP32 activation and
weight scales are applied, and split-K changes the BF16 reduction order.
Candidates are checked only against a trusted direct b12x plan and timed with
captured graphs in balanced order. Because K128 scales are applied between
FP32 accumulation segments, tile and unroll changes may legitimately alter
floating-point association even without split-K; every candidate therefore
uses a finite/nonzero cosine gate rather than byte equality. Reference backends
are final acceptance only.
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cutlass
import torch

from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from benchmarks.dense_autotune_common import (
    bench_events,
    capture_graph_replay,
    cosine_similarity,
)
from b12x._lib.dense_gemm import (
    DenseGemmKernel,
    _select_default_dense_gemm_plan,
    dense_gemm,
)
from b12x._lib.utils import get_num_sm


BASELINE_COSINE_THRESHOLD = 0.9999
PRODUCTION_TILES = (
    (16, 64),
    (16, 128),
    (32, 64),
    (32, 128),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    note: str


@dataclass(frozen=True)
class Candidate:
    tile_mn: tuple[int, int]
    split_k_slices: int
    large_m_unroll: bool | None

    def label(self) -> str:
        unroll = "auto" if self.large_m_unroll is None else int(self.large_m_unroll)
        return (
            f"{self.tile_mn[0]}x{self.tile_mn[1]}/BK128/"
            f"split{self.split_k_slices}/unroll{unroll}"
        )


@dataclass
class CandidateRun:
    candidate: Candidate
    cos: float
    max_abs: float
    samples_ms: list[float]


QWEN38_TP_SHAPES = tuple(
    shape
    for tp in (1, 2, 4, 8)
    for shape in (
        Shape(
            f"qwen38_27b_gate_or_up_tp{tp}",
            17408 // tp,
            5120,
            f"column-parallel dense SwiGLU gate or up, TP={tp}",
        ),
        Shape(
            f"qwen38_27b_down_tp{tp}",
            5120,
            17408 // tp,
            f"row-parallel dense SwiGLU down, TP={tp}",
        ),
    )
)

COMMON_SHAPES = (
    Shape("deepseek_wo_b", 4096, 4096, "WO-B projection"),
    Shape("deepseek_q_b", 16384, 1024, "q_b projection"),
    Shape("nemotron_shared_down", 4096, 5376, "shared expert down"),
    Shape("glm5_dense_down", 6144, 1536, "GLM dense down"),
)


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def _parse_tile_list(raw: str) -> list[tuple[int, int]]:
    try:
        values = [
            tuple(int(axis) for axis in item.lower().split("x"))
            for item in raw.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tiles must use MxN syntax") from exc
    if not values or any(len(tile) != 2 or min(tile) <= 0 for tile in values):
        raise argparse.ArgumentTypeError("tiles must use positive MxN syntax")
    return [(tile[0], tile[1]) for tile in values]


def _parse_unroll_list(raw: str) -> list[bool | None]:
    aliases: dict[str, bool | None] = {
        "auto": None,
        "0": False,
        "false": False,
        "1": True,
        "true": True,
    }
    try:
        values = [aliases[item.strip().lower()] for item in raw.split(",") if item.strip()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "large-M unroll values must be auto, 0/false, or 1/true"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("large-M unroll list must not be empty")
    return values


def _shapes(args: argparse.Namespace) -> list[Shape]:
    if args.n is not None:
        return [Shape(args.name, args.n, args.k, "explicit CLI shape")]
    if args.n_list is not None:
        return [
            Shape(f"grid_n{n}_k{k}", n, k, "CLI Cartesian grid")
            for n in args.n_list
            for k in args.k_list
        ]
    if args.shape_set == "common":
        return list(COMMON_SHAPES)
    if args.shape_set == "all":
        return list(QWEN38_TP_SHAPES + COMMON_SHAPES)
    return list(QWEN38_TP_SHAPES)


def _plan_candidates(
    *,
    m: int,
    tile_mn_list: list[tuple[int, int]],
    split_k_list: list[int],
    large_m_unroll_list: list[bool | None],
) -> list[Candidate]:
    slices = split_k_list if m <= 8 else [1]
    return [
        Candidate(tile, split_k, large_m_unroll)
        for tile in tile_mn_list
        for split_k in slices
        for large_m_unroll in large_m_unroll_list
    ]


def _make_operands(
    m: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 4).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 4).to(
        torch.float8_e4m3fn
    )
    a_scale = (torch.rand(m, k // 128, device="cuda") * 0.01 + 0.005).contiguous()
    b_scale = (
        torch.rand(n // 128, k // 128, device="cuda") * 0.01 + 0.005
    ).contiguous()
    return a, a_scale, b, b_scale


def _baseline(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
    plan: object,
) -> torch.Tensor:
    out = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    dense_gemm(
        (a.view(m, k, 1), a_scale),
        (b.view(n, k, 1), b_scale),
        out=out,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype="bfloat16",
        sf_vec_size=128,
        block_fp8=True,
        expected_m=m,
        mma_tiler_mn=plan.mma_tiler_mn,
        _split_k_slices_override=1,
    )
    torch.cuda.synchronize()
    return out[:, :, 0].clone()


def _check_candidate(
    output: torch.Tensor,
    baseline: torch.Tensor,
    *,
    shape: Shape,
    m: int,
    candidate: Candidate,
) -> tuple[float, float]:
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(
            f"non-finite output for {shape.name} M={m} {candidate.label()}"
        )
    if not bool(torch.count_nonzero(output).item()):
        raise RuntimeError(f"zero output for {shape.name} M={m} {candidate.label()}")
    max_abs = (output.float() - baseline.float()).abs().max().item()
    cos = cosine_similarity(output, baseline)
    if cos < BASELINE_COSINE_THRESHOLD:
        raise RuntimeError(
            f"correctness failure for {shape.name} M={m} {candidate.label()}: "
            f"max_abs={max_abs}, cos={cos}"
        )
    return cos, max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-set", choices=("qwen38-tp", "common", "all"), default="qwen38-tp"
    )
    parser.add_argument("--name", default="custom")
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n-list", type=_parse_int_list)
    parser.add_argument("--k-list", type=_parse_int_list)
    parser.add_argument(
        "--m-list",
        type=_parse_int_list,
        default=_parse_int_list("1,6,64,128,2048,4096"),
    )
    parser.add_argument(
        "--tile-mn-list", type=_parse_tile_list, default=list(PRODUCTION_TILES)
    )
    parser.add_argument(
        "--split-k-list", type=_parse_int_list, default=_parse_int_list("1,2,4")
    )
    parser.add_argument(
        "--large-m-unroll-list",
        type=_parse_unroll_list,
        default=_parse_unroll_list("auto"),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results.dense.block_fp8_plan.tsv"),
    )
    parser.add_argument("--no-l2-flush", action="store_true")
    args = parser.parse_args()

    if (args.n is None) != (args.k is None):
        parser.error("--n and --k must be supplied together")
    if (args.n_list is None) != (args.k_list is None):
        parser.error("--n-list and --k-list must be supplied together")
    if args.n is not None and args.n_list is not None:
        parser.error("explicit --n/--k and --n-list/--k-list are mutually exclusive")
    if args.n is not None and (args.n <= 0 or args.k <= 0):
        parser.error("--n and --k must be positive")
    if args.n_list is not None and (
        any(n <= 0 for n in args.n_list) or any(k <= 0 for k in args.k_list)
    ):
        parser.error("--n-list and --k-list values must be positive")
    shapes = _shapes(args)
    if any(shape.n % 128 or shape.k % 128 for shape in shapes):
        parser.error("all block-FP8 N/K dimensions must be divisible by 128")
    if any(m <= 0 for m in args.m_list):
        parser.error("--m-list values must be positive")
    if any(slices not in (1, 2, 4) for slices in args.split_k_list):
        parser.error("--split-k-list is restricted to 1,2,4")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative; iters and repeats must be positive")

    torch.empty(1, device="cuda")
    sm_count = get_num_sm(torch.device("cuda"))
    l2_flush = make_l2_flush_fn(enabled=not args.no_l2_flush, bytes_hint=0)
    l2_bytes = resolve_l2_flush_bytes(0) if l2_flush is not None else 0
    metadata = (
        f"gpu={torch.cuda.get_device_name()} sm_count={sm_count} "
        f"arch={os.getenv('CUTE_DSL_ARCH', '')} cutlass={cutlass.__version__} "
        f"warmup={args.warmup} iters={args.iters} repeats={args.repeats} "
        f"l2_flush_bytes={l2_bytes} calibration=b12x-only graph_replay=1 "
        "scale_contract=fp32_k128"
    )
    print(metadata)
    print(
        f"shapes={len(shapes)} M={args.m_list} tiles={args.tile_mn_list} "
        f"splitK={args.split_k_list} largeMUnroll={args.large_m_unroll_list}"
    )
    columns = (
        "shape\tm\tn\tk\ttile_m\ttile_n\ttile_k\twork_tiles\tsm_count\t"
        "split_k_slices\tlarge_m_unroll\tstatus\tmedian_us\tmin_us\tmax_us\t"
        "cos\tmax_abs\tsamples_us\n"
    )

    with args.output.open("w", encoding="utf-8") as output:
        output.write(f"# {metadata}\n")
        output.write(columns)
        for shape in shapes:
            print(f"\n### {shape.name}: Mx{shape.n}x{shape.k} ({shape.note})")
            for m in args.m_list:
                torch.manual_seed(42 + m + shape.n + shape.k)
                a, a_scale, b, b_scale = _make_operands(m, shape.n, shape.k)
                default_plan = _select_default_dense_gemm_plan(
                    m,
                    shape.n,
                    shape.k,
                    sm_count,
                    is_mxfp8=True,
                    block_fp8=True,
                    expected_m=m,
                )
                baseline = _baseline(
                    a,
                    a_scale,
                    b,
                    b_scale,
                    m=m,
                    n=shape.n,
                    k=shape.k,
                    plan=default_plan,
                )
                candidates = _plan_candidates(
                    m=m,
                    tile_mn_list=args.tile_mn_list,
                    split_k_list=args.split_k_list,
                    large_m_unroll_list=args.large_m_unroll_list,
                )
                runnable: list[Candidate] = []
                for candidate in candidates:
                    tile_m, tile_n = candidate.tile_mn
                    work_tiles = ((m + tile_m - 1) // tile_m) * (
                        (shape.n + tile_n - 1) // tile_n
                    )
                    prefix = (
                        f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t{tile_m}\t"
                        f"{tile_n}\t128\t{work_tiles}\t{sm_count}\t"
                        f"{candidate.split_k_slices}\t{candidate.large_m_unroll}\t"
                    )
                    if shape.k % (128 * candidate.split_k_slices):
                        output.write(prefix + "not_divisible\t\t\t\t\t\t\n")
                        continue
                    if candidate.split_k_slices > 1 and (
                        m > 8 or m > candidate.tile_mn[0]
                    ):
                        output.write(prefix + "unsupported_split\t\t\t\t\t\t\n")
                        continue
                    if not DenseGemmKernel.can_implement(
                        cutlass.Float8E4M3FN,
                        cutlass.Float32,
                        128,
                        cutlass.BFloat16,
                        candidate.tile_mn,
                        (1, 1),
                        shape.n,
                        shape.k,
                        1,
                        "k",
                        "k",
                        "n",
                        load_path="tma",
                        swap_ab=False,
                        block_fp8=True,
                    ):
                        output.write(prefix + "unsupported\t\t\t\t\t\t\n")
                        continue
                    runnable.append(candidate)

                measured: dict[Candidate, CandidateRun] = {}
                failed: set[Candidate] = set()
                for repeat in range(args.repeats):
                    ordered = runnable if repeat % 2 == 0 else list(reversed(runnable))
                    for candidate in ordered:
                        if candidate in failed:
                            continue
                        out = torch.empty(
                            (m, shape.n, 1), device="cuda", dtype=torch.bfloat16
                        )

                        def launch(
                            out: torch.Tensor = out,
                            candidate: Candidate = candidate,
                        ) -> None:
                            dense_gemm(
                                (a.view(m, shape.k, 1), a_scale),
                                (b.view(shape.n, shape.k, 1), b_scale),
                                out=out,
                                ab_dtype="float8_e4m3fn",
                                sf_dtype="float32",
                                c_dtype="bfloat16",
                                sf_vec_size=128,
                                block_fp8=True,
                                expected_m=m,
                                mma_tiler_mn=candidate.tile_mn,
                                _split_k_slices_override=candidate.split_k_slices,
                                _large_m_unroll_override=candidate.large_m_unroll,
                            )

                        replay = None
                        try:
                            replay = capture_graph_replay(launch)
                            replay()
                            torch.cuda.synchronize()
                            cos, max_abs = _check_candidate(
                                out[:, :, 0],
                                baseline,
                                shape=shape,
                                m=m,
                                candidate=candidate,
                            )
                            samples = bench_events(
                                replay,
                                warmup=args.warmup,
                                iters=args.iters,
                                l2_flush=l2_flush,
                            )
                        except Exception as exc:
                            message = str(exc).splitlines()[0].replace("\t", " ")[:160]
                            print(f"  M={m:<4} {candidate.label()} failed: {message}")
                            tile_m, tile_n = candidate.tile_mn
                            work_tiles = ((m + tile_m - 1) // tile_m) * (
                                (shape.n + tile_n - 1) // tile_n
                            )
                            output.write(
                                f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                                f"{tile_m}\t{tile_n}\t128\t{work_tiles}\t{sm_count}\t"
                                f"{candidate.split_k_slices}\t"
                                f"{candidate.large_m_unroll}\tfailed:{message}\t"
                                "\t\t\t\t\t\n"
                            )
                            failed.add(candidate)
                            measured.pop(candidate, None)
                        else:
                            run = measured.setdefault(
                                candidate, CandidateRun(candidate, cos, max_abs, [])
                            )
                            run.samples_ms.extend(samples)
                            run.cos = min(run.cos, cos)
                            run.max_abs = max(run.max_abs, max_abs)
                        finally:
                            del replay
                            del launch
                            del out
                            gc.collect()
                            torch.cuda.empty_cache()

                scored: list[tuple[float, CandidateRun]] = []
                for run in measured.values():
                    samples_us = [sample * 1000 for sample in run.samples_ms]
                    median_us = statistics.median(samples_us)
                    scored.append((median_us, run))
                    tile_m, tile_n = run.candidate.tile_mn
                    work_tiles = ((m + tile_m - 1) // tile_m) * (
                        (shape.n + tile_n - 1) // tile_n
                    )
                    sample_text = ",".join(f"{sample:.3f}" for sample in samples_us)
                    output.write(
                        f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t{tile_m}\t"
                        f"{tile_n}\t128\t{work_tiles}\t{sm_count}\t"
                        f"{run.candidate.split_k_slices}\t"
                        f"{run.candidate.large_m_unroll}\tpass\t{median_us:.3f}\t"
                        f"{min(samples_us):.3f}\t{max(samples_us):.3f}\t"
                        f"{run.cos:.10f}\t{run.max_abs:.8f}\t{sample_text}\n"
                    )
                    print(
                        f"  M={m:<4} {run.candidate.label():<34} "
                        f"{median_us:8.2f} us cos={run.cos:.8f}"
                    )
                if scored:
                    winner_us, winner = min(scored, key=lambda item: item[0])
                    default_us = next(
                        (
                            value
                            for value, item in scored
                            if item.candidate.tile_mn == default_plan.mma_tiler_mn
                            and item.candidate.split_k_slices == 1
                            and item.candidate.large_m_unroll is None
                        ),
                        None,
                    )
                    speedup = (
                        f" ({default_us / winner_us:.3f}x vs default/direct)"
                        if default_us is not None
                        else ""
                    )
                    print(
                        f"  -> M={m:<4} winner {winner.candidate.label()} "
                        f"{winner_us:.2f} us{speedup}"
                    )
    print(f"\nraw_results={args.output}")


if __name__ == "__main__":
    main()
