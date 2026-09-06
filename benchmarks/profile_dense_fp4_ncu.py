#!/usr/bin/env python3
"""Profile one prequantized dense NVFP4 graph replay with Nsight Compute."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.benchmark_dense_gemm import (
    REFERENCE_BACKEND,
    capture_graph_replay,
    make_quantized_operand,
)
from b12x._lib.dense_gemm import dense_gemm
from b12x._lib.utils import convert_sf_from_mma_layout
from flashinfer.gemm import mm_fp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("b12x", "flashinfer"), required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(42)
    a_packed, a_sf, a_gs = make_quantized_operand(args.m, args.k)
    b_packed, b_sf, b_gs = make_quantized_operand(args.n, args.k)
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)

    if args.arm == "b12x":
        out = torch.empty((args.m, args.n, 1), device="cuda", dtype=torch.bfloat16)

        def launch() -> None:
            dense_gemm(
                (a_packed, a_sf),
                (b_packed, b_sf),
                alpha=alpha,
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16",
                sf_vec_size=16,
                out=out,
                mma_tiler_mn=(args.tile_m, args.tile_n),
            )

    else:
        a_fp4 = a_packed[:, :, 0].contiguous()
        b_fp4 = b_packed[:, :, 0].contiguous()
        a_sf_linear = convert_sf_from_mma_layout(
            a_sf, m=args.m, k=args.k, num_groups=1
        )
        b_sf_linear = convert_sf_from_mma_layout(
            b_sf, m=args.n, k=args.k, num_groups=1
        )
        out = torch.empty((args.m, args.n), device="cuda", dtype=torch.bfloat16)

        def launch() -> None:
            mm_fp4(
                a_fp4,
                b_fp4.T,
                a_sf_linear,
                b_sf_linear.T,
                alpha,
                torch.bfloat16,
                out,
                block_size=16,
                use_8x4_sf_layout=False,
                backend=REFERENCE_BACKEND,
                use_nvfp4=True,
            )

    replay = capture_graph_replay(launch)
    for _ in range(3):
        replay()
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    replay()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()


if __name__ == "__main__":
    main()
