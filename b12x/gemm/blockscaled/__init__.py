"""One-shot dense block-scaled GEMM: ``C = (A·SFA) @ (B·SFB)``.

One-shot functional op over the shared SM120 warp-MMA engine (no TMEM, no
tcgen05, no 2-CTA). Recipes: NVFP4 (Float4E2M1 values, e4m3 scales, vec 16),
MXFP4 (e8m0 scales, vec 32), MXFP8 (e4m3 values, e8m0 scales, vec 32), and
tensor-scaled FP8. ``mm`` accepts raw ``(values, scales)`` operand pairs or a
weight returned by ``pack_weight``. A packed MXFP8 weight accepts either a
BF16/FP16 activation or a prequantized ``(values, scales)`` pair with compact
row-major or F8_128x4-swizzled UE8M0 scales. Pass swizzled storage flattened
(or as its native 6D view); a 2D ``[M,K/32]`` scale is interpreted as compact.
``expected_m`` is a DeepGEMM-style regime hint (decode vs prefill tiles).

On SM120/SM121, ``w4a16`` and ``w8a16`` accept BF16 activations and directly read
the same swizzled weight scales as NVFP4/MXFP8 GEMM. Packed BF16-input calls
accept ``mode='auto'|'a16'|'quantized'``. AUTO prefers an autotuned profile and
uses a conservative geometry heuristic for uncovered queries. A16 is a specialization
of ``DenseGemmKernel``, with BF16 warp MMA and inline weight dequantization.
Forced A16 also uses qualified profile tile configurations when available,
and retains BF16 activations at unprofiled row counts with the default tile.
Its packed BF16 conversions require a PTX 9.2 toolchain (CUDA 13.3).

NVFP4 packed weights require ``pack_weight(..., recipe='nvfp4', global_scale=g,
global_scale_kind='multiplier')``. ``'reciprocal'`` accepts a weight quantizer
multiplier without creating another tensor. NVFP4 quantized activation calls
also require ``activation_global_scale``, the activation quantizer multiplier.
Global scales must be finite and strictly positive; reconstructed weights must
fit BF16. Quantized activation execution requires K divisible by 128 and N by 8.

For allocation-stable capture, allocate BF16 ``out`` and uint8 ``workspace``
using ``workspace_size(weight, max_tokens)`` and call ``prewarm`` before capture.
Pass those buffers to ``mm``; concurrent calls need disjoint buffers. Prewarming
does not retain scratch or weight copies. Graph replay retains the captured
precision route. These operations do not expose a public planning API.

Example:
    from b12x.gemm import blockscaled

    out = blockscaled.mm(
        (a_fp4, a_sf), (b_fp4, b_sf),
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn", sf_vec_size=16,
        c_dtype="bfloat16", alpha=alpha, expected_m=m,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="blockscaled",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "Weight",
        "NVFP4LinearWeight",
        "mm",
        "mm_mxfp4",
        "mm_nvfp4",
        "mm_block_fp8",
        "pack_weight",
        "prewarm",
        "is_supported",
        "w4a16",
        "w8a16",
        "workspace_size",
    ),
    dtypes=("bf16", "fp16", "fp32", "fp8_e4m3", "fp4_e2m1"),
    recipes=("nvfp4", "mxfp4", "mxfp8"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/dense.py",),
    ),
    test_path="tests/gemm/test_blockscaled.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Weight,
        NVFP4LinearWeight,
        is_supported,
        mm,
        mm_block_fp8,
        mm_mxfp4,
        mm_nvfp4,
        pack_weight,
        prewarm,
        w4a16,
        w8a16,
        workspace_size,
    )

install_lazy_api(globals(), META)
