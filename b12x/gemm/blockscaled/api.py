"""Public surface for gemm.blockscaled (docs in the op ``__init__``)."""

from __future__ import annotations

import torch

from ..._lib.gating import default_is_supported
from ._linear import (
    Weight,
    blockscaled_mm as mm,
    mxfp8_linear_pair as mm_pair,
    pack_weight,
    prewarm,
)
from . import META
from ._a16 import NVFP4LinearWeight, w4a16, w8a16, workspace_size


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


def _output_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"block-scaled output must be bf16/fp16, got {dtype}")


def mm_mxfp4(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run serialized MXFP4 operands through ``blockscaled.mm``."""
    return mm(
        (lhs_values, lhs_scale_storage),
        (rhs_values, rhs_scale_storage),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=32,
        expected_m=expected_m,
        stream=stream,
    )


def mm_nvfp4(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run serialized NVFP4 operands through ``blockscaled.mm``."""
    return mm(
        (lhs_values, lhs_scale_storage),
        (rhs_values, rhs_scale_storage),
        alpha=alpha.reshape(1),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=16,
        expected_m=expected_m,
        stream=stream,
    )


def mm_block_fp8(
    lhs_values: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run compact 128x128 block-FP8 operands."""
    return mm(
        (lhs_values, lhs_scale),
        (rhs_values, rhs_scale),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=128,
        block_fp8=True,
        expected_m=expected_m,
        stream=stream,
    )


__all__ = [
    "Weight",
    "NVFP4LinearWeight",
    "is_supported",
    "mm",
    "mm_block_fp8",
    "mm_pair",
    "mm_mxfp4",
    "mm_nvfp4",
    "pack_weight",
    "prewarm",
    "w4a16",
    "w8a16",
    "workspace_size",
]
