"""Public API for :mod:`b12x.gemm.trellis_linear`."""

from __future__ import annotations

from typing import Optional

import torch

from ..._lib.gating import default_is_supported
from ...moe._shared.kernels.w4a16.kernel import (
    clear_w4a16_kernel_cache,
    run_trellis256_dense,
)
from ...moe._shared.kernels.w4a16.prepare import (
    PreparedTrellis256DenseWeight,
    prepare_trellis256_dense_weight,
    prepare_trellis256_pair_dense_weight,
)
from . import META

PreparedWeight = PreparedTrellis256DenseWeight


def prepare_weight(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    mcg: Optional[torch.Tensor] = None,
    mul1_e4m3: Optional[torch.Tensor] = None,
    codebook: Optional[str | int] = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: Optional[torch.Tensor] = None,
) -> PreparedWeight:
    """Validate one native EXL3 dense weight and retain zero-copy views."""
    return prepare_trellis256_dense_weight(
        trellis,
        suh,
        svh,
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
        codebook=codebook,
        params_dtype=params_dtype,
        dummy_scale=dummy_scale,
    )


def prepare_pair_weight(
    payload: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    pair_kind: str,
    rate_axis: str,
    mcg: Optional[torch.Tensor] = None,
    mul1_e4m3: Optional[torch.Tensor] = None,
    codebook: Optional[str | int] = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: Optional[torch.Tensor] = None,
) -> PreparedWeight:
    """Prepare one compact TP12 P24/P33 pair for the SM12x decoder."""
    return prepare_trellis256_pair_dense_weight(
        payload,
        suh,
        svh,
        pair_kind=pair_kind,
        rate_axis=rate_axis,
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
        codebook=codebook,
        params_dtype=params_dtype,
        dummy_scale=dummy_scale,
    )


def run(
    x: torch.Tensor,
    weight: PreparedWeight,
    *,
    output: Optional[torch.Tensor] = None,
    gemm_output: Optional[torch.Tensor] = None,
    c_tmp: Optional[torch.Tensor] = None,
    input_f16: Optional[torch.Tensor] = None,
    rotated_f16: Optional[torch.Tensor] = None,
    rotated_compute: Optional[torch.Tensor] = None,
    gemm_output_f16: Optional[torch.Tensor] = None,
    output_f16: Optional[torch.Tensor] = None,
    hadamard_128=None,
    _moe_block_size: int = 64,
    _force_tile_config: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Execute Trellis GEMM with CuTe rotations and optional caller-owned storage.

    The default tile depends only on weight geometry. Explicit tile choices
    must remain fixed while changing live rows under frozen kernel resolution.
    An explicit ``hadamard_128`` callback overrides the built-in rotation.
    """
    return run_trellis256_dense(
        x,
        weight,
        output=output,
        gemm_output=gemm_output,
        c_tmp=c_tmp,
        input_f16=input_f16,
        rotated_f16=rotated_f16,
        rotated_compute=rotated_compute,
        gemm_output_f16=gemm_output_f16,
        output_f16=output_f16,
        hadamard_128=hadamard_128,
        _moe_block_size=_moe_block_size,
        _force_tile_config=_force_tile_config,
    )


def is_supported(device=None) -> bool:
    """True when the SM120/SM121 Trellis kernel stack is available."""
    return default_is_supported(device, requires=META.requires)


def clear_caches() -> None:
    """Clear compiled GEMM and dense rotation specializations."""
    from ._rotation import _KERNEL_CACHE

    _KERNEL_CACHE.clear()
    clear_w4a16_kernel_cache()


__all__ = list(META.entry_points)
