"""Public API for :mod:`b12x.gemm.trellis_linear`."""

from __future__ import annotations

from dataclasses import replace
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
from ._k6_mcg_cute import (
    k6_mcg_small_m_scratch_elements as k6_mcg_small_m_scratch_elements,
    plan_k6_mcg_small_m,
)

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
    """Validate a native EXL3 weight and bind eligible serving launches."""
    weight = prepare_trellis256_dense_weight(
        trellis,
        suh,
        svh,
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
        codebook=codebook,
        params_dtype=params_dtype,
        dummy_scale=dummy_scale,
    )
    launch = plan_k6_mcg_small_m(weight)
    if launch is None:
        return weight
    return replace(weight, k6_mcg_small_m_launch=launch)


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
    """Execute Trellis GEMM, optionally reusing all capture-time storage."""
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
    """Clear compiled dense Trellis specializations."""
    from ._k6_mcg_cute import clear_k6_mcg_small_m_cache

    clear_k6_mcg_small_m_cache()
    clear_w4a16_kernel_cache()


__all__ = list(META.entry_points)
