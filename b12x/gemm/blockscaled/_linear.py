"""Packed-weight adapters for the public :mod:`b12x.gemm.blockscaled` API."""

from __future__ import annotations

import functools
import os
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeAlias

import cutlass.cute as cute
import torch

from b12x._lib.dense_gemm import (
    _dense_spark_policy_for_sm_count,
    dense_gemm,
)
from b12x._lib.intrinsics import as_grouped_scale_view, as_grouped_scale_view_mx
from b12x._lib.utils import cuda_stream_to_int, get_num_sm
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8Rows,
    MXFP8_SCALE_VEC_SIZE,
    _check_gpu_tensor,
    pack_mxfp8_scales_for_dense_gemm,
)
from ._a16 import NVFP4LinearWeight, pack_nvfp4_weight, workspace_size


@dataclass(frozen=True)
class MXFP8LinearWeight:
    """ModelOpt-style MXFP8 weight packed for ``blockscaled.mm``."""

    weight: MXFP8Rows
    in_features: int
    padded_in_features: int
    out_features: int


@dataclass(frozen=True)
class TensorFP8LinearWeight:
    """Tensor-scaled E4M3 weight packed for ``blockscaled.mm``."""

    values: torch.Tensor
    scale_mma: torch.Tensor
    block_scale: torch.Tensor
    output_scale: torch.Tensor
    in_features: int
    padded_in_features: int
    out_features: int


Weight: TypeAlias = MXFP8LinearWeight | TensorFP8LinearWeight | NVFP4LinearWeight


_PAIR_STREAM_LOCK = Lock()
_PAIR_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


def _paired_projection_stream(device: torch.device) -> torch.cuda.Stream:
    """Return the B12X-owned projection stream for this process and device.

    CUDA stream handles are process-local resources.  Keeping the stream
    behind the custom-op boundary prevents Torch AOT artifacts from embedding
    a handle created by the process that populated the compile cache.
    """
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (os.getpid(), int(device_index))
    with _PAIR_STREAM_LOCK:
        stream = _PAIR_STREAMS.get(key)
        if stream is not None:
            return stream
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the B12X paired-projection stream must be initialized before "
                "CUDA graph capture"
            )
        stream = torch.cuda.Stream(device=device)
        _PAIR_STREAMS[key] = stream
        return stream


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _output_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"blockscaled linear output must be bf16/fp16, got {dtype}")


def _output_dtype(c_dtype: str) -> torch.dtype:
    if c_dtype == "bfloat16":
        return torch.bfloat16
    if c_dtype == "float16":
        return torch.float16
    raise ValueError(f"blockscaled output must be bfloat16/float16, got {c_dtype!r}")


def _source_2d(source: torch.Tensor) -> torch.Tensor:
    if source.ndim < 2:
        raise ValueError(f"source must have at least 2 dims, got {tuple(source.shape)}")
    return source.reshape(-1, source.shape[-1]).contiguous()


def _pad_k(tensor: torch.Tensor, padded_k: int) -> torch.Tensor:
    rows, width = map(int, tensor.shape)
    if width == padded_k:
        return tensor.contiguous()
    padded = tensor.new_zeros((rows, padded_k))
    padded[:, :width] = tensor
    return padded.contiguous()


def _scale_rows_to_u8(scale_rows: torch.Tensor) -> torch.Tensor:
    if scale_rows.dtype == torch.uint8:
        return scale_rows.contiguous()
    if scale_rows.dtype == torch.float8_e8m0fnu:
        return scale_rows.view(torch.uint8).contiguous()
    raise ValueError(f"weight_scale must be uint8/e8m0, got {scale_rows.dtype}")


def _pad_scale_rows_k(
    scale_rows_u8: torch.Tensor,
    padded_sf_k: int,
) -> torch.Tensor:
    rows, sf_k = map(int, scale_rows_u8.shape)
    if sf_k == padded_sf_k:
        return scale_rows_u8.contiguous().view(torch.float8_e8m0fnu)
    padded = torch.full(
        (rows, padded_sf_k),
        127,
        dtype=torch.uint8,
        device=scale_rows_u8.device,
    )
    padded[:, :sf_k] = scale_rows_u8
    return padded.contiguous().view(torch.float8_e8m0fnu)


def _mxfp8_scale_mma_from_input(
    scale: torch.Tensor,
    *,
    rows: int,
    width: int,
    logical_width: int,
) -> torch.Tensor:
    """Normalize compact or F8_128x4 scales to the dense-GEMM MMA view."""

    if scale.dtype == torch.uint8:
        scale_u8 = scale
    elif scale.dtype != torch.float8_e8m0fnu:
        raise ValueError(
            f"MXFP8 activation scale must be uint8/e8m0, got {scale.dtype}"
        )
    else:
        scale_u8 = scale.view(torch.uint8)

    compact_shape = (rows, logical_width // MXFP8_SCALE_VEC_SIZE)
    if scale.ndim == 2 and tuple(scale.shape) == compact_shape:
        padded_scale = _pad_scale_rows_k(
            scale_u8,
            width // MXFP8_SCALE_VEC_SIZE,
        )
        return pack_mxfp8_scales_for_dense_gemm(
            padded_scale,
            m=rows,
            k=width,
            num_groups=1,
        )

    m_tiles = _align_up(rows, 128) // 128
    k_tiles = _align_up(width, 128) // 128
    expected_shape = (32, 4, m_tiles, 4, k_tiles, 1)
    if scale.ndim == 6:
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                "MXFP8 MMA scale has the wrong shape: expected "
                f"{expected_shape}, got {tuple(scale.shape)}"
            )
        return scale

    expected_numel = m_tiles * k_tiles * 32 * 4 * 4
    if not scale.is_contiguous() or scale.numel() != expected_numel:
        raise ValueError(
            "MXFP8 activation scale must use contiguous F8_128x4 swizzled "
            f"storage with {expected_numel} elements for M={rows}, K={width}; "
            f"got shape={tuple(scale.shape)}, contiguous={scale.is_contiguous()}"
        )
    physical = scale.view(m_tiles, k_tiles, 32, 4, 4)
    return physical.permute(2, 3, 0, 4, 1).unsqueeze(-1)


def _unit_scale_mma(rows: int, width: int, device: torch.device) -> torch.Tensor:
    scale_rows = torch.full(
        (rows, width // MXFP8_SCALE_VEC_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    return pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=rows,
        k=width,
        num_groups=1,
    )


def _unit_block_scale(rows: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.ones(
        (rows // 128, width // 128),
        dtype=torch.float32,
        device=device,
    )


@functools.cache
def _cached_unit_scale_mma(
    device_type: str,
    device_index: int | None,
    rows: int,
    width: int,
) -> torch.Tensor:
    return _unit_scale_mma(rows, width, torch.device(device_type, device_index))


def _activation_scale_mma(
    source: torch.Tensor,
    rows: int,
    width: int,
) -> torch.Tensor:
    device_index = source.device.index
    if source.device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return _cached_unit_scale_mma(
        source.device.type,
        device_index,
        int(rows),
        int(width),
    )


@functools.cache
def _cached_unit_activation_block_scale(
    device_type: str,
    device_index: int | None,
    rows: int,
    width: int,
) -> torch.Tensor:
    return torch.ones(
        (rows, width // 128),
        dtype=torch.float32,
        device=torch.device(device_type, device_index),
    )


def _activation_block_scale(
    source: torch.Tensor,
    rows: int,
    width: int,
) -> torch.Tensor:
    device_index = source.device.index
    if source.device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return _cached_unit_activation_block_scale(
        source.device.type,
        device_index,
        int(rows),
        int(width),
    )


def _use_block_fp8_recipe(
    *,
    live_m: int,
    expected_m: int,
    out_features: int,
    padded_in_features: int,
    sm_count: int,
) -> bool:
    """Select the measured degenerate K128 recipe for aligned decode GEMMs."""

    return (
        live_m <= 8
        and expected_m <= 8
        and out_features % 128 == 0
        and padded_in_features % 128 == 0
        and _dense_spark_policy_for_sm_count(sm_count)
    )


def is_mxfp8_linear_supported() -> tuple[bool, str | None]:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        return False, "CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op"
    return True, None


def pack_mxfp8_linear_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> MXFP8LinearWeight:
    """Pack ModelOpt ``[N,K]`` MXFP8 values and compact ``[N,K/32]`` scales."""

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("weight_scale", weight_scale)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if weight_scale.ndim != 2:
        raise ValueError(
            f"weight_scale must have shape [N,K/32], got {tuple(weight_scale.shape)}"
        )

    out_features, in_features = map(int, weight.shape)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features <= 0 or in_features % MXFP8_SCALE_VEC_SIZE != 0:
        raise ValueError(
            "ModelOpt MXFP8 weight K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {in_features}"
        )

    scale_k = in_features // MXFP8_SCALE_VEC_SIZE
    if (
        int(weight_scale.shape[0]) < out_features
        or int(weight_scale.shape[1]) < scale_k
    ):
        raise ValueError(
            "weight_scale must have at least shape "
            f"{(out_features, scale_k)}, got {tuple(weight_scale.shape)}"
        )

    padded_in_features = _align_up(in_features, 128)
    padded_scale_k = padded_in_features // MXFP8_SCALE_VEC_SIZE
    weight_values = _pad_k(
        weight[:out_features, :in_features],
        padded_in_features,
    )
    scale_rows_u8 = _scale_rows_to_u8(weight_scale[:out_features, :scale_k])
    scale_rows = _pad_scale_rows_k(scale_rows_u8, padded_scale_k)
    scale_mma = pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=out_features,
        k=padded_in_features,
        num_groups=1,
    )
    return MXFP8LinearWeight(
        weight=MXFP8Rows(
            values=weight_values,
            scale_rows=scale_rows.reshape(1, out_features, padded_scale_k),
            scale_mma=scale_mma,
        ),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


def is_tensor_fp8_linear_supported() -> tuple[bool, str | None]:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        return False, "CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op"
    return True, None


def pack_tensor_fp8_linear_weight(
    weight: torch.Tensor,
    output_scale: torch.Tensor,
) -> TensorFP8LinearWeight:
    """Pack an E4M3 ``[N,K]`` weight with one combined dequantization scale."""

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("output_scale", output_scale)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if output_scale.dtype != torch.float32 or output_scale.numel() != 1:
        raise ValueError(
            "output_scale must be one float32 value, got "
            f"dtype={output_scale.dtype}, shape={tuple(output_scale.shape)}"
        )
    if output_scale.device != weight.device:
        raise ValueError("weight and output_scale must be on the same device")
    if not bool(torch.isfinite(output_scale).all()) or bool((output_scale < 0).any()):
        raise ValueError("output_scale must be finite and non-negative")

    out_features, in_features = map(int, weight.shape)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features <= 0 or in_features % MXFP8_SCALE_VEC_SIZE != 0:
        raise ValueError(
            "tensor FP8 weight K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {in_features}"
        )

    padded_in_features = _align_up(in_features, 128)
    values = _pad_k(weight, padded_in_features)
    scale_mma = _unit_scale_mma(
        out_features,
        padded_in_features,
        weight.device,
    )
    return TensorFP8LinearWeight(
        values=values,
        scale_mma=scale_mma,
        block_scale=_unit_block_scale(
            out_features,
            padded_in_features,
            weight.device,
        ),
        output_scale=output_scale.reshape(1).contiguous(),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


def pack_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    recipe: str | None = None,
    global_scale: torch.Tensor | None = None,
    global_scale_kind: str = "multiplier",
) -> Weight:
    """Pack a serialized dense weight for ``blockscaled.mm``.

    A scalar FP32 ``scale`` selects tensor-scaled FP8 and represents the
    combined activation/weight dequantization scale. A 2D uint8/UE8M0 scale
    selects ModelOpt MXFP8.

    ``recipe='nvfp4'`` borrows packed uint8 values and swizzled E4M3 scales.
    Pass the weight-only ``global_scale`` and its ``global_scale_kind``
    (``'multiplier'`` or ``'reciprocal'``); no scale transformation occurs.
    """

    if recipe == "nvfp4":
        return pack_nvfp4_weight(weight, scale, global_scale,
                                 global_scale_kind=global_scale_kind)
    if recipe not in (None, "mxfp8", "tensor_fp8") or global_scale is not None or global_scale_kind != "multiplier":
        raise ValueError("global_scale/kind are specific to recipe='nvfp4'")
    if scale.dtype == torch.float32 and scale.numel() == 1 and recipe != "mxfp8":
        return pack_tensor_fp8_linear_weight(weight, scale)
    if scale.dtype in (torch.uint8, torch.float8_e8m0fnu) and scale.ndim == 2 and recipe != "tensor_fp8":
        return pack_mxfp8_linear_weight(weight, scale)
    raise ValueError(
        "unsupported blockscaled weight scale: expected one FP32 combined "
        "tensor scale or a 2D uint8/UE8M0 MXFP8 scale; got "
        f"dtype={scale.dtype}, shape={tuple(scale.shape)}"
    )


@torch.library.custom_op(
    "b12x::blockscaled_serialized",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _blockscaled_serialized_op(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor | None,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    block_fp8: bool,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    """Opaque adapter from checkpoint/quantizer storage to dense-GEMM views."""

    if lhs_values.ndim != 2 or rhs_values.ndim != 2:
        raise ValueError("serialized blockscaled operands must both be 2D")
    m, lhs_storage_k = map(int, lhs_values.shape)
    n, rhs_storage_k = map(int, rhs_values.shape)
    if rhs_storage_k != lhs_storage_k:
        raise ValueError(
            "blockscaled operands must have the same storage K extent, got "
            f"{lhs_storage_k} and {rhs_storage_k}"
        )

    if ab_dtype == "float4_e2m1fn":
        k = lhs_storage_k * 2
        if block_fp8:
            raise ValueError("block_fp8 is valid only for E4M3 operands")
        if sf_dtype == "float8_e4m3fn" and sf_vec_size == 16:
            lhs_scale = as_grouped_scale_view(
                lhs_scale_storage.view(torch.uint8).unsqueeze(0), m, k
            )
            rhs_scale = as_grouped_scale_view(
                rhs_scale_storage.view(torch.uint8).unsqueeze(0), n, k
            )
        elif sf_dtype == "float8_e8m0fnu" and sf_vec_size == 32:
            lhs_scale = as_grouped_scale_view_mx(
                lhs_scale_storage.view(torch.uint8).unsqueeze(0), m, k
            )
            rhs_scale = as_grouped_scale_view_mx(
                rhs_scale_storage.view(torch.uint8).unsqueeze(0), n, k
            )
        else:
            raise ValueError(
                "serialized FP4 requires NVFP4 E4M3/vec16 or MXFP4 "
                f"E8M0/vec32 scales, got {sf_dtype}/vec{sf_vec_size}"
            )
    elif (
        ab_dtype == "float8_e4m3fn"
        and sf_dtype == "float32"
        and sf_vec_size == 128
        and block_fp8
    ):
        k = lhs_storage_k
        lhs_scale = lhs_scale_storage
        rhs_scale = rhs_scale_storage
    else:
        raise ValueError(
            "unsupported serialized blockscaled recipe: "
            f"ab_dtype={ab_dtype}, sf_dtype={sf_dtype}, "
            f"sf_vec_size={sf_vec_size}, block_fp8={block_fp8}"
        )

    return dense_gemm(
        (lhs_values.reshape(m, lhs_storage_k, 1), lhs_scale),
        (rhs_values.reshape(n, rhs_storage_k, 1), rhs_scale),
        alpha=alpha,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        block_fp8=block_fp8,
        expected_m=expected_m,
        stream=stream_int,
    )[:, :, 0]


@_blockscaled_serialized_op.register_fake
def _blockscaled_serialized_fake(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor | None,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    block_fp8: bool,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del lhs_scale_storage, rhs_scale_storage, alpha
    del ab_dtype, sf_dtype, sf_vec_size, block_fp8, expected_m, stream_int
    return torch.empty(
        (lhs_values.shape[0], rhs_values.shape[0]),
        dtype=_output_dtype(c_dtype),
        device=lhs_values.device,
    )


@torch.library.custom_op(
    "b12x::blockscaled_packed_mxfp8",
    mutates_args=(),
)
def _packed_mxfp8_op(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_rows: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    in_features: int,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del weight_scale_rows, in_features
    tokens = int(source_2d.shape[0])
    source_for_quant = _pad_k(source_2d, int(padded_in_features))
    from b12x.gemm._shared.block_fp8 import (
        quantize_block_fp8_linear_input_mxfp8,
    )

    x_q = quantize_block_fp8_linear_input_mxfp8(source_for_quant)
    return dense_gemm(
        (x_q.values.reshape(tokens, padded_in_features, 1), x_q.scale_mma),
        (
            weight_values.reshape(out_features, padded_in_features, 1),
            weight_scale_mma,
        ),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_output_dtype_name(source_2d.dtype),
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        expected_m=expected_m,
        stream=stream_int,
    )[:, :, 0]


@_packed_mxfp8_op.register_fake
def _packed_mxfp8_fake(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_rows: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    in_features: int,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del weight_values, weight_scale_rows, weight_scale_mma
    del in_features, padded_in_features, expected_m, stream_int
    return torch.empty(
        (source_2d.shape[0], out_features),
        dtype=source_2d.dtype,
        device=source_2d.device,
    )


@torch.library.custom_op(
    "b12x::blockscaled_packed_mxfp8_pair",
    mutates_args=(),
)
def _packed_mxfp8_pair_op(
    source_2d: torch.Tensor,
    primary_weight_values: torch.Tensor,
    primary_weight_scale_rows: torch.Tensor,
    primary_weight_scale_mma: torch.Tensor,
    primary_padded_in_features: int,
    primary_out_features: int,
    secondary_weight_values: torch.Tensor,
    secondary_weight_scale_rows: torch.Tensor,
    secondary_weight_scale_mma: torch.Tensor,
    secondary_padded_in_features: int,
    secondary_out_features: int,
    expected_m: int,
    parallel_max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute independent packed MXFP8 projections with an optional overlap."""

    def project(
        weight_values: torch.Tensor,
        weight_scale_rows: torch.Tensor,
        weight_scale_mma: torch.Tensor,
        padded_in_features: int,
        out_features: int,
    ) -> torch.Tensor:
        return torch.ops.b12x.blockscaled_packed_mxfp8(
            source_2d,
            weight_values,
            weight_scale_rows,
            weight_scale_mma,
            int(source_2d.shape[1]),
            padded_in_features,
            out_features,
            expected_m,
            None,
        )

    tokens = int(source_2d.shape[0])
    if tokens > parallel_max_tokens:
        return (
            project(
                primary_weight_values,
                primary_weight_scale_rows,
                primary_weight_scale_mma,
                primary_padded_in_features,
                primary_out_features,
            ),
            project(
                secondary_weight_values,
                secondary_weight_scale_rows,
                secondary_weight_scale_mma,
                secondary_padded_in_features,
                secondary_out_features,
            ),
        )

    current_stream = torch.cuda.current_stream(source_2d.device)
    secondary_stream = _paired_projection_stream(source_2d.device)
    secondary_stream.wait_stream(current_stream)
    primary = project(
        primary_weight_values,
        primary_weight_scale_rows,
        primary_weight_scale_mma,
        primary_padded_in_features,
        primary_out_features,
    )
    with torch.cuda.stream(secondary_stream):
        secondary = project(
            secondary_weight_values,
            secondary_weight_scale_rows,
            secondary_weight_scale_mma,
            secondary_padded_in_features,
            secondary_out_features,
        )
    current_stream.wait_stream(secondary_stream)
    return primary, secondary


@_packed_mxfp8_pair_op.register_fake
def _packed_mxfp8_pair_fake(
    source_2d: torch.Tensor,
    primary_weight_values: torch.Tensor,
    primary_weight_scale_rows: torch.Tensor,
    primary_weight_scale_mma: torch.Tensor,
    primary_padded_in_features: int,
    primary_out_features: int,
    secondary_weight_values: torch.Tensor,
    secondary_weight_scale_rows: torch.Tensor,
    secondary_weight_scale_mma: torch.Tensor,
    secondary_padded_in_features: int,
    secondary_out_features: int,
    expected_m: int,
    parallel_max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe paired projection outputs without allocating CUDA storage."""
    del primary_weight_values, primary_weight_scale_rows, primary_weight_scale_mma
    del secondary_weight_values, secondary_weight_scale_rows
    del secondary_weight_scale_mma, primary_padded_in_features
    del secondary_padded_in_features, expected_m, parallel_max_tokens
    return (
        source_2d.new_empty((source_2d.shape[0], primary_out_features)),
        source_2d.new_empty((source_2d.shape[0], secondary_out_features)),
    )


@torch.library.custom_op(
    "b12x::blockscaled_packed_mxfp8_prequantized",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _packed_mxfp8_prequantized_op(
    source_values: torch.Tensor,
    source_scale_storage: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    padded_in_features: int,
    out_features: int,
    out_dtype: torch.dtype,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    tokens = int(source_values.shape[0])
    source_padded = _pad_k(source_values, int(padded_in_features))
    source_scale_mma = _mxfp8_scale_mma_from_input(
        source_scale_storage,
        rows=tokens,
        width=int(padded_in_features),
        logical_width=int(source_values.shape[1]),
    )
    return dense_gemm(
        (
            source_padded.reshape(tokens, padded_in_features, 1),
            source_scale_mma,
        ),
        (
            weight_values.reshape(out_features, padded_in_features, 1),
            weight_scale_mma,
        ),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        expected_m=expected_m,
        stream=stream_int,
    )[:, :, 0]


@_packed_mxfp8_prequantized_op.register_fake
def _packed_mxfp8_prequantized_fake(
    source_values: torch.Tensor,
    source_scale_storage: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    padded_in_features: int,
    out_features: int,
    out_dtype: torch.dtype,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del source_scale_storage, weight_values, weight_scale_mma
    del padded_in_features, expected_m, stream_int
    return torch.empty(
        (source_values.shape[0], out_features),
        dtype=out_dtype,
        device=source_values.device,
    )


def _validate_bias(
    bias: torch.Tensor | None,
    *,
    out_features: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> None:
    if bias is None:
        return
    _check_gpu_tensor("bias", bias)
    if bias.device != device:
        raise ValueError("bias must be on the same device as source")
    if bias.dtype != out_dtype or bias.shape != (out_features,):
        raise ValueError(
            f"bias must have shape {(out_features,)} and dtype {out_dtype}, "
            f"got shape={tuple(bias.shape)}, dtype={bias.dtype}"
        )


def mxfp8_linear(
    source: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    packed_weight: MXFP8LinearWeight,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run plain or prequantized activations through MXFP8 ``blockscaled.mm``."""

    if not isinstance(packed_weight, MXFP8LinearWeight):
        raise TypeError("packed_weight must be an MXFP8LinearWeight")

    prequantized = isinstance(source, tuple)
    if prequantized:
        if len(source) != 2:
            raise ValueError("prequantized source must be a (values, scale) pair")
        source_values, source_scale = source
    else:
        source_values = source
        source_scale = None
    _check_gpu_tensor("source", source_values)
    source_2d = _source_2d(source_values)
    tokens, in_features = map(int, source_2d.shape)
    if in_features != int(packed_weight.in_features):
        raise ValueError(
            f"input K={in_features} does not match packed weight K="
            f"{packed_weight.in_features}"
        )
    if packed_weight.weight.values.device != source_2d.device:
        raise ValueError("source and packed weight must be on the same device")
    if expected_m is not None and int(expected_m) <= 0:
        raise ValueError("expected_m must be positive when provided")

    if prequantized:
        if source_2d.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"prequantized MXFP8 source must be float8_e4m3fn, got {source_2d.dtype}"
            )
        assert source_scale is not None
        _check_gpu_tensor("source_scale", source_scale)
        if source_scale.device != source_2d.device:
            raise ValueError("source and source_scale must be on the same device")
        resolved_out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    else:
        if source_2d.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"source dtype must be bf16/fp16, got {source_2d.dtype}")
        resolved_out_dtype = source_2d.dtype if out_dtype is None else out_dtype
        if resolved_out_dtype != source_2d.dtype:
            raise ValueError(
                "plain MXFP8 output dtype must match the BF16/FP16 source dtype"
            )
    _output_dtype_name(resolved_out_dtype)

    out_features = int(packed_weight.out_features)
    _validate_bias(
        bias,
        out_features=out_features,
        out_dtype=resolved_out_dtype,
        device=source_2d.device,
    )
    if tokens == 0:
        output = torch.empty(
            (0, out_features),
            dtype=resolved_out_dtype,
            device=source_2d.device,
        )
    elif prequantized:
        assert source_scale is not None
        padded_k = int(packed_weight.padded_in_features)
        output = torch.ops.b12x.blockscaled_packed_mxfp8_prequantized(
            source_2d,
            source_scale,
            packed_weight.weight.values,
            packed_weight.weight.scale_mma,
            padded_k,
            out_features,
            resolved_out_dtype,
            int(expected_m) if expected_m is not None else tokens,
            cuda_stream_to_int(stream),
        )
    else:
        output = torch.ops.b12x.blockscaled_packed_mxfp8(
            source_2d,
            packed_weight.weight.values,
            packed_weight.weight.scale_rows if packed_weight.weight.scale_rows is not None else packed_weight.weight.scale_mma,
            packed_weight.weight.scale_mma,
            packed_weight.in_features,
            packed_weight.padded_in_features,
            packed_weight.out_features,
            int(expected_m) if expected_m is not None else tokens,
            cuda_stream_to_int(stream),
        )
    if bias is not None:
        output = output + bias
    return output.view(*source_values.shape[:-1], out_features)


def mxfp8_linear_pair(
    source: torch.Tensor,
    primary_weight: MXFP8LinearWeight,
    secondary_weight: MXFP8LinearWeight,
    *,
    expected_m: int | None = None,
    parallel_max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one activation tensor through two independent MXFP8 weights.

    The primary projection runs on the caller's current CUDA stream. When the
    row count does not exceed ``parallel_max_tokens``, the secondary projection
    runs on a B12X-owned process-local stream and the current stream waits for
    it before the function returns. Larger inputs execute serially on the
    current stream so bandwidth-bound prefill projections do not contend with
    each other.
    """
    if not isinstance(primary_weight, MXFP8LinearWeight):
        raise TypeError("primary_weight must be an MXFP8LinearWeight")
    if not isinstance(secondary_weight, MXFP8LinearWeight):
        raise TypeError("secondary_weight must be an MXFP8LinearWeight")
    _check_gpu_tensor("source", source)
    source_2d = _source_2d(source)
    tokens, in_features = map(int, source_2d.shape)
    if source_2d.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"source dtype must be bf16/fp16, got {source_2d.dtype}")
    if parallel_max_tokens <= 0:
        raise ValueError("parallel_max_tokens must be positive")
    if expected_m is not None and int(expected_m) <= 0:
        raise ValueError("expected_m must be positive when provided")
    for name, weight in (
        ("primary_weight", primary_weight),
        ("secondary_weight", secondary_weight),
    ):
        if in_features != int(weight.in_features):
            raise ValueError(
                f"source K={in_features} does not match {name} K={weight.in_features}"
            )
        if weight.weight.values.device != source_2d.device:
            raise ValueError(f"source and {name} must be on the same device")

    leading_shape = source.shape[:-1]
    if tokens == 0:
        return (
            source.new_empty((*leading_shape, primary_weight.out_features)),
            source.new_empty((*leading_shape, secondary_weight.out_features)),
        )

    primary, secondary = torch.ops.b12x.blockscaled_packed_mxfp8_pair(
        source_2d,
        primary_weight.weight.values,
        primary_weight.weight.scale_rows,
        primary_weight.weight.scale_mma,
        primary_weight.padded_in_features,
        primary_weight.out_features,
        secondary_weight.weight.values,
        secondary_weight.weight.scale_rows,
        secondary_weight.weight.scale_mma,
        secondary_weight.padded_in_features,
        secondary_weight.out_features,
        int(expected_m) if expected_m is not None else tokens,
        int(parallel_max_tokens),
    )
    return (
        primary.view(*leading_shape, primary_weight.out_features),
        secondary.view(*leading_shape, secondary_weight.out_features),
    )


@torch.library.custom_op(
    "b12x::blockscaled_packed_tensor_fp8",
    mutates_args=(),
)
def _packed_tensor_fp8_op(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    weight_block_scale: torch.Tensor,
    output_scale: torch.Tensor,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
) -> torch.Tensor:
    tokens = int(source_2d.shape[0])
    source_padded = _pad_k(source_2d, int(padded_in_features))
    if _use_block_fp8_recipe(
        live_m=tokens,
        expected_m=int(expected_m),
        out_features=int(out_features),
        padded_in_features=int(padded_in_features),
        sm_count=get_num_sm(source_2d.device),
    ):
        source_block_scale = _activation_block_scale(
            source_padded,
            tokens,
            int(padded_in_features),
        )
        return dense_gemm(
            (
                source_padded.reshape(tokens, padded_in_features, 1),
                source_block_scale,
            ),
            (
                weight_values.reshape(out_features, padded_in_features, 1),
                weight_block_scale,
            ),
            alpha=output_scale,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float32",
            c_dtype=_output_dtype_name(out_dtype),
            sf_vec_size=128,
            expected_m=expected_m,
            stream=stream_int,
            block_fp8=True,
        )[:, :, 0]

    source_scale_mma = _activation_scale_mma(
        source_padded,
        tokens,
        int(padded_in_features),
    )
    return dense_gemm(
        (
            source_padded.reshape(tokens, padded_in_features, 1),
            source_scale_mma,
        ),
        (
            weight_values.reshape(out_features, padded_in_features, 1),
            weight_scale_mma,
        ),
        alpha=output_scale,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        expected_m=expected_m,
        stream=stream_int,
        plain_fp8=True,
    )[:, :, 0]


@_packed_tensor_fp8_op.register_fake
def _packed_tensor_fp8_fake(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    weight_block_scale: torch.Tensor,
    output_scale: torch.Tensor,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
) -> torch.Tensor:
    del weight_values, weight_scale_mma, weight_block_scale, output_scale
    del padded_in_features, expected_m, stream_int
    return torch.empty(
        (source_2d.shape[0], out_features),
        dtype=out_dtype,
        device=source_2d.device,
    )


def tensor_fp8_linear(
    source: torch.Tensor,
    packed_weight: TensorFP8LinearWeight,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run static per-tensor E4M3 operands through the SM12x dense GEMM."""

    _check_gpu_tensor("source", source)
    if not isinstance(packed_weight, TensorFP8LinearWeight):
        raise TypeError("packed_weight must be a TensorFP8LinearWeight")
    source_2d = _source_2d(source)
    tokens, in_features = map(int, source_2d.shape)
    if source_2d.dtype != torch.float8_e4m3fn:
        raise ValueError(f"source must be float8_e4m3fn, got {source_2d.dtype}")
    if in_features != int(packed_weight.in_features):
        raise ValueError(
            f"input K={in_features} does not match packed weight K="
            f"{packed_weight.in_features}"
        )
    if packed_weight.values.device != source_2d.device:
        raise ValueError("source and packed weight must be on the same device")
    if expected_m is not None and int(expected_m) <= 0:
        raise ValueError("expected_m must be positive when provided")
    _output_dtype_name(out_dtype)

    out_features = int(packed_weight.out_features)
    _validate_bias(
        bias,
        out_features=out_features,
        out_dtype=out_dtype,
        device=source_2d.device,
    )
    if tokens == 0:
        output = torch.empty(
            (0, out_features),
            dtype=out_dtype,
            device=source_2d.device,
        )
    else:
        output = torch.ops.b12x.blockscaled_packed_tensor_fp8(
            source_2d,
            packed_weight.values,
            packed_weight.scale_mma,
            packed_weight.block_scale,
            packed_weight.output_scale,
            packed_weight.padded_in_features,
            packed_weight.out_features,
            int(expected_m) if expected_m is not None else tokens,
            out_dtype,
            cuda_stream_to_int(stream),
        )
    if bias is not None:
        output = output + bias
    return output.view(*source.shape[:-1], out_features)


def blockscaled_mm(
    lhs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    rhs: Weight | tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch packed linear weights or preserve the raw ``dense_gemm`` API."""

    if isinstance(rhs, NVFP4LinearWeight) or (
        isinstance(rhs, MXFP8LinearWeight) and isinstance(lhs, torch.Tensor)
        and lhs.dtype == torch.bfloat16
    ):
        if not isinstance(lhs, torch.Tensor):
            raise TypeError("packed NVFP4 linear requires a BF16 source tensor")
        options = dict(kwargs)
        bias = options.pop("bias", None)
        out_dtype = options.pop("out_dtype", None)
        if out_dtype not in (None, torch.bfloat16):
            raise ValueError("BF16 blockscaled linear output must be BF16")
        _validate_bias(bias, out_features=rhs.out_features, out_dtype=torch.bfloat16,
                       device=lhs.device)
        if bias is not None and out is not None:
            from ._a16 import _overlap
            if _overlap(out, bias):
                raise ValueError("out must not overlap bias")
        from ._ops import linear
        fp4 = isinstance(rhs, NVFP4LinearWeight)
        result = linear(lhs, rhs.values if fp4 else rhs.weight.values,
                        rhs.scale_mma if fp4 else rhs.weight.scale_mma,
                        rhs.global_scale if fp4 else None, fp4=fp4,
                        reciprocal=fp4 and rhs.global_scale_kind == "reciprocal",
                        input_k=rhs.in_features, out=out, **options)
        if bias is not None:
            from ._a16 import _stream_context
            with torch.cuda.device(lhs.device), _stream_context(options.get("stream"), lhs.device):
                result.add_(bias)
        return result
    if isinstance(rhs, MXFP8LinearWeight):
        if out is not None:
            raise ValueError("packed MXFP8 blockscaled.mm does not accept out")
        return mxfp8_linear(lhs, rhs, **kwargs)
    if isinstance(rhs, TensorFP8LinearWeight):
        if out is not None:
            raise ValueError("packed tensor-FP8 blockscaled.mm does not accept out")
        if isinstance(lhs, tuple):
            raise TypeError(
                "tensor-scaled FP8 blockscaled.mm accepts its prequantized "
                "values tensor directly; its static scale is already folded "
                "into the packed weight"
            )
        return tensor_fp8_linear(lhs, rhs, **kwargs)
    if not isinstance(lhs, tuple) or not isinstance(rhs, tuple):
        raise TypeError(
            "raw blockscaled.mm operands must be (values, scale) pairs, or rhs "
            "must be a weight returned by blockscaled.pack_weight"
        )
    lhs_values, lhs_scale = lhs
    rhs_values, rhs_scale = rhs
    if lhs_values.ndim == 2 or rhs_values.ndim == 2:
        if lhs_values.ndim != 2 or rhs_values.ndim != 2:
            raise ValueError(
                "serialized blockscaled values must either both be 2D or both "
                "use the native 3D dense-GEMM layout"
            )
        if out is not None:
            raise ValueError("serialized blockscaled.mm does not accept out")
        recipe = dict(kwargs)
        try:
            ab_dtype = recipe.pop("ab_dtype")
            sf_dtype = recipe.pop("sf_dtype")
            c_dtype = recipe.pop("c_dtype")
            sf_vec_size = recipe.pop("sf_vec_size")
        except KeyError as exc:
            raise TypeError(
                f"serialized blockscaled.mm requires {exc.args[0]}"
            ) from None
        alpha = recipe.pop("alpha", None)
        expected_m = recipe.pop("expected_m", None)
        stream = recipe.pop("stream", None)
        block_fp8 = bool(recipe.pop("block_fp8", False))
        if recipe:
            names = ", ".join(sorted(recipe))
            raise TypeError(
                "serialized blockscaled.mm does not support these raw-engine "
                f"options: {names}"
            )
        return torch.ops.b12x.blockscaled_serialized(
            lhs_values,
            lhs_scale,
            rhs_values,
            rhs_scale,
            alpha,
            str(ab_dtype),
            str(sf_dtype),
            str(c_dtype),
            int(sf_vec_size),
            block_fp8,
            (int(expected_m) if expected_m is not None else int(lhs_values.shape[0])),
            cuda_stream_to_int(stream),
        )
    return dense_gemm(lhs, rhs, out, **kwargs)


def prewarm(
    rhs: Weight | tuple[torch.Tensor, torch.Tensor],
    token_counts: Iterable[int],
    *,
    input_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
    **mm_kwargs: Any,
) -> int:
    """Compile ``blockscaled.mm`` calls for serving token counts.

    Packed weights infer their recipe.  For a raw serialized ``(values,
    scales)`` RHS, pass the same recipe keywords used by :func:`mm`.
    ``input_dtype=torch.float8_e4m3fn`` selects the prequantized-input path
    for an MXFP8 packed weight; BF16/FP16 selects its inline-quantized path.
    """

    counts = sorted({int(value) for value in token_counts if int(value) > 0})

    if isinstance(rhs, NVFP4LinearWeight) or (
        isinstance(rhs, MXFP8LinearWeight) and rhs.out_features % 8 == 0
        and (input_dtype if input_dtype is not None else out_dtype) == torch.bfloat16
        and torch.cuda.get_device_capability(rhs.weight.values.device) in ((12, 0), (12, 1))
    ):
        if input_dtype not in (None, torch.bfloat16) or out_dtype != torch.bfloat16:
            raise ValueError("NVFP4/MXFP8 BF16 warmup requires BF16 input/output")
        if not counts:
            return 0
        options = dict(mm_kwargs)
        mode = options.pop("mode", "auto")
        if mode not in ("auto", "a16", "quantized"):
            raise ValueError("mode must be 'auto', 'a16', or 'quantized'")
        device = rhs.values.device if isinstance(rhs, NVFP4LinearWeight) else rhs.weight.values.device
        workspace = options.pop("workspace", None)
        from ._a16 import _stream_context
        with torch.inference_mode(), torch.cuda.device(device), _stream_context(stream, device):
            if workspace is None:
                workspace = torch.empty(workspace_size(rhs, max(counts), _config=options.get("_config")),
                                        dtype=torch.uint8, device=device)
            for tokens in counts:
                source = torch.zeros((tokens, rhs.in_features), dtype=torch.bfloat16, device=device)
                output = torch.empty((tokens, rhs.out_features), dtype=torch.bfloat16, device=device)
                blockscaled_mm(source, rhs, out=output, workspace=workspace,
                               mode=mode, stream=stream, **options)
                if isinstance(rhs, MXFP8LinearWeight) and mode == "auto":
                    blockscaled_mm(source, rhs, mode=mode, stream=stream, **options)
        return len(counts)

    if isinstance(rhs, tuple):
        if len(rhs) != 2:
            raise ValueError("raw blockscaled RHS must be a (values, scale) pair")
        rhs_values, _ = rhs
        _check_gpu_tensor("rhs_values", rhs_values)
        if rhs_values.ndim != 2:
            raise ValueError("raw serialized blockscaled RHS values must be 2D")
        recipe = dict(mm_kwargs)
        try:
            ab_dtype = str(recipe["ab_dtype"])
            sf_dtype = str(recipe["sf_dtype"])
            sf_vec_size = int(recipe["sf_vec_size"])
        except KeyError as exc:
            raise TypeError(f"raw blockscaled prewarm requires {exc.args[0]}") from None

        storage_k = int(rhs_values.shape[1])
        if ab_dtype == "float4_e2m1fn":
            logical_k = storage_k * 2
            values_dtype = torch.uint8
            scale_cols = _align_up(logical_k // sf_vec_size, 4)
            if sf_dtype == "float8_e4m3fn" and sf_vec_size == 16:
                scale_dtype = torch.float8_e4m3fn
                scale_fill = 1.0
            elif sf_dtype == "float8_e8m0fnu" and sf_vec_size == 32:
                scale_dtype = torch.uint8
                scale_fill = 127
            else:
                raise ValueError(
                    "raw FP4 prewarm requires E4M3/vec16 or E8M0/vec32 scales"
                )
        elif (
            ab_dtype == "float8_e4m3fn"
            and sf_dtype == "float32"
            and sf_vec_size == 128
            and bool(recipe.get("block_fp8", False))
        ):
            logical_k = storage_k
            values_dtype = torch.float8_e4m3fn
            scale_cols = logical_k // 128
            scale_dtype = torch.float32
            scale_fill = 1.0
        else:
            raise ValueError(
                "unsupported raw blockscaled prewarm recipe: "
                f"ab_dtype={ab_dtype}, sf_dtype={sf_dtype}, "
                f"sf_vec_size={sf_vec_size}"
            )
        if input_dtype is not None and input_dtype != values_dtype:
            raise ValueError(
                f"raw recipe input dtype is {values_dtype}, got {input_dtype}"
            )

        warmed = 0
        with torch.inference_mode():
            for tokens in counts:
                lhs_values = torch.zeros(
                    (tokens, storage_k),
                    dtype=values_dtype,
                    device=rhs_values.device,
                )
                scale_rows = (
                    _align_up(tokens, 128) if ab_dtype == "float4_e2m1fn" else tokens
                )
                lhs_scale = torch.full(
                    (scale_rows, scale_cols),
                    scale_fill,
                    dtype=scale_dtype,
                    device=rhs_values.device,
                )
                blockscaled_mm(
                    (lhs_values, lhs_scale),
                    rhs,
                    stream=stream,
                    **recipe,
                )
                warmed += 1
        return warmed

    if isinstance(rhs, TensorFP8LinearWeight):
        source_dtype = torch.float8_e4m3fn
        if input_dtype is not None and input_dtype != source_dtype:
            raise ValueError("tensor-FP8 warmup input_dtype must be float8_e4m3fn")
    elif isinstance(rhs, MXFP8LinearWeight):
        source_dtype = out_dtype if input_dtype is None else input_dtype
        if source_dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
        ):
            raise ValueError("MXFP8 warmup input_dtype must be bf16/fp16/fp8_e4m3")
        if source_dtype != torch.float8_e4m3fn and source_dtype != out_dtype:
            raise ValueError("MXFP8 warmup input_dtype and out_dtype must match")
    else:
        raise TypeError("rhs must be a raw pair or returned by blockscaled.pack_weight")
    if mm_kwargs:
        names = ", ".join(sorted(mm_kwargs))
        raise TypeError(
            f"packed-weight prewarm does not accept recipe options: {names}"
        )

    warmed = 0
    with torch.inference_mode():
        for tokens in counts:
            source_values = torch.zeros(
                (tokens, rhs.in_features),
                dtype=source_dtype,
                device=(
                    rhs.values.device
                    if isinstance(rhs, TensorFP8LinearWeight)
                    else rhs.weight.values.device
                ),
            )
            source: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            if (
                isinstance(rhs, MXFP8LinearWeight)
                and source_dtype == torch.float8_e4m3fn
            ):
                scale_rows = _align_up(tokens, 128)
                scale_cols = _align_up(rhs.padded_in_features // 32, 4)
                source_scale = torch.full(
                    (scale_rows * scale_cols,),
                    127,
                    dtype=torch.uint8,
                    device=source_values.device,
                )
                source = (source_values, source_scale)
            else:
                source = source_values
            blockscaled_mm(
                source,
                rhs,
                out_dtype=out_dtype,
                expected_m=tokens,
                stream=stream,
            )
            warmed += 1
    return warmed


def prewarm_tensor_fp8_linear(
    packed_weight: TensorFP8LinearWeight,
    token_counts: Iterable[int],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
) -> int:
    """Compatibility alias for :func:`prewarm`."""

    return prewarm(
        packed_weight,
        token_counts,
        out_dtype=out_dtype,
        stream=stream,
    )


__all__ = [
    "MXFP8LinearWeight",
    "TensorFP8LinearWeight",
    "Weight",
    "blockscaled_mm",
    "is_mxfp8_linear_supported",
    "is_tensor_fp8_linear_supported",
    "mxfp8_linear",
    "mxfp8_linear_pair",
    "pack_mxfp8_linear_weight",
    "pack_weight",
    "pack_tensor_fp8_linear_weight",
    "prewarm",
    "prewarm_tensor_fp8_linear",
    "tensor_fp8_linear",
]
