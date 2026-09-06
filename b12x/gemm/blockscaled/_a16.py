"""One-shot BF16 activation GEMMs with shared NVFP4/MXFP8 weight storage."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
import triton

from . import _quantize as kernels

Mode = Literal["auto", "a16", "quantized"]


@dataclass(frozen=True)
class NVFP4LinearWeight:
    """Shared E2M1 values, swizzled K/16 E4M3 scales, and weight-only scale.

    ``global_scale_kind='multiplier'`` reconstructs weights as
    ``E2M1 * E4M3 * global_scale``. ``'reciprocal'`` divides by that tensor.
    Tensors are borrowed; callers must keep them alive and unmodified during use.
    """

    values: torch.Tensor
    scale_mma: torch.Tensor
    global_scale: torch.Tensor
    global_scale_kind: str
    in_features: int
    out_features: int

    @property
    def padded_in_features(self) -> int:
        return self.in_features


def _check_tensor(name, tensor, device, dtype=None):
    if not isinstance(tensor, torch.Tensor) or tensor.device != device:
        raise ValueError(f"{name} must be a tensor on {device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.numel() and tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def scale_storage(scale: torch.Tensor, n: int, k: int, group: int) -> torch.Tensor:
    """Return a zero-copy flat byte view of the F8_128x4 physical storage."""
    if scale.dtype not in (torch.uint8, torch.float8_e4m3fn, torch.float8_e8m0fnu):
        raise ValueError("block scales must have uint8/E4M3/UE8M0 storage")
    nt, kt = triton.cdiv(n, 128), triton.cdiv(k // group, 4)
    if scale.ndim == 6:
        if tuple(scale.shape) != (32, 4, nt, 4, kt, 1):
            raise ValueError("block scale MMA view has the wrong shape")
        physical = scale.permute(5, 2, 4, 0, 1, 3)
    else:
        physical = scale
    if not physical.is_contiguous() or physical.numel() != nt * kt * 512:
        raise ValueError("block scales must use contiguous F8_128x4 swizzled storage")
    return physical.view(torch.uint8).view(-1)


def pack_nvfp4_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    global_scale_kind: str = "multiplier",
) -> NVFP4LinearWeight:
    if weight.device.type != "cuda":
        raise ValueError("NVFP4 weights must be on CUDA")
    _check_tensor("weight", weight, weight.device, torch.uint8)
    if weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("NVFP4 weight must have positive shape [N,K/2]")
    n, storage_k = weight.shape
    k = storage_k * 2
    if k % 16:
        raise ValueError("NVFP4 K must be divisible by 16")
    if scale.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("NVFP4 requires E4M3 block scales")
    storage = scale_storage(scale, n, k, 16)
    _check_tensor("scale", storage, weight.device, torch.uint8)
    _check_tensor("global_scale", global_scale, weight.device, torch.float32)
    if global_scale.numel() != 1 or global_scale_kind not in ("multiplier", "reciprocal"):
        raise ValueError("global_scale must be scalar, with kind 'multiplier' or 'reciprocal'")
    from b12x._lib.intrinsics import as_grouped_scale_view
    mma = as_grouped_scale_view(storage.view(1, -1), n, k)
    return NVFP4LinearWeight(weight, mma, global_scale, global_scale_kind, k, n)


def _stream_context(stream, device):
    if stream is None:
        return nullcontext()
    if not isinstance(stream, torch.cuda.Stream):
        from b12x._lib.utils import cuda_stream_to_int
        stream = torch.cuda.ExternalStream(cuda_stream_to_int(stream), device=device)
    if stream.device != device:
        raise ValueError("stream must be on the operand device")
    return torch.cuda.stream(stream)


def _overlap(a, b):
    if not a.numel() or not b.numel():
        return False
    return (a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
            and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size())


def _validate_output(source, out, n):
    shape = (*source.shape[:-1], n)
    if out is None:
        return torch.empty(shape, device=source.device, dtype=torch.bfloat16)
    _check_tensor("out", out, source.device, torch.bfloat16)
    if tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape}")
    return out


def _validate_source(source, k):
    if source.device.type != "cuda":
        raise ValueError("blockscaled BF16 input requires CUDA")
    _check_tensor("source", source, source.device, torch.bfloat16)
    if source.ndim < 2 or source.shape[-1] != k:
        raise ValueError(f"source must have shape [...,M,{k}]")
    if torch.cuda.get_device_capability(source.device) not in ((12, 0), (12, 1)):
        raise ValueError("standalone blockscaled A16 kernels require SM120/SM121")


def _config(config):
    config = (64, 64, 1) if config is None else tuple(config)
    if len(config) != 3 or config[0] not in (64, 128) or config[1] not in (64, 128) or config[2] not in (1, 2, 4, 8):
        raise ValueError("A16 config must be (N tile 64/128, K tile 64/128, split-K 1/2/4/8)")
    return config


def _a16(source, values, scales, global_scale, out, workspace, *, fp4,
         reciprocal=False, input_k=None, config=None, stream=None):
    n, storage_k = values.shape
    k = storage_k * 2 if fp4 else storage_k
    input_k = k if input_k is None else input_k
    _validate_source(source, input_k)
    _check_tensor("weight", values, source.device,
                  torch.uint8 if fp4 else torch.float8_e4m3fn)
    if k % (16 if fp4 else 32) or n <= 0 or k <= 0 or not 0 < input_k <= k:
        raise ValueError("invalid blockscaled weight geometry")
    if k % 32 or input_k % 8 or n % 8:
        raise ValueError("A16 TMA requires K divisible by 32, input K by 8, and N by 8")
    scales = scale_storage(scales, n, k, 16 if fp4 else 32)
    _check_tensor("scale", scales, source.device, torch.uint8)
    if fp4:
        _check_tensor("global_scale", global_scale, source.device, torch.float32)
        if global_scale.numel() != 1:
            raise ValueError("global_scale must contain one value")
    out = _validate_output(source, out, n)
    reads = (source, values, scales) + ((global_scale,) if fp4 else ())
    if any(_overlap(out, tensor) for tensor in reads):
        raise ValueError("out must not overlap inputs")
    m = source.numel() // input_k
    bn, bk, split = _config(config)
    split = min(split, triton.cdiv(input_k, bk))
    if m == 0:
        return out
    if split > 1:
        needed = split * m * n * 4
        if workspace is None:
            workspace = torch.empty(needed, dtype=torch.uint8, device=source.device)
        _check_tensor("workspace", workspace, source.device, torch.uint8)
        if workspace.numel() < needed:
            raise ValueError(f"workspace requires at least {needed} bytes")
        if any(_overlap(workspace, tensor) for tensor in (*reads, out)):
            raise ValueError("workspace must not overlap inputs or output")
        result = workspace[:needed].view(torch.float32)
    else:
        result = out
    with torch.cuda.device(source.device), _stream_context(stream, source.device):
        from b12x._lib.dense_gemm import dense_gemm_a16, dense_gemm_a16_reduce
        dense_gemm_a16(source, values, scales, global_scale, result,
                       recipe="nvfp4" if fp4 else "mxfp8", input_k=input_k,
                       config=(bn, bk, split), reciprocal=reciprocal, stream=stream)
        if split > 1:
            dense_gemm_a16_reduce(result, out, n=n, m=m, slices=split, stream=stream)
    return out


def w4a16(
    source: torch.Tensor,
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    global_scale_kind: str = "multiplier",
    stream: object = None,
    _config: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """NVFP4 W4A16 using shared packed weights and swizzled block scales.

    No weight preparation or activation quantization is performed. The global
    scale is weight-only; specify ``'reciprocal'`` for a quantizer multiplier.
    ``workspace`` is contiguous uint8 storage for optional split-K partials.
    """
    if global_scale_kind not in ("multiplier", "reciprocal"):
        raise ValueError("global_scale_kind must be 'multiplier' or 'reciprocal'")
    if block_scale.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("NVFP4 requires E4M3 block scales")
    from ._ops import linear
    return linear(source, weight, block_scale, global_scale, out=out, workspace=workspace,
                  fp4=True, mode="a16", reciprocal=global_scale_kind == "reciprocal",
                  _config=_config, stream=stream)


def w8a16(
    source: torch.Tensor,
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    stream: object = None,
    _config: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """MXFP8 W8A16 using E4M3 weights and shared swizzled UE8M0 scales."""
    if block_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError("MXFP8 requires UE8M0 block scales")
    from ._ops import linear
    return linear(source, weight, block_scale, None, out=out, workspace=workspace,
                  fp4=False, mode="a16", _config=_config, stream=stream)


def _weight_parts(weight):
    from ._linear import MXFP8LinearWeight
    if isinstance(weight, NVFP4LinearWeight):
        return weight.values, weight.scale_mma, weight.global_scale, True
    if isinstance(weight, MXFP8LinearWeight):
        return weight.weight.values, weight.weight.scale_mma, None, False
    raise TypeError("BF16 blockscaled execution requires NVFP4 or MXFP8 weights")


def _layout(m, n, k, fp4, config=None):
    """Byte offsets into caller-owned scratch; all kernel pointers stay aligned."""
    value_bytes = m * (k // 2 if fp4 else k)
    scale_bytes = triton.cdiv(m, 128) * triton.cdiv(k // (16 if fp4 else 32), 4) * 512
    scale_start = triton.cdiv(value_bytes, 256) * 256
    alpha_start = triton.cdiv(scale_start + scale_bytes, 256) * 256
    partial_start = alpha_start + 256
    split = _config(config)[2]
    partial_bytes = max(
        split * m * n * 4 if split > 1 else 0,
        2 * min(m, 8) * n * 4 if not fp4 else 0,
    )
    return scale_start, alpha_start, partial_start, partial_start + partial_bytes


def workspace_size(weight, max_tokens: int, *, _config=None) -> int:
    """Required uint8 scratch bytes for one-shot BF16-input calls up to a capacity.

    The same allocation can serve either activation precision, sequentially.
    Concurrent invocations require disjoint workspaces and outputs.
    """
    _weight_parts(weight)
    if not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    needed = _layout(max_tokens, weight.out_features, weight.padded_in_features,
                     isinstance(weight, NVFP4LinearWeight), _config)[-1]
    if _config is None and weight.in_features == weight.padded_in_features:
        from ._policy import resolve_precision
        values, _, _, fp4 = _weight_parts(weight)
        resolution = resolve_precision(values.device, "nvfp4" if fp4 else "mxfp8",
                                       weight.in_features, weight.out_features)
        for m, _, bk, split in resolution.config.a16_rows:
            if m <= max_tokens:
                needed = max(needed, m * weight.out_features * 4 * min(split, triton.cdiv(weight.in_features, bk)))
    return needed


def _select_mode(mode, source, weight):
    if mode not in ("auto", "a16", "quantized"):
        raise ValueError("mode must be 'auto', 'a16', or 'quantized'")
    if mode == "quantized":
        return mode, None
    if (weight.in_features != weight.padded_in_features
            or not source.is_contiguous() or source.data_ptr() % 16):
        return ("a16" if mode == "a16" else "quantized"), None
    from ._policy import resolve_precision
    recipe = "nvfp4" if isinstance(weight, NVFP4LinearWeight) else "mxfp8"
    resolution = resolve_precision(source.device, recipe, weight.in_features, weight.out_features)
    config = resolution.config.select(source.numel() // weight.in_features)
    return ("a16", config) if mode == "a16" or config is not None else ("quantized", None)


def bf16_linear(source, weight, *, out=None, workspace=None, mode="auto",
                activation_global_scale=None, stream=None, expected_m=None,
                _config=None):
    """BF16-input one-shot dispatch, with all quantization buffers in scratch."""
    values, scales, global_scale, fp4 = _weight_parts(weight)
    if expected_m is not None and expected_m <= 0:
        raise ValueError("expected_m must be positive when provided")
    _validate_source(source, weight.in_features)
    _check_tensor("weight", values, source.device,
                  torch.uint8 if fp4 else torch.float8_e4m3fn)
    n, k = weight.out_features, weight.padded_in_features
    m = source.numel() // weight.in_features
    selected, selected_config = _select_mode(mode, source, weight)
    config = _config if _config is not None else selected_config
    if selected == "a16":
        return _a16(source, values, scales, global_scale, out, workspace, fp4=fp4,
                    reciprocal=fp4 and weight.global_scale_kind == "reciprocal",
                    input_k=weight.in_features, config=config, stream=stream)
    if _config is not None:
        raise ValueError("A16 launch configuration cannot be used with quantized execution")
    if k % 128 or n % 8:
        raise ValueError("quantized blockscaled execution requires K divisible by 128 and N by 8")
    if fp4:
        if activation_global_scale is None:
            raise ValueError("NVFP4 quantized execution requires activation_global_scale (quantizer multiplier)")
        _check_tensor("activation_global_scale", activation_global_scale, source.device, torch.float32)
        _check_tensor("global_scale", global_scale, source.device, torch.float32)
        if activation_global_scale.numel() != 1 or global_scale.numel() != 1:
            raise ValueError("activation and weight global scales must be scalar")
    elif activation_global_scale is not None:
        raise ValueError("MXFP8 does not use an activation global scale")
    out = _validate_output(source, out, n)
    if m == 0:
        return out
    scale_bytes = scale_storage(scales, n, k, 16 if fp4 else 32)
    _check_tensor("weight scale", scale_bytes, source.device, torch.uint8)
    reads = (source, values, scale_bytes) + ((global_scale, activation_global_scale) if fp4 else ())
    if any(_overlap(out, tensor) for tensor in reads):
        raise ValueError("out must not overlap inputs")
    sf_start, alpha_start, partial_start, needed = _layout(m, n, k, fp4)
    if workspace is None:
        workspace = torch.empty(needed, device=source.device, dtype=torch.uint8)
    _check_tensor("workspace", workspace, source.device, torch.uint8)
    if workspace.numel() < needed:
        raise ValueError(f"workspace requires at least {needed} bytes")
    if any(_overlap(workspace, tensor) for tensor in (*reads, out)):
        raise ValueError("workspace must not overlap inputs or output")
    storage_k = k // 2 if fp4 else k
    q = workspace[:m * storage_k].view(torch.uint8 if fp4 else torch.float8_e4m3fn)
    sf = workspace[sf_start:sf_start + triton.cdiv(m, 128) * triton.cdiv(k // (16 if fp4 else 32), 4) * 512]
    alpha = workspace[alpha_start:alpha_start + 4].view(torch.float32)
    partials = workspace[partial_start:needed].view(torch.float32)
    from b12x._lib.dense_gemm import dense_gemm
    from b12x._lib.intrinsics import as_grouped_scale_view, as_grouped_scale_view_mx
    sf_view = (as_grouped_scale_view if fp4 else as_grouped_scale_view_mx)(sf.view(1, -1), m, k)
    with torch.cuda.device(source.device), _stream_context(stream, source.device):
        kernels.launch(kernels._quantize,
                       (source, q, sf, activation_global_scale if fp4 else alpha,
                        global_scale if fp4 else alpha, alpha, m),
                       dict(INPUT_K=weight.in_features, K=k, FP4=fp4,
                            RECIPROCAL=fp4 and weight.global_scale_kind == "reciprocal",
                            GROUP=16 if fp4 else 32, CHUNKS=16),
                       (m, triton.cdiv(k // (16 if fp4 else 32), 16), 1),
                       device=source.device, num_stages=1)
        dense_gemm(
            (q.view(m, storage_k, 1), sf_view), (values.view(n, storage_k, 1), scales),
            out=out.view(m, n, 1), alpha=alpha,
            ab_dtype="float4_e2m1fn" if fp4 else "float8_e4m3fn",
            sf_dtype="float8_e4m3fn" if fp4 else "float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=16 if fp4 else 32,
            expected_m=expected_m, stream=stream, _split_k_workspace=partials,
        )
    return out
