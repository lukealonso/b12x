"""Opaque one-shot boundaries for Dynamo and caller-owned GEMM buffers."""

from __future__ import annotations

import torch


def _execute_impl(source, values, scales, global_scale, activation_scale, workspace, out,
                  fp4, reciprocal, input_k, mode, expected_m, config, stream):
    from ._a16 import NVFP4LinearWeight, bf16_linear
    from ._linear import MXFP8LinearWeight
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
    if fp4:
        weight = NVFP4LinearWeight(values, scales, global_scale,
                                   "reciprocal" if reciprocal else "multiplier",
                                   input_k, values.shape[0])
    else:
        # This invocation consumes only the shared MMA scale storage.
        scales = scales.view(torch.float8_e8m0fnu)
        weight = MXFP8LinearWeight(MXFP8Rows(values, None, scales), input_k,
                                   values.shape[1], values.shape[0])
        if mode in ("auto", "quantized") and out is None and workspace is None and config is None:
            from ._a16 import _select_mode
            from ._linear import mxfp8_linear
            selected, _ = _select_mode(mode, source, weight)
            if selected == "quantized":
                if activation_scale is not None:
                    raise ValueError("MXFP8 does not use an activation global scale")
                return mxfp8_linear(source, weight, expected_m=expected_m, stream=stream)
    return bf16_linear(source, weight, out=out, workspace=workspace, mode=mode,
                       activation_global_scale=activation_scale,
                       expected_m=expected_m, _config=config, stream=stream)


def _execute(source, values, scales, global_scale, activation_scale, workspace, out,
             fp4, reciprocal, input_k, mode, expected_m, config, stream):
    from ._a16 import _stream_context
    if source.device.type != "cuda":
        raise ValueError("blockscaled BF16 input requires CUDA")
    with torch.cuda.device(source.device), _stream_context(stream, source.device):
        return _execute_impl(source, values, scales, global_scale, activation_scale,
                             workspace, out, fp4, reciprocal, input_k, mode,
                             expected_m, config, stream)


@torch.library.custom_op("b12x::blockscaled_bf16", mutates_args=("workspace",))
def _functional(
    source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
    global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
    workspace: torch.Tensor | None, fp4: bool, reciprocal: bool, input_k: int,
    activation_mode: str, expected_m: int | None, config: list[int] | None, stream: int | None,
) -> torch.Tensor:
    return _execute(source, values, scales, global_scale, activation_scale, workspace,
                    None, fp4, reciprocal, input_k, activation_mode, expected_m, config, stream)


@_functional.register_fake
def _functional_fake(source, values, scales, global_scale, activation_scale,
                     workspace, fp4, reciprocal, input_k, activation_mode, expected_m, config, stream):
    return source.new_empty((*source.shape[:-1], values.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("b12x::blockscaled_bf16_out", mutates_args=("out", "workspace"))
def _out(
    source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
    global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
    workspace: torch.Tensor | None, out: torch.Tensor, fp4: bool, reciprocal: bool,
    input_k: int, activation_mode: str, expected_m: int | None, config: list[int] | None,
    stream: int | None,
) -> None:
    _execute(source, values, scales, global_scale, activation_scale, workspace,
             out, fp4, reciprocal, input_k, activation_mode, expected_m, config, stream)


@_out.register_fake
def _out_fake(source, values, scales, global_scale, activation_scale,
              workspace, out, fp4, reciprocal, input_k, activation_mode, expected_m, config, stream):
    return None


def linear(source, values, scales, global_scale, *, fp4, reciprocal=False,
           input_k=None, mode="auto", expected_m=None, _config=None,
           activation_global_scale=None, out=None, workspace=None, stream=None):
    from b12x._lib.utils import cuda_stream_to_int
    input_k = values.shape[1] * (2 if fp4 else 1) if input_k is None else input_k
    config = None if _config is None else list(_config)
    # Inductor cannot decompose auto-functionalization with UE8M0 inputs.
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    args = (source, values, scales, global_scale, activation_global_scale, workspace)
    options = (fp4, reciprocal, input_k, mode, expected_m, config, cuda_stream_to_int(stream))
    if out is None:
        return _functional(*args, *options)
    _out(*args, out, *options)
    return out
