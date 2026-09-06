"""Triton kernels for learned low-rank HyperConnection primitives."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._cute_config import require_cute_combine_norm


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    span = (
        0
        if tensor.numel() == 0
        else 1
        + sum(
            (int(size) - 1) * int(stride)
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    )
    return start, start + span * int(tensor.element_size())


def _require_disjoint(
    output_name: str,
    output: torch.Tensor,
    inputs: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    output_start, output_end = _byte_interval(output)
    for input_name, tensor in inputs:
        input_start, input_end = _byte_interval(tensor)
        if output_start < input_end and input_start < output_end:
            raise ValueError(f"{output_name} must not overlap {input_name}")


@triton.jit
def _grouped_rmsnorm_kernel(
    state_ptr,
    weight_ptr,
    out_ptr,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H).to(tl.int64)
    mask = cols < HIDDEN_SIZE
    offsets = row * HIDDEN_SIZE + cols
    values = tl.load(state_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / HIDDEN_SIZE
    inv_rms = tl.rsqrt(variance + eps)
    weight_offsets = (row % STREAMS) * HIDDEN_SIZE + cols
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, values * inv_rms * (1.0 + weight), mask=mask)


@triton.jit
def _scaled_silu_kernel(
    projected_ptr,
    out_ptr,
    elements,
    STREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < elements
    values = tl.load(projected_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = (values / STREAMS).to(tl.bfloat16)
    activated = scaled.to(tl.float32) * tl.sigmoid(scaled.to(tl.float32))
    tl.store(out_ptr + offsets, activated, mask=mask)


@triton.jit
def _gate_mean_kernel(
    normalized_ptr,
    gate_logits_ptr,
    out_ptr,
    HIDDEN_SIZE: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1).to(tl.int64)
    cols = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    mask = cols < HIDDEN_SIZE
    token_base = token * STREAMS * HIDDEN_SIZE
    total = tl.zeros((BLOCK_H,), tl.float32)
    for stream in tl.static_range(0, STREAMS):
        offsets = token_base + stream * HIDDEN_SIZE + cols
        values = tl.load(normalized_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
        logits = tl.load(gate_logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(logits).to(tl.bfloat16)
        product = (gate * values).to(tl.bfloat16)
        total += product.to(tl.float32)
    out_offsets = token * HIDDEN_SIZE + cols
    tl.store(out_ptr + out_offsets, total / STREAMS, mask=mask)


@triton.jit
def _combine_kernel(
    state_ptr,
    block_output_ptr,
    injection_logits_ptr,
    combined_ptr,
    HIDDEN_SIZE: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK_H).to(tl.int64)
    mask = cols < HIDDEN_SIZE
    state_offsets = (token * STREAMS + stream) * HIDDEN_SIZE + cols
    output_offsets = token * HIDDEN_SIZE + cols
    state = tl.load(state_ptr + state_offsets, mask=mask, other=0.0).to(tl.float32)
    block_output = tl.load(block_output_ptr + output_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    logit = tl.load(injection_logits_ptr + token * STREAMS + stream).to(tl.float32)
    scale = 2.0 * tl.sigmoid(logit / STREAMS)
    tl.store(
        combined_ptr + state_offsets,
        state + scale * block_output,
        mask=mask,
    )


def _norm_launch(
    state: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> None:
    _grouped_rmsnorm_kernel[(int(state.shape[0]) * streams,)](
        state,
        weight,
        out,
        float(eps),
        HIDDEN_SIZE=hidden_size,
        STREAMS=streams,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )


def _scaled_silu_launch(
    projected_down: torch.Tensor,
    out: torch.Tensor,
    streams: int,
    block: int,
) -> None:
    elements = int(projected_down.numel())
    _scaled_silu_kernel[(triton.cdiv(elements, block),)](
        projected_down,
        out,
        elements,
        STREAMS=streams,
        BLOCK=block,
        num_warps=4,
    )


def _gate_mean_launch(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    out: torch.Tensor,
    streams: int,
    hidden_size: int,
    block_h: int,
) -> None:
    _gate_mean_kernel[(int(normalized.shape[0]), triton.cdiv(hidden_size, block_h))](
        normalized,
        gate_logits,
        out,
        HIDDEN_SIZE=hidden_size,
        STREAMS=streams,
        BLOCK_H=block_h,
        num_warps=4,
    )


def _combine_launch(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    combined: torch.Tensor,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> None:
    _combine_kernel[(int(state.shape[0]), streams)](
        state,
        block_output,
        injection_logits,
        combined,
        HIDDEN_SIZE=hidden_size,
        STREAMS=streams,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )


@torch.library.custom_op("b12x::hyperconnection_grouped_rmsnorm", mutates_args=("out",))
def _grouped_rmsnorm_op(
    state: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> None:
    _require_disjoint("normalized", out, (("state", state), ("weight", weight)))
    _norm_launch(
        state,
        weight,
        out,
        eps,
        streams,
        hidden_size,
        block_h,
        num_warps,
    )


@_grouped_rmsnorm_op.register_fake
def _grouped_rmsnorm_fake(
    state: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> None:
    del state, weight, out, eps, streams, hidden_size, block_h, num_warps


@torch.library.custom_op("b12x::hyperconnection_scaled_silu", mutates_args=("out",))
def _scaled_silu_op(
    projected_down: torch.Tensor,
    out: torch.Tensor,
    streams: int,
    block: int,
) -> None:
    _require_disjoint("bottleneck", out, (("projected_down", projected_down),))
    from ._cute import scaled_silu

    scaled_silu(projected_down, out, streams=streams)


@_scaled_silu_op.register_fake
def _scaled_silu_fake(
    projected_down: torch.Tensor,
    out: torch.Tensor,
    streams: int,
    block: int,
) -> None:
    del projected_down, out, streams, block


@torch.library.custom_op("b12x::hyperconnection_gate_mean", mutates_args=("out",))
def _gate_mean_op(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    out: torch.Tensor,
    streams: int,
    hidden_size: int,
    block_h: int,
) -> None:
    _require_disjoint(
        "block_input",
        out,
        (("normalized", normalized), ("gate_logits", gate_logits)),
    )
    _gate_mean_launch(normalized, gate_logits, out, streams, hidden_size, block_h)


@_gate_mean_op.register_fake
def _gate_mean_fake(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    out: torch.Tensor,
    streams: int,
    hidden_size: int,
    block_h: int,
) -> None:
    del normalized, gate_logits, out, streams, hidden_size, block_h


@torch.library.custom_op(
    "b12x::hyperconnection_combine",
    mutates_args=(),
)
def _combine_op(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> torch.Tensor:
    combined = torch.empty_like(state)
    if int(state.shape[0]) != 0:
        from ._cute import combine

        combine(
            state,
            block_output,
            injection_logits,
            combined,
            streams=streams,
            hidden_size=hidden_size,
        )
    return combined


@_combine_op.register_fake
def _combine_fake(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> torch.Tensor:
    del block_output, injection_logits, streams, hidden_size, block_h, num_warps
    return torch.empty_like(state)


@torch.library.custom_op(
    "b12x::hyperconnection_combine_norm",
    mutates_args=(),
)
def _combine_norm_op(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    eps: float,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.empty_like(state)
    normalized = torch.empty_like(state)
    require_cute_combine_norm(
        state=state,
        block_output=block_output,
        injection_logits=injection_logits,
        next_norm_weight=next_norm_weight,
        combined=combined,
        normalized=normalized,
        streams=streams,
        hidden_size=hidden_size,
    )
    if int(state.shape[0]) != 0:
        from ._cute import combine_norm as combine_norm_cute

        combine_norm_cute(
            state,
            block_output,
            injection_logits,
            next_norm_weight,
            combined,
            normalized,
            eps=eps,
            streams=streams,
            hidden_size=hidden_size,
        )
    return combined, normalized


@_combine_norm_op.register_fake
def _combine_norm_fake(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    eps: float,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, next_norm_weight, eps
    del streams, hidden_size, block_h, num_warps
    return torch.empty_like(state), torch.empty_like(state)


def run_grouped_rmsnorm(
    state: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> None:
    torch.ops.b12x.hyperconnection_grouped_rmsnorm(
        state,
        weight,
        out,
        float(eps),
        int(streams),
        int(hidden_size),
        int(block_h),
        int(num_warps),
    )


def run_scaled_silu(
    projected_down: torch.Tensor,
    out: torch.Tensor,
    *,
    streams: int,
    block: int,
) -> None:
    torch.ops.b12x.hyperconnection_scaled_silu(
        projected_down, out, int(streams), int(block)
    )


def run_gate_mean(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    out: torch.Tensor,
    *,
    streams: int,
    hidden_size: int,
    block_h: int,
) -> None:
    torch.ops.b12x.hyperconnection_gate_mean(
        normalized,
        gate_logits,
        out,
        int(streams),
        int(hidden_size),
        int(block_h),
    )


def run_combine(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    *,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> torch.Tensor:
    return torch.ops.b12x.hyperconnection_combine(
        state,
        block_output,
        injection_logits,
        int(streams),
        int(hidden_size),
        int(block_h),
        int(num_warps),
    )


def run_combine_norm(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    *,
    eps: float,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.b12x.hyperconnection_combine_norm(
        state,
        block_output,
        injection_logits,
        next_norm_weight,
        float(eps),
        int(streams),
        int(hidden_size),
        int(block_h),
        int(num_warps),
    )


__all__ = [
    "run_grouped_rmsnorm",
    "run_scaled_silu",
    "run_gate_mean",
    "run_combine",
    "run_combine_norm",
]
