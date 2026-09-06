"""Opaque MTP feedback launches with mandatory Qwen CuTe projections."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._cute_prefill import (
    get_cached_mtp_prefill_bf16_gemm,
    is_mtp_prefill_bf16_gemm_warmed,
)
from ._cute_prefill_config import (
    projection_capacity_rows,
    require_qwen_cute_tensors,
)


@triton.jit
def _token_norm_kernel(
    token_embedding,
    token_norm_weight,
    token_normalized,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN_SIZE
    offsets = token * HIDDEN_SIZE + cols.to(tl.int64)
    values = tl.load(token_embedding + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / HIDDEN_SIZE
    normalized = values * tl.rsqrt(variance + eps)
    weight = tl.load(token_norm_weight + cols, mask=mask, other=0.0).to(tl.float32)
    result = (normalized * (1.0 + weight)).to(tl.bfloat16)
    tl.store(token_normalized + offsets, result, mask=mask)


@triton.jit
def _state_partial_sum_kernel(
    multi_state,
    state_partial_sums,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN_SIZE
    offsets = row * HIDDEN_SIZE + cols.to(tl.int64)
    values = tl.load(multi_state + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(state_partial_sums + row, tl.sum(values * values, axis=0))


@triton.jit
def _state_norm_kernel(
    multi_state,
    state_partial_sums,
    state_norm_weight,
    state_normalized,
    eps,
    STREAMS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1).to(tl.int64)
    stream_offsets = tl.arange(0, BLOCK_S)
    sum_squares = tl.sum(
        tl.load(
            state_partial_sums + token * STREAMS + stream_offsets,
            mask=stream_offsets < STREAMS,
            other=0.0,
        ),
        axis=0,
    )
    inverse_rms = tl.rsqrt(sum_squares / (STREAMS * HIDDEN_SIZE) + eps)

    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN_SIZE
    row = token * STREAMS + stream
    offsets = row * HIDDEN_SIZE + cols.to(tl.int64)
    weight_offsets = stream * HIDDEN_SIZE + cols.to(tl.int64)
    values = tl.load(multi_state + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(state_norm_weight + weight_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    result = (values * inverse_rms * (1.0 + weight)).to(tl.bfloat16)
    tl.store(state_normalized + offsets, result, mask=mask)


def _scratch_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    nbytes = numel * int(dtype.itemsize)
    return scratch.narrow(0, int(offset_bytes), nbytes).view(dtype).view(shape)


def _capacity_matrix(tensor: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    flat = tensor.reshape(-1)
    required = int(rows) * int(columns)
    available = (
        int(flat.untyped_storage().nbytes())
        - int(flat.storage_offset()) * int(flat.element_size())
    ) // int(flat.element_size())
    if required > available:
        raise ValueError(
            f"padded CuTe view needs {required} elements, storage has {available}"
        )
    return flat.as_strided((int(rows), int(columns)), (int(columns), 1))


def _qwen_cute_projections(
    token_normalized: torch.Tensor,
    state_normalized: torch.Tensor,
    embedding_fc_weight: torch.Tensor,
    hidden_fc_weight: torch.Tensor,
    token_path: torch.Tensor,
    output: torch.Tensor,
    *,
    tokens: int,
    token_rows: int,
    state_rows: int,
    streams: int,
    hidden_size: int,
) -> None:
    state_live_rows = tokens * streams
    token_input = _capacity_matrix(token_normalized, token_rows, hidden_size)
    token_output = _capacity_matrix(token_path, token_rows, hidden_size)
    state_input = _capacity_matrix(state_normalized, state_rows, hidden_size)
    state_output = output.reshape(-1)

    with torch.cuda.device(token_normalized.device):
        token_projection = get_cached_mtp_prefill_bf16_gemm(
            token_rows,
            hidden_size,
            hidden_size,
            device=token_normalized.device,
            streams=streams,
            add_token_path=False,
        )
        state_projection = get_cached_mtp_prefill_bf16_gemm(
            state_rows,
            hidden_size,
            hidden_size,
            device=token_normalized.device,
            streams=streams,
            add_token_path=True,
        )
        if token_projection is None or state_projection is None:
            raise RuntimeError(
                "MTP CuTe capacity kernels were not compiled by plan(); "
                "request-time projection compilation is disabled"
            )
        if torch.cuda.is_current_stream_capturing():
            token_warmed = token_projection is not None and (
                is_mtp_prefill_bf16_gemm_warmed(
                    token_rows,
                    hidden_size,
                    hidden_size,
                    device=token_normalized.device,
                    streams=streams,
                    add_token_path=False,
                )
            )
            state_warmed = state_projection is not None and (
                is_mtp_prefill_bf16_gemm_warmed(
                    state_rows,
                    hidden_size,
                    hidden_size,
                    device=token_normalized.device,
                    streams=streams,
                    add_token_path=True,
                )
            )
            if not token_warmed or not state_warmed:
                raise RuntimeError(
                    "MTP CuTe kernels must be warm-run before CUDA graph capture"
                )
        assert token_projection is not None
        assert state_projection is not None
        token_projection(
            token_input,
            embedding_fc_weight,
            token_output,
            live_rows=tokens,
        )
        state_projection(
            state_input,
            hidden_fc_weight,
            state_output,
            token_path=token_output,
            live_rows=state_live_rows,
        )


def _launch_mtp_feedback(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor,
    hidden_fc_weight: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    max_tokens: int,
    streams: int,
    hidden_size: int,
    token_normalized_offset_bytes: int,
    state_partial_sums_offset_bytes: int,
    state_normalized_offset_bytes: int,
    token_path_offset_bytes: int,
    token_projection_rows: int,
    state_projection_rows: int,
    norm_block_h: int,
    norm_block_s: int,
    norm_num_warps: int,
) -> None:
    tokens = int(token_embedding.shape[0])
    token_normalized = _scratch_view(
        scratch,
        offset_bytes=token_normalized_offset_bytes,
        shape=(token_projection_rows, hidden_size),
        dtype=torch.bfloat16,
    )[:tokens]
    state_partial_sums = _scratch_view(
        scratch,
        offset_bytes=state_partial_sums_offset_bytes,
        shape=(max_tokens, streams),
        dtype=torch.float32,
    )[:tokens]
    state_normalized = _scratch_view(
        scratch,
        offset_bytes=state_normalized_offset_bytes,
        shape=(state_projection_rows // streams, streams, hidden_size),
        dtype=torch.bfloat16,
    )[:tokens]
    token_path = _scratch_view(
        scratch,
        offset_bytes=token_path_offset_bytes,
        shape=(token_projection_rows, hidden_size),
        dtype=torch.bfloat16,
    )[:tokens]

    expected_token_rows, expected_state_rows = projection_capacity_rows(
        max_tokens=max_tokens,
        streams=streams,
        hidden_size=hidden_size,
    )
    if (token_projection_rows, state_projection_rows) != (
        expected_token_rows,
        expected_state_rows,
    ):
        raise ValueError(
            "MTP projection capacity does not match the planned CuTe "
            f"specialization: got {token_projection_rows}/{state_projection_rows}, "
            f"expected {expected_token_rows}/{expected_state_rows}"
        )
    projection_tensors = {
        "token_normalized": token_normalized,
        "state_normalized": state_normalized,
        "embedding_fc_weight": embedding_fc_weight,
        "hidden_fc_weight": hidden_fc_weight,
        "token_path": token_path,
        "output": output,
    }
    require_qwen_cute_tensors(**projection_tensors)
    _token_norm_kernel[(tokens,)](
        token_embedding,
        token_norm_weight,
        token_normalized,
        float(eps),
        HIDDEN_SIZE=int(hidden_size),
        BLOCK_H=int(norm_block_h),
        num_warps=int(norm_num_warps),
        num_stages=1,
    )
    _state_partial_sum_kernel[(tokens * streams,)](
        multi_state,
        state_partial_sums,
        HIDDEN_SIZE=int(hidden_size),
        BLOCK_H=int(norm_block_h),
        num_warps=int(norm_num_warps),
        num_stages=1,
    )
    _state_norm_kernel[(tokens, streams)](
        multi_state,
        state_partial_sums,
        state_norm_weight,
        state_normalized,
        float(eps),
        STREAMS=int(streams),
        HIDDEN_SIZE=int(hidden_size),
        BLOCK_S=int(norm_block_s),
        BLOCK_H=int(norm_block_h),
        num_warps=int(norm_num_warps),
        num_stages=1,
    )
    _qwen_cute_projections(
        token_normalized,
        state_normalized,
        embedding_fc_weight,
        hidden_fc_weight,
        token_path,
        output,
        tokens=tokens,
        token_rows=token_projection_rows,
        state_rows=state_projection_rows,
        streams=streams,
        hidden_size=hidden_size,
    )


@torch.library.custom_op(
    "b12x::mtp_feedback",
    mutates_args=("scratch", "output"),
)
def _mtp_feedback_op(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor,
    hidden_fc_weight: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    max_tokens: int,
    streams: int,
    hidden_size: int,
    token_normalized_offset_bytes: int,
    state_partial_sums_offset_bytes: int,
    state_normalized_offset_bytes: int,
    token_path_offset_bytes: int,
    token_projection_rows: int,
    state_projection_rows: int,
    norm_block_h: int,
    norm_block_s: int,
    norm_num_warps: int,
) -> None:
    _launch_mtp_feedback(
        token_embedding,
        multi_state,
        token_norm_weight,
        state_norm_weight,
        embedding_fc_weight,
        hidden_fc_weight,
        scratch,
        output,
        eps,
        max_tokens,
        streams,
        hidden_size,
        token_normalized_offset_bytes,
        state_partial_sums_offset_bytes,
        state_normalized_offset_bytes,
        token_path_offset_bytes,
        token_projection_rows,
        state_projection_rows,
        norm_block_h,
        norm_block_s,
        norm_num_warps,
    )


@_mtp_feedback_op.register_fake
def _mtp_feedback_fake(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor,
    hidden_fc_weight: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    max_tokens: int,
    streams: int,
    hidden_size: int,
    token_normalized_offset_bytes: int,
    state_partial_sums_offset_bytes: int,
    state_normalized_offset_bytes: int,
    token_path_offset_bytes: int,
    token_projection_rows: int,
    state_projection_rows: int,
    norm_block_h: int,
    norm_block_s: int,
    norm_num_warps: int,
) -> None:
    del token_embedding, multi_state, token_norm_weight, state_norm_weight
    del embedding_fc_weight, hidden_fc_weight, scratch, output, eps
    del max_tokens, streams, hidden_size
    del token_normalized_offset_bytes, state_partial_sums_offset_bytes
    del state_normalized_offset_bytes, token_path_offset_bytes
    del token_projection_rows, state_projection_rows
    del norm_block_h, norm_block_s, norm_num_warps


def run_mtp_feedback(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor,
    hidden_fc_weight: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    *,
    eps: float,
    max_tokens: int,
    streams: int,
    hidden_size: int,
    token_normalized_offset_bytes: int,
    state_partial_sums_offset_bytes: int,
    state_normalized_offset_bytes: int,
    token_path_offset_bytes: int,
    token_projection_rows: int,
    state_projection_rows: int,
    norm_block_h: int,
    norm_block_s: int,
    norm_num_warps: int,
) -> None:
    torch.ops.b12x.mtp_feedback(
        token_embedding,
        multi_state,
        token_norm_weight,
        state_norm_weight,
        embedding_fc_weight,
        hidden_fc_weight,
        scratch,
        output,
        float(eps),
        int(max_tokens),
        int(streams),
        int(hidden_size),
        int(token_normalized_offset_bytes),
        int(state_partial_sums_offset_bytes),
        int(state_normalized_offset_bytes),
        int(token_path_offset_bytes),
        int(token_projection_rows),
        int(state_projection_rows),
        int(norm_block_h),
        int(norm_block_s),
        int(norm_num_warps),
    )


__all__ = ["run_mtp_feedback"]
