"""Triton kernels for packed EOS-bounded PLE hashing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._contracts import Binding

_ERROR_CAPACITY = tl.constexpr(1)
_ERROR_QUERY_START = tl.constexpr(2)
_ERROR_TOKEN_ID = tl.constexpr(4)


@triton.jit
def _reset_error_kernel(error_code_ptr):
    tl.store(error_code_ptr, 0)


@triton.jit(do_not_specialize=["token_count"])
def _validate_metadata_kernel(
    token_ids_ptr,
    query_start_loc_ptr,
    committed_history_ptr,
    num_seqs_ptr,
    num_tokens_ptr,
    error_code_ptr,
    token_count,
    VOCAB_SIZE: tl.constexpr,
    MAX_ORDER: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    index = tl.program_id(0)
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    if index == 0:
        invalid_capacity = (
            (num_seqs < 0)
            | (num_seqs > MAX_SEQS)
            | (num_tokens < 0)
            | (num_tokens > MAX_TOKENS)
            | (num_tokens > token_count)
            | ((num_seqs == 0) & (num_tokens > 0))
        )
        tl.atomic_or(
            error_code_ptr,
            tl.where(invalid_capacity, _ERROR_CAPACITY, 0),
        )

    request_live = (index < MAX_SEQS) & (index < num_seqs)
    start = tl.load(query_start_loc_ptr + index, mask=request_live, other=0).to(
        tl.int32
    )
    end = tl.load(query_start_loc_ptr + index + 1, mask=request_live, other=0).to(
        tl.int32
    )
    invalid_start = request_live & (
        (start < 0)
        | (end < start)
        | (end > num_tokens)
        | ((index == 0) & (start != 0))
        | ((index == num_seqs - 1) & (end != num_tokens))
    )
    tl.atomic_or(
        error_code_ptr,
        tl.where(invalid_start, _ERROR_QUERY_START, 0),
    )
    for history in tl.static_range(0, MAX_ORDER - 1):
        history_offset = index.to(tl.int64) * (MAX_ORDER - 1) + history
        history_token = tl.load(
            committed_history_ptr + history_offset,
            mask=request_live,
            other=0,
        ).to(tl.int64)
        invalid_history = request_live & (
            (history_token < 0) | (history_token >= VOCAB_SIZE)
        )
        tl.atomic_or(
            error_code_ptr,
            tl.where(invalid_history, _ERROR_TOKEN_ID, 0),
        )

    token_live = (index < MAX_TOKENS) & (index < num_tokens)
    token_id = tl.load(token_ids_ptr + index, mask=token_live, other=0).to(tl.int64)
    invalid_token = token_live & ((token_id < 0) | (token_id >= VOCAB_SIZE))
    tl.atomic_or(
        error_code_ptr,
        tl.where(invalid_token, _ERROR_TOKEN_ID, 0),
    )


@triton.jit
def _request_ids_kernel(
    query_start_loc_ptr,
    num_seqs_ptr,
    num_tokens_ptr,
    request_ids_ptr,
    error_code_ptr,
    MAX_TOKENS: tl.constexpr,
):
    token = tl.program_id(0)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    valid_metadata = tl.load(error_code_ptr).to(tl.int32) == 0
    live = (token < num_tokens) & valid_metadata

    low = tl.zeros((), tl.int32)
    high = tl.maximum(num_seqs, 1)
    while low + 1 < high:
        middle = (low + high) // 2
        start = tl.load(query_start_loc_ptr + middle, mask=live, other=0).to(tl.int32)
        low = tl.where(start <= token, middle, low)
        high = tl.where(start <= token, high, middle)
    request = tl.where(live & (num_seqs > 0), low, -1)
    tl.store(request_ids_ptr + token, request)


@triton.jit
def _source_token(
    token_ids_ptr,
    committed_history_ptr,
    query_start,
    request,
    query_relative,
    relative_position,
    eos_token_id,
    live,
    MAX_ORDER: tl.constexpr,
):
    source_relative = query_relative + relative_position
    from_query = source_relative >= 0
    query_index = query_start + source_relative
    history_index = (MAX_ORDER - 1) + source_relative
    query_value = tl.load(
        token_ids_ptr + query_index.to(tl.int64),
        mask=live & from_query,
        other=eos_token_id,
    ).to(tl.int64)
    history_offset = request.to(tl.int64) * (MAX_ORDER - 1) + history_index.to(tl.int64)
    history_value = tl.load(
        committed_history_ptr + history_offset,
        mask=live & ~from_query & (history_index >= 0),
        other=eos_token_id,
    ).to(tl.int64)
    return tl.where(from_query, query_value, history_value)


@triton.jit
def _hash_ids_kernel(
    token_ids_ptr,
    query_start_loc_ptr,
    committed_history_ptr,
    num_tokens_ptr,
    request_ids_ptr,
    multipliers_ptr,
    prime_sizes_ptr,
    table_offsets_ptr,
    out_ptr,
    error_code_ptr,
    eos_token_id,
    MAX_TOKENS: tl.constexpr,
    MAX_ORDER: tl.constexpr,
    HEADS_PER_ORDER: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    request = tl.load(request_ids_ptr + token).to(tl.int32)
    valid_metadata = tl.load(error_code_ptr).to(tl.int32) == 0
    live = (token < num_tokens) & (request >= 0) & valid_metadata
    query_start = tl.load(query_start_loc_ptr + request, mask=live, other=0).to(
        tl.int32
    )
    query_relative = token - query_start
    order = head // HEADS_PER_ORDER + 2
    mixed = tl.zeros((), tl.int64)

    for position in tl.static_range(0, MAX_ORDER):
        in_order = position < order
        distance_from_current = order - 1 - position
        relative_position = -distance_from_current
        value = _source_token(
            token_ids_ptr,
            committed_history_ptr,
            query_start,
            request,
            query_relative,
            relative_position,
            eos_token_id,
            live & in_order,
            MAX_ORDER=MAX_ORDER,
        )
        bounded = tl.full((), True, tl.int1)
        for back in tl.static_range(1, MAX_ORDER):
            boundary_value = _source_token(
                token_ids_ptr,
                committed_history_ptr,
                query_start,
                request,
                query_relative,
                -back,
                eos_token_id,
                live & in_order & (back < distance_from_current),
                MAX_ORDER=MAX_ORDER,
            )
            is_boundary = (back < distance_from_current) & (
                boundary_value == eos_token_id
            )
            bounded &= ~is_boundary
        effective = tl.where(bounded, value, eos_token_id).to(tl.int64)
        multiplier = tl.load(
            multipliers_ptr + position,
            mask=in_order,
            other=1,
        ).to(tl.int64)
        product = effective * multiplier
        mixed ^= tl.where(in_order, product, 0)

    prime = tl.load(prime_sizes_ptr + head).to(tl.int64)
    table_offset = tl.load(table_offsets_ptr + head).to(tl.int64)
    remainder = mixed % prime
    remainder = tl.where(remainder < 0, remainder + prime, remainder)
    embedding_id = table_offset + remainder
    output_offset = token.to(tl.int64) * HEAD_COUNT + head
    tl.store(out_ptr + output_offset, tl.where(live, embedding_id, -1))


def _launch_hash_pipeline(
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    out: torch.Tensor,
    request_ids: torch.Tensor,
    error_code: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
    token_count: int | None = None,
) -> None:
    """Launch request mapping and hashing within preplanned storage."""
    if token_count is None:
        token_count = max_tokens
    head_count = (max_order - 1) * heads_per_order
    _reset_error_kernel[(1,)](error_code, num_warps=1)
    _validate_metadata_kernel[(max(token_count, max_seqs),)](
        token_ids,
        query_start_loc,
        committed_history,
        num_seqs,
        num_tokens,
        error_code,
        token_count,
        VOCAB_SIZE=vocab_size,
        MAX_ORDER=max_order,
        MAX_SEQS=max_seqs,
        MAX_TOKENS=max_tokens,
        num_warps=1,
    )
    _request_ids_kernel[(token_count,)](
        query_start_loc,
        num_seqs,
        num_tokens,
        request_ids,
        error_code,
        MAX_TOKENS=max_tokens,
        num_warps=1,
    )
    _hash_ids_kernel[(token_count, head_count)](
        token_ids,
        query_start_loc,
        committed_history,
        num_tokens,
        request_ids,
        multipliers,
        prime_sizes,
        table_offsets,
        out,
        error_code,
        eos_token_id,
        MAX_TOKENS=max_tokens,
        MAX_ORDER=max_order,
        HEADS_PER_ORDER=heads_per_order,
        HEAD_COUNT=head_count,
        num_warps=1,
    )


@torch.library.custom_op(
    "b12x::ple_hash_pipeline",
    mutates_args=("out", "request_ids", "error_code"),
)
def _hash_pipeline_op(
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    out: torch.Tensor,
    request_ids: torch.Tensor,
    error_code: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
) -> None:
    _launch_hash_pipeline(
        token_ids,
        query_start_loc,
        committed_history,
        num_seqs,
        num_tokens,
        multipliers,
        prime_sizes,
        table_offsets,
        out,
        request_ids,
        error_code,
        eos_token_id,
        vocab_size,
        max_order,
        heads_per_order,
        max_seqs,
        max_tokens,
    )


@_hash_pipeline_op.register_fake
def _hash_pipeline_fake(
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    out: torch.Tensor,
    request_ids: torch.Tensor,
    error_code: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
) -> None:
    del token_ids, query_start_loc, committed_history, num_seqs, num_tokens
    del multipliers, prime_sizes, table_offsets, out, request_ids, error_code
    del eos_token_id, vocab_size, max_order, heads_per_order, max_seqs, max_tokens


def run_hash_kernel(binding: Binding) -> None:
    """Dispatch the opaque, mutation-declared hash pipeline."""
    caps = binding.plan.caps
    torch.ops.b12x.ple_hash_pipeline(
        binding.token_ids,
        binding.query_start_loc,
        binding.committed_history,
        binding.num_seqs,
        binding.num_tokens,
        binding.plan.multipliers,
        binding.plan.prime_sizes,
        binding.plan.table_offsets,
        binding.out,
        binding.request_ids,
        binding.error_code,
        caps.eos_token_id,
        caps.vocab_size,
        caps.max_order,
        caps.heads_per_order,
        caps.max_seqs,
        caps.max_tokens,
    )


__all__ = ["run_hash_kernel"]
