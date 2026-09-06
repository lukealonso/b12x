"""Triton local-shard gather and dequantization for PLE embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from b12x.sequence.ple_hash._kernels import _launch_hash_pipeline

if TYPE_CHECKING:
    from ._contracts import Binding


_BLOCK_D = 128


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


def _pipeline_scratch_views(
    scratch: torch.Tensor,
    *,
    max_tokens: int,
    head_count: int,
    ids_offset_bytes: int,
    request_ids_offset_bytes: int,
    error_code_offset_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = _scratch_view(
        scratch,
        offset_bytes=ids_offset_bytes,
        shape=(max_tokens, head_count),
        dtype=torch.int64,
    )
    request_ids = _scratch_view(
        scratch,
        offset_bytes=request_ids_offset_bytes,
        shape=(max_tokens,),
        dtype=torch.int32,
    )
    error_code = _scratch_view(
        scratch,
        offset_bytes=error_code_offset_bytes,
        shape=(1,),
        dtype=torch.int32,
    )
    return ids, request_ids, error_code


@triton.jit
def _bf16_lookup_kernel(
    weight_ptr,
    ids_ptr,
    num_tokens_ptr,
    out_ptr,
    MAX_TOKENS: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    TABLE_VOCAB_SIZE: tl.constexpr,
    SHARD_START: tl.constexpr,
    SHARD_END: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    columns = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    column_mask = columns < HEAD_DIM
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    token_live = (token < num_tokens) & (num_tokens >= 0) & (num_tokens <= MAX_TOKENS)
    id_offset = token.to(tl.int64) * HEAD_COUNT + head.to(tl.int64)
    embedding_id = tl.load(ids_ptr + id_offset, mask=token_live, other=-1).to(tl.int64)
    local = (
        token_live
        & (embedding_id >= SHARD_START)
        & (embedding_id < SHARD_END)
        & (embedding_id < TABLE_VOCAB_SIZE)
    )
    local_row = tl.where(
        local,
        embedding_id - tl.full((), SHARD_START, tl.int64),
        0,
    ).to(tl.int64)
    row_base = local_row * tl.full((), HEAD_DIM, tl.int64)
    value = tl.load(
        weight_ptr + row_base + columns.to(tl.int64),
        mask=local & column_mask,
        other=0.0,
    ).to(tl.bfloat16)
    out_offset = (
        token.to(tl.int64) * EMBEDDING_DIM
        + head.to(tl.int64) * HEAD_DIM
        + columns.to(tl.int64)
    )
    tl.store(out_ptr + out_offset, tl.where(local, value, 0.0), mask=column_mask)


@triton.jit
def _fp8_lookup_kernel(
    weight_ptr,
    weight_scale_ptr,
    ids_ptr,
    num_tokens_ptr,
    out_ptr,
    MAX_TOKENS: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    TABLE_VOCAB_SIZE: tl.constexpr,
    SHARD_START: tl.constexpr,
    SHARD_END: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    column_block = tl.program_id(2)
    columns = column_block * BLOCK_D + tl.arange(0, BLOCK_D)
    column_mask = columns < HEAD_DIM
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    valid_count = (num_tokens >= 0) & (num_tokens <= MAX_TOKENS)
    token_live = (token < num_tokens) & valid_count
    id_offset = token.to(tl.int64) * HEAD_COUNT + head.to(tl.int64)
    embedding_id = tl.load(ids_ptr + id_offset, mask=token_live, other=-1).to(tl.int64)
    local = (
        token_live
        & (embedding_id >= SHARD_START)
        & (embedding_id < SHARD_END)
        & (embedding_id < TABLE_VOCAB_SIZE)
    )
    local_row = tl.where(
        local,
        embedding_id - tl.full((), SHARD_START, tl.int64),
        0,
    ).to(tl.int64)
    row_base = local_row * tl.full((), HEAD_DIM, tl.int64)
    quantized = tl.load(
        weight_ptr + row_base + columns.to(tl.int64),
        mask=local & column_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(weight_scale_ptr).to(tl.float32)
    dequantized = (quantized * scale).to(tl.bfloat16)
    out_offset = (
        token.to(tl.int64) * EMBEDDING_DIM
        + head.to(tl.int64) * HEAD_DIM
        + columns.to(tl.int64)
    )
    tl.store(
        out_ptr + out_offset,
        tl.where(local, dequantized, 0.0),
        mask=column_mask,
    )


@triton.jit
def _nvfp4_lookup_kernel(
    weight_ptr,
    weight_scale_ptr,
    weight_scale_2_ptr,
    ids_ptr,
    num_tokens_ptr,
    out_ptr,
    MAX_TOKENS: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    TABLE_VOCAB_SIZE: tl.constexpr,
    SHARD_START: tl.constexpr,
    SHARD_END: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    columns = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    column_mask = columns < HEAD_DIM
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    token_live = (token < num_tokens) & (num_tokens >= 0) & (num_tokens <= MAX_TOKENS)
    id_offset = token.to(tl.int64) * HEAD_COUNT + head.to(tl.int64)
    embedding_id = tl.load(ids_ptr + id_offset, mask=token_live, other=-1).to(tl.int64)
    local = (
        token_live
        & (embedding_id >= SHARD_START)
        & (embedding_id < SHARD_END)
        & (embedding_id < TABLE_VOCAB_SIZE)
    )
    local_row = tl.where(
        local,
        embedding_id - tl.full((), SHARD_START, tl.int64),
        0,
    ).to(tl.int64)

    packed_row_base = local_row * tl.full((), HEAD_DIM // 2, tl.int64)
    packed = tl.load(
        weight_ptr + packed_row_base + (columns // 2).to(tl.int64),
        mask=local & column_mask,
        other=0,
    ).to(tl.uint8)
    nibble = tl.where(
        (columns & 1) == 0,
        packed & 0x0F,
        (packed >> 4) & 0x0F,
    ).to(tl.int32)
    magnitude_code = nibble & 0x07
    magnitude = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    ).to(tl.float32)
    quantized = tl.where((nibble & 0x08) != 0, -magnitude, magnitude)

    scale_cols = HEAD_DIM // 16
    scale_row_base = local_row * tl.full((), scale_cols, tl.int64)
    block_scale = tl.load(
        weight_scale_ptr + scale_row_base + (columns // 16).to(tl.int64),
        mask=local & column_mask,
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(weight_scale_2_ptr).to(tl.float32)
    dequantized = (quantized * block_scale * global_scale).to(tl.bfloat16)
    out_offset = (
        token.to(tl.int64) * EMBEDDING_DIM
        + head.to(tl.int64) * HEAD_DIM
        + columns.to(tl.int64)
    )
    tl.store(
        out_ptr + out_offset,
        tl.where(local, dequantized, 0.0),
        mask=column_mask,
    )


def _launch_bf16_lookup(
    weight: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    grid = (out.shape[0], head_count, triton.cdiv(head_dim, _BLOCK_D))
    _bf16_lookup_kernel[grid](
        weight,
        ids,
        num_tokens,
        out,
        MAX_TOKENS=max_tokens,
        HEAD_COUNT=head_count,
        HEAD_DIM=head_dim,
        EMBEDDING_DIM=embedding_dim,
        TABLE_VOCAB_SIZE=table_vocab_size,
        SHARD_START=shard_start,
        SHARD_END=shard_end,
        BLOCK_D=_BLOCK_D,
        num_warps=4,
    )


def _launch_fp8_lookup(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    grid = (out.shape[0], head_count, triton.cdiv(head_dim, _BLOCK_D))
    _fp8_lookup_kernel[grid](
        weight,
        weight_scale,
        ids,
        num_tokens,
        out,
        MAX_TOKENS=max_tokens,
        HEAD_COUNT=head_count,
        HEAD_DIM=head_dim,
        EMBEDDING_DIM=embedding_dim,
        TABLE_VOCAB_SIZE=table_vocab_size,
        SHARD_START=shard_start,
        SHARD_END=shard_end,
        BLOCK_D=_BLOCK_D,
        num_warps=4,
    )


def _launch_nvfp4_lookup(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    grid = (out.shape[0], head_count, triton.cdiv(head_dim, _BLOCK_D))
    _nvfp4_lookup_kernel[grid](
        weight,
        weight_scale,
        weight_scale_2,
        ids,
        num_tokens,
        out,
        MAX_TOKENS=max_tokens,
        HEAD_COUNT=head_count,
        HEAD_DIM=head_dim,
        EMBEDDING_DIM=embedding_dim,
        TABLE_VOCAB_SIZE=table_vocab_size,
        SHARD_START=shard_start,
        SHARD_END=shard_end,
        BLOCK_D=_BLOCK_D,
        num_warps=4,
    )


@torch.library.custom_op(
    "b12x::ple_embedding_bf16_lookup",
    mutates_args=("out",),
)
def _bf16_lookup_op(
    weight: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    _launch_bf16_lookup(
        weight,
        ids,
        num_tokens,
        out,
        max_tokens,
        head_count,
        head_dim,
        embedding_dim,
        table_vocab_size,
        shard_start,
        shard_end,
    )


@_bf16_lookup_op.register_fake
def _bf16_lookup_fake(
    weight: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    del weight, ids, num_tokens, out
    del max_tokens, head_count, head_dim, embedding_dim
    del table_vocab_size, shard_start, shard_end


@torch.library.custom_op(
    "b12x::ple_embedding_fp8_lookup",
    mutates_args=("out",),
)
def _fp8_lookup_op(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    _launch_fp8_lookup(
        weight,
        weight_scale,
        ids,
        num_tokens,
        out,
        max_tokens,
        head_count,
        head_dim,
        embedding_dim,
        table_vocab_size,
        shard_start,
        shard_end,
    )


@_fp8_lookup_op.register_fake
def _fp8_lookup_fake(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    del weight, weight_scale, ids, num_tokens, out
    del max_tokens, head_count, head_dim, embedding_dim
    del table_vocab_size, shard_start, shard_end


@torch.library.custom_op(
    "b12x::ple_embedding_nvfp4_lookup",
    mutates_args=("out",),
)
def _nvfp4_lookup_op(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    _launch_nvfp4_lookup(
        weight,
        weight_scale,
        weight_scale_2,
        ids,
        num_tokens,
        out,
        max_tokens,
        head_count,
        head_dim,
        embedding_dim,
        table_vocab_size,
        shard_start,
        shard_end,
    )


@_nvfp4_lookup_op.register_fake
def _nvfp4_lookup_fake(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> None:
    del weight, weight_scale, weight_scale_2, ids, num_tokens, out
    del max_tokens, head_count, head_dim, embedding_dim
    del table_vocab_size, shard_start, shard_end


def _launch_hash(
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    ids: torch.Tensor,
    request_ids: torch.Tensor,
    error_code: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
    token_count: int,
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
        ids,
        request_ids,
        error_code,
        eos_token_id,
        vocab_size,
        max_order,
        heads_per_order,
        max_seqs,
        max_tokens,
        token_count=token_count,
    )


@torch.library.custom_op(
    "b12x::ple_embedding_bf16_pipeline",
    mutates_args=("scratch", "out"),
)
def _bf16_pipeline_op(
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    scratch: torch.Tensor,
    out: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
    ids_offset_bytes: int,
    request_ids_offset_bytes: int,
    error_code_offset_bytes: int,
) -> None:
    ids, request_ids, error_code = _pipeline_scratch_views(
        scratch,
        max_tokens=max_tokens,
        head_count=head_count,
        ids_offset_bytes=ids_offset_bytes,
        request_ids_offset_bytes=request_ids_offset_bytes,
        error_code_offset_bytes=error_code_offset_bytes,
    )
    _launch_hash(
        token_ids,
        query_start_loc,
        committed_history,
        num_seqs,
        num_tokens,
        multipliers,
        prime_sizes,
        table_offsets,
        ids,
        request_ids,
        error_code,
        eos_token_id,
        vocab_size,
        max_order,
        heads_per_order,
        max_seqs,
        max_tokens,
        out.shape[0],
    )
    _launch_bf16_lookup(
        weight,
        ids,
        num_tokens,
        out,
        max_tokens,
        head_count,
        head_dim,
        embedding_dim,
        table_vocab_size,
        shard_start,
        shard_end,
    )


@_bf16_pipeline_op.register_fake
def _bf16_pipeline_fake(
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    scratch: torch.Tensor,
    out: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
    ids_offset_bytes: int,
    request_ids_offset_bytes: int,
    error_code_offset_bytes: int,
) -> None:
    del weight, token_ids, query_start_loc, committed_history
    del num_seqs, num_tokens, multipliers, prime_sizes, table_offsets
    del scratch, out, eos_token_id, vocab_size, max_order, heads_per_order
    del max_seqs, max_tokens, head_count, head_dim, embedding_dim
    del table_vocab_size, shard_start, shard_end
    del ids_offset_bytes, request_ids_offset_bytes, error_code_offset_bytes


@torch.library.custom_op(
    "b12x::ple_embedding_fp8_pipeline",
    mutates_args=("scratch", "out"),
)
def _fp8_pipeline_op(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    scratch: torch.Tensor,
    out: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
    ids_offset_bytes: int,
    request_ids_offset_bytes: int,
    error_code_offset_bytes: int,
) -> None:
    ids, request_ids, error_code = _pipeline_scratch_views(
        scratch,
        max_tokens=max_tokens,
        head_count=head_count,
        ids_offset_bytes=ids_offset_bytes,
        request_ids_offset_bytes=request_ids_offset_bytes,
        error_code_offset_bytes=error_code_offset_bytes,
    )
    _launch_hash(
        token_ids,
        query_start_loc,
        committed_history,
        num_seqs,
        num_tokens,
        multipliers,
        prime_sizes,
        table_offsets,
        ids,
        request_ids,
        error_code,
        eos_token_id,
        vocab_size,
        max_order,
        heads_per_order,
        max_seqs,
        max_tokens,
        out.shape[0],
    )
    _launch_fp8_lookup(
        weight,
        weight_scale,
        ids,
        num_tokens,
        out,
        max_tokens,
        head_count,
        head_dim,
        embedding_dim,
        table_vocab_size,
        shard_start,
        shard_end,
    )


@_fp8_pipeline_op.register_fake
def _fp8_pipeline_fake(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    scratch: torch.Tensor,
    out: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
    ids_offset_bytes: int,
    request_ids_offset_bytes: int,
    error_code_offset_bytes: int,
) -> None:
    del weight, weight_scale, token_ids, query_start_loc, committed_history
    del num_seqs, num_tokens, multipliers, prime_sizes, table_offsets
    del scratch, out, eos_token_id, vocab_size, max_order, heads_per_order
    del max_seqs, max_tokens, head_count, head_dim, embedding_dim
    del table_vocab_size, shard_start, shard_end
    del ids_offset_bytes, request_ids_offset_bytes, error_code_offset_bytes


@torch.library.custom_op(
    "b12x::ple_embedding_nvfp4_pipeline",
    mutates_args=("scratch", "out"),
)
def _nvfp4_pipeline_op(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    scratch: torch.Tensor,
    out: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
    ids_offset_bytes: int,
    request_ids_offset_bytes: int,
    error_code_offset_bytes: int,
) -> None:
    ids, request_ids, error_code = _pipeline_scratch_views(
        scratch,
        max_tokens=max_tokens,
        head_count=head_count,
        ids_offset_bytes=ids_offset_bytes,
        request_ids_offset_bytes=request_ids_offset_bytes,
        error_code_offset_bytes=error_code_offset_bytes,
    )
    _launch_hash(
        token_ids,
        query_start_loc,
        committed_history,
        num_seqs,
        num_tokens,
        multipliers,
        prime_sizes,
        table_offsets,
        ids,
        request_ids,
        error_code,
        eos_token_id,
        vocab_size,
        max_order,
        heads_per_order,
        max_seqs,
        max_tokens,
        out.shape[0],
    )
    _launch_nvfp4_lookup(
        weight,
        weight_scale,
        weight_scale_2,
        ids,
        num_tokens,
        out,
        max_tokens,
        head_count,
        head_dim,
        embedding_dim,
        table_vocab_size,
        shard_start,
        shard_end,
    )


@_nvfp4_pipeline_op.register_fake
def _nvfp4_pipeline_fake(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    scratch: torch.Tensor,
    out: torch.Tensor,
    eos_token_id: int,
    vocab_size: int,
    max_order: int,
    heads_per_order: int,
    max_seqs: int,
    max_tokens: int,
    head_count: int,
    head_dim: int,
    embedding_dim: int,
    table_vocab_size: int,
    shard_start: int,
    shard_end: int,
    ids_offset_bytes: int,
    request_ids_offset_bytes: int,
    error_code_offset_bytes: int,
) -> None:
    del weight, weight_scale, weight_scale_2
    del token_ids, query_start_loc, committed_history
    del num_seqs, num_tokens, multipliers, prime_sizes, table_offsets
    del scratch, out, eos_token_id, vocab_size, max_order, heads_per_order
    del max_seqs, max_tokens, head_count, head_dim, embedding_dim
    del table_vocab_size, shard_start, shard_end
    del ids_offset_bytes, request_ids_offset_bytes, error_code_offset_bytes


def run_pipeline(binding: Binding, *, token_count: int) -> None:
    """Launch one opaque hash, local gather, and inline dequantization op."""
    plan = binding.plan
    caps = plan.caps
    hash_args = (
        binding.token_ids,
        binding.query_start_loc,
        binding.committed_history,
        binding.num_seqs,
        binding.num_tokens,
        plan.multipliers,
        plan.prime_sizes,
        plan.table_offsets,
        binding.scratch,
        binding.out[:token_count],
        caps.eos_token_id,
        caps.vocab_size,
        caps.max_order,
        caps.heads_per_order,
        caps.max_seqs,
        caps.max_tokens,
        plan.head_count,
        plan.head_dim,
        caps.embedding_dim,
        plan.table_vocab_size,
        plan.shard_start,
        plan.shard_end,
        plan._layout.ids_offset_bytes,
        plan._layout.hash_scratch_offset_bytes
        + plan._hash_plan.layout.request_ids_offset_bytes,
        plan._layout.hash_scratch_offset_bytes
        + plan._hash_plan.layout.error_code_offset_bytes,
    )
    if caps.quant_mode == "bf16":
        torch.ops.b12x.ple_embedding_bf16_pipeline(binding.weight, *hash_args)
    elif caps.quant_mode == "fp8_e4m3_per_tensor":
        assert binding.weight_scale is not None
        torch.ops.b12x.ple_embedding_fp8_pipeline(
            binding.weight, binding.weight_scale, *hash_args
        )
    elif caps.quant_mode == "nvfp4_group16":
        assert binding.weight_scale is not None
        assert binding.weight_scale_2 is not None
        torch.ops.b12x.ple_embedding_nvfp4_pipeline(
            binding.weight,
            binding.weight_scale,
            binding.weight_scale_2,
            *hash_args,
        )
    else:
        raise AssertionError(f"unplanned PLE storage mode {caps.quant_mode!r}")


__all__ = ["run_pipeline"]
