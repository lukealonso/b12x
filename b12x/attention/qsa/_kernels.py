"""Auxiliary Triton stages for the QSA selection and state transaction."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _round_fp32_to_bf16(value):
    """Round FP32 to BF16 and widen without permitting arithmetic fusion."""
    bits = value.to(tl.uint32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & 0xFFFF0000
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["rows"])
def _validate_packed_boundaries_kernel(
    query_start_loc,
    request_errors,
    rows,
    MAX_BATCH: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    requests = tl.arange(0, BLOCK_BATCH)
    request_mask = requests < MAX_BATCH
    starts = tl.load(
        query_start_loc + requests,
        mask=request_mask,
        other=rows,
    ).to(tl.int32)
    ends = tl.load(
        query_start_loc + requests + 1,
        mask=request_mask,
        other=rows,
    ).to(tl.int32)
    deltas = ends - starts
    active = request_mask & (deltas > 0)
    active_count = tl.sum(active.to(tl.int32), axis=0)
    dense_prefix = active == (request_mask & (requests < active_count))
    initial = tl.load(query_start_loc).to(tl.int32)
    terminal = tl.load(query_start_loc + MAX_BATCH).to(tl.int32)
    invalid = (
        (initial != 0)
        | (terminal < 0)
        | (terminal > rows)
        | (tl.sum((request_mask & (deltas < 0)).to(tl.int32), axis=0) != 0)
        | (tl.sum((request_mask & ~dense_prefix).to(tl.int32), axis=0) != 0)
    )
    tl.store(request_errors + MAX_BATCH, tl.where(invalid, 1, 0))


@triton.jit
def _validate_packed_requests_kernel(
    sequence_lengths,
    query_start_loc,
    num_accepted_tokens,
    is_prefilling,
    raw_state_slot_ids,
    raw_interval_start_positions,
    request_errors,
    raw_state_slot_stride,
    raw_interval_start_stride,
    MAX_BATCH: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    MAX_SPECULATIVE: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    request = tl.program_id(0)
    global_error = tl.load(request_errors + MAX_BATCH).to(tl.int32)
    start = tl.load(query_start_loc + request).to(tl.int32)
    end = tl.load(query_start_loc + request + 1).to(tl.int32)
    query_length = end - start
    active = (global_error == 0) & (query_length > 0)
    error = global_error

    accepted = tl.load(num_accepted_tokens + request, mask=active, other=0).to(tl.int32)
    prefill = tl.load(is_prefilling + request, mask=active, other=0).to(tl.int1)
    error = error | tl.where(
        active
        & (
            (accepted < 1)
            | (accepted > MAX_SPECULATIVE + 1)
            | (prefill & (accepted != 1))
        ),
        4,
        0,
    )
    sequence_length = tl.load(sequence_lengths + request, mask=active, other=0).to(
        tl.int64
    )
    first_position = sequence_length - query_length
    error = error | tl.where(
        active
        & (
            (sequence_length <= 0)
            | (sequence_length > MAX_SEQ_LEN)
            | (first_position < 0)
        ),
        8,
        0,
    )

    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=active,
        other=-1,
    ).to(tl.int64)
    valid_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    prior_interval_start = tl.load(
        raw_interval_start_positions + state_slot * raw_interval_start_stride,
        mask=active & valid_slot,
        other=-2,
    ).to(tl.int64)
    expected_first = prior_interval_start + accepted
    error = error | tl.where(
        active
        & (
            ~valid_slot
            | (prior_interval_start < -1)
            | ((prior_interval_start == -1) & ((accepted != 1) | (first_position != 0)))
            | (first_position != expected_first)
        ),
        16,
        0,
    )

    owners = tl.arange(0, BLOCK_BATCH)
    owner_mask = owners < MAX_BATCH
    owner_slots = tl.load(
        raw_state_slot_ids + owners * raw_state_slot_stride,
        mask=owner_mask,
        other=-1,
    ).to(tl.int64)
    invalid_slot_map = (
        tl.sum(
            owner_mask & ((owner_slots < -1) | (owner_slots >= MAX_RAW_STATE_SLOTS)),
            axis=0,
        )
        != 0
    )
    duplicate_slot = (
        active
        & valid_slot
        & (
            tl.sum(
                owner_mask
                & (owners != request)
                & (owner_slots == state_slot)
                & (owner_slots >= 0),
                axis=0,
            )
            != 0
        )
    )
    error = error | tl.where(active & (duplicate_slot | invalid_slot_map), 32, 0)

    tl.store(request_errors + request, error)


@triton.jit(do_not_specialize=["rows", "rope_position_rows"])
def _materialize_packed_row_errors_kernel(
    request_ids,
    query_positions,
    rope_positions,
    sequence_lengths,
    query_start_loc,
    request_errors,
    state_errors,
    rope_position_row_stride,
    rope_position_axis_stride,
    rows,
    rope_position_rows,
    MAX_BATCH: tl.constexpr,
    POSITION_AXES: tl.constexpr,
):
    row = tl.program_id(0)
    terminal = tl.load(query_start_loc + MAX_BATCH).to(tl.int32)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    live = row < terminal
    valid_request = live & (request >= 0) & (request < MAX_BATCH)
    start = tl.load(query_start_loc + request, mask=valid_request, other=0).to(tl.int32)
    end = tl.load(query_start_loc + request + 1, mask=valid_request, other=0).to(
        tl.int32
    )
    sequence_length = tl.load(
        sequence_lengths + request, mask=valid_request, other=0
    ).to(tl.int64)
    expected_position = sequence_length - (end - row)
    error = tl.load(request_errors + MAX_BATCH).to(tl.int32)
    error = error | tl.where(live & ~valid_request, 32, 0)
    error = error | tl.where(valid_request & ((row < start) | (row >= end)), 32, 0)
    error = error | tl.where(valid_request & (position != expected_position), 32, 0)
    error = error | tl.load(request_errors + request, mask=valid_request, other=0).to(
        tl.int32
    )

    axes = tl.arange(0, 4)
    axis_mask = axes < POSITION_AXES
    padding_rope = tl.load(
        rope_positions
        + row * rope_position_row_stride
        + axes * rope_position_axis_stride,
        mask=(~live) & axis_mask,
        other=-1,
    ).to(tl.int64)
    invalid_padding_rope = tl.sum(
        ((~live) & axis_mask & (padding_rope != -1)).to(tl.int32), axis=0
    )
    live_rope = tl.load(
        rope_positions
        + row * rope_position_row_stride
        + axes * rope_position_axis_stride,
        mask=live & axis_mask,
        other=-1,
    ).to(tl.int64)
    invalid_live_rope = tl.sum(
        (
            live
            & axis_mask
            & ((live_rope < 0) | (live_rope >= rope_position_rows))
        ).to(tl.int32),
        axis=0,
    )
    error = error | tl.where(invalid_live_rope != 0, 64, 0)
    error = error | tl.where((~live) & ((request != -1) | (position != -1)), 128, 0)
    error = error | tl.where(invalid_padding_rope != 0, 128, 0)
    tl.store(state_errors + row, error)


@triton.jit
def _validate_page_tables_kernel(
    request_ids,
    sequence_lengths,
    main_block_table,
    compressed_block_table,
    raw_state_slot_ids,
    state_errors,
    main_table_stride,
    compressed_table_stride,
    raw_state_slot_stride,
    num_main_pages,
    num_compressed_pages,
    MAIN_TABLE_WIDTH: tl.constexpr,
    COMPRESSED_TABLE_WIDTH: tl.constexpr,
    MAIN_PAGE_SIZE: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    MAX_BATCH: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    SHARED_COMPRESSED_RAW_POOL: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    page_block = tl.program_id(1)
    request = tl.load(request_ids + row).to(tl.int64)
    real_request = (tl.load(state_errors + row) == 0) & (request >= 0)
    sequence_length = tl.load(
        sequence_lengths + request,
        mask=real_request,
        other=0,
    ).to(tl.int64)
    pages = page_block * BLOCK_P + tl.arange(0, BLOCK_P)

    main_pages = tl.cdiv(sequence_length, MAIN_PAGE_SIZE)
    main_active = real_request & (pages < main_pages)
    main_offsets = (request * main_table_stride + pages.to(tl.int64)).to(tl.int64)
    main_physical = tl.load(
        main_block_table + main_offsets,
        mask=main_active,
        other=-1,
    ).to(tl.int64)
    bad_main = tl.sum(
        main_active & ((main_physical < 0) | (main_physical >= num_main_pages)),
        axis=0,
    )
    if bad_main != 0:
        tl.atomic_or(state_errors + row, 2048)

    completed_groups = sequence_length // COMPRESS_RATIO
    compressed_pages = tl.cdiv(completed_groups, COMPRESSED_PAGE_SIZE)
    compressed_active = real_request & (pages < compressed_pages)
    compressed_offsets = (request * compressed_table_stride + pages.to(tl.int64)).to(
        tl.int64
    )
    compressed_physical = tl.load(
        compressed_block_table + compressed_offsets,
        mask=compressed_active,
        other=-1,
    ).to(tl.int64)
    compressed_valid = (compressed_physical >= 0) & (
        compressed_physical < num_compressed_pages
    )
    bad_compressed = tl.sum(
        compressed_active & ~compressed_valid,
        axis=0,
    )
    if bad_compressed != 0:
        tl.atomic_or(state_errors + row, 1024)
    if SHARED_COMPRESSED_RAW_POOL:
        owners = tl.arange(0, BLOCK_BATCH)
        owner_mask = owners < MAX_BATCH
        owner_slots = tl.load(
            raw_state_slot_ids + owners * raw_state_slot_stride,
            mask=owner_mask,
            other=-1,
        ).to(tl.int64)
        owned = owner_mask & (owner_slots >= 0) & (owner_slots < MAX_RAW_STATE_SLOTS)
        aliases = (
            compressed_active[:, None]
            & compressed_valid[:, None]
            & owned[None, :]
            & (compressed_physical[:, None] == owner_slots[None, :])
        )
        if tl.sum(tl.sum(aliases.to(tl.int32), axis=1), axis=0) != 0:
            tl.atomic_or(state_errors + row, 4096)


@triton.jit
def _clear_shared_page_occupancy_kernel(
    occupancy,
    element_count,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    tl.store(occupancy + offsets, 0, mask=offsets < element_count)


@triton.jit
def _mark_live_compressed_pages_kernel(
    sequence_lengths,
    compressed_block_table,
    occupancy,
    table_error_offset,
    compressed_table_stride,
    num_compressed_pages,
    MAX_SEQ_LEN: tl.constexpr,
    COMPRESSED_TABLE_WIDTH: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    owner = tl.program_id(0)
    page_block = tl.program_id(1)
    sequence_length = tl.load(sequence_lengths + owner).to(tl.int64)
    valid_length = (sequence_length >= 0) & (sequence_length <= MAX_SEQ_LEN)
    if (page_block == 0) & ~valid_length:
        tl.atomic_or(occupancy + table_error_offset, 1)

    pages = page_block * BLOCK_P + tl.arange(0, BLOCK_P)
    completed_groups = tl.where(valid_length, sequence_length // COMPRESS_RATIO, 0)
    live_pages = tl.cdiv(completed_groups, COMPRESSED_PAGE_SIZE)
    active = valid_length & (pages < live_pages) & (pages < COMPRESSED_TABLE_WIDTH)
    offsets = owner.to(tl.int64) * compressed_table_stride + pages.to(tl.int64)
    physical = tl.load(
        compressed_block_table + offsets,
        mask=active,
        other=-1,
    ).to(tl.int64)
    valid_page = active & (physical >= 0) & (physical < num_compressed_pages)
    if tl.sum((active & ~valid_page).to(tl.int32), axis=0) != 0:
        tl.atomic_or(occupancy + table_error_offset, 1)
    tl.atomic_xchg(occupancy + physical, 1, mask=valid_page)


@triton.jit
def _validate_active_raw_slots_kernel(
    request_ids,
    raw_state_slot_ids,
    state_errors,
    occupancy,
    raw_state_slot_stride,
    table_error_offset,
):
    row = tl.program_id(0)
    request = tl.load(request_ids + row).to(tl.int64)
    active = (tl.load(state_errors + row) == 0) & (request >= 0)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=active,
        other=0,
    ).to(tl.int64)
    collision = tl.load(occupancy + state_slot, mask=active, other=0) != 0
    invalid_table = tl.load(occupancy + table_error_offset) != 0
    if active & (collision | invalid_table):
        tl.atomic_or(state_errors + row, 4096)


@triton.jit
def _prepare_index_query_kernel(
    index_query,
    request_ids,
    norm_weight,
    rope_positions,
    rope_cos,
    rope_sin,
    state_errors,
    prepared_query,
    rope_position_rows,
    eps,
    rope_position_row_stride,
    rope_position_axis_stride,
    rope_cos_row_stride,
    rope_sin_row_stride,
    INDEX_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    MROPE_SECTION_0: tl.constexpr,
    MROPE_SECTION_1: tl.constexpr,
    ROPE_IS_BF16: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // INDEX_HEADS
    head = program % INDEX_HEADS
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    valid_row = (tl.load(state_errors + row) == 0) & (
        tl.load(request_ids + row).to(tl.int64) >= 0
    )
    offsets = (row * INDEX_HEADS + head) * HEAD_DIM + dims
    values = tl.load(
        index_query + offsets,
        mask=valid_row & dim_mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(norm_weight + dims, mask=dim_mask, other=0.0).to(tl.float32)
    normalized = (values * inv_rms * (1.0 + weight)).to(tl.bfloat16).to(tl.float32)

    half_rotary = ROTARY_DIM // 2
    in_rotary = dims < ROTARY_DIM
    pair = dims % half_rotary
    partner_dim = tl.where(dims < half_rotary, dims + half_rotary, dims - half_rotary)
    partner_offsets = (row * INDEX_HEADS + head) * HEAD_DIM + partner_dim
    partner = tl.load(
        index_query + partner_offsets,
        mask=valid_row & in_rotary,
        other=0.0,
    ).to(tl.float32)
    partner_weight = tl.load(
        norm_weight + partner_dim,
        mask=in_rotary,
        other=0.0,
    ).to(tl.float32)
    partner = (
        (partner * inv_rms * (1.0 + partner_weight)).to(tl.bfloat16).to(tl.float32)
    )

    if POSITION_AXES == 1:
        axis = tl.zeros((BLOCK_D,), tl.int32)
    elif MROPE_INTERLEAVED:
        is_height = (pair % 3 == 1) & (pair < 3 * MROPE_SECTION_1)
        is_width = (pair % 3 == 2) & (
            pair < 3 * (ROTARY_DIM // 2 - MROPE_SECTION_0 - MROPE_SECTION_1)
        )
        axis = tl.where(is_height, 1, tl.where(is_width, 2, 0))
    else:
        axis = tl.where(
            pair < MROPE_SECTION_0,
            0,
            tl.where(pair < MROPE_SECTION_0 + MROPE_SECTION_1, 1, 2),
        )
    position = tl.load(
        rope_positions
        + row * rope_position_row_stride
        + axis * rope_position_axis_stride,
        mask=in_rotary,
        other=0,
    ).to(tl.int64)
    valid_position = (position >= 0) & (position < rope_position_rows)
    cosine = tl.load(
        rope_cos + position * rope_cos_row_stride + pair,
        mask=valid_row & in_rotary & valid_position,
        other=1.0,
    ).to(tl.float32)
    sine = tl.load(
        rope_sin + position * rope_sin_row_stride + pair,
        mask=valid_row & in_rotary & valid_position,
        other=0.0,
    ).to(tl.float32)
    rotated_partner = tl.where(dims < half_rotary, -partner, partner)
    if ROPE_IS_BF16:
        direct_product = _round_fp32_to_bf16(normalized * cosine)
        partner_product = _round_fp32_to_bf16(rotated_partner * sine)
        rotated = _round_fp32_to_bf16(direct_product + partner_product)
    else:
        rotated = normalized * cosine + rotated_partner * sine
    result = tl.where(in_rotary, rotated, normalized)
    tl.store(prepared_query + offsets, result, mask=dim_mask)
    invalid_position_count = tl.sum(valid_row & in_rotary & ~valid_position, axis=0)
    if invalid_position_count != 0:
        tl.atomic_or(state_errors + row, 16)


@triton.jit
def _validate_completed_groups_kernel(
    query_positions,
    rope_positions,
    request_ids,
    query_start_loc,
    raw_state_slot_ids,
    raw_logical_positions,
    raw_rope_positions,
    state_errors,
    rope_position_rows,
    rope_position_row_stride,
    rope_position_axis_stride,
    raw_state_slot_stride,
    raw_position_slot_stride,
    raw_rope_slot_stride,
    raw_rope_ring_stride,
    COMPRESS_RATIO: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
):
    row = tl.program_id(0)
    status = tl.load(state_errors + row).to(tl.int32)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    active = (status == 0) & (request >= 0)
    complete = ((position + 1) % COMPRESS_RATIO) == 0
    if active & complete:
        state_slot = tl.load(raw_state_slot_ids + request * raw_state_slot_stride).to(
            tl.int64
        )
        valid_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
        request_start = tl.load(query_start_loc + request).to(tl.int64)
        current_first = tl.load(query_positions + request_start).to(tl.int64)
        group_first = position - COMPRESS_RATIO + 1
        valid = valid_slot
        for offset in tl.static_range(0, COMPRESS_RATIO):
            source_position = group_first + offset
            from_ring = source_position < current_first
            ring_slot = source_position % RING_CAPACITY
            tag_offset = (state_slot * raw_position_slot_stride + ring_slot).to(
                tl.int64
            )
            observed = tl.load(
                raw_logical_positions + tag_offset,
                mask=valid_slot & from_ring,
                other=source_position,
            ).to(tl.int64)
            valid = valid & (~from_ring | (observed == source_position))

        from_ring = group_first < current_first
        ring_slot = group_first % RING_CAPACITY
        axes = tl.arange(0, 4)
        axis_mask = axes < POSITION_AXES
        rope_base = (
            state_slot * raw_rope_slot_stride + ring_slot * raw_rope_ring_stride
        ).to(tl.int64)
        current_row = request_start + group_first - current_first
        first_rope = tl.load(
            raw_rope_positions + rope_base + axes,
            mask=valid_slot & from_ring & axis_mask,
            other=0,
        ).to(tl.int64)
        current_rope = tl.load(
            rope_positions
            + current_row * rope_position_row_stride
            + axes * rope_position_axis_stride,
            mask=(~from_ring) & axis_mask,
            other=0,
        ).to(tl.int64)
        first_rope = tl.where(from_ring, first_rope, current_rope)
        valid_rope = (
            tl.sum(
                (
                    axis_mask & ((first_rope < 0) | (first_rope >= rope_position_rows))
                ).to(tl.int32),
                axis=0,
            )
            == 0
        )
        if ~(valid & valid_rope):
            tl.atomic_or(state_errors + row, 64)


@triton.jit
def _accumulate_request_errors_kernel(
    request_ids,
    state_errors,
    request_errors,
    MAX_BATCH: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(request_ids + row).to(tl.int64)
    valid_request = (request >= 0) & (request < MAX_BATCH)
    error = tl.load(state_errors + row).to(tl.int32)
    tl.atomic_or(request_errors + request, error, mask=valid_request & (error != 0))


@triton.jit
def _broadcast_request_errors_kernel(
    request_ids,
    request_errors,
    state_errors,
    MAX_BATCH: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(request_ids + row).to(tl.int64)
    valid_request = (request >= 0) & (request < MAX_BATCH)
    request_error = tl.load(
        request_errors + request,
        mask=valid_request,
        other=0,
    ).to(tl.int32)
    prior = tl.load(state_errors + row).to(tl.int32)
    tl.store(state_errors + row, prior | request_error)


@triton.jit
def _compress_completed_groups_kernel(
    raw_index_key,
    query_positions,
    rope_positions,
    request_ids,
    query_start_loc,
    raw_state_slot_ids,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    key_norm_weight,
    rope_cos,
    rope_sin,
    compressed_cache,
    compressed_block_table,
    state_errors,
    rope_position_rows,
    rope_position_row_stride,
    rope_position_axis_stride,
    rope_cos_row_stride,
    rope_sin_row_stride,
    raw_state_slot_stride,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_position_slot_stride,
    raw_rope_slot_stride,
    raw_rope_ring_stride,
    compressed_page_stride,
    compressed_token_stride,
    compressed_table_stride,
    num_compressed_pages,
    eps,
    INDEX_HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    MROPE_SECTION_0: tl.constexpr,
    MROPE_SECTION_1: tl.constexpr,
    ROPE_IS_BF16: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    status = tl.load(state_errors + row).to(tl.int32)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    real_request = (status == 0) & (request >= 0)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=real_request,
        other=-1,
    ).to(tl.int64)
    valid_state_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    complete = ((position + 1) % COMPRESS_RATIO) == 0
    if real_request & complete & valid_state_slot:
        request_start = tl.load(query_start_loc + request).to(tl.int64)
        current_first = tl.load(query_positions + request_start).to(tl.int64)
        group_first = position - COMPRESS_RATIO + 1
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < INDEX_HEAD_DIM
        half_rotary = ROTARY_DIM // 2
        partner_dim = tl.where(
            dims < half_rotary, dims + half_rotary, dims - half_rotary
        )
        total = tl.zeros((BLOCK_D,), tl.float32)
        partner_total = tl.zeros((BLOCK_D,), tl.float32)
        for offset in tl.static_range(0, COMPRESS_RATIO):
            source_position = group_first + offset
            from_current = source_position >= current_first
            current_row = request_start + source_position - current_first
            current = tl.load(
                raw_index_key + current_row * INDEX_HEAD_DIM + dims,
                mask=from_current & dim_mask,
                other=0.0,
            ).to(tl.float32)
            current_partner = tl.load(
                raw_index_key + current_row * INDEX_HEAD_DIM + partner_dim,
                mask=from_current & (dims < ROTARY_DIM),
                other=0.0,
            ).to(tl.float32)
            ring_slot = source_position % RING_CAPACITY
            key_base = (
                state_slot * raw_k_slot_stride + ring_slot * raw_k_ring_stride
            ).to(tl.int64)
            prior = tl.load(
                raw_k_ring + key_base + dims,
                mask=(~from_current) & dim_mask,
                other=0.0,
            ).to(tl.float32)
            prior_partner = tl.load(
                raw_k_ring + key_base + partner_dim,
                mask=(~from_current) & (dims < ROTARY_DIM),
                other=0.0,
            ).to(tl.float32)
            total += tl.where(from_current, current, prior)
            partner_total += tl.where(from_current, current_partner, prior_partner)

        pooled = (total / COMPRESS_RATIO).to(tl.bfloat16)
        pooled_fp32 = pooled.to(tl.float32)
        variance = tl.sum(pooled_fp32 * pooled_fp32, axis=0) / INDEX_HEAD_DIM
        inv_rms = tl.rsqrt(variance + eps)
        weight = tl.load(key_norm_weight + dims, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        normalized = (
            (pooled_fp32 * inv_rms * (1.0 + weight)).to(tl.bfloat16).to(tl.float32)
        )

        in_rotary = dims < ROTARY_DIM
        pair = dims % half_rotary
        partner_pooled = (partner_total / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)
        partner_weight = tl.load(
            key_norm_weight + partner_dim,
            mask=in_rotary,
            other=0.0,
        ).to(tl.float32)
        partner = (
            (partner_pooled * inv_rms * (1.0 + partner_weight))
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        if POSITION_AXES == 1:
            axis = tl.zeros((BLOCK_D,), tl.int32)
        elif MROPE_INTERLEAVED:
            is_height = (pair % 3 == 1) & (pair < 3 * MROPE_SECTION_1)
            is_width = (pair % 3 == 2) & (
                pair < 3 * (ROTARY_DIM // 2 - MROPE_SECTION_0 - MROPE_SECTION_1)
            )
            axis = tl.where(is_height, 1, tl.where(is_width, 2, 0))
        else:
            axis = tl.where(
                pair < MROPE_SECTION_0,
                0,
                tl.where(pair < MROPE_SECTION_0 + MROPE_SECTION_1, 1, 2),
            )
        first_from_current = group_first >= current_first
        first_current_row = request_start + group_first - current_first
        first_ring_slot = group_first % RING_CAPACITY
        rope_base = (
            state_slot * raw_rope_slot_stride + first_ring_slot * raw_rope_ring_stride
        ).to(tl.int64)
        current_rope = tl.load(
            rope_positions
            + first_current_row * rope_position_row_stride
            + axis * rope_position_axis_stride,
            mask=first_from_current & in_rotary,
            other=0,
        ).to(tl.int64)
        ring_rope = tl.load(
            raw_rope_positions + rope_base + axis,
            mask=(~first_from_current) & in_rotary,
            other=0,
        ).to(tl.int64)
        first_rope_position = tl.where(first_from_current, current_rope, ring_rope)
        valid_position = (first_rope_position >= 0) & (
            first_rope_position < rope_position_rows
        )
        cosine = tl.load(
            rope_cos + first_rope_position * rope_cos_row_stride + pair,
            mask=in_rotary & valid_position,
            other=1.0,
        ).to(tl.float32)
        sine = tl.load(
            rope_sin + first_rope_position * rope_sin_row_stride + pair,
            mask=in_rotary & valid_position,
            other=0.0,
        ).to(tl.float32)
        rotated_partner = tl.where(dims < half_rotary, -partner, partner)
        if ROPE_IS_BF16:
            direct_product = _round_fp32_to_bf16(normalized * cosine)
            partner_product = _round_fp32_to_bf16(rotated_partner * sine)
            rotated = _round_fp32_to_bf16(direct_product + partner_product)
        else:
            rotated = normalized * cosine + rotated_partner * sine
        representative = tl.where(in_rotary, rotated, normalized)
        group_id = position // COMPRESS_RATIO
        logical_page = group_id // COMPRESSED_PAGE_SIZE
        page_offset = group_id % COMPRESSED_PAGE_SIZE
        table_offset = (request * compressed_table_stride + logical_page).to(tl.int64)
        physical_page = tl.load(compressed_block_table + table_offset).to(tl.int64)
        valid_page = (physical_page >= 0) & (physical_page < num_compressed_pages)
        if valid_page:
            cache_base = (
                physical_page * compressed_page_stride
                + page_offset * compressed_token_stride
            ).to(tl.int64)
            tl.store(
                compressed_cache + cache_base + dims,
                representative,
                mask=dim_mask,
            )


@triton.jit
def _commit_raw_ring_kernel(
    raw_index_key,
    query_positions,
    rope_positions,
    request_ids,
    query_start_loc,
    sequence_lengths,
    is_prefilling,
    raw_state_slot_ids,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    raw_interval_start_positions,
    state_errors,
    rope_position_row_stride,
    rope_position_axis_stride,
    raw_state_slot_stride,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_position_slot_stride,
    raw_rope_slot_stride,
    raw_rope_ring_stride,
    raw_interval_start_stride,
    INDEX_HEAD_DIM: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    suffix_offset = tl.program_id(1)
    request_start = tl.load(query_start_loc + request).to(tl.int64)
    request_end = tl.load(query_start_loc + request + 1).to(tl.int64)
    query_length = request_end - request_start
    suffix_length = tl.minimum(query_length, RING_CAPACITY)
    row = request_end - suffix_length + suffix_offset
    active = suffix_offset < suffix_length
    status = tl.load(state_errors + request_start, mask=active, other=1)
    observed_request = tl.load(request_ids + row, mask=active, other=-1).to(tl.int64)
    real_request = active & (status == 0) & (observed_request == request)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=real_request,
        other=-1,
    ).to(tl.int64)
    valid_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    if real_request & valid_slot:
        position = tl.load(query_positions + row).to(tl.int64)
        ring_slot = position % RING_CAPACITY
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < INDEX_HEAD_DIM
        key_base = (state_slot * raw_k_slot_stride + ring_slot * raw_k_ring_stride).to(
            tl.int64
        )
        key = tl.load(
            raw_index_key + row * INDEX_HEAD_DIM + dims,
            mask=dim_mask,
            other=0.0,
        )
        tl.store(raw_k_ring + key_base + dims, key, mask=dim_mask)
        tag_offset = (state_slot * raw_position_slot_stride + ring_slot).to(tl.int64)
        tl.store(raw_logical_positions + tag_offset, position)
        axes = tl.arange(0, 4)
        axis_mask = axes < POSITION_AXES
        rope = tl.load(
            rope_positions
            + row * rope_position_row_stride
            + axes * rope_position_axis_stride,
            mask=axis_mask,
            other=-1,
        )
        rope_base = (
            state_slot * raw_rope_slot_stride + ring_slot * raw_rope_ring_stride
        ).to(tl.int64)
        tl.store(raw_rope_positions + rope_base + axes, rope, mask=axis_mask)
        if suffix_offset == 0:
            interval_start_offset = (state_slot * raw_interval_start_stride).to(
                tl.int64
            )
            prefill = tl.load(is_prefilling + request).to(tl.int1)
            sequence_length = tl.load(sequence_lengths + request).to(tl.int64)
            anchor = tl.where(prefill, sequence_length - 1, position)
            tl.store(
                raw_interval_start_positions + interval_start_offset,
                anchor,
            )
    elif real_request:
        tl.atomic_or(state_errors + row, 32)


@triton.jit
def _score_representatives_kernel(
    prepared_query,
    query_positions,
    request_ids,
    sequence_lengths,
    compressed_cache,
    compressed_block_table,
    state_errors,
    scores,
    eligible_counts,
    merge_lengths,
    compressed_page_stride,
    compressed_token_stride,
    compressed_table_stride,
    score_row_stride,
    num_compressed_pages,
    MAX_GROUPS: tl.constexpr,
    GROUP_OFFSET: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    group_block = tl.program_id(1)
    status = tl.load(state_errors + row)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    real_request = (status == 0) & (request >= 0)
    sequence_length = tl.load(
        sequence_lengths + request,
        mask=real_request,
        other=0,
    ).to(tl.int64)
    eligible = tl.minimum(
        (position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    eligible = tl.minimum(eligible, MAX_GROUPS)
    eligible = tl.where(real_request, eligible, 0)
    prior_eligible = tl.minimum(eligible, GROUP_OFFSET)
    carry_count = tl.minimum(prior_eligible, GROUP_BUDGET)
    chunk_eligible = tl.minimum(tl.maximum(eligible - GROUP_OFFSET, 0), GROUP_COUNT)
    if group_block == 0:
        tl.store(eligible_counts + row, eligible)
        tl.store(merge_lengths + row, carry_count + chunk_eligible)
    local_groups = group_block * BLOCK_G + tl.arange(0, BLOCK_G)
    groups = GROUP_OFFSET + local_groups
    group_mask = local_groups < GROUP_COUNT
    active = group_mask & (groups < eligible) & real_request
    logical_pages = groups // COMPRESSED_PAGE_SIZE
    page_offsets = groups % COMPRESSED_PAGE_SIZE
    table_offsets = request * compressed_table_stride + logical_pages.to(tl.int64)
    physical_pages = tl.load(
        compressed_block_table + table_offsets,
        mask=active,
        other=-1,
    ).to(tl.int64)
    valid_pages = (physical_pages >= 0) & (physical_pages < num_compressed_pages)
    bad_pages = tl.sum(active & ~valid_pages, axis=0)
    if bad_pages != 0:
        tl.atomic_or(state_errors + row, 512)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < INDEX_HEAD_DIM
    cache_offsets = (
        physical_pages[:, None] * compressed_page_stride
        + page_offsets[:, None].to(tl.int64) * compressed_token_stride
        + dims[None, :]
    )
    keys = tl.load(
        compressed_cache + cache_offsets,
        mask=active[:, None] & valid_pages[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    score = tl.zeros((BLOCK_G,), tl.float32)
    for head in tl.static_range(0, INDEX_HEADS):
        query_offsets = (row * INDEX_HEADS + head) * INDEX_HEAD_DIM + dims
        query = tl.load(
            prepared_query + query_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)
    score *= 1.0 / math.sqrt(INDEX_HEAD_DIM)
    score = tl.where(active & valid_pages, score, -float("inf"))
    output_columns = carry_count + local_groups
    tl.store(
        scores + row * score_row_stride + output_columns,
        score,
        mask=group_mask,
    )


@triton.jit
def _stage_topk_carry_kernel(
    prior_values,
    eligible_counts,
    scores,
    score_row_stride,
    GROUP_OFFSET: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    eligible = tl.load(eligible_counts + row)
    carry_count = tl.minimum(tl.minimum(eligible, GROUP_OFFSET), GROUP_BUDGET)
    values = tl.load(
        prior_values + row * GROUP_BUDGET + columns,
        mask=columns < carry_count,
        other=-float("inf"),
    )
    tl.store(
        scores + row * score_row_stride + columns,
        values,
        mask=columns < carry_count,
    )


@triton.jit
def _remap_topk_group_ids_kernel(
    local_ids,
    prior_ids,
    eligible_counts,
    merge_lengths,
    GROUP_OFFSET: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    eligible = tl.load(eligible_counts + row)
    carry_count = tl.minimum(tl.minimum(eligible, GROUP_OFFSET), GROUP_BUDGET)
    output_count = tl.minimum(tl.load(merge_lengths + row), GROUP_BUDGET)
    local = tl.load(
        local_ids + row * GROUP_BUDGET + columns,
        mask=columns < GROUP_BUDGET,
        other=-1,
    )
    carried = tl.load(
        prior_ids + row * GROUP_BUDGET + local,
        mask=(columns < output_count) & (local >= 0) & (local < carry_count),
        other=-1,
    )
    global_id = tl.where(
        local < carry_count,
        carried,
        GROUP_OFFSET + local - carry_count,
    )
    global_id = tl.where((columns < output_count) & (local >= 0), global_id, -1)
    tl.store(
        local_ids + row * GROUP_BUDGET + columns,
        global_id,
        mask=columns < GROUP_BUDGET,
    )


@triton.jit
def _stable_topk_threshold_kernel(
    topk_values,
    merge_lengths,
    thresholds,
    greater_totals,
    stable_values,
    stable_ids,
    GROUP_BUDGET: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    selected_count = tl.minimum(tl.load(merge_lengths + row), GROUP_BUDGET)
    active = columns < selected_count
    values = tl.load(
        topk_values + row * GROUP_BUDGET + columns,
        mask=active,
        other=float("inf"),
    ).to(tl.float32)
    threshold = tl.min(values, axis=0)
    greater = tl.sum((active & (values > threshold)).to(tl.int32), axis=0)
    tl.store(thresholds + row, threshold)
    tl.store(greater_totals + row, greater)
    tl.store(
        stable_values + row * GROUP_BUDGET + columns,
        -float("inf"),
        mask=columns < GROUP_BUDGET,
    )
    tl.store(
        stable_ids + row * GROUP_BUDGET + columns,
        -1,
        mask=columns < GROUP_BUDGET,
    )


@triton.jit
def _count_stable_topk_candidates_kernel(
    scores,
    merge_lengths,
    thresholds,
    tie_counts,
    greater_counts,
    score_row_stride,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    columns = block * BLOCK_C + tl.arange(0, BLOCK_C)
    length = tl.load(merge_lengths + row)
    active = columns < length
    values = tl.load(
        scores + row * score_row_stride + columns,
        mask=active,
        other=-float("inf"),
    ).to(tl.float32)
    threshold = tl.load(thresholds + row)
    ties = tl.sum((active & (values == threshold)).to(tl.int32), axis=0)
    greater = tl.sum((active & (values > threshold)).to(tl.int32), axis=0)
    tl.store(tie_counts + row * NUM_BLOCKS + block, ties)
    tl.store(greater_counts + row * NUM_BLOCKS + block, greater)


@triton.jit
def _emit_stable_topk_kernel(
    scores,
    merge_lengths,
    prior_ids,
    eligible_counts,
    thresholds,
    greater_totals,
    tie_counts,
    greater_counts,
    stable_values,
    stable_ids,
    score_row_stride,
    GROUP_OFFSET: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_COUNTS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    count_columns = tl.arange(0, BLOCK_COUNTS)
    prior_blocks = (count_columns < block) & (count_columns < NUM_BLOCKS)
    ties_before = tl.sum(
        tl.load(
            tie_counts + row * NUM_BLOCKS + count_columns,
            mask=prior_blocks,
            other=0,
        ),
        axis=0,
    )
    greater_before = tl.sum(
        tl.load(
            greater_counts + row * NUM_BLOCKS + count_columns,
            mask=prior_blocks,
            other=0,
        ),
        axis=0,
    )

    columns = block * BLOCK_C + tl.arange(0, BLOCK_C)
    length = tl.load(merge_lengths + row)
    active = columns < length
    values = tl.load(
        scores + row * score_row_stride + columns,
        mask=active,
        other=-float("inf"),
    ).to(tl.float32)
    threshold = tl.load(thresholds + row)
    greater = active & (values > threshold)
    ties = active & (values == threshold)
    selected_count = tl.minimum(length, GROUP_BUDGET)
    tie_needed = selected_count - tl.load(greater_totals + row)
    tie_rank = ties_before + tl.cumsum(ties.to(tl.int32), axis=0) - 1
    selected = greater | (ties & (tie_rank < tie_needed))
    selected_before = greater_before + tl.minimum(ties_before, tie_needed)
    output_columns = selected_before + tl.cumsum(selected.to(tl.int32), axis=0) - 1

    eligible = tl.load(eligible_counts + row)
    carry_count = tl.minimum(tl.minimum(eligible, GROUP_OFFSET), GROUP_BUDGET)
    carried = tl.load(
        prior_ids + row * GROUP_BUDGET + columns,
        mask=active & (columns < carry_count),
        other=-1,
    )
    global_ids = tl.where(
        columns < carry_count,
        carried,
        GROUP_OFFSET + columns - carry_count,
    )
    tl.store(
        stable_values + row * GROUP_BUDGET + output_columns,
        values,
        mask=selected,
    )
    tl.store(
        stable_ids + row * GROUP_BUDGET + output_columns,
        global_ids,
        mask=selected,
    )


@triton.jit
def _copy_stable_topk_kernel(
    stable_values,
    stable_ids,
    topk_values,
    topk_ids,
    GROUP_BUDGET: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    mask = columns < GROUP_BUDGET
    values = tl.load(
        stable_values + row * GROUP_BUDGET + columns,
        mask=mask,
        other=-float("inf"),
    )
    ids = tl.load(
        stable_ids + row * GROUP_BUDGET + columns,
        mask=mask,
        other=-1,
    )
    score_bits = values.to(tl.uint32, bitcast=True).to(tl.uint64)
    id_key = (0xFFFFFFFF - ids.to(tl.uint32)).to(tl.uint64)
    keys = tl.where(
        (columns < GROUP_BUDGET) & (ids >= 0), (score_bits << 32) | id_key, 0
    )
    keys = tl.sort(keys, dim=0, descending=True)
    sorted_ids = (0xFFFFFFFF - (keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    sorted_values = (keys >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    valid = mask & (keys != 0)
    tl.store(
        topk_values + row * GROUP_BUDGET + columns,
        tl.where(valid, sorted_values, -float("inf")),
        mask=mask,
    )
    tl.store(
        topk_ids + row * GROUP_BUDGET + columns,
        tl.where(valid, sorted_ids, -1),
        mask=mask,
    )


@triton.jit
def _expand_selected_groups_kernel(
    topk_group_ids,
    eligible_counts,
    query_positions,
    state_errors,
    selected_positions,
    topk_row_stride,
    selected_row_stride,
    GROUP_BUDGET: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_W)
    column_mask = columns < SELECTION_WIDTH
    status = tl.load(state_errors + row)
    eligible = tl.load(eligible_counts + row)
    selected_groups = tl.minimum(eligible, GROUP_BUDGET)
    expanded_count = selected_groups * COMPRESS_RATIO
    group_columns = columns // COMPRESS_RATIO
    group_ids = tl.load(
        topk_group_ids + row * topk_row_stride + group_columns,
        mask=column_mask & (columns < expanded_count),
        other=-1,
    )
    expanded = group_ids * COMPRESS_RATIO + columns % COMPRESS_RATIO
    position = tl.load(query_positions + row).to(tl.int64)
    tail_start = ((position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_length = position + 1 - tail_start
    tail_column = columns - expanded_count
    in_tail = (tail_column >= 0) & (tail_column < tail_length)
    result = tl.where(
        columns < expanded_count,
        expanded,
        tl.where(in_tail, tail_start + tail_column, -1),
    )
    result = tl.where(status == 0, result, -1)
    tl.store(
        selected_positions + row * selected_row_stride + columns,
        result,
        mask=column_mask,
    )


@triton.jit
def _poison_failed_rows_kernel(
    output,
    state_errors,
    elements_per_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements_per_row
    if tl.load(state_errors + row) != 0:
        tl.store(output + row * elements_per_row + offsets, float("nan"), mask=mask)


def _mrope_sections(caps) -> tuple[int, int]:
    if int(caps.position_axes) == 1:
        return 0, 0
    assert caps.mrope_sections is not None
    return int(caps.mrope_sections[0]), int(caps.mrope_sections[1])


def launch_validate_rows(
    *,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    request_errors: torch.Tensor,
    state_errors: torch.Tensor,
    rope_position_rows: int,
    caps,
) -> None:
    rows = int(request_ids.shape[0])
    block_batch = triton.next_power_of_2(int(caps.max_batch))
    _validate_packed_boundaries_kernel[(1,)](
        query_start_loc,
        request_errors,
        rows,
        MAX_BATCH=int(caps.max_batch),
        BLOCK_BATCH=block_batch,
        num_warps=1,
    )
    _validate_packed_requests_kernel[(int(caps.max_batch),)](
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        is_prefilling,
        raw_state_slot_ids,
        raw_interval_start_positions,
        request_errors,
        int(raw_state_slot_ids.stride(0)),
        int(raw_interval_start_positions.stride(0)),
        MAX_BATCH=int(caps.max_batch),
        MAX_SEQ_LEN=int(caps.max_seq_len),
        MAX_SPECULATIVE=int(caps.max_speculative_tokens),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        BLOCK_BATCH=block_batch,
        num_warps=1,
    )
    _materialize_packed_row_errors_kernel[(rows,)](
        request_ids,
        query_positions,
        rope_positions,
        sequence_lengths,
        query_start_loc,
        request_errors,
        state_errors,
        int(rope_positions.stride(0)),
        int(rope_positions.stride(1)),
        rows,
        int(rope_position_rows),
        MAX_BATCH=int(caps.max_batch),
        POSITION_AXES=int(caps.position_axes),
        num_warps=1,
    )


def launch_validate_page_tables(
    *,
    request_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
    main_block_table: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    state_errors: torch.Tensor,
    num_main_pages: int,
    num_compressed_pages: int,
    shared_compressed_raw_pool: bool,
    caps,
) -> None:
    rows = int(request_ids.shape[0])
    block_p = 32
    table_width = max(
        int(main_block_table.shape[1]), int(compressed_block_table.shape[1])
    )
    _validate_page_tables_kernel[(rows, triton.cdiv(table_width, block_p))](
        request_ids,
        sequence_lengths,
        main_block_table,
        compressed_block_table,
        raw_state_slot_ids,
        state_errors,
        int(main_block_table.stride(0)),
        int(compressed_block_table.stride(0)),
        int(raw_state_slot_ids.stride(0)),
        int(num_main_pages),
        int(num_compressed_pages),
        MAIN_TABLE_WIDTH=int(main_block_table.shape[1]),
        COMPRESSED_TABLE_WIDTH=int(compressed_block_table.shape[1]),
        MAIN_PAGE_SIZE=int(caps.main_page_size),
        COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
        COMPRESS_RATIO=int(caps.compress_ratio),
        MAX_BATCH=int(caps.max_batch),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        SHARED_COMPRESSED_RAW_POOL=bool(shared_compressed_raw_pool),
        BLOCK_BATCH=triton.next_power_of_2(int(caps.max_batch)),
        BLOCK_P=block_p,
        num_warps=1,
    )


def launch_validate_shared_pool_ownership(
    *,
    request_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    state_errors: torch.Tensor,
    occupancy: torch.Tensor,
    num_compressed_pages: int,
    caps,
) -> None:
    block = 256
    occupancy_and_error = int(num_compressed_pages) + 1
    _clear_shared_page_occupancy_kernel[(triton.cdiv(occupancy_and_error, block),)](
        occupancy,
        occupancy_and_error,
        BLOCK=block,
        num_warps=4,
    )
    block_p = 32
    _mark_live_compressed_pages_kernel[
        (
            int(caps.max_batch),
            triton.cdiv(int(compressed_block_table.shape[1]), block_p),
        )
    ](
        sequence_lengths,
        compressed_block_table,
        occupancy,
        int(num_compressed_pages),
        int(compressed_block_table.stride(0)),
        int(num_compressed_pages),
        MAX_SEQ_LEN=int(caps.max_seq_len),
        COMPRESSED_TABLE_WIDTH=int(compressed_block_table.shape[1]),
        COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
        COMPRESS_RATIO=int(caps.compress_ratio),
        BLOCK_P=block_p,
        num_warps=1,
    )
    _validate_active_raw_slots_kernel[(int(request_ids.shape[0]),)](
        request_ids,
        raw_state_slot_ids,
        state_errors,
        occupancy,
        int(raw_state_slot_ids.stride(0)),
        int(num_compressed_pages),
        num_warps=1,
    )


def launch_prepare_index_query(
    *,
    index_query: torch.Tensor,
    request_ids: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    state_errors: torch.Tensor,
    prepared_query: torch.Tensor,
    caps,
) -> None:
    rows = int(index_query.shape[0])
    section0, section1 = _mrope_sections(caps)
    _prepare_index_query_kernel[(rows * int(caps.index_heads),)](
        index_query,
        request_ids,
        norm_weight,
        rope_positions,
        rope_cos,
        rope_sin,
        state_errors,
        prepared_query,
        int(rope_cos.shape[0]),
        float(caps.rms_norm_eps),
        int(rope_positions.stride(0)),
        int(rope_positions.stride(1)),
        int(rope_cos.stride(0)),
        int(rope_sin.stride(0)),
        INDEX_HEADS=int(caps.index_heads),
        HEAD_DIM=int(caps.index_head_dim),
        ROTARY_DIM=int(caps.index_rotary_dim),
        POSITION_AXES=int(caps.position_axes),
        MROPE_INTERLEAVED=bool(caps.mrope_interleaved),
        MROPE_SECTION_0=section0,
        MROPE_SECTION_1=section1,
        ROPE_IS_BF16=rope_cos.dtype == torch.bfloat16,
        BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
        num_warps=4,
    )


def launch_validate_completed_groups(
    *,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    state_errors: torch.Tensor,
    rope_position_rows: int,
    caps,
) -> None:
    rows = int(query_positions.shape[0])
    _validate_completed_groups_kernel[(rows,)](
        query_positions,
        rope_positions,
        request_ids,
        query_start_loc,
        raw_state_slot_ids,
        raw_logical_positions,
        raw_rope_positions,
        state_errors,
        int(rope_position_rows),
        int(rope_positions.stride(0)),
        int(rope_positions.stride(1)),
        int(raw_state_slot_ids.stride(0)),
        int(raw_logical_positions.stride(0)),
        int(raw_rope_positions.stride(0)),
        int(raw_rope_positions.stride(1)),
        COMPRESS_RATIO=int(caps.compress_ratio),
        RING_CAPACITY=int(caps.raw_ring_capacity),
        POSITION_AXES=int(caps.position_axes),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        num_warps=1,
    )


def launch_propagate_request_errors(
    *,
    request_ids: torch.Tensor,
    request_errors: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    rows = int(request_ids.shape[0])
    _accumulate_request_errors_kernel[(rows,)](
        request_ids,
        state_errors,
        request_errors,
        MAX_BATCH=int(caps.max_batch),
        num_warps=1,
    )
    _broadcast_request_errors_kernel[(rows,)](
        request_ids,
        request_errors,
        state_errors,
        MAX_BATCH=int(caps.max_batch),
        num_warps=1,
    )


def launch_compress_completed_groups(
    *,
    raw_index_key: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    key_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    rows = int(raw_index_key.shape[0])
    section0, section1 = _mrope_sections(caps)
    _compress_completed_groups_kernel[(rows,)](
        raw_index_key,
        query_positions,
        rope_positions,
        request_ids,
        query_start_loc,
        raw_state_slot_ids,
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        key_norm_weight,
        rope_cos,
        rope_sin,
        compressed_cache,
        compressed_block_table,
        state_errors,
        int(rope_cos.shape[0]),
        int(rope_positions.stride(0)),
        int(rope_positions.stride(1)),
        int(rope_cos.stride(0)),
        int(rope_sin.stride(0)),
        int(raw_state_slot_ids.stride(0)),
        int(raw_k_ring.stride(0)),
        int(raw_k_ring.stride(1)),
        int(raw_logical_positions.stride(0)),
        int(raw_rope_positions.stride(0)),
        int(raw_rope_positions.stride(1)),
        int(compressed_cache.stride(0)),
        int(compressed_cache.stride(1)),
        int(compressed_block_table.stride(0)),
        int(compressed_cache.shape[0]),
        float(caps.rms_norm_eps),
        INDEX_HEAD_DIM=int(caps.index_head_dim),
        ROTARY_DIM=int(caps.index_rotary_dim),
        COMPRESS_RATIO=int(caps.compress_ratio),
        RING_CAPACITY=int(caps.raw_ring_capacity),
        COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
        POSITION_AXES=int(caps.position_axes),
        MROPE_INTERLEAVED=bool(caps.mrope_interleaved),
        MROPE_SECTION_0=section0,
        MROPE_SECTION_1=section1,
        ROPE_IS_BF16=rope_cos.dtype == torch.bfloat16,
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
        num_warps=4,
    )


def launch_commit_raw_ring(
    *,
    raw_index_key: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    sequence_lengths: torch.Tensor,
    is_prefilling: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    _commit_raw_ring_kernel[
        (int(caps.max_batch), int(caps.raw_ring_capacity))
    ](
        raw_index_key,
        query_positions,
        rope_positions,
        request_ids,
        query_start_loc,
        sequence_lengths,
        is_prefilling,
        raw_state_slot_ids,
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
        state_errors,
        int(rope_positions.stride(0)),
        int(rope_positions.stride(1)),
        int(raw_state_slot_ids.stride(0)),
        int(raw_k_ring.stride(0)),
        int(raw_k_ring.stride(1)),
        int(raw_logical_positions.stride(0)),
        int(raw_rope_positions.stride(0)),
        int(raw_rope_positions.stride(1)),
        int(raw_interval_start_positions.stride(0)),
        INDEX_HEAD_DIM=int(caps.index_head_dim),
        POSITION_AXES=int(caps.position_axes),
        RING_CAPACITY=int(caps.raw_ring_capacity),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
        num_warps=4,
    )


def launch_score_representatives(
    *,
    prepared_query: torch.Tensor,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    state_errors: torch.Tensor,
    scores: torch.Tensor,
    eligible_counts: torch.Tensor,
    merge_lengths: torch.Tensor,
    group_offset: int,
    group_count: int,
    caps,
) -> None:
    rows = int(prepared_query.shape[0])
    block_g = 32
    _score_representatives_kernel[(rows, triton.cdiv(int(group_count), block_g))](
        prepared_query,
        query_positions,
        request_ids,
        sequence_lengths,
        compressed_cache,
        compressed_block_table,
        state_errors,
        scores,
        eligible_counts,
        merge_lengths,
        int(compressed_cache.stride(0)),
        int(compressed_cache.stride(1)),
        int(compressed_block_table.stride(0)),
        int(scores.stride(0)),
        int(compressed_cache.shape[0]),
        MAX_GROUPS=int(caps.max_groups),
        GROUP_OFFSET=int(group_offset),
        GROUP_COUNT=int(group_count),
        GROUP_BUDGET=int(caps.group_budget),
        INDEX_HEADS=int(caps.index_heads),
        INDEX_HEAD_DIM=int(caps.index_head_dim),
        COMPRESS_RATIO=int(caps.compress_ratio),
        COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
        BLOCK_G=block_g,
        BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
        num_warps=4,
    )


def launch_stage_topk_carry(
    *,
    prior_values: torch.Tensor,
    eligible_counts: torch.Tensor,
    scores: torch.Tensor,
    group_offset: int,
    group_budget: int,
) -> None:
    rows = int(scores.shape[0])
    _stage_topk_carry_kernel[(rows,)](
        prior_values,
        eligible_counts,
        scores,
        int(scores.stride(0)),
        GROUP_OFFSET=int(group_offset),
        GROUP_BUDGET=int(group_budget),
        BLOCK_K=triton.next_power_of_2(int(group_budget)),
        num_warps=8,
    )


def launch_topk_groups(
    *,
    scores: torch.Tensor,
    eligible_counts: torch.Tensor,
    topk_values: torch.Tensor,
    topk_group_ids: torch.Tensor,
    group_budget: int,
) -> None:
    # This is the exact single-row radix kernel, not the cooperative multi-CTA
    # persistent path.
    from ..dsa_indexer.tiled_topk import run_row_topk

    run_row_topk(
        row_logits=scores,
        lengths=eligible_counts,
        topk=int(group_budget),
        output_values=topk_values,
        output_indices=topk_group_ids,
    )


def launch_remap_topk_group_ids(
    *,
    local_ids: torch.Tensor,
    prior_ids: torch.Tensor,
    eligible_counts: torch.Tensor,
    merge_lengths: torch.Tensor,
    group_offset: int,
    group_budget: int,
) -> None:
    rows = int(local_ids.shape[0])
    _remap_topk_group_ids_kernel[(rows,)](
        local_ids,
        prior_ids,
        eligible_counts,
        merge_lengths,
        GROUP_OFFSET=int(group_offset),
        GROUP_BUDGET=int(group_budget),
        BLOCK_K=triton.next_power_of_2(int(group_budget)),
        num_warps=8,
    )


def launch_stabilize_topk(
    *,
    scores: torch.Tensor,
    merge_lengths: torch.Tensor,
    prior_ids: torch.Tensor,
    eligible_counts: torch.Tensor,
    topk_values: torch.Tensor,
    topk_group_ids: torch.Tensor,
    tie_counts: torch.Tensor,
    greater_counts: torch.Tensor,
    stable_values: torch.Tensor,
    stable_ids: torch.Tensor,
    thresholds: torch.Tensor,
    greater_totals: torch.Tensor,
    group_offset: int,
    group_budget: int,
) -> None:
    """Make threshold ties exact and stable by retaining lower group IDs."""
    rows = int(scores.shape[0])
    num_blocks = int(tie_counts.shape[1])
    block_k = triton.next_power_of_2(int(group_budget))
    _stable_topk_threshold_kernel[(rows,)](
        topk_values,
        merge_lengths,
        thresholds,
        greater_totals,
        stable_values,
        stable_ids,
        GROUP_BUDGET=int(group_budget),
        BLOCK_K=block_k,
        num_warps=8,
    )
    _count_stable_topk_candidates_kernel[(rows, num_blocks)](
        scores,
        merge_lengths,
        thresholds,
        tie_counts,
        greater_counts,
        int(scores.stride(0)),
        NUM_BLOCKS=num_blocks,
        BLOCK_C=512,
        num_warps=8,
    )
    _emit_stable_topk_kernel[(rows, num_blocks)](
        scores,
        merge_lengths,
        prior_ids,
        eligible_counts,
        thresholds,
        greater_totals,
        tie_counts,
        greater_counts,
        stable_values,
        stable_ids,
        int(scores.stride(0)),
        GROUP_OFFSET=int(group_offset),
        GROUP_BUDGET=int(group_budget),
        NUM_BLOCKS=num_blocks,
        BLOCK_COUNTS=triton.next_power_of_2(num_blocks),
        BLOCK_C=512,
        num_warps=8,
    )
    _copy_stable_topk_kernel[(rows,)](
        stable_values,
        stable_ids,
        topk_values,
        topk_group_ids,
        GROUP_BUDGET=int(group_budget),
        BLOCK_K=block_k,
        num_warps=8,
    )


def launch_expand_selected_groups(
    *,
    topk_group_ids: torch.Tensor,
    eligible_counts: torch.Tensor,
    query_positions: torch.Tensor,
    state_errors: torch.Tensor,
    selected_positions: torch.Tensor,
    caps,
) -> None:
    rows = int(query_positions.shape[0])
    _expand_selected_groups_kernel[(rows,)](
        topk_group_ids,
        eligible_counts,
        query_positions,
        state_errors,
        selected_positions,
        int(topk_group_ids.stride(0)),
        int(selected_positions.stride(0)),
        GROUP_BUDGET=int(caps.group_budget),
        COMPRESS_RATIO=int(caps.compress_ratio),
        SELECTION_WIDTH=int(caps.selection_width),
        BLOCK_W=triton.next_power_of_2(int(caps.selection_width)),
        num_warps=8,
    )


def launch_poison_failed_rows(
    *,
    output: torch.Tensor,
    state_errors: torch.Tensor,
) -> None:
    rows = int(output.shape[0])
    elements = int(output.shape[1]) * int(output.shape[2])
    block = 256
    _poison_failed_rows_kernel[(rows, triton.cdiv(elements, block))](
        output,
        state_errors,
        elements,
        BLOCK=block,
        num_warps=4,
    )


__all__ = [
    "launch_validate_rows",
    "launch_validate_page_tables",
    "launch_validate_shared_pool_ownership",
    "launch_prepare_index_query",
    "launch_validate_completed_groups",
    "launch_propagate_request_errors",
    "launch_compress_completed_groups",
    "launch_commit_raw_ring",
    "launch_score_representatives",
    "launch_stage_topk_carry",
    "launch_topk_groups",
    "launch_remap_topk_group_ids",
    "launch_stabilize_topk",
    "launch_expand_selected_groups",
    "launch_poison_failed_rows",
]
