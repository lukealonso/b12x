"""Opaque launches for packed sequential GDN decode and output gating."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_VALIDATION_BLOCK = 256


@triton.jit
def _reset_validation_kernel(
    duplicate_slots,
    error_code,
    TABLE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    tl.store(duplicate_slots + offsets, -1, mask=offsets < TABLE_SIZE)
    if tl.program_id(0) == 0:
        tl.store(error_code, 0)


@triton.jit(
    do_not_specialize=["token_capacity", "sequence_capacity", "state_index_columns"]
)
def _validate_packed_metadata_kernel(
    query_start_loc,
    num_accepted_tokens,
    num_seqs,
    num_tokens,
    error_code,
    token_capacity,
    sequence_capacity,
    state_index_columns,
):
    request = tl.program_id(0)
    live_seqs = tl.load(num_seqs).to(tl.int32)
    live_tokens = tl.load(num_tokens).to(tl.int32)

    if request == 0:
        counts_invalid = (
            (live_seqs < 0)
            | (live_seqs > sequence_capacity)
            | (live_tokens < 0)
            | (live_tokens > token_capacity)
        )
        first = tl.load(query_start_loc).to(tl.int32)
        safe_last = tl.maximum(0, tl.minimum(live_seqs, sequence_capacity))
        last = tl.load(query_start_loc + safe_last).to(tl.int32)
        if counts_invalid | (first != 0) | (last != live_tokens):
            tl.atomic_or(error_code, 2)

    if request < tl.maximum(0, tl.minimum(live_seqs, sequence_capacity)):
        start = tl.load(query_start_loc + request).to(tl.int32)
        end = tl.load(query_start_loc + request + 1).to(tl.int32)
        accepted = tl.load(num_accepted_tokens + request).to(tl.int32)
        length = end - start
        invalid = (
            (start < 0)
            | (end < start)
            | (end > live_tokens)
            | (length > state_index_columns)
            | (accepted < 1)
            | (accepted > state_index_columns)
        )
        if invalid:
            tl.atomic_or(error_code, 2)


@triton.jit(
    do_not_specialize=[
        "sequence_capacity",
        "state_index_columns",
        "stride_indices_request",
        "stride_indices_column",
    ]
)
def _validate_active_state_slots_kernel(
    query_start_loc,
    num_accepted_tokens,
    state_indices,
    num_seqs,
    duplicate_slots,
    error_code,
    sequence_capacity,
    state_index_columns,
    stride_indices_request,
    stride_indices_column,
    MAX_STATE_SLOTS: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    HAS_NULL_STATE_INDEX: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
):
    cell = tl.program_id(0)
    request = cell // state_index_columns
    column = cell % state_index_columns
    live_seqs = tl.load(num_seqs).to(tl.int32)
    if request >= tl.maximum(0, tl.minimum(live_seqs, sequence_capacity)):
        return

    start = tl.load(query_start_loc + request).to(tl.int32)
    end = tl.load(query_start_loc + request + 1).to(tl.int32)
    accepted_column = tl.load(num_accepted_tokens + request).to(tl.int32) - 1
    active = (end > start) & ((column < (end - start)) | (column == accepted_column))
    if not active:
        return

    if HAS_NULL_STATE_INDEX:
        safe_accepted_column = tl.maximum(
            0, tl.minimum(accepted_column, state_index_columns - 1)
        )
        source_idx = tl.load(
            state_indices
            + request.to(tl.int64) * stride_indices_request
            + safe_accepted_column.to(tl.int64) * stride_indices_column
        ).to(tl.int64)
        if source_idx == NULL_STATE_INDEX:
            return

    state_idx = tl.load(
        state_indices
        + request.to(tl.int64) * stride_indices_request
        + column.to(tl.int64) * stride_indices_column
    ).to(tl.int64)
    if HAS_NULL_STATE_INDEX:
        if state_idx == NULL_STATE_INDEX:
            return
    if (state_idx < 0) | (state_idx >= MAX_STATE_SLOTS):
        tl.atomic_or(error_code, 4)
        return

    empty_slot = tl.full((), -1, tl.int64)
    slot = state_idx % TABLE_SIZE
    probes = tl.full((), 0, tl.int32)
    done = tl.full((), False, tl.int1)
    duplicate = tl.full((), False, tl.int1)
    while (probes < TABLE_SIZE) & ~done:
        previous = tl.atomic_cas(duplicate_slots + slot, empty_slot, state_idx)
        inserted = previous == empty_slot
        duplicate = previous == state_idx
        done = inserted | duplicate
        slot = (slot + 1) % TABLE_SIZE
        probes += 1

    if duplicate | ~done:
        tl.atomic_or(error_code, 1)


@triton.jit(
    do_not_specialize=[
        "sequence_capacity",
        "state_index_columns",
        "stride_indices_request",
        "stride_indices_column",
    ]
)
def _packed_sequential_kda_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    recurrent_state,
    query_start_loc,
    num_accepted_tokens,
    state_indices,
    num_seqs,
    output,
    error_code,
    scale,
    lower_bound,
    sequence_capacity,
    state_index_columns,
    stride_mixed_token: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_indices_request,
    stride_indices_column,
    stride_output_token: tl.constexpr,
    stride_output_head: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    KEY_HEADS: tl.constexpr,
    VALUE_HEADS: tl.constexpr,
    KEY_HEAD_DIM: tl.constexpr,
    VALUE_HEAD_DIM: tl.constexpr,
    STATE_INDEX_COLUMNS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    QK_L2NORM: tl.constexpr,
    HAS_NULL_STATE_INDEX: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
    VALIDATE_METADATA: tl.constexpr,
):
    value_tile = tl.program_id(0)
    request_value_head = tl.program_id(1)
    request = request_value_head // VALUE_HEADS
    value_head = request_value_head % VALUE_HEADS
    key_head = value_head

    live_seqs = tl.load(num_seqs).to(tl.int32)
    if request >= tl.maximum(0, tl.minimum(live_seqs, sequence_capacity)):
        return
    if VALIDATE_METADATA:
        if tl.load(error_code).to(tl.int32) != 0:
            return

    start = tl.load(query_start_loc + request).to(tl.int32)
    end = tl.load(query_start_loc + request + 1).to(tl.int32)
    if end <= start:
        return
    accepted_column = tl.load(num_accepted_tokens + request).to(tl.int32) - 1
    source_idx = tl.load(
        state_indices
        + request.to(tl.int64) * stride_indices_request
        + accepted_column.to(tl.int64) * stride_indices_column
    ).to(tl.int64)

    key_cols = tl.arange(0, KEY_HEAD_DIM)
    value_rows = value_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    key_mask = key_cols < KEY_HEAD_DIM
    value_mask = value_rows < VALUE_HEAD_DIM
    state_mask = value_mask[:, None] & key_mask[None, :]

    if HAS_NULL_STATE_INDEX:
        if source_idx == NULL_STATE_INDEX:
            for relative_token in range(STATE_INDEX_COLUMNS):
                if (relative_token < state_index_columns) & (
                    relative_token < (end - start)
                ):
                    token = start + relative_token
                    output_offsets = (
                        token.to(tl.int64) * stride_output_token
                        + value_head.to(tl.int64) * stride_output_head
                        + value_rows.to(tl.int64)
                    )
                    tl.store(output + output_offsets, 0.0, mask=value_mask)
            return

    # Pool-scaled arithmetic is explicitly Int64. Valid serving pools can place
    # recycled state slots beyond the signed-32-bit element-offset boundary.
    source_offsets = (
        source_idx * stride_state_slot
        + value_head.to(tl.int64) * stride_state_head
        + value_rows[:, None].to(tl.int64) * stride_state_v
        + key_cols[None, :].to(tl.int64)
    )
    state = tl.load(recurrent_state + source_offsets, mask=state_mask, other=0.0).to(
        tl.float32
    )

    for relative_token in range(STATE_INDEX_COLUMNS):
        if (relative_token < state_index_columns) & (relative_token < (end - start)):
            token = start + relative_token
            token_i64 = token.to(tl.int64)
            mixed_base = token_i64 * stride_mixed_token
            q_offsets = key_head * KEY_HEAD_DIM + key_cols
            k_offsets = KEY_HEADS * KEY_HEAD_DIM + q_offsets
            v_offsets = (
                2 * KEY_HEADS * KEY_HEAD_DIM + value_head * VALUE_HEAD_DIM + value_rows
            )
            q = tl.load(
                mixed_qkv + mixed_base + q_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            k = tl.load(
                mixed_qkv + mixed_base + k_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            value = tl.load(
                mixed_qkv + mixed_base + v_offsets, mask=value_mask, other=0.0
            ).to(tl.float32)
            if QK_L2NORM:
                q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6)
                k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)
            q *= scale

            b_value = tl.load(
                b + token_i64 * stride_b_token + value_head.to(tl.int64) * stride_b_head
            ).to(tl.float32)
            A_log_value = tl.load(A_log + value_head).to(tl.float32)
            raw_gate = tl.load(
                a
                + token_i64 * stride_a_token
                + value_head.to(tl.int64) * stride_a_head
                + key_cols.to(tl.int64),
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            gate_bias = tl.load(
                dt_bias
                + value_head.to(tl.int64) * stride_dt_bias_head
                + key_cols.to(tl.int64),
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            log_decay = lower_bound * tl.sigmoid(
                tl.exp(A_log_value) * (raw_gate + gate_bias)
            )
            state *= tl.exp(log_decay)[None, :]
            beta = tl.sigmoid(b_value)

            value -= tl.sum(state * k[None, :], axis=1)
            value *= beta
            state += value[:, None] * k[None, :]
            decoded = tl.sum(state * q[None, :], axis=1)
            output_offsets = (
                token_i64 * stride_output_token
                + value_head.to(tl.int64) * stride_output_head
                + value_rows.to(tl.int64)
            )
            tl.store(output + output_offsets, decoded, mask=value_mask)

            destination_idx = tl.load(
                state_indices
                + request.to(tl.int64) * stride_indices_request
                + relative_token * stride_indices_column
            ).to(tl.int64)
            destination_offsets = (
                destination_idx * stride_state_slot
                + value_head.to(tl.int64) * stride_state_head
                + value_rows[:, None].to(tl.int64) * stride_state_v
                + key_cols[None, :].to(tl.int64)
            )
            destination_mask = state_mask
            if HAS_NULL_STATE_INDEX:
                destination_mask &= destination_idx != NULL_STATE_INDEX
            tl.store(
                recurrent_state + destination_offsets,
                state,
                mask=destination_mask,
            )


@triton.jit(do_not_specialize=["token_capacity"])
def _gated_rmsnorm_kernel(
    output,
    z,
    norm_weight,
    num_tokens,
    error_code,
    eps,
    token_capacity,
    stride_output_token: tl.constexpr,
    stride_output_head: tl.constexpr,
    stride_z_token: tl.constexpr,
    stride_z_head: tl.constexpr,
    VALUE_HEADS: tl.constexpr,
    VALUE_HEAD_DIM: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
    NORM_WEIGHT_FP32: tl.constexpr,
    KDA_NORM_FP32: tl.constexpr,
    VALIDATE_METADATA: tl.constexpr = True,
):
    token_value_head = tl.program_id(0)
    token = token_value_head // VALUE_HEADS
    value_head = token_value_head % VALUE_HEADS
    cols = tl.arange(0, VALUE_HEAD_DIM)
    mask = cols < VALUE_HEAD_DIM
    token_i64 = token.to(tl.int64)
    output_offsets = (
        token_i64 * stride_output_token
        + value_head.to(tl.int64) * stride_output_head
        + cols.to(tl.int64)
    )
    if VALIDATE_METADATA:
        error = tl.load(error_code).to(tl.int32)
        if error != 0:
            tl.store(output + output_offsets, float("nan"), mask=mask)
            return
    live_tokens = tl.load(num_tokens).to(tl.int32)
    if token >= tl.maximum(0, tl.minimum(live_tokens, token_capacity)):
        tl.store(output + output_offsets, 0.0, mask=mask)
        return

    z_offsets = (
        token_i64 * stride_z_token
        + value_head.to(tl.int64) * stride_z_head
        + cols.to(tl.int64)
    )
    values = tl.load(output + output_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / VALUE_HEAD_DIM
    normalized = values * tl.rsqrt(variance + eps)
    if KDA_NORM_FP32:
        weight = tl.load(norm_weight + cols, mask=mask, other=0.0).to(tl.float32)
        weighted = normalized * weight
    elif NORM_WEIGHT_FP32:
        normalized = normalized.to(tl.bfloat16)
        weight = tl.load(norm_weight + cols, mask=mask, other=0.0).to(tl.float32)
        weighted = normalized.to(tl.float32) * weight
    else:
        normalized = normalized.to(tl.bfloat16)
        weight = tl.load(norm_weight + cols, mask=mask, other=0.0).to(tl.bfloat16)
        weighted = (normalized * weight).to(tl.bfloat16).to(tl.float32)
    gate_input = tl.load(z + z_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.sigmoid(gate_input)
    if not SIGMOID_GATE:
        gate *= gate_input
    tl.store(output + output_offsets, weighted * gate, mask=mask)


def _make_qwen_binding(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
    duplicate_slots: torch.Tensor,
    error_code: torch.Tensor,
    *,
    max_tokens: int,
    max_seqs: int,
    max_state_slots: int,
    state_index_columns: int,
    key_heads: int,
    value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    sigmoid_gate: bool,
    qk_l2norm: bool,
    null_state_index: int | None,
    duplicate_table_size: int,
):
    from ._impl import Binding, Caps, _materialize_plan

    caps = Caps(
        device=mixed_qkv.device,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        max_state_slots=max_state_slots,
        key_heads=key_heads,
        value_heads=value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        state_index_columns=state_index_columns,
        model_dtype=mixed_qkv.dtype,
        state_dtype=recurrent_state.dtype,
        gate_activation="sigmoid" if sigmoid_gate else "silu",
        qk_l2norm=qk_l2norm,
        null_state_index=null_state_index,
    )
    launch_plan = _materialize_plan(caps, policy_resolution=None)
    if launch_plan.duplicate_table_size != duplicate_table_size:
        raise ValueError("GDN duplicate-table capacity does not match the plan")
    return Binding(
        plan=launch_plan,
        scratch=duplicate_slots,
        duplicate_slots=duplicate_slots,
        error_code=error_code,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        z=z,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        recurrent_state=recurrent_state,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        state_indices=state_indices,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        output=output,
    )


def _launch_gdn_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
    duplicate_slots: torch.Tensor,
    error_code: torch.Tensor,
    eps: float,
    scale: float,
    lower_bound: float,
    max_tokens: int,
    max_seqs: int,
    max_state_slots: int,
    state_index_columns: int,
    key_heads: int,
    value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    sigmoid_gate: bool,
    qk_l2norm: bool,
    lower_bounded_kda: bool,
    has_null_state_index: bool,
    null_state_index: int,
    block_v: int,
    duplicate_table_size: int,
    recurrent_num_warps: int,
    norm_num_warps: int,
    validate_metadata: bool,
) -> None:
    if lower_bounded_kda:
        if key_heads != value_heads or not sigmoid_gate:
            raise RuntimeError(
                "GLM/KDA decode requires equal Q/K/V head counts and a "
                "sigmoid output gate"
            )
        token_capacity = int(output.shape[0])
        sequence_capacity = int(state_indices.shape[0])
        live_state_index_columns = int(state_indices.shape[1])
    elif value_heads != 3 * key_heads:
        raise RuntimeError(
            "Qwen GDN decode requires three value heads per key head, got "
            f"key_heads={key_heads}, value_heads={value_heads}"
        )
    else:
        token_capacity = int(max_tokens)
        sequence_capacity = int(max_seqs)
        live_state_index_columns = int(state_index_columns)

    if validate_metadata:
        _reset_validation_kernel[
            (triton.cdiv(duplicate_table_size, _VALIDATION_BLOCK),)
        ](
            duplicate_slots,
            error_code,
            TABLE_SIZE=int(duplicate_table_size),
            BLOCK=_VALIDATION_BLOCK,
            num_warps=1,
            num_stages=1,
        )
        _validate_packed_metadata_kernel[(sequence_capacity,)](
            query_start_loc,
            num_accepted_tokens,
            num_seqs,
            num_tokens,
            error_code,
            token_capacity,
            sequence_capacity,
            live_state_index_columns,
            num_warps=1,
            num_stages=1,
        )
        _validate_active_state_slots_kernel[
            (sequence_capacity * live_state_index_columns,)
        ](
            query_start_loc,
            num_accepted_tokens,
            state_indices,
            num_seqs,
            duplicate_slots,
            error_code,
            sequence_capacity,
            live_state_index_columns,
            stride_indices_request=int(state_indices.stride(0)),
            stride_indices_column=int(state_indices.stride(1)),
            MAX_STATE_SLOTS=int(max_state_slots),
            TABLE_SIZE=int(duplicate_table_size),
            HAS_NULL_STATE_INDEX=bool(has_null_state_index),
            NULL_STATE_INDEX=int(null_state_index),
            num_warps=1,
            num_stages=1,
        )
    if not lower_bounded_kda:
        binding = _make_qwen_binding(
            mixed_qkv,
            a,
            b,
            z,
            A_log,
            dt_bias,
            norm_weight,
            recurrent_state,
            query_start_loc,
            num_accepted_tokens,
            state_indices,
            num_seqs,
            num_tokens,
            output,
            duplicate_slots,
            error_code,
            max_tokens=max_tokens,
            max_seqs=max_seqs,
            max_state_slots=max_state_slots,
            state_index_columns=state_index_columns,
            key_heads=key_heads,
            value_heads=value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            sigmoid_gate=sigmoid_gate,
            qk_l2norm=qk_l2norm,
            null_state_index=(null_state_index if has_null_state_index else None),
            duplicate_table_size=duplicate_table_size,
        )
        from ._cute_kernels import run_packed_recurrent_qwen

        run_packed_recurrent_qwen(binding, scale=scale)
    else:
        value_tiles = triton.cdiv(value_head_dim, block_v)
        _packed_sequential_kda_decode_kernel[
            (value_tiles, sequence_capacity * value_heads)
        ](
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            recurrent_state,
            query_start_loc,
            num_accepted_tokens,
            state_indices,
            num_seqs,
            output,
            error_code,
            float(scale),
            float(lower_bound),
            sequence_capacity,
            live_state_index_columns,
            stride_mixed_token=int(mixed_qkv.stride(0)),
            stride_a_token=int(a.stride(0)),
            stride_a_head=int(a.stride(1)),
            stride_b_token=int(b.stride(0)),
            stride_b_head=int(b.stride(1)),
            stride_dt_bias_head=int(dt_bias.stride(0)),
            stride_state_slot=int(recurrent_state.stride(0)),
            stride_state_head=int(recurrent_state.stride(1)),
            stride_state_v=int(recurrent_state.stride(2)),
            stride_indices_request=int(state_indices.stride(0)),
            stride_indices_column=int(state_indices.stride(1)),
            stride_output_token=int(output.stride(0)),
            stride_output_head=int(output.stride(1)),
            MAX_SEQS=int(max_seqs),
            KEY_HEADS=int(key_heads),
            VALUE_HEADS=int(value_heads),
            KEY_HEAD_DIM=int(key_head_dim),
            VALUE_HEAD_DIM=int(value_head_dim),
            STATE_INDEX_COLUMNS=int(state_index_columns),
            BLOCK_V=int(block_v),
            QK_L2NORM=bool(qk_l2norm),
            HAS_NULL_STATE_INDEX=bool(has_null_state_index),
            NULL_STATE_INDEX=int(null_state_index),
            VALIDATE_METADATA=bool(validate_metadata),
            num_warps=int(recurrent_num_warps),
            num_stages=3,
        )
    _gated_rmsnorm_kernel[(token_capacity * value_heads,)](
        output,
        z,
        norm_weight,
        num_tokens,
        error_code,
        float(eps),
        token_capacity,
        stride_output_token=int(output.stride(0)),
        stride_output_head=int(output.stride(1)),
        stride_z_token=int(z.stride(0)),
        stride_z_head=int(z.stride(1)),
        VALUE_HEADS=int(value_heads),
        VALUE_HEAD_DIM=int(value_head_dim),
        SIGMOID_GATE=bool(sigmoid_gate),
        NORM_WEIGHT_FP32=norm_weight.dtype == torch.float32,
        KDA_NORM_FP32=bool(lower_bounded_kda),
        VALIDATE_METADATA=bool(validate_metadata),
        num_warps=int(norm_num_warps),
        num_stages=1,
    )


@torch.library.custom_op(
    "b12x::gdn_decode",
    mutates_args=(
        "recurrent_state",
        "output",
        "duplicate_slots",
        "error_code",
    ),
)
def _gdn_decode_op(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
    duplicate_slots: torch.Tensor,
    error_code: torch.Tensor,
    eps: float,
    scale: float,
    lower_bound: float,
    max_tokens: int,
    max_seqs: int,
    max_state_slots: int,
    state_index_columns: int,
    key_heads: int,
    value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    sigmoid_gate: bool,
    qk_l2norm: bool,
    lower_bounded_kda: bool,
    has_null_state_index: bool,
    null_state_index: int,
    block_v: int,
    duplicate_table_size: int,
    recurrent_num_warps: int,
    norm_num_warps: int,
    validate_metadata: bool,
) -> None:
    _launch_gdn_decode(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        recurrent_state,
        query_start_loc,
        num_accepted_tokens,
        state_indices,
        num_seqs,
        num_tokens,
        output,
        duplicate_slots,
        error_code,
        eps,
        scale,
        lower_bound,
        max_tokens,
        max_seqs,
        max_state_slots,
        state_index_columns,
        key_heads,
        value_heads,
        key_head_dim,
        value_head_dim,
        sigmoid_gate,
        qk_l2norm,
        lower_bounded_kda,
        has_null_state_index,
        null_state_index,
        block_v,
        duplicate_table_size,
        recurrent_num_warps,
        norm_num_warps,
        validate_metadata,
    )


@_gdn_decode_op.register_fake
def _gdn_decode_fake(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
    duplicate_slots: torch.Tensor,
    error_code: torch.Tensor,
    eps: float,
    scale: float,
    lower_bound: float,
    max_tokens: int,
    max_seqs: int,
    max_state_slots: int,
    state_index_columns: int,
    key_heads: int,
    value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    sigmoid_gate: bool,
    qk_l2norm: bool,
    lower_bounded_kda: bool,
    has_null_state_index: bool,
    null_state_index: int,
    block_v: int,
    duplicate_table_size: int,
    recurrent_num_warps: int,
    norm_num_warps: int,
    validate_metadata: bool,
) -> None:
    del mixed_qkv, a, b, z, A_log, dt_bias, norm_weight
    del recurrent_state, query_start_loc, num_accepted_tokens, state_indices
    del num_seqs, num_tokens, output, duplicate_slots, error_code
    del eps, scale, lower_bound, max_tokens, max_seqs, max_state_slots
    del state_index_columns
    del key_heads, value_heads, key_head_dim, value_head_dim, sigmoid_gate
    del qk_l2norm, lower_bounded_kda, has_null_state_index, null_state_index
    del block_v, duplicate_table_size
    del recurrent_num_warps, norm_num_warps, validate_metadata


def run_gdn_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
    duplicate_slots: torch.Tensor,
    error_code: torch.Tensor,
    *,
    eps: float,
    scale: float,
    max_tokens: int,
    max_seqs: int,
    max_state_slots: int,
    state_index_columns: int,
    key_heads: int,
    value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    gate_activation: str,
    decay_recipe: str,
    lower_bound: float,
    qk_l2norm: bool,
    null_state_index: int | None,
    block_v: int,
    duplicate_table_size: int,
    recurrent_num_warps: int,
    norm_num_warps: int,
    validate_metadata: bool = True,
) -> None:
    torch.ops.b12x.gdn_decode(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        recurrent_state,
        query_start_loc,
        num_accepted_tokens,
        state_indices,
        num_seqs,
        num_tokens,
        output,
        duplicate_slots,
        error_code,
        float(eps),
        float(scale),
        float(lower_bound),
        int(max_tokens),
        int(max_seqs),
        int(max_state_slots),
        int(state_index_columns),
        int(key_heads),
        int(value_heads),
        int(key_head_dim),
        int(value_head_dim),
        gate_activation == "sigmoid",
        bool(qk_l2norm),
        decay_recipe == "kda",
        null_state_index is not None,
        0 if null_state_index is None else int(null_state_index),
        int(block_v),
        int(duplicate_table_size),
        int(recurrent_num_warps),
        int(norm_num_warps),
        bool(validate_metadata),
    )


__all__ = ["run_gdn_decode"]
