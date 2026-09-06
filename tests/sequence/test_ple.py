from __future__ import annotations

import pytest
import torch

from b12x.sequence import ple, ple_hash
from b12x.sequence.ple.reference import (
    ple_projected_packed_reference,
    ple_projected_sequence_reference,
    ple_projected_u_reference,
)
from b12x.sequence.ple_hash.reference import (
    eos_bounded_windows,
    ple_hash_ids_reference,
    ple_hash_packed_reference,
    ple_multipliers,
    ple_table_geometry,
)

from ..conftest import require_b12x


def _scratch(plan) -> torch.Tensor:
    spec = plan.scratch_specs()[0]
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def test_ple_hash_geometry_is_distinct_deterministic_and_aligned() -> None:
    caps = ple_hash.Caps(
        device="cpu",
        max_tokens=8,
        max_seqs=3,
        vocab_size=1000,
        eos_token_id=999,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=100,
        table_alignment=128,
    )
    plan = ple_hash.plan(caps)

    assert plan.prime_sizes.tolist() == [101, 103, 107, 109]
    assert plan.table_offsets.tolist() == [0, 101, 204, 311]
    assert plan.padded_vocab_size == 512
    expected_multipliers = [
        5159850018220775,
        4902196785138501,
        6891410296393783,
    ]
    assert plan.multipliers.tolist() == expected_multipliers
    assert (
        ple_multipliers(
            vocab_size=1000,
            max_order=3,
            dense_layer_ordinal=0,
        ).tolist()
        == expected_multipliers
    )
    assert all(value & 1 for value in plan.multipliers.tolist())
    assert max(plan.multipliers.tolist()) <= ((1 << 63) - 1) // 1000

    next_sizes, _ = ple_table_geometry(
        base_size=100,
        dense_layer_ordinal=1,
        total_heads=4,
    )
    assert next_sizes.tolist() == [113, 127, 131, 137]
    minimum_sizes, minimum_offsets = ple_table_geometry(
        base_size=1,
        dense_layer_ordinal=0,
        total_heads=4,
    )
    assert minimum_sizes.tolist() == [2, 3, 5, 7]
    assert minimum_offsets.tolist() == [0, 2, 5, 10]


def test_ple_hash_plan_rejects_cumulative_table_extent_beyond_int64() -> None:
    caps = ple_hash.Caps(
        device="cpu",
        max_tokens=1,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=2,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )

    with pytest.raises(
        ValueError, match="cumulative table extent must fit signed int64"
    ):
        ple_hash.plan(
            caps,
            prime_sizes=torch.tensor([101, 9223372036854775783], dtype=torch.int64),
            table_offsets=torch.tensor([0, 101], dtype=torch.int64),
            multipliers=torch.tensor([1, 3], dtype=torch.int64),
        )


def test_ple_hash_plan_rejects_padded_table_extent_beyond_int64() -> None:
    caps = ple_hash.Caps(
        device="cpu",
        max_tokens=1,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=2,
        heads_per_order=1,
        dense_layer_ordinal=0,
        base_table_size=101,
        table_alignment=128,
    )

    with pytest.raises(ValueError, match="padded table extent must fit signed int64"):
        ple_hash.plan(
            caps,
            prime_sizes=torch.tensor([9223372036854775783], dtype=torch.int64),
            table_offsets=torch.tensor([0], dtype=torch.int64),
            multipliers=torch.tensor([1, 3], dtype=torch.int64),
        )


def test_ple_eos_bounded_windows_reset_left_context() -> None:
    windows = eos_bounded_windows(
        torch.tensor([4, 5, 99, 6], dtype=torch.int64),
        eos_token_id=99,
        max_order=3,
    )
    assert windows[2].tolist() == [[99, 4], [4, 5], [5, 99], [99, 6]]
    assert windows[3].tolist() == [
        [99, 99, 4],
        [99, 4, 5],
        [4, 5, 99],
        [99, 99, 6],
    ]


def test_ple_packed_hash_matches_per_request_complete_history() -> None:
    eos = 99
    token_ids = torch.tensor([9, eos, 10, 3, 4], dtype=torch.int64)
    starts = torch.tensor([0, 3, 5], dtype=torch.int32)
    history = torch.tensor([[7, 8], [eos, eos]], dtype=torch.int64)
    multipliers = torch.tensor([11, 13, 17], dtype=torch.int64)
    sizes = torch.tensor([101, 103, 107, 109], dtype=torch.int64)
    offsets = torch.tensor([0, 101, 204, 311], dtype=torch.int64)

    actual = ple_hash_packed_reference(
        token_ids,
        starts,
        history,
        eos_token_id=eos,
        multipliers=multipliers,
        prime_sizes=sizes,
        table_offsets=offsets,
        heads_per_order=2,
    )
    expected_parts = []
    for committed, query in ((history[0], token_ids[:3]), (history[1], token_ids[3:])):
        windows = eos_bounded_windows(
            torch.cat((committed, query)),
            eos_token_id=eos,
            max_order=3,
        )
        expected_parts.append(
            ple_hash_ids_reference(
                {order: rows[2:] for order, rows in windows.items()},
                multipliers=multipliers,
                prime_sizes=sizes,
                table_offsets=offsets,
                heads_per_order=2,
            )
        )
    torch.testing.assert_close(actual, torch.cat(expected_parts))


def test_ple_hash_plan_binds_fixed_capacity_and_fails_closed() -> None:
    caps = ple_hash.Caps(
        device="cpu",
        max_tokens=5,
        max_seqs=2,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    plan = ple_hash.plan(caps)
    binding = ple_hash.bind(
        plan,
        scratch=_scratch(plan),
        token_ids=torch.zeros(5, dtype=torch.int64),
        query_start_loc=torch.zeros(3, dtype=torch.int32),
        committed_history=torch.full((2, 2), 99, dtype=torch.int64),
        num_seqs=torch.tensor([0], dtype=torch.int32),
        num_tokens=torch.tensor([0], dtype=torch.int32),
        out=torch.empty((5, 4), dtype=torch.int64),
    )
    assert binding.scratch.untyped_storage().data_ptr() != 0
    assert not ple_hash.is_supported("cpu")
    with pytest.raises(ValueError, match="GPU run requires CUDA"):
        ple_hash.run(binding)


def _cpu_hash_bind_inputs(plan: ple_hash.Plan) -> dict[str, torch.Tensor]:
    caps = plan.caps
    return {
        "scratch": _scratch(plan),
        "token_ids": torch.zeros(caps.max_tokens, dtype=torch.int64),
        "query_start_loc": torch.zeros(caps.max_seqs + 1, dtype=torch.int32),
        "committed_history": torch.full(
            (caps.max_seqs, caps.max_order - 1),
            caps.eos_token_id,
            dtype=torch.int64,
        ),
        "num_seqs": torch.tensor([0], dtype=torch.int32),
        "num_tokens": torch.tensor([0], dtype=torch.int32),
        "out": torch.empty((caps.max_tokens, caps.head_count), dtype=torch.int64),
    }


def test_ple_hash_bind_rejects_output_aliasing_read_only_input() -> None:
    caps = ple_hash.Caps(
        device="cpu",
        max_tokens=2,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    plan = ple_hash.plan(caps)
    inputs = _cpu_hash_bind_inputs(plan)
    shared = torch.empty(caps.max_tokens * caps.head_count, dtype=torch.int64)
    inputs["token_ids"] = shared[: caps.max_tokens]
    inputs["out"] = shared.view(caps.max_tokens, caps.head_count)

    with pytest.raises(ValueError, match="out.*read-only tensor token_ids"):
        ple_hash.bind(plan, **inputs)


def test_ple_hash_bind_rejects_scratch_aliasing_read_only_input() -> None:
    caps = ple_hash.Caps(
        device="cpu",
        max_tokens=2,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    plan = ple_hash.plan(caps)
    inputs = _cpu_hash_bind_inputs(plan)
    shared = torch.empty(plan.scratch_specs()[0].shape, dtype=torch.uint8)
    inputs["scratch"] = shared
    inputs["token_ids"] = shared[: caps.max_tokens * 8].view(torch.int64)

    with pytest.raises(ValueError, match="scratch.*read-only tensor token_ids"):
        ple_hash.bind(plan, **inputs)


def test_ple_hash_bind_rejects_output_aliasing_plan_geometry() -> None:
    caps = ple_hash.Caps(
        device="cpu",
        max_tokens=1,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    shared = torch.tensor([101, 103, 107, 109], dtype=torch.int64)
    plan = ple_hash.plan(
        caps,
        prime_sizes=shared,
        table_offsets=torch.tensor([0, 101, 204, 311], dtype=torch.int64),
        multipliers=torch.tensor([11, 13, 17], dtype=torch.int64),
    )
    inputs = _cpu_hash_bind_inputs(plan)
    inputs["out"] = shared.view(1, caps.head_count)

    with pytest.raises(ValueError, match="out.*read-only tensor prime_sizes"):
        ple_hash.bind(plan, **inputs)


def _projected_inputs(tokens: int, streams: int, hidden: int):
    generator = torch.Generator().manual_seed(4107)
    residual = torch.randn(tokens, streams, hidden, generator=generator).to(
        torch.bfloat16
    )
    key = torch.randn(tokens, streams, hidden, generator=generator).to(torch.bfloat16)
    value = torch.randn(tokens, hidden, generator=generator).to(torch.bfloat16)
    weights = [
        (torch.randn(streams, hidden, generator=generator) / 10).to(torch.bfloat16)
        for _ in range(3)
    ]
    return residual, key, value, weights


def _cuda_projected_inputs(
    tokens: int,
    streams: int,
    hidden: int,
    *,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    residual = torch.randn(
        (tokens, streams, hidden),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(residual)
    value = torch.randn(
        (tokens, hidden),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    weights = [
        (
            torch.randn(
                (streams * hidden,),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            / 32
        ).contiguous()
        for _ in range(3)
    ]
    return residual, key, value, weights, generator


def _bind_cuda_layer(
    *,
    mode: str,
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weights: list[torch.Tensor],
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_seqs: int,
    num_tokens: int,
    conv_state: torch.Tensor,
    max_speculative_tokens: int,
    dilation: int,
    request_is_prefill: torch.Tensor | None = None,
):
    max_tokens, streams, hidden = residual.shape
    max_seqs = int(state_slot_ids.numel())
    plan = ple.plan(
        ple.Caps(
            device=residual.device,
            mode=mode,
            max_tokens=max_tokens,
            max_seqs=max_seqs,
            max_state_slots=conv_state.shape[0],
            max_speculative_tokens=max_speculative_tokens,
            streams=streams,
            hidden_size=hidden,
            kernel_size=conv_weight.shape[-1],
            dilation=dilation,
        )
    )
    out = torch.full_like(residual, 91)
    binding = ple.bind(
        plan,
        scratch=_scratch(plan),
        residual=residual,
        key=key,
        value=value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=torch.tensor([num_seqs], dtype=torch.int32, device=residual.device),
        num_tokens=torch.tensor(
            [num_tokens], dtype=torch.int32, device=residual.device
        ),
        conv_state=conv_state,
        out=out,
        request_is_prefill=request_is_prefill,
    )
    return plan, binding


def _mixed_layer_reference(
    *,
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weights: list[torch.Tensor],
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    request_is_prefill: torch.Tensor,
    num_seqs: int,
    num_tokens: int,
    conv_state: torch.Tensor,
    dilation: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_out = torch.zeros_like(residual)
    expected_state = conv_state.clone()
    starts = [
        int(value) for value in query_start_loc[: num_seqs + 1].detach().cpu().tolist()
    ]
    slots = [int(value) for value in state_slot_ids[:num_seqs].cpu().tolist()]
    fresh = [bool(value) for value in state_is_fresh[:num_seqs].cpu().tolist()]
    accepted = [int(value) for value in num_accepted_tokens[:num_seqs].cpu().tolist()]
    prefill = [bool(value) for value in request_is_prefill[:num_seqs].cpu().tolist()]
    state_length = dilation * (int(conv_weight.shape[-1]) - 1)
    max_speculative = int(conv_state.shape[-1]) - state_length

    for request in range(num_seqs):
        start, end = starts[request : request + 2]
        if start == end or slots[request] < 0:
            continue
        if fresh[request]:
            prior = torch.zeros_like(conv_state[slots[request], :, :state_length])
        else:
            rollback = 0 if prefill[request] else accepted[request] - 1
            prior = conv_state[slots[request], :, rollback : rollback + state_length]
        contribution, newest = ple_projected_sequence_reference(
            residual[start:end],
            key[start:end],
            value[start:end],
            k_norm_weight=weights[0],
            q_norm_weight=weights[1],
            u_norm_weight=weights[2],
            conv_weight=conv_weight,
            eps=eps,
            dilation=dilation,
            prior_state=prior,
        )
        expected_out[start:end].copy_(contribution)
        if prefill[request]:
            committed = newest
        else:
            _, committed = ple_projected_sequence_reference(
                residual[start : start + 1],
                key[start : start + 1],
                value[start : start + 1],
                k_norm_weight=weights[0],
                q_norm_weight=weights[1],
                u_norm_weight=weights[2],
                conv_weight=conv_weight,
                eps=eps,
                dilation=dilation,
                prior_state=prior,
            )
        state = expected_state[slots[request]]
        state[:, :state_length].copy_(committed)
        state[:, state_length:].zero_()
        if not prefill[request] and end - start > 1:
            _, normalized_u = ple_projected_u_reference(
                residual[start:end],
                key[start:end],
                value[start:end],
                k_norm_weight=weights[0],
                q_norm_weight=weights[1],
                u_norm_weight=weights[2],
                eps=eps,
            )
            candidates = min(end - start - 1, max_speculative)
            state[:, state_length : state_length + candidates].copy_(
                normalized_u[1 : candidates + 1].transpose(0, 1)
            )

    expected_out[num_tokens:].zero_()
    return expected_out, expected_state


def test_ple_stateful_decode_chunks_match_full_prefill_oracle() -> None:
    tokens, streams, hidden = 7, 2, 4
    kernel_size, dilation = 3, 2
    channels = streams * hidden
    state_length = dilation * (kernel_size - 1)
    residual, key, value, weights = _projected_inputs(tokens, streams, hidden)
    generator = torch.Generator().manual_seed(919)
    conv_weight = (torch.randn(channels, kernel_size, generator=generator) / 8).to(
        torch.bfloat16
    )
    zeros = torch.zeros(channels, state_length, dtype=torch.bfloat16)

    expected, expected_state = ple_projected_sequence_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=zeros,
    )
    first, state = ple_projected_sequence_reference(
        residual[:2],
        key[:2],
        value[:2],
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=zeros,
    )
    second, state = ple_projected_sequence_reference(
        residual[2:],
        key[2:],
        value[2:],
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=state,
    )
    torch.testing.assert_close(torch.cat((first, second)), expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_ple_packed_oracle_is_request_local() -> None:
    streams, hidden = 2, 4
    residual, key, value, weights = _projected_inputs(5, streams, hidden)
    generator = torch.Generator().manual_seed(721)
    conv_weight = (torch.randn(8, 3, generator=generator) / 8).to(torch.bfloat16)
    starts = torch.tensor([0, 3, 5], dtype=torch.int32)

    packed, states = ple_projected_packed_reference(
        residual,
        key,
        value,
        starts,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=2,
    )
    for request, (start, end) in enumerate(((0, 3), (3, 5))):
        expected, expected_state = ple_projected_sequence_reference(
            residual[start:end],
            key[start:end],
            value[start:end],
            k_norm_weight=weights[0],
            q_norm_weight=weights[1],
            u_norm_weight=weights[2],
            conv_weight=conv_weight,
            eps=1e-6,
            dilation=2,
        )
        torch.testing.assert_close(packed[start:end], expected, rtol=0, atol=0)
        torch.testing.assert_close(states[request], expected_state, rtol=0, atol=0)


def test_ple_layer_plan_exposes_state_capacity_and_fails_closed() -> None:
    caps = ple.Caps(
        device="cpu",
        mode="decode",
        max_tokens=2,
        max_seqs=2,
        max_state_slots=3,
        max_speculative_tokens=4,
        streams=2,
        hidden_size=4,
        kernel_size=4,
        dilation=3,
    )
    plan = ple.plan(caps)
    assert plan.state_length == 9
    assert plan.state_capacity == 13
    common = torch.empty((2, 2, 4), dtype=torch.bfloat16)
    binding = ple.bind(
        plan,
        scratch=_scratch(plan),
        residual=common,
        key=torch.empty_like(common),
        value=torch.empty((2, 4), dtype=torch.bfloat16),
        k_norm_weight=torch.empty((8,), dtype=torch.bfloat16),
        q_norm_weight=torch.empty((8,), dtype=torch.bfloat16),
        u_norm_weight=torch.empty((8,), dtype=torch.bfloat16),
        conv_weight=torch.empty((8, 4), dtype=torch.bfloat16),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        state_slot_ids=torch.tensor([0, -1], dtype=torch.int64),
        state_is_fresh=torch.tensor([True, False], dtype=torch.bool),
        num_accepted_tokens=torch.tensor([1, 0], dtype=torch.int32),
        num_seqs=torch.tensor([2], dtype=torch.int32),
        num_tokens=torch.tensor([2], dtype=torch.int32),
        conv_state=torch.empty((3, 8, 13), dtype=torch.bfloat16),
        out=torch.empty_like(common),
    )
    with pytest.raises(ValueError, match="GPU run requires CUDA"):
        ple.run_decode(binding, eps=1e-6)
    with pytest.raises(ValueError, match="prefill LayerPlan"):
        ple.run_prefill(binding, eps=1e-6)


def _cpu_layer_plan(mode: str = "decode") -> ple.Plan:
    return ple.plan(
        ple.Caps(
            device="cpu",
            mode=mode,
            max_tokens=2,
            max_seqs=2,
            max_state_slots=3,
            max_speculative_tokens=4,
            streams=2,
            hidden_size=4,
            kernel_size=4,
            dilation=3,
        )
    )


def _cpu_layer_bind_inputs(plan: ple.Plan) -> dict[str, torch.Tensor]:
    caps = plan.caps
    inputs = {
        "scratch": _scratch(plan),
        "residual": torch.empty(
            (caps.max_tokens, caps.streams, caps.hidden_size), dtype=caps.dtype
        ),
        "key": torch.empty(
            (caps.max_tokens, caps.streams, caps.hidden_size), dtype=caps.dtype
        ),
        "value": torch.empty((caps.max_tokens, caps.hidden_size), dtype=caps.dtype),
        "k_norm_weight": torch.empty(caps.channels, dtype=caps.dtype),
        "q_norm_weight": torch.empty(caps.channels, dtype=caps.dtype),
        "u_norm_weight": torch.empty(caps.channels, dtype=caps.dtype),
        "conv_weight": torch.empty((caps.channels, caps.kernel_size), dtype=caps.dtype),
        "query_start_loc": torch.tensor([0, 1, 2], dtype=torch.int32),
        "state_slot_ids": torch.tensor([0, 1], dtype=torch.int64),
        "state_is_fresh": torch.tensor([False, False], dtype=torch.bool),
        "num_accepted_tokens": torch.tensor([1, 1], dtype=torch.int32),
        "num_seqs": torch.tensor([2], dtype=torch.int32),
        "num_tokens": torch.tensor([2], dtype=torch.int32),
        "conv_state": torch.empty(
            (caps.max_state_slots, caps.channels, caps.state_capacity),
            dtype=caps.dtype,
        ),
        "out": torch.empty(
            (caps.max_tokens, caps.streams, caps.hidden_size), dtype=caps.dtype
        ),
    }
    if caps.mode == "mixed":
        inputs["request_is_prefill"] = torch.tensor([True, False])
    return inputs


def test_ple_mixed_plan_requires_fixed_request_modes() -> None:
    plan = _cpu_layer_plan("mixed")
    inputs = _cpu_layer_bind_inputs(plan)
    request_is_prefill = inputs.pop("request_is_prefill")
    with pytest.raises(ValueError, match="required for a mixed"):
        ple.bind(plan, **inputs)

    inputs["request_is_prefill"] = request_is_prefill.to(torch.int32)
    with pytest.raises(TypeError, match="request_is_prefill.*torch.bool"):
        ple.bind(plan, **inputs)

    inputs["request_is_prefill"] = request_is_prefill[:1]
    with pytest.raises(ValueError, match="request_is_prefill.*shape"):
        ple.bind(plan, **inputs)

    inputs["request_is_prefill"] = request_is_prefill
    binding = ple.bind(plan, **inputs)
    with pytest.raises(ValueError, match="GPU run requires CUDA"):
        ple.run_mixed(binding, eps=1e-6)
    with pytest.raises(ValueError, match="decode LayerPlan"):
        ple.run_decode(binding, eps=1e-6)


def test_ple_homogeneous_plan_rejects_request_modes() -> None:
    plan = _cpu_layer_plan()
    inputs = _cpu_layer_bind_inputs(plan)
    inputs["request_is_prefill"] = torch.tensor([True, False])

    with pytest.raises(ValueError, match="only valid for a mixed"):
        ple.bind(plan, **inputs)


def test_ple_mixed_request_modes_must_not_alias_mutable_scratch() -> None:
    plan = _cpu_layer_plan("mixed")
    inputs = _cpu_layer_bind_inputs(plan)
    shared = _scratch(plan)
    inputs["scratch"] = shared
    inputs["request_is_prefill"] = shared[: plan.caps.max_seqs].view(torch.bool)

    with pytest.raises(ValueError, match="scratch.*request_is_prefill"):
        ple.bind(plan, **inputs)


@pytest.mark.parametrize("alias_kind", ["out_input", "scratch_input", "state_output"])
def test_ple_bind_rejects_mutation_schema_aliases(alias_kind: str) -> None:
    plan = _cpu_layer_plan()
    caps = plan.caps
    inputs = _cpu_layer_bind_inputs(plan)
    if alias_kind == "out_input":
        inputs["out"] = inputs["residual"]
    elif alias_kind == "scratch_input":
        shared = torch.empty(plan.scratch_specs()[0].shape, dtype=torch.uint8)
        inputs["scratch"] = shared
        inputs["residual"] = (
            shared[: caps.max_tokens * caps.channels * 2]
            .view(torch.bfloat16)
            .view(caps.max_tokens, caps.streams, caps.hidden_size)
        )
    else:
        state_elements = caps.max_state_slots * caps.channels * caps.state_capacity
        shared = torch.empty(state_elements, dtype=caps.dtype)
        inputs["conv_state"] = shared.view(
            caps.max_state_slots, caps.channels, caps.state_capacity
        )
        inputs["out"] = shared[: caps.max_tokens * caps.channels].view(
            caps.max_tokens, caps.streams, caps.hidden_size
        )

    with pytest.raises(ValueError, match="must not overlap"):
        ple.bind(plan, **inputs)


def test_ple_bind_allows_read_only_aliases() -> None:
    plan = _cpu_layer_plan()
    inputs = _cpu_layer_bind_inputs(plan)
    inputs["key"] = inputs["residual"]

    binding = ple.bind(plan, **inputs)

    assert binding.key is binding.residual


def test_ple_bind_accepts_padded_state_slot_stride_without_copy() -> None:
    plan = _cpu_layer_plan()
    inputs = _cpu_layer_bind_inputs(plan)
    caps = plan.caps
    slot_elements = caps.channels * caps.state_capacity
    padded_slot_elements = slot_elements + 17
    storage = torch.empty(
        (caps.max_state_slots - 1) * padded_slot_elements + slot_elements,
        dtype=caps.dtype,
    )
    conv_state = torch.as_strided(
        storage,
        (caps.max_state_slots, caps.channels, caps.state_capacity),
        (padded_slot_elements, caps.state_capacity, 1),
    )
    inputs["conv_state"] = conv_state

    binding = ple.bind(plan, **inputs)

    assert binding.conv_state is conv_state
    assert binding.conv_state.stride(0) == padded_slot_elements
    assert not binding.conv_state.is_contiguous()


@pytest.mark.parametrize(
    "strides",
    [
        (8 * 13, 1, 8),
        (8 * 13 - 1, 13, 1),
    ],
)
def test_ple_bind_rejects_unsupported_or_overlapping_state_strides(
    strides: tuple[int, int, int],
) -> None:
    plan = _cpu_layer_plan()
    inputs = _cpu_layer_bind_inputs(plan)
    caps = plan.caps
    storage = torch.empty(4096, dtype=caps.dtype)
    inputs["conv_state"] = torch.as_strided(
        storage,
        (caps.max_state_slots, caps.channels, caps.state_capacity),
        strides,
    )

    with pytest.raises(ValueError, match="conv_state"):
        ple.bind(plan, **inputs)


@torch.inference_mode()
def test_ple_padded_state_stride_reaches_past_int32_offset_boundary() -> None:
    device = require_b12x()
    tokens, streams, hidden = 1, 1, 16
    dilation, max_speculative = 1, 0
    residual, key, value, weights, _ = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1401
    )
    conv_weight = torch.tensor(
        [[0.25, 0.5]] * hidden,
        dtype=torch.bfloat16,
        device=device,
    )

    # A representative 818176-byte aligned hybrid-cache page has this BF16
    # stride. The live tail slot starts beyond the signed-32-bit element range.
    state_slot_stride = 818176 // torch.tensor([], dtype=torch.bfloat16).element_size()
    tail_slot = (2**31) // state_slot_stride + 1
    state_elements = tail_slot * state_slot_stride + hidden
    state_storage = torch.empty(state_elements, dtype=torch.bfloat16, device=device)
    conv_state = torch.as_strided(
        state_storage,
        (tail_slot + 1, hidden, 1),
        (state_slot_stride, 1, 1),
    )
    assert tail_slot * conv_state.stride(0) > 2**31

    generator = torch.Generator(device=device).manual_seed(1402)
    prior = torch.randn(
        (hidden, 1), generator=generator, dtype=torch.bfloat16, device=device
    )
    conv_state[tail_slot].copy_(prior)
    conv_state[0].fill_(7)
    low_slot_before = conv_state[0].clone()
    expected, expected_state = ple_projected_sequence_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=prior,
    )
    _, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([tail_slot], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() == 0
    torch.testing.assert_close(binding.out, expected, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(
        conv_state[tail_slot], expected_state, rtol=0.02, atol=0.0078125
    )
    torch.testing.assert_close(conv_state[0], low_slot_before, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_hash_target_shape_matches_oracle_and_pads_output() -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=7,
        max_seqs=3,
        vocab_size=248320,
        eos_token_id=248044,
        max_order=3,
        heads_per_order=8,
        dense_layer_ordinal=0,
        base_table_size=20000000,
    )
    plan = ple_hash.plan(caps)
    token_ids = torch.tensor(
        [7, caps.eos_token_id, 8, 9, 10, 0, 0],
        dtype=torch.int64,
        device=device,
    )
    query_start_loc = torch.tensor([0, 3, 5, 5], dtype=torch.int32, device=device)
    committed_history = torch.tensor(
        [
            [3, 4],
            [5, caps.eos_token_id],
            [caps.eos_token_id, caps.eos_token_id],
        ],
        dtype=torch.int64,
        device=device,
    )
    out = torch.empty(
        (caps.max_tokens, caps.head_count), dtype=torch.int64, device=device
    )
    binding = ple_hash.bind(
        plan,
        scratch=_scratch(plan),
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=torch.tensor([2], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([5], dtype=torch.int32, device=device),
        out=out,
    )

    ple_hash.run(binding)
    torch.cuda.synchronize()
    expected = ple_hash_packed_reference(
        token_ids[:5],
        query_start_loc[:3],
        committed_history[:2],
        eos_token_id=caps.eos_token_id,
        multipliers=plan.multipliers,
        prime_sizes=plan.prime_sizes,
        table_offsets=plan.table_offsets,
        heads_per_order=caps.heads_per_order,
    )

    assert binding.error_code.item() == 0
    torch.testing.assert_close(out[:5], expected, rtol=0, atol=0)
    assert bool((out[5:] == -1).all().item())


@pytest.mark.parametrize(
    ("invalid_field", "error_mask"),
    [("query_start", 2), ("token_id", 4), ("history", 4)],
)
@torch.inference_mode()
def test_ple_hash_invalid_metadata_fails_closed(
    invalid_field: str, error_mask: int
) -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=2,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    plan = ple_hash.plan(caps)
    token_ids = torch.tensor([1, 0], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    committed_history = torch.tensor([[99, 99]], dtype=torch.int64, device=device)
    if invalid_field == "query_start":
        query_start_loc.copy_(torch.tensor([1, 1], dtype=torch.int32, device=device))
    elif invalid_field == "token_id":
        token_ids[0] = caps.vocab_size
    else:
        committed_history[0, 0] = -1
    history_before = committed_history.clone()
    binding = ple_hash.bind(
        plan,
        scratch=_scratch(plan),
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        out=torch.full(
            (caps.max_tokens, caps.head_count),
            91,
            dtype=torch.int64,
            device=device,
        ),
    )

    ple_hash.run(binding)
    torch.cuda.synchronize()

    assert binding.error_code.item() & error_mask
    assert bool((binding.out == -1).all().item())
    torch.testing.assert_close(committed_history, history_before, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_hash_zero_sequences_with_tokens_fails_closed() -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=2,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    plan = ple_hash.plan(caps)
    binding = ple_hash.bind(
        plan,
        scratch=_scratch(plan),
        token_ids=torch.tensor([1, 0], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        committed_history=torch.tensor([[99, 99]], dtype=torch.int64, device=device),
        num_seqs=torch.tensor([0], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        out=torch.full(
            (caps.max_tokens, caps.head_count),
            91,
            dtype=torch.int64,
            device=device,
        ),
    )

    ple_hash.run(binding)
    torch.cuda.synchronize()

    assert binding.error_code.item() & 1
    assert bool((binding.out == -1).all().item())


@torch.inference_mode()
def test_ple_hash_cuda_graph_replay_is_allocation_free() -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=2,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    plan = ple_hash.plan(caps)
    token_ids = torch.tensor([1, 2], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=device)
    committed_history = torch.tensor([[99, 99]], dtype=torch.int64, device=device)
    binding = ple_hash.bind(
        plan,
        scratch=_scratch(plan),
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([2], dtype=torch.int32, device=device),
        out=torch.empty(
            (caps.max_tokens, caps.head_count), dtype=torch.int64, device=device
        ),
    )
    ple_hash.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple_hash.run(binding)
    output_address = captured_out.data_ptr()

    token_ids.copy_(torch.tensor([3, 4], dtype=torch.int64, device=device))
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated(device)
    expected = ple_hash_packed_reference(
        token_ids,
        query_start_loc,
        committed_history,
        eos_token_id=caps.eos_token_id,
        multipliers=plan.multipliers,
        prime_sizes=plan.prime_sizes,
        table_offsets=plan.table_offsets,
        heads_per_order=caps.heads_per_order,
    )

    assert binding.error_code.item() == 0
    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_out, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_hash_target_cuda_graph_replays_dynamic_packed_metadata() -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=8,
        max_seqs=3,
        vocab_size=248320,
        eos_token_id=248044,
        max_order=3,
        heads_per_order=8,
        dense_layer_ordinal=0,
        base_table_size=20000000,
    )
    plan = ple_hash.plan(caps)
    token_ids = torch.tensor([1, 2, 0, 0, 0, 0, 0, 0], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 2, 2, 2], dtype=torch.int32, device=device)
    committed_history = torch.full(
        (caps.max_seqs, caps.max_order - 1),
        caps.eos_token_id,
        dtype=torch.int64,
        device=device,
    )
    num_seqs = torch.tensor([1], dtype=torch.int32, device=device)
    num_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    binding = ple_hash.bind(
        plan,
        scratch=_scratch(plan),
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        out=torch.empty(
            (caps.max_tokens, caps.head_count), dtype=torch.int64, device=device
        ),
    )

    ple_hash.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple_hash.run(binding)
    output_address = captured_out.data_ptr()

    token_ids.copy_(
        torch.tensor(
            [7, caps.eos_token_id, 8, 9, 10, 11, 0, 0],
            dtype=torch.int64,
            device=device,
        )
    )
    query_start_loc.copy_(torch.tensor([0, 3, 4, 6], dtype=torch.int32, device=device))
    committed_history.copy_(
        torch.tensor(
            [
                [3, 4],
                [5, caps.eos_token_id],
                [6, 7],
            ],
            dtype=torch.int64,
            device=device,
        )
    )
    num_seqs.fill_(3)
    num_tokens.fill_(6)
    expected = ple_hash_packed_reference(
        token_ids[:6],
        query_start_loc[:4],
        committed_history[:3],
        eos_token_id=caps.eos_token_id,
        multipliers=plan.multipliers,
        prime_sizes=plan.prime_sizes,
        table_offsets=plan.table_offsets,
        heads_per_order=caps.heads_per_order,
    )
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert binding.error_code.item() == 0
    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_out[:6], expected, rtol=0, atol=0)
    assert bool((captured_out[6:] == -1).all().item())


@torch.inference_mode()
def test_ple_target_shape_prefill_matches_oracle_and_recycles_dirty_slots() -> None:
    device = require_b12x()
    tokens, streams, hidden = 3, 4, 2560
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=991
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.full(
        (3, streams * hidden, state_length + max_speculative),
        7,
        dtype=torch.bfloat16,
        device=device,
    )
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    plan, binding = _bind_cuda_layer(
        mode="prefill",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=torch.tensor([0, 1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([True, True], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.zeros(2, dtype=torch.int32, device=device),
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_prefill(binding, eps=1e-6)
    torch.cuda.synchronize()
    expected, expected_state = ple_projected_packed_reference(
        residual,
        key,
        value,
        query_start_loc,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
    )

    assert binding.error_code.item() == 0
    torch.testing.assert_close(binding.out, expected, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(
        conv_state[:2, :, : plan.state_length],
        expected_state,
        rtol=0.02,
        atol=0.0078125,
    )
    assert bool((conv_state[:2, :, plan.state_length :] == 0).all().item())
    assert bool((conv_state[2] == 7).all().item())


@torch.inference_mode()
def test_ple_mixed_packed_order_matches_request_local_oracles() -> None:
    device = require_b12x()
    max_tokens, num_tokens, streams, hidden = 16, 12, 2, 64
    num_seqs = 4
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        max_tokens, streams, hidden, device=device, seed=1210
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (4, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_start_loc = torch.tensor(
        [0, 6, 9, 10, 12, 12], dtype=torch.int32, device=device
    )
    state_slot_ids = torch.tensor([2, 0, -1, 1, -1], device=device)
    state_is_fresh = torch.tensor(
        [True, False, False, False, False], dtype=torch.bool, device=device
    )
    num_accepted_tokens = torch.tensor(
        [0, 3, 1, 99, -7], dtype=torch.int32, device=device
    )
    request_is_prefill = torch.tensor(
        [True, False, False, True, False], dtype=torch.bool, device=device
    )
    expected_out, expected_state = _mixed_layer_reference(
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        request_is_prefill=request_is_prefill,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        conv_state=conv_state,
        dilation=dilation,
        eps=1e-6,
    )
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=request_is_prefill,
    )

    ple.run_mixed(binding, eps=1e-6)
    torch.cuda.synchronize()

    assert binding.error_code.item() == 0
    torch.testing.assert_close(binding.out, expected_out, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(conv_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("accepted", [1, 3, 5])
@torch.inference_mode()
def test_ple_decode_rolls_back_and_retains_candidates(accepted: int) -> None:
    device = require_b12x()
    tokens, streams, hidden = 5, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1200
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    state_capacity = state_length + max_speculative
    prior = torch.randn(
        (streams * hidden, state_capacity),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state = torch.full(
        (2, streams * hidden, state_capacity),
        11,
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state[0].copy_(prior)
    untouched = conv_state[1].clone()
    plan, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor(
            [0, tokens, tokens], dtype=torch.int32, device=device
        ),
        state_slot_ids=torch.tensor([0, -1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False, False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor(
            [accepted, 0], dtype=torch.int32, device=device
        ),
        num_seqs=1,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    rollback = accepted - 1
    effective_history = prior[:, rollback : rollback + state_length].contiguous()
    expected, _ = ple_projected_sequence_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=effective_history,
    )
    _, normalized_u = ple_projected_u_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        eps=1e-6,
    )
    expected_base = torch.cat(
        (effective_history[:, 1:], normalized_u[0].unsqueeze(1)), dim=1
    )
    expected_tail = torch.zeros(
        (streams * hidden, max_speculative),
        dtype=torch.bfloat16,
        device=device,
    )
    expected_tail[:, : tokens - 1].copy_(normalized_u[1:].transpose(0, 1))

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()

    assert binding.error_code.item() == 0
    torch.testing.assert_close(binding.out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        conv_state[0, :, : plan.state_length], expected_base, rtol=0, atol=0
    )
    torch.testing.assert_close(
        conv_state[0, :, plan.state_length :], expected_tail, rtol=0, atol=0
    )
    torch.testing.assert_close(conv_state[1], untouched, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_decode_fresh_flag_ignores_dirty_recycled_state() -> None:
    device = require_b12x()
    tokens, streams, hidden = 2, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1201
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.full(
        (1, streams * hidden, state_length + max_speculative),
        7,
        dtype=torch.bfloat16,
        device=device,
    )
    plan, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([0], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([True], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([3], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )
    zero_history = torch.zeros(
        (streams * hidden, state_length), dtype=torch.bfloat16, device=device
    )
    expected, _ = ple_projected_sequence_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=zero_history,
    )
    _, normalized_u = ple_projected_u_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        eps=1e-6,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()

    expected_base = torch.cat(
        (zero_history[:, 1:], normalized_u[0].unsqueeze(1)), dim=1
    )
    assert binding.error_code.item() == 0
    torch.testing.assert_close(binding.out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        conv_state[0, :, : plan.state_length], expected_base, rtol=0, atol=0
    )
    torch.testing.assert_close(
        conv_state[0, :, plan.state_length], normalized_u[1], rtol=0, atol=0
    )
    assert bool((conv_state[0, :, plan.state_length + 1 :] == 0).all().item())


@pytest.mark.parametrize(
    ("query_length", "accepted", "error_mask"),
    [(6, 1, 4), (1, 0, 8), (1, 6, 8)],
)
@torch.inference_mode()
def test_ple_decode_invalid_speculative_metadata_fails_without_mutation(
    query_length: int,
    accepted: int,
    error_mask: int,
) -> None:
    device = require_b12x()
    max_tokens, streams, hidden = 6, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        max_tokens, streams, hidden, device=device, seed=1202
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (1, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    state_before = conv_state.clone()
    _, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor(
            [0, query_length], dtype=torch.int32, device=device
        ),
        state_slot_ids=torch.tensor([0], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=query_length,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()

    assert binding.error_code.item() & error_mask
    assert bool((binding.out == 0).all().item())
    torch.testing.assert_close(conv_state, state_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("query_start_loc", "num_accepted_tokens", "error_mask", "token_count"),
    [
        ([0, 1, 6], [99, 1], 4, None),
        ([0, 1, 2], [0, 0], 8, None),
        ([0, 1, 2], [99, 6], 8, None),
        ([0, 1, 2], [99, 1], 1, 1),
        ([0, 1, 2], [99, 1], 1, 0),
    ],
)
@torch.inference_mode()
def test_ple_mixed_invalid_decode_metadata_fails_without_mutation(
    query_start_loc: list[int],
    num_accepted_tokens: list[int],
    error_mask: int,
    token_count: int | None,
) -> None:
    device = require_b12x()
    max_tokens, streams, hidden = 6, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 3
    residual, key, value, weights, generator = _cuda_projected_inputs(
        max_tokens, streams, hidden, device=device, seed=1211
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (2, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    state_before = conv_state.clone()
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([0, 1], dtype=torch.int64, device=device),
        state_is_fresh=torch.zeros(2, dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor(
            num_accepted_tokens, dtype=torch.int32, device=device
        ),
        num_seqs=2,
        num_tokens=query_start_loc[-1],
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=torch.tensor([True, False], dtype=torch.bool, device=device),
    )

    out = ple.run_mixed(binding, eps=1e-6, token_count=token_count)
    torch.cuda.synchronize()

    assert binding.error_code.item() & error_mask
    assert bool((out == 0).all().item())
    assert bool((binding.out[out.shape[0] :] == 91).all().item())
    torch.testing.assert_close(conv_state, state_before, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_mixed_duplicate_live_state_slots_fail_without_mutation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 2, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1212
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (1, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    state_before = conv_state.clone()
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([0, 0], dtype=torch.int64, device=device),
        state_is_fresh=torch.zeros(2, dtype=torch.bool, device=device),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device=device),
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=torch.tensor([True, False], dtype=torch.bool, device=device),
    )

    ple.run_mixed(binding, eps=1e-6)
    torch.cuda.synchronize()

    assert binding.error_code.item() & 32
    assert bool((binding.out == 0).all().item())
    torch.testing.assert_close(conv_state, state_before, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_duplicate_live_state_slots_fail_without_mutation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 2, 2, 32
    max_seqs = 130
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1206
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (1, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    state_before = conv_state.clone()
    query_start_loc = torch.ones(max_seqs + 1, dtype=torch.int32, device=device)
    query_start_loc[0] = 0
    query_start_loc[-1] = tokens
    state_slot_ids = torch.full((max_seqs,), -1, dtype=torch.int64, device=device)
    state_slot_ids[0] = 0
    state_slot_ids[-1] = 0
    _, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=torch.zeros(max_seqs, dtype=torch.bool, device=device),
        num_accepted_tokens=torch.ones(max_seqs, dtype=torch.int32, device=device),
        num_seqs=max_seqs,
        num_tokens=2,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()

    assert binding.error_code.item() & 32
    assert bool((binding.out == 0).all().item())
    torch.testing.assert_close(conv_state, state_before, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@torch.inference_mode()
def test_ple_zero_token_live_request_preserves_entire_state(mode: str) -> None:
    device = require_b12x()
    tokens, streams, hidden = 1, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1207
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (2, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    empty_state_before = conv_state[0].clone()
    _, binding = _bind_cuda_layer(
        mode=mode,
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([0, 1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([True, True], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32, device=device),
        num_seqs=2,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    if mode == "decode":
        ple.run_decode(binding, eps=1e-6)
    else:
        ple.run_prefill(binding, eps=1e-6)
    torch.cuda.synchronize()

    assert binding.error_code.item() == 0
    torch.testing.assert_close(conv_state[0], empty_state_before, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_layer_zero_sequences_with_tokens_fails_without_mutation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 1, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1208
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (1, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    state_before = conv_state.clone()
    _, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([0], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        num_seqs=0,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()

    assert binding.error_code.item() & 1
    assert bool((binding.out == 0).all().item())
    torch.testing.assert_close(conv_state, state_before, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_dummy_slots_replay_under_cuda_graph_without_state_mutation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 3, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1203
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (2, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    _, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([-1, -1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False, False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32, device=device),
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()
    state_before = conv_state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ple.run_decode(binding, eps=1e-6)
    allocated_before_replay = torch.cuda.memory_allocated(device)
    for _ in range(3):
        binding.out.fill_(91)
        graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert binding.error_code.item() == 0
    assert bool((binding.out == 0).all().item())
    torch.testing.assert_close(conv_state, state_before, rtol=0, atol=0)
    assert allocated_after_replay == allocated_before_replay


@torch.inference_mode()
def test_ple_prefill_replays_under_cuda_graph_without_allocation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 3, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1209
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    channels = streams * hidden
    state_capacity = state_length + max_speculative
    state_payload = channels * state_capacity
    state_slot_stride = state_payload + 19
    state_storage = torch.randn(
        state_slot_stride + state_payload,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state = torch.as_strided(
        state_storage,
        (2, channels, state_capacity),
        (state_slot_stride, state_capacity, 1),
    )
    untouched_state = conv_state[0].clone()
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    plan, binding = _bind_cuda_layer(
        mode="prefill",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=torch.tensor([1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([True], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([0], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_prefill(binding, eps=1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple.run_prefill(binding, eps=1e-6)
    output_address = captured_out.data_ptr()

    residual.copy_(torch.randn_like(residual))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated(device)
    expected, expected_state = ple_projected_packed_reference(
        residual,
        key,
        value,
        query_start_loc,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
    )

    assert binding.error_code.item() == 0
    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_out, expected, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(
        conv_state[1, :, : plan.state_length],
        expected_state[0],
        rtol=0.02,
        atol=0.0078125,
    )
    assert bool((conv_state[1, :, plan.state_length :] == 0).all().item())
    torch.testing.assert_close(conv_state[0], untouched_state, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_mixed_replays_runtime_request_modes_without_allocation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 4, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1213
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (2, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    state_slot_ids = torch.tensor([0, 1], dtype=torch.int64, device=device)
    state_is_fresh = torch.tensor([True, False], dtype=torch.bool, device=device)
    num_accepted_tokens = torch.tensor([99, 1], dtype=torch.int32, device=device)
    request_is_prefill = torch.tensor([True, False], dtype=torch.bool, device=device)
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=request_is_prefill,
    )

    ple.run_mixed(binding, eps=1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple.run_mixed(binding, eps=1e-6)
    torch.cuda.synchronize()
    output_address = captured_out.data_ptr()

    residual.copy_(torch.randn_like(residual))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    state_is_fresh.zero_()
    num_accepted_tokens.copy_(torch.tensor([2, 99], dtype=torch.int32, device=device))
    request_is_prefill.logical_not_()
    expected_out, expected_state = _mixed_layer_reference(
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        request_is_prefill=request_is_prefill,
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        dilation=dilation,
        eps=1e-6,
    )
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert binding.error_code.item() == 0
    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_out, expected_out, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(conv_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("token_count", "high_state_slots"), [(None, False), (8, False), (8, True)]
)
@torch.inference_mode()
def test_ple_target_mixed_graph_replays_dynamic_packed_metadata(
    token_count: int | None, high_state_slots: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from triton.runtime import JITFunction

    from b12x.sequence.ple import _kernels

    device = require_b12x()
    max_tokens, streams, hidden = 16, 4, 2560
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        max_tokens, streams, hidden, device=device, seed=1214
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    slot_stride = streams * hidden * (state_length + max_speculative)
    slot_offset = (1 << 31) // slot_stride + 1 if high_state_slots else 0
    conv_state = torch.empty(
        (slot_offset + 3, streams * hidden, state_length + max_speculative),
        dtype=torch.bfloat16,
        device=device,
    )
    live_state = conv_state[slot_offset:]
    live_state.normal_(generator=generator)
    query_start_loc = torch.tensor([0, 2, 5, 5], dtype=torch.int32, device=device)
    state_slot_ids = torch.arange(3, dtype=torch.int64, device=device) + slot_offset
    state_is_fresh = torch.tensor([True, False, False], dtype=torch.bool, device=device)
    num_accepted_tokens = torch.tensor([99, 2, 1], dtype=torch.int32, device=device)
    request_is_prefill = torch.tensor(
        [True, False, False], dtype=torch.bool, device=device
    )
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=2,
        num_tokens=5,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=request_is_prefill,
    )

    ple.run_mixed(binding, eps=1e-6)

    def reject_resolution(*args, **kwargs):
        pytest.fail("Live launch bounds must reuse the warmed PLE kernels")

    for kernel in vars(_kernels).values():
        if isinstance(kernel, JITFunction):
            monkeypatch.setattr(kernel, "_do_compile", reject_resolution)

    state_before = live_state.clone()
    binding.num_seqs.zero_()
    binding.num_tokens.zero_()
    for launch_bound in (0, 1, 3, 8):
        binding.out.fill_(91)
        out = ple.run_mixed(binding, eps=1e-6, token_count=launch_bound)
        assert bool((out == 0).all().item())
        assert bool((binding.out[launch_bound:] == 91).all().item())
    torch.testing.assert_close(live_state, state_before, rtol=0, atol=0)
    binding.num_seqs.fill_(2)
    binding.num_tokens.fill_(5)
    binding.out.fill_(91)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple.run_mixed(binding, eps=1e-6, token_count=token_count)
    output_address = captured_out.data_ptr()

    query_start_loc.copy_(torch.tensor([0, 1, 5, 8], dtype=torch.int32, device=device))
    binding.num_seqs.fill_(3)
    binding.num_tokens.fill_(8)
    request_is_prefill.copy_(
        torch.tensor([False, True, False], dtype=torch.bool, device=device)
    )
    num_accepted_tokens.copy_(
        torch.tensor([2, 99, 5], dtype=torch.int32, device=device)
    )
    state_is_fresh.copy_(
        torch.tensor([False, True, False], dtype=torch.bool, device=device)
    )
    residual.copy_(torch.randn_like(residual).mul_(0.2))
    key.copy_(torch.randn_like(key).mul_(0.2))
    value.copy_(torch.randn_like(value).mul_(0.2))
    live_state.copy_(torch.randn_like(live_state).mul_(0.2))
    expected_out, expected_state = _mixed_layer_reference(
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids - slot_offset,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        request_is_prefill=request_is_prefill,
        num_seqs=3,
        num_tokens=8,
        conv_state=live_state,
        dilation=dilation,
        eps=1e-6,
    )
    initial_state = live_state.clone()
    full_out = ple.run_mixed(binding, eps=1e-6).clone()
    full_state = live_state.clone()
    live_state.copy_(initial_state)
    binding.out.fill_(91)
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert binding.error_code.item() == 0
    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(
        captured_out, expected_out[: captured_out.shape[0]], rtol=0.02, atol=0.0078125
    )
    torch.testing.assert_close(
        captured_out, full_out[: captured_out.shape[0]], rtol=0, atol=0
    )
    assert bool((binding.out[captured_out.shape[0] :] == 91).all().item())
    torch.testing.assert_close(live_state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(live_state, full_state, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_public_runs_export_as_opaque_mutating_custom_ops() -> None:
    device = require_b12x()
    hash_plan = ple_hash.plan(
        ple_hash.Caps(
            device=device,
            max_tokens=2,
            max_seqs=1,
            vocab_size=100,
            eos_token_id=99,
            max_order=3,
            heads_per_order=2,
            dense_layer_ordinal=0,
            base_table_size=101,
        )
    )
    hash_binding = ple_hash.bind(
        hash_plan,
        scratch=_scratch(hash_plan),
        token_ids=torch.tensor([1, 2], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
        committed_history=torch.tensor([[99, 99]], dtype=torch.int64, device=device),
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([2], dtype=torch.int32, device=device),
        out=torch.empty((2, 4), dtype=torch.int64, device=device),
    )

    tokens, streams, hidden = 1, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1205
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (1, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    _, layer_binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([-1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )
    _, mixed_binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([-1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([99], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=torch.tensor([True], dtype=torch.bool, device=device),
    )

    hash_graph, _ = torch._dynamo.export(lambda: ple_hash.run(hash_binding))()
    layer_graph, _ = torch._dynamo.export(
        lambda: ple.run_decode(layer_binding, eps=1e-6)
    )()
    mixed_graph, _ = torch._dynamo.export(
        lambda: ple.run_mixed(mixed_binding, eps=1e-6)
    )()
    assert "torch.ops.b12x.ple_hash_pipeline" in hash_graph.code
    assert "torch.ops.b12x.ple_layer_pipeline" in layer_graph.code
    assert "torch.ops.b12x.ple_layer_mixed_pipeline" in mixed_graph.code
    assert "triton" not in hash_graph.code
    assert "triton" not in layer_graph.code
    assert "triton" not in mixed_graph.code

    compiled_hash = torch.compile(
        lambda: ple_hash.run(hash_binding), backend="eager", fullgraph=True
    )
    compiled_layer = torch.compile(
        lambda: ple.run_decode(layer_binding, eps=1e-6),
        backend="eager",
        fullgraph=True,
    )
    compiled_mixed = torch.compile(
        lambda: ple.run_mixed(mixed_binding, eps=1e-6),
        backend="eager",
        fullgraph=True,
    )
    compiled_hash()
    compiled_layer()
    compiled_mixed()
    torch.cuda.synchronize()
    assert hash_binding.error_code.item() == 0
    assert layer_binding.error_code.item() == 0
    assert mixed_binding.error_code.item() == 0
    assert bool((layer_binding.out == 0).all().item())
    assert bool((mixed_binding.out == 0).all().item())


@torch.inference_mode()
def test_ple_state_slot_past_int32_element_offset_matches_oracle() -> None:
    device = require_b12x()
    tokens, streams, hidden = 1, 4, 2560
    kernel_size, dilation, max_speculative = 4, 3, 4
    state_length = dilation * (kernel_size - 1)
    state_capacity = state_length + max_speculative
    channels = streams * hidden
    slot_stride_elements = channels * state_capacity
    int32_max = torch.iinfo(torch.int32).max
    high_slot = int32_max // slot_stride_elements + 2
    assert high_slot * slot_stride_elements > int32_max

    binding = None
    conv_state = None
    try:
        # The pool is intentionally mostly uninitialized. Only slot zero and
        # the live tail slot are touched, reproducing a high recycled slot ID.
        conv_state = torch.empty(
            (high_slot + 1, channels, state_capacity),
            dtype=torch.bfloat16,
            device=device,
        )
        assert conv_state.stride(0) == slot_stride_elements
        assert high_slot * conv_state.stride(0) > int32_max
        conv_state[0].fill_(7)
        low_slot_before = conv_state[0].clone()

        residual, key, value, weights, generator = _cuda_projected_inputs(
            tokens, streams, hidden, device=device, seed=1204
        )
        conv_weight = (
            torch.randn(
                (channels, kernel_size),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )
            / 32
        ).contiguous()
        prior = torch.randn(
            (channels, state_capacity),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        conv_state[high_slot].copy_(prior)
        plan, binding = _bind_cuda_layer(
            mode="decode",
            residual=residual,
            key=key,
            value=value,
            weights=weights,
            conv_weight=conv_weight,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            state_slot_ids=torch.tensor([high_slot], dtype=torch.int64, device=device),
            state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
            num_accepted_tokens=torch.tensor([3], dtype=torch.int32, device=device),
            num_seqs=1,
            num_tokens=1,
            conv_state=conv_state,
            max_speculative_tokens=max_speculative,
            dilation=dilation,
        )
        effective_history = prior[:, 2 : 2 + state_length].contiguous()
        expected, _ = ple_projected_sequence_reference(
            residual,
            key,
            value,
            k_norm_weight=weights[0],
            q_norm_weight=weights[1],
            u_norm_weight=weights[2],
            conv_weight=conv_weight,
            eps=1e-6,
            dilation=dilation,
            prior_state=effective_history,
        )
        _, normalized_u = ple_projected_u_reference(
            residual,
            key,
            value,
            k_norm_weight=weights[0],
            q_norm_weight=weights[1],
            u_norm_weight=weights[2],
            eps=1e-6,
        )
        expected_base = torch.cat(
            (effective_history[:, 1:], normalized_u[0].unsqueeze(1)), dim=1
        )

        ple.run_decode(binding, eps=1e-6)
        torch.cuda.synchronize()

        assert binding.error_code.item() == 0
        torch.testing.assert_close(binding.out, expected, rtol=0.02, atol=0.0078125)
        torch.testing.assert_close(
            conv_state[high_slot, :, : plan.state_length],
            expected_base,
            rtol=0.02,
            atol=0.0078125,
        )
        assert bool((conv_state[high_slot, :, plan.state_length :] == 0).all().item())
        torch.testing.assert_close(conv_state[0], low_slot_before, rtol=0, atol=0)
    finally:
        del binding
        del conv_state
        torch.cuda.empty_cache()
