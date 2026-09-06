from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe import rank_sliced_trellis


def _plan(*, hidden_size: int = 128, intermediate_size: int = 256):
    return fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="b12x_trellis",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=4,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_layout="w13",
        w4a16_layout="trellis_native",
        trellis_bits=3,
        trellis_tile_config=(64, 256, 64, 256),
        trellis_codebook="mcg",
        trellis_rate_granularity="uniform",
    )


def test_rank_sliced_trellis_adapter_is_public_and_transfers_prepared_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = SimpleNamespace(
        w13=torch.empty((2,), dtype=torch.uint8),
        w2=torch.empty((2,), dtype=torch.uint8),
        w13_scale=torch.empty((2,), dtype=torch.float16),
        w2_scale=torch.empty((2,), dtype=torch.float16),
        w13_global_scale=torch.empty((2,), dtype=torch.float32),
        w2_global_scale=torch.empty((2,), dtype=torch.float32),
    )
    calls: list[dict[str, object]] = []

    def fake_prepare(**kwargs):
        calls.append(kwargs)
        return prepared

    monkeypatch.setattr(
        rank_sliced_trellis, "prepare_trellis256_moe_weights", fake_prepare
    )
    tensors = {
        name: torch.empty((1,), dtype=torch.uint8)
        for name in ("w13", "w2", "gate_suh", "up_suh", "down_svh")
    }
    tensors["intermediate_rotations"] = torch.empty((1,), dtype=torch.float16)

    result = fused_moe.prepare_rank_sliced_trellis_weights(plan=_plan(), **tensors)

    assert result.w1_fp4 is prepared.w13
    assert result.w2_fp4 is prepared.w2
    assert result.w1_blockscale is prepared.w13_scale
    assert result.w2_blockscale is prepared.w2_scale
    assert calls == [
        {
            **tensors,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_experts": 4,
            "activation": "silu",
            "fc1_tile_n": 256,
            "fc2_tile_n": 256,
            "params_dtype": torch.float16,
            "w13_layout": "trellis_t256_proj",
            "trellis_bits": 3,
            "codebook": "mcg",
            "tile_config": (64, 256, 64, 256),
        }
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_rank_sliced_trellis_adapter_reuses_real_prepared_storage() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    experts = 4
    hidden = intermediate = 256
    bits = 3
    plan = _plan(hidden_size=hidden, intermediate_size=intermediate)

    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.ones(shape, dtype=torch.float16, device=device)

    result = fused_moe.prepare_rank_sliced_trellis_weights(
        plan=plan,
        w13=w13,
        w2=w2,
        gate_suh=scales((experts, hidden)),
        up_suh=scales((experts, hidden)),
        intermediate_rotations=scales((experts, 3 * intermediate)),
        down_svh=scales((experts, hidden)),
    )

    assert result.w1_fp4.data_ptr() == w13.data_ptr()
    assert result.w2_fp4.data_ptr() == w2.data_ptr()
    assert result.w1_fp4.dtype == torch.int32
    assert result.w2_fp4.dtype == torch.int32
    assert result.w1_fp4.numel() == w13.numel() // 2
    assert result.w2_fp4.numel() == w2.numel() // 2
    assert result.representation is not None
    assert result.representation.layout.value == "trellis_native"
    assert result.representation.value.weight_layout == "trellis_t256"
    assert result.representation.value.w13_layout == "trellis_t256_proj"
    for value in (
        result.a1_gscale,
        result.w1_alphas,
        result.a2_gscale,
        result.w2_alphas,
    ):
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) == value.numel()


def test_rank_sliced_trellis_adapter_rejects_nonuniform_rate_plan() -> None:
    plan = replace(_plan(), trellis_rate_granularity="per_expert")

    with pytest.raises(ValueError, match="requires uniform rates"):
        fused_moe.prepare_rank_sliced_trellis_weights(
            plan=plan,
            w13=torch.empty(0),
            w2=torch.empty(0),
            gate_suh=torch.empty(0),
            up_suh=torch.empty(0),
            intermediate_rotations=torch.empty(0),
            down_svh=torch.empty(0),
        )


def test_rank_sliced_trellis_adapter_rejects_nonplan() -> None:
    with pytest.raises(TypeError, match="MoEWeightPreparationPlan"):
        fused_moe.prepare_rank_sliced_trellis_weights(
            plan=object(),
            w13=torch.empty(0),
            w2=torch.empty(0),
            gate_suh=torch.empty(0),
            up_suh=torch.empty(0),
            intermediate_rotations=torch.empty(0),
            down_svh=torch.empty(0),
        )
