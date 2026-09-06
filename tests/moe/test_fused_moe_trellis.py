from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("cutlass")

from b12x.moe import fused_moe
from b12x.moe.fused_moe import META as FUSED_MOE_META
from b12x.moe._shared.kernels.w4a16.host import plan_w4a16_buffers
from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from b12x.moe._shared.execution import PreparedWeightLayout
from b12x.moe._shared.kernels.w4a16.prepare import (
    PreparedW4A16MoeWeights,
    prepare_trellis256_moe_weights,
)
from b12x.moe.fused_moe._impl import (
    B12XFP4ExpertWeights,
    _PreparedWeightRepresentation,
    plan_b12x_fp4_moe_weights,
)
from b12x._lib.quant.sqg_e4m3 import (
    sqg_xor_cheb_t12_direct_lut_cpu,
)
from b12x._lib.quant.sqg_fp16_d3l import sqg_fp16_d3l_direct_lut_cpu

_MCG = np.uint64(0xCBAC1FED)
_MCG_MASK = np.uint32(0x8FFF8FFF)
_MCG_OR = np.uint32(0x3B603B60)


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12 and minor in (0, 1)


def _weight_plan(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    input_dtype: torch.dtype,
    activation: str = "silu",
    trellis_bits: int = 3,
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
    codebook: str = "mcg",
    coupled_hadamard: bool | None = None,
) -> fused_moe.WeightsPlan:
    # These tests deliberately exercise private historical kernel recipes that
    # are not representable by the canonical checkpoint schema.
    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="btx",
        activation=activation,
        params_dtype=input_dtype,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_layout="w13",
        trellis_bits=trellis_bits,
        trellis_codebook=codebook,
        trellis_tile_config=tile_config,
        coupled_hadamard=coupled_hadamard,
    )
    return plan


def _caps(**overrides) -> fused_moe.Caps:
    values = {
        "max_tokens": 32,
        "num_topk": 8,
        "num_experts": 192,
        "hidden_size": 6144,
        "intermediate_size": 512,
        "route_num_experts": 256,
        "w4a16_block_size_m": 8,
        "input_dtype": torch.bfloat16,
        "device": "cuda:0",
    }
    values.update(overrides)
    weight_plan = _weight_plan(
        num_experts=values.pop("num_experts"),
        hidden_size=values.pop("hidden_size"),
        intermediate_size=values.pop("intermediate_size"),
        input_dtype=values.pop("input_dtype"),
    )
    return fused_moe.Caps(
        weight_plan=weight_plan,
        quant_mode="w4a16",
        **values,
    )


def test_fused_moe_metadata_advertises_trellis_input_dtypes() -> None:
    assert {"bf16", "fp16"}.issubset(FUSED_MOE_META.dtypes)

def test_rotation_placeholder_must_be_materialized_before_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    device = torch.device("cuda:127")
    w4a16_kernel._ROT_SCALES_DUMMY.pop(device, None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="prewarm the planned launch"):
        w4a16_kernel._rot_scales_dummy(device)


def test_dense_scratch_cannot_allocate_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="provide caller-owned storage"):
        w4a16_kernel._trellis_dense_buffer(
            "rotated_f16",
            None,
            shape=(2, 128),
            dtype=torch.float16,
            device=torch.device("cpu"),
        )


def test_caps_preserve_exact_block_m_and_input_dtypes() -> None:
    bf16 = _caps(w4a16_block_size_m=8, input_dtype=torch.bfloat16)
    fp16 = _caps(w4a16_block_size_m=64, input_dtype=torch.float16)

    assert bf16.w4a16_block_size_m == 8
    assert bf16.route_num_experts == 256
    assert bf16.dtype == torch.bfloat16
    assert fp16.w4a16_block_size_m == 64
    assert fp16.dtype == torch.float16


@pytest.mark.parametrize("block_size_m", [0, 4, 12, 128])
def test_caps_reject_unsupported_block_m(block_size_m: int) -> None:
    with pytest.raises(ValueError, match="w4a16_block_size_m"):
        _caps(w4a16_block_size_m=block_size_m)


def _prepare_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    input_dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
    codebook: str = "mcg",
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
) -> fused_moe.ExpertWeights:
    bits = int(w13.shape[-1]) // 16
    plan = _weight_plan(
        num_experts=int(w13.shape[1]),
        hidden_size=int(w13.shape[2]) * 16,
        intermediate_size=int(w13.shape[3]) * 16,
        input_dtype=input_dtype,
        activation=activation,
        trellis_bits=bits,
        tile_config=tile_config,
        codebook=codebook,
    )
    value = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=int(w13.shape[2]) * 16,
        intermediate_size=int(w13.shape[3]) * 16,
        num_experts=int(w13.shape[1]),
        activation=activation,
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook=codebook,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )
    unit = torch.ones((), dtype=torch.float32, device=value.w13.device)
    return B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=unit,
        w1_fp4=value.w13,
        w1_blockscale=value.w13_scale,
        w1_alphas=value.w13_global_scale,
        a2_gscale=unit,
        w2_fp4=value.w2,
        w2_blockscale=value.w2_scale,
        w2_alphas=value.w2_global_scale,
        representation=_PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=PreparedWeightLayout.TRELLIS_NATIVE,
            value=value,
        ),
    )


def _plan(
    weights: fused_moe.ExpertWeights,
    *,
    max_tokens: int,
    num_topk: int,
    route_num_experts: int,
    block_size_m: int,
    device: torch.device | str,
) -> fused_moe.Plan:
    return fused_moe.plan(
        fused_moe.Caps(
            max_tokens=max_tokens,
            num_topk=num_topk,
            route_num_experts=route_num_experts,
            device=device,
            weight_plan=weights.plan,
            quant_mode="w4a16",
            w4a16_block_size_m=block_size_m,
        )
    )


def test_low_level_buffer_plan_honors_explicit_block_m() -> None:
    prepared = SimpleNamespace(
        num_experts=256,
        hidden_size=6144,
        intermediate_size=512,
        is_gated=True,
    )
    automatic = plan_w4a16_buffers(
        prepared,
        m=1024,
        topk=8,
        route_num_experts=256,
        sms=120,
    )
    exact = plan_w4a16_buffers(
        prepared,
        m=1024,
        topk=8,
        route_num_experts=256,
        sms=120,
        full_rotation=True,
        block_size_m=8,
    )

    assert automatic.block_size_m != 8
    assert exact.block_size_m == 8
    assert exact.rotation_a_elements == 1024 * 8 * 6144
    with pytest.raises(ValueError, match="block_size_m"):
        plan_w4a16_buffers(
            prepared,
            m=1,
            topk=8,
            route_num_experts=256,
            sms=120,
            block_size_m=4,
        )


def test_planned_route_block_overrides_live_batch_heuristic() -> None:
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _resolve_route_block_size_m,
    )

    # m=1024/topk=8/E=256 selects 48 heuristically. A caller-owned arena
    # planned for 64 must retain 64 so route packing cannot outgrow its
    # block-expert table.
    assert _resolve_route_block_size_m(
        m=1024,
        topk=8,
        route_num_experts=256,
        planned_block_size_m=None,
        fused_launch=None,
    ) == 48
    assert _resolve_route_block_size_m(
        m=1024,
        topk=8,
        route_num_experts=256,
        planned_block_size_m=64,
        fused_launch=None,
    ) == 64
    with pytest.raises(RuntimeError, match="planned_block_size_m=64"):
        _resolve_route_block_size_m(
            m=1024,
            topk=8,
            route_num_experts=256,
            planned_block_size_m=64,
            fused_launch=SimpleNamespace(moe_block_size=48),
        )


def _cpu_weight_tensors(num_experts: int = 1) -> tuple[torch.Tensor, ...]:
    w13 = torch.zeros((2, num_experts, 8, 8, 48), dtype=torch.int16)
    w2 = torch.zeros((num_experts, 8, 8, 48), dtype=torch.int16)
    edge = torch.ones((1, 128), dtype=torch.float16)
    intermediate = torch.ones((num_experts, 384), dtype=torch.float16)
    return w13, w2, edge, edge.clone(), intermediate, edge.clone()

@lru_cache(maxsize=None)
def _sqg_xor_cheb_t12_table(bits: int) -> np.ndarray:
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported QSRT test rate K{bits}")
    labels = sqg_xor_cheb_t12_direct_lut_cpu()
    rate_labels = labels[(bits - 2) << 16 : (bits - 1) << 16]
    return rate_labels.view(torch.float8_e4m3fn).to(torch.float16).numpy()


@lru_cache(maxsize=None)
def _sqg_fp16_d3l_table(bits: int) -> np.ndarray:
    return sqg_fp16_d3l_direct_lut_cpu(bits).numpy()


def _decode_mcg_fp16(window: np.ndarray) -> np.ndarray:
    value = window.astype(np.uint64)
    value = ((value * _MCG) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    value = np.uint32((value & _MCG_MASK) ^ _MCG_OR)
    low = (value & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        ((value >> np.uint32(16)) & np.uint32(0xFFFF))
        .astype(np.uint16)
        .view(np.float16)
    )
    return (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)


def _decode_lane(
    tile_words: np.ndarray,
    lane: int,
    bits: int,
    *,
    codebook: str = "mcg",
) -> np.ndarray:
    if codebook not in {"mcg", "sqg_e4m3", "sqg_fp16"}:
        raise ValueError(f"unsupported test codebook {codebook!r}")
    width = 8 * bits
    values = []
    for weight in range(8):
        end_bit = (lane * 8 + weight + 257) * bits
        start_bit = end_bit - 16
        first_word = start_bit // 32
        last_word = (end_bit - 1) // 32
        shift = (last_word + 1) * 32 - end_bit
        first = tile_words[..., first_word % width].astype(np.uint64)
        last = tile_words[..., last_word % width].astype(np.uint64)
        merged = (first << np.uint64(32)) | last
        window = ((merged >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.uint32)
        if codebook == "mcg":
            values.append(_decode_mcg_fp16(window))
        elif codebook == "sqg_e4m3":
            values.append(_sqg_xor_cheb_t12_table(bits)[window])
        else:
            values.append(_sqg_fp16_d3l_table(bits)[window])
    return np.stack(values, axis=-1).astype(np.float16)


def _reconstruct_native(
    trellis: torch.Tensor,
    *,
    codebook: str = "mcg",
) -> torch.Tensor:
    native = trellis.detach().cpu().numpy()
    bits = int(native.shape[-1]) // 16
    k_tiles, n_tiles, _ = native.shape
    packed = native.view(np.uint16).reshape(k_tiles, n_tiles, 8 * bits, 2)
    words = packed[..., 0].astype(np.uint32) | (
        packed[..., 1].astype(np.uint32) << np.uint32(16)
    )
    output = np.zeros((k_tiles * 16, n_tiles * 16), dtype=np.float16)
    for k_tile in range(k_tiles):
        for n_tile in range(n_tiles):
            lanes = np.stack(
                [
                    _decode_lane(
                        words[k_tile, n_tile],
                        lane,
                        bits,
                        codebook=codebook,
                    )
                    for lane in range(32)
                ]
            )
            block = np.zeros((16, 16), dtype=np.float16)
            for lane in range(32):
                row0 = (lane % 4) * 2
                rows = (row0, row0 + 1, row0 + 8, row0 + 9)
                col0 = lane // 8
                col1 = col0 + 4
                parity = (lane >> 2) & 1
                for weight in range(8):
                    block[
                        rows[weight % 4], 2 * (col0 if weight < 4 else col1) + parity
                    ] = lanes[lane, weight]
            output[
                k_tile * 16 : (k_tile + 1) * 16,
                n_tile * 16 : (n_tile + 1) * 16,
            ] = block
    return torch.from_numpy(output)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [5, 6])
def test_sqg_fp16_d3l_uniform_w4a16_matches_reference_and_captures(
    bits: int,
) -> None:
    """Close native K5/K6 payloads through the actual A16 fused kernel."""

    torch.manual_seed(20260808 + bits)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = 1
    m = 2
    hidden = intermediate = 128
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
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((experts, hidden)).contiguous()
    up_suh = scales((experts, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((experts, hidden)).contiguous()
    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=128,
        fc2_tile_n=128,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook="sqg_fp16",
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=(64, 128, 64, 128),
    )
    assert prepared.source_format == "btx"
    assert prepared.trellis_codebook == "sqg_fp16"

    gate_weights = _reconstruct_native(
        w13[0, 0], codebook="sqg_fp16"
    ).unsqueeze(0).to(device)
    up_weights = _reconstruct_native(
        w13[1, 0], codebook="sqg_fp16"
    ).unsqueeze(0).to(device)
    down_weights = _reconstruct_native(
        w2[0], codebook="sqg_fp16"
    ).unsqueeze(0).to(device)
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    router_weights = torch.ones((m, topk), dtype=torch.float32, device=device)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            router_weights,
            ids,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=intermediate_rotations,
            full_rotation=True,
            suh_gate_table=gate_suh,
            suh_up_table=up_suh,
            svh_table=down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    actual = run().clone()
    torch.cuda.synchronize(device)
    reference = _reference_full_rotation_decoded(
        x,
        ids,
        router_weights,
        gate_weights,
        up_weights,
        down_weights,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)

def _hadamard_128(device: torch.device) -> torch.Tensor:
    indices = torch.arange(128, dtype=torch.int64)
    rows = []
    for row in range(128):
        parity = torch.tensor(
            [(row & int(col)).bit_count() & 1 for col in indices],
            dtype=torch.bool,
        )
        rows.append(torch.where(parity, -1.0, 1.0))
    return (torch.stack(rows) / (128.0**0.5)).to(device)


def _had128(
    value: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
    store_fp16: bool,
) -> torch.Tensor:
    work = value.float()
    if suh is not None:
        work = (work * suh.float()).to(torch.float16).float()
    rows, width = work.shape
    work = (work.view(rows, width // 128, 128) @ hadamard).view(rows, width)
    if svh is not None:
        work = work * svh.float()
    return work.to(torch.float16) if store_fp16 else work


def _coupled_h512(value: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """Apply the normalized H512 used at the coupled residual boundary."""

    work = value.float()
    rows, width = work.shape
    work = work.view(rows, width // 512, 4, 128) @ hadamard
    h4 = torch.tensor(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        ),
        dtype=torch.float32,
        device=value.device,
    ) * 0.5
    work = torch.einsum("rbgc,gh->rbhc", work, h4)
    return work.reshape(rows, width)


def _reference_full_rotation(
    x: torch.Tensor,
    local_ids: torch.Tensor,
    router_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    device = x.device
    experts = int(w2.shape[0])
    intermediate = int(w2.shape[1]) * 16
    hidden = int(w2.shape[2]) * 16
    hadamard = _hadamard_128(device)
    gate_weights = torch.stack(
        [_reconstruct_native(w13[0, expert]) for expert in range(experts)]
    ).to(device)
    up_weights = torch.stack(
        [_reconstruct_native(w13[1, expert]) for expert in range(experts)]
    ).to(device)
    down_weights = torch.stack(
        [_reconstruct_native(w2[expert]) for expert in range(experts)]
    ).to(device)
    return _reference_full_rotation_decoded(
        x,
        local_ids,
        router_weights,
        gate_weights,
        up_weights,
        down_weights,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation=activation,
    )


def _reference_full_rotation_decoded(
    x: torch.Tensor,
    local_ids: torch.Tensor,
    router_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    device = x.device
    experts = int(down_weights.shape[0])
    intermediate = int(down_weights.shape[1])
    hidden = int(down_weights.shape[2])
    hadamard = _hadamard_128(device)
    gate_svh = intermediate_rotations[:, :intermediate]
    up_svh = intermediate_rotations[:, intermediate : 2 * intermediate]
    down_suh = intermediate_rotations[:, 2 * intermediate :]
    output = torch.zeros((int(x.shape[0]), hidden), dtype=torch.float32, device=device)
    for token in range(int(x.shape[0])):
        for slot in range(int(local_ids.shape[1])):
            expert = int(local_ids[token, slot])
            h_side_expert = 0 if int(gate_suh.shape[0]) == 1 else expert
            source = x[token : token + 1].to(torch.float16)
            gate_a = _had128(
                source,
                hadamard,
                suh=gate_suh[h_side_expert],
                store_fp16=True,
            )
            up_a = _had128(
                source,
                hadamard,
                suh=up_suh[h_side_expert],
                store_fp16=True,
            )
            gate = (gate_a.float() @ gate_weights[expert].float()).to(torch.float16)
            up = (up_a.float() @ up_weights[expert].float()).to(torch.float16)
            gate = _had128(
                gate,
                hadamard,
                svh=gate_svh[expert],
                store_fp16=True,
            )
            up = _had128(
                up,
                hadamard,
                svh=up_svh[expert],
                store_fp16=True,
            )
            if activation == "situ":
                activated = (
                    4.0
                    * torch.tanh(gate.float() / 4.0)
                    * torch.sigmoid(gate.float())
                    * 25.0
                    * torch.tanh(up.float() / 25.0)
                ).to(torch.float16)
            else:
                activated = (
                    gate.float() * torch.sigmoid(gate.float()) * up.float()
                ).to(torch.float16)
            down_a = _had128(
                activated,
                hadamard,
                suh=down_suh[expert],
                store_fp16=True,
            )
            down = (down_a.float() @ down_weights[expert].float()).to(torch.float16)
            route = _had128(
                down,
                hadamard,
                svh=down_svh[0 if int(down_svh.shape[0]) == 1 else expert],
                store_fp16=False,
            )
            output[token] += router_weights[token, slot] * route[0]
    return output


def _reference_coupled_decoded(
    x: torch.Tensor,
    local_ids: torch.Tensor,
    router_weights: torch.Tensor,
    slot0_weights: torch.Tensor,
    slot1_weights: torch.Tensor,
    down_weights: torch.Tensor,
    source_suh: torch.Tensor,
    ordinary_rotations: torch.Tensor,
    coupled_signs: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    fc1_pair_kind: str | None = None,
    quantize_activations: bool = False,
) -> torch.Tensor:
    """Reference the exact transform order of a coupled uniform-rate path."""

    device = x.device
    experts = int(down_weights.shape[0])
    intermediate = int(down_weights.shape[1])
    hidden = int(down_weights.shape[2])
    hadamard = _hadamard_128(device)
    slot0_svh = ordinary_rotations[:, :intermediate]
    slot1_svh = ordinary_rotations[:, intermediate : 2 * intermediate]
    down_suh = ordinary_rotations[:, 2 * intermediate :]
    pre_signs = coupled_signs[:, : 2 * intermediate]
    post_signs = coupled_signs[:, 2 * intermediate :]
    output = torch.zeros((int(x.shape[0]), hidden), dtype=torch.float32, device=device)

    for token in range(int(x.shape[0])):
        route_values: list[torch.Tensor] = []
        for slot in range(int(local_ids.shape[1])):
            expert = int(local_ids[token, slot])
            h_side_expert = 0 if int(source_suh.shape[0]) == 1 else expert
            source = _coupled_h512(x[token : token + 1].to(torch.float16), hadamard)
            if quantize_activations:
                source = _had128(
                    source * source_suh[h_side_expert].float(),
                    hadamard,
                    store_fp16=False,
                )
                source = _reference_mxfp8_rows(source)
            else:
                source = (source * source_suh[h_side_expert].float()).to(
                    torch.float16
                )
                source = _had128(source, hadamard, store_fp16=True)

            atoms = intermediate // 32
            physical = [
                (source.float() @ weights[expert].float()).to(torch.float16)
                for weights in (slot0_weights, slot1_weights)
            ]
            if fc1_pair_kind is None:
                atoms = intermediate // 32
                raw = torch.stack(
                    (
                        physical[0].view(1, atoms, 32),
                        physical[1].view(1, atoms, 32),
                    ),
                    dim=2,
                ).reshape(1, 2 * intermediate)
                scales = torch.stack(
                    (
                        slot0_svh[expert].view(atoms, 32),
                        slot1_svh[expert].view(atoms, 32),
                    ),
                    dim=1,
                ).reshape(1, 2 * intermediate)
                pre = _had128(raw, hadamard, store_fp16=False)
                pre = _had128(
                    pre * scales.float(), hadamard, store_fp16=False
                )
            else:
                decoded = [
                    _had128(
                        physical[index],
                        hadamard,
                        svh=(slot0_svh, slot1_svh)[index][expert],
                        store_fp16=False,
                    )
                    for index in range(2)
                ]
            if fc1_pair_kind == "P43":
                # P43 stores each slot's funded K4 record before its logical
                # first K3 record.
                logical = torch.cat(
                    (
                        decoded[0][:, 128:],
                        decoded[1][:, 128:],
                        decoded[0][:, :128],
                        decoded[1][:, :128],
                    ),
                    dim=1,
                )
            elif fc1_pair_kind == "P33":
                logical = torch.cat(
                    (
                        decoded[0][:, :128],
                        decoded[1][:, :128],
                        decoded[0][:, 128:],
                        decoded[1][:, 128:],
                    ),
                    dim=1,
                )
            elif fc1_pair_kind is not None:
                raise ValueError(f"unsupported coupled FC1 pair kind {fc1_pair_kind}")
            if fc1_pair_kind is not None:
                pre = _had128(logical, hadamard, store_fp16=False)
            pre = pre * pre_signs[expert].float()
            gate = pre[:, 0::2]
            up = pre[:, 1::2]
            activated = (
                4.0
                * torch.tanh(gate / 4.0)
                * torch.sigmoid(gate)
                * 25.0
                * torch.tanh(up / 25.0)
            )
            activated = activated * post_signs[expert].float()
            activated = _had128(activated, hadamard, store_fp16=False)
            activated = _had128(
                activated * down_suh[expert].float(),
                hadamard,
                store_fp16=not quantize_activations,
            )
            if quantize_activations:
                activated = _reference_mxfp8_rows(activated)
            down = (
                activated.float() @ down_weights[expert].float()
            ).to(torch.float16)
            route = _had128(down, hadamard, store_fp16=False)
            route = route * down_svh[0 if int(down_svh.shape[0]) == 1 else expert]
            route_values.append(route)

        routed = sum(
            router_weights[token, slot] * route_values[slot]
            for slot in range(int(local_ids.shape[1]))
        )
        output[token] = _coupled_h512(routed, hadamard)[0]
    return output


def _reference_mxfp8_rows(source: torch.Tensor) -> torch.Tensor:
    """Quantize logical K32 groups to E4M3 with UE8M0 power-of-two scales."""

    m, k = source.shape
    blocks = source.float().reshape(m, k // 32, 32)
    max_abs = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(max_abs > 0, max_abs / 448.0, torch.ones_like(max_abs))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale = torch.pow(torch.tensor(2.0, device=source.device), exponent)
    scale = torch.where(max_abs > 0, scale, torch.ones_like(scale))
    return (
        (blocks / scale)
        .to(torch.float8_e4m3fn)
        .float()
        .mul(scale)
        .reshape(m, k)
    )

@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [3, 4])
def test_coupled_higher_rate_w4a16_matches_transform_reference(
    bits: int,
) -> None:
    """Close uniform K3/K4 payloads through the coupled transform contract."""

    torch.manual_seed(20260810 + bits)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = m = 1
    hidden, intermediate = 512, 128
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
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    source_suh = scales((1, hidden)).contiguous()
    ordinary_rotations = scales((experts, 3 * intermediate)).contiguous()
    coupled_signs = (
        torch.randint(
            0,
            2,
            (experts, 3 * intermediate),
            dtype=torch.int32,
            device=device,
        )
        .mul_(2)
        .sub_(1)
        .to(torch.float16)
    )
    down_svh = scales((1, hidden)).contiguous()
    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        params_dtype=torch.float16,
        fc1_tile_n=128,
        fc2_tile_n=128,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook="sqg_e4m3",
        gate_suh=source_suh,
        up_suh=source_suh,
        intermediate_rotations=ordinary_rotations,
        down_svh=down_svh,
        tile_config=(128, 128, 128, 128),
    )
    assert prepared.trellis is not None
    prepared = replace(
        prepared,
        trellis=replace(
            prepared.trellis,
            coupled_hadamard=True,
            intermediate_rotations=torch.cat(
                (ordinary_rotations, coupled_signs), dim=1
            ).contiguous(),
        ),
    )

    slot0_weights = _reconstruct_native(
        w13[0, 0], codebook="sqg_e4m3"
    ).unsqueeze(0).to(device)
    slot1_weights = _reconstruct_native(
        w13[1, 0], codebook="sqg_e4m3"
    ).unsqueeze(0).to(device)
    down_weights = _reconstruct_native(
        w2[0], codebook="sqg_e4m3"
    ).unsqueeze(0).to(device)
    x = torch.randn((m, hidden), dtype=torch.bfloat16, device=device)
    ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    router_weights = torch.ones((m, topk), dtype=torch.float32, device=device)

    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )
    actual_a16 = run_w4a16_moe(
        x,
        prepared,
        router_weights,
        ids,
        activation="situ",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        route_block_size_m=8,
        intermediate_rotation_scales=prepared.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=source_suh,
        suh_up_table=source_suh,
        svh_table=down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    ).clone()
    reference_a16 = _reference_coupled_decoded(
        x,
        ids,
        router_weights,
        slot0_weights,
        slot1_weights,
        down_weights,
        source_suh,
        ordinary_rotations,
        coupled_signs,
        down_svh,
        fc1_pair_kind=prepared.fc1_trellis_pair_kind,
    )
    relative_a16 = (
        (actual_a16 - reference_a16).norm()
        / reference_a16.norm().clamp_min(1.0e-9)
    )
    cosine_a16 = torch.nn.functional.cosine_similarity(
        actual_a16.flatten(), reference_a16.flatten(), dim=0
    )
    assert float(relative_a16) <= 2.0e-2
    assert float(cosine_a16) >= 0.9999

@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [3, 4])
def test_full_rotation_topk16_route_parallel_sum_matches_reference(bits: int) -> None:
    """Exercise legacy EXL3 K3/K4 through the eight-warp route partition."""
    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = 16
    hidden = intermediate = 128
    tile_config = (64, 128, 64, 128)
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
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((1, hidden)).contiguous()
    up_suh = scales((1, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((1, hidden)).contiguous()
    weights = _prepare_weights(
        w13,
        w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        activation="situ",
        tile_config=tile_config,
    )
    plan = _plan(
        weights,
        max_tokens=1,
        num_topk=topk,
        route_num_experts=experts,
        block_size_m=8,
        device=device,
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    x = (torch.randn((1, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.randperm(experts, device=device).to(torch.int32).view(1, topk)
    router_weights = torch.rand((1, topk), dtype=torch.float32, device=device)
    router_weights /= router_weights.sum(dim=1, keepdim=True)

    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=ids,
    )
    actual = binding.run().clone()
    torch.cuda.synchronize(device)
    reference = _reference_full_rotation(
        x,
        ids,
        router_weights,
        w13,
        w2,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation="situ",
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = binding.run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_planned_full_rotation_matches_reference_and_captures(
    input_dtype: torch.dtype,
    activation: str,
) -> None:
    torch.manual_seed(20260721)
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = 2, 128, 128
    bits = 3
    tile_config = (64, 128, 64, 128)
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
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((1, hidden)).contiguous()
    up_suh = scales((1, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((1, hidden)).contiguous()
    weights = _prepare_weights(
        w13,
        w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        codebook="mcg",
        tile_config=tile_config,
        input_dtype=input_dtype,
        activation=activation,
    )
    prepared = weights.representation_for("w4a16")
    assert prepared.w13.data_ptr() == w13.data_ptr()
    assert prepared.w2.data_ptr() == w2.data_ptr()

    plan = _plan(
        weights,
        max_tokens=2,
        num_topk=2,
        route_num_experts=4,
        block_size_m=8,
        device=device,
    )
    assert plan.caps.w4a16_block_size_m == 8
    assert plan.full_rotation

    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    x = (torch.randn((2, hidden), device=device) * 1.0e-3).to(input_dtype)
    local_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    global_ids = torch.tensor([[1, 3], [3, 1]], dtype=torch.int64, device=device)
    router_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    route_map = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=device)
    output_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32, device=device)
    external_output = torch.empty((2, hidden), dtype=input_dtype, device=device)

    mapped = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=global_ids,
        route_expert_map=route_map,
        output_expert_map=output_map,
        output=external_output,
    )
    mapped_output = fused_moe.run(binding=mapped)
    torch.cuda.synchronize(device)
    assert mapped_output.data_ptr() == external_output.data_ptr()
    mapped_eager = mapped_output.clone()

    identity = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=local_ids,
    )
    identity_output = identity.run()
    torch.cuda.synchronize(device)
    assert identity_output.dtype == torch.float32
    torch.testing.assert_close(
        mapped_eager,
        identity_output.to(input_dtype),
        rtol=0,
        atol=0,
    )

    reference = _reference_full_rotation(
        x,
        local_ids,
        router_weights,
        w13,
        w2,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation=activation,
    )
    relative_error = (mapped_eager - reference).norm() / reference.norm().clamp_min(
        1.0e-9
    )
    cosine = torch.nn.functional.cosine_similarity(
        mapped_eager.float().flatten(), reference.float().flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    mapped = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=global_ids,
        route_expert_map=route_map,
        output_expert_map=output_map,
        output=external_output,
    )
    mapped.run()
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    mapped.run()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated_before

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = mapped.run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_output.data_ptr() == external_output.data_ptr()
    assert torch.allclose(captured_output, mapped_eager, rtol=2.0e-3, atol=2.0e-3)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_full_rotation_reuses_compiled_kernels_across_expert_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact EXL3 layer sizes are runtime data, including the W13 plane stride."""
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    hidden = intermediate = 128
    topk = max_tokens = 2
    route_experts = 4
    bits = 3
    tile_config = (64, 128, 64, 128)
    w4a16_kernel.clear_w4a16_kernel_cache()

    compiled_after_first: tuple[dict[tuple, int], dict[tuple, int]] | None = None
    for experts in (2, 3):
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
            return (0.875 + 0.25 * torch.rand(shape, device=device)).to(
                torch.float16
            )

        gate_suh = scales((experts, hidden)).contiguous()
        up_suh = scales((experts, hidden)).contiguous()
        intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
        down_svh = scales((experts, hidden)).contiguous()
        weights = _prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate_rotations,
            down_svh=down_svh,
            tile_config=tile_config,
        )
        plan = _plan(
            weights,
            max_tokens=max_tokens,
            num_topk=topk,
            route_num_experts=route_experts,
            block_size_m=8,
            device=device,
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        x = (torch.randn((max_tokens, hidden), device=device) * 1.0e-3).to(
            torch.bfloat16
        )
        global_ids = torch.tensor(
            [[0, 3], [2, 1]], dtype=torch.int32, device=device
        )
        expert_map = (
            torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
            if experts == 2
            else torch.tensor([0, 1, 2, 2], dtype=torch.int32, device=device)
        )
        local_ids = expert_map[global_ids]
        router_weights = torch.tensor(
            [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
        )
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=weights,
            topk_weights=router_weights,
            topk_ids=global_ids,
            route_expert_map=expert_map,
            output_expert_map=expert_map,
        )
        if binding.fused_launch is not None:
            assert binding.fused_launch.direct_topk_routes
            assert binding.fused_launch.use_expert_map

        def _route_pack_must_not_run(*args, **kwargs):
            raise AssertionError("mapped decode unexpectedly invoked route packing")

        with monkeypatch.context() as patch:
            patch.setattr(
                w4a16_kernel,
                "pack_topk_routes_by_expert",
                _route_pack_must_not_run,
            )
            actual = fused_moe.run(binding=binding).clone()
        torch.cuda.synchronize(device)
        reference = _reference_full_rotation(
            x,
            local_ids,
            router_weights,
            w13,
            w2,
            gate_suh,
            up_suh,
            intermediate_rotations,
            down_svh,
        )
        relative_error = (actual - reference).norm() / reference.norm().clamp_min(
            1.0e-9
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        )
        assert float(relative_error) <= 2.0e-2
        assert float(cosine) >= 0.999

        if experts == 2:
            # Unmapped, negative, and out-of-range global ids must be rejected
            # inside the fused kernel before weight access. The top-k sum must
            # skip those stale route rows as well.
            sparse_map = torch.tensor(
                [0, -1, 1, -1], dtype=torch.int32, device=device
            )
            invalid_global_ids = torch.tensor(
                [[0, 1], [route_experts + 3, -1]],
                dtype=torch.int32,
                device=device,
            )
            masked_router_weights = router_weights.clone()
            masked_router_weights[0, 1] = 0.0
            masked_router_weights[1].zero_()
            invalid_binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=x,
                experts=weights,
                topk_weights=router_weights,
                topk_ids=invalid_global_ids,
                route_expert_map=sparse_map,
                output_expert_map=sparse_map,
            )
            with monkeypatch.context() as patch:
                patch.setattr(
                    w4a16_kernel,
                    "pack_topk_routes_by_expert",
                    _route_pack_must_not_run,
                )
                invalid_actual = invalid_binding.run().clone()
            invalid_reference = _reference_full_rotation(
                x,
                torch.zeros_like(invalid_global_ids),
                masked_router_weights,
                w13,
                w2,
                gate_suh,
                up_suh,
                intermediate_rotations,
                down_svh,
            )
            invalid_relative_error = (
                (invalid_actual - invalid_reference).norm()
                / invalid_reference.norm().clamp_min(1.0e-9)
            )
            assert float(invalid_relative_error) <= 2.0e-2
            assert torch.count_nonzero(invalid_actual[1]) == 0

            graph = torch.cuda.CUDAGraph()
            with monkeypatch.context() as patch:
                patch.setattr(
                    w4a16_kernel,
                    "pack_topk_routes_by_expert",
                    _route_pack_must_not_run,
                )
                with torch.cuda.graph(graph):
                    captured_invalid = invalid_binding.run()
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(captured_invalid, invalid_actual)

        compiled_now = (
            {
                key: id(value.compiled)
                for key, value in w4a16_kernel._FUSED_CACHE.items()
            },
            {
                key: id(value.compiled)
                for key, value in w4a16_kernel._SUM_CACHE.items()
            },
        )
        if compiled_after_first is None:
            compiled_after_first = compiled_now
            assert compiled_after_first[0]
            assert compiled_after_first[1]
        else:
            assert compiled_now == compiled_after_first


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_qsrt_pdynamic_cache_key_ignores_expert_count() -> None:
    """Profile-ID-5 layer populations are runtime data, like mixed tiers."""
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel

    def make_kernel(
        num_experts: int,
        pair_kind: str = "PDYNAMIC",
        fc2_pair_kind: str | None = None,
    ) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=2,
            hidden_size=256,
            intermediate_size=256,
            num_experts=num_experts,
            top_k=2,
            activation="situ",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=256,
            fc1_tile_k=64,
            fc2_tile_n=256,
            fc2_tile_k=64,
            moe_block_size=8,
            max_m_blocks=4,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=3,
            trellis_codebook="sqg_e4m3",
            fc1_trellis_pair_kind=pair_kind,
            fc2_trellis_pair_kind=(
                pair_kind if fc2_pair_kind is None else fc2_pair_kind
            ),
            schedule_whole_tiles=True,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            broadcast_suh=True,
        )

    compact = make_kernel(27)
    complete = make_kernel(896)

    assert compact.dynamic_num_experts
    assert complete.dynamic_num_experts
    assert compact.__cache_key__ == complete.__cache_key__
    assert compact.fc1.__cache_key__ == complete.fc1.__cache_key__
    assert compact.fc2.__cache_key__ == complete.fc2.__cache_key__

    # Static pair kinds are part of the fused contract (coupled H308 extents
    # run P43/P33 and P43/P44), so a static selection constructs; the fused
    # kernel fails closed only on mismatched dynamic kinds.
    make_kernel(27, "P24")
    make_kernel(27, "P43", "P33")
    with pytest.raises(
        ValueError, match="dynamic fused trellis pair kinds must match"
    ):
        make_kernel(27, "PDYNAMIC", "P33_P43")
    with pytest.raises(
        ValueError, match="dynamic fused trellis pair kinds must match"
    ):
        make_kernel(27, "P24", "PDYNAMIC")


def test_x4t_fc2_direct_uses_bounded_expert_capacity() -> None:
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _w4a16_fc2_direct_expert_capacity,
    )

    # Every possible Kimi layer endpoint population reuses one specialization.
    assert {
        _w4a16_fc2_direct_expert_capacity(num_experts)
        for num_experts in (1, 27, 192, 700, 896, 1024)
    } == {1024}
    assert _w4a16_fc2_direct_expert_capacity(1025) == 2048
    with pytest.raises(ValueError, match="at least one expert"):
        _w4a16_fc2_direct_expert_capacity(0)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
def test_planned_full_rotation_small_m_partial_blocks(m: int) -> None:
    """Small-m coverage at production geometry, with m < plan capacity.

    Motivated by vLLM issue #183, which reported that widening the Trellis window
    (VLLM_EXL3_TRELLIS_MIN_M=1) so an MTP-N draft's m=1..N GEMMs stay on the fused
    path silently corrupts output. m < moe_block_size means the route packer emits
    mostly-padding blocks (at m=3, topk=8 roughly 1-3 real rows of 8), so padding
    masking is load-bearing and was previously untested: the only numerical test on
    this path ran m == max_tokens == 2 with topk=2 and a non-production tile.

    Three oracles:
      * per-token bitwise equality against an m=1 run of the same token. A token's
        output depends only on its own routes, so batching must not change a bit.
      * the scratch arena pre-filled with 0xFF (NaN in fp32/fp16) before every
        bind, so any read of a padding/uninitialised slot surfaces as NaN.
      * plan capacity 32 vs m, i.e. the window-widening regime itself.
    """
    torch.manual_seed(20260726)
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = 32, 1024, 1024
    bits, topk, capacity = 3, 8, 32
    tile_config = (64, 256, 64, 256)

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
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    weights = _prepare_weights(
        w13,
        w2,
        gate_suh=scales((experts, hidden)).contiguous(),
        up_suh=scales((experts, hidden)).contiguous(),
        intermediate_rotations=scales((experts, 3 * intermediate)).contiguous(),
        down_svh=scales((experts, hidden)).contiguous(),
        codebook="mcg",
        tile_config=tile_config,
    )
    plan = _plan(
        weights,
        max_tokens=capacity,
        num_topk=topk,
        route_num_experts=experts,
        block_size_m=8,
        device=device,
    )
    assert plan.caps.w4a16_block_size_m == 8
    assert plan.full_rotation

    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(m)]
    ).to(torch.int32)
    router_weights = torch.rand((m, topk), dtype=torch.float32, device=device)
    router_weights /= router_weights.sum(dim=1, keepdim=True)

    def _run(rows: slice) -> torch.Tensor:
        # NaN-poison the arena so a padding-slot read cannot pass silently.
        scratch.fill_(0xFF)
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x[rows].contiguous(),
            experts=weights,
            topk_weights=router_weights[rows].contiguous(),
            topk_ids=ids[rows].contiguous(),
        )
        assert binding.route_block_size_m == plan.caps.w4a16_block_size_m
        out = fused_moe.run(binding=binding)
        torch.cuda.synchronize(device)
        return out.clone()

    batched = _run(slice(0, m))
    assert not torch.isnan(batched).any(), (
        f"NaN in fused output at m={m}: a padding or uninitialised arena slot was read"
    )

    for row in range(m):
        single = _run(slice(row, row + 1))
        assert not torch.isnan(single).any()
        assert torch.equal(batched[row], single[0]), (
            f"row {row} of an m={m} batch differs from its own m=1 run; a token's "
            "output must not depend on how many other tokens share the batch"
        )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("m", [1, 2, 3])
def test_planned_full_rotation_capture_below_capacity(m: int) -> None:
    """CUDA-graph capture and replay at m strictly below plan capacity.

    This is the surface vLLM #183 actually leaves untested. The existing capture
    test captures at ``m == max_tokens``, and a small-m eager check does not cover
    capture: ``bind`` runs inside the captured region, so the route-pack grid, the
    ``kernel_workspace`` zero-fill and the active-m constants are all baked into
    the graph. If any of those were captured for the wrong m, replay would produce
    stale or mismatched results while eager execution stayed correct -- which is
    exactly a "short prompts pass, long context corrupts" signature once an
    MTP-N draft replays its captured graph.

    Asserts replay matches the eager result for the same inputs bit-for-bit, and
    that a second replay is stable (no dependence on scratch left over from the
    capture pass).
    """
    torch.manual_seed(20260726)
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = 32, 1024, 1024
    bits, topk, capacity = 3, 8, 32
    tile_config = (64, 256, 64, 256)

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
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    weights = _prepare_weights(
        w13,
        w2,
        gate_suh=scales((experts, hidden)).contiguous(),
        up_suh=scales((experts, hidden)).contiguous(),
        intermediate_rotations=scales((experts, 3 * intermediate)).contiguous(),
        down_svh=scales((experts, hidden)).contiguous(),
        codebook="mcg",
        tile_config=tile_config,
    )
    plan = _plan(
        weights,
        max_tokens=capacity,
        num_topk=topk,
        route_num_experts=experts,
        block_size_m=8,
        device=device,
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(m)]
    ).to(torch.int32)
    router_weights = torch.rand((m, topk), dtype=torch.float32, device=device)
    router_weights /= router_weights.sum(dim=1, keepdim=True)

    eager_binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=ids,
    )
    assert eager_binding.route_block_size_m == plan.caps.w4a16_block_size_m
    eager = eager_binding.run().clone()
    torch.cuda.synchronize(device)
    assert not torch.isnan(eager).any()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=weights,
            topk_weights=router_weights,
            topk_ids=ids,
        )
        captured = binding.run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert not torch.isnan(captured).any(), (
        f"NaN after graph replay at m={m} (capacity {capacity})"
    )
    assert torch.equal(captured, eager), (
        f"graph replay at m={m} below capacity {capacity} disagrees with eager; "
        "capture baked constants or a route-pack grid for the wrong m"
    )

    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, eager), (
        f"second replay at m={m} drifted; replay depends on scratch state left "
        "over from the capture pass"
    )
