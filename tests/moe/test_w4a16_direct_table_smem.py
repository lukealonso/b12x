"""The staged direct SQG-XOR-Cheb-T12 table decodes bit-identically.

``B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM=1`` makes the fused W4A16 trellis kernel
stage the 64 KiB precomposed rate table (raw 16-bit window -> E4M3 byte)
instead of the 4 KiB modal staircase and skip the per-weight rank bijection.
Both decode paths produce the same bytes, so the fused kernel output must
match bit for bit across the flag for every supported uniform bitrate and
decode batch size, eagerly and under CUDA-graph capture. The flag is part of
the kernel cache key, so both variants coexist in one process.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel
from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import (
    W4A16FusedMoeKernel,
    compile_w4a16_fused_moe,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    W4A16MixedTrellisKernel,
    compile_mixed_trellis,
)
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

TILE = (128, 128, 128, 128)
FLAG = "B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM"


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major == 12


def _prepared(*, experts: int, hidden: int, intermediate: int, bits: int, seed: int,
              device: torch.device):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)).to(
            torch.float16
        )

    return prepare_trellis256_moe_weights(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=TILE[1],
        fc2_tile_n=TILE[3],
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook="sqg_e4m3",
        gate_suh=scales((experts, hidden)),
        up_suh=scales((experts, hidden)),
        intermediate_rotations=scales((experts, 3 * intermediate)),
        down_svh=scales((experts, hidden)),
        tile_config=TILE,
    )


def _run(x: torch.Tensor, prepared, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
         expert_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, topk = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    buffers = make_w4a16_packed_buffers(
        prepared, m=m, topk=topk, dtype=torch.float16, device=x.device,
        route_num_experts=int(expert_map.numel()), full_rotation=True, block_size_m=8,
    )

    def launch() -> torch.Tensor:
        return run_w4a16_moe(
            x, prepared, topk_weights, topk_ids, activation="silu",
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
            expert_map=expert_map,
            output_expert_map=expert_map,
            route_block_size_m=8,
            intermediate_rotation_scales=prepared.intermediate_rotations,
            full_rotation=True,
            suh_gate_table=prepared.gate_suh,
            suh_up_table=prepared.up_suh,
            svh_table=prepared.down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    eager = launch().clone()
    torch.cuda.synchronize(x.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    graph.replay()
    torch.cuda.synchronize(x.device)
    return eager, captured.clone()


def _compile_common(*, bits: int) -> dict:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(
        size_m=8, hidden_size=128, intermediate_size=128, num_experts=16, top_k=16,
        activation="silu", apply_router_weight_on_input=False, element_dtype="fp16",
        fast_math=True, sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        weight_layout="trellis_t256", w13_layout="trellis_t256_proj", scale_format="e4m3_k32",
        force_tile_config=TILE, trellis_bits=bits, trellis_codebook="sqg_e4m3",
        zero_fc2_output=False, moe_block_size=8, max_m_blocks=8 * 16,
        direct_topk_routes=True, use_expert_map=True, intermediate_rotation=True,
        full_rotation=True,
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("m", [1, 4, 8])
def test_direct_table_matches_modal_table_bitwise(
    monkeypatch: pytest.MonkeyPatch, m: int
) -> None:
    """2-bit trellis pipelines leave room for the 64 KiB table next to the
    GEMM stages; the direct decode must reproduce the modal decode exactly."""
    bits = 2
    device = torch.device("cuda", torch.cuda.current_device())
    experts, topk, hidden, intermediate = 16, 16, 128, 128
    prepared = _prepared(
        experts=experts, hidden=hidden, intermediate=intermediate, bits=bits,
        seed=20260902 + bits, device=device,
    )
    gen = torch.Generator(device="cpu").manual_seed(7 + m)
    x = (torch.randn((m, hidden), generator=gen) * 1.0e-3).to(torch.bfloat16).to(device)
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=gen)[:topk] for _ in range(m)]
    ).to(torch.int32).to(device)
    topk_weights = torch.rand((m, topk), generator=gen).to(device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    expert_map = torch.arange(experts, dtype=torch.int32, device=device)

    monkeypatch.setenv(FLAG, "0")
    assert not w4a16_kernel._sqg_xor_cheb_t12_direct_smem_enabled()
    modal_eager, modal_graph = _run(x, prepared, topk_weights, topk_ids, expert_map)

    monkeypatch.setenv(FLAG, "1")
    assert w4a16_kernel._sqg_xor_cheb_t12_direct_smem_enabled()
    assert compile_w4a16_fused_moe(**_compile_common(bits=bits)).sqg_xor_cheb_t12_direct_smem
    direct_eager, direct_graph = _run(x, prepared, topk_weights, topk_ids, expert_map)

    assert torch.isfinite(modal_eager).all()
    assert torch.count_nonzero(modal_eager).item() > 0
    assert torch.equal(modal_eager, direct_eager)
    assert torch.equal(modal_graph, direct_graph)
    assert torch.equal(modal_eager, modal_graph)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_direct_table_is_part_of_the_compiled_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiling with the flag on yields a launch that staged the 64 KiB table
    (and needs the matching rate slice) and whose shared-memory footprint grew
    by that table; compiling with it off does not."""
    staged = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(FLAG, flag)
        launch = compile_w4a16_fused_moe(**_compile_common(bits=2))
        staged[flag] = (
            bool(launch.sqg_xor_cheb_t12_direct_smem),
            int(launch.shared_memory_bytes),
        )
    assert staged["0"][0] is False
    assert staged["1"][0] is True
    direct_table_bytes = 1 << 16
    assert staged["1"][1] >= direct_table_bytes
    assert staged["1"][1] - staged["0"][1] >= direct_table_bytes - 4096


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [3, 4])
def test_direct_table_falls_back_to_modal_when_pipeline_leaves_no_room(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bits: int
) -> None:
    """3- and 4-bit pipelines hold wider weight stages; with the 128x128 tile
    the 64 KiB table exceeds the 99 KiB opt-in limit, so the launch keeps the
    modal table and reports the fallback once."""
    monkeypatch.setenv(FLAG, "1")
    w4a16_kernel._SQG_XOR_CHEB_T12_DIRECT_SMEM_FALLBACKS_REPORTED.clear()
    with caplog.at_level("WARNING", logger=w4a16_kernel.__name__):
        launch = compile_w4a16_fused_moe(**_compile_common(bits=bits))
    assert not launch.sqg_xor_cheb_t12_direct_smem
    assert launch.shared_memory_bytes < 1 << 16
    fallbacks = [r for r in caplog.records if "direct table does not fit" in r.getMessage()]
    assert len(fallbacks) == 1


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_direct_table_skips_multi_cta_grids(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The grid is sized before the table is added; a two-CTA-per-SM prefill
    grid cannot hold two 64 KiB tables, so those launches keep the modal
    table without a fallback report (the table targets the decode grid)."""
    monkeypatch.setenv(FLAG, "1")
    w4a16_kernel._SQG_XOR_CHEB_T12_DIRECT_SMEM_FALLBACKS_REPORTED.clear()
    common = _compile_common(bits=2)
    # 64-wide N tiles at 16 route rows per block fit two CTAs per SM
    common.update(size_m=512, moe_block_size=16, max_m_blocks=512 * 16 // 16,
                  force_tile_config=(128, 64, 128, 64),
                  direct_topk_routes=False, use_expert_map=False)
    with caplog.at_level("WARNING", logger=w4a16_kernel.__name__):
        launch = compile_w4a16_fused_moe(**common)
    assert launch.blocks_per_sm > 1
    assert not launch.sqg_xor_cheb_t12_direct_smem
    assert not [r for r in caplog.records if "direct table does not fit" in r.getMessage()]


def _uniform_rate_tier(*, bits: int, direct: bool, num_experts: int) -> W4A16FusedMoeKernel:
    """A tier-shaped fused kernel (the argument set ``compile_mixed_trellis``
    builds its driver and tiers from), with the direct table forced on or off."""
    return W4A16FusedMoeKernel(
        size_m=8, hidden_size=128, intermediate_size=128, num_experts=num_experts,
        top_k=4, activation="silu", apply_router_weight_on_input=False,
        zero_fc2_output=False,
        fc1_tile_n=128, fc1_tile_k=128, fc2_tile_n=128, fc2_tile_k=128,
        moe_block_size=8, max_m_blocks=8, fc2_moe_block_size=8,
        fc2_schedule_route_block_factor=1, element_dtype="fp16",
        weight_layout="trellis_t256", scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj", trellis_bits=bits,
        trellis_codebook="sqg_e4m3", intermediate_rotation=True,
        full_rotation=True, rotation_input_dtype="bf16", broadcast_suh=False,
        schedule_whole_tiles=True,
        sqg_xor_cheb_t12_direct_smem=direct,
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_mixed_rate_tiers_keep_the_modal_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mixed-rate kernel stages one 4 KiB modal table for all of its
    tiers, so its tiers are compiled with the direct table off even when the
    environment switch is on: the launch compiles with the same shared-memory
    footprint as with the switch off."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    footprint = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(FLAG, flag)
        launch = compile_mixed_trellis(
            size_m=8, hidden_size=128, intermediate_size=128,
            tier0_num_experts=2, tier1_num_experts=2, top_k=4, max_m_blocks=8,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(props.shared_memory_per_block_optin),
            force_tile_config=TILE, tier0_bits=2, tier1_bits=3,
            trellis_codebook="sqg_e4m3",
        )
        footprint[flag] = int(launch.shared_memory_bytes)
    assert footprint["0"] == footprint["1"]
    assert footprint["1"] < 1 << 16


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_mixed_rate_composition_rejects_direct_table_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tier compiled for the direct table would read the modal bytes the
    mixed kernel stages as if they were the 64 KiB rate slice, so composing
    such a tier is refused up front."""
    monkeypatch.setenv(FLAG, "1")
    # driver (all experts, rate of tier 0), tier 0, tier 1; the 2-bit tier is
    # the one whose pipeline leaves room for the direct table
    modal = [
        _uniform_rate_tier(bits=bits, direct=False, num_experts=experts)
        for bits, experts in ((2, 4), (2, 2), (3, 2))
    ]
    assert not any(k.sqg_xor_cheb_t12_direct_smem for k in modal)
    W4A16MixedTrellisKernel(driver=modal[0], tier0=modal[1], tier1=modal[2])

    direct_tier = _uniform_rate_tier(bits=2, direct=True, num_experts=2)
    assert direct_tier.sqg_xor_cheb_t12_direct_smem
    with pytest.raises(ValueError, match="sqg_xor_cheb_t12_direct_smem=False"):
        W4A16MixedTrellisKernel(driver=modal[0], tier0=modal[1], tier1=direct_tier)
