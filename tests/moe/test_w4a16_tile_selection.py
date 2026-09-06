from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.kernels.w4a16 import kernel as kernel_module

from b12x.moe._shared.kernels.w4a16.kernel import (
    _DEFAULT_MAX_SHARED_MEM,
    _DEVICE_MAX_REG_BYTES,
    _W4A16_MAX_LARGE_M_ACC_SETS,
    W4A16FusedMoeKernel,
    W4A16GemmKernel,
    _candidate_tile_fits,
    _covering_count,
    _select_tile_config,
    _shared_memory_footprint,
    _w4a16_accumulator_regs_per_thread,
    _w4a16_b_unit_bytes,
    compile_w4a16_fused_moe,
)


def _fits(
    *,
    tile_k: int,
    tile_n: int,
    cta_threads: int,
    allow_qualified_fc2_tile: bool = False,
) -> bool:
    return _candidate_tile_fits(
        problem_n=4096,
        problem_k=512,
        cta_m_blocks=1,
        tile_n=tile_n,
        tile_k=tile_k,
        cta_threads=cta_threads,
        max_shared_mem=1 << 30,
        scale_format="e8m0_k32",
        weight_layout="modelopt",
        weight_bits=4,
        allow_qualified_fc2_tile=allow_qualified_fc2_tile,
    )


def test_wave_balanced_fc2_tile_is_valid_as_an_explicit_pin() -> None:
    assert _fits(
        tile_k=32,
        tile_n=512,
        cta_threads=256,
        allow_qualified_fc2_tile=True,
    )


def test_wave_balanced_fc2_tile_is_rejected_for_fc1() -> None:
    assert not _fits(tile_k=32, tile_n=512, cta_threads=256)


def test_other_sub64_k_tiles_remain_unsupported() -> None:
    assert not _fits(tile_k=32, tile_n=256, cta_threads=128)
    assert not _fits(tile_k=16, tile_n=512, cta_threads=128)


# The served Kimi-K3 QSRT trellis launch: 2-bpw SQG-XOR-Cheb-T12 experts,
# E4M3 K/32 scales, FC1 K=3584 -> N=768 (384-channel extent, gate+up), FC2
# K=384 -> N=3584, one pinned 128x128 CTA tile with 256 threads, and the
# SM120 opt-in limit of 101,376 bytes minus the planner's 512-byte margin.
_K3_TILE = dict(tile_n=128, tile_k=128, cta_threads=256)
_K3_SMEM_LIMIT = _DEFAULT_MAX_SHARED_MEM - 512


def _k3_footprint(block: int, **kwargs) -> int:
    return _shared_memory_footprint(
        cta_m_blocks=_covering_count(block, 16),
        tile_n=128,
        tile_k=128,
        scale_format="e4m3_k32",
        weight_layout="trellis3_t256",
        **kwargs,
    )


def _k3_fits(block: int, *, problem_n: int, problem_k: int, **kwargs) -> bool:
    return _candidate_tile_fits(
        problem_n=problem_n,
        problem_k=problem_k,
        cta_m_blocks=_covering_count(block, 16),
        max_shared_mem=_K3_SMEM_LIMIT,
        scale_format="e4m3_k32",
        weight_layout="trellis3_t256",
        **_K3_TILE,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("weight_layout", "trellis_bits", "pair_kind", "rate_axis", "expected"),
    [
        ("packed", 3, None, None, 16),
        ("modelopt", 3, None, None, 16),
        ("trellis3_t256", 2, None, None, 8),
        ("trellis3_t256", 3, None, None, 12),
        ("trellis3_t256", 4, None, None, 16),
        ("trellis3_t256", 6, None, None, 24),
        ("trellis3_t256", 3, "P33", "n", 12),
        ("trellis3_t256", 3, "P24", "n", 12),
        ("trellis3_t256", 3, "P43", "n", 14),
        ("trellis3_t256", 3, "P44", "n", 16),
        ("trellis3_t256", 3, "P33_P43", "n", 14),
        ("trellis3_t256", 3, "PDYNAMIC", "n", 12),
        ("trellis3_t256", 3, "P33_P43", "k", 16),
        ("trellis3_t256", 3, "P24", "k", 16),
    ],
)
def test_b_unit_bytes_matches_the_kernel_staging_layout(
    weight_layout: str,
    trellis_bits: int,
    pair_kind: str | None,
    rate_axis: str | None,
    expected: int,
) -> None:
    assert (
        _w4a16_b_unit_bytes(
            weight_layout=weight_layout,
            trellis_bits=trellis_bits,
            trellis_pair_kind=pair_kind,
            trellis_rate_axis=rate_axis,
        )
        == expected
    )


def test_footprint_b_stage_uses_the_real_trellis_width() -> None:
    """The estimate must size the B stage exactly as the kernel stages it."""
    two_bpw = _w4a16_b_unit_bytes(weight_layout="trellis3_t256", trellis_bits=2)
    assert two_bpw == 8
    # Four 128x128 B stages: 32,768 B at 4 bpw, 16,384 B at 2 bpw. The
    # epilogue reduction buffer (cta_m x 136 x 2 B) aliases the B region, so
    # from block 64 up it bounds the region and absorbs part of the saving.
    for block, saving in ((48, 16_384), (64, 15_360), (96, 6_656)):
        contract = _k3_footprint(block, weight_bits=4)
        real = _k3_footprint(block, b_unit_bytes=two_bpw)
        assert contract - real == saving, block
    # Passing the 4-bpw unit width reproduces the weight_bits estimate.
    assert _k3_footprint(64, b_unit_bytes=16) == _k3_footprint(64, weight_bits=4)
    assert _k3_footprint(48, weight_bits=4) == 86_784
    assert _k3_footprint(64, weight_bits=4) == 103_424
    assert _k3_footprint(96, weight_bits=4) == 136_704
    assert _k3_footprint(48, b_unit_bytes=8) == 70_400
    assert _k3_footprint(64, b_unit_bytes=8) == 88_064
    assert _k3_footprint(96, b_unit_bytes=8) == 130_048


@pytest.mark.parametrize(
    ("problem_n", "problem_k"),
    [(768, 3584), (512, 3584), (3584, 384), (3584, 256)],
)
def test_route_block_64_fits_only_with_the_real_trellis_width(
    problem_n: int, problem_k: int
) -> None:
    two_bpw = _w4a16_b_unit_bytes(weight_layout="trellis3_t256", trellis_bits=2)
    for block, contract_fits, real_fits in ((48, True, True), (64, False, True), (96, False, False)):
        assert (
            _k3_fits(block, problem_n=problem_n, problem_k=problem_k, weight_bits=4)
            is contract_fits
        ), block
        assert (
            _k3_fits(
                block,
                problem_n=problem_n,
                problem_k=problem_k,
                weight_bits=4,
                b_unit_bytes=two_bpw,
            )
            is real_fits
        ), block


def test_fit_width_does_not_change_residency_planning() -> None:
    """``fit_b_unit_bytes`` only gates the fit; the tile choice and planned
    blocks per SM come from ``weight_bits`` as before."""
    two_bpw = _w4a16_b_unit_bytes(weight_layout="trellis3_t256", trellis_bits=2)
    for block in (32, 48, 64):
        for problem_m, problem_n, problem_k, top_k in (
            (4608, 768, 3584, 16),
            (4608 * 16, 3584, 384, 1),
        ):
            common = dict(
                problem_m=problem_m,
                problem_n=problem_n,
                problem_k=problem_k,
                top_k=top_k,
                moe_block_size=block,
                sms=188,
                max_shared_mem=_DEFAULT_MAX_SHARED_MEM,
                scale_format="e4m3_k32",
                weight_layout="trellis3_t256",
                weight_bits=4,
            )
            assert _select_tile_config(**common) == _select_tile_config(
                **common, fit_b_unit_bytes=two_bpw
            )


def _k3_gemm_kernel(block: int, *, size_n: int, size_k: int, **kwargs) -> W4A16GemmKernel:
    return W4A16GemmKernel(
        size_m=4608,
        size_n=size_n,
        size_k=size_k,
        num_experts=896,
        top_k=16,
        mul_topk_weights=False,
        tile_n=128,
        tile_k=128,
        moe_block_size=block,
        max_m_blocks=2000,
        element_dtype="fp16",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        trellis_bits=2,
        schedule_whole_tiles=True,
        dynamic_num_experts=True,
        **kwargs,
    )


def test_estimate_covers_the_real_kernel_layout_at_block_64() -> None:
    """The planner estimate stays an upper bound of the kernel's own layout."""
    two_bpw = _w4a16_b_unit_bytes(weight_layout="trellis3_t256", trellis_bits=2)
    for block, expected_real in ((48, 68_352), (64, 86_016)):
        fc1 = _k3_gemm_kernel(
            block,
            size_n=768,
            size_k=3584,
            w13_layout="trellis3_t256_proj",
            dual_a=True,
            route_major_a=True,
        )
        assert fc1.b_unit_bytes == two_bpw
        assert fc1.shared_words * 4 == expected_real
        assert fc1.shared_words * 4 <= _k3_footprint(block, b_unit_bytes=two_bpw)
        # One CTA per SM at the 255-register cap: the grid contract is unchanged.
        assert fc1.blocks_per_sm == 1


def test_fused_kernel_accepts_block_64_with_the_modal_table(monkeypatch) -> None:
    """Route block 64 leaves room for the 4 KiB T12 table: 90,128 <= 101,376."""
    monkeypatch.setenv("B12X_SQG_XOR_CHEB_T12_SMEM", "1")
    monkeypatch.setenv("B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM", "1")
    fused = W4A16FusedMoeKernel(
        size_m=4608,
        hidden_size=3584,
        intermediate_size=384,
        num_experts=896,
        top_k=16,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        fc1_tile_n=128,
        fc1_tile_k=128,
        fc2_tile_n=128,
        fc2_tile_k=128,
        moe_block_size=64,
        max_m_blocks=2000,
        element_dtype="fp16",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        intermediate_rotation=True,
        full_rotation=True,
        coupled_hadamard=True,
        rotation_input_dtype="bf16",
    )
    if not torch.cuda.is_available():
        assert fused.fc1.max_shared_mem == _DEFAULT_MAX_SHARED_MEM
    assert fused.shared_words * 4 + 16 == 90_128
    assert fused.shared_words * 4 + 16 <= fused.fc1.max_shared_mem
    # The 64 KiB direct table still does not fit next to a prefill pipeline.
    assert fused.sqg_xor_cheb_t12_smem
    assert not fused.sqg_xor_cheb_t12_direct_smem
    assert fused.blocks_per_sm == 1


@pytest.mark.parametrize(
    ("block", "accepted"),
    [(32, True), (48, True), (64, True), (96, False)],
)
def test_fused_planner_admits_block_64_at_the_pinned_k3_tile(
    block: int, accepted: bool, monkeypatch
) -> None:
    """``compile_w4a16_fused_moe`` reaches the kernel (cache miss) for blocks
    whose pinned 128x128 tile fits the real layout and rejects the rest in the
    ``force_tile_config`` fit check before constructing any kernel."""
    monkeypatch.setenv("B12X_SQG_XOR_CHEB_T12_SMEM", "1")
    kwargs = dict(
        size_m=4608,
        hidden_size=3584,
        intermediate_size=384,
        num_experts=896,
        top_k=16,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block,
        max_m_blocks=2000,
        element_dtype="fp16",
        sms=188,
        max_shared_mem=_DEFAULT_MAX_SHARED_MEM,
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        force_tile_config=(128, 128, 128, 128),
        intermediate_rotation=True,
        full_rotation=True,
        coupled_hadamard=True,
        rotation_input_dtype="bf16",
        _require_cached=True,
    )
    if accepted:
        # Past every planner gate; the uncompiled launch is reported as such.
        with pytest.raises(RuntimeError, match="not resolved for CUDA graph capture"):
            compile_w4a16_fused_moe(**kwargs)
    else:
        # Block 96 needs six 16-row accumulator sets; the large-M schedule
        # carries four. The serving planner reads the ValueError as "try a
        # narrower route block".
        with pytest.raises(ValueError, match="accumulator sets"):
            compile_w4a16_fused_moe(**kwargs)


# The register budget of one route block: the large-M schedule keeps every
# warp's fp32 partial of the whole CTA tile live across the K loop, so the
# accumulators alone claim a fixed share of the SM register file.


@pytest.mark.parametrize(
    ("block", "tile_n", "expected"),
    [
        # The pinned Kimi-K3 tile: 2 N-groups, so each warp holds
        # cta_m_blocks*16 rows by 64 columns -> 32 fp32 per m-block.
        (16, 128, 32),
        (32, 128, 64),
        (48, 128, 96),
        (64, 128, 128),
        (96, 128, 192),
        (128, 128, 256),
        # A 64-wide tile has one N-group, so a warp covers all 64 columns:
        # the same 32 fp32 per m-block.
        (48, 64, 96),
        # A 256-wide tile has four N-groups of 64 columns each: unchanged too.
        (48, 256, 96),
    ],
)
def test_accumulator_registers_per_thread(
    block: int, tile_n: int, expected: int
) -> None:
    assert (
        _w4a16_accumulator_regs_per_thread(
            cta_m_blocks=_covering_count(block, 16), tile_n=tile_n
        )
        == expected
    )


def test_pinned_k3_tile_accumulators_against_the_sm_budget() -> None:
    """Route block 48 already claims 37.6 % of the SM120 register file."""
    per_thread = _w4a16_accumulator_regs_per_thread(cta_m_blocks=3, tile_n=128)
    assert per_thread == 96
    # One 256-thread CTA per SM; _DEVICE_MAX_REG_BYTES is the SM register file.
    share = per_thread * 256 * 4 / _DEVICE_MAX_REG_BYTES
    assert round(share, 3) == 0.376
    # Block 96 would need 192 of the 255 registers a thread may hold, leaving
    # 63 for the kernel body; the compiled block-64 kernel uses 245 registers
    # with 128 accumulators, so the body needs far more than that.
    assert _w4a16_accumulator_regs_per_thread(cta_m_blocks=6, tile_n=128) == 192


def _k3_fc1_kernel(block: int) -> W4A16GemmKernel:
    return _k3_gemm_kernel(
        block,
        size_n=768,
        size_k=3584,
        w13_layout="trellis3_t256_proj",
        dual_a=True,
        route_major_a=True,
    )


@pytest.mark.parametrize("block", [80, 96, 128])
def test_route_blocks_above_four_m_blocks_are_rejected(
    block: int, monkeypatch
) -> None:
    """A fifth 16-row m-block has no accumulator set of its own.

    The allowed-route-size tuple already stops at 64, so this widens it to
    prove the kernel refuses on its own terms instead of silently folding rows
    64 and up into the fourth accumulator.
    """
    assert _covering_count(block, 16) > _W4A16_MAX_LARGE_M_ACC_SETS
    assert block not in kernel_module._ALLOWED_ROUTED_SIZES
    monkeypatch.setattr(
        kernel_module,
        "_ALLOWED_ROUTED_SIZES",
        (*kernel_module._ALLOWED_ROUTED_SIZES, block),
    )
    with pytest.raises(ValueError, match="accumulator sets"):
        _k3_fc1_kernel(block)


def test_allowed_route_sizes_stay_within_the_accumulator_sets() -> None:
    """The advertised route blocks must all be computable."""
    for size in kernel_module._ALLOWED_ROUTED_SIZES:
        assert _covering_count(size, 16) <= _W4A16_MAX_LARGE_M_ACC_SETS, size


@pytest.mark.parametrize("block", [16, 32, 48, 64])
def test_route_blocks_within_the_accumulator_sets_are_accepted(block: int) -> None:
    kernel = _k3_fc1_kernel(block)
    assert kernel.cta_m_blocks == _covering_count(block, 16)
    assert kernel.cta_m_blocks <= _W4A16_MAX_LARGE_M_ACC_SETS


def test_block_96_shared_memory_exceeds_the_sm120_limit_at_every_stage_count() -> None:
    """Even before the register budget, the pinned tile does not fit.

    The estimate stages ``_STAGES`` A tiles of ``96 x 128`` fp16 (24,576 B
    each), so the four-stage pipeline the kernel runs needs 130,048 B against
    the 101,376 B opt-in limit, and no stage count above two fits once the
    4 KiB modal trellis table is added.
    """
    two_bpw = _w4a16_b_unit_bytes(weight_layout="trellis3_t256", trellis_bits=2)
    assert _k3_footprint(96, b_unit_bytes=two_bpw) > _DEFAULT_MAX_SHARED_MEM
    # The kernel's own layout, recomputed for a hypothetical stage count:
    # 1,728 int4 of block metadata plus the aliased B/reduction region, and
    # 1,568 int4 per stage (1,536 for A, 32 for the scale stage).
    modal_table = 4096
    for stages, fits in ((4, False), (3, False), (2, True)):
        layout = (1_728 + stages * 1_568) * 16 + modal_table + 16
        assert (layout <= _DEFAULT_MAX_SHARED_MEM) is fits, stages
