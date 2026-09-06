"""The W4A16 decode-launch knobs: pipeline stages and persistent CTAs per SM.

Both knobs change no arithmetic order (k-tiles keep their pipe assignment
``i % stages`` and fold order; the small-M work loop is grid-agnostic); they
trade shared memory and barrier participants for latency hiding. These
tests pin their parsing, bounds and effect on the launch geometry without a
device.
"""

from __future__ import annotations

import importlib
import os

import pytest


def _reload_kernel(monkeypatch, **env):
    for key in ("B12X_W4A16_STAGES", "B12X_W4A16_SMALL_M_BLOCKS_PER_SM"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    import b12x.moe._shared.kernels.w4a16.kernel as kernel

    return importlib.reload(kernel)


@pytest.mark.parametrize("stages", [2, 3, 4, 6, 8])
def test_pipeline_stage_knob_sets_the_module_constant(monkeypatch, stages) -> None:
    kernel = _reload_kernel(monkeypatch, B12X_W4A16_STAGES=stages)
    assert kernel._STAGES == stages


@pytest.mark.parametrize("stages", [1, 9, 0])
def test_pipeline_stage_knob_rejects_out_of_range(monkeypatch, stages) -> None:
    with pytest.raises(ValueError, match="B12X_W4A16_STAGES"):
        _reload_kernel(monkeypatch, B12X_W4A16_STAGES=stages)


def test_default_is_the_served_geometry(monkeypatch) -> None:
    kernel = _reload_kernel(monkeypatch)
    assert kernel._STAGES == 4
    assert kernel._w4a16_small_m_blocks_per_sm() == 1


def test_small_m_launch_is_pinned_to_one_cta_per_sm_by_default(monkeypatch) -> None:
    kernel = _reload_kernel(monkeypatch)
    blocks = kernel._determine_blocks_per_sm(
        problem_m=4, problem_n=768, top_k=16, cta_threads=256, cta_m_blocks=1,
        tile_n=128, tile_k=128, uses_m_block_8=True, sms=188, max_shared_mem=101_376,
        scale_format="e4m3_k32", weight_layout="trellis3_t256", weight_bits=2,
    )
    assert blocks == 1


def test_small_m_blocks_per_sm_knob_raises_the_pin_within_the_resource_limit(monkeypatch) -> None:
    kernel = _reload_kernel(monkeypatch, B12X_W4A16_SMALL_M_BLOCKS_PER_SM=2)
    common = dict(
        problem_m=4, problem_n=768, top_k=16, cta_threads=256, cta_m_blocks=1,
        tile_n=128, tile_k=128, uses_m_block_8=True, sms=188,
        scale_format="e4m3_k32", weight_layout="trellis3_t256", weight_bits=2,
    )
    assert kernel._determine_blocks_per_sm(max_shared_mem=101_376, **common) == 2
    # A shared-memory limit that holds one footprint only keeps the pin at one.
    footprint = kernel._shared_memory_footprint(
        cta_m_blocks=1, tile_n=128, tile_k=128, scale_format="e4m3_k32",
        weight_layout="trellis3_t256", weight_bits=2,
    )
    assert kernel._determine_blocks_per_sm(max_shared_mem=footprint + 1536, **common) == 1


@pytest.mark.parametrize("value", [0, 5])
def test_small_m_blocks_per_sm_knob_rejects_out_of_range(monkeypatch, value) -> None:
    monkeypatch.setenv("B12X_W4A16_SMALL_M_BLOCKS_PER_SM", str(value))
    import b12x.moe._shared.kernels.w4a16.kernel as kernel

    with pytest.raises(ValueError, match="B12X_W4A16_SMALL_M_BLOCKS_PER_SM"):
        kernel._w4a16_small_m_blocks_per_sm()
