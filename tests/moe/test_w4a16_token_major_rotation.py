"""CPU checks of the token-major full-rotation A layout (fused W4A16 MoE).

With the coupled Hadamard and one broadcast ``suh`` row the rotated A row of
a route depends on its token only. The fused kernel then writes one row per
token and FC1 gathers rows through ``route // top_k`` (``route_major_a``
off); with per-expert ``suh`` rows, or with the switch off, the route-major
layout stays. This module checks the construction-time wiring without a
GPU; the row arithmetic itself is shared by both layouts
(``_rotate_coupled_row``).
"""

from __future__ import annotations

import pytest

kernel = pytest.importorskip("b12x.moe._shared.kernels.w4a16.kernel")


def _served_fused(monkeypatch: pytest.MonkeyPatch, *, broadcast_suh: bool, token_major: str):
    monkeypatch.setenv("B12X_W4A16_TOKEN_MAJOR_ROTATION", token_major)
    monkeypatch.delenv("B12X_W4A16_SMALL_M_SPLITK", raising=False)
    return kernel.W4A16FusedMoeKernel(
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
        moe_block_size=48,
        max_m_blocks=1536,
        element_dtype="fp16",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        trellis_codebook="sqg_xor_cheb_t12",
        intermediate_rotation=True,
        full_rotation=True,
        coupled_hadamard=True,
        rotation_input_dtype="bf16",
        broadcast_suh=broadcast_suh,
    )


def test_broadcast_suh_selects_token_major_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fused = _served_fused(monkeypatch, broadcast_suh=True, token_major="1")
    assert fused.token_major_rotation
    assert fused.dual_a
    assert not fused.fc1.route_major_a
    assert fused.fc2.top_k == 1 and not fused.fc2.route_major_a
    assert fused.token_major_rotation in fused.__cache_key__


def test_switch_off_keeps_route_major_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fused = _served_fused(monkeypatch, broadcast_suh=True, token_major="0")
    assert not fused.token_major_rotation
    assert fused.fc1.route_major_a
    on = _served_fused(monkeypatch, broadcast_suh=True, token_major="1")
    assert on.__cache_key__ != fused.__cache_key__


def test_per_expert_suh_keeps_route_major_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fused = _served_fused(monkeypatch, broadcast_suh=False, token_major="1")
    assert not fused.token_major_rotation
    assert fused.fc1.route_major_a


def test_token_rows_cover_every_route_row() -> None:
    """Row bookkeeping of the two layouts on the served capacity.

    Route ids are ``token * top_k + k`` (the rotation derives the token as
    ``route // top_k``); the token-major layout writes ``active_m`` rows and
    FC1 reads row ``route // top_k`` for every valid route, which is the row
    the route-major layout writes at index ``route`` from the same token.
    """

    top_k, active_m = 16, 4608
    routes = range(active_m * top_k)
    token_of_route = {r: r // top_k for r in routes}
    assert set(token_of_route.values()) == set(range(active_m))
    for route in (0, 15, 16, 4608 * 16 - 1):
        assert token_of_route[route] == route // top_k
    # Bytes written by the rotation phase per launch (fp16, hidden 3584).
    route_major_bytes = active_m * top_k * 3584 * 2
    token_major_bytes = active_m * 3584 * 2
    assert route_major_bytes == 16 * token_major_bytes == 528_482_304
