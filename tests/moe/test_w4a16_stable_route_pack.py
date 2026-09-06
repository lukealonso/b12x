"""Stable (ascending) W4A16 route packing against a pure-Python reference.

The kernels run on CUDA when a GPU is present and under the Triton
interpreter on CPU when ``TRITON_INTERPRET=1`` is set before ``triton`` is
imported; otherwise the kernel tests skip. The interpreter emulates
``tl.sort`` at roughly 0.2 s per program, so the shapes below keep the expert
count small except for one served-shape case (896 experts); the whole module
takes about 15 minutes under the interpreter and seconds on a GPU.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

import b12x.moe._shared.kernels.w4a16.route_pack as route_pack_module
from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
from b12x.moe._shared.kernels.w4a16.route_pack import pack_topk_routes_by_expert

_STABLE_MIN_ROUTES = route_pack_module._STABLE_SORT_MIN_ROUTES
_SORT_SMALL = route_pack_module._STABLE_SEGMENT_SORT_SMALL
_SORT_LARGE = route_pack_module._STABLE_SEGMENT_SORT_LARGE
_TOP_K = 16


def _kernel_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    pytest.skip("route-pack kernels need CUDA or TRITON_INTERPRET=1")


def reference_stable_route_pack(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the stable packed layout as plain host tensors.

    Contract shared by ``_pack_topk_routes_stable_kernel`` and the segment
    sort: expert segments in expert order, each padded to ``block_size``;
    inside a segment the live routes (flat ``topk_ids`` indices) ascend and
    the padding slots hold ``numel``; ``block_expert_ids`` names each block's
    expert and ``-1`` beyond the packed blocks; ``packed_route_count`` is the
    padded slot total. Slots beyond the packed count are ``numel``.
    """
    raw_ids = topk_ids.detach().cpu().reshape(-1).tolist()
    numel = len(raw_ids)
    host_map = None if expert_map is None else expert_map.detach().cpu().tolist()
    routes_by_expert: list[list[int]] = [[] for _ in range(num_experts)]
    for route, raw in enumerate(raw_ids):
        if raw < 0 or raw >= num_experts:
            continue
        expert = raw if host_map is None else host_map[raw]
        if expert < 0 or expert >= num_experts:
            continue
        routes_by_expert[expert].append(route)
    _, max_routes, max_blocks = route_pack_capacity(
        numel, block_size, num_experts, topk=int(topk_ids.shape[-1])
    )
    packed = [numel] * max_routes
    blocks = [-1] * max_blocks
    cursor = 0
    for expert, routes in enumerate(routes_by_expert):
        for rank, route in enumerate(routes):
            packed[cursor + rank] = route
        padded_blocks = math.ceil(len(routes) / block_size)
        for block in range(padded_blocks):
            blocks[cursor // block_size + block] = expert
        cursor += padded_blocks * block_size
    return (
        torch.tensor(packed, dtype=torch.int32),
        torch.tensor(blocks, dtype=torch.int32),
        torch.tensor([cursor], dtype=torch.int32),
    )


def _workspaces(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> dict[str, torch.Tensor]:
    _, max_routes, max_blocks = route_pack_capacity(
        int(topk_ids.numel()), block_size, num_experts, topk=int(topk_ids.shape[-1])
    )
    device = topk_ids.device
    return {
        "packed_route_indices": torch.full(
            (max_routes,), -7, dtype=torch.int32, device=device
        ),
        "block_expert_ids": torch.full((max_blocks,), -7, dtype=torch.int32, device=device),
        "packed_route_count": torch.empty(1, dtype=torch.int32, device=device),
        "expert_offsets": torch.empty(num_experts + 1, dtype=torch.int32, device=device),
        "expert_counts": torch.empty(num_experts, dtype=torch.int32, device=device),
    }


def _run(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    expert_map: torch.Tensor | None,
    scan: bool,
    monkeypatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    monkeypatch.setenv("B12X_W4A16_STABLE_ROUTE_PACK", "1")
    monkeypatch.setenv("B12X_W4A16_STABLE_ROUTE_PACK_SCAN", "1" if scan else "0")
    workspaces = _workspaces(topk_ids, block_size, num_experts)
    packed, blocks, count = pack_topk_routes_by_expert(
        topk_ids, block_size, num_experts, expert_map=expert_map, **workspaces
    )
    if topk_ids.is_cuda:
        torch.cuda.synchronize()
    return packed.cpu().clone(), blocks.cpu().clone(), count.cpu().clone()


def _rows_per_expert(
    topk_ids: torch.Tensor, num_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    ids = topk_ids.cpu().reshape(-1).to(torch.int64)
    if expert_map is not None:
        ids = expert_map.cpu().to(torch.int64)[ids]
    return torch.bincount(ids[ids >= 0], minlength=num_experts)


def _check_against_reference_and_scan(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None,
    monkeypatch,
) -> None:
    """The segment sort must reproduce both the Python reference and the
    sequential scan kernel (the same layout by two independent routes)."""
    assert topk_ids.numel() >= _STABLE_MIN_ROUTES
    expected = reference_stable_route_pack(topk_ids, block_size, num_experts, expert_map)
    sorted_result = _run(
        topk_ids, block_size, num_experts, expert_map=expert_map, scan=False, monkeypatch=monkeypatch
    )
    scanned_result = _run(
        topk_ids, block_size, num_experts, expert_map=expert_map, scan=True, monkeypatch=monkeypatch
    )
    for name, got, want, legacy in zip(
        ("packed_route_indices", "block_expert_ids", "packed_route_count"),
        sorted_result,
        expected,
        scanned_result,
        strict=True,
    ):
        assert torch.equal(got, want), name
        assert torch.equal(got, legacy), name


def _distinct_topk(weights: torch.Tensor, tokens: int, topk: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.multinomial(
        weights.unsqueeze(0).repeat(tokens, 1), topk, replacement=False, generator=generator
    ).to(torch.int32)


def _uniform(tokens: int, topk: int, num_experts: int, seed: int) -> torch.Tensor:
    return _distinct_topk(torch.ones(num_experts), tokens, topk, seed)


def _zipf(tokens: int, topk: int, num_experts: int, seed: int, s: float = 1.2) -> torch.Tensor:
    weights = torch.arange(1, num_experts + 1, dtype=torch.float32).pow(-s)
    return _distinct_topk(weights, tokens, topk, seed)


def _sparse(tokens: int, topk: int, num_experts: int, live: int, seed: int) -> torch.Tensor:
    """Only ``live`` experts receive routes; the rest stay empty."""
    weights = torch.zeros(num_experts)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights[torch.randperm(num_experts, generator=generator)[:live]] = 1.0
    return _distinct_topk(weights, tokens, topk, seed + 1)


def _pinned_expert(
    tokens: int, topk: int, num_experts: int, seed: int, expert: int = 5
) -> torch.Tensor:
    """``expert`` is in every token's top-k, so its segment has ``tokens``
    routes; the other slots are distinct experts drawn from the rest."""
    weights = torch.ones(num_experts)
    weights[expert] = 0.0
    others = _distinct_topk(weights, tokens, topk - 1, seed)
    return torch.cat([torch.full((tokens, 1), expert, dtype=torch.int32), others], dim=1)


def test_reference_layout_matches_the_documented_contract() -> None:
    topk_ids = torch.tensor([[0, 3, 1], [3, 0, 9], [1, -1, 3]], dtype=torch.int32)
    packed, blocks, count = reference_stable_route_pack(topk_ids, 2, 4)
    # expert 0: routes 0, 4; expert 1: routes 2, 6; expert 3: routes 1, 3, 8.
    assert count.tolist() == [8]
    assert packed[:8].tolist() == [0, 4, 2, 6, 1, 3, 8, 9]
    assert blocks[:4].tolist() == [0, 1, 3, 3]
    assert bool((packed[8:] == 9).all()) and bool((blocks[4:] == -1).all())


@pytest.mark.parametrize("block_size", [8, 48, 64])
@pytest.mark.parametrize("routing", ["uniform", "zipf", "sparse", "mapped"])
def test_segment_sort_matches_reference_and_scan(
    block_size: int, routing: str, monkeypatch
) -> None:
    """Random, skewed, empty-expert and expert-mapped routings at 128 experts."""
    device = _kernel_device()
    num_experts = 128
    seed = 20260906 + block_size
    expert_map = None
    if routing == "uniform":
        ids = _uniform(_STABLE_MIN_ROUTES // _TOP_K, _TOP_K, num_experts, seed)
    elif routing == "zipf":
        ids = _zipf(400, _TOP_K, num_experts, seed)
    elif routing == "sparse":
        ids = _sparse(320, _TOP_K, num_experts, 40, seed)
    else:
        ids = _uniform(320, _TOP_K, num_experts, seed)
        # Every odd raw id is dropped (-1); even ids compact onto 0..63.
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
        expert_map[::2] = torch.arange(num_experts // 2, dtype=torch.int32)
        expert_map = expert_map.to(device)
    topk_ids = ids.to(device=device)
    rows = _rows_per_expert(topk_ids, num_experts, expert_map)
    if routing in ("sparse", "mapped"):
        assert int((rows == 0).sum()) > 0
    if routing == "zipf":
        # The most popular experts exceed the small sorting width.
        assert _SORT_SMALL < int(rows.max()) <= _SORT_LARGE
    _check_against_reference_and_scan(topk_ids, block_size, num_experts, expert_map, monkeypatch)


def test_segment_sort_accepts_int64_route_ids(monkeypatch) -> None:
    device = _kernel_device()
    num_experts = 64
    ids = _uniform(_STABLE_MIN_ROUTES // _TOP_K, _TOP_K, num_experts, 7)
    topk_ids = ids.to(device=device, dtype=torch.int64)
    _check_against_reference_and_scan(topk_ids, 48, num_experts, None, monkeypatch)


@pytest.mark.parametrize(
    "segment_rows",
    [_SORT_SMALL, _SORT_SMALL + 1, _SORT_LARGE, _SORT_LARGE + 1],
)
def test_segment_sort_thresholds(segment_rows: int, monkeypatch) -> None:
    """One expert's segment sits exactly at each width boundary: up to
    ``_STABLE_SEGMENT_SORT_SMALL`` rows take the short sorting network, up to
    ``_STABLE_SEGMENT_SORT_LARGE`` the long one, and anything longer is
    rebuilt by the sequential scan."""
    device = _kernel_device()
    num_experts = 64
    ids = _pinned_expert(segment_rows, _TOP_K, num_experts, 11 + segment_rows)
    topk_ids = ids.to(device=device)
    rows = _rows_per_expert(topk_ids, num_experts, None)
    assert int(rows[5]) == segment_rows
    _check_against_reference_and_scan(topk_ids, 48, num_experts, None, monkeypatch)


def test_segment_sort_matches_reference_at_the_served_expert_count(monkeypatch) -> None:
    """Kimi-K3 shape: 896 experts, top-16, uniform routing, block 48."""
    device = _kernel_device()
    num_experts = 896
    ids = _uniform(320, _TOP_K, num_experts, 71903)
    topk_ids = ids.to(device=device)
    _check_against_reference_and_scan(topk_ids, 48, num_experts, None, monkeypatch)


def test_stable_flag_keeps_small_launches_on_the_atomic_path(monkeypatch) -> None:
    """Below the stable threshold the segment sort is not launched."""
    device = _kernel_device()

    class LaunchRecorder:
        def __init__(self) -> None:
            self.launches = 0

        def __getitem__(self, _grid):
            def launch(*_args, **_kwargs) -> None:
                self.launches += 1

            return launch

    recorder = LaunchRecorder()
    monkeypatch.setattr(
        route_pack_module, "_pack_topk_routes_segment_sort_kernel", recorder
    )
    monkeypatch.setenv("B12X_W4A16_STABLE_ROUTE_PACK", "1")
    monkeypatch.delenv("B12X_W4A16_STABLE_ROUTE_PACK_SCAN", raising=False)
    num_experts = 64
    small = _uniform(32, 8, num_experts, 1).to(device)
    assert small.numel() < _STABLE_MIN_ROUTES
    pack_topk_routes_by_expert(small, 16, num_experts, **_workspaces(small, 16, num_experts))
    assert recorder.launches == 0
    large = _uniform(_STABLE_MIN_ROUTES // 8, 8, num_experts, 2).to(device)
    pack_topk_routes_by_expert(large, 16, num_experts, **_workspaces(large, 16, num_experts))
    assert recorder.launches == 1


def test_stable_flag_off_keeps_the_arrival_order_layout(monkeypatch) -> None:
    """Without the stable flag the atomic scatter layout is returned as is:
    the same segments and block owners as the reference, any order inside."""
    device = _kernel_device()
    monkeypatch.delenv("B12X_W4A16_STABLE_ROUTE_PACK", raising=False)
    num_experts = 64
    ids = _uniform(_STABLE_MIN_ROUTES // _TOP_K, _TOP_K, num_experts, 3).to(device)
    packed, blocks, count = pack_topk_routes_by_expert(
        ids, 48, num_experts, **_workspaces(ids, 48, num_experts)
    )
    expected = reference_stable_route_pack(ids, 48, num_experts)
    assert torch.equal(blocks.cpu(), expected[1])
    assert torch.equal(count.cpu(), expected[2])
    packed = packed.cpu()
    numel = int(ids.numel())
    cursor = 0
    rows = _rows_per_expert(ids, num_experts, None)
    for expert in range(num_experts):
        live = int(rows[expert])
        padded = math.ceil(live / 48) * 48
        segment = packed[cursor : cursor + padded]
        assert sorted(segment[:live].tolist()) == expected[0][cursor : cursor + live].tolist()
        assert bool((segment[live:] == numel).all())
        cursor += padded
