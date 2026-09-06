"""Triton route-packing kernels for W4A16 MoE."""

from __future__ import annotations

import triton
import triton.language as tl
import torch

from b12x._lib.env import env_flag
from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity


_COUNT_BLOCK_T = 256
_SORT_BLOCK_T = 256
_POST_PREFIX_BLOCK_T = 256
_SMALL_PREFIX_MAX_PACKED_ROUTES = 4096
_SMALL_PREFIX_MAX_ROUTE_BLOCKS = 512


_FAST_COUNT_BLOCK_T = 1024
_STABLE_SORT_MIN_ROUTES = 4096
_STABLE_SORT_BLOCK_T = 4096
_STABLE_SORT_EXPERTS_PER_PROGRAM = 1
# Stable packing sorts each expert's scattered segment in registers. A
# program's lane count is a compile-time extent, so the width cannot be
# picked by a branch on the segment size: Triton requires both sides of a
# runtime ``if`` to yield the same type and rejects a ``tl.arange`` whose
# length differs between them. The widths are therefore spread over one
# kernel instance each, as ``(lane width, num_warps)`` in ascending
# power-of-two width order; instance ``i`` handles the experts whose live
# route count is in ``(width[i - 1], width[i]]`` and every other program
# exits after loading its count. Segments longer than the widest sort are
# rebuilt by a per-expert workspace scan, which needs no width at all.
_STABLE_SEGMENT_SORT_WIDTHS = ((256, 4), (2048, 8))
_STABLE_SEGMENT_SCAN_BLOCK_T = 4096
_STABLE_SEGMENT_SCAN_WARPS = 8
# Padding lanes of a sorted segment read this value so they order past every
# route id (a route id is a flat position in ``topk_ids``, far below 2**31-1)
# and the masked store leaves the segment's padding slots untouched.
_STABLE_SEGMENT_SORT_PAD = 2147483647


@triton.jit
def _w4a16_route_count_kernel(
    topk_ids,
    expert_map,
    counts,
    live_numel,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Parallel atomic histogram of routes per (mapped) expert.

    Same expert-id resolution as the sort kernel (expert_map aware).
    Writes ``counts[NUM_EXPERTS]``."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
        tl.int32
    )
    valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
    tl.atomic_add(counts + ids, 1, sem="relaxed", mask=valid)


@triton.jit
def _w4a16_route_prefix_from_counts_kernel(
    counts,
    packed_route_count,
    expert_offsets,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Tiny over-experts block-padded prefix from precomputed counts.

    Emits the clean block-padded ``expert_offsets`` / ``packed_route_count``
    contract; the sort kernel later advances expert_offsets in place."""
    experts = tl.arange(0, BLOCK_E)
    mask = experts < NUM_EXPERTS
    counts_v = tl.load(counts + experts, mask=mask, other=0)
    padded = ((counts_v + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)
    tl.store(expert_offsets + experts, prefix, mask=mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _numel_capacity_for_route_workspace(
    packed_routes: int,
    route_blocks: int,
    block_size: int,
    num_experts: int,
) -> int:
    """Recover the largest live route count covered by a fixed workspace."""
    block_size = int(block_size)
    num_experts = int(num_experts)
    padded_slots = min(int(packed_routes), int(route_blocks) * block_size)
    fully_padded_experts = num_experts * block_size
    if padded_slots <= fully_padded_experts:
        return padded_slots // block_size
    return padded_slots - num_experts * (block_size - 1)


def _workspace_slice(
    tensor: torch.Tensor | None,
    *,
    name: str,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    elements = int(elements)
    if tensor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{name} is not initialized for CUDA graph capture; "
                "provide a preallocated W4A16 route-packing workspace"
            )
        return torch.empty((elements,), dtype=dtype, device=device)
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if int(tensor.numel()) < elements:
        raise ValueError(
            f"{name} has {tensor.numel()} elements; need at least {elements}"
        )
    return tensor[:elements]


@triton.jit
def _pack_topk_routes_post_prefix_kernel(
    packed_route_indices,
    block_expert_ids,
    expert_offsets,
    live_numel,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_PACKED_ROUTES: tl.constexpr,
    MAX_ROUTE_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Sentinel-fill the packed slots and resolve each block's owning expert.

    Must run after the prefix is stored and before the sort kernel advances
    ``expert_offsets`` in place. The rightmost expert whose prefix is <= the
    block's first row owns the block: empty experts share their successor's
    offset, so the binary search cannot land on them."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tl.store(
        packed_route_indices + offsets,
        live_numel,
        mask=offsets < MAX_PACKED_ROUTES,
    )

    total = tl.load(expert_offsets + NUM_EXPERTS)
    block_rows = offsets * BLOCK_SIZE
    valid_blocks = (offsets < MAX_ROUTE_BLOCKS) & (block_rows < total)
    low = tl.zeros((BLOCK_T,), dtype=tl.int32)
    high = tl.full((BLOCK_T,), NUM_EXPERTS, dtype=tl.int32)
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = (low + high) // 2
        mid_offset = tl.load(expert_offsets + mid, mask=valid_blocks, other=0)
        take = mid_offset <= block_rows
        low = tl.where(take, mid, low)
        high = tl.where(take, high, mid)
    block_experts = tl.where(valid_blocks, low, -1)
    tl.store(
        block_expert_ids + offsets,
        block_experts,
        mask=offsets < MAX_ROUTE_BLOCKS,
    )


@triton.jit
def _pack_topk_routes_small_prefix_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    block_expert_ids,
    packed_route_count,
    expert_offsets,
    expert_counts,
    live_numel,
    NUMEL_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_PACKED_ROUTES: tl.constexpr,
    MAX_ROUTE_BLOCKS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_ROUTE_INIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Single-launch route packing whose cost scales with routed rows.

    Every expert-axis operation is one-dimensional: an atomic histogram of
    the routed ids into ``expert_counts``, one block-padded cumsum, and a
    per-route-block binary search of the stored prefix. Expert count only
    enters through the O(BLOCK_E) vector ops and the O(log BLOCK_E) search
    depth, so large-expert MoE decode does not regress the packing launch."""
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    tl.store(expert_counts + experts, 0, mask=expert_mask)
    tl.debug_barrier()

    route_offsets = tl.arange(0, BLOCK_T)
    for start in tl.range(0, NUMEL_CAPACITY, BLOCK_T):
        offsets = start + route_offsets
        raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
            tl.int32
        )
        valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
        ids = raw_ids
        if HAS_EXPERT_MAP:
            safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
            ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
            valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
        tl.atomic_add(expert_counts + ids, 1, sem="relaxed", mask=valid)
    tl.debug_barrier()

    counts = tl.load(expert_counts + experts, mask=expert_mask, other=0)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(expert_mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)

    tl.store(expert_offsets + experts, prefix, mask=expert_mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)

    route_init_offsets = tl.arange(0, BLOCK_ROUTE_INIT)
    tl.store(
        packed_route_indices + route_init_offsets,
        live_numel,
        mask=route_init_offsets < MAX_PACKED_ROUTES,
    )
    tl.debug_barrier()

    # Rightmost expert whose stored prefix is <= the block's first row. Empty
    # experts share their successor's offset, so the rightmost match is the
    # unique expert owning the block's padded segment.
    block_offsets = tl.arange(0, BLOCK_M)
    block_rows = block_offsets * BLOCK_SIZE
    valid_blocks = (block_offsets < MAX_ROUTE_BLOCKS) & (block_rows < total)
    low = tl.zeros((BLOCK_M,), dtype=tl.int32)
    high = tl.full((BLOCK_M,), NUM_EXPERTS, dtype=tl.int32)
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = (low + high) // 2
        mid_offset = tl.load(expert_offsets + mid, mask=valid_blocks, other=0)
        take = mid_offset <= block_rows
        low = tl.where(take, mid, low)
        high = tl.where(take, high, mid)
    block_experts = tl.where(valid_blocks, low, -1)
    tl.store(
        block_expert_ids + block_offsets,
        block_experts,
        mask=block_offsets < MAX_ROUTE_BLOCKS,
    )


@triton.jit
def _pack_topk_routes_sort_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    expert_offsets,
    live_numel,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
        tl.int32
    )
    valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)

    ranks = tl.atomic_add(expert_offsets + ids, 1, sem="relaxed", mask=valid)
    tl.store(packed_route_indices + ranks, offsets, mask=valid)


@triton.jit
def _pack_topk_routes_stable_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    expert_offsets,
    live_numel,
    NUMEL_CAPACITY: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EXPERTS_PER_PROGRAM: tl.constexpr,
):
    """Pack each expert's routes in ascending token-major route order."""
    expert_ids = (
        tl.program_id(0) * EXPERTS_PER_PROGRAM
        + tl.arange(0, EXPERTS_PER_PROGRAM)
    )
    expert_mask = expert_ids < NUM_EXPERTS
    output_starts = tl.load(
        expert_offsets + expert_ids,
        mask=expert_mask,
        other=0,
    )
    output_counts = tl.zeros((EXPERTS_PER_PROGRAM,), dtype=tl.int32)
    lanes = tl.arange(0, BLOCK_T)

    for start in tl.range(0, NUMEL_CAPACITY, BLOCK_T):
        offsets = start + lanes
        raw_ids = tl.load(
            topk_ids + offsets,
            mask=offsets < live_numel,
            other=-1,
        ).to(tl.int32)
        valid = (
            (offsets < live_numel)
            & (raw_ids >= 0)
            & (raw_ids < NUM_EXPERTS)
        )
        ids = raw_ids
        if HAS_EXPERT_MAP:
            safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
            ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(
                tl.int32
            )
            valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)

        matches = (
            expert_mask[:, None]
            & valid[None, :]
            & (expert_ids[:, None] == ids[None, :])
        )
        match_i32 = matches.to(tl.int32)
        local_ranks = tl.cumsum(match_i32, axis=1) - 1
        output_indices = (
            output_starts[:, None]
            + output_counts[:, None]
            + local_ranks
        )
        tl.store(
            packed_route_indices + output_indices,
            offsets[None, :],
            mask=matches,
        )
        output_counts += tl.sum(match_i32, axis=1)


@triton.jit
def _pack_topk_routes_segment_sort_kernel(
    packed_route_indices,
    expert_offsets,
    expert_counts,
    COUNT_MIN: tl.constexpr,
    SORT_WIDTH: tl.constexpr,
    PAD: tl.constexpr,
):
    """Sort one expert's packed segment when it fits this instance's width.

    Runs after ``_pack_topk_routes_sort_kernel`` scattered the routes and
    advanced ``expert_offsets[e]`` to the end of expert ``e``'s live routes;
    the segment is ``[expert_offsets[e] - expert_counts[e], expert_offsets[e])``
    and its padding slots keep their sentinel. Route ids inside a segment are
    distinct, so ascending order is the unique stable layout and equals what
    ``_pack_topk_routes_stable_kernel`` writes.

    One program per expert. ``SORT_WIDTH`` lanes are the whole program's
    extent, so the width is a property of the launch and never of a runtime
    branch; the program does nothing unless its live route count is in
    ``(COUNT_MIN, SORT_WIDTH]``, which is how the host covers all segment
    sizes with one launch per width.
    """
    expert = tl.program_id(0)
    count = tl.load(expert_counts + expert)
    if (count > COUNT_MIN) & (count <= SORT_WIDTH):
        start = tl.load(expert_offsets + expert) - count
        lanes = tl.arange(0, SORT_WIDTH)
        mask = lanes < count
        routes = tl.load(packed_route_indices + start + lanes, mask=mask, other=PAD)
        routes = tl.sort(routes)
        tl.store(packed_route_indices + start + lanes, routes, mask=mask)


@triton.jit
def _pack_topk_routes_segment_scan_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    expert_offsets,
    expert_counts,
    live_numel,
    NUMEL_CAPACITY: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    COUNT_MIN: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Rebuild one expert's packed segment by scanning the route workspace.

    Covers the segments too long to sort in registers (live route count above
    ``COUNT_MIN``): the program walks ``topk_ids`` once, keeping only the
    routes of its own expert, and writes them in workspace order, which is
    ascending. The result is the same segment layout the register sort
    produces, so the two paths can be mixed within one launch sequence.
    """
    expert = tl.program_id(0)
    count = tl.load(expert_counts + expert)
    if count > COUNT_MIN:
        start = tl.load(expert_offsets + expert) - count
        lanes = tl.arange(0, BLOCK_T)
        written = tl.zeros((1,), dtype=tl.int32)
        for chunk in tl.range(0, NUMEL_CAPACITY, BLOCK_T):
            offsets = chunk + lanes
            raw_ids = tl.load(
                topk_ids + offsets, mask=offsets < live_numel, other=-1
            ).to(tl.int32)
            valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
            ids = raw_ids
            if HAS_EXPERT_MAP:
                safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
                ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(
                    tl.int32
                )
                valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
            matches = valid & (ids == expert)
            match_i32 = matches.to(tl.int32)
            local_ranks = tl.cumsum(match_i32, axis=0) - 1
            tl.store(
                packed_route_indices + start + written + local_ranks,
                offsets,
                mask=matches,
            )
            written += tl.sum(match_i32, axis=0)


def _launch_stable_segment_pack(
    topk_ids: torch.Tensor,
    expert_map_tensor: torch.Tensor,
    packed_route_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_counts: torch.Tensor,
    *,
    live_numel: int,
    numel_capacity: int,
    num_experts: int,
    has_expert_map: bool,
) -> None:
    """Order every scattered expert segment by ascending route id.

    One sort launch per lane width in ``_STABLE_SEGMENT_SORT_WIDTHS`` and one
    scan launch for the segments above the widest sort. Each launch covers
    all experts; a program outside its launch's segment-size band retires
    after one load, and the bands partition ``[1, inf)``, so every non-empty
    segment is ordered exactly once. The counts live on the device, so the
    band a given expert falls into is only known there — splitting the widths
    across launches is what keeps one ``tl.arange`` extent per program.
    """
    grid = (int(num_experts),)
    count_min = 0
    for width, num_warps in _STABLE_SEGMENT_SORT_WIDTHS:
        _pack_topk_routes_segment_sort_kernel[grid](
            packed_route_indices,
            expert_offsets,
            expert_counts,
            COUNT_MIN=count_min,
            SORT_WIDTH=width,
            PAD=_STABLE_SEGMENT_SORT_PAD,
            num_warps=num_warps,
        )
        count_min = width
    _pack_topk_routes_segment_scan_kernel[grid](
        topk_ids,
        expert_map_tensor,
        packed_route_indices,
        expert_offsets,
        expert_counts,
        live_numel,
        NUMEL_CAPACITY=numel_capacity,
        NUM_EXPERTS=int(num_experts),
        HAS_EXPERT_MAP=has_expert_map,
        COUNT_MIN=count_min,
        BLOCK_T=_STABLE_SEGMENT_SCAN_BLOCK_T,
        num_warps=_STABLE_SEGMENT_SCAN_WARPS,
    )


def _stable_route_pack_uses_scan() -> bool:
    """Select the sequential per-expert scan for stable packing.

    The default stable path scatters with atomics and sorts each expert's
    segment; the scan is the reference implementation of the same layout and
    is kept for A/B comparison (``B12X_W4A16_STABLE_ROUTE_PACK_SCAN=1``).
    """
    return env_flag("W4A16_STABLE_ROUTE_PACK_SCAN")


def pack_topk_routes_by_expert(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    expert_map: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = int(topk_ids.numel())
    topk = int(topk_ids.shape[-1]) if topk_ids.ndim >= 2 else 1
    numel_capacity, capacity_packed_routes, capacity_route_blocks = route_pack_capacity(
        numel,
        int(block_size),
        int(num_experts),
        topk=topk,
    )
    provided_routes = (
        None if packed_route_indices is None else int(packed_route_indices.numel())
    )
    provided_blocks = (
        None if block_expert_ids is None else int(block_expert_ids.numel())
    )
    if (
        provided_routes is not None
        and provided_blocks is not None
        and (
            provided_routes < capacity_packed_routes
            or provided_blocks < capacity_route_blocks
        )
    ):
        (
            exact_numel_capacity,
            exact_packed_routes,
            exact_route_blocks,
        ) = route_pack_capacity(
            numel,
            int(block_size),
            int(num_experts),
            topk=topk,
            bucket_tokens=False,
        )
        if (
            provided_routes >= exact_packed_routes
            and provided_blocks >= exact_route_blocks
        ):
            # A serving caller can own one fixed arena sized for its configured
            # maximum while a live prefill tail belongs to a larger power-of-two
            # bucket. Reuse the full caller capacity instead of specializing each
            # exact tail. The small-prefix and post-prefix kernels fill unused
            # route slots with ``live_numel`` and unused blocks with ``-1``. The
            # recovered live-route capacity also keeps the small-prefix loop bound
            # stable without changing the caller's allocation or route semantics.
            numel_capacity = _numel_capacity_for_route_workspace(
                provided_routes,
                provided_blocks,
                int(block_size),
                int(num_experts),
            )
            capacity_packed_routes = provided_routes
            capacity_route_blocks = provided_blocks
        else:
            numel_capacity = exact_numel_capacity
            capacity_packed_routes = exact_packed_routes
            capacity_route_blocks = exact_route_blocks
    max_packed_routes = capacity_packed_routes
    max_route_blocks = capacity_route_blocks
    max_packed_routes = max(max_packed_routes, 1)
    max_route_blocks = max(max_route_blocks, 1)

    provided_routes = (
        None if packed_route_indices is None else int(packed_route_indices.numel())
    )
    provided_blocks = (
        None if block_expert_ids is None else int(block_expert_ids.numel())
    )
    if (
        provided_routes is not None
        and provided_blocks is not None
        and (provided_routes < max_packed_routes or provided_blocks < max_route_blocks)
    ):
        raise ValueError(
            "W4A16 route-packing workspace is too small: "
            f"topk_shape={tuple(topk_ids.shape)}, live_routes={numel}, "
            f"capacity_routes={numel_capacity}, block_size={int(block_size)}, "
            f"num_experts={int(num_experts)}, "
            f"packed_route_indices={provided_routes}/{max_packed_routes}, "
            f"block_expert_ids={provided_blocks}/{max_route_blocks}"
        )

    packed_route_indices = _workspace_slice(
        packed_route_indices,
        name="packed_route_indices",
        elements=max_packed_routes,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    block_expert_ids = _workspace_slice(
        block_expert_ids,
        name="block_expert_ids",
        elements=max_route_blocks,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    packed_route_count = _workspace_slice(
        packed_route_count,
        name="packed_route_count",
        elements=1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_offsets = _workspace_slice(
        expert_offsets,
        name="expert_offsets",
        elements=int(num_experts) + 1,
        dtype=torch.int32,
        device=topk_ids.device,
    )

    if numel == 0:
        packed_route_indices.fill_(0)
        block_expert_ids.fill_(-1)
        packed_route_count.zero_()
        return packed_route_indices, block_expert_ids, packed_route_count

    block_e = _next_power_of_2(num_experts)
    sort_grid = (triton.cdiv(numel, _SORT_BLOCK_T),)
    expert_map_tensor = expert_map if expert_map is not None else topk_ids

    block_route_init = _next_power_of_2(max(max_packed_routes, 1))
    block_m = _next_power_of_2(max(max_route_blocks, 1))
    use_small_prefix = (
        block_route_init <= _SMALL_PREFIX_MAX_PACKED_ROUTES
        and block_m <= _SMALL_PREFIX_MAX_ROUTE_BLOCKS
    )
    if use_small_prefix:
        # Decode-sized W4A16 MoE calls are launch-overhead sensitive. Keep the
        # large-shape split kernel below, but fold prefix/post-prefix work into
        # one launch when the vector sizes are safely bounded.
        expert_counts = _workspace_slice(
            expert_counts,
            name="expert_counts",
            elements=int(num_experts),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        _pack_topk_routes_small_prefix_kernel[(1,)](
            topk_ids,
            expert_map_tensor,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            expert_offsets,
            expert_counts,
            numel,
            NUMEL_CAPACITY=numel_capacity,
            BLOCK_SIZE=int(block_size),
            NUM_EXPERTS=int(num_experts),
            MAX_PACKED_ROUTES=max_packed_routes,
            MAX_ROUTE_BLOCKS=max_route_blocks,
            HAS_EXPERT_MAP=expert_map is not None,
            BLOCK_E=block_e,
            BLOCK_T=_COUNT_BLOCK_T,
            BLOCK_ROUTE_INIT=block_route_init,
            BLOCK_M=block_m,
            SEARCH_STEPS=block_e.bit_length(),
            num_warps=8,
        )
    else:
        # Parallel path for shapes above the single-launch caps: atomic
        # histogram over routed rows, one tiny 1-D prefix, and binary-search
        # block resolution. No stage scales with expert count beyond O(E)
        # vector work, so large-expert models keep prefill-scale packing
        # cheap. Capture-safe when the caller provides the routing
        # workspaces; ``_workspace_slice`` rejects capture without them.
        expert_counts = _workspace_slice(
            expert_counts,
            name="expert_counts",
            elements=int(num_experts),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        expert_counts.zero_()
        _w4a16_route_count_kernel[(triton.cdiv(numel, _FAST_COUNT_BLOCK_T),)](
            topk_ids,
            expert_map_tensor,
            expert_counts,
            numel,
            NUM_EXPERTS=int(num_experts),
            HAS_EXPERT_MAP=expert_map is not None,
            BLOCK_T=_FAST_COUNT_BLOCK_T,
            num_warps=4,
        )
        _w4a16_route_prefix_from_counts_kernel[(1,)](
            expert_counts,
            packed_route_count,
            expert_offsets,
            BLOCK_SIZE=int(block_size),
            NUM_EXPERTS=int(num_experts),
            BLOCK_E=block_e,
            num_warps=4,
        )
        post_prefix_grid = (
            max(
                triton.cdiv(max_packed_routes, _POST_PREFIX_BLOCK_T),
                triton.cdiv(max_route_blocks, _POST_PREFIX_BLOCK_T),
            ),
        )
        _pack_topk_routes_post_prefix_kernel[post_prefix_grid](
            packed_route_indices,
            block_expert_ids,
            expert_offsets,
            numel,
            BLOCK_SIZE=int(block_size),
            NUM_EXPERTS=int(num_experts),
            MAX_PACKED_ROUTES=max_packed_routes,
            MAX_ROUTE_BLOCKS=max_route_blocks,
            BLOCK_T=_POST_PREFIX_BLOCK_T,
            SEARCH_STEPS=block_e.bit_length(),
            num_warps=4,
        )
    stable = env_flag("W4A16_STABLE_ROUTE_PACK") and numel >= _STABLE_SORT_MIN_ROUTES
    if stable and _stable_route_pack_uses_scan():
        # Reference stable layout: one program per expert scans the whole
        # route workspace and writes its routes in ascending order.
        stable_grid = (
            triton.cdiv(num_experts, _STABLE_SORT_EXPERTS_PER_PROGRAM),
        )
        _pack_topk_routes_stable_kernel[stable_grid](
            topk_ids,
            expert_map_tensor,
            packed_route_indices,
            expert_offsets,
            numel,
            NUMEL_CAPACITY=numel_capacity,
            NUM_EXPERTS=int(num_experts),
            HAS_EXPERT_MAP=expert_map is not None,
            BLOCK_T=_STABLE_SORT_BLOCK_T,
            EXPERTS_PER_PROGRAM=_STABLE_SORT_EXPERTS_PER_PROGRAM,
            num_warps=8,
        )
        return packed_route_indices, block_expert_ids, packed_route_count
    _pack_topk_routes_sort_kernel[sort_grid](
        topk_ids,
        expert_map_tensor,
        packed_route_indices,
        expert_offsets,
        numel,
        NUM_EXPERTS=int(num_experts),
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_T=_SORT_BLOCK_T,
        num_warps=4,
    )
    if stable:
        # The atomic scatter above fills each expert's segment in arrival
        # order and leaves expert_offsets[e] at the segment end; ordering each
        # segment yields the ascending layout of the reference scan at a
        # fraction of its cost (the scan reads the whole workspace once per
        # expert).
        _launch_stable_segment_pack(
            topk_ids,
            expert_map_tensor,
            packed_route_indices,
            expert_offsets,
            expert_counts,
            live_numel=numel,
            numel_capacity=numel_capacity,
            num_experts=int(num_experts),
            has_expert_map=expert_map is not None,
        )
    return packed_route_indices, block_expert_ids, packed_route_count


__all__ = ["pack_topk_routes_by_expert"]
