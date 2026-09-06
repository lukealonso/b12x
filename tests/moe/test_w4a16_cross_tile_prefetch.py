"""CPU checks of the W4A16 cross-tile prefetch (shared-memory layout, schedule).

The pipelined whole-tile loop (``_run_persistent_gemm_pipelined``) must run
exactly the tile sequence of the legacy whole-tile loop and keep its
prefetch bookkeeping consistent; the double-buffered route metadata and the
de-aliased reduction scratch must not overlap any pipeline stage. Both are
pure host-side arithmetic and are modelled here without a GPU.
"""

from __future__ import annotations

import pytest

kernel = pytest.importorskip("b12x.moe._shared.kernels.w4a16.kernel")

_STAGES = kernel._STAGES
_MAX_SMEM = kernel._DEFAULT_MAX_SHARED_MEM


def _served_gemm(monkeypatch: pytest.MonkeyPatch, *, fc: str, block: int, width: int, prefetch: str):
    """Build the FC1 or FC2 GEMM of the served Kimi-K3 TP9 prefill kernel."""
    monkeypatch.setenv("B12X_W4A16_CROSS_TILE_PREFETCH", prefetch)
    monkeypatch.delenv("B12X_W4A16_SMALL_M_SPLITK", raising=False)
    common = dict(
        num_experts=896,
        tile_n=128,
        tile_k=128,
        moe_block_size=block,
        max_m_blocks=1536,
        element_dtype="fp16",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        trellis_bits=2,
        trellis_codebook="sqg_xor_cheb_t12",
        schedule_whole_tiles=True,
        dynamic_num_experts=True,
    )
    if fc == "fc1":
        return kernel.W4A16GemmKernel(
            size_m=4608,
            size_n=2 * width,
            size_k=3584,
            top_k=16,
            mul_topk_weights=False,
            w13_layout="trellis3_t256_proj",
            dual_a=True,
            route_major_a=True,
            **common,
        )
    return kernel.W4A16GemmKernel(
        size_m=4608 * 16,
        size_n=3584,
        size_k=width,
        top_k=1,
        mul_topk_weights=False,
        w13_layout="packed",
        fused_sum_topk=16,
        **common,
    )


def _regions(g) -> dict[str, tuple[int, int]]:
    """Byte ranges of every shared-memory region of one GEMM kernel."""
    meta = g.sh_meta_int4 * 16
    regions = {
        f"meta{i}": (i * meta, (i + 1) * meta) for i in range(g.sh_meta_copies)
    }
    b_bytes = _STAGES * g.b_sh_stage_bytes
    regions["b"] = (g.sh_b_off * 16, g.sh_b_off * 16 + b_bytes)
    red_bytes = (2 * g.cta_n_blocks + 1) * 16 * g.cta_m_blocks * 16
    regions["red"] = (g.sh_red_off * 16, g.sh_red_off * 16 + red_bytes)
    regions["s"] = (g.sh_s_off * 16, (g.sh_s_off + _STAGES * g.s_sh_stage) * 16)
    regions["a"] = (g.sh_a_off * 16, (g.sh_a_off + _STAGES * g.a_sh_stage) * 16)
    return regions


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@pytest.mark.parametrize("fc", ["fc1", "fc2"])
@pytest.mark.parametrize("width", [384, 256])
def test_block48_layout_dealiases_red_and_double_buffers_metadata(
    monkeypatch: pytest.MonkeyPatch, fc: str, width: int
) -> None:
    g = _served_gemm(monkeypatch, fc=fc, block=48, width=width, prefetch="1")
    assert g.cross_tile_prefetch
    assert g.sh_red_dealiased
    assert g.sh_meta_copies == 2
    assert g.blocks_per_sm == 1
    regions = _regions(g)
    names = list(regions)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            assert not _overlaps(regions[x], regions[y]), (x, y, regions)
    assert all(start % 16 == 0 for start, _ in regions.values())
    assert g.shared_words * 4 == max(end for _, end in regions.values())
    # Served geometry: 2 x 784 B metadata, 16 KiB B stages, 12.75 KiB red,
    # 256 B bias slack, 2 KiB scale stages, 48 KiB A stages = 82,464 B; the
    # fused launch adds the 4 KiB T12 table, one mbarrier and the 1 KiB
    # struct alignment (86,576 B dynamic shared memory).
    assert g.shared_words * 4 == 82_464
    assert (
        g.shared_words * 4 + kernel._CROSS_TILE_PREFETCH_SMEM_RESERVE_BYTES
        <= _MAX_SMEM
    )
    assert g.sh_meta_stride_bytes == g.sh_meta_int4 * 16 == 784


@pytest.mark.parametrize("fc", ["fc1", "fc2"])
def test_prefetch_off_keeps_the_served_layout(monkeypatch: pytest.MonkeyPatch, fc: str) -> None:
    g = _served_gemm(monkeypatch, fc=fc, block=48, width=384, prefetch="0")
    assert not g.cross_tile_prefetch
    assert not g.sh_red_dealiased
    assert g.sh_meta_copies == 1
    assert g.sh_b_off == g.sh_red_off == g.sh_valid_count_off == 48
    assert g.shared_words * 4 == 68_352


def test_block64_keeps_prefetch_but_aliases_red(monkeypatch: pytest.MonkeyPatch) -> None:
    # 64-row route blocks: 17 KiB red no longer fits next to the B stages
    # under the 99 KiB opt-in limit; the prefetch then issues B after the
    # drain (aliased layout), and the residency contract still holds.
    g = _served_gemm(monkeypatch, fc="fc1", block=64, width=384, prefetch="1")
    assert g.cross_tile_prefetch
    assert not g.sh_red_dealiased
    assert g.sh_meta_copies == 2
    regions = _regions(g)
    assert _overlaps(regions["b"], regions["red"])
    for name in ("meta0", "meta1", "s", "a"):
        assert not _overlaps(regions[name], regions["b"])
        assert not _overlaps(regions[name], regions["red"])
    assert (
        g.shared_words * 4 + kernel._CROSS_TILE_PREFETCH_SMEM_RESERVE_BYTES
        <= _MAX_SMEM // g.blocks_per_sm
    )


def test_prefetch_requires_whole_tile_route_packed_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B12X_W4A16_CROSS_TILE_PREFETCH", "1")
    g = kernel.W4A16GemmKernel(
        size_m=4608,
        size_n=768,
        size_k=3584,
        num_experts=896,
        top_k=16,
        mul_topk_weights=False,
        tile_n=128,
        tile_k=128,
        moe_block_size=48,
        max_m_blocks=1536,
        element_dtype="fp16",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        dual_a=True,
        route_major_a=True,
        schedule_whole_tiles=False,
    )
    assert not g.cross_tile_prefetch
    assert g.sh_meta_copies == 1


# --- schedule model -------------------------------------------------------


def _legacy_whole_tile_schedule(cta: int, grid_x: int, route_blocks: int, n_tiles: int, experts):
    """Tile sequence of the legacy whole-tile branch for one CTA."""
    global_mn_tiles = route_blocks * n_tiles
    iters = (global_mn_tiles + grid_x - 1) // grid_x
    work = cta
    executed = []
    for _ in range(iters):
        rb, nt = divmod(work, n_tiles)
        if rb < route_blocks and experts[rb] >= 0:
            executed.append((rb, nt))
        work += grid_x
    return executed


def _pipelined_schedule(cta: int, grid_x: int, route_blocks: int, n_tiles: int, experts):
    """Model of ``_run_persistent_gemm_pipelined`` with its bookkeeping."""
    global_mn_tiles = route_blocks * n_tiles
    remaining = (global_mn_tiles + grid_x - 1) // grid_x
    work = cta
    prefetched = 0
    parity = 0
    executed = []
    metadata_written = {0: None, 1: None}  # parity -> route block loaded
    while remaining > 0:
        remaining -= 1
        rb, nt = divmod(work, n_tiles)
        if rb < route_blocks and experts[rb] >= 0:
            nxt = work + grid_x
            nrb, nnt = divmod(nxt, n_tiles)
            has_next = int(remaining > 0 and nrb < route_blocks and experts[nrb] >= 0)
            if prefetched:
                assert metadata_written[parity] == rb, "prefetched metadata must belong to this tile"
            else:
                metadata_written[parity] = rb
            if has_next:
                metadata_written[1 - parity] = nrb
            executed.append((rb, nt, prefetched, parity, has_next))
            prefetched = has_next
            parity = 1 - parity
        work += grid_x
    assert prefetched == 0, "the last executed tile never prefetches"
    return executed


@pytest.mark.parametrize("grid_x", [188, 120, 7])
@pytest.mark.parametrize("route_blocks,n_tiles", [(1536, 6), (1537, 6), (1, 6), (0, 6), (25, 28), (188, 1)])
def test_pipelined_loop_runs_the_legacy_tile_sequence(grid_x: int, route_blocks: int, n_tiles: int) -> None:
    import random

    rng = random.Random(route_blocks * 31 + n_tiles)
    experts = [rng.randrange(896) if rng.random() > 0.05 else -1 for _ in range(route_blocks)]
    for cta in range(grid_x):
        legacy = _legacy_whole_tile_schedule(cta, grid_x, route_blocks, n_tiles, experts)
        pipelined = _pipelined_schedule(cta, grid_x, route_blocks, n_tiles, experts)
        assert [(rb, nt) for rb, nt, *_ in pipelined] == legacy
        # Every tile after a prefetching tile starts prefetched, and the
        # parity alternates per executed tile.
        for prev, cur in zip(pipelined, pipelined[1:]):
            assert cur[2] == prev[4]
            assert cur[3] == 1 - prev[3]
        if pipelined:
            assert pipelined[0][2] == 0


def test_cp_async_group_accounting_keeps_stage_order() -> None:
    """The FIFO of cp.async groups always exposes stage k before compute k.

    Model: groups are completed in issue order; ``wait_group(n)`` completes
    all but the last ``n``. The pipelined tile issues the next tile's route
    metadata after the stage-0 wait and its first three stages after the
    loop, so every stage must be complete when its pipe is consumed.
    """

    for k_tiles in (1, 2, 3, 28):
        fifo: list[str] = []
        done: set[str] = set()

        def wait(n: int) -> None:
            while len(fifo) > n:
                done.add(fifo.pop(0))

        def issue(name: str) -> None:
            fifo.append(name)

        # Tile T: stages 0..2 issued by the previous tile's epilogue.
        for p in range(_STAGES - 1):
            issue(f"s{p}" if p < k_tiles else f"empty{p}")
        wait(_STAGES - 2)
        assert "s0" in done or k_tiles == 0
        issue("meta_next")
        for tile in range(k_tiles):
            # lookahead at kk == 0 of every consumed pipe
            fetch = tile + _STAGES - 1
            issue(f"s{fetch}" if fetch < k_tiles else f"empty_la{tile}")
            wait(_STAGES - 2)
            assert f"s{tile}" in done, (k_tiles, tile, fifo, done)
        # finish_route_metadata for the next tile
        wait(0)
        assert "meta_next" in done
