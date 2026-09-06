from benchmarks.autotune_dense_mxfp8_plan import (
    PRODUCTION_TILES,
    QWEN38_TP_SHAPES,
    _plan_candidates,
)


def test_qwen38_mxfp8_plan_corpus_covers_tp_axes() -> None:
    assert [(shape.n, shape.k) for shape in QWEN38_TP_SHAPES] == [
        (17408, 5120),
        (5120, 17408),
        (8704, 5120),
        (5120, 8704),
        (4352, 5120),
        (5120, 4352),
        (2176, 5120),
        (5120, 2176),
    ]


def test_mxfp8_decode_plan_space_covers_tile_bk_and_split_axes() -> None:
    candidates = _plan_candidates(
        m=6,
        tile_mn_list=list(PRODUCTION_TILES),
        tile_k_list=[64, 128],
        split_k_list=[1, 2, 4],
        large_m_unroll_list=[None],
    )

    assert {candidate.tile_mn for candidate in candidates} == set(PRODUCTION_TILES)
    assert {candidate.tile_k for candidate in candidates} == {64, 128}
    assert {candidate.split_k_slices for candidate in candidates} == {1, 2, 4}
    assert all(
        candidate.tile_k != 64 or candidate.tile_mn[0] == 128
        for candidate in candidates
    )


def test_mxfp8_prefill_plan_space_disables_split_k() -> None:
    candidates = _plan_candidates(
        m=2048,
        tile_mn_list=list(PRODUCTION_TILES),
        tile_k_list=[64, 128],
        split_k_list=[1, 2, 4],
        large_m_unroll_list=[False, True],
    )

    assert {candidate.split_k_slices for candidate in candidates} == {1}
    assert {candidate.large_m_unroll for candidate in candidates} == {False, True}
