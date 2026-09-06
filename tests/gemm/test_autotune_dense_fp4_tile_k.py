from benchmarks.autotune_dense_fp4_tile_k import (
    COMMON_SHAPES,
    QWEN38_TP_SHAPES,
    _parse_bool_list,
    _plan_candidates,
)
from b12x._lib.dense_gemm import _DenseGemmPlan


def test_qwen38_tp_corpus_scales_the_correct_projection_axis() -> None:
    shapes = {shape.name: (shape.n, shape.k) for shape in QWEN38_TP_SHAPES}

    for tp in (1, 2, 4, 8):
        assert shapes[f"qwen38_27b_gate_or_up_tp{tp}"] == (17408 // tp, 5120)
        assert shapes[f"qwen38_27b_down_tp{tp}"] == (5120, 17408 // tp)


def test_common_corpus_keeps_unaligned_k_coverage() -> None:
    shapes = {shape.name: (shape.n, shape.k) for shape in COMMON_SHAPES}

    assert shapes["nano35_relu2_fc2_unaligned_k"] == (2688, 1856)


def test_joint_plan_space_couples_narrow_tile_with_swapped_storage() -> None:
    plan = _DenseGemmPlan((64, 128), "tma", False)
    candidates = _plan_candidates(
        plan,
        tile_k_list=[128, 256],
        joint=True,
        tile_mn_list=None,
        load_path_list=None,
        swap_ab_list=None,
    )

    assert len(candidates) == 10
    assert all(
        candidate.swap_ab == (candidate.tile_mn[1] < 64)
        for candidate in candidates
    )
    assert {candidate.load_path for candidate in candidates} == {"tma"}


def test_plan_space_excludes_smem_oversized_sm120_candidate() -> None:
    plan = _DenseGemmPlan((64, 128), "tma", False)
    candidates = _plan_candidates(
        plan,
        tile_k_list=[128, 256, 512],
        joint=True,
        tile_mn_list=None,
        load_path_list=None,
        swap_ab_list=None,
    )

    assert len(candidates) == 14
    assert not any(
        candidate.tile_mn == (128, 128) and candidate.tile_k == 512
        for candidate in candidates
    )


def test_explicit_plan_axes_remain_available() -> None:
    plan = _DenseGemmPlan((64, 128), "tma", False)
    candidates = _plan_candidates(
        plan,
        tile_k_list=[256],
        joint=False,
        tile_mn_list=[(64, 64), (64, 128)],
        load_path_list=["tma"],
        swap_ab_list=_parse_bool_list("0"),
    )

    assert [candidate.tile_mn for candidate in candidates] == [
        (64, 64),
        (64, 128),
    ]
