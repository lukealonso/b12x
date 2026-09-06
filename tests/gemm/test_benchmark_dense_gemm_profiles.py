from benchmarks.benchmark_dense_gemm import (
    COSINE_THRESHOLD,
    DEFAULT_PROFILE,
    FP4_GEMM_SPECS,
    FP8_BLOCK_GEMM_SPECS,
    FP8_GEMM_SPECS,
    QWEN38_27B_GEMM_SPECS,
    QWEN38_27B_PROFILE,
    gemm_specs_for_mode,
)


def test_quantized_reference_cosine_threshold_is_not_overly_strict() -> None:
    assert COSINE_THRESHOLD == 0.9999


def test_qwen38_27b_profile_covers_all_dense_quantization_modes() -> None:
    for mode in ("fp4", "fp8", "fp8-e2e", "fp8-block"):
        assert (
            gemm_specs_for_mode(mode, QWEN38_27B_PROFILE)
            is QWEN38_27B_GEMM_SPECS
        )


def test_qwen38_27b_profile_uses_checkpoint_ffn_shapes() -> None:
    assert [(k, n) for _name, k, n, _note in QWEN38_27B_GEMM_SPECS] == [
        (5120, 17408),
        (17408, 5120),
    ]


def test_default_profile_preserves_existing_mode_shapes() -> None:
    assert gemm_specs_for_mode("fp4", DEFAULT_PROFILE) is FP4_GEMM_SPECS
    assert gemm_specs_for_mode("fp8", DEFAULT_PROFILE) is FP8_GEMM_SPECS
    assert gemm_specs_for_mode("fp8-e2e", DEFAULT_PROFILE) is FP8_GEMM_SPECS
    assert gemm_specs_for_mode("fp8-block", DEFAULT_PROFILE) is FP8_BLOCK_GEMM_SPECS
