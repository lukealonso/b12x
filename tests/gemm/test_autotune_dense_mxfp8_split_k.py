from benchmarks.autotune_dense_mxfp8_split_k import QWEN38_TP_SHAPES


def test_mxfp8_split_k_qwen_tp_corpus_scales_the_correct_projection_axis() -> None:
    shapes = {shape.name: (shape.n, shape.k) for shape in QWEN38_TP_SHAPES}

    for tp in (1, 2, 4, 8):
        assert shapes[f"qwen38_27b_gate_or_up_tp{tp}"] == (17408 // tp, 5120)
        assert shapes[f"qwen38_27b_down_tp{tp}"] == (5120, 17408 // tp)
