from __future__ import annotations

import inspect

import pytest
import torch

from benchmarks import benchmark_mtp_feedback as benchmark
from benchmarks.benchmark_mtp_feedback import (
    PROFILES,
    _ATOL,
    _DTYPE,
    _HIDDEN_SIZE,
    _RTOL,
    _STREAMS,
    _comparison_metrics,
    _parse_args,
    _parse_filter,
    _profile_listing,
    _profile_seed,
    _select_profiles,
    _summary,
)
from b12x.sequence.mtp_feedback._cute_prefill_config import (
    projection_capacity_rows,
    require_qwen_cute_tensors,
    supports_prefill,
)


def test_qwen38_flash_next_profiles_cover_decode_spec_and_prefill() -> None:
    assert _STREAMS == 4
    assert _HIDDEN_SIZE == 2560
    assert torch.bfloat16 == _DTYPE
    assert [(profile.name, profile.phase, profile.tokens) for profile in PROFILES] == [
        ("decode-t1", "decode", 1),
        ("spec-t4", "spec", 4),
        ("prefill-t17", "prefill", 17),
        ("prefill-t128", "prefill", 128),
        ("prefill-t512", "prefill", 512),
        ("prefill-t4096", "prefill", 4096),
    ]


def test_profile_and_phase_filters_preserve_declared_order() -> None:
    selected = _select_profiles(
        ("prefill-t512", "decode-t1", "prefill-t128"),
        ("decode", "prefill"),
    )
    assert [profile.name for profile in selected] == [
        "decode-t1",
        "prefill-t128",
        "prefill-t512",
    ]
    assert [profile.name for profile in _select_profiles(None, ("prefill",))] == [
        "prefill-t17",
        "prefill-t128",
        "prefill-t512",
        "prefill-t4096",
    ]


def test_profile_filter_rejects_unknown_empty_and_duplicate_filters() -> None:
    with pytest.raises(ValueError, match="unknown profiles"):
        _select_profiles(("glm",), None)
    with pytest.raises(ValueError, match="unknown phases"):
        _select_profiles(None, ("train",))
    with pytest.raises(ValueError, match="select no benchmark cases"):
        _select_profiles(("decode-t1",), ("prefill",))
    with pytest.raises(Exception, match="comma-separated"):
        _parse_filter(",,")
    with pytest.raises(Exception, match="cannot be combined"):
        _parse_filter("all,decode-t1")
    with pytest.raises(Exception, match="unique"):
        _parse_filter("decode-t1,decode-t1")


def test_cli_filters_profiles_without_requiring_cuda() -> None:
    args = _parse_args(
        [
            "--profiles",
            "spec-t4,prefill-t512",
            "--phases",
            "prefill",
            "--list-profiles",
        ]
    )
    assert args.list_profiles is True
    assert [profile.name for profile in args.selected_profiles] == ["prefill-t512"]
    assert _profile_listing(args.selected_profiles) == [
        {"name": "prefill-t512", "phase": "prefill", "tokens": 512}
    ]
    assert _profile_seed(17, args.selected_profiles[0]) == 17 + 4 * 10_007


def test_default_cli_selects_all_profiles_and_both_timing_modes() -> None:
    args = _parse_args([])
    assert args.selected_profiles == PROFILES
    assert args.warmup > 0
    assert args.samples > 0
    assert args.capacity_tokens == benchmark.PLANNER_CAPACITY_TOKENS
    assert args.flush_l2 is False


def test_capacity_override_covers_selected_live_token_profiles() -> None:
    args = _parse_args(
        [
            "--profiles",
            "decode-t1,spec-t4",
            "--capacity-tokens",
            "4096",
            "--list-profiles",
        ]
    )
    assert args.capacity_tokens == 4096
    with pytest.raises(SystemExit):
        _parse_args(["--profiles", "prefill-t128", "--capacity-tokens", "64"])


def test_cute_projection_contract_accepts_every_positive_live_count() -> None:
    for tokens in range(1, 4097):
        assert supports_prefill(
            tokens=tokens,
            streams=4,
            hidden_size=2560,
        )
    assert projection_capacity_rows(
        max_tokens=4096,
        streams=4,
        hidden_size=2560,
    ) == (4096, 16384)


@pytest.mark.parametrize(
    ("capacity", "expected_rows"),
    [
        (1, (16, 16)),
        (3, (16, 16)),
        (4, (16, 16)),
        (17, (32, 80)),
        (4095, (4096, 16384)),
        (4096, (4096, 16384)),
    ],
)
def test_qwen_projection_pads_each_planner_capacity(
    capacity: int, expected_rows: tuple[int, int]
) -> None:
    assert projection_capacity_rows(
        max_tokens=capacity,
        streams=4,
        hidden_size=2560,
    ) == expected_rows


@pytest.mark.parametrize(
    ("streams", "hidden_size"),
    [
        (3, 2560),
        (4, 2048),
    ],
)
def test_non_qwen_projection_geometries_are_rejected(
    streams: int,
    hidden_size: int,
) -> None:
    with pytest.raises(ValueError, match="only implements the Qwen3.8 CuTe"):
        projection_capacity_rows(
            max_tokens=17,
            streams=streams,
            hidden_size=hidden_size,
        )


def test_qwen_projection_accepts_positive_capacity() -> None:
    assert supports_prefill(
        tokens=17,
        streams=4,
        hidden_size=2560,
    )
    assert projection_capacity_rows(
        max_tokens=32,
        streams=4,
        hidden_size=2560,
    ) == (32, 128)


def test_qwen_dispatch_rejects_tensors_outside_tma_contract() -> None:
    with pytest.raises(
        ValueError,
        match="Qwen MTP CuTe projection contract violation.*TMA",
    ):
        require_qwen_cute_tensors(token_path=torch.empty(1))


def test_graph_contract_poisons_output_and_scratch_before_replay() -> None:
    source = inspect.getsource(benchmark._benchmark_profile)

    output_poison = source.index('binding.output.fill_(float("nan"))')
    scratch_poison = source.index("binding.scratch.fill_(0xFF)")
    replay = source.index("graph.replay()", scratch_poison)
    replay_gate = source.index("graph_metrics = _comparison_metrics", replay)
    assert output_poison < scratch_poison < replay < replay_gate
    assert '"graph_replay_after_scratch_poison": True' in source


def test_correctness_metrics_cover_close_finite_and_nonzero_gates() -> None:
    expected = torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16)
    actual = expected.clone()
    metrics = _comparison_metrics(actual, expected)

    assert metrics["finite"] is True
    assert metrics["nonzero"] is True
    assert metrics["close"] is True
    assert metrics["max_abs"] == 0.0
    assert metrics["relative_l2"] == 0.0
    assert metrics["cosine"] == pytest.approx(1.0)
    assert _RTOL == 2.0e-2
    assert _ATOL == 4.0e-2

    nonfinite = _comparison_metrics(
        torch.tensor([float("nan"), 1.0], dtype=torch.float32),
        torch.ones(2),
    )
    assert nonfinite["finite"] is False
    assert nonfinite["close"] is False


def test_timing_summary_preserves_tail_and_median_semantics() -> None:
    assert _summary([50.0, 10.0, 40.0, 20.0, 30.0]) == {
        "median_us": 30.0,
        "p10_us": 10.0,
        "p90_us": 40.0,
        "min_us": 10.0,
        "max_us": 50.0,
    }
    with pytest.raises(ValueError, match="at least one"):
        _summary([])
