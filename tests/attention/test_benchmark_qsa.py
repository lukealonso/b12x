from __future__ import annotations

import ast
import inspect
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from benchmarks import benchmark_qsa


def test_qwen38_profiles_preserve_selector_and_rank_local_gqa_geometry() -> None:
    assert benchmark_qsa.INDEX_HEADS == 4
    assert benchmark_qsa.INDEX_KV_HEADS == 1
    assert benchmark_qsa.INDEX_HEAD_DIM == 128
    assert benchmark_qsa.INDEX_ROTARY_DIM == 64
    assert benchmark_qsa.COMPRESS_RATIO == 4
    assert benchmark_qsa.BUDGET == 2048
    assert benchmark_qsa.MAIN_PAGE_SIZE == 16
    assert benchmark_qsa.HEAD_DIM == 256
    assert benchmark_qsa.MROPE_SECTIONS == (11, 11, 10)
    assert {
        name: (profile.tensor_parallel_size, profile.q_heads, profile.kv_heads)
        for name, profile in benchmark_qsa.PROFILES.items()
    } == {
        "tp1": (1, 24, 2),
        "tp2": (2, 12, 1),
        "tp4": (4, 6, 1),
    }


def test_default_cases_use_tp2_and_bound_synthetic_cache_capacity() -> None:
    args = benchmark_qsa._parse_args([])
    cases = benchmark_qsa._resolve_cases(args)

    assert args.profiles == ("tp2",)
    assert args.main_cache_layout == "interleaved"
    assert args.kv_cache_dtype == "bf16"
    assert args.rows == (1, 4, 16, 64)
    assert args.contexts == (2048, 8192)
    assert len(cases) == 8
    assert {case.profile.name for case in cases} == {"tp2"}
    assert max(case.context for case in cases) < benchmark_qsa.MODEL_MAX_CONTEXT
    assert (
        max(benchmark_qsa._cache_capacity_bytes(case)["total"] for case in cases)
        < 1 << 30
    )


def test_cli_filters_tp_profiles_rows_and_full_context() -> None:
    args = benchmark_qsa._parse_args(
        [
            "--profiles",
            "tp1,tp4",
            "--rows",
            "1,16",
            "--contexts",
            "2048,full",
        ]
    )
    cases = benchmark_qsa._resolve_cases(args)

    assert args.profiles == ("tp1", "tp4")
    assert args.rows == (1, 16)
    assert args.contexts == (2048, 262_144)
    assert len(cases) == 8
    assert {case.name for case in cases} == {
        f"{profile}-r{rows}-c{context}"
        for profile in ("tp1", "tp4")
        for rows in (1, 16)
        for context in (2048, 262_144)
    }


def test_cli_builds_actual_packed_qwen_prefill_geometry() -> None:
    args = benchmark_qsa._parse_args(
        [
            "--profiles",
            "tp2",
            "--rows",
            "1",
            "--prefill-rows",
            "3008",
            "--contexts",
            "2048,8192",
        ]
    )
    cases = benchmark_qsa._resolve_cases(args)
    (prefill,) = [case for case in cases if case.kind == "prefill"]

    assert prefill.name == "tp2-prefill-r3008-c8192"
    assert prefill.request_count == 1
    assert prefill.positions[0] == 5184
    assert prefill.positions[-1] == 8191
    assert prefill.active_sequence_length == 8192
    assert prefill.rank_prefix_groups == 1296


def test_all_profile_alias_and_full_context_cases_are_explicit() -> None:
    args = benchmark_qsa._parse_args(
        ["--profiles", "all", "--contexts", "2048,8192,32768,131072,full"]
    )

    assert args.profiles == ("tp1", "tp2", "tp4")
    assert args.contexts == benchmark_qsa.FULL_CONTEXTS


@pytest.mark.parametrize(
    "value",
    ["", "0", "3", "2049", "262148", "2048,2048", "not-a-context"],
)
def test_context_filter_rejects_non_qsa_geometry(value: str) -> None:
    with pytest.raises(SystemExit):
        benchmark_qsa._parse_args(["--contexts", value])


def test_cache_estimate_uses_disjoint_main_kv_and_scales_state_by_request() -> None:
    one = benchmark_qsa.BenchmarkCase(benchmark_qsa.PROFILES["tp2"], 1, 8192)
    many = benchmark_qsa.BenchmarkCase(benchmark_qsa.PROFILES["tp2"], 64, 8192)
    one_bytes = benchmark_qsa._cache_capacity_bytes(one)
    many_bytes = benchmark_qsa._cache_capacity_bytes(many)
    fp8_bytes = benchmark_qsa._cache_capacity_bytes(
        one,
        kv_cache_dtype="fp8_e4m3",
    )

    assert many_bytes["main_kv"] == 64 * one_bytes["main_kv"]
    assert many_bytes["compressed"] == 64 * one_bytes["compressed"]
    assert many_bytes["raw_state"] == 64 * one_bytes["raw_state"]
    assert fp8_bytes["main_kv"] * 2 == one_bytes["main_kv"]
    assert fp8_bytes["compressed"] == one_bytes["compressed"]
    assert fp8_bytes["raw_state"] == one_bytes["raw_state"]
    assert many.compressed_page_size == 4
    assert many.compressed_pages_per_request == 512


def test_planned_capacity_is_independent_of_live_prefill_shape() -> None:
    case = benchmark_qsa.BenchmarkCase(
        benchmark_qsa.PROFILES["tp1"],
        6_016,
        32_768,
        kind="prefill",
        main_page_size=3_008,
        planned_max_batch=4,
        planned_max_q_rows=6_019,
        planned_max_speculative_tokens=3,
    )

    assert case.request_count == 1
    assert case.planned_batch == 4
    assert case.planned_q_rows == 6_019
    assert case.max_speculative_tokens == 0
    assert case.planned_speculative_tokens == 3
    assert case.main_pages_total == 11
    assert case.main_pages_capacity == 44
    assert case.compressed_pages_capacity == 44


def test_disjoint_main_page_tables_have_unique_request_ranges() -> None:
    table = benchmark_qsa._disjoint_page_table(
        4,
        8,
        device=torch.device("cpu"),
    )

    assert tuple(table.shape) == (4, 8)
    assert int(torch.unique(table).numel()) == int(table.numel())
    assert table[0].tolist() == list(range(8))
    assert table[3].tolist() == list(range(24, 32))


def test_contract_cases_cover_fixed_capacity_tails_and_speculative_rollback() -> None:
    args = benchmark_qsa._parse_args(
        [
            "--profiles",
            "tp2",
            "--rows",
            "1",
            "--contexts",
            "2048",
            "--contract-cases",
        ]
    )
    cases = benchmark_qsa._resolve_cases(args)
    phases = [case for case in cases if case.kind == "stream_phase"]
    speculative = [case for case in cases if case.kind == "speculative"]

    assert len(cases) == 6
    assert [case.tail_length for case in phases] == [0, 1, 2, 3]
    assert [case.positions for case in phases] == [
        (8191,),
        (8188,),
        (8189,),
        (8190,),
    ]
    assert [case.active_sequence_length for case in phases] == [8192, 8189, 8190, 8191]
    assert len(speculative) == 1
    spec = speculative[0]
    assert spec.request_count == 1
    assert spec.max_speculative_tokens == 3
    assert spec.preceding_accepted_tokens == 2
    assert spec.setup_positions == (8184, 8185, 8186, 8187)
    assert spec.positions == (8186, 8187, 8188, 8189)
    assert spec.active_sequence_length == 8190
    assert spec.rank_prefix_groups == 2046


def test_selector_rank_coefficients_are_distinct_and_descending() -> None:
    coefficients = benchmark_qsa._selector_rank_coefficients(
        benchmark_qsa.BUDGET // benchmark_qsa.COMPRESS_RATIO,
        device=torch.device("cpu"),
    )

    assert coefficients.dtype == torch.bfloat16
    assert int(torch.unique(coefficients).numel()) == int(coefficients.numel())
    assert bool(torch.all(coefficients[:-1] > coefficients[1:]))


def test_rank_safe_raw_keys_are_nonzero_distinct_and_score_below_markers() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260827)
    prepared_query = torch.randn(
        (benchmark_qsa.INDEX_HEADS, benchmark_qsa.INDEX_HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
    )
    key_norm_weight = torch.linspace(
        -0.1,
        0.1,
        benchmark_qsa.INDEX_HEAD_DIM,
        dtype=torch.float32,
    )

    first = benchmark_qsa._rank_safe_raw_keys(
        prepared_query,
        key_norm_weight,
        8,
        generator=generator,
        rms_norm_eps=1e-6,
    )
    second = benchmark_qsa._rank_safe_raw_keys(
        prepared_query,
        key_norm_weight,
        8,
        generator=generator,
        rms_norm_eps=1e-6,
    )
    representatives = benchmark_qsa.gemma_rmsnorm_reference(
        first,
        key_norm_weight,
        1e-6,
    )
    scores = torch.einsum(
        "hd,nd->hn",
        prepared_query.float(),
        representatives.float(),
    )

    assert first.dtype == torch.bfloat16
    assert tuple(first.shape) == (8, benchmark_qsa.INDEX_HEAD_DIM)
    assert int(torch.count_nonzero(first)) > 0
    assert int(torch.unique(first, dim=0).shape[0]) == 8
    assert not torch.equal(first, second)
    assert bool(torch.all(scores < -0.25))


def test_selector_rank_markers_span_completed_groups_without_duplicates() -> None:
    groups = benchmark_qsa.MODEL_MAX_CONTEXT // benchmark_qsa.COMPRESS_RATIO
    count = benchmark_qsa.BUDGET // benchmark_qsa.COMPRESS_RATIO
    group_ids = benchmark_qsa._selector_rank_group_ids(
        groups,
        count,
        device=torch.device("cpu"),
    )

    assert int(torch.unique(group_ids).numel()) == count
    assert bool(torch.all(group_ids[:-1] < group_ids[1:]))
    assert int(group_ids[0]) == 0
    assert int(group_ids[-1]) == groups - 2
    assert not bool(torch.any(group_ids == groups - 1))


def test_summary_preserves_raw_distribution_endpoints() -> None:
    assert benchmark_qsa._summary([5.0, 1.0, 3.0, 2.0, 4.0]) == {
        "median_us": 3.0,
        "p10_us": 1.0,
        "p90_us": 4.0,
        "min_us": 1.0,
        "max_us": 5.0,
    }


def test_benchmark_calls_only_the_public_qsa_lifecycle() -> None:
    path = Path(benchmark_qsa.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    qsa_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "qsa"
    }

    assert {"Caps", "plan", "bind", "run"} <= qsa_calls
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("b12x.attention.qsa._")
        for node in ast.walk(tree)
    )


def test_sparse_gqa_implementation_modules_are_imported_lazily() -> None:
    repository = Path(benchmark_qsa.__file__).resolve().parents[1]
    source = textwrap.dedent(
        """
        import inspect
        import sys

        from b12x.attention.qsa import _sparse_gqa

        cute_module = "b12x.attention.qsa._sparse_gqa_cute"
        selected_paged_module = "b12x.attention.paged._selected_forward"
        assert cute_module not in sys.modules
        assert selected_paged_module not in sys.modules
        source = inspect.getsource(_sparse_gqa).lower()
        assert "_cute_is_candidate" in source
        assert "launch_selected_paged_gqa_direct" in source
        assert "notimplementederror" in source
        assert cute_module not in sys.modules
        assert selected_paged_module not in sys.modules
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_graph_contract_poison_precedes_replay_and_replay_validation() -> None:
    source = inspect.getsource(benchmark_qsa._run_case)

    output_poison = source.index('output[: case.rows].fill_(float("nan"))')
    selector_poison = source.index(
        "selected_positions[: case.rows].fill_(SELECTED_POSITION_POISON)"
    )
    scratch_poison = source.index("scratch.fill_(0xFF)")
    replay = source.index("graph.replay()")
    validation = source.index("graph_output =")
    persistent_validation = source.index("eager_persistent_state.assert_matches(")
    assert torch.iinfo(torch.int32).min == benchmark_qsa.SELECTED_POSITION_POISON
    assert (
        scratch_poison
        < output_poison
        < selector_poison
        < replay
        < validation
        < persistent_validation
    )
    assert '"replay_after_output_selector_scratch_poison": True' in source
    assert 'correctness["graph_persistent_state_exact"] = True' in source
    assert 'correctness["graph_main_kv_read_only"] = True' in source
    assert "main_k_cache=prepared.binding.main_k_cache" in source
    assert "main_v_cache=prepared.binding.main_v_cache" in source
    assert "replay_allocation_delta_bytes" in source
