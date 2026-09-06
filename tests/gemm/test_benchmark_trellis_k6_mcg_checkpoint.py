from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.benchmark_trellis_k6_mcg_checkpoint import (
    _bound_fused_scratch_elements,
    _dot_report,
    _independent_oracle,
    _parse_params_dtype,
    _parse_rows,
    _resolve_checkpoint_binding,
    _sample_statistics,
    _run,
    _temporary_k6_mcg_grid_override,
    _tensor_metrics,
    _metrics_pass,
    parse_cuda_graph_dot,
)


def _kernel_node(symbol: str, *, cooperative: int, grid: str = "64") -> str:
    return f""""graph_1_node_0"[style="bold" shape="record" label="{{KERNEL
| {{ID | 0 (topoId: 3) | {symbol}\\<\\<\\<{grid},256,67840\\>\\>\\>}}
| {{cooperative | {cooperative}}}
}}"];
"""


def test_parse_rows_enforces_fused_boundary() -> None:
    assert _parse_rows("1,4,8,16") == (1, 4, 8, 16)
    with pytest.raises(Exception, match=r"\[1, 16\]"):
        _parse_rows("1,17")
    with pytest.raises(Exception, match="unique"):
        _parse_rows("4,4")


def test_parse_params_dtype_is_explicit_and_fail_closed() -> None:
    assert _parse_params_dtype("fp16") == torch.float16
    assert _parse_params_dtype("BF16") == torch.bfloat16
    with pytest.raises(Exception, match="fp16 or bf16"):
        _parse_params_dtype("float32")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_independent_oracle_preserves_selected_element_boundaries(
    dtype: torch.dtype,
) -> None:
    source = torch.linspace(-0.25, 0.25, 128, dtype=torch.float32).to(dtype)[None]
    raw_weight = torch.eye(128, dtype=torch.float16)
    signs = torch.ones(128, dtype=torch.float16)

    output = _independent_oracle(source, raw_weight, signs, signs)

    assert output.dtype == dtype
    assert output.shape == source.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output)


def test_checkpoint_binding_requires_one_manifest_bound_shard(tmp_path: Path) -> None:
    prefix = "model.layer.0.down_proj"
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"exact checkpoint bytes")
    weight_map = {
        f"{prefix}.{suffix}": shard.name for suffix in ("trellis", "suh", "svh", "mcg")
    }
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{shard_sha}  {shard.name}\n", encoding="utf-8"
    )

    resolved, metadata = _resolve_checkpoint_binding(tmp_path, prefix)

    assert resolved == shard
    assert metadata["expected_shard_sha256"] == shard_sha
    assert metadata["keys"]["trellis"] == f"{prefix}.trellis"


def test_cuda_dot_report_proves_one_cooperative_fused_launch(tmp_path: Path) -> None:
    symbol = (
        "kernel_cutlass_kernel_b12xgemmtrellis_linear_k6_mcg_cute"
        "K6McgSmallMKernel_object_at_test"
    )
    path = tmp_path / "fused.dot"
    path.write_text(
        "digraph dot {\n"
        + _kernel_node(symbol, cooperative=1, grid="\\{80,1,1\\}")
        + "}\n",
        encoding="utf-8",
    )

    nodes = parse_cuda_graph_dot(path)
    report = _dot_report(path, "fused_b12x")

    assert len(nodes) == 1
    assert nodes[0].grid == (80, 1, 1)
    assert nodes[0].cooperative
    assert report["pass"]
    assert report["k6_mcg_small_m_count"] == 1


def test_cuda_dot_report_rejects_noncooperative_fused_launch(tmp_path: Path) -> None:
    path = tmp_path / "fused.dot"
    path.write_text(
        "digraph dot {\n"
        + _kernel_node("prefix_K6McgSmallMKernel_suffix", cooperative=0)
        + "}\n",
        encoding="utf-8",
    )

    report = _dot_report(path, "fused_b12x")

    assert not report["pass"]
    assert "not cooperative" in " ".join(report["errors"])


def test_sample_statistics_retains_dispersion() -> None:
    report = _sample_statistics([1.0, 2.0, 3.0, 4.0])

    assert report["count"] == 4
    assert report["median_ms"] == 2.5
    assert report["mean_ms"] == 2.5
    assert report["mad_ms"] == 1.0
    assert report["p10_ms"] == pytest.approx(1.3)
    assert report["p90_ms"] == pytest.approx(3.7)


def test_run_rejects_unbalanced_route_iterations_before_cuda() -> None:
    args = SimpleNamespace(
        compile_warmups=1,
        cold_replays=1,
        warmups=1,
        iterations=5,
        replay_checks=1,
    )

    with pytest.raises(ValueError, match="multiple of 6"):
        _run(args, {})


def test_run_rejects_unbalanced_cold_replays_before_cuda() -> None:
    args = SimpleNamespace(
        compile_warmups=1,
        cold_replays=5,
        warmups=1,
        iterations=6,
        replay_checks=1,
    )

    with pytest.raises(ValueError, match="--cold-replays must be a multiple of 6"):
        _run(args, {})


def test_temporary_grid_override_restores_planner_state() -> None:
    from b12x.gemm.trellis_linear import _k6_mcg_cute

    table = _k6_mcg_cute._MEASURED_GRID_CTA
    existing_shape = (2048, 4096)
    previous = table[existing_shape]
    with (
        pytest.raises(RuntimeError, match="prepare failed"),
        _temporary_k6_mcg_grid_override(
            size_k=existing_shape[0],
            size_n=existing_shape[1],
            requested_grid_x=previous + 1,
        ) as report,
    ):
        assert table[existing_shape] == previous + 1
        assert report is not None
        raise RuntimeError("prepare failed")
    assert table[existing_shape] == previous

    absent_shape = (123, 456)
    assert absent_shape not in table
    with _temporary_k6_mcg_grid_override(
        size_k=absent_shape[0],
        size_n=absent_shape[1],
        requested_grid_x=7,
    ):
        assert table[absent_shape] == 7
    assert absent_shape not in table


def test_bound_scratch_uses_immutable_launch_contract() -> None:
    weight = SimpleNamespace(
        k6_mcg_small_m_launch=SimpleNamespace(required_scratch_elements=385_024)
    )

    assert _bound_fused_scratch_elements(weight) == 385_024


@pytest.mark.parametrize("required", (None, 0, -1))
def test_bound_scratch_rejects_missing_or_nonpositive_contract(required) -> None:
    launch = (
        None
        if required is None
        else SimpleNamespace(required_scratch_elements=required)
    )
    weight = SimpleNamespace(k6_mcg_small_m_launch=launch)

    with pytest.raises(RuntimeError, match="no positive bound"):
        _bound_fused_scratch_elements(weight)


def test_topk_gate_allows_only_numerically_ambiguous_boundary_swap() -> None:
    reference = torch.tensor([[10, 9, 8, 7, 6, 5, 4, 3.0001, 3.0]])
    actual = reference.clone()
    actual[0, 7] -= 0.0002
    actual[0, 8] += 0.0002

    metrics = _tensor_metrics(actual, reference, topk=8)
    passed, errors = _metrics_pass(
        metrics,
        min_cosine=0.99,
        max_relative_l2=0.01,
        max_abs=1.0,
        require_exact_topk=True,
    )

    assert not metrics["topk_all_rows_set_exact"]
    assert metrics["topk_ambiguity_qualified_mismatch_rows"] == 1
    assert metrics["topk_unambiguous_membership_mismatch_rows"] == 0
    assert passed
    assert not errors
