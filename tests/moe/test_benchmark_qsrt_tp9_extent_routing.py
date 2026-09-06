from __future__ import annotations

import math

import pytest
import torch

from benchmarks.benchmark_qsrt_tp9_extent import (
    NUM_EXPERTS,
    TOP_K,
    _load_topk_ids,
    _routing_histogram,
    _uniform_topk_ids,
    _zipf_topk_ids,
)


def _expected_blocks(ids: torch.Tensor, block: int) -> int:
    rows = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=NUM_EXPERTS)
    return int(sum(math.ceil(int(r) / block) for r in rows.tolist()))


def test_histogram_reports_blocks_for_the_chosen_and_candidate_blocks() -> None:
    torch.manual_seed(71903)
    ids = _uniform_topk_ids(4608, torch.device("cpu"))
    summary = _routing_histogram(ids, 48)
    assert summary["tokens"] == 4608
    assert summary["routes"] == 4608 * TOP_K
    assert set(summary["blocks"]) == {"48", "64"}
    for block in (48, 64):
        stats = summary["blocks"][str(block)]
        assert stats["total_blocks"] == _expected_blocks(ids, block)
        assert stats["padded_slots"] == stats["total_blocks"] * block
        fractions = stats["experts_needing"]
        assert abs(sum(fractions.values()) - 1.0) < 1e-6
    # Uniform routing at 82.3 rows/expert: nearly every expert needs two
    # 48-row blocks and the mean sits between the two block widths.
    assert 80.0 <= summary["rows_per_expert"]["mean"] <= 85.0
    assert summary["blocks"]["48"]["experts_needing"]["2"] > 0.9
    assert summary["blocks"]["64"]["total_blocks"] < summary["blocks"]["48"]["total_blocks"]


def test_histogram_includes_an_extra_chosen_block() -> None:
    ids = torch.zeros((3, TOP_K), dtype=torch.int32)
    ids[:] = torch.arange(TOP_K, dtype=torch.int32)
    summary = _routing_histogram(ids, 32)
    assert set(summary["blocks"]) == {"32", "48", "64"}
    assert summary["rows_per_expert"]["max"] == 3
    assert summary["rows_per_expert"]["empty_experts"] == NUM_EXPERTS - TOP_K
    assert summary["blocks"]["32"]["total_blocks"] == TOP_K


def test_zipf_routing_is_over_dispersed_and_seeded() -> None:
    torch.manual_seed(71903)
    first = _zipf_topk_ids(1024, 1.0, torch.device("cpu"))
    torch.manual_seed(71903)
    second = _zipf_topk_ids(1024, 1.0, torch.device("cpu"))
    assert torch.equal(first, second)
    assert first.shape == (1024, TOP_K)
    assert first.dtype == torch.int32
    # Distinct experts per token, all in range.
    assert int(first.min()) >= 0 and int(first.max()) < NUM_EXPERTS
    assert all(len(set(row.tolist())) == TOP_K for row in first)
    zipf_rows = torch.bincount(first.reshape(-1).to(torch.int64), minlength=NUM_EXPERTS)
    torch.manual_seed(71903)
    uniform_rows = torch.bincount(
        _uniform_topk_ids(1024, torch.device("cpu")).reshape(-1).to(torch.int64),
        minlength=NUM_EXPERTS,
    )
    assert zipf_rows.double().std() > 2 * uniform_rows.double().std()
    with pytest.raises(ValueError):
        _zipf_topk_ids(4, 0.0, torch.device("cpu"))


def test_captured_routing_loads_tensor_and_dict_payloads(tmp_path) -> None:
    ids = torch.randint(0, NUM_EXPERTS, (37, TOP_K), dtype=torch.int64)
    tensor_path = tmp_path / "ids.pt"
    torch.save(ids, tensor_path)
    loaded, weights = _load_topk_ids(tensor_path, torch.device("cpu"))
    assert loaded.dtype == torch.int32 and loaded.shape == (37, TOP_K)
    assert torch.equal(loaded.to(torch.int64), ids)
    assert weights is None

    routing = torch.softmax(torch.randn(37, TOP_K), dim=-1)
    dict_path = tmp_path / "routing.pt"
    torch.save({"topk_ids": ids.to(torch.int32), "topk_weights": routing}, dict_path)
    loaded, weights = _load_topk_ids(dict_path, torch.device("cpu"))
    assert torch.equal(loaded.to(torch.int64), ids)
    assert weights is not None and torch.equal(weights, routing)

    bad_path = tmp_path / "bad.pt"
    torch.save(torch.zeros((5, 8), dtype=torch.int32), bad_path)
    with pytest.raises(ValueError, match=r"\[tokens, 16\]"):
        _load_topk_ids(bad_path, torch.device("cpu"))
    torch.save(torch.full((5, TOP_K), NUM_EXPERTS, dtype=torch.int32), bad_path)
    with pytest.raises(ValueError, match="must lie in"):
        _load_topk_ids(bad_path, torch.device("cpu"))
