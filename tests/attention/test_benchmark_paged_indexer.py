from __future__ import annotations

import torch

from benchmarks.benchmark_paged_indexer import (
    _physical_slots,
    _validate_analytic_topk,
)


def test_analytic_topk_uses_64_bit_physical_slot_arithmetic() -> None:
    page_id = 33_554_432
    physical_slot = page_id * 64
    page_ids = torch.tensor([[page_id]], dtype=torch.int32)
    offsets = torch.zeros((1, 1), dtype=torch.int32)

    converted = _physical_slots(page_ids, offsets)

    assert converted.dtype == torch.int64
    assert converted.tolist() == [[physical_slot]]
    assert (
        _validate_analytic_topk(
            indices=converted,
            scores=None,
            seqlens=torch.ones(1, dtype=torch.int32),
            topk=1,
            analytic_scores=torch.empty(0),
            real_page_table=page_ids,
            output_physical_slots=True,
        )
        is None
    )
