"""The BF16 collective retains all nine peers in its host/kernel ABI."""

import os

import pytest
import torch

from b12x.comm.pcie.pcie_twoshot_bf16 import _make_layout, _pad_scalar_peer_ptrs


def test_ninth_peer_is_preserved_and_eight_rank_calls_pad_with_self():
    pointers = tuple(4096 * (rank + 1) for rank in range(9))
    assert _pad_scalar_peer_ptrs(pointers, rank=8, world_size=9) == pointers
    assert _pad_scalar_peer_ptrs(pointers[:8], rank=3, world_size=8) == (
        *pointers[:8],
        pointers[3],
    )
    layout = _make_layout(max_rows=27, row_elems=896, world_size=9)
    assert layout.slot_bytes >= 27 * 896 * 2
    assert layout.slab_bytes == layout.signal_bytes + 2 * layout.slot_bytes


@pytest.mark.skipif(
    os.getenv("B12X_PCIE_TEST_NINE_RANK_MATH") != "1",
    reason="set the GPU gate to compile the nine-peer ABI",
)
def test_bf16_nine_peer_launchers_compile_for_rank_eight():
    from b12x.comm.pcie._twoshot_bf16_cute import (
        get_twoshot_bf16_allreduce_launcher,
        get_twoshot_bf16_launcher,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    for operation in ("reduce_scatter", "all_gather"):
        assert callable(
            get_twoshot_bf16_launcher(operation, 9, 8, False, 0, 256, 896, 0)
        )
    assert callable(get_twoshot_bf16_allreduce_launcher(9, 8, False, 0, 256, 896, 0))
