"""Source ownership and resident-byte balance for whole-H128 TP9."""

import pytest

from b12x.moe._shared.qsrt_sharding import plan_qsrt_tp9_rank


def test_tp9_extents_cover_every_atom_without_crossing_scale_halves():
    totals = [0] * 9
    for layer in range(1, 93):
        owned = []
        channels = []
        for rank in range(9):
            extent = plan_qsrt_tp9_rank(layer, rank)
            first, count = extent.first_atom, extent.atom_count
            assert first % 4 == 0
            assert count in (8, 12)
            assert first // 48 == (first + count - 1) // 48
            owned.extend(range(first, first + count))
            channels.append(extent.intermediate_channels)
            totals[rank] += count
        assert sorted(owned) == list(range(96))
        assert sorted(channels) == [256] * 3 + [384] * 6
    # Full nine-layer cycles balance exactly. The two residual layers differ
    # by at most two H128 blocks, without padded expert payload.
    assert max(totals) - min(totals) <= 8
    assert sum(totals) == 92 * 96
    assert max(totals) < 92 * 12


@pytest.mark.parametrize("layer,rank", [(0, 0), (93, 0), (1, -1), (1, 9)])
def test_tp9_rejects_indices_outside_checkpoint_geometry(layer, rank):
    with pytest.raises(ValueError):
        plan_qsrt_tp9_rank(layer, rank)
