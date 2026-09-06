import pytest

from b12x.moe._shared.kernels.w4a16.host import route_pack_warmup_token_counts


@pytest.mark.parametrize("capacity", [1, 5, 6, 32, 3072])
def test_route_pack_warms_one_fixed_capacity(capacity):
    assert route_pack_warmup_token_counts(capacity) == (capacity,)


def test_route_pack_warmup_rejects_empty_capacity():
    with pytest.raises(ValueError, match="capacity must be positive"):
        route_pack_warmup_token_counts(0)
