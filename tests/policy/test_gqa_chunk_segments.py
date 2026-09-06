"""Compressed schedule arithmetic matches the scalar planner at every page."""

import itertools

import pytest
import torch

from b12x.attention.paged import planner
from b12x.attention.paged._policy import _compress_lut, _factor_chunk_pages_lut


def test_segment_maximum_preserves_the_first_witness():
    for divisor, floor, low, width in itertools.product(range(1, 25), (1, 3, 17), (1, 7, 31, 64, 99), (0, 1, 30, 100)):
        high = low + width
        chunks = [(page + max(floor, (page + divisor - 1) // divisor) - 1)
                  // max(floor, (page + divisor - 1) // divisor) for page in range(low, high + 1)]
        assert planner._chunk_segment_maximum(low, high, divisor, floor) == (max(chunks), low + chunks.index(max(chunks)))


@pytest.mark.parametrize("forced,minimum", ((None, None), (3, None), (None, 7), (5, 17)))
def test_segment_expansion_and_factoring_match_scalar_policy(monkeypatch, forced, minimum):
    for name, value in (("B12X_PAGED_DECODE_GRAPH_CHUNK_PAGES", forced),
                        ("B12X_PAGED_DECODE_GRAPH_MIN_CHUNK_PAGES", minimum)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))
    for dtype, batch, geometry, cap in itertools.product(
        (torch.bfloat16, torch.float8_e4m3fn), (1, 2, 3, 4, 8), ((128, 128, 6), (256, 256, 8)), (None, 1, 13, 94),
    ):
        qk, vo, group = geometry
        kwargs = dict(q_dtype=torch.bfloat16, kv_dtype=dtype, batch=batch, page_size=128,
                      head_dim_qk=qk, head_dim_vo=vo, gqa_group_size=group, max_chunks_per_req=cap)
        expected = tuple(planner.decode_chunk_pages_for_graph(**kwargs, max_effective_kv_pages=page)
                         for page in range(1, 1026))
        actual = planner.build_decode_chunk_pages_lut(**kwargs, max_effective_kv_pages=1025)
        assert actual == expected
        segments = planner._decode_chunk_segments(**kwargs, max_effective_kv_pages=1025)
        for max_chunks in (1, 7, 23, 96):
            expected_runs = _compress_lut(_factor_chunk_pages_lut(expected, max_chunks_per_request=max_chunks))
            assert planner._factored_chunk_segment_runs(segments, max_chunks) == expected_runs
