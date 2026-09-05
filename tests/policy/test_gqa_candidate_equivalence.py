from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.policy import FrozenMapping
from b12x.policy.generation.attention_corpus import gqa_cases
from b12x.policy.generation.providers.attention import GqaAttentionGenerator
from b12x.policy.generation.providers.gpu_workers import _GqaSession, _gqa_execution_key
from b12x.policy.generation.sweep import SweepCandidate


def test_gqa_races_distinct_schedules_and_keeps_the_first_representative():
    case = gqa_cases()[0]
    session = _GqaSession((case,), SimpleNamespace())

    def capacity(_case, *, graph_ctas_per_sm, force_split_kv):
        ctas = graph_ctas_per_sm or 2
        split = force_split_kv is not False
        return SimpleNamespace(
            graph_ctas_per_sm=ctas, cta_tile_q=16, query_tiles_per_request=1,
            architecture_max_chunks_per_request=ctas * 94,
            max_chunks_per_request=2 if split else 1,
            max_work_items=2 if split else 1,
            max_partial_rows=2 if split else 0,
            max_effective_kv_pages=2, worst_page_count=2 if split else 1,
            chunk_pages_lut=(1, 1) if split else (1, 2),
        )

    session._capacity = capacity
    candidates = session.candidates(case)

    assert len(candidates) == 2
    assert {item.config["max_chunks_per_request"] for item in candidates} == {1, 2}
    assert {item.config["graph_ctas_per_sm"] for item in candidates} == {2}
    assert GqaAttentionGenerator(cases=(case,))._candidate_contract_version == 2


@pytest.mark.parametrize("field,value", (
    ("base_chunk_pages_runs", [[2, 2]]),
    ("max_work_items", 4),
    ("max_partial_rows", 4),
    ("cta_tile_q", 64),
))
def test_gqa_equivalence_retains_execution_and_workspace_fields(field, value):
    case = gqa_cases()[0]
    config = dict(
        graph_ctas_per_sm=2, architecture_max_chunks_per_request=188,
        base_chunk_pages_runs=[[2, 1]], max_work_items=2, max_partial_rows=2,
        cta_tile_q=16,
    )
    original = SweepCandidate.create(config)
    metadata_only = SweepCandidate.create({**config, "graph_ctas_per_sm": 4,
                                          "architecture_max_chunks_per_request": 376})
    changed = SweepCandidate.create({**config, field: value})

    assert _gqa_execution_key(case, original) == _gqa_execution_key(case, metadata_only)
    assert _gqa_execution_key(case, original) != _gqa_execution_key(case, changed)
    verify = replace(case, query=FrozenMapping({**case.query.to_dict(), "mode": "verify"}))
    assert _gqa_execution_key(verify, original) != _gqa_execution_key(verify, metadata_only)
