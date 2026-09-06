import pytest
import torch

from b12x.policy import PolicyContext
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.attention import MlaAttentionGenerator
from b12x.policy.generation.sweep import SweepCase
from tests.conftest import require_b12x


@pytest.mark.parametrize("mode,kv_dtype", (("decode", "bfloat16"), ("extend", "float8_e4m3fn")))
def test_dense_mla_shape_races_use_high_pool_offsets_and_partial_pages(mode, kv_dtype, tmp_path):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(mode=mode, q_dtype="bfloat16", kv_dtype=kv_dtype, num_q_heads=8,
                 qk_head_dim=576, v_head_dim=512, page_size=64, query_rows=3,
                 max_batch=3 if mode == "decode" else 1, cache_tokens=193,
                 physical_record_width=576, window_size=None, use_cuda_graph=True)
    base, = MlaAttentionGenerator.cases_for_tuning_queries((query,))
    page_bytes = 64 * 576 * (2 if kv_dtype == "bfloat16" else 1)
    page_id_base = 2**31 // page_bytes + 1
    assert page_id_base * page_bytes > 2**31
    case = SweepCase.create(group_id=base.group_id, query=query, scenario="high-page-ids",
                            metadata={"page_id_base": page_id_base})
    provider = MlaAttentionGenerator(cases=(case,))
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    with provider._benchmark_factory(case.group_id, (case,), context) as session:
        candidates = session.candidates(case)
        measurements = session.measure(case, candidates)
        assert candidates and len(measurements) == len(candidates)
        for item in measurements:
            assert item.correct, (query, item.candidate.config, item.error, item.metrics)
            assert item.metrics["replay_allocation_bytes"] == 0
