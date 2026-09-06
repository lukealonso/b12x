import pytest
import torch

from b12x.policy import PolicyContext
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.tunable import VarlenAttentionGenerator
from tests.conftest import require_b12x


@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_production_varlen_races_use_declared_rows_and_sequence_capacities(tmp_path, dtype):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(variant="varlen", dtype=dtype, causal=True, batch_size=3,
                 q_heads=4, kv_heads=1, q_head_dim=128, v_head_dim=128,
                 query_rows=13, kv_rows=97, max_seqlen_q=8, max_seqlen_k=40)
    case, = VarlenAttentionGenerator.cases_for_tuning_queries((query,))
    provider = VarlenAttentionGenerator(cases=(case,))
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    with provider._benchmark_factory(case.group_id, (case,), context) as session:
        candidates = session.candidates(case)
        measurements = session.measure(case, candidates)
    assert len(measurements) == len(candidates) == 4
    for item in measurements:
        assert item.correct, (item.error, item.metrics)
        assert item.metrics["replay_allocation_bytes"] == 0
