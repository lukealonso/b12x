import pytest
import torch

from b12x.policy import PolicyContext
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.attention import GdnAttentionGenerator
from tests.conftest import require_b12x


@pytest.mark.parametrize("value_heads,gate,norm,state_dtype", (
    (12, "silu", False, "bfloat16"),
    (12, "sigmoid", True, "float32"),
    (4, "sigmoid", False, "float32"),
))
def test_gdn_shape_races_preserve_partial_capacity_gates_and_state(value_heads, gate, norm, state_dtype, tmp_path):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(gate_activation=gate, qk_l2norm=norm, state_dtype=state_dtype,
                 key_heads=4, value_heads=value_heads, max_seqs=3, max_tokens=11, state_index_columns=4)
    cases = tuple(GdnAttentionGenerator.cases_for_tuning_queries((query,)))
    provider = GdnAttentionGenerator(cases=cases)
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    with provider._benchmark_factory(cases[0].group_id, cases, context) as session:
        for case in cases:
            candidates = session.candidates(case)
            measurements = session.measure(case, candidates)
            assert candidates and len(measurements) == len(candidates)
            for item in measurements:
                assert item.correct, (query, case.scenario, item.error, item.metrics)
                assert item.metrics["replay_allocation_bytes"] == 0
                assert item.metrics["stable_addresses"]
