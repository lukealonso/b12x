import torch

from b12x.policy import PolicyContext
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.kda import KdaPrefillGenerator
from tests.conftest import require_b12x


def test_production_kda_races_qualify_arbitrary_capacity_and_partial_state(tmp_path):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(heads=8, head_dim=128, max_tokens=65, max_seqs=3,
                 model_dtype="bfloat16", state_dtype="float32", qk_l2norm=True, checkpoint_export=True)
    cases = tuple(KdaPrefillGenerator.cases_for_tuning_queries((query,)))
    provider = KdaPrefillGenerator(cases=cases)
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    with provider._benchmark_factory(cases[0].group_id, cases, context) as session:
        for case in cases:
            candidates = session.candidates(case)
            assert candidates
            measurements = session.measure(case, candidates)
            assert len(measurements) == len(candidates)
            for item in measurements:
                assert item.correct, (case.scenario, item.candidate.config, item.error, item.metrics)
                assert item.metrics["replay_allocation_bytes"] == 0
