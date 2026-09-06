import statistics

import pytest
import torch

from b12x.policy import PolicyContext
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.norm_sequence import HyperConnectionGenerator, MtpFeedbackGenerator, MhcGenerator
from tests.conftest import require_b12x


@pytest.mark.parametrize("provider_type", (HyperConnectionGenerator, MtpFeedbackGenerator))
@pytest.mark.parametrize("tokens", (1, 257))
def test_norm_profile_races_qualify_decode_and_irregular_prefill(provider_type, tokens, tmp_path):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(dtype="bfloat16", max_tokens=tokens, hidden_size=2560, streams=4)
    if provider_type is HyperConnectionGenerator:
        query["lowrank"] = 320
    cases = tuple(provider_type.cases_for_tuning_queries((query,)))
    provider = provider_type(cases=cases)
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    case, = cases
    with provider._benchmark_factory(case.group_id, cases, context) as session:
        candidates = session.candidates(case)
        measurements = session.measure(case, candidates)
        assert candidates and len(measurements) == len(candidates)
        for item in measurements:
            assert item.correct, (case.query, item.candidate.config, item.error, item.metrics)
            assert item.metrics["replay_allocation_bytes"] == 0
            timing = item.metrics["timing"]
            samples = timing["samples_us"]
            assert len(samples) == context.settings.groups * context.settings.repetitions
            if provider_type is MtpFeedbackGenerator:
                assert timing["aggregation"] == "median"
                assert item.latency_us == statistics.median(samples)


@pytest.mark.parametrize("tokens", (17, 385))
def test_mhc_profile_races_qualify_both_sides_of_backend_boundary(tokens, tmp_path):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(dtype="bfloat16", max_tokens=tokens, hidden_size=4096, split_k=64)
    case, = MhcGenerator.cases_for_tuning_queries((query,))
    provider = MhcGenerator(cases=(case,))
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    with provider._benchmark_factory(case.group_id, (case,), context) as session:
        candidates = session.candidates(case)
        measurements = session.measure(case, candidates)
        assert len(candidates) == (1 if tokens < 384 else 7)
        assert len(measurements) == len(candidates)
        for item in measurements:
            assert item.correct, (query, item.candidate.config, item.error, item.metrics)
            assert item.metrics["replay_allocation_bytes"] == 0
