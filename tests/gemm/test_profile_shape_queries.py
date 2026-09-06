import pytest
import torch

from b12x.policy import PolicyContext
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.gemm import Bf16VocabProjectionGenerator, BlockFp8LinearGenerator
from tests.conftest import require_b12x


@pytest.mark.parametrize("provider_type,query", (
    (Bf16VocabProjectionGenerator, dict(dtype="bfloat16", max_tokens=1, in_features=259, out_features=257)),
    (BlockFp8LinearGenerator, dict(output_dtype="bfloat16", max_tokens=17, in_features=384, out_features=256)),
))
def test_gemm_profile_races_qualify_declared_irregular_shapes(provider_type, query, tmp_path):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    case, = provider_type.cases_for_tuning_queries((query,))
    provider = provider_type(cases=(case,))
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


@pytest.mark.parametrize("recipe", ("nvfp4", "mxfp8"))
def test_precision_shape_factory_races_arbitrary_geometry_with_confirmation(recipe, tmp_path):
    import json

    from b12x.policy.generation.providers.blockscaled import BlockscaledPrecisionGenerator
    from b12x.policy.generation.provenance import capture_measurement_provenance

    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(recipe=recipe, in_features=384, out_features=256, measured_m=3)
    case, = BlockscaledPrecisionGenerator.cases_for_tuning_queries((query,))
    provider = BlockscaledPrecisionGenerator(cases=(case,))
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings(),
                                provenance=capture_measurement_provenance(device.index))
    with provider._benchmark_factory(case.group_id, (case,), context) as session:
        candidates = session.candidates(case)
        measurements = session.measure(case, candidates)
    (tmp_path / "precision-race.json").write_text(json.dumps({
        "query": query, "generation": context.checkpoint_metadata(),
        "measurements": [item.to_dict() for item in measurements],
    }, indent=2))
    assert len(measurements) == len(candidates) == 17
    for item in measurements:
        assert item.correct and item.error is None, item.to_dict()
        assert item.metrics["replay_allocation_bytes"] == 0
        assert item.metrics["samples_us"]
        if item.candidate.config["a16_rows"] and item.selection_eligible:
            assert item.metrics["confirmation_samples_us"]
            assert item.metrics["confirmation_clock_validation"]["valid"]
