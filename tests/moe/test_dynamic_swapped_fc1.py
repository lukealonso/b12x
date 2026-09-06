"""Qualify swapped FC1 fragments at narrow-width and routed-tile boundaries."""

from dataclasses import replace
import json

import pytest

from tests.conftest import require_b12x


@pytest.mark.parametrize("intermediate_size", (64, 96, 128))
def test_dynamic_m64_swapped_fc1(intermediate_size, tmp_path):
    import torch

    from b12x.policy import detect_device
    from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
    from b12x.policy.generation.moe_corpus import expand_physical_geometries, expand_sweep_cases
    from b12x.policy.generation.providers.moe_gpu_worker import _MoeGeometrySession

    require_b12x()
    ordinal = torch.cuda.current_device()
    context = GenerationContext(device=detect_device(ordinal).identity,
                                device_ordinal=ordinal, work_dir=tmp_path,
                                source_revision="gpu-regression", settings=GenerationSettings())
    geometry = next(item for item in expand_physical_geometries()
                    if item.key == ("modelopt-nvfp4", "silu", 256, 2048, 64))
    geometry = replace(geometry, intermediate_size=intermediate_size)
    cases = expand_sweep_cases(geometries=(geometry,), top_ks=(8,),
                              token_counts=(1, 7, 8, 9, 64, 128))
    with _MoeGeometrySession(geometry, context) as session:
        candidates = tuple(candidate for candidate in session.candidates
                           if candidate.config["backend"] == "dynamic"
                           and candidate.config["dynamic_tile_m"] == 64
                           and candidate.config["dynamic_route_mode"] == "grouped")
        assert len(candidates) == 1
        for case in cases:
            if case.route_pattern not in ("balanced", "hot"):
                continue
            inputs = session._stage_query_inputs(case)
            expected_before = session._independent_reference(
                x=inputs.x, topk_ids=inputs.topk_ids, topk_weights=inputs.topk_weights)
            measurements = session.measure(case, candidates, correctness=True)
            expected_after = session._independent_reference(
                x=inputs.x, topk_ids=inputs.topk_ids, topk_weights=inputs.topk_weights)
            torch.testing.assert_close(expected_before, expected_after, rtol=0, atol=0)
            record = {"query": case.query(), "route_pattern": case.route_pattern,
                      "measurements": [item.to_dict() for item in measurements]}
            print(json.dumps(record), flush=True)
            assert len(measurements) == 1
            measured = measurements[0]
            assert measured.passes(context.settings.minimum_cosine), record
            assert measured.metrics["comparison"] == "independent_nvfp4"
            assert measured.metrics["finite"]
            assert measured.metrics["output_nonzero"] > 0
            assert measured.metrics["allocation_delta_bytes"] == 0
