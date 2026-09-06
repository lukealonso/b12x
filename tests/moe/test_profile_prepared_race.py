"""Qualify shared candidate timing and census transport through the GPU worker."""

import pytest
import torch

from b12x.policy import detect_device
from b12x.policy.components import MOE_DECODE
from b12x.policy.generation.census import inventory_observations
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.moe_corpus import expand_physical_geometries, expand_sweep_cases
from b12x.policy.generation.observations import ObservationStore, measure_observation
from b12x.policy.generation.provenance import capture_measurement_provenance
from b12x.policy.generation.providers.moe_gpu_worker import _MoeProcessSession
from b12x.policy.types import FrozenMapping
from tests.conftest import require_b12x


@pytest.mark.parametrize("clock", ["cuda_event", "globaltimer"])
def test_moe_worker_races_retained_graphs_and_records_compilation_requests(tmp_path, clock):
    require_b12x()
    ordinal = torch.cuda.current_device()
    context = GenerationContext(device=detect_device(ordinal).identity, device_ordinal=ordinal,
        work_dir=tmp_path, source_revision="gpu-regression", settings=GenerationSettings(timing_clock=clock),
        provenance=capture_measurement_provenance(ordinal))
    geometry = next(item for item in expand_physical_geometries()
                    if item.key == ("modelopt-nvfp4", "silu", 256, 2048, 64))
    cases = tuple(case for case in expand_sweep_cases(geometries=(geometry,), top_ks=(8,), token_counts=(9,))
                  if case.route_pattern in ("balanced", "hot"))
    store = ObservationStore(tmp_path / "observations.sqlite3")
    with _MoeProcessSession(geometry, context) as session:
        candidates = tuple(candidate for candidate in session.eligible_candidates(cases[0], session.candidates)
                           if candidate.config["backend"] == "dynamic")[:2]
        assert len(candidates) == 2
        for case in cases:
            def measure():
                measurements = session.measure(case, candidates, correctness=True)
                for item in measurements:
                    assert item.passes(context.settings.minimum_cosine), item.to_dict()
                    assert item.metrics["timing"]["protocol"] == "balanced_candidate_replay_v1"
                    assert item.metrics["timing"]["clock"] == clock
                    assert item.metrics["replay_allocation_bytes"] == 0
                return {"measurements": [item.to_dict() for item in measurements]}
            observed = measure_observation(context=context, component_id=MOE_DECODE,
                inputs=FrozenMapping({"query": case.query(), "scenario": case.route_pattern}),
                candidates=tuple(candidate.config for candidate in candidates),
                oracle_contract="independent_nvfp4", store=store, measure=measure)
            assert observed.result["compilation_requests"]
            assert store.load(observed.identity) == observed.result
    inventory = inventory_observations((store.path,))
    assert inventory["counts"]["observations"] == 2
    assert inventory["counts"]["specializations"] > 0
    assert inventory["counts"]["unrecorded_observations"] == 0
