from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import sqlite3
from threading import Barrier

import pytest

from b12x.policy.generation.observations import ObservationIdentity, ObservationStore
from b12x.policy.types import FrozenMapping


def _identity():
    return ObservationIdentity(
        component_id="test.gemm",
        generation=FrozenMapping({
            "source_revision": "abc", "source_sha256": "source-content",
            "toolchain": {"cutlass": "4.6.2"}, "device": {"sm_count": 48},
            "physical_device": "GPU-a", "settings": {"seed": 42, "groups": 5},
        }),
        inputs=FrozenMapping({"rows": 16, "columns": 512, "scenario": "uniform"}),
        candidates=(FrozenMapping({"tile": 64}), FrozenMapping({"tile": 128})),
        oracle_contract="independent-fp32-1", cohort="initial",
    )


@pytest.mark.parametrize("field,value", [
    ("source_revision", "def"), ("source_sha256", "different-content"),
    ("toolchain", {"cutlass": "4.6.0"}), ("device", {"sm_count": 188}),
    ("physical_device", "GPU-b"), ("settings", {"seed": 43, "groups": 5}),
])
def test_measurement_inputs_invalidate_observation(tmp_path, field, value):
    identity = _identity()
    store = ObservationStore(tmp_path / "observations.sqlite3")
    store.save(identity, {"latencies": [[1., 2.], [3., 4.]]})
    generation = identity.generation.to_dict()
    generation[field] = value
    assert store.load(replace(identity, generation=FrozenMapping(generation))) is None


def test_independent_confirmation_retains_separate_paired_samples(tmp_path):
    initial = _identity()
    confirmation = replace(initial, cohort="confirmation-1")
    store = ObservationStore(tmp_path / "observations.sqlite3")
    samples = {"paired": [[1., 2.], [2., 1.]]}
    independent = {"paired": [[2., 2.], [1., 1.]]}
    store.save(initial, samples)
    store.save(confirmation, independent)
    assert store.load(initial) == samples
    assert store.load(confirmation) == independent
    assert store.save(initial, samples) == initial.key
    with pytest.raises(ValueError, match="different samples"):
        store.save(initial, independent)


def test_candidate_order_and_inputs_are_part_of_the_paired_protocol(tmp_path):
    identity = _identity()
    store = ObservationStore(tmp_path / "observations.sqlite3")
    store.save(identity, {"correct": True})
    assert store.load(replace(identity, candidates=identity.candidates[::-1])) is None
    assert store.load(replace(identity, oracle_contract="different-oracle")) is None
    assert store.load(replace(identity, inputs=FrozenMapping({"rows": 32}))) is None


@pytest.mark.parametrize("attempt", range(5))
def test_concurrent_gpu_writers_preserve_all_observations(tmp_path, attempt):
    store = ObservationStore(tmp_path / "observations.sqlite3")
    identities = [replace(_identity(), cohort=f"confirmation-{i}") for i in range(24)]
    barrier = Barrier(4)
    def save(item):
        barrier.wait(timeout=10)
        return store.save(item, {"correct": True})
    with ThreadPoolExecutor(max_workers=4) as executor:
        keys = list(executor.map(save, identities))
    assert len(set(keys)) == 24
    assert all(store.load(item) == {"correct": True} for item in identities)


def test_tampered_observation_fails_closed(tmp_path):
    store = ObservationStore(tmp_path / "observations.sqlite3")
    identity = _identity()
    store.save(identity, {"correct": True})
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE observations SET payload_sha256 = ?", ("wrong",))
    with pytest.raises(ValueError, match="content hash"):
        store.load(identity)


def test_incomplete_provenance_cannot_become_an_observation():
    with pytest.raises(ValueError, match="incomplete"):
        replace(_identity(), generation=FrozenMapping({"source_revision": "abc"}))


def test_assigned_gpu_reference_retains_actual_physical_identity(tmp_path):
    from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
    from b12x.policy.generation.observations import measure_observation
    from b12x.policy.types import DeviceIdentity

    context = GenerationContext(device=DeviceIdentity(vendor="nvidia", product_name="test",
                                compute_capability=(12, 1), sm_count=48), device_ordinal=0,
                                work_dir=tmp_path, source_revision="test", settings=GenerationSettings(),
                                provenance=FrozenMapping({"source_sha256": "contents", "physical_device": "GPU-a",
                                                          "toolchain": {"compiler": "test"}}))
    calls = []
    store = ObservationStore(tmp_path / "observations.sqlite3")
    def measure():
        calls.append(True)
        return {"samples": [1., 2.]}
    kwargs = dict(component_id="test.race", inputs=FrozenMapping({"rows": 8}),
                  candidates=(FrozenMapping({"tile": 64}),), oracle_contract="oracle:1", store=store, measure=measure)
    first = measure_observation(context=context, **kwargs)
    coordinator = replace(context, provenance=FrozenMapping({**context.provenance, "physical_device": "GPU-b"}),
                          accepted_physical_devices=("GPU-a", "GPU-b"))
    reused = measure_observation(context=coordinator, reference=first.identity.key, **kwargs)
    assert len(calls) == 1
    assert not reused.fresh
    assert reused.identity == first.identity
    assert reused.identity.generation["physical_device"] == "GPU-a"
    with pytest.raises(ValueError, match="different measurement"):
        measure_observation(context=coordinator, reference=first.identity.key,
                            **{**kwargs, "oracle_contract": "different"})
