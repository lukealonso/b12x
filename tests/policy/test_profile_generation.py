from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from b12x.policy import EMBEDDED_REGISTRY, ComponentPolicy, DeviceIdentity
from b12x.policy.generation import (
    CheckpointStore,
    ComponentGenerationResult,
    ComponentGeneratorRegistry,
    GenerationContext,
    GenerationSettings,
    MeasurementPartition,
    measurement_partitions,
    ProgressReporter,
    select_measurement_partitions,
    WorkEstimate,
)
from b12x.policy.generation import parallel
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.measured import (
    GpuProbeMeasurement,
    MeasuredPolicyGenerator,
)
from b12x.policy.generation.runner import (
    generate_profile_artifact,
    merge_profile_artifacts,
    write_artifact_atomic,
)
from b12x.tools.generate_gpu_profile import (
    _is_generated_profile_data,
    _parse_devices,
    _parse_partition_shard,
    _parser,
    _profile_id_for_device,
    _select_partition_shard,
)

_DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="Synthetic GPU",
)


def test_checkpoint_store_treats_truncated_checkpoint_as_cache_miss(
    tmp_path: Path,
) -> None:
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    path = checkpoints.save("test.component", "case", {"complete": True})
    path.write_text("", encoding="utf-8")

    assert checkpoints.load("test.component", "case") is None

    checkpoints.save("test.component", "case", {"complete": True})
    assert checkpoints.load("test.component", "case") == {"complete": True}


def test_default_measurement_protocol_is_two_warmups_and_five_by_five() -> None:
    settings = GenerationSettings()
    args = _parser().parse_args([])

    assert (settings.warmup, settings.groups, settings.repetitions) == (2, 5, 5)
    assert (args.warmup, args.groups, args.repetitions) == (2, 5, 5)
    assert settings.minimum_cosine == args.minimum_cosine == 0.998


def test_relaxed_cosine_gate_resumes_stricter_checkpoints(tmp_path) -> None:
    relaxed = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="relaxed",
        settings=GenerationSettings(),
    )
    strict = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="strict",
        settings=GenerationSettings(minimum_cosine=0.999),
    )

    assert relaxed.checkpoint_metadata_matches(strict.checkpoint_metadata())
    assert not strict.checkpoint_metadata_matches(relaxed.checkpoint_metadata())


def test_parallel_device_ranges_expand_without_duplicates() -> None:
    assert _parse_devices("0-3,5,cuda:7") == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:5",
        "cuda:7",
    )
    with pytest.raises(ValueError, match="duplicate"):
        _parse_devices("0-2,2")
    with pytest.raises(ValueError, match="ascending"):
        _parse_devices("3-1")


def test_partition_shard_parser_uses_human_friendly_indices() -> None:
    assert _parse_partition_shard("1/3") == (0, 3)
    assert _parse_partition_shard("3/3") == (2, 3)
    with pytest.raises(ValueError, match="look like"):
        _parse_partition_shard("0/3")
    with pytest.raises(ValueError, match="exceed"):
        _parse_partition_shard("4/3")


def test_default_profile_id_reuses_embedded_multi_target_profile() -> None:
    profile = EMBEDDED_REGISTRY.get("nvidia.rtx.pro.6000.blackwell")

    assert {
        _profile_id_for_device(target) for target in profile.targets
    } == {profile.profile_id}


def test_parallel_worker_reports_ready_after_initialization(monkeypatch) -> None:
    events = []
    device = SimpleNamespace(identity=object(), ordinal=3)

    class DeviceQueue:
        def get(self):
            return "cuda:3"

    class ProgressQueue:
        def put(self, event):
            events.append(event)

    monkeypatch.setattr(parallel, "detect_device", lambda _spec: device)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _ordinal: None)
    for name in (
        "_WORKER_DEVICE",
        "_WORKER_PROGRESS_QUEUE",
        "_WORKER_REGISTRY",
        "_WORKER_STOP_EVENT",
    ):
        monkeypatch.setattr(parallel, name, None)

    parallel._initialize_worker(
        DeviceQueue(),
        ProgressQueue(),
        object(),
        ComponentGeneratorRegistry,
    )

    assert events == [parallel._WorkerReady(device_ordinal=3)]


@dataclass(frozen=True)
class _Generator:
    component_id: str
    query_schema_version: int = 1
    config_schema_version: int = 1

    def estimate(self, context: GenerationContext) -> WorkEstimate:
        del context
        return WorkEstimate(
            component_id=self.component_id,
            work_units=1,
            case_count=1,
            description="synthetic",
            dimensions={"cases": 1},
        )

    def generate(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> ComponentGenerationResult:
        del context
        progress.start_stage(self.component_id, stage="race", total=1)
        progress.advance(self.component_id, detail="synthetic-case")
        checkpoints.save(self.component_id, "synthetic-case", {"done": True})
        return ComponentGenerationResult(
            component={
                "component_id": self.component_id,
                "query_schema_version": 1,
                "config_schema_version": 1,
                "rules": [
                    {
                        "name": "synthetic",
                        "exact": {"rows": 1},
                        "ranges": {},
                        "config": {"backend": "synthetic"},
                    }
                ],
            },
            evidence={"gpu_measurement_cases": 1},
            completed_work_units=1,
        )


@dataclass(frozen=True)
class _PartitionedGenerator(_Generator):
    partition_ids: tuple[str, ...] = ("large-a", "large-b", "small-a", "small-b")

    def measurement_partitions(
        self,
        context: GenerationContext,
    ) -> tuple[MeasurementPartition, ...]:
        del context
        work_units = {
            "large-a": 10,
            "large-b": 9,
            "small-a": 2,
            "small-b": 1,
        }
        return tuple(
            MeasurementPartition(
                component_id=self.component_id,
                partition_id=partition_id,
                work_units=work_units[partition_id],
                case_count=work_units[partition_id],
                description=partition_id,
            )
            for partition_id in self.partition_ids
        )

    def select_measurement_partitions(
        self,
        partition_ids: tuple[str, ...],
    ) -> _PartitionedGenerator:
        return _PartitionedGenerator(
            component_id=self.component_id,
            partition_ids=partition_ids,
        )


def test_nonpartitionable_generator_is_one_parallel_work_item(tmp_path) -> None:
    generator = _Generator("attention.gqa")
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )

    partitions = measurement_partitions((generator,), context)

    assert len(partitions) == 1
    assert partitions[0].partition_id == "full-component"
    assert (
        select_measurement_partitions(
            generator,
            (partitions[0].partition_id,),
        )
        is generator
    )


def test_cross_host_partition_shards_are_complete_disjoint_and_balanced(
    tmp_path,
) -> None:
    generator = _PartitionedGenerator("moe.decode")
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )

    first = _select_partition_shard(
        (generator,),
        context,
        shard_index=0,
        shard_count=2,
    )[0]
    second = _select_partition_shard(
        (generator,),
        context,
        shard_index=1,
        shard_count=2,
    )[0]

    first_ids = set(first.partition_ids)
    second_ids = set(second.partition_ids)
    assert first_ids.isdisjoint(second_ids)
    assert first_ids | second_ids == set(generator.partition_ids)
    assert sum(item.work_units for item in first.measurement_partitions(context)) == 11
    assert sum(item.work_units for item in second.measurement_partitions(context)) == 11


def test_registry_and_runner_assemble_all_components(tmp_path) -> None:
    registry = ComponentGeneratorRegistry()
    registry.register(_Generator("attention.gqa"))
    registry.register(_Generator("moe.decode"))
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )

    artifact = generate_profile_artifact(
        profile_id="nvidia.synthetic.48sm",
        generators=registry.select(None),
        context=context,
        progress=NullProgressReporter(),
    )

    profile = artifact["profile"]
    assert [component["component_id"] for component in profile["components"]] == [
        "attention.gqa",
        "moe.decode",
    ]
    assert set(artifact["evidence"]["components"]) == {
        "attention.gqa",
        "moe.decode",
    }
    assert (tmp_path / "checkpoints" / "moe.decode" / "synthetic-case.json").is_file()


def test_compact_profile_writer_round_trips_runtime_payload(tmp_path) -> None:
    path = tmp_path / "profile.json"
    payload = {"profile_id": "synthetic", "components": []}

    write_artifact_atomic(path, payload, overwrite=False, compact=True)

    assert json.loads(path.read_text()) == payload
    assert path.read_text().count("\n") == 1


def test_partial_artifact_merge_replaces_only_generated_components(
    tmp_path,
) -> None:
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )
    base = generate_profile_artifact(
        profile_id="nvidia.synthetic.48sm",
        generators=(_Generator("attention.gqa"), _Generator("moe.decode")),
        context=context,
        progress=NullProgressReporter(),
    )
    update = generate_profile_artifact(
        profile_id="nvidia.synthetic.48sm",
        generators=(_Generator("moe.decode"),),
        context=context,
        progress=NullProgressReporter(),
    )

    merged = merge_profile_artifacts(base, update)

    assert [
        component["component_id"] for component in merged["profile"]["components"]
    ] == ["attention.gqa", "moe.decode"]
    assert set(merged["evidence"]["components"]) == {
        "attention.gqa",
        "moe.decode",
    }


def test_partial_artifact_merge_preserves_base_target_aliases(tmp_path) -> None:
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )
    base = generate_profile_artifact(
        profile_id="nvidia.synthetic.48sm",
        generators=(_Generator("attention.gqa"), _Generator("moe.decode")),
        context=context,
        progress=NullProgressReporter(),
    )
    alias = DeviceIdentity(
        vendor=_DEVICE.vendor,
        compute_capability=_DEVICE.compute_capability,
        sm_count=_DEVICE.sm_count,
        product_name="Synthetic GPU Alias",
    )
    base["profile"]["targets"].append(
        {
            "vendor": alias.vendor,
            "compute_capability": list(alias.compute_capability),
            "sm_count": alias.sm_count,
            "product_name": alias.product_name,
        }
    )
    update = generate_profile_artifact(
        profile_id="nvidia.synthetic.48sm",
        generators=(_Generator("moe.decode"),),
        context=context,
        progress=NullProgressReporter(),
    )

    merged = merge_profile_artifacts(base, update)

    assert len(merged["profile"]["targets"]) == 2


def test_source_fingerprint_excludes_generated_profile_payloads() -> None:
    assert _is_generated_profile_data(
        Path("b12x/policy/_profiles/data/nvidia.gb10.48sm.json")
    )
    assert not _is_generated_profile_data(
        Path("b12x/policy/_profiles/data/__init__.py")
    )


@dataclass(frozen=True)
class _MeasuredQuery:
    rows: int


@dataclass(frozen=True)
class _MeasuredConfig:
    backend: str


@dataclass
class _Probe:
    case_ids: tuple[str, ...]
    calls: list[tuple[str, ...]]

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    @property
    def description(self) -> str:
        return "synthetic fixed-backend qualification"

    def __call__(self, context):
        del context
        self.calls.append(self.case_ids)
        return tuple(
            GpuProbeMeasurement(
                label=case_id,
                latency_us=1.0,
                correct=True,
            )
            for case_id in self.case_ids
        )


def _measured_generator(*, backend: str, probe: _Probe):
    policy = ComponentPolicy(
        component_id="test.measured",
        query_schema_version=1,
        config_schema_version=1,
        query_fields=frozenset({"rows"}),
        config_fields=frozenset({"backend"}),
        encode_query=lambda query: {"rows": query.rows},
        decode_profile=lambda payload: _MeasuredConfig(backend=str(payload["backend"])),
        heuristic=lambda _query, _device: _MeasuredConfig(backend=backend),
        validate_config=lambda _query, _config, _device: None,
    )
    return MeasuredPolicyGenerator(
        policy=policy,
        queries=(_MeasuredQuery(rows=1),),
        encode_config=lambda config: {"backend": config.backend},
        probe=probe,
    )


def test_measured_generator_resume_tracks_case_ids_and_config(tmp_path) -> None:
    calls = []
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    probe = _Probe(("case-a",), calls)
    generator = _measured_generator(backend="fixed", probe=probe)

    generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    assert calls == [("case-a",)]

    changed_source = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="def456",
        settings=GenerationSettings(),
    )
    generator.generate(
        changed_source,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    assert calls == [("case-a",)]

    changed_probe = _Probe(("case-b",), calls)
    _measured_generator(backend="fixed", probe=changed_probe).generate(
        changed_source,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    _measured_generator(backend="new-fixed", probe=changed_probe).generate(
        changed_source,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    assert calls == [("case-a",), ("case-b",), ("case-b",)]

    checkpoint = checkpoints.load("test.measured", "production-qualification")
    assert checkpoint is not None
    assert checkpoint["schema_version"] == 2
    assert checkpoint["case_ids"] == ["case-b"]
    assert checkpoint["config"] == {"backend": "new-fixed"}
