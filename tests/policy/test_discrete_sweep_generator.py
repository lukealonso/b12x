from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace

import pytest

from b12x.policy import DeviceIdentity
from b12x.policy.generation import (
    CheckpointStore,
    DiscreteSweepGenerator,
    GenerationContext,
    GenerationSettings,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.serialization import profile_from_dict
from b12x.policy.types import FrozenMapping

_DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="Synthetic GPU",
)


def test_shared_observations_survive_stage_checkpoint_removal(tmp_path):
    calls, candidate_calls, session_calls = [], [], []
    generator = DiscreteSweepGenerator(
        component_id="test.attention", query_schema_version=1, config_schema_version=1,
        query_fields=("family", "rows"), range_fields=frozenset({"rows"}), cases=_cases(),
        benchmark_factory=_Factory(calls, candidate_calls, session_calls), coverage={},
    )
    context = GenerationContext(
        device=_DEVICE, device_ordinal=0, work_dir=tmp_path, source_revision="source",
        settings=GenerationSettings(), provenance=FrozenMapping({
            "source_sha256": "content", "physical_device": "GPU-a", "toolchain": {"cutlass": "4.6.2"},
        }),
    )
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    first = generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert first.evidence["fresh_measurement_cases"] == 4
    assert first.evidence["reused_measurement_cases"] == 0
    for case in _cases():
        checkpoint = checkpoints.load(generator.component_id, case.case_id)
        assert checkpoint["schema_version"] == 3
        assert "observation_key" in checkpoint and "measurements" not in checkpoint
        checkpoints._path(generator.component_id, case.case_id).unlink()
    second = generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert second.component == first.component
    assert len(calls) == 4
    assert second.evidence["fresh_measurement_cases"] == 0
    assert second.evidence["reused_measurement_cases"] == 4
    third = generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert third.component == first.component
    assert len(session_calls) == 2

    changed = replace(context, provenance=FrozenMapping({
        **context.provenance.to_dict(), "physical_device": "GPU-b",
    }))
    fourth = generator.generate(changed, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert fourth.evidence["fresh_measurement_cases"] == 4
    assert len(calls) == 8


@pytest.mark.parametrize("search", (False, True))
def test_external_binary_identity_invalidates_sweep_and_search_observations(tmp_path, search):
    import hashlib
    from types import SimpleNamespace

    from b12x.policy.generation.engine import DiscreteSearch
    from b12x.policy.generation.search import SearchBudget, SearchStrategy
    from b12x.policy.problem import define_problem

    binary = tmp_path / "dependency.so"
    binary.write_bytes(b"dependency implementation one")
    calls, candidate_calls, session_calls = [], [], []
    generator = DiscreteSweepGenerator(
        component_id="test.attention", query_schema_version=1, config_schema_version=1,
        query_fields=("family", "rows"), range_fields=frozenset({"rows"}), cases=_cases(),
        benchmark_factory=_Factory(calls, candidate_calls, session_calls), coverage={},
    )
    def measurement_context(context):
        provenance = context.provenance.to_dict()
        dependencies = {**provenance.get("external_dependencies", {}),
                        "dependency": {"sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}}
        return replace(context, provenance=FrozenMapping({**provenance, "external_dependencies": dependencies}))

    generator.measurement_context = measurement_context
    context = GenerationContext(
        device=_DEVICE, device_ordinal=0, work_dir=tmp_path, source_revision="source",
        settings=GenerationSettings(), provenance=FrozenMapping({
            "source_sha256": "content", "physical_device": "GPU-a", "toolchain": {"cutlass": "4.6.2"},
            "external_dependencies": {"upstream": {"sha256": "upstream-binary"}},
        }),
    )
    assert generator.measurement_context(context).provenance["external_dependencies"]["upstream"] == FrozenMapping(
        {"sha256": "upstream-binary"})
    checkpoints = CheckpointStore(tmp_path / "checkpoints")

    @dataclass(frozen=True)
    class Query:
        family: str
        rows: int

    @dataclass(frozen=True)
    class Config:
        backend: str

    problem = define_problem(policy=SimpleNamespace(component_id=generator.component_id),
                             query_type=Query, config_type=Config, axes=("rows",), family=("family",),
                             decisions={"backend": ("left", "right")})

    def run():
        if search:
            with DiscreteSearch(generator, problem, context, checkpoints) as adapter:
                adapter.search(strategy=SearchStrategy.EXHAUSTIVE, budget=SearchBudget(queries=10))
        else:
            result = generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
            assert result.evidence["external_dependencies"] == generator.measurement_context(
                context).provenance.to_dict()["external_dependencies"]

    run()
    run()
    assert len(calls) == 4
    binary.write_bytes(b"dependency implementation two")
    run()
    assert len(calls) == 8
    run()
    assert len(calls) == 8


class _Session(AbstractContextManager["_Session"]):
    def __init__(
        self,
        calls: list[str],
        candidate_calls: list[str],
    ) -> None:
        self._calls = calls
        self._candidate_calls = candidate_calls
        self._candidates = (
            SweepCandidate.create({"backend": "left"}),
            SweepCandidate.create({"backend": "right"}),
        )

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def candidates(self, case):
        self._candidate_calls.append(case.case_id)
        return self._candidates

    def measure(self, case, candidates):
        self._calls.append(case.case_id)
        measurements = []
        for candidate in candidates:
            backend = candidate.config["backend"]
            if case.query["rows"] == 1:
                latency = 10.0 if backend == "left" else 20.0
            else:
                latency = 30.0 if backend == "left" else 15.0
            if case.scenario == "strided":
                latency *= 1.1
            measurements.append(
                SweepMeasurement(
                    candidate=candidate,
                    latency_us=latency,
                    correct=True,
                    metrics={"cosine": 0.9995},
                )
            )
        return tuple(measurements)


@dataclass
class _Factory:
    calls: list[str]
    candidate_calls: list[str]
    session_calls: list[str]

    def __call__(self, group_id, cases, context):
        self.session_calls.append(group_id)
        del cases, context
        return _Session(self.calls, self.candidate_calls)


def _cases():
    return tuple(
        SweepCase.create(
            group_id="geometry",
            query={"family": "a", "rows": rows},
            scenario=scenario,
            label=f"m{rows}-{scenario}",
        )
        for rows in (1, 4)
        for scenario in ("contiguous", "strided")
    )


@pytest.mark.parametrize("migration", (
    "subset", "missing", "undeclared", "settings", "boolean_contract",
    "float_contract", "boolean_schema", "legacy_subset",
))
def test_candidate_contract_migration_reuses_only_exact_recorded_candidates(
    tmp_path, migration,
) -> None:
    calls, candidate_calls, session_calls = [], [], []
    factory = _Factory(calls, candidate_calls, session_calls)
    arguments = dict(
        component_id="test.attention", query_schema_version=1, config_schema_version=1,
        query_fields=("family", "rows"), range_fields=frozenset({"rows"}),
        cases=_cases(), coverage={},
    )
    context = GenerationContext(
        device=_DEVICE, device_ordinal=0, work_dir=tmp_path,
        source_revision="measured-source", settings=GenerationSettings(),
    )
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    original = DiscreteSweepGenerator(benchmark_factory=factory, **arguments)
    original.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    saved = checkpoints.load("test.attention", _cases()[0].case_id)
    calls.clear()
    changes = {
        "boolean_contract": {"candidate_contract_version": True},
        "float_contract": {"candidate_contract_version": 1.0},
        "boolean_schema": {"schema_version": True},
        "legacy_subset": {"schema_version": 1},
    }.get(migration)
    if changes is not None:
        for case in _cases():
            payload = checkpoints.load("test.attention", case.case_id)
            checkpoints.save("test.attention", case.case_id, {**payload, **changes})

    def reduced_factory(group_id, cases, context):
        session = factory(group_id, cases, context)
        session._candidates = (
            SweepCandidate.create({"backend": "unmeasured"})
            if migration == "missing" else session._candidates[1],
        )
        return session

    reduced = DiscreteSweepGenerator(
        benchmark_factory=reduced_factory, candidate_contract_version=2,
        subset_reuse_contract_versions=() if migration == "undeclared" else (1,),
        **arguments,
    )
    if migration == "settings":
        context = replace(context, settings=GenerationSettings(groups=6))
    reduced.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)

    upgraded = checkpoints.load("test.attention", _cases()[0].case_id)
    assert upgraded["candidate_contract_version"] == 2
    assert len(upgraded["candidate_ids"]) == 1
    if migration == "subset":
        assert calls == []
        assert upgraded["measurements"] == saved["measurements"][1:]
        assert upgraded["generation"] == saved["generation"]
    else:
        assert len(calls) == len(_cases())
    enumerations = len(candidate_calls)
    measurements = len(calls)
    reduced.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert len(candidate_calls) == enumerations
    assert len(calls) == measurements


def test_discrete_sweep_partitions_preserve_allocation_groups(tmp_path) -> None:
    calls = []
    candidate_calls = []
    session_calls = []
    cases = (
        SweepCase.create(
            group_id="geometry-a",
            query={"family": "a", "rows": 1},
        ),
        SweepCase.create(
            group_id="geometry-a",
            query={"family": "a", "rows": 4},
        ),
        SweepCase.create(
            group_id="geometry-b",
            query={"family": "b", "rows": 1},
        ),
    )
    generator = DiscreteSweepGenerator(
        component_id="test.attention",
        query_schema_version=1,
        config_schema_version=1,
        query_fields=("family", "rows"),
        range_fields=frozenset({"rows"}),
        cases=cases,
        benchmark_factory=_Factory(calls, candidate_calls, session_calls),
        coverage={},
    )
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )

    partitions = generator.measurement_partitions(context)
    restricted = generator.select_measurement_partitions(("geometry-a",))

    assert [(item.partition_id, item.case_count) for item in partitions] == [
        ("geometry-a", 2),
        ("geometry-b", 1),
    ]
    assert restricted.estimate(context).case_count == 2
    with pytest.raises(ValueError, match="missing"):
        generator.select_measurement_partitions(("missing",))


def test_discrete_sweep_reduces_scenarios_and_resumes(tmp_path) -> None:
    calls = []
    candidate_calls = []
    session_calls = []
    generator = DiscreteSweepGenerator(
        component_id="test.attention",
        query_schema_version=1,
        config_schema_version=1,
        query_fields=("family", "rows"),
        range_fields=frozenset({"rows"}),
        cases=_cases(),
        benchmark_factory=_Factory(calls, candidate_calls, session_calls),
        coverage={"corpus_sha256": "synthetic"},
    )
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )
    checkpoints = CheckpointStore(tmp_path / "checkpoints")

    result = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    first_call_count = len(calls)
    resumed = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )

    assert first_call_count == 4
    assert len(calls) == first_call_count
    assert len(candidate_calls) == first_call_count
    assert len(session_calls) == 1
    assert result.component == resumed.component

    source_changed_context = replace(
        context,
        source_revision="def456",
        settings=GenerationSettings(warmup=1, repetitions=3, groups=3),
    )
    generator.generate(
        source_changed_context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    assert len(calls) == 2 * first_call_count
    assert len(candidate_calls) == 2 * first_call_count
    assert len(session_calls) == 2
    checkpoint = checkpoints.load("test.attention", _cases()[0].case_id)
    assert checkpoint is not None
    generation = checkpoint["generation"]
    assert isinstance(generation, dict)
    assert generation["source_revision"] == "def456"
    assert generation["settings"] == source_changed_context.settings.to_dict()

    generator.generate(
        source_changed_context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    assert len(calls) == 2 * first_call_count
    assert len(candidate_calls) == 2 * first_call_count
    assert len(session_calls) == 2

    changed_context = replace(
        source_changed_context,
        settings=GenerationSettings(repetitions=31),
    )
    generator.generate(
        changed_context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    assert len(calls) == 3 * first_call_count
    assert len(candidate_calls) == 3 * first_call_count
    assert len(session_calls) == 3

    changed_contract = DiscreteSweepGenerator(
        component_id="test.attention",
        query_schema_version=1,
        config_schema_version=1,
        query_fields=("family", "rows"),
        range_fields=frozenset({"rows"}),
        cases=_cases(),
        benchmark_factory=_Factory(calls, candidate_calls, session_calls),
        coverage={"corpus_sha256": "synthetic"},
        candidate_contract_version=2,
    )
    changed_contract.generate(
        changed_context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    assert len(calls) == 4 * first_call_count
    assert len(candidate_calls) == 4 * first_call_count
    assert len(session_calls) == 4
    profile = profile_from_dict(
        {
            "profile_id": "nvidia.synthetic.48sm",
            "targets": [
                {
                    "vendor": "nvidia",
                    "compute_capability": [12, 1],
                    "sm_count": 48,
                    "product_name": "Synthetic GPU",
                }
            ],
            "components": [result.component],
        }
    )
    component = profile.component("test.attention")
    assert component is not None
    assert component.lookup({"family": "a", "rows": 1}).config["backend"] == "left"
    assert component.lookup({"family": "a", "rows": 4}).config["backend"] == "right"


def test_shared_search_adapter_preserves_all_scenarios_and_production_candidates(tmp_path):
    from types import SimpleNamespace
    from b12x.policy.problem import define_problem
    from b12x.policy.generation.engine import DiscreteSearch
    from b12x.policy.generation.search import SearchBudget, SearchStrategy

    @dataclass(frozen=True)
    class Query:
        family: str
        rows: int

    @dataclass(frozen=True)
    class Config:
        backend: str

    problem = define_problem(policy=SimpleNamespace(component_id='test.attention'), query_type=Query,
                             config_type=Config, axes=('rows',), family=('family',),
                             decisions={'backend': ('left', 'right')})
    calls, candidate_calls, session_calls = [], [], []
    generator = DiscreteSweepGenerator(
        component_id='test.attention', query_schema_version=1, config_schema_version=1,
        query_fields=('family', 'rows'), range_fields=frozenset({'rows'}), cases=_cases(),
        benchmark_factory=_Factory(calls, candidate_calls, session_calls), coverage={},
    )
    context = GenerationContext(device=_DEVICE, device_ordinal=0, work_dir=tmp_path,
                                source_revision='source', settings=GenerationSettings())
    checkpoints = CheckpointStore(tmp_path/'checkpoints')
    with DiscreteSearch(generator, problem, context, checkpoints) as adapter:
        outcome = adapter.search(strategy=SearchStrategy.EXHAUSTIVE, budget=SearchBudget(queries=10))
        winners = {m.point.query['rows']: adapter.configs[m.winner]['backend'] for m in outcome.measurements}
    assert outcome.exhausted_domain
    assert winners == {1: 'left', 4: 'right'}
    assert len(calls) == 4
    assert len(session_calls) == 1
    assert all(len(m.candidate_ids) == 2 for m in outcome.measurements)
