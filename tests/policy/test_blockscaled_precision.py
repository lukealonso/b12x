from dataclasses import asdict, replace

import pytest

import b12x
from b12x.gemm.blockscaled._policy import (
    BLOCKSCALED_POLICY, BLOCKSCALED_PRECISION, BlockscaledConfig, BlockscaledQuery,
)
from b12x.policy import (
    EMBEDDED_REGISTRY, ComponentProfile, FrozenMapping, GpuProfile,
    InvalidPreplannedPolicyError, PolicyContext, PolicySource, ProfileLeaf, ProfileRegistry,
    PolicyMode,
)
from b12x.policy.catalog import ONESHOT_COMPONENTS, list_planning_components
from b12x.policy.generation.providers.blockscaled import BlockscaledPrecisionGenerator, precision_cases, qualifies
from b12x.policy.generation.reducer import DecisionRecord


def _context(config, device=None):
    if device is None:
        device = EMBEDDED_REGISTRY.get("nvidia.rtx.pro.6000.blackwell").targets[0]
    registry = ProfileRegistry()
    registry.register(GpuProfile(
        profile_id="test.blockscaled", targets=(device,), metadata=FrozenMapping(),
        components=(ComponentProfile(
            component_id=BLOCKSCALED_PRECISION, query_schema_version=1, config_schema_version=1,
            planner=ProfileLeaf(name="test-exact-rows", config=FrozenMapping(config)),
        ),),
    ))
    registry.freeze()
    return PolicyContext.for_identity(device, registry=registry)


def test_sm121_accepts_measured_a16_routes():
    device = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm").targets[0]
    context = _context({"a16_rows": [[8, 128, 64, 4]]}, device)
    query = BlockscaledQuery(recipe="nvfp4", in_features=5376, out_features=4096)
    resolution = context.resolve(BLOCKSCALED_POLICY, query)
    assert resolution.source is PolicySource.PREPLANNED
    assert resolution.config.select(8) == (128, 64, 4)


@pytest.mark.parametrize("recipe,n,k,rows", [
    ("nvfp4", 4096, 5376, (*range(1, 13), 14, 15, 16)),
    ("nvfp4", 16384, 1024, (*range(1, 15), 16)),
    ("nvfp4", 17408, 5120, (1, 2)),
    ("nvfp4", 5120, 17408, ()),
    ("mxfp8", 4096, 5376, tuple(range(1, 17))),
    ("mxfp8", 16384, 1024, (*range(1, 15), 16, 24, 32)),
    ("mxfp8", 17408, 5120, (1, 2, 3, *range(7, 17))),
    ("mxfp8", 5120, 17408, tuple(range(1, 17))),
    ("nvfp4", 248320, 2560, (1,)),
    ("mxfp8", 248320, 2560, (1, 4)),
])
def test_embedded_gb10_precision_selects_qualified_rows(recipe, n, k, rows):
    device = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm").targets[0]
    context = PolicyContext.for_identity(device, mode=PolicyMode.PREPLANNED_ONLY)
    query = BlockscaledQuery(recipe=recipe, in_features=k, out_features=n)
    resolution = context.resolve(BLOCKSCALED_POLICY, query)
    assert resolution.source is PolicySource.PREPLANNED
    assert tuple(row[0] for row in resolution.config.a16_rows) == rows
    assert resolution.config.select(2048) is None


def test_gb10_timing_gate_requires_p0_stable_sm_clock_and_no_throttling():
    from benchmarks.benchmark_blockscaled_precision import _clock_checks

    state = dict(name="NVIDIA GB10", uuid="GPU-test", pstate="P0",
                 **{"clocks.sm": "2411", "clocks.mem": "[N/A]",
                    "clocks_event_reasons.active": "0x0"})

    def snapshot(values):
        return dict(fields=list(values), values=list(values.values()))

    before = snapshot(state)
    result = _clock_checks(before, before)
    assert result["valid"] and not result["memory_clock_reported"]
    for change in (dict(pstate="P8"), dict(uuid="GPU-other"),
                   {"clocks.sm": "2442"}, {"clocks_event_reasons.active": "0x4"}):
        assert not _clock_checks(before, snapshot({**state, **change}))["valid"]


def test_one_shot_profile_registration_does_not_create_a_planned_api():
    assert ONESHOT_COMPONENTS[0].op_qualname == "gemm.blockscaled"
    assert "gemm.blockscaled" not in {item.op_qualname for item in list_planning_components()}
    assert next(op for op in b12x.list_ops() if op.qualname == "gemm.blockscaled").api_style == "oneshot"


def test_exact_row_dispatch_and_policy_precedence():
    context = _context({"a16_rows": [[1, 64, 64, 4], [8, 128, 64, 2]]})
    query = BlockscaledQuery(recipe="nvfp4", in_features=5376, out_features=4096)
    resolution = context.resolve(BLOCKSCALED_POLICY, query)
    assert resolution.source is PolicySource.PREPLANNED
    assert resolution.config.select(1) == (64, 64, 4)
    assert resolution.config.select(8) == (128, 64, 2)
    assert resolution.config.select(4) is None
    assert resolution.config.select(16) is None
    override = BlockscaledConfig()
    assert context.with_override(BLOCKSCALED_PRECISION, override).resolve(BLOCKSCALED_POLICY, query).config == override
    unknown = replace(context.device, product_name="synthetic unprofiled GPU")
    assert PolicyContext.for_identity(unknown).resolve(BLOCKSCALED_POLICY, query).source is PolicySource.HEURISTIC
    assert set(BLOCKSCALED_POLICY.encode_query(query)) == {"recipe", "in_features", "out_features"}


@pytest.mark.parametrize("profile_id", ["nvidia.rtx.pro.6000.blackwell.max-q", "nvidia.gb10.48sm"])
@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
def test_small_m_heuristic_and_autotuned_quantized_decision(profile_id, recipe):
    measured = _context({"a16_rows": []}, EMBEDDED_REGISTRY.get(profile_id).targets[0])
    context = PolicyContext.for_identity(measured.device, mode=PolicyMode.HEURISTIC_ONLY)
    for k, n in ((5376, 4096), (1024, 17408), (2560, 3584), (2560, 640),
                 (1536, 2560), (2560, 320)):
        query = BlockscaledQuery(recipe=recipe, in_features=k, out_features=n)
        predicted = context.resolve(BLOCKSCALED_POLICY, query)
        assert predicted.source is PolicySource.HEURISTIC
        assert all(predicted.config.select(m) == (128, 64, 4) for m in range(1, 9))
        assert predicted.config.select(0) is None
        assert predicted.config.select(9) is None
        assert predicted.config.select(16) is None
        assert measured.resolve(BLOCKSCALED_POLICY, query).config.select(1) is None
    assert not context.resolve(BLOCKSCALED_POLICY, replace(query, out_features=7)).config.a16_rows
    unsupported = replace(context.device, compute_capability=(9, 0))
    assert not PolicyContext.for_identity(unsupported).resolve(BLOCKSCALED_POLICY, query).config.a16_rows


def test_embedded_precision_coverage_is_exact_to_max_q_and_geometry():
    profile = EMBEDDED_REGISTRY.get("nvidia.rtx.pro.6000.blackwell.max-q")
    context = PolicyContext.for_identity(profile.targets[0])
    query = BlockscaledQuery(recipe="nvfp4", in_features=5376, out_features=4096)
    resolution = context.resolve(BLOCKSCALED_POLICY, query)
    assert resolution.source is PolicySource.PREPLANNED
    assert resolution.config.select(8) is not None
    assert resolution.config.select(32) is None
    uncovered = context.resolve(BLOCKSCALED_POLICY, replace(query, in_features=4096))
    assert uncovered.source is PolicySource.HEURISTIC
    assert uncovered.config.select(8) == (128, 64, 4)
    for device in EMBEDDED_REGISTRY.get("nvidia.rtx.pro.6000.blackwell").targets:
        assert PolicyContext.for_identity(device).resolve(BLOCKSCALED_POLICY, query).source is PolicySource.HEURISTIC


@pytest.mark.parametrize("rows", [[[1, 64, 64, 3]], [[8, 64, 64, 1], [1, 64, 64, 1]], [[1, 64, 64, 1], [1, 128, 64, 1]]])
def test_invalid_matching_precision_profile_fails_closed(rows):
    context = _context({"a16_rows": rows})
    with pytest.raises(InvalidPreplannedPolicyError):
        context.resolve(BLOCKSCALED_POLICY, BlockscaledQuery(recipe="mxfp8", in_features=1024, out_features=16384))


def test_generator_aggregates_measured_rows_without_putting_m_in_runtime_query():
    cases = precision_cases(geometries=((4096, 5376),), counts=(1, 4, 8), recipes=("nvfp4",))
    generator = BlockscaledPrecisionGenerator(cases=cases)
    records = tuple(DecisionRecord(query=case.query, config=FrozenMapping({
        "a16_rows": [] if case.query["measured_m"] == 4 else [[case.query["measured_m"], 64, 64, 4]],
    })) for case in cases)
    planner = generator.build_planner(records)
    query = dict(recipe="nvfp4", in_features=5376, out_features=4096)
    config = BlockscaledConfig.from_profile(planner.lookup(query).config)
    assert config.a16_rows == ((1, 64, 64, 4), (8, 64, 64, 4))
    assert planner.lookup({**query, "in_features": 5504}) is None
    assert all(case.metadata["source_toolchain_sha256"] for case in cases)


def test_precision_promotion_accepts_parity_and_any_speedup():
    assert qualifies([{"a": 9, "q": 10}] * 30, "a", "q")[0]
    assert qualifies([{"a": 9.6, "q": 10}] * 30, "a", "q")[0]
    assert qualifies([{"a": 9.999, "q": 10}] * 30, "a", "q")[0]
    assert qualifies([{"a": 10, "q": 10}] * 30, "a", "q")[0]
    assert not qualifies([{"a": 10.001, "q": 10}] * 30, "a", "q")[0]
    assert not qualifies([{"a": 9 if i % 2 else 12, "q": 10} for i in range(30)], "a", "q")[0]


def test_precision_parity_keeps_confidence_interval_as_evidence():
    samples = [{"a": 9 if i % 2 else 11, "q": 10} for i in range(30)]
    accepted, ratio, interval = qualifies(samples, "a", "q")
    assert accepted and ratio == 1.0
    assert interval[0] < 1.0 < interval[1]


def test_precision_generator_prefers_a16_at_equal_latency(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from b12x.gemm.blockscaled._policy import A16_CONFIGS
    from b12x.policy.generation import CheckpointStore, GenerationContext, GenerationSettings, SweepCandidate, SweepMeasurement
    from b12x.policy.generation.progress import NullProgressReporter
    from b12x.policy.serialization import profile_from_dict

    cases = precision_cases(geometries=((4096, 5376),), counts=(8,), recipes=("nvfp4",))
    generator = BlockscaledPrecisionGenerator(cases=cases)
    quantized = SweepCandidate.create({"a16_rows": []})
    a16 = max((SweepCandidate.create({"a16_rows": [[8, *config]]}) for config in A16_CONFIGS),
              key=lambda candidate: candidate.candidate_id)
    assert quantized.candidate_id < a16.candidate_id
    session = SimpleNamespace(
        candidates=lambda case: (quantized, a16),
        measure=lambda case, candidates: tuple(SweepMeasurement(
            candidate=candidate, latency_us=10.0, correct=True,
        ) for candidate in candidates),
    )
    generator._benchmark_factory = lambda *args: nullcontext(session)
    device = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm").targets[0]
    context = GenerationContext(device=device, device_ordinal=0, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    result = generator.generate(context, progress=NullProgressReporter(),
                                checkpoints=CheckpointStore(tmp_path))
    profile = profile_from_dict(dict(
        profile_id="test.precision-parity", targets=[asdict(device)],
        components=[result.component],
    ))
    leaf = profile.component(BLOCKSCALED_PRECISION).lookup(
        dict(recipe="nvfp4", in_features=5376, out_features=4096))
    assert leaf.config == a16.config


def test_auto_precision_preserves_explicit_mode_and_unqualified_source_layout(monkeypatch):
    from types import SimpleNamespace
    import torch
    from b12x.gemm.blockscaled._a16 import _select_mode

    config = BlockscaledConfig(a16_rows=((8, 64, 64, 1),))
    monkeypatch.setattr("b12x.gemm.blockscaled._policy.resolve_precision",
                        lambda *args: SimpleNamespace(config=config))
    weight = SimpleNamespace(in_features=128, padded_in_features=128, out_features=128)
    source = torch.empty(8, 128, dtype=torch.bfloat16)
    assert _select_mode("auto", source, weight) == ("a16", (64, 64, 1))
    assert _select_mode("a16", source, weight) == ("a16", (64, 64, 1))
    assert _select_mode("a16", source[:1], weight) == ("a16", None)
    assert _select_mode("quantized", source, weight) == ("quantized", None)
    assert _select_mode("auto", source.T.contiguous().T, weight) == ("quantized", None)
    misaligned = torch.empty(8 * 128 + 1, dtype=torch.bfloat16)[1:].view(8, 128)
    assert _select_mode("auto", misaligned, weight) == ("quantized", None)


def test_resume_retries_invalid_clock_samples_and_reuses_qualified_samples(tmp_path):
    from b12x.policy.generation import CheckpointStore, GenerationContext, GenerationSettings, SweepCandidate, SweepMeasurement

    case, = precision_cases(geometries=((4096, 5376),), counts=(1,), recipes=("nvfp4",))
    generator = BlockscaledPrecisionGenerator(cases=(case,))
    context = GenerationContext(device=_context({"a16_rows": []}).device,
                                device_ordinal=0, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    store = CheckpointStore(tmp_path)
    candidate = SweepCandidate.create({"a16_rows": []})
    for error in ("clock or replay-allocation qualification failed", None):
        measurement = SweepMeasurement(candidate=candidate, correct=True, latency_us=10, error=error)
        store.save(BLOCKSCALED_PRECISION, case.case_id, generator._checkpoint_payload(
            case=case, generation=context.checkpoint_metadata(),
            candidate_ids=(candidate.candidate_id,), measurements=(measurement,),
        ))
        cached = generator._load_checkpoint(case=case, context=context, checkpoints=store)
        assert (cached is None) == (error is not None)
