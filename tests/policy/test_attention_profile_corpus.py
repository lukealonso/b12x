from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace

from benchmarks.benchmark_gdn_decode import QWEN38_GDN_CASES
from benchmarks.benchmark_paged_attention import BENCHMARK_PROFILES
from benchmarks.benchmark_qsa import PROFILES as QSA_PROFILES
from b12x.policy import (
    EMBEDDED_REGISTRY,
    DeviceIdentity,
    list_profiled_components,
    profile_from_dict,
)
from b12x.policy.generation import (
    CheckpointStore,
    GenerationContext,
    GenerationSettings,
    SweepCandidate,
    SweepMeasurement,
)
from b12x.policy.generation.attention_corpus import (
    ATTENTION_BENCHMARK_PRESETS,
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_SEQUENCE_CAPACITIES,
    GDN_GEOMETRIES,
    GQA_GEOMETRIES,
    MLA_GEOMETRIES,
    QSA_DIRECT_PREFILL_ROWS,
    QSA_GEOMETRIES,
    QSA_PAGE_SIZES,
    SPARSE_MLA_GEOMETRIES,
    attention_corpus_manifest,
    gdn_cases,
    gqa_cases,
    mla_cases,
    qsa_cases,
    sparse_mla_cases,
)
from b12x.policy.generation.providers import register_builtin_generators
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers.attention import (
    GdnAttentionGenerator,
    QsaAttentionGenerator,
    _QsaSession,
)
from b12x.policy.generation.providers.gpu_workers import GdnBenchmarkFactory
from b12x.policy.generation.providers.qualification import (
    _DsaIndexerProbe,
    DsaIndexerGenerator,
    SparseMlaGenerator,
)
from b12x.policy.generation.providers.norm_sequence import (
    MhcGenerator,
    _MhcSession,
    _hyperconnection_cases,
    _mhc_cases,
    _mtp_feedback_cases,
)
from b12x.policy.generation.registry import ComponentGeneratorRegistry
from b12x.sequence.gdn_decode._policy import GDN_POLICY, GdnQuery


class _FixedGdnSession(AbstractContextManager["_FixedGdnSession"]):
    def __enter__(self) -> "_FixedGdnSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def candidates(self, _case):
        return (SweepCandidate.create({"backend": "triton"}),)

    def measure(self, _case, candidates):
        return (
            SweepMeasurement(
                candidate=candidates[0],
                latency_us=1.0,
                correct=True,
            ),
        )


class _FixedGdnFactory:
    def __call__(self, _group_id, _cases, _context):
        return _FixedGdnSession()


class _FixedQsaSession(AbstractContextManager["_FixedQsaSession"]):
    def __enter__(self) -> "_FixedQsaSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def candidates(self, _case):
        return _QsaSession._CANDIDATES

    def measure(self, _case, candidates):
        return tuple(
            SweepMeasurement(
                candidate=candidate,
                latency_us=100.0,
                correct=True,
            )
            for candidate in candidates
        )


class _FixedQsaFactory:
    def __call__(self, _group_id, _cases, _context):
        return _FixedQsaSession()


def test_builtin_registry_covers_every_top_level_component() -> None:
    registry = ComponentGeneratorRegistry()

    register_builtin_generators(registry)

    assert registry.component_ids() == tuple(
        str(item.component_id) for item in list_profiled_components()
    )


def test_qsa_generator_races_n32_n16_n64_and_prefers_n32_on_ties(
    tmp_path,
) -> None:
    case = qsa_cases()[0]
    generator = QsaAttentionGenerator(
        benchmark_factory=_FixedQsaFactory(),
        cases=(case,),
    )
    context = GenerationContext(
        device=DeviceIdentity(
            vendor="nvidia",
            product_name="Synthetic SM120",
            compute_capability=(12, 0),
            sm_count=188,
        ),
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="test",
        settings=GenerationSettings(),
    )
    checkpoints = CheckpointStore(tmp_path / "checkpoints")

    result = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    checkpoint = checkpoints.load("attention.qsa", case.case_id)
    assert checkpoint is not None
    assert checkpoint["candidate_contract_version"] == 2
    assert [
        item["config"]["sparse_gqa_direct_kv_warps"]
        for item in checkpoint["measurements"]
    ] == [2, 1, 4]
    profile = profile_from_dict(
        {
            "profile_id": "nvidia.synthetic.188sm",
            "targets": [
                {
                    "vendor": "nvidia",
                    "product_name": "Synthetic SM120",
                    "compute_capability": [12, 0],
                    "sm_count": 188,
                }
            ],
            "components": [result.component],
        }
    )
    component = profile.component("attention.qsa")
    assert component is not None
    hit = component.lookup(case.query)
    assert hit is not None
    assert hit.config["sparse_gqa_direct_kv_warps"] == 2


def test_attention_corpora_have_stable_reviewed_cross_products() -> None:
    assert len(GDN_GEOMETRIES) == 21
    assert len(GQA_GEOMETRIES) == 18
    assert len(MLA_GEOMETRIES) == 1
    assert len(QSA_GEOMETRIES) == 3
    assert len(SPARSE_MLA_GEOMETRIES) == 12
    assert len(gdn_cases()) == 1_462
    assert len(gqa_cases()) == 14_400
    assert len(mla_cases()) == 200
    assert len(qsa_cases()) == 384
    assert len(sparse_mla_cases()) == 288
    assert len({case.query for case in gqa_cases()}) == len(gqa_cases())

    all_cases = (
        *gdn_cases(),
        *gqa_cases(),
        *mla_cases(),
        *qsa_cases(),
        *sparse_mla_cases(),
    )
    assert len({case.case_id for case in all_cases}) == len(all_cases)


def test_gdn_corpus_includes_qwen_and_glm_decay_contracts() -> None:
    cases = gdn_cases()
    recipes = {case.metadata["decay_recipe"] for case in cases}
    glm_cases = [case for case in cases if case.metadata["decay_recipe"] == "kda"]

    assert recipes == {"gdn", "kda"}
    assert len(glm_cases) == 810
    assert {case.query["key_heads"] for case in glm_cases} == {4, 8, 16, 32, 64}
    assert all(
        case.query["key_heads"] == case.query["value_heads"] for case in glm_cases
    )
    glm_tp4_capacities = {
        (
            case.query["max_seqs"],
            case.query["max_tokens"],
            case.query["state_index_columns"],
        )
        for case in glm_cases
        if case.query["key_heads"] == 16
    }
    assert (16, 16, 1) in glm_tp4_capacities
    assert (16, 96, 6) in glm_tp4_capacities
    exercised = {
        case.query
        for case in cases
        if max(case.metadata["query_lengths"]) == int(case.query["state_index_columns"])
    }
    assert exercised == {case.query for case in cases}


def test_embedded_gdn_profiles_cover_every_corpus_query() -> None:
    cases_by_query = {case.query: case for case in gdn_cases()}

    for profile in EMBEDDED_REGISTRY.list_profiles():
        component = profile.component("attention.gdn")
        assert component is not None, profile.profile_id
        for query, case in cases_by_query.items():
            hit = component.lookup(query)
            assert hit is not None, (profile.profile_id, query.to_dict())
            expected_backend = (
                "triton" if case.metadata["decay_recipe"] == "kda" else "cutedsl"
            )
            assert hit.config["backend"] == expected_backend, (
                profile.profile_id,
                query.to_dict(),
                hit.config,
            )


def test_embedded_norm_sequence_profiles_cover_every_corpus_query() -> None:
    component_cases = {
        "norm.hyperconnection": _hyperconnection_cases(),
        "sequence.mtp_feedback": _mtp_feedback_cases(),
    }

    for profile in EMBEDDED_REGISTRY.list_profiles():
        for component_id, cases in component_cases.items():
            component = profile.component(component_id)
            assert component is not None, (profile.profile_id, component_id)
            for case in cases:
                hit = component.lookup(case.query)
                assert hit is not None, (
                    profile.profile_id,
                    component_id,
                    case.query.to_dict(),
                )
                assert hit.config["backend"] == "cutedsl"


def test_attention_capacity_axes_cover_serving_and_prefill_buckets() -> None:
    expected_sequence_capacities = (
        *range(1, 17),
        32,
        64,
        128,
        256,
    )
    assert expected_sequence_capacities == COMMON_SEQUENCE_CAPACITIES
    assert COMMON_PREFILL_TOKEN_CAPACITIES == (1_024, 2_048, 4_096, 8_192)

    for cases in (_hyperconnection_cases(), _mtp_feedback_cases()):
        capacities = {int(case.query["max_tokens"]) for case in cases}
        assert set(COMMON_SEQUENCE_CAPACITIES) <= capacities
        assert {512, *COMMON_PREFILL_TOKEN_CAPACITIES} <= capacities

    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["max_q_rows"])
        for case in qsa_cases()
        if int(case.query["max_q_rows"]) >= 1_024
    }
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["query_rows"])
        for case in mla_cases()
        if case.query["mode"] == "extend"
    }
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["query_rows"]) for case in sparse_mla_cases()
    }

    assert QSA_DIRECT_PREFILL_ROWS == {
        2_048: (65, 128, 1_024, 2_048),
        32_768: (65, 128, 1_024, 4_096, 6_016, 8_192),
        262_144: (65, 128, 1_024, 4_096, 6_016, 8_192),
    }
    assert QSA_PAGE_SIZES == (16, 64, 1_504, 3_008)
    assert {
        int(candidate.config["sparse_gqa_direct_kv_warps"])
        for candidate in _QsaSession._CANDIDATES
    } == {1, 2, 4}
    assert [
        int(candidate.config["sparse_gqa_direct_kv_warps"])
        for candidate in _QsaSession._CANDIDATES
    ] == [2, 1, 4]
    assert {
        int(case[0].removeprefix("glm52-extend-m"))
        for case in _DsaIndexerProbe._CASES
        if case[0].startswith("glm52-extend-m")
    } == set(COMMON_PREFILL_TOKEN_CAPACITIES)


def test_gdn_backend_identifies_decay_contract_from_head_geometry() -> None:
    common = {
        "gate_activation": "sigmoid",
        "qk_l2norm": True,
        "state_dtype": "float32",
        "max_seqs": 1,
        "max_tokens": 4,
        "state_index_columns": 4,
    }

    qwen = GdnQuery(key_heads=8, value_heads=24, **common)
    glm = GdnQuery(key_heads=8, value_heads=8, **common)

    assert GDN_POLICY.heuristic(qwen, None).backend == "cutedsl"
    assert GDN_POLICY.heuristic(glm, None).backend == "triton"


def test_generated_gdn_profile_covers_dense_and_sparse_capacity_ranges(
    tmp_path,
) -> None:
    cases = tuple(
        case
        for case in gdn_cases()
        if case.metadata["decay_recipe"] == "kda" and case.query["key_heads"] == 16
    )
    generator = GdnAttentionGenerator(
        benchmark_factory=_FixedGdnFactory(),
        cases=cases,
    )
    device = DeviceIdentity(
        vendor="nvidia",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="Synthetic RTX",
    )
    context = GenerationContext(
        device=device,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="test",
        settings=GenerationSettings(),
    )
    result = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )
    profile = profile_from_dict(
        {
            "profile_id": "synthetic",
            "targets": [
                {
                    "vendor": device.vendor,
                    "compute_capability": list(device.compute_capability),
                    "sm_count": device.sm_count,
                    "product_name": device.product_name,
                }
            ],
            "components": [result.component],
        }
    )
    component = profile.component("attention.gdn")
    assert component is not None

    for max_seqs, columns in ((16, 1), (24, 1), (24, 6), (256, 8)):
        leaf = component.lookup(
            {
                "gate_activation": "sigmoid",
                "qk_l2norm": True,
                "state_dtype": "float32",
                "key_heads": 16,
                "value_heads": 16,
                "max_seqs": max_seqs,
                "max_tokens": max_seqs * columns,
                "state_index_columns": columns,
            }
        )
        assert leaf is not None
        assert leaf.config["backend"] == "triton"

    assert (
        component.lookup(
            {
                "gate_activation": "sigmoid",
                "qk_l2norm": True,
                "state_dtype": "float32",
                "key_heads": 16,
                "value_heads": 16,
                "max_seqs": 257,
                "max_tokens": 257,
                "state_index_columns": 1,
            }
        )
        is None
    )


def test_gdn_benchmark_factory_accepts_grouped_capacity_cases() -> None:
    group_id = gdn_cases()[0].group_id
    cases = tuple(case for case in gdn_cases() if case.group_id == group_id)

    session = GdnBenchmarkFactory()(group_id, cases, object())

    assert len(cases) > 1
    assert session.candidates(cases[0])[0].config["backend"] == "cutedsl"


def test_attention_corpus_manifests_are_content_addressed() -> None:
    for component in ("gdn", "gqa", "mla", "qsa", "sparse_mla"):
        manifest = attention_corpus_manifest(component)

        assert manifest["schema_version"] == 1
        assert len(manifest["corpus_sha256"]) == 64


def test_mhc_tuner_races_the_medium_prefill_plan() -> None:
    case = next(
        case
        for case in _mhc_cases()
        if case.query["hidden_size"] == 4_096 and case.query["max_tokens"] == 3_072
    )
    configs = tuple(
        candidate.config.to_dict()
        for candidate in _MhcSession(SimpleNamespace(device=None)).candidates(case)
    )

    assert any(config["backend"] == "native" for config in configs)
    assert any(
        config
        == {
            "backend": "tf32_tma",
            "projection_tile_m": 64,
            "projection_tile_n": 24,
            "projection_tile_k": 64,
            "projection_num_stages": 2,
            "projection_num_m_warps": 4,
            "projection_num_n_warps": 1,
            "projection_k_splits": 8,
        }
        for config in configs
    )


def test_glm_profile_generation_envelope_matches_presets() -> None:
    dsa_queries = DsaIndexerGenerator().reviewed_queries()
    sparse_queries = SparseMlaGenerator().reviewed_queries()
    mhc_queries = MhcGenerator().reviewed_queries()

    assert any(
        query.num_q_heads == 32 and query.top_k == 2_048 for query in dsa_queries
    )
    assert any(query.num_q_heads == 32 and query.top_k == 512 for query in dsa_queries)
    assert {
        (query.qk_head_dim, query.v_head_dim, query.model_type)
        for query in sparse_queries
    } == {(576, 512, None), (512, 512, 2)}
    assert {query.num_q_heads for query in sparse_queries} == {8, 16, 32, 64}
    assert any(
        query.max_tokens == 6 and query.hidden_size == 4_096 and query.split_k == 64
        for query in mhc_queries
    )
    assert {4_096, 7_168} == {query.hidden_size for query in mhc_queries}
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        query.max_tokens for query in mhc_queries
    }
    assert {2_304, 3_072, 3_584} <= {query.max_tokens for query in mhc_queries}
    assert {query.score_mode for query in dsa_queries} == {"dsa", "msa"}
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        query.max_q_rows for query in dsa_queries if query.mode == "prefill"
    }
    assert set(COMMON_SEQUENCE_CAPACITIES) <= {
        query.max_q_rows for query in sparse_queries if query.mode == "decode"
    }


def test_named_attention_benchmark_presets_are_in_the_reviewed_inventory() -> None:
    preset_ids = {preset.preset_id for preset in ATTENTION_BENCHMARK_PRESETS}
    assert {
        name.removeprefix("paged:") for name in preset_ids if name.startswith("paged:")
    } == set(BENCHMARK_PROFILES)
    assert {
        name.removeprefix("qsa:") for name in preset_ids if name.startswith("qsa:")
    } == set(QSA_PROFILES)
    assert {
        name.removeprefix("gdn:") for name in preset_ids if name.startswith("gdn:")
    } == {case.name for case in QWEN38_GDN_CASES}
    assert preset_ids == {
        "compressed-mla:deepseek-v4-flash-default",
        "compressed-mla:vllm-dsv4-trace",
        "dense-mla:kimi-k3",
        "dsa-indexer:glm-5.1-default",
        "gdn:qk16-v48-decode-bs1",
        "gdn:qk2-v6-decode-bs1",
        "gdn:qk4-v12-decode-bs1",
        "gdn:qk8-v24-decode-bs1",
        "gdn:qk8-v24-decode-bs4",
        "gdn:qk8-v24-spec2-bs4",
        "gdn:qk8-v24-spec4-bs1",
        "gdn:qk8-v24-spec4-bs4",
        "gdn:qk8-v24-spec4-uneven",
        "mla:target-dsv4-trace",
        "mla:target-glm52-prefill4k-ctx16k",
        "mla:target-prefill64k-bs1",
        "mla:glm-5.2-default",
        "msa-indexer:minimax-m3-default",
        "paged-msa:minimax-m3-default",
        "paged:minimax-m2.7",
        "paged:qwen-gqa",
        "paged:qwen3.8-27b",
        "paged-indexer:deepseek-v4-flash-default",
        "qsa:tp1",
        "qsa:tp2",
        "qsa:tp4",
        "unified-mla:deepseek-v4-flash-decode",
        "unified-mla:deepseek-v4-flash-prefill",
        "unified-mla:glm-5.1-decode",
        "vllm-paged:minimax-m2.7",
        "vllm-paged:qwen-gqa",
    }
