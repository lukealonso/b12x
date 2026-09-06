from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest
import torch

from b12x.moe.fused_moe import _impl as fused_moe_impl
from b12x.policy import EMBEDDED_REGISTRY, DeviceIdentity
from b12x.policy.generation import (
    CheckpointStore,
    GenerationContext,
    GenerationSettings,
)
from b12x.policy.generation.moe_corpus import (
    MoeModelGeometry,
    MoeRecipe,
    expand_physical_geometries,
    expand_sweep_cases,
)
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers import moe_gpu_worker
from b12x.policy.generation.providers.moe import (
    _config_covers_query,
    _correctness_anchor_rank,
    _synthesize_token_capacity_coverage,
    MoeCandidate,
    MoeDecodeGenerator,
    MoeMeasurement,
)
from b12x.policy.generation.reducer import (
    DecisionRecord,
    build_axis_tree,
    decision_node_to_dict,
    synthesize_integer_axis_coverage,
)
from b12x.policy.generation.providers.moe_gpu_worker import (
    MoeGpuBenchmarkFactory,
    _activation_mode,
    _benchmark_input_scale,
    _candidate_environment,
    _candidate_reference_output,
    _candidates_for_geometry,
    _condition_benchmark_inputs,
    _concrete_candidate_path,
    _cosine_similarity,
    _finite_float_or_none,
    _is_fatal_accelerator_error,
    _measurement_seed,
    _maximum_relative_norm_error,
    _MoeGeometrySession,
    _MoeProcessSession,
    _MoeRemoteWorkerError,
    _packed_weights,
    _reset_cuda_graphs,
    _relative_norm_error,
    _trellis_weights,
    _uniform_w4a16_reference,
    _w4a16_direct_path,
    _w4a16_weight_layout,
)
from b12x.policy.serialization import profile_from_dict

_DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="Synthetic GPU",
)


def test_embedded_moe_profiles_and_heuristics_cover_corpus_queries(
) -> None:
    from b12x.policy import PolicyContext, PolicySource
    from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery

    cases = tuple(
        case for case in expand_sweep_cases()
        if case.geometry.recipe.quant_mode != "nvfp4_auto"
    )
    queries = {tuple(sorted(case.query().items())): case.query() for case in cases}
    base_queries = {}
    for case in cases:
        query = case.query()
        query.pop("num_tokens")
        query.pop("routed_rows")
        base_queries.setdefault(tuple(sorted(query.items())), query)
    for base_query in base_queries.values():
        top_k = int(base_query["top_k"])
        direct_limit = fused_moe_impl._DIRECT_ROUTING_MAX_ROUTED_ROWS // top_k
        triton_limit = (
            fused_moe_impl._DYNAMIC_EXTERNAL_ROUTE_PLAN_MAX_ROWS // top_k
        )
        for num_tokens in (9, direct_limit, direct_limit + 1, triton_limit + 1):
            if not 1 <= num_tokens <= 8_192:
                continue
            query = {
                **base_query,
                "num_tokens": num_tokens,
                "routed_rows": num_tokens * top_k,
            }
            queries.setdefault(tuple(sorted(query.items())), query)

    for profile in EMBEDDED_REGISTRY.list_profiles():
        component = profile.component("moe.decode")
        assert component is not None, profile.profile_id
        context = PolicyContext.for_identity(profile.targets[0])
        for query in queries.values():
            hit = component.lookup(query)
            if query["quant_mode"] == "w4a16" and query["source_format"] == "modelopt_nvfp4":
                # Native-layout measurements cannot qualify uniform MMA-packed A16.
                assert hit is None, (profile.profile_id, query)
                resolution = context.resolve(
                    MOE_DECODE_POLICY, MoeDecodeQuery(**query),
                )
                assert resolution.source is PolicySource.HEURISTIC
                assert _config_covers_query(query, asdict(resolution.config))
                continue
            assert hit is not None, (profile.profile_id, query)
            assert _config_covers_query(query, hit.config), (
                profile.profile_id,
                query,
                hit.config,
            )


def test_modelopt_w4a8_profile_worker_uses_a8_activation() -> None:
    from b12x.moe import fused_moe

    assert _activation_mode("w4a8_nvfp4") is fused_moe.ActivationMode.A8


def test_candidate_environment_uses_default_w4a8_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "B12X_DYNAMIC_W4A8_MATERIALIZED",
        "B12X_DYNAMIC_W4A8_SHARE_INPUT",
        "B12X_W4A8_TINY_DECODE",
    )
    for name in names:
        monkeypatch.setenv(name, "0")
    candidate = MoeCandidate.create(
        {
            "backend": "dynamic",
            "dynamic_route_mode": "grouped",
            "dynamic_tile_m": 16,
            "route_planner": "internal",
            "max_active_clusters": None,
            "w4a16_route_mode": None,
        }
    )

    with _candidate_environment(candidate):
        assert all(name not in os.environ for name in names)

    assert all(os.environ[name] == "0" for name in names)


def test_correctness_gate_detects_scale_errors_that_cosine_cannot() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    scaled = reference * 1.0e-8

    assert _cosine_similarity(scaled, reference) > 0.999
    assert _relative_norm_error(scaled, reference) > 0.99


@pytest.mark.parametrize("activation", ("silu", "relu2"))
def test_uniform_w4a16_tuner_reference_is_independent_of_routes(
    activation: str,
) -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.recipe_id == "modelopt-w4a16"
        and geometry.activation == activation
    )
    x = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    topk_weights = torch.tensor([[0.25, 0.75], [0.9, 0.1]])

    output = _uniform_w4a16_reference(
        geometry,
        x=x,
        topk_weights=topk_weights,
    )

    fc1 = x.sum(dim=-1) * 0.5
    if activation == "silu":
        intermediate = torch.nn.functional.silu(fc1) * fc1
    else:
        intermediate = torch.square(torch.relu(fc1))
    expected = intermediate * (0.5 * geometry.intermediate_size)
    assert torch.allclose(output[:, 0], expected)
    assert torch.equal(output, output[:, :1].expand_as(output))


@pytest.mark.parametrize(
    ("quant_mode", "backend", "w4a16_route_mode"),
    (
        ("w4a16", "dynamic", None),
        ("w4a8_mx", "w4a16", "packed"),
        ("w4a8_nvfp4", "w4a16", "packed"),
    ),
)
def test_profile_coverage_rejects_cross_family_backends(
    quant_mode: str,
    backend: str,
    w4a16_route_mode: str | None,
) -> None:
    query = {
        "quant_mode": quant_mode,
        "source_format": "modelopt_nvfp4",
        "activation": "silu",
        "num_experts": 256,
        "hidden_size": 4096,
        "intermediate_size": 1024,
        "top_k": 8,
        "num_tokens": 4,
    }
    config = {
        "backend": backend,
        "dynamic_route_mode": "grouped" if backend == "dynamic" else None,
        "dynamic_tile_m": 16 if backend == "dynamic" else None,
        "route_planner": "internal",
        "max_active_clusters": None,
        "w4a16_route_mode": w4a16_route_mode,
    }

    assert not _config_covers_query(query, config)


def test_relu2_w4a16_profile_fixture_uses_full_scale_conditioned_inputs() -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.recipe_id == "modelopt-w4a16"
        and geometry.activation == "relu2"
    )

    assert _benchmark_input_scale(geometry) == 1.0


def test_uniform_w4a16_profile_fixture_has_stable_positive_projections() -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.recipe_id == "modelopt-w4a16"
        and geometry.activation == "silu"
    )
    generator = torch.Generator(device="cpu").manual_seed(17)
    inputs = torch.randn(
        (8, geometry.hidden_size),
        dtype=torch.float32,
        generator=generator,
    )

    conditioned = _condition_benchmark_inputs(geometry, inputs)

    expected_sum = geometry.hidden_size**0.5
    assert torch.allclose(
        conditioned.sum(dim=-1),
        torch.full((8,), expected_sum),
        rtol=0.0,
        atol=1.0e-3,
    )


def test_w4a16_tuner_enumerates_distinct_kernel_routes() -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == "w4a16"
    )

    configs = tuple(
        candidate.config.to_dict()
        for candidate in _candidates_for_geometry(geometry, sm_count=48)
    )

    assert {config["backend"] for config in configs} == {"w4a16"}
    assert {config["w4a16_route_mode"] for config in configs} == {
        "direct",
        "packed",
    }
    assert all(config["dynamic_tile_m"] is None for config in configs)


def test_projection_mixed_trellis_enumerates_only_its_packed_route() -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.trellis_variant == "glm-mcg-projection-tiered"
    )
    configs = tuple(
        candidate.config.to_dict()
        for candidate in _candidates_for_geometry(geometry, sm_count=188)
    )

    assert [config["w4a16_route_mode"] for config in configs] == ["packed"]
    assert all(config["backend"] == "w4a16" for config in configs)


def test_w4a16_tuner_models_native_and_packed_route_kernels() -> None:
    cases = expand_sweep_cases()
    glm_decode = next(
        case
        for case in cases
        if case.geometry.recipe.recipe_id == "modelopt-w4a16"
        and case.geometry.activation == "silu"
        and case.top_k == 8
        and case.num_tokens == 8
        and case.route_pattern == "balanced"
    )
    e8m0_decode = next(
        case
        for case in cases
        if case.geometry.recipe.recipe_id == "e8m0-w4a16"
        and case.geometry.activation == "silu"
        and case.geometry.intermediate_size % 128 == 0
        and case.is_model_native_top_k
        and case.num_tokens == 8
        and case.route_pattern == "balanced"
    )
    compressed_direct = next(
        case
        for case in cases
        if case.geometry.recipe.recipe_id == "compressed-tensors-w4a16"
        and case.geometry.activation == "relu2"
        and case.is_model_native_top_k
        and case.num_tokens == 6
        and case.route_pattern == "balanced"
    )
    relu2_direct = next(
        case
        for case in cases
        if case.geometry.recipe.recipe_id == "modelopt-w4a16"
        and case.geometry.activation == "relu2"
        and case.num_tokens == 6
        and case.route_pattern == "balanced"
    )
    relu2_packed_only = next(
        case
        for case in cases
        if case.geometry.key == relu2_direct.geometry.key
        and case.top_k == relu2_direct.top_k
        and case.num_tokens == 16
        and case.route_pattern == "balanced"
    )

    assert _w4a16_weight_layout(glm_decode.geometry) == "packed"
    assert (
        _w4a16_direct_path(glm_decode.geometry, glm_decode)
        == "w4a16.tc_decode"
    )
    assert _w4a16_weight_layout(e8m0_decode.geometry) == "packed"
    assert (
        _w4a16_direct_path(e8m0_decode.geometry, e8m0_decode)
        == "w4a16.tc_decode"
    )
    assert _w4a16_weight_layout(compressed_direct.geometry) == "packed"
    assert (
        _w4a16_direct_path(compressed_direct.geometry, compressed_direct)
        == "w4a16.direct_topk"
    )
    assert (
        _w4a16_direct_path(relu2_direct.geometry, relu2_direct)
        == "w4a16.direct_topk"
    )
    assert _w4a16_direct_path(relu2_packed_only.geometry, relu2_packed_only) is None


def test_w4a16_tuner_layout_matches_canonical_weight_planning() -> None:
    from b12x.moe import fused_moe

    expected_layout = {
        fused_moe.WeightPacking.SOURCE_NATIVE: "modelopt",
        fused_moe.WeightPacking.MMA_PACKED: "packed",
    }
    geometries = (
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == "w4a16"
        and geometry.recipe.trellis_variant is None
    )
    for geometry in geometries:
        source = fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat(geometry.recipe.source_format),
            w13_layout=(
                fused_moe.W13Layout.W31
                if geometry.recipe.source_format == "modelopt_nvfp4"
                else fused_moe.W13Layout.W13
            ),
        )
        plan = fused_moe.plan_weights(
            source=source,
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A16,
                nonlinearity=geometry.activation,
                io_dtype=torch.bfloat16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=geometry.num_experts,
                hidden_size=geometry.hidden_size,
                intermediate_size=geometry.intermediate_size,
            ),
        )

        assert _w4a16_weight_layout(geometry) == expected_layout[
            plan.prepared_format.packing
        ]


def test_concrete_candidate_path_reads_the_exact_execution_variant() -> None:
    case = next(
        case
        for case in expand_sweep_cases()
        if case.geometry.recipe.recipe_id == "modelopt-nvfp4-auto"
        and case.geometry.activation == "silu"
        and case.top_k == 8
        and case.num_tokens == 8
        and case.route_pattern == "balanced"
    )
    candidate = MoeCandidate.create(
        {
            "backend": "w4a16",
            "dynamic_route_mode": None,
            "dynamic_tile_m": None,
            "route_planner": "internal",
            "max_active_clusters": None,
            "w4a16_route_mode": "direct",
        }
    )
    resolved_config = fused_moe_impl.MoeDecodeConfig.from_profile(candidate.config)
    actual_plan = SimpleNamespace(
        deterministic_output=False,
        policy_resolution=SimpleNamespace(config=resolved_config),
    )
    variant = SimpleNamespace(
        _impl=actual_plan,
        execution=SimpleNamespace(tile_m=None),
        implementation="w4a16",
    )
    capacity_plan = SimpleNamespace(variant_for=lambda _tokens: variant)

    assert (
        _concrete_candidate_path(
            geometry=case.geometry,
            case=case,
            candidate=candidate,
            plan=capacity_plan,
            binding=SimpleNamespace(implementation="w4a16"),
            prepared_payload=SimpleNamespace(weight_layout="modelopt"),
        )
        == "w4a16.small_m_direct"
    )


def test_concrete_candidate_path_reports_projection_mixed_trellis() -> None:
    case = next(
        case
        for case in expand_sweep_cases()
        if case.geometry.recipe.recipe_id == "trellis-glm-w4a16"
        and case.is_model_native_top_k
        and case.num_tokens == 8
        and case.route_pattern == "balanced"
    )
    candidate = MoeCandidate.create(
        {
            "backend": "w4a16",
            "dynamic_route_mode": None,
            "dynamic_tile_m": None,
            "route_planner": "internal",
            "max_active_clusters": None,
            "w4a16_route_mode": "packed",
        }
    )
    resolved_config = fused_moe_impl.MoeDecodeConfig.from_profile(candidate.config)
    variant = SimpleNamespace(
        _impl=SimpleNamespace(
            deterministic_output=False,
            policy_resolution=SimpleNamespace(config=resolved_config),
        ),
        execution=SimpleNamespace(tile_m=None),
        implementation="w4a16",
    )

    assert (
        _concrete_candidate_path(
            geometry=case.geometry,
            case=case,
            candidate=candidate,
            plan=SimpleNamespace(variant_for=lambda _tokens: variant),
            binding=SimpleNamespace(implementation="trellis_mixed3"),
            prepared_payload=SimpleNamespace(weight_layout="trellis_mixed3"),
        )
        == "w4a16.trellis_mixed3"
    )


@pytest.mark.parametrize(
    ("recipe_id", "backend", "tile_m", "route_mode", "expected"),
    (
        ("e8m0-w4a8", "micro", None, None, "w4a8.tiny_decode"),
        (
            "modelopt-w4a8-nvfp4",
            "micro",
            None,
            None,
            "w4a8.direct_micro",
        ),
        ("e8m0-w4a8", "dynamic", 16, "direct", "w4a8.dynamic.direct.m16"),
        ("e8m0-w4a8", "dynamic", 16, "grouped", "w4a8.dynamic.decode.m16"),
        ("e8m0-w4a8", "dynamic", 32, "grouped", "w4a8.dynamic.dense.m32"),
        (
            "modelopt-w4a8-nvfp4",
            "dynamic",
            16,
            "grouped",
            "w4a8.dynamic.grouped.m16",
        ),
    ),
)
def test_concrete_candidate_path_reports_w4a8_specializations(
    monkeypatch: pytest.MonkeyPatch,
    recipe_id: str,
    backend: str,
    tile_m: int | None,
    route_mode: str | None,
    expected: str,
) -> None:
    monkeypatch.delenv("B12X_DYNAMIC_WORK_SOURCE", raising=False)
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.recipe_id == recipe_id
        and geometry.activation == "silu"
    )
    case = next(
        case
        for case in expand_sweep_cases(geometries=(geometry,))
        if case.is_model_native_top_k
        and case.num_tokens == 1
        and case.route_pattern == "balanced"
    )
    candidate = MoeCandidate.create(
        {
            "backend": backend,
            "dynamic_route_mode": route_mode,
            "dynamic_tile_m": tile_m,
            "route_planner": "internal",
            "max_active_clusters": None,
            "w4a16_route_mode": None,
        }
    )
    resolved_config = fused_moe_impl.MoeDecodeConfig.from_profile(candidate.config)
    variant = SimpleNamespace(
        _impl=SimpleNamespace(
            deterministic_output=False,
            policy_resolution=SimpleNamespace(config=resolved_config),
        ),
        execution=SimpleNamespace(tile_m=tile_m),
        implementation=backend,
    )

    assert (
        _concrete_candidate_path(
            geometry=geometry,
            case=case,
            candidate=candidate,
            plan=SimpleNamespace(variant_for=lambda _tokens: variant),
            binding=SimpleNamespace(implementation=backend),
            prepared_payload=SimpleNamespace(weight_layout="prepared"),
        )
        == expected
    )


@pytest.mark.parametrize(
    ("quant_mode", "route_modes"),
    (("w4a8_mx", {"direct", "grouped"}), ("w4a8_nvfp4", {"grouped"})),
)
def test_w4a8_tuner_enumerates_micro_and_dynamic_tiles(
    quant_mode: str,
    route_modes: set[str],
) -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == quant_mode
    )

    configs = tuple(
        candidate.config.to_dict()
        for candidate in _candidates_for_geometry(geometry, sm_count=48)
    )

    assert sum(config["backend"] == "micro" for config in configs) == 1
    assert {
        config["dynamic_tile_m"] for config in configs if config["backend"] == "dynamic"
    } == {16, 32, 64, 128}
    assert {
        config["dynamic_route_mode"]
        for config in configs
        if config["backend"] == "dynamic"
    } == route_modes
    assert all(config["w4a16_route_mode"] is None for config in configs)


def test_nvfp4_triton_route_candidates_only_use_the_supported_tile() -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == "nvfp4"
        and geometry.activation == "silu"
    )

    configs = tuple(
        candidate.config
        for candidate in _candidates_for_geometry(geometry, sm_count=48)
        if candidate.config["route_planner"] == "triton"
    )

    assert configs
    assert {config["dynamic_tile_m"] for config in configs} == {16}
    assert {config["dynamic_route_mode"] for config in configs} == {"grouped"}


def test_every_corpus_geometry_and_capacity_has_an_applicable_candidate() -> None:
    candidates_by_geometry = {}
    checked = set()
    missing = []
    for case in expand_sweep_cases():
        key = (case.geometry.key, case.top_k, case.num_tokens)
        if key in checked:
            continue
        checked.add(key)
        candidates = candidates_by_geometry.get(case.geometry.key)
        if candidates is None:
            candidates = _candidates_for_geometry(case.geometry, sm_count=188)
            candidates_by_geometry[case.geometry.key] = candidates
        if not moe_gpu_worker._eligible_candidates_for_case(
            case.geometry,
            case,
            candidates,
        ):
            missing.append(case.case_id)

    assert not missing


@pytest.mark.parametrize(
    ("quant_mode", "expect_dynamic_direct"),
    (("w4a8_mx", True), ("w4a8_nvfp4", False)),
)
def test_w4a8_tuner_filters_dynamic_specializations_by_real_support(
    quant_mode: str,
    expect_dynamic_direct: bool,
) -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == quant_mode
        and geometry.activation == "silu"
    )
    cases = expand_sweep_cases(geometries=(geometry,))
    small = next(
        case
        for case in cases
        if case.is_model_native_top_k
        and case.num_tokens == 1
        and case.route_pattern == "balanced"
    )
    large = next(
        case
        for case in cases
        if case.is_model_native_top_k
        and case.num_tokens == 8
        and case.route_pattern == "balanced"
    )
    candidates = _candidates_for_geometry(geometry, sm_count=48)

    small_configs = tuple(
        candidate.config
        for candidate in moe_gpu_worker._eligible_candidates_for_case(
            geometry,
            small,
            candidates,
        )
    )
    large_configs = tuple(
        candidate.config
        for candidate in moe_gpu_worker._eligible_candidates_for_case(
            geometry,
            large,
            candidates,
        )
    )

    assert any(config["backend"] == "micro" for config in small_configs)
    assert any(
        config["backend"] == "dynamic"
        and config["dynamic_tile_m"] == 16
        and config["dynamic_route_mode"] == "grouped"
        for config in small_configs
    )
    assert (
        any(
            config["backend"] == "dynamic"
            and config["dynamic_tile_m"] == 16
            and config["dynamic_route_mode"] == "direct"
            for config in small_configs
        )
        is expect_dynamic_direct
    )
    assert not any(
        config["backend"] == "dynamic"
        and config["dynamic_route_mode"] == "direct"
        for config in large_configs
    )
    if quant_mode == "w4a8_nvfp4":
        assert not any(config["backend"] == "micro" for config in large_configs)


def test_nvfp4_relu2_tuner_excludes_oversized_dynamic_tiles() -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == "nvfp4"
        and geometry.activation == "relu2"
    )

    candidates = _candidates_for_geometry(geometry, sm_count=48)
    tile_ms = {
        candidate.config["dynamic_tile_m"]
        for candidate in candidates
        if candidate.config["backend"] == "dynamic"
    }

    assert tile_ms == {64, 128}


def test_w4a8_tuner_does_not_treat_gb10_micro_preference_as_support() -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == "w4a8_mx"
        and geometry.hidden_size == 6144
        and geometry.intermediate_size == 1024
    )
    case = next(
        case
        for case in expand_sweep_cases(geometries=(geometry,))
        if case.is_model_native_top_k
        and case.num_tokens == 3
        and case.route_pattern == "balanced"
    )
    candidates = _candidates_for_geometry(geometry, sm_count=_DEVICE.sm_count)

    eligible = moe_gpu_worker._eligible_candidates_for_case(
        geometry,
        case,
        candidates,
    )
    heuristic = fused_moe_impl._heuristic_moe_decode_config(
        fused_moe_impl.MoeDecodeQuery(
            quant_mode=geometry.recipe.quant_mode,
            source_format=geometry.recipe.source_format,
            activation=geometry.activation,
            num_experts=geometry.num_experts,
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
            top_k=case.top_k,
            num_tokens=case.num_tokens,
            routed_rows=case.routed_rows,
        ),
        _DEVICE,
    )

    assert any(candidate.config["backend"] == "micro" for candidate in eligible)
    assert heuristic.backend == "dynamic"


class _Session(AbstractContextManager["_Session"]):
    def __init__(
        self,
        calls: list[tuple[str, tuple[str, ...]]],
        *,
        quant_mode: str,
    ) -> None:
        self._calls = calls
        if quant_mode == "w4a16":
            configs = (
                {
                    "backend": "w4a16",
                    "dynamic_route_mode": None,
                    "dynamic_tile_m": None,
                    "route_planner": "internal",
                    "max_active_clusters": None,
                    "w4a16_route_mode": route_mode,
                }
                for route_mode in ("packed", "direct")
            )
        else:
            backends = ("dynamic",) if quant_mode == "w6a8_mx" else ("micro", "dynamic")
            configs = (
                {
                    "backend": backend,
                    "dynamic_route_mode": (
                        "grouped" if backend == "dynamic" else None
                    ),
                    "dynamic_tile_m": 128 if backend == "dynamic" else None,
                    "route_planner": "internal",
                    "max_active_clusters": None,
                    "w4a16_route_mode": None,
                }
                for backend in backends
            )
        self._candidates = tuple(MoeCandidate.create(config) for config in configs)

    @property
    def candidates(self) -> tuple[MoeCandidate, ...]:
        return self._candidates

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def eligible_candidates(self, case, candidates):
        return tuple(
            candidate
            for candidate in candidates
            if not (candidate.config["backend"] == "micro" and case.num_tokens > 8)
            and not (
                candidate.config.get("w4a16_route_mode") == "direct"
                and case.num_tokens > 8
            )
        )

    def measure(self, case, candidates, *, correctness=False):
        del correctness
        self._calls.append(
            (case.case_id, tuple(candidate.candidate_id for candidate in candidates))
        )
        measurements = []
        for candidate in candidates:
            backend = candidate.config["backend"]
            route_mode = candidate.config.get("w4a16_route_mode")
            is_decode_candidate = backend == "micro" or route_mode == "direct"
            if case.num_tokens == 1:
                latency = 10.0 if is_decode_candidate else 20.0
            else:
                latency = 30.0 if is_decode_candidate else 15.0
            if case.route_pattern == "hot":
                latency *= 1.1
            measurements.append(
                MoeMeasurement(
                    candidate=candidate,
                    latency_us=latency,
                    cosine=0.9995,
                    metrics={"graph_cosine": 1.0},
                )
            )
        return tuple(measurements)


@dataclass
class _Factory:
    calls: list[tuple[str, tuple[str, ...]]]

    def __call__(self, geometry, context):
        del context
        return _Session(
            self.calls,
            quant_mode=geometry.recipe.quant_mode,
        )


class _CandidateSpecificScreenSession(_Session):
    def eligible_candidates(self, case, candidates):
        return tuple(
            candidate
            for candidate in candidates
            if candidate.config["backend"] != "micro" or case.num_tokens == 1
        )


@dataclass
class _CandidateSpecificScreenFactory:
    calls: list[tuple[str, tuple[str, ...]]]

    def __call__(self, geometry, context):
        del context
        return _CandidateSpecificScreenSession(
            self.calls,
            quant_mode=geometry.recipe.quant_mode,
        )


def _generator(
    calls,
    *,
    quant_mode: str = "nvfp4",
    tp_sizes: tuple[int, ...] = (1,),
    token_counts: tuple[int, ...] = (1, 4),
    top_ks: tuple[int, ...] = (2,),
):
    recipe = MoeRecipe(
        recipe_id=f"test-{quant_mode}",
        family_id="test",
        quant_mode=quant_mode,
        source_format="modelopt_nvfp4",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
        compatible_activations=("silu",),
    )
    model = MoeModelGeometry(
        model_id="test-model",
        hidden_size=256,
        intermediate_size=64,
        num_experts=16,
        native_top_k=2,
        activation="silu",
        recipe_families=(recipe.family_id,),
        source="test",
        tp_sizes=tp_sizes,
    )
    geometries = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    cases = expand_sweep_cases(
        geometries=geometries,
        top_ks=top_ks,
        token_counts=token_counts,
        route_patterns=("balanced", "hot"),
    )
    return MoeDecodeGenerator(
        benchmark_factory=_Factory(calls),
        geometries=geometries,
        cases=cases,
    )


def _context(tmp_path):
    return GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )


def test_moe_measurement_partition_keeps_one_physical_geometry(tmp_path) -> None:
    generator = _generator([], tp_sizes=(1, 2))
    context = _context(tmp_path)

    partitions = generator.measurement_partitions(context)
    restricted = generator.select_measurement_partitions((partitions[0].partition_id,))

    assert len(partitions) == 2
    assert partitions[0].component_id == "moe.decode"
    assert restricted.estimate(context).case_count == partitions[0].case_count
    assert (
        restricted.estimate(context).case_count < generator.estimate(context).case_count
    )


def test_staged_moe_generator_keeps_regional_winners_and_resumes(tmp_path) -> None:
    calls = []
    generator = _generator(calls)
    context = _context(tmp_path)
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

    assert first_call_count == 9
    assert len(calls) == first_call_count
    assert result.component == resumed.component
    assert result.component["coverage"]["measured_query_points"] == 2
    assert result.component["coverage"]["runtime_query_points"] == 4

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
    component = profile.component("moe.decode")
    assert component is not None
    base_query = {
        "quant_mode": "nvfp4",
        "source_format": "modelopt_nvfp4",
        "activation": "silu",
        "num_experts": 16,
        "hidden_size": 256,
        "intermediate_size": 64,
        "top_k": 2,
    }
    assert (
        component.lookup({**base_query, "num_tokens": 1}).config["backend"] == "micro"
    )
    assert (
        component.lookup({**base_query, "num_tokens": 4}).config["backend"] == "dynamic"
    )
    assert (
        component.lookup({**base_query, "num_tokens": 2}).config["backend"] == "micro"
    )
    assert (
        component.lookup({**base_query, "num_tokens": 3}).config["backend"] == "dynamic"
    )
    assert component.lookup({**base_query, "num_tokens": 2_048}) is None


def test_moe_resume_reuses_a_cached_candidate_superset(tmp_path) -> None:
    calls = []
    generator = _generator(calls)
    context = _context(tmp_path)
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    case = generator._cases[0]
    session = _Session(calls, quant_mode="nvfp4")

    generator._race(
        stage="full",
        case=case,
        candidates=session.candidates,
        session=session,
        context=context,
        checkpoints=checkpoints,
    )
    calls.clear()
    requested = session.candidates[1:]

    resumed = generator._race(
        stage="full",
        case=case,
        candidates=requested,
        session=session,
        context=context,
        checkpoints=checkpoints,
    )

    assert calls == []
    assert tuple(item.candidate for item in resumed) == requested


def test_correctness_screen_uses_an_eligible_anchor_per_candidate(tmp_path) -> None:
    calls = []
    generator = _generator(calls, token_counts=(1, 4))
    generator._benchmark_factory = _CandidateSpecificScreenFactory(calls)

    result = generator.generate(
        _context(tmp_path),
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )

    cases_by_id = {case.case_id: case for case in generator._cases}
    screen_tokens = [cases_by_id[case_id].num_tokens for case_id, _ in calls[:2]]
    assert screen_tokens == [1, 4]
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
    component = profile.component("moe.decode")
    assert component is not None
    assert (
        component.lookup(
            {
                "quant_mode": "nvfp4",
                "source_format": "modelopt_nvfp4",
                "activation": "silu",
                "num_experts": 16,
                "hidden_size": 256,
                "intermediate_size": 64,
                "top_k": 2,
                "num_tokens": 1,
            }
        ).config["backend"]
        == "micro"
    )


def test_dynamic_correctness_screen_targets_its_route_tile() -> None:
    generator = _generator([], token_counts=(4, 8, 16), top_ks=(6,))
    cases = tuple(
        case
        for case in generator._cases
        if case.route_pattern == "balanced" and case.top_k == 6
    )
    candidate = MoeCandidate.create(
        {
            "backend": "dynamic",
            "dynamic_route_mode": "grouped",
            "dynamic_tile_m": 64,
            "route_planner": "internal",
            "max_active_clusters": None,
            "w4a16_route_mode": None,
        }
    )

    anchor = min(cases, key=lambda case: _correctness_anchor_rank(case, candidate))

    assert anchor.num_tokens == 8


def test_prefill_capacities_are_measured_for_each_top_k(
    tmp_path,
) -> None:
    calls = []
    generator = _generator(
        calls,
        token_counts=(1, 512, 1_024),
        top_ks=(1, 2),
    )

    result = generator.generate(
        _context(tmp_path),
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )

    cases_by_id = {case.case_id: case for case in generator._cases}
    capacity_calls = [
        cases_by_id[case_id]
        for case_id, _candidate_ids in calls
        if cases_by_id[case_id].num_tokens >= 512
    ]
    assert {(case.num_tokens, case.top_k) for case in capacity_calls} == {
        (512, 1),
        (512, 2),
        (1_024, 1),
        (1_024, 2),
    }
    assert result.component["coverage"]["measured_query_points"] == 6

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
    component = profile.component("moe.decode")
    assert component is not None
    for num_tokens in (512, 1_024):
        for top_k in (1, 2):
            hit = component.lookup(
                {
                    "quant_mode": "nvfp4",
                    "source_format": "modelopt_nvfp4",
                    "activation": "silu",
                    "num_experts": 16,
                    "hidden_size": 256,
                    "intermediate_size": 64,
                    "top_k": top_k,
                    "num_tokens": num_tokens,
                }
            )
            assert hit is not None
            assert hit.config["backend"] == "dynamic"


@pytest.mark.parametrize("top_k", (1, 10, 16))
def test_sparse_token_capacity_coverage_matches_dense_reference(top_k: int) -> None:
    base_query = {
        "quant_mode": "nvfp4",
        "source_format": "modelopt_nvfp4",
        "activation": "silu",
        "num_experts": 16,
        "hidden_size": 256,
        "intermediate_size": 64,
        "top_k": top_k,
    }
    configs = {
        "micro": {
            "backend": "micro",
            "dynamic_route_mode": None,
            "dynamic_tile_m": None,
            "route_planner": "internal",
            "max_active_clusters": None,
            "w4a16_route_mode": None,
        },
        "triton": {
            "backend": "dynamic",
            "dynamic_route_mode": "grouped",
            "dynamic_tile_m": 16,
            "route_planner": "triton",
            "max_active_clusters": 24,
            "w4a16_route_mode": None,
        },
        "dynamic": {
            "backend": "dynamic",
            "dynamic_route_mode": "grouped",
            "dynamic_tile_m": 128,
            "route_planner": "internal",
            "max_active_clusters": None,
            "w4a16_route_mode": None,
        },
    }
    measured = tuple(
        DecisionRecord.create(
            query={**base_query, "num_tokens": num_tokens},
            config=configs[config],
        )
        for num_tokens, config in (
            (1, "micro"),
            (min(128, 256 // top_k), "triton"),
            (512, "dynamic"),
            (1_024, "dynamic"),
        )
    )
    dense = synthesize_integer_axis_coverage(
        measured,
        field="num_tokens",
        minimum=1,
        maximum=1_024,
        config_is_valid=_config_covers_query,
    )
    sparse = _synthesize_token_capacity_coverage(
        measured,
        minimum=1,
        maximum=1_024,
    )
    assert len(sparse) < len(dense) // 10

    field_order = (*base_query, "num_tokens")
    dense_tree = build_axis_tree(
        dense,
        field_order=field_order,
        range_fields=frozenset({"num_tokens"}),
    )
    sparse_tree = build_axis_tree(
        sparse,
        field_order=field_order,
        range_fields=frozenset({"num_tokens"}),
        nearest_range_bounds={"num_tokens": (1, 1_024)},
    )

    def component(planner):
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
                "components": [
                    {
                        "component_id": "moe.decode",
                        "query_schema_version": 4,
                        "config_schema_version": 3,
                        "planner": decision_node_to_dict(planner),
                    }
                ],
            }
        )
        return profile.component("moe.decode")

    dense_component = component(dense_tree)
    sparse_component = component(sparse_tree)
    assert dense_component is not None
    assert sparse_component is not None
    for num_tokens in range(1, 1_025):
        query = {**base_query, "num_tokens": num_tokens}
        assert (
            sparse_component.lookup(query).config
            == dense_component.lookup(query).config
        )


@pytest.mark.parametrize("top_k", (1, 2))
def test_sparse_token_coverage_preserves_dynamic_direct_boundary(top_k: int) -> None:
    base_query = {
        "quant_mode": "nvfp4",
        "source_format": "modelopt_nvfp4",
        "activation": "silu",
        "num_experts": 256,
        "hidden_size": 4096,
        "intermediate_size": 1024,
        "top_k": top_k,
    }
    direct = {
        "backend": "dynamic",
        "dynamic_route_mode": "direct",
        "dynamic_tile_m": 16,
        "route_planner": "internal",
        "max_active_clusters": None,
        "w4a16_route_mode": None,
    }
    grouped = {**direct, "dynamic_route_mode": "grouped"}
    direct_limit = fused_moe_impl._DIRECT_ROUTING_MAX_ROUTED_ROWS // top_k
    measured = tuple(
        DecisionRecord.create(
            query={**base_query, "num_tokens": num_tokens},
            config=config,
        )
        for num_tokens, config in (
            (1, direct),
            (direct_limit, direct),
            (64, grouped),
            (128, grouped),
        )
    )
    dense = synthesize_integer_axis_coverage(
        measured,
        field="num_tokens",
        minimum=1,
        maximum=128,
        config_is_valid=_config_covers_query,
    )
    sparse = _synthesize_token_capacity_coverage(
        measured,
        minimum=1,
        maximum=128,
    )
    field_order = (*base_query, "num_tokens")
    dense_tree = build_axis_tree(
        dense,
        field_order=field_order,
        range_fields=frozenset({"num_tokens"}),
    )
    sparse_tree = build_axis_tree(
        sparse,
        field_order=field_order,
        range_fields=frozenset({"num_tokens"}),
        nearest_range_bounds={"num_tokens": (1, 128)},
    )

    for num_tokens in range(1, 129):
        query = {**base_query, "num_tokens": num_tokens}
        assert sparse_tree.lookup(query).config == dense_tree.lookup(query).config


def test_moe_resume_retries_checkpoint_when_every_candidate_errored(
    tmp_path,
) -> None:
    calls = []
    generator = _generator(calls)
    context = _context(tmp_path)
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    case = next(
        case
        for case in generator._cases
        if case.num_tokens == 4 and case.route_pattern == "balanced"
    )
    candidates = _Session([], quant_mode="nvfp4").candidates
    checkpoints.save(
        generator.component_id,
        f"screen-{case.case_id}",
        {
            "schema_version": 2,
            "candidate_contract_version": 1,
            "generation": context.checkpoint_metadata(),
            "case_id": case.case_id,
            "query": case.query(),
            "route_pattern": case.route_pattern,
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "measurements": [
                MoeMeasurement(
                    candidate=candidate,
                    latency_us=None,
                    cosine=None,
                    error="transient worker failure",
                ).to_dict()
                for candidate in candidates
            ],
        },
    )

    generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )

    assert calls[0][0] == case.case_id
    assert set(calls[0][1]) == {candidate.candidate_id for candidate in candidates}
    refreshed = checkpoints.load(
        generator.component_id,
        f"screen-{case.case_id}",
    )
    assert refreshed is not None
    assert all(
        measurement["error"] is None for measurement in refreshed["measurements"]
    )


def test_moe_measurement_inputs_are_route_pattern_controlled() -> None:
    cases = _generator([])._cases
    matched = [case for case in cases if case.top_k == 2 and case.num_tokens == 4]

    assert len({_measurement_seed(case, base_seed=17) for case in matched}) == 1


def test_w4a8_screening_allows_expected_quantized_norm_error() -> None:
    w4a8 = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == "w4a8_nvfp4"
    )
    nvfp4 = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == "nvfp4"
    )

    assert _maximum_relative_norm_error(w4a8) == 0.12
    assert _maximum_relative_norm_error(nvfp4) == 0.1


def test_moe_zero_outputs_have_defined_correctness_cosine() -> None:
    zeros = torch.zeros(8)

    assert _cosine_similarity(zeros, zeros.clone()) == 1.0
    assert _cosine_similarity(zeros, torch.ones(8)) == 0.0


def test_moe_cosine_does_not_overflow_on_large_finite_outputs() -> None:
    large = torch.full((8_192,), 1.0e30)

    assert _cosine_similarity(large, large.clone()) == pytest.approx(1.0)


def test_moe_nonfinite_diagnostics_are_strict_json_values() -> None:
    assert _finite_float_or_none(1.25) == 1.25
    assert _finite_float_or_none(float("inf")) is None
    assert _finite_float_or_none(float("nan")) is None


def test_moe_accelerator_errors_are_fatal() -> None:
    assert _is_fatal_accelerator_error(torch.AcceleratorError("CUDA fault"))
    assert not _is_fatal_accelerator_error(RuntimeError("candidate unsupported"))


def test_moe_query_teardown_explicitly_resets_cuda_graphs() -> None:
    resets = []
    prepared = tuple(
        SimpleNamespace(graph=SimpleNamespace(reset=lambda i=index: resets.append(i)))
        for index in range(3)
    )

    _reset_cuda_graphs(prepared)

    assert resets == [0, 1, 2]


def test_failed_first_candidate_cannot_become_cross_candidate_reference() -> None:
    output = object()

    assert (
        _candidate_reference_output(
            output,
            correct=False,
            has_reference=False,
        )
        is None
    )
    assert (
        _candidate_reference_output(
            output,
            correct=True,
            has_reference=False,
        )
        is output
    )


def test_independent_nvfp4_oracle_matches_kernel_scale_math(monkeypatch) -> None:
    from b12x.moe._shared.kernels import reference

    geometry = _generator([])._geometries[0]
    session = object.__new__(_MoeGeometrySession)
    session._geometry = geometry
    session._experts = SimpleNamespace(
        _impl=SimpleNamespace(
            w1_fp4=object(),
            w1_blockscale=object(),
            w1_alphas=object(),
            w2_fp4=object(),
            w2_blockscale=object(),
            w2_alphas=object(),
            a1_gscale=object(),
            a2_gscale=object(),
        )
    )
    captured = {}

    def fake_reference(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "reference"

    monkeypatch.setattr(reference, "moe_reference_nvfp4", fake_reference)

    result = session._independent_reference(
        x=object(),
        topk_ids=object(),
        topk_weights=object(),
    )

    assert result == "reference"
    assert captured["kwargs"]["quant_scale_math"] == "reciprocal_multiply"


def test_modelopt_profile_weights_pad_and_swizzle_scale_atoms() -> None:
    recipe = MoeRecipe(
        recipe_id="modelopt-nvfp4-test",
        family_id="test",
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
        compatible_activations=("silu",),
    )
    model = MoeModelGeometry(
        model_id="low-width-test",
        hidden_size=256,
        intermediate_size=32,
        num_experts=1,
        native_top_k=1,
        activation="silu",
        recipe_families=(recipe.family_id,),
        source="test",
        tp_sizes=(1,),
    )
    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )

    prepared = _packed_weights(geometry, device="cpu")._impl

    assert tuple(prepared.w1_blockscale.shape) == (1, 128, 16)
    assert tuple(prepared.w2_blockscale.shape) == (1, 256, 4)


@pytest.mark.parametrize(
    ("variant", "activation", "expected_dtype"),
    (
        ("glm-mcg-projection-tiered", "silu", torch.bfloat16),
        ("k3-sqg-uniform-coupled", "situ", torch.float16),
    ),
)
def test_trellis_profile_worker_uses_serving_dtype(
    monkeypatch,
    variant,
    activation,
    expected_dtype,
) -> None:
    from b12x.moe import fused_moe

    recipe = MoeRecipe(
        recipe_id="trellis-test",
        family_id="test",
        quant_mode="w4a16",
        source_format="b12x_trellis",
        intermediate_alignment=128,
        minimum_intermediate_size=128,
        compatible_activations=(activation,),
        trellis_variant=variant,
    )
    model = MoeModelGeometry(
        model_id="trellis-test",
        hidden_size=128,
        intermediate_size=128,
        num_experts=3,
        native_top_k=1,
        activation=activation,
        recipe_families=(recipe.family_id,),
        source="test",
        tp_sizes=(1,),
    )
    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    captured = {}

    def plan_weights(*, source, activation, geometry):
        captured.update(
            source=source,
            activation=activation,
            geometry=geometry,
        )
        return object()

    monkeypatch.setattr(fused_moe, "plan_weights", plan_weights)
    monkeypatch.setattr(
        fused_moe,
        "prepare_weights",
        lambda *, plan, weights: SimpleNamespace(plan=plan, weights=weights),
    )

    _trellis_weights(geometry, device="cpu")

    assert captured["activation"].io_dtype is expected_dtype


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability(torch.cuda.current_device())[0] != 12,
    reason="requires an SM120/SM121 GPU",
)
def test_projection_mixed_bind_initializes_grid_barrier_workspace() -> None:
    from b12x.moe import fused_moe
    from b12x.policy import MOE_DECODE, get_auto_policy

    recipe = MoeRecipe(
        recipe_id="trellis-bind-test",
        family_id="test",
        quant_mode="w4a16",
        source_format="b12x_trellis",
        intermediate_alignment=128,
        minimum_intermediate_size=128,
        compatible_activations=("silu",),
        trellis_variant="glm-mcg-projection-tiered",
    )
    model = MoeModelGeometry(
        model_id="trellis-bind-test",
        hidden_size=128,
        intermediate_size=256,
        num_experts=3,
        native_top_k=1,
        activation="silu",
        recipe_families=(recipe.family_id,),
        source="test",
        tp_sizes=(1,),
    )
    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    device = torch.device("cuda", torch.cuda.current_device())
    experts = _trellis_weights(geometry, device=device)
    policy = get_auto_policy(device).with_override(
        MOE_DECODE,
        fused_moe.MoeDecodeConfig.from_profile(
            {
                "backend": "w4a16",
                "dynamic_route_mode": None,
                "dynamic_tile_m": None,
                "route_planner": "internal",
                "max_active_clusters": None,
                "w4a16_route_mode": "packed",
            }
        ),
    )
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=1, top_k=1),
        policy=policy,
    )
    fused_moe.prewarm(plan)
    scratch = {
        spec.name: torch.empty(
            spec.shape,
            dtype=spec.dtype,
            device=spec.device,
        ).fill_(1)
        for spec in plan.scratch_specs()
    }
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=torch.zeros((1, 128), dtype=torch.bfloat16, device=device),
        experts=experts,
        topk_weights=torch.ones((1, 1), dtype=torch.float32, device=device),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32, device=device),
        input_scales_static=True,
    )

    workspace = binding.mixed_trellis_buffers.workspace
    assert int(torch.count_nonzero(workspace)) == 0


def test_w4a16_geometries_race_direct_and_packed_kernels(
    tmp_path,
) -> None:
    calls = []
    generator = _generator(calls, quant_mode="w4a16")

    result = generator.generate(
        _context(tmp_path),
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )

    assert len(calls) == 9
    assert result.evidence["gpu_measurement_cases"] == 9
    assert "precomputed_cases" not in result.evidence
    assert result.component["coverage"]["measured_query_points"] == 2
    assert result.component["coverage"]["runtime_query_points"] == 4


def test_w4a16_prefill_capacities_are_measured(tmp_path) -> None:
    calls = []
    generator = _generator(
        calls,
        quant_mode="w4a16",
        token_counts=(1, 512, 1_024),
    )

    result = generator.generate(
        _context(tmp_path),
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )

    cases_by_id = {case.case_id: case for case in generator._cases}
    measured_tokens = {
        cases_by_id[case_id].num_tokens for case_id, _candidate_ids in calls
    }
    assert measured_tokens == {1, 512, 1_024}
    assert result.component["coverage"]["measured_query_points"] == 3


def test_production_moe_factory_isolates_each_geometry_process(tmp_path) -> None:
    generator = _generator([])

    session = MoeGpuBenchmarkFactory()(generator._geometries[0], _context(tmp_path))

    assert isinstance(session, _MoeProcessSession)


def test_moe_process_session_retries_accelerator_failure_once(
    monkeypatch,
    tmp_path,
) -> None:
    generator = _generator([])
    geometry = generator._geometries[0]
    case = next(case for case in generator._cases if case.geometry == geometry)
    session = _MoeProcessSession(geometry, _context(tmp_path))
    candidate = session.candidates[0]
    measurement = MoeMeasurement(
        candidate=candidate,
        latency_us=10.0,
        cosine=1.0,
    )
    requests = []
    starts = []
    discards = []

    monkeypatch.setattr(session, "_start", lambda: starts.append("start"))

    def request(operation, **payload):
        requests.append((operation, payload))
        if len(requests) == 1:
            raise _MoeRemoteWorkerError(
                operation="measure",
                exception_type="AcceleratorError",
                error="CUDA illegal address",
            )
        return {"measurements": [measurement.to_dict()]}

    monkeypatch.setattr(session, "_request", request)
    monkeypatch.setattr(
        session,
        "_discard_worker",
        lambda: discards.append("discard"),
    )

    (result,) = session.measure(case, (candidate,), correctness=True)

    assert len(starts) == 2
    assert len(requests) == 2
    assert discards == ["discard"]
    assert result.metrics["worker_retries"] == 1


def test_moe_remote_worker_error_is_picklable() -> None:
    original = _MoeRemoteWorkerError(
        operation="measure",
        exception_type="AcceleratorError",
        error="CUDA illegal address",
        remote_traceback="remote stack",
    )

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        restored = executor.submit(
            _MoeRemoteWorkerError,
            original.operation,
            original.exception_type,
            original.remote_error,
            original.remote_traceback,
        ).result()

    assert restored.operation == original.operation
    assert restored.exception_type == original.exception_type
    assert restored.remote_error == original.remote_error
    assert restored.remote_traceback == original.remote_traceback
    assert str(restored) == str(original)


def test_moe_process_session_isolates_failed_candidate_after_batch_failure(
    monkeypatch,
    tmp_path,
) -> None:
    generator = _generator([])
    geometry = next(
        geometry
        for geometry in generator._geometries
        if len(_candidates_for_geometry(geometry, sm_count=48)) >= 3
    )
    case = next(case for case in generator._cases if case.geometry == geometry)
    session = _MoeProcessSession(geometry, _context(tmp_path))
    candidates = session.candidates[:3]
    requests = []
    discards = []

    monkeypatch.setattr(session, "_start", lambda: None)

    def request(operation, **payload):
        candidate_ids = payload["candidate_ids"]
        requests.append(candidate_ids)
        if len(candidate_ids) > 1 or candidate_ids == (candidates[1].candidate_id,):
            raise _MoeRemoteWorkerError(
                operation="measure",
                exception_type="AcceleratorError",
                error="CUDA illegal address",
            )
        candidate = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == candidate_ids[0]
        )
        return {
            "measurements": [
                MoeMeasurement(
                    candidate=candidate,
                    latency_us=10.0,
                    cosine=1.0,
                ).to_dict()
            ]
        }

    monkeypatch.setattr(session, "_request", request)
    monkeypatch.setattr(session, "_discard_worker", lambda: discards.append(True))

    results = session.measure(case, candidates, correctness=True)

    assert requests == [
        tuple(candidate.candidate_id for candidate in candidates),
        (candidates[0].candidate_id,),
        (candidates[1].candidate_id,),
        (candidates[2].candidate_id,),
    ]
    assert len(discards) == 2
    assert results[0].error is None
    assert results[1].latency_us is None
    assert results[1].error == "AcceleratorError: CUDA illegal address"
    assert results[2].error is None
    assert all(result.metrics["worker_retries"] == 1 for result in results)


def test_moe_geometry_worker_binds_device_before_starting_session(
    monkeypatch,
) -> None:
    events = []

    class Connection:
        def __init__(self) -> None:
            self.sent = []

        def send(self, message) -> None:
            self.sent.append(message)

        def recv(self):
            return {"operation": "close"}

        def close(self) -> None:
            events.append("connection.close")

    class Session(AbstractContextManager["Session"]):
        candidates = ()

        def __init__(self, geometry, context) -> None:
            del geometry, context
            events.append("session.init")

        def __enter__(self) -> "Session":
            events.append("session.enter")
            return self

        def __exit__(self, *_exc: object) -> None:
            events.append("session.exit")

    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda ordinal: events.append(("set_device", ordinal)),
    )
    monkeypatch.setattr(moe_gpu_worker, "_MoeGeometrySession", Session)
    connection = Connection()

    moe_gpu_worker._moe_geometry_worker(
        connection,
        object(),
        SimpleNamespace(device_ordinal=7),
    )

    assert events[:3] == [("set_device", 7), "session.init", "session.enter"]
    assert connection.sent == [
        {"ok": True, "candidates": []},
        {"ok": True, "closed": True},
    ]


def test_production_moe_resume_does_not_start_a_gpu_worker(tmp_path) -> None:
    generator = _generator([])
    geometry = generator._geometries[0]
    case = next(case for case in generator._cases if case.geometry == geometry)
    session = MoeGpuBenchmarkFactory()(geometry, _context(tmp_path))

    with session:
        candidates = session.candidates
        assert candidates
        assert session.eligible_candidates(case, candidates)
        assert session._process is None


@pytest.mark.parametrize("ratio,expected", [(1.0, True), (0.999, True), (0.97, True), (1.001, False)])
def test_moe_precision_prefers_measured_parity_without_a_five_percent_margin(ratio, expected):
    from b12x.policy.generation.providers.moe_precision import qualifies

    pairs = [{"a4": 10.0, "a16": 10.0 * ratio} for _ in range(20)]
    assert qualifies(pairs, "a16", "a4")[0] is expected


@pytest.mark.parametrize("confirmed,backend", [(True, "w4a16"), (False, "micro")])
def test_precision_reduction_confirms_aggregate_without_per_route_veto(confirmed, backend):
    from b12x.policy.generation.providers.moe_precision import select_winner

    a4 = _Session([], quant_mode="nvfp4").candidates[0]
    a16 = _Session([], quant_mode="w4a16").candidates[1]
    grouped = []
    for initial in (10.005, 8.0):
        confirmation = initial if confirmed else 12.0
        grouped.append((None, (
            MoeMeasurement(candidate=a4, latency_us=10.0, cosine=1.0,
                           metrics={"initial_latency_us": 10.0, "confirmation_latency_us": 10.0}),
            MoeMeasurement(candidate=a16, latency_us=confirmation, cosine=1.0,
                           metrics={"initial_latency_us": initial, "confirmation_latency_us": confirmation}),
        )))
    assert select_winner(grouped, 0.999).config["backend"] == backend


def test_auto_precision_qualifies_each_capacity_and_preserves_exact_holes(tmp_path):
    from b12x.policy import PolicyContext, PolicyMode, PolicySource, ProfileRegistry
    from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery

    calls = []
    class Session(_Session):
        def __init__(self):
            super().__init__(calls, quant_mode="nvfp4")
            self._candidates = (self._candidates[0], _Session([], quant_mode="w4a16").candidates[1])

        def measure(self, case, candidates, *, correctness=False):
            assert not correctness
            calls.append(case.case_id)
            return tuple(MoeMeasurement(
                candidate=candidate, latency_us=10.0, cosine=1.0,
                metrics={field: (11.0 if case.num_tokens == 1 and candidate.config["backend"] == "w4a16" else 10.0)
                         for field in ("initial_latency_us", "confirmation_latency_us")},
            ) for candidate in candidates)

    generator = _generator(calls, quant_mode="nvfp4_auto")
    generator._benchmark_factory = lambda geometry, context: Session()
    context = _context(tmp_path)
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    result = generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert len(calls) == 4
    profile = profile_from_dict({
        "profile_id": "nvidia.synthetic.48sm",
        "targets": [{"vendor": "nvidia", "compute_capability": [12, 1], "sm_count": 48,
                     "product_name": "Synthetic GPU"}],
        "components": [result.component],
    })
    component = profile.component("moe.decode")
    query = generator._cases[0].query()
    for count, backend in ((1, "micro"), (4, "w4a16")):
        hit = component.lookup({**query, "num_tokens": count, "routed_rows": 2 * count})
        assert hit.config["backend"] == backend
    assert component.lookup({**query, "num_tokens": 2, "routed_rows": 4}) is None
    registry = ProfileRegistry()
    registry.register(profile)
    policy = PolicyContext.for_identity(_DEVICE, registry=registry)
    resolved = policy.resolve(MOE_DECODE_POLICY, MoeDecodeQuery(**{**query, "num_tokens": 4, "routed_rows": 8}))
    assert resolved.source is PolicySource.PREPLANNED
    assert resolved.config.backend == "w4a16"
    miss = policy.resolve(MOE_DECODE_POLICY, MoeDecodeQuery(**{**query, "num_tokens": 2, "routed_rows": 4}))
    assert miss.source is PolicySource.HEURISTIC
    assert miss.config.backend == "w4a16"
    measured_a4 = policy.resolve(MOE_DECODE_POLICY, MoeDecodeQuery(**query))
    assert measured_a4.source is PolicySource.PREPLANNED
    assert measured_a4.config.backend == "micro"
    heuristic = PolicyContext.for_identity(_DEVICE, mode=PolicyMode.HEURISTIC_ONLY, registry=registry)
    assert heuristic.resolve(MOE_DECODE_POLICY, MoeDecodeQuery(**query)).source is PolicySource.HEURISTIC
    generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert len(calls) == 4
    generator._precision_identity = "different-source-or-toolchain"
    generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert len(calls) == 8


@pytest.mark.parametrize("profile_id,route", [
    ("nvidia.gb10.48sm", "packed"),
    ("nvidia.rtx.pro.6000.blackwell.max-q", "direct"),
])
def test_embedded_auto_precision_retains_overrides_and_exact_capacity_coverage(profile_id, route):
    from dataclasses import replace
    from b12x.policy import MOE_DECODE, PolicyContext, PolicyMode, PolicySource
    from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery, MoeDecodeConfig

    profile = EMBEDDED_REGISTRY.get(profile_id)
    context = PolicyContext.for_identity(profile.targets[0], mode=PolicyMode.PREPLANNED_ONLY)
    query = MoeDecodeQuery(
        quant_mode="nvfp4_auto", source_format="modelopt_nvfp4", activation="silu",
        num_experts=256, hidden_size=6144, intermediate_size=256, top_k=8, num_tokens=1, routed_rows=8,
    )
    for count in range(1, 9):
        resolution = context.resolve(MOE_DECODE_POLICY, replace(query, num_tokens=count, routed_rows=count * 8))
        assert resolution.source is PolicySource.PREPLANNED
        assert resolution.config.backend == "w4a16"
        assert resolution.config.w4a16_route_mode == route
    automatic = PolicyContext.for_identity(profile.targets[0])
    miss = automatic.resolve(MOE_DECODE_POLICY, replace(query, num_tokens=9, routed_rows=72))
    assert miss.source is PolicySource.HEURISTIC
    assert miss.config.backend != "w4a16"
    a4 = MoeDecodeConfig(backend="micro", route_planner="internal", max_active_clusters=None)
    override = automatic.with_override(MOE_DECODE, a4).resolve(MOE_DECODE_POLICY, query)
    assert override.source is PolicySource.OVERRIDE
    assert override.config == a4
    unknown = PolicyContext.for_identity(replace(profile.targets[0], product_name="Unmeasured GPU"))
    assert unknown.resolve(MOE_DECODE_POLICY, query).source is PolicySource.HEURISTIC
    assert unknown.resolve(MOE_DECODE_POLICY, query).config.backend == "w4a16"


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("capacity", [*range(1, 10), 16])
def test_auto_precision_heuristic_promotes_supported_decode(capability, capacity):
    from dataclasses import replace
    from b12x.policy import PolicyContext, PolicyMode, PolicySource
    from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery

    device = replace(_DEVICE, compute_capability=capability)
    policy = PolicyContext.for_identity(device, mode=PolicyMode.HEURISTIC_ONLY)
    query = MoeDecodeQuery(
        quant_mode="nvfp4_auto", source_format="modelopt_nvfp4", activation="silu",
        num_experts=256, hidden_size=6144, intermediate_size=256, top_k=8,
        num_tokens=capacity, routed_rows=capacity * 8,
    )
    resolution = policy.resolve(MOE_DECODE_POLICY, query)
    assert resolution.source is PolicySource.HEURISTIC
    assert (resolution.config.backend == "w4a16") is (capacity <= 8)
    for unsupported in (
        replace(query, hidden_size=192),
        replace(query, top_k=33, routed_rows=capacity * 33),
        replace(query, quant_mode="nvfp4"),
    ):
        assert policy.resolve(MOE_DECODE_POLICY, unsupported).config.backend != "w4a16"
    for other_device in (None, replace(device, compute_capability=(10, 0))):
        other = PolicyContext.for_identity(other_device, mode=PolicyMode.HEURISTIC_ONLY)
        assert other.resolve(MOE_DECODE_POLICY, query).config.backend != "w4a16"
