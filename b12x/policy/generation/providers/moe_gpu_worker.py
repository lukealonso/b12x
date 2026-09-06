"""Production GPU measurement worker for the built-in MoE corpus."""

from __future__ import annotations

import gc
import hashlib
import math
import multiprocessing
import os
import statistics
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection

from b12x.policy.generation.contracts import GenerationContext
from b12x.policy.generation.moe_corpus import (
    MoePhysicalGeometry,
    MoeSweepCase,
)

from .moe import MoeCandidate, MoeMeasurement

_MAX_RELATIVE_NORM_ERROR = 0.1
_W4A8_MAX_RELATIVE_NORM_ERROR = 0.12
_TUNER_OVERRIDE_ENV = (
    "B12X_DIRECT_CUTE_OPTIONS",
    "B12X_DYNAMIC_DETERMINISTIC_OUTPUT",
    "B12X_DYNAMIC_ENABLE_MULTICTA",
    "B12X_DYNAMIC_EXTERNAL_ROUTE_PLAN",
    "B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS",
    "B12X_DYNAMIC_SWAP_AB",
    "B12X_DYNAMIC_TILE_MN",
    "B12X_DYNAMIC_W4A8_MATERIALIZED",
    "B12X_DYNAMIC_W4A8_REPACKED",
    "B12X_DYNAMIC_W4A8_SHARE_INPUT",
    "B12X_DYNAMIC_WORK_SOURCE",
    "B12X_ENABLE_DYNAMIC_DOWN_SCALE",
    "B12X_LEVEL10_ENABLE_MULTICTA",
    "B12X_LEVEL10_MAX_ACTIVE_CLUSTERS",
    "B12X_MICRO_DYNAMIC_CUTOVER_PAIRS",
    "B12X_MICRO_MAX_ACTIVE_CLUSTERS",
    "B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS",
    "B12X_MOE_TILE_MN",
    "B12X_NVFP4_SPLIT_DECODE",
    "B12X_W4A16_SMALL_M_DIRECT",
    "B12X_W4A16_SMALL_M_HOST_BARRIER_RESET",
    "B12X_W4A16_SMALL_M_SPLITK",
    "B12X_W4A8_TINY_DECODE",
)


@contextmanager
def _candidate_environment(candidate: MoeCandidate):
    del candidate
    previous = {name: os.environ.get(name) for name in _TUNER_OVERRIDE_ENV}
    try:
        for name in _TUNER_OVERRIDE_ENV:
            os.environ.pop(name, None)
        from b12x.moe.fused_moe import _impl

        _impl._DYNAMIC_DOWN_SCALE_CACHE = None
        _impl._DYNAMIC_MULTICTA_CACHE = None
        _impl._MAC_CACHE.clear()
        _impl._MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE.clear()
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _activation_mode(quant_mode: str):
    from b12x.moe import fused_moe

    return {
        "nvfp4_auto": fused_moe.ActivationMode.AUTO,
        "nvfp4": fused_moe.ActivationMode.A4,
        "w4a16": fused_moe.ActivationMode.A16,
        "w4a8_mx": fused_moe.ActivationMode.A8,
        "w4a8_nvfp4": fused_moe.ActivationMode.A8,
    }[quant_mode]


def _benchmark_input_scale(geometry: MoePhysicalGeometry) -> float:
    if geometry.recipe.source_format in {"btx", "b12x_trellis"}:
        return 1.0e-2
    return 1.0


def _maximum_relative_norm_error(geometry: MoePhysicalGeometry) -> float:
    if geometry.recipe.quant_mode in {"w4a8_mx", "w4a8_nvfp4"}:
        return _W4A8_MAX_RELATIVE_NORM_ERROR
    return _MAX_RELATIVE_NORM_ERROR


def _condition_benchmark_inputs(
    geometry: MoePhysicalGeometry,
    x: object,
):
    """Keep uniform W4A16 fixtures out of saturated activation tails."""

    if not (
        geometry.recipe.quant_mode == "w4a16"
        and geometry.recipe.source_format not in {"btx", "b12x_trellis"}
    ):
        return x
    x.sub_(x.mean(dim=-1, keepdim=True))
    x.add_(geometry.hidden_size**-0.5)
    return x


def _packed_weights(
    geometry: MoePhysicalGeometry,
    *,
    device: object,
):
    import torch

    from b12x._lib.intrinsics import swizzle_block_scale
    from b12x.moe import fused_moe
    from b12x.moe._shared.kernels.activations import moe_activation_w1_rows

    source_format = geometry.recipe.source_format
    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat(source_format),
        w13_layout=(
            fused_moe.W13Layout.W31
            if source_format == "modelopt_nvfp4" and geometry.recipe.quant_mode != "nvfp4_auto"
            else fused_moe.W13Layout.W13
        ),
    )
    weight_plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode=_activation_mode(geometry.recipe.quant_mode),
            nonlinearity=geometry.activation,
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=geometry.num_experts,
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
        ),
    )
    experts = geometry.num_experts
    hidden = geometry.hidden_size
    intermediate = geometry.intermediate_size
    w13_rows = moe_activation_w1_rows(geometry.activation, intermediate)
    w13 = torch.full(
        (experts, w13_rows, hidden // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.full(
        (experts, hidden, intermediate // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    scale_group = 32 if source_format == "fp4_e8m0_k32" else 16
    if source_format == "fp4_e8m0_k32":
        scale_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
        scale_value = 127 if scale_dtype == torch.uint8 else 1.0
    else:
        scale_dtype = torch.float8_e4m3fn
        scale_value = 1.0
    w13_scales = torch.full(
        (experts, w13_rows, hidden // scale_group),
        scale_value,
        dtype=scale_dtype,
        device=device,
    )
    w2_scales = torch.full(
        (experts, hidden, _ceil_div(intermediate, scale_group)),
        scale_value,
        dtype=scale_dtype,
        device=device,
    )
    if source_format == "modelopt_nvfp4":
        w13_scales = swizzle_block_scale(w13_scales)
        w2_scales = swizzle_block_scale(w2_scales)
    unit = torch.ones(experts, dtype=torch.float32, device=device)
    automatic = geometry.recipe.quant_mode == "nvfp4_auto"
    weight_global = unit * (1.0 / 128.0) if automatic else unit
    activation_global = unit * 32.0 if automatic else unit
    packed = fused_moe.PackedWeights(
        w13=w13,
        w2=w2,
        w13_block_scales=w13_scales,
        w2_block_scales=w2_scales,
        w13_global_scales=weight_global,
        w2_global_scales=weight_global,
        input_scale=(activation_global if geometry.recipe.quant_mode != "w4a16" else None),
        intermediate_scale=(activation_global if geometry.recipe.quant_mode != "w4a16" else None),
    )
    return fused_moe.prepare_weights(plan=weight_plan, weights=packed)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _trellis_config(*, variant: str):
    from b12x.moe import fused_moe

    if variant == "k3-sqg-uniform-coupled":
        value = {
            "version": 2,
            "codebook": "sqg_e4m3",
            "rate": {"granularity": "uniform"},
            "scale": {
                "input_scales": {
                    "vectors": "per_layer",
                    "gains": "per_layer",
                },
                "intermediate_scales": {
                    "vectors": "per_layer",
                    "gains": "per_expert",
                },
                "output_scales": {
                    "vectors": "per_layer",
                    "gains": "per_layer",
                },
            },
            "transform": {
                "projection": {
                    "kind": "scaled_hadamard",
                    "block_size": 128,
                },
                "expert": {
                    "kind": "coupled_hadamard",
                    "pre_block_size": 512,
                    "post_block_size": 128,
                    "draw_granularity": "per_expert",
                },
            },
        }
    elif variant == "glm-mcg-projection-tiered":
        value = {
            "version": 2,
            "codebook": "mcg",
            "rate": {"granularity": "per_expert_projection"},
            "scale": {
                "input_scales": {
                    "vectors": "per_layer",
                    "gains": "none",
                },
                "intermediate_scales": {
                    "vectors": "per_expert",
                    "gains": "none",
                },
                "output_scales": {
                    "vectors": "per_layer",
                    "gains": "none",
                },
            },
            "transform": {
                "projection": {
                    "kind": "scaled_hadamard",
                    "block_size": 128,
                },
                "expert": {"kind": "none"},
            },
        }
    else:
        raise ValueError(f"unsupported Trellis benchmark variant {variant!r}")
    return fused_moe.TrellisConfig.from_dict(value)


def _trellis_weights(
    geometry: MoePhysicalGeometry,
    *,
    device: object,
):
    import torch

    from b12x.moe import fused_moe

    variant = geometry.recipe.trellis_variant
    if variant is None:
        raise ValueError("Trellis geometry is missing its Trellis variant")
    config = _trellis_config(variant=variant)
    io_dtype = torch.float16 if variant == "k3-sqg-uniform-coupled" else torch.bfloat16
    weight_plan = fused_moe.plan_weights(
        source=config,
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity=geometry.activation,
            io_dtype=io_dtype,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=geometry.num_experts,
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
        ),
    )
    experts = geometry.num_experts
    hidden = geometry.hidden_size
    intermediate = geometry.intermediate_size
    if variant == "k3-sqg-uniform-coupled":
        row_bytes = experts * 3 * (hidden // 16) * 64 * 3
        rate = torch.tensor([0x33], dtype=torch.uint8, device=device)
        input_scales = fused_moe.ScaleFactors(
            torch.ones(hidden, dtype=torch.float16, device=device),
            torch.ones(1, dtype=torch.float16, device=device),
        )
        intermediate_scales = fused_moe.ScaleFactors(
            torch.ones((3, intermediate), dtype=torch.float16, device=device),
            torch.ones(experts, dtype=torch.float16, device=device),
        )
        output_scales = fused_moe.ScaleFactors(
            torch.ones(hidden, dtype=torch.float16, device=device),
            torch.ones(1, dtype=torch.float16, device=device),
        )
        draws = torch.zeros(experts, dtype=torch.uint8, device=device)
    elif variant == "glm-mcg-projection-tiered":
        bit_rates = [
            [3 + (expert + projection) % 3 for projection in range(3)]
            for expert in range(experts)
        ]
        rate = torch.tensor(
            [[bits | (bits << 4) for bits in projections] for projections in bit_rates],
            dtype=torch.uint8,
            device=device,
        )
        row_bytes = (hidden // 16) * 64 * sum(map(sum, bit_rates))
        input_scales = fused_moe.ScaleFactors(
            torch.ones(hidden, dtype=torch.float16, device=device)
        )
        intermediate_scales = fused_moe.ScaleFactors(
            torch.ones(
                (experts, 3, intermediate),
                dtype=torch.float16,
                device=device,
            )
        )
        output_scales = fused_moe.ScaleFactors(
            torch.ones(hidden, dtype=torch.float16, device=device)
        )
        draws = None
    else:
        raise AssertionError(f"unhandled Trellis benchmark variant {variant!r}")
    generator = torch.Generator(device=device).manual_seed(
        20260828
        + experts * 17
        + hidden * 31
        + intermediate * 43
        + (1 if geometry.activation == "situ" else 0)
    )
    atoms = torch.randint(
        0,
        256,
        (intermediate // 32, row_bytes),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    return fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.TrellisWeights(
            atoms=atoms,
            rate=rate,
            input_scales=input_scales,
            intermediate_scales=intermediate_scales,
            output_scales=output_scales,
            expert_transform_draws=draws,
        ),
    )


def _prepare_experts(
    geometry: MoePhysicalGeometry,
    *,
    device: object,
):
    if geometry.recipe.source_format in {"btx", "b12x_trellis"}:
        return _trellis_weights(geometry, device=device)
    return _packed_weights(geometry, device=device)


def _routing(
    case: MoeSweepCase,
    *,
    device: object,
    seed: int,
):
    import torch

    tokens = case.num_tokens
    top_k = case.top_k
    experts = case.geometry.num_experts
    rows = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
    columns = torch.arange(top_k, dtype=torch.int64, device=device)[None, :]
    if case.route_pattern == "hot":
        ids = columns.expand(tokens, -1)
    elif case.route_pattern == "balanced":
        stride = max(1, experts // top_k)
        ids = (rows + columns * stride) % experts
    elif case.route_pattern == "disjoint":
        ids = (rows * top_k + columns) % experts
    elif case.route_pattern == "zipf":
        pool = min(experts, max(top_k, int(math.sqrt(tokens * top_k)) * top_k))
        ids = (columns + (rows % max(1, pool // top_k)) * top_k) % pool
    else:
        raise ValueError(f"unsupported MoE route pattern {case.route_pattern!r}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(tokens, top_k, generator=generator, dtype=torch.float32)
    weights = torch.softmax(logits, dim=-1).to(device=device)
    return ids.to(torch.int32).contiguous(), weights.contiguous()


def _measurement_seed(case: MoeSweepCase, *, base_seed: int) -> int:
    key = (
        *case.geometry.key,
        case.top_k,
        case.num_tokens,
    )
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def _cosine_similarity(left, right) -> float:
    import torch

    left = left.float().flatten()
    right = right.float().flatten()
    left_scale = torch.amax(torch.abs(left))
    right_scale = torch.amax(torch.abs(right))
    if float(left_scale) == 0.0 or float(right_scale) == 0.0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float(
        torch.nn.functional.cosine_similarity(
            left / left_scale,
            right / right_scale,
            dim=0,
        )
    )


def _stable_l2_norm(value) -> float:
    import torch

    flattened = value.float().flatten()
    scale = float(torch.amax(torch.abs(flattened)))
    if scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return math.inf
    return scale * float(torch.linalg.vector_norm(flattened / scale))


def _relative_norm_error(value, reference) -> float:
    value_norm = _stable_l2_norm(value)
    reference_norm = _stable_l2_norm(reference)
    if reference_norm == 0.0:
        return 0.0 if value_norm == 0.0 else math.inf
    return abs(value_norm - reference_norm) / reference_norm


def _uniform_w4a16_reference(
    geometry: MoePhysicalGeometry,
    *,
    x: object,
    topk_weights: object,
):
    """Independent oracle for the tuner's constant non-Trellis W4A16 weights."""

    import torch

    from b12x.moe._shared.kernels.activations import is_gated_moe_activation
    from b12x.moe._shared.kernels.reference import (
        _apply_gated_activation,
        _normalize_reference_swiglu_params,
    )

    activation, swiglu_limit, swiglu_alpha, swiglu_beta = (
        _normalize_reference_swiglu_params(
            geometry.activation,
            None,
            None,
            None,
        )
    )
    fc1 = x.float().sum(dim=-1) * 0.5
    if is_gated_moe_activation(activation):
        intermediate = _apply_gated_activation(
            fc1,
            fc1,
            activation=activation,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
    else:
        intermediate = torch.square(torch.relu(fc1))
    down = intermediate * (0.5 * geometry.intermediate_size)
    routed = down * topk_weights.float().sum(dim=-1)
    return routed[:, None].expand(-1, geometry.hidden_size).contiguous()


def _finite_float_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _verify_cooperative_workspace(binding: object) -> None:
    import torch

    buffers = getattr(binding, "mixed_trellis_buffers", None)
    if buffers is None:
        return
    barrier = buffers.workspace[-2:]
    if int(torch.count_nonzero(barrier).item()) != 0:
        raise RuntimeError("MoE cooperative grid barrier was not initialized")


def _candidate_reference_output(
    eager_output: object,
    *,
    correct: bool,
    has_reference: bool,
) -> object | None:
    return eager_output if correct and not has_reference else None


def _is_fatal_accelerator_error(exc: Exception) -> bool:
    import torch

    accelerator_error = getattr(torch, "AcceleratorError", None)
    return accelerator_error is not None and isinstance(exc, accelerator_error)


def _median_of_group_medians(
    samples: tuple[float, ...],
    *,
    groups: int,
    repetitions: int,
) -> float:
    medians = [
        statistics.median(samples[start : start + repetitions])
        for start in range(0, groups * repetitions, repetitions)
    ]
    return float(statistics.median(medians))


def _bounded_repetitions(settings, *, pilot_us: float) -> int:
    budget_us = float(settings.max_candidate_seconds) * 1_000_000.0
    budgeted = int(budget_us / (max(float(pilot_us), 1.0) * settings.groups))
    return max(1, min(settings.repetitions, budgeted))


def _cuda_event_samples_us(
    run,
    *,
    count: int,
    device: object,
    flush=None,
) -> tuple[float, ...]:
    import torch

    starts = tuple(torch.cuda.Event(enable_timing=True) for _ in range(count))
    ends = tuple(torch.cuda.Event(enable_timing=True) for _ in range(count))
    for start, end in zip(starts, ends, strict=True):
        if flush is not None:
            flush()
        start.record()
        run()
        end.record()
    torch.cuda.synchronize(device)
    return tuple(
        float(start.elapsed_time(end)) * 1_000.0
        for start, end in zip(starts, ends, strict=True)
    )


def _l2_flush_fn(device: object, *, enabled: bool):
    if not enabled:
        return None
    import torch

    properties = torch.cuda.get_device_properties(device)
    flush_bytes = max(2 * int(properties.L2_cache_size), 64 << 20)
    values = torch.ones(
        (flush_bytes + 3) // 4,
        dtype=torch.float32,
        device=device,
    )
    reduction = torch.empty((), dtype=torch.float32, device=device)

    def flush() -> None:
        torch.sum(values, dim=0, out=reduction)

    return flush


@dataclass(frozen=True, kw_only=True)
class _CandidateResult:
    measurement: MoeMeasurement
    reference_output: object | None


@dataclass(frozen=True, kw_only=True)
class _QueryInputs:
    x: object
    topk_ids: object
    topk_weights: object


@dataclass(frozen=True, kw_only=True)
class _PreparedCandidate:
    run: Callable[[], None]
    graph: object
    output: object
    actual_implementation: str
    concrete_path: str
    route_planner: str
    max_active_clusters: int | None
    dynamic_tile_m: int | None
    dynamic_route_mode: str | None
    w4a16_route_mode: str | None
    owners: tuple[object, ...]


class _CandidateContractError(RuntimeError):
    pass


def _reset_cuda_graphs(
    prepared_candidates: tuple[_PreparedCandidate, ...],
) -> None:
    for prepared in prepared_candidates:
        prepared.graph.reset()


def _candidates_for_geometry(
    geometry: MoePhysicalGeometry,
    *,
    sm_count: int,
) -> tuple[MoeCandidate, ...]:
    recipe = geometry.recipe
    if recipe.quant_mode == "nvfp4_auto":
        from dataclasses import replace
        return tuple(
            candidate
            for mode in ("nvfp4", "w4a16")
            for candidate in _candidates_for_geometry(
                replace(geometry, recipe=replace(recipe, quant_mode=mode)), sm_count=sm_count,
            )
        )
    if recipe.quant_mode == "w4a16":
        route_modes = (
            ("packed",)
            if _uses_projection_mixed_trellis(geometry)
            else ("packed", "direct")
        )
        return tuple(
            MoeCandidate.create(
                {
                    "backend": "w4a16",
                    "dynamic_route_mode": None,
                    "dynamic_tile_m": None,
                    "route_planner": "internal",
                    "max_active_clusters": None,
                    "w4a16_route_mode": route_mode,
                }
            )
            for route_mode in route_modes
        )
    if recipe.quant_mode == "w6a8_mx":
        dynamic_tile_ms = (128,)
    elif recipe.quant_mode == "nvfp4" and geometry.activation == "relu2":
        # The M16/M32 Relu2 kernels require 117760/121856 bytes of shared
        # memory, above the 101376-byte SM120/SM121 opt-in limit.
        dynamic_tile_ms = (64, 128)
    else:
        dynamic_tile_ms = (16, 32, 64, 128)
    candidates = []
    if recipe.quant_mode != "w6a8_mx":
        candidates.append(
            MoeCandidate.create(
                {
                    "backend": "micro",
                    "dynamic_route_mode": None,
                    "dynamic_tile_m": None,
                    "route_planner": "internal",
                    "max_active_clusters": None,
                    "w4a16_route_mode": None,
                }
            )
        )
    dynamic_route_modes = (
        ("grouped", "direct")
        if recipe.quant_mode in {"nvfp4", "w4a8_mx"}
        else ("grouped",)
    )
    candidates.extend(
        MoeCandidate.create(
            {
                "backend": "dynamic",
                "dynamic_route_mode": route_mode,
                "dynamic_tile_m": tile_m,
                "route_planner": "internal",
                "max_active_clusters": None,
                "w4a16_route_mode": None,
            }
        )
        for tile_m in dynamic_tile_ms
        for route_mode in dynamic_route_modes
    )
    if recipe.quant_mode == "nvfp4" and geometry.activation == "silu":
        cluster_caps = {
            max(1, sm_count // 4),
            max(1, sm_count // 2),
            max(1, 3 * sm_count // 4),
            sm_count,
        }
        candidates.extend(
            MoeCandidate.create(
                {
                    "backend": "dynamic",
                    "dynamic_route_mode": "grouped",
                    "dynamic_tile_m": tile_m,
                    "route_planner": "triton",
                    "max_active_clusters": cluster_cap,
                    "w4a16_route_mode": None,
                }
            )
            for tile_m in (16,)
            for cluster_cap in sorted(cluster_caps)
        )
    return tuple(candidates)


def _w4a16_weight_layout(geometry: MoePhysicalGeometry) -> str:
    from b12x.moe.fused_moe import _impl

    if geometry.recipe.recipe_id == "modelopt-nvfp4-auto":
        return "modelopt"
    return _impl._w4a16_weight_layout_for_source(
        geometry.recipe.source_format,
        intermediate_size=geometry.intermediate_size,
    )


def _uses_projection_mixed_trellis(geometry: MoePhysicalGeometry) -> bool:
    variant = geometry.recipe.trellis_variant
    return variant is not None and _trellis_config(variant=variant).projection_mixed


def _w4a16_prepared_weight_layout(geometry: MoePhysicalGeometry) -> str:
    if _uses_projection_mixed_trellis(geometry):
        return "trellis_mixed3"
    return _w4a16_weight_layout(geometry)


def _w4a16_direct_path(
    geometry: MoePhysicalGeometry,
    case: MoeSweepCase,
) -> str | None:
    from b12x.moe.fused_moe import _impl
    from b12x.moe._shared.kernels.activations import is_gated_moe_activation
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _MAX_DIRECT_TOPK_ROUTE_M,
        _TC_DECODE_MAX_M,
    )

    if _uses_projection_mixed_trellis(geometry):
        return None
    weight_layout = _w4a16_weight_layout(geometry)
    query = _impl.MoeDecodeQuery(
        quant_mode=("nvfp4_auto" if geometry.recipe.recipe_id == "modelopt-nvfp4-auto"
                    else geometry.recipe.quant_mode),
        source_format=geometry.recipe.source_format,
        activation=geometry.activation,
        num_experts=geometry.num_experts,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        top_k=case.top_k,
        num_tokens=case.num_tokens,
        routed_rows=case.routed_rows,
    )
    if not _impl._w4a16_direct_routing_supported(query):
        return None
    if weight_layout == "modelopt":
        return "w4a16.small_m_direct" if case.num_tokens <= 8 else None
    if weight_layout != "packed":
        return None
    if (
        case.num_tokens <= _TC_DECODE_MAX_M
        and is_gated_moe_activation(geometry.activation)
    ):
        return "w4a16.tc_decode"
    if case.num_tokens <= _MAX_DIRECT_TOPK_ROUTE_M:
        return "w4a16.direct_topk"
    return None


def _eligible_candidates_for_case(
    geometry: MoePhysicalGeometry,
    case: MoeSweepCase,
    candidates: tuple[MoeCandidate, ...],
) -> tuple[MoeCandidate, ...]:
    from b12x.moe.fused_moe import _impl

    if geometry.recipe.quant_mode == "nvfp4_auto":
        return tuple(
            candidate for candidate in candidates
            if _eligible_candidates_for_case(
                _precision_candidate_geometry(geometry, candidate), case, (candidate,),
            )
        )
    eligible = []
    for candidate in candidates:
        if candidate.config["backend"] == "dynamic":
            dynamic_n = _impl._dynamic_kernel_intermediate_size(
                geometry.intermediate_size,
                geometry.recipe.quant_mode,
            )
            try:
                _impl._select_dynamic_tile_mn(
                    case.routed_rows,
                    dynamic_n,
                    geometry.recipe.quant_mode,
                    num_experts=geometry.num_experts,
                    activation=geometry.activation,
                    planned_tile_m=candidate.config["dynamic_tile_m"],
                )
            except ValueError:
                continue
        if candidate.config["route_planner"] == "triton":
            if not _impl._dynamic_external_route_plan_supported(
                quant_mode=geometry.recipe.quant_mode,
                activation=geometry.activation,
                routed_rows=case.routed_rows,
                planned_tile_m=candidate.config["dynamic_tile_m"],
                dynamic_route_mode=candidate.config["dynamic_route_mode"],
                deterministic_output=False,
            ):
                continue
        if candidate.config["backend"] == "micro" and not _impl._policy_micro_supported(
            _impl.MoeDecodeQuery(
                quant_mode=geometry.recipe.quant_mode,
                source_format=_impl._canonical_moe_policy_source_format(
                    geometry.recipe.source_format
                ),
                activation=geometry.activation,
                num_experts=geometry.num_experts,
                hidden_size=geometry.hidden_size,
                intermediate_size=geometry.intermediate_size,
                top_k=case.top_k,
                num_tokens=case.num_tokens,
                routed_rows=case.routed_rows,
            )
        ):
            continue
        if (
            candidate.config["backend"] == "w4a16"
            and candidate.config["w4a16_route_mode"] == "direct"
            and _w4a16_direct_path(geometry, case) is None
        ):
            continue
        if (
            candidate.config["backend"] == "dynamic"
            and candidate.config["dynamic_route_mode"] == "direct"
        ):
            dynamic_n = _impl._dynamic_kernel_intermediate_size(
                geometry.intermediate_size,
                geometry.recipe.quant_mode,
            )
            try:
                _impl._dynamic_direct_routing_selected(
                    route_mode="direct",
                    quant_mode=geometry.recipe.quant_mode,
                    activation=geometry.activation,
                    routed_rows=case.routed_rows,
                    num_experts=geometry.num_experts,
                    n=dynamic_n,
                    deterministic_output=False,
                    planned_tile_m=candidate.config["dynamic_tile_m"],
                )
            except RuntimeError:
                continue
        eligible.append(candidate)
    return tuple(eligible)


def _precision_candidate_geometry(geometry, candidate):
    from dataclasses import replace
    if geometry.recipe.quant_mode != "nvfp4_auto":
        return geometry
    mode = "w4a16" if candidate.config["backend"] == "w4a16" else "nvfp4"
    return replace(geometry, recipe=replace(geometry.recipe, quant_mode=mode))


def _concrete_candidate_path(
    *,
    geometry: MoePhysicalGeometry,
    case: MoeSweepCase,
    candidate: MoeCandidate,
    plan: object,
    binding: object,
    prepared_payload: object,
) -> str:
    from b12x.moe.fused_moe import _impl

    config = candidate.config
    expected_implementation = str(config["backend"])
    variant = plan.variant_for(case.num_tokens)
    actual_plan = variant._impl
    actual_implementation = str(variant.implementation)
    if actual_implementation != expected_implementation:
        raise _CandidateContractError(
            f"candidate {expected_implementation!r} resolved to "
            f"{actual_implementation!r}"
        )
    bound_implementation = str(binding.implementation)
    resolution = actual_plan.policy_resolution
    if resolution is None:
        raise _CandidateContractError("candidate plan has no policy resolution")
    actual_config = resolution.config
    for field in ("route_planner", "max_active_clusters"):
        if getattr(actual_config, field) != config[field]:
            raise _CandidateContractError(
                f"candidate {field}={config[field]!r} resolved to "
                f"{getattr(actual_config, field)!r}"
            )

    if expected_implementation == "w4a16":
        route_mode = str(config["w4a16_route_mode"])
        actual_route_mode = actual_config.w4a16_route_mode
        if actual_route_mode != route_mode:
            raise _CandidateContractError(
                f"W4A16 candidate route mode {route_mode!r} resolved to "
                f"{actual_route_mode!r}"
            )
        expected_layout = _w4a16_prepared_weight_layout(geometry)
        actual_layout = getattr(prepared_payload, "weight_layout", None)
        if actual_layout != expected_layout:
            raise _CandidateContractError(
                f"W4A16 candidate expected {expected_layout!r} weights but "
                f"prepared {actual_layout!r}"
            )
        if bound_implementation == "trellis_mixed3":
            if actual_layout != "trellis_mixed3" or route_mode != "packed":
                raise _CandidateContractError(
                    "projection-mixed Trellis requires packed Trellis routing"
                )
            return "w4a16.trellis_mixed3"
        if bound_implementation != expected_implementation:
            raise _CandidateContractError(
                f"W4A16 candidate bound {bound_implementation!r}"
            )
        if route_mode == "packed":
            return "w4a16.route_packed"
        expected_path = _w4a16_direct_path(geometry, case)
        if expected_path is None:
            raise _CandidateContractError(
                "W4A16 direct candidate reached an unsupported route shape"
            )
        return expected_path

    if bound_implementation != expected_implementation:
        raise _CandidateContractError(
            f"candidate {expected_implementation!r} bound "
            f"{bound_implementation!r}"
        )

    if expected_implementation == "micro":
        quant_mode = geometry.recipe.quant_mode
        if quant_mode == "w4a8_mx":
            if getattr(prepared_payload, "weight_layout", None) == "trellis_t256":
                return "w4a8.trellis_micro"
            return "w4a8.tiny_decode"
        if quant_mode == "w4a8_nvfp4":
            return "w4a8.direct_micro"
        return "nvfp4.direct_micro"

    planned_tile_m = variant.execution.tile_m
    if planned_tile_m != config["dynamic_tile_m"]:
        raise _CandidateContractError(
            f"dynamic candidate tile M={config['dynamic_tile_m']} resolved to "
            f"M={planned_tile_m}"
        )
    planned_route_mode = actual_config.dynamic_route_mode
    if planned_route_mode != config["dynamic_route_mode"]:
        raise _CandidateContractError(
            f"dynamic candidate route mode {config['dynamic_route_mode']!r} "
            f"resolved to {planned_route_mode!r}"
        )
    if actual_config.route_planner == "triton":
        if not _impl._dynamic_external_route_plan_supported(
            quant_mode=geometry.recipe.quant_mode,
            activation=geometry.activation,
            routed_rows=case.routed_rows,
            planned_tile_m=planned_tile_m,
            dynamic_route_mode=planned_route_mode,
            deterministic_output=actual_plan.deterministic_output,
        ):
            raise _CandidateContractError(
                "Triton route-planner candidate resolved to an internal route plan"
            )
    quant_mode = geometry.recipe.quant_mode
    if quant_mode in {"w4a8_mx", "w4a8_nvfp4"}:
        if planned_route_mode == "direct":
            specialization = "direct"
        elif _impl._w4a8_dynamic_decode_candidate(
            quant_mode=quant_mode,
            activation=geometry.activation,
            routed_rows=case.routed_rows,
            num_experts=geometry.num_experts,
            n=geometry.intermediate_size,
            deterministic_output=actual_plan.deterministic_output,
            planned_tile_m=planned_tile_m,
        ):
            specialization = "decode"
        elif _impl._w4a8_dynamic_dense_candidate(
            quant_mode=quant_mode,
            activation=geometry.activation,
            routed_rows=case.routed_rows,
            num_experts=geometry.num_experts,
            k=geometry.hidden_size,
            n=geometry.intermediate_size,
            deterministic_output=actual_plan.deterministic_output,
            planned_tile_m=planned_tile_m,
        ):
            specialization = "dense"
        else:
            specialization = "grouped"
        return f"w4a8.dynamic.{specialization}.m{planned_tile_m}"
    if quant_mode == "nvfp4":
        specialization = str(planned_route_mode)
        return (
            f"nvfp4.dynamic.{specialization}.m{planned_tile_m}."
            f"route_{actual_config.route_planner}"
        )
    return f"{quant_mode}.dynamic.m{planned_tile_m}"


class _MoeGeometrySession(AbstractContextManager["_MoeGeometrySession"]):
    def __init__(
        self,
        geometry: MoePhysicalGeometry,
        context: GenerationContext,
    ) -> None:
        import torch

        from b12x.moe import fused_moe

        self._geometry = geometry
        self._context = context
        self._device = torch.device("cuda", context.device_ordinal)
        with torch.cuda.device(context.device_ordinal):
            fused_moe.clear_caches()
            self._experts = _prepare_experts(geometry, device=self._device)
            torch.cuda.synchronize(self._device)
            gc.collect()
            torch.cuda.empty_cache()
        self._candidates = _candidates_for_geometry(
            geometry,
            sm_count=context.device.sm_count,
        )
        self._query_key: tuple[int, int] | None = None
        self._query_inputs: _QueryInputs | None = None
        self._prepared_candidates: dict[str, _PreparedCandidate] = {}
        self._candidate_signatures: dict[tuple[object, ...], str] = {}
        self._flush = _l2_flush_fn(
            self._device,
            enabled=context.settings.cold_l2,
        )

    def __enter__(self) -> "_MoeGeometrySession":
        return self

    def __exit__(
        self,
        _exc_type: object,
        exc: object,
        _traceback: object,
    ) -> None:
        import torch

        from b12x.moe import fused_moe

        if isinstance(exc, Exception) and _is_fatal_accelerator_error(exc):
            self._prepared_candidates.clear()
            self._candidate_signatures.clear()
            self._query_inputs = None
            self._query_key = None
            self._flush = None
            self._experts = None
            return None
        self._clear_query_state()
        self._flush = None
        self._experts = None
        fused_moe.clear_caches()
        gc.collect()
        torch.cuda.synchronize(self._device)
        torch.cuda.empty_cache()
        return None

    def _clear_query_state(self) -> None:
        import torch

        torch.cuda.synchronize(self._device)
        _reset_cuda_graphs(tuple(self._prepared_candidates.values()))
        self._prepared_candidates.clear()
        self._candidate_signatures.clear()
        self._query_inputs = None
        self._query_key = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def candidates(self) -> tuple[MoeCandidate, ...]:
        return self._candidates

    def eligible_candidates(
        self,
        case: MoeSweepCase,
        candidates: tuple[MoeCandidate, ...],
    ) -> tuple[MoeCandidate, ...]:
        return _eligible_candidates_for_case(
            self._geometry,
            case,
            candidates,
        )

    def _stage_query_inputs(self, case: MoeSweepCase) -> _QueryInputs:
        import torch

        query_key = (case.top_k, case.num_tokens)
        measurement_seed = _measurement_seed(
            case,
            base_seed=self._context.settings.seed,
        )
        topk_ids, topk_weights = _routing(
            case,
            device=self._device,
            seed=measurement_seed,
        )
        if query_key == self._query_key:
            if self._query_inputs is None:
                raise RuntimeError("MoE query input cache is inconsistent")
            self._query_inputs.topk_ids.copy_(topk_ids)
            self._query_inputs.topk_weights.copy_(topk_weights)
            return self._query_inputs

        self._clear_query_state()
        generator = torch.Generator(device="cpu").manual_seed(measurement_seed)
        x = torch.randn(
            (case.num_tokens, self._geometry.hidden_size),
            dtype=torch.float32,
            generator=generator,
        )
        x = _condition_benchmark_inputs(self._geometry, x).to(
            device=self._device,
            dtype=self._experts.plan.activation.io_dtype,
        )
        input_scale = _benchmark_input_scale(self._geometry)
        if input_scale != 1.0:
            x.mul_(input_scale)
        self._query_key = query_key
        self._query_inputs = _QueryInputs(
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        return self._query_inputs

    def _prepare_candidate(
        self,
        *,
        case: MoeSweepCase,
        candidate: MoeCandidate,
        inputs: _QueryInputs,
    ) -> _PreparedCandidate:
        import torch

        from b12x.moe import fused_moe
        from b12x.policy import MOE_DECODE, get_auto_policy

        cached = self._prepared_candidates.get(candidate.candidate_id)
        if cached is not None:
            return cached
        candidate_config = fused_moe.MoeDecodeConfig.from_profile(candidate.config)
        policy = get_auto_policy(self._device).with_override(
            MOE_DECODE,
            candidate_config,
        )
        plan = fused_moe.plan_execution(
            experts=self._experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=case.num_tokens,
                top_k=case.top_k,
            ),
            policy=policy,
        )
        fused_moe.prewarm(plan)
        scratch = {
            spec.name: torch.empty(
                spec.shape,
                dtype=spec.dtype,
                device=spec.device,
            )
            for spec in plan.scratch_specs()
        }
        candidate_geometry = _precision_candidate_geometry(self._geometry, candidate)
        payload = self._experts._impl.representation_for(candidate_geometry.recipe.quant_mode)
        output_dtype = (
            torch.float32
            if getattr(payload, "weight_layout", "") == "trellis_t256"
            else inputs.x.dtype
        )
        output = torch.empty(
            (case.num_tokens, self._geometry.hidden_size),
            dtype=output_dtype,
            device=self._device,
        )
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=inputs.x,
            experts=self._experts,
            topk_weights=inputs.topk_weights,
            topk_ids=inputs.topk_ids,
            output=output,
            input_scales_static=True,
        )
        _verify_cooperative_workspace(binding)
        concrete_path = _concrete_candidate_path(
            geometry=candidate_geometry,
            case=case,
            candidate=candidate,
            plan=plan,
            binding=binding,
            prepared_payload=payload,
        )
        actual_implementation = str(binding.implementation)
        variant = plan.variant_for(case.num_tokens)
        resolution = variant._impl.policy_resolution
        if resolution is None:
            raise _CandidateContractError("candidate plan has no policy resolution")
        resolved_config = resolution.config
        signature = (
            concrete_path,
            resolved_config.route_planner,
            resolved_config.max_active_clusters,
            resolved_config.dynamic_tile_m,
            resolved_config.dynamic_route_mode,
            resolved_config.w4a16_route_mode,
        )
        aliased_candidate = self._candidate_signatures.get(signature)
        if (
            aliased_candidate is not None
            and aliased_candidate != candidate.candidate_id
        ):
            raise _CandidateContractError(
                f"MoE candidates {aliased_candidate} and "
                f"{candidate.candidate_id} resolve to the same concrete "
                f"launch contract {signature!r}"
            )
        self._candidate_signatures[signature] = candidate.candidate_id

        def run() -> None:
            fused_moe.run(binding=binding)

        run()
        torch.cuda.synchronize(self._device)
        graph = torch.cuda.CUDAGraph()
        if self._geometry.recipe.quant_mode == "nvfp4_auto":
            import b12x
            b12x.freeze_kernel_resolution("MoE precision candidate graph capture")
            try:
                with torch.cuda.graph(graph):
                    run()
            finally:
                b12x.unfreeze_kernel_resolution()
        else:
            with torch.cuda.graph(graph):
                run()
        torch.cuda.synchronize(self._device)
        prepared = _PreparedCandidate(
            run=run,
            graph=graph,
            output=output,
            actual_implementation=actual_implementation,
            concrete_path=concrete_path,
            route_planner=resolved_config.route_planner,
            max_active_clusters=resolved_config.max_active_clusters,
            dynamic_tile_m=resolved_config.dynamic_tile_m,
            dynamic_route_mode=resolved_config.dynamic_route_mode,
            w4a16_route_mode=resolved_config.w4a16_route_mode,
            owners=(plan, scratch, binding),
        )
        self._prepared_candidates[candidate.candidate_id] = prepared
        return prepared

    def _measure_candidate(
        self,
        *,
        case: MoeSweepCase,
        candidate: MoeCandidate,
        inputs: _QueryInputs,
        reference_output: object | None,
        reference_kind: str | None,
    ) -> _CandidateResult:
        import torch

        settings = self._context.settings
        try:
            with _candidate_environment(candidate):
                prepared = self._prepare_candidate(
                    case=case,
                    candidate=candidate,
                    inputs=inputs,
                )
                for _ in range(settings.warmup):
                    prepared.run()
                torch.cuda.synchronize(self._device)
                eager_output = prepared.output.detach().clone()
                prepared.output.fill_(float("nan"))
                prepared.graph.replay()
                torch.cuda.synchronize(self._device)
                comparison = (
                    eager_output if reference_output is None else reference_output
                )
                finite = bool(torch.isfinite(prepared.output).all().item())
                output_nonzero = int(torch.count_nonzero(prepared.output).item())
                cosine = _cosine_similarity(prepared.output, comparison)
                graph_cosine = _cosine_similarity(
                    prepared.output,
                    eager_output,
                )
                relative_norm_error = _relative_norm_error(
                    prepared.output,
                    comparison,
                )
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                prepared.graph.replay()
                end.record()
                end.synchronize()
                timed_repetitions = _bounded_repetitions(
                    settings,
                    pilot_us=float(start.elapsed_time(end)) * 1_000.0,
                )
                allocated_before = torch.cuda.memory_allocated(self._device)
                samples_us = _cuda_event_samples_us(
                    prepared.graph.replay,
                    count=settings.groups * timed_repetitions,
                    device=self._device,
                    flush=self._flush,
                )
                allocated_after = torch.cuda.memory_allocated(self._device)
                correct = (
                    finite
                    and cosine >= settings.minimum_cosine
                    and graph_cosine >= settings.minimum_cosine
                    and relative_norm_error
                    <= _maximum_relative_norm_error(self._geometry)
                    and allocated_after <= allocated_before
                )
                graph_cosine_metric = (
                    graph_cosine if math.isfinite(graph_cosine) else None
                )
                cross_cosine_metric = cosine if math.isfinite(cosine) else None
                return _CandidateResult(
                    measurement=MoeMeasurement(
                        candidate=candidate,
                        latency_us=_median_of_group_medians(
                            tuple(samples_us),
                            groups=settings.groups,
                            repetitions=timed_repetitions,
                        ),
                        cosine=(cosine if correct else None),
                        error=(
                            None if correct else "graph replay correctness gate failed"
                        ),
                        metrics={
                            "allocation_delta_bytes": (
                                allocated_after - allocated_before
                            ),
                            "comparison": (
                                "eager_self"
                                if reference_output is None
                                else reference_kind or "cross_candidate"
                            ),
                            "cross_candidate_cosine": cross_cosine_metric,
                            "finite": finite,
                            "graph_cosine": graph_cosine_metric,
                            "implementation": prepared.actual_implementation,
                            "concrete_path": prepared.concrete_path,
                            "dynamic_tile_m": prepared.dynamic_tile_m,
                            "dynamic_route_mode": prepared.dynamic_route_mode,
                            "input_scale": _benchmark_input_scale(self._geometry),
                            "route_planner": prepared.route_planner,
                            "max_active_clusters": (prepared.max_active_clusters),
                            "relative_norm_error": _finite_float_or_none(
                                relative_norm_error
                            ),
                            "w4a16_route_mode": prepared.w4a16_route_mode,
                            "output_nonzero": output_nonzero,
                            "output_norm": _finite_float_or_none(
                                _stable_l2_norm(prepared.output)
                            ),
                        },
                    ),
                    reference_output=_candidate_reference_output(
                        eager_output,
                        correct=correct,
                        has_reference=reference_output is not None,
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - one candidate may fail closed
            if isinstance(exc, _CandidateContractError):
                raise
            if _is_fatal_accelerator_error(exc):
                raise
            return _CandidateResult(
                measurement=MoeMeasurement(
                    candidate=candidate,
                    latency_us=None,
                    cosine=None,
                    error=f"{type(exc).__name__}: {exc}",
                    metrics={"exception_type": type(exc).__name__},
                ),
                reference_output=None,
            )

    def _independent_reference(
        self,
        *,
        x: object,
        topk_ids: object,
        topk_weights: object,
    ):
        recipe = self._geometry.recipe
        weights = self._experts._impl
        if (
            recipe.quant_mode == "w4a16"
            and recipe.source_format not in {"btx", "b12x_trellis"}
        ):
            return _uniform_w4a16_reference(
                self._geometry,
                x=x,
                topk_weights=topk_weights,
            )
        if recipe.quant_mode == "w4a8_nvfp4":
            import torch

            from b12x.moe.fused_moe import _impl
            from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

            w1_rows = int(weights.w1_fp4.shape[1])
            w1_mx, w1_residual = _impl._derive_w4a8_weight_grids(
                weights.w1_blockscale.view(torch.uint8),
                w1_rows,
                self._geometry.hidden_size,
            )
            w2_mx, w2_residual = _impl._derive_w4a8_weight_grids(
                weights.w2_blockscale.view(torch.uint8),
                self._geometry.hidden_size,
                self._geometry.intermediate_size,
            )
            normalized_key = (
                weights.w1_fp4.data_ptr(),
                weights.w1_blockscale.data_ptr(),
            )
            w13_layout = (
                "w13"
                if normalized_key in _impl._W13_NORMALIZED_STORAGES
                else weights.w13_layout
            )
            return moe_reference_w4a8_mx(
                x.float(),
                weights.w1_fp4,
                w1_mx,
                w1_residual.view(torch.float8_e4m3fn),
                weights.w1_alphas,
                weights.w2_fp4,
                w2_mx,
                w2_residual.view(torch.float8_e4m3fn),
                weights.w2_alphas,
                topk_ids,
                topk_weights,
                self._geometry.num_experts,
                self._geometry.hidden_size,
                self._geometry.intermediate_size,
                activation=self._geometry.activation,
                w13_layout=w13_layout,
            )
        if not (
            recipe.quant_mode == "nvfp4" and recipe.source_format == "modelopt_nvfp4"
        ):
            return None
        from b12x.moe._shared.kernels.reference import moe_reference_nvfp4

        return moe_reference_nvfp4(
            x,
            weights.w1_fp4,
            weights.w1_blockscale,
            weights.w1_alphas,
            weights.w2_fp4,
            weights.w2_blockscale,
            weights.w2_alphas,
            weights.a1_gscale,
            weights.a2_gscale,
            topk_ids,
            topk_weights,
            self._geometry.num_experts,
            self._geometry.hidden_size,
            self._geometry.intermediate_size,
            activation=self._geometry.activation,
            quant_scale_math="reciprocal_multiply",
        )

    def measure(
        self,
        case: MoeSweepCase,
        candidates: tuple[MoeCandidate, ...],
        *,
        correctness: bool = False,
    ) -> tuple[MoeMeasurement, ...]:
        if self._geometry.recipe.quant_mode == "nvfp4_auto":
            from .moe_precision import measure_precision
            return measure_precision(self, case, candidates)
        if any(candidate not in self._candidates for candidate in candidates):
            raise ValueError("MoE worker received an unknown candidate")
        eligible = self.eligible_candidates(case, candidates)
        inputs = self._stage_query_inputs(case)
        measurements_by_id: dict[str, MoeMeasurement] = {}
        reference_output = (
            self._independent_reference(
                x=inputs.x,
                topk_ids=inputs.topk_ids,
                topk_weights=inputs.topk_weights,
            )
            if correctness
            else None
        )
        reference_kind = (
            f"independent_{self._geometry.recipe.quant_mode}"
            if reference_output is not None
            else None
        )
        for candidate in eligible:
            result = self._measure_candidate(
                case=case,
                candidate=candidate,
                inputs=inputs,
                reference_output=reference_output,
                reference_kind=reference_kind,
            )
            measurements_by_id[candidate.candidate_id] = result.measurement
            if reference_output is None and result.reference_output is not None:
                reference_output = result.reference_output
                reference_kind = "cross_candidate"
        return tuple(
            measurements_by_id.get(
                candidate.candidate_id,
                MoeMeasurement(
                    candidate=candidate,
                    latency_us=None,
                    cosine=None,
                    error="candidate is unsupported for this route shape",
                ),
            )
            for candidate in candidates
        )


def _worker_candidates(
    session: _MoeGeometrySession,
    candidate_ids: tuple[str, ...],
) -> tuple[MoeCandidate, ...]:
    by_id = {candidate.candidate_id: candidate for candidate in session.candidates}
    try:
        return tuple(by_id[candidate_id] for candidate_id in candidate_ids)
    except KeyError as exc:
        raise ValueError(f"unknown MoE worker candidate {exc.args[0]!r}") from exc


def _moe_geometry_worker(
    connection: Connection,
    geometry: MoePhysicalGeometry,
    context: GenerationContext,
) -> None:
    try:
        import torch

        torch.cuda.set_device(context.device_ordinal)
        with _MoeGeometrySession(geometry, context) as session:
            connection.send(
                {
                    "ok": True,
                    "candidates": [
                        candidate.config.to_dict() for candidate in session.candidates
                    ],
                }
            )
            while True:
                request = connection.recv()
                operation = request.get("operation")
                if operation == "close":
                    break
                case = request.get("case")
                if not isinstance(case, MoeSweepCase):
                    raise TypeError("MoE worker request is missing a sweep case")
                raw_ids = request.get("candidate_ids")
                if not isinstance(raw_ids, tuple) or not all(
                    isinstance(candidate_id, str) for candidate_id in raw_ids
                ):
                    raise TypeError("MoE worker candidate IDs must be strings")
                candidates = _worker_candidates(session, raw_ids)
                if operation == "eligible":
                    eligible = session.eligible_candidates(case, candidates)
                    connection.send(
                        {
                            "ok": True,
                            "candidate_ids": tuple(
                                candidate.candidate_id for candidate in eligible
                            ),
                        }
                    )
                elif operation == "measure":
                    measurements = session.measure(
                        case,
                        candidates,
                        correctness=bool(request.get("correctness", False)),
                    )
                    connection.send(
                        {
                            "ok": True,
                            "measurements": [
                                measurement.to_dict() for measurement in measurements
                            ],
                        }
                    )
                else:
                    raise ValueError(f"unknown MoE worker operation {operation!r}")
        connection.send({"ok": True, "closed": True})
    except Exception as exc:  # noqa: BLE001 - preserve remote failure evidence
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(
                {
                    "ok": False,
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    finally:
        connection.close()


class _MoeRemoteWorkerError(RuntimeError):
    def __init__(
        self,
        operation: str,
        exception_type: str,
        error: str,
        remote_traceback: str = "",
    ) -> None:
        self.operation = operation
        self.exception_type = exception_type
        self.remote_error = error
        self.remote_traceback = remote_traceback
        super().__init__(
            f"MoE GPU worker failed during {operation}: "
            f"{exception_type}: {error}\n{remote_traceback}"
        )

    def __reduce__(self):
        return (
            type(self),
            (
                self.operation,
                self.exception_type,
                self.remote_error,
                self.remote_traceback,
            ),
        )

    @property
    def retryable(self) -> bool:
        return self.exception_type in {"AcceleratorError", "WorkerExit"}


class _MoeProcessSession(AbstractContextManager["_MoeProcessSession"]):
    def __init__(
        self,
        geometry: MoePhysicalGeometry,
        context: GenerationContext,
    ) -> None:
        self._geometry = geometry
        self._context = context
        self._connection: Connection | None = None
        self._process: multiprocessing.Process | None = None
        self._candidates = _candidates_for_geometry(
            geometry,
            sm_count=context.device.sm_count,
        )

    def _receive(self, *, operation: str) -> dict[str, object]:
        if self._connection is None:
            raise RuntimeError("MoE GPU worker is not running")
        try:
            response = self._connection.recv()
        except EOFError as exc:
            exitcode = None if self._process is None else self._process.exitcode
            raise _MoeRemoteWorkerError(
                operation=operation,
                exception_type="WorkerExit",
                error=f"exitcode={exitcode}",
            ) from exc
        if not isinstance(response, dict):
            raise TypeError("MoE GPU worker response must be an object")
        if response.get("ok") is not True:
            remote_type = response.get("exception_type", "Exception")
            error = response.get("error", "unknown worker failure")
            remote_traceback = response.get("traceback", "")
            raise _MoeRemoteWorkerError(
                operation=operation,
                exception_type=str(remote_type),
                error=str(error),
                remote_traceback=str(remote_traceback),
            )
        return response

    def _request(
        self,
        operation: str,
        **payload: object,
    ) -> dict[str, object]:
        if self._connection is None:
            raise RuntimeError("MoE GPU worker is not running")
        self._connection.send({"operation": operation, **payload})
        return self._receive(operation=operation)

    def _start(self) -> None:
        if self._process is not None:
            return
        process_context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = process_context.Pipe(duplex=True)
        process = process_context.Process(
            target=_moe_geometry_worker,
            args=(child_connection, self._geometry, self._context),
            name=(
                "b12x-moe-profile-"
                f"e{self._geometry.num_experts}-"
                f"k{self._geometry.hidden_size}-"
                f"n{self._geometry.intermediate_size}"
            ),
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process
        try:
            response = self._receive(operation="startup")
            raw_candidates = response.get("candidates")
            if not isinstance(raw_candidates, list):
                raise TypeError("MoE GPU worker startup is missing candidates")
            worker_candidates = tuple(
                MoeCandidate.create(config) for config in raw_candidates
            )
            if worker_candidates != self._candidates:
                raise RuntimeError(
                    "MoE GPU worker candidate enumeration disagrees with host"
                )
        except BaseException:
            parent_connection.close()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
            self._connection = None
            self._process = None
            raise

    def _discard_worker(self) -> None:
        connection = self._connection
        process = self._process
        if connection is not None:
            connection.close()
        if process is not None:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        self._connection = None
        self._process = None

    def __enter__(self) -> "_MoeProcessSession":
        return self

    def __exit__(self, exc_type: object, _exc: object, _tb: object) -> None:
        close_error: Exception | None = None
        process = self._process
        connection = self._connection
        if process is not None and process.is_alive() and connection is not None:
            try:
                self._request("close")
            except Exception as error:  # noqa: BLE001 - child may have failed
                close_error = error
        self._discard_worker()
        if close_error is not None and exc_type is None:
            raise close_error
        return None

    @property
    def candidates(self) -> tuple[MoeCandidate, ...]:
        return self._candidates

    def eligible_candidates(
        self,
        case: MoeSweepCase,
        candidates: tuple[MoeCandidate, ...],
    ) -> tuple[MoeCandidate, ...]:
        if any(candidate not in self._candidates for candidate in candidates):
            raise ValueError("MoE worker received an unknown candidate")
        return _eligible_candidates_for_case(
            self._geometry,
            case,
            candidates,
        )

    def measure(
        self,
        case: MoeSweepCase,
        candidates: tuple[MoeCandidate, ...],
        *,
        correctness: bool = False,
    ) -> tuple[MoeMeasurement, ...]:
        def request_measurement(
            requested: tuple[MoeCandidate, ...],
        ) -> dict[str, object]:
            self._start()
            return self._request(
                "measure",
                case=case,
                candidate_ids=tuple(
                    candidate.candidate_id for candidate in requested
                ),
                correctness=correctness,
            )

        def parse_measurements(
            response: dict[str, object],
        ) -> tuple[MoeMeasurement, ...]:
            raw_measurements = response.get("measurements")
            if not isinstance(raw_measurements, list):
                raise TypeError("MoE GPU worker measurement response is invalid")
            return tuple(
                MoeMeasurement.from_dict(measurement)
                for measurement in raw_measurements
            )

        def failed_measurement(
            candidate: MoeCandidate,
            error: _MoeRemoteWorkerError,
        ) -> MoeMeasurement:
            return MoeMeasurement(
                candidate=candidate,
                latency_us=None,
                cosine=None,
                error=f"{error.exception_type}: {error.remote_error}",
                metrics={
                    "exception_type": error.exception_type,
                    "worker_retries": 1,
                },
            )

        try:
            return parse_measurements(request_measurement(candidates))
        except _MoeRemoteWorkerError as exc:
            if not exc.retryable:
                raise
            self._discard_worker()
            if exc.operation == "startup" or len(candidates) == 1:
                measurements = parse_measurements(request_measurement(candidates))
                return tuple(
                    replace(
                        measurement,
                        metrics={
                            **measurement.metrics.to_dict(),
                            "worker_retries": 1,
                        },
                    )
                    for measurement in measurements
                )

        measurements = []
        for candidate in candidates:
            try:
                (measurement,) = parse_measurements(
                    request_measurement((candidate,))
                )
            except _MoeRemoteWorkerError as exc:
                if not exc.retryable or exc.operation == "startup":
                    raise
                self._discard_worker()
                measurement = failed_measurement(candidate, exc)
            measurements.append(
                replace(
                    measurement,
                    metrics={
                        **measurement.metrics.to_dict(),
                        "worker_retries": 1,
                    },
                )
            )
        return tuple(measurements)


class MoeGpuBenchmarkFactory:
    """Run every physical expert geometry in a fresh CUDA process."""

    def __call__(
        self,
        geometry: MoePhysicalGeometry,
        context: GenerationContext,
    ) -> _MoeProcessSession:
        return _MoeProcessSession(geometry, context)


__all__ = ["MoeGpuBenchmarkFactory"]
