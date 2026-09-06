"""Canonical fused-MoE weight planning and preparation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .._shared.execution import MoEWeightPreparationPlan
from ._impl import (
    plan_b12x_fp4_moe_weights,
    prepare_b12x_fp4_moe_weights,
    prepare_b12x_trellis_v2_weights,
)
from .config import TrellisConfig
from .source import PackedSource, WeightSource
from .weights import (
    PackedWeights,
    PreparedExperts,
    PreparedWeightFormat,
    ScaleEncoding,
    TrellisWeights,
    WeightEncoding,
    WeightPacking,
)


class ActivationMode(str, Enum):
    """Numeric activation contract at the fused GEMM boundaries."""

    A16 = "a16"
    A8 = "a8"
    A4 = "a4"
    AUTO = "auto"


@dataclass(frozen=True, kw_only=True)
class ActivationSpec:
    """Activation precision, nonlinearity, and public I/O dtype."""

    mode: ActivationMode
    nonlinearity: str
    io_dtype: torch.dtype
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ActivationMode(self.mode))
        object.__setattr__(self, "nonlinearity", str(self.nonlinearity).lower())
        if self.io_dtype not in {torch.bfloat16, torch.float16}:
            raise TypeError("io_dtype must be torch.bfloat16 or torch.float16")


@dataclass(frozen=True, kw_only=True)
class MoEGeometry:
    """Shape of one tensor-parallel expert shard."""

    num_experts: int
    hidden_size: int
    intermediate_size: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, kw_only=True)
class WeightPlanConstraints:
    """Integration requirements that constrain B12X preparation policy."""

    required_packing: WeightPacking | None = None

    def __post_init__(self) -> None:
        if self.required_packing is not None:
            object.__setattr__(
                self,
                "required_packing",
                WeightPacking(self.required_packing),
            )


@dataclass(frozen=True, kw_only=True)
class WeightPlan:
    """Canonical load-time plan with no public execution-recipe names."""

    source: WeightSource
    activation: ActivationSpec
    geometry: MoEGeometry
    prepared_format: PreparedWeightFormat
    _impl: MoEWeightPreparationPlan


def _packed_recipe(source: PackedSource, mode: ActivationMode) -> str:
    source_format = source.format.value
    if mode is ActivationMode.A16:
        if source_format == "mxfp6_e2m3":
            raise ValueError("MXFP6 weights require A8 activations")
        return "w4a16"
    if mode is ActivationMode.A4:
        if source_format != "modelopt_nvfp4":
            raise ValueError("A4 activations require ModelOpt NVFP4 weights")
        return "nvfp4"
    try:
        return {
            "fp4_e8m0_k32": "w4a8_mx",
            "modelopt_nvfp4": "w4a8_nvfp4",
            "mxfp6_e2m3": "w6a8_mx",
        }[source_format]
    except KeyError as exc:
        raise ValueError(
            f"source format {source_format!r} does not support A8 activations"
        ) from exc


def _validate_trellis_runtime(source: TrellisConfig) -> None:
    if source.codebook.value == "sqg_fp16":
        raise NotImplementedError(
            "sqg_fp16 is defined by the checkpoint schema but is not "
            "implemented by the routed fused MoE runtime"
        )
    if source.rate.group_size is not None:
        raise NotImplementedError(
            "grouped trellis rates are defined by the checkpoint schema but "
            "are not implemented by the fused MoE runtime"
        )
    if source.transform.projection.kind != "scaled_hadamard" or (
        source.transform.projection.block_size != 128
    ):
        raise NotImplementedError(
            "fused MoE trellis execution requires scaled_hadamard(128)"
        )
    expert = source.transform.expert
    if expert.kind == "coupled_hadamard" and (
        source.codebook.value != "sqg_e4m3"
        or source.rate.granularity.value != "uniform"
    ):
        raise NotImplementedError(
            "fused MoE coupled_hadamard execution currently requires the "
            "sqg_e4m3 codebook with uniform rates"
        )
    if expert.kind == "coupled_hadamard" and (
        expert.pre_block_size,
        expert.post_block_size,
    ) != (512, 128):
        raise NotImplementedError(
            "fused MoE coupled_hadamard requires block sizes (512, 128)"
        )


def _prepared_format(
    *,
    source: WeightSource,
    plan: MoEWeightPreparationPlan,
    recipe: str,
    constraints: WeightPlanConstraints,
) -> PreparedWeightFormat:
    available = frozenset(WeightPacking(layout.value) for layout in plan.weight_layouts)
    required = plan.required_weight_layout(recipe)
    default_packing = (
        WeightPacking.SOURCE_NATIVE
        if required is None
        else WeightPacking(required.value)
    )
    packing = constraints.required_packing or default_packing
    if packing not in available:
        raise ValueError(
            f"required packing {packing.value!r} is not available; "
            f"planner produced {sorted(value.value for value in available)}"
        )
    if isinstance(source, TrellisConfig):
        weights = WeightEncoding.TRELLIS
        scales = ScaleEncoding.TRELLIS_SCALES
    else:
        weights = (
            WeightEncoding.FP6_E2M3
            if source.format.value == "mxfp6_e2m3"
            else WeightEncoding.FP4_E2M1
        )
        scales = ScaleEncoding(plan.specs[0].weight_scale.value)
    return PreparedWeightFormat(
        weights=weights,
        scales=scales,
        packing=packing,
        available_packings=available,
    )


def plan_weights(
    *,
    source: WeightSource,
    activation: ActivationSpec,
    geometry: MoEGeometry,
    constraints: WeightPlanConstraints | None = None,
) -> WeightPlan:
    """Plan preparation from independent source, activation, and shape axes."""

    if not isinstance(activation, ActivationSpec):
        raise TypeError("activation must be an ActivationSpec")
    if not isinstance(geometry, MoEGeometry):
        raise TypeError("geometry must be a MoEGeometry")
    constraints = constraints or WeightPlanConstraints()
    if not isinstance(constraints, WeightPlanConstraints):
        raise TypeError("constraints must be WeightPlanConstraints")

    if isinstance(source, PackedSource):
        automatic = activation.mode is ActivationMode.AUTO
        if automatic and (
            source.format.value != "modelopt_nvfp4"
            or activation.io_dtype is not torch.bfloat16
            or activation.nonlinearity != "silu"
        ):
            raise ValueError("automatic MoE precision requires BF16 inputs, SiLU, and ModelOpt NVFP4 weights")
        if automatic and constraints.required_packing not in {None, WeightPacking.SOURCE_NATIVE}:
            raise ValueError("automatic MoE precision requires source-native weight storage")
        if automatic and source.w13_layout.value != "w13":
            raise ValueError("automatic MoE precision requires up/gate W13 row order")
        recipe = _packed_recipe(source, ActivationMode.A4 if automatic else activation.mode)
        requested_layout = None
        if automatic:
            requested_layout = WeightPacking.SOURCE_NATIVE.value
        if recipe == "w4a16" and constraints.required_packing is not None:
            if constraints.required_packing not in {
                WeightPacking.SOURCE_NATIVE,
                WeightPacking.MMA_PACKED,
            }:
                raise ValueError(
                    "A16 preparation requires source_native or mma_packed packing"
                )
            requested_layout = constraints.required_packing.value
        raw_plan = plan_b12x_fp4_moe_weights(
            quant_modes=("nvfp4", "w4a16") if automatic else recipe,
            source_format=source.format.value,
            activation=activation.nonlinearity,
            params_dtype=activation.io_dtype,
            num_experts=geometry.num_experts,
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
            w13_layout=source.w13_layout.value,
            w4a16_layout=requested_layout,
        )
    elif isinstance(source, TrellisConfig):
        if activation.mode is not ActivationMode.A16:
            raise ValueError("Trellis fused MoE currently requires A16 activations")
        if constraints.required_packing not in {
            None,
            WeightPacking.TRELLIS_NATIVE,
        }:
            raise ValueError("Trellis weights require trellis_native packing")
        _validate_trellis_runtime(source)
        expert = source.transform.expert
        recipe = "w4a16"
        raw_plan = plan_b12x_fp4_moe_weights(
            quant_modes=recipe,
            source_format="b12x_trellis",
            activation=activation.nonlinearity,
            params_dtype=activation.io_dtype,
            num_experts=geometry.num_experts,
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
            w13_layout="w31",
            w4a16_layout="trellis_native",
            trellis_bits=3,
            trellis_tile_config=(
                (128, 256, 64, 256)
                if source.codebook.value == "mcg"
                else (64, 256, 64, 256)
            ),
            coupled_hadamard=expert.kind == "coupled_hadamard",
            trellis_codebook=source.codebook.value,
            trellis_rate_granularity=source.rate.granularity.value,
            coupled_hadamard_blocks=(
                None
                if expert.kind == "none"
                else (expert.pre_block_size, expert.post_block_size)
            ),
        )
    else:
        raise TypeError("source must be a PackedSource or TrellisConfig")

    return WeightPlan(
        source=source,
        activation=activation,
        geometry=geometry,
        prepared_format=_prepared_format(
            source=source,
            plan=raw_plan,
            recipe=recipe,
            constraints=constraints,
        ),
        _impl=raw_plan,
    )


def prepare_weights(
    *,
    plan: WeightPlan,
    weights: PackedWeights | TrellisWeights,
) -> PreparedExperts:
    """Materialize the in-memory representation selected by ``plan_weights``."""

    if not isinstance(plan, WeightPlan):
        raise TypeError("plan must be a WeightPlan")
    if isinstance(plan.source, TrellisConfig):
        if not isinstance(weights, TrellisWeights):
            raise TypeError("Trellis preparation requires TrellisWeights")
        prepared = prepare_b12x_trellis_v2_weights(
            plan=plan._impl,
            config=plan.source,
            weights=weights,
        )
    else:
        if not isinstance(weights, PackedWeights):
            raise TypeError("packed preparation requires PackedWeights")
        input_scale = weights.input_scale
        intermediate_scale = weights.intermediate_scale
        if plan.activation.mode is ActivationMode.A16:
            input_scale = torch.ones(
                plan.geometry.num_experts,
                dtype=torch.float32,
                device=weights.w13.device,
            )
            intermediate_scale = input_scale
        elif (input_scale is None) != (intermediate_scale is None):
            raise ValueError(
                "input_scale and intermediate_scale must be supplied together"
            )
        elif input_scale is None:
            if plan.source.format.value == "modelopt_nvfp4":
                raise ValueError(
                    "ModelOpt NVFP4 A4/A8 preparation requires activation scales"
                )
            input_scale = torch.ones(
                plan.geometry.num_experts,
                dtype=torch.float32,
                device=weights.w13.device,
            )
            intermediate_scale = input_scale
        prepared = prepare_b12x_fp4_moe_weights(
            plan=plan._impl,
            params_dtype=plan.activation.io_dtype,
            w1_fp4=weights.w13,
            w2_fp4=weights.w2,
            w1_global_scale=weights.w13_global_scales,
            w2_global_scale=weights.w2_global_scales,
            w1_blockscale=weights.w13_block_scales,
            w2_blockscale=weights.w2_block_scales,
            a1_gscale=input_scale,
            a2_gscale=intermediate_scale,
        )
    return PreparedExperts(plan=plan, _impl=prepared)


__all__ = [
    "ActivationMode",
    "ActivationSpec",
    "MoEGeometry",
    "WeightPlan",
    "WeightPlanConstraints",
    "plan_weights",
    "prepare_weights",
]
