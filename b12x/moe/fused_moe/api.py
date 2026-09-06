"""Public surface for :mod:`b12x.moe.fused_moe`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import torch

from ..._lib.gating import default_is_supported
from ...policy import PolicyContext
from .._shared.routing import route_topk
from . import META
from . import _vllm_compat as _compat
from ._impl import (
    B12XFP4ExpertWeights as ExpertWeights,
    B12XTopKRouting as Routing,
    TPMoEFP4Binding as Binding,
    TPMoERouteBinding as RouteBinding,
    TPMoESparseFP4Binding as SparseBinding,
    build_tp_moe_route_binding as bind_route,
    build_tp_moe_sparse_fp4_binding as bind_sparse,
    clear_tp_moe_caches as clear_caches,
    prepare_w4a16_fc2_e8m0 as prepare_fc2_weights,
    prewarm_w4a16_fc2_e8m0 as prewarm_fc2,
    b12x_moe_fp4 as run,
    run_w4a16_fc2_e8m0 as run_fc2,
    b12x_route_experts_fast as route,
    b12x_sparse_moe_fp4 as run_sparse,
)
from .config import TrellisConfig
from ._policy import MoeDecodeConfig, MoeDecodeQuery
from .execution import (
    ExecutionCapacity,
    ExecutionPlan,
    ExecutionVariant,
    RoutingSpec,
    ScratchRequirement,
    plan_execution as _plan_execution,
    prewarm,
)
from .planning import (
    ActivationMode,
    ActivationSpec,
    MoEGeometry,
    WeightPlan,
    WeightPlanConstraints,
    plan_weights as _plan_weights,
    prepare_weights as _prepare_weights,
)
from .source import PackedSource, PackedSourceFormat, W13Layout, WeightSource
from .weights import (
    PackedWeights,
    PreparedExperts,
    PreparedWeightFormat,
    ScaleEncoding,
    ScaleFactors,
    TrellisWeights,
    WeightEncoding,
    WeightPacking,
)

Caps = _compat.Caps
Plan = _compat.Plan
WeightsPlan = _compat.WeightsPlan
plan = _compat.plan
required_nbytes = _compat.required_nbytes


def _reject_mixed(
    kwargs: dict[str, Any],
    *,
    canonical: str,
    compatibility: str,
) -> None:
    if canonical in kwargs and compatibility in kwargs:
        raise TypeError(
            f"cannot mix canonical {canonical}= with vLLM compatibility "
            f"{compatibility}="
        )


@overload
def plan_weights(
    *,
    source: WeightSource,
    activation: ActivationSpec,
    geometry: MoEGeometry,
    constraints: WeightPlanConstraints | None = None,
) -> WeightPlan: ...


@overload
def plan_weights(
    *,
    quant_modes: str | Sequence[str],
    source_format: str,
    activation: str,
    params_dtype: torch.dtype,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    w13_layout: str = "w13",
    w4a16_layout: str | None = None,
    trellis_bits: int | None = None,
    trellis_tile_config: tuple[int, int, int, int] | None = None,
    coupled_hadamard: bool | None = None,
    trellis_codebook: str | None = None,
    trellis_rate_granularity: str | None = None,
    trellis_pair_kinds: Sequence[str] | frozenset[str] | None = None,
    coupled_hadamard_blocks: tuple[int, int] | None = None,
) -> WeightsPlan: ...


def plan_weights(**kwargs: Any) -> WeightPlan | WeightsPlan:
    """Dispatch canonical source planning or the official vLLM contract."""

    _reject_mixed(
        kwargs,
        canonical="source",
        compatibility="quant_modes",
    )
    if "config" in kwargs:
        raise TypeError(
            "config= is not part of the fused-MoE API; use source=PackedSource(...)"
        )
    if "source" in kwargs:
        return _plan_weights(**kwargs)
    if "quant_modes" in kwargs:
        return _compat.plan_weights(**kwargs)
    raise TypeError("plan_weights requires source= or quant_modes=")


@overload
def prepare_weights(
    *,
    plan: WeightPlan,
    weights: PackedWeights | TrellisWeights,
) -> PreparedExperts: ...


@overload
def prepare_weights(
    *,
    plan: WeightsPlan,
    params_dtype: torch.dtype,
    w1_fp4: torch.Tensor | None = None,
    w2_fp4: torch.Tensor | None = None,
    w1_global_scale: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
    w1_blockscale: torch.Tensor | None = None,
    w2_blockscale: torch.Tensor | None = None,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    btx_layer: object | None = None,
    btx_device: torch.device | str | None = None,
    dummy_scale: torch.Tensor | None = None,
) -> ExpertWeights: ...


def prepare_weights(**kwargs: Any) -> PreparedExperts | ExpertWeights:
    """Dispatch canonical typed preparation or the official vLLM contract."""

    plan_value = kwargs.get("plan")
    if isinstance(plan_value, WeightPlan):
        legacy = {
            "params_dtype",
            "w1_fp4",
            "w2_fp4",
            "w1_global_scale",
            "w2_global_scale",
            "w1_blockscale",
            "w2_blockscale",
            "a1_gscale",
            "a2_gscale",
            "btx_layer",
            "btx_device",
            "dummy_scale",
        }
        mixed = sorted(legacy.intersection(kwargs))
        if mixed:
            raise TypeError(
                "canonical preparation cannot use vLLM tensor arguments: "
                + ", ".join(mixed)
            )
        return _prepare_weights(**kwargs)
    if "weights" in kwargs:
        raise TypeError("weights= requires a canonical WeightPlan")
    return _compat.prepare_weights(**kwargs)


@overload
def plan_execution(
    *,
    experts: PreparedExperts,
    capacity: ExecutionCapacity,
    routing: RoutingSpec | None = None,
    policy: PolicyContext | None = None,
) -> ExecutionPlan: ...


@overload
def plan_execution(
    *,
    num_tokens: int,
    num_topk: int,
    device: torch.device | str,
    weight_plan: WeightsPlan,
    quant_mode: str,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    apply_router_weight_on_input: bool = False,
    deterministic_output: bool | None = None,
) -> _compat.ExecutionPlan: ...


def plan_execution(**kwargs: Any) -> ExecutionPlan | _compat.ExecutionPlan:
    """Dispatch canonical capacity planning or vLLM's variant query."""

    _reject_mixed(
        kwargs,
        canonical="experts",
        compatibility="weight_plan",
    )
    if "experts" in kwargs:
        return _plan_execution(**kwargs)
    if "weight_plan" in kwargs:
        return _compat.plan_execution(**kwargs)
    raise TypeError("plan_execution requires experts= or weight_plan=")


def bind(plan: Plan | ExecutionPlan, **kwargs: Any) -> Binding:
    """Bind runtime tensors and caller-owned scratch without allocating."""

    if isinstance(plan, ExecutionPlan):
        if not plan.is_prewarmed:
            raise RuntimeError(
                "canonical execution plans must be prewarmed before bind"
            )
        experts = kwargs.get("experts")
        if not isinstance(experts, PreparedExperts):
            raise TypeError("experts must come from canonical prepare_weights")
        if experts.plan is not plan.experts.plan:
            raise ValueError("experts do not match the canonical execution plan")
        if "unit_scale_contract" in kwargs:
            raise TypeError(
                "unit_scale_contract is derived from the canonical activation mode"
            )
        kwargs["unit_scale_contract"] = (
            plan.activation_mode is ActivationMode.A16
        )
        kwargs["experts"] = experts._impl
        return plan._impl.bind(**kwargs)
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """Return whether the active device satisfies the fused-MoE requirements."""

    return default_is_supported(device, requires=META.requires)


__all__ = [
    "ActivationMode",
    "ActivationSpec",
    "Binding",
    "Caps",
    "ExecutionCapacity",
    "ExecutionPlan",
    "ExecutionVariant",
    "ExpertWeights",
    "MoEGeometry",
    "MoeDecodeConfig",
    "MoeDecodeQuery",
    "PackedSource",
    "PackedSourceFormat",
    "PackedWeights",
    "Plan",
    "PreparedExperts",
    "PreparedWeightFormat",
    "RouteBinding",
    "Routing",
    "RoutingSpec",
    "ScaleEncoding",
    "ScaleFactors",
    "ScratchRequirement",
    "SparseBinding",
    "TrellisConfig",
    "TrellisWeights",
    "W13Layout",
    "WeightEncoding",
    "WeightPacking",
    "WeightPlan",
    "WeightPlanConstraints",
    "WeightSource",
    "WeightsPlan",
    "bind",
    "bind_route",
    "bind_sparse",
    "clear_caches",
    "is_supported",
    "plan",
    "plan_execution",
    "plan_weights",
    "prepare_fc2_weights",
    "prepare_weights",
    "prewarm",
    "prewarm_fc2",
    "required_nbytes",
    "route",
    "route_topk",
    "run",
    "run_fc2",
    "run_sparse",
]
