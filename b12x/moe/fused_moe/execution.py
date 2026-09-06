"""Canonical capacity-based fused-MoE execution planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..._lib.scratch import ScratchBufferSpec
from ...policy import MOE_DECODE, PolicyContext, PolicyResolution, get_auto_policy
from .._shared.execution import MoEExecutionPlan
from ._impl import (
    TPMoEPlan,
    TPMoEScratchCaps,
    TPMoEScratchPlan,
    plan_tp_moe_execution,
    plan_tp_moe_scratch,
)
from .weights import PreparedExperts
from .planning import ActivationMode
from ._policy import MOE_DECODE_POLICY, MoeDecodeQuery


@dataclass(frozen=True, kw_only=True)
class ExecutionCapacity:
    """Serving capacity and token counts that must be warm before capture."""

    max_tokens: int
    top_k: int
    warmup_token_counts: tuple[int, ...] = ()
    route_num_experts: int | None = None

    def __post_init__(self) -> None:
        max_tokens = int(self.max_tokens)
        top_k = int(self.top_k)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        counts = tuple(sorted({int(value) for value in self.warmup_token_counts}))
        if any(value <= 0 or value > max_tokens for value in counts):
            raise ValueError(
                "warmup_token_counts must be positive and no larger than max_tokens"
            )
        route_num_experts = self.route_num_experts
        if route_num_experts is not None and int(route_num_experts) < 0:
            raise ValueError("route_num_experts cannot be negative")
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "warmup_token_counts", counts)
        if route_num_experts is not None:
            object.__setattr__(
                self,
                "route_num_experts",
                int(route_num_experts),
            )


@dataclass(frozen=True, kw_only=True)
class RoutingSpec:
    """Routing behavior that affects kernel planning or scratch sizing."""

    apply_router_weight_on_input: bool = False
    logits_dtype: torch.dtype | None = None
    deterministic_output: bool | None = None
    collect_activation_amax: bool = False


@dataclass(frozen=True, kw_only=True)
class ExecutionVariant:
    """One preplanned token-count lowering."""

    tokens: int
    implementation: str
    execution: MoEExecutionPlan
    max_tokens_per_launch: int
    _impl: TPMoEPlan


@dataclass(frozen=True, kw_only=True)
class ScratchRequirement:
    """Caller-owned scratch required by an execution plan."""

    specs: tuple[ScratchBufferSpec, ...]

    @property
    def nbytes(self) -> int:
        return sum(spec.nbytes for spec in self.specs)


class ExecutionPlan:
    """Capacity plan containing scratch and every requested launch variant."""

    def __init__(
        self,
        *,
        experts: PreparedExperts,
        capacity: ExecutionCapacity,
        routing: RoutingSpec,
        caps: TPMoEScratchCaps,
        impl: TPMoEScratchPlan,
        variants: tuple[ExecutionVariant, ...],
        policy: PolicyContext,
        activation_mode: ActivationMode,
        precision_resolution: PolicyResolution | None = None,
    ) -> None:
        self.experts = experts
        self.capacity = capacity
        self.routing = routing
        self._caps = caps
        self._impl = impl
        self.variants = variants
        self.policy = policy
        self.activation_mode = activation_mode
        self.precision_resolution = precision_resolution
        self._prewarmed = False

    @property
    def scratch(self) -> ScratchRequirement:
        return ScratchRequirement(specs=self._impl.scratch_specs())

    @property
    def is_prewarmed(self) -> bool:
        return self._prewarmed

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._impl.scratch_specs()

    def shapes_and_dtypes(
        self,
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return self._impl.shapes_and_dtypes()

    def variant_for(self, tokens: int) -> ExecutionVariant:
        tokens = int(tokens)
        for variant in self.variants:
            if variant.tokens == tokens:
                return variant
        planned = [variant.tokens for variant in self.variants]
        raise ValueError(
            f"token count {tokens} was not preplanned; planned counts are {planned}"
        )


def _variant(
    *,
    tokens: int,
    experts: PreparedExperts,
    capacity: ExecutionCapacity,
    routing: RoutingSpec,
    quant_mode: str,
    policy: PolicyContext,
) -> ExecutionVariant:
    activation = experts.plan.activation
    impl = plan_tp_moe_execution(
        num_tokens=tokens,
        num_topk=capacity.top_k,
        device=experts.device,
        weight_plan=experts.plan._impl,
        quant_mode=quant_mode,
        swiglu_limit=activation.swiglu_limit,
        swiglu_alpha=activation.swiglu_alpha,
        swiglu_beta=activation.swiglu_beta,
        apply_router_weight_on_input=routing.apply_router_weight_on_input,
        deterministic_output=routing.deterministic_output,
        policy_context=policy,
    )
    return ExecutionVariant(
        tokens=tokens,
        implementation=impl.implementation,
        execution=impl.execution,
        max_tokens_per_launch=impl.max_tokens_per_launch,
        _impl=impl,
    )


def plan_execution(
    *,
    experts: PreparedExperts,
    capacity: ExecutionCapacity,
    routing: RoutingSpec | None = None,
    policy: PolicyContext | None = None,
) -> ExecutionPlan:
    """Plan scratch and launch variants without compiling CUDA launches."""

    if not isinstance(experts, PreparedExperts):
        raise TypeError("experts must come from canonical prepare_weights")
    if not isinstance(capacity, ExecutionCapacity):
        raise TypeError("capacity must be an ExecutionCapacity")
    routing = routing or RoutingSpec()
    if not isinstance(routing, RoutingSpec):
        raise TypeError("routing must be a RoutingSpec")
    policy = policy or get_auto_policy(experts.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(experts.device)
    quant_modes = experts.plan._impl.quant_modes
    automatic = experts.plan.activation.mode is ActivationMode.AUTO
    if len(quant_modes) != 1 and not (automatic and quant_modes == {"nvfp4", "w4a16"}):
        raise ValueError("canonical weight plans must resolve one activation mode")
    activation = experts.plan.activation
    precision_resolution = None
    execution_policy = policy
    activation_mode = activation.mode
    if automatic:
        if routing.collect_activation_amax:
            raise ValueError("activation calibration requires an explicit MoE precision")
        precision_resolution = policy.resolve(
            MOE_DECODE_POLICY,
            MoeDecodeQuery(
                quant_mode="nvfp4_auto", source_format="modelopt_nvfp4",
                activation=activation.nonlinearity,
                num_experts=experts.num_experts, hidden_size=experts.hidden_size,
                intermediate_size=experts.intermediate_size,
                top_k=capacity.top_k, num_tokens=capacity.max_tokens,
                routed_rows=capacity.max_tokens * capacity.top_k,
            ),
        )
        activation_mode = (
            ActivationMode.A16 if precision_resolution.config.backend == "w4a16"
            else ActivationMode.A4
        )
        quant_mode = "w4a16" if activation_mode is ActivationMode.A16 else "nvfp4"
        execution_policy = policy.with_override(MOE_DECODE, precision_resolution.config)
    else:
        quant_mode = next(iter(quant_modes))
    counts = tuple(sorted({capacity.max_tokens, *capacity.warmup_token_counts}))
    caps = TPMoEScratchCaps(
        max_tokens=capacity.max_tokens,
        num_topk=capacity.top_k,
        device=experts.device,
        weight_plan=experts.plan._impl,
        quant_mode=quant_mode,
        core_token_counts=counts,
        route_num_experts=capacity.route_num_experts,
        route_logits_dtype=routing.logits_dtype,
        apply_router_weight_on_input=routing.apply_router_weight_on_input,
        swiglu_limit=activation.swiglu_limit,
        swiglu_alpha=activation.swiglu_alpha,
        swiglu_beta=activation.swiglu_beta,
        collect_activation_amax=routing.collect_activation_amax,
        deterministic_output=routing.deterministic_output,
        policy_context=execution_policy,
        frozen=True,
    )
    impl = plan_tp_moe_scratch(caps, prewarm_launches=False)
    variants = tuple(
        _variant(
            tokens=tokens,
            experts=experts,
            capacity=capacity,
            routing=routing,
            quant_mode=quant_mode,
            policy=execution_policy,
        )
        for tokens in counts
    )
    return ExecutionPlan(
        experts=experts,
        capacity=capacity,
        routing=routing,
        caps=caps,
        impl=impl,
        variants=variants,
        policy=policy,
        activation_mode=activation_mode,
        precision_resolution=precision_resolution,
    )


def prewarm(plan: ExecutionPlan) -> None:
    """Compile every launch variant owned by ``plan`` before CUDA capture."""

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    if plan._prewarmed:
        return
    warmed = plan_tp_moe_scratch(plan._caps, prewarm_launches=True)
    if warmed.shapes_and_dtypes() != plan._impl.shapes_and_dtypes():
        raise RuntimeError("prewarming changed the planned scratch contract")
    plan._impl = warmed
    if plan.experts.plan.activation.mode is ActivationMode.AUTO and plan.experts.device.type == "cuda":
        from .api import run

        scratch = {
            spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in plan.scratch_specs()
        }
        for count in plan._caps.core_token_counts:
            x = torch.ones(
                (count, plan.experts.hidden_size), dtype=torch.bfloat16, device=plan.experts.device,
            )
            output = torch.empty_like(x)
            ids = torch.arange(count * plan.capacity.top_k, device=x.device).reshape(
                count, plan.capacity.top_k,
            ) % plan.experts.num_experts
            weights = torch.full(ids.shape, 1.0 / plan.capacity.top_k, device=x.device)
            for dtype in (torch.int32, torch.int64):
                binding = plan._impl.bind(
                    scratch=scratch, experts=plan.experts._impl, a=x, output=output,
                    topk_ids=ids.to(dtype), topk_weights=weights, input_scales_static=True,
                    unit_scale_contract=plan.activation_mode is ActivationMode.A16,
                )
                run(binding=binding)
        torch.cuda.synchronize(plan.experts.device)
    plan._prewarmed = True


__all__ = [
    "ExecutionCapacity",
    "ExecutionPlan",
    "ExecutionVariant",
    "RoutingSpec",
    "ScratchRequirement",
    "plan_execution",
    "prewarm",
]
