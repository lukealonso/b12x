"""Planning, binding, and validation for HyperConnection primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from b12x._lib.scratch import ScratchBufferSpec
from b12x.policy import PolicyContext, get_auto_policy

from ._policy import (
    HYPERCONNECTION_POLICY,
    HyperConnectionQuery,
)

_MAX_TRITON_REDUCTION_WIDTH = 65_536


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass(frozen=True, kw_only=True)
class HyperConnectionCaps:
    """Token capacity and static multi-stream HyperConnection geometry."""

    device: torch.device | str
    max_tokens: int
    hidden_size: int
    streams: int = 4
    lowrank: int = 320
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = _canonical_device(self.device)
        max_tokens = int(self.max_tokens)
        hidden_size = int(self.hidden_size)
        streams = int(self.streams)
        lowrank = int(self.lowrank)
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if hidden_size <= 0 or hidden_size > _MAX_TRITON_REDUCTION_WIDTH:
            raise ValueError(f"hidden_size must be in [1, 65536], got {hidden_size}")
        if streams <= 0:
            raise ValueError(f"streams must be positive, got {streams}")
        if lowrank <= 0:
            raise ValueError(f"lowrank must be positive, got {lowrank}")
        if self.dtype != torch.bfloat16:
            raise TypeError(
                f"HyperConnection kernels require torch.bfloat16, got {self.dtype}"
            )
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "lowrank", lowrank)


@dataclass(frozen=True)
class HyperConnectionBinding:
    """Caller-owned output capacity with token-limited accessors.

    Mutating custom ops target the capacity tensors directly. The accessors
    expose only the live token prefix to downstream operators.
    """

    plan: "HyperConnectionPlan"
    tokens: int
    normalized_capacity: torch.Tensor
    bottleneck_capacity: torch.Tensor
    block_input_capacity: torch.Tensor

    @property
    def normalized(self) -> torch.Tensor:
        return self.normalized_capacity[: self.tokens]

    @property
    def bottleneck(self) -> torch.Tensor:
        return self.bottleneck_capacity[: self.tokens]

    @property
    def block_input(self) -> torch.Tensor:
        return self.block_input_capacity[: self.tokens]


@dataclass(frozen=True)
class HyperConnectionPlan:
    """HyperConnection launch policy with no anonymous scratch requirement."""

    caps: HyperConnectionCaps
    reduction_block_h: int
    pointwise_block: int
    reduction_num_warps: int
    policy_resolution: object | None = None

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        """HyperConnection primitives use no anonymous scratch allocation."""
        return ()

    def output_shapes(self, tokens: int | None = None) -> dict[str, tuple[int, ...]]:
        live_tokens = self._live_tokens(tokens)
        caps = self.caps
        width = caps.streams * caps.hidden_size
        return {
            "normalized": (live_tokens, width),
            "bottleneck": (live_tokens, caps.lowrank),
            "block_input": (live_tokens, caps.hidden_size),
        }

    def _live_tokens(self, tokens: int | None) -> int:
        live_tokens = self.caps.max_tokens if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > self.caps.max_tokens:
            raise ValueError(
                f"tokens={live_tokens} exceeds capacity {self.caps.max_tokens}"
            )
        return live_tokens

    def bind(
        self,
        *,
        normalized: torch.Tensor,
        bottleneck: torch.Tensor,
        block_input: torch.Tensor,
        tokens: int | None = None,
    ) -> HyperConnectionBinding:
        live_tokens = self._live_tokens(tokens)
        caps = self.caps
        width = caps.streams * caps.hidden_size
        capacity_outputs = {
            "normalized": normalized,
            "bottleneck": bottleneck,
            "block_input": block_input,
        }
        tails = {
            "normalized": (width,),
            "bottleneck": (caps.lowrank,),
            "block_input": (caps.hidden_size,),
        }
        for name, tensor in capacity_outputs.items():
            _validate_capacity(
                tensor,
                tokens=live_tokens,
                tail=tails[name],
                dtype=caps.dtype,
                device=caps.device,
                name=name,
            )
        # Dynamo cannot trace storage/data-pointer inspection. Callers that bind
        # fixed workspaces inside a compiled forward must validate the same
        # buffers once before compilation.
        if not torch.compiler.is_compiling():
            for other_name in ("bottleneck", "block_input"):
                if _overlaps(normalized, capacity_outputs[other_name]):
                    raise ValueError(
                        f"normalized and {other_name} outputs must not overlap"
                    )
        return HyperConnectionBinding(
            plan=self,
            tokens=live_tokens,
            normalized_capacity=normalized,
            bottleneck_capacity=bottleneck,
            block_input_capacity=block_input,
        )


def plan_hyperconnection(
    caps: HyperConnectionCaps,
    *,
    policy: PolicyContext | None = None,
) -> HyperConnectionPlan:
    """Plan fixed launch geometry for the supplied serving capacity."""

    if not isinstance(caps, HyperConnectionCaps):
        raise TypeError(f"caps must be HyperConnectionCaps, got {type(caps)!r}")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    resolution = policy.resolve(
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery(
            dtype=str(caps.dtype).removeprefix("torch."),
            max_tokens=caps.max_tokens,
            hidden_size=caps.hidden_size,
            streams=caps.streams,
            lowrank=caps.lowrank,
        ),
    )
    config = resolution.config
    return HyperConnectionPlan(
        caps=caps,
        reduction_block_h=config.reduction_block_h,
        pointwise_block=config.pointwise_block,
        reduction_num_warps=config.reduction_num_warps,
        policy_resolution=resolution,
    )


def _validate_capacity(
    tensor: torch.Tensor,
    *,
    tokens: int,
    tail: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> None:
    if tensor.ndim != len(tail) + 1 or tuple(tensor.shape[1:]) != tail:
        raise ValueError(
            f"{name} must have tail shape {tail}, got {tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) < tokens:
        raise ValueError(
            f"{name} capacity {int(tensor.shape[0])} is smaller than tokens={tokens}"
        )
    if tensor.dtype != dtype or tensor.device != device:
        raise ValueError(
            f"{name} must use dtype={dtype} and device={device}; "
            f"got dtype={tensor.dtype}, device={tensor.device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def _validate_input(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    caps: HyperConnectionCaps,
    name: str,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != caps.dtype or tensor.device != caps.device:
        raise ValueError(
            f"{name} must use dtype={caps.dtype} and device={caps.device}; "
            f"got dtype={tensor.dtype}, device={tensor.device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_output_disjoint(
    output_name: str,
    output: torch.Tensor,
    inputs: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    # Dynamo cannot trace storage/data-pointer inspection. The launch remains
    # opaque through the mutating custom op; eager warmup performs this alias
    # check before serving compilation.
    if torch.compiler.is_compiling():
        return
    for input_name, tensor in inputs:
        if _overlaps(output, tensor):
            raise ValueError(f"{output_name} must not overlap {input_name}")


def _require_cuda(plan: HyperConnectionPlan) -> None:
    if plan.caps.device.type != "cuda":
        raise ValueError(
            "HyperConnection GPU entry points require CUDA; use the "
            "explicit reference module for an oracle"
        )


def _require_eps(eps: float) -> float:
    value = float(eps)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {value}")
    return value


def run_grouped_rmsnorm_impl(
    state: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    binding: HyperConnectionBinding,
) -> torch.Tensor:
    _require_cuda(binding.plan)
    caps = binding.plan.caps
    width = caps.streams * caps.hidden_size
    shape = (binding.tokens, width)
    _validate_input(state, shape=shape, caps=caps, name="state")
    _validate_input(weight, shape=(width,), caps=caps, name="weight")
    _validate_output_disjoint(
        "normalized",
        binding.normalized_capacity,
        (("state", state), ("weight", weight)),
    )
    eps = _require_eps(eps)
    if binding.tokens:
        from ._kernels import run_grouped_rmsnorm

        run_grouped_rmsnorm(
            state,
            weight,
            binding.normalized_capacity,
            eps=eps,
            streams=caps.streams,
            hidden_size=caps.hidden_size,
            block_h=binding.plan.reduction_block_h,
            num_warps=binding.plan.reduction_num_warps,
        )
    return binding.normalized


def run_scaled_silu_impl(
    projected_down: torch.Tensor,
    *,
    binding: HyperConnectionBinding,
) -> torch.Tensor:
    _require_cuda(binding.plan)
    caps = binding.plan.caps
    _validate_input(
        projected_down,
        shape=(binding.tokens, caps.lowrank),
        caps=caps,
        name="projected_down",
    )
    _validate_output_disjoint(
        "bottleneck",
        binding.bottleneck_capacity,
        (("projected_down", projected_down),),
    )
    if binding.tokens:
        from ._kernels import run_scaled_silu

        run_scaled_silu(
            projected_down,
            binding.bottleneck_capacity,
            streams=caps.streams,
            block=binding.plan.pointwise_block,
        )
    return binding.bottleneck


def run_gate_mean_impl(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    binding: HyperConnectionBinding,
) -> torch.Tensor:
    _require_cuda(binding.plan)
    caps = binding.plan.caps
    shape = (binding.tokens, caps.streams * caps.hidden_size)
    _validate_input(normalized, shape=shape, caps=caps, name="normalized")
    _validate_input(gate_logits, shape=shape, caps=caps, name="gate_logits")
    _validate_output_disjoint(
        "block_input",
        binding.block_input_capacity,
        (("normalized", normalized), ("gate_logits", gate_logits)),
    )
    if binding.tokens:
        from ._kernels import run_gate_mean

        run_gate_mean(
            normalized,
            gate_logits,
            binding.block_input_capacity,
            streams=caps.streams,
            hidden_size=caps.hidden_size,
            block_h=binding.plan.pointwise_block,
        )
    return binding.block_input


def _validate_combine_inputs(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    *,
    plan: HyperConnectionPlan,
    tokens: int,
) -> None:
    caps = plan.caps
    _validate_input(
        state,
        shape=(tokens, caps.streams * caps.hidden_size),
        caps=caps,
        name="state",
    )
    _validate_input(
        block_output,
        shape=(tokens, caps.hidden_size),
        caps=caps,
        name="block_output",
    )
    _validate_input(
        injection_logits,
        shape=(tokens, caps.streams),
        caps=caps,
        name="injection_logits",
    )


def run_combine_impl(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    *,
    plan: HyperConnectionPlan,
) -> torch.Tensor:
    _require_cuda(plan)
    tokens = plan._live_tokens(state.shape[0])
    _validate_combine_inputs(
        state,
        block_output,
        injection_logits,
        plan=plan,
        tokens=tokens,
    )
    from ._kernels import run_combine

    caps = plan.caps
    return run_combine(
        state,
        block_output,
        injection_logits,
        streams=caps.streams,
        hidden_size=caps.hidden_size,
        block_h=plan.reduction_block_h,
        num_warps=plan.reduction_num_warps,
    )


def run_combine_norm_impl(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    *,
    eps: float,
    plan: HyperConnectionPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda(plan)
    caps = plan.caps
    tokens = plan._live_tokens(state.shape[0])
    _validate_combine_inputs(
        state,
        block_output,
        injection_logits,
        plan=plan,
        tokens=tokens,
    )
    _validate_input(
        next_norm_weight,
        shape=(caps.streams * caps.hidden_size,),
        caps=caps,
        name="next_norm_weight",
    )
    eps = _require_eps(eps)
    from ._kernels import run_combine_norm

    return run_combine_norm(
        state,
        block_output,
        injection_logits,
        next_norm_weight,
        eps=eps,
        streams=caps.streams,
        hidden_size=caps.hidden_size,
        block_h=plan.reduction_block_h,
        num_warps=plan.reduction_num_warps,
    )


__all__ = [
    "HyperConnectionCaps",
    "HyperConnectionPlan",
    "HyperConnectionBinding",
    "plan_hyperconnection",
    "run_grouped_rmsnorm_impl",
    "run_scaled_silu_impl",
    "run_gate_mean_impl",
    "run_combine_impl",
    "run_combine_norm_impl",
]
