"""Capacity plan and binding for BF16 vocabulary projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.scratch import ScratchBufferSpec
from b12x.policy import PolicyContext, PolicyResolution, get_auto_policy

from ._kernel import bf16_vocab_projection  # noqa: F401
from ._policy import (
    BF16_VOCAB_PROJECTION_POLICY,
    Bf16VocabProjectionConfig,
    Bf16VocabProjectionQuery,
)


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass(frozen=True, kw_only=True)
class Caps:
    device: torch.device | str
    max_tokens: int
    in_features: int
    out_features: int
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        for name in ("max_tokens", "in_features", "out_features"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        if self.dtype != torch.bfloat16:
            raise TypeError("BF16 vocabulary projection requires torch.bfloat16")


@dataclass(frozen=True, kw_only=True)
class Plan:
    caps: Caps
    config: Bf16VocabProjectionConfig
    policy_resolution: PolicyResolution[Bf16VocabProjectionConfig]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return ()

    def bind(self, **kwargs: object) -> "Binding":
        return bind(self, **kwargs)


@dataclass(frozen=True, kw_only=True)
class Binding:
    plan: Plan
    source: torch.Tensor
    weight: torch.Tensor


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Resolve one immutable projection plan for ``caps``."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be Caps")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    resolution = policy.resolve(
        BF16_VOCAB_PROJECTION_POLICY,
        Bf16VocabProjectionQuery(
            dtype="bfloat16",
            max_tokens=caps.max_tokens,
            in_features=caps.in_features,
            out_features=caps.out_features,
        ),
    )
    return Plan(caps=caps, config=resolution.config, policy_resolution=resolution)


def bind(
    planned: Plan,
    *,
    source: torch.Tensor,
    weight: torch.Tensor,
) -> Binding:
    """Validate runtime tensors and create an allocation-free binding."""
    if not isinstance(planned, Plan):
        raise TypeError("planned must be Plan")
    caps = planned.caps
    if source.ndim != 2 or not 0 < source.shape[0] <= caps.max_tokens:
        raise ValueError(
            f"source must have 1..{caps.max_tokens} rows, got {tuple(source.shape)}"
        )
    if source.shape[1] != caps.in_features:
        raise ValueError(f"source K must be {caps.in_features}, got {source.shape[1]}")
    if tuple(weight.shape) != (caps.out_features, caps.in_features):
        raise ValueError(
            "weight must have shape "
            f"{(caps.out_features, caps.in_features)}, got {tuple(weight.shape)}"
        )
    for name, tensor in (("source", source), ("weight", weight)):
        if tensor.device != caps.device:
            raise ValueError(f"{name} must be on {caps.device}, got {tensor.device}")
        if tensor.dtype != caps.dtype:
            raise TypeError(f"{name} must have dtype {caps.dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if planned.config.backend == "triton" and source.shape[0] != 1:
        raise ValueError("the planned Triton vocabulary projection requires one row")
    return Binding(plan=planned, source=source, weight=weight)


def run(binding: Binding) -> torch.Tensor:
    """Run a bound projection through its preselected backend."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be Binding")
    config = binding.plan.config
    if config.backend == "torch":
        return torch.nn.functional.linear(binding.source, binding.weight)
    algorithm = {"row": 0, "loop": 1}[config.algorithm]
    return torch.ops.b12x.bf16_vocab_projection(
        binding.source,
        binding.weight,
        algorithm,
        config.block_k,
        config.num_warps,
    )


__all__ = ["Binding", "Caps", "Plan", "bind", "plan", "run"]
