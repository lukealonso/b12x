"""Public surface for quantization.nvfp4 (docs in the op ``__init__``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from b12x.policy import PolicyContext, get_auto_policy

from ..._lib.gating import default_is_supported
from . import META
from ._impl import (
    BF16ToFP4TMAOutputs as Outputs,
)
from ._impl import (
    allocate_bf16_to_fp4_tma_outputs as _allocate,
)
from ._impl import (
    compile_bf16_to_fp4_tma as _compile,
)
from ._policy import (
    NVFP4_QUANTIZATION_POLICY,
    Nvfp4QuantizationConfig,
    Nvfp4QuantizationQuery,
)


@dataclass(frozen=True)
class Plan:
    """A compiled (m, k) shape; produced by :func:`plan`."""

    m: int
    k: int
    launch: Callable[..., None]
    policy_resolution: object | None = None


def plan(
    m: int,
    k: int,
    *,
    policy: PolicyContext | None = None,
) -> Plan:
    """Compile the quantizer for (m, k); host-side, cached per shape."""
    policy = policy or get_auto_policy()
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    if torch.cuda.is_available():
        policy.require_device(torch.device("cuda", torch.cuda.current_device()))
    resolution = policy.resolve(
        NVFP4_QUANTIZATION_POLICY,
        Nvfp4QuantizationQuery(
            dtype="bfloat16",
            rows=int(m),
            columns=int(k),
        ),
    )
    return Plan(
        m=int(m),
        k=int(k),
        launch=_compile(
            int(m),
            int(k),
            liveness_strategy=resolution.config.liveness_strategy,
        ),
        policy_resolution=resolution,
    )


def allocate_outputs(plan: Plan, *, device: torch.device | str = "cuda") -> Outputs:
    """Allocate the packed-FP4 + MMA-layout-scale output pair for a plan."""
    return _allocate(plan.m, plan.k, device=torch.device(device))


def run(
    *,
    plan: Plan,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    outputs: Outputs,
) -> None:
    """Quantize ``x`` into ``outputs`` (allocation-free, capture safe)."""
    plan.launch(x, global_scale, outputs.packed_a_flat, outputs.scale_flat)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Outputs",
    "Plan",
    "Nvfp4QuantizationConfig",
    "Nvfp4QuantizationQuery",
    "plan",
    "allocate_outputs",
    "run",
    "is_supported",
]
