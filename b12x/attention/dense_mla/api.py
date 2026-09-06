"""Public planned API for paged dense MLA."""

from __future__ import annotations

from dataclasses import replace

import torch

from ..._lib.gating import default_is_supported
from ...policy import NO_POLICY_OVERRIDE, PolicyContext, get_auto_policy
from . import META
from ._kernel import (
    clear_dense_mla_kernel_caches,
    compile_dense_mla,
    run_dense_mla,
)
from ._scratch import Binding, Caps, Plan, Scratch
from ._scratch import (
    plan_dense_mla_scratch,
)
from .planner import Budget
from ._policy import DENSE_MLA_POLICY, DenseMlaConfig, DenseMlaQuery
from .planner import (
    infer_dense_mla_mode,
)
from ._reference import dense_mla_reference


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Size fixed caller-owned storage and select shape-only kernel policy."""

    if not isinstance(caps, Caps):
        raise TypeError("caps must be dense_mla.Caps")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    query = DenseMlaQuery(
        mode=caps.mode,
        q_dtype=_dtype_name(caps.q_dtype),
        kv_dtype=_dtype_name(caps.kv_dtype),
        num_q_heads=caps.num_q_heads,
        qk_head_dim=caps.head_dim,
        v_head_dim=caps.v_head_dim,
        page_size=caps.page_size,
        query_rows=caps.max_total_q,
        max_batch=caps.max_batch,
        cache_tokens=caps.max_cache_tokens,
        physical_record_width=caps.physical_record_width,
        window_size=caps.window_size,
        use_cuda_graph=caps.use_cuda_graph,
    )
    existing_budget = caps.budget or Budget()
    override = NO_POLICY_OVERRIDE
    if existing_budget.max_splits is not None:
        override = DenseMlaConfig(max_splits=existing_budget.max_splits)
    resolution = policy.resolve(DENSE_MLA_POLICY, query, override=override)
    effective_caps = replace(
        caps,
        budget=Budget(
            max_splits=resolution.config.max_splits,
            max_partial_rows=existing_budget.max_partial_rows,
        ),
    )
    return replace(
        plan_dense_mla_scratch(effective_caps),
        policy_resolution=resolution,
    )


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors using views only; this function never allocates."""
    return plan.bind(**kwargs)


def compile(*, binding: Binding) -> None:
    """Compile every entry selected by ``binding`` without launching it."""
    compile_dense_mla(binding=binding)


def run(*, binding: Binding) -> tuple[torch.Tensor, torch.Tensor]:
    """Run dense MLA and return BF16 output plus FP32 natural-log LSE."""
    return run_dense_mla(binding=binding)


def reference(*args, **kwargs):
    """Run the FP32 paged dense-MLA oracle."""
    return dense_mla_reference(*args, **kwargs)


def infer_mode(cu_seqlens_q):
    return infer_dense_mla_mode(cu_seqlens_q)


def is_supported(device=None) -> bool:
    """True only for the fail-closed SM120/SM121 production envelope."""
    if not default_is_supported(device, requires=META.requires):
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    return tuple(torch.cuda.get_device_capability(device)) in ((12, 0), (12, 1))


def clear_caches() -> None:
    clear_dense_mla_kernel_caches()


__all__ = [
    "Binding",
    "Budget",
    "DenseMlaConfig",
    "DenseMlaQuery",
    "Caps",
    "Plan",
    "Scratch",
    "bind",
    "clear_caches",
    "compile",
    "infer_mode",
    "is_supported",
    "plan",
    "reference",
    "run",
]
