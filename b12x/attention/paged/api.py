"""Public surface for attention.paged (docs in the op ``__init__``)."""

from __future__ import annotations

from dataclasses import replace

from ..._lib.gating import default_is_supported
from ...policy import PolicyContext, get_auto_policy
from ._forward import (
    clear_paged_caches as clear_caches,
)
from ._forward import (
    compile_paged_attention as compile,
)
from ._forward import (
    paged_attention_forward as run,
)
from ._scratch import (
    B12XPagedAttentionBinding as Binding,
)
from ._scratch import (
    B12XPagedAttentionScratchCaps as Caps,
)
from ._scratch import (
    B12XPagedAttentionScratchPlan as Plan,
)
from ._scratch import (
    B12XPagedDecodeGraphScratchEnvelope as DecodeGraphScratchEnvelope,
)
from ._scratch import (
    plan_decode_graph_scratch_envelope as decode_graph_scratch_envelope,
)
from ._scratch import (
    plan_paged_attention_scratch,
)
from .planner import (
    PagedDecodeGraphCapacity as DecodeGraphCapacity,
)
from ._policy import GqaConfig, GqaQuery
from .planner import (
    PagedExtendGraphCapacity as ExtendGraphCapacity,
)
from .planner import (
    PagedVerifyGraphCapacity as VerifyGraphCapacity,
)
from .planner import (
    PagedPlanBudget as Budget,
)
from .planner import (
    infer_paged_mode as infer_mode,
)
from .planner import (
    plan_decode_graph_capacity as decode_graph_capacity,
)
from .planner import (
    plan_extend_graph_capacity as extend_graph_capacity,
)
from .planner import (
    plan_verify_graph_capacity as verify_graph_capacity,
)
from .workspace import (
    PagedAttentionWorkspace as Workspace,
)
from . import META


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Size fixed scratch under one immutable device policy context."""

    if not isinstance(caps, Caps):
        raise TypeError("caps must be paged.Caps")
    policy = policy or caps.policy_context or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    return plan_paged_attention_scratch(
        replace(caps, policy_context=policy),
    )


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "Workspace",
    "Budget",
    "DecodeGraphCapacity",
    "GqaConfig",
    "GqaQuery",
    "ExtendGraphCapacity",
    "VerifyGraphCapacity",
    "DecodeGraphScratchEnvelope",
    "decode_graph_capacity",
    "extend_graph_capacity",
    "verify_graph_capacity",
    "decode_graph_scratch_envelope",
    "plan",
    "bind",
    "compile",
    "run",
    "infer_mode",
    "is_supported",
    "clear_caches",
]
