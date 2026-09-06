"""Public surface for attention.compressed_sparse_mla (docs in the op ``__init__``)."""

from __future__ import annotations

from dataclasses import replace

from ..._lib.gating import default_is_supported
from ...policy import NO_POLICY_OVERRIDE, PolicyContext, get_auto_policy
from .._shared.mla.api import (
    clear_mla_caches as clear_caches,
)
from .._shared.mla.compressed_api import (
    compressed_sparse_mla_decode_forward as run,
)
from .._shared.mla.compressed_api import (
    compressed_sparse_mla_split_chunks_for_contract as split_chunks_for_contract,
)
from . import META
from ._policy import (
    COMPRESSED_SPARSE_MLA_POLICY,
    SparseMlaConfig,
    SparseMlaQuery,
)
from ._scratch import (
    B12XCompressedSparseMLABinding as Binding,
)
from ._scratch import (
    B12XCompressedSparseMLAScratch as Scratch,
)
from ._scratch import (
    B12XCompressedSparseMLAScratchCaps as Caps,
)
from ._scratch import (
    B12XCompressedSparseMLAScratchPlan as Plan,
)
from ._scratch import (
    plan_compressed_sparse_mla_scratch,
)


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Size fixed scratch and resolve the capture-static split contract."""

    if not isinstance(caps, Caps):
        raise TypeError("caps must be compressed_sparse_mla.Caps")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    query = SparseMlaQuery(
        layout=caps.layout,
        mode=caps.mode,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        num_q_heads=caps.num_q_heads,
        qk_head_dim=caps.head_dim,
        v_head_dim=caps.v_head_dim,
        swa_width=caps.swa_width,
        swa_page_size=caps.swa_page_size,
        indexed_width=caps.indexed_width,
        indexed_page_size=caps.indexed_page_size,
        query_rows=caps.max_q_rows,
    )
    override = NO_POLICY_OVERRIDE
    if caps.max_chunks_per_row is not None:
        override = SparseMlaConfig(
            max_chunks_per_row=caps.max_chunks_per_row,
        )
    resolution = policy.resolve(
        COMPRESSED_SPARSE_MLA_POLICY,
        query,
        override=override,
    )
    effective_caps = replace(
        caps,
        max_chunks_per_row=resolution.config.max_chunks_per_row,
    )
    return replace(
        plan_compressed_sparse_mla_scratch(effective_caps),
        policy_resolution=resolution,
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
    "Scratch",
    "SparseMlaConfig",
    "SparseMlaQuery",
    "plan",
    "bind",
    "run",
    "split_chunks_for_contract",
    "is_supported",
    "clear_caches",
]
