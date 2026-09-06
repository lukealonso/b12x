"""Public surface for attention.sparse_mla (docs in the op ``__init__``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..._lib.gating import default_is_supported
from ...policy import PolicyContext
from .._shared.mla.traits import ModelType
from .._shared.mla.api import (
    MLASparseDecodeMetadata as DecodeMetadata,
)
from .._shared.mla.api import (
    MLASparseExtendMetadata as ExtendMetadata,
)
from .._shared.mla.api import (
    clear_mla_caches as clear_caches,
)
from .._shared.mla.api import (
    sparse_mla_decode_forward as _run_decode,
)
from .._shared.mla.api import (
    sparse_mla_extend_forward as _run_extend,
)
from .._shared.mla.kv_cache import (
    compile_glm_next_mla_cache_writer,
    concat_and_cache_glm_next_mla,
    concat_and_cache_glm_next_mla_fp8,
    concat_and_cache_glm_next_mla_nvfp4,
    concat_and_cache_nvfp4_mla_fp8_rope,
)
from .pooled_selection import expand_pooled_topk_to_physical_slots
from ._scratch import (
    B12XSparseMLABinding as _RuntimeBinding,
)
from ._policy import SparseMlaConfig, SparseMlaQuery
from ._scratch import (
    B12XSparseMLAScratch as Scratch,
)
from ._scratch import (
    B12XSparseMLAScratchCaps as Caps,
)
from ._scratch import (
    B12XSparseMLAScratchPlan as Plan,
)
from ._scratch import (
    plan_sparse_mla_scratch,
)
from . import META


@dataclass(frozen=True, kw_only=True)
class Binding:
    """A complete sparse-MLA invocation bound to one immutable plan."""

    plan: Plan
    runtime: _RuntimeBinding
    kv_cache: torch.Tensor
    attention_sink: torch.Tensor | None = None


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Resolve policy and scratch layout once for a fixed capacity."""

    return plan_sparse_mla_scratch(caps, policy=policy)


def bind(
    plan: Plan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    cache_lengths: torch.Tensor,
    selected_lengths: torch.Tensor,
    attention_sink: torch.Tensor | None = None,
) -> Binding:
    """Bind every live tensor to a plan without allocating or launching work.

    Static execution semantics come from ``plan.caps``. The returned binding is
    the complete input to :func:`run`.
    """

    if not isinstance(plan, Plan):
        raise TypeError("plan must be sparse_mla.Plan")
    caps = plan.caps
    if kv_cache.ndim != 3:
        raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
    if kv_cache.device != caps.device:
        raise ValueError(f"kv_cache must be on {caps.device}, got {kv_cache.device}")
    if kv_cache.dtype != caps.kv_dtype:
        raise TypeError(
            f"kv_cache must have dtype {caps.kv_dtype}, got {kv_cache.dtype}"
        )
    if bool(attention_sink is not None) != bool(caps.has_attention_sink):
        raise ValueError(
            "attention_sink presence must match plan.caps.has_attention_sink"
        )
    if attention_sink is not None and caps.mode != "decode":
        raise ValueError("attention_sink is supported only by sparse MLA decode")
    if attention_sink is not None:
        if attention_sink.shape != (caps.num_q_heads,):
            raise ValueError(
                "attention_sink must have shape "
                f"({caps.num_q_heads},), got {tuple(attention_sink.shape)}"
            )
        if attention_sink.dtype != torch.float32:
            raise TypeError(
                f"attention_sink must have dtype torch.float32, got "
                f"{attention_sink.dtype}"
            )
        if attention_sink.device != caps.device or not attention_sink.is_contiguous():
            raise ValueError(f"attention_sink must be contiguous on {caps.device}")
    runtime = plan.bind(
        scratch=scratch,
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_lengths,
        nsa_cache_seqlens_int32=selected_lengths,
        kv_cache=kv_cache,
    )
    return Binding(
        plan=plan,
        runtime=runtime,
        kv_cache=kv_cache,
        attention_sink=attention_sink,
    )


def run(binding: Binding) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute a complete binding through the route selected by its plan."""

    if not isinstance(binding, Binding):
        raise TypeError("binding must be sparse_mla.Binding")
    caps = binding.plan.caps
    kwargs = dict(
        binding=binding.runtime,
        sm_scale=caps.softmax_scale,
        latent_scale=caps.latent_scale,
        v_head_dim=caps.v_head_dim,
        return_lse=caps.return_lse,
        lse_scale=caps.lse_scale,
    )
    if caps.mode == "decode":
        return _run_decode(attn_sink=binding.attention_sink, **kwargs)
    return _run_extend(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "ModelType",
    "Caps",
    "Plan",
    "Binding",
    "Scratch",
    "DecodeMetadata",
    "ExtendMetadata",
    "SparseMlaConfig",
    "SparseMlaQuery",
    "plan",
    "bind",
    "run",
    "compile_glm_next_mla_cache_writer",
    "concat_and_cache_glm_next_mla",
    "concat_and_cache_glm_next_mla_fp8",
    "concat_and_cache_glm_next_mla_nvfp4",
    "concat_and_cache_nvfp4_mla_fp8_rope",
    "expand_pooled_topk_to_physical_slots",
    "is_supported",
    "clear_caches",
]
