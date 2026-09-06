"""Typed component policy for expert-parallel MoE planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import EP_MOE, BackendConfig, make_fixed_backend_policy


@dataclass(frozen=True, kw_only=True)
class EpMoeQuery:
    max_tokens: int
    top_k: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    activation: str


EP_MOE_POLICY = make_fixed_backend_policy(
    component_id=EP_MOE,
    query_type=EpMoeQuery,
    backend="w4a16",
)
EpMoeConfig = BackendConfig


__all__ = ["EP_MOE_POLICY", "EpMoeConfig", "EpMoeQuery"]
