"""Typed component policy for PLE state planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import PLE, BackendConfig, make_fixed_backend_policy


@dataclass(frozen=True, kw_only=True)
class PleQuery:
    mode: str
    dtype: str
    max_tokens: int
    max_seqs: int
    max_speculative_tokens: int
    streams: int
    hidden_size: int
    kernel_size: int
    dilation: int


PLE_POLICY = make_fixed_backend_policy(
    component_id=PLE,
    query_type=PleQuery,
    backend="triton",
)
PleConfig = BackendConfig


__all__ = ["PLE_POLICY", "PleConfig", "PleQuery"]
