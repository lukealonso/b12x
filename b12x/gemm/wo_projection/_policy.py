"""Typed component policy for W_o projection planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import WO_PROJECTION, BackendConfig, make_fixed_backend_policy


@dataclass(frozen=True, kw_only=True)
class WoProjectionQuery:
    dtype: str
    max_tokens: int
    groups: int
    group_width: int
    rank: int
    hidden: int


WO_PROJECTION_POLICY = make_fixed_backend_policy(
    component_id=WO_PROJECTION,
    query_type=WoProjectionQuery,
    backend="mxfp8",
)
WoProjectionConfig = BackendConfig


__all__ = ["WO_PROJECTION_POLICY", "WoProjectionConfig", "WoProjectionQuery"]
