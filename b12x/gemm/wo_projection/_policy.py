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


from b12x.policy.problem import define_problem

TUNING_PROBLEM = define_problem(
    policy=WO_PROJECTION_POLICY, query_type=WoProjectionQuery, config_type=WoProjectionConfig,
    axes=('max_tokens', 'groups', 'group_width', 'rank', 'hidden'),
    family=('dtype',),
    constraints=(),
    environment=(),
    model_fields=('groups', 'group_width', 'rank', 'hidden'),
    decisions={'backend': ('mxfp8',)},
    derived_config_fields=(),
)
