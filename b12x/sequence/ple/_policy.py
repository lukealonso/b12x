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


from b12x.policy.problem import define_problem

TUNING_PROBLEM = define_problem(
    policy=PLE_POLICY, query_type=PleQuery, config_type=PleConfig,
    axes=('max_tokens', 'max_seqs', 'max_speculative_tokens', 'streams', 'hidden_size', 'kernel_size', 'dilation'),
    family=('mode', 'dtype'),
    constraints=(),
    environment=(),
    model_fields=('streams', 'hidden_size', 'kernel_size', 'dilation'),
    decisions={'backend': ('triton',)},
    derived_config_fields=(),
)
