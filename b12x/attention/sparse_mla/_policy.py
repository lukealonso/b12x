"""Typed component policy for sparse MLA planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    SPARSE_MLA_ATTENTION,
    BackendConfig,
    make_fixed_backend_policy,
)


@dataclass(frozen=True, kw_only=True)
class SparseMlaQuery:
    mode: str
    dtype: str
    kv_dtype: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    max_q_rows: int
    max_width: int
    page_size: int
    model_type: int | None
    head_major_output: bool


SPARSE_MLA_POLICY = make_fixed_backend_policy(
    component_id=SPARSE_MLA_ATTENTION,
    query_type=SparseMlaQuery,
    backend="native",
)
SparseMlaConfig = BackendConfig


__all__ = ["SPARSE_MLA_POLICY", "SparseMlaConfig", "SparseMlaQuery"]


from b12x.policy.problem import define_problem

TUNING_PROBLEM = define_problem(
    policy=SPARSE_MLA_POLICY, query_type=SparseMlaQuery, config_type=SparseMlaConfig,
    axes=('num_q_heads', 'qk_head_dim', 'v_head_dim', 'max_q_rows', 'max_width'),
    family=('mode', 'dtype', 'kv_dtype', 'page_size', 'model_type', 'head_major_output'),
    constraints=(),
    environment=(),
    model_fields=('num_q_heads', 'qk_head_dim', 'v_head_dim', 'page_size', 'model_type', 'head_major_output'),
    decisions={'backend': ('native',)},
    derived_config_fields=(),
)
