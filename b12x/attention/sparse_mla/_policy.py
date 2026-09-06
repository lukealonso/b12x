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
