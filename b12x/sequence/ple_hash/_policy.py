"""Typed component policy for PLE hash planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import PLE_HASH, BackendConfig, make_fixed_backend_policy


@dataclass(frozen=True, kw_only=True)
class PleHashQuery:
    max_tokens: int
    max_seqs: int
    vocab_size: int
    max_order: int
    heads_per_order: int
    base_table_size: int


PLE_HASH_POLICY = make_fixed_backend_policy(
    component_id=PLE_HASH,
    query_type=PleHashQuery,
    backend="triton",
)
PleHashConfig = BackendConfig


__all__ = ["PLE_HASH_POLICY", "PleHashConfig", "PleHashQuery"]
