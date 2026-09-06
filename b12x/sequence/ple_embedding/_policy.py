"""Typed component policy for PLE embedding planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import PLE_EMBEDDING, BackendConfig, make_fixed_backend_policy


@dataclass(frozen=True, kw_only=True)
class PleEmbeddingQuery:
    quant_mode: str
    table_memory: str
    output_dtype: str
    max_tokens: int
    max_seqs: int
    vocab_size: int
    max_order: int
    heads_per_order: int
    base_table_size: int
    embedding_dim: int
    tp_size: int


PLE_EMBEDDING_POLICY = make_fixed_backend_policy(
    component_id=PLE_EMBEDDING,
    query_type=PleEmbeddingQuery,
    backend="triton",
)
PleEmbeddingConfig = BackendConfig


__all__ = [
    "PLE_EMBEDDING_POLICY",
    "PleEmbeddingConfig",
    "PleEmbeddingQuery",
]


from b12x.policy.problem import define_problem

TUNING_PROBLEM = define_problem(
    policy=PLE_EMBEDDING_POLICY, query_type=PleEmbeddingQuery, config_type=PleEmbeddingConfig,
    axes=('max_tokens', 'max_seqs', 'vocab_size', 'max_order', 'heads_per_order', 'base_table_size', 'embedding_dim', 'tp_size'),
    family=('quant_mode', 'table_memory', 'output_dtype'),
    constraints=(),
    environment=(),
    model_fields=('vocab_size', 'max_order', 'heads_per_order', 'base_table_size', 'embedding_dim', 'tp_size'),
    decisions={'backend': ('triton',)},
    derived_config_fields=(),
)
