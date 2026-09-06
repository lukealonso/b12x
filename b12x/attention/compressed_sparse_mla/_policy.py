"""Typed component policy for compressed sparse MLA planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    COMPRESSED_SPARSE_MLA_ATTENTION,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True, kw_only=True)
class SparseMlaQuery:
    layout: str
    mode: str
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    swa_width: int
    swa_page_size: int
    indexed_width: int
    indexed_page_size: int
    query_rows: int

    def profile_fields(self) -> dict[str, object]:
        return {
            "layout": self.layout,
            "mode": self.mode,
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "num_q_heads": self.num_q_heads,
            "qk_head_dim": self.qk_head_dim,
            "v_head_dim": self.v_head_dim,
            "swa_width": self.swa_width,
            "swa_page_size": self.swa_page_size,
            "indexed_width": self.indexed_width,
            "indexed_page_size": self.indexed_page_size,
            "query_rows": self.query_rows,
        }


@dataclass(frozen=True, kw_only=True)
class SparseMlaConfig:
    max_chunks_per_row: int

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "SparseMlaConfig":
        if set(payload) != {"max_chunks_per_row"}:
            raise ValueError(
                "sparse MLA profiles require exactly max_chunks_per_row"
            )
        value = payload["max_chunks_per_row"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("sparse MLA max_chunks_per_row must be an integer")
        return cls(max_chunks_per_row=value)


def _heuristic(
    query: SparseMlaQuery,
    device: DeviceIdentity | None,
) -> SparseMlaConfig:
    from b12x.attention._shared.mla.compressed_config import compressed_sparse_mla_uses_single_pass_decode

    uses_single_pass = query.mode != "decode" or compressed_sparse_mla_uses_single_pass_decode(
        rows=query.query_rows, heads=query.num_q_heads, swa_width=query.swa_width,
        indexed_width=query.indexed_width, swa_page_size=query.swa_page_size,
        indexed_page_size=query.indexed_page_size,
        compute_capability=(0, 0) if device is None else device.compute_capability,
    )
    return SparseMlaConfig(max_chunks_per_row=1 if uses_single_pass else 64)


def _validate(
    _query: SparseMlaQuery,
    config: SparseMlaConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.max_chunks_per_row <= 0:
        raise ValueError("sparse MLA max_chunks_per_row must be positive")


COMPRESSED_SPARSE_MLA_POLICY = ComponentPolicy(
    component_id=COMPRESSED_SPARSE_MLA_ATTENTION,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(
        {
            "layout",
            "mode",
            "q_dtype",
            "kv_dtype",
            "num_q_heads",
            "qk_head_dim",
            "v_head_dim",
            "swa_width",
            "swa_page_size",
            "indexed_width",
            "indexed_page_size",
            "query_rows",
        }
    ),
    config_fields=frozenset({"max_chunks_per_row"}),
    encode_query=SparseMlaQuery.profile_fields,
    decode_profile=lambda query, device, payload: SparseMlaConfig.from_profile(payload),
    heuristic=_heuristic,
    validate_config=_validate,
)
SPARSE_MLA_POLICY = COMPRESSED_SPARSE_MLA_POLICY


__all__ = [
    "COMPRESSED_SPARSE_MLA_POLICY",
    "SPARSE_MLA_POLICY",
    "SparseMlaConfig",
    "SparseMlaQuery",
]


from b12x.policy.problem import define_problem

TUNING_PROBLEM = define_problem(
    policy=COMPRESSED_SPARSE_MLA_POLICY, query_type=SparseMlaQuery, config_type=SparseMlaConfig,
    axes=('num_q_heads', 'qk_head_dim', 'v_head_dim', 'swa_width', 'indexed_width', 'query_rows'),
    family=('layout', 'mode', 'q_dtype', 'kv_dtype', 'swa_page_size', 'indexed_page_size'),
    constraints=(),
    environment=(),
    model_fields=('num_q_heads', 'qk_head_dim', 'v_head_dim', 'swa_width', 'indexed_width', 'swa_page_size', 'indexed_page_size'),
    decisions={'max_chunks_per_row': (1, 2, 4, 8, 16, 32, 64, 256)},
    ordered=('max_chunks_per_row',),
    axis_domains={'num_q_heads': (1, 1), 'qk_head_dim': (1, 1), 'v_head_dim': (1, 1),
                  'swa_width': (0, 1), 'indexed_width': (0, 1), 'query_rows': (1, 1)},
    derived_config_fields=(),
)
