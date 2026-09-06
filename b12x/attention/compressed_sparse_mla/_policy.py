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
    capability = None if device is None else device.compute_capability
    uses_single_pass = query.mode != "decode" or (
        capability == (12, 1)
        and query.query_rows >= 16
        and query.num_q_heads == 32
        and query.swa_page_size == 64
        and (query.indexed_width == 0 or query.indexed_page_size == 64)
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
    decode_profile=SparseMlaConfig.from_profile,
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
