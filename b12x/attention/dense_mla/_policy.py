"""Typed component policy for dense MLA planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    MLA_ATTENTION,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)

from .planner import Budget, choose_num_splits


@dataclass(frozen=True, kw_only=True)
class DenseMlaQuery:
    mode: str
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    page_size: int
    query_rows: int
    max_batch: int
    cache_tokens: int
    physical_record_width: int
    window_size: int | None
    use_cuda_graph: bool

    def profile_fields(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "num_q_heads": self.num_q_heads,
            "qk_head_dim": self.qk_head_dim,
            "v_head_dim": self.v_head_dim,
            "page_size": self.page_size,
            "query_rows": self.query_rows,
            "max_batch": self.max_batch,
            "cache_tokens": self.cache_tokens,
            "physical_record_width": self.physical_record_width,
            "window_size": self.window_size,
            "use_cuda_graph": self.use_cuda_graph,
        }


@dataclass(frozen=True, kw_only=True)
class DenseMlaConfig:
    max_splits: int

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "DenseMlaConfig":
        if set(payload) != {"max_splits"}:
            raise ValueError("dense MLA profiles require exactly max_splits")
        value = payload["max_splits"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("dense MLA max_splits must be an integer")
        return cls(max_splits=value)


def _query_tile(query: DenseMlaQuery) -> int:
    if (
        query.mode == "decode"
        or query.max_batch != 1
        or query.window_size is not None
    ):
        return 1
    if query.kv_dtype == "float8_e4m3fn" and query.query_rows >= 3:
        return 4
    return 2


def _heuristic(
    query: DenseMlaQuery,
    device: DeviceIdentity | None,
) -> DenseMlaConfig:
    max_attended_tokens = query.cache_tokens
    if query.window_size is not None:
        max_attended_tokens = min(
            query.cache_tokens,
            query.window_size + query.page_size - 1,
        )
    splits = choose_num_splits(
        max_cache_tokens=max_attended_tokens,
        max_total_q=query.query_rows,
        num_q_heads=query.num_q_heads,
        query_tile=_query_tile(query),
        sm_count=1 if device is None else device.sm_count,
        budget=Budget(),
    )
    return DenseMlaConfig(max_splits=splits)


def _validate(
    query: DenseMlaQuery,
    config: DenseMlaConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.max_splits <= 0:
        raise ValueError("dense MLA max_splits must be positive")
    max_chunks = max(1, (query.cache_tokens + 63) // 64)
    if config.max_splits > max_chunks:
        raise ValueError(
            "dense MLA max_splits cannot exceed the cache chunk count"
        )


DENSE_MLA_POLICY = ComponentPolicy(
    component_id=MLA_ATTENTION,
    query_schema_version=2,
    config_schema_version=1,
    query_fields=frozenset(
        {
            "mode",
            "q_dtype",
            "kv_dtype",
            "num_q_heads",
            "qk_head_dim",
            "v_head_dim",
            "page_size",
            "query_rows",
            "max_batch",
            "cache_tokens",
            "physical_record_width",
            "window_size",
            "use_cuda_graph",
        }
    ),
    config_fields=frozenset({"max_splits"}),
    encode_query=DenseMlaQuery.profile_fields,
    decode_profile=DenseMlaConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = ["DENSE_MLA_POLICY", "DenseMlaConfig", "DenseMlaQuery"]
