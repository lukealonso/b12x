"""Typed component policy for Qwen grouped-selector sparse GQA."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    QSA_ATTENTION,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True, kw_only=True)
class QsaQuery:
    q_dtype: str
    kv_dtype: str
    q_heads: int
    kv_heads: int
    head_dim: int
    index_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_rotary_dim: int
    main_page_size: int
    max_batch: int
    max_q_rows: int
    max_seq_len: int
    max_speculative_tokens: int
    compress_ratio: int
    budget: int
    position_axes: int
    mrope_interleaved: bool

    def profile_fields(self) -> dict[str, object]:
        return {
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "index_heads": self.index_heads,
            "index_kv_heads": self.index_kv_heads,
            "index_head_dim": self.index_head_dim,
            "index_rotary_dim": self.index_rotary_dim,
            "main_page_size": self.main_page_size,
            "max_batch": self.max_batch,
            "max_q_rows": self.max_q_rows,
            "max_seq_len": self.max_seq_len,
            "max_speculative_tokens": self.max_speculative_tokens,
            "compress_ratio": self.compress_ratio,
            "budget": self.budget,
            "position_axes": self.position_axes,
            "mrope_interleaved": self.mrope_interleaved,
        }


@dataclass(frozen=True, kw_only=True)
class QsaConfig:
    backend: str
    sparse_gqa_direct_kv_warps: int = 2

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "QsaConfig":
        if set(payload) != {"backend", "sparse_gqa_direct_kv_warps"}:
            raise ValueError(
                "QSA profiles require exactly backend and sparse_gqa_direct_kv_warps"
            )
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("QSA backend must be a string")
        direct_kv_warps = payload["sparse_gqa_direct_kv_warps"]
        if not isinstance(direct_kv_warps, int) or isinstance(direct_kv_warps, bool):
            raise TypeError("QSA sparse_gqa_direct_kv_warps must be an integer")
        return cls(
            backend=backend,
            sparse_gqa_direct_kv_warps=direct_kv_warps,
        )


def _heuristic(
    _query: QsaQuery,
    _device: DeviceIdentity | None,
) -> QsaConfig:
    return QsaConfig(
        backend="cutedsl",
        sparse_gqa_direct_kv_warps=2,
    )


def _validate(
    query: QsaQuery,
    config: QsaConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend != "cutedsl":
        raise ValueError(f"unsupported QSA backend {config.backend!r}")
    if (
        not isinstance(config.sparse_gqa_direct_kv_warps, int)
        or isinstance(config.sparse_gqa_direct_kv_warps, bool)
        or config.sparse_gqa_direct_kv_warps not in (1, 2, 4)
    ):
        raise ValueError("QSA sparse_gqa_direct_kv_warps must be 1, 2, or 4")
    if query.q_dtype != "bfloat16":
        raise ValueError("QSA requires BF16 queries")
    if query.kv_dtype not in ("bfloat16", "float8_e4m3fn"):
        raise ValueError("QSA requires BF16 or FP8 E4M3 KV storage")
    if (query.q_heads, query.kv_heads) not in ((6, 1), (12, 1), (24, 2)):
        raise ValueError("unsupported QSA tensor-parallel head layout")
    if query.head_dim != 256 or query.index_head_dim != 128:
        raise ValueError("unsupported QSA head dimensions")
    if query.index_heads <= 0 or query.index_kv_heads != 1:
        raise ValueError("unsupported QSA index-head layout")
    if query.index_rotary_dim <= 0 or query.index_rotary_dim > query.index_head_dim:
        raise ValueError("unsupported QSA index rotary dimension")
    if (
        min(
            query.main_page_size,
            query.max_batch,
            query.max_q_rows,
            query.max_seq_len,
            query.compress_ratio,
            query.budget,
        )
        <= 0
    ):
        raise ValueError("QSA profile geometry must be positive")
    if query.max_q_rows < query.max_batch:
        raise ValueError("QSA max_q_rows must cover max_batch")
    if query.max_speculative_tokens < 0:
        raise ValueError("QSA speculative-token capacity must be nonnegative")
    if query.position_axes not in (1, 3):
        raise ValueError("QSA position_axes must be 1 or 3")
    if query.position_axes == 1 and query.mrope_interleaved:
        raise ValueError("scalar-position QSA cannot use interleaved M-RoPE")


QSA_POLICY = ComponentPolicy(
    component_id=QSA_ATTENTION,
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(QsaQuery.__dataclass_fields__),
    config_fields=frozenset(
        {
            "backend",
            "sparse_gqa_direct_kv_warps",
        }
    ),
    encode_query=QsaQuery.profile_fields,
    decode_profile=QsaConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = ["QSA_POLICY", "QsaConfig", "QsaQuery"]
