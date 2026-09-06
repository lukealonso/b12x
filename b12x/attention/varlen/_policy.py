"""Typed tile policy for contiguous batched and varlen attention."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    VARLEN_ATTENTION,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True, kw_only=True)
class VarlenAttentionQuery:
    variant: str
    dtype: str
    causal: bool
    batch_size: int
    q_heads: int
    kv_heads: int
    q_head_dim: int
    v_head_dim: int
    query_rows: int
    kv_rows: int
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass(frozen=True, kw_only=True)
class VarlenAttentionConfig:
    tile_m: int
    tile_n: int

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "VarlenAttentionConfig":
        if set(payload) != {"tile_m", "tile_n"}:
            raise ValueError("varlen attention configs require tile_m and tile_n")
        return cls(tile_m=int(payload["tile_m"]), tile_n=int(payload["tile_n"]))

    def to_dict(self) -> dict[str, object]:
        return {"tile_m": self.tile_m, "tile_n": self.tile_n}


def _encode(query: VarlenAttentionQuery) -> dict[str, object]:
    return {
        name: getattr(query, name) for name in VarlenAttentionQuery.__dataclass_fields__
    }


def _heuristic(
    query: VarlenAttentionQuery,
    _device: DeviceIdentity | None,
) -> VarlenAttentionConfig:
    if query.q_head_dim <= 64:
        return VarlenAttentionConfig(tile_m=128, tile_n=128)
    if query.q_head_dim <= 128:
        return VarlenAttentionConfig(tile_m=128, tile_n=64)
    if query.q_head_dim == 256:
        return VarlenAttentionConfig(
            tile_m=64,
            tile_n=32 if query.causal else 48,
        )
    raise ValueError(f"unsupported contiguous head_dim={query.q_head_dim}")


def _validate(
    query: VarlenAttentionQuery,
    config: VarlenAttentionConfig,
    _device: DeviceIdentity | None,
) -> None:
    if query.variant not in ("batched", "varlen"):
        raise ValueError(f"unsupported attention variant {query.variant!r}")
    if query.dtype not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported attention dtype {query.dtype!r}")
    if query.q_heads <= 0 or query.kv_heads <= 0:
        raise ValueError("attention head counts must be positive")
    if query.q_heads % query.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    if query.q_head_dim <= 0 or query.v_head_dim <= 0:
        raise ValueError("attention head dimensions must be positive")
    if config.tile_m <= 0 or config.tile_n <= 0:
        raise ValueError("attention tile dimensions must be positive")
    if config.tile_m % 16 or config.tile_n % 16:
        raise ValueError("attention tile dimensions must be multiples of 16")


VARLEN_ATTENTION_POLICY = ComponentPolicy(
    component_id=VARLEN_ATTENTION,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(VarlenAttentionQuery.__dataclass_fields__),
    config_fields=frozenset(VarlenAttentionConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=VarlenAttentionConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = [
    "VARLEN_ATTENTION_POLICY",
    "VarlenAttentionConfig",
    "VarlenAttentionQuery",
]
