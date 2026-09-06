"""Typed launch policy for block-FP8 linear planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    BLOCK_FP8_LINEAR,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True, kw_only=True)
class BlockFp8LinearQuery:
    max_tokens: int
    in_features: int
    out_features: int
    output_dtype: str


@dataclass(frozen=True, kw_only=True)
class BlockFp8LinearConfig:
    backend: str
    tile_m: int
    tile_n: int

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "BlockFp8LinearConfig":
        if set(payload) != {"backend", "tile_m", "tile_n"}:
            raise ValueError(
                "block-FP8 linear configs require backend, tile_m, and tile_n"
            )
        return cls(
            backend=str(payload["backend"]),
            tile_m=int(payload["tile_m"]),
            tile_n=int(payload["tile_n"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
        }


def _encode(query: BlockFp8LinearQuery) -> dict[str, object]:
    return {
        name: getattr(query, name)
        for name in BlockFp8LinearQuery.__dataclass_fields__
    }


def _heuristic(
    query: BlockFp8LinearQuery,
    _device: DeviceIdentity | None,
) -> BlockFp8LinearConfig:
    if query.max_tokens == 1:
        tile = (16, 64)
    elif query.max_tokens <= 8:
        tile = (16, 128)
    elif query.max_tokens <= 128 and query.out_features > 1_536:
        tile = (32, 128)
    elif query.max_tokens <= 128:
        tile = (64, 64)
    else:
        tile = (64, 128)
    return BlockFp8LinearConfig(
        backend="mxfp8",
        tile_m=tile[0],
        tile_n=tile[1],
    )


def _validate(
    query: BlockFp8LinearQuery,
    config: BlockFp8LinearConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend != "mxfp8":
        raise ValueError(f"unsupported block-FP8 backend {config.backend!r}")
    if (config.tile_m, config.tile_n) not in {
        (16, 64),
        (16, 128),
        (32, 64),
        (32, 128),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
    }:
        raise ValueError("unsupported block-FP8 MMA tile")
    if query.output_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported output dtype {query.output_dtype!r}")
    if query.in_features % 128:
        raise ValueError("block-FP8 in_features must be a multiple of 128")


BLOCK_FP8_LINEAR_POLICY = ComponentPolicy(
    component_id=BLOCK_FP8_LINEAR,
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(BlockFp8LinearQuery.__dataclass_fields__),
    config_fields=frozenset(BlockFp8LinearConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=BlockFp8LinearConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = [
    "BLOCK_FP8_LINEAR_POLICY",
    "BlockFp8LinearConfig",
    "BlockFp8LinearQuery",
]
