"""Typed launch policy for HyperConnection primitives."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    HYPERCONNECTION,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True, kw_only=True)
class HyperConnectionQuery:
    dtype: str
    max_tokens: int
    hidden_size: int
    streams: int
    lowrank: int


@dataclass(frozen=True, kw_only=True)
class HyperConnectionConfig:
    backend: str
    reduction_block_h: int
    pointwise_block: int
    reduction_num_warps: int

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "HyperConnectionConfig":
        expected = {
            "backend",
            "reduction_block_h",
            "pointwise_block",
            "reduction_num_warps",
        }
        if set(payload) != expected:
            raise ValueError(f"HyperConnection config fields must be {expected}")
        return cls(
            backend=str(payload["backend"]),
            reduction_block_h=int(payload["reduction_block_h"]),
            pointwise_block=int(payload["pointwise_block"]),
            reduction_num_warps=int(payload["reduction_num_warps"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "reduction_block_h": self.reduction_block_h,
            "pointwise_block": self.pointwise_block,
            "reduction_num_warps": self.reduction_num_warps,
        }


def _encode(query: HyperConnectionQuery) -> dict[str, object]:
    return {
        "dtype": query.dtype,
        "max_tokens": query.max_tokens,
        "hidden_size": query.hidden_size,
        "streams": query.streams,
        "lowrank": query.lowrank,
    }


def _heuristic(
    query: HyperConnectionQuery,
    _device: DeviceIdentity | None,
) -> HyperConnectionConfig:
    reduction_block_h = 1 << (query.hidden_size - 1).bit_length()
    return HyperConnectionConfig(
        backend="cutedsl",
        reduction_block_h=reduction_block_h,
        pointwise_block=256,
        reduction_num_warps=8 if reduction_block_h >= 2048 else 4,
    )


def _validate(
    query: HyperConnectionQuery,
    config: HyperConnectionConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend != "cutedsl":
        raise ValueError(f"unsupported HyperConnection backend {config.backend!r}")
    if config.reduction_block_h < query.hidden_size:
        raise ValueError("reduction_block_h must cover hidden_size")
    for name, value in (
        ("reduction_block_h", config.reduction_block_h),
        ("pointwise_block", config.pointwise_block),
    ):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if config.reduction_num_warps not in (1, 2, 4, 8):
        raise ValueError("reduction_num_warps must be one of 1, 2, 4, or 8")


HYPERCONNECTION_POLICY = ComponentPolicy(
    component_id=HYPERCONNECTION,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(HyperConnectionQuery.__dataclass_fields__),
    config_fields=frozenset(HyperConnectionConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=HyperConnectionConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = [
    "HYPERCONNECTION_POLICY",
    "HyperConnectionConfig",
    "HyperConnectionQuery",
]
