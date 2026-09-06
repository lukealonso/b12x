"""Typed launch policy for MTP feedback fusion."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    MTP_FEEDBACK,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True, kw_only=True)
class MtpFeedbackQuery:
    dtype: str
    max_tokens: int
    hidden_size: int
    streams: int


@dataclass(frozen=True, kw_only=True)
class MtpFeedbackConfig:
    backend: str
    norm_block_h: int
    norm_block_s: int
    norm_num_warps: int

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "MtpFeedbackConfig":
        expected = {
            "backend",
            "norm_block_h",
            "norm_block_s",
            "norm_num_warps",
        }
        if set(payload) != expected:
            raise ValueError(f"MTP feedback config fields must be {expected}")
        return cls(
            backend=str(payload["backend"]),
            norm_block_h=int(payload["norm_block_h"]),
            norm_block_s=int(payload["norm_block_s"]),
            norm_num_warps=int(payload["norm_num_warps"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "norm_block_h": self.norm_block_h,
            "norm_block_s": self.norm_block_s,
            "norm_num_warps": self.norm_num_warps,
        }


def _encode(query: MtpFeedbackQuery) -> dict[str, object]:
    return {
        "dtype": query.dtype,
        "max_tokens": query.max_tokens,
        "hidden_size": query.hidden_size,
        "streams": query.streams,
    }


def _heuristic(
    query: MtpFeedbackQuery,
    _device: DeviceIdentity | None,
) -> MtpFeedbackConfig:
    norm_block_h = 1 << (query.hidden_size - 1).bit_length()
    norm_block_s = 1 << (query.streams - 1).bit_length()
    return MtpFeedbackConfig(
        backend="cutedsl",
        norm_block_h=norm_block_h,
        norm_block_s=norm_block_s,
        norm_num_warps=8 if norm_block_h >= 2048 else 4,
    )


def _validate(
    query: MtpFeedbackQuery,
    config: MtpFeedbackConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend != "cutedsl":
        raise ValueError(f"unsupported MTP feedback backend {config.backend!r}")
    if config.norm_block_h < query.hidden_size:
        raise ValueError("norm_block_h must cover hidden_size")
    if config.norm_block_s < query.streams:
        raise ValueError("norm_block_s must cover streams")
    for name, value in (
        ("norm_block_h", config.norm_block_h),
        ("norm_block_s", config.norm_block_s),
    ):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if config.norm_num_warps not in (1, 2, 4, 8):
        raise ValueError("norm_num_warps must be one of 1, 2, 4, or 8")


MTP_FEEDBACK_POLICY = ComponentPolicy(
    component_id=MTP_FEEDBACK,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(MtpFeedbackQuery.__dataclass_fields__),
    config_fields=frozenset(MtpFeedbackConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=MtpFeedbackConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = ["MTP_FEEDBACK_POLICY", "MtpFeedbackConfig", "MtpFeedbackQuery"]
