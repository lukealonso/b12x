"""Typed component policy for GDN decode planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    GDN_ATTENTION,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True, kw_only=True)
class GdnQuery:
    gate_activation: str
    qk_l2norm: bool
    state_dtype: str
    key_heads: int
    value_heads: int
    max_seqs: int
    max_tokens: int
    state_index_columns: int

    def profile_fields(self) -> dict[str, object]:
        return {
            "gate_activation": self.gate_activation,
            "qk_l2norm": self.qk_l2norm,
            "state_dtype": self.state_dtype,
            "key_heads": self.key_heads,
            "value_heads": self.value_heads,
            "max_seqs": self.max_seqs,
            "max_tokens": self.max_tokens,
            "state_index_columns": self.state_index_columns,
        }


@dataclass(frozen=True, kw_only=True)
class GdnConfig:
    backend: str

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "GdnConfig":
        if set(payload) != {"backend"}:
            raise ValueError("GDN profiles require exactly backend")
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("GDN backend must be a string")
        return cls(backend=backend)


def _heuristic(
    query: GdnQuery,
    _device: DeviceIdentity | None,
) -> GdnConfig:
    backend = "triton" if query.key_heads == query.value_heads else "cutedsl"
    return GdnConfig(backend=backend)


def _validate(
    _query: GdnQuery,
    config: GdnConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend not in {"cutedsl", "triton"}:
        raise ValueError(f"unsupported GDN backend {config.backend!r}")


GDN_POLICY = ComponentPolicy(
    component_id=GDN_ATTENTION,
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(
        {
            "gate_activation",
            "qk_l2norm",
            "state_dtype",
            "key_heads",
            "value_heads",
            "max_seqs",
            "max_tokens",
            "state_index_columns",
        }
    ),
    config_fields=frozenset({"backend"}),
    encode_query=GdnQuery.profile_fields,
    decode_profile=GdnConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = ["GDN_POLICY", "GdnConfig", "GdnQuery"]
