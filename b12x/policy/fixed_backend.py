"""Reusable policy contract for components with one production backend."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TypeVar

from .context import ComponentPolicy
from .types import DeviceIdentity, FrozenMapping

QueryT = TypeVar("QueryT")


@dataclass(frozen=True, kw_only=True)
class BackendConfig:
    """Selected implementation for a fixed-backend component."""

    backend: str

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "BackendConfig":
        if set(payload) != {"backend"}:
            raise ValueError("fixed-backend profiles require exactly backend")
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        return cls(backend=backend)

    def to_dict(self) -> dict[str, object]:
        return {"backend": self.backend}


def make_fixed_backend_policy(
    *,
    component_id: str,
    query_type: type[QueryT],
    backend: str,
) -> ComponentPolicy[QueryT, BackendConfig]:
    """Create a typed policy for a component with one valid implementation."""

    query_fields = tuple(field.name for field in fields(query_type))
    if not query_fields:
        raise ValueError("fixed-backend queries must expose at least one field")

    def encode_query(query: QueryT) -> dict[str, object]:
        if not isinstance(query, query_type):
            raise TypeError(f"query must be {query_type.__name__}")
        return {name: getattr(query, name) for name in query_fields}

    def heuristic(
        _query: QueryT,
        _device: DeviceIdentity | None,
    ) -> BackendConfig:
        return BackendConfig(backend=backend)

    def validate(
        _query: QueryT,
        config: BackendConfig,
        _device: DeviceIdentity | None,
    ) -> None:
        if not isinstance(config, BackendConfig):
            raise TypeError("config must be BackendConfig")
        if config.backend != backend:
            raise ValueError(f"unsupported {component_id} backend {config.backend!r}")

    return ComponentPolicy(
        component_id=component_id,
        query_schema_version=1,
        config_schema_version=1,
        query_fields=frozenset(query_fields),
        config_fields=frozenset({"backend"}),
        encode_query=encode_query,
        decode_profile=BackendConfig.from_profile,
        heuristic=heuristic,
        validate_config=validate,
    )


__all__ = ["BackendConfig", "make_fixed_backend_policy"]
