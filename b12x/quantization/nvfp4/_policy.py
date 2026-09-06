"""Typed component policy for NVFP4 activation quantization."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import ComponentPolicy, DeviceIdentity, FrozenMapping
from b12x.policy.components import NVFP4_QUANTIZATION


@dataclass(frozen=True, kw_only=True)
class Nvfp4QuantizationQuery:
    dtype: str
    rows: int
    columns: int


@dataclass(frozen=True, kw_only=True)
class Nvfp4QuantizationConfig:
    backend: str
    liveness_strategy: str

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "Nvfp4QuantizationConfig":
        if set(payload) != {"backend", "liveness_strategy"}:
            raise ValueError("NVFP4 configs require backend and liveness_strategy")
        return cls(
            backend=str(payload["backend"]),
            liveness_strategy=str(payload["liveness_strategy"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "liveness_strategy": self.liveness_strategy,
        }


def _encode(query: Nvfp4QuantizationQuery) -> dict[str, object]:
    return {
        "dtype": query.dtype,
        "rows": query.rows,
        "columns": query.columns,
    }


def _heuristic(
    query: Nvfp4QuantizationQuery,
    _device: DeviceIdentity | None,
) -> Nvfp4QuantizationConfig:
    return Nvfp4QuantizationConfig(
        backend="cutedsl",
        liveness_strategy="retain" if query.rows == 128 else "packed",
    )


def _validate(
    query: Nvfp4QuantizationQuery,
    config: Nvfp4QuantizationConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend != "cutedsl":
        raise ValueError(f"unsupported NVFP4 backend {config.backend!r}")
    if config.liveness_strategy not in {"retain", "packed"}:
        raise ValueError("NVFP4 liveness_strategy must be retain or packed")
    if query.dtype != "bfloat16":
        raise ValueError("NVFP4 quantization requires bfloat16 input")
    if query.rows <= 0 or query.columns <= 0:
        raise ValueError("NVFP4 quantization dimensions must be positive")
    if query.rows % 128 or query.columns % 128:
        raise ValueError("NVFP4 quantization dimensions must be multiples of 128")


NVFP4_QUANTIZATION_POLICY = ComponentPolicy(
    component_id=NVFP4_QUANTIZATION,
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(Nvfp4QuantizationQuery.__dataclass_fields__),
    config_fields=frozenset(Nvfp4QuantizationConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=Nvfp4QuantizationConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = [
    "NVFP4_QUANTIZATION_POLICY",
    "Nvfp4QuantizationConfig",
    "Nvfp4QuantizationQuery",
]
