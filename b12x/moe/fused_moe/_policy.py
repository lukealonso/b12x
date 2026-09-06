"""Typed component-policy contract for fused-MoE decode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from b12x.policy import (
    MOE_DECODE,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)
@dataclass(frozen=True)
class MoeDecodeQuery:
    quant_mode: str
    source_format: str
    activation: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    num_tokens: int
    routed_rows: int

    def profile_fields(self) -> dict[str, object]:
        return {
            "activation": self.activation,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "num_tokens": self.num_tokens,
            "quant_mode": self.quant_mode,
            "routed_rows": self.routed_rows,
            "source_format": self.source_format,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class MoeDecodeConfig:
    backend: str
    route_planner: str
    max_active_clusters: int | None
    dynamic_tile_m: int | None = None
    dynamic_route_mode: str | None = None
    w4a16_route_mode: str | None = None

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "MoeDecodeConfig":
        expected = {
            "backend",
            "dynamic_route_mode",
            "dynamic_tile_m",
            "max_active_clusters",
            "route_planner",
            "w4a16_route_mode",
        }
        if set(payload) != expected:
            raise ValueError(
                "MoE decode profile fields must be exactly "
                f"{sorted(expected)}; got {sorted(payload)}"
            )
        backend = payload["backend"]
        route_planner = payload["route_planner"]
        max_active_clusters = payload["max_active_clusters"]
        dynamic_tile_m = payload["dynamic_tile_m"]
        dynamic_route_mode = payload["dynamic_route_mode"]
        w4a16_route_mode = payload["w4a16_route_mode"]
        if not isinstance(backend, str) or not isinstance(route_planner, str):
            raise TypeError("MoE backend and route_planner must be strings")
        if max_active_clusters is not None and (
            not isinstance(max_active_clusters, int)
            or isinstance(max_active_clusters, bool)
        ):
            raise TypeError("max_active_clusters must be an integer or null")
        if dynamic_tile_m is not None and (
            not isinstance(dynamic_tile_m, int) or isinstance(dynamic_tile_m, bool)
        ):
            raise TypeError("dynamic_tile_m must be an integer or null")
        if dynamic_route_mode is not None and not isinstance(dynamic_route_mode, str):
            raise TypeError("dynamic_route_mode must be a string or null")
        if w4a16_route_mode is not None and not isinstance(w4a16_route_mode, str):
            raise TypeError("w4a16_route_mode must be a string or null")
        return cls(
            backend=backend,
            route_planner=route_planner,
            max_active_clusters=max_active_clusters,
            dynamic_tile_m=dynamic_tile_m,
            dynamic_route_mode=dynamic_route_mode,
            w4a16_route_mode=w4a16_route_mode,
        )


def validate_moe_decode_config(
    query: MoeDecodeQuery,
    config: MoeDecodeConfig,
    _device: DeviceIdentity | None,
) -> None:
    if query.quant_mode == "nvfp4_auto":
        if query.source_format != "modelopt_nvfp4" or query.activation != "silu":
            raise ValueError("automatic MoE precision requires ModelOpt NVFP4 weights and SiLU")
        if config.backend == "w4a16" and config.w4a16_route_mode == "direct" and query.num_tokens > 8:
            raise ValueError("source-native A16 direct decode requires capacity at most 8")
        query = replace(
            query, quant_mode="w4a16" if config.backend == "w4a16" else "nvfp4",
        )
    if config.backend not in {"micro", "dynamic", "w4a16"}:
        raise ValueError(f"unsupported MoE backend {config.backend!r}")
    if query.quant_mode == "w4a16":
        if config.backend != "w4a16":
            raise ValueError("W4A16 queries require the W4A16 backend")
        if config.w4a16_route_mode not in {"direct", "packed"}:
            raise ValueError("W4A16 route mode must be 'direct' or 'packed'")
    else:
        if config.backend == "w4a16":
            raise ValueError("the W4A16 backend requires quant_mode='w4a16'")
        if config.w4a16_route_mode is not None:
            raise ValueError("w4a16_route_mode is only valid for W4A16")
    if query.quant_mode == "w6a8_mx" and config.backend != "dynamic":
        raise ValueError("W6A8-MX queries require the dynamic backend")
    if config.route_planner not in {"internal", "triton"}:
        raise ValueError(f"unsupported MoE route planner {config.route_planner!r}")
    if config.route_planner == "triton" and config.backend != "dynamic":
        raise ValueError("the Triton route planner requires dynamic MoE")
    if config.route_planner == "triton" and config.dynamic_route_mode != "grouped":
        raise ValueError("the Triton route planner requires grouped dynamic routing")
    if config.route_planner == "triton" and config.dynamic_tile_m != 16:
        raise ValueError("the Triton route planner requires dynamic_tile_m=16")
    if config.route_planner == "triton" and not (
        query.quant_mode == "nvfp4"
        and query.activation == "silu"
        and 0 < query.routed_rows <= 256
    ):
        raise ValueError(
            "the Triton route planner only supports small NVFP4 SiLU workloads"
        )
    if config.max_active_clusters is not None and config.max_active_clusters <= 0:
        raise ValueError("max_active_clusters must be positive when set")
    if config.route_planner != "triton" and config.max_active_clusters is not None:
        raise ValueError("max_active_clusters requires the Triton route planner")
    if config.backend == "dynamic":
        if config.dynamic_tile_m not in {16, 32, 64, 128}:
            raise ValueError("dynamic_tile_m must be one of 16, 32, 64, 128")
        if config.dynamic_route_mode not in {"direct", "grouped"}:
            raise ValueError("dynamic_route_mode must be 'direct' or 'grouped'")
    elif config.dynamic_tile_m is not None:
        raise ValueError("dynamic_tile_m is only valid for dynamic MoE")
    elif config.dynamic_route_mode is not None:
        raise ValueError("dynamic_route_mode is only valid for dynamic MoE")


def make_moe_decode_policy(
    heuristic: Callable[
        [MoeDecodeQuery, DeviceIdentity | None],
        MoeDecodeConfig,
    ],
) -> ComponentPolicy[MoeDecodeQuery, MoeDecodeConfig]:
    return ComponentPolicy(
        component_id=MOE_DECODE,
        query_schema_version=4,
        config_schema_version=3,
        query_fields=frozenset(
            {
                "activation",
                "hidden_size",
                "intermediate_size",
                "num_experts",
                "num_tokens",
                "quant_mode",
                "routed_rows",
                "source_format",
                "top_k",
            }
        ),
        config_fields=frozenset(
            {
                "backend",
                "dynamic_route_mode",
                "dynamic_tile_m",
                "max_active_clusters",
                "route_planner",
                "w4a16_route_mode",
            }
        ),
        encode_query=MoeDecodeQuery.profile_fields,
        decode_profile=MoeDecodeConfig.from_profile,
        heuristic=heuristic,
        validate_config=validate_moe_decode_config,
    )


def _heuristic(
    query: MoeDecodeQuery,
    device: DeviceIdentity | None,
) -> MoeDecodeConfig:
    from ._impl import _heuristic_moe_decode_config

    return _heuristic_moe_decode_config(query, device)


MOE_DECODE_POLICY = make_moe_decode_policy(_heuristic)


__all__ = [
    "MOE_DECODE_POLICY",
    "MoeDecodeConfig",
    "MoeDecodeQuery",
    "make_moe_decode_policy",
    "validate_moe_decode_config",
]
