"""Typed launch policy for BF16 vocabulary projections."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import ComponentPolicy, DeviceIdentity, FrozenMapping
from b12x.policy.components import BF16_VOCAB_PROJECTION

MAX_IN_FEATURES = 8_192
MIN_TRITON_OUT_FEATURES = 16_384
_TRITON_WARPS = frozenset((1, 2, 4, 8))
_LOOP_BLOCKS = frozenset((256, 512, 1_024))


@dataclass(frozen=True, kw_only=True)
class Bf16VocabProjectionQuery:
    dtype: str
    max_tokens: int
    in_features: int
    out_features: int


@dataclass(frozen=True, kw_only=True)
class Bf16VocabProjectionConfig:
    backend: str
    algorithm: str
    block_k: int
    num_warps: int

    @classmethod
    def from_profile(
        cls,
        payload: FrozenMapping,
    ) -> "Bf16VocabProjectionConfig":
        expected = {"backend", "algorithm", "block_k", "num_warps"}
        if set(payload) != expected:
            raise ValueError(
                "BF16 vocabulary projection configs require backend, "
                "algorithm, block_k, and num_warps"
            )
        return cls(
            backend=str(payload["backend"]),
            algorithm=str(payload["algorithm"]),
            block_k=int(payload["block_k"]),
            num_warps=int(payload["num_warps"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "algorithm": self.algorithm,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
        }


def _encode(query: Bf16VocabProjectionQuery) -> dict[str, object]:
    return {
        name: getattr(query, name)
        for name in Bf16VocabProjectionQuery.__dataclass_fields__
    }


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _heuristic(
    query: Bf16VocabProjectionQuery,
    device: DeviceIdentity | None,
) -> Bf16VocabProjectionConfig:
    supported_device = device is not None and device.compute_capability in {
        (12, 0),
        (12, 1),
    }
    if (
        supported_device
        and query.dtype == "bfloat16"
        and query.max_tokens == 1
        and 0 < query.in_features <= MAX_IN_FEATURES
        and query.out_features >= MIN_TRITON_OUT_FEATURES
    ):
        return Bf16VocabProjectionConfig(
            backend="triton",
            algorithm="row",
            block_k=_next_power_of_two(query.in_features),
            num_warps=8,
        )
    return Bf16VocabProjectionConfig(
        backend="torch",
        algorithm="torch",
        block_k=0,
        num_warps=0,
    )


def _validate(
    query: Bf16VocabProjectionQuery,
    config: Bf16VocabProjectionConfig,
    _device: DeviceIdentity | None,
) -> None:
    if query.dtype != "bfloat16":
        raise ValueError(f"unsupported vocabulary projection dtype {query.dtype!r}")
    if query.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if query.in_features <= 0 or query.out_features <= 0:
        raise ValueError("projection dimensions must be positive")
    if config.backend == "torch":
        if (config.algorithm, config.block_k, config.num_warps) != (
            "torch",
            0,
            0,
        ):
            raise ValueError("torch projection configs cannot carry Triton knobs")
        return
    if config.backend != "triton":
        raise ValueError(f"unsupported projection backend {config.backend!r}")
    if query.max_tokens != 1:
        raise ValueError("the Triton vocabulary GEMV requires max_tokens=1")
    if query.in_features > MAX_IN_FEATURES:
        raise ValueError(f"the Triton vocabulary GEMV supports K <= {MAX_IN_FEATURES}")
    if config.num_warps not in _TRITON_WARPS:
        raise ValueError(f"unsupported Triton warp count {config.num_warps}")
    if config.algorithm == "row":
        if (
            config.block_k < query.in_features
            or config.block_k > MAX_IN_FEATURES
            or config.block_k & (config.block_k - 1)
        ):
            raise ValueError("row block_k must be a covering power of two")
    elif config.algorithm == "loop":
        if config.block_k not in _LOOP_BLOCKS:
            raise ValueError(f"unsupported loop block_k {config.block_k}")
    else:
        raise ValueError(f"unsupported Triton algorithm {config.algorithm!r}")


BF16_VOCAB_PROJECTION_POLICY = ComponentPolicy(
    component_id=BF16_VOCAB_PROJECTION,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(Bf16VocabProjectionQuery.__dataclass_fields__),
    config_fields=frozenset(Bf16VocabProjectionConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=Bf16VocabProjectionConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = [
    "BF16_VOCAB_PROJECTION_POLICY",
    "Bf16VocabProjectionConfig",
    "Bf16VocabProjectionQuery",
    "MAX_IN_FEATURES",
    "MIN_TRITON_OUT_FEATURES",
]
