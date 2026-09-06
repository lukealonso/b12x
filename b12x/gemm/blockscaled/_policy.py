"""Static-geometry precision policy for one-shot block-scaled GEMM.

Resolution returns exact row-count routes. Live M selects within the
resolved config and never enters a policy or kernel resolution cache key.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from b12x.policy import ComponentPolicy, FrozenMapping, get_auto_policy

from b12x.policy.components import BLOCKSCALED_PRECISION
A16_CONFIGS = tuple((n, k, s) for n in (64, 128) for k in (64, 128) for s in (1, 2, 4, 8))


@dataclass(frozen=True, kw_only=True)
class BlockscaledQuery:
    recipe: str
    in_features: int
    out_features: int


@dataclass(frozen=True, kw_only=True)
class BlockscaledConfig:
    # Each entry is (M, N tile, K tile, split-K).
    a16_rows: tuple[tuple[int, int, int, int], ...] = ()

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> BlockscaledConfig:
        if set(payload) != {"a16_rows"}:
            raise ValueError("blockscaled precision config requires a16_rows")
        rows = tuple(tuple(row) for row in payload["a16_rows"])
        if any(len(row) != 4 or any(type(v) is not int for v in row) for row in rows):
            raise ValueError("a16_rows must contain integer (M, N tile, K tile, split-K) entries")
        return cls(a16_rows=rows)

    def to_dict(self):
        return {"a16_rows": [list(row) for row in self.a16_rows]}

    def select(self, m: int):
        return next((row[1:] for row in self.a16_rows if row[0] == m), None)


def _validate(query, config, device):
    if query.recipe not in ("nvfp4", "mxfp8") or min(query.in_features, query.out_features) <= 0:
        raise ValueError("invalid blockscaled precision geometry or recipe")
    if query.in_features % (16 if query.recipe == "nvfp4" else 32):
        raise ValueError("blockscaled precision K must be block-aligned")
    rows = config.a16_rows
    if any(len(row) != 4 or any(type(v) is not int for v in row)
           or row[0] <= 0 or row[1:] not in A16_CONFIGS for row in rows):
        raise ValueError("invalid blockscaled A16 row configuration")
    if tuple(row[0] for row in rows) != tuple(sorted({row[0] for row in rows})):
        raise ValueError("blockscaled A16 row counts must be unique and sorted")
    if rows and (query.in_features % 32 or query.out_features % 8):
        raise ValueError("A16 precision routes require K divisible by 32 and N by 8")
    if rows and (device is None or device.compute_capability not in ((12, 0), (12, 1))):
        raise ValueError("blockscaled A16 routes require SM120/SM121")


def _heuristic(query, device):
    if (device is not None and device.compute_capability in ((12, 0), (12, 1))
            and query.recipe in ("nvfp4", "mxfp8")
            and query.in_features % 32 == 0 and query.out_features % 8 == 0):
        return BlockscaledConfig(a16_rows=tuple(
            (m, 128, 64, 4) for m in range(1, 9)
        ))
    return BlockscaledConfig()


BLOCKSCALED_POLICY = ComponentPolicy(
    component_id=BLOCKSCALED_PRECISION,
    query_schema_version=1, config_schema_version=1,
    query_fields=frozenset(BlockscaledQuery.__dataclass_fields__),
    config_fields=frozenset(BlockscaledConfig.__dataclass_fields__),
    encode_query=lambda query: dict(vars(query)),
    decode_profile=BlockscaledConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


@lru_cache(maxsize=4096)
def resolve_precision(device, recipe, k, n):
    from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
    raise_if_kernel_resolution_frozen("blockscaled precision policy")
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("blockscaled precision policy must be resolved before CUDA graph capture")
    return get_auto_policy(device).resolve(
        BLOCKSCALED_POLICY, BlockscaledQuery(recipe=recipe, in_features=k, out_features=n)
    )
