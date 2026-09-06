"""Policy contract for chunked KDA prefill: query, config, heuristic."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import ComponentPolicy
from b12x.policy.components import KDA_PREFILL
from b12x.policy.types import FrozenMapping

BACKEND = "cutedsl"
V_SPLIT_CHOICES = (16, 32, 64, 128)
K_SPLIT_CHOICES = (1, 2, 4)
STAGE_CHOICES = (2, 3, 4)
CHUNK_TOKENS = 16


class WorkspaceRecord:
    """Byte layout of one prepared (tile, head) record in the workspace ring.

    The recurrence kernel copies ``[0, HEAD_BYTES)`` into a pipeline stage
    with one bulk copy; the stage then holds this CTA's value rows from
    offset ``V`` as ``v_split // 16`` groups of ``[16 tokens x 16 values]``,
    copied straight from the value tensor. The operand tiles are stored in
    the swizzled 16-byte-chunk order the consumer's ldmatrix reads.
    """

    Q_TILDE = 0
    K_TILDE = 4096
    K_R = 8192
    INV = 12288
    MQK = 12800
    LAMBDA_C = 13312
    BETA = 13824
    HEAD_BYTES = 13952
    BYTES = 14080
    V = 14080


WORKSPACE_RECORD_BYTES = WorkspaceRecord.BYTES
# Prepared-tile bytes one window may occupy; two windows stay L2 resident.
WINDOW_BYTES_BUDGET = 36 << 20


def tiles_capacity(max_tokens: int, max_seqs: int) -> int:
    """Upper bound on packed chunk tiles: one partial tile per sequence."""
    return -(-int(max_tokens) // CHUNK_TOKENS) + int(max_seqs)


def default_window_tiles(heads: int, max_tokens: int, max_seqs: int) -> int:
    """Tiles per pipeline window so a window's prepared tiles fit the L2 budget."""
    per_row = int(heads) * WORKSPACE_RECORD_BYTES
    return max(1, min(tiles_capacity(max_tokens, max_seqs), WINDOW_BYTES_BUDGET // per_row))


@dataclass(frozen=True, kw_only=True)
class KdaPrefillQuery:
    """Immutable geometry and planned capacity of one KDA prefill plan."""

    heads: int
    head_dim: int
    model_dtype: str
    state_dtype: str
    qk_l2norm: bool
    checkpoint_export: bool
    max_tokens: int
    max_seqs: int

    def profile_fields(self) -> dict[str, object]:
        return {
            "heads": int(self.heads),
            "head_dim": int(self.head_dim),
            "model_dtype": str(self.model_dtype),
            "state_dtype": str(self.state_dtype),
            "qk_l2norm": bool(self.qk_l2norm),
            "checkpoint_export": bool(self.checkpoint_export),
            "max_tokens": int(self.max_tokens),
            "max_seqs": int(self.max_seqs),
        }


@dataclass(frozen=True)
class KdaPrefillConfig:
    """Backend selection plus the recurrence kernel's launch geometry.

    ``v_split`` is the number of value rows one recurrence CTA owns (smaller
    splits launch more CTAs per sequence and head at the cost of re-reading
    the prepared tiles from L2); ``k_split`` is how many warps share each
    sixteen-row group by splitting its key columns (more warps shorten the
    per-tile tensor-core chain at the cost of shared-memory reductions);
    ``stages`` is the tile prefetch depth; ``window_tiles`` is the number of
    consecutive banded tile positions one pipeline window covers. The prepare
    kernel of a window runs concurrently with the recurrence of that window
    and the next window's prepare, and two windows of prepared records form
    the workspace ring, so the window size bounds the ring's footprint.
    """

    backend: str = BACKEND
    v_split: int = 64
    k_split: int = 1
    stages: int = 3
    window_tiles: int = 64

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "KdaPrefillConfig":
        keys = set(payload.keys())
        if "backend" not in keys or not keys <= {
            "backend", "v_split", "k_split", "stages", "window_tiles"
        }:
            raise ValueError(
                "KDA prefill profiles require backend and accept only v_split, "
                "k_split, stages, and window_tiles"
            )
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        values = {}
        for name, default in (("v_split", 64), ("k_split", 1), ("stages", 3), ("window_tiles", 64)):
            value = payload.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            values[name] = int(value)
        return cls(backend=backend, **values)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "v_split": int(self.v_split),
            "k_split": int(self.k_split),
            "stages": int(self.stages),
            "window_tiles": int(self.window_tiles),
        }


def _heuristic(query: KdaPrefillQuery, device) -> KdaPrefillConfig:
    del device
    return KdaPrefillConfig(
        backend=BACKEND,
        v_split=64,
        k_split=1,
        stages=3,
        window_tiles=default_window_tiles(query.heads, query.max_tokens, query.max_seqs),
    )


def _validate(query: KdaPrefillQuery, config: KdaPrefillConfig, device) -> None:
    del device
    if config.backend != BACKEND:
        raise ValueError(f"unsupported {KDA_PREFILL} backend {config.backend!r}")
    if config.v_split not in V_SPLIT_CHOICES:
        raise ValueError(
            f"unsupported {KDA_PREFILL} v_split {config.v_split!r}; expected one "
            f"of {V_SPLIT_CHOICES}"
        )
    if config.k_split not in K_SPLIT_CHOICES:
        raise ValueError(
            f"unsupported {KDA_PREFILL} k_split {config.k_split!r}; expected one "
            f"of {K_SPLIT_CHOICES}"
        )
    if config.stages not in STAGE_CHOICES:
        raise ValueError(
            f"unsupported {KDA_PREFILL} stages {config.stages!r}; expected one of "
            f"{STAGE_CHOICES}"
        )
    if 2 * config.v_split * config.k_split + 32 > 1024:
        raise ValueError(f"{KDA_PREFILL} v_split x k_split exceeds the thread limit")
    if isinstance(config.window_tiles, bool) or int(config.window_tiles) < 1:
        raise ValueError(f"{KDA_PREFILL} window_tiles must be a positive integer")
    if query.head_dim != 128:
        raise ValueError(f"{KDA_PREFILL} requires head_dim 128, got {query.head_dim}")
    if query.model_dtype != "bfloat16" or query.state_dtype != "float32":
        raise ValueError(
            f"{KDA_PREFILL} requires bfloat16 activations and float32 state, got "
            f"{query.model_dtype}/{query.state_dtype}"
        )


KDA_PREFILL_POLICY = ComponentPolicy(
    component_id=KDA_PREFILL,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(
        {
            "heads",
            "head_dim",
            "model_dtype",
            "state_dtype",
            "qk_l2norm",
            "checkpoint_export",
            "max_tokens",
            "max_seqs",
        }
    ),
    config_fields=frozenset({"backend", "v_split", "k_split", "stages", "window_tiles"}),
    encode_query=KdaPrefillQuery.profile_fields,
    decode_profile=KdaPrefillConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)

__all__ = [
    "BACKEND",
    "CHUNK_TOKENS",
    "K_SPLIT_CHOICES",
    "KDA_PREFILL_POLICY",
    "KdaPrefillConfig",
    "KdaPrefillQuery",
    "STAGE_CHOICES",
    "V_SPLIT_CHOICES",
    "WINDOW_BYTES_BUDGET",
    "WORKSPACE_RECORD_BYTES",
    "WorkspaceRecord",
    "default_window_tiles",
    "tiles_capacity",
]
