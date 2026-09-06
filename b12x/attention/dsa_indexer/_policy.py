"""Typed component policy for DSA indexer planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import DSA_INDEXER, ComponentPolicy, DeviceIdentity, FrozenMapping

# Cross-CTA merge arm of the fused paged indexer when a query row is split
# across cooperating CTAs. ``auto`` applies the planner default, ``cooperative``
# forces the grid-barrier radix at every live length, and ``serial`` forces the
# last-CTA reduction. The choice is a measured trade-off, so profiles may pin it
# per query shape.
FUSED_MERGE_AUTO = "auto"
FUSED_MERGE_COOPERATIVE = "cooperative"
FUSED_MERGE_SERIAL = "serial"
FUSED_MERGE_CHOICES = (FUSED_MERGE_AUTO, FUSED_MERGE_COOPERATIVE, FUSED_MERGE_SERIAL)

_BACKEND = "native"


@dataclass(frozen=True, kw_only=True)
class DsaIndexerQuery:
    source_layout: str
    mode: str
    dtype: str
    kv_dtype: str
    num_q_heads: int
    num_idx_heads: int
    max_q_rows: int
    max_k_rows: int
    top_k: int
    page_size: int
    score_mode: str
    shared_page_table: bool


@dataclass(frozen=True, kw_only=True)
class DsaIndexerConfig:
    """Selected implementation and tunable planner knobs for the DSA indexer.

    ``backend`` names the single production implementation. ``fused_merge``
    selects the fused paged indexer's cross-CTA merge arm; profiles written
    before the knob existed carry only ``backend`` and decode to ``auto``.
    """

    backend: str
    fused_merge: str = FUSED_MERGE_AUTO

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "DsaIndexerConfig":
        keys = set(payload)
        if "backend" not in keys or not keys <= {"backend", "fused_merge"}:
            raise ValueError(
                "DSA indexer profiles require backend and accept only fused_merge"
            )
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        fused_merge = payload.get("fused_merge", FUSED_MERGE_AUTO)
        if not isinstance(fused_merge, str):
            raise TypeError("fused_merge must be a string")
        return cls(backend=backend, fused_merge=fused_merge)

    def to_dict(self) -> dict[str, object]:
        return {"backend": self.backend, "fused_merge": self.fused_merge}


def _encode(query: DsaIndexerQuery) -> dict[str, object]:
    if not isinstance(query, DsaIndexerQuery):
        raise TypeError("query must be DsaIndexerQuery")
    return {name: getattr(query, name) for name in DsaIndexerQuery.__dataclass_fields__}


def _heuristic(
    _query: DsaIndexerQuery,
    _device: DeviceIdentity | None,
) -> DsaIndexerConfig:
    return DsaIndexerConfig(backend=_BACKEND, fused_merge=FUSED_MERGE_AUTO)


def _validate(
    _query: DsaIndexerQuery,
    config: DsaIndexerConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, DsaIndexerConfig):
        raise TypeError("config must be DsaIndexerConfig")
    if config.backend != _BACKEND:
        raise ValueError(f"unsupported {DSA_INDEXER} backend {config.backend!r}")
    if config.fused_merge not in FUSED_MERGE_CHOICES:
        raise ValueError(
            f"unsupported {DSA_INDEXER} fused_merge {config.fused_merge!r}; "
            f"expected one of {FUSED_MERGE_CHOICES}"
        )


DSA_INDEXER_POLICY = ComponentPolicy(
    component_id=DSA_INDEXER,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(DsaIndexerQuery.__dataclass_fields__),
    config_fields=frozenset(DsaIndexerConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=DsaIndexerConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = [
    "DSA_INDEXER_POLICY",
    "DsaIndexerConfig",
    "DsaIndexerQuery",
    "FUSED_MERGE_AUTO",
    "FUSED_MERGE_CHOICES",
    "FUSED_MERGE_COOPERATIVE",
    "FUSED_MERGE_SERIAL",
]
