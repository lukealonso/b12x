"""Public surface for comm.roce (docs in the op ``__init__``)."""

from __future__ import annotations

from .roce_oneshot import (
    API_VERSION,
    DEFAULT_MAX_GATHER_BYTES,
    DEFAULT_MAX_SIZE,
    SUPPORTED_DTYPES,
    SUPPORTED_WORLD_SIZES,
    RoceOneshotAllReduce as AllReduce,
    default_gid_index,
    discover_hcas,
    is_supported,
)

__all__ = [
    "API_VERSION",
    "AllReduce",
    "DEFAULT_MAX_GATHER_BYTES",
    "DEFAULT_MAX_SIZE",
    "SUPPORTED_DTYPES",
    "SUPPORTED_WORLD_SIZES",
    "default_gid_index",
    "discover_hcas",
    "is_supported",
]
