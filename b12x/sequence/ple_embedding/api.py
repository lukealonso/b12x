"""Public surface for :mod:`b12x.sequence.ple_embedding`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from . import META
from ._contracts import (
    Binding,
    Caps,
    Plan,
    QuantMode,
    TableMemory,
    bind,
    plan,
    run,
)
from ._storage import TableStorage, allocate_storage
from ._policy import PleEmbeddingConfig, PleEmbeddingQuery


def is_supported(device=None) -> bool:
    """True when the registered b12x architecture can run this Triton op."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "QuantMode",
    "TableMemory",
    "TableStorage",
    "Caps",
    "Plan",
    "Binding",
    "PleEmbeddingConfig",
    "PleEmbeddingQuery",
    "plan",
    "allocate_storage",
    "bind",
    "run",
    "is_supported",
]
