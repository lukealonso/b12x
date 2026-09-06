"""Public surface for :mod:`b12x.sequence.gdn_decode`."""

from __future__ import annotations

from ..._lib.gating import has_cutlass_dsl, has_triton

from . import reference
from ._impl import (
    Binding,
    Caps,
    KdaBinding,
    Plan,
    bind,
    bind_kda,
    plan,
    run,
    run_kda,
)
from ._policy import GdnConfig, GdnQuery


def is_supported(device=None) -> bool:
    """True when mandatory Qwen CuTe and its Triton auxiliaries are usable."""
    del device
    return has_cutlass_dsl() and has_triton()


__all__ = [
    "Binding",
    "Caps",
    "GdnConfig",
    "GdnQuery",
    "KdaBinding",
    "Plan",
    "bind",
    "bind_kda",
    "is_supported",
    "plan",
    "reference",
    "run",
    "run_kda",
]
