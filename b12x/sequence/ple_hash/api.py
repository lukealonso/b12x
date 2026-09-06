"""Public surface for :mod:`b12x.sequence.ple_hash`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._contracts import Binding, Caps, Plan, bind, plan, run
from ._policy import PleHashConfig, PleHashQuery


def is_supported(device=None) -> bool:
    """True on supported b12x devices with Triton available."""
    return default_is_supported(device, requires=("triton",))


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "PleHashConfig",
    "PleHashQuery",
    "plan",
    "bind",
    "run",
    "is_supported",
]
