"""Public surface for :mod:`b12x.sequence.ple`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._contracts import (
    LayerBinding as Binding,
    LayerCaps as Caps,
    LayerPlan as Plan,
    bind_layer as bind,
    plan_layer as plan,
    run_decode,
    run_mixed,
    run_prefill,
)
from ._policy import PleConfig, PleQuery


def is_supported(device=None) -> bool:
    """True on supported b12x devices with Triton available."""
    return default_is_supported(device, requires=("triton",))


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "PleConfig",
    "PleQuery",
    "plan",
    "bind",
    "run_decode",
    "run_mixed",
    "run_prefill",
    "is_supported",
]
