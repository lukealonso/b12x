"""Public surface for ``norm.hyperconnection``."""

from __future__ import annotations

from ..._lib.gating import has_cutlass_dsl, has_triton
from . import reference
from ._impl import HyperConnectionBinding as Binding
from ._impl import HyperConnectionCaps as Caps
from ._impl import HyperConnectionPlan as Plan
from ._impl import plan_hyperconnection as plan
from ._impl import run_combine_impl as run_combine
from ._impl import run_combine_norm_impl as run_combine_norm
from ._impl import run_gate_mean_impl as run_gate_mean
from ._impl import run_grouped_rmsnorm_impl as run_grouped_rmsnorm
from ._impl import run_scaled_silu_impl as run_scaled_silu
from ._policy import HyperConnectionConfig, HyperConnectionQuery


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind caller-owned capacity storage and expose live-prefix views."""
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True when the mandatory CuTe main path and Triton auxiliaries can run."""
    del device
    return has_cutlass_dsl() and has_triton()


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "HyperConnectionConfig",
    "HyperConnectionQuery",
    "plan",
    "bind",
    "run_grouped_rmsnorm",
    "run_scaled_silu",
    "run_gate_mean",
    "run_combine",
    "run_combine_norm",
    "reference",
    "is_supported",
]
