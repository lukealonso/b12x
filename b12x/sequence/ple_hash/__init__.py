"""Prime-hashed learned n-gram embedding IDs.

``plan`` owns the immutable signed-int64 multiplier and per-head table
geometry together with serving capacity. ``bind`` maps caller-owned runtime
tensors and scratch without allocation. ``run`` will write logical embedding
IDs while leaving committed history immutable.

Exact PyTorch oracles are available from
:mod:`b12x.sequence.ple_hash.reference` and are never runtime fallbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="ple_hash",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "PleHashConfig",
        "PleHashQuery",
        "plan",
        "bind",
        "run",
        "is_supported",
    ),
    dtypes=("int64",),
    recipes=("packed_eos_bounded",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="3a437ab51680",
        paths=(
            "b12x/_lib/scratch.py",
            "b12x/_lib/scratch_layout.py",
        ),
    ),
    test_path="tests/sequence/test_ple.py",
    since="1.3.0",
    notes=(
        "Packed hashing uses fixed-capacity caller-owned scratch and exposes "
        "a device error code for graph-safe metadata validation. The Triton "
        "implementation is a correctness reference and is not "
        "throughput-qualified."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        PleHashConfig,
        PleHashQuery,
        bind,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
