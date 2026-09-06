"""Compatibility alias for ModelOpt MXFP8 ``blockscaled`` calls.

New code should use ``b12x.gemm.blockscaled.pack_weight``, ``.mm``, and
``.prewarm``. This module retains its original names for compatibility.

Example:
    from b12x.gemm import blockscaled

    weight = blockscaled.pack_weight(w_mxfp8, w_scale)   # one-time
    out = blockscaled.mm(x, weight, expected_m=x.shape[0])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mxfp8_linear",
    group="gemm",
    api_style="oneshot",
    entry_points=("Weight", "mm", "mm_pair", "pack_weight", "is_supported"),
    dtypes=("bf16", "fp16"),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/mxfp8_linear.py",),
    ),
    test_path="tests/gemm/test_mxfp8_linear.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import Weight, is_supported, mm, mm_pair, pack_weight  # noqa: F401

install_lazy_api(globals(), META)
