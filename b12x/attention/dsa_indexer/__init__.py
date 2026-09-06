"""DeepSeek Sparse Attention indexer for SM12x.

A three-stage pipeline whose outputs feed ``attention.sparse_mla`` /
``attention.compressed_sparse_mla``:

1. quantize — ``quantize_q_fp8`` (index-Q to FP8 e4m3, MXFP8 tiling;
   INDEX_HEAD_DIM 128).
2. score — MQA logits between index-Q and the FP8 index-K cache:
   ``logits_paged`` / ``logits_contiguous``, block-score reduction via
   ``block_scores_paged`` / ``block_scores_contiguous``.
3. select — ``topk_blocks`` / ``topk_tiled`` and the q2k index builders
   ``q2k_indices_decode`` / ``q2k_indices_prefill`` (+ query-position
   helpers).

The production paged DSA lifecycle is ``plan(Caps(...))`` -> ``bind`` ->
``run(binding)``. Planning owns route and scratch selection, binding captures
all live tensors without allocating, and execution launches the selected route.
Lower-level scorer and selector stages remain implementation facets for kernel
development rather than integration entry points.

Pure-torch semantics live in ``reference.py`` and ``msa_reference.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="dsa_indexer",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "plan",
        "bind",
        "run",
        "INDEX_HEAD_DIM",
        "PAGED_INDEX_PAGE_SIZE",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("dsv4", "glm_nsa", "msa"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/attention/dsa_indexer/",),
    ),
    test_path="tests/attention/test_dsa_indexer.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import *  # noqa: F401,F403

install_lazy_api(globals(), META)
