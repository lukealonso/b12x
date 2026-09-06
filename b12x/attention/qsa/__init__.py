"""Grouped-selector sparse GQA for the Qwen3.8-Flash-Next QSA contract.

QSA keeps exact, original-token BF16 or globally scaled FP8 E4M3 GQA K/V and
uses a second compressed BF16 key cache only to select groups of logical token
positions. The public lifecycle is ``Caps -> plan -> bind -> prewarm -> run``.
Planning owns split and scratch policy; binding validates tensors and creates
references without allocating. ``prewarm`` compiles both request-ID widths and
both sparse-GQA row regimes without reading or mutating cache state and may be
omitted when first-use JIT latency is acceptable.

``run`` executes the decode transaction behind one opaque mutating dispatcher
boundary and never dispatches the slow functions in
:mod:`b12x.attention.qsa.reference`.  The bound main K/V cache and both page
tables are read-only.  The bound output, selected-position matrix, compressed
selector cache, and raw selector state are mutable.  The caller writes the
live original-token K/V before calling ``run``; QSA has no main-cache writer.

Raw selector state is indexed by persistent state slots, not batch indices.
``request_ids[row]`` selects a batch entry and ``raw_state_slot_ids[batch]``
selects its persistent slot; ``-1`` denotes padded work and forbids mutation.
Before assigning a fresh or recycled slot, the caller must fill its logical
and RoPE-position metadata with ``-1``.  Raw-key payload bytes need no
initialization because a key is readable only when its logical-position tag
matches the requested token.  A physical raw page stores the BF16 key payload
first, followed by bit-preserving int64 logical-position tags and RoPE
coordinates in the reserved tail.  ``cache_requirements`` reports whether the
complete raw page fits in one compressed-cache page; ``bind`` enforces that
condition only when the cache manager aliases their allocation slots.

When compressed and raw state share backing pages, every cached request keeps
owning its pages even while it has no rows in the current packed decode call.
Until eviction, the caller must retain that request's valid
``sequence_lengths``, ``compressed_block_table`` entries, and
``raw_state_slot_ids`` mapping.  Zero sequence lengths and ``-1`` table or slot
entries are reserved for unused or evicted capacity, not merely inactive
cached requests.

At the prefill-to-decode handoff for a first decode interval beginning at
logical position ``N``, the state-slot anchor is
``N - num_accepted_tokens``.  The raw ring must contain exact tagged raw keys
and RoPE positions for the trailing incomplete compression group; a decode row
that closes that group consumes the prefill state before overwriting the ring.
The ``-1`` anchor is reserved for initializing position zero with one accepted
token.

Bound cosine and sine tables accept positive-row-stride, unit-inner-stride
views, so the two halves of a combined RoPE table require no copies.  Dynamic
RoPE positions accept non-overlapping positive-stride views, including the
transpose of a native ``[axes, rows]`` MRoPE tensor.

Selected positions are request-relative original-token positions.  Every row
has fixed width ``budget + compress_ratio - 1``; valid positions are packed
first and unused entries are ``-1``.  Completed groups are expanded in their
selected order, followed by the causally visible incomplete-group tail.
The selected-position reader supports widths of at least 2051. Width is a
planned specialization; changing live rows preserves the warmed callable
within each split or direct reader regime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="qsa",
    group="attention",
    api_style="planned",
    entry_points=(
        "CacheRequirements",
        "Caps",
        "Plan",
        "Binding",
        "QsaConfig",
        "QsaQuery",
        "cache_requirements",
        "plan",
        "bind",
        "prewarm",
        "run",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("grouped_selector_sparse_gqa",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="3a437ab5168060e4d625f05e1625c04089f1ba37",
        paths=(
            "b12x/attention/dsa_indexer/",
            "b12x/attention/paged/",
            "b12x/attention/sparse_mla/",
        ),
    ),
    test_path="tests/attention/test_qsa_contract.py",
    since="1.3.0",
    notes=(
        "The Qwen sparse-GQA layout uses split CuTeDSL kernels for at most "
        "64 query rows. Larger prefills use the selected-position specialization "
        "of the CuTe paged-forward engine. "
        "Unsupported geometry fails closed. Main K/V cache writes are unsupported. "
        "Page- and state-slot-scaled addressing uses signed 64-bit arithmetic."
    ),
)

if TYPE_CHECKING:
    from .api import (
        Binding,
        CacheRequirements,
        Caps,
        Plan,
        QsaConfig,
        QsaQuery,
        bind,
        cache_requirements,
        is_supported,
        plan,
        prewarm,
        run,
    )

install_lazy_api(globals(), META)
