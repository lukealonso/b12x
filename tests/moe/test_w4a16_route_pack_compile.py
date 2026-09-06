"""Device-free Triton compilation of the W4A16 route-packing kernels.

Each kernel is compiled to cubin for the served SM 12.0 target with the
constexpr values ``pack_topk_routes_by_expert`` launches, on a host with no
GPU: ``triton.compile`` runs the frontend type-check and ptxas, neither of
which needs a device.

This is the gate the Triton interpreter cannot provide. Under
``TRITON_INTERPRET=1`` a Python ``if`` on a runtime value executes one branch
eagerly, so a kernel whose branches produce tensors of different shape or
dtype interprets correctly and still fails to compile ("Mismatched type for
<name> between then block and else block"). The kernel tests in
``test_w4a16_stable_route_pack.py`` prove the layout under the interpreter;
this module proves that every kernel instance those launches select is
compilable, so a shape or dtype join defect surfaces without a GPU.

Run it in its own process with the interpreter off (the module skips
otherwise), for example::

    python -m pytest -q tests/moe/test_w4a16_route_pack_compile.py
"""

from __future__ import annotations

import os

import pytest
import triton
import triton.language as tl

import b12x.moe._shared.kernels.w4a16.route_pack as route_pack_module

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") == "1",
    reason="triton.compile needs the real JIT frontend; run without TRITON_INTERPRET",
)


def _target():
    from triton.backends.compiler import GPUTarget

    # The served ranks are SM 12.0 with 32-lane warps. The target is fixed
    # rather than read from a device so the check runs on CPU-only hosts.
    return GPUTarget("cuda", 120, 32)


def _compile(fn, args: dict[str, object], *, num_warps: int):
    """Compile ``fn`` for the served target.

    ``args`` maps every kernel parameter either to a Triton signature string
    (``"*i32"``, ``"i32"``) or to the constexpr value it is launched with; a
    parameter added to a kernel without a matching entry raises here, which
    keeps the check exhaustive.
    """
    from triton.compiler import ASTSource

    signature: dict[str, str] = {}
    constexprs: dict[str, object] = {}
    for name in fn.arg_names:
        value = args[name]
        if isinstance(value, str):
            signature[name] = value
        else:
            signature[name] = "constexpr"
            constexprs[name] = value
    source = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(source, target=_target(), options={"num_warps": num_warps})


# The served Kimi-K3 prefill chunk: 4,608 tokens x top-16 over 896 experts at
# route block 48, i.e. 73,728 routes.
_NUM_EXPERTS = 896
_LIVE_ROUTES = 73728
_BLOCK_SIZE = 48
_TOPK_PTR = ("*i32", "*i64")

# Route packing is launch-latency work, not tiled GEMM work: every kernel here
# is expected to fit in the 48 KiB of shared memory a CUDA block gets without
# opting in to the larger per-block limit. The budget is a policy of these
# kernels, not a hardware cap; ``triton.compile`` reports the requirement in
# ``metadata.shared`` and only checks it against the device at launch, so
# holding the budget here keeps the check on CPU.
_SHARED_MEMORY_BUDGET = 48 * 1024


def _kernel_cases() -> list[tuple[str, object, dict[str, object], int]]:
    """Every (name, kernel, launch constexprs, num_warps) the packer selects."""
    module = route_pack_module
    packed_routes = _LIVE_ROUTES + _NUM_EXPERTS * (_BLOCK_SIZE - 1)
    route_blocks = -(-packed_routes // _BLOCK_SIZE)
    block_e = 1024  # next power of two at or above 896 experts
    cases: list[tuple[str, object, dict[str, object], int]] = []
    for topk_ptr in _TOPK_PTR:
        suffix = "i64" if topk_ptr == "*i64" else "i32"
        for has_map in (False, True):
            tag = f"{suffix}-map{int(has_map)}"
            cases.append(
                (
                    f"route_count-{tag}",
                    module._w4a16_route_count_kernel,
                    {
                        "topk_ids": topk_ptr,
                        "expert_map": "*i32",
                        "counts": "*i32",
                        "live_numel": "i32",
                        "NUM_EXPERTS": _NUM_EXPERTS,
                        "HAS_EXPERT_MAP": has_map,
                        "BLOCK_T": module._FAST_COUNT_BLOCK_T,
                    },
                    4,
                )
            )
            cases.append(
                (
                    f"sort-{tag}",
                    module._pack_topk_routes_sort_kernel,
                    {
                        "topk_ids": topk_ptr,
                        "expert_map": "*i32",
                        "packed_route_indices": "*i32",
                        "expert_offsets": "*i32",
                        "live_numel": "i32",
                        "NUM_EXPERTS": _NUM_EXPERTS,
                        "HAS_EXPERT_MAP": has_map,
                        "BLOCK_T": module._SORT_BLOCK_T,
                    },
                    4,
                )
            )
            cases.append(
                (
                    f"stable_scan-{tag}",
                    module._pack_topk_routes_stable_kernel,
                    {
                        "topk_ids": topk_ptr,
                        "expert_map": "*i32",
                        "packed_route_indices": "*i32",
                        "expert_offsets": "*i32",
                        "live_numel": "i32",
                        "NUMEL_CAPACITY": _LIVE_ROUTES,
                        "NUM_EXPERTS": _NUM_EXPERTS,
                        "HAS_EXPERT_MAP": has_map,
                        "BLOCK_T": module._STABLE_SORT_BLOCK_T,
                        "EXPERTS_PER_PROGRAM": module._STABLE_SORT_EXPERTS_PER_PROGRAM,
                    },
                    8,
                )
            )
            cases.append(
                (
                    f"segment_scan-{tag}",
                    module._pack_topk_routes_segment_scan_kernel,
                    {
                        "topk_ids": topk_ptr,
                        "expert_map": "*i32",
                        "packed_route_indices": "*i32",
                        "expert_offsets": "*i32",
                        "expert_counts": "*i32",
                        "live_numel": "i32",
                        "NUMEL_CAPACITY": _LIVE_ROUTES,
                        "NUM_EXPERTS": _NUM_EXPERTS,
                        "HAS_EXPERT_MAP": has_map,
                        "COUNT_MIN": module._STABLE_SEGMENT_SORT_WIDTHS[-1][0],
                        "BLOCK_T": module._STABLE_SEGMENT_SCAN_BLOCK_T,
                    },
                    module._STABLE_SEGMENT_SCAN_WARPS,
                )
            )
            # Decode-sized single-launch packing: 8 tokens x top-8 over 128
            # experts at block 16 keeps the vector extents under the caps in
            # pack_topk_routes_by_expert.
            cases.append(
                (
                    f"small_prefix-{tag}",
                    module._pack_topk_routes_small_prefix_kernel,
                    {
                        "topk_ids": topk_ptr,
                        "expert_map": "*i32",
                        "packed_route_indices": "*i32",
                        "block_expert_ids": "*i32",
                        "packed_route_count": "*i32",
                        "expert_offsets": "*i32",
                        "expert_counts": "*i32",
                        "live_numel": "i32",
                        "NUMEL_CAPACITY": 64,
                        "BLOCK_SIZE": 16,
                        "NUM_EXPERTS": 128,
                        "MAX_PACKED_ROUTES": 1088,
                        "MAX_ROUTE_BLOCKS": 68,
                        "HAS_EXPERT_MAP": has_map,
                        "BLOCK_E": 128,
                        "BLOCK_T": module._COUNT_BLOCK_T,
                        "BLOCK_ROUTE_INIT": 2048,
                        "BLOCK_M": 128,
                        "SEARCH_STEPS": 8,
                    },
                    8,
                )
            )
    cases.append(
        (
            "route_prefix",
            route_pack_module._w4a16_route_prefix_from_counts_kernel,
            {
                "counts": "*i32",
                "packed_route_count": "*i32",
                "expert_offsets": "*i32",
                "BLOCK_SIZE": _BLOCK_SIZE,
                "NUM_EXPERTS": _NUM_EXPERTS,
                "BLOCK_E": block_e,
            },
            4,
        )
    )
    cases.append(
        (
            "post_prefix",
            route_pack_module._pack_topk_routes_post_prefix_kernel,
            {
                "packed_route_indices": "*i32",
                "block_expert_ids": "*i32",
                "expert_offsets": "*i32",
                "live_numel": "i32",
                "BLOCK_SIZE": _BLOCK_SIZE,
                "NUM_EXPERTS": _NUM_EXPERTS,
                "MAX_PACKED_ROUTES": packed_routes,
                "MAX_ROUTE_BLOCKS": route_blocks,
                "BLOCK_T": route_pack_module._POST_PREFIX_BLOCK_T,
                "SEARCH_STEPS": block_e.bit_length(),
            },
            4,
        )
    )
    for width, num_warps in route_pack_module._STABLE_SEGMENT_SORT_WIDTHS:
        cases.append(
            (
                f"segment_sort-{width}",
                route_pack_module._pack_topk_routes_segment_sort_kernel,
                {
                    "packed_route_indices": "*i32",
                    "expert_offsets": "*i32",
                    "expert_counts": "*i32",
                    "COUNT_MIN": 0,
                    "SORT_WIDTH": width,
                    "PAD": route_pack_module._STABLE_SEGMENT_SORT_PAD,
                },
                num_warps,
            )
        )
    return cases


_CASES = _kernel_cases()


@pytest.mark.parametrize(
    "name,fn,args,num_warps", _CASES, ids=[case[0] for case in _CASES]
)
def test_route_pack_kernel_compiles(name, fn, args, num_warps) -> None:
    """The launched kernel instance compiles for the served target."""
    compiled = _compile(fn, args, num_warps=num_warps)
    assert compiled.asm["cubin"], name
    assert compiled.metadata.shared <= _SHARED_MEMORY_BUDGET, (
        f"{name} needs {compiled.metadata.shared} bytes of shared memory"
    )


@triton.jit
def _runtime_branch_width_kernel(
    values,
    THRESHOLD: tl.constexpr,
    NARROW: tl.constexpr,
    WIDE: tl.constexpr,
):
    """A kernel of exactly the shape this module exists to reject.

    Both branches bind ``lanes``, so Triton joins their types at the end of
    the ``if`` and refuses a lane extent that depends on a runtime value.
    Choosing the extent per program therefore has to happen on the host, one
    kernel instance per width, which is what ``_launch_stable_segment_pack``
    does.
    """
    count = tl.load(values)
    if count <= THRESHOLD:
        lanes = tl.arange(0, NARROW)
        tl.store(values + lanes, lanes)
    else:
        lanes = tl.arange(0, WIDE)
        tl.store(values + lanes, lanes)


def test_a_runtime_branch_over_lane_widths_is_rejected() -> None:
    """The check fails on the defect class it is here to catch.

    The Triton interpreter runs the taken branch of this kernel and reports
    nothing, so this negative control is what makes the module a gate rather
    than a restatement of the layout tests.
    """
    from triton.compiler.errors import CompilationError

    with pytest.raises(CompilationError, match="Mismatched type for lanes"):
        _compile(
            _runtime_branch_width_kernel,
            {"values": "*i32", "THRESHOLD": 0, "NARROW": 256, "WIDE": 2048},
            num_warps=4,
        )


def test_segment_sort_widths_partition_every_segment_size() -> None:
    """The launch bands cover every live segment size exactly once.

    ``_launch_stable_segment_pack`` walks the widths in order, giving band
    ``i`` the counts in ``(width[i - 1], width[i]]`` and the scan kernel
    everything above the last width. Ascending distinct powers of two make
    the bands a partition of ``[1, inf)``; a power of two is also what
    ``tl.sort`` requires of its lane extent.
    """
    widths = [width for width, _ in route_pack_module._STABLE_SEGMENT_SORT_WIDTHS]
    assert widths, "at least one register sort width is required"
    assert widths == sorted(set(widths))
    for width in widths:
        assert width > 0 and width & (width - 1) == 0, width
    scan_block = route_pack_module._STABLE_SEGMENT_SCAN_BLOCK_T
    assert scan_block > 0 and scan_block & (scan_block - 1) == 0
