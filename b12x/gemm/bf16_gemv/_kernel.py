"""BF16 small-N GEMV with static row groups and a runtime row grid."""
from __future__ import annotations

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Int64
from cutlass.cutlass_dsl import Int32, Uint32
from cutlass.cute.runtime import make_ptr

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    block_reduce,
    get_ptr_as_int64,
    ld_global_v4_u32,
    u32_as_f32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream

_THREADS = 128

#: Supported live-row envelope for decode and multi-token verification.
SMALL_M_MAX = 8

_KERNEL_CACHE: Dict[Tuple, object] = {}
_PRECOMPILED: set = set()


def _fadd(a, b):
    return a + b


class SmallNGemvKernel:
    """One CTA per output column and row group; live row counts are dynamic."""

    def __init__(self, n: int, k: int, rows_per_cta: int = 1):
        self.n = n
        self.k = k
        self.rows_per_cta = rows_per_cta

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        x = cute.make_tensor(x_ptr, cute.make_layout((m, self.k), stride=(self.k, 1)))
        w = cute.make_tensor(w_ptr, cute.make_layout((self.n, self.k), stride=(self.k, 1)))
        y = cute.make_tensor(y_ptr, cute.make_layout((Int64(m) * Int64(self.n),), stride=(1,)))
        row_tiles = 1 if self.rows_per_cta == SMALL_M_MAX else (m + self.rows_per_cta - 1) // self.rows_per_cta
        self.kernel(x, w, y).launch(
            grid=(self.n, row_tiles, 1),
            block=[_THREADS, 1, 1], cluster=[1, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        column, row_tile, _ = cute.arch.block_idx()
        row_base = Int32(0) if self.rows_per_cta == SMALL_M_MAX else Int32(row_tile) * Int32(self.rows_per_cta)
        k = Int32(self.k)
        w_base = Int64(column) * Int64(k)
        smem = cutlass.utils.SmemAllocator()
        red_buf = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((self.rows_per_cta, _THREADS // 32)), byte_alignment=16,
        )
        acc = cute.make_rmem_tensor((self.rows_per_cta,), cutlass.Float32)
        acc.fill(0.0)
        i = Int32(tidx)
        while i < k // Int32(8):
            col = Int64(i) * Int64(8)
            w0, w1, w2, w3 = ld_global_v4_u32(get_ptr_as_int64(mW, w_base + col))
            weights = tuple((u32_as_f32(wv << Uint32(16)), u32_as_f32(wv & Uint32(0xFFFF0000)))
                            for wv in (w0, w1, w2, w3))
            for r in cutlass.range_constexpr(self.rows_per_cta):
                row = row_base + Int32(r)
                if self.rows_per_cta == 1 or row < mX.shape[0]:
                    x0, x1, x2, x3 = ld_global_v4_u32(get_ptr_as_int64(mX, Int64(row) * Int64(k) + col))
                    value = acc[r]
                    for (low, high), xv in zip(weights, (x0, x1, x2, x3), strict=True):
                        value = value + low * u32_as_f32(xv << Uint32(16))
                        value = value + high * u32_as_f32(xv & Uint32(0xFFFF0000))
                    acc[r] = value
            i += Int32(_THREADS)
        if cutlass.const_expr(self.rows_per_cta == 1):
            total = block_reduce(warp_reduce(acc[0], _fadd), _fadd, red_buf, cutlass.Float32(0.0))
            if tidx == Int32(0):
                mY[Int64(row_base) * Int64(self.n) + Int64(column)] = cutlass.BFloat16(total)
        else:
            for r in cutlass.range_constexpr(self.rows_per_cta):
                if row_base + Int32(r) < mX.shape[0]:
                    value = warp_reduce(acc[r], _fadd)
                    if tidx % Int32(32) == Int32(0):
                        red_buf[r, tidx // Int32(32)] = value
            cute.arch.barrier()
            row = row_base + Int32(tidx)
            if tidx < Int32(self.rows_per_cta) and row < mX.shape[0]:
                total = cutlass.Float32(0.0)
                for warp in cutlass.range_constexpr(_THREADS // 32):
                    total += red_buf[tidx, warp]
                mY[Int64(row) * Int64(self.n) + Int64(column)] = cutlass.BFloat16(total)


def compile_bf16_gemv_small_n(m: int, n: int, k: int):
    """Compile one device/model specialization that accepts 1..8 live rows."""
    assert 1 <= m <= SMALL_M_MAX, f"small-N GEMV requires m<={SMALL_M_MAX}, got {m}"
    assert k % 8 == 0, f"K must be a multiple of 8, got {k}"
    assert n >= 1
    cache_key = (torch.cuda.current_device(), n, k)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from ._tuning import row_group_for_geometry

    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    rows_per_cta = row_group_for_geometry(n, k, sms)
    kernel = SmallNGemvKernel(n, k, rows_per_cta)
    def pointer(address=16):
        return make_ptr(cutlass.BFloat16, address, cute.AddressSpace.gmem, assumed_align=16)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    raw = b12x_compile(
        kernel,
        pointer(),
        pointer(),
        pointer(),
        Int32(SMALL_M_MAX),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.bf16_gemv_small_n",
            9,
            (*cache_key, rows_per_cta),
        ),
    )

    def launch(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor):
        raw(pointer(x.data_ptr()), pointer(w.data_ptr()), pointer(y.data_ptr()),
            Int32(x.shape[0]), current_cuda_stream())

    _KERNEL_CACHE[cache_key] = launch
    return launch


def get_cached_bf16_gemv_small_n(m: int, n: int, k: int):
    """Cache-only lookup (no JIT). Returns the launch fn or ``None``."""
    return _KERNEL_CACHE.get((torch.cuda.current_device(), n, k))


def precompile_bf16_gemv_small_n(weight: torch.Tensor, log=None) -> None:
    """Compile and warm the full row capacity before CUDA graph capture."""
    if log is None:
        import logging

        log = logging.getLogger("b12x.bf16_gemv")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if (
        not weight.is_cuda
        or k % 8 != 0
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.data_ptr() % 16 != 0
    ):
        log.info(
            "bf16 GEMV precompile: weight n=%d k=%d unsupported, skipping", n, k
        )
        return
    with torch.cuda.device(weight.device):
        if (weight.device.index, n, k) in _PRECOMPILED:
            return
        log.info(
            "bf16 GEMV precompile: n=%d k=%d, m=1..%d", n, k, SMALL_M_MAX
        )
        m = SMALL_M_MAX
        launch = compile_bf16_gemv_small_n(m, n, k)
        x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
        y = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)
        launch(x, weight, y)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _PRECOMPILED.add((weight.device.index, n, k))


def _use_kernel(x: torch.Tensor, weight: torch.Tensor) -> bool:
    m, k = x.shape
    return (
        x.is_cuda and weight.device == x.device
        and 1 <= m <= SMALL_M_MAX
        and k % 8 == 0
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


@torch.library.custom_op("b12x::bf16_gemv_small_n", mutates_args=())
def bf16_gemv_small_n(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``y = x @ weight.T`` for bf16 ``x (m, K)`` / ``weight (N, K)``.

    Routes small decode shapes (``m <= SMALL_M_MAX``) through the CUTE GEMV;
    anything else falls back to ``F.linear`` (cuBLAS) inside the op.
    """
    if not _use_kernel(x, weight):
        return torch.nn.functional.linear(x, weight)
    if not x.is_contiguous() or x.data_ptr() % 16 != 0:
        x = x.contiguous()
        if x.data_ptr() % 16 != 0:
            return torch.nn.functional.linear(x, weight)
    with torch.cuda.device(x.device):
        m, k = x.shape
        n = weight.shape[0]
        launch = get_cached_bf16_gemv_small_n(m, n, k)
        if launch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("BF16 GEMV compile miss during CUDA-graph capture; precompile the weight geometry")
            launch = compile_bf16_gemv_small_n(m, n, k)
        y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        launch(x, weight, y)
        return y


@bf16_gemv_small_n.register_fake
def _bf16_gemv_small_n_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)
