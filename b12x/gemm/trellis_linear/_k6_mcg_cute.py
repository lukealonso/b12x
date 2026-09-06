"""Cooperative SM120 dense K2-K6/MCG execution for small row counts.

``elem(x * suh) -> H128 -> K{2,3,4,5,6}/MCG GEMM -> H128 -> elem(* svh)``.
The kernel executes the complete native Trellis dense-linear transform in one resident
grid for either FP16 or BF16 activations:

``elem(x * suh) -> H128 -> K6/MCG GEMM -> H128 -> elem(* svh)``.

The GEMM uses B12X's native Trellis decoder and split-K reduction.  A
generation-counted grid barrier orders the input rotation before GEMM reads.
Cooperative launch admission is therefore part of the execution contract.

Only unpaired six-bit MCG payloads are eligible.  A paired QSRT payload has
explicit pair metadata and must use the generic dense Trellis scheduler.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.compiler import OptLevel
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    get_ptr_as_int64,
    half2_to_float2_scaled,
    half2_mul,
    ld_shared_u32,
    pack_f32x2_to_half2,
    shared_ptr_to_u32,
    st_global_u32,
)
from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_lut
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared.kernels.w4a16.kernel import (
    _SCALAR_ACC_FRAGMENT_WIDTH,
    _TRELLIS256_BITS,
    W4A16GemmKernel,
)


_MAX_ROWS = 16
_ROUTE_BLOCK = 16
_TILE_K = 128
_TILE_N = 128
_BARRIER_WORDS = 1
_STANDALONE_K6_DISABLED = os.environ.get("B12X_DISABLE_STANDALONE_K6", "0") == "1"

# The three K-by-N projection shapes below dominate GLM-5.2 dense online-K6
# decode. The resident CTA counts minimize full-chain latency on a 188-SM
# SM120 device while retaining capacity for concurrent model streams.
_GLM_GRID_CTA = {
    (2048, 4096): 64,
    (6144, 1024): 32,
    (512, 6144): 48,
}

# Exact checkpoint-native Qwen3.8-27B graph-replay sweeps on the 188-SM RTX
# PRO 6000 Blackwell selected these resident GEMM grids.  The prior generic
# 64-CTA fallback left the 40-N-tile projections substantially underfilled;
# the shape-specific grids preserve one cooperative launch while distributing
# split-K work across enough resident CTAs.  These are b12x planner decisions,
# not integration-side routing knobs.
_QWEN38_GRID_CTA = {
    # Linear-attention in_proj_z and in_proj_qkv.
    (5120, 6144): 120,
    (5120, 10240): 160,
    # Full-attention Q projection.
    (5120, 12288): 188,
    # Linear-attention output and MLP down projections.
    (6144, 5120): 120,
    (17408, 5120): 160,
    # MLP gate/up projections (K5 in Qwen3.8-27B K5K6 checkpoint).
    (5120, 17408): 128,
}

_MEASURED_GRID_CTA = {**_GLM_GRID_CTA, **_QWEN38_GRID_CTA}


@dataclass(frozen=True)
class K6McgSmallMCompileResult:
    compiled: object
    device_index: int
    size_k: int
    size_n: int
    params_dtype: torch.dtype
    grid_x: int
    cta_threads: int
    resident_ctas: int
    blocks_per_sm: int
    shared_memory_bytes: int
    required_scratch_elements: int
    required_workspace_elements: int
    trellis_lut: torch.Tensor = field(repr=False, compare=False)

    def accepts_input(self, x: torch.Tensor) -> bool:
        """Return whether a runtime input matches this bound launch."""
        return bool(
            isinstance(x, torch.Tensor)
            and x.is_cuda
            and x.dtype == self.params_dtype
            and x.ndim == 2
            and x.is_contiguous()
            and 1 <= int(x.shape[0]) <= _MAX_ROWS
            and int(x.shape[1]) == self.size_k
            and int(x.device.index if x.device.index is not None else 0)
            == self.device_index
        )

    def launch_grid_x(self, rows: int) -> int:
        """Return the CTA count needed to rotate each H128 block once."""
        warps_per_cta = self.cta_threads // 32
        rotation_blocks = int(rows) * (self.size_k // 128)
        rotation_ctas = (rotation_blocks + warps_per_cta - 1) // warps_per_cta
        if rotation_ctas > self.grid_x:
            # Keep the final rotation warp independent of the resident GEMM
            # subset when rotation needs more CTAs than split-K execution.
            rotation_ctas += 1
        return max(self.grid_x, min(rotation_ctas, self.resident_ctas))


_CACHE: dict[tuple[object, ...], K6McgSmallMCompileResult] = {}


@dsl_user_op
def _ld_global_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load one naturally aligned uint32 from global memory."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _grid_barrier_arrive_release(
    addr: Int64,
    increment: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Atomically arrive at a grid barrier with GPU release semantics."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(increment).ir_value(loc=loc, ip=ip),
            ],
            "atom.add.release.gpu.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _grid_barrier_wait_flip(
    addr: Int64,
    previous: Int32,
    *,
    loc=None,
    ip=None,
):
    """Wait until the barrier generation bit differs from ``previous``."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(previous).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred p;\n"
        ".reg .u32 current, delta;\n"
        "grid_barrier_wait:\n"
        "  ld.acquire.gpu.u32 current, [$0];\n"
        "  xor.b32 delta, current, $1;\n"
        "  and.b32 delta, delta, 0x80000000;\n"
        "  setp.eq.u32 p, delta, 0;\n"
        "  @p bra grid_barrier_wait;\n"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

def _is_unpaired_k6_mcg_weight(prepared) -> bool:
    """Return whether a prepared weight has the unpaired K2-K6/MCG contract."""
    return bool(
        getattr(prepared, "params_dtype", None) in (torch.float16, torch.bfloat16)
        and getattr(prepared, "weight_layout", None) == "trellis_t256"
        and int(getattr(prepared, "num_experts", 0)) == 1
        and int(getattr(prepared, "trellis_bits", 0)) in _TRELLIS256_BITS
        and str(getattr(prepared, "trellis_codebook", "")).lower() == "mcg"
        and getattr(prepared, "trellis_pair_kind", None) is None
        and getattr(prepared, "trellis_rate_axis", None) is None
        and int(getattr(prepared, "in_features", 0)) % 128 == 0
        and int(getattr(prepared, "out_features", 0)) % 128 == 0
    )


def is_k6_mcg_small_m_eligible(x: torch.Tensor, prepared) -> bool:
    """Return whether input and weight satisfy the fused K2-K6/MCG contract."""
    return bool(
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == getattr(prepared, "params_dtype", None)
        and x.ndim == 2
        and x.is_contiguous()
        and 1 <= int(x.shape[0]) <= _MAX_ROWS
        and int(x.shape[1]) == int(getattr(prepared, "in_features", -1))
        and _is_unpaired_k6_mcg_weight(prepared)
    )


def _requested_grid_x(size_k: int, size_n: int) -> int:
    """Return the measured or conservative GEMM CTA count for one shape."""
    requested = _MEASURED_GRID_CTA.get((int(size_k), int(size_n)))
    if requested is None:
        n_tiles = int(size_n) // _TILE_N
        k_tiles = int(size_k) // _TILE_K
        requested = max(n_tiles, min(n_tiles * k_tiles, 64))
    return max(1, int(requested))


def _grid_x(size_k: int, size_n: int, resident_ctas: int) -> int:
    """Cap the GEMM grid so every cooperative CTA can remain resident."""
    return min(_requested_grid_x(size_k, size_n), int(resident_ctas))


def k6_mcg_small_m_scratch_elements(size_k: int, size_n: int) -> int:
    """Return the FP32 split-K scratch required by an unpaired K2-K6 weight."""
    if min(int(size_k), int(size_n)) <= 0 or size_k % 128 or size_n % 128:
        raise ValueError("K2-K6/MCG K and N must be positive multiples of 128")
    return max(
        int(size_n) * _ROUTE_BLOCK,
        _requested_grid_x(size_k, size_n) * _ROUTE_BLOCK * _TILE_N,
    )


class K6McgSmallMKernel:
    """One cooperative grid for H128, split-K MCG GEMM, and H128 output."""

    ABI_VERSION = 4

    def __init__(self, *, size_k: int, size_n: int, element_dtype: str, trellis_bits: int = 6):
        """Build a compile-time specialization for one K-by-N projection."""
        if element_dtype not in ("fp16", "bf16"):
            raise ValueError(f"unsupported K6/MCG element dtype {element_dtype!r}")
        if trellis_bits not in _TRELLIS256_BITS:
            raise ValueError(f"unsupported trellis_bits {trellis_bits}, must be in {_TRELLIS256_BITS}")
        self.size_k = int(size_k)
        self.size_n = int(size_n)
        self.element_dtype = element_dtype
        self.trellis_bits = int(trellis_bits)
        self.gemm = W4A16GemmKernel(
            size_m=_MAX_ROWS,
            size_n=self.size_n,
            size_k=self.size_k,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=_TILE_N,
            tile_k=_TILE_K,
            moe_block_size=_ROUTE_BLOCK,
            max_m_blocks=1,
            element_dtype=self.element_dtype,
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="packed",
            trellis_bits=self.trellis_bits,
            trellis_codebook="mcg",
            dense_route_fast_path=True,
            schedule_whole_tiles=False,
        )
        self.cta_threads = int(self.gemm.cta_threads)
        self.blocks_per_sm = int(self.gemm.blocks_per_sm)
        self.sms = int(self.gemm.sms)
        self.shared_words = int(self.gemm.shared_words)
        self.barrier_count_off = self.sms * 4

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        """Return every compile-time value that affects generated code."""
        return (
            "k6_mcg_small_m",
            self.ABI_VERSION,
            self.size_k,
            self.size_n,
            self.element_dtype,
            self.trellis_bits,
            self.gemm.__cache_key__,
            self.cta_threads,
            self.blocks_per_sm,
            self.shared_words,
        )

    @cute.jit
    def _had128_quad(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
        lane: Int32,
    ):
        """Apply normalized Hadamard-128 to four values in each warp lane."""
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        h0 = s0 + s1
        h1 = d0 + d1
        h2 = s0 - s1
        h3 = d0 - d1
        for i in cutlass.range_constexpr(5):
            offset = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=offset)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=offset)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=offset)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=offset)
            if (lane & Int32(offset)) != Int32(0):
                h0 = p0 - h0
                h1 = p1 - h1
                h2 = p2 - h2
                h3 = p3 - h3
            else:
                h0 = p0 + h0
                h1 = p1 + h1
                h2 = p2 + h2
                h3 = p3 + h3
        scale = cutlass.Float32(0.088388347648)
        return h0 * scale, h1 * scale, h2 * scale, h3 * scale

    @cute.jit
    def _input_rotation(
        self,
        source: cute.Tensor,
        rotated: cute.Tensor,
        suh: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        """Rotate and scale all live input rows across the cooperative grid."""
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        blocks_per_row = Int32(self.size_k // 128)
        total = active_m * blocks_per_row
        cta_begin = (total * cta) // grid_x
        cta_end = (total * (cta + Int32(1))) // grid_x
        unit = cta_begin + warp
        elem = lane * Int32(4)
        while unit < cta_end:
            row = unit // blocks_per_row
            block = unit - row * blocks_per_row
            col = block * Int32(128) + elem
            source_base = row * Int32(self.size_k) + col
            source01 = _ld_global_u32(get_ptr_as_int64(source, source_base))
            source23 = _ld_global_u32(get_ptr_as_int64(source, source_base + Int32(2)))
            scale01 = _ld_global_u32(get_ptr_as_int64(suh, col))
            scale23 = _ld_global_u32(get_ptr_as_int64(suh, col + Int32(2)))
            if cutlass.const_expr(self.gemm.is_fp16):
                x0, x1 = half2_to_float2_scaled(
                    half2_mul(source01, scale01), cutlass.Float32(1.0)
                )
                x2, x3 = half2_to_float2_scaled(
                    half2_mul(source23, scale23), cutlass.Float32(1.0)
                )
                h0, h1, h2, h3 = self._had128_quad(x0, x1, x2, x3, lane)
                st_global_u32(
                    get_ptr_as_int64(rotated, source_base),
                    pack_f32x2_to_half2(h0, h1),
                )
                st_global_u32(
                    get_ptr_as_int64(rotated, source_base + Int32(2)),
                    pack_f32x2_to_half2(h2, h3),
                )
            else:
                x0, x1 = self.gemm._elem2_to_f32x2(source01)
                x2, x3 = self.gemm._elem2_to_f32x2(source23)
                s0, s1 = half2_to_float2_scaled(scale01, cutlass.Float32(1.0))
                s2, s3 = half2_to_float2_scaled(scale23, cutlass.Float32(1.0))
                h0, h1, h2, h3 = self._had128_quad(
                    x0 * s0,
                    x1 * s1,
                    x2 * s2,
                    x3 * s3,
                    lane,
                )
                st_global_u32(
                    get_ptr_as_int64(rotated, source_base),
                    self.gemm._pack_f32x2_to_elem2(h0, h1),
                )
                st_global_u32(
                    get_ptr_as_int64(rotated, source_base + Int32(2)),
                    self.gemm._pack_f32x2_to_elem2(h2, h3),
                )
            unit += warps_per_cta

    @cute.jit
    def _output_rotation_tile(
        self,
        output: cute.Tensor,
        svh: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        active_m: Int32,
    ):
        """Rotate one complete GEMM output tile to the selected element dtype."""
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        row = warp
        elem = lane * Int32(4)
        col = output_n_tile * Int32(128) + elem
        # The final split-K owner has staged the complete N128 tile in the
        # GEMM epilogue's padded shared-memory rows.  Reusing that tile avoids
        # reloading the intermediate element output from global memory.
        cute.arch.sync_threads()
        while row < active_m:
            base = row * Int32(self.size_n) + col
            shared_row_stride = Int32(2 * self.gemm.cta_n_blocks + 1)
            shared_addr = (
                smem_base
                + Int32(self.gemm.sh_red_off * 16)
                + row * shared_row_stride * Int32(16)
                + elem * Int32(2)
            )
            packed01 = ld_shared_u32(shared_addr)
            packed23 = ld_shared_u32(shared_addr + Int32(4))
            if cutlass.const_expr(self.gemm.is_fp16):
                v0, v1 = half2_to_float2_scaled(
                    packed01, cutlass.Float32(1.0)
                )
                v2, v3 = half2_to_float2_scaled(
                    packed23, cutlass.Float32(1.0)
                )
            else:
                v0, v1 = self.gemm._elem2_to_f32x2(packed01)
                v2, v3 = self.gemm._elem2_to_f32x2(packed23)
            h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
            scale01 = _ld_global_u32(get_ptr_as_int64(svh, col))
            scale23 = _ld_global_u32(get_ptr_as_int64(svh, col + Int32(2)))
            if cutlass.const_expr(self.gemm.is_fp16):
                st_global_u32(
                    get_ptr_as_int64(output, base),
                    half2_mul(pack_f32x2_to_half2(h0, h1), scale01),
                )
                st_global_u32(
                    get_ptr_as_int64(output, base + Int32(2)),
                    half2_mul(pack_f32x2_to_half2(h2, h3), scale23),
                )
            else:
                s0, s1 = half2_to_float2_scaled(scale01, cutlass.Float32(1.0))
                s2, s3 = half2_to_float2_scaled(scale23, cutlass.Float32(1.0))
                st_global_u32(
                    get_ptr_as_int64(output, base),
                    self.gemm._pack_f32x2_to_elem2(h0 * s0, h1 * s1),
                )
                st_global_u32(
                    get_ptr_as_int64(output, base + Int32(2)),
                    self.gemm._pack_f32x2_to_elem2(h2 * s2, h3 * s3),
                )
            row += warps_per_cta
        cute.arch.sync_threads()

    @cute.jit
    def _emit_tile(
        self,
        rotated: cute.Tensor,
        trellis: cute.Tensor,
        output: cute.Tensor,
        scales: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp: cute.Tensor,
        workspace: cute.Tensor,
        svh: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        active_m: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        """Run one split-K Trellis tile and let its final owner store H128."""
        # Compose the unpaired epilogue from B12X's existing Trellis
        # pipeline primitives. Keeping this orchestration local prevents the
        # K6 output transform from changing code generation for generic
        # W4A16 prefill kernels.
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self.gemm._tile_common_prologue(
            global_scale,
            workspace,
            global_scale,
            smem_base,
            tid,
            route_block_idx,
            Int32(0),
            output_n_tile,
            active_m,
        )
        a_sh_rd = self.gemm._a_shared_read_offset(tid, 16)
        acc0 = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
            for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc0[frag].fill(0.0)
        acc1 = acc0
        acc2 = acc0
        acc3 = acc0
        if cutlass.const_expr(self.gemm.cta_m_blocks > 1):
            acc1 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc1[frag].fill(0.0)
        if cutlass.const_expr(self.gemm.cta_m_blocks > 2):
            acc2 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc2[frag].fill(0.0)
        if cutlass.const_expr(self.gemm.cta_m_blocks > 3):
            acc3 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc3[frag].fill(0.0)

        k_tiles = reduce_tile_count
        self.gemm._prefetch_initial_tiles(
            rotated,
            rotated,
            trellis,
            scales,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            Int32(0),
            -1,
        )

        b_scale_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((2, 4), Uint32)
        self.gemm._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
            reduce_k_tile,
            -1,
        )
        a_regs = cute.make_rmem_tensor((self.gemm.cta_m_blocks, 4), Uint32)
        a_regs_next = cute.make_rmem_tensor((self.gemm.cta_m_blocks, 4), Uint32)
        self.gemm._load_a_register_bundle(
            a_regs,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            False,
        )
        self.gemm._run_mma_pipeline(
            rotated,
            rotated,
            trellis,
            scales,
            trellis_lut_addr,
            smem_base,
            tid,
            acc0,
            acc1,
            acc2,
            acc3,
            b_scale_cur,
            b_scale_next,
            a_regs,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            Int32(0),
            -1,
            False,
        )
        self.gemm._fold_cta_partials_large_m(
            acc0,
            acc1,
            acc2,
            acc3,
            smem_base,
            tid,
        )
        if reduce_slice_count > Int32(1):
            self.gemm._wait_for_reduction_turn(
                workspace, lock_slot, reduce_slice_idx, tid
            )
            self.gemm._combine_splitk_accumulators(
                acc0,
                acc1,
                acc2,
                acc3,
                c_tmp,
                block_valid_rows,
                lock_slot,
                reduce_slice_idx,
                reduce_slice_count,
                tid,
                False,
            )
            self.gemm._publish_reduction_turn(
                workspace,
                lock_slot,
                reduce_slice_idx == reduce_slice_count - Int32(1),
                tid,
            )
        if reduce_slice_idx == reduce_slice_count - Int32(1):
            self._store_h128_tile(
                output,
                svh,
                acc0,
                acc1,
                acc2,
                acc3,
                smem_base,
                tid,
                output_n_tile,
                block_valid_rows,
                global_scale_f32,
            )

    @cute.jit
    def _store_h128_tile(
        self,
        output: cute.Tensor,
        svh: cute.Tensor,
        acc0,
        acc1,
        acc2,
        acc3,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        """Stage the final accumulator tile and apply its output transform."""
        c_sh_stride = Int32(2 * self.gemm.cta_n_blocks + 1)
        c_sh_wr = (
            Int32(4) * c_sh_stride * ((tid & Int32(31)) // Int32(4))
            + (tid & Int32(31)) % Int32(4)
            + Int32(32) * (tid // Int32(32))
        )
        if tid // Int32(32) < Int32(self.gemm.tb_n_warps):
            self.gemm._store_tile_large_m_block(
                acc0,
                smem_base,
                c_sh_wr,
                c_sh_stride,
                tid // Int32(32),
                global_scale_f32,
            )
        self._output_rotation_tile(
            output,
            svh,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
        )

    @cute.jit
    def _grid_barrier(
        self,
        workspace: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
    ):
        """Order input rotation stores before every CTA starts GEMM reads."""
        cute.arch.sync_threads()
        if tid == Int32(0):
            barrier_addr = get_ptr_as_int64(workspace, Int32(self.barrier_count_off))
            increment = Int32(1)
            if cta == Int32(0):
                increment = Int32(-2147483647) - grid_x
            previous = _grid_barrier_arrive_release(barrier_addr, increment)
            _grid_barrier_wait_flip(barrier_addr, previous)
        cute.arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        trellis: cute.Tensor,
        output_ptr: cute.Pointer,
        rotated_ptr: cute.Pointer,
        scales: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp: cute.Tensor,
        workspace: cute.Tensor,
        suh_ptr: cute.Pointer,
        svh_ptr: cute.Pointer,
        trellis_lut: cute.Tensor,
        active_m: cutlass.Int32,
        launch_grid_x: cutlass.Int32,
        gemm_grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Bind pointer layouts and cooperatively launch one projection."""
        rows64 = active_m.to(cutlass.Int64)
        source = cute.make_tensor(
            source_ptr,
            layout=cute.make_layout((rows64 * Int64(self.size_k),), stride=(1,)),
        )
        rotated = cute.make_tensor(
            rotated_ptr,
            layout=cute.make_layout((rows64 * Int64(self.size_k),), stride=(1,)),
        )
        output = cute.make_tensor(
            output_ptr,
            layout=cute.make_layout((rows64 * Int64(self.size_n),), stride=(1,)),
        )
        suh = cute.make_tensor(
            suh_ptr, layout=cute.make_layout((Int64(self.size_k),), stride=(1,))
        )
        svh = cute.make_tensor(
            svh_ptr, layout=cute.make_layout((Int64(self.size_n),), stride=(1,))
        )
        self.kernel(
            source,
            trellis,
            output,
            rotated,
            scales,
            global_scale,
            c_tmp,
            workspace,
            suh,
            svh,
            trellis_lut,
            active_m,
            gemm_grid_x,
        ).launch(
            grid=(launch_grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            cooperative=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        trellis: cute.Tensor,
        output: cute.Tensor,
        rotated: cute.Tensor,
        scales: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp: cute.Tensor,
        workspace: cute.Tensor,
        suh: cute.Tensor,
        svh: cute.Tensor,
        trellis_lut: cute.Tensor,
        active_m: cutlass.Int32,
        gemm_grid_x: cutlass.Int32,
    ):
        """Execute input H128, split-K GEMM, and output H128 in one grid."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_raw)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        self._input_rotation(source, rotated, suh, tid, cta, grid_x, active_m)
        self._grid_barrier(workspace, tid, cta, grid_x)

        dummy = workspace
        if cta < gemm_grid_x:
            emit_tile = partial(
                self._emit_tile,
                rotated,
                trellis,
                output,
                scales,
                global_scale,
                c_tmp,
                workspace,
                svh,
                get_ptr_as_int64(trellis_lut, Int32(0)),
                smem_base,
                tid,
                active_m,
            )
            self.gemm._run_persistent_gemm(
                rotated,
                rotated,
                trellis,
                output,
                scales,
                global_scale,
                dummy,
                dummy,
                dummy,
                global_scale,
                c_tmp,
                workspace,
                get_ptr_as_int64(trellis_lut, Int32(0)),
                smem_base,
                tid,
                cta,
                gemm_grid_x,
                active_m,
                emit_tile,
            )


def compile_k6_mcg_small_m(
    *,
    size_k: int,
    size_n: int,
    params_dtype: torch.dtype,
    device: torch.device,
    trellis_bits: int = 6,
) -> K6McgSmallMCompileResult:
    """Compile the cooperative specialization on the weight's CUDA device."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"K6/MCG compilation requires a CUDA device, got {device}")
    with torch.cuda.device(device):
        return _compile_k6_mcg_small_m_current_device(
            size_k=size_k,
            size_n=size_n,
            params_dtype=params_dtype,
            device=device,
            trellis_bits=trellis_bits,
        )


def _compile_k6_mcg_small_m_current_device(
    *,
    size_k: int,
    size_n: int,
    params_dtype: torch.dtype,
    device: torch.device,
    trellis_bits: int = 6,
) -> K6McgSmallMCompileResult:
    """Compile after the public planner has selected the weight's device."""
    if params_dtype == torch.float16:
        element_dtype = "fp16"
        cutlass_dtype = cutlass.Float16
    elif params_dtype == torch.bfloat16:
        element_dtype = "bf16"
        cutlass_dtype = cutlass.BFloat16
    else:
        raise ValueError(
            "K6/MCG small-row compilation requires fp16 or bf16 parameters"
        )
    kernel = K6McgSmallMKernel(
        size_k=size_k,
        size_n=size_n,
        element_dtype=element_dtype,
        trellis_bits=trellis_bits,
    )
    resident_ctas = int(kernel.sms * kernel.blocks_per_sm)
    grid_x = _grid_x(size_k, size_n, resident_ctas)
    device_index = int(device.index if device.index is not None else 0)
    cache_key = ("k6_mcg_small_m", device_index, kernel.__cache_key__, grid_x)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    def tensor(dtype, elements: int, *, align: int = 16):
        """Construct one compile-only compact tensor descriptor."""
        return cute.runtime.make_fake_compact_tensor(
            dtype, (max(int(elements), 1),), assumed_align=align
        )

    trellis_elements = (int(size_k) // 16) * (int(size_n) // 16) * (trellis_bits * 8)
    scratch_elements = k6_mcg_small_m_scratch_elements(size_k, size_n)
    workspace_elements = 4 * int(kernel.sms) + _BARRIER_WORDS
    compile_args = (
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass.Int32, trellis_elements),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass.Int32, 1),
        tensor(cutlass.Float32, 1),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Int32, 4 * 256 + _BARRIER_WORDS),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass.Uint8, 1),
        Int32(1),
        Int32(grid_x),
        Int32(grid_x),
        current_cuda_stream(),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        *compile_args,
        compile_spec=KernelCompileSpec.from_key(
            "gemm.trellis.k6_mcg_small_m", K6McgSmallMKernel.ABI_VERSION, cache_key
        ),
        dsl_compile_options=OptLevel(2),
    )
    # Resolve and retain the trellis LUT at the model-load planning boundary.
    # MCG does not read the table, but the shared launch signature requires a
    # stable device pointer and CUDA graph capture must never populate its
    # process-global cache.
    trellis_lut = sqg_xor_cheb_t12_lut(device)
    result = K6McgSmallMCompileResult(
        compiled=compiled,
        device_index=device_index,
        size_k=int(size_k),
        size_n=int(size_n),
        params_dtype=params_dtype,
        grid_x=int(grid_x),
        cta_threads=int(kernel.cta_threads),
        resident_ctas=int(resident_ctas),
        blocks_per_sm=int(kernel.blocks_per_sm),
        shared_memory_bytes=int(kernel.shared_words * 4),
        required_scratch_elements=int(scratch_elements),
        required_workspace_elements=int(workspace_elements),
        trellis_lut=trellis_lut,
    )
    _CACHE[cache_key] = result
    return result


def _validate_k6_mcg_small_m_prepared_contract(
    prepared,
    launch: K6McgSmallMCompileResult,
) -> None:
    """Validate immutable weight, launch, workspace, and LUT bindings once."""
    device = prepared.trellis.device
    device_index = int(device.index if device.index is not None else 0)
    if (
        launch.device_index != device_index
        or launch.size_k != int(prepared.in_features)
        or launch.size_n != int(prepared.out_features)
        or launch.params_dtype != prepared.params_dtype
    ):
        raise RuntimeError(
            "compiled K6/MCG launch does not match its prepared weight contract"
        )

    workspace = prepared.workspace
    if (
        workspace.dtype != torch.int32
        or int(workspace.numel()) < launch.required_workspace_elements
        or workspace.device != device
        or not workspace.is_contiguous()
        or int(workspace.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            "prepared K6/MCG workspace must be contiguous, 16-byte-aligned "
            f"int32 on {device} with at least "
            f"{launch.required_workspace_elements} elements"
        )

    trellis_lut = launch.trellis_lut
    if (
        trellis_lut.dtype != torch.uint8
        or trellis_lut.device != device
        or not trellis_lut.is_contiguous()
        or int(trellis_lut.numel()) < 1
        or int(trellis_lut.data_ptr()) % 16 != 0
    ):
        raise RuntimeError(
            "bound K6/MCG LUT must be non-empty, contiguous, 16-byte-aligned "
            f"uint8 on {device}"
        )


def plan_k6_mcg_small_m(prepared) -> K6McgSmallMCompileResult | None:
    """Bind the cooperative launch for an eligible prepared K2-K6/MCG weight.

    This function is a model-load planning boundary. It resolves the compiled
    launch before serving and returns ``None`` when the weight must retain the
    generic dense Trellis scheduler.
    """
    if _STANDALONE_K6_DISABLED or not _is_unpaired_k6_mcg_weight(prepared):
        return None
    trellis_bits = int(getattr(prepared, "trellis_bits", 6))
    launch = compile_k6_mcg_small_m(
        size_k=int(prepared.in_features),
        size_n=int(prepared.out_features),
        params_dtype=prepared.params_dtype,
        device=prepared.trellis.device,
        trellis_bits=trellis_bits,
    )
    _validate_k6_mcg_small_m_prepared_contract(prepared, launch)
    return launch


def run_k6_mcg_small_m(
    x: torch.Tensor,
    prepared,
    *,
    output: torch.Tensor,
    rotated: torch.Tensor,
    c_tmp: torch.Tensor,
) -> torch.Tensor:
    """Execute a validated unpaired K6/MCG payload on the current stream."""
    launch = getattr(prepared, "k6_mcg_small_m_launch", None)
    if not isinstance(launch, K6McgSmallMCompileResult):
        raise RuntimeError(
            "prepared K6/MCG weight has no bound small-row launch; prepare the "
            "weight through b12x.gemm.trellis_linear.prepare_weight"
        )
    if not launch.accepts_input(x):
        raise ValueError("input does not satisfy the bound K6/MCG small-row ABI")
    m, size_k = (int(v) for v in x.shape)
    size_n = launch.size_n
    for name, tensor, shape, dtype in (
        ("output", output, (m, size_n), x.dtype),
        ("rotated", rotated, (m, size_k), x.dtype),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
            or int(tensor.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"{name} must be contiguous {dtype} with shape {shape} on {x.device}"
            )
    if (
        c_tmp.dtype != torch.float32
        or c_tmp.device != x.device
        or not c_tmp.is_contiguous()
        or int(c_tmp.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            "c_tmp must be contiguous, aligned float32 on the input device"
        )
    launch_grid_x = launch.launch_grid_x(m)
    if int(c_tmp.numel()) < launch.required_scratch_elements:
        raise ValueError(
            "K6/MCG split-K scratch is too small: "
            f"required={launch.required_scratch_elements}, "
            f"got={int(c_tmp.numel())}"
        )
    cutlass_dtype = (
        cutlass.Float16 if x.dtype == torch.float16 else cutlass.BFloat16
    )
    launch.compiled(
        make_ptr(
            cutlass_dtype,
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        prepared.trellis,
        make_ptr(
            cutlass_dtype,
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass_dtype,
            rotated.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        prepared.scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared.global_scale,
        c_tmp,
        prepared.workspace,
        make_ptr(
            cutlass.Float16,
            prepared.suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float16,
            prepared.svh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        launch.trellis_lut,
        m,
        launch_grid_x,
        launch.grid_x,
        current_cuda_stream(),
    )
    return output


def clear_k6_mcg_small_m_cache() -> None:
    """Drop process-global compiled-launch lookup entries."""
    _CACHE.clear()


__all__ = [
    "K6McgSmallMCompileResult",
    "clear_k6_mcg_small_m_cache",
    "compile_k6_mcg_small_m",
    "is_k6_mcg_small_m_eligible",
    "k6_mcg_small_m_scratch_elements",
    "plan_k6_mcg_small_m",
    "run_k6_mcg_small_m",
]
