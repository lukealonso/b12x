"""Paged QSA representative scoring with device-side context bounds.

Inactive CTAs initialize their score range without loading query or cache data.
BF16 tensor cores score 64 groups per CTA for up to eight index heads and
16-aligned head dimensions. Other geometries use scalar FP32 reductions over
32 groups per CTA. No live extent enters compilation keys.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    f32_to_raw_bits,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

_CACHE: dict[tuple, object] = {}
_THREADS = 128


@dsl_user_op
def _or_error(address: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.or.b32 [$0], 512;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _RepresentativeScoreKernel:
    def __init__(self, heads, dim, ratio, page_size, max_groups, budget):
        self.heads = heads
        self.dim = dim
        self.ratio = ratio
        self.page_size = page_size
        self.max_groups = max_groups
        self.budget = budget
        self.lane_values = (dim + 31) // 32
        self.scale = 1.0 / math.sqrt(dim)
        self.use_mma = heads <= 8 and dim % 16 == 0
        self.groups = 64 if self.use_mma else 32

    @cute.jit
    def load_pair(self, pointer, offset):
        address = pointer + offset
        value = Uint32(0)
        if (address.toint().to(Int64) & Int64(3)) == Int64(0):
            value = cute.recast_ptr(address.align(4), dtype=Uint32)[0].to(Uint32)
        else:
            # Odd storage offsets and token strides remain supported. Widening
            # and reassembling BF16 bits introduces no quantization.
            lo = f32_to_raw_bits(pointer[offset].to(Float32)) >> 16
            hi = f32_to_raw_bits(pointer[offset + Int64(1)].to(Float32))
            value = (lo | hi).to(Uint32)
        return value

    @cute.jit
    def score_mma(
        self,
        query,
        cache,
        table,
        errors,
        scores,
        request,
        row,
        page_stride,
        token_stride,
        table_stride,
        score_row,
        page_count,
        group_offset,
        group_count,
        local_start,
        eligible,
        carry,
        lane,
        warp,
    ):
        # m16n8k16: A rows are representatives; B columns are index heads.
        # Each four-lane subgroup owns two representative rows in C.
        matrix_row = lane // 4
        matrix_pair = lane % 4
        warp_start = local_start + warp * 16
        if group_offset + warp_start < eligible:
            base = cute.make_rmem_tensor((2,), Int64)
            valid = cute.make_rmem_tensor((2,), Int32)
            for part in cutlass.range_constexpr(2):
                local_group = warp_start + matrix_row + part * 8
                group = group_offset + local_group
                base[part] = Int64(0)
                valid[part] = Int32(0)
                if (local_group < group_count) & (group < eligible):
                    page = table[
                        request * table_stride + (group // self.page_size).to(Int64)
                    ].to(Int64)
                    if (page >= Int64(0)) & (page < page_count):
                        base[part] = (
                            page * page_stride
                            + (group % self.page_size).to(Int64) * token_stride
                        )
                        valid[part] = Int32(1)
                    elif matrix_pair == 0:
                        _or_error((errors + row.to(Int64)).toint().to(Int64))
            d0, d1, d2, d3 = Float32(0), Float32(0), Float32(0), Float32(0)
            for tile in cutlass.range_constexpr(self.dim // 16):
                mma_column = (tile * 16 + matrix_pair * 2).to(Int64)
                a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if valid[0] != 0:
                    a0 = self.load_pair(cache, base[0] + mma_column)
                    a2 = self.load_pair(cache, base[0] + mma_column + Int64(8))
                if valid[1] != 0:
                    a1 = self.load_pair(cache, base[1] + mma_column)
                    a3 = self.load_pair(cache, base[1] + mma_column + Int64(8))
                b0, b1 = Uint32(0), Uint32(0)
                if matrix_row < self.heads:
                    query_base = (
                        row.to(Int64) * self.heads * self.dim
                        + matrix_row.to(Int64) * self.dim
                        + mma_column
                    )
                    b0 = self.load_pair(query, query_base)
                    b1 = self.load_pair(query, query_base + Int64(8))
                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                    d0,
                    d1,
                    d2,
                    d3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                )
            for part in cutlass.range_constexpr(2):
                first = d0 if part == 0 else d2
                second = d1 if part == 0 else d3
                first = cutlass.max(first, Float32(0))
                second = cutlass.max(second, Float32(0))
                score = Float32(0)
                # Preserve head order, including zero-padded MMA columns.
                for head in cutlass.range_constexpr(self.heads):
                    value = first if head % 2 == 0 else second
                    value = cute.arch.shuffle_sync(
                        value,
                        (lane // 4) * 4 + head // 2,
                    )
                    score = score + value
                score = score * Float32(self.scale)
                if valid[part] == 0:
                    score = Float32(-float("inf"))
                local_group = warp_start + matrix_row + part * 8
                if (matrix_pair == 0) & (local_group < group_count):
                    scores[score_row + (carry + local_group).to(Int64)] = score
        else:
            local_group = warp_start + lane
            if (lane < 16) & (local_group < group_count):
                scores[score_row + (carry + local_group).to(Int64)] = Float32(
                    -float("inf")
                )

    @cute.jit
    def __call__(
        self,
        pointers: tuple,
        strides: tuple,
        rows: Int32,
        page_count: Int64,
        group_offset: Int32,
        group_count: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(pointers, strides, page_count, group_offset, group_count).launch(
            grid=((group_count + self.groups - 1) // self.groups, rows, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        pointers: tuple,
        strides: tuple,
        page_count: Int64,
        group_offset: Int32,
        group_count: Int32,
    ):
        (
            query,
            positions,
            requests,
            lengths,
            cache,
            table,
            errors,
            scores,
            counts,
            merges,
        ) = pointers
        page_stride, token_stride, table_stride, score_stride = strides
        block, row, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane = thread % 32
        warp = thread // 32
        request = requests[row].to(Int64)
        eligible = Int32(0)
        if (errors[row] == Int32(0)) & (request >= Int64(0)):
            position_groups = (positions[row].to(Int64) + Int64(1)) // self.ratio
            sequence_groups = lengths[request].to(Int64) // self.ratio
            eligible = cutlass.min(
                cutlass.min(position_groups, sequence_groups), Int64(self.max_groups)
            ).to(Int32)
        carry = cutlass.min(cutlass.min(eligible, group_offset), Int32(self.budget))
        if (block == 0) & (thread == 0):
            counts[row] = eligible
            merges[row] = carry + cutlass.min(
                cutlass.max(eligible - group_offset, Int32(0)), group_count
            )
        local_start = block * self.groups
        score_row = row.to(Int64) * score_stride
        if group_offset + local_start >= eligible:
            local_group = local_start + thread
            if (thread < self.groups) & (local_group < group_count):
                scores[score_row + (carry + local_group).to(Int64)] = Float32(
                    -float("inf")
                )
        elif cutlass.const_expr(self.use_mma):
            self.score_mma(
                query,
                cache,
                table,
                errors,
                scores,
                request,
                row,
                page_stride,
                token_stride,
                table_stride,
                score_row,
                page_count,
                group_offset,
                group_count,
                local_start,
                eligible,
                carry,
                lane,
                warp,
            )
        else:
            prepared = cute.make_rmem_tensor((self.heads, self.lane_values), Float32)
            for head in cutlass.range_constexpr(self.heads):
                for part in cutlass.range_constexpr(self.lane_values):
                    dim = lane + part * 32
                    value = Float32(0)
                    if dim < self.dim:
                        query_offset = (
                            row.to(Int64) * self.heads * self.dim
                            + head * self.dim
                            + dim
                        )
                        value = query[query_offset].to(Float32)
                    prepared[head, part] = value
            for item in cutlass.range(self.groups // (_THREADS // 32), unroll=1):
                local_group = local_start + warp + item * (_THREADS // 32)
                group = group_offset + local_group
                score = Float32(-float("inf"))
                if (local_group < group_count) & (group < eligible):
                    logical_page = group // self.page_size
                    page = table[request * table_stride + logical_page.to(Int64)].to(
                        Int64
                    )
                    if (page >= Int64(0)) & (page < page_count):
                        key = cute.make_rmem_tensor((self.lane_values,), Float32)
                        key_base = (
                            page * page_stride
                            + (group % self.page_size).to(Int64) * token_stride
                        )
                        for part in cutlass.range_constexpr(self.lane_values):
                            dim = lane + part * 32
                            value = Float32(0)
                            if dim < self.dim:
                                value = cache[key_base + dim].to(Float32)
                            key[part] = value
                        score = Float32(0)
                        for head in cutlass.range_constexpr(self.heads):
                            dot = Float32(0)
                            for part in cutlass.range_constexpr(self.lane_values):
                                dot = dot + key[part] * prepared[head, part]
                            for shift in cutlass.range_constexpr(5):
                                dot = dot + cute.arch.shuffle_sync_bfly(
                                    dot, offset=16 >> shift
                                )
                            score = score + cutlass.max(dot, Float32(0))
                        score = score * Float32(self.scale)
                    elif lane == 0:
                        _or_error((errors + row.to(Int64)).toint().to(Int64))
                if (lane == 0) & (local_group < group_count):
                    scores[score_row + (carry + local_group).to(Int64)] = score


def _pointer(tensor, dtype):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=dtype.width // 8
    )


def launch_score_representatives(
    *,
    prepared_query,
    query_positions,
    request_ids,
    sequence_lengths,
    compressed_cache,
    compressed_block_table,
    state_errors,
    scores,
    eligible_counts,
    merge_lengths,
    group_offset,
    group_count,
    caps,
):
    tensors = (
        prepared_query,
        query_positions,
        request_ids,
        sequence_lengths,
        compressed_cache,
        compressed_block_table,
        state_errors,
        scores,
        eligible_counts,
        merge_lengths,
    )
    types = tuple(
        {
            torch.bfloat16: BFloat16,
            torch.float32: Float32,
            torch.int32: Int32,
            torch.int64: Int64,
        }[t.dtype]
        for t in tensors
    )
    geometry = (
        int(caps.index_heads),
        int(caps.index_head_dim),
        int(caps.compress_ratio),
        int(caps.compressed_page_size),
        int(caps.max_groups),
        int(caps.group_budget),
    )
    with torch.cuda.device(prepared_query.device):
        key = (prepared_query.device.index, geometry, tuple(t.dtype for t in tensors))
        raw = _CACHE.get(key)
        if raw is None:
            kernel = _RepresentativeScoreKernel(*geometry)
            raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
            fake = tuple(
                make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8)
                for t in types
            )
            raw = b12x_compile(
                kernel,
                fake,
                (Int64(1),) * 4,
                Int32(1),
                Int64(1),
                Int32(0),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.qsa.representative_score", 1, key
                ),
            )
            _CACHE[key] = raw
        raw(
            tuple(_pointer(t, dt) for t, dt in zip(tensors, types, strict=True)),
            (
                Int64(compressed_cache.stride(0)),
                Int64(compressed_cache.stride(1)),
                Int64(compressed_block_table.stride(0)),
                Int64(scores.stride(0)),
            ),
            Int32(prepared_query.shape[0]),
            Int64(compressed_cache.shape[0]),
            Int32(group_offset),
            Int32(group_count),
            current_cuda_stream(),
        )
