"""Lean GLM-5.3 M8 NVFP4 route/activation preparation.

This is deliberately a narrow decode specialization.  The ordinary dynamic
kernel uses its full resident-grid kernel to initialize state, compact routes,
quantize the eight input rows, and publish the materialized work domain before
launching the non-cooperative compute body.  At the exact GLM-5.3 M8 shape that
preparation does not need a resident grid: one CTA per token can quantize the
row once and fan it out to its eight compact expert rows while CTA zero writes
the arithmetic task metadata.

The packed activation/scales and task layout are byte-for-byte the contracts
consumed by ``MoEDynamicKernelBackend(split_phase="compute")``.  No model math,
scale precision, or output reduction is changed.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32, Uint8, Uint32, Uint64

from b12x._lib.intrinsics import (
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    quantize_block_fp4_fast,
    st_global_f32,
    st_global_i32,
    st_global_u64,
)


class Nvfp4M8RoutePackKernel:
    """Prepare E288/K4096/top-k8/tile-M16 materialized compute work."""

    threads_per_cta = 128
    num_tokens = 8
    num_topk = 8
    num_experts = 288
    hidden_size = 4096
    tile_m = 16
    sf_vec_size = 16
    # N=512 is consumed as four independent N128 intermediate slices.  The
    # materialized queue uses one arithmetic slot per (M tile, slice).
    task_groups = 4

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        row_counts: cute.Tensor,
        expert_tile_base: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        all_work_published: cute.Tensor,
        input_global_scale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            a_input,
            topk_ids,
            topk_weights,
            packed_a_storage,
            scale_storage,
            row_counts,
            expert_tile_base,
            scatter_output,
            token_map,
            token_weights,
            task_expert,
            task_valid_rows,
            task_head,
            task_tail,
            all_work_published,
            input_global_scale,
        ).launch(
            # One CTA owns one (token, route) pair. The original token-only
            # grid reused each activation load but serialized all eight
            # expert-scale quantizations in every lane. M8 has only 64
            # routes, so exposing that route dimension still fits in one GPU
            # wave and removes the serial inner loop.
            grid=(self.num_tokens, self.num_topk, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        row_counts: cute.Tensor,
        expert_tile_base: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        all_work_published: cute.Tensor,
        input_global_scale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, route_idx, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            physical_row: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 4]
            expert: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 4]

        storage = smem.allocate(Storage)
        route_row = storage.physical_row.get_tensor(cute.make_layout(1))
        route_expert = storage.expert.get_tensor(cute.make_layout(1))

        # Derive a stable compact-row index for each route. With only 64
        # routes, an ordered prefix count in the eight route-owner lanes is
        # cheaper than a grid of atomics and removes the resident rendezvous.
        if tidx == Int32(0):
            pair_idx = Int32(token_idx * self.num_topk) + Int32(route_idx)
            route_expert_value = topk_ids[pair_idx].to(Int32)
            route_physical_row_value = Int32(-1)
            if route_expert_value >= Int32(0) and route_expert_value < Int32(
                self.num_experts
            ):
                row = Int32(0)
                prior = Int32(0)
                while prior < pair_idx:
                    if topk_ids[prior].to(Int32) == route_expert_value:
                        row += Int32(1)
                    prior += Int32(1)
                physical_tile = expert_tile_base[route_expert_value] + row // Int32(
                    self.tile_m
                )
                route_physical_row_value = physical_tile * Int32(
                    self.tile_m
                ) + row % Int32(self.tile_m)
                st_global_i32(
                    get_ptr_as_int64(token_map, route_physical_row_value),
                    Int32(token_idx),
                )
                st_global_f32(
                    get_ptr_as_int64(token_weights, route_physical_row_value),
                    topk_weights[pair_idx].to(cutlass.Float32),
                )
            route_row[Int32(0)] = route_physical_row_value
            route_expert[Int32(0)] = route_expert_value

        # CTA zero publishes the arithmetic materialized-task domain.  The
        # compute kernel recovers M-tile/slice coordinates from each slot, so
        # only expert and valid-row fields are materialized here.
        expert = Int32(0)
        if Int32(token_idx) == Int32(0) and Int32(route_idx) == Int32(0):
            expert = Int32(tidx)
            while expert < Int32(self.num_experts):
                rows = row_counts[expert].to(Int32)
                tile_offset = Int32(0)
                while rows > Int32(0):
                    physical_tile = expert_tile_base[expert] + tile_offset
                    valid_rows = rows
                    if valid_rows > Int32(self.tile_m):
                        valid_rows = Int32(self.tile_m)
                    group = Int32(0)
                    while group < Int32(self.task_groups):
                        slot = physical_tile * Int32(self.task_groups) + group
                        task_expert[slot] = expert
                        task_valid_rows[slot] = valid_rows
                        group += Int32(1)
                    rows -= Int32(self.tile_m)
                    tile_offset += Int32(1)
                expert += Int32(self.threads_per_cta)
            if tidx == Int32(0):
                task_head[Int32(0)] = Int32(0)
                task_tail[Int32(0)] = expert_tile_base[Int32(self.num_experts)] * Int32(
                    self.task_groups
                )
                all_work_published[Int32(0)] = Int32(1)

        # Compute-only deliberately skips the fused kernel's scatter clear.
        # Each preparation CTA owns one complete output row, so this is a
        # contiguous, conflict-free zero fill.
        if Int32(route_idx) == Int32(0):
            scatter_u32 = cute.recast_tensor(scatter_output, cutlass.Uint32)
            clear_idx = Int32(tidx)
            while clear_idx < Int32(self.hidden_size // 2):
                scatter_u32[Int32(token_idx * (self.hidden_size // 2)) + clear_idx] = (
                    Uint32(0)
                )
                clear_idx += Int32(self.threads_per_cta)

        cute.arch.sync_threads()

        # Load each K/16 block once, then quantize it with each route's exact
        # per-expert input scale. GLM-5.3 does not use one shared activation
        # scale; reusing a single packed payload would change model math.
        sf_blocks_per_row = Int32(self.hidden_size // self.sf_vec_size)
        packed_bytes_per_row = Int32(self.hidden_size // 2)
        num_k_tiles = Int32(self.hidden_size // 64)
        physical_row_value = route_row[Int32(0)]
        expert_idx = route_expert[Int32(0)]
        if physical_row_value >= Int32(0):
            global_scale = input_global_scale[expert_idx].to(cutlass.Float32)
            sf_idx = Int32(tidx)
            while sf_idx < sf_blocks_per_row:
                block_start = sf_idx * Int32(self.sf_vec_size)
                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for elem_idx in cutlass.range_constexpr(16):
                    value = cutlass.Float32(
                        a_input[Int32(token_idx), block_start + Int32(elem_idx)]
                    )
                    values[elem_idx] = value
                    block_max = fmax_f32(block_max, fabs_f32(value))
                packed64, scale_byte = quantize_block_fp4_fast(
                    values, block_max, global_scale
                )
                st_global_u64(
                    get_ptr_as_int64(
                        packed_a_storage,
                        physical_row_value * packed_bytes_per_row + sf_idx * Int32(8),
                    ),
                    Uint64(packed64),
                )
                k_tile_idx = sf_idx // Int32(4)
                inner_k_idx = sf_idx % Int32(4)
                sf_atom = physical_row_value >> Int32(7)
                sf_row = physical_row_value & Int32(127)
                scale_offset = (
                    sf_atom * num_k_tiles * Int32(32 * 4 * 4)
                    + k_tile_idx * Int32(32 * 4 * 4)
                    + (sf_row % Int32(32)) * Int32(4 * 4)
                    + (sf_row // Int32(32)) * Int32(4)
                    + inner_k_idx
                )
                scale_storage[scale_offset] = Uint8(scale_byte)
                sf_idx += Int32(self.threads_per_cta)
