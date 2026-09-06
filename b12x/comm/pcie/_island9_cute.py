"""CuTeDSL push-based nine-rank island all-reduce (two PCIe islands + one root rank).

Topology contract: ranks 0-3 share one PCIe switch cluster (island 0), ranks
4-7 share another (island 1), and rank 8 sits on the CPU root complex, so every
link to rank 8 crosses the root and every island-0 <-> island-1 transfer
crosses the two-tier cascade. Rank ``island * 4 + lane`` owns quarter ``lane``
of the vector; rank 8 owns nothing and contributes through island 0.

Three phases, every remote transfer a posted write into a peer's inbox:

1. Each rank scatters its input quarters into the owners' stage inboxes
   (island ranks to their three island peers, rank 8 to the four island-0
   owners). An owner reduces its quarter in fp32 in ascending source rank
   order: island 0 owners over ranks 0, 1, 2, 3, 8; island 1 owners over
   ranks 4, 5, 6, 7. The fp32 partial stays in fp32 (no intermediate
   rounding) and is pushed to the same-lane owner of the other island.
2. Both owners of a quarter add the island partials (island 0 first) and
   round once to bf16: the same value on both ranks.
3. Owners push their final quarter to their island peers (island 0 owners
   also to rank 8); every rank assembles the output from local memory.

Per-block arrival flags with generation counters double-buffer the inboxes,
so the kernel is CUDA-graph replayable without host involvement.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import (
    cuda_stream_from_int_or_current,
    cuda_stream_to_int,
    current_cuda_stream,
    make_ptr,
)

from ._island_rs_cute import (
    _atomic_add_global_u32,
    _fence_sc_sys,
    _load_acquire_sys_u32,
    _nanosleep,
    _store_release_sys_u32,
    pack_f32x2_to_bf16x2,
    unpack_bf16x2,
)

WORLD_SIZE = 9
ISLAND_SIZE = 4
ROOT_RANK = 8
# Stage inbox source slots: island lanes 0-3, rank 8 in slot 4.
_ROOT_SLOT = 4
_STAGE_SLOTS = 5
_FLAG_SLOTS = 8
MAX_BLOCKS = 32
_FLAG_LINE = 128
_FLAG_REGION_BYTES = MAX_BLOCKS * _FLAG_SLOTS * _FLAG_LINE
_P1_ARRIVED = 256
_P2_ARRIVED = _P1_ARRIVED + _FLAG_REGION_BYTES
_P3_ARRIVED = _P2_ARRIVED + _FLAG_REGION_BYTES
HEADER_BYTES = _P3_ARRIVED + _FLAG_REGION_BYTES


def island_of(rank: int) -> int:
    """Island index of an island rank (rank 8 has none)."""
    if not 0 <= rank < ROOT_RANK:
        raise ValueError(f"rank {rank} is not an island rank")
    return rank // ISLAND_SIZE


def lane_of(rank: int) -> int:
    if not 0 <= rank < ROOT_RANK:
        raise ValueError(f"rank {rank} is not an island rank")
    return rank % ISLAND_SIZE


def island9_peers(rank: int) -> tuple[int, ...]:
    """Ranks whose slabs this rank maps (inbox destinations and flag targets)."""
    if not 0 <= rank < WORLD_SIZE:
        raise ValueError(f"invalid rank {rank} for TP{WORLD_SIZE}")
    if rank == ROOT_RANK:
        return tuple(range(ISLAND_SIZE))
    island, lane = divmod(rank, ISLAND_SIZE)
    peers = {island * ISLAND_SIZE + p for p in range(ISLAND_SIZE)} - {rank}
    peers.add((1 - island) * ISLAND_SIZE + lane)
    if island == 0:
        peers.add(ROOT_RANK)
    return tuple(sorted(peers))


@cute.jit
def _wait_for(address: Int64, generation: Uint32, cycles: int) -> None:
    observed = _load_acquire_sys_u32(address)
    while observed < generation:
        if cutlass.const_expr(cycles > 0):
            _nanosleep(Uint32(cycles))
        observed = _load_acquire_sys_u32(address)


@cute.jit
def _flag_address(base: Int64, region: int, block: Int32, slot: int) -> Int64:
    return (
        base
        + Int64(region)
        + (Int64(block) * Int64(_FLAG_SLOTS) + Int64(slot)) * Int64(_FLAG_LINE)
    )


@cute.jit
def _word_ptr(address: Int64):
    """Uint32 (bf16x2) view of a 4-byte aligned global address."""
    return cute.recast_ptr(
        cute.make_ptr(
            cutlass.BFloat16,
            address,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ).align(4),
        dtype=Uint32,
    )


@cute.jit
def _float_ptr(address: Int64):
    """Float32 view of a 4-byte aligned global address."""
    return cute.make_ptr(
        cutlass.Float32,
        address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    ).align(4)


class _Island9Launch:
    """Rank-specialized launcher; the role and every peer index fold at compile time."""

    def __init__(
        self,
        rank: int,
        *,
        threads: int = 256,
        wait_nanosleep_cycles: int = 24,
    ) -> None:
        if not 0 <= rank < WORLD_SIZE:
            raise ValueError(f"invalid rank {rank} for world size {WORLD_SIZE}")
        if not 32 <= threads <= 1024 or threads % 32:
            raise ValueError("threads must be a multiple of 32 in [32, 1024]")
        self._rank = int(rank)
        self._threads = int(threads)
        self._wait_nanosleep_cycles = int(wait_nanosleep_cycles)
        self._is_root = self._rank == ROOT_RANK
        self._island = 0 if self._is_root else self._rank // ISLAND_SIZE
        self._lane = -1 if self._is_root else self._rank % ISLAND_SIZE
        self._partner = (
            -1 if self._is_root else (1 - self._island) * ISLAND_SIZE + self._lane
        )
        self._my_slot = _ROOT_SLOT if self._is_root else self._lane

    @cute.jit
    def __call__(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stage_offset: Int64,
        part_self_offset: Int64,
        part_inbox_offset: Int64,
        final_offset: Int64,
        quarter_capacity: Int64,
        elements: Int64,
        blocks: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            slab0,
            slab1,
            slab2,
            slab3,
            slab4,
            slab5,
            slab6,
            slab7,
            slab8,
            input_ptr,
            output_ptr,
            stage_offset,
            part_self_offset,
            part_inbox_offset,
            final_offset,
            quarter_capacity,
            elements,
        ).launch(
            grid=(blocks, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stage_offset: Int64,
        part_self_offset: Int64,
        part_inbox_offset: Int64,
        final_offset: Int64,
        quarter_capacity: Int64,
        elements: Int64,
    ) -> None:
        slabs = (slab0, slab1, slab2, slab3, slab4, slab5, slab6, slab7, slab8)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        self_base = Int64(slabs[self._rank].toint())

        allocator = cutlass.utils.SmemAllocator()
        shared_generation = allocator.allocate_tensor(
            element_type=cutlass.Uint32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )
        if tidx == Int32(0):
            shared_generation[0] = _atomic_add_global_u32(
                self_base + Int64(bidx) * Int64(4), Uint32(1)
            ) + Uint32(1)
        cute.arch.barrier()
        generation = Uint32(shared_generation[0])
        # Double-buffered inboxes: odd generations use the second slot set.
        if generation % Uint32(2) != Uint32(0):
            stage_offset += quarter_capacity * Int64(_STAGE_SLOTS * 4)
            part_self_offset += quarter_capacity * Int64(8)
            part_inbox_offset += quarter_capacity * Int64(8)
            final_offset += quarter_capacity * Int64(ISLAND_SIZE * 4)

        # Everything below counts bf16x2 words; callers guarantee an even
        # element count.
        pairs = elements // Int64(2)
        quarter = (pairs + Int64(ISLAND_SIZE - 1)) // Int64(ISLAND_SIZE)
        stride = Int64(gdim) * Int64(self._threads)
        start = Int64(bidx) * Int64(self._threads) + Int64(tidx)

        input_words = cute.recast_ptr(input_ptr.align(4), dtype=Uint32)
        output_words = cute.recast_ptr(output_ptr.align(4), dtype=Uint32)

        # ---- Phase 1: scatter my quarters into the owners' stage inboxes. ----
        my_slot = self._my_slot
        for lane in cutlass.range_constexpr(ISLAND_SIZE):
            if cutlass.const_expr(self._is_root or lane != self._lane):
                owner = self._island * ISLAND_SIZE + lane
                begin = Int64(lane) * quarter
                stop = begin + quarter
                if stop > pairs:
                    stop = pairs
                inbox = _word_ptr(
                    Int64(slabs[owner].toint())
                    + stage_offset
                    + Int64(my_slot) * quarter_capacity * Int64(4)
                )
                index = start
                while index < stop - begin:
                    inbox[index] = input_words[begin + index]
                    index += stride
        cute.arch.barrier()
        if tidx == Int32(0):
            _fence_sc_sys()
            for lane in cutlass.range_constexpr(ISLAND_SIZE):
                if cutlass.const_expr(self._is_root or lane != self._lane):
                    owner = self._island * ISLAND_SIZE + lane
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[owner].toint()), _P1_ARRIVED, bidx, my_slot
                        ),
                        generation,
                    )

        if cutlass.const_expr(not self._is_root):
            # ---- Phase 1 (owner): wait for the island contributions, reduce
            # my quarter in fp32 in ascending source rank order. ----
            if tidx == Int32(0):
                for slot in cutlass.range_constexpr(ISLAND_SIZE):
                    if cutlass.const_expr(slot != self._lane):
                        _wait_for(
                            _flag_address(self_base, _P1_ARRIVED, bidx, slot),
                            generation,
                            self._wait_nanosleep_cycles,
                        )
                if cutlass.const_expr(self._island == 0):
                    _wait_for(
                        _flag_address(self_base, _P1_ARRIVED, bidx, _ROOT_SLOT),
                        generation,
                        self._wait_nanosleep_cycles,
                    )
            cute.arch.barrier()

            mine_begin = Int64(self._lane) * quarter
            mine_stop = mine_begin + quarter
            if mine_stop > pairs:
                mine_stop = pairs
            mine_length = mine_stop - mine_begin

            part_self = _float_ptr(self_base + part_self_offset)
            partner_inbox = _float_ptr(
                Int64(slabs[self._partner].toint()) + part_inbox_offset
            )
            index = start
            while index < mine_length:
                total_lo = Float32(0.0)
                total_hi = Float32(0.0)
                for slot in cutlass.range_constexpr(ISLAND_SIZE):
                    if cutlass.const_expr(slot == self._lane):
                        lo, hi = unpack_bf16x2(input_words[mine_begin + index])
                    else:
                        stage = _word_ptr(
                            self_base
                            + stage_offset
                            + Int64(slot) * quarter_capacity * Int64(4)
                        )
                        lo, hi = unpack_bf16x2(stage[index])
                    total_lo += lo
                    total_hi += hi
                if cutlass.const_expr(self._island == 0):
                    stage = _word_ptr(
                        self_base
                        + stage_offset
                        + Int64(_ROOT_SLOT) * quarter_capacity * Int64(4)
                    )
                    lo, hi = unpack_bf16x2(stage[index])
                    total_lo += lo
                    total_hi += hi
                part_self[index * Int64(2)] = total_lo
                part_self[index * Int64(2) + Int64(1)] = total_hi
                partner_inbox[index * Int64(2)] = total_lo
                partner_inbox[index * Int64(2) + Int64(1)] = total_hi
                index += stride
            cute.arch.barrier()
            if tidx == Int32(0):
                _fence_sc_sys()
                _store_release_sys_u32(
                    _flag_address(
                        Int64(slabs[self._partner].toint()), _P2_ARRIVED, bidx, 0
                    ),
                    generation,
                )
                _wait_for(
                    _flag_address(self_base, _P2_ARRIVED, bidx, 0),
                    generation,
                    self._wait_nanosleep_cycles,
                )
            cute.arch.barrier()

            # ---- Phase 2: island 0 partial + island 1 partial, one rounding.
            # The same operand order on both owners gives identical bits. ----
            part_inbox = _float_ptr(self_base + part_inbox_offset)
            index = start
            while index < mine_length:
                self_lo = part_self[index * Int64(2)]
                self_hi = part_self[index * Int64(2) + Int64(1)]
                other_lo = part_inbox[index * Int64(2)]
                other_hi = part_inbox[index * Int64(2) + Int64(1)]
                if cutlass.const_expr(self._island == 0):
                    total_lo = self_lo + other_lo
                    total_hi = self_hi + other_hi
                else:
                    total_lo = other_lo + self_lo
                    total_hi = other_hi + self_hi
                value = pack_f32x2_to_bf16x2(total_lo, total_hi)
                output_words[mine_begin + index] = value
                for lane in cutlass.range_constexpr(ISLAND_SIZE):
                    if cutlass.const_expr(lane != self._lane):
                        peer = self._island * ISLAND_SIZE + lane
                        peer_final = _word_ptr(
                            Int64(slabs[peer].toint())
                            + final_offset
                            + Int64(self._lane) * quarter_capacity * Int64(4)
                        )
                        peer_final[index] = value
                if cutlass.const_expr(self._island == 0):
                    root_final = _word_ptr(
                        Int64(slabs[ROOT_RANK].toint())
                        + final_offset
                        + Int64(self._lane) * quarter_capacity * Int64(4)
                    )
                    root_final[index] = value
                index += stride
            cute.arch.barrier()
            if tidx == Int32(0):
                _fence_sc_sys()
                for lane in cutlass.range_constexpr(ISLAND_SIZE):
                    if cutlass.const_expr(lane != self._lane):
                        peer = self._island * ISLAND_SIZE + lane
                        _store_release_sys_u32(
                            _flag_address(
                                Int64(slabs[peer].toint()),
                                _P3_ARRIVED,
                                bidx,
                                self._lane,
                            ),
                            generation,
                        )
                if cutlass.const_expr(self._island == 0):
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[ROOT_RANK].toint()),
                            _P3_ARRIVED,
                            bidx,
                            self._lane,
                        ),
                        generation,
                    )

        # ---- Phase 3: assemble the quarters I do not own from my final inbox. ----
        if tidx == Int32(0):
            for lane in cutlass.range_constexpr(ISLAND_SIZE):
                if cutlass.const_expr(self._is_root or lane != self._lane):
                    _wait_for(
                        _flag_address(self_base, _P3_ARRIVED, bidx, lane),
                        generation,
                        self._wait_nanosleep_cycles,
                    )
        cute.arch.barrier()
        for lane in cutlass.range_constexpr(ISLAND_SIZE):
            if cutlass.const_expr(self._is_root or lane != self._lane):
                final_inbox = _word_ptr(
                    self_base + final_offset + Int64(lane) * quarter_capacity * Int64(4)
                )
                begin = Int64(lane) * quarter
                stop = begin + quarter
                if stop > pairs:
                    stop = pairs
                index = start
                while index < stop - begin:
                    output_words[begin + index] = final_inbox[index]
                    index += stride


@functools.lru_cache(maxsize=None)
def get_island9_launcher(
    rank: int,
    device_index: int,
    *,
    threads: int = 256,
    wait_nanosleep_cycles: int = 24,
) -> Callable[..., None]:
    """Return the rank-specialized nine-rank island all-reduce launcher."""

    del device_index  # retained in the process-local cache key
    launch = _Island9Launch(
        rank,
        threads=threads,
        wait_nanosleep_cycles=wait_nanosleep_cycles,
    )
    cache_key = (int(rank), int(threads), int(wait_nanosleep_cycles))
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    slab_placeholder = make_ptr(
        cutlass.Uint8,
        256,
        cute.AddressSpace.gmem,
        assumed_align=256,
    )
    raw = b12x_compile(
        launch,
        *(slab_placeholder for _ in range(WORLD_SIZE)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.island9.bf16",
            1,
            cache_key,
            labels=("rank", "threads", "wait_nanosleep_cycles"),
        ),
    )

    def run(
        slab_addresses: Sequence[int],
        input_address: int,
        output_address: int,
        stage_offset: int,
        part_self_offset: int,
        part_inbox_offset: int,
        final_offset: int,
        quarter_capacity: int,
        elements: int,
        blocks: int,
        stream: object = None,
    ) -> None:
        if len(slab_addresses) != WORLD_SIZE:
            raise ValueError(
                f"expected {WORLD_SIZE} slab addresses, got {len(slab_addresses)}"
            )
        if not 1 <= int(blocks) <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}], got {blocks}")
        raw(
            *(
                make_ptr(
                    cutlass.Uint8,
                    int(address),
                    cute.AddressSpace.gmem,
                    assumed_align=256,
                )
                for address in slab_addresses
            ),
            make_ptr(
                cutlass.BFloat16,
                input_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            make_ptr(
                cutlass.BFloat16,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            stage_offset,
            part_self_offset,
            part_inbox_offset,
            final_offset,
            quarter_capacity,
            elements,
            blocks,
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return run


__all__ = [
    "HEADER_BYTES",
    "ISLAND_SIZE",
    "MAX_BLOCKS",
    "ROOT_RANK",
    "WORLD_SIZE",
    "get_island9_launcher",
    "island9_peers",
    "island_of",
    "lane_of",
]
