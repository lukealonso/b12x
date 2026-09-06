"""CuTeDSL kernels for single-rounding BF16 PCIe two-shot collectives.

Structure mirrors :mod:`_twoshot_cute` (the fp8-transport variant): phase one
pushes this rank's shard into every peer's IPC staging slot (posted PCIe
writes), a per-CTA flag barrier follows, and phase two either reduces the
local rank's shard (reduce_scatter) or copies the staged shards into place
(all_gather).  Payload packs are 16 bytes = 8 bf16 values; the reduction
accumulates in fp32 in a fixed rank order (local rank first, then
``(local + i) % world`` for ``i = 1..world-1``) and rounds once to bf16, so a
given rank's output is deterministic across runs.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import ld_global_nc_v4_u32, ld_global_v4_u32, st_global_v4_u32
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._cute_intrinsics import (
    graph_epoch_arrive_serialized,
    ld_relaxed_gpu_u32,
    pack_f32x2_to_bf16x2,
    unpack_bf16x2,
)
from ._twoshot_cute import (
    _GRAPH_BLOCKS_ARRIVED_OFFSET,
    _GRAPH_EPOCH_OFFSET,
    _MAX_BLOCKS,
    _MAX_RANKS,
    _FLAG_STRIDE,
    _SELF_COUNTER_BYTES,
    _fence_sc_sys,
    _ld_generic_v4_u32,
    _ld_global_u32,
    _ld_relaxed_sys_u32,
    _st_generic_v4_u32,
    _st_global_u32,
    _st_relaxed_sys_u32,
)

_PREPARED_BF16_LAUNCHERS: set[tuple[object, ...]] = set()
_PACK_ELEMS = 8  # bf16 values per 16-byte pack
# Topologies with retained qualification evidence: TP4 (GLM-5.3 DCP4) and the
# Kimi-K3 TP8 / TP9 PCIe rings (docs/evidence, tests/comm/test_pcie_tp9_physical.py).
_SUPPORTED_WORLD_SIZES = (2, 4, 8, 9)


class _TwoShotBf16Launch:
    def __init__(
        self,
        operation: str,
        world_size: int,
        rank: int,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
        row_elems: int,
    ) -> None:
        if operation not in ("reduce_scatter", "all_gather"):
            raise ValueError(f"invalid two-shot operation {operation!r}")
        self._operation = operation
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._threads = int(threads)
        self._row_elems = int(row_elems)

    @cute.jit
    def __call__(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            payload,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            output,
            rank,
            pack_stride,
            slot_bytes,
            rows_per_rank,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _select_address(
        self,
        pointers: Sequence[cute.Pointer],
        index: Int32,
    ) -> Int64:
        """Select one scalar launch pointer without unrolling peer work."""

        address = Int64(pointers[0].toint())
        if cutlass.const_expr(self._world_size == 2):
            if index == Int32(1):
                address = Int64(pointers[1].toint())
            return address
        if index < Int32(4):
            if index < Int32(2):
                address = Int64(pointers[0].toint())
                if index == Int32(1):
                    address = Int64(pointers[1].toint())
            else:
                address = Int64(pointers[2].toint())
                if index == Int32(3):
                    address = Int64(pointers[3].toint())
        else:
            if index < Int32(6):
                address = Int64(pointers[4].toint())
                if index == Int32(5):
                    address = Int64(pointers[5].toint())
            else:
                address = Int64(pointers[6].toint())
                if cutlass.const_expr(self._world_size >= 8):
                    if index == Int32(7):
                        address = Int64(pointers[7].toint())
        if cutlass.const_expr(self._world_size == 9):
            if index == Int32(8):
                address = Int64(pointers[8].toint())
        return address

    @cute.jit
    def _barrier(
        self,
        signals: Sequence[cute.Pointer],
        rank: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cute.arch.barrier()
        if tidx < Int32(self._world_size):
            _fence_sc_sys()
            if cutlass.const_expr(self._operation == "all_gather"):
                self_base = Int64(signals[self._rank].toint())
            else:
                self_base = self._select_address(signals, rank)
            self_counter_address = self_base + (
                Int64(bidx) * Int64(_MAX_RANKS) + Int64(tidx)
            ) * Int64(4)
            value = _ld_global_u32(self_counter_address) + Uint32(1)
            _st_global_u32(self_counter_address, value)

            flag_slot = Int64(value % Uint32(2))
            peer_base = self._select_address(signals, tidx)
            peer_counter_address = (
                peer_base
                + Int64(_SELF_COUNTER_BYTES)
                + (
                    (flag_slot * Int64(_MAX_BLOCKS) + Int64(bidx))
                    * Int64(_MAX_RANKS * _FLAG_STRIDE)
                    + Int64(rank) * Int64(_FLAG_STRIDE)
                )
                * Int64(4)
            )
            self_counter_address = (
                self_base
                + Int64(_SELF_COUNTER_BYTES)
                + (
                    (flag_slot * Int64(_MAX_BLOCKS) + Int64(bidx))
                    * Int64(_MAX_RANKS * _FLAG_STRIDE)
                    + Int64(tidx) * Int64(_FLAG_STRIDE)
                )
                * Int64(4)
            )
            _st_relaxed_sys_u32(peer_counter_address, value)
            observed = _ld_relaxed_sys_u32(self_counter_address)
            while observed != value:
                observed = _ld_relaxed_sys_u32(self_counter_address)
        cute.arch.barrier()

    @cute.jit
    def _accumulate_words(
        self,
        accumulator: cute.Tensor,
        words,
    ) -> None:
        for word_index in cutlass.range_constexpr(4):
            lo, hi = unpack_bf16x2(words[word_index])
            element = word_index * 2
            accumulator[element] = accumulator[element] + lo
            accumulator[element + 1] = accumulator[element + 1] + hi

    @cute.jit
    def _load_accumulate_pack_global_nc(
        self,
        accumulator: cute.Tensor,
        address: Int64,
    ) -> None:
        words = ld_global_nc_v4_u32(address)
        self._accumulate_words(accumulator, words)

    @cute.jit
    def _load_accumulate_pack_generic(
        self,
        accumulator: cute.Tensor,
        address: Int64,
    ) -> None:
        words = _ld_generic_v4_u32(address)
        self._accumulate_words(accumulator, words)

    @cute.jit
    def _store_pack(self, output_address: Int64, accumulator: cute.Tensor) -> None:
        st_global_v4_u32(
            output_address,
            pack_f32x2_to_bf16x2(accumulator[0], accumulator[1]),
            pack_f32x2_to_bf16x2(accumulator[2], accumulator[3]),
            pack_f32x2_to_bf16x2(accumulator[4], accumulator[5]),
            pack_f32x2_to_bf16x2(accumulator[6], accumulator[7]),
        )

    @cute.kernel
    def kernel(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        block_threads, _, _ = cute.arch.block_dim()
        if cutlass.const_expr(self._operation == "all_gather"):
            local_rank = Int32(self._rank)
        else:
            local_rank = rank
        packs_per_row = Int32(self._row_elems // _PACK_ELEMS)
        shard_packs = Int64(rows_per_rank) * Int64(packs_per_row)
        chunk = (shard_packs + Int64(gdim) - Int64(1)) // Int64(gdim)
        begin = Int64(bidx) * chunk
        end = begin + chunk
        if end > shard_packs:
            end = shard_packs

        payload_address = Int64(payload.toint())
        staging_slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            if cutlass.const_expr(self._operation == "all_gather"):
                self_signal = Int64(signals[self._rank].toint())
            else:
                self_signal = self._select_address(signals, local_rank)
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_GRAPH_EPOCH_OFFSET))
            staging_slot_offset = (
                Int64((generation + Uint32(self._slot_bias)) % Uint32(2)) * slot_bytes
            )
            cute.arch.barrier()
            if Int32(tidx) == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_BLOCKS_ARRIVED_OFFSET),
                    Uint32(gdim),
                )

        # Phase one: push remote shards (posted PCIe writes), rank-staggered.
        peer_index = Int32(1)
        while peer_index < Int32(self._world_size):
            destination = (local_rank + peer_index) % Int32(self._world_size)
            destination_base = (
                self._select_address(staging, destination) + staging_slot_offset
            )
            destination_payload = destination_base + (
                Int64(local_rank) * pack_stride * Int64(16)
            )
            if cutlass.const_expr(self._operation == "reduce_scatter"):
                source_pack = Int64(destination) * shard_packs
            else:
                source_pack = Int64(0)

            index = begin + Int64(tidx)
            while index < end:
                words = ld_global_nc_v4_u32(
                    payload_address + (source_pack + index) * Int64(16)
                )
                _st_generic_v4_u32(
                    destination_payload + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += Int64(block_threads)
            peer_index += Int32(1)

        self._barrier(signals, local_rank)

        if cutlass.const_expr(self._operation == "all_gather"):
            self_base = Int64(staging[self._rank].toint())
        else:
            self_base = self._select_address(staging, local_rank)
        self_base += staging_slot_offset
        output_address = Int64(output.toint())
        if cutlass.const_expr(self._operation == "reduce_scatter"):
            index = begin + Int64(tidx)
            while index < end:
                accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
                for lane in cutlass.range_constexpr(_PACK_ELEMS):
                    accumulator[lane] = Float32(0.0)

                local_pack = Int64(local_rank) * shard_packs + index
                self._load_accumulate_pack_global_nc(
                    accumulator,
                    payload_address + local_pack * Int64(16),
                )

                for peer_index in cutlass.range(
                    Int32(1),
                    Int32(self._world_size),
                    Int32(1),
                    unroll=1,
                ):
                    source_rank = (local_rank + peer_index) % Int32(self._world_size)
                    staged_pack = (
                        self_base
                        + Int64(source_rank) * pack_stride * Int64(16)
                        + index * Int64(16)
                    )
                    self._load_accumulate_pack_generic(accumulator, staged_pack)

                self._store_pack(output_address + index * Int64(16), accumulator)
                index += Int64(self._threads)
        else:
            first_index = begin + Int64(tidx)
            iteration_count = Int32(
                (end - first_index + Int64(self._threads - 1)) // Int64(self._threads)
            )
            peer_index = Int32(0)
            index = Int64(0)
            while peer_index < Int32(self._world_size):
                source_rank = (local_rank + peer_index) % Int32(self._world_size)
                source_payload_base = Int64(0)
                if source_rank == local_rank:
                    source_payload_base = payload_address
                else:
                    source_payload_base = self_base + Int64(
                        source_rank
                    ) * pack_stride * Int64(16)
                destination_base = output_address + Int64(
                    source_rank
                ) * shard_packs * Int64(16)
                for iteration in cutlass.range(
                    Int32(0),
                    iteration_count,
                    Int32(1),
                    unroll=1,
                ):
                    index = first_index + Int64(iteration) * Int64(self._threads)
                    # Generic addressing serves both the local payload and the
                    # IPC-mapped peer slabs.
                    words = _ld_generic_v4_u32(source_payload_base + index * Int64(16))
                    st_global_v4_u32(
                        destination_base + index * Int64(16),
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
                peer_index += Int32(1)


def _bf16_process_key(
    operation: str,
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
) -> tuple[object, ...]:
    return (
        "bf16",
        str(operation),
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        int(slot_bias) & 1 if device_slot_selection else 0,
        int(threads),
        int(row_elems),
        int(device_index),
    )


def is_twoshot_bf16_launcher_prepared(
    operation: str,
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
) -> bool:
    return (
        _bf16_process_key(
            operation,
            world_size,
            rank,
            device_slot_selection,
            slot_bias,
            threads,
            row_elems,
            device_index,
        )
        in _PREPARED_BF16_LAUNCHERS
    )


@functools.cache
def get_twoshot_bf16_launcher(
    operation: str,
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
) -> Callable[..., None]:
    """Compile and return one static world/operation/thread specialization."""
    process_key = _bf16_process_key(
        operation,
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        device_index,
    )
    del device_index  # part of the process-local cache key
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "single-rounding BF16 two-shot launchers require a world size in "
            f"{_SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    if threads <= 0 or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a warp-aligned value in [32, 512]")
    if row_elems <= 0 or row_elems % _PACK_ELEMS != 0:
        raise ValueError("row_elems must be a positive multiple of 8")
    slot_bias = int(slot_bias) & 1
    launch = _TwoShotBf16Launch(
        operation,
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
    )
    cache_key = (
        "bf16",
        operation,
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
        int(row_elems),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            for _ in range(9)
        ),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            for _ in range(9)
        ),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        0,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            f"comm.pcie.twoshot_bf16.{operation}",
            2,
            cache_key,
        ),
    )

    def run(
        payload_address: int,
        staging_addresses: Sequence[int],
        signal_addresses: Sequence[int],
        output_address: int,
        rank: int,
        pack_stride: int,
        slot_bytes: int,
        rows_per_rank: int,
        grid_x: int,
    ) -> None:
        if len(staging_addresses) != 9 or len(signal_addresses) != 9:
            raise ValueError("two-shot scalar pointer ABI requires nine peer slots")
        raw_args = (
            make_ptr(
                cutlass.Uint32,
                payload_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                for address in staging_addresses
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for address in signal_addresses
            ),
            make_ptr(
                cutlass.Uint32,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            rank,
            pack_stride,
            slot_bytes,
            rows_per_rank,
            grid_x,
            current_cuda_stream(),
        )
        raw(*raw_args)

    _PREPARED_BF16_LAUNCHERS.add(process_key)
    return run


class _TwoShotPullAllReduceLaunch(_TwoShotBf16Launch):
    """Single-launch BF16 all-reduce with one rounding, using remote reads.

    The kernel stages this rank's full payload into its own IPC slab using a
    local copy. After the first barrier, every rank pulls its shard (P/world
    for a per-rank payload of P bytes) from every peer's staged payload,
    accumulates in fp32 (rank order: self, then
    ``(rank + i) % world``), rounds once to bf16 and writes the reduced shard
    both to the output and to its slab's reduced region. After the second
    barrier, every rank pulls the other ranks' reduced shards into the output.
    PCIe read volume per rank is 2P*(world - 1)/world, which is 1.5P for world
    size 4; synchronization uses two barriers.

    Shards are contiguous pack ranges. ``rows_per_rank`` rows of ``row_elems``
    form the base shard; ``remainder_packs`` (below the world size) extra packs
    are handed one each to the lowest ranks, so a payload whose pack count is
    not a multiple of the world size is reduced in place without wire padding.
    Rank ``k`` owns packs ``[k*base + min(k, r), (k+1)*base + min(k+1, r))``.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
        row_elems: int,
    ) -> None:
        super().__init__(
            "reduce_scatter",  # barrier addressing uses the dynamic-rank path
            world_size,
            rank,
            device_slot_selection,
            slot_bias,
            threads,
            row_elems,
        )

    @cute.jit
    def __call__(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        reduced_offset: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        remainder_packs: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            payload,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            output,
            rank,
            reduced_offset,
            slot_bytes,
            rows_per_rank,
            remainder_packs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _load_accumulate_pack_remote(
        self,
        accumulator: cute.Tensor,
        address: Int64,
    ) -> None:
        words = ld_global_v4_u32(address)
        self._accumulate_words(accumulator, words)

    @cute.kernel
    def kernel(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        reduced_offset: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        remainder_packs: Int32,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        local_rank = rank
        packs_per_row = Int32(self._row_elems // _PACK_ELEMS)
        base_packs = Int64(rows_per_rank) * Int64(packs_per_row)
        remainder = Int64(remainder_packs)
        full_packs = base_packs * Int64(self._world_size) + remainder
        # Balanced contiguous partition: ranks below the remainder own one
        # extra pack. The local shard bounds are runtime scalars.
        shard_base = Int64(local_rank) * base_packs + Int64(
            cutlass.min(local_rank, remainder_packs)
        )
        shard_packs = base_packs
        if local_rank < remainder_packs:
            shard_packs = base_packs + Int64(1)
        threads = Int64(self._threads)
        grid_threads = Int64(gdim) * threads
        flat = Int64(bidx) * threads + Int64(tidx)

        payload_address = Int64(payload.toint())
        output_address = Int64(output.toint())
        staging_slot_offset = Int64(0)
        self_signal = self._select_address(signals, local_rank)
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_GRAPH_EPOCH_OFFSET))
            staging_slot_offset = (
                Int64((generation + Uint32(self._slot_bias)) % Uint32(2)) * slot_bytes
            )
            cute.arch.barrier()
            if Int32(tidx) == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_BLOCKS_ARRIVED_OFFSET),
                    Uint32(gdim),
                )
        self_base = self._select_address(staging, local_rank) + staging_slot_offset

        # Stage the full local payload into this rank's own slab.
        index = flat
        while index < full_packs:
            words = ld_global_nc_v4_u32(payload_address + index * Int64(16))
            st_global_v4_u32(
                self_base + index * Int64(16),
                words[0],
                words[1],
                words[2],
                words[3],
            )
            index += grid_threads

        self._barrier(signals, local_rank)

        # Pull and reduce this rank's shard from every staged peer payload.
        index = flat
        while index < shard_packs:
            accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
            for lane in cutlass.range_constexpr(_PACK_ELEMS):
                accumulator[lane] = Float32(0.0)
            # Issue the local and all remote pack loads back-to-back (the peer
            # loop is unrolled at compile time), then accumulate in fixed
            # rank order: self, then (rank + i) % world for i = 1..world-1.
            pack_offset = (shard_base + index) * Int64(16)
            local_words = ld_global_nc_v4_u32(payload_address + pack_offset)
            peer_words = []
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                source_rank = (local_rank + Int32(peer_index)) % Int32(self._world_size)
                peer_base = (
                    self._select_address(staging, source_rank) + staging_slot_offset
                )
                peer_words.append(ld_global_v4_u32(peer_base + pack_offset))
            self._accumulate_words(accumulator, local_words)
            for peer_index in cutlass.range_constexpr(self._world_size - 1):
                self._accumulate_words(accumulator, peer_words[peer_index])
            self._store_pack(
                output_address + (shard_base + index) * Int64(16), accumulator
            )
            self._store_pack(
                self_base + reduced_offset + index * Int64(16), accumulator
            )
            index += grid_threads

        self._barrier(signals, local_rank)

        # Copy the other ranks' published reduced shards into the output.
        for peer_index in cutlass.range_constexpr(1, self._world_size):
            source_rank = (local_rank + Int32(peer_index)) % Int32(self._world_size)
            peer_reduced = (
                self._select_address(staging, source_rank)
                + staging_slot_offset
                + reduced_offset
            )
            peer_shard_base = Int64(source_rank) * base_packs + Int64(
                cutlass.min(source_rank, remainder_packs)
            )
            peer_shard_packs = base_packs
            if source_rank < remainder_packs:
                peer_shard_packs = base_packs + Int64(1)
            destination = output_address + peer_shard_base * Int64(16)
            index = flat
            while index < peer_shard_packs:
                words = ld_global_v4_u32(peer_reduced + index * Int64(16))
                st_global_v4_u32(
                    destination + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += grid_threads


class _TwoShotPushAllReduceLaunch(_TwoShotPullAllReduceLaunch):
    """Single-launch lossless bf16 all-reduce built on posted PCIe WRITES.

    Phase one pushes this rank's contribution to every peer's shard into that
    peer's staged payload region (rank-staggered destinations). After the
    first barrier every rank reduces its own shard from local memory only, in
    the same fixed order as the pull kernel (self, then ``(rank + i) % world``),
    writes the bf16 result to the output and pushes it into every peer's
    reduced region. After the second barrier the peers' reduced shards are
    copied from local memory into the output. Remote traffic is posted writes
    only; the shard partition matches the pull kernel, so outputs are
    bit-identical between the two kernels.
    """

    @cute.jit
    def __call__(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        reduced_offset: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        remainder_packs: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            payload,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            output,
            rank,
            pack_stride,
            reduced_offset,
            slot_bytes,
            rows_per_rank,
            remainder_packs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        reduced_offset: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        remainder_packs: Int32,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        local_rank = rank
        packs_per_row = Int32(self._row_elems // _PACK_ELEMS)
        base_packs = Int64(rows_per_rank) * Int64(packs_per_row)
        shard_base = Int64(local_rank) * base_packs + Int64(
            cutlass.min(local_rank, remainder_packs)
        )
        shard_packs = base_packs
        if local_rank < remainder_packs:
            shard_packs = base_packs + Int64(1)
        threads = Int64(self._threads)
        grid_threads = Int64(gdim) * threads
        flat = Int64(bidx) * threads + Int64(tidx)

        payload_address = Int64(payload.toint())
        output_address = Int64(output.toint())
        staging_slot_offset = Int64(0)
        self_signal = self._select_address(signals, local_rank)
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_GRAPH_EPOCH_OFFSET))
            staging_slot_offset = (
                Int64((generation + Uint32(self._slot_bias)) % Uint32(2)) * slot_bytes
            )
            cute.arch.barrier()
            if Int32(tidx) == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_BLOCKS_ARRIVED_OFFSET),
                    Uint32(gdim),
                )
        self_base = self._select_address(staging, local_rank) + staging_slot_offset

        # Phase one: push this rank's contribution to each peer's shard into
        # the peer's payload region, at this rank's source slot.
        peer_index = Int32(1)
        while peer_index < Int32(self._world_size):
            destination = (local_rank + peer_index) % Int32(self._world_size)
            destination_shard_base = Int64(destination) * base_packs + Int64(
                cutlass.min(destination, remainder_packs)
            )
            destination_shard_packs = base_packs
            if destination < remainder_packs:
                destination_shard_packs = base_packs + Int64(1)
            destination_slot = (
                self._select_address(staging, destination)
                + staging_slot_offset
                + Int64(local_rank) * pack_stride * Int64(16)
            )
            index = flat
            while index < destination_shard_packs:
                words = ld_global_nc_v4_u32(
                    payload_address + (destination_shard_base + index) * Int64(16)
                )
                _st_generic_v4_u32(
                    destination_slot + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += grid_threads
            peer_index += Int32(1)

        self._barrier(signals, local_rank)

        # Phase two: reduce this rank's shard from local memory, publish the
        # result to the output and to every peer's reduced region.
        index = flat
        while index < shard_packs:
            accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
            for lane in cutlass.range_constexpr(_PACK_ELEMS):
                accumulator[lane] = Float32(0.0)
            local_words = ld_global_nc_v4_u32(
                payload_address + (shard_base + index) * Int64(16)
            )
            peer_words = []
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                source_rank = (local_rank + Int32(peer_index)) % Int32(self._world_size)
                staged_pack = (
                    self_base
                    + Int64(source_rank) * pack_stride * Int64(16)
                    + index * Int64(16)
                )
                peer_words.append(_ld_generic_v4_u32(staged_pack))
            self._accumulate_words(accumulator, local_words)
            for peer_index in cutlass.range_constexpr(self._world_size - 1):
                self._accumulate_words(accumulator, peer_words[peer_index])
            self._store_pack(
                output_address + (shard_base + index) * Int64(16), accumulator
            )
            reduced_words = (
                pack_f32x2_to_bf16x2(accumulator[0], accumulator[1]),
                pack_f32x2_to_bf16x2(accumulator[2], accumulator[3]),
                pack_f32x2_to_bf16x2(accumulator[4], accumulator[5]),
                pack_f32x2_to_bf16x2(accumulator[6], accumulator[7]),
            )
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                destination = (local_rank + Int32(peer_index)) % Int32(self._world_size)
                destination_reduced = (
                    self._select_address(staging, destination)
                    + staging_slot_offset
                    + reduced_offset
                    + Int64(local_rank) * pack_stride * Int64(16)
                )
                _st_generic_v4_u32(
                    destination_reduced + index * Int64(16),
                    reduced_words[0],
                    reduced_words[1],
                    reduced_words[2],
                    reduced_words[3],
                )
            index += grid_threads

        self._barrier(signals, local_rank)

        # Phase three: copy the peers' reduced shards from local memory.
        for peer_index in cutlass.range_constexpr(1, self._world_size):
            source_rank = (local_rank + Int32(peer_index)) % Int32(self._world_size)
            source_reduced = (
                self_base
                + reduced_offset
                + Int64(source_rank) * pack_stride * Int64(16)
            )
            source_shard_base = Int64(source_rank) * base_packs + Int64(
                cutlass.min(source_rank, remainder_packs)
            )
            source_shard_packs = base_packs
            if source_rank < remainder_packs:
                source_shard_packs = base_packs + Int64(1)
            destination = output_address + source_shard_base * Int64(16)
            index = flat
            while index < source_shard_packs:
                words = _ld_generic_v4_u32(source_reduced + index * Int64(16))
                st_global_v4_u32(
                    destination + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += grid_threads


@functools.cache
def get_twoshot_bf16_allreduce_launcher(
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
    mode: str = "pull",
) -> Callable[..., None]:
    """Compile the single-launch bf16 all-reduce specialization.

    ``mode`` selects the remote-read (``"pull"``) or posted-write (``"push"``)
    kernel; both use the same shard partition and reduction order.
    """
    if mode not in ("pull", "push"):
        raise ValueError(f"invalid all-reduce mode {mode!r}")
    operation = f"all_reduce_{mode}"
    process_key = _bf16_process_key(
        operation,
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        device_index,
    )
    del device_index
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "single-rounding BF16 two-shot launchers require a world size in "
            f"{_SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    if threads <= 0 or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a warp-aligned value in [32, 512]")
    if row_elems <= 0 or row_elems % _PACK_ELEMS != 0:
        raise ValueError("row_elems must be a positive multiple of 8")
    slot_bias = int(slot_bias) & 1
    launch_cls = (
        _TwoShotPushAllReduceLaunch if mode == "push" else _TwoShotPullAllReduceLaunch
    )
    launch = launch_cls(
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
    )
    cache_key = (
        "bf16",
        operation,
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
        int(row_elems),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    # The push kernel takes the per-source pack stride ahead of the reduced
    # region offset; the pull kernel does not need it.
    scalar_samples = (0, 1, 1, 1, 1, 0, 1) if mode == "push" else (0, 1, 1, 1, 0, 1)
    raw = b12x_compile(
        launch,
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            for _ in range(9)
        ),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            for _ in range(9)
        ),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        *scalar_samples,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            f"comm.pcie.twoshot_bf16.{operation}",
            2,
            cache_key,
        ),
    )

    def run(
        payload_address: int,
        staging_addresses: Sequence[int],
        signal_addresses: Sequence[int],
        output_address: int,
        rank: int,
        reduced_offset: int,
        slot_bytes: int,
        rows_per_rank: int,
        remainder_packs: int,
        grid_x: int,
        pack_stride: int = 0,
    ) -> None:
        if len(staging_addresses) != 9 or len(signal_addresses) != 9:
            raise ValueError("two-shot scalar pointer ABI requires nine peer slots")
        if not 0 <= int(remainder_packs) < world_size:
            raise ValueError("remainder_packs must be below the world size")
        if mode == "push":
            if pack_stride <= 0:
                raise ValueError("the push all-reduce needs a positive pack_stride")
            scalars = (rank, pack_stride, reduced_offset, slot_bytes,
                       rows_per_rank, remainder_packs, grid_x)
        else:
            scalars = (rank, reduced_offset, slot_bytes, rows_per_rank,
                       remainder_packs, grid_x)
        raw_args = (
            make_ptr(
                cutlass.Uint32,
                payload_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                for address in staging_addresses
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for address in signal_addresses
            ),
            make_ptr(
                cutlass.Uint32,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            *scalars,
            current_cuda_stream(),
        )
        raw(*raw_args)

    _PREPARED_BF16_LAUNCHERS.add(process_key)
    return run


def is_twoshot_bf16_allreduce_launcher_prepared(
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
    mode: str = "pull",
) -> bool:
    return (
        _bf16_process_key(
            f"all_reduce_{mode}",
            world_size,
            rank,
            device_slot_selection,
            slot_bias,
            threads,
            row_elems,
            device_index,
        )
        in _PREPARED_BF16_LAUNCHERS
    )


__all__ = [
    "get_twoshot_bf16_launcher",
    "is_twoshot_bf16_launcher_prepared",
    "get_twoshot_bf16_allreduce_launcher",
    "is_twoshot_bf16_allreduce_launcher_prepared",
]
