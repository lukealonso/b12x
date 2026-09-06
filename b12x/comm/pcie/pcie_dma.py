"""CE-driven PCIe ring allreduce for prefill-size tensors.

NCCL's SM-copy transport sustains ~34 GB/s bus bandwidth on this fabric
while CE peer copies run at ~56 GB/s on every ring hop concurrently
(including the two root-complex crossings, which each own a partition
uplink per direction). This runtime drives a classic reduce-scatter +
all-gather ring where the data plane is CE copies and the SM only
synchronizes (monotonic flag kernels) and reduces, so captured graphs
replay without host patching.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import lru_cache
from statistics import median
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import PCIeOneshotAllReduce, _normalize_device

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 9, 10)
FLAG_STRIDE = 128
FLAG_SLOTS = 256
MAX_PIECES = 8
SCRATCH_ALIGN = 256
FP8_QUANT_BLOCK = 128
# Flag-slot ranges per collective. Every slot's monotonic counter is owned by
# exactly one op kind, so an all-reduce, a paired all-gather and a column
# reduce-scatter can be mixed in any order without agreeing on a piece count.
# The all-reduce ring (and the compressed all-to-all) publishes one slot per
# (step, piece) plus a done slot: at most 2 * (world - 1) * MAX_PIECES + 1
# slots, 145 at the largest supported world size. The paired all-gather uses
# one slot per step plus done (world slots); the column reduce-scatter one
# slot per (step, piece) with at most two pieces, plus done.
AR_SLOT_BASE = 0
AR_SLOT_COUNT = 2 * (max(SUPPORTED_WORLD_SIZES) - 1) * MAX_PIECES + 1
AG_PAIR_SLOT_BASE = 152
AG_PAIR_SLOT_COUNT = max(SUPPORTED_WORLD_SIZES)
RS_SLOT_BASE = 168
RS_MAX_PIECES = 2
RS_SLOT_COUNT = (max(SUPPORTED_WORLD_SIZES) - 1) * RS_MAX_PIECES + 1
assert AR_SLOT_BASE + AR_SLOT_COUNT <= AG_PAIR_SLOT_BASE
assert AG_PAIR_SLOT_BASE + AG_PAIR_SLOT_COUNT <= RS_SLOT_BASE
assert RS_SLOT_BASE + RS_SLOT_COUNT <= FLAG_SLOTS
# Replay entries used within this many ring operations of the newest one are
# never evicted: a borrowed static output is consumed by its caller before
# the caller's next op on the same entry, which in the Kimi-K3 layer is at
# most three ring ops later (attention all-reduce -> gather pair ->
# reduce-scatter -> MoE all-reduce).
REPLAY_EVICTION_GUARD_OPS = 3
RS_WIRE_MODES = ("fp32", "bf16")


def _fp8_mode() -> str:
    """Opt-in compressed wire transport mode.

    "ag" (also "1"): keep the saturated bf16 reduce-scatter ring and
    quantize only the allgather phase. Final values quantize exactly once
    at their owner and are forwarded verbatim around the ring, so the
    error cost is a single rounding while AG wire bytes halve.

    "ring": quantize every reduce-scatter hop and the allgather payload,
    keeping the saturated neighbor-only topology while halving both phases.

    "a2a": quantize-once all-to-all (two roundings, half the wire in both
    phases).

    "i8" / "i8_ring" / "i8_a2a": the matching topology with a symmetric
    signed-INT8 payload. Both codecs use the same layout: one payload byte
    per value plus one fp32 scale per 128 values.

    "mx" / "mx_ring" / "mx_a2a": standard MXFP8, with E4M3 payload values
    and one E8M0 scale per 32 values. Four scale bytes per 128 values retain
    the same wire footprint as the other compressed codecs.

    Every compressed mode materializes the locally owned reduced shard through
    the same wire payload as its peers. An all-reduce result must be
    rank-identical; retaining a pre-wire BF16 owner shard while peers dequantize
    that shard gives every TP rank a different replicated activation.
    """

    return _normalize_fp8_mode(os.getenv("B12X_PCIE_DMA_FP8", "0"))


def _normalize_fp8_mode(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return ""
    if raw in ("a2a", "ring"):
        return raw
    if raw in (
        "i8",
        "int8",
        "i8_ag",
        "i8-ag",
        "ag_i8",
        "int8_ag",
        "int8-ag",
    ):
        return "i8"
    if raw in ("i8_ring", "i8-ring", "int8_ring", "int8-ring", "ring_i8"):
        return "i8_ring"
    if raw in ("i8_a2a", "i8-a2a", "int8_a2a", "int8-a2a", "a2a_i8"):
        return "i8_a2a"
    if raw in (
        "mx",
        "mxfp8",
        "mx_ag",
        "mx-ag",
        "mxfp8_ag",
        "mxfp8-ag",
        "ag_mx",
    ):
        return "mx"
    if raw in (
        "mx_ring",
        "mx-ring",
        "mxfp8_ring",
        "mxfp8-ring",
        "ring_mx",
    ):
        return "mx_ring"
    if raw in (
        "mx_a2a",
        "mx-a2a",
        "mxfp8_a2a",
        "mxfp8-a2a",
        "a2a_mx",
    ):
        return "mx_a2a"
    if raw in ("ag", "1"):
        return "ag"
    raise ValueError(f"unrecognized PCIe DMA wire mode: {value!r}")


@lru_cache(maxsize=1)
def _load_kernels():
    """Return the CuTe/Python transport primitives.

    Importing the CuTe module stays lazy so the public comm package remains
    importable in CPU-only tooling.
    """

    from ._dma_kernels import DmaKernels

    return DmaKernels(CudaRTLibrary())


def _graph_replay_mode() -> bool:
    """Opt-in CUDA-graph replay of eager ring all-reduces.

    An eager call issues ~250 CUDA API calls from Python (per-piece copies,
    flag kernels, adds and the cross-stream events), which costs more host
    time than the transfer takes on the wire for prefill-size tensors. With
    replay enabled, every shape at or above ``_graph_replay_min_bytes()``
    is captured once into a CUDA graph over static buffers and later calls
    copy in, replay, and copy out. The kernels and their order are unchanged,
    so the result is bit-identical to the eager path. A caller that writes
    the static input itself (``all_reduce_input``) and consumes the static
    output directly (``borrow_output``) skips both copies. The static buffers
    for every cache entry are reserved when the ring is constructed
    (max_entries x 2 x max_bytes), so serving never allocates and never
    falls back for lack of device memory.
    """
    return os.getenv("B12X_PCIE_DMA_GRAPH_REPLAY", "0") == "1"


def _graph_replay_min_bytes() -> int:
    return int(os.getenv("B12X_PCIE_DMA_GRAPH_REPLAY_MIN_BYTES", str(8 << 20)))


def _graph_replay_max_entries() -> int:
    """Upper bound on cached shapes; the least recently used one is dropped
    first. Each entry holds two static buffers of the tensor size."""
    return max(1, int(os.getenv("B12X_PCIE_DMA_GRAPH_REPLAY_MAX_ENTRIES", "4")))


class _ReplayEntry:
    """A captured ring graph with its static buffers.

    ``inputs``/``outputs`` are the static views a caller may write into
    (``inputs``, via the ``*_input`` accessors) or read directly after the op
    (``outputs``, when borrowing). The lossless all-reduce entry is in place:
    its single static buffer is both input and output, so a producer that
    writes it and a consumer that reads it need no staging copy at all.
    ``scratch`` holds op-private static storage (the running partial sums of
    the reduce-scatter). ``last_use`` is the ring-op sequence number of the
    last issue; entries used within ``REPLAY_EVICTION_GUARD_OPS`` of the
    newest op are not evicted.
    """

    __slots__ = (
        "key",
        "inputs",
        "outputs",
        "scratch",
        "graph",
        "slot",
        "last_use",
    )

    def __init__(
        self,
        key: tuple,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
        graph,
        slot: int,
        scratch: tuple[torch.Tensor, ...] = (),
    ) -> None:
        self.key = key
        self.inputs = inputs
        self.outputs = outputs
        self.scratch = scratch
        self.graph = graph
        self.slot = slot
        self.last_use = 0

    # Source-compatible aliases for the single-tensor all-reduce entry.
    @property
    def inp(self) -> torch.Tensor:
        return self.inputs[0]

    @property
    def out(self) -> torch.Tensor:
        return self.outputs[0]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.numel() == b.numel()
        and a.dtype == b.dtype
    )


def _byte_ranges_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    a_start = a.data_ptr()
    a_end = a_start + a.numel() * a.element_size()
    b_start = b.data_ptr()
    b_end = b_start + b.numel() * b.element_size()
    return a_start < b_end and b_start < a_end


def _rs_wire_mode(value: str | None) -> str:
    mode = (value or "fp32").strip().lower()
    if mode not in RS_WIRE_MODES:
        raise ValueError(
            f"reduce-scatter wire must be one of {RS_WIRE_MODES}, got {value!r}"
        )
    return mode


class PCIeDmaAllReduce:
    """Single-channel ring allreduce over IPC scratch buffers.

    A channel is a single ordered stream context; concurrent use from
    multiple CUDA streams needs separate channels (same contract as the
    oneshot runtime).
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_bytes: int,
        ext_module=None,
        fp8: Optional[str] = None,
    ) -> None:
        # Kept only as a source-compatible keyword for callers that used to
        # inject the removed C++ extension.  Device work is always CuTe DSL.
        del ext_module
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = _normalize_device(device)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "PCIe DMA all-reduce supports only the reviewed world sizes "
                f"{SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )
        if self.device.type != "cuda":
            raise ValueError("PCIe ring allreduce requires a CUDA device")
        self.max_bytes = int(max_bytes)
        self._wire_input = None
        self._wire_output = None
        if self.world_size == 9:
            # Eight-element shards keep vector accesses aligned for every
            # supported dtype. Wire tails do not change the model tensors.
            wire_bytes = _align_up(self.max_bytes, self.world_size * 8 * 4)
            self._wire_input = torch.empty(
                wire_bytes, dtype=torch.uint8, device=self.device
            )
            self._wire_output = torch.empty_like(self._wire_input)
        self._kernels = _load_kernels()
        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._closed = False

        self.shard_capacity = _align_up(
            (self.max_bytes + self.world_size - 1) // self.world_size, SCRATCH_ALIGN
        )
        steps = 2 * (self.world_size - 1)
        flags_bytes = FLAG_SLOTS * FLAG_STRIDE
        slab_bytes = flags_bytes + steps * self.shard_capacity
        self._slab = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group, slab_bytes, zero_fill=True, ipc=self._ipc
        )
        self._flags_base = list(self._slab.peer_ptrs)
        self._scratch_base = [ptr + flags_bytes for ptr in self._slab.peer_ptrs]
        # Device-resident monotonic counters: one per flag slot for the
        # publisher role and one for the waiter role.
        self._send_counters = torch.zeros(
            FLAG_SLOTS, dtype=torch.int32, device=self.device
        )
        self._wait_counters = torch.zeros(
            FLAG_SLOTS, dtype=torch.int32, device=self.device
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._flag_stream = torch.cuda.Stream(device=self.device)
        # Separate CE/flag streams for the a2a broadcast phase so allgather
        # traffic overlaps reduce-scatter traffic instead of queueing
        # behind it.
        self._ag_copy_stream = torch.cuda.Stream(device=self.device)
        self._ag_flag_stream = torch.cuda.Stream(device=self.device)
        # Persistent cross-stream events: captured graphs keep references to
        # recorded events, so per-call temporaries must not be destroyed.
        self._piece_events = [torch.cuda.Event() for _ in range(MAX_PIECES)]
        self._copied_events = [
            torch.cuda.Event() for _ in range(2 * (self.world_size - 1) * MAX_PIECES)
        ]
        self._input_ready = torch.cuda.Event()
        self._ag_ready = torch.cuda.Event()
        self._a2a_qdone = [torch.cuda.Event() for _ in range(MAX_PIECES)]
        self._a2a_ownq = [torch.cuda.Event() for _ in range(MAX_PIECES)]
        # Explicit argument wins over the environment so integrations can
        # plumb the mode through their own configuration.
        self._fp8 = _normalize_fp8_mode(fp8) if fp8 is not None else _fp8_mode()
        self._fp8_stage = None
        self._fp8_stage_stride = 0
        if self._fp8:
            max_shard_elems = self.max_bytes // 2 // self.world_size
            stride = _align_up(
                max_shard_elems + max_shard_elems // FP8_QUANT_BLOCK * 4,
                SCRATCH_ALIGN,
            )
            self._fp8_stage = torch.empty(
                self.world_size * stride, dtype=torch.uint8, device=self.device
            )
            self._fp8_stage_stride = stride
        self.min_bytes = 0
        self._graph_replay = _graph_replay_mode()
        self._graph_replay_min_bytes = _graph_replay_min_bytes()
        self._graph_replay_max_entries = _graph_replay_max_entries()
        # Replay key ``(op, numel, dtype)`` -> captured graph with its static
        # buffers (insertion order doubles as the LRU order); a shape is
        # captured on its second eager-eligible call so one-off sizes
        # (prefill tail chunks) stay eager instead of churning captures. The
        # op tag keeps an all-reduce, a paired all-gather and a column
        # reduce-scatter of equal element counts on separate entries.
        self._replay_entries: "OrderedDict[tuple, _ReplayEntry]" = OrderedDict()
        self._replay_seen: dict[tuple, int] = {}
        self._replay_capture_stream: torch.cuda.Stream | None = None
        # Ring-op sequence number: every issued op (eager or replayed)
        # increments it; entries record it as their last use.
        self._op_seq = 0
        self._rs_prepared_wires: set[str] = set()
        # The lossless ring reads each input chunk before the step that
        # overwrites the same chunk of the output (the reduce adds are
        # elementwise on one chunk; the first send of the owner chunk
        # completes, through the flag chain around the ring, before the
        # all-gather step that receives that chunk), so its replay entry can
        # use one buffer as both static input and output. The compressed
        # wire modes keep separate buffers.
        self._replay_in_place = not self._fp8
        # Static storage for every cached shape is reserved here, once, so a
        # capture during serving never allocates device memory (and can never
        # fail for lack of it): one slot of 2 x max_bytes per cache entry,
        # carved into views per op (an in-place all-reduce uses one buffer of
        # the tensor size; a paired all-gather or a reduce-scatter carves its
        # inputs, outputs and partial sums out of the same slot).
        self._replay_slot_bytes = 2 * _align_up(self.max_bytes, SCRATCH_ALIGN)
        self._replay_arena: torch.Tensor | None = None
        self._replay_free_slots: list[int] = []
        if self._graph_replay:
            self._replay_arena = torch.empty(
                self._graph_replay_max_entries * self._replay_slot_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
            self._replay_free_slots = list(range(self._graph_replay_max_entries))
        wire_modes = {
            "i8": "int8-ag",
            "i8_ring": "int8-ring",
            "i8_a2a": "int8-a2a",
            "mx": "mxfp8-ag",
            "mx_ring": "mxfp8-ring",
            "mx_a2a": "mxfp8-a2a",
        }
        self.wire_mode = wire_modes.get(
            self._fp8, f"fp8-{self._fp8}" if self._fp8 else "bf16"
        )
        logger.debug("[PCIe DMA allreduce] wire mode: %s", self.wire_mode)
        prepare = getattr(self._kernels, "prepare", None)
        if prepare is not None:
            prepare(world_size=self.world_size, wire_mode=self._fp8)
        if logger.isEnabledFor(logging.DEBUG):
            self._log_peer_copy_bandwidth()

    def _log_peer_copy_bandwidth(self, iters: int = 20) -> None:
        """One-time raw cudaMemcpyAsync bandwidth check, bypassing the ring
        schedule and flag sync entirely, so a slow deployment environment
        shows up here (bandwidth) rather than only in the full ring's
        latency (which would also be sensitive to sync/launch overhead).

        Every rank concurrently writes step 1 of its successor's scratch
        from step 0 of its own; no rank's step 0 (read) or step 1 (write)
        is touched by anyone else, so this measures true full-ring-style
        concurrent peer bandwidth with no self-inflicted read/write race.
        """

        if self.world_size < 2 or 2 * (self.world_size - 1) < 2:
            return
        nxt = (self.rank + 1) % self.world_size
        probe_bytes = min(self.shard_capacity, 4 << 20)
        probe_bytes -= probe_bytes % 16
        if probe_bytes <= 0:
            return
        stream = torch.cuda.Stream(device=self.device)
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        dist.barrier(group=self.group, device_ids=[device_index])
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._kernels.dma_copy(
                    self._scratch_ptr(nxt, 1),
                    self._scratch_ptr(self.rank, 0),
                    probe_bytes,
                )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for _ in range(iters):
                self._kernels.dma_copy(
                    self._scratch_ptr(nxt, 1),
                    self._scratch_ptr(self.rank, 0),
                    probe_bytes,
                )
            end.record(stream)
        stream.synchronize()
        ms = start.elapsed_time(end)
        gbps = probe_bytes * iters / (ms * 1e-3) / 1e9
        logger.debug(
            "[PCIe DMA allreduce] rank %d -> %d raw peer copy: %.1f GB/s "
            "(%d bytes x %d iters)",
            self.rank,
            nxt,
            gbps,
            probe_bytes,
            iters,
        )

    def _flag_ptr(self, rank: int, slot: int) -> int:
        return self._flags_base[rank] + slot * FLAG_STRIDE

    def _counter_ptr(self, counters: torch.Tensor, slot: int) -> int:
        return counters.data_ptr() + slot * 4

    def _scratch_ptr(self, rank: int, step: int) -> int:
        return self._scratch_base[rank] + step * self.shard_capacity

    @staticmethod
    def _pick_pieces(shard_elems: int, shard_bytes: int) -> int:
        override = int(os.getenv("B12X_PCIE_DMA_PIECES", "0"))
        # pieces=2 measured best at every size (deeper chunking pays an
        # extra wait+add launch chain per piece on the main stream).
        candidates = (override,) if 1 <= override <= MAX_PIECES else (2,)
        for pieces in candidates:
            if shard_elems % (pieces * 8) == 0 and shard_bytes // pieces >= 512 << 10:
                return pieces
        return 1

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed or inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        numel = inp.numel()
        alignment = 8 if self.world_size == 9 else self.world_size * 8
        if numel <= 0 or numel % alignment != 0:
            return False
        size_bytes = numel * inp.element_size()
        if size_bytes < self.min_bytes:
            return False
        return inp.is_contiguous() and size_bytes <= self.max_bytes

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        borrow_output: bool = False,
    ) -> torch.Tensor:
        """Ring all-reduce of ``inp``.

        ``borrow_output=True`` lets a replayed call return the entry's static
        output instead of copying it into fresh storage. The returned tensor
        then aliases ring-owned memory that the next all-reduce with the same
        ``(numel, dtype)`` overwrites, so the caller must consume it before
        issuing that op (and must not pass ``out``). When ``inp`` is the
        entry's static input (see ``all_reduce_input``; for the lossless
        in-place entry that is also the buffer a previous borrow returned),
        the input staging copy is skipped as well. The result is bit-identical
        either way because the captured kernels and their order do not
        change. Eager calls (unreplayed shapes) keep out-of-place semantics
        regardless of ``borrow_output``.
        """
        if borrow_output and out is not None:
            raise ValueError("borrow_output cannot be combined with out=")
        with torch.cuda.device(self.device):
            if self._graph_replay and self._graph_replay_eligible(inp):
                return self._all_reduce_replayed(inp, out, borrow_output)
            self._op_seq += 1
            return self._all_reduce_on_device(inp, out=out)

    def _graph_replay_eligible(self, inp: torch.Tensor) -> bool:
        # Inside an enclosing capture the eager sequence is recorded into
        # that graph as before; a graph cannot be replayed while capturing.
        if torch.cuda.is_current_stream_capturing():
            return False
        if inp.numel() * inp.element_size() < self._graph_replay_min_bytes:
            return False
        return self.should_allreduce(inp)

    @staticmethod
    def _all_reduce_key(inp: torch.Tensor) -> tuple:
        return ("ar", inp.numel(), inp.dtype)

    def _replay_entry_for(
        self, key: tuple, capture
    ) -> _ReplayEntry | None:
        """Return the replay entry for ``key``, capturing it on the second
        request. ``capture`` builds the entry once a slot is available;
        ``None`` means the caller must run eagerly this time."""
        entry = self._replay_entries.get(key)
        if entry is not None:
            self._replay_entries.move_to_end(key)
            return entry
        seen = self._replay_seen.get(key, 0) + 1
        if len(self._replay_seen) > 1024:
            self._replay_seen.clear()
        self._replay_seen[key] = seen
        if seen < 2:
            return None
        if not self._reserve_replay_slot():
            return None
        entry = capture()
        self._replay_entries[key] = entry
        logger.debug(
            "[PCIe DMA ring] captured replay graph for key=%s (%d cached)",
            key,
            len(self._replay_entries),
        )
        return entry

    def _reserve_replay_slot(self) -> bool:
        """Free one arena slot for a new entry, evicting the least recently
        used entry that no caller can still be reading. Returns False when
        every entry is inside the eviction guard window."""
        while (
            not self._replay_free_slots
            and len(self._replay_entries) >= self._graph_replay_max_entries
        ):
            victim_key = None
            for key, entry in self._replay_entries.items():
                if self._op_seq - entry.last_use > REPLAY_EVICTION_GUARD_OPS:
                    victim_key = key
                    break
            if victim_key is None:
                return False
            evicted = self._replay_entries.pop(victim_key)
            self._replay_free_slots.append(evicted.slot)
            del evicted
        assert self._replay_arena is not None
        return bool(self._replay_free_slots)

    def _slot_views(
        self, slot: int, specs: list[tuple[tuple[int, ...], torch.dtype]]
    ) -> list[torch.Tensor]:
        """Carve consecutive static views out of one arena slot."""
        assert self._replay_arena is not None
        base = slot * self._replay_slot_bytes
        limit = base + self._replay_slot_bytes
        views = []
        for shape, dtype in specs:
            numel = 1
            for dim in shape:
                numel *= int(dim)
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            if base + nbytes > limit:
                raise ValueError(
                    "replay arena slot cannot hold the requested static buffers"
                )
            views.append(self._replay_arena[base : base + nbytes].view(dtype).view(shape))
            base = _align_up(base + nbytes, SCRATCH_ALIGN)
        return views

    def _capture_graph(self, run) -> "torch.cuda.CUDAGraph":
        """Record ``run()`` into a CUDA graph on a joined side stream.

        Nothing executes during capture, and the device-side flag counters
        advance only when the graph runs, exactly as the eager kernels
        would. torch.cuda.graph() would also synchronize the device, run gc
        and empty the caching allocator; the ring allocates nothing inside
        the capture, so a bare capture suffices.
        """
        main = torch.cuda.current_stream(self.device)
        side = self._replay_capture_stream
        if side is None:
            side = self._replay_capture_stream = torch.cuda.Stream(device=self.device)
        side.wait_stream(main)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                run()
            finally:
                graph.capture_end()
        main.wait_stream(side)
        return graph

    def _all_reduce_replayed(
        self,
        inp: torch.Tensor,
        out: Optional[torch.Tensor],
        borrow_output: bool = False,
    ) -> torch.Tensor:
        key = self._all_reduce_key(inp)
        entry = self._replay_entry_for(key, lambda: self._capture_replay_entry(inp))
        self._op_seq += 1
        if entry is None:
            return self._all_reduce_on_device(inp, out=out)
        entry.last_use = self._op_seq
        # Entries are keyed by element count: the ring treats its input as a
        # flat buffer, so tensors of different shapes but equal size share a
        # graph. Copy through flat views so a [768, 7168] call can reuse the
        # [1536, 3584] entry (and vice versa). A producer that wrote into the
        # static input (``all_reduce_input``, or a borrowed output of the
        # in-place entry) skips this copy.
        if not _same_storage(entry.inp, inp):
            if _byte_ranges_overlap(entry.inp, inp):
                raise ValueError(
                    "all-reduce input partially overlaps the ring's static "
                    "buffer; pass the whole borrowed tensor or a copy"
                )
            entry.inp.view(-1).copy_(inp.view(-1))
        entry.graph.replay()
        if borrow_output:
            return entry.out.view(inp.shape)
        if out is None:
            out = torch.empty_like(inp)
        out.view(-1).copy_(entry.out.view(-1))
        return out

    def _capture_replay_entry(self, inp: torch.Tensor) -> _ReplayEntry:
        """Capture the eager ring sequence for ``inp``'s shape over static
        buffers. Every rank captures the same shape sequence, so the cache
        contents stay symmetric across the group."""
        slot = self._replay_free_slots.pop()
        spec = (tuple(inp.shape), inp.dtype)
        if self._replay_in_place:
            (static_in,) = self._slot_views(slot, [spec])
            static_out = static_in
        else:
            static_in, static_out = self._slot_views(slot, [spec, spec])
        graph = self._capture_graph(
            lambda: self._all_reduce_on_device(static_in, out=static_out)
        )
        return _ReplayEntry(
            self._all_reduce_key(inp), (static_in,), (static_out,), graph, slot
        )

    def all_reduce_input(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Return the static input of the replay entry for ``shape``/``dtype``
        so its producer can write the all-reduce input in place.

        ``None`` means the shape is not replayed (replay disabled, below the
        replay threshold, ineligible, first sighting, or no evictable slot);
        the caller then supplies its own buffer as before. Requesting the
        buffer counts as a sighting, so a shape is captured on the second
        request just as a plain all-reduce would be. The buffer is valid
        until the entry is evicted, which cannot happen while the caller's
        all-reduce of it is within ``REPLAY_EVICTION_GUARD_OPS`` ring ops.
        For the in-place entry this is the same buffer ``borrow_output``
        returns, so a caller chain producer -> all-reduce -> consumer ->
        producer stays inside one buffer.
        """
        if not self._graph_replay or torch.cuda.is_current_stream_capturing():
            return None
        probe = torch.empty(shape, dtype=dtype, device="meta")
        if probe.numel() * probe.element_size() < self._graph_replay_min_bytes:
            return None
        if self._closed or dtype not in SUPPORTED_DTYPES:
            return None
        numel = probe.numel()
        alignment = 8 if self.world_size == 9 else self.world_size * 8
        if numel <= 0 or numel % alignment != 0:
            return None
        if numel * probe.element_size() > self.max_bytes:
            return None
        with torch.cuda.device(self.device):
            entry = self._replay_entry_for(
                self._all_reduce_key(probe),
                lambda: self._capture_replay_entry(probe),
            )
        if entry is None:
            return None
        return entry.inp.view(shape)

    def is_ring_storage(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` lives inside the replay arena (a static input,
        static output or scratch view handed out by this ring)."""
        arena = self._replay_arena
        if arena is None or tensor.device != self.device:
            return False
        start = arena.data_ptr()
        end = start + arena.numel()
        ptr = tensor.data_ptr()
        return start <= ptr < end

    def _all_reduce_on_device(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy ring allreduce requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if out is None:
            # Preserve normal out-of-place collective semantics. Callers can
            # retain this result while a later collective is in flight.
            out = torch.empty_like(inp)
        elif (
            out.shape != inp.shape
            or out.dtype != inp.dtype
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "output must match input shape/dtype/device and be contiguous"
            )
        if self.world_size == 9 and inp.numel() % (self.world_size * 8):
            numel = inp.numel()
            wire_numel = _align_up(numel, self.world_size * 8)
            wire_bytes = wire_numel * inp.element_size()
            assert self._wire_input is not None and self._wire_output is not None
            wire_in = self._wire_input[:wire_bytes].view(inp.dtype)
            wire_out = self._wire_output[:wire_bytes].view(inp.dtype)
            wire_in[:numel].copy_(inp.view(-1))
            wire_in[numel:].zero_()
            self._all_reduce_aligned(wire_in, wire_out)
            out.view(-1).copy_(wire_out[:numel])
            return out
        return self._all_reduce_aligned(inp, out)

    def _all_reduce_aligned(
        self, inp: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Reduce equal, eight-element-aligned shards into the supplied output."""
        kernels = self._kernels
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        dtype_code = SUPPORTED_DTYPES[inp.dtype]
        elem = inp.element_size()
        shard_elems = inp.numel() // world
        shard_bytes = shard_elems * elem

        compressed_eligible = (
            bool(self._fp8)
            and inp.dtype == torch.bfloat16
            and shard_elems % FP8_QUANT_BLOCK == 0
        )
        int8_wire = compressed_eligible and self._fp8.startswith("i8")
        mxfp8_wire = compressed_eligible and self._fp8.startswith("mx")
        wire_codec = "i8" if int8_wire else "mx" if mxfp8_wire else "e4m3"
        if compressed_eligible and self._fp8 in ("a2a", "i8_a2a", "mx_a2a"):
            return self._all_reduce_fp8(inp, out, shard_elems, wire_codec=wire_codec)
        compressed_ring = compressed_eligible and self._fp8 in (
            "ring",
            "i8_ring",
            "mx_ring",
        )
        compressed_ag = compressed_eligible and self._fp8 in (
            "ag",
            "ring",
            "i8",
            "i8_ring",
            "mx",
            "mx_ring",
        )
        if wire_codec == "i8":
            quantize = kernels.dma_quant_i8
            dequantize_store = kernels.dma_dequant_store_i8
            dequantize_add_quant = kernels.dma_dequant_add_quant_i8
        elif wire_codec == "mx":
            quantize = kernels.dma_quant_mx
            dequantize_store = kernels.dma_dequant_store_mx
            dequantize_add_quant = kernels.dma_dequant_add_quant_mx
        else:
            quantize = kernels.dma_quant
            dequantize_store = kernels.dma_dequant_store
            dequantize_add_quant = kernels.dma_dequant_add_quant

        base = out.data_ptr()

        # Sub-chunking with a dedicated copy stream keeps the copy engine
        # busy: the CE never waits for a flag round trip or an add because
        # sub-chunk c+1's copy overlaps sub-chunk c's wait+reduce. Deeper
        # chunking amortizes the flag round trip further as long as each
        # piece's copy time dominates the ~5us sub-step overhead.
        pieces = self._pick_pieces(shard_elems, shard_bytes)
        if compressed_ag and (shard_elems // pieces) % FP8_QUANT_BLOCK != 0:
            pieces = 1
        piece_elems = shard_elems // pieces
        piece_bytes = piece_elems * elem
        # Compressed slices are piece-contiguous: [payload][scales] per piece.
        piece_slice_bytes = piece_elems + piece_elems // FP8_QUANT_BLOCK * 4
        steps = 2 * (world - 1)

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream

        # No upfront out.copy_(inp): the first send of each chunk reads the
        # caller's input directly and every reduce-scatter add is a first
        # touch (out = inp + scratch), so the accumulation base folds into
        # the add instead of a full-size copy on the critical path.
        in_base = inp.data_ptr()
        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)

        def piece_ptr(chunk: int, piece: int) -> int:
            return base + chunk * shard_bytes + piece * piece_bytes

        def in_piece_ptr(chunk: int, piece: int) -> int:
            return in_base + chunk * shard_bytes + piece * piece_bytes

        def scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_bytes

        def slot(step: int, piece: int) -> int:
            return step * pieces + piece

        # Events gating each step's send on the previous step's reduce of
        # the same payload piece (persistent; re-recorded per step). Flag
        # kernels run on their own stream, gated per copy by copied[] events,
        # so the copy stream is pure back-to-back CE work: an SM kernel
        # between CE ops stalls the engine for the launch round trip, which
        # is what made deeper sub-chunking regress.
        add_done = self._piece_events
        flag_stream = self._flag_stream
        copied = self._copied_events
        flag_stream.wait_event(self._input_ready)

        def fp8_scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_slice_bytes

        stage = self._fp8_stage.data_ptr() if compressed_ag else 0

        def fp8_stage_piece(chunk: int, piece: int) -> int:
            return stage + chunk * self._fp8_stage_stride + piece * piece_slice_bytes

        for k in range(steps):
            reduce_phase = k < world - 1
            if reduce_phase:
                send_chunk = (rank - k) % world
                recv_chunk = (rank - k - 1) % world
            else:
                send_chunk = (rank + 1 - (k - (world - 1))) % world
                recv_chunk = (rank - (k - (world - 1))) % world
            compressed_reduce = compressed_ring and reduce_phase
            compressed_step = compressed_reduce or (compressed_ag and not reduce_phase)
            if compressed_step and k == world - 1:
                # The AG-only mode quantizes the fully reduced owner chunk
                # here.  The FP8 ring's fused final reduce hop already
                # emitted the same payload.  Both modes forward those bytes
                # verbatim, with no additional all-gather rounding.
                if not compressed_ring:
                    for p in range(pieces):
                        ag_stage = fp8_stage_piece(send_chunk, p)
                        quantize(
                            piece_ptr(send_chunk, p),
                            ag_stage,
                            ag_stage + piece_elems,
                            piece_elems,
                        )
                # Publish the payload before the local materialization so the
                # CE broadcast can overlap this read-only dequant kernel.
                self._ag_ready.record(main)
                # The owner used to retain its pre-wire BF16 shard while the
                # other ranks materialized this same shard from FP8.  That
                # violates the replicated-output contract of all-reduce and
                # lets the next TP layer consume rank-dependent activations.
                # Round-trip the owner through the exact forwarded payload so
                # all ranks receive bit-identical BF16 values for every shard.
                for p in range(pieces):
                    owner_stage = fp8_stage_piece(send_chunk, p)
                    dequantize_store(
                        piece_ptr(send_chunk, p),
                        owner_stage,
                        owner_stage + piece_elems,
                        piece_elems,
                    )
            for p in range(pieces):
                if compressed_reduce:
                    send_src = fp8_stage_piece(send_chunk, p)
                    if k == 0:
                        quantize(
                            in_piece_ptr(send_chunk, p),
                            send_src,
                            send_src + piece_elems,
                            piece_elems,
                        )
                        self._a2a_qdone[p].record(main)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                elif not compressed_step:
                    send_src = (
                        in_piece_ptr(send_chunk, p)
                        if k == 0
                        else piece_ptr(send_chunk, p)
                    )
                    send_bytes = piece_bytes
                    send_dst = scratch_piece(nxt, k, p)
                elif k == world - 1:
                    send_src = fp8_stage_piece(send_chunk, p)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                else:
                    send_src = fp8_scratch_piece(rank, k - 1, p)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                with torch.cuda.stream(copy_stream):
                    if compressed_reduce:
                        copy_stream.wait_event(
                            self._a2a_qdone[p] if k == 0 else add_done[p]
                        )
                    elif compressed_step and k == world - 1:
                        copy_stream.wait_event(self._ag_ready)
                    elif k > 0:
                        copy_stream.wait_event(add_done[p])
                    kernels.dma_copy(send_dst, send_src, send_bytes)
                    copied[slot(k, p)].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied[slot(k, p)])
                    kernels.dma_set_flag(
                        self._flag_ptr(nxt, slot(k, p)),
                        self._counter_ptr(self._send_counters, slot(k, p)),
                    )
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot(k, p)),
                    self._counter_ptr(self._wait_counters, slot(k, p)),
                )
                if reduce_phase:
                    if compressed_reduce:
                        payload = fp8_scratch_piece(rank, k, p)
                        reduced = fp8_stage_piece(recv_chunk, p)
                        dequantize_add_quant(
                            piece_ptr(recv_chunk, p),
                            in_piece_ptr(recv_chunk, p),
                            payload,
                            payload + piece_elems,
                            reduced,
                            reduced + piece_elems,
                            piece_elems,
                            k == world - 2,
                        )
                    else:
                        kernels.dma_add(
                            piece_ptr(recv_chunk, p),
                            in_piece_ptr(recv_chunk, p),
                            scratch_piece(rank, k, p),
                            piece_elems,
                            dtype_code,
                        )
                elif compressed_step:
                    payload = fp8_scratch_piece(rank, k, p)
                    # Forwarding reads the received FP8 payload verbatim, so
                    # it only depends on the receive flag, not on the local
                    # BF16 materialization below.  Publish readiness before
                    # dequantization to overlap the next hop's CE copy with
                    # this rank's read-only dequant/store.
                    add_done[p].record(main)
                    dequantize_store(
                        piece_ptr(recv_chunk, p),
                        payload,
                        payload + piece_elems,
                        piece_elems,
                    )
                else:
                    kernels.dma_copy(
                        piece_ptr(recv_chunk, p),
                        scratch_piece(rank, k, p),
                        piece_bytes,
                    )
                if reduce_phase or not compressed_step:
                    add_done[p].record(main)

        # Neighbor handshake so the next call (or graph replay) cannot
        # overwrite scratch a lagging neighbor still reads. The main stream
        # must also drain the copy and flag streams before the op is done.
        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        done = steps * pieces
        kernels.dma_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        kernels.dma_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )
        return out

    # ------------------------------------------------------------------
    # Paired all-gather (copy-engine ring, rank-major outputs)
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_layout(
        first: torch.Tensor, second: torch.Tensor
    ) -> tuple[int, int, int]:
        """Return (first_bytes, second_offset, payload_bytes) of one rank's
        block in the per-step scratch area: the second tensor's block follows
        the first at the next 256-byte boundary."""
        first_bytes = first.numel() * first.element_size()
        second_offset = _align_up(first_bytes, SCRATCH_ALIGN)
        return (
            first_bytes,
            second_offset,
            second_offset + second.numel() * second.element_size(),
        )

    def should_all_gather_pair(self, first: torch.Tensor, second: torch.Tensor) -> bool:
        """Whether ``all_gather_pair`` accepts these rank-local blocks."""
        if self._closed or self.world_size < 2:
            return False
        for tensor in (first, second):
            if (
                tensor.device != self.device
                or tensor.dtype not in SUPPORTED_DTYPES
                or tensor.ndim != 2
                or tensor.numel() <= 0
                or not tensor.is_contiguous()
            ):
                return False
        if first.shape[0] != second.shape[0]:
            return False
        return self._pair_layout(first, second)[2] <= self.shard_capacity

    @staticmethod
    def _all_gather_pair_key(first: torch.Tensor, second: torch.Tensor) -> tuple:
        return (
            "ag_pair",
            (first.numel(), second.numel()),
            (first.dtype, second.dtype),
            (first.shape[0], first.shape[1], second.shape[1]),
        )

    def all_gather_pair(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather two rank-local ``[rows, c]`` blocks in one ring pass.

        Returns rank-major ``[world, rows, c_first]`` and
        ``[world, rows, c_second]`` tensors whose block ``r`` is rank ``r``'s
        input, byte for byte (an all-gather only copies). Both blocks travel
        the ring together: one flag per step gates one copy-engine transfer of
        the combined payload, so the SM only runs the single-thread flag
        kernels. Issued on a side stream, the gather overlaps whatever the
        caller keeps on its main stream; the caller orders the consumer after
        the side stream with an event. The op shares the channel's scratch
        and streams with the all-reduce and reduce-scatter, so it must be
        ordered against them like any other op on this channel.

        Replayed calls return the entry's static outputs; they stay valid until
        the next ``all_gather_pair`` with the same key, and a producer that
        wrote into ``all_gather_pair_inputs`` skips the input staging copies.
        Eager calls allocate their outputs on the current stream.
        """
        if not self.should_all_gather_pair(first, second):
            raise ValueError(
                "inputs do not satisfy paired all-gather requirements "
                f"(shapes={tuple(first.shape)}, {tuple(second.shape)}, "
                f"dtypes={first.dtype}, {second.dtype})"
            )
        with torch.cuda.device(self.device):
            key = self._all_gather_pair_key(first, second)
            entry = None
            if self._graph_replay and not torch.cuda.is_current_stream_capturing():
                entry = self._replay_entry_for(
                    key, lambda: self._capture_all_gather_pair(first, second)
                )
            self._op_seq += 1
            if entry is None:
                out_first = first.new_empty((self.world_size, *first.shape))
                out_second = second.new_empty((self.world_size, *second.shape))
                self._all_gather_pair_on_device(first, second, out_first, out_second)
                return out_first, out_second
            entry.last_use = self._op_seq
            static_first, static_second = entry.inputs
            if not _same_storage(static_first, first):
                static_first.copy_(first)
            if not _same_storage(static_second, second):
                static_second.copy_(second)
            entry.graph.replay()
            return entry.outputs[0], entry.outputs[1]

    def all_gather_pair_inputs(
        self,
        first_shape: tuple[int, int],
        first_dtype: torch.dtype,
        second_shape: tuple[int, int],
        second_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Static inputs of the paired all-gather replay entry for these
        shapes, for producers that write the blocks in place (``None`` when
        the shapes are not replayed; same sighting rule as
        ``all_reduce_input``)."""
        if not self._graph_replay or torch.cuda.is_current_stream_capturing():
            return None
        first = torch.empty(first_shape, dtype=first_dtype, device="meta")
        second = torch.empty(second_shape, dtype=second_dtype, device="meta")
        if self._closed or self.world_size < 2:
            return None
        for tensor in (first, second):
            if tensor.dtype not in SUPPORTED_DTYPES or tensor.ndim != 2:
                return None
        if first.shape[0] != second.shape[0]:
            return None
        if self._pair_layout(first, second)[2] > self.shard_capacity:
            return None
        with torch.cuda.device(self.device):
            entry = self._replay_entry_for(
                self._all_gather_pair_key(first, second),
                lambda: self._capture_all_gather_pair(first, second),
            )
        if entry is None:
            return None
        return entry.inputs[0], entry.inputs[1]

    def _capture_all_gather_pair(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> _ReplayEntry:
        slot = self._replay_free_slots.pop()
        world = self.world_size
        static_first, static_second, out_first, out_second = self._slot_views(
            slot,
            [
                (tuple(first.shape), first.dtype),
                (tuple(second.shape), second.dtype),
                ((world, *first.shape), first.dtype),
                ((world, *second.shape), second.dtype),
            ],
        )
        graph = self._capture_graph(
            lambda: self._all_gather_pair_on_device(
                static_first, static_second, out_first, out_second
            )
        )
        return _ReplayEntry(
            self._all_gather_pair_key(first, second),
            (static_first, static_second),
            (out_first, out_second),
            graph,
            slot,
        )

    def _all_gather_pair_on_device(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        out_first: torch.Tensor,
        out_second: torch.Tensor,
    ) -> None:
        kernels = self._kernels
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        first_bytes, second_offset, payload_bytes = self._pair_layout(first, second)
        second_bytes = second.numel() * second.element_size()
        steps = world - 1

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        # Persistent events: the first half of ``copied`` gates each step's
        # flag on its transfer, the second half marks the receipt of each
        # step on the main stream (the next step's forwarding copy reads that
        # scratch area).
        copied = self._copied_events
        received = self._copied_events[len(self._copied_events) // 2 :]

        def out_block(tensor: torch.Tensor, block: int) -> int:
            return tensor.data_ptr() + block * tensor.stride(0) * tensor.element_size()

        def slot(step: int) -> int:
            return AG_PAIR_SLOT_BASE + step

        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)
        flag_stream.wait_event(self._input_ready)
        # Own block: the caller's inputs become output block ``rank``.
        kernels.dma_copy(out_block(out_first, rank), first.data_ptr(), first_bytes)
        kernels.dma_copy(out_block(out_second, rank), second.data_ptr(), second_bytes)

        for k in range(steps):
            # Step k forwards block (rank - k) % world: the own block at
            # k == 0, afterwards the block received at step k - 1, still in
            # scratch area k - 1 as one contiguous [first | second] payload.
            recv_block = (rank - k - 1) % world
            dst = self._scratch_ptr(nxt, k)
            with torch.cuda.stream(copy_stream):
                if k == 0:
                    kernels.dma_copy(dst, first.data_ptr(), first_bytes)
                    kernels.dma_copy(dst + second_offset, second.data_ptr(), second_bytes)
                else:
                    copy_stream.wait_event(received[k - 1])
                    kernels.dma_copy(dst, self._scratch_ptr(rank, k - 1), payload_bytes)
                copied[k].record(copy_stream)
            with torch.cuda.stream(flag_stream):
                flag_stream.wait_event(copied[k])
                kernels.dma_set_flag(
                    self._flag_ptr(nxt, slot(k)),
                    self._counter_ptr(self._send_counters, slot(k)),
                )
            kernels.dma_wait_flag(
                self._flag_ptr(rank, slot(k)),
                self._counter_ptr(self._wait_counters, slot(k)),
            )
            received[k].record(main)
            src = self._scratch_ptr(rank, k)
            kernels.dma_copy(out_block(out_first, recv_block), src, first_bytes)
            kernels.dma_copy(
                out_block(out_second, recv_block), src + second_offset, second_bytes
            )

        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        done = slot(steps)
        kernels.dma_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        kernels.dma_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )

    # ------------------------------------------------------------------
    # Column reduce-scatter (fp32 or bf16 wire)
    # ------------------------------------------------------------------

    @staticmethod
    def _rs_cols(width: int, world: int, cols: int | None = None) -> int:
        """Column-block width: the caller's ``cols`` (the consumer's shard
        width, at least ``ceil(K / world)``), or ``ceil(K / world)``."""
        minimum = (width + world - 1) // world
        return minimum if cols is None else int(cols)

    def should_reduce_scatter_columns(
        self, inp: torch.Tensor, *, cols: int | None = None
    ) -> bool:
        """Whether ``reduce_scatter_columns`` accepts ``inp`` (``[rows, K]``
        bf16, contiguous, on this device) with block width ``cols`` (default
        ``ceil(K / world)``; a caller's width must cover ``K`` with ``world``
        blocks). The block's element count must be a multiple of eight and
        its fp32 pieces must fit the scratch areas."""
        if self._closed or self.world_size < 2:
            return False
        if (
            inp.device != self.device
            or inp.dtype != torch.bfloat16
            or inp.ndim != 2
            or inp.numel() <= 0
            or not inp.is_contiguous()
        ):
            return False
        rows, width = inp.shape
        cols = self._rs_cols(width, self.world_size, cols)
        if cols < 1 or cols * self.world_size < width:
            return False
        shard_elems = rows * cols
        if shard_elems % 8:
            return False
        pieces = self._pick_pieces(shard_elems, shard_elems * 2)
        piece_elems = shard_elems // pieces
        # Every piece travels as fp32 on the wide-wire hops and owns one
        # scratch area per step; the area must hold the fp32 piece and the
        # slab's 2 * (world - 1) areas must cover (world - 1) * pieces steps.
        if piece_elems * 4 > self.shard_capacity:
            return False
        return pieces <= RS_MAX_PIECES

    @staticmethod
    def _reduce_scatter_key(inp: torch.Tensor, wire: str, cols: int) -> tuple:
        return ("rs_" + wire, inp.numel(), inp.dtype, tuple(inp.shape), int(cols))

    @staticmethod
    def _rs_partial_dtype(wire: str) -> torch.dtype:
        return torch.float32 if wire == "fp32" else torch.bfloat16

    def reduce_scatter_columns(
        self, inp: torch.Tensor, *, wire: str = "fp32", cols: int | None = None
    ) -> torch.Tensor:
        """Reduce ``inp`` (``[rows, K]`` bf16) across ranks and return this
        rank's column block ``[rows, cols]``: columns
        ``[rank * cols, (rank + 1) * cols)`` of the sum, zero beyond ``K``.
        ``cols`` defaults to ``ceil(K / world)``; a row-parallel consumer
        whose input shards are wider (an eight-aligned shard width) passes
        its own so the block is exactly its input shard.

        ``wire="fp32"``: the first hop carries the bf16 source, later hops an
        fp32 running sum; every add accumulates in fp32 and only the final
        hop rounds to bf16 (one rounding, the two-shot precision class).
        ``wire="bf16"``: every hop carries bf16 and every add rounds to bf16
        (the precision class of the ring all-reduce, in column-block order).
        Summation order for block ``c``: ``(((x[c+1] + x[c+2]) + ...) + x[c])``
        with rank indices modulo world.

        Replayed calls return the entry's static output, valid until the next
        ``reduce_scatter_columns`` with the same key.
        """
        wire = _rs_wire_mode(wire)
        if not self.should_reduce_scatter_columns(inp, cols=cols):
            raise ValueError(
                "input does not satisfy column reduce-scatter requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype}, cols={cols})"
            )
        with torch.cuda.device(self.device):
            self.prepare_reduce_scatter(wire)
            rows, width = inp.shape
            cols = self._rs_cols(width, self.world_size, cols)
            key = self._reduce_scatter_key(inp, wire, cols)
            entry = None
            if self._graph_replay and not torch.cuda.is_current_stream_capturing():
                entry = self._replay_entry_for(
                    key, lambda: self._capture_reduce_scatter(inp, wire, cols)
                )
            self._op_seq += 1
            if entry is None:
                blocks = inp.new_empty((self.world_size, rows, cols))
                out = inp.new_empty((rows, cols))
                partials = tuple(
                    inp.new_empty((rows, cols), dtype=self._rs_partial_dtype(wire))
                    for _ in range(2)
                )
                self._pack_column_blocks(inp, blocks)
                self._reduce_scatter_on_device(blocks, out, partials, wire)
                return out
            entry.last_use = self._op_seq
            self._pack_column_blocks(inp, entry.inputs[0])
            entry.graph.replay()
            return entry.outputs[0]

    def prepare_reduce_scatter(self, wire: str = "fp32") -> None:
        """Compile and warm the mixed-precision add kernels of ``wire`` before
        any graph capture can need them."""
        wire = _rs_wire_mode(wire)
        if wire in self._rs_prepared_wires:
            return
        prepare = getattr(self._kernels, "prepare_reduce_scatter", None)
        if prepare is not None:
            prepare(wire=wire)
        self._rs_prepared_wires.add(wire)

    def _pack_column_blocks(self, inp: torch.Tensor, blocks: torch.Tensor) -> None:
        """Relayout ``inp[rows, K]`` into ``blocks[world, rows, cols]``: block
        ``b`` holds columns ``[b*cols, (b+1)*cols)``, and the last block's
        columns past ``K`` are zero (they contribute nothing to the sum, so
        the consumer's padded weight rows see zeros)."""
        world, rows, cols = blocks.shape
        width = inp.shape[1]
        full = width // cols
        if full:
            blocks[:full].copy_(
                inp[:, : full * cols].view(rows, full, cols).permute(1, 0, 2)
            )
        rest = width - full * cols
        if rest:
            blocks[full, :, :rest].copy_(inp[:, full * cols :])
            blocks[full, :, rest:].zero_()

    def _capture_reduce_scatter(
        self, inp: torch.Tensor, wire: str, cols: int
    ) -> _ReplayEntry:
        slot = self._replay_free_slots.pop()
        world = self.world_size
        rows, _ = inp.shape
        partial_dtype = self._rs_partial_dtype(wire)
        specs = [
            ((world, rows, cols), inp.dtype),
            ((rows, cols), inp.dtype),
            ((rows, cols), partial_dtype),
            ((rows, cols), partial_dtype),
        ]
        views = self._slot_views(slot, specs)
        blocks, out = views[0], views[1]
        partials = tuple(views[2:])
        graph = self._capture_graph(
            lambda: self._reduce_scatter_on_device(blocks, out, partials, wire)
        )
        return _ReplayEntry(
            self._reduce_scatter_key(inp, wire, cols),
            (blocks,),
            (out,),
            graph,
            slot,
            scratch=partials,
        )

    def _reduce_scatter_on_device(
        self,
        blocks: torch.Tensor,
        out: torch.Tensor,
        partials: tuple[torch.Tensor, ...],
        wire: str,
    ) -> None:
        """Ring reduce-scatter over ``blocks[world, rows, cols]`` into
        ``out[rows, cols]`` (this rank's block).

        Step ``k`` sends block ``(rank - k - 1) % world`` and receives block
        ``(rank - k - 2) % world`` from the predecessor; after ``world - 1``
        steps rank ``r`` holds the full sum of block ``r``. The payload of
        step 0 is the sender's bf16 block; the payload of step ``k >= 1`` is
        the running sum produced at step ``k - 1``, held in ``partials``
        (two buffers alternating by step parity: fp32 for the fp32 wire, bf16
        for the bf16 wire). Every add reads the local bf16 block and the
        received payload; with the fp32 wire it accumulates in fp32 and only
        the final step rounds to bf16 into ``out``.
        """
        kernels = self._kernels
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        elem = blocks.element_size()
        shard_elems = out.numel()
        shard_bytes = shard_elems * elem
        pieces = self._pick_pieces(shard_elems, shard_bytes)
        piece_elems = shard_elems // pieces
        piece_bytes = piece_elems * elem
        steps = world - 1
        fp32_wire = wire == "fp32"
        if len(partials) != 2 or any(
            partial.dtype != self._rs_partial_dtype(wire) for partial in partials
        ):
            raise ValueError(
                f"{wire} wire reduce-scatter needs two "
                f"{self._rs_partial_dtype(wire)} partial buffers"
            )
        partial_bytes = piece_elems * partials[0].element_size()

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        copied = self._copied_events
        add_done = self._piece_events
        in_base = blocks.data_ptr()
        out_base = out.data_ptr()

        def in_piece(block: int, piece: int) -> int:
            return in_base + block * shard_bytes + piece * piece_bytes

        def partial_piece(step: int, piece: int) -> int:
            return partials[step % 2].data_ptr() + piece * partial_bytes

        def scratch_piece(owner: int, step: int, piece: int) -> int:
            # One scratch area per (step, piece): an fp32 piece can exceed
            # half an area, so pieces do not share one.
            return self._scratch_ptr(owner, step * pieces + piece)

        def slot(step: int, piece: int) -> int:
            return RS_SLOT_BASE + step * pieces + piece

        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)
        flag_stream.wait_event(self._input_ready)

        for k in range(steps):
            send_block = (rank - k - 1) % world
            recv_block = (rank - k - 2) % world
            last = k == steps - 1
            for p in range(pieces):
                if k == 0:
                    send_src = in_piece(send_block, p)
                    send_bytes = piece_bytes
                else:
                    send_src = partial_piece(k - 1, p)
                    send_bytes = partial_bytes
                with torch.cuda.stream(copy_stream):
                    if k > 0:
                        copy_stream.wait_event(add_done[p])
                    kernels.dma_copy(scratch_piece(nxt, k, p), send_src, send_bytes)
                    copied[slot(k, p) - RS_SLOT_BASE].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied[slot(k, p) - RS_SLOT_BASE])
                    kernels.dma_set_flag(
                        self._flag_ptr(nxt, slot(k, p)),
                        self._counter_ptr(self._send_counters, slot(k, p)),
                    )
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot(k, p)),
                    self._counter_ptr(self._wait_counters, slot(k, p)),
                )
                recv = scratch_piece(rank, k, p)
                own = in_piece(recv_block, p)
                dst = out_base + p * piece_bytes if last else partial_piece(k, p)
                if k >= 2 and not last:
                    # Step k's add writes the partial buffer of parity k % 2,
                    # which is the payload the copy engine is still reading for
                    # step k - 1: two buffers alternate, so a send has exactly
                    # one step of slack and nothing else in the schedule holds
                    # the main stream behind the copy engine (the flag waits
                    # only bind this rank to its predecessor's copies, and
                    # add_done runs the other way). Without this edge a rank
                    # whose outgoing hop is slower than the incoming chain
                    # overwrites a payload in flight, and the block that
                    # payload carries reaches its owner corrupted.
                    main.wait_event(copied[slot(k - 1, p) - RS_SLOT_BASE])
                if not fp32_wire or (k == 0 and last):
                    # bf16 payload in, bf16 out: the plain ring add (fp32 sum
                    # rounded once to bf16), which for a single-step ring is
                    # also the exact fp32-wire result.
                    kernels.dma_add(
                        dst, own, recv, piece_elems, SUPPORTED_DTYPES[blocks.dtype]
                    )
                elif k == 0:
                    kernels.dma_add_mixed(dst, own, recv, piece_elems, "bf16_bf16_f32")
                elif last:
                    kernels.dma_add_mixed(dst, own, recv, piece_elems, "bf16_f32_bf16")
                else:
                    kernels.dma_add_mixed(dst, own, recv, piece_elems, "bf16_f32_f32")
                add_done[p].record(main)

        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        done = RS_SLOT_BASE + steps * pieces
        kernels.dma_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        kernels.dma_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )

    def _pick_a2a_chunks(self, shard_elems: int) -> int:
        override = int(os.getenv("B12X_PCIE_DMA_A2A_CHUNKS", "0"))
        candidates = (override,) if 1 <= override <= MAX_PIECES else (4, 3, 2)
        for chunks in candidates:
            if (
                shard_elems % (chunks * FP8_QUANT_BLOCK) == 0
                and shard_elems // chunks >= 384 << 10
            ):
                return chunks
        return 1

    def _all_reduce_fp8(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        shard_elems: int,
        *,
        wire_codec: str = "e4m3",
    ) -> torch.Tensor:
        """Pipelined quantize-once compressed all-to-all.

        Slices are split into chunks; each chunk's quantize -> scatter ->
        fp32 dequant-accumulate -> quantize-once broadcast wave overlaps the
        next chunk's, with broadcast copies on their own CE stream so the
        two phases' wire time overlaps rather than queues.

        No end handshake is needed: a rank re-enters the op only after its
        stream finished, which required every peer's broadcast of every
        chunk and therefore every peer's accumulate and placement; peers'
        next-call writes are stream-ordered after that.
        """

        kernels = self._kernels
        if wire_codec == "i8":
            quantize = kernels.dma_quant_i8
            dequantize_accum = kernels.dma_dequant_accum_i8
            dequantize_store = kernels.dma_dequant_store_i8
        elif wire_codec == "mx":
            quantize = kernels.dma_quant_mx
            dequantize_accum = kernels.dma_dequant_accum_mx
            dequantize_store = kernels.dma_dequant_store_mx
        else:
            quantize = kernels.dma_quant
            dequantize_accum = kernels.dma_dequant_accum
            dequantize_store = kernels.dma_dequant_store
        world = self.world_size
        rank = self.rank
        shard_bytes = shard_elems * 2
        chunks = self._pick_a2a_chunks(shard_elems)
        chunk_elems = shard_elems // chunks
        chunk_bytes = chunk_elems * 2
        chunk_payload = chunk_elems
        chunk_slice = chunk_payload + chunk_elems // FP8_QUANT_BLOCK * 4
        in_base = inp.data_ptr()
        out_base = out.data_ptr()
        stage_base = self._fp8_stage.data_ptr()
        stride = self._fp8_stage_stride

        def stage_chunk(shard: int, c: int) -> int:
            return stage_base + shard * stride + c * chunk_slice

        def rs_chunk(owner: int, srcpos: int, c: int) -> int:
            return self._scratch_ptr(owner, srcpos) + c * chunk_slice

        def ag_chunk(owner: int, srcpos: int, c: int) -> int:
            return self._scratch_ptr(owner, (world - 1) + srcpos) + c * chunk_slice

        def rs_slot(srcpos: int, c: int) -> int:
            return srcpos * chunks + c

        def ag_slot(srcpos: int, c: int) -> int:
            return (world - 1) * chunks + srcpos * chunks + c

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        ag_copy = self._ag_copy_stream
        ag_flag = self._ag_flag_stream
        copied = self._copied_events
        half = len(copied) // 2
        peers = [(rank + 1 + i) % world for i in range(world - 1)]
        pos_at = [(rank - j - 1) % world for j in peers]

        # Quantize all outgoing chunks up front (cheap kernels on main);
        # per-chunk events let the scatter start as soon as its chunk is
        # ready while later quants still run.
        for c in range(chunks):
            for j in peers:
                quantize(
                    in_base + j * shard_bytes + c * chunk_bytes,
                    stage_chunk(j, c),
                    stage_chunk(j, c) + chunk_payload,
                    chunk_elems,
                )
            self._a2a_qdone[c].record(main)

        # Scatter: reduce-scatter slices, chunk-pipelined.
        for c in range(chunks):
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(self._a2a_qdone[c])
                for i, j in enumerate(peers):
                    kernels.dma_copy(
                        rs_chunk(j, pos_at[i], c), stage_chunk(j, c), chunk_slice
                    )
                    copied[i * chunks + c].record(copy_stream)
            with torch.cuda.stream(flag_stream):
                for i, j in enumerate(peers):
                    flag_stream.wait_event(copied[i * chunks + c])
                    slot = rs_slot(pos_at[i], c)
                    kernels.dma_set_flag(
                        self._flag_ptr(j, slot),
                        self._counter_ptr(self._send_counters, slot),
                    )

        # Accumulate own shard chunk by chunk; broadcast each chunk as soon
        # as it is reduced and quantized (once).
        for c in range(chunks):
            for i in range(world - 1):
                slot = rs_slot(i, c)
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._counter_ptr(self._wait_counters, slot),
                )
            payloads = [rs_chunk(rank, i, c) for i in range(world - 1)]
            scales = [ptr + chunk_payload for ptr in payloads]
            own = rank * shard_bytes + c * chunk_bytes
            dequantize_accum(
                out_base + own, in_base + own, payloads, scales, chunk_elems
            )
            quantize(
                out_base + own,
                stage_chunk(rank, c),
                stage_chunk(rank, c) + chunk_payload,
                chunk_elems,
            )
            # Publish the payload first so its CE broadcast can overlap the
            # local read-only materialization below.
            self._a2a_ownq[c].record(main)
            # Peers materialize this reduced shard from the broadcast FP8
            # payload.  The owner must do the same or every rank enters the
            # next TP layer with a different replicated activation.
            dequantize_store(
                out_base + own,
                stage_chunk(rank, c),
                stage_chunk(rank, c) + chunk_payload,
                chunk_elems,
            )
            with torch.cuda.stream(ag_copy):
                ag_copy.wait_event(self._a2a_ownq[c])
                for i, j in enumerate(peers):
                    kernels.dma_copy(
                        ag_chunk(j, pos_at[i], c), stage_chunk(rank, c), chunk_slice
                    )
                    copied[half + i * chunks + c].record(ag_copy)
            with torch.cuda.stream(ag_flag):
                for i, j in enumerate(peers):
                    ag_flag.wait_event(copied[half + i * chunks + c])
                    slot = ag_slot(pos_at[i], c)
                    kernels.dma_set_flag(
                        self._flag_ptr(j, slot),
                        self._counter_ptr(self._send_counters, slot),
                    )

        # Place incoming reduced shards.
        for c in range(chunks):
            for i in range(world - 1):
                src = peers[i]
                slot = ag_slot(i, c)
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._counter_ptr(self._wait_counters, slot),
                )
                payload = ag_chunk(rank, i, c)
                dequantize_store(
                    out_base + src * shard_bytes + c * chunk_bytes,
                    payload,
                    payload + chunk_payload,
                    chunk_elems,
                )
        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        main.wait_stream(ag_copy)
        main.wait_stream(ag_flag)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._replay_entries.clear()
        self._replay_arena = None
        # Drain the main and four auxiliary streams before any rank unmaps a
        # peer allocation.  Every importer must unmap before its owner frees
        # the exported slab.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for ptr in self._slab.remote_ptrs:
            with suppress(Exception):
                self._ipc.cudaIpcCloseMemHandle(ptr)
        dist.barrier(group=self.group)
        with suppress(Exception):
            self._ipc.cudaFree(self._slab.local_ptr)
        dist.barrier(group=self.group)

    def __enter__(self) -> "PCIeDmaAllReduce":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        # Distributed barriers are unsafe during asymmetric interpreter
        # teardown. Explicit/context-manager close owns coordinated release.
        return None


def autotune_crossovers(
    oneshot,
    dma: Optional[PCIeDmaAllReduce],
    nccl_group: ProcessGroup,
    *,
    hidden_size: int,
    max_rows: int,
    rms_norm_op=None,
    epsilon: float = 1e-6,
    warmup: int = 5,
    iters: int = 50,
    samples: int = 5,
    win_margin: float = 0.02,
) -> tuple[int, int]:
    """Single sweep from 1 row to the prefill chunk size with the real
    kernels: the oneshot channel (fused AR+RMSNorm when ``rms_norm_op`` is
    given, plain otherwise), the CE ring, and NCCL (plus ``rms_norm_op``)
    as the fallback. Returns (oneshot_max_bytes, dma_min_bytes) and sets
    ``dma.min_bytes``. Each timing is the median of multiple CUDA-event
    samples after MAX-reducing every sample across ranks. A backend must win
    by ``win_margin`` and DMA must do so at two consecutive sizes before its
    crossover is committed.
    """

    device = oneshot.device if oneshot is not None else dma.device
    stream = torch.cuda.Stream(device=device)
    dtype = torch.bfloat16
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    inf = float("inf")
    oneshot_max = 0
    dma_min = 0
    if dma is not None:
        original_dma_min = dma.min_bytes
        dma.min_bytes = 0
    wire = "bf16" if dma is None else dma.wire_mode
    lines = [
        f"[PCIe allreduce] Crossover sweep (dma wire={wire}, "
        f"hidden={hidden_size}, fused={rms_norm_op is not None}):"
    ]

    def bench(build) -> float:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            replay = build()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            replay()
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        dist.barrier(group=nccl_group, device_ids=[device_index])
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                graph.replay()
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        rank_max = torch.empty((), dtype=torch.float64, device=device)
        timings = []
        for _ in range(samples):
            dist.barrier(group=nccl_group, device_ids=[device_index])
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(iters):
                    graph.replay()
                end.record(stream)
            end.synchronize()
            rank_max.fill_(start.elapsed_time(end) * 1e3 / iters)
            dist.all_reduce(rank_max, op=dist.ReduceOp.MAX, group=nccl_group)
            timings.append(float(rank_max.item()))
        return float(median(timings))

    try:
        # Fully dense through 8 rows (the decode regime, where every row
        # count occurs), quarter steps through 32-128 rows (where the
        # NCCL/DMA boundary lives), and powers of two with midpoints
        # elsewhere. The sweep stops once the DMA allreduce has won twice
        # in a row: the curves are monotone above the boundary and the
        # large probes are the expensive ones.
        ladder = list(range(1, min(8, max_rows) + 1))
        step = 8
        while step <= max_rows:
            if step not in ladder:
                ladder.append(step)
            if 32 <= step <= 64:
                extra = (step + step // 4, step + step // 2, step + 3 * step // 4)
            else:
                extra = (step + step // 2,)
            ladder.extend(rows for rows in extra if rows <= max_rows)
            step *= 2
        oneshot_losses = 0
        dma_wins = 0
        dma_candidate = 0
        rank0 = dist.get_rank(group=nccl_group) == 0
        if rank0:
            logger.debug(lines[0])
        for rows in ladder:
            point_start = time.perf_counter()
            shape = (rows, hidden_size)
            size_bytes = rows * hidden_size * dtype.itemsize

            def build_nccl():
                inp = torch.randn(shape, dtype=dtype, device=device) * 0.01
                residual = torch.randn(shape, dtype=dtype, device=device)
                if rms_norm_op is None:
                    return lambda: dist.all_reduce(inp, group=nccl_group)
                return lambda: (
                    dist.all_reduce(inp, group=nccl_group),
                    rms_norm_op(inp, residual, weight, epsilon),
                )

            nccl_us = bench(build_nccl)

            # Stop probing the oneshot after it has clearly lost (its curve
            # is monotone against NCCL); a probe the kernel refuses (row or
            # capacity limits) counts as a loss. Every rank takes the same
            # branch because verdicts come from MAX-reduced timings.
            oneshot_us = inf
            if (
                oneshot is not None
                and oneshot_losses < 2
                and size_bytes <= oneshot.max_size
            ):

                def build_oneshot():
                    inp = torch.randn(shape, dtype=dtype, device=device) * 0.01
                    residual = torch.randn(shape, dtype=dtype, device=device)
                    out = torch.empty_like(inp)
                    residual_out = torch.empty_like(inp)
                    if rms_norm_op is None:
                        return lambda: oneshot.all_reduce(inp, out=out)
                    return lambda: oneshot.all_reduce_fused_add_rms_norm(
                        inp,
                        residual,
                        weight,
                        epsilon,
                        out=out,
                        residual_out=residual_out,
                    )

                try:
                    oneshot_us = bench(build_oneshot)
                except Exception:
                    oneshot_us = inf

            dma_us = inf
            probe = torch.empty(shape, dtype=dtype, device=device)
            if dma is not None and dma.should_allreduce(probe):

                def build_dma():
                    inp = torch.randn(shape, dtype=dtype, device=device) * 0.01
                    out = torch.empty_like(inp)
                    return lambda: dma.all_reduce(inp, out=out)

                dma_us = bench(build_dma)
            del probe

            stats = torch.tensor(
                [nccl_us, oneshot_us, dma_us], dtype=torch.float64, device=device
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=nccl_group)
            nccl_us, oneshot_us, dma_us = (float(v) for v in stats.tolist())
            oneshot_limit = (1.0 + win_margin) * min(nccl_us, dma_us)
            if oneshot_us < oneshot_limit:
                oneshot_max = size_bytes
                oneshot_losses = 0
            else:
                oneshot_losses += 1
            dma_limit = (1.0 - win_margin) * min(nccl_us, oneshot_us)
            if dma_us < dma_limit:
                if dma_wins == 0:
                    dma_candidate = size_bytes
                dma_wins += 1
            else:
                dma_wins = 0
                dma_candidate = 0
            line = (
                f"  rows={rows:5d} ({size_bytes >> 10:6d}KB): "
                f"oneshot {oneshot_us:9.1f}  dma {dma_us:9.1f}  "
                f"nccl {nccl_us:9.1f} us"
                f"  [{time.perf_counter() - point_start:.2f}s]"
            )
            lines.append(line)
            if rank0:
                logger.debug(line)
            if dma_wins >= 2:
                dma_min = dma_candidate
                break
    except Exception:
        if dma is not None:
            dma.min_bytes = original_dma_min
        raise

    if dma is not None:
        dma.min_bytes = dma_min if dma_min > 0 else dma.max_bytes + 1
    if dist.get_rank(group=nccl_group) == 0:
        logger.debug("  oneshot_max_bytes=%d dma_min_bytes=%d", oneshot_max, dma_min)
    return oneshot_max, dma_min


__all__ = ["PCIeDmaAllReduce", "autotune_crossovers"]
