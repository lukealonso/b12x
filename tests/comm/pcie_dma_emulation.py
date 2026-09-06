"""CPU emulation of the PCIe DMA ring for protocol-level tests.

One ``PCIeDmaAllReduce`` per rank runs in its own thread over host memory:

* every rank's slab (flag slots + scratch areas) is a CPU ``uint8`` tensor,
  so ``_flag_ptr`` / ``_scratch_ptr`` are real host addresses and the peer
  copies of the ring are ``ctypes.memmove``;
* the flag kernels implement the device protocol (publisher increments its
  counter and stores it at the peer's slot; the waiter increments its expected
  value and blocks until the slot reaches it) on a shared condition variable;
* the add kernels reproduce the device arithmetic (fp32 sums, one
  round-to-nearest-even to bf16 per bf16 store);
* CUDA streams and events are no-ops, so each rank executes its ring schedule
  in program order, which is one valid serialization of the multi-stream
  schedule (cross-stream races are outside what this harness can observe);
* a graph capture records the kernel calls made while capturing and replays
  them verbatim, so the replay path drives the same static-buffer addresses
  as the device graph would.

The emulated object mirrors the state ``PCIeDmaAllReduce.__init__`` builds
(``test_pcie_dma_replay_cpu.py`` checks the mirror against the constructor's
attribute list) so the ring's own methods run unmodified.
"""

from __future__ import annotations

import ctypes
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable

import torch

from b12x.comm.pcie import pcie_dma

_DTYPE_CODES = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}
_MIXED_MODES = {
    "bf16_bf16_f32": (torch.bfloat16, torch.float32),
    "bf16_f32_f32": (torch.float32, torch.float32),
    "bf16_f32_bf16": (torch.float32, torch.bfloat16),
}


def _view(ptr: int, elems: int, dtype: torch.dtype) -> torch.Tensor:
    nbytes = elems * torch.empty((), dtype=dtype).element_size()
    raw = (ctypes.c_uint8 * nbytes).from_address(int(ptr))
    return torch.frombuffer(raw, dtype=torch.uint8).view(dtype)


class _FakeStream:
    def __init__(self, device=None) -> None:
        self.device = device

    def wait_event(self, event) -> None:
        return None

    def wait_stream(self, stream) -> None:
        return None

    def synchronize(self) -> None:
        return None


class _FakeEvent:
    def __init__(self, enable_timing: bool = False) -> None:
        return None

    def record(self, stream=None) -> None:
        return None

    def wait(self, stream=None) -> None:
        return None


class _FakeGraph:
    """Records kernel calls during capture; ``replay`` re-issues them."""

    def __init__(self) -> None:
        self.calls: list[tuple[Callable[..., None], tuple[Any, ...]]] = []
        self._kernels: EmulatedKernels | None = None

    def capture_begin(self, capture_error_mode: str = "global") -> None:
        kernels = EmulatedKernels.current()
        if kernels is None:
            raise RuntimeError("capture outside an emulated ring thread")
        self._kernels = kernels
        kernels.begin_capture(self.calls)

    def capture_end(self) -> None:
        assert self._kernels is not None
        self._kernels.end_capture()

    def replay(self) -> None:
        for fn, args in self.calls:
            fn(*args)


class _FakeDeviceContext:
    def __init__(self, device) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


@contextmanager
def _fake_stream_context(stream):
    yield


def install_cuda_fakes(monkeypatch) -> None:
    """Replace the torch.cuda entry points the ring uses with the fakes."""
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "device", _FakeDeviceContext)
    monkeypatch.setattr(torch.cuda, "stream", _fake_stream_context)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeStream())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


class EmulatedKernels:
    """Host implementation of the ``DmaKernels`` facade shared by all ranks."""

    _local = threading.local()

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.copies = 0
        self.flags_set = 0
        self.flags_waited = 0
        self.adds = 0

    @classmethod
    def current(cls) -> "EmulatedKernels | None":
        return getattr(cls._local, "kernels", None)

    def bind_thread(self) -> None:
        self._local.kernels = self
        self._local.recording = None

    def begin_capture(self, calls: list) -> None:
        self._local.recording = calls

    def end_capture(self) -> None:
        self._local.recording = None

    def _issue(self, fn: Callable[..., None], *args: Any) -> None:
        recording = getattr(self._local, "recording", None)
        if recording is not None:
            recording.append((fn, args))
            return
        fn(*args)

    # -- primitives -------------------------------------------------------
    def dma_copy(self, dst_ptr: int, src_ptr: int, bytes_: int) -> None:
        self._issue(self._copy, int(dst_ptr), int(src_ptr), int(bytes_))

    def _copy(self, dst: int, src: int, nbytes: int) -> None:
        ctypes.memmove(dst, src, nbytes)
        self.copies += 1

    def dma_set_flag(self, peer_flag_ptr: int, counter_ptr: int) -> None:
        self._issue(self._set_flag, int(peer_flag_ptr), int(counter_ptr))

    def _set_flag(self, peer_flag_ptr: int, counter_ptr: int) -> None:
        with self._cond:
            counter = ctypes.c_int32.from_address(counter_ptr)
            counter.value = counter.value + 1
            ctypes.c_int32.from_address(peer_flag_ptr).value = counter.value
            self.flags_set += 1
            self._cond.notify_all()

    def dma_wait_flag(self, flag_ptr: int, counter_ptr: int) -> None:
        self._issue(self._wait_flag, int(flag_ptr), int(counter_ptr))

    def _wait_flag(self, flag_ptr: int, counter_ptr: int) -> None:
        with self._cond:
            counter = ctypes.c_int32.from_address(counter_ptr)
            counter.value = counter.value + 1
            expected = counter.value
            flag = ctypes.c_int32.from_address(flag_ptr)
            if not self._cond.wait_for(lambda: flag.value - expected >= 0, timeout=60):
                raise TimeoutError("emulated ring flag wait timed out")
            self.flags_waited += 1

    def dma_add(
        self, dst_ptr: int, a_ptr: int, b_ptr: int, elems: int, dtype_code: int
    ) -> None:
        self._issue(
            self._add, int(dst_ptr), int(a_ptr), int(b_ptr), int(elems), int(dtype_code)
        )

    def _add(self, dst: int, a: int, b: int, elems: int, dtype_code: int) -> None:
        dtype = _DTYPE_CODES[dtype_code]
        av = _view(a, elems, dtype).float()
        bv = _view(b, elems, dtype).float()
        _view(dst, elems, dtype).copy_((av + bv).to(dtype))
        self.adds += 1

    def dma_add_mixed(
        self, dst_ptr: int, a_ptr: int, b_ptr: int, elems: int, mode: str
    ) -> None:
        if mode not in _MIXED_MODES:
            raise ValueError(mode)
        self._issue(
            self._add_mixed, int(dst_ptr), int(a_ptr), int(b_ptr), int(elems), mode
        )

    def _add_mixed(self, dst: int, a: int, b: int, elems: int, mode: str) -> None:
        b_dtype, out_dtype = _MIXED_MODES[mode]
        av = _view(a, elems, torch.bfloat16).float()
        bv = _view(b, elems, b_dtype).float()
        _view(dst, elems, out_dtype).copy_((av + bv).to(out_dtype))
        self.adds += 1

    def prepare_reduce_scatter(self, *, wire: str) -> None:
        return None

    # The lossless ring binds the compressed-wire codecs by attribute even
    # when it never calls them; the emulation covers the lossless wire only.
    def _unsupported(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("compressed wire modes are not emulated")

    dma_quant = dma_dequant_store = dma_dequant_add_quant = _unsupported
    dma_quant_i8 = dma_dequant_store_i8 = dma_dequant_add_quant_i8 = _unsupported
    dma_quant_mx = dma_dequant_store_mx = dma_dequant_add_quant_mx = _unsupported
    dma_dequant_accum = dma_dequant_accum_i8 = dma_dequant_accum_mx = _unsupported


class EmulatedRing:
    """A ``world``-rank ring over host memory; ``rings[r]`` is rank ``r``."""

    def __init__(
        self,
        world: int,
        max_bytes: int,
        *,
        graph_replay: bool = True,
        replay_min_bytes: int = 0,
        replay_max_entries: int = 4,
    ) -> None:
        self.world = world
        self.kernels = EmulatedKernels()
        cls = pcie_dma.PCIeDmaAllReduce
        shard_capacity = pcie_dma._align_up(
            (max_bytes + world - 1) // world, pcie_dma.SCRATCH_ALIGN
        )
        steps = 2 * (world - 1)
        flags_bytes = pcie_dma.FLAG_SLOTS * pcie_dma.FLAG_STRIDE
        slab_bytes = flags_bytes + steps * shard_capacity
        self.slabs = [torch.zeros(slab_bytes, dtype=torch.uint8) for _ in range(world)]
        flags_base = [int(slab.data_ptr()) for slab in self.slabs]
        scratch_base = [ptr + flags_bytes for ptr in flags_base]
        device = torch.device("cpu")
        self.rings: list[pcie_dma.PCIeDmaAllReduce] = []
        for rank in range(world):
            ring = cls.__new__(cls)
            ring.group = None
            ring.rank = rank
            ring.world_size = world
            ring.device = device
            ring.max_bytes = int(max_bytes)
            ring._wire_input = None
            ring._wire_output = None
            if world == 9:
                wire_bytes = pcie_dma._align_up(ring.max_bytes, world * 8 * 4)
                ring._wire_input = torch.empty(wire_bytes, dtype=torch.uint8)
                ring._wire_output = torch.empty_like(ring._wire_input)
            ring._kernels = self.kernels
            ring._ipc = None
            ring._closed = False
            ring.shard_capacity = shard_capacity
            ring._slab = None
            ring._flags_base = list(flags_base)
            ring._scratch_base = list(scratch_base)
            ring._send_counters = torch.zeros(pcie_dma.FLAG_SLOTS, dtype=torch.int32)
            ring._wait_counters = torch.zeros(pcie_dma.FLAG_SLOTS, dtype=torch.int32)
            ring._copy_stream = _FakeStream(device)
            ring._flag_stream = _FakeStream(device)
            ring._ag_copy_stream = _FakeStream(device)
            ring._ag_flag_stream = _FakeStream(device)
            ring._piece_events = [_FakeEvent() for _ in range(pcie_dma.MAX_PIECES)]
            ring._copied_events = [
                _FakeEvent() for _ in range(2 * (world - 1) * pcie_dma.MAX_PIECES)
            ]
            ring._input_ready = _FakeEvent()
            ring._ag_ready = _FakeEvent()
            ring._a2a_qdone = [_FakeEvent() for _ in range(pcie_dma.MAX_PIECES)]
            ring._a2a_ownq = [_FakeEvent() for _ in range(pcie_dma.MAX_PIECES)]
            ring._fp8 = ""
            ring._fp8_stage = None
            ring._fp8_stage_stride = 0
            ring.min_bytes = 0
            ring._graph_replay = bool(graph_replay)
            ring._graph_replay_min_bytes = int(replay_min_bytes)
            ring._graph_replay_max_entries = int(replay_max_entries)
            ring._replay_entries = OrderedDict()
            ring._replay_seen = {}
            ring._replay_capture_stream = None
            ring._op_seq = 0
            ring._rs_prepared_wires = set()
            ring._replay_in_place = True
            ring._replay_slot_bytes = 2 * pcie_dma._align_up(
                ring.max_bytes, pcie_dma.SCRATCH_ALIGN
            )
            ring._replay_arena = None
            ring._replay_free_slots = []
            if graph_replay:
                ring._replay_arena = torch.empty(
                    ring._graph_replay_max_entries * ring._replay_slot_bytes,
                    dtype=torch.uint8,
                )
                ring._replay_free_slots = list(range(ring._graph_replay_max_entries))
            ring.wire_mode = "bf16"
            self.rings.append(ring)

    def run(self, fn: Callable[[pcie_dma.PCIeDmaAllReduce, int], Any]) -> list[Any]:
        """Run ``fn(ring, rank)`` on every rank concurrently; return results."""
        results: list[Any] = [None] * self.world
        errors: list[BaseException | None] = [None] * self.world

        def worker(rank: int) -> None:
            self.kernels.bind_thread()
            try:
                results[rank] = fn(self.rings[rank], rank)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors[rank] = exc
                with self.kernels._cond:
                    self.kernels._cond.notify_all()

        threads = [
            threading.Thread(target=worker, args=(rank,), name=f"rank{rank}")
            for rank in range(self.world)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        for rank, exc in enumerate(errors):
            if exc is not None:
                raise RuntimeError(f"rank {rank} failed") from exc
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("emulated ring did not finish")
        return results


def _chain_sum(
    terms: list[torch.Tensor], order: list[int], round_dtype: torch.dtype | None
) -> torch.Tensor:
    """``(((x[o0] + x[o1]) + x[o2]) + ...)`` in fp32, rounding the running
    sum to ``round_dtype`` after every add when given (``None``: fp32
    accumulation throughout)."""
    acc = terms[order[0]].float()
    for r in order[1:]:
        acc = terms[r].float() + acc
        if round_dtype is not None:
            acc = acc.to(round_dtype).float()
    return acc


def ring_all_reduce_reference(inputs: list[torch.Tensor], world: int) -> torch.Tensor:
    """The ring all-reduce's arithmetic: flat chunk ``c`` (``numel / world``
    elements) is summed as ``(((x[c] + x[c+1]) + x[c+2]) + ... + x[c-1])``
    with a bf16 rounding per add (fp32 for fp32 inputs is exact per add)."""
    flat = [x.reshape(-1) for x in inputs]
    dtype = flat[0].dtype
    shard = flat[0].numel() // world
    out = torch.empty_like(flat[0])
    for chunk in range(world):
        sl = slice(chunk * shard, (chunk + 1) * shard)
        terms = [x[sl] for x in flat]
        order = [(chunk + i) % world for i in range(world)]
        out[sl] = _chain_sum(terms, order, dtype).to(dtype)
    return out.view_as(inputs[0])


def column_reduce_scatter_reference(
    inputs: list[torch.Tensor], world: int, wire: str, cols: int | None = None
) -> list[torch.Tensor]:
    """Per-rank column block of the ring reduce-scatter: block ``c`` is summed
    as ``(((x[c+1] + x[c+2]) + ...) + x[c])``; ``wire="bf16"`` rounds after
    every add, ``wire="fp32"`` once at the end. The last block is zero past
    the input width. ``cols`` is the block width (default ``ceil(K/world)``)."""
    rows, width = inputs[0].shape
    if cols is None:
        cols = (width + world - 1) // world
    blocks = []
    for c in range(world):
        start = c * cols
        terms = []
        for x in inputs:
            block = x.new_zeros((rows, cols))
            valid = min(cols, width - start)
            if valid > 0:
                block[:, :valid] = x[:, start : start + valid]
            terms.append(block)
        order = [(c + 1 + i) % world for i in range(world)]
        acc = _chain_sum(terms, order, torch.bfloat16 if wire == "bf16" else None)
        blocks.append(acc.to(torch.bfloat16))
    return blocks
