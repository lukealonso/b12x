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
* each rank executes its ring schedule in program order, which is one valid
  serialization of the multi-stream schedule, so the values a test reads back
  are those of a schedule where every stream keeps up with the main stream;
* a graph capture records the kernel calls and the stream/event operations
  made while capturing and replays them verbatim, so the replay path drives
  the same static-buffer addresses and the same ordering the device graph
  would.

``ScheduleModel`` makes the orderings the schedule does *not* impose visible
even though the emulation runs in program order. Streams and events carry
vector clocks, the flag protocol contributes the cross-rank edges (a waiter
that observes counter value ``v`` is ordered after the publisher's stream at
the point it published ``v``), and every kernel declares the byte ranges it
reads and writes. Two accesses to overlapping bytes where at least one is a
write and neither happens before the other are recorded in
``ScheduleModel.conflicts``: on the device those two kernels may run
concurrently, so the result depends on which finishes first.

The emulated object mirrors the state ``PCIeDmaAllReduce.__init__`` builds
(``test_pcie_dma_replay_cpu.py`` checks the mirror against the constructor's
attribute list) so the ring's own methods run unmodified.
"""

from __future__ import annotations

import ctypes
import itertools
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, NamedTuple

import torch

from b12x.comm.pcie import pcie_dma

_DTYPE_CODES = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}
_MIXED_MODES = {
    "bf16_bf16_f32": (torch.bfloat16, torch.float32),
    "bf16_f32_f32": (torch.float32, torch.float32),
    "bf16_f32_bf16": (torch.float32, torch.bfloat16),
}
# Shadow-memory granularity of the conflict check and the cap on how many
# accesses one page keeps: a conflicting pair is always separated by a few
# ring steps, so an older access can be dropped without losing a report.
_PAGE_BYTES = 1 << 20
_PAGE_HISTORY = 256

_TLS = threading.local()


def _view(ptr: int, elems: int, dtype: torch.dtype) -> torch.Tensor:
    nbytes = elems * torch.empty((), dtype=dtype).element_size()
    raw = (ctypes.c_uint8 * nbytes).from_address(int(ptr))
    return torch.frombuffer(raw, dtype=torch.uint8).view(dtype)


def _submit(fn: Callable[..., None], *args: Any) -> None:
    """Run ``fn(*args)`` now, or append it to the graph being captured."""
    recording = getattr(_TLS, "recording", None)
    if recording is not None:
        recording.append((fn, args))
        return
    fn(*args)


def _model() -> "ScheduleModel | None":
    return getattr(_TLS, "model", None)


def _current_stream() -> "_FakeStream":
    stream = getattr(_TLS, "current", None)
    if stream is None:
        stream = _FakeStream(name="main")
        _TLS.current = stream
    return stream


# ---------------------------------------------------------------------------
# Happens-before model
# ---------------------------------------------------------------------------


class Conflict(NamedTuple):
    """Two accesses to overlapping bytes that the schedule does not order."""

    rank: int
    kind: str  # write-after-read, write-after-write or read-after-write
    first: str
    second: str
    start: int
    nbytes: int

    def __str__(self) -> str:  # pragma: no cover - diagnostic text
        return (
            f"rank {self.rank} {self.kind}: {self.second} conflicts with "
            f"{self.first} on {self.nbytes} bytes at 0x{self.start:x}"
        )


class _Access(NamedTuple):
    start: int
    end: int
    clock: dict[int, int]
    stream: int
    seq: int
    write: bool
    label: str
    rank: int


class ScheduleModel:
    """Vector-clock model of the ring's streams, events and flag protocol."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._view: dict[int, dict[int, int]] = {}
        self._published: dict[tuple[int, int], dict[int, int]] = {}
        self._pages: dict[int, list[_Access]] = {}
        self.conflicts: list[Conflict] = []

    # -- clocks -----------------------------------------------------------
    def _bump(self, stream: "_FakeStream") -> dict[int, int]:
        view = self._view.setdefault(id(stream), {})
        view[id(stream)] = view.get(id(stream), 0) + 1
        return dict(view)

    def _merge(self, stream: "_FakeStream", clock: dict[int, int]) -> None:
        view = self._view.setdefault(id(stream), {})
        for key, value in clock.items():
            if value > view.get(key, 0):
                view[key] = value

    def snapshot(self, stream: "_FakeStream") -> dict[int, int]:
        with self._lock:
            return dict(self._view.setdefault(id(stream), {}))

    def record_event(self, event: "_FakeEvent", stream: "_FakeStream") -> None:
        with self._lock:
            event.clock = dict(self._view.setdefault(id(stream), {}))

    def wait_event(self, stream: "_FakeStream", event: "_FakeEvent") -> None:
        with self._lock:
            self._merge(stream, event.clock)

    def wait_stream(self, stream: "_FakeStream", other: "_FakeStream") -> None:
        with self._lock:
            self._merge(stream, self._view.setdefault(id(other), {}))

    # -- cross-rank flag edges -------------------------------------------
    def publish(self, stream: "_FakeStream", flag_ptr: int, value: int) -> None:
        with self._lock:
            self._bump(stream)
            self._published[(flag_ptr, value)] = dict(self._view[id(stream)])

    def observe(self, stream: "_FakeStream", flag_ptr: int, expected: int) -> None:
        """Order ``stream`` after the publisher of counter value ``expected``.

        A waiter that returns has seen a flag value of at least ``expected``;
        the guarantee the protocol gives is the publisher's state at the point
        it published ``expected``, so only that edge is added.
        """
        with self._lock:
            self._bump(stream)
            clock = self._published.get((flag_ptr, expected))
            if clock is not None:
                self._merge(stream, clock)

    # -- shadow memory ----------------------------------------------------
    def access(
        self,
        stream: "_FakeStream",
        ptr: int,
        nbytes: int,
        *,
        write: bool,
        label: str,
    ) -> None:
        if nbytes <= 0:
            return
        with self._lock:
            clock = self._bump(stream)
            entry = _Access(
                start=int(ptr),
                end=int(ptr) + int(nbytes),
                clock=clock,
                stream=id(stream),
                seq=clock[id(stream)],
                write=bool(write),
                label=label,
                rank=stream.rank,
            )
            pages = range(entry.start // _PAGE_BYTES, (entry.end - 1) // _PAGE_BYTES + 1)
            seen: set[tuple[int, int]] = set()
            for page in pages:
                history = self._pages.setdefault(page, [])
                for old in history:
                    if (old.stream, old.seq) in seen:
                        continue
                    if old.end <= entry.start or entry.end <= old.start:
                        continue
                    if not old.write and not entry.write:
                        continue
                    if clock.get(old.stream, 0) >= old.seq:
                        continue
                    seen.add((old.stream, old.seq))
                    kind = (
                        "write-after-write"
                        if old.write and entry.write
                        else "write-after-read"
                        if entry.write
                        else "read-after-write"
                    )
                    self.conflicts.append(
                        Conflict(
                            rank=entry.rank,
                            kind=kind,
                            first=old.label,
                            second=entry.label,
                            start=max(old.start, entry.start),
                            nbytes=min(old.end, entry.end) - max(old.start, entry.start),
                        )
                    )
                history.append(entry)
                if len(history) > _PAGE_HISTORY:
                    del history[: len(history) - _PAGE_HISTORY]


# ---------------------------------------------------------------------------
# torch.cuda fakes
# ---------------------------------------------------------------------------


_STREAM_IDS = itertools.count()


class _FakeStream:
    def __init__(self, device=None, *, rank: int | None = None, name: str = "") -> None:
        self.device = device
        self.rank = _tls_rank() if rank is None else int(rank)
        self.name = name or f"stream{next(_STREAM_IDS)}"

    def __repr__(self) -> str:  # pragma: no cover - diagnostic text
        return f"<rank {self.rank} {self.name}>"

    def wait_event(self, event) -> None:
        model = _model()
        if model is not None:
            _submit(model.wait_event, self, event)

    def wait_stream(self, stream) -> None:
        model = _model()
        if model is not None:
            _submit(model.wait_stream, self, stream)

    def synchronize(self) -> None:
        return None


class _FakeEvent:
    def __init__(self, enable_timing: bool = False) -> None:
        self.clock: dict[int, int] = {}

    def record(self, stream=None) -> None:
        model = _model()
        if model is None:
            return
        target = stream if stream is not None else _current_stream()
        if target is not None:
            _submit(model.record_event, self, target)

    def wait(self, stream=None) -> None:
        model = _model()
        if model is None:
            return
        target = stream if stream is not None else _current_stream()
        if target is not None:
            _submit(model.wait_event, target, self)


def _tls_rank() -> int:
    return int(getattr(_TLS, "rank", 0))


class _FakeGraph:
    """Records kernel and ordering calls during capture; ``replay`` re-issues
    them, so a replayed op drives the same addresses and the same
    stream/event structure as the captured one.

    A graph launch is one operation of the launching stream: every node runs
    after the launch point and the launching stream continues only once the
    whole graph is done. ``replay`` brackets the recorded calls with those two
    edges, which is also what keeps consecutive replays ordered against each
    other."""

    def __init__(self) -> None:
        self.calls: list[tuple[Callable[..., None], tuple[Any, ...]]] = []
        self.streams: list[_FakeStream] = []

    def capture_begin(self, capture_error_mode: str = "global") -> None:
        kernels = EmulatedKernels.current()
        if kernels is None:
            raise RuntimeError("capture outside an emulated ring thread")
        kernels.begin_capture(self.calls)

    def capture_end(self) -> None:
        kernels = EmulatedKernels.current()
        assert kernels is not None
        kernels.end_capture()
        seen: dict[int, _FakeStream] = {}
        for _fn, args in self.calls:
            for arg in args:
                if isinstance(arg, _FakeStream):
                    seen.setdefault(id(arg), arg)
        self.streams = list(seen.values())

    def replay(self) -> None:
        model = _model()
        launch = _current_stream()
        if model is not None:
            for stream in self.streams:
                model.wait_stream(stream, launch)
        for fn, args in self.calls:
            fn(*args)
        if model is not None:
            for stream in self.streams:
                model.wait_stream(launch, stream)


class _FakeDeviceContext:
    def __init__(self, device) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


@contextmanager
def _fake_stream_context(stream):
    previous = getattr(_TLS, "current", None)
    _TLS.current = stream
    try:
        yield
    finally:
        _TLS.current = previous


def install_cuda_fakes(monkeypatch) -> None:
    """Replace the torch.cuda entry points the ring uses with the fakes."""
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "device", _FakeDeviceContext)
    monkeypatch.setattr(torch.cuda, "stream", _fake_stream_context)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: _current_stream()
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: getattr(_TLS, "recording", None) is not None,
    )


class EmulatedKernels:
    """Host implementation of the ``DmaKernels`` facade shared by all ranks."""

    _local = threading.local()

    def __init__(self, model: ScheduleModel | None = None) -> None:
        self._cond = threading.Condition()
        self.model = model
        self.copies = 0
        self.flags_set = 0
        self.flags_waited = 0
        self.adds = 0

    @classmethod
    def current(cls) -> "EmulatedKernels | None":
        return getattr(_TLS, "kernels", None)

    def bind_thread(self, rank: int = 0, current: _FakeStream | None = None) -> None:
        _TLS.kernels = self
        _TLS.recording = None
        _TLS.rank = int(rank)
        _TLS.model = self.model
        _TLS.current = current

    def begin_capture(self, calls: list) -> None:
        _TLS.recording = calls

    def end_capture(self) -> None:
        _TLS.recording = None

    def _issue(self, fn: Callable[..., None], *args: Any) -> None:
        _submit(fn, _current_stream(), *args)

    def _access(
        self, stream, ptr: int, nbytes: int, *, write: bool, label: str
    ) -> None:
        if self.model is not None and stream is not None:
            self.model.access(stream, ptr, nbytes, write=write, label=label)

    def host_write(self, ptr: int, nbytes: int, label: str) -> None:
        """Declare a main-stream write the ring performs with plain tensor
        operations (the reduce-scatter's column pack) rather than a kernel."""
        self._access(_current_stream(), ptr, nbytes, write=True, label=label)

    # -- primitives -------------------------------------------------------
    def dma_copy(self, dst_ptr: int, src_ptr: int, bytes_: int) -> None:
        self._issue(self._copy, int(dst_ptr), int(src_ptr), int(bytes_))

    def _copy(self, stream, dst: int, src: int, nbytes: int) -> None:
        self._access(stream, src, nbytes, write=False, label="copy source")
        self._access(stream, dst, nbytes, write=True, label="copy destination")
        ctypes.memmove(dst, src, nbytes)
        self.copies += 1

    def dma_set_flag(self, peer_flag_ptr: int, counter_ptr: int) -> None:
        self._issue(self._set_flag, int(peer_flag_ptr), int(counter_ptr))

    def _set_flag(self, stream, peer_flag_ptr: int, counter_ptr: int) -> None:
        with self._cond:
            counter = ctypes.c_int32.from_address(counter_ptr)
            counter.value = counter.value + 1
            if self.model is not None and stream is not None:
                self.model.publish(stream, peer_flag_ptr, counter.value)
            ctypes.c_int32.from_address(peer_flag_ptr).value = counter.value
            self.flags_set += 1
            self._cond.notify_all()

    def dma_wait_flag(self, flag_ptr: int, counter_ptr: int) -> None:
        self._issue(self._wait_flag, int(flag_ptr), int(counter_ptr))

    def _wait_flag(self, stream, flag_ptr: int, counter_ptr: int) -> None:
        with self._cond:
            counter = ctypes.c_int32.from_address(counter_ptr)
            counter.value = counter.value + 1
            expected = counter.value
            flag = ctypes.c_int32.from_address(flag_ptr)
            if not self._cond.wait_for(lambda: flag.value - expected >= 0, timeout=60):
                raise TimeoutError("emulated ring flag wait timed out")
            if self.model is not None and stream is not None:
                self.model.observe(stream, flag_ptr, expected)
            self.flags_waited += 1

    def dma_add(
        self, dst_ptr: int, a_ptr: int, b_ptr: int, elems: int, dtype_code: int
    ) -> None:
        self._issue(
            self._add, int(dst_ptr), int(a_ptr), int(b_ptr), int(elems), int(dtype_code)
        )

    def _add(self, stream, dst: int, a: int, b: int, elems: int, dtype_code: int) -> None:
        dtype = _DTYPE_CODES[dtype_code]
        width = torch.empty((), dtype=dtype).element_size()
        self._access(stream, a, elems * width, write=False, label="add operand")
        self._access(stream, b, elems * width, write=False, label="add payload")
        self._access(stream, dst, elems * width, write=True, label="add result")
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

    def _add_mixed(self, stream, dst: int, a: int, b: int, elems: int, mode: str) -> None:
        b_dtype, out_dtype = _MIXED_MODES[mode]
        b_width = torch.empty((), dtype=b_dtype).element_size()
        out_width = torch.empty((), dtype=out_dtype).element_size()
        self._access(stream, a, elems * 2, write=False, label="add operand")
        self._access(stream, b, elems * b_width, write=False, label="add payload")
        self._access(stream, dst, elems * out_width, write=True, label="add result")
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


def _instrumented_pack(ring, inp: torch.Tensor, blocks: torch.Tensor) -> None:
    """``_pack_column_blocks`` with its main-stream write declared to the
    schedule model, so cross-call reuse of the static block buffer is checked
    like any kernel write."""
    ring._kernels.host_write(
        int(blocks.data_ptr()),
        blocks.numel() * blocks.element_size(),
        "column pack",
    )
    pcie_dma.PCIeDmaAllReduce._pack_column_blocks(ring, inp, blocks)


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
        model: bool = False,
    ) -> None:
        self.world = world
        self.model = ScheduleModel() if model else None
        self.kernels = EmulatedKernels(self.model)
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
        self.main_streams: list[_FakeStream] = []
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
            ring._copy_stream = _FakeStream(device, rank=rank, name="copy")
            ring._flag_stream = _FakeStream(device, rank=rank, name="flag")
            ring._ag_copy_stream = _FakeStream(device, rank=rank, name="ag-copy")
            ring._ag_flag_stream = _FakeStream(device, rank=rank, name="ag-flag")
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
            if self.model is not None:
                ring._pack_column_blocks = _instrumented_pack.__get__(ring, cls)
            self.main_streams.append(_FakeStream(device, rank=rank, name="main"))
            self.rings.append(ring)

    @property
    def conflicts(self) -> list[Conflict]:
        """Overlapping accesses the schedule leaves unordered (empty unless
        the ring was built with ``model=True``)."""
        return [] if self.model is None else list(self.model.conflicts)

    def run(self, fn: Callable[[pcie_dma.PCIeDmaAllReduce, int], Any]) -> list[Any]:
        """Run ``fn(ring, rank)`` on every rank concurrently; return results."""
        results: list[Any] = [None] * self.world
        errors: list[BaseException | None] = [None] * self.world

        def worker(rank: int) -> None:
            self.kernels.bind_thread(rank, self.main_streams[rank])
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
