"""Replay-cache bookkeeping of PCIeDmaAllReduce on one GPU, without a ring.

The ring's collective needs several GPUs; these tests drive the replay
entry points of a bare instance whose eager collective is stubbed, so they
check only the cache contract: the output contract holds on the replay
path, an evicted shape earns its capture again with one eager call, and a
capture that fails returns its static-buffer slot.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from b12x.comm.pcie.pcie_dma import SCRATCH_ALIGN, PCIeDmaAllReduce, _align_up

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs one CUDA device"
)


def _bare_ring(*, max_entries: int, max_bytes: int = 1 << 16) -> PCIeDmaAllReduce:
    ring = PCIeDmaAllReduce.__new__(PCIeDmaAllReduce)
    ring.device = torch.device("cuda", torch.cuda.current_device())
    ring.world_size = 2
    ring.rank = 0
    ring.max_bytes = max_bytes
    ring.min_bytes = 0
    ring._graph_replay = True
    ring._graph_replay_min_bytes = 0
    ring._graph_replay_max_entries = max_entries
    ring._replay_entries = OrderedDict()
    ring._replay_seen = {}
    ring._replay_capture_stream = None
    ring._replay_slot_bytes = 2 * _align_up(max_bytes, SCRATCH_ALIGN)
    ring._replay_arena = torch.empty(
        max_entries * ring._replay_slot_bytes, dtype=torch.uint8, device=ring.device
    )
    ring._replay_free_slots = list(range(max_entries))
    ring.should_allreduce = lambda inp: True  # type: ignore[method-assign]
    return ring


def _stub_eager(ring: PCIeDmaAllReduce, calls: list) -> None:
    """Record (numel, capturing) per eager call; the collective is a copy."""

    def eager(inp, *, out=None):
        calls.append((inp.numel(), torch.cuda.is_current_stream_capturing()))
        if out is None:
            out = torch.empty_like(inp)
        out.copy_(inp)
        return out

    ring._all_reduce_on_device = eager  # type: ignore[method-assign]


def test_replay_path_enforces_the_output_contract() -> None:
    ring = _bare_ring(max_entries=2)
    calls: list = []
    _stub_eager(ring, calls)
    inp = torch.ones(16, 64, dtype=torch.bfloat16, device=ring.device)
    wrong_dtype = torch.empty(16, 64, dtype=torch.float16, device=ring.device)
    cpu_out = torch.empty(16, 64, dtype=torch.bfloat16)
    strided = torch.empty(16, 128, dtype=torch.bfloat16, device=ring.device)[:, :64]
    for bad in (wrong_dtype, cpu_out, strided):
        with pytest.raises(ValueError, match="output must match"):
            ring._all_reduce_replayed(inp, bad)
    assert not calls and not ring._replay_entries


def test_evicted_shape_runs_eager_once_before_it_is_captured_again() -> None:
    ring = _bare_ring(max_entries=1)
    calls: list = []
    _stub_eager(ring, calls)
    a = torch.full((8, 64), 3.0, dtype=torch.bfloat16, device=ring.device)
    b = torch.full((8, 128), 5.0, dtype=torch.bfloat16, device=ring.device)
    eager, captured = (a.numel(), False), (a.numel(), True)
    # a: eager, then captured. b: eager, then captured (evicts a). a again:
    # eager first (its warm-up state was dropped with the eviction), then
    # captured; the capture after that is a replay with no eager call.
    ring._all_reduce_replayed(a, None)
    assert calls == [eager]
    out = ring._all_reduce_replayed(a, None)
    assert calls == [eager, captured]
    assert torch.equal(out, a)
    ring._all_reduce_replayed(b, None)
    ring._all_reduce_replayed(b, None)
    assert calls[2:] == [(b.numel(), False), (b.numel(), True)]
    assert list(ring._replay_entries) == [(b.numel(), b.dtype)]
    assert (a.numel(), a.dtype) not in ring._replay_seen
    ring._all_reduce_replayed(a, None)
    assert calls[4:] == [eager]
    out = ring._all_reduce_replayed(a, None)
    assert calls[5:] == [captured]
    assert torch.equal(out, a)
    out = ring._all_reduce_replayed(a, None)
    assert calls[6:] == []
    assert torch.equal(out, a)
    assert list(ring._replay_entries) == [(a.numel(), a.dtype)]
    assert ring._replay_free_slots == []


def test_failed_capture_returns_its_slot() -> None:
    ring = _bare_ring(max_entries=1)

    def failing_eager(inp, *, out=None):
        raise RuntimeError("capture-time failure")

    ring._all_reduce_on_device = failing_eager  # type: ignore[method-assign]
    inp = torch.ones(8, 64, dtype=torch.bfloat16, device=ring.device)
    with pytest.raises(RuntimeError, match="capture-time failure"):
        ring._capture_replay_entry(inp)
    assert ring._replay_free_slots == [0]
    assert not ring._replay_entries
    # The slot is usable again: a capture that completes owns it.
    calls: list = []
    _stub_eager(ring, calls)
    entry = ring._capture_replay_entry(inp)
    assert entry.slot == 0 and ring._replay_free_slots == []
    assert list(ring._replay_entries) == [(inp.numel(), inp.dtype)]
