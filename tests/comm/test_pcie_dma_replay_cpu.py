"""CPU-emulated ring tests: graph-replay static buffers of
``pcie_dma.PCIeDmaAllReduce``.

The emulation (``pcie_dma_emulation.py``) runs the ring's own Python
schedule per rank over host memory with the device flag protocol and the
device add arithmetic, so these tests check protocol correctness, replay
bookkeeping and numerics; multi-stream timing needs the GPU tests.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from b12x.comm.pcie import pcie_dma
from tests.comm.pcie_dma_emulation import (
    EmulatedRing,
    install_cuda_fakes,
    ring_all_reduce_reference,
)

WORLD = 9
HIDDEN = 7168
LATENT = 3584
ROUTER_COLS = 104
LATENT_COLS = 400
MAX_BYTES = 4608 * HIDDEN * 2


@pytest.fixture(autouse=True)
def _cuda_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("B12X_PCIE_DMA_PIECES", raising=False)
    install_cuda_fakes(monkeypatch)


def _heavy_tailed(
    rows: int, cols: int, rank: int, seed: int, dtype: torch.dtype
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 1000 + rank)
    bulk = torch.randn(rows, cols, generator=generator) * 2.0 ** (rank - 4)
    outliers = torch.rand(rows, cols, generator=generator) < 0.01
    spikes = torch.randn(rows, cols, generator=generator) * 1.0e4
    return torch.where(outliers, spikes, bulk).to(dtype)


def _inputs(rows: int, cols: int, seed: int, dtype: torch.dtype) -> list[torch.Tensor]:
    return [_heavy_tailed(rows, cols, rank, seed, dtype) for rank in range(WORLD)]


def _same_on_all_ranks(outputs: list[torch.Tensor]) -> bool:
    return all(torch.equal(out, outputs[0]) for out in outputs[1:])


# ---------------------------------------------------------------------------
# Harness self-checks
# ---------------------------------------------------------------------------


def test_emulated_ring_mirrors_constructor_state() -> None:
    """Every attribute ``PCIeDmaAllReduce.__init__`` assigns exists on the
    emulated object, so the ring's methods see the state they expect."""
    source = textwrap.dedent(inspect.getsource(pcie_dma.PCIeDmaAllReduce.__init__))
    assigned: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    assert assigned, "constructor attribute scan found nothing"
    ring = EmulatedRing(WORLD, MAX_BYTES).rings[0]
    missing = sorted(name for name in assigned if not hasattr(ring, name))
    assert not missing, f"emulation lacks constructor attributes: {missing}"


def test_flag_slot_range_covers_every_world_size() -> None:
    assert pcie_dma.AR_SLOT_BASE + pcie_dma.AR_SLOT_COUNT <= pcie_dma.FLAG_SLOTS
    for world in pcie_dma.SUPPORTED_WORLD_SIZES:
        # All-reduce: 2(world-1) steps x MAX_PIECES pieces + done.
        assert 2 * (world - 1) * pcie_dma.MAX_PIECES < pcie_dma.AR_SLOT_COUNT


# ---------------------------------------------------------------------------
# Item 7: replay over static buffers without staging copies
# ---------------------------------------------------------------------------


def test_eager_all_reduce_matches_ring_arithmetic() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False)
    inputs = _inputs(144, HIDDEN, seed=1, dtype=torch.bfloat16)
    outs = emu.run(lambda ring, rank: ring.all_reduce(inputs[rank]))
    expected = ring_all_reduce_reference(inputs, WORLD)
    assert _same_on_all_ranks(outs)
    assert torch.equal(outs[0], expected)


def test_eager_all_reduce_wire_padding_path() -> None:
    """Nine ranks with numel not divisible by 72 go through the padded wire."""
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False)
    inputs = _inputs(8, HIDDEN, seed=2, dtype=torch.bfloat16)
    assert inputs[0].numel() % 8 == 0 and inputs[0].numel() % 72 != 0
    outs = emu.run(lambda ring, rank: ring.all_reduce(inputs[rank]))
    padded = [torch.nn.functional.pad(x.reshape(-1), (0, 72 - x.numel() % 72)) for x in inputs]
    expected = ring_all_reduce_reference(padded, WORLD)[: inputs[0].numel()]
    assert _same_on_all_ranks(outs)
    assert torch.equal(outs[0].reshape(-1), expected)


def test_replay_captures_on_second_sighting_in_place_and_matches_eager() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES)
    inputs = _inputs(144, HIDDEN, seed=3, dtype=torch.bfloat16)
    expected = ring_all_reduce_reference(inputs, WORLD)

    def two_calls(ring, rank):
        first = ring.all_reduce(inputs[rank])
        captured_after_first = len(ring._replay_entries)
        second = ring.all_reduce(inputs[rank])
        return first, captured_after_first, second

    results = emu.run(two_calls)
    for first, captured_after_first, second in results:
        assert captured_after_first == 0
        assert torch.equal(first, expected)
        assert torch.equal(second, expected)
    ring = emu.rings[0]
    key = ("ar", inputs[0].numel(), torch.bfloat16)
    assert list(ring._replay_entries) == [key]
    entry = ring._replay_entries[key]
    assert entry.inp is entry.out, "lossless all-reduce entry is in place"
    assert ring.is_ring_storage(entry.inp)
    # A plain call still returns caller-owned storage.
    assert not ring.is_ring_storage(results[0][2])


def test_all_reduce_input_and_borrowed_output_share_the_static_buffer() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES)
    shape = (144, HIDDEN)
    inputs = _inputs(*shape, seed=4, dtype=torch.bfloat16)
    expected = ring_all_reduce_reference(inputs, WORLD)

    def chain(ring, rank):
        # First sighting: no entry yet, producer keeps its own buffer.
        assert ring.all_reduce_input(shape, torch.bfloat16) is None
        # Second sighting captures; the producer writes the static input.
        static = ring.all_reduce_input(shape, torch.bfloat16)
        assert static is not None and ring.is_ring_storage(static)
        static.copy_(inputs[rank])
        out = ring.all_reduce(static, borrow_output=True)
        assert out.data_ptr() == static.data_ptr()
        # The consumer may write the borrowed buffer; the next producer
        # writes it again and the next all-reduce takes it in place.
        out.copy_(inputs[rank])
        again = ring.all_reduce(out, borrow_output=True)
        assert again.data_ptr() == static.data_ptr()
        return out.clone(), again.clone(), ring.all_reduce_input(shape, torch.bfloat16)

    results = emu.run(chain)
    for out, again, static in results:
        assert torch.equal(out, expected)
        assert torch.equal(again, expected)
        assert static is not None
    ring = emu.rings[0]
    assert len(ring._replay_entries) == 1
    with pytest.raises(ValueError, match="borrow_output"):
        ring.all_reduce(inputs[0], out=torch.empty_like(inputs[0]), borrow_output=True)


def test_replay_rejects_partially_overlapping_static_input() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES)
    shape = (144, HIDDEN)
    inputs = _inputs(*shape, seed=5, dtype=torch.bfloat16)

    def capture(ring, rank):
        ring.all_reduce(inputs[rank])
        return ring.all_reduce(inputs[rank], borrow_output=True)

    borrowed = emu.run(capture)
    ring = emu.rings[0]
    nbytes = borrowed[0].numel() * 2
    arena = ring._replay_arena
    assert arena is not None
    entry = next(iter(ring._replay_entries.values()))
    offset = entry.inp.data_ptr() - arena.data_ptr() + 256
    shifted = arena[offset : offset + nbytes].view(torch.bfloat16).view(shape)
    assert ring.is_ring_storage(shifted)
    with pytest.raises(ValueError, match="partially overlaps"):
        ring._all_reduce_replayed(shifted, None, True)


def test_repeated_same_shape_calls_keep_one_entry_and_counters_advance() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES)
    inputs = _inputs(144, HIDDEN, seed=6, dtype=torch.bfloat16)
    expected = ring_all_reduce_reference(inputs, WORLD)
    calls = 6

    def repeat(ring, rank):
        outs = []
        seqs = []
        for _ in range(calls):
            outs.append(ring.all_reduce(inputs[rank], borrow_output=True).clone())
            seqs.append(ring._op_seq)
        return outs, seqs

    results = emu.run(repeat)
    for outs, seqs in results:
        assert all(torch.equal(out, expected) for out in outs)
        assert seqs == list(range(1, calls + 1))
    ring = emu.rings[0]
    assert len(ring._replay_entries) == 1
    entry = next(iter(ring._replay_entries.values()))
    assert entry.last_use == calls
    # Every slot the op uses advanced exactly once per call on every rank,
    # for the publisher and the waiter alike, and the published flag values
    # equal the counters (the graph runs the same flag kernels as eager).
    for rank, r in enumerate(emu.rings):
        used = r._send_counters.nonzero().flatten().tolist()
        assert used, "all-reduce published no flags"
        assert all(int(r._send_counters[slot]) == calls for slot in used)
        assert torch.equal(r._send_counters, r._wait_counters)
        for slot in used:
            peer_slot_ptr = r._flag_ptr(rank, slot)
            offset = peer_slot_ptr - r._flags_base[rank]
            flag = emu.slabs[rank][offset : offset + 4].view(torch.int32)
            assert int(flag) == calls
        assert max(used) < pcie_dma.AR_SLOT_BASE + pcie_dma.AR_SLOT_COUNT


def test_eviction_guard_keeps_a_borrowed_output_until_its_consumer_reads() -> None:
    """With one arena slot, a second shape cannot evict the entry whose
    output was borrowed within the last REPLAY_EVICTION_GUARD_OPS ops."""
    emu = EmulatedRing(WORLD, MAX_BYTES, replay_max_entries=1)
    shape_a = (144, HIDDEN)
    shape_b = (216, HIDDEN)
    inputs_a = _inputs(*shape_a, seed=7, dtype=torch.bfloat16)
    inputs_b = _inputs(*shape_b, seed=8, dtype=torch.bfloat16)
    expected_a = ring_all_reduce_reference(inputs_a, WORLD)
    expected_b = ring_all_reduce_reference(inputs_b, WORLD)
    key_a = ("ar", inputs_a[0].numel(), torch.bfloat16)
    key_b = ("ar", inputs_b[0].numel(), torch.bfloat16)
    guard = pcie_dma.REPLAY_EVICTION_GUARD_OPS

    def scenario(ring, rank):
        ring.all_reduce(inputs_a[rank])  # op 1: eager sighting
        borrowed = ring.all_reduce(inputs_a[rank], borrow_output=True)  # op 2
        assert ring.is_ring_storage(borrowed)
        snapshots = []
        # Ops 3 .. 2+guard+1: B is sighted repeatedly; each attempt to
        # capture B finds A used within the guard and runs eagerly.
        for _ in range(guard + 1):
            out_b = ring.all_reduce(inputs_b[rank])
            snapshots.append((list(ring._replay_entries), borrowed.clone(), out_b))
        # One op later A is outside the guard: B captures and takes the slot.
        out_b_last = ring.all_reduce(inputs_b[rank])
        return snapshots, list(ring._replay_entries), out_b_last

    results = emu.run(scenario)
    for snapshots, final_keys, out_b_last in results:
        for keys, borrowed_view, out_b in snapshots:
            assert keys == [key_a]
            assert torch.equal(borrowed_view, expected_a)
            assert torch.equal(out_b, expected_b)
        assert final_keys == [key_b]
        assert torch.equal(out_b_last, expected_b)


def test_replay_keys_carry_the_op_tag() -> None:
    latent = torch.empty(144, LATENT, dtype=torch.bfloat16)
    ring = pcie_dma.PCIeDmaAllReduce
    assert ring._all_reduce_key(latent) == ("ar", latent.numel(), latent.dtype)
    assert ring._all_reduce_key(latent.view(-1)) == ring._all_reduce_key(latent)
