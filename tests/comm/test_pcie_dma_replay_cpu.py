"""CPU-emulated ring tests: graph-replay static buffers, the paired
all-gather and the column reduce-scatter of ``pcie_dma.PCIeDmaAllReduce``.

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
    column_reduce_scatter_reference,
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


def test_flag_slot_ranges_are_disjoint_for_every_world_size() -> None:
    ar_end = pcie_dma.AR_SLOT_BASE + pcie_dma.AR_SLOT_COUNT
    ag_end = pcie_dma.AG_PAIR_SLOT_BASE + pcie_dma.AG_PAIR_SLOT_COUNT
    rs_end = pcie_dma.RS_SLOT_BASE + pcie_dma.RS_SLOT_COUNT
    assert ar_end <= pcie_dma.AG_PAIR_SLOT_BASE < ag_end <= pcie_dma.RS_SLOT_BASE
    assert rs_end <= pcie_dma.FLAG_SLOTS
    for world in pcie_dma.SUPPORTED_WORLD_SIZES:
        # All-reduce: 2(world-1) steps x MAX_PIECES pieces + done.
        assert 2 * (world - 1) * pcie_dma.MAX_PIECES < pcie_dma.AR_SLOT_COUNT
        # Paired all-gather: world-1 steps + done.
        assert world <= pcie_dma.AG_PAIR_SLOT_COUNT
        # Column reduce-scatter: (world-1) x RS_MAX_PIECES + done.
        assert (world - 1) * pcie_dma.RS_MAX_PIECES < pcie_dma.RS_SLOT_COUNT


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
        assert max(used) < pcie_dma.AG_PAIR_SLOT_BASE


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
    rows = 144
    latent = torch.empty(rows, LATENT, dtype=torch.bfloat16)
    first = torch.empty(rows, ROUTER_COLS, dtype=torch.float32)
    second = torch.empty(rows, LATENT_COLS, dtype=torch.bfloat16)
    ring = pcie_dma.PCIeDmaAllReduce
    assert ring._all_reduce_key(latent)[0] == "ar"
    assert ring._reduce_scatter_key(latent, "fp32", 399)[0] == "rs_fp32"
    assert ring._reduce_scatter_key(latent, "bf16", 399)[0] == "rs_bf16"
    assert ring._all_gather_pair_key(first, second)[0] == "ag_pair"
    assert ring._all_reduce_key(latent)[1:] == (latent.numel(), latent.dtype)
    assert ring._reduce_scatter_key(latent, "fp32", 399)[1:3] == (
        latent.numel(),
        latent.dtype,
    )


# ---------------------------------------------------------------------------
# Item 1: paired all-gather
# ---------------------------------------------------------------------------


def _pair_inputs(rows: int, seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    firsts = _inputs(rows, ROUTER_COLS, seed, torch.float32)
    seconds = _inputs(rows, LATENT_COLS, seed + 50, torch.bfloat16)
    return firsts, seconds


def _check_gathered(
    outs: list[tuple[torch.Tensor, torch.Tensor]],
    firsts: list[torch.Tensor],
    seconds: list[torch.Tensor],
) -> None:
    expected_first = torch.stack(firsts)
    expected_second = torch.stack(seconds)
    for out_first, out_second in outs:
        assert out_first.shape == expected_first.shape
        assert out_second.shape == expected_second.shape
        # Byte equality, as torch.distributed.all_gather would produce.
        assert torch.equal(out_first.view(torch.int32), expected_first.view(torch.int32))
        assert torch.equal(out_second.view(torch.int16), expected_second.view(torch.int16))


def test_all_gather_pair_eager_matches_all_gather_semantics() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False)
    firsts, seconds = _pair_inputs(144, seed=10)
    outs = emu.run(lambda ring, rank: ring.all_gather_pair(firsts[rank], seconds[rank]))
    _check_gathered(outs, firsts, seconds)
    for rank, ring in enumerate(emu.rings):
        used = ring._send_counters.nonzero().flatten().tolist()
        assert used == list(range(pcie_dma.AG_PAIR_SLOT_BASE, pcie_dma.AG_PAIR_SLOT_BASE + WORLD))


def test_all_gather_pair_replay_returns_static_outputs_that_track_new_inputs() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES)
    firsts_a, seconds_a = _pair_inputs(144, seed=11)
    firsts_b, seconds_b = _pair_inputs(144, seed=12)

    def three_calls(ring, rank):
        eager = ring.all_gather_pair(firsts_a[rank], seconds_a[rank])
        replayed = ring.all_gather_pair(firsts_a[rank], seconds_a[rank])
        assert ring.is_ring_storage(replayed[0]) and ring.is_ring_storage(replayed[1])
        assert not ring.is_ring_storage(eager[0])
        replayed_clone = (replayed[0].clone(), replayed[1].clone())
        again = ring.all_gather_pair(firsts_b[rank], seconds_b[rank])
        assert again[0].data_ptr() == replayed[0].data_ptr()
        assert again[1].data_ptr() == replayed[1].data_ptr()
        return eager, replayed_clone, (again[0].clone(), again[1].clone())

    results = emu.run(three_calls)
    _check_gathered([r[0] for r in results], firsts_a, seconds_a)
    _check_gathered([r[1] for r in results], firsts_a, seconds_a)
    _check_gathered([r[2] for r in results], firsts_b, seconds_b)
    ring = emu.rings[0]
    assert list(ring._replay_entries) == [
        ring._all_gather_pair_key(firsts_a[0], seconds_a[0])
    ]


def test_all_gather_pair_static_inputs_skip_staging() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES)
    firsts, seconds = _pair_inputs(144, seed=13)

    def chain(ring, rank):
        assert (
            ring.all_gather_pair_inputs(
                tuple(firsts[rank].shape), torch.float32, tuple(seconds[rank].shape), torch.bfloat16
            )
            is None
        )
        statics = ring.all_gather_pair_inputs(
            tuple(firsts[rank].shape), torch.float32, tuple(seconds[rank].shape), torch.bfloat16
        )
        assert statics is not None
        statics[0].copy_(firsts[rank])
        statics[1].copy_(seconds[rank])
        out = ring.all_gather_pair(statics[0], statics[1])
        return out[0].clone(), out[1].clone()

    _check_gathered(emu.run(chain), firsts, seconds)


def test_all_gather_pair_rejects_unsupported_inputs() -> None:
    ring = EmulatedRing(WORLD, MAX_BYTES).rings[0]
    first = torch.empty(144, ROUTER_COLS, dtype=torch.float32)
    second = torch.empty(144, LATENT_COLS, dtype=torch.bfloat16)
    assert ring.should_all_gather_pair(first, second)
    assert not ring.should_all_gather_pair(first, torch.empty(72, LATENT_COLS, dtype=torch.bfloat16))
    assert not ring.should_all_gather_pair(first.view(-1), second)
    assert not ring.should_all_gather_pair(first.t(), second)
    huge = torch.empty(144, ring.shard_capacity // 2, dtype=torch.bfloat16)
    assert not ring.should_all_gather_pair(first, huge)
    with pytest.raises(ValueError, match="paired all-gather"):
        ring.all_gather_pair(first, huge)


# ---------------------------------------------------------------------------
# Item 3: column reduce-scatter
# ---------------------------------------------------------------------------


def _rs_mismatch_vs_fp64(blocks: list[torch.Tensor], inputs: list[torch.Tensor]) -> int:
    rows, width = inputs[0].shape
    cols = (width + WORLD - 1) // WORLD
    reference = sum(x.double() for x in inputs).to(torch.bfloat16)
    mismatches = 0
    for rank, block in enumerate(blocks):
        start = rank * cols
        valid = min(cols, width - start)
        mismatches += int((block[:, :valid] != reference[:, start : start + valid]).sum())
        assert torch.equal(block[:, valid:], torch.zeros_like(block[:, valid:]))
    return mismatches


@pytest.mark.parametrize("wire", ["fp32", "bf16"])
def test_reduce_scatter_columns_matches_reference(wire: str) -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False)
    inputs = _inputs(144, LATENT, seed=20, dtype=torch.bfloat16)
    blocks = emu.run(lambda ring, rank: ring.reduce_scatter_columns(inputs[rank], wire=wire))
    expected = column_reduce_scatter_reference(inputs, WORLD, wire)
    for rank, block in enumerate(blocks):
        assert block.shape == (144, 399)
        assert torch.equal(block, expected[rank])
    for rank, ring in enumerate(emu.rings):
        used = ring._send_counters.nonzero().flatten().tolist()
        assert min(used) == pcie_dma.RS_SLOT_BASE
        assert max(used) < pcie_dma.RS_SLOT_BASE + pcie_dma.RS_SLOT_COUNT


def test_reduce_scatter_fp32_wire_is_at_least_as_precise_as_bf16_wire() -> None:
    inputs = _inputs(144, LATENT, seed=21, dtype=torch.bfloat16)
    fp32_blocks = column_reduce_scatter_reference(inputs, WORLD, "fp32")
    bf16_blocks = column_reduce_scatter_reference(inputs, WORLD, "bf16")
    ring_blocks = []
    ring_out = ring_all_reduce_reference(inputs, WORLD)
    for rank in range(WORLD):
        start = rank * 399
        block = torch.zeros(144, 399, dtype=torch.bfloat16)
        valid = min(399, LATENT - start)
        block[:, :valid] = ring_out[:, start : start + valid]
        ring_blocks.append(block)
    fp32_bad = _rs_mismatch_vs_fp64(fp32_blocks, inputs)
    bf16_bad = _rs_mismatch_vs_fp64(bf16_blocks, inputs)
    ring_bad = _rs_mismatch_vs_fp64(ring_blocks, inputs)
    # One rounding versus eight: the fp32 wire deviates from the correctly
    # rounded sum on far fewer elements than either eight-rounding scheme.
    assert fp32_bad < bf16_bad
    assert fp32_bad < ring_bad
    assert fp32_bad * 4 < ring_bad


def test_reduce_scatter_replay_matches_eager_and_returns_static_block() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES)
    inputs_a = _inputs(144, LATENT, seed=22, dtype=torch.bfloat16)
    inputs_b = _inputs(144, LATENT, seed=23, dtype=torch.bfloat16)

    def calls(ring, rank):
        eager = ring.reduce_scatter_columns(inputs_a[rank], wire="fp32")
        replayed = ring.reduce_scatter_columns(inputs_a[rank], wire="fp32")
        assert ring.is_ring_storage(replayed) and not ring.is_ring_storage(eager)
        snapshot = replayed.clone()
        again = ring.reduce_scatter_columns(inputs_b[rank], wire="fp32")
        assert again.data_ptr() == replayed.data_ptr()
        return eager, snapshot, again.clone()

    results = emu.run(calls)
    expected_a = column_reduce_scatter_reference(inputs_a, WORLD, "fp32")
    expected_b = column_reduce_scatter_reference(inputs_b, WORLD, "fp32")
    for rank, (eager, replayed, again) in enumerate(results):
        assert torch.equal(eager, expected_a[rank])
        assert torch.equal(replayed, expected_a[rank])
        assert torch.equal(again, expected_b[rank])
    ring = emu.rings[0]
    entry = next(iter(ring._replay_entries.values()))
    assert entry.key[0] == "rs_fp32"
    assert len(entry.scratch) == 2 and all(p.dtype == torch.float32 for p in entry.scratch)


def test_reduce_scatter_two_pieces_matches_reference() -> None:
    """A shard above 1 MB per piece splits into two pieces, each with its own
    scratch area and flag slot."""
    rows = 1328  # 1328 x 399 elements: divisible by 16, > 1 MB per bf16 piece
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False)
    inputs = _inputs(rows, LATENT, seed=24, dtype=torch.bfloat16)
    shard_elems = rows * 399
    assert pcie_dma.PCIeDmaAllReduce._pick_pieces(shard_elems, shard_elems * 2) == 2
    blocks = emu.run(lambda ring, rank: ring.reduce_scatter_columns(inputs[rank], wire="fp32"))
    expected = column_reduce_scatter_reference(inputs, WORLD, "fp32")
    for rank, block in enumerate(blocks):
        assert torch.equal(block, expected[rank])
    used = emu.rings[0]._send_counters.nonzero().flatten().tolist()
    assert len(used) == (WORLD - 1) * 2 + 1


def test_reduce_scatter_consumer_block_width_matches_aligned_shards() -> None:
    """A row-parallel consumer with eight-aligned 400-column input shards
    (TP9 latent up-projection) gets blocks of its own width: the last rank
    holds 384 real columns and 16 zeros."""
    emu = EmulatedRing(WORLD, MAX_BYTES)
    inputs = _inputs(144, LATENT, seed=25, dtype=torch.bfloat16)

    def calls(ring, rank):
        eager = ring.reduce_scatter_columns(inputs[rank], wire="fp32", cols=400)
        replayed = ring.reduce_scatter_columns(inputs[rank], wire="fp32", cols=400)
        return eager, replayed.clone(), list(ring._replay_entries)

    results = emu.run(calls)
    expected = column_reduce_scatter_reference(inputs, WORLD, "fp32", cols=400)
    for rank, (eager, replayed, keys) in enumerate(results):
        assert eager.shape == (144, 400)
        assert torch.equal(eager, expected[rank])
        assert torch.equal(replayed, expected[rank])
        assert keys[0][-1] == 400
    last = results[WORLD - 1][0]
    assert torch.equal(last[:, 384:], torch.zeros_like(last[:, 384:]))
    reference = sum(x.double() for x in inputs).to(torch.bfloat16)
    assert torch.equal(last[:, :384], reference[:, 3200:])
    ring = emu.rings[0]
    # The blocks of the default and the consumer width live on separate entries.
    assert ring._reduce_scatter_key(inputs[0], "fp32", 399) != ring._reduce_scatter_key(
        inputs[0], "fp32", 400
    )


def test_reduce_scatter_rejects_unsupported_inputs() -> None:
    ring = EmulatedRing(WORLD, MAX_BYTES).rings[0]
    good = torch.empty(144, LATENT, dtype=torch.bfloat16)
    assert ring.should_reduce_scatter_columns(good)
    assert ring.should_reduce_scatter_columns(good, cols=400)
    assert not ring.should_reduce_scatter_columns(good, cols=398)
    assert not ring.should_reduce_scatter_columns(good.float())
    assert not ring.should_reduce_scatter_columns(good.t())
    assert not ring.should_reduce_scatter_columns(torch.empty(7, LATENT, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="wire must be one of"):
        ring.reduce_scatter_columns(good, wire="fp16")
    with pytest.raises(ValueError, match="column reduce-scatter"):
        ring.reduce_scatter_columns(good.float())
    with pytest.raises(ValueError, match="column reduce-scatter"):
        ring.reduce_scatter_columns(good, cols=300)


# ---------------------------------------------------------------------------
# The Kimi-K3 layer sequence on one channel
# ---------------------------------------------------------------------------


def test_layer_sequence_mixes_ops_on_one_channel_with_replay() -> None:
    """attention all-reduce -> gather pair -> reduce-scatter -> MoE
    all-reduce, twice: three replay entries, every result correct, and the
    borrowed all-reduce buffer survives the two other ops in between."""
    emu = EmulatedRing(WORLD, MAX_BYTES, replay_max_entries=4)
    rows = 144
    hidden_a = _inputs(rows, HIDDEN, seed=30, dtype=torch.bfloat16)
    hidden_b = _inputs(rows, HIDDEN, seed=31, dtype=torch.bfloat16)
    firsts, seconds = _pair_inputs(rows, seed=32)
    latent = _inputs(rows, LATENT, seed=33, dtype=torch.bfloat16)
    expected_a = ring_all_reduce_reference(hidden_a, WORLD)
    expected_b = ring_all_reduce_reference(hidden_b, WORLD)
    expected_rs = column_reduce_scatter_reference(latent, WORLD, "fp32")

    def layer(ring, rank, warm: bool):
        a1 = ring.all_reduce(hidden_a[rank], borrow_output=True)
        a1_view = a1
        gathered = ring.all_gather_pair(firsts[rank], seconds[rank])
        block = ring.reduce_scatter_columns(latent[rank], wire="fp32")
        a1_after = a1_view.clone()
        # The MoE all-reduce consumes the attention output's buffer in place
        # when it is the ring's static buffer (warm layers).
        source = a1 if warm and ring.is_ring_storage(a1) else hidden_b[rank].clone()
        source.copy_(hidden_b[rank])
        a5 = ring.all_reduce(source, borrow_output=True)
        return a1_after, (gathered[0].clone(), gathered[1].clone()), block.clone(), a5.clone()

    def two_layers(ring, rank):
        cold = layer(ring, rank, warm=False)
        warm = layer(ring, rank, warm=True)
        return cold, warm, list(ring._replay_entries)

    results = emu.run(two_layers)
    for cold, warm, keys in results:
        for a1, gathered, block, a5 in (cold, warm):
            assert torch.equal(a1, expected_a)
            assert torch.equal(a5, expected_b)
        _check_gathered([cold[1], warm[1]], firsts, seconds)
    for rank, (cold, warm, keys) in enumerate(results):
        assert torch.equal(cold[2], expected_rs[rank])
        assert torch.equal(warm[2], expected_rs[rank])
        # Both all-reduces of a layer share one entry (same numel/dtype).
        assert sorted(k[0] for k in keys) == ["ag_pair", "ar", "rs_fp32"]
