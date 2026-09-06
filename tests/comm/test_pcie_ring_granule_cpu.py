"""CPU proofs of the lossless PCIe ring's chunk mapping and hop precision.

The ring's arithmetic contract lives in ``pcie_dma_reference``: the step table
``ring_schedule`` that ``PCIeDmaAllReduce._all_reduce_lossless`` executes on
the device, the element mapping ``piece_offset_elems`` it addresses pieces
with, and ``ring_all_reduce_reference``, which states the intended result
directly from the chain definition. These tests execute the step table over
CPU tensors and check four properties:

* the served mapping (``B12X_PCIE_RING_GRANULE_ROWS=0``) reduces chunk ``c``
  as the contiguous flat block ``c`` summed from rank ``c`` onwards, which is
  what makes an element's summation order depend on the tensor's row count;
* the granule mapping reduces a row block bit-identically whether it is
  all-reduced whole or in row slices that are multiples of
  ``world * granule`` rows;
* ``p`` fp32 reduce-scatter hops leave ``world - 1 - p`` bf16 roundings per
  element, and ``p = world - 2`` is one rounding of the fp32 chain sum;
* the executed step table agrees with the reference on random inputs.

No CUDA: the tests import the pure geometry, the step table and the two
decision functions of the channel class.
"""

from __future__ import annotations

import pytest
import torch

from b12x.comm.pcie.pcie_dma import MAX_PIECES, PCIeDmaAllReduce
from b12x.comm.pcie.pcie_dma_reference import (
    chain_order,
    chunk_of_flat_elements,
    execute_schedule,
    granule_elems_for,
    piece_offset_elems,
    ring_all_reduce_reference,
    ring_schedule,
    rounding_count,
)

WORLD = 9
# Production all-reduce shapes: a 4,608-token prefill chunk and its halves,
# hidden 7,168 (o_proj, final residual, embedding) and 3,584 (latent).
PROD_ROWS = (4608, 2304)
PROD_WIDTHS = (7168, 3584)


def _inputs(
    world: int, rows: int, width: int, seed: int = 0, scale_spread: bool = True
) -> list[torch.Tensor]:
    """Heavy-tailed bf16 activations, one tensor per rank, with a per-rank
    magnitude spread so that the summation order is visible in the bits."""
    generator = torch.Generator().manual_seed(seed)
    out = []
    for rank in range(world):
        values = torch.randn(rows, width, generator=generator, dtype=torch.float32)
        if scale_spread:
            values = values * float(2.0 ** (rank - world // 2))
        out.append(values.to(torch.bfloat16))
    return out


def _stub_channel(granule_rows: int, fp32_hops: int = 0) -> PCIeDmaAllReduce:
    """A channel object carrying only the fields the two mapping decisions
    read. Constructing a real channel needs nine GPUs and IPC handles."""
    channel = PCIeDmaAllReduce.__new__(PCIeDmaAllReduce)
    channel.world_size = WORLD
    channel._granule_rows = granule_rows
    channel._fp32_hops = fp32_hops
    channel._fp8 = ""
    return channel


# --------------------------------------------------------------------------
# (b) the served mapping and order
# --------------------------------------------------------------------------


def test_served_mapping_is_contiguous_shards_summed_from_the_owner() -> None:
    numel, shard = 9 * 32, 32
    chunks = chunk_of_flat_elements(numel, WORLD)
    assert torch.equal(chunks, torch.arange(numel) // shard)
    for chunk in range(WORLD):
        assert chain_order(chunk, WORLD) == [(chunk + i) % WORLD for i in range(WORLD)]
        for piece in range(2):
            assert piece_offset_elems(
                chunk,
                piece,
                world=WORLD,
                shard_elems=shard,
                piece_elems=shard // 2,
                granule=False,
            ) == chunk * shard + piece * (shard // 2)


def test_served_schedule_reproduces_the_served_order() -> None:
    inputs = _inputs(WORLD, rows=18, width=8, seed=1)
    outputs = execute_schedule(inputs, world=WORLD, pieces=2)
    expected = ring_all_reduce_reference(inputs, WORLD)
    for rank, out in enumerate(outputs):
        assert torch.equal(out, expected), f"rank {rank}"


def test_served_mapping_is_not_row_count_invariant() -> None:
    """The structural reason the granule mapping exists: with the served
    mapping the two halves of a row block do not reproduce the whole."""
    inputs = _inputs(WORLD, rows=36, width=8, seed=2)
    whole = execute_schedule(inputs, world=WORLD, pieces=1)[0]
    halves = torch.cat(
        [
            execute_schedule([x[:18] for x in inputs], world=WORLD, pieces=1)[0],
            execute_schedule([x[18:] for x in inputs], world=WORLD, pieces=1)[0],
        ]
    )
    assert not torch.equal(whole, halves)
    differing = int((whole != halves).sum())
    assert differing > whole.numel() // 4


# --------------------------------------------------------------------------
# (a) the granule mapping is row-count invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize("granule_rows", [1, 2, 4])
def test_granule_mapping_splits_bit_identically(granule_rows: int) -> None:
    rows = 8 * WORLD * granule_rows
    width = 8
    inputs = _inputs(WORLD, rows=rows, width=width, seed=3)
    granule_elems = granule_rows * width
    whole = execute_schedule(inputs, world=WORLD, granule_elems=granule_elems)
    half = rows // 2
    first = execute_schedule(
        [x[:half] for x in inputs], world=WORLD, granule_elems=granule_elems
    )
    second = execute_schedule(
        [x[half:] for x in inputs], world=WORLD, granule_elems=granule_elems
    )
    for rank in range(WORLD):
        assert torch.equal(whole[rank][:half], first[rank]), f"rank {rank} first half"
        assert torch.equal(whole[rank][half:], second[rank]), f"rank {rank} second half"


def test_granule_mapping_splits_bit_identically_with_fp32_hops() -> None:
    rows, width, granule_rows = 4 * WORLD * 2, 8, 2
    inputs = _inputs(WORLD, rows=rows, width=width, seed=4)
    granule_elems = granule_rows * width
    for hops in (1, 3, WORLD - 2):
        whole = execute_schedule(
            inputs, world=WORLD, granule_elems=granule_elems, fp32_hops=hops
        )[0]
        half = rows // 2
        parts = [
            execute_schedule(
                [x[:half] for x in inputs],
                world=WORLD,
                granule_elems=granule_elems,
                fp32_hops=hops,
            )[0],
            execute_schedule(
                [x[half:] for x in inputs],
                world=WORLD,
                granule_elems=granule_elems,
                fp32_hops=hops,
            )[0],
        ]
        assert torch.equal(whole, torch.cat(parts)), f"{hops} fp32 hops"


@pytest.mark.parametrize("granule_rows", [128, 256])
@pytest.mark.parametrize("width", PROD_WIDTHS)
def test_production_rows_share_the_granule_order(granule_rows: int, width: int) -> None:
    """At the production shapes every row of the 4,608-row chunk keeps the
    chunk index it has inside its 2,304-row half, so both halves reduce every
    element in the same rank order as the whole chunk."""
    whole, half = PROD_ROWS
    assert granule_elems_for((whole, width), WORLD, granule_rows, MAX_PIECES) > 0
    assert granule_elems_for((half, width), WORLD, granule_rows, MAX_PIECES) > 0
    rows = torch.arange(whole)
    chunk_whole = (rows // granule_rows) % WORLD
    chunk_halves = torch.cat(
        [((torch.arange(half) // granule_rows) % WORLD) for _ in range(whole // half)]
    )
    assert torch.equal(chunk_whole, chunk_halves)


# --------------------------------------------------------------------------
# (c) hop precision
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hops", list(range(WORLD - 1)))
def test_rounding_count_per_element(hops: int) -> None:
    schedule = ring_schedule(WORLD, hops)
    reduce_steps = [step for step in schedule if step.reduce]
    assert len(reduce_steps) == WORLD - 1
    rounded = [step for step in reduce_steps if step.sum_to == "out"]
    assert len(rounded) == WORLD - 1 - hops == rounding_count(WORLD, hops)
    # The final add always rounds into the output tensor; the adds that keep
    # fp32 are the ``hops`` adds that feed the trailing fp32 hops, i.e. the
    # ones just before the final add.
    assert reduce_steps[-1].sum_to == "out"
    kept = [index for index, step in enumerate(reduce_steps) if step.sum_to != "out"]
    assert kept == list(range(WORLD - 2 - hops, WORLD - 2))


def test_full_fp32_wire_is_the_chain_sum_rounded_once() -> None:
    inputs = _inputs(WORLD, rows=WORLD, width=8, seed=5)
    outputs = execute_schedule(inputs, world=WORLD, fp32_hops=WORLD - 2)
    flat = [x.reshape(-1) for x in inputs]
    numel = flat[0].numel()
    shard = numel // WORLD
    expected = torch.empty(numel, dtype=torch.bfloat16)
    for chunk in range(WORLD):
        span = slice(chunk * shard, (chunk + 1) * shard)
        acc = torch.zeros(shard, dtype=torch.float32)
        for rank in chain_order(chunk, WORLD):
            acc = acc + flat[rank][span].float()
        expected[span] = acc.to(torch.bfloat16)
    for rank, out in enumerate(outputs):
        assert torch.equal(out.reshape(-1), expected), f"rank {rank}"


def test_error_against_fp64_falls_with_every_fp32_hop() -> None:
    inputs = _inputs(WORLD, rows=72, width=64, seed=6)
    exact = sum(x.double() for x in inputs)
    errors = []
    for hops in range(WORLD - 1):
        out = execute_schedule(inputs, world=WORLD, fp32_hops=hops)[0]
        errors.append(float((out.double() - exact).norm() / exact.norm()))
    print("rel-L2 vs fp64 by fp32 hop count:", [f"{e:.3e}" for e in errors])
    assert all(b <= a for a, b in zip(errors, errors[1:], strict=False)), errors
    assert errors[-1] < errors[0] / 2


# --------------------------------------------------------------------------
# (d) the step table agrees with the reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("world", [8, 9])
@pytest.mark.parametrize("granule_rows", [0, 1, 2])
@pytest.mark.parametrize("hops", [0, 1, 3, 6])
def test_schedule_matches_reference(world: int, granule_rows: int, hops: int) -> None:
    if hops > world - 2:
        pytest.skip(f"world {world} allows at most {world - 2} fp32 hops")
    width = 8
    rows = 4 * world * max(granule_rows, 1)
    inputs = _inputs(world, rows=rows, width=width, seed=7 + world)
    granule_elems = granule_rows * width
    pieces = 1 if granule_elems else 2
    outputs = execute_schedule(
        inputs,
        world=world,
        pieces=pieces,
        granule_elems=granule_elems,
        fp32_hops=hops,
    )
    expected = ring_all_reduce_reference(
        inputs, world, granule_elems=granule_elems, fp32_hops=hops
    )
    for rank, out in enumerate(outputs):
        assert torch.equal(out, expected), f"rank {rank}"


def test_pieces_do_not_change_the_result() -> None:
    inputs = _inputs(WORLD, rows=WORLD * 4, width=8, seed=8)
    base = execute_schedule(inputs, world=WORLD, pieces=1)[0]
    for pieces in (2, 4):
        assert torch.equal(
            execute_schedule(inputs, world=WORLD, pieces=pieces)[0], base
        )


# --------------------------------------------------------------------------
# Channel-side decisions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width", PROD_WIDTHS)
def test_granule_elems_for_production_shapes(width: int) -> None:
    for rows in PROD_ROWS:
        assert (
            granule_elems_for((rows, width), WORLD, 128, MAX_PIECES) == 128 * width
        ), rows
        assert (
            granule_elems_for((rows, width), WORLD, 256, MAX_PIECES) == 256 * width
        ), rows


def test_granule_elems_for_falls_back_to_the_served_mapping() -> None:
    width = 7168
    # Disabled by default.
    assert granule_elems_for((4608, width), WORLD, 0, MAX_PIECES) == 0
    # Row count not a multiple of world * granule.
    assert granule_elems_for((4600, width), WORLD, 128, MAX_PIECES) == 0
    assert granule_elems_for((1152 + 16, width), WORLD, 128, MAX_PIECES) == 0
    # Flat input: no row width to granulate by.
    assert granule_elems_for((4608 * width,), WORLD, 128, MAX_PIECES) == 0
    # More granules per chunk than the ring has piece slots.
    assert granule_elems_for((WORLD * (MAX_PIECES + 1), 8), WORLD, 1, MAX_PIECES) == 0
    # Granule not a multiple of the eight-element pack.
    assert granule_elems_for((WORLD * 4, 4), WORLD, 1, MAX_PIECES) == 0


def test_channel_granule_decision_follows_the_wire_mode() -> None:
    inp = torch.empty(0)
    shape = (4608, 7168)
    served = _stub_channel(granule_rows=0)
    assert served._granule_elems(torch.empty(shape, device="meta")) == 0
    granule = _stub_channel(granule_rows=128)
    assert granule._granule_elems(torch.empty(shape, device="meta")) == 128 * 7168
    # A compressed wire keeps the served mapping and no fp32 hops.
    compressed = _stub_channel(granule_rows=128, fp32_hops=3)
    compressed._fp8 = "ring"
    assert compressed._granule_elems(torch.empty(shape, device="meta")) == 0
    assert compressed._fp32_hops_for(torch.empty(shape, device="meta")) == 0
    del inp


def test_channel_fp32_hops_only_for_bf16() -> None:
    channel = _stub_channel(granule_rows=0, fp32_hops=3)
    bf16 = torch.empty(8, dtype=torch.bfloat16, device="meta")
    assert channel._fp32_hops_for(bf16) == 3
    assert (
        channel._fp32_hops_for(torch.empty(8, dtype=torch.float32, device="meta")) == 0
    )


def test_replay_key_separates_equal_sized_shapes() -> None:
    channel = _stub_channel(granule_rows=128)
    wide = torch.empty((2304, 7168), device="meta")
    narrow = torch.empty((4608, 3584), device="meta")
    assert wide.numel() == narrow.numel()
    assert channel._replay_key(wide) != channel._replay_key(narrow)
    served = _stub_channel(granule_rows=0)
    assert served._replay_key(wide) == served._replay_key(narrow)


@pytest.mark.parametrize("hops", list(range(WORLD - 1)))
def test_scratch_layout_grows_by_one_shard_per_fp32_hop(hops: int) -> None:
    capacity = 1024
    offsets, total = PCIeDmaAllReduce._scratch_layout(capacity, WORLD, hops)
    steps = 2 * (WORLD - 1)
    assert len(offsets) == steps
    assert total == (steps + hops) * capacity
    assert offsets[0] == 0
    assert all(b > a for a, b in zip(offsets, offsets[1:], strict=False))
    if hops == 0:
        assert offsets == tuple(step * capacity for step in range(steps))
