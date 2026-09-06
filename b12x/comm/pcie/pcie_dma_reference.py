"""Arithmetic contract of the lossless PCIe DMA ring all-reduce.

``PCIeDmaAllReduce`` reduces a flat buffer as ``world`` chunks. Chunk ``c``
travels the rank ring starting at rank ``c``: rank ``c`` sends its own values,
rank ``c + 1`` adds its values to them, rank ``c + 2`` adds its values to that
sum, and so on, so every element of chunk ``c`` is summed in the rank order
``c, c + 1, ..., c - 1`` with one add per reduce-scatter hop. Each add reads
both operands as fp32 and stores the sum, which rounds it to bf16
(``cvt.rn.bf16x2.f32``). The all-gather phase copies finished chunks, so all
ranks end with the same bits.

Two properties of that schedule are configurable and are defined here so the
CUDA implementation and its CPU proofs cannot drift apart:

* **Chunk mapping** — which elements form chunk ``c`` (``piece_offset_elems``,
  ``granule_elems_for``, ``chunk_of_flat_elements``). The served mapping cuts
  the flat buffer into ``world`` contiguous shards, which makes an element's
  summation order depend on the tensor's row count. The granule mapping
  assigns fixed-size granules round-robin, which makes it depend only on the
  element's position inside a ``world * granule`` row period.
* **Hop precision** — how many trailing reduce-scatter hops carry the running
  sum as fp32 instead of bf16 (``ring_schedule``, ``chain_sum``,
  ``rounding_count``).

``ring_schedule`` is the step table the ring executes; ``execute_schedule``
runs that same table over CPU tensors, and ``ring_all_reduce_reference``
computes the intended result directly from the chain definition. Agreement of
the two is what makes the schedule table a proof rather than a restatement.

CPU only, torch only; no CUDA dependency.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch

__all__ = [
    "RingStep",
    "chain_order",
    "chain_sum",
    "chunk_of_flat_elements",
    "execute_schedule",
    "granule_elems_for",
    "piece_offset_elems",
    "ring_all_reduce_reference",
    "ring_schedule",
    "rounding_count",
    "scratch_step_widths",
]


# --------------------------------------------------------------------------
# Chunk mapping
# --------------------------------------------------------------------------


def granule_elems_for(
    shape: tuple[int, ...],
    world: int,
    rows_per_granule: int,
    max_pieces: int,
) -> int:
    """Elements per granule of the row-count-invariant mapping for a tensor of
    ``shape``, or 0 when that tensor must use the served contiguous mapping.

    The granule mapping needs a 2-D interpretation (the last dimension is the
    row width), a row count that is a multiple of ``world * rows_per_granule``
    so that every chunk owns the same number of whole granules, at most
    ``max_pieces`` granules per chunk (the ring's per-step piece budget), and
    a granule of a multiple of eight elements (the add kernel's pack size and
    the 16-byte alignment of every copy). ``rows_per_granule <= 0`` selects
    the served mapping. Every rank sees the same shape, so every rank reaches
    the same decision.
    """
    if rows_per_granule <= 0 or len(shape) < 2:
        return 0
    width = int(shape[-1])
    if width <= 0:
        return 0
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    granules_per_chunk, tail = divmod(rows, world * rows_per_granule)
    elems = rows_per_granule * width
    if tail or granules_per_chunk < 1 or granules_per_chunk > max_pieces:
        return 0
    if elems % 8:
        return 0
    return elems


def piece_offset_elems(
    chunk: int,
    piece: int,
    *,
    world: int,
    shard_elems: int,
    piece_elems: int,
    granule: bool,
) -> int:
    """First flat element of piece ``piece`` of chunk ``chunk``.

    Served mapping (``granule`` false): chunk ``c`` is the contiguous range
    ``[c * shard_elems, (c + 1) * shard_elems)``, cut into equal pieces.
    Granule mapping (``granule`` true): a piece is one granule of
    ``piece_elems`` elements and granule ``b`` of the flat buffer belongs to
    chunk ``b % world``, so piece ``p`` of chunk ``c`` is granule
    ``p * world + c``.
    """
    if granule:
        return (piece * world + chunk) * piece_elems
    return chunk * shard_elems + piece * piece_elems


def chunk_of_flat_elements(
    numel: int, world: int, granule_elems: int = 0
) -> torch.Tensor:
    """Ring chunk index of every flat element (``int64[numel]``)."""
    if numel % world:
        raise ValueError(f"{numel} elements do not split into {world} shards")
    index = torch.arange(numel, dtype=torch.int64)
    if granule_elems <= 0:
        return index // (numel // world)
    if numel % (world * granule_elems):
        raise ValueError(
            f"{numel} elements are not a multiple of world * granule "
            f"({world} * {granule_elems})"
        )
    return (index // granule_elems) % world


# --------------------------------------------------------------------------
# Chain arithmetic
# --------------------------------------------------------------------------


def chain_order(chunk: int, world: int) -> list[int]:
    """Rank order in which chunk ``chunk`` is summed."""
    return [(chunk + i) % world for i in range(world)]


def _check_hops(world: int, fp32_hops: int) -> None:
    if not 0 <= fp32_hops <= world - 2:
        raise ValueError(
            f"fp32 hops must lie in [0, world - 2] = [0, {world - 2}], got {fp32_hops}"
        )


def rounding_count(world: int, fp32_hops: int = 0) -> int:
    """bf16 roundings applied to one element (``world - 1`` adds, the last
    ``fp32_hops`` of the non-final ones kept in fp32)."""
    _check_hops(world, fp32_hops)
    return world - 1 - fp32_hops


def chain_sum(
    terms: list[torch.Tensor], order: list[int], fp32_hops: int = 0
) -> torch.Tensor:
    """``(((x[o0] + x[o1]) + x[o2]) + ...)`` with the ring's roundings.

    Add ``i`` (``1 <= i < len(order)``) is an fp32 sum; its result is rounded
    to the input dtype unless the add feeds an fp32 hop, i.e. unless
    ``len(order) - 1 - fp32_hops <= i < len(order) - 1``. The final add is
    always rounded (the output dtype is the input dtype). Returns the rounded
    result in the input dtype.
    """
    world = len(order)
    _check_hops(world, fp32_hops)
    dtype = terms[0].dtype
    acc = terms[order[0]].float()
    for i in range(1, world):
        acc = acc + terms[order[i]].float()
        keep_fp32 = world - 1 - fp32_hops <= i < world - 1
        if not keep_fp32:
            acc = acc.to(dtype).float()
    return acc.to(dtype)


def ring_all_reduce_reference(
    inputs: list[torch.Tensor],
    world: int,
    *,
    granule_elems: int = 0,
    fp32_hops: int = 0,
) -> torch.Tensor:
    """Bit-level reference of the ring all-reduce of ``inputs`` (one tensor per
    rank, equal shapes) for the given mapping and fp32 hop count."""
    if len(inputs) != world:
        raise ValueError(f"expected {world} inputs, got {len(inputs)}")
    flat = [x.reshape(-1) for x in inputs]
    numel = flat[0].numel()
    chunks = chunk_of_flat_elements(numel, world, granule_elems)
    out = torch.empty_like(flat[0])
    for chunk in range(world):
        sel = chunks == chunk
        terms = [x[sel] for x in flat]
        out[sel] = chain_sum(terms, chain_order(chunk, world), fp32_hops)
    return out.view_as(inputs[0])


# --------------------------------------------------------------------------
# Step table
# --------------------------------------------------------------------------


class RingStep(NamedTuple):
    """One step of the ring, identical for every piece and every rank.

    ``send_offset`` / ``recv_offset`` are relative to the executing rank:
    the step sends chunk ``(rank + send_offset) % world`` to the next rank and
    reduces or stores chunk ``(rank + recv_offset) % world`` received from the
    previous one. ``send_from`` names the buffer holding the payload
    (``"input"`` the caller's tensor, ``"out"`` the output tensor,
    ``"stage"`` the rank-local fp32 stage, ``"scratch"`` the rank's receive
    area of step ``send_step``), ``payload_fp32`` says whether the payload is
    an fp32 running sum (twice the bytes, and the receive area of this step is
    twice as wide), ``op`` is the main-stream operation (``"add"`` the served
    bf16 add, ``"add_mixed"`` an add of mode ``add_mode``, ``"copy"`` the
    all-gather placement copy) and ``sum_to`` names the buffer the add writes
    (``"scratch"`` means this step's own receive area, i.e. in place).
    """

    step: int
    reduce: bool
    send_offset: int
    recv_offset: int
    send_from: str
    send_step: int
    payload_fp32: bool
    op: str
    add_mode: str
    sum_to: str


def _sum_location(step: int, world: int, fp32_hops: int) -> tuple[str, int]:
    """Buffer holding the running sum that reduce-scatter step ``step``
    produces, and the step whose receive area it is (-1 if not a receive
    area).

    The final add always rounds into the output tensor. With fp32 hops, the
    add that feeds the first fp32 hop receives a bf16 payload and cannot
    widen it in place, so its fp32 sum goes to the rank-local stage; the
    later fp32 adds accumulate in place in the (double width) area that
    received their payload. Every other add stores bf16 into the output
    tensor, exactly as the served schedule does.
    """
    rs_steps = world - 1
    first_fp32_payload = rs_steps - fp32_hops
    if not fp32_hops or step >= rs_steps - 1 or step < first_fp32_payload - 1:
        return "out", -1
    if step == first_fp32_payload - 1:
        return "stage", -1
    return "scratch", step


@functools.cache
def ring_schedule(world: int, fp32_hops: int = 0) -> tuple[RingStep, ...]:
    """The ``2 * (world - 1)`` steps of the lossless ring for ``fp32_hops``
    trailing fp32 reduce-scatter hops (0 = the served bf16 schedule)."""
    _check_hops(world, fp32_hops)
    rs_steps = world - 1
    first_fp32_payload = rs_steps - fp32_hops
    steps: list[RingStep] = []
    for k in range(2 * rs_steps):
        if k < rs_steps:
            if k == 0:
                send_from, send_step = "input", -1
            else:
                send_from, send_step = _sum_location(k - 1, world, fp32_hops)
            received_fp32 = bool(fp32_hops) and first_fp32_payload <= k < rs_steps
            sum_fp32 = bool(fp32_hops) and first_fp32_payload - 1 <= k < rs_steps - 1
            if received_fp32 or sum_fp32:
                op = "add_mixed"
                add_mode = (
                    "bf16_f32_f32"
                    if received_fp32 and sum_fp32
                    else "bf16_f32_bf16"
                    if received_fp32
                    else "bf16_bf16_f32"
                )
            else:
                op, add_mode = "add", ""
            sum_to, _ = _sum_location(k, world, fp32_hops)
            steps.append(
                RingStep(
                    step=k,
                    reduce=True,
                    send_offset=-k,
                    recv_offset=-k - 1,
                    send_from=send_from,
                    send_step=send_step,
                    payload_fp32=received_fp32,
                    op=op,
                    add_mode=add_mode,
                    sum_to=sum_to,
                )
            )
        else:
            gather = k - rs_steps
            steps.append(
                RingStep(
                    step=k,
                    reduce=False,
                    send_offset=1 - gather,
                    recv_offset=-gather,
                    send_from="out",
                    send_step=-1,
                    payload_fp32=False,
                    op="copy",
                    add_mode="",
                    sum_to="",
                )
            )
    return tuple(steps)


def scratch_step_widths(world: int, fp32_hops: int = 0) -> tuple[int, ...]:
    """Receive-area width of every step in shard capacities (1 for a bf16
    payload, 2 for an fp32 running sum)."""
    return tuple(
        2 if step.payload_fp32 else 1 for step in ring_schedule(world, fp32_hops)
    )


# --------------------------------------------------------------------------
# CPU execution of the step table
# --------------------------------------------------------------------------


def _check_add_mode(step: RingStep, received: torch.dtype) -> None:
    """Fail if the step's declared add does not match the dtypes it meets:
    the local operand is always bf16, the received operand carries the payload
    dtype and the result is bf16 only when it lands in the output tensor."""
    if step.op == "add":
        expected = ""
        if received != torch.bfloat16 or step.sum_to != "out":
            raise ValueError(f"step {step.step}: served add on a widened payload")
    else:
        b = "f32" if received == torch.float32 else "bf16"
        expected = f"bf16_{b}_{'bf16' if step.sum_to == 'out' else 'f32'}"
    if step.add_mode != expected:
        raise ValueError(
            f"step {step.step}: add mode {step.add_mode!r} does not match the "
            f"payload dtypes ({expected!r})"
        )


def execute_schedule(
    inputs: list[torch.Tensor],
    *,
    world: int,
    pieces: int = 1,
    granule_elems: int = 0,
    fp32_hops: int = 0,
) -> list[torch.Tensor]:
    """Run ``ring_schedule(world, fp32_hops)`` over CPU tensors and return the
    output of every rank.

    The buffers of the CUDA implementation are modelled one for one: an output
    tensor per rank, a receive area per (rank, step, piece) whose dtype follows
    ``payload_fp32``, and a rank-local fp32 stage per piece. Every step first
    performs all ranks' copies (a send reads only values produced by earlier
    steps) and then all ranks' adds or placement copies, which is the order the
    flag handshake enforces on the device.
    """
    if len(inputs) != world:
        raise ValueError(f"expected {world} inputs, got {len(inputs)}")
    dtype = inputs[0].dtype
    flat = [x.reshape(-1) for x in inputs]
    numel = flat[0].numel()
    shard_elems = numel // world
    if shard_elems * world != numel:
        raise ValueError(f"{numel} elements do not split into {world} shards")
    if granule_elems:
        if shard_elems % granule_elems:
            raise ValueError("granule does not tile the shard")
        pieces = shard_elems // granule_elems
    piece_elems = shard_elems // pieces
    if piece_elems * pieces != shard_elems:
        raise ValueError(f"{pieces} pieces do not tile the shard")
    schedule = ring_schedule(world, fp32_hops)

    out = [torch.zeros(numel, dtype=dtype) for _ in range(world)]
    scratch = [
        [
            [
                torch.zeros(
                    piece_elems,
                    dtype=torch.float32 if step.payload_fp32 else dtype,
                )
                for _ in range(pieces)
            ]
            for step in schedule
        ]
        for _ in range(world)
    ]
    stage = [
        [torch.zeros(piece_elems, dtype=torch.float32) for _ in range(pieces)]
        for _ in range(world)
    ]

    def piece_slice(chunk: int, piece: int) -> slice:
        start = piece_offset_elems(
            chunk,
            piece,
            world=world,
            shard_elems=shard_elems,
            piece_elems=piece_elems,
            granule=bool(granule_elems),
        )
        return slice(start, start + piece_elems)

    for step in schedule:
        payloads = []
        for rank in range(world):
            send_chunk = (rank + step.send_offset) % world
            row = []
            for piece in range(pieces):
                if step.send_from == "input":
                    row.append(flat[rank][piece_slice(send_chunk, piece)].clone())
                elif step.send_from == "out":
                    row.append(out[rank][piece_slice(send_chunk, piece)].clone())
                elif step.send_from == "stage":
                    row.append(stage[rank][piece].clone())
                else:
                    row.append(scratch[rank][step.send_step][piece].clone())
            payloads.append(row)
        for rank in range(world):
            nxt = (rank + 1) % world
            for piece in range(pieces):
                scratch[nxt][step.step][piece].copy_(payloads[rank][piece])
        for rank in range(world):
            recv_chunk = (rank + step.recv_offset) % world
            for piece in range(pieces):
                received = scratch[rank][step.step][piece]
                if not step.reduce:
                    out[rank][piece_slice(recv_chunk, piece)] = received.to(dtype)
                    continue
                _check_add_mode(step, received.dtype)
                local = flat[rank][piece_slice(recv_chunk, piece)].float()
                total = local + received.float()
                if step.sum_to == "out":
                    out[rank][piece_slice(recv_chunk, piece)] = total.to(dtype)
                elif step.sum_to == "stage":
                    stage[rank][piece].copy_(total)
                else:
                    scratch[rank][step.step][piece].copy_(total)
    return [x.view_as(inputs[0]) for x in out]
