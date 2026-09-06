"""Arithmetic contract of the lossless PCIe DMA ring all-reduce.

``PCIeDmaAllReduce`` reduces a flat buffer as ``world`` chunks. Chunk ``c``
travels the rank ring starting at rank ``c``: rank ``c`` sends its own values,
rank ``c + 1`` adds its values to them, rank ``c + 2`` adds its values to that
sum, and so on, so every element of chunk ``c`` is summed in the rank order
``c, c + 1, ..., c - 1`` with one add per reduce-scatter hop. Each add reads
both operands as fp32 and stores the sum, which rounds it to bf16
(``cvt.rn.bf16x2.f32``). The all-gather phase copies finished chunks, so all
ranks end with the same bits.

Which elements form chunk ``c`` is configurable and is defined here so the
CUDA implementation and its CPU proofs cannot drift apart (
``piece_offset_elems``, ``granule_elems_for``, ``chunk_of_flat_elements``).
The served mapping cuts the flat buffer into ``world`` contiguous shards,
which makes an element's summation order depend on the tensor's row count.
The granule mapping assigns fixed-size granules round-robin, which makes it
depend only on the element's position inside a ``world * granule`` row period.

``ring_schedule`` is the step table the ring executes.

CPU only, torch only; no CUDA dependency.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch

__all__ = [
    "RingStep",
    "chunk_of_flat_elements",
    "granule_elems_for",
    "piece_offset_elems",
    "ring_schedule",
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


# --------------------------------------------------------------------------
# Step table
# --------------------------------------------------------------------------


class RingStep(NamedTuple):
    """One step of the ring, identical for every piece and every rank.

    ``send_offset`` / ``recv_offset`` are relative to the executing rank:
    the step sends chunk ``(rank + send_offset) % world`` to the next rank and
    reduces or stores chunk ``(rank + recv_offset) % world`` received from the
    previous one. ``send_from`` names the buffer holding the payload
    (``"input"`` the caller's tensor, ``"out"`` the output tensor), ``op`` is
    the main-stream operation (``"add"`` the bf16 add of the reduce-scatter
    phase, ``"copy"`` the all-gather placement copy) and ``sum_to`` names the
    buffer the add writes.
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


@functools.cache
def ring_schedule(world: int) -> tuple[RingStep, ...]:
    """The ``2 * (world - 1)`` steps of the lossless ring: ``world - 1``
    reduce-scatter steps that add the received payload to the local input and
    store the bf16 running sum, then ``world - 1`` all-gather steps that place
    the received chunk."""
    rs_steps = world - 1
    steps: list[RingStep] = []
    for k in range(2 * rs_steps):
        if k < rs_steps:
            steps.append(
                RingStep(
                    step=k,
                    reduce=True,
                    send_offset=-k,
                    recv_offset=-k - 1,
                    send_from="input" if k == 0 else "out",
                    send_step=-1,
                    payload_fp32=False,
                    op="add",
                    add_mode="",
                    sum_to="out",
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
