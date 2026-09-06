"""Whole-H128 ownership for the coupled K2 QSRT expert representation.

Source atom indices select the rotation signs and ordinary input scale, so
they remain part of the loading contract after rank ownership changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QSRTAtomExtent:
    first_atom: int
    atom_count: int

    @property
    def intermediate_channels(self) -> int:
        return self.atom_count * 32


def plan_qsrt_tp9_rank(layer: int, rank: int) -> QSRTAtomExtent:
    """Assign all atoms once and rotate smaller extents across model layers.

    This policy balances resident bytes. Six ranks still own three H128 blocks
    per expert, so it does not imply lower per-layer critical-path work.
    """
    if not 1 <= layer <= 92:
        raise ValueError("coupled K2 QSRT layer must lie in 1..92")
    if not 0 <= rank < 9:
        raise ValueError("TP9 rank must lie in 0..8")
    owner = (rank - (layer - 1)) % 9
    if owner < 5:
        half, half_rank, ranks_in_half = 0, owner, 5
    else:
        half, half_rank, ranks_in_half = 1, owner - 5, 4
    quotient, remainder = divmod(12, ranks_in_half)
    blocks = quotient + int(half_rank < remainder)
    begin = half_rank * quotient + min(half_rank, remainder)
    return QSRTAtomExtent(48 * half + 4 * begin, 4 * blocks)
