"""Bit-exact equivalence of the two SQG-XOR-Cheb-T12 decode chains (CPU).

The W4A16 trellis kernels decode eight L16 windows per call through the
modal T12 staircase. Two PTX chains exist (``legacy`` and ``funnel``,
selected by ``B12X_SQG_XOR_CHEB_T12_DECODE_CHAIN``); they must read the same
table byte for every window. This module executes the exact PTX text each
builder emits with a small vectorized interpreter and compares every byte
with the reference rank map ``_sqg_xor_cheb_t12_rank_for_codewords`` for all
65,536 codewords in every window position, with random bits everywhere else
in the two 32-bit source words. No GPU is involved.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch

from b12x._lib.intrinsics import (
    SQG_XOR_CHEB_T12_DECODE_CHAINS,
    sqg_xor_cheb_t12_decode_asm,
    sqg_xor_cheb_t12_funnel_decode_asm,
)
from b12x._lib.quant.sqg_e4m3 import (
    _sqg_xor_cheb_t12_rank_for_codewords,
    sqg_xor_cheb_t12_lut_cpu,
)

_MASK32 = np.uint32(0xFFFFFFFF)
_TOKEN_RE = re.compile(r"[,\s]+")


def _imm(token: str) -> int:
    return int(token, 0)


class _PtxVectorInterpreter:
    """Execute the restricted PTX subset of the decode chains on arrays.

    Registers hold ``numpy.uint32`` (or ``uint64``) vectors; ``$k`` operands
    are bound by the caller. Byte loads index ``table`` after subtracting the
    table base bound to the address operand.
    """

    def __init__(self, table: np.ndarray, base: int, base_is_64bit: bool):
        self.table = table
        self.base = base
        self.base_is_64bit = base_is_64bit
        self.regs: dict[str, np.ndarray] = {}

    def _read(self, token: str, width: int = 32):
        if token in self.regs:
            return self.regs[token]
        value = _imm(token) & ((1 << width) - 1)
        return np.uint64(value) if width == 64 else np.uint32(value)

    def run(self, asm: str, bindings: dict[str, np.ndarray | int]) -> None:
        for name, value in bindings.items():
            self.regs[name] = value
        for raw in asm.splitlines():
            line = raw.split("//", 1)[0].strip()
            if not line or line in "{}" or line.startswith(".reg"):
                continue
            if not line.endswith(";"):
                raise ValueError(f"unterminated PTX statement: {raw!r}")
            line = line[:-1]
            if line.startswith("ld."):
                self._load(line)
                continue
            tokens = [t for t in _TOKEN_RE.split(line) if t]
            op, args = tokens[0], tokens[1:]
            self._execute(op, args)

    def _load(self, line: str) -> None:
        op, rest = line.split(None, 1)
        dst, addr = [t.strip() for t in rest.split(",", 1)]
        addr = addr.strip("[]")
        address = self.regs[addr]
        if op == "ld.shared.u8":
            offset = (address - np.uint32(self.base)) & _MASK32
        elif op == "ld.global.u8":
            offset = (address - np.uint64(self.base)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        else:
            raise ValueError(f"unsupported load {op}")
        offset = offset.astype(np.int64)
        if offset.min() < 0 or offset.max() >= self.table.shape[0]:
            raise AssertionError("table index out of range")
        self.regs[dst] = self.table[offset].astype(np.uint32)

    def _execute(self, op: str, args: list[str]) -> None:
        r = self.regs
        if op == "mov.b32":
            r[args[0]] = self._read(args[1]) + np.uint32(0)
        elif op == "bfe.u32":
            a = self._read(args[1])
            pos, length = _imm(args[2]), _imm(args[3])
            r[args[0]] = (a >> np.uint32(pos)) & np.uint32((1 << length) - 1)
        elif op == "shr.u32":
            r[args[0]] = self._read(args[1]) >> np.uint32(_imm(args[2]))
        elif op == "shl.b32":
            r[args[0]] = (self._read(args[1]) << np.uint32(_imm(args[2]))) & _MASK32
        elif op == "bfi.b32":
            a, b = self._read(args[1]), self._read(args[2])
            pos, length = _imm(args[3]), _imm(args[4])
            field = np.uint32(((1 << length) - 1) << pos)
            r[args[0]] = (b & ~field) | ((a << np.uint32(pos)) & field)
        elif op == "xor.b32":
            r[args[0]] = self._read(args[1]) ^ self._read(args[2])
        elif op == "or.b32":
            r[args[0]] = self._read(args[1]) | self._read(args[2])
        elif op == "and.b32":
            r[args[0]] = self._read(args[1]) & self._read(args[2])
        elif op == "add.u32":
            r[args[0]] = (self._read(args[1]) + self._read(args[2])) & _MASK32
        elif op == "mad.lo.u32":
            a = self._read(args[1]).astype(np.uint64)
            b = np.uint64(_imm(args[2]))
            c = np.uint64(_imm(args[3]))
            r[args[0]] = ((a * b + c) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        elif op == "brev.b32":
            a = self._read(args[1])
            out = np.zeros_like(a)
            for bit in range(32):
                out |= ((a >> np.uint32(bit)) & np.uint32(1)) << np.uint32(31 - bit)
            r[args[0]] = out
        elif op == "shf.r.clamp.b32":
            low = self._read(args[1]).astype(np.uint64)
            high = self._read(args[2]).astype(np.uint64)
            amount = min(_imm(args[3]), 32)
            merged = (high << np.uint64(32)) | low
            r[args[0]] = ((merged >> np.uint64(amount)) & np.uint64(0xFFFFFFFF)).astype(
                np.uint32
            )
        elif op == "cvt.u64.u32":
            r[args[0]] = self._read(args[1]).astype(np.uint64)
        elif op == "add.u64":
            r[args[0]] = self._read(args[1], 64) + self._read(args[2], 64)
        else:
            raise ValueError(f"unsupported PTX instruction {op}")


def _reference_bytes(windows: np.ndarray, bits: int) -> np.ndarray:
    codewords = torch.from_numpy(windows.astype(np.int64))
    index = _sqg_xor_cheb_t12_rank_for_codewords(codewords, bits) >> 4
    return sqg_xor_cheb_t12_lut_cpu()[index].numpy().astype(np.uint32)


def _window(word: np.ndarray, shift: int) -> np.ndarray:
    return (word >> np.uint32(shift)) & np.uint32(0xFFFF)


def _run_chain(
    chain: str, bits: int, in_shared: bool, win_a: np.ndarray, win_b: np.ndarray
) -> list[np.ndarray]:
    """Return the eight decoded bytes (window order 0..7) of one chain."""
    table = sqg_xor_cheb_t12_lut_cpu().numpy()
    base = 0x10B00 if in_shared else 0x7F3A_0000_1000
    interpreter = _PtxVectorInterpreter(table, base, base_is_64bit=not in_shared)
    if chain == "legacy":
        asm = sqg_xor_cheb_t12_decode_asm(bits, in_shared)
        bindings = {
            "$2": win_a,
            "$3": win_b,
            "$4": np.uint32(base) if in_shared else np.uint64(base),
        }
        interpreter.run(asm, bindings)
        lo, hi = interpreter.regs["$0"], interpreter.regs["$1"]
        return [(word >> np.uint32(8 * k)) & np.uint32(0xFF) for word in (lo, hi) for k in range(4)]
    asm = sqg_xor_cheb_t12_funnel_decode_asm(bits, in_shared)
    bindings = {
        "$4": win_a,
        "$5": win_b,
        "$6": np.uint32(base) if in_shared else np.uint64(base),
    }
    interpreter.run(asm, bindings)
    pairs = [interpreter.regs[f"${k}"] for k in range(4)]
    for pair in pairs:
        assert not np.any(pair >> np.uint32(16)), "pair words must keep bits 31:16 clear"
    return [(pair >> np.uint32(8 * k)) & np.uint32(0xFF) for pair in pairs for k in range(2)]


@pytest.mark.parametrize("chain", SQG_XOR_CHEB_T12_DECODE_CHAINS)
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("in_shared", [True, False])
def test_decode_chain_matches_reference_for_every_codeword(
    chain: str, bits: int, in_shared: bool
) -> None:
    rng = np.random.default_rng(20260906 + 7 * bits + (1 if in_shared else 0))
    codewords = np.arange(1 << 16, dtype=np.uint32)
    for index in range(8):
        # Window ``index`` takes every codeword; every other bit of the two
        # source words is random so garbage outside the window is exercised.
        shift = (3 - (index & 3)) * bits
        win_a = rng.integers(0, 1 << 32, size=codewords.shape[0], dtype=np.uint64).astype(np.uint32)
        win_b = rng.integers(0, 1 << 32, size=codewords.shape[0], dtype=np.uint64).astype(np.uint32)
        window_mask = np.uint32(0xFFFF << shift)
        if index < 4:
            win_b = (win_b & ~window_mask) | (codewords << np.uint32(shift))
        else:
            win_a = (win_a & ~window_mask) | (codewords << np.uint32(shift))
        decoded = _run_chain(chain, bits, in_shared, win_a, win_b)
        for other in range(8):
            other_shift = (3 - (other & 3)) * bits
            source = win_b if other < 4 else win_a
            expected = _reference_bytes(_window(source, other_shift), bits)
            mismatches = int(np.count_nonzero(decoded[other] != expected))
            assert mismatches == 0, (chain, bits, in_shared, index, other, mismatches)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_funnel_pairs_equal_legacy_halves(bits: int) -> None:
    rng = np.random.default_rng(777 + bits)
    n = 1 << 15
    win_a = rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(np.uint32)
    win_b = rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(np.uint32)
    legacy = _run_chain("legacy", bits, True, win_a, win_b)
    funnel = _run_chain("funnel", bits, True, win_a, win_b)
    for k in range(8):
        assert np.array_equal(legacy[k], funnel[k]), (bits, k)


def test_funnel_chain_uses_the_expected_instructions() -> None:
    asm = sqg_xor_cheb_t12_funnel_decode_asm(2, True)
    assert asm.count("brev.b32") == 2
    assert asm.count("shf.r.clamp.b32") == 8
    assert asm.count("ld.shared.u8") == 8
    assert asm.count("bfe.u32") == 8
    assert "and.b32" not in asm
    legacy = sqg_xor_cheb_t12_decode_asm(2, True)
    assert legacy.count("brev.b32") == 8
    assert legacy.count("and.b32") == 8


def test_decode_chain_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = pytest.importorskip("b12x.moe._shared.kernels.w4a16.kernel")
    monkeypatch.delenv("B12X_SQG_XOR_CHEB_T12_DECODE_CHAIN", raising=False)
    assert kernel._sqg_xor_cheb_t12_decode_chain() == "funnel"
    monkeypatch.setenv("B12X_SQG_XOR_CHEB_T12_DECODE_CHAIN", "legacy")
    assert kernel._sqg_xor_cheb_t12_decode_chain() == "legacy"
    monkeypatch.setenv("B12X_SQG_XOR_CHEB_T12_DECODE_CHAIN", "Funnel ")
    assert kernel._sqg_xor_cheb_t12_decode_chain() == "funnel"
    monkeypatch.setenv("B12X_SQG_XOR_CHEB_T12_DECODE_CHAIN", "direct")
    with pytest.raises(ValueError):
        kernel._sqg_xor_cheb_t12_decode_chain()
