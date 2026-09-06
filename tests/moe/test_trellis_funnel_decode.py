"""Bit-equality of the funnel-shift T12 decode against the legacy chain (GPU).

``packed_decode_sqg_xor_cheb_t12_to_e4m3x2x4`` must return, as four byte
pairs, exactly the bytes ``packed_decode_sqg_xor_cheb_t12_to_e4m3x8`` packs
into its two words, and the pair converter must produce the same f16x2 as
the low/high halves of the legacy converter. The probe drives both on the
same random ring windows through both table address forms (global and
shared) and compares every word.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    fp8x2_e4m3_pair_to_half2,
    fp8x4_e4m3_to_half2x2,
    packed_decode_sqg_xor_cheb_t12_to_e4m3x2x4,
    packed_decode_sqg_xor_cheb_t12_to_e4m3x8,
    shared_ptr_to_u32,
    st_shared_u32,
)
from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_lut_cpu
from b12x._lib.utils import current_cuda_stream
from tests._reference.helpers import require_b12x

require_b12x()

_TABLE_BYTES = 1 << 12


class _FunnelProbe:
    def __init__(self, bits: int, in_shared: bool):
        self.bits = int(bits)
        self.in_shared = bool(in_shared)

    @cute.jit
    def __call__(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(wins, t12, out).launch(
            grid=(1, 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, wins: cute.Tensor, t12: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        lane = Int32(tidx)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            table: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, _TABLE_BYTES // 4], 16
            ]

        storage = smem.allocate(Storage)
        table_smem = shared_ptr_to_u32(storage.table.data_ptr())
        table_gmem = t12.iterator.toint()
        if cutlass.const_expr(self.in_shared):
            # Stage the 4 KiB table: 32 lanes x 32 words.
            for i in cutlass.range_constexpr(_TABLE_BYTES // 4 // 32):
                word = Int32(i * 32) + lane
                st_shared_u32(
                    table_smem + word * Int32(4),
                    Uint32(t12[word * 4].to(Int32))
                    | (Uint32(t12[word * 4 + 1].to(Int32)) << Uint32(8))
                    | (Uint32(t12[word * 4 + 2].to(Int32)) << Uint32(16))
                    | (Uint32(t12[word * 4 + 3].to(Int32)) << Uint32(24)),
                )
            cute.arch.sync_threads()
        wa = Uint32(wins[2 * lane])
        wb = Uint32(wins[2 * lane + 1])
        if cutlass.const_expr(self.in_shared):
            lo, hi = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
                wa, wb, table_smem, self.bits, t12_in_shared=True
            )
            p01, p23, p45, p67 = packed_decode_sqg_xor_cheb_t12_to_e4m3x2x4(
                wa, wb, table_smem, self.bits, t12_in_shared=True
            )
        else:
            lo, hi = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
                wa, wb, table_gmem, self.bits, t12_in_shared=False
            )
            p01, p23, p45, p67 = packed_decode_sqg_xor_cheb_t12_to_e4m3x2x4(
                wa, wb, table_gmem, self.bits, t12_in_shared=False
            )
        h0, h1 = fp8x4_e4m3_to_half2x2(lo)
        h2, h3 = fp8x4_e4m3_to_half2x2(hi)
        f0 = fp8x2_e4m3_pair_to_half2(p01)
        f1 = fp8x2_e4m3_pair_to_half2(p23)
        f2 = fp8x2_e4m3_pair_to_half2(p45)
        f3 = fp8x2_e4m3_pair_to_half2(p67)
        base = 14 * lane
        out[base + 0] = Int32(lo)
        out[base + 1] = Int32(hi)
        out[base + 2] = Int32(p01)
        out[base + 3] = Int32(p23)
        out[base + 4] = Int32(p45)
        out[base + 5] = Int32(p67)
        out[base + 6] = Int32(h0)
        out[base + 7] = Int32(h1)
        out[base + 8] = Int32(h2)
        out[base + 9] = Int32(h3)
        out[base + 10] = Int32(f0)
        out[base + 11] = Int32(f1)
        out[base + 12] = Int32(f2)
        out[base + 13] = Int32(f3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("in_shared", [True, False])
def test_funnel_decode_bit_equals_legacy(bits: int, in_shared: bool) -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260906 + 3 * bits + int(in_shared))
    wins = torch.zeros(64, dtype=torch.int32, device=device)
    t12 = sqg_xor_cheb_t12_lut_cpu().to(device)
    out = torch.zeros(32 * 14, dtype=torch.int32, device=device)

    def args():
        return (
            from_dlpack(wins, assumed_align=16),
            from_dlpack(t12, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_FunnelProbe(bits, in_shared), *args())
    mismatched = 0
    for _ in range(512):
        wins.copy_(
            torch.randint(
                -(2**31), 2**31 - 1, (64,), dtype=torch.int32, device=device
            )
        )
        compiled(*args())
        torch.cuda.synchronize()
        o = out.view(32, 14).to(torch.int64) & 0xFFFFFFFF
        lo, hi = o[:, 0], o[:, 1]
        expected_pairs = torch.stack(
            [lo & 0xFFFF, lo >> 16, hi & 0xFFFF, hi >> 16], dim=1
        )
        mismatched += int((o[:, 2:6] != expected_pairs).sum())
        mismatched += int((o[:, 6:10] != o[:, 10:14]).sum())
    assert mismatched == 0, mismatched
