"""Independent native Trellis decoder used by offline production qualification."""

from functools import lru_cache
import numpy as np
import torch
from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_direct_lut_cpu
from b12x._lib.quant.sqg_fp16_d3l import sqg_fp16_d3l_direct_lut_cpu

_MCG = np.uint64(0xCBAC1FED)




_MASK = np.uint32(0x8FFF8FFF)


_ORC = np.uint32(0x3B603B60)


def _decode_3inst_fp16(window: np.ndarray) -> np.ndarray:
    value = window.astype(np.uint64)
    value = ((value * _MCG) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    value = np.uint32((value & _MASK) ^ _ORC)
    low = (value & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        ((value >> np.uint32(16)) & np.uint32(0xFFFF))
        .astype(np.uint16)
        .view(np.float16)
    )
    return (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)








@lru_cache(maxsize=None)
def _sqg_xor_cheb_t12_table(bits: int) -> np.ndarray:
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported SQG-XOR-Cheb-T12 test rate K{bits}")
    rate_index = bits - 2
    labels = sqg_xor_cheb_t12_direct_lut_cpu()[
        rate_index << 16 : (rate_index + 1) << 16
    ]
    return labels.view(torch.float8_e4m3fn).to(torch.float16).numpy()


def _decode_sqg_xor_cheb_t12_fp16(
    window: np.ndarray, bits: int
) -> np.ndarray:
    indices = np.asarray(window, dtype=np.uint32) & np.uint32(0xFFFF)
    return _sqg_xor_cheb_t12_table(bits)[indices]


@lru_cache(maxsize=None)
def _sqg_fp16_table(bits):
    return sqg_fp16_d3l_direct_lut_cpu(bits).numpy()


def _decode_lane(
    tile_words: np.ndarray,
    lane: int,
    bits: int,
    *,
    codebook: str = "mcg",
) -> np.ndarray:
    width = 8 * bits
    values = []
    for weight in range(8):
        end_bit = (lane * 8 + weight + 257) * bits
        start_bit = end_bit - 16
        first_word = start_bit // 32
        last_word = (end_bit - 1) // 32
        shift = (last_word + 1) * 32 - end_bit
        first = tile_words[..., first_word % width].astype(np.uint64)
        last = tile_words[..., last_word % width].astype(np.uint64)
        merged = (first << np.uint64(32)) | last
        window = ((merged >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(
            np.uint32
        )
        if codebook == "mcg":
            values.append(_decode_3inst_fp16(window))
        elif codebook == "sqg_e4m3":
            values.append(_decode_sqg_xor_cheb_t12_fp16(window, bits))
        elif codebook == "sqg_fp16":
            values.append(_sqg_fp16_table(bits)[window])
        else:
            raise ValueError(f"unsupported reference codebook {codebook!r}")
    return np.stack(values, axis=-1).astype(np.float16)


def _reconstruct_native(
    trellis: torch.Tensor, *, codebook: str = "mcg"
) -> torch.Tensor:
    native = trellis.detach().cpu().numpy()
    bits = int(native.shape[-1]) // 16
    k_tiles, n_tiles, _ = native.shape
    packed = native.view(np.uint16).reshape(k_tiles, n_tiles, 8 * bits, 2)
    words = packed[..., 0].astype(np.uint32) | (
        packed[..., 1].astype(np.uint32) << np.uint32(16)
    )
    output = np.zeros((k_tiles * 16, n_tiles * 16), dtype=np.float16)
    for k_tile in range(k_tiles):
        for n_tile in range(n_tiles):
            lanes = np.stack(
                [
                    _decode_lane(
                        words[k_tile, n_tile], lane, bits, codebook=codebook
                    )
                    for lane in range(32)
                ]
            )
            block = np.zeros((16, 16), dtype=np.float16)
            for lane in range(32):
                row0 = (lane % 4) * 2
                rows = (row0, row0 + 1, row0 + 8, row0 + 9)
                col0 = lane // 8
                col1 = col0 + 4
                parity = (lane >> 2) & 1
                for weight in range(8):
                    block[
                        rows[weight % 4],
                        2 * (col0 if weight < 4 else col1) + parity,
                    ] = lanes[lane, weight]
            output[
                k_tile * 16 : (k_tile + 1) * 16,
                n_tile * 16 : (n_tile + 1) * 16,
            ] = block
    return torch.from_numpy(output)
