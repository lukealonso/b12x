"""gemm.mxfp8_linear: ModelOpt MXFP8 linear vs quantized reference, K-padding
semantics, and CUDA-graph capture of the default fused path.

Curated from b12x tests/test_gemm_mxfp8_linear.py (kept whole — it was
already tight).
"""

from __future__ import annotations

import cutlass.cute as cute
import pytest
import torch

from b12x._lib.utils import convert_sf_from_mma_layout
from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm import blockscaled, mxfp8_linear
from b12x.gemm.blockscaled import _linear as blockscaled_linear
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
)

from ..conftest import require_b12x


def require_mxf8_mma() -> None:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        pytest.skip("CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op")


def _quantize_modelopt_mxfp8_rows(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = map(int, source.shape)
    chunks = width // 32
    blocked = source.to(torch.float32).reshape(rows, chunks, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(max_abs > 0.0, max_abs / 448.0, torch.ones_like(max_abs))
    scale_exp = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale_u8 = (scale_exp + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    values = (
        (blocked / scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, width)
        .contiguous()
    )
    return values, scale_u8.contiguous()


def _reference_from_packed(source: torch.Tensor, packed_weight) -> torch.Tensor:
    rows, width = map(int, source.shape)
    padded_width = int(packed_weight.padded_in_features)
    if width != padded_width:
        padded = source.new_zeros((rows, padded_width))
        padded[:, :width] = source
        source = padded.contiguous()
    x_q = bfl.quantize_input(source)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    w_deq = dequantize_mxfp8_rows_torch(
        packed_weight.weight.values, packed_weight.weight.scale_rows
    )
    return x_deq @ w_deq.T


def _make_inputs(tokens: int, in_features: int, out_features: int):
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight_bf16 = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).contiguous()
    weight, weight_scale = _quantize_modelopt_mxfp8_rows(weight_bf16)
    packed = mxfp8_linear.pack_weight(weight, weight_scale)
    return source, weight_scale, packed


def test_mm_matches_quantized_reference_small_n() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260614)

    source, _, packed = _make_inputs(7, 128, 32)
    actual = mxfp8_linear.mm(source, packed)
    expected = _reference_from_packed(source, packed)
    torch.cuda.synchronize()

    assert actual.shape == (7, 32)
    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
    )


def test_mm_persistent_ctas_complete_single_stage_epilogue_stores() -> None:
    require_b12x()
    require_mxf8_mma()

    tokens, in_features, out_features = 1372, 128, 4096
    source_values = torch.ones(
        (tokens, in_features), device="cuda", dtype=torch.float8_e4m3fn
    )
    source_scale = torch.full(
        (tokens, in_features // 32), 127, device="cuda", dtype=torch.uint8
    )
    weight = torch.ones(
        (out_features, in_features), device="cuda", dtype=torch.float8_e4m3fn
    )
    weight_scale = torch.full(
        (out_features, in_features // 32),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    packed = mxfp8_linear.pack_weight(weight, weight_scale)

    for _ in range(4):
        actual = mxfp8_linear.mm(
            (source_values, source_scale), packed, expected_m=tokens
        )
        torch.cuda.synchronize()
        assert torch.all(actual == in_features)


@pytest.mark.parametrize("tokens", (2, 3, 8, 15, 16, 17, 32, 99))
def test_mm_writes_all_rows_for_unaligned_output_width(tokens: int) -> None:
    """A small-batch GEMM must store every live row when N spans multiple tiles."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260814 + tokens)

    source, _, packed = _make_inputs(tokens, 7168, 132)
    actual = mxfp8_linear.mm(source, packed, expected_m=tokens)
    expected = _reference_from_packed(source, packed)
    torch.cuda.synchronize()

    assert actual.shape == (tokens, 132)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-2,
    )


def test_mm_unaligned_output_stride_captures_and_replays() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260822)

    source, _, packed = _make_inputs(8, 7168, 132)
    replacement = torch.randn_like(source).div_(4)
    mxfp8_linear.mm(source, packed)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mxfp8_linear.mm(source, packed)
    source.copy_(replacement)
    expected = _reference_from_packed(source, packed)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-2,
    )


def test_mm_pads_k32_to_dense_tile() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260615)

    source, weight_scale, packed = _make_inputs(3, 160, 40)

    assert packed.in_features == 160
    assert packed.padded_in_features == 256
    assert packed.weight.values.shape == (40, 256)
    assert packed.weight.scale_rows.shape == (1, 40, 8)
    torch.testing.assert_close(
        packed.weight.scale_rows.view(torch.uint8)[0, :, :5], weight_scale
    )
    assert torch.all(packed.weight.scale_rows.view(torch.uint8)[0, :, 5:] == 127)

    actual = mxfp8_linear.mm(source, packed)
    expected = _reference_from_packed(source, packed)
    torch.cuda.synchronize()

    assert actual.shape == (3, 40)
    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
    )


def test_mm_default_fused_path_captures_with_k_padding() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260616)

    source, _, packed = _make_inputs(1, 160, 40)

    eager = mxfp8_linear.mm(source, packed).clone()
    torch.cuda.synchronize()

    mxfp8_linear.mm(source, packed)  # warm before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mxfp8_linear.mm(source, packed)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", (0, 4, 8, 9))
def test_mm_pair_matches_independent_projections(tokens: int) -> None:
    """Paired execution preserves both independently computed MXFP8 results."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260904 + tokens)

    source, _, primary = _make_inputs(tokens, 256, 384)
    _, _, secondary = _make_inputs(tokens, 256, 64)
    expected_primary = mxfp8_linear.mm(source, primary)
    expected_secondary = mxfp8_linear.mm(source, secondary)
    actual_primary, actual_secondary = mxfp8_linear.mm_pair(
        source,
        primary,
        secondary,
        parallel_max_tokens=8,
    )
    current_stream = torch.cuda.current_stream(source.device)
    observed_primary = actual_primary.clone()
    observed_secondary = actual_secondary.clone()
    current_stream.synchronize()

    torch.testing.assert_close(observed_primary, expected_primary, rtol=0, atol=0)
    torch.testing.assert_close(observed_secondary, expected_secondary, rtol=0, atol=0)


def test_mm_pair_joins_secondary_work_to_the_caller_stream() -> None:
    """A caller-stream read observes secondary output without a device sync."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260906)

    source, _, primary = _make_inputs(4, 256, 384)
    _, _, secondary = _make_inputs(4, 256, 64)
    expected_secondary = mxfp8_linear.mm(source, secondary)
    current_stream = torch.cuda.current_stream(source.device)
    secondary_stream = blockscaled_linear._paired_projection_stream(source.device)
    current_stream.synchronize()
    with torch.cuda.stream(secondary_stream):
        torch.cuda._sleep(20_000_000)

    _, actual_secondary = mxfp8_linear.mm_pair(
        source,
        primary,
        secondary,
        parallel_max_tokens=8,
    )
    observed_secondary = actual_secondary.clone()
    current_stream.synchronize()

    torch.testing.assert_close(observed_secondary, expected_secondary, rtol=0, atol=0)


def test_mm_pair_captures_both_streams_and_replays() -> None:
    """Graph replay waits for the secondary projection before returning outputs."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260905)

    source, _, primary = _make_inputs(4, 256, 384)
    _, _, secondary = _make_inputs(4, 256, 64)
    replacement = torch.randn_like(source).div_(4)
    mxfp8_linear.mm_pair(
        source,
        primary,
        secondary,
        parallel_max_tokens=8,
    )
    current_stream = torch.cuda.current_stream(source.device)
    current_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_primary, actual_secondary = mxfp8_linear.mm_pair(
            source,
            primary,
            secondary,
            parallel_max_tokens=8,
        )
    source.copy_(replacement)
    expected_primary = mxfp8_linear.mm(source, primary)
    expected_secondary = mxfp8_linear.mm(source, secondary)
    graph.replay()
    observed_primary = actual_primary.clone()
    observed_secondary = actual_secondary.clone()
    current_stream.synchronize()

    torch.testing.assert_close(observed_primary, expected_primary, rtol=0, atol=0)
    torch.testing.assert_close(observed_secondary, expected_secondary, rtol=0, atol=0)


def test_blockscaled_mm_accepts_prequantized_mxfp8_and_replays() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260823)

    tokens, in_features, out_features = 6, 128, 64
    source, _, packed = _make_inputs(tokens, in_features, out_features)
    source_q = bfl.quantize_input(source)
    source_scale_storage = convert_sf_from_mma_layout(
        source_q.scale_mma,
        m=tokens,
        k=in_features,
        num_groups=1,
        sf_vec_size=32,
    )
    expected = blockscaled.mm(source, packed)
    actual_native_scale = blockscaled.mm(
        (source_q.values, source_q.scale_mma),
        packed,
        out_dtype=torch.bfloat16,
    )
    actual_serialized_scale = blockscaled.mm(
        (source_q.values, source_scale_storage),
        packed,
        out_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual_native_scale, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_serialized_scale, expected, rtol=0, atol=0)

    blockscaled.prewarm(
        packed,
        [tokens],
        input_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = blockscaled.mm(
            (source_q.values, source_scale_storage),
            packed,
            out_dtype=torch.bfloat16,
        )
    output_ptr = graph_output.data_ptr()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert graph_output.data_ptr() == output_ptr
    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)


def test_blockscaled_mm_accepts_compact_mxfp8_scales_with_k_padding() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260824)

    source, _, packed = _make_inputs(6, 160, 40)
    source_values, source_scale_rows = _quantize_modelopt_mxfp8_rows(source)
    actual = blockscaled.mm(
        (source_values, source_scale_rows),
        packed,
        out_dtype=torch.bfloat16,
    )
    expected = blockscaled.mm(source, packed)

    assert actual.shape == (6, 40)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
