"""gemm.blockscaled: NVFP4/MXFP8 dense block-scaled GEMM.

Curated from b12x tests/test_gemm_stack.py: flashinfer-core cuDNN oracle
parity (NVFP4), grouped MXFP8 per-batch scale correctness, and CUDA-graph
replay. Exhaustive tile/support-matrix sweeps stay in the b12x repo.
"""

from __future__ import annotations

import pytest
import torch

from b12x._lib import dense_gemm as dense_module
from b12x._lib.intrinsics import (
    fp4_quantize_values_torch,
    quantize_grouped_nvfp4_torch,
    swizzle_block_scale,
)
from b12x._lib.utils import convert_sf_from_mma_layout
from b12x.gemm import blockscaled
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    pack_fp8_block_scaled_weight_mxfp8,
    quantize_mxfp8_rows_torch,
)

from ..conftest import require_b12x


def _quantize_mxfp4_rows(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent row-major MXFP4 quantizer for dense correctness tests."""

    rows, width = map(int, source.shape)
    blocked = source.float().view(rows, width // 32, 32)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    safe_scale = torch.where(
        block_max > 0,
        block_max / 6.0,
        torch.ones_like(block_max),
    )
    exponent = torch.ceil(torch.log2(safe_scale)).clamp(-127, 127)
    scale_byte = torch.where(
        block_max > 0,
        exponent + 127,
        torch.zeros_like(exponent),
    ).to(torch.uint8)
    scale = torch.where(
        block_max > 0,
        torch.exp2(exponent),
        torch.zeros_like(exponent),
    )
    values = fp4_quantize_values_torch(
        torch.where(
            scale > 0,
            blocked / scale.clamp_min(1e-30),
            torch.zeros_like(blocked),
        ).view(rows, width)
    )
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    nibbles = torch.zeros_like(values, dtype=torch.uint8)
    for code, magnitude in enumerate(magnitudes):
        nibbles = torch.where(
            values.abs() == magnitude,
            torch.full_like(nibbles, code),
            nibbles,
        )
    nibbles |= (values < 0).to(torch.uint8) << 3
    pairs = nibbles.view(rows, width // 2, 2)
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()
    return packed, scale_byte.squeeze(-1).contiguous()


def _dequantize_mxfp4_rows(
    packed: torch.Tensor,
    scale_byte: torch.Tensor,
) -> torch.Tensor:
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=packed.device,
    )
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(1)
    values = magnitudes[(nibbles & 0x7).long()]
    values *= torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    scale = torch.where(
        scale_byte == 0,
        0.0,
        torch.exp2(scale_byte.float() - 127.0),
    ).repeat_interleave(32, dim=1)
    return values * scale


def _require_cudnn_fp4_oracle():
    """flashinfer core mm_fp4 (cuDNN backend) as the correctness oracle.

    Tests may import both core and experimental; only package code is bound
    by the isolation rule.
    """
    try:
        from flashinfer.gemm import mm_fp4
        from flashinfer.gemm.gemm_base import (
            CUDNN_AVAILABLE,
            _check_cudnn_fp4_availability,
        )
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"flashinfer core mm_fp4 oracle unavailable: {exc}")
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN Python bindings not installed")
    try:
        _check_cudnn_fp4_availability()
    except RuntimeError as exc:
        pytest.skip(f"cuDNN FP4 not available: {exc}")
    return mm_fp4


def test_serialized_mm_keeps_planning_opaque_to_dynamo() -> None:
    lhs_values = torch.empty((6, 64), dtype=torch.uint8)
    lhs_scale = torch.empty((128, 4), dtype=torch.uint8)
    rhs_values = torch.empty((48, 64), dtype=torch.uint8)
    rhs_scale = torch.empty((128, 4), dtype=torch.uint8)

    def run(lhs_values, lhs_scale, rhs_values, rhs_scale):
        return blockscaled.mm(
            (lhs_values, lhs_scale),
            (rhs_values, rhs_scale),
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
        )

    graph, _ = torch._dynamo.export(run)(
        lhs_values,
        lhs_scale,
        rhs_values,
        rhs_scale,
    )
    targets = {node.target for node in graph.graph.nodes if node.op == "call_function"}

    assert torch.ops.b12x.blockscaled_serialized in targets
    assert torch.ops.b12x.dense_gemm_launch not in targets


def test_recipe_wrappers_keep_planning_opaque_to_dynamo() -> None:
    lhs_values = torch.empty((6, 64), dtype=torch.uint8)
    lhs_scale = torch.empty((128, 4), dtype=torch.uint8)
    rhs_values = torch.empty((48, 64), dtype=torch.uint8)
    rhs_scale = torch.empty((128, 4), dtype=torch.uint8)
    alpha = torch.ones(1, dtype=torch.float32)

    def run_mxfp4(lhs_values, lhs_scale, rhs_values, rhs_scale):
        return blockscaled.mm_mxfp4(
            lhs_values,
            lhs_scale,
            rhs_values,
            rhs_scale,
        )

    def run_nvfp4(lhs_values, lhs_scale, rhs_values, rhs_scale, alpha):
        return blockscaled.mm_nvfp4(
            lhs_values,
            lhs_scale,
            rhs_values,
            rhs_scale,
            alpha,
        )

    for run, args in (
        (run_mxfp4, (lhs_values, lhs_scale, rhs_values, rhs_scale)),
        (run_nvfp4, (lhs_values, lhs_scale, rhs_values, rhs_scale, alpha)),
    ):
        graph, _ = torch._dynamo.export(run)(*args)
        targets = {
            node.target for node in graph.graph.nodes if node.op == "call_function"
        }
        assert torch.ops.b12x.blockscaled_serialized in targets
        assert torch.ops.b12x.dense_gemm_launch not in targets


def _make_quantized_operand(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    source = torch.randn(shape, device="cuda", dtype=dtype) / 4
    row_counts = torch.full(
        (shape[0],), shape[1], dtype=torch.int32, device=source.device
    )
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return (packed, scales), global_scale


def _mm_nvfp4(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    *,
    c_dtype: str = "bfloat16",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)
    return blockscaled.mm(
        lhs,
        rhs,
        out=out,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=c_dtype,
        sf_vec_size=16,
    )


@pytest.mark.parametrize(
    ("m", "n", "k", "c_dtype"),
    [
        (128, 128, 128, "bfloat16"),
        (256, 512, 128, "bfloat16"),
        (512, 256, 256, "bfloat16"),
        (128, 128, 128, "float16"),
    ],
)
def test_mm_nvfp4_matches_flashinfer_cudnn(m, n, k, c_dtype) -> None:
    require_b12x()
    mm_fp4 = _require_cudnn_fp4_oracle()
    torch.manual_seed(42)

    lhs, lhs_scale = _make_quantized_operand((1, m, k), dtype=torch.bfloat16)
    rhs, rhs_scale = _make_quantized_operand((1, n, k), dtype=torch.bfloat16)
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)

    actual = _mm_nvfp4(lhs, rhs, lhs_scale, rhs_scale, c_dtype=c_dtype)

    packed_a, sfa = lhs
    packed_b, sfb = rhs
    oracle = mm_fp4(
        packed_a[:, :, 0].contiguous(),
        packed_b[:, :, 0].contiguous().T,
        convert_sf_from_mma_layout(sfa, m=m, k=k, num_groups=1),
        convert_sf_from_mma_layout(sfb, m=n, k=k, num_groups=1).T,
        alpha,
        torch.bfloat16 if c_dtype == "bfloat16" else torch.float16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cudnn",
        use_nvfp4=True,
    )

    torch.testing.assert_close(actual[:, :, 0], oracle, rtol=0, atol=0)


def test_mm_mxfp8_grouped_batches_use_their_own_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    torch.manual_seed(29)

    # Real grouped WO-A geometry; force the shape-gated BK64 specialization so
    # this compact test covers its packed-scale address arithmetic for L>1.
    m, n, k = 64, 1024, 512
    groups = 4
    group_multipliers = torch.tensor(
        [1.0, 2.0, 4.0, 0.5], device="cuda", dtype=torch.bfloat16
    ).view(1, 1, groups)
    a = torch.randn((m, k, groups), device="cuda", dtype=torch.bfloat16) / 4
    a_q = quantize_mxfp8_rows_torch(a * group_multipliers)
    b_values = (
        torch.randn((groups * n, k), device="cuda", dtype=torch.bfloat16) / 32
    ).to(torch.float8_e4m3fn)
    b_scales = (
        torch.tensor([1.0, 2.0, 4.0, 0.5], device="cuda", dtype=torch.float32)
        .view(groups, 1, 1)
        .expand(groups, n // 128, k // 128)
        .reshape(groups * (n // 128), k // 128)
        .contiguous()
    )
    b_q = pack_fp8_block_scaled_weight_mxfp8(
        b_values, b_scales, m=n, k=k, num_groups=groups
    )
    assert not torch.equal(a_q.scale_rows[0], a_q.scale_rows[1])
    assert not torch.equal(b_q.scale_rows[0], b_q.scale_rows[1])

    monkeypatch.setattr(dense_module, "_select_mxfp8_tile_k", lambda *_: 64)
    out = blockscaled.mm(
        (a_q.values, a_q.scale_mma),
        (b_q.values, b_q.scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        mma_tiler_mn=(128, 128),
        expected_m=2048,
        sfb_k_replicated=True,
    )
    a_deq = dequantize_mxfp8_rows_torch(a_q.values, a_q.scale_rows).to(torch.bfloat16)
    b_deq = dequantize_mxfp8_rows_torch(b_q.values, b_q.scale_rows).to(torch.bfloat16)
    ref = torch.einsum("mkl,nkl->mnl", a_deq, b_deq).to(torch.bfloat16)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_mm_pair_replays_under_cuda_graph() -> None:
    require_b12x()
    torch.manual_seed(1234)

    gate_m, gate_n, gate_k = 32, 2048, 512
    down_m, down_n, down_k = 32, 1024, 2048

    gate_lhs, gate_ls = _make_quantized_operand(
        (1, gate_m, gate_k), dtype=torch.bfloat16
    )
    gate_rhs, gate_rs = _make_quantized_operand(
        (1, gate_n, gate_k), dtype=torch.bfloat16
    )
    down_lhs, down_ls = _make_quantized_operand(
        (1, down_m, down_k), dtype=torch.bfloat16
    )
    down_rhs, down_rs = _make_quantized_operand(
        (1, down_n, down_k), dtype=torch.bfloat16
    )

    eager_gate = _mm_nvfp4(gate_lhs, gate_rhs, gate_ls, gate_rs)
    eager_down = _mm_nvfp4(down_lhs, down_rhs, down_ls, down_rs)
    torch.cuda.synchronize()

    graph_gate = torch.empty_like(eager_gate)
    graph_down = torch.empty_like(eager_down)

    # Prime compiled kernels before capture, matching the serving warmup path.
    _mm_nvfp4(gate_lhs, gate_rhs, gate_ls, gate_rs)
    _mm_nvfp4(down_lhs, down_rhs, down_ls, down_rs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _mm_nvfp4(gate_lhs, gate_rhs, gate_ls, gate_rs, out=graph_gate)
        _mm_nvfp4(down_lhs, down_rhs, down_ls, down_rs, out=graph_down)

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_gate, eager_gate, rtol=0, atol=0)
    torch.testing.assert_close(graph_down, eager_down, rtol=0, atol=0)


def test_mm_serialized_mxfp4_matches_independent_dequantized_reference() -> None:
    """Non-unit E8M0 scales catch scale-fragment ordering regressions."""

    require_b12x()
    torch.manual_seed(20260823)
    m, n, k = 6, 128, 256
    lhs_source = torch.randn((m, k), device="cuda") * 0.3
    rhs_source = torch.randn((n, k), device="cuda") * 0.3
    lhs_values, lhs_scale_rows = _quantize_mxfp4_rows(lhs_source)
    rhs_values, rhs_scale_rows = _quantize_mxfp4_rows(rhs_source)
    lhs_scale_storage = swizzle_block_scale(lhs_scale_rows)
    rhs_scale_storage = swizzle_block_scale(rhs_scale_rows)

    actual = blockscaled.mm(
        (lhs_values, lhs_scale_storage),
        (rhs_values, rhs_scale_storage),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=m,
    )
    compatibility = blockscaled.mm_mxfp4(
        lhs_values,
        lhs_scale_storage,
        rhs_values,
        rhs_scale_storage,
        expected_m=m,
    )
    lhs_dequant = _dequantize_mxfp4_rows(lhs_values, lhs_scale_rows)
    rhs_dequant = _dequantize_mxfp4_rows(rhs_values, rhs_scale_rows)
    expected = (lhs_dequant.to(torch.bfloat16) @ rhs_dequant.to(torch.bfloat16).T).to(
        torch.bfloat16
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(compatibility, actual, rtol=0, atol=0)

    # Serving qualification: compile first, then capture/replay without a
    # workspace or output-address change.
    blockscaled.prewarm(
        (rhs_values, rhs_scale_storage),
        [m],
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = blockscaled.mm_mxfp4(
            lhs_values,
            lhs_scale_storage,
            rhs_values,
            rhs_scale_storage,
            expected_m=m,
        )
    output_ptr = graph_output.data_ptr()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert graph_output.data_ptr() == output_ptr
    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)


def test_mm_serialized_nvfp4_and_block_fp8_match_native_views() -> None:
    require_b12x()
    torch.manual_seed(20260824)
    m, n, k = 6, 128, 256

    lhs, lhs_global_scale = _make_quantized_operand((1, m, k), dtype=torch.bfloat16)
    rhs, rhs_global_scale = _make_quantized_operand((1, n, k), dtype=torch.bfloat16)
    alpha = (1.0 / (lhs_global_scale[0] * rhs_global_scale[0])).view(1)
    native_nvfp4 = _mm_nvfp4(lhs, rhs, lhs_global_scale, rhs_global_scale)[:, :, 0]
    serialized_nvfp4 = blockscaled.mm(
        (
            lhs[0][:, :, 0],
            convert_sf_from_mma_layout(lhs[1], m=m, k=k, num_groups=1),
        ),
        (
            rhs[0][:, :, 0],
            convert_sf_from_mma_layout(rhs[1], m=n, k=k, num_groups=1),
        ),
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        expected_m=m,
    )
    torch.testing.assert_close(serialized_nvfp4, native_nvfp4, rtol=0, atol=0)
    compatibility_nvfp4 = blockscaled.mm_nvfp4(
        lhs[0][:, :, 0],
        convert_sf_from_mma_layout(lhs[1], m=m, k=k, num_groups=1),
        rhs[0][:, :, 0],
        convert_sf_from_mma_layout(rhs[1], m=n, k=k, num_groups=1),
        alpha,
        expected_m=m,
    )
    torch.testing.assert_close(compatibility_nvfp4, serialized_nvfp4, rtol=0, atol=0)

    lhs_fp8 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    rhs_fp8 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    lhs_scale = torch.rand((m, k // 128), device="cuda", dtype=torch.float32)
    rhs_scale = torch.rand((n // 128, k // 128), device="cuda", dtype=torch.float32)
    native_block_fp8 = blockscaled.mm(
        (lhs_fp8.unsqueeze(-1), lhs_scale),
        (rhs_fp8.unsqueeze(-1), rhs_scale),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype="bfloat16",
        sf_vec_size=128,
        block_fp8=True,
        expected_m=m,
    )[:, :, 0]
    serialized_block_fp8 = blockscaled.mm(
        (lhs_fp8, lhs_scale),
        (rhs_fp8, rhs_scale),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype="bfloat16",
        sf_vec_size=128,
        block_fp8=True,
        expected_m=m,
    )
    torch.testing.assert_close(serialized_block_fp8, native_block_fp8, rtol=0, atol=0)

    compatibility_block_fp8 = blockscaled.mm_block_fp8(
        lhs_fp8,
        lhs_scale,
        rhs_fp8,
        rhs_scale,
        out_dtype=torch.bfloat16,
        expected_m=m,
    )
    torch.testing.assert_close(
        compatibility_block_fp8, native_block_fp8, rtol=0, atol=0
    )


def test_blockscaled_public_surface_and_compatibility_aliases() -> None:
    from b12x.gemm import mxfp8_linear, tensor_fp8_linear

    assert blockscaled.META.entry_points == (
        "Weight",
        "NVFP4LinearWeight",
        "mm",
        "mm_mxfp4",
        "mm_nvfp4",
        "mm_block_fp8",
        "pack_weight",
        "prewarm",
        "is_supported",
        "w4a16",
        "w8a16",
        "workspace_size",
    )
    assert mxfp8_linear.mm is blockscaled.mm
    assert mxfp8_linear.pack_weight is blockscaled.pack_weight
    assert tensor_fp8_linear.mm is blockscaled.mm
    assert tensor_fp8_linear.pack_weight is blockscaled.pack_weight
    assert tensor_fp8_linear.prewarm is blockscaled.prewarm
    assert not hasattr(blockscaled, "mm_fused_quant_a")
    assert not hasattr(blockscaled, "mm_fused_quant_a_grouped")
