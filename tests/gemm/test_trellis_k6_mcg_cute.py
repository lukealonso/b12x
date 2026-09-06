"""Contracts for the fused SM120 unpaired K6/MCG dense linear."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import trellis_linear
from b12x.gemm.trellis_linear import _k6_mcg_cute
from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel


def _weight_descriptor(**overrides):
    descriptor = {
        "params_dtype": torch.float16,
        "weight_layout": "trellis_t256",
        "num_experts": 1,
        "trellis_bits": 6,
        "trellis_codebook": "mcg",
        "trellis_pair_kind": None,
        "trellis_rate_axis": None,
        "in_features": 2048,
        "out_features": 4096,
    }
    descriptor.update(overrides)
    return SimpleNamespace(**descriptor)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"params_dtype": torch.bfloat16}, True),
        ({"params_dtype": torch.float32}, False),
        ({"weight_layout": "trellis3_t256"}, False),
        ({"weight_layout": "mxfp4"}, False),
        ({"num_experts": 2}, False),
        ({"trellis_bits": 5}, False),
        ({"trellis_codebook": "sqg_e4m3"}, False),
        ({"trellis_pair_kind": "P24", "trellis_rate_axis": "n"}, False),
        ({"trellis_pair_kind": "P33", "trellis_rate_axis": "k"}, False),
        ({"trellis_pair_kind": "P24"}, False),
        ({"trellis_rate_axis": "n"}, False),
        ({"in_features": 2112}, False),
        ({"out_features": 4032}, False),
    ],
)
def test_unpaired_weight_gate_excludes_qsrt_pairs(
    overrides: dict[str, object], expected: bool
) -> None:
    assert (
        _k6_mcg_cute._is_unpaired_k6_mcg_weight(_weight_descriptor(**overrides))
        is expected
    )


@pytest.mark.parametrize(
    ("size_k", "size_n", "expected"),
    [
        (2048, 4096, 131072),
        (6144, 1024, 65536),
        (512, 6144, 98304),
        # Qwen3.8-27B K6 down projection uses the measured 160-CTA grid.
        (17408, 5120, 327680),
        (128, 128, 2048),
    ],
)
def test_scratch_contract_covers_split_k_owners(
    size_k: int, size_n: int, expected: int
) -> None:
    assert trellis_linear.k6_mcg_small_m_scratch_elements(size_k, size_n) == expected


@pytest.mark.parametrize(
    ("size_k", "size_n", "expected_grid_x"),
    [
        (5120, 6144, 120),
        (5120, 10240, 160),
        (5120, 12288, 188),
        (6144, 5120, 120),
        (17408, 5120, 160),
    ],
)
def test_qwen38_grid_policy_is_shape_specific(
    size_k: int, size_n: int, expected_grid_x: int
) -> None:
    assert _k6_mcg_cute._requested_grid_x(size_k, size_n) == expected_grid_x


@pytest.mark.parametrize("shape", [(0, 128), (128, 0), (192, 128), (128, 192)])
def test_scratch_contract_rejects_unaligned_shapes(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive multiples of 128"):
        trellis_linear.k6_mcg_small_m_scratch_elements(*shape)


def test_rotation_grid_and_gemm_grid_have_independent_capacity() -> None:
    launch = _k6_mcg_cute.K6McgSmallMCompileResult(
        compiled=None,
        device_index=0,
        size_k=6144,
        size_n=1024,
        params_dtype=torch.float16,
        grid_x=32,
        cta_threads=256,
        resident_ctas=376,
        blocks_per_sm=2,
        shared_memory_bytes=0,
        required_scratch_elements=65536,
        required_workspace_elements=1,
        trellis_lut=torch.empty(1, dtype=torch.uint8),
    )

    assert launch.launch_grid_x(1) == 32
    assert launch.launch_grid_x(4) == 32
    assert launch.launch_grid_x(8) == 49
    assert launch.launch_grid_x(16) == 97


def test_bound_launch_rejects_non_cuda_runtime_inputs() -> None:
    launch = _k6_mcg_cute.K6McgSmallMCompileResult(
        compiled=None,
        device_index=0,
        size_k=128,
        size_n=128,
        params_dtype=torch.float16,
        grid_x=1,
        cta_threads=256,
        resident_ctas=1,
        blocks_per_sm=1,
        shared_memory_bytes=0,
        required_scratch_elements=2048,
        required_workspace_elements=1,
        trellis_lut=torch.empty(1, dtype=torch.uint8),
    )

    assert not launch.accepts_input(torch.empty((1, 128), dtype=torch.float16))


def test_compile_rejects_non_cuda_device_before_jit() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        _k6_mcg_cute.compile_k6_mcg_small_m(
            size_k=128,
            size_n=128,
            params_dtype=torch.float16,
            device=torch.device("cpu"),
        )


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(props.major) == 12


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_planning_rejects_invalid_static_workspace_before_serving() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    size_k = size_n = 128
    trellis = torch.zeros(
        (size_k // 16, size_n // 16, 96), dtype=torch.int16, device=device
    )
    signs = torch.ones(size_k, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        signs,
        signs.clone(),
        codebook="mcg",
        params_dtype=torch.float16,
    )
    invalid = replace(
        weight,
        workspace=torch.empty(1, dtype=torch.int32, device=device),
        k6_mcg_small_m_launch=None,
    )

    with pytest.raises(ValueError, match="workspace"):
        _k6_mcg_cute.plan_k6_mcg_small_m(invalid)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_bound_fused_dispatch_skips_generic_static_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 1
    size_k = size_n = 128
    scratch_elements = trellis_linear.k6_mcg_small_m_scratch_elements(
        size_k, size_n
    )
    launch = _k6_mcg_cute.K6McgSmallMCompileResult(
        compiled=None,
        device_index=int(device.index),
        size_k=size_k,
        size_n=size_n,
        params_dtype=torch.float16,
        grid_x=1,
        cta_threads=256,
        resident_ctas=1,
        blocks_per_sm=1,
        shared_memory_bytes=0,
        required_scratch_elements=scratch_elements,
        required_workspace_elements=1,
        trellis_lut=torch.empty(1, dtype=torch.uint8, device=device),
    )
    # A bound launch is the immutable serving contract. Deliberately omit every
    # generic static weight attribute so any per-call revalidation fails here.
    prepared = SimpleNamespace(k6_mcg_small_m_launch=launch)
    source = torch.zeros((rows, size_k), dtype=torch.float16, device=device)
    output = torch.empty((rows, size_n), dtype=torch.float16, device=device)
    rotated = torch.empty_like(source)
    c_tmp = torch.empty(scratch_elements, dtype=torch.float32, device=device)
    calls: list[tuple[torch.Tensor, object]] = []

    def fake_run(
        actual_source,
        actual_prepared,
        *,
        output,
        rotated,
        c_tmp,
    ):
        calls.append((actual_source, actual_prepared))
        return output

    monkeypatch.setattr(_k6_mcg_cute, "run_k6_mcg_small_m", fake_run)

    actual = w4a16_kernel._run_trellis256_dense_current_device(
        source,
        prepared,
        output=output,
        rotated_compute=rotated,
        c_tmp=c_tmp,
    )

    assert actual is output
    assert len(calls) == 1
    assert calls[0][0] is source
    assert calls[0][1] is prepared


def _buffers(
    rows: int,
    size_k: int,
    size_n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    return {
        "output": torch.empty((rows, size_n), dtype=dtype, device=device),
        "rotated_f16": torch.empty(
            (rows, size_k), dtype=torch.float16, device=device
        ),
        "rotated_compute": torch.empty(
            (rows, size_k), dtype=dtype, device=device
        ),
        "c_tmp": torch.empty(
            (trellis_linear.k6_mcg_small_m_scratch_elements(size_k, size_n),),
            dtype=torch.float32,
            device=device,
        ),
    }


def _hadamard_128_reference(
    source: torch.Tensor,
    output: torch.Tensor,
    pre_scale: torch.Tensor | None,
    post_scale: torch.Tensor | None,
    scale: float,
) -> None:
    """Apply normalized Hadamard-128 from its mathematical definition."""
    values = source.float()
    if pre_scale is not None:
        values = values * pre_scale.float()

    blocks = values.reshape(*values.shape[:-1], -1, 128)
    stride = 1
    while stride < 128:
        stages = blocks.reshape(*blocks.shape[:-1], -1, 2, stride)
        lower = stages[..., 0, :]
        upper = stages[..., 1, :]
        blocks = torch.stack((lower + upper, lower - upper), dim=-2).flatten(-3, -1)
        stride *= 2
    values = blocks.reshape_as(values) * (float(scale) / (128.0**0.5))

    if post_scale is not None:
        values = values * post_scale.float()
    output.copy_(values.to(output.dtype))


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(
    ("rows", "size_k", "size_n", "dtype"),
    [
        (1, 128, 128, torch.float16),
        (4, 128, 128, torch.float16),
        (8, 128, 128, torch.float16),
        (11, 128, 128, torch.float16),
        (16, 128, 128, torch.float16),
        # Exercise split-K reduction and the multi-CTA input rotation on a
        # production GLM-5.2 projection geometry.
        (8, 6144, 1024, torch.float16),
        # Exercise the largest unpaired K6/MCG dense projection observed in
        # the Qwen3.8-27B container's b12x launch records.
        (1, 17408, 5120, torch.float16),
        # Native BF16 coverage is focused on one boundary case and the live
        # Qwen down-projection geometry rather than duplicating every FP16 row.
        (8, 128, 128, torch.bfloat16),
        (1, 17408, 5120, torch.bfloat16),
    ],
)
def test_fused_k6_mcg_matches_separate_rotation_pipeline(
    rows: int,
    size_k: int,
    size_n: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0x4B364D43 + rows)
    device = torch.device("cuda", torch.cuda.current_device())
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    suh = torch.where(
        torch.rand(size_k, device=device) > 0.5,
        torch.ones(size_k, device=device),
        -torch.ones(size_k, device=device),
    ).to(torch.float16)
    svh = torch.where(
        torch.rand(size_n, device=device) > 0.5,
        torch.ones(size_n, device=device),
        -torch.ones(size_n, device=device),
    ).to(torch.float16)
    weight = trellis_linear.prepare_weight(
        trellis,
        suh,
        svh,
        codebook="mcg",
        params_dtype=dtype,
    )
    launch = weight.k6_mcg_small_m_launch
    assert isinstance(launch, _k6_mcg_cute.K6McgSmallMCompileResult)
    if (size_k, size_n) == (6144, 1024):
        assert launch.grid_x > 1
        assert launch.launch_grid_x(rows) > launch.grid_x
    source = (torch.randn((rows, size_k), device=device) * 0.05).to(dtype)
    noncontiguous = torch.empty((rows, size_k * 2), dtype=dtype, device=device)[
        :, ::2
    ]
    assert not noncontiguous.is_contiguous()
    assert not _k6_mcg_cute.is_k6_mcg_small_m_eligible(noncontiguous, weight)

    reference_buffers = _buffers(rows, size_k, size_n, device, dtype)
    reference = trellis_linear.run(
        source,
        weight,
        hadamard_128=_hadamard_128_reference,
        **reference_buffers,
    ).clone()
    weight.workspace.zero_()
    fused_buffers = _buffers(rows, size_k, size_n, device, dtype)
    actual = trellis_linear.run(source, weight, **fused_buffers).clone()
    torch.cuda.synchronize(device)

    delta = actual.float() - reference.float()
    relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
        reference.float()
    ).clamp_min(1.0e-12)
    relative_limit = 2.0e-3 if dtype == torch.float16 else 1.0e-2
    assert float(relative_l2) <= relative_limit
    if dtype == torch.float16:
        assert float(delta.abs().max()) <= 1.0
    else:
        output_range = reference.float().abs().max().clamp_min(1.0e-12)
        assert float(delta.abs().max() / output_range) <= 2.0e-2


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(
    ("rows", "size_k", "size_n", "dtype"),
    [
        (8, 128, 128, torch.float16),
        (8, 128, 128, torch.bfloat16),
        # The live Qwen3.8-27B decode path captures this K6 down projection.
        (1, 17408, 5120, torch.float16),
        (1, 17408, 5120, torch.bfloat16),
    ],
)
def test_fused_k6_mcg_cuda_graph_replay_is_stable(
    rows: int,
    size_k: int,
    size_n: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0x4B364752)
    device = torch.device("cuda", torch.cuda.current_device())
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    suh = torch.ones(size_k, dtype=torch.float16, device=device)
    svh = torch.ones(size_n, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        suh,
        svh,
        codebook="mcg",
        params_dtype=dtype,
    )
    assert isinstance(
        weight.k6_mcg_small_m_launch, _k6_mcg_cute.K6McgSmallMCompileResult
    )
    launch = weight.k6_mcg_small_m_launch
    assert launch.trellis_lut.device == device
    assert launch.trellis_lut.dtype == torch.uint8
    assert launch.required_workspace_elements <= int(weight.workspace.numel())
    assert (
        launch.required_scratch_elements
        == trellis_linear.k6_mcg_small_m_scratch_elements(size_k, size_n)
    )
    source = (torch.randn((rows, size_k), device=device) * 0.05).to(dtype)
    buffers = _buffers(rows, size_k, size_n, device, dtype)
    _k6_mcg_cute.clear_k6_mcg_small_m_cache()
    freeze_kernel_resolution("bound K6/MCG launch must survive cache clearing")
    try:
        expected = trellis_linear.run(source, weight, **buffers).clone()
        torch.cuda.synchronize(device)

        def fail_late_lut_resolution(*_args, **_kwargs):
            pytest.fail("runtime launch must reuse the LUT bound during preparation")

        monkeypatch.setattr(
            _k6_mcg_cute,
            "sqg_xor_cheb_t12_lut",
            fail_late_lut_resolution,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = trellis_linear.run(source, weight, **buffers)
        for _ in range(5):
            buffers["output"].fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize(device)
    finally:
        unfreeze_kernel_resolution()

    assert torch.equal(captured, expected)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("missing", ["output", "rotated_compute", "c_tmp"])
def test_fused_k6_mcg_capture_requires_caller_owned_buffers(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 8
    size_k = size_n = 128
    trellis = torch.zeros(
        (size_k // 16, size_n // 16, 96), dtype=torch.int16, device=device
    )
    signs = torch.ones(size_k, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        signs,
        signs.clone(),
        codebook="mcg",
        params_dtype=torch.float16,
    )
    source = torch.zeros((rows, size_k), dtype=torch.float16, device=device)
    buffers = _buffers(rows, size_k, size_n, device)

    buffers.pop(missing)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        trellis_linear.run(source, weight, **buffers)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_fused_k6_mcg_declines_generic_sized_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 8
    size_k = size_n = 128
    trellis = torch.zeros(
        (size_k // 16, size_n // 16, 96), dtype=torch.int16, device=device
    )
    signs = torch.ones(size_k, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        signs,
        signs.clone(),
        codebook="mcg",
        params_dtype=torch.bfloat16,
    )
    source = torch.zeros((rows, size_k), dtype=torch.bfloat16, device=device)
    buffers = {
        "output": torch.empty_like(source),
        "gemm_output": torch.empty_like(source),
        "c_tmp": torch.empty((size_n * 64,), dtype=torch.float32, device=device),
        "input_f16": torch.empty_like(source, dtype=torch.float16),
        "rotated_f16": torch.empty_like(source, dtype=torch.float16),
        "rotated_compute": torch.empty_like(source),
        "gemm_output_f16": torch.empty_like(source, dtype=torch.float16),
        "output_f16": torch.empty_like(source, dtype=torch.float16),
    }

    weight = replace(
        weight,
        k6_mcg_small_m_launch=replace(
            weight.k6_mcg_small_m_launch,
            required_scratch_elements=int(buffers["c_tmp"].numel()) + 1,
        ),
    )

    def fail_fused(*_args, **_kwargs):
        pytest.fail("undersized fused scratch must decline the fused route")

    monkeypatch.setattr(_k6_mcg_cute, "run_k6_mcg_small_m", fail_fused)
    monkeypatch.setattr(
        w4a16_kernel,
        "_resolve_exl3_hadamard_128",
        lambda _callback: _hadamard_128_reference,
    )

    actual = trellis_linear.run(source, weight, **buffers)
    torch.cuda.synchronize(device)

    assert actual is buffers["output"]
    assert torch.isfinite(actual).all()
