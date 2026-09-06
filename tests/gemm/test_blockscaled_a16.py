"""Shared-storage A16 correctness, one-shot dispatch, and graph contracts."""

from __future__ import annotations

import pytest
import torch
import cutlass.cute as cute

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _a16, _quantize
from b12x._lib.intrinsics import swizzle_block_scale
from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
from tests._reference.helpers import require_b12x


def test_a16_functional_and_out_export_are_opaque():
    from b12x.gemm.blockscaled import _ops  # noqa: F401
    source = torch.empty(8, 128, dtype=torch.bfloat16)
    values = torch.empty(64, 64, dtype=torch.uint8)
    scales = torch.empty(128, 8, dtype=torch.uint8)
    global_scale = torch.ones(1)
    output = torch.empty(8, 64, dtype=torch.bfloat16)
    workspace = torch.empty(4096, dtype=torch.uint8)

    def functional(x, w, s, g):
        return blockscaled.w4a16(x, w, s, g)

    def allocated(x, w, s, g, out, scratch):
        return blockscaled.w4a16(x, w, s, g, out=out, workspace=scratch)

    for fn, args, target in [
        (functional, (source, values, scales, global_scale), torch.ops.b12x.blockscaled_bf16.default),
        (allocated, (source, values, scales, global_scale, output, workspace), torch.ops.b12x.blockscaled_bf16_out.default),
    ]:
        graph, _ = torch._dynamo.export(fn)(*args)
        targets = {node.target for node in graph.graph.nodes if node.op == "call_function"}
        assert target in targets


@pytest.mark.parametrize("n,k,group", [(37, 80, 16), (129, 160, 32)])
def test_scale_storage_views_are_zero_copy(n, k, group):
    from b12x._lib.intrinsics import as_grouped_scale_view, as_grouped_scale_view_mx
    physical = torch.empty((n + 127) // 128 * 128, (k // group + 3) // 4 * 4, dtype=torch.uint8)
    view = (as_grouped_scale_view if group == 16 else as_grouped_scale_view_mx)(physical[None], n, k)
    flat = _a16.scale_storage(view, n, k, group)
    assert flat.data_ptr() == physical.data_ptr()
    with pytest.raises(ValueError, match="contiguous"):
        _a16.scale_storage(physical.T, n, k, group)


def make_weight(recipe, n, k, device="cuda"):
    if recipe == "nvfp4":
        codes = torch.arange(n * k, device=device).reshape(n, k) % 16
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        scales = (torch.rand(n, k // 16, device=device) * 2 + 0.0625).to(torch.float8_e4m3fn)
        storage = swizzle_block_scale(scales)
        global_scale = torch.tensor([0.125], dtype=torch.float32, device=device)
        weight = blockscaled.pack_weight(packed, storage, recipe="nvfp4", global_scale=global_scale)
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=device)
        local = lut[codes] * scales.float().repeat_interleave(16, 1)
        return weight, local.to(torch.bfloat16).float() * global_scale, storage
    values = (torch.randn(n, k, device=device) * 2).to(torch.float8_e4m3fn)
    exponent = torch.randint(120, 133, (n, k // 32), device=device, dtype=torch.uint8)
    weight = blockscaled.pack_weight(values, exponent)
    local = values.float() * torch.exp2(exponent.float() - 127).repeat_interleave(32, 1)
    return weight, local.to(torch.bfloat16).float(), weight.weight.scale_mma


def assert_close(actual, expected):
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    assert float(error / denominator) < 0.004
    torch.testing.assert_close(actual.float(), expected.float(), atol=float(expected.abs().max()) * 0.008 + 1e-6, rtol=0.008)


@pytest.mark.parametrize("allocation", ["system", "pinned", "pinned_wc", "registered", "managed", "file"])
@pytest.mark.parametrize("recipe,m", [("nvfp4", 1), ("nvfp4", 16), ("mxfp8", 1), ("mxfp8", 16)])
def test_shared_checkpoint_storage_matches_cuda_weights_in_graphs(tmp_path, allocation, recipe, m):
    """Both ordinary-load and TMA paths must consume the owned shared weights."""
    require_b12x()
    from b12x.loader import capabilities
    from benchmarks.loader._utils import WeightFiles
    caps = capabilities()
    if allocation in ("system", "file") and not caps["host_page_tables"]:
        pytest.skip("requires GPU host page tables")
    if allocation == "registered" and not caps["host_register_supported"]:
        pytest.skip("requires host registration")
    if allocation == "managed" and not caps["concurrent_managed_access"]:
        pytest.skip("requires concurrent managed access")
    weight, decoded, _ = make_weight(recipe, 128, 256)
    loaded = WeightFiles(tmp_path, allocation).load(weight)
    source = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(m, 128, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(blockscaled.workspace_size(weight, m), device="cuda", dtype=torch.uint8)
    options = {"activation_global_scale": torch.tensor([128.], device="cuda")} if recipe == "nvfp4" else {}
    for mode in ("a16", "quantized"):
        expected = blockscaled.mm(source, weight, mode=mode, **options)
        if mode == "a16":
            assert_close(expected, source.float() @ decoded.T)
        def run():
            blockscaled.mm(source, loaded, mode=mode, out=output, workspace=workspace, **options)
        if recipe == "nvfp4" and mode == "quantized" and allocation in ("system", "file"):
            # Triton's launcher rejects the unregistered global-scale pointer.
            with pytest.raises(ValueError, match="Pointer argument cannot be accessed"):
                run()
            continue
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            output.fill_(float("nan"))
            graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("mode", ["auto", "a16", "quantized"])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_aot_functionalization_preserves_precision_and_output(recipe, mode, use_out, backend, monkeypatch):
    require_b12x()
    import torch._inductor.config as inductor_config
    from b12x.gemm.blockscaled import _ops

    torch._dynamo.reset()
    weight, _, storage = make_weight(recipe, 128, 256)
    saved_scales = storage.view(torch.uint8).clone()
    execute = _ops._execute_impl

    def check_storage(source, values, scales, *args):
        assert scales.data_ptr() == storage.data_ptr()
        if recipe == "mxfp8":
            assert scales.dtype == torch.uint8
            assert scales.shape == storage.shape
            assert scales.stride() == storage.stride()
        return execute(source, values, scales, *args)

    monkeypatch.setattr(_ops, "_execute_impl", check_storage)
    workspace = torch.empty(blockscaled.workspace_size(weight, 8), device="cuda", dtype=torch.uint8)
    options = dict(activation_global_scale=torch.tensor([128.], device="cuda")) if recipe == "nvfp4" else {}

    def run(source, output):
        return blockscaled.mm(source, weight, mode=mode, expected_m=source.shape[0],
                              out=output if use_out else None,
                              workspace=workspace if use_out else None, **options)

    with inductor_config.patch(enable_auto_functionalized_v2=False):
        compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
        for m in (2, 8):
            source = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
            output = torch.full((m, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
            expected = blockscaled.mm(source, weight, mode=mode, **options)
            actual = compiled(source, output)
            if use_out:
                assert actual is output
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(storage.view(torch.uint8), saved_scales, atol=0, rtol=0)
    torch._dynamo.reset()


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("allocated", [False, True])
@pytest.mark.parametrize("functionalize_v2", [False, True])
def test_a16_aot_compile_functionalizes_workspace(recipe, allocated, functionalize_v2):
    require_b12x()
    weight, decoded, _ = make_weight(recipe, 128, 256)
    source = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty(blockscaled.workspace_size(weight, 4, _config=(64, 128, 2)),
                          device="cuda", dtype=torch.uint8)
    out = torch.empty(4, 128, device="cuda", dtype=torch.bfloat16) if allocated else None

    def project(x):
        return blockscaled.mm(x, weight, mode="a16", out=out,
                              workspace=scratch, _config=(64, 128, 2))

    project(source)
    compiled = torch.compile(project, fullgraph=True, options={
        "enable_auto_functionalized_v2": functionalize_v2,
    }).aot_compile(((source,), {}))
    for _ in range(2):
        source.normal_()
        assert_close(compiled(source), source.float() @ decoded.T)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("m,n,k", [(1, 40, 128), (4, 128, 256), (8, 136, 256), (16, 64, 128), (19, 40, 160), (33, 4096, 128)])
@pytest.mark.parametrize("split", [1, 2, 4, 8])
def test_a16_reference(recipe, m, n, k, split):
    require_b12x()
    weight, decoded, _ = make_weight(recipe, n, k)
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    output = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    scratch = torch.full((blockscaled.workspace_size(weight, m, _config=(64, 64, split)),), 255,
                         device="cuda", dtype=torch.uint8)
    actual = blockscaled.mm(source, weight, mode="a16", out=output,
                            workspace=scratch, _config=(64, 64, split))
    assert actual.data_ptr() == output.data_ptr()
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
    assert_close(actual, source.float() @ decoded.T)


@pytest.mark.parametrize("kind", ["multiplier", "reciprocal"])
def test_w4a16_raw_scale_identity_and_rounding(kind):
    require_b12x()
    n, k = 136, 96
    weight, _, storage = make_weight("nvfp4", n, k)
    original = storage.view(torch.uint8).clone()
    source = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    global_scale = weight.global_scale if kind == "multiplier" else weight.global_scale.reciprocal()
    actual = blockscaled.w4a16(source, weight.values, storage, global_scale,
                               global_scale_kind=kind)
    expected = blockscaled.mm(source, weight, mode="a16")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert _a16.scale_storage(weight.scale_mma, n, k, 16).data_ptr() == storage.data_ptr()
    torch.testing.assert_close(storage.view(torch.uint8), original, atol=0, rtol=0)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
def test_a16_frozen_callable_and_graph_replay(recipe, monkeypatch):
    require_b12x()
    weight, decoded, storage = make_weight(recipe, 136, 256)
    saved = storage.view(torch.uint8).clone()
    config = (64, 128, 4)
    workspace = torch.empty(blockscaled.workspace_size(weight, 32, _config=config), device="cuda", dtype=torch.uint8)
    source = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(32, 136, device="cuda", dtype=torch.bfloat16)
    blockscaled.mm(source[:1], weight, out=output[:1], workspace=workspace, mode="a16", _config=config)
    from b12x._lib.dense_gemm import _get_compiled_dense_gemm
    compiled = _get_compiled_dense_gemm.cache_info()
    freeze_kernel_resolution("A16 live-M reuse test")
    try:
        for m in [2, 4, 8, 16, 19, 32]:
            result = blockscaled.mm(source[:m], weight, out=output[:m], workspace=workspace,
                                    mode="a16", _config=config)
            assert_close(result, source[:m].float() @ decoded.T)
        assert compiled.misses == _get_compiled_dense_gemm.cache_info().misses
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            blockscaled.mm(source, weight, out=output, workspace=workspace,
                            mode="a16", _config=config)
        pointers = (source.data_ptr(), output.data_ptr(), workspace.data_ptr())
        for _ in range(3):
            source.normal_()
            workspace.fill_(255)
            output.fill_(float("nan"))
            graph.replay()
            assert_close(output, source.float() @ decoded.T)
        assert pointers == (source.data_ptr(), output.data_ptr(), workspace.data_ptr())
        torch.testing.assert_close(storage.view(torch.uint8), saved, atol=0, rtol=0)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("unexpected device allocation"))
            blockscaled.mm(source, weight, out=output, workspace=workspace,
                            mode="a16", _config=config)
    finally:
        unfreeze_kernel_resolution()


def test_quantized_nvfp4_activation_contract():
    require_b12x()
    from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
    from b12x._lib.dense_gemm import dense_gemm
    weight, _, _ = make_weight("nvfp4", 128, 256)
    source = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    ag = torch.tensor([256.], device="cuda")
    q, sf = quantize_grouped_nvfp4_torch(source[None], torch.tensor([8], device="cuda"), ag)
    expected = dense_gemm((q, sf),
                          (weight.values[:, :, None], weight.scale_mma),
                          ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
                          c_dtype="bfloat16", sf_vec_size=16,
                          alpha=weight.global_scale / ag)[:, :, 0]
    scratch = torch.full((blockscaled.workspace_size(weight, 16),), 255, device="cuda", dtype=torch.uint8)
    actual = blockscaled.mm(source, weight, mode="quantized", activation_global_scale=ag, workspace=scratch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    _, _, _, total = _a16._layout(8, 128, 256, True)
    sf_start, alpha_start, _, _ = _a16._layout(8, 128, 256, True)
    assert total <= scratch.numel()
    expected_bytes = _a16.scale_storage(sf, 8, 256, 16)
    torch.testing.assert_close(scratch[sf_start:sf_start + expected_bytes.numel()], expected_bytes, atol=0, rtol=0)
    torch.testing.assert_close(scratch[:8 * 128], q.flatten(), atol=0, rtol=0)
    torch.testing.assert_close(scratch[alpha_start:alpha_start + 4].view(torch.float32), weight.global_scale / ag)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
def test_quantized_graph_and_dispatch(recipe):
    require_b12x()
    weight, _, _ = make_weight(recipe, 128, 256)
    source = torch.randn(9, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(9, 128, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(blockscaled.workspace_size(weight, 16), device="cuda", dtype=torch.uint8)
    options = {"activation_global_scale": torch.tensor([128.], device="cuda")} if recipe == "nvfp4" else {}
    blockscaled.prewarm(weight, [1, 2, 8, 9, 16], workspace=workspace, **options)
    expected = blockscaled.mm(source, weight, mode="quantized", **options)
    actual = blockscaled.mm(source, weight, mode="auto", **options)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    freeze_kernel_resolution()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            blockscaled.mm(source, weight, mode="auto", out=output, workspace=workspace, **options)
        source.normal_()
        expected = blockscaled.mm(source, weight, mode="quantized", **options)
        workspace.fill_(255)
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
    finally:
        unfreeze_kernel_resolution()


def test_invalid_mode_and_aliasing():
    require_b12x()
    weight, _, _ = make_weight("nvfp4", 128, 128)
    source = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="mode"):
        blockscaled.mm(source, weight, mode="typo")
    with pytest.raises(ValueError, match="activation_global_scale"):
        blockscaled.mm(source, weight, mode="quantized")
    with pytest.raises(ValueError, match="overlap"):
        blockscaled.mm(source, weight, out=source, mode="a16")
    with pytest.raises(ValueError, match="workspace requires"):
        blockscaled.mm(source, weight, mode="a16", workspace=torch.empty(1, dtype=torch.uint8, device="cuda"),
                       _config=(64, 64, 4))


def test_mxfp8_prewarm_covers_functional_and_out_under_frozen_resolution():
    require_b12x()
    weight, _, _ = make_weight("mxfp8", 152, 384)
    source = torch.randn(7, 384, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(7, 152, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(blockscaled.workspace_size(weight, 7), device="cuda", dtype=torch.uint8)
    blockscaled.prewarm(weight, [7], workspace=workspace)
    freeze_kernel_resolution("blockscaled prewarm covers both output forms")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            functional = blockscaled.mm(source, weight)
            blockscaled.mm(source, weight, out=output, workspace=workspace)
        graph.replay()
        torch.testing.assert_close(functional, output, atol=0, rtol=0)
    finally:
        unfreeze_kernel_resolution()


def test_heuristic_promotes_inside_bf16_api_before_activation_quantization(monkeypatch):
    require_b12x()
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the geometry promotion heuristic is qualified on SM120")
    from b12x.gemm.blockscaled import _policy
    from b12x.policy import PolicyContext, PolicyMode, PolicySource

    policy = PolicyContext.for_device("cuda", mode=PolicyMode.HEURISTIC_ONLY)
    monkeypatch.setattr(_policy, "get_auto_policy", lambda device: policy)
    _policy.resolve_precision.cache_clear()
    weight, decoded, _ = make_weight("nvfp4", 4096, 5376)
    source = torch.randn(8, 5376, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(8, 4096, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(blockscaled.workspace_size(weight, 8), device="cuda", dtype=torch.uint8)
    blockscaled.prewarm(weight, [1], workspace=workspace)
    resolution = _policy.resolve_precision(source.device, "nvfp4", 5376, 4096)
    assert resolution.source is PolicySource.HEURISTIC
    monkeypatch.setattr(_quantize, "launch", lambda *args, **kwargs: pytest.fail("A16 quantized an activation"))
    freeze_kernel_resolution("heuristic promotion reuses one dense specialization")
    try:
        for m in (1, 4, 8):
            blockscaled.mm(source[:m], weight, out=output[:m], workspace=workspace)
            assert_close(output[:m], source[:m].float() @ decoded.T)
    finally:
        unfreeze_kernel_resolution()
        _policy.resolve_precision.cache_clear()


@pytest.mark.parametrize("recipe,n,k,m", [("nvfp4", 4096, 5376, 8), ("mxfp8", 5120, 17408, 16)])
def test_embedded_precision_promotion_captures_the_real_one_shot_path(recipe, n, k, m, monkeypatch):
    require_b12x()
    from b12x.gemm.blockscaled import _policy
    from b12x.policy import PolicyContext, PolicyMode, PolicySource

    policy = PolicyContext.for_device("cuda", mode=PolicyMode.PREPLANNED_ONLY)
    if policy.device.product_name != "nvidia rtx pro 6000 blackwell max-q workstation edition":
        pytest.skip("precision profile is qualified for the RTX PRO 6000 Max-Q")
    monkeypatch.setattr(_policy, "get_auto_policy", lambda device: policy)
    _policy.resolve_precision.cache_clear()
    weight, decoded, _ = make_weight(recipe, n, k)
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(blockscaled.workspace_size(weight, m), device="cuda", dtype=torch.uint8)
    blockscaled.prewarm(weight, [m], workspace=workspace)
    resolution = _policy.resolve_precision(source.device, recipe, k, n)
    assert resolution.source is PolicySource.PREPLANNED
    assert resolution.config.select(m) is not None
    monkeypatch.setattr(_quantize, "launch", lambda *args, **kwargs: pytest.fail("promoted GEMM quantized its activation"))
    freeze_kernel_resolution("embedded precision promotion")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            blockscaled.mm(source, weight, out=output, workspace=workspace)
        source.normal_()
        workspace.fill_(255)
        output.fill_(float("nan"))
        graph.replay()
        assert_close(output, source.float() @ decoded.T)
    finally:
        unfreeze_kernel_resolution()
        _policy.resolve_precision.cache_clear()


@pytest.mark.parametrize("fp4", [True, False])
def test_native_weight_pairs_exhaustive(fp4):
    """Every value byte and scale byte, including UE8M0 zero and NaNs."""
    require_b12x()
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.compiler import compile as compile_kernel
    from b12x._lib.intrinsics import nvfp4_pair_to_bf16x2_sm120, mxfp8_pair_to_bf16x2_sm120
    from b12x._lib.utils import make_ptr, current_cuda_stream

    class Decode:
        def __init__(self, fp4):
            self.fp4 = fp4

        @cute.jit
        def __call__(self, q: cute.Pointer, sf: cute.Pointer, out: cute.Pointer, stream):
            self.kernel(q, sf, out).launch(grid=(256, 1, 1), block=(256, 1, 1), stream=stream)

        @cute.kernel
        def kernel(self, q: cute.Pointer, sf: cute.Pointer, out: cute.Pointer):
            i = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
            if cutlass.const_expr(self.fp4):
                out[i] = nvfp4_pair_to_bf16x2_sm120(q[i], sf[i])
            else:
                out[i] = mxfp8_pair_to_bf16x2_sm120(q[i], sf[i])

    code = torch.arange(256, device="cuda", dtype=torch.int64).repeat(256)
    sf = torch.arange(256, device="cuda", dtype=torch.int64).repeat_interleave(256)
    if fp4:
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64, device="cuda")
        values = torch.stack((lut[code & 15], lut[code >> 4]), -1)
        factors = sf.to(torch.uint8).view(torch.float8_e4m3fn).double()
        packed = code
    else:
        values = torch.stack((code, 255 - code), -1).to(torch.uint8).view(torch.float8_e4m3fn).double()
        factors = torch.exp2(sf.double() - 127)
        factors[sf == 255] = float("nan")
        packed = code | ((255 - code) << 8)
    expected = (values * factors[:, None]).to(torch.bfloat16)
    packed, sf = packed.to(torch.uint32), sf.to(torch.uint32)
    out = torch.empty_like(expected)
    pointers = [make_ptr(cutlass.Uint32, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                for tensor in (packed, sf, out)]
    fn = compile_kernel(Decode(fp4), *pointers, current_cuda_stream())
    fn(*pointers, current_cuda_stream())
    torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("m", [8, 2048])
def test_gb10_embedded_precision_captures_selected_route(recipe, m, monkeypatch):
    require_b12x()
    from b12x.gemm.blockscaled import _policy
    from b12x.policy import PolicyContext, PolicyMode, PolicySource

    policy = PolicyContext.for_device("cuda", mode=PolicyMode.PREPLANNED_ONLY)
    if policy.device.product_name != "nvidia gb10":
        pytest.skip("GB10 precision profile qualification")
    monkeypatch.setattr(_policy, "get_auto_policy", lambda device: policy)
    _policy.resolve_precision.cache_clear()
    weight, decoded, _ = make_weight(recipe, 4096, 5376)
    source = torch.randn(m, 5376, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(m, 4096, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(blockscaled.workspace_size(weight, m), device="cuda", dtype=torch.uint8)
    options = dict(activation_global_scale=torch.tensor([128.], device="cuda")) if recipe == "nvfp4" else {}
    blockscaled.prewarm(weight, [1, m], workspace=workspace, **options)
    resolution = _policy.resolve_precision(source.device, recipe, 5376, 4096)
    assert resolution.source is PolicySource.PREPLANNED
    promoted = resolution.config.select(m) is not None
    assert promoted == (m == 8)
    if promoted:
        monkeypatch.setattr(_quantize, "launch", lambda *a, **kw: pytest.fail("GB10 promoted GEMM quantized an activation"))
    else:
        monkeypatch.setattr(_a16, "_a16", lambda *a, **kw: pytest.fail("GB10 quantized route executed A16"))
    freeze_kernel_resolution("GB10 embedded precision dispatch")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            blockscaled.mm(source, weight, out=output, workspace=workspace, **options)
        source.normal_()
        expected = (source.float() @ decoded.T) if promoted else blockscaled.mm(source, weight, mode="quantized", **options)
        workspace.fill_(255)
        output.fill_(float("nan"))
        graph.replay()
        if promoted:
            assert_close(output, expected)
        else:
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
    finally:
        unfreeze_kernel_resolution()
        _policy.resolve_precision.cache_clear()


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("k", [32, 96, 160])
def test_a16_short_scale_tile_tail(recipe, k):
    require_b12x()
    weight, decoded, _ = make_weight(recipe, 40, k)
    source = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    actual = blockscaled.mm(source, weight, mode="a16", _config=(128, 128, 4))
    assert_close(actual, source.float() @ decoded.T)


def test_a16_rejects_tma_misalignment():
    require_b12x()
    weight, _, _ = make_weight("nvfp4", 37, 80)
    source = torch.randn(2, 80, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="A16 TMA"):
        blockscaled.mm(source, weight, mode="a16")
