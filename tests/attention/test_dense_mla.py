"""Native dense MLA correctness, serving, and addressing gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from b12x.attention import dense_mla
from b12x.attention.dense_mla._reference import K3_SM_SCALE

from ..conftest import require_b12x

FP8 = torch.float8_e4m3fn
HEADS = 8
QK_DIM = 576
VALUE_DIM = 512


def _scratch(plan: dense_mla.Plan) -> torch.Tensor:
    (spec,) = plan.scratch_specs()
    return torch.empty(
        spec.shape,
        dtype=spec.dtype,
        device=spec.device,
    )


def test_is_supported_accepts_implicit_current_device() -> None:
    require_b12x()
    assert dense_mla.is_supported()


def _guarded_scratch(
    plan: dense_mla.Plan,
    *,
    guard_bytes: int = 16 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact planned scratch surrounded by initialized canaries."""
    (spec,) = plan.scratch_specs()
    assert spec.dtype == torch.uint8
    storage = torch.full(
        (spec.nbytes + 2 * guard_bytes,),
        0xA5,
        dtype=torch.uint8,
        device=spec.device,
    )
    scratch = storage.narrow(0, guard_bytes, spec.nbytes)
    return storage, scratch


def _assert_matches(
    output: torch.Tensor,
    lse: torch.Tensor,
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
) -> None:
    torch.cuda.synchronize()
    assert bool(torch.isfinite(output).all().item())
    assert bool(torch.isfinite(lse).all().item())
    assert int(torch.count_nonzero(output).item()) == output.numel()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().reshape(output.shape[0], -1),
        reference_output.float().reshape(output.shape[0], -1),
        dim=1,
    )
    assert float(cosine.min().item()) > 0.999
    torch.testing.assert_close(
        output.float(),
        reference_output.float(),
        rtol=2e-2,
        atol=5e-4,
    )
    torch.testing.assert_close(lse, reference_lse, rtol=2e-5, atol=2e-5)


def test_source_is_standalone_cute() -> None:
    root = Path(dense_mla.__file__).resolve().parent
    forbidden = (
        "triton",
        "b12x.attention.paged",
        "b12x.attention.sparse_mla",
        "b12x.attention.dsa_indexer",
        "b12x.attention._shared.mla",
    )
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        for name in imports:
            assert not name.startswith(forbidden), (path.name, name)


def test_public_types_are_module_scoped_names() -> None:
    assert dense_mla.Caps.__name__ == "Caps"
    assert dense_mla.Plan.__name__ == "Plan"
    assert dense_mla.Binding.__name__ == "Binding"
    assert dense_mla.Scratch.__name__ == "Scratch"
    assert dense_mla.Budget.__name__ == "Budget"


@torch.inference_mode()
def test_fp8_physical_record_stride_ignores_padding() -> None:
    device = require_b12x()
    torch.manual_seed(20260813)
    rows = 2
    heads = 16
    pages = 3
    page_size = 64
    physical_width = 1088
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=rows,
            max_batch=rows,
            max_cache_tokens=65,
            max_page_table_width=2,
            num_cache_pages=pages,
            physical_record_width=physical_width,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(rows, heads, QK_DIM, device=device) * 10).to(FP8)
    cache = torch.empty(
        pages,
        page_size,
        physical_width,
        dtype=FP8,
        device=device,
    )
    cache[..., :QK_DIM] = (
        torch.randn(pages, page_size, QK_DIM, device=device) * 10
    ).to(FP8)
    cache[..., QK_DIM:] = (
        (torch.randn(pages, page_size, physical_width - QK_DIM, device=device) * 100)
        .clamp(-448, 448)
        .to(FP8)
    )
    page_table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([64, 65], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(rows + 1, dtype=torch.int32, device=device)
    output = torch.empty(rows, heads, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    actual, actual_lse = dense_mla.run(binding=binding)
    expected, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    _assert_matches(actual, actual_lse, expected, expected_lse)


def test_partial_row_budget_changes_native_split_policy() -> None:
    device = require_b12x()
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=16,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=8,
            budget=dense_mla.Budget(max_partial_rows=0),
        )
    )
    assert plan.num_splits == 1
    assert plan.chunks_per_split == 2


@pytest.mark.parametrize("heads", [8, 12])
@torch.inference_mode()
def test_bf16_multi_request_decode_matches_reference(heads: int) -> None:
    device = require_b12x()
    torch.manual_seed(20260730)
    batch = 4
    page_size = 16
    pages = 24
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=batch,
            max_batch=batch,
            max_cache_tokens=96,
            max_page_table_width=6,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(batch, heads, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.tensor(
        [
            [4, 7, 1, 5, 0, 3],
            [18, 2, 19, 6, 8, 21],
            [10, 9, 12, 16, 15, 11],
            [23, 17, 20, 14, 13, 22],
        ],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(
        [1, 17, 63, 91],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.arange(
        batch + 1,
        dtype=torch.int32,
        device=device,
    )
    output = torch.full(
        (batch, heads, VALUE_DIM),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    scratch = _scratch(plan)

    # Compile first through a smaller live batch. The same capacity-planned
    # specialization must then accept the full batch without recompilation or
    # stale tensor-layout assumptions.
    small_output = torch.empty(
        1,
        heads,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    small_binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q[:1],
        kv_cache=cache,
        output=small_output,
        page_table=page_table[:1],
        cache_seqlens=cache_seqlens[:1],
        cu_seqlens_q=torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device=device,
        ),
    )
    small_actual, small_lse = dense_mla.run(binding=small_binding)
    small_expected, small_expected_lse = dense_mla.reference(
        small_binding.q,
        small_binding.kv_cache,
        small_binding.page_table,
        small_binding.cache_seqlens,
        small_binding.cu_seqlens_q,
    )
    _assert_matches(
        small_actual,
        small_lse,
        small_expected,
        small_expected_lse,
    )

    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )

    # Compilation must not launch or mutate graph-visible destinations.
    dense_mla.compile(binding=binding)
    torch.cuda.synchronize()
    assert bool(torch.isnan(output).all().item())
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@pytest.mark.parametrize("heads", [8, 12])
@torch.inference_mode()
def test_fp8_query_tiled_causal_extend_matches_reference(heads: int) -> None:
    device = require_b12x()
    torch.manual_seed(20260731)
    query_rows = 5
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="extend",
            kv_dtype=FP8,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=query_rows,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
        )
    )
    assert plan.query_tile == 4
    q_float = torch.randn(query_rows, heads, QK_DIM, device=device) * 0.14
    cache_float = torch.randn(pages, page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.tensor(
        [[4, 7, 1, 5, 0, 3, 6, 2]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([77], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, query_rows],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        query_rows,
        heads,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_bf16_query_tiled_causal_extend_matches_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260732)
    query_rows = 7
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="extend",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=query_rows,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
        )
    )
    assert plan.query_tile == 2
    q = (torch.randn(query_rows, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.tensor(
        [[4, 7, 1, 5, 0, 3, 6, 2]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([101], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, query_rows],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        query_rows,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_padded_page_stride_matches_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260801)
    page_size = 16
    pages = 5
    page_payload = page_size * QK_DIM
    page_stride = page_payload + 128
    storage = torch.empty(
        pages * page_stride,
        dtype=torch.bfloat16,
        device=device,
    )
    cache = torch.as_strided(
        storage,
        size=(pages, page_size, QK_DIM),
        stride=(page_stride, QK_DIM, 1),
    )
    cache.normal_().mul_(0.1)
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=64,
            max_page_table_width=4,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor(
        [[4, 1, 3, 0]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([61], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    q_before = q.clone()
    cache_before = cache.clone()
    page_table_before = page_table.clone()
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.testing.assert_close(q, q_before, rtol=0, atol=0)
    torch.testing.assert_close(cache, cache_before, rtol=0, atol=0)
    torch.testing.assert_close(page_table, page_table_before, rtol=0, atol=0)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_cuda_graph_replay_is_allocation_stable_and_reads_live_inputs() -> None:
    device = require_b12x()
    torch.manual_seed(20260802)
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
            use_cuda_graph=True,
        )
    )
    assert plan.num_splits > 1
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.arange(
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, pages)
    cache_seqlens = torch.tensor([97], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated(device)
    q.mul_(0.75)
    cache_seqlens.fill_(65)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated(device)
    assert allocated_after == allocated_before

    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        captured_output,
        captured_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_fp8_production_split_plan_handles_short_live_sequence() -> None:
    """The 1M K3 plan must not touch inactive split storage or cache pages."""
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 768
    page_width = (131_072 + page_size - 1) // page_size
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=48,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=131_072,
            max_page_table_width=page_width,
            num_cache_pages=1,
            use_cuda_graph=True,
        )
    )
    assert plan.num_splits == 94

    q_float = torch.randn(1, 48, QK_DIM, device=device) * 0.1
    cache_float = torch.randn(1, page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([257], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        48,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    guarded_storage, scratch = _guarded_scratch(plan)
    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
        # One launched split per live 64-token chunk. The five chunks fit
        # the plan's single-split threshold (chunks_per_split = 22), so
        # split 0 scans all of them and splits 1-4 publish empty partials.
        active_splits=5,
    )
    assert binding.active_splits == 5
    assert plan.single_split_chunks == plan.chunks_per_split == 22
    dense_mla.compile(binding=binding)
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    graph.replay()
    torch.cuda.synchronize()
    guard_bytes = scratch.storage_offset()
    expected_guard = torch.full(
        (guard_bytes,),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    torch.testing.assert_close(guarded_storage[:guard_bytes], expected_guard)
    torch.testing.assert_close(guarded_storage[-guard_bytes:], expected_guard)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        captured_output,
        captured_lse,
        expected_output,
        expected_lse,
    )

    # A prefix of min(num_splits, live chunks) launched splits assigns the
    # live chunks exactly like the full-plan launch (here split 0 takes all
    # five chunks under either launch and every other split stays empty),
    # so the two must match including the BF16 output bits.
    full_output = torch.empty_like(output)
    full_binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=full_output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    assert full_binding.active_splits == plan.num_splits
    full_actual, full_lse = dense_mla.run(binding=full_binding)
    torch.cuda.synchronize()
    torch.testing.assert_close(full_actual, captured_output, rtol=0, atol=0)
    torch.testing.assert_close(full_lse, captured_lse, rtol=0, atol=0)
    finite_partials = torch.isfinite(full_binding.scratch.partial_lse[0]).sum(dim=-1)
    assert int(finite_partials.max().item()) == 1


@torch.inference_mode()
def test_page_ids_past_int32_scaled_offset_match_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 16
    record_bytes = (
        QK_DIM
        * torch.empty(
            (),
            dtype=torch.bfloat16,
        ).element_size()
    )
    page_stride_bytes = page_size * record_bytes
    high_page = torch.iinfo(torch.int32).max // page_stride_bytes + 2
    live_pages = 2
    pages = high_page + live_pages
    assert high_page * page_stride_bytes > torch.iinfo(torch.int32).max

    # Roughly 2 GiB is intentionally mostly uninitialized. Only the live tail
    # pages are touched; this reproduces high recycled pool ids.
    cache = torch.empty(
        pages,
        page_size,
        QK_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    cache[high_page:].normal_().mul_(0.1)
    page_table = torch.arange(
        high_page,
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, live_pages)
    assert (
        int(page_table.min().item()) * cache.stride(0) * cache.element_size()
        > torch.iinfo(torch.int32).max
    )

    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * live_pages,
            max_page_table_width=live_pages,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache_seqlens = torch.tensor(
        [page_size + 9],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_fp8_page_ids_past_int32_scaled_offset_match_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 768
    record_bytes = QK_DIM
    page_stride_bytes = page_size * record_bytes
    high_page = torch.iinfo(torch.int32).max // page_stride_bytes + 2
    live_pages = 2
    pages = high_page + live_pages
    assert high_page * page_stride_bytes > torch.iinfo(torch.int32).max

    cache = torch.empty(
        pages,
        page_size,
        QK_DIM,
        dtype=FP8,
        device=device,
    )
    cache_float = (
        torch.randn(
            live_pages,
            page_size,
            QK_DIM,
            device=device,
        )
        * 0.1
    )
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    cache[high_page:] = (cache_float / kv_scale).to(FP8)
    page_table = torch.arange(
        high_page,
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, live_pages)
    assert (
        int(page_table.min().item()) * cache.stride(0) * cache.element_size()
        > torch.iinfo(torch.int32).max
    )

    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * live_pages,
            max_page_table_width=live_pages,
            num_cache_pages=pages,
        )
    )
    q_float = torch.randn(1, HEADS, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache_seqlens = torch.tensor(
        [page_size + 9],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_balanced_splits_cover_a_long_live_sequence_with_every_split() -> None:
    """A live sequence far below the planned capacity is shared by every
    launched split (about live_chunks / active_splits chunks each) instead
    of a few splits scanning capacity-sized ranges; the result matches the
    reference, and a one-split launch of the same rows agrees within the
    reassociation tolerance."""
    device = require_b12x()
    torch.manual_seed(20260902)
    page_size = 64
    max_cache_tokens = 131_072
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=8,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=max_cache_tokens,
            max_page_table_width=max_cache_tokens // page_size,
            num_cache_pages=1 << 20,
            use_cuda_graph=True,
        )
    )
    assert plan.num_splits > 8
    live = 10_240  # 160 chunks: fewer than the planned 2,048, more than the splits
    num_pages = live // page_size
    q = torch.randn(1, 8, QK_DIM, device=device, dtype=torch.bfloat16) * 0.1
    cache = torch.randn(num_pages, page_size, QK_DIM, device=device, dtype=torch.bfloat16) * 0.1
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, -1)
    cache_seqlens = torch.tensor([live], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, cache_seqlens, cu_seqlens_q
    )

    outputs = {}
    for active in (plan.num_splits, 1):
        output = torch.empty(1, 8, VALUE_DIM, dtype=torch.bfloat16, device=device)
        binding = dense_mla.bind(
            plan,
            scratch=_scratch(plan),
            q=q,
            kv_cache=cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_splits=active,
        )
        dense_mla.compile(binding=binding)
        actual_output, actual_lse = dense_mla.run(binding=binding)
        torch.cuda.synchronize()
        _assert_matches(actual_output, actual_lse, expected_output, expected_lse)
        outputs[active] = (actual_output.clone(), actual_lse.clone())
    full_output, full_lse = outputs[plan.num_splits]
    one_output, one_lse = outputs[1]
    torch.testing.assert_close(full_output, one_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(full_lse, one_lse, rtol=2e-5, atol=2e-5)


def _plan_131k(device, **overrides) -> dense_mla.Plan:
    """A decode plan sized for one DCP-8 shard of a 1M-token window with four
    query rows (one 8-head tile each), which the wave-balanced planner maps
    to 47 splits of 44 chunks on a 188-SM device."""
    page_size = 64
    max_cache_tokens = 131_072
    caps = dict(
        device=device,
        mode="decode",
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        page_size=page_size,
        max_total_q=4,
        max_batch=4,
        max_cache_tokens=max_cache_tokens,
        max_page_table_width=max_cache_tokens // page_size,
        num_cache_pages=1 << 20,
        use_cuda_graph=True,
    )
    caps.update(overrides)
    return dense_mla.plan(dense_mla.Caps(**caps))


def _run_131k(plan, scratch, q, cache, live, *, active_splits=None):
    device = q.device
    page_size = 64
    num_pages = (live + page_size - 1) // page_size
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, -1)
    cache_seqlens = torch.tensor([live], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, 8, VALUE_DIM, dtype=torch.bfloat16, device=device)
    kwargs = {} if active_splits is None else {"active_splits": active_splits}
    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        **kwargs,
    )
    dense_mla.compile(binding=binding)
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    return actual_output.clone(), actual_lse.clone(), binding


def test_partial_dtype_and_single_split_chunks_are_validated() -> None:
    common = dict(
        device="cpu",
        mode="decode",
        kv_dtype=torch.bfloat16,
        num_q_heads=HEADS,
        page_size=16,
        max_total_q=1,
        max_batch=1,
        max_cache_tokens=128,
        max_page_table_width=8,
        num_cache_pages=8,
    )
    with pytest.raises(TypeError, match="partial_dtype"):
        dense_mla.Caps(partial_dtype=torch.float16, **common)
    with pytest.raises(ValueError, match="single_split_chunks"):
        dense_mla.Caps(single_split_chunks=-1, **common)
    plan = dense_mla.plan(dense_mla.Caps(**common))
    assert plan.single_split_chunks == plan.chunks_per_split
    assert plan.partial_dtype == torch.bfloat16
    balanced = dense_mla.plan(dense_mla.Caps(single_split_chunks=0, **common))
    assert balanced.single_split_chunks == 0


def test_fp32_partials_double_the_partial_scratch() -> None:
    """The partial-output region is the only scratch that scales with the
    partial element type; the LSE regions keep their float32 size."""
    device = require_b12x()
    bf16_plan = _plan_131k(device)
    fp32_plan = _plan_131k(device, partial_dtype=torch.float32)
    assert bf16_plan.num_splits == fp32_plan.num_splits > 1
    (bf16_spec,) = bf16_plan.scratch_specs()
    (fp32_spec,) = fp32_plan.scratch_specs()
    rows = bf16_plan.caps.max_total_q
    partial_bf16_bytes = rows * 8 * bf16_plan.num_splits * VALUE_DIM * 2
    grown = fp32_spec.shape[0] * fp32_spec.dtype.itemsize - (
        bf16_spec.shape[0] * bf16_spec.dtype.itemsize
    )
    assert grown == partial_bf16_bytes


@torch.inference_mode()
@pytest.mark.parametrize("partial_dtype", [torch.bfloat16, torch.float32])
def test_single_split_requests_match_the_one_split_launch_bitwise(
    partial_dtype: torch.dtype,
) -> None:
    """A request with at most single_split_chunks live chunks is scanned by
    split 0 alone under a full-plan launch, so its merged result equals the
    direct one-split write bit for bit (the fixed-range association a
    chunks_per_split-run kernel produces); the 44-chunk run boundary is the
    last such length. One chunk past it the request is balanced over the
    launched splits, which the full-plan and eager launches partition
    identically."""
    device = require_b12x()
    torch.manual_seed(20260903)
    plan = _plan_131k(device, partial_dtype=partial_dtype)
    assert plan.num_splits > 8
    run = plan.chunks_per_split
    assert plan.single_split_chunks == run
    scratch = _scratch(plan)
    q = torch.randn(1, 8, QK_DIM, device=device, dtype=torch.bfloat16) * 0.3
    cache = torch.randn(2_048, 64, QK_DIM, device=device, dtype=torch.bfloat16) * 0.3
    for live in (1, 63, 64, 65, run * 64 // 2, run * 64 - 1, run * 64):
        full_output, full_lse, binding = _run_131k(plan, scratch, q, cache, live)
        finite = torch.isfinite(binding.scratch.partial_lse[0]).sum(dim=-1)
        assert int(finite.max().item()) == 1, live
        one_output, one_lse, _ = _run_131k(
            plan, scratch, q, cache, live, active_splits=1
        )
        assert torch.equal(full_output, one_output), live
        assert torch.equal(full_lse, one_lse), live
    live = run * 64 + 1
    full_output, full_lse, binding = _run_131k(plan, scratch, q, cache, live)
    finite = torch.isfinite(binding.scratch.partial_lse[0]).sum(dim=-1)
    assert int(finite.min().item()) == min(plan.num_splits, run + 1)
    eager_output, eager_lse, _ = _run_131k(
        plan, scratch, q, cache, live, active_splits=min(plan.num_splits, run + 1)
    )
    assert torch.equal(full_output, eager_output)
    assert torch.equal(full_lse, eager_lse)


@torch.inference_mode()
def test_single_split_chunks_zero_balances_short_requests() -> None:
    device = require_b12x()
    torch.manual_seed(20260903)
    plan = _plan_131k(device, single_split_chunks=0)
    assert plan.single_split_chunks == 0
    scratch = _scratch(plan)
    q = torch.randn(1, 8, QK_DIM, device=device, dtype=torch.bfloat16) * 0.3
    cache = torch.randn(64, 64, QK_DIM, device=device, dtype=torch.bfloat16) * 0.3
    output, lse, binding = _run_131k(plan, scratch, q, cache, 1_000)
    finite = torch.isfinite(binding.scratch.partial_lse[0]).sum(dim=-1)
    assert int(finite.min().item()) == 16  # one split per live chunk
    page_table = torch.arange(16, dtype=torch.int32, device=device).view(1, -1)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache[:16],
        page_table,
        torch.tensor([1_000], dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
    )
    _assert_matches(output, lse, expected_output, expected_lse)


@torch.inference_mode()
def test_fp32_partials_remove_the_second_rounding() -> None:
    """With bf16 partials a merged result is rounded twice, so balancing a
    short request over many one-chunk splits is measurably less accurate
    than the one-split scan; float32 partials bring the merged result back
    to the one-split accuracy, and improve every multi-split result."""
    device = require_b12x()
    torch.manual_seed(20260903)
    q = torch.randn(1, 8, QK_DIM, device=device, dtype=torch.bfloat16) * 0.3
    cache = torch.randn(2_048, 64, QK_DIM, device=device, dtype=torch.bfloat16) * 0.3

    def relative_error(output, live):
        num_pages = (live + 63) // 64
        page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, -1)
        records = cache[:num_pages].reshape(-1, QK_DIM)[:live].double()
        scores = (q[0].double() @ records.t()) * K3_SM_SCALE
        probabilities = torch.softmax(scores, dim=-1)
        reference = probabilities @ records[:, :VALUE_DIM]
        error = output[0].double() - reference
        return float(error.norm() / reference.norm())

    plans = {
        dtype: _plan_131k(device, single_split_chunks=0, partial_dtype=dtype)
        for dtype in (torch.bfloat16, torch.float32)
    }
    one_split = _plan_131k(device)
    assert one_split.chunks_per_split * 64 >= 2_816
    scratch = _scratch(plans[torch.float32])
    for live in (1_000, 2_816):
        one_output, _, _ = _run_131k(one_split, scratch, q, cache, live)
        bf16_output, _, _ = _run_131k(plans[torch.bfloat16], scratch, q, cache, live)
        fp32_output, _, _ = _run_131k(plans[torch.float32], scratch, q, cache, live)
        single = relative_error(one_output, live)
        bf16 = relative_error(bf16_output, live)
        fp32 = relative_error(fp32_output, live)
        assert bf16 > single * 1.05, (live, single, bf16)
        assert fp32 < bf16 * 0.95, (live, bf16, fp32)
        assert abs(fp32 - single) < single * 0.05, (live, single, fp32)
    live = 10_240  # 160 chunks: every plan merges several partials
    bf16_output, _, _ = _run_131k(plans[torch.bfloat16], scratch, q, cache, live)
    fp32_output, _, _ = _run_131k(plans[torch.float32], scratch, q, cache, live)
    assert relative_error(fp32_output, live) < relative_error(bf16_output, live) * 0.95
