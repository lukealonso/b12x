from contextlib import contextmanager

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.policy import PolicyContext, PolicyMode, PolicySource
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.attention import CompressedSparseMlaAttentionGenerator
from b12x.policy.generation.sweep import SweepCase
from tests.conftest import require_b12x


@pytest.mark.parametrize("mode,rows,swa,indexed,indexed_page", (
    ("decode", 16, 128, 512, 64),
    ("decode", 16, 128, 513, 64),
    ("extend", 17, 128, 65, 2),
))
def test_compressed_mla_shape_races_cover_chunk_boundary_and_high_pool_offsets(mode, rows, swa, indexed, indexed_page, tmp_path, monkeypatch):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(layout="compressed_dsv4", mode=mode, q_dtype="bfloat16", kv_dtype="float8_e4m3fn",
                 num_q_heads=32, qk_head_dim=512, v_head_dim=512, query_rows=rows,
                 swa_width=swa, indexed_width=indexed, swa_page_size=64, indexed_page_size=indexed_page)
    base, = CompressedSparseMlaAttentionGenerator.cases_for_tuning_queries((query,))
    case = SweepCase.create(group_id=base.group_id, query=query, scenario="high-pool-offsets",
                            metadata={"minimum_pool_offset_bytes": 2**31})
    provider = CompressedSparseMlaAttentionGenerator(cases=(case,))
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    graph_context = torch.cuda.graph

    @contextmanager
    def frozen_capture(*args, **kwargs):
        freeze_kernel_resolution("compressed MLA capture must reuse warmed kernels")
        try:
            with graph_context(*args, **kwargs):
                allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                yield
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
        finally:
            unfreeze_kernel_resolution()

    monkeypatch.setattr(torch.cuda, "graph", frozen_capture)
    with provider._benchmark_factory(case.group_id, (case,), context) as session:
        candidates = session.candidates(case)
        measurements = session.measure(case, candidates)
        assert candidates and len(measurements) == len(candidates)
        for item in measurements:
            assert item.correct, (query, item.candidate.config, item.error, item.metrics)
            assert item.metrics["replay_allocation_bytes"] == 0


def test_embedded_compressed_mla_corpus_constructs_preplanned_public_plans():
    from b12x.attention import compressed_sparse_mla

    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    policy = PolicyContext.for_device(device, mode=PolicyMode.PREPLANNED_ONLY)
    cases = CompressedSparseMlaAttentionGenerator()._cases
    assert len(cases) == 288
    for case in cases:
        query = case.query
        plan = compressed_sparse_mla.plan(
            compressed_sparse_mla.Caps(
                device=device, num_q_heads=query["num_q_heads"], max_q_rows=query["query_rows"],
                max_width=query["swa_width"] + query["indexed_width"],
                head_dim=query["qk_head_dim"], v_head_dim=query["v_head_dim"],
                max_batch=query["query_rows"], page_size=query["swa_page_size"],
                layout=query["layout"], mode=query["mode"], swa_width=query["swa_width"],
                indexed_width=query["indexed_width"], swa_page_size=query["swa_page_size"],
                indexed_page_size=query["indexed_page_size"], use_cuda_graph=True,
            ), policy=policy,
        )
        assert plan.policy_resolution.source is PolicySource.PREPLANNED
        assert plan.caps.max_chunks_per_row == plan.policy_resolution.config.max_chunks_per_row
