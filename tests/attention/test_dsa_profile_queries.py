from contextlib import contextmanager

import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x._lib.runtime_control import kernel_resolution_frozen
from b12x.policy import PolicyContext
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.tunable import DsaIndexerMergeGenerator
from b12x.policy.generation.sweep import SweepCase
from tests.conftest import require_b12x


def test_dsa_merge_races_public_plans_with_partial_counts_and_high_pool_offsets(tmp_path, monkeypatch):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(source_layout="paged", mode="decode", dtype="bfloat16", kv_dtype="uint8",
                 num_q_heads=32, num_idx_heads=1, max_q_rows=3, max_k_rows=4160,
                 top_k=512, page_size=64, score_mode="dsa", shared_page_table=False)
    cases = tuple(SweepCase.create(group_id=case.group_id, query=case.query, scenario=case.scenario,
                   metadata={**case.metadata, "minimum_pool_offset_bytes": 2**31})
                  for case in DsaIndexerMergeGenerator.cases_for_tuning_queries((query,)))
    provider = DsaIndexerMergeGenerator(cases=cases)
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    graph_context = torch.cuda.graph

    @contextmanager
    def frozen_capture(*args, **kwargs):
        already_frozen = kernel_resolution_frozen()
        if not already_frozen:
            freeze_kernel_resolution("DSA capture must reuse warmed kernels")
        try:
            with graph_context(*args, **kwargs):
                allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                yield
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
        finally:
            if not already_frozen:
                unfreeze_kernel_resolution()

    monkeypatch.setattr(torch.cuda, "graph", frozen_capture)
    with provider._benchmark_factory(cases[0].group_id, cases, context) as session:
        for case in cases:
            candidates = session.candidates(case)
            if case.scenario == "partial":
                freeze_kernel_resolution("partial live counts must reuse the full-capacity DSA kernels")
            try:
                measurements = session.measure(case, candidates)
            finally:
                unfreeze_kernel_resolution()
            assert len(measurements) == len(candidates) == 2
            for item in measurements:
                assert item.correct, (case.scenario, item.candidate.config, item.error, item.metrics)
                assert item.metrics["replay_allocation_bytes"] == 0
                assert item.metrics["ctas_per_group"] == min(65, context.device.sm_count // 3)
