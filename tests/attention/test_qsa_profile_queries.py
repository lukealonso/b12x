from contextlib import contextmanager

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.policy import PolicyContext
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.attention import QsaAttentionGenerator
from tests.conftest import require_b12x


@pytest.mark.parametrize("kv_dtype", ("bfloat16", "float8_e4m3fn"))
def test_qsa_query_races_full_partial_and_decode_graphs(kv_dtype, tmp_path, monkeypatch):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = dict(q_dtype="bfloat16", kv_dtype=kv_dtype, q_heads=12, kv_heads=1, head_dim=256,
                 index_heads=4, index_kv_heads=1, index_head_dim=128, index_rotary_dim=64,
                 main_page_size=16, max_batch=3, max_q_rows=97, max_seq_len=260,
                 max_speculative_tokens=3, compress_ratio=4, budget=2048,
                 position_axes=3, mrope_interleaved=True)
    cases = tuple(QsaAttentionGenerator.cases_for_tuning_queries((query,)))
    provider = QsaAttentionGenerator(cases=cases)
    context = GenerationContext(device=PolicyContext.for_device(device).device,
                                device_ordinal=device.index, work_dir=tmp_path,
                                source_revision="test", settings=GenerationSettings())
    graph_context = torch.cuda.graph

    @contextmanager
    def frozen_capture(*args, **kwargs):
        freeze_kernel_resolution("QSA capture must reuse warmed kernels")
        try:
            with graph_context(*args, **kwargs):
                yield
        finally:
            unfreeze_kernel_resolution()

    monkeypatch.setattr(torch.cuda, "graph", frozen_capture)
    with provider._benchmark_factory(cases[0].group_id, cases, context) as session:
        for case in cases:
            candidates = session.candidates(case)
            measurements = session.measure(case, candidates)
            assert len(measurements) == len(candidates) == 3
            assert all(item.correct for item in measurements), [
                (item.candidate.config, item.error) for item in measurements if not item.correct]
