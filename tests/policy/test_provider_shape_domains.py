from types import SimpleNamespace

import pytest

from b12x.policy.catalog import list_profiled_components
from b12x.policy.types import FrozenMapping


@pytest.mark.parametrize("component", (
    "quantization.nvfp4", "attention.gqa", "attention.varlen", "sequence.kda_prefill",
    "gemm.bf16_vocab_projection", "gemm.block_fp8_linear",
    "norm.hyperconnection", "sequence.mtp_feedback", "norm.mhc", "attention.mla", "attention.compressed_sparse_mla", "attention.gdn", "attention.qsa",
))
def test_shape_factories_preserve_query_and_validate_every_production_candidate(component):
    registration = next(item for item in list_profiled_components() if item.component_id == component)
    generator = registration.create_generator()
    original = generator._cases[0].query
    cases = tuple(generator.cases_for_tuning_queries((original,)))
    assert cases and all(case.query == original for case in cases)
    assert len({case.case_id for case in cases}) == len(cases)
    assert cases == tuple(generator.cases_for_tuning_queries((original,)))
    if component == "attention.gqa":
        return
    session = generator._benchmark_factory(cases[0].group_id, cases, SimpleNamespace(device=None))
    for case in cases:
        candidates = session.candidates(case)
        assert candidates
        for candidate in candidates:
            generator.validate_region_decision(case.query, None, candidate.config)


def test_kda_shape_factory_retains_partial_live_count_and_long_memory_scenario():
    from b12x.policy.generation.providers.kda import KdaPrefillGenerator

    query = dict(heads=8, head_dim=128, max_tokens=65, max_seqs=3,
                 model_dtype="bfloat16", state_dtype="float32", qk_l2norm=True, checkpoint_export=True)
    cases = tuple(KdaPrefillGenerator.cases_for_tuning_queries((query,)))
    assert [(case.scenario, case.metadata["live_tokens"], case.metadata["gate"]) for case in cases] == [
        ("full", 65, "random"), ("partial-long-memory", 33, "long_memory")]
    with pytest.raises(ValueError):
        tuple(KdaPrefillGenerator.cases_for_tuning_queries(({**query, "head_dim": 64},)))


def test_varlen_shape_factory_preserves_unequal_total_rows_and_capacity():
    from b12x.policy.generation.providers.tunable import VarlenAttentionGenerator, _attention_lengths

    query = dict(variant="varlen", dtype="float16", causal=True, batch_size=3,
                 q_heads=4, kv_heads=1, q_head_dim=128, v_head_dim=128,
                 query_rows=13, kv_rows=97, max_seqlen_q=8, max_seqlen_k=40)
    cases = tuple(VarlenAttentionGenerator.cases_for_tuning_queries((query,)))
    assert cases[0].query == FrozenMapping(query)
    assert _attention_lengths(query, "query_rows", "max_seqlen_q") == (5, 4, 4)
    assert _attention_lengths(query, "kv_rows", "max_seqlen_k") == (33, 32, 32)
    with pytest.raises(ValueError, match="row counts"):
        tuple(VarlenAttentionGenerator.cases_for_tuning_queries(({**query, "variant": "batched"},)))
    with pytest.raises(ValueError, match="capacities"):
        tuple(VarlenAttentionGenerator.cases_for_tuning_queries(({**query, "query_rows": 25},)))


def test_vocabulary_search_excludes_illegal_triton_candidates():
    from b12x.gemm.bf16_vocab_projection._policy import MAX_IN_FEATURES
    from b12x.policy.generation.providers.gemm import Bf16VocabProjectionGenerator, _Bf16VocabProjectionSession

    query = dict(dtype="bfloat16", max_tokens=1, in_features=MAX_IN_FEATURES + 1, out_features=256)
    case, = Bf16VocabProjectionGenerator.cases_for_tuning_queries((query,))
    candidates = _Bf16VocabProjectionSession(SimpleNamespace(device=None)).candidates(case)
    assert len(candidates) == 1 and candidates[0].config["backend"] == "torch"
    with pytest.raises(ValueError, match="one-token"):
        tuple(Bf16VocabProjectionGenerator.cases_for_tuning_queries(({**query, "max_tokens": 2},)))


def test_dense_mla_shape_factory_rejects_unrepresented_layout_and_resource_contracts():
    from b12x.policy.generation.providers.attention import MlaAttentionGenerator

    query = dict(mode="decode", q_dtype="bfloat16", kv_dtype="bfloat16", num_q_heads=8,
                 qk_head_dim=576, v_head_dim=512, page_size=64, query_rows=3, max_batch=3,
                 cache_tokens=193, physical_record_width=576, window_size=None, use_cuda_graph=True)
    case, = MlaAttentionGenerator.cases_for_tuning_queries((query,))
    assert case.query == FrozenMapping(query)
    for change in ({"max_batch": 1}, {"window_size": 128}, {"physical_record_width": 640}):
        with pytest.raises(ValueError, match="fixtures require"):
            tuple(MlaAttentionGenerator.cases_for_tuning_queries(({**query, **change},)))
    with pytest.raises(ValueError, match="shared-memory"):
        tuple(MlaAttentionGenerator.cases_for_tuning_queries((
            {**query, "qk_head_dim": 1088, "v_head_dim": 1024, "physical_record_width": 1088},)))


@pytest.mark.parametrize("rows,heads,swa,indexed,indexed_page,capability,single_pass", (
    (16, 32, 128, 512, 64, (12, 1), True),
    (16, 32, 128, 513, 64, (12, 1), False),
    (16, 32, 129, 511, 64, (12, 1), False),
    (15, 32, 128, 512, 64, (12, 1), False),
    (16, 16, 128, 512, 64, (12, 1), False),
    (16, 32, 128, 512, 2, (12, 1), False),
    (16, 32, 128, 512, 64, (12, 0), False),
))
def test_compressed_mla_policy_and_candidates_share_production_chunk_limit(rows, heads, swa, indexed, indexed_page, capability, single_pass):
    from b12x.attention._shared.mla.compressed_api import _should_use_sm121_single_pass_decode
    from b12x.attention.compressed_sparse_mla._policy import SPARSE_MLA_POLICY, SparseMlaQuery
    from b12x.policy.generation.providers.attention import CompressedSparseMlaAttentionGenerator, _compressed_mla_candidates

    query = dict(layout="compressed_dsv4", mode="decode", q_dtype="bfloat16", kv_dtype="float8_e4m3fn",
                 num_q_heads=heads, qk_head_dim=512, v_head_dim=512, query_rows=rows,
                 swa_width=swa, indexed_width=indexed, swa_page_size=64, indexed_page_size=indexed_page)
    device = SimpleNamespace(compute_capability=capability)
    assert _should_use_sm121_single_pass_decode(rows=rows, heads=heads, swa_width=swa, indexed_width=indexed,
                                                swa_page_size=64, indexed_page_size=indexed_page,
                                                compute_capability=capability) == single_pass
    assert SPARSE_MLA_POLICY.heuristic(SparseMlaQuery(**query), device).max_chunks_per_row == (1 if single_pass else 64)
    candidates = _compressed_mla_candidates(query, device)
    assert (len(candidates) == 1) == single_pass
    for candidate in candidates:
        CompressedSparseMlaAttentionGenerator.validate_region_decision(query, device, candidate.config)
    with pytest.raises(ValueError, match="512-wide"):
        tuple(CompressedSparseMlaAttentionGenerator.cases_for_tuning_queries(({**query, "v_head_dim": 448},)))


def test_compressed_mla_rejects_unsupported_single_pass_swa_width():
    from b12x.policy.generation.providers.attention import CompressedSparseMlaAttentionGenerator

    query = dict(layout="compressed_dsv4", mode="extend", q_dtype="bfloat16", kv_dtype="float8_e4m3fn",
                 num_q_heads=32, qk_head_dim=512, v_head_dim=512, query_rows=17,
                 swa_width=127, indexed_width=65, swa_page_size=64, indexed_page_size=2)
    with pytest.raises(ValueError, match="require swa_width=128"):
        tuple(CompressedSparseMlaAttentionGenerator.cases_for_tuning_queries((query,)))
    with pytest.raises(ValueError, match="require swa_width=128"):
        CompressedSparseMlaAttentionGenerator.validate_region_decision(
            {**query, "mode": "decode", "swa_width": 640, "indexed_width": 0},
            SimpleNamespace(compute_capability=(12, 1)), {"max_chunks_per_row": 1},
        )


def test_gdn_shape_factory_preserves_partial_capacity_and_decay_contract():
    from b12x.policy.generation.providers.attention import GdnAttentionGenerator

    query = dict(gate_activation="silu", qk_l2norm=False, state_dtype="bfloat16",
                 key_heads=4, value_heads=12, max_seqs=3, max_tokens=11, state_index_columns=4)
    cases = tuple(GdnAttentionGenerator.cases_for_tuning_queries((query,)))
    assert [case.metadata["query_lengths"] for case in cases] == [(4, 4, 3), (2, 2, 1)]
    assert all(case.query == FrozenMapping(query) and case.metadata["decay_recipe"] == "gdn" for case in cases)
    for change in ({"value_heads": 4}, {"max_tokens": 13}, {"value_heads": 8}, {"state_index_columns": 9}):
        with pytest.raises(ValueError, match="GDN fixtures require"):
            tuple(GdnAttentionGenerator.cases_for_tuning_queries(({**query, **change},)))


def test_qsa_shape_factory_covers_capacity_and_live_path_boundary_without_changing_geometry():
    from b12x.policy.generation.providers.attention import QsaAttentionGenerator

    query = dict(q_dtype="bfloat16", kv_dtype="bfloat16", q_heads=12, kv_heads=1, head_dim=256,
                 index_heads=4, index_kv_heads=1, index_head_dim=128, index_rotary_dim=64,
                 main_page_size=16, max_batch=3, max_q_rows=97, max_seq_len=260,
                 max_speculative_tokens=3, compress_ratio=4, budget=2048,
                 position_axes=3, mrope_interleaved=True)
    cases = tuple(QsaAttentionGenerator.cases_for_tuning_queries((query,)))
    assert [(case.scenario, case.metadata["rows"], case.metadata["kind"]) for case in cases] == [
        ("full-prefill", 97, "prefill"), ("partial-prefill", 48, "prefill"), ("decode", 3, "throughput")]
    assert all(case.query == FrozenMapping(query) for case in cases)
    for change in ({"index_heads": 8}, {"budget": 8192}, {"max_q_rows": 261}, {"max_seq_len": 261}):
        with pytest.raises(ValueError, match="QSA fixture"):
            tuple(QsaAttentionGenerator.cases_for_tuning_queries(({**query, **change},)))


def test_dsa_merge_shape_factory_keeps_live_counts_separate_from_planned_capacity():
    from b12x.policy.generation.providers.tunable import DsaIndexerMergeGenerator

    query = dict(source_layout="paged", mode="decode", dtype="bfloat16", kv_dtype="uint8",
                 num_q_heads=32, num_idx_heads=1, max_q_rows=3, max_k_rows=4160,
                 top_k=512, page_size=64, score_mode="dsa", shared_page_table=False)
    cases = tuple(DsaIndexerMergeGenerator.cases_for_tuning_queries((query,)))
    assert [dict(case.metadata) for case in cases] == [
        dict(seq_len=4160, live_rows=3, page_table_width=65),
        dict(seq_len=2080, live_rows=1, page_table_width=65)]
    assert all(case.query == FrozenMapping(query) for case in cases)
    for change in ({"max_k_rows": 0}, {"max_k_rows": 4159}, {"mode": "prefill"}, {"max_q_rows": 17}):
        with pytest.raises(ValueError):
            tuple(DsaIndexerMergeGenerator.cases_for_tuning_queries(({**query, **change},)))


@pytest.mark.parametrize("layout", ("p24_k", "p24_n", "p33_k", "p33_n"))
def test_trellis_pair_family_preserves_rate_axis_and_candidate_constraints(layout):
    from b12x.policy.generation.providers.trellis import TrellisLinearGenerator, _candidates

    query = dict(max_rows=65, in_features=256, out_features=256, input_dtype="bfloat16",
                 compute_dtype="bfloat16", codebook="sqg_e4m3", bits=3, weight_layout=layout)
    generator = TrellisLinearGenerator()
    case, = generator.cases_for_tuning_queries((query,))
    assert case.query == FrozenMapping(query)
    candidates = _candidates(case.query, None)
    assert candidates
    if layout.endswith("_n"):
        assert {candidate.config["tile_n"] for candidate in candidates} == {256}
    axis = "out_features" if layout.endswith("_n") else "in_features"
    for change in ({axis: 384}, {"bits": 4}, {"codebook": "sqg_fp16"}):
        with pytest.raises(ValueError):
            tuple(generator.cases_for_tuning_queries(({**query, **change},)))


def test_gemv_configuration_records_device_geometry_without_live_row_specialization():
    from b12x.gemm.bf16_gemv._tuning import GemvQuery, heuristic, validate_config, GemvConfig

    for sms, n, k, grouping in ((188, 112, 1024, 1), (48, 112, 1024, 8), (48, 128, 2048, 4)):
        device = SimpleNamespace(sm_count=sms)
        expected = GemvConfig(backend="cutedsl", rows_per_cta=grouping)
        for rows in (1, 3, 8):
            query = GemvQuery(dtype="bfloat16", max_rows=rows, in_features=k, out_features=n)
            assert heuristic(query, device) == expected
            validate_config(query, expected, device)
            with pytest.raises(ValueError, match="row grouping"):
                validate_config(query, GemvConfig(backend="cutedsl", rows_per_cta=True), device)


def test_trellis_qualification_uses_builtin_cute_rotation(tmp_path):
    from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
    from b12x.policy.generation.providers.trellis import TrellisLinearGenerator
    from b12x.policy import DeviceIdentity

    generator = TrellisLinearGenerator()
    context = GenerationContext(device=DeviceIdentity(vendor="nvidia", product_name="synthetic",
        compute_capability=(12, 1), sm_count=48), device_ordinal=0, work_dir=tmp_path,
        source_revision="test", settings=GenerationSettings())
    estimate = generator.estimate(context)
    assert estimate.dimensions["rotation_backend"] == "cutedsl"
    assert "external_dependencies" not in estimate.dimensions
    assert estimate.case_count == 72
    assert estimate.dimensions["candidate_measurements"] == 656


def test_dsa_embedded_merge_coverage_preserves_measured_page_capacity():
    from b12x.attention.dsa_indexer._policy import DSA_INDEXER_POLICY, DsaIndexerQuery
    from b12x.policy import PolicyContext, DeviceIdentity

    device = DeviceIdentity(vendor="nvidia", product_name="nvidia rtx pro 6000 blackwell max-q workstation edition",
                            compute_capability=(12, 0), sm_count=188)
    context = PolicyContext(device=device)
    query = dict(source_layout="paged", mode="decode", dtype="bfloat16", kv_dtype="uint8",
                 num_q_heads=32, num_idx_heads=1, max_q_rows=4, max_k_rows=32768,
                 top_k=2048, page_size=64, score_mode="dsa", shared_page_table=False)
    assert DSA_INDEXER_POLICY.query_schema_version == 2
    assert context.resolve(DSA_INDEXER_POLICY, DsaIndexerQuery(**query)).config.fused_merge == "cooperative"
    assert context.resolve(DSA_INDEXER_POLICY, DsaIndexerQuery(**{**query, "max_k_rows": 32832})).config.fused_merge == "auto"
