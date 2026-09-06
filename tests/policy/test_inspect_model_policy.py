from __future__ import annotations

import pytest

from b12x.policy import DetectedDevice, EMBEDDED_REGISTRY
from b12x.policy.generation.attention_corpus import (
    ATTENTION_BENCHMARK_PRESETS,
    GQA_GEOMETRIES,
)
from b12x.policy.generation.moe_corpus import (
    MOE_BENCHMARK_PRESETS,
    MOE_RECIPES,
)
from b12x.policy.generation.providers.gemm import _bf16_vocab_projection_cases
from b12x.tools.inspect_model_policy import (
    _MODEL_FACTORIES,
    _canonical_model,
    _deepseek_v4_flash_queries,
    _device_selection,
    _glm52_queries,
    _glm53_queries,
    _glm53_flash_queries,
    _minimax_m27_queries,
    _minimax_m3_queries,
    _qwen_dense_queries,
    _qwen_flash_next_queries,
    _qwen_gqa_queries,
    inspect_model_policy,
    main,
)


def test_qwen_flash_next_preset_slices_rank_local_tp_geometry() -> None:
    selections = {
        item.scenario: item
        for item in _qwen_flash_next_queries(4, runtime_device="cuda:0")
    }

    qsa = selections["qsa-spec4"].query
    gdn = selections["gdn-spec4"].query
    moe = selections["moe-m4"].query
    vocab = selections["vocab-projection-m1"].query
    assert (qsa.q_heads, qsa.kv_heads) == (6, 1)
    assert (gdn.key_heads, gdn.value_heads) == (4, 12)
    assert (gdn.max_seqs, gdn.max_tokens, gdn.state_index_columns) == (4, 16, 4)
    assert moe.intermediate_size == 160
    assert moe.routed_rows == 40
    assert (vocab.in_features, vocab.out_features) == (2_560, 62_080)


def test_qwen_flash_next_preset_rejects_unprofiled_qsa_tp() -> None:
    with pytest.raises(ValueError, match="TP 1, 2, or 4"):
        _qwen_flash_next_queries(8, runtime_device="cuda:0")


def test_model_aliases_are_canonicalized() -> None:
    assert _canonical_model("glm52") == "glm-5.2"
    assert _canonical_model("GLM 5.2") == "glm-5.2"
    assert _canonical_model("glm53") == "glm-5.3"
    assert _canonical_model("GLM 5.3") == "glm-5.3"
    assert _canonical_model("glm5.3-flash") == "glm-5.3-flash"
    assert _canonical_model("GLM 5.3 Flash") == "glm-5.3-flash"
    assert _canonical_model("glm53-flash-shape") == "glm-5.3-flash"
    assert _canonical_model("qwen38-flash-next") == "qwen3.8-flash-next-180b"
    assert _canonical_model("Qwen 3.8 Flash Next") == (
        "qwen3.8-flash-next-180b"
    )
    assert _canonical_model("qwen38-27b") == "qwen3.8-27b"
    assert _canonical_model("nemotron-backbone") == (
        "nvidia-nemotron-3-super-120b"
    )
    assert _canonical_model("minimax-m27") == "minimax-m2.7"


def test_every_moe_benchmark_preset_spelling_is_accepted() -> None:
    recipes = {recipe.recipe_id: recipe for recipe in MOE_RECIPES}
    for preset in MOE_BENCHMARK_PRESETS:
        canonical = _canonical_model(preset.preset_id)
        assert canonical in _MODEL_FACTORIES
        recipe = recipes[preset.recipe_id]
        query = next(
            item.query
            for item in _MODEL_FACTORIES[canonical](
                preset.tp_size,
                runtime_device="cuda:0",
            )
            if item.policy.component_id == "moe.decode"
            and item.query.quant_mode == recipe.quant_mode
            and item.query.source_format == recipe.source_format
        )
        assert (query.quant_mode, query.source_format) == (
            recipe.quant_mode,
            recipe.source_format,
        )


def test_every_attention_benchmark_model_is_accepted() -> None:
    benchmark_tp = {
        "deepseek-v4-flash": 4,
        "glm-5.1": 8,
        "glm-5.2": 8,
        "kimi-k3": 12,
        "minimax-m2.7": 2,
        "minimax-m3": 2,
        "qwen-gqa": 1,
        "qwen3.8-27b": 1,
        "qwen3.8-flash-next-180b": 1,
    }
    for preset in ATTENTION_BENCHMARK_PRESETS:
        canonical = _canonical_model(preset.model_id)
        assert canonical in _MODEL_FACTORIES
        queries = _MODEL_FACTORIES[canonical](
            benchmark_tp[canonical],
            runtime_device="cuda:0",
        )
        assert preset.component in {item.policy.component_id for item in queries}


@pytest.mark.parametrize(
    ("factory", "tp_sizes"),
    (
        (_qwen_flash_next_queries, (1, 2, 4)),
        (_qwen_dense_queries, (1, 2, 4, 8)),
        (_minimax_m27_queries, (1, 2, 3, 4, 6, 8, 12, 16)),
        (_minimax_m3_queries, (1, 2, 4, 8, 16)),
        (_qwen_gqa_queries, (1,)),
    ),
)
def test_every_inspector_gqa_shape_is_in_the_profile_corpus(
    factory,
    tp_sizes: tuple[int, ...],
) -> None:
    profiled = {
        (geometry.q_heads, geometry.kv_heads, geometry.head_dim)
        for geometry in GQA_GEOMETRIES
    }

    for tp_size in tp_sizes:
        query = next(
            item.query
            for item in factory(tp_size, runtime_device="cuda:0")
            if item.policy.component_id == "attention.gqa"
        )
        assert (query.q_heads, query.kv_heads, query.head_dim_qk) in profiled


def test_embedded_device_can_be_selected_by_product_fragment() -> None:
    selected = _device_selection("gb10")

    assert selected.identity.product_name == "nvidia gb10"
    assert selected.identity.sm_count == 48


def test_multi_target_profile_id_uses_matching_detected_device(monkeypatch) -> None:
    profile = EMBEDDED_REGISTRY.get("nvidia.rtx.pro.6000.blackwell")
    identity = profile.targets[1]
    monkeypatch.setattr(
        "b12x.tools.inspect_model_policy.detect_device",
        lambda device=None: DetectedDevice(ordinal=4, identity=identity),
    )

    selected = _device_selection(profile.profile_id)

    assert selected.identity == identity
    assert selected.runtime_device == "cuda:4"


def test_auto_device_selects_detected_cuda_device(monkeypatch) -> None:
    identity = EMBEDDED_REGISTRY.get(
        "nvidia.rtx.pro.6000.blackwell"
    ).targets[0]
    requested: list[object | None] = []

    def detect(device=None):
        requested.append(device)
        return DetectedDevice(ordinal=3, identity=identity)

    monkeypatch.setattr("b12x.tools.inspect_model_policy.detect_device", detect)

    selected = _device_selection("auto")

    assert requested == [None]
    assert selected.identity == identity
    assert selected.runtime_device == "cuda:3"


def test_numeric_device_selects_cuda_ordinal(monkeypatch) -> None:
    identity = EMBEDDED_REGISTRY.get(
        "nvidia.rtx.pro.6000.blackwell"
    ).targets[0]
    requested: list[object | None] = []

    def detect(device=None):
        requested.append(device)
        return DetectedDevice(ordinal=7, identity=identity)

    monkeypatch.setattr("b12x.tools.inspect_model_policy.detect_device", detect)

    selected = _device_selection("7")

    assert requested == ["cuda:7"]
    assert selected.identity == identity
    assert selected.runtime_device == "cuda:7"


def test_cli_lists_model_presets(capsys) -> None:
    assert main(["--list-models"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "deepseek-v4-flash",
        "dsv4f",
        "dsv4f-nvfp4",
        "glm-5.1",
        "glm-5.2",
        "glm-5.3",
        "glm-5.3-flash",
        "kimi-k3",
        "laguna-s2.1",
        "minimax-m2.7",
        "minimax-m3",
        "nvidia-nano3.5",
        "nvidia-nemotron-3-super-120b",
        "qwen-gqa",
        "qwen3.5-397b-a17b",
        "qwen3.8-27b",
        "qwen3.8-flash-next-180b",
    ]


def test_glm52_preset_uses_benchmark_attention_and_moe_contracts() -> None:
    selections = {
        item.scenario: item for item in _glm52_queries(8, runtime_device="cuda:0")
    }

    indexer = selections["dsa-decode-spec4"].query
    sparse_mla = selections["sparse-mla-spec4"].query
    moe = selections["moe-m4"].query
    vocab = selections["vocab-projection-m1"].query
    assert (indexer.num_q_heads, indexer.top_k) == (32, 2_048)
    assert (sparse_mla.num_q_heads, sparse_mla.qk_head_dim) == (8, 576)
    assert (moe.quant_mode, moe.intermediate_size, moe.top_k) == (
        "w4a8_nvfp4",
        256,
        8,
    )
    assert (vocab.in_features, vocab.out_features) == (6_144, 20_480)


def test_glm53_flash_preset_composes_kda_sparse_mla_mhc_and_moe() -> None:
    selections = {
        item.scenario: item
        for item in _glm53_flash_queries(4, runtime_device="cuda:0")
    }

    kda = selections["kda-spec6"].query
    indexer = selections["pooled-indexer-spec6"].query
    sparse_mla = selections["sparse-mla-spec6"].query
    mhc = selections["mhc-spec6"].query
    main_moe = selections["main-moe-m4"].query
    mtp_moe = selections["mtp-moe-m4"].query
    vocab = selections["vocab-projection-m1"].query
    assert (kda.key_heads, kda.value_heads) == (16, 16)
    assert (kda.max_seqs, kda.max_tokens, kda.state_index_columns) == (4, 24, 6)
    assert (indexer.num_q_heads, indexer.top_k) == (32, 512)
    assert (sparse_mla.qk_head_dim, sparse_mla.v_head_dim) == (512, 512)
    assert int(sparse_mla.model_type) == 2
    assert (mhc.hidden_size, mhc.split_k) == (4_096, 64)
    assert (main_moe.quant_mode, main_moe.intermediate_size, main_moe.top_k) == (
        "nvfp4",
        512,
        8,
    )
    assert (mtp_moe.quant_mode, mtp_moe.intermediate_size, mtp_moe.top_k) == (
        "w4a16",
        512,
        8,
    )
    assert (vocab.in_features, vocab.out_features) == (4_096, 38_720)


def test_glm53_preset_composes_dsa_sparse_mla_main_and_mtp_moe() -> None:
    selections = {
        item.scenario: item
        for item in _glm53_queries(8, runtime_device="cuda:0")
    }

    indexer = selections["dsa-decode-spec4"].query
    sparse_mla = selections["sparse-mla-spec4"].query
    main_moe = selections["main-moe-m4"].query
    mtp_moe = selections["mtp-moe-m4"].query
    vocab = selections["vocab-projection-m1"].query
    assert (indexer.num_q_heads, indexer.top_k) == (32, 2_048)
    assert (sparse_mla.num_q_heads, sparse_mla.qk_head_dim) == (8, 576)
    assert (main_moe.quant_mode, main_moe.intermediate_size, main_moe.top_k) == (
        "nvfp4",
        256,
        8,
    )
    assert (mtp_moe.quant_mode, mtp_moe.intermediate_size, mtp_moe.top_k) == (
        "w4a16",
        256,
        8,
    )
    assert (vocab.in_features, vocab.out_features) == (6_144, 19_360)


def test_glm53_vocab_projection_corpus_uses_checkpoint_vocabulary() -> None:
    cases = _bf16_vocab_projection_cases()
    glm53_cases = tuple(
        case
        for case in cases
        if case.metadata["model_id"] in {"glm-5.3", "glm-5.3-flash"}
    )

    assert {case.metadata["global_vocab_size"] for case in glm53_cases} == {
        154_880
    }
    assert {
        (case.query["in_features"], case.query["out_features"])
        for case in glm53_cases
        if case.metadata["tp_size"] == 8
    } == {(4_096, 19_360), (6_144, 19_360)}


def test_deepseek_v4_flash_preset_composes_indexer_sparse_mla_and_moe() -> None:
    selections = _deepseek_v4_flash_queries(2, runtime_device="cuda:0")

    assert [item.scenario for item in selections] == [
        "paged-indexer-decode",
        "compressed-mla-swa",
        "compressed-mla-swa-c4",
        "compressed-mla-swa-c128",
        "moe-m1",
        "moe-m4",
        "moe-m7",
    ]
    assert selections[1].query.num_q_heads == 32
    assert selections[-1].query.top_k == 6


def test_minimax_m3_preset_includes_paged_attention_msa_and_moe() -> None:
    selections = _minimax_m3_queries(4, runtime_device="cuda:0")

    assert [item.policy.component_id for item in selections] == [
        "attention.gqa",
        "attention.dsa_indexer",
        "moe.decode",
        "moe.decode",
        "moe.decode",
    ]
    assert selections[0].query.q_heads == 16
    assert selections[1].query.score_mode == "msa"


@pytest.mark.parametrize(
    "factory", (_glm52_queries, _glm53_queries, _glm53_flash_queries)
)
def test_glm_presets_reject_unqualified_four_head_attention_shards(factory) -> None:
    with pytest.raises(ValueError, match="TP 1, 2, 4, or 8"):
        factory(16, runtime_device="cuda:0")


@pytest.mark.parametrize("tp_size", (1, 2, 4))
def test_qwen_flash_next_inspection_is_fully_preplanned_on_gb10(
    tp_size: int,
) -> None:
    payload = inspect_model_policy(
        "qwen3.8-flash-next-180b",
        tp_size=tp_size,
        device="gb10",
    )

    assert payload["profile_id"] == "nvidia.gb10.48sm"
    assert {selection["source"] for selection in payload["selections"]} == {
        "preplanned"
    }


@pytest.mark.parametrize("tp_size", (1, 2, 4, 8))
def test_qwen_dense_inspection_is_fully_preplanned_on_gb10(
    tp_size: int,
) -> None:
    payload = inspect_model_policy(
        "qwen3.8-27b",
        tp_size=tp_size,
        device="gb10",
    )

    assert payload["profile_id"] == "nvidia.gb10.48sm"
    assert {selection["source"] for selection in payload["selections"]} == {
        "preplanned"
    }


@pytest.mark.parametrize("model", ("glm-5.2", "glm-5.3", "glm-5.3-flash"))
@pytest.mark.parametrize("tp_size", (1, 2, 4, 8))
def test_glm_inspection_reports_packed_a16_heuristics_on_gb10(
    model: str,
    tp_size: int,
) -> None:
    payload = inspect_model_policy(model, tp_size=tp_size, device="gb10")

    assert payload["profile_id"] == "nvidia.gb10.48sm"
    _assert_profile_coverage_with_uniform_nvfp4_a16_heuristics(payload)


@pytest.mark.parametrize(
    ("model", "tp_size"),
    (
        ("deepseek-v4-flash", 4),
        ("dsv4f", 2),
        ("dsv4f-nvfp4", 2),
        ("glm-5.1", 8),
        ("glm-5.2", 8),
        ("glm-5.3", 8),
        ("glm-5.3-flash", 1),
        ("kimi-k3", 12),
        ("laguna-s2.1", 1),
        ("minimax-m2.7", 2),
        ("minimax-m3", 2),
        ("nvidia-nano3.5", 1),
        ("nvidia-nemotron-3-super-120b", 1),
        ("qwen-gqa", 1),
        ("qwen3.5-397b-a17b", 4),
        ("qwen3.8-27b", 1),
        ("qwen3.8-flash-next-180b", 1),
    ),
)
def test_canonical_model_profile_coverage_at_its_benchmark_tp(
    model: str,
    tp_size: int,
) -> None:
    payload = inspect_model_policy(model, tp_size=tp_size, device="gb10")

    _assert_profile_coverage_with_uniform_nvfp4_a16_heuristics(payload)


def _assert_profile_coverage_with_uniform_nvfp4_a16_heuristics(payload):
    for selection in payload["selections"]:
        query = selection["query"]
        uniform_nvfp4_a16 = (
            query.get("quant_mode") == "w4a16"
            and query.get("source_format") == "modelopt_nvfp4"
        )
        assert selection["source"] == ("heuristic" if uniform_nvfp4_a16 else "preplanned")
