"""Inspect model-level kernel selections without allocating model weights."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from b12x.policy import (
    EMBEDDED_REGISTRY,
    ComponentPolicy,
    DeviceIdentity,
    GpuProfile,
    PolicyContext,
    PolicyResolution,
    detect_device,
)


@dataclass(frozen=True, kw_only=True)
class KernelQuery:
    scenario: str
    kernel_family: str
    policy: ComponentPolicy[Any, Any]
    query: object


@dataclass(frozen=True, kw_only=True)
class DeviceSelection:
    identity: DeviceIdentity
    runtime_device: str


@dataclass(frozen=True, kw_only=True)
class MoePreset:
    quant_mode: str
    source_format: str
    activation: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    intermediate_alignment: int
    minimum_intermediate_size: int


def _aligned_shard_size(total: int, tp_size: int, alignment: int, minimum: int) -> int:
    logical_size = (total + tp_size - 1) // tp_size
    return ((max(logical_size, minimum) + alignment - 1) // alignment) * alignment


def _moe_queries(
    tp_size: int,
    preset: MoePreset,
    *,
    scenario_prefix: str = "moe",
) -> tuple[KernelQuery, ...]:
    from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery

    if not 1 <= tp_size <= 16 or tp_size > preset.intermediate_size:
        raise ValueError("MoE benchmark presets support sliceable TP 1 through 16")
    intermediate = _aligned_shard_size(
        preset.intermediate_size,
        tp_size,
        preset.intermediate_alignment,
        preset.minimum_intermediate_size,
    )
    return tuple(
        KernelQuery(
            scenario=f"{scenario_prefix}-m{tokens}",
            kernel_family="fused-moe",
            policy=MOE_DECODE_POLICY,
            query=MoeDecodeQuery(
                quant_mode=preset.quant_mode,
                source_format=preset.source_format,
                activation=preset.activation,
                num_experts=preset.num_experts,
                hidden_size=preset.hidden_size,
                intermediate_size=intermediate,
                top_k=preset.top_k,
                num_tokens=tokens,
                routed_rows=tokens * preset.top_k,
            ),
        )
        for tokens in (1, 4, 7)
    )


def _paged_gqa_query(
    *,
    runtime_device: str,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    kv_dtype: str = "bfloat16",
    page_size: int = 64,
) -> KernelQuery:
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery

    return KernelQuery(
        scenario="full-attention-decode",
        kernel_family="paged-gqa",
        policy=GQA_POLICY,
        query=GqaQuery(
            device=runtime_device,
            mode="decode",
            q_dtype="bfloat16",
            kv_dtype=kv_dtype,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            kv_cache_layout="separate",
            batch_size=1,
            query_len=1,
            cache_tokens=65_536,
            window_left=-1,
            requested_graph_ctas_per_sm=None,
            requested_max_work_items=None,
            requested_max_partial_rows=None,
            force_split_kv=None,
        ),
    )


def _sliced_gqa_query(
    *,
    model: str,
    tp_size: int,
    runtime_device: str,
    global_q_heads: int,
    global_kv_heads: int,
    head_dim: int,
) -> KernelQuery:
    if not 1 <= tp_size <= 16 or global_q_heads % tp_size:
        raise ValueError(f"{model} does not have an integral TP {tp_size} Q-head shard")
    q_heads = global_q_heads // tp_size
    kv_heads = max(1, global_kv_heads // tp_size)
    if q_heads % kv_heads:
        raise ValueError(f"{model} TP {tp_size} does not produce a valid GQA shard")
    return _paged_gqa_query(
        runtime_device=runtime_device,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )


def _vocab_projection_query(
    *,
    model: str,
    tp_size: int,
    hidden_size: int,
    global_vocab_size: int,
) -> KernelQuery:
    from b12x.gemm.bf16_vocab_projection._policy import (
        BF16_VOCAB_PROJECTION_POLICY,
        Bf16VocabProjectionQuery,
    )

    if global_vocab_size % tp_size:
        raise ValueError(f"{model} vocabulary is not sliceable at TP {tp_size}")
    return KernelQuery(
        scenario="vocab-projection-m1",
        kernel_family="bf16-vocab-projection",
        policy=BF16_VOCAB_PROJECTION_POLICY,
        query=Bf16VocabProjectionQuery(
            dtype="bfloat16",
            max_tokens=1,
            in_features=hidden_size,
            out_features=global_vocab_size // tp_size,
        ),
    )


def _qwen_flash_next_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
    from b12x.attention.qsa._policy import QSA_POLICY, QsaQuery
    from b12x.gemm.block_fp8_linear._policy import (
        BLOCK_FP8_LINEAR_POLICY,
        BlockFp8LinearQuery,
    )
    from b12x.gemm.wo_projection._policy import (
        WO_PROJECTION_POLICY,
        WoProjectionQuery,
    )
    from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery
    from b12x.norm.hyperconnection._policy import (
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery,
    )
    from b12x.sequence.gdn_decode._policy import GDN_POLICY, GdnQuery
    from b12x.sequence.mtp_feedback._policy import (
        MTP_FEEDBACK_POLICY,
        MtpFeedbackQuery,
    )
    from b12x.sequence.ple._policy import PLE_POLICY, PleQuery
    from b12x.sequence.ple_embedding._policy import (
        PLE_EMBEDDING_POLICY,
        PleEmbeddingQuery,
    )
    from b12x.sequence.ple_hash._policy import PLE_HASH_POLICY, PleHashQuery

    if tp_size not in (1, 2, 4):
        raise ValueError(
            "qwen3.8-flash-next-180b supports TP 1, 2, or 4 in the "
            "profiled QSA contract"
        )
    q_heads = 24 // tp_size
    kv_heads = max(1, 2 // tp_size)
    gdn_key_heads = 16 // tp_size
    gdn_value_heads = 48 // tp_size
    intermediate = ((640 + tp_size - 1) // tp_size + 15) // 16 * 16
    queries = [
        _vocab_projection_query(
            model="qwen3.8-flash-next-180b",
            tp_size=tp_size,
            hidden_size=2_560,
            global_vocab_size=248_320,
        ),
        KernelQuery(
            scenario="full-attention-decode",
            kernel_family="paged-gqa",
            policy=GQA_POLICY,
            query=GqaQuery(
                device=runtime_device,
                mode="decode",
                q_dtype="bfloat16",
                kv_dtype="float8_e4m3fn",
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim_qk=256,
                head_dim_vo=256,
                page_size=128,
                kv_cache_layout="separate",
                batch_size=1,
                query_len=1,
                cache_tokens=65_536,
                window_left=-1,
                requested_graph_ctas_per_sm=None,
                requested_max_work_items=None,
                requested_max_partial_rows=None,
                force_split_kv=None,
            ),
        ),
        KernelQuery(
            scenario="qsa-spec4",
            kernel_family="qsa",
            policy=QSA_POLICY,
            query=QsaQuery(
                q_dtype="bfloat16",
                kv_dtype="float8_e4m3fn",
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=256,
                index_heads=4,
                index_kv_heads=1,
                index_head_dim=128,
                index_rotary_dim=64,
                main_page_size=16,
                max_batch=1,
                max_q_rows=4,
                max_seq_len=65_536,
                max_speculative_tokens=3,
                compress_ratio=4,
                budget=2_048,
                position_axes=3,
                mrope_interleaved=True,
            ),
        ),
        KernelQuery(
            scenario="gdn-spec4",
            kernel_family="gdn",
            policy=GDN_POLICY,
            query=GdnQuery(
                gate_activation="sigmoid",
                qk_l2norm=True,
                state_dtype="float32",
                key_heads=gdn_key_heads,
                value_heads=gdn_value_heads,
                max_seqs=4,
                max_tokens=16,
                state_index_columns=4,
            ),
        ),
        KernelQuery(
            scenario="attention-output",
            kernel_family="wo-projection",
            policy=WO_PROJECTION_POLICY,
            query=WoProjectionQuery(
                dtype="bfloat16",
                max_tokens=4,
                groups=q_heads,
                group_width=512,
                rank=512,
                hidden=2_560,
            ),
        ),
        KernelQuery(
            scenario="mxfp8-linear",
            kernel_family="block-fp8-linear",
            policy=BLOCK_FP8_LINEAR_POLICY,
            query=BlockFp8LinearQuery(
                max_tokens=4,
                in_features=2_560,
                out_features=2_560,
                output_dtype="bfloat16",
            ),
        ),
    ]
    for tokens in (1, 4, 7):
        queries.append(
            KernelQuery(
                scenario=f"moe-m{tokens}",
                kernel_family="fused-moe",
                policy=MOE_DECODE_POLICY,
                query=MoeDecodeQuery(
                    quant_mode="nvfp4",
                    source_format="modelopt_nvfp4",
                    activation="silu",
                    num_experts=512,
                    hidden_size=2_560,
                    intermediate_size=intermediate,
                    top_k=10,
                    num_tokens=tokens,
                    routed_rows=tokens * 10,
                ),
            )
        )
    queries.extend(
        (
            KernelQuery(
                scenario="residual-spec4",
                kernel_family="hyperconnection",
                policy=HYPERCONNECTION_POLICY,
                query=HyperConnectionQuery(
                    dtype="bfloat16",
                    max_tokens=4,
                    hidden_size=2_560,
                    streams=4,
                    lowrank=320,
                ),
            ),
            KernelQuery(
                scenario="mtp-feedback-spec4",
                kernel_family="mtp-feedback",
                policy=MTP_FEEDBACK_POLICY,
                query=MtpFeedbackQuery(
                    dtype="bfloat16",
                    max_tokens=4,
                    hidden_size=2_560,
                    streams=4,
                ),
            ),
            KernelQuery(
                scenario="ple-spec4",
                kernel_family="ple",
                policy=PLE_POLICY,
                query=PleQuery(
                    mode="decode",
                    dtype="bfloat16",
                    max_tokens=4,
                    max_seqs=1,
                    max_speculative_tokens=3,
                    streams=4,
                    hidden_size=2_560,
                    kernel_size=4,
                    dilation=3,
                ),
            ),
            KernelQuery(
                scenario="ple-hash-spec4",
                kernel_family="ple-hash",
                policy=PLE_HASH_POLICY,
                query=PleHashQuery(
                    max_tokens=4,
                    max_seqs=1,
                    vocab_size=248_320,
                    max_order=3,
                    heads_per_order=8,
                    base_table_size=20_000_000,
                ),
            ),
            KernelQuery(
                scenario="ple-embedding-spec4",
                kernel_family="ple-embedding",
                policy=PLE_EMBEDDING_POLICY,
                query=PleEmbeddingQuery(
                    quant_mode="nvfp4_group16",
                    table_memory="mapped_host",
                    output_dtype="bfloat16",
                    max_tokens=4,
                    max_seqs=1,
                    vocab_size=248_320,
                    max_order=3,
                    heads_per_order=8,
                    base_table_size=20_000_000,
                    embedding_dim=2_560,
                    tp_size=tp_size,
                ),
            ),
        )
    )
    return tuple(queries)


def _qwen_dense_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.quantization.nvfp4._policy import (
        NVFP4_QUANTIZATION_POLICY,
        Nvfp4QuantizationQuery,
    )

    if tp_size not in (1, 2, 4, 8):
        raise ValueError("qwen3.8-27b supports profiled TP 1, 2, 4, or 8")
    attention = _paged_gqa_query(
        runtime_device=runtime_device,
        q_heads=24 // tp_size,
        kv_heads=max(1, 4 // tp_size),
        head_dim=256,
        kv_dtype="float8_e4m3fn",
        page_size=128,
    )
    return (
        _vocab_projection_query(
            model="qwen3.8-27b",
            tp_size=tp_size,
            hidden_size=5_120,
            global_vocab_size=248_320,
        ),
        attention,
        KernelQuery(
            scenario="nvfp4-activation-block",
            kernel_family="nvfp4-quantization",
            policy=NVFP4_QUANTIZATION_POLICY,
            query=Nvfp4QuantizationQuery(
                dtype="bfloat16",
                rows=128,
                columns=5_120,
            ),
        ),
    )


_MOE_PRESETS = {
    "qwen3.5-397b-a17b": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=512,
        hidden_size=4_096,
        intermediate_size=1_024,
        top_k=10,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "nvidia-nemotron-3-super-120b": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="relu2",
        num_experts=512,
        hidden_size=1_024,
        intermediate_size=2_688,
        top_k=22,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "nvidia-nano3.5": MoePreset(
        quant_mode="w4a16",
        source_format="modelopt_nvfp4",
        activation="relu2",
        num_experts=128,
        hidden_size=2_688,
        intermediate_size=1_856,
        top_k=6,
        intermediate_alignment=64,
        minimum_intermediate_size=64,
    ),
    "dsv4f": MoePreset(
        quant_mode="w4a16",
        source_format="fp4_e8m0_k32",
        activation="silu",
        num_experts=256,
        hidden_size=6_144,
        intermediate_size=2_048,
        top_k=8,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "dsv4f-nvfp4": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=256,
        hidden_size=6_144,
        intermediate_size=2_048,
        top_k=8,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "minimax-m3": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="swigluoai_uninterleave",
        num_experts=128,
        hidden_size=6_144,
        intermediate_size=3_072,
        top_k=4,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "laguna-s2.1": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=256,
        hidden_size=3_072,
        intermediate_size=1_024,
        top_k=10,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "deepseek-v4-flash": MoePreset(
        quant_mode="w4a16",
        source_format="fp4_e8m0_k32",
        activation="silu",
        num_experts=256,
        hidden_size=4_096,
        intermediate_size=2_048,
        top_k=6,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "glm-5.1": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=256,
        hidden_size=6_144,
        intermediate_size=2_048,
        top_k=8,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "glm-5.3": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=256,
        hidden_size=6_144,
        intermediate_size=2_048,
        top_k=8,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "glm-5.3-mtp": MoePreset(
        quant_mode="w4a16",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=256,
        hidden_size=6_144,
        intermediate_size=2_048,
        top_k=8,
        intermediate_alignment=64,
        minimum_intermediate_size=64,
    ),
    "glm-5.3-flash": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=288,
        hidden_size=4_096,
        intermediate_size=2_048,
        top_k=8,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "glm-5.3-flash-mtp": MoePreset(
        quant_mode="w4a16",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=288,
        hidden_size=4_096,
        intermediate_size=2_048,
        top_k=8,
        intermediate_alignment=64,
        minimum_intermediate_size=64,
    ),
    "minimax-m2.7": MoePreset(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=256,
        hidden_size=3_072,
        intermediate_size=1_536,
        top_k=8,
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    ),
    "kimi-k3": MoePreset(
        quant_mode="w4a16",
        source_format="b12x_trellis",
        activation="situ",
        num_experts=896,
        hidden_size=3_584,
        intermediate_size=3_072,
        top_k=16,
        intermediate_alignment=256,
        minimum_intermediate_size=256,
    ),
}


def _moe_only_factory(model: str):
    def factory(
        tp_size: int,
        *,
        runtime_device: str,
    ) -> tuple[KernelQuery, ...]:
        del runtime_device
        return _moe_queries(tp_size, _MOE_PRESETS[model])

    return factory


def _minimax_m27_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    attention = _sliced_gqa_query(
        model="minimax-m2.7",
        tp_size=tp_size,
        runtime_device=runtime_device,
        global_q_heads=48,
        global_kv_heads=8,
        head_dim=128,
    )
    return (attention, *_moe_queries(tp_size, _MOE_PRESETS["minimax-m2.7"]))


def _minimax_m3_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.attention.dsa_indexer._policy import (
        DSA_INDEXER_POLICY,
        DsaIndexerQuery,
    )

    attention = _sliced_gqa_query(
        model="minimax-m3",
        tp_size=tp_size,
        runtime_device=runtime_device,
        global_q_heads=64,
        global_kv_heads=4,
        head_dim=128,
    )
    indexer = KernelQuery(
        scenario="msa-indexer-decode",
        kernel_family="msa-indexer",
        policy=DSA_INDEXER_POLICY,
        query=DsaIndexerQuery(
            source_layout="paged",
            mode="decode",
            dtype="float8_e4m3fn",
            kv_dtype="uint8",
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=4,
            max_k_rows=0,
            top_k=16,
            page_size=64,
            score_mode="msa",
            shared_page_table=False,
        ),
    )
    return (
        attention,
        indexer,
        *_moe_queries(tp_size, _MOE_PRESETS["minimax-m3"]),
    )


def _qwen_gqa_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    if tp_size != 1:
        raise ValueError("the synthetic qwen-gqa benchmark preset is rank-local TP1")
    return (
        _paged_gqa_query(
            runtime_device=runtime_device,
            q_heads=8,
            kv_heads=1,
            head_dim=256,
        ),
    )


def _kimi_k3_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.attention.dense_mla._policy import DENSE_MLA_POLICY, DenseMlaQuery

    del runtime_device
    if tp_size != 12:
        raise ValueError("the reviewed Kimi-K3 dense-MLA preset is TP12")
    attention = KernelQuery(
        scenario="dense-mla-decode",
        kernel_family="dense-mla",
        policy=DENSE_MLA_POLICY,
        query=DenseMlaQuery(
            mode="decode",
            q_dtype="float8_e4m3fn",
            kv_dtype="float8_e4m3fn",
            num_q_heads=8,
            qk_head_dim=576,
            v_head_dim=512,
            page_size=944,
            query_rows=1,
            max_batch=1,
            cache_tokens=65_536,
            physical_record_width=576,
            window_size=None,
            use_cuda_graph=True,
        ),
    )
    return (attention, *_moe_queries(tp_size, _MOE_PRESETS["kimi-k3"]))


def _deepseek_v4_flash_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.attention.compressed_sparse_mla._policy import (
        COMPRESSED_SPARSE_MLA_POLICY,
        SparseMlaQuery,
    )
    from b12x.attention.dsa_indexer._policy import (
        DSA_INDEXER_POLICY,
        DsaIndexerQuery,
    )

    del runtime_device
    if tp_size not in (1, 2, 4, 8):
        raise ValueError("deepseek-v4-flash supports profiled TP 1, 2, 4, or 8")
    local_heads = 64 // tp_size
    queries = [
        KernelQuery(
            scenario="paged-indexer-decode",
            kernel_family="dsa-indexer",
            policy=DSA_INDEXER_POLICY,
            query=DsaIndexerQuery(
                source_layout="paged",
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=local_heads,
                num_idx_heads=1,
                max_q_rows=1,
                max_k_rows=0,
                top_k=512,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
        )
    ]
    for contract, indexed_width, indexed_page_size in (
        ("swa", 0, 64),
        ("swa-c4", 512, 64),
        ("swa-c128", 512, 2),
    ):
        queries.append(
            KernelQuery(
                scenario=f"compressed-mla-{contract}",
                kernel_family="compressed-sparse-mla",
                policy=COMPRESSED_SPARSE_MLA_POLICY,
                query=SparseMlaQuery(
                    layout="compressed_dsv4",
                    mode="decode",
                    q_dtype="bfloat16",
                    kv_dtype="float8_e4m3fn",
                    num_q_heads=local_heads,
                    qk_head_dim=512,
                    v_head_dim=448,
                    swa_width=128,
                    swa_page_size=64,
                    indexed_width=indexed_width,
                    indexed_page_size=indexed_page_size,
                    query_rows=1,
                ),
            )
        )
    queries.extend(_moe_queries(tp_size, _MOE_PRESETS["deepseek-v4-flash"]))
    return tuple(queries)

def _validate_glm_tp(model: str, tp_size: int) -> None:
    if tp_size not in (1, 2, 4, 8):
        raise ValueError(f"{model} supports profiled TP 1, 2, 4, or 8")


def _glm52_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.attention.dsa_indexer._policy import (
        DSA_INDEXER_POLICY,
        DsaIndexerQuery,
    )
    from b12x.attention.sparse_mla._policy import (
        SPARSE_MLA_POLICY,
        SparseMlaQuery,
    )
    from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery

    del runtime_device
    _validate_glm_tp("glm-5.2", tp_size)
    local_heads = 64 // tp_size
    intermediate = ((2_048 + tp_size - 1) // tp_size + 31) // 32 * 32
    queries = [
        _vocab_projection_query(
            model="glm-5.2",
            tp_size=tp_size,
            hidden_size=6_144,
            global_vocab_size=163_840 if tp_size == 8 else 163_968,
        ),
        KernelQuery(
            scenario="dsa-decode-spec4",
            kernel_family="dsa-indexer",
            policy=DSA_INDEXER_POLICY,
            query=DsaIndexerQuery(
                source_layout="paged",
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=32,
                num_idx_heads=1,
                max_q_rows=4,
                max_k_rows=0,
                top_k=2_048,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
        ),
        KernelQuery(
            scenario="sparse-mla-spec4",
            kernel_family="sparse-mla",
            policy=SPARSE_MLA_POLICY,
            query=SparseMlaQuery(
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=local_heads,
                qk_head_dim=576,
                v_head_dim=512,
                max_q_rows=4,
                max_width=2_048,
                page_size=64,
                model_type=None,
                head_major_output=False,
            ),
        ),
    ]
    for tokens in (1, 4, 7):
        queries.append(
            KernelQuery(
                scenario=f"moe-m{tokens}",
                kernel_family="fused-moe",
                policy=MOE_DECODE_POLICY,
                query=MoeDecodeQuery(
                    quant_mode="w4a8_nvfp4",
                    source_format="modelopt_nvfp4",
                    activation="silu",
                    num_experts=256,
                    hidden_size=6_144,
                    intermediate_size=intermediate,
                    top_k=8,
                    num_tokens=tokens,
                    routed_rows=tokens * 8,
                ),
            )
        )
    return tuple(queries)


def _glm51_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    attention = tuple(
        item
        for item in _glm52_queries(tp_size, runtime_device=runtime_device)
        if item.policy.component_id != "moe.decode"
    )
    return (*attention, *_moe_queries(tp_size, _MOE_PRESETS["glm-5.1"]))


def _glm53_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    _validate_glm_tp("glm-5.3", tp_size)
    attention = tuple(
        item
        for item in _glm52_queries(tp_size, runtime_device=runtime_device)
        if item.policy.component_id.startswith("attention.")
    )
    return (
        _vocab_projection_query(
            model="glm-5.3",
            tp_size=tp_size,
            hidden_size=6_144,
            global_vocab_size=154_880,
        ),
        *attention,
        *_moe_queries(
            tp_size,
            _MOE_PRESETS["glm-5.3"],
            scenario_prefix="main-moe",
        ),
        *_moe_queries(
            tp_size,
            _MOE_PRESETS["glm-5.3-mtp"],
            scenario_prefix="mtp-moe",
        ),
    )


def _glm53_flash_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.attention._shared.mla.traits import ModelType
    from b12x.attention.dsa_indexer._policy import (
        DSA_INDEXER_POLICY,
        DsaIndexerQuery,
    )
    from b12x.attention.sparse_mla._policy import (
        SPARSE_MLA_POLICY,
        SparseMlaQuery,
    )
    from b12x.norm.mhc._policy import MHC_POLICY, MhcQuery
    from b12x.sequence.gdn_decode._policy import GDN_POLICY, GdnQuery

    del runtime_device
    _validate_glm_tp("glm-5.3-flash", tp_size)
    local_heads = 64 // tp_size
    queries = [
        _vocab_projection_query(
            model="glm-5.3-flash",
            tp_size=tp_size,
            hidden_size=4_096,
            global_vocab_size=154_880,
        ),
        KernelQuery(
            scenario="kda-spec6",
            kernel_family="kda",
            policy=GDN_POLICY,
            query=GdnQuery(
                gate_activation="sigmoid",
                qk_l2norm=True,
                state_dtype="float32",
                key_heads=local_heads,
                value_heads=local_heads,
                max_seqs=4,
                max_tokens=24,
                state_index_columns=6,
            ),
        ),
        KernelQuery(
            scenario="pooled-indexer-spec6",
            kernel_family="dsa-indexer",
            policy=DSA_INDEXER_POLICY,
            query=DsaIndexerQuery(
                source_layout="paged",
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=32,
                num_idx_heads=1,
                max_q_rows=6,
                max_k_rows=0,
                top_k=512,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
        ),
        KernelQuery(
            scenario="sparse-mla-spec6",
            kernel_family="sparse-mla",
            policy=SPARSE_MLA_POLICY,
            query=SparseMlaQuery(
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=local_heads,
                qk_head_dim=512,
                v_head_dim=512,
                max_q_rows=6,
                max_width=2_051,
                page_size=256,
                model_type=ModelType.GLM_NEXT,
                head_major_output=False,
            ),
        ),
        KernelQuery(
            scenario="mhc-spec6",
            kernel_family="mhc",
            policy=MHC_POLICY,
            query=MhcQuery(
                dtype="bfloat16",
                max_tokens=6,
                hidden_size=4_096,
                split_k=64,
            ),
        ),
    ]
    queries.extend(
        _moe_queries(
            tp_size,
            _MOE_PRESETS["glm-5.3-flash"],
            scenario_prefix="main-moe",
        )
    )
    queries.extend(
        _moe_queries(
            tp_size,
            _MOE_PRESETS["glm-5.3-flash-mtp"],
            scenario_prefix="mtp-moe",
        )
    )
    return tuple(queries)


_MODEL_FACTORIES = {
    "deepseek-v4-flash": _deepseek_v4_flash_queries,
    "dsv4f": _moe_only_factory("dsv4f"),
    "dsv4f-nvfp4": _moe_only_factory("dsv4f-nvfp4"),
    "glm-5.1": _glm51_queries,
    "glm-5.2": _glm52_queries,
    "glm-5.3": _glm53_queries,
    "glm-5.3-flash": _glm53_flash_queries,
    "kimi-k3": _kimi_k3_queries,
    "laguna-s2.1": _moe_only_factory("laguna-s2.1"),
    "minimax-m2.7": _minimax_m27_queries,
    "minimax-m3": _minimax_m3_queries,
    "nvidia-nano3.5": _moe_only_factory("nvidia-nano3.5"),
    "nvidia-nemotron-3-super-120b": _moe_only_factory(
        "nvidia-nemotron-3-super-120b"
    ),
    "qwen-gqa": _qwen_gqa_queries,
    "qwen3.5-397b-a17b": _moe_only_factory("qwen3.5-397b-a17b"),
    "qwen3.8-flash-next-180b": _qwen_flash_next_queries,
    "qwen3.8-27b": _qwen_dense_queries,
}
_MODEL_ALIASES = {
    "glm5.1": "glm-5.1",
    "glm51": "glm-5.1",
    "glm5.2": "glm-5.2",
    "glm52": "glm-5.2",
    "glm5.3": "glm-5.3",
    "glm5.3-flash": "glm-5.3-flash",
    "glm53": "glm-5.3",
    "glm53-flash": "glm-5.3-flash",
    "glm53-flash-shape": "glm-5.3-flash",
    "qwen3.8-flash-next": "qwen3.8-flash-next-180b",
    "qwen38-flash-next": "qwen3.8-flash-next-180b",
    "qwen38-flash-next-shape": "qwen3.8-flash-next-180b",
    "qwen38-flash-next-180b": "qwen3.8-flash-next-180b",
    "qwen38-27b": "qwen3.8-27b",
    "qwen397b": "qwen3.5-397b-a17b",
    "nemotron-backbone": "nvidia-nemotron-3-super-120b",
    "nano35-w4a16": "nvidia-nano3.5",
    "nano35-w4a16-shape": "nvidia-nano3.5",
    "minimax-m3-shape": "minimax-m3",
    "laguna-s21": "laguna-s2.1",
    "laguna-s21-shape": "laguna-s2.1",
    "minimax-m27": "minimax-m2.7",
    "minimax-m2": "minimax-m2.7",
    "mimimax-m2.7": "minimax-m2.7",
}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _canonical_model(value: str) -> str:
    key = value.casefold()
    canonical = _MODEL_ALIASES.get(key, key)
    if canonical in _MODEL_FACTORIES:
        return canonical
    needle = _normalize_name(value)
    normalized = {
        _normalize_name(name): name for name in _MODEL_FACTORIES
    }
    for alias, target in _MODEL_ALIASES.items():
        normalized[_normalize_name(alias)] = target
    canonical = normalized.get(needle)
    if canonical is not None:
        return canonical
    choices = ", ".join(sorted(_MODEL_FACTORIES))
    raise ValueError(f"unknown model preset {value!r}; choose one of {choices}")


def _detected_selection(device: str | None) -> DeviceSelection | None:
    detected = detect_device(device)
    if detected.identity is None or detected.ordinal is None:
        return None
    return DeviceSelection(
        identity=detected.identity,
        runtime_device=f"cuda:{detected.ordinal}",
    )


def _profile_selection(profile: GpuProfile) -> DeviceSelection:
    detected = _detected_selection(None)
    if detected is not None and detected.identity in profile.targets:
        return detected
    return DeviceSelection(identity=profile.targets[0], runtime_device="cuda:0")


def _device_selection(value: str) -> DeviceSelection:
    if value.casefold() == "auto":
        selected = _detected_selection(None)
        if selected is None:
            raise ValueError(
                "--device auto requires a CUDA-enabled PyTorch interpreter"
            )
        return selected

    if value.isdecimal():
        selected = _detected_selection(f"cuda:{value}")
        if selected is None:
            raise ValueError(
                f"--device {value} did not find CUDA device ordinal {value}"
            )
        return selected

    needle = _normalize_name(value)
    profiles = EMBEDDED_REGISTRY.list_profiles()
    exact_profiles = [
        profile
        for profile in profiles
        if needle == _normalize_name(profile.profile_id)
    ]
    partial_profiles = [
        profile
        for profile in profiles
        if needle and needle in _normalize_name(profile.profile_id)
    ]
    matched_profiles = exact_profiles or partial_profiles
    if len(matched_profiles) == 1:
        return _profile_selection(matched_profiles[0])
    if len(matched_profiles) > 1:
        matched = ", ".join(profile.profile_id for profile in matched_profiles)
        raise ValueError(f"ambiguous profile name {value!r}; matches {matched}")

    exact_targets: list[tuple[object, DeviceIdentity]] = []
    partial_targets: list[tuple[object, DeviceIdentity]] = []
    for profile in profiles:
        for target in profile.targets:
            normalized = _normalize_name(target.product_name)
            if needle == normalized:
                exact_targets.append((profile, target))
            elif needle and needle in normalized:
                partial_targets.append((profile, target))
    matches = exact_targets or partial_targets
    unique = {
        (profile.profile_id, target): (profile, target)
        for profile, target in matches
    }
    if len(unique) != 1:
        choices = ", ".join(
            profile.profile_id for profile in EMBEDDED_REGISTRY.list_profiles()
        )
        if not unique:
            raise ValueError(
                f"unknown embedded device/profile {value!r}; choose one of {choices}"
            )
        matched = ", ".join(
            sorted(target.product_name for _profile, target in unique.values())
        )
        raise ValueError(f"ambiguous device name {value!r}; matches {matched}")
    _key, (_profile, identity) = next(iter(unique.items()))
    detected = detect_device()
    ordinal = (
        detected.ordinal
        if detected.identity == identity and detected.ordinal is not None
        else 0
    )
    return DeviceSelection(identity=identity, runtime_device=f"cuda:{ordinal}")


def _config_dict(config: object) -> dict[str, object]:
    if is_dataclass(config):
        return asdict(config)
    for method_name in ("to_dict", "profile_dict"):
        method = getattr(config, method_name, None)
        if callable(method):
            payload = method()
            if isinstance(payload, dict):
                return payload
    raise TypeError(f"cannot serialize policy config {type(config).__name__}")


def _kernel_name(item: KernelQuery, config: dict[str, object]) -> str:
    backend = config.get("backend")
    if item.policy.component_id == "moe.decode":
        return f"{backend}/{config['route_planner']}"
    return str(backend) if backend is not None else item.kernel_family


def _record(
    item: KernelQuery,
    resolution: PolicyResolution[Any],
) -> dict[str, object]:
    config = _config_dict(resolution.config)
    return {
        "scenario": item.scenario,
        "component": item.policy.component_id,
        "kernel": _kernel_name(item, config),
        "source": resolution.source.value,
        "profile_id": resolution.profile_id,
        "rule": resolution.rule_name,
        "query": dict(item.policy.encode_query(item.query)),
        "config": config,
    }


def inspect_model_policy(
    model: str,
    *,
    tp_size: int,
    device: str,
) -> dict[str, object]:
    if tp_size <= 0:
        raise ValueError("TP size must be positive")
    canonical = _canonical_model(model)
    selected = _device_selection(device)
    context = PolicyContext.for_identity(selected.identity)
    queries = _MODEL_FACTORIES[canonical](
        tp_size,
        runtime_device=selected.runtime_device,
    )
    return {
        "model": canonical,
        "tp_size": tp_size,
        "device": {
            "product_name": selected.identity.product_name,
            "compute_capability": list(selected.identity.compute_capability),
            "sm_count": selected.identity.sm_count,
        },
        "profile_id": context.profile_id,
        "selections": [
            _record(item, context.resolve(item.policy, item.query))
            for item in queries
        ],
    }


def _render_table(console: Console, payload: dict[str, object]) -> None:
    table = Table(
        title=(
            f"{payload['model']} TP={payload['tp_size']} on "
            f"{payload['device']['product_name']}"
        )
    )
    table.add_column("Scenario")
    table.add_column("Component")
    table.add_column("Kernel")
    table.add_column("Source")
    table.add_column("Rule")
    table.add_column("Config")
    for selection in payload["selections"]:
        table.add_row(
            str(selection["scenario"]),
            str(selection["component"]),
            str(selection["kernel"]),
            str(selection["source"]),
            str(selection["rule"] or "-"),
            json.dumps(selection["config"], sort_keys=True, separators=(",", ":")),
        )
    console.print(table)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the model-level kernels selected by embedded GPU profiles, "
            "including heuristic fallbacks."
        )
    )
    parser.add_argument("model", nargs="?", help="model preset name")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "auto, a CUDA ordinal, an embedded profile ID, or a device-name "
            "fragment"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_models:
        for model in sorted(_MODEL_FACTORIES):
            print(model)
        return 0
    if args.list_devices:
        for profile in EMBEDDED_REGISTRY.list_profiles():
            names = ", ".join(target.product_name for target in profile.targets)
            print(f"{profile.profile_id}: {names}")
        return 0
    if args.model is None:
        parser.error("model is required unless a --list option is used")
    try:
        payload = inspect_model_policy(
            args.model,
            tp_size=args.tp,
            device=args.device,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_table(Console(), payload)
    return 0


__all__ = ["inspect_model_policy", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
