"""Stable component identifiers used by generated GPU profiles."""

COMPRESSED_SPARSE_MLA_ATTENTION = "attention.compressed_sparse_mla"
DSA_INDEXER = "attention.dsa_indexer"
GDN_ATTENTION = "attention.gdn"
GQA_ATTENTION = "attention.gqa"
MLA_ATTENTION = "attention.mla"
MOE_DECODE = "moe.decode"
QSA_ATTENTION = "attention.qsa"
SPARSE_MLA_ATTENTION = "attention.sparse_mla"
VARLEN_ATTENTION = "attention.varlen"
BLOCKSCALED_PRECISION = "gemm.blockscaled_precision"
BLOCK_FP8_LINEAR = "gemm.block_fp8_linear"
BF16_VOCAB_PROJECTION = "gemm.bf16_vocab_projection"
WO_PROJECTION = "gemm.wo_projection"
EP_MOE = "moe.ep_moe"
HYPERCONNECTION = "norm.hyperconnection"
KDA_PREFILL = "sequence.kda_prefill"
MHC = "norm.mhc"
NVFP4_QUANTIZATION = "quantization.nvfp4"
MTP_FEEDBACK = "sequence.mtp_feedback"
PLE = "sequence.ple"
PLE_EMBEDDING = "sequence.ple_embedding"
PLE_HASH = "sequence.ple_hash"

__all__ = [
    "BLOCKSCALED_PRECISION",
    "BLOCK_FP8_LINEAR",
    "BF16_VOCAB_PROJECTION",
    "COMPRESSED_SPARSE_MLA_ATTENTION",
    "DSA_INDEXER",
    "EP_MOE",
    "GDN_ATTENTION",
    "GQA_ATTENTION",
    "HYPERCONNECTION",
    "KDA_PREFILL",
    "MLA_ATTENTION",
    "MHC",
    "MOE_DECODE",
    "MTP_FEEDBACK",
    "NVFP4_QUANTIZATION",
    "PLE",
    "PLE_EMBEDDING",
    "PLE_HASH",
    "QSA_ATTENTION",
    "SPARSE_MLA_ATTENTION",
    "VARLEN_ATTENTION",
    "WO_PROJECTION",
]
