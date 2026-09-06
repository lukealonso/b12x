"""Offline family, shape, and launch-decision contract for Trellis linear."""

from dataclasses import asdict, dataclass, fields

from b12x.policy.context import ComponentPolicy
from b12x.policy.problem import define_problem


@dataclass(frozen=True, kw_only=True)
class TrellisQuery:
    max_rows: int
    in_features: int
    out_features: int
    input_dtype: str
    compute_dtype: str
    codebook: str
    bits: int
    weight_layout: str = "native"


@dataclass(frozen=True, kw_only=True)
class TrellisConfig:
    backend: str
    block_rows: int
    tile_k: int
    tile_n: int

    @classmethod
    def from_profile(cls, payload):
        return cls(**dict(payload))


def validate_query(query):
    if (query.max_rows <= 0 or min(query.in_features, query.out_features) <= 0
            or query.in_features % 128 or query.out_features % 128
            or query.input_dtype not in ("float16", "bfloat16")
            or query.compute_dtype not in ("float16", "bfloat16")
            or query.weight_layout not in ("native", "p24_k", "p24_n", "p33_k", "p33_n")):
        raise ValueError("Trellis qualification requires positive BF16/FP16 rows, 128-aligned N/K, and a supported layout")
    from b12x.moe._shared.trellis_codebooks import normalize_codebook, validate_codebook_bits
    if normalize_codebook(query.codebook) != query.codebook or query.bits not in (2, 3, 4, 5, 6):
        raise ValueError("Trellis queries require a canonical codebook and supported native rate")
    validate_codebook_bits(query.codebook, query.bits)
    if query.weight_layout != "native":
        axis = query.weight_layout[-1]
        size = query.in_features if axis == "k" else query.out_features
        if query.bits != 3 or size != 256 or query.codebook not in ("mcg", "sqg_e4m3"):
            raise ValueError("compact Trellis pairs require three stored bits per weight, a 256-channel rate axis, and MCG or SQG E4M3")


def validate_config(query, config, device):
    from b12x.moe._shared.kernels.w4a16.kernel import _candidate_tile_fits, _w4a16_num_regs

    validate_query(query)
    if (not isinstance(config, TrellisConfig) or config.backend != "cutedsl"
            or config.block_rows not in (16, 32, 48, 64)
            or config.tile_k not in (64, 128) or config.tile_n not in (64, 128, 256)):
        raise ValueError("invalid native Trellis launch decision")
    if query.weight_layout.endswith("_n") and config.tile_n != 256:
        raise ValueError("N-axis Trellis pairs require a complete 256-column tile")
    threads = config.tile_n * config.tile_k // 64
    if not _candidate_tile_fits(problem_n=query.out_features, problem_k=query.in_features,
                               cta_m_blocks=config.block_rows // 16, tile_n=config.tile_n,
                               tile_k=config.tile_k, cta_threads=threads, max_shared_mem=101_376 - 512,
                               scale_format="e4m3_k32", weight_layout="trellis_t256", weight_bits=max(4, query.bits)):
        raise ValueError("native Trellis tile exceeds shape or shared-memory limits")
    _w4a16_num_regs(cta_threads=threads, cta_m_blocks=config.block_rows // 16,
                   cta_n_blocks=config.tile_n // 16, cta_k_blocks=config.tile_k // 16,
                   uses_m_block_8=False, weight_layout="trellis_t256")


def heuristic(query, device):
    from b12x.moe._shared.kernels.w4a16.kernel import _trellis256_dense_tile_config

    tile_k, tile_n = _trellis256_dense_tile_config(query.in_features, query.out_features)
    return TrellisConfig(backend="cutedsl", block_rows=64, tile_k=tile_k, tile_n=tile_n)


EXECUTION_CONTRACT = ComponentPolicy(
    component_id="gemm.trellis_linear", query_schema_version=2, config_schema_version=1,
    query_fields=frozenset(field.name for field in fields(TrellisQuery)),
    config_fields=frozenset(field.name for field in fields(TrellisConfig)), encode_query=asdict,
    decode_profile=lambda query, device, payload: TrellisConfig.from_profile(payload),
    heuristic=heuristic, validate_config=validate_config,
)
TUNING_PROBLEM = define_problem(
    policy=EXECUTION_CONTRACT, query_type=TrellisQuery, config_type=TrellisConfig,
    axes=("max_rows", "in_features", "out_features"),
    family=("input_dtype", "compute_dtype", "codebook", "bits", "weight_layout"),
    model_fields=("in_features", "out_features"),
    decisions={"backend": ("cutedsl",), "block_rows": (16, 32, 48, 64),
               "tile_k": (64, 128), "tile_n": (64, 128, 256)},
    axis_domains={"max_rows": (1, 1), "in_features": (128, 128), "out_features": (128, 128)},
)
