"""Offline execution contract for the CuTe small-N BF16 GEMV path."""

from dataclasses import asdict, dataclass, fields

from b12x.policy.context import ComponentPolicy
from b12x.policy.problem import define_problem


@dataclass(frozen=True, kw_only=True)
class GemvQuery:
    dtype: str
    max_rows: int
    in_features: int
    out_features: int


@dataclass(frozen=True, kw_only=True)
class GemvConfig:
    backend: str
    rows_per_cta: int


def row_group_for_geometry(n: int, k: int, sms: int) -> int:
    if n < sms:
        return 1
    # A single K step fits 128 threads carrying eight BF16 values each.
    return 8 if k <= 1024 else 4


def validate_query(query):
    if (query.dtype != "bfloat16" or not 1 <= query.max_rows <= 8
            or query.in_features < 32 or query.in_features % 32 or not 1 <= query.out_features <= 1024):
        raise ValueError("BF16 GEMV qualification requires the CuTe small-M/small-N contract")


def heuristic(query, device):
    if device is None:
        raise ValueError("BF16 GEMV launch selection requires the device SM count")
    return GemvConfig(backend="cutedsl", rows_per_cta=row_group_for_geometry(
        query.out_features, query.in_features, device.sm_count))


def validate_config(query, config, device):
    validate_query(query)
    if (not isinstance(config, GemvConfig) or type(config.rows_per_cta) is not int
            or config != heuristic(query, device)):
        raise ValueError("BF16 GEMV qualification must match the production row grouping")


EXECUTION_CONTRACT = ComponentPolicy(
    component_id="gemm.bf16_gemv", query_schema_version=1, config_schema_version=2,
    query_fields=frozenset(field.name for field in fields(GemvQuery)),
    config_fields=frozenset(field.name for field in fields(GemvConfig)), encode_query=asdict,
    decode_profile=lambda query, device, payload: GemvConfig(**dict(payload)),
    heuristic=heuristic, validate_config=validate_config,
)
TUNING_PROBLEM = define_problem(
    policy=EXECUTION_CONTRACT, query_type=GemvQuery, config_type=GemvConfig,
    axes=("max_rows", "in_features", "out_features"), family=("dtype",),
    model_fields=("in_features", "out_features"), decisions={"backend": ("cutedsl",), "rows_per_cta": (1, 4, 8)},
    axis_domains={"max_rows": (1, 1), "in_features": (32, 32), "out_features": (1, 1)},
)
