"""Offline execution contract for MLA query projection and RoPE assembly."""

from dataclasses import asdict, dataclass, fields

from b12x.policy.context import ComponentPolicy
from b12x.policy.fixed_backend import BackendConfig
from b12x.policy.problem import define_problem


@dataclass(frozen=True, kw_only=True)
class ProjectionQuery:
    heads: int
    max_rows: int
    weight_format: str
    output_dtype: str


def validate_query(query):
    if (query.heads not in ((8, 16) if query.weight_format == "mxfp8" else (8, 11, 16))
            or not 1 <= query.max_rows <= 32 or query.weight_format not in ("bf16", "mxfp8")
            or query.output_dtype not in ("bfloat16", "float8_e4m3fn")):
        raise ValueError("MLA query qualification requires a supported 192-to-512 projection and 64-wide RoPE suffix")


def execution_backend(query):
    return "triton" if query.weight_format == "bf16" else "cutedsl"


def _validate(query, config, device):
    validate_query(query)
    if not isinstance(config, BackendConfig) or config.backend != execution_backend(query):
        raise ValueError("MLA query backend must match the production weight-format dispatch")


EXECUTION_CONTRACT = ComponentPolicy(
    component_id="gemm.mla_query_projection", query_schema_version=1, config_schema_version=1,
    query_fields=frozenset(field.name for field in fields(ProjectionQuery)), config_fields=frozenset({"backend"}),
    encode_query=asdict, decode_profile=lambda query, device, payload: BackendConfig.from_profile(payload),
    heuristic=lambda query, device: BackendConfig(backend=execution_backend(query)), validate_config=_validate,
)
TUNING_PROBLEM = define_problem(
    policy=EXECUTION_CONTRACT, query_type=ProjectionQuery, config_type=BackendConfig,
    axes=("heads", "max_rows"), family=("weight_format", "output_dtype"),
    model_fields=("heads",), decisions={"backend": ("cutedsl", "triton")},
    axis_domains={"heads": (1, 1), "max_rows": (1, 1)},
)
