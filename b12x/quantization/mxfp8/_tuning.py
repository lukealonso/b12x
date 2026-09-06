"""Offline execution contract for per-32 MXFP8 activation quantization."""

from dataclasses import dataclass

from b12x.policy.fixed_backend import BackendConfig, make_fixed_backend_policy
from b12x.policy.problem import define_problem


@dataclass(frozen=True, kw_only=True)
class Mxfp8Query:
    max_rows: int
    columns: int
    dtype: str
    value_order: str


def validate_query(query):
    if (min(query.max_rows, query.columns) <= 0 or query.columns % 128
            or query.dtype not in ("bfloat16", "float16")
            or query.value_order not in ("linear", "trellis_native_mma")):
        raise ValueError("MXFP8 qualification requires BF16/FP16 rows and 128-aligned columns")


EXECUTION_CONTRACT = make_fixed_backend_policy(component_id="quantization.mxfp8", query_type=Mxfp8Query,
                                               backend="cutedsl", validate_query=validate_query)
TUNING_PROBLEM = define_problem(
    policy=EXECUTION_CONTRACT, query_type=Mxfp8Query, config_type=BackendConfig,
    axes=("max_rows", "columns"), family=("dtype", "value_order"), model_fields=("columns",),
    decisions={"backend": ("cutedsl",)}, axis_domains={"max_rows": (1, 1), "columns": (128, 128)},
)
