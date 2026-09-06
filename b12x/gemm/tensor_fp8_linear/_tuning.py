"""Offline execution contract for tensor-scaled E4M3 linear projection."""

from dataclasses import dataclass

from b12x.policy.fixed_backend import BackendConfig, make_fixed_backend_policy
from b12x.policy.problem import define_problem


@dataclass(frozen=True, kw_only=True)
class TensorFp8Query:
    max_rows: int
    in_features: int
    out_features: int
    output_dtype: str


def validate_query(query):
    if (min(query.max_rows, query.in_features, query.out_features) <= 0 or query.in_features % 32
            or query.output_dtype not in ("bfloat16", "float16")):
        raise ValueError("tensor-FP8 qualification requires positive dimensions, K32, and BF16/FP16 output")


EXECUTION_CONTRACT = make_fixed_backend_policy(component_id="gemm.tensor_fp8_linear", query_type=TensorFp8Query,
                                               backend="cutedsl", validate_query=validate_query)
TUNING_PROBLEM = define_problem(
    policy=EXECUTION_CONTRACT, query_type=TensorFp8Query, config_type=BackendConfig,
    axes=("max_rows", "in_features", "out_features"), family=("output_dtype",),
    model_fields=("in_features", "out_features"), decisions={"backend": ("cutedsl",)},
    axis_domains={name: (1, 1) for name in ("max_rows", "in_features", "out_features")},
)
