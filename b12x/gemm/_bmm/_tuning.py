"""Offline execution contract for native rowwise-MXFP8 BMM."""

from dataclasses import dataclass

from b12x.policy.fixed_backend import BackendConfig, make_fixed_backend_policy
from b12x.policy.problem import define_problem


@dataclass(frozen=True, kw_only=True)
class BmmQuery:
    batch: int
    max_rows: int
    in_features: int
    out_features: int
    b_major: str


def validate_query(query):
    from b12x.gemm._shared.mxfp8_bmm import can_implement

    if not can_implement(batch=query.batch, max_m=query.max_rows, n=query.out_features,
                         k=query.in_features, b_major=query.b_major, sf_axis=query.b_major):
        raise ValueError("BMM qualification requires a supported BF16 x MXFP8 geometry")


EXECUTION_CONTRACT = make_fixed_backend_policy(component_id="gemm.bmm", query_type=BmmQuery,
                                               backend="cutedsl", validate_query=validate_query)
TUNING_PROBLEM = define_problem(
    policy=EXECUTION_CONTRACT, query_type=BmmQuery, config_type=BackendConfig,
    axes=("batch", "max_rows", "in_features", "out_features"), family=("b_major",),
    model_fields=("batch", "in_features", "out_features"), decisions={"backend": ("cutedsl",)},
    axis_domains={name: (1, 1) for name in ("batch", "max_rows", "in_features", "out_features")},
)
