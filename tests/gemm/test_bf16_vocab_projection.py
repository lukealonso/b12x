from __future__ import annotations

import pytest
import torch

from b12x.gemm.bf16_vocab_projection._policy import (
    BF16_VOCAB_PROJECTION_POLICY,
    Bf16VocabProjectionConfig,
    Bf16VocabProjectionQuery,
)
from b12x.policy import BF16_VOCAB_PROJECTION, DeviceIdentity

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_unknown_device_heuristic_uses_torch() -> None:
    query = Bf16VocabProjectionQuery(
        dtype="bfloat16",
        max_tokens=1,
        in_features=2_560,
        out_features=248_320,
    )
    device = DeviceIdentity(
        vendor="nvidia",
        compute_capability=(9, 0),
        sm_count=120,
        product_name="Synthetic GPU",
    )

    config = BF16_VOCAB_PROJECTION_POLICY.heuristic(query, device)

    assert config.backend == "torch"


@cuda_required
def test_planned_projection_matches_reference_and_replays_graph() -> None:
    from b12x.gemm import bf16_vocab_projection as projection
    from b12x.policy import PolicyContext, PolicyMode

    torch.manual_seed(4)
    device = torch.device("cuda")
    source = torch.randn(1, 256, device=device, dtype=torch.bfloat16)
    weight = torch.randn(4_096, 256, device=device, dtype=torch.bfloat16)
    planned = projection.plan(
        projection.Caps(
            device=device,
            max_tokens=1,
            in_features=256,
            out_features=4_096,
        ),
        policy=PolicyContext.for_device(
            device, mode=PolicyMode.HEURISTIC_ONLY
        ).with_override(
            BF16_VOCAB_PROJECTION,
            Bf16VocabProjectionConfig(
                backend="triton",
                algorithm="row",
                block_k=256,
                num_warps=8,
            ),
        ),
    )
    binding = projection.bind(planned, source=source, weight=weight)
    expected = torch.nn.functional.linear(source, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = projection.run(binding)
    actual.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
    assert planned.scratch_specs() == ()
