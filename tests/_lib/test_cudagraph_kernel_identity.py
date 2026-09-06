from __future__ import annotations

import hashlib
import json
from pathlib import Path

from validation.trellis_decode.cudagraph_kernel_identity import analyze_receipt


def _dot_node(node_id: int, grid_x: int, *, cooperative: int = 1) -> str:
    return f""""graph_1_node_{node_id}"[style="bold" shape="record" label="{{KERNEL
| {{ID | {node_id} (topoId: {10 - node_id}) | kernel_cutlass_kernel_b12xgemmtrellis_linear_k6_mcg_cuteK6McgSmallMKernel_object_at_test\\<\\<\\<{grid_x},256,68608\\>\\>\\>}}
| {{cooperative | {cooperative}}}
}}"];
"""


def _multidim_dot_node() -> str:
    return """"graph_1_node_2"[style="bold" shape="record" label="{KERNEL
| {ID | 2 (topoId: 8) | unrelated_kernel\\<\\<\\<\\{40,4\\},\\{32,16\\},0\\>\\>\\>}
| {cooperative | 0}
}"];
"""


def _write_receipt(
    tmp_path: Path,
    *,
    cooperative: int = 1,
    runtime_route: str = "b12x_k6_mcg_small_m",
    include_inactive_fp16_plan: bool = False,
    malformed_runtime_launch: bool = False,
) -> Path:
    receipt = tmp_path / "receipt"
    graphs = receipt / "graphs"
    graphs.mkdir(parents=True)
    dot_path = graphs / "graph.dot"
    dot_path.write_text(
        "digraph dot {\n"
        + _dot_node(0, 64, cooperative=cooperative)
        + _dot_node(1, 80)
        + _multidim_dot_node()
        + "}\n"
    )
    dot_bytes = dot_path.read_bytes()
    dump = {
        "schema": "b12x.cudagraph.kernel_dump.v1",
        "channel_id": "vllm:target:production",
        "batch_descriptor": (
            "BatchExecutionDescriptor(cg_mode=<CUDAGraphMode.FULL: 2>, "
            "num_tokens=4, num_reqs=1, uniform_token_count=4, "
            "max_req_tokens=None, num_active_loras=0)"
        ),
        "path": "/tmp/b12x-cudagraph-dumps/graph.dot",
        "sha256": hashlib.sha256(dot_bytes).hexdigest(),
        "bytes": len(dot_bytes),
    }
    records = []
    plan_route = f"{runtime_route}_planned"
    for projection, grid_x in (("layer.0.down_proj", 64), ("layer.1.down_proj", 80)):
        records.append(
            {
                "phase": "plan",
                "model_role": "target_or_shared",
                "projection": projection,
                "shard": "None",
                "bound_k6_mcg_small_m_launch": True,
                "reason": "bound_cooperative_launch",
                "selected_route": plan_route,
            }
        )
        if include_inactive_fp16_plan:
            records.append(
                {
                    "phase": "plan",
                    "model_role": "target_or_shared",
                    "projection": projection,
                    "shard": "None",
                    "bound_k6_mcg_small_m_launch": True,
                    "reason": "bound_cooperative_launch",
                    "selected_route": "b12x_k6_mcg_small_m_planned",
                }
            )
        records.append(
            {
                "phase": "runtime",
                "model_role": "target_or_shared",
                "projection": projection,
                "m": 4,
                "cuda_graph_capture": True,
                "selected_route": runtime_route,
                "launch": (
                    {} if malformed_runtime_launch else {"launch_grid_x": grid_x}
                ),
            }
        )
    log_lines = [
        f"B12X_TRELLIS_REACHABILITY {json.dumps(record)}" for record in records
    ]
    log_lines.append(f"B12X_CUDAGRAPH_KERNEL_DUMP {json.dumps(dump)}")
    (receipt / "service.log").write_text("\n".join(log_lines) + "\n")
    return receipt


def test_analyze_receipt_correlates_exact_cooperative_launches(
    tmp_path: Path,
) -> None:
    report = analyze_receipt(_write_receipt(tmp_path))

    assert report["status"] == "pass"
    assert report["summary"]["one_cooperative_launch_per_bound_projection"]
    assert report["summary"]["all_k6_mcg_nodes_cooperative"]
    graph = report["graphs"][0]
    assert graph["k6_mcg_node_count"] == 2
    assert graph["k6_mcg_grid_x_counts"] == {"64": 1, "80": 1}


def test_analyze_receipt_pairs_bf16_runtime_with_only_its_active_plan(
    tmp_path: Path,
) -> None:
    report = analyze_receipt(
        _write_receipt(
            tmp_path,
            runtime_route="b12x_k6_mcg_small_m_bf16",
            include_inactive_fp16_plan=True,
        )
    )

    assert report["status"] == "pass"
    assert report["summary"]["bound_projection_counts"] == {"target_or_shared": 2}
    assert report["summary"]["active_fused_runtime_routes"] == [
        "b12x_k6_mcg_small_m_bf16"
    ]


def test_analyze_receipt_fails_closed_for_noncooperative_k6_node(
    tmp_path: Path,
) -> None:
    report = analyze_receipt(_write_receipt(tmp_path, cooperative=0))

    assert report["status"] == "fail"
    assert not report["summary"]["all_k6_mcg_nodes_cooperative"]
    assert report["graphs"][0]["k6_mcg_noncooperative_nodes"]


def test_analyze_receipt_fails_closed_for_malformed_runtime_launch(
    tmp_path: Path,
) -> None:
    report = analyze_receipt(
        _write_receipt(tmp_path, malformed_runtime_launch=True)
    )

    assert report["status"] == "fail"
    graph = report["graphs"][0]
    assert graph["status"] == "fail"
    assert not graph["checks"]["launch_grid_multiset_matches_runtime"]
    assert any(
        "invalid launch.launch_grid_x" in error for error in graph["errors"]
    )
