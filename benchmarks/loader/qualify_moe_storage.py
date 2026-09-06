"""Qualify native NVFP4 MoE weights in shared memory for A4 and A16 decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x.loader import capabilities
from b12x.moe import fused_moe
from b12x.policy import MOE_DECODE, get_auto_policy
from benchmarks.loader._utils import WeightFiles, paired_times, source_identity
from benchmarks.loader.qualify_storage import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation",
        default="pinned",
        choices=["pinned", "pinned_wc", "system", "managed", "registered", "file"],
    )
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--intermediate", type=int, default=640)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.init()
    torch.manual_seed(782)
    device = torch.device("cuda", 0)
    e, h, i = args.experts, args.hidden, args.intermediate

    def scales(shape):
        return swizzle_block_scale(
            (torch.rand(shape, device=device) * 0.1 + 0.03).to(torch.float8_e4m3fn)
        )

    weights = fused_moe.PackedWeights(
        w13=torch.randint(0, 256, (e, 2 * i, h // 2), device=device, dtype=torch.uint8),
        w2=torch.randint(0, 256, (e, h, i // 2), device=device, dtype=torch.uint8),
        w13_block_scales=scales((e, 2 * i, h // 16)),
        w2_block_scales=scales((e, h, i // 16)),
        w13_global_scales=torch.rand(e, device=device) * 0.1 + 0.1,
        w2_global_scales=torch.rand(e, device=device) * 0.1 + 0.1,
        input_scale=torch.full((e,), 128.0, device=device),
        intermediate_scale=torch.full((e,), 512.0, device=device),
    )
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MODELOPT_NVFP4,
            w13_layout=fused_moe.W13Layout.W13,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=e, hidden_size=h, intermediate_size=i
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.AUTO,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
    )
    report = {
        **source_identity(),
        "command": sys.argv,
        "before": snapshot(),
        "capabilities": capabilities(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "allocation": args.allocation,
        "shape": {
            "m": args.m,
            "hidden": h,
            "intermediate": i,
            "experts": e,
            "top_k": args.top_k,
        },
        "cases": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        files = WeightFiles(directory, args.allocation)
        loaded = files.load(weights)
        prepared = [
            fused_moe.prepare_weights(plan=weight_plan, weights=w)
            for w in (weights, loaded)
        ]
        native = prepared[1]._impl.representation_for("w4a16")
        assert native.w13.data_ptr() == loaded.w13.data_ptr()
        assert native.w2.data_ptr() == loaded.w2.data_ptr()
        assert native.w13_scale.data_ptr() == loaded.w13_block_scales.data_ptr()
        assert native.w2_scale.data_ptr() == loaded.w2_block_scales.data_ptr()
        source = (torch.randn(args.m, h, device=device) * 0.25).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(e, device=device)[: args.top_k] for _ in range(args.m)]
        ).to(torch.int32)
        routes = torch.softmax(torch.randn(args.m, args.top_k, device=device), dim=-1)
        for mode, backend, route_mode in (
            ("a4", "micro", None),
            ("a16", "w4a16", "direct"),
        ):
            bindings = []
            outputs = [torch.empty_like(source), torch.empty_like(source)]
            for index, experts in enumerate(prepared):
                config = fused_moe.MoeDecodeConfig(
                    backend=backend,
                    route_planner="internal",
                    max_active_clusters=None,
                    w4a16_route_mode=route_mode,
                )
                plan = fused_moe.plan_execution(
                    experts=experts,
                    capacity=fused_moe.ExecutionCapacity(
                        max_tokens=args.m,
                        top_k=args.top_k,
                        warmup_token_counts=(args.m,),
                    ),
                    policy=get_auto_policy(device).with_override(MOE_DECODE, config),
                )
                fused_moe.prewarm(plan)
                (spec,) = plan.scratch_specs()
                scratch = torch.empty(spec.shape, device=device, dtype=spec.dtype)
                bindings.append(
                    fused_moe.bind(
                        plan,
                        scratch=scratch,
                        a=source,
                        experts=experts,
                        topk_ids=ids,
                        topk_weights=routes,
                        output=outputs[index],
                        input_scales_static=True,
                    )
                )

            def run(index):
                fused_moe.run(binding=bindings[index])

            graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
            for index in range(2):
                run(index)
                with torch.cuda.graph(graphs[index]):
                    run(index)
            for _ in range(3):
                for output in outputs:
                    output.fill_(float("nan"))
                for graph in graphs:
                    graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(outputs[0]).all() and outputs[0].count_nonzero() > 0
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            case = {
                "mode": mode,
                "exact_match": True,
                "graph_replay": True,
                **paired_times(run),
            }
            print(
                json.dumps({k: v for k, v in case.items() if k != "raw_ms"}), flush=True
            )
            report["cases"].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["persistent_shared_bytes"] = files.storage_bytes
        report["after"] = snapshot()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
