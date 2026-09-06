#!/usr/bin/env python3
"""Measure HC projection staging plus consumers against row-strided views."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b12x.norm.hyperconnection import _cute, _kernels
from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    nvidia_smi_gpu_mode_snapshot,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(20260905)
    result = {
        "status": "research-only",
        "command": sys.argv,
        "source_sha256": {
            str(Path(module.__file__).name): hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (_cute, _kernels)
        },
        "timing": "warm-L2; staging + scaled SiLU + combine/norm; ABBA; 60 samples/arm",
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "cases": [],
    }
    for rows in (1, 4, 16, 128, 4096, 6019):
        streams, lowrank, hidden = 4, 320, 2560

        def randn(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        merged = randn(rows, 336)
        down = merged[:, :lowrank]
        injection = merged[:, lowrank : lowrank + streams]
        staged_down, staged_injection = (
            torch.empty_like(down),
            torch.empty_like(injection),
        )
        state, block = randn(rows, streams * hidden), randn(rows, hidden)
        weights = randn(streams * hidden)
        bottleneck = torch.empty_like(down)
        combined, normalized = torch.empty_like(state), torch.empty_like(state)

        def launch_staged():
            staged_down.copy_(down)
            staged_injection.copy_(injection)
            _kernels._scaled_silu_launch(staged_down, bottleneck, streams, 256)
            _cute.combine_norm(
                state,
                block,
                staged_injection,
                weights,
                combined,
                normalized,
                eps=1e-6,
                streams=streams,
                hidden_size=hidden,
            )

        def launch_views():
            _cute.scaled_silu(down, bottleneck, streams=streams)
            _cute.combine_norm(
                state,
                block,
                injection,
                weights,
                combined,
                normalized,
                eps=1e-6,
                streams=streams,
                hidden_size=hidden,
            )

        launch_staged()
        expected = [x.clone() for x in (bottleneck, combined, normalized)]
        launch_views()
        for actual, oracle in zip(
            (bottleneck, combined, normalized), expected, strict=True
        ):
            torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
        del expected
        graphs = {}
        names = {}
        launches = 32 if rows <= 128 else 2
        for label, launch in (("staged", launch_staged), ("views", launch_views)):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                launch()
                torch.cuda.synchronize()
            names[label] = [
                x.name for x in profile.events() if x.device_type.name == "CUDA"
            ]
            assert any("_ScaledSilu" in name for name in names[label]) == (
                label == "views"
            )

            def repeated():
                for _ in range(launches):
                    launch()

            graphs[label] = capture_cuda_graph(repeated, warmup=2)
        samples = {label: [] for label in graphs}
        for label in ("staged", "views", "views", "staged"):
            samples[label].extend(
                value / launches
                for value in bench_cuda_graph(graphs[label], replays=30)["replay_us"]
            )
        medians = {
            label: statistics.median(values) for label, values in samples.items()
        }
        case = {
            "rows": rows,
            "median_us": medians,
            "samples_us": samples,
            "kernel_names": names,
            "bit_exact": True,
            "staged_over_views_ratio": medians["staged"] / medians["views"],
        }
        result["cases"].append(case)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in case.items()
                    if k not in ("samples_us", "kernel_names")
                }
            ),
            flush=True,
        )
    result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
