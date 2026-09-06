#!/usr/bin/env python3
"""Compare ordered QSA split reductions using balanced warm-L2 graph replay."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b12x.attention.paged import _selected_forward as implementation
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
    source = Path(implementation.__file__)
    result = {
        "status": "research-only",
        "measurement_scope": "proxy: isolated merge kernel; end-to-end serving is not measured",
        "command": sys.argv,
        "worktree": str(Path(__file__).resolve().parents[1]),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True,
        ).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "cases": [],
        "timing": "warm-L2 merge only; ABBA, 32 launches/replay, 60 samples/arm",
    }
    original = implementation._SparseGqaMergeKernel
    serving_kernel = implementation._MERGE_KERNEL
    try:
        for heads in (6, 12, 24):
            for rows in (1, 4, 16, 64):
                for splits in (1, 16, 32, 64):
                    partials = torch.randn(rows, splits, heads, 256, device="cuda")
                    lse = torch.randn(rows, splits, heads, device="cuda") * 8
                    lse[0, :, 0] = -torch.inf
                    output = torch.empty(
                        rows, heads, 256, device="cuda", dtype=torch.bfloat16
                    )

                    def launch():
                        implementation.launch_sparse_gqa_merge(
                            partial_output=partials,
                            partial_lse=lse,
                            output=output,
                            rows=rows,
                            splits=splits,
                        )

                    graphs = {}
                    kernel_names = {}
                    for label, kernel in (
                        ("serial", original),
                        (
                            "cooperative",
                            implementation._ShapeAdaptiveSparseGqaMergeKernel,
                        ),
                    ):
                        implementation._MERGE_KERNEL = kernel
                        implementation.clear_caches()
                        launch()
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profile:
                            launch()
                            torch.cuda.synchronize()
                        names = [
                            event.name
                            for event in profile.events()
                            if event.device_type.name == "CUDA"
                        ]
                        assert any(
                            "CooperativeSparseGqaMergeKernel" in name for name in names
                        ) == (label == "cooperative" and rows <= splits), names
                        kernel_names[label] = names
                        if label == "serial":
                            expected = output.clone()
                        else:
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)

                        def repeated():
                            for _ in range(32):
                                launch()

                        graphs[label] = capture_cuda_graph(repeated, warmup=2)
                    samples = {label: [] for label in graphs}
                    for label in ("serial", "cooperative", "cooperative", "serial"):
                        samples[label].extend(
                            v / 32
                            for v in bench_cuda_graph(graphs[label], replays=30)[
                                "replay_us"
                            ]
                        )
                    medians = {
                        label: statistics.median(v) for label, v in samples.items()
                    }
                    row = {
                        "correctness": "passed",
                        "rtol": 0,
                        "atol": 0,
                        "heads": heads,
                        "rows": rows,
                        "splits": splits,
                        "median_us": medians,
                        "kernel_names": kernel_names,
                        "samples_us": samples,
                        "serial_over_cooperative_ratio": medians["serial"]
                        / medians["cooperative"],
                    }
                    result["cases"].append(row)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                        flush=True,
                    )
    finally:
        implementation._MERGE_KERNEL = serving_kernel
        implementation.clear_caches()
    result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
