#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the B12X project
"""Benchmark paired MXFP8 projections used by Qwen3.8 Flash Next GDN.

The Qwen Gated DeltaNet input path projects one BF16 activation matrix to a
wide QKVZ output and a narrow BA output. This benchmark compares two unchanged
``mxfp8_linear.mm`` calls with ``mxfp8_linear.mm_pair`` under CUDA-graph
replay. Both arms use the same activation and packed weights, and exact output
parity is required before timing results are written.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.gemm import mxfp8_linear


def _command_output(command: list[str]) -> str:
    """Return stripped stdout from a successful command."""
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _packed_weight(rows: int, columns: int) -> object:
    """Create a deterministic finite MXFP8 weight for timing and parity."""
    values = torch.ones(
        (rows, columns),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    scale = torch.full(
        (rows, columns // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    return mxfp8_linear.pack_weight(values, scale)


def _capture(operation: Callable[[], tuple[torch.Tensor, torch.Tensor]]):
    """Capture one operation and retain its graph-owned output tensors."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = operation()
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def _elapsed_ms(operation: Callable[[], None]) -> float:
    """Measure GPU elapsed time through the caller-stream completion point."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _measure_shape(
    tokens: int,
    *,
    input_width: int,
    qkvz_width: int,
    ba_width: int,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    """Validate and time one Qwen decode row count."""
    source = torch.randn(
        (tokens, input_width),
        dtype=torch.bfloat16,
        device="cuda",
    )
    qkvz_weight = _packed_weight(qkvz_width, input_width)
    ba_weight = _packed_weight(ba_width, input_width)
    secondary_stream = torch.cuda.Stream()

    def serial() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            mxfp8_linear.mm(source, qkvz_weight, expected_m=tokens),
            mxfp8_linear.mm(source, ba_weight, expected_m=tokens),
        )

    def paired() -> tuple[torch.Tensor, torch.Tensor]:
        return mxfp8_linear.mm_pair(
            source,
            qkvz_weight,
            ba_weight,
            expected_m=tokens,
            parallel_max_tokens=1023,
            secondary_stream=secondary_stream,
        )

    # Compile every kernel before capture. CUDA graph construction must not
    # include a CuTe JIT miss.
    serial_expected = serial()
    paired_actual = paired()
    torch.cuda.synchronize()
    for expected, actual in zip(serial_expected, paired_actual, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    serial_graph, serial_outputs = _capture(serial)
    paired_graph, paired_outputs = _capture(paired)
    for expected, actual in zip(serial_outputs, paired_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    for _ in range(warmups):
        serial_graph.replay()
        paired_graph.replay()
    torch.cuda.synchronize()

    timings: dict[str, list[float]] = {"serial_ms": [], "paired_ms": []}
    for sample in range(samples):
        arms = (
            (("serial_ms", serial_graph), ("paired_ms", paired_graph))
            if sample % 2 == 0
            else (("paired_ms", paired_graph), ("serial_ms", serial_graph))
        )
        for name, graph in arms:
            timings[name].append(_elapsed_ms(graph.replay))

    serial_median = statistics.median(timings["serial_ms"])
    paired_median = statistics.median(timings["paired_ms"])
    return {
        "tokens": tokens,
        "input_shape": [tokens, input_width],
        "qkvz_weight_shape": [qkvz_width, input_width],
        "ba_weight_shape": [ba_width, input_width],
        "correctness": "bit-identical",
        "serial_ms": timings["serial_ms"],
        "paired_ms": timings["paired_ms"],
        "serial_median_ms": serial_median,
        "paired_median_ms": paired_median,
        "speedup_ratio_serial_over_paired": serial_median / paired_median,
        "latency_reduction_percent": 100.0
        * (serial_median - paired_median)
        / serial_median,
    }


def main() -> None:
    """Parse the benchmark contract, execute it, and write JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--input-width", type=int, default=2560)
    parser.add_argument("--qkvz-width", type=int, default=16384)
    parser.add_argument("--ba-width", type=int, default=96)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--operating-mode", required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    repository = pathlib.Path(__file__).resolve().parents[1]
    try:
        revision = _command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
        status = _command_output(["git", "-C", str(repository), "status", "--short"])
        worktree_state = "clean" if not status else "modified"
    except subprocess.CalledProcessError:
        revision = os.environ.get("B12X_BENCHMARK_SOURCE_REVISION", "unavailable")
        status = os.environ.get("B12X_BENCHMARK_WORKTREE_STATUS", "")
        worktree_state = os.environ.get("B12X_BENCHMARK_WORKTREE_STATE", "unavailable")
    gpu = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,pstate,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()[torch.cuda.current_device()]
    result = {
        "schema_version": 1,
        "kind": "qwen3.8-flash-next-gdn-mxfp8-projection-pair",
        "command": shlex.join(sys.argv),
        "source": {
            "repository": str(repository),
            "candidate_revision": revision,
            "serial_implementation_baseline_revision": args.baseline_revision,
            "worktree_state": worktree_state,
            "worktree_status": status.splitlines(),
        },
        "hardware": {
            "visible_cuda_device": torch.cuda.current_device(),
            "physical_gpu_index": args.physical_gpu_index,
            "nvidia_smi_record": gpu,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "operating_mode": args.operating_mode,
        },
        "measurement": {
            "mode": "CUDA graph replay",
            "warmups_per_arm": args.warmups,
            "samples_per_arm": args.samples,
            "order": "alternating serial-first and paired-first",
            "ratio": "serial median milliseconds / paired median milliseconds",
        },
        "cases": [
            _measure_shape(
                tokens,
                input_width=args.input_width,
                qkvz_width=args.qkvz_width,
                ba_width=args.ba_width,
                warmups=args.warmups,
                samples=args.samples,
            )
            for tokens in args.tokens
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
