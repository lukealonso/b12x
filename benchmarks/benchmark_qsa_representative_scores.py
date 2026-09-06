#!/usr/bin/env python3
"""Compare QSA score-stage kernels under balanced CUDA graph replay.

This measures the paged representative scoring stage, not the full selector or
model throughput. Both arms use BF16 operands, FP32 output, identical page
tables and fixed 262144-token capacity. Live context remains device metadata.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b12x.attention import qsa
from b12x.attention.qsa._kernels import launch_score_representatives as triton_score
from b12x.attention.qsa._score_cute import launch_score_representatives as cute_score
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
    torch.manual_seed(98799)
    page, groups, dim = 752, 65536, 128
    pages = (groups + page - 1) // page
    caps = qsa.Caps(
        device="cuda",
        max_batch=128,
        max_raw_state_slots=128,
        max_q_rows=128,
        max_seq_len=groups * 4,
        num_main_cache_pages=pages,
        num_compressed_cache_pages=pages,
        main_page_size=page * 4,
        compressed_page_size=page,
    )
    root = Path(__file__).resolve().parents[1]
    sources = (
        Path(__file__),
        root / "b12x/attention/qsa/_kernels.py",
        root / "b12x/attention/qsa/_score_cute.py",
    )
    result = {
        "status": "research-only",
        "command": sys.argv,
        "worktree": str(Path(__file__).resolve().parents[1]),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True,
        ).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "cases": [],
        "timing": "warm-L2 score stage only; ABBA CUDA graphs, 32 launches per replay",
    }
    cache = torch.randn((pages, page, dim), device="cuda", dtype=torch.bfloat16)
    for rows in (1, 4, 16, 128):
        query = torch.randn((rows, 4, dim), device="cuda", dtype=torch.bfloat16)
        positions = torch.empty(rows, device="cuda", dtype=torch.int64)
        lengths = torch.full((rows,), groups * 4, device="cuda", dtype=torch.int32)
        requests = torch.arange(rows, device="cuda", dtype=torch.int32)
        table = torch.arange(pages, device="cuda", dtype=torch.int32).flip(0)
        table = table.expand(rows, -1).contiguous()
        output = torch.empty((rows, groups), device="cuda")
        metadata = [
            torch.zeros(rows, device="cuda", dtype=torch.int32) for _ in range(3)
        ]
        kwargs = dict(
            prepared_query=query,
            query_positions=positions,
            request_ids=requests,
            sequence_lengths=lengths,
            compressed_cache=cache,
            compressed_block_table=table,
            state_errors=metadata[0],
            scores=output,
            eligible_counts=metadata[1],
            merge_lengths=metadata[2],
            group_offset=0,
            group_count=groups,
            caps=caps,
        )
        for context in (128, 1024, 8192, 32768, 262144):
            positions.fill_(context - 1)
            triton_score(**kwargs)
            expected = output.clone()
            cute_score(**kwargs)
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
            assert not metadata[0].any()
            graphs = {}
            for name, launch in (("triton", triton_score), ("cute", cute_score)):

                def repeated(launch=launch):
                    for _ in range(32):
                        launch(**kwargs)

                graphs[name] = capture_cuda_graph(repeated, warmup=2)
            gpu_before_timing = nvidia_smi_gpu_mode_snapshot()
            samples = {name: [] for name in graphs}
            for name in ("triton", "cute", "cute", "triton"):
                samples[name].extend(
                    v / 32
                    for v in bench_cuda_graph(graphs[name], replays=30)["replay_us"]
                )
            gpu_after_timing = nvidia_smi_gpu_mode_snapshot()
            record = {
                "gpu_before_timing": gpu_before_timing,
                "gpu_after_timing": gpu_after_timing,
                "correctness": "passed",
                "rtol": 1e-5,
                "atol": 1e-5,
                "rows": rows,
                "context": context,
                "samples_us": samples,
                "median_us": {k: statistics.median(v) for k, v in samples.items()},
            }
            record["triton_over_cute_ratio"] = (
                record["median_us"]["triton"] / record["median_us"]["cute"]
            )
            result["cases"].append(record)
            print(
                json.dumps({k: v for k, v in record.items() if k not in ("samples_us", "gpu_before_timing", "gpu_after_timing")}),
                flush=True,
            )
    result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
