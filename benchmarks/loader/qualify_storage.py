"""Compare actual b12x GEMMs reading shared weights with CUDA-owned controls.

Run each allocation in a separate process to isolate CUDA faults. This is an
allocation qualification, not a full checkpoint startup benchmark. Timing uses
FlashInfer CUPTI with graph replay and cold L2, and reports raw paired samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from b12x.gemm import blockscaled
from b12x.loader import capabilities, storage_stats
from benchmarks.loader._utils import WeightFiles, paired_times, source_identity
from tests.gemm.test_blockscaled_a16 import make_weight, assert_close


def snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation",
        required=True,
        choices=[
            "cuda",
            "system",
            "pinned",
            "pinned_wc",
            "registered",
            "managed",
            "file",
        ],
        help="Use cuda for a second CUDA allocation as the null comparison",
    )
    parser.add_argument("--recipe", default="nvfp4", choices=["nvfp4", "mxfp8"])
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2560)
    parser.add_argument(
        "--modes", nargs="+", choices=["a16", "quantized"], default=["a16", "quantized"]
    )
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Rotate through distinct weight copies within each graph replay",
    )
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.copies < 1:
        parser.error("copies must be positive")
    torch.cuda.init()
    torch.manual_seed(782)
    repo = Path(__file__).resolve().parents[2]
    report = {
        **source_identity(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=repo)
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("B12X_", "CUDA_"))
        },
        "cold_l2_cache": not args.warm_l2,
        "allocation": args.allocation,
        "recipe": args.recipe,
        "shape_m_n_k": [args.m, args.n, args.k],
        "copies": args.copies,
        "capabilities": capabilities(),
        "before": snapshot(),
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="b12x-loader-") as directory:
        files = WeightFiles(directory, args.allocation)

        original, decoded, _ = make_weight(args.recipe, args.n, args.k)
        originals = [original] + [
            WeightFiles(directory, "cuda").load(original)
            for _ in range(args.copies - 1)
        ]
        loaded = [files.load(original) for _ in range(args.copies)]
        source = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
        oracle = source.float() @ decoded.T
        report["persistent_shared_bytes"] = files.storage_bytes
        physical_weight = original if args.recipe == "nvfp4" else original.weight
        weight_read_bytes = (
            physical_weight.values.nbytes + physical_weight.scale_mma.nbytes
        )
        if args.recipe == "nvfp4":
            weight_read_bytes += original.global_scale.nbytes
        report["weight_read_bytes"] = weight_read_bytes
        report["weight_working_set_bytes_per_arm"] = weight_read_bytes * args.copies
        report["storage_stats"] = storage_stats()
        for mode in args.modes:
            workspace = torch.empty(
                blockscaled.workspace_size(original, args.m),
                dtype=torch.uint8,
                device="cuda",
            )
            outputs = [
                [
                    torch.empty(args.m, args.n, dtype=torch.bfloat16, device="cuda")
                    for _ in range(args.copies)
                ]
                for _ in range(2)
            ]
            options = (
                {"activation_global_scale": torch.tensor([128.0], device="cuda")}
                if args.recipe == "nvfp4"
                else {}
            )

            def run(index):
                for bank in range(args.copies):
                    blockscaled.mm(
                        source,
                        (originals, loaded)[index][bank],
                        mode=mode,
                        out=outputs[index][bank],
                        workspace=workspace,
                        **options,
                    )

            for index in range(2):
                run(index)
            torch.cuda.synchronize()
            assert (
                torch.isfinite(outputs[0][0]).all()
                and outputs[0][0].count_nonzero() > 0
            )
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
            if mode == "a16":
                for output in outputs[0]:
                    assert_close(output, oracle)
            graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
            for index in range(2):
                with torch.cuda.graph(graphs[index]):
                    run(index)
            for _ in range(3):
                for arm in outputs:
                    for output in arm:
                        output.fill_(float("nan"))
                for graph in graphs:
                    graph.replay()
            torch.cuda.synchronize()
            assert all(
                torch.isfinite(output).all() and output.count_nonzero() > 0
                for output in outputs[0]
            )
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
            case = {"mode": mode, "exact_match": True, "graph_replay": True}
            if args.timing:
                case.update(paired_times(run, cold_l2_cache=not args.warm_l2))
                case["raw_unit"] = "milliseconds per full rotation"
                case["cuda_ms"] /= args.copies
                case["shared_ms"] /= args.copies
                case["weight_read_GB_s"] = weight_read_bytes / (case["shared_ms"] * 1e6)
            report["cases"].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in case.items() if k != "raw_ms"}), flush=True
            )
        report["after"] = snapshot()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
