"""Compare timing methods inside the registered production GQA provider."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from b12x.policy.device import detect_device
from b12x.policy.generation.attention_corpus import gqa_cases
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers import gpu_workers
from benchmarks.benchmark_profile_timing import compare_timing, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--outer-graph", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detected = detect_device(f"cuda:{args.device}")
    assert detected.identity is not None
    torch.cuda.set_device(args.device)
    context = GenerationContext(
        device=detected.identity, device_ordinal=args.device,
        work_dir=args.output.parent,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        settings=GenerationSettings(),
    )
    cases = tuple(
        case for case in gqa_cases()
        if case.metadata["model_id"] == "qwen3.8-flash-next-180b"
        and case.query["batch_size"] in (1, 8)
        and case.query["cache_tokens"] in (128, 16384)
        and case.query["kv_dtype"] == "bfloat16"
        and case.query["page_size"] == 64
        and case.query["kv_cache_layout"] == "separate"
    )
    assert len(cases) == 4
    paths = (Path(__file__), Path("benchmarks/benchmark_profile_timing.py"),
             Path(gpu_workers.__file__))
    report = {
        "command": sys.argv, "worktree": str(Path.cwd()),
        "generation": context.checkpoint_metadata(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original = gpu_workers._cuda_event_samples_us
    factory = gpu_workers.GqaBenchmarkFactory()
    for case in cases:
        row = {"case_id": case.case_id, "query": case.query.to_dict(),
               "before": snapshot(args.device), "timing": []}
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"stage": "qualify_and_compare", "case_id": case.case_id}), flush=True)

        def observe(run, *, count, device, flush=None):
            baseline = original(run, count=count, device=device, flush=flush)
            row["timing"].append(compare_timing(
                run, count=count, device=device, flush=flush,
                rounds=args.rounds, sample=original,
                include_outer=args.outer_graph,
            ))
            return baseline

        gpu_workers._cuda_event_samples_us = observe
        started = time.perf_counter()
        try:
            with factory(case.group_id, (case,), context) as session:
                candidates = session.candidates(case)
                measurements = session.measure(case, candidates)
        finally:
            gpu_workers._cuda_event_samples_us = original
        row["total_seconds"] = time.perf_counter() - started
        row["qualification"] = [item.to_dict() for item in measurements]
        row["after"] = snapshot(args.device)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not all(item.correct for item in measurements):
            raise RuntimeError(f"GQA qualification failed: {row['qualification']}")
        assert len(measurements) == len(row["timing"])
        print(json.dumps({"case_id": case.case_id, "candidates": len(measurements),
                          "total_seconds": row["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
