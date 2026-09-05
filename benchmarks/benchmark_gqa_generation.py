"""Measure registered GQA generation with a fixed production corpus subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from b12x.policy.device import detect_device
from b12x.policy.generation.attention_corpus import gqa_cases
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers.attention import GqaAttentionGenerator
from b12x.policy.generation.runner import generate_profile_artifact


def snapshot(device):
    return subprocess.check_output([
        "nvidia-smi", "-i", str(device),
        "--query-gpu=uuid,pstate,clocks.sm,clocks.mem,clocks_throttle_reasons.active",
        "--format=csv,noheader",
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--qualify-layouts", action="store_true")
    args = parser.parse_args()
    detected = detect_device(f"cuda:{args.device}")
    assert detected.identity is not None
    cases = tuple(
        case for case in gqa_cases()
        if case.metadata["model_id"] == "qwen3.8-flash-next-180b"
        and case.query["batch_size"] in (1, 8)
        and case.query["cache_tokens"] in (128, 16384)
        and (args.qualify_layouts or (
            case.query["kv_dtype"] == "bfloat16"
            and case.query["page_size"] == 64
            and case.query["kv_cache_layout"] == "separate"
        ))
    )
    assert len(cases) == (32 if args.qualify_layouts else 4)
    paths = (Path("b12x/policy/generation/providers/gpu_workers.py"),
             Path("b12x/policy/generation/providers/attention.py"))
    report = {
        "command": sys.argv, "worktree": str(Path.cwd()),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths},
        "case_ids": [case.case_id for case in cases], "rounds": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(args.repetitions):
        work = args.output.parent / f"{args.output.stem}-round{repetition}"
        if work.exists():
            raise FileExistsError(f"fresh measurements require an unused work directory: {work}")
        context = GenerationContext(
            device=detected.identity, device_ordinal=args.device,
            work_dir=work, source_revision=report["revision"],
            settings=GenerationSettings(),
        )
        before = snapshot(args.device)
        started = time.perf_counter()
        artifact = generate_profile_artifact(
            profile_id="nvidia.gqa-generation-benchmark",
            generators=(GqaAttentionGenerator(cases=cases),), context=context,
            progress=NullProgressReporter(),
        )
        elapsed = time.perf_counter() - started
        checkpoints = [json.loads(path.read_text()) for path in sorted(
            (work / "checkpoints" / "attention.gqa").glob("*.json")
        )]
        row = {"seconds": elapsed, "before": before, "after": snapshot(args.device),
               "artifact": artifact, "checkpoints": checkpoints}
        report["rounds"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"repetition": repetition, "seconds": elapsed,
                          "measurements": sum(len(c['measurements']) for c in checkpoints)}), flush=True)


if __name__ == "__main__":
    main()
