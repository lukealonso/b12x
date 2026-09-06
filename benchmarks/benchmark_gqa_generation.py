"""Measure registered GQA generation with a fixed production corpus subset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from rich.console import Console

from b12x.policy.device import detect_device
from b12x.policy.generation.attention_corpus import gqa_cases
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers.attention import GqaAttentionGenerator
from b12x.policy.generation.parallel import run_parallel_measurements
from b12x.policy.generation.registry import ComponentGeneratorRegistry
from b12x.policy.generation.runner import generate_profile_artifact


def snapshot(device):
    return subprocess.check_output([
        "nvidia-smi", "-i", str(device),
        "--query-gpu=uuid,pstate,clocks.sm,clocks.mem,clocks_throttle_reasons.active",
        "--format=csv,noheader",
    ], text=True).strip()


def registry(*, qualify_layouts):
    cases = tuple(
        case for case in gqa_cases()
        if case.metadata["model_id"] == "qwen3.8-flash-next-180b"
        and case.query["batch_size"] in (1, 8)
        and case.query["cache_tokens"] in (128, 16384)
        and (qualify_layouts or (
            case.query["kv_dtype"] == "bfloat16"
            and case.query["page_size"] == 64
            and case.query["kv_cache_layout"] == "separate"
        ))
    )
    assert len(cases) == (32 if qualify_layouts else 4)
    result = ComponentGeneratorRegistry()
    result.register(GqaAttentionGenerator(cases=cases))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", type=int, default=0)
    devices.add_argument("--devices", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--qualify-layouts", action="store_true")
    args = parser.parse_args()
    ordinals = args.devices or (args.device,)
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("GPU ordinals must be distinct")
    detected = detect_device(f"cuda:{ordinals[0]}")
    assert detected.identity is not None
    for ordinal in ordinals[1:]:
        if detect_device(f"cuda:{ordinal}").identity != detected.identity:
            raise ValueError("parallel generation requires identical device identities")
    factory = partial(registry, qualify_layouts=args.qualify_layouts)
    generators = factory().select(None)
    cases = generators[0]._cases
    paths = (Path(__file__),
             Path("b12x/policy/generation/providers/gpu_workers.py"),
             Path("b12x/policy/generation/providers/attention.py"),
             Path("b12x/policy/generation/parallel.py"),
             Path("b12x/policy/generation/sweep.py"),
             Path("b12x/policy/generation/store.py"),
             Path("b12x/policy/decision_dag.py"))
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
            device=detected.identity, device_ordinal=ordinals[0],
            work_dir=work, source_revision=report["revision"],
            settings=GenerationSettings(),
        )
        before = {ordinal: snapshot(ordinal) for ordinal in ordinals}
        started = time.perf_counter()
        parallel = None
        if len(ordinals) > 1:
            parallel = asdict(run_parallel_measurements(
                console=Console(),
                device_specs=tuple(f"cuda:{ordinal}" for ordinal in ordinals),
                generators=generators, context=context, registry_factory=factory,
            ))
        artifact = generate_profile_artifact(
            profile_id="nvidia.gqa-generation-benchmark",
            generators=generators, context=context,
            progress=NullProgressReporter(),
        )
        elapsed = time.perf_counter() - started
        checkpoints = [json.loads(path.read_text()) for path in sorted(
            (work / "checkpoints" / "attention.gqa").glob("*.json")
        )]
        from b12x.policy.generation.providers import gpu_workers

        def forbid_session(*_args, **_kwargs):
            raise AssertionError("a complete resume must not enter a GPU session")

        original_enter = gpu_workers._GqaSession.__enter__
        gpu_workers._GqaSession.__enter__ = forbid_session
        resume_started = time.perf_counter()
        try:
            resumed = generate_profile_artifact(
                profile_id="nvidia.gqa-generation-benchmark",
                generators=generators, context=context,
                progress=NullProgressReporter(),
            )
        finally:
            gpu_workers._GqaSession.__enter__ = original_enter
        assert resumed == artifact
        resume_seconds = time.perf_counter() - resume_started
        row = {"seconds": elapsed, "before": before,
               "after": {ordinal: snapshot(ordinal) for ordinal in ordinals},
               "parallel": parallel,
               "resume_seconds": resume_seconds,
               "resume_without_gpu_session": True,
               "artifact": artifact, "checkpoints": checkpoints}
        report["rounds"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"repetition": repetition, "seconds": elapsed,
                          "measurements": sum(len(c['measurements']) for c in checkpoints)}), flush=True)


if __name__ == "__main__":
    main()
