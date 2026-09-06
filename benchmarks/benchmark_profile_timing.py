"""Compare timing submission methods on qualified production MoE graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from b12x.policy.generation.timing import cuda_event_samples_us

from b12x.policy.device import detect_device
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.provenance import capture_measurement_provenance
from b12x.tools.generate_gpu_profile import _source_revision
from b12x.policy.generation.moe_corpus import (
    expand_physical_geometries,
    expand_sweep_cases,
)
from b12x.policy.generation.providers.moe_gpu_worker import (
    _MoeGeometrySession,
)


def snapshot(device: int) -> str:
    return subprocess.check_output(
        [
            "nvidia-smi", "-i", str(device),
            "--query-gpu=uuid,name,pstate,clocks.sm,clocks.mem,"
            "clocks_throttle_reasons.active,power.draw,temperature.gpu",
            "--format=csv,noheader",
        ], text=True,
    ).strip()


def compare_timing(
    run, *, count, device, flush, rounds, sample=cuda_event_samples_us,
    include_outer=False,
):
    setup_started = time.perf_counter()
    starts = tuple(torch.cuda.Event(enable_timing=True, external=True)
                   for _ in range(count))
    ends = tuple(torch.cuda.Event(enable_timing=True, external=True)
                 for _ in range(count))

    def submit() -> None:
        for start, end in zip(starts, ends, strict=True):
            if flush is not None:
                flush()
            start.record()
            run()
            end.record()

    def read() -> tuple[float, ...]:
        torch.cuda.synchronize(device)
        return tuple(float(start.elapsed_time(end)) * 1_000.0
                     for start, end in zip(starts, ends, strict=True))

    submit()
    read()
    event_setup_seconds = time.perf_counter() - setup_started
    outer = None
    capture_seconds = None
    if include_outer:
        capture_started = time.perf_counter()
        outer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(outer):
            submit()
        outer.replay()
        read()
        capture_seconds = time.perf_counter() - capture_started
    timing = {
        "event_setup_seconds": event_setup_seconds,
        "capture_seconds": capture_seconds,
        "rounds": [],
    }
    allocated = torch.cuda.memory_allocated(device)
    for round_index in range(rounds):
        modes = ["events", "pooled_events"]
        if outer is not None:
            modes.append("outer_graph")
        shift = round_index % len(modes)
        modes = modes[shift:] + modes[:shift]
        if (round_index // len(modes)) % 2:
            modes.reverse()
        for mode in modes:
            started = time.perf_counter()
            if mode == "events":
                samples = sample(
                    run, count=count, device=device,
                    flush=flush,
                )
            else:
                if mode == "pooled_events":
                    submit()
                else:
                    outer.replay()
                samples = read()
            timing["rounds"].append({
                "round": round_index, "mode": mode,
                "seconds": time.perf_counter() - started,
                "samples_us": samples,
                "median_us": statistics.median(samples),
            })
    timing["allocation_delta_bytes"] = (
        torch.cuda.memory_allocated(device) - allocated
    )
    assert timing["allocation_delta_bytes"] == 0
    if outer is not None:
        outer.reset()
    return timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--outer-graph", action="store_true")
    args = parser.parse_args()
    detected = detect_device(f"cuda:{args.device}")
    assert detected.identity is not None
    torch.cuda.set_device(args.device)
    context = GenerationContext(
        device=detected.identity, device_ordinal=args.device,
        work_dir=args.output.parent,
        source_revision=_source_revision(),
        provenance=capture_measurement_provenance(args.device),
        settings=GenerationSettings(),
    )
    geometry = next(
        item for item in expand_physical_geometries()
        if item.key == ("modelopt-nvfp4", "silu", 256, 2048, 64)
    )
    cases = expand_sweep_cases(
        geometries=(geometry,), top_ks=(8,), token_counts=(1, 8, 32),
    )
    report = {
        "command": __import__("sys").argv,
        "worktree": str(Path.cwd()),
        "generation": context.checkpoint_metadata(),
        "geometry": asdict(geometry),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(
                "b12x/policy/generation/providers/moe_gpu_worker.py"
            ))
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = context.settings.groups * context.settings.repetitions
    print(json.dumps({"stage": "prepare_geometry", "geometry": geometry.key}), flush=True)
    with _MoeGeometrySession(geometry, context) as session:
        for case in cases:
            print(json.dumps({"stage": "qualify", "case_id": case.case_id}), flush=True)
            before = snapshot(args.device)
            started = time.perf_counter()
            candidates = session.eligible_candidates(case, session.candidates)
            qualified = session.measure(case, candidates, correctness=True)
            qualification_seconds = time.perf_counter() - started
            row = {
                "case_id": case.case_id, "query": case.query(),
                "route_pattern": case.route_pattern,
                "before": before,
                "qualification_seconds": qualification_seconds,
                "qualification": [item.to_dict() for item in qualified],
                "timing": [],
            }
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            failures = [item for item in qualified if not item.passes(
                context.settings.minimum_cosine
            )]
            if failures:
                raise RuntimeError(f"qualification failed: {failures}")
            print(json.dumps({"stage": "timing", "case_id": case.case_id}), flush=True)
            for candidate in candidates:
                prepared = session._prepared_candidates[candidate.candidate_id]
                run = prepared.graph.replay
                timing = compare_timing(
                    run, count=count, device=args.device, flush=session._flush,
                    rounds=args.rounds,
                    include_outer=args.outer_graph,
                )
                timing["candidate"] = {"candidate_id": candidate.candidate_id,
                                       "config": candidate.config.to_dict()}
                row["timing"].append(timing)
            row["after"] = snapshot(args.device)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"case_id": case.case_id,
                              "candidates": len(candidates),
                              "qualification_seconds": qualification_seconds}),
                  flush=True)


if __name__ == "__main__":
    main()
