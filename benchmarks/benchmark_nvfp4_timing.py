"""Compare host and child-graph timing on the complete NVFP4 provider corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("at least two rounds are required to balance method ordering")

    import torch
    from b12x.policy import detect_device
    from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
    from b12x.policy.generation.providers import tunable
    from b12x.policy.generation.provenance import capture_measurement_provenance
    from b12x.policy.generation.timing import CapturedGraphTimer
    from b12x.tools.generate_gpu_profile import _source_revision
    from benchmarks.benchmark_profile_timing import snapshot

    torch.cuda.set_device(args.device)
    context = GenerationContext(device=detect_device(args.device).identity, device_ordinal=args.device,
                                work_dir=args.output.parent, source_revision=_source_revision(),
                                provenance=capture_measurement_provenance(args.device), settings=GenerationSettings())
    report = {"status": "diagnostic", "command": sys.argv, "worktree": str(Path.cwd()),
              "generation": context.checkpoint_metadata(), "cases": []}
    original = tunable.cuda_event_samples_us
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        for case in tunable._nvfp4_cases():
            row = {"query": case.query.to_dict(), "case_id": case.case_id,
                   "before": snapshot(args.device), "timing": []}
            report["cases"].append(row)

            def compare(run, *, count, device, flush=None):
                baseline = original(run, count=count, device=device, flush=flush)
                allocated = torch.cuda.memory_allocated(device)
                setup = time.monotonic()
                with CapturedGraphTimer(run.__self__, count=count, device=device, flush=flush) as timer:
                    creation_seconds = time.monotonic() - setup
                    timer.replay()
                    timer.samples_us()
                    result = {"creation_seconds": creation_seconds, "rounds": []}
                    row["timing"].append(result)
                    for repetition in range(args.rounds):
                        methods = ("host_events", "child_graph") if repetition % 2 == 0 else ("child_graph", "host_events")
                        for method in methods:
                            started = time.monotonic()
                            if method == "host_events":
                                samples = original(run, count=count, device=device, flush=flush)
                            else:
                                timer.replay()
                                samples = timer.samples_us()
                            result["rounds"].append({"round": repetition, "method": method,
                                                     "seconds": time.monotonic() - started,
                                                     "samples_us": list(samples), "median_us": statistics.median(samples)})
                result["allocation_delta_bytes"] = torch.cuda.memory_allocated(device) - allocated
                assert result["allocation_delta_bytes"] == 0
                return baseline

            tunable.cuda_event_samples_us = compare
            with tunable._Nvfp4Session(context) as session:
                measurements = session.measure(case, session.candidates(case))
            row["qualification"] = [measurement.to_dict() for measurement in measurements]
            row["after"] = snapshot(args.device)
            save()
            if not all(measurement.passes() for measurement in measurements):
                raise RuntimeError(f"NVFP4 production qualification failed: {row['qualification']}")
            print(json.dumps({"case_id": case.case_id, "candidates": len(measurements)}), flush=True)
    finally:
        tunable.cuda_event_samples_us = original
        save()


if __name__ == "__main__":
    main()
