"""Compare event placement around identical, correctness-checked GQA graphs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--groups", type=int, default=52)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--globaltimer", action="store_true", help="Include diagnostic GPU timestamp kernels; changes timing instrumentation.")
    parser.add_argument("--profile-cuda", action="store_true", help="Bracket the measurement loop for Nsight Systems capture.")
    args = parser.parse_args()

    import torch
    from b12x.policy import detect_device
    from b12x.policy.generation.providers import gpu_workers
    from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
    from b12x.policy.generation.providers.attention import GqaAttentionGenerator
    from b12x.policy.generation.provenance import capture_measurement_provenance
    from b12x.policy.generation.sweep import SweepMeasurement
    from b12x.policy.generation.timing import CapturedGraphRaceTimer, balanced_race_samples_us, grouped_timing_evidence, median_of_group_medians
    from b12x.tools.generate_gpu_profile import _source_revision
    from benchmarks.benchmark_profile_timing import snapshot
    from contextlib import ExitStack
    from b12x.policy.generation import timestamps
    from b12x.policy.generation.timestamps import GlobaltimerGraphRaceTimer

    torch.cuda.set_device(args.device)
    context = GenerationContext(device=detect_device(args.device).identity, device_ordinal=args.device,
        work_dir=args.output.parent, source_revision=_source_revision(),
        settings=GenerationSettings(groups=args.groups, repetitions=args.repetitions),
        provenance=capture_measurement_provenance(args.device))
    cases = tuple(GqaAttentionGenerator.cases_for_tuning_queries(json.loads(args.queries.read_text())))
    provider = GqaAttentionGenerator(cases=cases)
    report = {"status": "research-only", "command": sys.argv, "worktree": str(Path.cwd()),
              "generation": context.checkpoint_metadata(),
              "globaltimer_driver_sha256": hashlib.sha256(Path(timestamps.__file__).read_bytes()).hexdigest() if args.globaltimer else None,
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": []}
    for case in cases:
        row = {"query": case.query.to_dict(), "rounds": [], "before": snapshot(args.device)}
        def compare(prepared, *, settings, device, flush):
            prepared = tuple(prepared)
            graphs = {item.candidate.candidate_id: item.graph for item in prepared}
            def replay_candidate(name):
                with torch.cuda.nvtx.range(f"candidate:{name}"):
                    graphs[name].replay()
            runs = {name: lambda name=name: replay_candidate(name) for name in graphs}
            expected = {item.candidate.candidate_id: item.owners[2].clone() for item in prepared}
            count = settings.groups * settings.repetitions
            row["candidates"] = {item.candidate.candidate_id: dict(item.candidate.config) for item in prepared}
            if args.profile_cuda:
                torch.cuda.cudart().cudaProfilerStart()
            try:
                with ExitStack() as timers:
                    timer = timers.enter_context(CapturedGraphRaceTimer(graphs, count=count, device=device, flush=flush))
                    globaltimer = timers.enter_context(GlobaltimerGraphRaceTimer(graphs, count=count, device=device, flush=flush)) if args.globaltimer else None
                    timer.replay()
                    timer.samples_us()
                    if globaltimer is not None:
                        globaltimer.replay()
                        globaltimer.samples_us()
                    for index in range(args.rounds):
                        samples = {}
                        methods = ("host_events", "device_events", "device_globaltimer") if globaltimer is not None else ("host_events", "device_events")
                        offset = index % len(methods)
                        methods = methods[offset:] + methods[:offset]
                        for method in methods:
                            for item in prepared:
                                item.owners[2].fill_(float("nan"))
                            allocated = torch.cuda.memory_allocated(device)
                            with torch.cuda.nvtx.range(f"{case.case_id}:round{index}:{method}"):
                                if method == "host_events":
                                    samples[method] = balanced_race_samples_us(runs, count=count, device=device, flush=flush)
                                elif method == "device_globaltimer":
                                    globaltimer.replay()
                                    samples[method] = globaltimer.samples_us()
                                else:
                                    timer.replay()
                                    samples[method] = timer.samples_us()
                            assert torch.cuda.memory_allocated(device) == allocated
                            assert all(torch.equal(item.owners[2], expected[item.candidate.candidate_id]) for item in prepared)
                        row["rounds"].append({"order": methods, "samples": {
                            method: {name: grouped_timing_evidence(values, groups=settings.groups, repetitions=settings.repetitions)
                                     for name, values in arms.items()} for method, arms in samples.items()},
                            "medians_us": {method: {name: median_of_group_medians(values, groups=settings.groups, repetitions=settings.repetitions)
                                                   for name, values in arms.items()} for method, arms in samples.items()},
                            "bitwise_graph_outputs": True, "replay_allocation_bytes": 0})
                return tuple(SweepMeasurement(candidate=item.candidate, correct=item.correct,
                    latency_us=row["rounds"][-1]["medians_us"]["device_events"][item.candidate.candidate_id]) for item in prepared)
            finally:
                if args.profile_cuda:
                    torch.cuda.cudart().cudaProfilerStop()
                for graph in graphs.values():
                    graph.reset()

        with provider._benchmark_factory(case.group_id, (case,), context) as session, \
                patch.object(gpu_workers, "measure_prepared_candidates", compare):
            measurements = session.measure(case, session.candidates(case))
            assert all(item.correct for item in measurements)
        if len(row["rounds"]) != args.rounds:
            raise RuntimeError("GQA preparation did not execute the requested timing comparison")
        row["after"] = snapshot(args.device)
        report["cases"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"query": row["query"], "medians": [item["medians_us"] for item in row["rounds"]]}), flush=True)


if __name__ == "__main__":
    main()
