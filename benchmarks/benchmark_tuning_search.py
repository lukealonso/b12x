"""Compare offline samplers on production races with isolated caches and holdouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def run_worker(args):
    started = time.monotonic()
    import torch

    from b12x.policy import detect_device
    from b12x.policy.catalog import list_profiled_components
    from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
    from b12x.policy.generation.provenance import capture_measurement_provenance
    from b12x.policy.generation.qualification import QualificationCase, qualify_policy
    from b12x.policy.generation.regions import fit_regret_regions
    from b12x.policy.generation.search import SearchBudget, SearchStrategy
    from b12x.policy.generation.store import CheckpointStore
    from b12x.policy.problem import AxisInterval, SearchDomain
    from b12x.tools.generate_gpu_profile import _source_revision

    torch.cuda.set_device(args.device)
    registration = next(item for item in list_profiled_components() if item.component_id == args.component)
    generator = registration.create_generator()
    task = next(item for item in generator.measurement_program if item.name == args.task)
    context = GenerationContext(device=detect_device(args.device).identity, device_ordinal=args.device,
                                source_revision=_source_revision(), work_dir=args.work_dir,
                                settings=GenerationSettings(), provenance=capture_measurement_provenance(args.device))
    checkpoints = CheckpointStore(args.work_dir / "checkpoints")
    with task.open_search(context, checkpoints) as search:
        all_points = search.points
        by_family = defaultdict(list)
        for point in all_points:
            by_family[point.family].append(point)
        holdout_keys = {point.key for points in by_family.values() for index, point in enumerate(points)
                        if len(points) > 1 and index % 5 == 1}
        holdouts = tuple(point for point in all_points if point.key in holdout_keys)
        training = tuple(point for point in all_points if point.key not in holdout_keys)
        if not holdouts:
            raise ValueError("the declared corpus has no independent within-family holdouts")
        domains = tuple(SearchDomain(fixed=family, axes=tuple(
            AxisInterval(name=axis.name, minimum=min(point.coordinates[index] for point in points),
                         maximum=max(point.coordinates[index] for point in points), alignment=axis.alignment)
            for index, axis in enumerate(generator.problem.axes)
        )) for family, points in by_family.items())
        search.points = training
        strategy = SearchStrategy(args.strategy)
        budget = len(training) if strategy is SearchStrategy.EXHAUSTIVE else max(
            len(by_family), math.ceil(len(training) * args.fraction))
        outcome = search.search(strategy=strategy, budget=SearchBudget(queries=budget))
        decisions = dict(search.configs)
    fit_started = time.monotonic()
    fit = fit_regret_regions(outcome.measurements, decisions,
                             axes=tuple(field.name for field in generator.problem.axes), domains=domains)
    fit_seconds = time.monotonic() - fit_started
    independent = replace(context, measurement_cohort="independent-corpus-holdout")
    qualified_cases = []
    with task.open_search(independent, checkpoints) as search:
        for point in holdouts:
            measured = search.measure(point)
            qualified_cases.append(QualificationCase(measurement=measured, selected_candidate=fit.select(point.query),
                                   partition="corpus-heldout", cohort=independent.measurement_cohort))
    report = qualify_policy(qualified_cases, training_queries=fit.training_queries,
                            required_partitions=frozenset({"corpus-heldout"}))
    artifact = {
        "schema_version": 1, "status": "research-only", "component": args.component, "task": args.task,
        "command": sys.argv, "worktree": str(Path.cwd()), "generation": context.checkpoint_metadata(),
        "generation_seconds": time.monotonic() - started, "fit_seconds": fit_seconds,
        "sampling": outcome.accounting(), "fit": fit.describe(), "holdout": report.to_dict(),
        "scope": "deterministic within-family corpus holdouts; additional geometry and decision-boundary qualification required",
        "training": [{"query": item.point.query.to_dict(), "candidate_ids": list(item.candidate_ids),
                      "latencies_us": item.latencies_us.to_dict(), "selected": item.winner,
                      "costs_seconds": item.costs_seconds.to_dict()} for item in outcome.measurements],
        "holdout_cases": [{"query": case.measurement.point.query.to_dict(), "selected": case.selected_candidate,
                           "candidate_ids": list(case.measurement.candidate_ids),
                           "latencies_us": case.measurement.latencies_us.to_dict(),
                           "regret_ratio": case.ratio if math.isfinite(case.ratio) else None} for case in qualified_cases],
        "decisions": {key: value.to_dict() for key, value in decisions.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: artifact[key] for key in ("component", "generation_seconds", "sampling", "fit", "holdout")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True)
    parser.add_argument("--task", default="production")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=.5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--strategy", choices=("exhaustive", "space_filling", "adaptive", "bayesian"))
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or args.rounds < 1:
        parser.error("fraction must be in (0,1] and rounds must be positive")
    if args.strategy:
        run_worker(args)
        return
    results = []
    for repetition in range(args.rounds):
        strategies = ("exhaustive", "space_filling", "adaptive", "bayesian")
        if repetition % 2:
            strategies = strategies[::-1]
        for strategy in strategies:
            work_dir = args.work_dir / f"round-{repetition}-{strategy}"
            work_dir.mkdir(parents=True, exist_ok=False)
            output = work_dir / "result.json"
            environment = {**os.environ, "B12X_COMPILE_CACHE_DIR": str(work_dir / "compile-cache")}
            command = [sys.executable, str(Path(__file__).resolve()), "--component", args.component,
                       "--task", args.task, "--device", str(args.device), "--work-dir", str(work_dir),
                       "--output", str(output), "--fraction", str(args.fraction), "--strategy", strategy]
            with (work_dir / "worker.log").open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(output.read_text())
            results.append({"round": repetition, "strategy": strategy, "result": str(output),
                            "generation_seconds": result["generation_seconds"], "holdout": result["holdout"]})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"status": "research-only", "command": sys.argv, "results": results}, indent=2) + "\n")
            print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
