"""Repeat complete production races without changing a fitted policy or its holdouts."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True)
    parser.add_argument("--task", default="production")
    parser.add_argument("--queries", type=Path, required=True, help="JSON array of complete tuning queries")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=52)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("rechecks require at least two independent rounds")

    import torch
    from b12x.policy import detect_device
    from b12x.policy.catalog import list_profiled_components
    from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
    from b12x.policy.generation.program import _complete_inputs, _scope_task
    from b12x.policy.generation.provenance import capture_measurement_provenance
    from b12x.policy.generation.store import CheckpointStore
    from b12x.tools.generate_gpu_profile import _source_revision
    from benchmarks.benchmark_profile_timing import snapshot

    torch.cuda.set_device(args.device)
    registration = next(item for item in list_profiled_components() if item.component_id == args.component)
    generator = registration.create_generator()
    task = next(item for item in generator.measurement_program if item.name == args.task)
    queries = tuple(_complete_inputs(generator.problem, query) for query in json.loads(args.queries.read_text()))
    if not queries or len(queries) != len(set(queries)):
        parser.error("queries must be nonempty and unique")
    task = _scope_task(task, queries)
    context = GenerationContext(device=detect_device(args.device).identity, device_ordinal=args.device,
        work_dir=args.work_dir, source_revision=_source_revision(),
        settings=GenerationSettings(repetitions=args.repetitions, groups=args.groups),
        provenance=capture_measurement_provenance(args.device))
    report = dict(status="research-only", purpose="independent repeats; does not replace failed qualification",
                  command=sys.argv, worktree=str(Path.cwd()), generation=context.checkpoint_metadata(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  query_sha256=hashlib.sha256(args.queries.read_bytes()).hexdigest(), rounds=[])
    args.work_dir.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.rounds):
        cohort = f"diagnostic-repeat-{index}"
        checkpoints = CheckpointStore(args.work_dir / "checkpoints" / cohort,
                                      observations_path=args.work_dir / "observations.sqlite3")
        row = dict(cohort=cohort, before=snapshot(args.device), measurements=[])
        started = time.monotonic()
        with task.open_search(replace(context, measurement_cohort=cohort), checkpoints) as search:
            points = search.points if index % 2 == 0 else search.points[::-1]
            for point in points:
                item = search.measure(point)
                row["measurements"].append(dict(query=point.query.to_dict(), winner=item.winner,
                    candidate_ids=list(item.candidate_ids), latencies_us=item.latencies_us.to_dict(),
                    eligible_candidates=item.eligible_candidates,
                    all_candidates_correct=set(item.latencies_us) == set(item.candidate_ids),
                    costs_seconds=item.costs_seconds.to_dict()))
            row["decisions"] = {key: value.to_dict() for key, value in search.configs.items()}
        row.update(seconds=time.monotonic() - started, after=snapshot(args.device))
        report["rounds"].append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
