"""Diagnose cold-replay estimator variance with sequential and interleaved races."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--counts", type=int, nargs="+", default=(25, 256))
    parser.add_argument("--corpus", choices=("regressions", "provider"), default="regressions")
    args = parser.parse_args()
    if args.rounds < 2 or any(count <= 0 for count in args.counts):
        parser.error("timing requires at least two rounds and positive sample counts")

    import torch
    import b12x
    from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
    from b12x.policy import PolicyContext, PolicyMode, detect_device
    from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
    from b12x.policy.generation.providers.gpu_workers import _l2_flush_fn
    from b12x.policy.generation.providers.tunable import Nvfp4QuantizationGenerator, _nvfp4_cases
    from b12x.policy.generation.provenance import capture_measurement_provenance
    from b12x.policy.generation.timing import balanced_race_samples_us, cuda_event_samples_us
    from b12x.quantization import nvfp4
    from b12x.quantization.nvfp4._policy import NVFP4_QUANTIZATION_POLICY, Nvfp4QuantizationConfig
    from b12x.tools.generate_gpu_profile import _source_revision
    from benchmarks.benchmark_profile_timing import snapshot

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    context = GenerationContext(device=detect_device(device).identity, device_ordinal=args.device,
        work_dir=args.output.parent, source_revision=_source_revision(), settings=GenerationSettings(),
        provenance=capture_measurement_provenance(args.device))
    report = {"status": "diagnostic", "command": sys.argv, "worktree": str(Path.cwd()),
              "generation": context.checkpoint_metadata(), "ratio_direction": "retain / packed; lower favors retain",
              "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    cases = tuple(Nvfp4QuantizationGenerator.cases_for_tuning_queries((
        {"dtype": "bfloat16", "rows": 128, "columns": 4096},
        {"dtype": "bfloat16", "rows": 256, "columns": 2048},
    )))
    if args.corpus == "provider":
        cases = tuple({case.case_id: case for case in (*_nvfp4_cases(), *cases)}.values())
    for case in cases:
        rows, columns = case.query["rows"], case.query["columns"]
        seed = context.settings.seed + int(case.case_id[-8:], 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        source = torch.randn((rows, columns), device=device, dtype=torch.bfloat16, generator=generator).mul_(.25)
        scale = torch.tensor([.5], device=device)
        counts = torch.tensor([rows], device=device, dtype=torch.int32)
        expected, sf = quantize_grouped_nvfp4_torch(source.unsqueeze(0), counts, scale)
        expected_sf = sf.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8).reshape(-1)
        row = {"case_id": case.case_id, "query": case.query.to_dict(), "input_seed": seed,
               "before": snapshot(args.device), "rounds": [], "qualification": {}}
        report["cases"].append(row)
        graphs, storage, addresses = {}, {}, {}
        for name in ("retain", "packed"):
            policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY).with_override(
                NVFP4_QUANTIZATION_POLICY.component_id, Nvfp4QuantizationConfig(backend="cutedsl", liveness_strategy=name))
            plan = nvfp4.plan(rows, columns, policy=policy)
            outputs = nvfp4.allocate_outputs(plan, device=device)
            def run():
                nvfp4.run(plan=plan, x=source, global_scale=scale, outputs=outputs)
            for _ in range(context.settings.warmup):
                run()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            outputs.packed_a_storage.zero_()
            outputs.scale_flat.zero_()
            graph.replay()
            torch.cuda.synchronize(device)
            packed_exact = bool(torch.equal(outputs.packed_a_storage.permute(1, 2, 0), expected))
            scales_exact = bool(torch.equal(outputs.scale_flat, expected_sf))
            nonzero = bool(torch.count_nonzero(outputs.packed_a_storage) and torch.count_nonzero(outputs.scale_flat))
            row["qualification"][name] = dict(packed_exact=packed_exact, scales_exact=scales_exact, nonzero=nonzero)
            save()
            if not packed_exact or not scales_exact or not nonzero:
                raise RuntimeError(f"NVFP4 production oracle failed for {case.case_id}: {name}")
            graphs[name], storage[name] = graph, outputs
            addresses[name] = (outputs.packed_a_storage.data_ptr(), outputs.scale_flat.data_ptr())
        flush = _l2_flush_fn(device, enabled=True)
        allocated = torch.cuda.memory_allocated(device)
        b12x.freeze_kernel_resolution("balanced NVFP4 timing diagnostic")
        try:
            for repetition in range(args.rounds):
                for count in (args.counts if repetition % 2 == 0 else args.counts[::-1]):
                    methods = ("sequential", "interleaved") if repetition % 2 == 0 else ("interleaved", "sequential")
                    names = ("retain", "packed") if repetition % 2 == 0 else ("packed", "retain")
                    for method in methods:
                        before = snapshot(args.device)
                        started = time.monotonic()
                        if method == "sequential":
                            samples = {name: cuda_event_samples_us(graphs[name].replay, count=count, device=device, flush=flush)
                                       for name in names}
                        else:
                            samples = balanced_race_samples_us({name: graphs[name].replay for name in names},
                                                              count=count, device=device, flush=flush)
                        seconds = time.monotonic() - started
                        statistics_by_candidate = {}
                        for name, values in samples.items():
                            trimmed = sorted(values)[count // 10:count - count // 10]
                            statistics_by_candidate[name] = {"mean": statistics.fmean(values),
                                "median": statistics.median(values), "trimmed_mean": statistics.fmean(trimmed)}
                        row["rounds"].append(dict(round=repetition, method=method, count=count, seconds=seconds,
                            samples_us=samples, statistics=statistics_by_candidate, before=before, after=snapshot(args.device)))
                        assert torch.cuda.memory_allocated(device) == allocated
                        for name, outputs in storage.items():
                            assert addresses[name] == (outputs.packed_a_storage.data_ptr(), outputs.scale_flat.data_ptr())
                            assert torch.equal(outputs.packed_a_storage.permute(1, 2, 0), expected)
                            assert torch.equal(outputs.scale_flat, expected_sf)
                        save()
        finally:
            b12x.unfreeze_kernel_resolution()
            for graph in graphs.values():
                graph.reset()
        row["after"] = snapshot(args.device)
        row["replay_allocation_bytes"] = 0
        save()
        print(json.dumps({"case_id": case.case_id, "rounds": len(row["rounds"])}), flush=True)


if __name__ == "__main__":
    main()
