"""Compare complete GQA policy resolution for an identical covered query corpus."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def export_queries(source, output):
    sys.path.insert(0, str(source))
    from b12x.policy import EMBEDDED_REGISTRY
    from b12x.policy.components import GQA_ATTENTION
    from b12x.policy.types import ExactDecisionNode, ProfileLeaf

    def walk(node, query):
        if isinstance(node, ProfileLeaf):
            yield query
            return
        workspace_limits = {"requested_max_work_items", "requested_max_partial_rows"}
        if node.default is not None and node.field not in workspace_limits:
            raise ValueError("query export requires explicitly bounded GQA coverage")
        for value, child in node.branches:
            values = (value,) if isinstance(node, ExactDecisionNode) else range(value.minimum, value.maximum + 1)
            for scalar in values:
                yield from walk(child, {**query, node.field: scalar})

    profiles = []
    for profile in EMBEDDED_REGISTRY.list_profiles():
        component = profile.component(GQA_ATTENTION)
        if component is None or component.planner is None:
            continue
        queries = []
        for query in walk(component.planner, {}):
            if len(queries) >= 100_000:
                raise ValueError("GQA query export exceeded its 100,000-query budget")
            queries.append(query)
        profiles.append({"profile_id": profile.profile_id, "device": asdict(profile.targets[0]), "queries": queries})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 2, "profiles": profiles}, separators=(",", ":")) + "\n")


def probe(source, manifests):
    sys.path.insert(0, str(source))
    from b12x.policy import DeviceIdentity, PolicyContext, PolicyMode
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
    from b12x.policy.generation.provenance import measurement_source_sha256

    batches = []
    corpus = json.loads(manifests.read_text())
    if corpus.get("schema_version") != 2:
        raise ValueError("GQA resolution benchmark requires query corpus schema 2")
    for manifest in corpus["profiles"]:
        device = DeviceIdentity(**manifest["device"])
        context = PolicyContext.for_identity(device, mode=PolicyMode.PREPLANNED_ONLY)
        queries = tuple(GqaQuery(**{"device": None, "requested_max_work_items": None,
                                   "requested_max_partial_rows": None, **query}) for query in manifest["queries"])
        batches.append((device, context, queries))
    if not batches:
        raise ValueError("GQA comparison requires covered-query manifests")
    identity = hashlib.sha256()
    for device, context, queries in batches:
        for query in queries:
            config = context.resolve(GQA_POLICY, query).config
            identity.update(json.dumps((asdict(device), asdict(query), asdict(config)), sort_keys=True).encode())
    uncached, cached = [], []
    for _ in range(5):
        elapsed, count = 0, 0
        for _device, context, queries in batches:
            context._resolution_cache.clear()
            started = time.perf_counter_ns()
            for query in queries:
                context.resolve(GQA_POLICY, query)
            elapsed += time.perf_counter_ns() - started
            count += len(queries)
        uncached.append(elapsed / (count * 1000.))
        elapsed, count = 0, 0
        for _device, context, queries in batches:
            selected = queries[::max(1, len(queries) // 64)]
            context._resolution_cache.clear()
            for query in selected:
                context.resolve(GQA_POLICY, query)
            started = time.perf_counter_ns()
            for _ in range(20):
                for query in selected:
                    context.resolve(GQA_POLICY, query)
            elapsed += time.perf_counter_ns() - started
            count += 20 * len(selected)
        cached.append(elapsed / (count * 1000.))
    print(json.dumps({"source": str(source), "config_identity": identity.hexdigest(),
                      "source_sha256": measurement_source_sha256(),
                      "query_corpus_sha256": hashlib.sha256(manifests.read_bytes()).hexdigest(),
                      "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "queries": sum(len(queries) for _, _, queries in batches),
                      "uncached_us": uncached, "cached_us": cached}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--export-queries", type=Path, metavar="SOURCE")
    args = parser.parse_args()
    if args.export_queries:
        export_queries(args.export_queries.resolve(), args.manifests.resolve())
        return
    if args.probe:
        probe(args.probe.resolve(), args.manifests.resolve())
        return
    if args.baseline is None or args.output is None or args.repetitions < 1:
        parser.error("baseline, output, and positive repetitions are required")
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    samples = {name: [] for name in roots}
    for repetition in range(args.repetitions):
        for name in (("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")):
            raw = subprocess.check_output([sys.executable, "-P", str(Path(__file__).resolve()),
                "--probe", str(roots[name]), "--manifests", str(args.manifests.resolve())], text=True)
            samples[name].append(json.loads(raw))
    if len({item["config_identity"] for values in samples.values() for item in values}) != 1:
        raise ValueError("resolved GQA configurations differ between source checkouts")
    medians = {name: {mode: statistics.median(value for item in items for value in item[mode])
                     for mode in ("uncached_us", "cached_us")} for name, items in samples.items()}
    result = {"command": sys.argv, "samples": samples, "medians": medians,
              "candidate_over_baseline": {mode: medians["candidate"][mode] / medians["baseline"][mode]
                                          for mode in ("uncached_us", "cached_us")},
              "ratio_direction": "lower_is_faster"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))


if __name__ == "__main__":
    main()
