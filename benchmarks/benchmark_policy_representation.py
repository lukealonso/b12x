"""Compare lossless policy encodings and fresh-process runtime imports.

Run with the repository's Python environment and --baseline-repo pointing to
the comparison checkout. Every alternative uses the complete embedded corpus;
canonical path constraints, configs, names, and evidence must remain identical.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from b12x.policy.decision_dag import encode_planner_dag
from b12x.policy.generation.reducer import decision_node_to_dict
from b12x.policy.serialization import profile_from_dict
from b12x.policy.types import ExactDecisionNode, ProfileLeaf, _decision_nodes


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _regions(node, predicates=()):
    if isinstance(node, ProfileLeaf):
        yield _canonical((sorted(predicates), node.name, node.config.to_dict(), node.evidence))
        return
    if isinstance(node, ExactDecisionNode):
        grouped = {}
        for value, child in node.branches:
            grouped.setdefault(id(child), (child, []))[1].append(value)
        branches = [(values, child) for child, values in grouped.values()]
        labels = [_canonical(("exact", sorted(values, key=_canonical))) for values, _ in branches]
    else:
        branches = [(bounds, child) for bounds, child in node.branches]
        labels = [_canonical(("range", bounds.minimum, bounds.maximum)) for bounds, _ in branches]
    for label, (_, child) in zip(labels, branches, strict=True):
        yield from _regions(child, (*predicates, (node.field, label)))
    if node.default is not None:
        yield from _regions(node.default, (*predicates, (node.field, _canonical(("default", sorted(labels))))))


def _signature(profile):
    return {
        component.component_id: Counter(_regions(component.planner))
        for component in profile.components if component.planner is not None
    }


def _witnesses(node, query=None):
    query = {} if query is None else query
    if isinstance(node, ProfileLeaf):
        yield query
        return
    for value, child in node.branches:
        choices = (value,) if isinstance(node, ExactDecisionNode) else (
            value.minimum, value.maximum,
        )
        for scalar in choices:
            yield from _witnesses(child, {**query, node.field: scalar})


def _lookups(originals, replacements, repeats):
    arms = [[], []]
    for original, replacement in zip(originals, replacements, strict=True):
        for left, right in zip(original.components, replacement.components, strict=True):
            if left.planner is None:
                continue
            queries = list(_witnesses(left.planner))
            # Spread probes across the entire component; bound benchmark work.
            queries = queries[::max(1, len(queries) // 1024)]
            for query in queries:
                a, b = left.lookup(query), right.lookup(query)
                assert a is not None and b is not None
                assert (a.config, a.name, a.evidence) == (b.config, b.name, b.evidence)
                arms[0].append((left, query))
                arms[1].append((right, query))
    samples = [[], []]
    for repeat in range(repeats):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            started = time.perf_counter()
            for component, query in arms[index]:
                component.lookup(query)
            samples[index].append((time.perf_counter() - started) * 1e6 / len(arms[index]))
    return {"queries": len(arms[0]), "tree_us_per_query": samples[0],
            "dag_us_per_query": samples[1]}


def _decode(blobs):
    return [profile_from_dict(json.loads(gzip.decompress(blob))) for blob in blobs]


def _measure(blobs, repeats):
    timings = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        profiles = _decode(blobs)
        timings.append(time.perf_counter() - start)
        del profiles
    gc.collect()
    tracemalloc.start()
    profiles = _decode(blobs)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    nodes = sum(
        sum(1 for _ in _decision_nodes(component.planner))
        for profile in profiles for component in profile.components
        if component.planner is not None
    )
    return {
        "compressed_bytes": sum(map(len, blobs)),
        "json_bytes": sum(len(gzip.decompress(blob)) for blob in blobs),
        "decode_seconds": timings, "median_decode_seconds": statistics.median(timings),
        "retained_bytes": retained, "peak_bytes": peak, "decision_nodes": nodes,
    }


def _imports(repo, repeats):
    code = """import json,time
start=time.perf_counter()
import b12x.policy
seconds=time.perf_counter()-start
rss=next(line.strip() for line in open('/proc/self/status') if line.startswith('VmRSS:'))
print(json.dumps(dict(seconds=seconds,rss=rss,source=b12x.policy.__file__)))
"""
    return [json.loads(subprocess.check_output(
        [sys.executable, "-B", "-c", code], cwd=repo, text=True,
    )) for _ in range(repeats)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline_repo.resolve()
    paths = sorted((baseline / "b12x/policy/_profiles/data").glob("*.json.gz"))
    sources = [json.loads(gzip.decompress(path.read_bytes())) for path in paths]
    originals = [profile_from_dict(source) for source in sources]
    signatures = [_signature(profile) for profile in originals]
    results = {"baseline_repo": str(baseline), "candidate_repo": str(_ROOT),
               "baseline_revision": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=baseline, text=True,
               ).strip(), "profiles": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }, "encodings": {}}
    for name in ("tree", "dag", "guarded_dag"):
        blobs = []
        for source, original, signature in zip(sources, originals, signatures, strict=True):
            payload = {**source, "components": []}
            for raw, component in zip(source["components"], original.components, strict=True):
                planner = component.planner
                encoded = (decision_node_to_dict(planner) if name == "tree" else
                           encode_planner_dag(planner, hoist_guards=name == "guarded_dag"))
                payload["components"].append({**raw, "planner": encoded})
            assert _signature(profile_from_dict(payload)) == signature, source["profile_id"]
            blobs.append(gzip.compress(_canonical(payload).encode(), mtime=0))
        results["encodings"][name] = {**_measure(blobs, args.repetitions), "equivalent": True}
        print(name, _canonical(results["encodings"][name]), flush=True)
    results["lookups"] = _lookups(originals, _decode(blobs), args.repetitions)
    results["imports"] = {
        "baseline": _imports(baseline, args.repetitions),
        "candidate": _imports(_ROOT, args.repetitions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(_canonical(results["imports"]))


if __name__ == "__main__":
    main()
