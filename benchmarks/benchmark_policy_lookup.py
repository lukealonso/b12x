"""Compare complete embedded-policy lookup paths between source checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def probe(root: Path):
    sys.path.insert(0, str(root))
    started = time.perf_counter()
    from b12x.policy import EMBEDDED_REGISTRY
    from b12x.policy.types import ExactDecisionNode, ProfileLeaf
    import_seconds = time.perf_counter() - started

    def witnesses(node, query):
        if isinstance(node, ProfileLeaf):
            yield query
        else:
            for value, child in node.branches:
                values = (value,) if isinstance(node, ExactDecisionNode) else (value.minimum, value.maximum)
                for scalar in values:
                    yield from witnesses(child, {**query, node.field: scalar})

    probes = []
    identity = hashlib.sha256()
    for profile in EMBEDDED_REGISTRY.list_profiles():
        for component in profile.components:
            if component.planner is None:
                continue
            queries = list(witnesses(component.planner, {}))
            queries = queries[::max(1, len(queries) // 1024)]
            for query in queries:
                hit = component.lookup(query)
                payload = (profile.profile_id, component.component_id, query,
                           None if hit is None else (hit.name, hit.config.to_dict(), hit.evidence))
                identity.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
                probes.append((component, query))
    for component, query in probes:
        component.lookup(query)
    samples = []
    for _ in range(5):
        started = time.perf_counter_ns()
        for component, query in probes:
            component.lookup(query)
        samples.append((time.perf_counter_ns() - started) / (1000. * len(probes)))
    rss = next(line.strip() for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmRSS:"))
    print(json.dumps({"source": str(root), "types_sha256": hashlib.sha256((root / "b12x/policy/types.py").read_bytes()).hexdigest(),
                      "probe_count": len(probes), "lookup_identity": identity.hexdigest(),
                      "microseconds_per_lookup": samples, "import_seconds": import_seconds, "rss": rss}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--probe", type=Path)
    args = parser.parse_args()
    if args.probe is not None:
        probe(args.probe.resolve())
        return
    if args.baseline is None or args.output is None:
        parser.error("--baseline and --output are required")
    samples = {"baseline": [], "candidate": []}
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    for repetition in range(args.repetitions):
        for arm in (("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")):
            raw = subprocess.check_output([sys.executable, "-B", str(Path(__file__).resolve()), "--probe", str(roots[arm])], text=True)
            samples[arm].append(json.loads(raw))
    identities = {item["lookup_identity"] for values in samples.values() for item in values}
    if len(identities) != 1:
        raise ValueError("lookup corpora or results differ between checkouts")
    median = {arm: statistics.median(value for item in items for value in item["microseconds_per_lookup"])
              for arm, items in samples.items()}
    result = {"command": sys.argv, "samples": samples, "median_microseconds": median,
              "candidate_over_baseline": median["candidate"] / median["baseline"],
              "ratio_direction": "lower_is_faster"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))


if __name__ == "__main__":
    main()
