#!/usr/bin/env python3
"""Inventory GPU specialization sources and verify explicitly supplied objects."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b12x.policy.generation.census import inventory_cuda_trace, inventory_observations, inventory_sources, verify_compile_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--observations", type=Path, action="append", default=[])
    parser.add_argument("--trace-sqlite", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = inventory_sources(Path(__file__).resolve().parents[1])
    report["artifacts"] = [verify_compile_manifest(path) for path in args.manifest]
    report["observations"] = inventory_observations(args.observations)
    report["traces"] = [inventory_cuda_trace(path) for path in args.trace_sqlite]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": report["counts"], "unowned_compile_sites": report["unowned_compile_sites"],
                      "verified_artifacts": len(report["artifacts"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
