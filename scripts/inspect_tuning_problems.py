#!/usr/bin/env python3
"""Emit the catalog's complete input and kernel-decision inventory as JSON."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b12x.policy.catalog import API_ALIASES, list_generation_components


def api_coverage(registrations):
    """Report registration scope without equating it to kernel qualification."""
    import b12x

    ops = b12x.list_ops()
    owners = {item.op_qualname: item.component_id for item in registrations}
    aliases = {item.op_qualname: item for item in API_ALIASES}
    return {
        "scope": "public_api_registration; internal kernel specialization coverage is not established",
        "public_apis": len(ops),
        "api_styles": dict(Counter(op.api_style for op in ops)),
        "registered_components": len(registrations),
        "provider_roles": dict(Counter(item.mode.value for item in registrations)),
        "api_aliases": [{"op": item.op_qualname, "owner": item.owner_op,
                         "entry_points": item.entry_points, "recipes": item.recipes} for item in API_ALIASES],
        "unregistered_apis": [
            {"op": op.qualname, "api_style": op.api_style}
            for op in ops if op.qualname not in owners and op.qualname not in aliases
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", action="append")
    args = parser.parse_args()
    registrations = list_generation_components()
    selected = set(args.component or (item.component_id for item in registrations))
    unknown = selected - {item.component_id for item in registrations}
    if unknown:
        parser.error(f"unknown components: {sorted(unknown)}")
    problems = []
    for item in registrations:
        if item.component_id not in selected:
            continue
        generator = item.create_generator()
        problems.append({**generator.problem.describe(), "artifact_kind": generator.artifact_kind, "measurement_program": [
            task.describe() for task in generator.measurement_program
        ]})
    print(json.dumps({"schema_version": 1, "api_coverage": api_coverage(registrations),
                      "problems": problems}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
