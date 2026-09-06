#!/usr/bin/env python3
"""Mirror b12x's qualified GLM 4096x6144 grid policy into v39 Sparkinfer."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


EXPECTED_SOURCE_SHA256 = (
    "70edc835c03a0054ef29d56f869996fc4cf22df5b1cc4636734b9ea812a77f94"
)
EXPECTED_BEFORE = {
    (2048, 4096): 64,
    (6144, 1024): 32,
    (512, 6144): 48,
}
EXPECTED_AFTER = {
    (2048, 4096): 64,
    (4096, 6144): 144,
    (6144, 1024): 32,
    (512, 6144): 48,
}
OLD_BLOCK = """# The three K-by-N projection shapes below dominate GLM-5.2 dense online-K6
# decode. The resident CTA counts minimize full-chain latency on a 188-SM
# SM120 device while retaining capacity for concurrent model streams.
_GLM_GRID_CTA = {
    (2048, 4096): 64,
    (6144, 1024): 32,
    (512, 6144): 48,
}
"""
NEW_BLOCK = """# The K-by-N projection shapes below dominate GLM-5.2 dense online-K6 decode.
# The resident CTA counts minimize full-chain latency on a 188-SM SM120 device
# while retaining capacity for concurrent model streams.  The 4096x6144 TP4
# o_proj uses 144 CTAs: exact-checkpoint graph replay is faster than the served
# ExLlamaV3 route at M=1/4 and reaches parity at M=8/12/16.
_GLM_GRID_CTA = {
    (2048, 4096): 64,
    (4096, 6144): 144,
    (6144, 1024): 32,
    (512, 6144): 48,
}
"""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def grid_table(source: str) -> dict[tuple[int, int], int]:
    tree = ast.parse(source)
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_GLM_GRID_CTA"
            for target in node.targets
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one _GLM_GRID_CTA assignment, found {len(matches)}"
        )
    value = ast.literal_eval(matches[0])
    if not isinstance(value, dict):
        raise RuntimeError("_GLM_GRID_CTA is not a literal dictionary")
    return value


def write_atomic(path: Path, value: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.chmod(path.stat().st_mode)
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--path", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    result.add_argument("--b12x-source-sha256", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    before = args.path.read_bytes()
    before_hash = sha256_bytes(before)
    if before_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "v39 source hash differs from the reviewed port: "
            f"expected={EXPECTED_SOURCE_SHA256}, actual={before_hash}"
        )
    source = before.decode("utf-8")
    if grid_table(source) != EXPECTED_BEFORE:
        raise RuntimeError("v39 GLM planner table differs from the reviewed baseline")
    if source.count(OLD_BLOCK) != 1:
        raise RuntimeError("reviewed v39 GLM planner block is not unique")
    updated = source.replace(OLD_BLOCK, NEW_BLOCK)
    if grid_table(updated) != EXPECTED_AFTER:
        raise RuntimeError("updated v39 GLM planner table is incorrect")
    after = updated.encode("utf-8")
    write_atomic(args.path, after)
    if args.path.read_bytes() != after:
        raise RuntimeError("v39 policy file did not round-trip after replacement")
    receipt = {
        "schema": "b12x.glm52_o4096x6144_v39_policy_patch.v1",
        "patched_at": datetime.now(timezone.utc).isoformat(),
        "path": str(args.path),
        "before_sha256": before_hash,
        "after_sha256": sha256_bytes(after),
        "b12x_source_sha256": args.b12x_source_sha256,
        "before_table": {f"{k}x{n}": grid for (k, n), grid in EXPECTED_BEFORE.items()},
        "after_table": {f"{k}x{n}": grid for (k, n), grid in EXPECTED_AFTER.items()},
        "policy_owner": "b12x planner; v39 Sparkinfer is the pinned package namespace",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
