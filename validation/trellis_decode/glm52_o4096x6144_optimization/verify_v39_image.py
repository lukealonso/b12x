#!/usr/bin/env python3
"""Verify and record the optimized v39 image identity and internal patch receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any


_INNER_SCRIPT = r"""
import hashlib
import json
from pathlib import Path

source = Path('/opt/venv/lib/python3.12/site-packages/sparkinfer/gemm/trellis_linear/_k6_mcg_cute.py')
receipt = Path('/opt/qualification/glm52_o4096x6144_v39_policy_patch.json')
print(json.dumps({
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
    'receipt': json.loads(receipt.read_text(encoding='utf-8')),
    'policy_lines': [
        line.strip()
        for line in source.read_text(encoding='utf-8').splitlines()
        if '(4096, 6144)' in line or '(5120, 6144)' in line
    ],
}, sort_keys=True))
"""


def capture(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--image", required=True)
    result.add_argument("--expected-image-id", required=True)
    result.add_argument("--expected-grid", type=int, default=144)
    result.add_argument("--expected-b12x-source-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    inspect = capture(["docker", "image", "inspect", args.image])
    if inspect["returncode"]:
        raise RuntimeError(inspect["stderr"].strip())
    inspected = json.loads(inspect["stdout"])[0]
    if inspected["Id"] != args.expected_image_id:
        raise RuntimeError(
            f"image ID mismatch: {inspected['Id']} != {args.expected_image_id}"
        )
    internal = capture(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--entrypoint",
            "/opt/venv/bin/python",
            args.image,
            "-c",
            _INNER_SCRIPT,
        ]
    )
    if internal["returncode"]:
        raise RuntimeError(internal["stderr"].strip())
    internal_payload = json.loads(internal["stdout"])
    patch_receipt = internal_payload["receipt"]
    expected_key = "4096x6144"
    if patch_receipt["after_table"].get(expected_key) != args.expected_grid:
        raise RuntimeError(
            f"optimized policy missing {expected_key}={args.expected_grid}"
        )
    if patch_receipt["b12x_source_sha256"] != args.expected_b12x_source_sha256:
        raise RuntimeError("embedded b12x source identity mismatch")
    if patch_receipt["after_sha256"] != internal_payload["source_sha256"]:
        raise RuntimeError("embedded source hash does not match patch receipt")
    labels = inspected["Config"].get("Labels", {})
    if labels.get("local-inference-lab.b12x.glm-o-proj-grid") != str(
        args.expected_grid
    ):
        raise RuntimeError("image grid label mismatch")
    output = {
        "schema": "b12x.glm52_o4096x6144_v39_image_verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "image": args.image,
        "image_id": inspected["Id"],
        "repo_digests": inspected.get("RepoDigests", []),
        "created": inspected.get("Created"),
        "size": inspected.get("Size"),
        "architecture": inspected.get("Architecture"),
        "labels": labels,
        "internal": internal_payload,
        "verification_pass": True,
    }
    write_json(args.output.resolve(), output)
    print(json.dumps({"output": str(args.output.resolve()), "verification_pass": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
