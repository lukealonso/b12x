#!/usr/bin/env python3
"""Capture an auditable receipt for one isolated GLM serving arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def run(*args: str, check: bool = True) -> dict[str, Any]:
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {args!r}\n{completed.stderr}"
        )
    return {
        "argv": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30.0) as response:
        payload = response.read()
    return json.loads(payload) if payload else None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inspect_command = run("docker", "inspect", args.container)
    container_inspect = json.loads(inspect_command["stdout"])[0]
    image_id = container_inspect["Image"]
    image_command = run("docker", "image", "inspect", image_id)
    image_inspect = json.loads(image_command["stdout"])[0]
    scale_command = run(
        "docker",
        "exec",
        args.container,
        "sha256sum",
        "/opt/kv-scale/nvfp4_mla_outer_scales.json",
    )
    model_scale_command = run(
        "docker",
        "exec",
        args.container,
        "sha256sum",
        "/models/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78/nvfp4_mla_outer_scales.json",
    )
    logs_command = run("docker", "logs", args.container)
    logs_path = args.output_dir / "container.log"
    logs_payload = (logs_command["stdout"] + logs_command["stderr"]).encode("utf-8")
    logs_path.write_bytes(logs_payload)

    nvidia_query = run(
        "nvidia-smi",
        "--query-gpu=index,uuid,name,pstate,power.draw,power.limit,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.active,compute_mode,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    )
    topology = run("nvidia-smi", "topo", "-m")
    processes = run("nvidia-smi", "pmon", "-c", "1")
    docker_ps = run("docker", "ps", "--no-trunc")
    docker_top = run("docker", "top", args.container, "-eo", "pid,ppid,user,etime,args")

    receipt = {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "client_host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "base_url": args.base_url,
        "health": get_json(f"{args.base_url}/health"),
        "models": get_json(f"{args.base_url}/v1/models"),
        "container": container_inspect,
        "image": image_inspect,
        "outer_scales": {
            "kv_mount_sha256sum": scale_command["stdout"].strip(),
            "model_mount_sha256sum": model_scale_command["stdout"].strip(),
        },
        "runtime": {
            "docker_top": docker_top,
            "docker_ps": docker_ps,
            "nvidia_smi_query": nvidia_query,
            "nvidia_smi_topology": topology,
            "nvidia_smi_processes": processes,
        },
        "logs": {
            "path": str(logs_path.resolve()),
            "bytes": len(logs_payload),
            "sha256": sha256_bytes(logs_payload),
        },
    }
    output = args.output_dir / "service_receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(output.resolve())
    print(f"logs_sha256={receipt['logs']['sha256']}")
    print(scale_command["stdout"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
