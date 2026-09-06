#!/usr/bin/env python3
"""Run one isolated full-profile GLM service arm and its cache-proof workloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


MODEL_DIR = Path("/data/models/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78")
SERVED_MODEL = "GLM-5.2-EXL3-TR3v4-3.5bpw"
GPU_INDICES = "0,1,2,3"
DEFAULT_QUALIFICATION_TOOL_ROOT = Path(__file__).resolve().parent
WORKLOADS = (
    ("short_c1_p128_o256", 128, 256, 1, 1, 5),
    ("short_c4_p128_o256", 128, 256, 4, 1, 5),
    ("long_c1_p32768_o256", 32768, 256, 1, 1, 3),
    ("long_c4_p32768_o256", 32768, 256, 4, 1, 3),
    ("prefill_c1_p8039_o1", 8039, 1, 1, 1, 5),
)
GPU_QUERY = (
    "index,uuid,name,pstate,power.draw,power.limit,clocks.current.sm,"
    "clocks.current.memory,clocks_throttle_reasons.active,memory.used,"
    "pcie.link.gen.current,compute_mode,driver_version"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(command: list[str], *, check: bool = False) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result = {
        "command": command,
        "command_shell": shlex.join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {result['command_shell']}\n"
            f"{completed.stderr}"
        )
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gpu_snapshot() -> dict[str, Any]:
    return capture(
        [
            "nvidia-smi",
            f"--id={GPU_INDICES}",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
    )


def wait_for_idle(timeout_seconds: int) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = gpu_snapshot()
        snapshots.append(snapshot)
        rows = [line.split(",") for line in snapshot["stdout"].splitlines() if line]
        idle = len(rows) == 4
        for row in rows:
            values = [value.strip() for value in row]
            idle &= (
                values[3] == "P8"
                and int(values[9]) <= 16
                and values[10] == "1"
                and values[8] == "0x0000000000000000"
            )
        if idle:
            return snapshots
        time.sleep(2.0)
    raise RuntimeError(
        "target GPUs did not settle to P8/Gen1 with clear throttle masks"
    )


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        return client.connect_ex(("127.0.0.1", port)) != 0


def service_command(args: argparse.Namespace, cache_dir: Path) -> list[str]:
    model_mount = "/models/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
    environment = {
        "MODEL_FAMILY": "glm52-exl3",
        "GPUS": GPU_INDICES,
        "PORT": str(args.port),
        "MODEL": model_mount,
        "SERVED_MODEL_NAME": SERVED_MODEL,
        "TP": "4",
        "DCP": "4",
        "MTP": "3",
        "GPU_MEMORY_UTILIZATION": "0.955",
        "MAX_BATCHED_TOKENS": "2048",
        "MAX_NUM_SEQS": "4",
        "GRAPH": "16",
        "MAX_MODEL_LEN": "262144",
        "MOE_MODE": "a16",
        "QUANTIZATION": "exl3",
        "ONLINE_QUANT": "none",
        "KV_CACHE_DTYPE": "nvfp4_ds_mla",
        "LOAD_FORMAT": "safetensors",
        "KV_FP8_ROPE": "0",
        "VLLM_NVFP4_MLA_SCALES_FILE": "/opt/kv-scale/nvfp4_mla_outer_scales.json",
        "VLLM_EXL3_R7_FUSED": "1",
        "VLLM_EXL3_R7_FUSED_LAYERS": "48",
        "VLLM_EXL3_R7_A1_MIN_ROWS": "0",
        "VLLM_EXL3_PREFILL_CAPACITY": "1024",
        "VLLM_EXL3_PREFILL_BLOCK_M": "64",
        "VLLM_DCP_INDEXER_SHARDS": "4",
        "MTP_MOE_BACKEND": "triton",
        "MTP_DRAFT_SAMPLE_METHOD": "greedy",
        "ASYNC_SCHEDULING": "0",
        "PCIE_CALIBRATION": "auto",
    }
    if args.kill_switch:
        environment["B12X_DISABLE_STANDALONE_K6"] = "1"
    command = [
        "docker",
        "run",
        "-d",
        "--pull=never",
        "--name",
        args.container,
        "--entrypoint",
        "/usr/local/bin/serve-gilded-gnosis.sh",
        "--network",
        "host",
        "--ipc",
        "host",
        "--privileged",
        "--shm-size",
        "64g",
        "--ulimit",
        "memlock=-1:-1",
        "--ulimit",
        "nofile=1048576:1048576",
        "--ulimit",
        "stack=67108864:67108864",
        "--gpus",
        '"device=0,1,2,3"',
        "-v",
        f"{MODEL_DIR}:{model_mount}:ro",
        "-v",
        f"{MODEL_DIR}:/opt/kv-scale:ro",
        "-v",
        f"{cache_dir}:/cache",
    ]
    for name, value in environment.items():
        command.extend(["-e", f"{name}={value}"])
    command.append(args.image)
    return command


def wait_for_health(
    container: str,
    base_url: str,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    next_update = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5.0) as response:
                payload = response.read().decode("utf-8", errors="replace")
            observations.append(
                {
                    "at": utc_now(),
                    "status": response.status,
                    "payload": payload,
                }
            )
            if response.status == 200:
                return observations
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if now >= next_update:
                print(
                    f"{utc_now()} waiting for {container}: {type(exc).__name__}",
                    flush=True,
                )
                next_update = now + 30.0
            observations.append(
                {
                    "at": utc_now(),
                    "status": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        state = capture(
            ["docker", "inspect", "--format", "{{.State.Running}}", container]
        )
        if state["returncode"] or state["stdout"].strip() != "true":
            logs = capture(["docker", "logs", "--tail", "200", container])
            raise RuntimeError(
                f"service exited before health:\n{logs['stdout']}{logs['stderr']}"
            )
        time.sleep(5.0)
    raise TimeoutError(f"service health timeout after {timeout_seconds}s")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--container", required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--image-id", required=True)
    result.add_argument("--arm", required=True)
    result.add_argument("--panel", required=True)
    result.add_argument("--nonce-base", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--qualification-tool-root",
        type=Path,
        default=DEFAULT_QUALIFICATION_TOOL_ROOT,
        help="directory containing the service benchmark and capture tools",
    )
    result.add_argument("--port", type=int, default=18080)
    result.add_argument("--kill-switch", action="store_true")
    result.add_argument("--idle-timeout", type=int, default=120)
    result.add_argument("--health-timeout", type=int, default=1200)
    return result


def main() -> int:
    args = parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    args.qualification_tool_root = args.qualification_tool_root.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"arm evidence already exists: {args.output_dir}")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(MODEL_DIR)
    benchmark_tool = args.qualification_tool_root / "benchmark_openai_stream.py"
    capture_tool = args.qualification_tool_root / "capture_service_evidence.py"
    if not benchmark_tool.is_file() or not capture_tool.is_file():
        raise FileNotFoundError(
            "qualification service tools are missing from "
            f"{args.qualification_tool_root}"
        )
    existing = capture(["docker", "container", "inspect", args.container])
    if existing["returncode"] == 0:
        raise RuntimeError(f"container already exists: {args.container}")
    if not port_available(args.port):
        raise RuntimeError(f"port is already in use: {args.port}")
    inspected_image = capture(["docker", "image", "inspect", args.image], check=True)
    image_payload = json.loads(inspected_image["stdout"])[0]
    if image_payload["Id"] != args.image_id:
        raise RuntimeError(
            f"image identity mismatch: {image_payload['Id']} != {args.image_id}"
        )

    args.output_dir.mkdir(parents=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir()
    receipt_path = args.output_dir / "arm_runner_receipt.json"
    receipt: dict[str, Any] = {
        "schema": "b12x.glm52_o4096x6144_service_arm.v1",
        "started_at": utc_now(),
        "completed_at": None,
        "argv": sys.argv,
        "host": platform.node(),
        "platform": platform.platform(),
        "arm": args.arm,
        "panel": args.panel,
        "nonce_base": args.nonce_base,
        "container": args.container,
        "image": args.image,
        "image_id": args.image_id,
        "kill_switch": args.kill_switch,
        "port": args.port,
        "model_dir": str(MODEL_DIR),
        "qualification_tool_root": str(args.qualification_tool_root),
        "serving_fused_sha256": sha256_file(MODEL_DIR / "SERVING_FUSED.md"),
        "docker_compose_sha256": sha256_file(MODEL_DIR / "docker-compose.yaml"),
        "tools": {
            "benchmark": {
                "path": str(benchmark_tool),
                "sha256": sha256_file(benchmark_tool),
            },
            "capture": {
                "path": str(capture_tool),
                "sha256": sha256_file(capture_tool),
            },
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "image_inspect": image_payload,
        "docker_ps_before": capture(["docker", "ps", "--no-trunc"]),
        "gpu_idle_observations": wait_for_idle(args.idle_timeout),
        "gpu_before": gpu_snapshot(),
        "workloads": [],
    }
    launch_command = service_command(args, cache_dir)
    receipt["launch_command"] = launch_command
    receipt["launch_command_shell"] = shlex.join(launch_command)
    write_json(receipt_path, receipt)

    launched = False
    failure: str | None = None
    try:
        launch = capture(launch_command, check=True)
        launched = True
        receipt["launch"] = launch
        print(f"{utc_now()} launched {args.arm} container {args.container}", flush=True)
        base_url = f"http://127.0.0.1:{args.port}"
        receipt["health_observations"] = wait_for_health(
            args.container, base_url, args.health_timeout
        )
        receipt["healthy_at"] = utc_now()
        write_json(receipt_path, receipt)
        print(f"{utc_now()} {args.arm} is healthy", flush=True)

        for workload, prompt, output, concurrency, warmups, samples in WORKLOADS:
            command = [
                sys.executable,
                str(benchmark_tool),
                "--base-url",
                base_url,
                "--arm",
                args.arm,
                "--stage",
                "mtp3",
                "--workload",
                workload,
                "--prompt-tokens",
                str(prompt),
                "--output-tokens",
                str(output),
                "--concurrency",
                str(concurrency),
                "--warmups",
                str(warmups),
                "--samples",
                str(samples),
                "--nonce-base",
                args.nonce_base,
                "--telemetry-gpus",
                GPU_INDICES,
                "--output",
                str(args.output_dir / f"{workload}.json"),
            ]
            print(f"{utc_now()} {args.arm} starting {workload}", flush=True)
            started_at = utc_now()
            result = capture(command)
            stdout_path = args.output_dir / f"{workload}.stdout.log"
            stderr_path = args.output_dir / f"{workload}.stderr.log"
            stdout_path.write_text(result.pop("stdout"), encoding="utf-8")
            stderr_path.write_text(result.pop("stderr"), encoding="utf-8")
            result.update(
                {
                    "workload": workload,
                    "started_at": started_at,
                    "completed_at": utc_now(),
                    "stdout_path": str(stdout_path),
                    "stdout_sha256": sha256_file(stdout_path),
                    "stderr_path": str(stderr_path),
                    "stderr_sha256": sha256_file(stderr_path),
                }
            )
            receipt["workloads"].append(result)
            write_json(receipt_path, receipt)
            if result["returncode"]:
                raise RuntimeError(f"{workload} failed; see {stderr_path}")
            print(f"{utc_now()} {args.arm} completed {workload}", flush=True)

        capture_command = [
            sys.executable,
            str(capture_tool),
            "--container",
            args.container,
            "--arm",
            args.arm,
            "--base-url",
            base_url,
            "--output-dir",
            str(args.output_dir),
        ]
        receipt["service_capture"] = capture(capture_command, check=True)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        receipt["failure"] = failure
        raise
    finally:
        if launched:
            receipt["container_logs_final"] = capture(
                ["docker", "logs", args.container]
            )
            stop = capture(["docker", "stop", "--time", "30", args.container])
            receipt["stop"] = stop
            receipt["container_final_inspect"] = capture(
                ["docker", "container", "inspect", args.container]
            )
            receipt["remove"] = capture(["docker", "rm", args.container])
        receipt["gpu_after"] = gpu_snapshot()
        receipt["docker_ps_after"] = capture(["docker", "ps", "--no-trunc"])
        receipt["completed_at"] = utc_now()
        receipt["success"] = failure is None
        write_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "success": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
