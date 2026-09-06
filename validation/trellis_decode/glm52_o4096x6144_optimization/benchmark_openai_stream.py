#!/usr/bin/env python3
"""Measure an OpenAI-compatible vLLM endpoint with cache-proof prompts.

The client intentionally depends only on Python's standard library.  It records
the raw SSE events, client-side timing, vLLM counter deltas, and GPU telemetry
needed to audit a serving benchmark after the fact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MODEL = "GLM-5.2-EXL3-TR3v4-3.5bpw"
SAFE_TOKENS = (30903, 2404, 21678, 825, 1378, 2326, 13)
METRIC_NAMES = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)
METRIC_SOURCES = (
    "local_compute",
    "local_cache_hit",
    "external_kv_transfer",
)
METRIC_LINE = re.compile(
    r"^(?P<name>[^ {]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
    }


def http_bytes(url: str, *, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def scrape_metrics(base_url: str) -> dict[str, float]:
    text = http_bytes(f"{base_url}/metrics").decode("utf-8")
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        labels = match.group("labels") or ""
        value = float(match.group("value"))
        if name in METRIC_NAMES:
            values[name] = values.get(name, 0.0) + value
        if name == "vllm:prompt_tokens_by_source_total":
            for source in METRIC_SOURCES:
                if f'source="{source}"' in labels:
                    values[f"{name}:{source}"] = value
    return values


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(set(before) | set(after))
    }


def cache_proof_prompt(length: int, nonce: str) -> list[int]:
    if length < 1:
        raise ValueError("prompt length must be positive")
    # The first 64 tokens are derived from the nonce.  Prefix caching can only
    # reuse contiguous blocks from token zero, so a unique first block prevents
    # the repeated body from becoming a cache hit while preserving exact length.
    digest = hashlib.shake_256(nonce.encode("utf-8")).digest(64)
    prefix = [SAFE_TOKENS[value % len(SAFE_TOKENS)] for value in digest]
    body = [SAFE_TOKENS[index % len(SAFE_TOKENS)] for index in range(length)]
    return (prefix + body)[:length]


@dataclass
class RequestResult:
    request_index: int
    nonce: str
    prompt_tokens_requested: int
    output_tokens_requested: int
    prompt_sha256: str
    prompt_prefix: list[int]
    prompt_suffix: list[int]
    start_monotonic_ns: int
    first_token_monotonic_ns: int | None
    end_monotonic_ns: int
    ttft_seconds: float | None
    e2e_seconds: float
    decode_window_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    prefill_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    finish_reason: str | None
    response_id: str | None
    system_fingerprint: str | None
    text_sha256: str
    text_preview: str
    sse_events: list[dict[str, Any]]
    error: str | None


def run_request(
    *,
    base_url: str,
    request_index: int,
    nonce: str,
    prompt_tokens: int,
    output_tokens: int,
    timeout: float,
) -> RequestResult:
    prompt = cache_proof_prompt(prompt_tokens, nonce)
    prompt_bytes = json.dumps(prompt, separators=(",", ":")).encode("ascii")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_ns = time.monotonic_ns()
    first_token_ns: int | None = None
    end_ns = start_ns
    events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    prompt_count: int | None = None
    completion_count: int | None = None
    cached_count: int | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                received_ns = time.monotonic_ns()
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    end_ns = received_ns
                    break
                event = json.loads(data)
                event["client_received_monotonic_ns"] = received_ns
                events.append(event)
                response_id = event.get("id") or response_id
                system_fingerprint = (
                    event.get("system_fingerprint") or system_fingerprint
                )
                usage = event.get("usage") or {}
                prompt_count = usage.get("prompt_tokens", prompt_count)
                new_completion_count = usage.get("completion_tokens")
                if new_completion_count is not None:
                    completion_count = new_completion_count
                    if new_completion_count > 0 and first_token_ns is None:
                        first_token_ns = received_ns
                details = usage.get("prompt_tokens_details") or {}
                cached_count = details.get("cached_tokens", cached_count)
                for choice in event.get("choices") or []:
                    text_parts.append(choice.get("text") or "")
                    finish_reason = choice.get("finish_reason") or finish_reason
            else:
                end_ns = time.monotonic_ns()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        end_ns = time.monotonic_ns()
        error = f"{type(exc).__name__}: {exc}"

    text = "".join(text_parts)
    ttft = (first_token_ns - start_ns) / 1e9 if first_token_ns else None
    e2e = (end_ns - start_ns) / 1e9
    decode_window = (end_ns - first_token_ns) / 1e9 if first_token_ns else None
    prefill_rate = prompt_count / ttft if prompt_count and ttft and ttft > 0 else None
    decode_rate = None
    if completion_count and decode_window and decode_window > 0:
        decode_rate = max(completion_count - 1, 0) / decode_window
    return RequestResult(
        request_index=request_index,
        nonce=nonce,
        prompt_tokens_requested=prompt_tokens,
        output_tokens_requested=output_tokens,
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        prompt_prefix=prompt[:64],
        prompt_suffix=prompt[-64:],
        start_monotonic_ns=start_ns,
        first_token_monotonic_ns=first_token_ns,
        end_monotonic_ns=end_ns,
        ttft_seconds=ttft,
        e2e_seconds=e2e,
        decode_window_seconds=decode_window,
        prompt_tokens=prompt_count,
        completion_tokens=completion_count,
        cached_tokens=cached_count,
        prefill_tokens_per_second=prefill_rate,
        decode_tokens_per_second=decode_rate,
        finish_reason=finish_reason,
        response_id=response_id,
        system_fingerprint=system_fingerprint,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        text_preview=text[:240],
        sse_events=events,
        error=error,
    )


class GpuTelemetry:
    QUERY_FIELDS = (
        "timestamp,index,uuid,pstate,power.draw,power.limit,clocks.current.sm,"
        "clocks.current.memory,clocks_throttle_reasons.active,temperature.gpu,"
        "utilization.gpu,memory.used"
    )

    def __init__(self, gpu_indices: str, interval: float = 0.25) -> None:
        self.gpu_indices = gpu_indices
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            started_ns = time.monotonic_ns()
            command = [
                "nvidia-smi",
                f"--id={self.gpu_indices}",
                f"--query-gpu={self.QUERY_FIELDS}",
                "--format=csv,noheader,nounits",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.samples.append(
                {
                    "client_monotonic_ns": started_ns,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.splitlines(),
                    "stderr": completed.stderr.strip(),
                }
            )
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(self.interval * 3, 2.0))


def run_round(
    *,
    args: argparse.Namespace,
    round_kind: str,
    round_index: int,
) -> dict[str, Any]:
    before = scrape_metrics(args.base_url)
    telemetry = GpuTelemetry(args.telemetry_gpus, args.telemetry_interval)
    telemetry.start()
    round_start = time.monotonic_ns()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                run_request,
                base_url=args.base_url,
                request_index=index,
                nonce=(
                    f"{args.stage}:{args.workload}:{args.nonce_base}:"
                    f"{round_kind}:{round_index}:{index}"
                ),
                prompt_tokens=args.prompt_tokens,
                output_tokens=args.output_tokens,
                timeout=args.timeout,
            )
            for index in range(args.concurrency)
        ]
        requests = [future.result() for future in futures]
    round_end = time.monotonic_ns()
    telemetry.stop()
    after = scrape_metrics(args.base_url)
    deltas = metric_delta(before, after)
    draft_tokens = deltas.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted_tokens = deltas.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    return {
        "kind": round_kind,
        "index": round_index,
        "started_monotonic_ns": round_start,
        "ended_monotonic_ns": round_end,
        "wall_seconds": (round_end - round_start) / 1e9,
        "metrics_before": before,
        "metrics_after": after,
        "metrics_delta": deltas,
        "strict_acceptance_rate": (
            accepted_tokens / draft_tokens if draft_tokens > 0 else None
        ),
        "requests": [asdict(result) for result in requests],
        "gpu_telemetry": telemetry.samples,
    }


def validate_round(round_result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    requests = round_result["requests"]
    if len(requests) != args.concurrency:
        failures.append(f"expected {args.concurrency} requests, got {len(requests)}")
    for result in requests:
        prefix = f"request {result['request_index']}"
        if result["error"]:
            failures.append(f"{prefix}: {result['error']}")
        if result["prompt_tokens"] != args.prompt_tokens:
            failures.append(
                f"{prefix}: prompt tokens {result['prompt_tokens']} != {args.prompt_tokens}"
            )
        if result["completion_tokens"] != args.output_tokens:
            failures.append(
                f"{prefix}: completion tokens {result['completion_tokens']} "
                f"!= {args.output_tokens}"
            )
        if result["cached_tokens"] not in (0, None):
            failures.append(f"{prefix}: cached tokens {result['cached_tokens']} != 0")
        if result["finish_reason"] != "length":
            failures.append(
                f"{prefix}: finish reason {result['finish_reason']!r} != 'length'"
            )
    delta = round_result["metrics_delta"]
    expected_prompt_tokens = args.prompt_tokens * args.concurrency
    local_compute = delta.get("vllm:prompt_tokens_by_source_total:local_compute", 0.0)
    local_cache = delta.get("vllm:prompt_tokens_by_source_total:local_cache_hit", 0.0)
    if local_compute != expected_prompt_tokens:
        failures.append(
            f"local-compute prompt delta {local_compute} != {expected_prompt_tokens}"
        )
    if local_cache != 0:
        failures.append(f"local-cache-hit prompt delta {local_cache} != 0")
    return failures


def make_summary(
    rounds: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    measured = [item for item in rounds if item["kind"] == "measured"]
    requests = [request for item in measured for request in item["requests"]]
    successful = [request for request in requests if request["error"] is None]
    fields = {
        "ttft_seconds": [item["ttft_seconds"] for item in successful],
        "e2e_seconds": [item["e2e_seconds"] for item in successful],
        "decode_window_seconds": [item["decode_window_seconds"] for item in successful],
        "prefill_tokens_per_second": [
            item["prefill_tokens_per_second"] for item in successful
        ],
        "decode_tokens_per_second": [
            item["decode_tokens_per_second"] for item in successful
        ],
    }
    metric_deltas: dict[str, float] = {}
    for item in measured:
        for name, value in item["metrics_delta"].items():
            metric_deltas[name] = metric_deltas.get(name, 0.0) + value
    draft_tokens = metric_deltas.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted_tokens = metric_deltas.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0.0
    )
    return {
        "measured_rounds": len(measured),
        "measured_requests": len(requests),
        "successful_requests": len(successful),
        "request_metrics": {
            name: summarize([float(value) for value in values if value is not None])
            for name, values in fields.items()
        },
        "metrics_delta_total": metric_deltas,
        "strict_acceptance_rate": (
            accepted_tokens / draft_tokens if draft_tokens > 0 else None
        ),
        "validation_failures": [
            failure for item in measured for failure in validate_round(item, args)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--stage", choices=("mtp0", "mtp3"), required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--nonce-base", required=True)
    parser.add_argument("--telemetry-gpus", default="0,1,2,3")
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.warmups < 0 or args.samples < 1:
        parser.error("concurrency/samples must be positive and warmups nonnegative")
    if args.prompt_tokens < 1 or args.output_tokens < 1:
        parser.error("prompt/output tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    health = http_bytes(f"{args.base_url}/health", timeout=10.0)
    models = json.loads(http_bytes(f"{args.base_url}/v1/models", timeout=10.0))
    document: dict[str, Any] = {
        "schema_version": 2,
        "started_at": utc_now(),
        "client": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "argv": sys.argv,
        },
        "configuration": {
            "base_url": args.base_url,
            "arm": args.arm,
            "stage": args.stage,
            "workload": args.workload,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "concurrency": args.concurrency,
            "warmups": args.warmups,
            "samples": args.samples,
            "nonce_base": args.nonce_base,
            "telemetry_gpus": args.telemetry_gpus,
            "telemetry_interval": args.telemetry_interval,
        },
        "health_response": health.decode("utf-8", errors="replace"),
        "models_response": models,
        "rounds": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(args.warmups):
            document["rounds"].append(
                run_round(args=args, round_kind="warmup", round_index=index)
            )
        for index in range(args.samples):
            document["rounds"].append(
                run_round(args=args, round_kind="measured", round_index=index)
            )
    finally:
        document["finished_at"] = utc_now()
        document["summary"] = make_summary(document["rounds"], args)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
    failures = document["summary"]["validation_failures"]
    print(json.dumps(document["summary"], indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
