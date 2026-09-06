"""Generate one resumable, multi-component GPU profile artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

from rich.console import Console
from rich.table import Table

from b12x.policy import (
    EMBEDDED_REGISTRY,
    DetectedDevice,
    DeviceIdentity,
    detect_device,
)
from b12x.policy.generation import (
    ComponentGenerator,
    ComponentGeneratorRegistry,
    GenerationContext,
    GenerationSettings,
    MeasurementPartition,
    measurement_partitions,
    select_measurement_partitions,
)
from b12x.policy.generation.parallel import run_parallel_measurements
from b12x.policy.generation.progress import RichProgressReporter
from b12x.policy.generation.runner import (
    estimate_generators,
    generate_profile_artifact,
    merge_profile_artifacts,
    runtime_profile_payload,
    write_artifact_atomic,
)

_ENTRY_POINT_GROUP = "b12x.profile_generators"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATED_PROFILE_DATA = Path("b12x/policy/_profiles/data")
_GENERATED_PROFILE_PATHSPEC = ":(exclude)b12x/policy/_profiles/data/*.json*"


def _is_generated_profile_data(path: Path) -> bool:
    return path.parent == _GENERATED_PROFILE_DATA and (
        path.name.endswith(".json") or path.name.endswith(".json.gz")
    )


def _package_source_revision() -> str:
    try:
        version = metadata.version("b12x")
    except metadata.PackageNotFoundError:
        version = "uninstalled"
    fingerprint = hashlib.sha256()
    package_root = _REPO_ROOT / "b12x"
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        fingerprint.update(str(path.relative_to(package_root)).encode())
        fingerprint.update(path.read_bytes())
    return f"package.{version}.{fingerprint.hexdigest()[:16]}"


def _source_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "HEAD",
                "--",
                "b12x",
                "benchmarks",
                "pyproject.toml",
                _GENERATED_PROFILE_PATHSPEC,
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "b12x",
                "benchmarks",
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError):
        return _package_source_revision()
    fingerprint = hashlib.sha256(diff)
    for raw_path in sorted(path for path in untracked if path):
        relative_path = Path(os.fsdecode(raw_path))
        if _is_generated_profile_data(relative_path):
            continue
        path = _REPO_ROOT / relative_path
        if not path.is_file():
            continue
        fingerprint.update(raw_path)
        fingerprint.update(path.read_bytes())
    digest = fingerprint.hexdigest()
    return (
        revision
        if not diff and not any(untracked)
        else f"{revision}-worktree.{digest[:16]}"
    )


def _profile_id(product_name: str, sm_count: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", product_name.casefold()).strip(".")
    return f"{slug}.{sm_count}sm"


def _profile_id_for_device(device: DeviceIdentity) -> str:
    embedded = EMBEDDED_REGISTRY.find(device)
    if embedded is not None:
        return embedded.profile_id
    return _profile_id(device.product_name, device.sm_count)


def _profile_output_paths(
    profile_id: str,
    *,
    output: Path | None,
    embed: bool,
) -> tuple[Path, Path]:
    embedded_output = (
        _REPO_ROOT
        / "b12x"
        / "policy"
        / "_profiles"
        / "data"
        / f"{profile_id}.json.gz"
    ).resolve()
    if output is not None:
        return output.resolve(), embedded_output
    if embed:
        return embedded_output, embedded_output
    generated_output = (
        Path("validation/gpu_profiles/generated") / f"{profile_id}.json.gz"
    ).resolve()
    return generated_output, embedded_output


def _load_registry() -> ComponentGeneratorRegistry:
    from b12x.policy.generation.providers import register_builtin_generators

    registry = ComponentGeneratorRegistry()
    register_builtin_generators(registry)
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=_ENTRY_POINT_GROUP)
    for entry_point in sorted(selected, key=lambda item: item.name):
        loaded = entry_point.load()
        generator = loaded() if isinstance(loaded, type) else loaded
        registry.register(generator)
    return registry


def _parse_components(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip().casefold() == "all":
        return None
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--components must select at least one component")
    return values


def _parse_devices(raw: str) -> tuple[str, ...]:
    value = raw.strip().casefold()
    if value == "all":
        import torch

        count = torch.cuda.device_count()
        if count <= 0:
            raise ValueError("--devices all found no visible CUDA GPUs")
        return tuple(f"cuda:{ordinal}" for ordinal in range(count))

    ordinals: list[int] = []
    for item in (part.strip().casefold() for part in raw.split(",")):
        if not item:
            continue
        if item.startswith("cuda:"):
            item = item.removeprefix("cuda:")
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if match is None:
            raise ValueError("--devices must be 'all' or a list such as 0-3,5,7")
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        if last < first:
            raise ValueError("--devices ranges must be ascending")
        ordinals.extend(range(first, last + 1))
    if not ordinals:
        raise ValueError("--devices must select at least one CUDA GPU")
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("--devices contains duplicate CUDA ordinals")
    return tuple(f"cuda:{ordinal}" for ordinal in ordinals)


def _parse_partition_shard(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", raw.strip())
    if match is None:
        raise ValueError("--partition-shard must look like 1/2")
    ordinal, count = (int(value) for value in match.groups())
    if ordinal > count:
        raise ValueError("--partition-shard index cannot exceed its shard count")
    return ordinal - 1, count


def _select_partition_shard(
    generators: tuple[ComponentGenerator, ...],
    context: GenerationContext,
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[ComponentGenerator, ...]:
    partitions = measurement_partitions(generators, context)
    if shard_count > len(partitions):
        raise ValueError(
            f"cannot split {len(partitions)} measurement partitions into "
            f"{shard_count} shards"
        )
    assignments: list[list[MeasurementPartition]] = [
        [] for _ in range(shard_count)
    ]
    loads = [0] * shard_count
    for partition in sorted(
        partitions,
        key=lambda item: (
            -item.work_units,
            item.component_id,
            item.partition_id,
        ),
    ):
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[target].append(partition)
        loads[target] += partition.work_units

    selected_by_component: dict[str, list[str]] = defaultdict(list)
    for partition in assignments[shard_index]:
        selected_by_component[partition.component_id].append(partition.partition_id)
    return tuple(
        select_measurement_partitions(
            generator,
            tuple(sorted(selected_by_component[generator.component_id])),
        )
        for generator in generators
        if generator.component_id in selected_by_component
    )


def _detect_devices(device_specs: tuple[str, ...]) -> tuple[DetectedDevice, ...]:
    if not device_specs:
        raise ValueError("at least one CUDA device is required")
    detected_devices = tuple(detect_device(spec) for spec in device_specs)
    if any(
        detected.identity is None or detected.ordinal is None
        for detected in detected_devices
    ):
        unresolved = [
            spec
            for spec, detected in zip(device_specs, detected_devices, strict=True)
            if detected.identity is None or detected.ordinal is None
        ]
        raise ValueError(f"CUDA devices did not resolve: {', '.join(unresolved)}")
    identity = detected_devices[0].identity
    if any(detected.identity != identity for detected in detected_devices[1:]):
        descriptions = ", ".join(
            f"{spec}={detected.identity}"
            for spec, detected in zip(device_specs, detected_devices, strict=True)
        )
        raise ValueError(
            "parallel profile generation requires identical GPUs; " + descriptions
        )
    return detected_devices


def _render_estimates(
    console: Console,
    generators: tuple[ComponentGenerator, ...],
    context: GenerationContext,
) -> None:
    estimates = estimate_generators(generators, context)
    table = Table(title="GPU profile generation plan")
    table.add_column("Component")
    table.add_column("Cases", justify="right")
    table.add_column("Work units", justify="right")
    table.add_column("Scope")
    for estimate in estimates:
        table.add_row(
            estimate.component_id,
            f"{estimate.case_count:,}",
            f"{estimate.work_units:,}",
            estimate.description,
        )
    table.add_section()
    table.add_row(
        "total",
        f"{sum(item.case_count for item in estimates):,}",
        f"{sum(item.work_units for item in estimates):,}",
        "",
    )
    console.print(table)


def _render_parallel_plan(
    console: Console,
    partitions: tuple[MeasurementPartition, ...],
    worker_count: int,
) -> None:
    table = Table(title="Parallel measurement plan")
    table.add_column("GPU workers", justify="right")
    table.add_column("Partitions", justify="right")
    table.add_column("Measurement cases", justify="right")
    table.add_column("Largest partition", justify="right")
    table.add_row(
        f"{worker_count:,}",
        f"{len(partitions):,}",
        f"{sum(item.case_count for item in partitions):,}",
        f"{max(item.case_count for item in partitions):,}",
    )
    console.print(table)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        help="single CUDA device (default: current device)",
    )
    device_group.add_argument(
        "--devices",
        help="parallel CUDA ordinals, for example 0-11,13 or all",
    )
    parser.add_argument("--profile-id")
    parser.add_argument("--components", default="all")
    parser.add_argument(
        "--partition-shard",
        help="measure one workload-balanced cross-host shard, for example 1/2",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--merge-from",
        type=Path,
        help="base full profile whose unselected components are retained",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--minimum-cosine", type=float, default=0.998)
    parser.add_argument(
        "--max-candidate-seconds",
        type=float,
        default=2.0,
        help="cap timed replay work per candidate while retaining every group",
    )
    parser.add_argument(
        "--cold-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--embed",
        action="store_true",
        help=(
            "write into b12x package data; this is the sole output unless "
            "--output is provided"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-components", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    console = Console()
    registry = _load_registry()
    if args.list_components:
        for component_id in registry.component_ids():
            console.print(component_id)
        return 0
    try:
        selected_ids = _parse_components(args.components)
        generators = registry.select(selected_ids)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not generators:
        raise SystemExit(
            "no component profile generators are installed; "
            f"register providers through {_ENTRY_POINT_GROUP!r}"
        )

    try:
        device_specs = (
            _parse_devices(args.devices)
            if args.devices is not None
            else (args.device or "cuda",)
        )
        detected_devices = _detect_devices(device_specs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    detected = detected_devices[0]
    if detected.identity is None or detected.ordinal is None:
        raise RuntimeError("validated CUDA device lost its identity")
    profile_id = args.profile_id or _profile_id_for_device(detected.identity)
    work_dir = (args.work_dir or Path(".b12x-profile-work") / profile_id).resolve()
    output, embedded_output = _profile_output_paths(
        profile_id,
        output=args.output,
        embed=args.embed,
    )
    merge_from = None if args.merge_from is None else args.merge_from.resolve()
    if selected_ids is not None and merge_from is None:
        if output.exists():
            merge_from = output
        elif args.embed and embedded_output.exists():
            merge_from = embedded_output
    if args.embed and selected_ids is not None and merge_from is None:
        raise SystemExit(
            "embedding a component subset requires an existing output profile "
            "or --merge-from"
        )
    if not args.dry_run and output.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing profile {output}; "
            "pass --overwrite or choose --output"
        )
    if (
        args.embed
        and not args.dry_run
        and embedded_output != output
        and embedded_output.exists()
        and not args.overwrite
    ):
        raise SystemExit(
            f"refusing to overwrite embedded profile {embedded_output}; "
            "pass --overwrite"
        )
    context = GenerationContext(
        device=detected.identity,
        device_ordinal=detected.ordinal,
        work_dir=work_dir,
        source_revision=_source_revision(),
        settings=GenerationSettings(
            warmup=args.warmup,
            repetitions=args.repetitions,
            groups=args.groups,
            seed=args.seed,
            minimum_cosine=args.minimum_cosine,
            cold_l2=args.cold_l2,
            max_candidate_seconds=args.max_candidate_seconds,
        ),
    )
    if args.partition_shard is not None:
        try:
            shard_index, shard_count = _parse_partition_shard(args.partition_shard)
            generators = _select_partition_shard(
                generators,
                context,
                shard_index=shard_index,
                shard_count=shard_count,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        console.print(
            f"Partition shard: {shard_index + 1}/{shard_count} "
            f"({len(measurement_partitions(generators, context))} partitions)"
        )
    device_labels = ", ".join(f"cuda:{item.ordinal}" for item in detected_devices)
    console.print(
        f"[bold]{profile_id}[/bold] on {device_labels} "
        f"({detected.identity.product_name}, {detected.identity.sm_count} SMs)"
    )
    console.print(f"Checkpoint directory: {work_dir}")
    if merge_from is not None:
        console.print(f"Merge base: {merge_from}")
    _render_estimates(console, generators, context)
    partitions = measurement_partitions(generators, context)
    if len(detected_devices) > 1:
        _render_parallel_plan(
            console,
            partitions,
            min(len(detected_devices), len(partitions)),
        )
    if args.dry_run:
        return 0

    estimates = estimate_generators(generators, context)
    parallel_summary = None
    try:
        if len(detected_devices) > 1:
            parallel_summary = run_parallel_measurements(
                console=console,
                device_specs=device_specs,
                generators=generators,
                context=context,
                registry_factory=_load_registry,
            )
            console.print("Parallel measurements complete; assembling profile.")
        with RichProgressReporter(estimates) as progress:
            artifact = generate_profile_artifact(
                profile_id=profile_id,
                generators=generators,
                context=context,
                progress=progress,
            )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted.[/yellow] Completed races are checkpointed; "
            f"rerun with the same --work-dir {work_dir} to resume."
        )
        return 130
    if parallel_summary is not None:
        evidence = artifact.get("evidence")
        if not isinstance(evidence, dict):
            raise TypeError("generated artifact evidence must be mutable")
        evidence["parallel_measurement"] = {
            "device_ordinals": list(parallel_summary.device_ordinals),
            "partition_count": parallel_summary.partition_count,
            "worker_count": parallel_summary.worker_count,
        }
    if merge_from is not None:
        raw = merge_from.read_bytes()
        if merge_from.suffix == ".gz":
            raw = gzip.decompress(raw)
        base_artifact = json.loads(raw)
        if not isinstance(base_artifact, Mapping):
            raise TypeError("merge base must contain a JSON object")
        artifact = merge_profile_artifacts(base_artifact, artifact)
    profile = artifact["profile"]
    if not isinstance(profile, Mapping):
        raise TypeError("generated artifact profile must be an object")
    output_is_embedded = args.embed and embedded_output == output
    embedded_profile = runtime_profile_payload(profile)
    write_artifact_atomic(
        output,
        embedded_profile if output_is_embedded else artifact,
        overwrite=args.overwrite,
        compact=output_is_embedded,
    )
    console.print(f"Wrote [bold]{output}[/bold]")
    if args.embed and embedded_output != output:
        write_artifact_atomic(
            embedded_output,
            embedded_profile,
            overwrite=args.overwrite,
            compact=True,
        )
        console.print(f"Embedded [bold]{embedded_output}[/bold]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
