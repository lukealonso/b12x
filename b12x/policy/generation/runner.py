"""Top-level orchestration and artifact assembly for GPU profile generation."""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from b12x.policy.serialization import profile_from_dict

from .contracts import (
    ComponentGenerator,
    GenerationContext,
    ProgressReporter,
    WorkEstimate,
)
from .store import CheckpointStore


def estimate_generators(
    generators: tuple[ComponentGenerator, ...],
    context: GenerationContext,
) -> tuple[WorkEstimate, ...]:
    estimates = tuple(generator.estimate(context) for generator in generators)
    expected = tuple(generator.component_id for generator in generators)
    actual = tuple(estimate.component_id for estimate in estimates)
    if actual != expected:
        raise ValueError(
            "component generators must estimate their own component; "
            f"expected {expected}, got {actual}"
        )
    return estimates


def generate_profile_artifact(
    *,
    profile_id: str,
    generators: tuple[ComponentGenerator, ...],
    context: GenerationContext,
    progress: ProgressReporter,
) -> dict[str, object]:
    """Run all selected providers and assemble one validated profile artifact."""

    estimates = estimate_generators(generators, context)
    checkpoints = CheckpointStore(context.work_dir / "checkpoints")
    components: list[Mapping[str, object]] = []
    component_evidence: dict[str, object] = {}
    for generator, estimate in zip(generators, estimates, strict=True):
        progress.start_component(estimate)
        result = generator.generate(
            context,
            progress=progress,
            checkpoints=checkpoints,
        )
        if result.completed_work_units != estimate.work_units:
            raise ValueError(
                f"generator {generator.component_id!r} completed "
                f"{result.completed_work_units} work units, expected "
                f"{estimate.work_units}"
            )
        component = result.component
        if component.get("component_id") != generator.component_id:
            raise ValueError(
                f"generator {generator.component_id!r} emitted component "
                f"{component.get('component_id')!r}"
            )
        if component.get("query_schema_version") != generator.query_schema_version:
            raise ValueError(
                f"generator {generator.component_id!r} emitted the wrong "
                "query schema version"
            )
        if component.get("config_schema_version") != generator.config_schema_version:
            raise ValueError(
                f"generator {generator.component_id!r} emitted the wrong "
                "config schema version"
            )
        gpu_measurement_cases = result.evidence.get("gpu_measurement_cases")
        if (
            not isinstance(gpu_measurement_cases, int)
            or isinstance(gpu_measurement_cases, bool)
            or gpu_measurement_cases <= 0
        ):
            raise ValueError(
                f"generator {generator.component_id!r} did not report any "
                "production-path GPU measurements"
            )
        if "precomputed_cases" in result.evidence:
            raise ValueError(
                f"generator {generator.component_id!r} reported precomputed "
                "results; generated profiles require real GPU measurements"
            )
        components.append(component)
        component_evidence[generator.component_id] = result.evidence
        progress.finish_component(generator.component_id)

    identity = context.device
    profile: dict[str, object] = {
        "profile_id": profile_id,
        "targets": [
            {
                "vendor": identity.vendor,
                "compute_capability": list(identity.compute_capability),
                "sm_count": identity.sm_count,
                "product_name": identity.product_name,
            }
        ],
        "components": components,
        "metadata": {
            "generated_by": "b12x-generate-gpu-profile",
            "source_revision": context.source_revision,
        },
    }
    profile_from_dict(profile)
    return {
        "schema_version": 1,
        "profile": profile,
        "evidence": {
            "device_ordinal": context.device_ordinal,
            "settings": context.settings.to_dict(),
            "components": component_evidence,
        },
    }


def runtime_profile_payload(profile: Mapping[str, object]) -> dict[str, object]:
    """Return the minimal planner payload embedded in the runtime package."""

    def strip_audit_fields(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: strip_audit_fields(item)
                for key, item in value.items()
                if key not in {"coverage", "evidence", "metadata"}
            }
        if isinstance(value, list):
            return [strip_audit_fields(item) for item in value]
        return value

    payload = strip_audit_fields(profile)
    if not isinstance(payload, dict):
        raise TypeError("runtime profile payload must be an object")
    profile_from_dict(payload)
    return payload


def merge_profile_artifacts(
    base: Mapping[str, object],
    update: Mapping[str, object],
) -> dict[str, object]:
    """Replace generated components in an existing full profile artifact."""

    def profile_payload(artifact: Mapping[str, object]) -> Mapping[str, object]:
        value = artifact.get("profile", artifact)
        if not isinstance(value, Mapping):
            raise TypeError("profile artifact must contain a profile object")
        return value

    base_profile = profile_payload(base)
    update_profile = profile_payload(update)
    parsed_base = profile_from_dict(base_profile)
    parsed_update = profile_from_dict(update_profile)
    if parsed_base.profile_id != parsed_update.profile_id:
        raise ValueError("cannot merge profiles with different profile IDs")
    if not frozenset(parsed_update.targets) <= frozenset(parsed_base.targets):
        raise ValueError(
            "cannot merge a profile whose device targets are not owned by "
            "the base profile"
        )

    components = {
        str(component["component_id"]): component
        for component in base_profile["components"]
    }
    components.update(
        {
            str(component["component_id"]): component
            for component in update_profile["components"]
        }
    )
    metadata = dict(base_profile.get("metadata", {}))
    metadata.update(update_profile.get("metadata", {}))
    merged_profile: dict[str, object] = {
        "profile_id": parsed_base.profile_id,
        "targets": list(base_profile["targets"]),
        "components": [components[key] for key in sorted(components)],
    }
    if metadata:
        merged_profile["metadata"] = metadata
    profile_from_dict(merged_profile)

    evidence = dict(base.get("evidence", {}))
    update_evidence = update.get("evidence", {})
    if not isinstance(update_evidence, Mapping):
        raise TypeError("generated artifact evidence must be an object")
    base_components = evidence.get("components", {})
    update_components = update_evidence.get("components", {})
    if not isinstance(base_components, Mapping) or not isinstance(
        update_components, Mapping
    ):
        raise TypeError("artifact component evidence must be an object")
    evidence.update(update_evidence)
    evidence["components"] = {**base_components, **update_components}
    return {
        "schema_version": int(update.get("schema_version", 1)),
        "profile": merged_profile,
        "evidence": evidence,
    }


def write_artifact_atomic(
    path: Path,
    artifact: Mapping[str, object],
    *,
    overwrite: bool,
    compact: bool = False,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing profile {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    if compact:
        serialized = json.dumps(
            artifact,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    else:
        serialized = json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    payload = (serialized + "\n").encode("utf-8")
    if path.suffix == ".gz":
        temporary.write_bytes(gzip.compress(payload, mtime=0))
    else:
        temporary.write_bytes(payload)
    os.replace(temporary, path)


__all__ = [
    "estimate_generators",
    "generate_profile_artifact",
    "merge_profile_artifacts",
    "runtime_profile_payload",
    "write_artifact_atomic",
]
