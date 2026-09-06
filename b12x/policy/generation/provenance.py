"""Physical-device and source identity for offline GPU measurements."""

from __future__ import annotations

import hashlib
import platform
from functools import lru_cache
from pathlib import Path

from b12x.policy.types import FrozenMapping


@lru_cache(maxsize=1)
def measurement_source_sha256() -> str:
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for directory in ("b12x", "benchmarks", "tests"):
        for path in sorted((root / directory).rglob("*.py")):
            name = path.relative_to(root).as_posix().encode()
            content = path.read_bytes()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def physical_device_uuid(ordinal: int) -> str:
    import torch

    value = str(torch.cuda.get_device_properties(ordinal).uuid).strip()
    if not value:
        raise RuntimeError("GPU measurements require a physical CUDA device UUID")
    return value


def capture_measurement_provenance(ordinal: int) -> FrozenMapping:
    from cuda.bindings import driver
    from b12x._lib.compiler import _compile_environment_key, _runtime_toolchain_key

    status, version = driver.cuDriverGetVersion()
    if int(status) != 0:
        raise RuntimeError(f"cannot identify the CUDA driver: {status}")
    return FrozenMapping({
        "source_sha256": measurement_source_sha256(),
        "physical_device": physical_device_uuid(ordinal),
        "toolchain": {"runtime": _runtime_toolchain_key(), "cuda_driver": int(version),
                      "machine": platform.machine(), "environment": _compile_environment_key()},
    })


__all__ = ["capture_measurement_provenance", "measurement_source_sha256", "physical_device_uuid"]
