"""Lazy CUDA-device detection for policy contexts."""

from __future__ import annotations

from dataclasses import dataclass

from .types import DeviceIdentity


@dataclass(frozen=True)
class DetectedDevice:
    ordinal: int | None
    identity: DeviceIdentity | None


_DEVICE_CACHE: dict[int, DeviceIdentity] = {}


def detect_device(device: object | None = None) -> DetectedDevice:
    """Resolve one CUDA device without importing torch at package import."""

    try:
        import torch
    except ImportError:
        return DetectedDevice(ordinal=None, identity=None)
    if not torch.cuda.is_available():
        return DetectedDevice(ordinal=None, identity=None)
    resolved = torch.device("cuda" if device is None else device)
    if resolved.type != "cuda":
        return DetectedDevice(ordinal=None, identity=None)
    ordinal = resolved.index
    if ordinal is None:
        ordinal = int(torch.cuda.current_device())
    identity = _DEVICE_CACHE.get(ordinal)
    if identity is None:
        properties = torch.cuda.get_device_properties(ordinal)
        identity = DeviceIdentity(
            vendor="nvidia",
            compute_capability=(
                int(properties.major),
                int(properties.minor),
            ),
            sm_count=int(properties.multi_processor_count),
            product_name=str(properties.name),
        )
        _DEVICE_CACHE[ordinal] = identity
    return DetectedDevice(ordinal=ordinal, identity=identity)


__all__ = ["DetectedDevice", "detect_device"]
