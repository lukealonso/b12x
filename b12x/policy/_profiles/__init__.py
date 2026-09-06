"""Generated GPU profiles shipped with b12x."""

from __future__ import annotations

import gzip
import json
from importlib.resources import files

from ..catalog import list_profiled_components
from ..registry import EMBEDDED_REGISTRY
from ..serialization import profile_from_dict


def _load_embedded_profiles() -> None:
    data_root = files(__package__).joinpath("data")
    for resource in sorted(data_root.iterdir(), key=lambda item: item.name):
        if resource.name.endswith(".json.gz"):
            payload = json.loads(gzip.decompress(resource.read_bytes()))
        elif resource.name.endswith(".json"):
            payload = json.loads(resource.read_text(encoding="utf-8"))
        else:
            continue
        if not isinstance(payload, dict):
            raise TypeError(f"embedded GPU profile {resource.name!r} must be an object")
        profile_payload = payload.get("profile", payload)
        EMBEDDED_REGISTRY.register(profile_from_dict(profile_payload))

    expected = {
        str(registration.component_id) for registration in list_profiled_components()
    }
    for profile in EMBEDDED_REGISTRY.list_profiles():
        actual = {component.component_id for component in profile.components}
        if actual != expected:
            raise ValueError(
                f"embedded profile {profile.profile_id!r} component drift; "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}"
            )


_load_embedded_profiles()
