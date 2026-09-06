"""Atomic, resumable checkpoint storage for offline profile generation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path


def _safe_part(value: str) -> str:
    if not value or value in {".", ".."}:
        raise ValueError("checkpoint path parts must be non-empty")
    if any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in value
    ):
        raise ValueError(f"unsafe checkpoint path part {value!r}")
    return value


class CheckpointStore:
    """Namespaced JSON checkpoints with atomic replacement."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, component_id: str, key: str) -> Path:
        component = _safe_part(component_id)
        leaf = _safe_part(key)
        return self.root / component / f"{leaf}.json"

    def load(self, component_id: str, key: str) -> dict[str, object] | None:
        path = self._path(component_id, key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            raise TypeError(f"checkpoint {path} must contain an object")
        return payload

    def save(
        self,
        component_id: str,
        key: str,
        payload: Mapping[str, object],
    ) -> Path:
        path = self._path(component_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


__all__ = ["CheckpointStore"]
