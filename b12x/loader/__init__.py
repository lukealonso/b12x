"""Experimental checkpoint loading into CPU-addressable CUDA storage.

Importing this namespace does not initialize CUDA or build the native helper.
"""

from __future__ import annotations

__all__ = ["capabilities", "read_tensor", "storage_stats"]


def __getattr__(name):
    if name in __all__:
        from . import _api

        value = getattr(_api, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
