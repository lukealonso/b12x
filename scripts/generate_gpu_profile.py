#!/usr/bin/env python3
"""Run the resumable multi-component GPU profile generator."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from b12x.tools.generate_gpu_profile import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
