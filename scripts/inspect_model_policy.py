#!/usr/bin/env python3
"""Inspect model-level kernel selections from GPU policies."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from b12x.tools.inspect_model_policy import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
