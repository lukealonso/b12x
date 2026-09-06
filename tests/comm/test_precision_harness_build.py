"""The nine-rank precision harness builds a runtime when any kernel it serves
is selected, and reports a constructor failure with its message instead of
the generic "could not be built"."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[2] / "benchmarks" / "precision_pcie_tp9_allreduce.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("precision_pcie_tp9_allreduce", _HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_serving_several_kernels_is_built_when_wanted(harness) -> None:
    skipped: dict = {}
    built = harness.build_runtime(
        "dma", lambda: "ring", ["dma-ring", "dma-rs-fp32"], skipped, wanted=True
    )
    assert built == "ring"
    assert skipped == {}


def test_runtime_not_selected_is_not_built(harness) -> None:
    skipped: dict = {}
    assert harness.build_runtime("dma", lambda: "ring", ["oneshot"], skipped) is None
    assert skipped == {}


def test_constructor_failure_is_reported_with_its_message(harness) -> None:
    skipped: dict = {}

    def construct():
        raise ValueError("world size 9 needs B12X_PCIE_DMA_GRAPH_REPLAY=1")

    assert harness.build_runtime("dma", construct, [], skipped, wanted=True) is None
    reason = skipped["dma"]
    assert reason.startswith("ValueError while building: world size 9 needs")
    assert "Traceback" in reason and "construct" in reason
