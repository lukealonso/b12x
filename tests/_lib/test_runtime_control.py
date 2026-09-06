"""Serving freeze guards (ported semantics from b12x runtime_control)."""

from __future__ import annotations

import importlib

import pytest

rc = importlib.import_module("b12x._lib.runtime_control")


def test_freeze_blocks_kernel_resolution_with_context():
    assert not rc.kernel_resolution_frozen()
    rc.freeze_kernel_resolution("unit-test")
    try:
        assert rc.kernel_resolution_frozen()
        with pytest.raises(rc.KernelResolutionFrozenError) as excinfo:
            rc.raise_if_kernel_resolution_frozen(
                "compile",
                target=test_freeze_blocks_kernel_resolution_with_context,
                cache_key=("shape", 128),
            )
        message = str(excinfo.value)
        assert "unit-test" in message
        assert "compile" in message
        assert "freeze_kernel_resolution" in message
    finally:
        rc.unfreeze_kernel_resolution()
    assert not rc.kernel_resolution_frozen()
    rc.raise_if_kernel_resolution_frozen("compile")  # no-op when unfrozen


def test_compilation_aliases_are_the_same_functions():
    assert rc.freeze_compilation is rc.freeze_kernel_resolution
    assert rc.unfreeze_compilation is rc.unfreeze_kernel_resolution
    assert rc.compilation_frozen is rc.kernel_resolution_frozen


def test_namespace_root_reexports():
    b12x = importlib.import_module("b12x")
    assert b12x.freeze_kernel_resolution is rc.freeze_kernel_resolution
    assert b12x.KernelResolutionFrozenError is rc.KernelResolutionFrozenError


def test_triton_cache_misses_are_observed_and_frozen_without_replacing_other_hooks(monkeypatch):
    from types import SimpleNamespace
    from triton import knobs
    from b12x.policy.generation.census import collect_compilation_requests

    called = []
    monkeypatch.setattr(knobs.runtime, "jit_cache_hook", lambda **kwargs: called.append(kwargs["key"]))
    def kernel():
        pass
    request = dict(fn=SimpleNamespace(module="b12x.example", name="kernel", jit_function=SimpleNamespace(fn=kernel)),
                   key="specialization", compile={"specialization_data": '{"constants":[128]}'} )
    with collect_compilation_requests() as requests:
        knobs.runtime.jit_cache_hook(**request)
    assert called == ["specialization"]
    assert len(requests) == 1
    assert next(iter(requests.values()))["compile_spec"]["dialect"] == "triton"
    rc.freeze_kernel_resolution("reject unprepared Triton specializations")
    try:
        with pytest.raises(rc.KernelResolutionFrozenError, match="triton.compile"):
            knobs.runtime.jit_cache_hook(**request)
        external = {**request, "fn": SimpleNamespace(module="external", name="kernel")}
        knobs.runtime.jit_cache_hook(**external)
        assert called == ["specialization", "specialization"]
    finally:
        rc.unfreeze_kernel_resolution()
