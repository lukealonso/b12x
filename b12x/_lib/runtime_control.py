from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock


class KernelResolutionFrozenError(RuntimeError):
    """Raised when b12x is asked to resolve a new kernel after freeze."""


_STATE_LOCK = Lock()
_FROZEN = False
_FREEZE_REASON: str | None = None
_COMPILATION_OBSERVERS = ContextVar("b12x_compilation_observers", default=())
_TRITON_CACHE_HOOK = None


def _ensure_triton_cache_hook() -> None:
    import json
    from types import SimpleNamespace
    from triton import knobs

    global _TRITON_CACHE_HOOK
    with _STATE_LOCK:
        previous = knobs.runtime.jit_cache_hook
        if previous is _TRITON_CACHE_HOOK and _TRITON_CACHE_HOOK is not None:
            return

        def hook(**kwargs):
            function = kwargs["fn"]
            if str(function.module).startswith("b12x."):
                target = function.jit_function.fn
                if _COMPILATION_OBSERVERS.get():
                    spec = {"dialect": "triton", "kernel": f"{function.module}.{function.name}",
                            "specialization": json.loads(kwargs["compile"]["specialization_data"])}
                    report_compilation_request(target, SimpleNamespace(
                        json_key=json.dumps(spec, sort_keys=True, separators=(",", ":"))))
                raise_if_kernel_resolution_frozen("triton.compile", target=target, cache_key=kwargs["key"])
            return None if previous is None else previous(**kwargs)

        _TRITON_CACHE_HOOK = hook
        knobs.runtime.jit_cache_hook = hook


@contextmanager
def observe_compilation_requests(observer):
    """Observe compiler requests in this context without changing cache identity."""
    _ensure_triton_cache_hook()
    token = _COMPILATION_OBSERVERS.set((*_COMPILATION_OBSERVERS.get(), observer))
    try:
        yield
    finally:
        _COMPILATION_OBSERVERS.reset(token)


def report_compilation_request(target, spec):
    for observer in _COMPILATION_OBSERVERS.get():
        observer(target, spec)


def freeze_kernel_resolution(reason: str | None = None) -> None:
    _ensure_triton_cache_hook()
    global _FROZEN, _FREEZE_REASON
    with _STATE_LOCK:
        _FROZEN = True
        _FREEZE_REASON = reason


def unfreeze_kernel_resolution() -> None:
    global _FROZEN, _FREEZE_REASON
    with _STATE_LOCK:
        _FROZEN = False
        _FREEZE_REASON = None


def kernel_resolution_frozen() -> bool:
    with _STATE_LOCK:
        return _FROZEN


freeze_compilation = freeze_kernel_resolution
unfreeze_compilation = unfreeze_kernel_resolution
compilation_frozen = kernel_resolution_frozen


def raise_if_kernel_resolution_frozen(
    kind: str,
    *,
    target: object | None = None,
    cache_key: object | None = None,
) -> None:
    with _STATE_LOCK:
        frozen = _FROZEN
        reason = _FREEZE_REASON
    if not frozen:
        return

    details = [f"b12x kernel resolution is frozen; refusing {kind}"]
    target_name = _describe_target(target)
    if target_name is not None:
        details.append(f"target={target_name}")
    if cache_key is not None:
        details.append(f"key={_summarize_cache_key(cache_key)}")
    if reason is not None:
        details.append(f"reason={reason}")
    details.append(
        "warm up this kernel shape before calling b12x.freeze_kernel_resolution()"
    )
    raise KernelResolutionFrozenError("; ".join(details))


def _describe_target(target: object | None) -> str | None:
    if target is None:
        return None
    if inspect.ismethod(target):
        module = getattr(target.__func__, "__module__", "")
        qualname = getattr(
            target.__func__, "__qualname__", getattr(target.__func__, "__name__", "")
        )
        return f"{module}.{qualname}" if module else qualname
    if inspect.isfunction(target):
        module = getattr(target, "__module__", "")
        qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
        return f"{module}.{qualname}" if module else qualname
    target_type = type(target)
    module = getattr(target_type, "__module__", "")
    qualname = getattr(target_type, "__qualname__", target_type.__name__)
    return f"{module}.{qualname}" if module else qualname


def _summarize_cache_key(cache_key: object) -> str:
    text = repr(cache_key)
    if len(text) > 120:
        text = text[:117] + "..."
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()[:12]
    return f"{text} [{digest}]"
