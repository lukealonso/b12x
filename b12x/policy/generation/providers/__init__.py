"""Built-in component generators used by the top-level profile tool."""

from __future__ import annotations

from b12x.policy.catalog import list_profiled_components
from b12x.policy.generation.registry import ComponentGeneratorRegistry


def register_builtin_generators(registry: ComponentGeneratorRegistry) -> None:
    for registration in list_profiled_components():
        registry.register(registration.create_generator())


__all__ = ["register_builtin_generators"]
