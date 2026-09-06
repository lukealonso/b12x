"""Built-in component generators used by the top-level profile tool."""

from __future__ import annotations

from b12x.policy.catalog import list_profiled_components
from b12x.policy.generation.registry import ComponentGeneratorRegistry


def register_builtin_generators(registry: ComponentGeneratorRegistry) -> None:
    for registration in list_profiled_components():
        # Profile-generation overlay (TP3 work): a provider whose probe cannot
        # initialise in this image (the QSA benchmark-case kinds drifted) must
        # not block generating the other components.
        try:
            registry.register(registration.create_generator())
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "skipping profile generator %s: %s", registration.component_id, exc
            )


__all__ = ["register_builtin_generators"]
