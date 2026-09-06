"""Registry for offline component profile generators."""

from __future__ import annotations

from .contracts import ComponentGenerator


class ComponentGeneratorRegistry:
    """Validated collection of component-owned offline generators."""

    def __init__(self) -> None:
        self._generators: dict[str, ComponentGenerator] = {}

    def register(self, generator: ComponentGenerator) -> None:
        if not isinstance(generator, ComponentGenerator):
            raise TypeError("generator must implement ComponentGenerator")
        component_id = generator.component_id
        if component_id in self._generators:
            raise ValueError(f"duplicate component generator {component_id!r}")
        if generator.query_schema_version <= 0:
            raise ValueError("query_schema_version must be positive")
        if generator.config_schema_version <= 0:
            raise ValueError("config_schema_version must be positive")
        self._generators[component_id] = generator

    def get(self, component_id: str) -> ComponentGenerator:
        try:
            return self._generators[component_id]
        except KeyError as exc:
            choices = ", ".join(self.component_ids()) or "none"
            raise KeyError(
                f"no generator for {component_id!r}; registered: {choices}"
            ) from exc

    def component_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._generators))

    def select(
        self, component_ids: tuple[str, ...] | None
    ) -> tuple[ComponentGenerator, ...]:
        selected = self.component_ids() if component_ids is None else component_ids
        if len(selected) != len(set(selected)):
            raise ValueError("component selection contains duplicates")
        return tuple(self.get(component_id) for component_id in selected)


__all__ = ["ComponentGeneratorRegistry"]
