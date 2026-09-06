"""Generic measurement partitioning for multi-GPU profile generation."""

from __future__ import annotations

from .contracts import (
    ComponentGenerator,
    GenerationContext,
    MeasurementPartition,
    PartitionableComponentGenerator,
)

_FULL_COMPONENT_PARTITION = "full-component"


def measurement_partitions(
    generators: tuple[ComponentGenerator, ...],
    context: GenerationContext,
) -> tuple[MeasurementPartition, ...]:
    """Return deterministic, checkpoint-disjoint work for ``generators``."""
    partitions: list[MeasurementPartition] = []
    for generator in generators:
        if isinstance(generator, PartitionableComponentGenerator):
            component_partitions = generator.measurement_partitions(context)
        else:
            estimate = generator.estimate(context)
            component_partitions = (
                MeasurementPartition(
                    component_id=generator.component_id,
                    partition_id=_FULL_COMPONENT_PARTITION,
                    work_units=max(estimate.work_units, 1),
                    case_count=max(estimate.case_count, 1),
                    description="complete component qualification",
                ),
            )
        if not component_partitions:
            raise ValueError(
                f"generator {generator.component_id!r} has no measurement partitions"
            )
        if any(
            partition.component_id != generator.component_id
            for partition in component_partitions
        ):
            raise ValueError(
                f"generator {generator.component_id!r} returned a foreign partition"
            )
        partition_ids = [item.partition_id for item in component_partitions]
        if len(partition_ids) != len(set(partition_ids)):
            raise ValueError(
                f"generator {generator.component_id!r} returned duplicate partitions"
            )
        partitions.extend(component_partitions)
    return tuple(
        sorted(
            partitions,
            key=lambda item: (item.component_id, item.partition_id),
        )
    )


def select_measurement_partitions(
    generator: ComponentGenerator,
    partition_ids: tuple[str, ...],
) -> ComponentGenerator:
    """Restrict ``generator`` to the requested independent measurements."""
    if not partition_ids or len(partition_ids) != len(set(partition_ids)):
        raise ValueError("measurement partition selection must be non-empty and unique")
    if isinstance(generator, PartitionableComponentGenerator):
        return generator.select_measurement_partitions(partition_ids)
    if partition_ids != (_FULL_COMPONENT_PARTITION,):
        raise ValueError(
            f"generator {generator.component_id!r} is only selectable as a component"
        )
    return generator


__all__ = ["measurement_partitions", "select_measurement_partitions"]
