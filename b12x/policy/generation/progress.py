"""Rich and no-op progress reporters for profile generation."""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .contracts import WorkEstimate


class NullProgressReporter:
    def start_component(self, estimate: WorkEstimate) -> None:
        del estimate

    def start_stage(
        self,
        component_id: str,
        *,
        stage: str,
        total: int,
    ) -> None:
        del component_id, stage, total

    def advance(
        self,
        component_id: str,
        *,
        units: int = 1,
        detail: str | None = None,
    ) -> None:
        del component_id, units, detail

    def finish_component(self, component_id: str) -> None:
        del component_id


class RichProgressReporter:
    """Nested overall/component progress with ETA and current-case detail."""

    def __init__(
        self,
        estimates: tuple[WorkEstimate, ...],
        *,
        transient: bool = False,
    ) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}"),
            transient=transient,
        )
        self._overall = self._progress.add_task(
            "all components",
            total=sum(estimate.work_units for estimate in estimates),
            detail="",
        )
        self._tasks: dict[str, TaskID] = {}
        self._stage_tasks: dict[str, TaskID] = {}
        self._totals: dict[str, int] = {}
        self._completed: dict[str, int] = {}

    def __enter__(self) -> "RichProgressReporter":
        self._progress.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._progress.stop()

    def start_component(self, estimate: WorkEstimate) -> None:
        self._tasks[estimate.component_id] = self._progress.add_task(
            estimate.component_id,
            total=estimate.work_units,
            detail=estimate.description,
        )
        self._totals[estimate.component_id] = estimate.work_units
        self._completed[estimate.component_id] = 0

    def start_stage(
        self,
        component_id: str,
        *,
        stage: str,
        total: int,
    ) -> None:
        if total < 0:
            raise ValueError("stage progress totals cannot be negative")
        previous = self._stage_tasks.pop(component_id, None)
        if previous is not None:
            self._progress.remove_task(previous)
        self._stage_tasks[component_id] = self._progress.add_task(
            f"  ↳ {stage}",
            total=total,
            detail="",
        )

    def advance(
        self,
        component_id: str,
        *,
        units: int = 1,
        detail: str | None = None,
    ) -> None:
        if units < 0:
            raise ValueError("progress units cannot be negative")
        fields = {} if detail is None else {"detail": detail}
        self._progress.advance(self._tasks[component_id], units)
        self._progress.advance(self._overall, units)
        stage_task = self._stage_tasks.get(component_id)
        if stage_task is not None:
            self._progress.advance(stage_task, units)
        if fields:
            self._progress.update(self._tasks[component_id], **fields)
            if stage_task is not None:
                self._progress.update(stage_task, **fields)
        self._completed[component_id] += units

    def finish_component(self, component_id: str) -> None:
        task_id = self._tasks[component_id]
        remaining = max(
            0,
            self._totals[component_id] - self._completed[component_id],
        )
        if remaining:
            self._progress.advance(task_id, remaining)
            self._progress.advance(self._overall, remaining)
            self._completed[component_id] += remaining
        stage_task = self._stage_tasks.pop(component_id, None)
        if stage_task is not None:
            self._progress.remove_task(stage_task)
        self._progress.update(
            task_id,
            description=f"{component_id}: complete",
            detail="",
        )


__all__ = ["NullProgressReporter", "RichProgressReporter"]
