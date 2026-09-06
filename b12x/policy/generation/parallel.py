"""Spawn-isolated multi-GPU measurement orchestration."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from b12x.policy import DetectedDevice, DeviceIdentity, detect_device

from .contracts import (
    ComponentGenerator,
    GenerationContext,
    GenerationSettings,
    MeasurementPartition,
    WorkEstimate,
)
from .registry import ComponentGeneratorRegistry
from .sharding import measurement_partitions, select_measurement_partitions
from .store import CheckpointStore

RegistryFactory = Callable[[], ComponentGeneratorRegistry]

_WORKER_DEVICE: DetectedDevice | None = None
_WORKER_PROGRESS_QUEUE: Any = None
_WORKER_REGISTRY: ComponentGeneratorRegistry | None = None
_WORKER_STOP_EVENT: Any = None


@dataclass(frozen=True, kw_only=True)
class ParallelMeasurementSummary:
    device_ordinals: tuple[int, ...]
    partition_count: int
    worker_count: int


@dataclass(frozen=True, kw_only=True)
class _MeasurementTask:
    partition: MeasurementPartition
    expected_device: DeviceIdentity
    work_dir: Path
    source_revision: str
    settings: GenerationSettings


@dataclass(frozen=True, kw_only=True)
class _MeasurementResult:
    partition: MeasurementPartition
    device_ordinal: int


@dataclass(frozen=True, kw_only=True)
class _MeasurementProgress:
    partition: MeasurementPartition
    device_ordinal: int
    units: int
    detail: str


@dataclass(frozen=True, kw_only=True)
class _WorkerReady:
    device_ordinal: int


class _WorkerProgressReporter:
    def __init__(self, partition: MeasurementPartition) -> None:
        self._partition = partition

    def _send(self, *, units: int, detail: str) -> None:
        detected = _WORKER_DEVICE
        if detected is None or detected.ordinal is None:
            raise RuntimeError("profile measurement worker was not initialized")
        if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
            raise InterruptedError("parallel profile measurement was cancelled")
        if _WORKER_PROGRESS_QUEUE is None:
            return
        _WORKER_PROGRESS_QUEUE.put(
            _MeasurementProgress(
                partition=self._partition,
                device_ordinal=detected.ordinal,
                units=units,
                detail=detail,
            )
        )

    def start_component(self, estimate: WorkEstimate) -> None:
        self._send(units=0, detail=estimate.description)

    def start_stage(
        self,
        component_id: str,
        *,
        stage: str,
        total: int,
    ) -> None:
        del component_id, total
        self._send(units=0, detail=stage)

    def advance(
        self,
        component_id: str,
        *,
        units: int = 1,
        detail: str | None = None,
    ) -> None:
        del component_id
        self._send(units=units, detail=detail or "")

    def finish_component(self, component_id: str) -> None:
        del component_id


def _initialize_worker(
    device_queue: Any,
    progress_queue: Any,
    stop_event: Any,
    registry_factory: RegistryFactory,
) -> None:
    global _WORKER_DEVICE, _WORKER_PROGRESS_QUEUE, _WORKER_REGISTRY
    global _WORKER_STOP_EVENT

    device_spec = device_queue.get()
    detected = detect_device(device_spec)
    if detected.identity is None or detected.ordinal is None:
        raise RuntimeError(f"{device_spec!r} did not resolve to a CUDA GPU")
    import torch

    torch.cuda.set_device(detected.ordinal)
    _WORKER_DEVICE = detected
    _WORKER_PROGRESS_QUEUE = progress_queue
    _WORKER_REGISTRY = registry_factory()
    _WORKER_STOP_EVENT = stop_event
    progress_queue.put(_WorkerReady(device_ordinal=detected.ordinal))


def _run_task(task: _MeasurementTask) -> _MeasurementResult:
    detected = _WORKER_DEVICE
    registry = _WORKER_REGISTRY
    if detected is None or registry is None:
        raise RuntimeError("profile measurement worker was not initialized")
    if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
        raise InterruptedError("parallel profile measurement was cancelled")
    if detected.identity != task.expected_device:
        raise RuntimeError(
            f"cuda:{detected.ordinal} is {detected.identity}, expected "
            f"{task.expected_device}"
        )
    generator = select_measurement_partitions(
        registry.get(task.partition.component_id),
        (task.partition.partition_id,),
    )
    context = GenerationContext(
        device=task.expected_device,
        device_ordinal=detected.ordinal,
        work_dir=task.work_dir,
        source_revision=task.source_revision,
        settings=task.settings,
    )
    estimate = generator.estimate(context)
    if (
        estimate.component_id != task.partition.component_id
        or estimate.work_units != task.partition.work_units
        or estimate.case_count != task.partition.case_count
    ):
        raise RuntimeError(
            f"measurement partition {task.partition.partition_id!r} no longer "
            "matches its selected generator"
        )
    progress = _WorkerProgressReporter(task.partition)
    progress.start_component(estimate)
    result = generator.generate(
        context,
        progress=progress,
        checkpoints=CheckpointStore(task.work_dir / "checkpoints"),
    )
    if result.completed_work_units != task.partition.work_units:
        raise RuntimeError(
            f"measurement partition {task.partition.partition_id!r} completed "
            f"{result.completed_work_units} work units; expected "
            f"{task.partition.work_units}"
        )
    progress.finish_component(task.partition.component_id)
    return _MeasurementResult(
        partition=task.partition,
        device_ordinal=detected.ordinal,
    )


def run_parallel_measurements(
    *,
    console: Console,
    device_specs: tuple[str, ...],
    generators: tuple[ComponentGenerator, ...],
    context: GenerationContext,
    registry_factory: RegistryFactory,
) -> ParallelMeasurementSummary:
    """Measure generators on identical GPUs and leave reduction to the parent."""
    partitions = measurement_partitions(generators, context)
    worker_count = min(len(device_specs), len(partitions))
    ordered = tuple(
        sorted(
            partitions,
            key=lambda item: (
                -item.work_units,
                item.component_id,
                item.partition_id,
            ),
        )
    )
    process_context = multiprocessing.get_context("spawn")
    device_queue = process_context.Queue()
    progress_queue = process_context.Queue()
    stop_event = process_context.Event()
    for device_spec in device_specs[:worker_count]:
        device_queue.put(device_spec)
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
        initializer=_initialize_worker,
        initargs=(device_queue, progress_queue, stop_event, registry_factory),
    )
    futures = {}
    results = []
    try:
        for partition in ordered:
            future = executor.submit(
                _run_task,
                _MeasurementTask(
                    partition=partition,
                    expected_device=context.device,
                    work_dir=context.work_dir,
                    source_revision=context.source_revision,
                    settings=context.settings,
                ),
            )
            futures[future] = partition
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}"),
            console=console,
        ) as progress:
            progress_task = progress.add_task(
                "parallel GPU measurements",
                total=sum(item.work_units for item in partitions),
                detail=f"0/{worker_count} GPUs online · {len(partitions)} partitions",
            )
            reported = {partition: 0 for partition in partitions}
            completed: set[MeasurementPartition] = set()
            worker_tasks: dict[int, int] = {}
            worker_completed: dict[int, int] = {}

            def ensure_worker_task(device_ordinal: int) -> int:
                task = worker_tasks.get(device_ordinal)
                if task is not None:
                    return task
                task = progress.add_task(
                    f"cuda:{device_ordinal}",
                    total=None,
                    detail="ready",
                )
                worker_tasks[device_ordinal] = task
                worker_completed[device_ordinal] = 0
                return task

            def update_overall(*, advance: int = 0) -> None:
                progress.update(
                    progress_task,
                    advance=advance,
                    detail=(
                        f"{len(worker_tasks)}/{worker_count} GPUs online · "
                        f"{len(completed)}/{len(partitions)} partitions"
                    ),
                )

            def drain_progress() -> None:
                while True:
                    try:
                        event = progress_queue.get_nowait()
                    except Empty:
                        return
                    if isinstance(event, _WorkerReady):
                        ensure_worker_task(event.device_ordinal)
                        update_overall()
                        continue
                    if not isinstance(event, _MeasurementProgress):
                        raise TypeError("worker progress event has the wrong type")
                    partition = event.partition
                    if partition in completed:
                        continue
                    worker_task = ensure_worker_task(event.device_ordinal)
                    remaining = partition.work_units - reported[partition]
                    units = min(max(event.units, 0), remaining)
                    reported[partition] += units
                    progress.update(
                        worker_task,
                        detail=(
                            f"{worker_completed[event.device_ordinal]} done · "
                            f"{partition.component_id} · {event.detail}"
                        ),
                    )
                    update_overall(advance=units)

            pending = set(futures)
            while pending:
                done, pending = wait(
                    pending,
                    timeout=0.25,
                    return_when=FIRST_COMPLETED,
                )
                drain_progress()
                for future in done:
                    partition = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        console.print(
                            "[bold red]GPU measurement partition failed:[/bold red] "
                            f"{partition.component_id}/{partition.partition_id}"
                        )
                        console.print_exception(show_locals=False)
                        raise
                    if result.partition != partition:
                        raise RuntimeError(
                            "measurement worker returned the wrong partition"
                        )
                    results.append(result)
                    remaining = partition.work_units - reported[partition]
                    reported[partition] += remaining
                    completed.add(partition)
                    worker_task = ensure_worker_task(result.device_ordinal)
                    worker_completed[result.device_ordinal] += 1
                    progress.update(
                        worker_task,
                        detail=(
                            f"{worker_completed[result.device_ordinal]} done · "
                            f"completed {partition.component_id}"
                        ),
                    )
                    update_overall(advance=remaining)
            drain_progress()
            for device_ordinal, worker_task in worker_tasks.items():
                progress.update(
                    worker_task,
                    total=1,
                    completed=1,
                    detail=(
                        f"{worker_completed[device_ordinal]} partitions · complete"
                    ),
                )
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        device_queue.close()
        device_queue.join_thread()
        progress_queue.close()
        progress_queue.join_thread()
    return ParallelMeasurementSummary(
        device_ordinals=tuple(sorted({result.device_ordinal for result in results})),
        partition_count=len(results),
        worker_count=worker_count,
    )


__all__ = ["ParallelMeasurementSummary", "run_parallel_measurements"]
