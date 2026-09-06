"""CUDA replay sampling and lossless grouped timing evidence."""

from __future__ import annotations

import math
import statistics
from contextlib import ExitStack


def _group_medians(samples, *, groups: int, repetitions: int):
    if any(type(value) is not int or value <= 0 for value in (groups, repetitions)):
        raise ValueError("timing groups and repetitions must be positive integers")
    expected = groups * repetitions
    if len(samples) != expected:
        raise ValueError(f"expected {expected} timing samples, received {len(samples)}")
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    return tuple(float(statistics.median(samples[start : start + repetitions]))
                 for start in range(0, expected, repetitions))


def median_of_group_medians(samples, *, groups: int, repetitions: int) -> float:
    return float(statistics.median(_group_medians(samples, groups=groups, repetitions=repetitions)))


def grouped_timing_evidence(samples, *, groups: int, repetitions: int) -> dict[str, object]:
    """Retain every unrounded duration and the partition used to reduce it."""
    medians = _group_medians(samples, groups=groups, repetitions=repetitions)
    return {"schema_version": 1, "unit": "us", "aggregation": "median_of_group_medians",
            "groups": groups, "repetitions": repetitions, "samples_us": list(samples),
            "group_medians_us": list(medians)}


def bounded_repetitions(settings, *, pilot_us: float) -> int:
    budget_us = float(settings.max_candidate_seconds) * 1_000_000.0
    budgeted = int(budget_us / (max(float(pilot_us), 1.0) * settings.groups))
    return max(1, min(settings.repetitions, budgeted))


def cuda_event_samples_us(run, *, count: int, device: object, flush=None,
                          before_each=None) -> tuple[float, ...]:
    """Bracket each replay separately, keeping resets and L2 flushes outside it."""
    import torch

    if type(count) is not int or count <= 0:
        raise ValueError("timing requires a positive integer sample count")
    starts = tuple(torch.cuda.Event(enable_timing=True) for _ in range(count))
    ends = tuple(torch.cuda.Event(enable_timing=True) for _ in range(count))
    for start, end in zip(starts, ends, strict=True):
        if before_each is not None:
            before_each()
        if flush is not None:
            flush()
        start.record()
        run()
        end.record()
    torch.cuda.synchronize(device)
    return tuple(float(start.elapsed_time(end)) * 1_000.0
                 for start, end in zip(starts, ends, strict=True))


def balanced_race_samples_us(runs, *, count: int, device: object, flush=None, before_each=None):
    """Interleave independent cold replays with balanced candidate positions.

    Every timing interval contains exactly one production replay; reset and
    flush work is excluded. Per-candidate counts support sample aggregation.
    """
    import torch

    from collections.abc import Mapping

    if type(count) is not int and not isinstance(count, Mapping):
        raise ValueError("a balanced race requires a positive count or per-candidate count mapping")
    counts = {name: count for name in runs} if type(count) is int else dict(count)
    if not runs or set(counts) != set(runs) or any(type(value) is not int or value <= 0 for value in counts.values()):
        raise ValueError("a balanced race requires candidates and a positive sample count")
    names = tuple(runs)
    with torch.cuda.device(device):
        events = {name: tuple((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                              for _ in range(counts[name])) for name in names}
        for index in range(max(counts.values())):
            offset = index % len(names)
            order = names[offset:] + names[:offset]
            if (index // len(names)) % 2:
                order = order[::-1]
            for name in order:
                if index >= counts[name]:
                    continue
                if before_each is not None:
                    before_each(name)
                if flush is not None:
                    flush()
                start, end = events[name][index]
                start.record()
                runs[name]()
                end.record()
        torch.cuda.synchronize(device)
        return {name: tuple(float(start.elapsed_time(end)) * 1000. for start, end in events[name])
                for name in names}


class CapturedGraphRaceTimer:
    """Schedule unchanged production graphs between device-side timing events.

    The production graph must retain its raw graph with ``keep_graph=True``.
    Child nodes clone that graph; they do not recapture the production callable.
    Each sample runs its reset and cold-L2 flush before the start event.
    """

    def __init__(self, graphs, *, count: int, device: object, flush=None, before_each=None):
        import torch
        from cuda.bindings import runtime

        if not graphs or type(count) is not int or count <= 0:
            raise ValueError("timing requires graphs and a positive integer sample count")
        self._runtime, self._device, self._production = runtime, device, dict(graphs)
        self._owners = ExitStack()
        self._executable = None
        try:
            with torch.cuda.device(device):
                names = tuple(graphs)
                self._events = {name: tuple((self._event(), self._event()) for _ in range(count)) for name in names}
                preparations = {}
                if before_each is not None or flush is not None:
                    for name in names:
                        preparation = torch.cuda.CUDAGraph(keep_graph=True)
                        self._owners.callback(preparation.reset)
                        with torch.cuda.graph(preparation):
                            if before_each is not None:
                                before_each(name)
                            if flush is not None:
                                flush()
                        preparations[name] = preparation
                parent = self._checked(runtime.cudaGraphCreate(0))
                self._owners.callback(self._checked_call, runtime.cudaGraphDestroy, parent)
                previous = []
                for index in range(count):
                    offset = index % len(names)
                    order = names[offset:] + names[:offset]
                    if (index // len(names)) % 2:
                        order = order[::-1]
                    for name in order:
                        start, end = self._events[name][index]
                        if name in preparations:
                            previous = [self._checked(runtime.cudaGraphAddChildGraphNode(
                                parent, previous, len(previous),
                                runtime.cudaGraph_t(preparations[name].raw_cuda_graph())))]
                        previous = [self._checked(runtime.cudaGraphAddEventRecordNode(
                            parent, previous, len(previous), start))]
                        previous = [self._checked(runtime.cudaGraphAddChildGraphNode(
                            parent, previous, 1, runtime.cudaGraph_t(graphs[name].raw_cuda_graph())))]
                        previous = [self._checked(runtime.cudaGraphAddEventRecordNode(
                            parent, previous, 1, end))]
                self._executable = self._checked(runtime.cudaGraphInstantiate(parent, 0))
                self._owners.callback(self._checked_call, runtime.cudaGraphExecDestroy, self._executable)
        except BaseException:
            self._owners.close()
            raise

    def _checked(self, result):
        error, *values = result
        if error != self._runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA graph timer failed: {error}")
        return None if not values else values[0]

    def _checked_call(self, function, *args):
        return self._checked(function(*args))

    def _event(self):
        event = self._checked(self._runtime.cudaEventCreate())
        self._owners.callback(self._checked_call, self._runtime.cudaEventDestroy, event)
        return event

    def replay(self):
        import torch

        if self._executable is None:
            raise RuntimeError("CUDA graph timer is closed")
        stream = self._runtime.cudaStream_t(torch.cuda.current_stream(self._device).cuda_stream)
        self._checked(self._runtime.cudaGraphLaunch(self._executable, stream))

    def samples_us(self):
        import torch

        if self._executable is None:
            raise RuntimeError("CUDA graph timer is closed")
        torch.cuda.synchronize(self._device)
        return {name: tuple(float(self._checked(self._runtime.cudaEventElapsedTime(start, end))) * 1_000.
                            for start, end in events) for name, events in self._events.items()}

    def close(self):
        import torch

        if self._executable is not None:
            torch.cuda.synchronize(self._device)
            self._executable = None
        self._owners.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class CapturedGraphTimer(CapturedGraphRaceTimer):
    """Bracket repeated execution of one unchanged production graph."""

    def __init__(self, graph, *, count, device, flush=None, before_each=None):
        super().__init__({"production": graph}, count=count, device=device, flush=flush,
                         before_each=None if before_each is None else lambda name: before_each())

    def samples_us(self):
        return super().samples_us()["production"]
