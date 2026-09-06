"""Prepare production graphs before timing a balanced candidate race."""

from collections.abc import Callable
from dataclasses import dataclass, field
import statistics

from .sweep import SweepMeasurement
from .timing import balanced_race_samples_us, bounded_repetitions, grouped_timing_evidence, median_of_group_medians


def capture_warmed_graph(run, *, device):
    """Capture a warmed callable with kernel resolution frozen."""
    import torch
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x._lib.runtime_control import kernel_resolution_frozen

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    already_frozen = kernel_resolution_frozen()
    if not already_frozen:
        freeze_kernel_resolution("profile capture must reuse prepared production kernels")
    try:
        with torch.cuda.device(device), torch.cuda.graph(graph):
            output = run()
    except BaseException:
        graph.reset()
        raise
    finally:
        if not already_frozen:
            unfreeze_kernel_resolution()
    return graph, output


@dataclass(kw_only=True)
class PreparedCandidate:
    """A correctness-checked graph with explicit storage ownership and reset."""

    candidate: object
    graph: object
    correct: bool
    metrics: dict = field(default_factory=dict)
    before_each: Callable | None = None
    owners: tuple = ()
    aggregation: str = "median_of_group_medians"
    owns_graph: bool = True
    sample_width: Callable[[float], int] | None = None
    pilot_replays: int = 1

    def __post_init__(self):
        if self.aggregation not in ("median", "median_of_group_medians"):
            raise ValueError("unsupported prepared-race estimator")
        if type(self.pilot_replays) is not int or self.pilot_replays <= 0:
            raise ValueError("prepared race pilot replays must be a positive integer")


def measure_prepared_candidates(prepared, *, settings, device, flush=None):
    """Time only prepared graphs, preserving failures and candidate order.

    Failed preparations may be supplied as SweepMeasurement objects. Every
    successful graph remains alive until all candidates finish timing. State
    resets and L2 flushes precede each event interval.
    """
    import torch

    prepared = tuple(prepared)
    active = {item.candidate.candidate_id: item for item in prepared if isinstance(item, PreparedCandidate)}
    if len({item.candidate.candidate_id for item in prepared}) != len(prepared):
        raise ValueError("prepared race contains duplicate candidates")
    if not active:
        return prepared

    def reset(name):
        callback = active[name].before_each
        if callback is not None:
            callback()

    runs = {name: item.graph.replay for name, item in active.items()}
    def sample(count):
        if settings.timing_clock == "globaltimer":
            from .timestamps import globaltimer_race_samples_us
            return globaltimer_race_samples_us({name: item.graph for name, item in active.items()},
                count=count, device=device, flush=flush, before_each=reset)
        return balanced_race_samples_us(runs, count=count, device=device, flush=flush, before_each=reset)

    try:
        with torch.cuda.device(device):
            for _ in range(settings.warmup):
                for name, run in runs.items():
                    reset(name)
                    run()
            torch.cuda.synchronize(device)
            pilot = sample(max(item.pilot_replays for item in active.values()))
            widths = {name: 1 if item.sample_width is None else item.sample_width(min(pilot[name]))
                      for name, item in active.items()}
            if any(type(width) is not int or width <= 0 for width in widths.values()):
                raise ValueError("prepared race sample widths must be positive integers")
            repetitions = min(bounded_repetitions(settings, pilot_us=min(pilot[name]) * widths[name]) for name in active)
            allocated = torch.cuda.memory_allocated(device)
            counts = {name: settings.groups * repetitions * widths[name] for name in active}
            raw_samples = sample(counts)
            allocation = torch.cuda.memory_allocated(device) - allocated
            samples = {name: tuple(statistics.mean(values[start:start + widths[name]])
                                  for start in range(0, len(values), widths[name])) for name, values in raw_samples.items()}
        return tuple(
            SweepMeasurement(
                candidate=item.candidate,
                latency_us=(float(statistics.median(samples[item.candidate.candidate_id]))
                            if item.aggregation == "median" else
                            median_of_group_medians(samples[item.candidate.candidate_id],
                                                    groups=settings.groups, repetitions=repetitions)),
                correct=item.correct and allocation == 0,
                metrics={**item.metrics,
                         "timing": {**grouped_timing_evidence(samples[item.candidate.candidate_id],
                                      groups=settings.groups, repetitions=repetitions),
                                    "protocol": "balanced_candidate_replay_v1",
                                    "clock": settings.timing_clock,
                                    "aggregation": item.aggregation,
                                    "replays_per_sample": widths[item.candidate.candidate_id],
                                    **({"raw_samples_us": list(raw_samples[item.candidate.candidate_id])}
                                       if widths[item.candidate.candidate_id] > 1 else {}),
                                    "candidate_order": list(active)},
                         "replay_allocation_bytes": allocation},
            ) if isinstance(item, PreparedCandidate) else item
            for item in prepared
        )
    finally:
        torch.cuda.synchronize(device)
        for item in active.values():
            if item.owns_graph:
                item.graph.reset()
