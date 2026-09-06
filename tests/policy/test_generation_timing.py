"""Timing evidence preserves the measured estimator and replay boundaries."""

import json
import statistics
import sys
from types import SimpleNamespace

import pytest

from b12x.policy.generation.timing import (
    cuda_event_samples_us,
    grouped_timing_evidence,
    median_of_group_medians,
)


def test_unrounded_samples_reproduce_grouped_estimator():
    samples = (1.123456789, 2., 100., 3., 4., 5., 6., 100., 200.)
    record = json.loads(json.dumps(grouped_timing_evidence(samples, groups=3, repetitions=3)))
    assert record["samples_us"] == list(samples)
    assert record["group_medians_us"] == [2., 4., 100.]
    assert median_of_group_medians(record["samples_us"], groups=record["groups"],
                                   repetitions=record["repetitions"]) == 4.
    assert statistics.median(samples) != 4.


@pytest.mark.parametrize("samples,groups,repetitions", [
    ((1.,), 1, 2), ((1.,), True, 1), ((1.,), 1, 0),
    ((float("nan"),), 1, 1), ((float("inf"),), 1, 1), ((0.,), 1, 1),
])
def test_invalid_timing_evidence_is_rejected(samples, groups, repetitions):
    with pytest.raises(ValueError):
        grouped_timing_evidence(samples, groups=groups, repetitions=repetitions)


def test_reset_and_cold_l2_flush_remain_outside_each_timed_replay(monkeypatch):
    trace = []
    events = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.index = len(events)
            events.append(self)

        def record(self):
            trace.append(("event", self.index))

        def elapsed_time(self, other):
            assert trace[-1] == ("synchronize", "assigned-device")
            assert other.index == self.index + 2
            return .00123456789

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        Event=Event, synchronize=lambda device: trace.append(("synchronize", device)))))
    samples = cuda_event_samples_us(lambda: trace.append("run"), count=2, device="assigned-device",
                                    before_each=lambda: trace.append("restore"),
                                    flush=lambda: trace.append("flush"))
    assert trace == ["restore", "flush", ("event", 0), "run", ("event", 2),
                     "restore", "flush", ("event", 1), "run", ("event", 3),
                     ("synchronize", "assigned-device")]
    assert samples == (1.23456789, 1.23456789)


def test_balanced_races_keep_preparation_outside_events_and_balance_positions(monkeypatch):
    from contextlib import nullcontext
    from collections import Counter
    import torch
    from b12x.policy.generation.timing import balanced_race_samples_us

    log = []
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            log.append("event")
        def elapsed_time(self, end):
            return .001

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    samples = balanced_race_samples_us({name: lambda name=name: log.append(name) for name in ("a", "b", "c")},
        count=6, device="cuda:5", flush=lambda: log.append("flush"), before_each=lambda name: log.append("reset"))
    assert samples == {name: (1.,) * 6 for name in ("a", "b", "c")}
    names = []
    for offset in range(0, len(log), 5):
        reset, flush, start, name, end = log[offset:offset + 5]
        assert (reset, flush, start, end) == ("reset", "flush", "event", "event")
        names.append(name)
    for position in range(3):
        assert Counter(names[position::3]) == {"a": 2, "b": 2, "c": 2}
