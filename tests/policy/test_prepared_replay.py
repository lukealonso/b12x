from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from b12x.policy.generation import replay
from b12x.policy.generation.contracts import GenerationSettings
from b12x.policy.generation.sweep import SweepCandidate, SweepMeasurement


@pytest.mark.parametrize("allocation", (0, 16))
def test_prepared_race_preserves_failures_resets_and_storage_until_all_samples(monkeypatch, allocation):
    trace = []
    allocated = iter((100, 100 + allocation))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: next(allocated))
    prepared = []
    for name in ("a", "b"):
        graph = SimpleNamespace(replay=lambda name=name: trace.append(("run", name)),
                                reset=lambda name=name: trace.append(("close", name)))
        prepared.append(replay.PreparedCandidate(
            candidate=SweepCandidate.create({"backend": name}), graph=graph, correct=True,
            before_each=lambda name=name: trace.append(("reset", name)), owners=(object(),)))
    failure = SweepMeasurement(candidate=SweepCandidate.create({"backend": "failed"}),
                               correct=False, latency_us=None, error="compilation failed")
    prepared.insert(1, failure)
    calls = []

    def sample(runs, *, count, device, flush, before_each):
        assert not any(item[0] == "close" for item in trace)
        calls.append(count)
        for name, run in runs.items():
            before_each(name)
            flush()
            run()
        return {name: (2.,) * (count if isinstance(count, int) else count[name]) for name in runs}

    monkeypatch.setattr(replay, "balanced_race_samples_us", sample)
    result = replay.measure_prepared_candidates(
        prepared, settings=GenerationSettings(groups=3, repetitions=5),
        device="cuda:5", flush=lambda: trace.append(("flush", None)))
    assert result[1] is failure
    assert calls == [1, {item.candidate.candidate_id: 15 for item in (prepared[0], prepared[2])}]
    assert trace[-2:] == [("close", "a"), ("close", "b")]
    for item in (result[0], result[2]):
        assert item.correct == (allocation == 0)
        assert item.latency_us == 2.
        assert item.metrics["replay_allocation_bytes"] == allocation
        assert item.metrics["timing"]["protocol"] == "balanced_candidate_replay_v1"
        assert len(item.metrics["timing"]["samples_us"]) == 15


def test_failed_race_releases_every_graph(monkeypatch):
    closed = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    def fail(*args, **kwargs):
        raise RuntimeError("event failure")
    monkeypatch.setattr(replay, "balanced_race_samples_us", fail)
    prepared = [replay.PreparedCandidate(
        candidate=SweepCandidate.create({"backend": name}), correct=True,
        graph=SimpleNamespace(replay=lambda: None, reset=lambda name=name: closed.append(name)))
        for name in ("a", "b")]
    with pytest.raises(RuntimeError, match="event failure"):
        replay.measure_prepared_candidates(prepared, settings=GenerationSettings(), device="cuda:5")
    assert closed == ["a", "b"]


def test_averaged_samples_preserve_raw_intervals_and_borrowed_graphs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 100)
    closed = []
    item = replay.PreparedCandidate(candidate=SweepCandidate.create({"backend": "borrowed"}),
        graph=SimpleNamespace(replay=lambda: None, reset=lambda: closed.append(True)),
        correct=True, sample_width=lambda pilot: 2, pilot_replays=3, owns_graph=False, aggregation="median")
    name = item.candidate.candidate_id
    calls = []
    def sample(runs, *, count, **kwargs):
        calls.append(count)
        return {name: (1., 3., 2.) if isinstance(count, int) else (1., 9., 2., 4., 8., 12.)}
    monkeypatch.setattr(replay, "balanced_race_samples_us", sample)
    result, = replay.measure_prepared_candidates((item,), device="cuda:5",
        settings=GenerationSettings(groups=3, repetitions=1))
    assert calls == [3, {name: 6}]
    assert not closed
    assert result.latency_us == 5.
    assert result.metrics["timing"]["samples_us"] == (5., 3., 10.)
    assert result.metrics["timing"]["raw_samples_us"] == (1., 9., 2., 4., 8., 12.)
