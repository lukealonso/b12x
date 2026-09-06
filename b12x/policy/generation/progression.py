"""Audit ordered kernel choices along observed coordinate lines."""

from __future__ import annotations

from collections import defaultdict


def audit_progressions(measurements, decisions, *, axes, ordered_knobs):
    """Report reversals without extending monotonicity beyond observed samples."""
    reports = []
    for dimension, axis in enumerate(axes):
        lines = defaultdict(list)
        for item in measurements:
            coordinates = item.point.coordinates
            key = (item.point.family, coordinates[:dimension], coordinates[dimension + 1:])
            lines[key].append(item)
        for knob in ordered_knobs:
            segments = reversals = constant = increasing = decreasing = 0
            examples = []
            for line in lines.values():
                ordered = sorted(line, key=lambda item: item.point.coordinates[dimension])
                runs = [[]]
                for item in ordered:
                    value = decisions[item.winner].get(knob)
                    if type(value) is not int:
                        if runs[-1]:
                            runs.append([])
                    else:
                        runs[-1].append((item.point.coordinates[dimension], value, item.point.key))
                for run in runs:
                    if len(run) < 2:
                        continue
                    segments += 1
                    signs = {(right[1] > left[1]) - (right[1] < left[1])
                             for left, right in zip(run, run[1:], strict=False)} - {0}
                    if not signs:
                        constant += 1
                    elif signs == {1}:
                        increasing += 1
                    elif signs == {-1}:
                        decreasing += 1
                    else:
                        reversals += 1
                        if len(examples) < 16:
                            examples.append({"coordinates": [item[0] for item in run],
                                             "values": [item[1] for item in run],
                                             "queries": [item[2] for item in run]})
            reports.append({"axis": axis, "knob": knob, "observed_segments": segments,
                            "constant": constant, "nondecreasing": increasing, "nonincreasing": decreasing,
                            "nonmonotone": reversals, "reversal_examples": examples,
                            "guarantee": "observed_samples_only"})
    return tuple(reports)


__all__ = ["audit_progressions"]
