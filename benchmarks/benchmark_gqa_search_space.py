"""Count serialized candidates and distinct execution configs over all GQA cases."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from b12x.attention.paged._policy import GqaConfig
from b12x.policy.device import detect_device
from b12x.policy.generation.attention_corpus import gqa_cases
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.providers.gpu_workers import _GqaSession, _gqa_execution_key
from b12x.policy.generation.sweep import SweepCandidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = detect_device(f"cuda:{args.device}")
    assert device.identity is not None
    context = GenerationContext(
        device=device.identity, device_ordinal=args.device, work_dir=args.output.parent,
        source_revision="enumeration-only", settings=GenerationSettings(),
    )
    grouped = defaultdict(list)
    cases = gqa_cases()
    for case in cases:
        grouped[case.group_id].append(case)
    serialized = distinct = 0
    histogram = Counter()
    started = time.perf_counter()
    for group_index, group in enumerate(grouped.values(), start=1):
        session = _GqaSession(tuple(group), context)
        capacity = session._capacity
        all_candidates = {}

        def observe(*args, **kwargs):
            result = capacity(*args, **kwargs)
            candidate = SweepCandidate.create(GqaConfig.from_capacity(result).profile_dict())
            all_candidates.setdefault(candidate.candidate_id, candidate)
            return result

        session._capacity = observe
        for case in group:
            all_candidates.clear()
            candidates = session.candidates(case)
            representatives = {_gqa_execution_key(case, item) for item in candidates}
            assert representatives == {_gqa_execution_key(case, item)
                                       for item in all_candidates.values()}
            assert all(item.candidate_id in all_candidates for item in candidates)
            serialized += len(all_candidates)
            distinct += len(candidates)
            histogram[f"{len(all_candidates)}->{len(candidates)}"] += 1
        if group_index % 30 == 0:
            print(json.dumps({"groups": group_index, "serialized": serialized,
                              "distinct": distinct}), flush=True)
    report = {
        "device": context.checkpoint_metadata()["device"],
        "case_ids_sha256": hashlib.sha256("\n".join(c.case_id for c in cases).encode()).hexdigest(),
        "cases": len(cases), "groups": len(grouped),
        "serialized_candidates": serialized, "execution_candidates": distinct,
        "histogram": dict(sorted(histogram.items())),
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
