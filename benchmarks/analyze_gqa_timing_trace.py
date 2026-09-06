"""Compare GQA timing instrumentation with correlated Nsight kernel intervals."""

import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def _balanced_names(names, count):
    for index in range(count):
        offset = index % len(names)
        order = names[offset:] + names[:offset]
        if (index // len(names)) % 2:
            order = order[::-1]
        yield from order


def _role(name):
    for pattern, role in (("update_regular_decode_graph_metadata", "metadata"),
                          ("PagedForwardKernel", "forward"), ("PagedPersistentMergeKernel", "merge")):
        if pattern in name:
            return role
    return None


def analyze(path, case):
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        launches = [dict(row) for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")
                    if strings[row["nameId"]].startswith("cudaGraphLaunch")]
        kernels = defaultdict(list)
        for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
            kernels[row["correlationId"]].append({**dict(row), "symbol": strings[row["demangledName"]]})
        ranges = [dict(row) for row in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start")]
    methods = [row for row in ranges if row["text"] and ":round" in row["text"]]
    expected_methods = sum(len(row["medians_us"]) for row in case["rounds"])
    if len(methods) != expected_methods:
        raise ValueError(f"trace contains {len(methods)} timing ranges, expected {expected_methods}")
    results = []
    for region in methods:
        _, index, method = region["text"].rsplit(":", 2)
        round_index = int(index.removeprefix("round"))
        measured = case["rounds"][round_index]
        sample_count = len(next(iter(measured["samples"][method].values()))["samples_us"])
        names = tuple(case["candidates"])
        selected = [item for item in launches if region["start"] <= item["start"] < region["end"]]
        samples = defaultdict(list)
        if method == "host_events":
            candidate_ranges = [r for r in ranges if r["text"] and r["text"].startswith("candidate:")
                                and region["start"] <= r["start"] < region["end"]]
            for launch in selected:
                owners = [r for r in candidate_ranges if r["start"] <= launch["start"] < r["end"]]
                if len(owners) != 1:
                    raise ValueError("host replay does not have one candidate NVTX owner")
                samples[owners[0]["text"].removeprefix("candidate:")].append(kernels[launch["correlationId"]])
        else:
            if len(selected) != 1:
                raise ValueError("device timing range must launch one parent graph")
            nodes = kernels[selected[0]["correlationId"]]
            replays, active = [], []
            flushes = 0
            for node in nodes:
                if "at::native::reduce_kernel" in node["symbol"] and "sum_functor" in node["symbol"]:
                    flushes += 1
                    if active:
                        replays.append(active)
                        active = []
                elif _role(node["symbol"]) is not None:
                    active.append(node)
                elif node["symbol"] != "_timestamp":
                    raise ValueError(f"unexpected timing-graph kernel: {node['symbol']}")
            if active:
                replays.append(active)
            expected_replays = sample_count * len(names)
            if flushes != expected_replays or len(replays) != expected_replays:
                raise ValueError("device timing graph has incomplete cold-replay coverage")
            for name, replay in zip(_balanced_names(names, sample_count), replays, strict=True):
                samples[name].append(replay)
        if set(samples) != set(names) or any(len(items) != sample_count for items in samples.values()):
            raise ValueError("timing trace does not match the declared candidate/sample counts")
        rows = {}
        for name, replays in samples.items():
            spans, sums, durations = [], [], defaultdict(list)
            for nodes in replays:
                nodes = [node for node in nodes if _role(node["symbol"]) is not None]
                expected_roles = ["forward"] if case["candidates"][name]["max_partial_rows"] == 0 else ["metadata", "forward", "merge"]
                if [_role(node["symbol"]) for node in nodes] != expected_roles:
                    raise ValueError("GQA replay kernels differ from the candidate split contract")
                spans.append((nodes[-1]["end"]-nodes[0]["start"])/1000.)
                sums.append(sum(node["end"]-node["start"] for node in nodes)/1000.)
                for node in nodes:
                    durations[_role(node["symbol"])].append((node["end"]-node["start"])/1000.)
            median_span = statistics.median(spans)
            rows[name] = {"samples": sample_count, "kernel_span_us": spans, "kernel_sum_us": sums,
                          "median_span_us": median_span, "median_sum_us": statistics.median(sums),
                          "median_kernels_us": {role: statistics.median(values) for role, values in durations.items()},
                          "instrument_median_us": measured["medians_us"][method][name],
                          "instrument_minus_span_us": measured["medians_us"][method][name]-median_span}
        results.append({"round": round_index, "method": method, "candidates": rows})
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"query": case["query"], "trace": str(path.resolve()), "trace_sha256": digest, "ranges": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timing = json.loads(args.timing.read_text())
    cases = [analyze(path, case) for path, case in zip(args.trace, timing["cases"], strict=True)]
    result = {"status": "research-only", "qualification": "CUPTI trace diagnostics; profiling and timestamp kernels perturb scheduling",
              "timing_sha256": hashlib.sha256(args.timing.read_bytes()).hexdigest(),
              "generation": timing["generation"], "cases": cases}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        for method in dict.fromkeys(row["method"] for row in case["ranges"]):
            rows = [item for row in case["ranges"] if row["method"] == method for item in row["candidates"].values()]
            overhead = [row["instrument_minus_span_us"] for row in rows]
            print(case["query"]["batch_size"], case["query"]["cache_tokens"], method,
                  "instrument-minus-kernel-span median range us", min(overhead), max(overhead))


if __name__ == "__main__":
    main()
