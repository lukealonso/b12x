"""Replay query-selection policies against existing complete MoE race tables.

This is an offline counterfactual study, not GPU qualification. Source revisions,
measurement settings, candidate contracts, and eligibility sets remain separate.
Every query observation contains all four route scenarios and all candidates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from b12x.policy.generation.moe_corpus import COMMON_ROUTE_PATTERNS


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_tables(root):
    grouped = defaultdict(dict)
    digest = hashlib.sha256()
    revisions = defaultdict(int)
    files = sorted(root.glob("full-*.json"))
    for path in files:
        raw = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + hashlib.sha256(raw).digest())
        record = json.loads(raw)
        query = record["query"]
        generation = record["generation"]
        revisions[generation["source_revision"]] += 1
        key = canonical((generation, record["candidate_contract_version"],
                         {k: v for k, v in query.items() if k != "routed_rows"}))
        grouped[key][record["route_pattern"]] = record
    series = defaultdict(list)
    for key, routes in grouped.items():
        if set(routes) != set(COMMON_ROUTE_PATTERNS):
            continue
        by_route = []
        for record in routes.values():
            by_route.append({
                item["candidate_id"]: item["latency_us"]
                for item in record["measurements"]
                if item["error"] is None and item["latency_us"] is not None
                and item["cosine"] is not None
                and item["cosine"] >= record["generation"]["settings"]["minimum_cosine"]
            })
        candidates = sorted(set.intersection(*(set(item) for item in by_route)))
        if len(candidates) < 2:
            continue
        generation, contract, query = json.loads(key)
        tokens = query.pop("num_tokens")
        series_key = canonical((generation, contract, query, candidates))
        scores = [sum(math.log(item[c]) for item in by_route) / len(by_route)
                  for c in candidates]
        series[series_key].append((tokens, scores))
    tables = []
    for key, rows in sorted(series.items()):
        if len(rows) < 8:
            continue
        rows.sort()
        tables.append((key, np.array([row[0] for row in rows]),
                       np.array([row[1] for row in rows])))
    return tables, {"files": len(files), "corpus_sha256": digest.hexdigest(),
                    "source_revisions": dict(revisions), "series": len(tables)}


def posterior(x, y, selected):
    distance = np.abs(x[:, None] - x[None, :])
    scaled = np.sqrt(3) * distance
    kernel = 0.5**2 * (1 + scaled) * np.exp(-scaled)
    cross = kernel[:, selected]
    train = kernel[np.ix_(selected, selected)] + np.eye(len(selected)) * 0.02**2
    mean = cross @ np.linalg.solve(train, y[selected])
    variance = np.maximum(0, np.diag(kernel) - np.sum(
        cross * np.linalg.solve(train, cross.T).T, axis=1,
    ))
    return mean, np.sqrt(variance)


def predict(tokens, scores, budget, method):
    x = np.log2(tokens)
    selected = [0, len(x) - 1]
    # Relative log latency removes the common growth with capacity.
    y = scores - scores[:, :1]
    while len(selected) < budget:
        remaining = np.array([i for i in range(len(x)) if i not in selected])
        if method == "bayesian":
            mean, sigma = posterior(x, y, selected)
            uncertainty = np.repeat(sigma[:, None], y.shape[1], axis=1)
            uncertainty[:, 0] = 0
            winner = mean.argmin(axis=1)
            best_mean = mean[np.arange(len(x)), winner]
            best_sigma = uncertainty[np.arange(len(x)), winner]
            acquisition = np.max(
                best_mean[:, None] - mean
                + 2 * (best_sigma[:, None] + uncertainty), axis=1,
            )
        else:
            acquisition = np.min(np.abs(x[:, None] - x[selected]), axis=1)
            if method == "boundary":
                ordered = sorted(selected)
                winners = scores[ordered].argmin(axis=1)
                for left, right, wl, wr in zip(
                    ordered, ordered[1:], winners, winners[1:],
                ):
                    if wl != wr:
                        acquisition[left + 1:right] *= 4
        selected.append(int(remaining[np.argmax(acquisition[remaining])]))
    if method == "bayesian":
        mean, _ = posterior(x, y, selected)
        choices = mean.argmin(axis=1)
    else:
        nearest = np.argmin(np.abs(x[:, None] - x[selected]), axis=1)
        choices = scores[selected].argmin(axis=1)[nearest]
    # Measured queries always retain their measured decision.
    choices[selected] = scores[selected].argmin(axis=1)
    regrets = np.exp(scores[np.arange(len(x)), choices] - scores.min(axis=1)) - 1
    held_out = np.ones(len(x), dtype=bool)
    held_out[selected] = False
    return regrets, held_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    tables, provenance = load_tables(args.checkpoints)
    report = {"status": "research-only", "checkpoints": str(args.checkpoints),
              "provenance": provenance, "load_seconds": time.perf_counter() - started,
              "experiments": []}
    print(canonical({**provenance, "load_seconds": report["load_seconds"]}), flush=True)
    for fraction in (0.125, 0.25, 0.5):
        for method in ("space_filling", "boundary", "bayesian"):
            started = time.perf_counter()
            regrets, held_out_regrets = [], []
            total, measured = 0, 0
            for _, tokens, scores in tables:
                budget = min(len(tokens), max(2, math.ceil(len(tokens) * fraction)))
                values, held_out = predict(tokens, scores, budget, method)
                regrets.extend(values.tolist())
                held_out_regrets.extend(values[held_out].tolist())
                total += len(tokens)
                measured += budget
            row = {"method": method, "budget_fraction": fraction,
                   "queries": total, "measured_queries": measured,
                   "seconds": time.perf_counter() - started,
                   "geomean_regret": float(np.expm1(np.log1p(regrets).mean())),
                   "p99_regret": float(np.quantile(regrets, 0.99)),
                   "max_regret": max(regrets),
                   "fraction_above_5pct": float((np.array(regrets) > 0.05).mean()),
                   "held_out_p99_regret": float(np.quantile(held_out_regrets, 0.99))}
            report["experiments"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(canonical(row), flush=True)


if __name__ == "__main__":
    main()
