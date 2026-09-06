"""Compare two llm-decode-bench arms (decode matrix, coding peak, standalone prefill)."""
import json, sys, os
R = os.path.dirname(os.path.abspath(__file__))
a, b = sys.argv[1], sys.argv[2]
def load(name):
    p = f"{R}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None
def cells(d):
    out = {}
    for r in d.get("results", []):
        out[(r["concurrency"], r["context_tokens"])] = r
    return out
da, db = load(f"{a}-decode.json"), load(f"{b}-decode.json")
if da and db:
    ca, cb = cells(da), cells(db)
    print(f"{'cell':<12}{a+' tok/s':>16}{b+' tok/s':>16}{'delta':>9}{a+' itl':>12}{b+' itl':>12}")
    for k in sorted(set(ca) | set(cb)):
        ra, rb = ca.get(k), cb.get(k)
        ta = ra["client_output_tokens"] / 30 if ra else float("nan")
        tb = rb["client_output_tokens"] / 30 if rb else float("nan")
        ia = ra["chunk_inter_token_latency_p50"] * 1000 if ra else float("nan")
        ib = rb["chunk_inter_token_latency_p50"] * 1000 if rb else float("nan")
        d = (ta / tb - 1) * 100 if rb and tb else float("nan")
        print(f"c{k[0]}@{k[1]//1024}k".ljust(12) + f"{ta:>16.1f}{tb:>16.1f}{d:>8.1f}%{ia:>10.0f}ms{ib:>10.0f}ms")
    for name, d in ((a, da), (b, db)):
        cp = d.get("coding_peak")
        if cp:
            keys = [k for k in cp if isinstance(cp[k], (int, float))]
            print(f"coding_peak {name}: " + ", ".join(f"{k}={cp[k]:.2f}" if isinstance(cp[k], float) else f"{k}={cp[k]}" for k in keys[:8]))
pa, pb = load(f"{a}-prefill.json"), load(f"{b}-prefill.json")
if pa and pb:
    print(f"{'prefill':<12}{a+' tok/s':>16}{b+' tok/s':>16}{'delta':>9}")
    for k in pa.get("prefill", {}):
        ta = pa["prefill"][k]["tok_per_sec"]; tb = pb["prefill"].get(k, {}).get("tok_per_sec", float("nan"))
        print(f"{k:<12}{ta:>16.0f}{tb:>16.0f}{(ta/tb-1)*100:>8.1f}%")
