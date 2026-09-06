# Serving A/B on the final PR heads: RoCEnante vs NCCL, 2026-09-03

Measured configuration, identical for both arms except one environment variable:

- Nodes: four DGX Spark GB10 (`sm121a`), one GPU each, both ConnectX-7 functions per node, RoCE v2 GID index 3.
- Serving: vLLM fork `local-inference-lab/vllm` branch `roce/rocenante-shim` at `aef3576` (adapter, worker health check, config vote) on base `dev/jovian-judgement`; b12x `roce-oneshot-allreduce` at `00280f2` (fail-stop, rank-invariant dispatch, stream ordering). Image `spark-vllm:jovian-r12-glm53-sm121-arm64-0190180-aace94c-experimental-roce10` on all four nodes, built from those revisions (label `local-inference.b12x.roce.commit`). Later commits on both branches touch only the benchmark script, docs and receipts.
- Model and engine settings: GLM-5.3-Flash NVFP4 with the DFlash2 MTP5 draft, TP4, KV-33G profile, dual-HCA NCCL settings, chunk budget 8192 (compose stack `docker-compose-glm53-flash-dflash2-tp4-spark.yml` + the kv33g, dual-hca and chunk8192 overrides on each node). Synchronous scheduling.
- Arm ROCE: `docker-compose-roce-allreduce.override.yml` (`VLLM_ENABLE_ROCE_ALLREDUCE=1`, all-reduce cap 2 MB, all-gather cap 16 MB, `B12X_ROCE_SPIN_LIMIT=50000000`). Startup log: `Using ['B12X_ROCENANTE', 'PYNCCL'] all-reduce backends ... for group 'tp:0'`.
- Arm NCCL: `docker-compose-roce-off.override.yml`, the same file with `VLLM_ENABLE_ROCE_ALLREDUCE=0`. Startup log (`nccl-startup.txt`): `Using ['PYNCCL'] all-reduce backends ... for group 'tp:0'`.
- Correctness state: temperature-0 sanity request ("Write a Python one-liner that reverses a string.") returns the same slice-based answer under both arms (`nccl-startup.txt` for the NCCL arm; the RoCE arm's is in the deploy log of the session). Every benchmark request completed (`runs_ok=5/5` coding peak, no capacity-limited cells).
- Client: `llm_decode_bench.py` 0.4.32 on stinger against `maxwell:8000`, exact command in `run-arm.sh` (decode matrix c=1,2,4,8,16 x 0/16k/32k context, 30 s sustained per cell, max_tokens 512, exact token targeting, coding peak 5 runs x 2000 tokens; then standalone cold prefill 8k/32k/128k, 30 s each). `run-nccl-after-roce.sh` is the sequencing script that restarted the cluster between arms. The ROCE arm ran 21:47 to 22:03 local, the NCCL arm 22:06 to 22:22, back to back on the same nodes.
- Per-request raw samples, per-cell latency percentiles, event logs and the tool's metadata are in the four result JSON files (`*-decode.json`, `*-prefill.json`, compact layout, values untouched). `ROCE-vs-NCCL.txt` is `compare.py ROCE-final NCCL-final`.

| Node | Role | GPU | UUID | Driver |
|---|---|---|---|---|
| ampere | vLLM TP rank 0 | NVIDIA GB10 | `9ac97432-8546-9fc0-337d-3c0428d21656` | driver 580.173.02 |
| faraday | vLLM TP rank 1 | NVIDIA GB10 | `35fddba6-2b9c-a9dd-27cb-141630ca8f37` | driver 580.173.02 |
| hertz | vLLM TP rank 2 | NVIDIA GB10 | `99fe1d41-bd13-d96a-be78-faa7c33af618` | driver 580.173.02 |
| maxwell | vLLM TP rank 0 | NVIDIA GB10 | `128ca980-53cf-6524-40c5-a861a7941888` | driver 580.173.02 |

## Result (throughput is client-observed output tokens over the 30 s cell; itl is the median inter-token latency)

```
cell        ROCE-final tok/sNCCL-final tok/s    deltaROCE-final itlNCCL-final itl
c1@0k                   40.8            39.4     3.7%        62ms        65ms
c1@16k                  37.0            39.1    -5.5%        64ms        66ms
c1@32k                  40.8            36.5    11.8%        64ms        66ms
c2@0k                   61.9            58.4     6.0%        89ms        88ms
c2@16k                  54.4            53.7     1.4%        88ms        88ms
c2@32k                  59.4            55.9     6.3%        86ms        89ms
c4@0k                  108.9            92.1    18.2%        99ms       107ms
c4@16k                  94.8            89.6     5.8%       105ms       107ms
c4@32k                  97.9            80.4    21.8%       102ms       113ms
c8@0k                  126.3           120.9     4.5%       148ms       160ms
c8@16k                 130.7           114.2    14.4%       151ms       165ms
c8@32k                 131.3           122.2     7.5%       160ms       166ms
c16@0k                 215.2           163.7    31.5%       194ms       246ms
c16@16k                193.9           159.2    21.8%       212ms       256ms
c16@32k                191.6           152.2    25.9%       211ms       259ms
coding_peak ROCE-final: runs_requested=5, runs_ok=5, max_tokens=2000
coding_peak NCCL-final: runs_requested=5, runs_ok=5, max_tokens=2000
prefill     ROCE-final tok/sNCCL-final tok/s    delta
8192                    2601            2599     0.1%
32768                   2763            2758     0.2%
131072                  2701            2702    -0.0%
```

Coding peak (c=1, real coding prompt, generation tok/s over 5 runs): ROCE median 70.5 (mean 70.0, min 65.0, max 72.6); NCCL median 72.1 (mean 72.4, min 66.7, max 76.2). Difference -2%, inside the run-to-run spread of speculative acceptance; the +12.7% seen in the 2026-09-02 A/B was not reproduced here.

Reading: per-step latency is lower with RoCEnante in 14 of 15 cells (c=1: 62-64 vs 65-66 ms; c=16: 194-212 vs 246-259 ms) and equal in one; throughput gains grow with concurrency (c=4: +6 to +22%, c=8: +5 to +14%, c=16: +22 to +32%) because the all-reduce share of the step grows with batch size. c=1 throughput on padding text moves +-6% between cells from speculative acceptance, so read the step latency and the coding peak there. Prefill is unchanged (its 64 MB all-reduces are above the cap and stay on NCCL).
