# GLM-5.2 4096×6144 K6/MCG optimization qualification

Date: 2026-08-26

## Verdict

The b12x-owned 144-CTA planner policy is qualified for the exact TP4 rank-zero
FP16 `model.layers.3.self_attn.o_proj` payload from
`GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`.

The result is deliberately workload-scoped:

- **Kernel performance: pass.** Under balanced CUDA-graph replay, the normal
  planner route is 9.41% lower latency than the served ExLlamaV3 projection at
  M=1 and 6.51% lower at M=4. It is at timing-resolution parity at M=8, M=12,
  and M=16.
- **Correctness and serving invariants: pass.** Exact checkpoint tensors,
  numerical oracles, finite/nonzero and overwrite checks, one-kernel graph
  identity, stable addresses, mutation replay, fixed caller-owned capacity,
  and zero replay allocation all passed.
- **Unrelated-shape protection: pass.** GLM `q_b_proj` and the Qwen
  17408×5120 BF16 shape stayed within the 1% regression budget and compiled to
  identical cubins in base/candidate/base panels.
- **Full-service safety: pass.** All six TP4/DCP4/MTP3 service arms started,
  served the cache-proof matrix, preserved the complete checkpoint profile,
  and shut down cleanly.
- **Full-service performance: mixed.** The upstream A-B-A panel measured a
  short-concurrency-1 E2E improvement of **+0.346%** against **0.025%** outer
  E2E drift. Other workload classes were mixed or within drift. This report
  does not claim that the complete service is universally faster.
- **KLD: pass.** Five optimized-image runs averaged 0.072603, within the
  checkpoint's historical qualified range and far from the broken-FC2
  signature.

The machine-readable disposition and evidence hashes are in
[`evidence/qualification_receipt.json`](evidence/qualification_receipt.json).

## What changed

The b12x planner change is one shape-specific entry:

```python
_GLM_GRID_CTA[(4096, 6144)] = 144
```

That binds a 144-CTA launch and 294,912 FP32 scratch elements (1,179,648
bytes) for the exact GLM TP4 `o_proj` shape. b12x remains the policy owner; the
vLLM integration only consumes the prepared launch and preallocates its
declared capacity.

The benchmark harness also now sizes fused scratch from the immutable bound
launch's `required_scratch_elements`. Previously, an experimental grid could
be bound while the temporary override was active, but scratch was recomputed
after the override had been restored. Grids larger than the public default
could therefore decline silently because the benchmark had underallocated
scratch. The fix makes the benchmark exercise the launch it says it is
testing.

No K6/MCG kernel math, quantization semantics, tile, thread count, H128 work,
barrier protocol, supported-row bound, generic fallback, or vLLM policy was
changed. In particular:

- W4A16 remains activation-typed FP16/BF16 input with inline FP4/NVFP4 weight
  dequantization; no activation-scale math was introduced.
- `_MAX_ROWS` remains 16.
- `B12X_DISABLE_STANDALONE_K6=1` remains an import-time kill switch.
- The Qwen `(5120, 6144)` grid remains 120, and unrelated compiled objects are
  unchanged.

PR #221 was reviewed as a reference implementation, not merged. It and
PR #243 cover the same M≤16 kernel class; the evidence supported a planner-only
parallelism correction rather than combining branches or creating an
M=17–128 path.

## Source and artifact identity

| Item | Identity |
|---|---|
| Optimization worktree | `/home/jon/git/local-inference-lab/b12x-glm52-pr243-o-opt` |
| Qualified PR #243 base revision | `706ad0eb54014ed9156dc82a7e0acff691662a89` |
| Qualified PR #243 base tree | `5e4c5566062306b76a6860d6a40c0f743aac0c89` |
| Qualified source diff SHA-256 | `a33e6f38db7b3ac594183b312fb234fa4fdadd1bb01c12ee27f9bead05b355a4` |
| Qualified implementation revision | `e7465a425bfa04c816901ee30a5f9c4f6a092a43` |
| Qualified/published K6/MCG runtime AST SHA-256 | `3b070bc70075e8db6f0014faf83de9d02162d405b67672f88ce2931ce4a8c7dc` |
| Fetched `origin/master` | `3a437ab5168060e4d625f05e1625c04089f1ba37` |
| Fetched PR #221 | `413f96e889dad1ae0752fd1f4be9d37f56849600` |
| master/#221 merge base | `6714ff09bc5be749c6f674ac8e2ba6a3b6a40ab4` |
| master/#243 merge base | `9ae32e297c7d8d5258e0953c111a933adcb687f5` |
| vLLM integration HEAD/tree | `8e7be4d5c97fb86d983bd5f83c825153452efaec` / `b930c0f215e4c18f105b4480022eca59299d6072` |
| Pre-existing vLLM diff SHA-256 | `329ccd993a5fad0c15fe75b3d2483e73da7e3b72c3779f43d84faa05ee33007e` |
| Optimized image | `local/glm52-pr243-o144:706ad0e-v39-v1` |
| Optimized image ID/digest | `sha256:c728d06ab95ee208bacecbf4b5948c735b5c0057cfffd179ef8b53b390c777d1` |
| Resolved candidate base digest | `sha256:d4cdc039cc3b7ef7be8e64ae51a70768cc7977994cd0547c01c3411a61339795` |

The pinned v39 image exposes the reviewed runtime through the `sparkinfer`
package namespace. The optimized image was built as a strict policy-only port
against that image's source: internal source SHA-256 changed from
`70edc835...a77f94` to `4fde57ae...a672f`, and image verification proves the
4096×6144 entry is 144 while the Qwen 5120×6144 entry remains 120. This is a
semantic integration of the b12x policy into the v39 ABI, not a claim that the
image file is byte-identical to the qualified b12x-namespaced source file.
The image receipt's `b12x_source_sha256` value
`99318df5...c186872` identifies the reviewed source snapshot used to authorize
the port. The published b12x file with SHA-256 `26aaed0f...dc988` differs from
that snapshot only in comments and formatting. Its runtime AST SHA-256 is
identical to qualified implementation revision `e7465a4`, so the executable
4096×6144 policy is unchanged.
See [`image-verification.json`](evidence/optimized_image_v1/image-verification.json).

Qualification did not deploy or promote an image and did not alter production
services. Publishing this report and its source changes to PR #243 does not
change that runtime scope.

## Checkpoint and hardware identity

The exact model source was:

- checkpoint: `/data/models/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`;
- shard: `model-layer-003.safetensors`, 165,527,424 bytes,
  SHA-256 `4bfb3a07979529760f2643a6491b05c84027d7ebf510564a913f1d7962fcb5aa`;
- index SHA-256:
  `20ed38193742d442a5ba444b07945527d49b5d68f152aa88131adf668ca54727`;
- verified `SOURCE_R7_MANIFEST.json` SHA-256:
  `df2f4c87b22c21c5234ef216149f5b5adc556820bb97ae4bc6dd7f4f0647b8db`;
- local TP4 rank-zero shape: K=4096, N=6144;
- exact activation contract: FP16;
- exact local tensor hashes are recorded in the consolidated receipt and the
  normal-planner result.

The checkpoint does not ship an independent shard checksum list. The shard
hash above is an exact recorded identity, while the checkpoint-owned source
manifest hash was independently verified. Those two facts are kept distinct.

Projection tuning and focused qualification ran on host `epyc`, physical
GPU 6:

| Property | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| UUID | `GPU-5e983ae0-0c39-779f-3f35-fc3744a2398d` |
| SMs / architecture | 188 / `sm_120a` |
| Memory | 102,014,189,568 bytes |
| Power limit / compute mode | 300 W / Default |
| Driver | 610.57.04 |
| Torch / CUDA | 2.12.0+cu132 / 13.2 |
| CUTLASS DSL | 4.6.0 |
| Runtime nvcc/ptxas | 13.2.78 |

Formal resource extraction reassembled retained PTX with ptxas 13.3.27; that
toolchain distinction is recorded because runtime compilation and timing used
13.2.78.

## CTA-grid sweep

Each grid used a fresh isolated compile/cache directory, exact checkpoint
tensors, FP16 correctness gates, CUDA-graph capture/replay, balanced route
ordering, 6,000 warmups, 240 timed iterations, and 12 cold replays. All eleven
grids passed correctness, route identity, graph safety, stable-address,
mutation, overwrite, and replay-allocation gates before their raw timings were
recorded. The sweep locates the CTA plateau; grid selection uses the
clock-comparable repeat samples described below.

Each retained sweep manifest contains a Docker inspect payload whose image ID
matches the declared
`sha256:d4cdc039cc3b7ef7be8e64ae51a70768cc7977994cd0547c01c3411a61339795`.
`run_grid_sweep.py` resolves that ID and rejects a mismatch before creating run
output; its containers use `--pull=never`, `--network=none`, and `--ipc=none`.

The table reports fused median latency in microseconds; lower is better.

| CTAs | FP32 scratch elements | M=1 | M=4 | M=8 | M=12 | M=16 |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 98,304 | 32.256 | 32.320 | 32.352 | 32.352 | 32.352 |
| 64 | 131,072 | 28.256 | 28.256 | 28.224 | 28.256 | 30.064 |
| 80 | 163,840 | 26.208 | 26.176 | 26.208 | 26.208 | 26.192 |
| 96 | 196,608 | 22.112 | 22.112 | 22.080 | 22.112 | 22.080 |
| 112 | 229,376 | 22.112 | 22.112 | 24.128 | 24.160 | 24.160 |
| 120 | 245,760 | 22.080 | 22.112 | 22.112 | 24.128 | 24.160 |
| 128 | 262,144 | 22.080 | 22.112 | 22.112 | 22.112 | 22.112 |
| **144** | **294,912** | **20.032** | **20.064** | 22.112 | 22.112 | 22.112 |
| 160 | 327,680 | 20.064 | 20.064 | 22.080 | 22.080 | 22.080 |
| 176 | 360,448 | 20.096 | 22.112 | 22.112 | 22.112 | 22.112 |
| 188 | 385,024 | 20.096 | 22.112 | 22.080 | 22.112 | 22.080 |

The 64-CTA starting policy was under-parallelized: it took 28.256 µs at M=1
and M=4 versus 22.112 µs for the served ExLlamaV3 route in this valid-telemetry
run. The plateau moved at 96, 128, and 144 CTAs. This made structural kernel
changes unnecessary.

The leaders were repeated in both orders, with 600 iterations per arm:

| Repeat order | 144 CTAs M=1 / M=4 | 160 CTAs M=1 / M=4 |
|---|---:|---:|
| 160 → 144 | 20.032 / 20.064 | 20.032 / 20.064 |
| 144 → 160 | 20.032 / 20.064 | excluded / 20.064 |

The fully valid 160→144 repeat measured both M=1 arms at 2,272 MHz and found
identical 20.032 µs medians. In the reverse repeat, the grid-160 M=1
post-timing snapshot had already fallen to 832 MHz while grid 144 remained at
2,280 MHz, so that one comparison is excluded. The reverse-repeat M=4 clocks
were 2,257 and 2,265 MHz and both medians were 20.064 µs. The exploratory sweep
also captured post-timing M=1 downclocks and is not used to distinguish grids
144 and 160 at that row.

Grid 144 was selected from the fully valid repeat tie and the valid M=4
reverse-order tie because it uses 10% less FP32 scratch and launches fewer
CTAs. The independent normal-planner run below then qualified the selected
policy against ExLlamaV3 with balanced within-run route ordering. Some
non-leading 176/188 samples observed throttle mask `0x4`; they were not used to
support the speed claim.

Raw evidence:

- [`sweep_manifest.json`](evidence/grid_sweep_fp16_v1_20260826/sweep_manifest.json)
- [`leader repeat 160→144`](evidence/leader_repeat_1_160_144/sweep_manifest.json)
- [`leader repeat 144→160`](evidence/leader_repeat_2_144_160/sweep_manifest.json)
- [`reverse-repeat telemetry disposition`](evidence/leader_repeat_2_144_160/telemetry_disposition.json)

## Normal-planner performance

After integrating the policy, a separate run used the packaged normal planner
with no experimental override. It used 6,000 warmups, 600 balanced iterations,
and all six route permutations equally. The ratio direction is
`ExLlamaV3 latency / fused latency`; greater than one favors b12x.

| M | b12x fused (µs) | served ExLlamaV3 (µs) | Throughput ratio | b12x latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 20.032 | 22.112 | 1.1038× | 9.41% |
| 4 | 20.672 | 22.112 | 1.0697× | 6.51% |
| 8 | 22.112 | 22.112 | 1.0000× | parity |
| 12 | 22.112 | 22.112 | 1.0000× | parity |
| 16 | 22.112 | 22.112 | 1.0000× | parity |

The small M=4 movement from 20.064 µs in the sweep/repeat panels to 20.672 µs
in the independent packaged-planner run does not change the disposition: it
remains repeatably faster than the 22.112 µs control.

See the exact command, environment, telemetry, raw samples, and hashes in
[`normal planner qualification receipt`](evidence/normal_planner_o_fp16_v1_20260826/qualification_receipt.json).

## Correctness, graph, and allocation gates

Ten exact-checkpoint fused-vs-oracle comparisons across the row matrix and
adversarial scenarios produced:

| Gate | Result |
|---|---:|
| Minimum cosine similarity | 0.999999702 |
| Maximum relative L2 | 0.0005874504 |
| Maximum absolute error | 0.0001220703125 |
| Minimum top-8 overlap | 0.875 |
| Top-8 membership mismatch rows | 2 |
| Ambiguity-qualified mismatches | 2 |
| Unambiguous mismatches | 0 |
| Finite and nonzero outputs | pass |
| Fully overwritten outputs | pass |

Every fused graph contained exactly one cooperative
`K6McgSmallMKernel`, with a 144-block grid, 256 threads, and 50,176 bytes of
dynamic shared memory. No separate rotation-like kernel appeared. Caller-owned
source, output, activation-typed rotated storage, FP32 scratch, workspace, and
weight addresses stayed stable. Eight repeated/mutated replays per row had
zero allocator deltas and fully overwritten output.

CUDA graph construction itself showed the same 1,024-byte/two-allocation
framework bookkeeping signature for fused b12x, generic b12x, and ExLlamaV3.
That route-independent graph bookkeeping is recorded separately from route
execution: all route storage was caller-owned, and replay allocated nothing.

## Resource evidence

The selected launch retained the original kernel resources:

| Resource | Value |
|---|---:|
| Allocated registers | 132 |
| Exact SASS register sets | R=130, UR=13, P=7, UP=1 |
| Frame / local loads / local stores | 0 / 0 / 0 |
| Static / dynamic / total launch shared memory | 1,024 / 50,176 / 51,200 bytes |
| Occupancy | 1 CTA/SM |
| SASS instructions | 2,608 |
| Code size | 41,728 bytes |
| Cubin SHA-256 | `b8d6ce59ab173addbb7b5af56f93cbb5b00d63810fbb29443e41026e7e827370` |

All eleven grid values produced that same extracted cubin. The optimization is
therefore a launch-policy and fixed-capacity change, not a hidden compiled-code
change. See [`resource audit`](evidence/grid_sweep_fp16_v1_20260826/resource_audit_v1/report.json).

## Existing-shape regression protection

### GLM `q_b_proj`, FP16, 2048×4096

The base-candidate-base panel kept grid 64 and 131,072 scratch elements. Every
arm produced cubin SHA-256
`5748a801e29e76804c8dc7df9e3a52882d2efa4667808992368543bce1761c5d`.
The worst positive candidate delta was +0.1003%, below the 1% budget. All
correctness, graph, telemetry, and route-identity checks passed. Per-row
SM-clock spreads were 8–15 MHz, below the declared 30 MHz limit.

### Qwen down projection, BF16, 17408×5120

The original exact Qwen checkpoint is not present at its recorded path, so this
qualification contains no exact-checkpoint rerun. Its Qwen protection combines:

1. retained exact-checkpoint evidence with verified shard SHA-256
   `792b46b5...365f0d` and a passing 160-CTA fused result; and
2. a deterministic shape-only BF16 base-candidate-base latency/object panel.

The deterministic panel kept grid 160. Every arm produced cubin SHA-256
`615fcb7d1c6d8e79ad74a141c56f7ed03d2d9dd97f2953aea516299fe63959e0`;
candidate deltas ranged from -0.0685% to +0.0717%. This proves that the
shape-specific GLM entry does not change Qwen dispatch, compiled code, or
representative replay latency. The fixture is not used as checkpoint
correctness evidence. Per-row SM-clock spreads were 0–7 MHz, below the declared
30 MHz limit.

Evidence:

- [`GLM q_b panel`](evidence/regression_qb_fp16_panel_20260826.json)
- [`Qwen shape panel`](evidence/regression_qwen_shape_bf16_panel_20260826.json)
- [`retained exact Qwen result`](../evidence/qwen38_down_bf16_8159533/result.json)

## Full GLM service qualification

All arms used the complete checkpoint-owned serving profile:

- GPUs 0–3, TP4 and DCP4 with `a2a`;
- MTP3, Triton draft MoE, and greedy draft sampling;
- EXL3 weights, `nvfp4_ds_mla` KV cache with the checkpoint's outer scales,
  BF16 RoPE, and 48 R7 fused layers;
- GPU memory utilization 0.955, max 2,048 batched tokens, max 4 sequences,
  and max model length 262,144;
- full/piecewise CUDA graphs at 4, 8, 12, and 16;
- prefix caching and chunked prefill enabled;
- stable fixed allocation with 280,320 KV-cache tokens.

Prompts used a per-request nonce. Metrics confirmed zero prefix-cache hits and
zero cached prompt tokens for every measured workload, so the results are
cache-proof while preserving the production prefix-caching setting.

The arms were:

- **A:** v39 integration/fallback image
  `sha256:87fc18209cf28e58b79b52c7cdc194be8aeb2f0624d2e222915dba364f1685fa`;
- **B:** optimized image
  `sha256:c728d06ab95ee208bacecbf4b5948c735b5c0057cfffd179ef8b53b390c777d1`;
- **C:** the same optimized image as B, restarted with
  `B12X_DISABLE_STANDALONE_K6=1`.

Each workload had one warmup. Short and prefill workloads used five measured
samples; long workloads used three. Positive deltas in the tables below mean B
is faster than the midpoint of the two outer arms. `C1` means concurrency 1;
`C4` means concurrency 4.

### Upstream/fallback A1 → B1 → A2

| Workload | Decode Δ | Decode outer drift | E2E Δ | E2E outer drift | Acceptance Δ |
|---|---:|---:|---:|---:|---:|
| short C1, p128/o256 | +0.107% | 0.546% | **+0.346%** | **0.025%** | 0.000 pp |
| short C4, p128/o256 | -0.558% | 0.157% | -0.433% | 1.251% | -5.920 pp |
| long C1, p32768/o256 | +0.220% | 0.393% | -0.084% | 0.129% | 0.000 pp |
| long C4, p32768/o256 | -0.823% | 0.191% | -0.321% | 0.148% | +0.151 pp |
| prefill C1, p8039/o1 | n/a | n/a | -0.134% | 0.278% | n/a |

For the prefill row, prefill throughput changed -0.133% against 0.279% outer
drift.

The short-C1 E2E result is the narrow service-level win: +0.346% is materially
larger than 0.025% outer drift and acceptance was unchanged. It satisfies the
requirement that the kernel improvement appear in a real service workload.
The matrix does not support extrapolating that result to concurrency 4 or long
prompts.

### Same-image kill switch C1 → B2 → C2

| Workload | Decode Δ | Decode outer drift | E2E Δ | E2E outer drift | Acceptance Δ |
|---|---:|---:|---:|---:|---:|
| short C1, p128/o256 | -0.014% | 0.283% | +0.846% | 0.649% | -0.301 pp |
| short C4, p128/o256 | +2.518% | 3.047% | +1.121% | 0.440% | -6.098 pp |
| long C1, p32768/o256 | +0.020% | 0.266% | -0.244% | 0.033% | 0.000 pp |
| long C4, p32768/o256 | +0.521% | 0.171% | +0.107% | 0.049% | +0.022 pp |
| prefill C1, p8039/o1 | n/a | n/a | -0.088% | 0.342% | n/a |

For the prefill row, prefill throughput changed -0.088% against 0.343% outer
drift.

The C-B-C panel confirms that the same image is safe with the route enabled or
disabled. It is not used to claim a robust short-C4 speedup: its 3.047% decode
outer drift exceeds the 2.518% candidate gain, and speculative acceptance
moved by about six percentage points. Long-C4 is directionally positive, but
the absolute service effect is small.

All six arm receipts report success. Logs show the expected fixed-capacity
startup and CUDA-graph profile with no K6 illegal access, replay allocation, or
fallback-route error. An unrelated Triton draft-routing kernel compiled during
the first workload warmup in each applicable service process; measured samples
followed the warmup. K6 no-JIT/capture safety is established independently by
the exact projection graph qualification.

Raw panels:

- [`upstream A-B-A`](evidence/service_mtp3_panel_v1_20260826/e2e_upstream_aba.json)
- [`same-image C-B-C`](evidence/service_mtp3_panel_v1_20260826/e2e_kill_switch_cbc.json)
- [`cache inventory before cleanup`](evidence/service_mtp3_panel_v1_20260826/cache_inventory_before_cleanup.json)

The six generated service cache directories were isolated per arm, inventoried
and hashed, then removed after qualification. Route attribution comes from
image/source receipts, normal-planner graph evidence, and the same-image kill
switch panel—not from assuming cache files are byte-identical.

## KLD gate

The optimized image ran the checkpoint-owned eager full-logit KLD gate five
times over 2,047 positions per run:

| Run | KLD |
|---:|---:|
| 1 | 0.0742051 |
| 2 | 0.0747398 |
| 3 | 0.0732294 |
| 4 | 0.0718451 |
| 5 | 0.0689948 |
| **Mean ± sample SD** | **0.0726028 ± 0.0022978** |

The prior qualified integration mean was 0.0713090, the historical run range
was approximately 0.06398–0.07462, the documented reference was 0.069527, and
the known broken-FC2 result was approximately 2.36. The optimized-image mean is
inside the historical range and 1.81% above the prior mean. One individual run
is 0.00012 above the prior observed maximum; that is disclosed here and does
not resemble the broken-FC2 failure mode.

See [`KLD summary`](evidence/kld_optimized_v1_20260826/summary.json).

## Reproduction commands

The exact container commands, mounts, environment, image IDs, telemetry, and
output hashes are stored in each raw run receipt. The top-level commands were:

```bash
python validation/trellis_decode/glm52_o4096x6144_optimization/run_grid_sweep.py \
  --source-tree /home/jon/git/local-inference-lab/b12x-glm52-pr243-o-opt \
  --model-dir /data/models/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 \
  --output-root /home/jon/git/local-inference-lab/b12x-glm52-pr243-o-opt/validation/trellis_decode/glm52_o4096x6144_optimization/evidence/grid_sweep_fp16_v1_20260826 \
  --image local/glm52-pr243-candidate-integrated:706ad0e-v39-final \
  --image-id sha256:d4cdc039cc3b7ef7be8e64ae51a70768cc7977994cd0547c01c3411a61339795 \
  --source-revision 706ad0eb54014ed9156dc82a7e0acff691662a89 \
  --integration-tree 8e7be4d5c97fb86d983bd5f83c825153452efaec+diff-329ccd993a5fad0c15fe75b3d2483e73da7e3b72c3779f43d84faa05ee33007e \
  --gpu 6 \
  --grids 48,64,80,96,112,120,128,144,160,176,188 \
  --warmups 6000 --iterations 240 --cold-replays 12
```

```bash
python validation/trellis_decode/glm52_o4096x6144_optimization/run_checkpoint_qualification.py \
  --source-tree /home/jon/git/local-inference-lab/b12x-glm52-pr243-o-opt \
  --model-dir /data/models/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 \
  --output-root /home/jon/git/local-inference-lab/b12x-glm52-pr243-o-opt/validation/trellis_decode/glm52_o4096x6144_optimization/evidence/normal_planner_o_fp16_v1_20260826 \
  --image local/glm52-pr243-o144:706ad0e-v39-v1 \
  --image-id sha256:c728d06ab95ee208bacecbf4b5948c735b5c0057cfffd179ef8b53b390c777d1 \
  --source-revision 706ad0eb54014ed9156dc82a7e0acff691662a89+o144-policy-99318df5db87635e845bcd1ba93d6714e17b1cebdad51150ac892b227c186872 \
  --integration-tree 8e7be4d5c97fb86d983bd5f83c825153452efaec+diff-329ccd993a5fad0c15fe75b3d2483e73da7e3b72c3779f43d84faa05ee33007e \
  --tensor-prefix model.layers.3.self_attn.o_proj \
  --params-dtype fp16 --expected-grid-x 144 \
  --expected-scratch-elements 294912 --rows 1,4,8,12,16 \
  --gpu 6 --warmups 6000 --iterations 600 --cold-replays 12
```

Service arms used `run_service_arm.py` with isolated container names, ports,
cache roots, nonce bases, and output directories. For example, B1 was:

```bash
python validation/trellis_decode/glm52_o4096x6144_optimization/run_service_arm.py \
  --container glm52-o144-qual-b1 \
  --image local/glm52-pr243-o144:706ad0e-v39-v1 \
  --image-id sha256:c728d06ab95ee208bacecbf4b5948c735b5c0057cfffd179ef8b53b390c777d1 \
  --arm B1 --panel upstream_aba \
  --nonce-base glm52-o144-aba-v1-20260826 \
  --qualification-tool-root validation/trellis_decode/glm52_o4096x6144_optimization \
  --output-dir validation/trellis_decode/glm52_o4096x6144_optimization/evidence/service_mtp3_panel_v1_20260826/b1 \
  --port 18080
```

The complete generated `docker run` command and every workload command are in
[`B1 arm receipt`](evidence/service_mtp3_panel_v1_20260826/b1/arm_runner_receipt.json);
the other arm directories contain equivalent receipts.

## Verification and repository state

The required focused test command passed against the published source:

```text
71 passed in 20.82s
```

Ruff passed for all 17 changed Python files. Explicit Python source
compilation, the qualification-tool smoke gates, JSON parsing, evidence hash
verification, relative-link verification, and `git diff --check` also passed.
The focused test log is
[`focused_pytest.log`](evidence/final_verification_20260826/focused_pytest.log).

The qualified executable source is bound to PR #243 revision
`706ad0eb54014ed9156dc82a7e0acff691662a89` plus tracked diff SHA-256
`a33e6f38db7b3ac594183b312fb234fa4fdadd1bb01c12ee27f9bead05b355a4`.
That implementation is revision `e7465a425bfa04c816901ee30a5f9c4f6a092a43`.
The published planner-table comment preserves runtime AST SHA-256
`3b070bc70075e8db6f0014faf83de9d02162d405b67672f88ce2931ce4a8c7dc`;
qualification-tooling and evidence-interpretation changes do not affect the
serving executable.
The vLLM integration worktree with diff SHA-256
`329ccd993a5fad0c15fe75b3d2483e73da7e3b72c3779f43d84faa05ee33007e`
was read but not modified. Qualification containers and generated per-arm
caches were removed; the host services listed in the qualification receipt
were not stopped, recreated, or changed.

## Recommendation

Accept the 144-CTA b12x planner entry for the exact GLM FP16 4096×6144 M≤16
route. It fixes the demonstrated under-parallelization, is faster than the
served projection on the important M=1/M=4 kernel cases, preserves larger-row
parity, and does not disturb Qwen or GLM `q_b_proj`.

Describe its service benefit narrowly: **faster exact `o_proj` decode kernel,
with a measured short-C1 E2E gain; broad service throughput is otherwise mixed
and should be remeasured for any different production workload mix.**
