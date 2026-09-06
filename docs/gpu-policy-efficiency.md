# GPU policy representation and generation

Implemented: lossless decision-DAG serialization, GQA search over distinct
execution configurations, and compact JSON checkpoints. Runtime policies retain
exact device matching, explicit coverage, validation, override precedence, and
plan-time resolution.
The fixed timing protocol and MoE search remain unchanged.

## Mathematical model

For a planned query `q`, let `C(q)` contain its legal execution configurations.
For components with several scenarios, use the generator's existing geometric
mean latency `T(q,c)` across those scenarios. The policy objective is low
dispatch regret, not the smallest absolute latency across unrelated shapes:

```text
regret(q,c) = T(q,c) / min_{a in C(q)} T(q,a) - 1

policy loss = sum_q weight(q) * log(1 + regret(q, policy(q)))
              + complexity_penalty * representation_size(policy)
```

Runtime representation, candidate enumeration, and experimental query selection
solve separate parts of this problem:

- A shared decision DAG compresses an existing piecewise dispatch function
  without changing its decisions or coverage. It shares suffixes and configs
  and hoists universally required equality guards. It is not guaranteed to be
  the globally smallest diagram. The underlying principle is shared decision
  subfunctions, as in [decision-diagram reduction](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf).
- Search should operate on equivalence classes of configurations that produce
  the same execution. For GQA decode, identical tile geometry, complete chunk
  tables, chunk limits, and workspace capacities define the implemented
  equivalence. CTA budgets are excluded only after these results are fixed.
- Approximate query selection should spend measurements where they reduce
  expected dispatch regret, accounting for preparation and timing cost. A
  geometric nearest-neighbor index alone does not establish that a neighboring
  query has a legal or fast configuration. Conditional and categorical search
  spaces also motivate model-based configurators such as
  [SMAC](https://www.jmlr.org/papers/volume23/21-0888/21-0888.pdf).

## Representation qualification

Qualified against all three embedded profiles at source revision
`f9986fa6f94faf7317f9a13433791f229618e8ed`. The benchmark compares canonical
accepted path predicates, selected configs, leaf names, and evidence. It also
checks 8,694 covered query witnesses, including range endpoints. Unit tests
exercise holes, missing fields, defaults, Boolean/integer distinctions,
malformed references, excessive depth, and heavily shared graphs.

| Representation | Recompressed bytes | Unique nodes | Retained decoder allocations |
|---|---:|---:|---:|
| Nested tree | 308,821 | 93,648 | 55.7 MB |
| Shared DAG | 210,443 | 19,969 | 12.8 MB |
| DAG with common guards hoisted | 176,476 | 11,813 | 10.8 MB |

The checked-in gzip assets total 176,484 bytes versus 308,822 bytes for the
baseline assets. Their uncompressed JSON totals 2,838,222 versus 15,316,551
bytes. These asset sizes differ slightly from the table because the benchmark
reserializes all alternatives with the same canonical JSON writer.

Five fresh-process imports had median policy import times of 0.743 s for the
baseline and 0.142 s for the implemented encoding. Median process RSS was
102.0 MiB and 37.8 MiB, respectively. Direct lookup over the fixed witness
corpus had medians of 1.69 and 1.31 microseconds per query. These are CPU
measurements; policy lookup does not occur during GPU replay.

The runtime decoder accepts both nested trees and DAG format 1. Component query
and config schema versions do not change. Generated full artifacts and embedded
runtime artifacts both use compact planners. See
[the serialization contract](gpu-profiles.md#planner-encoding).

## GQA generation qualification

Implemented: GQA candidate contract 2 selects the first representative of each
decode execution class. It retains the complete config for validation and
provenance. Verifier queries preserve CTA-budget distinctions because their
kernel selection can consume that budget directly.

The complete 14,400-case, 360-group corpus was enumerated on the 188-SM Max-Q
device. Its 144,628 distinct serialized candidates reduce to 37,352 execution
representatives: 74.17% fewer candidates to prepare, qualify, and time. Every
original execution key has a retained representative; no measured performance
or neighboring query was used to discard a candidate. Enumeration took
300.5 s. This is a host-side search-space census, not a GPU timing sweep.

The production generator was exercised on Qwen3.8 Flash Next geometry with 24 Q
heads, 2 KV heads, 256-dimensional heads, batch capacities 1 and 8, and cache
capacities 128 and 16,384. BF16 KV, separate caches, and 64-token pages formed
the fixed four-case timing subset. Every case reduced from ten candidates to
two, removing 32 of 40 preparation, correctness, and timing cycles.

Three alternating-arm provider runs measured 2.198, 2.353, and 2.292 s for the
baseline and 1.854, 1.900, and 1.847 s for candidate deduplication. The median
reduction is 19.1%. These are diagnostic generation wall times on GPU UUID
`GPU-ac6fcbb2-ae5f-231d-cc3e-e843c305baff`, an RTX PRO 6000 Blackwell Max-Q with
188 SMs. Clocks varied and one baseline start was P3; these results do not
qualify a kernel latency improvement under fixed clocks. All reported throttle
masks were zero. Warmup, cold-L2 flushing, five groups of five replays, and
correctness thresholds were identical between arms.

Qualified: the 32-case extension across BF16/FP8 KV, separate/combined caches,
and page sizes 64/128 passed all 64 measured production plans. Every replay
allocation delta was zero; minimum cosine was 0.9999913 and maximum relative
L2 error was 0.0047807. This is targeted GPU qualification, not a GPU run of the
entire 14,400-case corpus.

Qualified: spawned workers on two 188-SM Max-Q GPUs completed the same 32 cases
as eight allocation groups. All 64 production plans passed correctness with
zero replay allocation. The minimum cosine and maximum relative L2 matched the
single-GPU qualification above. A complete resume reproduced the full artifact
in 3.53 ms with GPU-session entry prohibited. The physical devices were
`GPU-a0816187-68b2-b679-587f-0e56bac804f5` and
`GPU-9c204557-77b4-7ffb-c9f2-effcb51d054a`. This run verifies parallel generation
and resume; its idle-to-active clock transition does not qualify a latency or
generation-throughput comparison.

GQA declares compatible subset migration from candidate contract 1. A copied
four-case baseline checkpoint set retained eight exact recorded candidate
measurements, preserving their complete records and source provenance, with
GPU measurement calls prohibited. A second resume succeeded with session entry
prohibited. Other candidate-contract changes retain their invalidation behavior.

## Checkpoint representation

Implemented: checkpoints use compact, deterministic JSON with atomic per-record
replacement. Paths, payload schemas, compatibility checks, and independent
worker writes retain their existing contracts. Existing indented JSON files
remain readable and are not rewritten during resume.

Qualified: the 174,399-record measurement corpus was copied through each writer
on the same ext4 filesystem as the source checkpoints. Canonical record digests
match exactly across the complete corpus. Timed writes include source reads and
digest computation; random-lookup times are the median of three complete passes
with warm filesystem caches.

| Representation | Logical bytes | Allocated bytes | Write seconds | Lookup seconds |
|---|---:|---:|---:|---:|
| Indented JSON | 816,096,347 | 1,188,511,744 | 22.412 | 3.698 |
| Compact JSON | 588,192,014 | 926,629,888 | 13.391 | 3.552 |
| Gzip JSON, level 1 | 180,632,422 | 714,338,304 | 18.595 | 5.034 |

Compact JSON reduces logical bytes by 27.9% and allocated bytes by 22.0% while
preserving the atomic-file storage model. Per-file gzip saves more space but
increases lookup and bulk-resume cost; it is not enabled.

Three complete copies in alternating format order measured indented JSON writes
at 22.412, 22.314, and 22.228 s, and compact JSON at 13.391, 14.064, and 13.718 s.
The median write-wall-time reduction is 38.5%. This is a checkpoint-storage
measurement, not a measurement of complete GPU profile generation.

Research-only: SQLite prototypes intern generation metadata and store either
compact JSON or level-1 DEFLATE records. On ext4, rollback journals with
`synchronous=FULL` committed only 6,293 and 6,308 records, respectively, within
each 60-second write budget. The completed prefixes passed exact record
comparison and database integrity checks. A separate reader timed out after
five seconds during sustained writes. These commits have stronger power-loss
durability than the JSON writer, which uses atomic rename without `fsync`;
their times are not a comparison at equal durability.

A complete RAM-filesystem study demonstrated that compressed SQLite can reduce
allocated storage to 307,609,600 bytes, but it does not establish SSD write
performance. WAL was not tested: the installed SQLite 3.50.4 predates the
[upstream WAL-reset corruption fix](https://www.sqlite.org/wal.html).
The database prototypes are not production checkpoint backends.

## Sampling and timing alternatives

Research-only: a Matérn-3/2 Gaussian-process sampler over log2 token capacity
was compared with space-filling and winner-boundary sampling. It models relative
log latency with a one-octave length scale and 2% observation noise. Its
acquisition prioritizes uncertainty in the selected configuration's regret.
Device, source revision, measurement protocol, candidate contract, and eligible
candidate set are stratified; each observation contains all four route
scenarios. No GPU results are inferred or added to checkpoints.

The replay corpus contains 165,283 saved full-stage MoE records. Requiring
complete scenarios and at least eight capacities per stratum leaves 14,336
query points in 1,783 strata. At a half-query budget:

| Selection method | Geometric-mean regret | P99 regret | Maximum regret |
|---|---:|---:|---:|
| Space-filling | 0.647% | 17.47% | 436.98% |
| Winner-boundary refinement | 0.351% | 10.43% | 229.49% |
| Gaussian-process sampling | 0.303% | 9.87% | 74.77% |

These are counterfactual decisions evaluated against saved measurements, not
independent GPU qualification. The large tail losses preclude enabling this
sampler for profile generation. A useful sampler must account for support
boundaries and validate its uncertainty against held-out measurements; average
regret alone is insufficient.

Reusing CUDA event objects in four GQA cases saved less than 1% of timing wall
time. Capturing existing graph replays inside an outer graph failed with
`cudaErrorStreamCaptureUnsupported`; the runtime timing implementation is
unchanged. The equivalent MoE timing experiment aborts with SIGABRT inside
CUTLASS 4.6.2 `build_module` for `(E,K,N,top_k,tokens)=(256,2048,64,8,1)`, including
in the main checkout. It also aborts after integration onto
`f9986fa6f94faf7317f9a13433791f229618e8ed` on GPU
`GPU-a0816187-68b2-b679-587f-0e56bac804f5`. Its failure occurs before comparative
measurements.

Coarse/full MoE result reuse is not implemented. Activation seeds depend on
geometry, top-k, token count, and the base seed; they exclude the route pattern.
The input cache therefore does not make activation values depend on which route
is measured first. Reuse must account for qualification and timing semantics:
screening can use an independent reference, other races select a reference from
their candidate cohort, and automatic precision races retain paired samples and
independent confirmation passes. Matching a case and config alone does not
establish that those measurement contracts are equivalent.

## Reproduction

Use the project Python environment from the candidate checkout. Raw evidence
for the measurements above is under `/tmp/b12x-policy-efficiency.IT4SFh/` on the
qualification host; the commands recreate the studies in a fresh output directory.

```bash
python -m benchmarks.benchmark_policy_representation \
  --baseline-repo /path/to/f9986fa6-checkout --repetitions 5 \
  --output /tmp/policy-study/representation.json
python -m benchmarks.benchmark_gqa_search_space \
  --device 0 --output /tmp/policy-study/gqa-search.json
python -m benchmarks.benchmark_gqa_generation \
  --device 0 --repetitions 3 --output /tmp/policy-study/gqa-generation.json
python -m benchmarks.benchmark_gqa_generation --qualify-layouts \
  --device 0 --repetitions 1 --output /tmp/policy-study/gqa-qualification.json
python -m benchmarks.benchmark_gqa_generation --qualify-layouts \
  --devices 4 5 --repetitions 1 --output /tmp/policy-study/gqa-parallel.json
python -m benchmarks.benchmark_checkpoint_storage \
  --source /path/to/checkpoints --output-dir /var/tmp/checkpoint-study \
  --repetitions 3 --max-write-seconds 60
python -m benchmarks.benchmark_profile_sampling \
  --checkpoints /path/to/checkpoints/moe.decode \
  --output /tmp/policy-study/sampling.json
python -m benchmarks.benchmark_gqa_profile_timing \
  --device 0 --output /tmp/policy-study/gqa-timing.json
python -m pytest -q -p no:cacheprovider tests/policy
```

Policy tests: 277 passed; the baseline catalog consistency test fails because
the planned `sequence.kda_prefill` op has no registration. The installed
environment does not contain Ruff. No compiler, GPU clocks, existing checkpoint
corpus, or main-checkout files were modified for this study.
