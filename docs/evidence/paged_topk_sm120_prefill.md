# Paged top-k-512 candidate-capacity qualification

## Status and operation contract

Status: **implemented** as an architecture-neutral top-k policy and
**qualified** on NVIDIA compute capability 12.0 (SM120).

The top-k-512 selector uses a 1,024-entry shared candidate buffer. Selectors
with top-k 1,024 or 2,048 use an 8,192-entry buffer. Candidate capacity is an
immutable compile-time property derived only from top-k width; selector launch
does not query CUDA capability or request state.

The exact overflow rescan preserves selection semantics when a threshold bucket
contains more candidates than the shared buffer. For a row with fewer live
candidates than top-k, each live candidate appears at most once and unused
entries are `-1`. An indices-only terminal fold leaves the score output buffer
untouched. Intermediate folds continue writing scores because later folds read
them.

Performance qualification is specific to the hardware and production geometry
below. The operation contract and compile-time policy are not architecture
gated.

## Source, hardware, and command

- Selector performance source revision:
  `03ba9855d28717096135f675852a8888fa79c189`, based on `master` revision
  `a45d3f7690e5b2f2e9bdcc0f76d76a48a0c490aa`.
- Correctness and benchmark-oracle revision:
  `a8306ab1d117bb53285b511c71ba4bcb5aad4f18`.
- Worktree: `/root/vllm/tmp/b12x-pr-paged-topk-20260830`.
- Physical GPU 2: `GPU-167fbc3f-fd06-7f08-9e06-ee02946d041c`, NVIDIA RTX PRO
  6000 Blackwell Workstation Edition, compute capability 12.0, 188 SMs, PCIe
  Gen5 x16, stock clocks, and a 600 W power limit.
- Toolchain: CUDA Python 13.3.1, PyTorch 2.13.0+cu130, CUTLASS DSL 4.6.2, and
  `CUTE_DSL_ARCH=sm_120a`.
- Performance environment: the worktree `.venv`.
- Correctness environment: `/opt/venv` from
  `voipmonitor/vllm:jovian-judgement-community-20260831-r10`, with the B12X
  checkout mounted read-only at `/src`.

The timed selector objects share source revision
`03ba9855d28717096135f675852a8888fa79c189`, package fingerprint
`351a2593b55428b9d81840384e4cddcf34a2fc76e3be8b5628366d451655a2e0`,
CUTLASS DSL 4.6.2, and PTXAS 13.3.73. The PTXAS executable has SHA-256
`afd8d1e1fa6e310f7faee44f6621e4c1315fb7fd6da7d4d87414358e12a651dc`.

| Candidate capacity | Compile manifest cache key | Compile manifest SHA-256 | Compile specification SHA-256 | Object SHA-256 |
| ---: | --- | --- | --- | --- |
| 1,024 | `377868b15c4457c61e769a9534df456d8f2cc067857bb977363aaed549d55f7a` | `153be41d246f0a8f70dc28b7f553766004a778b80027b57600dc461746d6af5c` | `75e920ecbecdc715ccc4061644e1370f66afc778d876909c791663bfe1871806` | `771c9ddf8f831bff42c45846744167faf9e51134cc15c297fc2df48632bf55ce` |
| 8,192 | `a77894d49ebd74b4d94927aff50f109effb6fc348540cbf855e1542eb707e31f` | `42bf35cb808e4771d005680c4e5c598cb98aa7824e82587ace50a4fd214a92ab` | `4044c4e1510075ae5f8b62e2eba1cc903b677c5713e6e70597a7a0eac0a64b08` | `3302781c9b37e18662de61381d25bdf4db96c1b4366da7ac30ab90772760b5e3` |

Both arms used the same source, analytic inputs, graph-replay path,
indices-only output contract, and cache state. The benchmark-only
`--topk-candidate-capacity` argument selected the compile-time capacity:

```bash
CUDA_VISIBLE_DEVICES=2 CUTE_DSL_ARCH=sm_120a \
  .venv/bin/python -B benchmarks/benchmark_paged_indexer.py \
  --rows 4080 \
  --global-heads 64 \
  --tp-size 4 \
  --page-table-width 128 \
  --seq-len 8192 \
  --mode supertile-topk \
  --route paged-tiled \
  --topk 512 \
  --topk-candidate-capacity 1024 \
  --indices-only \
  --check \
  --warmup 10 \
  --iters 30
```

The 8,192-entry arm changed only
`--topk-candidate-capacity 1024` to
`--topk-candidate-capacity 8192`. One preconditioning process per arm preceded
five measured processes per arm. Measured processes ran in balanced order, and
each process first passed the analytic top-k oracle and verified that the
indices-only launch left the binding-owned terminal workspace score slice
unchanged. Each process then reported the median of 30 graph replays.

## Correctness

The focused selector suite used the production source and 1,024-entry policy.
The command ran from the B12X repository root and exposed physical GPU 2 as
CUDA device 0 in the container:

```bash
docker run --rm --gpus '"device=2"' --ipc=host --shm-size 16g \
  --entrypoint /bin/bash \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTHONPATH=/src \
  -v "$PWD:/src:ro" \
  -w /src \
  voipmonitor/vllm:jovian-judgement-community-20260831-r10 \
  -lc '/opt/venv/bin/python -B -m pytest -q \
    tests/attention/test_benchmark_paged_indexer.py \
    tests/attention/test_attention_dsa_indexer_api.py \
    tests/attention/test_paged_prefill_topk_long_context.py \
    -k "analytic_topk or topk_candidate_capacity or row_topk or paged_prefill_topk"'
```

Result: `17 passed, 29 deselected`. The selection covers top-k policy,
64-bit physical-slot oracle arithmetic, indices-only output, short-row padding,
exact overflow selection, and CUDA graph replay with changing live inputs.

For indices-only `index_topk_fp8` runs, the benchmark initializes the
binding-owned terminal workspace score slice and verifies that the selector
leaves it unchanged. The following graph-replay samples provide correctness
evidence, not performance qualification:

| Rows | Visible tokens | Page-table width | Candidate capacity | Chunks | Oracle state |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4,080 | 8,192 | 128 | 1,024 | 1 | indices and untouched terminal scores pass |
| 4,080 | 8,192 | 128 | 8,192 | 1 | indices and untouched terminal scores pass |
| 1 | 64 | 1 | 1,024 | 1 | short-row padding and untouched terminal scores pass |
| 32 | 16,384 | 256 | 1,024 | 2 | carry fold, indices, and untouched terminal scores pass |

## Selector performance

| Candidate capacity | Process medians, microseconds | Median |
| ---: | --- | ---: |
| 1,024 | 967.66, 961.54, 969.73, 961.54, 961.54 | 961.54 us |
| 8,192 | 974.18, 984.86, 985.44, 986.05, 970.77 | 984.86 us |

The 1,024-entry policy reduces selector latency by 2.37%. The ratio direction is
8,192-entry latency divided by 1,024-entry latency:
`984.86 / 961.54 = 1.02425`, or 2.43% higher selector throughput.

This comparison isolates candidate capacity. It does not attribute the separate
gain from omitting unused terminal score writes, because both arms use the same
indices-only contract.

### Raw graph-replay samples

Arm A is the 1,024-entry candidate capacity and arm B is the 8,192-entry
candidate capacity. `pre-A` and `pre-B` are the independent preconditioning
processes. The measured process order was A1, B1, B2, A2, A3, B3, B4, A4, A5,
B5. Every value below is one graph-replay latency in microseconds; each line
contains all 30 replay samples from that process. The benchmark computed each
median before rounding individual samples to two decimal places for display.

```text
pre-A (median 973.39): 974.85,974.94,975.46,973.76,975.42,974.43,973.44,973.63,973.44,973.41,974.50,973.38,970.37,973.50,976.67,972.70,971.81,970.78,971.81,972.83,970.78,971.81,972.83,971.81,970.75,973.86,972.83,971.78,972.83,973.82
pre-B (median 980.00): 978.94,981.76,981.57,979.90,980.93,979.97,979.52,979.71,982.94,982.98,982.85,978.85,980.80,978.85,980.61,980.93,979.97,978.94,980.00,980.00,978.94,980.00,981.02,980.00,978.98,979.97,980.99,979.94,978.94,981.02
A1 (median 967.66): 968.70,968.45,968.64,968.26,968.51,967.65,968.80,968.32,968.61,968.48,966.59,966.69,965.54,966.66,967.58,966.66,967.68,966.66,965.63,964.61,965.63,966.69,968.70,967.68,968.70,967.65,966.66,965.63,968.70,967.68
B1 (median 974.18): 976.93,975.94,975.42,975.81,974.78,973.89,975.42,975.55,975.52,972.74,974.46,974.82,974.46,972.67,974.46,974.75,971.78,972.83,973.82,974.88,972.83,973.86,972.83,973.86,972.83,971.81,972.80,971.78,970.75,975.90
B2 (median 984.86): 987.14,986.98,987.78,985.92,987.87,985.95,985.12,983.78,983.84,985.02,984.74,985.12,986.75,985.02,984.74,984.99,984.06,983.04,983.04,984.06,983.07,984.10,985.09,982.05,983.04,984.06,983.07,985.09,984.10,985.09
A2 (median 961.54): 962.56,964.38,963.14,961.60,963.14,964.22,963.14,962.08,962.53,961.47,962.53,961.06,961.54,963.10,961.54,962.46,961.54,960.48,961.54,962.56,959.46,960.48,960.51,960.51,960.48,961.54,960.51,959.49,960.51,959.49
A3 (median 969.73): 970.75,973.66,973.38,971.81,971.33,971.74,971.33,969.63,971.39,969.57,970.37,968.61,969.73,972.70,971.78,968.48,969.73,968.70,967.68,970.75,968.70,969.70,968.70,969.73,968.70,967.68,968.70,967.71,968.70,969.73
B3 (median 985.44): 987.14,987.81,987.71,986.88,987.71,986.05,985.73,986.05,985.73,984.96,984.06,987.04,986.75,985.12,986.78,985.15,984.06,985.09,983.04,984.06,985.12,984.06,985.09,984.06,985.09,986.11,985.09,985.12,988.16,987.14
B4 (median 986.05): 985.09,983.90,987.17,984.64,989.28,988.74,985.02,984.77,989.09,989.82,987.94,987.78,985.98,985.73,985.86,987.81,986.11,985.09,985.09,986.11,987.14,984.06,985.09,986.11,985.09,988.19,989.18,987.10,984.06,985.09
A4 (median 961.54): 964.61,963.42,963.36,963.30,963.17,964.16,963.17,962.30,964.61,961.44,960.51,963.30,961.57,963.10,961.50,964.51,959.49,960.51,961.54,960.51,961.54,960.51,960.51,960.51,960.51,961.54,962.56,959.49,960.51,959.49
A5 (median 961.54): 964.61,962.56,961.09,961.47,963.14,960.22,963.14,961.44,961.15,962.11,962.53,963.10,961.54,965.15,961.54,962.46,961.54,962.56,961.54,960.54,959.49,962.56,961.54,962.56,962.56,961.54,960.51,959.49,960.48,959.49
B5 (median 970.77): 972.80,971.74,971.39,972.38,971.42,972.51,971.33,972.45,970.72,970.37,970.27,971.42,972.32,971.39,971.74,971.39,969.73,970.75,967.68,968.74,968.70,969.73,970.78,969.76,968.74,971.81,968.70,967.68,970.75,969.73
```

## Compiled-resource evidence

The B12X compile manifests and the embedded `sm_120a` fatbins identify the exact
selector objects:

| Candidate capacity | Object SHA-256 | Object bytes | Dynamic shared memory | Registers/thread | Stack | Local memory | Static shared memory |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | `771c9ddf8f831bff42c45846744167faf9e51134cc15c297fc2df48632bf55ce` | 61,288 | 20,096 B | 27 | 0 B | 0 B | 1,024 B |
| 8,192 | `3302781c9b37e18662de61381d25bdf4db96c1b4366da7ac30ab90772760b5e3` | 63,576 | 77,440 B | 27 | 0 B | 0 B | 1,024 B |

`cuobjdump --dump-resource-usage` reports no stack, spill, or local-memory use
for either object. The GPU permits 1,536 threads, 65,536 registers, and 102,400
shared-memory bytes per SM. Each selector CTA has 1,024 threads, so both objects
are limited to one resident CTA per SM by the thread limit. Both therefore have
32 active warps out of 48, or 66.7% theoretical warp occupancy. The smaller
candidate buffer improves latency without changing CTA occupancy, register use,
or spill behavior.

## Scope

The selector result does not establish end-to-end serving throughput. GLM-5.3
qualification must also include C4 scoring, page-table preparation,
communication, attention, MoE, and the remaining model work.
