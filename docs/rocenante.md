# RoCEnante: one-shot RoCE collectives for multi-node DGX Spark TP

`b12x.comm.roce` (RoCEnante) is an all-reduce and all-gather runtime for tensor
parallelism across DGX Spark nodes joined by their ConnectX-7 200 GbE ports, one
GPU per node. It is a runtime API: a serving integration routes eligible
collectives to it and keeps everything else on its own backend. The vLLM adapter
in local-inference-lab/vllm#597 (`vllm/distributed/device_communicators/b12x_roce_all_reduce.py`,
base branch `dev/jovian-judgement`) does that for decode, which takes NCCL out of
the decode path for the models measured below.

Status: implemented and qualified on four DGX Spark (GB10, `sm121a`) nodes with
the tests and receipts named in this page. Other GPUs, hosts with GPUDirect RDMA,
and integrations other than that adapter are unsupported.

## Why it works on GB10 without GPUDirect RDMA

The GB10 is an integrated GPU with unified memory: pinned host memory is directly
addressable by GPU kernels at full bandwidth, and the NIC can register it with a
plain `ibv_reg_mr`. So peers RDMA-write into pinned slots that the local kernel
reads in place. No dmabuf or `nvidia_peermem` support is needed (neither is
available on the DGX Spark driver stack; NCCL therefore stays host-staged there).

Each Spark's single cabled QSFP port is exposed as two PCIe Gen5 x4 functions
(`rocep1s0f0`, `roceP2p1s0f0`). The runtime stripes every peer payload across
both functions so one rank pair can use both SoC-facing PCIe links.

## Protocol

One pinned region per rank: `recv[src][slot]`, `flag[src][slot][hca]`,
`send[slot]`, and a control record. One kernel launch per collective:

1. stage the input into `send[seq & 1]`;
2. the last block to finish staging publishes `nbytes` (per slot) and `seq` to
   the control record, which a C proxy thread (`_roce_proxy.c`, libibverbs)
   polls; the proxy divides each peer payload into one stripe per HCA, then
   posts each stripe followed by a 4-byte write of `seq` on the same reliable
   QP, so a stripe flag cannot land before its data. The doorbell holds only
   the newest `seq`, and a rank's kernel for
   op N finishes on the peers' payloads alone, so op N+1 can ring before the
   proxy has seen op N; the proxy posts every sequence between the last one it
   posted and the doorbell (at most two are ever pending). After a run of polls
   without a doorbell the thread requests a short sleep between polls (the OS
   decides the actual delay) so an idle runtime does not hold a core; the
   catch-up keeps the protocol correct however long the thread is away;
3. wait on `flag[peer][seq & 1][hca] == seq` for every peer and HCA (bounded; a
   timeout records the missing peer and HCA, the kernel skips its data phase
   and keeps the epoch, later launches do nothing, and the host raises: see
   Contract);
4. all-reduce: sum the local input and every peer slot in fixed rank order, so
   all ranks produce bit-identical output; all-gather: strided copy that writes
   the concatenated layout directly (dim 0 or last dim);
5. the last block to finish advances a device-resident epoch, which makes `seq` a
   runtime value and keeps CUDA-graph replay correct.

Two slots suffice because a peer cannot start op k+2 before finishing op k+1,
which needs our op k+1 data, which we only post after our op k kernel completed.
Staging and tail arrivals use separate counters (a block with nothing to stage can
pass the wait before a slower block has staged).

## Interface

`AllReduce.from_exchange_group(exchange_group=<gloo group>, device=..., max_size=,
max_gather_bytes=)` mirrors `comm.pcie.AllReduce`: `should_allreduce`,
`all_reduce`, `should_all_gather`, `all_gather(inp, dim=)`,
`prepare(dtypes, padded_gather=)` (compile and allocate before graph capture),
`for_stream`, `capture`, `check_health`, `poisoned`, `close`. `API_VERSION` is
bumped on incompatible surface changes; integrations pin the value they target.
Exchange setup over a CPU (gloo) group: using a torch NCCL group would create a
torch NCCL communicator costing about 3.4 GB of unified memory per rank.

Environment: `B12X_ROCE_HCA` (falls back to `NCCL_IB_HCA`), `B12X_ROCE_GID_INDEX`
(falls back to `NCCL_IB_GID_INDEX`, default 3), `B12X_ROCE_SPIN_LIMIT`,
`B12X_ROCE_CACHE_DIR` (where the proxy .so is built with the host C compiler).

Constraints: 2 to 16 ranks, one collective in flight per runtime, integrated
GPU with unified addressing, active RDMA devices.

## Contract

**Eligibility is rank-invariant.** `should_allreduce` and `should_all_gather`
decide from dtype, shape, contiguity and byte size only, which tensor-parallel
ranks share; a pointer that is not 16-byte aligned is staged through runtime
scratch rather than declined, and strides are part of the contract (a
non-contiguous tensor is declined on every rank alike). A closed runtime raises
instead of declining, so no rank can fall back to another backend alone. Both
gather paths stage the shard contiguously and only the reader is strided, so
ranks may take different paths for the same collective.

**Configuration is checked at setup.** Every rank publishes its API version,
proxy ABI, HCA count, slot geometry, size limits, spin limit and launch geometry
in the setup exchange; any difference fails construction on every rank.

**Failures are fail-stop, never a fallback.** A wait that exceeds the spin limit
records the sequence and the missing peer in the control record; the kernel
skips its data phase and does not advance the epoch, so every later launch on
that rank is a no-op. `check_health` raises from then on, before any further
launch and whenever the integration calls it. The integration calls it after
each step's own device-to-host synchronization (vLLM copies the sampled tokens
to the host every step), so no extra synchronization is added and the step's
output never leaves the worker. Inside the step, kernels after the failed
collective do consume its output, inside a worker that is about to exit. Ranks
converge without a supervisor: a rank that stops posting starves its peers'
next wait, so each peer times out and poisons itself within one spin limit,
including the asymmetric case where one rank received every flag and advanced
while another timed out. Choose the spin limit as the failure-detection latency
(`B12X_ROCE_SPIN_LIMIT`, about a microsecond per poll; the default 20M is about
half a minute); it must exceed the longest legitimate rank skew, which for
tensor-parallel decode is milliseconds.

**Streams.** Collectives on different streams are ordered with an event
recorded under the admission lock, so they execute in launch order; inside a
CUDA graph capture every collective must use the capture stream.

**Memory.** The pinned transport region is `world_size x 2` receive slots plus
2 send slots of `max(max_size, max_gather_bytes)` bytes each, plus flags: 160 MB
for four ranks at the 16 MB gather limit, 576 MB for sixteen. `prepare`
allocates two `max_size` alignment buffers; the padded-gather scratch
(`max_gather_bytes x (world_size + 1)`) is allocated only by
`prepare(padded_gather=True)` or the first eager padded gather, so a workload
with 16-byte-aligned rows never pays for it.

**Native proxy.** `_roce_proxy.c` is compiled once per source hash with the host
C compiler into `B12X_ROCE_CACHE_DIR` (default `~/.cache/b12x/roce`); it needs a
C compiler and the libibverbs headers. Serving images build it at image build
time (one import in the Dockerfile) so no compiler runs at startup.

## Results

Measured configuration: four DGX Spark GB10 nodes (`sm121a`, one GPU each,
unified memory), both ConnectX-7 functions per node, RoCE v2 GID index 3, bf16,
world size 4. Timing is CUDA events around one call, median over samples per
rank, then the slowest rank; graph replay is the decode path. The receipt
`docs/evidence/rocenante/20260903-4spark-bf16-standalone-00280f2.json` (emitted by
`benchmarks/benchmark_roce_oneshot.py`) holds the command, source revision,
worktree state, per-rank GPU identity, correctness results, raw samples, the
executed arm order, and ratios with their direction. The earlier receipt
`20260902-4spark-bf16-standalone.json` (commit 3478415, before the fail-stop
change) is kept for comparison: graph replay is within 1.3 us at every size.

| Collective | NCCL | RoCEnante eager | RoCEnante, graph replay | NCCL / graph |
|---|---|---|---|---|
| all-reduce 8 KB | 52.6 us | 46.6 us | 16.8 us | 3.1 |
| all-reduce 48 KB (6-token decode step) | 65.5 us | 52.8 us | 23.6 us | 2.8 |
| all-reduce 256 KB | 174.3 us | 88.4 us | 58.3 us | 3.0 |
| all-reduce 1 MB (128-token batch) | 899.2 us | 261.7 us | 238.9 us | 3.8 |
| all-gather [6, 38720] logits shard | 337.3 us (incl. reshape copy) | 190.8 us | 96.9 us | 3.48 |
| all-gather [96, 38720] (7.4 MB shard) | 1540.3 us | 1552.0 us | 1528.2 us | 1.01 |

Correctness on the same run: bf16 sums within 6.8e-3 relative error of NCCL
(float32 accumulation, rank-identical bits), all-gathers bit-exact, checked before
and after timing. Shards of several megabytes are bandwidth-bound for both
implementations; the gains are in the latency-bound region that decode lives in.

Serving A/B on the final revisions (b12x `00280f2`, vLLM adapter `aef3576`, one
image, RoCEnante toggled by one environment variable, arms back to back;
`docs/evidence/rocenante/serving-ab-final-20260903/` with commands, revisions,
GPU identities, startup lines, sanity output and raw samples): per-step decode
latency lower in 14 of 15 cells of the concurrency-by-context matrix (c=1 62-64
vs 65-66 ms, c=16 194-212 vs 246-259 ms), throughput c=4 +6 to +22%, c=8 +5 to
+14%, c=16 +22 to +32%, prefill unchanged, coding-peak c=1 within noise (-2%).

Earlier serving A/B (2026-09-02, before the fail-stop change), GLM-5.3-Flash TP4
with MTP5 on the same four nodes through the vLLM adapter, RoCEnante versus NCCL
for the same image and configuration (`docs/evidence/rocenante/serving-ab-20260902/`: per-arm decode and prefill
result JSON from `llm_decode_bench.py`, the arm comparison tables, and the README
that names the image, compose overrides, and commands): per-step decode latency
lower in every cell of a 15-cell concurrency-by-context matrix (5 to 19%), c=16
throughput +15 to +23%, coding-peak c=1 +12.7%, prefill unchanged (its 64 MB
all-reduces stay above the size cap and remain on NCCL). With both collectives
routed, a decode-step profile shows zero NCCL kernels.

## vLLM integration

The adapter lives in the vLLM fork, local-inference-lab/vllm#597 on
`dev/jovian-judgement` (`vllm/distributed/device_communicators/b12x_roce_all_reduce.py`
plus hooks in `cuda_communicator.py` and three entries in `envs.py`, about 180
lines). It dispatches eligible all-reduces and all-gathers to the runtime;
enable with `VLLM_ENABLE_ROCE_ALLREDUCE=1`, bound with
`VLLM_ROCE_ALLREDUCE_MAX_SIZE` (2MB) and `VLLM_ROCE_ALLGATHER_MAX_SIZE` (16MB).
The backend appears as `B12X_ROCENANTE` in the communicator's dispatch list.
Status: qualified with the serving A/B above against that fork revision; the
adapter is supported once #597 merges, and no other integration is.

## Tests and benchmark

`tests/comm/test_roce_oneshot_gpu.py` (torchrun, 2+ nodes): NCCL parity,
bit-identical ranks, dtype/shape eligibility and unaligned staging, dim-0/
last-dim/padded gathers, CUDA-graph replay mixing both collectives, the exact
adapter call pattern replayed with stable addresses, alternating eager streams,
a proxy that misses a doorbell, and fault injection (a stopped proxy) in eager
and graph mode checking the fail-stop contract on every rank. `benchmarks/benchmark_roce_oneshot.py`
times both collectives against NCCL.

## Unsupported

Fused all-reduce + residual + RMSNorm (the PCIe runtime has it), all-to-all for
expert parallelism, GPU-initiated posting (needs GDR support the platform lacks),
more than one collective in flight per runtime, and hosts without an integrated
GPU or without active RDMA devices (`is_supported()` returns False there).
