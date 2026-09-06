# Single-rounding BF16 PCIe two-shot collective qualification

Implementation status: **qualified** for B12X commit
`bf3699b87fc1751e4eccf10f18361799d5ef8b86`, repository tree
`48744e1ae3cff42021df1ca478b2e64f36afea61`, and `b12x/` tree
`6d6af03e23c7b535aaaa82ae1dddb10f5c216edf`. The TP4-only gate,
caller-owned graph outputs, and expanded replay checks passed on physical GPUs
0 through 3 of the qualification host. All four devices are NVIDIA RTX PRO
6000 Blackwell Max-Q Workstation Edition GPUs.

The qualification used a fresh isolated compile cache. It produced 36 CUTLASS
DSL manifests and 36 objects: reduce-scatter, all-gather, and pull all-reduce
for four ranks in eager, graph-slot-0, and graph-slot-1 modes. Every manifest
and object hash verified, coverage was exact, and every manifest carried
package fingerprint
`7171ff95b5efdb9bb27787ef64e788866560c83394cbf665db4df38e7afedbfd`.

The retained serving measurements and indexed artifact table below remain
historical evidence for commit `7edd604a621ddbc3db1545e54d0e7031090bace5`.
They do not attribute serving throughput to the qualified implementation
commit above.

The `PCIeTwoShotBF16` collective transfers BF16 payloads over CUDA peer memory,
accumulates values in FP32 in a fixed rank order, and rounds the result to BF16
once. Its public operations are reduce-scatter, all-gather, and all-reduce. The
implementation rejects overlapping input and output storage because its
non-coherent peer loads cannot safely observe storage written by the same
kernel launch. Only world size 4 is supported. CUDA graph capture requires a
caller-owned preallocated output for every operation.

## Correctness contract

The distributed qualification test is:

```bash
NCCL_ALGO=Ring CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.run --nproc-per-node=4 \
  tests/comm/test_pcie_twoshot_bf16.py
```

The implementation qualification used the already selected physical devices
0 through 3 without changing `CUDA_VISIBLE_DEVICES`:

```bash
readonly CACHE_DIR=/path/to/fresh/compile-cache
PYTHONPATH="$PWD" \
B12X_COMPILE_CACHE_DIR="$CACHE_DIR" \
B12X_CUTE_COMPILE_CACHE_DIR="$CACHE_DIR" \
NCCL_ALGO=Ring \
python -m torch.distributed.run --nproc-per-node=4 \
  tests/comm/test_pcie_twoshot_bf16.py
```

The run completed with:

```text
pcie_twoshot_bf16 correctness OK (4 ranks, all_reduce_rows=(8, 16, 32, 64, 96, 128, 192, 256, 512), workspace_max_rows=512)
```

The test in this source tree executes all three public collectives at multiple
tensor heights under frozen kernel resolution after warming one static launcher
geometry. It checks the reduction against an exact FP32 sum, requires at most
one BF16 rounding, verifies deterministic eager execution, rejects overlapping
storage, and rejects rank-divergent graph-slot selection. It captures and
replays both all-reduce and a reduce-scatter-to-all-gather chain with
caller-owned preallocated outputs. Before each replay, the test mutates the live
inputs and poisons every output; replay must restore correct results, retain all
output addresses, and leave PyTorch CUDA allocator usage unchanged.

The retained successful run predates the TP4-only gate and expanded replay
checks. It exercised all three collectives eagerly and all-reduce under CUDA
graph replay, but it did not capture reduce-scatter or all-gather, mutate graph
inputs, poison graph outputs, or freeze kernel resolution across multiple live
row counts. Its result and artifacts therefore qualify only B12X commit
`7edd604a621ddbc3db1545e54d0e7031090bace5`, not the implementation described
by the strengthened test.

The test allocates a fixed IPC workspace for `max_rows=512`,
`row_elems=4096`, and world size 4. Its layout contains 266,240 signal bytes,
a 4,194,304-byte staged payload in each slot, a 1,048,576-byte reduced shard in
each slot, two 5,242,880-byte slots, and 10,752,000 bytes in the complete slab.
The 786,432-byte serving dispatch limit described below is a message-size
routing boundary; it is not the workspace capacity. The retained historical
test report is:

```text
pcie_twoshot_bf16 correctness OK (4 ranks, all_reduce_rows=(8, 16, 32, 64, 96, 128, 192, 256, 512), workspace_max_rows=512)
```

The retained compiled kernel artifacts use Python 3.12.3, PyTorch 2.13.0 with
CUDA 13.3, CUTLASS DSL 4.6.2, cuda-bindings 13.3.1, and PTXAS 13.3.73. Every
reduce-scatter, all-gather, and pull all-reduce artifact for ranks 0–3 uses
`opt-level=3`, relocatable device code disabled, assertions disabled, line
information disabled, 512 threads, and a 4,096-element row. The compile
manifests bind the rank and physical GPU UUID and contain separate exact
objects for eager slot selection and graph slot biases 0 and 1. The artifacts
bind B12X commit `7edd604a621ddbc3db1545e54d0e7031090bace5` and package tree
`5c13b2d9809025c5bf83c9ddb9071352acb60c0f`. Both serving arms resolved the
same source package fingerprint and toolchain mapping; only the dispatch limit
determined whether the pull all-reduce object executed. The rank, GPU, slot
mode, manifest, and object mapping is recorded in the
[SM120 artifact map](pcie_twoshot_bf16_sm120_artifacts.md).

## GLM-5.3-Flash serving measurement

The measured B12X integration source was commit
`cd89e4c7cf36e3366e49b0c09ef5e2deed45b8ea` with Git tree
`7f7107f7973a65aa5a94162b12a80b430abee252`. The measured vLLM integration
source was commit `c057c05522ca4b158be97a22a935633a00506124` with Git tree
`171465307585ecae2319284fe72d3a67610c5998`. These identifiers define the
software boundary for the measurements below; the enabled and disabled arms
used the same source trees.

The serving comparison used physical GPUs 4–7 at stock clocks, tensor
parallelism 4, decode context parallelism 1, an NVFP4 target checkpoint, an
MXFP8 DFlash2 draft checkpoint, seven speculative tokens per verifier step,
FP8 KV cache, B12X attention/MoE/linear kernels, and identical source and launch
arguments in both arms. The control disabled this collective with
`VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=0`; the candidate enabled it through
768 KiB. Each concurrency cell used a 15-second warmup followed by three
30-second samples. The matched enabled-versus-disabled comparison covers 8 and
12 concurrent requests (`C8` and `C12`). The enabled arm also measured one and
24 concurrent requests (`C1` and `C24`) to exercise its supported dispatch
shapes; no disabled-arm samples were retained for those two cells, so they are
not performance comparisons.

The qualification host used these GPU identities:

| Physical index | GPU UUID | PCI address |
|---:|:---|:---|
| 4 | `GPU-8800cf0c-1ba5-7136-d796-2a91f9e9586e` | `00000000:43:00.0` |
| 5 | `GPU-4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | `00000000:44:00.0` |
| 6 | `GPU-1a0323f7-8113-a1e1-c68b-f23fecf77171` | `00000000:63:00.0` |
| 7 | `GPU-0027fc86-3322-ce2a-856c-f49eb61eb63e` | `00000000:64:00.0` |

The B12X and vLLM checkouts were
`/root/vllm/worktrees/b12x-glm53-r17-perf-20260902` and
`/root/vllm/worktrees/vllm-glm53-r17-perf-20260902`. The container launcher
expanded to the following serving command; `ARM_LIMIT` was `0` for the NCCL
control and `786432` for the PCIe two-shot candidate:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
VLLM_ENABLE_PCIE_ALLREDUCE=1 \
VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE="$ARM_LIMIT" \
NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 \
NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=0 \
/opt/venv/bin/vllm serve /model \
  --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port 5051 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --decode-context-parallel-size 1 --cp-kv-cache-interleave-size 4 \
  --dcp-kv-cache-interleave-size 4 --max-num-seqs 32 \
  --max-model-len 262144 --max-num-batched-tokens 4096 \
  --prefill-schedule-interval 8 --max-cudagraph-capture-size 256 \
  --gpu-memory-utilization 0.90 --mamba-cache-mode align \
  --enable-chunked-prefill --enable-prefix-caching --dtype bfloat16 \
  --kv-cache-dtype fp8 --quantization modelopt_mixed --block-size 256 \
  --load-format instanttensor --attention-backend B12X \
  --moe-backend b12x --linear-backend b12x \
  --no-enable-flashinfer-autotune \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"flashkda"}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --speculative-config \
  '{"method":"dflash","model":"/draft","num_speculative_tokens":7,"attention_backend":"FLASH_ATTN"}' \
  --cudagraph-capture-sizes 1 2 4 8 16 32 40 48 64 96 128 192 256
```

The benchmark client was `llm_decode_bench.py` version 0.4.29 with SHA-256
`a17ee69dd2ee5aa59d9c9a1b03e28cae6fe2837545ecc967256b2828215deab7`.
Each sample used this command, with a distinct output path:

```bash
ARM=enabled  # use disabled for the NCCL control
N=1          # use 1, 2, and 3 for the three recorded samples
if [[ "$ARM" == enabled ]]; then
  CONCURRENCY=1,8,12,24
else
  CONCURRENCY=8,12
fi
mkdir -p "$ARM"
python /root/llm_decode_bench.py \
  --host 127.0.0.1 --port 5051 --model GLM-5.3-Flash \
  --contexts 0 --concurrency "$CONCURRENCY" \
  --duration 30 --decode-warmup-seconds 15 \
  --max-tokens 8192 --temperature 0 --skip-prefill \
  --display-mode plain --no-resume --output "$ARM/decode-run-$N.json"
```

The NCCL arm set `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=0`. The PCIe two-shot
arm set `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=786432`. No other server option
or environment value changed between the arms.

The reported cells are the raw post-warmup samples. Cold-start throughput was
not retained and is outside the steady-state decode claim. Both measured arms
passed the distributed collective test described above before serving. The
serving processes captured their declared CUDA graphs, retained stable graph
tensor addresses, completed every request without an API error, and selected
the intended NCCL or PCIe two-shot route at the declared boundary.

| Arm | Concurrency | Output tok/s samples | Verifier steps/s samples | Median output tok/s | Median verifier steps/s |
|:--|--:|:--|:--|--:|--:|
| Disabled | 8 | 732.047, 731.156, 729.200 | 281.098, 280.580, 282.060 | 731.156 | 281.098 |
| Enabled | 8 | 754.231, 748.416, 753.032 | 285.293, 286.877, 293.970 | 753.032 | 286.877 |
| Disabled | 12 | 865.016, 897.032, 900.058 | 341.857, 338.963, 342.022 | 897.032 | 341.857 |
| Enabled | 12 | 906.205, 909.330, 896.085 | 350.595, 351.172, 348.355 | 906.205 | 350.595 |

The enabled-arm shape and capacity samples excluded from the matched A/B
calculation were:

| Arm | Concurrency | Output tok/s samples | Verifier steps/s samples | Median output tok/s | Median verifier steps/s |
|:--|--:|:--|:--|--:|--:|
| Enabled | 1 | 246.317, 208.170, 219.334 | 85.739, 86.877, 86.634 | 219.334 | 86.634 |
| Enabled | 24 | 1259.033, 1245.146, 1269.194 | 476.939, 478.546, 477.249 | 1259.033 | 477.249 |

The enabled-minus-disabled median change, calculated as
`(enabled / disabled - 1) * 100`, was:

- C8: **+2.99% output tok/s** and **+2.06% verifier steps/s**.
- C12: **+1.02% output tok/s** and **+2.56% verifier steps/s**.

The retained conclusion is limited to the declared four-GPU SM120 topology,
the identified source revisions, and tensor sizes selected by the vLLM
integration. The performance-gain conclusion is limited further to the matched
C8 and C12 cells. Other GPU architectures, world sizes, message sizes, and
unmatched concurrency cells are not qualified as performance comparisons by
this report. These serving measurements do not qualify the later TP4-only and
graph-contract hardening changes.
