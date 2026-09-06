# W4A16 direct-table bulk-copy qualification on SM120 TP8

Status: **qualified** for exact output identity and CUDA graph replay on
Kimi-K3 QSRT K2 layer 10 using one SM120 GPU at TP8; **not deployed**.
Hot-cache latency values are independent observations, not paired performance
qualification.

## Implementations

The vector-copy arm stages each CTA's 64 KiB direct rate table with distributed
global loads, shared stores, and a CTA barrier. Its measured source revision is
`46bfc80e0c5d0d1aa4a5a4a67a6463323047c7c5`, with source digest
`ba47552b9404836c03a4f392ba6fcbb1030efa427a0c0649a062729a7ce1d300`.

The bulk-copy arm stages the same bytes with one
`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` transaction and
waits on the CTA-scope barrier before FC1 or FC2 reads the table. Its measured
source revision is `ca02251b4ee41cfd343fcabf3a50d1d01b3e83df`, with source
digest `d012d17ac6a4af4b123fb6186ab2e8b6714fb12bb9662c136d160cc410921dbc`.
The compile result owns the selected device LUT slice, so capture-time launches
do not allocate or populate the process-lifetime direct-table cache.

Both Git worktrees were clean when the benchmark collected provenance. The
complete records, including every replay sample, are:

- `benchmarks/results/w4a16_direct_table_vector_copy_sm120_tp8.json`
- `benchmarks/results/w4a16_direct_table_bulk_copy_sm120_tp8.json`

## Measurement contract

- Model artifact: Kimi-K3 uniform-K2 coupled-Hadamard QSRT atoms-v2,
  `/mnt/models/Kimi-K3-QSRT-K2`.
- Completion manifest SHA-256:
  `c5f216807d27a31b7d1fbd7d0f6b745f767b1a27fb5edea23c2f1276e34a84b4`.
- Candidate-pool content SHA-256:
  `dec9725c3a05fffd53c2026db188db8aa32bfd914848110ad50f19d5b3b76708`.
- Layer artifact: `qsrt-layer-00010.safetensors`, 7,415,300,096 bytes.
- Geometry: TP8 rank 0, 896 experts, top-k 16, hidden width 3,584, local
  intermediate width 384, SiTU activation, BF16 input, FP16 prepared weights,
  and `(128, 128, 128, 128)` W4A16 tiles.
- Timed region: complete fused route packing, FC1, activation, FC2, and output
  reduction through CUDA graph replay. Checkpoint loading, weight preparation,
  compilation, graph capture, and 50 warmup replays are excluded.
- Samples: 400 hot-cache replays per implementation and routed-token count.
  The implementations ran in separate processes, with the vector-copy process
  followed by the bulk-copy process. Lower time is faster; ratios below are
  descriptive bulk-copy medians divided by vector-copy medians and do not
  control for run-order drift.
- Runtime image:
  `kimi-k3-production-issue75-dspark@sha256:bb9843ca63fe61b258077a3231a4136f143f942e259676225446df030afda767`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
  `GPU-9361804b-020b-3f67-9026-fca5d0264d5f`, SM120. Both arms recorded P1,
  3,082 MHz SM clock, 16,365 MHz memory clock, default compute mode, and no
  active clock-throttle reason before and after measurement.
- Environment: `CUDA_VISIBLE_DEVICES=0` and
  `B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM=1`. Each source revision used a distinct
  b12x compile-cache directory.

Each JSON artifact records its exact executable command. The shared arguments
were:

```text
B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM=1 CUDA_VISIBLE_DEVICES=0 \
  /opt/venv/bin/python3 benchmarks/benchmark_qsrt_checkpoint_profiles.py \
  --k2-checkpoint /mnt/models/Kimi-K3-QSRT-K2 \
  --profiles uniform_k2_coupled --tp-size 8 --tp-rank 0 --layer 10 \
  --tokens 4,8 --warmup 50 --replays 400 --cold-replays 0 \
  --bootstrap-replicates 100 --seed 20260904 --output <arm-json-path>
```

## Results

| Routed tokens | Vector median | Bulk median | Independent bulk/vector observation | Output SHA-256 equal |
|---:|---:|---:|---:|:---:|
| 4 | 116.384 µs | 114.336 µs | 0.9824 | yes |
| 8 | 171.680 µs | 167.600 µs | 0.9762 | yes |

For four routed tokens, the vector-copy p10/p90 interval was
115.360/119.424 µs and the bulk-copy interval was 114.304/117.888 µs. For
eight routed tokens, the corresponding intervals were 171.616/175.488 µs and
167.520/171.392 µs.

The exact output byte digests match between implementations:

- Four routed tokens:
  `9bb1e8c897b4c932282c78a1978abee317390ba62c07273a0e0951e788cb6a9d`.
- Eight routed tokens:
  `fdc7688894587397e5dca98a3121ee55f7e054c7c4bab7c271a5e7f998aa53aa`.

Every measured output was finite and nonzero, and each eager output was
bit-identical to its CUDA graph replay output.

## Conclusion and limitations

The bulk-copy prologue preserves the real checkpoint output exactly. Its
independent process run recorded a lower complete fused graph-replay median at
both routed-token counts, but the measurements do not isolate the copy
mechanism from run-order drift. This evidence qualifies correctness and capture
for the stated SM120 TP8 layer contract; a balanced or paired A/B is required
for performance qualification.

The benchmark uses one workstation-class GPU that also hosts a resident draft
model process. It does not measure an eight-GPU serving step, request-level
throughput, cold-L2 behavior, or Max-Q GPU clocks. Those cases are unsupported
by this result and require separate qualification before deployment.
