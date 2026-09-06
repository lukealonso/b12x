# Qwen3.8 Flash Next paired MXFP8 projection qualification

Status: qualified for NVIDIA RTX PRO 6000 Blackwell Workstation Edition decode
shapes.

The Qwen3.8 Flash Next Gated DeltaNet input path applies two independent MXFP8
linear projections to the same BF16 hidden state. The wide projection produces
16,384 QKVZ values per row and the narrow projection produces 96 BA values per
row; both consume 2,560 input values. `b12x.gemm.mxfp8_linear.mm_pair` overlaps
the narrow projection with the wide projection and joins the streams before
returning either output.

## Qualification contract

- Serial implementation revision: `9ae41c5cb9935d740456479954b0089f80bd2ef2`.
- Paired implementation revision: `d79342688246bfbd2693262114d45310076317d6`.
- Benchmark: `benchmarks/benchmark_mxfp8_linear_pair.py`.
- Physical GPU: index 2, UUID
  `GPU-167fbc3f-fd06-7f08-9e06-ee02946d041c`, PCI address
  `00000000:23:00.0`.
- Operating mode: stock automatic clocks, without an application clock lock or
  memory overclock.
- Software: CUDA 13.3 and PyTorch 2.13.0.
- Measurement: 20 warmup replays and 100 measured CUDA-graph replays per arm.
  Serial-first and paired-first order alternates between samples.
- Correctness gate: both output tensors must be bit-identical before timing.
- Ratio: serial median milliseconds divided by paired median milliseconds. A
  ratio above 1 means the paired operation is faster.

The command executed inside the container was:

```bash
python benchmarks/benchmark_mxfp8_linear_pair.py \
  --tokens 1 4 16 32 64 128 256 512 768 1023 1024 \
  --warmups 20 \
  --samples 100 \
  --physical-gpu-index 2 \
  --operating-mode "stock automatic clocks; no application clock lock or memory overclock" \
  --baseline-revision 9ae41c5cb9935d740456479954b0089f80bd2ef2 \
  --output docs/evidence/qwen38_flash_next_mxfp8_projection_pair_rtx6000_20260904.json
```

## Results

| Rows | Serial median | Paired median | Speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.025840 ms | 0.020768 ms | 1.244x | 19.63% |
| 4 | 0.027760 ms | 0.016816 ms | 1.651x | 39.42% |
| 16 | 0.023744 ms | 0.016608 ms | 1.430x | 30.05% |
| 32 | 0.021664 ms | 0.014560 ms | 1.488x | 32.79% |
| 64 | 0.029888 ms | 0.020912 ms | 1.429x | 30.03% |
| 128 | 0.042256 ms | 0.034944 ms | 1.209x | 17.30% |
| 256 | 0.058608 ms | 0.045328 ms | 1.293x | 22.66% |
| 512 | 0.105600 ms | 0.091712 ms | 1.151x | 13.15% |
| 768 | 0.140320 ms | 0.127184 ms | 1.103x | 9.36% |
| 1023 | 0.169952 ms | 0.154048 ms | 1.103x | 9.36% |
| 1024 | 0.168688 ms | 0.166800 ms | 1.011x | 1.12% |

The paired operation preserves exact outputs and reduces the complete
projection-pair GPU interval at every measured overlapping row count from 1
through 1023. The 1024-row case selects serial execution and measures the
wrapper overhead within timing noise. The committed JSON artifact
`docs/evidence/qwen38_flash_next_mxfp8_projection_pair_rtx6000_20260904.json`
contains all 100 raw timings for both arms and every row count.

This qualification covers CUDA-graph replay with the stated Qwen dimensions.
It does not claim a benefit for bandwidth-saturating prefill shapes; callers
must select a bounded row policy and retain serial execution above that bound.
