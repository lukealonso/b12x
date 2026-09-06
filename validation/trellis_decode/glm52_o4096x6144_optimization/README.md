# GLM 4096×6144 K6/MCG optimization

This directory contains durable tooling and evidence for optimizing b12x
PR `#243` on the exact TP4 rank-zero `o_proj` payload from
`GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`.

The qualification disposition is documented in
[`QUALIFICATION_REPORT.md`](QUALIFICATION_REPORT.md), with machine-readable
identities, metrics, limitations, and evidence hashes in
[`evidence/qualification_receipt.json`](evidence/qualification_receipt.json).
[`evidence/README.md`](evidence/README.md) defines the compact evidence retained
in the repository and the raw artifact classes intentionally kept host-local.
The selected b12x policy uses 144 CTAs for FP16 M×4096×6144. The report records
the exact target, comparison identities, correctness gates, raw timings, ratio
direction, and workload-scoped service disposition.

`benchmark_v39_checkpoint.py` reuses the repository checkpoint benchmark for
the checkpoint's pinned v39 serving image. That image packages the b12x runtime
under the historical `sparkinfer` module name. The adapter changes only that
module name and the checkpoint-owned TP slicing contract; it does not implement
a kernel, planner policy, route, or comparator. For non-GLM prefixes it
delegates to the upstream checkpoint loader so the same runtime can protect the
exact Qwen projection without changing load semantics.

`run_grid_sweep.py` is the benchmark-only CTA tuner.
`run_checkpoint_qualification.py` runs one projection through the packaged
normal planner by default and rejects any undeclared benchmark override.
`verify_v39_image.py` binds image identity to the image's internal policy patch
receipt.

`generate_k6_mcg_fixture.py` creates deterministic shape-only payloads for
compiled-object and latency regression panels when the original checkpoint is
not locally available. Such a fixture is never presented as checkpoint
correctness evidence. `analyze_regression_panel.py` validates a balanced
base-candidate-base panel, including exact tensor identity, planner dispatch,
embedded cubin identity, telemetry, a default 30 MHz per-row SM-clock spread
gate, and the one-percent regression budget.

`run_service_arm.py` launches one isolated service with the complete
checkpoint-owned MTP3/TP4/DCP4 performance profile, waits for health, runs the
fixed cache-proof workload matrix, captures the image/runtime/log receipt, and
removes only its own test container. The cache-proof request client
`benchmark_openai_stream.py` and runtime receipt collector
`capture_service_evidence.py` are committed beside the runner. The optional
`--qualification-tool-root` argument selects an alternate directory; every arm
receipt records the resolved paths and hashes. The qualification used
`benchmark_openai_stream.py` SHA-256
`5d3924204b3a3abd71166c052e858957d0c1d22c43bd63b82d9b054e3dfc3439`
and `capture_service_evidence.py` SHA-256
`acfb066ac45148473c050939c689dffc216256ec6ae781ad442234dcebd030f8`.

The checkpoint has no independent model-shard checksum list. Every result
therefore records the exact `model-layer-003.safetensors` digest and verifies
the checkpoint's `SOURCE_R7_MANIFEST.sha256`, while describing the distinction
explicitly in its JSON metadata.
