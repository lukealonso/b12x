# Committed GLM 4096×6144 qualification evidence

Status: **qualified with workload scope**.

This directory retains the compact evidence needed to review the b12x
144-CTA planner policy for the exact TP4 rank-zero FP16 GLM `o_proj` shape
M×4096×6144. The authoritative machine-readable disposition is
[`qualification_receipt.json`](qualification_receipt.json). Its `evidence`
object records the relative path and SHA-256 digest for every result used by
the qualification conclusion.

The repository retains:

- the complete CTA sweep summary and the selected grid-144 result;
- both balanced grid-144/grid-160 repeat summaries and the explicit disposition
  for one reverse-order M=1 post-timing clock anomaly;
- the normal-planner exact-checkpoint qualification receipt;
- the selected launch's resource audit;
- GLM `q_b_proj` and Qwen shape-regression summaries;
- upstream/fallback and same-image kill-switch service panel summaries;
- the optimized-image B1 service-arm command and runtime receipt from the
  fallback/optimized/fallback panel;
- the optimized-image verification receipt and build provenance;
- the five-run KLD summary; and
- the focused unit-test log.

The repository excludes artifact classes that are redundant, generated, or
too large for code review:

- compile, CUDA, Torch-extension, and container caches;
- the generated 66 MiB Qwen shape fixture;
- model weights and checkpoint shards;
- per-request service payloads and container logs;
- duplicated per-grid and per-arm raw timing payloads; and
- extracted objects, cubins, PTX, and SASS files.

The full raw evidence remains in the isolated qualification worktree at:

```text
/home/jon/git/local-inference-lab/b12x-glm52-pr243-o-opt/validation/trellis_decode/glm52_o4096x6144_optimization/evidence
```

That host-local path is an operational location, not a repository interface.
The committed receipt binds omitted artifacts through exact source, image,
checkpoint, tensor, manifest, result, and resource hashes. Performance claims
remain limited to the conditions and workloads stated in
[`../QUALIFICATION_REPORT.md`](../QUALIFICATION_REPORT.md).
