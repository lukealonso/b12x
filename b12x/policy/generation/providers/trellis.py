"""Trellis launch races including production input/output rotations."""

from dataclasses import asdict, replace

from b12x.gemm.trellis_linear._tuning import TrellisConfig, TrellisQuery, TUNING_PROBLEM
from b12x.policy.generation.sweep import SweepCandidate

from .oneshot import _ApiGenerator, _Session
from .gpu_workers import _l2_flush_fn
from b12x.policy.generation.replay import measure_prepared_candidates


def _candidates(query, device):
    typed = TrellisQuery(**query)
    candidates = []
    for block in (16, 32, 48, 64):
        for k, n in ((64, 128), (64, 256), (128, 64)):
            config = TrellisConfig(backend="cutedsl", block_rows=block, tile_k=k, tile_n=n)
            try:
                TUNING_PROBLEM.lower(typed, device, asdict(config))
            except ValueError:
                continue
            candidates.append(SweepCandidate.create(asdict(config)))
    if not candidates:
        raise ValueError("native Trellis query has no legal launch decisions")
    return tuple(candidates)


def _hadamard_reference(source):
    import torch

    original = source.shape
    value = source.float().reshape(-1, 128)
    for width in (1, 2, 4, 8, 16, 32, 64):
        pairs = value.reshape(-1, 128 // (2 * width), 2, width)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        value = torch.cat((left + right, left - right), dim=-1).reshape(-1, 128)
    return (value * (128 ** -0.5)).to(torch.float16).reshape(original)


class _TrellisSession(_Session):
    def candidates(self, case):
        return _candidates(case.query, self.context.device)

    def measure(self, case, candidates):
        import torch
        from b12x.gemm import trellis_linear
        from b12x.moe._shared.kernels.w4a16.host import packed_gemm_scratch_elements
        from b12x.policy.generation.trellis_reference import _reconstruct_native

        if candidates != self.candidates(case):
            raise ValueError("native Trellis candidate set differs from its execution contract")
        query = TrellisQuery(**case.query)
        device = torch.device("cuda", self.context.device_ordinal)
        rng = torch.Generator(device=device).manual_seed(self.context.settings.seed + int(case.case_id[-8:], 16))
        m, k, n = query.max_rows, query.in_features, query.out_features
        with torch.cuda.device(device):
            def payload_for(rows, columns, bits):
                return torch.randint(-32768, 32767, (rows // 16, columns // 16, 16 * bits),
                                     device=device, dtype=torch.int16, generator=rng)

            scales = [torch.randn(size, device=device, generator=rng).sign().mul_(0.5).to(torch.float16)
                      for size in (k, n)]
            if query.weight_layout == "native":
                payload = payload_for(k, n, query.bits)
                decoded = _reconstruct_native(payload, codebook=query.codebook)
                weight = trellis_linear.prepare_weight(payload, *scales, codebook=query.codebook,
                                                       params_dtype=getattr(torch, query.compute_dtype))
            else:
                pair_kind, axis = query.weight_layout.split("_")
                rates = (2, 4) if pair_kind == "p24" else (3, 3)
                records = [payload_for(k if axis == "n" else 128, n if axis == "k" else 128, bits)
                           for bits in rates]
                decoded = torch.cat([_reconstruct_native(record, codebook=query.codebook) for record in records],
                                    dim=1 if axis == "n" else 0)
                payload = torch.cat([record.reshape(-1) for record in records])
                weight = trellis_linear.prepare_pair_weight(
                    payload, *scales, pair_kind=pair_kind.upper(), rate_axis=axis, codebook=query.codebook,
                    params_dtype=getattr(torch, query.compute_dtype))
            source = torch.randn(m, k, device=device, generator=rng).mul_(0.01).to(getattr(torch, query.input_dtype))
            decoded = decoded.to(device=device, dtype=getattr(torch, query.compute_dtype))
            rotated = _hadamard_reference(source.to(torch.float16) * scales[0]).to(getattr(torch, query.compute_dtype))
            projected = (rotated.float() @ decoded.float()).to(getattr(torch, query.compute_dtype)).to(torch.float16)
            expected = (_hadamard_reference(projected) * scales[1]).to(source.dtype)
            def allocate(shape, dtype):
                return torch.empty(shape, device=device, dtype=dtype)
            output = allocate((m, n), source.dtype)
            maximum_scratch = max(packed_gemm_scratch_elements(
                size_n=n, route_slots=((m + c.config["block_rows"] - 1) // c.config["block_rows"]) * c.config["block_rows"],
                moe_block_size=c.config["block_rows"], sms=self.context.device.sm_count,
            ) for c in candidates)
            kwargs = dict(output=output, gemm_output=allocate((m, n), getattr(torch, query.compute_dtype)),
                          c_tmp=allocate((maximum_scratch,), torch.float32),
                          input_f16=allocate((m, k), torch.float16), rotated_f16=allocate((m, k), torch.float16),
                          rotated_compute=allocate((m, k), getattr(torch, query.compute_dtype)),
                          gemm_output_f16=allocate((m, n), torch.float16), output_f16=allocate((m, n), torch.float16))
            measurements = []
            for candidate in candidates:
                def run(candidate=candidate):
                    result = trellis_linear.run(source, weight, **kwargs,
                                                _moe_block_size=candidate.config["block_rows"],
                                                _force_tile_config=(candidate.config["tile_k"], candidate.config["tile_n"]))
                    return (result,)
                measurements.append(self.prepare_candidate(candidate, run, (expected,)))
            return measure_prepared_candidates(measurements, settings=self.context.settings, device=device,
                flush=_l2_flush_fn(device, enabled=self.context.settings.cold_l2))


class TrellisLinearGenerator(_ApiGenerator):
    module = "b12x.gemm.trellis_linear"
    query_name = "TrellisQuery"
    measurement_kind = "candidate_race"
    candidate_contract_version = 3
    default_queries = tuple(
        dict(max_rows=m, in_features=k, out_features=n, input_dtype=dtype, compute_dtype=dtype,
             codebook=codebook, bits=bits)
        for codebook in ("mcg", "sqg_e4m3", "sqg_fp16")
        for bits in ((2, 3, 4, 5, 6) if codebook == "mcg" else ((5, 6) if codebook == "sqg_fp16" else (2, 3, 4)))
        for dtype in ("float16", "bfloat16")
        for m, k, n in ((3, 128, 128), (65, 256, 256))
    ) + tuple(
        dict(max_rows=m, in_features=256, out_features=256, input_dtype=dtype, compute_dtype=dtype,
             codebook=codebook, bits=3, weight_layout=layout)
        for layout in ("p24_k", "p24_n", "p33_k", "p33_n")
        for codebook in ("mcg", "sqg_e4m3")
        for dtype in ("float16", "bfloat16")
        for m in (3, 65)
    )

    def __init__(self):
        super().__init__()
        self._benchmark_factory = lambda _group, _cases, context: _TrellisSession(
            self.component_id, context)

    def estimate(self, context):
        estimate = super().estimate(context)
        count = sum(len(_candidates(case.query, context.device)) for case in self._cases)
        return replace(estimate, description=f"{count} Trellis launch measurements with input/output rotations",
                       dimensions={**estimate.dimensions, "candidate_measurements": count,
                                   "rotation_backend": "cutedsl"})
