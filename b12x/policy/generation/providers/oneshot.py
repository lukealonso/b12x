"""Production graph qualification for APIs without runtime policy lookup."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, fields, replace
import gc
import hashlib
import importlib
import json

from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator, SweepCandidate, SweepCase,
)
from b12x.policy.generation.replay import PreparedCandidate, capture_warmed_graph, measure_prepared_candidates

from .gpu_workers import _l2_flush_fn


_CANDIDATES = (SweepCandidate.create({"backend": "cutedsl"}),)


def _mxfp8_weight(batch, k, n, major, device, rng):
    import torch

    physical = (batch, k, n) if major == "n" else (batch, n, k)
    values = torch.randn(physical, device=device, generator=rng).mul_(0.25).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 126, (*physical[:-1], physical[-1] // 32),
                           device=device, dtype=torch.uint8, generator=rng)
    decoded = values.to(torch.bfloat16) * scales.view(torch.float8_e8m0fnu).to(torch.bfloat16).repeat_interleave(32, -1)
    return (values, scales), decoded if major == "n" else decoded.transpose(1, 2)


def _prepare(component, query, device, rng):
    import torch

    def random(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device=device, generator=rng).mul_(0.25).to(dtype)

    m = query["max_rows"]
    if component == "gemm.bf16_gemv":
        from b12x.gemm import bf16_gemv

        source = random((m, query["in_features"]))
        weight = random((query["out_features"], query["in_features"]))
        expected = (source.float() @ weight.float().T).to(source.dtype)
        return lambda: (bf16_gemv.mm(source, weight),), (expected,), False
    if component == "gemm.tensor_fp8_linear":
        from b12x.gemm import tensor_fp8_linear

        source = random((m, query["in_features"]), torch.float8_e4m3fn)
        weight = random((query["out_features"], query["in_features"]), torch.float8_e4m3fn)
        scale = torch.tensor([0.125], device=device)
        packed = tensor_fp8_linear.pack_weight(weight, scale)
        dtype = getattr(torch, query["output_dtype"])
        expected = ((source.float() @ weight.float().T) * scale).to(dtype)
        return lambda: (tensor_fp8_linear.mm(source, packed, out_dtype=dtype, expected_m=m),), (expected,), False
    if component == "gemm.bmm":
        from b12x import gemm

        b, k, n = (query[name] for name in ("batch", "in_features", "out_features"))
        source = random((b, m, k))
        packed, decoded = _mxfp8_weight(b, k, n, query["b_major"], device, rng)
        output = torch.empty((b, m, n), device=device, dtype=torch.bfloat16)
        expected = torch.bmm(source.float(), decoded.float()).to(output.dtype)

        def run():
            gemm.bmm(source, packed, output, a_dtype="bfloat16", b_dtype="float8_e4m3fn",
                     sf_dtype="float8_e8m0fnu", c_dtype="bfloat16", sf_vec_size=32,
                     b_major=query["b_major"], sf_axis=query["b_major"])
            return (output,)

        return run, (expected,), False
    if component == "gemm.mla_query_projection":
        from b12x.gemm import mla_query_projection

        h = query["heads"]
        source = random((h, m, 192))
        rope = random((m, h, 576))[..., 512:]
        if query["weight_format"] == "mxfp8":
            weight, decoded = _mxfp8_weight(h, 192, 512, "n", device, rng)
        else:
            weight = decoded = random((h, 192, 512))
        output = torch.empty((m, h, 576), device=device, dtype=getattr(torch, query["output_dtype"]))
        scale = torch.tensor([0.037], device=device) if output.dtype == torch.float8_e4m3fn else None
        projected = torch.bmm(source.float(), decoded.float()).to(torch.bfloat16).transpose(0, 1)
        expected = torch.cat((projected, rope), dim=-1)
        if scale is not None:
            expected = (expected.float() / scale).clamp(-448, 448).to(output.dtype)

        def run():
            mla_query_projection.run(source, weight, rope, output, q_scale=scale)
            return (output,)

        return run, (expected,), False
    if component == "quantization.mxfp8":
        from b12x.quantization import mxfp8
        from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch

        source = random((m, query["columns"]), getattr(torch, query["dtype"]))
        ref = quantize_mxfp8_rows_torch(source)
        expected_values = ref.values
        if query["value_order"] == "trellis_native_mma":
            # Native MMA assigns eight four-byte words within each K32 scale group.
            order = (0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15,
                     20, 21, 28, 29, 16, 17, 24, 25, 22, 23, 30, 31, 18, 19, 26, 27)
            index = torch.tensor(order, device=device)
            expected_values = ref.values.view(torch.uint8).reshape(m, -1, 32)[..., index].reshape_as(ref.values)
        values = torch.empty_like(ref.values)
        scales = torch.empty_like(ref.scale_rows)
        mma = torch.empty_like(ref.scale_mma)
        mma.view(torch.uint8).fill_(127)

        def run():
            mxfp8.quantize_rows(source, values, scales, mma, value_order=query["value_order"])
            return values, scales, mma

        return run, (expected_values, ref.scale_rows, ref.scale_mma), True
    raise ValueError(f"no production fixture for {component}")


class _Session(AbstractContextManager):
    def __init__(self, component, context):
        self.component = component
        self.context = context

    def __exit__(self, *_exc):
        import torch

        gc.collect()
        torch.cuda.synchronize(self.context.device_ordinal)
        torch.cuda.empty_cache()

    def candidates(self, case):
        if self.component == "gemm.bf16_gemv":
            from b12x.gemm.bf16_gemv._tuning import GemvQuery, heuristic

            return (SweepCandidate.create(asdict(heuristic(GemvQuery(**case.query), self.context.device))),)
        if self.component == "gemm.mla_query_projection" and case.query["weight_format"] == "bf16":
            return (SweepCandidate.create({"backend": "triton"}),)
        return _CANDIDATES

    def measure(self, case, candidates):
        import torch

        if candidates != self.candidates(case):
            raise ValueError("qualification provider received an unknown execution config")
        settings = self.context.settings
        device = torch.device("cuda", self.context.device_ordinal)
        rng = torch.Generator(device=device).manual_seed(settings.seed + int(case.case_id[-8:], 16))
        with torch.cuda.device(device):
            run, expected, exact = _prepare(self.component, case.query, device, rng)
            prepared = self.prepare_candidate(candidates[0], run, expected, exact=exact)
            return measure_prepared_candidates((prepared,), settings=settings, device=device,
                flush=_l2_flush_fn(device, enabled=settings.cold_l2))

    def prepare_candidate(self, candidate, run, expected, *, exact=False):
        import torch
        import torch.nn.functional as F

        settings = self.context.settings
        device = torch.device("cuda", self.context.device_ordinal)
        with torch.cuda.device(device):
            for _ in range(settings.warmup):
                outputs = run()
            torch.cuda.synchronize(device)
            graph, outputs = capture_warmed_graph(run, device=device)
            pointers = tuple(out.data_ptr() for out in outputs)
            correct = True
            cosines = []
            errors = []
            for _ in range(3):
                for index, output in enumerate(outputs):
                    if exact:
                        # Scale-MMA padding is initialized by the caller and is not a kernel output.
                        if index < 2:
                            output.view(torch.uint8).zero_()
                    else:
                        output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                for output, reference in zip(outputs, expected, strict=True):
                    if exact:
                        correct &= bool(torch.equal(output.view(torch.uint8), reference.view(torch.uint8)))
                    else:
                        actual, ref = output.float(), reference.float()
                        cosine = float(F.cosine_similarity(actual.reshape(1, -1), ref.reshape(1, -1)).item())
                        cosines.append(cosine)
                        errors.append(float((actual - ref).abs().max().item()))
                        correct &= bool(torch.isfinite(actual).all().item()) and cosine >= settings.minimum_cosine
                        tolerance = 0.13 if output.dtype == torch.float8_e4m3fn else 0.02
                        correct &= bool(torch.allclose(actual, ref, rtol=tolerance, atol=0.01))
                    correct &= bool(torch.count_nonzero(output.float()).item())
            stable = tuple(out.data_ptr() for out in outputs) == pointers
            return PreparedCandidate(
                candidate=candidate, graph=graph, correct=bool(correct and stable), owners=(run, outputs, expected),
                metrics={"minimum_cosine": min(cosines) if cosines else None,
                         "maximum_absolute_error": max(errors) if errors else 0,
                         "bitwise_reference": exact, "stable_addresses": stable,
                         "poison_replays": 3, "frozen_resolution_capture": True},
            )


class _ApiGenerator(DiscreteSweepGenerator):
    module: str
    query_name: str
    default_queries: tuple
    artifact_kind = "qualification"
    measurement_kind = "fixed_backend_probe"
    candidate_contract_version = 1

    def __init__(self):
        module = importlib.import_module(self.module + "._tuning")
        self.query_type = getattr(module, self.query_name)
        self.validate_query = module.validate_query
        self.problem = module.TUNING_PROBLEM
        policy = module.EXECUTION_CONTRACT
        super().__init__(component_id=policy.component_id, query_schema_version=policy.query_schema_version,
                         config_schema_version=policy.config_schema_version,
                         query_fields=tuple(field.name for field in fields(self.query_type)), range_fields=frozenset(),
                         cases=self.cases_for_tuning_queries(self.default_queries),
                         benchmark_factory=lambda _group, _cases, context: _Session(policy.component_id, context),
                         coverage={"scope": "measured production API cases", "artifact_kind": "qualification"},
                         candidate_contract_version=self.candidate_contract_version)

    def cases_for_tuning_queries(self, queries):
        result = []
        for query in queries:
            typed = self.query_type(**query)
            self.problem.canonical_inputs(typed)
            self.validate_query(typed)
            normalized = asdict(typed)
            digest = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:16]
            result.append(SweepCase.create(group_id=digest, query=normalized))
        return tuple(result)

    def estimate(self, context):
        estimate = super().estimate(context)
        return replace(estimate, description="production API references and CUDA graph qualification",
                       dimensions={**estimate.dimensions, "artifact_kind": self.artifact_kind,
                                   "candidate_measurements": len(self._cases)})


class Bf16GemvGenerator(_ApiGenerator):
    module = "b12x.gemm.bf16_gemv"
    query_name = "GemvQuery"
    candidate_contract_version = 2
    default_queries = tuple(dict(dtype="bfloat16", max_rows=m, in_features=k, out_features=n)
                            for m in (1, 3, 8) for n, k in ((1, 5120), (112, 1024), (128, 2048)))


class BmmGenerator(_ApiGenerator):
    module = "b12x.gemm._bmm"
    query_name = "BmmQuery"
    default_queries = tuple(dict(batch=b, max_rows=m, in_features=k, out_features=n, b_major=major)
                            for b in (8, 16) for m in (1, 7, 17, 32)
                            for major, k, n in (("n", 192, 512), ("k", 512, 256)))


class MlaQueryProjectionGenerator(_ApiGenerator):
    module = "b12x.gemm.mla_query_projection"
    query_name = "ProjectionQuery"
    candidate_contract_version = 2
    default_queries = tuple(dict(heads=h, max_rows=m, weight_format=weight, output_dtype=dtype)
                            for weight in ("bf16", "mxfp8")
                            for h in ((8, 11, 16) if weight == "bf16" else (8, 16))
                            for m in (1, 7, 17, 32) for dtype in ("bfloat16", "float8_e4m3fn"))


class TensorFp8LinearGenerator(_ApiGenerator):
    module = "b12x.gemm.tensor_fp8_linear"
    query_name = "TensorFp8Query"
    default_queries = tuple(dict(max_rows=m, in_features=k, out_features=n, output_dtype=dtype)
                            for m, n, k in ((3, 132, 128), (8, 128, 128), (17, 64, 160))
                            for dtype in ("bfloat16", "float16"))


class Mxfp8QuantizationGenerator(_ApiGenerator):
    module = "b12x.quantization.mxfp8"
    query_name = "Mxfp8Query"
    default_queries = tuple(dict(max_rows=m, columns=k, dtype=dtype, value_order=order)
                            for m, k in ((1, 128), (7, 256), (9, 384), (129, 1024))
                            for dtype in ("bfloat16", "float16") for order in ("linear", "trellis_native_mma"))
