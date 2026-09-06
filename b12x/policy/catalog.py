"""Ownership of runtime GPU policies and offline API qualification providers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .components import (
    BF16_VOCAB_PROJECTION,
    BLOCKSCALED_PRECISION,
    BLOCK_FP8_LINEAR,
    COMPRESSED_SPARSE_MLA_ATTENTION,
    DSA_INDEXER,
    EP_MOE,
    GDN_ATTENTION,
    GQA_ATTENTION,
    HYPERCONNECTION,
    KDA_PREFILL,
    MHC,
    MLA_ATTENTION,
    MOE_DECODE,
    MTP_FEEDBACK,
    NVFP4_QUANTIZATION,
    PLE,
    PLE_EMBEDDING,
    PLE_HASH,
    QSA_ATTENTION,
    SPARSE_MLA_ATTENTION,
    VARLEN_ATTENTION,
    WO_PROJECTION,
)

if TYPE_CHECKING:
    from .context import ComponentPolicy
    from .generation.contracts import ComponentGenerator


class PlanningPolicyMode(str, Enum):
    """Runtime selection, offline qualification, or component-local planning."""

    LOCAL = "local"
    PROFILED = "profiled"
    QUALIFICATION = "qualification"


@dataclass(frozen=True, kw_only=True)
class PlanningComponentRegistration:
    """An API's typed execution contract and optional measurement provider."""

    op_qualname: str
    mode: PlanningPolicyMode
    component_id: str | None = None
    policy_ref: str | None = None
    generator_ref: str | None = None

    def __post_init__(self) -> None:
        references = (self.component_id, self.policy_ref, self.generator_ref)
        if not self.op_qualname or "." not in self.op_qualname:
            raise ValueError("planned op qualname must use '<group>.<op>'")
        if self.mode in (PlanningPolicyMode.PROFILED, PlanningPolicyMode.QUALIFICATION):
            if any(value is None for value in references):
                raise ValueError(
                    "profiled components require an ID, policy, and generator"
                )
        elif any(value is not None for value in references):
            raise ValueError("local planners cannot register profile providers")

    @staticmethod
    def _load(reference: str) -> Any:
        module_name, separator, attribute = reference.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(f"invalid provider reference {reference!r}")
        return getattr(importlib.import_module(module_name), attribute)

    def load_policy(self) -> ComponentPolicy[Any, Any]:
        """Load and validate the runtime policy owned by this component."""

        if self.policy_ref is None or self.component_id is None:
            raise LookupError(f"{self.op_qualname} has no device-profile policy")
        from .context import ComponentPolicy

        policy = self._load(self.policy_ref)
        if not isinstance(policy, ComponentPolicy):
            raise TypeError(f"{self.policy_ref} did not resolve to ComponentPolicy")
        if policy.component_id != self.component_id:
            raise ValueError(
                f"{self.policy_ref} owns {policy.component_id!r}, expected "
                f"{self.component_id!r}"
            )
        return policy

    def create_generator(self) -> ComponentGenerator:
        """Construct and validate this component's offline generator."""

        if self.generator_ref is None or self.component_id is None:
            raise LookupError(f"{self.op_qualname} has no profile generator")
        from .generation.contracts import ComponentGenerator

        provider = self._load(self.generator_ref)
        generator = provider() if isinstance(provider, type) else provider
        if not isinstance(generator, ComponentGenerator):
            raise TypeError(
                f"{self.generator_ref} did not resolve to ComponentGenerator"
            )
        policy = self.load_policy()
        contract = (
            generator.component_id,
            generator.query_schema_version,
            generator.config_schema_version,
        )
        expected = (
            self.component_id,
            policy.query_schema_version,
            policy.config_schema_version,
        )
        if contract != expected:
            raise ValueError(
                f"generator contract {contract!r} does not match runtime policy "
                f"{expected!r}"
            )
        generator.problem = self.load_problem()
        from .generation.engine import measurement_program

        generator.measurement_program = measurement_program(generator, generator.problem)
        generator.artifact_kind = "qualification" if self.mode is PlanningPolicyMode.QUALIFICATION else "runtime_profile"
        return generator

    def load_problem(self):
        """Load the component-owned executable definition of its tuning problem."""
        from .problem import TuningProblem

        if self.policy_ref is None:
            raise LookupError(f"{self.op_qualname} has no tuning problem")
        reference = self.policy_ref.split(":", 1)[0] + ":TUNING_PROBLEM"
        problem = self._load(reference)
        if not isinstance(problem, TuningProblem) or problem.policy is not self.load_policy():
            raise ValueError(f"{reference} must describe the registered runtime policy")
        return problem


PLANNING_COMPONENTS = (
    PlanningComponentRegistration(
        op_qualname="attention.compressed_sparse_mla",
        mode=PlanningPolicyMode.PROFILED,
        component_id=COMPRESSED_SPARSE_MLA_ATTENTION,
        policy_ref=(
            "b12x.attention.compressed_sparse_mla._policy:COMPRESSED_SPARSE_MLA_POLICY"
        ),
        generator_ref=(
            "b12x.policy.generation.providers.attention:"
            "CompressedSparseMlaAttentionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="attention.dense_mla",
        mode=PlanningPolicyMode.PROFILED,
        component_id=MLA_ATTENTION,
        policy_ref="b12x.attention.dense_mla._policy:DENSE_MLA_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.attention:MlaAttentionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="attention.dsa_indexer",
        mode=PlanningPolicyMode.PROFILED,
        component_id=DSA_INDEXER,
        policy_ref="b12x.attention.dsa_indexer._policy:DSA_INDEXER_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.tunable:DsaIndexerProfileGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="attention.paged",
        mode=PlanningPolicyMode.PROFILED,
        component_id=GQA_ATTENTION,
        policy_ref="b12x.attention.paged._policy:GQA_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.attention:GqaAttentionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="attention.qsa",
        mode=PlanningPolicyMode.PROFILED,
        component_id=QSA_ATTENTION,
        policy_ref="b12x.attention.qsa._policy:QSA_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.attention:QsaAttentionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="attention.sparse_mla",
        mode=PlanningPolicyMode.PROFILED,
        component_id=SPARSE_MLA_ATTENTION,
        policy_ref="b12x.attention.sparse_mla._policy:SPARSE_MLA_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.qualification:SparseMlaGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="attention.varlen",
        mode=PlanningPolicyMode.PROFILED,
        component_id=VARLEN_ATTENTION,
        policy_ref="b12x.attention.varlen._policy:VARLEN_ATTENTION_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.tunable:VarlenAttentionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="gemm.bf16_vocab_projection",
        mode=PlanningPolicyMode.PROFILED,
        component_id=BF16_VOCAB_PROJECTION,
        policy_ref=(
            "b12x.gemm.bf16_vocab_projection._policy:"
            "BF16_VOCAB_PROJECTION_POLICY"
        ),
        generator_ref=(
            "b12x.policy.generation.providers.gemm:"
            "Bf16VocabProjectionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="gemm.block_fp8_linear",
        mode=PlanningPolicyMode.PROFILED,
        component_id=BLOCK_FP8_LINEAR,
        policy_ref=("b12x.gemm.block_fp8_linear._policy:BLOCK_FP8_LINEAR_POLICY"),
        generator_ref=(
            "b12x.policy.generation.providers.gemm:BlockFp8LinearGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="gemm.wo_projection",
        mode=PlanningPolicyMode.PROFILED,
        component_id=WO_PROJECTION,
        policy_ref="b12x.gemm.wo_projection._policy:WO_PROJECTION_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.gemm:WoProjectionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="moe.fused_moe",
        mode=PlanningPolicyMode.PROFILED,
        component_id=MOE_DECODE,
        policy_ref="b12x.moe.fused_moe._policy:MOE_DECODE_POLICY",
        generator_ref="b12x.policy.generation.providers.moe:MoeDecodeGenerator",
    ),
    PlanningComponentRegistration(
        op_qualname="moe.ep_moe",
        mode=PlanningPolicyMode.PROFILED,
        component_id=EP_MOE,
        policy_ref="b12x.moe.ep_moe._policy:EP_MOE_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.qualification:EpMoeGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="norm.hyperconnection",
        mode=PlanningPolicyMode.PROFILED,
        component_id=HYPERCONNECTION,
        policy_ref=("b12x.norm.hyperconnection._policy:HYPERCONNECTION_POLICY"),
        generator_ref=(
            "b12x.policy.generation.providers.norm_sequence:"
            "HyperConnectionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="norm.mhc",
        mode=PlanningPolicyMode.PROFILED,
        component_id=MHC,
        policy_ref="b12x.norm.mhc._policy:MHC_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.norm_sequence:MhcGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="quantization.nvfp4",
        mode=PlanningPolicyMode.PROFILED,
        component_id=NVFP4_QUANTIZATION,
        policy_ref=("b12x.quantization.nvfp4._policy:NVFP4_QUANTIZATION_POLICY"),
        generator_ref=(
            "b12x.policy.generation.providers.tunable:"
            "Nvfp4QuantizationGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="sequence.gdn_decode",
        mode=PlanningPolicyMode.PROFILED,
        component_id=GDN_ATTENTION,
        policy_ref="b12x.sequence.gdn_decode._policy:GDN_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.attention:GdnAttentionGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="sequence.kda_prefill",
        mode=PlanningPolicyMode.PROFILED,
        component_id=KDA_PREFILL,
        policy_ref="b12x.sequence.kda_prefill._policy:KDA_PREFILL_POLICY",
        generator_ref="b12x.policy.generation.providers.kda:KdaPrefillGenerator",
    ),
    PlanningComponentRegistration(
        op_qualname="sequence.mtp_feedback",
        mode=PlanningPolicyMode.PROFILED,
        component_id=MTP_FEEDBACK,
        policy_ref=("b12x.sequence.mtp_feedback._policy:MTP_FEEDBACK_POLICY"),
        generator_ref=(
            "b12x.policy.generation.providers.norm_sequence:MtpFeedbackGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="sequence.ple",
        mode=PlanningPolicyMode.PROFILED,
        component_id=PLE,
        policy_ref="b12x.sequence.ple._policy:PLE_POLICY",
        generator_ref="b12x.policy.generation.providers.ple:PleGenerator",
    ),
    PlanningComponentRegistration(
        op_qualname="sequence.ple_embedding",
        mode=PlanningPolicyMode.PROFILED,
        component_id=PLE_EMBEDDING,
        policy_ref=("b12x.sequence.ple_embedding._policy:PLE_EMBEDDING_POLICY"),
        generator_ref=(
            "b12x.policy.generation.providers.ple:PleEmbeddingGenerator"
        ),
    ),
    PlanningComponentRegistration(
        op_qualname="sequence.ple_hash",
        mode=PlanningPolicyMode.PROFILED,
        component_id=PLE_HASH,
        policy_ref="b12x.sequence.ple_hash._policy:PLE_HASH_POLICY",
        generator_ref=(
            "b12x.policy.generation.providers.ple:PleHashGenerator"
        ),
    ),
)


# One-shot APIs resolve static geometry before capture. They do not
# acquire a public plan/bind/run surface by participating in offline profiling.
ONESHOT_COMPONENTS = (
    PlanningComponentRegistration(
        op_qualname="gemm.blockscaled",
        mode=PlanningPolicyMode.PROFILED,
        component_id=BLOCKSCALED_PRECISION,
        policy_ref="b12x.gemm.blockscaled._policy:BLOCKSCALED_POLICY",
        generator_ref="b12x.policy.generation.providers.blockscaled:BlockscaledPrecisionGenerator",
    ),
)


QUALIFICATION_COMPONENTS = (*tuple(
    PlanningComponentRegistration(
        op_qualname=op, mode=PlanningPolicyMode.QUALIFICATION, component_id=op,
        policy_ref=f"{module}._tuning:EXECUTION_CONTRACT",
        generator_ref=f"b12x.policy.generation.providers.oneshot:{generator}",
    )
    for op, module, generator in (
        ("gemm.bf16_gemv", "b12x.gemm.bf16_gemv", "Bf16GemvGenerator"),
        ("gemm.bmm", "b12x.gemm._bmm", "BmmGenerator"),
        ("gemm.mla_query_projection", "b12x.gemm.mla_query_projection", "MlaQueryProjectionGenerator"),
        ("gemm.tensor_fp8_linear", "b12x.gemm.tensor_fp8_linear", "TensorFp8LinearGenerator"),
        ("quantization.mxfp8", "b12x.quantization.mxfp8", "Mxfp8QuantizationGenerator"),
    )
), PlanningComponentRegistration(
    op_qualname="gemm.trellis_linear", mode=PlanningPolicyMode.QUALIFICATION,
    component_id="gemm.trellis_linear", policy_ref="b12x.gemm.trellis_linear._tuning:EXECUTION_CONTRACT",
    generator_ref="b12x.policy.generation.providers.trellis:TrellisLinearGenerator",
))


@dataclass(frozen=True, kw_only=True)
class ApiAliasRegistration:
    """Public names that resolve to the same production callables as their owner."""

    op_qualname: str
    owner_op: str
    entry_points: tuple[str, ...]
    recipes: tuple[str, ...]


API_ALIASES = (
    ApiAliasRegistration(op_qualname="gemm.mxfp8_linear", owner_op="gemm.blockscaled",
                         entry_points=("mm", "pack_weight"), recipes=("mxfp8",)),
)


def list_generation_components() -> tuple[PlanningComponentRegistration, ...]:
    """Return runtime profile providers and API-only qualification providers."""
    return tuple(sorted((*list_profiled_components(), *QUALIFICATION_COMPONENTS),
                        key=lambda item: item.component_id))


def _validate_catalog() -> None:
    registrations = (*PLANNING_COMPONENTS, *ONESHOT_COMPONENTS, *QUALIFICATION_COMPONENTS)
    op_qualnames = tuple(item.op_qualname for item in registrations)
    if len(op_qualnames) != len(set(op_qualnames)):
        raise ValueError("planned ops cannot have duplicate policy registrations")
    component_ids = tuple(
        item.component_id
        for item in registrations if item.mode is not PlanningPolicyMode.LOCAL
    )
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("profile component IDs must be unique")


_validate_catalog()


def list_planning_components() -> tuple[PlanningComponentRegistration, ...]:
    """Return every planned op's explicit policy registration."""

    return tuple(sorted(PLANNING_COMPONENTS, key=lambda item: item.op_qualname))


def list_profiled_components() -> tuple[PlanningComponentRegistration, ...]:
    """Return built-in components owned by generated device profiles."""

    return tuple(
        sorted(
            (
                item
                for item in (*PLANNING_COMPONENTS, *ONESHOT_COMPONENTS)
                if item.mode is PlanningPolicyMode.PROFILED
            ),
            key=lambda item: str(item.component_id),
        )
    )


__all__ = [
    "PLANNING_COMPONENTS",
    "ONESHOT_COMPONENTS",
    "PlanningComponentRegistration",
    "PlanningPolicyMode",
    "list_planning_components",
    "list_profiled_components",
    "list_generation_components",
    "QUALIFICATION_COMPONENTS",
    "API_ALIASES",
    "ApiAliasRegistration",
]
