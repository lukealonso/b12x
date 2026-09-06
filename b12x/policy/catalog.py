"""Authoritative registration of planned ops and profiled components."""

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
    """Whether a planned op consults the device-profile policy layer."""

    LOCAL = "local"
    PROFILED = "profiled"


@dataclass(frozen=True, kw_only=True)
class PlanningComponentRegistration:
    """One planned op's policy ownership and optional profile providers."""

    op_qualname: str
    mode: PlanningPolicyMode
    component_id: str | None = None
    policy_ref: str | None = None
    generator_ref: str | None = None

    def __post_init__(self) -> None:
        references = (self.component_id, self.policy_ref, self.generator_ref)
        if not self.op_qualname or "." not in self.op_qualname:
            raise ValueError("planned op qualname must use '<group>.<op>'")
        if self.mode is PlanningPolicyMode.PROFILED:
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
        return generator


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


def _validate_catalog() -> None:
    op_qualnames = tuple(item.op_qualname for item in (*PLANNING_COMPONENTS, *ONESHOT_COMPONENTS))
    if len(op_qualnames) != len(set(op_qualnames)):
        raise ValueError("planned ops cannot have duplicate policy registrations")
    component_ids = tuple(
        item.component_id
        for item in (*PLANNING_COMPONENTS, *ONESHOT_COMPONENTS)
        if item.mode is PlanningPolicyMode.PROFILED
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
]
