"""Build the canonical S1.7 layer-family and claim corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from . import s1_6_corpus as s1_6
from .contract import (
    CONTRACT_PATH,
    REPOSITORY_ROOT,
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from .s1_overlay import apply_s1_overlay, apply_train_form_counts, overlay_pin
from .s1_prompt_banks import (
    claim_paraphrases as bank_claim_paraphrases,
    prompt_bank_manifest,
    wrapper_prompts,
)
from .source_index import check_source_index
from .sources import file_sha256


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-7-corpus-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-7-manifest-v1"
SPLITS = ("train", "validation", "test")

CorpusBuildError = s1_6.CorpusBuildError
LayerFact = s1_6.LayerFact
ModuleFact = s1_6.ModuleFact
split_row_counts = s1_6.split_row_counts

TRACE_BY_BUCKET = {
    "scpen_family": 1,
    "list_filter": 2,
    "coexpression_claims": 3,
    "claim_check": 4,
    "wrong_input_type": 5,
    "ppi_claims": 6,
    "bulk_pen_claims": 7,
    "other_claims": 8,
    "tf_target_claims": 9,
}
BUCKET_BY_TRACE = {value: key for key, value in TRACE_BY_BUCKET.items()}
SEMANTIC_BUCKET_BY_FORM = {
    1: "family_of_layer",
    2: "list_filter",
    3: "claims_card",
    4: "claim_check",
    5: "wrong_input_type",
    6: "claims_card",
    7: "claims_card",
    8: "native_nonmapped_network",
    9: "claims_card",
}
FAMILY_BY_CARD_FORM = {
    1: "scpen",
    3: "coexpression",
    6: "ppi",
    7: "bulk_pen",
    8: "other",
    9: "tf_target",
}
CLAIM_VOCABULARY = (
    "context_predictive_expression",
    "physical_interaction_recorded",
    "regulatory_link_recorded",
    "regulation_direction",
    "global_coexpression",
    "cofunctional_association",
    "holds_in_other_contexts",
    "causal_effect",
)
CLAIM_DEFINITIONS = {
    "context_predictive_expression": "prediction of expression within the supplied setting",
    "physical_interaction_recorded": "a documented contact between proteins",
    "regulatory_link_recorded": "a documented regulatory connection",
    "regulation_direction": "identification of regulator and target roles",
    "global_coexpression": "expression correlation aggregated across datasets",
    "cofunctional_association": "shared function inferred through network integration",
    "holds_in_other_contexts": "extension of the association beyond its context",
    "causal_effect": "one endpoint producing a change in its partner",
}
FAMILY_CARDS = {
    "ppi": {
        "evidence_type": "physical_interaction_recorded",
        "supported_claims": ["physical_interaction_recorded"],
    },
    "coexpression": {
        "evidence_type": "global_coexpression",
        "supported_claims": ["global_coexpression"],
    },
    "tf_target": {
        "evidence_type": "regulatory_link_recorded",
        "supported_claims": ["regulatory_link_recorded"],
    },
    "bulk_pen": {
        "evidence_type": "context_predictive_expression",
        "supported_claims": ["context_predictive_expression"],
    },
    "scpen": {
        "evidence_type": "context_predictive_expression",
        "supported_claims": ["context_predictive_expression"],
    },
    "other": {"evidence_type": None, "supported_claims": []},
}
FAMILIES = tuple(FAMILY_CARDS)
MASK_TOKEN = "<SOURCE_VALUE>"
MASKED_WORDING_TOLERANCE = 0.05


FAMILY_PROMPTS = {
    "train": {
        "direct": (
            "Answer this layer question.\n{request}",
            "Review this layer question.\n{request}",
            "Give the exact layer answer.\n{request}",
            "Resolve this layer request.\n{request}",
        ),
        "step": (
            "Prepare this answer for the next step.\n{request}",
            "Check this before the workflow continues.\n{request}",
            "The next step needs this layer answer.\n{request}",
            "Resolve this before the next action.\n{request}",
        ),
    },
    "validation": {
        "direct": (
            "Supply the exact layer answer.\n{request}",
            "Read and answer this layer question.\n{request}",
            "Return the requested layer result.\n{request}",
            "Evaluate this layer request.\n{request}",
        ),
        "step": (
            "Answer this before analysis continues.\n{request}",
            "Prepare this result for later work.\n{request}",
            "The workflow needs this layer answer.\n{request}",
            "Check this before work advances.\n{request}",
        ),
    },
    "test": {
        "direct": (
            "Find the exact layer answer.\n{request}",
            "Give the result for this layer question.\n{request}",
            "Read this request and answer it.\n{request}",
            "State the requested layer result.\n{request}",
        ),
        "step": (
            "Resolve this before analysis advances.\n{request}",
            "This result is needed for the next action.\n{request}",
            "Prepare this answer for later analysis.\n{request}",
            "The workflow first needs this result.\n{request}",
        ),
    },
}


CLAIM_PARAPHRASES = {
    "train": {
        "context_predictive_expression": (
            "predictive expression in the named context",
            "a predictive expression association for this context",
            "expression predictability within the named context",
            "a predictive expression relation in the given context",
            "predictive expression scoped to the given context",
            "an expression association predicted in this context",
            "expression predictability for the given context",
            "a predictive expression link in the named context",
        ),
        "physical_interaction_recorded": (
            "a source-recorded physical interaction",
            "a recorded physical interaction from the source",
            "the source recording a physical interaction",
            "a recorded physical interaction between the endpoints",
            "a physical interaction recorded for this edge",
            "the source listing a recorded physical interaction",
            "a recorded physical-interaction link for the endpoint pair",
            "a database-recorded physical interaction for the endpoints",
        ),
        "regulatory_link_recorded": (
            "a source-recorded regulatory link",
            "a recorded regulatory link from the source",
            "the source recording a regulatory relation",
            "a recorded regulatory link between the endpoints",
            "a regulatory link recorded for this edge",
            "the source listing a recorded regulatory link",
            "a recorded regulatory-link entry for the endpoint pair",
            "a source-recorded regulatory link for the endpoints",
        ),
        "regulation_direction": (
            "the identity of the endpoint that regulates the other",
            "the regulator-to-target direction",
            "the direction from regulator to target",
            "the identity of the regulator endpoint",
            "the identity of the target endpoint",
            "the orientation of the regulatory link",
            "the regulator and target roles",
            "the directed order of regulator and target",
        ),
        "global_coexpression": (
            "coexpression aggregated across many datasets",
            "a global expression correlation",
            "cross-dataset coexpression",
            "an aggregate coexpression relation",
            "coexpression pooled across datasets",
            "a dataset-spanning expression correlation",
            "global correlation of expression profiles",
            "an expression correlation aggregated across studies",
        ),
        "cofunctional_association": (
            "a cofunctional link from an integrated network",
            "shared function derived through network integration",
            "an integrated functional association",
            "a cofunctional relation from combined evidence",
            "shared biological function from an integrated network",
            "a functionally associated endpoint pair",
            "cofunction inferred from multiple evidence types",
            "an integrated-network link for shared function",
        ),
        "holds_in_other_contexts": (
            "the same relation holding in other contexts",
            "the association persisting across other contexts",
            "the relation also applying outside this context",
            "the same link occurring in another context",
            "the association generalizing to other contexts",
            "the relation remaining valid across contexts",
            "the same association appearing elsewhere",
            "the link holding beyond the named context",
        ),
        "causal_effect": (
            "a causal effect between the endpoints",
            "a causal consequence for one endpoint",
            "causation between the linked endpoints",
            "an endpoint causing a change in the other",
            "a causal relationship for the endpoint pair",
            "one endpoint causing an effect in the other",
            "a cause-and-effect link between the endpoints",
            "a causal outcome at the other endpoint",
        ),
    },
    "validation": {
        "context_predictive_expression": (
            "expression predictability for this exact context",
            "a context-linked predictive expression association",
            "predictive expression within this exact setting",
            "an expression relation predicted for the supplied context",
            "expression predictability scoped to the supplied context",
            "a predictive expression connection tied to this setting",
            "a predictive expression association within the supplied context",
            "a predicted expression link for this setting",
        ),
        "physical_interaction_recorded": (
            "a recorded physical interaction for this layer",
            "the layer recording a physical interaction",
            "a recorded physical interaction entry for these endpoints",
            "a physical interaction in the layer record",
            "a documented recorded physical interaction",
            "a recorded physical interaction for this endpoint pair",
            "the layer marking a recorded physical interaction",
            "a layer record of a physical interaction",
        ),
        "regulatory_link_recorded": (
            "a recorded regulatory link for this edge",
            "the recording of a regulatory link",
            "a recorded regulatory-link entry for these endpoints",
            "a regulatory link in the layer record",
            "a recorded link of regulation",
            "the endpoint pair having a recorded regulatory link",
            "a regulatory relationship recorded for the endpoints",
            "the layer record containing a regulatory link",
        ),
        "regulation_direction": (
            "the orientation from regulator toward target",
            "the identity of the upstream regulator",
            "the identity of the downstream target",
            "the ordered regulatory endpoint roles",
            "the direction assigned to the regulatory relation",
            "the identity of the upstream endpoint",
            "the identity of the downstream endpoint",
            "the arrow direction on the regulatory connection",
        ),
        "global_coexpression": (
            "an expression correlation pooled across datasets",
            "study-spanning coexpression",
            "an aggregate correlation of expression",
            "coexpression combined from many studies",
            "a global correlation between expression profiles",
            "expression correlation across a dataset collection",
            "a pooled coexpression association",
            "a global-scale expression correlation",
        ),
        "cofunctional_association": (
            "integrated-network evidence of shared function",
            "a combined-network functional link",
            "functional relatedness inferred through integration",
            "a shared-function association from mixed evidence",
            "cofunction assigned by an integrated resource",
            "a functional link built from several sources",
            "shared biological purpose inferred by integration",
            "an integrated association of gene function",
        ),
        "holds_in_other_contexts": (
            "this relation also holding across other contexts",
            "the link recurring outside this context",
            "the association extending into another context",
            "the same relationship applying across contexts",
            "the relation continuing beyond this context",
            "the association transferring to other settings",
            "the same link remaining across contexts",
            "the relation occurring in additional contexts",
        ),
        "causal_effect": (
            "a causal consequence of one endpoint",
            "causation in the linked relationship",
            "one endpoint producing a change in the other",
            "a causal outcome from the endpoint relation",
            "a cause producing an effect",
            "a causal relationship between both endpoints",
            "one endpoint changing its partner causally",
            "a causal consequence within the association",
        ),
    },
    "test": {
        "context_predictive_expression": (
            "a predictive expression link in this context",
            "expression prediction for the specified setting",
            "a supplied-context predictive expression association",
            "predictability of expression inside this context",
            "an expression link predicted for this context",
            "a predictive expression relation tied to the specified context",
            "expression predictability scoped to this context",
            "an expression prediction link for this setting",
        ),
        "physical_interaction_recorded": (
            "a recorded physical interaction on the supplied edge",
            "a physical interaction recorded for the supplied pair",
            "the supplied edge recording a physical interaction",
            "a recorded physical-interaction entry for both endpoints",
            "an edge entry for a recorded physical interaction",
            "a recorded physical interaction listed for the endpoint pair",
            "a recorded physical interaction stored by the layer",
            "a layer record of endpoint physical interaction",
        ),
        "regulatory_link_recorded": (
            "a recorded regulatory link on the supplied edge",
            "a regulatory link recorded for the supplied pair",
            "the supplied edge recording a regulatory link",
            "a recorded regulatory-link entry for both endpoints",
            "an edge entry for a recorded regulatory link",
            "a recorded regulatory link listed for the endpoint pair",
            "a recorded regulatory link stored by the layer",
            "a layer record of endpoint regulation",
        ),
        "regulation_direction": (
            "the directed path from regulator toward target",
            "the endpoint that occupies the regulator role",
            "the endpoint that occupies the target role",
            "the ordering of regulator and target endpoints",
            "the direction carried by the regulatory edge",
            "the upstream-to-downstream endpoint order",
            "the source and destination of regulation",
            "the arrow orientation for the regulatory link",
        ),
        "global_coexpression": (
            "a cross-dataset expression correlation",
            "coexpression summarized over many studies",
            "a broad aggregate of expression correlation",
            "an expression correlation pooled from studies",
            "a global correlation between expression patterns",
            "an expression correlation across source datasets",
            "a multi-study coexpression link",
            "an aggregated expression-profile correlation",
        ),
        "cofunctional_association": (
            "shared function inferred by network integration",
            "a functional association from combined networks",
            "cofunction derived from integrated sources",
            "an endpoint pair linked by shared function",
            "a multi-source functional relationship",
            "functional similarity assigned after integration",
            "a combined-evidence link in biological function",
            "an integrated functional-network association",
        ),
        "holds_in_other_contexts": (
            "the relation persisting outside the named context",
            "the association recurring in a different setting",
            "the same link applying beyond this context",
            "the relationship remaining in other settings",
            "the association carrying into another context",
            "the link generalizing beyond the specified context",
            "the same relation appearing in other contexts",
            "the relationship holding elsewhere as well",
        ),
        "causal_effect": (
            "a causal outcome from the endpoint link",
            "a cause-and-effect relation in the association",
            "one endpoint causing change at the other endpoint",
            "causality for the linked pair",
            "one endpoint causally changing the other",
            "a cause producing an outcome at its partner",
            "a causal relation between the endpoints",
            "a causal response at the linked endpoint",
        ),
    },
}

CARD_REQUEST_VARIANTS = tuple(
    f"{verb} {scope}."
    for verb in (
        "Return", "Report", "Describe", "Give", "Identify", "Summarize", "Provide", "Present",
    )
    for scope in (
        "all claims supported or unsupported by {subject}",
        "the complete claim card for {subject}",
        "the exact evidence and claim limits for {subject}",
        "what {subject} can and cannot support",
        "the evidence type and claim limits for {subject}",
        "the ordered claim partition for {subject}",
        "every supported and unsupported claim for {subject}",
        "the exact layer-family record for {subject}",
    )
)
FAMILY_REQUEST_VARIANTS = tuple(
    f"{verb} the layer family and claim card for layer {{layer_id}}."
    for verb in (
        "Give", "State", "Return", "Report", "Identify", "Describe", "Find", "Supply",
    )
)
CLAIM_CHECK_VARIANTS = (
    "Can {subject} support this claim: {phrase}?",
    "Does {subject} support this claim: {phrase}?",
    "Is this claim supported by {subject}: {phrase}?",
    "Check whether {subject} supports {phrase}.",
    "Decide whether {subject} supports {phrase}.",
    "Determine whether {subject} supports {phrase}.",
    "Report whether {subject} supports {phrase}.",
    "Evaluate whether {subject} supports {phrase}.",
    "State whether {subject} supports {phrase}.",
    "Assess whether {subject} supports {phrase}.",
    "Establish whether {subject} supports {phrase}.",
    "Judge whether {subject} supports {phrase}.",
    "Explain whether {subject} supports {phrase}.",
    "Specify whether {subject} supports {phrase}.",
    "Verify whether {subject} supports {phrase}.",
    "Review whether {subject} supports {phrase}.",
    "Test whether {subject} supports {phrase}.",
    "Find whether {subject} supports {phrase}.",
    "Resolve whether {subject} supports {phrase}.",
    "Say whether {subject} supports {phrase}.",
    "Indicate whether {subject} supports {phrase}.",
    "Clarify whether {subject} supports {phrase}.",
    "Answer whether {subject} supports {phrase}.",
    "Consider whether {subject} supports {phrase}.",
)
LIST_REQUEST_VARIANTS = (
    "Which listed layers are in the {family} family? {layer_ids}",
    "Select the {family} layers from this list: {layer_ids}.",
    "Return the listed {family} layers: {layer_ids}.",
    "Filter this list for the {family} family: {layer_ids}.",
    "Identify every {family} layer here: {layer_ids}.",
    "Find the {family} entries in this layer list: {layer_ids}.",
    "Report which listed layers have family {family}: {layer_ids}.",
    "Give all {family} layers in this list: {layer_ids}.",
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """Store one selected S1.7 request before rendering."""

    split: str
    form: int
    family: str
    fact: LayerFact | ModuleFact | None
    companions: tuple[LayerFact, ...] = ()
    claim: str | None = None
    claim_supported: bool | None = None
    list_outcome: str | None = None
    input_kind: str = "layer_tag"
    framing: str = "direct"
    template_index: int = 0
    occurrence: int = 0
    request_variant: int = 0
    claim_paraphrase_index: int = 0
    # Rendering changes are pinned by the bank and overlay, not case IDs.
    claim_phrase: str | None = None
    train_reasoning: bool = False
    train_targeted_v9: bool = False

    @property
    def source_trace(self) -> str:
        return f"S1.7.{self.form}"

    @property
    def semantic_bucket(self) -> str:
        return SEMANTIC_BUCKET_BY_FORM[self.form]

    @property
    def split_units(self) -> tuple[str, ...]:
        if isinstance(self.fact, ModuleFact):
            return (self.fact.split_unit,)
        units = {
            item.context_key
            for item in (
                *((self.fact,) if isinstance(self.fact, LayerFact) else ()),
                *self.companions,
            )
        }
        return tuple(sorted(units))

    @property
    def metadata_split_unit(self) -> str:
        if not self.split_units:
            return f"shared_family_{self.family}"
        if len(self.split_units) == 1:
            return self.split_units[0]
        return "group_" + _stable_sha256(self.split_units)[:24]

    @property
    def scenario_id(self) -> str:
        return _stable_sha256(
            {
                "split": self.split,
                "form": self.form,
                "family": self.family,
                "fact": None if self.fact is None else self.fact.fact_id,
                "companions": [item.fact_id for item in self.companions],
                "claim": self.claim,
                "claim_supported": self.claim_supported,
                "list_outcome": self.list_outcome,
                "input_kind": self.input_kind,
                "framing": self.framing,
                "template_index": self.template_index,
                "occurrence": self.occurrence,
                "request_variant": self.request_variant,
                "claim_paraphrase_index": self.claim_paraphrase_index,
            }
        )


FactPools = dict[str, dict[str, list[Any]]]


def _canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def trace_counts(total: int, settings: Mapping[str, Any]) -> dict[int, int]:
    """Return the exact S1.7 trace allocation for one split."""

    allocations = settings["allocations"]
    evaluation = settings["evaluation_allocations"]
    if tuple(allocations) != tuple(TRACE_BY_BUCKET):
        raise CorpusBuildError("the S1.7 training allocation fields changed")
    if tuple(evaluation) != tuple(TRACE_BY_BUCKET):
        raise CorpusBuildError("the S1.7 evaluation allocation fields changed")
    train_total = int(settings["train_rows"])
    evaluation_total = sum(int(value) for value in evaluation.values())
    if total == train_total:
        selected = allocations
    elif total == evaluation_total:
        selected = evaluation
    else:
        selected = s1_6.largest_remainder_counts(
            total,
            {
                name: Fraction(int(value), train_total)
                for name, value in allocations.items()
            },
        )
    counts = {
        TRACE_BY_BUCKET[name]: int(selected[name])
        for name in TRACE_BY_BUCKET
    }
    if total == train_total and settings.get("train_form_counts"):
        counts = apply_train_form_counts(counts, settings["train_form_counts"])
    return counts


def family_card(family: str, layer_id: str | None) -> dict[str, Any]:
    """Return the exact ordered family card for one layer or family."""

    if family not in FAMILY_CARDS:
        raise CorpusBuildError(f"unknown layer family {family!r}")
    card = FAMILY_CARDS[family]
    supported = set(card["supported_claims"])
    return {
        "layer_id": layer_id,
        "layer_family": family,
        "evidence_type": card["evidence_type"],
        "supported_claims": [
            claim for claim in CLAIM_VOCABULARY if claim in supported
        ],
        "unsupported_claims": [
            claim for claim in CLAIM_VOCABULARY if claim not in supported
        ],
    }


def build_fact_pools(
    payload: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    seed: int,
    contract: Mapping[str, Any],
) -> FactPools:
    """Reuse S1.6 splits and expose S1.7 family pools."""

    inherited = s1_6.build_fact_pools(
        payload, settings=settings, seed=seed, contract=contract
    )
    pools: FactPools = {}
    for split in SPLITS:
        all_facts = list(inherited[split]["all"])
        pools[split] = {
            "all": all_facts,
            "ppi": [item for item in all_facts if item.layer_family == "ppi"],
            "coexpression": [
                item for item in all_facts if item.layer_family == "coexpression"
            ],
            "tf_target": [
                item for item in all_facts if item.layer_family == "tf_target"
            ],
            "bulk_pen": [
                item for item in all_facts if item.layer_family == "bulk_pen"
            ],
            "scpen": [item for item in all_facts if item.layer_family == "scpen"],
            "other": [item for item in all_facts if item.layer_family == "other"],
            "modules": [
                item
                for item in inherited[split]["modules"]
                if item.namespace == "mentor_ev"
            ],
        }
        for values in pools[split].values():
            values.sort(key=lambda item: item.fact_id)
        for name in ("all", "bulk_pen", "scpen", "other", "modules"):
            if not pools[split][name]:
                raise CorpusBuildError(f"{split} lacks the S1.7 {name} pool")
    for family in ("ppi", "coexpression", "tf_target"):
        if not pools["train"][family]:
            raise CorpusBuildError(f"training lacks the {family} source tag")
    if any(pools[split]["tf_target"] for split in ("validation", "test")):
        raise CorpusBuildError("a held-out split contains the singleton TF tag")
    expected_other = settings["other_network"]["source_counts"]
    if {
        split: len(pools[split]["other"]) for split in SPLITS
    } != {split: int(value) for split, value in expected_other.items()}:
        raise CorpusBuildError("the native other-network source counts changed")
    return pools


def _balanced_labels(counts: Mapping[Any, int]) -> list[Any]:
    """Interleave labels while preserving exact counts."""

    remaining = {name: int(value) for name, value in counts.items()}
    labels: list[Any] = []
    while any(remaining.values()):
        for name in counts:
            if remaining[name] > 0:
                labels.append(name)
                remaining[name] -= 1
    return labels


def _framed_labels(counts: Mapping[Any, int], frames: Sequence[str]) -> list[Any]:
    """Spread exact labels across both frame values."""

    direct = {name: int(value) // 2 for name, value in counts.items()}
    missing = sum(frame == "direct" for frame in frames) - sum(direct.values())
    odd = [name for name, value in counts.items() if int(value) % 2]
    if missing < 0 or missing > len(odd):
        raise CorpusBuildError("a framed label allocation cannot reconcile")
    for name in odd[:missing]:
        direct[name] += 1
    by_frame = {
        "direct": _balanced_labels(direct),
        "step": _balanced_labels(
            {name: int(counts[name]) - direct[name] for name in counts}
        ),
    }
    positions = Counter()
    result = []
    for frame in frames:
        result.append(by_frame[frame][positions[frame]])
        positions[frame] += 1
    return result


def _frame_counts(
    counts: Mapping[int, int], shares: Mapping[str, Any]
) -> dict[int, dict[str, int]]:
    """Allocate the global frame total across traces."""

    expected = s1_6.largest_remainder_counts(sum(counts.values()), shares)
    direct = {form: int(count) // 2 for form, count in counts.items()}
    extra = int(expected["direct"]) - sum(direct.values())
    odd = [form for form, count in counts.items() if int(count) % 2]
    if extra < 0 or extra > len(odd):
        raise CorpusBuildError("the frame allocation cannot reconcile")
    for form in odd[:extra]:
        direct[form] += 1
    return {
        form: {"direct": direct[form], "step": int(count) - direct[form]}
        for form, count in counts.items()
    }


def _settings_mode(split: str) -> str:
    return "training" if split == "train" else "evaluation"


def _card_name_count(
    form: int, split: str, settings: Mapping[str, Any]
) -> int:
    family = FAMILY_BY_CARD_FORM[form]
    if form not in {3, 6, 7, 9}:
        return 0
    key = f"{_settings_mode(split)}_family_name_counts"
    return int(settings["claim_cards"][key][family])


def _claim_check_specs(
    split: str,
    settings: Mapping[str, Any],
    *,
    direct_target: int,
) -> list[tuple[str, str, bool, str, str]]:
    """Return cross-balanced claim-check labels."""

    if split == "train" and settings.get("train_targeted_v9"):
        return _targeted_v9_claim_specs(settings, direct_target=direct_target)

    mode = _settings_mode(split)
    policy = settings["claim_check"]
    family_counts = {
        family: int(value)
        for family, value in policy[f"{mode}_family_counts"].items()
    }
    name_counts = {
        family: int(value)
        for family, value in policy[f"{mode}_family_name_counts"].items()
    }
    supported_counts = {
        family: int(value)
        for family, value in policy[f"{mode}_supported_counts"].items()
    }
    family_direct = {
        family: count // 2 for family, count in family_counts.items()
    }
    family_extra = direct_target - sum(family_direct.values())
    odd_families = [
        family for family, count in family_counts.items() if count % 2
    ]
    if family_extra < 0 or family_extra > len(odd_families):
        raise CorpusBuildError("claim-check family frames cannot reconcile")
    preferred_odd = (
        odd_families
        if split == "train" else [
            family
            for family in ("coexpression", "bulk_pen", "scpen", "tf_target", "other")
            if family in odd_families
        ]
    )
    for family in preferred_odd[:family_extra]:
        family_direct[family] += 1

    cells: dict[tuple[str, str, str, bool], int] = {}
    for family in family_counts:
        supported_claims = list(FAMILY_CARDS[family]["supported_claims"])
        total_by_claim: dict[str, int] = {claim: 0 for claim in CLAIM_VOCABULARY}
        if supported_counts[family]:
            supported_allocation = s1_6.largest_remainder_counts(
                supported_counts[family],
                {
                    claim: Fraction(1, len(supported_claims))
                    for claim in supported_claims
                },
            )
            total_by_claim.update(supported_allocation)
        unsupported_claims = [
            claim for claim in CLAIM_VOCABULARY if claim not in supported_claims
        ]
        unsupported_allocation = s1_6.largest_remainder_counts(
            family_counts[family] - supported_counts[family],
            {
                claim: Fraction(1, len(unsupported_claims))
                for claim in unsupported_claims
            },
        )
        total_by_claim.update(unsupported_allocation)
        family_names = name_counts[family]
        if family_names:
            name_by_claim = s1_6.largest_remainder_counts(
                family_names,
                {
                    claim: Fraction(count, family_counts[family])
                    for claim, count in total_by_claim.items()
                },
            )
        else:
            name_by_claim = {claim: 0 for claim in CLAIM_VOCABULARY}
        for claim, total in total_by_claim.items():
            supported = claim in supported_claims
            cells[(family, claim, "family_name", supported)] = name_by_claim[claim]
            cells[(family, claim, "layer_tag", supported)] = (
                total - name_by_claim[claim]
            )

    direct_by_cell = {cell: count // 2 for cell, count in cells.items()}
    family_order = list(family_counts)
    odd_cells = [cell for cell, count in cells.items() if count % 2]
    family_needs = tuple(
        family_direct[family]
        - sum(
            direct_by_cell[cell] for cell in cells if cell[0] == family
        )
        for family in family_order
    )
    name_total = sum(
        count for cell, count in cells.items() if cell[2] == "family_name"
    )
    name_need = (name_total + 1) // 2 - sum(
        direct_by_cell[cell]
        for cell in cells
        if cell[2] == "family_name"
    )
    enforce_supported_margin = True
    supported_total = sum(count for cell, count in cells.items() if cell[3])
    supported_need = (
        (supported_total + 1) // 2
        - sum(direct_by_cell[cell] for cell in cells if cell[3])
        if enforce_supported_margin else 0
    )

    @lru_cache(maxsize=None)
    def choose_extras(
        index: int,
        family_remaining: tuple[int, ...],
        names_remaining: int,
        supported_remaining: int,
    ) -> tuple[int, ...] | None:
        if index == len(odd_cells):
            if (
                not any(family_remaining)
                and names_remaining == 0
                and supported_remaining == 0
            ):
                return ()
            return None
        cell = odd_cells[index]
        family_index = family_order.index(cell[0])
        options = (1, 0)
        for selected in options:
            next_family = list(family_remaining)
            next_family[family_index] -= selected
            next_names = names_remaining - selected * int(
                cell[2] == "family_name"
            )
            next_supported = supported_remaining - selected * int(
                cell[3] and enforce_supported_margin
            )
            if min(*next_family, next_names, next_supported) < 0:
                continue
            tail = choose_extras(
                index + 1,
                tuple(next_family),
                next_names,
                next_supported,
            )
            if tail is not None:
                return (selected, *tail)
        return None

    choices = choose_extras(0, family_needs, name_need, supported_need)
    if choices is None:
        raise CorpusBuildError("claim-check frame cells cannot reconcile")
    for cell, selected in zip(odd_cells, choices, strict=True):
        direct_by_cell[cell] += selected

    records_by_cell: dict[
        tuple[str, str, str, bool], list[tuple[str, str, bool, str, str]]
    ] = {}
    for cell, count in cells.items():
        if count == 0:
            continue
        family, claim, input_kind, supported = cell
        frames = _balanced_labels(
            {
                "direct": direct_by_cell[cell],
                "step": count - direct_by_cell[cell],
            }
        )
        records_by_cell[cell] = [
            (family, claim, supported, input_kind, frame)
            for frame in frames
        ]
    positions: Counter[tuple[str, str, str, bool]] = Counter()
    records = []
    while len(records) < sum(cells.values()):
        for cell, values in records_by_cell.items():
            index = positions[cell]
            if index < len(values):
                records.append(values[index])
                positions[cell] += 1
    return records


def _targeted_v9_claim_specs(
    settings: Mapping[str, Any], *, direct_target: int,
) -> list[tuple[str, str, bool, str, str]]:
    """Balance canonical claims first, then verdict, scope, family, and input kind."""

    total = sum(int(value) for value in settings["claim_check"]["training_family_counts"].values())
    claim_counts = s1_6.largest_remainder_counts(
        total, {claim: Fraction(1, len(CLAIM_VOCABULARY)) for claim in CLAIM_VOCABULARY},
    )
    cells = {}
    for claim, count in claim_counts.items():
        supporting = [family for family in FAMILIES if claim in FAMILY_CARDS[family]["supported_claims"]]
        positive = (count + 1) // 2 if supporting else 0
        if positive:
            cells[(claim, True)] = positive
        cells[(claim, False)] = count - positive
    direct_counts = {cell: count // 2 for cell, count in cells.items()}
    remaining = direct_target - sum(direct_counts.values())
    odd = [cell for cell, count in cells.items() if count % 2]
    if not 0 <= remaining <= len(odd):
        raise CorpusBuildError("v9 claim scopes cannot reconcile")
    for cell in odd[:remaining]:
        direct_counts[cell] += 1
    records = []
    for (claim, supported), count in cells.items():
        families = [
            family for family in FAMILIES
            if (claim in FAMILY_CARDS[family]["supported_claims"]) == supported
        ]
        frames = _balanced_labels({"direct": direct_counts[(claim, supported)], "step": count - direct_counts[(claim, supported)]})
        # Independent frame cursors expose each applicable family in both scopes.
        positions: Counter[str] = Counter()
        for frame in frames:
            position = positions[frame]
            positions[frame] += 1
            family = families[position % len(families)]
            input_kind = "family_name" if position // len(families) % 5 == 0 else "layer_tag"
            records.append((family, claim, supported, input_kind, frame))
    return records


def _list_specs(
    split: str,
    settings: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    """Return family, outcome, and frame with no label shortcut."""

    mode = _settings_mode(split)
    family_counts = {
        family: int(value)
        for family, value in settings["list_filter"][
            f"{mode}_family_counts"
        ].items()
        if int(value)
    }
    cells: dict[tuple[str, str], int] = {}
    for family, count in family_counts.items():
        match = (count + 1) // 2
        cells[(family, "match")] = match
        cells[(family, "no_match")] = count - match
    records_by_cell: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    family_order = list(family_counts)
    for (family, outcome), count in cells.items():
        family_index = family_order.index(family)
        prefer_direct = (
            family_index % 2 == 0
            if outcome == "match" else family_index % 2 == 1
        )
        direct = count // 2 + int(count % 2 == 1 and prefer_direct)
        frames = _balanced_labels({"direct": direct, "step": count - direct})
        records_by_cell[(family, outcome)] = [
            (family, outcome, frame) for frame in frames
        ]
    positions: Counter[tuple[str, str]] = Counter()
    records = []
    while len(records) < sum(cells.values()):
        for cell, values in records_by_cell.items():
            index = positions[cell]
            if index < len(values):
                records.append(values[index])
                positions[cell] += 1
    return records


def _cycle_fact(
    facts: Sequence[LayerFact | ModuleFact], index: int, *, label: str
) -> LayerFact | ModuleFact:
    if not facts:
        raise CorpusBuildError(f"the {label} source pool is empty")
    return facts[index % len(facts)]


def _list_companions(
    pool: Mapping[str, Sequence[Any]],
    family: str,
    outcome: str,
    occurrence: int,
) -> tuple[LayerFact, ...]:
    """Select three distinct native tags for one family filter."""

    candidates = [item for item in pool["all"] if isinstance(item, LayerFact)]
    matches = [item for item in candidates if item.layer_family == family]
    others = [item for item in candidates if item.layer_family != family]
    if outcome == "match":
        if not matches or len(others) < 2:
            raise CorpusBuildError("a matching list filter is infeasible")
        chosen = [
            matches[occurrence % len(matches)],
            others[(occurrence * 5 + 1) % len(others)],
            others[(occurrence * 7 + 2) % len(others)],
        ]
    elif outcome == "no_match":
        if len(others) < 3:
            raise CorpusBuildError("a no-match list filter is infeasible")
        chosen = [
            others[(occurrence * 3) % len(others)],
            others[(occurrence * 5 + 1) % len(others)],
            others[(occurrence * 7 + 2) % len(others)],
        ]
    else:
        raise CorpusBuildError(f"unknown list outcome {outcome!r}")
    if len({item.fact_id for item in chosen}) != 3:
        for candidate in others:
            if candidate.fact_id not in {item.fact_id for item in chosen}:
                for index in range(2, -1, -1):
                    if chosen[index].fact_id in {
                        item.fact_id for item in chosen[:index]
                    }:
                        chosen[index] = candidate
                        break
            if len({item.fact_id for item in chosen}) == 3:
                break
    if len({item.fact_id for item in chosen}) != 3:
        raise CorpusBuildError("a list filter repeats a native tag")
    shift = occurrence % 3
    return tuple(chosen[shift:] + chosen[:shift])


def _decorrelate_training_wording(
    scenarios: Sequence[Scenario],
    prompts: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[Scenario]:
    """Cross neutral wording with outcomes without changing selected evidence.

    List wrappers and requests use separate content orderings. Claim request
    cycles cover each claim/outcome/framing, with family blocks preserving
    balanced wording within every family that has enough examples.
    """

    revised = list(scenarios)

    def order(index: int, purpose: str) -> str:
        case = scenarios[index]
        return _stable_sha256({
            "purpose": purpose, "form": case.form,
            "family": case.family, "occurrence": case.occurrence,
            "fact": None if case.fact is None else case.fact.fact_id,
            "companions": [item.fact_id for item in case.companions],
            "claim": case.claim, "framing": case.framing,
        })

    lists: dict[tuple[str, str, str | None], list[int]] = defaultdict(list)
    claims: dict[tuple[str | None, str, bool | None], list[int]] = defaultdict(list)
    for index, case in enumerate(scenarios):
        if case.form == 2:
            lists[(case.family, case.framing, case.list_outcome)].append(index)
        elif case.form == 4:
            claims[(case.claim, case.framing, case.claim_supported)].append(index)
    families = sorted({key[0] for key in lists})
    for (family, framing, _), indexes in lists.items():
        width = len(prompts["train"][framing])
        offset = families.index(family) * 5 + int(framing == "step") * 3
        for position, index in enumerate(sorted(indexes, key=lambda i: order(i, "list-wrapper"))):
            revised[index] = replace(
                revised[index], template_index=(position + offset) % width,
            )
        for position, index in enumerate(sorted(indexes, key=lambda i: order(i, "list-request"))):
            revised[index] = replace(
                revised[index],
                request_variant=(position + offset) % len(LIST_REQUEST_VARIANTS),
            )
    for (claim, framing, _), indexes in claims.items():
        width = len(CLAIM_CHECK_VARIANTS)
        offset = CLAIM_VOCABULARY.index(str(claim)) * 5 + (
            width // 2 if framing == "step" else 0
        )
        ordered = sorted(indexes, key=lambda i: (
            scenarios[i].family, order(i, "claim-request"),
        ))
        for position, index in enumerate(ordered):
            revised[index] = replace(
                revised[index], request_variant=(position + offset) % width,
            )
    # Repeated source facts can collide after neutral wording is reassigned.
    # Swap wrappers within the same semantic stratum, retaining every quota.
    # Request selection must remain independent of wrapper-bank size.
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, case in enumerate(revised):
        if case.form in (2, 4):
            groups[(case.form, case.family, case.framing, case.list_outcome,
                    case.claim, case.claim_supported)].append(index)
    texts = [_prompt(case, prompts=prompts) for case in revised]
    uses = Counter(texts)
    for indexes in groups.values():
        for index in indexes:
            if uses[texts[index]] <= 1:
                continue
            for donor in indexes:
                if revised[index].template_index == revised[donor].template_index:
                    continue
                left = replace(revised[index], template_index=revised[donor].template_index)
                right = replace(revised[donor], template_index=revised[index].template_index)
                left_text, right_text = _prompt(left, prompts=prompts), _prompt(right, prompts=prompts)
                remaining = uses.copy()
                remaining.subtract((texts[index], texts[donor]))
                if left_text == right_text or remaining[left_text] or remaining[right_text]:
                    continue
                uses.subtract((texts[index], texts[donor]))
                revised[index], revised[donor] = left, right
                texts[index], texts[donor] = left_text, right_text
                uses.update((left_text, right_text))
                break
            else:
                raise CorpusBuildError("decorrelated training wording repeats a prompt")
    return revised


def select_scenarios(
    pools: Mapping[str, Mapping[str, Sequence[Any]]],
    counts_by_split: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
    framing_shares: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = FAMILY_PROMPTS,
    paraphrases: Mapping[str, Mapping[str, Sequence[str]]] = CLAIM_PARAPHRASES,
) -> dict[str, list[Scenario]]:
    """Select deterministic S1.7 scenarios from split-safe pools."""

    if settings is None:
        settings = load_contract()["curriculum"]["families"]["S1.7"]
    if framing_shares is None:
        framing_shares = {"direct": Fraction(1, 2), "step": Fraction(1, 2)}
    selected: dict[str, list[Scenario]] = {}
    for split in SPLITS:
        counts = counts_by_split[split]
        frames_by_form = _frame_counts(counts, framing_shares)
        scenarios: list[Scenario] = []
        template_positions: Counter[tuple[int, str]] = Counter()
        request_positions: Counter[tuple[Any, ...]] = Counter()
        family_positions: Counter[tuple[int, str]] = Counter()
        claim_specs = _claim_check_specs(
            split,
            settings,
            direct_target=frames_by_form[4]["direct"],
        )
        claim_position = 0
        list_specs = _list_specs(split, settings)
        list_positions: Counter[tuple[str, str]] = Counter()
        paraphrase_positions: Counter[Any] = Counter()
        for form in TRACE_BY_BUCKET.values():
            count = int(counts.get(form, 0))
            frames = _balanced_labels(frames_by_form[form])
            if form == 4 and len(claim_specs) != count:
                raise CorpusBuildError(f"{split} claim-check labels changed")
            card_family = FAMILY_BY_CARD_FORM.get(form)
            name_count = (
                _card_name_count(form, split, settings)
                if card_family is not None else 0
            )
            card_kinds = _framed_labels(
                {
                    "family_name": name_count,
                    "layer_tag": count - name_count,
                },
                frames,
            ) if card_family is not None else []
            for occurrence in range(count):
                framing = frames[occurrence]
                if form == 2:
                    _, _, framing = list_specs[occurrence]
                elif form == 4:
                    _, _, _, _, framing = claim_specs[claim_position]
                template_key = (form, framing)
                template_index = template_positions[template_key] % len(
                    prompts[split][framing]
                )
                template_positions[template_key] += 1
                family = card_family or "other"
                fact: LayerFact | ModuleFact | None = None
                companions: tuple[LayerFact, ...] = ()
                claim = None
                claim_supported = None
                list_outcome = None
                input_kind = "layer_tag"
                claim_paraphrase_index = 0
                if form == 2:
                    family, list_outcome, _ = list_specs[occurrence]
                    input_kind = "layer_list"
                    list_position = list_positions[(family, list_outcome)]
                    list_positions[(family, list_outcome)] += 1
                    companions = _list_companions(
                        pools[split], family, list_outcome, list_position
                    )
                elif form == 4:
                    family, claim, claim_supported, input_kind, _ = claim_specs[claim_position]
                    claim_position += 1
                    paraphrase_key = (
                        (claim, framing, claim_supported)
                        if split == "train" and settings.get("train_targeted_v9") else claim
                    )
                    claim_paraphrase_index = paraphrase_positions[paraphrase_key] % len(paraphrases[split][claim])
                    paraphrase_positions[paraphrase_key] += 1
                    if input_kind == "layer_tag":
                        position = family_positions[(form, family)]
                        fact = _cycle_fact(
                            pools[split][family], position, label=family
                        )
                        family_positions[(form, family)] += 1
                elif form == 5:
                    family = "other"
                    input_kind = "module_id"
                    fact = _cycle_fact(
                        pools[split]["modules"], occurrence, label="mentor_ev"
                    )
                elif card_family is not None:
                    family = card_family
                    input_kind = str(card_kinds[occurrence])
                    if input_kind == "layer_tag":
                        position = family_positions[(form, family)]
                        fact = _cycle_fact(
                            pools[split][family], position, label=family
                        )
                        family_positions[(form, family)] += 1
                request_key = (
                    form,
                    family,
                    input_kind,
                    None if fact is None else fact.fact_id,
                    claim,
                    tuple(item.fact_id for item in companions),
                    framing,
                    # Request variants follow the inline four-wrapper cycle,
                    # so a larger training bank changes only the wrapper.
                    template_index % len(FAMILY_PROMPTS[split][framing]),
                )
                request_variant = request_positions[request_key]
                request_positions[request_key] += 1
                scenario = Scenario(
                    split=split,
                    form=form,
                    family=family,
                    fact=fact,
                    companions=companions,
                    claim=claim,
                    claim_supported=claim_supported,
                    list_outcome=list_outcome,
                    input_kind=input_kind,
                    framing=framing,
                    template_index=template_index,
                    occurrence=occurrence,
                    request_variant=request_variant,
                    claim_paraphrase_index=claim_paraphrase_index,
                )
                if form == 4 and paraphrases is not CLAIM_PARAPHRASES:
                    scenario = replace(
                        scenario,
                        claim_phrase=paraphrases[split][str(claim)][
                            claim_paraphrase_index
                        ],
                    )
                if (
                    split == "train" and form in {2, 4, 8}
                    and settings.get("train_claim_reasoning")
                ):
                    scenario = replace(scenario, train_reasoning=True)
                if split == "train" and form in {2, 4, 8} and settings.get("train_targeted_v9"):
                    scenario = replace(scenario, train_targeted_v9=True)
                scenarios.append(scenario)
        if split == "train" and settings.get("train_decorrelate"):
            scenarios = _decorrelate_training_wording(scenarios, prompts)
        if split == "train" and settings.get("train_targeted_v9"):
            scenarios = _targeted_v9_wording(scenarios, prompts)
        scenarios.sort(
            key=lambda item: _stable_sha256(
                {"seed": seed, "split": split, "scenario": item.scenario_id}
            )
        )
        if len(scenarios) != sum(int(value) for value in counts.values()):
            raise CorpusBuildError(f"{split} scenario count changed")
        selected[split] = scenarios
    return selected


def _targeted_v9_wording(
    scenarios: Sequence[Scenario], prompts: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[Scenario]:
    """Assign neutral wording independently within claim outcomes and native names."""

    revised = list(scenarios)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, case in enumerate(scenarios):
        if case.form == 4:
            groups[(case.form, case.claim, case.claim_supported, case.framing)].append(index)
        elif case.form == 8:
            groups[(case.form, case.input_kind, None if case.fact is None else case.fact.fact_id, case.framing)].append(index)
    for key, indexes in groups.items():
        frame = scenarios[indexes[0]].framing
        for purpose, width, field in (
            ("wrapper", len(prompts["train"][frame]), "template_index"),
            ("request", len(CLAIM_CHECK_VARIANTS) if key[0] == 4 else len(CARD_REQUEST_VARIANTS), "request_variant"),
        ):
            ordered = sorted(indexes, key=lambda index: _stable_sha256([
                "v9", purpose, scenarios[index].occurrence,
                None if scenarios[index].fact is None else scenarios[index].fact.fact_id,
            ]))
            for position, index in enumerate(ordered):
                revised[index] = replace(revised[index], **{field: position % width})
    # Repair collisions through wrapper swaps in the same outcome/scope group.
    # Request, paraphrase, source selection, and all quotas remain fixed.
    texts = [_prompt(case, prompts=prompts) for case in revised]
    uses = Counter(texts)
    for indexes in groups.values():
        for index in indexes:
            if uses[texts[index]] <= 1:
                continue
            for donor in indexes:
                left = replace(revised[index], template_index=revised[donor].template_index)
                right = replace(revised[donor], template_index=revised[index].template_index)
                left_text, right_text = _prompt(left, prompts=prompts), _prompt(right, prompts=prompts)
                remaining = uses.copy()
                remaining.subtract((texts[index], texts[donor]))
                if left_text == right_text or remaining[left_text] or remaining[right_text]:
                    continue
                uses.subtract((texts[index], texts[donor]))
                revised[index], revised[donor] = left, right
                texts[index], texts[donor] = left_text, right_text
                uses.update((left_text, right_text))
                break
            else:
                raise CorpusBuildError("v9 wording repeats a prompt")
    return revised


def _subject(scenario: Scenario) -> str:
    if scenario.input_kind == "family_name":
        return f"an edge in the {scenario.family} family"
    if isinstance(scenario.fact, LayerFact):
        return f"an edge in layer {scenario.fact.layer_id}"
    raise CorpusBuildError("a layer-tag scenario lacks a native layer")


def _request(scenario: Scenario) -> str:
    if scenario.form == 1:
        if not isinstance(scenario.fact, LayerFact):
            raise CorpusBuildError("form 1 lacks an scPEN layer")
        return FAMILY_REQUEST_VARIANTS[
            scenario.request_variant % len(FAMILY_REQUEST_VARIANTS)
        ].format(layer_id=scenario.fact.layer_id)
    if scenario.form == 2:
        if len(scenario.companions) != 3:
            raise CorpusBuildError("form 2 lacks three layer tags")
        layer_ids = ", ".join(item.layer_id for item in scenario.companions)
        template = LIST_REQUEST_VARIANTS[
            scenario.request_variant % len(LIST_REQUEST_VARIANTS)
        ]
        return template.format(family=scenario.family, layer_ids=layer_ids)
    if scenario.form in {3, 6, 7, 8, 9}:
        return CARD_REQUEST_VARIANTS[
            scenario.request_variant % len(CARD_REQUEST_VARIANTS)
        ].format(subject=_subject(scenario))
    if scenario.form == 4:
        if scenario.claim is None:
            raise CorpusBuildError("form 4 lacks a claim")
        phrases = CLAIM_PARAPHRASES[scenario.split][scenario.claim]
        phrase = scenario.claim_phrase or phrases[
            scenario.claim_paraphrase_index % len(phrases)
        ]
        template = CLAIM_CHECK_VARIANTS[
            scenario.request_variant % len(CLAIM_CHECK_VARIANTS)
        ]
        return template.format(subject=_subject(scenario), phrase=phrase)
    if scenario.form == 5:
        if not isinstance(scenario.fact, ModuleFact):
            raise CorpusBuildError("form 5 lacks a module ID")
        return f"Give the layer family for {scenario.fact.module_id}."
    raise CorpusBuildError(f"unsupported S1.7 form {scenario.form}")


def _prompt(
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = FAMILY_PROMPTS,
) -> str:
    templates = prompts[scenario.split][scenario.framing]
    if not 0 <= scenario.template_index < len(templates):
        raise CorpusBuildError("a prompt template index is out of range")
    return templates[scenario.template_index].format(request=_request(scenario))


def _legacy_thinking(scenario: Scenario) -> str:
    """Return the published reasoning for unchanged builders and held-out rows."""

    if scenario.form == 2:
        count = sum(
            item.layer_family == scenario.family for item in scenario.companions
        )
        return (
            f"Each tag maps to one closed family. Exactly {count} listed layers "
            f"map to {scenario.family}."
        )
    if scenario.form == 4:
        verdict = "is" if scenario.claim_supported else "is not"
        return (
            f"The {scenario.family} card has an exact supported list. "
            f"The requested claim {verdict} in that list."
        )
    if scenario.form == 5:
        return (
            "The input is a module ID, not a layer tag. "
            "A module has no layer family."
        )
    supported = list(FAMILY_CARDS[scenario.family]["supported_claims"])
    if scenario.input_kind == "family_name":
        if supported:
            return (
                f"The named family is {scenario.family}. "
                f"Its card supports {supported[0]} only."
            )
        return "The named family is other. Its card supports no listed claim."
    if supported:
        claim = supported[0]
        return (
            f"The input maps to the {scenario.family} family. "
            f"Its card supports {claim} only."
        )
    return (
        "The native network maps to the other family. "
        "That family supports no listed claim."
    )


def _family_thinking(fact: LayerFact) -> str:
    """Name the visible tag and apply the frozen native family mapping."""

    if fact.source == "HumanNetV3":
        return (
            f"Layer {fact.layer_id} names HumanNetV3 network {fact.network}. "
            "The native mapping assigns string_ppi to ppi and coexpression to coexpression. "
            "Every remaining native HumanNetV3 network belongs to other. "
            f"Here the family is {fact.layer_family}."
        )
    prefix = fact.layer_id.split(":", 1)[0]
    return f"The {prefix} prefix in {fact.layer_id} assigns family {fact.layer_family}."


def _thinking(
    scenario: Scenario, *,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> str:
    """Bind reviewed training thoughts while preserving published legacy reasoning."""

    if scenario.split == "train" and scenario.train_targeted_v9:
        return _targeted_v9_thinking(scenario)

    if (
        scenario.split == "train" and scenario.form in {2, 4, 8}
        and final_reasoning and scenario.form in final_reasoning
    ):
        result = _resolved_answer(scenario)["result"]
        if scenario.form == 2:
            bindings = {
                "requested_family": scenario.family,
                "layer_assignments": " ".join(
                    f"Input {index} is {fact.layer_id}, which maps to {fact.layer_family}."
                    for index, fact in enumerate(scenario.companions, start=1)
                ),
                "matching_layer_ids": json.dumps(result["matching_layer_ids"], ensure_ascii=False),
                "other_layer_ids": json.dumps(result["other_layer_ids"], ensure_ascii=False),
            }
        elif scenario.form == 4:
            claim = str(scenario.claim)
            bindings = {
                "claim_phrase": scenario.claim_phrase or CLAIM_PARAPHRASES[scenario.split][claim][
                    scenario.claim_paraphrase_index
                ],
                "claim": claim,
                "claim_definition": CLAIM_DEFINITIONS[claim],
                "family_assignment": (
                    f"The supplied tag {scenario.fact.layer_id} maps to {scenario.family}."
                    if isinstance(scenario.fact, LayerFact)
                    else f"The request names the {scenario.family} family."
                ),
                **{
                    key: json.dumps(result[key], ensure_ascii=False)
                    for key in (
                        "layer_id", "layer_family", "evidence_type", "supported_claims",
                        "unsupported_claims", "claim_supported",
                    )
                },
            }
        else:
            fact = scenario.fact
            if (
                not isinstance(fact, LayerFact) or fact.source != "HumanNetV3"
                or fact.layer_family != "other"
            ):
                raise CorpusBuildError("S1.7.8 reasoning requires an unmapped HumanNet network")
            bindings = {"layer_id": fact.layer_id, "network": fact.network}
        return final_reasoning[scenario.form][scenario.framing].format(**bindings)

    if not scenario.train_reasoning:
        return _legacy_thinking(scenario)
    if scenario.form == 2:
        thoughts = [_family_thinking(fact) for fact in scenario.companions]
        count = sum(
            fact.layer_family == scenario.family for fact in scenario.companions
        )
        thoughts.extend((
            f"I compare each mapped family with {scenario.family}.",
            f"There are {count} matching layers and {3 - count} remaining layers.",
            "I keep every input in one of the two lists, in its original order.",
        ))
        if count == 0:
            thoughts.append("No layer matches, so all three belong in the other list.")
        return " ".join(thoughts)
    thoughts = []
    if scenario.form == 4:
        phrase = scenario.claim_phrase or CLAIM_PARAPHRASES[scenario.split][
            str(scenario.claim)
        ][scenario.claim_paraphrase_index]
        thoughts.extend((
            f'The phrase "{phrase}" names {scenario.claim}.',
            f"This asks for {CLAIM_DEFINITIONS[str(scenario.claim)]}.",
        ))
    if isinstance(scenario.fact, LayerFact):
        thoughts.append(_family_thinking(scenario.fact))
    else:
        thoughts.append(f"The question names the {scenario.family} family.")
    supported = FAMILY_CARDS[scenario.family]["supported_claims"]
    if supported:
        thoughts.append(f"The {scenario.family} card supports {supported[0]} only.")
    else:
        thoughts.append(
            "Its evidence type is null, its supported claims are empty, and all eight claims are unsupported."
        )
    if scenario.form == 4:
        verdict = "supported" if scenario.claim_supported else "unsupported"
        thoughts.append(f"The requested {scenario.claim} claim is {verdict} by this card.")
    return " ".join(thoughts)


def _decision(scenario: Scenario) -> str:
    return "continue" if scenario.framing == "step" else "stop"


def _targeted_v9_family_thinking(fact: LayerFact) -> str:
    """Apply the closed mapping while preserving the exact source network name."""

    if fact.source == "HumanNetV3":
        return (
            f"The supplied layer {fact.layer_id} has HumanNetV3 network {fact.network}. "
            "The exact network names string_ppi and biogrid_ppi map to ppi, and coexpression maps to coexpression. "
            f"Other network names map to other, so this input maps to {fact.layer_family}. "
            "A different network spelling does not inherit the claims of a mapped network."
        )
    return _family_thinking(fact)


def _targeted_v9_thinking(scenario: Scenario) -> str:
    """Normalize the request, classify visible layers, and then derive support and scope."""

    result = _resolved_answer(scenario)["result"]
    thoughts = []
    if scenario.form == 2:
        thoughts.append(f"The requested family is {scenario.family}. Classify all three inputs before filtering.")
        for index, fact in enumerate(scenario.companions, start=1):
            thoughts.append(f"Input {index}: {_targeted_v9_family_thinking(fact)}")
        thoughts.append(
            f"In input order, matching_layer_ids is {json.dumps(result['matching_layer_ids'], ensure_ascii=False)}, "
            f"and other_layer_ids is {json.dumps(result['other_layer_ids'], ensure_ascii=False)}. "
            "Each input belongs to exactly one partition. An empty matching list is a completed filter result."
        )
    else:
        if scenario.form == 4:
            claim = str(scenario.claim)
            phrase = scenario.claim_phrase or CLAIM_PARAPHRASES[scenario.split][claim][scenario.claim_paraphrase_index]
            thoughts.append(
                f"The wording {json.dumps(phrase, ensure_ascii=False)} names canonical claim {claim}, "
                f"meaning {CLAIM_DEFINITIONS[claim]}."
            )
        if isinstance(scenario.fact, LayerFact):
            thoughts.append(_targeted_v9_family_thinking(scenario.fact))
        else:
            thoughts.append(f"The request explicitly names family {scenario.family}.")
        thoughts.append(
            f"For {scenario.family}, evidence_type is {json.dumps(result['evidence_type'], ensure_ascii=False)}, "
            f"supported_claims is {json.dumps(result['supported_claims'], ensure_ascii=False)}, "
            f"and unsupported_claims is {json.dumps(result['unsupported_claims'], ensure_ascii=False)}."
        )
        if scenario.form == 4:
            thoughts.append(
                f"Canonical claim {scenario.claim} {'belongs' if scenario.claim_supported else 'does not belong'} "
                f"to supported_claims, so claim_supported is {json.dumps(scenario.claim_supported)}. "
                "A false support verdict still answers the claim question."
            )
        else:
            thoughts.append("This well-formed native network has a valid other classification with no supported claims.")
    thoughts.append("Return resolved with no blocked reason.")
    thoughts.append(
        "The user needs this answer for a later workflow step, so continue regardless of the support verdict or match count."
        if scenario.framing == "step" else
        "The user asked for this answer as a standalone task, so stop regardless of the support verdict or match count."
    )
    return " ".join(thoughts)


def _answer(
    *,
    status: str,
    result: Mapping[str, Any],
    decision: str,
    blocked_reason: str | None,
    summary: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "provenance": None,
        "result": dict(result),
        "continuation_decision": decision,
        "blocked_reason": blocked_reason,
        "summary": summary,
    }


def _card_summary(scenario: Scenario) -> str:
    card = FAMILY_CARDS[scenario.family]
    supported = list(card["supported_claims"])
    if scenario.input_kind == "family_name":
        subject = f"The {scenario.family} family"
    elif isinstance(scenario.fact, LayerFact):
        subject = f"<LAYER:{scenario.fact.layer_id}>"
    else:
        raise CorpusBuildError("a claim card lacks its input")
    if not supported:
        if scenario.input_kind == "family_name":
            return "The other family supports no listed claim."
        return f"{subject} is in the other family and supports no listed claim."
    if scenario.input_kind == "family_name":
        return f"The {scenario.family} family supports {supported[0]} only."
    return (
        f"{subject} is in the {scenario.family} family and supports "
        f"{supported[0]} only."
    )


def _resolved_answer(scenario: Scenario) -> dict[str, Any]:
    if scenario.form == 2:
        matching = [
            item.layer_id
            for item in scenario.companions
            if item.layer_family == scenario.family
        ]
        other = [
            item.layer_id
            for item in scenario.companions
            if item.layer_family != scenario.family
        ]
        result = {
            "layer_family": scenario.family,
            "matching_layer_ids": matching,
            "other_layer_ids": other,
        }
        matching_anchors = ", ".join(
            f"<LAYER:{value}>" for value in matching
        )
        other_anchors = ", ".join(f"<LAYER:{value}>" for value in other)
        if matching:
            summary = (
                f"The {scenario.family} matches are: {matching_anchors}. "
                f"The other layers are: {other_anchors}."
            )
        else:
            summary = (
                f"No listed layer is in the {scenario.family} family. "
                f"The other layers are: {other_anchors}."
            )
        return _answer(
            status="resolved",
            result=result,
            decision=_decision(scenario),
            blocked_reason=None,
            summary=summary,
        )
    layer_id = (
        scenario.fact.layer_id if isinstance(scenario.fact, LayerFact) else None
    )
    result = family_card(scenario.family, layer_id)
    if scenario.form == 4:
        if scenario.claim is None or scenario.claim_supported is None:
            raise CorpusBuildError("form 4 lacks its claim outcome")
        result["claim"] = scenario.claim
        result["claim_supported"] = scenario.claim_supported
        prefix = "Yes" if scenario.claim_supported else "No"
        if layer_id is None:
            subject = f"The {scenario.family} family"
        else:
            subject = f"<LAYER:{layer_id}>"
        verb = "supports" if scenario.claim_supported else "does not support"
        summary = f"{prefix}. {subject} {verb} {scenario.claim}."
    else:
        summary = _card_summary(scenario)
    return _answer(
        status="resolved",
        result=result,
        decision=_decision(scenario),
        blocked_reason=None,
        summary=summary,
    )


def _invalid_answer(scenario: Scenario) -> dict[str, Any]:
    if not isinstance(scenario.fact, ModuleFact):
        raise CorpusBuildError("form 5 lacks a module ID")
    value = scenario.fact.module_id
    return _answer(
        status="invalid_input",
        result={"input": value, "expected": "layer_tag"},
        decision="defer",
        blocked_reason=f"The input {value} is a module ID, not a layer tag.",
        summary=(
            "The input is a module ID, not a layer tag. "
            "Give a layer tag to continue."
        ),
    )


def _row_id(scenario: Scenario, schema_version: str) -> str:
    return "wm3_s1_7_" + _stable_sha256(
        {"schema_version": schema_version, "scenario": scenario.scenario_id}
    )[:24]


def build_row(
    scenario: Scenario,
    *,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = FAMILY_PROMPTS,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Render one closed-book S1.7 row without a tool call."""

    schema_version = str(contract["runtime"]["row_schema_version"])
    answer = _invalid_answer(scenario) if scenario.form == 5 else _resolved_answer(scenario)
    metadata: dict[str, Any] = {
        "book_mode": "closed_book",
        "step": "S1",
        "question_family": "layer_family_claims",
        "form": scenario.form,
        "source_trace": scenario.source_trace,
        "split_unit": scenario.metadata_split_unit,
        "blocked_reason_code": "wrong_input_type" if scenario.form == 5 else None,
        "schema_version": schema_version,
        "system_prompt_sha256": system_prompt_sha256(contract),
        "tool_contract_version": contract["runtime"]["tool_contract_version"],
        "ensembl_release": contract["coordinate_system"]["ensembl_release"],
        "identifier_registry_id": contract["coordinate_system"][
            "identifier_registry_id"
        ],
    }
    if scenario.form == 5:
        metadata["found_input_type"] = "module_id"
    row = {
        "row_id": _row_id(scenario, schema_version),
        "messages": [
            {"role": "system", "content": contract["runtime"]["system_prompt"]},
            {"role": "user", "content": _prompt(scenario, prompts=prompts)},
            {
                "role": "assistant",
                "thinking": _thinking(scenario, final_reasoning=final_reasoning),
                "content": answer,
            },
        ],
        "tools": tool_definitions(contract) if tools is None else tools,
        "metadata": metadata,
    }
    return row, []


THINKING_FILTER_RULES = (
    "one thinking step per row",
    "thinking matches the exact family card",
    "claim support uses supported-list membership only",
    "no tool call or tool-result claim",
    "no gene or edge-instance fact",
    "status and decision agree with the answer",
)


def _final_answer(row: Mapping[str, Any]) -> Mapping[str, Any]:
    assistants = [
        message
        for message in row["messages"]
        if message.get("role") == "assistant" and "content" in message
    ]
    if len(assistants) != 1 or not isinstance(assistants[0]["content"], Mapping):
        raise CorpusBuildError(f"row {row['row_id']} lacks one final answer")
    return assistants[0]["content"]


def _tool_calls(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        message["tool_call"]
        for message in row["messages"]
        if message.get("role") == "assistant" and "tool_call" in message
    ]


def _current_user_message(row: Mapping[str, Any]) -> str:
    users = [message for message in row["messages"] if message.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0].get("content"), str):
        raise CorpusBuildError(f"row {row['row_id']} lacks one user message")
    return str(users[0]["content"])


def filter_row_thinking(
    row: Mapping[str, Any], scenario: Scenario, *,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> int:
    """Reject a thought that violates an S1.7 runtime rule."""

    if _tool_calls(row) or any(
        message.get("role") == "tool" for message in row["messages"]
    ):
        raise CorpusBuildError(f"row {row['row_id']} contains a tool action")
    assistants = [
        message for message in row["messages"] if message.get("role") == "assistant"
    ]
    if len(assistants) != 1 or assistants[0].get("thinking") != _thinking(
        scenario, final_reasoning=final_reasoning,
    ):
        raise CorpusBuildError(f"row {row['row_id']} changes its exact thinking")
    thought = str(assistants[0]["thinking"])
    lowered = thought.casefold()
    if any(value in lowered for value in ("provenance", "tool call", "tool result")):
        raise CorpusBuildError(f"row {row['row_id']} discusses runtime fields")
    tag_pattern = r"[A-Za-z][A-Za-z0-9_-]*(?::[A-Za-z0-9_-]+)+"
    visible_tags = set(re.findall(tag_pattern, _current_user_message(row)))
    for tag in re.findall(tag_pattern, thought):
        if tag not in visible_tags:
            raise CorpusBuildError(f"row {row['row_id']} invents the tag {tag}")
    answer = _final_answer(row)
    expected = _invalid_answer(scenario) if scenario.form == 5 else _resolved_answer(scenario)
    if answer != expected:
        raise CorpusBuildError(f"row {row['row_id']} changes its exact answer")
    return 1


def _prompt_and_outcome_signatures(
    row: Mapping[str, Any],
) -> tuple[str, str, str]:
    messages = row["messages"]
    last_user = max(
        index for index, message in enumerate(messages) if message.get("role") == "user"
    )
    prompt = messages[: last_user + 1]
    outcomes = [
        {"tool_name": message["name"], "result_status": message["content"]["status"]}
        for message in messages[last_user + 1 :]
        if message.get("role") == "tool"
    ]
    prompt_key = _canonical_json(prompt, sort_keys=False)
    outcome_key = _canonical_json(outcomes, sort_keys=False)
    combined = _canonical_json(
        {"prompt": prompt, "tool_outcomes": outcomes}, sort_keys=False
    )
    return prompt_key, outcome_key, combined


def duplicate_prompt_outcome_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
) -> dict[str, Any]:
    """Reject each repeated closed-book prompt and outcome."""

    split_results = {}
    for split in SPLITS:
        prompts: set[str] = set()
        combined: set[str] = set()
        for row, _ in zip(
            rows_by_split[split], scenarios_by_split[split], strict=True
        ):
            prompt, _, signature = _prompt_and_outcome_signatures(row)
            if prompt in prompts or signature in combined:
                raise CorpusBuildError(
                    f"{split} repeats a prompt and closed-book outcome"
                )
            prompts.add(prompt)
            combined.add(signature)
        split_results[split] = {
            "row_count": len(rows_by_split[split]),
            "normalized_prompt_count": len(prompts),
            "normalized_prompt_tool_outcome_count": len(combined),
            "duplicate_prompt_tool_outcome_count": 0,
            "intended_success_failure_pair_count": 0,
            "pass": True,
        }
    return {
        "normalization": "retain the exact visible prompt and every tool outcome",
        "splits": split_results,
        "pass": True,
    }


def _masked_user_wording(
    row: Mapping[str, Any],
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = FAMILY_PROMPTS,
) -> str:
    user = _current_user_message(row)
    template = prompts[scenario.split][scenario.framing][scenario.template_index]
    prefix, suffix = template.split("{request}")
    if not user.startswith(prefix) or (suffix and not user.endswith(suffix)):
        raise CorpusBuildError(f"row {row['row_id']} does not match its template")
    return " ".join(template.format(request=MASK_TOKEN).casefold().split())


def _source_values(scenario: Scenario) -> set[str]:
    values = {scenario.family}
    if isinstance(scenario.fact, LayerFact):
        values.add(scenario.fact.layer_id)
    elif isinstance(scenario.fact, ModuleFact):
        values.add(scenario.fact.module_id)
    values.update(item.layer_id for item in scenario.companions)
    if scenario.claim is not None:
        values.add(scenario.claim)
        values.update(CLAIM_PARAPHRASES[scenario.split][scenario.claim])
        if scenario.claim_phrase is not None:
            values.add(scenario.claim_phrase)
    return values


def masked_visible_prompt(row: Mapping[str, Any], scenario: Scenario) -> str:
    """Mask source values but retain visible prompt structure."""

    visible = _current_user_message(row)
    for value in sorted(_source_values(scenario), key=len, reverse=True):
        visible = re.sub(re.escape(value), MASK_TOKEN, visible, flags=re.IGNORECASE)
    return visible


def _wording_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", value))


def _wording_classifier(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    masker: Callable[[Mapping[str, Any], Scenario], str],
    scope: str,
    gated: bool,
    tolerance: float = MASKED_WORDING_TOLERANCE,
) -> dict[str, Any]:
    """Train and evaluate one class-balanced word model."""

    labels = tuple(sorted({item.form for item in scenarios_by_split["train"]}))
    token_counts: dict[int, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[int] = Counter()
    vocabulary: set[str] = set()
    for row, scenario in zip(
        rows_by_split["train"], scenarios_by_split["train"], strict=True
    ):
        tokens = _wording_tokens(masker(row, scenario))
        token_counts[scenario.form].update(tokens)
        token_totals[scenario.form] += len(tokens)
        vocabulary.update(tokens)
    if not vocabulary:
        raise CorpusBuildError("the wording classifier has no vocabulary")

    def predict(tokens: Sequence[str]) -> int:
        known = Counter(token for token in tokens if token in vocabulary)
        scores = []
        for label in labels:
            denominator = token_totals[label] + len(vocabulary)
            score = sum(
                count * math.log((token_counts[label][token] + 1) / denominator)
                for token, count in known.items()
            )
            scores.append((score, -label, label))
        return max(scores)[2]

    evaluations = {}
    passed_all = True
    for split in SPLITS:
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        distribution = Counter(item.form for item in scenarios)
        correct = sum(
            predict(_wording_tokens(masker(row, scenario))) == scenario.form
            for row, scenario in zip(rows, scenarios, strict=True)
        )
        accuracy = correct / len(rows)
        baseline = max(distribution.values()) / len(rows)
        apply_gate = gated and split != "train"
        threshold = baseline + tolerance if apply_gate else None
        passed = accuracy <= threshold if apply_gate else None
        if passed is False:
            passed_all = False
        evaluations[split] = {
            "row_count": len(rows),
            "correct": correct,
            "accuracy": accuracy,
            "largest_form_baseline": baseline,
            "threshold": threshold,
            "gate_applied": apply_gate,
            "pass": passed,
        }
    result = {
        "algorithm": "class_balanced_multinomial_naive_bayes_v1",
        "scope": scope,
        "diagnostic_only": not gated,
        "mask_token": MASK_TOKEN,
        "absolute_tolerance": tolerance,
        "training_vocabulary_size": len(vocabulary),
        "splits": evaluations,
        "pass": passed_all if gated else None,
    }
    if gated and not passed_all:
        raise CorpusBuildError("masked user wording predicts the S1.7 form")
    return result


def masked_wording_classifier_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    tolerance: float = MASKED_WORDING_TOLERANCE,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = FAMILY_PROMPTS,
) -> dict[str, Any]:
    """Gate leakage from split-held-out outer wrappers."""

    return _wording_classifier(
        rows_by_split,
        scenarios_by_split,
        masker=lambda row, scenario: _masked_user_wording(
            row, scenario, prompts=prompts
        ),
        scope="outer_wrapper_only",
        gated=True,
        tolerance=tolerance,
    )


def full_visible_prompt_classifier_diagnostic(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
) -> dict[str, Any]:
    """Measure full masked prompt cues without a pass gate."""

    return _wording_classifier(
        rows_by_split,
        scenarios_by_split,
        masker=masked_visible_prompt,
        scope="full_visible_prompt",
        gated=False,
    )


def _context_strata(
    facts: Iterable[LayerFact],
) -> tuple[dict[str, int], set[str]]:
    return s1_6._context_strata(facts)


def source_pool_coverage(
    pools: Mapping[str, Mapping[str, Sequence[Any]]]
) -> dict[str, Any]:
    """Summarize S1.7 source pools and exact context strata."""

    layer_counts = {}
    context_counts = {}
    strata = {}
    for split in SPLITS:
        layer_counts[split] = {
            name: len(pools[split][name])
            for name in (*FAMILIES, "all", "modules")
        }
        split_strata, contexts = _context_strata(pools[split]["all"])
        strata[split] = split_strata
        context_counts[split] = len(contexts)
    return {
        "layer_pool_counts": layer_counts,
        "context_counts": context_counts,
        "context_stratum_counts": strata,
        "native_layer_count": sum(len(pools[split]["all"]) for split in SPLITS),
    }


def _sentence_word_counts(text: str) -> list[int]:
    normalized = re.sub(r"<[^>]+>", " CODE ", text)
    normalized = re.sub(r"\S*:\S+", " CODE ", normalized)
    normalized = re.sub(r"\b[A-Za-z0-9]+_[A-Za-z0-9_-]+\b", " CODE ", normalized)
    return [
        len(re.findall(r"[A-Za-z]+|CODE", sentence))
        for sentence in re.split(r"[.!?]+", normalized)
        if sentence.strip()
    ]


def _prose_length_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    splits = {}
    for split in SPLITS:
        maximum_instruction = 0
        maximum_description = 0
        violations = 0
        for row in rows_by_split[split]:
            instructions = _sentence_word_counts(_current_user_message(row))
            maximum_instruction = max(maximum_instruction, max(instructions, default=0))
            if any(count > 20 for count in instructions):
                violations += 1
            assistant = next(
                message
                for message in row["messages"]
                if message.get("role") == "assistant" and "content" in message
            )
            answer = assistant["content"]
            descriptions = [str(answer["summary"])]
            if answer["blocked_reason"] is not None:
                descriptions.append(str(answer["blocked_reason"]))
            counts = [
                count
                for description in descriptions
                for count in _sentence_word_counts(description)
            ]
            maximum_description = max(
                maximum_description, max(counts, default=0),
                max(_sentence_word_counts(str(assistant["thinking"])), default=0),
            )
            if any(count > 25 for count in counts):
                violations += 1
        if violations:
            raise CorpusBuildError(f"{split} violates the prose length limits")
        splits[split] = {
            "row_count": len(rows_by_split[split]),
            "maximum_instruction_words": maximum_instruction,
            "maximum_description_words": maximum_description,
            "violation_count": 0,
            "pass": True,
        }
    return {"splits": splits, "pass": True}


def _expected_answer(scenario: Scenario) -> dict[str, Any]:
    return _invalid_answer(scenario) if scenario.form == 5 else _resolved_answer(scenario)


def audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    expected_counts: Mapping[str, Mapping[int, int]],
    *,
    framing_shares: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = FAMILY_PROMPTS,
    paraphrases: Mapping[str, Mapping[str, Sequence[str]]] = CLAIM_PARAPHRASES,
) -> dict[str, Any]:
    """Audit rendered rows against all S1.7 invariants."""

    if contract is None:
        contract = load_contract()
    if settings is None:
        settings = contract["curriculum"]["families"]["S1.7"]
    if framing_shares is None:
        framing_shares = {"direct": Fraction(1, 2), "step": Fraction(1, 2)}
    row_ids: set[str] = set()
    status_counts = {}
    tool_counts = {}
    framing_counts = {}
    template_counts = {}
    semantic_counts = {}
    family_counts = {}
    input_kind_counts = {}
    claim_counts = {}
    claim_outcomes = {}
    list_family_counts = {}
    list_outcomes = {}
    list_joint_counts = {}
    context_counts = {}
    context_strata = {}
    context_sets = {}
    per_form = {}
    thinking = {}
    structures = {}
    family_confusion = {}
    claim_set_gates = {}
    paraphrase_gates = {}
    claim_joint_counts = {}
    paraphrase_coverage = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} row and scenario counts differ")
        forms = Counter(item.form for item in scenarios)
        expected_forms = {form: int(value) for form, value in expected_counts[split].items()}
        if {form: forms[form] for form in expected_forms} != expected_forms:
            raise CorpusBuildError(f"{split} trace counts changed")
        frames = Counter(item.framing for item in scenarios)
        if dict(frames) != s1_6.largest_remainder_counts(len(rows), framing_shares):
            raise CorpusBuildError(f"{split} framing counts changed")
        statuses: Counter[str] = Counter()
        calls: Counter[str] = Counter()
        templates: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        semantics: Counter[str] = Counter()
        families: Counter[str] = Counter()
        kinds: Counter[str] = Counter()
        checks: Counter[str] = Counter()
        check_pairs: Counter[str] = Counter()
        claim_joint: Counter[tuple[str, str, bool, str, str]] = Counter()
        claim_phrase_indexes: dict[str, set[int]] = defaultdict(set)
        filters: Counter[str] = Counter()
        filter_families: Counter[str] = Counter()
        list_joint: Counter[tuple[str, str, str]] = Counter()
        matching_positions: dict[str, set[int]] = defaultdict(set)
        gates: dict[str, Counter[str]] = defaultdict(Counter)
        confusion: dict[str, Counter[str]] = defaultdict(Counter)
        facts_by_id: dict[str, LayerFact] = {}
        exact_claim_sets = 0
        paraphrase_correct = 0
        thinking_steps = 0
        no_calls = 0
        harnesses = 0
        for row, scenario in zip(rows, scenarios, strict=True):
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise CorpusBuildError(f"duplicate row ID {row_id}")
            row_ids.add(row_id)
            form_key = scenario.source_trace
            gates[form_key]["row_count"] += 1
            semantics[scenario.semantic_bucket] += 1
            if scenario.form != 5:
                families[scenario.family] += 1
            kinds[scenario.input_kind] += 1
            templates[form_key][scenario.framing][str(scenario.template_index)] += 1
            if any(message.get("role") == "harness" for message in row["messages"]):
                harnesses += 1
            row_calls = _tool_calls(row)
            calls[str(len(row_calls))] += 1
            if row_calls:
                raise CorpusBuildError(f"row {row_id} makes a forbidden tool call")
            no_calls += 1
            expected = _expected_answer(scenario)
            answer = _final_answer(row)
            for key in (
                "status", "result", "continuation_decision", "blocked_reason"
            ):
                if answer.get(key) != expected[key]:
                    raise CorpusBuildError(f"row {row_id} has the wrong {key}")
                gates[form_key][f"exact_{key}_count"] += 1
            if answer.get("provenance") is not None:
                raise CorpusBuildError(f"row {row_id} has closed-book provenance")
            gates[form_key]["null_provenance_count"] += 1
            gates[form_key]["exact_call_count"] += 1
            statuses[str(answer["status"])] += 1
            thinking_steps += filter_row_thinking(
                row, scenario, final_reasoning=settings.get("train_final_reasoning"),
            )
            result = answer["result"]
            if scenario.form != 5:
                if scenario.form == 2:
                    filters[str(scenario.list_outcome)] += 1
                    filter_families[scenario.family] += 1
                    list_joint[
                        (scenario.family, str(scenario.list_outcome), scenario.framing)
                    ] += 1
                    if scenario.list_outcome == "match":
                        for position, fact in enumerate(scenario.companions):
                            if fact.layer_family == scenario.family:
                                matching_positions[scenario.family].add(position)
                    for fact in scenario.companions:
                        facts_by_id[fact.fact_id] = fact
                        confusion[fact.layer_family][fact.layer_family] += 1
                else:
                    expected_card = family_card(
                        scenario.family,
                        scenario.fact.layer_id
                        if isinstance(scenario.fact, LayerFact) else None,
                    )
                    if all(result.get(key) == value for key, value in expected_card.items()):
                        exact_claim_sets += 1
                    else:
                        raise CorpusBuildError(f"row {row_id} changes a claim card")
                    if isinstance(scenario.fact, LayerFact):
                        facts_by_id[scenario.fact.fact_id] = scenario.fact
                        confusion[scenario.fact.layer_family][scenario.family] += 1
            if scenario.form == 4:
                label = "supported" if scenario.claim_supported else "unsupported"
                checks[label] += 1
                check_pairs[f"{scenario.family}:{scenario.claim}:{label}"] += 1
                claim_joint[
                    (
                        scenario.family,
                        str(scenario.claim),
                        bool(scenario.claim_supported),
                        scenario.input_kind,
                        scenario.framing,
                    )
                ] += 1
                claim_phrase_indexes[str(scenario.claim)].add(
                    scenario.claim_paraphrase_index
                )
                if scenario.claim_supported != (
                    scenario.claim in FAMILY_CARDS[scenario.family]["supported_claims"]
                ):
                    raise CorpusBuildError(f"row {row_id} leaks an unsupported claim")
                if any(
                    phrase in _current_user_message(row)
                    for phrase in paraphrases[split][str(scenario.claim)]
                ):
                    paraphrase_correct += 1
        for form, count in expected_forms.items():
            if not count:
                continue
            form_key = f"S1.7.{form}"
            for frame in ("direct", "step"):
                if set(templates[form_key][frame]) != {
                    str(index) for index in range(len(prompts[split][frame]))
                }:
                    raise CorpusBuildError(
                        f"{split} form {form} {frame} lacks template coverage"
                    )
        mode = _settings_mode(split)
        expected_filter_families = {
            key: int(value)
            for key, value in settings["list_filter"][f"{mode}_family_counts"].items()
        }
        expected_filter_outcomes = {
            key: int(value)
            for key, value in settings["list_filter"][f"{mode}_outcome_counts"].items()
        }
        if {
            key: filter_families[key] for key in expected_filter_families
        } != expected_filter_families:
            raise CorpusBuildError(f"{split} list-family counts changed")
        if dict(filters) != expected_filter_outcomes:
            raise CorpusBuildError(f"{split} list outcomes changed")
        expected_check_outcomes = {
            key: int(value)
            for key, value in settings["claim_check"][f"{mode}_outcome_counts"].items()
        }
        if split == "train" and settings.get("train_targeted_v9"):
            expected_check_outcomes = dict(Counter(
                "supported" if supported else "unsupported"
                for _, _, supported, _, _ in _claim_check_specs(
                    split, settings,
                    direct_target=_frame_counts(expected_counts[split], framing_shares)[4]["direct"],
                )
            ))
        if dict(checks) != expected_check_outcomes:
            raise CorpusBuildError(f"{split} claim-check outcomes changed")
        expected_list_joint = Counter(_list_specs(split, settings))
        if list_joint != expected_list_joint:
            raise CorpusBuildError(f"{split} list joint allocation changed")
        expected_target_families = {
            family
            for family, count in expected_filter_families.items()
            if count
        }
        if any(
            matching_positions[family] != {0, 1, 2}
            for family in expected_target_families
        ):
            raise CorpusBuildError(f"{split} list target positions changed")
        direct_target = _frame_counts(
            expected_counts[split], framing_shares
        )[4]["direct"]
        expected_claim_joint = Counter(
            _claim_check_specs(
                split,
                settings,
                direct_target=direct_target,
            )
        )
        if claim_joint != expected_claim_joint:
            raise CorpusBuildError(f"{split} claim-check joint allocation changed")
        for claim in CLAIM_VOCABULARY:
            expected_indexes = set(range(len(paraphrases[split][claim])))
            if claim_phrase_indexes[claim] != expected_indexes:
                raise CorpusBuildError(
                    f"{split} claim paraphrase coverage changed for {claim}"
                )
        split_strata, contexts = _context_strata(facts_by_id.values())
        expected_context_count = int(settings["context_split"]["counts"][split])
        expected_strata = {
            key: int(value)
            for key, value in settings["context_split"]["strata"][split].items()
        }
        if len(contexts) != expected_context_count or split_strata != expected_strata:
            raise CorpusBuildError(f"{split} context coverage changed")
        context_sets[split] = contexts
        context_counts[split] = len(contexts)
        context_strata[split] = split_strata
        status_counts[split] = dict(sorted(statuses.items()))
        tool_counts[split] = dict(sorted(calls.items()))
        framing_counts[split] = dict(sorted(frames.items()))
        template_counts[split] = {
            form: {
                frame: dict(sorted(values.items()))
                for frame, values in sorted(frame_values.items())
            }
            for form, frame_values in sorted(templates.items())
        }
        semantic_counts[split] = dict(sorted(semantics.items()))
        family_counts[split] = {family: families[family] for family in FAMILIES}
        input_kind_counts[split] = dict(sorted(kinds.items()))
        claim_outcomes[split] = {
            "supported": checks["supported"],
            "unsupported": checks["unsupported"],
        }
        claim_counts[split] = dict(sorted(check_pairs.items()))
        list_family_counts[split] = expected_filter_families
        list_outcomes[split] = expected_filter_outcomes
        list_joint_counts[split] = {
            f"{family}:{outcome}:{frame}": count
            for (family, outcome, frame), count in sorted(list_joint.items())
        }
        claim_joint_counts[split] = {
            f"{family}:{claim}:"
            f"{'supported' if supported else 'unsupported'}:"
            f"{input_kind}:{frame}": count
            for (
                family, claim, supported, input_kind, frame
            ), count in sorted(claim_joint.items())
        }
        paraphrase_coverage[split] = {
            claim: sorted(claim_phrase_indexes[claim])
            for claim in CLAIM_VOCABULARY
        }
        family_confusion[split] = {
            expected: dict(sorted(observed.items()))
            for expected, observed in sorted(confusion.items())
        }
        card_rows = sum(scenario.form not in {2, 5} for scenario in scenarios)
        claim_set_gates[split] = {
            "row_count": card_rows,
            "exact_claim_set_count": exact_claim_sets,
            "accuracy": exact_claim_sets / card_rows,
            "unsupported_claim_marked_supported_count": 0,
            "pass": exact_claim_sets == card_rows,
        }
        check_rows = forms[4]
        paraphrase_gates[split] = {
            "row_count": check_rows,
            "correct_count": paraphrase_correct,
            "accuracy": paraphrase_correct / check_rows,
            "pass": split == "train" or paraphrase_correct / check_rows >= 0.99,
        }
        per_form[split] = {}
        for form, count in expected_forms.items():
            form_key = f"S1.7.{form}"
            values = gates[form_key]
            gate = {
                "row_count": count,
                "exact_status_count": values["exact_status_count"],
                "exact_result_count": values["exact_result_count"],
                "exact_continuation_decision_count": values[
                    "exact_continuation_decision_count"
                ],
                "exact_blocked_reason_count": values["exact_blocked_reason_count"],
                "null_provenance_count": values["null_provenance_count"],
                "exact_call_count": values["exact_call_count"],
            }
            gate["pass"] = all(
                value == count
                for key, value in gate.items()
                if key.endswith("_count") and key != "row_count"
            )
            per_form[split][form_key] = gate
        thinking[split] = {
            "row_count": len(rows),
            "thinking_step_count": thinking_steps,
            "pass": thinking_steps == len(rows),
        }
        structures[split] = {
            "row_count": len(rows),
            "harness_message_count": harnesses,
            "no_tool_call_count": no_calls,
            "family_name_input_count": kinds["family_name"],
            "native_layer_input_count": kinds["layer_tag"],
            "layer_list_input_count": kinds["layer_list"],
            "wrong_type_input_count": kinds["module_id"],
            "pass": harnesses == 0 and no_calls == len(rows),
        }
    for first, second in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ):
        if context_sets[first] & context_sets[second]:
            raise CorpusBuildError(f"layer contexts overlap between {first} and {second}")
    duplicate = duplicate_prompt_outcome_audit(rows_by_split, scenarios_by_split)
    wording = masked_wording_classifier_audit(
        rows_by_split, scenarios_by_split, prompts=prompts
    )
    full_prompt = full_visible_prompt_classifier_diagnostic(
        rows_by_split, scenarios_by_split
    )
    prose = _prose_length_audit(rows_by_split)
    heldout_family_splits = {}
    for split in ("validation", "test"):
        total = sum(
            count
            for observed in family_confusion[split].values()
            for count in observed.values()
        )
        correct = sum(
            observed.get(expected, 0)
            for expected, observed in family_confusion[split].items()
        )
        accuracy = correct / total if total else 1.0
        heldout_family_splits[split] = {
            "row_count": total,
            "correct_count": correct,
            "accuracy": accuracy,
            "pass": accuracy >= 0.99,
        }
    heldout_family = {
        "splits": heldout_family_splits,
        "confusion_matrix": {
            split: family_confusion[split] for split in ("validation", "test")
        },
        "pass": all(item["pass"] for item in heldout_family_splits.values()),
    }
    heldout_paraphrase = {
        "splits": {
            split: paraphrase_gates[split] for split in ("validation", "test")
        },
        "pass": all(
            paraphrase_gates[split]["pass"] for split in ("validation", "test")
        ),
    }
    return {
        "status_counts": status_counts,
        "tool_call_counts": tool_counts,
        "framing_counts": framing_counts,
        "prompt_template_counts": template_counts,
        "semantic_bucket_counts": semantic_counts,
        "family_counts": family_counts,
        "input_kind_counts": input_kind_counts,
        "list_family_counts": list_family_counts,
        "list_outcome_counts": list_outcomes,
        "list_joint_counts": list_joint_counts,
        "claim_check_counts": claim_counts,
        "claim_outcome_counts": claim_outcomes,
        "claim_check_outcome_counts": claim_outcomes,
        "claim_check_joint_counts": claim_joint_counts,
        "claim_paraphrase_coverage": paraphrase_coverage,
        "context_counts": context_counts,
        "context_stratum_counts": context_strata,
        "per_form_gates": {
            "splits": per_form,
            "family_confusion_matrix": family_confusion,
            "heldout_family_accuracy": heldout_family,
            "claim_set_accuracy": claim_set_gates,
            "claim_paraphrase_accuracy": paraphrase_gates,
            "heldout_claim_paraphrase_accuracy": heldout_paraphrase,
            "unsupported_claim_leak_count": 0,
            "false_defer_count": 0,
            "pass": all(
                gate["pass"]
                for split_gates in per_form.values()
                for gate in split_gates.values()
            ) and all(gate["pass"] for gate in claim_set_gates.values())
            and all(gate["pass"] for gate in paraphrase_gates.values())
            and heldout_family["pass"]
            and heldout_paraphrase["pass"],
        },
        "prompt_structure_audit": {
            "splits": structures,
            "full_visible_prompt_classifier": full_prompt,
            "prose_length_audit": prose,
            "pass": all(item["pass"] for item in structures.values()) and prose["pass"],
        },
        "masked_wording_classifier": wording,
        "thinking_filter": {
            "rules": list(THINKING_FILTER_RULES),
            "splits": thinking,
            "pass": all(item["pass"] for item in thinking.values()),
        },
        "duplicate_prompt_tool_outcome": duplicate,
        "split_context_sha256": {
            split: _stable_sha256(sorted(values))
            for split, values in context_sets.items()
        },
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row, sort_keys=False) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_corpus(path: Path) -> None:
    validator = (
        REPOSITORY_ROOT
        / "agents"
        / "world_model_v3_curriculum"
        / "s1_kit"
        / "validate.py"
    )
    result = subprocess.run(
        [sys.executable, str(validator), str(path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CorpusBuildError(
            f"semantic validation failed for {path}:\n" + result.stdout + result.stderr
        )


def _configured_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def build_corpus(
    source_index_path: Path,
    output_directory: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path = CONTRACT_PATH,
    validate: bool = True,
    overlay: Mapping[str, str] | None = None,
    prompt_banks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate S1.7, then publish its manifest last.

    The v5 overlay changes training counts, claim phrases, and reasoning.
    Validation and test keep their published rows byte for byte.
    """

    source_index_path = source_index_path.resolve()
    payload = check_source_index(source_index_path, contract)
    curriculum = contract["curriculum"]
    settings = curriculum["families"]["S1.7"]
    seed = int(settings["assignment_seed"])
    prompts = (
        wrapper_prompts(prompt_banks, FAMILY_PROMPTS, bank_name="layer_family")
        if prompt_banks is not None
        else FAMILY_PROMPTS
    )
    paraphrases = (
        bank_claim_paraphrases(prompt_banks, CLAIM_PARAPHRASES)
        if prompt_banks is not None else CLAIM_PARAPHRASES
    )
    if any(
        len(prompts["train"][framing]) % len(FAMILY_PROMPTS["train"][framing])
        for framing in ("direct", "step")
    ):
        raise CorpusBuildError(
            "the S1.7 training bank must be a multiple of the inline bank"
        )
    row_counts = split_row_counts(
        int(settings["train_rows"]), curriculum["split_fractions"]
    )
    counts_by_split = {
        split: trace_counts(row_counts[split], settings) for split in SPLITS
    }
    pools = build_fact_pools(
        payload, settings=settings, seed=seed, contract=contract
    )
    coverage = source_pool_coverage(pools)
    scenarios_by_split = select_scenarios(
        pools,
        counts_by_split,
        seed=seed,
        framing_shares=curriculum["framing"],
        settings=settings,
        prompts=prompts,
        paraphrases=paraphrases,
    )
    tools = tool_definitions(contract)
    rows_by_split = {
        split: [
            build_row(
                scenario, contract=contract, tools=tools, prompts=prompts,
                final_reasoning=settings.get("train_final_reasoning"),
            )[0]
            for scenario in scenarios_by_split[split]
        ]
        for split in SPLITS
    }
    audit = audit_rows(
        rows_by_split,
        scenarios_by_split,
        counts_by_split,
        framing_shares=curriculum["framing"],
        settings=settings,
        contract=contract,
        prompts=prompts,
        paraphrases=paraphrases,
    )
    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".s1_7_build_", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        paths = {split: temporary / f"{split}.jsonl" for split in SPLITS}
        for split, path in paths.items():
            _write_jsonl(path, rows_by_split[split])
            if validate:
                _validate_corpus(path)
        resolved_contract_path = contract_path.resolve()
        s1_6_builder_path = REPOSITORY_ROOT / "runtime/world_model/s1_6_corpus.py"
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "section": "S1.7",
            "dataset_id": settings["dataset_id"],
            "seed": seed,
            "prompt_templates_sha256": _stable_sha256(prompts),
            "claim_paraphrases_sha256": _stable_sha256(paraphrases),
            "scenario_assignment_sha256": {
                split: _stable_sha256(
                    [item.scenario_id for item in scenarios_by_split[split]]
                )
                for split in SPLITS
            },
            "row_counts": row_counts,
            "form_counts": {
                split: {
                    f"S1.7.{form}": int(count)
                    for form, count in counts_by_split[split].items()
                }
                for split in SPLITS
            },
            "semantic_bucket_counts": audit["semantic_bucket_counts"],
            "status_counts": audit["status_counts"],
            "tool_call_counts": audit["tool_call_counts"],
            "framing_counts": audit["framing_counts"],
            "prompt_template_counts": audit["prompt_template_counts"],
            "family_counts": audit["family_counts"],
            "input_kind_counts": audit["input_kind_counts"],
            "list_family_counts": audit["list_family_counts"],
            "list_outcome_counts": audit["list_outcome_counts"],
            "list_joint_counts": audit["list_joint_counts"],
            "claim_check_counts": audit["claim_check_counts"],
            "claim_check_outcome_counts": audit["claim_check_outcome_counts"],
            "claim_check_joint_counts": audit["claim_check_joint_counts"],
            "claim_paraphrase_coverage": audit["claim_paraphrase_coverage"],
            "context_counts": audit["context_counts"],
            "context_stratum_counts": audit["context_stratum_counts"],
            "sources": {
                "source_index": {
                    "path": (
                        str(source_index_path.relative_to(REPOSITORY_ROOT))
                        if source_index_path.is_relative_to(REPOSITORY_ROOT)
                        else str(source_index_path)
                    ),
                    "sha256": file_sha256(source_index_path),
                    "content_sha256": contract["source_index"]["content_sha256"],
                },
                "multiplex_flist": dict(payload["sources"]["multiplex_flist"]),
                "contract": {
                    "path": (
                        str(resolved_contract_path.relative_to(REPOSITORY_ROOT))
                        if resolved_contract_path.is_relative_to(REPOSITORY_ROOT)
                        else str(resolved_contract_path)
                    ),
                    "sha256": file_sha256(resolved_contract_path),
                },
                "builder": {
                    "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
                    "sha256": file_sha256(Path(__file__).resolve()),
                },
                "s1_6_builder": {
                    "path": "runtime/world_model/s1_6_corpus.py",
                    "sha256": file_sha256(s1_6_builder_path),
                },
                "native_layer_policy": "use_only_the_353_pinned_flist_tags",
                "records": payload["sources"],
            },
            "source_pool_coverage": {
                **coverage,
                "split_context_sha256": audit["split_context_sha256"],
            },
            "per_form_gates": audit["per_form_gates"],
            "prompt_structure_audit": audit["prompt_structure_audit"],
            "masked_wording_classifier": audit["masked_wording_classifier"],
            "thinking_filter": audit["thinking_filter"],
            "duplicate_prompt_tool_outcome": audit[
                "duplicate_prompt_tool_outcome"
            ],
            "evidence": {
                "call_fields": ["evidence_id", "row_id", "producer"],
                "calls": [],
                "call_count": 0,
            },
            "files": {
                path.name: {
                    "row_count": row_counts[split],
                    "sha256": file_sha256(path),
                }
                for split, path in paths.items()
            },
        }
        expected_manifest_keys = {
            "schema_version", "corpus_schema_version", "section", "dataset_id",
            "seed", "prompt_templates_sha256", "claim_paraphrases_sha256",
            "scenario_assignment_sha256", "row_counts", "form_counts",
            "semantic_bucket_counts", "status_counts", "tool_call_counts",
            "framing_counts", "prompt_template_counts", "family_counts",
            "input_kind_counts", "list_family_counts", "list_outcome_counts",
            "list_joint_counts", "claim_check_counts",
            "claim_check_outcome_counts", "claim_check_joint_counts",
            "claim_paraphrase_coverage", "context_counts",
            "context_stratum_counts", "sources", "source_pool_coverage",
            "per_form_gates", "prompt_structure_audit",
            "masked_wording_classifier", "thinking_filter",
            "duplicate_prompt_tool_outcome", "evidence", "files",
        }
        if set(manifest) != expected_manifest_keys:
            raise CorpusBuildError("the S1.7 manifest fields changed")
        if overlay is not None:
            manifest["s1_overlay"] = dict(overlay)
        if prompt_banks is not None:
            manifest["prompt_bank"] = prompt_bank_manifest(prompt_banks)
        if "train_final_reasoning" in settings:
            manifest["train_final_reasoning"] = dict(settings["train_final_reasoning"])
        if settings.get("train_targeted_v9"):
            manifest["train_targeted_v9"] = targeted_v9_audit(scenarios_by_split["train"], paraphrases, prompts)
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        output_directory.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            (temporary / name).replace(output_directory / name)
        manifest_path.replace(output_directory / "manifest.json")
    return manifest


def targeted_v9_audit(
    scenarios: Sequence[Scenario], paraphrases: Mapping[str, Mapping[str, Sequence[str]]],
    prompts: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    """Require balanced canonical claims and expose every training paraphrase in both scopes."""

    cases = [case for case in scenarios if case.form == 4]
    counts = Counter(str(case.claim) for case in cases)
    expected = s1_6.largest_remainder_counts(
        len(cases), {claim: Fraction(1, len(CLAIM_VOCABULARY)) for claim in CLAIM_VOCABULARY},
    )
    if counts != Counter(expected):
        raise CorpusBuildError("v9 canonical claim counts are not balanced")
    cells: dict[tuple[str, bool], Counter[str]] = defaultdict(Counter)
    phrases: dict[tuple[str, bool, str], set[int]] = defaultdict(set)
    for case in cases:
        claim = str(case.claim)
        cells[(claim, bool(case.claim_supported))][case.framing] += 1
        phrases[(claim, bool(case.claim_supported), case.framing)].add(case.claim_paraphrase_index)
        if case.claim_supported != (claim in FAMILY_CARDS[case.family]["supported_claims"]):
            raise CorpusBuildError("v9 claim verdict contradicts its family card")
    for claim, count in counts.items():
        supportable = any(claim in card["supported_claims"] for card in FAMILY_CARDS.values())
        positive = sum(cells.get((claim, True), {}).values())
        if (supportable and abs(positive - (count - positive)) > 1) or (not supportable and positive):
            raise CorpusBuildError("v9 claim verdict margins changed")
        for verdict in ((False, True) if supportable else (False,)):
            frames = cells[(claim, verdict)]
            if abs(frames["direct"] - frames["step"]) > 1:
                raise CorpusBuildError("v9 claim verdict predicts scope")
            for frame in ("direct", "step"):
                if phrases[(claim, verdict, frame)] != set(range(len(paraphrases["train"][claim]))):
                    raise CorpusBuildError("v9 claim paraphrase lacks an applicable verdict or scope")
    wording: dict[tuple[int, str, str, str, int], Counter[str]] = defaultdict(Counter)
    other_names: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    native_mapping: dict[str, Counter[str]] = defaultdict(Counter)
    for case in scenarios:
        if case.form == 2:
            for attribute in ("template_index", "request_variant"):
                wording[(case.form, case.family, case.framing, attribute, getattr(case, attribute))][str(case.list_outcome)] += 1
        elif case.form == 4:
            for attribute in ("template_index", "request_variant"):
                wording[(case.form, str(case.claim), case.framing, attribute, getattr(case, attribute))][str(bool(case.claim_supported))] += 1
        elif case.form == 8:
            if not isinstance(case.fact, LayerFact) or case.fact.source != "HumanNetV3" or case.fact.layer_family != "other":
                raise CorpusBuildError("v9 other claims must use an existing native unmapped HumanNet tag")
            for attribute in ("template_index", "request_variant"):
                other_names[(str(case.fact.network), case.framing, attribute)].add(getattr(case, attribute))
        if case.form in {2, 8}:
            for fact in (*case.companions, *((case.fact,) if isinstance(case.fact, LayerFact) else ())):
                if fact.source == "HumanNetV3":
                    native_mapping[f"{fact.network}:{fact.layer_family}"][case.framing] += 1
    for (form, subject, frame, attribute, _), outcomes in wording.items():
        expected_outcomes = {"match", "no_match"} if form == 2 else (
            {"False", "True"} if any(subject in card["supported_claims"] for card in FAMILY_CARDS.values()) else {"False"}
        )
        if set(outcomes) != expected_outcomes:
            raise CorpusBuildError("v9 neutral wording predicts a list or support outcome")
    if len({network for network, _, _ in other_names}) != 3:
        raise CorpusBuildError("v9 changed the three native unmapped HumanNet training names")
    for (network, frame, attribute), indexes in other_names.items():
        width = len(prompts["train"][frame]) if attribute == "template_index" else len(CARD_REQUEST_VARIANTS)
        if indexes != set(range(width)):
            raise CorpusBuildError(f"v9 other-network wording coverage is incomplete for {network}")
    if len(other_names) != 3 * 2 * 2 or any(set(frames) != {"direct", "step"} for frames in native_mapping.values()):
        raise CorpusBuildError("v9 native HumanNet names do not cover both task scopes")
    return {
        "pass": True,
        "canonical_claim_counts": dict(counts),
        "claim_verdict_scope_counts": {f"{claim}:{verdict}": dict(frames) for (claim, verdict), frames in sorted(cells.items())},
        "paraphrase_verdict_scope_coverage": {f"{claim}:{verdict}:{frame}": len(indexes) for (claim, verdict, frame), indexes in sorted(phrases.items())},
        "wording_outcome_counts": {f"{form}:{subject}:{frame}:{attribute}:{index}": dict(outcomes) for (form, subject, frame, attribute, index), outcomes in sorted(wording.items())},
        "other_network_wording_coverage": {f"{network}:{frame}:{attribute}": len(indexes) for (network, frame, attribute), indexes in sorted(other_names.items())},
        "native_humannet_mapping_scope_counts": {key: dict(frames) for key, frames in sorted(native_mapping.items())},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument(
        "--overlay",
        type=Path,
        help="apply an S1 overlay such as runtime/world_model/s1_v3.yaml",
    )
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip semantic validation for a focused test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    contract = load_contract(args.contract)
    overlay = None
    prompt_banks = None
    if args.overlay is not None:
        contract, payload = apply_s1_overlay(
            contract, args.overlay, base_contract_path=args.contract
        )
        overlay = overlay_pin(args.overlay, payload)
        prompt_banks = payload.get("prompt_banks")
    digest = contract["source_index"]["content_sha256"]
    source_index = args.source_index or (
        _configured_path(contract["source_index"]["output_directory"])
        / f"world_model_source_index_{digest}.json.gz"
    )
    settings = contract["curriculum"]["families"]["S1.7"]
    output_directory = args.output_directory or _configured_path(
        settings["output_directory"]
    )
    canonical_output = _configured_path(settings["output_directory"]).resolve()
    if args.no_validate and output_directory.resolve() == canonical_output:
        parser.error(
            "--no-validate requires an explicit noncanonical output directory"
        )
    manifest = build_corpus(
        source_index,
        output_directory,
        contract=contract,
        contract_path=args.contract,
        validate=not args.no_validate,
        overlay=overlay,
        prompt_banks=prompt_banks,
    )
    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                "row_counts": manifest["row_counts"],
                "files": manifest["files"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
