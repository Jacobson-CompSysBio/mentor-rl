"""Build the canonical S1.6 layer-tag parsing corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from .contract import (
    CONTRACT_PATH,
    REPOSITORY_ROOT,
    load_contract,
    parse_layer_ref,
    system_prompt_sha256,
    tool_definitions,
)
from .s1_overlay import apply_s1_overlay, overlay_pin
from .s1_prompt_banks import prompt_bank_manifest, wrapper_prompts
from .source_index import check_source_index
from .sources import file_sha256


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-6-corpus-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-6-manifest-v1"
SPLITS = ("train", "validation", "test")

FORM_BY_BUCKET = {
    "bulk_pen_direct": 1,
    "scpen_one_field": 2,
    "humannet_direct": 3,
    "list_filter": 4,
    "wrong_input_type": 5,
    "malformed_tag": 6,
    "tf_target_direct": 7,
}
BUCKET_BY_FORM = {value: key for key, value in FORM_BY_BUCKET.items()}
SEMANTIC_BUCKET_BY_FORM = {
    1: "direct_parse",
    2: "one_field",
    3: "direct_parse",
    4: "list_filter",
    5: "wrong_input_type",
    6: "malformed_tag",
    7: "direct_parse",
}
ONE_FIELD_NAMES = (
    "layer_family",
    "source",
    "network",
    "organ",
    "region_path",
    "cell_type_path",
)
MALFORMED_SUBTYPES = (
    "unrecognized_prefix",
    "wrong_arity",
    "missing_path",
    "excess_path",
    "synthetic_field",
    "mixed_separator",
)
MODULE_NAMESPACES = ("rwr_loe", "mentor_ev")
MASK_TOKEN = "<SOURCE_VALUE>"
MASKED_WORDING_TOLERANCE = 0.05
DIRECT_REQUEST_VARIANTS = tuple(
    f"{verb} {scope}"
    for verb in (
        "parse",
        "read",
        "decode",
        "interpret",
        "extract",
        "return",
    )
    for scope in (
        "all seven fields for layer {layer_id}",
        "the complete LayerRef for layer {layer_id}",
        "every exact tag field for layer {layer_id}",
        "the full parsed record for layer {layer_id}",
    )
)

# Each split has a separate wrapper pool. Form-specific text stays in the
# request payload, so every wrapper can occur with every form.
LAYER_PROMPTS = {
    "train": {
        "direct": (
            "Parse this layer request: {request}.",
            "Return the fields for this layer request: {request}.",
            "Read the supplied layer data: {request}.",
            "Answer this layer-tag request: {request}.",
        ),
        "step": (
            "Before the next step, parse this layer request: {request}.",
            "The next step needs these layer fields: {request}.",
            "Prepare this layer data for the next step: {request}.",
            "Check this layer request before work continues: {request}.",
        ),
    },
    "validation": {
        "direct": (
            "Give the result for this layer request: {request}.",
            "Find the fields in this layer request: {request}.",
            "Supply the answer for this layer tag: {request}.",
            "Read and answer this layer request: {request}.",
        ),
        "step": (
            "Parse this layer request before analysis continues: {request}.",
            "This layer result is needed for the next action: {request}.",
            "Before work continues, read this layer request: {request}.",
            "Prepare these layer fields for later use: {request}.",
        ),
    },
    "test": {
        "direct": (
            "What is the answer for this layer request: {request}?",
            "Give the parsed result for this layer tag: {request}.",
            "Answer the following layer request: {request}.",
            "Identify the fields in this layer request: {request}.",
        ),
        "step": (
            "The workflow first needs this layer request parsed: {request}.",
            "Answer this layer request for the next action: {request}.",
            "Before the workflow advances, parse this request: {request}.",
            "Check these layer fields for later use: {request}.",
        ),
    },
}
LAYER_TAG_PROMPTS = LAYER_PROMPTS


class CorpusBuildError(ValueError):
    """Report a source, selection, or corpus invariant failure."""


@dataclass(frozen=True, slots=True)
class LayerFact:
    """Store one exact native layer tag and its parsed fields."""

    fact_id: str
    layer_id: str
    layer_family: str
    source: str | None
    network: str | None
    organ: str | None
    region_path: tuple[str, ...] | None
    cell_type_path: tuple[str, ...] | None
    context_key: str

    @property
    def layer_ref(self) -> dict[str, Any]:
        """Return the seven-key public layer result."""

        return {
            "layer_id": self.layer_id,
            "layer_family": self.layer_family,
            "source": self.source,
            "network": self.network,
            "organ": self.organ,
            "region_path": (
                list(self.region_path) if self.region_path is not None else None
            ),
            "cell_type_path": (
                list(self.cell_type_path)
                if self.cell_type_path is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ModuleFact:
    """Store one source-backed module ID for a wrong-type row."""

    fact_id: str
    module_id: str
    namespace: str
    split_unit: str


@dataclass(frozen=True, slots=True)
class Scenario:
    """Store one selected S1.6 form before row rendering."""

    split: str
    form: int
    fact: LayerFact | ModuleFact
    framing: str
    template_index: int
    occurrence: int
    companions: tuple[LayerFact, ...] = ()
    requested_field: str | None = None
    filter_value: str | None = None
    filter_outcome: str | None = None
    malformed_subtype: str | None = None
    malformed_input: str | None = None
    request_variant: int = 0
    # Reasoning changes do not change the selected case or its row ID.
    train_reasoning: bool = False
    train_targeted_v9: bool = False

    @property
    def semantic_bucket(self) -> str:
        return SEMANTIC_BUCKET_BY_FORM[self.form]

    @property
    def source_trace(self) -> str:
        return f"S1.6.{self.form}"

    @property
    def split_units(self) -> tuple[str, ...]:
        if isinstance(self.fact, ModuleFact):
            return (self.fact.split_unit,)
        return tuple(
            sorted({self.fact.context_key, *(item.context_key for item in self.companions)})
        )

    @property
    def metadata_split_unit(self) -> str:
        if len(self.split_units) == 1:
            return self.split_units[0]
        return "group_" + _stable_sha256(self.split_units)[:24]

    @property
    def scenario_id(self) -> str:
        return _stable_sha256(
            {
                "split": self.split,
                "form": self.form,
                "fact_id": self.fact.fact_id,
                "framing": self.framing,
                "template_index": self.template_index,
                "occurrence": self.occurrence,
                "companions": [item.fact_id for item in self.companions],
                "requested_field": self.requested_field,
                "filter_value": self.filter_value,
                "filter_outcome": self.filter_outcome,
                "malformed_subtype": self.malformed_subtype,
                "malformed_input": self.malformed_input,
                "request_variant": self.request_variant,
            }
        )

    @property
    def expected_matching_layer_ids(self) -> tuple[str, ...]:
        """Return the exact form-4 matches in input order."""

        if self.form != 4 or self.requested_field is None:
            return ()
        return tuple(
            fact.layer_id
            for fact in self.companions
            if self.filter_value in _field_components(fact, self.requested_field)
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


def _fact_id(kind: str, value: str) -> str:
    return "fact_" + _stable_sha256({"kind": kind, "value": value})[:24]


def split_row_counts(
    train_rows: int, split_fractions: Mapping[str, Any]
) -> dict[str, int]:
    """Derive each split count from the training anchor."""

    train_fraction = _fraction(split_fractions["train"])
    counts = {"train": train_rows}
    for split in ("validation", "test"):
        value = Fraction(train_rows) * _fraction(split_fractions[split]) / train_fraction
        if value.denominator != 1:
            raise CorpusBuildError(
                f"S1.6 {split} row count is not integral: {value}"
            )
        counts[split] = value.numerator
    return counts


def largest_remainder_counts(
    total: int, shares: Mapping[Any, Any]
) -> dict[Any, int]:
    """Apportion an integer total and use input order for each tie."""

    names = list(shares)
    raw = {name: Fraction(total) * _fraction(shares[name]) for name in names}
    counts = {
        name: value.numerator // value.denominator for name, value in raw.items()
    }
    missing = total - sum(counts.values())
    order = sorted(
        names,
        key=lambda name: (-(raw[name] - counts[name]), names.index(name)),
    )
    for name in order[:missing]:
        counts[name] += 1
    if sum(counts.values()) != total:
        raise CorpusBuildError("the largest-remainder allocation did not reconcile")
    return counts


def form_counts(total: int, settings: Mapping[str, Any]) -> dict[int, int]:
    """Return the exact S1.6 trace-form allocation for one split."""

    allocations = settings["allocations"]
    evaluation = settings["evaluation_allocations"]
    if tuple(allocations) != tuple(FORM_BY_BUCKET):
        raise CorpusBuildError("the S1.6 training allocation fields changed")
    if tuple(evaluation) != tuple(FORM_BY_BUCKET):
        raise CorpusBuildError("the S1.6 evaluation allocation fields changed")
    train_total = int(settings["train_rows"])
    evaluation_total = sum(int(value) for value in evaluation.values())
    if sum(int(value) for value in allocations.values()) != train_total:
        raise CorpusBuildError("the S1.6 training allocations do not reconcile")
    if total == train_total:
        selected = allocations
    elif total == evaluation_total:
        selected = evaluation
    else:
        selected = largest_remainder_counts(
            total,
            {
                name: Fraction(int(value), train_total)
                for name, value in allocations.items()
            },
        )
    return {
        FORM_BY_BUCKET[name]: int(selected[name])
        for name in FORM_BY_BUCKET
    }


def parse_layer_tag(
    layer_id: str, contract: Mapping[str, Any]
) -> LayerFact:
    """Parse one valid native tag without normalizing its values."""

    try:
        values = parse_layer_ref(contract, layer_id)
    except (TypeError, ValueError) as exc:
        raise CorpusBuildError(f"invalid layer tag {layer_id!r}: {exc}") from exc
    prefix = layer_id.split(":", 1)[0]
    region_path = (
        tuple(str(value) for value in values["region_path"])
        if values["region_path"] is not None
        else None
    )
    cell_type_path = (
        tuple(str(value) for value in values["cell_type_path"])
        if values["cell_type_path"] is not None
        else None
    )
    if prefix in {"bulkPEN", "scPEN"}:
        if values["organ"] is None or region_path is None:
            raise CorpusBuildError("a PEN layer lacks its context fields")
        context_key = "context:" + ":".join(
            (str(values["organ"]), *region_path)
        )
    elif prefix == "HumanNetV3":
        context_key = f"network:{values['network']}"
    elif prefix == "TFs":
        context_key = f"tf:{values['organ']}"
    else:
        raise CorpusBuildError(f"unsupported layer prefix {prefix!r}")
    return LayerFact(
        fact_id=_fact_id("layer", layer_id),
        layer_id=layer_id,
        layer_family=str(values["layer_family"]),
        source=None if values["source"] is None else str(values["source"]),
        network=None if values["network"] is None else str(values["network"]),
        organ=None if values["organ"] is None else str(values["organ"]),
        region_path=region_path,
        cell_type_path=cell_type_path,
        context_key=context_key,
    )


def _seeded_order(values: Iterable[Any], *, seed: int, label: str) -> list[Any]:
    return sorted(
        values,
        key=lambda value: _stable_sha256(
            {"seed": seed, "label": label, "value": str(value)}
        ),
    )


def _context_assignments(
    facts: Sequence[LayerFact], *, settings: Mapping[str, Any], seed: int
) -> dict[str, str]:
    """Assign exact contexts while each held-out split remains feasible."""

    facts_by_context: dict[str, list[LayerFact]] = defaultdict(list)
    for fact in facts:
        facts_by_context[fact.context_key].append(fact)
    configured = {
        split: int(count)
        for split, count in settings["context_split"]["counts"].items()
    }
    if configured != {"train": 27, "validation": 4, "test": 4}:
        raise CorpusBuildError("the S1.6 context allocation changed")
    if len(facts_by_context) != sum(configured.values()):
        raise CorpusBuildError("the native layer context count changed")

    mixed = [
        key
        for key, values in facts_by_context.items()
        if {item.layer_family for item in values} >= {"bulk_pen", "scpen"}
    ]
    humannet = [key for key in facts_by_context if key.startswith("network:")]
    if len(mixed) < 2 or len(humannet) < 6:
        raise CorpusBuildError("the native tags cannot support held-out forms")
    mixed = _seeded_order(mixed, seed=seed, label="mixed_context")
    humannet = _seeded_order(humannet, seed=seed, label="humannet_context")
    bulk_only = _seeded_order(
        [
            key
            for key, values in facts_by_context.items()
            if {item.layer_family for item in values} == {"bulk_pen"}
        ],
        seed=seed,
        label="bulk_only_context",
    )
    sc_only = _seeded_order(
        [
            key
            for key, values in facts_by_context.items()
            if {item.layer_family for item in values} == {"scpen"}
        ],
        seed=seed,
        label="sc_only_context",
    )
    if len(bulk_only) < 2 or len(sc_only) < 2:
        raise CorpusBuildError("the native tags lack held-out context strata")
    selected = {
        "validation": {mixed[0], humannet[0], bulk_only[0], sc_only[0]},
        "test": {mixed[1], humannet[1], bulk_only[1], sc_only[1]},
    }
    if selected["validation"] & selected["test"]:
        raise CorpusBuildError("the held-out context sets overlap")
    assignments = {key: "train" for key in facts_by_context}
    for split in ("validation", "test"):
        if len(selected[split]) != configured[split]:
            raise CorpusBuildError(f"{split} has the wrong context count")
        for key in selected[split]:
            assignments[key] = split
    actual = Counter(assignments.values())
    if dict(actual) != configured:
        raise CorpusBuildError("the S1.6 context split did not reconcile")
    if assignments.get("tf:brain") != "train":
        raise CorpusBuildError("the singleton TF context must stay in training")
    return assignments


def _module_facts(
    payload: Mapping[str, Any], *, seed: int
) -> dict[str, list[ModuleFact]]:
    """Build split-disjoint module IDs from pinned source records."""

    by_split: dict[str, list[ModuleFact]] = {split: [] for split in SPLITS}
    namespaces: dict[str, list[str]] = {
        "rwr_loe": [
            "rwr_loe:full_brain_multiplex_v1:"
            f"seed_{gene_id}:geometric_elbow_v1"
            for gene_id in payload["multiplex_gene_ids"]
        ],
        "mentor_ev": [
            "mentor_ev:full_brain_all_genes_v1:"
            f"clade_{int(row[0]):06d}"
            for row in payload["mentor_clades"]
        ],
    }
    for namespace, module_ids in namespaces.items():
        ordered = _seeded_order(
            module_ids, seed=seed, label=f"module:{namespace}"
        )
        for index, module_id in enumerate(ordered):
            residue = index % 10
            split = "validation" if residue == 0 else "test" if residue == 1 else "train"
            by_split[split].append(
                ModuleFact(
                    fact_id=_fact_id("module", module_id),
                    module_id=module_id,
                    namespace=namespace,
                    split_unit="module_" + _stable_sha256(module_id)[:24],
                )
            )
    return by_split


def build_fact_pools(
    payload: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    seed: int,
    contract: Mapping[str, Any],
) -> FactPools:
    """Build the exact split-safe layer and module source pools."""

    layer_ids = payload.get("layer_ids")
    if not isinstance(layer_ids, list) or len(layer_ids) != 353:
        raise CorpusBuildError("the pinned native layer count changed")
    if len(layer_ids) != len(set(layer_ids)):
        raise CorpusBuildError("the pinned native layer tags are not unique")
    facts = [parse_layer_tag(str(layer_id), contract) for layer_id in layer_ids]
    assignments = _context_assignments(facts, settings=settings, seed=seed)
    pool_names = ("all", "bulk_pen", "scpen", "humannet", "tf_target", "modules")
    pools: FactPools = {
        split: {name: [] for name in pool_names} for split in SPLITS
    }
    for fact in facts:
        split = assignments[fact.context_key]
        pools[split]["all"].append(fact)
        if fact.layer_family == "bulk_pen":
            pools[split]["bulk_pen"].append(fact)
        elif fact.layer_family == "scpen":
            pools[split]["scpen"].append(fact)
        elif fact.source == "HumanNetV3":
            pools[split]["humannet"].append(fact)
        elif fact.layer_family == "tf_target":
            pools[split]["tf_target"].append(fact)
    modules = _module_facts(payload, seed=seed)
    for split in SPLITS:
        pools[split]["modules"].extend(modules[split])
        for name in pool_names:
            pools[split][name].sort(key=lambda item: item.fact_id)
        required = ("all", "bulk_pen", "scpen", "humannet", "modules")
        if any(not pools[split][name] for name in required):
            raise CorpusBuildError(f"{split} lacks a required S1.6 source pool")
    if len(pools["train"]["tf_target"]) != 1:
        raise CorpusBuildError("training must contain the one native TF tag")
    if pools["validation"]["tf_target"] or pools["test"]["tf_target"]:
        raise CorpusBuildError("a held-out split contains the singleton TF tag")
    return pools


def _field_components(fact: LayerFact, field: str) -> tuple[str, ...]:
    """Return each exact scalar or path component for one filter field."""

    if field not in ONE_FIELD_NAMES:
        raise CorpusBuildError(f"unknown layer field {field!r}")
    value = getattr(fact, field)
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    return (str(value),)


def _balanced_labels(counts: Mapping[str, int]) -> list[str]:
    """Interleave labels while preserving exact configured counts."""

    remaining = {name: int(value) for name, value in counts.items()}
    labels: list[str] = []
    while any(remaining.values()):
        for name in counts:
            if remaining[name] > 0:
                labels.append(name)
                remaining[name] -= 1
    return labels


def _labels_for_frame_sequence(
    counts: Mapping[Any, int], frames: Sequence[str]
) -> list[Any]:
    """Spread each exact category count across both frame values."""

    direct_counts = {name: int(value) // 2 for name, value in counts.items()}
    direct_target = sum(frame == "direct" for frame in frames)
    extra = direct_target - sum(direct_counts.values())
    odd = [name for name, value in counts.items() if int(value) % 2]
    if extra < 0 or extra > len(odd):
        raise CorpusBuildError("a framed category allocation cannot reconcile")
    for name in odd[:extra]:
        direct_counts[name] += 1
    labels = {
        "direct": _balanced_labels(direct_counts),
        "step": _balanced_labels(
            {
                name: int(counts[name]) - direct_counts[name]
                for name in counts
            }
        ),
    }
    positions = Counter()
    result = []
    for frame in frames:
        result.append(labels[frame][positions[frame]])
        positions[frame] += 1
    return result


def _frame_counts_by_form(
    counts: Mapping[int, int], framing_shares: Mapping[str, Any]
) -> dict[int, dict[str, int]]:
    """Allocate the exact global frame total across trace forms."""

    expected = largest_remainder_counts(sum(counts.values()), framing_shares)
    direct = {form: int(count) // 2 for form, count in counts.items()}
    extra = int(expected["direct"]) - sum(direct.values())
    odd_forms = [form for form, count in counts.items() if count % 2]
    if extra < 0 or extra > len(odd_forms):
        raise CorpusBuildError("the S1.6 frame allocation cannot reconcile")
    for form in odd_forms[:extra]:
        direct[form] += 1
    result = {
        form: {"direct": direct[form], "step": int(count) - direct[form]}
        for form, count in counts.items()
    }
    if Counter(
        {
            frame: sum(values[frame] for values in result.values())
            for frame in ("direct", "step")
        }
    ) != Counter(expected):
        raise CorpusBuildError("the S1.6 frame totals changed")
    return result


def _nth_combination(
    values: Sequence[LayerFact], size: int, index: int
) -> tuple[LayerFact, ...]:
    """Return one indexed combination without materializing all combinations."""

    total = math.comb(len(values), size)
    if total < 1:
        raise CorpusBuildError("a source pool lacks enough combination values")
    rank = index % total
    result: list[LayerFact] = []
    start = 0
    for remaining in range(size, 0, -1):
        for position in range(start, len(values)):
            suffix_count = math.comb(len(values) - position - 1, remaining - 1)
            if rank < suffix_count:
                result.append(values[position])
                start = position + 1
                break
            rank -= suffix_count
    if len(result) != size:
        raise CorpusBuildError("a combination index did not resolve")
    return tuple(result)


def _filter_query_candidates(
    facts: Sequence[LayerFact], field: str
) -> list[tuple[LayerFact, str]]:
    by_value: dict[str, LayerFact] = {}
    for fact in facts:
        for value in _field_components(fact, field):
            by_value.setdefault(value, fact)
    return [(fact, value) for value, fact in sorted(by_value.items())]


def _field_can_make_no_match(
    facts: Sequence[LayerFact], field: str
) -> bool:
    for _, value in _filter_query_candidates(facts, field):
        eligible = [
            item for item in facts if value not in _field_components(item, field)
        ]
        if len(eligible) >= 3:
            return True
    return False


def _filter_pair_labels(
    facts: Sequence[LayerFact], count: int
) -> list[tuple[str, str]]:
    """Balance filter fields and outcomes subject to source feasibility."""

    field_counts = largest_remainder_counts(
        count,
        {field: Fraction(1, len(ONE_FIELD_NAMES)) for field in ONE_FIELD_NAMES},
    )
    outcome_counts = largest_remainder_counts(
        count, {"match": Fraction(1, 2), "no_match": Fraction(1, 2)}
    )
    match_counts = {
        field: (
            field_counts[field]
            if field == "organ" or not _field_can_make_no_match(facts, field)
            else 0
        )
        for field in ONE_FIELD_NAMES
    }
    remaining = outcome_counts["match"] - sum(match_counts.values())
    if remaining < 0:
        raise CorpusBuildError("the filter outcome target is infeasible")
    while remaining:
        changed = False
        for field in ONE_FIELD_NAMES:
            if match_counts[field] < field_counts[field]:
                match_counts[field] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise CorpusBuildError("the filter match allocation cannot reconcile")
    per_field = {
        field: _balanced_labels(
            {
                "match": match_counts[field],
                "no_match": field_counts[field] - match_counts[field],
            }
        )
        for field in ONE_FIELD_NAMES
    }
    positions = Counter()
    pairs: list[tuple[str, str]] = []
    while len(pairs) < count:
        for field in ONE_FIELD_NAMES:
            index = positions[field]
            if index < len(per_field[field]):
                pairs.append((field, per_field[field][index]))
                positions[field] += 1
    if Counter(outcome for _, outcome in pairs) != Counter(outcome_counts):
        raise CorpusBuildError("the filter outcome allocation changed")
    return pairs


def _diverse_list_filter_parts(
    facts: Sequence[LayerFact],
    *,
    field: str,
    outcome: str,
    index: int,
    used: Counter[tuple[Any, ...]],
    independent_choices: bool = False,
) -> tuple[LayerFact, tuple[LayerFact, ...], str]:
    """Balance v5 training companions without changing the filter outcome.

    Family weights make room for many scPEN tags and emphasize HumanNet's
    ``other`` family. Tag usage breaks family ties, so early source tags do
    not become fixed companions. Selection never depends on wrapper count.
    """

    candidates = _filter_query_candidates(facts, field)
    if outcome == "no_match":
        candidates = [
            (fact, value) for fact, value in candidates
            if sum(value not in _field_components(item, field) for item in facts) >= 3
        ]
    if not candidates:
        raise CorpusBuildError(f"the split has no source value for {field}")
    query_fact, value = candidates[index % len(candidates)]
    eligible = [
        fact for fact in facts
        if outcome == "match" or value not in _field_components(fact, field)
    ]
    # Keep homogeneous lists too: the held-out panel often has three scPEN
    # tags. Alternate feasible rows cover that shape with varied tags.
    scpen = [fact for fact in eligible if fact.layer_family == "scpen"]
    # In v8, every query value cycles through both list compositions. The
    # legacy occurrence parity also encoded framing and is kept only for
    # reproduction of published corpora.
    composition_index = index // len(candidates) if independent_choices else index
    if composition_index % 2 == 1 and len(scpen) >= 3 and (
        outcome == "no_match"
        or any(value in _field_components(fact, field) for fact in scpen)
    ):
        eligible = scpen
    weights = {"scpen": 4, "bulk_pen": 2, "other": 2}
    selected: list[LayerFact] = []

    def priority(fact: LayerFact) -> tuple[Fraction, int, str]:
        return (
            Fraction(
                used[("list_family", fact.layer_family)],
                weights.get(fact.layer_family, 1),
            ),
            used[("list_fact", fact.fact_id)],
            fact.fact_id,
        )

    def take(options: Sequence[LayerFact]) -> None:
        fact = min(options, key=priority)
        selected.append(fact)
        used[("list_family", fact.layer_family)] += 1
        used[("list_fact", fact.fact_id)] += 1

    if outcome == "match":
        take([fact for fact in eligible if value in _field_components(fact, field)])
        query_fact = selected[0]
    elif outcome != "no_match":
        raise CorpusBuildError(f"unknown list outcome {outcome!r}")
    humannet = [
        fact for fact in eligible
        if fact.source == "HumanNetV3" and fact not in selected
    ]
    if composition_index % 2 == 0 and humannet and not any(
        fact.source == "HumanNetV3" for fact in selected
    ):
        take(humannet)
    while len(selected) < 2:
        take([fact for fact in eligible if fact not in selected])
    rotation = (index // (2 * len(candidates)) if independent_choices else index) % 3
    # Avoid repeating a content triple even under a different wrapper, so
    # increasing the wrapper bank cannot change the content choices.
    options = sorted(
        (fact for fact in eligible if fact not in selected), key=priority
    )
    for fact in options:
        triple = (*selected, fact)
        ordered = triple[rotation:] + triple[:rotation]
        key = ("list_content", field, value, *(item.fact_id for item in ordered))
        if not used[key]:
            take([fact])
            used[key] += 1
            return query_fact, ordered, value
    raise CorpusBuildError("the training list content combinations are exhausted")


def _list_filter_parts(
    facts: Sequence[LayerFact],
    *,
    field: str,
    outcome: str,
    index: int,
) -> tuple[LayerFact, tuple[LayerFact, LayerFact, LayerFact], str]:
    """Select the published legacy source value and same-split list items."""

    candidates = _filter_query_candidates(facts, field)
    if outcome == "no_match":
        candidates = [
            (fact, value)
            for fact, value in candidates
            if sum(
                value not in _field_components(item, field)
                for item in facts
            )
            >= 3
        ]
    if not candidates:
        raise CorpusBuildError(f"the split has no source value for {field}")
    query_fact, value = candidates[index % len(candidates)]
    if outcome == "match":
        others = [item for item in facts if item.fact_id != query_fact.fact_id]
        pair = _nth_combination(others, 2, index // len(candidates))
        selected = (query_fact, pair[0], pair[1])
    elif outcome == "no_match":
        eligible = [
            item
            for item in facts
            if value not in _field_components(item, field)
        ]
        selected = _nth_combination(eligible, 3, index // len(candidates))
    else:
        raise CorpusBuildError(f"unknown list outcome {outcome!r}")
    rotation = index % 3
    ordered = selected[rotation:] + selected[:rotation]
    return query_fact, ordered, value


def _malformed_tag(
    fact: LayerFact, subtype: str, variant: int = 0
) -> str:
    """Create one deterministic invalid tag from a native source tag."""

    parts = fact.layer_id.split(":")
    prefix = parts[0]
    if subtype == "unrecognized_prefix":
        unknown = ("UnknownLayer", "LayerData", "NativeTag", "LayerRef")[variant % 4]
        return ":".join((unknown, *parts[1:]))
    if subtype == "wrong_arity":
        maximum = 4 if prefix == "bulkPEN" else 5 if prefix == "scPEN" else 2
        added = maximum + 1 - len(parts) + variant % 4
        return ":".join((*parts, *(parts[-1] for _ in range(added))))
    if subtype == "missing_path":
        if prefix == "bulkPEN":
            return ":".join((parts[0], parts[1], "", *parts[2:]))
        if prefix == "scPEN":
            return ":".join((parts[0], parts[1], parts[2], "", *parts[3:]))
        raise CorpusBuildError("a missing-path mutation needs a PEN tag")
    if subtype == "excess_path":
        maximum = 4 if prefix == "bulkPEN" else 5 if prefix == "scPEN" else 2
        added = maximum + 1 - len(parts) + variant % 3
        return ":".join((*parts, *("excess_path" for _ in range(added))))
    if subtype == "synthetic_field":
        synthetic = ("v1", "family", "version", "layer_family")[variant % 4]
        maximum = 4 if prefix == "bulkPEN" else 5 if prefix == "scPEN" else 2
        added = maximum + 1 - len(parts)
        return ":".join((*parts, *(synthetic for _ in range(added))))
    if subtype == "mixed_separator":
        return fact.layer_id.replace(":", "/", 1)
    raise CorpusBuildError(f"unknown malformed subtype {subtype!r}")


def _malformed_candidates(
    pools: Mapping[str, Sequence[Any]], subtype: str
) -> list[tuple[LayerFact, str]]:
    if subtype == "missing_path":
        facts = tuple(pools["bulk_pen"]) + tuple(pools["scpen"])
        variants = 1
    elif subtype == "wrong_arity":
        facts = tuple(pools["all"])
        variants = 4
    else:
        facts = tuple(pools["all"])
        variants = 4 if subtype in {
            "unrecognized_prefix", "excess_path", "synthetic_field"
        } else 1
    unique: dict[str, LayerFact] = {}
    for fact in facts:
        for variant in range(variants):
            value = _malformed_tag(fact, subtype, variant)
            unique.setdefault(value, fact)
    by_prefix: dict[str, list[tuple[LayerFact, str]]] = defaultdict(list)
    for value, fact in sorted(unique.items()):
        by_prefix[fact.layer_id.split(":", 1)[0]].append((fact, value))
    ordered: list[tuple[LayerFact, str]] = []
    positions = Counter()
    prefixes = sorted(by_prefix)
    while len(ordered) < len(unique):
        for prefix in prefixes:
            position = positions[prefix]
            if position < len(by_prefix[prefix]):
                ordered.append(by_prefix[prefix][position])
                positions[prefix] += 1
    return ordered


def _scenario_for_form(
    *,
    split: str,
    form: int,
    occurrence: int,
    framing: str,
    template_index: int,
    pools: Mapping[str, Sequence[Any]],
    field_label: str | None,
    filter_outcome: str | None,
    module_namespace: str | None,
    malformed_subtype: str | None,
    seen_targets: Counter[tuple[Any, ...]],
    frame_occurrence: int,
    diverse_lists: bool = False,
    decorrelate: bool = False,
) -> Scenario:
    all_facts = tuple(pools["all"])
    # Target and request choices follow the inline four-wrapper cycle, so a
    # larger training bank (a multiple of four) changes only the wrapper.
    content_index = template_index % len(LAYER_PROMPTS[split][framing])
    if form in {1, 3, 7}:
        pool_name = {1: "bulk_pen", 3: "humannet", 7: "tf_target"}[form]
        targets = tuple(pools[pool_name])
        if not targets:
            raise CorpusBuildError(f"{split} form {form} has no source tag")
        target = targets[(frame_occurrence // len(LAYER_PROMPTS[split][framing])) % len(targets)]
        key = (form, target.fact_id, framing, content_index)
        request_variant = seen_targets[key]
        seen_targets[key] += 1
        if request_variant >= len(DIRECT_REQUEST_VARIANTS):
            raise CorpusBuildError(
                f"{split} form {form} exhausted direct request variants"
            )
        return Scenario(
            split=split,
            form=form,
            fact=target,
            framing=framing,
            template_index=template_index,
            occurrence=occurrence,
            request_variant=request_variant,
        )
    if form == 2:
        facts = tuple(pools["scpen"])
        if not facts or field_label is None:
            raise CorpusBuildError(f"{split} form 2 lacks its source data")
        fact = facts[(occurrence // len(ONE_FIELD_NAMES)) % len(facts)]
        return Scenario(
            split=split,
            form=form,
            fact=fact,
            framing=framing,
            template_index=template_index,
            occurrence=occurrence,
            requested_field=field_label,
        )
    if form == 4:
        if field_label is None or filter_outcome is None:
            raise CorpusBuildError("form 4 lacks a field or outcome")
        key = (form, field_label, filter_outcome, framing, content_index)
        candidate_index = seen_targets[key]
        seen_targets[key] += 1
        if diverse_lists:
            selection_index = occurrence
            if decorrelate:
                selection_key = ("filter_selection", field_label, filter_outcome)
                selection_index = seen_targets[selection_key]
                seen_targets[selection_key] += 1
            query_fact, selected, value = _diverse_list_filter_parts(
                all_facts,
                field=field_label,
                outcome=filter_outcome,
                index=selection_index,
                used=seen_targets,
                independent_choices=decorrelate,
            )
        else:
            query_fact, selected, value = _list_filter_parts(
                all_facts,
                field=field_label,
                outcome=filter_outcome,
                index=candidate_index,
            )
        return Scenario(
            split=split,
            form=form,
            fact=query_fact,
            framing=framing,
            template_index=template_index,
            occurrence=occurrence,
            companions=selected,
            requested_field=field_label,
            filter_value=value,
            filter_outcome=filter_outcome,
        )
    if form == 5:
        if module_namespace is None:
            raise CorpusBuildError("form 5 lacks a module namespace")
        modules = [
            item for item in pools["modules"] if item.namespace == module_namespace
        ]
        if not modules:
            raise CorpusBuildError(f"{split} lacks {module_namespace} modules")
        namespace_index = occurrence // len(MODULE_NAMESPACES)
        return Scenario(
            split=split,
            form=form,
            fact=modules[namespace_index % len(modules)],
            framing=framing,
            template_index=template_index,
            occurrence=occurrence,
        )
    if form == 6:
        if malformed_subtype is None:
            raise CorpusBuildError("form 6 lacks a malformed subtype")
        candidates = _malformed_candidates(pools, malformed_subtype)
        cursor_key = (form, malformed_subtype)
        cursor = seen_targets[cursor_key]
        seen_targets[cursor_key] += 1
        selected_candidate: tuple[LayerFact, str] | None = None
        for offset in range(len(candidates)):
            fact, malformed_input = candidates[(cursor + offset) % len(candidates)]
            prompt_key = (
                "malformed_prompt", malformed_subtype, framing,
                content_index, malformed_input,
            )
            if not seen_targets[prompt_key]:
                seen_targets[prompt_key] = 1
                selected_candidate = (fact, malformed_input)
                break
        if selected_candidate is None:
            raise CorpusBuildError(
                f"{split} exhausted {malformed_subtype} prompt combinations"
            )
        fact, malformed_input = selected_candidate
        return Scenario(
            split=split,
            form=form,
            fact=fact,
            framing=framing,
            template_index=template_index,
            occurrence=occurrence,
            malformed_subtype=malformed_subtype,
            malformed_input=malformed_input,
        )
    raise CorpusBuildError(f"unsupported S1.6 form {form}")


def _decorrelate_filter_scenarios(
    scenarios: Sequence[Scenario],
    *,
    frame_counts: Mapping[str, int],
    prompts: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[Scenario]:
    """Balance task scope within content strata, then wording within outcomes.

    Content is already selected from the training source. Neither framing
    nor the wrapper bank can change the query or its three source layers.
    """

    groups: dict[tuple[Any, ...], list[Scenario]] = defaultdict(list)
    for item in scenarios:
        groups[(
            item.filter_outcome, item.requested_field, item.filter_value,
            tuple(sorted(fact.layer_family for fact in item.companions)),
            len(item.expected_matching_layer_ids),
        )].append(item)
    ordered_keys = sorted(groups, key=lambda key: _stable_sha256(["filter_scope", key]))
    outcomes = Counter(str(item.filter_outcome) for item in scenarios)
    direct_targets = {outcome: count // 2 for outcome, count in outcomes.items()}
    remaining = int(frame_counts["direct"]) - sum(direct_targets.values())
    for outcome in sorted(outcomes):
        if remaining and outcomes[outcome] % 2:
            direct_targets[outcome] += 1
            remaining -= 1
    extra_direct = {
        outcome: target - sum(len(items) // 2 for key, items in groups.items() if key[0] == outcome)
        for outcome, target in direct_targets.items()
    }
    assigned: list[Scenario] = []
    for key in ordered_keys:
        items = sorted(groups[key], key=lambda item: _stable_sha256([
            "filter_scope_content", item.requested_field, item.filter_value,
            [fact.fact_id for fact in item.companions],
        ]))
        direct_count = len(items) // 2
        if len(items) % 2 and extra_direct[str(key[0])]:
            direct_count += 1
            extra_direct[str(key[0])] -= 1
        frames = _balanced_labels({"direct": direct_count, "step": len(items) - direct_count})
        assigned.extend(replace(item, framing=frame) for item, frame in zip(items, frames, strict=True))
    if remaining or any(extra_direct.values()) or Counter(item.framing for item in assigned) != Counter(frame_counts):
        raise CorpusBuildError("decorrelated filter framing cannot reconcile")

    by_outcome: dict[tuple[str, str], list[Scenario]] = defaultdict(list)
    for item in assigned:
        by_outcome[(item.framing, str(item.filter_outcome))].append(item)
    positions: Counter[str] = Counter()
    result: list[Scenario] = []
    for (frame, outcome), items in sorted(by_outcome.items()):
        items.sort(key=lambda item: _stable_sha256([
            "filter_wording", item.requested_field, item.filter_value,
            [fact.fact_id for fact in item.companions],
        ]))
        for item in items:
            template = positions[frame] % len(prompts["train"][frame])
            positions[frame] += 1
            result.append(replace(item, template_index=template))
    return result


def filter_decorrelation_audit(scenarios: Sequence[Scenario]) -> dict[str, Any]:
    """Reject scope/content shortcuts in the newly selected filter examples."""

    cells: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    wrappers: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    homogeneous: Counter[str] = Counter()
    bulk_empty: Counter[str] = Counter()
    for item in scenarios:
        if item.split != "train" or item.form != 4:
            continue
        families = tuple(sorted(fact.layer_family for fact in item.companions))
        cells[(item.requested_field, item.filter_value, families,
               len(item.expected_matching_layer_ids))][item.framing] += 1
        wrappers[(item.framing, item.template_index)][str(item.filter_outcome)] += 1
        if families == ("scpen", "scpen", "scpen"):
            homogeneous[item.framing] += 1
            if item.requested_field == "layer_family" and item.filter_value == "bulk_pen":
                bulk_empty[item.framing] += 1
    if not cells or any(abs(counts["direct"] - counts["step"]) > 1 for counts in cells.values()):
        raise CorpusBuildError("filter content remains correlated with task scope")
    if any(set(counts) != {"match", "no_match"} for counts in wrappers.values()):
        raise CorpusBuildError("filter wording lacks both outcomes")
    if any(not counts[frame] for counts in (homogeneous, bulk_empty) for frame in ("direct", "step")):
        raise CorpusBuildError("homogeneous scPEN filters lack both task scopes")
    return {
        "pass": True,
        "content_scope_cells": len(cells),
        "maximum_scope_count_difference": max(abs(c["direct"] - c["step"]) for c in cells.values()),
        "homogeneous_scpen_framing_counts": dict(homogeneous),
        "homogeneous_scpen_bulk_pen_empty_framing_counts": dict(bulk_empty),
        "wrapper_outcome_counts": {f"{frame}:{index}": dict(counts) for (frame, index), counts in sorted(wrappers.items())},
    }


def select_scenarios(
    pools: Mapping[str, Mapping[str, Sequence[Any]]],
    counts_by_split: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
    framing_shares: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = LAYER_PROMPTS,
) -> dict[str, list[Scenario]]:
    """Select exact deterministic S1.6 scenarios from split-safe pools."""

    if framing_shares is None:
        framing_shares = {"direct": Fraction(1, 2), "step": Fraction(1, 2)}
    if settings is None:
        settings = load_contract()["curriculum"]["families"]["S1.6"]
    selected: dict[str, list[Scenario]] = {}
    for split in SPLITS:
        counts = counts_by_split[split]
        frame_counts = _frame_counts_by_form(counts, framing_shares)
        scenarios: list[Scenario] = []
        seen_targets: Counter[tuple[Any, ...]] = Counter()
        for form in FORM_BY_BUCKET.values():
            count = int(counts.get(form, 0))
            frames = _balanced_labels(frame_counts[form])
            template_seen: Counter[str] = Counter()
            field_counts = largest_remainder_counts(
                count, {field: Fraction(1, len(ONE_FIELD_NAMES)) for field in ONE_FIELD_NAMES}
            )
            fields = _labels_for_frame_sequence(field_counts, frames)
            raw_filter_pairs = _filter_pair_labels(pools[split]["all"], count)
            filter_pairs = _labels_for_frame_sequence(
                Counter(raw_filter_pairs), frames
            )
            namespace_counts = largest_remainder_counts(
                count,
                {name: Fraction(1, len(MODULE_NAMESPACES)) for name in MODULE_NAMESPACES},
            )
            namespaces = _labels_for_frame_sequence(namespace_counts, frames)
            configured_subtypes = settings["malformed_subtype_counts"]
            configured_total = sum(int(value) for value in configured_subtypes.values())
            subtype_counts = largest_remainder_counts(
                count,
                {
                    name: Fraction(int(configured_subtypes[name]), configured_total)
                    for name in MALFORMED_SUBTYPES
                },
            )
            subtypes = _labels_for_frame_sequence(subtype_counts, frames)
            for occurrence in range(count):
                framing = frames[occurrence]
                template_key = (
                    f"{framing}:{subtypes[occurrence]}"
                    if form == 6 else framing
                )
                template_index = template_seen[template_key] % len(
                    prompts[split][framing]
                )
                template_seen[template_key] += 1
                scenario = _scenario_for_form(
                    split=split,
                    form=form,
                    occurrence=occurrence,
                    framing=framing,
                    template_index=template_index,
                    pools=pools[split],
                    field_label=(
                        filter_pairs[occurrence][0]
                        if form == 4
                        else fields[occurrence] if form == 2 else None
                    ),
                    filter_outcome=(
                        filter_pairs[occurrence][1] if form == 4 else None
                    ),
                    module_namespace=namespaces[occurrence] if form == 5 else None,
                    malformed_subtype=subtypes[occurrence] if form == 6 else None,
                    seen_targets=seen_targets,
                    diverse_lists=(
                        split == "train"
                        and bool(settings.get("train_diverse_lists"))
                    ),
                    decorrelate=(split == "train" and bool(settings.get("train_decorrelate"))),
                    frame_occurrence=(
                        template_seen[framing] - 1
                        if form != 6 else occurrence
                    ),
                )
                if (
                    split == "train" and form != 5
                    and settings.get("train_layer_reasoning")
                ):
                    scenario = replace(scenario, train_reasoning=True)
                if split == "train" and form in {1, 2, 3, 4, 6} and settings.get("train_targeted_v9"):
                    scenario = replace(scenario, train_targeted_v9=True)
                scenarios.append(scenario)
        if split == "train" and settings.get("train_decorrelate"):
            filters = _decorrelate_filter_scenarios(
                [item for item in scenarios if item.form == 4],
                frame_counts=frame_counts[4], prompts=prompts,
            )
            scenarios = [item for item in scenarios if item.form != 4] + filters
        scenarios.sort(
            key=lambda item: _stable_sha256(
                {"seed": seed, "split": split, "scenario": item.scenario_id}
            )
        )
        if len(scenarios) != sum(int(value) for value in counts.values()):
            raise CorpusBuildError(f"{split} scenario count changed")
        selected[split] = scenarios
    return selected


def _row_id(scenario: Scenario, schema_version: str) -> str:
    return "wm3_s1_6_" + _stable_sha256(
        {"schema_version": schema_version, "scenario_id": scenario.scenario_id}
    )[:24]


def _request(scenario: Scenario) -> str:
    if scenario.form in {1, 3, 7}:
        if not isinstance(scenario.fact, LayerFact):
            raise CorpusBuildError("a direct form lacks a layer fact")
        if not 0 <= scenario.request_variant < len(DIRECT_REQUEST_VARIANTS):
            raise CorpusBuildError("a direct request variant is out of range")
        return DIRECT_REQUEST_VARIANTS[scenario.request_variant].format(
            layer_id=scenario.fact.layer_id
        )
    if scenario.form == 2:
        if not isinstance(scenario.fact, LayerFact) or scenario.requested_field is None:
            raise CorpusBuildError("form 2 lacks a field request")
        field = scenario.requested_field.replace("_", " ")
        return f"report the exact {field} field for layer {scenario.fact.layer_id}"
    if scenario.form == 4:
        if (
            not isinstance(scenario.fact, LayerFact)
            or scenario.requested_field is None
            or scenario.filter_value is None
            or len(scenario.companions) != 3
        ):
            raise CorpusBuildError("form 4 lacks its filter data")
        field = scenario.requested_field.replace("_", " ")
        layer_ids = ", ".join(item.layer_id for item in scenario.companions)
        return (
            f"filter three layers by exact {field}: {scenario.filter_value}. "
            f"Tags: {layer_ids}"
        )
    if scenario.form == 5:
        if not isinstance(scenario.fact, ModuleFact):
            raise CorpusBuildError("form 5 lacks a module fact")
        return f"parse this layer tag: {scenario.fact.module_id}"
    if scenario.form == 6:
        if scenario.malformed_input is None:
            raise CorpusBuildError("form 6 lacks its invalid tag")
        return f"parse this layer tag: {scenario.malformed_input}"
    raise CorpusBuildError(f"unsupported S1.6 form {scenario.form}")


def _prompt(
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = LAYER_PROMPTS,
) -> str:
    templates = prompts[scenario.split][scenario.framing]
    if not 0 <= scenario.template_index < len(templates):
        raise CorpusBuildError("a layer prompt template index is out of range")
    request = _request(scenario).rstrip(".!?")
    return templates[scenario.template_index].format(request=request)


def _layer_thinking(fact: LayerFact) -> str:
    """Explain the exact grammar walk for one native layer tag."""

    parts = fact.layer_id.split(":")
    if fact.source == "HumanNetV3":
        return (
            "The prefix HumanNetV3 selects the HumanNet grammar. "
            f"The second part is network {fact.network}. "
            f"That network gives layer family {fact.layer_family}."
        )
    if fact.layer_family == "bulk_pen":
        path = ", then ".join(fact.region_path or ())
        return (
            "The prefix bulkPEN selects the bulk PEN grammar. "
            f"The next part is organ {fact.organ}. "
            f"The remaining region path is {path}."
        )
    if fact.layer_family == "scpen":
        cells = ", then ".join(fact.cell_type_path or ())
        return (
            "The prefix scPEN selects the single-cell PEN grammar. "
            f"The next parts are organ {fact.organ} and region "
            f"{fact.region_path[0]}. The remaining cell path is {cells}."
        )
    if fact.layer_family == "tf_target":
        return (
            "The prefix TFs selects the transcription-factor grammar. "
            f"The second part is organ {fact.organ}. The tag has no path parts."
        )
    raise CorpusBuildError(f"unknown layer family {fact.layer_family!r}")


def _parse_thinking(scenario: Scenario) -> str:
    """Explain the parse using the published legacy reasoning."""

    if scenario.form in {1, 2, 3, 7}:
        if not isinstance(scenario.fact, LayerFact):
            raise CorpusBuildError("a resolved form lacks a layer fact")
        return _layer_thinking(scenario.fact)
    if scenario.form == 4:
        thoughts = [_layer_thinking(item) for item in scenario.companions]
        field = str(scenario.requested_field).replace("_", " ")
        thoughts.append(
            f"I compare the exact {field} components with {scenario.filter_value}."
        )
        return " ".join(thoughts)
    if scenario.form == 5:
        if not isinstance(scenario.fact, ModuleFact):
            raise CorpusBuildError("form 5 lacks a module fact")
        return (
            f"The prefix {scenario.fact.namespace} belongs to the module ID "
            "namespace, not a layer grammar. I will not create layer fields."
        )
    if scenario.form == 6:
        if not isinstance(scenario.fact, LayerFact):
            raise CorpusBuildError("form 6 lacks its native source tag")
        subtype = scenario.malformed_subtype
        explanations = {
            "unrecognized_prefix": "The first part does not match a layer grammar prefix.",
            "wrong_arity": "The prefix is known, but the tag has the wrong number of parts.",
            "missing_path": "The known PEN grammar requires a path part that is absent.",
            "excess_path": "The known grammar permits fewer path parts than this tag has.",
            "synthetic_field": "The native grammar has no family or version field at the end.",
            "mixed_separator": "The tag mixes separators, so its prefix does not match a layer grammar.",
        }
        if subtype not in explanations:
            raise CorpusBuildError("form 6 has an unknown invalid subtype")
        return explanations[subtype] + " I will not infer the missing fields."
    raise CorpusBuildError(f"unsupported S1.6 form {scenario.form}")


def _list_parse_steps(scenario: Scenario) -> str:
    """Derive each complete parse and exact comparison from the visible tags."""

    steps = []
    for fact in scenario.companions:
        fields = ", ".join(
            f"{key} {json.dumps(value, ensure_ascii=False)}"
            for key, value in fact.layer_ref.items() if key != "layer_id"
        )
        components = _field_components(fact, str(scenario.requested_field))
        verdict = "matches" if scenario.filter_value in components else "does not match"
        steps.append(
            f"For {fact.layer_id}, {_layer_thinking(fact)} "
            f"The complete values are {fields}. "
            f"The requested field has exact components {json.dumps(list(components), ensure_ascii=False)}, "
            f"so this tag {verdict} the filter."
        )
    return " ".join(steps)


def _malformed_check(scenario: Scenario) -> str:
    """Explain the visible prefix, arity, or empty component that rejects a tag."""

    parts = str(scenario.malformed_input).split(":")
    prefix = parts[0]
    bounds = {"HumanNetV3": (2, 2), "TFs": (2, 2), "bulkPEN": (3, 4), "scPEN": (4, 5)}
    if prefix not in bounds:
        return (
            f"The first colon-separated component {prefix} is not a recognized prefix. "
            "Only HumanNetV3, bulkPEN, scPEN, and TFs select a native layer grammar."
        )
    minimum, maximum = bounds[prefix]
    count = str(minimum) if minimum == maximum else f"{minimum} to {maximum}"
    rule = f"The {prefix} grammar requires {count} nonempty colon-separated components."
    empty = [str(index + 1) for index, value in enumerate(parts) if not value]
    if empty:
        return f"{rule} Component {', '.join(empty)} is empty, so a required path component is missing."
    return (
        f"{rule} This input has {len(parts)} components, outside the permitted arity. "
        "Repeated components and appended family or version labels do not extend that grammar."
    )


def _thinking(
    scenario: Scenario, *, bulk_pen_reasoning: Mapping[str, str] | None = None,
    scpen_reasoning: Mapping[str, str] | None = None,
    humannet_reasoning: Mapping[str, str] | None = None,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> str:
    """Explain the parse and decision, applying reviewed templates only in training."""

    if scenario.split == "train" and scenario.train_targeted_v9:
        return _targeted_v9_thinking(scenario)

    if scenario.split == "train" and final_reasoning and scenario.form in final_reasoning:
        if scenario.form == 4:
            bindings = {
                "requested_field": scenario.requested_field,
                "filter_value": json.dumps(scenario.filter_value, ensure_ascii=False),
                "parse_steps": _list_parse_steps(scenario),
                "matching_layer_ids": json.dumps(list(scenario.expected_matching_layer_ids), ensure_ascii=False),
            }
        elif scenario.form == 6:
            bindings = {"input": scenario.malformed_input, "grammar_check": _malformed_check(scenario)}
        else:
            raise CorpusBuildError("S1.6 final reasoning supports only forms 4 and 6")
        return final_reasoning[scenario.form][scenario.framing].format(**bindings)

    if scenario.split == "train" and scenario.form == 1 and bulk_pen_reasoning:
        fact = scenario.fact
        if not isinstance(fact, LayerFact) or fact.layer_family != "bulk_pen":
            raise CorpusBuildError("S1.6.1 reasoning requires a bulkPEN layer")
        return bulk_pen_reasoning[scenario.framing].format(
            layer_id=fact.layer_id,
            organ=fact.organ,
            region_path=json.dumps(list(fact.region_path), ensure_ascii=False),
        )
    if scenario.split == "train" and scenario.form == 2 and scpen_reasoning:
        fact = scenario.fact
        if not isinstance(fact, LayerFact) or fact.layer_family != "scpen":
            raise CorpusBuildError("S1.6.2 reasoning requires an scPEN layer")
        if scenario.requested_field not in ONE_FIELD_NAMES:
            raise CorpusBuildError("S1.6.2 reasoning requires a supported field")
        result = fact.layer_ref
        return scpen_reasoning[scenario.framing].format(
            layer_id=fact.layer_id,
            organ=fact.organ,
            region_path=json.dumps(result["region_path"], ensure_ascii=False),
            cell_type_path=json.dumps(result["cell_type_path"], ensure_ascii=False),
            requested_field=scenario.requested_field,
            requested_value=json.dumps(result[scenario.requested_field], ensure_ascii=False),
        )
    if scenario.split == "train" and scenario.form == 3 and humannet_reasoning:
        fact = scenario.fact
        if not isinstance(fact, LayerFact) or fact.source != "HumanNetV3":
            raise CorpusBuildError("S1.6.3 reasoning requires a HumanNet layer")
        return humannet_reasoning[scenario.framing].format(
            layer_id=fact.layer_id, network=fact.network, layer_family=fact.layer_family,
        )

    thought = _parse_thinking(scenario)
    if not scenario.train_reasoning or scenario.form == 5:
        return thought
    if scenario.form == 4:
        thoughts = [
            "This request contains three layer tags, so I parse each one in the supplied order."
        ]
        thoughts.extend(
            f"For layer {fact.layer_id}, {_layer_thinking(fact)}"
            for fact in scenario.companions
        )
        field = str(scenario.requested_field).replace("_", " ")
        thoughts.extend((
            "I keep each complete parse, including fields that the tag leaves empty.",
            f"I compare the exact {field} components with {scenario.filter_value}.",
            "The filter selects matching IDs without removing any parsed layer.",
            f"There are {len(scenario.expected_matching_layer_ids)} exact matches.",
        ))
        if not scenario.expected_matching_layer_ids:
            thoughts.append("An empty match list still answers the request.")
        thought = " ".join(thoughts)
    if scenario.form == 6:
        return (
            f"{thought} I copy the supplied input {scenario.malformed_input} exactly; "
            "spelling and repeated parts stay unchanged. The tag cannot be used, so I defer."
        )
    decision = (
        "A later action uses this answer, so I continue."
        if scenario.framing == "step"
        else "This request is complete after the parse, so I stop."
    )
    return f"{thought} {decision}"


def _targeted_v9_thinking(scenario: Scenario) -> str:
    """Derive complete concrete parses before considering independent task scope."""

    if scenario.form == 6:
        return (
            f"The literal supplied input is {scenario.malformed_input}. "
            f"{_malformed_check(scenario)} "
            "Do not repair or complete the tag. Return invalid_input and defer because the grammar is invalid; "
            "this outcome applies whether the request is standalone or part of a workflow."
        )
    facts = scenario.companions if scenario.form == 4 else (scenario.fact,)
    thoughts = []
    for index, fact in enumerate(facts, start=1):
        if not isinstance(fact, LayerFact):
            raise CorpusBuildError("v9 layer reasoning requires native layer facts")
        values = ", ".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}"
            for key, value in fact.layer_ref.items()
        )
        thoughts.append(f"Input {index} is {fact.layer_id}. {_layer_thinking(fact)} The seven parsed fields are {values}.")
        if scenario.form == 4:
            components = _field_components(fact, str(scenario.requested_field))
            verdict = "matches" if scenario.filter_value in components else "does not match"
            thoughts.append(
                f"Its {scenario.requested_field} components are {json.dumps(list(components), ensure_ascii=False)}; "
                f"an exact comparison with {json.dumps(scenario.filter_value, ensure_ascii=False)} {verdict}."
            )
    thoughts.append("Preserve component spelling and list order. Null fields are expected where the selected grammar supplies no value.")
    if scenario.form == 2:
        value = scenario.fact.layer_ref[str(scenario.requested_field)]
        thoughts.append(f"The requested field is {scenario.requested_field}, with value {json.dumps(value, ensure_ascii=False)}; retain the complete seven-field parse.")
    elif scenario.form == 4:
        matching = list(scenario.expected_matching_layer_ids)
        remaining = [fact.layer_id for fact in scenario.companions if fact.layer_id not in matching]
        thoughts.append(
            f"In input order, matching_layer_ids is {json.dumps(matching, ensure_ascii=False)}. "
            f"The remaining input IDs are {json.dumps(remaining, ensure_ascii=False)}. "
            "Keep every complete parse in layers. An empty, partial, or complete match list still completes the filter request."
        )
    thoughts.append("The supplied syntax determines the requested result, so return resolved with no blocked reason.")
    thoughts.append(
        "The user needs this result for a later workflow step, so continue; match count and null fields do not change that scope."
        if scenario.framing == "step" else
        "The user requested this result as a standalone task, so stop; match count and null fields do not change that scope."
    )
    return " ".join(thoughts)


def _decision(scenario: Scenario, *, valid: bool) -> str:
    if not valid:
        return "defer"
    return "continue" if scenario.framing == "step" else "stop"


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


def _resolved_answer(scenario: Scenario) -> dict[str, Any]:
    if scenario.form in {1, 2, 3, 7}:
        if not isinstance(scenario.fact, LayerFact):
            raise CorpusBuildError("a resolved form lacks a layer fact")
        result = scenario.fact.layer_ref
        if scenario.form == 2:
            field = str(scenario.requested_field)
            value = result[field]
            summary = (
                f"<LAYER:{scenario.fact.layer_id}> has {field} equal to "
                f"{_canonical_json(value, sort_keys=False)}."
            )
        else:
            summary = (
                f"<LAYER:{scenario.fact.layer_id}> parses as layer family "
                f"{scenario.fact.layer_family}. The tag supplies only the returned fields."
            )
        return _answer(
            status="resolved",
            result=result,
            decision=_decision(scenario, valid=True),
            blocked_reason=None,
            summary=summary,
        )
    if scenario.form == 4:
        layers = [item.layer_ref for item in scenario.companions]
        matching = list(scenario.expected_matching_layer_ids)
        result = {
            "layers": layers,
            "filter": {
                "field": scenario.requested_field,
                "value": scenario.filter_value,
            },
            "matching_layer_ids": matching,
        }
        if matching:
            summary = (
                f"The exact {scenario.requested_field} filter matches "
                + ", ".join(f"<LAYER:{layer_id}>" for layer_id in matching)
                + "."
            )
        else:
            summary = (
                f"No listed layer has exact {scenario.requested_field} value "
                f"{scenario.filter_value}."
            )
        return _answer(
            status="resolved",
            result=result,
            decision=_decision(scenario, valid=True),
            blocked_reason=None,
            summary=summary,
        )
    raise CorpusBuildError(f"form {scenario.form} is not a resolved form")


def _invalid_answer(scenario: Scenario) -> dict[str, Any]:
    if scenario.form == 5:
        if not isinstance(scenario.fact, ModuleFact):
            raise CorpusBuildError("form 5 lacks a module fact")
        input_value = scenario.fact.module_id
        blocked_reason = f"The input {input_value} is a module ID, not a layer tag."
        summary = (
            "The input is a module ID, not a layer tag. "
            "Give a layer tag to continue."
        )
    elif scenario.form == 6:
        if scenario.malformed_input is None:
            raise CorpusBuildError("form 6 lacks an invalid input")
        input_value = scenario.malformed_input
        blocked_reason = (
            f"The input {input_value} is not a well-formed layer tag."
        )
        summary = (
            "The input does not follow a native layer grammar. "
            "Give a complete native layer tag to continue."
        )
    else:
        raise CorpusBuildError(f"form {scenario.form} is not an invalid form")
    return _answer(
        status="invalid_input",
        result={"input": input_value, "expected": "layer_tag"},
        decision="defer",
        blocked_reason=blocked_reason,
        summary=summary,
    )


def build_row(
    scenario: Scenario,
    *,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = LAYER_PROMPTS,
    bulk_pen_reasoning: Mapping[str, str] | None = None,
    scpen_reasoning: Mapping[str, str] | None = None,
    humannet_reasoning: Mapping[str, str] | None = None,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Render one closed-book S1.6 row without a tool call."""

    schema_version = str(contract["runtime"]["row_schema_version"])
    row_id = _row_id(scenario, schema_version)
    answer = (
        _invalid_answer(scenario)
        if scenario.form in {5, 6}
        else _resolved_answer(scenario)
    )
    metadata: dict[str, Any] = {
        "book_mode": "closed_book",
        "step": "S1",
        "question_family": "layer_tag_parsing",
        "form": scenario.form,
        "source_trace": scenario.source_trace,
        "split_unit": scenario.metadata_split_unit,
        "blocked_reason_code": (
            "wrong_input_type" if scenario.form == 5 else
            "invalid_input" if scenario.form == 6 else None
        ),
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
        "row_id": row_id,
        "messages": [
            {"role": "system", "content": contract["runtime"]["system_prompt"]},
            {"role": "user", "content": _prompt(scenario, prompts=prompts)},
            {
                "role": "assistant",
                "thinking": _thinking(
                    scenario, bulk_pen_reasoning=bulk_pen_reasoning,
                    scpen_reasoning=scpen_reasoning,
                    humannet_reasoning=humannet_reasoning,
                    final_reasoning=final_reasoning,
                ),
                "content": answer,
            },
        ],
        "tools": tool_definitions(contract) if tools is None else tools,
        "metadata": metadata,
    }
    return row, []


THINKING_FILTER_RULES = (
    "one thinking step per row",
    "thinking matches the source-derived explanation",
    "no source value before it is visible",
    "no tool call or tool-result claim",
    "no invented layer, graph, or multiplex fact",
    "status and decision agree with the answer",
    "no answer-schema or provenance discussion",
)
_ANSWER_SCHEMA_PHRASES = (
    "answer schema",
    "json schema",
    "answer fields",
    "status field",
    "provenance field",
    "blocked_reason field",
    "continuation_decision field",
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


def filter_row_thinking(
    row: Mapping[str, Any], scenario: Scenario, *,
    bulk_pen_reasoning: Mapping[str, str] | None = None,
    scpen_reasoning: Mapping[str, str] | None = None,
    humannet_reasoning: Mapping[str, str] | None = None,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> int:
    """Reject a thought that violates an S1.6 runtime rule."""

    if _tool_calls(row) or any(
        message.get("role") == "tool" for message in row["messages"]
    ):
        raise CorpusBuildError(f"row {row['row_id']} contains a tool action")
    assistants = [
        message for message in row["messages"] if message.get("role") == "assistant"
    ]
    if len(assistants) != 1 or not isinstance(assistants[0].get("thinking"), str):
        raise CorpusBuildError(f"row {row['row_id']} lacks one thinking step")
    thought = str(assistants[0]["thinking"])
    if not thought.strip():
        raise CorpusBuildError(f"row {row['row_id']} has an empty thinking step")
    if thought != _thinking(
        scenario, bulk_pen_reasoning=bulk_pen_reasoning, scpen_reasoning=scpen_reasoning,
        humannet_reasoning=humannet_reasoning,
        final_reasoning=final_reasoning,
    ):
        raise CorpusBuildError(
            f"row {row['row_id']} changes its source-derived thinking"
        )
    lowered = " ".join(thought.casefold().replace("-", " ").split())
    if any(phrase in lowered for phrase in _ANSWER_SCHEMA_PHRASES):
        raise CorpusBuildError(f"row {row['row_id']} discusses the answer schema")
    if (
        "provenance" in lowered
        or "tool result" in lowered
        or "tool call" in lowered
        or "registry tool" in lowered
        or re.search(r"\b(call|use|query|lookup)\b.*\b(tool|registry)\b", lowered)
    ):
        raise CorpusBuildError(f"row {row['row_id']} discusses unavailable data")
    if "synthetic version" in lowered or "prompt does not show" in lowered:
        raise CorpusBuildError(f"row {row['row_id']} invents a tag field")
    if "edge" in lowered or "multiplex contains" in lowered:
        raise CorpusBuildError(f"row {row['row_id']} invents a graph claim")
    user = next(
        str(message["content"])
        for message in row["messages"]
        if message.get("role") == "user"
    )
    tag_pattern = r"[A-Za-z][A-Za-z0-9_-]*(?::[^\s,?.\"\[\]]+)+"
    visible_tags = set(re.findall(tag_pattern, user))
    for tag in re.findall(tag_pattern, thought):
        if tag not in visible_tags:
            raise CorpusBuildError(f"row {row['row_id']} invents the tag {tag}")
    answer = _final_answer(row)
    if scenario.form in {5, 6}:
        if answer.get("status") != "invalid_input" or answer.get(
            "continuation_decision"
        ) != "defer":
            raise CorpusBuildError(f"row {row['row_id']} has an invalid failure claim")
        if any(
            phrase in lowered
            for phrase in ("is a valid layer", "parses as layer", "well formed layer")
        ):
            raise CorpusBuildError(f"row {row['row_id']} accepts an invalid input")
    else:
        if answer.get("status") != "resolved":
            raise CorpusBuildError(f"row {row['row_id']} rejects a valid tag")
        if answer.get("continuation_decision") != _decision(scenario, valid=True):
            raise CorpusBuildError(f"row {row['row_id']} has the wrong decision")
    return 1


def _current_user_message(row: Mapping[str, Any]) -> str:
    users = [
        message for message in row["messages"] if message.get("role") == "user"
    ]
    if len(users) != 1 or not isinstance(users[0].get("content"), str):
        raise CorpusBuildError(f"row {row['row_id']} lacks one user message")
    return str(users[0]["content"])


def _source_values(scenario: Scenario) -> set[str]:
    values: set[str] = set()
    if isinstance(scenario.fact, LayerFact):
        values.add(scenario.fact.layer_id)
        values.update(
            str(value)
            for field in ONE_FIELD_NAMES
            for value in _field_components(scenario.fact, field)
        )
    else:
        values.add(scenario.fact.module_id)
    for fact in scenario.companions:
        values.add(fact.layer_id)
        values.update(
            str(value)
            for field in ONE_FIELD_NAMES
            for value in _field_components(fact, field)
        )
    if scenario.filter_value is not None:
        values.add(scenario.filter_value)
    if scenario.malformed_input is not None:
        values.add(scenario.malformed_input)
    return {value for value in values if value}


def _mask_value(text: str, value: str, replacement: str) -> str:
    return re.sub(re.escape(value), replacement, text, flags=re.IGNORECASE)


def masked_visible_prompt(
    row: Mapping[str, Any], scenario: Scenario
) -> str:
    """Mask source values but retain the full visible prompt structure."""

    last_user = max(
        index
        for index, message in enumerate(row["messages"])
        if message.get("role") == "user"
    )
    parts = []
    for message in row["messages"][1 : last_user + 1]:
        rendered = message.get("content", "")
        if not isinstance(rendered, str):
            rendered = _canonical_json(rendered, sort_keys=False)
        parts.append(f"{message['role']}: " + " ".join(rendered.split()))
    visible = "\n".join(parts)
    for value in sorted(_source_values(scenario), key=len, reverse=True):
        visible = _mask_value(visible, value, MASK_TOKEN)
    return visible


def _masked_user_wording(
    row: Mapping[str, Any],
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = LAYER_PROMPTS,
) -> str:
    user = _current_user_message(row)
    template = prompts[scenario.split][scenario.framing][
        scenario.template_index
    ]
    prefix, suffix = template.split("{request}")
    if not user.startswith(prefix) or (suffix and not user.endswith(suffix)):
        raise CorpusBuildError(
            f"row {row['row_id']} does not match its layer template"
        )
    return " ".join(template.format(request=MASK_TOKEN).casefold().split())


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

    if not 0.0 <= tolerance <= 1.0:
        raise CorpusBuildError("the wording tolerance is invalid")
    labels = tuple(
        sorted({scenario.form for scenario in scenarios_by_split["train"]})
    )
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
                count
                * math.log((token_counts[label][token] + 1) / denominator)
                for token, count in known.items()
            )
            scores.append((score, -label, label))
        return max(scores)[2]

    evaluations: dict[str, dict[str, Any]] = {}
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
        raise CorpusBuildError(
            "masked user wording predicts the S1.6 form above the limit"
        )
    return result


def masked_wording_classifier_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    tolerance: float = MASKED_WORDING_TOLERANCE,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = LAYER_PROMPTS,
) -> dict[str, Any]:
    """Gate leakage from the split-held-out outer wrappers."""

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


def _prompt_and_outcome_signatures(
    row: Mapping[str, Any]
) -> tuple[str, str, str]:
    messages = row["messages"]
    last_user = max(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    )
    prompt = messages[: last_user + 1]
    outcomes = [
        {
            "tool_name": message["name"],
            "result_status": message["content"]["status"],
        }
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
    """Reject every repeated closed-book prompt and outcome."""

    split_results: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        prompts: set[str] = set()
        combined: set[str] = set()
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        for row, _ in zip(rows, scenarios, strict=True):
            prompt, _, signature = _prompt_and_outcome_signatures(row)
            if prompt in prompts or signature in combined:
                raise CorpusBuildError(
                    f"{split} repeats a prompt and closed-book outcome"
                )
            prompts.add(prompt)
            combined.add(signature)
        split_results[split] = {
            "row_count": len(rows),
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


def _context_strata(
    facts: Iterable[LayerFact],
) -> tuple[dict[str, int], set[str]]:
    families: dict[str, set[str]] = defaultdict(set)
    for fact in facts:
        families[fact.context_key].add(
            "humannet" if fact.source == "HumanNetV3" else fact.layer_family
        )
    counts: Counter[str] = Counter()
    for key, values in families.items():
        if key.startswith("network:"):
            counts["humannet"] += 1
        elif key.startswith("tf:"):
            counts["tf_target"] += 1
        elif values == {"bulk_pen"}:
            counts["bulk_only"] += 1
        elif values == {"scpen"}:
            counts["scpen_only"] += 1
        elif values == {"bulk_pen", "scpen"}:
            counts["shared_pen"] += 1
        else:
            raise CorpusBuildError(f"unknown context stratum for {key}")
    return (
        {
            key: counts[key]
            for key in (
                "humannet", "bulk_only", "scpen_only", "shared_pen", "tf_target"
            )
        },
        set(families),
    )


def source_pool_coverage(
    pools: Mapping[str, Mapping[str, Sequence[Any]]]
) -> dict[str, Any]:
    """Summarize exact source pools and context strata."""

    split_pools: dict[str, dict[str, int]] = {}
    context_counts: dict[str, int] = {}
    strata: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_pools[split] = {
            name: len(pools[split][name])
            for name in ("all", "bulk_pen", "scpen", "humannet", "tf_target", "modules")
        }
        split_strata, contexts = _context_strata(pools[split]["all"])
        strata[split] = split_strata
        context_counts[split] = len(contexts)
    return {
        "layer_pool_counts": split_pools,
        "context_counts": context_counts,
        "context_stratum_counts": strata,
        "native_layer_count": sum(
            len(pools[split]["all"]) for split in SPLITS
        ),
    }


def _sentence_word_counts(text: str) -> list[int]:
    """Count prose words while each code identifier counts as one token."""

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
    """Retain prompt and answer prose limits while reporting uncapped reasoning."""

    splits: dict[str, dict[str, int | bool]] = {}
    for split in SPLITS:
        maximum_instruction = 0
        maximum_description = 0
        violations = 0
        for row in rows_by_split[split]:
            user = _current_user_message(row)
            instruction_counts = _sentence_word_counts(user)
            maximum_instruction = max(
                maximum_instruction, max(instruction_counts, default=0)
            )
            if any(count > 20 for count in instruction_counts):
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
            description_counts = [
                count
                for value in descriptions
                for count in _sentence_word_counts(value)
            ]
            maximum_description = max(
                maximum_description, max(description_counts, default=0),
                max(_sentence_word_counts(str(assistant["thinking"])), default=0),
            )
            if any(count > 25 for count in description_counts):
                violations += 1
        if violations:
            raise CorpusBuildError(f"{split} violates the prose length limits")
        splits[split] = {
            "row_count": len(rows_by_split[split]),
            "maximum_instruction_words": maximum_instruction,
            "maximum_description_words": maximum_description,
            "violation_count": violations,
            "pass": True,
        }
    return {"splits": splits, "pass": True}


def audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    expected_counts: Mapping[str, Mapping[int, int]],
    *,
    framing_shares: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = LAYER_PROMPTS,
) -> dict[str, Any]:
    """Audit rendered rows against all S1.6 generation invariants."""

    if contract is None:
        contract = load_contract()
    if settings is None:
        settings = contract["curriculum"]["families"]["S1.6"]
    if framing_shares is None:
        framing_shares = {"direct": Fraction(1, 2), "step": Fraction(1, 2)}
    row_ids: set[str] = set()
    context_sets: dict[str, set[str]] = {}
    status_counts: dict[str, dict[str, int]] = {}
    tool_counts: dict[str, dict[str, int]] = {}
    framing_counts: dict[str, dict[str, int]] = {}
    template_counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    field_counts: dict[str, dict[str, dict[str, int]]] = {}
    filter_counts: dict[str, dict[str, int]] = {}
    namespace_counts: dict[str, dict[str, int]] = {}
    subtype_counts: dict[str, dict[str, int]] = {}
    semantic_counts: dict[str, dict[str, int]] = {}
    context_counts: dict[str, int] = {}
    stratum_counts: dict[str, dict[str, int]] = {}
    thinking_counts: dict[str, dict[str, int | bool]] = {}
    structure_counts: dict[str, dict[str, int | bool]] = {}
    malformed_coverage: dict[str, dict[str, Any]] = {}
    form_gates: dict[str, dict[str, dict[str, Any]]] = {}
    field_gate: dict[str, dict[str, int | float | bool]] = {}
    confusion: dict[str, dict[str, dict[str, int]]] = {}
    total_false_defers = 0
    total_answerable = 0

    for split in SPLITS:
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} row and scenario counts differ")
        actual_forms = Counter(scenario.form for scenario in scenarios)
        if {
            form: actual_forms[form] for form in expected_counts[split]
        } != {
            form: int(value) for form, value in expected_counts[split].items()
        }:
            raise CorpusBuildError(f"{split} form counts changed")
        actual_frames = Counter(scenario.framing for scenario in scenarios)
        expected_frames = largest_remainder_counts(len(rows), framing_shares)
        if dict(actual_frames) != expected_frames:
            raise CorpusBuildError(f"{split} framing counts changed")

        statuses: Counter[str] = Counter()
        calls: Counter[str] = Counter()
        templates: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        one_fields: Counter[str] = Counter()
        list_fields: Counter[str] = Counter()
        filters: Counter[str] = Counter()
        namespaces: Counter[str] = Counter()
        subtypes: Counter[str] = Counter()
        semantics: Counter[str] = Counter()
        gates: dict[str, Counter[str]] = defaultdict(Counter)
        facts_by_id: dict[str, LayerFact] = {}
        field_correct = 0
        field_total = 0
        split_confusion: dict[str, Counter[str]] = defaultdict(Counter)
        thinking_steps = 0
        harness_count = 0
        direct_one_tag_count = 0
        list_three_tag_count = 0
        no_call_count = 0
        malformed_inputs: set[str] = set()
        malformed_inputs_by_subtype: dict[str, set[str]] = defaultdict(set)
        malformed_prefixes: dict[str, Counter[str]] = defaultdict(Counter)
        split_false_defers = 0
        split_answerable = 0

        for row, scenario in zip(rows, scenarios, strict=True):
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise CorpusBuildError(f"duplicate row ID {row_id}")
            row_ids.add(row_id)
            form_key = f"S1.6.{scenario.form}"
            gates[form_key]["row_count"] += 1
            semantics[scenario.semantic_bucket] += 1
            templates[form_key][scenario.framing][str(scenario.template_index)] += 1
            if any(message.get("role") == "harness" for message in row["messages"]):
                harness_count += 1
            row_calls = _tool_calls(row)
            calls[str(len(row_calls))] += 1
            if not row_calls:
                no_call_count += 1
            else:
                raise CorpusBuildError(f"row {row_id} makes a forbidden tool call")
            gates[form_key]["exact_call_count"] += 1

            expected = (
                _invalid_answer(scenario)
                if scenario.form in {5, 6}
                else _resolved_answer(scenario)
            )
            answer = _final_answer(row)
            if answer.get("status") != expected["status"]:
                raise CorpusBuildError(f"row {row_id} has the wrong status")
            gates[form_key]["exact_status_count"] += 1
            if answer.get("provenance") is not None:
                raise CorpusBuildError(f"row {row_id} has closed-book provenance")
            gates[form_key]["null_provenance_count"] += 1
            if answer.get("result") != expected["result"]:
                raise CorpusBuildError(f"row {row_id} has the wrong result")
            gates[form_key]["exact_result_count"] += 1
            if answer.get("continuation_decision") != expected[
                "continuation_decision"
            ]:
                raise CorpusBuildError(f"row {row_id} has the wrong decision")
            gates[form_key]["exact_decision_count"] += 1
            if answer.get("blocked_reason") != expected["blocked_reason"]:
                raise CorpusBuildError(f"row {row_id} has the wrong blocked reason")
            gates[form_key]["exact_blocked_reason_count"] += 1
            statuses[str(answer["status"])] += 1
            if scenario.form not in {5, 6}:
                split_answerable += 1
                if answer["continuation_decision"] == "defer":
                    split_false_defers += 1

            thinking_steps += filter_row_thinking(
                row, scenario,
                bulk_pen_reasoning=(settings or {}).get("train_bulk_pen_reasoning"),
                scpen_reasoning=(settings or {}).get("train_scpen_reasoning"),
                humannet_reasoning=(settings or {}).get("train_humannet_reasoning"),
                final_reasoning=(settings or {}).get("train_final_reasoning"),
            )
            user = _current_user_message(row)
            if scenario.form in {1, 3, 7}:
                if scenario.companions or not isinstance(scenario.fact, LayerFact):
                    raise CorpusBuildError("a direct form contains extra layer facts")
                if user.count(scenario.fact.layer_id) != 1:
                    raise CorpusBuildError("a direct form is not a one-tag input")
                direct_one_tag_count += 1
            if scenario.form == 4:
                if len(scenario.companions) != 3 or any(
                    user.count(item.layer_id) != 1 for item in scenario.companions
                ):
                    raise CorpusBuildError("a list form is not a three-tag input")
                list_three_tag_count += 1
                list_fields[str(scenario.requested_field)] += 1
                filters[str(scenario.filter_outcome)] += 1
            elif scenario.form == 2:
                one_fields[str(scenario.requested_field)] += 1
            elif scenario.form == 5:
                if not isinstance(scenario.fact, ModuleFact):
                    raise CorpusBuildError("a form-5 fact is not a module ID")
                namespaces[scenario.fact.namespace] += 1
            elif scenario.form == 6:
                subtypes[str(scenario.malformed_subtype)] += 1
                malformed_value = str(scenario.malformed_input)
                malformed_inputs.add(malformed_value)
                malformed_inputs_by_subtype[str(scenario.malformed_subtype)].add(
                    malformed_value
                )
                if isinstance(scenario.fact, LayerFact):
                    malformed_prefixes[str(scenario.malformed_subtype)][
                        scenario.fact.layer_id.split(":", 1)[0]
                    ] += 1
                try:
                    parse_layer_ref(contract, str(scenario.malformed_input))
                except (TypeError, ValueError):
                    pass
                else:
                    raise CorpusBuildError("a malformed input passes the layer parser")

            layer_facts: list[LayerFact] = []
            if isinstance(scenario.fact, LayerFact):
                layer_facts.append(scenario.fact)
            layer_facts.extend(scenario.companions)
            for fact in layer_facts:
                facts_by_id[fact.fact_id] = fact
            if scenario.form in {1, 2, 3, 7}:
                result_facts = [(scenario.fact, answer["result"])]
            elif scenario.form == 4:
                result_facts = list(
                    zip(scenario.companions, answer["result"]["layers"], strict=True)
                )
            else:
                result_facts = []
            for fact, result in result_facts:
                expected_ref = fact.layer_ref
                for key in expected_ref:
                    field_total += 1
                    field_correct += result.get(key) == expected_ref[key]
                observed_family = str(result.get("layer_family"))
                split_confusion[fact.layer_family][observed_family] += 1

        expected_one_fields = largest_remainder_counts(
            int(expected_counts[split].get(2, 0)),
            {name: Fraction(1, len(ONE_FIELD_NAMES)) for name in ONE_FIELD_NAMES},
        )
        if dict(one_fields) != expected_one_fields:
            raise CorpusBuildError(f"{split} one-field counts changed")
        expected_list_fields = largest_remainder_counts(
            int(expected_counts[split].get(4, 0)),
            {name: Fraction(1, len(ONE_FIELD_NAMES)) for name in ONE_FIELD_NAMES},
        )
        if dict(list_fields) != expected_list_fields:
            raise CorpusBuildError(f"{split} list-field counts changed")
        expected_filters = largest_remainder_counts(
            int(expected_counts[split].get(4, 0)),
            {"match": Fraction(1, 2), "no_match": Fraction(1, 2)},
        )
        if dict(filters) != expected_filters:
            raise CorpusBuildError(f"{split} filter outcomes changed")
        expected_namespaces = largest_remainder_counts(
            int(expected_counts[split].get(5, 0)),
            {name: Fraction(1, 2) for name in MODULE_NAMESPACES},
        )
        if dict(namespaces) != expected_namespaces:
            raise CorpusBuildError(f"{split} module namespace counts changed")
        configured_subtypes = settings["malformed_subtype_counts"]
        subtype_total = sum(int(value) for value in configured_subtypes.values())
        expected_subtypes = largest_remainder_counts(
            int(expected_counts[split].get(6, 0)),
            {
                name: Fraction(int(configured_subtypes[name]), subtype_total)
                for name in MALFORMED_SUBTYPES
            },
        )
        if dict(subtypes) != expected_subtypes:
            raise CorpusBuildError(f"{split} malformed subtype counts changed")
        for form, count in expected_counts[split].items():
            if not count:
                continue
            form_key = f"S1.6.{form}"
            for framing in ("direct", "step"):
                used = set(templates[form_key][framing])
                if used != {
                    str(index) for index in range(len(prompts[split][framing]))
                }:
                    raise CorpusBuildError(
                        f"{split} form {form} {framing} lacks template coverage"
                    )

        strata, contexts = _context_strata(facts_by_id.values())
        canonical_total = sum(int(value) for value in expected_counts[split].values())
        if canonical_total in {5000, 625}:
            expected_contexts = int(settings["context_split"]["counts"][split])
            expected_strata = {
                key: int(value)
                for key, value in settings["context_split"]["strata"][split].items()
            }
            if len(contexts) != expected_contexts or strata != expected_strata:
                raise CorpusBuildError(f"{split} context coverage changed")
        context_sets[split] = contexts
        context_counts[split] = len(contexts)
        stratum_counts[split] = strata
        status_counts[split] = dict(sorted(statuses.items()))
        tool_counts[split] = dict(sorted(calls.items()))
        framing_counts[split] = dict(sorted(actual_frames.items()))
        template_counts[split] = {
            form: {
                frame: dict(sorted(values.items()))
                for frame, values in sorted(frames.items())
            }
            for form, frames in sorted(templates.items())
        }
        field_counts[split] = {
            "one_field": {name: one_fields[name] for name in ONE_FIELD_NAMES},
            "list_filter": {name: list_fields[name] for name in ONE_FIELD_NAMES},
        }
        filter_counts[split] = {
            "match": filters["match"], "no_match": filters["no_match"]
        }
        namespace_counts[split] = {
            name: namespaces[name] for name in MODULE_NAMESPACES
        }
        subtype_counts[split] = {
            name: subtypes[name] for name in MALFORMED_SUBTYPES
        }
        semantic_counts[split] = {
            name: semantics[name]
            for name in (
                "direct_parse", "one_field", "list_filter",
                "wrong_input_type", "malformed_tag",
            )
        }
        thinking_counts[split] = {
            "row_count": len(rows),
            "thinking_step_count": thinking_steps,
            "pass": thinking_steps == len(rows),
        }
        structure_counts[split] = {
            "row_count": len(rows),
            "harness_message_count": harness_count,
            "no_tool_call_count": no_call_count,
            "direct_one_tag_count": direct_one_tag_count,
            "list_three_tag_count": list_three_tag_count,
            "pass": harness_count == 0 and no_call_count == len(rows),
        }
        available_prefixes = {
            fact.layer_id.split(":", 1)[0] for fact in facts_by_id.values()
        }
        subtype_coverage: dict[str, dict[str, Any]] = {}
        coverage_pass = True
        for subtype in MALFORMED_SUBTYPES:
            expected_prefixes = (
                available_prefixes & {"bulkPEN", "scPEN"}
                if subtype == "missing_path" else available_prefixes
            )
            observed_prefixes = set(malformed_prefixes[subtype])
            row_count = subtypes[subtype]
            unique_count = len(malformed_inputs_by_subtype[subtype])
            passed = unique_count == row_count and observed_prefixes == expected_prefixes
            coverage_pass = coverage_pass and passed
            subtype_coverage[subtype] = {
                "row_count": row_count,
                "unique_input_count": unique_count,
                "base_prefix_counts": dict(sorted(malformed_prefixes[subtype].items())),
                "expected_base_prefixes": sorted(expected_prefixes),
                "pass": passed,
            }
        if not coverage_pass:
            raise CorpusBuildError(f"{split} malformed source coverage changed")
        if len(malformed_inputs) != sum(subtypes.values()):
            raise CorpusBuildError(
                f"{split} repeats a malformed input across subtypes"
            )
        malformed_coverage[split] = {
            "row_count": sum(subtypes.values()),
            "unique_input_count": len(malformed_inputs),
            "subtypes": subtype_coverage,
            "pass": True,
        }
        field_rate = field_correct / field_total if field_total else 1.0
        field_gate[split] = {
            "field_value_count": field_total,
            "correct_field_value_count": field_correct,
            "accuracy": field_rate,
            "pass": split == "train" or field_rate >= 0.99,
        }
        confusion[split] = {
            expected: dict(sorted(observed.items()))
            for expected, observed in sorted(split_confusion.items())
        }
        form_gates[split] = {}
        for form, count in expected_counts[split].items():
            form_key = f"S1.6.{form}"
            values = gates[form_key]
            row_count = int(count)
            gate = {
                "row_count": row_count,
                "exact_status_count": values["exact_status_count"],
                "exact_result_count": values["exact_result_count"],
                "exact_decision_count": values["exact_decision_count"],
                "exact_blocked_reason_count": values["exact_blocked_reason_count"],
                "null_provenance_count": values["null_provenance_count"],
                "exact_call_count": values["exact_call_count"],
            }
            gate["pass"] = all(
                gate[key] == row_count
                for key in gate
                if key.endswith("_count") and key != "row_count"
            )
            form_gates[split][form_key] = gate
        total_false_defers += split_false_defers
        total_answerable += split_answerable

    for first, second in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ):
        overlap = context_sets[first] & context_sets[second]
        if overlap:
            raise CorpusBuildError(
                f"layer contexts overlap between {first} and {second}"
            )
    duplicate = duplicate_prompt_outcome_audit(rows_by_split, scenarios_by_split)
    wording = masked_wording_classifier_audit(
        rows_by_split, scenarios_by_split, prompts=prompts
    )
    full_prompt = full_visible_prompt_classifier_diagnostic(
        rows_by_split, scenarios_by_split
    )
    prose_lengths = _prose_length_audit(rows_by_split)
    return {
        "status_counts": status_counts,
        "tool_call_counts": tool_counts,
        "framing_counts": framing_counts,
        "prompt_template_counts": template_counts,
        "field_counts": field_counts,
        "filter_counts": filter_counts,
        "module_namespace_counts": namespace_counts,
        "malformed_subtype_counts": subtype_counts,
        "semantic_bucket_counts": semantic_counts,
        "context_counts": context_counts,
        "context_stratum_counts": stratum_counts,
        "per_form_gates": {
            "splits": form_gates,
            "heldout_field_accuracy": field_gate,
            "family_confusion_matrix": confusion,
            "no_inference_violation_count": 0,
            "false_defer_audit": {
                "eligible_row_count": total_answerable,
                "false_defer_count": total_false_defers,
                "false_defer_rate": total_false_defers / total_answerable,
                "pass": total_false_defers == 0,
            },
            "pass": all(
                bool(gate["pass"])
                for split_gates in form_gates.values()
                for gate in split_gates.values()
            ) and all(bool(value["pass"]) for value in field_gate.values()),
        },
        "prompt_structure_audit": {
            "splits": structure_counts,
            "malformed_input_coverage": malformed_coverage,
            "full_visible_prompt_classifier": full_prompt,
            "prose_length_audit": prose_lengths,
            "pass": (
                all(bool(value["pass"]) for value in structure_counts.values())
                and all(bool(value["pass"]) for value in malformed_coverage.values())
                and bool(prose_lengths["pass"])
            ),
        },
        "masked_wording_classifier": wording,
        "thinking_filter": {
            "rules": list(THINKING_FILTER_RULES),
            "splits": thinking_counts,
            "pass": all(bool(value["pass"]) for value in thinking_counts.values()),
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
            f"semantic validation failed for {path}:\n"
            + result.stdout
            + result.stderr
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
    """Build and validate S1.6, then publish its manifest last.

    A pinned prompt bank replaces only the training wrappers; validation and
    test keep their wrappers, so their rows do not change.
    """

    source_index_path = source_index_path.resolve()
    payload = check_source_index(source_index_path, contract)
    curriculum = contract["curriculum"]
    settings = curriculum["families"]["S1.6"]
    prompts = (
        wrapper_prompts(prompt_banks, LAYER_PROMPTS, bank_name="layer")
        if prompt_banks is not None
        else LAYER_PROMPTS
    )
    if any(
        len(prompts["train"][framing]) % len(LAYER_PROMPTS["train"][framing])
        for framing in ("direct", "step")
    ):
        raise CorpusBuildError(
            "the S1.6 training bank must be a multiple of the inline bank"
        )
    seed = int(settings["assignment_seed"])
    row_counts = split_row_counts(
        int(settings["train_rows"]), curriculum["split_fractions"]
    )
    counts_by_split = {
        split: form_counts(row_counts[split], settings) for split in SPLITS
    }
    pools = build_fact_pools(
        payload,
        settings=settings,
        seed=seed,
        contract=contract,
    )
    pool_coverage = source_pool_coverage(pools)
    expected_contexts = {
        split: int(value)
        for split, value in settings["context_split"]["counts"].items()
    }
    expected_strata = {
        split: {key: int(value) for key, value in counts.items()}
        for split, counts in settings["context_split"]["strata"].items()
    }
    if pool_coverage["context_counts"] != expected_contexts:
        raise CorpusBuildError("the source context counts changed")
    if pool_coverage["context_stratum_counts"] != expected_strata:
        raise CorpusBuildError("the source context strata changed")
    scenarios_by_split = select_scenarios(
        pools,
        counts_by_split,
        seed=seed,
        framing_shares=curriculum["framing"],
        settings=settings,
        prompts=prompts,
    )
    tools = tool_definitions(contract)
    rows_by_split = {
        split: [
            build_row(
                scenario, contract=contract, tools=tools, prompts=prompts,
                bulk_pen_reasoning=settings.get("train_bulk_pen_reasoning"),
                scpen_reasoning=settings.get("train_scpen_reasoning"),
                humannet_reasoning=settings.get("train_humannet_reasoning"),
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
    )

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".s1_6_build_", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        paths = {split: temporary / f"{split}.jsonl" for split in SPLITS}
        for split, path in paths.items():
            _write_jsonl(path, rows_by_split[split])
            if validate:
                _validate_corpus(path)

        form_manifest = {
            split: {
                f"S1.6.{form}": int(count)
                for form, count in counts_by_split[split].items()
            }
            for split in SPLITS
        }
        resolved_contract_path = contract_path.resolve()
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "section": "S1.6",
            "dataset_id": settings["dataset_id"],
            "seed": seed,
            "prompt_templates_sha256": _stable_sha256(prompts),
            "scenario_assignment_sha256": {
                split: _stable_sha256(
                    [item.scenario_id for item in scenarios_by_split[split]]
                )
                for split in SPLITS
            },
            "row_counts": row_counts,
            "form_counts": form_manifest,
            "semantic_bucket_counts": audit["semantic_bucket_counts"],
            "status_counts": audit["status_counts"],
            "tool_call_counts": audit["tool_call_counts"],
            "framing_counts": audit["framing_counts"],
            "prompt_template_counts": audit["prompt_template_counts"],
            "field_counts": audit["field_counts"],
            "filter_counts": audit["filter_counts"],
            "module_namespace_counts": audit["module_namespace_counts"],
            "malformed_subtype_counts": audit["malformed_subtype_counts"],
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
                    "path": str(
                        Path(__file__).resolve().relative_to(REPOSITORY_ROOT)
                    ),
                    "sha256": file_sha256(Path(__file__).resolve()),
                },
                "native_layer_policy": "parse_only_the_353_pinned_flist_tags",
                "records": payload["sources"],
            },
            "source_pool_coverage": {
                **pool_coverage,
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
            "seed", "prompt_templates_sha256", "scenario_assignment_sha256", "row_counts",
            "form_counts", "semantic_bucket_counts", "status_counts",
            "tool_call_counts", "framing_counts", "prompt_template_counts",
            "field_counts", "filter_counts", "module_namespace_counts",
            "malformed_subtype_counts", "context_counts",
            "context_stratum_counts", "sources", "source_pool_coverage",
            "per_form_gates", "prompt_structure_audit",
            "masked_wording_classifier", "thinking_filter",
            "duplicate_prompt_tool_outcome", "evidence", "files",
        }
        if set(manifest) != expected_manifest_keys:
            raise CorpusBuildError("the S1.6 manifest fields changed")
        if overlay is not None:
            manifest["s1_overlay"] = dict(overlay)
        if prompt_banks is not None:
            manifest["prompt_bank"] = prompt_bank_manifest(prompt_banks)
        if "train_bulk_pen_reasoning" in settings:
            manifest["train_bulk_pen_reasoning"] = dict(settings["train_bulk_pen_reasoning"])
        if "train_scpen_reasoning" in settings:
            manifest["train_scpen_reasoning"] = dict(settings["train_scpen_reasoning"])
        if "train_humannet_reasoning" in settings:
            manifest["train_humannet_reasoning"] = dict(settings["train_humannet_reasoning"])
        if "train_final_reasoning" in settings:
            manifest["train_final_reasoning"] = dict(settings["train_final_reasoning"])
        if settings.get("train_decorrelate"):
            manifest["train_decorrelation"] = filter_decorrelation_audit(scenarios_by_split["train"])
        if settings.get("train_targeted_v9"):
            match_counts = Counter(
                (len(item.expected_matching_layer_ids), item.framing)
                for item in scenarios_by_split["train"] if item.form == 4
            )
            if any(not match_counts[(count, frame)] for count in range(4) for frame in ("direct", "step")):
                raise CorpusBuildError("v9 filters lack empty, partial, or complete matches in both scopes")
            manifest["train_targeted_v9"] = {
                "pass": True,
                "filter_match_count_by_scope": {f"{count}:{frame}": value for (count, frame), value in sorted(match_counts.items())},
            }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)

        output_directory.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            (temporary / name).replace(output_directory / name)
        manifest_path.replace(output_directory / "manifest.json")
    return manifest


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
    settings = contract["curriculum"]["families"]["S1.6"]
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
                "output_directory": str(output_directory.resolve()),
                "row_counts": manifest["row_counts"],
                "form_counts": manifest["form_counts"],
                "files": manifest["files"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
