"""Build the canonical S1.2 Ensembl-to-symbol corpus.

The shared runtime contract owns sources, counts, proportions, and failure
policy.  This module adds deterministic fact selection and rendering for the
twelve reviewed S1.2 forms.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
    system_prompt_sha256,
    tool_definitions,
)
from .s0_corpus import (
    EvidenceSequence,
    assign_evaluation_components,
    build_identity_components,
)
from .s1_overlay import (
    apply_s1_overlay,
    apply_train_form_counts,
    overlay_pin,
)
from .s1_prompt_banks import (
    bank_reasoning,
    identity_prompts,
    prompt_bank_manifest,
)
from .source_index import check_source_index
from .s1_lookup_retry import assign_retry_cases, audit_retry_row, load_retry_policy, render_retry_case
from .sources import (
    GENE_ID_PATTERN,
    MOUSE_GENE_ID_PATTERN,
    file_sha256,
    read_ensembl_gtf,
    read_ensembl_gtf_identifiers,
)
from .tools import (
    GENE_ID_TOOL,
    SYMBOL_TOOL,
    ToolCallResult,
    WorldModelToolRuntime,
)


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-2-corpus-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-2-manifest-v1"

FORM_BY_BUCKET = {
    "direct": 1,
    "inside_task": 2,
    "version_suffix": 3,
    "no_symbol": 4,
    "not_in_release": 5,
    "malformed": 6,
    "harness_exact": 7,
    "harness_near_miss": 8,
    "symbol_input": 9,
    "other_species": 10,
    "backend_unavailable": 11,
    "transcript_wrong_type": 12,
}
BUCKET_BY_FORM = {value: key for key, value in FORM_BY_BUCKET.items()}
BASE_TOOL_CALL_BUCKETS = frozenset(
    {
        "direct",
        "inside_task",
        "version_suffix",
        "no_symbol",
        "not_in_release",
        "malformed",
        "harness_near_miss",
        "symbol_input",
    }
)
FLEXIBLE_FRAMING_FORMS = (3, 4, 5, 6, 7, 8, 9, 10, 12)

# The wrapper deliberately does not reveal the input type or outcome.  The
# request payload is the only cue, and validation/test paraphrases are held
# out from training.
IDENTITY_PROMPTS = {
    "train": {
        "direct": (
            "Resolve this gene-label request: {request}.",
            "Answer this request for a gene label: {request}.",
            "Use the supplied input to answer this request: {request}.",
            "Determine the requested human gene label: {request}.",
        ),
        "step": (
            "Before the next analysis step, resolve this request: {request}.",
            "The next operation needs this gene-label request answered: {request}.",
            "Prepare the requested gene label for the next operation: {request}.",
            "Resolve this request so the larger analysis can proceed: {request}.",
        ),
    },
    "validation": {
        "direct": (
            "Find the answer to this gene-label request: {request}.",
            "Return the requested human gene label: {request}.",
            "Work out this request about a gene label: {request}.",
            "Supply the answer for this gene-label request: {request}.",
        ),
        "step": (
            "Resolve this gene-label request before the analysis continues: {request}.",
            "The following gene label is needed for the next step: {request}.",
            "Before continuing, answer this gene-label request: {request}.",
            "Prepare this gene label for the next analysis action: {request}.",
        ),
    },
    "test": {
        "direct": (
            "What is the answer to this gene-label request: {request}?",
            "Give the requested human gene label: {request}.",
            "Answer the following request about a gene label: {request}.",
            "Identify the requested human gene label: {request}.",
        ),
        "step": (
            "The workflow first needs this gene-label request resolved: {request}.",
            "Answer this gene-label request for the next workflow action: {request}.",
            "Before the workflow advances, determine this gene label: {request}.",
            "Resolve the following gene-label request for downstream use: {request}.",
        ),
    },
}


class CorpusBuildError(ValueError):
    """Report a source, selection, or rendered-corpus invariant failure."""


@dataclass(frozen=True, slots=True)
class GeneFact:
    """One source-backed identifier with its leakage-safe split units."""

    fact_id: str
    split_units: tuple[str, ...]
    gene_id: str
    gene_symbol: str | None = None
    gene_version: int | None = None
    species: str = "human"
    kind: str = "current"
    transcript_id: str | None = None
    renamed: bool = False


@dataclass(frozen=True, slots=True)
class Scenario:
    """One selected form before canonical runtime execution."""

    split: str
    form: int
    fact: GeneFact
    framing: str
    template_index: int
    occurrence: int
    input_value: str
    distractor: GeneFact | None = None
    paired_success_id: str | None = None

    @property
    def source_trace(self) -> str:
        return f"S1.2.{self.form}"

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
                "input_value": self.input_value,
                "distractor": (
                    self.distractor.fact_id if self.distractor else None
                ),
                "paired_success_id": self.paired_success_id,
            }
        )

    @property
    def split_units(self) -> tuple[str, ...]:
        units = set(self.fact.split_units)
        if self.distractor is not None:
            units.update(self.distractor.split_units)
        return tuple(sorted(units))

    @property
    def metadata_split_unit(self) -> str:
        if len(self.split_units) == 1:
            return self.split_units[0]
        return "group_" + _stable_sha256(self.split_units)[:24]


def _canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def split_row_counts(
    train_rows: int, split_fractions: Mapping[str, Any]
) -> dict[str, int]:
    """Derive all split counts from the contract's train anchor."""

    train_fraction = _fraction(split_fractions["train"])
    counts = {"train": train_rows}
    for split in ("validation", "test"):
        value = Fraction(train_rows) * _fraction(
            split_fractions[split]
        ) / train_fraction
        if value.denominator != 1:
            raise CorpusBuildError(
                f"S1.2 {split} row count is not integral: {value}"
            )
        counts[split] = value.numerator
    return counts


def largest_remainder_counts(
    total: int, shares: Mapping[str, Any]
) -> dict[str, int]:
    """Apportion an exact integer total with contract order as tie-breaker."""

    names = list(shares)
    raw = {name: Fraction(total) * _fraction(shares[name]) for name in names}
    counts = {
        name: value.numerator // value.denominator
        for name, value in raw.items()
    }
    missing = total - sum(counts.values())
    order = sorted(
        names,
        key=lambda name: (
            -(raw[name] - counts[name]),
            names.index(name),
        ),
    )
    for name in order[:missing]:
        counts[name] += 1
    if sum(counts.values()) != total:
        raise CorpusBuildError("largest-remainder allocation did not reconcile")
    return counts


def form_counts(
    total: int,
    settings: Mapping[str, Any],
    curriculum: Mapping[str, Any],
) -> dict[int, int]:
    """Allocate the ten base forms, added transcript form, and failures."""

    policy = settings["failure_injection"]
    if policy != {
        "source_form": "direct",
        "allocation": "replace",
        "call_stage": "first",
    }:
        raise CorpusBuildError("unsupported S1.2 failure policy")

    train_rows = int(settings["train_rows"])
    base_train_rows = int(settings["base_train_rows"])
    added = settings["added_form_rows"]
    if set(added) != {"transcript_wrong_type"}:
        raise CorpusBuildError("unsupported S1.2 added-form allocation")
    added_train_rows = int(added["transcript_wrong_type"])
    if base_train_rows + added_train_rows != train_rows:
        raise CorpusBuildError("S1.2 base and added rows do not reconcile")

    base_value = Fraction(total * base_train_rows, train_rows)
    added_value = Fraction(total * added_train_rows, train_rows)
    if base_value.denominator != 1 or added_value.denominator != 1:
        raise CorpusBuildError("S1.2 split base/addition is not integral")
    base_total = base_value.numerator
    added_total = added_value.numerator

    base = largest_remainder_counts(base_total, settings["shares"])
    if set(base) != set(FORM_BY_BUCKET).difference(
        {"backend_unavailable", "transcript_wrong_type"}
    ):
        raise CorpusBuildError("S1.2 base share fields changed")
    tool_rows = sum(base[name] for name in BASE_TOOL_CALL_BUCKETS)
    failure_value = Fraction(tool_rows) * _fraction(
        curriculum["tool_failure_share"]
    )
    if failure_value.denominator != 1:
        raise CorpusBuildError(
            f"S1.2 failure allocation is not integral: {failure_value}"
        )
    failures = failure_value.numerator
    base["direct"] -= failures
    if base["direct"] < failures:
        raise CorpusBuildError(
            "the retained direct rows cannot supply every failure pair"
        )
    counts = {FORM_BY_BUCKET[name]: count for name, count in base.items()}
    counts[11] = failures
    counts[12] = added_total
    if sum(counts.values()) != total:
        raise CorpusBuildError("S1.2 form counts do not match the split total")
    return dict(sorted(counts.items()))


def override_train_counts(
    counts: Mapping[int, int], overrides: Mapping[int, int]
) -> dict[int, int]:
    """Apply S1 v5 train-only form counts.

    The train total stays the same, and every tool failure (form 11) still
    copies its own retained direct row (form 1).
    """

    updated = apply_train_form_counts(counts, overrides)
    if updated[1] < updated[11]:
        raise CorpusBuildError(
            "the retained direct rows cannot supply every failure pair"
        )
    return updated


def _split_for_hash(
    key: str, *, seed: int, fractions: Mapping[str, Any]
) -> str:
    point = Fraction(
        int(_stable_sha256({"assignment_seed": seed, "key": key})[:16], 16),
        16**16,
    )
    train_edge = _fraction(fractions["train"])
    validation_edge = train_edge + _fraction(fractions["validation"])
    if point < train_edge:
        return "train"
    if point < validation_edge:
        return "validation"
    return "test"


def _fact_order(fact: GeneFact, *, seed: int, split: str, pool: str) -> str:
    return _stable_sha256(
        {
            "assignment_seed": seed,
            "split": split,
            "pool": pool,
            "fact_id": fact.fact_id,
        }
    )


def _fact_id(kind: str, value: str) -> str:
    return "fact_" + _stable_sha256({"kind": kind, "value": value})[:24]


def build_fact_pools(
    payload: Mapping[str, Any],
    human_identifiers: Any,
    mouse_genes: Iterable[Any],
    *,
    s0_settings: Mapping[str, Any],
    seed: int,
    fractions: Mapping[str, Any],
) -> dict[str, dict[str, list[GeneFact]]]:
    """Build leakage-safe current, retired, mouse, and transcript pools."""

    components, component_by_gene = build_identity_components(
        payload["genes_by_id"]
    )
    assignments = assign_evaluation_components(
        components,
        s0_settings["evaluation"],
        seed=int(s0_settings["assignment_seed"]),
    )
    versions: dict[str, int] = {}
    for item in human_identifiers.genes:
        gene_id = str(item.gene_id)
        version = int(item.gene_version)
        if gene_id in versions or version < 1:
            raise CorpusBuildError(f"invalid duplicate gene version for {gene_id}")
        versions[gene_id] = version
    if set(versions) != set(payload["genes_by_id"]):
        raise CorpusBuildError(
            "the GTF version table and source-index registry differ"
        )

    symbol_to_ids: dict[str, list[str]] = defaultdict(list)
    for gene_id, values in payload["genes_by_id"].items():
        if values[0] is not None:
            symbol_to_ids[str(values[0]).casefold()].append(str(gene_id))
    renamed_gene_ids = {
        str(gene_id)
        for row in payload["historical_symbols"]
        if "previous" in row[1]
        for gene_id in row[2]
        if str(gene_id) in payload["genes_by_id"]
    }

    pool_names = (
        "current_named",
        "current_unnamed",
        "symbol_input",
        "retired",
        "mouse",
        "transcript",
    )
    pools = {
        split: {name: [] for name in pool_names}
        for split in ("train", "validation", "test")
    }
    for component in components:
        split = assignments.get(component.split_unit, "train")
        for gene_id in component.gene_ids:
            raw_symbol = payload["genes_by_id"][gene_id][0]
            symbol = None if raw_symbol is None else str(raw_symbol)
            fact = GeneFact(
                fact_id=_fact_id("current_gene_id", gene_id),
                split_units=(component.split_unit,),
                gene_id=gene_id,
                gene_symbol=symbol,
                gene_version=versions[gene_id],
                renamed=gene_id in renamed_gene_ids,
            )
            pool = "current_named" if symbol is not None else "current_unnamed"
            pools[split][pool].append(fact)
            if symbol is not None and len(symbol_to_ids[symbol.casefold()]) == 1:
                pools[split]["symbol_input"].append(fact)

    for row in payload["retired_genes"]:
        gene_id = str(row[0])
        linked_units = {
            component_by_gene[str(current_id)]
            for current_id in row[1]
            if str(current_id) in component_by_gene
        }
        linked_splits = {
            assignments.get(unit, "train") for unit in linked_units
        }
        if len(linked_splits) > 1:
            continue
        split = (
            next(iter(linked_splits))
            if linked_splits
            else _split_for_hash(
                f"retired:{gene_id}", seed=seed, fractions=fractions
            )
        )
        units = tuple(sorted(linked_units)) or (
            "retired_" + _stable_sha256(gene_id)[:24],
        )
        pools[split]["retired"].append(
            GeneFact(
                fact_id=_fact_id("retired_gene_id", gene_id),
                split_units=units,
                gene_id=gene_id,
                kind="retired",
            )
        )

    mouse_by_id: dict[str, Any] = {}
    for gene in mouse_genes:
        mouse_by_id.setdefault(str(gene.gene_id), gene)
    for gene_id, gene in mouse_by_id.items():
        split = _split_for_hash(
            f"mouse:{gene_id}", seed=seed, fractions=fractions
        )
        pools[split]["mouse"].append(
            GeneFact(
                fact_id=_fact_id("mouse_gene_id", gene_id),
                split_units=("mouse_" + _stable_sha256(gene_id)[:24],),
                gene_id=gene_id,
                gene_symbol=(
                    None
                    if gene.gene_symbol is None
                    else str(gene.gene_symbol)
                ),
                species="mouse",
                kind="mouse",
            )
        )

    seen_transcripts: set[str] = set()
    for item in human_identifiers.transcripts:
        transcript_id = str(item.transcript_id)
        parent_id = str(item.gene_id)
        if transcript_id in seen_transcripts:
            raise CorpusBuildError(f"duplicate transcript ID {transcript_id}")
        seen_transcripts.add(transcript_id)
        split_unit = component_by_gene.get(parent_id)
        if split_unit is None:
            raise CorpusBuildError(
                f"transcript {transcript_id} has unknown parent {parent_id}"
            )
        split = assignments.get(split_unit, "train")
        raw_symbol = payload["genes_by_id"][parent_id][0]
        pools[split]["transcript"].append(
            GeneFact(
                fact_id=_fact_id("transcript_id", transcript_id),
                split_units=(split_unit,),
                gene_id=parent_id,
                gene_symbol=(
                    None if raw_symbol is None else str(raw_symbol)
                ),
                gene_version=versions[parent_id],
                kind="transcript",
                transcript_id=transcript_id,
            )
        )

    for split, split_pools in pools.items():
        for name, facts in split_pools.items():
            facts.sort(
                key=lambda fact: _fact_order(
                    fact, seed=seed, split=split, pool=name
                )
            )
    return pools


def _mixed_case(value: str) -> str:
    letters = 0
    rendered = []
    for char in value:
        if not char.isalpha():
            rendered.append(char)
            continue
        rendered.append(char.lower() if letters % 2 == 0 else char.upper())
        letters += 1
    mixed = "".join(rendered)
    return mixed if mixed != value else value.lower()


def _canonical_input(fact: GeneFact, form: int) -> str:
    if form == 3:
        if fact.gene_version is None:
            raise CorpusBuildError("a version-suffix fact has no version")
        return f"{fact.gene_id}.{fact.gene_version}"
    if form == 6:
        match = GENE_ID_PATTERN.fullmatch(fact.gene_id)
        if match is None:
            raise CorpusBuildError("a malformed form lacks a canonical source ID")
        return fact.gene_id[:-1]
    if form == 9:
        if fact.gene_symbol is None:
            raise CorpusBuildError("a symbol-input fact has no symbol")
        return fact.gene_symbol
    if form == 12:
        if fact.transcript_id is None:
            raise CorpusBuildError("a transcript form has no transcript ID")
        return fact.transcript_id
    return fact.gene_id


def _input_value(
    fact: GeneFact,
    *,
    form: int,
    split: str,
    occurrence: int,
    seed: int,
    case_rate: Any,
) -> str:
    value = _canonical_input(fact, form)
    if fact.species != "human":
        return value
    point = Fraction(
        int(
            _stable_sha256(
                {
                    "assignment_seed": seed,
                    "split": split,
                    "form": form,
                    "fact_id": fact.fact_id,
                    "occurrence": occurrence,
                    "pool": "input_case",
                }
            )[:16],
            16,
        ),
        16**16,
    )
    if point >= _fraction(case_rate):
        return value
    return value.lower() if point < _fraction(case_rate) / 2 else _mixed_case(value)


def _select(
    facts: Sequence[GeneFact],
    count: int,
    *,
    split: str,
    form: int,
    seed: int,
    case_rate: Any,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
) -> list[Scenario]:
    if not facts:
        raise CorpusBuildError(f"the {split} form {form} pool is empty")
    framing_variants = 1 if form in {1, 2} else 2
    maximum_variants = (
        framing_variants * len(prompts[split]["direct"])
    )
    if count > len(facts) * maximum_variants:
        raise CorpusBuildError(
            f"the {split} form {form} pool cannot make {count} unique prompts"
        )
    chosen = [facts[ordinal % len(facts)] for ordinal in range(count)]
    if form in {1, 2, 3} and not any(fact.renamed for fact in chosen):
        renamed = next((fact for fact in facts if fact.renamed), None)
        if renamed is None:
            raise CorpusBuildError(
                f"the {split} form {form} pool has no renamed gene"
            )
        chosen[-1] = renamed
    selected = []
    for ordinal, fact in enumerate(chosen):
        occurrence = ordinal // len(facts)
        framing = "direct" if form == 1 else "step" if form == 2 else ""
        template_index = (ordinal % len(facts) + occurrence) % len(
            prompts[split][framing or "direct"]
        )
        selected.append(
            Scenario(
                split=split,
                form=form,
                fact=fact,
                framing=framing,
                template_index=template_index,
                occurrence=occurrence,
                input_value=_input_value(
                    fact,
                    form=form,
                    split=split,
                    occurrence=occurrence,
                    seed=seed,
                    case_rate=case_rate,
                ),
            )
        )
    return selected


def _assign_flexible_framing(
    selected: dict[int, list[Scenario]],
    counts: Mapping[int, int],
    *,
    split: str,
    total: int,
    framing_shares: Mapping[str, Any],
    seed: int,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    balanced: bool = False,
) -> None:
    """Assign flexible forms while meeting the family-wide framing quota.

    A balanced split starts each row's wrapper search round-robin instead of
    at a fact hash, so a large training bank covers every wrapper.
    """

    target = largest_remainder_counts(total, framing_shares)
    direct_needed = target["direct"] - counts[1] - counts[11]
    step_needed = target["step"] - counts[2]
    flexible_total = sum(counts[form] for form in FLEXIBLE_FRAMING_FORMS)
    if direct_needed < 0 or step_needed < 0:
        raise CorpusBuildError(f"{split} fixed forms exceed framing targets")
    if direct_needed + step_needed != flexible_total:
        raise CorpusBuildError(f"{split} flexible framing does not reconcile")

    shares = {
        str(form): Fraction(counts[form], flexible_total)
        for form in FLEXIBLE_FRAMING_FORMS
    }
    direct_by_form = {
        int(form): value
        for form, value in largest_remainder_counts(
            direct_needed, shares
        ).items()
    }
    for form in FLEXIBLE_FRAMING_FORMS:
        scenarios = selected[form]
        direct_count = direct_by_form[form]
        if direct_count < len(prompts[split]["direct"]) or len(
            scenarios
        ) - direct_count < len(prompts[split]["step"]):
            raise CorpusBuildError(
                f"{split} form {form} cannot cover framing templates"
            )
        ranked = sorted(
            scenarios,
            key=lambda item: _stable_sha256(
                {
                    "assignment_seed": seed,
                    "split": split,
                    "form": form,
                    "fact_id": item.fact.fact_id,
                    "occurrence": item.occurrence,
                    "pool": "framing",
                }
            ),
        )
        direct_ids = {
            (item.fact.fact_id, item.occurrence)
            for item in ranked[:direct_count]
        }
        framed = [
            replace(
                item,
                framing=(
                    "direct"
                    if (item.fact.fact_id, item.occurrence) in direct_ids
                    else "step"
                ),
            )
            for item in scenarios
        ]
        for framing in ("direct", "step"):
            group = sorted(
                (item for item in framed if item.framing == framing),
                key=lambda item: _stable_sha256(
                    {
                        "assignment_seed": seed,
                        "split": split,
                        "form": form,
                        "framing": framing,
                        "fact_id": item.fact.fact_id,
                        "occurrence": item.occurrence,
                        "pool": "template",
                    }
                ),
            )
            template_count = len(prompts[split][framing])
            used_by_fact: dict[str, set[int]] = defaultdict(set)
            indexes: dict[tuple[str, int], int] = {}
            for position, item in enumerate(group):
                start = (
                    position
                    if balanced
                    else int(_stable_sha256(item.fact.fact_id)[:8], 16)
                    + item.occurrence
                ) % template_count
                index = next(
                    (
                        (start + offset) % template_count
                        for offset in range(template_count)
                        if (start + offset) % template_count
                        not in used_by_fact[item.fact.fact_id]
                    ),
                    None,
                )
                if index is None:
                    raise CorpusBuildError(
                        f"{split} form {form} repeats one prompt too often"
                    )
                used_by_fact[item.fact.fact_id].add(index)
                indexes[(item.fact.fact_id, item.occurrence)] = index
            framed = [
                replace(
                    item,
                    template_index=indexes[
                        (item.fact.fact_id, item.occurrence)
                    ],
                )
                if item.framing == framing
                else item
                for item in framed
            ]
        selected[form] = framed


def select_scenarios(
    pools: Mapping[str, Mapping[str, Sequence[GeneFact]]],
    counts_by_split: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
    case_rate: Any,
    framing_shares: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    balanced_splits: Iterable[str] = (),
    train_decorrelate: bool = False,
) -> dict[str, list[Scenario]]:
    """Select every form, exact framings, and paired failure wrapper positions."""

    balanced_splits = frozenset(balanced_splits)
    framing_shares = framing_shares or {"direct": 0.5, "step": 0.5}
    scenarios_by_split: dict[str, list[Scenario]] = {}
    for split in ("train", "validation", "test"):
        counts = counts_by_split[split]
        split_pools = pools[split]
        selected: dict[int, list[Scenario]] = {}
        pool_for_form = {
            1: "current_named",
            2: "current_named",
            3: "current_named",
            4: "current_unnamed",
            5: "retired",
            6: "current_named",
            7: "current_named",
            8: "current_named",
            9: "symbol_input",
            10: "mouse",
            12: "transcript",
        }
        for form in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12):
            selected[form] = _select(
                split_pools[pool_for_form[form]],
                counts[form],
                split=split,
                form=form,
                seed=seed,
                case_rate=case_rate,
                prompts=prompts,
            )

        _assign_flexible_framing(
            selected,
            counts,
            split=split,
            total=sum(counts.values()),
            framing_shares=framing_shares,
            seed=seed,
            prompts=prompts,
            balanced=split in balanced_splits,
        )

        distractors = split_pools["current_named"]
        near_misses = []
        for scenario in selected[8]:
            # The requested identity chooses content independently of wording.
            distractor_key = (
                _stable_sha256({
                    "seed": seed, "split": split, "form": scenario.form,
                    "fact_id": scenario.fact.fact_id,
                    "occurrence": scenario.occurrence, "pool": "distractor",
                })
                if train_decorrelate and split == "train"
                else scenario.scenario_id
            )
            start = int(distractor_key[:16], 16) % len(distractors)
            distractor = next(
                (
                    distractors[(start + offset) % len(distractors)]
                    for offset in range(len(distractors))
                    if distractors[(start + offset) % len(distractors)].gene_id
                    != scenario.fact.gene_id
                ),
                None,
            )
            if distractor is None:
                raise CorpusBuildError("a near-miss distractor is unavailable")
            near_misses.append(replace(scenario, distractor=distractor))
        selected[8] = near_misses

        ordered_successes = sorted(
            selected[1], key=lambda scenario: scenario.scenario_id
        )
        pair_sources: list[Scenario] = []
        paired_ids: set[str] = set()
        for template_index in range(len(prompts[split]["direct"])):
            source = next(
                item
                for item in ordered_successes
                if item.template_index == template_index
            )
            pair_sources.append(source)
            paired_ids.add(source.scenario_id)
        pair_sources.extend(
            item
            for item in ordered_successes
            if item.scenario_id not in paired_ids
        )
        if train_decorrelate and split == "train":
            # Keep failure targets stable when the wrapper bank grows.
            pair_sources = selected[1]
        selected[11] = [
            replace(
                success,
                form=11,
                paired_success_id=success.scenario_id,
            )
            for success in pair_sources[: counts[11]]
        ]

        combined = [
            scenario
            for form in range(1, 13)
            for scenario in selected[form]
        ]
        if len(combined) != sum(counts.values()):
            raise CorpusBuildError(f"{split} scenario selection did not reconcile")
        scenarios_by_split[split] = sorted(
            combined,
            key=lambda scenario: _stable_sha256(
                {
                    "assignment_seed": seed,
                    "split": split,
                    "scenario_id": scenario.scenario_id,
                    "pool": "output",
                }
            ),
        )
    return scenarios_by_split


def _audit_train_decorrelation(
    rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Scenario],
    prompts: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    """Check joint wrapper coverage across evidence and lookup-outcome cases."""

    coverage: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    for row, scenario in zip(rows, scenarios, strict=True):
        if _user_message(row) != _question(scenario, prompts=prompts):
            raise CorpusBuildError("decorrelated training must use its shared wrapper bank")
        coverage[(scenario.form, scenario.framing)][scenario.template_index] += 1
    for (_, framing), counts in coverage.items():
        if set(counts) != set(range(len(prompts["train"][framing]))):
            raise CorpusBuildError("a decorrelated training case lacks wrapper coverage")
    return {
        "wrapper_counts_by_form_and_framing": {
            f"{form}:{framing}": dict(sorted(counts.items()))
            for (form, framing), counts in sorted(coverage.items())
        },
        "failure_prompts_match_success": True,
        "content_selection_independent_of_wrapper_count": True,
    }


def _request(scenario: Scenario) -> str:
    return f"the human gene symbol corresponding to {scenario.input_value}"


def _question(
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
) -> str:
    template = prompts[scenario.split][scenario.framing][
        scenario.template_index
    ]
    return template.format(request=_request(scenario))


def _answer(
    *,
    status: str,
    provenance: Mapping[str, Any] | None,
    result: Mapping[str, Any],
    decision: str,
    blocked_reason: str | None,
    summary: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "provenance": dict(provenance) if provenance is not None else None,
        "result": dict(result),
        "continuation_decision": decision,
        "blocked_reason": blocked_reason,
        "summary": summary,
    }


def _evidence_row(
    call_result: ToolCallResult,
    *,
    row_id: str,
    producer: str | None = None,
) -> list[str] | None:
    if call_result.evidence is None:
        return None
    return [
        str(call_result.evidence["evidence_id"]),
        row_id,
        producer or str(call_result.evidence["tool_name"]),
    ]


def _append_call(
    messages: list[dict[str, Any]],
    *,
    thinking: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    call_result: ToolCallResult,
) -> None:
    messages.extend(
        [
            {
                "role": "assistant",
                "thinking": thinking,
                "tool_call": {
                    "name": tool_name,
                    "arguments": dict(arguments),
                },
            },
            {
                "role": "tool",
                "name": tool_name,
                "content": call_result.output,
            },
        ]
    )


def _harness_content(call_result: ToolCallResult) -> str:
    output = call_result.output
    if output.get("status") != "resolved":
        raise CorpusBuildError("a harness entry did not resolve")
    block = {
        "provenance": output["provenance"],
        "entries": [
            {
                "gene_id": output["gene_id"],
                "gene_symbol": output["gene_symbol"],
            }
        ],
    }
    return (
        "Registry entries, Ensembl release 116:\n"
        + _canonical_json(block, sort_keys=False)
    )


def _row_id(scenario: Scenario, schema_version: str) -> str:
    return "wm3_s1_2_" + _stable_sha256(
        {
            "scenario_id": scenario.scenario_id,
            "schema_version": schema_version,
        }
    )[:24]


def _resolved_answer(
    scenario: Scenario,
    output: Mapping[str, Any],
    *,
    release: str,
    no_symbol_reasoning: Mapping[str, str | None] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render a resolved answer.

    ``no_symbol_reasoning`` maps framing to S1.2.4 training reasoning.
    V5 supplies static bank text; v6 templates bind the returned gene ID.
    """

    if output.get("status") != "resolved":
        raise CorpusBuildError(
            f"form {scenario.form} selected a non-resolving identity"
        )
    gene_id = str(output["gene_id"])
    symbol = output["gene_symbol"]
    decision = "continue" if scenario.framing == "step" else "stop"

    if scenario.form == 4:
        if symbol is not None:
            raise CorpusBuildError("a no-symbol fact unexpectedly has a symbol")
        thinking = (
            "The registry has this ID but no symbol for it. The ID is still "
            "valid, so "
            + (
                "the analysis can continue with it."
                if decision == "continue"
                else "the task is complete with the valid ID."
            )
        )
        bank_thinking = (no_symbol_reasoning or {}).get(scenario.framing)
        if scenario.split == "train" and bank_thinking:
            thinking = bank_thinking.format(gene_id=gene_id)
        summary = (
            f"<GENE:{gene_id}> is a human gene in Ensembl release {release} "
            "with no known symbol."
        )
        if decision == "continue":
            summary += " The analysis continues with the ID."
    else:
        if not isinstance(symbol, str):
            raise CorpusBuildError(f"form {scenario.form} selected an unnamed gene")
        if scenario.form == 3:
            thinking = (
                f"The registry maps the ID to {symbol}. The version suffix "
                "does not change the gene. "
                + (
                    "The next step can use the symbol."
                    if decision == "continue"
                    else "The question asks only for the symbol, so the task is complete."
                )
            )
            summary = (
                f"{scenario.input_value} is <GENE:{gene_id}|{symbol}> in "
                f"Ensembl release {release}. The version suffix does not "
                "change the gene."
            )
            if decision == "continue":
                summary += " The analysis continues with this gene."
        elif scenario.form in {2, 8} and decision == "continue":
            next_action = (
                "next operation" if scenario.form == 2 else "next step"
            )
            thinking = (
                f"The registry gives the symbol {symbol} for the seed ID. "
                f"The {next_action} can use it."
            )
            summary = (
                f"The seed <GENE:{gene_id}|{symbol}> has the symbol "
                f"{symbol} in Ensembl release {release}. The analysis "
                "continues with this gene."
            )
        elif scenario.form == 9:
            thinking = (
                f"{symbol} is a symbol in release {release}, with the ID "
                f"{gene_id}. The input is already the symbol, so "
                + (
                    "the next step can use it."
                    if decision == "continue"
                    else "the task is complete."
                )
            )
            summary = (
                "The input is already a gene symbol: "
                f"<GENE:{gene_id}|{symbol}> in Ensembl release {release}."
            )
            if decision == "continue":
                summary += " The analysis continues with this gene."
        else:
            thinking = (
                f"The registry gives the symbol {symbol} for this ID in "
                f"release {release}. "
                + (
                    "The next operation can use it."
                    if decision == "continue"
                    else "The question asks only for the symbol, so the task is complete."
                )
            )
            if decision == "continue":
                summary = (
                    f"The seed <GENE:{gene_id}|{symbol}> has the symbol "
                    f"{symbol} in Ensembl release {release}. The analysis "
                    "continues with this gene."
                )
            else:
                summary = (
                    f"<GENE:{gene_id}|{symbol}> has the symbol {symbol} in "
                    f"Ensembl release {release}."
                )

    return thinking, _answer(
        status="resolved",
        provenance=output["provenance"],
        result={"gene_id": gene_id, "gene_symbol": symbol},
        decision=decision,
        blocked_reason=None,
        summary=summary,
    )


def _pre_call_thinking(scenario: Scenario) -> str:
    if scenario.form == 3:
        return (
            "The ID has a version suffix. The tool ignores the suffix, so I "
            "will pass the ID as given."
        )
    if scenario.form == 4 and scenario.framing == "step":
        return (
            "The analysis needs the symbol for the seed ID. I will look it "
            "up in the registry."
        )
    if scenario.form == 5:
        return (
            "I need the symbol for this Ensembl ID. I will pass the ID to "
            "the registry tool as given."
        )
    if scenario.form == 6:
        return (
            "The input starts with ENSG, so it is meant as a human gene ID. "
            "The tool checks the format, so I will pass it as given."
        )
    if scenario.form == 8:
        if scenario.distractor is None:
            raise CorpusBuildError("a near miss has no distractor")
        requested_kind = (
            "seed ID" if scenario.framing == "step" else "requested ID"
        )
        return (
            f"The registry entries list {scenario.distractor.gene_id} but "
            f"not {scenario.input_value}. I will look up the {requested_kind} "
            "in the registry."
        )
    if scenario.form == 9:
        return (
            f"The input {scenario.input_value} is a gene symbol, not an "
            "Ensembl ID. The ID tool cannot match it, so I will check it "
            "with the symbol tool."
        )
    if scenario.framing == "step":
        return (
            "The next operation needs the symbol for the seed ID. I will "
            "look it up in the registry."
        )
    return "I need the symbol for this Ensembl ID. I will look it up in the registry."


def build_row(
    scenario: Scenario,
    *,
    runtime: WorldModelToolRuntime,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]],
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    no_symbol_reasoning: Mapping[str, str | None] | None = None,
    final_reasoning: str | None = None,
    version_suffix_reasoning: Mapping[str, str] | None = None,
    near_miss_reasoning: Mapping[str, str] | None = None,
    symbol_input_reasoning: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Execute a scenario, with reviewed training thinking for forms 1, 3-6, and 8-9."""

    schema_version = str(contract["runtime"]["row_schema_version"])
    row_id = _row_id(scenario, schema_version)
    release = runtime.ensembl_release
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": contract["runtime"]["system_prompt"]}
    ]
    evidence: list[list[str]] = []

    harness_result: ToolCallResult | None = None
    if scenario.form in {7, 8}:
        harness_fact = (
            scenario.fact if scenario.form == 7 else scenario.distractor
        )
        if harness_fact is None:
            raise CorpusBuildError("a near miss has no distractor")
        harness_result = runtime.execute(
            GENE_ID_TOOL, {"gene_id": harness_fact.gene_id}
        )
        harness_evidence = _evidence_row(
            harness_result,
            row_id=row_id,
            producer="harness_registry_block",
        )
        if harness_evidence is None:
            raise CorpusBuildError("a harness entry produced no evidence")
        evidence.append(harness_evidence)
        messages.append(
            {"role": "harness", "content": _harness_content(harness_result)}
        )

    messages.append(
        {"role": "user", "content": _question(scenario, prompts=prompts)}
    )

    final_thinking: str
    final_answer: dict[str, Any]
    if scenario.form == 7:
        if harness_result is None:
            raise CorpusBuildError("an exact harness has no result")
        output = harness_result.output
        symbol = output["gene_symbol"]
        if not isinstance(symbol, str):
            raise CorpusBuildError("an exact harness selected an unnamed gene")
        gene_id = str(output["gene_id"])
        decision = "continue" if scenario.framing == "step" else "stop"
        final_thinking = (
            "The registry entries from the harness list this ID for Ensembl "
            f"release {release}, with the symbol {symbol}. The lookup is "
            "already done, so I will not call the tool."
        )
        summary = (
            f"<GENE:{gene_id}|{symbol}> has the symbol {symbol} in Ensembl "
            f"release {release}."
        )
        if decision == "continue":
            summary += " The analysis continues with this gene."
        final_answer = _answer(
            status="resolved",
            provenance=output["provenance"],
            result={"gene_id": gene_id, "gene_symbol": symbol},
            decision=decision,
            blocked_reason=None,
            summary=summary,
        )
    elif scenario.form == 10:
        value = scenario.input_value
        final_thinking = (
            "The ID starts with ENSMUSG, which marks a mouse gene. The "
            "registry holds human genes only, so I will not call the tool."
        )
        final_answer = _answer(
            status="invalid_input",
            provenance=None,
            result={"input": value, "expected": "human_gene_id"},
            decision="defer",
            blocked_reason=(
                f"The input {value} is a mouse gene ID, and the registry "
                "holds human genes only."
            ),
            summary=(
                f"{value} is a mouse gene ID, and the registry holds human "
                "genes only. Give a human Ensembl gene ID to continue."
            ),
        )
    elif scenario.form == 12:
        value = scenario.input_value
        final_thinking = (
            "This is a transcript ID, not an Ensembl gene ID, so the lookup "
            "does not apply."
        )
        final_answer = _answer(
            status="invalid_input",
            provenance=None,
            result={"input": value, "expected": "human_gene_id"},
            decision="defer",
            blocked_reason=(
                f"The input {value} is a transcript ID, not a human gene ID."
            ),
            summary=(
                f"{value} is a transcript ID. Give a human Ensembl gene ID "
                "to continue."
            ),
        )
    elif scenario.form == 11:
        arguments = {"gene_id": scenario.input_value}
        call_result = runtime.execute(
            GENE_ID_TOOL, arguments, inject_backend_failure=True
        )
        if call_result.evidence is not None:
            raise CorpusBuildError("a failed call produced evidence")
        _append_call(
            messages,
            thinking=_pre_call_thinking(scenario),
            tool_name=GENE_ID_TOOL,
            arguments=arguments,
            call_result=call_result,
        )
        error_code = str(call_result.output["error_code"])
        final_thinking = (
            "The call failed because the registry could not be reached. This "
            "does not show that the ID is absent from release 116. I have no "
            "registry result, so I must defer."
        )
        final_answer = _answer(
            status="error",
            provenance=None,
            result={
                "failed_tool": GENE_ID_TOOL,
                "error_code": error_code,
                "gene_id": scenario.input_value,
            },
            decision="defer",
            blocked_reason=(
                f"The call to {GENE_ID_TOOL} failed with {error_code}, so "
                f"there is no registry result for {scenario.input_value}."
            ),
            summary=(
                f"The registry lookup for {scenario.input_value} failed "
                "because the backend was unreachable. This does not show "
                "that the ID is invalid. Retry the lookup to continue."
            ),
        )
    else:
        tool_name = SYMBOL_TOOL if scenario.form == 9 else GENE_ID_TOOL
        argument_name = "gene_symbol" if scenario.form == 9 else "gene_id"
        arguments = {argument_name: scenario.input_value}
        call_result = runtime.execute(tool_name, arguments)
        call_evidence = _evidence_row(call_result, row_id=row_id)
        if call_evidence is None:
            raise CorpusBuildError("a registry lookup produced no evidence")
        evidence.append(call_evidence)
        _append_call(
            messages,
            thinking=_pre_call_thinking(scenario),
            tool_name=tool_name,
            arguments=arguments,
            call_result=call_result,
        )
        output = call_result.output
        if scenario.form == 5:
            if output.get("status") != "not_found":
                raise CorpusBuildError("a retired ID unexpectedly resolved")
            value = str(output["gene_id"])
            if value.upper().endswith("_PAR_Y"):
                final_thinking = (
                    "The registry has no gene with this ID in release 116. "
                    "The _PAR_Y suffix marks the chromosome Y copy of a PAR "
                    "gene in an older GENCODE release. I will not map it to "
                    "another gene from memory, so I must defer."
                )
            else:
                final_thinking = (
                    "The registry has no gene with this ID in release 116. I "
                    "will not map it to another gene from memory, so I must "
                    "defer."
                )
            if scenario.split == "train" and final_reasoning:
                final_thinking = final_reasoning.format(gene_id=value)
            final_answer = _answer(
                status="not_found",
                provenance=output["provenance"],
                result={"gene_id": value},
                decision="defer",
                blocked_reason=(
                    f"The registry has no gene with the ID {value} in "
                    f"Ensembl release {release}."
                ),
                summary=(
                    f"{value} is not a gene ID in Ensembl release {release}. "
                    "Give the release 116 gene ID to continue."
                ),
            )
        elif scenario.form == 6:
            if output.get("status") != "invalid_input":
                raise CorpusBuildError("a malformed ID was not rejected")
            value = str(output["input"])
            final_thinking = (
                "The tool reports that the input is not a well-formed human "
                "gene ID. I will not guess a missing digit, so I must defer."
            )
            if scenario.split == "train" and final_reasoning:
                final_thinking = final_reasoning.format(input=value)
            final_answer = _answer(
                status="invalid_input",
                provenance=output["provenance"],
                result={"input": value, "expected": "human_gene_id"},
                decision="defer",
                blocked_reason=(
                    f"The input {value} is not a well-formed human gene ID."
                ),
                summary=(
                    f"{value} is not a well-formed human gene ID. A human "
                    "gene ID is ENSG followed by 11 digits. Give the full ID "
                    "to continue."
                ),
            )
        else:
            final_thinking, final_answer = _resolved_answer(
                scenario,
                output,
                release=release,
                no_symbol_reasoning=no_symbol_reasoning,
            )
            if scenario.split == "train" and scenario.form == 1 and final_reasoning:
                final_thinking = final_reasoning.format(
                    gene_id=output["gene_id"], symbol=output["gene_symbol"],
                )
            elif scenario.split == "train" and scenario.form == 3 and version_suffix_reasoning:
                final_thinking = version_suffix_reasoning[scenario.framing].format(
                    gene_id=output["gene_id"], symbol=output["gene_symbol"],
                )
            elif scenario.split == "train" and scenario.form == 8 and near_miss_reasoning:
                final_thinking = near_miss_reasoning[scenario.framing].format(
                    gene_id=output["gene_id"], symbol=output["gene_symbol"],
                )
            elif scenario.split == "train" and scenario.form == 9 and symbol_input_reasoning:
                final_thinking = symbol_input_reasoning[scenario.framing].format(
                    gene_id=output["gene_id"], symbol=output["gene_symbol"],
                )

    messages.append(
        {
            "role": "assistant",
            "thinking": final_thinking,
            "content": final_answer,
        }
    )
    blocked_code = {
        5: "not_found",
        6: "invalid_input",
        10: "out_of_scope_species",
        11: "tool_error",
        12: "wrong_input_type",
    }.get(scenario.form)
    metadata = {
        "book_mode": "tool_call",
        "step": "S1",
        "question_family": contract["curriculum"]["families"]["S1.2"][
            "question_family"
        ],
        "form": scenario.form,
        "source_trace": scenario.source_trace,
        "split_unit": scenario.metadata_split_unit,
        "blocked_reason_code": blocked_code,
        "schema_version": schema_version,
        "system_prompt_sha256": system_prompt_sha256(contract),
        "tool_contract_version": contract["runtime"]["tool_contract_version"],
        "ensembl_release": release,
        "identifier_registry_id": runtime.identifier_registry_id,
    }
    if scenario.form == 10:
        metadata["input_species"] = "mouse"
    elif scenario.form == 12:
        metadata["found_input_type"] = "transcript_id"
    row = {
        "row_id": row_id,
        "messages": messages,
        "tools": tools,
        "metadata": metadata,
    }
    return row, evidence


def _tool_calls(row: Mapping[str, Any]) -> list[str]:
    return [
        str(message["tool_call"]["name"])
        for message in row["messages"]
        if message.get("role") == "assistant" and "tool_call" in message
    ]


def _tool_arguments(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(message["tool_call"]["arguments"])
        for message in row["messages"]
        if message.get("role") == "assistant" and "tool_call" in message
    ]


MASKED_WORDING_TOLERANCE = 0.02
MASK_TOKEN = "<INPUT>"


def _user_message(row: Mapping[str, Any]) -> str:
    users = [
        message["content"]
        for message in row["messages"]
        if message.get("role") == "user"
    ]
    if len(users) != 1 or not isinstance(users[0], str):
        raise CorpusBuildError("an S1.2 row must have exactly one user message")
    return users[0]


def _masked_user_wording(
    row: Mapping[str, Any], scenario: Scenario
) -> str:
    """Mask the full scenario input and normalize only case/whitespace."""

    user = _user_message(row)
    request_suffix = f"corresponding to {scenario.input_value}"
    occurrences = user.count(request_suffix)
    if occurrences != 1:
        raise CorpusBuildError(
            f"row {row['row_id']} contains its request suffix "
            f"{occurrences} times"
        )
    masked = user.replace(
        request_suffix,
        f"corresponding to {MASK_TOKEN}",
    )
    if request_suffix in masked:
        raise CorpusBuildError(f"row {row['row_id']} did not mask its input")
    return " ".join(masked.casefold().split())


def _majority_label(counts: Mapping[int, int]) -> int:
    if not counts:
        raise CorpusBuildError("the wording classifier has no labels")
    return min(counts, key=lambda label: (-counts[label], label))


def _wording_tokens(masked_wording: str) -> tuple[str, ...]:
    """Return dependency-free lowercase word tokens for the leakage model."""

    return tuple(re.findall(r"[a-z]+", masked_wording))


def masked_wording_classifier_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    tolerance: float = MASKED_WORDING_TOLERANCE,
) -> dict[str, Any]:
    """Fit class-balanced multinomial Naive Bayes and test held-out splits.

    Uniform class priors intentionally remove the known form-allocation and
    direct/step imbalance, leaving token likelihoods to measure lexical form
    leakage.  Add-one smoothing and contract-order tie-breaking make the
    dependency-free implementation deterministic.  Held-out accuracy may not
    lift more than two absolute percentage points above the largest-form
    baseline.
    """

    if tolerance < 0.0 or tolerance > 1.0:
        raise CorpusBuildError("the wording-classifier tolerance is invalid")
    train_rows = rows_by_split["train"]
    train_scenarios = scenarios_by_split["train"]
    if len(train_rows) != len(train_scenarios):
        raise CorpusBuildError("train rows and scenarios differ")

    label_counts: Counter[int] = Counter()
    token_counts: dict[int, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[int] = Counter()
    vocabulary: set[str] = set()
    for row, scenario in zip(train_rows, train_scenarios, strict=True):
        label_counts[scenario.form] += 1
        tokens = _wording_tokens(_masked_user_wording(row, scenario))
        token_counts[scenario.form].update(tokens)
        token_totals[scenario.form] += len(tokens)
        vocabulary.update(tokens)
    labels = tuple(sorted(label_counts))
    if not vocabulary:
        raise CorpusBuildError("the wording classifier has no vocabulary")

    def predict(tokens: Sequence[str]) -> int:
        known = Counter(token for token in tokens if token in vocabulary)
        scored = []
        for label in labels:
            denominator = token_totals[label] + len(vocabulary)
            score = sum(
                count
                * math.log(
                    (token_counts[label][token] + 1) / denominator
                )
                for token, count in known.items()
            )
            scored.append((score, -label, label))
        return max(scored)[2]

    evaluations: dict[str, dict[str, Any]] = {}
    overall_pass = True
    for split in ("train", "validation", "test"):
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} rows and scenarios differ")
        split_labels = Counter(item.form for item in scenarios)
        largest_count = max(split_labels.values())
        correct = 0
        known_tokens = unknown_tokens = 0
        for row, scenario in zip(rows, scenarios, strict=True):
            tokens = _wording_tokens(_masked_user_wording(row, scenario))
            known_tokens += sum(token in vocabulary for token in tokens)
            unknown_tokens += sum(token not in vocabulary for token in tokens)
            prediction = predict(tokens)
            correct += prediction == scenario.form
        accuracy = correct / len(rows)
        baseline = largest_count / len(rows)
        gate_applied = split != "train"
        threshold = baseline + tolerance if gate_applied else None
        passed = accuracy <= threshold if threshold is not None else None
        if passed is not None:
            overall_pass &= passed
        evaluations[split] = {
            "row_count": len(rows),
            "correct": correct,
            "accuracy": accuracy,
            "largest_form": _majority_label(split_labels),
            "largest_form_baseline": baseline,
            "threshold": threshold,
            "gate_applied": gate_applied,
            "known_token_occurrences": known_tokens,
            "unknown_token_occurrences": unknown_tokens,
            "pass": passed,
        }
    result = {
        "algorithm": "class_balanced_multinomial_naive_bayes_v1",
        "mask_token": MASK_TOKEN,
        "tokenizer": "lowercase_ascii_words",
        "smoothing_alpha": 1,
        "class_prior": "uniform",
        "tie_break": "lowest_form_number",
        "absolute_tolerance": tolerance,
        "threshold_rule": (
            "held-out accuracy must be <= largest-form baseline + tolerance"
        ),
        "training_vocabulary_size": len(vocabulary),
        "splits": evaluations,
        "pass": overall_pass,
    }
    if not overall_pass:
        failed = {
            split: values
            for split, values in evaluations.items()
            if values["pass"] is False
        }
        raise CorpusBuildError(
            "masked user wording predicts S1.2 form above the allowed "
            f"baseline tolerance: {failed!r}"
        )
    return result


THINKING_FILTER_RULES = (
    "no_premature_identifier_symbol_or_value",
    "status_agrees_with_result",
    "decision_agrees_with_answer",
    "no_answer_schema_or_provenance_discussion",
)

_ANSWER_SCHEMA_PHRASES = (
    "answer schema",
    "answer object",
    "json object",
    "provenance",
    "evidence_id",
    "blocked_reason",
    "continuation_decision",
    "status field",
    "result field",
    "summary field",
)
_STATUS_CLAIM_PHRASES = {
    "resolved": (
        "the registry gives",
        "the registry maps",
        "the registry has this id",
        "the registry entries from the harness list this id",
        " is a symbol in release",
    ),
    "not_found": (
        "the registry has no gene",
        "the registry has no match",
    ),
    "invalid_input": (
        "not a well-formed human gene id",
        "marks a mouse gene",
        "this is a transcript id",
    ),
    "error": (
        "the call failed",
        "registry could not be reached",
    ),
}
_DECISION_CLAIM_PHRASES = {
    "continue": (
        "analysis can continue",
        "analysis continues",
        "next step can use",
        "next operation can use",
    ),
    "stop": ("task is complete",),
    "defer": ("must defer",),
}


def _protected_thinking_values(
    row: Mapping[str, Any], scenario: Scenario
) -> set[str]:
    """Return row-specific biological/runtime values thinking may reveal."""

    values = {
        scenario.input_value,
        scenario.fact.gene_id,
        scenario.fact.gene_symbol,
        scenario.fact.transcript_id,
    }
    if scenario.distractor is not None:
        values.update(
            {
                scenario.distractor.gene_id,
                scenario.distractor.gene_symbol,
                scenario.distractor.transcript_id,
            }
        )

    protected_keys = {
        "gene_id",
        "gene_symbol",
        "input",
        "error_code",
        "multiplex_id",
        "candidate_gene_ids",
        "present_gene_ids",
        "absent_gene_ids",
    }

    def collect(value: Any, *, active: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                collect(item, active=active or str(key) in protected_keys)
        elif isinstance(value, list):
            for item in value:
                collect(item, active=active)
        elif active and isinstance(value, (str, int, float)):
            values.add(str(value))

    for message in row["messages"]:
        if message.get("role") == "tool":
            collect(message.get("content"))
    collect(row["messages"][-1].get("content"))
    return {
        value
        for value in values
        if isinstance(value, str) and len(value.strip()) >= 2
    }


def _thinking_status_claims(thinking: str) -> set[str]:
    lowered = thinking.casefold()
    claims = {
        status
        for status, phrases in _STATUS_CLAIM_PHRASES.items()
        if any(phrase in lowered for phrase in phrases)
    }
    explicit = {
        "resolved": (r"\bresolved\b",),
        "not_found": (r"\bnot_found\b", r"\bnot found\b"),
        "invalid_input": (r"\binvalid_input\b", r"\binvalid input\b"),
        "error": (r"\berror\b",),
    }
    claims.update(
        status
        for status, patterns in explicit.items()
        if any(re.search(pattern, lowered) for pattern in patterns)
    )
    return claims


def _thinking_decision_claims(thinking: str) -> set[str]:
    lowered = thinking.casefold()
    claims = {
        decision
        for decision, phrases in _DECISION_CLAIM_PHRASES.items()
        if any(phrase in lowered for phrase in phrases)
    }
    claims.update(
        decision
        for decision in ("continue", "stop", "defer")
        if re.search(rf"\b{decision}\b", lowered)
    )
    return claims


def _contains_protected_value(text: str, value: str) -> bool:
    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(value)
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def filter_row_thinking(
    row: Mapping[str, Any], scenario: Scenario, *, final_reasoning: str | None = None,
) -> int:
    """Check all rules, allowing the pinned form-6 final decision-field phrase."""

    protected_values = _protected_thinking_values(row, scenario)
    visible = ""
    checked = 0
    for message in row["messages"]:
        role = message.get("role")
        if role != "assistant":
            content = message.get("content")
            visible += " " + (
                content
                if isinstance(content, str)
                else _canonical_json(content, sort_keys=False)
            )
            continue

        thinking = message.get("thinking")
        if not isinstance(thinking, str) or not thinking.strip():
            raise CorpusBuildError(
                f"row {row['row_id']} has an assistant action without thinking"
            )
        checked += 1
        lowered = thinking.casefold()
        schema_text = lowered
        if (
            scenario.split == "train" and scenario.form == 6 and final_reasoning
            and message is row["messages"][-1]
            and thinking == final_reasoning.format(input=scenario.input_value)
        ):
            # The reviewed template names this field to correct invented decision values.
            schema_text = schema_text.replace("continuation_decision:defer", "")
        if any(phrase in schema_text for phrase in _ANSWER_SCHEMA_PHRASES):
            raise CorpusBuildError(
                f"row {row['row_id']} thinking discusses answer schema/provenance"
            )

        for value in protected_values:
            if (
                _contains_protected_value(thinking, value)
                and not _contains_protected_value(visible, value)
            ):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking reveals {value!r} before "
                    "the prompt or tool result"
                )

        status_claims = _thinking_status_claims(thinking)
        if "tool_call" in message:
            if status_claims:
                raise CorpusBuildError(
                    f"row {row['row_id']} predicts status before its tool call"
                )
        else:
            answer = message.get("content")
            if not isinstance(answer, Mapping):
                raise CorpusBuildError(
                    f"row {row['row_id']} final action has no answer object"
                )
            expected_status = str(answer["status"])
            if status_claims.difference({expected_status}):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking status {status_claims!r} "
                    f"disagrees with {expected_status!r}"
                )
            decision_claims = _thinking_decision_claims(thinking)
            expected_decision = str(answer["continuation_decision"])
            if decision_claims.difference({expected_decision}):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking decision "
                    f"{decision_claims!r} disagrees with "
                    f"{expected_decision!r}"
                )
    return checked


def _without_evidence_ids(value: Any) -> Any:
    """Recursively remove evidence IDs, including serialized harness JSON."""

    if isinstance(value, Mapping):
        return {
            key: _without_evidence_ids(item)
            for key, item in value.items()
            if key != "evidence_id"
        }
    if isinstance(value, list):
        return [_without_evidence_ids(item) for item in value]
    if isinstance(value, str) and "evidence_id" in value:
        header, separator, payload = value.partition("\n")
        if separator:
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                pass
            else:
                return header + "\n" + _canonical_json(
                    _without_evidence_ids(parsed), sort_keys=False
                )
        return re.sub(
            r'("evidence_id"\s*:\s*)"[^"]*"',
            r'\1"<REMOVED>"',
            value,
        )
    return value


def _prompt_and_outcome_signatures(
    row: Mapping[str, Any]
) -> tuple[str, str, str]:
    """Return normalized visible-prompt, tool-outcome, and combined keys."""

    prompt = []
    outcomes = []
    reached_user = False
    for message in row["messages"]:
        role = message.get("role")
        if not reached_user:
            prompt.append(_without_evidence_ids(message))
            reached_user = role == "user"
            continue
        if role == "tool":
            outcomes.append(
                {
                    "tool_name": message["name"],
                    "result_status": message["content"]["status"],
                }
            )
        # The final assistant thinking and answer are intentionally excluded.
    prompt_key = _canonical_json(prompt, sort_keys=False)
    outcome_key = _canonical_json(outcomes, sort_keys=False)
    combined_key = _canonical_json(
        {"prompt": prompt, "tool_outcomes": outcomes}, sort_keys=False
    )
    return prompt_key, outcome_key, combined_key


def duplicate_prompt_outcome_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    train_retry_cases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reject evidence-ID-hidden duplicates and unintended prompt pairs."""

    split_results: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} rows and scenarios differ")
        retry_cases = {
            scenario.scenario_id: (train_retry_cases or {}).get(str(row["row_id"]))
            for scenario, row in zip(scenarios, rows, strict=True)
        } if split == "train" else {}
        prompt_groups: dict[str, list[tuple[Scenario, str]]] = defaultdict(list)
        combined_groups: dict[str, list[Scenario]] = defaultdict(list)
        for row, scenario in zip(rows, scenarios, strict=True):
            prompt_key, outcome_key, combined_key = (
                _prompt_and_outcome_signatures(row)
            )
            prompt_groups[prompt_key].append((scenario, outcome_key))
            combined_groups[combined_key].append(scenario)

        duplicate_outcomes = [
            group for group in combined_groups.values() if len(group) > 1
        ]
        if duplicate_outcomes:
            example = [
                item.scenario_id for item in duplicate_outcomes[0]
            ]
            raise CorpusBuildError(
                f"{split} repeats a prompt and tool outcome after removing "
                f"evidence IDs: {example!r}"
            )

        intended_pairs = 0
        for group in prompt_groups.values():
            if len(group) == 1:
                continue
            if len(group) != 2:
                raise CorpusBuildError(
                    f"{split} has a prompt shared by {len(group)} rows"
                )
            success = next(
                ((item, outcome) for item, outcome in group if item.form == 1),
                None,
            )
            failure = next(
                (
                    (item, outcome)
                    for item, outcome in group
                    if item.form == 11
                ),
                None,
            )
            if success is None or failure is None:
                raise CorpusBuildError(f"{split} has an unintended prompt pair")
            success_scenario, success_outcome = success
            failure_scenario, failure_outcome = failure
            if (
                failure_scenario.paired_success_id
                != success_scenario.scenario_id
                or success_outcome == failure_outcome
            ):
                raise CorpusBuildError(
                    f"{split} has an invalid success/failure prompt pair"
                )
            success_results = json.loads(success_outcome)
            failure_results = json.loads(failure_outcome)
            case = retry_cases.get(failure_scenario.scenario_id)
            wanted_failures = [{"tool_name": GENE_ID_TOOL, "result_status": "error"}]
            if case in {"recovered", "exhausted"}:
                wanted_failures.append({
                    "tool_name": GENE_ID_TOOL,
                    "result_status": "resolved" if case == "recovered" else "error",
                })
            if (
                len(success_results) != 1
                or success_results[0]["tool_name"] != GENE_ID_TOOL
                or success_results[0]["result_status"] != "resolved"
                or failure_results != wanted_failures
            ):
                raise CorpusBuildError(
                    f"{split} minimal pair has the wrong tool outcomes"
                )
            intended_pairs += 1

        split_results[split] = {
            "row_count": len(rows),
            "normalized_prompt_count": len(prompt_groups),
            "normalized_prompt_tool_outcome_count": len(combined_groups),
            "duplicate_prompt_tool_outcome_count": 0,
            "intended_success_failure_pair_count": intended_pairs,
            "pass": True,
        }
    return {
        "normalization": (
            "remove evidence_id recursively; ignore final assistant thinking "
            "and answer; retain only each tool name and result status"
        ),
        "splits": split_results,
        "pass": True,
    }


def audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    expected_counts: Mapping[str, Mapping[int, int]],
    *,
    framing_shares: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    train_form_prompts: Mapping[int, Mapping[str, Mapping[str, Sequence[str]]]] | None = None,
    train_final_reasoning: Mapping[int, str] | None = None,
    train_retry_cases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Enforce S1.2 semantics beyond the generic row validator."""

    framing_shares = framing_shares or {"direct": 0.5, "step": 0.5}
    row_ids: set[str] = set()
    split_units: dict[str, set[str]] = {}
    expected_calls = {
        1: [GENE_ID_TOOL],
        2: [GENE_ID_TOOL],
        3: [GENE_ID_TOOL],
        4: [GENE_ID_TOOL],
        5: [GENE_ID_TOOL],
        6: [GENE_ID_TOOL],
        7: [],
        8: [GENE_ID_TOOL],
        9: [SYMBOL_TOOL],
        10: [],
        11: [GENE_ID_TOOL],
        12: [],
    }
    expected_status = {
        1: "resolved",
        2: "resolved",
        3: "resolved",
        4: "resolved",
        5: "not_found",
        6: "invalid_input",
        7: "resolved",
        8: "resolved",
        9: "resolved",
        10: "invalid_input",
        11: "error",
        12: "invalid_input",
    }
    status_counts: dict[str, dict[str, int]] = {}
    call_counts: dict[str, dict[str, int]] = {}
    framing_counts: dict[str, dict[str, int]] = {}
    case_counts: dict[str, dict[str, int]] = {}
    pair_counts: dict[str, int] = {}
    template_counts: dict[str, dict[str, int]] = {}
    renamed_counts: dict[str, dict[str, int]] = {}
    thinking_counts: dict[str, dict[str, int]] = {}

    for split in ("train", "validation", "test"):
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} rows and scenarios differ")
        actual_forms = Counter(int(row["metadata"]["form"]) for row in rows)
        if dict(sorted(actual_forms.items())) != dict(expected_counts[split]):
            raise CorpusBuildError(f"{split} form counts changed")

        successes = {
            scenario.scenario_id: (scenario, row)
            for scenario, row in zip(scenarios, rows, strict=True)
            if scenario.form == 1
        }
        units: set[str] = set()
        prompt_groups: dict[str, list[Scenario]] = defaultdict(list)
        pairs = 0
        case_eligible = case_changed = 0
        renamed = Counter()
        thinking_steps = 0
        for scenario, row in zip(scenarios, rows, strict=True):
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise CorpusBuildError(f"duplicate row ID {row_id}")
            row_ids.add(row_id)
            units.update(scenario.split_units)
            thinking_steps += filter_row_thinking(
                row, scenario, final_reasoning=(train_final_reasoning or {}).get(scenario.form),
            )

            calls = _tool_calls(row)
            arguments = _tool_arguments(row)
            retry_case = (train_retry_cases or {}).get(row_id) if split == "train" else None
            wanted_calls = expected_calls[scenario.form]
            wanted_status = expected_status[scenario.form]
            if retry_case is not None:
                if scenario.form != 11:
                    raise CorpusBuildError("an ID lookup retry changed another form")
                audit_retry_row(row, retry_case)
                wanted_calls = [GENE_ID_TOOL] * (1 if retry_case == "no_retry" else 2)
                wanted_status = "resolved" if retry_case == "recovered" else "error"
            if calls != wanted_calls:
                raise CorpusBuildError(
                    f"form {scenario.form} emitted calls {calls!r}"
                )
            if arguments:
                argument_name = (
                    "gene_symbol" if scenario.form == 9 else "gene_id"
                )
                if arguments[0] != {argument_name: scenario.input_value}:
                    raise CorpusBuildError(
                        f"form {scenario.form} changed the tool argument"
                    )
            answer = row["messages"][-1]["content"]
            if answer["status"] != wanted_status:
                raise CorpusBuildError(
                    f"form {scenario.form} emitted status {answer['status']!r}"
                )
            if scenario.form in {1, 2, 3, 4, 7, 8, 9}:
                if scenario.form == 7:
                    harness = next(
                        message["content"]
                        for message in row["messages"]
                        if message["role"] == "harness"
                    )
                    _, _, serialized = harness.partition("\n")
                    harness_payload = json.loads(serialized)
                    source_result = harness_payload["entries"][0]
                else:
                    source_result = next(
                        message["content"]
                        for message in reversed(row["messages"])
                        if message["role"] == "tool"
                    )
                expected_result = {
                    "gene_id": source_result["gene_id"],
                    "gene_symbol": source_result["gene_symbol"],
                }
                if answer["result"] != expected_result:
                    raise CorpusBuildError(
                        f"form {scenario.form} answer changed the successful "
                        "registry result"
                    )
            if scenario.form == 4 and answer["result"] != {
                "gene_id": scenario.fact.gene_id,
                "gene_symbol": None,
            }:
                raise CorpusBuildError("a no-symbol row changed its valid ID")
            if scenario.form == 5 and set(answer["result"]) != {"gene_id"}:
                raise CorpusBuildError("a retired-ID row supplied a successor")
            if scenario.form == 6 and answer["result"]["input"] != scenario.input_value:
                raise CorpusBuildError("a malformed ID was repaired")
            if scenario.form in {10, 12} and calls:
                raise CorpusBuildError("a recognized wrong-type input made a call")

            prompt_messages = []
            for message in row["messages"]:
                prompt_messages.append(message)
                if message["role"] == "user":
                    break
            prompt = _canonical_json(prompt_messages, sort_keys=False)
            prompt_groups[prompt].append(scenario)

            if scenario.form == 11:
                paired = successes.get(str(scenario.paired_success_id))
                if paired is None:
                    raise CorpusBuildError("a failure has no retained success pair")
                success_scenario, success_row = paired
                same_prompt = (
                    row["messages"][1]["content"]
                    == success_row["messages"][1]["content"]
                )
                if split == "train" and train_form_prompts:
                    # Reviewed wrappers can differ; both members must still
                    # render their pinned template at the same position.
                    same_prompt = all(
                        member["messages"][1]["content"] == _question(
                            case, prompts=train_form_prompts.get(case.form, prompts),
                        )
                        for case, member in (
                            (scenario, row), (success_scenario, success_row)
                        )
                    ) and _question(scenario, prompts=prompts) == _question(
                        success_scenario, prompts=prompts,
                    )
                if (
                    not same_prompt
                    or scenario.input_value != success_scenario.input_value
                    or scenario.template_index != success_scenario.template_index
                    or scenario.framing != success_scenario.framing
                ):
                    raise CorpusBuildError("a failure pair changed its prompt")
                pairs += 1

            if scenario.fact.species == "human":
                case_eligible += 1
                case_changed += (
                    scenario.input_value
                    != _canonical_input(scenario.fact, scenario.form)
                )
            if scenario.form in {1, 2, 3} and scenario.fact.renamed:
                renamed[f"S1.2.{scenario.form}"] += 1

        for prompt_scenarios in prompt_groups.values():
            if len(prompt_scenarios) == 1:
                continue
            if len(prompt_scenarios) != 2:
                raise CorpusBuildError(f"duplicate non-paired prompt in {split}")
            success = next(
                (item for item in prompt_scenarios if item.form == 1), None
            )
            failure = next(
                (item for item in prompt_scenarios if item.form == 11), None
            )
            if (
                success is None
                or failure is None
                or failure.paired_success_id != success.scenario_id
            ):
                raise CorpusBuildError(f"duplicate non-paired prompt in {split}")

        expected_framing = largest_remainder_counts(
            len(rows), framing_shares
        )
        actual_framing = Counter(item.framing for item in scenarios)
        if dict(actual_framing) != expected_framing:
            raise CorpusBuildError(
                f"{split} framing counts are {dict(actual_framing)!r}, "
                f"expected {expected_framing!r}"
            )
        for form in (1, 2, 3):
            if renamed[f"S1.2.{form}"] < 1:
                raise CorpusBuildError(
                    f"{split} form {form} has no renamed-gene coverage"
                )

        split_units[split] = units
        pair_counts[split] = pairs
        if pairs != expected_counts[split][11]:
            raise CorpusBuildError(f"{split} failure pair count changed")
        status_counts[split] = dict(
            sorted(
                Counter(
                    str(row["messages"][-1]["content"]["status"])
                    for row in rows
                ).items()
            )
        )
        call_counts[split] = dict(
            sorted(Counter(str(len(_tool_calls(row))) for row in rows).items())
        )
        framing_counts[split] = dict(sorted(actual_framing.items()))
        case_counts[split] = {
            "eligible": case_eligible,
            "perturbed": case_changed,
        }
        renamed_counts[split] = dict(sorted(renamed.items()))
        thinking_counts[split] = {
            "rows_checked": len(rows),
            "thinking_steps_checked": thinking_steps,
            "violations": 0,
        }
        coverage = Counter(
            f"S1.2.{scenario.form}:{scenario.framing}:"
            f"{scenario.template_index}"
            for scenario in scenarios
        )
        template_counts[split] = dict(sorted(coverage.items()))
        for form in range(1, 13):
            frames = (
                ("direct",)
                if form in {1, 11}
                else ("step",)
                if form == 2
                else ("direct", "step")
            )
            for framing in frames:
                used = {
                    scenario.template_index
                    for scenario in scenarios
                    if scenario.form == form and scenario.framing == framing
                }
                expected = set(range(len(prompts[split][framing])))
                if used != expected:
                    raise CorpusBuildError(
                        f"{split} form {form} {framing} misses templates"
                    )

    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = split_units[first].intersection(split_units[second])
        if overlap:
            raise CorpusBuildError(
                f"identity leakage between {first} and {second}: "
                f"{sorted(overlap)[:3]!r}"
            )

    duplicate_audit = duplicate_prompt_outcome_audit(
        rows_by_split, scenarios_by_split, train_retry_cases=train_retry_cases,
    )
    wording_audit = masked_wording_classifier_audit(
        rows_by_split, scenarios_by_split
    )
    return {
        "status_counts": status_counts,
        "tool_call_counts": call_counts,
        "framing_counts": framing_counts,
        "case_counts": case_counts,
        "failure_pair_counts": pair_counts,
        "prompt_template_counts": template_counts,
        "renamed_gene_counts": renamed_counts,
        "thinking_filter": {
            "rules": list(THINKING_FILTER_RULES),
            "splits": thinking_counts,
            "pass": True,
        },
        "duplicate_prompt_tool_outcome": duplicate_audit,
        "masked_wording_classifier": wording_audit,
        "split_unit_hashes": {
            split: _stable_sha256(sorted(units))
            for split, units in split_units.items()
        },
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row, sort_keys=False) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
    validate: bool = True,
    overlay: Mapping[str, str] | None = None,
    prompt_banks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, validate, and atomically publish all three S1.2 splits.

    A pinned prompt bank replaces only the training wrappers and assigns them
    round-robin, and from S1 v5 it may also give the S1.2.4 training
    reasoning; validation and test keep their wrappers, assignment, and
    reasoning, so their rows do not change. The S1 v5 overlay's
    ``train_form_counts`` moves training rows between forms.
    S1 v6 can replace S1.2.1/S1.2.3/S1.2.4/S1.2.6/S1.2.8/S1.2.9 training wrappers and final
    thinking, plus S1.2.5 final thinking only. S1.2.11 can teach one explicitly
    allowed retry, allocating new recovery evidence after all baseline rows.
    """

    source_index_path = source_index_path.resolve()
    payload = check_source_index(source_index_path, contract)
    curriculum = contract["curriculum"]
    settings = curriculum["families"]["S1.2"]
    seed = int(settings["assignment_seed"])
    prompts = (
        identity_prompts(prompt_banks, IDENTITY_PROMPTS)
        if prompt_banks is not None
        else IDENTITY_PROMPTS
    )
    train_form_pins = settings.get("train_prompt_banks", {})
    train_decorrelate = settings.get("train_decorrelate", False)
    if train_decorrelate and train_form_pins:
        raise CorpusBuildError("decorrelated training forbids form-specific wrapper banks")
    train_form_prompts = {
        form: identity_prompts(pin, IDENTITY_PROMPTS)
        for form, pin in train_form_pins.items()
    }
    for form, form_prompts in train_form_prompts.items():
        if any(
            len(form_prompts["train"][framing]) != len(prompts["train"][framing])
            for framing in ("direct", "step")
        ):
            raise CorpusBuildError(
                f"form {form} training wrappers change the template count"
            )
    train_final_reasoning = settings.get("train_final_reasoning", {})
    version_suffix_reasoning = settings.get("train_version_suffix_reasoning")
    near_miss_reasoning = settings.get("train_near_miss_reasoning")
    symbol_input_reasoning = settings.get("train_symbol_input_reasoning")
    balanced_splits = ("train",) if prompt_banks is not None else ()
    no_symbol_reasoning = (
        {
            framing: bank_reasoning(prompt_banks, f"s1_2_no_symbol_{framing}")
            for framing in ("direct", "step")
        }
        if prompt_banks is not None
        else None
    )
    reviewed_no_symbol_reasoning = settings.get("train_no_symbol_reasoning")
    if reviewed_no_symbol_reasoning is not None:
        no_symbol_reasoning = reviewed_no_symbol_reasoning
    row_counts = split_row_counts(
        int(settings["train_rows"]), curriculum["split_fractions"]
    )
    shared_counts = {
        split: form_counts(row_counts[split], settings, curriculum)
        for split in ("train", "validation", "test")
    }
    counts_by_split = dict(shared_counts)
    if "train_form_counts" in settings:
        counts_by_split["train"] = override_train_counts(
            shared_counts["train"], settings["train_form_counts"]
        )

    registry_spec = contract["coordinate_system"]["identifier_registry"]
    human_path = _configured_path(str(registry_spec["source_path"])).resolve()
    actual_human_hash = file_sha256(human_path)
    if actual_human_hash != registry_spec["source_sha256"]:
        raise CorpusBuildError(
            f"human source SHA-256 is {actual_human_hash}, expected "
            f"{registry_spec['source_sha256']}"
        )
    human_identifiers = read_ensembl_gtf_identifiers(human_path)

    mouse_spec = contract["coordinate_system"]["out_of_scope_species"][
        "mouse"
    ]
    mouse_path = _configured_path(str(mouse_spec["source_path"])).resolve()
    actual_mouse_hash = file_sha256(mouse_path)
    if actual_mouse_hash != mouse_spec["source_sha256"]:
        raise CorpusBuildError(
            f"mouse source SHA-256 is {actual_mouse_hash}, expected "
            f"{mouse_spec['source_sha256']}"
        )
    if mouse_spec["gene_id_pattern"] != MOUSE_GENE_ID_PATTERN.pattern:
        raise CorpusBuildError("mouse gene-ID pattern changed")
    mouse_genes = read_ensembl_gtf(
        mouse_path,
        gene_id_pattern=MOUSE_GENE_ID_PATTERN,
        species="mouse",
    )

    pools = build_fact_pools(
        payload,
        human_identifiers,
        mouse_genes,
        s0_settings=curriculum["families"]["S0"],
        seed=seed,
        fractions=curriculum["split_fractions"],
    )
    scenarios_by_split = select_scenarios(
        pools,
        counts_by_split,
        seed=seed,
        case_rate=curriculum["identity_case_perturbation"],
        framing_shares=curriculum["framing"],
        prompts=prompts,
        balanced_splits=balanced_splits,
        train_decorrelate=train_decorrelate,
    )

    # The splits share one evidence-ID sequence, train first. Revised train
    # counts would shift every held-out ID, so replay the shared-allocation
    # train rows to advance the sequence, build validation and test with the
    # IDs they always had, and build the revised train rows last. The
    # aggregate train copy re-keys train evidence IDs anyway.
    replay: list[Scenario] = []
    build_order = ("train", "validation", "test")
    if counts_by_split["train"] != shared_counts["train"]:
        replay = select_scenarios(
            pools,
            shared_counts,
            seed=seed,
            case_rate=curriculum["identity_case_perturbation"],
            framing_shares=curriculum["framing"],
            prompts=prompts,
            balanced_splits=balanced_splits,
        )["train"]
        build_order = ("validation", "test", "train")

    runtime = WorldModelToolRuntime.from_source_index(
        source_index_path,
        contract=contract,
        evidence_id_factory=EvidenceSequence(),
    )
    tools = tool_definitions(contract)
    for scenario in replay:
        build_row(
            scenario,
            runtime=runtime,
            contract=contract,
            tools=tools,
            prompts=prompts,
        )
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    evidence_calls: list[list[str]] = []
    for split in build_order:
        rows = []
        for scenario in scenarios_by_split[split]:
            row, row_evidence = build_row(
                scenario,
                runtime=runtime,
                contract=contract,
                tools=tools,
                prompts=(train_form_prompts.get(scenario.form, prompts) if split == "train" else prompts),
                no_symbol_reasoning=no_symbol_reasoning,
                final_reasoning=train_final_reasoning.get(scenario.form),
                version_suffix_reasoning=version_suffix_reasoning,
                near_miss_reasoning=near_miss_reasoning,
                symbol_input_reasoning=symbol_input_reasoning,
            )
            rows.append(row)
            evidence_calls.extend(row_evidence)
        rows_by_split[split] = rows

    retry_pin = settings.get("train_lookup_retries")
    train_retry_cases = {}
    if retry_pin is not None:
        retry_policy = load_retry_policy(retry_pin, tool_name=GENE_ID_TOOL)
        failures = [row for row in rows_by_split["train"] if row["metadata"]["form"] == 11]
        train_retry_cases = assign_retry_cases(failures, tool_name=GENE_ID_TOOL)
        if Counter(train_retry_cases.values()) != retry_policy["case_counts"]:
            raise CorpusBuildError("ID lookup retry allocation differs from its pinned counts")
        revised = []
        for row in rows_by_split["train"]:
            case = train_retry_cases.get(row["row_id"])
            if case is not None:
                row, added_evidence = render_retry_case(
                    row, case=case, runtime=runtime, policy=retry_policy,
                )
                evidence_calls.extend(added_evidence)
            revised.append(row)
        rows_by_split["train"] = revised

    evidence_ids = [item[0] for item in evidence_calls]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise CorpusBuildError("an evidence ID appears more than once")
    audit = audit_rows(
        rows_by_split,
        scenarios_by_split,
        counts_by_split,
        framing_shares=curriculum["framing"],
        prompts=prompts,
        train_form_prompts=train_form_prompts,
        train_final_reasoning=train_final_reasoning,
        train_retry_cases=train_retry_cases,
    )
    decorrelation_audit = (
        _audit_train_decorrelation(rows_by_split["train"], scenarios_by_split["train"], prompts)
        if train_decorrelate else None
    )

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".s1_2_build_", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        paths = {
            split: temporary / f"{split}.jsonl"
            for split in ("train", "validation", "test")
        }
        for split, path in paths.items():
            _write_jsonl(path, rows_by_split[split])
            if validate:
                _validate_corpus(path)

        form_count_manifest = {
            split: {
                f"S1.2.{form}": count
                for form, count in counts_by_split[split].items()
            }
            for split in ("train", "validation", "test")
        }
        pool_counts = {
            split: {
                name: len(facts)
                for name, facts in sorted(pools[split].items())
            }
            for split in ("train", "validation", "test")
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "dataset_id": settings["dataset_id"],
            "section": "S1.2",
            "implementation": {
                "builder": {
                    "path": str(
                        Path(__file__).resolve().relative_to(REPOSITORY_ROOT)
                    ),
                    "sha256": file_sha256(Path(__file__).resolve()),
                },
                "tool_runtime": {
                    "path": "runtime/world_model/tools.py",
                    "sha256": file_sha256(Path(__file__).with_name("tools.py")),
                },
            },
            "contract": {
                "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
                "sha256": file_sha256(CONTRACT_PATH),
                "row_schema_version": contract["runtime"][
                    "row_schema_version"
                ],
                "tool_contract_version": contract["runtime"][
                    "tool_contract_version"
                ],
                **({"s1_overlay": dict(overlay)} if overlay else {}),
            },
            "sources": {
                "source_index_sha256": runtime.source_index_sha256,
                "human_gtf": {
                    "file_name": human_path.name,
                    "sha256": actual_human_hash,
                    "gene_version_count": len(human_identifiers.genes),
                    "transcript_count": len(human_identifiers.transcripts),
                    "identifier_registry_id": runtime.identifier_registry_id,
                },
                "mouse_gtf": {
                    "file_name": mouse_path.name,
                    "sha256": actual_mouse_hash,
                    "identifier_registry_id": mouse_spec[
                        "identifier_registry_id"
                    ],
                },
            },
            "assignment": {
                "seed": seed,
                "split_fractions": dict(curriculum["split_fractions"]),
                "split_unit_sha256": audit["split_unit_hashes"],
                "source_pool_counts": pool_counts,
            },
            "failure_injection": {
                **dict(settings["failure_injection"]),
                "share_of_tool_call_rows": curriculum[
                    "tool_failure_share"
                ],
                "pair_counts": audit["failure_pair_counts"],
            },
            "added_form_rows": {
                "train": int(settings["added_form_rows"][
                    "transcript_wrong_type"
                ]),
                "validation": counts_by_split["validation"][12],
                "test": counts_by_split["test"][12],
            },
            "prompt_templates_sha256": _stable_sha256(prompts),
            "prompt_template_counts": audit["prompt_template_counts"],
            "row_counts": row_counts,
            "form_counts": form_count_manifest,
            "status_counts": audit["status_counts"],
            "tool_call_counts": audit["tool_call_counts"],
            "framing_counts": audit["framing_counts"],
            "renamed_gene_counts": audit["renamed_gene_counts"],
            "masked_wording_classifier": audit[
                "masked_wording_classifier"
            ],
            "thinking_filter": audit["thinking_filter"],
            "duplicate_prompt_tool_outcome": audit[
                "duplicate_prompt_tool_outcome"
            ],
            "case_perturbation": {
                "target_share": curriculum["identity_case_perturbation"],
                "counts": audit["case_counts"],
            },
            "evidence": {
                "sources": payload["sources"],
                "call_fields": ["evidence_id", "row_id", "producer"],
                "calls": evidence_calls,
            },
            "files": {
                f"{split}.jsonl": {
                    "row_count": row_counts[split],
                    "sha256": file_sha256(path),
                }
                for split, path in paths.items()
            },
        }
        if decorrelation_audit is not None:
            manifest["train_decorrelation"] = decorrelation_audit
        if prompt_banks is not None:
            manifest["prompt_bank"] = prompt_bank_manifest(prompt_banks)
        if train_form_pins:
            manifest["train_prompt_banks"] = {
                str(form): prompt_bank_manifest(pin)
                for form, pin in train_form_pins.items()
            }
        if train_final_reasoning:
            manifest["train_final_reasoning"] = {
                str(form): text for form, text in train_final_reasoning.items()
            }
        if version_suffix_reasoning is not None:
            manifest["train_version_suffix_reasoning"] = dict(version_suffix_reasoning)
        if reviewed_no_symbol_reasoning is not None:
            manifest["train_no_symbol_reasoning"] = dict(reviewed_no_symbol_reasoning)
        if near_miss_reasoning is not None:
            manifest["train_near_miss_reasoning"] = dict(near_miss_reasoning)
        if symbol_input_reasoning is not None:
            manifest["train_symbol_input_reasoning"] = dict(symbol_input_reasoning)
        if retry_pin is not None:
            manifest["train_lookup_retries"] = {
                **dict(retry_pin), "max_retries": 1,
                "case_counts": dict(sorted(Counter(train_retry_cases.values()).items())),
                "case_row_ids_sha256": _stable_sha256(train_retry_cases),
                "implementation_sha256": file_sha256(Path(__file__).with_name("s1_lookup_retry.py")),
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
        help="skip the semantic validator (intended only for focused tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
    settings = contract["curriculum"]["families"]["S1.2"]
    output_directory = args.output_directory or _configured_path(
        settings["output_directory"]
    )
    manifest = build_corpus(
        source_index,
        output_directory,
        contract=contract,
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
