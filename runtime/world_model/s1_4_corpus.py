"""Build the canonical S1.4 cross-context alignment corpus."""

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
    system_prompt_sha256,
    tool_definitions,
)
from .s0_corpus import (
    EvidenceSequence,
    assign_evaluation_components,
    build_identity_components,
)
from .s1_overlay import apply_s1_overlay, overlay_pin
from .s1_prompt_banks import prompt_bank_manifest, wrapper_prompts
from .source_index import check_source_index
from .sources import file_sha256, read_ensembl_gtf_identifiers
from .tools import (
    GENE_ID_TOOL,
    SYMBOL_TOOL,
    ToolCallResult,
    WorldModelToolRuntime,
)


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-4-corpus-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-4-manifest-v1"
MYGENE_SOURCE_SHA256 = (
    "44f2fbcea56595caea5f59651532cfea50ce033296ce441e3b539a9f5859740c"
)
SPLITS = ("train", "validation", "test")

FORM_BY_BUCKET = {
    "symbol_id_match": 1,
    "symbol_id_mismatch": 2,
    "id_notation_match": 3,
    "id_near_miss": 4,
    "historical_symbol_unresolved": 5,
    "ambiguous_symbol": 6,
    "registry_block_match": 7,
    "mygene_id_match": 8,
    "backend_unavailable": 9,
}
BUCKET_BY_FORM = {value: key for key, value in FORM_BY_BUCKET.items()}
TOOL_FORMS = frozenset({1, 2, 5, 6, 8, 9})

# Each split has its own wrappers. The request payload holds all form-specific
# structure. Each wrapper can therefore occur with each form.
ALIGNMENT_PROMPTS = {
    "train": {
        "direct": (
            "Check this gene alignment request: {request}.",
            "Answer this gene comparison: {request}.",
            "Compare these supplied gene mentions: {request}.",
            "Determine whether these gene mentions align: {request}.",
        ),
        "step": (
            "Before the next step, check this alignment: {request}.",
            "The next operation needs this gene comparison: {request}.",
            "Prepare this gene alignment for the next operation: {request}.",
            "Check this comparison so analysis can proceed: {request}.",
        ),
    },
    "validation": {
        "direct": (
            "Find the answer to this gene alignment request: {request}.",
            "Return the result of this gene comparison: {request}.",
            "Work out whether these gene mentions align: {request}.",
            "Supply the answer for this gene comparison: {request}.",
        ),
        "step": (
            "Check this gene alignment before the analysis continues: {request}.",
            "This comparison is needed for the next step: {request}.",
            "Before continuing, compare these gene mentions: {request}.",
            "Prepare this alignment for the next action: {request}.",
        ),
    },
    "test": {
        "direct": (
            "What is the answer to this gene comparison: {request}?",
            "Give the result of this gene alignment: {request}.",
            "Answer the following request about two gene mentions: {request}.",
            "Identify whether these gene mentions align: {request}.",
        ),
        "step": (
            "The workflow first needs this gene alignment checked: {request}.",
            "Answer this comparison for the next action: {request}.",
            "Before the workflow advances, compare these gene mentions: {request}.",
            "Check this gene alignment for downstream use: {request}.",
        ),
    },
}


class CorpusBuildError(ValueError):
    """Report a source, selection, or corpus invariant failure."""


@dataclass(frozen=True, slots=True)
class GeneFact:
    """Store one current gene and its split units."""

    fact_id: str
    split_units: tuple[str, ...]
    gene_symbol: str | None
    gene_id: str
    gene_version: int | None = None


@dataclass(frozen=True, slots=True)
class AmbiguousFact:
    """Store one source-backed ambiguous symbol."""

    fact_id: str
    split_units: tuple[str, ...]
    gene_symbol: str
    candidate_gene_ids: tuple[str, ...]
    candidate_details: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class ExternalFact:
    """Store one external name-to-current-gene mapping."""

    fact_id: str
    split_units: tuple[str, ...]
    query: str
    gene_symbol: str | None
    gene_id: str


Fact = GeneFact | AmbiguousFact | ExternalFact


@dataclass(frozen=True, slots=True)
class Scenario:
    """Store one selected form before runtime execution."""

    split: str
    form: int
    fact: Fact
    target: GeneFact
    framing: str
    template_index: int
    occurrence: int
    orientation: int = 0
    distractor: GeneFact | None = None
    ambiguous_target_is_candidate: bool | None = None
    source_form: int | None = None
    paired_success_id: str | None = None
    case_perturbed: bool = False
    training_variant: int | None = None

    @property
    def effective_form(self) -> int:
        return self.source_form if self.form == 9 else self.form

    @property
    def source_trace(self) -> str:
        return f"S1.4.{self.form}"

    @property
    def split_units(self) -> tuple[str, ...]:
        units = set(self.fact.split_units)
        units.update(self.target.split_units)
        if self.distractor is not None:
            units.update(self.distractor.split_units)
        return tuple(sorted(units))

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
                "target_id": self.target.fact_id,
                "framing": self.framing,
                "template_index": self.template_index,
                "occurrence": self.occurrence,
                "orientation": self.orientation,
                "distractor_id": (
                    self.distractor.fact_id if self.distractor else None
                ),
                "ambiguous_target_is_candidate": (
                    self.ambiguous_target_is_candidate
                ),
                "source_form": self.source_form,
                "paired_success_id": self.paired_success_id,
                "case_perturbed": self.case_perturbed,
                **({"training_variant": self.training_variant}
                   if self.training_variant is not None else {}),
            }
        )


def _canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _fact_id(kind: str, *values: str) -> str:
    return "fact_" + _stable_sha256({"kind": kind, "values": values})[:24]


def split_row_counts(
    train_rows: int, split_fractions: Mapping[str, Any]
) -> dict[str, int]:
    """Derive each split count from the training anchor."""

    train_fraction = _fraction(split_fractions["train"])
    counts = {"train": train_rows}
    for split in ("validation", "test"):
        value = Fraction(train_rows) * _fraction(
            split_fractions[split]
        ) / train_fraction
        if value.denominator != 1:
            raise CorpusBuildError(
                f"S1.4 {split} row count is not integral: {value}"
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
        name: value.numerator // value.denominator
        for name, value in raw.items()
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


def form_counts(
    total: int,
    settings: Mapping[str, Any],
    curriculum: Mapping[str, Any],
) -> dict[int, int]:
    """Scale the final training allocations with largest-remainder rounding."""

    expected_policy = {
        "source_form": "symbol_id_match",
        "allocation": "replace",
        "call_stage": "first",
    }
    if dict(settings["failure_injection"]) != expected_policy:
        raise CorpusBuildError("the S1.4 failure policy changed")
    allocations = settings["allocations"]
    if set(allocations) != set(FORM_BY_BUCKET):
        raise CorpusBuildError("the S1.4 allocation fields changed")
    train_rows = int(settings["train_rows"])
    if sum(int(value) for value in allocations.values()) != train_rows:
        raise CorpusBuildError("the S1.4 training allocations do not reconcile")
    counts = largest_remainder_counts(
        total,
        {
            FORM_BY_BUCKET[name]: Fraction(int(value), train_rows)
            for name, value in allocations.items()
        },
    )
    if counts[9] > counts[1]:
        raise CorpusBuildError("form 9 lacks enough retained form-1 pairs")
    if total == train_rows:
        ordinary_tool_rows = sum(
            int(allocations[name])
            for name in (
                "symbol_id_match",
                "symbol_id_mismatch",
                "historical_symbol_unresolved",
                "ambiguous_symbol",
                "mygene_id_match",
            )
        ) + int(allocations["backend_unavailable"])
        expected = Fraction(ordinary_tool_rows) * _fraction(
            curriculum["tool_failure_share"]
        )
        if expected.denominator != 1 or counts[9] != expected:
            raise CorpusBuildError("the S1.4 training failure share changed")
    return dict(sorted((int(form), count) for form, count in counts.items()))


def ambiguous_outcome_counts(
    total: int, settings: Mapping[str, Any]
) -> dict[str, int]:
    """Scale the candidate and noncandidate outcomes for form 6."""

    configured = settings["ambiguous_outcome_counts"]
    if set(configured) != {"candidate", "noncandidate"}:
        raise CorpusBuildError("the S1.4 ambiguous outcome fields changed")
    configured_total = sum(int(value) for value in configured.values())
    if configured_total != int(settings["allocations"]["ambiguous_symbol"]):
        raise CorpusBuildError("the S1.4 ambiguous outcomes do not reconcile")
    return largest_remainder_counts(
        total,
        {
            name: Fraction(int(value), configured_total)
            for name, value in configured.items()
        },
    )


def _versions_from_input(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {str(gene_id): int(version) for gene_id, version in value.items()}
    genes = getattr(value, "genes", value)
    versions: dict[str, int] = {}
    for item in genes:
        gene_id = str(getattr(item, "gene_id"))
        version = int(getattr(item, "gene_version"))
        if gene_id in versions:
            raise CorpusBuildError(f"duplicate gene version for {gene_id}")
        versions[gene_id] = version
    return versions


def _mygene_gene_ids(hit: Mapping[str, Any]) -> tuple[str, ...]:
    value = hit.get("ensembl")
    if isinstance(value, Mapping):
        gene_id = value.get("gene")
        return (str(gene_id),) if isinstance(gene_id, str) else ()
    if isinstance(value, list):
        return tuple(
            sorted(
                {
                    str(item["gene"])
                    for item in value
                    if isinstance(item, Mapping)
                    and isinstance(item.get("gene"), str)
                }
            )
        )
    return ()


def _usable_mygene_records(
    payload: Mapping[str, Any], mygene_cache: Mapping[str, Any]
) -> list[tuple[str, str, str]]:
    """Return cache aliases with one exact current human mapping."""

    genes_by_id = payload["genes_by_id"]
    records: list[tuple[str, str, str]] = []
    for cache_key, raw_hits in mygene_cache.items():
        if not str(cache_key).startswith("alias::"):
            continue
        query = str(cache_key).split("::", 1)[1]
        if not query or not isinstance(raw_hits, list) or len(raw_hits) != 1:
            continue
        hit = raw_hits[0]
        if (
            not isinstance(hit, Mapping)
            or hit.get("notfound")
            or hit.get("taxid") != 9606
        ):
            continue
        candidate_ids = _mygene_gene_ids(hit)
        if len(candidate_ids) != 1:
            continue
        gene_id = candidate_ids[0]
        if gene_id not in genes_by_id:
            continue
        raw_symbol = genes_by_id[gene_id][0]
        if raw_symbol is None or hit.get("symbol") != raw_symbol:
            continue
        records.append((query, gene_id, str(raw_symbol)))
    return sorted(
        records,
        key=lambda item: (item[0].casefold(), item[0], item[1]),
    )


def build_fact_pools(
    payload: Mapping[str, Any],
    gene_versions: Any,
    mygene_cache: Mapping[str, Any],
    *,
    s0_settings: Mapping[str, Any],
    seed: int,
    fractions: Mapping[str, Any],
    maximum_candidate_gene_ids: int = 49,
) -> dict[str, dict[str, list[Any]]]:
    """Build all split-safe source pools for the nine forms."""

    del fractions
    components, component_by_gene = build_identity_components(
        payload["genes_by_id"]
    )
    assignments = assign_evaluation_components(
        components,
        s0_settings["evaluation"],
        seed=int(s0_settings["assignment_seed"]),
    )
    versions = _versions_from_input(gene_versions)
    genes_by_id = payload["genes_by_id"]
    if set(versions) != set(genes_by_id):
        raise CorpusBuildError("the GTF version table and source index differ")

    split_by_unit = {
        component.split_unit: assignments.get(component.split_unit, "train")
        for component in components
    }
    split_by_gene = {
        gene_id: split_by_unit[split_unit]
        for gene_id, split_unit in component_by_gene.items()
    }
    units_by_gene = {
        gene_id: (split_unit,)
        for gene_id, split_unit in component_by_gene.items()
    }
    ids_by_symbol: dict[str, list[str]] = defaultdict(list)
    for gene_id, values in genes_by_id.items():
        if values[0] is not None:
            ids_by_symbol[str(values[0]).casefold()].append(str(gene_id))

    pool_names = (
        "resolved",
        "current",
        "near_miss",
        "historical",
        "ambiguous",
        "external",
    )
    pools: dict[str, dict[str, list[Any]]] = {
        split: {name: [] for name in pool_names} for split in SPLITS
    }
    current_by_id: dict[str, GeneFact] = {}
    for gene_id, values in genes_by_id.items():
        raw_symbol = values[0]
        symbol = None if raw_symbol is None else str(raw_symbol)
        fact = GeneFact(
            fact_id=_fact_id("current_gene", gene_id),
            split_units=units_by_gene[gene_id],
            gene_symbol=symbol,
            gene_id=gene_id,
            gene_version=versions[gene_id],
        )
        current_by_id[gene_id] = fact
        split = split_by_gene[gene_id]
        pools[split]["current"].append(fact)
        if symbol is not None and len(ids_by_symbol[symbol.casefold()]) == 1:
            pools[split]["resolved"].append(fact)

    by_prefix: dict[str, list[str]] = defaultdict(list)
    for gene_id in genes_by_id:
        by_prefix[gene_id[:-1]].append(gene_id)
    for gene_ids in by_prefix.values():
        ordered = sorted(gene_ids)
        for left_index, left_id in enumerate(ordered):
            for right_id in ordered[left_index + 1 :]:
                split = split_by_gene[left_id]
                if split != split_by_gene[right_id]:
                    continue
                pools[split]["near_miss"].append(
                    (current_by_id[left_id], current_by_id[right_id])
                )

    historical_by_key: dict[str, list[Sequence[Any]]] = defaultdict(list)
    for row in payload["historical_symbols"]:
        historical_by_key[str(row[0]).casefold()].append(row)
    for key, rows in historical_by_key.items():
        if key in ids_by_symbol:
            continue
        linked_ids = sorted(
            {
                str(gene_id)
                for row in rows
                for gene_id in row[2]
                if str(gene_id) in current_by_id
            }
        )
        if not linked_ids:
            continue
        linked_splits = {split_by_gene[gene_id] for gene_id in linked_ids}
        if len(linked_splits) != 1:
            continue
        split = next(iter(linked_splits))
        spelling = min(
            (str(row[0]) for row in rows),
            key=lambda item: _stable_sha256(
                {"seed": seed, "historical_key": key, "spelling": item}
            ),
        )
        target_id = min(
            linked_ids,
            key=lambda item: _stable_sha256(
                {"seed": seed, "historical_key": key, "gene_id": item}
            ),
        )
        units = tuple(
            sorted({component_by_gene[gene_id] for gene_id in linked_ids})
        )
        target_symbol = genes_by_id[target_id][0]
        pools[split]["historical"].append(
            ExternalFact(
                fact_id=_fact_id("historical_symbol", spelling, target_id),
                split_units=units,
                query=spelling,
                gene_symbol=(
                    None if target_symbol is None else str(target_symbol)
                ),
                gene_id=target_id,
            )
        )

    for key, candidate_ids_value in ids_by_symbol.items():
        candidate_ids = tuple(sorted(candidate_ids_value))
        if not 2 <= len(candidate_ids) <= maximum_candidate_gene_ids:
            continue
        units = {component_by_gene[gene_id] for gene_id in candidate_ids}
        if len(units) != 1:
            raise CorpusBuildError("one ambiguous symbol crosses split units")
        split_unit = next(iter(units))
        split = split_by_unit[split_unit]
        symbol = str(genes_by_id[candidate_ids[0]][0])
        pools[split]["ambiguous"].append(
            AmbiguousFact(
                fact_id=_fact_id("ambiguous_symbol", symbol),
                split_units=(split_unit,),
                gene_symbol=symbol,
                candidate_gene_ids=candidate_ids,
                candidate_details={
                    gene_id: {
                        "chromosome": str(genes_by_id[gene_id][1]),
                        "biotype": str(genes_by_id[gene_id][2]),
                    }
                    for gene_id in candidate_ids
                },
            )
        )

    for query, gene_id, canonical_symbol in _usable_mygene_records(
        payload, mygene_cache
    ):
        lexical_gene_ids = {
            str(item)
            for item in ids_by_symbol.get(query.casefold(), ())
        }
        lexical_gene_ids.update(
            str(item)
            for row in historical_by_key.get(query.casefold(), ())
            for item in row[2]
            if str(item) in component_by_gene
        )
        linked_gene_ids = lexical_gene_ids | {gene_id}
        linked_units = {
            component_by_gene[item] for item in linked_gene_ids
        }
        linked_splits = {split_by_unit[item] for item in linked_units}
        if len(linked_splits) != 1:
            continue
        split = next(iter(linked_splits))
        pools[split]["external"].append(
            ExternalFact(
                fact_id=_fact_id("mygene_alias", query, gene_id),
                split_units=tuple(sorted(linked_units)),
                query=query,
                gene_symbol=canonical_symbol,
                gene_id=gene_id,
            )
        )

    for split, split_pools in pools.items():
        for name, values in split_pools.items():
            values.sort(
                key=lambda item: _stable_sha256(
                    {
                        "seed": seed,
                        "split": split,
                        "pool": name,
                        "fact": (
                            [value.fact_id for value in item]
                            if isinstance(item, tuple)
                            else item.fact_id
                        ),
                    }
                )
            )
            if not values:
                raise CorpusBuildError(f"the {split} {name} source pool is empty")
    return pools


def source_pool_coverage(
    pools: Mapping[str, Mapping[str, Sequence[Any]]]
) -> dict[str, Any]:
    """Report source-pool sizes and key source distributions."""

    return {
        "pool_counts": {
            split: {name: len(values) for name, values in split_pools.items()}
            for split, split_pools in pools.items()
        },
        "ambiguous_candidate_count_distribution": {
            split: dict(
                sorted(
                    Counter(
                        str(len(fact.candidate_gene_ids))
                        for fact in split_pools["ambiguous"]
                    ).items(),
                    key=lambda item: int(item[0]),
                )
            )
            for split, split_pools in pools.items()
        },
    }


def _mixed_case(value: str) -> str:
    letters = 0
    rendered = []
    for character in value:
        if not character.isalpha():
            rendered.append(character)
            continue
        rendered.append(
            character.lower() if letters % 2 == 0 else character.upper()
        )
        letters += 1
    result = "".join(rendered)
    return result if result != value else value.lower()


def _case_can_change(scenario: Scenario) -> bool:
    """Report whether the audited source input has a different case form."""

    form = scenario.effective_form
    if form in {1, 2, 7} and isinstance(scenario.fact, GeneFact):
        value = scenario.fact.gene_symbol
        return value is not None and value.lower() != value
    if form == 5 and isinstance(scenario.fact, ExternalFact):
        return scenario.fact.query.lower() != scenario.fact.query
    if form == 6 and isinstance(scenario.fact, AmbiguousFact):
        return scenario.fact.gene_symbol.lower() != scenario.fact.gene_symbol
    if form == 8 and isinstance(scenario.fact, ExternalFact):
        return _mixed_case(scenario.fact.gene_id) != scenario.fact.gene_id
    return False


def _case_value(
    value: str,
    *,
    split: str,
    form: int,
    occurrence: int,
    seed: int,
    case_rate: Any,
) -> str:
    point = Fraction(
        int(
            _stable_sha256(
                {
                    "seed": seed,
                    "split": split,
                    "form": form,
                    "occurrence": occurrence,
                    "value": value,
                }
            )[:16],
            16,
        ),
        16**16,
    )
    if point >= _fraction(case_rate):
        return value
    return value.lower() if occurrence % 2 == 0 else _mixed_case(value)


def _ordered_pool(
    values: Sequence[Any], *, seed: int, split: str, form: int
) -> list[Any]:
    return sorted(
        values,
        key=lambda value: _stable_sha256(
            {
                "seed": seed,
                "split": split,
                "form": form,
                "fact": (
                    [item.fact_id for item in value]
                    if isinstance(value, tuple)
                    else value.fact_id
                ),
            }
        ),
    )


def _pick_distinct(
    facts: Sequence[GeneFact],
    excluded_ids: set[str],
    *,
    offset: int,
) -> GeneFact:
    for step in range(len(facts)):
        fact = facts[(offset + step) % len(facts)]
        if fact.gene_id not in excluded_ids:
            return fact
    raise CorpusBuildError("a source pool has no distinct target gene")


def _scenario_for_form(
    *,
    split: str,
    form: int,
    ordinal: int,
    occurrence: int,
    pools: Mapping[str, Sequence[Any]],
    current_by_id: Mapping[str, GeneFact],
    seed: int,
    case_rate: Any,
    ambiguous_target_is_candidate: bool | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = ALIGNMENT_PROMPTS,
) -> Scenario:
    framing = "direct" if ordinal % 2 == 0 else "step"
    templates = prompts[split][framing]
    orientation = occurrence % 2
    current = pools["current"]
    resolved = pools["resolved"]
    distractor: GeneFact | None = None

    if form in {1, 2, 7}:
        ordered = resolved
        fact = ordered[occurrence % len(ordered)]
        if form == 2:
            target = _pick_distinct(
                resolved,
                {fact.gene_id},
                offset=occurrence * 17 + 1,
            )
        else:
            target = fact
        distractor = _pick_distinct(
            current,
            {target.gene_id},
            offset=occurrence * 19 + 3,
        )
    elif form == 3:
        ordered = current
        fact = ordered[occurrence % len(ordered)]
        target = fact
        distractor = _pick_distinct(
            current,
            {fact.gene_id},
            offset=occurrence * 43 + 19,
        )
    elif form == 4:
        ordered = pools["near_miss"]
        fact, target = ordered[occurrence % len(ordered)]
        distractor = _pick_distinct(
            current,
            {fact.gene_id, target.gene_id},
            offset=occurrence * 41 + 17,
        )
    elif form == 5:
        ordered = pools["historical"]
        fact = ordered[occurrence % len(ordered)]
        target = current_by_id[fact.gene_id]
        distractor = _pick_distinct(
            current,
            {target.gene_id},
            offset=occurrence * 23 + 5,
        )
    elif form == 6:
        ordered = pools["ambiguous"]
        fact = ordered[occurrence % len(ordered)]
        if ambiguous_target_is_candidate is None:
            raise CorpusBuildError("form 6 lacks its comparison outcome")
        if ambiguous_target_is_candidate:
            target_id = fact.candidate_gene_ids[
                (occurrence // len(ordered)) % len(fact.candidate_gene_ids)
            ]
            target = current_by_id[target_id]
        else:
            target = _pick_distinct(
                current,
                set(fact.candidate_gene_ids),
                offset=occurrence * 29 + 7,
            )
        distractor = _pick_distinct(
            current,
            set(fact.candidate_gene_ids) | {target.gene_id},
            offset=occurrence * 31 + 11,
        )
    elif form == 8:
        ordered = pools["external"]
        fact = ordered[occurrence % len(ordered)]
        target = current_by_id[fact.gene_id]
        distractor = _pick_distinct(
            current,
            {target.gene_id},
            offset=occurrence * 37 + 13,
        )
    else:
        raise CorpusBuildError(f"unsupported ordinary S1.4 form {form}")

    template_index = occurrence % len(templates)
    scenario = Scenario(
        split=split,
        form=form,
        fact=fact,
        target=target,
        framing=framing,
        template_index=template_index,
        occurrence=occurrence,
        orientation=orientation,
        distractor=distractor,
        ambiguous_target_is_candidate=ambiguous_target_is_candidate,
    )
    return scenario


def select_scenarios(
    pools: Mapping[str, Mapping[str, Sequence[Any]]],
    counts_by_split: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
    case_rate: Any = Fraction(1, 10),
    framing_shares: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = ALIGNMENT_PROMPTS,
) -> dict[str, list[Scenario]]:
    """Select deterministic scenarios from synthetic or source pools."""

    if framing_shares is not None and dict(framing_shares) != {
        "direct": 0.5,
        "step": 0.5,
    }:
        raise CorpusBuildError("S1.4 supports equal direct and step framing")
    training_prompts = prompts["train"]
    if settings and settings.get("train_decorrelate"):
        # Freeze semantic selection before assigning the requested wording bank.
        prompts = {**prompts, "train": ALIGNMENT_PROMPTS["train"]}
    selected: dict[str, list[Scenario]] = {}
    for split in SPLITS:
        scenarios: list[Scenario] = []
        counts = counts_by_split[split]
        global_ordinal = 0
        by_form: dict[int, list[Scenario]] = {}
        current_by_id = {
            fact.gene_id: fact for fact in pools[split]["current"]
        }
        ordered_pools = {
            name: _ordered_pool(
                values, seed=seed, split=split, form=0
            )
            for name, values in pools[split].items()
        }
        for form in range(1, 9):
            form_scenarios: list[Scenario] = []
            if form == 6:
                if settings is None:
                    outcome_counts = largest_remainder_counts(
                        int(counts[form]),
                        {"candidate": Fraction(1, 2), "noncandidate": Fraction(1, 2)},
                    )
                else:
                    outcome_counts = ambiguous_outcome_counts(
                        int(counts[form]), settings
                    )
                outcomes = [True] * outcome_counts["candidate"] + [
                    False
                ] * outcome_counts["noncandidate"]
            else:
                outcomes = [None] * int(counts[form])
            for occurrence, outcome in enumerate(outcomes):
                scenario = _scenario_for_form(
                    split=split,
                    form=form,
                    ordinal=global_ordinal,
                    occurrence=occurrence,
                    pools=ordered_pools,
                    current_by_id=current_by_id,
                    seed=seed,
                    case_rate=case_rate,
                    ambiguous_target_is_candidate=outcome,
                    prompts=prompts,
                )
                form_scenarios.append(scenario)
                scenarios.append(scenario)
                global_ordinal += 1
            by_form[form] = form_scenarios

        success_pairs = by_form[1]
        failure_count = int(counts[9])
        if failure_count > len(success_pairs):
            raise CorpusBuildError("form 9 lacks retained form-1 success pairs")
        pair_buckets: dict[int, list[Scenario]] = defaultdict(list)
        for success in success_pairs:
            pair_buckets[success.template_index].append(success)
        for template_index, bucket in pair_buckets.items():
            bucket.sort(
                key=lambda item: _stable_sha256(
                    {
                        "seed": seed,
                        "split": split,
                        "failure_template": template_index,
                        "pair": item.scenario_id,
                    }
                )
            )
        ordered_pairs: list[Scenario] = []
        bucket_offsets = {index: 0 for index in sorted(pair_buckets)}
        while len(ordered_pairs) < failure_count:
            made_progress = False
            for template_index in sorted(pair_buckets):
                offset = bucket_offsets[template_index]
                bucket = pair_buckets[template_index]
                if offset >= len(bucket):
                    continue
                ordered_pairs.append(bucket[offset])
                bucket_offsets[template_index] += 1
                made_progress = True
                if len(ordered_pairs) == failure_count:
                    break
            if not made_progress:
                raise CorpusBuildError(
                    "form 9 cannot cover the retained success templates"
                )
        failures = [
            replace(
                success,
                form=9,
                source_form=1,
                paired_success_id=success.scenario_id,
            )
            for success in ordered_pairs[:failure_count]
        ]
        scenarios.extend(failures)

        # Apply case changes after the source rows and their failure copies exist.
        # Prompt rendering uses the scenario hash, so paired rows stay identical.
        selected[split] = sorted(
            scenarios,
            key=lambda item: (item.form, item.scenario_id),
        )

    # Rebalance framing globally without changing a failure pair.
    for split, scenarios in selected.items():
        target = largest_remainder_counts(
            len(scenarios), {"direct": Fraction(1, 2), "step": Fraction(1, 2)}
        )
        pair_groups: list[list[Scenario]] = []
        single_groups: list[list[Scenario]] = []
        paired_ids = {
            item.paired_success_id for item in scenarios if item.form == 9
        }
        by_id = {item.scenario_id: item for item in scenarios if item.form != 9}
        used: set[str] = set()
        failures_by_pair = {
            item.paired_success_id: item for item in scenarios if item.form == 9
        }
        for success_id in sorted(paired_ids):
            if success_id is None:
                continue
            success = by_id[success_id]
            pair_groups.append([success, failures_by_pair[success_id]])
            used.add(success.scenario_id)
        for item in scenarios:
            if item.form != 9 and item.scenario_id not in used:
                single_groups.append([item])
        grouped_pairs: dict[int, list[list[Scenario]]] = defaultdict(list)
        for group in pair_groups:
            grouped_pairs[group[0].template_index].append(group)
        for template_index, groups in grouped_pairs.items():
            groups.sort(
                key=lambda group: _stable_sha256(
                    {
                        "seed": seed,
                        "split": split,
                        "paired_template": template_index,
                        "group": group[0].scenario_id,
                    }
                )
            )
        pair_groups = []
        pair_offsets = {index: 0 for index in sorted(grouped_pairs)}
        while len(pair_groups) < len(paired_ids):
            made_progress = False
            for template_index in sorted(grouped_pairs):
                offset = pair_offsets[template_index]
                groups = grouped_pairs[template_index]
                if offset >= len(groups):
                    continue
                pair_groups.append(groups[offset])
                pair_offsets[template_index] += 1
                made_progress = True
            if not made_progress:
                raise CorpusBuildError("failure-pair framing groups are incomplete")
        direct_pair_count = largest_remainder_counts(
            len(pair_groups),
            {"direct": Fraction(1, 2), "step": Fraction(1, 2)},
        )["direct"]
        direct_left = target["direct"] - 2 * direct_pair_count
        if not 0 <= direct_left <= len(single_groups):
            raise CorpusBuildError(f"the {split} framing groups cannot balance")
        singles_by_form: dict[int, list[Scenario]] = defaultdict(list)
        for group in single_groups:
            singles_by_form[group[0].form].append(group[0])
        direct_quota_by_form = {
            form: len(items) // 2
            for form, items in singles_by_form.items()
        }
        quota_left = direct_left - sum(direct_quota_by_form.values())
        odd_forms = [
            form
            for form in sorted(singles_by_form)
            if len(singles_by_form[form]) % 2
        ]
        if not 0 <= quota_left <= len(odd_forms):
            raise CorpusBuildError(
                f"the {split} per-form framing quotas cannot balance"
            )
        for form in odd_forms[:quota_left]:
            direct_quota_by_form[form] += 1

        direct_single_ids: set[str] = set()
        ordered_single_groups: list[list[Scenario]] = []
        for form in sorted(singles_by_form):
            strata: dict[tuple[int, int, int], list[Scenario]] = defaultdict(list)
            for item in singles_by_form[form]:
                outcome = (
                    int(bool(item.ambiguous_target_is_candidate))
                    if form == 6
                    else 0
                )
                strata[(item.template_index, item.orientation, outcome)].append(
                    item
                )
            for stratum, items in strata.items():
                items.sort(
                    key=lambda item: _stable_sha256(
                        {
                            "seed": seed,
                            "split": split,
                            "form": form,
                            "framing_stratum": stratum,
                            "scenario": item.scenario_id,
                        }
                    )
                )
            offsets = {stratum: 0 for stratum in sorted(strata)}
            ordered: list[Scenario] = []
            while len(ordered) < len(singles_by_form[form]):
                made_progress = False
                for stratum in sorted(strata):
                    offset = offsets[stratum]
                    items = strata[stratum]
                    if offset >= len(items):
                        continue
                    ordered.append(items[offset])
                    offsets[stratum] += 1
                    made_progress = True
                if not made_progress:
                    raise CorpusBuildError(
                        f"the {split} form {form} framing strata are incomplete"
                    )
            direct_single_ids.update(
                item.scenario_id
                for item in ordered[: direct_quota_by_form[form]]
            )
            ordered_single_groups.extend([item] for item in ordered)
        if len(direct_single_ids) != direct_left:
            raise CorpusBuildError(f"the {split} direct framing quota changed")
        rebuilt: list[Scenario] = []
        for index, group in enumerate(pair_groups):
            framing = "direct" if index < direct_pair_count else "step"
            template_count = len(prompts[split][framing])
            success, failure = group
            rebuilt_success = replace(
                success,
                framing=framing,
                template_index=success.template_index % template_count,
            )
            rebuilt_failure = replace(
                failure,
                framing=framing,
                template_index=rebuilt_success.template_index,
                paired_success_id=rebuilt_success.scenario_id,
            )
            rebuilt.extend((rebuilt_success, rebuilt_failure))
        for group in ordered_single_groups:
            item = group[0]
            framing = (
                "direct" if item.scenario_id in direct_single_ids else "step"
            )
            template_count = len(prompts[split][framing])
            rebuilt.append(
                replace(
                    item,
                    framing=framing,
                    template_index=item.template_index % template_count,
                )
            )
        eligible_forms = {1, 2, 5, 6, 7, 8, 9}
        eligible_rows = [
            item for item in rebuilt if item.form in eligible_forms
        ]
        case_target = largest_remainder_counts(
            len(eligible_rows),
            {
                "perturbed": _fraction(case_rate),
                "canonical": 1 - _fraction(case_rate),
            },
        )["perturbed"]
        rebuilt_by_id = {
            item.scenario_id: item for item in rebuilt if item.form != 9
        }
        rebuilt_failures = {
            item.paired_success_id: item for item in rebuilt if item.form == 9
        }
        case_groups: list[list[Scenario]] = []
        grouped_ids: set[str] = set()
        for success_id, failure in rebuilt_failures.items():
            if success_id is None:
                raise CorpusBuildError("a failure lacks its paired success")
            success = rebuilt_by_id[success_id]
            grouped_ids.update((success.scenario_id, failure.scenario_id))
            if _case_can_change(success):
                case_groups.append([success, failure])
        case_groups.extend(
            [item]
            for item in rebuilt
            if (
                item.form in eligible_forms
                and item.scenario_id not in grouped_ids
                and _case_can_change(item)
            )
        )
        case_groups.sort(
            key=lambda group: _stable_sha256(
                {
                    "seed": seed,
                    "split": split,
                    "case_group": [item.scenario_id for item in group],
                }
            )
        )
        perturbed_ids: set[str] = set()
        case_left = case_target
        for group in case_groups:
            if len(group) <= case_left:
                perturbed_ids.update(item.scenario_id for item in group)
                case_left -= len(group)
            if case_left == 0:
                break
        if case_left:
            raise CorpusBuildError(
                f"the {split} case-perturbation quota cannot reconcile"
            )

        case_rebuilt: list[Scenario] = []
        failures_for_case = {
            item.paired_success_id: item for item in rebuilt if item.form == 9
        }
        handled_failures: set[str] = set()
        for item in rebuilt:
            if item.form == 9:
                continue
            paired_failure = failures_for_case.get(item.scenario_id)
            changed = replace(
                item, case_perturbed=item.scenario_id in perturbed_ids
            )
            case_rebuilt.append(changed)
            if paired_failure is not None:
                changed_failure = replace(
                    paired_failure,
                    case_perturbed=(
                        paired_failure.scenario_id in perturbed_ids
                    ),
                    paired_success_id=changed.scenario_id,
                )
                case_rebuilt.append(changed_failure)
                handled_failures.add(paired_failure.scenario_id)
        if len(handled_failures) != len(rebuilt_failures):
            raise CorpusBuildError(
                f"the {split} case rebuild lost a failure pair"
            )
        selected[split] = sorted(
            case_rebuilt,
            key=lambda item: _stable_sha256(
                {
                    "seed": seed,
                    "split": split,
                    "output_scenario": item.scenario_id,
                }
            ),
        )
    if settings and settings.get("train_decorrelate"):
        selected["train"] = _decorrelate_training(
            selected["train"], seed=seed, prompts=training_prompts,
        )
    return selected


def _decorrelate_training(
    scenarios: Sequence[Scenario], *, seed: int,
    prompts: Mapping[str, Sequence[str]],
) -> list[Scenario]:
    """Select independent request variants, then distribute their wrappers."""

    failures = {
        item.paired_success_id: item for item in scenarios if item.form == 9
    }
    groups: dict[tuple[Any, ...], list[Scenario]] = defaultdict(list)
    for item in scenarios:
        if item.form != 9:
            groups[(item.form, item.framing,
                    item.ambiguous_target_is_candidate,
                    item.scenario_id in failures)].append(item)
    result = []
    for key, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        # Identical phases for the two ambiguity outcomes keep their visible
        # structural frequencies balanced. Facts and failure pairs stay intact.
        phase = int(_stable_sha256({
            "seed": seed, "form": key[0], "framing": key[1],
            "purpose": "training_variants",
        })[:8], 16)
        period = 16 if key[0] in {3, 4} else 4
        for rank, item in enumerate(sorted(items, key=lambda value: value.scenario_id)):
            variant = (rank + phase) % period
            wrapper = (rank // period + variant * 7) % len(prompts[item.framing])
            changed = replace(item, orientation=variant % 2,
                              training_variant=variant, template_index=wrapper)
            result.append(changed)
            if item.scenario_id in failures:
                result.append(replace(
                    failures[item.scenario_id], orientation=changed.orientation,
                    training_variant=variant, template_index=wrapper,
                    paired_success_id=changed.scenario_id,
                ))
    return sorted(result, key=lambda item: _stable_sha256({
        "seed": seed, "split": "train", "output_scenario": item.scenario_id,
    }))


def _row_id(scenario: Scenario, schema_version: str) -> str:
    return "wm3_s1_4_" + _stable_sha256(
        {"scenario_id": scenario.scenario_id, "schema_version": schema_version}
    )[:24]


def _context_group_id(scenario: Scenario) -> str:
    return "group_" + _stable_sha256(
        {
            "split": scenario.split,
            "effective_form": scenario.effective_form,
            "fact": scenario.fact.fact_id,
            "target": scenario.target.fact_id,
            "distractor": (
                scenario.distractor.fact_id if scenario.distractor else None
            ),
            "occurrence": scenario.occurrence,
            "orientation": scenario.orientation,
        }
    )[:8]


def _case_input(
    value: str, scenario: Scenario, *, side: int, contract: Mapping[str, Any]
) -> str:
    if scenario.effective_form == 3 or not scenario.case_perturbed:
        return value
    del contract
    return value.lower() if side == 0 else _mixed_case(value)


def _mentions(
    scenario: Scenario, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    form = scenario.effective_form
    fact = scenario.fact
    target = scenario.target
    if form in {1, 2, 7}:
        if not isinstance(fact, GeneFact) or fact.gene_symbol is None:
            raise CorpusBuildError(f"form {form} lacks a resolved symbol")
        source_item = "abstract" if form == 2 else "typed_seed_list"
        target_item = "seed_gene_set" if form == 2 else "module"
        source = {
            "item": source_item,
            "input": _case_input(fact.gene_symbol, scenario, side=0, contract=contract),
            "gene_id": fact.gene_id,
            "gene_symbol": fact.gene_symbol,
            "source": True,
        }
        second = {
            "item": target_item,
            "input": target.gene_id,
            "gene_id": target.gene_id,
            "gene_symbol": target.gene_symbol,
            "source": False,
        }
    elif form == 3:
        if not isinstance(fact, GeneFact) or fact.gene_version is None:
            raise CorpusBuildError("form 3 lacks a versioned current ID")
        source = {
            "item": "seed_gene_set",
            "input": f"{fact.gene_id}.{fact.gene_version}",
            "gene_id": fact.gene_id,
            "gene_symbol": fact.gene_symbol,
            "source": True,
        }
        second = {
            "item": "module",
            "input": fact.gene_id.lower(),
            "gene_id": fact.gene_id,
            "gene_symbol": fact.gene_symbol,
            "source": False,
        }
    elif form == 4:
        if not isinstance(fact, GeneFact):
            raise CorpusBuildError("form 4 lacks its first current ID")
        source = {
            "item": "module",
            "input": fact.gene_id,
            "gene_id": fact.gene_id,
            "gene_symbol": fact.gene_symbol,
            "source": True,
        }
        second = {
            "item": "seed_gene_set",
            "input": target.gene_id,
            "gene_id": target.gene_id,
            "gene_symbol": target.gene_symbol,
            "source": False,
        }
    elif form == 5:
        if not isinstance(fact, ExternalFact):
            raise CorpusBuildError("form 5 lacks a historical symbol")
        source = {
            "item": "abstract",
            "input": _case_input(fact.query, scenario, side=0, contract=contract),
            "gene_id": None,
            "gene_symbol": None,
            "source": True,
        }
        second = {
            "item": "module",
            "input": target.gene_id,
            "gene_id": target.gene_id,
            "gene_symbol": target.gene_symbol,
            "source": False,
        }
    elif form == 6:
        if not isinstance(fact, AmbiguousFact):
            raise CorpusBuildError("form 6 lacks an ambiguous symbol")
        source = {
            "item": "typed_seed_list",
            "input": _case_input(fact.gene_symbol, scenario, side=0, contract=contract),
            "gene_id": None,
            "gene_symbol": fact.gene_symbol,
            "candidate_gene_ids": list(fact.candidate_gene_ids),
            "candidate_details": {
                gene_id: dict(details)
                for gene_id, details in fact.candidate_details.items()
            },
            "source": True,
        }
        second = {
            "item": "module",
            "input": target.gene_id,
            "gene_id": target.gene_id,
            "gene_symbol": target.gene_symbol,
            "source": False,
        }
    elif form == 8:
        if not isinstance(fact, ExternalFact):
            raise CorpusBuildError("form 8 lacks a MyGene mapping")
        source = {
            "item": "query_mygene",
            "input": fact.query,
            "gene_id": fact.gene_id,
            "gene_symbol": fact.gene_symbol,
            "source": True,
            "lookup_input": fact.gene_id,
        }
        second = {
            "item": "module",
            "input": _case_input(
                fact.gene_id, scenario, side=1, contract=contract
            ),
            "gene_id": target.gene_id,
            "gene_symbol": target.gene_symbol,
            "source": False,
        }
    else:
        raise CorpusBuildError(f"unsupported S1.4 form {form}")
    if scenario.training_variant is not None and form in {3, 4}:
        if not isinstance(fact, GeneFact):
            raise CorpusBuildError("ID notation requires a current registry fact")
        source["input"], second["input"] = fact.gene_id, target.gene_id
        seed_mention = next(item for item in (source, second) if item["item"] == "seed_gene_set")
        module_mention = next(item for item in (source, second) if item["item"] == "module")
        versions = {fact.gene_id: fact.gene_version, target.gene_id: target.gene_version}
        notation = scenario.training_variant // 4
        if notation == 1:
            version = versions[seed_mention["gene_id"]]
            if version is None:
                raise CorpusBuildError("seed ID lacks its real version")
            seed_mention["input"] = f"{seed_mention['gene_id']}.{version}"
            module_mention["input"] = module_mention["gene_id"].lower()
        elif notation == 2:
            seed_mention["input"] = seed_mention["gene_id"].lower()
        elif notation == 3:
            version = versions[module_mention["gene_id"]]
            if version is None:
                raise CorpusBuildError("module ID lacks its real version")
            module_mention["input"] = f"{module_mention['gene_id'].lower()}.{version}"
        normalized = tuple(
            str(item["input"]).split(".", 1)[0].upper()
            for item in (source, second)
        )
        if normalized != (fact.gene_id, target.gene_id) or (
            normalized[0] == normalized[1]
        ) != (form == 3):
            raise CorpusBuildError("ID notation changed the source identity or equality")
    first, last = (source, second)
    if scenario.orientation % 2:
        first, last = last, first
    return first, last


def _source_mention(
    mentions: tuple[Mapping[str, Any], Mapping[str, Any]]
) -> Mapping[str, Any]:
    for mention in mentions:
        if mention["source"]:
            return mention
    raise CorpusBuildError("a comparison has no source mention")


def _public_item(mention: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "item": mention["item"],
        "input": mention["input"],
        "gene_id": mention["gene_id"],
    }
    if "candidate_gene_ids" in mention:
        result["candidate_gene_ids"] = list(mention["candidate_gene_ids"])
        result["candidate_details"] = {
            gene_id: dict(details)
            for gene_id, details in mention["candidate_details"].items()
        }
    return result


def _mention_text(mention: Mapping[str, Any], *, group_id: str) -> str:
    value = str(mention["input"])
    item = mention["item"]
    if item == "typed_seed_list":
        return f"the typed seed {value}"
    if item == "seed_gene_set":
        return f"{value} in the seed gene set"
    if item == "module":
        return f"{value} in module {group_id}"
    if item == "abstract":
        return f"{value} in the example abstract"
    if item == "query_mygene":
        return f"the query_mygene entry for {value}"
    raise CorpusBuildError(f"unknown S1.4 item {item!r}")


def _request(scenario: Scenario, contract: Mapping[str, Any]) -> str:
    first, second = _mentions(scenario, contract)
    group_id = _context_group_id(scenario)
    return (
        f"whether {_mention_text(first, group_id=group_id)} and "
        f"{_mention_text(second, group_id=group_id)} name the same gene"
    )


def _prompt(
    scenario: Scenario,
    contract: Mapping[str, Any],
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = ALIGNMENT_PROMPTS,
) -> str:
    request = _request(scenario, contract).rstrip(".!?")
    templates = prompts[scenario.split][scenario.framing]
    if not 0 <= scenario.template_index < len(templates):
        raise CorpusBuildError("an alignment template index is out of range")
    return templates[scenario.template_index].format(request=request)


def _module_line(
    scenario: Scenario,
    mentions: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> str:
    module = next(
        (mention for mention in mentions if mention["item"] == "module"), None
    )
    if module is None:
        raise CorpusBuildError("a module harness has no module mention")
    gene_ids = [str(module["input"])]
    if scenario.distractor is not None:
        gene_ids.append(scenario.distractor.gene_id)
    if scenario.training_variant is not None and scenario.training_variant // 2 % 2:
        gene_ids.reverse()
    return (
        f"Module {_context_group_id(scenario)}, gene_ids: "
        + _canonical_json(gene_ids, sort_keys=False)
    )


def _seed_set_block(
    scenario: Scenario,
    mentions: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> str:
    seed = next(
        (mention for mention in mentions if mention["item"] == "seed_gene_set"),
        None,
    )
    if seed is None:
        raise CorpusBuildError("a seed-set harness has no seed mention")
    gene_ids = [str(seed["input"])]
    symbols = [seed.get("gene_symbol")]
    if scenario.distractor is not None:
        gene_ids.append(scenario.distractor.gene_id)
        symbols.append(scenario.distractor.gene_symbol)
    if scenario.training_variant is not None and scenario.training_variant // 2 % 2:
        gene_ids.reverse()
        symbols.reverse()
    payload = {
        "seed_gene_ids": gene_ids,
        "seed_gene_symbols": symbols,
    }
    return "Seed gene set:\n" + _canonical_json(payload, sort_keys=False)


def _registry_block(call_result: ToolCallResult) -> str:
    output = call_result.output
    if output.get("status") != "resolved":
        raise CorpusBuildError("an S1.4 registry block is not resolved")
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


def _harness_content(
    scenario: Scenario,
    contract: Mapping[str, Any],
    *,
    registry_result: ToolCallResult | None = None,
) -> str | None:
    form = scenario.effective_form
    mentions = _mentions(scenario, contract)
    source = _source_mention(mentions)
    if form == 1:
        return _module_line(scenario, mentions)
    if form == 2:
        return (
            _seed_set_block(scenario, mentions)
            + "\nAbstract (example text, not a real abstract):\n"
            + _canonical_json(
                f"This example text mentions {source['input']}.",
                sort_keys=False,
            )
        )
    if form in {3, 4}:
        return _module_line(scenario, mentions)
    if form == 5:
        return (
            _module_line(scenario, mentions)
            + "\nAbstract (example text, not a real abstract):\n"
            + _canonical_json(
                f"This example text mentions {source['input']}.",
                sort_keys=False,
            )
        )
    if form == 6:
        return _module_line(scenario, mentions)
    if form == 7:
        if registry_result is None:
            raise CorpusBuildError("form 7 lacks its registry block")
        return _registry_block(registry_result) + "\n" + _module_line(
            scenario, mentions
        )
    if form == 8:
        if not isinstance(scenario.fact, ExternalFact):
            raise CorpusBuildError("form 8 lacks an external fact")
        query_result = {
            "query": source["input"],
            "requested_fields": ["symbol", "ensembl"],
            "results": [
                {
                    "symbol": scenario.fact.gene_symbol,
                    "ensembl": {"gene": source["lookup_input"]},
                }
            ],
            "result_count": 1,
        }
        return (
            "query_mygene result:\n"
            + _canonical_json(query_result, sort_keys=False)
            + "\n"
            + _module_line(scenario, mentions)
        )
    raise CorpusBuildError(f"unsupported S1.4 harness form {form}")


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


def _anchor(gene_id: str, symbol: str | None = None) -> str:
    return (
        f"<GENE:{gene_id}|{symbol}>" if symbol is not None else f"<GENE:{gene_id}>"
    )


def _decision(scenario: Scenario, *, usable: bool) -> str:
    if not usable:
        return "defer"
    return "continue" if scenario.framing == "step" else "stop"


def _comparison_result(
    mentions: tuple[Mapping[str, Any], Mapping[str, Any]],
    same_gene: bool | None,
) -> dict[str, Any]:
    return {
        "first": _public_item(mentions[0]),
        "second": _public_item(mentions[1]),
        "same_gene": same_gene,
    }


def _candidate_summary(fact: AmbiguousFact) -> str:
    count = len(fact.candidate_gene_ids)
    if count > 5:
        return (
            f"{fact.gene_symbol} matches {count} genes in Ensembl release 116"
        )
    parts = []
    for gene_id in fact.candidate_gene_ids:
        details = fact.candidate_details[gene_id]
        parts.append(
            f"Candidate {_anchor(gene_id, fact.gene_symbol)} "
            f"(chromosome {details['chromosome']}, "
            f"{str(details['biotype']).replace('_', ' ')})"
        )
    return (
        f"{fact.gene_symbol} matches {count} genes in Ensembl release 116. "
        + ". ".join(parts)
    )


def _pre_call_thinking(
    scenario: Scenario,
    mentions: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> str:
    form = scenario.effective_form
    source = _source_mention(mentions)
    target = next(item for item in mentions if not item["source"])
    if form == 8:
        return (
            f"The query_mygene result gives {source['lookup_input']} for "
            f"{source['input']}. I will check this ID in the release 116 "
            "registry before I compare."
        )
    if form == 6:
        return (
            f"{source['input']} is a symbol, and {target['input']} is an "
            f"Ensembl ID. I will look up {source['input']} in the registry."
        )
    return (
        f"{source['input']} is a symbol, and {target['input']} is an "
        f"Ensembl ID. I cannot compare them as strings, so I will look up "
        f"{source['input']} in the registry."
    )


def _no_call_answer(
    scenario: Scenario,
    mentions: tuple[Mapping[str, Any], Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    form = scenario.effective_form
    first, second = mentions
    decision = _decision(scenario, usable=True)
    if form == 3:
        thinking = (
            "Both inputs are Ensembl gene IDs. They are equal after I remove "
            "the version suffix and ignore case, so no lookup is needed."
        )
        gene_id = scenario.target.gene_id
        summary = (
            f"{first['input']} and {second['input']} are the same gene ID, "
            f"{_anchor(gene_id)}. Only the version suffix and case differ."
        )
        same_gene = True
    elif form == 4:
        thinking = (
            "Both inputs are well-formed Ensembl gene IDs. They differ in "
            "the last digit, so they are different keys and need no lookup."
        )
        summary = (
            f"{_anchor(str(first['gene_id']))} and "
            f"{_anchor(str(second['gene_id']))} are different gene IDs, so "
            "they are not the same gene."
        )
        same_gene = False
    elif form == 7:
        source = _source_mention(mentions)
        thinking = (
            f"The registry block maps {source['input']} to "
            f"{source['gene_id']}. The lookup is already done, and the two "
            "mentions have the same ID."
        )
        summary = (
            f"The two mentions are the same gene, "
            f"{_anchor(str(source['gene_id']), str(source['gene_symbol']))}."
        )
        same_gene = True
    else:
        raise CorpusBuildError(f"form {form} is not a no-call form")
    if scenario.training_variant is not None and form in {3, 4}:
        relation = "equal" if same_gene else "different"
        thinking = (
            "Both inputs are Ensembl gene IDs. Remove any version suffix and "
            f"ignore case before comparing the complete IDs: {first['gene_id']} "
            f"and {second['gene_id']} are {relation}, so no lookup is needed."
        )
        if same_gene:
            summary = (
                f"{first['input']} and {second['input']} identify the same gene, "
                f"{_anchor(str(first['gene_id']))}."
            )
    return thinking, _answer(
        status="resolved",
        provenance=provenance,
        result=_comparison_result(mentions, same_gene),
        decision=decision,
        blocked_reason=None,
        summary=summary,
    )


def _tool_answer(
    scenario: Scenario,
    mentions: tuple[Mapping[str, Any], Mapping[str, Any]],
    call_result: ToolCallResult,
    *,
    name_candidate: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Render the answer after one lookup.

    ``name_candidate`` (S1 v5, training rows) makes the S1.4.6 thinking name
    the other ID, say whether it is a candidate, and say that the module is
    not a multiplex to check.
    """

    form = scenario.effective_form
    output = call_result.output
    source = _source_mention(mentions)
    target = next(item for item in mentions if not item["source"])
    if scenario.form == 9:
        if output.get("status") != "error":
            raise CorpusBuildError("form 9 did not receive an error result")
        tool_name = SYMBOL_TOOL
        input_value = str(source["input"])
        return (
            "The lookup failed, so I cannot make the comparison.",
            _answer(
                status="error",
                provenance=None,
                result={
                    "failed_tool": tool_name,
                    "error_code": output["error_code"],
                    "gene_symbol": input_value,
                },
                decision="defer",
                blocked_reason=(
                    f"The call to {tool_name} failed with "
                    f"{output['error_code']}, so there is no registry result "
                    f"for {input_value}."
                ),
                summary="The registry lookup failed. Retry the lookup to continue.",
            ),
        )
    provenance = output.get("provenance")
    if form == 1:
        if (
            output.get("status") != "resolved"
            or output.get("gene_id") != target["gene_id"]
            or output.get("gene_id") != scenario.fact.gene_id
            or output.get("gene_symbol") != scenario.fact.gene_symbol
        ):
            raise CorpusBuildError("form 1 source data did not resolve as a match")
        thinking = (
            f"{output['gene_symbol']} maps to {output['gene_id']}, the same "
            "ID as the other item. The two mentions name the same gene."
        )
        answer = _answer(
            status="resolved",
            provenance=provenance,
            result=_comparison_result(mentions, True),
            decision=_decision(scenario, usable=True),
            blocked_reason=None,
            summary=(
                f"The two mentions are the same gene, "
                f"{_anchor(str(output['gene_id']), str(output['gene_symbol']))}."
            ),
        )
        return thinking, answer
    if form == 2:
        if (
            output.get("status") != "resolved"
            or output.get("gene_id") != scenario.fact.gene_id
            or output.get("gene_symbol") != scenario.fact.gene_symbol
            or output.get("gene_id") == target["gene_id"]
        ):
            raise CorpusBuildError("form 2 source data did not resolve as a mismatch")
        thinking = (
            f"{output['gene_symbol']} maps to {output['gene_id']}, not to "
            f"{target['gene_id']}. Different Ensembl IDs name different genes."
        )
        answer = _answer(
            status="resolved",
            provenance=provenance,
            result=_comparison_result(mentions, False),
            decision=_decision(scenario, usable=True),
            blocked_reason=None,
            summary=(
                f"{source['input']} is "
                f"{_anchor(str(output['gene_id']), str(output['gene_symbol']))}. "
                f"{_anchor(str(target['gene_id']), target.get('gene_symbol'))} "
                "is a different gene."
            ),
        )
        return thinking, answer
    if form == 5:
        if (
            output.get("status") != "not_found"
            or output.get("gene_symbol") != source["input"]
            or list(output.get("candidate_gene_ids", ()))
        ):
            raise CorpusBuildError("form 5 historical symbol unexpectedly resolved")
        input_value = str(source["input"])
        thinking = (
            f"The registry has no match for {input_value}. I will not map it "
            "from memory, so I cannot make the comparison."
        )
        answer = _answer(
            status="not_found",
            provenance=provenance,
            result=_comparison_result(mentions, None),
            decision="defer",
            blocked_reason=(
                f"The registry has no gene with the symbol {input_value} in "
                "Ensembl release 116."
            ),
            summary=(
                f"{input_value} has no match in Ensembl release 116, so this "
                f"answer does not compare it with {_anchor(str(target['gene_id']))}. "
                f"Give the gene symbol or Ensembl ID for {input_value} to continue."
            ),
        )
        return thinking, answer
    if form == 6:
        if output.get("status") != "ambiguous" or not isinstance(
            scenario.fact, AmbiguousFact
        ):
            raise CorpusBuildError("form 6 source symbol is not ambiguous")
        expected_details = {
            gene_id: dict(details)
            for gene_id, details in scenario.fact.candidate_details.items()
        }
        if (
            output.get("gene_symbol") != scenario.fact.gene_symbol
            or
            list(output["candidate_gene_ids"])
            != list(scenario.fact.candidate_gene_ids)
            or output["candidate_details"] != expected_details
        ):
            raise CorpusBuildError("form 6 runtime candidates changed")
        candidate = bool(scenario.ambiguous_target_is_candidate)
        target_is_candidate = target["gene_id"] in output["candidate_gene_ids"]
        if candidate != target_is_candidate:
            raise CorpusBuildError("form 6 selected the wrong target outcome")
        candidate_text = _candidate_summary(scenario.fact)
        symbol = output["gene_symbol"]
        count = len(output["candidate_gene_ids"])
        target_id = target["gene_id"]
        module_cue = "The module only lists that ID; no multiplex is named."
        if candidate:
            thinking = (
                f"The registry maps {symbol} to {count} IDs, and {target_id} "
                f"is one of them. {module_cue} {symbol} could still name "
                "another candidate, so the two mentions are undecided. I will "
                "ask for one ID."
                if name_candidate
                else f"The registry maps {symbol} to {count} IDs. The other ID "
                "is one candidate, but the symbol does not choose it."
            )
            answer = _answer(
                status="ambiguous",
                provenance=provenance,
                result=_comparison_result(mentions, None),
                decision="defer",
                blocked_reason=(
                    f"The symbol {source['input']} matches "
                    f"{len(output['candidate_gene_ids'])} genes in Ensembl "
                    "release 116, and nothing in the task chooses one."
                ),
                summary=(
                    f"{candidate_text}. The other ID is one candidate, so this "
                    "answer does not say whether the mentions are the same gene. "
                    f"Give one Ensembl gene ID for {output['gene_symbol']} to continue."
                ),
            )
        else:
            thinking = (
                f"The registry maps {symbol} to {count} IDs, and {target_id} "
                f"is not one of them. {module_cue} Whichever candidate "
                f"{symbol} names, it is not {target_id}, so the two mentions "
                "name different genes."
                if name_candidate
                else f"The registry maps {symbol} to {count} IDs. The other ID "
                "is not one of them, so the mentions cannot name the same gene."
            )
            answer = _answer(
                status="resolved",
                provenance=provenance,
                result=_comparison_result(mentions, False),
                decision=_decision(scenario, usable=True),
                blocked_reason=None,
                summary=(
                    f"{candidate_text}. {_anchor(str(target['gene_id']))} is "
                    "not a candidate, so the two mentions are not the same gene."
                ),
            )
        return thinking, answer
    if form == 8:
        if (
            output.get("status") != "resolved"
            or output.get("gene_id") != scenario.fact.gene_id
            or output.get("gene_symbol") != scenario.fact.gene_symbol
            or output.get("gene_id") != target["gene_id"]
        ):
            raise CorpusBuildError("form 8 MyGene ID did not resolve as a match")
        thinking = (
            f"The ID is in release 116 with symbol {output['gene_symbol']}, "
            "and it is the same ID as the module item."
        )
        answer = _answer(
            status="resolved",
            provenance=provenance,
            result=_comparison_result(mentions, True),
            decision=_decision(scenario, usable=True),
            blocked_reason=None,
            summary=(
                f"The query_mygene result maps {source['input']} to "
                f"{_anchor(str(output['gene_id']), str(output['gene_symbol']))}, "
                "which is the same gene as the module item."
            ),
        )
        return thinking, answer
    raise CorpusBuildError(f"unsupported tool form {form}")


def build_row(
    scenario: Scenario,
    *,
    runtime: WorldModelToolRuntime,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]],
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = ALIGNMENT_PROMPTS,
    train_candidate_reasoning: bool = False,
    match_reasoning: Mapping[str, str] | None = None,
    mismatch_reasoning: Mapping[str, str] | None = None,
    unresolved_symbol_reasoning: Mapping[str, str] | None = None,
    ambiguous_candidate_reasoning: Mapping[str, str] | None = None,
    ambiguous_noncandidate_reasoning: Mapping[str, str] | None = None,
    mygene_match_reasoning: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Execute and render one source-backed S1.4 scenario.

    ``train_candidate_reasoning`` (S1 v5) rewrites the S1.4.6 thinking of
    training rows; validation and test keep the inline text.
    ``match_reasoning`` (S1 v6) replaces only S1.4.1 final training thinking,
    binding the successful lookup and the user's direct or workflow framing.
    ``mismatch_reasoning`` does the same for S1.4.2 and also binds the
    supplied seed ID, so the negative comparison completes without another call.
    ``unresolved_symbol_reasoning`` binds the not_found input and supplied
    module ID for S1.4.5, preserving unknown identity and deferral.
    The two ``ambiguous_*_reasoning`` templates bind S1.4.6's candidates and
    comparison ID, distinguishing an unknown match from a proven mismatch.
    ``mygene_match_reasoning`` binds the original MyGene query and returned
    registry identity for S1.4.8 without changing the comparison's inputs.
    """

    schema_version = str(contract["runtime"]["row_schema_version"])
    row_id = _row_id(scenario, schema_version)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": contract["runtime"]["system_prompt"]}
    ]
    evidence: list[list[str]] = []
    mentions = _mentions(scenario, contract)
    form = scenario.effective_form

    registry_result: ToolCallResult | None = None
    if form == 7:
        source = _source_mention(mentions)
        registry_result = runtime.execute(
            SYMBOL_TOOL, {"gene_symbol": source["input"]}
        )
        if (
            registry_result.output.get("status") != "resolved"
            or registry_result.output.get("gene_id") != source["gene_id"]
            or registry_result.output.get("gene_symbol")
            != source["gene_symbol"]
        ):
            raise CorpusBuildError("form 7 registry mapping changed")
        record = _evidence_row(
            registry_result,
            row_id=row_id,
            producer="harness_registry_block",
        )
        if record is None:
            raise CorpusBuildError("a form-7 registry block has no evidence")
        evidence.append(record)

    harness = _harness_content(
        scenario, contract, registry_result=registry_result
    )
    if harness is not None:
        messages.append({"role": "harness", "content": harness})
    messages.append(
        {"role": "user", "content": _prompt(scenario, contract, prompts=prompts)}
    )

    if form in {3, 4, 7}:
        provenance = (
            None if registry_result is None else registry_result.output["provenance"]
        )
        final_thinking, final_answer = _no_call_answer(
            scenario, mentions, provenance=provenance
        )
    else:
        source = _source_mention(mentions)
        if form == 8:
            tool_name = GENE_ID_TOOL
            arguments = {"gene_id": source["lookup_input"]}
        else:
            tool_name = SYMBOL_TOOL
            arguments = {"gene_symbol": source["input"]}
        call_result = runtime.execute(
            tool_name,
            arguments,
            inject_backend_failure=scenario.form == 9,
        )
        record = _evidence_row(call_result, row_id=row_id)
        if scenario.form != 9:
            if record is None:
                raise CorpusBuildError("a successful lookup has no evidence")
            evidence.append(record)
        elif record is not None:
            raise CorpusBuildError("a failed lookup produced evidence")
        _append_call(
            messages,
            thinking=_pre_call_thinking(scenario, mentions),
            tool_name=tool_name,
            arguments=arguments,
            call_result=call_result,
        )
        final_thinking, final_answer = _tool_answer(
            scenario,
            mentions,
            call_result,
            name_candidate=(
                train_candidate_reasoning and scenario.split == "train"
            ),
        )
        if scenario.split == "train" and scenario.form == 1 and match_reasoning:
            final_thinking = match_reasoning[scenario.framing].format(
                symbol=call_result.output["gene_symbol"],
                gene_id=call_result.output["gene_id"],
            )
        if scenario.split == "train" and scenario.form == 2 and mismatch_reasoning:
            final_thinking = mismatch_reasoning[scenario.framing].format(
                symbol=call_result.output["gene_symbol"],
                gene_id=call_result.output["gene_id"],
                target_id=scenario.target.gene_id,
            )
        if scenario.split == "train" and scenario.form == 5 and unresolved_symbol_reasoning:
            final_thinking = unresolved_symbol_reasoning[scenario.framing].format(
                input=call_result.output["gene_symbol"],
                target_id=scenario.target.gene_id,
            )
        if scenario.split == "train" and scenario.form == 6:
            reasoning = (
                ambiguous_candidate_reasoning if scenario.ambiguous_target_is_candidate
                else ambiguous_noncandidate_reasoning
            )
            if reasoning:
                final_thinking = reasoning[scenario.framing].format(
                    symbol=call_result.output["gene_symbol"],
                    count=len(call_result.output["candidate_gene_ids"]),
                    target_id=scenario.target.gene_id,
                )
        if scenario.split == "train" and scenario.form == 8 and mygene_match_reasoning:
            final_thinking = mygene_match_reasoning[scenario.framing].format(
                query=source["input"], gene_id=call_result.output["gene_id"],
                symbol=call_result.output["gene_symbol"],
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
        9: "tool_error",
    }.get(scenario.form)
    if form == 6 and scenario.ambiguous_target_is_candidate:
        blocked_code = "ambiguous_symbol"
    row = {
        "row_id": row_id,
        "messages": messages,
        "tools": tools,
        "metadata": {
            "book_mode": "tool_call",
            "step": "S1",
            "question_family": "cross_context_alignment",
            "form": scenario.form,
            "source_trace": scenario.source_trace,
            "split_unit": scenario.metadata_split_unit,
            "blocked_reason_code": blocked_code,
            "schema_version": schema_version,
            "system_prompt_sha256": system_prompt_sha256(contract),
            "tool_contract_version": contract["runtime"][
                "tool_contract_version"
            ],
            "ensembl_release": contract["coordinate_system"][
                "ensembl_release"
            ],
            "identifier_registry_id": contract["coordinate_system"][
                "identifier_registry_id"
            ],
        },
    }
    return row, evidence


THINKING_FILTER_RULES = (
    "no source value before it is visible",
    "tool status agrees with the final answer",
    "decision agrees with the final answer",
    "no answer-schema or provenance discussion",
)
_ANSWER_SCHEMA_PHRASES = (
    "answer schema",
    "json schema",
    "answer fields",
    "result field",
    "result fields",
    "status field",
    "status fields",
    "provenance and status",
    "provenance field",
    "blocked_reason field",
    "continuation_decision field",
)
_PREMATURE_OUTCOME_PHRASES = (
    "are the same gene",
    "is the same gene",
    "name the same gene",
    "names the same gene",
    "not the same gene",
    "different gene",
    "resolves to",
    "maps to",
    "is a candidate",
    "not a candidate",
    "lookup failed",
    "not found",
)


def _final_answer(row: Mapping[str, Any]) -> Mapping[str, Any]:
    assistants = [
        message
        for message in row["messages"]
        if message.get("role") == "assistant" and "content" in message
    ]
    if not assistants or not isinstance(assistants[-1]["content"], Mapping):
        raise CorpusBuildError(f"row {row['row_id']} has no final answer")
    return assistants[-1]["content"]


def _protected_values(scenario: Scenario) -> set[str]:
    values: set[str] = {scenario.target.gene_id}
    if scenario.target.gene_symbol:
        values.add(scenario.target.gene_symbol)
    if isinstance(scenario.fact, GeneFact):
        values.add(scenario.fact.gene_id)
        if scenario.fact.gene_symbol:
            values.add(scenario.fact.gene_symbol)
    elif isinstance(scenario.fact, AmbiguousFact):
        values.add(scenario.fact.gene_symbol)
        values.update(scenario.fact.candidate_gene_ids)
    else:
        values.add(scenario.fact.query)
        values.add(scenario.fact.gene_id)
        if scenario.fact.gene_symbol:
            values.add(scenario.fact.gene_symbol)
    if scenario.distractor is not None:
        values.add(scenario.distractor.gene_id)
        if scenario.distractor.gene_symbol:
            values.add(scenario.distractor.gene_symbol)
    return {value for value in values if len(value) >= 2}


def _contains_value(text: str, value: str) -> bool:
    return re.search(
        r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    ) is not None


def filter_row_thinking(
    row: Mapping[str, Any], scenario: Scenario, *,
    match_reasoning: Mapping[str, str] | None = None,
    mismatch_reasoning: Mapping[str, str] | None = None,
    unresolved_symbol_reasoning: Mapping[str, str] | None = None,
    ambiguous_candidate_reasoning: Mapping[str, str] | None = None,
    ambiguous_noncandidate_reasoning: Mapping[str, str] | None = None,
    mygene_match_reasoning: Mapping[str, str] | None = None,
) -> int:
    """Reject a thought that violates an S1 runtime rule."""

    messages = row["messages"]
    if scenario.split == "train" and scenario.form == 1 and match_reasoning:
        output = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
        expected = match_reasoning[scenario.framing].format(
            symbol=output["gene_symbol"], gene_id=output["gene_id"],
        )
        if messages[-1].get("thinking") != expected:
            raise CorpusBuildError(f"row {row['row_id']} has unbound match reasoning")
    if scenario.split == "train" and scenario.form == 2 and mismatch_reasoning:
        output = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
        expected = mismatch_reasoning[scenario.framing].format(
            symbol=output["gene_symbol"], gene_id=output["gene_id"],
            target_id=scenario.target.gene_id,
        )
        if messages[-1].get("thinking") != expected:
            raise CorpusBuildError(f"row {row['row_id']} has unbound mismatch reasoning")
    if scenario.split == "train" and scenario.form == 5 and unresolved_symbol_reasoning:
        output = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
        expected = unresolved_symbol_reasoning[scenario.framing].format(
            input=output["gene_symbol"], target_id=scenario.target.gene_id,
        )
        if messages[-1].get("thinking") != expected:
            raise CorpusBuildError(f"row {row['row_id']} has unbound unresolved-symbol reasoning")
    if scenario.split == "train" and scenario.form == 6:
        reasoning = (
            ambiguous_candidate_reasoning if scenario.ambiguous_target_is_candidate
            else ambiguous_noncandidate_reasoning
        )
        if reasoning:
            output = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
            expected = reasoning[scenario.framing].format(
                symbol=output["gene_symbol"], count=len(output["candidate_gene_ids"]),
                target_id=scenario.target.gene_id,
            )
            if messages[-1].get("thinking") != expected:
                raise CorpusBuildError(f"row {row['row_id']} has unbound ambiguous-comparison reasoning")
    if scenario.split == "train" and scenario.form == 8 and mygene_match_reasoning:
        if not isinstance(scenario.fact, ExternalFact):
            raise CorpusBuildError("form 8 lacks an external fact")
        output = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
        expected = mygene_match_reasoning[scenario.framing].format(
            query=scenario.fact.query, gene_id=output["gene_id"], symbol=output["gene_symbol"],
        )
        if messages[-1].get("thinking") != expected:
            raise CorpusBuildError(f"row {row['row_id']} has unbound mygene-match reasoning")
    user_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    if not user_indexes:
        raise CorpusBuildError(f"row {row['row_id']} has no user message")
    current_turn = user_indexes[-1]
    protected = _protected_values(scenario)
    visible = ""
    checked = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        if role != "assistant":
            content = message.get("content")
            visible += " " + (
                content
                if isinstance(content, str)
                else _canonical_json(content, sort_keys=False)
            )
            continue
        if index < current_turn:
            if "thinking" in message:
                raise CorpusBuildError(
                    f"row {row['row_id']} trains earlier-turn thinking"
                )
            continue
        thinking = message.get("thinking")
        if not isinstance(thinking, str) or not thinking.strip():
            raise CorpusBuildError(
                f"row {row['row_id']} has an assistant action without thinking"
            )
        checked += 1
        lowered = thinking.casefold()
        if any(phrase in lowered for phrase in _ANSWER_SCHEMA_PHRASES):
            raise CorpusBuildError(
                f"row {row['row_id']} thinking discusses the answer schema"
            )
        for value in protected:
            if _contains_value(thinking, value) and not _contains_value(
                visible, value
            ):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking reveals {value!r} too early"
                )
        if "tool_call" in message and any(
            phrase in lowered for phrase in _PREMATURE_OUTCOME_PHRASES
        ):
            raise CorpusBuildError(
                f"row {row['row_id']} thinking predicts the lookup outcome"
            )
        if "tool_call" not in message:
            answer = message.get("content")
            if not isinstance(answer, Mapping):
                raise CorpusBuildError(
                    f"row {row['row_id']} has no final answer object"
                )
            statuses = {
                status
                for status in (
                    "resolved",
                    "ambiguous",
                    "not_found",
                    "invalid_input",
                    "error",
                )
                if re.search(rf"\b{re.escape(status)}\b", lowered)
            }
            if "not found" in lowered or "not-found" in lowered:
                statuses.add("not_found")
            if statuses.difference({str(answer["status"])}):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking has the wrong status"
                )
            decisions = {
                decision
                for decision in ("continue", "stop", "defer")
                if re.search(rf"\b{decision}\b", lowered)
            }
            if decisions.difference(
                {str(answer["continuation_decision"])}
            ):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking has the wrong decision"
                )
            result = answer.get("result")
            expected_same = (
                result.get("same_gene")
                if isinstance(result, Mapping) and "same_gene" in result
                else "unavailable"
            )
            negative_same_gene = any(
                phrase in lowered
                for phrase in (
                    "different",
                    "not the same gene",
                    "cannot be the same gene",
                    "cannot name the same gene",
                )
            )
            claims_true = any(
                phrase in lowered
                for phrase in (
                    "are the same gene",
                    "name the same gene",
                    "names the same gene",
                    "the same id as",
                    "ids are equal",
                )
            ) and not negative_same_gene
            claims_false = negative_same_gene
            claims_unknown = any(
                phrase in lowered
                for phrase in (
                    "cannot make the comparison",
                    "cannot say whether",
                    "does not choose",
                    "set same_gene to null",
                )
            )
            wrong_claim = (
                (expected_same is True and (claims_false or claims_unknown))
                or (expected_same is False and (claims_true or claims_unknown))
                or (expected_same is None and (claims_true or claims_false))
                or (
                    expected_same == "unavailable"
                    and (claims_true or claims_false)
                )
            )
            if wrong_claim:
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking makes the wrong "
                    "alignment claim"
                )
            if scenario.effective_form == 6:
                target_is_candidate = bool(
                    scenario.ambiguous_target_is_candidate
                )
                if target_is_candidate and any(
                    phrase in lowered
                    for phrase in ("not a candidate", "not one of them")
                ):
                    raise CorpusBuildError(
                        f"row {row['row_id']} thinking misstates candidacy"
                    )
                if not target_is_candidate and any(
                    phrase in lowered
                    for phrase in ("is one candidate", "is one of them")
                ):
                    raise CorpusBuildError(
                        f"row {row['row_id']} thinking misstates candidacy"
                    )
    return checked


MASKED_WORDING_TOLERANCE = 0.02
MASK_TOKEN = "<INPUT>"


def _current_user_message(row: Mapping[str, Any]) -> str:
    users = [
        message["content"]
        for message in row["messages"]
        if message.get("role") == "user"
    ]
    if not users or not isinstance(users[-1], str):
        raise CorpusBuildError("an S1.4 row has no current user text")
    return users[-1]


def _mask_value(text: str, value: str, replacement: str) -> str:
    if not value:
        return text
    return re.sub(
        r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])",
        replacement,
        text,
        flags=re.IGNORECASE,
    )


def masked_visible_prompt(
    row: Mapping[str, Any], scenario: Scenario
) -> str:
    """Mask source values but retain the full visible prompt structure."""

    messages = row["messages"]
    last_user = max(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    )
    parts = []
    for message in messages[1 : last_user + 1]:
        content = message.get("content")
        rendered = (
            content
            if isinstance(content, str)
            else _canonical_json(content, sort_keys=False)
        )
        parts.append(
            f"{message['role']}: " + " ".join(str(rendered).split())
        )
    visible = "\n".join(parts)
    visible = re.sub(
        r"ev_reg_[a-z0-9]{6}",
        "<EVIDENCE_ID>",
        visible,
        flags=re.IGNORECASE,
    )
    visible = re.sub(
        r"group_[a-f0-9]{8}", "<GROUP_ID>", visible, flags=re.IGNORECASE
    )
    identifiers = _protected_values(scenario)
    source_values = {"116"}
    if isinstance(scenario.fact, AmbiguousFact):
        source_values.add(str(len(scenario.fact.candidate_gene_ids)))
        for details in scenario.fact.candidate_details.values():
            source_values.update(str(value) for value in details.values())
    for value in sorted(identifiers, key=len, reverse=True):
        visible = _mask_value(visible, value, MASK_TOKEN)
    for value in sorted(source_values, key=len, reverse=True):
        visible = _mask_value(visible, value, "<SOURCE_VALUE>")
    return "\n".join(" ".join(line.split()) for line in visible.splitlines())


def _masked_user_wording(
    row: Mapping[str, Any],
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = ALIGNMENT_PROMPTS,
) -> str:
    user = _current_user_message(row)
    template = prompts[scenario.split][scenario.framing][
        scenario.template_index
    ]
    prefix, suffix = template.split("{request}")
    if not user.startswith(prefix) or (suffix and not user.endswith(suffix)):
        raise CorpusBuildError(
            f"row {row['row_id']} does not match its alignment template"
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
    train_rows = rows_by_split["train"]
    train_scenarios = scenarios_by_split["train"]
    labels = tuple(sorted({scenario.form for scenario in train_scenarios}))
    token_counts: dict[int, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[int] = Counter()
    vocabulary: set[str] = set()
    for row, scenario in zip(train_rows, train_scenarios, strict=True):
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
        form_distribution = Counter(item.form for item in scenarios)
        correct = sum(
            predict(_wording_tokens(masker(row, scenario))) == scenario.form
            for row, scenario in zip(rows, scenarios, strict=True)
        )
        accuracy = correct / len(rows)
        baseline = max(form_distribution.values()) / len(rows)
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
            "masked user wording predicts the S1.4 form above the limit"
        )
    return result


def masked_wording_classifier_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    tolerance: float = MASKED_WORDING_TOLERANCE,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = ALIGNMENT_PROMPTS,
) -> dict[str, Any]:
    """Gate leakage from the outer alignment wrappers."""

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


def _without_evidence_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_evidence_ids(item)
            for key, item in value.items()
            if key != "evidence_id"
        }
    if isinstance(value, list):
        return [_without_evidence_ids(item) for item in value]
    if isinstance(value, str):
        value = re.sub(
            r"group_[a-f0-9]{8}",
            "<GROUP_ID>",
            value,
            flags=re.IGNORECASE,
        )
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
            r'("evidence_id"\s*:\s*)"[^"]*"', r'\1"<REMOVED>"', value
        )
    return value


def _prompt_and_outcome_signatures(
    row: Mapping[str, Any],
) -> tuple[str, str, str]:
    messages = row["messages"]
    last_user = max(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    )
    prompt = [_without_evidence_ids(item) for item in messages[: last_user + 1]]
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
    """Reject repeated prompt outcomes except required failure pairs."""

    split_results: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        prompt_groups: dict[str, list[tuple[Scenario, str]]] = defaultdict(list)
        combined_groups: dict[str, list[Scenario]] = defaultdict(list)
        for row, scenario in zip(rows, scenarios, strict=True):
            prompt, outcome, combined = _prompt_and_outcome_signatures(row)
            prompt_groups[prompt].append((scenario, outcome))
            combined_groups[combined].append(scenario)
        duplicates = [items for items in combined_groups.values() if len(items) > 1]
        if duplicates:
            raise CorpusBuildError(
                f"{split} repeats a prompt and tool outcome after normalization"
            )
        pair_count = 0
        for items in prompt_groups.values():
            if len(items) == 1:
                continue
            if len(items) != 2:
                raise CorpusBuildError(f"{split} has a prompt shared by many rows")
            failure = next((item for item in items if item[0].form == 9), None)
            success = next((item for item in items if item[0].form == 1), None)
            if failure is None or success is None:
                raise CorpusBuildError(f"{split} has an unintended prompt pair")
            if (
                failure[0].paired_success_id != success[0].scenario_id
                or failure[1] == success[1]
            ):
                raise CorpusBuildError(f"{split} has an invalid failure pair")
            pair_count += 1
        split_results[split] = {
            "row_count": len(rows),
            "normalized_prompt_count": len(prompt_groups),
            "normalized_prompt_tool_outcome_count": len(combined_groups),
            "duplicate_prompt_tool_outcome_count": 0,
            "intended_success_failure_pair_count": pair_count,
            "pass": True,
        }
    return {
        "normalization": (
            "remove evidence IDs, normalize generated group IDs, and retain "
            "tool names and statuses"
        ),
        "splits": split_results,
        "pass": True,
    }


def _tool_calls(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        message["tool_call"]
        for message in row["messages"]
        if message.get("role") == "assistant" and "tool_call" in message
    ]


def _case_input_pair(
    scenario: Scenario, contract: Mapping[str, Any]
) -> tuple[str, str] | None:
    form = scenario.effective_form
    source = _source_mention(_mentions(scenario, contract))
    if form in {1, 2, 7}:
        if not isinstance(scenario.fact, GeneFact):
            return None
        return str(source["input"]), str(scenario.fact.gene_symbol)
    if form == 5:
        if not isinstance(scenario.fact, ExternalFact):
            return None
        return str(source["input"]), scenario.fact.query
    if form == 6:
        if not isinstance(scenario.fact, AmbiguousFact):
            return None
        return str(source["input"]), scenario.fact.gene_symbol
    if form == 8:
        if not isinstance(scenario.fact, ExternalFact):
            return None
        module = next(
            mention
            for mention in _mentions(scenario, contract)
            if mention["item"] == "module"
        )
        return str(module["input"]), scenario.fact.gene_id
    return None


def _lexical_values(scenario: Scenario) -> set[str]:
    values = _protected_values(scenario)
    return {value.casefold() for value in values if value}


def audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    expected_counts: Mapping[str, Mapping[int, int]],
    *,
    framing_shares: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = ALIGNMENT_PROMPTS,
) -> dict[str, Any]:
    """Audit rendered rows against every S1.4 generation invariant."""

    if framing_shares is None:
        framing_shares = {"direct": Fraction(1, 2), "step": Fraction(1, 2)}
    if contract is None:
        contract = load_contract()
    row_ids: set[str] = set()
    units_by_split: dict[str, set[str]] = {}
    lexical_by_split: dict[str, set[str]] = {}
    status_counts: dict[str, dict[str, int]] = {}
    same_gene_counts: dict[str, dict[str, int]] = {}
    call_counts: dict[str, dict[str, int]] = {}
    tool_name_counts: dict[str, dict[str, int]] = {}
    framing_counts: dict[str, dict[str, int]] = {}
    case_counts: dict[str, dict[str, int]] = {}
    notation_counts: dict[str, dict[str, int]] = {}
    template_counts: dict[str, dict[str, int]] = {}
    orientation_counts: dict[str, dict[str, dict[str, int]]] = {}
    item_pair_counts: dict[str, dict[str, dict[str, int]]] = {}
    outcome_counts: dict[str, dict[str, int]] = {}
    pair_counts: dict[str, int] = {}
    false_defer_counts: dict[str, dict[str, int | float | bool]] = {}
    thinking_counts: dict[str, dict[str, int | bool]] = {}
    registry_counts: dict[str, dict[str, int]] = {}
    mygene_counts: dict[str, dict[str, int]] = {}
    abstract_counts: dict[str, dict[str, int]] = {}
    cue_counts: dict[str, dict[str, int]] = {}
    form_gate_counts: dict[str, dict[str, dict[str, Any]]] = {}

    for split in SPLITS:
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} rows and scenarios differ")
        actual_forms = Counter(
            int(row["metadata"]["form"]) for row in rows
        )
        if dict(sorted(actual_forms.items())) != dict(expected_counts[split]):
            raise CorpusBuildError(f"{split} form counts changed")
        expected_frames = largest_remainder_counts(
            len(rows), framing_shares
        )
        actual_frames = Counter(item.framing for item in scenarios)
        if dict(actual_frames) != expected_frames:
            raise CorpusBuildError(f"{split} framing counts changed")

        successes = {
            scenario.scenario_id: (scenario, row)
            for scenario, row in zip(scenarios, rows, strict=True)
            if scenario.form == 1
        }
        statuses: Counter[str] = Counter()
        same_genes: Counter[str] = Counter()
        calls: Counter[str] = Counter()
        tools_used: Counter[str] = Counter()
        templates: Counter[str] = Counter()
        orientations: dict[str, Counter[str]] = defaultdict(Counter)
        item_pairs: dict[str, Counter[str]] = defaultdict(Counter)
        outcomes: Counter[str] = Counter()
        units: set[str] = set()
        lexical_values: set[str] = set()
        eligible_case = 0
        perturbed_case = 0
        mandatory_notation = 0
        failure_pairs = 0
        answerable = 0
        false_defers = 0
        thinking_steps = 0
        registry_structure: Counter[str] = Counter()
        mygene_structure: Counter[str] = Counter()
        abstract_structure: Counter[str] = Counter()
        ambiguous_cues: dict[str, Counter[str]] = {
            "candidate": Counter(),
            "noncandidate": Counter(),
        }
        form_gates: dict[str, Counter[str]] = defaultdict(Counter)

        for scenario, row in zip(scenarios, rows, strict=True):
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise CorpusBuildError(f"duplicate row ID {row_id}")
            row_ids.add(row_id)
            units.update(scenario.split_units)
            lexical_values.update(_lexical_values(scenario))
            answer = _final_answer(row)
            status = str(answer["status"])
            statuses[status] += 1
            result = answer["result"]
            if isinstance(result, Mapping) and "same_gene" in result:
                same = result["same_gene"]
                same_key = (
                    "true" if same is True else "false" if same is False else "null"
                )
                first_item = str(result["first"]["item"])
                second_item = str(result["second"]["item"])
            else:
                same_key = "unavailable"
                mentions = _mentions(scenario, contract)
                first_item = str(mentions[0]["item"])
                second_item = str(mentions[1]["item"])
            same_genes[same_key] += 1
            form_key = f"S1.4.{scenario.form}"
            form_gates[form_key]["row_count"] += 1
            orientations[form_key][str(scenario.orientation % 2)] += 1
            item_pairs[form_key][f"{first_item}->{second_item}"] += 1
            templates[
                f"{form_key}:{scenario.framing}:{scenario.template_index}"
            ] += 1

            row_calls = _tool_calls(row)
            calls[str(len(row_calls))] += 1
            tool_key = row_calls[0]["name"] if row_calls else "none"
            tools_used[str(tool_key)] += 1
            expected_call_count = 0 if scenario.effective_form in {3, 4, 7} else 1
            if len(row_calls) != expected_call_count:
                raise CorpusBuildError(
                    f"row {row_id} has the wrong number of calls"
                )
            form_gates[form_key]["exact_call_count"] += 1

            mentions = _mentions(scenario, contract)
            source = _source_mention(mentions)
            expected_tool_name: str | None
            expected_arguments: Mapping[str, Any] | None
            if scenario.effective_form in {3, 4, 7}:
                expected_tool_name = None
                expected_arguments = None
            elif scenario.effective_form == 8:
                expected_tool_name = GENE_ID_TOOL
                expected_arguments = {"gene_id": source["lookup_input"]}
            else:
                expected_tool_name = SYMBOL_TOOL
                expected_arguments = {"gene_symbol": source["input"]}
            if expected_tool_name is not None and (
                row_calls[0].get("name") != expected_tool_name
                or row_calls[0].get("arguments") != expected_arguments
            ):
                raise CorpusBuildError(f"row {row_id} has the wrong tool call")
            form_gates[form_key]["exact_tool_count"] += 1

            expected_status = {
                1: "resolved",
                2: "resolved",
                3: "resolved",
                4: "resolved",
                5: "not_found",
                7: "resolved",
                8: "resolved",
                9: "error",
            }.get(scenario.form)
            if scenario.form == 6:
                expected_status = (
                    "ambiguous"
                    if scenario.ambiguous_target_is_candidate
                    else "resolved"
                )
                outcome = (
                    "candidate"
                    if scenario.ambiguous_target_is_candidate
                    else "noncandidate"
                )
                outcomes[outcome] += 1
                ambiguous_cues[outcome][
                    masked_visible_prompt(row, scenario)
                ] += 1
            if status != expected_status:
                raise CorpusBuildError(f"row {row_id} has the wrong status")
            form_gates[form_key]["exact_status_count"] += 1

            expected_same = {
                1: True,
                2: False,
                3: True,
                4: False,
                5: None,
                7: True,
                8: True,
            }.get(scenario.form)
            if scenario.form == 6:
                expected_same = (
                    None
                    if scenario.ambiguous_target_is_candidate
                    else False
                )
            if scenario.form == 9:
                expected_failure_result = {
                    "failed_tool": SYMBOL_TOOL,
                    "error_code": "BACKEND_UNAVAILABLE",
                    "gene_symbol": source["input"],
                }
                if result != expected_failure_result:
                    raise CorpusBuildError(
                        f"row {row_id} has the wrong failure result"
                    )
            else:
                expected_result = _comparison_result(
                    mentions, expected_same
                )
                if result != expected_result:
                    raise CorpusBuildError(
                        f"row {row_id} has the wrong comparison result"
                    )
            form_gates[form_key]["exact_result_count"] += 1

            if scenario.form in {5, 9} or (
                scenario.form == 6
                and scenario.ambiguous_target_is_candidate
            ):
                expected_decision = "defer"
            else:
                expected_decision = _decision(scenario, usable=True)
            if answer.get("continuation_decision") != expected_decision:
                raise CorpusBuildError(f"row {row_id} has the wrong decision")
            form_gates[form_key]["exact_decision_count"] += 1
            if scenario.form != 9 and expected_same is None:
                form_gates[form_key]["null_result_count"] += 1
                if answer.get("continuation_decision") != "defer":
                    raise CorpusBuildError(
                        f"row {row_id} does not defer after a null result"
                    )
                form_gates[form_key]["null_result_defer_count"] += 1

            if same_key in {"true", "false"}:
                answerable += 1
                if answer["continuation_decision"] == "defer":
                    false_defers += 1
            thinking_steps += filter_row_thinking(
                row, scenario, match_reasoning=(settings or {}).get("train_match_reasoning"),
                mismatch_reasoning=(settings or {}).get("train_mismatch_reasoning"),
                unresolved_symbol_reasoning=(settings or {}).get("train_unresolved_symbol_reasoning"),
                ambiguous_candidate_reasoning=(settings or {}).get("train_ambiguous_candidate_reasoning"),
                ambiguous_noncandidate_reasoning=(settings or {}).get("train_ambiguous_noncandidate_reasoning"),
                mygene_match_reasoning=(settings or {}).get("train_mygene_match_reasoning"),
            )
            case_pair = _case_input_pair(scenario, contract)
            if case_pair is not None:
                eligible_case += 1
                visible_input, canonical_input = case_pair
                if (
                    visible_input != canonical_input
                    and visible_input.casefold() == canonical_input.casefold()
                ):
                    perturbed_case += 1
            if scenario.effective_form == 3:
                mandatory_notation += 1

            visible_prefix = "\n".join(
                str(message.get("content", ""))
                for message in row["messages"]
                if message.get("role") in {"harness", "user"}
            )
            has_registry = "Registry entries, Ensembl release 116:" in visible_prefix
            has_mygene = "query_mygene result:" in visible_prefix
            has_abstract = "Abstract (example text, not a real abstract):" in visible_prefix
            if has_registry:
                registry_structure[form_key] += 1
            if has_mygene:
                mygene_structure[form_key] += 1
            if has_abstract:
                abstract_structure[form_key] += 1
            if has_registry != (scenario.effective_form == 7):
                raise CorpusBuildError(f"row {row_id} has a wrong registry block")
            if has_mygene != (scenario.effective_form == 8):
                raise CorpusBuildError(f"row {row_id} has a wrong MyGene block")
            if has_abstract != (scenario.effective_form in {2, 5}):
                raise CorpusBuildError(f"row {row_id} has a wrong abstract block")

            if scenario.form == 9:
                pair = successes.get(str(scenario.paired_success_id))
                if pair is None:
                    raise CorpusBuildError("a failure has no retained success")
                success_scenario, success_row = pair
                prompt, _, _ = _prompt_and_outcome_signatures(row)
                success_prompt, _, _ = _prompt_and_outcome_signatures(
                    success_row
                )
                if (
                    prompt != success_prompt
                    or scenario.fact != success_scenario.fact
                    or scenario.target != success_scenario.target
                    or scenario.orientation != success_scenario.orientation
                ):
                    raise CorpusBuildError("a failure changed its success prompt")
                failure_pairs += 1

        if failure_pairs != int(expected_counts[split][9]):
            raise CorpusBuildError(f"{split} failure-pair count changed")
        expected_outcomes = (
            ambiguous_outcome_counts(int(expected_counts[split][6]), settings)
            if settings is not None
            else largest_remainder_counts(
                int(expected_counts[split][6]),
                {"candidate": Fraction(1, 2), "noncandidate": Fraction(1, 2)},
            )
        )
        if dict(outcomes) != expected_outcomes:
            raise CorpusBuildError(f"{split} ambiguous outcomes changed")
        all_cues = set(ambiguous_cues["candidate"]) | set(
            ambiguous_cues["noncandidate"]
        )
        if any(
            abs(
                ambiguous_cues["candidate"][cue]
                - ambiguous_cues["noncandidate"][cue]
            )
            > 1
            for cue in all_cues
        ):
            raise CorpusBuildError(
                f"{split} prompt structure predicts the form-6 outcome"
            )
        for form in range(1, 10):
            form_key = f"S1.4.{form}"
            if set(orientations[form_key]) != {"0", "1"}:
                raise CorpusBuildError(
                    f"{split} form {form} lacks both orientations"
                )
            for framing in ("direct", "step"):
                used = {
                    scenario.template_index
                    for scenario in scenarios
                    if scenario.form == form and scenario.framing == framing
                }
                expected_templates = set(range(len(prompts[split][framing])))
                if used != expected_templates:
                    raise CorpusBuildError(
                        f"{split} form {form} {framing} lacks template coverage"
                    )

        units_by_split[split] = units
        lexical_by_split[split] = lexical_values
        status_counts[split] = dict(sorted(statuses.items()))
        same_gene_counts[split] = {
            key: same_genes[key]
            for key in ("true", "false", "null", "unavailable")
        }
        call_counts[split] = dict(sorted(calls.items()))
        tool_name_counts[split] = dict(sorted(tools_used.items()))
        framing_counts[split] = dict(sorted(actual_frames.items()))
        case_counts[split] = {
            "eligible": eligible_case,
            "perturbed": perturbed_case,
        }
        notation_counts[split] = {"mandatory_form3": mandatory_notation}
        template_counts[split] = dict(sorted(templates.items()))
        orientation_counts[split] = {
            key: dict(sorted(value.items()))
            for key, value in sorted(orientations.items())
        }
        item_pair_counts[split] = {
            key: dict(sorted(value.items()))
            for key, value in sorted(item_pairs.items())
        }
        outcome_counts[split] = dict(sorted(outcomes.items()))
        pair_counts[split] = failure_pairs
        false_defer_counts[split] = {
            "eligible_row_count": answerable,
            "false_defer_count": false_defers,
            "false_defer_rate": false_defers / answerable,
            "pass": false_defers == 0,
        }
        thinking_counts[split] = {
            "row_count": len(rows),
            "thinking_step_count": thinking_steps,
            "pass": True,
        }
        registry_counts[split] = {
            "S1.4.7": registry_structure["S1.4.7"],
            "unexpected": sum(
                count
                for key, count in registry_structure.items()
                if key != "S1.4.7"
            ),
        }
        mygene_counts[split] = {
            "S1.4.8": mygene_structure["S1.4.8"],
            "unexpected": sum(
                count
                for key, count in mygene_structure.items()
                if key != "S1.4.8"
            ),
        }
        abstract_counts[split] = {
            "S1.4.2": abstract_structure["S1.4.2"],
            "S1.4.5": abstract_structure["S1.4.5"],
            "unexpected": sum(
                count
                for key, count in abstract_structure.items()
                if key not in {"S1.4.2", "S1.4.5"}
            ),
        }
        cue_counts[split] = {
            "candidate_row_count": sum(ambiguous_cues["candidate"].values()),
            "noncandidate_row_count": sum(
                ambiguous_cues["noncandidate"].values()
            ),
            "cue_count": len(all_cues),
            "maximum_count_difference": max(
                (
                    abs(
                        ambiguous_cues["candidate"][cue]
                        - ambiguous_cues["noncandidate"][cue]
                    )
                    for cue in all_cues
                ),
                default=0,
            ),
        }
        form_gate_counts[split] = {}
        for form in range(1, 10):
            form_key = f"S1.4.{form}"
            values = form_gates[form_key]
            row_count = values["row_count"]
            gate = {
                "row_count": row_count,
                "exact_status_count": values["exact_status_count"],
                "exact_status_rate": values["exact_status_count"] / row_count,
                "exact_result_count": values["exact_result_count"],
                "exact_result_rate": values["exact_result_count"] / row_count,
                "exact_decision_count": values["exact_decision_count"],
                "exact_decision_rate": values["exact_decision_count"] / row_count,
                "exact_call_count": values["exact_call_count"],
                "exact_call_rate": values["exact_call_count"] / row_count,
                "exact_tool_count": values["exact_tool_count"],
                "exact_tool_rate": values["exact_tool_count"] / row_count,
                "null_result_count": values["null_result_count"],
                "null_result_defer_count": values[
                    "null_result_defer_count"
                ],
            }
            gate["pass"] = all(
                gate[key] == row_count
                for key in (
                    "exact_status_count",
                    "exact_result_count",
                    "exact_decision_count",
                    "exact_call_count",
                    "exact_tool_count",
                )
            ) and (
                gate["null_result_count"]
                == gate["null_result_defer_count"]
            )
            form_gate_counts[split][form_key] = gate

    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = units_by_split[first] & units_by_split[second]
        if overlap:
            raise CorpusBuildError(
                f"identity leakage between {first} and {second}: "
                f"{sorted(overlap)[:3]!r}"
            )
        lexical_overlap = lexical_by_split[first] & lexical_by_split[second]
        if lexical_overlap:
            raise CorpusBuildError(
                f"lexical identity leakage between {first} and {second}: "
                f"{sorted(lexical_overlap)[:3]!r}"
            )

    duplicate_audit = duplicate_prompt_outcome_audit(
        rows_by_split, scenarios_by_split
    )
    wording_audit = masked_wording_classifier_audit(
        rows_by_split, scenarios_by_split, prompts=prompts
    )
    full_prompt = full_visible_prompt_classifier_diagnostic(
        rows_by_split, scenarios_by_split
    )
    total_answerable = sum(
        int(value["eligible_row_count"])
        for value in false_defer_counts.values()
    )
    total_false_defers = sum(
        int(value["false_defer_count"])
        for value in false_defer_counts.values()
    )
    return {
        "status_counts": status_counts,
        "same_gene_counts": same_gene_counts,
        "tool_call_counts": call_counts,
        "tool_name_counts": tool_name_counts,
        "framing_counts": framing_counts,
        "case_counts": case_counts,
        "notation_counts": notation_counts,
        "prompt_template_counts": template_counts,
        "orientation_counts": orientation_counts,
        "item_pair_counts": item_pair_counts,
        "ambiguous_outcome_counts": outcome_counts,
        "per_form_gates": form_gate_counts,
        "failure_pair_counts": pair_counts,
        "false_defer_audit": {
            "definition": "a row defers although the comparison is answerable",
            "eligible_row_count": total_answerable,
            "false_defer_count": total_false_defers,
            "false_defer_rate": total_false_defers / total_answerable,
            "splits": false_defer_counts,
            "pass": total_false_defers == 0,
        },
        "prompt_structure_audit": {
            "registry_block_counts": registry_counts,
            "mygene_block_counts": mygene_counts,
            "abstract_block_counts": abstract_counts,
            "form6_pretool_cue_counts": cue_counts,
            "pass": True,
        },
        "thinking_filter": {
            "rules": list(THINKING_FILTER_RULES),
            "splits": thinking_counts,
            "pass": True,
        },
        "duplicate_prompt_tool_outcome": duplicate_audit,
        "masked_wording_classifier": wording_audit,
        "full_visible_prompt_classifier": full_prompt,
        "split_unit_hashes": {
            split: _stable_sha256(sorted(values))
            for split, values in units_by_split.items()
        },
        "lexical_input_hashes": {
            split: _stable_sha256(sorted(values))
            for split, values in lexical_by_split.items()
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
    mygene_source_path: Path | None = None,
    validate: bool = True,
    overlay: Mapping[str, str] | None = None,
    prompt_banks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate S1.4, then publish its manifest last.

    A pinned prompt bank replaces only the training wrappers; validation and
    test keep their wrappers, so their rows do not change.
    """

    source_index_path = source_index_path.resolve()
    payload = check_source_index(source_index_path, contract)
    curriculum = contract["curriculum"]
    settings = curriculum["families"]["S1.4"]
    seed = int(settings["assignment_seed"])
    prompts = (
        wrapper_prompts(prompt_banks, ALIGNMENT_PROMPTS, bank_name="alignment")
        if prompt_banks is not None
        else ALIGNMENT_PROMPTS
    )
    row_counts = split_row_counts(
        int(settings["train_rows"]), curriculum["split_fractions"]
    )
    counts_by_split = {
        split: form_counts(row_counts[split], settings, curriculum)
        for split in SPLITS
    }

    registry_spec = contract["coordinate_system"]["identifier_registry"]
    registry_path = _configured_path(str(registry_spec["source_path"]))
    if file_sha256(registry_path) != str(registry_spec["source_sha256"]):
        raise CorpusBuildError("the Ensembl registry source hash changed")
    gene_versions = read_ensembl_gtf_identifiers(registry_path)

    mygene_spec = settings["mygene_source"]
    configured_mygene_path = _configured_path(str(mygene_spec["source_path"]))
    mygene_path = (
        configured_mygene_path
        if mygene_source_path is None
        else mygene_source_path.resolve()
    )
    configured_mygene_sha = str(mygene_spec["source_sha256"])
    if configured_mygene_sha != MYGENE_SOURCE_SHA256:
        raise CorpusBuildError("the configured MyGene source pin changed")
    if file_sha256(mygene_path) != configured_mygene_sha:
        raise CorpusBuildError("the MyGene source hash changed")
    raw_mygene = json.loads(mygene_path.read_text(encoding="utf-8"))
    if not isinstance(raw_mygene, Mapping):
        raise CorpusBuildError("the MyGene source is not a query mapping")
    if len(raw_mygene) != int(mygene_spec["query_count"]):
        raise CorpusBuildError("the MyGene query count changed")
    usable_mygene_count = len(_usable_mygene_records(payload, raw_mygene))
    if usable_mygene_count != int(mygene_spec["usable_mapping_count"]):
        raise CorpusBuildError("the usable MyGene mapping count changed")

    pools = build_fact_pools(
        payload,
        gene_versions,
        raw_mygene,
        s0_settings=curriculum["families"]["S0"],
        seed=seed,
        fractions=curriculum["split_fractions"],
        maximum_candidate_gene_ids=int(
            settings["maximum_candidate_gene_ids"]
        ),
    )
    pool_coverage = source_pool_coverage(pools)
    eligible_external_count = sum(
        len(pools[split]["external"]) for split in SPLITS
    )
    excluded_external_count = usable_mygene_count - eligible_external_count
    if excluded_external_count < 0:
        raise CorpusBuildError("eligible MyGene mappings exceed the raw source")

    scenarios_by_split = select_scenarios(
        pools,
        counts_by_split,
        seed=seed,
        case_rate=curriculum["identity_case_perturbation"],
        framing_shares=curriculum["framing"],
        settings=settings,
        prompts=prompts,
    )
    runtime = WorldModelToolRuntime.from_source_index(
        source_index_path,
        contract=contract,
        evidence_id_factory=EvidenceSequence(),
    )
    tools = tool_definitions(contract)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    evidence_calls: list[list[str]] = []
    for split in SPLITS:
        rows: list[dict[str, Any]] = []
        for scenario in scenarios_by_split[split]:
            row, row_evidence = build_row(
                scenario,
                runtime=runtime,
                contract=contract,
                tools=tools,
                prompts=prompts,
                train_candidate_reasoning=bool(
                    settings.get("train_candidate_reasoning")
                ),
                match_reasoning=settings.get("train_match_reasoning"),
                mismatch_reasoning=settings.get("train_mismatch_reasoning"),
                unresolved_symbol_reasoning=settings.get("train_unresolved_symbol_reasoning"),
                ambiguous_candidate_reasoning=settings.get("train_ambiguous_candidate_reasoning"),
                ambiguous_noncandidate_reasoning=settings.get("train_ambiguous_noncandidate_reasoning"),
                mygene_match_reasoning=settings.get("train_mygene_match_reasoning"),
            )
            rows.append(row)
            evidence_calls.extend(row_evidence)
        rows_by_split[split] = rows

    evidence_ids = [record[0] for record in evidence_calls]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise CorpusBuildError("an evidence ID appears in more than one row")
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
        prefix=".s1_4_build_", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        paths = {
            split: temporary / f"{split}.jsonl" for split in SPLITS
        }
        for split, path in paths.items():
            _write_jsonl(path, rows_by_split[split])
            if validate:
                _validate_corpus(path)

        form_count_manifest = {
            split: {
                f"S1.4.{form}": count
                for form, count in counts_by_split[split].items()
            }
            for split in SPLITS
        }
        mygene_manifest = {
            key: mygene_spec[key]
            for key in (
                "snapshot_id",
                "source_path",
                "source_url",
                "source_sha256",
                "query_count",
                "usable_mapping_count",
            )
        }
        mygene_manifest.update(
            {
                "eligible_external_mapping_count": eligible_external_count,
                "split_eligible_mapping_count": eligible_external_count,
                "excluded_cross_split_mapping_count": excluded_external_count,
                "split_eligibility_rule": (
                    "the mapped gene and every registry or historical gene "
                    "that uses the query spelling belong to one S0 split"
                ),
            }
        )
        resolved_contract_path = contract_path.resolve()
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "dataset_id": settings["dataset_id"],
            "section": "S1.4",
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
                "path": (
                    str(resolved_contract_path.relative_to(REPOSITORY_ROOT))
                    if resolved_contract_path.is_relative_to(REPOSITORY_ROOT)
                    else str(resolved_contract_path)
                ),
                "sha256": file_sha256(resolved_contract_path),
                "row_schema_version": contract["runtime"][
                    "row_schema_version"
                ],
                "tool_contract_version": contract["runtime"][
                    "tool_contract_version"
                ],
            },
            "sources": {
                "source_index_sha256": runtime.source_index_sha256,
                "identifier_registry_id": runtime.identifier_registry_id,
                "identifier_registry": dict(registry_spec),
                "mygene_source": mygene_manifest,
                "maximum_candidate_gene_ids": int(
                    settings["maximum_candidate_gene_ids"]
                ),
                **pool_coverage,
                "records": payload["sources"],
            },
            "assignment": {
                "seed": seed,
                "split_fractions": dict(curriculum["split_fractions"]),
                "split_unit_sha256": audit["split_unit_hashes"],
                "lexical_input_sha256": audit["lexical_input_hashes"],
            },
            "failure_injection": {
                **dict(settings["failure_injection"]),
                "share_of_training_tool_call_rows": curriculum[
                    "tool_failure_share"
                ],
                "pair_counts": audit["failure_pair_counts"],
            },
            "prompt_templates_sha256": _stable_sha256(prompts),
            "prompt_template_counts": audit["prompt_template_counts"],
            "row_counts": row_counts,
            "form_counts": form_count_manifest,
            "status_counts": audit["status_counts"],
            "same_gene_counts": audit["same_gene_counts"],
            "tool_call_counts": audit["tool_call_counts"],
            "tool_name_counts": audit["tool_name_counts"],
            "framing_counts": audit["framing_counts"],
            "case_counts": audit["case_counts"],
            "notation_counts": audit["notation_counts"],
            "orientation_counts": audit["orientation_counts"],
            "item_pair_counts": audit["item_pair_counts"],
            "ambiguous_outcome_counts": audit[
                "ambiguous_outcome_counts"
            ],
            "per_form_gates": audit["per_form_gates"],
            "false_defer_audit": audit["false_defer_audit"],
            "prompt_structure_audit": audit["prompt_structure_audit"],
            "masked_wording_classifier": audit[
                "masked_wording_classifier"
            ],
            "full_visible_prompt_classifier": audit[
                "full_visible_prompt_classifier"
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
                path.name: {
                    "row_count": row_counts[split],
                    "sha256": file_sha256(path),
                }
                for split, path in paths.items()
            },
        }
        if overlay is not None:
            manifest["contract"]["s1_overlay"] = dict(overlay)
        if prompt_banks is not None:
            manifest["prompt_bank"] = prompt_bank_manifest(prompt_banks)
        if settings.get("train_match_reasoning"):
            manifest["train_match_reasoning"] = dict(settings["train_match_reasoning"])
        if settings.get("train_mismatch_reasoning"):
            manifest["train_mismatch_reasoning"] = dict(settings["train_mismatch_reasoning"])
        if settings.get("train_unresolved_symbol_reasoning"):
            manifest["train_unresolved_symbol_reasoning"] = dict(settings["train_unresolved_symbol_reasoning"])
        if settings.get("train_ambiguous_candidate_reasoning"):
            manifest["train_ambiguous_candidate_reasoning"] = dict(settings["train_ambiguous_candidate_reasoning"])
        if settings.get("train_ambiguous_noncandidate_reasoning"):
            manifest["train_ambiguous_noncandidate_reasoning"] = dict(settings["train_ambiguous_noncandidate_reasoning"])
        if settings.get("train_mygene_match_reasoning"):
            manifest["train_mygene_match_reasoning"] = dict(settings["train_mygene_match_reasoning"])
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
    parser.add_argument("--mygene-source", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip semantic validation for a focused test",
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
    settings = contract["curriculum"]["families"]["S1.4"]
    output_directory = args.output_directory or _configured_path(
        settings["output_directory"]
    )
    manifest = build_corpus(
        source_index,
        output_directory,
        contract=contract,
        contract_path=args.contract,
        mygene_source_path=args.mygene_source,
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
