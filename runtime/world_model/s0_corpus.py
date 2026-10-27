"""Build the canonical source-index-backed S0 corpus.

The shared S0/S1 contract remains the immutable runtime wire contract. A small
content-pinned S0-v4 overlay owns only the revised corpus allocations and
additive row metadata. Generated data is published as an immutable versioned
corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from .contract import (
    CONTRACT_PATH,
    REPOSITORY_ROOT,
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from .source_index import check_source_index
from .sources import file_sha256
from .tools import (
    GENE_ID_TOOL,
    SYMBOL_TOOL,
    ToolCallResult,
    WorldModelToolRuntime,
)


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s0-corpus-v2"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-manifest-v2"
S0_V4_OVERLAY_SCHEMA_VERSION = "mentor-rl-world-model-s0-overlay-v1"
S0_V4_OVERLAY_PATH = Path(__file__).with_name("s0_v4.yaml")
S0_V4_METADATA_ADDITIONS = (
    "fact_kind",
    "framing",
    "prompt_template_id",
    "rendering_index",
    "input_variant",
)

RESOLVED_SYMBOL = "resolved_symbol"
AMBIGUOUS_SYMBOL = "ambiguous_symbol"
NAMED_GENE_ID = "named_gene_id"
UNNAMED_GENE_ID = "unnamed_gene_id"
HISTORICAL_SYMBOL = "historical_symbol"
RETIRED_GENE_ID = "retired_gene_id"

PROMPT_TEMPLATES = {
    "train": {
        "symbol": {
            "direct": (
                "What is the human Ensembl gene ID for {value}?",
                "Which human Ensembl gene ID belongs to the gene symbol {value}?",
                "Look up the human gene symbol {value} and report its Ensembl gene ID.",
                "For the input gene symbol {value}, return the matching human Ensembl gene ID.",
                "Resolve the supplied human gene symbol {value} to its Ensembl gene ID.",
                "Find the human Ensembl gene ID associated with the input symbol {value}.",
                "A human Ensembl gene ID is requested for the supplied gene symbol {value}.",
                "Return the human Ensembl gene ID after checking the gene symbol {value}.",
            ),
            "step": (
                "Resolve the human gene symbol {value} for the next analysis "
                "step.",
                "The next step uses an Ensembl gene ID; look up the supplied "
                "gene symbol {value}.",
                "Look up {value} as a human gene symbol so the analysis can "
                "continue with its Ensembl gene ID.",
                "Convert the input gene symbol {value} to a human Ensembl gene "
                "ID for the following analysis step.",
                "The analysis needs to continue from the supplied symbol "
                "{value}; resolve it to a human Ensembl gene ID.",
                "Before continuing the analysis, find the Ensembl gene ID "
                "associated with the human gene symbol {value}.",
                "A downstream operation requires an Ensembl gene ID; obtain it "
                "from the input gene symbol {value}.",
                "Use the current gene symbol {value} to obtain the Ensembl gene "
                "ID needed by the subsequent operation.",
            ),
        },
        "gene_id": {
            "direct": (
                "What is the human gene symbol for {value}?",
                "Which human gene symbol belongs to the Ensembl gene ID {value}?",
                "Look up the human Ensembl gene ID {value} and report its gene symbol.",
                "For the input Ensembl gene ID {value}, return the matching human gene symbol.",
                "Resolve the supplied human Ensembl gene ID {value} to its gene symbol.",
                "Find the human gene symbol associated with the input Ensembl ID {value}.",
                "A human gene symbol is requested for the supplied Ensembl gene ID {value}.",
                "Return the human gene symbol after checking the Ensembl gene ID {value}.",
            ),
            "step": (
                "Give the symbol for {value} before the next analysis step.",
                "The next step uses a human gene symbol; look up the supplied "
                "Ensembl gene ID {value}.",
                "Look up {value} as a human Ensembl gene ID so the analysis can "
                "continue with its gene symbol.",
                "Convert the input Ensembl gene ID {value} to a human gene "
                "symbol for the following analysis step.",
                "The analysis needs to continue from the supplied Ensembl gene "
                "ID {value}; resolve it to a human gene symbol.",
                "Before continuing the analysis, find the gene symbol associated "
                "with the human Ensembl gene ID {value}.",
                "A downstream operation requires a human gene symbol; obtain it "
                "from the input Ensembl gene ID {value}.",
                "Use the current Ensembl gene ID {value} to obtain the human gene "
                "symbol needed by the subsequent operation.",
            ),
        },
    },
    "validation": {
        "symbol": {
            "direct": (
                "Give the human Ensembl gene ID for {value}.",
                "Report the Ensembl gene identifier assigned to human gene symbol {value}.",
                "Using {value} as the input symbol, find its human Ensembl gene ID.",
                "I have the human gene symbol {value}; what is its Ensembl gene ID?",
            ),
            "step": (
                "Before the analysis continues, resolve {value} as a human "
                "Ensembl gene ID.",
                "Resolve the input gene symbol {value}; its Ensembl gene ID is "
                "needed by the next stage.",
                "The following analysis stage expects an Ensembl gene ID, so "
                "look up the human gene symbol {value}.",
                "Find the Ensembl gene ID for the supplied symbol {value} and "
                "then continue the workflow.",
            ),
        },
        "gene_id": {
            "direct": (
                "Give the human gene symbol for {value}.",
                "Report the gene symbol assigned to human Ensembl gene ID {value}.",
                "Using {value} as the input Ensembl ID, find its human gene symbol.",
                "I have the human Ensembl gene ID {value}; what is its gene symbol?",
            ),
            "step": (
                "Before the analysis continues, resolve Ensembl gene ID "
                "{value} as a human gene symbol.",
                "Resolve the input Ensembl gene ID {value}; its human gene "
                "symbol is needed by the next stage.",
                "The following analysis stage expects a human gene symbol, so "
                "look up the Ensembl gene ID {value}.",
                "Find the human gene symbol for the supplied Ensembl ID {value} "
                "and then continue the workflow.",
            ),
        },
    },
    "test": {
        "symbol": {
            "direct": ("Which human Ensembl gene ID matches {value}?",),
            "step": (
                "The next analysis operation needs a human Ensembl gene ID. "
                "Resolve the current symbol {value}.",
            ),
        },
        "gene_id": {
            "direct": ("Which human gene symbol matches {value}?",),
            "step": (
                "The next analysis operation needs a human gene symbol. "
                "Resolve the current Ensembl gene ID {value}.",
            ),
        },
    },
}


class CorpusBuildError(ValueError):
    """Report an invalid source pool or corpus invariant."""


@dataclass(frozen=True, slots=True)
class IdentityFact:
    """One source fact before prompt rendering or tool execution."""

    fact_id: str
    split_unit: str
    kind: str
    value: str
    candidate_count: int = 0

    @property
    def tool_name(self) -> str:
        return (
            SYMBOL_TOOL
            if self.kind in {RESOLVED_SYMBOL, AMBIGUOUS_SYMBOL, HISTORICAL_SYMBOL}
            else GENE_ID_TOOL
        )


@dataclass(frozen=True, slots=True)
class IdentityComponent:
    """One connected current ID/symbol split unit."""

    split_unit: str
    gene_ids: tuple[str, ...]
    gene_symbols: tuple[str, ...]
    is_ambiguous: bool
    facts: tuple[IdentityFact, ...]


@dataclass(frozen=True, slots=True)
class SupplementalEntry:
    """One absent symbol or retired ID and any linked current components."""

    key: str
    kind: str
    value: str
    linked_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RowRendering:
    """One deterministic prompt rendering of an identity fact."""

    fact: IdentityFact
    framing: str
    template_index: int
    rendering_index: int
    input_variant: str


class EvidenceSequence:
    """Allocate deterministic, collision-free six-character evidence IDs."""

    def __init__(self) -> None:
        self._next = 0

    def __call__(self, channel: str) -> str:
        if channel not in {"registry", "graph"}:
            raise CorpusBuildError(f"unknown evidence channel {channel!r}")
        value = self._next
        self._next += 1
        suffix = _base36(value).rjust(6, "0")
        if len(suffix) != 6:
            raise CorpusBuildError("the evidence-ID namespace is exhausted")
        prefix = "ev_reg_" if channel == "registry" else "ev_mem_"
        return prefix + suffix


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    digits = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits))


def _canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_s0_v4_settings(settings: Mapping[str, Any]) -> None:
    expected_rows = {"train": 16384, "validation": 622, "test": 600}
    if settings.get("dataset_id") != "world_model_v4_s0_human_identifiers":
        raise CorpusBuildError("the S0-v4 dataset ID changed")
    if settings.get("rows") != expected_rows:
        raise CorpusBuildError("the S0-v4 row counts changed")
    if settings.get("output_directory") != "data/world_model_v4/corpora/s0":
        raise CorpusBuildError("the S0-v4 output directory changed")
    if settings.get("input_casefold_share") != 0.125:
        raise CorpusBuildError("the S0-v4 case-fold share must be 0.125")
    if settings.get("max_length") != 4096:
        raise CorpusBuildError("the S0-v4 max length must be 4096")
    if settings.get("train_prompt_template_families") != 8:
        raise CorpusBuildError("S0-v4 must have eight train prompt pairs")
    if settings.get("validation_prompt_template_families") != 4:
        raise CorpusBuildError("S0-v4 must have four validation prompt pairs")
    if settings.get("maximum_train_ambiguous_candidates") != 49:
        raise CorpusBuildError("S0-v4 must support 49 ambiguity candidates")

    train = settings.get("train")
    if not isinstance(train, Mapping):
        raise CorpusBuildError("the S0-v4 train allocation must be a mapping")
    expected_train = {
        "registry_symbol_rows": 5952,
        "resolved_symbol_rows": 5120,
        "ambiguous_symbol_rows": 832,
        "named_gene_id_rows": 4608,
        "unnamed_gene_id_rows": 5056,
        "historical_symbol_rows": 384,
        "retired_gene_id_rows": 384,
        "unique_fact_counts": {
            RESOLVED_SYMBOL: 2560,
            AMBIGUOUS_SYMBOL: 385,
            NAMED_GENE_ID: 2304,
            UNNAMED_GENE_ID: 2528,
            HISTORICAL_SYMBOL: 192,
            RETIRED_GENE_ID: 192,
        },
    }
    if dict(train) != expected_train:
        raise CorpusBuildError("the S0-v4 train allocation changed")
    evaluation = settings.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise CorpusBuildError("S0-v4 did not inherit the evaluation allocation")
    for split in ("validation", "test"):
        split_settings = evaluation.get(split)
        if not isinstance(split_settings, Mapping):
            raise CorpusBuildError(f"S0-v4 has no {split} allocation")
        count = sum(
            int(split_settings.get(key, 0))
            for key in (
                "registry_rows",
                "historical_symbol_rows",
                "retired_gene_id_rows",
            )
        )
        if count != expected_rows[split]:
            raise CorpusBuildError(f"S0-v4 {split} rows do not reconcile")


def load_s0_v4_settings(
    contract: Mapping[str, Any],
    overlay_path: Path = S0_V4_OVERLAY_PATH,
    *,
    base_contract_path: Path = CONTRACT_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the S0-v4 overlay without mutating the base contract."""

    overlay_path = overlay_path.resolve()
    base_contract_path = base_contract_path.resolve()
    try:
        payload = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CorpusBuildError(f"cannot load S0-v4 overlay {overlay_path}: {error}") from error
    if not isinstance(payload, dict):
        raise CorpusBuildError("the S0-v4 overlay root must be a mapping")
    if list(payload) != [
        "schema_version",
        "base_contract",
        "metadata_optional_additions",
        "family_overrides",
    ]:
        raise CorpusBuildError("the S0-v4 overlay keys changed")
    if payload["schema_version"] != S0_V4_OVERLAY_SCHEMA_VERSION:
        raise CorpusBuildError("the S0-v4 overlay schema changed")

    base_identity = payload["base_contract"]
    if not isinstance(base_identity, Mapping) or list(base_identity) != [
        "path",
        "sha256",
    ]:
        raise CorpusBuildError("the S0-v4 base-contract identity is invalid")
    declared_path = (REPOSITORY_ROOT / str(base_identity["path"])).resolve()
    if declared_path != base_contract_path:
        raise CorpusBuildError(
            "the S0-v4 overlay names a different shared contract"
        )
    actual_contract_hash = file_sha256(base_contract_path)
    if base_identity["sha256"] != actual_contract_hash:
        raise CorpusBuildError(
            "the shared contract does not match the S0-v4 overlay pin"
        )

    additions = payload["metadata_optional_additions"]
    if additions != list(S0_V4_METADATA_ADDITIONS):
        raise CorpusBuildError("the S0-v4 additive metadata contract changed")
    base_optional = contract.get("rows", {}).get("metadata", {}).get("optional")
    if not isinstance(base_optional, list) or set(additions) & set(base_optional):
        raise CorpusBuildError("the S0-v4 metadata additions are not additive")

    overrides = payload["family_overrides"]
    if not isinstance(overrides, Mapping):
        raise CorpusBuildError("the S0-v4 family overrides must be a mapping")
    base_settings = contract.get("curriculum", {}).get("families", {}).get("S0")
    if not isinstance(base_settings, Mapping):
        raise CorpusBuildError("the shared contract has no S0 family")
    if base_settings.get("dataset_id") != "world_model_v3_s0_human_identifiers":
        raise CorpusBuildError("the S0-v4 overlay requires the legacy S0 base")
    settings = copy.deepcopy(dict(base_settings))
    settings.update(copy.deepcopy(dict(overrides)))
    _validate_s0_v4_settings(settings)
    return settings, payload


def _supplemental_fact_id(kind: str, value: str) -> str:
    return "fact_" + _stable_sha256({"kind": kind, "value": value})[:24]


def _registry_fact_id(
    kind: str,
    value: str,
    *,
    candidate_ids: Sequence[str] = (),
    gene_symbols: Sequence[str] = (),
) -> str:
    """Keep the legacy fact hash so the pinned train selection stays stable."""

    if kind == RESOLVED_SYMBOL:
        family = "human_symbol_to_ensembl"
        inputs = {"gene_symbol": value}
        answer = {
            "status": "resolved",
            "gene_id": candidate_ids[0],
            "gene_symbol": value,
        }
    elif kind == AMBIGUOUS_SYMBOL:
        family = "human_ambiguous_symbol"
        inputs = {"gene_symbol": value}
        answer = {
            "status": "ambiguous",
            "gene_symbol": value,
            "candidate_gene_ids": list(candidate_ids),
            "action": "defer",
        }
    else:
        family = "human_ensembl_to_symbol"
        inputs = {"gene_id": value}
        answer = {
            "status": "resolved",
            "gene_id": value,
            "gene_symbols": list(gene_symbols),
        }
    return "fact_" + _stable_sha256(
        {"family": family, "input": inputs, "answer": answer}
    )[:24]


def _find(parent: dict[str, str], value: str) -> str:
    parent.setdefault(value, value)
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        first, second = sorted((left_root, right_root))
        parent[second] = first


def build_identity_components(
    genes_by_id: Mapping[str, Sequence[str | None]],
) -> tuple[list[IdentityComponent], dict[str, str]]:
    """Build case-insensitive ID/symbol components and their natural facts."""

    parent: dict[str, str] = {}
    ids_by_symbol: dict[str, set[str]] = defaultdict(set)
    for gene_id, values in genes_by_id.items():
        _find(parent, gene_id)
        symbol = values[0]
        if symbol is None:
            continue
        key = str(symbol).casefold()
        ids_by_symbol[key].add(gene_id)

    for gene_ids in ids_by_symbol.values():
        first, *rest = sorted(gene_ids)
        for gene_id in rest:
            _union(parent, first, gene_id)

    genes_by_root: dict[str, list[str]] = defaultdict(list)
    for gene_id in sorted(genes_by_id):
        genes_by_root[_find(parent, gene_id)].append(gene_id)

    components: list[IdentityComponent] = []
    component_by_gene: dict[str, str] = {}
    kind_order = {
        RESOLVED_SYMBOL: 0,
        AMBIGUOUS_SYMBOL: 1,
        NAMED_GENE_ID: 2,
        UNNAMED_GENE_ID: 3,
    }
    for gene_ids in genes_by_root.values():
        symbols = sorted(
            {
                str(genes_by_id[gene_id][0])
                for gene_id in gene_ids
                if genes_by_id[gene_id][0] is not None
            }
        )
        split_unit = "group_" + _stable_sha256(
            {"gene_ids": gene_ids, "gene_symbols": symbols}
        )[:24]
        for gene_id in gene_ids:
            component_by_gene[gene_id] = split_unit

        facts: list[IdentityFact] = []
        ambiguous = False
        for symbol in symbols:
            candidate_ids = ids_by_symbol[symbol.casefold()]
            candidate_count = len(candidate_ids)
            kind = RESOLVED_SYMBOL if candidate_count == 1 else AMBIGUOUS_SYMBOL
            ambiguous |= candidate_count > 1
            facts.append(
                IdentityFact(
                    fact_id=_registry_fact_id(
                        kind,
                        symbol,
                        candidate_ids=tuple(sorted(candidate_ids)),
                    ),
                    split_unit=split_unit,
                    kind=kind,
                    value=symbol,
                    candidate_count=candidate_count,
                )
            )
        for gene_id in gene_ids:
            kind = (
                NAMED_GENE_ID
                if genes_by_id[gene_id][0] is not None
                else UNNAMED_GENE_ID
            )
            facts.append(
                IdentityFact(
                    fact_id=_registry_fact_id(
                        kind,
                        gene_id,
                        gene_symbols=(
                            ()
                            if genes_by_id[gene_id][0] is None
                            else (str(genes_by_id[gene_id][0]),)
                        ),
                    ),
                    split_unit=split_unit,
                    kind=kind,
                    value=gene_id,
                )
            )
        facts.sort(key=lambda fact: (kind_order[fact.kind], fact.fact_id))
        components.append(
            IdentityComponent(
                split_unit=split_unit,
                gene_ids=tuple(gene_ids),
                gene_symbols=tuple(symbols),
                is_ambiguous=ambiguous,
                facts=tuple(facts),
            )
        )
    components.sort(key=lambda component: component.split_unit)
    return components, component_by_gene


def assign_evaluation_components(
    components: Sequence[IdentityComponent],
    settings: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, str]:
    """Assign deterministic, disjoint components to validation and test."""

    pools = {
        "unambiguous": [item for item in components if not item.is_ambiguous],
        "ambiguous": [item for item in components if item.is_ambiguous],
    }
    assignments: dict[str, str] = {}
    for split in ("validation", "test"):
        for component_type in ("unambiguous", "ambiguous"):
            count = int(settings[split][f"{component_type}_component_count"])
            ordered = sorted(
                pools[component_type],
                key=lambda item: _stable_sha256(
                    {
                        "assignment_seed": seed,
                        "split": split,
                        "fact_group_id": item.split_unit,
                    }
                ),
            )
            if len(ordered) < count:
                raise CorpusBuildError(
                    f"{component_type} components cannot fill {split}"
                )
            selected = ordered[:count]
            pools[component_type] = ordered[count:]
            assignments.update({item.split_unit: split for item in selected})
    return assignments


def _ordered_facts(
    facts: Iterable[IdentityFact], *, seed: int, pool: str
) -> list[IdentityFact]:
    return sorted(
        facts,
        key=lambda fact: _stable_sha256(
            {"assignment_seed": seed, "pool": pool, "fact_id": fact.fact_id}
        ),
    )


def select_registry_facts(
    components: Sequence[IdentityComponent],
    assignments: Mapping[str, str],
    settings: Mapping[str, Any],
    *,
    seed: int,
    maximum_ambiguous_candidates: int,
) -> dict[str, list[IdentityFact]]:
    """Select the configured registry facts without crossing split units."""

    available = [
        fact
        for component in components
        if component.split_unit not in assignments
        for fact in component.facts
    ]
    ambiguous = [
        fact
        for fact in available
        if fact.kind == AMBIGUOUS_SYMBOL
        and fact.candidate_count <= maximum_ambiguous_candidates
    ]
    unique_fact_counts = settings["train"]["unique_fact_counts"]
    expected_ambiguous = int(unique_fact_counts[AMBIGUOUS_SYMBOL])
    if len(ambiguous) != expected_ambiguous:
        raise CorpusBuildError(
            f"eligible train ambiguities are {len(ambiguous)}, "
            f"expected {expected_ambiguous}"
        )
    resolved = _ordered_facts(
        (fact for fact in available if fact.kind == RESOLVED_SYMBOL),
        seed=seed,
        pool="resolved_symbols",
    )[: int(unique_fact_counts[RESOLVED_SYMBOL])]
    named = _ordered_facts(
        (fact for fact in available if fact.kind == NAMED_GENE_ID),
        seed=seed,
        pool="named_gene_ids",
    )[: int(unique_fact_counts[NAMED_GENE_ID])]
    unnamed = _ordered_facts(
        (fact for fact in available if fact.kind == UNNAMED_GENE_ID),
        seed=seed,
        pool="unnamed_gene_ids",
    )[: int(unique_fact_counts[UNNAMED_GENE_ID])]
    train = ambiguous + resolved + named + unnamed
    expected_registry_facts = sum(
        int(unique_fact_counts[kind])
        for kind in (
            RESOLVED_SYMBOL,
            AMBIGUOUS_SYMBOL,
            NAMED_GENE_ID,
            UNNAMED_GENE_ID,
        )
    )
    if len(train) != expected_registry_facts:
        raise CorpusBuildError("the registry cannot fill the S0 train allocation")

    result = {"train": _ordered_facts(train, seed=seed, pool="train_registry")}
    for split in ("validation", "test"):
        facts = sorted(
            (
                fact
                for component in components
                if assignments.get(component.split_unit) == split
                for fact in component.facts
            ),
            key=lambda fact: fact.fact_id,
        )
        expected = int(settings["evaluation"][split]["registry_rows"])
        if len(facts) != expected:
            raise CorpusBuildError(
                f"{split} registry rows are {len(facts)}, expected {expected}"
            )
        result[split] = facts
    return result


def _supplemental_entries(
    payload: Mapping[str, Any],
    component_by_gene: Mapping[str, str],
) -> dict[str, list[SupplementalEntry]]:
    symbols_by_key: dict[str, list[Sequence[Any]]] = defaultdict(list)
    for row in payload["historical_symbols"]:
        symbols_by_key[str(row[0]).casefold()].append(row)
    historical: list[SupplementalEntry] = []
    for key, rows in sorted(symbols_by_key.items()):
        chosen = min(rows, key=lambda row: _stable_sha256(row[0]))
        linked = sorted(
            {
                component_by_gene[gene_id]
                for row in rows
                for gene_id in row[2]
                if gene_id in component_by_gene
            }
        )
        historical.append(
            SupplementalEntry(
                key=f"historical:{key}",
                kind=HISTORICAL_SYMBOL,
                value=str(chosen[0]),
                linked_components=tuple(linked),
            )
        )

    retired = [
        SupplementalEntry(
            key=f"retired:{row[0]}",
            kind=RETIRED_GENE_ID,
            value=str(row[0]),
            linked_components=tuple(
                sorted(
                    {
                        component_by_gene[gene_id]
                        for gene_id in row[1]
                        if gene_id in component_by_gene
                    }
                )
            ),
        )
        for row in payload["retired_genes"]
    ]
    return {HISTORICAL_SYMBOL: historical, RETIRED_GENE_ID: retired}


def select_supplemental_facts(
    payload: Mapping[str, Any],
    component_by_gene: Mapping[str, str],
    assignments: Mapping[str, str],
    settings: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, list[IdentityFact]]:
    """Assign absent symbols and retired IDs without identity leakage."""

    pools = _supplemental_entries(payload, component_by_gene)
    selected: dict[str, list[IdentityFact]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    used: set[str] = set()
    for split in ("validation", "test", "train"):
        split_settings = (
            settings["train"]
            if split == "train"
            else settings["evaluation"][split]
        )
        for kind, count_key in (
            (HISTORICAL_SYMBOL, "historical_symbol_rows"),
            (RETIRED_GENE_ID, "retired_gene_id_rows"),
        ):
            if split == "train" and "unique_fact_counts" in split_settings:
                count = int(split_settings["unique_fact_counts"][kind])
            else:
                count = int(split_settings[count_key])

            def compatible(entry: SupplementalEntry) -> bool:
                if entry.key in used:
                    return False
                linked_splits = {
                    assignments.get(component) for component in entry.linked_components
                }
                if not entry.linked_components:
                    return True
                if split == "train":
                    return linked_splits == {None}
                return linked_splits == {split}

            ordered = sorted(
                (entry for entry in pools[kind] if compatible(entry)),
                key=lambda entry: _stable_sha256(
                    {
                        "assignment_seed": seed,
                        "pool": kind,
                        "split": split,
                        "key": entry.key,
                    }
                ),
            )
            if len(ordered) < count:
                raise CorpusBuildError(f"{kind} cannot fill {split}")
            for entry in ordered[:count]:
                used.add(entry.key)
                split_unit = (
                    entry.linked_components[0]
                    if len(entry.linked_components) == 1
                    else "group_" + _stable_sha256(
                        {
                            "kind": kind,
                            "key": entry.key,
                            "linked_components": entry.linked_components,
                        }
                    )[:24]
                )
                selected[split].append(
                    IdentityFact(
                        fact_id=_supplemental_fact_id(kind, entry.value),
                        split_unit=split_unit,
                        kind=kind,
                        value=entry.value,
                    )
                )
    return selected


def _input_value(
    fact: IdentityFact,
    *,
    seed: int,
    input_variant: str | None = None,
    rendering_index: int = 0,
) -> str:
    if input_variant is None:
        perturb = int(
            _stable_sha256(
                {
                    "assignment_seed": seed,
                    "pool": "input_case",
                    "fact_id": fact.fact_id,
                    "rendering_index": rendering_index,
                }
            )[:16],
            16,
        ) % 8 == 0
        input_variant = "casefold" if perturb else "canonical"
    if input_variant not in {"canonical", "casefold"}:
        raise CorpusBuildError(f"unknown input variant {input_variant!r}")
    return fact.value.lower() if input_variant == "casefold" else fact.value


def _source_trace(fact: IdentityFact, input_value: str) -> tuple[str, int]:
    if fact.kind == RESOLVED_SYMBOL:
        form = 3 if input_value != fact.value else 1
    else:
        form = {
            NAMED_GENE_ID: 2,
            UNNAMED_GENE_ID: 4,
            AMBIGUOUS_SYMBOL: 5,
            HISTORICAL_SYMBOL: 6,
            RETIRED_GENE_ID: 7,
        }[fact.kind]
    return f"S0.{form}", form


def _framing(fact: IdentityFact, *, seed: int) -> str:
    value = int(
        _stable_sha256(
            {"assignment_seed": seed, "pool": "framing", "fact_id": fact.fact_id}
        )[:16],
        16,
    )
    return "direct" if value % 2 == 0 else "step"


def _prompt_input_kind(fact: IdentityFact) -> str:
    return "symbol" if fact.tool_name == SYMBOL_TOOL else "gene_id"


def prompt_template_id(
    fact: IdentityFact, split: str, template_index: int
) -> str:
    """Return the stable ID shared by one direct/step prompt pair."""

    return f"s0.{split}.{_prompt_input_kind(fact)}.t{template_index:02d}"


def _default_template_index(
    fact: IdentityFact, *, split: str, seed: int
) -> int:
    input_kind = _prompt_input_kind(fact)
    count = len(PROMPT_TEMPLATES[split][input_kind]["direct"])
    value = int(
        _stable_sha256(
            {
                "assignment_seed": seed,
                "pool": "prompt_template",
                "split": split,
                "fact_id": fact.fact_id,
            }
        )[:16],
        16,
    )
    return value % count


def _question(
    fact: IdentityFact,
    value: str,
    split: str,
    framing: str,
    template_index: int = 0,
) -> str:
    input_kind = "symbol" if fact.tool_name == SYMBOL_TOOL else "gene_id"
    try:
        template = PROMPT_TEMPLATES[split][input_kind][framing][template_index]
    except (KeyError, IndexError) as error:
        raise CorpusBuildError(
            f"no {split}/{input_kind}/{framing} prompt template "
            f"{template_index}"
        ) from error
    return template.format(value=value)


def build_train_renderings(
    facts: Sequence[IdentityFact],
    settings: Mapping[str, Any],
    *,
    seed: int,
) -> list[RowRendering]:
    """Expand selected facts into balanced direct/step training rows."""

    rows_by_kind = {
        RESOLVED_SYMBOL: "resolved_symbol_rows",
        AMBIGUOUS_SYMBOL: "ambiguous_symbol_rows",
        NAMED_GENE_ID: "named_gene_id_rows",
        UNNAMED_GENE_ID: "unnamed_gene_id_rows",
        HISTORICAL_SYMBOL: "historical_symbol_rows",
        RETIRED_GENE_ID: "retired_gene_id_rows",
    }
    facts_by_kind: dict[str, list[IdentityFact]] = defaultdict(list)
    for fact in facts:
        facts_by_kind[fact.kind].append(fact)

    pair_records: list[tuple[IdentityFact, int, str]] = []
    template_count = int(settings["train_prompt_template_families"])
    if template_count != len(PROMPT_TEMPLATES["train"]["symbol"]["direct"]):
        raise CorpusBuildError("the train prompt-template count changed")
    for input_kind in ("symbol", "gene_id"):
        for framing in ("direct", "step"):
            if len(PROMPT_TEMPLATES["train"][input_kind][framing]) != template_count:
                raise CorpusBuildError("train prompt pairs have unequal lengths")

    for kind, row_key in rows_by_kind.items():
        selected = sorted(facts_by_kind[kind], key=lambda fact: fact.fact_id)
        target_rows = int(settings["train"][row_key])
        if target_rows % 2:
            raise CorpusBuildError(f"{kind} cannot be direct/step paired")
        target_pairs = target_rows // 2
        if len(selected) > target_pairs:
            raise CorpusBuildError(f"too many unique {kind} facts for its rows")

        pairs: dict[tuple[str, int], IdentityFact] = {}
        for fact in selected:
            template_index = _default_template_index(
                fact, split="train", seed=seed
            )
            pairs[(fact.fact_id, template_index)] = fact

        if kind == AMBIGUOUS_SYMBOL:
            for fact in selected:
                if fact.candidate_count > 20:
                    for template_index in range(template_count):
                        pairs[(fact.fact_id, template_index)] = fact

        candidates = sorted(
            (
                (fact, template_index)
                for fact in selected
                for template_index in range(template_count)
                if (fact.fact_id, template_index) not in pairs
            ),
            key=lambda item: _stable_sha256(
                {
                    "assignment_seed": seed,
                    "pool": "extra_train_rendering",
                    "kind": kind,
                    "fact_id": item[0].fact_id,
                    "template_index": item[1],
                }
            ),
        )
        needed = target_pairs - len(pairs)
        if needed < 0 or len(candidates) < needed:
            raise CorpusBuildError(f"cannot fill the paired {kind} allocation")
        for fact, template_index in candidates[:needed]:
            pairs[(fact.fact_id, template_index)] = fact
        if len(pairs) != target_pairs:
            raise CorpusBuildError(f"paired {kind} allocation is incomplete")

        ordered_pairs = sorted(
            ((fact, template_index) for (_, template_index), fact in pairs.items()),
            key=lambda item: _stable_sha256(
                {
                    "assignment_seed": seed,
                    "pool": "train_rendering_order",
                    "kind": kind,
                    "fact_id": item[0].fact_id,
                    "template_index": item[1],
                }
            ),
        )
        casefold_share = float(settings["input_casefold_share"])
        target_casefold_pairs = int(target_pairs * casefold_share)
        if target_casefold_pairs / target_pairs != casefold_share:
            raise CorpusBuildError(f"{kind} cannot represent the case-fold share")
        casefold_candidates = sorted(
            (
                (fact, template_index)
                for fact, template_index in ordered_pairs
                if fact.value.lower() != fact.value
            ),
            key=lambda item: _stable_sha256(
                {
                    "assignment_seed": seed,
                    "pool": "train_input_case",
                    "kind": kind,
                    "fact_id": item[0].fact_id,
                    "template_index": item[1],
                }
            ),
        )
        if len(casefold_candidates) < target_casefold_pairs:
            raise CorpusBuildError(f"{kind} cannot fill the case-fold share")
        casefold_keys = {
            (fact.fact_id, template_index)
            for fact, template_index in casefold_candidates[:target_casefold_pairs]
        }
        for fact, template_index in ordered_pairs:
            key = (fact.fact_id, template_index)
            input_variant = "casefold" if key in casefold_keys else "canonical"
            for framing in ("direct", "step"):
                pair_records.append(
                    RowRendering(
                        fact=fact,
                        framing=framing,
                        template_index=template_index,
                        rendering_index=template_index,
                        input_variant=input_variant,
                    )
                )
    return pair_records


def build_evaluation_renderings(
    facts: Sequence[IdentityFact],
    split: str,
    *,
    seed: int,
) -> list[RowRendering]:
    """Render one deterministic held-out prompt for each evaluation fact."""

    if split not in {"validation", "test"}:
        raise CorpusBuildError(f"unknown evaluation split {split!r}")
    renderings = []
    for fact in facts:
        framing = _framing(fact, seed=seed)
        template_index = _default_template_index(fact, split=split, seed=seed)
        rendering_index = template_index
        input_variant = (
            "casefold"
            if _input_value(
                fact,
                seed=seed,
                rendering_index=rendering_index,
            )
            != fact.value
            else "canonical"
        )
        renderings.append(
            RowRendering(
                fact=fact,
                framing=framing,
                template_index=template_index,
                rendering_index=rendering_index,
                input_variant=input_variant,
            )
        )
    return renderings


def _pre_call_thinking(fact: IdentityFact, value: str) -> str:
    if fact.tool_name == SYMBOL_TOOL:
        return (
            f"{value} is the supplied gene symbol, so I will call "
            f'{SYMBOL_TOOL} with gene_symbol="{value}" to resolve it to an '
            "Ensembl gene ID."
        )
    return (
        f"{value} is the supplied Ensembl gene ID, so I will call "
        f'{GENE_ID_TOOL} with gene_id="{value}" to resolve its gene symbol.'
    )


def _resolved_decision_thinking(framing: str) -> str:
    if framing == "step":
        return "The next analysis step can use this result, so I should continue."
    if framing == "direct":
        return "This completes the direct lookup request, so I should stop."
    raise CorpusBuildError(f"unknown prompt framing {framing!r}")


def _answer(
    fact: IdentityFact,
    value: str,
    tool_result: Mapping[str, Any],
    *,
    release: str,
    framing: str,
) -> tuple[str, dict[str, Any]]:
    status = str(tool_result["status"])
    provenance = tool_result["provenance"]
    blocked_reason: str | None = None

    if fact.tool_name == SYMBOL_TOOL and status == "resolved":
        gene_id = str(tool_result["gene_id"])
        symbol = str(tool_result["gene_symbol"])
        result = {"gene_id": gene_id, "gene_symbol": symbol}
        thinking = (
            f"The registry gives one ID for {symbol}, so I can use the "
            f"canonical mapping. {_resolved_decision_thinking(framing)}"
        )
        summary = (
            f"{value} is <GENE:{gene_id}|{symbol}> in Ensembl release {release}."
        )
        decision = "continue" if framing == "step" else "stop"
    elif fact.tool_name == GENE_ID_TOOL and status == "resolved":
        gene_id = str(tool_result["gene_id"])
        symbol = tool_result["gene_symbol"]
        result = {"gene_id": gene_id, "gene_symbol": symbol}
        if symbol is None:
            thinking = (
                "The tool returned gene_symbol as null. The ID is valid, but I "
                "must preserve that null rather than invent a symbol. "
                f"{_resolved_decision_thinking(framing)}"
            )
            summary = (
                f"<GENE:{gene_id}> is a human gene in Ensembl release {release} "
                "with no known symbol."
            )
        else:
            thinking = (
                "The registry gives one symbol for this ID, so I can use the "
                f"canonical mapping. {_resolved_decision_thinking(framing)}"
            )
            summary = (
                f"<GENE:{gene_id}|{symbol}> has the symbol {symbol} in Ensembl "
                f"release {release}."
            )
        decision = "continue" if framing == "step" else "stop"
    elif status == "ambiguous":
        symbol = str(tool_result["gene_symbol"])
        candidate_ids = list(tool_result["candidate_gene_ids"])
        details = dict(tool_result["candidate_details"])
        result = {
            "gene_symbol": symbol,
            "candidate_gene_ids": candidate_ids,
            "candidate_details": details,
        }
        thinking = (
            f"The registry maps {symbol} to {len(candidate_ids)} IDs and cannot "
            "choose one. I will keep every candidate and defer because the "
            "task supplies no disambiguating evidence."
        )
        blocked_reason = (
            f"The symbol {symbol} matches {len(candidate_ids)} genes in Ensembl "
            f"release {release}, and nothing in the task chooses one."
        )
        anchors = ", ".join(
            f"<GENE:{gene_id}|{symbol}>" for gene_id in candidate_ids
        )
        summary = (
            f"The symbol {symbol} matches {len(candidate_ids)} genes in Ensembl "
            f"release {release}: {anchors}. Give one Ensembl gene ID to continue."
        )
        decision = "defer"
    elif fact.tool_name == SYMBOL_TOOL and status == "not_found":
        symbol = str(tool_result["gene_symbol"])
        result = {"gene_symbol": symbol}
        thinking = (
            f"The registry has no match for {symbol}. I will not replace it "
            "with a name from memory, and I must defer until the input changes."
        )
        blocked_reason = (
            f"The registry has no gene with the symbol {symbol} in Ensembl "
            f"release {release}."
        )
        summary = (
            f"{symbol} has no match in Ensembl release {release}. Give a current "
            "symbol or an Ensembl gene ID to continue."
        )
        decision = "defer"
    elif fact.tool_name == GENE_ID_TOOL and status == "not_found":
        gene_id = str(tool_result["gene_id"])
        result = {"gene_id": gene_id}
        thinking = (
            "The registry has no gene with this ID in this release. I will not "
            "replace it with an ID or a symbol from memory, and I must defer "
            "until the input changes."
        )
        blocked_reason = (
            f"The registry has no gene with the ID {gene_id} in Ensembl release "
            f"{release}."
        )
        summary = (
            f"{gene_id} is not a gene ID in Ensembl release {release}. Give a "
            "current Ensembl gene ID to continue."
        )
        decision = "defer"
    else:
        raise CorpusBuildError(
            f"unexpected S0 result {status!r} for {fact.kind}"
        )

    answer = {
        "status": status,
        "provenance": provenance,
        "result": result,
        "continuation_decision": decision,
        "blocked_reason": blocked_reason,
        "summary": summary,
    }
    return thinking, answer


def _build_row(
    fact: IdentityFact,
    *,
    split: str,
    seed: int,
    runtime: WorldModelToolRuntime,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]],
    framing: str | None = None,
    template_index: int | None = None,
    rendering_index: int | None = None,
    input_variant: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if framing is None:
        framing = _framing(fact, seed=seed)
    if framing not in {"direct", "step"}:
        raise CorpusBuildError(f"unknown prompt framing {framing!r}")
    if template_index is None:
        template_index = _default_template_index(fact, split=split, seed=seed)
    if rendering_index is None:
        rendering_index = template_index
    if input_variant is None:
        value = _input_value(
            fact,
            seed=seed,
            rendering_index=rendering_index,
        )
        input_variant = "casefold" if value != fact.value else "canonical"
    else:
        value = _input_value(
            fact,
            seed=seed,
            input_variant=input_variant,
            rendering_index=rendering_index,
        )
    source_trace, form = _source_trace(fact, value)
    template_id = prompt_template_id(fact, split, template_index)
    arguments = (
        {"gene_symbol": value}
        if fact.tool_name == SYMBOL_TOOL
        else {"gene_id": value}
    )
    row_id = "wm3_s0_" + _stable_sha256(
        {
            "fact_id": fact.fact_id,
            "split": split,
            "schema_version": contract["runtime"]["row_schema_version"],
            "framing": framing,
            "prompt_template_id": template_id,
            "rendering_index": rendering_index,
            "input_variant": input_variant,
        }
    )[:24]
    call_result: ToolCallResult = runtime.execute(fact.tool_name, arguments)
    final_thinking, final_answer = _answer(
        fact,
        value,
        call_result.output,
        release=runtime.ensembl_release,
        framing=framing,
    )
    if call_result.output["status"] == "ambiguous":
        question_family = "human_ambiguous_symbol"
        blocked_reason_code = "ambiguous_symbol"
    elif fact.tool_name == SYMBOL_TOOL:
        question_family = "human_symbol_to_ensembl"
        blocked_reason_code = (
            "not_found" if call_result.output["status"] == "not_found" else None
        )
    else:
        question_family = "human_ensembl_to_symbol"
        blocked_reason_code = (
            "not_found" if call_result.output["status"] == "not_found" else None
        )

    row = {
        "row_id": row_id,
        "messages": [
            {"role": "system", "content": contract["runtime"]["system_prompt"]},
            {
                "role": "user",
                "content": _question(
                    fact,
                    value,
                    split,
                    framing,
                    template_index,
                ),
            },
            {
                "role": "assistant",
                "thinking": _pre_call_thinking(fact, value),
                "tool_call": {"name": fact.tool_name, "arguments": arguments},
            },
            {
                "role": "tool",
                "name": fact.tool_name,
                "content": call_result.output,
            },
            {
                "role": "assistant",
                "thinking": final_thinking,
                "content": final_answer,
            },
        ],
        "tools": tools,
        "metadata": {
            "book_mode": "tool_call",
            "step": "S0",
            "question_family": question_family,
            "form": form,
            "source_trace": source_trace,
            "split_unit": fact.split_unit,
            "blocked_reason_code": blocked_reason_code,
            "fact_kind": fact.kind,
            "framing": framing,
            "prompt_template_id": template_id,
            "rendering_index": rendering_index,
            "input_variant": input_variant,
            "schema_version": contract["runtime"]["row_schema_version"],
            "system_prompt_sha256": system_prompt_sha256(contract),
            "tool_contract_version": contract["runtime"][
                "tool_contract_version"
            ],
            "ensembl_release": runtime.ensembl_release,
            "identifier_registry_id": runtime.identifier_registry_id,
        },
    }
    if call_result.evidence is None:
        raise CorpusBuildError("an S0 source fact unexpectedly produced no evidence")
    evidence = {
        "evidence_id": call_result.evidence["evidence_id"],
        "row_id": row_id,
        "tool_name": fact.tool_name,
    }
    return row, evidence


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row, sort_keys=False) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validate_corpus(
    path: Path,
    *,
    metadata_optional_additions: Sequence[str],
) -> None:
    if tuple(metadata_optional_additions) != S0_V4_METADATA_ADDITIONS:
        raise CorpusBuildError("the S0-v4 validator received unknown metadata")
    split = path.stem
    sanitized_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{split}_base_contract_",
            suffix=".jsonl",
            dir=path.parent,
            delete=False,
        ) as sanitized:
            sanitized_path = Path(sanitized.name)
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError as error:
                        raise CorpusBuildError(
                            f"invalid JSON at {path}:{line_number}"
                        ) from error
                    metadata = row.get("metadata")
                    if not isinstance(metadata, dict):
                        raise CorpusBuildError(
                            f"missing metadata at {path}:{line_number}"
                        )
                    values = {
                        key: metadata.get(key)
                        for key in metadata_optional_additions
                    }
                    if values["fact_kind"] not in {
                        RESOLVED_SYMBOL,
                        AMBIGUOUS_SYMBOL,
                        NAMED_GENE_ID,
                        UNNAMED_GENE_ID,
                        HISTORICAL_SYMBOL,
                        RETIRED_GENE_ID,
                    }:
                        raise CorpusBuildError(
                            f"invalid fact_kind at {path}:{line_number}"
                        )
                    if values["framing"] not in {"direct", "step"}:
                        raise CorpusBuildError(
                            f"invalid framing at {path}:{line_number}"
                        )
                    template_id = values["prompt_template_id"]
                    if not isinstance(template_id, str) or not template_id.startswith(
                        f"s0.{split}."
                    ):
                        raise CorpusBuildError(
                            f"invalid prompt_template_id at {path}:{line_number}"
                        )
                    rendering_index = values["rendering_index"]
                    if (
                        isinstance(rendering_index, bool)
                        or not isinstance(rendering_index, int)
                        or rendering_index < 0
                    ):
                        raise CorpusBuildError(
                            f"invalid rendering_index at {path}:{line_number}"
                        )
                    if values["input_variant"] not in {"canonical", "casefold"}:
                        raise CorpusBuildError(
                            f"invalid input_variant at {path}:{line_number}"
                        )
                    for key in metadata_optional_additions:
                        del metadata[key]
                    sanitized.write(_canonical_json(row, sort_keys=False) + "\n")

        if sanitized_path is None:
            raise AssertionError("the sanitized S0 corpus was not created")
        validator = (
            REPOSITORY_ROOT
            / "agents"
            / "world_model_v3_curriculum"
            / "s1_kit"
            / "validate.py"
        )
        result = subprocess.run(
            [sys.executable, str(validator), str(sanitized_path)],
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
    finally:
        if sanitized_path is not None:
            sanitized_path.unlink(missing_ok=True)


def build_corpus(
    source_index_path: Path,
    output_directory: Path,
    *,
    contract: Mapping[str, Any],
    overlay_path: Path = S0_V4_OVERLAY_PATH,
    base_contract_path: Path = CONTRACT_PATH,
    validate: bool = True,
) -> dict[str, Any]:
    """Build, validate, and atomically publish the three S0 splits."""

    payload = check_source_index(source_index_path.resolve(), contract)
    settings, overlay = load_s0_v4_settings(
        contract,
        overlay_path,
        base_contract_path=base_contract_path,
    )
    overlay_path = overlay_path.resolve()
    base_contract_path = base_contract_path.resolve()
    seed = int(settings["assignment_seed"])
    components, component_by_gene = build_identity_components(
        payload["genes_by_id"]
    )
    assignments = assign_evaluation_components(
        components,
        settings["evaluation"],
        seed=seed,
    )
    registry_facts = select_registry_facts(
        components,
        assignments,
        settings,
        seed=seed,
        maximum_ambiguous_candidates=int(
            settings["maximum_train_ambiguous_candidates"]
        ),
    )
    supplemental = select_supplemental_facts(
        payload,
        component_by_gene,
        assignments,
        settings,
        seed=seed,
    )
    facts_by_split = {
        split: _ordered_facts(
            registry_facts[split] + supplemental[split],
            seed=seed,
            pool=f"{split}_output",
        )
        for split in ("train", "validation", "test")
    }
    expected_train_facts = sum(
        int(value) for value in settings["train"]["unique_fact_counts"].values()
    )
    if len(facts_by_split["train"]) != expected_train_facts:
        raise CorpusBuildError(
            f"train has {len(facts_by_split['train'])} facts, "
            f"expected {expected_train_facts}"
        )
    for split in ("validation", "test"):
        expected = int(settings["rows"][split])
        if len(facts_by_split[split]) != expected:
            raise CorpusBuildError(
                f"{split} has {len(facts_by_split[split])} facts, "
                f"expected {expected}"
            )

    renderings_by_split = {
        "train": build_train_renderings(
            facts_by_split["train"], settings, seed=seed
        ),
        "validation": build_evaluation_renderings(
            facts_by_split["validation"], "validation", seed=seed
        ),
        "test": build_evaluation_renderings(
            facts_by_split["test"], "test", seed=seed
        ),
    }
    for split, renderings in renderings_by_split.items():
        expected = int(settings["rows"][split])
        if len(renderings) != expected:
            raise CorpusBuildError(
                f"{split} has {len(renderings)} renderings, expected {expected}"
            )

    evidence_sequence = EvidenceSequence()
    runtime = WorldModelToolRuntime.from_source_index(
        source_index_path.resolve(),
        contract=contract,
        evidence_id_factory=evidence_sequence,
    )
    tools = tool_definitions(contract)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    evidence_calls: list[list[str]] = []
    for split in ("train", "validation", "test"):
        rows = []
        for rendering in renderings_by_split[split]:
            row, evidence = _build_row(
                rendering.fact,
                split=split,
                seed=seed,
                runtime=runtime,
                contract=contract,
                tools=tools,
                framing=rendering.framing,
                template_index=rendering.template_index,
                rendering_index=rendering.rendering_index,
                input_variant=rendering.input_variant,
            )
            rows.append(row)
            evidence_calls.append(
                [
                    evidence["evidence_id"],
                    evidence["row_id"],
                    evidence["tool_name"],
                ]
            )
        rows_by_split[split] = rows

    row_ids = [row["row_id"] for rows in rows_by_split.values() for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise CorpusBuildError("row IDs are not globally unique")
    split_units = {
        split: {row["metadata"]["split_unit"] for row in rows}
        for split, rows in rows_by_split.items()
    }
    if any(
        split_units[left] & split_units[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise CorpusBuildError("an identity split unit crosses corpus splits")

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".s0_build_", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        paths = {
            split: temporary / f"{split}.jsonl"
            for split in ("train", "validation", "test")
        }
        for split, path in paths.items():
            _write_jsonl(path, rows_by_split[split])
            if validate:
                _validate_corpus(
                    path,
                    metadata_optional_additions=overlay[
                        "metadata_optional_additions"
                    ],
                )

        row_counts = {split: len(rows) for split, rows in rows_by_split.items()}
        status_counts = {
            split: dict(
                sorted(
                    Counter(row["messages"][-1]["content"]["status"] for row in rows).items()
                )
            )
            for split, rows in rows_by_split.items()
        }
        form_counts = {
            split: dict(
                sorted(
                    Counter(row["metadata"]["source_trace"] for row in rows).items()
                )
            )
            for split, rows in rows_by_split.items()
        }
        framing_counts = {
            split: dict(
                sorted(
                    Counter(row["metadata"]["framing"] for row in rows).items()
                )
            )
            for split, rows in rows_by_split.items()
        }
        fact_kind_counts = {
            split: dict(
                sorted(
                    Counter(row["metadata"]["fact_kind"] for row in rows).items()
                )
            )
            for split, rows in rows_by_split.items()
        }
        input_variant_counts = {
            split: dict(
                sorted(
                    Counter(
                        row["metadata"]["input_variant"] for row in rows
                    ).items()
                )
            )
            for split, rows in rows_by_split.items()
        }
        prompt_template_counts = {
            split: dict(
                sorted(
                    Counter(
                        row["metadata"]["prompt_template_id"] for row in rows
                    ).items()
                )
            )
            for split, rows in rows_by_split.items()
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "dataset_id": settings["dataset_id"],
            "implementation": {
                "builder": {
                    "path": str(
                        Path(__file__).resolve().relative_to(REPOSITORY_ROOT)
                    ),
                    "sha256": _sha256(Path(__file__).resolve()),
                },
                "tool_runtime": {
                    "path": str(
                        Path(__file__)
                        .with_name("tools.py")
                        .resolve()
                        .relative_to(REPOSITORY_ROOT)
                    ),
                    "sha256": _sha256(Path(__file__).with_name("tools.py")),
                },
            },
            "contract": {
                "path": str(base_contract_path.relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(base_contract_path),
                "s0_overlay": {
                    "path": str(overlay_path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(overlay_path),
                    "schema_version": overlay["schema_version"],
                    "metadata_optional_additions": overlay[
                        "metadata_optional_additions"
                    ],
                },
                "row_schema_version": contract["runtime"]["row_schema_version"],
                "tool_contract_version": contract["runtime"][
                    "tool_contract_version"
                ],
            },
            "source_index": {
                "content_sha256": runtime.source_index_sha256,
                "file_name": source_index_path.name,
            },
            "assignment": {
                "seed": seed,
                "maximum_train_ambiguous_candidates": settings[
                    "maximum_train_ambiguous_candidates"
                ],
                "input_casefold_share": settings["input_casefold_share"],
                "prompt_templates_sha256": _stable_sha256(PROMPT_TEMPLATES),
                "train_split_units_sha256": _stable_sha256(
                    sorted(split_units["train"])
                ),
                "validation_split_units": sorted(split_units["validation"]),
                "test_split_units": sorted(split_units["test"]),
            },
            "row_counts": row_counts,
            "status_counts": status_counts,
            "form_counts": form_counts,
            "framing_counts": framing_counts,
            "fact_kind_counts": fact_kind_counts,
            "unique_fact_counts": {
                split: dict(
                    sorted(Counter(fact.kind for fact in facts).items())
                )
                for split, facts in facts_by_split.items()
            },
            "input_variant_counts": input_variant_counts,
            "prompt_template_counts": prompt_template_counts,
            "evidence": {
                "sources": payload["sources"],
                "call_fields": ["evidence_id", "row_id", "tool_name"],
                "calls": evidence_calls,
            },
            "files": {
                f"{split}.jsonl": {
                    "row_count": row_counts[split],
                    "sha256": _sha256(path),
                }
                for split, path in paths.items()
            },
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)

        output_directory.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            (temporary / name).replace(output_directory / name)
        manifest_path.replace(output_directory / "manifest.json")
    return manifest


def _configured_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--overlay", type=Path, default=S0_V4_OVERLAY_PATH)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = load_contract(args.contract)
    settings, _ = load_s0_v4_settings(
        contract,
        args.overlay,
        base_contract_path=args.contract,
    )
    digest = contract["source_index"]["content_sha256"]
    source_index = args.source_index or (
        _configured_path(contract["source_index"]["output_directory"])
        / f"world_model_source_index_{digest}.json.gz"
    )
    output_directory = args.output_directory or _configured_path(
        settings["output_directory"]
    )
    manifest = build_corpus(
        source_index,
        output_directory,
        contract=contract,
        overlay_path=args.overlay,
        base_contract_path=args.contract,
    )
    print(
        json.dumps(
            {
                "output_directory": str(output_directory.resolve()),
                "row_counts": manifest["row_counts"],
                "files": manifest["files"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
