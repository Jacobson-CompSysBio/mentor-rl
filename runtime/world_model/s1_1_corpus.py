"""Build the canonical S1.1 symbol-to-Ensembl corpus.

The shared runtime contract owns sources, counts, proportions, and failure
policy.  This module adds only deterministic fact selection and rendering for
the eleven reviewed S1.1 forms.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from .contract import (
    CONTRACT_PATH,
    REPOSITORY_ROOT,
    composite_provenance,
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from .s0_corpus import (
    EvidenceSequence,
    RESOLVED_SYMBOL,
    assign_evaluation_components,
    build_identity_components,
)
from .s1_overlay import apply_s1_overlay, overlay_pin
from .s1_lookup_retry import (
    assign_retry_cases,
    audit_retry_row,
    load_retry_policy,
    render_retry_case,
)
from .s1_prompt_banks import (
    bank_reasoning,
    identity_prompts,
    prompt_bank_manifest,
)
from .source_index import check_source_index
from .sources import (
    MOUSE_GENE_ID_PATTERN,
    file_sha256,
    read_ensembl_gtf,
)
from .tools import (
    GENE_ID_TOOL,
    MEMBERSHIP_TOOL,
    SYMBOL_TOOL,
    ToolCallResult,
    WorldModelToolRuntime,
)


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-1-corpus-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-1-manifest-v1"

FORM_BY_BUCKET = {
    "direct": 1,
    "inside_task": 2,
    "absent_symbol": 3,
    "harness_exact": 4,
    "harness_near_miss": 5,
    "ensembl_input": 6,
    "other_species": 7,
    "backend_unavailable": 8,
    "membership_present": 9,
    "membership_absent": 10,
    "unknown_multiplex": 11,
}
BUCKET_BY_FORM = {value: key for key, value in FORM_BY_BUCKET.items()}
BASE_TOOL_CALL_BUCKETS = frozenset(
    {
        "direct",
        "inside_task",
        "absent_symbol",
        "harness_near_miss",
        "ensembl_input",
        "membership_present",
        "membership_absent",
        "unknown_multiplex",
    }
)
FLEXIBLE_FRAMING_FORMS = (3, 4, 5, 6, 7, 9, 10, 11)

# S1.3 will use the same outer wording.  The request payload, not the wrapper,
# carries task-required facts such as a species or multiplex name.
IDENTITY_PROMPTS = {
    "train": {
        "direct": (
            "Resolve this gene request: {request}.",
            "Answer this gene request: {request}.",
            "Use the supplied inputs to answer this request: {request}.",
            "Determine the requested gene fact: {request}.",
        ),
        "step": (
            "Before the next analysis step, resolve this request: {request}.",
            "The next operation needs this gene request answered: {request}.",
            "Prepare the requested gene fact for the next operation: {request}.",
            "Resolve this request so the larger analysis can proceed: {request}.",
        ),
    },
    "validation": {
        "direct": (
            "Find the answer to this gene request: {request}.",
            "Return the requested gene fact: {request}.",
            "Work out this gene request: {request}.",
            "Supply the answer for this gene request: {request}.",
        ),
        "step": (
            "Resolve this gene request before the analysis continues: {request}.",
            "The following gene fact is needed for the next step: {request}.",
            "Before continuing, answer this gene request: {request}.",
            "Prepare this gene fact for the next analysis action: {request}.",
        ),
    },
    "test": {
        "direct": (
            "What is the answer to this gene request: {request}?",
            "Give the requested gene fact: {request}.",
            "Answer the following request about a gene: {request}.",
            "Identify the requested gene fact: {request}.",
        ),
        "step": (
            "The workflow first needs this gene request resolved: {request}.",
            "Answer this gene request for the next workflow action: {request}.",
            "Before the workflow advances, determine this gene fact: {request}.",
            "Resolve the following gene request for downstream use: {request}.",
        ),
    },
}


class CorpusBuildError(ValueError):
    """Report a source, selection, or rendered-corpus invariant failure."""


@dataclass(frozen=True, slots=True)
class GeneFact:
    """One source-backed identity with its leakage-safe split units."""

    fact_id: str
    split_units: tuple[str, ...]
    gene_symbol: str
    gene_id: str | None
    species: str = "human"


@dataclass(frozen=True, slots=True)
class Scenario:
    """One selected form before runtime execution."""

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
        return f"S1.1.{self.form}"

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
    return Fraction(str(value))


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
                f"S1.1 {split} row count is not integral: {value}"
            )
        counts[split] = value.numerator
    return counts


def largest_remainder_counts(
    total: int, shares: Mapping[str, Any]
) -> dict[str, int]:
    """Apportion an exact integer total without order-dependent rounding."""

    names = list(shares)
    raw = {name: Fraction(total) * _fraction(shares[name]) for name in names}
    counts = {name: value.numerator // value.denominator for name, value in raw.items()}
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
    """Allocate ordinary forms, then replace direct rows with failure pairs."""

    policy = settings["failure_injection"]
    if policy != {
        "source_form": "direct",
        "allocation": "replace",
        "call_stage": "first",
    }:
        raise CorpusBuildError("unsupported S1.1 failure policy")
    base = largest_remainder_counts(total, settings["shares"])
    tool_rows = sum(base[name] for name in BASE_TOOL_CALL_BUCKETS)
    failure_value = Fraction(tool_rows) * _fraction(
        curriculum["tool_failure_share"]
    )
    if failure_value.denominator != 1:
        raise CorpusBuildError(
            f"S1.1 failure allocation is not integral: {failure_value}"
        )
    failures = failure_value.numerator
    base["direct"] -= failures
    if base["direct"] < failures:
        raise CorpusBuildError(
            "the retained direct rows cannot supply every failure pair"
        )
    counts = {
        FORM_BY_BUCKET[name]: count for name, count in base.items()
    }
    counts[8] = failures
    if sum(counts.values()) != total:
        raise CorpusBuildError("S1.1 form counts do not match the split total")
    return dict(sorted(counts.items()))


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


def build_fact_pools(
    payload: Mapping[str, Any],
    mouse_genes: Iterable[Any],
    *,
    s0_settings: Mapping[str, Any],
    seed: int,
    fractions: Mapping[str, Any],
) -> dict[str, dict[str, list[GeneFact]]]:
    """Build split-safe current, historical, membership, and mouse pools."""

    components, component_by_gene = build_identity_components(
        payload["genes_by_id"]
    )
    assignments = assign_evaluation_components(
        components,
        s0_settings["evaluation"],
        seed=int(s0_settings["assignment_seed"]),
    )
    symbol_to_ids: dict[str, list[str]] = defaultdict(list)
    for gene_id, values in payload["genes_by_id"].items():
        if values[0] is not None:
            symbol_to_ids[str(values[0]).casefold()].append(str(gene_id))

    pools = {
        split: {
            "current": [],
            "membership_present": [],
            "membership_absent": [],
            "historical": [],
            "mouse": [],
        }
        for split in ("train", "validation", "test")
    }
    multiplex_genes = frozenset(payload["multiplex_gene_ids"])
    for component in components:
        split = assignments.get(component.split_unit, "train")
        for identity in component.facts:
            if identity.kind != RESOLVED_SYMBOL:
                continue
            gene_ids = symbol_to_ids[identity.value.casefold()]
            if len(gene_ids) != 1:
                raise CorpusBuildError("a resolved symbol does not have one ID")
            gene_id = gene_ids[0]
            fact = GeneFact(
                fact_id=identity.fact_id,
                split_units=(component.split_unit,),
                gene_symbol=identity.value,
                gene_id=gene_id,
            )
            pools[split]["current"].append(fact)
            membership_pool = (
                "membership_present"
                if gene_id in multiplex_genes
                else "membership_absent"
            )
            pools[split][membership_pool].append(fact)

    historical_by_key: dict[str, list[Sequence[Any]]] = defaultdict(list)
    for row in payload["historical_symbols"]:
        historical_by_key[str(row[0]).casefold()].append(row)
    for key, rows in historical_by_key.items():
        linked_units = {
            component_by_gene[str(gene_id)]
            for row in rows
            for gene_id in row[2]
            if str(gene_id) in component_by_gene
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
                f"historical:{key}", seed=seed, fractions=fractions
            )
        )
        spelling = min(
            (str(row[0]) for row in rows),
            key=lambda value: _stable_sha256(
                {"historical_key": key, "spelling": value}
            ),
        )
        units = tuple(sorted(linked_units)) or (
            "historical_" + _stable_sha256(key)[:24],
        )
        pools[split]["historical"].append(
            GeneFact(
                fact_id="fact_" + _stable_sha256(
                    {"kind": "historical_symbol", "value": spelling}
                )[:24],
                split_units=units,
                gene_symbol=spelling,
                gene_id=None,
            )
        )

    mouse_by_symbol: dict[str, Any] = {}
    for gene in mouse_genes:
        if gene.gene_symbol is None:
            continue
        key = gene.gene_symbol.casefold()
        previous = mouse_by_symbol.get(key)
        if previous is None or gene.gene_id < previous.gene_id:
            mouse_by_symbol[key] = gene
    for key, gene in mouse_by_symbol.items():
        split = _split_for_hash(
            f"mouse:{gene.gene_id}", seed=seed, fractions=fractions
        )
        pools[split]["mouse"].append(
            GeneFact(
                fact_id="fact_" + _stable_sha256(
                    {
                        "kind": "mouse_symbol",
                        "gene_id": gene.gene_id,
                        "gene_symbol": gene.gene_symbol,
                    }
                )[:24],
                split_units=(
                    "mouse_" + _stable_sha256(gene.gene_id)[:24],
                ),
                gene_symbol=gene.gene_symbol,
                gene_id=gene.gene_id,
                species="mouse",
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


def _input_value(
    fact: GeneFact,
    *,
    form: int,
    split: str,
    occurrence: int,
    seed: int,
    case_rate: Any,
) -> str:
    value = fact.gene_id if form == 6 else fact.gene_symbol
    if value is None:
        raise CorpusBuildError(f"form {form} selected a fact without its input")
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
    framing_offset: int,
    case_rate: Any,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
) -> list[Scenario]:
    if not facts:
        raise CorpusBuildError(f"the {split} form {form} pool is empty")
    maximum_variants = len(prompts[split]["direct"])
    if count > len(facts) * maximum_variants:
        raise CorpusBuildError(
            f"the {split} form {form} pool cannot make {count} unique prompts"
        )
    selected = []
    for ordinal in range(count):
        rank = ordinal % len(facts)
        occurrence = ordinal // len(facts)
        fact = facts[rank]
        if form == 1:
            framing = "direct"
        elif form == 2:
            framing = "step"
        else:
            framing = (
                "direct" if (framing_offset + ordinal) % 2 == 0 else "step"
            )
        templates = prompts[split][framing]
        base_template = rank if form in {1, 2} else rank // 2
        template_index = (base_template + occurrence) % len(templates)
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


def select_scenarios(
    pools: Mapping[str, Mapping[str, Sequence[GeneFact]]],
    counts_by_split: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
    case_rate: Any,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    train_decorrelate: bool = False,
) -> dict[str, list[Scenario]]:
    """Select every form and create exact form-1/form-8 minimal pairs."""

    scenarios_by_split: dict[str, list[Scenario]] = {}
    for split in ("train", "validation", "test"):
        counts = counts_by_split[split]
        split_pools = pools[split]
        selected: dict[int, list[Scenario]] = {}
        flexible_offset = 0
        pool_for_form = {
            1: "current",
            2: "current",
            3: "historical",
            4: "current",
            5: "current",
            6: "current",
            7: "mouse",
            9: "membership_present",
            10: "membership_absent",
            11: "current",
        }
        for form in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
            selected[form] = _select(
                split_pools[pool_for_form[form]],
                counts[form],
                split=split,
                form=form,
                seed=seed,
                framing_offset=flexible_offset,
                case_rate=case_rate,
                prompts=prompts,
            )
            if form in FLEXIBLE_FRAMING_FORMS:
                flexible_offset += counts[form]

        distractors = split_pools["current"]
        near_misses = []
        for scenario in selected[5]:
            # Wording and framing must not choose the unrelated registry fact.
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
                    if distractors[(start + offset) % len(distractors)]
                    .gene_symbol.casefold()
                    != scenario.fact.gene_symbol.casefold()
                ),
                None,
            )
            if distractor is None:
                raise CorpusBuildError("a near-miss distractor is unavailable")
            near_misses.append(replace(scenario, distractor=distractor))
        selected[5] = near_misses

        ordered_successes = sorted(
            selected[1], key=lambda scenario: scenario.scenario_id
        )
        pair_sources = []
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
            # Source order is independent of the wrapper bank. Its round-robin
            # wrappers also cover every position in the failure subset.
            pair_sources = selected[1]
        failures = [
            replace(
                success,
                form=8,
                paired_success_id=success.scenario_id,
            )
            for success in pair_sources[: counts[8]]
        ]
        selected[8] = failures
        combined = [
            scenario
            for form in range(1, 12)
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
    """Require the same complete wrapper bank in each applicable train case."""

    coverage: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    for row, scenario in zip(rows, scenarios, strict=True):
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        multiplex_id = row["messages"][-1]["content"]["result"].get("multiplex_id", "")
        if user != _question(scenario, multiplex_id=multiplex_id, prompts=prompts):
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


def _request(scenario: Scenario, *, multiplex_id: str) -> str:
    if scenario.form == 7:
        return (
            "the human Ensembl gene ID for the mouse gene "
            f"{scenario.input_value}"
        )
    if scenario.form in {9, 10, 11}:
        return (
            f"whether the human gene {scenario.input_value} is a node in "
            f"{multiplex_id}"
        )
    return f"the human Ensembl gene ID for {scenario.input_value}"


def _question(
    scenario: Scenario,
    *,
    multiplex_id: str,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
) -> str:
    template = prompts[scenario.split][scenario.framing][
        scenario.template_index
    ]
    return template.format(request=_request(scenario, multiplex_id=multiplex_id))


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


def _resolved_answer(
    scenario: Scenario,
    output: Mapping[str, Any],
    *,
    release: str,
) -> tuple[str, dict[str, Any]]:
    if output.get("status") != "resolved":
        raise CorpusBuildError(
            f"form {scenario.form} selected a non-resolving identity"
        )
    gene_id = str(output["gene_id"])
    symbol = output["gene_symbol"]
    if not isinstance(symbol, str):
        raise CorpusBuildError("S1.1 selected an unnamed current gene")
    decision = "continue" if scenario.framing == "step" else "stop"
    if scenario.form == 6:
        thinking = (
            f"The ID is in release {release} and has the symbol {symbol}. "
            + (
                "The next step can use the confirmed ID."
                if decision == "continue"
                else "The task asks only for the ID, so the task is complete."
            )
        )
        summary = (
            f"The input is already an Ensembl gene ID: "
            f"<GENE:{gene_id}|{symbol}> in Ensembl release {release}."
        )
        if decision == "continue":
            summary += " The analysis continues with this ID."
    else:
        thinking = (
            f"The registry maps {symbol} to one ID, {gene_id}. "
            + (
                "The next step can use it."
                if decision == "continue"
                else "The task asks only for this ID, so the task is complete."
            )
        )
        summary = (
            f"{symbol} is <GENE:{gene_id}|{symbol}> in Ensembl release "
            f"{release}."
        )
        if decision == "continue":
            summary += " The analysis continues with this ID."
    return thinking, _answer(
        status="resolved",
        provenance=output["provenance"],
        result={"gene_id": gene_id, "gene_symbol": symbol},
        decision=decision,
        blocked_reason=None,
        summary=summary,
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
    return "wm3_s1_1_" + _stable_sha256(
        {
            "scenario_id": scenario.scenario_id,
            "schema_version": schema_version,
        }
    )[:24]


def build_row(
    scenario: Scenario,
    *,
    runtime: WorldModelToolRuntime,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]],
    unknown_multiplex_id: str,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    absent_reasoning: str | None = None,
    membership_answers: Mapping[str, Any] | None = None,
    final_reasoning: str | None = None,
    membership_reasoning: Mapping[str, str] | None = None,
    id_confirmation_reasoning: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Execute and render one source-backed S1.1 scenario.

    ``absent_reasoning``, when given, replaces the final reasoning of
    training S1.1.10 rows. ``membership_answers`` (S1 v5 overlay) sets the
    S1.1.10 status and adds ``present`` to the S1.1.11 result in every split.
    ``final_reasoning`` replaces only S1.1.2 or S1.1.3 final training thinking.
    ``membership_reasoning`` replaces both post-tool thoughts only in
    training S1.1.9-11 rows. The caller selects the reviewed templates
    for the form, preserving the lookup and membership evidence.
    ``id_confirmation_reasoning`` binds S1.1.6 final training thinking to
    the confirmed identity and the user's existing request scope.
    """

    schema_version = str(contract["runtime"]["row_schema_version"])
    row_id = _row_id(scenario, schema_version)
    release = runtime.ensembl_release
    multiplex_id = (
        unknown_multiplex_id if scenario.form == 11 else runtime.multiplex_id
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": contract["runtime"]["system_prompt"]}
    ]
    evidence: list[list[str]] = []
    membership_reasoning = (
        membership_reasoning
        if scenario.split == "train" and scenario.form in {9, 10, 11} else None
    )

    if scenario.form in {4, 5}:
        harness_fact = (
            scenario.fact if scenario.form == 4 else scenario.distractor
        )
        if harness_fact is None:
            raise CorpusBuildError("a near miss has no distractor")
        harness_result = runtime.execute(
            SYMBOL_TOOL, {"gene_symbol": harness_fact.gene_symbol}
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
        {
            "role": "user",
            "content": _question(
                scenario, multiplex_id=multiplex_id, prompts=prompts
            ),
        }
    )

    final_thinking: str
    final_answer: dict[str, Any]
    if scenario.form == 4:
        gene_id = scenario.fact.gene_id
        if gene_id is None:
            raise CorpusBuildError("an exact harness fact has no gene ID")
        harness_output = harness_result.output
        symbol = str(harness_output["gene_symbol"])
        decision = "continue" if scenario.framing == "step" else "stop"
        final_thinking = (
            f"The registry entries from the harness list {symbol} for "
            f"Ensembl release {release}. The lookup is already done, so I "
            "will not call the tool."
        )
        summary = (
            f"{symbol} is <GENE:{gene_id}|{symbol}> in Ensembl release "
            f"{release}."
        )
        if decision == "continue":
            summary += " The analysis continues with this ID."
        final_answer = _answer(
            status="resolved",
            provenance=harness_output["provenance"],
            result={"gene_id": gene_id, "gene_symbol": symbol},
            decision=decision,
            blocked_reason=None,
            summary=summary,
        )
    elif scenario.form == 7:
        value = scenario.input_value
        final_thinking = (
            "The task names a mouse gene. The registry holds human genes "
            "only, and the lookup ignores case, so it could match a human "
            "symbol with the same letters. I will not call the tool."
        )
        final_answer = _answer(
            status="invalid_input",
            provenance=None,
            result={"input": value, "expected": "human_gene_symbol"},
            decision="defer",
            blocked_reason=(
                f"The input {value} is a mouse gene symbol, and the registry "
                "holds human genes only."
            ),
            summary=(
                f"{value} is named as a mouse gene, and the registry holds "
                "human genes only. Give a human gene symbol to continue."
            ),
        )
    elif scenario.form == 6:
        arguments = {"gene_id": scenario.input_value}
        call_result = runtime.execute(GENE_ID_TOOL, arguments)
        call_evidence = _evidence_row(call_result, row_id=row_id)
        if call_evidence is None:
            raise CorpusBuildError("the ID lookup produced no evidence")
        evidence.append(call_evidence)
        _append_call(
            messages,
            thinking=(
                "The input is already an Ensembl gene ID, not a symbol. The "
                "symbol tool cannot match it, so I will check it with the ID "
                "tool."
            ),
            tool_name=GENE_ID_TOOL,
            arguments=arguments,
            call_result=call_result,
        )
        final_thinking, final_answer = _resolved_answer(
            scenario, call_result.output, release=release
        )
    elif scenario.form == 8:
        arguments = {"gene_symbol": scenario.input_value}
        call_result = runtime.execute(
            SYMBOL_TOOL, arguments, inject_backend_failure=True
        )
        if call_result.evidence is not None:
            raise CorpusBuildError("a failed call produced evidence")
        _append_call(
            messages,
            thinking=(
                f"The task gives only the gene input {scenario.input_value}. "
                "I will look it up in the registry."
            ),
            tool_name=SYMBOL_TOOL,
            arguments=arguments,
            call_result=call_result,
        )
        error_code = str(call_result.output["error_code"])
        final_thinking = (
            "The call failed because the registry could not be reached. This "
            f"does not show that {scenario.input_value} is absent, so I have "
            "no result for it. I must defer."
        )
        final_answer = _answer(
            status="error",
            provenance=None,
            result={
                "failed_tool": SYMBOL_TOOL,
                "error_code": error_code,
                "gene_symbol": scenario.input_value,
            },
            decision="defer",
            blocked_reason=(
                f"The call to {SYMBOL_TOOL} failed with {error_code}, so "
                f"there is no registry result for {scenario.input_value}."
            ),
            summary=(
                f"The registry lookup for {scenario.input_value} failed "
                "because the backend was unreachable. This does not mean the "
                "gene is absent. Retry the lookup to continue."
            ),
        )
    elif scenario.form in {9, 10, 11}:
        lookup_arguments = {"gene_symbol": scenario.input_value}
        lookup = runtime.execute(SYMBOL_TOOL, lookup_arguments)
        lookup_evidence = _evidence_row(lookup, row_id=row_id)
        if lookup_evidence is None or lookup.output.get("status") != "resolved":
            raise CorpusBuildError("a membership symbol did not resolve")
        evidence.append(lookup_evidence)
        _append_call(
            messages,
            thinking="I need the registry ID before checking graph membership.",
            tool_name=SYMBOL_TOOL,
            arguments=lookup_arguments,
            call_result=lookup,
        )
        gene_id = str(lookup.output["gene_id"])
        symbol = str(lookup.output["gene_symbol"])
        membership_arguments = {
            "multiplex_id": multiplex_id,
            "gene_ids": [gene_id],
        }
        membership = runtime.execute(MEMBERSHIP_TOOL, membership_arguments)
        membership_evidence = _evidence_row(membership, row_id=row_id)
        if membership_evidence is None:
            raise CorpusBuildError("a membership lookup produced no evidence")
        evidence.append(membership_evidence)
        membership_thinking = (
            "The symbol resolved, so I will check the ID in the named multiplex."
        )
        if membership_reasoning is not None:
            membership_thinking = membership_reasoning["after_lookup"].format(
                gene_id=gene_id, symbol=symbol,
            )
        _append_call(
            messages,
            thinking=membership_thinking,
            tool_name=MEMBERSHIP_TOOL,
            arguments=membership_arguments,
            call_result=membership,
        )
        provenance = composite_provenance(
            contract,
            registry=lookup.output["provenance"]["registry"],
            graph=membership.output["provenance"]["graph"],
        )
        if scenario.form == 11:
            if membership.output.get("status") != "not_found":
                raise CorpusBuildError("the unknown multiplex unexpectedly exists")
            final_thinking = "The graph registry does not know this multiplex."
            if membership_reasoning is not None:
                final_thinking = membership_reasoning[scenario.framing]
            result = {
                "gene_id": gene_id,
                "gene_symbol": symbol,
                "multiplex_id": multiplex_id,
            }
            if membership_answers is not None:
                result["present"] = membership_answers["unknown_multiplex_present"]
            final_answer = _answer(
                status="not_found",
                provenance=provenance,
                result=result,
                decision="defer",
                blocked_reason=(
                    f"The multiplex {multiplex_id} is not registered."
                ),
                summary=(
                    f"<MULTIPLEX:{multiplex_id}> is not registered. Give a "
                    "registered multiplex to continue."
                ),
            )
        else:
            if membership.output.get("status") != "resolved":
                raise CorpusBuildError("the registered multiplex was not found")
            present = gene_id in membership.output["present_gene_ids"]
            if present != (scenario.form == 9):
                raise CorpusBuildError("membership selection and runtime disagree")
            decision = "continue" if scenario.framing == "step" else "stop"
            final_thinking = (
                "The graph result directly answers the membership question."
            )
            if not present and scenario.split == "train" and absent_reasoning:
                final_thinking = absent_reasoning
            if membership_reasoning is not None:
                final_thinking = membership_reasoning[scenario.framing].format(
                    gene_id=gene_id,
                )
            relation = "is" if present else "is not"
            summary = (
                f"<GENE:{gene_id}|{symbol}> {relation} a node in "
                f"<MULTIPLEX:{multiplex_id}>."
            )
            if decision == "continue":
                summary += " The analysis proceeds with this membership result."
            absent_status = (
                membership_answers["absent_status"]
                if membership_answers is not None
                else "not_found"
            )
            final_answer = _answer(
                status="resolved" if present else absent_status,
                provenance=provenance,
                result={
                    "gene_id": gene_id,
                    "gene_symbol": symbol,
                    "multiplex_id": multiplex_id,
                    "present": present,
                },
                decision=decision,
                blocked_reason=None,
                summary=summary,
            )
    else:
        arguments = {"gene_symbol": scenario.input_value}
        call_result = runtime.execute(SYMBOL_TOOL, arguments)
        call_evidence = _evidence_row(call_result, row_id=row_id)
        if call_evidence is None:
            raise CorpusBuildError("the symbol lookup produced no evidence")
        evidence.append(call_evidence)
        if scenario.form == 5:
            if scenario.distractor is None:
                raise CorpusBuildError("a near miss has no distractor")
            pre_call = (
                f"The registry entries list {scenario.distractor.gene_symbol} "
                f"but not {scenario.input_value}. I will look up "
                f"{scenario.input_value} in the registry."
            )
        elif scenario.framing == "step":
            pre_call = (
                f"The next step needs the Ensembl ID for "
                f"{scenario.input_value}. I will look it up in the registry."
            )
        else:
            pre_call = (
                f"The task gives only the gene input {scenario.input_value}. "
                "I will look it up in the registry."
            )
        _append_call(
            messages,
            thinking=pre_call,
            tool_name=SYMBOL_TOOL,
            arguments=arguments,
            call_result=call_result,
        )
        if scenario.form == 3:
            output = call_result.output
            if output.get("status") != "not_found":
                raise CorpusBuildError("an absent symbol unexpectedly resolved")
            value = str(output["gene_symbol"])
            final_thinking = (
                f"The registry has no match for {value}. I will not replace "
                "it with a symbol or an ID from memory, so I must defer."
            )
            final_answer = _answer(
                status="not_found",
                provenance=output["provenance"],
                result={"gene_symbol": value},
                decision="defer",
                blocked_reason=(
                    f"The registry has no gene with the symbol {value} in "
                    f"Ensembl release {release}."
                ),
                summary=(
                    f"{value} has no match in Ensembl release {release}, so "
                    "no graph lookup can use it. Give a current symbol or an "
                    "Ensembl gene ID to continue."
                ),
            )
        else:
            final_thinking, final_answer = _resolved_answer(
                scenario, call_result.output, release=release
            )

    if scenario.split == "train" and scenario.form in {2, 3} and final_reasoning:
        bindings = {"symbol": final_answer["result"]["gene_symbol"]}
        if scenario.form == 2:
            bindings["gene_id"] = final_answer["result"]["gene_id"]
        final_thinking = final_reasoning.format(**bindings)
    if (scenario.split == "train" and scenario.form == 6
            and id_confirmation_reasoning):
        final_thinking = id_confirmation_reasoning[scenario.framing].format(
            gene_id=call_result.output["gene_id"],
            symbol=call_result.output["gene_symbol"],
            user_request=messages[1]["content"],
        )

    messages.append(
        {
            "role": "assistant",
            "thinking": final_thinking,
            "content": final_answer,
        }
    )
    blocked_code = {
        3: "not_found",
        7: "out_of_scope_species",
        8: "tool_error",
        11: "unknown_multiplex",
    }.get(scenario.form)
    metadata = {
        "book_mode": "tool_call",
        "step": "S1",
        "question_family": contract["curriculum"]["families"]["S1.1"][
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
    if scenario.form == 7:
        metadata["input_species"] = "mouse"
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


def audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    expected_counts: Mapping[str, Mapping[int, int]],
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    membership_answers: Mapping[str, Any] | None = None,
    train_form_prompts: Mapping[
        int, Mapping[str, Mapping[str, Sequence[str]]]
    ] | None = None,
    train_retry_cases: Mapping[str, str] | None = None,
    train_id_confirmation_reasoning: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Enforce S1.1 semantics that the generic row validator cannot see."""

    row_ids: set[str] = set()
    split_units: dict[str, set[str]] = {}
    expected_calls = {
        1: [SYMBOL_TOOL],
        2: [SYMBOL_TOOL],
        3: [SYMBOL_TOOL],
        4: [],
        5: [SYMBOL_TOOL],
        6: [GENE_ID_TOOL],
        7: [],
        8: [SYMBOL_TOOL],
        9: [SYMBOL_TOOL, MEMBERSHIP_TOOL],
        10: [SYMBOL_TOOL, MEMBERSHIP_TOOL],
        11: [SYMBOL_TOOL, MEMBERSHIP_TOOL],
    }
    expected_status = {
        1: "resolved",
        2: "resolved",
        3: "not_found",
        4: "resolved",
        5: "resolved",
        6: "resolved",
        7: "invalid_input",
        8: "error",
        9: "resolved",
        10: "not_found",
        11: "not_found",
    }
    if membership_answers is not None:
        expected_status[10] = membership_answers["absent_status"]
    status_counts: dict[str, dict[str, int]] = {}
    call_counts: dict[str, dict[str, int]] = {}
    framing_counts: dict[str, dict[str, int]] = {}
    case_counts: dict[str, dict[str, int]] = {}
    pair_counts: dict[str, int] = {}
    template_counts: dict[str, dict[str, int]] = {}

    for split in ("train", "validation", "test"):
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} rows and scenarios differ")
        actual_forms = Counter(
            int(row["metadata"]["form"]) for row in rows
        )
        if dict(sorted(actual_forms.items())) != dict(expected_counts[split]):
            raise CorpusBuildError(f"{split} form counts changed")

        successes = {
            scenario.scenario_id: (scenario, row)
            for scenario, row in zip(scenarios, rows, strict=True)
            if scenario.form == 1
        }
        pairs = 0
        units: set[str] = set()
        case_eligible = case_changed = 0
        for scenario, row in zip(scenarios, rows, strict=True):
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise CorpusBuildError(f"duplicate row ID {row_id}")
            row_ids.add(row_id)
            units.update(scenario.split_units)
            calls = _tool_calls(row)
            retry_case = (
                (train_retry_cases or {}).get(row_id) if split == "train" else None
            )
            wanted_calls = expected_calls[scenario.form]
            wanted_status = expected_status[scenario.form]
            if retry_case is not None:
                if scenario.form != 8:
                    raise CorpusBuildError("a lookup retry changed another form")
                audit_retry_row(row, retry_case)
                wanted_calls = [SYMBOL_TOOL] * (1 if retry_case == "no_retry" else 2)
                wanted_status = "resolved" if retry_case == "recovered" else "error"
            if calls != wanted_calls:
                raise CorpusBuildError(
                    f"form {scenario.form} emitted calls {calls!r}"
                )
            answer = row["messages"][-1]["content"]
            if answer["status"] != wanted_status:
                raise CorpusBuildError(
                    f"form {scenario.form} emitted status {answer['status']!r}"
                )
            if (split == "train" and scenario.form == 6
                    and train_id_confirmation_reasoning):
                lookup = row["messages"][-2]["content"]
                expected_request = _question(scenario, multiplex_id="", prompts=prompts)
                expected_thinking = train_id_confirmation_reasoning[scenario.framing].format(
                    gene_id=lookup["gene_id"], symbol=lookup["gene_symbol"],
                    user_request=expected_request,
                )
                if (row["messages"][1]["content"] != expected_request
                        or row["messages"][-1]["thinking"] != expected_thinking):
                    raise CorpusBuildError("ID confirmation reasoning changed its evidence or request binding")
                if (
                    answer["result"] != {key: lookup[key] for key in ("gene_id", "gene_symbol")}
                    or answer["provenance"] != lookup["provenance"]
                    or answer["continuation_decision"] != ("continue" if scenario.framing == "step" else "stop")
                    or answer["blocked_reason"] is not None
                ):
                    raise CorpusBuildError("ID confirmation answer changed its evidence or request scope")
            if scenario.form == 8:
                paired = successes.get(str(scenario.paired_success_id))
                if paired is None:
                    raise CorpusBuildError("a failure has no retained success pair")
                success_scenario, success_row = paired
                same_prompt = (
                    row["messages"][1]["content"]
                    == success_row["messages"][1]["content"]
                )
                if split == "train" and train_form_prompts:
                    # A reviewed form may use another frozen wrapper bank.
                    # Both prompts must still render their declared templates
                    # and have the same request and original wrapper position.
                    same_prompt = all(
                        member["messages"][1]["content"] == _question(
                            case, multiplex_id="",
                            prompts=train_form_prompts.get(case.form, prompts),
                        )
                        for case, member in (
                            (scenario, row), (success_scenario, success_row)
                        )
                    ) and _question(
                        scenario, multiplex_id="", prompts=prompts
                    ) == _question(
                        success_scenario, multiplex_id="", prompts=prompts
                    )
                if (
                    row["messages"][-2]["role"] != "tool"
                    or not same_prompt
                    or scenario.input_value != success_scenario.input_value
                    or scenario.template_index != success_scenario.template_index
                    or scenario.framing != success_scenario.framing
                ):
                    raise CorpusBuildError("a failure pair changed its prompt")
                pairs += 1
            if scenario.fact.species == "human":
                canonical = (
                    scenario.fact.gene_id
                    if scenario.form == 6
                    else scenario.fact.gene_symbol
                )
                case_eligible += 1
                case_changed += scenario.input_value != canonical
        split_units[split] = units
        pair_counts[split] = pairs
        if pairs != expected_counts[split][8]:
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
        framing_counts[split] = dict(
            sorted(Counter(item.framing for item in scenarios).items())
        )
        case_counts[split] = {
            "eligible": case_eligible,
            "perturbed": case_changed,
        }
        coverage = Counter(
            f"S1.1.{scenario.form}:{scenario.framing}:"
            f"{scenario.template_index}"
            for scenario in scenarios
        )
        template_counts[split] = dict(sorted(coverage.items()))
        for form in range(1, 12):
            frames = (
                ("direct",)
                if form in {1, 8}
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
                        f"{split} form {form} does not cover every "
                        f"{framing} template"
                    )
        if abs(framing_counts[split].get("direct", 0) - len(rows) / 2) > 1:
            raise CorpusBuildError(f"{split} framing is not balanced")

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = split_units[left] & split_units[right]
        if overlap:
            raise CorpusBuildError(
                f"{left} and {right} share {len(overlap)} split units"
            )
    return {
        "status_counts": status_counts,
        "tool_call_counts": call_counts,
        "framing_counts": framing_counts,
        "case_counts": case_counts,
        "failure_pair_counts": pair_counts,
        "prompt_template_counts": template_counts,
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
    """Build, validate, and atomically publish all three S1.1 splits.

    A pinned prompt bank replaces only the training wrappers (and, from S1
    v4, the S1.1.10 training reasoning); validation and test keep the inline
    text, so their rows do not change. The S1 v5 overlay's membership answers
    change the S1.1.10 and S1.1.11 answers in every split.
    S1 v6 can replace one form's training wrappers at their existing indices;
    selection and unreviewed reasoning continue to use the default bank.
    Reviewed final-reasoning templates apply to S1.1.2 and S1.1.3 training only.
    S1.1.9-11 training can replace both post-tool thoughts by framing.
    S1 v10 binds S1.1.6 final training thinking to the existing user request.
    """

    source_index_path = source_index_path.resolve()
    payload = check_source_index(source_index_path, contract)
    curriculum = contract["curriculum"]
    settings = curriculum["families"]["S1.1"]
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
            len(form_prompts["train"][framing])
            != len(prompts["train"][framing])
            for framing in ("direct", "step")
        ):
            raise CorpusBuildError(
                f"form {form} training wrappers change the template count"
            )
    absent_reasoning = (
        bank_reasoning(prompt_banks, "s1_1_absent_membership")
        if prompt_banks is not None
        else None
    )
    membership_answers = settings.get("membership_answers")
    train_final_reasoning = settings.get("train_final_reasoning", {})
    train_id_confirmation_reasoning = settings.get("train_id_confirmation_reasoning")
    train_membership_reasoning_by_form = {
        9: settings.get("train_membership_reasoning"),
        10: settings.get("train_absent_membership_reasoning"),
        11: settings.get("train_unknown_multiplex_reasoning"),
    }
    row_counts = split_row_counts(
        int(settings["train_rows"]), curriculum["split_fractions"]
    )
    counts_by_split = {
        split: form_counts(row_counts[split], settings, curriculum)
        for split in ("train", "validation", "test")
    }

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
        prompts=prompts,
        train_decorrelate=train_decorrelate,
    )

    runtime = WorldModelToolRuntime.from_source_index(
        source_index_path,
        contract=contract,
        evidence_id_factory=EvidenceSequence(),
    )
    tools = tool_definitions(contract)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    evidence_calls: list[list[str]] = []
    for split in ("train", "validation", "test"):
        rows = []
        for scenario in scenarios_by_split[split]:
            row, row_evidence = build_row(
                scenario,
                runtime=runtime,
                contract=contract,
                tools=tools,
                unknown_multiplex_id=str(settings["unknown_multiplex_id"]),
                prompts=(
                    train_form_prompts.get(scenario.form, prompts)
                    if split == "train" else prompts
                ),
                absent_reasoning=absent_reasoning,
                membership_answers=membership_answers,
                final_reasoning=train_final_reasoning.get(scenario.form),
                membership_reasoning=train_membership_reasoning_by_form.get(scenario.form),
                id_confirmation_reasoning=train_id_confirmation_reasoning,
            )
            rows.append(row)
            evidence_calls.extend(row_evidence)
        rows_by_split[split] = rows

    retry_pin = settings.get("train_lookup_retries")
    train_retry_cases = {}
    if retry_pin is not None:
        retry_policy = load_retry_policy(retry_pin)
        failures = [
            row for row in rows_by_split["train"]
            if row["metadata"]["form"] == 8
        ]
        train_retry_cases = assign_retry_cases(failures)
        if Counter(train_retry_cases.values()) != retry_policy["case_counts"]:
            raise CorpusBuildError("lookup retry allocation differs from its pinned counts")
        # Allocate recovery evidence after the entire baseline build, keeping
        # every other row's evidence IDs and both held-out files unchanged.
        for index, row in enumerate(rows_by_split["train"]):
            case = train_retry_cases.get(row["row_id"])
            if case is not None:
                row, added_evidence = render_retry_case(
                    row, case=case, runtime=runtime, policy=retry_policy,
                )
                rows_by_split["train"][index] = row
                evidence_calls.extend(added_evidence)

    evidence_ids = [item[0] for item in evidence_calls]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise CorpusBuildError("an evidence ID appears more than once")
    audit = audit_rows(
        rows_by_split,
        scenarios_by_split,
        counts_by_split,
        prompts=prompts,
        membership_answers=membership_answers,
        train_form_prompts=train_form_prompts,
        train_retry_cases=train_retry_cases,
        train_id_confirmation_reasoning=train_id_confirmation_reasoning,
    )
    decorrelation_audit = (
        _audit_train_decorrelation(rows_by_split["train"], scenarios_by_split["train"], prompts)
        if train_decorrelate else None
    )

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".s1_1_build_", dir=output_directory.parent
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
                f"S1.1.{form}": count
                for form, count in counts_by_split[split].items()
            }
            for split in ("train", "validation", "test")
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "dataset_id": settings["dataset_id"],
            "section": "S1.1",
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
            },
            "sources": {
                "source_index_sha256": runtime.source_index_sha256,
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
            },
            "failure_injection": {
                **dict(settings["failure_injection"]),
                "share_of_tool_call_rows": curriculum[
                    "tool_failure_share"
                ],
                "pair_counts": audit["failure_pair_counts"],
            },
            "prompt_templates_sha256": _stable_sha256(
                {"default": prompts, "train_form_overrides": train_form_prompts}
                if train_form_prompts else prompts
            ),
            "prompt_template_counts": audit["prompt_template_counts"],
            "row_counts": row_counts,
            "form_counts": form_count_manifest,
            "status_counts": audit["status_counts"],
            "tool_call_counts": audit["tool_call_counts"],
            "framing_counts": audit["framing_counts"],
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
        if overlay is not None:
            manifest["contract"]["s1_overlay"] = dict(overlay)
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
        if train_id_confirmation_reasoning is not None:
            manifest["train_id_confirmation_reasoning"] = dict(train_id_confirmation_reasoning)
        for form, key in (
            (9, "train_membership_reasoning"),
            (10, "train_absent_membership_reasoning"),
            (11, "train_unknown_multiplex_reasoning"),
        ):
            reasoning = train_membership_reasoning_by_form[form]
            if reasoning is not None:
                manifest[key] = dict(reasoning)
        if retry_pin is not None:
            manifest["train_lookup_retries"] = {
                **dict(retry_pin), "max_retries": 1,
                "case_counts": dict(sorted(
                    Counter(train_retry_cases.values()).items()
                )),
                "case_row_ids_sha256": _stable_sha256(train_retry_cases),
                "runtime_sha256": file_sha256(
                    Path(__file__).with_name("s1_lookup_retry.py")
                ),
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
    settings = contract["curriculum"]["families"]["S1.1"]
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
