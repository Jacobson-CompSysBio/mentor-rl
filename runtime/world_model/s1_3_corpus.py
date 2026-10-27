"""Build the canonical S1.3 ambiguous-symbol corpus."""

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
    AMBIGUOUS_SYMBOL,
    EvidenceSequence,
    assign_evaluation_components,
    build_identity_components,
)
from .s1_1_corpus import IDENTITY_PROMPTS
from .s1_membership_retry import (
    assign_membership_retry_cases,
    audit_membership_retry_row,
    load_membership_retry_policy,
    render_membership_retry_case,
)
from .s1_overlay import (
    apply_s1_overlay,
    apply_train_form_counts,
    overlay_pin,
)
from .s1_prompt_banks import (
    LATER_STEP_CUES, identity_prompts, load_choice_replies, prompt_bank_manifest,
)
from .source_index import check_source_index
from .sources import file_sha256
from .tools import (
    MEMBERSHIP_TOOL,
    SYMBOL_TOOL,
    ToolCallResult,
    WorldModelToolRuntime,
)


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-3-corpus-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-3-manifest-v1"
MULTIPLEX_ID = "full_brain_multiplex_v1"

FORM_BY_BUCKET = {
    "direct": 1,
    "inside_task": 2,
    "already_marked": 3,
    "harness_near_miss": 4,
    "membership_keeps_one": 5,
    "membership_keeps_several": 6,
    "membership_keeps_none": 7,
    "user_chooses": 8,
    "reply_does_not_choose": 9,
    "membership_backend_unavailable": 10,
}
BUCKET_BY_FORM = {value: key for key, value in FORM_BY_BUCKET.items()}
MEMBERSHIP_POOL_BY_FORM = {
    5: "keeps_one",
    6: "keeps_several",
    7: "keeps_none",
}
FAILURE_SOURCE_SHARES = {
    "keeps_one": Fraction(230, 300),
    "keeps_several": Fraction(24, 300),
    "keeps_none": Fraction(46, 300),
}

CHOICE_REPLIES = (
    "Use {gene_id}.",
    "Choose the candidate {gene_id}.",
    "Continue with {gene_id}.",
    "The intended gene is {gene_id}.",
)
NONCHOICE_REPLIES = (
    "Use any candidate.",
    "Continue with all candidates.",
    "Use the matching gene.",
    "Choose whichever candidate fits.",
)
CHOICE_STANDALONE_SCOPE = "The gene-ID selection is the whole request."
CHOICE_AMBIGUITY_SUFFIX = "Please identify the intended Ensembl entry."


class CorpusBuildError(ValueError):
    """Report a source, selection, or corpus invariant failure."""


@dataclass(frozen=True, slots=True)
class GeneFact:
    """Store one source-backed ambiguous symbol and its membership result."""

    fact_id: str
    split_unit: str
    gene_symbol: str
    candidate_gene_ids: tuple[str, ...]
    candidate_details: Mapping[str, Mapping[str, str]]
    present_gene_ids: tuple[str, ...]

    @property
    def membership_outcome(self) -> str:
        if len(self.present_gene_ids) == 1:
            return "keeps_one"
        if len(self.present_gene_ids) > 1:
            return "keeps_several"
        return "keeps_none"


@dataclass(frozen=True, slots=True)
class Scenario:
    """Store one selected form before the runtime executes its calls."""

    split: str
    form: int
    fact: GeneFact
    framing: str
    template_index: int
    occurrence: int
    input_value: str
    distractor: GeneFact | None = None
    chosen_gene_id: str | None = None
    reply: str | None = None
    reply_index: int | None = None
    multiplex_in_harness: bool = False
    paired_success_id: str | None = None

    @property
    def source_trace(self) -> str:
        return f"S1.3.{self.form}"

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
                "distractor": self.distractor.fact_id if self.distractor else None,
                "chosen_gene_id": self.chosen_gene_id,
                "reply": self.reply,
                "reply_index": self.reply_index,
                "multiplex_in_harness": self.multiplex_in_harness,
                "paired_success_id": self.paired_success_id,
            }
        )

    @property
    def split_units(self) -> tuple[str, ...]:
        units = {self.fact.split_unit}
        if self.distractor is not None:
            units.add(self.distractor.split_unit)
        return tuple(sorted(units))

    @property
    def metadata_split_unit(self) -> str:
        if len(self.split_units) == 1:
            return self.split_units[0]
        return "group_" + _stable_sha256(self.split_units)[:24]


def _canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def split_row_counts(
    train_rows: int, split_fractions: Mapping[str, Any]
) -> dict[str, int]:
    """Derive each split count from the training anchor."""

    train_fraction = _fraction(split_fractions["train"])
    counts = {"train": train_rows}
    for split in ("validation", "test"):
        count = Fraction(train_rows) * _fraction(
            split_fractions[split]
        ) / train_fraction
        if count.denominator != 1:
            raise CorpusBuildError(
                f"S1.3 {split} row count is not integral: {count}"
            )
        counts[split] = count.numerator
    return counts


def largest_remainder_counts(
    total: int, shares: Mapping[str, Any]
) -> dict[str, int]:
    """Apportion an integer total and use contract order for each tie."""

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
    """Scale the training allocations to one exact split total."""

    if settings["failure_injection"] != {
        "source_forms": [
            "membership_keeps_one",
            "membership_keeps_several",
            "membership_keeps_none",
        ],
        "allocation": "replace",
        "call_stage": "membership",
    }:
        raise CorpusBuildError("the S1.3 failure policy changed")
    train_rows = int(settings["train_rows"])
    allocations = settings["allocations"]
    if sum(int(value) for value in allocations.values()) != train_rows:
        raise CorpusBuildError("the S1.3 allocations do not match train_rows")
    shares = {
        name: Fraction(int(value), train_rows)
        for name, value in allocations.items()
    }
    counts = largest_remainder_counts(total, shares)
    result = {FORM_BY_BUCKET[name]: count for name, count in counts.items()}
    if total == train_rows:
        tool_rows = total - result[3] - result[8] - result[9]
        expected_failure = Fraction(tool_rows) * _fraction(
            curriculum["tool_failure_share"]
        )
        if expected_failure.denominator != 1 or result[10] != expected_failure:
            raise CorpusBuildError("the S1.3 failure share changed")
    return dict(sorted(result.items()))


def build_fact_pools(
    payload: Mapping[str, Any],
    *,
    s0_settings: Mapping[str, Any],
    maximum_candidate_gene_ids: int,
) -> tuple[dict[str, dict[str, list[GeneFact]]], dict[str, int]]:
    """Build split-safe ambiguous-symbol pools from the source index."""

    components, _ = build_identity_components(payload["genes_by_id"])
    assignments = assign_evaluation_components(
        components,
        s0_settings["evaluation"],
        seed=int(s0_settings["assignment_seed"]),
    )
    genes_by_id = payload["genes_by_id"]
    multiplex_gene_ids = set(payload["multiplex_gene_ids"])
    ids_by_symbol: dict[str, list[str]] = defaultdict(list)
    for gene_id, values in genes_by_id.items():
        if values[0] is not None:
            ids_by_symbol[str(values[0]).casefold()].append(str(gene_id))

    pools = {
        split: {
            "all": [],
            "keeps_one": [],
            "keeps_several": [],
            "keeps_none": [],
        }
        for split in ("train", "validation", "test")
    }
    source_outcomes: Counter[str] = Counter()
    for component in components:
        split = assignments.get(component.split_unit, "train")
        for identity in component.facts:
            if identity.kind != AMBIGUOUS_SYMBOL:
                continue
            candidate_ids = tuple(
                sorted(ids_by_symbol[identity.value.casefold()])
            )
            present_ids = tuple(
                gene_id
                for gene_id in candidate_ids
                if gene_id in multiplex_gene_ids
            )
            outcome = (
                "keeps_one"
                if len(present_ids) == 1
                else "keeps_several"
                if present_ids
                else "keeps_none"
            )
            source_outcomes[outcome] += 1
            if len(candidate_ids) > maximum_candidate_gene_ids:
                continue
            fact = GeneFact(
                fact_id=identity.fact_id,
                split_unit=identity.split_unit,
                gene_symbol=identity.value,
                candidate_gene_ids=candidate_ids,
                candidate_details={
                    gene_id: {
                        "chromosome": str(genes_by_id[gene_id][1]),
                        "biotype": str(genes_by_id[gene_id][2]),
                    }
                    for gene_id in candidate_ids
                },
                present_gene_ids=present_ids,
            )
            pools[split]["all"].append(fact)
            pools[split][outcome].append(fact)

    for split, split_pools in pools.items():
        for name, facts in split_pools.items():
            split_pools[name] = sorted(
                facts,
                key=lambda fact: _stable_sha256(
                    {"split": split, "pool": name, "fact_id": fact.fact_id}
                ),
            )
        if not split_pools["all"]:
            raise CorpusBuildError(f"the {split} ambiguous pool is empty")
    return pools, dict(source_outcomes)


def source_pool_coverage(
    pools: Mapping[str, Mapping[str, Sequence[GeneFact]]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Measure candidate counts and independent ambiguity classes."""

    candidate_distributions: dict[str, dict[str, int]] = {}
    present_distributions: dict[str, dict[str, int]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        facts = pools[split]["all"]
        candidate_distributions[split] = dict(
            sorted(
                Counter(str(len(fact.candidate_gene_ids)) for fact in facts).items(),
                key=lambda item: int(item[0]),
            )
        )
        present_distributions[split] = dict(
            sorted(
                Counter(str(len(fact.present_gene_ids)) for fact in facts).items(),
                key=lambda item: int(item[0]),
            )
        )
        classes = Counter(
            {
                "par_x_y_protein_coding": 0,
                "pseudogene_lncRNA": 0,
                "multicopy": 0,
                "long_list_above_five": 0,
            }
        )
        for fact in facts:
            details = list(fact.candidate_details.values())
            chromosomes = {str(item["chromosome"]) for item in details}
            biotypes = [str(item["biotype"]) for item in details]
            if (
                {"X", "Y"}.issubset(chromosomes)
                and biotypes
                and all(value == "protein_coding" for value in biotypes)
            ):
                classes["par_x_y_protein_coding"] += 1
            if any("pseudogene" in value for value in biotypes) and any(
                value.casefold() == "lncrna" for value in biotypes
            ):
                classes["pseudogene_lncRNA"] += 1
            if len(fact.candidate_gene_ids) > 2:
                classes["multicopy"] += 1
            if len(fact.candidate_gene_ids) > 5:
                classes["long_list_above_five"] += 1
        class_counts[split] = dict(classes)

    for split in ("train", "validation"):
        missing = [
            name for name, count in class_counts[split].items() if count < 1
        ]
        if missing:
            raise CorpusBuildError(
                f"the {split} ambiguity pool misses classes {missing!r}"
            )
    return {
        "candidate_count_distribution": candidate_distributions,
        "present_candidate_count_distribution": present_distributions,
        "ambiguity_class_counts": class_counts,
    }


def _mixed_case(value: str) -> str:
    return "".join(
        char.lower() if index % 2 == 0 else char.upper()
        for index, char in enumerate(value)
    )


def _scenario_input(
    fact: GeneFact, *, seed: int, split: str, form: int, ordinal: int, rate: Any,
    decorrelate: bool = False,
) -> str:
    point = Fraction(
        int(
            _stable_sha256(
                {
                    "assignment_seed": seed,
                    "split": split,
                    "form": form,
                    "ordinal": ordinal,
                    "fact_id": fact.fact_id,
                    "purpose": "case",
                }
            )[:16],
            16,
        ),
        16**16,
    )
    if point >= _fraction(rate):
        return fact.gene_symbol
    lower = ordinal % 2 == 0
    if decorrelate:
        lower = int(_stable_sha256({
            "seed": seed, "form": form, "ordinal": ordinal,
            "fact_id": fact.fact_id, "purpose": "case_style",
        })[-8:], 16) % 2 == 0
    return (
        fact.gene_symbol.lower()
        if lower
        else _mixed_case(fact.gene_symbol)
    )


def _single_survivor_schedule(
    facts: Sequence[GeneFact], *, count: int, seed: int,
) -> dict[int, GeneFact]:
    """Sample real nonfirst survivors once per four inputs, before wording."""

    pools = {
        nonfirst: [fact for fact in facts if (
            fact.present_gene_ids[0] != fact.candidate_gene_ids[0]
        ) == nonfirst]
        for nonfirst in (False, True)
    }
    if not all(pools.values()):
        raise CorpusBuildError("decorrelation needs first and nonfirst survivors")
    frame_count = (count + 1) // 2
    nonfirst_indexes = set()
    for start in range(0, frame_count, 4):
        nonfirst_indexes.add(min(range(start, min(start + 4, frame_count)),
            key=lambda index: _stable_sha256({
            "seed": seed, "index": index,
            "purpose": "nonfirst_survivor",
        })))
    positions: Counter[bool] = Counter()
    schedule = {}
    for index in range(frame_count):
        nonfirst = index in nonfirst_indexes
        pool = pools[nonfirst]
        schedule[index] = pool[positions[nonfirst] % len(pool)]
        positions[nonfirst] += 1
    return schedule


def _ordered_facts(
    facts: Sequence[GeneFact], *, seed: int, split: str, form: int
) -> list[GeneFact]:
    return sorted(
        facts,
        key=lambda fact: _stable_sha256(
            {
                "assignment_seed": seed,
                "split": split,
                "form": form,
                "fact_id": fact.fact_id,
            }
        ),
    )


def _framing_sequence(count: int, form: int) -> list[str]:
    if form == 1:
        return ["direct"] * count
    if form == 2:
        return ["step"] * count
    first = "direct" if form in {3, 5, 7, 8, 10} else "step"
    second = "step" if first == "direct" else "direct"
    return [first if index % 2 == 0 else second for index in range(count)]


def _template_balanced_order(
    scenarios: Iterable[Scenario],
    *,
    seed: int,
    split: str,
    source_form: int,
    placement_offset: int,
) -> list[Scenario]:
    """Balance pair candidates across templates and multiplex placement."""

    groups: dict[tuple[int, bool], list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[
            (scenario.template_index, scenario.multiplex_in_harness)
        ].append(scenario)
    for group in groups.values():
        group.sort(
            key=lambda item: _stable_sha256(
                {
                    "assignment_seed": seed,
                    "split": split,
                    "source_form": source_form,
                    "scenario_id": item.scenario_id,
                    "purpose": "failure_pair",
                }
            )
        )
    ordered: list[Scenario] = []
    indexes = sorted({key[0] for key in groups})
    while any(groups.values()):
        for template_index in indexes:
            prefer_harness = (placement_offset + len(ordered)) % 2 == 0
            for placement in (prefer_harness, not prefer_harness):
                group = groups[(template_index, placement)]
                if group:
                    ordered.append(group.pop(0))
                    break
    return ordered


def _select_form_scenarios(
    *,
    split: str,
    form: int,
    count: int,
    facts: Sequence[GeneFact],
    all_facts: Sequence[GeneFact],
    seed: int,
    case_rate: Any,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    balanced: bool = False,
    decorrelate: bool = False,
) -> list[Scenario]:
    """Select one form's scenarios.

    A balanced split assigns wrappers round-robin so a large training bank
    covers every wrapper, and for forms 5-7 every wrapper with both multiplex
    placements.  Unbalanced splits keep the original assignment.
    """

    if count and not facts:
        raise CorpusBuildError(f"form {form} cannot fill {split}")
    ordered = _ordered_facts(facts, seed=seed, split=split, form=form)
    all_ordered = _ordered_facts(
        all_facts, seed=seed, split=split, form=form + 100
    )
    frames = _framing_sequence(count, form)
    frame_indexes = Counter()
    scenarios: list[Scenario] = []
    used_membership_inputs: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    survivor_wrappers: Counter[tuple[str, bool, bool, int]] = Counter()
    used_survivor_wrappers: dict[tuple[str, str, bool], set[int]] = defaultdict(set)
    survivor_occurrences: Counter[tuple[str, str]] = Counter()
    survivor_schedule = (
        _single_survivor_schedule(
            ordered, count=count, seed=seed,
        )
        if decorrelate and form == 5 else {}
    )
    for ordinal in range(count):
        framing = frames[ordinal]
        frame_index = frame_indexes[framing]
        frame_indexes[framing] += 1
        fact_index = frame_index % len(ordered)
        fact = survivor_schedule.get(frame_index, ordered[fact_index])
        cycle = frame_index // len(ordered)
        if survivor_schedule:
            occurrence_key = (framing, fact.fact_id)
            cycle = survivor_occurrences[occurrence_key]
            survivor_occurrences[occurrence_key] += 1
        template_count = len(prompts[split][framing])
        template_start = fact_index * template_count // len(ordered)
        template_index = (template_start + cycle) % template_count
        if balanced:
            if form in {5, 6, 7}:
                template_index = frame_index % (2 * template_count) // 2
            else:
                template_index = frame_index % template_count
        distractor = None
        if form == 4:
            candidates = [
                item for item in all_ordered if item.split_unit != fact.split_unit
            ]
            if not candidates:
                raise CorpusBuildError("a near-miss row has no split-safe distractor")
            distractor = candidates[(frame_index + cycle) % len(candidates)]
        chosen = None
        reply = None
        reply_index = None
        if form == 8:
            chosen = fact.candidate_gene_ids[
                (frame_index + cycle) % len(fact.candidate_gene_ids)
            ]
            reply_index = (
                fact_index * len(CHOICE_REPLIES) // len(ordered) + cycle
            ) % len(CHOICE_REPLIES)
            reply = CHOICE_REPLIES[reply_index].format(
                gene_id=chosen
            )
        elif form == 9:
            reply_index = (
                fact_index * len(NONCHOICE_REPLIES) // len(ordered) + cycle
            ) % len(NONCHOICE_REPLIES)
            reply = NONCHOICE_REPLIES[reply_index]
        multiplex_in_harness = (
            form in {5, 6, 7}
            and (
                (frame_index if balanced else fact_index + cycle // template_count)
                + (1 if framing == "step" else 0)
            )
            % 2
            == 0
        )
        if survivor_schedule:
            nonfirst = fact.present_gene_ids[0] != fact.candidate_gene_ids[0]
            used = used_survivor_wrappers[(framing, fact.fact_id, multiplex_in_harness)]
            if len(used) == template_count:
                used.clear()
            template_index = min(
                (index for index in range(template_count) if index not in used),
                key=lambda index: (
                    survivor_wrappers[(framing, nonfirst, multiplex_in_harness, index)],
                    _stable_sha256({"seed": seed, "framing": framing,
                        "ordinal": ordinal, "wrapper": index}),
                ),
            )
            used.add(template_index)
            survivor_wrappers[(framing, nonfirst, multiplex_in_harness, template_index)] += 1
        input_value = _scenario_input(
            fact,
            seed=seed,
            split=split,
            form=form,
            ordinal=ordinal,
            rate=case_rate,
            decorrelate=decorrelate,
        )
        if form in {5, 6, 7}:
            input_key = (
                fact.fact_id,
                framing,
                template_index,
                multiplex_in_harness,
            )
            used = used_membership_inputs[input_key]
            if input_value.casefold() in {item.casefold() for item in used}:
                variants = (
                    fact.gene_symbol,
                    fact.gene_symbol.lower(),
                    _mixed_case(fact.gene_symbol),
                )
                input_value = next(
                    (
                        value
                        for value in variants
                        if value not in used
                    ),
                    "",
                )
                if not input_value:
                    raise CorpusBuildError(
                        f"form {form} cannot make a unique visible prompt"
                    )
            used.add(input_value)
        scenario = Scenario(
            split=split,
            form=form,
            fact=fact,
            framing=framing,
            template_index=template_index,
            occurrence=cycle,
            input_value=input_value,
            distractor=distractor,
            chosen_gene_id=chosen,
            reply=reply,
            reply_index=reply_index,
            multiplex_in_harness=multiplex_in_harness,
        )
        scenarios.append(scenario)
    return scenarios


def select_scenarios(
    pools: Mapping[str, Mapping[str, Sequence[GeneFact]]],
    counts_by_split: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
    case_rate: Any = Fraction(1, 10),
    framing_shares: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    balanced_splits: Iterable[str] = (),
    train_decorrelate: bool = False,
) -> dict[str, list[Scenario]]:
    """Select deterministic scenarios from synthetic or source pools."""

    balanced_splits = frozenset(balanced_splits)

    if framing_shares is not None and dict(framing_shares) != {
        "direct": 0.5,
        "step": 0.5,
    }:
        raise CorpusBuildError("S1.3 supports equal direct and step framing")
    selected: dict[str, list[Scenario]] = {}
    for split in ("train", "validation", "test"):
        split_pools = pools[split]
        all_facts = split_pools["all"]
        counts = counts_by_split[split]
        scenarios: list[Scenario] = []
        by_form: dict[int, list[Scenario]] = {}
        for form in range(1, 10):
            pool_name = MEMBERSHIP_POOL_BY_FORM.get(form, "all")
            items = _select_form_scenarios(
                split=split,
                form=form,
                count=int(counts[form]),
                facts=split_pools[pool_name],
                all_facts=all_facts,
                seed=seed,
                case_rate=case_rate,
                prompts=prompts,
                balanced=split in balanced_splits or (train_decorrelate and split == "train"),
                decorrelate=train_decorrelate and split == "train",
            )
            by_form[form] = items
            scenarios.extend(items)

        failure_counts = largest_remainder_counts(
            int(counts[10]), FAILURE_SOURCE_SHARES
        )
        failures: list[Scenario] = []
        failure_frame_offset = 0
        failure_placement_offset = 0
        for pool_name, failure_count in failure_counts.items():
            source_form = {
                "keeps_one": 5,
                "keeps_several": 6,
                "keeps_none": 7,
            }[pool_name]
            successes = by_form[source_form]
            if failure_count > len(successes):
                raise CorpusBuildError(
                    f"form 10 cannot pair with enough form {source_form} rows"
                )
            failure_frames = [
                "direct"
                if (failure_frame_offset + index) % 2 == 0
                else "step"
                for index in range(failure_count)
            ]
            failure_frame_offset += failure_count
            wanted_frames = Counter(failure_frames)
            paired = []
            for framing, frame_count in wanted_frames.items():
                ordered_successes = _template_balanced_order(
                    (item for item in successes if item.framing == framing),
                    seed=seed,
                    split=split,
                    source_form=source_form,
                    placement_offset=failure_placement_offset,
                )
                if train_decorrelate and split == "train":
                    ordered_successes = sorted(ordered_successes, key=lambda item: _stable_sha256({
                        "seed": seed, "source_form": source_form, "framing": framing,
                        "fact": item.fact.fact_id, "occurrence": item.occurrence,
                        "input": item.input_value, "placement": item.multiplex_in_harness,
                        "purpose": "failure_pair_without_wording",
                    }))
                if len(ordered_successes) < frame_count:
                    raise CorpusBuildError(
                        f"form 10 lacks {framing} form {source_form} pairs"
                    )
                paired.extend(ordered_successes[:frame_count])
                failure_placement_offset += frame_count
            failures.extend(
                replace(
                    item,
                    form=10,
                    paired_success_id=item.scenario_id,
                )
                for item in paired
            )
        scenarios.extend(failures)
        selected[split] = sorted(
            scenarios,
            key=lambda item: _stable_sha256(
                {
                    "assignment_seed": seed,
                    "split": split,
                    "scenario_id": item.scenario_id,
                    "purpose": "output",
                }
            ),
        )
    return selected


def _initial_request(scenario: Scenario) -> str:
    request = f"give the human Ensembl gene ID for {scenario.input_value}"
    if scenario.form in {5, 6, 7, 10} and not scenario.multiplex_in_harness:
        request = (
            f"give the human Ensembl gene ID for {scenario.input_value} "
            f"in multiplex {MULTIPLEX_ID}"
        )
    return request


def _prompt(
    scenario: Scenario,
    request: str,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
) -> str:
    request = request.rstrip().rstrip(".!?")
    return prompts[scenario.split][scenario.framing][
        scenario.template_index
    ].format(request=request)


def _choice_assignments(
    scenarios: Sequence[Scenario], *, seed: int, reply_count: int,
) -> dict[str, tuple[str, int]]:
    """Give each existing direct/workflow pair the same scope and reply variant."""

    pairs: dict[tuple[str, int, str | None], list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        if scenario.form == 8:
            pairs[scenario.fact.fact_id, scenario.occurrence, scenario.chosen_gene_id].append(scenario)
    if len(pairs) != 500 or any(
        len(pair) != 2 or {item.framing for item in pair} != {"direct", "step"}
        for pair in pairs.values()
    ):
        raise CorpusBuildError("targeted user-choice revision requires 500 direct/workflow pairs")
    ordered = sorted(pairs, key=lambda key: _stable_sha256({
        "seed": seed, "pair": key, "purpose": "v9_choice_scope",
    }))
    placements = ("both", "both", "earlier", "latest")
    reply_indexes = sorted(range(reply_count), key=lambda index: _stable_sha256({
        "seed": seed, "index": index, "purpose": "v9_choice_reply",
    }))
    placement_counts: Counter[str] = Counter()
    assignments = {}
    for index, key in enumerate(ordered):
        placement = placements[index % len(placements)]
        reply_index = reply_indexes[placement_counts[placement] % reply_count]
        placement_counts[placement] += 1
        for scenario in pairs[key]:
            assignments[scenario.scenario_id] = placement, reply_index
    return assignments


def _choice_users(
    scenario: Scenario, placement: str, reply: str,
    prompts: Mapping[str, Mapping[str, Sequence[str]]],
    *, train_targeted_v10: bool = False,
) -> tuple[str, str]:
    """Render scope only in the designated user turns, never in an ID choice."""

    users = []
    for turn, request in (("earlier", _initial_request(scenario)), ("latest", reply)):
        explicit = placement in {"both", turn}
        framed = scenario if explicit else replace(scenario, framing="direct")
        text = _prompt(framed, request, prompts=prompts)
        if explicit and scenario.framing == "direct" and not train_targeted_v10:
            text = f"{CHOICE_STANDALONE_SCOPE} {text}"
        users.append(text)
    return users[0], users[1]


def _choice_thinking(
    scenario: Scenario, placement: str, reasoning: Mapping[str, str],
    *, train_targeted_v10: bool = False,
) -> str:
    """Explain identity selection and cite the separate visible scope evidence."""

    original = reasoning[scenario.framing].format(
        gene_id=scenario.chosen_gene_id, symbol=scenario.fact.gene_symbol,
    )
    body, separator, ending = original.rpartition(". ")
    if train_targeted_v10 and scenario.framing == "direct":
        return (
            f"{body}{separator}Selecting the ID establishes identity; neither user turn "
            f"requests further work beyond this gene-ID selection. {ending}"
        )
    source = {"both": "both user turns", "earlier": "the earlier user turn", "latest": "the latest user turn"}[placement]
    scope = "standalone" if scenario.framing == "direct" else "later workflow"
    return (
        f"{body}{separator}Selecting the ID establishes identity; the {scope} scope "
        f"is explicit in {source}, independently of that selection. {ending}"
    )


def _audit_choice_scope(
    rows: Sequence[Mapping[str, Any]], scenarios: Sequence[Scenario], *,
    assignments: Mapping[str, tuple[str, int]], replies: Sequence[str],
    prompts: Mapping[str, Mapping[str, Sequence[str]]],
    reasoning: Mapping[str, str],
    train_targeted_v10: bool = False,
) -> dict[str, Any]:
    """Check all visible scope cues and full reply-by-scope-by-framing coverage."""

    coverage: Counter[str] = Counter()
    for row, scenario in zip(rows, scenarios, strict=True):
        if scenario.form != 8:
            continue
        placement, reply_index = assignments[scenario.scenario_id]
        reply = replies[reply_index].format(gene_id=scenario.chosen_gene_id)
        expected_users = _choice_users(
            scenario, placement, reply, prompts, train_targeted_v10=train_targeted_v10,
        )
        users = [message["content"] for message in row["messages"] if message["role"] == "user"]
        if tuple(users) != expected_users:
            raise CorpusBuildError("user-choice history changed its designated scope cues")
        for turn, user in zip(("earlier", "latest"), users, strict=True):
            explicit = placement in {"both", turn}
            if bool(LATER_STEP_CUES.search(user)) != (explicit and scenario.framing == "step"):
                raise CorpusBuildError("user-choice history leaks or contradicts workflow scope")
            if (CHOICE_STANDALONE_SCOPE in user) != (
                explicit and scenario.framing == "direct" and not train_targeted_v10
            ):
                raise CorpusBuildError("user-choice history leaks or contradicts standalone scope")
        current_user = max(i for i, m in enumerate(row["messages"]) if m["role"] == "user")
        for message in row["messages"][:current_user]:
            if message["role"] != "assistant":
                continue
            content = message.get("content", {})
            texts = [message.get("thinking", "")]
            texts.extend(content.get(key) or "" for key in ("summary", "blocked_reason"))
            if any(
                LATER_STEP_CUES.search(text)
                or re.search(r"\b(standalone|whole request)\b", text, re.I)
                for text in texts
            ):
                raise CorpusBuildError("earlier assistant text leaks task scope")
        if row["messages"][-1]["thinking"] != _choice_thinking(
            scenario, placement, reasoning, train_targeted_v10=train_targeted_v10,
        ):
            raise CorpusBuildError("user-choice thinking does not cite the visible scope")
        coverage[f"{placement}:{scenario.framing}:{reply_index}"] += 1
    if len(coverage) != 3 * 2 * len(replies):
        raise CorpusBuildError("user-choice revision lost reply/scope/framing coverage")
    return {
        "pair_counts": {"both": 250, "earlier": 125, "latest": 125},
        "reply_count": len(replies),
        "assignment_sha256": _stable_sha256(assignments),
        "coverage": dict(sorted(coverage.items())),
        "pass": True,
    }


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
    if output.get("status") != "ambiguous":
        raise CorpusBuildError("an S1.3 harness entry is not ambiguous")
    entry = {
        "status": output["status"],
        "gene_symbol": output["gene_symbol"],
        "candidate_gene_ids": output["candidate_gene_ids"],
        "candidate_details": output["candidate_details"],
    }
    block = {"provenance": output["provenance"], "entries": [entry]}
    return (
        "Registry entries, Ensembl release 116:\n"
        + _canonical_json(block, sort_keys=False)
    )


def _row_id(scenario: Scenario, schema_version: str) -> str:
    return "wm3_s1_3_" + _stable_sha256(
        {
            "scenario_id": scenario.scenario_id,
            "schema_version": schema_version,
        }
    )[:24]


def _merge_provenance(
    registry: Mapping[str, Any], graph: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {"registry": dict(registry), "graph": dict(graph) if graph else None}


def _candidate_result(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gene_symbol": output["gene_symbol"],
        "candidate_gene_ids": list(output["candidate_gene_ids"]),
        "candidate_details": {
            gene_id: dict(details)
            for gene_id, details in output["candidate_details"].items()
        },
    }


def _anchor(gene_id: str, symbol: str) -> str:
    return f"<GENE:{gene_id}|{symbol}>"


def _candidate_summary(
    output: Mapping[str, Any], *, suffix: str
) -> str:
    symbol = str(output["gene_symbol"])
    candidate_ids = list(output["candidate_gene_ids"])
    if len(candidate_ids) > 5:
        return (
            f"{symbol} matches {len(candidate_ids)} genes in Ensembl release "
            f"116. {suffix}"
        )
    parts = []
    for gene_id in candidate_ids:
        detail = output["candidate_details"][gene_id]
        label = str(detail["biotype"]).replace("_", " ")
        parts.append(f"{_anchor(gene_id, symbol)} ({label})")
    return (
        f"{symbol} matches {len(candidate_ids)} genes in Ensembl release "
        f"116: {', '.join(parts)}. {suffix}"
    )


def _ambiguous_reason(output: Mapping[str, Any]) -> str:
    return (
        f"The symbol {output['gene_symbol']} matches "
        f"{len(output['candidate_gene_ids'])} genes in Ensembl release 116, "
        "and nothing in the task chooses one."
    )


def _ambiguous_answer(
    output: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    return _answer(
        status="ambiguous",
        provenance=provenance,
        result=_candidate_result(output),
        decision="defer",
        blocked_reason=_ambiguous_reason(output),
        summary=_candidate_summary(
            output, suffix="Give one Ensembl gene ID to continue."
        ),
    )


def _pre_lookup_thinking(scenario: Scenario) -> str:
    if scenario.form == 4:
        if scenario.distractor is None:
            raise CorpusBuildError("a near-miss scenario has no distractor")
        return (
            f"The registry entries list {scenario.distractor.gene_symbol}, "
            f"but not {scenario.input_value}. I will look up the requested "
            "symbol in the registry."
        )
    if scenario.framing == "step":
        return (
            f"The next step needs the Ensembl ID for {scenario.input_value}. "
            "I will look up the symbol in the registry."
        )
    return (
        f"The task gives the symbol {scenario.input_value}. I will look it "
        "up in the registry."
    )


def _post_lookup_thinking(
    output: Mapping[str, Any],
    *,
    multiplex_in_harness: bool,
    name_cue: bool = False,
) -> str:
    """Thinking before the membership call.

    ``name_cue`` (S1 v5, training rows) names the multiplex and its source,
    the cue that separates a membership check from a deferral.
    """

    symbol = str(output["gene_symbol"])
    count = len(output["candidate_gene_ids"])
    if name_cue:
        source = (
            f"The harness names {MULTIPLEX_ID} as the active multiplex"
            if multiplex_in_harness
            else f"The task names {MULTIPLEX_ID}"
        )
        return (
            f"The registry maps {symbol} to {count} IDs. {source}, so its "
            "membership can choose one. I will check which candidates are "
            "nodes in it."
        )
    source = "The harness names" if multiplex_in_harness else "The task names"
    return (
        f"The registry maps {symbol} to {count} IDs. {source} the active "
        "multiplex, so I will check which candidates are nodes in it."
    )


def _no_multiplex_thinking(
    output: Mapping[str, Any], *, form: int, name_cue: bool = False
) -> str:
    """Final thinking of a deferral with no multiplex (forms 1 to 4).

    ``name_cue`` (S1 v5, training rows) says what the task or harness names
    instead of a multiplex.
    """

    symbol = str(output["gene_symbol"])
    count = len(output["candidate_gene_ids"])
    if form == 3 and name_cue:
        return (
            f"The harness marks {symbol} as ambiguous with {count} "
            "candidates and names no multiplex, so nothing chooses one. I "
            "will ask for one ID."
        )
    if form == 3:
        return (
            f"The harness marks {symbol} as ambiguous with {count} "
            "candidates. The lookup is complete, but nothing chooses one, so "
            "I will ask for one ID."
        )
    if not name_cue:
        cue = "No multiplex is named"
    elif form == 4:
        cue = "The harness lists registry entries, not a multiplex"
    else:
        cue = "The task names no multiplex"
    return (
        f"The registry maps {symbol} to {count} IDs. {cue}, so nothing "
        "chooses one. I will ask for one ID."
    )


def _earlier_answer(output: Mapping[str, Any]) -> dict[str, Any]:
    return _ambiguous_answer(output, provenance=output["provenance"])


def build_row(
    scenario: Scenario,
    *,
    runtime: WorldModelToolRuntime,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]],
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    membership_answers: Mapping[str, Any] | None = None,
    train_multiplex_reasoning: bool = False,
    final_reasoning: str | None = None,
    single_membership_reasoning: Mapping[str, str] | None = None,
    multiple_membership_reasoning: Mapping[str, str] | None = None,
    no_membership_reasoning: Mapping[str, str] | None = None,
    user_choice_reasoning: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Execute and render one source-backed S1.3 scenario.

    ``membership_answers`` (S1 v5) sets the S1.3.5 to S1.3.7 answers in every
    split. ``train_multiplex_reasoning`` (S1 v5) names the multiplex cue in
    the thinking of training rows and binds the selected-candidate fields.
    ``final_reasoning`` replaces only forms 1 and 4's training conclusions in v6.
    ``single_membership_reasoning`` replaces form 5's two post-tool thoughts.
    ``multiple_membership_reasoning`` replaces form 6's two post-tool thoughts.
    ``no_membership_reasoning`` replaces form 7's two post-tool thoughts.
    ``user_choice_reasoning`` replaces only form 8's final training thought.
    """

    name_cue = train_multiplex_reasoning and scenario.split == "train"
    member_reasoning = (
        {5: single_membership_reasoning, 6: multiple_membership_reasoning, 7: no_membership_reasoning}.get(scenario.form)
        if scenario.split == "train" else None
    )
    keep_tool_fields = bool(
        membership_answers and membership_answers["keep_tool_fields"]
    )
    schema_version = str(contract["runtime"]["row_schema_version"])
    row_id = _row_id(scenario, schema_version)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": contract["runtime"]["system_prompt"]}
    ]
    evidence: list[list[str]] = []
    lookup: ToolCallResult | None = None

    if scenario.form in {3, 4}:
        harness_fact = scenario.fact if scenario.form == 3 else scenario.distractor
        if harness_fact is None:
            raise CorpusBuildError("a near-miss scenario has no distractor")
        harness_result = runtime.execute(
            SYMBOL_TOOL, {"gene_symbol": harness_fact.gene_symbol}
        )
        record = _evidence_row(
            harness_result, row_id=row_id, producer="harness_registry_block"
        )
        if record is None:
            raise CorpusBuildError("a harness lookup produced no evidence")
        evidence.append(record)
        messages.append({"role": "harness", "content": _harness_content(harness_result)})
        if scenario.form == 3:
            lookup = harness_result

    if scenario.form in {5, 6, 7, 10} and scenario.multiplex_in_harness:
        messages.append(
            {"role": "harness", "content": f"Active multiplex: {MULTIPLEX_ID}"}
        )

    if scenario.form in {8, 9}:
        lookup = runtime.execute(
            SYMBOL_TOOL, {"gene_symbol": scenario.input_value}
        )
        record = _evidence_row(
            lookup, row_id=row_id, producer="earlier_turn_registry_lookup"
        )
        if record is None:
            raise CorpusBuildError("an earlier lookup produced no evidence")
        evidence.append(record)
        messages.extend(
            [
                {"role": "user", "content": _prompt(scenario, _initial_request(scenario), prompts=prompts)},
                {"role": "assistant", "content": _earlier_answer(lookup.output)},
                {"role": "user", "content": _prompt(scenario, str(scenario.reply), prompts=prompts)},
            ]
        )
    else:
        messages.append(
            {"role": "user", "content": _prompt(scenario, _initial_request(scenario), prompts=prompts)}
        )

    final_thinking: str
    final_answer: dict[str, Any]
    if scenario.form == 3:
        if lookup is None:
            raise CorpusBuildError("an exact harness has no lookup result")
        output = lookup.output
        final_thinking = _no_multiplex_thinking(
            output, form=3, name_cue=name_cue
        )
        final_answer = _ambiguous_answer(output, provenance=output["provenance"])
    elif scenario.form in {8, 9}:
        if lookup is None:
            raise CorpusBuildError("a multi-turn scenario has no earlier lookup")
        output = lookup.output
        candidate_ids = list(output["candidate_gene_ids"])
        symbol = str(output["gene_symbol"])
        if scenario.form == 8:
            chosen = scenario.chosen_gene_id
            if chosen not in candidate_ids:
                raise CorpusBuildError("the user choice is not a candidate")
            decision = "continue" if scenario.framing == "step" else "stop"
            final_thinking = (
                f"The earlier answer lists {chosen} as a candidate for "
                f"{symbol}. The reply selects it, so the choice is resolved."
            )
            if name_cue:
                final_thinking += (
                    f" Set gene_id to {chosen} and gene_symbol to {symbol}."
                    " Copy the earlier candidate_gene_ids list in its order."
                    " The result has exactly gene_id, gene_symbol, and "
                    "candidate_gene_ids."
                )
            if user_choice_reasoning is not None and scenario.split == "train":
                final_thinking = user_choice_reasoning[scenario.framing].format(
                    gene_id=chosen, symbol=symbol,
                )
            summary = f"The user chose {_anchor(str(chosen), symbol)}."
            if decision == "continue":
                summary += " The analysis continues with this ID."
            final_answer = _answer(
                status="resolved",
                provenance=output["provenance"],
                result={
                    "gene_id": chosen,
                    "gene_symbol": symbol,
                    "candidate_gene_ids": candidate_ids,
                },
                decision=decision,
                blocked_reason=None,
                summary=summary,
            )
        else:
            final_thinking = (
                f"The earlier answer lists {len(candidate_ids)} candidates "
                f"for {symbol}. The reply does not select one, so I will ask "
                "again."
            )
            final_answer = _answer(
                status="ambiguous",
                provenance=output["provenance"],
                result=_candidate_result(output),
                decision="defer",
                blocked_reason=(
                    f"The reply does not choose one of the {len(candidate_ids)} "
                    f"candidates for {symbol}."
                ),
                summary=_candidate_summary(
                    output, suffix="Give one of these IDs to continue."
                ),
            )
    else:
        if scenario.form != 3:
            arguments = {"gene_symbol": scenario.input_value}
            lookup = runtime.execute(SYMBOL_TOOL, arguments)
            record = _evidence_row(lookup, row_id=row_id)
            if record is None:
                raise CorpusBuildError("a symbol lookup produced no evidence")
            evidence.append(record)
            _append_call(
                messages,
                thinking=_pre_lookup_thinking(scenario),
                tool_name=SYMBOL_TOOL,
                arguments=arguments,
                call_result=lookup,
            )
        if lookup is None or lookup.output.get("status") != "ambiguous":
            raise CorpusBuildError("an S1.3 symbol lookup is not ambiguous")
        output = lookup.output
        if scenario.form in {1, 2, 4}:
            final_thinking = _no_multiplex_thinking(
                output, form=scenario.form, name_cue=name_cue
            )
            if scenario.split == "train" and scenario.form in {1, 4} and final_reasoning is not None:
                final_thinking = final_reasoning.format(
                    symbol=output["gene_symbol"], count=len(output["candidate_gene_ids"]),
                )
            final_answer = _ambiguous_answer(
                output, provenance=output["provenance"]
            )
        else:
            membership_arguments = {
                "multiplex_id": MULTIPLEX_ID,
                "gene_ids": list(output["candidate_gene_ids"]),
            }
            membership = runtime.execute(
                MEMBERSHIP_TOOL,
                membership_arguments,
                inject_backend_failure=scenario.form == 10,
            )
            record = _evidence_row(membership, row_id=row_id)
            if record is not None:
                evidence.append(record)
            post_lookup_thinking = _post_lookup_thinking(
                output,
                multiplex_in_harness=scenario.multiplex_in_harness,
                name_cue=name_cue,
            )
            if member_reasoning is not None:
                post_lookup_thinking = member_reasoning["after_lookup"].format(
                    symbol=output["gene_symbol"],
                    count=len(output["candidate_gene_ids"]),
                    source="harness" if scenario.multiplex_in_harness else "task",
                    multiplex_id=membership_arguments["multiplex_id"],
                )
            _append_call(
                messages,
                thinking=post_lookup_thinking,
                tool_name=MEMBERSHIP_TOOL,
                arguments=membership_arguments,
                call_result=membership,
            )
            registry_block = output["provenance"]["registry"]
            candidate_ids = list(output["candidate_gene_ids"])
            symbol = str(output["gene_symbol"])
            if scenario.form == 10:
                error_code = str(membership.output["error_code"])
                final_thinking = (
                    "The membership check failed, so I do not know which "
                    "candidate is a node. The failure does not rule out a "
                    "candidate, so I must defer."
                )
                result = _candidate_result(output)
                result.update(
                    {
                        "multiplex_id": MULTIPLEX_ID,
                        "failed_tool": MEMBERSHIP_TOOL,
                        "error_code": error_code,
                    }
                )
                final_answer = _answer(
                    status="ambiguous",
                    provenance=_merge_provenance(registry_block),
                    result=result,
                    decision="defer",
                    blocked_reason=(
                        f"The lookup found {len(candidate_ids)} candidates, "
                        f"but the call to {MEMBERSHIP_TOOL} failed with "
                        f"{error_code}, so no candidate is checked in "
                        f"{MULTIPLEX_ID}."
                    ),
                    summary=_candidate_summary(
                        output,
                        suffix=(
                            f"The membership check in <MULTIPLEX:{MULTIPLEX_ID}> "
                            "failed, so no candidate is ruled out. Retry the "
                            "check to continue."
                        ),
                    ),
                )
            else:
                member_output = membership.output
                present_ids = list(member_output["present_gene_ids"])
                provenance = _merge_provenance(
                    registry_block, member_output["provenance"]["graph"]
                )
                # S1 v5 keeps every field of both tool results.
                details = _candidate_result(output)["candidate_details"]
                absent_ids = list(member_output["absent_gene_ids"])
                if scenario.form == 5:
                    if len(present_ids) != 1:
                        raise CorpusBuildError("form 5 did not keep one candidate")
                    chosen = present_ids[0]
                    decision = "continue" if scenario.framing == "step" else "stop"
                    final_thinking = (
                        f"Only {chosen} is a node in this multiplex. The "
                        "multiplex therefore selects this candidate."
                    )
                    if member_reasoning is not None:
                        final_thinking = member_reasoning[scenario.framing].format(
                            gene_id=chosen, multiplex_id=member_output["multiplex_id"],
                        )
                    summary = (
                        f"Only {_anchor(chosen, symbol)} is a node in "
                        f"<MULTIPLEX:{MULTIPLEX_ID}>."
                    )
                    if decision == "continue":
                        summary += " The analysis continues with this ID."
                    result = {
                        "gene_id": chosen,
                        "gene_symbol": symbol,
                        "candidate_gene_ids": candidate_ids,
                    }
                    if keep_tool_fields:
                        result["candidate_details"] = details
                    result["multiplex_id"] = MULTIPLEX_ID
                    result["present_gene_ids"] = present_ids
                    if keep_tool_fields:
                        result["absent_gene_ids"] = absent_ids
                    final_answer = _answer(
                        status="resolved",
                        provenance=provenance,
                        result=result,
                        decision=decision,
                        blocked_reason=None,
                        summary=summary,
                    )
                elif scenario.form == 6:
                    if len(present_ids) < 2:
                        raise CorpusBuildError("form 6 did not keep several candidates")
                    final_thinking = (
                        f"The multiplex keeps {len(present_ids)} candidates, "
                        "so membership does not choose one. I will ask for "
                        "one ID."
                    )
                    if member_reasoning is not None:
                        final_thinking = member_reasoning["final"].format(
                            symbol=symbol, count=len(candidate_ids),
                            present_count=len(present_ids), multiplex_id=member_output["multiplex_id"],
                        )
                    result = _candidate_result(output)
                    result.update(
                        {
                            "multiplex_id": MULTIPLEX_ID,
                            "present_gene_ids": present_ids,
                        }
                    )
                    if keep_tool_fields:
                        result["absent_gene_ids"] = absent_ids
                    if len(present_ids) <= 5:
                        listed = ", ".join(
                            _anchor(gene_id, symbol) for gene_id in present_ids
                        )
                        summary = (
                            f"{len(present_ids)} candidates are nodes in "
                            f"<MULTIPLEX:{MULTIPLEX_ID}>: {listed}. Give one "
                            "Ensembl gene ID to continue."
                        )
                    else:
                        summary = (
                            f"{len(present_ids)} candidates are nodes in "
                            f"<MULTIPLEX:{MULTIPLEX_ID}>. Give one Ensembl "
                            "gene ID to continue."
                        )
                    final_answer = _answer(
                        status="ambiguous",
                        provenance=provenance,
                        result=result,
                        decision="defer",
                        blocked_reason=(
                            f"The symbol {symbol} matches {len(candidate_ids)} "
                            f"genes, and {len(present_ids)} of them are nodes "
                            f"in {MULTIPLEX_ID}, so the multiplex does not "
                            "choose one."
                        ),
                        summary=summary,
                    )
                else:
                    if present_ids:
                        raise CorpusBuildError("form 7 kept a candidate")
                    final_thinking = (
                        "No candidate is a node in this multiplex. A choice "
                        "between the candidates cannot supply a usable node, "
                        "so I must defer."
                    )
                    if member_reasoning is not None:
                        final_thinking = member_reasoning["final"].format(
                            symbol=symbol, count=len(candidate_ids),
                            multiplex_id=member_output["multiplex_id"],
                        )
                    result = {
                        "gene_symbol": symbol,
                        "candidate_gene_ids": candidate_ids,
                    }
                    if keep_tool_fields:
                        result["candidate_details"] = details
                    result["multiplex_id"] = MULTIPLEX_ID
                    result["present_gene_ids"] = []
                    if keep_tool_fields:
                        result["absent_gene_ids"] = absent_ids
                    final_answer = _answer(
                        status=(
                            membership_answers["none_present_status"]
                            if membership_answers is not None
                            else "not_found"
                        ),
                        provenance=provenance,
                        result=result,
                        decision="defer",
                        blocked_reason=(
                            f"No candidate for {symbol} is a node in "
                            f"{MULTIPLEX_ID}."
                        ),
                        summary=(
                            f"{symbol} has no candidate node in "
                            f"<MULTIPLEX:{MULTIPLEX_ID}>. Choose another gene "
                            "or another multiplex to continue."
                        ),
                    )

    messages.append(
        {"role": "assistant", "thinking": final_thinking, "content": final_answer}
    )
    blocked_reason_code = {
        1: "ambiguous_symbol",
        2: "ambiguous_symbol",
        3: "ambiguous_symbol",
        4: "ambiguous_symbol",
        6: "ambiguous_symbol",
        7: "no_multiplex_membership",
        9: "ambiguous_symbol",
        10: "tool_error",
    }.get(scenario.form)
    metadata = {
        "book_mode": "tool_call",
        "step": "S1",
        "question_family": contract["curriculum"]["families"]["S1.3"][
            "question_family"
        ],
        "form": scenario.form,
        "source_trace": scenario.source_trace,
        "split_unit": scenario.metadata_split_unit,
        "blocked_reason_code": blocked_reason_code,
        "schema_version": schema_version,
        "system_prompt_sha256": system_prompt_sha256(contract),
        "tool_contract_version": contract["runtime"]["tool_contract_version"],
        "ensembl_release": runtime.ensembl_release,
        "identifier_registry_id": runtime.identifier_registry_id,
    }
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


def _contains_value(text: str, value: str) -> bool:
    return re.search(
        r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    ) is not None


def _protected_values(row: Mapping[str, Any], scenario: Scenario) -> set[str]:
    values = {
        scenario.fact.gene_symbol,
        *scenario.fact.candidate_gene_ids,
        *scenario.fact.present_gene_ids,
        MULTIPLEX_ID,
        "BACKEND_UNAVAILABLE",
    }
    if scenario.distractor is not None:
        values.add(scenario.distractor.gene_symbol)
        values.update(scenario.distractor.candidate_gene_ids)
    return {value for value in values if len(value) >= 2}


def filter_row_thinking(
    row: Mapping[str, Any], scenario: Scenario, *,
    user_choice_reasoning: Mapping[str, str] | None = None,
    membership_retry_case: str | None = None,
    choice_placement: str | None = None,
    train_targeted_v10: bool = False,
) -> int:
    """Check visible evidence, including the explicitly permitted retry outcomes."""

    if membership_retry_case is not None:
        if scenario.split != "train" or scenario.form != 10:
            raise CorpusBuildError("membership retry thinking is scoped to training form 10")
        audit_membership_retry_row(row, membership_retry_case, framing=scenario.framing)
    messages = row["messages"]
    user_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    if not user_indexes:
        raise CorpusBuildError(f"row {row['row_id']} has no user message")
    current_turn = user_indexes[-1]
    protected = _protected_values(row, scenario)
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
                    f"row {row['row_id']} trains thinking in an earlier turn"
                )
            visible += " " + _canonical_json(
                message.get("content"), sort_keys=False
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
            scenario.split == "train" and scenario.form == 8 and user_choice_reasoning
            and message is messages[-1]
            and thinking == (
                _choice_thinking(
                    scenario, choice_placement, user_choice_reasoning,
                    train_targeted_v10=train_targeted_v10,
                )
                if choice_placement else user_choice_reasoning[scenario.framing].format(
                    gene_id=scenario.chosen_gene_id, symbol=scenario.fact.gene_symbol,
                )
            )
        ):
            # The reviewed thought explicitly preserves the earlier null graph channel.
            schema_text = schema_text.replace("null graph provenance", "")
        if any(phrase in schema_text for phrase in _ANSWER_SCHEMA_PHRASES):
            raise CorpusBuildError(
                f"row {row['row_id']} thinking discusses the answer "
                "schema/provenance"
            )
        for value in protected:
            if _contains_value(thinking, value) and not _contains_value(
                visible, value
            ):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking reveals {value!r} too early"
                )
        if "tool_call" not in message:
            answer = message.get("content")
            if not isinstance(answer, Mapping):
                raise CorpusBuildError(
                    f"row {row['row_id']} has no final answer object"
                )
            explicit_statuses = {
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
                explicit_statuses.add("not_found")
            if explicit_statuses.difference({str(answer["status"])}):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking has wrong status"
                )
            explicit_decisions = {
                decision
                for decision in ("continue", "stop", "defer")
                if re.search(rf"\b{decision}\b", lowered)
            }
            if explicit_decisions.difference(
                {str(answer["continuation_decision"])}
            ):
                raise CorpusBuildError(
                    f"row {row['row_id']} thinking has wrong decision"
                )
            if scenario.form == 10 and membership_retry_case != "recovered" and any(
                phrase in lowered
                for phrase in (
                    "neither candidate is present",
                    "neither candidate is absent",
                    "neither candidate is a node",
                    "no candidate is present",
                    "no candidate is a node",
                    "candidates are absent",
                )
            ):
                raise CorpusBuildError(
                    f"row {row['row_id']} claims absence after a failed call"
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
        raise CorpusBuildError("an S1.3 row has no current user text")
    return users[-1]


def masked_visible_prompt(
    row: Mapping[str, Any], scenario: Scenario
) -> str:
    """Mask source values but retain the full visible prompt structure."""

    messages = row["messages"]
    user_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    if not user_indexes:
        raise CorpusBuildError(f"row {row['row_id']} has no user message")
    last_user = user_indexes[-1]
    parts = []
    for message in messages[1 : last_user + 1]:
        role = str(message["role"])
        content = message.get("content")
        text = (
            content
            if isinstance(content, str)
            else _canonical_json(content, sort_keys=False)
        )
        text = " ".join(text.split())
        parts.append(f"{role}: {text}")
    visible = "\n".join(parts)
    visible = re.sub(
        r"ev_(?:reg|mem)_[a-z0-9]{6}",
        "<EVIDENCE_ID>",
        visible,
        flags=re.IGNORECASE,
    )

    facts = [scenario.fact]
    if scenario.distractor is not None:
        facts.append(scenario.distractor)
    identifiers = {scenario.input_value, MULTIPLEX_ID}
    source_values = {"116"}
    for fact in facts:
        identifiers.add(fact.gene_symbol)
        identifiers.update(fact.candidate_gene_ids)
        source_values.add(str(len(fact.candidate_gene_ids)))
        source_values.add(str(len(fact.present_gene_ids)))
        for details in fact.candidate_details.values():
            source_values.update(str(value) for value in details.values())
    if scenario.chosen_gene_id is not None:
        identifiers.add(scenario.chosen_gene_id)
    for value in sorted(identifiers, key=len, reverse=True):
        if value:
            visible = re.sub(
                r"(?<![A-Za-z0-9])"
                + re.escape(value)
                + r"(?![A-Za-z0-9])",
                MASK_TOKEN,
                visible,
                flags=re.IGNORECASE,
            )
    for value in sorted(source_values, key=len, reverse=True):
        if value:
            visible = re.sub(
                r"(?<![A-Za-z0-9])"
                + re.escape(value)
                + r"(?![A-Za-z0-9])",
                "<SOURCE_VALUE>",
                visible,
                flags=re.IGNORECASE,
            )
    return "\n".join(" ".join(line.split()) for line in visible.splitlines())


def _masked_user_wording(
    row: Mapping[str, Any],
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    choice_placement: str | None = None,
) -> str:
    user = _current_user_message(row)
    framing = scenario.framing
    if choice_placement:
        # Scope placement is audited separately; this classifier measures only
        # the shared outer wording, including its actual direct/step wrapper.
        if choice_placement == "earlier":
            framing = "direct"
        user = user.removeprefix(CHOICE_STANDALONE_SCOPE + " ")
    template = prompts[scenario.split][framing][
        scenario.template_index
    ]
    prefix, suffix = template.split("{request}")
    if not user.startswith(prefix) or (
        suffix and not user.endswith(suffix)
    ):
        raise CorpusBuildError(
            f"row {row['row_id']} does not match its shared prompt template"
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

    if tolerance < 0.0 or tolerance > 1.0:
        raise CorpusBuildError("the wording tolerance is invalid")
    train_rows = rows_by_split["train"]
    train_scenarios = scenarios_by_split["train"]
    if len(train_rows) != len(train_scenarios):
        raise CorpusBuildError("train rows and scenarios differ")
    labels = tuple(sorted({scenario.form for scenario in train_scenarios}))
    label_counts: Counter[int] = Counter()
    token_counts: dict[int, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[int] = Counter()
    vocabulary: set[str] = set()
    for row, scenario in zip(train_rows, train_scenarios, strict=True):
        tokens = _wording_tokens(masker(row, scenario))
        label_counts[scenario.form] += 1
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
    for split in ("train", "validation", "test"):
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} rows and scenarios differ")
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
            "masked user wording predicts the S1.3 form above the limit"
        )
    return result


def masked_wording_classifier_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    tolerance: float = MASKED_WORDING_TOLERANCE,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = IDENTITY_PROMPTS,
    train_form_prompts: Mapping[int, Mapping[str, Mapping[str, Sequence[str]]]] | None = None,
    train_choice_assignments: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Gate leakage from the shared outer identity wrappers."""

    return _wording_classifier(
        rows_by_split,
        scenarios_by_split,
        masker=lambda row, scenario: _masked_user_wording(
            row, scenario,
            prompts=((train_form_prompts or {}).get(scenario.form, prompts) if scenario.split == "train" else prompts),
            choice_placement=(train_choice_assignments or {}).get(scenario.scenario_id, (None, 0))[0],
        ),
        scope="outer_wrapper_only",
        gated=True,
        tolerance=tolerance,
    )


def full_visible_prompt_classifier_diagnostic(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
) -> dict[str, Any]:
    """Measure form cues in the full masked prompt without a pass gate."""

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


def _failure_pair_prompts_match(
    failure: Scenario, failure_row: Mapping[str, Any],
    success: Scenario, success_row: Mapping[str, Any], *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]],
    train_form_prompts: Mapping[int, Mapping[str, Mapping[str, Sequence[str]]]] | None,
) -> bool:
    """Allow only the declared training wrapper difference in a failure pair."""

    for field in ("split", "fact", "input_value", "framing", "template_index", "multiplex_in_harness"):
        if getattr(failure, field) != getattr(success, field):
            return False
    prefixes = []
    for scenario, row in ((failure, failure_row), (success, success_row)):
        prefix = json.loads(_prompt_and_outcome_signatures(row)[0])
        if failure.split == "train" and success.form in {5, 6, 7} and success.form in (train_form_prompts or {}):
            # Validate each original wrapper before normalizing just its text.
            bank = (train_form_prompts or {}).get(scenario.form, prompts)
            if prefix[-1]["content"] != _prompt(scenario, _initial_request(scenario), prompts=bank):
                return False
            prefix[-1]["content"] = _prompt(scenario, _initial_request(scenario), prompts=prompts)
        prefixes.append(prefix)
    return prefixes[0] == prefixes[1]


def _membership_prompt_cue(
    row: Mapping[str, Any], scenario: Scenario, *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]],
    train_form_prompts: Mapping[int, Mapping[str, Mapping[str, Sequence[str]]]] | None,
) -> str:
    """Compare membership context after validating a reviewed membership wrapper."""

    if scenario.split == "train" and scenario.form in {5, 6, 7, 10} and scenario.form in (train_form_prompts or {}):
        bank = train_form_prompts[scenario.form]
        request = _initial_request(scenario)
        if _current_user_message(row) != _prompt(scenario, request, prompts=bank):
            raise CorpusBuildError("a membership prompt changed its pinned wrapper")
        messages = [dict(message) for message in row["messages"]]
        user_index = next(i for i, message in enumerate(messages) if message["role"] == "user")
        messages[user_index]["content"] = _prompt(scenario, request, prompts=prompts)
        row = {**row, "messages": messages}
    return masked_visible_prompt(row, scenario)


def duplicate_prompt_outcome_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
) -> dict[str, Any]:
    """Reject repeated prompt outcomes except the required failure pairs."""

    split_results: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
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
                f"{split} repeats a prompt and tool outcome after removing "
                "evidence IDs"
            )
        pair_count = 0
        for items in prompt_groups.values():
            if len(items) == 1:
                continue
            if len(items) != 2:
                raise CorpusBuildError(f"{split} has a prompt shared by many rows")
            failure = next((item for item in items if item[0].form == 10), None)
            success = next((item for item in items if item[0].form in {5, 6, 7}), None)
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
        "normalization": "remove evidence IDs and retain tool names and statuses",
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
    membership_answers: Mapping[str, Any] | None = None,
    train_form_prompts: Mapping[int, Mapping[str, Mapping[str, Sequence[str]]]] | None = None,
    user_choice_reasoning: Mapping[str, str] | None = None,
    train_retry_cases: Mapping[str, str] | None = None,
    train_choice_assignments: Mapping[str, tuple[str, int]] | None = None,
    train_targeted_v10: bool = False,
) -> dict[str, Any]:
    """Check S1.3 rules that the generic row validator cannot check."""

    framing_shares = framing_shares or {"direct": 0.5, "step": 0.5}
    expected_calls = {
        1: [SYMBOL_TOOL],
        2: [SYMBOL_TOOL],
        3: [],
        4: [SYMBOL_TOOL],
        5: [SYMBOL_TOOL, MEMBERSHIP_TOOL],
        6: [SYMBOL_TOOL, MEMBERSHIP_TOOL],
        7: [SYMBOL_TOOL, MEMBERSHIP_TOOL],
        8: [],
        9: [],
        10: [SYMBOL_TOOL, MEMBERSHIP_TOOL],
    }
    expected_status = {
        1: "ambiguous",
        2: "ambiguous",
        3: "ambiguous",
        4: "ambiguous",
        5: "resolved",
        6: "ambiguous",
        7: "not_found",
        8: "resolved",
        9: "ambiguous",
        10: "ambiguous",
    }
    detail_forms = {1, 2, 3, 4, 6, 9, 10}
    if membership_answers is not None:
        expected_status[7] = membership_answers["none_present_status"]
        if membership_answers["keep_tool_fields"]:
            detail_forms |= {5, 7}
    row_ids: set[str] = set()
    units_by_split: dict[str, set[str]] = {}
    status_counts: dict[str, dict[str, int]] = {}
    call_counts: dict[str, dict[str, int]] = {}
    framing_counts: dict[str, dict[str, int]] = {}
    case_counts: dict[str, dict[str, int]] = {}
    pair_counts: dict[str, int] = {}
    outcome_counts: dict[str, dict[str, int]] = {}
    placement_counts: dict[str, dict[str, dict[str, int]]] = {}
    reply_counts: dict[str, dict[str, dict[str, int]]] = {}
    false_defer_counts: dict[str, dict[str, int | float]] = {}
    registry_harness_counts: dict[str, dict[str, int]] = {}
    earlier_turn_counts: dict[str, dict[str, int]] = {}
    membership_cue_counts: dict[str, dict[str, int]] = {}
    raw_membership_cue_counts: dict[str, dict[str, int]] = {}
    successful_membership_cue_counts: dict[str, dict[str, int]] = {}
    template_counts: dict[str, dict[str, int]] = {}
    thinking_counts: dict[str, dict[str, int]] = {}
    targeted_reply_count = max((index for _, index in (train_choice_assignments or {}).values()), default=3) + 1

    for split in ("train", "validation", "test"):
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} rows and scenarios differ")
        forms = Counter(int(row["metadata"]["form"]) for row in rows)
        if dict(sorted(forms.items())) != dict(expected_counts[split]):
            raise CorpusBuildError(f"{split} form counts changed")
        expected_frames = largest_remainder_counts(
            len(rows), framing_shares
        )
        actual_frames = Counter(item.framing for item in scenarios)
        if dict(actual_frames) != expected_frames:
            raise CorpusBuildError(
                f"{split} framing is {dict(actual_frames)!r}, expected "
                f"{expected_frames!r}"
            )

        successes = {
            scenario.scenario_id: (scenario, row)
            for scenario, row in zip(scenarios, rows, strict=True)
            if scenario.form in {5, 6, 7}
        }
        units: set[str] = set()
        thinking_steps = 0
        changed_case = 0
        failures = 0
        membership_outcomes: Counter[str] = Counter()
        multiplex_placements: Counter[str] = Counter()
        reply_variants: Counter[str] = Counter()
        answerable_rows = 0
        false_defers = 0
        registry_counts: Counter[str] = Counter()
        earlier_counts: Counter[str] = Counter()
        membership_cues: dict[str, set[str]] = {
            "keeps_one": set(),
            "keeps_several": set(),
            "keeps_none": set(),
        }
        raw_membership_cues = {name: set() for name in membership_cues}
        successful_membership_cues = {name: set() for name in membership_cues}
        for scenario, row in zip(scenarios, rows, strict=True):
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise CorpusBuildError(f"duplicate row ID {row_id}")
            row_ids.add(row_id)
            units.update(scenario.split_units)
            retry_case = (train_retry_cases or {}).get(scenario.scenario_id) if split == "train" else None
            recovered = retry_case == "recovered"
            if retry_case:
                audit_membership_retry_row(row, retry_case, framing=scenario.framing)
                if recovered and row["messages"][-1]["content"]["result"]["present_gene_ids"] != list(scenario.fact.present_gene_ids):
                    raise CorpusBuildError("membership recovery changed the source outcome")
            thinking_steps += filter_row_thinking(
                row, scenario, user_choice_reasoning=user_choice_reasoning, membership_retry_case=retry_case,
                choice_placement=(train_choice_assignments or {}).get(scenario.scenario_id, (None, 0))[0],
                train_targeted_v10=train_targeted_v10,
            )
            changed_case += scenario.input_value != scenario.fact.gene_symbol

            registry_harness = any(
                message.get("role") == "harness"
                and str(message.get("content", "")).startswith(
                    "Registry entries, Ensembl release 116:"
                )
                for message in row["messages"]
            )
            expects_registry_harness = scenario.form in {3, 4}
            if registry_harness != expects_registry_harness:
                registry_counts["unexpected"] += 1
            elif registry_harness:
                registry_counts[f"S1.3.{scenario.form}"] += 1

            users = [
                index
                for index, message in enumerate(row["messages"])
                if message.get("role") == "user"
            ]
            has_earlier_turn = (
                len(users) == 2
                and any(
                    message.get("role") == "assistant"
                    for message in row["messages"][users[0] + 1 : users[1]]
                )
            )
            expects_earlier_turn = scenario.form in {8, 9}
            if has_earlier_turn != expects_earlier_turn:
                earlier_counts["unexpected"] += 1
            elif has_earlier_turn:
                earlier_counts[f"S1.3.{scenario.form}"] += 1

            if scenario.form in {5, 6, 7, 10}:
                raw_membership_cues[scenario.fact.membership_outcome].add(
                    masked_visible_prompt(row, scenario)
                )
                if scenario.form != 10:
                    successful_membership_cues[scenario.fact.membership_outcome].add(
                        masked_visible_prompt(row, scenario)
                    )
                membership_cues[scenario.fact.membership_outcome].add(
                    _membership_prompt_cue(
                        row, scenario, prompts=prompts, train_form_prompts=train_form_prompts,
                    )
                )

            calls = _tool_calls(row)
            arguments = _tool_arguments(row)
            wanted_calls = expected_calls[scenario.form] + ([MEMBERSHIP_TOOL] if retry_case in {"recovered", "exhausted"} else [])
            if calls != wanted_calls:
                raise CorpusBuildError(
                    f"form {scenario.form} emitted calls {calls!r}"
                )
            if calls and arguments[0] != {"gene_symbol": scenario.input_value}:
                raise CorpusBuildError(
                    f"form {scenario.form} changed its symbol argument"
                )
            if len(calls) >= 2 and arguments[1] != {
                "multiplex_id": MULTIPLEX_ID,
                "gene_ids": list(scenario.fact.candidate_gene_ids),
            }:
                raise CorpusBuildError(
                    f"form {scenario.form} changed its candidate set"
                )

            answer = row["messages"][-1]["content"]
            answerable = scenario.form in {5, 8} or (recovered and len(scenario.fact.present_gene_ids) == 1)
            answerable_rows += int(answerable)
            false_defers += int(
                answerable and answer["continuation_decision"] == "defer"
            )
            wanted_status = "resolved" if recovered and len(scenario.fact.present_gene_ids) == 1 else expected_status[scenario.form]
            if answer["status"] != wanted_status:
                raise CorpusBuildError(
                    f"form {scenario.form} emitted status {answer['status']!r}"
                )
            result = answer["result"]
            if list(result["candidate_gene_ids"]) != list(
                scenario.fact.candidate_gene_ids
            ):
                raise CorpusBuildError(
                    f"form {scenario.form} changed its candidate IDs"
                )
            if scenario.form in detail_forms and result.get(
                "candidate_details"
            ) != scenario.fact.candidate_details:
                raise CorpusBuildError(
                    f"form {scenario.form} changed candidate details"
                )
            if (
                membership_answers is not None
                and membership_answers["keep_tool_fields"]
                and scenario.form in {5, 6, 7}
                and set(result["absent_gene_ids"])
                != set(scenario.fact.candidate_gene_ids).difference(
                    scenario.fact.present_gene_ids
                )
            ):
                raise CorpusBuildError(
                    f"form {scenario.form} changed absent candidates"
                )
            if scenario.form == 5:
                if (
                    result["gene_id"]
                    != scenario.fact.present_gene_ids[0]
                    or result["present_gene_ids"]
                    != list(scenario.fact.present_gene_ids)
                ):
                    raise CorpusBuildError("form 5 chose the wrong candidate")
            elif scenario.form == 6:
                if result["present_gene_ids"] != list(
                    scenario.fact.present_gene_ids
                ):
                    raise CorpusBuildError("form 6 changed present candidates")
            elif scenario.form == 7:
                if result["present_gene_ids"] != []:
                    raise CorpusBuildError("form 7 reports a present candidate")
            elif scenario.form == 8:
                if result["gene_id"] not in scenario.fact.candidate_gene_ids:
                    raise CorpusBuildError("form 8 chose a noncandidate")
            elif scenario.form in {1, 2, 3, 4, 9} and "gene_id" in result:
                raise CorpusBuildError(
                    f"form {scenario.form} chose an ambiguous candidate"
                )
            elif scenario.form == 10 and not recovered:
                if (
                    result.get("failed_tool") != MEMBERSHIP_TOOL
                    or result.get("error_code") != "BACKEND_UNAVAILABLE"
                    or answer["provenance"]["graph"] is not None
                    or "present_gene_ids" in result
                ):
                    raise CorpusBuildError("form 10 changed partial-failure semantics")

            if scenario.form in {5, 6, 7, 10}:
                membership_outcomes[scenario.fact.membership_outcome] += 1
                placement = (
                    "harness"
                    if scenario.multiplex_in_harness
                    else "user_task"
                )
                multiplex_placements[f"S1.3.{scenario.form}:{placement}"] += 1
            if scenario.form in {8, 9}:
                targeted_reply = (train_choice_assignments or {}).get(scenario.scenario_id)
                reply_index = targeted_reply[1] if targeted_reply else scenario.reply_index
                reply_count = targeted_reply_count if targeted_reply else 4
                if reply_index not in range(reply_count):
                    raise CorpusBuildError(
                        f"form {scenario.form} has no valid reply index"
                    )
                reply_variants[
                    f"S1.3.{scenario.form}:{reply_index}"
                ] += 1
            if scenario.form == 10:
                pair = successes.get(str(scenario.paired_success_id))
                if pair is None:
                    raise CorpusBuildError("a failure has no retained success")
                success_scenario, success_row = pair
                if not _failure_pair_prompts_match(
                    scenario, row, success_scenario, success_row,
                    prompts=prompts, train_form_prompts=train_form_prompts,
                ):
                    raise CorpusBuildError("a failure changed its success prompt")
                failures += 1

        if failures != expected_counts[split][10]:
            raise CorpusBuildError(f"{split} failure-pair count changed")
        units_by_split[split] = units
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
        framing_counts[split] = dict(sorted(actual_frames.items()))
        case_counts[split] = {
            "eligible": len(scenarios),
            "perturbed": changed_case,
        }
        pair_counts[split] = failures
        outcome_counts[split] = dict(sorted(membership_outcomes.items()))
        split_placements: dict[str, dict[str, int]] = {}
        for form in (5, 6, 7, 10):
            form_counts_by_placement = {}
            for placement in ("harness", "user_task"):
                key = f"S1.3.{form}:{placement}"
                if multiplex_placements[key] < 1:
                    raise CorpusBuildError(
                        f"{split} form {form} has no {placement}-named multiplex"
                    )
                form_counts_by_placement[placement] = multiplex_placements[key]
            split_placements[f"S1.3.{form}"] = form_counts_by_placement
        placement_counts[split] = split_placements
        split_replies: dict[str, dict[str, int]] = {}
        for form in (8, 9):
            form_reply_counts = {}
            reply_count = targeted_reply_count if split == "train" and form == 8 and train_choice_assignments else 4
            for reply_index in range(reply_count):
                key = f"S1.3.{form}:{reply_index}"
                if reply_variants[key] < 1:
                    raise CorpusBuildError(
                        f"{split} form {form} misses reply {reply_index}"
                    )
                form_reply_counts[str(reply_index)] = reply_variants[key]
            split_replies[f"S1.3.{form}"] = form_reply_counts
        reply_counts[split] = split_replies
        registry_result = {
            "S1.3.3": registry_counts["S1.3.3"],
            "S1.3.4": registry_counts["S1.3.4"],
            "unexpected": registry_counts["unexpected"],
        }
        if (
            registry_result["S1.3.3"] != expected_counts[split][3]
            or registry_result["S1.3.4"] != expected_counts[split][4]
            or registry_result["unexpected"]
        ):
            raise CorpusBuildError(f"{split} registry harness structure changed")
        registry_harness_counts[split] = registry_result

        earlier_result = {
            "S1.3.8": earlier_counts["S1.3.8"],
            "S1.3.9": earlier_counts["S1.3.9"],
            "unexpected": earlier_counts["unexpected"],
        }
        if (
            earlier_result["S1.3.8"] != expected_counts[split][8]
            or earlier_result["S1.3.9"] != expected_counts[split][9]
            or earlier_result["unexpected"]
        ):
            raise CorpusBuildError(f"{split} earlier-turn structure changed")
        earlier_turn_counts[split] = earlier_result

        shared_cues = set.intersection(*membership_cues.values())
        cue_result = {
            name: len(cues) for name, cues in membership_cues.items()
        }
        cue_result["shared_cue_count"] = len(shared_cues)
        expected_cue_count = sum(
            len(prompts[split][framing])
            for framing in ("direct", "step")
        ) * 2
        if any(
            cues != shared_cues for cues in membership_cues.values()
        ) or len(shared_cues) != expected_cue_count:
            raise CorpusBuildError(
                f"{split} has a pre-tool membership-outcome cue"
            )
        membership_cue_counts[split] = cue_result
        if train_form_prompts and 5 in train_form_prompts:
            raw_membership_cue_counts[split] = {
                **{name: len(cues) for name, cues in raw_membership_cues.items()},
                "shared_cue_count": len(set.intersection(*raw_membership_cues.values())),
            }
        if (
            train_form_prompts and {5, 6, 7} <= train_form_prompts.keys()
            and train_form_prompts[5] == train_form_prompts[6] == train_form_prompts[7]
        ):
            # The successful outcomes must share the same visible wrapper coverage.
            shared_successful_cues = set.intersection(*successful_membership_cues.values())
            if any(cues != shared_successful_cues for cues in successful_membership_cues.values()) or len(shared_successful_cues) != expected_cue_count:
                raise CorpusBuildError(f"{split} has a raw successful membership-outcome cue")
            successful_membership_cue_counts[split] = {
                **{name: len(cues) for name, cues in successful_membership_cues.items()},
                "shared_cue_count": len(shared_successful_cues),
            }
        if train_form_prompts and {5, 6, 7, 10} <= train_form_prompts.keys() and all(
            train_form_prompts[form] == train_form_prompts[5] for form in (6, 7, 10)
        ):
            shared_raw = set.intersection(*raw_membership_cues.values())
            if len(shared_raw) != expected_cue_count or any(cues != shared_raw for cues in raw_membership_cues.values()):
                raise CorpusBuildError(f"{split} has a raw membership-outcome cue")
        if false_defers:
            raise CorpusBuildError(
                f"{split} has {false_defers} false-defer rows"
            )
        false_defer_counts[split] = {
            "eligible_row_count": answerable_rows,
            "false_defer_count": false_defers,
            "false_defer_rate": false_defers / answerable_rows,
            "pass": false_defers == 0,
        }
        thinking_counts[split] = {
            "rows_checked": len(rows),
            "thinking_steps_checked": thinking_steps,
            "violations": 0,
        }
        coverage = Counter(
            f"S1.3.{scenario.form}:{scenario.framing}:"
            f"{scenario.template_index}"
            for scenario in scenarios
        )
        template_counts[split] = dict(sorted(coverage.items()))
        for form in range(1, 11):
            frames = (
                ("direct",)
                if form == 1
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
                        f"{split} form {form} {framing} uses templates "
                        f"{sorted(used)!r}, expected {sorted(expected)!r}"
                    )

    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = units_by_split[first].intersection(units_by_split[second])
        if overlap:
            raise CorpusBuildError(
                f"identity leakage between {first} and {second}: "
                f"{sorted(overlap)[:3]!r}"
            )

    duplicate_audit = duplicate_prompt_outcome_audit(
        rows_by_split, scenarios_by_split
    )
    wording_audit = masked_wording_classifier_audit(
        rows_by_split, scenarios_by_split, prompts=prompts,
        train_form_prompts=train_form_prompts,
        train_choice_assignments=train_choice_assignments,
    )
    full_prompt_diagnostic = full_visible_prompt_classifier_diagnostic(
        rows_by_split, scenarios_by_split
    )
    total_answerable = sum(
        int(values["eligible_row_count"])
        for values in false_defer_counts.values()
    )
    total_false_defers = sum(
        int(values["false_defer_count"])
        for values in false_defer_counts.values()
    )
    return {
        "status_counts": status_counts,
        "tool_call_counts": call_counts,
        "framing_counts": framing_counts,
        "case_counts": case_counts,
        "failure_pair_counts": pair_counts,
        "membership_outcome_counts": outcome_counts,
        "multiplex_placement_counts": placement_counts,
        "reply_template_counts": reply_counts,
        "false_defer_audit": {
            "definition": "a row defers although the task is answerable",
            "eligible_row_count": total_answerable,
            "false_defer_count": total_false_defers,
            "false_defer_rate": total_false_defers / total_answerable,
            "splits": false_defer_counts,
            "pass": total_false_defers == 0,
        },
        "prompt_template_counts": template_counts,
        "prompt_structure_audit": {
            "registry_harness_counts": registry_harness_counts,
            "earlier_turn_counts": earlier_turn_counts,
            "membership_pretool_cue_counts": membership_cue_counts,
            **({"membership_pretool_raw_cue_counts": raw_membership_cue_counts}
               if raw_membership_cue_counts else {}),
            **({"successful_membership_pretool_raw_cue_counts": successful_membership_cue_counts}
               if successful_membership_cue_counts else {}),
            "pass": True,
        },
        "thinking_filter": {
            "rules": list(THINKING_FILTER_RULES),
            "splits": thinking_counts,
            "pass": True,
        },
        "duplicate_prompt_tool_outcome": duplicate_audit,
        "masked_wording_classifier": wording_audit,
        "full_visible_prompt_classifier": full_prompt_diagnostic,
        "split_unit_hashes": {
            split: _stable_sha256(sorted(units))
            for split, units in units_by_split.items()
        },
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row, sort_keys=False) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validate_corpus(
    path: Path, overlay: Mapping[str, str] | None = None
) -> None:
    """Run the semantic validator, against the overlaid contract if any."""

    validator = (
        REPOSITORY_ROOT
        / "agents"
        / "world_model_v3_curriculum"
        / "s1_kit"
        / "validate.py"
    )
    overlay_args = (
        ["--overlay", str(REPOSITORY_ROOT / overlay["path"])]
        if overlay is not None
        else []
    )
    result = subprocess.run(
        [sys.executable, str(validator), *overlay_args, str(path)],
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
    """Build and validate S1.3, then publish its manifest last.

    A pinned prompt bank replaces only the training wrappers and assigns them
    round-robin; validation and test keep their wrappers and assignment, so
    their rows do not change. The S1 v5 overlay also sets the S1.3.5 to
    S1.3.7 answers in every split, and the training reasoning and form counts.
    """

    source_index_path = source_index_path.resolve()
    payload = check_source_index(source_index_path, contract)
    curriculum = contract["curriculum"]
    settings = curriculum["families"]["S1.3"]
    seed = int(settings["assignment_seed"])
    prompts = (
        identity_prompts(
            prompt_banks, IDENTITY_PROMPTS, bank_name="identity_s1_3"
        )
        if prompt_banks is not None
        else IDENTITY_PROMPTS
    )
    balanced_splits = ("train",) if prompt_banks is not None else ()
    train_form_pins = settings.get("train_prompt_banks", {})
    train_form_prompts = {
        form: identity_prompts(pin, IDENTITY_PROMPTS, bank_name="identity_s1_3")
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
    single_membership_reasoning = settings.get("train_single_membership_reasoning")
    multiple_membership_reasoning = settings.get("train_multiple_membership_reasoning")
    no_membership_reasoning = settings.get("train_no_membership_reasoning")
    user_choice_reasoning = settings.get("train_user_choice_reasoning")
    train_targeted_v9 = bool(settings.get("train_targeted_v9"))
    train_targeted_v10 = bool(settings.get("train_targeted_v10"))
    if train_targeted_v10 and not train_targeted_v9:
        raise CorpusBuildError("v10 user-choice revision requires the v9 paired scope assignment")
    choice_pin = settings.get("train_choice_replies")
    choice_replies = load_choice_replies(choice_pin) if train_targeted_v9 and choice_pin else ()
    if train_targeted_v9 and (not choice_replies or user_choice_reasoning is None):
        raise CorpusBuildError("targeted user-choice revision needs pinned replies and reviewed reasoning")
    membership_answers = settings.get("membership_answers")
    train_multiplex_reasoning = bool(settings.get("train_multiplex_reasoning"))
    row_counts = split_row_counts(
        int(settings["train_rows"]), curriculum["split_fractions"]
    )
    shared_counts = {
        split: form_counts(row_counts[split], settings, curriculum)
        for split in ("train", "validation", "test")
    }
    counts_by_split = dict(shared_counts)
    if "train_form_counts" in settings:
        counts_by_split["train"] = apply_train_form_counts(
            shared_counts["train"], settings["train_form_counts"]
        )
    pools, source_outcomes = build_fact_pools(
        payload,
        s0_settings=curriculum["families"]["S0"],
        maximum_candidate_gene_ids=int(settings["maximum_candidate_gene_ids"]),
    )
    pool_coverage = source_pool_coverage(pools)
    expected_source_outcomes = {
        str(name): int(count)
        for name, count in settings["membership_outcome_counts"].items()
    }
    if source_outcomes != expected_source_outcomes:
        raise CorpusBuildError(
            f"membership source outcomes are {source_outcomes!r}, expected "
            f"{expected_source_outcomes!r}"
        )
    eligible_count = len(
        {
            fact.fact_id
            for split_pools in pools.values()
            for fact in split_pools["all"]
        }
    )
    source_count = sum(source_outcomes.values())
    excluded_over_cap = source_count - eligible_count
    if excluded_over_cap < 0:
        raise CorpusBuildError("the eligible ambiguity count exceeds its source")

    scenarios_by_split = select_scenarios(
        pools,
        counts_by_split,
        seed=seed,
        case_rate=curriculum["identity_case_perturbation"],
        framing_shares=curriculum["framing"],
        prompts=prompts,
        balanced_splits=balanced_splits,
        train_decorrelate=bool(settings.get("train_decorrelate")),
    )
    choice_assignments = (
        _choice_assignments(scenarios_by_split["train"], seed=seed, reply_count=len(choice_replies))
        if train_targeted_v9 else {}
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
                membership_answers=membership_answers,
                train_multiplex_reasoning=train_multiplex_reasoning,
                final_reasoning=train_final_reasoning.get(scenario.form),
                single_membership_reasoning=single_membership_reasoning,
                multiple_membership_reasoning=multiple_membership_reasoning,
                no_membership_reasoning=no_membership_reasoning,
                user_choice_reasoning=user_choice_reasoning,
            )
            if scenario.scenario_id in choice_assignments:
                placement, reply_index = choice_assignments[scenario.scenario_id]
                earlier, latest = _choice_users(
                    scenario, placement,
                    choice_replies[reply_index].format(gene_id=scenario.chosen_gene_id),
                    prompts,
                    train_targeted_v10=train_targeted_v10,
                )
                users = [message for message in row["messages"] if message["role"] == "user"]
                users[0]["content"], users[1]["content"] = earlier, latest
                earlier_answer = row["messages"][2]["content"]
                earlier_answer["summary"] = _candidate_summary(
                    earlier_answer["result"], suffix=CHOICE_AMBIGUITY_SUFFIX,
                )
                row["messages"][-1]["thinking"] = _choice_thinking(
                    scenario, placement, user_choice_reasoning,
                    train_targeted_v10=train_targeted_v10,
                )
            rows.append(row)
            evidence_calls.extend(row_evidence)
        rows_by_split[split] = rows

    retry_pin = settings.get("train_membership_retries")
    retry_cases = {}
    retry_coverage = Counter()
    if retry_pin:
        policy = load_membership_retry_policy(retry_pin)
        scenarios = scenarios_by_split["train"]
        retry_cases = assign_membership_retry_cases([s for s in scenarios if s.form == 10])
        if Counter(retry_cases.values()) != policy["case_counts"]:
            raise CorpusBuildError("membership retry allocation changed")
        successes = {s.scenario_id: row for s, row in zip(scenarios, rows_by_split["train"], strict=True) if s.form in {5, 6, 7}}
        for index, scenario in enumerate(scenarios):
            if scenario.form != 10:
                continue
            case = retry_cases[scenario.scenario_id]
            row, evidence = render_membership_retry_case(
                rows_by_split["train"][index], success=successes[scenario.paired_success_id],
                case=case, runtime=runtime, policy=policy,
            )
            rows_by_split["train"][index] = row
            evidence_calls.extend(evidence)
            retry_coverage[f"{scenario.fact.membership_outcome}:{case}:{scenario.framing}:{'harness' if scenario.multiplex_in_harness else 'task'}"] += 1
        if len(retry_coverage) != 36:
            raise CorpusBuildError("membership retry coverage lost an outcome/framing/placement branch")

    evidence_ids = [item[0] for item in evidence_calls]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise CorpusBuildError("an evidence ID appears in more than one row")
    choice_audit = (
        _audit_choice_scope(
            rows_by_split["train"], scenarios_by_split["train"],
            assignments=choice_assignments, replies=choice_replies,
            prompts=prompts, reasoning=user_choice_reasoning,
            train_targeted_v10=train_targeted_v10,
        ) if train_targeted_v9 else None
    )
    audit = audit_rows(
        rows_by_split,
        scenarios_by_split,
        counts_by_split,
        framing_shares=curriculum["framing"],
        prompts=prompts,
        membership_answers=membership_answers,
        train_form_prompts=train_form_prompts,
        user_choice_reasoning=user_choice_reasoning,
        train_retry_cases=retry_cases,
        train_choice_assignments=choice_assignments,
        train_targeted_v10=train_targeted_v10,
    )

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".s1_3_build_", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        paths = {
            split: temporary / f"{split}.jsonl"
            for split in ("train", "validation", "test")
        }
        for split, path in paths.items():
            _write_jsonl(path, rows_by_split[split])
            if validate:
                _validate_corpus(path, overlay)

        form_count_manifest = {
            split: {
                f"S1.3.{form}": count
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
            "section": "S1.3",
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
                "row_schema_version": contract["runtime"]["row_schema_version"],
                "tool_contract_version": contract["runtime"][
                    "tool_contract_version"
                ],
            },
            "sources": {
                "source_index_sha256": runtime.source_index_sha256,
                "identifier_registry_id": runtime.identifier_registry_id,
                "ambiguous_symbol_count": source_count,
                "membership_outcome_counts": expected_source_outcomes,
                "maximum_candidate_gene_ids": int(
                    settings["maximum_candidate_gene_ids"]
                ),
                "eligible_ambiguous_symbol_count": eligible_count,
                "excluded_over_cap_count": excluded_over_cap,
                **pool_coverage,
                "records": payload["sources"],
            },
            "assignment": {
                "seed": seed,
                "split_fractions": dict(curriculum["split_fractions"]),
                "split_unit_sha256": audit["split_unit_hashes"],
                "source_pool_counts": pool_counts,
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
            "tool_call_counts": audit["tool_call_counts"],
            "framing_counts": audit["framing_counts"],
            "membership_outcome_counts": audit[
                "membership_outcome_counts"
            ],
            "multiplex_placement_counts": audit[
                "multiplex_placement_counts"
            ],
            "reply_template_counts": audit["reply_template_counts"],
            "false_defer_audit": audit["false_defer_audit"],
            "prompt_structure_audit": audit[
                "prompt_structure_audit"
            ],
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
                f"{split}.jsonl": {
                    "row_count": row_counts[split],
                    "sha256": file_sha256(path),
                }
                for split, path in paths.items()
            },
        }
        if retry_pin:
            manifest["train_membership_retries"] = {
                **dict(retry_pin), "case_counts": dict(Counter(retry_cases.values())),
                "assignment_sha256": _stable_sha256(retry_cases),
                "coverage": dict(sorted(retry_coverage.items())),
                "implementation_sha256": file_sha256(Path(__file__).with_name("s1_membership_retry.py")),
            }
        if overlay is not None:
            manifest["contract"]["s1_overlay"] = dict(overlay)
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
        if single_membership_reasoning is not None:
            manifest["train_single_membership_reasoning"] = dict(single_membership_reasoning)
        if multiple_membership_reasoning is not None:
            manifest["train_multiple_membership_reasoning"] = dict(multiple_membership_reasoning)
        if no_membership_reasoning is not None:
            manifest["train_no_membership_reasoning"] = dict(no_membership_reasoning)
        if user_choice_reasoning is not None:
            manifest["train_user_choice_reasoning"] = dict(user_choice_reasoning)
        if choice_audit is not None:
            manifest["train_targeted_v10" if train_targeted_v10 else "train_targeted_v9"] = {
                "choice_replies": dict(choice_pin), **choice_audit,
            }
            if train_targeted_v10:
                manifest["train_targeted_v10"]["standalone_scope"] = "no later work requested in either user turn"
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
    settings = contract["curriculum"]["families"]["S1.3"]
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
