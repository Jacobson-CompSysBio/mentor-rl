"""Focused tests for the canonical S1.4 alignment corpus builder."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s0_corpus import EvidenceSequence
from runtime.world_model.s1_4_corpus import (
    ALIGNMENT_PROMPTS,
    AmbiguousFact,
    CorpusBuildError,
    ExternalFact,
    GeneFact,
    Scenario,
    build_fact_pools,
    build_row,
    duplicate_prompt_outcome_audit,
    filter_row_thinking,
    form_counts,
    full_visible_prompt_classifier_diagnostic,
    largest_remainder_counts,
    masked_visible_prompt,
    masked_wording_classifier_audit,
    select_scenarios,
    source_pool_coverage,
    split_row_counts,
)
from runtime.world_model.source_index import check_source_index
from runtime.world_model.sources import (
    file_sha256,
    read_ensembl_gtf_identifiers,
)
from runtime.world_model.tools import (
    GENE_ID_TOOL,
    SYMBOL_TOOL,
    WorldModelToolRuntime,
)
from runtime.world_model_training import (
    flatten_v3_trace_for_arrow,
    tokenize_v3_trace_for_sft,
)


SOURCE_INDEX = Path(
    "data/world_model_v3/source_indexes/"
    "world_model_source_index_"
    "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e.json.gz"
)
MYGENE_SOURCE = Path(
    "data/runtime/alias_registry/mygene_query_cache.json"
)
MYGENE_SHA256 = (
    "44f2fbcea56595caea5f59651532cfea50ce033296ce441e3b539a9f5859740c"
)
ITEMS = {
    "typed_seed_list",
    "seed_gene_set",
    "module",
    "abstract",
    "query_mygene",
}
EXPECTED_SOURCE_COVERAGE = {
    "pool_counts": {
        "train": {
            "resolved": 41067,
            "current": 78220,
            "near_miss": 208105,
            "historical": 54923,
            "ambiguous": 385,
            "external": 4574,
        },
        "validation": {
            "resolved": 144,
            "current": 366,
            "near_miss": 10,
            "historical": 231,
            "ambiguous": 48,
            "external": 13,
        },
        "test": {
            "resolved": 133,
            "current": 355,
            "near_miss": 12,
            "historical": 227,
            "ambiguous": 48,
            "external": 8,
        },
    },
    "ambiguous_candidate_count_distribution": {
        "train": {
            "2": 363,
            "3": 6,
            "4": 4,
            "6": 1,
            "7": 5,
            "9": 1,
            "10": 1,
            "21": 1,
            "22": 1,
            "27": 1,
            "37": 1,
        },
        "validation": {
            "2": 44,
            "3": 1,
            "4": 1,
            "7": 1,
            "8": 1,
        },
        "test": {"2": 47, "5": 1},
    },
}


def _gene_id(value: int) -> str:
    return f"ENSG{value:011d}"


def _fact(
    value: int,
    *,
    symbol: str,
    split: str = "train",
    gene_id: str | None = None,
    version: int | None = 3,
    units: tuple[str, ...] | None = None,
) -> GeneFact:
    return GeneFact(
        fact_id=f"fact_{split}_{symbol}_{value}",
        split_units=units or (f"unit_{split}_{value}",),
        gene_symbol=symbol,
        gene_id=gene_id if gene_id is not None else _gene_id(value),
        gene_version=version,
    )


def _historical(
    value: int,
    *,
    symbol: str,
    current_symbol: str = "GENEA",
    split: str = "train",
    linked_unit: str | None = None,
) -> ExternalFact:
    return ExternalFact(
        fact_id=f"historical_{split}_{symbol}_{value}",
        split_units=(linked_unit or f"unit_{split}_{value}",),
        query=symbol,
        gene_symbol=current_symbol,
        gene_id=_gene_id(value),
    )


def _ambiguous(
    value: int,
    *,
    symbol: str,
    split: str = "train",
    candidate_count: int = 2,
) -> AmbiguousFact:
    candidate_gene_ids = tuple(
        _gene_id(value + index) for index in range(candidate_count)
    )
    return AmbiguousFact(
        fact_id=f"ambiguous_{split}_{symbol}_{value}",
        split_units=(f"ambiguous_unit_{split}_{value}",),
        gene_symbol=symbol,
        candidate_gene_ids=candidate_gene_ids,
        candidate_details={
            gene_id: {
                "chromosome": str((value + index) % 22 + 1),
                "biotype": (
                    "protein_coding" if index % 2 == 0 else "lncRNA"
                ),
            }
            for index, gene_id in enumerate(candidate_gene_ids)
        },
    )


def _external(
    value: int,
    *,
    query: str,
    symbol: str,
    split: str = "train",
) -> ExternalFact:
    return ExternalFact(
        fact_id=f"external_{split}_{query}_{value}",
        split_units=(f"unit_{split}_{value}",),
        query=query,
        gene_symbol=symbol,
        gene_id=_gene_id(value),
    )


def _calls(row: dict[str, object]) -> list[dict[str, object]]:
    return [
        message["tool_call"]
        for message in row["messages"]
        if message["role"] == "assistant" and "tool_call" in message
    ]


def _tool_results(row: dict[str, object]) -> list[dict[str, object]]:
    return [
        message["content"]
        for message in row["messages"]
        if message["role"] == "tool"
    ]


def _matches_template(text: str, template: str) -> bool:
    prefix, suffix = template.split("{request}")
    return text.startswith(prefix) and text.endswith(suffix)


class _Tokenizer:
    def __init__(self) -> None:
        self.rendered_messages: list[list[dict[str, object]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, object]], **options: object
    ) -> list[int]:
        tools = options["tools"]
        assert all("description" in tool["function"] for tool in tools)
        self.rendered_messages.append(deepcopy(messages))
        result = [1]
        for index, message in enumerate(messages):
            if message["role"] == "system":
                result.append(10)
            elif message["role"] == "user":
                result.append(20)
            elif message["role"] == "tool":
                result.append(50)
            elif "tool_calls" in message:
                has_final = any(
                    future["role"] == "assistant"
                    and "tool_calls" not in future
                    for future in messages[index + 1 :]
                )
                if message.get("thinking") and not has_final:
                    result.append(30)
                result.append(40)
            else:
                if message.get("thinking"):
                    result.append(60)
                result.append(61)
        return result


class WorldModelS14AllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.curriculum = self.contract["curriculum"]
        self.settings = self.curriculum["families"]["S1.4"]

    def test_split_and_form_allocations_are_exact(self) -> None:
        rows = split_row_counts(
            self.settings["train_rows"],
            self.curriculum["split_fractions"],
        )
        self.assertEqual(
            rows, {"train": 8000, "validation": 1000, "test": 1000}
        )
        expected = {
            "train": {
                1: 750,
                2: 1000,
                3: 1000,
                4: 1000,
                5: 1000,
                6: 1000,
                7: 1000,
                8: 1000,
                9: 250,
            },
            "validation": {
                1: 94,
                2: 125,
                3: 125,
                4: 125,
                5: 125,
                6: 125,
                7: 125,
                8: 125,
                9: 31,
            },
            "test": {
                1: 94,
                2: 125,
                3: 125,
                4: 125,
                5: 125,
                6: 125,
                7: 125,
                8: 125,
                9: 31,
            },
        }
        actual = {
            split: form_counts(total, self.settings, self.curriculum)
            for split, total in rows.items()
        }
        self.assertEqual(actual, expected)
        for split, counts in actual.items():
            tool_rows = sum(counts[form] for form in (1, 2, 5, 6, 8, 9))
            ordinary_tool_rows = tool_rows
            self.assertEqual(
                counts[9],
                largest_remainder_counts(
                    ordinary_tool_rows,
                    {"failure": 0.05, "success": 0.95},
                )["failure"],
                split,
            )

    def test_contract_records_every_build_decision(self) -> None:
        self.assertEqual(
            list(self.settings),
            [
                "dataset_id",
                "question_family",
                "book_mode",
                "output_directory",
                "assignment_seed",
                "train_rows",
                "allocations",
                "ambiguous_outcome_counts",
                "maximum_candidate_gene_ids",
                "mygene_source",
                "failure_injection",
            ],
        )
        self.assertEqual(
            self.settings["allocations"],
            {
                "symbol_id_match": 750,
                "symbol_id_mismatch": 1000,
                "id_notation_match": 1000,
                "id_near_miss": 1000,
                "historical_symbol_unresolved": 1000,
                "ambiguous_symbol": 1000,
                "registry_block_match": 1000,
                "mygene_id_match": 1000,
                "backend_unavailable": 250,
            },
        )
        self.assertEqual(
            self.settings["ambiguous_outcome_counts"],
            {"candidate": 500, "noncandidate": 500},
        )
        self.assertEqual(self.settings["maximum_candidate_gene_ids"], 49)
        self.assertEqual(
            self.settings["failure_injection"],
            {
                "source_form": "symbol_id_match",
                "allocation": "replace",
                "call_stage": "first",
            },
        )
        mygene = self.settings["mygene_source"]
        self.assertEqual(
            list(mygene),
            [
                "snapshot_id",
                "source_path",
                "source_url",
                "source_sha256",
                "query_count",
                "usable_mapping_count",
            ],
        )
        self.assertEqual(mygene["source_path"], str(MYGENE_SOURCE))
        self.assertEqual(mygene["source_sha256"], MYGENE_SHA256)
        self.assertGreater(mygene["query_count"], 0)
        self.assertGreater(mygene["usable_mapping_count"], 0)

    def test_largest_remainder_ties_follow_form_order(self) -> None:
        shares = {
            name: count / self.settings["train_rows"]
            for name, count in self.settings["allocations"].items()
        }
        counts = largest_remainder_counts(1000, shares)
        self.assertEqual(counts["symbol_id_match"], 94)
        self.assertEqual(counts["backend_unavailable"], 31)
        self.assertTrue(
            all(
                counts[name] == 125
                for name in list(self.settings["allocations"])[1:-1]
            )
        )
        self.assertEqual(sum(counts.values()), 1000)

    def test_prompt_pool_is_split_held_out_and_complete(self) -> None:
        phrases_by_split = {
            split: {
                template
                for templates in framings.values()
                for template in templates
            }
            for split, framings in ALIGNMENT_PROMPTS.items()
        }
        self.assertEqual(
            set(phrases_by_split), {"train", "validation", "test"}
        )
        for split, framings in ALIGNMENT_PROMPTS.items():
            self.assertEqual(set(framings), {"direct", "step"})
            self.assertTrue(all(framings.values()))
            self.assertTrue(
                all(
                    template.count("{request}") == 1
                    for templates in framings.values()
                    for template in templates
                )
            )
            held_out = set().union(
                *(
                    phrases
                    for other, phrases in phrases_by_split.items()
                    if other != split
                )
            )
            self.assertTrue(phrases_by_split[split].isdisjoint(held_out))

    def test_mygene_snapshot_matches_the_contract_pin(self) -> None:
        if not MYGENE_SOURCE.is_file():
            self.skipTest(f"MyGene snapshot is absent: {MYGENE_SOURCE}")
        self.assertEqual(file_sha256(MYGENE_SOURCE), MYGENE_SHA256)
        payload = json.loads(MYGENE_SOURCE.read_text())
        self.assertEqual(
            len(payload), self.settings["mygene_source"]["query_count"]
        )
        self.assertTrue(all("::" in query for query in payload))


class WorldModelS14SourceTests(unittest.TestCase):
    def test_real_source_pools_are_pinned_capped_and_split_safe(self) -> None:
        contract = load_contract()
        settings = contract["curriculum"]["families"]["S1.4"]
        registry_path = Path(
            contract["coordinate_system"]["identifier_registry"][
                "source_path"
            ]
        )
        required = (SOURCE_INDEX, MYGENE_SOURCE, registry_path)
        missing = [path for path in required if not path.is_file()]
        if missing:
            self.skipTest(f"canonical source is absent: {missing[0]}")

        payload = check_source_index(SOURCE_INDEX, contract)
        gene_versions = read_ensembl_gtf_identifiers(registry_path)
        mygene_cache = json.loads(MYGENE_SOURCE.read_text())
        pools = build_fact_pools(
            payload,
            gene_versions,
            mygene_cache,
            s0_settings=contract["curriculum"]["families"]["S0"],
            seed=settings["assignment_seed"],
            fractions=contract["curriculum"]["split_fractions"],
            maximum_candidate_gene_ids=settings[
                "maximum_candidate_gene_ids"
            ],
        )
        self.assertEqual(source_pool_coverage(pools), EXPECTED_SOURCE_COVERAGE)
        self.assertEqual(set(pools), {"train", "validation", "test"})
        for split_pools in pools.values():
            self.assertEqual(
                set(split_pools),
                {
                    "resolved",
                    "current",
                    "near_miss",
                    "historical",
                    "ambiguous",
                    "external",
                },
            )
            self.assertTrue(all(split_pools.values()))
            self.assertTrue(
                all(
                    isinstance(fact, GeneFact)
                    and fact.gene_id in payload["genes_by_id"]
                    and fact.gene_version is not None
                    for fact in split_pools["current"]
                )
            )
            self.assertTrue(
                all(
                    left.gene_id != right.gene_id
                    for left, right in split_pools["near_miss"]
                )
            )
            self.assertTrue(
                all(
                    isinstance(fact, ExternalFact)
                    and fact.gene_id in payload["genes_by_id"]
                    for fact in split_pools["historical"]
                )
            )
            self.assertTrue(
                all(
                    isinstance(fact, AmbiguousFact)
                    and 1 < len(fact.candidate_gene_ids) <= 49
                    and set(fact.candidate_gene_ids)
                    == set(fact.candidate_details)
                    for fact in split_pools["ambiguous"]
                )
            )
            self.assertTrue(
                all(
                    isinstance(fact, ExternalFact)
                    and fact.gene_id in payload["genes_by_id"]
                    for fact in split_pools["external"]
                )
            )

        external_count = sum(
            len(split_pools["external"])
            for split_pools in pools.values()
        )
        self.assertEqual(
            external_count,
            settings["mygene_source"]["usable_mapping_count"] - 1,
        )
        self.assertEqual(external_count, 4595)

        current_symbol_splits: dict[str, set[str]] = {}
        for split, split_pools in pools.items():
            for fact in split_pools["current"]:
                if fact.gene_symbol is None:
                    continue
                current_symbol_splits.setdefault(
                    fact.gene_symbol.casefold(), set()
                ).add(split)
        for split, split_pools in pools.items():
            for fact in split_pools["external"]:
                lexical_splits = current_symbol_splits.get(
                    fact.query.casefold(), set()
                )
                self.assertTrue(
                    not lexical_splits or lexical_splits == {split},
                    (fact.query, split, lexical_splits),
                )

        units_by_split: dict[str, set[str]] = {}
        for split, split_pools in pools.items():
            facts: list[GeneFact | AmbiguousFact | ExternalFact] = []
            for name in (
                "resolved",
                "current",
                "historical",
                "ambiguous",
                "external",
            ):
                facts.extend(split_pools[name])
            for left, right in split_pools["near_miss"]:
                facts.extend((left, right))
            units_by_split[split] = {
                unit for fact in facts for unit in fact.split_units
            }
        self.assertTrue(
            units_by_split["train"].isdisjoint(units_by_split["validation"])
        )
        self.assertTrue(
            units_by_split["train"].isdisjoint(units_by_split["test"])
        )
        self.assertTrue(
            units_by_split["validation"].isdisjoint(units_by_split["test"])
        )


class WorldModelS14SelectionTests(unittest.TestCase):
    @staticmethod
    def _pools() -> dict[str, dict[str, list[object]]]:
        pools: dict[str, dict[str, list[object]]] = {}
        for split_index, split in enumerate(("train", "validation", "test")):
            start = 1000 * (split_index + 1)
            current = [
                _fact(
                    start + index * 10,
                    symbol=f"GENE{split_index}{index}",
                    split=split,
                )
                for index in range(4)
            ]
            historical = [
                _historical(
                    start + index * 10,
                    symbol=f"OLD{split_index}{index}",
                    current_symbol=current[index].gene_symbol,
                    split=split,
                    linked_unit=current[index].split_units[0],
                )
                for index in range(3)
            ]
            ambiguous = [
                _ambiguous(
                    start + 100 + index * 10,
                    symbol=f"AMB{split_index}{index}",
                    split=split,
                )
                for index in range(3)
            ]
            ambiguous_candidates = [
                GeneFact(
                    fact_id=f"candidate_{split}_{gene_id}",
                    split_units=fact.split_units,
                    gene_symbol=fact.gene_symbol,
                    gene_id=gene_id,
                    gene_version=1,
                )
                for fact in ambiguous
                for gene_id in fact.candidate_gene_ids
            ]
            all_current = current + ambiguous_candidates
            external = [
                ExternalFact(
                    fact_id=f"external_{split}_{index}",
                    split_units=current[index].split_units,
                    query=f"ALIAS{split_index}{index}",
                    gene_symbol=current[index].gene_symbol,
                    gene_id=current[index].gene_id,
                )
                for index in range(3)
            ]
            pools[split] = {
                "resolved": current,
                "current": all_current,
                "near_miss": [
                    (current[index], current[(index + 1) % len(current)])
                    for index in range(len(current))
                ],
                "historical": historical,
                "ambiguous": ambiguous,
                "external": external,
            }
        return pools

    def test_selection_routes_forms_balances_outcomes_and_pairs_failures(self) -> None:
        counts = {
            split: {
                1: 16,
                2: 16,
                3: 16,
                4: 16,
                5: 16,
                6: 16,
                7: 16,
                8: 16,
                9: 8,
            }
            for split in ("train", "validation", "test")
        }
        pools = self._pools()
        selected = select_scenarios(
            pools, counts, seed=1729, case_rate=0
        )
        repeated = select_scenarios(
            pools, counts, seed=1729, case_rate=0
        )

        units_by_split: dict[str, set[str]] = {}
        for split, scenarios in selected.items():
            with self.subTest(split=split):
                self.assertEqual(
                    Counter(item.form for item in scenarios),
                    Counter(counts[split]),
                )
                self.assertEqual(
                    Counter(item.framing for item in scenarios),
                    {"direct": 68, "step": 68},
                )
                self.assertEqual(
                    {item.source_trace for item in scenarios},
                    {f"S1.4.{form}" for form in range(1, 10)},
                )
                by_form = {
                    form: [item for item in scenarios if item.form == form]
                    for form in range(1, 10)
                }
                self.assertEqual(
                    Counter(
                        item.ambiguous_target_is_candidate
                        for item in by_form[6]
                    ),
                    {True: 8, False: 8},
                )
                self.assertTrue(
                    all(isinstance(item.fact, GeneFact) for item in by_form[1])
                )
                self.assertTrue(
                    all(
                        isinstance(item.fact, ExternalFact)
                        for item in by_form[5]
                    )
                )
                self.assertTrue(
                    all(
                        isinstance(item.fact, AmbiguousFact)
                        for item in by_form[6]
                    )
                )
                self.assertTrue(
                    all(
                        isinstance(item.fact, ExternalFact)
                        for item in by_form[8]
                    )
                )
                for form in range(1, 10):
                    for framing in ("direct", "step"):
                        framed = [
                            item
                            for item in by_form[form]
                            if item.framing == framing
                        ]
                        self.assertTrue(framed)
                        self.assertTrue(
                            all(
                                0
                                <= item.template_index
                                < len(ALIGNMENT_PROMPTS[split][framing])
                                for item in framed
                            )
                        )
                    self.assertEqual(
                        {item.orientation for item in by_form[form]}, {0, 1}
                    )

                successes = {
                    item.scenario_id: item for item in by_form[1]
                }
                for failure in by_form[9]:
                    success = successes[failure.paired_success_id]
                    self.assertEqual(failure.source_form, 1)
                    self.assertEqual(failure.fact, success.fact)
                    self.assertEqual(failure.target, success.target)
                    self.assertEqual(failure.framing, success.framing)
                    self.assertEqual(
                        failure.template_index, success.template_index
                    )
                    self.assertEqual(failure.orientation, success.orientation)
                    self.assertEqual(failure.distractor, success.distractor)

                units_by_split[split] = {
                    unit for item in scenarios for unit in item.split_units
                }

        self.assertEqual(
            {
                split: [item.scenario_id for item in scenarios]
                for split, scenarios in selected.items()
            },
            {
                split: [item.scenario_id for item in scenarios]
                for split, scenarios in repeated.items()
            },
        )
        self.assertTrue(
            units_by_split["train"].isdisjoint(units_by_split["validation"])
        )
        self.assertTrue(
            units_by_split["train"].isdisjoint(units_by_split["test"])
        )
        self.assertTrue(
            units_by_split["validation"].isdisjoint(units_by_split["test"])
        )

    def test_real_allocations_cover_templates_and_form6_suboutcomes(self) -> None:
        contract = load_contract()
        curriculum = contract["curriculum"]
        settings = curriculum["families"]["S1.4"]
        row_counts = split_row_counts(
            settings["train_rows"], curriculum["split_fractions"]
        )
        counts_by_split = {
            split: form_counts(total, settings, curriculum)
            for split, total in row_counts.items()
        }
        scenarios_by_split = select_scenarios(
            self._pools(),
            counts_by_split,
            seed=settings["assignment_seed"],
            case_rate=0,
            framing_shares=curriculum["framing"],
            settings=settings,
        )
        expected_outcomes = {
            "train": {True: 500, False: 500},
            "validation": {True: 63, False: 62},
            "test": {True: 63, False: 62},
        }
        for split, scenarios in scenarios_by_split.items():
            by_form = {
                form: [item for item in scenarios if item.form == form]
                for form in range(1, 10)
            }
            self.assertEqual(
                Counter(
                    item.ambiguous_target_is_candidate
                    for item in by_form[6]
                ),
                expected_outcomes[split],
            )
            for form in range(1, 10):
                for framing in ("direct", "step"):
                    with self.subTest(
                        split=split, form=form, framing=framing
                    ):
                        self.assertEqual(
                            {
                                item.template_index
                                for item in by_form[form]
                                if item.framing == framing
                            },
                            set(
                                range(
                                    len(ALIGNMENT_PROMPTS[split][framing])
                                )
                            ),
                        )
                self.assertEqual(
                    {item.orientation for item in by_form[form]}, {0, 1}
                )


class WorldModelS14RowSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.tools = tool_definitions(self.contract)
        self.primary = _fact(100, symbol="GENEA")
        self.second = _fact(200, symbol="GENEB")
        self.near_miss = _fact(101, symbol="GENENEAR")
        self.distractor = _fact(300, symbol="GENEC")
        self.historical = _historical(
            100,
            symbol="OLDA",
            current_symbol=self.primary.gene_symbol,
            linked_unit=self.primary.split_units[0],
        )
        self.ambiguous = _ambiguous(400, symbol="AMBIG")
        self.candidate_target = _fact(
            400,
            symbol="AMBIG",
            units=self.ambiguous.split_units,
        )
        self.external = ExternalFact(
            fact_id="external_alias_a",
            split_units=self.primary.split_units,
            query="ALIASA",
            gene_symbol=self.primary.gene_symbol,
            gene_id=self.primary.gene_id,
        )
        genes_by_id = {
            self.primary.gene_id: ["GENEA", "1", "protein_coding"],
            self.second.gene_id: ["GENEB", "2", "protein_coding"],
            self.near_miss.gene_id: [
                "GENENEAR",
                "1",
                "protein_coding",
            ],
            self.distractor.gene_id: ["GENEC", "3", "lncRNA"],
            **{
                gene_id: [
                    self.ambiguous.gene_symbol,
                    detail["chromosome"],
                    detail["biotype"],
                ]
                for gene_id, detail in self.ambiguous.candidate_details.items()
            },
        }
        payload = {
            "coordinate_system": {
                "ensembl_release": "116",
                "identifier_registry_id": "ensembl_homo_sapiens_116_v3",
                "multiplex_id": "full_brain_multiplex_v1",
                "multiplex_gene_rule": "test_fixture",
            },
            "genes_by_id": genes_by_id,
            "multiplex_gene_ids": [],
            "sources": {
                "ensembl_gtf": {
                    "file_name": "registry.gtf.gz",
                    "sha256": "0" * 64,
                }
            },
        }
        self.runtime = WorldModelToolRuntime(
            payload=payload,
            contract=self.contract,
            evidence_id_factory=EvidenceSequence(),
        )

    def _scenario(
        self,
        form: int,
        *,
        fact: GeneFact | AmbiguousFact | ExternalFact | None = None,
        target: GeneFact | None = None,
        framing: str = "direct",
        orientation: int = 0,
        ambiguous_target_is_candidate: bool | None = None,
        paired_success_id: str | None = None,
        source_form: int | None = None,
    ) -> Scenario:
        default_fact: GeneFact | AmbiguousFact | ExternalFact = self.primary
        default_target = self.primary
        if form == 2:
            default_target = self.second
        elif form == 4:
            default_target = self.near_miss
        elif form == 5:
            default_fact = self.historical
        elif form == 6:
            default_fact = self.ambiguous
            default_target = (
                self.candidate_target
                if ambiguous_target_is_candidate is not False
                else self.second
            )
        elif form == 8:
            default_fact = self.external
        return Scenario(
            split="train",
            form=form,
            fact=fact or default_fact,
            target=target or default_target,
            framing=framing,
            template_index=0,
            occurrence=0,
            orientation=orientation,
            distractor=self.distractor,
            ambiguous_target_is_candidate=ambiguous_target_is_candidate,
            source_form=source_form,
            paired_success_id=paired_success_id,
        )

    def _build(self, scenario: Scenario) -> dict[str, object]:
        row, evidence = build_row(
            scenario,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        self.assertTrue(all(len(record) == 3 for record in evidence))
        expected_evidence = (
            1 if scenario.form in {1, 2, 5, 6, 7, 8} else 0
        )
        self.assertEqual(len(evidence), expected_evidence)
        return row

    def test_all_nine_forms_have_the_reviewed_action_semantics(self) -> None:
        success = self._scenario(1)
        scenarios: list[tuple[str, Scenario, list[str], str, bool | None]] = [
            ("1", success, [SYMBOL_TOOL], "resolved", True),
            ("2", self._scenario(2), [SYMBOL_TOOL], "resolved", False),
            ("3", self._scenario(3), [], "resolved", True),
            ("4", self._scenario(4), [], "resolved", False),
            ("5", self._scenario(5), [SYMBOL_TOOL], "not_found", None),
            (
                "6_candidate",
                self._scenario(6, ambiguous_target_is_candidate=True),
                [SYMBOL_TOOL],
                "ambiguous",
                None,
            ),
            (
                "6_noncandidate",
                self._scenario(6, ambiguous_target_is_candidate=False),
                [SYMBOL_TOOL],
                "resolved",
                False,
            ),
            ("7", self._scenario(7), [], "resolved", True),
            ("8", self._scenario(8), [GENE_ID_TOOL], "resolved", True),
            (
                "9",
                self._scenario(
                    9,
                    source_form=1,
                    paired_success_id=success.scenario_id,
                ),
                [SYMBOL_TOOL],
                "error",
                None,
            ),
        ]
        for label, scenario, expected_calls, status, same_gene in scenarios:
            with self.subTest(form=label):
                row = self._build(scenario)
                self.assertEqual(
                    list(row), ["row_id", "messages", "tools", "metadata"]
                )
                self.assertEqual(
                    [call["name"] for call in _calls(row)], expected_calls
                )
                answer = row["messages"][-1]["content"]
                self.assertEqual(
                    list(answer),
                    [
                        "status",
                        "provenance",
                        "result",
                        "continuation_decision",
                        "blocked_reason",
                        "summary",
                    ],
                )
                self.assertEqual(answer["status"], status)
                self.assertEqual(
                    answer["continuation_decision"],
                    "defer" if status in {"not_found", "ambiguous", "error"}
                    else "stop",
                )
                self.assertEqual(
                    answer["blocked_reason"] is None,
                    status == "resolved",
                )
                if scenario.form == 9:
                    self.assertEqual(
                        answer["result"]["failed_tool"], SYMBOL_TOOL
                    )
                    self.assertEqual(
                        answer["result"]["error_code"],
                        "BACKEND_UNAVAILABLE",
                    )
                    self.assertIsNone(answer["provenance"])
                else:
                    result = answer["result"]
                    self.assertEqual(result["same_gene"], same_gene)
                    self.assertEqual(
                        {result["first"]["item"], result["second"]["item"]}
                        <= ITEMS,
                        True,
                    )
                    if same_gene is None:
                        self.assertEqual(answer["continuation_decision"], "defer")
                metadata = row["metadata"]
                self.assertEqual(metadata["book_mode"], "tool_call")
                self.assertEqual(metadata["step"], "S1")
                self.assertEqual(
                    metadata["question_family"], "cross_context_alignment"
                )
                self.assertEqual(metadata["form"], scenario.form)
                self.assertEqual(metadata["source_trace"], scenario.source_trace)
                expected_code = {
                    "5": "not_found",
                    "6_candidate": "ambiguous_symbol",
                    "9": "tool_error",
                }.get(label)
                self.assertEqual(metadata["blocked_reason_code"], expected_code)

    def test_ambiguous_candidate_fields_are_exact_and_side_local(self) -> None:
        for orientation in (0, 1):
            for candidate, expected_same in ((True, None), (False, False)):
                with self.subTest(orientation=orientation, candidate=candidate):
                    scenario = self._scenario(
                        6,
                        orientation=orientation,
                        ambiguous_target_is_candidate=candidate,
                    )
                    row = self._build(scenario)
                    result = row["messages"][-1]["content"]["result"]
                    ambiguous_side = (
                        result["first"] if orientation == 0 else result["second"]
                    )
                    other_side = (
                        result["second"] if orientation == 0 else result["first"]
                    )
                    self.assertIsNone(ambiguous_side["gene_id"])
                    self.assertEqual(
                        ambiguous_side["candidate_gene_ids"],
                        list(self.ambiguous.candidate_gene_ids),
                    )
                    self.assertEqual(
                        ambiguous_side["candidate_details"],
                        self.ambiguous.candidate_details,
                    )
                    self.assertNotIn("candidate_gene_ids", other_side)
                    self.assertEqual(result["same_gene"], expected_same)
                    self.assertEqual(
                        other_side["gene_id"] in self.ambiguous.candidate_gene_ids,
                        candidate,
                    )

    def test_orientation_preserves_request_and_result_order(self) -> None:
        rows = {
            orientation: self._build(
                self._scenario(2, orientation=orientation)
            )
            for orientation in (0, 1)
        }
        for orientation, row in rows.items():
            result = row["messages"][-1]["content"]["result"]
            expected = (
                (self.primary.gene_symbol, self.second.gene_id)
                if orientation == 0
                else (self.second.gene_id, self.primary.gene_symbol)
            )
            self.assertEqual(
                (result["first"]["input"], result["second"]["input"]),
                expected,
            )
        self.assertNotEqual(
            rows[0]["messages"][-1]["content"]["result"]["first"]["item"],
            rows[1]["messages"][-1]["content"]["result"]["first"]["item"],
        )

    def test_scenario_split_unit_covers_both_mentions_and_distractors(self) -> None:
        scenario = self._scenario(2)
        self.assertEqual(
            set(scenario.split_units),
            {
                *self.primary.split_units,
                *self.second.split_units,
                *self.distractor.split_units,
            },
        )
        self.assertTrue(scenario.metadata_split_unit.startswith("group_"))
        self.assertEqual(
            scenario.metadata_split_unit,
            replace(scenario, orientation=1).metadata_split_unit,
        )

    def test_rendered_prompts_use_only_their_split_pool(self) -> None:
        for split in ("train", "validation", "test"):
            scenario = replace(self._scenario(1), split=split)
            row = self._build(scenario)
            user = next(
                message["content"]
                for message in row["messages"]
                if message["role"] == "user"
            )
            matches = [
                (framing, index)
                for framing, templates in ALIGNMENT_PROMPTS[split].items()
                for index, template in enumerate(templates)
                if _matches_template(user, template)
            ]
            self.assertEqual(matches, [(scenario.framing, 0)])
            self.assertFalse(
                any(
                    _matches_template(user, template)
                    for other, framings in ALIGNMENT_PROMPTS.items()
                    if other != split
                    for templates in framings.values()
                    for template in templates
                )
            )

    def test_id_notation_uses_native_version_and_needs_no_call(self) -> None:
        row = self._build(self._scenario(3))
        self.assertEqual(_calls(row), [])
        answer = row["messages"][-1]["content"]
        self.assertIsNone(answer["provenance"])
        first = answer["result"]["first"]
        second = answer["result"]["second"]
        self.assertTrue(
            first["input"].endswith(f".{self.primary.gene_version}")
            or second["input"].endswith(f".{self.primary.gene_version}")
        )
        self.assertEqual(first["gene_id"], self.primary.gene_id)
        self.assertEqual(second["gene_id"], self.primary.gene_id)

    def test_historical_alias_is_not_mapped_from_memory(self) -> None:
        row = self._build(self._scenario(5, framing="step"))
        self.assertEqual(_calls(row)[0]["arguments"], {"gene_symbol": "OLDA"})
        result = row["messages"][-1]["content"]["result"]
        unresolved = next(side for side in ("first", "second") if result[side]["input"] == "OLDA")
        self.assertIsNone(result[unresolved]["gene_id"])
        self.assertIsNone(result["same_gene"])
        self.assertEqual(row["messages"][-1]["content"]["continuation_decision"], "defer")

    def test_registry_block_and_mygene_use_their_required_evidence_paths(self) -> None:
        registry_row = self._build(self._scenario(7, framing="step"))
        self.assertEqual(_calls(registry_row), [])
        harness = next(
            item["content"]
            for item in registry_row["messages"]
            if item["role"] == "harness"
        )
        self.assertIn("Registry entries, Ensembl release 116:", harness)
        registry_answer = registry_row["messages"][-1]["content"]
        self.assertIsNotNone(registry_answer["provenance"]["registry"])
        self.assertEqual(registry_answer["continuation_decision"], "continue")

        external_row = self._build(self._scenario(8, framing="step"))
        external_harness = next(
            item["content"]
            for item in external_row["messages"]
            if item["role"] == "harness"
        )
        self.assertIn("query_mygene result:", external_harness)
        self.assertEqual(
            _calls(external_row)[0],
            {
                "name": GENE_ID_TOOL,
                "arguments": {"gene_id": self.primary.gene_id},
            },
        )
        result = external_row["messages"][-1]["content"]["result"]
        external_side = next(
            side
            for side in (result["first"], result["second"])
            if side["item"] == "query_mygene"
        )
        self.assertEqual(external_side["input"], "ALIASA")
        self.assertEqual(external_side["gene_id"], self.primary.gene_id)

    def test_failure_keeps_the_success_prefix_and_changes_only_the_result(self) -> None:
        success = self._scenario(1, framing="step", orientation=1)
        failure = replace(
            success,
            form=9,
            source_form=1,
            paired_success_id=success.scenario_id,
        )
        success_row = self._build(success)
        failure_row = self._build(failure)
        success_tool_index = next(
            index
            for index, message in enumerate(success_row["messages"])
            if message["role"] == "tool"
        )
        failure_tool_index = next(
            index
            for index, message in enumerate(failure_row["messages"])
            if message["role"] == "tool"
        )
        self.assertEqual(success_tool_index, failure_tool_index)
        self.assertEqual(
            success_row["messages"][:success_tool_index],
            failure_row["messages"][:failure_tool_index],
        )
        self.assertEqual(
            _tool_results(success_row)[0]["status"], "resolved"
        )
        self.assertEqual(_tool_results(failure_row)[0]["status"], "error")
        self.assertIsNone(_tool_results(failure_row)[0]["provenance"])
        failure_text = json.dumps(failure_row["messages"][-1]["content"])
        self.assertNotIn('"same_gene": true', failure_text)
        self.assertNotIn('"same_gene": false', failure_text)

    def test_resolved_continuation_follows_the_question_frame(self) -> None:
        for form in (1, 2, 3, 4, 7, 8):
            for framing, expected in (("direct", "stop"), ("step", "continue")):
                with self.subTest(form=form, framing=framing):
                    row = self._build(self._scenario(form, framing=framing))
                    answer = row["messages"][-1]["content"]
                    self.assertEqual(answer["status"], "resolved")
                    self.assertEqual(answer["continuation_decision"], expected)
                    self.assertIsNone(answer["blocked_reason"])

        for candidate in (True, False):
            scenario = self._scenario(
                6,
                framing="step",
                ambiguous_target_is_candidate=candidate,
            )
            answer = self._build(scenario)["messages"][-1]["content"]
            self.assertEqual(
                answer["continuation_decision"],
                "defer" if candidate else "continue",
            )

    def test_message_orders_match_call_and_no_call_forms(self) -> None:
        for form in range(1, 10):
            success = self._scenario(1)
            options: dict[str, object] = {}
            if form == 6:
                options["ambiguous_target_is_candidate"] = True
            if form == 9:
                options.update(
                    source_form=1,
                    paired_success_id=success.scenario_id,
                )
            scenario = success if form == 1 else self._scenario(form, **options)
            row = self._build(scenario)
            roles = [message["role"] for message in row["messages"]]
            call_count = len(_calls(row))
            self.assertEqual(call_count, 0 if form in {3, 4, 7} else 1)
            self.assertEqual(roles[0], "system")
            self.assertEqual(roles[-1], "assistant")
            self.assertEqual(roles.count("user"), 1)
            self.assertEqual(roles.count("tool"), call_count)
            assistants = [
                message
                for message in row["messages"]
                if message["role"] == "assistant"
            ]
            self.assertTrue(all(message.get("thinking") for message in assistants))
            self.assertEqual(
                len(row["tools"]), len(self.contract["tools"]["callable"])
            )

    def test_training_adapter_accepts_s1_4_and_masks_nonassistant_turns(self) -> None:
        scenario = self._scenario(8, framing="step")
        row = self._build(scenario)
        flattened = flatten_v3_trace_for_arrow(row)
        self.assertEqual(flattened["record_id"], row["row_id"])
        self.assertEqual(flattened["fact_id"], row["metadata"]["split_unit"])
        self.assertEqual(
            flattened["question_family"], "cross_context_alignment"
        )
        self.assertEqual(json.loads(flattened["trace_json"]), row)

        tokenizer = _Tokenizer()
        tokenized = tokenize_v3_trace_for_sft(row, tokenizer, 64)
        self.assertEqual(
            tokenized["input_ids"], [1, 10, 20, 30, 40, 50, 60, 61]
        )
        self.assertEqual(
            tokenized["completion_mask"], [0, 0, 0, 1, 1, 0, 1, 1]
        )
        self.assertTrue(
            all(
                message["role"] != "harness"
                for rendered in tokenizer.rendered_messages
                for message in rendered
            )
        )
        first_user = tokenizer.rendered_messages[0][1]["content"]
        self.assertIn("Context (from the harness):", first_user)
        self.assertIn("query_mygene result:", first_user)
        self.assertIn("User request:", first_user)

    def test_training_adapter_supports_no_call_registry_context(self) -> None:
        row = self._build(self._scenario(7))
        flattened = flatten_v3_trace_for_arrow(row)
        self.assertEqual(
            flattened["question_family"], "cross_context_alignment"
        )
        tokenizer = _Tokenizer()
        tokenized = tokenize_v3_trace_for_sft(row, tokenizer, 64)
        self.assertEqual(tokenized["input_ids"], [1, 10, 20, 60, 61])
        self.assertEqual(tokenized["completion_mask"], [0, 0, 0, 1, 1])
        self.assertIn(
            "Registry entries, Ensembl release 116:",
            tokenizer.rendered_messages[0][1]["content"],
        )

    def test_thinking_filter_rejects_outcome_leaks_and_false_claims(self) -> None:
        scenario = self._scenario(1, framing="step")
        row = self._build(scenario)
        self.assertGreater(filter_row_thinking(row, scenario), 0)

        mutations = (
            "GENEA resolves to the target ID, so they are the same gene.",
            "I will follow the answer schema for this comparison.",
        )
        for thinking in mutations:
            with self.subTest(thinking=thinking):
                changed = deepcopy(row)
                first_assistant = next(
                    message
                    for message in changed["messages"]
                    if message["role"] == "assistant"
                )
                first_assistant["thinking"] = thinking
                with self.assertRaises(CorpusBuildError):
                    filter_row_thinking(changed, scenario)

        mismatch = self._scenario(2)
        mismatch_row = self._build(mismatch)
        mismatch_row["messages"][-1]["thinking"] = (
            "The registry IDs are equal, so these are the same gene."
        )
        with self.assertRaises(CorpusBuildError):
            filter_row_thinking(mismatch_row, mismatch)

        success = self._scenario(1)
        failure = replace(
            success,
            form=9,
            source_form=1,
            paired_success_id=success.scenario_id,
        )
        failure_row = self._build(failure)
        failure_row["messages"][-1]["thinking"] = (
            "The failed lookup proves that the two mentions are different."
        )
        with self.assertRaises(CorpusBuildError):
            filter_row_thinking(failure_row, failure)

    def test_visible_prompt_mask_keeps_structure_and_masks_source_values(self) -> None:
        scenario = self._scenario(7, framing="step")
        row = self._build(scenario)
        masked = masked_visible_prompt(row, scenario)
        self.assertEqual(
            [line.split(":", 1)[0] for line in masked.splitlines()],
            ["harness", "user"],
        )
        self.assertIn("Registry entries, Ensembl release", masked)
        self.assertIn("Module", masked)
        self.assertNotIn("ev_reg_", masked)
        for value in (
            self.primary.gene_symbol,
            self.primary.gene_id,
            self.distractor.gene_symbol,
            self.distractor.gene_id,
        ):
            self.assertNotIn(value.casefold(), masked.casefold())

        external = self._scenario(8)
        external_masked = masked_visible_prompt(self._build(external), external)
        self.assertIn("query_mygene result", external_masked)
        for value in (
            self.external.query,
            self.external.gene_symbol,
            self.external.gene_id,
        ):
            self.assertNotIn(value.casefold(), external_masked.casefold())

    def test_prompt_classifier_gates_only_the_outer_wrapper(self) -> None:
        rows_by_split: dict[str, list[dict[str, object]]] = {}
        scenarios_by_split: dict[str, list[Scenario]] = {}
        for split in ("train", "validation", "test"):
            rows = []
            scenarios = []
            for form, cue in ((1, "match-route"), (2, "mismatch-route")):
                for occurrence in range(4):
                    scenario = replace(
                        self._scenario(form),
                        split=split,
                        occurrence=occurrence,
                    )
                    wrapper = ALIGNMENT_PROMPTS[split]["direct"][0].format(
                        request=(
                            f"Compare {scenario.fact.gene_symbol} with "
                            f"{scenario.target.gene_id}."
                        )
                    )
                    rows.append(
                        {
                            "row_id": f"classifier_{split}_{form}_{occurrence}",
                            "messages": [
                                {"role": "system", "content": "Shared system."},
                                {
                                    "role": "harness",
                                    "content": (
                                        f"Structure cue {cue}. Module values: "
                                        f"{scenario.target.gene_id}."
                                    ),
                                },
                                {"role": "user", "content": wrapper},
                            ],
                        }
                    )
                    scenarios.append(scenario)
            rows_by_split[split] = rows
            scenarios_by_split[split] = scenarios

        wrapper_result = masked_wording_classifier_audit(
            rows_by_split, scenarios_by_split
        )
        self.assertEqual(wrapper_result["scope"], "outer_wrapper_only")
        self.assertTrue(wrapper_result["pass"])
        for split in ("validation", "test"):
            self.assertEqual(
                wrapper_result["splits"][split]["accuracy"], 0.5
            )

        diagnostic = full_visible_prompt_classifier_diagnostic(
            rows_by_split, scenarios_by_split
        )
        self.assertEqual(diagnostic["scope"], "full_visible_prompt")
        self.assertTrue(diagnostic["diagnostic_only"])
        self.assertIsNone(diagnostic["pass"])
        for split in ("train", "validation", "test"):
            metrics = diagnostic["splits"][split]
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertFalse(metrics["gate_applied"])
            self.assertIsNone(metrics["threshold"])
            self.assertIsNone(metrics["pass"])

    def test_duplicate_audit_rejects_an_ordinary_copy(self) -> None:
        scenario = self._scenario(1)
        row = self._build(scenario)
        duplicate_scenario = replace(scenario, occurrence=1)
        duplicate = deepcopy(row)
        duplicate["row_id"] = "s1_4_duplicate"
        duplicate["messages"][-1]["content"]["summary"] = (
            "Different final wording with the same outcome."
        )
        with self.assertRaises(CorpusBuildError):
            duplicate_prompt_outcome_audit(
                {"train": [row, duplicate], "validation": [], "test": []},
                {
                    "train": [scenario, duplicate_scenario],
                    "validation": [],
                    "test": [],
                },
            )

    def test_duplicate_audit_allows_only_the_declared_failure_pair(self) -> None:
        success = self._scenario(1, framing="step")
        failure = replace(
            success,
            form=9,
            source_form=1,
            paired_success_id=success.scenario_id,
        )
        result = duplicate_prompt_outcome_audit(
            {
                "train": [self._build(success), self._build(failure)],
                "validation": [],
                "test": [],
            },
            {
                "train": [success, failure],
                "validation": [],
                "test": [],
            },
        )
        self.assertTrue(result["pass"])
        self.assertEqual(
            result["splits"]["train"][
                "intended_success_failure_pair_count"
            ],
            1,
        )


class WorldModelS14GeneratedCorpusTests(unittest.TestCase):
    def test_generated_manifest_and_rows_reconcile_when_present(self) -> None:
        root = Path("data/world_model_v3/corpora/s1/s1_4")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"generated corpus is absent: {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        contract = load_contract()
        curriculum = contract["curriculum"]
        settings = curriculum["families"]["S1.4"]
        expected_rows = {
            "train": 8000,
            "validation": 1000,
            "test": 1000,
        }
        expected_counts = {
            split: form_counts(total, settings, curriculum)
            for split, total in expected_rows.items()
        }
        expected_outcomes = {
            split: largest_remainder_counts(
                counts[6], {"candidate": 0.5, "noncandidate": 0.5}
            )
            for split, counts in expected_counts.items()
        }
        expected_same_gene = {
            split: {
                "true": sum(counts[form] for form in (1, 3, 7, 8)),
                "false": (
                    counts[2]
                    + counts[4]
                    + expected_outcomes[split]["noncandidate"]
                ),
                "null": (
                    counts[5] + expected_outcomes[split]["candidate"]
                ),
                "unavailable": counts[9],
            }
            for split, counts in expected_counts.items()
        }
        expected_status = {
            split: {
                "ambiguous": expected_outcomes[split]["candidate"],
                "error": counts[9],
                "not_found": counts[5],
                "resolved": (
                    expected_same_gene[split]["true"]
                    + expected_same_gene[split]["false"]
                ),
            }
            for split, counts in expected_counts.items()
        }
        expected_call_counts = {
            split: {
                "0": sum(counts[form] for form in (3, 4, 7)),
                "1": sum(counts[form] for form in (1, 2, 5, 6, 8, 9)),
            }
            for split, counts in expected_counts.items()
        }
        expected_tool_names = {
            split: {
                GENE_ID_TOOL: counts[8],
                SYMBOL_TOOL: sum(
                    counts[form] for form in (1, 2, 5, 6, 9)
                ),
                "none": sum(counts[form] for form in (3, 4, 7)),
            }
            for split, counts in expected_counts.items()
        }

        self.assertEqual(manifest["section"], "S1.4")
        self.assertEqual(manifest["dataset_id"], settings["dataset_id"])
        self.assertEqual(manifest["row_counts"], expected_rows)
        self.assertEqual(
            manifest["form_counts"],
            {
                split: {
                    f"S1.4.{form}": count
                    for form, count in counts.items()
                }
                for split, counts in expected_counts.items()
            },
        )
        self.assertEqual(manifest["status_counts"], expected_status)
        self.assertEqual(manifest["same_gene_counts"], expected_same_gene)
        self.assertEqual(manifest["tool_call_counts"], expected_call_counts)
        self.assertEqual(manifest["tool_name_counts"], expected_tool_names)
        self.assertEqual(
            manifest["ambiguous_outcome_counts"], expected_outcomes
        )
        self.assertEqual(
            manifest["framing_counts"],
            {
                split: {
                    "direct": total // 2,
                    "step": total // 2,
                }
                for split, total in expected_rows.items()
            },
        )
        expected_case_eligible = {
            split: sum(
                counts[form] for form in (1, 2, 5, 6, 7, 8, 9)
            )
            for split, counts in expected_counts.items()
        }
        for split, eligible in expected_case_eligible.items():
            case_counts = manifest["case_counts"][split]
            self.assertEqual(set(case_counts), {"eligible", "perturbed"})
            self.assertEqual(case_counts["eligible"], eligible)
            self.assertEqual(
                manifest["notation_counts"][split],
                {"mandatory_form3": expected_counts[split][3]},
            )
            self.assertEqual(
                manifest["notation_counts"][split]["mandatory_form3"],
                expected_counts[split][3],
            )
            self.assertGreater(case_counts["perturbed"], 0)
            self.assertLessEqual(case_counts["perturbed"], eligible)
            self.assertLessEqual(
                abs(
                    case_counts["perturbed"] / eligible
                    - curriculum["identity_case_perturbation"]
                ),
                0.02,
            )

        expected_prompt_hash = hashlib.sha256(
            json.dumps(
                ALIGNMENT_PROMPTS,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            manifest["prompt_templates_sha256"], expected_prompt_hash
        )
        for split in ("train", "validation", "test"):
            expected_template_keys = {
                f"S1.4.{form}:{framing}:{index}"
                for form in range(1, 10)
                for framing in ("direct", "step")
                for index in range(
                    len(ALIGNMENT_PROMPTS[split][framing])
                )
            }
            template_counts = manifest["prompt_template_counts"][split]
            self.assertEqual(set(template_counts), expected_template_keys)
            self.assertTrue(all(count > 0 for count in template_counts.values()))
            self.assertEqual(
                sum(template_counts.values()), expected_rows[split]
            )

            orientation_counts = manifest["orientation_counts"][split]
            self.assertEqual(
                set(orientation_counts),
                {f"S1.4.{form}" for form in range(1, 10)},
            )
            for form in range(1, 10):
                counts = orientation_counts[f"S1.4.{form}"]
                self.assertEqual(set(counts), {"0", "1"})
                self.assertTrue(all(count > 0 for count in counts.values()))
                self.assertEqual(
                    sum(counts.values()), expected_counts[split][form]
                )

            item_pair_counts = manifest["item_pair_counts"][split]
            self.assertEqual(
                set(item_pair_counts),
                {f"S1.4.{form}" for form in range(1, 10)},
            )
            for form in range(1, 10):
                counts = item_pair_counts[f"S1.4.{form}"]
                self.assertTrue(counts)
                self.assertTrue(all(count > 0 for count in counts.values()))
                self.assertEqual(
                    sum(counts.values()), expected_counts[split][form]
                )

        self.assertEqual(
            manifest["failure_injection"]["pair_counts"],
            {
                split: counts[9]
                for split, counts in expected_counts.items()
            },
        )
        self.assertEqual(
            {
                key: manifest["failure_injection"][key]
                for key in ("source_form", "allocation", "call_stage")
            },
            settings["failure_injection"],
        )

        sources = manifest["sources"]
        self.assertEqual(
            sources["mygene_source"]["source_sha256"], MYGENE_SHA256
        )
        self.assertEqual(
            sources["mygene_source"]["source_path"], str(MYGENE_SOURCE)
        )
        self.assertEqual(
            sources["mygene_source"]["query_count"],
            settings["mygene_source"]["query_count"],
        )
        self.assertEqual(
            sources["mygene_source"]["usable_mapping_count"],
            settings["mygene_source"]["usable_mapping_count"],
        )
        self.assertEqual(
            set(sources["pool_counts"]),
            {"train", "validation", "test"},
        )
        self.assertEqual(
            set(sources["ambiguous_candidate_count_distribution"]),
            {"train", "validation", "test"},
        )
        self.assertEqual(
            sources["pool_counts"], EXPECTED_SOURCE_COVERAGE["pool_counts"]
        )
        self.assertEqual(
            sources["ambiguous_candidate_count_distribution"],
            EXPECTED_SOURCE_COVERAGE[
                "ambiguous_candidate_count_distribution"
            ],
        )
        for split in ("train", "validation", "test"):
            self.assertEqual(
                set(sources["pool_counts"][split]),
                {
                    "resolved",
                    "current",
                    "near_miss",
                    "historical",
                    "ambiguous",
                    "external",
                },
            )
            self.assertTrue(
                all(count > 0 for count in sources["pool_counts"][split].values())
            )
            distribution = sources[
                "ambiguous_candidate_count_distribution"
            ][split]
            self.assertTrue(distribution)
            self.assertTrue(
                all(2 <= int(count) <= 49 for count in distribution)
            )

        false_defer = manifest["false_defer_audit"]
        expected_eligible = {
            split: values["true"] + values["false"]
            for split, values in expected_same_gene.items()
        }
        self.assertEqual(
            false_defer["eligible_row_count"], sum(expected_eligible.values())
        )
        self.assertEqual(false_defer["false_defer_count"], 0)
        self.assertEqual(false_defer["false_defer_rate"], 0.0)
        self.assertTrue(false_defer["pass"])
        for split, eligible in expected_eligible.items():
            self.assertEqual(
                false_defer["splits"][split],
                {
                    "eligible_row_count": eligible,
                    "false_defer_count": 0,
                    "false_defer_rate": 0.0,
                    "pass": True,
                },
            )

        wrapper = manifest["masked_wording_classifier"]
        self.assertEqual(wrapper["scope"], "outer_wrapper_only")
        self.assertTrue(wrapper["pass"])
        for split in ("validation", "test"):
            self.assertTrue(wrapper["splits"][split]["gate_applied"])
            self.assertTrue(wrapper["splits"][split]["pass"])

        full = manifest["full_visible_prompt_classifier"]
        self.assertEqual(full["scope"], "full_visible_prompt")
        self.assertTrue(full["diagnostic_only"])
        self.assertIsNone(full["pass"])
        for split in ("train", "validation", "test"):
            metrics = full["splits"][split]
            self.assertEqual(metrics["row_count"], expected_rows[split])
            self.assertFalse(metrics["gate_applied"])
            self.assertIsNone(metrics["threshold"])
            self.assertIsNone(metrics["pass"])

        self.assertTrue(manifest["prompt_structure_audit"]["pass"])
        self.assertTrue(manifest["thinking_filter"]["pass"])
        self.assertTrue(manifest["duplicate_prompt_tool_outcome"]["pass"])

        all_row_ids: set[str] = set()
        units_by_split: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            path = root / f"{split}.jsonl"
            with path.open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), expected_rows[split])
            self.assertEqual(
                file_sha256(path), manifest["files"][path.name]["sha256"]
            )
            self.assertEqual(
                Counter(row["metadata"]["form"] for row in rows),
                Counter(expected_counts[split]),
            )
            self.assertEqual(
                {row["metadata"]["source_trace"] for row in rows},
                {f"S1.4.{form}" for form in range(1, 10)},
            )
            units_by_split[split] = {
                row["metadata"]["split_unit"] for row in rows
            }
            observed_status: Counter[str] = Counter()
            observed_same: Counter[str] = Counter()
            observed_calls: Counter[str] = Counter()
            observed_tools: Counter[str] = Counter()
            observed_pairs = {
                f"S1.4.{form}": Counter() for form in range(1, 10)
            }
            for row in rows:
                self.assertNotIn(row["row_id"], all_row_ids)
                all_row_ids.add(row["row_id"])
                answer = row["messages"][-1]["content"]
                observed_status[answer["status"]] += 1
                same_gene = answer["result"].get(
                    "same_gene", "unavailable"
                )
                if same_gene == "unavailable":
                    same_key = "unavailable"
                elif same_gene is None:
                    same_key = "null"
                else:
                    same_key = str(same_gene).lower()
                observed_same[same_key] += 1
                calls = _calls(row)
                observed_calls[str(len(calls))] += 1
                if calls:
                    observed_tools.update(call["name"] for call in calls)
                else:
                    observed_tools["none"] += 1
                form = row["metadata"]["form"]
                if form != 9:
                    result = answer["result"]
                    observed_pairs[f"S1.4.{form}"][
                        f"{result['first']['item']}->{result['second']['item']}"
                    ] += 1
                    self.assertEqual(
                        {result["first"]["item"], result["second"]["item"]}
                        <= ITEMS,
                        True,
                    )
                    if result["same_gene"] is None:
                        self.assertEqual(
                            answer["continuation_decision"], "defer"
                        )
                current_user = next(
                    message["content"]
                    for message in row["messages"]
                    if message["role"] == "user"
                )
                self.assertEqual(
                    sum(
                        _matches_template(current_user, template)
                        for template in ALIGNMENT_PROMPTS[split][
                            "direct"
                        ]
                        + ALIGNMENT_PROMPTS[split]["step"]
                    ),
                    1,
                )
                self.assertFalse(
                    any(
                        _matches_template(current_user, template)
                        for other, framings in ALIGNMENT_PROMPTS.items()
                        if other != split
                        for templates in framings.values()
                        for template in templates
                    )
                )
            self.assertEqual(dict(observed_status), expected_status[split])
            self.assertEqual(dict(observed_same), expected_same_gene[split])
            self.assertEqual(dict(observed_calls), expected_call_counts[split])
            self.assertEqual(dict(observed_tools), expected_tool_names[split])
            pair_by_prefix: dict[str, str] = {}
            for row in rows:
                if row["metadata"]["form"] != 1:
                    continue
                tool_index = next(
                    index
                    for index, message in enumerate(row["messages"])
                    if message["role"] == "tool"
                )
                key = json.dumps(
                    row["messages"][:tool_index],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                result = row["messages"][-1]["content"]["result"]
                pair_by_prefix[key] = (
                    f"{result['first']['item']}->{result['second']['item']}"
                )
            for row in rows:
                if row["metadata"]["form"] != 9:
                    continue
                tool_index = next(
                    index
                    for index, message in enumerate(row["messages"])
                    if message["role"] == "tool"
                )
                key = json.dumps(
                    row["messages"][:tool_index],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                self.assertIn(key, pair_by_prefix)
                observed_pairs["S1.4.9"][pair_by_prefix[key]] += 1
            observed_pair_dict = {
                trace: dict(counts)
                for trace, counts in observed_pairs.items()
            }
            for trace, counts in observed_pair_dict.items():
                self.assertEqual(
                    counts, manifest["item_pair_counts"][split][trace]
                )

        self.assertTrue(
            units_by_split["train"].isdisjoint(units_by_split["validation"])
        )
        self.assertTrue(
            units_by_split["train"].isdisjoint(units_by_split["test"])
        )
        self.assertTrue(
            units_by_split["validation"].isdisjoint(units_by_split["test"])
        )
        evidence_ids = [record[0] for record in manifest["evidence"]["calls"]]
        self.assertEqual(len(evidence_ids), len(set(evidence_ids)))


if __name__ == "__main__":
    unittest.main()
