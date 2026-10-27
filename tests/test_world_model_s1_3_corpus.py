"""Focused tests for the canonical S1.3 ambiguous-symbol corpus builder."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import unittest

from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s0_corpus import EvidenceSequence
from runtime.world_model.s1_1_corpus import (
    IDENTITY_PROMPTS as S1_1_IDENTITY_PROMPTS,
)
from runtime.world_model.s1_3_corpus import (
    CHOICE_REPLIES,
    IDENTITY_PROMPTS,
    MULTIPLEX_ID,
    NONCHOICE_REPLIES,
    CorpusBuildError,
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
    split_row_counts,
)
from runtime.world_model.source_index import check_source_index
from runtime.world_model.sources import file_sha256
from runtime.world_model.tools import (
    MEMBERSHIP_TOOL,
    SYMBOL_TOOL,
    WorldModelToolRuntime,
)


SOURCE_INDEX = Path(
    "data/world_model_v3/source_indexes/"
    "world_model_source_index_"
    "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e.json.gz"
)


def _gene_id(value: int) -> str:
    return f"ENSG{value:011d}"


def _matches_template(text: str, template: str) -> bool:
    prefix, suffix = template.split("{request}")
    return text.startswith(prefix) and text.endswith(suffix)


def _fact(
    value: int,
    *,
    symbol: str,
    present_indexes: tuple[int, ...],
    split: str = "train",
    candidate_count: int = 2,
) -> GeneFact:
    candidate_gene_ids = tuple(
        _gene_id(value + index) for index in range(candidate_count)
    )
    candidate_details = {
        gene_id: {
            "chromosome": str((value + index) % 22 + 1),
            "biotype": (
                "protein_coding" if index % 2 == 0 else "lncRNA"
            ),
        }
        for index, gene_id in enumerate(candidate_gene_ids)
    }
    return GeneFact(
        fact_id=f"ambiguous_{split}_{symbol}",
        split_unit=f"component_{split}_{symbol}",
        gene_symbol=symbol,
        candidate_gene_ids=candidate_gene_ids,
        candidate_details=candidate_details,
        present_gene_ids=tuple(
            candidate_gene_ids[index] for index in present_indexes
        ),
    )


class WorldModelS13AllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.curriculum = self.contract["curriculum"]
        self.settings = self.curriculum["families"]["S1.3"]

    def test_split_and_form_allocations_are_exact(self) -> None:
        rows = split_row_counts(
            self.settings["train_rows"],
            self.curriculum["split_fractions"],
        )
        self.assertEqual(
            rows, {"train": 9000, "validation": 1125, "test": 1125}
        )
        expected = {
            "train": {
                1: 1000,
                2: 1000,
                3: 1000,
                4: 1000,
                5: 2070,
                6: 218,
                7: 412,
                8: 1000,
                9: 1000,
                10: 300,
            },
            "validation": {
                1: 125,
                2: 125,
                3: 125,
                4: 125,
                5: 259,
                6: 27,
                7: 52,
                8: 125,
                9: 125,
                10: 37,
            },
            "test": {
                1: 125,
                2: 125,
                3: 125,
                4: 125,
                5: 259,
                6: 27,
                7: 52,
                8: 125,
                9: 125,
                10: 37,
            },
        }
        actual = {
            split: form_counts(total, self.settings, self.curriculum)
            for split, total in rows.items()
        }
        self.assertEqual(actual, expected)
        for split, counts in actual.items():
            self.assertEqual(
                {
                    "0": counts[3] + counts[8] + counts[9],
                    "1": counts[1] + counts[2] + counts[4],
                    "2": counts[5] + counts[6] + counts[7] + counts[10],
                },
                {
                    "0": rows[split] // 3,
                    "1": rows[split] // 3,
                    "2": rows[split] // 3,
                },
            )

    def test_contract_records_every_allocation_decision(self) -> None:
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
                "membership_outcome_counts",
                "maximum_candidate_gene_ids",
                "failure_injection",
            ],
        )
        self.assertEqual(
            self.settings["allocations"],
            {
                "direct": 1000,
                "inside_task": 1000,
                "already_marked": 1000,
                "harness_near_miss": 1000,
                "membership_keeps_one": 2070,
                "membership_keeps_several": 218,
                "membership_keeps_none": 412,
                "user_chooses": 1000,
                "reply_does_not_choose": 1000,
                "membership_backend_unavailable": 300,
            },
        )
        self.assertEqual(
            self.settings["membership_outcome_counts"],
            {"keeps_one": 371, "keeps_several": 39, "keeps_none": 74},
        )
        self.assertEqual(self.settings["maximum_candidate_gene_ids"], 49)
        self.assertEqual(
            self.settings["failure_injection"],
            {
                "source_forms": [
                    "membership_keeps_one",
                    "membership_keeps_several",
                    "membership_keeps_none",
                ],
                "allocation": "replace",
                "call_stage": "membership",
            },
        )

    def test_largest_remainder_ties_follow_form_order(self) -> None:
        shares = {
            name: count / self.settings["train_rows"]
            for name, count in self.settings["allocations"].items()
        }
        counts = largest_remainder_counts(1125, shares)
        self.assertEqual(counts["membership_keeps_one"], 259)
        self.assertEqual(counts["membership_keeps_several"], 27)
        self.assertEqual(counts["membership_keeps_none"], 52)
        self.assertEqual(counts["membership_backend_unavailable"], 37)
        self.assertEqual(sum(counts.values()), 1125)

    def test_prompt_pool_is_the_shared_split_held_out_pool(self) -> None:
        self.assertIs(IDENTITY_PROMPTS, S1_1_IDENTITY_PROMPTS)
        phrases_by_split = {
            split: {
                template
                for templates in framings.values()
                for template in templates
            }
            for split, framings in IDENTITY_PROMPTS.items()
        }
        self.assertEqual(
            set(phrases_by_split), {"train", "validation", "test"}
        )
        for split, framings in IDENTITY_PROMPTS.items():
            self.assertEqual(set(framings), {"direct", "step"})
            self.assertTrue(
                all(templates for templates in framings.values())
            )
            held_out = set().union(
                *(
                    phrases
                    for other, phrases in phrases_by_split.items()
                    if other != split
                )
            )
            self.assertTrue(phrases_by_split[split].isdisjoint(held_out))

    def test_source_pool_counts_and_candidate_cap_are_exact(self) -> None:
        if not SOURCE_INDEX.is_file():
            self.skipTest(f"source index is absent: {SOURCE_INDEX}")
        payload = check_source_index(SOURCE_INDEX, self.contract)
        pools, source_outcomes = build_fact_pools(
            payload,
            s0_settings=self.curriculum["families"]["S0"],
            maximum_candidate_gene_ids=self.settings[
                "maximum_candidate_gene_ids"
            ],
        )
        self.assertEqual(
            source_outcomes,
            {"keeps_one": 371, "keeps_several": 39, "keeps_none": 74},
        )
        self.assertEqual(
            {
                split: {name: len(facts) for name, facts in split_pools.items()}
                for split, split_pools in pools.items()
            },
            {
                "train": {
                    "all": 385,
                    "keeps_one": 299,
                    "keeps_several": 31,
                    "keeps_none": 55,
                },
                "validation": {
                    "all": 48,
                    "keeps_one": 34,
                    "keeps_several": 4,
                    "keeps_none": 10,
                },
                "test": {
                    "all": 48,
                    "keeps_one": 38,
                    "keeps_several": 1,
                    "keeps_none": 9,
                },
            },
        )
        self.assertEqual(
            sum(source_outcomes.values())
            - sum(len(split_pools["all"]) for split_pools in pools.values()),
            3,
        )
        for split_pools in pools.values():
            self.assertTrue(
                all(
                    len(fact.candidate_gene_ids) <= 49
                    for fact in split_pools["all"]
                )
            )

        row_counts = split_row_counts(
            self.settings["train_rows"],
            self.curriculum["split_fractions"],
        )
        counts_by_split = {
            split: form_counts(total, self.settings, self.curriculum)
            for split, total in row_counts.items()
        }
        scenarios_by_split = select_scenarios(
            pools,
            counts_by_split,
            seed=self.settings["assignment_seed"],
            case_rate=self.curriculum["identity_case_perturbation"],
            framing_shares=self.curriculum["framing"],
        )
        for split, scenarios in scenarios_by_split.items():
            by_form = {
                form: [item for item in scenarios if item.form == form]
                for form in range(1, 11)
            }
            for form, items in by_form.items():
                applicable_framings = (
                    ("direct",)
                    if form == 1
                    else ("step",)
                    if form == 2
                    else ("direct", "step")
                )
                for framing in applicable_framings:
                    with self.subTest(
                        split=split, form=form, framing=framing
                    ):
                        self.assertEqual(
                            {
                                item.template_index
                                for item in items
                                if item.framing == framing
                            },
                            set(
                                range(
                                    len(IDENTITY_PROMPTS[split][framing])
                                )
                            ),
                        )

            for form in (5, 6, 7, 10):
                with self.subTest(split=split, form=form):
                    placements = Counter(
                        "harness"
                        if item.multiplex_in_harness
                        else "user_task"
                        for item in by_form[form]
                    )
                    self.assertGreater(placements["harness"], 0)
                    self.assertGreater(placements["user_task"], 0)

            expected_membership_structures = {
                (framing, template_index, multiplex_in_harness)
                for framing in ("direct", "step")
                for template_index in range(
                    len(IDENTITY_PROMPTS[split][framing])
                )
                for multiplex_in_harness in (False, True)
            }
            for form in (5, 6, 7):
                with self.subTest(
                    split=split, form=form, audit="pretool_structure"
                ):
                    self.assertEqual(
                        {
                            (
                                item.framing,
                                item.template_index,
                                item.multiplex_in_harness,
                            )
                            for item in by_form[form]
                        },
                        expected_membership_structures,
                    )

            successes = {
                item.scenario_id: item
                for form in (5, 6, 7)
                for item in by_form[form]
            }
            for failure in by_form[10]:
                success = successes[failure.paired_success_id]
                self.assertEqual(
                    failure.multiplex_in_harness,
                    success.multiplex_in_harness,
                )

            choice_templates = Counter(
                next(
                    index
                    for index, template in enumerate(CHOICE_REPLIES)
                    if template.format(gene_id=item.chosen_gene_id)
                    == item.reply
                )
                for item in by_form[8]
            )
            nonchoice_templates = Counter(
                NONCHOICE_REPLIES.index(item.reply)
                for item in by_form[9]
            )
            self.assertEqual(set(choice_templates), set(range(4)))
            self.assertEqual(set(nonchoice_templates), set(range(4)))
            for item in by_form[9]:
                if len(item.fact.candidate_gene_ids) > 2:
                    self.assertIsNone(
                        re.search(
                            r"\b(?:either|both)\b",
                            item.reply or "",
                            flags=re.IGNORECASE,
                        )
                    )


class WorldModelS13SelectionTests(unittest.TestCase):
    @staticmethod
    def _pools() -> dict[str, dict[str, list[GeneFact]]]:
        pools = {}
        for split_index, split in enumerate(("train", "validation", "test")):
            start = 1000 * (split_index + 1)
            keeps_one = [
                _fact(
                    start + 10 * index,
                    symbol=f"ONE{split_index}{index}",
                    present_indexes=(0,),
                    split=split,
                )
                for index in range(3)
            ]
            keeps_several = [
                _fact(
                    start + 100 + 10 * index,
                    symbol=f"SEVERAL{split_index}{index}",
                    present_indexes=(0, 1),
                    split=split,
                )
                for index in range(3)
            ]
            keeps_none = [
                _fact(
                    start + 200 + 10 * index,
                    symbol=f"NONE{split_index}{index}",
                    present_indexes=(),
                    split=split,
                )
                for index in range(3)
            ]
            pools[split] = {
                "all": keeps_one + keeps_several + keeps_none,
                "keeps_one": keeps_one,
                "keeps_several": keeps_several,
                "keeps_none": keeps_none,
            }
        return pools

    def test_selection_routes_forms_and_builds_exact_failure_pairs(self) -> None:
        counts = {
            split: {
                1: 8,
                2: 8,
                3: 8,
                4: 8,
                5: 24,
                6: 8,
                7: 8,
                8: 8,
                9: 8,
                10: 30,
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

        units_by_split = {}
        for split, scenarios in selected.items():
            with self.subTest(split=split):
                self.assertEqual(
                    Counter(item.form for item in scenarios),
                    Counter(counts[split]),
                )
                self.assertEqual(
                    Counter(item.framing for item in scenarios),
                    {"direct": 59, "step": 59},
                )
                self.assertEqual(
                    {item.source_trace for item in scenarios},
                    {f"S1.3.{form}" for form in range(1, 11)},
                )
                self.assertTrue(
                    all(
                        item.fact.membership_outcome
                        == {5: "keeps_one", 6: "keeps_several", 7: "keeps_none"}[
                            item.form
                        ]
                        for item in scenarios
                        if item.form in {5, 6, 7}
                    )
                )
                self.assertTrue(
                    all(
                        item.distractor is not None
                        and item.distractor.split_unit != item.fact.split_unit
                        for item in scenarios
                        if item.form == 4
                    )
                )
                self.assertTrue(
                    all(
                        item.chosen_gene_id in item.fact.candidate_gene_ids
                        and item.chosen_gene_id in (item.reply or "")
                        for item in scenarios
                        if item.form == 8
                    )
                )
                self.assertTrue(
                    all(
                        item.chosen_gene_id is None and item.reply is not None
                        for item in scenarios
                        if item.form == 9
                    )
                )

                successes = {
                    item.scenario_id: item
                    for item in scenarios
                    if item.form in {5, 6, 7}
                }
                failures = [item for item in scenarios if item.form == 10]
                self.assertEqual(
                    Counter(item.fact.membership_outcome for item in failures),
                    {"keeps_one": 23, "keeps_several": 2, "keeps_none": 5},
                )
                for failure in failures:
                    success = successes[failure.paired_success_id]
                    self.assertEqual(failure.fact, success.fact)
                    self.assertEqual(failure.input_value, success.input_value)
                    self.assertEqual(
                        failure.multiplex_in_harness,
                        success.multiplex_in_harness,
                    )
                    self.assertEqual(failure.framing, success.framing)
                    self.assertEqual(
                        failure.template_index, success.template_index
                    )
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


class WorldModelS13RowSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.tools = tool_definitions(self.contract)
        self.keeps_one = _fact(
            100, symbol="KEEPONE", present_indexes=(0,)
        )
        self.keeps_several = _fact(
            200, symbol="KEEPSEVERAL", present_indexes=(0, 1)
        )
        self.keeps_none = _fact(
            300, symbol="KEEPNONE", present_indexes=()
        )
        self.distractor = _fact(
            400, symbol="DISTRACTOR", present_indexes=(0,)
        )
        self.long_fact = _fact(
            500,
            symbol="LONGAMBIGUOUS",
            present_indexes=(),
            candidate_count=6,
        )
        facts = (
            self.keeps_one,
            self.keeps_several,
            self.keeps_none,
            self.distractor,
            self.long_fact,
        )
        genes_by_id = {
            gene_id: [
                fact.gene_symbol,
                fact.candidate_details[gene_id]["chromosome"],
                fact.candidate_details[gene_id]["biotype"],
            ]
            for fact in facts
            for gene_id in fact.candidate_gene_ids
        }
        multiplex_gene_ids = sorted(
            {
                gene_id
                for fact in facts
                for gene_id in fact.present_gene_ids
            }
        )
        payload = {
            "coordinate_system": {
                "ensembl_release": "116",
                "identifier_registry_id": "ensembl_homo_sapiens_116_v3",
                "multiplex_id": "full_brain_multiplex_v1",
                "multiplex_gene_rule": "test_fixture",
            },
            "genes_by_id": genes_by_id,
            "multiplex_gene_ids": multiplex_gene_ids,
            "sources": {
                "ensembl_gtf": {
                    "file_name": "registry.gtf.gz",
                    "sha256": "0" * 64,
                },
                "mentor_ev_dendrogram": {
                    "file_name": "dendrogram.tsv",
                    "sha256": "1" * 64,
                },
            },
        }
        self.runtime_payload = payload
        self.runtime = WorldModelToolRuntime(
            payload=payload,
            contract=self.contract,
            evidence_id_factory=EvidenceSequence(),
        )

    def _scenario(
        self,
        form: int,
        fact: GeneFact,
        *,
        framing: str = "direct",
        distractor: GeneFact | None = None,
        chosen_gene_id: str | None = None,
        reply: str | None = None,
        multiplex_in_harness: bool | None = None,
        paired_success_id: str | None = None,
    ) -> Scenario:
        return Scenario(
            split="train",
            form=form,
            fact=fact,
            framing=framing,
            template_index=0,
            occurrence=0,
            input_value=fact.gene_symbol,
            distractor=distractor,
            chosen_gene_id=chosen_gene_id,
            reply=reply,
            multiplex_in_harness=(
                form in {5, 6, 7, 10}
                if multiplex_in_harness is None
                else multiplex_in_harness
            ),
            paired_success_id=paired_success_id,
        )

    @staticmethod
    def _calls(row: dict[str, object]) -> list[dict[str, object]]:
        return [
            message["tool_call"]
            for message in row["messages"]
            if message["role"] == "assistant" and "tool_call" in message
        ]

    def test_selected_candidate_reasoning_changes_only_training_thinking(
        self,
    ) -> None:
        """Bind selection fields without changing answers or held-out rows."""

        chosen = self.keeps_none.candidate_gene_ids[1]
        scenario = self._scenario(
            8, self.keeps_none, chosen_gene_id=chosen, reply=f"Use {chosen}."
        )
        for split in ("train", "validation", "test"):
            for framing in ("direct", "step"):
                selected = replace(scenario, split=split, framing=framing)
                legacy, _ = build_row(
                    selected,
                    runtime=WorldModelToolRuntime(
                        payload=self.runtime_payload,
                        contract=self.contract,
                        evidence_id_factory=EvidenceSequence(),
                    ),
                    contract=self.contract,
                    tools=self.tools,
                )
                revised, _ = build_row(
                    selected,
                    runtime=WorldModelToolRuntime(
                        payload=self.runtime_payload,
                        contract=self.contract,
                        evidence_id_factory=EvidenceSequence(),
                    ),
                    contract=self.contract,
                    tools=self.tools,
                    train_multiplex_reasoning=True,
                )
                if split != "train":
                    self.assertEqual(revised, legacy)
                    continue
                final = revised["messages"][-1]
                self.assertIn(
                    "exactly gene_id, gene_symbol, and candidate_gene_ids",
                    final["thinking"],
                )
                self.assertIn(
                    "candidate_gene_ids list in its order", final["thinking"]
                )
                final["thinking"] = legacy["messages"][-1]["thinking"]
                self.assertEqual(revised, legacy)

    def test_all_ten_forms_have_the_reviewed_action_semantics(self) -> None:
        chosen = self.keeps_none.candidate_gene_ids[1]
        scenarios = {
            1: self._scenario(1, self.keeps_one),
            2: self._scenario(2, self.keeps_none, framing="step"),
            3: self._scenario(3, self.keeps_several),
            4: self._scenario(
                4, self.keeps_none, distractor=self.distractor
            ),
            5: self._scenario(5, self.keeps_one, framing="step"),
            6: self._scenario(6, self.keeps_several),
            7: self._scenario(7, self.keeps_none),
            8: self._scenario(
                8,
                self.keeps_none,
                framing="step",
                chosen_gene_id=chosen,
                reply=f"Use {chosen}.",
            ),
            9: self._scenario(
                9,
                self.keeps_none,
                reply="Use either candidate.",
            ),
            10: self._scenario(10, self.keeps_one, framing="step"),
        }
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
        expected_decision = {
            1: "defer",
            2: "defer",
            3: "defer",
            4: "defer",
            5: "continue",
            6: "defer",
            7: "defer",
            8: "continue",
            9: "defer",
            10: "defer",
        }
        expected_code = {
            1: "ambiguous_symbol",
            2: "ambiguous_symbol",
            3: "ambiguous_symbol",
            4: "ambiguous_symbol",
            5: None,
            6: "ambiguous_symbol",
            7: "no_multiplex_membership",
            8: None,
            9: "ambiguous_symbol",
            10: "tool_error",
        }
        expected_evidence = {
            1: 1,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 2,
            7: 2,
            8: 1,
            9: 1,
            10: 1,
        }

        for form, scenario in scenarios.items():
            with self.subTest(form=form):
                row, evidence = build_row(
                    scenario,
                    runtime=self.runtime,
                    contract=self.contract,
                    tools=self.tools,
                )
                calls = self._calls(row)
                answer = row["messages"][-1]["content"]
                self.assertEqual(
                    [call["name"] for call in calls], expected_calls[form]
                )
                self.assertEqual(answer["status"], expected_status[form])
                self.assertEqual(
                    answer["continuation_decision"], expected_decision[form]
                )
                self.assertEqual(
                    row["metadata"]["blocked_reason_code"],
                    expected_code[form],
                )
                self.assertEqual(
                    row["metadata"]["source_trace"], f"S1.3.{form}"
                )
                self.assertEqual(len(evidence), expected_evidence[form])
                self.assertEqual(row["tools"], self.tools)

                if calls:
                    self.assertEqual(
                        calls[0]["arguments"],
                        {"gene_symbol": scenario.input_value},
                    )
                if len(calls) == 2:
                    self.assertEqual(
                        calls[1]["arguments"],
                        {
                            "multiplex_id": "full_brain_multiplex_v1",
                            "gene_ids": list(scenario.fact.candidate_gene_ids),
                        },
                    )

                result = answer["result"]
                if answer["status"] == "ambiguous":
                    self.assertEqual(
                        result["candidate_details"],
                        scenario.fact.candidate_details,
                    )
                if form in {5, 6, 7}:
                    self.assertIsNotNone(answer["provenance"]["graph"])
                    self.assertEqual(
                        result["present_gene_ids"],
                        list(scenario.fact.present_gene_ids),
                    )
                else:
                    self.assertIsNone(answer["provenance"]["graph"])

        form_five = build_row(
            scenarios[5],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )[0]["messages"][-1]["content"]
        self.assertEqual(
            form_five["result"]["gene_id"],
            self.keeps_one.present_gene_ids[0],
        )
        form_eight = build_row(
            scenarios[8],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )[0]["messages"][-1]["content"]
        self.assertEqual(form_eight["result"]["gene_id"], chosen)
        failure = build_row(
            scenarios[10],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )[0]["messages"][-1]["content"]
        self.assertEqual(failure["result"]["failed_tool"], MEMBERSHIP_TOOL)
        self.assertEqual(failure["result"]["error_code"], "BACKEND_UNAVAILABLE")
        self.assertNotIn("present_gene_ids", failure["result"])

    def test_message_orders_match_each_form(self) -> None:
        chosen = self.keeps_none.candidate_gene_ids[0]
        scenarios = {
            1: self._scenario(1, self.keeps_one),
            2: self._scenario(2, self.keeps_one, framing="step"),
            3: self._scenario(3, self.keeps_one),
            4: self._scenario(
                4, self.keeps_one, distractor=self.distractor
            ),
            5: self._scenario(5, self.keeps_one),
            6: self._scenario(6, self.keeps_several),
            7: self._scenario(7, self.keeps_none),
            8: self._scenario(
                8,
                self.keeps_none,
                chosen_gene_id=chosen,
                reply=f"Choose {chosen}.",
            ),
            9: self._scenario(
                9, self.keeps_none, reply="Use either candidate."
            ),
            10: self._scenario(10, self.keeps_one),
        }
        expected = {
            1: ["system", "user", "assistant", "tool", "assistant"],
            2: ["system", "user", "assistant", "tool", "assistant"],
            3: ["system", "harness", "user", "assistant"],
            4: [
                "system",
                "harness",
                "user",
                "assistant",
                "tool",
                "assistant",
            ],
            5: [
                "system",
                "harness",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
            ],
            6: [
                "system",
                "harness",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
            ],
            7: [
                "system",
                "harness",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
            ],
            8: ["system", "user", "assistant", "user", "assistant"],
            9: ["system", "user", "assistant", "user", "assistant"],
            10: [
                "system",
                "harness",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
            ],
        }
        for form, scenario in scenarios.items():
            with self.subTest(form=form):
                row, _ = build_row(
                    scenario,
                    runtime=self.runtime,
                    contract=self.contract,
                    tools=self.tools,
                )
                self.assertEqual(
                    [message["role"] for message in row["messages"]],
                    expected[form],
                )

    def test_rendered_prompts_use_only_their_split_pool(self) -> None:
        for split, split_templates in IDENTITY_PROMPTS.items():
            foreign_templates = [
                template
                for other_split, framings in IDENTITY_PROMPTS.items()
                if other_split != split
                for templates in framings.values()
                for template in templates
            ]
            for framing, templates in split_templates.items():
                form = 1 if framing == "direct" else 2
                for template_index, template in enumerate(templates):
                    with self.subTest(
                        split=split,
                        framing=framing,
                        template_index=template_index,
                    ):
                        scenario = replace(
                            self._scenario(form, self.keeps_one),
                            split=split,
                            framing=framing,
                            template_index=template_index,
                        )
                        row, _ = build_row(
                            scenario,
                            runtime=self.runtime,
                            contract=self.contract,
                            tools=self.tools,
                        )
                        user_text = next(
                            message["content"]
                            for message in row["messages"]
                            if message["role"] == "user"
                        )
                        self.assertTrue(
                            _matches_template(user_text, template)
                        )
                        self.assertFalse(
                            any(
                                _matches_template(user_text, foreign)
                                for foreign in foreign_templates
                            )
                        )

    def test_resolved_results_follow_the_question_framing(self) -> None:
        chosen = self.keeps_one.candidate_gene_ids[0]
        for form in (5, 8):
            for framing, decision in (("direct", "stop"), ("step", "continue")):
                with self.subTest(form=form, framing=framing):
                    scenario = self._scenario(
                        form,
                        self.keeps_one,
                        framing=framing,
                        chosen_gene_id=chosen if form == 8 else None,
                        reply=f"Use {chosen}." if form == 8 else None,
                    )
                    row, _ = build_row(
                        scenario,
                        runtime=self.runtime,
                        contract=self.contract,
                        tools=self.tools,
                    )
                    answer = row["messages"][-1]["content"]
                    self.assertEqual(answer["continuation_decision"], decision)
                    self.assertIsNone(answer["blocked_reason"])

    def test_long_candidate_summaries_have_no_gene_anchor(self) -> None:
        row, _ = build_row(
            self._scenario(1, self.long_fact),
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        answer = row["messages"][-1]["content"]
        self.assertEqual(
            answer["result"]["candidate_gene_ids"],
            list(self.long_fact.candidate_gene_ids),
        )
        self.assertNotIn("<GENE:", answer["summary"])
        self.assertIn(str(len(self.long_fact.candidate_gene_ids)), answer["summary"])

    def test_failure_keeps_the_success_prompt_and_first_result(self) -> None:
        for form, fact in (
            (5, self.keeps_one),
            (6, self.keeps_several),
            (7, self.keeps_none),
        ):
            with self.subTest(source_form=form):
                success = self._scenario(form, fact, framing="step")
                failure = replace(
                    success,
                    form=10,
                    paired_success_id=success.scenario_id,
                )
                success_row, _ = build_row(
                    success,
                    runtime=self.runtime,
                    contract=self.contract,
                    tools=self.tools,
                )
                failure_row, _ = build_row(
                    failure,
                    runtime=self.runtime,
                    contract=self.contract,
                    tools=self.tools,
                )
                success_prefix = deepcopy(success_row["messages"][:6])
                failure_prefix = deepcopy(failure_row["messages"][:6])
                for messages in (success_prefix, failure_prefix):
                    for message in messages:
                        if message["role"] == "tool":
                            message["content"]["provenance"]["registry"].pop(
                                "evidence_id"
                            )
                self.assertEqual(success_prefix, failure_prefix)
                failure_membership = next(
                    message["content"]
                    for message in failure_row["messages"]
                    if message["role"] == "tool"
                    and message["name"] == MEMBERSHIP_TOOL
                )
                self.assertEqual(failure_membership["status"], "error")
                self.assertIsNone(failure_membership["provenance"])


class WorldModelS13AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        WorldModelS13RowSemanticsTests.setUp(self)

    def _scenario(
        self,
        form: int,
        fact: GeneFact,
        *,
        framing: str = "direct",
        distractor: GeneFact | None = None,
        chosen_gene_id: str | None = None,
        reply: str | None = None,
        multiplex_in_harness: bool | None = None,
        paired_success_id: str | None = None,
    ) -> Scenario:
        return WorldModelS13RowSemanticsTests._scenario(
            self,
            form,
            fact,
            framing=framing,
            distractor=distractor,
            chosen_gene_id=chosen_gene_id,
            reply=reply,
            multiplex_in_harness=multiplex_in_harness,
            paired_success_id=paired_success_id,
        )

    def test_thinking_filter_rejects_hidden_and_false_claims(self) -> None:
        scenario = self._scenario(5, self.keeps_one, framing="step")
        row, _ = build_row(
            scenario,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        self.assertEqual(filter_row_thinking(row, scenario), 3)

        mutations = {
            "premature_gene_id": (
                0,
                f"I know {self.keeps_one.candidate_gene_ids[0]} is the answer.",
                "reveals",
            ),
            "wrong_status": (
                2,
                "No candidate is present, so this is not_found. I can continue.",
                "(?:status.*wrong|wrong.*status)",
            ),
            "wrong_decision": (
                2,
                "One candidate is present. I must stop.",
                "(?:decision.*wrong|wrong.*decision)",
            ),
            "answer_schema": (
                2,
                "I will put both evidence blocks in the provenance field.",
                "discusses the answer schema",
            ),
        }
        for name, (assistant_index, thinking, message) in mutations.items():
            with self.subTest(rule=name):
                mutated = deepcopy(row)
                assistants = [
                    item
                    for item in mutated["messages"]
                    if item["role"] == "assistant"
                ]
                assistants[assistant_index]["thinking"] = thinking
                with self.assertRaisesRegex(CorpusBuildError, message):
                    filter_row_thinking(mutated, scenario)

    def test_thinking_filter_rejects_absence_after_membership_failure(self) -> None:
        success = self._scenario(5, self.keeps_one, framing="step")
        failure = replace(
            success,
            form=10,
            paired_success_id=success.scenario_id,
        )
        row, _ = build_row(
            failure,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        assistants = [
            item for item in row["messages"] if item["role"] == "assistant"
        ]
        assistants[-1]["thinking"] = (
            "The membership call failed. Neither candidate is present, so I "
            "must defer."
        )
        with self.assertRaises(CorpusBuildError):
            filter_row_thinking(row, failure)

    def test_visible_prompt_mask_keeps_context_and_masks_values(self) -> None:
        near_miss = self._scenario(
            4, self.keeps_none, distractor=self.distractor
        )
        near_miss_row, _ = build_row(
            near_miss,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        masked_near_miss = masked_visible_prompt(near_miss_row, near_miss)
        self.assertEqual(
            [line.split(":", 1)[0] for line in masked_near_miss.splitlines()],
            ["harness", "user"],
        )
        self.assertIn("Registry entries, Ensembl release", masked_near_miss)
        for value in (
            near_miss.fact.gene_symbol,
            *near_miss.fact.candidate_gene_ids,
            near_miss.distractor.gene_symbol,
            *near_miss.distractor.candidate_gene_ids,
        ):
            self.assertNotIn(value.casefold(), masked_near_miss.casefold())
        self.assertNotIn("ev_reg_", masked_near_miss)

        reply = "Use either candidate."
        multi_turn = self._scenario(9, self.keeps_none, reply=reply)
        multi_turn_row, _ = build_row(
            multi_turn,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        masked_multi_turn = masked_visible_prompt(
            multi_turn_row, multi_turn
        )
        self.assertEqual(
            [line.split(":", 1)[0] for line in masked_multi_turn.splitlines()],
            ["user", "assistant", "user"],
        )
        self.assertIn("status", masked_multi_turn)
        self.assertIn(reply.casefold(), masked_multi_turn.casefold())
        self.assertNotIn("ev_reg_", masked_multi_turn)
        for value in (
            multi_turn.fact.gene_symbol,
            *multi_turn.fact.candidate_gene_ids,
        ):
            self.assertNotIn(value.casefold(), masked_multi_turn.casefold())

    def test_masked_wording_classifier_retains_form_correlated_context(self) -> None:
        def fixtures(
            context: str, *, include_cue: bool
        ) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[Scenario]]]:
            rows_by_split: dict[str, list[dict[str, object]]] = {}
            scenarios_by_split: dict[str, list[Scenario]] = {}
            for split in ("train", "validation", "test"):
                rows = []
                scenarios = []
                for form, fact, cue in (
                    (3, self.keeps_one, "alpha"),
                    (4, self.keeps_none, "beta"),
                ):
                    for occurrence in range(4):
                        scenario = replace(
                            self._scenario(form, fact),
                            split=split,
                            occurrence=occurrence,
                        )
                        current_user = IDENTITY_PROMPTS[split]["direct"][
                            0
                        ].format(
                            request=f"source {fact.gene_symbol}"
                        )
                        if context == "harness":
                            messages = [
                                {"role": "system", "content": "Shared system."},
                                {
                                    "role": "harness",
                                    "content": (
                                        f"Routing cue {cue}. "
                                        if include_cue
                                        else "Registry item. "
                                    )
                                    + f"Symbol {fact.gene_symbol}; candidates "
                                    + ", ".join(fact.candidate_gene_ids),
                                },
                                {
                                    "role": "user",
                                    "content": current_user,
                                },
                            ]
                        else:
                            messages = [
                                {"role": "system", "content": "Shared system."},
                                {
                                    "role": "user",
                                    "content": (
                                        f"Earlier request: {fact.gene_symbol}"
                                    ),
                                },
                                {
                                    "role": "assistant",
                                    "content": (
                                        f"The {cue} route remains active."
                                        if include_cue
                                        else "The earlier request remains active."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": current_user,
                                },
                            ]
                        rows.append(
                            {
                                "row_id": (
                                    f"masked_{context}_{split}_{form}_{occurrence}"
                                ),
                                "messages": messages,
                            }
                        )
                        scenarios.append(scenario)
                rows_by_split[split] = rows
                scenarios_by_split[split] = scenarios
            return rows_by_split, scenarios_by_split

        rows, scenarios = fixtures("harness", include_cue=False)
        result = masked_wording_classifier_audit(rows, scenarios)
        self.assertTrue(result["pass"])
        self.assertEqual(result["scope"], "outer_wrapper_only")
        self.assertEqual(result["splits"]["validation"]["accuracy"], 0.5)
        self.assertEqual(result["splits"]["test"]["accuracy"], 0.5)

        for context in ("harness", "earlier_turn"):
            with self.subTest(context=context):
                rows, scenarios = fixtures(context, include_cue=True)
                wrapper_result = masked_wording_classifier_audit(
                    rows, scenarios
                )
                self.assertTrue(wrapper_result["pass"])
                diagnostic = full_visible_prompt_classifier_diagnostic(
                    rows, scenarios
                )
                self.assertEqual(
                    diagnostic["scope"], "full_visible_prompt"
                )
                self.assertTrue(diagnostic["diagnostic_only"])
                self.assertIsNone(diagnostic["pass"])
                for split in ("train", "validation", "test"):
                    metrics = diagnostic["splits"][split]
                    self.assertEqual(metrics["accuracy"], 1.0)
                    self.assertFalse(metrics["gate_applied"])
                    self.assertIsNone(metrics["threshold"])
                    self.assertIsNone(metrics["pass"])

    def test_duplicate_audit_rejects_an_ordinary_copy(self) -> None:
        scenario = self._scenario(1, self.keeps_one)
        row, _ = build_row(
            scenario,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        duplicate_scenario = replace(scenario, occurrence=1)
        duplicate = deepcopy(row)
        duplicate["row_id"] = "duplicate_with_new_evidence"
        tool_result = next(
            item["content"]
            for item in duplicate["messages"]
            if item["role"] == "tool"
        )
        tool_result["provenance"]["registry"]["evidence_id"] = (
            "ev_reg_change1"
        )
        answer = duplicate["messages"][-1]["content"]
        answer["provenance"]["registry"]["evidence_id"] = "ev_reg_change2"
        answer["summary"] = "A different final answer with the same outcome."

        with self.assertRaisesRegex(
            CorpusBuildError,
            "repeats a .*prompt and tool outcome",
        ):
            duplicate_prompt_outcome_audit(
                {
                    "train": [row, duplicate],
                    "validation": [],
                    "test": [],
                },
                {
                    "train": [scenario, duplicate_scenario],
                    "validation": [],
                    "test": [],
                },
            )

    def test_duplicate_audit_allows_the_failure_pair(self) -> None:
        success = self._scenario(5, self.keeps_one, framing="step")
        failure = replace(
            success,
            form=10,
            paired_success_id=success.scenario_id,
        )
        success_row, _ = build_row(
            success,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        failure_row, _ = build_row(
            failure,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        result = duplicate_prompt_outcome_audit(
            {
                "train": [success_row, failure_row],
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


class WorldModelS13GeneratedCorpusTests(unittest.TestCase):
    def test_generated_manifest_and_rows_reconcile_when_present(self) -> None:
        root = Path("data/world_model_v3/corpora/s1/s1_3")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"generated corpus is absent: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["section"], "S1.3")
        self.assertEqual(
            manifest["row_counts"],
            {"train": 9000, "validation": 1125, "test": 1125},
        )
        contract = load_contract()
        curriculum = contract["curriculum"]
        settings = curriculum["families"]["S1.3"]
        expected_counts = {
            split: form_counts(total, settings, curriculum)
            for split, total in manifest["row_counts"].items()
        }
        expected_status = {
            split: {
                "ambiguous": sum(
                    counts[form] for form in (1, 2, 3, 4, 6, 9, 10)
                ),
                "not_found": counts[7],
                "resolved": counts[5] + counts[8],
            }
            for split, counts in expected_counts.items()
        }
        expected_call_counts = {
            split: {
                "0": counts[3] + counts[8] + counts[9],
                "1": counts[1] + counts[2] + counts[4],
                "2": counts[5] + counts[6] + counts[7] + counts[10],
            }
            for split, counts in expected_counts.items()
        }
        self.assertEqual(
            manifest["form_counts"],
            {
                split: {
                    f"S1.3.{form}": count
                    for form, count in counts.items()
                }
                for split, counts in expected_counts.items()
            },
        )
        self.assertEqual(manifest["status_counts"], expected_status)
        self.assertEqual(manifest["tool_call_counts"], expected_call_counts)
        expected_prompt_hash = hashlib.sha256(
            json.dumps(
                S1_1_IDENTITY_PROMPTS,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            manifest["prompt_templates_sha256"], expected_prompt_hash
        )
        self.assertEqual(
            manifest["failure_injection"]["pair_counts"],
            {split: counts[10] for split, counts in expected_counts.items()},
        )
        self.assertEqual(
            manifest["sources"]["ambiguous_symbol_count"], 484
        )
        self.assertEqual(
            manifest["sources"]["membership_outcome_counts"],
            {"keeps_one": 371, "keeps_several": 39, "keeps_none": 74},
        )
        self.assertEqual(manifest["sources"]["excluded_over_cap_count"], 3)
        self.assertEqual(
            manifest["sources"]["candidate_count_distribution"],
            {
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
        )
        self.assertEqual(
            manifest["sources"]["present_candidate_count_distribution"],
            {
                "train": {
                    "0": 55,
                    "1": 299,
                    "2": 23,
                    "3": 2,
                    "4": 2,
                    "5": 1,
                    "9": 1,
                    "10": 1,
                    "13": 1,
                },
                "validation": {"0": 10, "1": 34, "2": 2, "3": 2},
                "test": {"0": 9, "1": 38, "2": 1},
            },
        )
        self.assertEqual(
            manifest["sources"]["ambiguity_class_counts"],
            {
                "train": {
                    "par_x_y_protein_coding": 15,
                    "pseudogene_lncRNA": 302,
                    "multicopy": 22,
                    "long_list_above_five": 12,
                },
                "validation": {
                    "par_x_y_protein_coding": 3,
                    "pseudogene_lncRNA": 34,
                    "multicopy": 4,
                    "long_list_above_five": 2,
                },
                "test": {
                    "par_x_y_protein_coding": 0,
                    "pseudogene_lncRNA": 42,
                    "multicopy": 1,
                    "long_list_above_five": 0,
                },
            },
        )
        for split in ("train", "validation"):
            self.assertTrue(
                all(
                    count > 0
                    for count in manifest["sources"][
                        "ambiguity_class_counts"
                    ][split].values()
                )
            )
        wrapper_classifier = manifest["masked_wording_classifier"]
        self.assertEqual(
            wrapper_classifier["scope"], "outer_wrapper_only"
        )
        self.assertTrue(wrapper_classifier["pass"])
        self.assertIsNone(wrapper_classifier["splits"]["train"]["pass"])
        for split in ("validation", "test"):
            self.assertTrue(
                wrapper_classifier["splits"][split]["gate_applied"]
            )
            self.assertTrue(wrapper_classifier["splits"][split]["pass"])

        full_classifier = manifest["full_visible_prompt_classifier"]
        self.assertEqual(full_classifier["scope"], "full_visible_prompt")
        self.assertTrue(full_classifier["diagnostic_only"])
        self.assertIsNone(full_classifier["pass"])
        for split in ("train", "validation", "test"):
            metrics = full_classifier["splits"][split]
            self.assertEqual(
                metrics["row_count"], manifest["row_counts"][split]
            )
            self.assertEqual(
                metrics["accuracy"],
                metrics["correct"] / metrics["row_count"],
            )
            self.assertFalse(metrics["gate_applied"])
            self.assertIsNone(metrics["threshold"])
            self.assertIsNone(metrics["pass"])

        prompt_structure = manifest["prompt_structure_audit"]
        self.assertTrue(prompt_structure["pass"])
        for split in ("train", "validation", "test"):
            self.assertEqual(
                prompt_structure["registry_harness_counts"][split],
                {
                    "S1.3.3": expected_counts[split][3],
                    "S1.3.4": expected_counts[split][4],
                    "unexpected": 0,
                },
            )
            self.assertEqual(
                prompt_structure["earlier_turn_counts"][split],
                {
                    "S1.3.8": expected_counts[split][8],
                    "S1.3.9": expected_counts[split][9],
                    "unexpected": 0,
                },
            )
            shared_cue_count = 2 * sum(
                len(templates)
                for templates in IDENTITY_PROMPTS[split].values()
            )
            self.assertEqual(
                prompt_structure["membership_pretool_cue_counts"][split],
                {
                    "keeps_one": shared_cue_count,
                    "keeps_several": shared_cue_count,
                    "keeps_none": shared_cue_count,
                    "shared_cue_count": shared_cue_count,
                },
            )
        self.assertTrue(manifest["thinking_filter"]["pass"])
        self.assertTrue(manifest["duplicate_prompt_tool_outcome"]["pass"])

        false_defer = manifest["false_defer_audit"]
        expected_false_defer_eligible = {
            split: counts[5] + counts[8]
            for split, counts in expected_counts.items()
        }
        self.assertEqual(
            false_defer["eligible_row_count"],
            sum(expected_false_defer_eligible.values()),
        )
        self.assertEqual(false_defer["false_defer_count"], 0)
        self.assertEqual(false_defer["false_defer_rate"], 0.0)
        self.assertTrue(false_defer["pass"])
        for split, eligible_count in expected_false_defer_eligible.items():
            self.assertEqual(
                false_defer["splits"][split],
                {
                    "eligible_row_count": eligible_count,
                    "false_defer_count": 0,
                    "false_defer_rate": 0.0,
                    "pass": True,
                },
            )

        expected_template_keys = {}
        for split in ("train", "validation", "test"):
            keys = set()
            for form in range(1, 11):
                framings = (
                    ("direct",)
                    if form == 1
                    else ("step",)
                    if form == 2
                    else ("direct", "step")
                )
                keys.update(
                    f"S1.3.{form}:{framing}:{index}"
                    for framing in framings
                    for index in range(
                        len(IDENTITY_PROMPTS[split][framing])
                    )
                )
            expected_template_keys[split] = keys
            template_counts = manifest["prompt_template_counts"][split]
            self.assertEqual(set(template_counts), keys)
            self.assertTrue(all(count > 0 for count in template_counts.values()))
            self.assertEqual(
                sum(template_counts.values()), manifest["row_counts"][split]
            )

            reply_counts = manifest["reply_template_counts"][split]
            self.assertEqual(set(reply_counts), {"S1.3.8", "S1.3.9"})
            for form in (8, 9):
                counts = reply_counts[f"S1.3.{form}"]
                self.assertEqual(set(counts), {"0", "1", "2", "3"})
                self.assertTrue(all(count > 0 for count in counts.values()))
                self.assertEqual(
                    sum(counts.values()), expected_counts[split][form]
                )

            placements = manifest["multiplex_placement_counts"][split]
            self.assertEqual(
                set(placements),
                {"S1.3.5", "S1.3.6", "S1.3.7", "S1.3.10"},
            )
            for form in (5, 6, 7, 10):
                counts = placements[f"S1.3.{form}"]
                self.assertEqual(set(counts), {"harness", "user_task"})
                self.assertGreater(counts["harness"], 0)
                self.assertGreater(counts["user_task"], 0)
                self.assertEqual(
                    sum(counts.values()), expected_counts[split][form]
                )

        all_row_ids: set[str] = set()
        units_by_split: dict[str, set[str]] = {}
        observed_templates: dict[str, Counter[str]] = {}
        observed_replies: dict[str, dict[str, Counter[str]]] = {}
        observed_placements: dict[str, dict[str, Counter[str]]] = {}
        for split in ("train", "validation", "test"):
            path = root / f"{split}.jsonl"
            with path.open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), manifest["row_counts"][split])
            self.assertEqual(
                file_sha256(path), manifest["files"][path.name]["sha256"]
            )
            self.assertEqual(
                Counter(row["metadata"]["form"] for row in rows),
                Counter(expected_counts[split]),
            )
            self.assertEqual(
                {row["metadata"]["source_trace"] for row in rows},
                {f"S1.3.{form}" for form in range(1, 11)},
            )
            units_by_split[split] = {
                row["metadata"]["split_unit"] for row in rows
            }
            observed_templates[split] = Counter()
            observed_replies[split] = {
                "S1.3.8": Counter(),
                "S1.3.9": Counter(),
            }
            observed_placements[split] = {
                f"S1.3.{form}": Counter() for form in (5, 6, 7, 10)
            }
            for row in rows:
                row_id = row["row_id"]
                self.assertNotIn(row_id, all_row_ids)
                all_row_ids.add(row_id)
                form = row["metadata"]["form"]
                calls = [
                    message["tool_call"]
                    for message in row["messages"]
                    if message["role"] == "assistant"
                    and "tool_call" in message
                ]
                answer = row["messages"][-1]["content"]
                current_user = [
                    message["content"]
                    for message in row["messages"]
                    if message["role"] == "user"
                ][-1]
                template_matches = []
                for framing, templates in IDENTITY_PROMPTS[split].items():
                    for index, template in enumerate(templates):
                        if _matches_template(current_user, template):
                            template_matches.append((framing, index))
                self.assertEqual(len(template_matches), 1)
                self.assertFalse(
                    any(
                        _matches_template(current_user, template)
                        for other_split, framings in IDENTITY_PROMPTS.items()
                        if other_split != split
                        for templates in framings.values()
                        for template in templates
                    )
                )
                framing, template_index = template_matches[0]
                observed_templates[split][
                    f"S1.3.{form}:{framing}:{template_index}"
                ] += 1

                if form in {5, 6, 7, 10}:
                    harness_has_multiplex = any(
                        message["role"] == "harness"
                        and MULTIPLEX_ID in message["content"]
                        for message in row["messages"]
                    )
                    user_has_multiplex = MULTIPLEX_ID in current_user
                    self.assertNotEqual(harness_has_multiplex, user_has_multiplex)
                    placement = (
                        "harness" if harness_has_multiplex else "user_task"
                    )
                    observed_placements[split][f"S1.3.{form}"][placement] += 1

                if form == 8:
                    chosen = answer["result"]["gene_id"]
                    indexes = [
                        index
                        for index, template in enumerate(CHOICE_REPLIES)
                        if template.format(gene_id=chosen).rstrip(".!?")
                        in current_user
                    ]
                    self.assertEqual(len(indexes), 1)
                    observed_replies[split]["S1.3.8"][str(indexes[0])] += 1
                elif form == 9:
                    indexes = [
                        index
                        for index, template in enumerate(NONCHOICE_REPLIES)
                        if template.rstrip(".!?") in current_user
                    ]
                    self.assertEqual(len(indexes), 1)
                    observed_replies[split]["S1.3.9"][str(indexes[0])] += 1
                    if len(answer["result"]["candidate_gene_ids"]) > 2:
                        self.assertIsNone(
                            re.search(
                                r"\b(?:either|both)\b",
                                current_user,
                                flags=re.IGNORECASE,
                            )
                        )
                self.assertEqual(
                    len(calls),
                    0 if form in {3, 8, 9} else 1 if form in {1, 2, 4} else 2,
                )
                if calls:
                    self.assertEqual(calls[0]["name"], SYMBOL_TOOL)
                registry_harnesses = [
                    message
                    for message in row["messages"]
                    if message["role"] == "harness"
                    and message["content"].startswith(
                        "Registry entries, Ensembl release 116:"
                    )
                ]
                self.assertEqual(
                    len(registry_harnesses), 1 if form in {3, 4} else 0
                )
                if form in {8, 9}:
                    self.assertEqual(
                        [
                            message["role"]
                            for message in row["messages"]
                        ],
                        [
                            "system",
                            "user",
                            "assistant",
                            "user",
                            "assistant",
                        ],
                    )
                    self.assertNotIn("thinking", row["messages"][2])
                    self.assertNotIn("tool_call", row["messages"][2])
                if len(calls) == 2:
                    self.assertEqual(calls[1]["name"], MEMBERSHIP_TOOL)
                    self.assertEqual(
                        calls[1]["arguments"]["gene_ids"],
                        next(
                            message["content"]["candidate_gene_ids"]
                            for message in row["messages"]
                            if message["role"] == "tool"
                            and message["name"] == SYMBOL_TOOL
                        ),
                    )
                if form in {1, 2, 3, 4, 6, 9, 10}:
                    self.assertIn("candidate_details", answer["result"])
                if form in {5, 8}:
                    self.assertIn(
                        answer["result"]["gene_id"],
                        answer["result"]["candidate_gene_ids"],
                    )
                if form == 7:
                    self.assertEqual(answer["result"]["present_gene_ids"], [])
                if form == 10:
                    membership_result = next(
                        message["content"]
                        for message in row["messages"]
                        if message["role"] == "tool"
                        and message["name"] == MEMBERSHIP_TOOL
                    )
                    self.assertEqual(membership_result["status"], "error")
                    self.assertIsNone(membership_result["provenance"])
                    self.assertIsNone(answer["provenance"]["graph"])

            self.assertEqual(
                dict(observed_templates[split]),
                manifest["prompt_template_counts"][split],
            )
            self.assertEqual(
                {
                    trace: dict(counts)
                    for trace, counts in observed_replies[split].items()
                },
                manifest["reply_template_counts"][split],
            )
            self.assertEqual(
                {
                    trace: dict(counts)
                    for trace, counts in observed_placements[split].items()
                },
                manifest["multiplex_placement_counts"][split],
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


if __name__ == "__main__":
    unittest.main()
