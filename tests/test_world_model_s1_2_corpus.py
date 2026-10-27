"""Focused tests for the canonical S1.2 corpus builder."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s0_corpus import EvidenceSequence
from runtime.world_model.s1_overlay import S1OverlayError
from runtime.world_model.s1_2_corpus import (
    IDENTITY_PROMPTS,
    CorpusBuildError,
    GeneFact,
    Scenario,
    audit_rows,
    build_row,
    duplicate_prompt_outcome_audit,
    filter_row_thinking,
    form_counts,
    largest_remainder_counts,
    masked_wording_classifier_audit,
    override_train_counts,
    select_scenarios,
    split_row_counts,
)
from runtime.world_model.sources import file_sha256
from runtime.world_model.tools import (
    GENE_ID_TOOL,
    SYMBOL_TOOL,
    WorldModelToolRuntime,
)


def _gene_id(value: int) -> str:
    return f"ENSG{value:011d}"


def _transcript_id(value: int) -> str:
    return f"ENST{value:011d}"


def _fact(
    value: int,
    *,
    split: str = "train",
    symbol: str | None = None,
    version: int | None = 1,
    kind: str = "current",
    species: str = "human",
    transcript_id: str | None = None,
    renamed: bool = False,
) -> GeneFact:
    return GeneFact(
        fact_id=f"fact_{split}_{kind}_{value}",
        split_units=(f"unit_{split}_{kind}_{value}",),
        gene_id=(
            f"ENSMUSG{value:011d}" if species == "mouse" else _gene_id(value)
        ),
        gene_symbol=symbol,
        gene_version=version,
        species=species,
        kind=kind,
        transcript_id=transcript_id,
        renamed=renamed,
    )


class WorldModelS12AllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.curriculum = self.contract["curriculum"]
        self.settings = self.curriculum["families"]["S1.2"]

    def test_split_and_form_allocations_are_exact(self) -> None:
        rows = split_row_counts(
            self.settings["train_rows"],
            self.curriculum["split_fractions"],
        )
        self.assertEqual(
            rows, {"train": 11000, "validation": 1375, "test": 1375}
        )
        expected = {
            "train": {
                1: 1600,
                2: 1500,
                3: 1000,
                4: 1000,
                5: 1000,
                6: 500,
                7: 1000,
                8: 500,
                9: 500,
                10: 1000,
                11: 400,
                12: 1000,
            },
            "validation": {
                1: 200,
                2: 188,
                3: 125,
                4: 125,
                5: 125,
                6: 63,
                7: 125,
                8: 62,
                9: 62,
                10: 125,
                11: 50,
                12: 125,
            },
            "test": {
                1: 200,
                2: 188,
                3: 125,
                4: 125,
                5: 125,
                6: 63,
                7: 125,
                8: 62,
                9: 62,
                10: 125,
                11: 50,
                12: 125,
            },
        }
        actual = {
            split: form_counts(total, self.settings, self.curriculum)
            for split, total in rows.items()
        }
        self.assertEqual(actual, expected)
        for split, counts in actual.items():
            tool_rows = sum(
                counts[form]
                for form in (1, 2, 3, 4, 5, 6, 8, 9, 11)
            )
            self.assertEqual(counts[11] * 20, tool_rows, split)

    def test_contract_records_every_allocation_decision(self) -> None:
        self.assertEqual(self.settings["base_train_rows"], 10000)
        self.assertEqual(
            self.settings["added_form_rows"],
            {"transcript_wrong_type": 1000},
        )
        self.assertEqual(
            self.settings["failure_injection"],
            {
                "source_form": "direct",
                "allocation": "replace",
                "call_stage": "first",
            },
        )
        self.assertEqual(
            self.settings["shares"],
            {
                "direct": 0.20,
                "inside_task": 0.15,
                "version_suffix": 0.10,
                "no_symbol": 0.10,
                "not_in_release": 0.10,
                "malformed": 0.05,
                "harness_exact": 0.10,
                "harness_near_miss": 0.05,
                "symbol_input": 0.05,
                "other_species": 0.10,
            },
        )

    def test_largest_remainder_ties_follow_contract_order(self) -> None:
        counts = largest_remainder_counts(1250, self.settings["shares"])
        self.assertEqual(counts["inside_task"], 188)
        self.assertEqual(counts["malformed"], 63)
        self.assertEqual(counts["harness_near_miss"], 62)
        self.assertEqual(counts["symbol_input"], 62)
        self.assertEqual(sum(counts.values()), 1250)

    def test_train_override_keeps_the_total_and_every_failure_pair(self) -> None:
        counts = {1: 500, 10: 1000, 11: 400, 12: 1000}
        moved = {1: 800, 10: 650, 11: 800, 12: 650}
        self.assertEqual(override_train_counts(counts, moved), moved)
        with self.assertRaisesRegex(S1OverlayError, "change the total"):
            override_train_counts(counts, {11: 800})
        with self.assertRaisesRegex(CorpusBuildError, "failure pair"):
            override_train_counts(counts, {1: 400, 11: 500})
        with self.assertRaisesRegex(S1OverlayError, "no form 13"):
            override_train_counts(counts, {13: 1})


class WorldModelS12SelectionTests(unittest.TestCase):
    def _pools(self) -> dict[str, dict[str, list[GeneFact]]]:
        pools = {}
        for split_index, split in enumerate(("train", "validation", "test")):
            start = 100 * (split_index + 1)
            named = [
                _fact(
                    start + index,
                    split=split,
                    symbol=f"G{split_index}{index}",
                    renamed=index == 0,
                )
                for index in range(3)
            ]
            pools[split] = {
                "current_named": named,
                "current_unnamed": [
                    _fact(start + 10, split=split, symbol=None)
                ],
                "symbol_input": named,
                "retired": [
                    _fact(
                        start + 20,
                        split=split,
                        version=None,
                        kind="retired",
                    )
                ],
                "mouse": [
                    _fact(
                        start + 30,
                        split=split,
                        symbol=f"Mouse{split_index}",
                        version=None,
                        kind="mouse",
                        species="mouse",
                    )
                ],
                "transcript": [
                    _fact(
                        start + 40,
                        split=split,
                        symbol=f"T{split_index}",
                        kind="transcript",
                        transcript_id=_transcript_id(start + 40),
                    )
                ],
            }
        return pools

    def test_selection_routes_forms_and_builds_exact_failure_pairs(self) -> None:
        counts = {
            split: {
                1: 8,
                2: 12,
                3: 8,
                4: 8,
                5: 8,
                6: 8,
                7: 8,
                8: 8,
                9: 8,
                10: 8,
                11: 4,
                12: 8,
            }
            for split in ("train", "validation", "test")
        }
        selected = select_scenarios(
            self._pools(), counts, seed=1729, case_rate=0
        )

        expected_kinds = {
            1: "current",
            2: "current",
            3: "current",
            4: "current",
            5: "retired",
            6: "current",
            7: "current",
            8: "current",
            9: "current",
            10: "mouse",
            11: "current",
            12: "transcript",
        }
        for split, scenarios in selected.items():
            self.assertEqual(len(scenarios), 96)
            self.assertEqual(
                Counter(item.form for item in scenarios), Counter(counts[split])
            )
            self.assertEqual(
                Counter(item.framing for item in scenarios),
                {"direct": 48, "step": 48},
            )
            self.assertEqual(
                {item.source_trace for item in scenarios},
                {f"S1.2.{form}" for form in range(1, 13)},
            )
            self.assertTrue(
                all(item.fact.kind == expected_kinds[item.form] for item in scenarios)
            )
            self.assertTrue(
                all(
                    item.distractor is not None
                    and item.distractor.gene_id != item.fact.gene_id
                    for item in scenarios
                    if item.form == 8
                )
            )

            successes = {
                item.scenario_id: item for item in scenarios if item.form == 1
            }
            failures = [item for item in scenarios if item.form == 11]
            self.assertEqual(len(failures), 4)
            self.assertEqual(
                {item.template_index for item in failures}, {0, 1, 2, 3}
            )
            for failure in failures:
                success = successes[failure.paired_success_id]
                self.assertEqual(failure.fact, success.fact)
                self.assertEqual(failure.input_value, success.input_value)
                self.assertEqual(failure.template_index, success.template_index)
                self.assertEqual(failure.framing, success.framing)
                self.assertEqual(failure.framing, "direct")


class WorldModelS12RowSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.tools = tool_definitions(self.contract)
        self.named = _fact(1, symbol="ONE", version=3, renamed=True)
        self.unnamed = _fact(2, symbol=None, version=4)
        self.retired = _fact(3, version=None, kind="retired")
        self.other_named = _fact(4, symbol="FOUR", version=2)
        self.mouse = _fact(
            5,
            symbol="MouseFive",
            version=None,
            kind="mouse",
            species="mouse",
        )
        self.transcript = _fact(
            1,
            symbol="ONE",
            version=3,
            kind="transcript",
            transcript_id=_transcript_id(1),
        )
        payload = {
            "coordinate_system": {
                "ensembl_release": "116",
                "identifier_registry_id": "ensembl_homo_sapiens_116_v3",
                "multiplex_id": "full_brain_multiplex_v1",
                "multiplex_gene_rule": "test_fixture",
            },
            "genes_by_id": {
                self.named.gene_id: ["ONE", "1", "protein_coding"],
                self.unnamed.gene_id: [None, "1", "lncRNA"],
                self.other_named.gene_id: ["FOUR", "1", "protein_coding"],
            },
            "multiplex_gene_ids": [],
            "sources": {
                "ensembl_gtf": {"file_name": "registry.gtf.gz", "sha256": "0" * 64},
                "mentor_ev_dendrogram": {
                    "file_name": "dendrogram.tsv",
                    "sha256": "1" * 64,
                },
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
        fact: GeneFact,
        input_value: str,
        *,
        framing: str = "direct",
        distractor: GeneFact | None = None,
    ) -> Scenario:
        return Scenario(
            split="train",
            form=form,
            fact=fact,
            framing=framing,
            template_index=0,
            occurrence=0,
            input_value=input_value,
            distractor=distractor,
        )

    @staticmethod
    def _calls(row: dict[str, object]) -> list[dict[str, object]]:
        return [
            message["tool_call"]
            for message in row["messages"]
            if message["role"] == "assistant" and "tool_call" in message
        ]

    def test_declared_banks_reject_arbitrary_failure_pair_prompts(self) -> None:
        """A scoped wrapper override must still validate both paired requests."""

        success = self._scenario(1, self.named, self.named.gene_id)
        failure = replace(success, form=11, paired_success_id=success.scenario_id)
        restored = deepcopy(IDENTITY_PROMPTS)
        restored["train"]["direct"] = (
            "Only this needs an answer: {request}.",
            *restored["train"]["direct"][1:],
        )
        for corrupt_form in (1, 11):
            with self.subTest(corrupt_form=corrupt_form):
                rows = []
                for case in (success, failure):
                    row, _ = build_row(
                        case, runtime=self.runtime, contract=self.contract,
                        tools=self.tools,
                        prompts=restored if case.form == 1 else IDENTITY_PROMPTS,
                    )
                    if case.form == corrupt_form:
                        row["messages"][1]["content"] += " An undeclared request."
                    rows.append(row)
                with self.assertRaisesRegex(CorpusBuildError, "failure pair changed its prompt"):
                    audit_rows(
                        {"train": rows}, {"train": [success, failure]},
                        {"train": {1: 1, 11: 1}}, train_form_prompts={1: restored},
                    )

    def test_all_twelve_forms_have_the_reviewed_action_semantics(self) -> None:
        scenarios = {
            1: self._scenario(1, self.named, self.named.gene_id),
            2: self._scenario(
                2, self.named, self.named.gene_id, framing="step"
            ),
            3: self._scenario(
                3,
                self.named,
                f"{self.named.gene_id}.{self.named.gene_version}",
            ),
            4: self._scenario(4, self.unnamed, self.unnamed.gene_id),
            5: self._scenario(5, self.retired, self.retired.gene_id),
            6: self._scenario(6, self.named, self.named.gene_id[:-1]),
            7: self._scenario(
                7, self.named, self.named.gene_id, framing="step"
            ),
            8: self._scenario(
                8,
                self.named,
                self.named.gene_id,
                distractor=self.other_named,
            ),
            9: self._scenario(9, self.named, "ONE", framing="step"),
            10: self._scenario(10, self.mouse, self.mouse.gene_id),
            11: self._scenario(11, self.named, self.named.gene_id),
            12: self._scenario(
                12, self.transcript, self.transcript.transcript_id or ""
            ),
        }
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
        expected_code = {
            5: "not_found",
            6: "invalid_input",
            10: "out_of_scope_species",
            11: "tool_error",
            12: "wrong_input_type",
        }
        expected_evidence = {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 1,
            10: 0,
            11: 0,
            12: 0,
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
                    row["metadata"]["blocked_reason_code"],
                    expected_code.get(form),
                )
                self.assertEqual(
                    row["metadata"]["source_trace"], f"S1.2.{form}"
                )
                self.assertEqual(len(evidence), expected_evidence[form])
                self.assertEqual(row["tools"], self.tools)

                if calls:
                    argument_name = "gene_symbol" if form == 9 else "gene_id"
                    self.assertEqual(
                        calls[0]["arguments"],
                        {argument_name: scenario.input_value},
                    )
                if form in {10, 11, 12}:
                    self.assertIsNone(answer["provenance"])
                else:
                    self.assertIsNotNone(answer["provenance"])

        version_row, _ = build_row(
            scenarios[3],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        version_call = self._calls(version_row)[0]
        self.assertEqual(
            version_call["arguments"], {"gene_id": f"{self.named.gene_id}.3"}
        )
        self.assertEqual(
            version_row["messages"][-1]["content"]["result"]["gene_id"],
            self.named.gene_id,
        )

        no_symbol = build_row(
            scenarios[4],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )[0]["messages"][-1]["content"]
        self.assertEqual(
            no_symbol["result"],
            {"gene_id": self.unnamed.gene_id, "gene_symbol": None},
        )
        self.assertEqual(no_symbol["continuation_decision"], "stop")

        retired = build_row(
            scenarios[5],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )[0]["messages"][-1]["content"]
        self.assertEqual(retired["result"], {"gene_id": self.retired.gene_id})

        malformed = build_row(
            scenarios[6],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )[0]["messages"][-1]["content"]
        self.assertEqual(
            malformed["result"],
            {"input": self.named.gene_id[:-1], "expected": "human_gene_id"},
        )

        transcript_row, _ = build_row(
            scenarios[12],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        self.assertEqual(
            transcript_row["metadata"]["found_input_type"], "transcript_id"
        )
        self.assertNotIn("input_species", transcript_row["metadata"])

        mouse_row, _ = build_row(
            scenarios[10],
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        self.assertEqual(mouse_row["metadata"]["input_species"], "mouse")
        self.assertNotIn("found_input_type", mouse_row["metadata"])

    def test_usable_results_follow_the_question_framing(self) -> None:
        for framing, decision in (("direct", "stop"), ("step", "continue")):
            for form, fact, value in (
                (3, self.named, f"{self.named.gene_id}.3"),
                (4, self.unnamed, self.unnamed.gene_id),
                (7, self.named, self.named.gene_id),
                (9, self.named, self.named.gene_symbol or ""),
            ):
                with self.subTest(form=form, framing=framing):
                    row, _ = build_row(
                        self._scenario(
                            form, fact, value, framing=framing
                        ),
                        runtime=self.runtime,
                        contract=self.contract,
                        tools=self.tools,
                    )
                    answer = row["messages"][-1]["content"]
                    self.assertEqual(answer["continuation_decision"], decision)
                    self.assertIsNone(answer["blocked_reason"])

    def test_reviewed_thinking_wording_in_golden_framings(self) -> None:
        cases = {
            2: (
                self._scenario(
                    2, self.named, self.named.gene_id, framing="step"
                ),
                [
                    "The next operation needs the symbol for the seed ID. "
                    "I will look it up in the registry.",
                    "The registry gives the symbol ONE for the seed ID. The "
                    "next operation can use it.",
                ],
            ),
            4: (
                self._scenario(
                    4, self.unnamed, self.unnamed.gene_id, framing="step"
                ),
                [
                    "The analysis needs the symbol for the seed ID. I will "
                    "look it up in the registry.",
                    "The registry has this ID but no symbol for it. The ID is "
                    "still valid, so the analysis can continue with it.",
                ],
            ),
            8: (
                self._scenario(
                    8,
                    self.named,
                    self.named.gene_id,
                    framing="step",
                    distractor=self.other_named,
                ),
                [
                    f"The registry entries list {self.other_named.gene_id} "
                    f"but not {self.named.gene_id}. I will look up the seed "
                    "ID in the registry.",
                    "The registry gives the symbol ONE for the seed ID. The "
                    "next step can use it.",
                ],
            ),
        }
        for form, (scenario, expected) in cases.items():
            with self.subTest(form=form):
                row, _ = build_row(
                    scenario,
                    runtime=self.runtime,
                    contract=self.contract,
                    tools=self.tools,
                )
                self.assertEqual(
                    [
                        message["thinking"]
                        for message in row["messages"]
                        if message["role"] == "assistant"
                    ],
                    expected,
                )

        par_y = self._scenario(
            5,
            self.retired,
            "ENSG00000182378.14_PAR_Y",
        )
        par_row, _ = build_row(
            par_y,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        self.assertEqual(
            [
                message["thinking"]
                for message in par_row["messages"]
                if message["role"] == "assistant"
            ],
            [
                "I need the symbol for this Ensembl ID. I will pass the ID "
                "to the registry tool as given.",
                "The registry has no gene with this ID in release 116. The "
                "_PAR_Y suffix marks the chromosome Y copy of a PAR gene in "
                "an older GENCODE release. I will not map it to another gene "
                "from memory, so I must defer.",
            ],
        )

        direct = self._scenario(1, self.named, self.named.gene_id)
        failure = Scenario(
            split=direct.split,
            form=11,
            fact=direct.fact,
            framing=direct.framing,
            template_index=direct.template_index,
            occurrence=direct.occurrence,
            input_value=direct.input_value,
            paired_success_id=direct.scenario_id,
        )
        direct_row, _ = build_row(
            direct,
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
        self.assertEqual(
            direct_row["messages"][2]["thinking"],
            failure_row["messages"][2]["thinking"],
        )


class WorldModelS12AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        WorldModelS12RowSemanticsTests.setUp(self)

    def _scenario(
        self,
        form: int,
        fact: GeneFact,
        input_value: str,
        *,
        framing: str = "direct",
        distractor: GeneFact | None = None,
    ) -> Scenario:
        return WorldModelS12RowSemanticsTests._scenario(
            self,
            form,
            fact,
            input_value,
            framing=framing,
            distractor=distractor,
        )

    @staticmethod
    def _classifier_row(
        split: str, form: int, input_value: str
    ) -> dict[str, object]:
        return {
            "row_id": f"classifier_{split}_{form}",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Find the official symbol corresponding to "
                        f"{input_value}"
                    ),
                }
            ],
        }

    @staticmethod
    def _classifier_scenario(
        split: str, form: int, input_value: str
    ) -> Scenario:
        return Scenario(
            split=split,
            form=form,
            fact=_fact(form, split=split, symbol=f"FORM{form}"),
            framing="direct",
            template_index=0,
            occurrence=0,
            input_value=input_value,
        )

    def test_masked_wording_classifier_masks_form_correlated_inputs(self) -> None:
        inputs = {1: "AlphaAlpha", 2: "BetaBeta"}
        rows_by_split = {
            split: [
                self._classifier_row(split, form, inputs[form])
                for form in (1, 2)
            ]
            for split in ("train", "validation", "test")
        }
        scenarios_by_split = {
            split: [
                self._classifier_scenario(split, form, inputs[form])
                for form in (1, 2)
            ]
            for split in ("train", "validation", "test")
        }

        result = masked_wording_classifier_audit(
            rows_by_split, scenarios_by_split
        )

        self.assertTrue(result["pass"])
        self.assertEqual(result["mask_token"], "<INPUT>")
        self.assertEqual(
            result["algorithm"],
            "class_balanced_multinomial_naive_bayes_v1",
        )
        self.assertEqual(result["training_vocabulary_size"], 7)
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                metrics = result["splits"][split]
                self.assertEqual(metrics["row_count"], 2)
                self.assertEqual(metrics["correct"], 1)
                self.assertEqual(metrics["accuracy"], 0.5)
                self.assertEqual(metrics["largest_form"], 1)
                self.assertEqual(metrics["largest_form_baseline"], 0.5)
                self.assertEqual(metrics["known_token_occurrences"], 14)
                self.assertEqual(metrics["unknown_token_occurrences"], 0)
                if split == "train":
                    self.assertFalse(metrics["gate_applied"])
                    self.assertIsNone(metrics["threshold"])
                    self.assertIsNone(metrics["pass"])
                else:
                    self.assertTrue(metrics["gate_applied"])
                    self.assertEqual(metrics["threshold"], 0.52)
                    self.assertTrue(metrics["pass"])

    def test_thinking_filter_rejects_each_forbidden_claim(self) -> None:
        scenario = self._scenario(1, self.named, self.named.gene_id)
        row, _ = build_row(
            scenario,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        self.assertEqual(filter_row_thinking(row, scenario), 2)

        mutations = {
            "premature_symbol": (
                0,
                "The registry result is the symbol UNSEENSYMBOL.",
                "reveals 'UNSEENSYMBOL' before",
            ),
            "wrong_status": (
                1,
                "The registry has no gene with this ID, so I must defer.",
                "thinking status",
            ),
            "wrong_decision": (
                1,
                "The registry gives the symbol for the seed ID. The next "
                "step can use it.",
                "thinking decision",
            ),
            "answer_schema": (
                1,
                "I will put this evidence in the provenance field.",
                "discusses answer schema/provenance",
            ),
        }
        for name, (assistant_index, thinking, message) in mutations.items():
            with self.subTest(rule=name):
                mutated = deepcopy(row)
                if name == "premature_symbol":
                    tool_message = next(
                        item
                        for item in mutated["messages"]
                        if item["role"] == "tool"
                    )
                    tool_message["content"]["gene_symbol"] = (
                        "UNSEENSYMBOL"
                    )
                    final_answer = mutated["messages"][-1]["content"]
                    final_answer["result"]["gene_symbol"] = (
                        "UNSEENSYMBOL"
                    )
                assistants = [
                    item
                    for item in mutated["messages"]
                    if item["role"] == "assistant"
                ]
                assistants[assistant_index]["thinking"] = thinking
                with self.assertRaisesRegex(CorpusBuildError, message):
                    filter_row_thinking(mutated, scenario)

    def test_duplicate_audit_ignores_evidence_and_final_answer(self) -> None:
        scenario = self._scenario(1, self.named, self.named.gene_id)
        row, _ = build_row(
            scenario,
            runtime=self.runtime,
            contract=self.contract,
            tools=self.tools,
        )
        duplicate_scenario = Scenario(
            split=scenario.split,
            form=scenario.form,
            fact=scenario.fact,
            framing=scenario.framing,
            template_index=scenario.template_index,
            occurrence=1,
            input_value=scenario.input_value,
        )
        duplicate = deepcopy(row)
        duplicate["row_id"] = "duplicate_with_new_evidence"
        tool_message = next(
            item
            for item in duplicate["messages"]
            if item["role"] == "tool"
        )
        tool_message["content"]["provenance"]["registry"][
            "evidence_id"
        ] = "ev_changed_tool"
        final_answer = duplicate["messages"][-1]["content"]
        final_answer["provenance"]["registry"]["evidence_id"] = (
            "ev_changed_answer"
        )
        final_answer["summary"] = "A deliberately different final answer."

        with self.assertRaisesRegex(
            CorpusBuildError,
            "repeats a prompt and tool outcome after removing evidence IDs",
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

    def test_duplicate_audit_allows_the_intended_failure_pair(self) -> None:
        success = self._scenario(1, self.named, self.named.gene_id)
        failure = Scenario(
            split=success.split,
            form=11,
            fact=success.fact,
            framing=success.framing,
            template_index=success.template_index,
            occurrence=success.occurrence,
            input_value=success.input_value,
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
            result["splits"]["train"],
            {
                "row_count": 2,
                "normalized_prompt_count": 1,
                "normalized_prompt_tool_outcome_count": 2,
                "duplicate_prompt_tool_outcome_count": 0,
                "intended_success_failure_pair_count": 1,
                "pass": True,
            },
        )


class WorldModelS12GeneratedCorpusTests(unittest.TestCase):
    def test_generated_manifest_reconciles_when_present(self) -> None:
        root = Path("data/world_model_v3/corpora/s1/s1_2")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"generated corpus is absent: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["section"], "S1.2")
        self.assertEqual(
            manifest["row_counts"],
            {"train": 11000, "validation": 1375, "test": 1375},
        )
        contract = load_contract()
        curriculum = contract["curriculum"]
        settings = curriculum["families"]["S1.2"]
        expected_counts = {
            split: form_counts(total, settings, curriculum)
            for split, total in manifest["row_counts"].items()
        }
        expected_status = {
            split: {
                "error": counts[11],
                "invalid_input": counts[6] + counts[10] + counts[12],
                "not_found": counts[5],
                "resolved": sum(
                    counts[form] for form in (1, 2, 3, 4, 7, 8, 9)
                ),
            }
            for split, counts in expected_counts.items()
        }
        expected_call_counts = {
            split: {
                "0": counts[7] + counts[10] + counts[12],
                "1": sum(
                    counts[form]
                    for form in (1, 2, 3, 4, 5, 6, 8, 9, 11)
                ),
            }
            for split, counts in expected_counts.items()
        }
        expected_framing = {
            split: largest_remainder_counts(
                total, curriculum["framing"]
            )
            for split, total in manifest["row_counts"].items()
        }
        self.assertEqual(
            manifest["form_counts"],
            {
                split: {
                    f"S1.2.{form}": count
                    for form, count in counts.items()
                }
                for split, counts in expected_counts.items()
            },
        )
        self.assertEqual(manifest["status_counts"], expected_status)
        self.assertEqual(manifest["tool_call_counts"], expected_call_counts)
        self.assertEqual(manifest["framing_counts"], expected_framing)
        self.assertEqual(
            manifest["failure_injection"]["pair_counts"],
            {split: counts[11] for split, counts in expected_counts.items()},
        )
        self.assertEqual(
            manifest["added_form_rows"],
            {split: counts[12] for split, counts in expected_counts.items()},
        )
        self.assertEqual(
            manifest["sources"]["human_gtf"]["gene_version_count"],
            78941,
        )
        self.assertEqual(
            manifest["sources"]["human_gtf"]["transcript_count"],
            646577,
        )
        self.assertTrue(manifest["masked_wording_classifier"]["pass"])
        self.assertTrue(manifest["thinking_filter"]["pass"])
        self.assertTrue(manifest["duplicate_prompt_tool_outcome"]["pass"])
        self.assertIsNone(
            manifest["masked_wording_classifier"]["splits"]["train"]["pass"]
        )
        for split in ("validation", "test"):
            self.assertTrue(
                manifest["masked_wording_classifier"]["splits"][split][
                    "pass"
                ]
            )
            self.assertEqual(
                manifest["duplicate_prompt_tool_outcome"]["splits"][split][
                    "duplicate_prompt_tool_outcome_count"
                ],
                0,
            )

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
        expected_codes = {
            5: "not_found",
            6: "invalid_input",
            10: "out_of_scope_species",
            11: "tool_error",
            12: "wrong_input_type",
        }
        all_row_ids: set[str] = set()
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
                {f"S1.2.{form}" for form in range(1, 13)},
            )
            successes: dict[str, dict[str, object]] = {}
            failures: list[dict[str, object]] = []
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
                self.assertEqual(
                    [call["name"] for call in calls], expected_calls[form]
                )
                self.assertEqual(
                    row["metadata"]["blocked_reason_code"],
                    expected_codes.get(form),
                )
                if calls:
                    argument_name = "gene_symbol" if form == 9 else "gene_id"
                    argument = calls[0]["arguments"][argument_name]
                    self.assertIn(argument, row["messages"][-4]["content"])

                if form in {10, 11, 12}:
                    self.assertIsNone(answer["provenance"])
                else:
                    self.assertIsNotNone(answer["provenance"])
                if form == 1:
                    self.assertEqual(answer["continuation_decision"], "stop")
                    key = json.dumps(row["messages"][:3], sort_keys=True)
                    self.assertNotIn(key, successes)
                    successes[key] = row
                elif form == 2:
                    self.assertEqual(
                        answer["continuation_decision"], "continue"
                    )
                elif form == 3:
                    input_id = calls[0]["arguments"]["gene_id"]
                    canonical_id, version = input_id.rsplit(".", 1)
                    self.assertTrue(version.isdigit())
                    self.assertEqual(answer["result"]["gene_id"], canonical_id.upper())
                elif form == 4:
                    self.assertIsNone(answer["result"]["gene_symbol"])
                    self.assertEqual(answer["status"], "resolved")
                elif form == 5:
                    self.assertEqual(set(answer["result"]), {"gene_id"})
                elif form == 6:
                    value = answer["result"]["input"]
                    self.assertEqual(value, calls[0]["arguments"]["gene_id"])
                    self.assertEqual(len(value), 14)
                    self.assertTrue(value.upper().startswith("ENSG"))
                elif form == 7:
                    self.assertEqual(
                        [message["role"] for message in row["messages"]].count(
                            "harness"
                        ),
                        1,
                    )
                elif form == 8:
                    harness = next(
                        message["content"]
                        for message in row["messages"]
                        if message["role"] == "harness"
                    )
                    block = json.loads(harness.split("\n", 1)[1])
                    self.assertNotEqual(
                        block["entries"][0]["gene_id"].casefold(),
                        calls[0]["arguments"]["gene_id"].casefold(),
                    )
                elif form == 10:
                    self.assertEqual(row["metadata"]["input_species"], "mouse")
                    self.assertTrue(
                        answer["result"]["input"].startswith("ENSMUSG")
                    )
                elif form == 11:
                    self.assertEqual(answer["status"], "error")
                    failures.append(row)
                elif form == 12:
                    self.assertEqual(
                        row["metadata"]["found_input_type"], "transcript_id"
                    )
                    self.assertTrue(
                        answer["result"]["input"].upper().startswith("ENST")
                    )

            self.assertEqual(len(failures), expected_counts[split][11])
            for failure in failures:
                key = json.dumps(failure["messages"][:3], sort_keys=True)
                self.assertIn(key, successes)
                success = successes[key]
                self.assertEqual(
                    failure["messages"][2], success["messages"][2]
                )
                self.assertEqual(
                    failure["messages"][-2]["content"]["status"], "error"
                )
                self.assertEqual(
                    success["messages"][-2]["content"]["status"], "resolved"
                )


if __name__ == "__main__":
    unittest.main()
