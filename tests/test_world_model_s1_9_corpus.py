"""Focused tests for the canonical S1.9 module-ID corpus builder."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import runtime.world_model.s1_9_corpus as s1_9_module

from runtime.world_model.contract import (
    REPOSITORY_ROOT,
    load_contract,
    tool_definitions,
)
from runtime.world_model.s1_9_corpus import (
    FORM_BY_FAMILY,
    MODULE_FAMILIES,
    MODULE_GRAMMARS,
    MODULE_PROMPTS,
    PARSE_REQUESTS,
    CorpusBuildError,
    Scenario,
    audit_rows,
    build_corpus,
    build_fact_pools,
    build_row,
    duplicate_prompt_outcome_audit,
    filter_row_thinking,
    masked_wording_classifier_audit,
    parse_module_id,
    select_scenarios,
    split_row_counts,
    trace_counts,
)
from runtime.world_model.s1_overlay import S1_V3_OVERLAY_PATH, apply_s1_overlay
from runtime.world_model.s1_prompt_banks import wrapper_prompts
from runtime.world_model.source_index import check_source_index
from runtime.world_model.sources import file_sha256
from runtime.world_model_training import (
    flatten_v3_trace_for_arrow,
    tokenize_v3_trace_for_sft,
)


FORM_BY_ALLOCATION = {"rwr_loe_parse": 1, "mentor_ev_parse": 2}
EXPECTED_TRACE_COUNTS = {
    "train": {1: 1000, 2: 1000},
    "validation": {1: 125, 2: 125},
    "test": {1: 125, 2: 125},
}
EXPECTED_POOL_COUNTS = {
    "train": {"rwr_loe": 37_613, "mentor_ev": 12_543},
    "validation": {"rwr_loe": 4_702, "mentor_ev": 1_568},
    "test": {"rwr_loe": 4_702, "mentor_ev": 1_568},
}
EXPECTED_GRAMMARS = {
    "rwr_loe": {
        "pattern": (
            "^rwr_loe:full_brain_multiplex_v1:"
            "seed_(ENSG[0-9]{11}):geometric_elbow_v1$"
        ),
        "field_count": 4,
        "multiplex_id": "full_brain_multiplex_v1",
        "threshold_recipe": "geometric_elbow_v1",
    },
    "mentor_ev": {
        "pattern": "^mentor_ev:full_brain_all_genes_v1:clade_[0-9]{6}$",
        "field_count": 3,
        "dendrogram_id": "full_brain_all_genes_v1",
        "clade_digits": 6,
    },
}
EXPECTED_RESULT_SHAPES = {
    "rwr_loe": {
        "keys": [
            "module_id",
            "module_family",
            "multiplex_id",
            "seed_gene_id",
            "threshold_recipe",
            "existence_asserted",
        ],
        "module_family": "rwr_loe",
        "multiplex_id": "full_brain_multiplex_v1",
        "threshold_recipe": "geometric_elbow_v1",
        "existence_asserted": False,
    },
    "mentor_ev": {
        "keys": [
            "module_id",
            "module_family",
            "dendrogram_id",
            "clade_id",
            "existence_asserted",
        ],
        "module_family": "mentor_ev",
        "dendrogram_id": "full_brain_all_genes_v1",
        "existence_asserted": False,
    },
}
EXPECTED_MANIFEST_KEYS = {
    "schema_version",
    "corpus_schema_version",
    "section",
    "dataset_id",
    "seed",
    "contract_s1_9_sha256",
    "prompt_templates_sha256",
    "request_templates_sha256",
    "module_grammars_sha256",
    "result_shapes_sha256",
    "scenario_assignment_sha256",
    "row_counts",
    "form_counts",
    "semantic_bucket_counts",
    "status_counts",
    "tool_call_counts",
    "framing_counts",
    "prompt_template_counts",
    "request_template_counts",
    "module_family_counts",
    "continuation_counts",
    "selected_source_counts",
    "selected_source_sha256",
    "sources",
    "source_pool_coverage",
    "split_unit_sha256",
    "per_form_gates",
    "prompt_structure_audit",
    "masked_wording_classifier",
    "thinking_filter",
    "duplicate_prompt_tool_outcome",
    "evidence",
    "files",
}
EXPECTED_SOURCE_KEYS = {
    "source_index",
    "contract",
    "builder",
    "s1_6_builder",
    "golden_traces",
    "question_classes",
    "coordinate_spec",
    "multiplex_flist",
    "mentor_ev_dendrogram",
    "records",
    "module_policy",
}


def _source_index(contract: dict[str, object]) -> Path:
    digest = contract["source_index"]["content_sha256"]
    configured = Path(contract["source_index"]["output_directory"])
    root = configured if configured.is_absolute() else REPOSITORY_ROOT / configured
    return root / f"world_model_source_index_{digest}.json.gz"


def _answer(row: dict[str, object]) -> dict[str, object]:
    return row["messages"][-1]["content"]


def _calls(row: dict[str, object]) -> list[dict[str, object]]:
    return [
        message["tool_call"]
        for message in row["messages"]
        if message["role"] == "assistant" and "tool_call" in message
    ]


def _matches_template(text: str, template: str) -> bool:
    prefix, suffix = template.split("{request}")
    return text.startswith(prefix) and text.endswith(suffix)


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _Tokenizer:
    def apply_chat_template(
        self, messages: list[dict[str, object]], **options: object
    ) -> list[int]:
        assert all(
            "description" in tool["function"] for tool in options["tools"]
        )
        result = [1]
        for message in messages:
            if message["role"] == "system":
                result.append(10)
            elif message["role"] == "user":
                result.append(20)
            elif message["role"] == "tool":
                result.append(50)
            elif "tool_calls" in message:
                result.append(40)
            else:
                if message.get("thinking"):
                    result.append(60)
                result.append(61)
        return result


class WorldModelS19ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.curriculum = self.contract["curriculum"]
        self.settings = self.curriculum["families"]["S1.9"]

    def test_contract_records_every_frozen_build_decision(self) -> None:
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
                "evaluation_allocations",
                "semantic_buckets",
                "source_trace_by_allocation",
                "grammars",
                "result_shapes",
                "source_pools",
                "split_policy",
                "nonpositive_policy",
                "status_counts",
                "framing_counts",
            ],
        )
        self.assertEqual(
            {
                key: self.settings[key]
                for key in (
                    "dataset_id",
                    "question_family",
                    "book_mode",
                    "output_directory",
                    "assignment_seed",
                    "train_rows",
                )
            },
            {
                "dataset_id": "world_model_v3_s1_9_module_id_parsing",
                "question_family": "module_id_parsing",
                "book_mode": "closed_book",
                "output_directory": "data/world_model_v3/corpora/s1/s1_9",
                "assignment_seed": 1729,
                "train_rows": 2000,
            },
        )
        self.assertEqual(
            self.settings["allocations"],
            {
                name: EXPECTED_TRACE_COUNTS["train"][form]
                for name, form in FORM_BY_ALLOCATION.items()
            },
        )
        self.assertEqual(
            self.settings["evaluation_allocations"],
            {
                name: EXPECTED_TRACE_COUNTS["validation"][form]
                for name, form in FORM_BY_ALLOCATION.items()
            },
        )
        self.assertEqual(
            self.settings["semantic_buckets"],
            {"rwr_loe": ["rwr_loe_parse"], "mentor_ev": ["mentor_ev_parse"]},
        )
        self.assertEqual(
            self.settings["source_trace_by_allocation"],
            {"rwr_loe_parse": "S1.9.1", "mentor_ev_parse": "S1.9.2"},
        )

    def test_split_and_form_allocations_are_exact(self) -> None:
        row_counts = split_row_counts(
            self.settings["train_rows"], self.curriculum["split_fractions"]
        )
        self.assertEqual(
            row_counts, {"train": 2000, "validation": 250, "test": 250}
        )
        counts = {
            split: trace_counts(total, self.settings)
            for split, total in row_counts.items()
        }
        self.assertEqual(counts, EXPECTED_TRACE_COUNTS)
        for split, values in counts.items():
            self.assertEqual(sum(values.values()), row_counts[split])

    def test_grammars_and_result_shapes_match_the_golden_contract(self) -> None:
        self.assertEqual(self.settings["grammars"], EXPECTED_GRAMMARS)
        self.assertEqual(MODULE_GRAMMARS, EXPECTED_GRAMMARS)
        self.assertEqual(self.settings["result_shapes"], EXPECTED_RESULT_SHAPES)
        self.assertEqual(FORM_BY_FAMILY, {"rwr_loe": 1, "mentor_ev": 2})
        self.assertEqual(MODULE_FAMILIES, ("rwr_loe", "mentor_ev"))

    def test_source_and_split_policies_are_exact(self) -> None:
        self.assertEqual(
            self.settings["source_pools"],
            {
                "rwr_loe": {
                    "source_index_field": "multiplex_gene_ids",
                    "record_count": 47_017,
                    "split_counts": {
                        "train": 37_613,
                        "validation": 4_702,
                        "test": 4_702,
                    },
                },
                "mentor_ev": {
                    "source_index_field": "mentor_clades",
                    "record_count": 15_679,
                    "eligible_leaf_count": {"min": 5, "max": 30},
                    "split_counts": {
                        "train": 12_543,
                        "validation": 1_568,
                        "test": 1_568,
                    },
                },
            },
        )
        self.assertEqual(
            self.settings["split_policy"],
            {
                "reuse": "S1.6.module_facts",
                "unit": "exact_module_id",
                "order": "seeded_sha256",
                "modulus": 10,
                "residues": {
                    "train": [2, 3, 4, 5, 6, 7, 8, 9],
                    "validation": [0],
                    "test": [1],
                },
                "namespace_labels": {
                    "rwr_loe": "module:rwr_loe",
                    "mentor_ev": "module:mentor_ev",
                },
                "ids_disjoint": True,
            },
        )

    def test_nonpositive_and_existence_cases_are_excluded(self) -> None:
        self.assertEqual(
            self.settings["nonpositive_policy"],
            {
                "included": False,
                "invalid": "excluded",
                "ambiguous": "excluded",
                "unrecognized": "excluded",
                "syntactically_valid_unobserved": "excluded",
            },
        )
        self.assertEqual(
            self.settings["status_counts"],
            {"train": {"resolved": 2000}, "evaluation": {"resolved": 250}},
        )
        for shape in self.settings["result_shapes"].values():
            self.assertIs(shape["existence_asserted"], False)

    def test_builder_rejects_direct_contract_mapping_drift(self) -> None:
        changed_settings = []

        changed = deepcopy(self.settings)
        changed["allocations"]["rwr_loe_parse"] = 999
        changed_settings.append(("allocation", changed))

        changed = deepcopy(self.settings)
        changed["source_trace_by_allocation"]["rwr_loe_parse"] = "S1.9.2"
        changed_settings.append(("source trace", changed))

        changed = deepcopy(self.settings)
        changed["grammars"]["mentor_ev"]["clade_digits"] = 5
        changed_settings.append(("grammar", changed))

        changed = deepcopy(self.settings)
        changed["result_shapes"]["rwr_loe"]["existence_asserted"] = True
        changed_settings.append(("result", changed))

        changed = deepcopy(self.settings)
        changed["source_pools"]["mentor_ev"]["record_count"] = 15_678
        changed_settings.append(("source pool", changed))

        changed = deepcopy(self.settings)
        changed["split_policy"]["residues"]["validation"] = [1]
        changed_settings.append(("split", changed))

        changed = deepcopy(self.settings)
        changed["nonpositive_policy"]["invalid"] = "included"
        changed_settings.append(("nonpositive", changed))

        for name, settings in changed_settings:
            with self.subTest(name=name):
                with self.assertRaises(CorpusBuildError):
                    trace_counts(2000, settings)


class WorldModelS19ParserTests(unittest.TestCase):
    def test_parser_returns_exact_rwr_loe_fields(self) -> None:
        module_id = (
            "rwr_loe:full_brain_multiplex_v1:"
            "seed_ENSG00000141510:geometric_elbow_v1"
        )
        result = parse_module_id(module_id)
        self.assertEqual(
            result,
            {
                "module_id": module_id,
                "module_family": "rwr_loe",
                "multiplex_id": "full_brain_multiplex_v1",
                "seed_gene_id": "ENSG00000141510",
                "threshold_recipe": "geometric_elbow_v1",
                "existence_asserted": False,
            },
        )
        self.assertEqual(list(result), EXPECTED_RESULT_SHAPES["rwr_loe"]["keys"])

    def test_parser_returns_exact_mentor_ev_fields(self) -> None:
        module_id = "mentor_ev:full_brain_all_genes_v1:clade_000417"
        result = parse_module_id(module_id)
        self.assertEqual(
            result,
            {
                "module_id": module_id,
                "module_family": "mentor_ev",
                "dendrogram_id": "full_brain_all_genes_v1",
                "clade_id": "clade_000417",
                "existence_asserted": False,
            },
        )
        self.assertEqual(list(result), EXPECTED_RESULT_SHAPES["mentor_ev"]["keys"])

    def test_parser_accepts_valid_unobserved_ids_without_existence_inference(self) -> None:
        fabricated = (
            "rwr_loe:full_brain_multiplex_v1:"
            "seed_ENSG99999999999:geometric_elbow_v1",
            "mentor_ev:full_brain_all_genes_v1:clade_999999",
        )
        for module_id in fabricated:
            with self.subTest(module_id=module_id):
                result = parse_module_id(module_id)
                self.assertIs(result["existence_asserted"], False)
                self.assertEqual(result["module_id"], module_id)

    def test_parser_rejects_every_malformed_or_wrong_context_case(self) -> None:
        malformed = (
            "",
            "unknown:full_brain_all_genes_v1:clade_000417",
            "rwr_loe:full_brain_multiplex_v1",
            "rwr_loe:retired_brain_multiplex_v0:seed_ENSG00000141510:geometric_elbow_v1",
            "rwr_loe:full_brain_multiplex_v1:ENSG00000141510:geometric_elbow_v1",
            "rwr_loe:full_brain_multiplex_v1:seed_ENSG0000014151:geometric_elbow_v1",
            "rwr_loe:full_brain_multiplex_v1:seed_ENSG000001415100:geometric_elbow_v1",
            "rwr_loe:full_brain_multiplex_v1:seed_ENST00000141510:geometric_elbow_v1",
            "rwr_loe:full_brain_multiplex_v1:seed_ENSG00000141510:fixed_cut_v1",
            "rwr_loe:full_brain_multiplex_v1:seed_ENSG00000141510:geometric_elbow_v1:extra",
            "mentor_ev:full_brain_all_genes_v1",
            "mentor_ev:other_tree:clade_000417",
            "mentor_ev:full_brain_all_genes_v1:clade_00417",
            "mentor_ev:full_brain_all_genes_v1:clade_0000417",
            "mentor_ev:full_brain_all_genes_v1:node_000417",
            "mentor_ev:full_brain_all_genes_v1:clade_-00417",
            "mentor_ev:full_brain_all_genes_v1:clade_000417:extra",
        )
        for module_id in malformed:
            with self.subTest(module_id=module_id):
                with self.assertRaises(CorpusBuildError):
                    parse_module_id(module_id)
        with self.assertRaises(CorpusBuildError):
            parse_module_id(None)  # type: ignore[arg-type]


class WorldModelS19SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.9"]
        cls.source_index = _source_index(cls.contract)
        if not cls.source_index.is_file():
            raise unittest.SkipTest(
                f"canonical source index is absent: {cls.source_index}"
            )
        cls.payload = check_source_index(cls.source_index, cls.contract)
        cls.pools = build_fact_pools(
            cls.payload,
            settings=cls.settings,
            seed=cls.settings["assignment_seed"],
            contract=cls.contract,
        )
        cls.scenarios = select_scenarios(
            cls.pools,
            EXPECTED_TRACE_COUNTS,
            seed=cls.settings["assignment_seed"],
            framing_shares=cls.contract["curriculum"]["framing"],
            settings=cls.settings,
        )

    def test_source_pools_have_exact_counts_and_disjoint_ids(self) -> None:
        split_ids = {}
        for split, families in self.pools.items():
            self.assertEqual(set(families), set(MODULE_FAMILIES))
            split_ids[split] = set()
            for family, facts in families.items():
                with self.subTest(split=split, family=family):
                    self.assertEqual(len(facts), EXPECTED_POOL_COUNTS[split][family])
                    self.assertEqual(len({fact.module_id for fact in facts}), len(facts))
                    self.assertTrue(all(fact.namespace == family for fact in facts))
                    for fact in facts:
                        result = parse_module_id(fact.module_id)
                        self.assertEqual(result["module_family"], family)
                        self.assertIs(result["existence_asserted"], False)
                        split_ids[split].add(fact.module_id)
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            self.assertTrue(split_ids[first].isdisjoint(split_ids[second]))

    def test_source_pools_use_every_pinned_gene_and_eligible_clade(self) -> None:
        rwr_ids = {
            fact.module_id
            for split in self.pools.values()
            for fact in split["rwr_loe"]
        }
        mentor_ids = {
            fact.module_id
            for split in self.pools.values()
            for fact in split["mentor_ev"]
        }
        self.assertEqual(len(rwr_ids), len(self.payload["multiplex_gene_ids"]))
        self.assertEqual(len(mentor_ids), len(self.payload["mentor_clades"]))
        self.assertEqual(
            {parse_module_id(value)["seed_gene_id"] for value in rwr_ids},
            set(self.payload["multiplex_gene_ids"]),
        )
        self.assertEqual(
            {
                int(parse_module_id(value)["clade_id"].removeprefix("clade_"))
                for value in mentor_ids
            },
            {int(row[0]) for row in self.payload["mentor_clades"]},
        )

    def test_scenario_counts_families_and_frames_are_exact(self) -> None:
        expected_frames = {
            "train": {"direct": 1000, "step": 1000},
            "validation": {"direct": 125, "step": 125},
            "test": {"direct": 125, "step": 125},
        }
        expected_by_form = {
            "train": {
                1: {"direct": 500, "step": 500},
                2: {"direct": 500, "step": 500},
            },
            "validation": {
                1: {"direct": 63, "step": 62},
                2: {"direct": 62, "step": 63},
            },
            "test": {
                1: {"direct": 63, "step": 62},
                2: {"direct": 62, "step": 63},
            },
        }
        row_ids = set()
        for split, scenarios in self.scenarios.items():
            self.assertEqual(
                Counter(item.form for item in scenarios),
                Counter(EXPECTED_TRACE_COUNTS[split]),
            )
            self.assertEqual(
                Counter(item.framing for item in scenarios),
                Counter(expected_frames[split]),
            )
            for form in (1, 2):
                self.assertEqual(
                    Counter(item.framing for item in scenarios if item.form == form),
                    Counter(expected_by_form[split][form]),
            )
            for item in scenarios:
                self.assertEqual(
                    item.module_family,
                    "rwr_loe" if item.form == 1 else "mentor_ev",
                )
                self.assertNotIn(item.scenario_id, row_ids)
                row_ids.add(item.scenario_id)

    def test_selected_ids_are_unique_and_source_backed_in_their_split(self) -> None:
        all_selected = set()
        for split, scenarios in self.scenarios.items():
            for family in MODULE_FAMILIES:
                source = {
                    fact.module_id: fact for fact in self.pools[split][family]
                }
                selected = [
                    item for item in scenarios if item.module_family == family
                ]
                self.assertEqual(
                    len({item.module_id for item in selected}), len(selected)
                )
                for item in selected:
                    self.assertIn(item.module_id, source)
                    self.assertEqual(
                        item.split_unit, source[item.module_id].split_unit
                    )
                    self.assertNotIn(item.module_id, all_selected)
                    all_selected.add(item.module_id)

    def test_prompt_and_request_paraphrases_have_full_heldout_coverage(self) -> None:
        wrappers = {}
        requests = {}
        for split, framings in MODULE_PROMPTS.items():
            self.assertEqual(set(framings), {"direct", "step"})
            wrappers[split] = {
                template for values in framings.values() for template in values
            }
            requests[split] = set(PARSE_REQUESTS[split])
            self.assertTrue(
                all(template.count("{request}") == 1 for template in wrappers[split])
            )
            scenarios = self.scenarios[split]
            for form in (1, 2):
                for framing, templates in framings.items():
                    self.assertEqual(
                        {
                            item.template_index
                            for item in scenarios
                            if item.form == form and item.framing == framing
                        },
                        set(range(len(templates))),
                    )
                self.assertEqual(
                    {item.request_index for item in scenarios if item.form == form},
                    set(range(len(PARSE_REQUESTS[split]))),
                )
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            self.assertTrue(wrappers[first].isdisjoint(wrappers[second]))
            self.assertTrue(requests[first].isdisjoint(requests[second]))


class WorldModelS19RowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.9"]
        cls.tools = tool_definitions(cls.contract)
        cls.source_index = _source_index(cls.contract)
        if not cls.source_index.is_file():
            raise unittest.SkipTest(
                f"canonical source index is absent: {cls.source_index}"
            )
        payload = check_source_index(cls.source_index, cls.contract)
        cls.pools = build_fact_pools(
            payload,
            settings=cls.settings,
            seed=cls.settings["assignment_seed"],
            contract=cls.contract,
        )
        cls.scenarios = select_scenarios(
            cls.pools,
            EXPECTED_TRACE_COUNTS,
            seed=cls.settings["assignment_seed"],
            framing_shares=cls.contract["curriculum"]["framing"],
            settings=cls.settings,
        )

    def _first(self, form: int, **attributes: object) -> Scenario:
        return next(
            item
            for scenarios in self.scenarios.values()
            for item in scenarios
            if item.form == form
            and all(getattr(item, key) == value for key, value in attributes.items())
        )

    def _build(self, scenario: Scenario) -> dict[str, object]:
        row, evidence = build_row(
            scenario, contract=self.contract, tools=self.tools
        )
        self.assertEqual(evidence, [])
        return row

    def test_both_forms_have_closed_book_metadata_and_zero_calls(self) -> None:
        for form in (1, 2):
            scenario = self._first(form)
            row = self._build(scenario)
            answer = _answer(row)
            metadata = row["metadata"]
            with self.subTest(form=form):
                self.assertEqual(
                    list(row), ["row_id", "messages", "tools", "metadata"]
                )
                self.assertEqual(row["tools"], self.tools)
                self.assertEqual(
                    [message["role"] for message in row["messages"]],
                    ["system", "user", "assistant"],
                )
                self.assertEqual(_calls(row), [])
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
                self.assertEqual(answer["status"], "resolved")
                self.assertIsNone(answer["provenance"])
                self.assertIsNone(answer["blocked_reason"])
                self.assertEqual(
                    set(metadata),
                    set(self.contract["rows"]["metadata"]["required"]),
                )
                self.assertEqual(metadata["book_mode"], "closed_book")
                self.assertEqual(metadata["step"], "S1")
                self.assertEqual(metadata["question_family"], "module_id_parsing")
                self.assertEqual(metadata["form"], form)
                self.assertEqual(metadata["source_trace"], f"S1.9.{form}")
                self.assertEqual(metadata["split_unit"], scenario.split_unit)

    def test_every_row_matches_the_exact_parse_without_existence(self) -> None:
        for split, scenarios in self.scenarios.items():
            for scenario in scenarios:
                row = self._build(scenario)
                answer = _answer(row)
                result = answer["result"]
                self.assertEqual(result, parse_module_id(scenario.module_id))
                self.assertEqual(
                    list(result),
                    EXPECTED_RESULT_SHAPES[scenario.module_family]["keys"],
                )
                self.assertIs(result["existence_asserted"], False)
                self.assertEqual(
                    answer["continuation_decision"],
                    "continue" if scenario.framing == "step" else "stop",
                )

    def test_rwr_rows_anchor_the_exact_seed_and_multiplex(self) -> None:
        for framing in ("direct", "step"):
            scenario = self._first(1, framing=framing)
            answer = _answer(self._build(scenario))
            result = answer["result"]
            self.assertEqual(result["module_family"], "rwr_loe")
            self.assertRegex(result["seed_gene_id"], r"^ENSG[0-9]{11}$")
            self.assertEqual(result["multiplex_id"], "full_brain_multiplex_v1")
            self.assertEqual(result["threshold_recipe"], "geometric_elbow_v1")
            self.assertIn(f'<GENE:{result["seed_gene_id"]}>', answer["summary"])
            self.assertIn(
                "<MULTIPLEX:full_brain_multiplex_v1>", answer["summary"]
            )

    def test_mentor_rows_return_exact_dendrogram_and_clade_fields(self) -> None:
        for framing in ("direct", "step"):
            scenario = self._first(2, framing=framing)
            answer = _answer(self._build(scenario))
            result = answer["result"]
            self.assertEqual(result["module_family"], "mentor_ev")
            self.assertEqual(result["dendrogram_id"], "full_brain_all_genes_v1")
            self.assertRegex(result["clade_id"], r"^clade_[0-9]{6}$")
            self.assertIn(result["clade_id"], answer["summary"])
            self.assertIn(result["dendrogram_id"], answer["summary"])

    def test_prompts_request_parsing_only_and_use_heldout_wording(self) -> None:
        forbidden = re.compile(
            r"\b(?:exist|exists|existing|real|present|found|observed|valid)\b",
            re.IGNORECASE,
        )
        for split, scenarios in self.scenarios.items():
            held_out_templates = [
                template
                for other, framings in MODULE_PROMPTS.items()
                if other != split
                for values in framings.values()
                for template in values
            ]
            held_out_requests = {
                template
                for other, templates in PARSE_REQUESTS.items()
                if other != split
                for template in templates
            }
            for scenario in scenarios:
                row = self._build(scenario)
                user = row["messages"][1]["content"]
                self.assertEqual(user.count(scenario.module_id), 1)
                self.assertIsNone(forbidden.search(user), user)
                self.assertEqual(
                    sum(
                        _matches_template(user, template)
                        for values in MODULE_PROMPTS[split].values()
                        for template in values
                    ),
                    1,
                )
                self.assertFalse(
                    any(
                        _matches_template(user, template)
                        for template in held_out_templates
                    )
                )
                selected_request = PARSE_REQUESTS[split][scenario.request_index]
                self.assertIn(
                    selected_request.format(module_id=scenario.module_id), user
                )
                self.assertNotIn(selected_request, held_out_requests)
                self.assertEqual(filter_row_thinking(row, scenario), 1)

    def test_thinking_and_summary_state_the_parse_only_limit(self) -> None:
        for form in (1, 2):
            scenario = self._first(form)
            row = self._build(scenario)
            thought = row["messages"][-1]["thinking"].casefold()
            summary = _answer(row)["summary"].casefold()
            self.assertIn("parsing", thought)
            self.assertIn("does not assert existence", thought)
            self.assertIn("parsing", summary)
            self.assertIn("does not assert existence", summary)
            self.assertNotIn(" exists", thought)
            self.assertNotIn(" exists", summary)

    def test_thinking_filter_rejects_recall_and_runtime_mutations(self) -> None:
        scenario = self._first(2)
        row = self._build(scenario)
        mutations = (
            "",
            "I will follow the answer schema.",
            row["messages"][-1]["thinking"] + " This clade exists.",
            row["messages"][-1]["thinking"] + " I will call a tool.",
        )
        for thinking in mutations:
            changed = deepcopy(row)
            changed["messages"][-1]["thinking"] = thinking
            with self.subTest(thinking=thinking):
                with self.assertRaises(CorpusBuildError):
                    filter_row_thinking(changed, scenario)

    def test_training_adapter_accepts_a_closed_book_row(self) -> None:
        scenario = self._first(1)
        row = self._build(scenario)
        flattened = flatten_v3_trace_for_arrow(row, expected_split="train")
        self.assertEqual(flattened["question_family"], "module_id_parsing")
        self.assertEqual(flattened["record_id"], row["row_id"])
        self.assertEqual(flattened["fact_id"], scenario.split_unit)
        tokenized = tokenize_v3_trace_for_sft(row, _Tokenizer(), max_length=64)
        self.assertEqual(tokenized["input_ids"], [1, 10, 20, 60, 61])
        self.assertEqual(tokenized["completion_mask"], [0, 0, 0, 1, 1])


class WorldModelS19AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.9"]
        cls.tools = tool_definitions(cls.contract)
        cls.source_index = _source_index(cls.contract)
        if not cls.source_index.is_file():
            raise unittest.SkipTest(
                f"canonical source index is absent: {cls.source_index}"
            )
        payload = check_source_index(cls.source_index, cls.contract)
        cls.pools = build_fact_pools(
            payload,
            settings=cls.settings,
            seed=cls.settings["assignment_seed"],
            contract=cls.contract,
        )
        cls.scenarios = select_scenarios(
            cls.pools,
            EXPECTED_TRACE_COUNTS,
            seed=cls.settings["assignment_seed"],
            framing_shares=cls.contract["curriculum"]["framing"],
            settings=cls.settings,
        )
        cls.rows = {
            split: [
                build_row(item, contract=cls.contract, tools=cls.tools)[0]
                for item in scenarios
            ]
            for split, scenarios in cls.scenarios.items()
        }

    def _audit(
        self,
        rows: dict[str, list[dict[str, object]]] | None = None,
        scenarios: dict[str, list[Scenario]] | None = None,
    ) -> dict[str, object]:
        return audit_rows(
            self.rows if rows is None else rows,
            self.scenarios if scenarios is None else scenarios,
            EXPECTED_TRACE_COUNTS,
            source_pools=self.pools,
            framing_shares=self.contract["curriculum"]["framing"],
            settings=self.settings,
            contract=self.contract,
        )

    def test_duplicate_and_masked_wording_audits_pass(self) -> None:
        duplicate = duplicate_prompt_outcome_audit(self.rows, self.scenarios)
        self.assertTrue(duplicate["pass"])
        wording = masked_wording_classifier_audit(self.rows, self.scenarios)
        self.assertTrue(wording["pass"])
        for split in ("validation", "test"):
            self.assertTrue(wording["splits"][split]["gate_applied"])
            self.assertTrue(wording["splits"][split]["pass"])

    def test_aggregate_audit_reports_exact_core_counts(self) -> None:
        audit = self._audit()
        self.assertEqual(
            audit["status_counts"],
            {
                "train": {"resolved": 2000},
                "validation": {"resolved": 250},
                "test": {"resolved": 250},
            },
        )
        self.assertEqual(
            audit["tool_call_counts"],
            {
                "train": {"0": 2000},
                "validation": {"0": 250},
                "test": {"0": 250},
            },
        )
        self.assertEqual(
            audit["framing_counts"],
            {
                "train": {"direct": 1000, "step": 1000},
                "validation": {"direct": 125, "step": 125},
                "test": {"direct": 125, "step": 125},
            },
        )
        self.assertEqual(
            audit["semantic_bucket_counts"],
            {
                "train": {"mentor_ev": 1000, "rwr_loe": 1000},
                "validation": {"mentor_ev": 125, "rwr_loe": 125},
                "test": {"mentor_ev": 125, "rwr_loe": 125},
            },
        )
        self.assertEqual(
            audit["module_family_counts"], audit["semantic_bucket_counts"]
        )
        self.assertTrue(audit["per_form_gates"]["pass"])
        self.assertEqual(
            audit["per_form_gates"]["existence_asserted_true_count"], 0
        )
        self.assertEqual(
            audit["per_form_gates"]["non_source_id_count"], 0
        )
        self.assertTrue(audit["prompt_structure_audit"]["pass"])
        prose = audit["prompt_structure_audit"]["prose_length_audit"]
        for split in ("train", "validation", "test"):
            self.assertLessEqual(
                prose["splits"][split]["maximum_instruction_words"], 20
            )
            self.assertLessEqual(
                prose["splits"][split]["maximum_description_words"], 25
            )
            self.assertEqual(prose["splits"][split]["violation_count"], 0)
        self.assertTrue(audit["masked_wording_classifier"]["pass"])
        self.assertTrue(audit["thinking_filter"]["pass"])
        self.assertTrue(audit["duplicate_prompt_tool_outcome"]["pass"])

    def test_audit_rejects_field_existence_and_recall_mutations(self) -> None:
        mutations = []

        first_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 1
        )
        changed_family = deepcopy(self.rows)
        _answer(changed_family["train"][first_index])["result"][
            "module_family"
        ] = "mentor_ev"
        mutations.append(("family", changed_family))

        second_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 2
        )
        changed_existence = deepcopy(self.rows)
        _answer(changed_existence["train"][second_index])["result"][
            "existence_asserted"
        ] = True
        mutations.append(("existence", changed_existence))

        changed_status = deepcopy(self.rows)
        _answer(changed_status["train"][second_index])["status"] = "not_found"
        mutations.append(("recall status", changed_status))

        changed_prompt = deepcopy(self.rows)
        module_id = self.scenarios["train"][second_index].module_id
        changed_prompt["train"][second_index]["messages"][1]["content"] = (
            f"Does module {module_id} exist?"
        )
        mutations.append(("recall prompt", changed_prompt))

        for name, rows in mutations:
            with self.subTest(name=name):
                with self.assertRaises(CorpusBuildError):
                    self._audit(rows)

    def test_audit_rejects_a_syntactically_valid_non_source_id(self) -> None:
        index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 2
        )
        changed_scenarios = deepcopy(self.scenarios)
        original = changed_scenarios["train"][index]
        fabricated = "mentor_ev:full_brain_all_genes_v1:clade_999999"
        changed_scenarios["train"][index] = replace(
            original,
            module_id=fabricated,
            split_unit="module_" + _stable_sha256(fabricated)[:24],
            scenario_id="s1_9_fabricated",
        )
        changed_rows = deepcopy(self.rows)
        changed_rows["train"][index] = build_row(
            changed_scenarios["train"][index],
            contract=self.contract,
            tools=self.tools,
        )[0]
        with self.assertRaises(CorpusBuildError):
            self._audit(changed_rows, changed_scenarios)

    def test_duplicate_audit_rejects_an_ordinary_copy(self) -> None:
        scenario = self.scenarios["train"][0]
        duplicate_scenario = replace(scenario, occurrence=100_000)
        row = deepcopy(self.rows["train"][0])
        duplicate = deepcopy(row)
        duplicate["row_id"] = "wm3_s1_9_duplicate"
        duplicate["messages"][-1]["content"]["summary"] = (
            "Different summary text for the same parse."
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


class WorldModelS19GeneratedCorpusTests(unittest.TestCase):
    def _check_source_records(
        self,
        manifest: dict[str, object],
        source_index: Path,
        *,
        exact_shared: bool,
    ) -> None:
        sources = manifest["sources"]
        self.assertEqual(set(sources), EXPECTED_SOURCE_KEYS)
        mutable = {
            "contract",
            # Historical manifests retain their original builder pins;
            # fresh builds below must pin today's sources.
            "builder",
            "s1_6_builder",
            "golden_traces",
            "question_classes",
            "coordinate_spec",
        }
        for name in (
            "contract",
            "builder",
            "s1_6_builder",
            "golden_traces",
            "question_classes",
            "coordinate_spec",
        ):
            record = sources[name]
            self.assertEqual(set(record), {"path", "sha256"})
            recorded_path = Path(record["path"])
            path = (
                recorded_path
                if recorded_path.is_absolute()
                else REPOSITORY_ROOT / recorded_path
            )
            self.assertTrue(path.is_file())
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            if exact_shared or name not in mutable:
                self.assertEqual(record["sha256"], file_sha256(path))

        expected_paths = {
            "contract": Path("runtime/world_model/contract.yaml"),
            "builder": Path("runtime/world_model/s1_9_corpus.py"),
            "s1_6_builder": Path("runtime/world_model/s1_6_corpus.py"),
            "golden_traces": Path("runtime/world_model/golden_traces.jsonl"),
            "question_classes": Path(
                "agents/world_model_v3_curriculum/S1-question-classes.md"
            ),
            "coordinate_spec": Path(
                "agents/world_model_v3_curriculum/S1-coordinate-system.md"
            ),
        }
        for name, expected_path in expected_paths.items():
            self.assertEqual(Path(sources[name]["path"]), expected_path)

        source_record = sources["source_index"]
        self.assertEqual(
            set(source_record), {"path", "sha256", "content_sha256"}
        )
        self.assertEqual(
            Path(source_record["path"]), source_index.relative_to(REPOSITORY_ROOT)
        )
        self.assertEqual(source_record["sha256"], file_sha256(source_index))
        contract = load_contract()
        self.assertEqual(
            source_record["content_sha256"],
            contract["source_index"]["content_sha256"],
        )
        payload = check_source_index(source_index, contract)
        self.assertEqual(sources["records"], payload["sources"])
        self.assertEqual(
            sources["multiplex_flist"], payload["sources"]["multiplex_flist"]
        )
        self.assertEqual(
            sources["mentor_ev_dendrogram"],
            payload["sources"]["mentor_ev_dendrogram"],
        )
        self.assertEqual(
            sources["module_policy"],
            "parse_only_split_safe_source_ids_without_existence_claims",
        )

    def test_build_is_deterministic_and_validates_before_publication(self) -> None:
        contract = load_contract()
        source_index = _source_index(contract)
        if not source_index.is_file():
            self.skipTest(f"canonical source index is absent: {source_index}")
        real_validate = s1_9_module._validate_corpus
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            first_root = temporary / "first" / "corpus"
            second_root = temporary / "second" / "corpus"
            active_root = [first_root]
            validated_names = []

            def validate_before_publish(path: Path) -> None:
                self.assertFalse((active_root[0] / "manifest.json").exists())
                validated_names.append(path.name)
                real_validate(path)

            with patch.object(
                s1_9_module, "_validate_corpus", side_effect=validate_before_publish
            ):
                first = build_corpus(
                    source_index, first_root, contract=contract, validate=True
                )
                active_root[0] = second_root
                second = build_corpus(
                    source_index, second_root, contract=contract, validate=True
                )

            self.assertEqual(
                validated_names,
                [
                    "train.jsonl",
                    "validation.jsonl",
                    "test.jsonl",
                    "train.jsonl",
                    "validation.jsonl",
                    "test.jsonl",
                ],
            )
            self.assertEqual(first, second)
            self.assertEqual(set(first), EXPECTED_MANIFEST_KEYS)
            for file_name in (
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
                "manifest.json",
            ):
                self.assertEqual(
                    (first_root / file_name).read_bytes(),
                    (second_root / file_name).read_bytes(),
                )
            for file_name, values in first["files"].items():
                self.assertEqual(
                    file_sha256(first_root / file_name), values["sha256"]
                )
            self.assertEqual(
                json.loads((first_root / "manifest.json").read_text(encoding="utf-8")),
                first,
            )
            self._check_source_records(
                first, source_index, exact_shared=True
            )

    def test_generated_manifest_and_rows_reconcile_when_present(self) -> None:
        # The current builders publish S1 v3, which trains on the pinned
        # prompt bank; the v1 directory predates prompt-bank support.
        contract, overlay = apply_s1_overlay(load_contract(), S1_V3_OVERLAY_PATH)
        settings = contract["curriculum"]["families"]["S1.9"]
        prompts = wrapper_prompts(
            overlay["prompt_banks"], MODULE_PROMPTS, bank_name="module"
        )
        configured = Path(settings["output_directory"])
        root = (
            configured
            if configured.is_absolute()
            else REPOSITORY_ROOT / configured
        )
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"generated corpus is absent: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(manifest), EXPECTED_MANIFEST_KEYS | {"s1_overlay", "prompt_bank"}
        )
        self.assertEqual(
            manifest["prompt_bank"]["sha256"], overlay["prompt_banks"]["sha256"]
        )
        self.assertEqual(
            manifest["schema_version"], "mentor-rl-world-model-s1-9-manifest-v1"
        )
        self.assertEqual(
            manifest["corpus_schema_version"],
            "mentor-rl-world-model-s1-9-corpus-v1",
        )
        self.assertEqual(manifest["section"], "S1.9")
        self.assertEqual(manifest["dataset_id"], settings["dataset_id"])
        self.assertEqual(manifest["seed"], 1729)
        self.assertEqual(
            manifest["row_counts"], {"train": 2000, "validation": 250, "test": 250}
        )
        self.assertEqual(
            manifest["form_counts"],
            {
                split: {
                    f"S1.9.{form}": count for form, count in counts.items()
                }
                for split, counts in EXPECTED_TRACE_COUNTS.items()
            },
        )
        self.assertEqual(
            manifest["contract_s1_9_sha256"], _stable_sha256(settings)
        )
        self.assertEqual(
            manifest["prompt_templates_sha256"], _stable_sha256(prompts)
        )
        self.assertEqual(
            manifest["request_templates_sha256"], _stable_sha256(PARSE_REQUESTS)
        )
        self.assertEqual(
            manifest["module_grammars_sha256"], _stable_sha256(MODULE_GRAMMARS)
        )
        self.assertEqual(
            manifest["result_shapes_sha256"],
            _stable_sha256(settings["result_shapes"]),
        )
        self._check_source_records(
            manifest, _source_index(contract), exact_shared=False
        )
        self.assertEqual(manifest["evidence"]["call_count"], 0)
        self.assertEqual(manifest["evidence"]["calls"], [])
        for split, count in manifest["row_counts"].items():
            path = root / f"{split}.jsonl"
            self.assertEqual(file_sha256(path), manifest["files"][path.name]["sha256"])
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), count)
            self.assertEqual(len({row["row_id"] for row in rows}), count)
            self.assertTrue(all(_calls(row) == [] for row in rows))
            self.assertTrue(all(_answer(row)["provenance"] is None for row in rows))
            self.assertTrue(
                all(_answer(row)["result"]["existence_asserted"] is False
                    for row in rows)
            )


if __name__ == "__main__":
    unittest.main()
