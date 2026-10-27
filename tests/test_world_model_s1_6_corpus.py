"""Focused tests for the canonical S1.6 layer-tag corpus builder."""

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

import runtime.world_model.s1_6_corpus as s1_6_module

from runtime.world_model.contract import (
    ContractError,
    load_contract,
    parse_layer_ref,
    tool_definitions,
)
from runtime.world_model.s1_6_corpus import (
    LAYER_PROMPTS,
    CorpusBuildError,
    LayerFact,
    ModuleFact,
    Scenario,
    audit_rows,
    build_fact_pools,
    build_corpus,
    build_row,
    duplicate_prompt_outcome_audit,
    filter_row_thinking,
    form_counts,
    largest_remainder_counts,
    masked_wording_classifier_audit,
    parse_layer_tag,
    select_scenarios,
    split_row_counts,
)
from runtime.world_model.s1_overlay import S1_V3_OVERLAY_PATH, apply_s1_overlay
from runtime.world_model.s1_prompt_banks import wrapper_prompts
from runtime.world_model.source_index import check_source_index
from runtime.world_model.sources import file_sha256
from runtime.world_model_training import (
    flatten_v3_trace_for_arrow,
    tokenize_v3_trace_for_sft,
)


SOURCE_INDEX = Path(
    "data/world_model_v3/source_indexes/"
    "world_model_source_index_"
    "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e.json.gz"
)
FORM_BY_BUCKET = {
    "bulk_pen_direct": 1,
    "scpen_one_field": 2,
    "humannet_direct": 3,
    "list_filter": 4,
    "wrong_input_type": 5,
    "malformed_tag": 6,
    "tf_target_direct": 7,
}
FIELD_NAMES = (
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
EXPECTED_PREFIX_COUNTS = {
    "HumanNetV3": 7,
    "TFs": 1,
    "bulkPEN": 14,
    "scPEN": 331,
}
EXPECTED_ARITY_COUNTS = {
    "HumanNetV3:2": 7,
    "TFs:2": 1,
    "bulkPEN:3": 10,
    "bulkPEN:4": 4,
    "scPEN:4": 310,
    "scPEN:5": 21,
}
EXPECTED_FAMILY_COUNTS = {
    "bulk_pen": 14,
    "coexpression": 1,
    "other": 5,
    "ppi": 1,
    "scpen": 331,
    "tf_target": 1,
}
EXPECTED_FORM_COUNTS = {
    "train": {1: 1250, 2: 1250, 3: 625, 4: 750, 5: 500, 6: 500, 7: 125},
    "validation": {1: 167, 2: 156, 3: 83, 4: 94, 5: 63, 6: 62, 7: 0},
    "test": {1: 167, 2: 156, 3: 83, 4: 94, 5: 63, 6: 62, 7: 0},
}
EXPECTED_POOL_COUNTS = {
    "train": {
        "all": 261,
        "bulk_pen": 10,
        "scpen": 245,
        "humannet": 5,
        "tf_target": 1,
        "modules": 50_156,
    },
    "validation": {
        "all": 53,
        "bulk_pen": 2,
        "scpen": 50,
        "humannet": 1,
        "tf_target": 0,
        "modules": 6_270,
    },
    "test": {
        "all": 39,
        "bulk_pen": 2,
        "scpen": 36,
        "humannet": 1,
        "tf_target": 0,
        "modules": 6_270,
    },
}


def _calls(row: dict[str, object]) -> list[dict[str, object]]:
    return [
        message["tool_call"]
        for message in row["messages"]
        if message["role"] == "assistant" and "tool_call" in message
    ]


def _matches_template(text: str, template: str) -> bool:
    prefix, suffix = template.split("{request}")
    return text.startswith(prefix) and text.endswith(suffix)


def _native_tags(text: str) -> list[str]:
    return re.findall(
        r"(?:HumanNetV3|TFs|bulkPEN|scPEN)"
        r":[A-Za-z0-9_-]+(?:\:[A-Za-z0-9_-]+)*",
        text,
    )


def _instruction_word_counts(text: str) -> list[int]:
    text = re.sub(
        r"(?<![A-Za-z0-9_-])"
        r"[A-Za-z0-9_-]+(?::[A-Za-z0-9_/-]+)+"
        r"(?![A-Za-z0-9_-])",
        "TECHNICALID",
        text,
    )
    return [
        len(re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*", sentence))
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]


def _layer_ref(fact: LayerFact) -> dict[str, object]:
    return {
        "layer_id": fact.layer_id,
        "layer_family": fact.layer_family,
        "source": fact.source,
        "network": fact.network,
        "organ": fact.organ,
        "region_path": (
            list(fact.region_path) if fact.region_path is not None else None
        ),
        "cell_type_path": (
            list(fact.cell_type_path)
            if fact.cell_type_path is not None
            else None
        ),
    }


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


class WorldModelS16AllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.curriculum = self.contract["curriculum"]
        self.settings = self.curriculum["families"]["S1.6"]

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
                "evaluation_allocations",
                "semantic_buckets",
                "one_field_names",
                "list_filter",
                "wrong_input_type",
                "malformed_subtype_counts",
                "context_split",
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
                "dataset_id": "world_model_v3_s1_6_layer_tag_parsing",
                "question_family": "layer_tag_parsing",
                "book_mode": "closed_book",
                "output_directory": "data/world_model_v3/corpora/s1/s1_6",
                "assignment_seed": 1729,
                "train_rows": 5000,
            },
        )
        self.assertEqual(
            self.settings["allocations"],
            {
                name: EXPECTED_FORM_COUNTS["train"][form]
                for name, form in FORM_BY_BUCKET.items()
            },
        )
        self.assertEqual(
            self.settings["evaluation_allocations"],
            {
                name: EXPECTED_FORM_COUNTS["validation"][form]
                for name, form in FORM_BY_BUCKET.items()
            },
        )
        self.assertEqual(
            self.settings["semantic_buckets"],
            {
                "direct_parse": [
                    "bulk_pen_direct",
                    "humannet_direct",
                    "tf_target_direct",
                ],
                "one_field": ["scpen_one_field"],
                "list_filter": ["list_filter"],
                "wrong_input_type": ["wrong_input_type"],
                "malformed_tag": ["malformed_tag"],
            },
        )
        self.assertEqual(self.settings["one_field_names"], list(FIELD_NAMES))
        self.assertEqual(
            self.settings["list_filter"],
            {
                "list_length": 3,
                "fields": list(FIELD_NAMES),
                "path_match": "exact_component",
                "training_outcome_counts": {
                    "match": 375,
                    "no_match": 375,
                },
            },
        )
        self.assertEqual(
            self.settings["wrong_input_type"],
            {
                "found_input_type": "module_id",
                "module_namespace_counts": {
                    "rwr_loe": 250,
                    "mentor_ev": 250,
                },
            },
        )
        self.assertEqual(
            self.settings["malformed_subtype_counts"],
            dict(zip(MALFORMED_SUBTYPES, (84, 84, 83, 83, 83, 83))),
        )
        self.assertEqual(
            self.settings["context_split"],
            {
                "unit": "exact_organ_and_region_path_or_network",
                "counts": {"train": 27, "validation": 4, "test": 4},
                "strata": {
                    "train": {
                        "humannet": 5,
                        "bulk_only": 9,
                        "scpen_only": 11,
                        "shared_pen": 1,
                        "tf_target": 1,
                    },
                    "validation": {
                        "humannet": 1,
                        "bulk_only": 1,
                        "scpen_only": 1,
                        "shared_pen": 1,
                        "tf_target": 0,
                    },
                    "test": {
                        "humannet": 1,
                        "bulk_only": 1,
                        "scpen_only": 1,
                        "shared_pen": 1,
                        "tf_target": 0,
                    },
                },
                "singleton_policy": "train_only",
            },
        )

    def test_split_and_form_allocations_are_exact(self) -> None:
        row_counts = split_row_counts(
            self.settings["train_rows"],
            self.curriculum["split_fractions"],
        )
        self.assertEqual(
            row_counts, {"train": 5000, "validation": 625, "test": 625}
        )
        self.assertEqual(
            {
                split: form_counts(total, self.settings)
                for split, total in row_counts.items()
            },
            EXPECTED_FORM_COUNTS,
        )
        self.assertEqual(sum(EXPECTED_FORM_COUNTS["train"].values()), 5000)
        self.assertEqual(
            sum(EXPECTED_FORM_COUNTS["validation"].values()), 625
        )

    def test_ordered_suballocations_use_largest_remainder(self) -> None:
        one_field = largest_remainder_counts(
            1250, {name: 1 / 6 for name in FIELD_NAMES}
        )
        self.assertEqual(
            one_field,
            dict(zip(FIELD_NAMES, (209, 209, 208, 208, 208, 208))),
        )
        self.assertEqual(
            largest_remainder_counts(
                156, {name: 1 / 6 for name in FIELD_NAMES}
            ),
            {name: 26 for name in FIELD_NAMES},
        )
        self.assertEqual(
            largest_remainder_counts(
                63, {"rwr_loe": 0.5, "mentor_ev": 0.5}
            ),
            {"rwr_loe": 32, "mentor_ev": 31},
        )
        self.assertEqual(
            largest_remainder_counts(
                62, {name: 1 / 6 for name in MALFORMED_SUBTYPES}
            ),
            dict(zip(MALFORMED_SUBTYPES, (11, 11, 10, 10, 10, 10))),
        )

    def test_prompt_pool_is_split_held_out(self) -> None:
        phrases_by_split = {
            split: {
                template
                for templates in framings.values()
                for template in templates
            }
            for split, framings in LAYER_PROMPTS.items()
        }
        self.assertEqual(
            set(phrases_by_split), {"train", "validation", "test"}
        )
        for split, framings in LAYER_PROMPTS.items():
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


class WorldModelS16SourceAndParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.flist_path = Path(
            self.contract["coordinate_system"]["source_flist_path"]
        )

    def test_pinned_flist_and_source_index_have_exact_native_counts(self) -> None:
        if not SOURCE_INDEX.is_file():
            self.skipTest(f"canonical source index is absent: {SOURCE_INDEX}")
        # The live flist changed after this corpus was frozen. Check the
        # source snapshot that the builder actually consumes.
        payload = check_source_index(SOURCE_INDEX, self.contract)
        self.assertEqual(
            payload["sources"]["multiplex_flist"],
            {
                "file_name": self.flist_path.name,
                "sha256": self.contract["coordinate_system"]["source_flist_sha256"],
            },
        )
        layer_ids = payload["layer_ids"]
        self.assertEqual(len(layer_ids), 353)
        self.assertEqual(
            Counter(layer_id.split(":", 1)[0] for layer_id in layer_ids),
            EXPECTED_PREFIX_COUNTS,
        )
        self.assertEqual(
            Counter(
                f"{parts[0]}:{len(parts)}"
                for layer_id in layer_ids
                for parts in (layer_id.split(":"),)
            ),
            EXPECTED_ARITY_COUNTS,
        )

        self.assertEqual(payload["counts"]["layer_count"], 353)

    def test_every_native_tag_round_trips_with_exact_fields(self) -> None:
        if not SOURCE_INDEX.is_file():
            self.skipTest(f"canonical source index is absent: {SOURCE_INDEX}")
        payload = check_source_index(SOURCE_INDEX, self.contract)
        facts = [
            parse_layer_tag(layer_id, self.contract)
            for layer_id in payload["layer_ids"]
        ]
        self.assertTrue(all(isinstance(fact, LayerFact) for fact in facts))
        self.assertEqual(
            Counter(fact.layer_family for fact in facts),
            EXPECTED_FAMILY_COUNTS,
        )
        for fact in facts:
            self.assertEqual(
                _layer_ref(fact), parse_layer_ref(self.contract, fact.layer_id)
            )
            self.assertEqual(
                list(_layer_ref(fact)),
                [
                    "layer_id",
                    "layer_family",
                    "source",
                    "network",
                    "organ",
                    "region_path",
                    "cell_type_path",
                ],
            )
            self.assertEqual(_layer_ref(fact)["layer_id"], fact.layer_id)

        by_id = {fact.layer_id: fact for fact in facts}
        self.assertEqual(
            _layer_ref(by_id["HumanNetV3:string_ppi"]),
            {
                "layer_id": "HumanNetV3:string_ppi",
                "layer_family": "ppi",
                "source": "HumanNetV3",
                "network": "string_ppi",
                "organ": None,
                "region_path": None,
                "cell_type_path": None,
            },
        )
        self.assertEqual(
            by_id["bulkPEN:brain:basal_ganglia:caudate"].region_path,
            ("basal_ganglia", "caudate"),
        )
        self.assertEqual(
            by_id[
                "scPEN:brain:pfc:inhibitory_interneuron:pv"
            ].cell_type_path,
            ("inhibitory_interneuron", "pv"),
        )
        self.assertEqual(
            _layer_ref(by_id["TFs:brain"]),
            {
                "layer_id": "TFs:brain",
                "layer_family": "tf_target",
                "source": None,
                "network": None,
                "organ": "brain",
                "region_path": None,
                "cell_type_path": None,
            },
        )

    def test_parser_rejects_all_six_malformed_minimal_pairs(self) -> None:
        invalid = {
            "unrecognized_prefix": "scPENX:brain:hippocampus:astrocyte",
            "wrong_arity": "TFs:brain:extra",
            "missing_path": "scPEN:brain:hippocampus",
            "excess_path": "bulkPEN:brain:basal_ganglia:caudate:extra",
            "synthetic_field": "HumanNetV3:string_ppi:v3",
            "mixed_separator": "scPEN/brain:hippocampus:astrocyte",
        }
        self.assertEqual(tuple(invalid), MALFORMED_SUBTYPES)
        for subtype, value in invalid.items():
            with self.subTest(subtype=subtype):
                with self.assertRaises(ContractError):
                    parse_layer_ref(self.contract, value)
                with self.assertRaises(CorpusBuildError):
                    parse_layer_tag(value, self.contract)

    def test_real_fact_pools_hold_out_exact_context_units(self) -> None:
        if not SOURCE_INDEX.is_file():
            self.skipTest(f"canonical source index is absent: {SOURCE_INDEX}")
        payload = check_source_index(SOURCE_INDEX, self.contract)
        settings = self.contract["curriculum"]["families"]["S1.6"]
        pools = build_fact_pools(
            payload,
            settings=settings,
            seed=settings["assignment_seed"],
            contract=self.contract,
        )
        self.assertEqual(set(pools), {"train", "validation", "test"})
        expected_pool_names = {
            "all",
            "bulk_pen",
            "scpen",
            "humannet",
            "tf_target",
            "modules",
        }
        for split_pools in pools.values():
            self.assertEqual(set(split_pools), expected_pool_names)
        self.assertEqual(
            {
                split: {name: len(values) for name, values in split_pools.items()}
                for split, split_pools in pools.items()
            },
            EXPECTED_POOL_COUNTS,
        )
        self.assertEqual(
            {
                split: Counter(item.namespace for item in split_pools["modules"])
                for split, split_pools in pools.items()
            },
            {
                "train": Counter({"rwr_loe": 37_613, "mentor_ev": 12_543}),
                "validation": Counter({"rwr_loe": 4_702, "mentor_ev": 1_568}),
                "test": Counter({"rwr_loe": 4_702, "mentor_ev": 1_568}),
            },
        )

        all_facts = [fact for split in pools.values() for fact in split["all"]]
        self.assertEqual(len(all_facts), 353)
        self.assertEqual(
            {fact.layer_id for fact in all_facts}, set(payload["layer_ids"])
        )
        self.assertEqual(
            Counter(fact.layer_family for fact in all_facts),
            EXPECTED_FAMILY_COUNTS,
        )

        contexts = {
            split: {fact.context_key for fact in split_pools["all"]}
            for split, split_pools in pools.items()
        }
        self.assertEqual(
            {split: len(values) for split, values in contexts.items()},
            {"train": 27, "validation": 4, "test": 4},
        )
        expected_strata = {
            "train": {
                "humannet": 5,
                "bulk_only": 9,
                "scpen_only": 11,
                "shared_pen": 1,
                "tf_target": 1,
            },
            "validation": {
                "humannet": 1,
                "bulk_only": 1,
                "scpen_only": 1,
                "shared_pen": 1,
                "tf_target": 0,
            },
            "test": {
                "humannet": 1,
                "bulk_only": 1,
                "scpen_only": 1,
                "shared_pen": 1,
                "tf_target": 0,
            },
        }
        for split, split_pools in pools.items():
            families_by_context: dict[str, set[str]] = {}
            for fact in split_pools["all"]:
                families_by_context.setdefault(fact.context_key, set()).add(
                    fact.layer_family
                )
            observed = Counter()
            for context_key, families in families_by_context.items():
                if context_key.startswith("network:"):
                    observed["humannet"] += 1
                elif context_key.startswith("tf:"):
                    observed["tf_target"] += 1
                elif families >= {"bulk_pen", "scpen"}:
                    observed["shared_pen"] += 1
                elif families == {"bulk_pen"}:
                    observed["bulk_only"] += 1
                elif families == {"scpen"}:
                    observed["scpen_only"] += 1
                else:
                    self.fail((split, context_key, families))
            self.assertEqual(observed, Counter(expected_strata[split]))
        self.assertTrue(contexts["train"].isdisjoint(contexts["validation"]))
        self.assertTrue(contexts["train"].isdisjoint(contexts["test"]))
        self.assertTrue(
            contexts["validation"].isdisjoint(contexts["test"])
        )
        self.assertEqual(
            [fact.layer_id for fact in pools["train"]["tf_target"]],
            ["TFs:brain"],
        )
        self.assertEqual(pools["validation"]["tf_target"], [])
        self.assertEqual(pools["test"]["tf_target"], [])

        for split, split_pools in pools.items():
            for name, family in (
                ("bulk_pen", "bulk_pen"),
                ("scpen", "scpen"),
                ("tf_target", "tf_target"),
            ):
                self.assertTrue(
                    all(fact.layer_family == family for fact in split_pools[name])
                )
            self.assertTrue(
                all(
                    fact.layer_id.startswith("HumanNetV3:")
                    for fact in split_pools["humannet"]
                )
            )
            self.assertTrue(split_pools["modules"], split)


class WorldModelS16SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.6"]
        if not SOURCE_INDEX.is_file():
            raise unittest.SkipTest(
                f"canonical source index is absent: {SOURCE_INDEX}"
            )
        cls.payload = check_source_index(SOURCE_INDEX, cls.contract)
        cls.pools = build_fact_pools(
            cls.payload,
            settings=cls.settings,
            seed=cls.settings["assignment_seed"],
            contract=cls.contract,
        )
        cls.scenarios = select_scenarios(
            cls.pools,
            EXPECTED_FORM_COUNTS,
            seed=cls.settings["assignment_seed"],
            framing_shares=cls.contract["curriculum"]["framing"],
            settings=cls.settings,
        )

    def test_selection_is_exact_deterministic_and_source_backed(self) -> None:
        repeated = select_scenarios(
            self.pools,
            EXPECTED_FORM_COUNTS,
            seed=self.settings["assignment_seed"],
            framing_shares=self.contract["curriculum"]["framing"],
            settings=self.settings,
        )
        source_ids = set(self.payload["layer_ids"])
        self.assertNotIn("HumanNetV3:biogrid_ppi", source_ids)
        for split, scenarios in self.scenarios.items():
            with self.subTest(split=split):
                self.assertEqual(
                    Counter(item.form for item in scenarios),
                    Counter(EXPECTED_FORM_COUNTS[split]),
                )
                self.assertEqual(
                    Counter(item.framing for item in scenarios),
                    largest_remainder_counts(
                        len(scenarios), self.contract["curriculum"]["framing"]
                    ),
                )
                self.assertEqual(
                    {item.source_trace for item in scenarios},
                    {
                        f"S1.6.{form}"
                        for form, count in EXPECTED_FORM_COUNTS[split].items()
                        if count
                    },
                )
                for item in scenarios:
                    if isinstance(item.fact, LayerFact):
                        self.assertIn(item.fact.layer_id, source_ids)
                    for companion in item.companions:
                        self.assertIn(companion.layer_id, source_ids)
                self.assertFalse(
                    any(
                        isinstance(item.fact, LayerFact)
                        and item.fact.layer_id == "HumanNetV3:biogrid_ppi"
                        for item in scenarios
                    )
                )
                self.assertTrue(
                    all(
                        item.companions == ()
                        for item in scenarios
                        if item.form in {1, 3, 7}
                    )
                )

        self.assertEqual(
            {
                split: [item.scenario_id for item in scenarios]
                for split, scenarios in self.scenarios.items()
            },
            {
                split: [item.scenario_id for item in scenarios]
                for split, scenarios in repeated.items()
            },
        )
        self.assertEqual(
            {
                item.fact.layer_family
                for scenarios in self.scenarios.values()
                for item in scenarios
                if item.form == 3
            },
            {"ppi", "coexpression", "other"},
        )

    def test_selection_balances_fields_outcomes_and_negative_classes(self) -> None:
        expected_field_counts = {
            "train": dict(
                zip(FIELD_NAMES, (209, 209, 208, 208, 208, 208))
            ),
            "validation": {name: 26 for name in FIELD_NAMES},
            "test": {name: 26 for name in FIELD_NAMES},
        }
        expected_filter_fields = {
            "train": {name: 125 for name in FIELD_NAMES},
            "validation": dict(zip(FIELD_NAMES, (16, 16, 16, 16, 15, 15))),
            "test": dict(zip(FIELD_NAMES, (16, 16, 16, 16, 15, 15))),
        }
        expected_filter_outcomes = {
            "train": {"match": 375, "no_match": 375},
            "validation": {"match": 47, "no_match": 47},
            "test": {"match": 47, "no_match": 47},
        }
        expected_modules = {
            "train": {"rwr_loe": 250, "mentor_ev": 250},
            "validation": {"rwr_loe": 32, "mentor_ev": 31},
            "test": {"rwr_loe": 32, "mentor_ev": 31},
        }
        expected_malformed = {
            "train": dict(
                zip(MALFORMED_SUBTYPES, (84, 84, 83, 83, 83, 83))
            ),
            "validation": dict(
                zip(MALFORMED_SUBTYPES, (11, 11, 10, 10, 10, 10))
            ),
            "test": dict(
                zip(MALFORMED_SUBTYPES, (11, 11, 10, 10, 10, 10))
            ),
        }
        for split, scenarios in self.scenarios.items():
            by_form = {
                form: [item for item in scenarios if item.form == form]
                for form in range(1, 8)
            }
            with self.subTest(split=split):
                self.assertEqual(
                    Counter(item.requested_field for item in by_form[2]),
                    expected_field_counts[split],
                )
                self.assertEqual(
                    Counter(item.requested_field for item in by_form[4]),
                    expected_filter_fields[split],
                )
                self.assertEqual(
                    Counter(item.filter_outcome for item in by_form[4]),
                    expected_filter_outcomes[split],
                )
                self.assertTrue(
                    all(len(item.companions) == 3 for item in by_form[4])
                )
                self.assertTrue(
                    all(isinstance(item.fact, ModuleFact) for item in by_form[5])
                )
                self.assertEqual(
                    Counter(item.fact.namespace for item in by_form[5]),
                    expected_modules[split],
                )
                self.assertEqual(
                    Counter(item.malformed_subtype for item in by_form[6]),
                    expected_malformed[split],
                )
                self.assertTrue(
                    all(item.malformed_input for item in by_form[6])
                )
                available_prefixes = {
                    fact.layer_id.split(":", 1)[0]
                    for fact in self.pools[split]["all"]
                }
                for subtype, expected_count in expected_malformed[split].items():
                    malformed = [
                        item
                        for item in by_form[6]
                        if item.malformed_subtype == subtype
                    ]
                    self.assertEqual(
                        len({item.malformed_input for item in malformed}),
                        expected_count,
                    )
                    expected_prefixes = (
                        available_prefixes & {"bulkPEN", "scPEN"}
                        if subtype == "missing_path"
                        else available_prefixes
                    )
                    self.assertEqual(
                        {
                            item.fact.layer_id.split(":", 1)[0]
                            for item in malformed
                        },
                        expected_prefixes,
                    )
                if split == "train":
                    self.assertTrue(
                        all(
                            item.fact.layer_id == "TFs:brain"
                            for item in by_form[7]
                        )
                    )
                else:
                    self.assertEqual(by_form[7], [])

    def test_list_filters_use_exact_components_without_biological_inference(self) -> None:
        for split, scenarios in self.scenarios.items():
            for item in (value for value in scenarios if value.form == 4):
                facts = item.companions
                expected_matches = []
                for fact in facts:
                    value = getattr(fact, item.requested_field)
                    if item.requested_field in {"region_path", "cell_type_path"}:
                        matches = (
                            value is not None and item.filter_value in value
                        )
                    else:
                        matches = value == item.filter_value
                    if matches:
                        expected_matches.append(fact.layer_id)
                with self.subTest(split=split, scenario=item.scenario_id):
                    self.assertEqual(bool(expected_matches), item.filter_outcome == "match")

    def test_templates_and_context_units_are_held_out(self) -> None:
        split_units = {
            split: {
                unit
                for item in scenarios
                for unit in item.split_units
            }
            for split, scenarios in self.scenarios.items()
        }
        self.assertTrue(
            split_units["train"].isdisjoint(split_units["validation"])
        )
        self.assertTrue(split_units["train"].isdisjoint(split_units["test"]))
        self.assertTrue(
            split_units["validation"].isdisjoint(split_units["test"])
        )
        for split, scenarios in self.scenarios.items():
            for form, expected_count in EXPECTED_FORM_COUNTS[split].items():
                if not expected_count:
                    continue
                items = [item for item in scenarios if item.form == form]
                for framing in ("direct", "step"):
                    framed = [item for item in items if item.framing == framing]
                    self.assertTrue(framed, (split, form, framing))
                    self.assertEqual(
                        {item.template_index for item in framed},
                        set(range(len(LAYER_PROMPTS[split][framing]))),
                    )


class WorldModelS16RowSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.6"]
        cls.tools = tool_definitions(cls.contract)
        if not SOURCE_INDEX.is_file():
            raise unittest.SkipTest(
                f"canonical source index is absent: {SOURCE_INDEX}"
            )
        cls.payload = check_source_index(SOURCE_INDEX, cls.contract)
        cls.pools = build_fact_pools(
            cls.payload,
            settings=cls.settings,
            seed=cls.settings["assignment_seed"],
            contract=cls.contract,
        )
        cls.scenarios = select_scenarios(
            cls.pools,
            EXPECTED_FORM_COUNTS,
            seed=cls.settings["assignment_seed"],
            framing_shares=cls.contract["curriculum"]["framing"],
            settings=cls.settings,
        )

    def _build(self, scenario: Scenario) -> dict[str, object]:
        row, evidence = build_row(
            scenario,
            contract=self.contract,
            tools=self.tools,
        )
        self.assertEqual(evidence, [])
        return row

    def _first(self, form: int, **attributes: object) -> Scenario:
        return next(
            scenario
            for scenarios in self.scenarios.values()
            for scenario in scenarios
            if scenario.form == form
            and all(
                getattr(scenario, key) == value
                for key, value in attributes.items()
            )
        )

    @staticmethod
    def _expected_filter_matches(scenario: Scenario) -> list[str]:
        matches: list[str] = []
        for fact in scenario.companions:
            value = getattr(fact, scenario.requested_field)
            if scenario.requested_field in {"region_path", "cell_type_path"}:
                keep = value is not None and scenario.filter_value in value
            else:
                keep = value == scenario.filter_value
            if keep:
                matches.append(fact.layer_id)
        return matches

    def test_all_seven_forms_have_exact_closed_book_semantics(self) -> None:
        scenarios = [self._first(form) for form in range(1, 8)]
        for scenario in scenarios:
            with self.subTest(form=scenario.form):
                row = self._build(scenario)
                self.assertEqual(
                    list(row), ["row_id", "messages", "tools", "metadata"]
                )
                self.assertEqual(row["tools"], self.tools)
                self.assertEqual(
                    [message["role"] for message in row["messages"]],
                    ["system", "user", "assistant"],
                )
                self.assertEqual(_calls(row), [])
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
                self.assertIsNone(answer["provenance"])
                metadata = row["metadata"]
                expected_metadata_keys = set(
                    self.contract["rows"]["metadata"]["required"]
                )
                if scenario.form == 5:
                    expected_metadata_keys.add("found_input_type")
                self.assertEqual(set(metadata), expected_metadata_keys)
                self.assertEqual(metadata["book_mode"], "closed_book")
                self.assertEqual(metadata["step"], "S1")
                self.assertEqual(
                    metadata["question_family"], "layer_tag_parsing"
                )
                self.assertEqual(metadata["form"], scenario.form)
                self.assertEqual(metadata["source_trace"], scenario.source_trace)
                self.assertEqual(
                    metadata["split_unit"], scenario.metadata_split_unit
                )

                if scenario.form in {5, 6}:
                    self.assertEqual(answer["status"], "invalid_input")
                    self.assertEqual(
                        answer["continuation_decision"], "defer"
                    )
                    self.assertEqual(
                        answer["result"]["expected"], "layer_tag"
                    )
                    self.assertNotIn("<LAYER:", answer["summary"])
                else:
                    self.assertEqual(answer["status"], "resolved")
                    self.assertEqual(
                        answer["continuation_decision"],
                        "continue" if scenario.framing == "step" else "stop",
                    )
                    self.assertIsNone(answer["blocked_reason"])
                self.assertNotIn("version v", answer["summary"].casefold())
                if scenario.form not in {5, 6}:
                    lowered_summary = answer["summary"].casefold()
                    for invented in (
                        "organ global",
                        "cell type all",
                        "region all",
                        "edge",
                        "ontology",
                        "exists",
                    ):
                        self.assertNotIn(invented, lowered_summary)

    def test_valid_single_tag_results_have_all_seven_fields(self) -> None:
        for form in (1, 2, 3, 7):
            scenario = self._first(form)
            row = self._build(scenario)
            result = row["messages"][-1]["content"]["result"]
            with self.subTest(form=form):
                self.assertEqual(result, _layer_ref(scenario.fact))
                self.assertEqual(
                    list(result),
                    [
                        "layer_id",
                        "layer_family",
                        "source",
                        "network",
                        "organ",
                        "region_path",
                        "cell_type_path",
                    ],
                )
                self.assertIn(
                    f"<LAYER:{scenario.fact.layer_id}>",
                    row["messages"][-1]["content"]["summary"],
                )
                user = row["messages"][1]["content"]
                self.assertEqual(user.count(scenario.fact.layer_id), 1)
                self.assertEqual(_native_tags(user), [scenario.fact.layer_id])

    def test_one_field_questions_still_return_the_full_layer_ref(self) -> None:
        for field in FIELD_NAMES:
            scenario = self._first(2, requested_field=field)
            row = self._build(scenario)
            result = row["messages"][-1]["content"]["result"]
            with self.subTest(field=field):
                self.assertEqual(result, _layer_ref(scenario.fact))
                self.assertEqual(result[field], _layer_ref(scenario.fact)[field])
                self.assertIn(
                    field.replace("_", " "), row["messages"][1]["content"]
                )

    def test_list_filter_rows_preserve_order_and_match_exact_components(self) -> None:
        samples = {
            (scenario.requested_field, scenario.filter_outcome): scenario
            for scenarios in self.scenarios.values()
            for scenario in scenarios
            if scenario.form == 4
        }
        self.assertEqual({field for field, _ in samples}, set(FIELD_NAMES))
        self.assertEqual(
            {outcome for _, outcome in samples}, {"match", "no_match"}
        )
        for (field, outcome), scenario in samples.items():
            row = self._build(scenario)
            result = row["messages"][-1]["content"]["result"]
            facts = scenario.companions
            with self.subTest(field=field, outcome=outcome):
                self.assertEqual(
                    list(result),
                    ["layers", "filter", "matching_layer_ids"],
                )
                self.assertEqual(
                    result["layers"], [_layer_ref(fact) for fact in facts]
                )
                self.assertEqual(
                    _native_tags(row["messages"][1]["content"]),
                    [fact.layer_id for fact in facts],
                )
                self.assertEqual(
                    result["filter"],
                    {
                        "field": field,
                        "value": scenario.filter_value,
                    },
                )
                self.assertEqual(
                    result["matching_layer_ids"],
                    self._expected_filter_matches(scenario),
                )
                self.assertEqual(
                    tuple(result["matching_layer_ids"]),
                    scenario.expected_matching_layer_ids,
                )
                self.assertEqual(
                    bool(result["matching_layer_ids"]), outcome == "match"
                )

        facts_by_id = {
            fact.layer_id: fact
            for split_pools in self.pools.values()
            for fact in split_pools["all"]
        }
        base = self._first(4)
        negative = replace(
            base,
            fact=facts_by_id["bulkPEN:brain:amygdala"],
            companions=(
                facts_by_id[
                    "scPEN:brain:amygdaloid-complex:astrocyte"
                ],
                facts_by_id["bulkPEN:brain:hippocampus"],
                facts_by_id["HumanNetV3:string_ppi"],
            ),
            requested_field="region_path",
            filter_value="amygdala",
            filter_outcome="no_match",
            occurrence=90_001,
        )
        result = self._build(negative)["messages"][-1]["content"]["result"]
        self.assertNotIn(
            "scPEN:brain:amygdaloid-complex:astrocyte",
            result["matching_layer_ids"],
        )

        two_part = replace(
            base,
            fact=facts_by_id["bulkPEN:brain:basal_ganglia:caudate"],
            companions=(
                facts_by_id["bulkPEN:brain:basal_ganglia:caudate"],
                facts_by_id["bulkPEN:brain:hippocampus"],
                facts_by_id["HumanNetV3:string_ppi"],
            ),
            requested_field="region_path",
            filter_value="caudate",
            filter_outcome="match",
            occurrence=90_002,
        )
        two_part_result = self._build(two_part)["messages"][-1]["content"][
            "result"
        ]
        self.assertEqual(
            two_part_result["matching_layer_ids"],
            self._expected_filter_matches(two_part),
        )

    def test_wrong_type_and_malformed_rows_use_exact_blocked_reasons(self) -> None:
        for namespace in ("rwr_loe", "mentor_ev"):
            scenario = next(
                item
                for item in self.scenarios["train"]
                if item.form == 5 and item.fact.namespace == namespace
            )
            row = self._build(scenario)
            answer = row["messages"][-1]["content"]
            self.assertEqual(
                answer["result"],
                {"input": scenario.fact.module_id, "expected": "layer_tag"},
            )
            self.assertEqual(
                answer["blocked_reason"],
                f"The input {scenario.fact.module_id} is a module ID, "
                "not a layer tag.",
            )
            self.assertEqual(
                row["metadata"]["blocked_reason_code"], "wrong_input_type"
            )
            self.assertEqual(
                row["metadata"]["found_input_type"], "module_id"
            )

        for subtype in MALFORMED_SUBTYPES:
            scenario = self._first(6, malformed_subtype=subtype)
            row = self._build(scenario)
            answer = row["messages"][-1]["content"]
            with self.subTest(subtype=subtype):
                self.assertEqual(
                    answer["result"],
                    {"input": scenario.malformed_input, "expected": "layer_tag"},
                )
                self.assertEqual(
                    answer["blocked_reason"],
                    f"The input {scenario.malformed_input} is not a "
                    "well-formed layer tag.",
                )
                self.assertEqual(
                    row["metadata"]["blocked_reason_code"], "invalid_input"
                )
                self.assertNotIn("found_input_type", row["metadata"])
                with self.assertRaises(CorpusBuildError):
                    parse_layer_tag(scenario.malformed_input, self.contract)

    def test_rendered_prompts_and_thinking_are_split_safe(self) -> None:
        for split, scenarios in self.scenarios.items():
            held_out = [
                template
                for other, framings in LAYER_PROMPTS.items()
                if other != split
                for templates in framings.values()
                for template in templates
            ]
            for scenario in scenarios[:50]:
                row = self._build(scenario)
                user = row["messages"][1]["content"]
                thinking = row["messages"][-1]["thinking"]
                with self.subTest(split=split, scenario=scenario.scenario_id):
                    self.assertEqual(
                        sum(
                            _matches_template(user, template)
                            for templates in LAYER_PROMPTS[split].values()
                            for template in templates
                        ),
                        1,
                    )
                    self.assertFalse(
                        any(_matches_template(user, template) for template in held_out)
                    )
                    self.assertTrue(thinking.strip())
                    self.assertNotIn("answer schema", thinking.casefold())
                    self.assertEqual(filter_row_thinking(row, scenario), 1)

    def test_thinking_filter_rejects_schema_and_false_parse_claims(self) -> None:
        scenario = self._first(1)
        row = self._build(scenario)
        mutations = (
            "",
            "I will follow the answer schema for this layer.",
            "The tag has a synthetic version field that the prompt does not show.",
            "I will call a registry tool to parse this tag.",
            row["messages"][-1]["thinking"] + " The region is invented_cortex.",
        )
        for thinking in mutations:
            changed = deepcopy(row)
            changed["messages"][-1]["thinking"] = thinking
            with self.subTest(thinking=thinking):
                with self.assertRaises(CorpusBuildError):
                    filter_row_thinking(changed, scenario)

    def test_row_rendering_is_deterministic(self) -> None:
        for form in range(1, 8):
            scenario = self._first(form)
            self.assertEqual(self._build(scenario), self._build(scenario))

    def test_training_adapter_accepts_closed_book_no_call_rows(self) -> None:
        scenario = self._first(1)
        row = self._build(scenario)
        flattened = flatten_v3_trace_for_arrow(row, expected_split="train")
        self.assertEqual(flattened["question_family"], "layer_tag_parsing")
        self.assertEqual(flattened["record_id"], row["row_id"])
        self.assertEqual(flattened["fact_id"], scenario.metadata_split_unit)
        self.assertEqual(flattened["split"], "train")

        tokenizer = _Tokenizer()
        tokenized = tokenize_v3_trace_for_sft(row, tokenizer, max_length=64)
        self.assertEqual(tokenized["input_ids"], [1, 10, 20, 60, 61])
        self.assertEqual(tokenized["completion_mask"], [0, 0, 0, 1, 1])
        self.assertEqual(len(tokenizer.rendered_messages), 2)
        self.assertEqual(
            [message["role"] for message in tokenizer.rendered_messages[-1]],
            ["system", "user", "assistant"],
        )


class WorldModelS16AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.6"]
        cls.tools = tool_definitions(cls.contract)
        if not SOURCE_INDEX.is_file():
            raise unittest.SkipTest(
                f"canonical source index is absent: {SOURCE_INDEX}"
            )
        payload = check_source_index(SOURCE_INDEX, cls.contract)
        cls.pools = build_fact_pools(
            payload,
            settings=cls.settings,
            seed=cls.settings["assignment_seed"],
            contract=cls.contract,
        )
        cls.scenarios = select_scenarios(
            cls.pools,
            EXPECTED_FORM_COUNTS,
            seed=cls.settings["assignment_seed"],
            framing_shares=cls.contract["curriculum"]["framing"],
            settings=cls.settings,
        )
        cls.rows: dict[str, list[dict[str, object]]] = {}
        for split, scenarios in cls.scenarios.items():
            cls.rows[split] = [
                build_row(
                    scenario,
                    contract=cls.contract,
                    tools=cls.tools,
                )[0]
                for scenario in scenarios
            ]

    def test_every_row_has_zero_calls_null_provenance_and_one_thought(self) -> None:
        for split, rows in self.rows.items():
            self.assertEqual(len(rows), sum(EXPECTED_FORM_COUNTS[split].values()))
            held_out_templates = [
                template
                for other, framings in LAYER_PROMPTS.items()
                if other != split
                for templates in framings.values()
                for template in templates
            ]
            for row, scenario in zip(rows, self.scenarios[split]):
                self.assertEqual(_calls(row), [])
                self.assertFalse(
                    any(message["role"] == "tool" for message in row["messages"])
                )
                user = row["messages"][1]["content"]
                self.assertEqual(
                    sum(
                        _matches_template(user, template)
                        for templates in LAYER_PROMPTS[split].values()
                        for template in templates
                    ),
                    1,
                )
                self.assertFalse(
                    any(
                        _matches_template(user, template)
                        for template in held_out_templates
                    )
                )
                self.assertTrue(row["messages"][-1]["thinking"].strip())
                self.assertTrue(_instruction_word_counts(user))
                self.assertLessEqual(max(_instruction_word_counts(user)), 20)
                if scenario.form in {1, 2, 3, 7}:
                    self.assertEqual(scenario.companions, ())
                    self.assertEqual(
                        _native_tags(user),
                        [scenario.fact.layer_id],
                    )
                elif scenario.form == 4:
                    self.assertEqual(
                        _native_tags(user),
                        [fact.layer_id for fact in scenario.companions],
                    )
                self.assertIsNone(row["messages"][-1]["content"]["provenance"])
                self.assertEqual(filter_row_thinking(row, scenario), 1)

    def test_duplicate_and_masked_wrapper_audits_pass_canonical_rows(self) -> None:
        duplicate = duplicate_prompt_outcome_audit(self.rows, self.scenarios)
        self.assertTrue(duplicate["pass"])
        for split in ("train", "validation", "test"):
            self.assertEqual(
                duplicate["splits"][split][
                    "duplicate_prompt_tool_outcome_count"
                ],
                0,
            )

        wording = masked_wording_classifier_audit(self.rows, self.scenarios)
        self.assertEqual(wording["scope"], "outer_wrapper_only")
        self.assertTrue(wording["pass"])
        for split in ("validation", "test"):
            self.assertTrue(wording["splits"][split]["gate_applied"])
            self.assertTrue(wording["splits"][split]["pass"])

    def test_aggregate_audit_reports_exact_counts_and_gates(self) -> None:
        audit = audit_rows(
            self.rows,
            self.scenarios,
            EXPECTED_FORM_COUNTS,
            framing_shares=self.contract["curriculum"]["framing"],
            settings=self.settings,
            contract=self.contract,
        )
        self.assertEqual(
            set(audit),
            {
                "status_counts",
                "tool_call_counts",
                "framing_counts",
                "prompt_template_counts",
                "field_counts",
                "filter_counts",
                "module_namespace_counts",
                "malformed_subtype_counts",
                "semantic_bucket_counts",
                "context_counts",
                "context_stratum_counts",
                "per_form_gates",
                "prompt_structure_audit",
                "masked_wording_classifier",
                "thinking_filter",
                "duplicate_prompt_tool_outcome",
                "split_context_sha256",
            },
        )
        expected_status = {
            "train": {"invalid_input": 1000, "resolved": 4000},
            "validation": {"invalid_input": 125, "resolved": 500},
            "test": {"invalid_input": 125, "resolved": 500},
        }
        self.assertEqual(audit["status_counts"], expected_status)
        self.assertEqual(
            audit["tool_call_counts"],
            {
                "train": {"0": 5000},
                "validation": {"0": 625},
                "test": {"0": 625},
            },
        )
        self.assertEqual(
            audit["framing_counts"],
            {
                "train": {"direct": 2500, "step": 2500},
                "validation": {"direct": 313, "step": 312},
                "test": {"direct": 313, "step": 312},
            },
        )
        self.assertEqual(
            audit["semantic_bucket_counts"],
            {
                "train": {
                    "direct_parse": 2000,
                    "one_field": 1250,
                    "list_filter": 750,
                    "wrong_input_type": 500,
                    "malformed_tag": 500,
                },
                "validation": {
                    "direct_parse": 250,
                    "one_field": 156,
                    "list_filter": 94,
                    "wrong_input_type": 63,
                    "malformed_tag": 62,
                },
                "test": {
                    "direct_parse": 250,
                    "one_field": 156,
                    "list_filter": 94,
                    "wrong_input_type": 63,
                    "malformed_tag": 62,
                },
            },
        )
        self.assertEqual(
            audit["context_counts"],
            {"train": 27, "validation": 4, "test": 4},
        )
        self.assertEqual(
            audit["context_stratum_counts"],
            self.settings["context_split"]["strata"],
        )

        expected_one_fields = {
            "train": dict(zip(FIELD_NAMES, (209, 209, 208, 208, 208, 208))),
            "validation": {name: 26 for name in FIELD_NAMES},
            "test": {name: 26 for name in FIELD_NAMES},
        }
        expected_list_fields = {
            "train": {name: 125 for name in FIELD_NAMES},
            "validation": dict(zip(FIELD_NAMES, (16, 16, 16, 16, 15, 15))),
            "test": dict(zip(FIELD_NAMES, (16, 16, 16, 16, 15, 15))),
        }
        for split in ("train", "validation", "test"):
            self.assertEqual(
                audit["field_counts"][split]["one_field"],
                expected_one_fields[split],
            )
            self.assertEqual(
                audit["field_counts"][split]["list_filter"],
                expected_list_fields[split],
            )
            row_count = sum(EXPECTED_FORM_COUNTS[split].values())
            structure = audit["prompt_structure_audit"]["splits"][split]
            self.assertEqual(structure["harness_message_count"], 0)
            self.assertEqual(structure["no_tool_call_count"], row_count)
            self.assertEqual(
                structure["direct_one_tag_count"],
                sum(EXPECTED_FORM_COUNTS[split][form] for form in (1, 3, 7)),
            )
            self.assertEqual(
                structure["list_three_tag_count"],
                EXPECTED_FORM_COUNTS[split][4],
            )
            self.assertTrue(structure["pass"])
            malformed_coverage = audit["prompt_structure_audit"][
                "malformed_input_coverage"
            ][split]
            self.assertEqual(
                malformed_coverage["row_count"], EXPECTED_FORM_COUNTS[split][6]
            )
            self.assertEqual(
                malformed_coverage["unique_input_count"],
                EXPECTED_FORM_COUNTS[split][6],
            )
            self.assertTrue(malformed_coverage["pass"])
            available_prefixes = (
                {"HumanNetV3", "TFs", "bulkPEN", "scPEN"}
                if split == "train"
                else {"HumanNetV3", "bulkPEN", "scPEN"}
            )
            for subtype, values in malformed_coverage["subtypes"].items():
                expected_prefixes = (
                    available_prefixes & {"bulkPEN", "scPEN"}
                    if subtype == "missing_path"
                    else available_prefixes
                )
                self.assertEqual(
                    values["unique_input_count"], values["row_count"]
                )
                self.assertEqual(
                    set(values["base_prefix_counts"]), expected_prefixes
                )
                self.assertEqual(
                    values["expected_base_prefixes"], sorted(expected_prefixes)
                )
                self.assertEqual(
                    sum(values["base_prefix_counts"].values()),
                    values["row_count"],
                )
                self.assertTrue(values["pass"])
            for form, count in EXPECTED_FORM_COUNTS[split].items():
                gate = audit["per_form_gates"]["splits"][split][
                    f"S1.6.{form}"
                ]
                self.assertEqual(gate["row_count"], count)
                self.assertTrue(gate["pass"])
                if count:
                    templates = audit["prompt_template_counts"][split][
                        f"S1.6.{form}"
                    ]
                    self.assertEqual(set(templates), {"direct", "step"})
                    for framing in ("direct", "step"):
                        self.assertEqual(
                            set(templates[framing]), {"0", "1", "2", "3"}
                        )
                        self.assertTrue(
                            all(value > 0 for value in templates[framing].values())
                        )
                    self.assertEqual(
                        sum(
                            value
                            for values in templates.values()
                            for value in values.values()
                        ),
                        count,
                    )

        self.assertEqual(
            audit["filter_counts"],
            {
                "train": {"match": 375, "no_match": 375},
                "validation": {"match": 47, "no_match": 47},
                "test": {"match": 47, "no_match": 47},
            },
        )
        self.assertEqual(
            audit["per_form_gates"]["no_inference_violation_count"], 0
        )
        self.assertEqual(
            audit["per_form_gates"]["false_defer_audit"],
            {
                "eligible_row_count": 5000,
                "false_defer_count": 0,
                "false_defer_rate": 0.0,
                "pass": True,
            },
        )
        self.assertTrue(audit["per_form_gates"]["pass"])
        for split, field_gate in audit["per_form_gates"][
            "heldout_field_accuracy"
        ].items():
            self.assertEqual(field_gate["accuracy"], 1.0, split)
            self.assertTrue(field_gate["pass"])
        for matrix in audit["per_form_gates"][
            "family_confusion_matrix"
        ].values():
            for expected, observed in matrix.items():
                self.assertEqual(set(observed), {expected})
        self.assertTrue(audit["prompt_structure_audit"]["pass"])
        full = audit["prompt_structure_audit"][
            "full_visible_prompt_classifier"
        ]
        self.assertEqual(full["scope"], "full_visible_prompt")
        self.assertTrue(full["diagnostic_only"])
        self.assertIsNone(full["pass"])
        for metrics in full["splits"].values():
            self.assertFalse(metrics["gate_applied"])
            self.assertIsNone(metrics["pass"])
        prose_audit = audit["prompt_structure_audit"]["prose_length_audit"]
        self.assertTrue(prose_audit["pass"])
        for prose in prose_audit["splits"].values():
            self.assertLessEqual(prose["maximum_instruction_words"], 20)
            self.assertLessEqual(prose["maximum_description_words"], 25)
            self.assertEqual(prose["violation_count"], 0)
            self.assertTrue(prose["pass"])
        self.assertTrue(audit["masked_wording_classifier"]["pass"])
        self.assertTrue(audit["thinking_filter"]["pass"])
        self.assertTrue(audit["duplicate_prompt_tool_outcome"]["pass"])

    def test_duplicate_audit_rejects_an_ordinary_copy(self) -> None:
        scenario = self.scenarios["train"][0]
        row = deepcopy(self.rows["train"][0])
        duplicate_scenario = replace(scenario, occurrence=100_000)
        duplicate = deepcopy(row)
        duplicate["row_id"] = "s1_6_duplicate"
        duplicate["messages"][-1]["content"]["summary"] = (
            "Different final wording with the same parsed result."
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


class WorldModelS16GeneratedCorpusTests(unittest.TestCase):
    def test_build_is_deterministic_and_validates_before_publication(self) -> None:
        if not SOURCE_INDEX.is_file():
            self.skipTest(f"canonical source index is absent: {SOURCE_INDEX}")
        contract = load_contract()
        real_validate = s1_6_module._validate_corpus
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            first_root = temporary / "first" / "corpus"
            second_root = temporary / "second" / "corpus"
            active_root = [first_root]
            validated_names: list[str] = []

            def validate_before_publish(path: Path) -> None:
                self.assertFalse((active_root[0] / "manifest.json").exists())
                validated_names.append(path.name)
                real_validate(path)

            with patch.object(
                s1_6_module,
                "_validate_corpus",
                side_effect=validate_before_publish,
            ):
                first = build_corpus(
                    SOURCE_INDEX,
                    first_root,
                    contract=contract,
                    validate=True,
                )
                active_root[0] = second_root
                second = build_corpus(
                    SOURCE_INDEX,
                    second_root,
                    contract=contract,
                    validate=True,
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
            published = Path(
                contract["curriculum"]["families"]["S1.6"]["output_directory"]
            )
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    (first_root / f"{split}.jsonl").read_bytes(),
                    (published / f"{split}.jsonl").read_bytes(),
                )
            for file_name in (
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
                "manifest.json",
            ):
                first_path = first_root / file_name
                second_path = second_root / file_name
                self.assertTrue(first_path.is_file())
                self.assertTrue(second_path.is_file())
                self.assertEqual(
                    first_path.read_bytes(), second_path.read_bytes()
                )
            for file_name, values in first["files"].items():
                self.assertEqual(
                    file_sha256(first_root / file_name), values["sha256"]
                )
            first_manifest = json.loads(
                (first_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first_manifest, first)
            for name in ("contract", "builder"):
                source = first_manifest["sources"][name]
                self.assertEqual(
                    source["sha256"], file_sha256(Path(source["path"]))
                )

    def test_generated_manifest_and_rows_reconcile_when_present(self) -> None:
        # The current builders publish S1 v3, which trains on the pinned
        # prompt bank; the v1 directory predates prompt-bank support.
        contract, overlay = apply_s1_overlay(load_contract(), S1_V3_OVERLAY_PATH)
        settings = contract["curriculum"]["families"]["S1.6"]
        prompts = wrapper_prompts(
            overlay["prompt_banks"], LAYER_PROMPTS, bank_name="layer"
        )
        root = Path(settings["output_directory"])
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"generated corpus is absent: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "corpus_schema_version",
                "section",
                "dataset_id",
                "seed",
                "scenario_assignment_sha256",
                "prompt_templates_sha256",
                "row_counts",
                "form_counts",
                "semantic_bucket_counts",
                "status_counts",
                "tool_call_counts",
                "framing_counts",
                "prompt_template_counts",
                "field_counts",
                "filter_counts",
                "module_namespace_counts",
                "malformed_subtype_counts",
                "context_counts",
                "context_stratum_counts",
                "sources",
                "source_pool_coverage",
                "per_form_gates",
                "prompt_structure_audit",
                "masked_wording_classifier",
                "thinking_filter",
                "duplicate_prompt_tool_outcome",
                "evidence",
                "files",
                "s1_overlay",
                "prompt_bank",
            },
        )
        self.assertEqual(
            manifest["schema_version"],
            "mentor-rl-world-model-s1-6-manifest-v1",
        )
        self.assertEqual(
            manifest["corpus_schema_version"],
            "mentor-rl-world-model-s1-6-corpus-v1",
        )
        self.assertEqual(manifest["section"], "S1.6")
        self.assertEqual(manifest["dataset_id"], settings["dataset_id"])
        self.assertEqual(manifest["seed"], settings["assignment_seed"])
        self.assertEqual(
            manifest["prompt_templates_sha256"],
            hashlib.sha256(
                json.dumps(
                    prompts,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(
            set(manifest["scenario_assignment_sha256"]),
            {"train", "validation", "test"},
        )
        self.assertTrue(
            all(
                re.fullmatch(r"[0-9a-f]{64}", value)
                for value in manifest["scenario_assignment_sha256"].values()
            )
        )
        expected_rows = {"train": 5000, "validation": 625, "test": 625}
        self.assertEqual(manifest["row_counts"], expected_rows)
        self.assertEqual(
            manifest["form_counts"],
            {
                split: {
                    f"S1.6.{form}": count
                    for form, count in counts.items()
                }
                for split, counts in EXPECTED_FORM_COUNTS.items()
            },
        )
        self.assertEqual(
            manifest["status_counts"],
            {
                "train": {"invalid_input": 1000, "resolved": 4000},
                "validation": {"invalid_input": 125, "resolved": 500},
                "test": {"invalid_input": 125, "resolved": 500},
            },
        )
        self.assertEqual(
            manifest["tool_call_counts"],
            {
                "train": {"0": 5000},
                "validation": {"0": 625},
                "test": {"0": 625},
            },
        )
        self.assertEqual(
            manifest["context_counts"],
            settings["context_split"]["counts"],
        )
        self.assertEqual(
            manifest["context_stratum_counts"],
            settings["context_split"]["strata"],
        )

        expected_one_fields = {
            "train": dict(zip(FIELD_NAMES, (209, 209, 208, 208, 208, 208))),
            "validation": {name: 26 for name in FIELD_NAMES},
            "test": {name: 26 for name in FIELD_NAMES},
        }
        expected_list_fields = {
            "train": {name: 125 for name in FIELD_NAMES},
            "validation": dict(zip(FIELD_NAMES, (16, 16, 16, 16, 15, 15))),
            "test": dict(zip(FIELD_NAMES, (16, 16, 16, 16, 15, 15))),
        }
        expected_modules = {
            "train": {"rwr_loe": 250, "mentor_ev": 250},
            "validation": {"rwr_loe": 32, "mentor_ev": 31},
            "test": {"rwr_loe": 32, "mentor_ev": 31},
        }
        expected_malformed = {
            "train": dict(zip(MALFORMED_SUBTYPES, (84, 84, 83, 83, 83, 83))),
            "validation": dict(zip(MALFORMED_SUBTYPES, (11, 11, 10, 10, 10, 10))),
            "test": dict(zip(MALFORMED_SUBTYPES, (11, 11, 10, 10, 10, 10))),
        }
        for split in ("train", "validation", "test"):
            self.assertEqual(
                manifest["field_counts"][split],
                {
                    "one_field": expected_one_fields[split],
                    "list_filter": expected_list_fields[split],
                },
            )
        self.assertEqual(
            manifest["filter_counts"],
            {
                "train": {"match": 375, "no_match": 375},
                "validation": {"match": 47, "no_match": 47},
                "test": {"match": 47, "no_match": 47},
            },
        )
        self.assertEqual(manifest["module_namespace_counts"], expected_modules)
        self.assertEqual(
            manifest["malformed_subtype_counts"], expected_malformed
        )
        for split, counts in EXPECTED_FORM_COUNTS.items():
            templates = manifest["prompt_template_counts"][split]
            self.assertEqual(
                set(templates),
                {
                    f"S1.6.{form}"
                    for form, count in counts.items()
                    if count
                },
            )
            for form, count in counts.items():
                if not count:
                    continue
                by_frame = templates[f"S1.6.{form}"]
                self.assertEqual(set(by_frame), {"direct", "step"})
                for framing, values in by_frame.items():
                    self.assertEqual(
                        set(values),
                        {str(index) for index in range(len(prompts[split][framing]))},
                    )
                    self.assertTrue(all(value > 0 for value in values.values()))
                self.assertEqual(
                    sum(
                        value
                        for values in by_frame.values()
                        for value in values.values()
                    ),
                    count,
                )

        coverage = manifest["source_pool_coverage"]
        self.assertEqual(
            set(coverage),
            {
                "layer_pool_counts",
                "context_counts",
                "context_stratum_counts",
                "native_layer_count",
                "split_context_sha256",
            },
        )
        self.assertEqual(coverage["layer_pool_counts"], EXPECTED_POOL_COUNTS)
        self.assertEqual(coverage["context_counts"], manifest["context_counts"])
        self.assertEqual(
            coverage["context_stratum_counts"],
            manifest["context_stratum_counts"],
        )
        self.assertEqual(coverage["native_layer_count"], 353)
        self.assertEqual(
            set(coverage["split_context_sha256"]),
            {"train", "validation", "test"},
        )
        self.assertTrue(
            all(
                re.fullmatch(r"[0-9a-f]{64}", value)
                for value in coverage["split_context_sha256"].values()
            )
        )

        sources = manifest["sources"]
        self.assertEqual(
            set(sources),
            {
                "source_index",
                "multiplex_flist",
                "contract",
                "builder",
                "native_layer_policy",
                "records",
            },
        )
        source_payload = check_source_index(SOURCE_INDEX, contract)
        self.assertEqual(sources["records"], source_payload["sources"])
        self.assertEqual(
            sources["native_layer_policy"],
            "parse_only_the_353_pinned_flist_tags",
        )
        self.assertEqual(
            sources["source_index"]["sha256"], file_sha256(SOURCE_INDEX)
        )
        self.assertEqual(
            sources["source_index"]["content_sha256"],
            contract["source_index"]["content_sha256"],
        )
        flist_path = Path(contract["coordinate_system"]["source_flist_path"])
        self.assertEqual(
            sources["multiplex_flist"],
            {
                "file_name": flist_path.name,
                "sha256": contract["coordinate_system"]["source_flist_sha256"],
            },
        )
        contract_source = sources["contract"]
        self.assertEqual(set(contract_source), {"path", "sha256"})
        self.assertEqual(
            Path(contract_source["path"]),
            Path("runtime/world_model/contract.yaml"),
        )
        self.assertRegex(contract_source["sha256"], r"[0-9a-f]{64}")

        builder_source = sources["builder"]
        self.assertEqual(set(builder_source), {"path", "sha256"})
        builder_path = Path(builder_source["path"])
        self.assertEqual(
            builder_path, Path("runtime/world_model/s1_6_corpus.py")
        )
        self.assertTrue(builder_path.is_file())
        # This manifest records the historical builder. Fresh builds above
        # check the current builder pin and reproduce every published row.
        self.assertRegex(builder_source["sha256"], r"^[0-9a-f]{64}$")

        self.assertEqual(
            manifest["evidence"],
            {
                "call_fields": ["evidence_id", "row_id", "producer"],
                "calls": [],
                "call_count": 0,
            },
        )
        self.assertTrue(manifest["per_form_gates"]["pass"])
        self.assertEqual(
            manifest["per_form_gates"]["no_inference_violation_count"], 0
        )
        self.assertEqual(
            manifest["per_form_gates"]["false_defer_audit"][
                "false_defer_count"
            ],
            0,
        )
        self.assertTrue(manifest["prompt_structure_audit"]["pass"])
        prose_audit = manifest["prompt_structure_audit"][
            "prose_length_audit"
        ]
        self.assertTrue(prose_audit["pass"])
        for prose in prose_audit["splits"].values():
            self.assertLessEqual(prose["maximum_instruction_words"], 20)
            self.assertLessEqual(prose["maximum_description_words"], 25)
            self.assertEqual(prose["violation_count"], 0)
            self.assertTrue(prose["pass"])
        for split, coverage_values in manifest["prompt_structure_audit"][
            "malformed_input_coverage"
        ].items():
            self.assertEqual(
                coverage_values["unique_input_count"],
                EXPECTED_FORM_COUNTS[split][6],
            )
            self.assertTrue(coverage_values["pass"])
            for subtype_values in coverage_values["subtypes"].values():
                self.assertEqual(
                    subtype_values["unique_input_count"],
                    subtype_values["row_count"],
                )
                self.assertEqual(
                    set(subtype_values["base_prefix_counts"]),
                    set(subtype_values["expected_base_prefixes"]),
                )
                self.assertTrue(subtype_values["pass"])
        self.assertTrue(manifest["masked_wording_classifier"]["pass"])
        self.assertTrue(manifest["thinking_filter"]["pass"])
        self.assertTrue(manifest["duplicate_prompt_tool_outcome"]["pass"])

        all_row_ids: set[str] = set()
        units_by_split: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            path = root / f"{split}.jsonl"
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), expected_rows[split])
            self.assertEqual(
                manifest["files"][path.name],
                {"row_count": expected_rows[split], "sha256": file_sha256(path)},
            )
            self.assertEqual(
                Counter(row["metadata"]["form"] for row in rows),
                Counter(EXPECTED_FORM_COUNTS[split]),
            )
            units_by_split[split] = {
                row["metadata"]["split_unit"] for row in rows
            }
            for row in rows:
                self.assertNotIn(row["row_id"], all_row_ids)
                all_row_ids.add(row["row_id"])
                self.assertEqual(_calls(row), [])
                self.assertFalse(
                    any(message["role"] == "tool" for message in row["messages"])
                )
                self.assertIsNone(row["messages"][-1]["content"]["provenance"])
                user = row["messages"][1]["content"]
                self.assertLessEqual(max(_instruction_word_counts(user)), 20)
                self.assertEqual(
                    sum(
                        _matches_template(user, template)
                        for templates in prompts[split].values()
                        for template in templates
                    ),
                    1,
                )
                self.assertFalse(
                    any(
                        _matches_template(user, template)
                        for other, framings in prompts.items()
                        if other != split
                        for templates in framings.values()
                        for template in templates
                    )
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
