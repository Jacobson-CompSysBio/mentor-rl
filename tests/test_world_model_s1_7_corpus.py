"""Focused tests for the canonical S1.7 layer-family corpus builder."""

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

import runtime.world_model.s1_7_corpus as s1_7_module

from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s1_7_corpus import (
    CLAIM_PARAPHRASES,
    CLAIM_VOCABULARY,
    FAMILY_CARDS,
    FAMILY_PROMPTS,
    CorpusBuildError,
    Scenario,
    audit_rows,
    build_corpus,
    build_fact_pools,
    build_row,
    duplicate_prompt_outcome_audit,
    family_card,
    filter_row_thinking,
    masked_wording_classifier_audit,
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


FORM_BY_BUCKET = {
    "scpen_family": 1,
    "list_filter": 2,
    "coexpression_claims": 3,
    "claim_check": 4,
    "wrong_input_type": 5,
    "ppi_claims": 6,
    "bulk_pen_claims": 7,
    "other_claims": 8,
    "tf_target_claims": 9,
}
EXPECTED_TRACE_COUNTS = {
    "train": {1: 1500, 2: 900, 3: 375, 4: 1500, 5: 300,
              6: 375, 7: 375, 8: 300, 9: 375},
    "validation": {1: 188, 2: 113, 3: 47, 4: 187, 5: 37,
                   6: 47, 7: 47, 8: 37, 9: 47},
    "test": {1: 188, 2: 113, 3: 47, 4: 187, 5: 37,
             6: 47, 7: 47, 8: 37, 9: 47},
}
EXPECTED_POOL_COUNTS = {
    "train": {
        "all": 261,
        "ppi": 1,
        "coexpression": 1,
        "tf_target": 1,
        "bulk_pen": 10,
        "scpen": 245,
        "other": 3,
        "modules": 12_543,
    },
    "validation": {
        "all": 53,
        "ppi": 0,
        "coexpression": 0,
        "tf_target": 0,
        "bulk_pen": 2,
        "scpen": 50,
        "other": 1,
        "modules": 1_568,
    },
    "test": {
        "all": 39,
        "ppi": 0,
        "coexpression": 0,
        "tf_target": 0,
        "bulk_pen": 2,
        "scpen": 36,
        "other": 1,
        "modules": 1_568,
    },
}
EXPECTED_FAMILY_CARDS = {
    "ppi": ("physical_interaction_recorded", "physical_interaction_recorded"),
    "coexpression": ("global_coexpression", "global_coexpression"),
    "tf_target": ("regulatory_link_recorded", "regulatory_link_recorded"),
    "bulk_pen": (
        "context_predictive_expression",
        "context_predictive_expression",
    ),
    "scpen": (
        "context_predictive_expression",
        "context_predictive_expression",
    ),
    "other": (None, None),
}
TRACE_FAMILY = {
    1: "scpen",
    3: "coexpression",
    6: "ppi",
    7: "bulk_pen",
    8: "other",
    9: "tf_target",
}


def _source_index(contract: dict[str, object]) -> Path:
    digest = contract["source_index"]["content_sha256"]
    return Path(
        "data/world_model_v3/source_indexes/"
        f"world_model_source_index_{digest}.json.gz"
    )


def _calls(row: dict[str, object]) -> list[dict[str, object]]:
    return [
        message["tool_call"]
        for message in row["messages"]
        if message["role"] == "assistant" and "tool_call" in message
    ]


def _answer(row: dict[str, object]) -> dict[str, object]:
    return row["messages"][-1]["content"]


def _matches_template(text: str, template: str) -> bool:
    prefix, suffix = template.split("{request}")
    return text.startswith(prefix) and text.endswith(suffix)


def _native_tags(text: str) -> list[str]:
    return re.findall(
        r"(?:HumanNetV3|TFs|bulkPEN|scPEN)"
        r":[A-Za-z0-9_-]+(?:\:[A-Za-z0-9_-]+)*",
        text,
    )


def _expected_card(family: str, layer_id: str | None) -> dict[str, object]:
    evidence, supported = EXPECTED_FAMILY_CARDS[family]
    supported_claims = [] if supported is None else [supported]
    return {
        "layer_id": layer_id,
        "layer_family": family,
        "evidence_type": evidence,
        "supported_claims": supported_claims,
        "unsupported_claims": [
            claim for claim in CLAIM_VOCABULARY if claim not in supported_claims
        ],
    }


def _expected_unsupported_counts(
    total: int, family: str
) -> dict[str, int]:
    supported = set(_expected_card(family, None)["supported_claims"])
    candidates = [claim for claim in CLAIM_VOCABULARY if claim not in supported]
    quotient, remainder = divmod(total, len(candidates))
    return {
        claim: quotient + (index < remainder)
        for index, claim in enumerate(candidates)
    }


def _largest_remainder_subset(
    counts: dict[str, int], target: int
) -> dict[str, int]:
    total = sum(counts.values())
    result = {
        claim: count * target // total for claim, count in counts.items()
    }
    remaining = target - sum(result.values())
    order = {claim: index for index, claim in enumerate(CLAIM_VOCABULARY)}
    ranked = sorted(
        counts,
        key=lambda claim: (-(counts[claim] * target % total), order[claim]),
    )
    for claim in ranked[:remaining]:
        result[claim] += 1
    return result


class _Tokenizer:
    def __init__(self) -> None:
        self.rendered_messages: list[list[dict[str, object]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, object]], **options: object
    ) -> list[int]:
        assert all(
            "description" in tool["function"] for tool in options["tools"]
        )
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


class WorldModelS17ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.curriculum = self.contract["curriculum"]
        self.settings = self.curriculum["families"]["S1.7"]

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
                "claim_vocabulary",
                "family_cards",
                "family_name_policy",
                "list_filter",
                "claim_cards",
                "claim_check",
                "wrong_input_type",
                "other_network",
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
                "dataset_id": "world_model_v3_s1_7_layer_family_claims",
                "question_family": "layer_family_claims",
                "book_mode": "closed_book",
                "output_directory": "data/world_model_v3/corpora/s1/s1_7",
                "assignment_seed": 1729,
                "train_rows": 6000,
            },
        )
        self.assertEqual(
            self.settings["allocations"],
            {
                name: EXPECTED_TRACE_COUNTS["train"][form]
                for name, form in FORM_BY_BUCKET.items()
            },
        )
        self.assertEqual(
            self.settings["evaluation_allocations"],
            {
                name: EXPECTED_TRACE_COUNTS["validation"][form]
                for name, form in FORM_BY_BUCKET.items()
            },
        )
        self.assertEqual(
            self.settings["semantic_buckets"],
            {
                "family_of_layer": ["scpen_family"],
                "list_filter": ["list_filter"],
                "claims_card": [
                    "coexpression_claims",
                    "ppi_claims",
                    "bulk_pen_claims",
                    "tf_target_claims",
                ],
                "claim_check": ["claim_check"],
                "wrong_input_type": ["wrong_input_type"],
                "native_nonmapped_network": ["other_claims"],
            },
        )
        self.assertEqual(self.settings["claim_vocabulary"], list(CLAIM_VOCABULARY))
        self.assertEqual(
            self.settings["family_cards"],
            {
                family: {
                    "evidence_type": card["evidence_type"],
                    "supported_claims": list(card["supported_claims"]),
                }
                for family, card in FAMILY_CARDS.items()
            },
        )
        self.assertEqual(
            self.settings["family_name_policy"],
            {"shared_closed_vocabulary": True, "result_layer_id": None},
        )

    def test_split_and_trace_allocations_are_exact(self) -> None:
        row_counts = split_row_counts(
            self.settings["train_rows"], self.curriculum["split_fractions"]
        )
        self.assertEqual(
            row_counts, {"train": 6000, "validation": 750, "test": 750}
        )
        self.assertEqual(
            {
                split: trace_counts(total, self.settings)
                for split, total in row_counts.items()
            },
            EXPECTED_TRACE_COUNTS,
        )
        for split, counts in EXPECTED_TRACE_COUNTS.items():
            self.assertEqual(sum(counts.values()), row_counts[split])

    def test_all_six_family_cards_have_exact_ordered_partitions(self) -> None:
        self.assertEqual(
            list(FAMILY_CARDS),
            ["ppi", "coexpression", "tf_target", "bulk_pen", "scpen", "other"],
        )
        self.assertEqual(
            list(CLAIM_VOCABULARY),
            [
                "context_predictive_expression",
                "physical_interaction_recorded",
                "regulatory_link_recorded",
                "regulation_direction",
                "global_coexpression",
                "cofunctional_association",
                "holds_in_other_contexts",
                "causal_effect",
            ],
        )
        for family in FAMILY_CARDS:
            with self.subTest(family=family):
                expected = _expected_card(family, None)
                actual = family_card(family, None)
                self.assertEqual(actual, expected)
                self.assertEqual(
                    list(actual),
                    [
                        "layer_id",
                        "layer_family",
                        "evidence_type",
                        "supported_claims",
                        "unsupported_claims",
                    ],
                )
                self.assertEqual(
                    set(actual["supported_claims"])
                    | set(actual["unsupported_claims"]),
                    set(CLAIM_VOCABULARY),
                )
                self.assertTrue(
                    set(actual["supported_claims"]).isdisjoint(
                        actual["unsupported_claims"]
                    )
                )
                for key in ("supported_claims", "unsupported_claims"):
                    selected = set(actual[key])
                    self.assertEqual(
                        actual[key],
                        [claim for claim in CLAIM_VOCABULARY if claim in selected],
                    )

    def test_prompt_wrappers_and_claim_paraphrases_are_held_out(self) -> None:
        self.assertEqual(set(FAMILY_PROMPTS), {"train", "validation", "test"})
        wrappers: dict[str, set[str]] = {}
        paraphrases: dict[str, set[str]] = {}
        for split in FAMILY_PROMPTS:
            self.assertEqual(set(FAMILY_PROMPTS[split]), {"direct", "step"})
            wrappers[split] = {
                template
                for values in FAMILY_PROMPTS[split].values()
                for template in values
            }
            self.assertTrue(wrappers[split])
            self.assertTrue(
                all(template.count("{request}") == 1 for template in wrappers[split])
            )
            self.assertEqual(set(CLAIM_PARAPHRASES[split]), set(CLAIM_VOCABULARY))
            paraphrases[split] = {
                phrase
                for values in CLAIM_PARAPHRASES[split].values()
                for phrase in values
            }
            generic_carrier = re.compile(
                r"^(?:evidence|support|a finding|a conclusion|a claim|"
                r"a report|confirmation|an assertion)\b",
                re.IGNORECASE,
            )
            self.assertTrue(
                all(
                    generic_carrier.search(phrase) is None
                    for phrase in paraphrases[split]
                )
            )
            forbidden_context_words = re.compile(
                r"\b(?:specific|limited|bounded|restricted|confined|only|tissue)\b",
                re.IGNORECASE,
            )
            self.assertTrue(
                all(
                    forbidden_context_words.search(phrase) is None
                    for phrase in CLAIM_PARAPHRASES[split][
                        "context_predictive_expression"
                    ]
                )
            )
            self.assertTrue(
                all(
                    re.search(r"\bpredict\w*", phrase, re.IGNORECASE) is not None
                    and re.search(r"\bexpression\b", phrase, re.IGNORECASE)
                    is not None
                    for phrase in CLAIM_PARAPHRASES[split][
                        "context_predictive_expression"
                    ]
                )
            )
            self.assertTrue(
                all(
                    re.search(
                        r"\b(?:coexpression|correlat\w*)\b",
                        phrase,
                        re.IGNORECASE,
                    )
                    is not None
                    for phrase in CLAIM_PARAPHRASES[split][
                        "global_coexpression"
                    ]
                )
            )
            self.assertTrue(
                all(CLAIM_PARAPHRASES[split][claim] for claim in CLAIM_VOCABULARY)
            )
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            self.assertTrue(wrappers[first].isdisjoint(wrappers[second]))
            self.assertTrue(paraphrases[first].isdisjoint(paraphrases[second]))


class WorldModelS17SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.7"]
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

    def test_source_pools_cover_all_families_and_only_mentor_modules(self) -> None:
        self.assertEqual(set(self.pools), {"train", "validation", "test"})
        for split, pools in self.pools.items():
            with self.subTest(split=split):
                self.assertEqual(set(pools), set(EXPECTED_POOL_COUNTS[split]))
                self.assertEqual(
                    {name: len(values) for name, values in pools.items()},
                    EXPECTED_POOL_COUNTS[split],
                )
                self.assertEqual(
                    Counter(fact.layer_family for fact in pools["all"]),
                    Counter(
                        {
                            family: count
                            for family, count in EXPECTED_POOL_COUNTS[split].items()
                            if family not in {"all", "modules"} and count
                        }
                    ),
                )
                self.assertEqual(
                    {fact.namespace for fact in pools["modules"]}, {"mentor_ev"}
                )
                for family in FAMILY_CARDS:
                    self.assertTrue(
                        all(fact.layer_family == family for fact in pools[family])
                    )

        native_other = {
            fact.layer_id
            for pools in self.pools.values()
            for fact in pools["other"]
        }
        self.assertEqual(
            native_other,
            {
                "HumanNetV3:cocitation",
                "HumanNetV3:database",
                "HumanNetV3:gene_interaction",
                "HumanNetV3:gene_neighbor",
                "HumanNetV3:phylogenetic",
            },
        )
        all_positive_ids = {
            fact.layer_id
            for pools in self.pools.values()
            for fact in pools["all"]
        }
        self.assertNotIn("HumanNetV3:biogrid_ppi", all_positive_ids)
        module_sets = {
            split: {fact.module_id for fact in pools["modules"]}
            for split, pools in self.pools.items()
        }
        self.assertTrue(module_sets["train"].isdisjoint(module_sets["validation"]))
        self.assertTrue(module_sets["train"].isdisjoint(module_sets["test"]))
        self.assertTrue(module_sets["validation"].isdisjoint(module_sets["test"]))

        context_sets = {
            split: {fact.context_key for fact in pools["all"]}
            for split, pools in self.pools.items()
        }
        self.assertEqual(
            {split: len(values) for split, values in context_sets.items()},
            self.settings["context_split"]["counts"],
        )
        self.assertTrue(context_sets["train"].isdisjoint(context_sets["validation"]))
        self.assertTrue(context_sets["train"].isdisjoint(context_sets["test"]))
        self.assertTrue(context_sets["validation"].isdisjoint(context_sets["test"]))

    def test_scenario_counts_and_trace_families_are_exact(self) -> None:
        expected_input_kinds = {
            "train": {
                "layer_tag": 4200,
                "layer_list": 900,
                "family_name": 600,
                "module_id": 300,
            },
            "validation": {
                "layer_tag": 365,
                "layer_list": 113,
                "family_name": 235,
                "module_id": 37,
            },
            "test": {
                "layer_tag": 365,
                "layer_list": 113,
                "family_name": 235,
                "module_id": 37,
            },
        }
        for split, scenarios in self.scenarios.items():
            with self.subTest(split=split):
                self.assertEqual(
                    Counter(item.form for item in scenarios),
                    Counter(EXPECTED_TRACE_COUNTS[split]),
                )
                self.assertEqual(
                    Counter(item.source_trace for item in scenarios),
                    Counter(
                        {
                            f"S1.7.{form}": count
                            for form, count in EXPECTED_TRACE_COUNTS[split].items()
                        }
                    ),
                )
                self.assertEqual(
                    Counter(item.input_kind for item in scenarios),
                    Counter(expected_input_kinds[split]),
                )
                self.assertEqual(len({item.scenario_id for item in scenarios}), len(scenarios))
                for form, family in TRACE_FAMILY.items():
                    self.assertEqual(
                        {item.family for item in scenarios if item.form == form},
                        {family},
                    )

    def test_family_name_inputs_follow_exact_train_and_eval_policy(self) -> None:
        expected = {
            "train": {
                (3, "coexpression"): 75,
                (4, "ppi"): 50,
                (4, "coexpression"): 50,
                (4, "tf_target"): 50,
                (4, "bulk_pen"): 50,
                (4, "scpen"): 50,
                (4, "other"): 50,
                (6, "ppi"): 75,
                (7, "bulk_pen"): 75,
                (9, "tf_target"): 75,
            },
            "validation": {
                (3, "coexpression"): 47,
                (4, "ppi"): 32,
                (4, "coexpression"): 31,
                (4, "tf_target"): 31,
                (6, "ppi"): 47,
                (9, "tf_target"): 47,
            },
            "test": {
                (3, "coexpression"): 47,
                (4, "ppi"): 32,
                (4, "coexpression"): 31,
                (4, "tf_target"): 31,
                (6, "ppi"): 47,
                (9, "tf_target"): 47,
            },
        }
        for split, scenarios in self.scenarios.items():
            counts = Counter(
                (item.form, item.family)
                for item in scenarios
                if item.input_kind == "family_name"
            )
            self.assertEqual(counts, Counter(expected[split]))
            for item in scenarios:
                if item.input_kind == "family_name":
                    self.assertIsNone(item.fact)
                    self.assertIn(item.family, FAMILY_CARDS)

        for split in ("validation", "test"):
            by_form = {
                form: [item for item in self.scenarios[split] if item.form == form]
                for form in (3, 6, 7, 9)
            }
            for form in (3, 6, 9):
                self.assertTrue(
                    all(item.input_kind == "family_name" for item in by_form[form])
                )
            self.assertTrue(
                all(
                    item.input_kind == "layer_tag"
                    and item.fact.layer_id.startswith("bulkPEN:")
                    for item in by_form[7]
                )
            )

    def test_list_rows_preserve_input_partition_and_exact_order(self) -> None:
        expected_families = {
            "train": {family: 150 for family in FAMILY_CARDS},
            "validation": {"bulk_pen": 38, "scpen": 38, "other": 37},
            "test": {"bulk_pen": 38, "scpen": 38, "other": 37},
        }
        expected_outcomes = {
            "train": {"match": 450, "no_match": 450},
            "validation": {"match": 57, "no_match": 56},
            "test": {"match": 57, "no_match": 56},
        }
        expected_cross = {
            "train": {
                (family, outcome): 75
                for family in FAMILY_CARDS
                for outcome in ("match", "no_match")
            },
            "validation": {
                ("bulk_pen", "match"): 19,
                ("bulk_pen", "no_match"): 19,
                ("scpen", "match"): 19,
                ("scpen", "no_match"): 19,
                ("other", "match"): 19,
                ("other", "no_match"): 18,
            },
            "test": {
                ("bulk_pen", "match"): 19,
                ("bulk_pen", "no_match"): 19,
                ("scpen", "match"): 19,
                ("scpen", "no_match"): 19,
                ("other", "match"): 19,
                ("other", "no_match"): 18,
            },
        }
        expected_outcome_frames = {
            "train": {
                ("match", "direct"): 225,
                ("match", "step"): 225,
                ("no_match", "direct"): 225,
                ("no_match", "step"): 225,
            },
            "validation": {
                ("match", "direct"): 29,
                ("match", "step"): 28,
                ("no_match", "direct"): 28,
                ("no_match", "step"): 28,
            },
            "test": {
                ("match", "direct"): 29,
                ("match", "step"): 28,
                ("no_match", "direct"): 28,
                ("no_match", "step"): 28,
            },
        }
        for split, scenarios in self.scenarios.items():
            rows = [item for item in scenarios if item.form == 2]
            self.assertTrue(all(item.input_kind == "layer_list" for item in rows))
            self.assertEqual(
                Counter(item.family for item in rows),
                Counter(expected_families[split]),
            )
            self.assertEqual(
                Counter(item.list_outcome for item in rows),
                Counter(expected_outcomes[split]),
            )
            self.assertEqual(
                Counter((item.family, item.list_outcome) for item in rows),
                Counter(expected_cross[split]),
            )
            self.assertEqual(
                Counter((item.list_outcome, item.framing) for item in rows),
                Counter(expected_outcome_frames[split]),
            )
            for family, outcome in expected_cross[split]:
                frames = {
                    item.framing
                    for item in rows
                    if item.family == family and item.list_outcome == outcome
                }
                self.assertEqual(frames, {"direct", "step"})
            for family in expected_families[split]:
                positions = {
                    index
                    for item in rows
                    if item.family == family and item.list_outcome == "match"
                    for index, fact in enumerate(item.companions)
                    if fact.layer_family == family
                }
                self.assertEqual(positions, {0, 1, 2}, (split, family))
            for item in rows:
                self.assertEqual(len(item.companions), 3)
                self.assertEqual(
                    len({fact.layer_id for fact in item.companions}), 3
                )
                matching = tuple(
                    fact.layer_id
                    for fact in item.companions
                    if fact.layer_family == item.family
                )
                other = tuple(
                    fact.layer_id
                    for fact in item.companions
                    if fact.layer_family != item.family
                )
                self.assertEqual(bool(matching), item.list_outcome == "match")
                self.assertEqual(
                    Counter(matching + other),
                    Counter(fact.layer_id for fact in item.companions),
                )

    def test_claim_checks_are_exactly_40_60_with_hard_negatives(self) -> None:
        policy = self.settings["claim_check"]
        required = policy["required_unsupported_pairs"]
        for split, scenarios in self.scenarios.items():
            label = "training" if split == "train" else "evaluation"
            rows = [item for item in scenarios if item.form == 4]
            self.assertEqual(
                Counter(item.family for item in rows),
                Counter(policy[f"{label}_family_counts"]),
            )
            self.assertEqual(
                Counter(
                    "supported" if item.claim_supported else "unsupported"
                    for item in rows
                ),
                Counter(policy[f"{label}_outcome_counts"]),
            )
            self.assertEqual(
                sum(item.claim_supported for item in rows) / len(rows),
                0.4 if split == "train" else 75 / 187,
            )
            for family, family_total in policy[f"{label}_family_counts"].items():
                family_rows = [item for item in rows if item.family == family]
                supported_count = policy[f"{label}_supported_counts"][family]
                card = _expected_card(family, None)
                observed = Counter(item.claim for item in family_rows)
                expected = Counter(
                    _expected_unsupported_counts(family_total - supported_count, family)
                )
                for claim in card["supported_claims"]:
                    expected[claim] += supported_count
                self.assertEqual(observed, expected, (split, family))
                for item in family_rows:
                    self.assertEqual(
                        item.claim_supported,
                        item.claim in card["supported_claims"],
                    )
                for claim in required.get(family, []):
                    self.assertTrue(
                        any(
                            item.claim == claim and not item.claim_supported
                            for item in family_rows
                        ),
                        (split, family, claim),
                    )
                for claim in required["every_family"]:
                    self.assertTrue(
                        any(
                            item.claim == claim and not item.claim_supported
                            for item in family_rows
                        ),
                        (split, family, claim),
                    )
            self.assertNotIn(
                policy["excluded_scoped_negative"],
                {item.claim for item in rows},
            )

    def test_list_and_claim_frames_are_balanced_within_each_family(self) -> None:
        for split, scenarios in self.scenarios.items():
            for form, expected_train_count in ((2, 75), (4, 125)):
                rows = [item for item in scenarios if item.form == form]
                for family in {item.family for item in rows}:
                    counts = Counter(
                        item.framing for item in rows if item.family == family
                    )
                    self.assertEqual(set(counts), {"direct", "step"})
                    if split == "train":
                        self.assertEqual(
                            counts,
                            Counter(
                                {
                                    "direct": expected_train_count,
                                    "step": expected_train_count,
                                }
                            ),
                        )
                    else:
                        self.assertLessEqual(
                            abs(counts["direct"] - counts["step"]), 1
                        )

        train_claims = [
            item for item in self.scenarios["train"] if item.form == 4
        ]
        expected_joint = Counter()
        for family in FAMILY_CARDS:
            if family == "other":
                expected_joint[(family, "family_name", "unsupported")] = 50
                expected_joint[(family, "layer_tag", "unsupported")] = 200
            else:
                expected_joint[(family, "family_name", "supported")] = 24
                expected_joint[(family, "family_name", "unsupported")] = 26
                expected_joint[(family, "layer_tag", "supported")] = 96
                expected_joint[(family, "layer_tag", "unsupported")] = 104
        actual_joint = Counter(
            (
                item.family,
                item.input_kind,
                "supported" if item.claim_supported else "unsupported",
            )
            for item in train_claims
        )
        self.assertEqual(actual_joint, expected_joint)
        self.assertEqual(
            Counter((item.input_kind, item.framing) for item in train_claims),
            Counter(
                {
                    ("family_name", "direct"): 150,
                    ("family_name", "step"): 150,
                    ("layer_tag", "direct"): 600,
                    ("layer_tag", "step"): 600,
                }
            ),
        )
        self.assertEqual(
            Counter(
                (
                    "supported" if item.claim_supported else "unsupported",
                    item.framing,
                )
                for item in train_claims
            ),
            Counter(
                {
                    ("supported", "direct"): 300,
                    ("supported", "step"): 300,
                    ("unsupported", "direct"): 450,
                    ("unsupported", "step"): 450,
                }
            ),
        )
        for cell, count in expected_joint.items():
            frames = {
                item.framing
                for item in train_claims
                if (
                    item.family,
                    item.input_kind,
                    "supported" if item.claim_supported else "unsupported",
                )
                == cell
            }
            if count > 1:
                self.assertEqual(frames, {"direct", "step"})

        for split in ("validation", "test"):
            eval_claims = [
                item for item in self.scenarios[split] if item.form == 4
            ]
            self.assertEqual(
                Counter((item.input_kind, item.framing) for item in eval_claims),
                Counter(
                    {
                        ("family_name", "direct"): 47,
                        ("family_name", "step"): 47,
                        ("layer_tag", "direct"): 47,
                        ("layer_tag", "step"): 46,
                    }
                ),
            )
            self.assertEqual(
                Counter(
                    (
                        "supported" if item.claim_supported else "unsupported",
                        item.framing,
                    )
                    for item in eval_claims
                ),
                Counter(
                    {
                        ("supported", "direct"): 38,
                        ("supported", "step"): 37,
                        ("unsupported", "direct"): 56,
                        ("unsupported", "step"): 56,
                    }
                ),
            )

        expected_claim_input = Counter()
        for family in FAMILY_CARDS:
            family_rows = [item for item in train_claims if item.family == family]
            claim_counts = Counter(item.claim for item in family_rows)
            name_counts = _largest_remainder_subset(dict(claim_counts), 50)
            for claim in CLAIM_VOCABULARY:
                expected_claim_input[(family, claim, "family_name")] = name_counts[
                    claim
                ]
                expected_claim_input[(family, claim, "layer_tag")] = (
                    claim_counts[claim] - name_counts[claim]
                )
        actual_claim_input = Counter(
            (item.family, item.claim, item.input_kind) for item in train_claims
        )
        self.assertEqual(actual_claim_input, expected_claim_input)
        for cell, count in expected_claim_input.items():
            self.assertGreater(count, 0)
            frames = {
                item.framing
                for item in train_claims
                if (item.family, item.claim, item.input_kind) == cell
            }
            if count > 1:
                self.assertEqual(frames, {"direct", "step"})

    def test_contexts_templates_and_singleton_tag_are_split_safe(self) -> None:
        split_units = {
            split: {
                fact.context_key
                for item in rows
                for fact in (
                    *((item.fact,) if getattr(item.fact, "context_key", None) else ()),
                    *item.companions,
                )
            }
            for split, rows in self.scenarios.items()
        }
        self.assertTrue(split_units["train"].isdisjoint(split_units["validation"]))
        self.assertTrue(split_units["train"].isdisjoint(split_units["test"]))
        self.assertTrue(split_units["validation"].isdisjoint(split_units["test"]))
        train_tags = {
            fact.layer_id
            for item in self.scenarios["train"]
            for fact in ((item.fact,) if getattr(item.fact, "layer_id", None) else ())
        } | {
            fact.layer_id
            for item in self.scenarios["train"]
            for fact in item.companions
        }
        self.assertIn("TFs:brain", train_tags)
        for split in ("validation", "test"):
            tags = {
                fact.layer_id
                for item in self.scenarios[split]
                for fact in ((item.fact,) if getattr(item.fact, "layer_id", None) else ())
            } | {
                fact.layer_id
                for item in self.scenarios[split]
                for fact in item.companions
            }
            self.assertNotIn("TFs:brain", tags)
            self.assertTrue(
                all(
                    item.input_kind == "family_name"
                    for item in self.scenarios[split]
                    if item.family == "tf_target"
                )
            )

        for split, scenarios in self.scenarios.items():
            for form, count in EXPECTED_TRACE_COUNTS[split].items():
                if not count:
                    continue
                items = [item for item in scenarios if item.form == form]
                for framing in ("direct", "step"):
                    framed = [item for item in items if item.framing == framing]
                    self.assertTrue(framed, (split, form, framing))
                    self.assertEqual(
                        {item.template_index for item in framed},
                        set(range(len(FAMILY_PROMPTS[split][framing]))),
                    )


class WorldModelS17RowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.7"]
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

    def test_all_nine_traces_have_closed_book_metadata_and_zero_calls(self) -> None:
        for form in range(1, 10):
            scenario = self._first(form)
            row = self._build(scenario)
            answer = _answer(row)
            with self.subTest(form=form):
                self.assertEqual(list(row), ["row_id", "messages", "tools", "metadata"])
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
                self.assertIsNone(answer["provenance"])
                metadata = row["metadata"]
                expected_keys = set(self.contract["rows"]["metadata"]["required"])
                if form == 5:
                    expected_keys.add("found_input_type")
                self.assertEqual(set(metadata), expected_keys)
                self.assertEqual(metadata["book_mode"], "closed_book")
                self.assertEqual(metadata["step"], "S1")
                self.assertEqual(metadata["question_family"], "layer_family_claims")
                self.assertEqual(metadata["form"], form)
                self.assertEqual(metadata["source_trace"], f"S1.7.{form}")
                self.assertEqual(metadata["split_unit"], scenario.metadata_split_unit)
                if form == 5:
                    self.assertEqual(answer["status"], "invalid_input")
                    self.assertEqual(answer["continuation_decision"], "defer")
                    self.assertEqual(metadata["blocked_reason_code"], "wrong_input_type")
                    self.assertEqual(metadata["found_input_type"], "module_id")
                else:
                    self.assertEqual(answer["status"], "resolved")
                    self.assertEqual(
                        answer["continuation_decision"],
                        "continue" if scenario.framing == "step" else "stop",
                    )
                    self.assertIsNone(answer["blocked_reason"])

    def test_family_card_rows_have_exact_claim_partitions(self) -> None:
        for form in (1, 3, 6, 7, 8, 9):
            for input_kind in {item.input_kind for rows in self.scenarios.values()
                               for item in rows if item.form == form}:
                scenario = self._first(form, input_kind=input_kind)
                result = _answer(self._build(scenario))["result"]
                layer_id = (
                    None if input_kind == "family_name" else scenario.fact.layer_id
                )
                with self.subTest(form=form, input_kind=input_kind):
                    self.assertEqual(result, _expected_card(scenario.family, layer_id))
                    self.assertEqual(
                        set(result["supported_claims"])
                        | set(result["unsupported_claims"]),
                        set(CLAIM_VOCABULARY),
                    )
                    for key in ("supported_claims", "unsupported_claims"):
                        selected = set(result[key])
                        self.assertEqual(
                            result[key],
                            [
                                claim
                                for claim in CLAIM_VOCABULARY
                                if claim in selected
                            ],
                        )
                    if input_kind == "family_name":
                        self.assertIsNone(result["layer_id"])
                        self.assertEqual(_native_tags(
                            self._build(scenario)["messages"][1]["content"]
                        ), [])
                    else:
                        self.assertIn(
                            f"<LAYER:{layer_id}>",
                            _answer(self._build(scenario))["summary"],
                        )

    def test_list_results_partition_ids_without_reordering(self) -> None:
        samples = {
            (item.family, item.list_outcome): item
            for scenarios in self.scenarios.values()
            for item in scenarios
            if item.form == 2
        }
        self.assertEqual({family for family, _ in samples}, set(FAMILY_CARDS))
        self.assertEqual({outcome for _, outcome in samples}, {"match", "no_match"})
        for key, scenario in samples.items():
            row = self._build(scenario)
            result = _answer(row)["result"]
            matching = [
                fact.layer_id
                for fact in scenario.companions
                if fact.layer_family == scenario.family
            ]
            other = [
                fact.layer_id
                for fact in scenario.companions
                if fact.layer_family != scenario.family
            ]
            with self.subTest(family=key[0], outcome=key[1]):
                self.assertEqual(
                    list(result),
                    ["layer_family", "matching_layer_ids", "other_layer_ids"],
                )
                self.assertEqual(result["layer_family"], scenario.family)
                self.assertEqual(result["matching_layer_ids"], matching)
                self.assertEqual(result["other_layer_ids"], other)
                self.assertEqual(
                    _native_tags(row["messages"][1]["content"]),
                    [fact.layer_id for fact in scenario.companions],
                )
                self.assertEqual(
                    len(
                        set(
                            result["matching_layer_ids"]
                            + result["other_layer_ids"]
                        )
                    ),
                    3,
                )
                self.assertEqual(
                    sorted(result["matching_layer_ids"] + result["other_layer_ids"]),
                    sorted(fact.layer_id for fact in scenario.companions),
                )
                summary = _answer(row)["summary"]
                for fact in scenario.companions:
                    self.assertEqual(
                        summary.count(f"<LAYER:{fact.layer_id}>"), 1
                    )

    def test_claim_check_matches_supported_list_membership(self) -> None:
        samples = {
            (item.family, item.claim_supported, item.input_kind): item
            for scenarios in self.scenarios.values()
            for item in scenarios
            if item.form == 4
        }
        self.assertEqual({family for family, _, _ in samples}, set(FAMILY_CARDS))
        for key, scenario in samples.items():
            result = _answer(self._build(scenario))["result"]
            with self.subTest(key=key):
                self.assertEqual(
                    list(result),
                    [
                        "layer_id",
                        "layer_family",
                        "evidence_type",
                        "supported_claims",
                        "unsupported_claims",
                        "claim",
                        "claim_supported",
                    ],
                )
                self.assertEqual(
                    {name: result[name] for name in list(result)[:5]},
                    _expected_card(
                        scenario.family,
                        None
                        if scenario.input_kind == "family_name"
                        else scenario.fact.layer_id,
                    ),
                )
                self.assertEqual(result["claim"], scenario.claim)
                self.assertEqual(result["claim_supported"], scenario.claim_supported)
                self.assertEqual(
                    result["claim_supported"],
                    result["claim"] in result["supported_claims"],
                )
                user = self._build(scenario)["messages"][1]["content"].casefold()
                if scenario.family == "bulk_pen":
                    self.assertNotIn("one cell type", user)
                    self.assertNotIn("single cell", user)
                    self.assertNotIn("bulk_pen_single_cell", user)
                if scenario.input_kind == "family_name":
                    self.assertIn(
                        f"an edge in the {scenario.family} family", user
                    )

    def test_prompts_and_family_name_summaries_avoid_mechanical_text(self) -> None:
        for scenarios in self.scenarios.values():
            for scenario in scenarios:
                row = self._build(scenario)
                user = row["messages"][1]["content"].casefold()
                self.assertIsNone(
                    re.search(r"what an edge\b.*\bfor an edge\b", user),
                    user,
                )
                if scenario.input_kind == "family_name":
                    summary = _answer(row)["summary"].casefold()
                    self.assertNotIn(
                        f"{scenario.family} family is in the "
                        f"{scenario.family} family",
                        summary,
                    )

    def test_wrong_type_uses_only_source_backed_mentor_ev_ids(self) -> None:
        for split, scenarios in self.scenarios.items():
            source_ids = {
                fact.module_id for fact in self.pools[split]["modules"]
            }
            wrong = [item for item in scenarios if item.form == 5]
            self.assertEqual(len(wrong), EXPECTED_TRACE_COUNTS[split][5])
            for scenario in wrong:
                row = self._build(scenario)
                answer = _answer(row)
                self.assertEqual(scenario.fact.namespace, "mentor_ev")
                self.assertIn(scenario.fact.module_id, source_ids)
                self.assertEqual(
                    answer["result"],
                    {"input": scenario.fact.module_id, "expected": "layer_tag"},
                )
                self.assertEqual(
                    answer["blocked_reason"],
                    f"The input {scenario.fact.module_id} is a module ID, "
                    "not a layer tag.",
                )
                user = row["messages"][1]["content"]
                self.assertNotIn("ENSG", user)
                self.assertNotIn("layer tag", user.casefold())

    def test_eval_rows_use_held_out_wrappers_and_claim_paraphrases(self) -> None:
        for split, scenarios in self.scenarios.items():
            held_out_wrappers = [
                template
                for other, framings in FAMILY_PROMPTS.items()
                if other != split
                for values in framings.values()
                for template in values
            ]
            held_out_claims = {
                phrase
                for other, by_claim in CLAIM_PARAPHRASES.items()
                if other != split
                for values in by_claim.values()
                for phrase in values
            }
            for scenario in scenarios:
                row = self._build(scenario)
                user = row["messages"][1]["content"]
                self.assertEqual(
                    sum(
                        _matches_template(user, template)
                        for values in FAMILY_PROMPTS[split].values()
                        for template in values
                    ),
                    1,
                )
                self.assertFalse(
                    any(_matches_template(user, template) for template in held_out_wrappers)
                )
                if scenario.form == 4:
                    phrase = CLAIM_PARAPHRASES[split][scenario.claim][
                        scenario.claim_paraphrase_index
                    ]
                    self.assertIn(phrase, user)
                    self.assertFalse(any(phrase in user for phrase in held_out_claims))
                self.assertEqual(filter_row_thinking(row, scenario), 1)

        for split, scenarios in self.scenarios.items():
            claim_rows = [item for item in scenarios if item.form == 4]
            for claim in CLAIM_VOCABULARY:
                observed = {
                    item.claim_paraphrase_index
                    for item in claim_rows
                    if item.claim == claim
                }
                self.assertEqual(
                    observed,
                    set(range(len(CLAIM_PARAPHRASES[split][claim]))),
                    (split, claim),
                )

    def test_thinking_filter_rejects_semantic_mutations(self) -> None:
        scenario = self._first(4, claim_supported=False)
        row = self._build(scenario)
        mutations = (
            "",
            "I will follow the answer schema.",
            row["messages"][-1]["thinking"]
            + " This unsupported claim is supported.",
            row["messages"][-1]["thinking"]
            + " I will call a tool to decide the family.",
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
        self.assertEqual(flattened["question_family"], "layer_family_claims")
        self.assertEqual(flattened["record_id"], row["row_id"])
        self.assertEqual(flattened["fact_id"], scenario.metadata_split_unit)
        tokenizer = _Tokenizer()
        tokenized = tokenize_v3_trace_for_sft(row, tokenizer, max_length=64)
        self.assertEqual(tokenized["input_ids"], [1, 10, 20, 60, 61])
        self.assertEqual(tokenized["completion_mask"], [0, 0, 0, 1, 1])


class WorldModelS17AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.7"]
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
                build_row(scenario, contract=cls.contract, tools=cls.tools)[0]
                for scenario in scenarios
            ]
            for split, scenarios in cls.scenarios.items()
        }

    def test_duplicate_and_masked_wording_audits_pass(self) -> None:
        duplicate = duplicate_prompt_outcome_audit(self.rows, self.scenarios)
        self.assertTrue(duplicate["pass"])
        wording = masked_wording_classifier_audit(self.rows, self.scenarios)
        self.assertTrue(wording["pass"])
        for split in ("validation", "test"):
            self.assertTrue(wording["splits"][split]["gate_applied"])
            self.assertTrue(wording["splits"][split]["pass"])

    def test_aggregate_audit_reports_exact_core_counts(self) -> None:
        audit = audit_rows(
            self.rows,
            self.scenarios,
            EXPECTED_TRACE_COUNTS,
            framing_shares=self.contract["curriculum"]["framing"],
            settings=self.settings,
            contract=self.contract,
        )
        self.assertEqual(
            audit["status_counts"],
            {
                "train": {"invalid_input": 300, "resolved": 5700},
                "validation": {"invalid_input": 37, "resolved": 713},
                "test": {"invalid_input": 37, "resolved": 713},
            },
        )
        self.assertEqual(
            audit["tool_call_counts"],
            {
                "train": {"0": 6000},
                "validation": {"0": 750},
                "test": {"0": 750},
            },
        )
        self.assertEqual(
            audit["framing_counts"],
            {
                "train": {"direct": 3000, "step": 3000},
                "validation": {"direct": 375, "step": 375},
                "test": {"direct": 375, "step": 375},
            },
        )
        self.assertEqual(
            audit["semantic_bucket_counts"],
            {
                "train": {
                    "family_of_layer": 1500,
                    "list_filter": 900,
                    "claims_card": 1500,
                    "claim_check": 1500,
                    "wrong_input_type": 300,
                    "native_nonmapped_network": 300,
                },
                "validation": {
                    "family_of_layer": 188,
                    "list_filter": 113,
                    "claims_card": 188,
                    "claim_check": 187,
                    "wrong_input_type": 37,
                    "native_nonmapped_network": 37,
                },
                "test": {
                    "family_of_layer": 188,
                    "list_filter": 113,
                    "claims_card": 188,
                    "claim_check": 187,
                    "wrong_input_type": 37,
                    "native_nonmapped_network": 37,
                },
            },
        )
        self.assertEqual(
            {
                split: sum(counts.values())
                for split, counts in audit["family_counts"].items()
            },
            {"train": 5700, "validation": 713, "test": 713},
        )
        self.assertEqual(
            audit["claim_outcome_counts"],
            {
                "train": {"supported": 600, "unsupported": 900},
                "validation": {"supported": 75, "unsupported": 112},
                "test": {"supported": 75, "unsupported": 112},
            },
        )
        self.assertEqual(
            audit["list_outcome_counts"],
            {
                "train": {"match": 450, "no_match": 450},
                "validation": {"match": 57, "no_match": 56},
                "test": {"match": 57, "no_match": 56},
            },
        )
        self.assertEqual(
            audit["context_counts"], {"train": 27, "validation": 4, "test": 4}
        )
        self.assertEqual(
            audit["context_stratum_counts"], self.settings["context_split"]["strata"]
        )
        self.assertTrue(audit["per_form_gates"]["pass"])
        self.assertEqual(
            audit["per_form_gates"]["unsupported_claim_leak_count"], 0
        )
        self.assertTrue(
            audit["per_form_gates"]["heldout_family_accuracy"]["pass"]
        )
        self.assertTrue(
            audit["per_form_gates"]["heldout_claim_paraphrase_accuracy"]["pass"]
        )
        self.assertTrue(audit["prompt_structure_audit"]["pass"])
        self.assertTrue(audit["masked_wording_classifier"]["pass"])
        self.assertTrue(audit["thinking_filter"]["pass"])
        self.assertTrue(audit["duplicate_prompt_tool_outcome"]["pass"])

    def test_audits_reject_result_and_order_mutations(self) -> None:
        mutation_specs = []
        claim_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 4
        )
        changed_claim = deepcopy(self.rows)
        claim_result = _answer(changed_claim["train"][claim_index])["result"]
        claim_result["claim_supported"] = not claim_result["claim_supported"]
        mutation_specs.append(("claim membership", changed_claim))

        card_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 3
        )
        changed_order = deepcopy(self.rows)
        unsupported = _answer(changed_order["train"][card_index])["result"][
            "unsupported_claims"
        ]
        unsupported.reverse()
        mutation_specs.append(("claim order", changed_order))

        list_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 2
            and len(_answer(self.rows["train"][index])["result"]["other_layer_ids"]) > 1
        )
        changed_list = deepcopy(self.rows)
        _answer(changed_list["train"][list_index])["result"][
            "other_layer_ids"
        ].reverse()
        mutation_specs.append(("list order", changed_list))

        for name, rows in mutation_specs:
            with self.subTest(name=name):
                with self.assertRaises(CorpusBuildError):
                    audit_rows(
                        rows,
                        self.scenarios,
                        EXPECTED_TRACE_COUNTS,
                        framing_shares=self.contract["curriculum"]["framing"],
                        settings=self.settings,
                        contract=self.contract,
                    )

    def test_audit_rejects_the_excluded_bulk_cell_proposition(self) -> None:
        index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 4 and item.family == "bulk_pen"
        )
        changed_scenarios = deepcopy(self.scenarios)
        changed_scenarios["train"][index] = replace(
            changed_scenarios["train"][index],
            claim=self.settings["claim_check"]["excluded_scoped_negative"],
            claim_supported=False,
        )
        with self.assertRaises(CorpusBuildError):
            audit_rows(
                self.rows,
                changed_scenarios,
                EXPECTED_TRACE_COUNTS,
                framing_shares=self.contract["curriculum"]["framing"],
                settings=self.settings,
                contract=self.contract,
            )

    def test_duplicate_audit_rejects_an_ordinary_copy(self) -> None:
        scenario = self.scenarios["train"][0]
        duplicate_scenario = replace(scenario, occurrence=100_000)
        row = deepcopy(self.rows["train"][0])
        duplicate = deepcopy(row)
        duplicate["row_id"] = "wm3_s1_7_duplicate"
        duplicate["messages"][-1]["content"]["summary"] = (
            "Different summary text for the same result."
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


class WorldModelS17GeneratedCorpusTests(unittest.TestCase):
    def test_build_is_deterministic_and_validates_before_publication(self) -> None:
        contract = load_contract()
        source_index = _source_index(contract)
        if not source_index.is_file():
            self.skipTest(f"canonical source index is absent: {source_index}")
        real_validate = s1_7_module._validate_corpus
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
                s1_7_module, "_validate_corpus", side_effect=validate_before_publish
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
            published = Path(
                contract["curriculum"]["families"]["S1.7"]["output_directory"]
            )
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    (first_root / f"{split}.jsonl").read_bytes(),
                    (published / f"{split}.jsonl").read_bytes(),
                )
            for key in ("builder", "s1_6_builder"):
                source = first["sources"][key]
                self.assertEqual(source["sha256"], file_sha256(Path(source["path"])))
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

    def test_generated_manifest_and_rows_reconcile_when_present(self) -> None:
        # The current builders publish S1 v3, which trains on the pinned
        # prompt bank; the v1 directory predates prompt-bank support.
        contract, overlay = apply_s1_overlay(load_contract(), S1_V3_OVERLAY_PATH)
        settings = contract["curriculum"]["families"]["S1.7"]
        prompts = wrapper_prompts(
            overlay["prompt_banks"], FAMILY_PROMPTS, bank_name="layer_family"
        )
        root = Path(settings["output_directory"])
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"generated corpus is absent: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"], "mentor-rl-world-model-s1-7-manifest-v1"
        )
        self.assertEqual(
            manifest["corpus_schema_version"],
            "mentor-rl-world-model-s1-7-corpus-v1",
        )
        self.assertEqual(manifest["section"], "S1.7")
        self.assertEqual(manifest["dataset_id"], settings["dataset_id"])
        self.assertEqual(manifest["seed"], 1729)
        self.assertEqual(
            manifest["row_counts"], {"train": 6000, "validation": 750, "test": 750}
        )
        self.assertEqual(
            manifest["form_counts"],
            {
                split: {
                    f"S1.7.{form}": count for form, count in counts.items()
                }
                for split, counts in EXPECTED_TRACE_COUNTS.items()
            },
        )
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
            manifest["claim_paraphrases_sha256"],
            hashlib.sha256(
                json.dumps(
                    CLAIM_PARAPHRASES,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        )
        dependency = manifest["sources"]["s1_6_builder"]
        dependency_path = Path(dependency["path"])
        self.assertEqual(dependency_path, Path("runtime/world_model/s1_6_corpus.py"))
        # The published manifest keeps its historical dependency pin.
        # Fresh builds above check current pins and reproduce the old rows.
        self.assertRegex(dependency["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["evidence"]["call_count"], 0)
        self.assertEqual(manifest["evidence"]["calls"], [])
        self.assertEqual(
            {
                split: sum(counts.values())
                for split, counts in manifest["family_counts"].items()
            },
            {"train": 5700, "validation": 713, "test": 713},
        )
        for split, count in manifest["row_counts"].items():
            path = root / f"{split}.jsonl"
            self.assertEqual(file_sha256(path), manifest["files"][path.name]["sha256"])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), count)
            self.assertEqual(len({row["row_id"] for row in rows}), count)
            self.assertTrue(all(_calls(row) == [] for row in rows))
            self.assertTrue(all(_answer(row)["provenance"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
