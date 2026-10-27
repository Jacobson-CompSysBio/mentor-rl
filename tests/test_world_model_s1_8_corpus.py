"""Focused tests for the canonical S1.8 direction corpus builder."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runtime.world_model.s1_8_corpus as s1_8_module

from runtime.world_model.contract import (
    REPOSITORY_ROOT,
    load_contract,
    tool_definitions,
)
from runtime.world_model.s1_8_corpus import (
    AMBIGUOUS_INPUTS,
    METRIC_CONVENTIONS,
    NAME_VARIANTS,
    PROMPT_TEMPLATES,
    SHARED_ALIASES,
    UNKNOWN_INPUTS,
    CorpusBuildError,
    Scenario,
    audit_rows,
    build_corpus,
    build_row,
    convention_card,
    duplicate_prompt_outcome_audit,
    filter_row_thinking,
    masked_wording_classifier_audit,
    select_scenarios,
    split_row_counts,
    trace_counts,
)
from runtime.world_model.s1_overlay import S1_V3_OVERLAY_PATH, apply_s1_overlay
from runtime.world_model.s1_prompt_banks import metric_names, wrapper_prompts
from runtime.world_model.sources import file_sha256
from runtime.world_model_training import (
    flatten_v3_trace_for_arrow,
    tokenize_v3_trace_for_sft,
)


FORM_BY_ALLOCATION = {
    "direction_lookup": 1,
    "true_false_distance_style": 2,
    "true_false_similarity_style": 3,
    "true_false_workflow_style": 4,
    "ambiguous_percentile": 5,
    "unrecognized_metric": 6,
}
EXPECTED_TRACE_COUNTS = {
    "train": {1: 1200, 2: 667, 3: 667, 4: 666, 5: 400, 6: 400},
    "validation": {1: 150, 2: 84, 3: 83, 4: 83, 5: 50, 6: 50},
    "test": {1: 150, 2: 84, 3: 83, 4: 83, 5: 50, 6: 50},
}
EXPECTED_CONVENTIONS = {
    "rank": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "lower_rank_closer",
    },
    "score": {
        "lower_is_stronger": False,
        "meaning": "proximity",
        "rule": "higher_score_closer",
    },
    "spearman_distance": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "lower_distance_closer",
    },
    "pearson_distance": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "lower_distance_closer",
    },
    "spearman_correlation": {
        "lower_is_stronger": False,
        "meaning": "similarity",
        "rule": "higher_correlation_more_similar",
    },
    "pearson_correlation": {
        "lower_is_stronger": False,
        "meaning": "similarity",
        "rule": "higher_correlation_more_similar",
    },
    "dot_similarity": {
        "lower_is_stronger": False,
        "meaning": "similarity",
        "rule": "higher_similarity_more_similar",
    },
    "distance_percentile": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "lower_distance_percentile_closer",
    },
    "score_percentile": {
        "lower_is_stronger": False,
        "meaning": "proximity",
        "rule": "higher_score_percentile_closer",
    },
    "hop_count": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "fewer_hops_closer",
    },
    "empirical_p_value": {
        "lower_is_stronger": True,
        "meaning": "evidence",
        "rule": "lower_p_stronger_evidence",
    },
    "q": {
        "lower_is_stronger": True,
        "meaning": "evidence",
        "rule": "lower_p_stronger_evidence",
    },
    "enrichment_score": {
        "lower_is_stronger": False,
        "meaning": "evidence",
        "rule": "higher_enrichment_stronger_evidence",
    },
}
POLARITY_ORDER = (
    "lower_stronger",
    "lower_weaker",
    "higher_stronger",
    "higher_weaker",
)
EXPECTED_MANIFEST_KEYS = {
    "schema_version",
    "corpus_schema_version",
    "section",
    "dataset_id",
    "seed",
    "contract_s1_8_sha256",
    "prompt_templates_sha256",
    "request_templates_sha256",
    "metric_conventions_sha256",
    "name_sources_sha256",
    "scenario_assignment_sha256",
    "row_counts",
    "form_counts",
    "semantic_bucket_counts",
    "status_counts",
    "tool_call_counts",
    "framing_counts",
    "prompt_template_counts",
    "metric_counts",
    "candidate_metric_counts",
    "name_kind_counts",
    "statement_truth_counts",
    "statement_polarity_counts",
    "metric_polarity_counts",
    "agreeing_alias_counts",
    "ambiguous_input_counts",
    "unknown_input_counts",
    "continuation_counts",
    "sources",
    "source_coverage",
    "split_unit_sha256",
    "per_form_gates",
    "prompt_structure_audit",
    "masked_wording_classifier",
    "thinking_filter",
    "duplicate_prompt_tool_outcome",
    "evidence",
    "files",
}


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


def _plain(value: object) -> object:
    return json.loads(json.dumps(value))


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _statement_truth(lower_is_stronger: bool, polarity: str) -> bool:
    if polarity in {"lower_stronger", "higher_weaker"}:
        return lower_is_stronger
    if polarity in {"lower_weaker", "higher_stronger"}:
        return not lower_is_stronger
    raise AssertionError(f"unknown polarity {polarity}")


def _expected_result(scenario: Scenario) -> dict[str, object]:
    result: dict[str, object] = {
        "metric": scenario.metric,
        "input": scenario.input_name,
        "lower_is_stronger": scenario.lower_is_stronger,
        "meaning": scenario.meaning,
        "rule": scenario.rule,
        "statement_true": scenario.statement_true,
    }
    if scenario.candidate_metrics:
        result["candidate_metrics"] = list(scenario.candidate_metrics)
    return result


def _expected_split_unit(scenario: Scenario) -> str:
    if scenario.name_kind == "canonical":
        return f"shared_metric_{scenario.metric}"
    digest = hashlib.sha256(
        scenario.input_name.casefold().encode("utf-8")
    ).hexdigest()
    return f"heldout_metric_name_{digest[:20]}"


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


class WorldModelS18ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.curriculum = self.contract["curriculum"]
        self.settings = self.curriculum["families"]["S1.8"]

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
                "result_shape",
                "metric_conventions",
                "name_variants",
                "shared_aliases",
                "name_policy",
                "true_false",
                "ambiguous_metric",
                "unrecognized_metric",
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
                "dataset_id": "world_model_v3_s1_8_direction_conventions",
                "question_family": "direction_conventions",
                "book_mode": "closed_book",
                "output_directory": "data/world_model_v3/corpora/s1/s1_8",
                "assignment_seed": 1729,
                "train_rows": 4000,
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
            self.settings["source_trace_by_allocation"],
            {name: f"S1.8.{form}" for name, form in FORM_BY_ALLOCATION.items()},
        )
        self.assertEqual(
            self.settings["semantic_buckets"],
            {
                "direction_lookup": ["direction_lookup"],
                "true_false": [
                    "true_false_distance_style",
                    "true_false_similarity_style",
                    "true_false_workflow_style",
                ],
                "ambiguous_metric": ["ambiguous_percentile"],
                "unrecognized_metric": ["unrecognized_metric"],
            },
        )

    def test_split_and_trace_allocations_are_exact(self) -> None:
        row_counts = split_row_counts(
            self.settings["train_rows"], self.curriculum["split_fractions"]
        )
        self.assertEqual(
            row_counts, {"train": 4000, "validation": 500, "test": 500}
        )
        counts = {
            split: trace_counts(total, self.settings)
            for split, total in row_counts.items()
        }
        self.assertEqual(counts, EXPECTED_TRACE_COUNTS)
        for split, values in counts.items():
            self.assertEqual(sum(values.values()), row_counts[split])

    def test_metric_truth_table_is_exact_and_ordered(self) -> None:
        self.assertEqual(self.settings["metric_conventions"], EXPECTED_CONVENTIONS)
        self.assertEqual(_plain(METRIC_CONVENTIONS), EXPECTED_CONVENTIONS)
        self.assertEqual(list(METRIC_CONVENTIONS), list(EXPECTED_CONVENTIONS))
        for metric, expected in EXPECTED_CONVENTIONS.items():
            with self.subTest(metric=metric):
                self.assertEqual(convention_card(metric), expected)
        self.assertNotIn("lca_depth", METRIC_CONVENTIONS)

    def test_result_shapes_and_statement_polarities_are_frozen(self) -> None:
        self.assertEqual(
            self.settings["result_shape"],
            {
                "base_keys": [
                    "metric",
                    "input",
                    "lower_is_stronger",
                    "meaning",
                    "rule",
                    "statement_true",
                ],
                "candidate_metrics_position": "append",
                "single_metric": {
                    "metric": "canonical_metric",
                    "candidate_metrics": "absent",
                },
                "shared_alias": {
                    "metric": None,
                    "candidate_metrics": "ordered_candidates",
                    "direction_fields": "shared_non_null",
                },
                "ambiguous_metric": {
                    "metric": None,
                    "candidate_metrics": "ordered_candidates",
                    "direction_fields": None,
                },
                "unrecognized_metric": {
                    "metric": None,
                    "candidate_metrics": "absent",
                    "direction_fields": None,
                },
            },
        )
        self.assertEqual(
            self.settings["true_false"]["polarity_order"],
            list(POLARITY_ORDER),
        )
        self.assertEqual(
            self.settings["true_false"]["train_truth_counts"],
            {"true": 1000, "false": 1000},
        )
        self.assertEqual(
            self.settings["true_false"]["evaluation_truth_counts"],
            {"true": 125, "false": 125},
        )

    def test_names_aliases_and_negative_inputs_are_split_disjoint(self) -> None:
        self.assertEqual(_plain(NAME_VARIANTS), self.settings["name_variants"])
        self.assertEqual(_plain(SHARED_ALIASES), self.settings["shared_aliases"])
        self.assertEqual(
            _plain(AMBIGUOUS_INPUTS),
            {
                "train": self.settings["ambiguous_metric"]["train_inputs"],
                "validation": self.settings["ambiguous_metric"][
                    "validation_inputs"
                ],
                "test": self.settings["ambiguous_metric"]["test_inputs"],
            },
        )
        self.assertEqual(
            _plain(UNKNOWN_INPUTS),
            {
                "train": self.settings["unrecognized_metric"]["train_inputs"],
                "validation": self.settings["unrecognized_metric"][
                    "validation_inputs"
                ],
                "test": self.settings["unrecognized_metric"]["test_inputs"],
            },
        )
        categories: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            self.assertEqual(set(NAME_VARIANTS[split]), set(EXPECTED_CONVENTIONS))
            self.assertTrue(
                all(len(NAME_VARIANTS[split][metric]) == 3
                    for metric in EXPECTED_CONVENTIONS)
            )
            values = {
                value.casefold()
                for variants in NAME_VARIANTS[split].values()
                for value in variants
            }
            values.update(value.casefold() for value in SHARED_ALIASES[split])
            values.update(value.casefold() for value in AMBIGUOUS_INPUTS[split])
            values.update(value.casefold() for value in UNKNOWN_INPUTS[split])
            self.assertNotIn("lca_depth", values)
            self.assertNotIn("lca depth", values)
            categories[split] = values
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            self.assertTrue(categories[first].isdisjoint(categories[second]))

    def test_shared_and_ambiguous_candidates_have_required_directions(self) -> None:
        for split, aliases in SHARED_ALIASES.items():
            for name, card in aliases.items():
                with self.subTest(split=split, name=name):
                    candidates = card["candidate_metrics"]
                    self.assertEqual(len(candidates), 2)
                    self.assertEqual(
                        {METRIC_CONVENTIONS[item]["lower_is_stronger"]
                         for item in candidates},
                        {card["lower_is_stronger"]},
                    )
                    for field in ("meaning", "rule"):
                        self.assertEqual(
                            {METRIC_CONVENTIONS[item][field]
                             for item in candidates},
                            {card[field]},
                        )
        candidates = self.settings["ambiguous_metric"]["candidate_metrics"]
        self.assertEqual(candidates, ["distance_percentile", "score_percentile"])
        self.assertEqual(
            {METRIC_CONVENTIONS[item]["lower_is_stronger"] for item in candidates},
            {False, True},
        )

    def test_builder_rejects_direct_contract_mapping_drift(self) -> None:
        changed_settings = []

        changed = deepcopy(self.settings)
        changed["allocations"]["direction_lookup"] = 1199
        changed_settings.append(("allocation", changed))

        changed = deepcopy(self.settings)
        changed["source_trace_by_allocation"]["direction_lookup"] = "S1.8.2"
        changed_settings.append(("source trace", changed))

        changed = deepcopy(self.settings)
        changed["result_shape"]["base_keys"].reverse()
        changed_settings.append(("result shape", changed))

        changed = deepcopy(self.settings)
        changed["semantic_buckets"]["direction_lookup"] = [
            "direction_lookup",
            "unrecognized_metric",
        ]
        changed_settings.append(("semantic bucket", changed))

        changed = deepcopy(self.settings)
        changed["name_policy"]["variant_share_of_resolved"] = 0.31
        changed_settings.append(("name policy", changed))

        changed = deepcopy(self.settings)
        changed["true_false"]["allocation_policy"] = "changed"
        changed_settings.append(("truth policy", changed))

        changed = deepcopy(self.settings)
        changed["status_counts"]["train"]["resolved"] = 3199
        changed_settings.append(("status", changed))

        changed = deepcopy(self.settings)
        changed["framing_counts"]["train"]["direct"] = 1999
        changed_settings.append(("framing", changed))

        for name, settings in changed_settings:
            with self.subTest(name=name):
                with self.assertRaises(CorpusBuildError):
                    trace_counts(4000, settings)


class WorldModelS18SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.8"]
        cls.scenarios = select_scenarios(
            EXPECTED_TRACE_COUNTS,
            seed=cls.settings["assignment_seed"],
            framing_shares=cls.contract["curriculum"]["framing"],
            settings=cls.settings,
        )

    def test_scenario_counts_forms_and_framing_are_exact(self) -> None:
        expected_frames = {
            "train": {"direct": 2000, "step": 2000},
            "validation": {"direct": 250, "step": 250},
            "test": {"direct": 250, "step": 250},
        }
        scenario_ids: set[str] = set()
        for split, scenarios in self.scenarios.items():
            with self.subTest(split=split):
                self.assertEqual(len(scenarios), sum(EXPECTED_TRACE_COUNTS[split].values()))
                self.assertEqual(
                    Counter(item.form for item in scenarios),
                    Counter(EXPECTED_TRACE_COUNTS[split]),
                )
                self.assertEqual(
                    Counter(item.framing for item in scenarios),
                    Counter(expected_frames[split]),
                )
                self.assertTrue(all(item.split == split for item in scenarios))
                for item in scenarios:
                    self.assertNotIn(item.scenario_id, scenario_ids)
                    scenario_ids.add(item.scenario_id)

    def test_name_kind_and_form_allocations_are_exact(self) -> None:
        policy = self.settings["name_policy"]
        for split, scenarios in self.scenarios.items():
            resolved = [item for item in scenarios if item.form <= 4]
            mode = "train" if split == "train" else "evaluation"
            self.assertEqual(
                Counter(item.name_kind for item in resolved),
                Counter(policy[f"{mode}_counts"]),
            )
            for allocation, form in FORM_BY_ALLOCATION.items():
                if form > 4:
                    continue
                selected = [item for item in resolved if item.form == form]
                self.assertEqual(
                    Counter(item.name_kind for item in selected),
                    Counter(policy[f"{mode}_form_counts"][allocation]),
                )

    def test_each_input_kind_uses_only_its_declared_split_pool(self) -> None:
        for split, scenarios in self.scenarios.items():
            for scenario in scenarios:
                with self.subTest(split=split, scenario=scenario.scenario_id):
                    if scenario.name_kind == "canonical":
                        self.assertEqual(scenario.input_name, scenario.metric)
                        self.assertFalse(scenario.candidate_metrics)
                    elif scenario.name_kind == "metric_variant":
                        self.assertIn(
                            scenario.input_name,
                            NAME_VARIANTS[split][scenario.metric],
                        )
                        self.assertFalse(scenario.candidate_metrics)
                    elif scenario.name_kind == "shared_alias":
                        self.assertIsNone(scenario.metric)
                        card = SHARED_ALIASES[split][scenario.input_name]
                        self.assertEqual(
                            list(scenario.candidate_metrics),
                            list(card["candidate_metrics"]),
                        )
                        self.assertEqual(scenario.lower_is_stronger,
                                         card["lower_is_stronger"])
                        self.assertEqual(scenario.meaning, card["meaning"])
                        self.assertEqual(scenario.rule, card["rule"])
                    elif scenario.name_kind == "ambiguous":
                        self.assertIn(scenario.input_name, AMBIGUOUS_INPUTS[split])
                    elif scenario.name_kind == "unknown":
                        self.assertIn(scenario.input_name, UNKNOWN_INPUTS[split])
                    else:
                        self.fail(f"unexpected name kind {scenario.name_kind}")
                    if scenario.metric is not None:
                        card = EXPECTED_CONVENTIONS[scenario.metric]
                        self.assertEqual(
                            scenario.lower_is_stronger,
                            card["lower_is_stronger"],
                        )
                        self.assertEqual(scenario.meaning, card["meaning"])
                        self.assertEqual(scenario.rule, card["rule"])

    def test_every_declared_paraphrase_and_shared_alias_is_used(self) -> None:
        for split, scenarios in self.scenarios.items():
            metric_variants = {
                item.input_name
                for item in scenarios
                if item.name_kind == "metric_variant"
            }
            aliases = {
                item.input_name
                for item in scenarios
                if item.name_kind == "shared_alias"
            }
            self.assertEqual(
                metric_variants,
                {
                    value
                    for variants in NAME_VARIANTS[split].values()
                    for value in variants
                },
                split,
            )
            self.assertEqual(aliases, set(SHARED_ALIASES[split]), split)

    def test_true_false_rows_cover_the_complete_direction_truth_table(self) -> None:
        settings = self.settings["true_false"]
        for split, scenarios in self.scenarios.items():
            selected = [item for item in scenarios if item.form in {2, 3, 4}]
            mode = "train" if split == "train" else "evaluation"
            self.assertEqual(
                Counter(str(item.statement_true).lower() for item in selected),
                Counter(settings[f"{mode}_truth_counts"]),
            )
            for allocation, form in FORM_BY_ALLOCATION.items():
                if form not in {2, 3, 4}:
                    continue
                form_rows = [item for item in selected if item.form == form]
                self.assertEqual(
                    Counter(str(item.statement_true).lower() for item in form_rows),
                    Counter(settings[f"{mode}_form_truth_counts"][allocation]),
                )
            single_metric = [item for item in selected if item.metric is not None]
            coverage = {
                (item.form, item.metric, item.statement_polarity)
                for item in single_metric
            }
            self.assertEqual(
                coverage,
                {
                    (form, metric, polarity)
                    for form in (2, 3, 4)
                    for metric in EXPECTED_CONVENTIONS
                    for polarity in POLARITY_ORDER
                },
            )
            for item in selected:
                self.assertEqual(
                    item.statement_true,
                    _statement_truth(item.lower_is_stronger, item.statement_polarity),
                )

    def test_metric_and_polarity_allocations_stay_balanced(self) -> None:
        for split, scenarios in self.scenarios.items():
            for form in range(1, 5):
                single_metric = [
                    item
                    for item in scenarios
                    if item.form == form and item.metric is not None
                ]
                metric_counts = Counter(item.metric for item in single_metric)
                self.assertEqual(set(metric_counts), set(EXPECTED_CONVENTIONS))
                self.assertLessEqual(
                    max(metric_counts.values()) - min(metric_counts.values()), 1
                )
                if form > 1:
                    for metric in EXPECTED_CONVENTIONS:
                        polarity_counts = Counter(
                            item.statement_polarity
                            for item in single_metric
                            if item.metric == metric
                        )
                        self.assertEqual(set(polarity_counts), set(POLARITY_ORDER))
                        self.assertLessEqual(
                            max(polarity_counts.values())
                            - min(polarity_counts.values()),
                            2,
                            (split, form, metric, polarity_counts),
                        )

            lookup_rows = [item for item in scenarios if item.form == 1]
            for name_kind in ("canonical", "metric_variant", "shared_alias"):
                frames = Counter(
                    item.framing
                    for item in lookup_rows
                    if item.name_kind == name_kind
                )
                self.assertEqual(set(frames), {"direct", "step"})
                self.assertLessEqual(
                    max(frames.values()) - min(frames.values()), 1
                )

    def test_each_metric_occurs_in_each_resolved_form_and_framing(self) -> None:
        for split, scenarios in self.scenarios.items():
            single_metric = [item for item in scenarios if item.metric is not None]
            coverage = {
                (item.metric, item.form, item.framing) for item in single_metric
            }
            self.assertEqual(
                coverage,
                {
                    (metric, form, framing)
                    for metric in EXPECTED_CONVENTIONS
                    for form in range(1, 5)
                    for framing in ("direct", "step")
                },
                split,
            )

    def test_truth_labels_do_not_follow_name_kind_or_framing(self) -> None:
        for split, scenarios in self.scenarios.items():
            selected = [item for item in scenarios if item.form in {2, 3, 4}]
            groups: dict[tuple[object, ...], Counter[object]] = {}
            for item in selected:
                truth_key = (item.form, item.name_kind)
                groups.setdefault(truth_key, Counter())[item.statement_true] += 1
                frame_key = (item.form, item.name_kind, item.statement_true)
                groups.setdefault(frame_key, Counter())[item.framing] += 1
            for key, counts in groups.items():
                with self.subTest(split=split, key=key):
                    self.assertEqual(len(counts), 2)
                    self.assertLessEqual(
                        max(counts.values()) - min(counts.values()), 1
                    )

    def test_ambiguous_and_unknown_inputs_have_balanced_full_coverage(self) -> None:
        for split, scenarios in self.scenarios.items():
            ambiguous = [item for item in scenarios if item.form == 5]
            unknown = [item for item in scenarios if item.form == 6]
            self.assertEqual(
                {item.input_name for item in ambiguous}, set(AMBIGUOUS_INPUTS[split])
            )
            self.assertEqual(
                {item.input_name for item in unknown}, set(UNKNOWN_INPUTS[split])
            )
            for selected in (ambiguous, unknown):
                counts = Counter(item.input_name for item in selected)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                for input_name in counts:
                    frames = Counter(
                        item.framing for item in selected if item.input_name == input_name
                    )
                    self.assertLessEqual(abs(frames["direct"] - frames["step"]), 1)
            for item in ambiguous:
                self.assertEqual(
                    list(item.candidate_metrics),
                    self.settings["ambiguous_metric"]["candidate_metrics"],
                )
                self.assertIsNone(item.metric)
                self.assertIsNone(item.lower_is_stronger)
                self.assertIsNone(item.meaning)
                self.assertIsNone(item.rule)
                self.assertIsNone(item.statement_true)
            for item in unknown:
                self.assertIsNone(item.metric)
                self.assertFalse(item.candidate_metrics)
                self.assertIsNone(item.lower_is_stronger)
                self.assertIsNone(item.meaning)
                self.assertIsNone(item.rule)
                self.assertIsNone(item.statement_true)

    def test_templates_are_complete_and_split_held_out(self) -> None:
        self.assertEqual(set(PROMPT_TEMPLATES), {"train", "validation", "test"})
        wrappers = {}
        for split, framings in PROMPT_TEMPLATES.items():
            self.assertEqual(set(framings), {"direct", "step"})
            wrappers[split] = {
                template for values in framings.values() for template in values
            }
            self.assertTrue(wrappers[split])
            self.assertTrue(
                all(template.count("{request}") == 1 for template in wrappers[split])
            )
            scenarios = self.scenarios[split]
            for framing, templates in framings.items():
                observed = {
                    item.template_index
                    for item in scenarios
                    if item.framing == framing
                }
                self.assertEqual(observed, set(range(len(templates))))
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            self.assertTrue(wrappers[first].isdisjoint(wrappers[second]))

    def test_every_request_paraphrase_is_used_for_its_form(self) -> None:
        for split, scenarios in self.scenarios.items():
            for form in range(1, 7):
                requests = (
                    s1_8_module.DIRECTION_REQUESTS
                    if form == 1
                    else s1_8_module.TRUE_FALSE_REQUESTS[form]
                )
                observed = {
                    item.request_index for item in scenarios if item.form == form
                }
                self.assertEqual(
                    observed,
                    set(range(len(requests))),
                    (split, form),
                )
        request_sets = {
            form: set(s1_8_module.TRUE_FALSE_REQUESTS[form])
            for form in (2, 3, 4)
        }
        self.assertTrue(request_sets[2].isdisjoint(request_sets[3]))
        self.assertTrue(request_sets[2].isdisjoint(request_sets[4]))
        self.assertTrue(request_sets[3].isdisjoint(request_sets[4]))


class WorldModelS18RowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.8"]
        cls.tools = tool_definitions(cls.contract)
        cls.scenarios = select_scenarios(
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

    def test_all_six_forms_have_closed_book_metadata_and_zero_calls(self) -> None:
        for form in range(1, 7):
            scenario = self._first(form)
            row = self._build(scenario)
            answer = _answer(row)
            metadata = row["metadata"]
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
                self.assertEqual(set(metadata), set(self.contract["rows"]["metadata"]["required"]))
                self.assertEqual(metadata["book_mode"], "closed_book")
                self.assertEqual(metadata["step"], "S1")
                self.assertEqual(metadata["question_family"], "direction_conventions")
                self.assertEqual(metadata["form"], form)
                self.assertEqual(metadata["source_trace"], f"S1.8.{form}")
                self.assertEqual(metadata["split_unit"], _expected_split_unit(scenario))
                if form == 5:
                    self.assertEqual(answer["status"], "ambiguous")
                    self.assertEqual(metadata["blocked_reason_code"], "ambiguous_metric")
                elif form == 6:
                    self.assertEqual(answer["status"], "not_found")
                    self.assertEqual(metadata["blocked_reason_code"], "unrecognized_metric")
                else:
                    self.assertEqual(answer["status"], "resolved")
                    self.assertIsNone(metadata["blocked_reason_code"])

    def test_single_metric_and_alias_rows_have_exact_ordered_results(self) -> None:
        samples = [
            next(
                item
                for scenarios in self.scenarios.values()
                for item in scenarios
                if item.form == form and item.name_kind == kind
            )
            for form in range(1, 5)
            for kind in ("canonical", "metric_variant", "shared_alias")
        ]
        for scenario in samples:
            answer = _answer(self._build(scenario))
            result = answer["result"]
            with self.subTest(form=scenario.form, kind=scenario.name_kind):
                self.assertEqual(result, _expected_result(scenario))
                expected_keys = [
                    "metric",
                    "input",
                    "lower_is_stronger",
                    "meaning",
                    "rule",
                    "statement_true",
                ]
                if scenario.name_kind == "shared_alias":
                    expected_keys.append("candidate_metrics")
                    self.assertIsNone(result["metric"])
                    self.assertTrue(result["candidate_metrics"])
                else:
                    self.assertNotIn("candidate_metrics", result)
                    self.assertIn(result["metric"], EXPECTED_CONVENTIONS)
                self.assertEqual(list(result), expected_keys)
                self.assertEqual(
                    answer["continuation_decision"],
                    "continue" if scenario.framing == "step" else "stop",
                )
                self.assertIsNone(answer["blocked_reason"])

    def test_true_false_answers_follow_the_direction_truth_table(self) -> None:
        samples: dict[tuple[str, str], Scenario] = {}
        for scenarios in self.scenarios.values():
            for item in scenarios:
                if item.form in {2, 3, 4} and item.metric is not None:
                    samples.setdefault((item.metric, item.statement_polarity), item)
        self.assertEqual(
            set(samples),
            {
                (metric, polarity)
                for metric in EXPECTED_CONVENTIONS
                for polarity in POLARITY_ORDER
            },
        )
        for key, scenario in samples.items():
            result = _answer(self._build(scenario))["result"]
            with self.subTest(metric=key[0], polarity=key[1]):
                self.assertEqual(result, _expected_result(scenario))
                self.assertEqual(
                    result["statement_true"],
                    _statement_truth(
                        EXPECTED_CONVENTIONS[scenario.metric]["lower_is_stronger"],
                        scenario.statement_polarity,
                    ),
                )

    def test_ambiguous_rows_defer_without_a_direction_claim(self) -> None:
        for split in ("train", "validation", "test"):
            scenario = next(item for item in self.scenarios[split] if item.form == 5)
            answer = _answer(self._build(scenario))
            result = answer["result"]
            with self.subTest(split=split):
                self.assertEqual(answer["status"], "ambiguous")
                self.assertEqual(answer["continuation_decision"], "defer")
                self.assertEqual(result, _expected_result(scenario))
                self.assertEqual(
                    list(result),
                    [
                        "metric",
                        "input",
                        "lower_is_stronger",
                        "meaning",
                        "rule",
                        "statement_true",
                        "candidate_metrics",
                    ],
                )
                self.assertEqual(
                    answer["blocked_reason"],
                    f'The name "{scenario.input_name}" can mean '
                    "distance_percentile or score_percentile, and the two "
                    "have opposite directions.",
                )
                self.assertTrue(
                    all(result[key] is None for key in (
                        "metric",
                        "lower_is_stronger",
                        "meaning",
                        "rule",
                        "statement_true",
                    ))
                )

    def test_unknown_rows_defer_without_candidates_or_assumptions(self) -> None:
        for split in ("train", "validation", "test"):
            scenario = next(item for item in self.scenarios[split] if item.form == 6)
            answer = _answer(self._build(scenario))
            result = answer["result"]
            with self.subTest(split=split):
                self.assertEqual(answer["status"], "not_found")
                self.assertEqual(answer["continuation_decision"], "defer")
                self.assertEqual(result, _expected_result(scenario))
                self.assertNotIn("candidate_metrics", result)
                self.assertEqual(
                    answer["blocked_reason"],
                    f'The metric "{scenario.input_name}" is not in the '
                    "direction conventions.",
                )
                self.assertTrue(
                    all(result[key] is None for key in (
                        "metric",
                        "lower_is_stronger",
                        "meaning",
                        "rule",
                        "statement_true",
                    ))
                )

    def test_rows_use_only_split_specific_templates_and_names(self) -> None:
        for split, scenarios in self.scenarios.items():
            held_out_templates = [
                template
                for other, framings in PROMPT_TEMPLATES.items()
                if other != split
                for values in framings.values()
                for template in values
            ]
            held_out_names = {
                value
                for other in ("train", "validation", "test")
                if other != split
                for variants in NAME_VARIANTS[other].values()
                for value in variants
            }
            held_out_names.update(
                name
                for other in ("train", "validation", "test")
                if other != split
                for name in SHARED_ALIASES[other]
            )
            held_out_names.update(
                name
                for other in ("train", "validation", "test")
                if other != split
                for name in AMBIGUOUS_INPUTS[other]
            )
            held_out_names.update(
                name
                for other in ("train", "validation", "test")
                if other != split
                for name in UNKNOWN_INPUTS[other]
            )
            for scenario in scenarios:
                row = self._build(scenario)
                user = row["messages"][1]["content"]
                self.assertEqual(
                    sum(
                        _matches_template(user, template)
                        for values in PROMPT_TEMPLATES[split].values()
                        for template in values
                    ),
                    1,
                )
                self.assertFalse(
                    any(_matches_template(user, template)
                        for template in held_out_templates)
                )
                self.assertIn(scenario.input_name.casefold(), user.casefold())
                if scenario.name_kind != "canonical":
                    self.assertNotIn(scenario.input_name, held_out_names)
                self.assertEqual(filter_row_thinking(row, scenario), 1)

    def test_prompts_state_the_selected_direction_and_relation(self) -> None:
        relation_words = {
            "proximity": ("stronger proximity", "weaker proximity"),
            "similarity": ("greater similarity", "less similarity"),
            "evidence": ("stronger evidence", "weaker evidence"),
            None: ("a stronger relation", "a weaker relation"),
        }
        for split, scenarios in self.scenarios.items():
            for scenario in scenarios:
                row = self._build(scenario)
                user = row["messages"][1]["content"]
                lowered = user.casefold()
                if scenario.form == 1:
                    self.assertIn(scenario.input_name.casefold(), lowered)
                    self.assertTrue(
                        ("lower" in lowered and "higher" in lowered)
                        or "direction" in lowered,
                        user,
                    )
                    continue
                direction = (
                    "lower"
                    if scenario.statement_polarity.startswith("lower_")
                    else "higher"
                )
                meaning = "proximity" if scenario.form == 5 else scenario.meaning
                stronger, weaker = relation_words[meaning]
                relation = (
                    stronger
                    if scenario.statement_polarity.endswith("_stronger")
                    else weaker
                )
                statement = (
                    f'a {direction} value for "{scenario.input_name}" means '
                    f"{relation}"
                )
                self.assertIn(statement.casefold(), lowered, (split, scenario))

    def test_thinking_filter_rejects_direction_and_runtime_mutations(self) -> None:
        scenario = self._first(2)
        row = self._build(scenario)
        mutations = (
            "",
            "I will follow the answer schema.",
            row["messages"][-1]["thinking"] + " The opposite direction also holds.",
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
        self.assertEqual(flattened["question_family"], "direction_conventions")
        self.assertEqual(flattened["record_id"], row["row_id"])
        self.assertEqual(flattened["fact_id"], _expected_split_unit(scenario))
        tokenized = tokenize_v3_trace_for_sft(row, _Tokenizer(), max_length=64)
        self.assertEqual(tokenized["input_ids"], [1, 10, 20, 60, 61])
        self.assertEqual(tokenized["completion_mask"], [0, 0, 0, 1, 1])


class WorldModelS18AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings = cls.contract["curriculum"]["families"]["S1.8"]
        cls.tools = tool_definitions(cls.contract)
        cls.scenarios = select_scenarios(
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

    def _audit(self, rows: dict[str, list[dict[str, object]]] | None = None) -> dict[str, object]:
        return audit_rows(
            self.rows if rows is None else rows,
            self.scenarios,
            EXPECTED_TRACE_COUNTS,
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
                "train": {"ambiguous": 400, "not_found": 400, "resolved": 3200},
                "validation": {"ambiguous": 50, "not_found": 50, "resolved": 400},
                "test": {"ambiguous": 50, "not_found": 50, "resolved": 400},
            },
        )
        self.assertEqual(
            audit["tool_call_counts"],
            {
                "train": {"0": 4000},
                "validation": {"0": 500},
                "test": {"0": 500},
            },
        )
        self.assertEqual(
            audit["framing_counts"],
            {
                "train": {"direct": 2000, "step": 2000},
                "validation": {"direct": 250, "step": 250},
                "test": {"direct": 250, "step": 250},
            },
        )
        self.assertEqual(
            audit["semantic_bucket_counts"],
            {
                "train": {
                    "ambiguous_metric": 400,
                    "direction_lookup": 1200,
                    "true_false": 2000,
                    "unrecognized_metric": 400,
                },
                "validation": {
                    "ambiguous_metric": 50,
                    "direction_lookup": 150,
                    "true_false": 250,
                    "unrecognized_metric": 50,
                },
                "test": {
                    "ambiguous_metric": 50,
                    "direction_lookup": 150,
                    "true_false": 250,
                    "unrecognized_metric": 50,
                },
            },
        )
        self.assertEqual(
            audit["name_kind_counts"],
            {
                "train": {
                    "ambiguous": 400,
                    "canonical": 2240,
                    "metric_variant": 640,
                    "shared_alias": 320,
                    "unknown": 400,
                },
                "validation": {
                    "ambiguous": 50,
                    "canonical": 280,
                    "metric_variant": 80,
                    "shared_alias": 40,
                    "unknown": 50,
                },
                "test": {
                    "ambiguous": 50,
                    "canonical": 280,
                    "metric_variant": 80,
                    "shared_alias": 40,
                    "unknown": 50,
                },
            },
        )
        self.assertEqual(
            audit["statement_truth_counts"],
            {
                "train": {"false": 1000, "true": 1000},
                "validation": {"false": 125, "true": 125},
                "test": {"false": 125, "true": 125},
            },
        )
        self.assertTrue(audit["per_form_gates"]["pass"])
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
            self.assertTrue(prose["splits"][split]["pass"])
        self.assertTrue(prose["pass"])
        self.assertTrue(audit["masked_wording_classifier"]["pass"])
        self.assertTrue(audit["thinking_filter"]["pass"])
        self.assertTrue(audit["duplicate_prompt_tool_outcome"]["pass"])

    def test_audit_rejects_direction_truth_and_defer_mutations(self) -> None:
        mutations: list[tuple[str, dict[str, list[dict[str, object]]]]] = []

        direction_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 1 and item.metric is not None
        )
        changed_direction = deepcopy(self.rows)
        direction_result = _answer(changed_direction["train"][direction_index])["result"]
        direction_result["lower_is_stronger"] = not direction_result["lower_is_stronger"]
        mutations.append(("direction", changed_direction))

        truth_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form in {2, 3, 4}
        )
        changed_truth = deepcopy(self.rows)
        truth_result = _answer(changed_truth["train"][truth_index])["result"]
        truth_result["statement_true"] = not truth_result["statement_true"]
        mutations.append(("truth", changed_truth))

        ambiguous_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 5
        )
        changed_ambiguous = deepcopy(self.rows)
        ambiguous_result = _answer(changed_ambiguous["train"][ambiguous_index])["result"]
        ambiguous_result["lower_is_stronger"] = True
        mutations.append(("ambiguous direction", changed_ambiguous))

        unknown_index = next(
            index
            for index, item in enumerate(self.scenarios["train"])
            if item.form == 6
        )
        changed_unknown = deepcopy(self.rows)
        unknown_result = _answer(changed_unknown["train"][unknown_index])["result"]
        unknown_result["metric"] = "spearman_distance"
        mutations.append(("unknown assumption", changed_unknown))

        for name, rows in mutations:
            with self.subTest(name=name):
                with self.assertRaises(CorpusBuildError):
                    self._audit(rows)

    def test_duplicate_audit_rejects_an_ordinary_copy(self) -> None:
        scenario = self.scenarios["train"][0]
        duplicate_scenario = replace(scenario, occurrence=100_000)
        row = deepcopy(self.rows["train"][0])
        duplicate = deepcopy(row)
        duplicate["row_id"] = "wm3_s1_8_duplicate"
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


class WorldModelS18GeneratedCorpusTests(unittest.TestCase):
    def _check_source_records(
        self, manifest: dict[str, object], *, exact_shared: bool
    ) -> None:
        sources = manifest["sources"]
        self.assertEqual(
            set(sources),
            {
                "contract",
                "builder",
                "s1_6_builder",
                "golden_traces",
                "question_classes",
                "coordinate_spec",
                "convention_policy",
            },
        )
        # Published manifests pin the builders used then. Fresh builds below
        # must pin today's sources; historical corpora retain their old pins.
        mutable = {
            "contract", "builder", "s1_6_builder", "golden_traces",
            "question_classes", "coordinate_spec",
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
        self.assertEqual(
            Path(sources["contract"]["path"]),
            Path("runtime/world_model/contract.yaml"),
        )
        self.assertEqual(
            Path(sources["builder"]["path"]),
            Path("runtime/world_model/s1_8_corpus.py"),
        )
        self.assertEqual(
            Path(sources["s1_6_builder"]["path"]),
            Path("runtime/world_model/s1_6_corpus.py"),
        )
        self.assertEqual(
            sources["convention_policy"],
            "use_only_the_closed_metric_table_and_split_specific_name_pools",
        )

    def test_build_is_deterministic_and_validates_before_publication(self) -> None:
        contract = load_contract()
        real_validate = s1_8_module._validate_corpus
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
                s1_8_module, "_validate_corpus", side_effect=validate_before_publish
            ):
                first = build_corpus(first_root, contract=contract, validate=True)
                active_root[0] = second_root
                second = build_corpus(second_root, contract=contract, validate=True)

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
            self._check_source_records(first, exact_shared=True)

    def test_generated_manifest_and_rows_reconcile_when_present(self) -> None:
        # The current builders publish S1 v3, which trains on the pinned
        # prompt bank; the v1 directory predates prompt-bank support.
        contract, overlay = apply_s1_overlay(load_contract(), S1_V3_OVERLAY_PATH)
        settings = contract["curriculum"]["families"]["S1.8"]
        pin = overlay["prompt_banks"]
        prompts = wrapper_prompts(pin, PROMPT_TEMPLATES, bank_name="metric")
        name_variants, alias_cards = metric_names(
            pin, NAME_VARIANTS, SHARED_ALIASES
        )
        root = Path(settings["output_directory"])
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"generated corpus is absent: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(manifest), EXPECTED_MANIFEST_KEYS | {"s1_overlay", "prompt_bank"}
        )
        self.assertEqual(manifest["prompt_bank"]["sha256"], pin["sha256"])
        self.assertEqual(
            manifest["schema_version"], "mentor-rl-world-model-s1-8-manifest-v1"
        )
        self.assertEqual(
            manifest["corpus_schema_version"],
            "mentor-rl-world-model-s1-8-corpus-v1",
        )
        self.assertEqual(manifest["section"], "S1.8")
        self.assertEqual(manifest["dataset_id"], settings["dataset_id"])
        self.assertEqual(manifest["seed"], 1729)
        self.assertEqual(
            manifest["row_counts"], {"train": 4000, "validation": 500, "test": 500}
        )
        self.assertEqual(
            manifest["form_counts"],
            {
                split: {
                    f"S1.8.{form}": count for form, count in counts.items()
                }
                for split, counts in EXPECTED_TRACE_COUNTS.items()
            },
        )
        self.assertEqual(
            manifest["contract_s1_8_sha256"], _stable_sha256(settings)
        )
        self.assertEqual(
            manifest["prompt_templates_sha256"],
            _stable_sha256(prompts),
        )
        self.assertEqual(
            manifest["request_templates_sha256"],
            _stable_sha256(
                {
                    "direction": s1_8_module.DIRECTION_REQUESTS,
                    "true_false": s1_8_module.TRUE_FALSE_REQUESTS,
                }
            ),
        )
        self.assertEqual(
            manifest["metric_conventions_sha256"],
            _stable_sha256(METRIC_CONVENTIONS),
        )
        self.assertEqual(
            manifest["name_sources_sha256"],
            _stable_sha256(
                {
                    "name_variants": name_variants,
                    "shared_aliases": alias_cards,
                    "ambiguous_inputs": AMBIGUOUS_INPUTS,
                    "unknown_inputs": UNKNOWN_INPUTS,
                }
            ),
        )
        self._check_source_records(manifest, exact_shared=False)
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


if __name__ == "__main__":
    unittest.main()
