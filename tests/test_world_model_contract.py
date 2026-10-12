"""Regression tests for the consolidated S0/S1 world-model contract."""

from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from runtime.world_model.contract import (
    ContractError,
    GOLDEN_TRACES_PATH,
    REPOSITORY_ROOT,
    check_golden,
    load_contract,
    parse_layer_ref,
    verify_source_inputs,
)


class WorldModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.rows = check_golden(cls.contract)
        cls.by_trace = {
            row["metadata"]["source_trace"]: row for row in cls.rows
        }

    def test_coordinate_system_pins_current_hpc_inputs(self) -> None:
        coordinate = self.contract["coordinate_system"]
        self.assertEqual(coordinate["layer_count"], 353)
        self.assertEqual(
            coordinate["identifier_registry"]["source_sha256"],
            "ed992f0eac7197d9627bda618f8f831ba355c95bd5d0796af785387d462828b6",
        )
        self.assertEqual(
            coordinate["source_flist_path"],
            "/lustre/orion/syb111/proj-shared/Data/H_sap/"
            "mentor-ev_multiplexes/brain_multiplexes/full_brain/"
            "full_brain_flist.tsv",
        )
        self.assertEqual(
            coordinate["mentor_ev"]["source_directory"],
            "/lustre/orion/syb111/proj-shared/Projects/mentor-ev/brain/full_brain",
        )
        self.assertEqual(
            coordinate["mentor_ev"]["module_id_pattern"],
            r"^mentor_ev:full_brain_all_genes_v1:clade_[0-9]{6}$",
        )
        self.assertEqual(
            self.contract["source_index"]["content_sha256"],
            "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e",
        )

    def test_contract_contains_the_settled_wire_decisions(self) -> None:
        self.assertEqual(
            self.contract["answer"]["keys"],
            [
                "status",
                "provenance",
                "result",
                "continuation_decision",
                "blocked_reason",
                "summary",
            ],
        )
        vocabulary = self.contract["vocabularies"]
        self.assertIn("ambiguous_metric", vocabulary["blocked_reason_code"])
        self.assertIn("unknown_multiplex", vocabulary["blocked_reason_code"])
        self.assertIn("MULTIPLEX", vocabulary["anchor_type"])
        self.assertEqual(
            self.contract["provenance"]["channels"], ["registry", "graph"]
        )
        self.assertEqual(
            self.contract["runtime"]["row_schema_version"], "multiplex_sft_v3"
        )
        self.assertEqual(
            self.contract["runtime"]["tool_contract_version"],
            "mentor-rl-world-model-s0-identifier-tools-v3",
        )
        id_schema = self.rows[0]["tools"][1]["function"]["parameters"][
            "properties"
        ]["gene_id"]
        self.assertNotIn("pattern", id_schema)

    def test_s1_1_membership_mix_is_explicit(self) -> None:
        settings = self.contract["curriculum"]["families"]["S1.1"]
        shares = settings["shares"]
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertEqual(shares["membership_present"], 0.10)
        self.assertEqual(shares["membership_absent"], 0.07)
        self.assertEqual(shares["unknown_multiplex"], 0.03)
        self.assertEqual(
            settings["failure_injection"],
            {
                "source_form": "direct",
                "allocation": "replace",
                "call_stage": "first",
            },
        )
        self.assertEqual(
            settings["unknown_multiplex_id"],
            "retired_brain_multiplex_v0",
        )

    def test_s1_2_build_mix_is_explicit(self) -> None:
        settings = self.contract["curriculum"]["families"]["S1.2"]
        self.assertEqual(
            settings,
            {
                "dataset_id": "world_model_v3_s1_2_ensembl_to_symbol",
                "question_family": "human_ensembl_to_symbol",
                "book_mode": "tool_call",
                "output_directory": "data/world_model_v3/corpora/s1/s1_2",
                "assignment_seed": 1729,
                "train_rows": 11000,
                "base_train_rows": 10000,
                "failure_injection": {
                    "source_form": "direct",
                    "allocation": "replace",
                    "call_stage": "first",
                },
                "shares": {
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
                "added_form_rows": {"transcript_wrong_type": 1000},
            },
        )
        self.assertEqual(
            settings["base_train_rows"]
            + sum(settings["added_form_rows"].values()),
            settings["train_rows"],
        )

    def test_s1_2_build_mix_rejects_drift(self) -> None:
        mutations = {
            "dataset": lambda settings: settings.__setitem__(
                "dataset_id", "changed"
            ),
            "base rows": lambda settings: settings.__setitem__(
                "base_train_rows", 9999
            ),
            "share": lambda settings: settings["shares"].__setitem__(
                "direct", 0.19
            ),
            "share order": lambda settings: settings.__setitem__(
                "shares", dict(reversed(settings["shares"].items()))
            ),
            "added form": lambda settings: settings.__setitem__(
                "added_form_rows", {"transcript_wrong_type": 999}
            ),
            "failure policy": lambda settings: settings[
                "failure_injection"
            ].__setitem__("allocation", "add"),
            "extra setting": lambda settings: settings.__setitem__(
                "unexpected", True
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = copy.deepcopy(self.contract)
                settings = payload["curriculum"]["families"]["S1.2"]
                mutate(settings)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "contract.yaml"
                    path.write_text(yaml.safe_dump(payload, sort_keys=False))
                    with self.assertRaisesRegex(ContractError, "S1.2"):
                        load_contract(path)

    def test_s1_4_build_contract_is_explicit(self) -> None:
        settings = self.contract["curriculum"]["families"]["S1.4"]
        self.assertEqual(
            settings,
            {
                "dataset_id": "world_model_v3_s1_4_cross_context_alignment",
                "question_family": "cross_context_alignment",
                "book_mode": "tool_call",
                "output_directory": "data/world_model_v3/corpora/s1/s1_4",
                "assignment_seed": 1729,
                "train_rows": 8000,
                "allocations": {
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
                "ambiguous_outcome_counts": {
                    "candidate": 500,
                    "noncandidate": 500,
                },
                "maximum_candidate_gene_ids": 49,
                "mygene_source": {
                    "snapshot_id": "mygene_query_cache_2026_04_15",
                    "source_path": (
                        "data/runtime/alias_registry/mygene_query_cache.json"
                    ),
                    "source_url": "https://mygene.info/v3/query",
                    "source_sha256": (
                        "44f2fbcea56595caea5f59651532cfea50ce033296ce441e3b539a9f5859740c"
                    ),
                    "query_count": 33086,
                    "usable_mapping_count": 4596,
                },
                "failure_injection": {
                    "source_form": "symbol_id_match",
                    "allocation": "replace",
                    "call_stage": "first",
                },
            },
        )
        allocations = settings["allocations"]
        self.assertEqual(sum(allocations.values()), settings["train_rows"])
        self.assertEqual(
            sum(settings["ambiguous_outcome_counts"].values()),
            allocations["ambiguous_symbol"],
        )
        self.assertEqual(
            allocations["symbol_id_match"]
            + allocations["backend_unavailable"],
            1000,
        )
        tool_call_rows = settings["train_rows"] - sum(
            allocations[name]
            for name in (
                "id_notation_match",
                "id_near_miss",
                "registry_block_match",
            )
        )
        self.assertEqual(
            allocations["backend_unavailable"] / tool_call_rows,
            self.contract["curriculum"]["tool_failure_share"],
        )

    def test_s1_4_build_contract_rejects_drift(self) -> None:
        mutations = {
            "dataset": lambda settings: settings.__setitem__(
                "dataset_id", "changed"
            ),
            "rows": lambda settings: settings.__setitem__("train_rows", 8001),
            "allocation value": lambda settings: settings[
                "allocations"
            ].__setitem__("symbol_id_match", 749),
            "balanced allocation drift": lambda settings: (
                settings["allocations"].__setitem__("symbol_id_match", 751),
                settings["allocations"].__setitem__("backend_unavailable", 249),
            ),
            "allocation order": lambda settings: settings.__setitem__(
                "allocations", dict(reversed(settings["allocations"].items()))
            ),
            "ambiguous outcome": lambda settings: (
                settings["ambiguous_outcome_counts"].__setitem__(
                    "candidate", 499
                ),
                settings["ambiguous_outcome_counts"].__setitem__(
                    "noncandidate", 501
                ),
            ),
            "ambiguous outcome order": lambda settings: settings.__setitem__(
                "ambiguous_outcome_counts",
                dict(reversed(settings["ambiguous_outcome_counts"].items())),
            ),
            "candidate cap": lambda settings: settings.__setitem__(
                "maximum_candidate_gene_ids", 50
            ),
            "MyGene hash": lambda settings: settings[
                "mygene_source"
            ].__setitem__("source_sha256", "0" * 64),
            "MyGene count": lambda settings: settings[
                "mygene_source"
            ].__setitem__("usable_mapping_count", 4595),
            "MyGene source order": lambda settings: settings.__setitem__(
                "mygene_source",
                dict(reversed(settings["mygene_source"].items())),
            ),
            "failure policy": lambda settings: settings[
                "failure_injection"
            ].__setitem__("allocation", "add"),
            "failure policy order": lambda settings: settings.__setitem__(
                "failure_injection",
                dict(reversed(settings["failure_injection"].items())),
            ),
            "extra setting": lambda settings: settings.__setitem__(
                "unexpected", True
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = copy.deepcopy(self.contract)
                settings = payload["curriculum"]["families"]["S1.4"]
                mutate(settings)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "contract.yaml"
                    path.write_text(yaml.safe_dump(payload, sort_keys=False))
                    with self.assertRaisesRegex(ContractError, "S1.4"):
                        load_contract(path)

    def test_s1_6_build_contract_is_explicit(self) -> None:
        settings = self.contract["curriculum"]["families"]["S1.6"]
        self.assertEqual(
            settings,
            {
                "dataset_id": "world_model_v3_s1_6_layer_tag_parsing",
                "question_family": "layer_tag_parsing",
                "book_mode": "closed_book",
                "output_directory": "data/world_model_v3/corpora/s1/s1_6",
                "assignment_seed": 1729,
                "train_rows": 5000,
                "allocations": {
                    "bulk_pen_direct": 1250,
                    "scpen_one_field": 1250,
                    "humannet_direct": 625,
                    "list_filter": 750,
                    "wrong_input_type": 500,
                    "malformed_tag": 500,
                    "tf_target_direct": 125,
                },
                "evaluation_allocations": {
                    "bulk_pen_direct": 167,
                    "scpen_one_field": 156,
                    "humannet_direct": 83,
                    "list_filter": 94,
                    "wrong_input_type": 63,
                    "malformed_tag": 62,
                    "tf_target_direct": 0,
                },
                "semantic_buckets": {
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
                "one_field_names": [
                    "layer_family",
                    "source",
                    "network",
                    "organ",
                    "region_path",
                    "cell_type_path",
                ],
                "list_filter": {
                    "list_length": 3,
                    "fields": [
                        "layer_family",
                        "source",
                        "network",
                        "organ",
                        "region_path",
                        "cell_type_path",
                    ],
                    "path_match": "exact_component",
                    "training_outcome_counts": {
                        "match": 375,
                        "no_match": 375,
                    },
                },
                "wrong_input_type": {
                    "found_input_type": "module_id",
                    "module_namespace_counts": {
                        "rwr_loe": 250,
                        "mentor_ev": 250,
                    },
                },
                "malformed_subtype_counts": {
                    "unrecognized_prefix": 84,
                    "wrong_arity": 84,
                    "missing_path": 83,
                    "excess_path": 83,
                    "synthetic_field": 83,
                    "mixed_separator": 83,
                },
                "context_split": {
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
            },
        )
        self.assertEqual(sum(settings["allocations"].values()), 5000)
        self.assertEqual(
            sum(settings["evaluation_allocations"].values()), 625
        )

    def test_s1_6_build_contract_rejects_drift(self) -> None:
        mutations = {
            "dataset": lambda settings: settings.__setitem__(
                "dataset_id", "changed"
            ),
            "allocation": lambda settings: settings[
                "allocations"
            ].__setitem__("bulk_pen_direct", 1249),
            "allocation order": lambda settings: settings.__setitem__(
                "allocations", dict(reversed(settings["allocations"].items()))
            ),
            "evaluation": lambda settings: settings[
                "evaluation_allocations"
            ].__setitem__("bulk_pen_direct", 166),
            "semantic bucket": lambda settings: settings[
                "semantic_buckets"
            ]["direct_parse"].pop(),
            "one field": lambda settings: settings[
                "one_field_names"
            ].append("layer_id"),
            "filter policy": lambda settings: settings[
                "list_filter"
            ].__setitem__("path_match", "substring"),
            "wrong type": lambda settings: settings[
                "wrong_input_type"
            ].__setitem__("found_input_type", "context_id"),
            "malformed subtype": lambda settings: settings[
                "malformed_subtype_counts"
            ].__setitem__("mixed_separator", 82),
            "split policy": lambda settings: settings[
                "context_split"
            ].__setitem__("singleton_policy", "reuse"),
            "extra setting": lambda settings: settings.__setitem__(
                "unexpected", True
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = copy.deepcopy(self.contract)
                settings = payload["curriculum"]["families"]["S1.6"]
                mutate(settings)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "contract.yaml"
                    path.write_text(yaml.safe_dump(payload, sort_keys=False))
                    with self.assertRaisesRegex(ContractError, "S1.6"):
                        load_contract(path)

    def test_s1_7_build_contract_is_explicit(self) -> None:
        settings = self.contract["curriculum"]["families"]["S1.7"]
        self.assertEqual(
            settings,
            {
                "dataset_id": "world_model_v3_s1_7_layer_family_claims",
                "question_family": "layer_family_claims",
                "book_mode": "closed_book",
                "output_directory": "data/world_model_v3/corpora/s1/s1_7",
                "assignment_seed": 1729,
                "train_rows": 6000,
                "allocations": {
                    "scpen_family": 1500,
                    "list_filter": 900,
                    "coexpression_claims": 375,
                    "claim_check": 1500,
                    "wrong_input_type": 300,
                    "ppi_claims": 375,
                    "bulk_pen_claims": 375,
                    "other_claims": 300,
                    "tf_target_claims": 375,
                },
                "evaluation_allocations": {
                    "scpen_family": 188,
                    "list_filter": 113,
                    "coexpression_claims": 47,
                    "claim_check": 187,
                    "wrong_input_type": 37,
                    "ppi_claims": 47,
                    "bulk_pen_claims": 47,
                    "other_claims": 37,
                    "tf_target_claims": 47,
                },
                "semantic_buckets": {
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
                "claim_vocabulary": [
                    "context_predictive_expression",
                    "physical_interaction_recorded",
                    "regulatory_link_recorded",
                    "regulation_direction",
                    "global_coexpression",
                    "cofunctional_association",
                    "holds_in_other_contexts",
                    "causal_effect",
                ],
                "family_cards": {
                    "ppi": {
                        "evidence_type": "physical_interaction_recorded",
                        "supported_claims": [
                            "physical_interaction_recorded"
                        ],
                    },
                    "coexpression": {
                        "evidence_type": "global_coexpression",
                        "supported_claims": ["global_coexpression"],
                    },
                    "tf_target": {
                        "evidence_type": "regulatory_link_recorded",
                        "supported_claims": ["regulatory_link_recorded"],
                    },
                    "bulk_pen": {
                        "evidence_type": "context_predictive_expression",
                        "supported_claims": [
                            "context_predictive_expression"
                        ],
                    },
                    "scpen": {
                        "evidence_type": "context_predictive_expression",
                        "supported_claims": [
                            "context_predictive_expression"
                        ],
                    },
                    "other": {
                        "evidence_type": None,
                        "supported_claims": [],
                    },
                },
                "family_name_policy": {
                    "shared_closed_vocabulary": True,
                    "result_layer_id": None,
                },
                "list_filter": {
                    "list_length": 3,
                    "training_family_counts": {
                        "ppi": 150,
                        "coexpression": 150,
                        "tf_target": 150,
                        "bulk_pen": 150,
                        "scpen": 150,
                        "other": 150,
                    },
                    "evaluation_family_counts": {
                        "ppi": 0,
                        "coexpression": 0,
                        "tf_target": 0,
                        "bulk_pen": 38,
                        "scpen": 38,
                        "other": 37,
                    },
                    "training_outcome_counts": {
                        "match": 450,
                        "no_match": 450,
                    },
                    "evaluation_outcome_counts": {
                        "match": 57,
                        "no_match": 56,
                    },
                },
                "claim_cards": {
                    "training_family_name_counts": {
                        "coexpression": 75,
                        "ppi": 75,
                        "bulk_pen": 75,
                        "tf_target": 75,
                    },
                    "evaluation_family_name_counts": {
                        "coexpression": 47,
                        "ppi": 47,
                        "bulk_pen": 0,
                        "tf_target": 47,
                    },
                },
                "claim_check": {
                    "scope_policy": "broad_claim_only",
                    "excluded_scoped_negative": "bulk_pen_single_cell",
                    "unsupported_allocation": (
                        "largest_remainder_in_claim_order"
                    ),
                    "required_unsupported_pairs": {
                        "coexpression": [
                            "physical_interaction_recorded"
                        ],
                        "ppi": ["regulatory_link_recorded"],
                        "tf_target": ["regulation_direction"],
                        "bulk_pen": ["holds_in_other_contexts"],
                        "scpen": ["holds_in_other_contexts"],
                        "every_family": ["causal_effect"],
                    },
                    "training_family_counts": {
                        "ppi": 250,
                        "coexpression": 250,
                        "tf_target": 250,
                        "bulk_pen": 250,
                        "scpen": 250,
                        "other": 250,
                    },
                    "training_family_name_counts": {
                        "ppi": 50,
                        "coexpression": 50,
                        "tf_target": 50,
                        "bulk_pen": 50,
                        "scpen": 50,
                        "other": 50,
                    },
                    "training_supported_counts": {
                        "ppi": 120,
                        "coexpression": 120,
                        "tf_target": 120,
                        "bulk_pen": 120,
                        "scpen": 120,
                        "other": 0,
                    },
                    "training_outcome_counts": {
                        "supported": 600,
                        "unsupported": 900,
                    },
                    "evaluation_family_counts": {
                        "ppi": 32,
                        "coexpression": 31,
                        "tf_target": 31,
                        "bulk_pen": 31,
                        "scpen": 31,
                        "other": 31,
                    },
                    "evaluation_family_name_counts": {
                        "ppi": 32,
                        "coexpression": 31,
                        "tf_target": 31,
                        "bulk_pen": 0,
                        "scpen": 0,
                        "other": 0,
                    },
                    "evaluation_supported_counts": {
                        "ppi": 15,
                        "coexpression": 15,
                        "tf_target": 15,
                        "bulk_pen": 15,
                        "scpen": 15,
                        "other": 0,
                    },
                    "evaluation_outcome_counts": {
                        "supported": 75,
                        "unsupported": 112,
                    },
                },
                "wrong_input_type": {
                    "found_input_type": "module_id",
                    "module_namespace": "mentor_ev",
                    "training_count": 300,
                    "evaluation_count": 37,
                },
                "other_network": {
                    "family": "other",
                    "native_only": True,
                    "continuation_decision": "by_framing",
                    "source_counts": {
                        "train": 3,
                        "validation": 1,
                        "test": 1,
                    },
                },
                "context_split": {
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
                    "singleton_tag_policy": "train_only",
                },
            },
        )
        self.assertEqual(sum(settings["allocations"].values()), 6000)
        self.assertEqual(
            sum(settings["evaluation_allocations"].values()), 750
        )

    def test_s1_7_build_contract_rejects_drift(self) -> None:
        mutations = {
            "dataset": lambda settings: settings.__setitem__(
                "dataset_id", "changed"
            ),
            "allocation": lambda settings: settings[
                "allocations"
            ].__setitem__("scpen_family", 1499),
            "allocation order": lambda settings: settings.__setitem__(
                "allocations", dict(reversed(settings["allocations"].items()))
            ),
            "evaluation": lambda settings: settings[
                "evaluation_allocations"
            ].__setitem__("scpen_family", 187),
            "semantic bucket": lambda settings: settings[
                "semantic_buckets"
            ]["claims_card"].pop(),
            "claim vocabulary": lambda settings: settings[
                "claim_vocabulary"
            ].reverse(),
            "family card": lambda settings: settings["family_cards"][
                "ppi"
            ].__setitem__("supported_claims", []),
            "family-name policy": lambda settings: settings[
                "family_name_policy"
            ].__setitem__("result_layer_id", "required"),
            "list filter": lambda settings: settings["list_filter"][
                "evaluation_family_counts"
            ].__setitem__("other", 36),
            "claim-card input": lambda settings: settings["claim_cards"][
                "training_family_name_counts"
            ].__setitem__("ppi", 74),
            "claim scope": lambda settings: settings["claim_check"].__setitem__(
                "scope_policy", "scoped_claims"
            ),
            "claim outcome": lambda settings: settings["claim_check"][
                "training_outcome_counts"
            ].__setitem__("supported", 599),
            "wrong type": lambda settings: settings[
                "wrong_input_type"
            ].__setitem__("module_namespace", "rwr_loe"),
            "other source": lambda settings: settings["other_network"][
                "source_counts"
            ].__setitem__("train", 2),
            "split policy": lambda settings: settings[
                "context_split"
            ].__setitem__("singleton_tag_policy", "reuse"),
            "extra setting": lambda settings: settings.__setitem__(
                "unexpected", True
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = copy.deepcopy(self.contract)
                settings = payload["curriculum"]["families"]["S1.7"]
                mutate(settings)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "contract.yaml"
                    path.write_text(yaml.safe_dump(payload, sort_keys=False))
                    with self.assertRaisesRegex(ContractError, "S1.7"):
                        load_contract(path)

    def test_s1_8_build_contract_is_explicit(self) -> None:
        settings = self.contract["curriculum"]["families"]["S1.8"]

        self.assertEqual(
            settings["allocations"],
            {
                "direction_lookup": 1200,
                "true_false_distance_style": 667,
                "true_false_similarity_style": 667,
                "true_false_workflow_style": 666,
                "ambiguous_percentile": 400,
                "unrecognized_metric": 400,
            },
        )
        self.assertEqual(
            settings["evaluation_allocations"],
            {
                "direction_lookup": 150,
                "true_false_distance_style": 84,
                "true_false_similarity_style": 83,
                "true_false_workflow_style": 83,
                "ambiguous_percentile": 50,
                "unrecognized_metric": 50,
            },
        )
        self.assertEqual(len(settings["metric_conventions"]), 13)
        self.assertEqual(
            list(settings["metric_conventions"]),
            [
                "rank",
                "score",
                "spearman_distance",
                "pearson_distance",
                "spearman_correlation",
                "pearson_correlation",
                "dot_similarity",
                "distance_percentile",
                "score_percentile",
                "hop_count",
                "empirical_p_value",
                "q",
                "enrichment_score",
            ],
        )
        self.assertEqual(
            settings["shared_aliases"]["train"]["distance"],
            {
                "candidate_metrics": [
                    "spearman_distance",
                    "pearson_distance",
                ],
                "lower_is_stronger": True,
                "meaning": "proximity",
                "rule": "lower_distance_closer",
            },
        )
        self.assertEqual(
            settings["ambiguous_metric"]["candidate_metrics"],
            ["distance_percentile", "score_percentile"],
        )
        self.assertEqual(
            settings["name_policy"]["train_counts"],
            {"canonical": 2240, "metric_variant": 640, "shared_alias": 320},
        )
        self.assertEqual(
            settings["true_false"]["train_truth_counts"],
            {"true": 1000, "false": 1000},
        )
        self.assertEqual(
            settings["status_counts"]["train"],
            {"resolved": 3200, "ambiguous": 400, "not_found": 400},
        )
        self.assertEqual(sum(settings["allocations"].values()), 4000)
        self.assertEqual(
            sum(settings["evaluation_allocations"].values()), 500
        )

    def test_s1_8_build_contract_rejects_drift(self) -> None:
        mutations = {
            "dataset": lambda settings: settings.__setitem__(
                "dataset_id", "changed"
            ),
            "allocation": lambda settings: settings[
                "allocations"
            ].__setitem__("direction_lookup", 1199),
            "evaluation": lambda settings: settings[
                "evaluation_allocations"
            ].__setitem__("direction_lookup", 149),
            "semantic bucket": lambda settings: settings[
                "semantic_buckets"
            ]["true_false"].pop(),
            "source trace": lambda settings: settings[
                "source_trace_by_allocation"
            ].__setitem__("direction_lookup", "S1.8.2"),
            "result shape": lambda settings: settings[
                "result_shape"
            ].__setitem__("candidate_metrics_position", "before_rule"),
            "metric": lambda settings: settings["metric_conventions"][
                "rank"
            ].__setitem__("lower_is_stronger", False),
            "variant": lambda settings: settings["name_variants"]["test"][
                "rank"
            ].pop(),
            "shared alias": lambda settings: settings["shared_aliases"][
                "train"
            ].pop("distance"),
            "name policy": lambda settings: settings["name_policy"][
                "train_counts"
            ].__setitem__("metric_variant", 639),
            "truth": lambda settings: settings["true_false"][
                "train_truth_counts"
            ].__setitem__("true", 999),
            "ambiguous": lambda settings: settings["ambiguous_metric"][
                "candidate_metrics"
            ].reverse(),
            "unknown": lambda settings: settings["unrecognized_metric"][
                "test_inputs"
            ].pop(),
            "status": lambda settings: settings["status_counts"][
                "train"
            ].__setitem__("resolved", 3199),
            "framing": lambda settings: settings["framing_counts"][
                "resolved_decision"
            ].__setitem__("direct", "continue"),
            "extra setting": lambda settings: settings.__setitem__(
                "unexpected", True
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = copy.deepcopy(self.contract)
                settings = payload["curriculum"]["families"]["S1.8"]
                mutate(settings)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "contract.yaml"
                    path.write_text(yaml.safe_dump(payload, sort_keys=False))
                    with self.assertRaisesRegex(ContractError, "S1.8"):
                        load_contract(path)

    def test_s1_9_build_contract_is_explicit(self) -> None:
        settings = self.contract["curriculum"]["families"]["S1.9"]

        self.assertEqual(
            settings["allocations"],
            {"rwr_loe_parse": 1000, "mentor_ev_parse": 1000},
        )
        self.assertEqual(
            settings["evaluation_allocations"],
            {"rwr_loe_parse": 125, "mentor_ev_parse": 125},
        )
        self.assertEqual(
            settings["result_shapes"]["rwr_loe"]["keys"],
            [
                "module_id",
                "module_family",
                "multiplex_id",
                "seed_gene_id",
                "threshold_recipe",
                "existence_asserted",
            ],
        )
        self.assertEqual(
            settings["result_shapes"]["mentor_ev"]["keys"],
            [
                "module_id",
                "module_family",
                "dendrogram_id",
                "clade_id",
                "existence_asserted",
            ],
        )
        self.assertFalse(
            settings["result_shapes"]["rwr_loe"]["existence_asserted"]
        )
        self.assertFalse(
            settings["result_shapes"]["mentor_ev"]["existence_asserted"]
        )
        self.assertEqual(
            settings["source_pools"]["rwr_loe"]["split_counts"],
            {"train": 37613, "validation": 4702, "test": 4702},
        )
        self.assertEqual(
            settings["source_pools"]["mentor_ev"]["split_counts"],
            {"train": 12543, "validation": 1568, "test": 1568},
        )
        self.assertEqual(
            settings["split_policy"]["residues"],
            {
                "train": [2, 3, 4, 5, 6, 7, 8, 9],
                "validation": [0],
                "test": [1],
            },
        )
        self.assertEqual(
            settings["framing_counts"]["train"],
            {"direct": 1000, "step": 1000},
        )
        self.assertEqual(
            settings["framing_counts"]["evaluation"],
            {"direct": 125, "step": 125},
        )
        self.assertFalse(settings["nonpositive_policy"]["included"])

    def test_s1_9_build_contract_rejects_drift(self) -> None:
        mutations = {
            "dataset": lambda settings: settings.__setitem__(
                "dataset_id", "changed"
            ),
            "allocation": lambda settings: settings[
                "allocations"
            ].__setitem__("rwr_loe_parse", 999),
            "evaluation": lambda settings: settings[
                "evaluation_allocations"
            ].__setitem__("mentor_ev_parse", 124),
            "semantic bucket": lambda settings: settings[
                "semantic_buckets"
            ].__setitem__("rwr_loe", ["mentor_ev_parse"]),
            "source trace": lambda settings: settings[
                "source_trace_by_allocation"
            ].__setitem__("rwr_loe_parse", "S1.9.2"),
            "grammar": lambda settings: settings["grammars"][
                "mentor_ev"
            ].__setitem__("clade_digits", 5),
            "result shape": lambda settings: settings["result_shapes"][
                "rwr_loe"
            ]["keys"].remove("existence_asserted"),
            "source pool": lambda settings: settings["source_pools"][
                "mentor_ev"
            ].__setitem__("record_count", 15678),
            "split": lambda settings: settings["split_policy"][
                "residues"
            ].__setitem__("validation", [1]),
            "nonpositive": lambda settings: settings[
                "nonpositive_policy"
            ].__setitem__("included", True),
            "status": lambda settings: settings["status_counts"][
                "train"
            ].__setitem__("resolved", 1999),
            "framing": lambda settings: settings["framing_counts"][
                "train"
            ].__setitem__("direct", 999),
            "extra setting": lambda settings: settings.__setitem__(
                "unexpected", True
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = copy.deepcopy(self.contract)
                settings = payload["curriculum"]["families"]["S1.9"]
                mutate(settings)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "contract.yaml"
                    path.write_text(yaml.safe_dump(payload, sort_keys=False))
                    with self.assertRaisesRegex(ContractError, "S1.9"):
                        load_contract(path)

    def test_layer_parser_returns_the_exact_normalized_fields(self) -> None:
        self.assertEqual(
            parse_layer_ref(
                self.contract,
                "bulkPEN:brain:basal_ganglia:caudate",
            ),
            {
                "layer_id": "bulkPEN:brain:basal_ganglia:caudate",
                "layer_family": "bulk_pen",
                "source": None,
                "network": None,
                "organ": "brain",
                "region_path": ["basal_ganglia", "caudate"],
                "cell_type_path": None,
            },
        )
        self.assertEqual(
            parse_layer_ref(
                self.contract,
                "scPEN:brain:pfc:inhibitory_interneuron:pv",
            )["cell_type_path"],
            ["inhibitory_interneuron", "pv"],
        )
        self.assertEqual(
            parse_layer_ref(self.contract, "HumanNetV3:cocitation")[
                "layer_family"
            ],
            "other",
        )
        self.assertEqual(
            parse_layer_ref(self.contract, "TFs:brain")["layer_family"],
            "tf_target",
        )
        for value in (
            "scPEN:brain::astrocyte",
            "scPEN:brain:hippocampus",
            "unknown:brain",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ContractError):
                    parse_layer_ref(self.contract, value)

    def test_pinned_hpc_sources_match_hashes_and_counts(self) -> None:
        coordinate = self.contract["coordinate_system"]
        registry_spec = coordinate["identifier_registry"]
        paths = [
            Path(coordinate["source_flist_path"]),
            Path(coordinate["mentor_ev"]["source_path"]),
            Path(registry_spec["source_path"]),
            Path(registry_spec["stable_id_history"]["source_path"]),
            Path(registry_spec["hgnc_history"]["source_path"]),
            Path(
                coordinate["out_of_scope_species"]["mouse"]["source_path"]
            ),
            Path(
                self.contract["curriculum"]["families"]["S1.4"][
                    "mygene_source"
                ]["source_path"]
            ),
        ]
        paths = [
            path if path.is_absolute() else REPOSITORY_ROOT / path
            for path in paths
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            self.skipTest(f"pinned source is not mounted: {missing[0]}")
        verify_source_inputs(self.contract)

    def test_golden_rows_cover_new_decision_cases(self) -> None:
        self.assertEqual(len(self.rows), 73)
        for trace in (
            "S1.1.9",
            "S1.1.10",
            "S1.1.11",
            "S1.2.12",
            "S1.4.9",
            "S1.9.1",
            "S1.9.2",
        ):
            self.assertIn(trace, self.by_trace)

        membership_answer = self.by_trace["S1.1.9"]["messages"][-1]["content"]
        self.assertIsNotNone(membership_answer["provenance"]["registry"])
        self.assertIsNotNone(membership_answer["provenance"]["graph"])
        self.assertIn("<MULTIPLEX:", membership_answer["summary"])

        failure = self.by_trace["S1.4.9"]["messages"][-1]["content"]
        self.assertIn("failed_tool", failure["result"])
        self.assertNotIn("tool", failure["result"])

    def test_semantic_validator_accepts_the_golden_rows(self) -> None:
        validator = (
            REPOSITORY_ROOT
            / "agents"
            / "world_model_v3_curriculum"
            / "s1_kit"
            / "validate.py"
        )
        result = subprocess.run(
            [sys.executable, str(validator), str(GOLDEN_TRACES_PATH)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("violations                  : 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
