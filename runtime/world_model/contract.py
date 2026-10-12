"""Load and expose the single S0/S1 world-model training contract.

The YAML file beside this module is the machine-readable source of truth.
This module deliberately contains no second copy of its vocabularies or row
shapes; it only validates the document and derives runtime values from it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .sources import SourceError, file_sha256, read_dendrogram, read_flist


CONTRACT_PATH = Path(__file__).with_name("contract.yaml")
GOLDEN_TRACES_PATH = Path(__file__).with_name("golden_traces.jsonl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ContractError(ValueError):
    """Report a malformed world-model contract or golden trace."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently taking the last one."""


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ContractError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Return the validated world-model contract."""

    try:
        payload = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("the contract root must be a mapping")
    _validate_contract(payload)
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    required_sections = (
        "schema_version",
        "runtime",
        "coordinate_system",
        "source_index",
        "answer",
        "vocabularies",
        "rows",
        "provenance",
        "tools",
        "layer_grammars",
        "blocked_reasons",
        "curriculum",
    )
    missing = [key for key in required_sections if key not in contract]
    if missing:
        raise ContractError(f"contract is missing sections: {', '.join(missing)}")

    runtime = contract["runtime"]
    if runtime.get("row_schema_version") != "multiplex_sft_v3":
        raise ContractError("runtime.row_schema_version must name the v3 wire")
    if runtime.get("tool_contract_version") != (
        "mentor-rl-world-model-s0-identifier-tools-v3"
    ):
        raise ContractError("runtime.tool_contract_version must name the v3 tools")

    answer_keys = contract["answer"].get("keys")
    if answer_keys != [
        "status",
        "provenance",
        "result",
        "continuation_decision",
        "blocked_reason",
        "summary",
    ]:
        raise ContractError("answer.keys is not the frozen six-key wire order")

    channels = contract["provenance"].get("channels")
    if channels != ["registry", "graph"]:
        raise ContractError("provenance.channels must be [registry, graph]")

    callables = contract["tools"].get("callable")
    definitions = contract["tools"].get("definitions", {})
    if not isinstance(callables, list) or not set(callables) <= set(definitions):
        raise ContractError("tools.callable names a missing tool definition")

    vocabularies = contract["vocabularies"]
    reason_codes = set(vocabularies.get("blocked_reason_code", []))
    templates = set(contract["blocked_reasons"].get("codes", {}))
    if reason_codes != templates:
        raise ContractError(
            "blocked_reason_code vocabulary and blocked_reasons.codes disagree"
        )

    fractions = contract["curriculum"].get("split_fractions", {})
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ContractError("curriculum split fractions must sum to 1")

    families = contract["curriculum"].get("families", {})
    for family, settings in families.items():
        shares = settings.get("shares")
        if isinstance(shares, (dict, list)):
            values = shares.values() if isinstance(shares, dict) else shares
            if abs(sum(values) - 1.0) > 1e-9:
                raise ContractError(f"{family} shares must sum to 1")

    s0 = families.get("S0", {})
    s0_rows = s0.get("rows", {})
    s0_train = s0.get("train", {})
    train_count = sum(
        int(s0_train.get(key, 0))
        for key in (
            "registry_symbol_rows",
            "named_gene_id_rows",
            "unnamed_gene_id_rows",
            "historical_symbol_rows",
            "retired_gene_id_rows",
        )
    )
    if train_count != s0_rows.get("train"):
        raise ContractError("S0 train allocations do not match its row count")
    if s0_train.get("ambiguous_symbol_rows", 0) > s0_train.get(
        "registry_symbol_rows", 0
    ):
        raise ContractError("S0 ambiguous rows exceed its symbol allocation")
    for split in ("validation", "test"):
        settings = s0.get("evaluation", {}).get(split, {})
        count = sum(
            int(settings.get(key, 0))
            for key in (
                "registry_rows",
                "historical_symbol_rows",
                "retired_gene_id_rows",
            )
        )
        if count != s0_rows.get(split):
            raise ContractError(
                f"S0 {split} allocations do not match its row count"
            )

    s1_1 = families.get("S1.1", {})
    if not isinstance(s1_1.get("train_rows"), int) or s1_1["train_rows"] < 1:
        raise ContractError("S1.1 train_rows must be a positive integer")
    failure = s1_1.get("failure_injection", {})
    if failure != {
        "source_form": "direct",
        "allocation": "replace",
        "call_stage": "first",
    }:
        raise ContractError("S1.1 failure injection policy changed")
    if s1_1.get("unknown_multiplex_id") == contract["coordinate_system"].get(
        "multiplex_id"
    ):
        raise ContractError("S1.1 unknown multiplex must not be registered")

    s1_2 = families.get("S1.2", {})
    expected_s1_2_keys = [
        "dataset_id",
        "question_family",
        "book_mode",
        "output_directory",
        "assignment_seed",
        "train_rows",
        "base_train_rows",
        "failure_injection",
        "shares",
        "added_form_rows",
    ]
    if not isinstance(s1_2, Mapping) or list(s1_2) != expected_s1_2_keys:
        raise ContractError("S1.2 build setting keys changed")
    expected_s1_2_fields = {
        "dataset_id": "world_model_v3_s1_2_ensembl_to_symbol",
        "question_family": "human_ensembl_to_symbol",
        "book_mode": "tool_call",
        "output_directory": "data/world_model_v3/corpora/s1/s1_2",
        "assignment_seed": 1729,
        "train_rows": 11000,
        "base_train_rows": 10000,
    }
    for key, expected in expected_s1_2_fields.items():
        if s1_2.get(key) != expected:
            raise ContractError(
                f"S1.2 {key} is {s1_2.get(key)!r}, expected {expected!r}"
            )

    expected_s1_2_shares = {
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
    }
    shares = s1_2.get("shares")
    if (
        not isinstance(shares, Mapping)
        or list(shares) != list(expected_s1_2_shares)
        or dict(shares) != expected_s1_2_shares
    ):
        raise ContractError("S1.2 ordered base shares changed")

    added_rows = s1_2.get("added_form_rows")
    expected_added_rows = {"transcript_wrong_type": 1000}
    if added_rows != expected_added_rows:
        raise ContractError("S1.2 added form rows changed")
    if s1_2["base_train_rows"] + sum(added_rows.values()) != s1_2["train_rows"]:
        raise ContractError("S1.2 base and added rows do not match train_rows")

    failure = s1_2.get("failure_injection", {})
    if failure != {
        "source_form": "direct",
        "allocation": "replace",
        "call_stage": "first",
    }:
        raise ContractError("S1.2 failure injection policy changed")

    s1_3 = families.get("S1.3", {})
    expected_s1_3_keys = [
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
    ]
    if not isinstance(s1_3, Mapping) or list(s1_3) != expected_s1_3_keys:
        raise ContractError("S1.3 build setting keys changed")
    expected_s1_3_fields = {
        "dataset_id": "world_model_v3_s1_3_ambiguous_symbol",
        "question_family": "human_ambiguous_symbol",
        "book_mode": "tool_call",
        "output_directory": "data/world_model_v3/corpora/s1/s1_3",
        "assignment_seed": 1729,
        "train_rows": 9000,
        "maximum_candidate_gene_ids": 49,
    }
    for key, expected in expected_s1_3_fields.items():
        if s1_3.get(key) != expected:
            raise ContractError(
                f"S1.3 {key} is {s1_3.get(key)!r}, expected {expected!r}"
            )

    expected_s1_3_allocations = {
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
    }
    allocations = s1_3.get("allocations")
    if (
        not isinstance(allocations, Mapping)
        or list(allocations) != list(expected_s1_3_allocations)
        or dict(allocations) != expected_s1_3_allocations
    ):
        raise ContractError("S1.3 ordered allocations changed")
    if sum(allocations.values()) != s1_3["train_rows"]:
        raise ContractError("S1.3 allocations do not match train_rows")

    if s1_3.get("membership_outcome_counts") != {
        "keeps_one": 371,
        "keeps_several": 39,
        "keeps_none": 74,
    }:
        raise ContractError("S1.3 membership source counts changed")
    if s1_3.get("failure_injection") != {
        "source_forms": [
            "membership_keeps_one",
            "membership_keeps_several",
            "membership_keeps_none",
        ],
        "allocation": "replace",
        "call_stage": "membership",
    }:
        raise ContractError("S1.3 failure injection policy changed")
    no_call_rows = sum(
        allocations[name]
        for name in (
            "already_marked",
            "user_chooses",
            "reply_does_not_choose",
        )
    )
    tool_call_rows = s1_3["train_rows"] - no_call_rows
    failure_rows = allocations["membership_backend_unavailable"]
    if failure_rows / tool_call_rows != contract["curriculum"][
        "tool_failure_share"
    ]:
        raise ContractError("S1.3 tool failure share changed")

    s1_4 = families.get("S1.4", {})
    expected_s1_4_keys = [
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
    ]
    if not isinstance(s1_4, Mapping) or list(s1_4) != expected_s1_4_keys:
        raise ContractError("S1.4 build setting keys changed")
    expected_s1_4_fields = {
        "dataset_id": "world_model_v3_s1_4_cross_context_alignment",
        "question_family": "cross_context_alignment",
        "book_mode": "tool_call",
        "output_directory": "data/world_model_v3/corpora/s1/s1_4",
        "assignment_seed": 1729,
        "train_rows": 8000,
        "maximum_candidate_gene_ids": 49,
    }
    for key, expected in expected_s1_4_fields.items():
        if s1_4.get(key) != expected:
            raise ContractError(
                f"S1.4 {key} is {s1_4.get(key)!r}, expected {expected!r}"
            )

    expected_s1_4_allocations = {
        "symbol_id_match": 750,
        "symbol_id_mismatch": 1000,
        "id_notation_match": 1000,
        "id_near_miss": 1000,
        "historical_symbol_unresolved": 1000,
        "ambiguous_symbol": 1000,
        "registry_block_match": 1000,
        "mygene_id_match": 1000,
        "backend_unavailable": 250,
    }
    allocations = s1_4.get("allocations")
    if (
        not isinstance(allocations, Mapping)
        or list(allocations) != list(expected_s1_4_allocations)
    ):
        raise ContractError("S1.4 allocation keys or order changed")
    if sum(allocations.values()) != s1_4["train_rows"]:
        raise ContractError("S1.4 allocations do not match train_rows")
    if dict(allocations) != expected_s1_4_allocations:
        raise ContractError("S1.4 allocation values changed")

    ambiguous_outcomes = s1_4.get("ambiguous_outcome_counts")
    expected_ambiguous_outcomes = {"candidate": 500, "noncandidate": 500}
    if (
        not isinstance(ambiguous_outcomes, Mapping)
        or list(ambiguous_outcomes) != list(expected_ambiguous_outcomes)
    ):
        raise ContractError("S1.4 ambiguous outcome keys or order changed")
    if sum(ambiguous_outcomes.values()) != allocations["ambiguous_symbol"]:
        raise ContractError(
            "S1.4 ambiguous outcomes do not match the ambiguous allocation"
        )
    if dict(ambiguous_outcomes) != expected_ambiguous_outcomes:
        raise ContractError("S1.4 ambiguous outcome values changed")

    expected_mygene_source = {
        "snapshot_id": "mygene_query_cache_2026_04_15",
        "source_path": "data/runtime/alias_registry/mygene_query_cache.json",
        "source_url": "https://mygene.info/v3/query",
        "source_sha256": (
            "44f2fbcea56595caea5f59651532cfea50ce033296ce441e3b539a9f5859740c"
        ),
        "query_count": 33086,
        "usable_mapping_count": 4596,
    }
    mygene_source = s1_4.get("mygene_source")
    if (
        not isinstance(mygene_source, Mapping)
        or list(mygene_source) != list(expected_mygene_source)
    ):
        raise ContractError("S1.4 MyGene source keys or order changed")
    if mygene_source["usable_mapping_count"] > mygene_source["query_count"]:
        raise ContractError("S1.4 usable MyGene mappings exceed its queries")
    if dict(mygene_source) != expected_mygene_source:
        raise ContractError("S1.4 MyGene source values changed")

    failure = s1_4.get("failure_injection")
    expected_failure = {
        "source_form": "symbol_id_match",
        "allocation": "replace",
        "call_stage": "first",
    }
    if (
        not isinstance(failure, Mapping)
        or list(failure) != list(expected_failure)
        or dict(failure) != expected_failure
    ):
        raise ContractError("S1.4 failure injection policy changed")
    failure_rows = allocations["backend_unavailable"]
    source_rows = allocations[failure["source_form"]]
    if source_rows < failure_rows:
        raise ContractError("S1.4 retained successes cannot pair every failure")
    if source_rows + failure_rows != 1000:
        raise ContractError("S1.4 replacement rows do not restore equal base forms")
    no_call_rows = sum(
        allocations[name]
        for name in (
            "id_notation_match",
            "id_near_miss",
            "registry_block_match",
        )
    )
    tool_call_rows = s1_4["train_rows"] - no_call_rows
    if failure_rows / tool_call_rows != contract["curriculum"][
        "tool_failure_share"
    ]:
        raise ContractError("S1.4 tool failure share changed")

    s1_6 = families.get("S1.6", {})
    expected_s1_6_keys = [
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
    ]
    if not isinstance(s1_6, Mapping) or list(s1_6) != expected_s1_6_keys:
        raise ContractError("S1.6 build setting keys changed")
    expected_s1_6_fields = {
        "dataset_id": "world_model_v3_s1_6_layer_tag_parsing",
        "question_family": "layer_tag_parsing",
        "book_mode": "closed_book",
        "output_directory": "data/world_model_v3/corpora/s1/s1_6",
        "assignment_seed": 1729,
        "train_rows": 5000,
    }
    for key, expected in expected_s1_6_fields.items():
        if s1_6.get(key) != expected:
            raise ContractError(
                f"S1.6 {key} is {s1_6.get(key)!r}, expected {expected!r}"
            )

    expected_s1_6_allocations = {
        "bulk_pen_direct": 1250,
        "scpen_one_field": 1250,
        "humannet_direct": 625,
        "list_filter": 750,
        "wrong_input_type": 500,
        "malformed_tag": 500,
        "tf_target_direct": 125,
    }
    allocations = s1_6.get("allocations")
    if (
        not isinstance(allocations, Mapping)
        or list(allocations) != list(expected_s1_6_allocations)
        or dict(allocations) != expected_s1_6_allocations
    ):
        raise ContractError("S1.6 ordered allocations changed")
    if sum(allocations.values()) != s1_6["train_rows"]:
        raise ContractError("S1.6 allocations do not match train_rows")

    expected_evaluation_allocations = {
        "bulk_pen_direct": 167,
        "scpen_one_field": 156,
        "humannet_direct": 83,
        "list_filter": 94,
        "wrong_input_type": 63,
        "malformed_tag": 62,
        "tf_target_direct": 0,
    }
    evaluation_allocations = s1_6.get("evaluation_allocations")
    if (
        not isinstance(evaluation_allocations, Mapping)
        or list(evaluation_allocations)
        != list(expected_evaluation_allocations)
        or dict(evaluation_allocations) != expected_evaluation_allocations
    ):
        raise ContractError("S1.6 evaluation allocations changed")
    split_fractions = contract["curriculum"]["split_fractions"]
    evaluation_rows = (
        s1_6["train_rows"]
        * split_fractions["validation"]
        / split_fractions["train"]
    )
    if evaluation_rows != 625 or sum(evaluation_allocations.values()) != 625:
        raise ContractError("S1.6 evaluation allocations do not reconcile")

    expected_semantic_buckets = {
        "direct_parse": [
            "bulk_pen_direct",
            "humannet_direct",
            "tf_target_direct",
        ],
        "one_field": ["scpen_one_field"],
        "list_filter": ["list_filter"],
        "wrong_input_type": ["wrong_input_type"],
        "malformed_tag": ["malformed_tag"],
    }
    if s1_6.get("semantic_buckets") != expected_semantic_buckets:
        raise ContractError("S1.6 semantic buckets changed")
    semantic_counts = {
        name: sum(allocations[item] for item in members)
        for name, members in expected_semantic_buckets.items()
    }
    if semantic_counts != {
        "direct_parse": 2000,
        "one_field": 1250,
        "list_filter": 750,
        "wrong_input_type": 500,
        "malformed_tag": 500,
    }:
        raise ContractError("S1.6 semantic shares changed")

    expected_fields = [
        "layer_family",
        "source",
        "network",
        "organ",
        "region_path",
        "cell_type_path",
    ]
    if s1_6.get("one_field_names") != expected_fields:
        raise ContractError("S1.6 one-field names changed")
    expected_list_filter = {
        "list_length": 3,
        "fields": expected_fields,
        "path_match": "exact_component",
        "training_outcome_counts": {"match": 375, "no_match": 375},
    }
    if s1_6.get("list_filter") != expected_list_filter:
        raise ContractError("S1.6 list-filter policy changed")
    if sum(
        s1_6["list_filter"]["training_outcome_counts"].values()
    ) != allocations["list_filter"]:
        raise ContractError("S1.6 list outcomes do not match their allocation")

    expected_wrong_type = {
        "found_input_type": "module_id",
        "module_namespace_counts": {"rwr_loe": 250, "mentor_ev": 250},
    }
    if s1_6.get("wrong_input_type") != expected_wrong_type:
        raise ContractError("S1.6 wrong-input-type policy changed")
    if sum(expected_wrong_type["module_namespace_counts"].values()) != (
        allocations["wrong_input_type"]
    ):
        raise ContractError("S1.6 module namespaces do not reconcile")

    expected_malformed = {
        "unrecognized_prefix": 84,
        "wrong_arity": 84,
        "missing_path": 83,
        "excess_path": 83,
        "synthetic_field": 83,
        "mixed_separator": 83,
    }
    if s1_6.get("malformed_subtype_counts") != expected_malformed:
        raise ContractError("S1.6 malformed-tag subtype counts changed")
    if sum(expected_malformed.values()) != allocations["malformed_tag"]:
        raise ContractError("S1.6 malformed subtypes do not reconcile")

    expected_context_split = {
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
    }
    if s1_6.get("context_split") != expected_context_split:
        raise ContractError("S1.6 context split policy changed")
    if sum(s1_6["context_split"]["counts"].values()) != 35:
        raise ContractError("S1.6 context split counts do not reconcile")
    for split, count in s1_6["context_split"]["counts"].items():
        if sum(s1_6["context_split"]["strata"][split].values()) != count:
            raise ContractError(f"S1.6 {split} context strata do not reconcile")

    s1_7 = families.get("S1.7", {})
    expected_s1_7_keys = [
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
    ]
    if not isinstance(s1_7, Mapping) or list(s1_7) != expected_s1_7_keys:
        raise ContractError("S1.7 build setting keys changed")
    expected_s1_7_scalars = {
        "dataset_id": "world_model_v3_s1_7_layer_family_claims",
        "question_family": "layer_family_claims",
        "book_mode": "closed_book",
        "output_directory": "data/world_model_v3/corpora/s1/s1_7",
        "assignment_seed": 1729,
        "train_rows": 6000,
    }
    for key, expected in expected_s1_7_scalars.items():
        if s1_7.get(key) != expected:
            raise ContractError(
                f"S1.7 {key} is {s1_7.get(key)!r}, expected {expected!r}"
            )

    expected_s1_7_allocations = {
        "scpen_family": 1500,
        "list_filter": 900,
        "coexpression_claims": 375,
        "claim_check": 1500,
        "wrong_input_type": 300,
        "ppi_claims": 375,
        "bulk_pen_claims": 375,
        "other_claims": 300,
        "tf_target_claims": 375,
    }
    allocations_1_7 = s1_7.get("allocations")
    if (
        not isinstance(allocations_1_7, Mapping)
        or list(allocations_1_7) != list(expected_s1_7_allocations)
        or dict(allocations_1_7) != expected_s1_7_allocations
    ):
        raise ContractError("S1.7 ordered allocations changed")
    if sum(allocations_1_7.values()) != s1_7["train_rows"]:
        raise ContractError("S1.7 allocations do not match train_rows")

    expected_s1_7_evaluation = {
        "scpen_family": 188,
        "list_filter": 113,
        "coexpression_claims": 47,
        "claim_check": 187,
        "wrong_input_type": 37,
        "ppi_claims": 47,
        "bulk_pen_claims": 47,
        "other_claims": 37,
        "tf_target_claims": 47,
    }
    evaluation_1_7 = s1_7.get("evaluation_allocations")
    if (
        not isinstance(evaluation_1_7, Mapping)
        or list(evaluation_1_7) != list(expected_s1_7_evaluation)
        or dict(evaluation_1_7) != expected_s1_7_evaluation
    ):
        raise ContractError("S1.7 evaluation allocations changed")
    evaluation_rows_1_7 = (
        s1_7["train_rows"]
        * split_fractions["validation"]
        / split_fractions["train"]
    )
    if evaluation_rows_1_7 != 750 or sum(evaluation_1_7.values()) != 750:
        raise ContractError("S1.7 evaluation allocations do not reconcile")

    expected_s1_7_buckets = {
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
    }
    if s1_7.get("semantic_buckets") != expected_s1_7_buckets:
        raise ContractError("S1.7 semantic buckets changed")
    semantic_1_7 = {
        name: sum(allocations_1_7[item] for item in members)
        for name, members in expected_s1_7_buckets.items()
    }
    if semantic_1_7 != {
        "family_of_layer": 1500,
        "list_filter": 900,
        "claims_card": 1500,
        "claim_check": 1500,
        "wrong_input_type": 300,
        "native_nonmapped_network": 300,
    }:
        raise ContractError("S1.7 semantic shares changed")

    claim_vocabulary = contract["vocabularies"]["layer_claims"]
    if s1_7.get("claim_vocabulary") != claim_vocabulary:
        raise ContractError("S1.7 claim vocabulary changed")
    expected_family_cards = {
        "ppi": {
            "evidence_type": "physical_interaction_recorded",
            "supported_claims": ["physical_interaction_recorded"],
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
            "supported_claims": ["context_predictive_expression"],
        },
        "scpen": {
            "evidence_type": "context_predictive_expression",
            "supported_claims": ["context_predictive_expression"],
        },
        "other": {"evidence_type": None, "supported_claims": []},
    }
    if s1_7.get("family_cards") != expected_family_cards:
        raise ContractError("S1.7 family cards changed")
    if list(s1_7["family_cards"]) != contract["vocabularies"]["layer_family"]:
        raise ContractError("S1.7 family card order changed")
    for family, card in s1_7["family_cards"].items():
        supported = card["supported_claims"]
        if supported != [item for item in claim_vocabulary if item in supported]:
            raise ContractError(f"S1.7 {family} claim order changed")
    if s1_7.get("family_name_policy") != {
        "shared_closed_vocabulary": True,
        "result_layer_id": None,
    }:
        raise ContractError("S1.7 family-name policy changed")

    expected_list_filter_1_7 = {
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
        "training_outcome_counts": {"match": 450, "no_match": 450},
        "evaluation_outcome_counts": {"match": 57, "no_match": 56},
    }
    if s1_7.get("list_filter") != expected_list_filter_1_7:
        raise ContractError("S1.7 list-filter policy changed")
    if sum(expected_list_filter_1_7["training_family_counts"].values()) != 900:
        raise ContractError("S1.7 training list families do not reconcile")
    if sum(expected_list_filter_1_7["evaluation_family_counts"].values()) != 113:
        raise ContractError("S1.7 evaluation list families do not reconcile")
    for split_name in ("training", "evaluation"):
        if sum(
            expected_list_filter_1_7[f"{split_name}_outcome_counts"].values()
        ) != sum(
            expected_list_filter_1_7[f"{split_name}_family_counts"].values()
        ):
            raise ContractError(f"S1.7 {split_name} list outcomes do not reconcile")

    expected_claim_cards = {
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
    }
    if s1_7.get("claim_cards") != expected_claim_cards:
        raise ContractError("S1.7 claim-card input policy changed")

    expected_claim_check = {
        "scope_policy": "broad_claim_only",
        "excluded_scoped_negative": "bulk_pen_single_cell",
        "unsupported_allocation": "largest_remainder_in_claim_order",
        "required_unsupported_pairs": {
            "coexpression": ["physical_interaction_recorded"],
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
        "training_outcome_counts": {"supported": 600, "unsupported": 900},
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
        "evaluation_outcome_counts": {"supported": 75, "unsupported": 112},
    }
    if s1_7.get("claim_check") != expected_claim_check:
        raise ContractError("S1.7 claim-check policy changed")
    for split_name in ("training", "evaluation"):
        family_counts = expected_claim_check[f"{split_name}_family_counts"]
        supported_counts = expected_claim_check[
            f"{split_name}_supported_counts"
        ]
        outcome_counts = expected_claim_check[f"{split_name}_outcome_counts"]
        if sum(family_counts.values()) != sum(outcome_counts.values()):
            raise ContractError(f"S1.7 {split_name} claim rows do not reconcile")
        if sum(supported_counts.values()) != outcome_counts["supported"]:
            raise ContractError(
                f"S1.7 {split_name} supported claims do not reconcile"
            )
        if any(supported_counts[key] > family_counts[key] for key in family_counts):
            raise ContractError(f"S1.7 {split_name} family claims are infeasible")

    if s1_7.get("wrong_input_type") != {
        "found_input_type": "module_id",
        "module_namespace": "mentor_ev",
        "training_count": 300,
        "evaluation_count": 37,
    }:
        raise ContractError("S1.7 wrong-input-type policy changed")
    if s1_7.get("other_network") != {
        "family": "other",
        "native_only": True,
        "continuation_decision": "by_framing",
        "source_counts": {"train": 3, "validation": 1, "test": 1},
    }:
        raise ContractError("S1.7 other-network policy changed")

    expected_context_split_1_7 = {
        "unit": "exact_organ_and_region_path_or_network",
        "counts": {"train": 27, "validation": 4, "test": 4},
        "strata": expected_context_split["strata"],
        "singleton_tag_policy": "train_only",
    }
    if s1_7.get("context_split") != expected_context_split_1_7:
        raise ContractError("S1.7 context split policy changed")
    for split, count in s1_7["context_split"]["counts"].items():
        if sum(s1_7["context_split"]["strata"][split].values()) != count:
            raise ContractError(f"S1.7 {split} context strata do not reconcile")

    s1_8 = families.get("S1.8", {})
    expected_s1_8_keys = [
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
    ]
    if not isinstance(s1_8, Mapping) or list(s1_8) != expected_s1_8_keys:
        raise ContractError("S1.8 build setting keys changed")
    expected_s1_8_scalars = {
        "dataset_id": "world_model_v3_s1_8_direction_conventions",
        "question_family": "direction_conventions",
        "book_mode": "closed_book",
        "output_directory": "data/world_model_v3/corpora/s1/s1_8",
        "assignment_seed": 1729,
        "train_rows": 4000,
    }
    for key, expected in expected_s1_8_scalars.items():
        if s1_8.get(key) != expected:
            raise ContractError(
                f"S1.8 {key} is {s1_8.get(key)!r}, expected {expected!r}"
            )

    expected_s1_8_allocations = {
        "direction_lookup": 1200,
        "true_false_distance_style": 667,
        "true_false_similarity_style": 667,
        "true_false_workflow_style": 666,
        "ambiguous_percentile": 400,
        "unrecognized_metric": 400,
    }
    allocations_1_8 = s1_8.get("allocations")
    if (
        not isinstance(allocations_1_8, Mapping)
        or list(allocations_1_8) != list(expected_s1_8_allocations)
        or dict(allocations_1_8) != expected_s1_8_allocations
    ):
        raise ContractError("S1.8 ordered allocations changed")
    if sum(allocations_1_8.values()) != s1_8["train_rows"]:
        raise ContractError("S1.8 allocations do not match train_rows")

    expected_s1_8_evaluation = {
        "direction_lookup": 150,
        "true_false_distance_style": 84,
        "true_false_similarity_style": 83,
        "true_false_workflow_style": 83,
        "ambiguous_percentile": 50,
        "unrecognized_metric": 50,
    }
    evaluation_1_8 = s1_8.get("evaluation_allocations")
    if (
        not isinstance(evaluation_1_8, Mapping)
        or list(evaluation_1_8) != list(expected_s1_8_evaluation)
        or dict(evaluation_1_8) != expected_s1_8_evaluation
    ):
        raise ContractError("S1.8 evaluation allocations changed")
    evaluation_rows_1_8 = (
        s1_8["train_rows"]
        * split_fractions["validation"]
        / split_fractions["train"]
    )
    if evaluation_rows_1_8 != 500 or sum(evaluation_1_8.values()) != 500:
        raise ContractError("S1.8 evaluation allocations do not reconcile")

    expected_s1_8_buckets = {
        "direction_lookup": ["direction_lookup"],
        "true_false": [
            "true_false_distance_style",
            "true_false_similarity_style",
            "true_false_workflow_style",
        ],
        "ambiguous_metric": ["ambiguous_percentile"],
        "unrecognized_metric": ["unrecognized_metric"],
    }
    if s1_8.get("semantic_buckets") != expected_s1_8_buckets:
        raise ContractError("S1.8 semantic buckets changed")
    semantic_1_8 = {
        name: sum(allocations_1_8[item] for item in members)
        for name, members in expected_s1_8_buckets.items()
    }
    if semantic_1_8 != {
        "direction_lookup": 1200,
        "true_false": 2000,
        "ambiguous_metric": 400,
        "unrecognized_metric": 400,
    }:
        raise ContractError("S1.8 semantic shares changed")

    expected_metric_conventions = {
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
    if s1_8.get("metric_conventions") != expected_metric_conventions:
        raise ContractError("S1.8 metric conventions changed")

    name_variants_1_8 = s1_8.get("name_variants", {})
    if list(name_variants_1_8) != ["train", "validation", "test"]:
        raise ContractError("S1.8 variant splits changed")
    variant_owners: dict[str, str] = {}
    for split, variants_by_metric in name_variants_1_8.items():
        if list(variants_by_metric) != list(expected_metric_conventions):
            raise ContractError(f"S1.8 {split} variant metric order changed")
        for metric, variants in variants_by_metric.items():
            if not isinstance(variants, list) or len(variants) != 3:
                raise ContractError(
                    f"S1.8 {split} {metric} must have three variants"
                )
            for variant in variants:
                normalized = variant.casefold()
                owner = variant_owners.setdefault(normalized, split)
                if owner != split:
                    raise ContractError("S1.8 metric variants cross splits")

    expected_name_counts = {
        "train": {"canonical": 2240, "metric_variant": 640, "shared_alias": 320},
        "evaluation": {"canonical": 280, "metric_variant": 80, "shared_alias": 40},
    }
    name_policy_1_8 = s1_8.get("name_policy", {})
    for split, expected in expected_name_counts.items():
        if name_policy_1_8.get(f"{split}_counts") != expected:
            raise ContractError(f"S1.8 {split} name counts changed")
    if sum(expected_name_counts["train"].values()) != 3200:
        raise ContractError("S1.8 training name counts do not reconcile")
    if sum(expected_name_counts["evaluation"].values()) != 400:
        raise ContractError("S1.8 evaluation name counts do not reconcile")
    if (
        expected_name_counts["train"]["metric_variant"]
        + expected_name_counts["train"]["shared_alias"]
    ) / 3200 != 0.30:
        raise ContractError("S1.8 variant share does not reconcile")

    true_false_1_8 = s1_8.get("true_false", {})
    if true_false_1_8.get("polarity_order") != [
        "lower_stronger",
        "lower_weaker",
        "higher_stronger",
        "higher_weaker",
    ]:
        raise ContractError("S1.8 polarity order changed")
    if true_false_1_8.get("train_truth_counts") != {
        "true": 1000,
        "false": 1000,
    }:
        raise ContractError("S1.8 training truth counts changed")
    if true_false_1_8.get("evaluation_truth_counts") != {
        "true": 125,
        "false": 125,
    }:
        raise ContractError("S1.8 evaluation truth counts changed")

    if s1_8.get("status_counts") != {
        "train": {"resolved": 3200, "ambiguous": 400, "not_found": 400},
        "evaluation": {"resolved": 400, "ambiguous": 50, "not_found": 50},
    }:
        raise ContractError("S1.8 status counts changed")
    if s1_8.get("framing_counts") != {
        "train": {"direct": 2000, "step": 2000},
        "evaluation": {"direct": 250, "step": 250},
        "resolved_decision": {"direct": "stop", "step": "continue"},
        "deferred_decision": "defer",
    }:
        raise ContractError("S1.8 framing counts changed")

    s1_8_digest = hashlib.sha256(
        json.dumps(
            s1_8,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if s1_8_digest != (
        "966fdf1b4f5723d428910ee0641ff9ecb25570c7b8525dcb77a1449fb3059a29"
    ):
        raise ContractError("S1.8 exact build settings changed")

    s1_9 = families.get("S1.9", {})
    expected_s1_9_keys = [
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
    ]
    if not isinstance(s1_9, Mapping) or list(s1_9) != expected_s1_9_keys:
        raise ContractError("S1.9 build setting keys changed")
    expected_s1_9_scalars = {
        "dataset_id": "world_model_v3_s1_9_module_id_parsing",
        "question_family": "module_id_parsing",
        "book_mode": "closed_book",
        "output_directory": "data/world_model_v3/corpora/s1/s1_9",
        "assignment_seed": 1729,
        "train_rows": 2000,
    }
    for key, expected in expected_s1_9_scalars.items():
        if s1_9.get(key) != expected:
            raise ContractError(
                f"S1.9 {key} is {s1_9.get(key)!r}, expected {expected!r}"
            )

    expected_s1_9_allocations = {
        "rwr_loe_parse": 1000,
        "mentor_ev_parse": 1000,
    }
    allocations_1_9 = s1_9.get("allocations")
    if (
        not isinstance(allocations_1_9, Mapping)
        or list(allocations_1_9) != list(expected_s1_9_allocations)
        or dict(allocations_1_9) != expected_s1_9_allocations
    ):
        raise ContractError("S1.9 ordered allocations changed")
    if sum(allocations_1_9.values()) != s1_9["train_rows"]:
        raise ContractError("S1.9 allocations do not match train_rows")

    expected_s1_9_evaluation = {
        "rwr_loe_parse": 125,
        "mentor_ev_parse": 125,
    }
    evaluation_1_9 = s1_9.get("evaluation_allocations")
    if (
        not isinstance(evaluation_1_9, Mapping)
        or list(evaluation_1_9) != list(expected_s1_9_evaluation)
        or dict(evaluation_1_9) != expected_s1_9_evaluation
    ):
        raise ContractError("S1.9 evaluation allocations changed")
    evaluation_rows_1_9 = (
        s1_9["train_rows"]
        * split_fractions["validation"]
        / split_fractions["train"]
    )
    if evaluation_rows_1_9 != 250 or sum(evaluation_1_9.values()) != 250:
        raise ContractError("S1.9 evaluation allocations do not reconcile")

    if s1_9.get("semantic_buckets") != {
        "rwr_loe": ["rwr_loe_parse"],
        "mentor_ev": ["mentor_ev_parse"],
    }:
        raise ContractError("S1.9 semantic buckets changed")
    if s1_9.get("source_trace_by_allocation") != {
        "rwr_loe_parse": "S1.9.1",
        "mentor_ev_parse": "S1.9.2",
    }:
        raise ContractError("S1.9 source traces changed")

    expected_s1_9_grammars = {
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
            "pattern": (
                "^mentor_ev:full_brain_all_genes_v1:clade_[0-9]{6}$"
            ),
            "field_count": 3,
            "dendrogram_id": "full_brain_all_genes_v1",
            "clade_digits": 6,
        },
    }
    if s1_9.get("grammars") != expected_s1_9_grammars:
        raise ContractError("S1.9 grammars changed")

    expected_s1_9_shapes = {
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
    if s1_9.get("result_shapes") != expected_s1_9_shapes:
        raise ContractError("S1.9 result shapes changed")

    expected_s1_9_pools = {
        "rwr_loe": {
            "source_index_field": "multiplex_gene_ids",
            "record_count": 47017,
            "split_counts": {
                "train": 37613,
                "validation": 4702,
                "test": 4702,
            },
        },
        "mentor_ev": {
            "source_index_field": "mentor_clades",
            "record_count": 15679,
            "eligible_leaf_count": {"min": 5, "max": 30},
            "split_counts": {
                "train": 12543,
                "validation": 1568,
                "test": 1568,
            },
        },
    }
    if s1_9.get("source_pools") != expected_s1_9_pools:
        raise ContractError("S1.9 source pools changed")
    for namespace, pool in expected_s1_9_pools.items():
        if sum(pool["split_counts"].values()) != pool["record_count"]:
            raise ContractError(f"S1.9 {namespace} source pool does not reconcile")

    expected_s1_9_split_policy = {
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
    }
    if s1_9.get("split_policy") != expected_s1_9_split_policy:
        raise ContractError("S1.9 split policy changed")
    residues_1_9 = s1_9["split_policy"]["residues"]
    if sorted(sum(residues_1_9.values(), [])) != list(range(10)):
        raise ContractError("S1.9 split residues do not partition the modulus")

    if s1_9.get("nonpositive_policy") != {
        "included": False,
        "invalid": "excluded",
        "ambiguous": "excluded",
        "unrecognized": "excluded",
        "syntactically_valid_unobserved": "excluded",
    }:
        raise ContractError("S1.9 nonpositive policy changed")
    if s1_9.get("status_counts") != {
        "train": {"resolved": 2000},
        "evaluation": {"resolved": 250},
    }:
        raise ContractError("S1.9 status counts changed")

    expected_s1_9_framing = {
        "train": {"direct": 1000, "step": 1000},
        "evaluation": {"direct": 125, "step": 125},
        "train_form_counts": {
            "rwr_loe_parse": {"direct": 500, "step": 500},
            "mentor_ev_parse": {"direct": 500, "step": 500},
        },
        "evaluation_form_counts": {
            "rwr_loe_parse": {"direct": 63, "step": 62},
            "mentor_ev_parse": {"direct": 62, "step": 63},
        },
        "resolved_decision": {"direct": "stop", "step": "continue"},
    }
    if s1_9.get("framing_counts") != expected_s1_9_framing:
        raise ContractError("S1.9 framing counts changed")
    for form, counts in expected_s1_9_framing["train_form_counts"].items():
        if sum(counts.values()) != expected_s1_9_allocations[form]:
            raise ContractError(f"S1.9 {form} training frames do not reconcile")
    for form, counts in expected_s1_9_framing[
        "evaluation_form_counts"
    ].items():
        if sum(counts.values()) != expected_s1_9_evaluation[form]:
            raise ContractError(f"S1.9 {form} evaluation frames do not reconcile")

    s1_9_digest = hashlib.sha256(
        json.dumps(
            s1_9,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if s1_9_digest != (
        "98badcd4b359a5c94025732cdc20f2773812cdc716a66d7078caf4fa82f872ba"
    ):
        raise ContractError("S1.9 exact build settings changed")

    coordinate = contract["coordinate_system"]
    if expected_s1_9_pools["rwr_loe"]["record_count"] != coordinate.get(
        "graph_gene_count"
    ):
        raise ContractError("S1.9 RWR-LOE source count changed")
    mentor_coordinate = coordinate.get("mentor_ev", {})
    if expected_s1_9_pools["mentor_ev"]["record_count"] != (
        mentor_coordinate.get("eligible_clade_count")
    ):
        raise ContractError("S1.9 MENTOR-EV source count changed")
    if expected_s1_9_pools["mentor_ev"]["eligible_leaf_count"] != (
        mentor_coordinate.get("eligible_leaf_count")
    ):
        raise ContractError("S1.9 MENTOR-EV leaf limits changed")
    if sum(coordinate.get("layer_counts", {}).values()) != coordinate.get(
        "layer_count"
    ):
        raise ContractError("coordinate-system layer counts do not reconcile")
    for digest_name in ("source_flist_sha256",):
        if re.fullmatch(r"[0-9a-f]{64}", str(coordinate.get(digest_name))) is None:
            raise ContractError(f"coordinate_system.{digest_name} is not SHA-256")
    registry = coordinate.get("identifier_registry", {})
    registry_digest = registry.get("source_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(registry_digest)) is None:
        raise ContractError(
            "coordinate_system.identifier_registry.source_sha256 is not SHA-256"
        )
    for source_name in ("stable_id_history", "hgnc_history"):
        source_digest = registry.get(source_name, {}).get("source_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", str(source_digest)) is None:
            raise ContractError(
                "coordinate_system.identifier_registry."
                f"{source_name}.source_sha256 is not SHA-256"
            )
    mouse = coordinate.get("out_of_scope_species", {}).get("mouse", {})
    if mouse.get("ensembl_release") != coordinate.get("ensembl_release"):
        raise ContractError("mouse and human Ensembl releases must match")
    if re.fullmatch(
        r"[0-9a-f]{64}", str(mouse.get("source_sha256"))
    ) is None:
        raise ContractError("mouse source SHA-256 is invalid")
    mentor_digest = coordinate.get("mentor_ev", {}).get("sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(mentor_digest)) is None:
        raise ContractError("coordinate_system.mentor_ev.sha256 is not SHA-256")
    source_index = contract["source_index"]
    if source_index.get("schema_version") != (
        "mentor-rl-world-model-source-index-v1"
    ):
        raise ContractError("source_index.schema_version is unsupported")
    content_digest = source_index.get("content_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(content_digest)) is None:
        raise ContractError("source_index.content_sha256 is not SHA-256")


def system_prompt_sha256(contract: Mapping[str, Any]) -> str:
    """Return the digest of the sole S1 system prompt."""

    prompt = contract["runtime"]["system_prompt"]
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def verify_source_inputs(contract: Mapping[str, Any]) -> None:
    """Verify the live source snapshots pinned by the contract."""

    coordinate = contract["coordinate_system"]
    registry_spec = coordinate["identifier_registry"]
    registry = Path(registry_spec["source_path"])
    if not registry.is_absolute():
        registry = REPOSITORY_ROOT / registry
    stable_id_spec = registry_spec["stable_id_history"]
    stable_id_history = Path(stable_id_spec["source_path"])
    if not stable_id_history.is_absolute():
        stable_id_history = REPOSITORY_ROOT / stable_id_history
    hgnc_spec = registry_spec["hgnc_history"]
    hgnc_history = Path(hgnc_spec["source_path"])
    if not hgnc_history.is_absolute():
        hgnc_history = REPOSITORY_ROOT / hgnc_history
    mouse_spec = coordinate["out_of_scope_species"]["mouse"]
    mouse_registry = Path(mouse_spec["source_path"])
    if not mouse_registry.is_absolute():
        mouse_registry = REPOSITORY_ROOT / mouse_registry
    mygene_spec = contract["curriculum"]["families"]["S1.4"][
        "mygene_source"
    ]
    mygene_cache = Path(mygene_spec["source_path"])
    if not mygene_cache.is_absolute():
        mygene_cache = REPOSITORY_ROOT / mygene_cache
    flist = Path(coordinate["source_flist_path"])
    dendrogram = Path(coordinate["mentor_ev"]["source_path"])
    for source in (
        registry,
        stable_id_history,
        hgnc_history,
        mouse_registry,
        mygene_cache,
        flist,
        dendrogram,
    ):
        if not source.is_file():
            raise ContractError(f"pinned source is not mounted: {source}")

    actual_registry_hash = file_sha256(registry)
    if actual_registry_hash != registry_spec["source_sha256"]:
        raise ContractError(
            f"Ensembl GTF SHA-256 is {actual_registry_hash}, "
            f"expected {registry_spec['source_sha256']}"
        )
    actual_stable_id_hash = file_sha256(stable_id_history)
    if actual_stable_id_hash != stable_id_spec["source_sha256"]:
        raise ContractError(
            f"Ensembl stable-ID history SHA-256 is {actual_stable_id_hash}, "
            f"expected {stable_id_spec['source_sha256']}"
        )
    actual_hgnc_hash = file_sha256(hgnc_history)
    if actual_hgnc_hash != hgnc_spec["source_sha256"]:
        raise ContractError(
            f"HGNC history SHA-256 is {actual_hgnc_hash}, "
            f"expected {hgnc_spec['source_sha256']}"
        )
    actual_mouse_hash = file_sha256(mouse_registry)
    if actual_mouse_hash != mouse_spec["source_sha256"]:
        raise ContractError(
            f"mouse Ensembl GTF SHA-256 is {actual_mouse_hash}, "
            f"expected {mouse_spec['source_sha256']}"
        )
    actual_mygene_hash = file_sha256(mygene_cache)
    if actual_mygene_hash != mygene_spec["source_sha256"]:
        raise ContractError(
            f"MyGene cache SHA-256 is {actual_mygene_hash}, "
            f"expected {mygene_spec['source_sha256']}"
        )

    actual_flist_hash = file_sha256(flist)
    if actual_flist_hash != coordinate["source_flist_sha256"]:
        raise ContractError(
            f"flist SHA-256 is {actual_flist_hash}, "
            f"expected {coordinate['source_flist_sha256']}"
        )
    try:
        layers = read_flist(flist)
    except SourceError as exc:
        raise ContractError(str(exc)) from exc
    prefix_counts = Counter(layer.layer_id.split(":", 1)[0] for layer in layers)
    if len(layers) != coordinate["layer_count"]:
        raise ContractError(
            "flist row count does not match coordinate_system.layer_count"
        )
    if dict(prefix_counts) != coordinate["layer_counts"]:
        raise ContractError(
            f"flist family counts {dict(prefix_counts)!r} do not match the contract"
        )
    bad_tags = {
        layer.layer_id: problem
        for layer in layers
        if (problem := layer_tag_problem(contract, layer.layer_id)) is not None
    }
    if bad_tags:
        first = next(iter(bad_tags.items()))
        raise ContractError(f"flist layer grammar failure: {first[0]}: {first[1]}")

    mentor = coordinate["mentor_ev"]
    actual_tree_hash = file_sha256(dendrogram)
    if actual_tree_hash != mentor["sha256"]:
        raise ContractError(
            f"MENTOR-EV SHA-256 is {actual_tree_hash}, expected {mentor['sha256']}"
        )
    try:
        tree = read_dendrogram(dendrogram)
    except SourceError as exc:
        raise ContractError(str(exc)) from exc
    if len(tree.nodes) != mentor["node_count"]:
        raise ContractError("MENTOR-EV node count does not match the contract")
    leaf_counts = tree.leaf_counts()
    leaf_count = len(tree.leaf_gene_ids())
    if leaf_count != mentor["leaf_count"]:
        raise ContractError("MENTOR-EV leaf count does not match the contract")
    if leaf_count != coordinate["graph_gene_count"]:
        raise ContractError("multiplex and MENTOR-EV gene counts disagree")
    bounds = mentor["eligible_leaf_count"]
    eligible = sum(
        bounds["min"] <= count <= bounds["max"]
        and not tree.nodes[node_id].is_leaf
        for node_id, count in leaf_counts.items()
    )
    if eligible != mentor["eligible_clade_count"]:
        raise ContractError(
            f"MENTOR-EV has {eligible} eligible clades, "
            f"expected {mentor['eligible_clade_count']}"
        )


def composite_provenance(
    contract: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
    graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a composite provenance object in the contract's wire order."""

    if registry is None and graph is None:
        raise ContractError(
            "use provenance: null when no evidence channel is populated"
        )
    values = {"registry": registry, "graph": graph}
    return {
        channel: dict(values[channel]) if values[channel] is not None else None
        for channel in contract["provenance"]["channels"]
    }


def tool_definitions(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive the three callable JSON-schema tool definitions."""

    definitions: list[dict[str, Any]] = []
    for name in contract["tools"]["callable"]:
        spec = contract["tools"]["definitions"][name]
        properties: dict[str, Any] = {}
        for argument, kind in spec["args"].items():
            if kind == "str":
                argument_schema: dict[str, Any] = {"type": "string", "minLength": 1}
            elif kind == "list":
                argument_schema = {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            else:
                raise ContractError(f"unsupported argument kind {kind!r} for {name}")
            properties[argument] = argument_schema
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(spec["args"]),
                        "additionalProperties": False,
                    },
                },
            }
        )
    return definitions


def parse_layer_ref(
    contract: Mapping[str, Any], layer_id: str
) -> dict[str, Any]:
    """Parse one layer tag with the canonical grammar."""

    if not isinstance(layer_id, str) or not layer_id:
        raise ContractError("layer ID must be a nonempty string")
    grammar = contract["layer_grammars"]
    parts = layer_id.split(grammar["sep"])
    if any(not part for part in parts):
        raise ContractError("layer ID contains an empty part")
    for family, spec in grammar["families"].items():
        if parts[0] != spec["match_prefix"]:
            continue
        if len(parts) not in spec["arity"]:
            raise ContractError(
                f"{family} layer has arity {len(parts)}; "
                f"expected one of {spec['arity']}"
            )
        variadic = spec.get("variadic_field")
        if variadic is not None:
            count = len(parts[variadic["start"] :])
            if not variadic["min_items"] <= count <= variadic["max_items"]:
                raise ContractError(
                    f"{variadic['name']} has {count} parts; expected "
                    f"{variadic['min_items']} to {variadic['max_items']}"
                )
        values: dict[str, Any] = {
            "layer_id": layer_id,
            "layer_family": None,
            "source": None,
            "network": None,
            "organ": None,
            "region_path": None,
            "cell_type_path": None,
        }
        if family == "humannet":
            values["source"] = parts[0]
            values["network"] = parts[1]
            values["layer_family"] = spec["network_family"].get(
                parts[1], spec["unrecognized_network_family"]
            )
        elif family == "bulk_pen":
            values["layer_family"] = spec["family"]
            values["organ"] = parts[1]
            values["region_path"] = parts[2:]
        elif family == "scpen":
            values["layer_family"] = spec["family"]
            values["organ"] = parts[1]
            values["region_path"] = [parts[2]]
            values["cell_type_path"] = parts[3:]
        elif family == "tf_target":
            values["layer_family"] = spec["family"]
            values["organ"] = parts[1]
        else:
            raise ContractError(f"unsupported layer grammar family {family!r}")
        if list(values) != grammar["normalized_keys"]:
            raise ContractError("parsed layer keys differ from the contract")
        return values
    raise ContractError(f"no layer family matches prefix {parts[0]!r}")


def layer_tag_problem(contract: Mapping[str, Any], layer_id: str) -> str | None:
    """Return why a native layer ID violates its family grammar, if it does."""

    try:
        parse_layer_ref(contract, layer_id)
    except ContractError as exc:
        return str(exc)
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, failing with the exact bad line."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    if not rows:
        raise ContractError(f"{path}: no rows")
    return rows


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _message_lines(message: Mapping[str, Any]) -> Iterable[str]:
    role = message.get("role")
    if role == "harness":
        yield "Context (from the harness):"
        yield f"`{message.get('content', '')}`"
    elif role == "user":
        yield "User:"
        yield f"`{message.get('content', '')}`"
    elif role == "assistant" and "tool_call" in message:
        if message.get("thinking"):
            yield "Thinking:"
            yield f"`{message['thinking']}`"
        call = message["tool_call"]
        yield "Tool call:"
        yield f"`{call['name']}({_json(call['arguments'])})`"
    elif role == "tool":
        yield "Tool result:"
        yield f"`{_json(message.get('content'))}`"
    elif role == "assistant":
        if message.get("thinking"):
            yield "Thinking:"
            yield f"`{message['thinking']}`"
        yield "Answer:"
        yield f"`{_json(message.get('content'))}`"


def render_golden_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render golden JSONL rows as the human review document."""

    lines = [
        "# S1 Question Classes and Golden Traces",
        "",
        "<!-- Generated from runtime/world_model/golden_traces.jsonl. -->",
        "<!-- Edit the JSONL source and run "
        "`python -m runtime.world_model.contract render`. -->",
        "",
        "The row fixture is normative; this rendering is for review.",
        "",
    ]
    current_family: str | None = None
    for row in rows:
        metadata = row.get("metadata", {})
        source_trace = str(metadata.get("source_trace", row.get("row_id", "trace")))
        match = re.match(r"^(S[0-9]+(?:\.[0-9]+)?)", source_trace)
        family = match.group(1) if match else "Other"
        if family != current_family:
            lines.extend([f"## {family}", ""])
            current_family = family
        lines.extend([f"**{source_trace}**", ""])
        messages = row.get("messages", [])
        users = [
            i
            for i, message in enumerate(messages)
            if message.get("role") == "user"
        ]
        trained_turn = users[-1] if users else 0
        for index, message in enumerate(messages):
            if message.get("role") == "system":
                continue
            if len(users) > 1 and index == users[0]:
                lines.extend(["[Earlier turn: in context, masked]", ""])
            if len(users) > 1 and index == trained_turn:
                lines.extend(["[This turn: trained]", ""])
            lines.extend(_message_lines(message))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _check_golden_rows(
    contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    row_keys = contract["rows"]["row_keys"]
    metadata_spec = contract["rows"]["metadata"]
    allowed_metadata = set(metadata_spec["required"] + metadata_spec["optional"])
    prompt = contract["runtime"]["system_prompt"]
    digest = system_prompt_sha256(contract)
    tools = tool_definitions(contract)
    row_ids: set[str] = set()
    traces: set[str] = set()
    for index, row in enumerate(rows, 1):
        where = f"golden row {index}"
        if list(row) != row_keys:
            raise ContractError(f"{where}: keys {list(row)!r}, expected {row_keys!r}")
        row_id = row["row_id"]
        if row_id in row_ids:
            raise ContractError(f"{where}: duplicate row_id {row_id!r}")
        row_ids.add(row_id)
        messages = row["messages"]
        if not messages or messages[0] != {"role": "system", "content": prompt}:
            raise ContractError(f"{where}: does not use the contract system prompt")
        if row["tools"] != tools:
            raise ContractError(f"{where}: does not use the contract tool definitions")
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            raise ContractError(f"{where}: metadata is not an object")
        missing = [key for key in metadata_spec["required"] if key not in metadata]
        extra = [key for key in metadata if key not in allowed_metadata]
        if missing or extra:
            raise ContractError(
                f"{where}: metadata missing={missing!r}, extra={extra!r}"
            )
        if metadata["system_prompt_sha256"] != digest:
            raise ContractError(f"{where}: wrong system_prompt_sha256")
        expected_metadata = {
            "schema_version": contract["runtime"]["row_schema_version"],
            "tool_contract_version": contract["runtime"][
                "tool_contract_version"
            ],
            "ensembl_release": contract["coordinate_system"][
                "ensembl_release"
            ],
            "identifier_registry_id": contract["coordinate_system"][
                "identifier_registry_id"
            ],
        }
        for key, expected in expected_metadata.items():
            if metadata[key] != expected:
                raise ContractError(
                    f"{where}: metadata.{key} is {metadata[key]!r}, "
                    f"expected {expected!r}"
                )
        trace = metadata["source_trace"]
        if trace in traces:
            raise ContractError(f"{where}: duplicate source_trace {trace!r}")
        traces.add(trace)


def check_golden(
    contract: Mapping[str, Any], path: Path = GOLDEN_TRACES_PATH
) -> list[dict[str, Any]]:
    """Load the golden rows and enforce the non-negotiable row envelope."""

    rows = load_jsonl(path)
    _check_golden_rows(contract, rows)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="validate the contract and golden rows")
    subparsers.add_parser("render", help="print the golden rows for review")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = load_contract()
    rows = check_golden(contract)
    if args.command == "render":
        print(render_golden_markdown(rows), end="")
    else:
        verify_source_inputs(contract)
        print(
            f"contract, pinned sources, and golden rows are valid "
            f"({len(rows)} traces)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
