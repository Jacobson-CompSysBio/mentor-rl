"""Apply the pinned S1 v10 training overlay to the frozen shared contract.

The overlay owns training revisions, reviewed prompt banks, replay selection,
and output locations. The shared S0/S1 runtime contract stays byte-identical.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from .contract import CONTRACT_PATH, REPOSITORY_ROOT
from .s1_prompt_banks import (
    PromptBankError,
    load_prompt_banks,
    validate_identity_completion_reasoning,
)
from .sources import file_sha256
from .s1_lookup_retry import load_retry_policy
from .s1_membership_retry import load_membership_retry_policy
from .tools import GENE_ID_TOOL, SYMBOL_TOOL


# Exact training-template bindings for the reviewed closed-book forms.
FINAL_REASONING_FIELDS = {
    "S1.6": {
        4: ("requested_field", "filter_value", "parse_steps", "matching_layer_ids"),
        6: ("input", "grammar_check"),
    },
    "S1.7": {
        2: ("requested_family", "layer_assignments", "matching_layer_ids", "other_layer_ids"),
        4: ("claim_phrase", "claim", "claim_definition", "family_assignment", "layer_id",
            "layer_family", "evidence_type", "supported_claims", "unsupported_claims", "claim_supported"),
        8: ("layer_id", "network"),
    },
    "S1.8": {
        1: ("input_name", "name_resolution", "metric", "lower_is_stronger", "meaning", "rule",
            "direction_explanation", "candidate_schema"),
        2: ("input_name", "name_resolution", "metric", "lower_is_stronger", "meaning", "rule",
            "direction_explanation", "claim_direction", "claim_outcome", "actual_outcome",
            "statement_true", "candidate_schema"),
        3: ("input_name", "name_resolution", "metric", "lower_is_stronger", "meaning", "rule",
            "direction_explanation", "claim_direction", "claim_outcome", "actual_outcome",
            "statement_true", "candidate_schema"),
        4: ("input_name", "name_resolution", "metric", "lower_is_stronger", "meaning", "rule",
            "direction_explanation", "claim_direction", "claim_outcome", "actual_outcome",
            "statement_true", "candidate_schema"),
        5: ("input_name",),
        6: ("input_name",),
    },
    "S1.9": {
        1: ("module_id", "multiplex_id", "seed_component", "seed_gene_id", "threshold_recipe"),
        2: ("module_id", "dendrogram_id", "clade_id"),
    },
}


S1_OVERLAY_V9_SCHEMA_VERSION = "mentor-rl-world-model-s1-overlay-v9"
S1_V10_OVERLAY_PATH = Path(__file__).with_name("s1_v10.yaml")
OVERLAY_KEYS = ['schema_version',
 'base_contract',
 'corpus_root',
 'evaluator_root',
 'family_overrides',
 'prompt_banks',
 'train_normalization',
 's0_replay']
FAMILY_OVERRIDE_KEYS = {'S1.1': {'membership_answers',
          'output_directory',
          'train_absent_membership_reasoning',
          'train_final_reasoning',
          'train_id_confirmation_reasoning',
          'train_lookup_retries',
          'train_membership_reasoning',
          'train_prompt_banks',
          'train_unknown_multiplex_reasoning'},
 'S1.2': {'output_directory',
          'shares',
          'train_final_reasoning',
          'train_form_counts',
          'train_lookup_retries',
          'train_near_miss_reasoning',
          'train_no_symbol_reasoning',
          'train_prompt_banks',
          'train_symbol_input_reasoning',
          'train_version_suffix_reasoning'},
 'S1.3': {'membership_answers',
          'output_directory',
          'train_choice_replies',
          'train_final_reasoning',
          'train_form_counts',
          'train_membership_retries',
          'train_multiple_membership_reasoning',
          'train_multiplex_reasoning',
          'train_no_membership_reasoning',
          'train_prompt_banks',
          'train_single_membership_reasoning',
          'train_targeted_v10',
          'train_targeted_v9',
          'train_user_choice_reasoning'},
 'S1.4': {'output_directory',
          'train_ambiguous_candidate_reasoning',
          'train_ambiguous_noncandidate_reasoning',
          'train_candidate_reasoning',
          'train_match_reasoning',
          'train_mismatch_reasoning',
          'train_mygene_match_reasoning',
          'train_unresolved_symbol_reasoning'},
 'S1.6': {'output_directory',
          'train_bulk_pen_reasoning',
          'train_decorrelate',
          'train_diverse_lists',
          'train_final_reasoning',
          'train_humannet_reasoning',
          'train_layer_reasoning',
          'train_scpen_reasoning',
          'train_targeted_v9'},
 'S1.7': {'output_directory',
          'train_claim_reasoning',
          'train_decorrelate',
          'train_final_reasoning',
          'train_form_counts',
          'train_targeted_v9'},
 'S1.8': {'output_directory',
          'train_decorrelate',
          'train_final_reasoning',
          'train_metric_reasoning',
          'train_targeted_v9',
          'train_variant_coverage'},
 'S1.9': {'output_directory',
          'train_balanced_requests',
          'train_decorrelate',
          'train_final_reasoning',
          'train_module_reasoning'}}
SCHEMA_LAYOUTS = {
    S1_OVERLAY_V9_SCHEMA_VERSION: (OVERLAY_KEYS, FAMILY_OVERRIDE_KEYS),
}
TRAIN_NORMALIZATION = {
    "evidence_id_policy": "aggregate_sequence_v1",
    "strip_s0_metadata_additions": True,
}
# S1 v5 membership answers, by stage, in every split.
MEMBERSHIP_ANSWERS = {
    # A gene absent from the named multiplex answers the question, so
    # S1.1.10 is resolved, as the membership tool reports. S1.1.11 keeps the
    # forms 9-10 result shape, with present null: an unknown multiplex gives
    # no membership answer.
    "S1.1": {
        "absent_status": "resolved",
        "unknown_multiplex_present": None,
    },
    # S1.3.5 to S1.3.7 keep every field of both tool results. With no
    # candidate in the multiplex (S1.3.7), nothing chooses one, so the answer
    # is ambiguous, as in S1.3.6 and S1.3.10.
    "S1.3": {
        "keep_tool_fields": True,
        "none_present_status": "ambiguous",
    },
}
REPLAY_KEYS = [
    "dataset_id",
    "output_directory",
    "manifest_schema_version",
    "corpus_schema_version",
]
V9_REPLAY_COUNTS = {
    "resolved_symbol": 92,
    "named_gene_id": 56,
    "unnamed_gene_id": 56,
    "ambiguous_symbol": 48,
    "retired_gene_id": 16,
    "historical_symbol": 8,
}


class S1OverlayError(ValueError):
    """Raised when the S1 overlay does not fit the shared contract."""


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise S1OverlayError(f"the S1 overlay {label} must be a string")
    return value


def _validate_shares(base: Any, shares: Any) -> None:
    """Require the same share names, each in [0, 1], with an exact sum of 1."""

    if not isinstance(base, Mapping) or not isinstance(shares, Mapping):
        raise S1OverlayError("the S1 overlay shares must be a mapping")
    if list(shares) != list(base):
        raise S1OverlayError("the S1 overlay must keep the contract share names")
    total = Fraction(0)
    for name, value in shares.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise S1OverlayError(f"the S1 overlay share {name} must be numeric")
        share = Fraction(str(value))
        if not 0 <= share <= 1:
            raise S1OverlayError(f"the S1 overlay share {name} is out of range")
        total += share
    if total != 1:
        raise S1OverlayError(f"the S1 overlay shares sum to {total}, not 1")


def apply_train_form_counts(
    counts: Mapping[int, int], overrides: Mapping[int, int]
) -> dict[int, int]:
    """Return train form counts with an overlay's ``train_form_counts``.

    The train total stays the same, so the corpus size and the number of
    training steps do not change.
    """

    updated = dict(counts)
    for form, count in overrides.items():
        if form not in updated:
            raise S1OverlayError(f"the stage has no form {form}")
        updated[form] = count
    if sum(updated.values()) != sum(counts.values()):
        raise S1OverlayError("the train form counts change the total")
    return dict(sorted(updated.items()))


def overlay_pin(overlay_path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Return the manifest record that identifies one applied overlay."""

    overlay_path = overlay_path.resolve()
    return {
        "path": overlay_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "sha256": file_sha256(overlay_path),
        "schema_version": str(payload["schema_version"]),
    }


def apply_s1_overlay(
    contract: Mapping[str, Any],
    overlay_path: Path = S1_V10_OVERLAY_PATH,
    *,
    base_contract_path: Path = CONTRACT_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy of the contract with the S1 overlay applied."""

    overlay_path = overlay_path.resolve()
    try:
        payload = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise S1OverlayError(f"cannot load S1 overlay {overlay_path}: {error}") from error
    if not isinstance(payload, dict):
        raise S1OverlayError("the S1 overlay keys changed")
    layout = SCHEMA_LAYOUTS.get(payload.get("schema_version"))
    if layout is None:
        raise S1OverlayError("the S1 overlay schema changed")
    overlay_keys, family_override_keys = layout
    if list(payload) != overlay_keys:
        raise S1OverlayError("the S1 overlay keys changed")

    # The overlay applies only to the exact shared contract that it names.
    base_identity = payload["base_contract"]
    if not isinstance(base_identity, Mapping) or list(base_identity) != [
        "path",
        "sha256",
    ]:
        raise S1OverlayError("the S1 overlay base-contract identity is invalid")
    declared_path = (REPOSITORY_ROOT / str(base_identity["path"])).resolve()
    if declared_path != base_contract_path.resolve():
        raise S1OverlayError("the S1 overlay names a different shared contract")
    if base_identity["sha256"] != file_sha256(base_contract_path.resolve()):
        raise S1OverlayError("the shared contract does not match the S1 overlay pin")
    _required_string(payload["corpus_root"], "corpus_root")
    _required_string(payload["evaluator_root"], "evaluator_root")

    overlaid = copy.deepcopy(dict(contract))
    families = overlaid["curriculum"]["families"]
    overrides = payload["family_overrides"]
    if not isinstance(overrides, Mapping) or not set(overrides) <= set(
        family_override_keys
    ):
        raise S1OverlayError("the S1 overlay overrides an unsupported family")
    for section, values in overrides.items():
        if not isinstance(values, Mapping) or set(values) != family_override_keys[section]:
            raise S1OverlayError(f"the S1 overlay {section} override keys changed")
        _required_string(values["output_directory"], f"{section} output_directory")
        if values.get("train_targeted_v10") and values.get("train_targeted_v9") is not True:
            raise S1OverlayError("the S1 v10 choice revision requires the v9 choice coverage")
        if "train_choice_replies" in values:
            from .s1_prompt_banks import load_choice_replies

            try:
                load_choice_replies(values["train_choice_replies"])
            except (PromptBankError, ValueError, OSError) as error:
                raise S1OverlayError(f"the S1 overlay choice replies: {error}") from error
        if "train_membership_retries" in values:
            try:
                load_membership_retry_policy(values["train_membership_retries"])
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise S1OverlayError(f"the S1 overlay membership retry policy: {exc}") from exc
        if "train_lookup_retries" in values:
            try:
                load_retry_policy(
                    values["train_lookup_retries"],
                    tool_name=GENE_ID_TOOL if section == "S1.2" else SYMBOL_TOOL,
                )
            except (ValueError, OSError, TypeError, KeyError) as error:
                raise S1OverlayError(f"the S1 overlay lookup retry policy: {error}") from error
        if "shares" in values:
            _validate_shares(families[section].get("shares"), values["shares"])
        if "membership_answers" in values and (
            not isinstance(values["membership_answers"], Mapping)
            or dict(values["membership_answers"]) != MEMBERSHIP_ANSWERS[section]
        ):
            raise S1OverlayError("the S1 overlay membership answers changed")
        if "train_prompt_banks" in values:
            pins = values["train_prompt_banks"]
            allowed_forms = {
                "S1.1": {1, 2, 9, 10, 11},
                "S1.2": {1, 3, 4, 6, 8, 9},
                "S1.3": {1, 4, 5, 6, 7, 10},
            }[section]
            if (
                not isinstance(pins, Mapping)
                or not pins or not set(pins) <= allowed_forms
                or any(type(form) is not int for form in pins)
            ):
                raise S1OverlayError(
                    "the S1 overlay train prompt banks must name only S1.2.1, S1.2.3, S1.2.4, S1.2.6, S1.2.8, or S1.2.9"
                    if section == "S1.2" else
                    "the S1 overlay train prompt banks must name only S1.3.1, S1.3.4, S1.3.5, S1.3.6, S1.3.7, or S1.3.10"
                    if section == "S1.3" else
                    "the S1 overlay train prompt banks must name only S1.1.1, S1.1.2, S1.1.9, S1.1.10, or S1.1.11"
                )
            try:
                for pin in pins.values():
                    load_prompt_banks(pin)
            except PromptBankError as error:
                raise S1OverlayError(
                    f"the S1 overlay train prompt bank: {error}"
                ) from error
        if "train_final_reasoning" in values:
            reasoning = values["train_final_reasoning"]
            allowed_forms = {
                "S1.1": {2, 3}, "S1.2": {1, 5, 6}, "S1.3": {1, 4},
                "S1.6": {4, 6}, "S1.7": {2, 4, 8},
                "S1.8": {1, 2, 3, 4, 5, 6}, "S1.9": {1, 2},
            }[section]
            if (
                not isinstance(reasoning, Mapping)
                or not reasoning or not set(reasoning) <= allowed_forms
                or any(type(form) is not int for form in reasoning)
            ):
                raise S1OverlayError(
                    f"the S1 overlay final reasoning must name only reviewed {section} forms"
                    if section in {"S1.6", "S1.7", "S1.8", "S1.9"} else
                    "the S1 overlay final reasoning must name only S1.2.1, S1.2.5, or S1.2.6"
                    if section == "S1.2" else
                    "the S1 overlay final reasoning must name only S1.3.1 or S1.3.4"
                    if section == "S1.3" else
                    "the S1 overlay final reasoning must name only S1.1.2 or S1.1.3"
                )
            try:
                for form, text in reasoning.items():
                    if section in FINAL_REASONING_FIELDS:
                        if not isinstance(text, Mapping) or set(text) != {"direct", "step"}:
                            raise S1OverlayError(f"the S1 overlay {section} final reasoning framing keys changed")
                        for template in text.values():
                            fields = FINAL_REASONING_FIELDS[section][form]
                            if payload["schema_version"] == S1_OVERLAY_V9_SCHEMA_VERSION and section == "S1.8" and form == 4:
                                fields = (*fields, "outer_scope")
                            validate_identity_completion_reasoning(template, fields=fields)
                    elif section == "S1.3":
                        validate_identity_completion_reasoning(text, fields=("symbol", "count"))
                    elif section == "S1.2" and form == 5:
                        validate_identity_completion_reasoning(text, fields=("gene_id",))
                    elif section == "S1.2" and form == 6:
                        validate_identity_completion_reasoning(text, fields=("input",))
                    else:
                        fields = ("symbol",) if section == "S1.1" and form == 3 else ("symbol", "gene_id")
                        validate_identity_completion_reasoning(text, fields=fields)
            except PromptBankError as error:
                raise S1OverlayError(f"the S1 overlay final reasoning: {error}") from error
        for key, label, fields in (
            ("train_id_confirmation_reasoning", "ID-confirmation", ("gene_id", "symbol", "user_request")),
            ("train_version_suffix_reasoning", "version-suffix", ("symbol", "gene_id")),
            ("train_no_symbol_reasoning", "no-symbol", ("gene_id",)),
            ("train_near_miss_reasoning", "near-miss", ("symbol", "gene_id")),
            ("train_symbol_input_reasoning", "symbol-input", ("symbol", "gene_id")),
            ("train_user_choice_reasoning", "user-choice", ("gene_id", "symbol")),
            ("train_match_reasoning", "alignment-match", ("symbol", "gene_id", "gene_id")),
            ("train_mismatch_reasoning", "alignment-mismatch", ("symbol", "gene_id", "gene_id", "target_id", "target_id")),
            ("train_unresolved_symbol_reasoning", "alignment-unresolved-symbol", ("input", "target_id")),
            ("train_ambiguous_candidate_reasoning", "alignment-ambiguous-candidate", ("symbol", "count", "target_id", "target_id")),
            ("train_ambiguous_noncandidate_reasoning", "alignment-ambiguous-noncandidate", ("symbol", "count", "target_id", "target_id")),
            ("train_mygene_match_reasoning", "alignment-mygene-match", ("symbol", "gene_id", "gene_id", "query", "query")),
            ("train_bulk_pen_reasoning", "bulk-pen", ("layer_id", "organ", "region_path")),
            ("train_scpen_reasoning", "scpen", ("layer_id", "organ", "region_path", "cell_type_path", "requested_field", "requested_value")),
            ("train_humannet_reasoning", "humannet", ("layer_id", "network", "layer_family")),
        ):
            if key not in values:
                continue
            reasoning = values[key]
            if not isinstance(reasoning, Mapping) or set(reasoning) != {"direct", "step"}:
                raise S1OverlayError(f"the S1 overlay {label} reasoning keys changed")
            try:
                for text in reasoning.values():
                    validate_identity_completion_reasoning(text, fields=fields)
            except PromptBankError as error:
                raise S1OverlayError(f"the S1 overlay {label} reasoning: {error}") from error
        for key in (
            "train_membership_reasoning", "train_absent_membership_reasoning",
            "train_unknown_multiplex_reasoning", "train_single_membership_reasoning",
            "train_multiple_membership_reasoning", "train_no_membership_reasoning",
        ):
            if key not in values:
                continue
            reasoning = values[key]
            phases = (
                {"after_lookup", "final"} if key in {"train_multiple_membership_reasoning", "train_no_membership_reasoning"}
                else {"after_lookup", "direct", "step"}
            )
            if not isinstance(reasoning, Mapping) or set(reasoning) != phases:
                raise S1OverlayError("the S1 overlay membership reasoning keys changed")
            try:
                for phase, text in reasoning.items():
                    if key == "train_multiple_membership_reasoning":
                        fields = ("symbol", "count", "source", "multiplex_id") if phase == "after_lookup" else ("symbol", "count", "present_count", "multiplex_id")
                    elif key == "train_no_membership_reasoning":
                        fields = ("symbol", "count", "source", "multiplex_id") if phase == "after_lookup" else ("symbol", "count", "multiplex_id")
                    elif key == "train_single_membership_reasoning":
                        fields = ("symbol", "count", "source", "multiplex_id") if phase == "after_lookup" else ("gene_id", "multiplex_id")
                    else:
                        fields = ("symbol", "gene_id") if phase == "after_lookup" else (
                            () if key == "train_unknown_multiplex_reasoning" else ("gene_id",)
                        )
                    validate_identity_completion_reasoning(text, fields=fields)
            except PromptBankError as error:
                raise S1OverlayError(f"the S1 overlay membership reasoning: {error}") from error
        # A train-only switch is either absent or on.
        for switch in (
            "train_multiplex_reasoning",
            "train_candidate_reasoning",
            "train_diverse_lists",
            "train_layer_reasoning",
            "train_claim_reasoning",
            "train_variant_coverage",
            "train_metric_reasoning",
            "train_module_reasoning",
            "train_balanced_requests",
            "train_decorrelate",
            "train_targeted_v9",
            "train_targeted_v10",
        ):
            if values.get(switch, True) is not True:
                raise S1OverlayError(f"the S1 overlay {switch} changed")
        # The S1.2 builder checks the train total and the failure pairs.
        if "train_form_counts" in values and (
            not isinstance(values["train_form_counts"], Mapping)
            or not values["train_form_counts"]
            or not all(
                type(form) is int and type(count) is int and count >= 0
                for form, count in values["train_form_counts"].items()
            )
        ):
            raise S1OverlayError("the S1 overlay train form counts are invalid")
        families[section] = {**families[section], **copy.deepcopy(dict(values))}

    # An ambiguous S1.3.7 answer keeps the status of its lookup, so its
    # blocked-reason template takes the "lookup" status, as tool_error.partial
    # does, in the overlaid copy only.
    if "membership_answers" in overrides.get("S1.3", {}):
        for template in overlaid["blocked_reasons"]["codes"][
            "no_multiplex_membership"
        ]:
            if template["id"] == "no_multiplex_membership.candidates":
                template["status"] = "lookup"

    if "prompt_banks" in payload:
        try:
            load_prompt_banks(payload["prompt_banks"])
        except PromptBankError as error:
            raise S1OverlayError(f"the S1 overlay prompt bank: {error}") from error

    if "train_normalization" in payload:
        normalization = payload["train_normalization"]
        if not isinstance(normalization, Mapping) or dict(normalization) != (
            TRAIN_NORMALIZATION
        ):
            raise S1OverlayError(
                "the S1 overlay train normalization contract changed"
            )

    replay = payload["s0_replay"]
    replay_keys = REPLAY_KEYS + ["selection"]
    if not isinstance(replay, Mapping) or list(replay) != replay_keys:
        raise S1OverlayError("the S1 overlay replay keys changed")
    for key in REPLAY_KEYS:
        _required_string(replay[key], f"s0_replay {key}")
    if "selection" in replay and replay["selection"] != {
        "algorithm": "stratified_sha256_v1", "counts": V9_REPLAY_COUNTS,
    }:
        raise S1OverlayError("the S1 v9 replay allocation changed")
    return overlaid, payload
