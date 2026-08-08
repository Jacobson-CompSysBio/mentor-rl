"""Define and check exact S0 contracts for human gene identifiers.

This module checks a record after the schema parser creates it. It does not
compare a model answer with the private answer key or calculate metrics.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from .world_model_schemas import (
    FieldValidatorSpec,
    IdentifierSFTMetadata,
    WorldModelRecord,
)


# These names define the only question families that S0 accepts.
SYMBOL_TO_ENSEMBL_FAMILY = "human_symbol_to_ensembl"
ENSEMBL_TO_SYMBOL_FAMILY = "human_ensembl_to_symbol"
AMBIGUOUS_SYMBOL_FAMILY = "human_ambiguous_symbol"
S0_FAMILIES = (
    SYMBOL_TO_ENSEMBL_FAMILY,
    ENSEMBL_TO_SYMBOL_FAMILY,
    AMBIGUOUS_SYMBOL_FAMILY,
)
RESOLVED_STATUS = "resolved"
AMBIGUOUS_STATUS = "ambiguous"
AMBIGUITY_ACTION = "defer"

# Ensembl human gene IDs use ENSG and exactly 11 decimal digits.
ENSEMBL_PATTERN = re.compile(r"^ENSG[0-9]{11}$")

# Each dataset split uses a different question form for the same fact.
SPLIT_PROMPT_FORMS = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


def expected_answer_keys(family: str) -> set[str]:
    """Return the exact answer keys for one S0 family.

    Exact key sets reject omitted fields and extra model output fields.
    """

    if family == SYMBOL_TO_ENSEMBL_FAMILY:
        return {"status", "gene_id", "gene_symbol"}
    if family == ENSEMBL_TO_SYMBOL_FAMILY:
        return {"status", "gene_id", "gene_symbols"}
    if family == AMBIGUOUS_SYMBOL_FAMILY:
        return {
            "status",
            "gene_symbol",
            "candidate_gene_ids",
            "action",
        }
    raise ValueError(f"Unknown S0 family: {family!r}")


def validator_specs_for_answer(
    family: str,
) -> dict[str, FieldValidatorSpec]:
    """Return the exact field validator set for one S0 family.

    The corpus builder writes these contracts into each record. This function
    supplies an independent copy for runtime checks.
    """

    # The first two families require one resolved Ensembl gene ID.
    common = {
        "status": FieldValidatorSpec(
            primitive="ENUM", options={"values": [RESOLVED_STATUS]}
        ),
        "gene_id": FieldValidatorSpec(
            primitive="EXACT_ID",
            options={"pattern": ENSEMBL_PATTERN.pattern},
        ),
    }
    if family == SYMBOL_TO_ENSEMBL_FAMILY:
        # S0.1 returns the input symbol and its one resolved gene ID.
        return {
            **common,
            "gene_symbol": FieldValidatorSpec(primitive="SYMBOL"),
        }
    if family == ENSEMBL_TO_SYMBOL_FAMILY:
        # S0.2 returns every symbol for the input gene ID as one set.
        return {
            **common,
            "gene_symbols": FieldValidatorSpec(primitive="SYMBOL_SET"),
        }
    if family == AMBIGUOUS_SYMBOL_FAMILY:
        # S0.3 returns every candidate gene ID and requires defer.
        return {
            "status": FieldValidatorSpec(
                primitive="ENUM", options={"values": [AMBIGUOUS_STATUS]}
            ),
            "gene_symbol": FieldValidatorSpec(primitive="SYMBOL"),
            "candidate_gene_ids": FieldValidatorSpec(
                primitive="ID_SET",
                options={"item_pattern": ENSEMBL_PATTERN.pattern},
            ),
            "action": FieldValidatorSpec(
                primitive="ENUM", options={"values": [AMBIGUITY_ACTION]}
            ),
        }
    raise ValueError(f"Unknown S0 family: {family!r}")


def _valid_ensembl(value: Any) -> bool:
    """Return true for one canonical human Ensembl gene ID."""

    return (
        isinstance(value, str)
        and ENSEMBL_PATTERN.fullmatch(value) is not None
    )


def _valid_symbol(value: Any) -> bool:
    """Return true for one nonempty gene symbol string."""

    return isinstance(value, str) and bool(value.strip())


def _validate_sorted_unique_strings(
    value: Any,
    *,
    field_name: str,
    minimum_size: int,
    item_validator: Callable[[Any], bool],
) -> list[str]:
    """Check one canonical set that uses a JSON list.

    The JSON list must have enough valid, unique values in sorted order.
    """

    # Check the container before operations that require a list.
    if not isinstance(value, list) or len(value) < minimum_size:
        return [
            f"{field_name} must contain at least {minimum_size} values"
        ]
    if any(not item_validator(item) for item in value):
        return [f"{field_name} contains an invalid value"]

    # The source builder defines canonical set order as Python string order.
    if len(value) != len(set(value)):
        return [f"{field_name} contains a duplicate"]
    if value != sorted(value):
        return [f"{field_name} must use canonical order"]
    return []


def _validator_payload(
    validators: Mapping[str, FieldValidatorSpec],
) -> dict[str, dict[str, Any]]:
    """Convert validator objects to their canonical JSON form.

    This form permits an exact comparison with the record validator block.
    """

    return {
        key: {
            "primitive": value.primitive,
            "required": value.required,
            "options": dict(value.options),
        }
        for key, value in validators.items()
    }


def _validate_identity(record: WorldModelRecord) -> list[str]:
    """Check the split, fact identity, and prompt form fields.

    These checks keep one fact stable across train, validation, and test forms.
    """

    errors: list[str] = []
    provenance = record.provenance
    if not isinstance(provenance, Mapping):
        return ["S0 provenance must be one object"]

    # The fact ID joins render forms. The group ID keeps linked facts together.
    for field_name in ("fact_id", "fact_group_id"):
        value = provenance.get(field_name)
        if not isinstance(value, str) or not value:
            errors.append(f"provenance.{field_name} must be a non-empty string")
    if provenance.get("fact_role") != "seen":
        errors.append("provenance.fact_role must be seen")

    # Each prompt form has a fixed nonnegative render index.
    rendering_index = provenance.get("rendering_index")
    if not isinstance(rendering_index, int) or isinstance(
        rendering_index, bool
    ) or rendering_index < 0:
        errors.append(
            "provenance.rendering_index must be a nonnegative integer"
        )

    # Reject a train question in a validation or test record.
    if record.split is not None:
        expected_form = SPLIT_PROMPT_FORMS[record.split]
        if provenance.get("prompt_form_id") != expected_form:
            errors.append(
                "provenance.prompt_form_id does not match the record split"
            )
    return errors


def validate_s0_contract(record: WorldModelRecord) -> list[str]:
    """Return all contract errors for one complete S0 record.

    This function checks self-consistency only. A scorer must compare model
    output with the private answer key after this check.
    """

    errors: list[str] = []
    metadata = record.metadata
    if not isinstance(metadata, IdentifierSFTMetadata):
        return ["S0 records require identifier_sft_v2 metadata"]

    # Reject another curriculum stage or an unknown S0 family.
    family = metadata.question_family
    if family not in S0_FAMILIES:
        return [f"unknown S0 family {family!r}"]
    if record.context is not None:
        errors.append("S0 records require null context")
    errors.extend(_validate_identity(record))

    # S0.2 accepts one gene ID. S0.1 and S0.3 accept one gene symbol.
    input_value = record.input
    expected_input_keys = (
        {"gene_id"}
        if family == ENSEMBL_TO_SYMBOL_FAMILY
        else {"gene_symbol"}
    )
    if not isinstance(input_value, Mapping) or set(input_value) != (
        expected_input_keys
    ):
        errors.append(
            f"input keys differ from {family} contract: "
            f"expected={sorted(expected_input_keys)} "
            f"actual={sorted(input_value) if isinstance(input_value, Mapping) else []}"
        )
    elif family == ENSEMBL_TO_SYMBOL_FAMILY:
        if not _valid_ensembl(input_value.get("gene_id")):
            errors.append(
                "input.gene_id is not a canonical Ensembl gene ID"
            )
        elif input_value.get("gene_id") != record.answer.get("gene_id"):
            errors.append("input.gene_id does not match answer.gene_id")
    else:
        if not _valid_symbol(input_value.get("gene_symbol")):
            errors.append("input.gene_symbol must be a non-empty string")
        elif input_value.get("gene_symbol") != record.answer.get(
            "gene_symbol"
        ):
            errors.append(
                "input.gene_symbol does not match answer.gene_symbol"
            )

    # Require the exact answer and validator shapes for the selected family.
    expected_keys = expected_answer_keys(family)
    if set(record.answer) != expected_keys:
        errors.append(
            f"answer keys differ from {family} contract: "
            f"expected={sorted(expected_keys)} "
            f"actual={sorted(record.answer)}"
        )
    if _validator_payload(record.validators) != _validator_payload(
        validator_specs_for_answer(family)
    ):
        errors.append("validator specs differ from the S0 family contract")

    answer = record.answer
    if family == SYMBOL_TO_ENSEMBL_FAMILY:
        # S0.1 requires one resolved gene ID and the original symbol.
        if answer.get("status") != RESOLVED_STATUS:
            errors.append("resolved mapping status must be resolved")
        if not _valid_ensembl(answer.get("gene_id")):
            errors.append("gene_id is not a canonical Ensembl gene ID")
        if not _valid_symbol(answer.get("gene_symbol")):
            errors.append("gene_symbol must be a non-empty string")
    elif family == ENSEMBL_TO_SYMBOL_FAMILY:
        # S0.2 requires the complete canonical symbol set for one gene ID.
        if answer.get("status") != RESOLVED_STATUS:
            errors.append("resolved mapping status must be resolved")
        if not _valid_ensembl(answer.get("gene_id")):
            errors.append("gene_id is not a canonical Ensembl gene ID")
        errors.extend(
            _validate_sorted_unique_strings(
                answer.get("gene_symbols"),
                field_name="gene_symbols",
                minimum_size=1,
                item_validator=_valid_symbol,
            )
        )
    else:
        # S0.3 requires all candidate IDs and an explicit defer action.
        if answer.get("status") != AMBIGUOUS_STATUS:
            errors.append("ambiguous mapping status must be ambiguous")
        if answer.get("action") != AMBIGUITY_ACTION:
            errors.append("ambiguous mapping action must be defer")
        if not _valid_symbol(answer.get("gene_symbol")):
            errors.append("gene_symbol must be a non-empty string")
        errors.extend(
            _validate_sorted_unique_strings(
                answer.get("candidate_gene_ids"),
                field_name="candidate_gene_ids",
                minimum_size=2,
                item_validator=_valid_ensembl,
            )
        )
    return errors
