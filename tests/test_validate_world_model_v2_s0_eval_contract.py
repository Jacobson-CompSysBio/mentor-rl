"""Test the S0 test contract validator."""

from __future__ import annotations

import pytest

from scripts.validate_world_model_v2_s0_eval_contract import (
    EXPOSURE_SCHEMA_VERSION,
    require_full_exposure_checkpoint,
)


METHOD_ID = "test-method"


def full_exposure(*, train_rows: int) -> dict:
    """Return one complete exposure receipt."""

    return {
        "schema_version": EXPOSURE_SCHEMA_VERSION,
        "status": "complete",
        "method_id": METHOD_ID,
        "corpus": {"eligible_train_rows": train_rows},
        "logical_exposure": {
            "all_eligible_train_rows_exposed": True,
        },
        "exposure_contract": {
            "scope": "all_eligible_train_rows",
            "satisfied": True,
        },
    }


def test_full_exposure_requires_source_row_count() -> None:
    """The test gate rejects a subset exposure receipt."""

    exposure = full_exposure(train_rows=3)

    with pytest.raises(
        SystemExit,
        match="complete full-exposure checkpoint",
    ):
        require_full_exposure_checkpoint(
            exposure,
            method_id=METHOD_ID,
            eligible_train_rows=4,
        )


def test_full_exposure_accepts_source_row_count() -> None:
    """The test gate accepts the full source row count."""

    exposure = full_exposure(train_rows=4)

    require_full_exposure_checkpoint(
        exposure,
        method_id=METHOD_ID,
        eligible_train_rows=4,
    )


def test_full_exposure_rejects_debug_scope() -> None:
    """The test gate rejects a debug exposure receipt."""

    exposure = full_exposure(train_rows=4)
    exposure["exposure_contract"]["scope"] = "bounded_debug_subset"

    with pytest.raises(
        SystemExit,
        match="complete full-exposure checkpoint",
    ):
        require_full_exposure_checkpoint(
            exposure,
            method_id=METHOD_ID,
            eligible_train_rows=4,
        )
