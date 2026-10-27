"""Test the S1 v2 overlay and the corpus built from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import (
    S1_V2_OVERLAY_PATH,
    S1_V4_OVERLAY_PATH,
    S1_V5_OVERLAY_PATH,
    S1_OVERLAY_V3_SCHEMA_VERSION,
    S1_OVERLAY_V4_SCHEMA_VERSION,
    S1OverlayError,
    apply_s1_overlay,
    apply_train_form_counts,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
V2_CORPUS = REPO_ROOT / "data/world_model_v4/corpora/s1"


def _overlay_copy(tmp_path: Path, change) -> Path:
    payload = yaml.safe_load(S1_V2_OVERLAY_PATH.read_text(encoding="utf-8"))
    change(payload)
    path = tmp_path / "s1_overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_overlay_changes_only_s1_2_and_keeps_the_contract() -> None:
    contract = load_contract()
    before = json.dumps(contract, sort_keys=True)

    overlaid, payload = apply_s1_overlay(contract)

    assert json.dumps(contract, sort_keys=True) == before
    families = overlaid["curriculum"]["families"]
    base_families = contract["curriculum"]["families"]
    assert families["S1.2"]["shares"]["no_symbol"] == 0.31
    assert families["S1.2"]["output_directory"] == (
        "data/world_model_v4/corpora/s1/s1_2"
    )
    assert families["S1.2"]["train_rows"] == base_families["S1.2"]["train_rows"]
    for section, settings in families.items():
        if section != "S1.2":
            assert settings == base_families[section]
    assert payload["s0_replay"]["dataset_id"] == (
        "world_model_v4_s0_human_identifiers"
    )


def test_v4_overlay_enables_only_the_pinned_train_normalization() -> None:
    _, payload = apply_s1_overlay(load_contract(), S1_V4_OVERLAY_PATH)

    assert payload["schema_version"] == S1_OVERLAY_V3_SCHEMA_VERSION
    assert payload["train_normalization"] == {
        "evidence_id_policy": "aggregate_sequence_v1",
        "strip_s0_metadata_additions": True,
    }


def test_v4_overlay_rejects_a_train_normalization_change(tmp_path: Path) -> None:
    payload = yaml.safe_load(S1_V4_OVERLAY_PATH.read_text(encoding="utf-8"))
    payload["train_normalization"]["evidence_id_policy"] = "changed"
    path = tmp_path / "s1_v4.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(S1OverlayError, match="train normalization"):
        apply_s1_overlay(load_contract(), path)


def test_v5_overlay_sets_the_s1_1_answers_and_s1_2_train_counts() -> None:
    contract = load_contract()
    overlaid, payload = apply_s1_overlay(contract, S1_V5_OVERLAY_PATH)

    assert payload["schema_version"] == S1_OVERLAY_V4_SCHEMA_VERSION
    families = overlaid["curriculum"]["families"]
    settings = families["S1.1"]
    assert settings["membership_answers"] == {
        "absent_status": "resolved",
        "unknown_multiplex_present": None,
    }
    base = contract["curriculum"]["families"]["S1.1"]
    assert {key: value for key, value in settings.items() if key in base} == {
        **base,
        "output_directory": "data/world_model_v4/corpora/s1_v5/s1_1",
    }
    assert families["S1.2"]["train_form_counts"] == {
        1: 800,
        10: 650,
        11: 800,
        12: 650,
    }
    assert families["S1.3"]["membership_answers"] == {
        "keep_tool_fields": True,
        "none_present_status": "ambiguous",
    }
    assert families["S1.3"]["train_multiplex_reasoning"] is True
    assert families["S1.3"]["train_form_counts"] == {3: 850, 9: 850, 10: 600}
    assert families["S1.4"]["train_candidate_reasoning"] is True


def test_v5_overlay_gives_s1_3_7_the_lookup_status_in_its_copy_only() -> None:
    contract = load_contract()
    overlaid, _ = apply_s1_overlay(contract, S1_V5_OVERLAY_PATH)

    def status(source: dict) -> str:
        return next(
            template["status"]
            for template in source["blocked_reasons"]["codes"][
                "no_multiplex_membership"
            ]
            if template["id"] == "no_multiplex_membership.candidates"
        )

    assert status(overlaid) == "lookup"
    assert status(contract) == "not_found"
    v4, _ = apply_s1_overlay(contract, S1_V4_OVERLAY_PATH)
    assert status(v4) == "not_found"


def test_train_form_counts_keep_the_total() -> None:
    counts = {3: 1000, 9: 1000, 10: 300}
    assert apply_train_form_counts(counts, {3: 850, 9: 850, 10: 600}) == {
        3: 850,
        9: 850,
        10: 600,
    }
    with pytest.raises(S1OverlayError, match="change the total"):
        apply_train_form_counts(counts, {10: 600})


def test_v5_overlay_rejects_invalid_train_form_counts(tmp_path: Path) -> None:
    payload = yaml.safe_load(S1_V5_OVERLAY_PATH.read_text(encoding="utf-8"))
    payload["family_overrides"]["S1.2"]["train_form_counts"][11] = "800"
    path = tmp_path / "s1_v5.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(S1OverlayError, match="train form counts"):
        apply_s1_overlay(load_contract(), path)


def test_v5_overlay_rejects_a_membership_answer_change(tmp_path: Path) -> None:
    payload = yaml.safe_load(S1_V5_OVERLAY_PATH.read_text(encoding="utf-8"))
    payload["family_overrides"]["S1.1"]["membership_answers"]["absent_status"] = (
        "not_found"
    )
    path = tmp_path / "s1_v5.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(S1OverlayError, match="membership answers"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda payload: payload["base_contract"].update(sha256="0" * 64),
            "does not match the S1 overlay pin",
        ),
        (
            lambda payload: payload["family_overrides"]["S1.2"]["shares"].update(
                no_symbol=0.30
            ),
            "sum to",
        ),
        (
            lambda payload: payload["family_overrides"]["S1.2"]["shares"].pop(
                "malformed"
            ),
            "keep the contract share names",
        ),
        (
            lambda payload: payload["family_overrides"].update(
                {"S1.3": {"output_directory": "x", "shares": {}}}
            ),
            "unsupported family",
        ),
    ],
)
def test_overlay_rejects_an_invalid_change(
    tmp_path: Path, change, message: str
) -> None:
    with pytest.raises(S1OverlayError, match=message):
        apply_s1_overlay(load_contract(), _overlay_copy(tmp_path, change))


@pytest.mark.skipif(
    not (V2_CORPUS / "manifest.json").is_file(), reason="S1 v2 is not built"
)
def test_published_v2_corpus_uses_the_overlay_sources() -> None:
    manifest = json.loads((V2_CORPUS / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["contract"]["s1_overlay"]["path"] == (
        "runtime/world_model/s1_v2.yaml"
    )
    assert manifest["components"]["s1_2"]["manifest_path"].startswith(
        "data/world_model_v4/corpora/s1/s1_2/"
    )
    assert all(
        entry["manifest_path"].startswith("data/world_model_v3/corpora/s1/")
        for stage, entry in manifest["components"].items()
        if stage != "s1_2"
    )
    assert manifest["s0_replay"]["dataset_id"] == (
        "world_model_v4_s0_human_identifiers"
    )
    assert manifest["row_counts"] == {
        "train": 55276,
        "validation": 6875,
        "test": 6875,
    }
