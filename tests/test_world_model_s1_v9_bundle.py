"""Check selective v9 settings and stratified replay without changing S0 rows."""

from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest
import yaml

from runtime.world_model.contract import load_contract, system_prompt_sha256, tool_definitions
from runtime.world_model.s1_overlay import (
    S1OverlayError, S1_V6_OVERLAY_PATH, S1_V8_OVERLAY_PATH, S1_V9_OVERLAY_PATH,
    V9_REPLAY_COUNTS, apply_s1_overlay,
)
from scripts.build_world_model_s1_bundle import (
    REPO_ROOT, _load_replay_selection, replay_stratum,
)
from scripts.audit_world_model_s1_decorrelation import check_candidate_identity


def test_v9_keeps_identity_settings_and_selective_decorrelation():
    """The revision must not reintroduce the broad v8 identity changes."""
    base = load_contract()
    before = deepcopy(base)
    v6, _ = apply_s1_overlay(base, S1_V6_OVERLAY_PATH)
    v8, _ = apply_s1_overlay(base, S1_V8_OVERLAY_PATH)
    v9, _ = apply_s1_overlay(base, S1_V9_OVERLAY_PATH)
    assert base == before
    for section in ("S1.1", "S1.2", "S1.4", "S1.9"):
        expected = deepcopy((v8 if section == "S1.9" else v6)["curriculum"]["families"][section])
        actual = deepcopy(v9["curriculum"]["families"][section])
        expected.pop("output_directory")
        actual.pop("output_directory")
        assert actual == expected
    for section in ("S1.3", "S1.6", "S1.7", "S1.8"):
        assert v9["curriculum"]["families"][section]["train_targeted_v9"] is True


@pytest.mark.parametrize("change", ["replay", "scope_pin", "off"])
def test_v9_rejects_changed_allocation_or_disabled_settings(tmp_path, change):
    """V9 pins its replay counts, choice phrases, and training-only switches."""
    payload = yaml.safe_load(S1_V9_OVERLAY_PATH.read_text())
    if change == "replay":
        payload["s0_replay"]["selection"]["counts"]["ambiguous_symbol"] -= 1
    elif change == "scope_pin":
        payload["family_overrides"]["S1.3"]["train_choice_replies"]["sha256"] = "0" * 64
    else:
        payload["family_overrides"]["S1.6"]["train_targeted_v9"] = False
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError):
        apply_s1_overlay(load_contract(), path)


def test_v9_replay_preserves_source_rows_and_balances_framing():
    """Cover rare ambiguity inputs while keeping source facts and text intact."""
    contract, overlay = apply_s1_overlay(load_contract(), S1_V9_OVERLAY_PATH)
    selected = _load_replay_selection(
        repo_root=REPO_ROOT, contract=contract,
        expected_tools=tool_definitions(contract),
        prompt_sha256=system_prompt_sha256(contract), s1_train_row_count=55000,
        replay_source=overlay["s0_replay"],
    )
    assert len(selected.rows) == 276
    assert Counter(replay_stratum(row) for _, row in selected.rows) == V9_REPLAY_COUNTS
    assert len({row["row_id"] for _, row in selected.rows}) == 276
    for kind, count in V9_REPLAY_COUNTS.items():
        frames = Counter(row["metadata"]["framing"] for _, row in selected.rows
                         if replay_stratum(row) == kind)
        assert frames == {"direct": count // 2, "step": count // 2}
    source_rows = {json.loads(line)["row_id"]: line for line in selected.source_path.open("rb")}
    assert all(source_rows[row["row_id"]] == raw for raw, row in selected.rows)
    assert selected.selection_strata["ambiguous_cells"]
    ambiguous = [row for _, row in selected.rows if replay_stratum(row) == "ambiguous_symbol"]
    assert len({len(row["messages"][-1]["content"]["result"]["candidate_gene_ids"])
                for row in ambiguous}) > 2
    assert {row["metadata"]["input_variant"] for row in ambiguous} == {"canonical", "casefold"}
    expected_ids = [row["row_id"] for _, row in selected.rows]
    expected_ids.sort(key=lambda value: (hashlib.sha256(f"{selected.seed}:{value}".encode()).hexdigest(), value))
    assert [row["row_id"] for _, row in selected.rows] == expected_ids


def test_audit_rejects_an_unrelated_overlay_before_tokenization(tmp_path):
    """An audit of v9 must not silently record the default v8 overlay hash."""
    pin = {
        "path": str(S1_V9_OVERLAY_PATH.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(S1_V9_OVERLAY_PATH.read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"contract": {"s1_overlay": pin}}))
    with pytest.raises(ValueError, match="overlay differs"):
        check_candidate_identity(tmp_path, S1_V8_OVERLAY_PATH)
