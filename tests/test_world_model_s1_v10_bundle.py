"""Keep v10 scope fixes isolated from frozen data and historical overlays."""

from copy import deepcopy

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import (
    S1OverlayError, S1_V9_OVERLAY_PATH, S1_V10_OVERLAY_PATH, apply_s1_overlay,
)


def test_v10_changes_only_approved_settings():
    """Keep all allocations, source pins, replay selection, and old settings intact."""
    base = load_contract()
    before = deepcopy(base)
    v9, old = apply_s1_overlay(base, S1_V9_OVERLAY_PATH)
    v10, new = apply_s1_overlay(base, S1_V10_OVERLAY_PATH)
    assert base == before
    for section in v9["curriculum"]["families"]:
        expected = deepcopy(v9["curriculum"]["families"][section])
        actual = deepcopy(v10["curriculum"]["families"][section])
        expected.pop("output_directory", None)
        actual.pop("output_directory", None)
        if section == "S1.1":
            assert actual.pop("train_id_confirmation_reasoning")
        elif section == "S1.3":
            assert actual.pop("train_targeted_v10") is True
            actual["train_user_choice_reasoning"] = expected["train_user_choice_reasoning"]
        elif section == "S1.8":
            for template in actual["train_final_reasoning"][4].values():
                assert template.count("{outer_scope}") == 1
            actual["train_final_reasoning"][4] = expected["train_final_reasoning"][4]
        assert actual == expected, section
    for key in ("base_contract", "prompt_banks", "train_normalization", "s0_replay"):
        assert old[key] == new[key]


@pytest.mark.parametrize("change", ["id_slot", "metric_slot", "choice_off", "v9_off", "legacy_schema", "replay"])
def test_v10_rejects_incomplete_or_unpinned_changes(tmp_path, change):
    """Fail closed when scope binding, prerequisite coverage, or replay changes."""
    payload = yaml.safe_load(S1_V10_OVERLAY_PATH.read_text())
    families = payload["family_overrides"]
    if change == "id_slot":
        templates = families["S1.1"]["train_id_confirmation_reasoning"]
        templates["step"] = templates["step"].replace("{user_request}", "the request")
    elif change == "metric_slot":
        templates = families["S1.8"]["train_final_reasoning"][4]
        templates["direct"] = templates["direct"].replace("{outer_scope}", "the request")
    elif change == "choice_off":
        families["S1.3"]["train_targeted_v10"] = False
    elif change == "v9_off":
        families["S1.3"]["train_targeted_v9"] = False
    elif change == "legacy_schema":
        payload["schema_version"] = "mentor-rl-world-model-s1-overlay-v8"
    else:
        payload["s0_replay"]["selection"]["counts"]["ambiguous_symbol"] -= 1
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError):
        apply_s1_overlay(load_contract(), path)
