"""Check multiplex-map selection, native roles, scope slots and split isolation."""

import json

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_4_prompts import (
    DIRECTIONS, MODES, load_s2_4_requests, render_s2_4_request,
    validate_s2_4_rendered_prompts, validate_s2_4_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / "runtime/world_model/s2_4_v1_requests.json"
SCOPE = {"kind": "layers", "layer_ids": ["LayerB", "LayerA"], "layer_family": None}


@pytest.fixture
def requests():
    """Give each mutation test an independent frozen request bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read immutable neutral wrappers and the three preceding request banks."""
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in (
        "s2_v1_prompt_banks.json", "s2_v1_edge_requests.json",
        "s2_2_v1_requests.json", "s2_3_v1_requests.json",
    ))


def test_pinned_bank_declares_both_selections_and_all_native_directions(requests):
    contract = json.loads((REQUEST_PATH.parent / "s2_4_v1.json").read_text())
    assert load_s2_4_requests(contract["request_bank"]) == requests
    report = validate_s2_4_requests(requests)
    assert sum(n for split in report["template_counts"].values()
               for mode in split.values() for n in mode.values()) == 96
    assert report["ranking_template_counts"] == {"train": 4, "validation": 4, "test": 4}


def test_rendered_questions_remain_disjoint_from_every_prior_family(requests, prior_banks):
    report = validate_s2_4_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report["s2_4_rendered_questions"] == {"train": 2304, "validation": 576, "test": 576}
    assert report["cross_split_pairs_checked"] == 28901376
    assert report["maximum_cross_split_jaccard"] < 0.6
    assert report["shared_cross_split_5grams"] == 0


def test_prior_family_heldout_wording_cannot_enter_new_training_requests(requests, prior_banks):
    requests["train"]["complete"]["incident"][0] += " Return the associated weight without changing it."
    validate_s2_4_requests(requests)
    with pytest.raises(PromptBankError, match="share a 5-word run"):
        validate_s2_4_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_rendering_preserves_scope_order_and_only_binds_preselected_content(requests, mode, direction):
    text = render_s2_4_request(requests, split="train", mode=mode, direction=direction,
        variant_index=5, multiplex_id="MultiplexExample", layer_scope=SCOPE,
        gene="GeneExample", top_k=7 if mode == "top_k" else None)
    values = {"multiplex_id": "MultiplexExample", "scope_text": "layers [LayerB, LayerA]", "gene": "GeneExample"}
    if mode == "top_k":
        values["top_k"] = 7
    assert text == requests["train"][mode][direction][1].format(**values)
    assert text.count("GeneExample") == 1
    assert "LayerB, LayerA" in text


@pytest.mark.parametrize("family,expected", [(None, "all layers"), ("TFs", "all layers in family TFs")])
def test_all_selectors_remain_exhaustive_in_user_wording(requests, family, expected):
    text = render_s2_4_request(requests, split="test", mode="complete", direction="incident",
        variant_index=0, multiplex_id="M", gene="Q",
        layer_scope={"kind": "all", "layer_ids": [], "layer_family": family})
    assert expected in text


@pytest.mark.parametrize("mode,top_k", [
    ("top_k", None), ("top_k", 0), ("top_k", -1), ("top_k", True),
    ("top_k", "7"), ("complete", 7),
])
def test_mode_and_bound_cannot_disagree(requests, mode, top_k):
    with pytest.raises(PromptBankError, match="top_k"):
        render_s2_4_request(requests, split="train", direction="incident", mode=mode,
            variant_index=0, multiplex_id="M", layer_scope=SCOPE, gene="Q", top_k=top_k)


def test_unresolved_strength_request_uses_its_frozen_diagnostic_wording(requests):
    text = render_s2_4_request(requests, split="validation", direction="incident", mode="complete",
        variant_index=5, multiplex_id="M", layer_scope=SCOPE, gene="Q", ranking="strongest_unspecified")
    assert text == requests["ranking_requests"]["validation"][1].format(
        multiplex_id="M", scope_text="layers [LayerB, LayerA]", gene="Q")


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "conversion", "format"])
def test_exact_request_slots_cannot_change(requests, change):
    template = requests["train"]["complete"]["incident"][0]
    if change == "missing":
        template = template.replace("{scope_text}", "the current layers")
    elif change == "duplicate":
        template += " {gene}"
    elif change == "extra":
        template += " {outcome}"
    elif change == "conversion":
        template = template.replace("{gene}", "{gene!r}")
    else:
        template = template.replace("{gene}", "{gene:>20}")
    requests["train"]["complete"]["incident"][0] = template
    with pytest.raises(PromptBankError, match="declared fields once"):
        validate_s2_4_requests(requests)


def test_bounded_union_cannot_silently_change_to_weight_ranking(requests):
    requests["train"]["top_k"]["incident"][0] += " Choose the strongest rows by descending weight."
    with pytest.raises(PromptBankError, match="rank across source weights"):
        validate_s2_4_requests(requests)


@pytest.mark.parametrize("direction,old,new", [
    ("in", "into target {gene}", "from source {gene}"),
    ("out", "from source {gene}", "into target {gene}"),
])
def test_directional_query_roles_cannot_be_reversed(requests, direction, old, new):
    requests["train"]["complete"][direction][0] = requests["train"]["complete"][direction][0].replace(old, new)
    with pytest.raises(PromptBankError, match="role"):
        validate_s2_4_requests(requests)


def test_incident_wording_cannot_drop_outgoing_support(requests):
    requests["train"]["complete"]["incident"][0] = requests["train"]["complete"]["incident"][0].replace(
        "incoming and outgoing native arcs", "incoming native arcs")
    with pytest.raises(PromptBankError, match="all native orientations"):
        validate_s2_4_requests(requests)


def test_request_file_tampering_is_rejected(requests, tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(requests))
    pin = {"path": str(path), "sha256": file_sha256(path)}
    path.write_text(path.read_text() + "\n")
    with pytest.raises(PromptBankError, match="does not match its pin"):
        load_s2_4_requests(pin)
