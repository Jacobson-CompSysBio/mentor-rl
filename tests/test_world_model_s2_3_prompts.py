"""Check neighbor request bindings, directional scope, and frozen split wording."""

import json

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError, _tokens
from runtime.world_model.s2_3_prompts import (
    DIRECTIONS,
    MODES,
    load_s2_3_requests,
    render_s2_3_request,
    validate_s2_3_rendered_prompts,
    validate_s2_3_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / "runtime/world_model/s2_3_v1_requests.json"


@pytest.fixture
def requests():
    """Read an independent bank for each failure-injection test."""

    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def frozen_banks():
    """Load immutable neutral wrappers and preceding S2 family requests."""

    return tuple(json.loads((REPOSITORY_ROOT / "runtime/world_model" / name).read_text()) for name in (
        "s2_v1_prompt_banks.json", "s2_v1_edge_requests.json", "s2_2_v1_requests.json",
    ))


def test_pinned_bank_covers_both_modes_and_all_directions(requests):
    assert load_s2_3_requests({"path": str(REQUEST_PATH), "sha256": file_sha256(REQUEST_PATH)}) == requests
    report = validate_s2_3_requests(requests)
    assert sum(count for split in report["template_counts"].values() for mode in split.values() for count in mode.values()) == 96


def test_full_questions_preserve_cross_split_separation_from_prior_families(requests, frozen_banks):
    report = validate_s2_3_rendered_prompts(frozen_banks[0], requests, *frozen_banks[1:])
    assert report["s2_3_rendered_questions"] == {"train": 2048, "validation": 512, "test": 512}
    assert report["s2_2_rendered_questions"] == {"train": 2304, "validation": 576, "test": 576}
    assert report["s2_1_rendered_questions"] == {"train": 512, "validation": 128, "test": 128}
    assert report["cross_split_pairs_checked"] == 13307904
    assert report["maximum_cross_split_jaccard"] < 0.6
    assert report["shared_cross_split_5grams"] == 0


@pytest.mark.parametrize("phrase", [
    "Return the associated weight without changing it.",
    "Select the earliest qualifying layers according to manifest position.",
])
def test_prior_family_heldout_runs_are_detected_in_new_training_wording(requests, frozen_banks, phrase):
    requests["train"]["complete"]["undirected"][0] += " " + phrase
    validate_s2_3_requests(requests)
    with pytest.raises(PromptBankError, match="share a 5-word run"):
        validate_s2_3_rendered_prompts(frozen_banks[0], requests, *frozen_banks[1:])


def test_cross_split_similarity_is_checked_even_without_copied_word_runs(requests, frozen_banks):
    # A reversed token set retains vocabulary overlap but creates no copied
    # five-word run. The operation and binding checks must still pass first.
    template = requests["validation"]["complete"]["undirected"][0]
    tokens = _tokens(template.format(multiplex_id="", layer_id="", gene=""))
    scrambled = " ".join(reversed(tokens))
    request = "{multiplex_id} {layer_id} {gene} " + scrambled
    requests["train"]["complete"]["undirected"][0] = request
    validate_s2_3_requests(requests)
    with pytest.raises(PromptBankError, match="Jaccard="):
        validate_s2_3_rendered_prompts(frozen_banks[0], requests, *frozen_banks[1:])


def test_rendering_binds_only_preselected_content_and_wraps_wording_index(requests):
    fields = dict(multiplex_id="multiplex_example", layer_id="layer_example", gene="GENE_EXAMPLE")
    for split in ("train", "validation", "test"):
        for mode in MODES:
            for direction in DIRECTIONS:
                for index in range(8):
                    text = render_s2_3_request(
                        requests, split=split, mode=mode, direction=direction,
                        variant_index=index, **fields, top_k=7 if mode == "top_k" else None,
                    )
                    expected = dict(fields, **({"top_k": 7} if mode == "top_k" else {}))
                    assert text == requests[split][mode][direction][index % 4].format(**expected)
                    assert all(text.count(value) == 1 for value in fields.values())


@pytest.mark.parametrize("mode,top_k", [
    ("top_k", None), ("top_k", 0), ("top_k", -1), ("top_k", True),
    ("top_k", "7"), ("complete", 7),
])
def test_selection_and_bound_cannot_conflict(requests, mode, top_k):
    with pytest.raises(PromptBankError, match="top_k"):
        render_s2_3_request(
            requests, split="train", mode=mode, direction="undirected",
            variant_index=0, multiplex_id="M", layer_id="L", gene="Q", top_k=top_k,
        )


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "conversion", "format"])
def test_template_fields_are_bound_exactly_once(requests, change):
    template = requests["train"]["complete"]["in"][0]
    if change == "missing":
        template = template.replace("{layer_id}", "the active layer")
    elif change == "duplicate":
        template += " {gene}"
    elif change == "extra":
        template += " {outcome}"
    elif change == "conversion":
        template = template.replace("{gene}", "{gene!r}")
    else:
        template = template.replace("{gene}", "{gene:>20}")
    requests["train"]["complete"]["in"][0] = template
    with pytest.raises(PromptBankError, match="declared fields once"):
        validate_s2_3_requests(requests)


@pytest.mark.parametrize("direction,role", [("in", "source"), ("out", "target")])
def test_single_direction_endpoint_inversion_is_rejected(requests, direction, role):
    name = "incoming" if direction == "in" else "outgoing"
    requests["train"]["complete"][direction][0] = (
        f"List every {name} neighbor of {role} {{gene}} in {{multiplex_id}}/{{layer_id}}, with weights."
    )
    with pytest.raises(PromptBankError, match="endpoint role"):
        validate_s2_3_requests(requests)


def test_incoming_origin_is_not_mistaken_for_target_binding(requests):
    requests["train"]["complete"]["in"][0] = (
        "List every incoming neighbor whose arc originates at {gene} "
        "in {multiplex_id}/{layer_id}, with weights."
    )
    with pytest.raises(PromptBankError, match="endpoint role"):
        validate_s2_3_requests(requests)


def test_undirected_request_cannot_invent_regulator_roles(requests):
    requests["train"]["complete"]["undirected"][0] += " Treat these as outgoing targets."
    with pytest.raises(PromptBankError, match="invent directional roles"):
        validate_s2_3_requests(requests)


def test_both_direction_roles_cannot_be_reversed(requests):
    template = requests["train"]["complete"]["both"][0]
    requests["train"]["complete"]["both"][0] = template.replace("incoming N → Q", "incoming Q → N")
    with pytest.raises(PromptBankError, match="both endpoint roles"):
        validate_s2_3_requests(requests)


def test_both_direction_bound_must_apply_per_side(requests):
    template = requests["train"]["top_k"]["both"][0]
    requests["train"]["top_k"]["both"][0] = template.split(" Apply that bound")[0]
    with pytest.raises(PromptBankError, match="K separately per direction"):
        validate_s2_3_requests(requests)


def test_combined_ranking_cannot_be_added_to_independent_bounds(requests):
    requests["train"]["top_k"]["both"][0] += " Use a global rank by summed weights."
    with pytest.raises(PromptBankError, match="combined directional ranking"):
        validate_s2_3_requests(requests)


def test_bounded_request_requires_weight_ranking(requests):
    requests["train"]["top_k"]["undirected"][0] = requests["train"]["top_k"]["undirected"][0].replace("strongest", "first")
    with pytest.raises(PromptBankError, match="strongest neighbors"):
        validate_s2_3_requests(requests)


def test_complete_request_cannot_silently_become_bounded(requests):
    requests["train"]["complete"]["undirected"][0] += " Return only the strongest neighbors."
    with pytest.raises(PromptBankError, match="ranked bound"):
        validate_s2_3_requests(requests)


def test_request_file_tampering_is_rejected(requests, tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(requests))
    pin = {"path": str(path), "sha256": file_sha256(path)}
    path.write_text(path.read_text() + "\n")
    with pytest.raises(PromptBankError, match="does not match its pin"):
        load_s2_3_requests(pin)
