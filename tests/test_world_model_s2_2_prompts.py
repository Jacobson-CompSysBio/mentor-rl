"""Test S2.2 framing semantics and full-question separation from frozen S2.1."""

import json

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_2_prompts import (
    EDGE_QUERIES,
    MODES,
    load_s2_2_requests,
    render_s2_2_request,
    validate_s2_2_rendered_prompts,
    validate_s2_2_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / "runtime/world_model/s2_2_v1_requests.json"


@pytest.fixture
def requests():
    """Read a fresh request bank so failure tests cannot contaminate each other."""

    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def frozen_banks():
    """Read the unchanged neutral wrappers and S2.1 core requests."""

    paths = (
        REPOSITORY_ROOT / "runtime/world_model/s2_v1_prompt_banks.json",
        REPOSITORY_ROOT / "runtime/world_model/s2_v1_edge_requests.json",
    )
    return tuple(json.loads(path.read_text()) for path in paths)


def test_pinned_bank_covers_all_modes_and_predicates(requests):
    pin = {"path": str(REQUEST_PATH), "sha256": file_sha256(REQUEST_PATH)}
    assert load_s2_2_requests(pin) == requests
    report = validate_s2_2_requests(requests)
    assert sum(
        count
        for split in report["template_counts"].values()
        for mode in split.values()
        for count in mode.values()
    ) == 108


def test_full_questions_are_separated_across_families_and_splits(requests, frozen_banks):
    wrappers, previous = frozen_banks
    report = validate_s2_2_rendered_prompts(wrappers, requests, previous)
    assert report["s2_2_rendered_questions"] == {"train": 2304, "validation": 576, "test": 576}
    assert report["s2_1_rendered_questions"] == {"train": 512, "validation": 128, "test": 128}
    assert report["cross_split_pairs_checked"] == 4460544
    assert report["maximum_cross_split_jaccard"] < 0.6
    assert report["shared_cross_split_5grams"] == 0


@pytest.mark.parametrize("split,phrase", [
    ("train", "Return the associated weight without changing it."),
    ("test", "Give its recorded weight if it exists."),
])
def test_frozen_s2_1_wording_cannot_leak_across_splits(requests, frozen_banks, split, phrase):
    requests[split]["complete"]["any_direction"][0] += " " + phrase
    validate_s2_2_requests(requests)
    with pytest.raises(PromptBankError, match="share a 5-word run"):
        validate_s2_2_rendered_prompts(frozen_banks[0], requests, frozen_banks[1])


def test_each_renderer_binds_the_selected_content_once(requests):
    fields = dict(
        multiplex_id="multiplex_example", first="NFYA", second="KLF3",
        scope_text="the selected TF-target layers",
    )
    for split in ("train", "validation", "test"):
        for mode in MODES:
            for query in EDGE_QUERIES:
                for index in range(8):
                    kwargs = {**fields, "top_k": 3 if mode == "top_k" else None}
                    text = render_s2_2_request(
                        requests, split=split, mode=mode, edge_query=query,
                        variant_index=index, **kwargs,
                    )
                    expected_fields = dict(fields)
                    if mode == "top_k":
                        expected_fields["top_k"] = "3"
                    assert text == requests[split][mode][query][index % 4].format(**expected_fields)
                    assert all(text.count(value) == 1 for value in fields.values())
                    assert text.index("NFYA") < text.index("KLF3")


@pytest.mark.parametrize("mode,top_k", [
    ("top_k", None), ("top_k", 0), ("top_k", True), ("top_k", "3"),
    ("complete", 1), ("exists", 1),
])
def test_mode_and_k_cannot_conflict(requests, mode, top_k):
    with pytest.raises(PromptBankError, match="top_k"):
        render_s2_2_request(
            requests, split="train", mode=mode, edge_query="any_direction",
            variant_index=0, multiplex_id="M", first="A", second="B",
            scope_text="all registered layers", top_k=top_k,
        )


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "conversion"])
def test_template_slots_are_exact(requests, change):
    template = requests["train"]["complete"]["source_to_target"][0]
    if change == "missing":
        template = template.replace("{scope_text}", "some layers")
    elif change == "duplicate":
        template += " {first}"
    elif change == "extra":
        template += " {unknown}"
    else:
        template = template.replace("{second}", "{second!r}")
    requests["train"]["complete"]["source_to_target"][0] = template
    with pytest.raises(PromptBankError, match="declared fields once"):
        validate_s2_2_requests(requests)


def test_directional_role_inversion_is_rejected(requests):
    requests["train"]["exists"]["source_to_target"][0] = (
        "Within {scope_text} of {multiplex_id}, does an arc with target {first} "
        "and source {second} occur in any layer?"
    )
    with pytest.raises(PromptBankError, match="bind ordered roles"):
        validate_s2_2_requests(requests)


def test_reciprocity_cannot_be_split_across_different_layers(requests):
    requests["train"]["exists"]["both_directions"][0] = (
        "Within {scope_text} of {multiplex_id}, do both directed orientations "
        "between {first} and {second} occur across different layers?"
    )
    with pytest.raises(PromptBankError, match="both orientations in one layer"):
        validate_s2_2_requests(requests)


def test_any_direction_cannot_silently_drop_directed_edges(requests):
    requests["train"]["exists"]["any_direction"][0] = (
        "Within {scope_text} of {multiplex_id}, does an undirected edge join "
        "{first} with {second} in either reference order?"
    )
    with pytest.raises(PromptBankError, match="undirected or directed support"):
        validate_s2_2_requests(requests)


@pytest.mark.parametrize("extra", ["Give every weight.", "Find the first matching layer."])
def test_existence_remains_boolean_without_unrequested_acquisition(requests, extra):
    requests["train"]["exists"]["any_direction"][0] += " " + extra
    with pytest.raises(PromptBankError, match="Boolean and unordered"):
        validate_s2_2_requests(requests)


def test_top_k_cannot_promise_an_exhaustive_support_list(requests):
    requests["train"]["top_k"]["any_direction"][0] += " Return every supporting layer."
    with pytest.raises(PromptBankError, match="cannot promise exhaustive"):
        validate_s2_2_requests(requests)


def test_tampered_request_file_is_rejected_before_rendering(requests, tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(requests))
    pin = {"path": str(path), "sha256": file_sha256(path)}
    path.write_text(path.read_text() + "\n")
    with pytest.raises(PromptBankError, match="does not match its pin"):
        load_s2_2_requests(pin)
