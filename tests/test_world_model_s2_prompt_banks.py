"""Check S2 framing diversity, wording isolation, pinning, and S1 compatibility."""

from copy import deepcopy
import json

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import (
    LATER_STEP_CUES,
    PromptBankError,
    load_s2_prompt_banks,
    load_s2_edge_requests,
    prompt_bank_manifest,
    render_s2_prompt,
    render_s2_edge_request,
    validate_s2_edge_requests,
    validate_s2_rendered_prompts,
    validate_s2_prompt_banks,
    validate_wrapper_bank,
    S2_WRAPPER_SPEC,
)
from runtime.world_model.sources import file_sha256


BANK_PATH = REPOSITORY_ROOT / "runtime/world_model/s2_v1_prompt_banks.json"
EDGE_PATH = REPOSITORY_ROOT / "runtime/world_model/s2_v1_edge_requests.json"


@pytest.fixture
def bank():
    """Load the frozen wording without hiding mutations made by each test."""

    return json.loads(BANK_PATH.read_text())


@pytest.fixture
def edge_requests():
    """Read the independently frozen core wording for binding and leakage tests."""

    return json.loads(EDGE_PATH.read_text())


def test_frozen_s2_bank_is_pinned_diverse_and_separated(bank):
    pin = {"path": str(BANK_PATH), "sha256": file_sha256(BANK_PATH)}
    assert load_s2_prompt_banks(pin) == bank
    audit = validate_s2_prompt_banks(bank)
    assert audit["s1_held_out_wrappers_checked"] >= 100
    for split, count in (("train", 32), ("validation", 8), ("test", 8)):
        assert audit["splits"][split]["wrapper_counts"] == {
            "direct": count, "step": count,
        }
        assert audit["splits"][split]["maximum_held_out_jaccard"] < 0.6


def test_tampered_prompt_bank_fails_before_use(tmp_path, bank):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    pin = {"path": str(path), "sha256": file_sha256(path)}
    path.write_text(path.read_text() + "\n")
    with pytest.raises(PromptBankError, match="does not match its pin"):
        load_s2_prompt_banks(pin)


def test_s2_training_minimum_does_not_shrink_to_held_out_size(bank):
    for framing in ("direct", "step"):
        bank["train"][framing] = bank["train"][framing][:8]
    with pytest.raises(PromptBankError, match="at least 32"):
        validate_s2_prompt_banks(bank)


def test_cross_split_copy_is_rejected(bank):
    for framing in ("direct", "step"):
        bank["validation"][framing][0] = bank["train"][framing][0]
    with pytest.raises(PromptBankError, match="held-out wrapper"):
        validate_s2_prompt_banks(bank)


def test_frozen_s1_held_out_wording_cannot_enter_s2_training(bank):
    bank["train"]["direct"][0] = "Read this request and answer it.\n{request}"
    with pytest.raises(PromptBankError, match="held-out wrapper"):
        validate_s2_prompt_banks(bank)


@pytest.mark.parametrize("leak", ["edge", "missing", "registry", "degree", "cursor"])
def test_wrapper_cannot_name_an_operation_or_outcome(bank, leak):
    bank["train"]["direct"][0] = f"Address the {leak} request below.\n{{request}}"
    with pytest.raises(PromptBankError, match="names"):
        validate_s2_prompt_banks(bank)


def test_direct_framing_cannot_imply_later_work(bank):
    bank["train"]["direct"][0] = bank["train"]["step"][0]
    with pytest.raises(PromptBankError, match="implies a later step"):
        validate_s2_prompt_banks(bank)


def test_framing_pair_cannot_change_the_base_request_wording(bank):
    bank["train"]["step"][0], bank["train"]["step"][1] = (
        bank["train"]["step"][1], bank["train"]["step"][0],
    )
    with pytest.raises(PromptBankError, match="pairs must differ only"):
        validate_s2_prompt_banks(bank)


def test_renderer_preserves_the_selected_request_and_only_changes_framing(bank):
    request = 'Evaluate the supplied pair {"gene_ids":["A","B"]} in the specified layer.'
    for split, count in (("train", 32), ("validation", 8), ("test", 8)):
        for variant_index in range(count * 2):
            outputs = {}
            for framing in ("direct", "step"):
                text, index = render_s2_prompt(
                    bank, split=split, framing=framing,
                    variant_index=variant_index, request=request,
                )
                assert index == variant_index % count
                assert text.partition("\n")[2] == request
                assert text.count(request) == 1
                assert bool(LATER_STEP_CUES.search(text.partition("\n")[0])) == (
                    framing == "step"
                )
                outputs[framing] = text.partition("\n")[0]
            assert outputs["step"].startswith(outputs["direct"][:-1] + "; ")


@pytest.mark.parametrize("variant_index", [-1, True, 1.5])
def test_renderer_rejects_invalid_variant_index(bank, variant_index):
    with pytest.raises(PromptBankError, match="must be nonnegative"):
        render_s2_prompt(
            bank, split="train", framing="direct",
            variant_index=variant_index, request="Answer this question.",
        )


def test_existing_s1_bank_loader_and_training_minimum_are_unchanged(bank):
    path = REPOSITORY_ROOT / "runtime/world_model/s1_v5_prompt_banks.json"
    manifest = prompt_bank_manifest({"path": str(path), "sha256": file_sha256(path)})
    assert manifest["schema_version"] == "mentor-rl-world-model-s1-prompt-banks-v1"
    assert manifest["audit"]["identity"]["wrapper_counts"] == {
        "direct": 64, "step": 64,
    }
    with pytest.raises(PromptBankError, match="at least 32"):
        validate_wrapper_bank(
            deepcopy(bank["validation"]), spec=S2_WRAPPER_SPEC, held_out=(),
        )


def test_complete_rendered_questions_are_separated_across_every_split(bank, edge_requests):
    pin = {"path": str(EDGE_PATH), "sha256": file_sha256(EDGE_PATH)}
    assert load_s2_edge_requests(pin) == edge_requests
    audit = validate_s2_rendered_prompts(bank, edge_requests)
    assert audit["rendered_questions"] == {"train": 512, "validation": 128, "test": 128}
    assert audit["cross_split_pairs_checked"] == 147456
    assert audit["maximum_cross_split_jaccard"] < 0.6
    assert audit["shared_cross_split_5grams"] == 0


def test_core_leak_is_rejected_even_when_wrappers_and_representations_differ(bank, edge_requests):
    phrase = " Give its weight when recorded."
    edge_requests["train"]["directed"][0] += phrase
    edge_requests["validation"]["undirected"][0] += phrase
    validate_s2_prompt_banks(bank)
    validate_s2_edge_requests(edge_requests)
    with pytest.raises(PromptBankError, match="shared_5gram=True"):
        validate_s2_rendered_prompts(bank, edge_requests)


def test_every_core_binds_source_roles_and_scope_once(edge_requests):
    fields = {
        "multiplex_id": "multiplex_example",
        "layer_id": "layer_example",
        "first": "ENSG00000109787",
        "second": "ENSG00000001167",
    }
    for split in ("train", "validation", "test"):
        for directed in (True, False):
            representation = "directed" if directed else "undirected"
            for index in range(8):
                text = render_s2_edge_request(edge_requests, split, directed, index, **fields)
                assert text == edge_requests[split][representation][index % 4].format(**fields)
                assert all(text.count(value) == 1 for value in fields.values())
                assert text.index(fields["first"]) < text.index(fields["second"])


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "conversion"])
def test_core_template_cannot_drop_or_change_a_binding(edge_requests, change):
    template = edge_requests["train"]["directed"][0]
    if change == "missing":
        template = template.replace("{layer_id}", "some layer")
    elif change == "extra":
        template += " {unexpected}"
    elif change == "duplicate":
        template += " {first}"
    else:
        template = template.replace("{first}", "{first!r}")
    edge_requests["train"]["directed"][0] = template
    with pytest.raises(PromptBankError, match="four fields once"):
        validate_s2_edge_requests(edge_requests)


def test_directed_template_cannot_reverse_source_target_roles(edge_requests):
    edge_requests["train"]["directed"][0] = (
        "In {multiplex_id}/{layer_id}, does an arc have target {first} and source "
        "{second}? Include its weight when recorded."
    )
    with pytest.raises(PromptBankError, match="ordered orientation"):
        validate_s2_edge_requests(edge_requests)


def test_undirected_template_cannot_invent_endpoint_roles(edge_requests):
    edge_requests["train"]["undirected"][0] = (
        "In {multiplex_id}/{layer_id}, is an undirected edge recorded with source "
        "{first} and target {second}? Include its weight if recorded."
    )
    with pytest.raises(PromptBankError, match="unordered endpoints"):
        validate_s2_edge_requests(edge_requests)


@pytest.mark.parametrize("bad_fields", [
    {"multiplex_id": "M", "layer_id": "L", "first": "A"},
    {"multiplex_id": "M", "layer_id": "L", "first": "A", "second": "B", "extra": "C"},
    {"multiplex_id": "M", "layer_id": "L", "first": "", "second": "B"},
    {"multiplex_id": "M", "layer_id": "L", "first": 1, "second": "B"},
])
def test_core_renderer_requires_the_exact_four_field_values(edge_requests, bad_fields):
    with pytest.raises(PromptBankError, match="exactly four nonempty"):
        render_s2_edge_request(edge_requests, "train", True, 0, **bad_fields)
