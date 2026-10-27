"""Verify the reviewed S1.6 thoughts and frozen boundaries of the v6 corpus."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_6_corpus import (
    LAYER_PROMPTS, CorpusBuildError, LayerFact, Scenario,
    build_corpus, build_row, filter_row_thinking,
)
from runtime.world_model.s1_overlay import (
    S1_V4_OVERLAY_PATH, S1_V5_OVERLAY_PATH, S1_V6_OVERLAY_PATH,
    S1OverlayError, apply_s1_overlay, overlay_pin,
)
from runtime.world_model.s1_prompt_banks import wrapper_prompts


BASELINE = Path("data/world_model_v4/corpora/s1_v5/s1_6")
BODY = (
    "The supplied tag {layer_id} follows the bulkPEN grammar. "
    "The prefix sets layer_family to bulk_pen. "
    "The next component gives organ {organ}, and the remaining colon-separated components give region_path {region_path}. "
    "Copy layer_id exactly and preserve each region component’s spelling and order, including underscores and hyphens. "
    "Return exactly layer_id, layer_family, source, network, organ, region_path, and cell_type_path. "
    "This grammar supplies no source, network, or cell-type path, so source, network, and cell_type_path remain null. "
    "These null values are expected for a valid bulkPEN tag. "
    "The supplied tag is sufficient to complete the parse. "
    "Return resolved with no blocked reason. "
)
REASONING = {
    "direct": BODY + "The user asked only for this layer parse. The requested fields are complete, so stop.",
    "step": BODY + "The user needs these parsed layer fields for a later workflow step. Completing the parse supplies that step’s input, so continue with the parsed result.",
}
SCPEN_BODY = (
    "The supplied tag {layer_id} follows the scPEN grammar. "
    "The prefix sets layer_family to scpen. "
    "The second colon-separated component gives organ {organ}. "
    "The third component alone gives region_path {region_path}; all remaining components give cell_type_path {cell_type_path}. "
    "Preserve the region and cell-type paths as separate ordered lists, including a second cell-type component when present. "
    "Copy layer_id exactly and preserve component spelling, underscores, and hyphens. "
    "The requested field is {requested_field}, whose value is {requested_value}. "
    "Return the complete parse with exactly layer_id, layer_family, source, network, organ, region_path, and cell_type_path. "
    "This grammar supplies no source or network, so both remain null. "
    "These null values are expected for a valid scPEN tag, including when source or network is the requested field. "
    "The supplied tag is sufficient to answer the request. "
    "Return resolved with no blocked reason. "
)
SCPEN_REASONING = {
    "direct": SCPEN_BODY + "The user asked only for this layer field. The requested value is established, so stop.",
    "step": SCPEN_BODY + "The user needs this layer field for a later workflow step. Providing the requested value supplies that step’s input, so continue with the parsed result.",
}
HUMANNET_BODY = (
    "The supplied tag {layer_id} follows the two-component HumanNet grammar. "
    "The prefix sets source to HumanNetV3. "
    "The second component gives network {network}. "
    "The family mapping assigns string_ppi and biogrid_ppi to ppi, coexpression to coexpression, and every other network name to other. "
    "For this network, set layer_family to {layer_family}. "
    "The family value other is a valid classification for a well-formed HumanNet tag. "
    "Copy layer_id and the network spelling exactly, including underscores. "
    "Return exactly layer_id, layer_family, source, network, organ, region_path, and cell_type_path. "
    "This grammar supplies no organ, region path, or cell-type path, so organ, region_path, and cell_type_path remain null. "
    "These null values are expected for a valid HumanNet tag. "
    "The supplied tag is sufficient to complete the parse. "
    "Return resolved with no blocked reason. "
)
HUMANNET_REASONING = {
    "direct": HUMANNET_BODY + "The user asked only for this layer parse. The requested fields are complete, so stop.",
    "step": HUMANNET_BODY + "The user needs these parsed layer fields for a later workflow step. Completing the parse supplies that step’s input, so continue with the parsed result.",
}


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build and audit the full component away from published artifacts."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_6_v6")
    manifest = build_corpus(
        source, root, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    return contract, payload, source, root, manifest


def test_only_reviewed_training_thoughts_change(revision):
    """Check exact approved wording, source bindings, and every preserved row field."""

    contract, payload, _, root, manifest = revision
    assert contract["curriculum"]["families"]["S1.6"]["train_bulk_pen_reasoning"] == REASONING
    assert manifest["train_bulk_pen_reasoning"] == REASONING
    assert contract["curriculum"]["families"]["S1.6"]["train_scpen_reasoning"] == SCPEN_REASONING
    assert manifest["train_scpen_reasoning"] == SCPEN_REASONING
    assert contract["curriculum"]["families"]["S1.6"]["train_humannet_reasoning"] == HUMANNET_REASONING
    assert manifest["train_humannet_reasoning"] == HUMANNET_REASONING
    prompts = wrapper_prompts(payload["prompt_banks"], LAYER_PROMPTS, bank_name="layer")
    changed, depths, tags = Counter(), Counter(), set()
    scpen_fields, scpen_depths, scpen_tags, scpen_regions = Counter(), Counter(), set(), set()
    humannet_networks, humannet_families = Counter(), Counter()
    unchanged = 0
    batch_counts = Counter()
    for old_line, new_line in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (root / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_line), json.loads(new_line)
        form = old["metadata"]["form"]
        if form in {4, 6}:
            assert old_line != new_line
            expected = deepcopy(old)
            expected["messages"][-1]["thinking"] = new["messages"][-1]["thinking"]
            assert new == expected
            thought = new["messages"][-1]["thinking"]
            result = new["messages"][-1]["content"]["result"]
            if form == 4:
                assert "Return exactly layers, filter, and matching_layer_ids." in thought
                assert f'Set matching_layer_ids to {json.dumps(result["matching_layer_ids"], ensure_ascii=False)}' in thought
                assert all(layer["layer_id"] in thought for layer in result["layers"])
            else:
                assert result["input"] in thought
                assert "Return exactly input and expected in result." in thought
                assert "so defer" in thought
            batch_counts[form] += 1
            continue
        if form not in {1, 2, 3}:
            assert old_line == new_line
            unchanged += 1
            continue
        user = old["messages"][1]["content"]
        framings = {
            framing for framing, templates in prompts["train"].items()
            if any(user.startswith(text.split("{request}")[0]) for text in templates)
        }
        assert len(framings) == 1
        framing = framings.pop()
        result = old["messages"][-1]["content"]["result"]
        expected = deepcopy(old)
        bindings = dict(layer_id=result["layer_id"], organ=result["organ"],
                        region_path=json.dumps(result["region_path"], ensure_ascii=False))
        if form == 2:
            field = re.search(r"report the exact (.+?) field for layer", user).group(1).replace(" ", "_")
            bindings.update(
                cell_type_path=json.dumps(result["cell_type_path"], ensure_ascii=False),
                requested_field=field,
                requested_value=json.dumps(result[field], ensure_ascii=False),
            )
            scpen_fields[field] += 1
            scpen_depths[len(result["cell_type_path"])] += 1
            scpen_tags.add(result["layer_id"])
            scpen_regions.add(tuple(result["region_path"]))
        elif form == 3:
            bindings = {key: result[key] for key in ("layer_id", "network", "layer_family")}
            humannet_networks[result["network"]] += 1
            humannet_families[result["layer_family"]] += 1
        reasoning = {1: REASONING, 2: SCPEN_REASONING, 3: HUMANNET_REASONING}[form]
        expected["messages"][-1]["thinking"] = reasoning[framing].format(**bindings)
        assert old_line != new_line
        assert new == expected
        assert len(new["messages"]) == 3
        assert not any("tool_call" in m or m["role"] == "tool" for m in new["messages"])
        answer = new["messages"][-1]["content"]
        assert answer["status"] == "resolved"
        assert answer["blocked_reason"] is None and answer["provenance"] is None
        assert answer["continuation_decision"] == ("stop" if framing == "direct" else "continue")
        assert set(result) == {
            "layer_id", "layer_family", "source", "network", "organ", "region_path", "cell_type_path",
        }
        if form in {1, 2}:
            assert result["organ"] == "brain"
            assert result["source"] is None and result["network"] is None
        parts = result["layer_id"].split(":")
        if form == 1:
            assert result["layer_family"] == "bulk_pen" and result["cell_type_path"] is None
            assert result["region_path"] == parts[2:]
            depths[len(result["region_path"])] += 1
            tags.add(result["layer_id"])
        elif form == 2:
            assert result["layer_family"] == "scpen"
            assert result["region_path"] == [parts[2]]
            assert result["cell_type_path"] == parts[3:]
        else:
            assert len(parts) == 2
            assert result["source"] == parts[0] == "HumanNetV3"
            assert result["network"] == parts[1]
            assert all(result[key] is None for key in ("organ", "region_path", "cell_type_path"))
            assert result["layer_family"] == {
                "string_ppi": "ppi", "biogrid_ppi": "ppi", "coexpression": "coexpression",
            }.get(result["network"], "other")
        changed[form, framing] += 1
    assert changed == {(1, "direct"): 625, (1, "step"): 625, (2, "direct"): 625, (2, "step"): 625,
                       (3, "direct"): 313, (3, "step"): 312}
    assert depths == {1: 754, 2: 496} and len(tags) == 10
    assert scpen_fields == {"layer_family": 209, "source": 209, "network": 208,
                            "organ": 208, "region_path": 208, "cell_type_path": 208}
    assert scpen_depths == {1: 1130, 2: 120}
    assert len(scpen_tags) == 209 and len(scpen_regions) == 12
    assert humannet_networks == {"string_ppi": 121, "coexpression": 128, "gene_neighbor": 120,
                                "cocitation": 128, "gene_interaction": 128}
    assert humannet_families == {"ppi": 121, "coexpression": 128, "other": 376}
    assert batch_counts == {4: 750, 6: 500}
    assert unchanged == 625
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()
        assert manifest["row_counts"][split] == 625
    assert manifest["evidence"] == json.loads((BASELINE / "manifest.json").read_text())["evidence"]
    assert manifest["train_final_reasoning"] == contract["curriculum"]["families"]["S1.6"]["train_final_reasoning"]


def test_repeated_build_is_byte_identical(revision, tmp_path):
    """Reproduce all splits and manifest from the same pinned inputs."""

    contract, payload, source, root, _ = revision
    build_corpus(
        source, tmp_path, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize(("overlay_path", "baseline"), [
    (None, Path("data/world_model_v3/corpora/s1/s1_6")),
    (S1_V4_OVERLAY_PATH, Path("data/world_model_v4/corpora/s1_v4/s1_6")),
    (S1_V5_OVERLAY_PATH, BASELINE),
])
def test_legacy_builds_reproduce_frozen_splits(overlay_path, baseline, tmp_path):
    """Preserve every published no-overlay, v4, and v5 row."""

    contract, payload = load_contract(), {}
    if overlay_path is not None:
        contract, payload = apply_s1_overlay(contract, overlay_path)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    build_corpus(
        source, tmp_path, contract=contract, prompt_banks=payload.get("prompt_banks"),
        overlay=overlay_pin(overlay_path, payload) if overlay_path is not None else None,
    )
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (baseline / f"{split}.jsonl").read_bytes()


@pytest.mark.parametrize("mutation", ["tag", "second_binding", "third_binding", "extra", "slot", "newline", "framing"])
@pytest.mark.parametrize(("kind", "label"), [("bulk_pen", "bulk-pen"), ("scpen", "scpen"), ("humannet", "humannet")])
def test_overlay_rejects_invalid_templates(tmp_path, mutation, kind, label):
    """Reject missing or substituted evidence bindings and malformed templates."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.6"][f"train_{kind}_reasoning"]
    if mutation == "framing":
        reasoning["workflow"] = reasoning.pop("step")
    else:
        text = reasoning["direct"]
        second_slot = "{network}" if kind == "humannet" else "{organ}"
        third_slot = "{layer_family}" if kind == "humannet" else "{region_path}"
        reasoning["direct"] = {
            "tag": text.replace("{layer_id}", "the tag"),
            "second_binding": text.replace(second_slot, "a value"),
            "third_binding": text.replace(third_slot, second_slot),
            "extra": text + " {layer_id}.",
            "slot": text.replace("{layer_id}", "{layer_id.real}"),
            "newline": text + "\nAnother sentence.",
        }[mutation]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match=f"{label} reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("field", ["cell_type_path", "requested_field", "requested_value"])
@pytest.mark.parametrize("replacement", ["a value", "{region_path}"])
def test_scpen_requires_each_field_binding(tmp_path, field, replacement):
    """Reject missing or confused cell-path, requested-field, and typed-value slots."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.6"]["train_scpen_reasoning"]
    reasoning["direct"] = reasoning["direct"].replace("{" + field + "}", replacement)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="scpen reasoning"):
        apply_s1_overlay(load_contract(), path)


def _scenario(*, split="train", form=1, framing="direct"):
    """Supply a native tag with distinct fields for the reviewed-form filter checks."""

    fact = LayerFact(
        fact_id="bulk-pen-test", layer_id="bulkPEN:brain:basal_ganglia:caudate",
        layer_family="bulk_pen", source=None, network=None, organ="brain",
        region_path=("basal_ganglia", "caudate"), cell_type_path=None,
        context_key="bulk-pen-test-context",
    )
    if form == 2:
        fact = LayerFact(
            fact_id="scpen-test", layer_id="scPEN:brain:amygdaloid-complex:deep_layer:near_projecting",
            layer_family="scpen", source=None, network=None, organ="brain",
            region_path=("amygdaloid-complex",), cell_type_path=("deep_layer", "near_projecting"),
            context_key="scpen-test-context",
        )
    elif form == 3:
        fact = LayerFact(
            fact_id="humannet-test", layer_id="HumanNetV3:gene_neighbor",
            layer_family="other", source="HumanNetV3", network="gene_neighbor",
            organ=None, region_path=None, cell_type_path=None,
            context_key="humannet-test-context",
        )
    return Scenario(split, form, fact, framing, 0, 0, train_reasoning=split == "train",
                    requested_field="network" if form == 2 else None)


@pytest.mark.parametrize(("split", "form"), [("validation", 1), ("test", 1), ("validation", 2), ("test", 2), ("validation", 3), ("test", 3), ("train", 7)])
def test_templates_cannot_change_heldout_or_unreviewed_forms(split, form):
    """Even supplied templates must leave other scenarios on their legacy path."""

    scenario = _scenario(split=split, form=form)
    original, _ = build_row(scenario, contract=load_contract())
    reasoning = dict(bulk_pen_reasoning=REASONING, scpen_reasoning=SCPEN_REASONING,
                     humannet_reasoning=HUMANNET_REASONING)
    revised, _ = build_row(scenario, contract=load_contract(), **reasoning)
    assert revised == original
    assert filter_row_thinking(revised, scenario, **reasoning) == 1


@pytest.mark.parametrize("framing", ["direct", "step"])
@pytest.mark.parametrize("mutation", ["thought", "status", "decision"])
@pytest.mark.parametrize("form", [1, 2, 3])
def test_thinking_filter_keeps_binding_and_decision_checks(framing, mutation, form):
    """Reject altered source-derived thoughts and conflicting final outcomes."""

    scenario = _scenario(framing=framing, form=form)
    reasoning = dict(bulk_pen_reasoning=REASONING, scpen_reasoning=SCPEN_REASONING,
                     humannet_reasoning=HUMANNET_REASONING)
    row, _ = build_row(scenario, contract=load_contract(), **reasoning)
    assert filter_row_thinking(row, scenario, **reasoning) == 1
    if mutation == "thought":
        value = {1: "caudate", 2: "near_projecting", 3: "gene_neighbor"}[form]
        row["messages"][-1]["thinking"] = row["messages"][-1]["thinking"].replace(value, "putamen")
    elif mutation == "status":
        row["messages"][-1]["content"]["status"] = "invalid_input"
    else:
        row["messages"][-1]["content"]["continuation_decision"] = "continue" if framing == "direct" else "stop"
    with pytest.raises(CorpusBuildError):
        filter_row_thinking(row, scenario, **reasoning)


@pytest.mark.parametrize(("claim", "error"), [
    ("The tag bulkPEN:brain:unseen_region is available.", "invents the tag"),
    ("The tool result confirms this.", "unavailable data"),
    ("Preserve provenance.", "unavailable data"),
    ("The graph edge is present.", "graph claim"),
    ("Return the answer schema.", "answer schema"),
])
@pytest.mark.parametrize("form", [1, 2, 3])
def test_templates_do_not_bypass_visibility_or_claim_filters(claim, error, form):
    """A template match alone must not authorize unsupported evidence or claims."""

    scenario = _scenario(form=form)
    templates = {1: REASONING, 2: SCPEN_REASONING, 3: HUMANNET_REASONING}[form]
    reasoning = {key: value + " " + claim for key, value in templates.items()}
    key = {1: "bulk_pen_reasoning", 2: "scpen_reasoning", 3: "humannet_reasoning"}[form]
    kwargs = {key: reasoning}
    row, _ = build_row(scenario, contract=load_contract(), **kwargs)
    with pytest.raises(CorpusBuildError, match=error):
        filter_row_thinking(row, scenario, **kwargs)


def _batch_scenario(form, framing="direct", split="train"):
    """Make visible mixed grammars or a malformed native input for the new batch."""

    bulk, single, human = (_scenario(form=i).fact for i in (1, 2, 3))
    if form == 4:
        return Scenario(split, form, bulk, framing, 0, 0,
                        companions=(single, human, bulk), requested_field="region_path",
                        filter_value="caudate", filter_outcome="match", train_reasoning=True)
    return Scenario(split, form, single, framing, 0, 0, train_reasoning=True,
                    malformed_subtype="missing_path", malformed_input="scPEN:brain:amygdala")


@pytest.mark.parametrize("form", [4, 6])
@pytest.mark.parametrize("framing", ["direct", "step"])
def test_batch_reasoning_derives_visible_grammar_and_preserves_scope(form, framing):
    """Distinguish region/cell paths and reject missing components in either frame."""

    contract, _ = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    templates = contract["curriculum"]["families"]["S1.6"]["train_final_reasoning"]
    scenario = _batch_scenario(form, framing)
    row, _ = build_row(scenario, contract=contract, final_reasoning=templates)
    assert filter_row_thinking(row, scenario, final_reasoning=templates) == 1
    thought = row["messages"][-1]["thinking"]
    result = row["messages"][-1]["content"]["result"]
    if form == 4:
        assert result["matching_layer_ids"] == [scenario.fact.layer_id]
        assert 'region_path ["amygdaloid-complex"]' in thought
        assert 'cell_type_path ["deep_layer", "near_projecting"]' in thought
        assert 'exact components ["basal_ganglia", "caudate"]' in thought
        assert "layer_family \"other\"" in thought
    else:
        assert "requires 4 to 5 nonempty" in thought
        assert "has 3 components, outside" in thought
        assert row["messages"][-1]["content"]["continuation_decision"] == "defer"
    for heldout in ("validation", "test"):
        scenario = replace(scenario, split=heldout)
        legacy, _ = build_row(scenario, contract=contract)
        supplied, _ = build_row(scenario, contract=contract, final_reasoning=templates)
        assert supplied == legacy


@pytest.mark.parametrize("form", [4, 6])
@pytest.mark.parametrize(("claim", "error"), [
    ('The tag bulkPEN:brain:unseen_region is present.', 'invents the tag'),
    ('The tool result confirms this.', 'unavailable data'),
    ('The graph edge is present.', 'graph claim'),
])
def test_batch_templates_retain_visibility_and_claim_checks(form, claim, error):
    """Matching a supplied template cannot permit hidden identifiers or evidence."""

    contract, _ = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    templates = deepcopy(contract["curriculum"]["families"]["S1.6"]["train_final_reasoning"])
    templates[form]["direct"] += " " + claim
    scenario = _batch_scenario(form)
    row, _ = build_row(scenario, contract=contract, final_reasoning=templates)
    with pytest.raises(CorpusBuildError, match=error):
        filter_row_thinking(row, scenario, final_reasoning=templates)


@pytest.mark.parametrize("form", [4, 6])
@pytest.mark.parametrize("mutation", ["binding", "framing", "extra_form"])
def test_batch_overlay_rejects_invalid_templates(tmp_path, form, mutation):
    """Keep the exact binding and reviewed-form boundary enforced in the overlay."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    templates = payload["family_overrides"]["S1.6"]["train_final_reasoning"]
    if mutation == "binding":
        slot = "{requested_field}" if form == 4 else "{input}"
        templates[form]["direct"] = templates[form]["direct"].replace(slot, "a value")
    elif mutation == "framing":
        templates[form]["workflow"] = templates[form].pop("step")
    else:
        templates[5] = templates[form]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="final reasoning"):
        apply_s1_overlay(load_contract(), path)
