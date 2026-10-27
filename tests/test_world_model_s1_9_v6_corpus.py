"""Verify S1.9 v6 changes only the reviewed module parsing thoughts."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_9_corpus import (
    CorpusBuildError, Scenario, _prose_length_audit, build_corpus, build_row,
    filter_row_thinking,
)
from runtime.world_model.s1_overlay import (
    S1_V4_OVERLAY_PATH, S1_V5_OVERLAY_PATH, S1_V6_OVERLAY_PATH,
    S1OverlayError, apply_s1_overlay, overlay_pin,
)


BASELINE = Path("data/world_model_v4/corpora/s1_v5/s1_9")
REASONING = {1: {'direct': 'The user asks to decode a module identifier, so apply the module grammar to the '
               'supplied ID. A module ID is the expected input for this request. Parse {module_id} '
               'as four colon-separated components. The prefix rwr_loe sets module_family to '
               'rwr_loe. The second component sets multiplex_id to {multiplex_id}. The third '
               'component is {seed_component}; remove only seed_ to obtain seed_gene_id '
               '{seed_gene_id}. The fourth component sets threshold_recipe to {threshold_recipe}. '
               'Preserve the entire supplied identifier as module_id and copy every other '
               'component exactly. Return exactly module_id, module_family, multiplex_id, '
               'seed_gene_id, threshold_recipe, and existence_asserted. These are the fields '
               'encoded by this module grammar. This syntax task is complete from the identifier; '
               'no lookup or existence check is needed. Parsing establishes the encoded fields and '
               'does not establish module existence, so existence_asserted is false. The grammar '
               'matches, so status is resolved and there is no blocked reason. The answer uses '
               'null provenance because the fields come from the supplied identifier. The user '
               'asked only for the encoded module fields. Returning the complete parse finishes '
               'that request, so stop.',
     'step': 'The user asks to decode a module identifier, so apply the module grammar to the '
             'supplied ID. A module ID is the expected input for this request. Parse {module_id} '
             'as four colon-separated components. The prefix rwr_loe sets module_family to '
             'rwr_loe. The second component sets multiplex_id to {multiplex_id}. The third '
             'component is {seed_component}; remove only seed_ to obtain seed_gene_id '
             '{seed_gene_id}. The fourth component sets threshold_recipe to {threshold_recipe}. '
             'Preserve the entire supplied identifier as module_id and copy every other component '
             'exactly. Return exactly module_id, module_family, multiplex_id, seed_gene_id, '
             'threshold_recipe, and existence_asserted. These are the fields encoded by this '
             'module grammar. This syntax task is complete from the identifier; no lookup or '
             'existence check is needed. Parsing establishes the encoded fields and does not '
             'establish module existence, so existence_asserted is false. The grammar matches, so '
             'status is resolved and there is no blocked reason. The answer uses null provenance '
             'because the fields come from the supplied identifier. The user needs the encoded '
             'module fields for a later workflow step. Returning the complete parse supplies that '
             'step’s input, so continue with the parsed result.'},
 2: {'direct': 'The user asks to decode a module identifier, so apply the module grammar to the '
               'supplied ID. A module ID is the expected input for this request. Parse {module_id} '
               'as three colon-separated components. The prefix mentor_ev sets module_family to '
               'mentor_ev. The second component sets dendrogram_id to {dendrogram_id}. The third '
               'component sets clade_id to {clade_id}; preserve the clade_ prefix and all six '
               'digits, including leading zeros. Preserve the entire supplied identifier as '
               'module_id and keep the clade identifier as a string. Return exactly module_id, '
               'module_family, dendrogram_id, clade_id, and existence_asserted. These are the '
               'fields encoded by this module grammar. This syntax task is complete from the '
               'identifier; no lookup or existence check is needed. Parsing establishes the '
               'encoded fields and does not establish module existence, so existence_asserted is '
               'false. The grammar matches, so status is resolved and there is no blocked reason. '
               'The answer uses null provenance because the fields come from the supplied '
               'identifier. The user asked only for the encoded module fields. Returning the '
               'complete parse finishes that request, so stop.',
     'step': 'The user asks to decode a module identifier, so apply the module grammar to the '
             'supplied ID. A module ID is the expected input for this request. Parse {module_id} '
             'as three colon-separated components. The prefix mentor_ev sets module_family to '
             'mentor_ev. The second component sets dendrogram_id to {dendrogram_id}. The third '
             'component sets clade_id to {clade_id}; preserve the clade_ prefix and all six '
             'digits, including leading zeros. Preserve the entire supplied identifier as '
             'module_id and keep the clade identifier as a string. Return exactly module_id, '
             'module_family, dendrogram_id, clade_id, and existence_asserted. These are the fields '
             'encoded by this module grammar. This syntax task is complete from the identifier; no '
             'lookup or existence check is needed. Parsing establishes the encoded fields and does '
             'not establish module existence, so existence_asserted is false. The grammar matches, '
             'so status is resolved and there is no blocked reason. The answer uses null '
             'provenance because the fields come from the supplied identifier. The user needs the '
             'encoded module fields for a later workflow step. Returning the complete parse '
             'supplies that step’s input, so continue with the parsed result.'}}


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build the complete revision with its pinned overlay and semantic audits."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_9_v6")
    manifest = build_corpus(
        source, root, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    return contract, payload, source, root, manifest


def test_only_all_reviewed_thoughts_change(revision):
    """Compare every row field and exact approved source-bound thought with v5."""

    contract, _, _, root, manifest = revision
    assert contract["curriculum"]["families"]["S1.9"]["train_final_reasoning"] == REASONING
    assert manifest["train_final_reasoning"] == REASONING
    changes, decisions, requests = Counter(), Counter(), {}
    for old_line, new_line in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (root / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_line), json.loads(new_line)
        form = old["metadata"]["form"]
        answer = old["messages"][-1]["content"]
        result = answer["result"]
        framing = "direct" if answer["continuation_decision"] == "stop" else "step"
        bindings = dict(result)
        parts = result["module_id"].split(":")
        if form == 1:
            bindings["seed_component"] = parts[2]
            assert result["seed_gene_id"] == parts[2].removeprefix("seed_")
            assert result["multiplex_id"] == parts[1]
            assert result["threshold_recipe"] == parts[3]
        else:
            assert result["dendrogram_id"] == parts[1]
            assert result["clade_id"] == parts[2]
            assert len(parts[2].removeprefix("clade_")) == 6
            assert parts[2].startswith("clade_0")
        expected = deepcopy(old)
        expected["messages"][-1]["thinking"] = REASONING[form][framing].format(**bindings)
        assert new == expected and old_line != new_line
        assert answer["status"] == "resolved"
        assert answer["provenance"] is None and answer["blocked_reason"] is None
        assert result["existence_asserted"] is False
        request = old["messages"][1]["content"].splitlines()[-1].replace(
            result["module_id"], "{module_id}",
        )
        requests.setdefault((form, request), Counter())[framing] += 1
        changes[form] += 1
        decisions[(form, framing)] += 1
    assert changes == {1: 1000, 2: 1000}
    assert decisions == {(1, "direct"): 500, (1, "step"): 500,
                         (2, "direct"): 500, (2, "step"): 500}
    assert len(requests) == 16
    assert all(set(counts) == {"direct", "step"}
               and abs(counts["direct"] - counts["step"]) <= 1
               for counts in requests.values())
    prior = json.loads((BASELINE / "manifest.json").read_text())
    for key in ("scenario_assignment_sha256", "selected_source_sha256", "source_pool_coverage"):
        assert manifest[key] == prior[key]
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()


def test_build_is_deterministic(revision, tmp_path):
    """Repeated builds reproduce every split and the pinned manifest."""

    contract, payload, source, root, _ = revision
    build_corpus(source, tmp_path, contract=contract,
                 overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
                 prompt_banks=payload["prompt_banks"])
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize(("overlay_path", "baseline"), [
    (None, Path("data/world_model_v3/corpora/s1/s1_9")),
    (S1_V4_OVERLAY_PATH, Path("data/world_model_v4/corpora/s1_v4/s1_9")),
    (S1_V5_OVERLAY_PATH, BASELINE),
])
def test_legacy_builds_remain_byte_identical(revision, tmp_path, overlay_path, baseline):
    """The optional templates cannot alter any published no-overlay/v4/v5 row."""

    contract, payload = load_contract(), None
    if overlay_path is not None:
        contract, payload = apply_s1_overlay(contract, overlay_path)
    build_corpus(revision[2], tmp_path, contract=contract,
                 prompt_banks=payload["prompt_banks"] if payload else None)
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (baseline / f"{split}.jsonl").read_bytes()


def _scenario(form, *, split="train", framing="direct"):
    """Use literal compositional IDs to check prefixes, zeros, and source visibility."""

    module_id = (
        "rwr_loe:full_brain_multiplex_v1:seed_ENSG00000141510:geometric_elbow_v1"
        if form == 1 else "mentor_ev:full_brain_all_genes_v1:clade_000417"
    )
    return Scenario(split, form, module_id, "rwr_loe" if form == 1 else "mentor_ev",
                    "unit", framing, 0, 0, 0, "scenario", train_reasoning=split == "train")


@pytest.mark.parametrize("form", [1, 2])
@pytest.mark.parametrize("split", ["validation", "test"])
def test_templates_do_not_change_heldout_scenarios(form, split):
    """Even explicitly supplied templates must be ignored outside training."""

    scenario = _scenario(form, split=split)
    old, _ = build_row(scenario, contract=load_contract())
    new, _ = build_row(scenario, contract=load_contract(), final_reasoning=REASONING)
    assert new == old
    assert filter_row_thinking(new, scenario, final_reasoning=REASONING) == 1


@pytest.mark.parametrize("form", [1, 2])
@pytest.mark.parametrize("framing", ["direct", "step"])
@pytest.mark.parametrize("mutation", ["thinking", "input", "status", "decision", "existence", "provenance"])
def test_thinking_filter_preserves_source_and_answer_checks(form, framing, mutation):
    """Reject substituted evidence, altered thoughts, and incompatible output fields."""

    scenario = _scenario(form, framing=framing)
    row, _ = build_row(scenario, contract=load_contract(), final_reasoning=REASONING)
    assert filter_row_thinking(row, scenario, final_reasoning=REASONING) == 1
    answer = row["messages"][-1]["content"]
    if mutation == "thinking":
        row["messages"][-1]["thinking"] += " This module exists."
    elif mutation == "input":
        row["messages"][1]["content"] = row["messages"][1]["content"].replace(scenario.module_id, "another identifier")
    elif mutation == "status":
        answer["status"] = "invalid_input"
    elif mutation == "decision":
        answer["continuation_decision"] = "continue" if framing == "direct" else "stop"
    elif mutation == "existence":
        answer["result"]["existence_asserted"] = True
    else:
        answer["provenance"] = "local_match"
    with pytest.raises(CorpusBuildError):
        filter_row_thinking(row, scenario, final_reasoning=REASONING)


@pytest.mark.parametrize("form", [1, 2])
@pytest.mark.parametrize(("claim", "error"), [
    ("The gene is ENSG99999999999.", "invents the identifier"),
    ("The clade is clade_999999.", "invents the identifier"),
    ("The module is mentor_ev:full_brain_all_genes_v1:clade_999999.", "invents the identifier"),
    ("The tool result confirms this.", "runtime fields"),
    ("Preserve provenance.", "runtime fields"),
    ("This module exists.", "asserts module existence"),
    ("The existence_asserted is true.", "asserts module existence"),
])
def test_template_match_cannot_bypass_visibility_or_claim_filters(form, claim, error):
    """Exact template matching must still reject unsupported identifiers and claims."""

    scenario = _scenario(form)
    templates = deepcopy(REASONING)
    templates[form]["direct"] += " " + claim
    row, _ = build_row(scenario, contract=load_contract(), final_reasoning=templates)
    with pytest.raises(CorpusBuildError, match=error):
        filter_row_thinking(row, scenario, final_reasoning=templates)


@pytest.mark.parametrize("form", [1, 2])
@pytest.mark.parametrize("mutation", ["missing", "repeat", "unsupported", "newline"])
def test_overlay_rejects_invalid_template_bindings(tmp_path, form, mutation):
    """Each approved source slot must occur exactly once in one trimmed line."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    templates = payload["family_overrides"]["S1.9"]["train_final_reasoning"]
    original = templates[form]["direct"]
    templates[form]["direct"] = {
        "missing": original.replace("{module_id}", "the module"),
        "repeat": original + " Copy {module_id}.",
        "unsupported": original.replace("{module_id}", "{unsupported}"),
        "newline": original + "\nAnother sentence.",
    }[mutation]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError):
        apply_s1_overlay(load_contract(), path)


def test_reasoning_has_no_sentence_length_gate():
    """Longer reasoning is allowed while user and summary limits remain active."""

    row, _ = build_row(_scenario(1), contract=load_contract(), final_reasoning=REASONING)
    row["messages"][-1]["thinking"] += " " + "The supplied identifier is available " * 10 + "."
    audit = _prose_length_audit({split: [row] for split in ("train", "validation", "test")})
    assert audit["pass"]
    assert audit["splits"]["train"]["maximum_description_words"] > 25
    row["messages"][-1]["content"]["summary"] = "summary " * 26 + "."
    with pytest.raises(CorpusBuildError, match="prose length"):
        _prose_length_audit({split: [row] for split in ("train", "validation", "test")})
