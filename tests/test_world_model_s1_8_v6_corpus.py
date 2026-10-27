"""Verify S1.8's explicit result construction and frozen v6 boundaries."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_8_corpus import (
    CorpusBuildError, METRIC_CONVENTIONS, Scenario, build_corpus, build_row,
    filter_row_thinking,
)
from runtime.world_model.s1_overlay import (
    S1_V4_OVERLAY_PATH, S1_V5_OVERLAY_PATH, S1_V6_OVERLAY_PATH,
    S1OverlayError, apply_s1_overlay, overlay_pin,
)


BASELINE = Path("data/world_model_v4/corpora/s1_v5/s1_8")
TEMPLATE_SHA256 = "077418c4eb7ca9e55438679b499dcfed5e54c4f82374f64cdc825ad02a31cfa7"
RESULT_FIELDS = {
    "metric", "input", "lower_is_stronger", "meaning", "rule", "statement_true",
}


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build the approved overlay without touching published corpus files."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    root = tmp_path_factory.mktemp("s1_8_v6")
    manifest = build_corpus(
        root, contract=contract, prompt_banks=payload["prompt_banks"],
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
    )
    return contract, payload, root, manifest


def test_only_training_thoughts_change(revision):
    """Keep every prompt, target, source choice, row ID, and held-out byte fixed."""

    contract, _, root, manifest = revision
    templates = contract["curriculum"]["families"]["S1.8"]["train_final_reasoning"]
    serialized = json.dumps(templates, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(serialized.encode()).hexdigest() == TEMPLATE_SHA256
    assert manifest["train_final_reasoning"] == templates
    changed, kinds, decisions = Counter(), Counter(), Counter()
    for old_line, new_line in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (root / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_line), json.loads(new_line)
        form = old["metadata"]["form"]
        assert old["messages"][-1]["thinking"] != new["messages"][-1]["thinking"]
        expected = deepcopy(old)
        expected["messages"][-1]["thinking"] = new["messages"][-1]["thinking"]
        assert new == expected
        changed[form] += 1
        answer = new["messages"][-1]["content"]
        result = answer["result"]
        thought = new["messages"][-1]["thinking"]
        assert json.dumps(result["input"], ensure_ascii=False) in thought
        assert len(new["messages"]) == 3
        assert not any("tool_call" in m or m["role"] == "tool" for m in new["messages"])
        assert answer["provenance"] is None
        decisions[form, answer["continuation_decision"]] += 1
        if form <= 4:
            assert answer["status"] == "resolved" and answer["blocked_reason"] is None
            assert f'set metric to {json.dumps(result["metric"])}.' in thought
            for field in ("lower_is_stronger", "meaning", "rule"):
                assert f'{field} to {json.dumps(result[field])}' in thought
            if "candidate_metrics" in result:
                kinds["shared_alias"] += 1
                assert set(result) == RESULT_FIELDS | {"candidate_metrics"}
                assert result["metric"] is None
                assert "Their direction, meaning, and rule agree" in thought
                assert f'candidate_metrics with the ordered list {json.dumps(result["candidate_metrics"])}.' in thought
            else:
                kinds["canonical" if result["input"] == result["metric"] else "metric_variant"] += 1
                assert set(result) == RESULT_FIELDS
                assert {key: result[key] for key in METRIC_CONVENTIONS[result["metric"]]} == METRIC_CONVENTIONS[result["metric"]]
                assert "omit candidate_metrics" in thought
            if form == 1:
                assert result["statement_true"] is None
                assert "include statement_true as null" in thought
            else:
                claim = re.search(r'a (lower|higher) value for "[^"]+" means (stronger|weaker|greater|less)', new["messages"][1]["content"])
                assert claim
                actual_stronger = (claim[1] == "lower") == result["lower_is_stronger"]
                asserted_stronger = claim[2] in {"stronger", "greater"}
                assert result["statement_true"] == (actual_stronger == asserted_stronger)
                assert f'makes statement_true {json.dumps(result["statement_true"])}.' in thought
                assert "A false claim is still fully answered" in thought
            assert thought.endswith("so stop." if answer["continuation_decision"] == "stop" else "so continue.")
            if form == 4 and answer["continuation_decision"] == "stop":
                assert "enclosing request asks only for this convention check" in thought
                assert "does not add another requested action" in thought
        else:
            kinds["ambiguous" if form == 5 else "unknown"] += 1
            assert answer["continuation_decision"] == "defer"
            assert answer["blocked_reason"]
            assert all(result[field] is None for field in RESULT_FIELDS - {"input"})
            assert "explicitly null" in thought and "required fields" in thought
            assert thought.endswith("so defer.")
            if form == 5:
                assert answer["status"] == "ambiguous"
                assert set(result) == RESULT_FIELDS | {"candidate_metrics"}
                assert result["candidate_metrics"] == ["distance_percentile", "score_percentile"]
                assert "candidate conventions disagree" in thought
                assert "retaining that exact field name" in thought
            else:
                assert answer["status"] == "not_found"
                assert set(result) == RESULT_FIELDS
                assert "does not establish a convention here" in thought
                assert "no candidate_metrics field" in thought
    assert changed == {1: 1200, 2: 667, 3: 667, 4: 666, 5: 400, 6: 400}
    assert kinds == {"canonical": 1600, "metric_variant": 1280, "shared_alias": 320, "ambiguous": 400, "unknown": 400}
    assert decisions == {
        (1, "stop"): 600, (1, "continue"): 600,
        (2, "stop"): 334, (2, "continue"): 333,
        (3, "stop"): 333, (3, "continue"): 334,
        (4, "stop"): 333, (4, "continue"): 333,
        (5, "defer"): 400, (6, "defer"): 400,
    }
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()
    original = json.loads((BASELINE / "manifest.json").read_text())
    for field in ("row_counts", "form_counts", "scenario_assignment_sha256", "name_sources_sha256", "source_coverage", "evidence"):
        assert manifest[field] == original[field]


def test_repeated_build_is_byte_identical(revision, tmp_path):
    """The same pinned inputs reproduce all splits and the manifest."""

    contract, payload, root, _ = revision
    build_corpus(
        tmp_path, contract=contract, prompt_banks=payload["prompt_banks"],
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
    )
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize(("overlay_path", "baseline"), [
    (None, Path("data/world_model_v3/corpora/s1/s1_8")),
    (S1_V4_OVERLAY_PATH, Path("data/world_model_v4/corpora/s1_v4/s1_8")),
    (S1_V5_OVERLAY_PATH, BASELINE),
])
def test_legacy_builds_reproduce_frozen_splits(overlay_path, baseline, tmp_path):
    """Preserve the no-overlay, v4, and v5 training and evaluation bytes."""

    contract, payload = load_contract(), {}
    if overlay_path is not None:
        contract, payload = apply_s1_overlay(contract, overlay_path)
    build_corpus(tmp_path, contract=contract, prompt_banks=payload.get("prompt_banks"))
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (baseline / f"{split}.jsonl").read_bytes()


def _scenario(form=1, *, framing="direct", split="train"):
    """Use a supported convention and both kinds of deferred training names."""

    scenario = Scenario(
        split=split, form=form, metric="dot_similarity", input_name="dot_similarity",
        name_kind="canonical", lower_is_stronger=False, meaning="similarity",
        rule="higher_similarity_more_similar", statement_polarity=None if form == 1 else "lower_stronger",
        statement_true=None if form == 1 else False, candidate_metrics=(), framing=framing,
        template_index=0, occurrence=0, scenario_id="metric-reasoning-test", train_reasoning=split == "train",
    )
    if form in (5, 6):
        scenario = replace(
            scenario, metric=None, input_name="percentile" if form == 5 else "cosine distance",
            name_kind="ambiguous" if form == 5 else "unknown", lower_is_stronger=None,
            meaning=None, rule=None, statement_true=None,
            candidate_metrics=("distance_percentile", "score_percentile") if form == 5 else (),
        )
    return scenario


@pytest.mark.parametrize("form", range(1, 7))
@pytest.mark.parametrize("split", ["validation", "test"])
def test_supplied_templates_cannot_change_heldout(form, split, revision):
    """The template path is training-only even when callers supply it directly."""

    contract, _, _, _ = revision
    templates = contract["curriculum"]["families"]["S1.8"]["train_final_reasoning"]
    scenario = _scenario(form, split=split)
    old, _ = build_row(scenario, contract=contract)
    new, _ = build_row(scenario, contract=contract, final_reasoning=templates)
    assert old == new
    assert filter_row_thinking(new, scenario, final_reasoning=templates) == 1


@pytest.mark.parametrize("form", range(1, 7))
@pytest.mark.parametrize("mutation", ["thought", "result", "status", "decision", "call"])
def test_filter_preserves_evidence_and_outcome_checks(form, mutation, revision):
    """Reject altered bindings, missing schema fields, and unsupported actions."""

    contract, _, _, _ = revision
    templates = contract["curriculum"]["families"]["S1.8"]["train_final_reasoning"]
    scenario = _scenario(form)
    row, _ = build_row(scenario, contract=contract, final_reasoning=templates)
    assert filter_row_thinking(row, scenario, final_reasoning=templates) == 1
    final = row["messages"][-1]
    if mutation == "thought":
        final["thinking"] = final["thinking"].replace(scenario.input_name, "invented_metric")
    elif mutation == "result":
        del final["content"]["result"]["metric"]
    elif mutation == "status":
        final["content"]["status"] = "invalid_input"
    elif mutation == "decision":
        final["content"]["continuation_decision"] = "continue"
    else:
        final["tool_call"] = {"name": "lookup_human_gene_symbol", "arguments": {"gene_symbol": "TP53"}}
    with pytest.raises(CorpusBuildError):
        filter_row_thinking(row, scenario, final_reasoning=templates)


@pytest.mark.parametrize("claim", ["The tool result establishes this.", "A tool call is needed.", "Preserve provenance."])
def test_templates_do_not_bypass_visibility_filter(claim, revision):
    """A matching template cannot authorize unavailable tool evidence."""

    contract, _, _, _ = revision
    templates = deepcopy(contract["curriculum"]["families"]["S1.8"]["train_final_reasoning"])
    templates[1]["direct"] += " " + claim
    scenario = _scenario()
    row, _ = build_row(scenario, contract=contract, final_reasoning=templates)
    with pytest.raises(CorpusBuildError, match="runtime fields"):
        filter_row_thinking(row, scenario, final_reasoning=templates)


@pytest.mark.parametrize("form", range(1, 7))
@pytest.mark.parametrize("mutation", ["missing", "extra", "slot", "newline", "framing"])
def test_overlay_requires_exact_template_bindings(form, mutation, tmp_path):
    """Reject malformed templates and omitted or substituted supplied names."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    frames = payload["family_overrides"]["S1.8"]["train_final_reasoning"][form]
    if mutation == "framing":
        frames["workflow"] = frames.pop("step")
    else:
        text = frames["direct"]
        frames["direct"] = {
            "missing": text.replace("{input_name}", "the name"),
            "extra": text + " {input_name}.",
            "slot": text.replace("{input_name}", "{input_name.real}"),
            "newline": text + "\nAnother sentence.",
        }[mutation]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError):
        apply_s1_overlay(load_contract(), path)
