"""Verify reviewed S1.3 wording and ambiguity reasoning against frozen rows."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest
import yaml

from runtime.world_model.contract import load_contract
from dataclasses import replace

from runtime.world_model.s1_3_corpus import (
    GeneFact, Scenario, _failure_pair_prompts_match, _initial_request, _membership_prompt_cue, _prompt, build_corpus,
    CorpusBuildError, filter_row_thinking, masked_visible_prompt,
)
from runtime.world_model.s1_overlay import (
    S1_V4_OVERLAY_PATH, S1_V5_OVERLAY_PATH, S1_V6_OVERLAY_PATH,
    S1OverlayError, apply_s1_overlay, overlay_pin,
)


BASELINE = Path("data/world_model_v4/corpora/s1_v5/s1_3")
REASONING = (
    "{symbol} matches {count} candidates; the user supplied no choice or multiplex to disambiguate them. "
    "Return ambiguous/defer with all candidates and details, registry evidence, and a blocked reason; "
    "omit gene_id because order alone cannot resolve ambiguity."
)
NEAR_MISS_REASONING = (
    "The lookup returns {count} candidates for {symbol}; unrelated harness entries supply no choice or multiplex. "
    "Keep only gene_symbol, candidate_gene_ids, and candidate_details in result; "
    "preserve lookup evidence and return ambiguous/defer with a blocked reason."
)
REASONING_BY_FORM = {1: REASONING, 4: NEAR_MISS_REASONING}
SINGLE_MEMBERSHIP_REASONING = {
    "after_lookup": "The lookup returns {count} candidates for {symbol}, and the {source} names {multiplex_id}. "
        "Check all candidate_gene_ids in that multiplex before selecting an ID.",
    "direct": "Only {gene_id} is present in {multiplex_id}, resolving this request; the user requested only this result, so stop. "
        "Return resolved with gene_id, gene_symbol, candidate_gene_ids, candidate_details, multiplex_id, "
        "present_gene_ids, absent_gene_ids, both evidence channels, and no blocked reason.",
    "step": "Only {gene_id} is present in {multiplex_id}, resolving this request; the user needs this result for a later workflow step, so continue. "
        "Return resolved with gene_id, gene_symbol, candidate_gene_ids, candidate_details, multiplex_id, "
        "present_gene_ids, absent_gene_ids, both evidence channels, and no blocked reason.",
}
MULTIPLE_MEMBERSHIP_REASONING = {
    "after_lookup": SINGLE_MEMBERSHIP_REASONING["after_lookup"],
    "final": "The membership check leaves {present_count} of {count} candidates for {symbol} in {multiplex_id}, so it does not identify one gene. "
        "Return ambiguous/defer with a blocked reason asking the user to choose one of the present IDs. "
        "Keep gene_symbol, candidate_gene_ids, candidate_details, multiplex_id, present_gene_ids, and absent_gene_ids exactly as returned, "
        "retaining absent_gene_ids even when empty and preserving both registry and graph evidence. "
        "Omit a singular gene_id, and wait for that choice even when a later workflow step needs the result.",
}
NO_MEMBERSHIP_REASONING = {
    "after_lookup": SINGLE_MEMBERSHIP_REASONING["after_lookup"],
    "final": "The membership check finds no present candidates for {symbol} in {multiplex_id}; all {count} registry candidates are absent from this multiplex. "
        "The lookup remains ambiguous, and choosing one of these IDs cannot supply a usable node for the requested multiplex. "
        "Return ambiguous/defer with a blocked reason explaining that another gene or multiplex is needed. "
        "Keep exactly gene_symbol, candidate_gene_ids, candidate_details, multiplex_id, present_gene_ids, and absent_gene_ids, "
        "including the empty present_gene_ids list and every returned absent ID. "
        "Preserve both registry and graph evidence and omit a singular gene_id. "
        "Both direct and workflow requests remain blocked, so defer until the request changes.",
}
USER_CHOICE_BODY = (
    "The reply selects {gene_id}, which appears in the earlier candidate_gene_ids list for {symbol}, so the user has resolved the choice. "
    "Build the result with exactly three fields: gene_id for the selected ID, gene_symbol for the earlier canonical symbol, "
    "and candidate_gene_ids for the entire earlier list in its original order. "
    "Keep the full list after selection; candidate_details and other fields do not belong in this resolved result. "
    "Preserve the earlier registry evidence and null graph provenance, and return resolved with no blocked reason. "
    "No new lookup or membership check is needed. "
)
USER_CHOICE_REASONING = {
    "direct": USER_CHOICE_BODY + "This completes the user's standalone gene-ID request, so stop.",
    "step": USER_CHOICE_BODY + "The user's later workflow step needs this selected mapping, so continue.",
}
WRAPPER_FORMS = {1, 4, 5, 6, 7}
REVIEWED_FORMS = WRAPPER_FORMS | {8, 10}


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build the component with all normal semantic and wording audits enabled."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_3_v6")
    manifest = build_corpus(
        source, root, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    return contract, payload, source, root, manifest


def test_only_reviewed_wording_and_post_tool_thinking_change(revision):
    """Compare every field and raw unreviewed row, retaining V5 evidence nonces."""

    root = revision[3]
    banks = {
        version: json.loads(Path(f"runtime/world_model/s1_{version}_prompt_banks.json").read_text())["identity_s1_3"]
        for version in ("v4", "v5")
    }
    counts = {form: Counter() for form in WRAPPER_FORMS}
    placements = Counter()
    coverage, other_rows = Counter(), 0
    for old_line, new_line in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (root / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_line), json.loads(new_line)
        form = old["metadata"]["form"]
        if form in {8, 10}:
            continue  # These deltas have separate complete-row checks.
        if form not in REVIEWED_FORMS:
            assert old_line == new_line
            other_rows += 1
            continue
        harness = any(m["role"] == "harness" for m in old["messages"])
        roles = ["system"] + (["harness"] if harness else []) + ["user", "assistant", "tool"]
        roles += ["assistant", "tool", "assistant"] if form in {5, 6, 7} else ["assistant"]
        assert [m["role"] for m in new["messages"]] == roles
        user_index = roles.index("user")
        call = old["messages"][user_index + 1]["tool_call"]
        assert call["name"] == "lookup_human_gene_symbol"
        request = f"give the human Ensembl gene ID for {call['arguments']['gene_symbol']}"
        if form in {5, 6, 7} and not harness:
            request += " in multiplex full_brain_multiplex_v1"
        indices = [
            (framing, i) for framing in ("direct", "step")
            for i, wrapper in enumerate(banks["v5"][framing])
            if wrapper.format(request=request) == old["messages"][user_index]["content"]
        ]
        assert len(indices) == 1
        framing, index = indices[0]
        coverage[form, framing, index] += 1
        output = old["messages"][user_index + 2]["content"]
        expected = deepcopy(old)
        expected["messages"][user_index]["content"] = banks["v4"][framing][index].format(request=request)
        if form in {5, 6, 7}:
            member = old["messages"][-2]["content"]
            reasoning = {5: SINGLE_MEMBERSHIP_REASONING, 6: MULTIPLE_MEMBERSHIP_REASONING, 7: NO_MEMBERSHIP_REASONING}[form]
            expected["messages"][user_index + 3]["thinking"] = reasoning["after_lookup"].format(
                symbol=output["gene_symbol"], count=len(output["candidate_gene_ids"]),
                source="harness" if harness else "task", multiplex_id=member["multiplex_id"],
            )
            expected["messages"][-1]["thinking"] = reasoning[framing if form == 5 else "final"].format(
                gene_id=member["present_gene_ids"][0] if form == 5 else None, multiplex_id=member["multiplex_id"],
                symbol=output["gene_symbol"], count=len(output["candidate_gene_ids"]),
                present_count=len(member["present_gene_ids"]),
            )
            counts[form]["intermediate_thoughts"] += old["messages"][user_index + 3]["thinking"] != expected["messages"][user_index + 3]["thinking"]
            placements[form, framing, "harness" if harness else "task"] += 1
        else:
            expected["messages"][-1]["thinking"] = REASONING_BY_FORM[form].format(
                symbol=output["gene_symbol"], count=len(output["candidate_gene_ids"]),
            )
        assert new == expected, old["row_id"]
        assert new_line == json.dumps(expected, ensure_ascii=False, separators=(",", ":")).encode()
        counts[form][framing] += 1
        counts[form][f"{framing}_users"] += old["messages"][user_index] != new["messages"][user_index]
        counts[form]["thoughts"] += old["messages"][-1]["thinking"] != new["messages"][-1]["thinking"]
        final = new["messages"][-1]["content"]
        if form in {5, 6, 7}:
            assert final["status"] == ("resolved" if form == 5 else "ambiguous")
            assert final["continuation_decision"] == (("continue" if framing == "step" else "stop") if form == 5 else "defer")
            assert final["result"] == {
                **({"gene_id": member["present_gene_ids"][0]} if form == 5 else {}),
                **{key: output[key] for key in ("gene_symbol", "candidate_gene_ids", "candidate_details")},
                **{key: member[key] for key in ("multiplex_id", "present_gene_ids", "absent_gene_ids")},
            }
            assert final["provenance"] == {"registry": output["provenance"]["registry"], "graph": member["provenance"]["graph"]}
            if form == 5:
                assert final["blocked_reason"] is None
            else:
                assert final["blocked_reason"]
                if form == 6:
                    assert len(member["present_gene_ids"]) >= 2
                else:
                    assert member["present_gene_ids"] == []
                    assert member["absent_gene_ids"] == output["candidate_gene_ids"]
        else:
            assert final["status"] == "ambiguous" and final["continuation_decision"] == "defer"
            assert final["result"] == {key: output[key] for key in ("gene_symbol", "candidate_gene_ids", "candidate_details")}
            assert final["provenance"] == output["provenance"]
            assert final["provenance"]["registry"] and final["provenance"]["graph"] is None
            assert final["blocked_reason"]
    assert other_rows == 2700
    assert counts == {
        1: {"direct": 1000, "direct_users": 155, "thoughts": 1000},
        4: {"direct": 500, "step": 500, "direct_users": 76, "step_users": 500, "thoughts": 1000},
        5: {"direct": 1035, "step": 1035, "direct_users": 160, "step_users": 1035, "thoughts": 2070, "intermediate_thoughts": 2070},
        6: {"direct": 109, "step": 109, "direct_users": 12, "step_users": 109, "thoughts": 218, "intermediate_thoughts": 218},
        7: {"direct": 206, "step": 206, "direct_users": 30, "step_users": 206, "thoughts": 412, "intermediate_thoughts": 412},
    }
    assert placements == {
        (5, "direct", "harness"): 518, (5, "direct", "task"): 517, (5, "step", "harness"): 517, (5, "step", "task"): 518,
        (6, "direct", "harness"): 55, (6, "direct", "task"): 54, (6, "step", "harness"): 54, (6, "step", "task"): 55,
        (7, "direct", "harness"): 103, (7, "direct", "task"): 103, (7, "step", "harness"): 103, (7, "step", "task"): 103,
    }
    for form, framing in ((1, "direct"), (4, "direct"), (4, "step"), (5, "direct"), (5, "step"), (6, "direct"), (6, "step"), (7, "direct"), (7, "step")):
        uses = [n for (f, kind, _), n in coverage.items() if (f, kind) == (form, framing)]
        assert len(uses) == 32
        assert set(uses) == {1: {31, 32}, 4: {15, 16}, 5: {32, 33, 34}, 6: {2, 3, 4}, 7: {6, 8}}[form]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_entire_heldout_split_is_byte_identical(revision, split):
    """Freeze all 1,125 held-out rows per split, including all reviewed forms."""

    path = revision[3] / f"{split}.jsonl"
    assert path.read_bytes() == (BASELINE / path.name).read_bytes()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1125
    for form in REVIEWED_FORMS:
        assert sum(row["metadata"]["form"] == form for row in rows) == {1: 125, 4: 125, 5: 259, 6: 27, 7: 52, 8: 125, 10: 37}[form]


def test_user_choice_changes_only_final_training_thinking(revision):
    """Bind the explicit reply and preserve both user turns, evidence, and answers."""

    counts = Counter()
    for old_line, new_line in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (revision[3] / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_line), json.loads(new_line)
        if old["metadata"]["form"] != 8:
            continue
        assert [m["role"] for m in new["messages"]] == ["system", "user", "assistant", "user", "assistant"]
        earlier, final = new["messages"][2]["content"], new["messages"][-1]["content"]
        selected, = re.findall(r"ENSG\d{11}", new["messages"][3]["content"])
        assert selected in earlier["result"]["candidate_gene_ids"]
        assert final["result"] == {
            "gene_id": selected,
            "gene_symbol": earlier["result"]["gene_symbol"],
            "candidate_gene_ids": earlier["result"]["candidate_gene_ids"],
        }
        assert final["status"] == "resolved" and final["blocked_reason"] is None
        assert final["provenance"] == earlier["provenance"]
        assert final["provenance"]["registry"] and final["provenance"]["graph"] is None
        assert final["continuation_decision"] in {"stop", "continue"}
        framing = "step" if final["continuation_decision"] == "continue" else "direct"
        expected = deepcopy(old)
        expected["messages"][-1]["thinking"] = USER_CHOICE_REASONING[framing].format(
            gene_id=selected, symbol=earlier["result"]["gene_symbol"],
        )
        assert old["messages"][-1]["thinking"] != expected["messages"][-1]["thinking"]
        assert new == expected, old["row_id"]
        assert new_line == json.dumps(expected, ensure_ascii=False, separators=(",", ":")).encode()
        counts[framing] += 1
    assert counts == {"direct": 500, "step": 500}


def test_manifest_pins_scoped_wrappers_and_reviewed_reasoning(revision):
    """Pin reviewed wrappers and reasoning without changing allocations or evidence."""

    _, payload, _, _, manifest = revision
    pins = manifest["train_prompt_banks"]
    assert set(pins) == {str(form) for form in WRAPPER_FORMS | {10}}
    for form in WRAPPER_FORMS:
        assert pins[str(form)]["path"] == "runtime/world_model/s1_v4_prompt_banks.json"
        assert pins[str(form)]["sha256"] == payload["family_overrides"]["S1.3"]["train_prompt_banks"][form]["sha256"]
    assert manifest["train_final_reasoning"] == {str(form): text for form, text in REASONING_BY_FORM.items()}
    assert manifest["train_single_membership_reasoning"] == SINGLE_MEMBERSHIP_REASONING
    assert manifest["train_multiple_membership_reasoning"] == MULTIPLE_MEMBERSHIP_REASONING
    assert manifest["train_no_membership_reasoning"] == NO_MEMBERSHIP_REASONING
    assert manifest["train_user_choice_reasoning"] == USER_CHOICE_REASONING
    structure = manifest["prompt_structure_audit"]
    assert structure["membership_pretool_cue_counts"]["train"] == {"keeps_one": 128, "keeps_several": 128, "keeps_none": 128, "shared_cue_count": 128}
    assert structure["membership_pretool_raw_cue_counts"]["train"] == {'keeps_none': 128, 'keeps_one': 128, 'keeps_several': 128, 'shared_cue_count': 128}
    for split, count in (("train", 128), ("validation", 16), ("test", 16)):
        assert structure["successful_membership_pretool_raw_cue_counts"][split] == {
            "keeps_one": count, "keeps_several": count, "keeps_none": count, "shared_cue_count": count,
        }
    assert manifest["prompt_bank"]["sha256"] == payload["prompt_banks"]["sha256"]
    baseline = json.loads((BASELINE / "manifest.json").read_text())
    for key in ("form_counts", "framing_counts"):
        assert manifest[key] == baseline[key]


@pytest.mark.parametrize("framing", ["direct", "step"])
def test_user_choice_provenance_phrase_keeps_other_thinking_checks(revision, framing):
    """Restrict the phrase exception to the pinned final training thought."""

    row = next(
        row for row in map(json.loads, (revision[3] / "train.jsonl").open())
        if row["metadata"]["form"] == 8 and row["messages"][-1]["content"]["continuation_decision"] == ("continue" if framing == "step" else "stop")
    )
    earlier, final = row["messages"][2]["content"], row["messages"][-1]["content"]
    symbol = earlier["result"]["gene_symbol"]
    selected = final["result"]["gene_id"]
    fact = GeneFact("filter-check", "unit", symbol, tuple(earlier["result"]["candidate_gene_ids"]), earlier["result"]["candidate_details"], ())
    scenario = Scenario("train", 8, fact, framing, 0, 0, symbol, chosen_gene_id=selected)
    assert filter_row_thinking(row, scenario, user_choice_reasoning=USER_CHOICE_REASONING) == 1
    with pytest.raises(CorpusBuildError, match="schema/provenance"):
        filter_row_thinking(row, scenario)
    for changed in (replace(scenario, split="test"), replace(scenario, form=9)):
        with pytest.raises(CorpusBuildError, match="schema/provenance"):
            filter_row_thinking(row, changed, user_choice_reasoning=USER_CHOICE_REASONING)
    altered = deepcopy(row)
    altered["messages"][-1]["thinking"] += " Another sentence."
    with pytest.raises(CorpusBuildError, match="schema/provenance"):
        filter_row_thinking(altered, scenario, user_choice_reasoning=USER_CHOICE_REASONING)
    extra_schema = deepcopy(row)
    templates = {key: text + " Copy provenance." for key, text in USER_CHOICE_REASONING.items()}
    extra_schema["messages"][-1]["thinking"] = templates[framing].format(gene_id=selected, symbol=symbol)
    with pytest.raises(CorpusBuildError, match="schema/provenance"):
        filter_row_thinking(extra_schema, scenario, user_choice_reasoning=templates)
    hidden = deepcopy(row)
    hidden["messages"] = json.loads(json.dumps(hidden["messages"][:-1]).replace(selected, "ENSG00000000000")) + [hidden["messages"][-1]]
    with pytest.raises(CorpusBuildError, match="reveals"):
        filter_row_thinking(hidden, scenario, user_choice_reasoning=USER_CHOICE_REASONING)
    for field, value, error in (
        ("status", "ambiguous", "wrong status"),
        ("continuation_decision", "defer", "wrong decision"),
    ):
        inconsistent = deepcopy(row)
        inconsistent["messages"][-1]["content"][field] = value
        with pytest.raises(CorpusBuildError, match=error):
            filter_row_thinking(inconsistent, scenario, user_choice_reasoning=USER_CHOICE_REASONING)


def test_repeated_build_is_byte_identical(revision, tmp_path):
    """Reproduce all row bytes and pins from the same inputs."""

    contract, payload, source, root, _ = revision
    build_corpus(
        source, tmp_path, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize(("key", "form"), [
    (key, form)
    for key in ("train_prompt_banks", "train_final_reasoning")
    for form in (2, 3, 6, 7, 8, 9, 10, "1", "4", "5", "6", "7", True)
    if key != "train_prompt_banks" or form not in {6, 7, 10}
])
def test_overlay_rejects_unreviewed_or_untyped_forms(tmp_path, key, form):
    """Only explicitly reviewed integer forms have S1.3 overrides."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    overrides = payload["family_overrides"]["S1.3"]
    overrides[key] = {form: overrides[key][1]}
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="only S1.3.1"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("form", [1, 4, 5, 6, 7])
def test_overlay_rejects_wrong_bank_pin(tmp_path, form):
    """Never use an unverified wrapper bank."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.3"]["train_prompt_banks"][form]["sha256"] = "0" * 64
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="train prompt bank"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("form", [1, 4])
@pytest.mark.parametrize(("old", "new"), [
    ("{count}", "two"),
    ("{count}", "{count} {count}"),
    ("{count}", "{count.real}"),
    ("{count}", "{gene_id}"),
    ("{symbol}", "EXAMPLE"),
    ("{count}", "{count} {extra}"),
])
def test_overlay_rejects_unbound_reasoning(tmp_path, form, old, new):
    """Require both visible-value bindings without permitting invented slots."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.3"]["train_final_reasoning"][form] = REASONING_BY_FORM[form].replace(old, new)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="final reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize(("overlay_path", "baseline"), [
    (None, Path("data/world_model_v3/corpora/s1/s1_3")),
    (S1_V4_OVERLAY_PATH, Path("data/world_model_v4/corpora/s1_v4/s1_3")),
    (S1_V5_OVERLAY_PATH, BASELINE),
])
def test_legacy_builds_reproduce_frozen_split_bytes(overlay_path, baseline, tmp_path):
    """Keep the changed builder compatible with each published baseline."""

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


@pytest.mark.parametrize("phase", ["after_lookup", "direct", "step"])
@pytest.mark.parametrize("mutation", ["missing", "repeated", "unsupported", "newline"])
def test_single_membership_reasoning_requires_visible_bindings(tmp_path, phase, mutation):
    """Keep the candidate lookup and selected-member bindings distinct."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.3"]["train_single_membership_reasoning"]
    slot = "{count}" if phase == "after_lookup" else "{gene_id}"
    replacement = {"missing": "a value", "repeated": slot + " " + slot, "unsupported": "{unexpected}", "newline": slot + "\n"}[mutation]
    reasoning[phase] = reasoning[phase].replace(slot, replacement)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="membership reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("mutation", ["missing_phase", "extra_phase", "flat_form5"])
def test_single_membership_reasoning_requires_scoped_phases(tmp_path, mutation):
    """Form 5 needs intermediate reasoning and separate final framing templates."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    settings = payload["family_overrides"]["S1.3"]
    if mutation == "missing_phase":
        settings["train_single_membership_reasoning"].pop("step")
    elif mutation == "extra_phase":
        settings["train_single_membership_reasoning"]["other"] = SINGLE_MEMBERSHIP_REASONING["direct"]
    else:
        settings["train_final_reasoning"][5] = REASONING
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize(("key", "phase", "slot"), [
    (key, phase, slot) for key in ("train_multiple_membership_reasoning", "train_no_membership_reasoning")
    for phase, slots in {
        "after_lookup": ("symbol", "count", "source", "multiplex_id"),
        "final": ("symbol", "count", "present_count", "multiplex_id"),
    }.items() for slot in slots if key != "train_no_membership_reasoning" or slot != "present_count"
])
@pytest.mark.parametrize("mutation", ["missing", "repeated", "unsupported", "newline"])
def test_unresolved_membership_reasoning_requires_visible_bindings(tmp_path, key, phase, slot, mutation):
    """Bind every candidate count and context cue to visible tool evidence."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.3"][key]
    field = "{" + slot + "}"
    replacement = {"missing": "a value", "repeated": field + " " + field, "unsupported": "{gene_id}", "newline": field + "\n"}[mutation]
    reasoning[phase] = reasoning[phase].replace(field, replacement)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="membership reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize(("form", "key", "templates"), [
    (6, "train_multiple_membership_reasoning", MULTIPLE_MEMBERSHIP_REASONING),
    (7, "train_no_membership_reasoning", NO_MEMBERSHIP_REASONING),
])
@pytest.mark.parametrize("mutation", ["missing_phase", "extra_phase", "flat_form", "wrong_family"])
def test_unresolved_membership_reasoning_requires_scoped_phases(tmp_path, form, key, templates, mutation):
    """Unresolved outcomes share final reasoning across framings only in S1.3."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    settings = payload["family_overrides"]["S1.3"]
    if mutation == "missing_phase":
        settings[key].pop("final")
    elif mutation == "extra_phase":
        settings[key]["step"] = templates["final"]
    elif mutation == "flat_form":
        settings["train_final_reasoning"][form] = REASONING
    else:
        payload["family_overrides"]["S1.1"][key] = templates
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError):
        apply_s1_overlay(load_contract(), path)


def test_all_reviewed_reasoning_templates_allow_longer_explanations(tmp_path):
    """Remove word and sentence caps across the shared identity overlay paths."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    extension = " More explanation follows. " + "The available evidence supports this step " * 12 + ". This completes the explanation."

    def extend_templates(templates):
        """Append to text leaves while retaining per-form/framing mappings."""

        checked = 0
        for phase, text in templates.items():
            if isinstance(text, dict):
                checked += extend_templates(text)
            else:
                assert isinstance(text, str)
                templates[phase] = text + extension
                checked += 1
        return checked

    checked = 0
    for values in payload["family_overrides"].values():
        for key, templates in values.items():
            if "reasoning" in key and isinstance(templates, dict):
                checked += extend_templates(templates)
    assert checked > 20
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    _, loaded = apply_s1_overlay(load_contract(), path)
    assert loaded == payload


@pytest.mark.parametrize("phase", ["direct", "step"])
@pytest.mark.parametrize("slot", ["gene_id", "symbol"])
@pytest.mark.parametrize("mutation", ["missing", "repeated", "unsupported", "newline"])
def test_user_choice_reasoning_requires_visible_bindings(tmp_path, phase, slot, mutation):
    """Reject templates that cannot bind the chosen ID and earlier canonical symbol."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.3"]["train_user_choice_reasoning"]
    field = "{" + slot + "}"
    replacement = {"missing": "a value", "repeated": field + " " + field, "unsupported": "{unexpected}", "newline": field + "\n"}[mutation]
    reasoning[phase] = reasoning[phase].replace(field, replacement)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="user-choice reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("mutation", ["missing_phase", "extra_phase", "wrong_family"])
def test_user_choice_reasoning_requires_scoped_framings(tmp_path, mutation):
    """Require the two framing templates and keep this override inside S1.3."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.3"]["train_user_choice_reasoning"]
    if mutation == "missing_phase":
        reasoning.pop("step")
    elif mutation == "extra_phase":
        reasoning["final"] = reasoning["direct"]
    else:
        payload["family_overrides"]["S1.1"]["train_user_choice_reasoning"] = reasoning
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError):
        apply_s1_overlay(load_contract(), path)


@pytest.fixture(params=[5, 6, 7])
def wrapper_pair(request):
    """Build a minimal success/failure prompt pair with different pinned wrappers."""

    banks = {
        v: {split: json.loads(Path(f"runtime/world_model/s1_v{v}_prompt_banks.json").read_text())["identity_s1_3"] for split in ("train", "test")}
        for v in (4, 5)
    }
    ids = ("ENSG00000000001", "ENSG00000000002")
    fact = GeneFact("fact", "unit", "EXAMPLE", ids, {}, {5: ids[:1], 6: ids, 7: ()}[request.param])
    success = Scenario("train", request.param, fact, "step", 0, 0, "EXAMPLE", multiplex_in_harness=True)
    failure = replace(success, form=10, paired_success_id=success.scenario_id)
    def row(scenario, bank):
        return {"messages": [
            {"role": "system", "content": "System instruction."},
            {"role": "harness", "content": "Active multiplex: full_brain_multiplex_v1"},
            {"role": "user", "content": _prompt(scenario, _initial_request(scenario), prompts=bank)},
        ]}
    return failure, row(failure, banks[5]), success, row(success, banks[4]), banks


def test_failure_pair_allows_only_its_declared_wrapper_difference(wrapper_pair):
    """Keep the held-out and unconfigured pair comparisons exact."""

    failure, failure_row, success, success_row, banks = wrapper_pair
    assert _failure_pair_prompts_match(failure, failure_row, success, success_row, prompts=banks[5], train_form_prompts={success.form: banks[4]})
    assert not _failure_pair_prompts_match(failure, failure_row, success, success_row, prompts=banks[5], train_form_prompts=None)
    assert not _failure_pair_prompts_match(replace(failure, split="test"), failure_row, replace(success, split="test"), success_row, prompts=banks[5], train_form_prompts={success.form: banks[4]})


@pytest.mark.parametrize("mutation", ["failure_user", "success_user", "harness", "system", "input", "framing", "template", "placement", "fact"])
def test_failure_pair_rejects_undeclared_prompt_or_scenario_changes(wrapper_pair, mutation):
    """Wrapper normalization cannot hide altered context or case assignments."""

    failure, failure_row, success, success_row, banks = wrapper_pair
    if mutation == "failure_user":
        failure_row["messages"][-1]["content"] += " Changed."
    elif mutation == "success_user":
        success_row["messages"][-1]["content"] += " Changed."
    elif mutation in {"harness", "system"}:
        success_row["messages"][1 if mutation == "harness" else 0]["content"] += " Changed."
    else:
        updates = {"input": {"input_value": "other"}, "framing": {"framing": "direct"}, "template": {"template_index": 1}, "placement": {"multiplex_in_harness": False}, "fact": {"fact": replace(success.fact, fact_id="other")}}
        success = replace(success, **updates[mutation])
    assert not _failure_pair_prompts_match(failure, failure_row, success, success_row, prompts=banks[5], train_form_prompts={success.form: banks[4]})


def test_membership_cue_normalizes_only_verified_training_wrapper(wrapper_pair):
    """Preserve context and reject wording mutations before any cue normalization."""

    failure, failure_row, success, success_row, banks = wrapper_pair
    expected = masked_visible_prompt(failure_row, failure)
    assert _membership_prompt_cue(success_row, success, prompts=banks[5], train_form_prompts={success.form: banks[4]}) == expected
    assert _membership_prompt_cue(success_row, success, prompts=banks[5], train_form_prompts=None) != expected
    assert _membership_prompt_cue(success_row, replace(success, split="test"), prompts=banks[5], train_form_prompts={success.form: banks[4]}) != expected
    changed = deepcopy(success_row)
    changed["messages"][1]["content"] += " Unexpected context."
    assert _membership_prompt_cue(changed, success, prompts=banks[5], train_form_prompts={success.form: banks[4]}) != expected
    changed["messages"][-1]["content"] += " Unexpected request."
    with pytest.raises(CorpusBuildError, match="pinned wrapper"):
        _membership_prompt_cue(changed, success, prompts=banks[5], train_form_prompts={success.form: banks[4]})


def test_membership_retry_delta_and_allocation(revision):
    """Keep all 600 identities, first calls, and old nonces; append only 150 graph IDs."""
    from runtime.world_model.s1_membership_retry import audit_membership_retry_row

    banks = {version: json.loads(Path(f"runtime/world_model/s1_{version}_prompt_banks.json").read_text())["identity_s1_3"] for version in ("v4", "v5")}
    counts, changes, outcomes = Counter(), Counter(), Counter()
    old_rows = list(map(json.loads, (BASELINE / "train.jsonl").read_text().splitlines()))
    new_rows = list(map(json.loads, (revision[3] / "train.jsonl").read_text().splitlines()))
    old_manifest = json.loads((BASELINE / "manifest.json").read_text())
    calls = revision[4]["evidence"]["calls"]
    old_calls = old_manifest["evidence"]["calls"]
    assert calls[:len(old_calls)] == old_calls
    assert len(calls) == len(old_calls) + 150
    assert len({call[0] for call in calls}) == len(calls)
    for old, new in zip(old_rows, new_rows, strict=True):
        if old["metadata"]["form"] != 10:
            continue
        old_messages, messages = old["messages"], new["messages"]
        user = next(i for i, m in enumerate(messages) if m["role"] == "user")
        harness = user == 2
        initial = old_messages[user + 1]
        request = f"give the human Ensembl gene ID for {initial['tool_call']['arguments']['gene_symbol']}"
        if not harness:
            request += " in multiplex full_brain_multiplex_v1"
        framing, index = next((f, i) for f in ("direct", "step") for i, w in enumerate(banks["v5"][f]) if w.format(request=request) == old_messages[user]["content"])
        assert messages[user]["content"] == banks["v4"][framing][index].format(request=request)
        changes[framing] += old_messages[user] != messages[user]
        assert new["row_id"] == old["row_id"]
        assert messages[:user] == old_messages[:user]
        assert messages[user + 1:user + 3] == old_messages[user + 1:user + 3]
        assert messages[user + 3]["tool_call"] == old_messages[user + 3]["tool_call"]
        tools = [m["content"] for m in messages if m["role"] == "tool"]
        case = "no_retry" if len(tools) == 2 else "exhausted" if tools[-1]["status"] == "error" else "recovered"
        audit_membership_retry_row(new, case, framing=framing)
        counts[case] += 1
        changes["after_lookup"] += messages[user + 3]["thinking"] != old_messages[user + 3]["thinking"]
        changes["final"] += messages[-1]["thinking"] != old_messages[-1]["thinking"]
        meta = dict(new["metadata"], blocked_reason_code=old["metadata"]["blocked_reason_code"])
        assert meta == old["metadata"] and new["tools"] == old["tools"]
        final = messages[-1]["content"]
        if case == "recovered":
            present = len(final["result"]["present_gene_ids"])
            outcomes["one" if present == 1 else "multiple" if present else "none"] += 1
        else:
            expected = deepcopy(old_messages[-1]["content"])
            expected["summary"] = expected["summary"].replace("Retry the check to continue.", "Graph access must be restored to continue.")
            assert final == expected
    assert counts == {"no_retry": 300, "recovered": 150, "exhausted": 150}
    assert changes == {"direct": 42, "step": 300, "after_lookup": 600, "final": 600}
    assert outcomes == {"one": 115, "multiple": 12, "none": 23}
    assert len(revision[4]["train_membership_retries"]["coverage"]) == 36
