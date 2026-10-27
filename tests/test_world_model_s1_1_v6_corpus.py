"""Check the reviewed S1.1 edits, bounded retries, and frozen corpus boundary."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_1_corpus import (
    IDENTITY_PROMPTS,
    CorpusBuildError,
    GeneFact,
    Scenario,
    audit_rows,
    build_corpus,
)
from runtime.world_model.s1_overlay import (
    S1_V4_OVERLAY_PATH,
    S1_V5_OVERLAY_PATH,
    S1_V6_OVERLAY_PATH,
    S1OverlayError,
    apply_s1_overlay,
    overlay_pin,
)


BASELINE = Path("data/world_model_v4/corpora/s1_v5/s1_1")
V4 = Path("data/world_model_v4/corpora/s1_v4/s1_1")
INSIDE_TASK_REASONING = (
    "The registry resolves {symbol} to {gene_id}, completing the requested lookup. "
    "Because the user needs this mapping for a later workflow step, return it with registry evidence and continue."
)
ABSENT_SYMBOL_REASONING = (
    "The registry reports not_found for {symbol}; this release provides no mapping. "
    "Preserve the queried symbol and registry evidence, then defer with a blocked reason."
)
MEMBERSHIP_REASONING = {
    "after_lookup": (
        "The lookup supplies {gene_id} for {symbol}, and the membership request remains unanswered. "
        "Check this ID in the requested multiplex."
    ),
    "direct": (
        "The membership result lists {gene_id} as present, so return resolved with present:true and both evidence channels. "
        "The request is complete, so stop."
    ),
    "step": (
        "The membership result lists {gene_id} as present, so return resolved with present:true and both evidence channels. "
        "The next workflow step needs this membership result, so continue."
    ),
}
ABSENT_MEMBERSHIP_REASONING = {
    "after_lookup": MEMBERSHIP_REASONING["after_lookup"],
    "direct": (
        "The membership result lists {gene_id} as absent, so return resolved with present:false, both evidence channels, and no blocked reason. "
        "The request is complete, so stop."
    ),
    "step": (
        "The membership result lists {gene_id} as absent, so return resolved with present:false, both evidence channels, and no blocked reason. "
        "The next workflow step needs this membership result, so continue."
    ),
}
UNKNOWN_MULTIPLEX_REASONING = {
    "after_lookup": MEMBERSHIP_REASONING["after_lookup"],
    "direct": (
        "The requested multiplex is unregistered, so the gene's membership is unknown; return not_found with present:null. "
        "Preserve both evidence channels and defer with a blocked reason until a registered multiplex is supplied."
    ),
    "step": (
        "The requested multiplex is unregistered, so the gene's membership is unknown; return not_found with present:null. "
        "Preserve both evidence channels and defer with a blocked reason until a registered multiplex is supplied."
    ),
}


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build the reviewed change from pinned sources in a temporary directory."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_1_v6")
    manifest = build_corpus(
        source, root, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    return contract, payload, source, root, manifest


def test_only_reviewed_training_messages_change(revision):
    """Apply only reviewed wording and thinking changes, preserving targets."""

    _, _, _, root, manifest = revision
    v4 = {
        row["row_id"]: row
        for row in map(json.loads, (V4 / "train.jsonl").read_text().splitlines())
    }
    changed, form_counts, retry_outcomes = Counter(), Counter(), Counter()
    inside_task_changes = Counter()
    membership_changes = {form: Counter() for form in (9, 10, 11)}
    membership_framing = {form: Counter() for form in (9, 10, 11)}
    v4_wrappers = json.loads(Path("runtime/world_model/s1_v4_prompt_banks.json").read_text())["identity"]
    for old_line, new_line in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (root / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_line), json.loads(new_line)
        expected = deepcopy(old)
        form = old["metadata"]["form"]
        form_counts[form] += 1
        if form in {1, 2, 9, 10, 11}:
            expected["messages"][1]["content"] = (
                v4[old["row_id"]]["messages"][1]["content"]
            )
            if form == 2:
                result = old["messages"][-1]["content"]["result"]
                expected["messages"][-1]["thinking"] = INSIDE_TASK_REASONING.format(
                    symbol=result["gene_symbol"], gene_id=result["gene_id"],
                )
                inside_task_changes["user"] += (
                    old["messages"][1] != new["messages"][1]
                )
                inside_task_changes["thinking"] += (
                    old["messages"][-1]["thinking"] != new["messages"][-1]["thinking"]
                )
        if form in {9, 10, 11}:
            present = form == 9
            reasoning = {
                9: MEMBERSHIP_REASONING, 10: ABSENT_MEMBERSHIP_REASONING,
                11: UNKNOWN_MULTIPLEX_REASONING,
            }[form]
            lookup = old["messages"][3]["content"]
            final = old["messages"][-1]["content"]
            framing = "direct" if final["continuation_decision"] == "stop" else "step"
            if form == 11:
                request = (
                    f"whether the human gene {old['messages'][2]['tool_call']['arguments']['gene_symbol']} is a node in "
                    f"{final['result']['multiplex_id']}"
                )
                matches = [
                    frame for frame, wrappers in v4_wrappers.items()
                    if v4[old["row_id"]]["messages"][1]["content"] in {
                        wrapper.format(request=request) for wrapper in wrappers
                    }
                ]
                assert len(matches) == 1
                framing = matches[0]
            expected["messages"][4]["thinking"] = reasoning["after_lookup"].format(
                gene_id=lookup["gene_id"], symbol=lookup["gene_symbol"],
            )
            expected["messages"][-1]["thinking"] = reasoning[framing].format(
                gene_id=lookup["gene_id"],
            )
            membership_framing[form][framing] += 1
            membership_changes[form][framing] += old["messages"][1] != new["messages"][1]
            for index in (4, 6):
                membership_changes[form][f"thinking_{index}"] += (
                    old["messages"][index]["thinking"] != new["messages"][index]["thinking"]
                )
            assert [m["tool_call"]["name"] for m in new["messages"] if "tool_call" in m] == [
                "lookup_human_gene_symbol", "check_multiplex_membership",
            ]
            membership = new["messages"][5]["content"]
            if form == 11:
                assert membership["status"] == "not_found"
                assert "present_gene_ids" not in membership and "absent_gene_ids" not in membership
                assert final["status"] == "not_found" and final["result"]["present"] is None
                assert final["continuation_decision"] == "defer" and final["blocked_reason"]
            else:
                assert lookup["gene_id"] in membership["present_gene_ids" if present else "absent_gene_ids"]
                assert final["status"] == "resolved" and final["result"]["present"] is present
                assert final["blocked_reason"] is None
            assert set(final["result"]) == {"gene_id", "gene_symbol", "multiplex_id", "present"}
            assert final["provenance"] == {
                "registry": lookup["provenance"]["registry"],
                "graph": membership["provenance"]["graph"],
            }
        if form == 3:
            symbol = old["messages"][-2]["content"]["gene_symbol"]
            expected["messages"][-1]["thinking"] = ABSENT_SYMBOL_REASONING.format(
                symbol=symbol,
            )
            assert new["messages"][1] == old["messages"][1]
        if form == 8:
            assert new["row_id"] == old["row_id"]
            assert new["tools"] == old["tools"]
            assert new["messages"][:3] == old["messages"][:3]
            calls = [m["tool_call"] for m in new["messages"] if "tool_call" in m]
            assert calls in ([old["messages"][2]["tool_call"]], [old["messages"][2]["tool_call"]] * 2)
            final = new["messages"][-1]["content"]
            case = "no_retry" if len(calls) == 1 else "recovered" if final["status"] == "resolved" else "exhausted"
            retry_outcomes[case] += 1
            assert new["metadata"] == {**old["metadata"], "blocked_reason_code": None if case == "recovered" else "tool_error"}
            assert new["messages"][3]["content"] == (old["messages"][3]["content"] if case == "no_retry" else {
                **old["messages"][3]["content"],
                "message": "registry unreachable; retries_remaining=1",
            })
            if case == "recovered":
                output = new["messages"][-2]["content"]
                assert final["result"] == {key: output[key] for key in ("gene_id", "gene_symbol")}
                assert final["provenance"] == output["provenance"]
                assert final["continuation_decision"] == "stop" and final["blocked_reason"] is None
                assert final["status"] == "resolved"
            else:
                assert {**final, "summary": None} == {**old["messages"][-1]["content"], "summary": None}
                assert "Retry " not in final["summary"]
                if case == "exhausted":
                    assert new["messages"][-2]["content"] == {**old["messages"][3]["content"], "message": "registry unreachable; retries_remaining=0"}
            # Other forms are compared as entire rows below; these trajectories
            # instead have the explicit preservation and evidence checks above.
            changed[form] += new_line != old_line
            continue
        if form in {1, 2, 3}:
            calls = [m["tool_call"] for m in new["messages"] if "tool_call" in m]
            assert len(calls) == 1
            assert calls[0]["name"] == "lookup_human_gene_symbol"
            final = new["messages"][-1]["content"]
            assert final["status"] == ("not_found" if form == 3 else "resolved")
            assert final["continuation_decision"] == (
                "defer" if form == 3 else "continue" if form == 2 else "stop"
            )
            if form == 3:
                assert final["blocked_reason"]
                assert set(final["result"]) == {"gene_symbol"}
            else:
                assert final["blocked_reason"] is None
                assert set(final["result"]) == {"gene_id", "gene_symbol"}
            assert final["provenance"]["graph"] is None
        assert new == expected
        if new_line != old_line:
            changed[form] += 1
    assert form_counts[1] == 1200 and form_counts[2] == 1600 and form_counts[3] == 1200
    assert changed == {1: 168, 2: 1600, 3: 1200, 8: 400, 9: 1000, 10: 700, 11: 300}
    assert inside_task_changes == {"user": 1600, "thinking": 1600}
    assert membership_framing == {
        9: {"direct": 500, "step": 500}, 10: {"direct": 350, "step": 350},
        11: {"direct": 150, "step": 150},
    }
    assert membership_changes == {
        9: {"direct": 71, "step": 500, "thinking_4": 1000, "thinking_6": 1000},
        10: {"direct": 47, "step": 350, "thinking_4": 700, "thinking_6": 700},
        11: {"direct": 20, "step": 150, "thinking_4": 300, "thinking_6": 300},
    }
    assert retry_outcomes == {"no_retry": 200, "recovered": 100, "exhausted": 100}
    assert manifest["row_counts"] == {
        "train": 10000, "validation": 1250, "test": 1250,
    }
    assert manifest["train_prompt_banks"]["1"]["path"] == (
        "runtime/world_model/s1_v4_prompt_banks.json"
    )
    assert manifest["train_prompt_banks"]["2"]["path"] == (
        "runtime/world_model/s1_v4_prompt_banks.json"
    )
    assert manifest["train_prompt_banks"]["9"]["path"] == (
        "runtime/world_model/s1_v4_prompt_banks.json"
    )
    assert manifest["train_prompt_banks"]["10"]["path"] == (
        "runtime/world_model/s1_v4_prompt_banks.json"
    )
    assert manifest["train_prompt_banks"]["11"]["path"] == (
        "runtime/world_model/s1_v4_prompt_banks.json"
    )
    assert manifest["train_membership_reasoning"] == MEMBERSHIP_REASONING
    assert manifest["train_absent_membership_reasoning"] == ABSENT_MEMBERSHIP_REASONING
    assert manifest["train_unknown_multiplex_reasoning"] == UNKNOWN_MULTIPLEX_REASONING
    assert manifest["train_final_reasoning"] == {
        "2": INSIDE_TASK_REASONING, "3": ABSENT_SYMBOL_REASONING,
    }


def test_heldout_rows_remain_byte_identical(revision):
    """Training wording must not change any held-out question or answer."""

    root = revision[3]
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (
            BASELINE / f"{split}.jsonl"
        ).read_bytes()


def test_repeated_build_is_byte_identical(revision, tmp_path):
    """The form-specific bank must remain part of a deterministic build."""

    contract, payload, source, root, _ = revision
    build_corpus(
        source, tmp_path, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize("form", [3, 4, 8, 12, "1", True])
def test_overlay_rejects_unreviewed_or_untyped_form_overrides(tmp_path, form):
    """Only integer forms 1, 2, 9, 10, and 11 have approved wrapper overrides."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    pins = payload["family_overrides"]["S1.1"]["train_prompt_banks"]
    payload["family_overrides"]["S1.1"]["train_prompt_banks"] = {form: pins[1]}
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="only S1.1.1, S1.1.2, S1.1.9, S1.1.10, or S1.1.11"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("form", [1, 2, 9, 10, 11])
def test_overlay_rejects_an_unpinned_form_bank(tmp_path, form):
    """A per-form wrapper bank needs the same content pin as the shared bank."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.1"]["train_prompt_banks"][form]["sha256"] = "0" * 64
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="does not match its pin"):
        apply_s1_overlay(load_contract(), path)


def test_unreviewed_policies_and_default_bank_remain_v5():
    """Reviewed edits must not alter another family's content policy."""

    v5, old_payload = apply_s1_overlay(load_contract(), S1_V5_OVERLAY_PATH)
    v6, new_payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    for section, settings in v6["curriculum"]["families"].items():
        old = dict(v5["curriculum"]["families"][section])
        new = dict(settings)
        old.pop("output_directory", None)
        new.pop("output_directory", None)
        if section == "S1.1":
            new.pop("train_prompt_banks")
            new.pop("train_final_reasoning")
            new.pop("train_lookup_retries")
            new.pop("train_membership_reasoning")
            new.pop("train_absent_membership_reasoning")
            new.pop("train_unknown_multiplex_reasoning")
        elif section == "S1.2":
            new.pop("train_prompt_banks")
            new.pop("train_final_reasoning")
            new.pop("train_version_suffix_reasoning")
            new.pop("train_no_symbol_reasoning")
            new.pop("train_near_miss_reasoning")
            new.pop("train_symbol_input_reasoning")
            new.pop("train_lookup_retries")
        elif section == "S1.3":
            new.pop("train_prompt_banks")
            new.pop("train_final_reasoning")
            assert set(new.pop("train_single_membership_reasoning")) == {"after_lookup", "direct", "step"}
            assert set(new.pop("train_multiple_membership_reasoning")) == {"after_lookup", "final"}
            assert set(new.pop("train_no_membership_reasoning")) == {"after_lookup", "final"}
            assert set(new.pop("train_user_choice_reasoning")) == {"direct", "step"}
            new.pop("train_membership_retries")
        elif section == "S1.4":
            assert set(new.pop("train_match_reasoning")) == {"direct", "step"}
            assert set(new.pop("train_mismatch_reasoning")) == {"direct", "step"}
            assert set(new.pop("train_unresolved_symbol_reasoning")) == {"direct", "step"}
            assert set(new.pop("train_ambiguous_candidate_reasoning")) == {"direct", "step"}
            assert set(new.pop("train_ambiguous_noncandidate_reasoning")) == {"direct", "step"}
            assert set(new.pop("train_mygene_match_reasoning")) == {"direct", "step"}
        elif section == "S1.6":
            assert set(new.pop("train_bulk_pen_reasoning")) == {"direct", "step"}
            assert set(new.pop("train_scpen_reasoning")) == {"direct", "step"}
            assert set(new.pop("train_humannet_reasoning")) == {"direct", "step"}
        if section in {"S1.6", "S1.7", "S1.8", "S1.9"}:
            templates = new.pop("train_final_reasoning")
            assert set(templates) == {
                "S1.6": {4, 6}, "S1.7": {2, 4, 8},
                "S1.8": {1, 2, 3, 4, 5, 6}, "S1.9": {1, 2},
            }[section]
            assert all(set(frames) == {"direct", "step"} for frames in templates.values())
        assert new == old
    assert new_payload["prompt_banks"] == old_payload["prompt_banks"]
    assert set(v6["curriculum"]["families"]["S1.1"]["train_prompt_banks"]) == {1, 2, 9, 10, 11}


@pytest.mark.parametrize("form", [1, 4, 9, 10, "2", True])
def test_final_reasoning_cannot_target_another_form(tmp_path, form):
    """Only forms 2 and 3 have reviewed final-thinking replacements."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.1"]["train_final_reasoning"] = {
        form: INSIDE_TASK_REASONING,
    }
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="only S1.1.2 or S1.1.3"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize(
    "text",
    [
        "Return the mapping with continue.",
        INSIDE_TASK_REASONING.replace("{symbol}", "{symbol.name}"),
        INSIDE_TASK_REASONING + " Use {multiplex_id}.",
        INSIDE_TASK_REASONING.replace(". Because", ".\nBecause"),
    ],
)
def test_completion_reasoning_requires_bound_fields(tmp_path, text):
    """Reject missing or invented fields and malformed template formatting."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.1"]["train_final_reasoning"][2] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="final reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize(
    "text",
    [
        "No mapping is available, so defer with a blocked reason.",
        INSIDE_TASK_REASONING,
        ABSENT_SYMBOL_REASONING + " Check {multiplex_id}.",
    ],
)
def test_absent_symbol_reasoning_binds_only_the_returned_symbol(tmp_path, text):
    """An absent-symbol answer has no resolved ID or multiplex to interpolate."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.1"]["train_final_reasoning"][3] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="final reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("reasoning_key", [
    "train_membership_reasoning", "train_absent_membership_reasoning",
    "train_unknown_multiplex_reasoning",
])
@pytest.mark.parametrize(
    ("phase", "text"),
    [
        ("after_lookup", "The lookup supplies {gene_id}. Check the multiplex."),
        ("direct", MEMBERSHIP_REASONING["after_lookup"]),
        ("step", MEMBERSHIP_REASONING["step"].replace("{gene_id}", "{gene_id.name}")),
        ("after_lookup", MEMBERSHIP_REASONING["after_lookup"] + "\n"),
    ],
)
def test_membership_reasoning_requires_bound_fields(tmp_path, reasoning_key, phase, text):
    """Bind each membership thought only to available evidence."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.1"][reasoning_key][phase] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="membership reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("remove", [True, False])
@pytest.mark.parametrize("reasoning_key", [
    "train_membership_reasoning", "train_absent_membership_reasoning",
    "train_unknown_multiplex_reasoning",
])
def test_membership_reasoning_requires_exactly_the_reviewed_phases(tmp_path, remove, reasoning_key):
    """The membership templates cover one intermediate step and both framings."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.1"][reasoning_key]
    if remove:
        reasoning.pop("after_lookup")
    else:
        reasoning["absent"] = MEMBERSHIP_REASONING["direct"]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="membership reasoning keys"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("corrupt_form", [1, 8])
def test_form_override_does_not_allow_arbitrary_failure_pair_prompts(corrupt_form):
    """Both members must match their pinned wrapper, even when banks differ."""

    fact = GeneFact("fact", ("unit",), "EXAMPLE", "ENSG00000000001")
    success = Scenario("train", 1, fact, "direct", 0, 0, "EXAMPLE")
    failure = replace(success, form=8, paired_success_id=success.scenario_id)
    restored = deepcopy(IDENTITY_PROMPTS)
    restored["train"]["direct"] = (
        "Only this needs an answer: {request}.",
        *restored["train"]["direct"][1:],
    )
    request = "the human Ensembl gene ID for EXAMPLE"
    rows = []
    for case in (success, failure):
        bank = restored if case.form == 1 else IDENTITY_PROMPTS
        prompt = bank["train"]["direct"][0].format(request=request)
        if case.form == corrupt_form:
            prompt = "An unconfigured request about ANOTHER_GENE."
        rows.append({
            "row_id": str(case.form), "metadata": {"form": case.form},
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "tool_call": {
                    "name": "lookup_human_gene_symbol",
                    "arguments": {"gene_symbol": "EXAMPLE"},
                }},
                {"role": "tool", "content": {}},
                {"role": "assistant", "content": {
                    "status": "resolved" if case.form == 1 else "error",
                }},
            ],
        })
    with pytest.raises(CorpusBuildError, match="failure pair changed its prompt"):
        audit_rows(
            {"train": rows}, {"train": [success, failure]},
            {"train": {1: 1, 8: 1}}, train_form_prompts={1: restored},
        )


@pytest.mark.parametrize(
    ("overlay_path", "baseline"),
    [
        (None, Path("data/world_model_v3/corpora/s1/s1_1")),
        (S1_V4_OVERLAY_PATH, V4),
        (S1_V5_OVERLAY_PATH, BASELINE),
    ],
)
def test_legacy_split_bytes_are_unchanged(overlay_path, baseline, tmp_path):
    """New rendering and audit support must reproduce frozen older splits."""

    contract = load_contract()
    payload = {}
    if overlay_path is not None:
        contract, payload = apply_s1_overlay(contract, overlay_path)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    build_corpus(
        source, tmp_path, contract=contract,
        prompt_banks=payload.get("prompt_banks"),
    )
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (
            baseline / f"{split}.jsonl"
        ).read_bytes()
