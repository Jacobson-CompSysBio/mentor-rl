"""Verify reviewed S1.2 revisions and preserve every unreviewed or frozen row."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re

import pytest
import yaml

from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s0_corpus import EvidenceSequence
from runtime.world_model.s1_2_corpus import (
    CorpusBuildError, GeneFact, Scenario, build_corpus, build_row, filter_row_thinking,
)
from runtime.world_model.s1_overlay import (
    S1_V4_OVERLAY_PATH,
    S1_V5_OVERLAY_PATH,
    S1_V6_OVERLAY_PATH,
    S1OverlayError,
    apply_s1_overlay,
    overlay_pin,
)
from runtime.world_model.tools import WorldModelToolRuntime


BASELINE = Path("data/world_model_v4/corpora/s1_v5/s1_2")
DIRECT_REASONING = (
    "The registry resolves {gene_id} to {symbol}, completing the requested lookup. "
    "Return this mapping with registry evidence and stop."
)
NOT_FOUND_REASONING = (
    "The registry reports not_found for {gene_id} in release 116, so the requested symbol is unavailable. "
    "Return only the queried gene_id with registry evidence and defer with a blocked reason until a current ID is supplied. "
    "Do not retry this negative result, guess a successor, or invent a graph lookup."
)
INVALID_INPUT_REASONING = (
    "The registry rejects {input}; return invalid_input with exactly input and expected:human_gene_id, preserving the original text. "
    "Keep registry evidence and set continuation_decision:defer with a blocked reason; do not repair the input or call again."
)
VERSION_SUFFIX_REASONING = {
    "direct": (
        "The registry ignores the numeric version suffix and returns {gene_id} with symbol {symbol}, completing the requested lookup. "
        "Return this mapping with registry evidence and stop."
    ),
    "step": (
        "The registry ignores the numeric version suffix and returns {gene_id} with symbol {symbol}, completing the requested lookup. "
        "Because the user needs this mapping for a later workflow step, return it with registry evidence and continue."
    ),
}
NO_SYMBOL_REASONING = {
    "direct": (
        "The registry reports resolved for {gene_id} with gene_symbol:null, completing the lookup without an assigned symbol. "
        "Return the ID and null symbol with registry evidence, no blocked reason, and stop."
    ),
    "step": (
        "The registry reports resolved for {gene_id} with gene_symbol:null, completing the lookup without an assigned symbol. "
        "Because the user's workflow needs this result, return the ID and null symbol with registry evidence, no blocked reason, and continue."
    ),
}
NEAR_MISS_REASONING = {
    "direct": (
        "The registry resolves {gene_id} to {symbol}. "
        "Because the user requested only this lookup, return the mapping with registry evidence and stop."
    ),
    "step": (
        "The registry resolves {gene_id} to {symbol}. "
        "Because the user needs this mapping for a later workflow step, return it with registry evidence and continue."
    ),
}
SYMBOL_INPUT_REASONING = {
    "direct": (
        "The registry confirms {symbol} as the symbol for {gene_id}; no membership check is requested. "
        "Because the user requested only this lookup, return the mapping with registry evidence and stop."
    ),
    "step": (
        "The registry confirms {symbol} as the symbol for {gene_id}; no membership check is requested. "
        "Because the user needs this mapping for a later workflow step, return it with registry evidence and continue."
    ),
}


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build v6 from pinned inputs, including semantic and wording audits."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_2_v6")
    manifest = build_corpus(
        source, root, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    return contract, payload, source, root, manifest


def test_only_reviewed_lookup_wording_and_final_thinking_change(revision):
    """Retain V5 identities, framing, evidence, targets, and failure partners."""

    root, manifest = revision[3:]
    banks = {
        version: json.loads(Path(f"runtime/world_model/s1_{version}_prompt_banks.json").read_text())["identity"]
        for version in ("v4", "v5")
    }
    changed = {form: Counter() for form in (1, 3, 4, 5, 6, 8, 9)}
    coverage = {
        1: {"direct": Counter()},
        3: {"direct": Counter(), "step": Counter()},
        4: {"direct": Counter(), "step": Counter()},
        5: {"direct": Counter(), "step": Counter()},
        6: {"direct": Counter(), "step": Counter()},
        8: {"direct": Counter(), "step": Counter()},
        9: {"direct": Counter(), "step": Counter()},
    }
    framings = {form: Counter() for form in (3, 4, 5, 6, 8, 9)}
    unchanged = 0
    for old_line, new_line in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (root / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_line), json.loads(new_line)
        form = old["metadata"]["form"]
        if form == 11:
            # Its bounded retry trajectories have a separate full-row audit.
            continue
        if form not in changed:
            assert old_line == new_line
            unchanged += 1
            continue
        user_index = next(i for i, message in enumerate(old["messages"]) if message["role"] == "user")
        call = next(message["tool_call"] for message in old["messages"] if "tool_call" in message)
        argument_name = "gene_symbol" if form == 9 else "gene_id"
        tool_name = "lookup_human_gene_symbol" if form == 9 else "lookup_human_ensembl_gene_id"
        query = call["arguments"][argument_name]
        decision = old["messages"][-1]["content"]["continuation_decision"]
        request = f"the human gene symbol corresponding to {query}"
        indices = [
            (framing, index)
            for framing in ("direct", "step")
            for index, wrapper in enumerate(banks["v5"][framing])
            if wrapper.format(request=request) == old["messages"][user_index]["content"]
        ]
        assert len(indices) == 1
        framing, index = indices[0]
        coverage[form][framing][index] += 1
        output = next(message["content"] for message in old["messages"] if message["role"] == "tool")
        expected = deepcopy(old)
        if form != 5:
            expected["messages"][user_index]["content"] = banks["v4"][framing][index].format(request=request)
        reasoning = {
            1: DIRECT_REASONING,
            3: VERSION_SUFFIX_REASONING[framing],
            4: NO_SYMBOL_REASONING[framing],
            5: NOT_FOUND_REASONING,
            6: INVALID_INPUT_REASONING,
            8: NEAR_MISS_REASONING[framing],
            9: SYMBOL_INPUT_REASONING[framing],
        }[form]
        expected["messages"][-1]["thinking"] = (
            reasoning.format(input=output["input"]) if form == 6 else
            reasoning.format(gene_id=output["gene_id"], symbol=output["gene_symbol"])
        )
        assert new == expected
        changed[form]["rows"] += new_line != old_line
        changed[form]["users"] += old["messages"][user_index] != new["messages"][user_index]
        changed[form]["final_thinking"] += old["messages"][-1]["thinking"] != new["messages"][-1]["thinking"]
        calls = [m["tool_call"] for m in new["messages"] if "tool_call" in m]
        assert calls == [{"name": tool_name, "arguments": {argument_name: query}}]
        final = new["messages"][-1]["content"]
        assert final["status"] == {5: "not_found", 6: "invalid_input"}.get(form, "resolved")
        assert final["continuation_decision"] == decision
        result_keys = {
            5: ("gene_id",), 6: ("input", "expected"),
        }.get(form, ("gene_id", "gene_symbol"))
        assert final["result"] == {key: output[key] for key in result_keys}
        assert final["provenance"] == output["provenance"]
        assert final["provenance"]["graph"] is None
        if form in {5, 6}:
            assert decision == "defer" and final["blocked_reason"]
            if form == 5:
                assert output["status"] == "not_found" and output["gene_symbol"] is None
                assert final["result"]["gene_id"] == query
            else:
                assert output["status"] == "invalid_input"
                assert final["result"] == {"input": query, "expected": "human_gene_id"}
                assert re.fullmatch(r"(?i:ENSG)\d{10}", query)
        else:
            assert final["blocked_reason"] is None
        if form == 3:
            assert re.fullmatch(r"(?i:ENSG)\d{11}\.\d+", query)
            assert final["result"]["gene_id"] == query.rsplit(".", 1)[0].upper()
        if form == 4:
            assert output["gene_symbol"] is None and final["result"]["gene_symbol"] is None
            assert final["result"]["gene_id"] == query.upper()
        if form in {3, 4, 5, 6, 8, 9}:
            framings[form][framing] += 1
            changed[form][framing] += old["messages"][user_index] != new["messages"][user_index]
    assert changed == {
        1: {"rows": 800, "users": 110, "final_thinking": 800},
        3: {"rows": 1000, "users": 624, "final_thinking": 1000, "direct": 62, "step": 562},
        4: {"rows": 3100, "users": 1931, "final_thinking": 3100, "direct": 190, "step": 1741},
        5: {"rows": 1000, "users": 0, "final_thinking": 1000, "direct": 0, "step": 0},
        6: {"rows": 500, "users": 310, "final_thinking": 500, "direct": 29, "step": 281},
        8: {"rows": 500, "users": 310, "final_thinking": 500, "direct": 29, "step": 281},
        9: {"rows": 500, "users": 310, "final_thinking": 500, "direct": 29, "step": 281},
    }
    assert framings == {
        3: {"direct": 438, "step": 562},
        4: {"direct": 1359, "step": 1741},
        5: {"direct": 438, "step": 562},
        6: {"direct": 219, "step": 281},
        8: {"direct": 219, "step": 281},
        9: {"direct": 219, "step": 281},
    }
    assert unchanged == 2800
    assert len(coverage[1]["direct"]) == 64 and set(coverage[1]["direct"].values()) == {12, 13}
    for framing, uses in (("direct", {6, 7}), ("step", {8, 9})):
        assert len(coverage[3][framing]) == 64 and set(coverage[3][framing].values()) == uses
        assert len(coverage[5][framing]) == 64 and set(coverage[5][framing].values()) == uses
    for framing, uses in (("direct", {21, 22}), ("step", {27, 28})):
        assert len(coverage[4][framing]) == 64 and set(coverage[4][framing].values()) == uses
    for framing, uses in (("direct", {3, 4}), ("step", {4, 5})):
        assert len(coverage[6][framing]) == 64 and set(coverage[6][framing].values()) == uses
        assert len(coverage[8][framing]) == 64 and set(coverage[8][framing].values()) == uses
        assert len(coverage[9][framing]) == 64 and set(coverage[9][framing].values()) == uses
    assert manifest["row_counts"] == {"train": 11000, "validation": 1375, "test": 1375}
    assert manifest["train_final_reasoning"] == {
        "1": DIRECT_REASONING, "5": NOT_FOUND_REASONING, "6": INVALID_INPUT_REASONING,
    }
    assert manifest["train_version_suffix_reasoning"] == VERSION_SUFFIX_REASONING
    assert manifest["train_no_symbol_reasoning"] == NO_SYMBOL_REASONING
    assert manifest["train_near_miss_reasoning"] == NEAR_MISS_REASONING
    assert manifest["train_symbol_input_reasoning"] == SYMBOL_INPUT_REASONING
    assert set(manifest["train_prompt_banks"]) == {"1", "3", "4", "6", "8", "9"}
    assert manifest["train_prompt_banks"]["1"]["path"] == "runtime/world_model/s1_v4_prompt_banks.json"
    assert manifest["train_prompt_banks"]["3"]["path"] == "runtime/world_model/s1_v4_prompt_banks.json"
    assert manifest["train_prompt_banks"]["4"]["path"] == "runtime/world_model/s1_v4_prompt_banks.json"
    assert manifest["train_prompt_banks"]["6"]["path"] == "runtime/world_model/s1_v4_prompt_banks.json"
    assert manifest["train_prompt_banks"]["8"]["path"] == "runtime/world_model/s1_v4_prompt_banks.json"
    assert manifest["train_prompt_banks"]["9"]["path"] == "runtime/world_model/s1_v4_prompt_banks.json"
    assert manifest["failure_injection"]["pair_counts"] == {"train": 800, "validation": 50, "test": 50}
    assert manifest["duplicate_prompt_tool_outcome"]["splits"]["train"]["intended_success_failure_pair_count"] == 690
    assert manifest["masked_wording_classifier"]["pass"]


def test_id_retry_rows_preserve_inputs_and_bind_each_final_result(revision):
    """Audit all 800 failure partners, including bounded recovery and exhaustion."""

    root, manifest = revision[3:]
    policy = json.loads(Path("runtime/world_model/s1_v6_id_lookup_retries.json").read_text())
    baseline = [json.loads(line) for line in (BASELINE / "train.jsonl").read_text().splitlines()]
    failures = {row["row_id"]: row for row in baseline if row["metadata"]["form"] == 11}
    successful_results = {
        row["messages"][2]["tool_call"]["arguments"]["gene_id"]: row["messages"][3]["content"]
        for row in baseline if row["metadata"]["form"] == 1
    }
    counts = Counter()
    evidence_ids = set()
    lowercase = 0
    for row in map(json.loads, (root / "train.jsonl").open()):
        if row["metadata"]["form"] != 11:
            continue
        old = failures[row["row_id"]]
        assert row["messages"][:3] == old["messages"][:3]
        assert row["tools"] == old["tools"]
        query = old["messages"][2]["tool_call"]["arguments"]["gene_id"]
        lowercase += query.startswith("ensg")
        calls = [m["tool_call"] for m in row["messages"] if "tool_call" in m]
        outputs = [m["content"] for m in row["messages"] if m["role"] == "tool"]
        assert len(calls) in (1, 2) and calls == [old["messages"][2]["tool_call"]] * len(calls)
        case = "no_retry" if len(calls) == 1 else "recovered" if outputs[-1]["status"] == "resolved" else "exhausted"
        counts[case] += 1
        first = deepcopy(old["messages"][3]["content"])
        if case != "no_retry":
            first["message"] = "registry unreachable; retries_remaining=1"
            assert row["messages"][4]["thinking"] == policy["thinking"]["retry"]
        assert outputs[0] == first
        final = row["messages"][-1]
        metadata = deepcopy(old["metadata"])
        if case == "recovered":
            canonical = successful_results[query]
            assert outputs[-1]["status"] == "resolved"
            assert outputs[-1]["gene_id"] == canonical["gene_id"]
            assert outputs[-1]["gene_symbol"] == canonical["gene_symbol"]
            answer = final["content"]
            assert answer["status"] == "resolved" and answer["continuation_decision"] == "stop"
            assert answer["result"] == {key: canonical[key] for key in ("gene_id", "gene_symbol")}
            assert answer["provenance"] == outputs[-1]["provenance"]
            assert answer["provenance"]["graph"] is None and answer["blocked_reason"] is None
            evidence_id = answer["provenance"]["registry"]["evidence_id"]
            assert evidence_id not in evidence_ids
            evidence_ids.add(evidence_id)
            metadata["blocked_reason_code"] = None
            expected_thought = policy["thinking"][case].format(
                input=query, gene_id=canonical["gene_id"], symbol=canonical["gene_symbol"],
            )
        else:
            if case == "exhausted":
                assert outputs[-1] == {**first, "message": "registry unreachable; retries_remaining=0"}
            expected_answer = deepcopy(old["messages"][-1]["content"])
            expected_answer["summary"] = policy["summaries"][case].format(input=query)
            assert final["content"] == expected_answer
            assert final["content"]["result"] == {
                "failed_tool": "lookup_human_ensembl_gene_id", "error_code": "BACKEND_UNAVAILABLE", "gene_id": query,
            }
            expected_thought = policy["thinking"][case].format(input=query)
        assert row["metadata"] == metadata
        assert final["thinking"] == expected_thought
    assert counts == {"no_retry": 400, "recovered": 200, "exhausted": 200}
    assert lowercase == 46 and len(evidence_ids) == 200
    assert manifest["train_lookup_retries"]["case_counts"] == counts
    assert manifest["train_lookup_retries"]["max_retries"] == 1


def test_all_heldout_rows_remain_byte_identical(revision):
    """Freeze whole splits, including direct, workflow, and versioned lookups."""

    for split in ("validation", "test"):
        path = revision[3] / f"{split}.jsonl"
        assert path.read_bytes() == (BASELINE / path.name).read_bytes()
        counts = Counter(row["metadata"]["form"] for row in map(json.loads, path.open()))
        assert counts[1] == 63 and counts[2] == 63 and counts[3] == 125 and counts[4] == 388 and counts[5] == 125
        assert counts[6] == 62 and counts[8] == 62 and counts[9] == 62


def test_reviewed_decision_phrase_keeps_the_other_thinking_checks(revision):
    """Permit only the exact pinned final thought in form-6 training rows."""

    row = next(
        row for row in map(json.loads, (revision[3] / "train.jsonl").open())
        if row["metadata"]["form"] == 6
    )
    query = row["messages"][2]["tool_call"]["arguments"]["gene_id"]
    scenario = Scenario(
        split="train", form=6,
        fact=GeneFact(fact_id="filter-check", split_units=("filter-check",), gene_id=query.upper() + "0"),
        framing="direct", template_index=0, occurrence=0, input_value=query,
    )
    assert filter_row_thinking(row, scenario, final_reasoning=INVALID_INPUT_REASONING) == 2
    with pytest.raises(CorpusBuildError, match="schema/provenance"):
        filter_row_thinking(row, scenario)
    for changed in (replace(scenario, split="test"), replace(scenario, form=5)):
        with pytest.raises(CorpusBuildError, match="schema/provenance"):
            filter_row_thinking(row, changed, final_reasoning=INVALID_INPUT_REASONING)

    altered = deepcopy(row)
    altered["messages"][-1]["thinking"] += " Another sentence."
    with pytest.raises(CorpusBuildError, match="schema/provenance"):
        filter_row_thinking(altered, scenario, final_reasoning=INVALID_INPUT_REASONING)
    premature = deepcopy(row)
    premature["messages"][2]["thinking"] = row["messages"][-1]["thinking"]
    with pytest.raises(CorpusBuildError, match="schema/provenance"):
        filter_row_thinking(premature, scenario, final_reasoning=INVALID_INPUT_REASONING)
    extra_schema = deepcopy(row)
    template = INVALID_INPUT_REASONING + " Copy provenance."
    extra_schema["messages"][-1]["thinking"] = template.format(input=query)
    with pytest.raises(CorpusBuildError, match="schema/provenance"):
        filter_row_thinking(extra_schema, scenario, final_reasoning=template)
    for field, value, error in (
        ("status", "not_found", "thinking status"),
        ("continuation_decision", "stop", "thinking decision"),
    ):
        inconsistent = deepcopy(row)
        inconsistent["messages"][-1]["content"][field] = value
        with pytest.raises(CorpusBuildError, match=error):
            filter_row_thinking(inconsistent, scenario, final_reasoning=INVALID_INPUT_REASONING)


def test_repeated_build_is_byte_identical(revision, tmp_path):
    """The new form-specific inputs remain part of the deterministic manifest."""

    contract, payload, source, root, _ = revision
    build_corpus(
        source, tmp_path, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize(("key", "form"), [
    *(("train_prompt_banks", form) for form in (2, 5, 11, "1", True)),
    *(("train_final_reasoning", form) for form in (2, 8, 11, "1", True)),
])
def test_overrides_cannot_target_an_unreviewed_or_untyped_form(tmp_path, key, form):
    """Reviewed S1.2 overrides permit no changes to another form."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    values = payload["family_overrides"]["S1.2"][key]
    payload["family_overrides"]["S1.2"][key] = {form: values[1]}
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="only S1.2.1"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("form", [1, 3, 4, 6, 8, 9])
def test_reviewed_bank_requires_its_content_pin(tmp_path, form):
    """A historical bank is reusable only with its verified bytes."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.2"]["train_prompt_banks"][form]["sha256"] = "0" * 64
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="does not match its pin"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("text", [
    "The task is complete, so stop.",
    DIRECT_REASONING.replace("{symbol}", "{symbol.name}"),
    DIRECT_REASONING + " Use {multiplex_id}.",
    DIRECT_REASONING + "\n",
])
def test_final_thinking_requires_returned_fields(tmp_path, text):
    """Reject missing evidence bindings, invented slots, and malformed formatting."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.2"]["train_final_reasoning"][1] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="final reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("form", [3, 4, 8, 9])
def test_flat_final_reasoning_cannot_replace_framed_lookup_thinking(tmp_path, form):
    """Framed lookup forms use their reviewed maps, never the flat one."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.2"]["train_final_reasoning"] = {form: DIRECT_REASONING}
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="final reasoning must name only S1.2.1"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("text", [
    NOT_FOUND_REASONING.replace("{gene_id}", "this ID"),
    NOT_FOUND_REASONING.replace("{gene_id}", "{gene_id.name}"),
    NOT_FOUND_REASONING.replace("requested symbol", "{symbol}"),
    NOT_FOUND_REASONING + "\n",
])
def test_not_found_thinking_binds_only_the_queried_id(tmp_path, text):
    """Keep the longer deferral rule bound to the exact input, with no symbol slot."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.2"]["train_final_reasoning"][5] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="final reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("text", [
    INVALID_INPUT_REASONING.replace("{input}", "this input"),
    INVALID_INPUT_REASONING.replace("{input}", "{gene_id}"),
    INVALID_INPUT_REASONING.replace("{input}", "{input.name}"),
    INVALID_INPUT_REASONING + " Use {symbol}.",
    INVALID_INPUT_REASONING + "\n",
])
def test_invalid_input_thinking_binds_only_the_unchanged_input(tmp_path, text):
    """Bind malformed text without implying a resolved ID or allowing extra slots."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.2"]["train_final_reasoning"][6] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="final reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("framing", ["direct", "step"])
@pytest.mark.parametrize("text", [
    "The task is complete, so stop.",
    VERSION_SUFFIX_REASONING["direct"].replace("{gene_id}", "{gene_id.name}"),
    VERSION_SUFFIX_REASONING["direct"] + " Use {multiplex_id}.",
    VERSION_SUFFIX_REASONING["direct"] + "\n",
])
def test_version_suffix_thinking_binds_evidence(tmp_path, framing, text):
    """Both framing templates bind only the registry's returned ID and symbol."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.2"]["train_version_suffix_reasoning"][framing] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="version-suffix reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("remove", [True, False])
def test_version_suffix_thinking_requires_both_reviewed_framings(tmp_path, remove):
    """Reject missing framings and additions outside the approved map."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.2"]["train_version_suffix_reasoning"]
    if remove:
        reasoning.pop("step")
    else:
        reasoning["other"] = VERSION_SUFFIX_REASONING["direct"]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="version-suffix reasoning keys"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("framing", ["direct", "step"])
@pytest.mark.parametrize("text", [
    "The task is complete, so stop.",
    NO_SYMBOL_REASONING["direct"].replace("{gene_id}", "{gene_id.name}"),
    NO_SYMBOL_REASONING["direct"].replace("gene_symbol:null", "{symbol}"),
    NO_SYMBOL_REASONING["direct"] + "\n",
])
def test_no_symbol_thinking_binds_only_the_returned_id(tmp_path, framing, text):
    """A null-symbol thought binds the valid ID without inventing a symbol."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.2"]["train_no_symbol_reasoning"][framing] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="no-symbol reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("remove", [True, False])
def test_no_symbol_thinking_requires_both_reviewed_framings(tmp_path, remove):
    """Keep the null-symbol completion rule scoped to both approved framings."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.2"]["train_no_symbol_reasoning"]
    if remove:
        reasoning.pop("step")
    else:
        reasoning["other"] = NO_SYMBOL_REASONING["direct"]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="no-symbol reasoning keys"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize(("reasoning_key", "label"), [
    ("train_near_miss_reasoning", "near-miss"),
    ("train_symbol_input_reasoning", "symbol-input"),
])
@pytest.mark.parametrize("framing", ["direct", "step"])
@pytest.mark.parametrize("text", [
    "The task is complete, so stop.",
    NEAR_MISS_REASONING["direct"].replace("{gene_id}", "{gene_id.name}"),
    NEAR_MISS_REASONING["direct"].replace("{symbol}", "the symbol"),
    NEAR_MISS_REASONING["direct"] + " Use {multiplex_id}.",
    NEAR_MISS_REASONING["direct"] + "\n",
])
def test_framed_lookup_thinking_binds_returned_fields(tmp_path, reasoning_key, label, framing, text):
    """Near-miss and symbol-input thoughts bind only the returned ID and symbol."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.2"][reasoning_key][framing] = text
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match=f"{label} reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize(("reasoning_key", "label"), [
    ("train_near_miss_reasoning", "near-miss"),
    ("train_symbol_input_reasoning", "symbol-input"),
])
@pytest.mark.parametrize("remove", [True, False])
def test_framed_lookup_thinking_requires_exactly_both_reviewed_framings(tmp_path, reasoning_key, label, remove):
    """Reject incomplete maps and additions outside the direct/workflow pair."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.2"][reasoning_key]
    if remove:
        reasoning.pop("step")
    else:
        reasoning["other"] = reasoning["direct"]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match=f"{label} reasoning keys"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("form", [3, 4, 5, 6, 8, 9])
@pytest.mark.parametrize("split", ["train", "test"])
def test_matched_task_scope_changes_only_unblocked_continuation(form, split):
    """Hold lookup evidence fixed; workflow requests continue only when resolved."""

    contract = load_contract()
    named_id, unnamed_id, absent_id, other_id = (f"ENSG{value:011d}" for value in (1, 2, 3, 4))
    payload = {
        "coordinate_system": {
            "ensembl_release": "116",
            "identifier_registry_id": "ensembl_homo_sapiens_116_v3",
            "multiplex_id": "full_brain_multiplex_v1",
            "multiplex_gene_rule": "test_fixture",
        },
        "genes_by_id": {
            named_id: ["ONE", "1", "protein_coding"],
            unnamed_id: [None, "1", "lncRNA"],
            other_id: ["FOUR", "1", "protein_coding"],
        },
        "multiplex_gene_ids": [],
        "sources": {
            "ensembl_gtf": {"file_name": "registry.gtf.gz", "sha256": "0" * 64},
            "mentor_ev_dendrogram": {"file_name": "dendrogram.tsv", "sha256": "1" * 64},
        },
    }
    gene_id = {4: unnamed_id, 5: absent_id}.get(form, named_id)
    query = {3: named_id + ".3", 6: named_id[:-1], 9: "one"}.get(form, gene_id)
    fact = GeneFact(
        fact_id="scope-test", split_units=("scope-test",),
        gene_id=gene_id, gene_symbol=None if form in {4, 5} else "ONE",
        kind="retired" if form == 5 else "current",
    )
    rows = []
    for framing in ("direct", "step"):
        scenario = Scenario(
            split=split, form=form, fact=fact, framing=framing,
            template_index=0, occurrence=0, input_value=query,
            distractor=replace(fact, fact_id="distractor", gene_id=other_id, gene_symbol="FOUR") if form == 8 else None,
        )
        row, _ = build_row(
            scenario,
            runtime=WorldModelToolRuntime(
                payload=payload, contract=contract, evidence_id_factory=EvidenceSequence(),
            ),
            contract=contract, tools=tool_definitions(contract),
            no_symbol_reasoning=NO_SYMBOL_REASONING,
            version_suffix_reasoning=VERSION_SUFFIX_REASONING,
            near_miss_reasoning=NEAR_MISS_REASONING,
            symbol_input_reasoning=SYMBOL_INPUT_REASONING,
            final_reasoning={5: NOT_FOUND_REASONING, 6: INVALID_INPUT_REASONING}.get(form),
        )
        rows.append(row)
        final = row["messages"][-1]
        assert final["content"]["continuation_decision"] == (
            "defer" if form in {5, 6} else "continue" if framing == "step" else "stop"
        )
        if split == "train" and form not in {5, 6}:
            clause = "user's workflow needs this result" if form == 4 else "user needs this mapping for a later workflow step"
            assert (clause in final["thinking"]) == (framing == "step")
        if split == "test":
            assert "Because the user" not in final["thinking"]
        if form == 9:
            assert [m["tool_call"] for m in row["messages"] if "tool_call" in m] == [
                {"name": "lookup_human_gene_symbol", "arguments": {"gene_symbol": "one"}}
            ]
            assert final["content"]["result"] == {"gene_id": named_id, "gene_symbol": "ONE"}
    direct, step = rows
    assert [message for message in direct["messages"] if message["role"] == "tool"] == [
        message for message in step["messages"] if message["role"] == "tool"
    ]
    assert [message["tool_call"] for message in direct["messages"] if "tool_call" in message] == [
        message["tool_call"] for message in step["messages"] if "tool_call" in message
    ]
    for key in ("status", "result", "provenance", "blocked_reason"):
        assert direct["messages"][-1]["content"][key] == step["messages"][-1]["content"][key]


@pytest.mark.parametrize(("overlay_path", "baseline"), [
    (None, Path("data/world_model_v3/corpora/s1/s1_2")),
    (S1_V4_OVERLAY_PATH, Path("data/world_model_v4/corpora/s1_v4/s1_2")),
    (S1_V5_OVERLAY_PATH, BASELINE),
])
def test_legacy_builds_reproduce_frozen_split_bytes(overlay_path, baseline, tmp_path):
    """Keep the no-overlay, V4, and V5 builders compatible with trained data."""

    contract, payload = load_contract(), {}
    if overlay_path is not None:
        contract, payload = apply_s1_overlay(contract, overlay_path)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    build_corpus(source, tmp_path, contract=contract, prompt_banks=payload.get("prompt_banks"))
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (baseline / f"{split}.jsonl").read_bytes()
