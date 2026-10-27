"""Protect historical rows and verify v9 identity/scope separation."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_3_corpus import (
    CHOICE_STANDALONE_SCOPE, CorpusBuildError, GeneFact, Scenario,
    _audit_choice_scope, _choice_assignments, _choice_thinking, _choice_users,
    build_corpus,
)
from runtime.world_model.s1_overlay import (
    S1_V6_OVERLAY_PATH, apply_s1_overlay, overlay_pin,
)
from runtime.world_model.s1_prompt_banks import (
    LATER_STEP_CUES, PromptBankError, identity_prompts, load_choice_replies,
)
from runtime.world_model.s1_1_corpus import IDENTITY_PROMPTS
from runtime.world_model.sources import file_sha256


BASELINE = Path("data/world_model_v4/corpora/s1_v6/s1_3")
CHOICE_PATH = Path("runtime/world_model/s1_v9_choice_replies.json")
CHOICE_PIN = {"path": str(CHOICE_PATH), "sha256": file_sha256(CHOICE_PATH)}


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build the historical baseline and targeted variant through normal audits."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    baseline = tmp_path_factory.mktemp("s1_3_v9_baseline")
    build_corpus(source, baseline, contract=contract,
                 overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
                 prompt_banks=payload["prompt_banks"])
    contract["curriculum"]["families"]["S1.3"].update(
        train_targeted_v9=True, train_choice_replies=CHOICE_PIN,
    )
    revised = tmp_path_factory.mktemp("s1_3_v9")
    manifest = build_corpus(source, revised, contract=contract,
                            overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
                            prompt_banks=payload["prompt_banks"])
    return baseline, revised, manifest


def test_revision_changes_only_choice_users_and_current_thinking(revision):
    """Keep every answer, evidence nonce, source fact, call, and other form intact."""

    baseline, revised, _ = revision
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        assert (baseline / name).read_bytes() == (BASELINE / name).read_bytes()
    for split in ("validation", "test"):
        assert (revised / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()
    changed = 0
    for old_raw, new_raw in zip(
        (baseline / "train.jsonl").read_bytes().splitlines(),
        (revised / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_raw), json.loads(new_raw)
        if old["metadata"]["form"] != 8:
            assert old_raw == new_raw
            continue
        changed += 1
        expected = deepcopy(old)
        expected["messages"][1]["content"] = new["messages"][1]["content"]
        expected["messages"][2]["content"]["summary"] = new["messages"][2]["content"]["summary"]
        expected["messages"][3]["content"] = new["messages"][3]["content"]
        expected["messages"][-1]["thinking"] = new["messages"][-1]["thinking"]
        assert new == expected
        assert new["messages"][2]["content"]["summary"] == old["messages"][2]["content"]["summary"].replace(
            "Give one Ensembl gene ID to continue.", "Please identify the intended Ensembl entry.",
        )
        earlier, final = new["messages"][2]["content"], new["messages"][-1]["content"]
        assert set(final["result"]) == {"gene_id", "gene_symbol", "candidate_gene_ids"}
        assert final["result"]["candidate_gene_ids"] == earlier["result"]["candidate_gene_ids"]
        assert final["provenance"] == earlier["provenance"]
        assert final["status"] == "resolved" and final["blocked_reason"] is None
        assert not any("tool_call" in message for message in new["messages"])
        assert "Selecting the ID establishes identity" in new["messages"][-1]["thinking"]
    assert changed == 1000


def test_actual_visible_scope_and_joint_coverage(revision):
    """Exercise both decisions and all phrases when either turn supplies scope."""

    _, revised, manifest = revision
    counts = Counter()
    phrases = load_choice_replies(CHOICE_PIN)
    phrase_coverage = defaultdict(set)
    for line in (revised / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["metadata"]["form"] != 8:
            continue
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        final = row["messages"][-1]["content"]
        workflow = final["continuation_decision"] == "continue"
        cues = tuple(bool(LATER_STEP_CUES.search(u)) if workflow else CHOICE_STANDALONE_SCOPE in u for u in users)
        placement = {(True, True): "both", (True, False): "earlier", (False, True): "latest"}[cues]
        # Standalone prompts have no workflow cues anywhere in either user turn.
        if not workflow:
            assert not any(LATER_STEP_CUES.search(u) for u in users)
        selected = final["result"]["gene_id"]
        variants = [i for i, phrase in enumerate(phrases) if phrase.format(gene_id=selected).rstrip(".") in users[-1]]
        assert len(variants) == 1
        phrase_coverage[placement, workflow].add(variants[0])
        counts[placement, workflow] += 1
    assert counts == {(p, w): n for p, n in (("both", 250), ("earlier", 125), ("latest", 125)) for w in (False, True)}
    assert all(variants == set(range(16)) for variants in phrase_coverage.values())
    assert len(phrase_coverage) == 6
    assert manifest["train_targeted_v9"]["pair_counts"] == {"both": 250, "earlier": 125, "latest": 125}
    assert manifest["train_targeted_v9"]["choice_replies"] == CHOICE_PIN
    assert len(manifest["train_targeted_v9"]["coverage"]) == 96
    assert len(manifest["reply_template_counts"]["train"]["S1.3.8"]) == 16


def test_scope_schedule_ignores_wrapper_and_phrase_bank_sizes():
    """Changing wording resources cannot change selected identities or scope."""

    scenarios = []
    for index in range(500):
        fact = GeneFact(f"fact-{index}", f"unit-{index}", f"GENE{index}", ("ENSG00000000001", "ENSG00000000002"), {}, ())
        for framing in ("direct", "step"):
            scenarios.append(Scenario("train", 8, fact, framing, index % 32, index,
                                      fact.gene_symbol, chosen_gene_id=fact.candidate_gene_ids[index % 2]))
    first = _choice_assignments(scenarios, seed=42, reply_count=16)
    altered = [replace(s, template_index=s.template_index + 100) for s in scenarios]
    second = _choice_assignments(altered, seed=42, reply_count=32)
    assert [first[s.scenario_id][0] for s in scenarios] == [second[s.scenario_id][0] for s in altered]
    assert all(first[a.scenario_id] == first[b.scenario_id] for a, b in zip(scenarios[::2], scenarios[1::2], strict=True))
    assert Counter(value[0] for value in first.values()) == {"both": 500, "earlier": 250, "latest": 250}


@pytest.mark.parametrize("placement", ["both", "earlier", "latest"])
@pytest.mark.parametrize("framing", ["direct", "step"])
def test_whole_history_audit_rejects_scope_leaks(placement, framing):
    """A cue inserted into either visible user turn must fail the targeted audit."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    reasoning = contract["curriculum"]["families"]["S1.3"]["train_user_choice_reasoning"]
    prompts = identity_prompts(payload["prompt_banks"], IDENTITY_PROMPTS, bank_name="identity_s1_3")
    fact = GeneFact("f", "u", "GENE", ("ENSG00000000001", "ENSG00000000002"), {}, ())
    scenario = Scenario("train", 8, fact, framing, 0, 0, "GENE", chosen_gene_id=fact.candidate_gene_ids[0])
    replies = load_choice_replies(CHOICE_PIN)
    users = _choice_users(scenario, placement, replies[0].format(gene_id=scenario.chosen_gene_id), prompts)
    for turn in (0, 1):
        row = {"messages": [{"role": "user", "content": u} for u in users] + [{"role": "assistant", "thinking": _choice_thinking(scenario, placement, reasoning)}]}
        row["messages"][turn]["content"] += " A later workflow requires this value."
        with pytest.raises(CorpusBuildError, match="scope cues"):
            _audit_choice_scope([row], [scenario], assignments={scenario.scenario_id: (placement, 0)}, replies=replies, prompts=prompts, reasoning=reasoning)


@pytest.mark.parametrize("field", ["summary", "blocked_reason", "thinking"])
def test_earlier_assistant_cannot_add_scope(field):
    """Check prior assistant prose as well as user instructions for scope leaks."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    reasoning = contract["curriculum"]["families"]["S1.3"]["train_user_choice_reasoning"]
    prompts = identity_prompts(payload["prompt_banks"], IDENTITY_PROMPTS, bank_name="identity_s1_3")
    fact = GeneFact("f", "u", "GENE", ("ENSG00000000001", "ENSG00000000002"), {}, ())
    scenario = Scenario("train", 8, fact, "step", 0, 0, "GENE", chosen_gene_id=fact.candidate_gene_ids[0])
    replies = load_choice_replies(CHOICE_PIN)
    earlier, latest = _choice_users(scenario, "latest", replies[0].format(gene_id=scenario.chosen_gene_id), prompts)
    prior = {"role": "assistant", "content": {"summary": "Please choose an entry.", "blocked_reason": "An entry must be chosen."}}
    (prior if field == "thinking" else prior["content"])[field] = "A later workflow needs the selected entry."
    row = {"messages": [
        {"role": "user", "content": earlier}, prior,
        {"role": "user", "content": latest},
        {"role": "assistant", "thinking": _choice_thinking(scenario, "latest", reasoning)},
    ]}
    with pytest.raises(CorpusBuildError, match="earlier assistant text leaks"):
        _audit_choice_scope([row], [scenario], assignments={scenario.scenario_id: ("latest", 0)}, replies=replies, prompts=prompts, reasoning=reasoning)


@pytest.mark.parametrize("mutation", ["scope", "heldout", "duplicate", "binding", "short", "digest"])
def test_choice_bank_rejects_invalid_content(tmp_path, mutation):
    """Reject hidden scope labels, held-out copying, weak variety, and stale pins."""

    payload = json.loads(CHOICE_PATH.read_text())
    if mutation == "scope":
        payload["replies"][0] = "Continue with {gene_id}."
    elif mutation == "heldout":
        payload["replies"][0] = "Use {gene_id}."
    elif mutation == "duplicate":
        payload["replies"][0] = payload["replies"][1]
    elif mutation == "binding":
        payload["replies"][0] = "I choose {symbol}."
    elif mutation == "short":
        payload["replies"] = payload["replies"][:-1]
    path = tmp_path / "replies.json"
    path.write_text(json.dumps(payload))
    pin = {"path": str(path), "sha256": "0" * 64 if mutation == "digest" else file_sha256(path)}
    with pytest.raises(PromptBankError):
        load_choice_replies(pin)
