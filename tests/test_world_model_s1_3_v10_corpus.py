"""Keep choice scope grounded in user turns without a fixed completion cue."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_1_corpus import IDENTITY_PROMPTS
from runtime.world_model.s1_3_corpus import (
    CHOICE_STANDALONE_SCOPE, CorpusBuildError, GeneFact, Scenario,
    _audit_choice_scope, _choice_thinking, _choice_users, build_corpus,
)
from runtime.world_model.s1_overlay import apply_s1_overlay, overlay_pin
from runtime.world_model.s1_prompt_banks import (
    LATER_STEP_CUES, identity_prompts, load_choice_replies,
)


BASELINE = Path("data/world_model_v4/corpora/s1_v9/s1_3")
V9_OVERLAY = Path("runtime/world_model/s1_v9.yaml")
V10_OVERLAY = Path("runtime/world_model/s1_v10.yaml")


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build both overlays through their semantic, wording, and source audits."""

    results = []
    for version, path in (("v9", V9_OVERLAY), ("v10", V10_OVERLAY)):
        contract, payload = apply_s1_overlay(load_contract(), path)
        source = Path(contract["source_index"]["output_directory"]) / (
            f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
        )
        output = tmp_path_factory.mktemp(f"s1_3_{version}")
        manifest = build_corpus(
            source, output, contract=contract, overlay=overlay_pin(path, payload),
            prompt_banks=payload["prompt_banks"],
        )
        results.append((output, manifest))
    return results


def test_only_choice_scope_prose_and_final_thinking_change(revision):
    """Freeze historical bytes, held-out splits, facts, answers, and other forms."""

    (baseline, _), (revised, _) = revision
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        assert (baseline / name).read_bytes() == (BASELINE / name).read_bytes()
    for split in ("validation", "test"):
        assert (revised / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()
    changed = Counter()
    for old_raw, new_raw in zip(
        (baseline / "train.jsonl").read_bytes().splitlines(),
        (revised / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_raw), json.loads(new_raw)
        if old["metadata"]["form"] != 8:
            assert old_raw == new_raw
            continue
        final = new["messages"][-1]["content"]
        decision = final["continuation_decision"]
        changed[decision] += 1
        expected = deepcopy(old)
        for message in expected["messages"]:
            if message["role"] == "user" and decision == "stop":
                message["content"] = message["content"].removeprefix(CHOICE_STANDALONE_SCOPE + " ")
        expected["messages"][-1]["thinking"] = new["messages"][-1]["thinking"]
        assert old["messages"][-1]["thinking"] != new["messages"][-1]["thinking"]
        assert new == expected
        earlier = new["messages"][2]["content"]
        assert final["status"] == "resolved" and final["blocked_reason"] is None
        assert set(final["result"]) == {"gene_id", "gene_symbol", "candidate_gene_ids"}
        assert final["result"]["candidate_gene_ids"] == earlier["result"]["candidate_gene_ids"]
        assert final["provenance"] == earlier["provenance"]
        assert not any("tool_call" in message for message in new["messages"])
        thinking = new["messages"][-1]["thinking"]
        assert "Set status to resolved" in thinking
        assert f"set the continuation decision to {decision}." in thinking
    assert changed == {"stop": 500, "continue": 500}


def test_scope_coverage_uses_actual_visible_requests(revision):
    """Cover ordinary standalone requests and workflow scope in either user turn."""

    (_, historical_manifest), (revised, manifest) = revision
    historical = historical_manifest["train_targeted_v9"]
    current = manifest["train_targeted_v10"]
    for key in ("pair_counts", "reply_count", "assignment_sha256", "coverage", "choice_replies"):
        assert current[key] == historical[key]
    assert current["pass"]
    phrases = load_choice_replies(current["choice_replies"])
    counts = Counter()
    phrase_coverage = defaultdict(set)
    standalone_wrappers = set()
    for line in (revised / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["metadata"]["form"] != 8:
            continue
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        thinking = row["messages"][-1]["thinking"]
        final = row["messages"][-1]["content"]
        cues = tuple(bool(LATER_STEP_CUES.search(user)) for user in users)
        assert not any(CHOICE_STANDALONE_SCOPE in user for user in users)
        if final["continuation_decision"] == "stop":
            assert cues == (False, False)
            assert "neither user turn requests further work" in thinking
            assert "scope is explicit" not in thinking
            placement = "ordinary_standalone"
            standalone_wrappers.add(re.sub(
                rf"(?<!\w){re.escape(final['result']['gene_symbol'])}(?!\w)",
                "<SYMBOL>", users[0], flags=re.I,
            ))
        else:
            placement = {(True, True): "both", (True, False): "earlier", (False, True): "latest"}[cues]
            source = {"both": "both user turns", "earlier": "the earlier user turn", "latest": "the latest user turn"}[placement]
            assert f"later workflow scope is explicit in {source}" in thinking
        selected = final["result"]["gene_id"]
        variants = [i for i, phrase in enumerate(phrases) if phrase.format(gene_id=selected).rstrip(".") in users[-1]]
        assert len(variants) == 1
        phrase_coverage[placement].add(variants[0])
        counts[placement] += 1
    assert counts == {"ordinary_standalone": 500, "both": 250, "earlier": 125, "latest": 125}
    assert all(variants == set(range(16)) for variants in phrase_coverage.values())
    assert len(standalone_wrappers) >= 32


@pytest.mark.parametrize("placement", ["both", "earlier", "latest"])
@pytest.mark.parametrize("framing", ["direct", "step"])
@pytest.mark.parametrize("turn", [0, 1])
def test_audit_rejects_changed_scope_in_either_user_turn(placement, framing, turn):
    """A workflow cue or fixed standalone marker cannot bypass the history audit."""

    contract, payload = apply_s1_overlay(load_contract(), V10_OVERLAY)
    settings = contract["curriculum"]["families"]["S1.3"]
    reasoning = settings["train_user_choice_reasoning"]
    prompts = identity_prompts(payload["prompt_banks"], IDENTITY_PROMPTS, bank_name="identity_s1_3")
    replies = load_choice_replies(settings["train_choice_replies"])
    fact = GeneFact("f", "u", "GENE", ("ENSG00000000001", "ENSG00000000002"), {}, ())
    scenario = Scenario("train", 8, fact, framing, 0, 0, "GENE", chosen_gene_id=fact.candidate_gene_ids[0])
    users = _choice_users(
        scenario, placement, replies[0].format(gene_id=scenario.chosen_gene_id),
        prompts, train_targeted_v10=True,
    )
    row = {"messages": [{"role": "user", "content": user} for user in users] + [{
        "role": "assistant", "thinking": _choice_thinking(
            scenario, placement, reasoning, train_targeted_v10=True,
        ),
    }]}
    row["messages"][turn]["content"] += (
        " A later workflow requires this value." if framing == "direct" else " " + CHOICE_STANDALONE_SCOPE
    )
    with pytest.raises(CorpusBuildError, match="scope cues"):
        _audit_choice_scope(
            [row], [scenario], assignments={scenario.scenario_id: (placement, 0)},
            replies=replies, prompts=prompts, reasoning=reasoning,
            train_targeted_v10=True,
        )
