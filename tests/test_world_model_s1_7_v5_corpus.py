"""Check S1.7 v5's claim meanings, training allocation, and frozen panels."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_7_corpus import (
    CLAIM_PARAPHRASES,
    CLAIM_VOCABULARY,
    FAMILY_PROMPTS,
    build_corpus,
    build_fact_pools,
    select_scenarios,
    trace_counts,
)
from runtime.world_model.s1_overlay import S1_V5_OVERLAY_PATH, apply_s1_overlay
from runtime.world_model.s1_prompt_banks import (
    PromptBankError,
    claim_paraphrases,
    load_prompt_banks,
    validate_claim_paraphrases,
    wrapper_prompts,
)
from runtime.world_model.source_index import check_source_index


BASELINE = Path("data/world_model_v4/corpora/s1_v4/s1_7")


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build the revision with semantic validation, away from publication."""

    contract, overlay = apply_s1_overlay(load_contract(), S1_V5_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_7_v5")
    manifest = build_corpus(
        source, root, contract=contract, prompt_banks=overlay["prompt_banks"]
    )
    rows = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    return contract, overlay, source, root, manifest, rows


def test_revision_keeps_panels_and_unaffected_training_answers(revision):
    """Only the requested training forms and phrases may change."""

    _, _, _, root, manifest, rows = revision
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (
            BASELINE / f"{split}.jsonl"
        ).read_bytes()
    assert Counter(row["metadata"]["form"] for row in rows) == {
        1: 1200, 2: 900, 3: 375, 4: 1500, 5: 300,
        6: 375, 7: 375, 8: 600, 9: 375,
    }
    baseline = {
        row["row_id"]: row
        for row in map(json.loads, (BASELINE / "train.jsonl").read_text().splitlines())
    }
    for row in rows:
        form = row["metadata"]["form"]
        if form in {4, 8}:
            continue
        original = baseline[row["row_id"]]
        assert row["messages"][-1]["content"] == original["messages"][-1]["content"]
        if form != 2:
            assert row == original
    old_manifest = json.loads((BASELINE / "manifest.json").read_text())
    for key in ("claim_check_joint_counts", "list_joint_counts"):
        assert manifest[key] == old_manifest[key]


def test_other_cards_teach_the_closed_mapping_without_heldout_tags(revision):
    """The extra cards must reject invented families and claims explicitly."""

    _, _, _, _, _, rows = revision
    cards = [row for row in rows if row["metadata"]["form"] == 8]
    networks = Counter()
    for row in cards:
        final = row["messages"][-1]
        card = final["content"]["result"]
        assert card["layer_family"] == "other"
        assert card["evidence_type"] is None
        assert card["supported_claims"] == []
        assert card["unsupported_claims"] == list(CLAIM_VOCABULARY)
        assert card["layer_id"] in final["thinking"]
        assert "Every remaining native HumanNetV3 network belongs to other" in final["thinking"]
        networks[card["layer_id"]] += 1
    assert networks == {
        "HumanNetV3:cocitation": 200,
        "HumanNetV3:gene_interaction": 200,
        "HumanNetV3:gene_neighbor": 200,
    }
    assert all(
        "HumanNetV3:database" not in json.dumps(row)
        and "HumanNetV3:phylogenetic" not in json.dumps(row)
        for row in rows
    )


def test_claim_meaning_precedes_support_and_contrasts_regulatory_roles(revision):
    """Every phrase must be used, with opposite TF link and role verdicts."""

    _, overlay, _, _, _, rows = revision
    bank = load_prompt_banks(overlay["prompt_banks"])["claim_paraphrases"]
    seen = defaultdict(set)
    paired = defaultdict(set)
    for row in rows:
        if row["metadata"]["form"] != 4:
            continue
        final = row["messages"][-1]
        card = final["content"]["result"]
        match = re.match(r'The phrase "([^"]+)" names ([a-z_]+)\.', final["thinking"])
        assert match and match[2] == card["claim"]
        assert match[1] in row["messages"][1]["content"]
        seen[card["claim"]].add(match[1])
        if card["layer_family"] == "tf_target" and card["claim"] in {
            "regulatory_link_recorded", "regulation_direction"
        }:
            assert card["claim_supported"] == (card["claim"] == "regulatory_link_recorded")
            key = (
                row["messages"][1]["content"].split("\n")[0],
                card["layer_id"], final["content"]["continuation_decision"],
            )
            paired[key].add(card["claim"])
    assert {claim: len(phrases) for claim, phrases in seen.items()} == {
        claim: 32 for claim in CLAIM_VOCABULARY
    }
    assert all(seen[claim] == set(bank[claim]) for claim in CLAIM_VOCABULARY)
    assert any(len(claims) == 2 for claims in paired.values())


def test_list_reasoning_preserves_exhaustive_partition(revision):
    """Empty matches must retain all three input tags in the other list."""

    _, _, _, _, _, rows = revision
    empty = 0
    for row in rows:
        if row["metadata"]["form"] != 2:
            continue
        final = row["messages"][-1]
        result = final["content"]["result"]
        tags = result["matching_layer_ids"] + result["other_layer_ids"]
        assert len(tags) == len(set(tags)) == 3
        assert all(tag in final["thinking"] for tag in tags)
        if not result["matching_layer_ids"]:
            empty += 1
            assert final["content"]["status"] == "resolved"
            assert len(result["other_layer_ids"]) == 3
    assert empty == 450


def test_claim_bank_rejects_leakage_duplicates_and_wrong_vocabulary(revision):
    """Frozen phrases need the same leakage boundary as wrapper banks."""

    _, overlay, _, _, _, _ = revision
    bank = load_prompt_banks(overlay["prompt_banks"])["claim_paraphrases"]
    audit = validate_claim_paraphrases(bank)
    assert audit["maximum_held_out_jaccard"] < 0.6
    changed = deepcopy(bank)
    changed["regulation_direction"][0] = CLAIM_PARAPHRASES["test"]["regulation_direction"][0]
    with pytest.raises(PromptBankError, match="held-out"):
        validate_claim_paraphrases(changed)
    changed = deepcopy(bank)
    changed["regulation_direction"][0] = changed["regulatory_link_recorded"][0]
    with pytest.raises(PromptBankError, match="repeated"):
        validate_claim_paraphrases(changed)
    changed = deepcopy(bank)
    changed["invented_claim"] = changed.pop("causal_effect")
    with pytest.raises(PromptBankError, match="vocabulary"):
        validate_claim_paraphrases(changed)


def test_wrapper_count_keeps_claim_targets_and_request_choices(revision):
    """Expanding wrappers must change only wording, not selected content."""

    contract, overlay, source, _, _, _ = revision
    settings = contract["curriculum"]["families"]["S1.7"]
    pools = build_fact_pools(
        check_source_index(source, contract), settings=settings,
        seed=settings["assignment_seed"], contract=contract,
    )
    prompts = wrapper_prompts(overlay["prompt_banks"], FAMILY_PROMPTS, bank_name="layer_family")
    phrases = claim_paraphrases(overlay["prompt_banks"], CLAIM_PARAPHRASES)
    expanded = deepcopy(prompts)
    for framing in ("direct", "step"):
        expanded["train"][framing] = (*expanded["train"][framing], *expanded["train"][framing])
    counts = {split: trace_counts(size, settings) for split, size in {
        "train": 6000, "validation": 750, "test": 750
    }.items()}
    selections = [
        select_scenarios(
            pools, counts, seed=settings["assignment_seed"], settings=settings,
            prompts=bank, paraphrases=phrases,
            framing_shares=contract["curriculum"]["framing"],
        ) for bank in (prompts, expanded)
    ]
    for split in counts:
        signatures = [
            {
                (case.form, case.occurrence): (
                    case.fact, case.family, case.framing, case.companions,
                    case.claim, case.claim_supported, case.input_kind,
                    case.request_variant, case.claim_phrase,
                ) for case in cases[split]
            } for cases in selections
        ]
        assert signatures[0] == signatures[1]
