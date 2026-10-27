"""Check the v5 S1.6 revision's held-out boundary and training coverage."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_6_corpus import (
    LAYER_PROMPTS,
    build_corpus,
    build_fact_pools,
    form_counts,
    select_scenarios,
    split_row_counts,
)
from runtime.world_model.s1_overlay import S1_V5_OVERLAY_PATH, apply_s1_overlay
from runtime.world_model.s1_prompt_banks import wrapper_prompts
from runtime.world_model.source_index import check_source_index


BASELINE = Path("data/world_model_v4/corpora/s1_v4/s1_6")


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build a validated revision from the pinned source, away from publication."""

    contract, overlay = apply_s1_overlay(load_contract(), S1_V5_OVERLAY_PATH)
    digest = contract["source_index"]["content_sha256"]
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{digest}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_6_v5")
    build_corpus(
        source, root, contract=contract, prompt_banks=overlay["prompt_banks"]
    )
    rows = [
        json.loads(line)
        for line in (root / "train.jsonl").read_text().splitlines()
    ]
    return contract, overlay, source, root, rows


def test_revision_keeps_heldout_rows_and_existing_answer_rules(revision):
    """A training revision must preserve the frozen panels and answer rules."""

    contract, _, _, root, rows = revision
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (
            BASELINE / f"{split}.jsonl"
        ).read_bytes()
    assert len(rows) == 5000
    assert Counter(row["metadata"]["form"] for row in rows) == {
        1: 1250, 2: 1250, 3: 625, 4: 750, 5: 500, 6: 500, 7: 125
    }
    baseline = {
        row["row_id"]: row
        for row in map(json.loads, (BASELINE / "train.jsonl").read_text().splitlines())
    }
    for row in rows:
        if row["metadata"]["form"] == 4:
            continue
        original = baseline[row["row_id"]]["messages"][-1]
        assert row["messages"][-1]["content"] == original["content"]
        if row["metadata"]["form"] == 5:
            assert row["messages"][-1] == original
    assert contract["curriculum"]["families"]["S1.6"]["train_diverse_lists"]


def test_lists_cover_all_training_tags_and_both_response_outcomes(revision):
    """Lists must vary source tags, families, positions, and match outcomes."""

    contract, _, source, _, rows = revision
    settings = contract["curriculum"]["families"]["S1.6"]
    pools = build_fact_pools(
        check_source_index(source, contract), settings=settings,
        seed=settings["assignment_seed"], contract=contract,
    )
    lists = [
        row["messages"][-1]["content"]["result"]
        for row in rows if row["metadata"]["form"] == 4
    ]
    usage = Counter()
    positions = defaultdict(set)
    for result in lists:
        assert set(result) == {"layers", "filter", "matching_layer_ids"}
        assert len(result["layers"]) == 3
        for position, layer in enumerate(result["layers"]):
            usage[layer["layer_id"]] += 1
            positions[layer["layer_family"]].add(position)
    assert set(usage) == {fact.layer_id for fact in pools["train"]["all"]}
    assert all(value == {0, 1, 2} for value in positions.values())
    pen_ids = {
        fact.layer_id for fact in pools["train"]["all"]
        if fact.layer_family in {"bulk_pen", "scpen"}
    }
    assert max(usage[tag] for tag in pen_ids) < len(lists) // 10
    assert sum(not result["matching_layer_ids"] for result in lists) == 375
    assert sum(
        all(layer["layer_family"] == "scpen" for layer in result["layers"])
        for result in lists
    ) >= 200
    assert sum(
        any(layer["source"] == "HumanNetV3" for layer in result["layers"])
        for result in lists
    ) >= 375


def test_reasoning_binds_lists_to_tags_and_explains_decisions(revision):
    """Reasoning must explain the list behavior and the visible framing cue."""

    _, overlay, _, _, rows = revision
    prompts = wrapper_prompts(
        overlay["prompt_banks"], LAYER_PROMPTS, bank_name="layer"
    )
    for direct, step in zip(
        prompts["train"]["direct"], prompts["train"]["step"], strict=True
    ):
        assert step.startswith(direct.split(": {request}")[0])
    for row in rows:
        final = row["messages"][-1]
        form = row["metadata"]["form"]
        if form == 5:
            continue
        if form == 6:
            assert final["content"]["result"]["input"] in final["thinking"]
            assert "spelling and repeated parts stay unchanged" in final["thinking"]
            continue
        assert f"so I {final['content']['continuation_decision']}" in final["thinking"]
        if form == 4:
            result = final["content"]["result"]
            assert all(
                layer["layer_id"] in final["thinking"] for layer in result["layers"]
            )
            assert "without removing any parsed layer" in final["thinking"]
            if not result["matching_layer_ids"]:
                assert (
                    "An empty match list still answers the request" in final["thinking"]
                )


def test_wrapper_count_does_not_change_training_content(revision):
    """More wordings must leave every selected input and filter unchanged."""

    contract, overlay, source, _, _ = revision
    settings = contract["curriculum"]["families"]["S1.6"]
    pools = build_fact_pools(
        check_source_index(source, contract), settings=settings,
        seed=settings["assignment_seed"], contract=contract,
    )
    sizes = split_row_counts(5000, contract["curriculum"]["split_fractions"])
    counts = {split: form_counts(size, settings) for split, size in sizes.items()}
    prompts = wrapper_prompts(
        overlay["prompt_banks"], LAYER_PROMPTS, bank_name="layer"
    )
    expanded = deepcopy(prompts)
    for framing in ("direct", "step"):
        expanded["train"][framing] = (
            *expanded["train"][framing], *expanded["train"][framing]
        )
    selections = [
        select_scenarios(
            pools, counts, seed=settings["assignment_seed"], settings=settings,
            framing_shares=contract["curriculum"]["framing"], prompts=bank,
        )
        for bank in (prompts, expanded)
    ]
    for split in sizes:
        cases = [
            {
                (case.form, case.occurrence): (
                    case.fact, case.framing, case.companions, case.requested_field,
                    case.filter_value, case.filter_outcome, case.malformed_input,
                    case.request_variant,
                )
                for case in selected[split]
            }
            for selected in selections
        ]
        assert cases[0] == cases[1]
