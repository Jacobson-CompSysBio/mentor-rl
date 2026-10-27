"""Check S1.8's revised name coverage, exact cards, and frozen panels."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_8_corpus import (
    METRIC_CONVENTIONS, NAME_VARIANTS, POLARITIES, PROMPT_TEMPLATES,
    SHARED_ALIASES, build_corpus, select_scenarios, split_row_counts, trace_counts,
)
from runtime.world_model.s1_overlay import (
    S1_V5_OVERLAY_PATH, S1OverlayError, apply_s1_overlay,
)
from runtime.world_model.s1_prompt_banks import (
    PromptBankError, load_prompt_banks, metric_names, validate_metric_names,
    wrapper_prompts,
)


BASELINE = Path("data/world_model_v4/corpora/s1_v4/s1_8")
LEGACY = Path("data/world_model_v3/corpora/s1/s1_8")


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build and validate the revised corpus away from publication."""

    contract, overlay = apply_s1_overlay(load_contract(), S1_V5_OVERLAY_PATH)
    root = tmp_path_factory.mktemp("s1_8_v5")
    manifest = build_corpus(
        root, contract=contract, prompt_banks=overlay["prompt_banks"],
    )
    rows = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    return contract, overlay, root, manifest, rows


def test_revision_keeps_panels_and_allocation(revision):
    """Name examples replace canonical repeats; blocked cases retain their quotas."""

    _, _, root, manifest, rows = revision
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()
    assert Counter(row["metadata"]["form"] for row in rows) == {
        1: 1200, 2: 667, 3: 667, 4: 666, 5: 400, 6: 400,
    }
    assert manifest["name_kind_counts"]["train"] == {
        "canonical": 1600, "metric_variant": 1280, "shared_alias": 320,
        "ambiguous": 400, "unknown": 400,
    }
    original = json.loads((BASELINE / "manifest.json").read_text())
    for field in (
        "row_counts", "form_counts", "status_counts", "framing_counts",
        "continuation_counts", "statement_truth_counts",
    ):
        assert manifest[field] == original[field]


def test_each_name_covers_all_polarities_and_both_frames(revision):
    """Coverage must hold for individual names, including symbolic synonyms."""

    _, overlay, _, _, rows = revision
    names = load_prompt_banks(overlay["prompt_banks"])["metric_name_variants"]
    by_name = {name: metric for metric, values in names.items() for name in values}
    polarities = defaultdict(set)
    lookup_frames = defaultdict(set)
    uses = Counter()
    for row in rows:
        final = row["messages"][-1]["content"]
        result = final["result"]
        name = result["input"]
        if name not in by_name:
            continue
        uses[name] += 1
        assert result["metric"] == by_name[name]
        if row["metadata"]["form"] == 1:
            lookup_frames[name].add(final["continuation_decision"])
            continue
        match = re.search(r'a (lower|higher) value for "[^"]+" means (stronger|weaker|greater|less)', row["messages"][1]["content"])
        assert match
        outcome = "stronger" if match[2] in {"stronger", "greater"} else "weaker"
        polarity = f"{match[1]}_{outcome}"
        polarities[name].add(polarity)
        actual_stronger = (match[1] == "lower") == result["lower_is_stronger"]
        assert result["statement_true"] == (actual_stronger == (outcome == "stronger"))
    assert len(by_name) == len(uses) == 156
    assert all(polarities[name] == set(POLARITIES) for name in by_name)
    assert all(lookup_frames[name] == {"stop", "continue"} for name in by_name)


def test_exact_cards_input_copying_and_workflow_scope(revision):
    """Truth checks use the fixed card; workflow text alone does not select continue."""

    _, _, _, _, rows = revision
    workflow_decisions = set()
    for row in rows:
        final = row["messages"][-1]
        result = final["content"]["result"]
        thought = final["thinking"]
        assert f'Keep the input name exactly as "{result["input"]}".' in thought
        if result["metric"] is not None:
            card = METRIC_CONVENTIONS[result["metric"]]
            assert {field: result[field] for field in card} == card
            assert f'canonical metric {result["metric"]}.' in thought
            assert f'rule to {card["rule"]}.' in thought
        if row["metadata"]["form"] in (2, 3, 4):
            assert "The claim pairs" in thought
            assert f'statement_true is {str(result["statement_true"]).lower()}.' in thought
        decision = final["content"]["continuation_decision"]
        assert f"continuation_decision is {decision}." in thought
        if row["metadata"]["form"] == 4:
            workflow_decisions.add(decision)
            if decision == "stop":
                assert "the enclosing request asks only for this answer" in thought
                assert "only" in row["messages"][1]["content"].split("\n")[0].casefold() or any(
                    scope in row["messages"][1]["content"].split("\n")[0]
                    for scope in ("whole task", "complete task", "single task", "on its own", "by itself")
                )
    assert workflow_decisions == {"stop", "continue"}


def test_revised_name_bank_rejects_leakage_and_duplicates(revision):
    """New names use the stricter boundary without rewriting published banks."""

    _, overlay, _, _, _ = revision
    bank = load_prompt_banks(overlay["prompt_banks"])
    variants = bank["metric_name_variants"]
    assert validate_metric_names(variants, bank["shared_aliases"])["maximum_new_name_jaccard"] < 0.6
    changed = deepcopy(variants)
    changed["dot_similarity"][-1] = NAME_VARIANTS["test"]["dot_similarity"][0]
    with pytest.raises(PromptBankError, match="held-out"):
        validate_metric_names(changed, bank["shared_aliases"])
    changed = deepcopy(variants)
    changed["dot_similarity"][-1] = "RWR profile dot product"
    with pytest.raises(PromptBankError, match="too close"):
        validate_metric_names(changed, bank["shared_aliases"])
    changed = deepcopy(variants)
    changed["dot_similarity"][-1] = changed["score"][-1]
    with pytest.raises(PromptBankError, match="repeats"):
        validate_metric_names(changed, bank["shared_aliases"])


def test_wrapper_count_keeps_content_and_request_choices(revision):
    """Expanding a wrapper bank must change only wording and row identities."""

    contract, overlay, _, _, _ = revision
    settings = contract["curriculum"]["families"]["S1.8"]
    prompts = wrapper_prompts(overlay["prompt_banks"], PROMPT_TEMPLATES, bank_name="metric")
    variants, aliases = metric_names(overlay["prompt_banks"], NAME_VARIANTS, SHARED_ALIASES)
    expanded = deepcopy(prompts)
    for framing in ("direct", "step"):
        expanded["train"][framing] *= 2
    sizes = split_row_counts(4000, contract["curriculum"]["split_fractions"])
    counts = {split: trace_counts(size, settings) for split, size in sizes.items()}
    selections = [select_scenarios(
        counts, seed=settings["assignment_seed"], settings=settings, prompts=bank,
        name_variants=variants, alias_cards=aliases,
        framing_shares=contract["curriculum"]["framing"],
    ) for bank in (prompts, expanded)]
    for split in counts:
        signatures = [Counter(
            (case.form, case.input_name, case.metric, case.name_kind, case.framing,
             case.statement_polarity, case.statement_true, case.candidate_metrics,
             case.request_index, case.occurrence)
            for case in selected[split]
        ) for selected in selections]
        assert signatures[0] == signatures[1]


def test_legacy_builder_reproduces_published_splits(tmp_path):
    """New training switches must not alter any no-overlay corpus row."""

    build_corpus(tmp_path, contract=load_contract())
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (LEGACY / f"{split}.jsonl").read_bytes()


def test_overlay_rejects_disabled_training_switch(tmp_path):
    """Coverage changes must be explicitly enabled by the pinned overlay."""

    payload = yaml.safe_load(S1_V5_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.8"]["train_variant_coverage"] = False
    path = tmp_path / "s1_v5.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="train_variant_coverage"):
        apply_s1_overlay(load_contract(), path)
