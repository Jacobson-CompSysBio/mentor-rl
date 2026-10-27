"""Check S1.9's grammar wording, closed schema, and frozen source IDs."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_9_corpus import (
    MODULE_PROMPTS, PARSE_REQUESTS, build_corpus, build_fact_pools, select_scenarios,
    split_row_counts, trace_counts,
)
from runtime.world_model.s1_overlay import (
    S1_V5_OVERLAY_PATH, S1OverlayError, apply_s1_overlay,
)
from runtime.world_model.s1_prompt_banks import (
    _validated_banks, load_prompt_banks, wrapper_prompts,
)
from runtime.world_model.source_index import check_source_index


BASELINE = Path("data/world_model_v4/corpora/s1_v4/s1_9")
LEGACY = Path("data/world_model_v3/corpora/s1/s1_9")
RESULT_KEYS = {
    "rwr_loe": [
        "module_id", "module_family", "multiplex_id", "seed_gene_id",
        "threshold_recipe", "existence_asserted",
    ],
    "mentor_ev": [
        "module_id", "module_family", "dendrogram_id", "clade_id",
        "existence_asserted",
    ],
}


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build the revision against the pinned source snapshot with validation."""

    contract, overlay = apply_s1_overlay(load_contract(), S1_V5_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_9_v5")
    manifest = build_corpus(
        source, root, contract=contract, prompt_banks=overlay["prompt_banks"],
    )
    rows = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    return contract, overlay, source, root, manifest, rows


def test_revision_keeps_source_ids_answers_and_panels(revision):
    """All modules, answers, quotas, and held-out bytes must stay fixed."""

    _, _, _, root, manifest, rows = revision
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()
    original = {
        row["messages"][-1]["content"]["result"]["module_id"]: row
        for row in map(json.loads, (BASELINE / "train.jsonl").read_text().splitlines())
    }
    assert {row["messages"][-1]["content"]["result"]["module_id"] for row in rows} == set(original)
    assert Counter(row["metadata"]["form"] for row in rows) == {1: 1000, 2: 1000}
    for row in rows:
        old = original[row["messages"][-1]["content"]["result"]["module_id"]]
        assert row["metadata"] == old["metadata"]
        assert row["tools"] == old["tools"]
        assert row["messages"][-1]["content"] == old["messages"][-1]["content"]
    previous = json.loads((BASELINE / "manifest.json").read_text())
    for field in ("row_counts", "form_counts", "status_counts", "framing_counts", "continuation_counts", "source_pool_coverage"):
        assert manifest[field] == previous[field]


def test_each_parse_request_occurs_in_both_frames(revision):
    """Request wording must not predict stop or continue by itself."""

    _, _, _, _, _, rows = revision
    coverage = {}
    for row in rows:
        answer = row["messages"][-1]["content"]
        result = answer["result"]
        request = row["messages"][1]["content"].split("\n")[-1].replace(result["module_id"], "{module_id}")
        key = (result["module_family"], request)
        coverage.setdefault(key, Counter())[answer["continuation_decision"]] += 1
    assert set(coverage) == {
        (family, request) for family in RESULT_KEYS for request in PARSE_REQUESTS["train"]
    }
    for counts in coverage.values():
        assert set(counts) == {"stop", "continue"}
        assert abs(counts["stop"] - counts["continue"]) <= 1


def test_reasoning_binds_components_to_exact_fields(revision):
    """Reject the invented field sets observed in all twelve test failures."""

    _, _, _, _, _, rows = revision
    leading_zero_clades = 0
    for row in rows:
        final = row["messages"][-1]
        answer = final["content"]
        result = answer["result"]
        thought = final["thinking"]
        parts = result["module_id"].split(":")
        assert list(result) == RESULT_KEYS[parts[0]]
        assert f'Copy the full input "{result["module_id"]}" into module_id.' in thought
        assert answer["status"] == "resolved"
        assert result["existence_asserted"] is False
        assert "existence_asserted is false" in thought
        assert "Emit exactly " + ", ".join(RESULT_KEYS[parts[0]][:-1]) in thought
        assert f'continuation_decision is {answer["continuation_decision"]}.' in thought
        if parts[0] == "rwr_loe":
            assert result["multiplex_id"] == parts[1]
            assert result["seed_gene_id"] == parts[2].removeprefix("seed_")
            assert result["threshold_recipe"] == parts[3]
            assert "four colon-separated components" in thought
            assert "sets threshold_recipe to geometric_elbow_v1" in thought
        else:
            assert result["dendrogram_id"] == parts[1]
            assert result["clade_id"] == parts[2]
            assert len(parts[2].removeprefix("clade_")) == 6
            leading_zero_clades += parts[2].startswith("clade_0")
            assert "three colon-separated components" in thought
            assert "sets dendrogram_id" in thought
            assert "including leading zeros" in thought
    assert leading_zero_clades > 0


def test_grammar_wording_covers_both_families_and_framings(revision):
    """Grammar wording must not select a family or continuation outcome."""

    _, overlay, _, _, _, rows = revision
    bank = load_prompt_banks(overlay["prompt_banks"])
    audit = _validated_banks(bank)["module"]
    assert audit["wrapper_counts"] == {"direct": 32, "step": 32}
    assert audit["maximum_held_out_jaccard"] < 0.6
    grammar = Counter()
    for row in rows:
        if "grammar" in row["messages"][1]["content"].split("\n")[0]:
            answer = row["messages"][-1]["content"]
            grammar[(answer["result"]["module_family"], answer["continuation_decision"])] += 1
    assert set(grammar) == {
        (family, decision) for family in RESULT_KEYS for decision in ("stop", "continue")
    }
    assert max(grammar.values()) - min(grammar.values()) <= 1
    assert 0 < sum(grammar.values()) < len(rows)


def test_wrapper_count_does_not_change_identifiers_or_requests(revision):
    """A larger wording bank must retain the same selected parse tasks."""

    contract, overlay, source, _, _, _ = revision
    settings = contract["curriculum"]["families"]["S1.9"]
    pools = build_fact_pools(
        check_source_index(source, contract), settings=settings,
        seed=settings["assignment_seed"], contract=contract,
    )
    prompts = wrapper_prompts(overlay["prompt_banks"], MODULE_PROMPTS, bank_name="module")
    expanded = deepcopy(prompts)
    for framing in ("direct", "step"):
        expanded["train"][framing] *= 2
    sizes = split_row_counts(2000, contract["curriculum"]["split_fractions"])
    counts = {split: trace_counts(size, settings) for split, size in sizes.items()}
    selections = [select_scenarios(
        pools, counts, seed=settings["assignment_seed"], settings=settings,
        prompts=bank, framing_shares=contract["curriculum"]["framing"],
    ) for bank in (prompts, expanded)]
    for split in counts:
        signatures = [Counter(
            (case.form, case.module_id, case.module_family, case.split_unit,
             case.framing, case.request_index, case.occurrence)
            for case in selected[split]
        ) for selected in selections]
        assert signatures[0] == signatures[1]


def test_no_overlay_reproduces_published_splits(revision, tmp_path):
    """The training switch must leave every legacy corpus row reproducible."""

    _, _, source, _, _, _ = revision
    build_corpus(source, tmp_path, contract=load_contract())
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (LEGACY / f"{split}.jsonl").read_bytes()


@pytest.mark.parametrize("switch", ["train_module_reasoning", "train_balanced_requests"])
def test_overlay_rejects_disabled_module_switch(tmp_path, switch):
    """Training reasoning is enabled explicitly by the pinned v5 overlay."""

    payload = yaml.safe_load(S1_V5_OVERLAY_PATH.read_text())
    payload["family_overrides"]["S1.9"][switch] = False
    path = tmp_path / "s1_v5.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match=switch):
        apply_s1_overlay(load_contract(), path)
