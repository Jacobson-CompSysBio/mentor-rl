"""Check source-backed v8 identity decorrelation and frozen legacy outputs."""

from collections import Counter, defaultdict
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import re

import pytest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import apply_s1_overlay, overlay_pin


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/world_model_v3/source_indexes/world_model_source_index_08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e.json.gz"
PROMPT_BANKS = json.loads((ROOT / "runtime/world_model/s1_v5_prompt_banks.json").read_text())


@pytest.fixture(scope="module", params=["s1_3", "s1_4"])
def built(request, tmp_path_factory):
    """Build real components once, including their normal semantic audits."""
    family = request.param
    module = importlib.import_module(f"runtime.world_model.{family}_corpus")
    directories = {}
    original_select = module.select_scenarios
    width_checks = []

    def checked_select(*args, **kwargs):
        selected = original_select(*args, **kwargs)
        enabled = kwargs.get("train_decorrelate") or kwargs.get("settings", {}).get("train_decorrelate")
        if enabled:
            prompts = kwargs["prompts"]
            expanded = {**prompts, "train": {
                frame: list(bank) * 2 for frame, bank in prompts["train"].items()
            }}
            other = original_select(*args, **{**kwargs, "prompts": expanded})
            assert Counter(map(_semantic_signature, selected["train"])) == Counter(
                map(_semantic_signature, other["train"])
            )
            width_checks.append(len(selected["train"]))
        return selected

    for version in ("v6", "v8"):
        overlay = ROOT / f"runtime/world_model/s1_{version}.yaml"
        contract, payload = apply_s1_overlay(load_contract(), overlay)
        directory = tmp_path_factory.mktemp(f"{family}_{version}")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, "select_scenarios", checked_select)
            module.build_corpus(
                SOURCE, directory, contract=contract, validate=False,
                overlay=overlay_pin(overlay, payload), prompt_banks=payload["prompt_banks"],
            )
        directories[version] = directory
    return family, directories, width_checks


def _semantic_signature(scenario):
    value = asdict(scenario)
    # The pair's full content is present in both copies; only identity hashes
    # and wrapper positions change when the requested bank gets larger.
    value.pop("template_index")
    value.pop("paired_success_id")
    return json.dumps(value, sort_keys=True)


def test_wrapper_bank_expansion_preserves_training_content(built):
    _, directories, width_checks = built
    assert width_checks == [len(_rows(directories["v8"] / "train.jsonl"))]


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wrapper(row, family):
    user = [m["content"] for m in row["messages"] if m["role"] == "user"][-1]
    bank = PROMPT_BANKS["identity_s1_3" if family == "s1_3" else "alignment"]
    matches = []
    for framing, templates in bank.items():
        for index, template in enumerate(templates):
            prefix, suffix = template.split("{request}")
            if user.startswith(prefix) and user.endswith(suffix):
                matches.append((framing, index))
    assert len(matches) == 1
    return matches[0]


def test_legacy_rebuild_and_v8_heldout_are_byte_identical(built):
    family, directories, _ = built
    frozen = ROOT / "data/world_model_v4/corpora/s1_v6" / family
    for split in ("train", "validation", "test"):
        assert (directories["v6"] / f"{split}.jsonl").read_bytes() == (
            frozen / f"{split}.jsonl"
        ).read_bytes()
        if split != "train":
            assert (directories["v8"] / f"{split}.jsonl").read_bytes() == (
                frozen / f"{split}.jsonl"
            ).read_bytes()
    old, new = (_rows(directories[v] / "train.jsonl") for v in ("v6", "v8"))
    assert Counter(r["metadata"]["form"] for r in old) == Counter(
        r["metadata"]["form"] for r in new
    )
    assert old != new


def test_training_wrappers_and_source_backed_survivors(built):
    family, directories, _ = built
    rows = _rows(directories["v8"] / "train.jsonl")
    for row in rows:
        _wrapper(row, family)
    if family != "s1_3":
        return
    original = {
        r["messages"][-1]["content"]["result"]["gene_symbol"]:
        r["messages"][-1]["content"]["result"]
        for r in _rows(directories["v6"] / "train.jsonl")
        if r["metadata"]["form"] == 5
    }
    cells = defaultdict(set)
    styles = defaultdict(Counter)
    nonfirst = 0
    seen = set()
    for row in rows:
        if row["metadata"]["form"] != 5:
            continue
        result = row["messages"][-1]["content"]["result"]
        source = original[result["gene_symbol"]]
        for field in ("candidate_gene_ids", "candidate_details", "present_gene_ids", "absent_gene_ids", "gene_id"):
            assert result[field] == source[field]
        assert result["candidate_gene_ids"] == sorted(result["candidate_gene_ids"])
        nonfirst_here = result["gene_id"] != result["candidate_gene_ids"][0]
        nonfirst += nonfirst_here
        seen.add(result["gene_symbol"])
        framing, wrapper = _wrapper(row, family)
        location = any(m["role"] == "harness" for m in row["messages"])
        cells[(framing, wrapper, location)].add(nonfirst_here)
        query = next(m["tool_call"]["arguments"]["gene_symbol"] for m in row["messages"] if "tool_call" in m)
        canonical = result["gene_symbol"]
        if query != canonical:
            styles[framing]["lower" if query == canonical.lower() else "mixed"] += 1
    assert len(seen) == len(original)
    assert nonfirst >= 2070 // 4
    assert len(cells) == 128 and all(v == {False, True} for v in cells.values())
    for framing in ("direct", "step"):
        assert set(styles[framing]) == {"lower", "mixed"}
        assert .25 < styles[framing]["lower"] / sum(styles[framing].values()) < .75


def test_alignment_order_and_evidence_positions_cross_with_wrappers(built):
    family, directories, _ = built
    if family != "s1_4":
        return
    order = defaultdict(set)
    position = defaultdict(set)
    source_items = {1: "typed_seed_list", 2: "abstract", 3: "seed_gene_set", 4: "module", 5: "abstract", 6: "typed_seed_list", 7: "typed_seed_list", 8: "query_mygene"}
    for row in _rows(directories["v8"] / "train.jsonl"):
        form = row["metadata"]["form"]
        if form == 9:
            continue  # The component audit separately checks every failure pair.
        key = (form, *_wrapper(row, family))
        result = row["messages"][-1]["content"]["result"]
        first, second = result["first"], result["second"]
        order[key].add(first["item"] == source_items[form])
        harness = "\n".join(m["content"] for m in row["messages"] if m["role"] == "harness")
        module = next((m for m in (first, second) if m["item"] == "module"), None)
        if module:
            ids = json.loads(re.search(r"Module [^\n]+, gene_ids: (\[[^\n]+\])", harness).group(1))
            position[key].add(ids.index(module["input"]))
        else:
            block = json.loads(harness.split("Seed gene set:\n")[1].split("\n")[0])
            seed = next(m for m in (first, second) if m["item"] == "seed_gene_set")
            position[key].add(block["seed_gene_ids"].index(seed["input"]))
    assert len(order) == 8 * 64
    assert all(v == {False, True} for v in order.values())
    assert all(v == {0, 1} for v in position.values())


def test_alignment_notation_is_crossed_and_normalizes_to_real_target(built):
    family, directories, _ = built
    if family != "s1_4":
        return
    cells = defaultdict(set)
    item_styles = defaultdict(set)
    for row in _rows(directories["v8"] / "train.jsonl"):
        form = row["metadata"]["form"]
        if form not in (3, 4):
            continue
        answer = row["messages"][-1]["content"]
        result = answer["result"]
        mentions = [result["first"], result["second"]]
        normalized = [m["input"].split(".")[0].upper() for m in mentions]
        assert normalized == [m["gene_id"] for m in mentions]
        assert (normalized[0] == normalized[1]) == result["same_gene"] == (form == 3)
        notation = "versioned" if any("." in m["input"] for m in mentions) else (
            "case_only" if any(m["input"] != m["input"].upper() for m in mentions) else "plain"
        )
        cells[(form, *_wrapper(row, family))].add(notation)
        item_styles[form].add(tuple(sorted(
            (m["item"], "." in m["input"], m["input"].split(".")[0].islower())
            for m in mentions
        )))
        thought = row["messages"][-1]["thinking"]
        assert ("are equal" if form == 3 else "are different") in thought
    assert len(cells) == 2 * 64
    assert all(v == {"plain", "case_only", "versioned"} for v in cells.values())
    assert len(item_styles[3]) == 4 and item_styles[3] == item_styles[4]
