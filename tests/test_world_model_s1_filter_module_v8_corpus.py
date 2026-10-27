"""Check v8 joint coverage against real rows and preserve frozen corpus bytes."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import (
    S1OverlayError, S1_V6_OVERLAY_PATH, S1_V8_OVERLAY_PATH,
    apply_s1_overlay, overlay_pin,
)


@pytest.fixture(scope="module", params=[6, 9])
def built(request, tmp_path_factory):
    """Rebuild each real component with both the old and new overlay."""

    module = importlib.import_module(f"runtime.world_model.s1_{request.param}_corpus")
    roots = {}
    for version, overlay_path in ((6, S1_V6_OVERLAY_PATH), (8, S1_V8_OVERLAY_PATH)):
        contract, payload = apply_s1_overlay(load_contract(), overlay_path)
        source = Path(contract["source_index"]["output_directory"]) / (
            f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
        )
        root = tmp_path_factory.mktemp(f"s1_{request.param}_v{version}")
        module.build_corpus(
            source, root, contract=contract, overlay=overlay_pin(overlay_path, payload),
            prompt_banks=payload["prompt_banks"], validate=False,
        )
        roots[version] = root
    return request.param, roots


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_v6_reproduction_and_v8_heldout_identity(built):
    """A train-only switch must not alter any frozen evaluation example."""

    section, roots = built
    published = Path(f"data/world_model_v4/corpora/s1_v6/s1_{section}")
    for split in ("train", "validation", "test"):
        assert (roots[6] / f"{split}.jsonl").read_bytes() == (published / f"{split}.jsonl").read_bytes()
    for split in ("validation", "test"):
        assert (roots[8] / f"{split}.jsonl").read_bytes() == (published / f"{split}.jsonl").read_bytes()
    assert len(_rows(roots[8] / "train.jsonl")) == len(_rows(roots[6] / "train.jsonl"))


def test_filter_scope_and_wording_cover_independent_outcomes(built):
    """Check the visible examples, including the formerly absent scPEN case."""

    section, roots = built
    if section != 6:
        return
    rows = [row for row in _rows(roots[8] / "train.jsonl") if row["metadata"]["form"] == 4]
    cells, wrappers = defaultdict(Counter), defaultdict(Counter)
    homogeneous, bulk_empty, decisions = Counter(), Counter(), Counter()
    for row in rows:
        final = row["messages"][-1]["content"]
        result, decision = final["result"], final["continuation_decision"]
        layers = result["layers"]
        families = tuple(sorted(layer["layer_family"] for layer in layers))
        key = (result["filter"]["field"], result["filter"]["value"], families,
               len(result["matching_layer_ids"]))
        cells[key][decision] += 1
        wrapper = row["messages"][1]["content"].split("filter three layers by exact ")[0]
        wrappers[wrapper][bool(result["matching_layer_ids"])] += 1
        decisions[decision] += 1
        if families == ("scpen", "scpen", "scpen"):
            homogeneous[decision] += 1
            if result["filter"] == {"field": "layer_family", "value": "bulk_pen"}:
                assert not result["matching_layer_ids"]
                bulk_empty[decision] += 1
        assert all(set(layer) == {"layer_id", "layer_family", "source", "network",
                                 "organ", "region_path", "cell_type_path"} for layer in layers)
    assert decisions == {"stop": 375, "continue": 375}
    assert all(abs(counts["stop"] - counts["continue"]) <= 1 for counts in cells.values())
    assert set(homogeneous) == set(bulk_empty) == {"stop", "continue"}
    assert len(wrappers) == 64
    assert all(set(counts) == {True, False} for counts in wrappers.values())


def test_every_module_wrapper_covers_every_request(built):
    """Keep source IDs fixed while covering all 512 wording pairs per form."""

    section, roots = built
    if section != 9:
        return
    rows = _rows(roots[8] / "train.jsonl")
    requests = defaultdict(set)
    for row in rows:
        result = row["messages"][-1]["content"]["result"]
        wrapper, request = row["messages"][1]["content"].split("\n", 1)
        requests[(row["metadata"]["form"], wrapper)].add(request.replace(result["module_id"], "{module_id}"))
        assert result["existence_asserted"] is False
    assert len(requests) == 128
    assert all(len(variants) == 8 for variants in requests.values())
    for version in (6, 8):
        ids = {row["messages"][-1]["content"]["result"]["module_id"]
               for row in _rows(roots[version] / "train.jsonl")}
        if version == 6:
            old_ids = ids
        else:
            assert ids == old_ids


@pytest.mark.parametrize("section", (6, 9))
def test_doubled_wrapper_bank_preserves_content(section, monkeypatch, tmp_path):
    """More wrappers must leave evidence, framing and request choices unchanged."""

    module = importlib.import_module(f"runtime.world_model.s1_{section}_corpus")
    contract, payload = apply_s1_overlay(load_contract(), S1_V8_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    original = module.select_scenarios
    checked = []

    def content(case):
        """Ignore only the wrapper choice and its derived scenario identifier."""

        record = asdict(case)
        record.pop("template_index")
        record.pop("scenario_id", None)
        return json.dumps(record, sort_keys=True)

    def compare_banks(*args, **kwargs):
        baseline = original(*args, **kwargs)
        expanded_options = deepcopy(kwargs)
        for framing in ("direct", "step"):
            templates = expanded_options["prompts"]["train"][framing]
            expanded_options["prompts"]["train"][framing] = tuple(templates) * 2
        expanded = original(*args, **expanded_options)
        assert any(case.template_index >= 32 for case in expanded["train"])
        for split in ("train", "validation", "test"):
            assert Counter(map(content, baseline[split])) == Counter(map(content, expanded[split]))
            if split != "train":
                assert baseline[split] == expanded[split]
        checked.append(True)
        return baseline

    monkeypatch.setattr(module, "select_scenarios", compare_banks)
    module.build_corpus(
        source, tmp_path, contract=contract, prompt_banks=payload["prompt_banks"],
        overlay=overlay_pin(S1_V8_OVERLAY_PATH, payload), validate=False,
    )
    assert checked == [True]


def test_v8_overlay_requires_explicit_decorrelation(tmp_path):
    """The revision must not silently disable the declared sampling fixes."""

    payload = yaml.safe_load(S1_V8_OVERLAY_PATH.read_text())
    assert all("train_prompt_banks" not in values for values in payload["family_overrides"].values())
    payload["family_overrides"]["S1.6"]["train_decorrelate"] = False
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="train_decorrelate"):
        apply_s1_overlay(load_contract(), path)
