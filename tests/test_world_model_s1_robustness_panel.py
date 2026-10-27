"""Check source truth, controlled pairs, separation, and reproducible publishing."""
from collections import Counter
from copy import deepcopy
import json

import pytest

from scripts import build_world_model_s1_robustness_panel as panel


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    directory = tmp_path_factory.mktemp("s1_robustness")
    historical = panel.read_json(panel.ROOT / "diagnostics/s1_v8_decorrelation/baseline_pins.json")
    manifest = panel.build_panel(directory / "first")
    panel.build_panel(directory / "second")
    for path, digest in historical.items():
        assert panel.sha256_file(panel.ROOT / path) == digest
    for path, digest in manifest["inputs"].items():
        assert panel.sha256_file(panel.ROOT / path) == digest
    return directory, manifest


def _records(built):
    path = built[0] / "first"
    rows = {r["row_id"]: r for r in panel.read_jsonl(path / "rows.jsonl")}
    groups = panel.read_json(path / "pairs.json")["groups"]
    return path, rows, groups


def test_exact_rebuild_and_separate_native_questions(built):
    directory, manifest = built
    assert manifest["row_count"] == 344 and manifest["group_count"] == 172
    assert manifest["diagnostic_only"] and manifest["source_split"] == "train"
    assert manifest["validation"]["external_semantic_violations"] == 0
    for name in (*manifest["files"], "manifest.json"):
        assert (directory / "first" / name).read_bytes() == (directory / "second" / name).read_bytes()
    path, rows, groups = _records(built)
    questions = panel.read_jsonl(path / "test_questions.jsonl")
    answers = {r["record_id"]: r for r in panel.read_jsonl(path / "test_answer_key.jsonl")}
    assert len(rows) == len(questions) == len(answers) == 344
    assert Counter(g["axis"] for g in groups) == panel.AXIS_COUNTS
    assert Counter(m["record_id"] for g in groups for m in g["members"]) == Counter(rows.keys())
    for question in questions:
        rid = question["record_id"]
        assert question["messages"][-1]["role"] == "user"
        assert not any(m["role"] in {"assistant", "tool"} for m in question["messages"])
        assert rows[rid]["messages"] == question["messages"] + answers[rid]["expected_messages"]
        assert rows[rid]["metadata"] == question["metadata"]
        assert rows[rid]["tools"] == question["tools"]
    assert "source_result" not in json.dumps(groups)


def test_native_pairs_change_only_the_declared_axis(built):
    _, rows, groups = _records(built)
    order_modes = Counter()
    for group in groups:
        panel.validate_pair(group, rows)
        a, b = [rows[m["record_id"]] for m in group["members"]]
        axis = group["axis"]
        users = [[m["content"] for m in row["messages"] if m["role"] == "user"] for row in (a, b)]
        harnesses = [[m["content"] for m in row["messages"] if m["role"] == "harness"] for row in (a, b)]
        if axis in {"wording", "task_scope"}:
            assert harnesses[0] == harnesses[1]
            assert users[0] != users[1]
        elif axis == "evidence_order":
            relation = group["expected_relationship"]
            order_modes[relation] += 1
            if relation == "invariant":
                assert users[0] == users[1] and harnesses[0] != harnesses[1]
            else:
                assert harnesses[0] == harnesses[1] and users[0] != users[1]
                left, right = panel.final(a)["result"], panel.final(b)["result"]
                assert left["first"] == right["second"] and left["second"] == right["first"]
    assert order_modes == {"invariant": 16, "mention_swap": 16}


def test_training_ownership_and_real_survivor_strata(built):
    _, rows, groups = _records(built)
    sources = {component: {r["row_id"]: r for r in panel.read_jsonl(
        panel.ROOT / "data/world_model_v4/corpora/s1_v8" / component / "train.jsonl")}
        for component in panel.COMPONENTS}
    nonfirst_symbols, count_matched, count_unmatched = set(), 0, 0
    for group in groups:
        for member in group["members"]:
            original = sources[member["source_component"]][member["source_row_id"]]
            row = rows[member["record_id"]]
            assert row["metadata"]["source_trace"] == original["metadata"]["source_trace"]
            if member["source_component"] == "s1_3":
                result, source = panel.final(row)["result"], panel.final(original)["result"]
                assert result == source
                calls = [m["tool_call"] for m in row["messages"] if "tool_call" in m]
                assert calls[1]["arguments"]["gene_ids"] == result["candidate_gene_ids"] == sorted(result["candidate_gene_ids"])
            elif member["source_component"] == "s1_4":
                result, source = panel.final(row)["result"], panel.final(original)["result"]
                assert sorted(result[s]["gene_id"] for s in ("first", "second")) == sorted(source[s]["gene_id"] for s in ("first", "second"))
                versions = {m["gene_id"]: m["input"].split(".")[1] for m in (source["first"], source["second"]) if "." in m["input"]}
                for mention in (result["first"], result["second"]):
                    assert mention["input"].split(".")[0].upper() == mention["gene_id"]
                    if "." in mention["input"]:
                        assert mention["input"].split(".")[1] == versions[mention["gene_id"]]
        if group["axis"] == "survivor_position":
            assert group["kind"] == "coverage_group"
            a, b = [rows[m["record_id"]] for m in group["members"]]
            assert not panel._nonfirst(a) and panel._nonfirst(b)
            nonfirst_symbols.add(panel.final(b)["result"]["gene_symbol"])
            if "candidate_count" in group["coverage"]["matching_fields"]:
                count_matched += 1
            else:
                assert "candidate_count" in group["coverage"]["unmatched_fields"]
                count_unmatched += 1
    assert len(nonfirst_symbols) == 6
    assert count_matched and count_unmatched


def test_layer_filter_claim_and_module_coverage(built):
    _, rows, groups = _records(built)
    seen = Counter()
    derived = 0
    for group in groups:
        if group["axis"] not in {"wording", "task_scope"}:
            continue
        row = rows[group["members"][0]["record_id"]]
        trace = row["metadata"]["source_trace"]
        result = panel.final(row)["result"]
        if trace in {"S1.6.4", "S1.7.2"}:
            homogeneous = all(t.startswith("scPEN:") for t in panel._layer_tags(row))
            seen[(trace, homogeneous, bool(result["matching_layer_ids"]))] += 1
            if homogeneous and not result["matching_layer_ids"]:
                assert result.get("layer_family") == "bulk_pen" or result.get("filter") == {"field": "layer_family", "value": "bulk_pen"}
            if "derivation" in group["members"][0]:
                assert trace == "S1.7.2" and homogeneous and result["layer_family"] == "scpen"
                assert len(result["matching_layer_ids"]) == 3
                derived += 1
        elif trace == "S1.7.4":
            assert result["claim_supported"] == (result["claim"] in result["supported_claims"])
            seen[(trace, result["claim_supported"])] += 1
        elif trace.startswith("S1.9"):
            assert not result["existence_asserted"]
            seen[(trace,)] += 1
    for trace in ("S1.6.4", "S1.7.2"):
        for homogeneous in (False, True):
            for positive in (False, True):
                assert seen[(trace, homogeneous, positive)]
    assert derived == 4
    assert seen[("S1.7.4", False)] and seen[("S1.7.4", True)]
    assert seen[("S1.9.1",)] and seen[("S1.9.2",)]


@pytest.mark.parametrize("axis", ["wording", "task_scope", "evidence_order", "id_notation_case", "survivor_position"])
def test_pair_audit_rejects_wrong_gold(built, axis):
    _, rows, groups = _records(built)
    group = next(g for g in groups if g["axis"] == axis)
    changed = deepcopy(rows)
    row = changed[group["members"][1]["record_id"]]
    if axis == "survivor_position":
        panel.final(row)["result"]["gene_id"] = panel.final(row)["result"]["candidate_gene_ids"][0]
    else:
        panel.final(row)["status"] = "not_found"
    with pytest.raises(ValueError):
        panel.validate_pair(group, changed)


def test_existing_corpora_are_not_valid_output_locations():
    with pytest.raises(ValueError, match="cannot overwrite"):
        panel.build_panel(panel.ROOT / "data/world_model_v4/corpora/s1_v8/s1_3")
