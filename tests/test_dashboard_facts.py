"""Gate thresholds, split counts, tool documentation, and source pins for the stage page."""

import json

from dashboard import facts
from dashboard.gates import thresholds


def test_thresholds_read_targets_and_minimums():
    structured = {"details": {"structured_answer_accuracy": {"target": 0.95}}}
    s0 = {"minimum_status_exact": 0.995, "minimum_unrelated": 0.5}
    assert thresholds(structured) == {"structured_answer_accuracy": 0.95}
    assert thresholds(s0) == {"status_exact": 0.995}
    assert thresholds({"minimum_target_rate": 0.9}) == {}


def test_splits_count_types_from_component_manifests(tmp_path):
    corpus = tmp_path / "s1"
    (corpus / "s1_1").mkdir(parents=True)
    (corpus / "s1_1" / "manifest.json").write_text(json.dumps(
        {"form_counts": {"train": {"S1.1.1": 10}, "test": {"S1.1.1": 2}}}))
    (corpus / "manifest.json").write_text(json.dumps({
        "row_counts": {"train": 11, "validation": 0, "test": 2}, "stage_order": ["s1_1"],
        "stage_counts": {"train": {"s0_replay": 1, "s1_1": 10}}}))
    found = facts.splits({"version": "v1", "corpus_root": str(corpus)})
    assert found == {"rows": {"train": 11, "validation": 0, "test": 2}, "replay": 1,
                     "types": {"S1.1.1": {"train": 10, "test": 2}}}
    assert facts.splits({"version": "v0", "corpus_root": None}) is None


def test_tool_docs_join_schemas_with_the_contract(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"tools": {
        "error_result": {"keys": ["status"], "status": "error"},
        "definitions": {
            "lookup": {"args": {"symbol": "str"}, "normalizes": ["ignore_case"], "returns": {"resolved": ["gene_id"]}},
            "later": {"available_from": "S1.5", "args": {"context_id": "str"}},
        }}}))
    schema = {"type": "function", "function": {"name": "lookup", "parameters": {
        "properties": {"symbol": {"type": "string", "minLength": 1}}, "required": ["symbol"]}}}
    docs = facts.tool_docs(str(contract), [schema])
    [tool] = docs["tools"]
    assert tool["parameters"] == [{"name": "symbol", "type": "string", "required": True, "notes": ["at least 1 character"]}]
    assert tool["normalizes"] == ["ignore_case"] and tool["returns"] == {"resolved": ["gene_id"]}
    [later] = docs["other"]
    assert later["note"] == "Available from S1.5."
    assert later["parameters"][0] == {"name": "context_id", "type": "string", "required": True, "notes": []}


def test_sources_follow_a_pinned_source_manifest(tmp_path):
    # The S2 config pins a source manifest, which pins the multiplex layer files.
    sha = "b" * 64
    folder = tmp_path / "checkout" / "runtime" / "world_model"
    folder.mkdir(parents=True)
    (folder / "sources.json").write_text(json.dumps(
        {"descriptor": {"layers": [{"layer_id": "ppi", "source_path": "/lustre/ppi.tsv", "source_sha256": sha}]}}))
    (folder / "config.json").write_text(json.dumps(
        {"source_manifest": {"path": "runtime/world_model/sources.json", "sha256": sha}}))
    found = facts.sources({"contract": str(folder / "config.json")})
    assert [(pin["name"], pin["path"]) for pin in found["pins"]] == [
        ("source_manifest", "runtime/world_model/sources.json"),
        ("source_manifest.descriptor.layers.0", "/lustre/ppi.tsv"),
    ]
    assert found["pins"][1]["facts"] == {"layer_id": "ppi"}
    assert facts.sources({"contract": None}) is None


def test_pins_find_paths_hashes_and_facts():
    sha = "a" * 64
    contract = {
        "registry": {"source_path": "data/genes.gtf.gz", "source_sha256": sha, "gene_count": 10, "nested": {"path": "x", "sha256": sha}},
        "source_pins": {"/lustre/layer.tsv": sha},
        "family": {"output_directory": "data/corpora/s1"},  # a build output, not a source
        "index": {"output_directory": "data/source_indexes", "content_sha256": sha},
    }
    found = {(pin["name"], pin["path"]): pin for pin in facts.pins(contract)}
    assert ("family", "data/corpora/s1") not in found
    assert found[("index", "data/source_indexes")]["sha256"] == sha
    assert found[("registry", "data/genes.gtf.gz")] == {"name": "registry", "path": "data/genes.gtf.gz", "sha256": sha, "facts": {"gene_count": 10}}
    assert found[("registry.nested", "x")]["sha256"] == sha
    assert found[("source_pins", "/lustre/layer.tsv")]["sha256"] == sha
