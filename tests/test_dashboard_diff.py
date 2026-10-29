"""The version diff classifies gold traces, counts edits, and checks comparability."""

import json
from pathlib import Path

from dashboard.diff import compare, scan, type_names
from dashboard.gates import comparability


def row(row_id, thinking, status="resolved", decision="stop"):
    return {
        "row_id": row_id,
        "messages": [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "Find the ID."},
            {"role": "assistant", "thinking": thinking,
             "content": {"status": status, "continuation_decision": decision}},
        ],
        "tools": [],
        "metadata": {"source_trace": "S1.1.1", "blocked_reason_code": None},
    }


def version(root: Path, rows, test_sha="same"):
    (root / "s1_1").mkdir(parents=True)
    (root / "s1_1" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = {"validation.jsonl": {"sha256": "same"}, "test.jsonl": {"sha256": test_sha}}
    (root / "manifest.json").write_text(json.dumps({"stage_order": ["s1_1"], "files": files}))
    return {"version": root.name, "corpus_root": str(root)}


def run(old, new):
    stage = {"stage": "S1", "versions": [old, new]}
    scans = {entry["version"]: scan(entry) for entry in stage["versions"]}
    return compare(stage, old, new, scans)


def test_compare_classifies_gold_traces_and_counts_edits(tmp_path):
    old = version(tmp_path / "v1", [
        row("a", "same"),
        row("b", "old words", decision="continue"),
        row("c", "replaced", status="not_found", decision="defer"),
    ])
    new = version(tmp_path / "v2", [
        row("a", "same"),
        row("b", "new words", decision="continue"),
        row("d", "regenerated", status="not_found", decision="defer"),
    ], test_sha="changed")
    record, needed = run(old, new)
    assert record["gold"] == {"identical": 1, "edited": 1, "regenerated": 1}
    assert record["rows"] == {"total": 3, "edited": 1, "regenerated": 1, "removed": 1}
    assert record["types"][0]["fields"] == {"final thinking": 1}
    assert sorted(t["state"] for t in record["traces"]) == ["edited", "identical", "regenerated"]
    assert record["held_out_frozen"] == {"validation.jsonl": True, "test.jsonl": False}
    assert needed == set()


def test_compare_asks_for_older_rows_that_are_not_gold_there(tmp_path):
    # In v1 the gold trace for this outcome is row x, so the edited row b
    # must be fetched from v1 for the page to show its old side.
    old = version(tmp_path / "v1", [row("x", "first"), row("b", "old words")])
    new = version(tmp_path / "v2", [row("b", "new words")])
    record, needed = run(old, new)
    assert [(t["state"], t["old"], t["new"]) for t in record["traces"]] == [("edited", "b", "b")]
    assert needed == {"b"}


def test_single_file_corpus_and_fact_kind_names(tmp_path):
    # S0 has no component folders, and names its types by fact kind and input variant.
    root = tmp_path / "s0"
    root.mkdir()
    s0_row = row("a", "lookup")
    s0_row["metadata"].update(source_trace="S0.3", fact_kind="resolved_symbol", input_variant="casefold")
    (root / "train.jsonl").write_text(json.dumps(s0_row) + "\n")
    (root / "manifest.json").write_text(json.dumps({"files": {}}))
    found = scan({"version": "v4", "corpus_root": str(root)})
    assert list(found["parts"]) == ["a"]
    assert type_names(list(found["gold"].values())) == {"S0.3": "resolved symbol, casefold input"}


def gate(**changes):
    return {"panel": "p", "profile": "structured_95_v1", "history_policy": "retain_current_turn",
            "answer_key": "k", "overall": [1, 2]} | changes


def test_comparability_names_each_difference():
    assert comparability(gate(), gate()) == []
    assert comparability(gate(), gate(profile="tiered_v3", answer_key="other")) == [
        "the gate profiles differ (structured_95_v1, tiered_v3)", "the answer keys differ"]
    assert comparability(gate(), None) == ["a gate report is missing"]
