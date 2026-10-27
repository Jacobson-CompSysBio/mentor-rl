"""Verify v10 changes only the three approved training types and their text."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).parent
FAMILIES = (1, 2, 3, 4, 6, 7, 8, 9)
EXPECTED = {"S1.1.6": 800, "S1.3.8": 1000, "S1.8.4": 666}


def sha256(path):
    """Hash one artifact without retaining its contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_training(old_path, new_path):
    """Require stable ordering and exact equality outside approved message text."""
    changed, user_edits, total = Counter(), Counter(), 0
    with old_path.open("rb") as old_stream, new_path.open("rb") as new_stream:
        for old_raw, new_raw in zip(old_stream, new_stream, strict=True):
            total += 1
            old, new = json.loads(old_raw), json.loads(new_raw)
            assert old["row_id"] == new["row_id"], "row IDs or order changed"
            trace = old["metadata"]["source_trace"]
            if trace not in EXPECTED:
                assert old_raw == new_raw, (trace, old["row_id"])
                continue
            assert old_raw != new_raw, ("target row was not revised", old["row_id"])
            allowed = deepcopy(old)
            allowed["messages"][-1]["thinking"] = new["messages"][-1]["thinking"]
            if trace == "S1.3.8":
                for index in (1, 3):
                    if old["messages"][index]["content"] != new["messages"][index]["content"]:
                        assert old["messages"][-1]["content"]["continuation_decision"] == "stop"
                        assert new["messages"][index]["content"] == old["messages"][index]["content"].removeprefix(
                            "The gene-ID selection is the whole request. "
                        )
                        user_edits[trace] += 1
                    allowed["messages"][index]["content"] = new["messages"][index]["content"]
            assert allowed == new, ("unapproved field changed", old["row_id"])
            changed[trace] += 1
    return {"rows": total, "changed_rows": dict(changed),
            "unchanged_rows": total - sum(changed.values()),
            "changed_user_messages": dict(user_edits)}


def main():
    """Publish evidence only after every frozen-file and training comparison passes."""
    pins_path = OUTPUT / "baseline_pins.json"
    pins = json.loads(pins_path.read_text())
    for name, expected in pins.items():
        assert sha256(ROOT / name) == expected, ("historical file changed", name)
    old = ROOT / "data/world_model_v4/corpora/s1_v9"
    new = ROOT / "data/world_model_v4/corpora/s1_v10"
    component_results = {}
    for family in FAMILIES:
        component = f"s1_{family}"
        result = compare_training(old / component / "train.jsonl", new / component / "train.jsonl")
        expected = {trace: count for trace, count in EXPECTED.items() if trace.startswith(f"S1.{family}.")}
        assert result["changed_rows"] == expected, component
        component_results[component] = result
    aggregate = compare_training(old / "train.jsonl", new / "train.jsonl")
    assert aggregate["rows"] == 55276 and aggregate["changed_rows"] == EXPECTED
    assert aggregate["changed_user_messages"] == {"S1.3.8": 750}
    heldout = {}
    for component in ("", *(f"s1_{family}" for family in FAMILIES)):
        for split in ("validation", "test"):
            name = Path(component) / f"{split}.jsonl"
            digest = sha256(new / name)
            assert digest == sha256(old / name), name
            heldout[f"corpus/{name}"] = digest
    for split in ("validation", "test"):
        for kind in ("questions", "answer_key"):
            name = f"{split}_{kind}.jsonl"
            base = ROOT / "data/world_model_v4/eval"
            digest = sha256(base / "s1_v10" / name)
            assert digest == sha256(base / "s1_v9" / name), name
            heldout[f"evaluator/{name}"] = digest
    report = {
        "schema_version": "mentor-rl-s1-v10-corpus-change-audit-v1", "passed": True,
        "baseline_files_verified": len(pins), "baseline_pins_sha256": sha256(pins_path),
        "component_training": component_results, "aggregate_training": aggregate,
        "identical_heldout_files": heldout,
        "protected_fields": "All row IDs/order, metadata, tools, assistant actions, tool results, answers, evidence, and S0 replay remain unchanged. Only final reasoning and 750 fixed standalone prefixes in S1.3.8 user messages change.",
        "audit_script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "corpus_change_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "historical_files": len(pins),
                      "heldout_files": len(heldout), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
