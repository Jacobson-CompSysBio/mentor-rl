"""The S2 Failures tab data: categories, JSON-path differences, examples, and gate rates."""

import hashlib
import json
from pathlib import Path

import pytest

from dashboard import failures
from dashboard.gates import gate_info

CALL = {"name": "get_edge", "arguments": {"layer_id": "L1", "gene_ids": ["A", "B"]}}


def test_json_diff_names_each_kind_of_difference():
    expected = {"ids": ["A", "B"], "order": [1, 2], "count": 2, "gone": 1}
    got = {"ids": ["A"], "order": [2, 1], "count": 3, "new": 1}
    found = {d["path"]: d for d in failures.json_diff(expected, got, ".result")}
    assert found[".result.ids[]"]["kind"] == "length"
    assert found[".result.ids[]"]["only_expected"] == ["B"]
    assert found[".result.order[]"]["kind"] == "order"
    assert found[".result.count"] == {"path": ".result.count", "kind": "differs", "expected": 2, "got": 3}
    assert found[".result.gone"]["kind"] == "missing" and found[".result.new"]["kind"] == "extra"
    assert failures.json_diff(expected, expected) == []
    assert failures.general_path(".result.items[3].gene_id") == ".result.items[].gene_id"


def record(**fields):
    base = {name: True for name in failures.FIELDS}
    base.update(valid_trajectory=True, trajectory_error=None, parse_error=None, stop_reason="final")
    base.update(fields)
    return base


@pytest.mark.parametrize(("fields", "expected", "generated", "label"), [
    ({"valid_trajectory": False, "trajectory_error": "unexpected_tool_call_or_arguments"}, [], [CALL],
     "invalid: called a tool when none was expected"),
    ({"valid_trajectory": False, "trajectory_error": "unexpected_tool_call_or_arguments"}, [CALL],
     [{**CALL, "arguments": {"layer_id": "L2"}}], "invalid: wrong arguments at call 1"),
    ({"valid_trajectory": False, "trajectory_error": "unexpected_tool_call_or_arguments"}, [CALL], [CALL, CALL],
     "invalid: extra call after the expected calls"),
    ({"valid_trajectory": False, "trajectory_error": "final_answer:the final answer is not strict JSON: "
      "Expecting ',' delimiter: line 1 column 9 (char 8)"}, [], [], "invalid: the final answer is not strict JSON: Expecting ',' delimiter"),
    ({"exact_tool_calls": False}, [CALL], [], "answered before the expected tool calls"),
    ({"result_accuracy": False, "continuation_decision_accuracy": False}, [CALL], [CALL],
     "final answer: result + decision"),
])
def test_each_failed_row_gets_one_category(fields, expected, generated, label):
    assert failures.category(record(**fields), expected, generated) == label


def test_examples_cover_every_category_before_repeating_one():
    rows = [{"category": "a"}] * 5 + [{"category": "b"}, {"category": "c"}]
    assert [row["category"] for row in failures.examples(rows, 4)] == ["a", "b", "c", "a"]


def write_run(root: Path, *, tamper: bool = False) -> Path:
    """A two-row S2 run: one correct row and one with a wrong result."""
    evaluator = root / "eval"
    evaluator.mkdir()
    questions = [{"row_id": row_id, "messages": [{"role": "system", "content": "s"}, {"role": "harness", "content": "h" * 3000},
                                                  {"role": "user", "content": "Find the edge."}]} for row_id in ("ok", "bad")]
    target = {"status": "resolved", "result": {"ids": ["A", "B"]}, "continuation_decision": "stop", "provenance": {},
              "blocked_reason_required": False, "expected_tool_calls": [CALL], "expected_tool_responses": []}
    answers = [{"row_id": row_id, "target": target} for row_id in ("ok", "bad")]
    for name, rows in (("q.jsonl", questions), ("a.jsonl", answers)):
        (evaluator / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    run = root / "run"
    for part in ("report", "generation", "run_contract"):
        (run / part).mkdir(parents=True)
    digest = lambda name: hashlib.sha256((evaluator / name).read_bytes()).hexdigest()
    (run / "run_contract/test_config.json").write_text(json.dumps({"test_panel": {
        "evaluator_root": str(evaluator), "questions_file": "q.jsonl", "answer_key_file": "a.jsonl",
        "questions_sha256": digest("q.jsonl"), "answer_key_sha256": "0" * 64 if tamper else digest("a.jsonl")}}))
    bad = record(result_accuracy=False, structured_accuracy=False, row_id="bad", catalog_case_id="S2.1.1", question_family="S2.1")
    good = record(structured_accuracy=True, row_id="ok", catalog_case_id="S2.1.1", question_family="S2.1")
    (run / "report/report.json").write_text(json.dumps({"records": [good, bad]}))
    counts = {"row_count": 2, "structured_accuracy_count": 1, "result_accuracy_count": 1,
              **{f"{field}_count": 2 for field in failures.FIELDS if field != "result_accuracy"}}
    (run / "report/metrics.json").write_text(json.dumps({"test_panel_id": "p", "gate": {"profile": "report_only_v1"},
                                                         "overall": counts, "families": {"S2.1": counts}, "cases": {"S2.1.1": counts}}))
    final = {"status": "resolved", "result": {"ids": ["A"]}, "continuation_decision": "stop", "provenance": {},
             "blocked_reason": None, "summary": "Edge."}
    generated = [{"role": "assistant", "thinking": "Call it.", "tool_call": CALL},
                 {"role": "tool", "name": "get_edge", "content": {"edges": ["x"] * 2000}},
                 {"role": "assistant", "thinking": "Done.", "content": final}]
    (run / "generation/predictions.jsonl").write_text(json.dumps({"row_id": "bad", "generated_messages": generated}) + "\n")
    return run


def test_collect_reads_one_run_and_cuts_long_text(tmp_path):
    summary, rows = failures.collect(write_run(tmp_path))
    assert len(rows) == 1 and summary["correct"] == 1 and summary["categories"] == {"final answer: result": 1}
    row = rows[0]
    assert row["failed"] == ["result"] and row["diffs"][0]["path"] == ".result.ids[]"
    assert row["messages"][0]["content"].endswith("more characters]")  # the harness block is cut
    assert row["messages"][3]["content"].endswith("more characters]")  # and so is the long tool result
    case = summary["cases"]["S2.1.1"]
    assert case["paths"] == [[".result.ids[] (length)", 1]] and case["examples"][0]["row_id"] == "bad"


def test_collect_refuses_evaluator_files_that_changed(tmp_path):
    with pytest.raises(SystemExit, match="no longer matches"):
        failures.collect(write_run(tmp_path, tamper=True))


def test_gate_info_reads_s2_case_rates(tmp_path):
    run = tmp_path / "eval_runs" / "s2_test" / "method" / "slurm-1"
    (run / "report").mkdir(parents=True)
    counts = {"row_count": 4, "structured_accuracy_count": 3, "structured_accuracy_rate": 0.75, "result_accuracy_rate": 0.5}
    (run / "report/metrics.json").write_text(json.dumps({
        "test_panel_id": "p", "gate": {"profile": "report_only_v1", "passed": None},
        "overall": counts, "families": {"S2.1": counts}, "cases": {"S2.1.1": counts}}))
    gate = gate_info(str(run / "report/metrics.json"))
    assert gate["overall"] == [3, 4] and gate["types"] == {"S2.1.1": [3, 4]}
    assert gate["groups"]["by"] == "question type"
    assert gate["metrics"] == {"structured_accuracy": 0.75, "result_accuracy": 0.5}
    assert gate["run"] == "s2_test" and gate["passed"] is None
