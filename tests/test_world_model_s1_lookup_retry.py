"""Verify bounded recovery and exhaustion through the real S1 trajectory runner."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_lookup_retry import assign_retry_cases, diagnostic_retry_case
from runtime.world_model.s1_overlay import S1_V6_OVERLAY_PATH, S1OverlayError, apply_s1_overlay
from scripts.build_world_model_s1_lookup_retry_panel import build_panel
from scripts.evaluate_world_model_s1 import score_s1_record
from scripts.run_world_model_s1_exact_generation import DeterministicEvidenceRuntime, orchestrate_s1_trajectory
from scripts.world_model_s1_eval_common import ordered_json


@pytest.fixture(scope="module", params=("S1.1", "S1.2"))
def panel(tmp_path_factory, request):
    """Build held-out retry cases once, with the pinned local registry."""

    root = tmp_path_factory.mktemp("lookup_retry_panel")
    manifest = build_panel(root, section=request.param)
    questions = [json.loads(line) for line in (root / "test_questions.jsonl").read_text().splitlines()]
    answers = {row["record_id"]: row for row in map(json.loads, (root / "test_answer_key.jsonl").read_text().splitlines())}
    return root, manifest, questions, answers


def _native_text(message):
    """Serialize a scripted assistant action with the production native syntax."""

    prefix = "<|channel|>analysis<|message|>" + message["thinking"] + "<|end|><|start|>assistant"
    if "tool_call" in message:
        call = message["tool_call"]
        return prefix + " to=functions." + call["name"] + "<|channel|>commentary json<|message|>" + json.dumps(call["arguments"], separators=(",", ":")) + "<|call|>"
    return prefix + "<|channel|>final<|message|>" + ordered_json(message["content"]) + "<|return|>"


def _run(panel, case, *, premature=False, third_call=False, change_argument=False):
    """Drive the production runner using real failures and successful tool evidence."""

    _, _, questions, answers = panel
    question = next(row for row in questions if diagnostic_retry_case(row["record_id"]) == case)
    answer = answers[question["record_id"]]
    contract = load_contract()
    source = Path(contract["source_index"]["output_directory"]) / f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    runtime = DeterministicEvidenceRuntime.from_source_index(source, contract=contract, record_indexes={question["record_id"]: 0})
    actions = [deepcopy(message) for message in answer["expected_messages"] if message["role"] == "assistant"]

    def generate(messages, tools, turn_index):
        action = deepcopy(actions[min(turn_index, len(actions) - 1)])
        if premature and turn_index == 1:
            # Stop before the permitted retry, with an otherwise valid error.
            other = next(row for row in questions if diagnostic_retry_case(row["record_id"]) == "no_retry")
            action = deepcopy(answers[other["record_id"]]["expected_messages"][-1])
        if third_call and turn_index == 2:
            action = deepcopy(actions[0])
        if change_argument and turn_index == 1:
            arguments = action["tool_call"]["arguments"]
            key = next(iter(arguments))
            arguments[key] = "ENSG00000141510" if key == "gene_id" else "TP53"
        if "content" in action and action["content"]["status"] == "resolved":
            # Copy evidence actually emitted for this record and call, since
            # gold evidence nonces intentionally differ from runner nonces.
            actual = messages[-1]["content"]
            action["content"]["provenance"] = actual["provenance"]
            action["content"]["result"] = {key: actual[key] for key in ("gene_id", "gene_symbol")}
        return _native_text(action)

    prediction = orchestrate_s1_trajectory(question, contract=contract, runtime=runtime, generate_turn=generate)
    return prediction, score_s1_record(question, answer, prediction, contract=contract)


@pytest.mark.parametrize("case,failures,calls", [("no_retry", 1, 1), ("recovered", 1, 2), ("exhausted", 2, 2)])
def test_permitted_recovery_and_exhaustion_score_correctly(panel, case, failures, calls):
    """The runner and scorer agree on budgets, final evidence, and fault counts."""

    prediction, result = _run(panel, case)
    assert prediction["stop_reason"] == "final"
    assert prediction["injected_failure_count"] == failures
    assert prediction["executed_tool_call_count"] == calls
    assert result["failure_harness_accuracy"] is True
    assert result["structured_answer_accuracy"] is True
    assert result["exact_behavior"] is True


def test_premature_deferral_is_not_successful_recovery(panel):
    prediction, result = _run(panel, "recovered", premature=True)
    assert prediction["executed_tool_call_count"] == 1
    assert result["structured_answer_accuracy"] is False
    assert result["exact_behavior"] is False


def test_exhausted_budget_cannot_execute_a_third_call(panel):
    prediction, result = _run(panel, "exhausted", third_call=True)
    assert prediction["stop_reason"] == "tool_call_limit"
    assert prediction["executed_tool_call_count"] == 2
    assert prediction["injected_failure_count"] == 2
    assert result["exact_behavior"] is False


def test_retry_must_keep_the_original_arguments(panel):
    prediction, result = _run(panel, "recovered", change_argument=True)
    assert prediction["executed_tool_call_count"] == 2
    assert result["exact_tool_arguments"] is False
    assert result["structured_answer_accuracy"] is False
    assert result["exact_behavior"] is False


def test_diagnostic_is_matched_separate_and_reproducible(panel, tmp_path):
    root, manifest, questions, answers = panel
    assert len(questions) == len(answers) == 150
    assert manifest["case_counts"] == {"no_retry": 50, "recovered": 50, "exhausted": 50}
    assert manifest["test_panel_id"] != "eff174050b8d82ee2702cad2e4efebb8fa0938785e2ec6266640c23363ee7904"
    for index in range(0, 150, 3):
        group = questions[index:index + 3]
        assert group[0]["messages"] == group[1]["messages"] == group[2]["messages"]
        assert len({row["metadata"]["split_unit"] for row in group}) == 1
    section = "S1.2" if questions[0]["metadata"]["form"] == 11 else "S1.1"
    build_panel(tmp_path, section=section)
    for name in ("test_questions.jsonl", "test_answer_key.jsonl", "manifest.json"):
        assert (root / name).read_bytes() == (tmp_path / name).read_bytes()


@pytest.mark.parametrize("corruption", ["pin", "budget", "counts", "reasoning"])
@pytest.mark.parametrize("section", ["S1.1", "S1.2"])
def test_retry_policy_rejects_unpinned_or_unsupported_changes(tmp_path, corruption, section):
    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    pin = payload["family_overrides"][section]["train_lookup_retries"]
    policy = json.loads(Path(pin["path"]).read_text())
    if corruption == "budget":
        policy["max_retries"] = 2
    if corruption == "counts":
        policy["case_counts"]["recovered"] += 1
    if corruption == "reasoning":
        policy["thinking"]["retry"] += " Invent {multiplex_id}."
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    pin.update(path=str(path), sha256="0" * 64 if corruption == "pin" else hashlib.sha256(path.read_bytes()).hexdigest())
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="lookup retry policy"):
        apply_s1_overlay(load_contract(), overlay)


def test_diagnostic_namespace_cannot_change_legacy_fault_plans():
    assert diagnostic_retry_case("wm3_s1_1_1234567890abcdef12345678") is None
    with pytest.raises(ValueError, match="diagnostic record ID"):
        diagnostic_retry_case("wm_s1_lookup_retry_v1_unbounded_1234567890abcdef12345678")


@pytest.mark.parametrize(("component", "form", "tool_name"), [
    ("s1_1", 8, "lookup_human_gene_symbol"),
    ("s1_2", 11, "lookup_human_ensembl_gene_id"),
])
def test_retry_outcomes_do_not_depend_on_wording_or_wrapper_row_ids(component, form, tool_name):
    """Changing a wrapper bank must change wording alone, never fault outcomes."""

    baseline = Path("data/world_model_v4/corpora/s1_v5") / component / "train.jsonl"
    rows = [row for row in map(json.loads, baseline.read_text().splitlines()) if row["metadata"]["form"] == form]
    original = assign_retry_cases(rows, tool_name=tool_name)
    revised = deepcopy(rows)
    for index, row in enumerate(revised):
        row["row_id"] = "different_wrapper_row_" + str(index)
        row["messages"][1]["content"] = "A different request wording."
    changed = assign_retry_cases(list(reversed(revised)), tool_name=tool_name)
    assert [original[row["row_id"]] for row in rows] == [changed[row["row_id"]] for row in revised]


@pytest.mark.parametrize("corruption", ["zero_budget", "no_allowance", "changed_argument", "changed_tool", "third_call", "wrong_mapping"])
def test_semantic_validator_rejects_false_recovery(panel, tmp_path, corruption):
    """The recovery exception must not bypass failure or evidence checks."""

    _, _, questions, answers = panel
    question = next(row for row in questions if diagnostic_retry_case(row["record_id"]) == "recovered")
    row = {
        "row_id": question["record_id"],
        "messages": deepcopy(question["messages"] + answers[question["record_id"]]["expected_messages"]),
        "tools": question["tools"], "metadata": question["metadata"],
    }
    if corruption == "zero_budget":
        row["messages"][3]["content"]["message"] = "registry unreachable; retries_remaining=0"
    elif corruption == "no_allowance":
        row["messages"][3]["content"]["message"] = "registry unreachable"
    elif corruption == "changed_argument":
        arguments = row["messages"][4]["tool_call"]["arguments"]
        key = next(iter(arguments))
        arguments[key] = "ENSG00000141510" if key == "gene_id" else "TP53"
    elif corruption == "changed_tool":
        row["messages"][4]["tool_call"]["name"] = "check_multiplex_membership"
    elif corruption == "third_call":
        row["messages"][-1:-1] = deepcopy(row["messages"][4:6])
    else:
        row["messages"][-1]["content"]["result"]["gene_symbol"] = "TP53"
    path = tmp_path / "bad_recovery.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = subprocess.run(
        [sys.executable, "agents/world_model_v3_curriculum/s1_kit/validate.py", str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "failure keys" in result.stdout
