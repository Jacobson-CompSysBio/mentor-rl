"""Exercise membership retries through native generation, scoring, and validation."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_membership_retry import (
    DIAGNOSTIC_PREFIX, assign_membership_retry_cases, audit_membership_retry_row,
    diagnostic_membership_retry_case,
)
from runtime.world_model.s1_overlay import S1_V6_OVERLAY_PATH, S1OverlayError, apply_s1_overlay
from scripts.build_world_model_s1_membership_retry_panel import build_panel
from scripts.evaluate_world_model_s1 import score_s1_record, validate_generated_evidence_policy
from scripts.run_world_model_s1_exact_generation import (
    DeterministicEvidenceRuntime, S1GenerationError, deterministic_evidence_id, maximum_tool_calls,
    orchestrate_s1_trajectory, validate_evidence_namespace,
)
from scripts.world_model_s1_eval_common import ordered_json


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    """Build the diagnostic once and share its real local runtime across tests."""
    root = tmp_path_factory.mktemp("membership_retry_panel")
    manifest = build_panel(root)
    questions = list(map(json.loads, (root / "test_questions.jsonl").read_text().splitlines()))
    answers = {r["record_id"]: r for r in map(json.loads, (root / "test_answer_key.jsonl").read_text().splitlines())}
    contract, _ = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    runtime = DeterministicEvidenceRuntime.from_source_index(source, contract=contract, record_indexes={r["record_id"]: i for i, r in enumerate(questions)})
    return root, manifest, questions, answers, contract, runtime


def _native_text(message):
    """Use the production channel and call syntax for scripted actions."""
    prefix = "<|channel|>analysis<|message|>" + message["thinking"] + "<|end|><|start|>assistant"
    if "tool_call" in message:
        call = message["tool_call"]
        return prefix + " to=functions." + call["name"] + "<|channel|>commentary json<|message|>" + json.dumps(call["arguments"], separators=(",", ":")) + "<|call|>"
    return prefix + "<|channel|>final<|message|>" + ordered_json(message["content"]) + "<|return|>"


def _run(panel, question, corruption=None, *, runtime=None):
    """Script actions while copying only evidence actually returned by the runner."""
    _, _, _, answers, contract, _ = panel
    if runtime is None:
        source = Path(contract["source_index"]["output_directory"]) / f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
        runtime = DeterministicEvidenceRuntime.from_source_index(source, contract=contract, record_indexes={question["record_id"]: 0})
    answer = answers[question["record_id"]]
    actions = [m for m in answer["expected_messages"] if m["role"] == "assistant"]

    def generate(messages, tools, turn_index):
        action = deepcopy(actions[min(turn_index, len(actions) - 1)])
        if corruption == "extra_call" and turn_index == len(actions) - 1:
            action = deepcopy(actions[1])
        if corruption in ("changed_order", "changed_multiplex", "repeat_lookup") and turn_index == 2:
            if corruption == "repeat_lookup":
                action = deepcopy(actions[0])
            elif corruption == "changed_order":
                action["tool_call"]["arguments"]["gene_ids"].reverse()
            else:
                action["tool_call"]["arguments"]["multiplex_id"] = "invented_multiplex_v1"
        if "content" in action:
            provenance = {"registry": None, "graph": None}
            for message in messages:
                if message["role"] == "tool" and message["content"].get("provenance"):
                    for channel, block in message["content"]["provenance"].items():
                        if block:
                            provenance[channel] = block
            action["content"]["provenance"] = provenance
        return _native_text(action)

    prediction = orchestrate_s1_trajectory(question, contract=contract, runtime=runtime, generate_turn=generate)
    return prediction, score_s1_record(question, answer, prediction, contract=contract)


def test_all_111_native_trajectories_and_evidence(panel):
    """All three schedules and all three recovery outcomes pass actual runner/scorer checks."""
    _, manifest, questions, _, contract, _ = panel
    assert manifest["row_count"] == 111 and manifest["split_unit_count"] == 27
    assert manifest["recovery_outcomes"] == {"5": 28, "6": 3, "7": 6}
    validate_evidence_namespace(questions)
    predictions = {}
    for question in questions:
        case = diagnostic_membership_retry_case(question["record_id"])
        prediction, result = _run(panel, question, runtime=panel[5])
        assert prediction["stop_reason"] == "final", prediction
        assert prediction["executed_tool_call_count"] == (2 if case == "no_retry" else 3)
        assert prediction["injected_failure_count"] == (2 if case == "exhausted" else 1)
        assert result["structured_answer_accuracy"] is True, (question["record_id"], result)
        assert result["failure_harness_accuracy"] is True
        assert result["exact_behavior"] is True, (question["record_id"], result)
        predictions[question["record_id"]] = prediction
    validate_generated_evidence_policy({r["record_id"]: r for r in questions}, predictions, contract)


@pytest.mark.parametrize("case", ["no_retry", "recovered", "exhausted"])
def test_no_additional_calls_after_terminal_outcome(panel, case):
    question = next(r for r in panel[2] if diagnostic_membership_retry_case(r["record_id"]) == case)
    prediction, result = _run(panel, question, "extra_call")
    assert prediction["stop_reason"] == "tool_call_limit"
    assert prediction["executed_tool_call_count"] == (2 if case == "no_retry" else 3)
    assert result["exact_behavior"] is False


@pytest.mark.parametrize("corruption", ["changed_order", "changed_multiplex", "repeat_lookup"])
def test_retry_keeps_full_original_membership_arguments(panel, corruption):
    question = next(r for r in panel[2] if diagnostic_membership_retry_case(r["record_id"]) == "recovered")
    _, result = _run(panel, question, corruption)
    assert result["exact_behavior"] is False
    assert result["exact_tool_arguments"] is False


def test_frozen_prompts_and_deterministic_panel(panel, tmp_path):
    root, manifest, questions, _, _, _ = panel
    frozen = [r for r in map(json.loads, Path("data/world_model_v4/corpora/s1_v5/s1_3/test.jsonl").read_text().splitlines()) if r["metadata"]["form"] == 10]
    for index, original in enumerate(frozen):
        prompt = original["messages"][:next(i for i, m in enumerate(original["messages"]) if m["role"] == "assistant")]
        for question in questions[index * 3:index * 3 + 3]:
            assert question["messages"] == prompt
    assert manifest["case_counts"] == {c: 37 for c in ("no_retry", "recovered", "exhausted")}
    build_panel(tmp_path)
    for name in ("test_questions.jsonl", "test_answer_key.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


def _row(panel, case="recovered"):
    question = next(r for r in panel[2] if diagnostic_membership_retry_case(r["record_id"]) == case)
    return deepcopy({"row_id": question["record_id"], "messages": question["messages"] + panel[3][question["record_id"]]["expected_messages"], "tools": question["tools"], "metadata": question["metadata"]})


@pytest.mark.parametrize("corruption", ["zero_budget", "no_allowance", "changed_order", "changed_multiplex", "changed_tool", "extra_call", "stale_error", "lost_registry", "invented_graph", "wrong_exhaustion", "blocked_object"])
def test_semantic_validator_rejects_invalid_retry(panel, tmp_path, corruption):
    row = _row(panel, "exhausted" if corruption == "wrong_exhaustion" else "recovered")
    start = next(i for i, m in enumerate(row["messages"]) if m["role"] == "assistant")
    suffix = row["messages"][start:]
    if corruption in ("zero_budget", "no_allowance"):
        suffix[3]["content"]["message"] = "graph service unreachable" + ("; retries_remaining=0" if corruption == "zero_budget" else "")
    elif corruption == "changed_order":
        suffix[4]["tool_call"]["arguments"]["gene_ids"].reverse()
    elif corruption == "changed_multiplex":
        suffix[4]["tool_call"]["arguments"]["multiplex_id"] = "invented_multiplex_v1"
    elif corruption == "changed_tool":
        suffix[4]["tool_call"] = deepcopy(suffix[0]["tool_call"])
    elif corruption == "extra_call":
        row["messages"][-1:-1] = deepcopy(suffix[4:6])
    elif corruption == "stale_error":
        suffix[-1]["content"]["result"]["failed_tool"] = "check_multiplex_membership"
    elif corruption == "lost_registry":
        suffix[-1]["content"]["provenance"]["registry"] = None
    elif corruption == "invented_graph":
        suffix[-1]["content"]["provenance"]["graph"]["evidence_id"] = "ev_mem_zzzzzz"
    elif corruption == "wrong_exhaustion":
        suffix[5]["content"]["message"] = "graph service unreachable; retries_remaining=1"
    else:
        suffix[-1]["content"]["blocked_reason"] = {"message": "blocked"}
    case = "exhausted" if corruption == "wrong_exhaustion" else "recovered"
    with pytest.raises(ValueError):
        audit_membership_retry_row(row, case)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = subprocess.run([sys.executable, "agents/world_model_v3_curriculum/s1_kit/validate.py", "--overlay", str(S1_V6_OVERLAY_PATH), str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "membership retry budget and evidence" in result.stdout


def test_complete_diagnostic_passes_semantic_validator(panel, tmp_path):
    rows = [{"row_id": q["record_id"], "messages": q["messages"] + panel[3][q["record_id"]]["expected_messages"], "tools": q["tools"], "metadata": q["metadata"]} for q in panel[2]]
    path = tmp_path / "valid.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = subprocess.run([sys.executable, "agents/world_model_v3_curriculum/s1_kit/validate.py", "--overlay", str(S1_V6_OVERLAY_PATH), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("corruption", ["pin", "budget", "counts", "binding"])
def test_policy_rejects_unapproved_changes(tmp_path, corruption):
    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    pin = payload["family_overrides"]["S1.3"]["train_membership_retries"]
    policy = json.loads(Path(pin["path"]).read_text())
    if corruption == "budget":
        policy["max_retries"] = 2
    elif corruption == "counts":
        policy["case_counts"]["recovered"] = 151
    elif corruption == "binding":
        policy["thinking"]["retry"] += " Use {invented}."
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    pin.update(path=str(path), sha256="0" * 64 if corruption == "pin" else hashlib.sha256(path.read_bytes()).hexdigest())
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="membership retry policy"):
        apply_s1_overlay(load_contract(), overlay)


def test_legacy_limits_nonces_and_new_namespace_are_disjoint():
    legacy = "wm3_s1_3_" + "a" * 24
    diagnostic = DIAGNOSTIC_PREFIX + "recovered_" + "a" * 24
    assert maximum_tool_calls(legacy) == 2 and maximum_tool_calls(diagnostic) == 3
    assert deterministic_evidence_id(channel="registry", record_index=0, call_index=0, record_id=legacy) == "ev_reg_zzzzzz"
    assert deterministic_evidence_id(channel="graph", record_index=0, call_index=1, record_id=legacy) == "ev_mem_zzzzzy"
    with pytest.raises(S1GenerationError):
        deterministic_evidence_id(channel="graph", record_index=0, call_index=2, record_id=legacy)
    for channel in ("registry", "graph"):
        old = {deterministic_evidence_id(channel=channel, record_index=i, call_index=j) for i in range(111) for j in range(2)}
        new = {deterministic_evidence_id(channel=channel, record_index=i, call_index=j, record_id=diagnostic) for i in range(111) for j in range(3)}
        assert len(new) == 333 and not old & new


def test_assignment_ignores_wrapper_indices_row_identity_and_input_order():
    """The fault schedule follows stable source identities and occurrences."""
    from runtime.world_model.s1_3_corpus import GeneFact, Scenario

    ids = ("ENSG00000000001", "ENSG00000000002")
    scenarios = []
    for outcome, present in enumerate((ids[:1], ids, ())):
        fact = GeneFact(f"fact{outcome}", f"unit{outcome}", f"EXAMPLE{outcome}", ids, {}, present)
        scenarios.extend(Scenario("train", 10, fact, "direct", index % 32, index, fact.gene_symbol) for index in range(40))
    original = assign_membership_retry_cases(scenarios)
    changed = [replace(s, template_index=(s.template_index + 17) % 32) for s in scenarios]
    revised = assign_membership_retry_cases(list(reversed(changed)))
    assert [original[s.scenario_id] for s in scenarios] == [revised[s.scenario_id] for s in changed]
    assert Counter(original.values()) == {"no_retry": 60, "recovered": 30, "exhausted": 30}


def test_mixed_batch_runs_fourth_assistant_turn_only_when_needed(panel, monkeypatch):
    """A shorter record can finish while a permitted retry still needs its final turn."""
    from scripts import run_world_model_s1_exact_generation as runner

    questions = panel[2][:3]
    contract = panel[4]
    source = Path(contract["source_index"]["output_directory"]) / f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    runtime = DeterministicEvidenceRuntime.from_source_index(source, contract=contract, record_indexes={q["record_id"]: i for i, q in enumerate(questions)})
    monkeypatch.setattr(runner, "_render_prompt", lambda state, **kwargs: json.dumps(state.prompt_messages))
    monkeypatch.setattr(runner, "_prompt_token_count", lambda tokenizer, prompt: 10)
    batches = []

    def generate(**kwargs):
        batches.append(kwargs["record_ids"])
        index = int(kwargs["stage"].rsplit("_", 1)[-1]) - 1
        generated = []
        for record_id, prompt in zip(kwargs["record_ids"], kwargs["prompts"], strict=True):
            actions = [m for m in panel[3][record_id]["expected_messages"] if m["role"] == "assistant"]
            action = deepcopy(actions[index])
            if "content" in action:
                evidence = {"registry": None, "graph": None}
                for message in json.loads(prompt):
                    if message["role"] == "tool":
                        payload = message["content"]
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        for channel, block in (payload.get("provenance") or {}).items():
                            if block:
                                evidence[channel] = block
                action["content"]["provenance"] = evidence
            generated.append({"raw_generation": _native_text(action)})
        return generated

    monkeypatch.setattr(runner, "generate_prompt_batch_with_oom_backoff", generate)
    predictions = runner.generate_trajectory_batch(
        questions, model=None, tokenizer=None, device=None, torch_module=None,
        runtime=runtime, contract=contract, max_new_tokens=4096, max_total_tokens=8192,
        max_tool_call_tokens=1024, max_final_prompt_tokens=4096,
        reasoning_effort="low", shard_index=0,
    )
    assert [len(batch) for batch in batches] == [3, 3, 3, 2]
    for question, prediction in zip(questions, predictions, strict=True):
        assert prediction["stop_reason"] == "final"
        assert score_s1_record(question, panel[3][question["record_id"]], prediction, contract=contract)["exact_behavior"]
