"""Separate robustness correctness from pair consistency using native S1 scoring."""

from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import pytest

from runtime.world_model.contract import load_contract
from scripts import evaluate_world_model_s1_robustness as robustness
from scripts.run_world_model_s1_exact_generation import (
    EVIDENCE_ID_POLICY, NETWORK_POLICY, deterministic_evidence_id,
)
from scripts.world_model_s1_eval_common import (
    EVALUATION_CONTRACT, PREDICTION_SCHEMA_VERSION, S1EvaluationError,
    final_message, read_json, sha256_file, stable_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/world_model_v4/diagnostics/s1_robustness_v1/manifest.json"
TOKENIZER = ROOT / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/plain_base_tokenizer/tokenizer_manifest.json"


def oracle_predictions(questions, answers):
    """Replay native gold with the exact per-record evidence nonces used at generation."""
    predictions = {}
    for record_index, (rid, question) in enumerate(questions.items()):
        messages = answers[rid]["expected_messages"]
        replacements, call_index = {}, -1
        for message in messages:
            if "tool_call" in message:
                call_index += 1
            if message["role"] == "tool":
                for old in re.findall(r"ev_(?:reg|mem)_[a-z0-9]{6}", json.dumps(message["content"])):
                    replacements[old] = deterministic_evidence_id(
                        channel="registry" if old.startswith("ev_reg") else "graph",
                        record_index=record_index, call_index=call_index, record_id=rid,
                    )
        encoded = json.dumps(messages)
        for old, new in replacements.items():
            encoded = encoded.replace(old, new)
        predictions[rid] = {
            "record_id": rid, "generated_messages": json.loads(encoded), "generations": [],
            "stop_reason": "final", "trajectory_error": None,
            "tool_call_count": call_index + 1, "executed_tool_call_count": call_index + 1,
            "injected_failure_count": sum(m["role"] == "tool" and m["content"]["status"] == "error" for m in messages),
            "network_used_false": True,
        }
    return predictions


@pytest.fixture(scope="module")
def panel():
    """Use the pinned published training-owned panel, never the promotion test rows."""
    metadata, rows, answers = robustness.load_robustness_panel(MANIFEST, sha256_file(MANIFEST))
    return metadata, {row["record_id"]: row for row in rows}, answers, load_contract()


def select_group(panel, axis, *, relation=None, trace=None, with_tools=False):
    """Choose a real pair and retain its complete local generation ordering."""
    metadata, questions, answers, contract = panel
    group = next(group for group in metadata["groups"]
                 if group["axis"] == axis
                 and (relation is None or group["expected_relationship"] == relation)
                 and (trace is None or questions[group["members"][0]["record_id"]]["metadata"]["source_trace"] == trace)
                 and (not with_tools or any("tool_call" in m for m in answers[group["members"][0]["record_id"]]["expected_messages"])))
    ids = {member["record_id"] for member in group["members"]}
    selected = {rid: q for rid, q in questions.items() if rid in ids}
    selected_answers = {rid: answers[rid] for rid in selected}
    return selected, selected_answers, oracle_predictions(selected, selected_answers), [group], contract


def score(selected):
    """Run the production row and relationship checks on a complete chosen group."""
    questions, answers, predictions, groups, contract = selected
    return robustness.score_robustness_records(questions, answers, predictions, groups, contract=contract)


def content(prediction):
    """Find the structured final object without touching free-form reasoning."""
    return final_message(prediction["generated_messages"])["content"]


def test_gold_oracle_all_axes_and_coverage_groups(panel):
    """Every native gold row and declared transformation must score correctly."""
    metadata, questions, answers, contract = panel
    report = robustness.score_robustness_records(
        questions, answers, oracle_predictions(questions, answers), metadata["groups"], contract=contract,
    )
    assert report["overall"]["structured_accuracy_rate"] == 1.0
    assert report["overall"]["both_correct_rate"] == 1.0
    assert report["overall"]["pair_consistency_rate"] == 1.0
    assert report["overall"]["stable_wrong_pair_count"] == 0
    assert report["overall"]["coverage_group_all_correct_count"] == report["overall"]["coverage_group_count"]
    assert set(report["by_axis"]) == set(robustness.RELATION_BY_AXIS)
    assert all(group["consistent"] is None for group in report["groups"] if group["kind"] == "coverage_group")
    assert all(item["structured_accuracy_rate"] == 1.0 for item in report["worst_strata"])


@pytest.mark.parametrize("axis,relation", [
    ("wording", "invariant"), ("evidence_order", "invariant"),
    ("evidence_order", "mention_swap"), ("id_notation_case", "normalized_identity_invariant"),
])
def test_stable_wrong_identity_is_consistent_but_not_correct(panel, axis, relation):
    """Agreement never turns a shared factual mistake into successful accuracy."""
    selected = select_group(panel, axis, relation=relation, trace="S1.4.3")
    for prediction in selected[2].values():
        result = content(prediction)["result"]
        result["same_gene"] = not result["same_gene"]
    report = score(selected)
    assert report["overall"]["structured_accuracy_count"] == 0
    assert report["overall"]["both_correct_count"] == 0
    assert report["overall"]["consistent_pair_count"] == 1
    assert report["overall"]["stable_wrong_pair_count"] == 1


def test_scope_stop_for_both_is_inconsistent(panel):
    """Scope consistency requires the expected stop-to-continue transformation."""
    selected = select_group(panel, "task_scope")
    assert score(selected)["overall"]["consistent_pair_count"] == 1
    for prediction in selected[2].values():
        content(prediction)["continuation_decision"] = "stop"
    report = score(selected)
    assert report["overall"]["structured_accuracy_count"] == 1
    assert report["overall"]["both_correct_count"] == 0
    assert report["overall"]["consistent_pair_count"] == 0


def test_mention_swap_requires_swapped_output_fields(panel):
    """Keeping the first/second answer order after swapping the prompt is an error."""
    selected = select_group(panel, "evidence_order", relation="mention_swap", trace="S1.4.4")
    predictions = list(selected[2].values())
    result = content(predictions[1])["result"]
    result["first"], result["second"] = result["second"], result["first"]
    report = score(selected)
    assert report["overall"]["structured_accuracy_count"] == 1
    assert report["overall"]["consistent_pair_count"] == 0


def test_notation_must_echo_each_actual_input_literal(panel):
    """Case/version normalization cannot hide incorrect first/second input literals."""
    selected = select_group(panel, "id_notation_case")
    predictions = list(selected[2].values())
    for side in ("first", "second"):
        content(predictions[1])["result"][side]["input"] = content(predictions[0])["result"][side]["input"]
    report = score(selected)
    assert report["overall"]["structured_accuracy_count"] == 1
    assert report["overall"]["consistent_pair_count"] == 0


def test_summary_wording_does_not_define_accuracy(panel):
    """Reader-facing prose may vary while structured answers stay correct."""
    selected = select_group(panel, "wording")
    for index, prediction in enumerate(selected[2].values()):
        final = content(prediction)
        final["summary"] = f"A different explanation of this result ({index})."
    report = score(selected)
    assert report["overall"]["structured_accuracy_count"] == 2
    assert report["overall"]["consistent_pair_count"] == 1


def test_comparison_keeps_blocked_reason_presence_without_matching_text(panel):
    """Nonempty blocked explanations compare alike; absence stays a separate field."""
    selected = select_group(panel, "wording")
    original = next(iter(selected[2].values()))["generated_messages"]
    first, second = deepcopy(original), deepcopy(original)
    final_message(first)["content"]["blocked_reason"] = "The requested evidence is unavailable."
    final_message(second)["content"]["blocked_reason"] = "This result cannot be established yet."
    assert robustness._structured_view(first) == robustness._structured_view(second)
    assert robustness._structured_view(first) != robustness._structured_view(original)


def test_wrong_final_evidence_nonce_is_not_normalized_into_consistency(panel):
    """Pair comparison must validate citation binding before removing nonce spelling."""
    selected = select_group(panel, "wording", with_tools=True)
    for prediction in selected[2].values():
        final = content(prediction)
        encoded = json.dumps(final)
        assert re.search(r"ev_(?:reg|mem)_[a-z0-9]{6}", encoded)
        encoded = re.sub(r"ev_(reg|mem)_[a-z0-9]{6}", r"ev_\1_000000", encoded)
        final.clear()
        final.update(json.loads(encoded))
    report = score(selected)
    assert report["overall"]["structured_accuracy_count"] == 0
    assert report["overall"]["consistent_pair_count"] == 0
    assert all(not row["evidence_consistency"] for row in report["records"])


def test_wrong_tool_evidence_nonce_is_a_harness_error(panel):
    """Fabricated tool payload identity must abort instead of becoming a model score."""
    selected = select_group(panel, "wording", with_tools=True)
    prediction = next(iter(selected[2].values()))
    message = next(message for message in prediction["generated_messages"] if message["role"] == "tool")
    message["content"] = json.loads(re.sub(r"ev_(reg|mem)_[a-z0-9]{6}", r"ev_\1_000000", json.dumps(message["content"])))
    with pytest.raises(S1EvaluationError):
        score(selected)


def test_survivor_groups_are_coverage_not_equivalence_pairs(panel):
    """Different source facts receive row accuracy and all-correct coverage only."""
    selected = select_group(panel, "survivor_position")
    prediction = next(iter(selected[2].values()))
    result = content(prediction)["result"]
    result["gene_id"] = next(gene for gene in result["candidate_gene_ids"] if gene != result["gene_id"])
    report = score(selected)
    assert report["overall"]["structured_accuracy_count"] == 1
    assert report["overall"]["pair_count"] == 0
    assert report["overall"]["pair_consistency_rate"] is None
    assert report["overall"]["coverage_group_all_correct_count"] == 0
    assert report["groups"][0]["consistent"] is None


@pytest.mark.parametrize("fault", ["missing", "record_id", "network", "group_missing"])
def test_incomplete_mismatched_or_harness_invalid_records_abort(panel, fault):
    """Only complete, identity-bound, isolated prediction sets can be reported."""
    selected = list(select_group(panel, "wording"))
    predictions = selected[2]
    first = next(iter(predictions))
    if fault == "missing":
        predictions.pop(first)
    elif fault == "record_id":
        predictions[first]["record_id"] = "other-record"
    elif fault == "network":
        predictions[first]["network_used_false"] = False
    else:
        selected[3] = []
    with pytest.raises(S1EvaluationError):
        score(selected)


@pytest.fixture(scope="module")
def artifacts(panel, tmp_path_factory):
    """Write complete offline generation artifacts with the production identity schema."""
    metadata, questions, answers, contract = panel
    work = tmp_path_factory.mktemp("robustness_scoring")
    predictions = oracle_predictions(questions, answers)
    predictions_path = work / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in predictions.values()))
    tokenizer = read_json(TOKENIZER)
    identity = {
        "train_run_id": "scripted-robustness-test", "method_id": "scripted-robustness-test",
        "checkpoint_identity_sha256": "1" * 64, "base_model_identity_sha256": "2" * 64,
        "base_model_artifact_sha256": "3" * 64, "tokenizer_manifest_sha256": tokenizer["manifest_sha256"],
        "tokenizer_artifacts_sha256": "4" * 64,
        "source_index_sha256": contract["source_index"]["content_sha256"], "generation_bundle_sha256": "5" * 64,
    }
    generation_config = {
        "maximum_tool_calls": 2, "evidence_id_policy": EVIDENCE_ID_POLICY,
        "network_policy": NETWORK_POLICY, "enable_thinking": True, "history_policy": "retain_current_turn",
    }
    generation = {
        "schema_version": PREDICTION_SCHEMA_VERSION, "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": metadata["test_panel_id"], "reads_private_answer_keys": False,
        "predictions_sha256": sha256_file(predictions_path), "record_count": len(predictions),
        "generation_config": generation_config, "generation_config_sha256": stable_sha256(generation_config),
        "elapsed_seconds": 0.0, **identity,
    }
    generation["manifest_sha256"] = stable_sha256(generation)
    generation_path = work / "generation_manifest.json"
    generation_path.write_text(json.dumps(generation))
    return dict(
        manifest_path=MANIFEST, manifest_sha256=sha256_file(MANIFEST), predictions_path=predictions_path,
        generation_manifest_path=generation_path, tokenizer_manifest_path=TOKENIZER,
        output_dir=work / "report", expected_identity=identity,
    )


def test_complete_artifacts_write_only_diagnostic_reports(artifacts):
    """The production entry point validates identities and writes no promotion score."""
    metrics = robustness.score_robustness_diagnostic(**artifacts)
    assert metrics["diagnostic_only"] is True and "gate" not in metrics
    assert metrics["overall"]["structured_accuracy_rate"] == 1.0
    assert {path.name for path in artifacts["output_dir"].iterdir()} == {
        "robustness_metrics.json", "robustness_report.json",
    }


@pytest.mark.parametrize("fault", ["missing", "mismatched"])
def test_launch_identity_is_required_and_bound(artifacts, fault):
    """Missing or incorrect launch identities cannot be replaced by artifact claims."""
    kwargs = {**artifacts, "expected_identity": dict(artifacts["expected_identity"])}
    kwargs["expected_identity"]["checkpoint_identity_sha256"] = None if fault == "missing" else "6" * 64
    with pytest.raises(S1EvaluationError):
        robustness.score_robustness_diagnostic(**kwargs)


def test_generation_with_stripped_history_is_rejected(artifacts, tmp_path):
    """A valid outer manifest hash does not authorize the wrong reasoning policy."""
    manifest = read_json(artifacts["generation_manifest_path"])
    manifest["generation_config"]["history_policy"] = "strip"
    manifest["generation_config_sha256"] = stable_sha256(manifest["generation_config"])
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = stable_sha256(manifest)
    path = tmp_path / "generation_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(S1EvaluationError, match="retain"):
        robustness.score_robustness_diagnostic(**{**artifacts, "generation_manifest_path": path})


def test_main_binds_all_nine_environment_identities(monkeypatch, artifacts):
    """CLI launch identities are supplied by the launcher, never inferred from outputs."""
    names = {
        "train_run_id": "S1_TRAIN_RUN_ID", "method_id": "S1_METHOD_ID",
        "checkpoint_identity_sha256": "S1_CHECKPOINT_ARTIFACT_SHA256",
        "base_model_identity_sha256": "MODEL_IDENTITY_SHA256", "base_model_artifact_sha256": "BASE_MODEL_ARTIFACT_SHA256",
        "tokenizer_manifest_sha256": "S1_TOKENIZER_MANIFEST_SHA256", "tokenizer_artifacts_sha256": "S1_TOKENIZER_ARTIFACTS_SHA256",
        "source_index_sha256": "S1_SOURCE_INDEX_SHA256", "generation_bundle_sha256": "S1_GENERATION_BUNDLE_SHA256",
    }
    for key, env in names.items():
        monkeypatch.setenv(env, artifacts["expected_identity"][key])
    arguments = ["evaluate_world_model_s1_robustness.py"]
    for key in ("manifest", "predictions", "generation_manifest", "tokenizer_manifest", "output_dir", "manifest_sha256"):
        artifact_key = key + "_path" if key in {"manifest", "predictions", "generation_manifest", "tokenizer_manifest"} else key
        arguments.extend(["--" + key.replace("_", "-"), str(artifacts[artifact_key])])
    monkeypatch.setattr(sys, "argv", arguments)
    captured = []
    monkeypatch.setattr(robustness, "score_robustness_diagnostic", lambda **kwargs: captured.append(kwargs) or {"overall": {"row_count": 344}})
    assert robustness.main() == 0
    assert captured[0]["expected_identity"] == artifacts["expected_identity"]
    monkeypatch.delenv("S1_CHECKPOINT_ARTIFACT_SHA256")
    assert robustness.main() == 0
    assert captured[1]["expected_identity"]["checkpoint_identity_sha256"] is None
