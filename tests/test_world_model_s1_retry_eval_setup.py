"""Keep retry launch integration exact, answer-free, and separate from promotion."""

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from runtime.world_model.contract import load_contract
from scripts.build_world_model_s1_generation_bundle import build_generation_bundle
from scripts.run_world_model_s1_exact_generation import (
    EVIDENCE_ID_POLICY, NETWORK_POLICY, S1GenerationError, deterministic_evidence_id,
    load_generation_bundle, maximum_tool_calls,
)
from scripts.world_model_s1_eval_common import (
    EVALUATION_CONTRACT, PREDICTION_SCHEMA_VERSION, S1EvaluationError,
    read_json, sha256_file, stable_sha256,
)
from scripts.world_model_s1_retry_eval import (
    PANELS, load_retry_panel, score_retry_diagnostic, validate_retry_config,
)

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/plain_base_tokenizer/tokenizer_manifest.json"


@pytest.fixture(scope="module", params=list(PANELS))
def diagnostic(request, tmp_path_factory):
    """Load each real frozen panel and build its production generation bundle."""

    kind = request.param
    path = ROOT / "data/world_model_v4/eval/s1_v6" / kind / "manifest.json"
    panel, questions, answers = load_retry_panel(path, sha256_file(path))
    contract = load_contract()
    source_sha = contract["source_index"]["content_sha256"]
    source = ROOT / contract["source_index"]["output_directory"] / f"world_model_source_index_{source_sha}.json.gz"
    work = tmp_path_factory.mktemp(kind)
    bundle = build_generation_bundle(
        evaluator_manifest_path=path, evaluator_manifest_sha256=sha256_file(path),
        source_index_path=source, source_index_sha256=source_sha, output_dir=work / "bundle",
    )
    return kind, path, panel, questions, answers, work, bundle


def test_native_loader_accepts_only_the_separate_diagnostic(diagnostic):
    kind, _, panel, questions, _, work, bundle = diagnostic
    loaded, rows, _ = load_generation_bundle(work / "bundle")
    assert loaded == bundle and rows == questions
    assert loaded["diagnostic_only"] is True and loaded["diagnostic_kind"] == kind
    assert loaded["failure_harness"] == {
        "policy_source": f"runtime/world_model/s1_v6_{kind}.json",
        "selector": "diagnostic_record_id_retry_case",
        "metadata_visible_to_model": False,
    }
    assert loaded["record_count"] == PANELS[kind][0]
    assert len(loaded["family_counts"]) == 1
    assert loaded["reads_private_answer_keys"] is False
    assert not (work / "bundle/test_answer_key.jsonl").exists()
    assert panel["test_panel_id"] != "eff174050b8d82ee2702cad2e4efebb8fa0938785e2ec6266640c23363ee7904"
    assert max(maximum_tool_calls(q["record_id"]) for q in rows) == (3 if kind == "membership_retries" else 2)


def test_public_loader_never_reads_answer_keys(diagnostic, monkeypatch):
    _, path, _, _, _, _, _ = diagnostic
    import scripts.world_model_s1_retry_eval as module

    original = module.read_jsonl
    def public_read(file):
        assert Path(file).name != "test_answer_key.jsonl"
        return original(file)
    monkeypatch.setattr(module, "read_jsonl", public_read)
    _, _, answers = load_retry_panel(path, sha256_file(path), read_answers=False)
    assert answers == {}


@pytest.mark.parametrize("mutation", ["marker", "kind", "promotion"])
def test_diagnostic_cannot_relax_promotion_family_coverage(diagnostic, mutation, tmp_path):
    _, _, _, _, _, work, bundle = diagnostic
    revised = deepcopy(bundle)
    if mutation == "marker":
        revised["diagnostic_only"] = False
    elif mutation == "kind":
        revised["diagnostic_kind"] = "promotion"
    else:
        del revised["diagnostic_only"]
        del revised["diagnostic_kind"]
    revised.pop("bundle_sha256")
    revised["bundle_sha256"] = stable_sha256(revised)
    (tmp_path / "manifest.json").write_text(json.dumps(revised))
    (tmp_path / "questions.jsonl").write_bytes((work / "bundle/questions.jsonl").read_bytes())
    with pytest.raises(S1GenerationError):
        load_generation_bundle(tmp_path)


def _config(kind, path):
    """Construct the diagnostic-specific config fields independently of checkpoint data."""

    return {
        "diagnostic_panel": {"kind": kind, "manifest": path.relative_to(ROOT).as_posix(), "manifest_sha256": sha256_file(path)},
        "gate": {"profile": "diagnostic_only", "require_network_used_false": True},
        "tracking": {"resume_training_run": False},
        "generation": {"maximum_tool_calls": 3 if kind == "membership_retries" else 2},
    }


def test_diagnostic_preflight_checks_private_answers_and_isolation(diagnostic):
    kind, path, panel, _, _, _, _ = diagnostic
    config = _config(kind, path)
    verified, _, _ = validate_retry_config(config, repo_root=ROOT, contract=load_contract())
    assert verified == panel
    for changed in ("gate", "tracking", "calls", "pin"):
        bad = deepcopy(config)
        if changed == "gate":
            bad["gate"]["profile"] = "tiered_v3"
        elif changed == "tracking":
            bad["tracking"]["resume_training_run"] = True
        elif changed == "calls":
            bad["generation"]["maximum_tool_calls"] = 4
        else:
            bad["diagnostic_panel"]["manifest_sha256"] = "0" * 64
        with pytest.raises(S1EvaluationError):
            validate_retry_config(bad, repo_root=ROOT, contract=load_contract())


def test_pinned_diagnostic_config_exports_without_promotion_gates(diagnostic, monkeypatch):
    """Validate final pins with the future checkpoint directory and receipt stubbed."""

    from scripts import validate_world_model_s1_eval_contract as validator

    kind, path, panel, _, _, _, _ = diagnostic
    config_path = ROOT / "config" / f"world_model_s1_120b_32n_{kind}_v6_32n.json"
    config = read_json(config_path)
    promotion = read_json(ROOT / "config/world_model_s1_120b_32n_test_v6_512n.json")
    assert all(config[key] == promotion[key] for key in ("training", "model", "corpus", "test_panel", "source_index"))
    assert config["methods"][0]["output_root"] != promotion["methods"][0]["output_root"]
    for key, value in promotion["generation"].items():
        if key not in {"num_nodes", "maximum_tool_calls"}:
            assert config["generation"][key] == value
    checked = []
    def checkpoint_stub(**kwargs):
        checked.append(kwargs)
        return "scripted-test", "1" * 64
    monkeypatch.setattr(validator, "_validate_checkpoint", checkpoint_stub)
    resolve = validator.resolve_repo_path
    def future_checkpoint_path(repo_root, value, label, **kwargs):
        if label in {"checkpoint root", "training output root"}:
            kwargs["must_exist"] = False
        return resolve(repo_root, value, label, **kwargs)
    monkeypatch.setattr(validator, "resolve_repo_path", future_checkpoint_path)
    values = validator.validate_and_export(
        repo_root=ROOT, config_path=config_path, method_id=config["methods"][0]["method_id"],
        checkpoint_path=ROOT / config["training"]["checkpoint_root"] / "scripted-test",
    )
    evaluator = read_json(ROOT / promotion["test_panel"]["evaluator_root"] / "manifest.json")
    assert checked[0]["validation_answer_key_sha256"] == evaluator["validation"]["answer_key_sha256"]
    assert values["S1_DIAGNOSTIC_ONLY"] == "1"
    assert values["S1_TEST_PANEL_ID"] == panel["test_panel_id"]
    assert values["S1_TEST_ROWS"] == str(PANELS[kind][0])
    assert values["S1_EVAL_NUM_NODES"] == "32"
    assert Path(values["S1_EVALUATOR_MANIFEST"]) == path
    assert "S1_GATE_PROFILE" not in values


def test_scripted_complete_artifacts_score_without_a_promotion_gate(diagnostic):
    """Exercise loading, nonce validation, every diagnostic row, and separate report writing."""

    kind, path, panel, questions, answers, work, bundle = diagnostic
    predictions = []
    for record_index, question in enumerate(questions):
        rid = question["record_id"]
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
        predictions.append({
            "record_id": rid, "generated_messages": json.loads(encoded), "generations": [],
            "stop_reason": "final", "trajectory_error": None,
            "tool_call_count": call_index + 1, "executed_tool_call_count": call_index + 1,
            "injected_failure_count": sum(m["role"] == "tool" and m["content"]["status"] == "error" for m in messages),
            "network_used_false": True,
        })
    predictions_path = work / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    tokenizer = read_json(TOKENIZER)
    identity = {
        "train_run_id": "scripted-test", "method_id": "scripted-test",
        "checkpoint_identity_sha256": "1" * 64, "base_model_identity_sha256": "2" * 64,
        "base_model_artifact_sha256": "3" * 64, "tokenizer_manifest_sha256": tokenizer["manifest_sha256"],
        "tokenizer_artifacts_sha256": "4" * 64, "source_index_sha256": bundle["source_index"]["sha256"],
        "generation_bundle_sha256": bundle["bundle_sha256"],
    }
    generation_config = {
        "maximum_tool_calls": 3 if kind == "membership_retries" else 2,
        "evidence_id_policy": EVIDENCE_ID_POLICY, "network_policy": NETWORK_POLICY, "enable_thinking": True,
    }
    generation = {
        "schema_version": PREDICTION_SCHEMA_VERSION, "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": panel["test_panel_id"], "reads_private_answer_keys": False,
        "predictions_sha256": sha256_file(predictions_path), "record_count": len(predictions),
        "generation_config": generation_config, "generation_config_sha256": stable_sha256(generation_config),
        "elapsed_seconds": 0.0, **identity,
    }
    generation["manifest_sha256"] = stable_sha256(generation)
    generation_path = work / "generation_manifest.json"
    generation_path.write_text(json.dumps(generation))
    kwargs = dict(
        manifest_path=path, manifest_sha256=sha256_file(path), predictions_path=predictions_path,
        generation_manifest_path=generation_path, tokenizer_manifest_path=TOKENIZER,
        output_dir=work / "report", expected_identity=identity,
    )
    metrics = score_retry_diagnostic(**kwargs)
    assert metrics["diagnostic_only"] is True
    assert "gate" not in metrics
    assert metrics["overall"]["structured_answer_accuracy_rate"] == 1.0
    assert set(metrics["by_case"]) == {"no_retry", "recovered", "exhausted"}
    assert all(s["row_count"] == len(predictions) // 3 for s in metrics["by_case"].values())
    assert (work / "report/retry_metrics.json").exists()
    assert not (work / "report/test_metrics.json").exists()
    kwargs["expected_identity"] = {**identity, "checkpoint_identity_sha256": "5" * 64}
    with pytest.raises(S1EvaluationError, match="checkpoint_identity"):
        score_retry_diagnostic(**kwargs)


def test_v6_promotion_bundle_preserves_the_frozen_panel(tmp_path):
    """The ordinary v6 launch keeps all eight families and the unchanged panel."""

    from runtime.world_model.s1_overlay import S1_OVERLAY_V5_SCHEMA_VERSION
    from scripts.build_world_model_s1_generation_bundle import S1_OVERLAY_SCHEMAS
    from scripts.validate_world_model_s1_eval_contract import S1_OVERLAY_SCHEMAS as EVAL_SCHEMAS
    from scripts.world_model_s1_eval_common import S1_FAMILIES

    config = read_json(ROOT / "config/world_model_s1_120b_32n_test_v6_512n.json")
    path = ROOT / config["test_panel"]["evaluator_root"] / "manifest.json"
    overlay = Path("runtime/world_model/s1_v6.yaml")
    assert S1_OVERLAY_SCHEMAS[overlay] == EVAL_SCHEMAS[overlay.as_posix()] == S1_OVERLAY_V5_SCHEMA_VERSION
    bundle = build_generation_bundle(
        evaluator_manifest_path=path,
        evaluator_manifest_sha256=config["test_panel"]["evaluator_manifest_sha256"],
        source_index_path=ROOT / config["source_index"]["path"],
        source_index_sha256=config["source_index"]["sha256"], output_dir=tmp_path,
    )
    loaded, questions, _ = load_generation_bundle(tmp_path)
    assert loaded == bundle
    assert len(questions) == 6875 and set(bundle["family_counts"]) == set(S1_FAMILIES)
    assert bundle["test_panel_id"] == "eff174050b8d82ee2702cad2e4efebb8fa0938785e2ec6266640c23363ee7904"
    assert bundle["reads_private_answer_keys"] is False
    assert "diagnostic_only" not in bundle and "diagnostic_kind" not in bundle
    assert not (tmp_path / "test_answer_key.jsonl").exists()
