"""Keep robustness generation answer-free and separate from the S1 gate."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from runtime.world_model.contract import load_contract
from scripts.build_world_model_s1_generation_bundle import build_generation_bundle
from scripts.run_world_model_s1_exact_generation import (
    S1GenerationError, load_generation_bundle,
)
from scripts import validate_world_model_s1_eval_contract as validator
from scripts.world_model_s1_eval_common import (
    QUESTION_KEYS, S1EvaluationError, read_json, sha256_file, stable_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/world_model_v4/diagnostics/s1_robustness_v1/manifest.json"
METHOD = "oss120b-plain-base-tokenizer-lora-r32"
PROMOTION_PANEL = "eff174050b8d82ee2702cad2e4efebb8fa0938785e2ec6266640c23363ee7904"


@pytest.fixture(scope="module")
def diagnostic(tmp_path_factory):
    """Use the actual panel and production bundle builder, without answers."""
    from scripts.evaluate_world_model_s1_robustness import load_robustness_panel

    contract = load_contract()
    panel, questions, answers = load_robustness_panel(
        MANIFEST, sha256_file(MANIFEST), read_answers=False,
    )
    assert answers == {}
    source_sha = contract["source_index"]["content_sha256"]
    source = ROOT / contract["source_index"]["output_directory"] / (
        f"world_model_source_index_{source_sha}.json.gz"
    )
    work = tmp_path_factory.mktemp("s1_robustness_setup")
    arguments = {
        "evaluator_manifest_path": MANIFEST,
        "evaluator_manifest_sha256": sha256_file(MANIFEST),
        "source_index_path": source, "source_index_sha256": source_sha,
    }
    bundle = build_generation_bundle(**arguments, output_dir=work / "bundle")
    return panel, questions, work, bundle, arguments


def test_native_bundle_is_separate_and_answer_free(diagnostic):
    """The native runner must accept the exact diagnostic question set."""
    panel, questions, work, bundle, _ = diagnostic
    loaded, rows, _ = load_generation_bundle(work / "bundle")
    assert loaded == bundle and rows == questions
    assert loaded["diagnostic_only"] is True
    assert loaded["diagnostic_kind"] == "robustness_v1"
    assert loaded["reads_private_answer_keys"] is False
    assert loaded["record_count"] == panel["row_count"] == 344
    assert loaded["failure_row_count"] == 0
    assert loaded["test_panel_id"] != PROMOTION_PANEL
    assert all(set(row) == set(QUESTION_KEYS) for row in rows)
    assert {p.name for p in (work / "bundle").iterdir()} == {
        "manifest.json", "questions.jsonl",
    }
    assert "expected_messages" not in (work / "bundle/questions.jsonl").read_text()


def test_public_panel_builder_and_native_loader_never_open_answers(
    diagnostic, tmp_path, monkeypatch,
):
    """Guard filesystem reads across all three public-generation entry points."""
    from scripts.evaluate_world_model_s1_robustness import load_robustness_panel

    panel, questions, _, _, arguments = diagnostic
    original = Path.open
    reads = []

    def public_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in mode or "+" in mode:
            assert "answer_key" not in path.name and path.name != "rows.jsonl", (
                f"private answer read: {path}"
            )
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", public_open)
    verified, rows, answers = load_robustness_panel(
        MANIFEST, sha256_file(MANIFEST), read_answers=False,
    )
    assert verified == panel and rows == questions and answers == {}
    build_generation_bundle(**arguments, output_dir=tmp_path / "bundle")
    _, loaded, _ = load_generation_bundle(tmp_path / "bundle")
    assert loaded == questions and reads


@pytest.mark.parametrize("mutation", [
    "marker", "kind", "promotion", "panel_pin", "panel_path", "question", "failure_harness",
])
def test_rehashed_bundle_mutations_do_not_bypass_panel_binding(
    diagnostic, tmp_path, mutation,
):
    """A new outer hash cannot authorize changed records or diagnostic identity."""
    _, questions, _, bundle, _ = diagnostic
    revised = deepcopy(bundle)
    rows = deepcopy(questions)
    if mutation == "marker":
        revised["diagnostic_only"] = False
    elif mutation == "kind":
        revised["diagnostic_kind"] = "promotion"
    elif mutation == "promotion":
        for key in ("diagnostic_only", "diagnostic_kind", "diagnostic_manifest_path"):
            revised.pop(key)
    elif mutation == "panel_pin":
        revised["source_evaluator_manifest_sha256"] = "0" * 64
    elif mutation == "panel_path":
        revised["diagnostic_manifest_path"] = "data/world_model_v4/eval/s1_v6/manifest.json"
    elif mutation == "question":
        user = next(m for m in reversed(rows[0]["messages"]) if m["role"] == "user")
        user["content"] += " Please check carefully."
    else:
        revised["failure_harness"]["selector"] = "never"
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    revised["questions_sha256"] = sha256_file(questions_path)
    revised.pop("bundle_sha256")
    revised["bundle_sha256"] = stable_sha256(revised)
    (tmp_path / "manifest.json").write_text(json.dumps(revised))
    with pytest.raises((S1GenerationError, S1EvaluationError)):
        load_generation_bundle(tmp_path)


@pytest.mark.parametrize("version", ["v6", "v8"])
def test_diagnostic_config_exports_with_retained_reasoning(version, diagnostic, monkeypatch):
    """Validate real v6 checkpoint data; stub only the unavailable v8 checkpoint."""
    panel, _, _, _, _ = diagnostic
    path = ROOT / "config" / f"world_model_s1_120b_32n_robustness_{version}_32n.json"
    config = read_json(path)
    training = read_json(ROOT / config["training"]["config"])
    assert config["model"] == training["model"]
    assert config["corpus"]["manifest_sha256"] == training["corpus"]["manifest_sha256"]
    assert config["generation"]["history_policy"] == "retain_current_turn"
    assert config["tracking"]["resume_training_run"] is False
    checked = []
    if version == "v8":
        def checkpoint_stub(**kwargs):
            checked.append(kwargs)
            return "explicit-future-v8-checkpoint-stub", "1" * 64

        monkeypatch.setattr(validator, "_validate_checkpoint", checkpoint_stub)
        resolve = validator.resolve_repo_path

        def future_path(repo_root, value, label, **kwargs):
            if label in {"checkpoint root", "training output root"}:
                kwargs["must_exist"] = False
            return resolve(repo_root, value, label, **kwargs)

        monkeypatch.setattr(validator, "resolve_repo_path", future_path)
        checkpoint = ROOT / config["training"]["checkpoint_root"] / "explicit-future-v8-checkpoint-stub"
    else:
        checkpoint = ROOT / config["training"]["checkpoint_root"] / "slurm-5712804"
    values = validator.validate_and_export(
        repo_root=ROOT, config_path=path, method_id=METHOD, checkpoint_path=checkpoint,
    )
    assert values["S1_DIAGNOSTIC_ONLY"] == "1"
    assert values["S1_DIAGNOSTIC_KIND"] == "robustness_v1"
    assert values["S1_HISTORY_POLICY"] == "retain_current_turn"
    assert values["S1_TEST_PANEL_ID"] == panel["test_panel_id"]
    assert values["S1_TEST_ROWS"] == "344"
    assert values["S1_EVAL_NUM_NODES"] == "32"
    assert Path(values["S1_EVALUATOR_MANIFEST"]) == MANIFEST
    assert "S1_GATE_PROFILE" not in values
    if version == "v8":
        evaluator = read_json(ROOT / config["test_panel"]["evaluator_root"] / "manifest.json")
        assert len(checked) == 1
        assert checked[0]["corpus_manifest_sha256"] == training["corpus"]["manifest_sha256"]
        assert checked[0]["validation_answer_key_sha256"] == evaluator["validation"]["answer_key_sha256"]
    else:
        assert values["S1_TRAIN_RUN_ID"] == METHOD + "-slurm-5712804"


@pytest.mark.parametrize("mutation", ["strip", "gate", "tracking", "pin", "calls", "kind"])
def test_diagnostic_config_rejects_legacy_history_and_promotion_settings(diagnostic, mutation):
    """Reject relaxed history, promotion publication, and mismatched panel pins."""
    from scripts.evaluate_world_model_s1_robustness import validate_robustness_config

    config = read_json(ROOT / "config/world_model_s1_120b_32n_robustness_v6_32n.json")
    if mutation == "strip":
        config["generation"]["history_policy"] = "strip"
    elif mutation == "gate":
        config["gate"]["profile"] = "tiered_v3"
    elif mutation == "tracking":
        config["tracking"]["resume_training_run"] = True
    elif mutation == "pin":
        config["diagnostic_panel"]["manifest_sha256"] = "0" * 64
    elif mutation == "calls":
        config["generation"]["maximum_tool_calls"] = 3
    else:
        config["diagnostic_panel"]["kind"] = "lookup_retries"
    with pytest.raises(S1EvaluationError):
        validate_robustness_config(config, repo_root=ROOT, contract=load_contract())


def test_direct_generation_rejects_stripping_before_model_checks(tmp_path, monkeypatch):
    """A direct runner call cannot bypass the diagnostic's retained-history rule."""
    from scripts import run_world_model_s1_exact_generation as runner

    monkeypatch.setattr(
        runner, "load_generation_bundle",
        lambda path: ({"diagnostic_kind": "robustness_v1"}, [], tmp_path / "source"),
    )

    def forbidden_artifact_check(*args, **kwargs):
        pytest.fail("strip history reached model verification")

    monkeypatch.setattr(runner, "validate_base_model_artifact_binding", forbidden_artifact_check)
    with pytest.raises(S1GenerationError, match="requires retained current-turn reasoning"):
        runner.generate_predictions(
            bundle_root=tmp_path / "bundle", base_model_path=tmp_path / "model",
            checkpoint_path=tmp_path / "checkpoint", tokenizer_path=tmp_path / "tokenizer",
            tokenizer_manifest_path=tmp_path / "tokenizer_manifest.json",
            output_dir=tmp_path / "output", method_id=METHOD,
            corpus_manifest_sha256="1" * 64, tokenizer_manifest_sha256="2" * 64,
            model_identity_sha256="3" * 64, base_model_artifact_sha256="4" * 64,
            max_new_tokens=4096, max_total_tokens=8192, max_tool_call_tokens=1024,
            max_final_prompt_tokens=6144, enable_thinking=True, reasoning_effort="low",
            seed=900913, batch_size=1, local_files_only=True, history_policy="strip",
        )
    assert not (tmp_path / "output").exists()


def test_slurm_dispatch_retains_history_and_exits_before_promotion():
    """Check syntax and the dedicated diagnostic dispatch without launching jobs."""
    path = ROOT / "world_model_s1_eval.slurm"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert '--history-policy "${S1_HISTORY_POLICY}"' in text
    assert 'if [[ "${S1_DIAGNOSTIC_ONLY}" == "1" ]]; then\n  PUBLISH_WANDB=0' in text
    start = text.index('  diagnostic_scorer=scripts/world_model_s1_retry_eval.py')
    stop = text.index('\npublish_args=()', start)
    diagnostic = text[start:stop]
    assert '"${S1_DIAGNOSTIC_KIND}" == "robustness_v1"' in diagnostic
    assert 'diagnostic_scorer=scripts/evaluate_world_model_s1_robustness.py' in diagnostic
    assert 'diagnostic_report=robustness' in diagnostic
    assert 'exit 0' in diagnostic
    assert '--publish-wandb' not in diagnostic
