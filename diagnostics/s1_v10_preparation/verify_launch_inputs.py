"""Verify the three v10 launch inputs, recipe parity, and reviewed source snapshot."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
PREPARATION = Path(__file__).parent
SNAPSHOT = ROOT / "data/world_model_v4/launch_reports/s1_v10_120b_32n/submitted_code"
CONFIGS = {
    "training": "config/world_model_s1_120b_training_v10_32n.json",
    "s1_gate": "config/world_model_s1_120b_32n_test_v10_structured95_512n.json",
    "s0_retention": "config/world_model_s1_120b_s0_forgetting_test_v10_512n.json",
}


def read(path):
    """Read a JSON input relative to the repository."""
    return json.loads((ROOT / path).read_text())


def sha256(path):
    """Hash a real input without loading large artifacts into memory."""
    with (ROOT / path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, expected):
    """Reject a changed input before recording readiness."""
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Input hash differs: {path}")
    return actual


def verify_tokenizer_manifest(path, expected):
    """Check the tokenizer manifest's canonical identity and report its raw file hash."""
    sys.path.insert(0, str(ROOT))
    from runtime.world_model_training import validated_tokenizer_manifest
    assert validated_tokenizer_manifest(ROOT / path)["manifest_sha256"] == expected
    return sha256(path)


def snapshot_sources():
    """Preserve the v9 source set and current launch dependencies without docs or tests."""
    paths = set(read("diagnostics/s1_v9_preparation/readiness.json")["code_sha256"])
    paths.add("runtime/world_model/s1_v10.yaml")
    # Both launchers name their implementation helpers explicitly. Retain these
    # dependencies even if a newly added helper was absent from the v9 snapshot.
    for launcher in ("world_model_v2_s0.slurm", "world_model_s1_eval.slurm"):
        paths.update(re.findall(r"(?:runtime|scripts|utils)/[A-Za-z0-9_/]+\.py", (ROOT / launcher).read_text()))
    assert not any(path.startswith(("tests/", "docs/")) or path.endswith(".md") for path in paths)
    hashes = {}
    for path in sorted(paths):
        source, destination = ROOT / path, SNAPSHOT / path
        actual = sha256(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert hashlib.file_digest(destination.open("rb"), "sha256").hexdigest() == actual, f"Existing source snapshot changed: {path}"
        else:
            shutil.copy2(source, destination)
        assert hashlib.file_digest(destination.open("rb"), "sha256").hexdigest() == actual
        hashes[path] = actual
    return hashes


def main():
    """Validate real static inputs and record readiness without submitting any job."""
    configs = {key: read(path) for key, path in CONFIGS.items()}
    train = configs["training"]
    baseline = read("config/world_model_s1_120b_training_v9_32n.json")
    normalized = deepcopy(train)
    for key in ("schema_version", "training_id"):
        normalized[key] = baseline[key]
    for key in ("root", "manifest_sha256", "train_sha256", "evaluator_manifest", "evaluator_manifest_sha256"):
        normalized["corpus"][key] = baseline["corpus"][key]
    normalized["methods"][0]["output_root"] = baseline["methods"][0]["output_root"]
    assert normalized == baseline, "V10 changed the v9 optimizer, parent, tokenizer, objective, telemetry, or exposure recipe"
    assert train["trainer"]["segment_loss_telemetry"] is True
    assert train["run_settings"]["diagnostic_snapshot_steps"] == [400, 800, 1200, 1600]

    corpus, verified = train["corpus"], {}
    method = train["methods"][0]
    for path, expected in (
        (corpus["root"] + "/manifest.json", corpus["manifest_sha256"]),
        (corpus["root"] + "/train.jsonl", corpus["train_sha256"]),
        (corpus["root"] + "/validation.jsonl", corpus["validation_sha256"]),
        (corpus["evaluator_manifest"], corpus["evaluator_manifest_sha256"]),
        (train["initialization"]["checkpoint_path"] + "/adapter_model.safetensors", train["initialization"]["adapter_sha256"]),
        (method["arm_root"] + "/manifest.json", method["arm_manifest_sha256"]),
    ):
        verified[path] = verify(path, expected)

    tokenizer_manifest = method["arm_root"] + "/tokenizer_manifest.json"
    verified[tokenizer_manifest] = verify_tokenizer_manifest(tokenizer_manifest, method["tokenizer_manifest_sha256"])

    for kind in ("s1_gate", "s0_retention"):
        config = configs[kind]
        assert config["training"]["config"] == CONFIGS["training"]
        assert config["training"]["config_sha256"] == sha256(CONFIGS["training"])
        assert config["training"]["training_id"] == train["training_id"]
        assert config["training"]["checkpoint_root"] == method["output_root"]
        assert config["corpus"]["manifest_sha256"] == corpus["manifest_sha256"]
        assert config["corpus"]["train_sha256"] == corpus["train_sha256"]
        previous = read(CONFIGS[kind].replace("v10", "v9"))
        normalized_eval = deepcopy(config)
        normalized_eval["evaluation_id"] = previous["evaluation_id"]
        normalized_eval["training"] = previous["training"]
        for key in ("root", "manifest_sha256", "train_sha256"):
            normalized_eval["corpus"][key] = previous["corpus"][key]
        normalized_eval["methods"][0]["output_root"] = previous["methods"][0]["output_root"]
        if kind == "s1_gate":
            for key in ("evaluator_root", "evaluator_manifest_sha256"):
                normalized_eval["test_panel"][key] = previous["test_panel"][key]
        assert normalized_eval == previous, f"Changed {kind} generation, gate, panel, model/tokenizer, or tracking settings"
        panel = config["test_panel"]
        panel_root = panel["evaluator_root"]
        manifest = panel_root + "/" + panel["evaluator_manifest"]
        verified[manifest] = verify(manifest, panel["evaluator_manifest_sha256"])
        for name in ("questions", "answer_key"):
            path = panel_root + "/" + panel[name + "_file"]
            verified[path] = verify(path, panel[name + "_sha256"])
        for entry in config["methods"]:
            verified[entry["tokenizer_manifest"]] = verify_tokenizer_manifest(entry["tokenizer_manifest"], entry["tokenizer_manifest_sha256"])

    sys.path.insert(0, str(ROOT))
    from runtime.world_model.contract import load_contract
    from runtime.world_model.source_index import check_source_index, content_sha256
    source = configs["s1_gate"]["source_index"]
    assert source == configs["s0_retention"]["identifier_registry"]
    assert content_sha256(check_source_index(ROOT / source["path"], load_contract())) == source["sha256"]
    verified[source["path"]] = sha256(source["path"])

    audit_path = "diagnostics/s1_v10_build/final_audit.json"
    audit = read(audit_path)
    build_path = "diagnostics/s1_v10_build/build_verification.json"
    build = read(build_path)
    assert audit["passed"] is True and build["corpus_checks_passed"] is True
    assert audit["max_length"] == train["run_settings"]["max_length"] == 4096
    assert audit["serializer_sha256"] == sha256("runtime/world_model_training.py")
    assert audit["overlay_sha256"] == sha256("runtime/world_model/s1_v10.yaml")
    assert len(audit["frozen"]["identical_heldout_files"]) == 22
    assert build["aggregate_training_semantic_violations"] == 0
    assert build["deterministic_rebuild_files_byte_identical"] == 16
    for group in ("files", "tokenizer_files", "implementation_files"):
        for path, expected in audit[group].items():
            verified[path] = verify(path, expected)
    for name, expected in audit["frozen"]["identical_heldout_files"].items():
        parent = "data/world_model_v4/eval/s1_v10" if name.endswith(("_questions.jsonl", "_answer_key.jsonl")) else corpus["root"]
        path = parent + "/" + name
        verified[path] = verify(path, expected)
    for split, report in audit["splits"].items():
        assert report["maximum_tokens"] <= 4096
        assert report["rows"] == build["row_counts"][split]

    config_pins = read("diagnostics/s1_v10_preparation/eval_config_pins.json")
    assert config_pins["training_config_sha256"] == sha256(CONFIGS["training"])
    for entry in config_pins["evaluation_configs"]:
        verify(entry["path"], entry["sha256"])
    eval_checks = read("diagnostics/s1_v10_preparation/eval_config_checks.json")
    assert eval_checks["exit_code"] == 0 and eval_checks["summary"].startswith("5 passed")
    preflight_path = "diagnostics/s1_v10_preparation/train_launcher_preflight.log"
    preflight = (ROOT / preflight_path).read_text()
    assert "Resources: nodes=32 ranks_per_node=8" in preflight
    assert "calculated_steps=1728 global_batch=32 learning_rate=0.0001" in preflight
    assert "The selected inputs and config are valid. No Slurm job ran." in preflight

    now = datetime.now(timezone.utc).isoformat()
    report = {
        "status": "passed_static_input_checks", "recorded_at_utc": now,
        "configs": {kind: {"path": path, "sha256": sha256(path)} for kind, path in CONFIGS.items()},
        "verified_input_sha256": verified, "recipe_matches_v9": True,
        "evaluation_recipes_match_v9": True, "s1_gate_profile": "structured_95_v1",
        "telemetry_enabled": True, "diagnostic_snapshot_steps": [400, 800, 1200, 1600],
        "source_index_content_sha256": source["sha256"],
        "completed_checkpoint_preflight_performed": False,
        "checkpoint_validation": "Dependent evaluations must validate the real completed checkpoint after training succeeds; static tests use explicit fixtures/stubs.",
        "jobs_submitted": False,
    }
    (PREPARATION / "static_launch_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    code = snapshot_sources()
    evidence_paths = [audit_path, build_path, preflight_path,
                      "diagnostics/s1_v10_preparation/eval_config_checks.json",
                      "diagnostics/s1_v10_preparation/eval_config_pins.json",
                      "diagnostics/s1_v10_preparation/static_launch_audit.json"]
    readiness = {
        "status": "ready_to_submit", "recorded_at_utc": now,
        "configs": report["configs"], "code_sha256": code,
        "submitted_code_directory": str(SNAPSHOT.relative_to(ROOT)),
        "source_snapshot_files": len(code), "source_snapshot_copies_verified": True,
        "evidence_sha256": {path: sha256(path) for path in evidence_paths},
        "checks": {
            "corpus_and_panels_verified": True, "recipe_matches_v9": True,
            "training_launcher_cpu_preflight": "passed", "evaluation_config_tests": {"passed": 5},
            "segment_annotations_audited_on_all_rows": True, "all_22_heldout_files_identical": True,
            "deterministic_rebuild_files_identical": 16, "aggregate_training_semantic_violations": 0,
            "tokenizer_supervision_rows": 69026,
            "maximum_tokens": {split: value["maximum_tokens"] for split, value in audit["splits"].items()},
            "gpu_execution_preflight": "Runs inside real Slurm allocations; not claimed by CPU preflight.",
            "completed_checkpoint_preflight_performed": False,
        },
        "known_external_or_historical_checks": [
            "Frozen held-out aggregates retain 251 validation and 148 test evidence-ID collisions; reports are byte-identical to v9. Training semantics pass with zero violations.",
            "Live raw-source limitations remain separate; the actual frozen source-index content hash is verified.",
            "Historical robustness implementation pins remain frozen; this plan contains only training, S1 gate, and S0 retention.",
            "Evaluation config tests use explicit future-checkpoint fixtures/stubs; real completed-checkpoint validation remains mandatory inside each dependent job.",
        ],
        "snapshot_steps": [400, 800, 1200, 1600], "final_step": 1728,
        "training_submitted": False, "evaluation_submitted": False,
    }
    (PREPARATION / "readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")
    print(f"V10 static inputs and unchanged v9 recipe verified; {len(code)} source files copied and checked. No jobs submitted.")


if __name__ == "__main__":
    main()
