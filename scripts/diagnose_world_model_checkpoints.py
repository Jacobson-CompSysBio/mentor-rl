#!/usr/bin/env python3
"""Pair selected S1 checkpoint token losses with generated validation accuracy."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.world_model.contract import load_contract
from runtime.world_model.source_index import check_source_index, content_sha256
from runtime.world_model.tools import WorldModelToolRuntime
from runtime.world_model_training import tokenizer_artifact_hashes, tokenize_v3_trace_for_sft, validated_tokenizer_manifest
from scripts.build_world_model_v2_s0_generation_bundle import build_generation_bundle
from scripts.evaluate_world_model_s1 import score_s1_record
from scripts.evaluate_world_model_v2_s0 import (
    WorldModelContractError, _normalize_evidence_ids, parse_v3_answer,
    score_tool_trajectory_record,
)
from scripts.run_world_model_s1_exact_generation import (
    DeterministicEvidenceRuntime, generate_trajectory_batch, install_network_guard,
    validate_tokenizer_artifact_binding, write_prediction_jsonl,
)
from scripts import run_world_model_v2_s0_exact_generation as s0_generation
from scripts.validate_world_model_v2_s0_train_contract import validate_base_model
from scripts.world_model_s1_eval_common import read_json, read_jsonl, sha256_file, stable_sha256, validate_answer_row, validate_question, write_json

SCHEMA = "mentor-rl-checkpoint-validation-diagnostic-v1"
SEGMENTS = ("reasoning", "tool_call", "structured_final", "free_text", "formatting")
FIELDS = {1: "status", 2: "result", 4: "continuation_decision", 8: "provenance"}
GENERATION = dict(max_new_tokens=4096, max_total_tokens=8192, max_tool_call_tokens=1024,
                  max_final_prompt_tokens=6144, reasoning_effort="low")
ACCURACIES = ("structured", "result", "continuation", "tool", "json")


def checked(path: Path, digest: str) -> Path:
    """Require a pinned local artifact before reading its contents."""
    if sha256_file(path) != digest:
        raise ValueError(f"Diagnostic artifact identity changed: {path}")
    return path


def read_checked_rows(spec: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Read a pinned public panel or private reference artifact."""
    return read_jsonl(checked(Path(spec[kind + "_path"]), spec[kind + "_sha256"]))


def checkpoint_identity(path: Path, training: dict[str, Any], config_hash: str, step: int, final_step: int) -> dict[str, Any]:
    """Validate real completed artifacts and all training identity bindings."""
    method = training["methods"][0]
    artifact, run_id = s0_generation.validate_checkpoint(
        path, method_id=method["method_id"],
        corpus_manifest_sha256=training["corpus"]["manifest_sha256"],
        tokenizer_manifest_sha256=method["tokenizer_manifest_sha256"],
        model_identity_sha256=training["model"]["model_identity_sha256"],
        snapshot_validation=step != final_step,
    )
    manifest = read_json(path / "tp_adapter_manifest.json")
    required = {
        "run_config_sha256": config_hash, "train_sha256": training["corpus"]["train_sha256"],
        "tokenizer_arm_manifest_sha256": method["arm_manifest_sha256"],
        "validation_sha256": training["corpus"]["validation_sha256"],
    }
    if any(manifest["identity"].get(key) != value for key, value in required.items()):
        raise ValueError("Checkpoint config/corpus/tokenizer bindings differ from the diagnostic")
    exposure_path = path / ("run_contract/training_exposure.json" if step == final_step else "training_exposure.json")
    exposure = read_json(exposure_path)
    if exposure.get("run_id") != run_id or any(exposure.get("identity", {}).get(key) != value for key, value in required.items() if key != "validation_sha256"):
        raise ValueError("Checkpoint exposure belongs to a different run or input contract")
    if exposure["schedule"]["completed_global_step"] != step:
        raise ValueError("Checkpoint step and exposure receipt disagree")
    if step == final_step:
        if not exposure["logical_exposure"]["all_eligible_train_rows_exposed"]:
            raise ValueError("Final diagnostic requires a fully exposed completed checkpoint")
        state = read_json(path / "training_state.json")
        if state.get("global_step") != step:
            raise ValueError("Final checkpoint training state has a different step")
    elif artifact.get("snapshot_step") != step:
        raise ValueError("Snapshot path and snapshot metadata step disagree")
    tokenizer_files = tokenizer_artifact_hashes(ROOT / method["tokenizer_path"])
    validate_tokenizer_artifact_binding(path, tokenizer_files)
    return {"path": str(path), "step": step, "run_id": run_id,
            "checkpoint_identity_sha256": artifact["checkpoint_identity_sha256"],
            "adapter_sha256": sha256_file(path / "adapter_model.safetensors"),
            "exposure_sha256": sha256_file(exposure_path)}


def prepare(config_path: Path, checkpoint_path: Path, output: Path) -> dict[str, Any]:
    """Require completed training, pin public panels, and prepare no private labels."""
    config = read_json(config_path)
    if config.get("schema_version") != "mentor-rl-s1-checkpoint-diagnostic-v1":
        raise ValueError("Unsupported checkpoint diagnostic config")
    training_path = checked(ROOT / config["training_config"], config["training_config_sha256"])
    training = read_json(training_path)
    if training.get("training_id") != "world_model_s1_120b_training_v9_32n":
        raise ValueError("This diagnostic requires the frozen v9 run")
    if config["selected_steps"] != [400, 800, 1200, 1600, 1728] or config["final_step"] != 1728:
        raise ValueError("Selected checkpoint schedule changed")
    checkpoint_path = checkpoint_path.resolve()
    expected_root = (ROOT / training["methods"][0]["output_root"]).resolve()
    if checkpoint_path.parent != expected_root or not checkpoint_path.name.startswith("slurm-"):
        raise ValueError("Selected final checkpoint is outside the pinned training output")
    snapshots = []
    checkpoint_identity(checkpoint_path, training, config["training_config_sha256"], config["final_step"], config["final_step"])
    for step in config["selected_steps"]:
        path = checkpoint_path if step == config["final_step"] else checkpoint_path / "diagnostic_snapshots" / f"step-{step:06d}"
        snapshots.append(checkpoint_identity(path, training, config["training_config_sha256"], step, config["final_step"]))
    if len({item["run_id"] for item in snapshots}) != 1:
        raise ValueError("Selected checkpoints came from different runs")
    corpus = training["corpus"]
    checked(ROOT / corpus["root"] / "manifest.json", corpus["manifest_sha256"])
    evaluator = read_json(checked(ROOT / corpus["evaluator_manifest"], corpus["evaluator_manifest_sha256"]))
    s0_evaluator_path = checked(ROOT / config["s0_evaluator_manifest"], config["s0_evaluator_manifest_sha256"])
    s0_evaluator = read_json(s0_evaluator_path)
    contract = load_contract()
    source_spec = s0_evaluator["identifier_registry"]
    source_path = ROOT / source_spec["path"]
    if content_sha256(check_source_index(source_path, contract)) != source_spec["sha256"]:
        raise ValueError("Frozen source index identity changed")
    s0_corpus_path = checked(ROOT / s0_evaluator["corpus_manifest"], s0_evaluator["corpus_manifest_sha256"])
    s0_corpus = read_json(s0_corpus_path)
    panels = {}
    for name, spec in (("s1", evaluator["validation"]), ("s0", s0_evaluator["behavioral_validation"])):
        if "validation" not in Path(spec["questions_path"]).name or "validation" not in Path(spec["answer_key_path"]).name:
            raise ValueError("The checkpoint diagnostic must use validation panels")
        panels[name] = {key: str(ROOT / value) if key.endswith("_path") else value for key, value in spec.items()}
        rows = read_checked_rows(panels[name], "questions")
        ids = [row["record_id"] for row in rows]
        if len(rows) != spec["row_count"] or len(set(ids)) != len(ids) or any(row["split"] != "validation" for row in rows):
            raise ValueError("Validation coverage or split changed")
        if name == "s1":
            for row in rows:
                validate_question(row, contract, expected_split="validation")
        panels[name]["record_ids"] = ids
    panels["s1"].update(reference_path=str(ROOT / corpus["root"] / corpus["validation_file"]), reference_sha256=corpus["validation_sha256"])
    panels["s0_teacher_forced"] = {
        "reference_path": str(s0_corpus_path.parent / "validation.jsonl"),
        "reference_sha256": s0_corpus["files"]["validation.jsonl"]["sha256"],
        "row_count": 622,
        "validation_panel_id": s0_evaluator["teacher_forced_validation"]["validation_panel_id"],
    }
    teacher_spec = s0_evaluator["teacher_forced_validation"]
    teacher_questions = read_jsonl(checked(ROOT / teacher_spec["questions_path"], teacher_spec["questions_sha256"]))
    teacher_ids = [row["record_id"] for row in teacher_questions]
    if len(teacher_ids) != 622 or len(set(teacher_ids)) != 622 or any(row.get("split") != "validation" for row in teacher_questions):
        raise ValueError("S0 teacher-forced public panel coverage changed")
    panels["s0_teacher_forced"]["record_ids"] = teacher_ids
    tokenizer_manifest = validated_tokenizer_manifest((ROOT / training["methods"][0]["tokenizer_path"]).parent / "tokenizer_manifest.json")
    if tokenizer_manifest["manifest_sha256"] != training["methods"][0]["tokenizer_manifest_sha256"]:
        raise ValueError("Tokenizer manifest changed")
    plan = {
        "schema_version": SCHEMA, "diagnostic_only": True, "promotion_eligible": False,
        "test_panel_access": False, "history_policy": "retain_current_turn",
        "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
        "training": training, "training_config_sha256": config["training_config_sha256"],
        "checkpoints": snapshots, "panels": panels,
        "source_index_path": str(source_path), "source_index_sha256": source_spec["sha256"],
        "shard_count": config["shard_count"], "batch_size": config["batch_size"], "seed": config["seed"],
        "tracking": config["tracking"],
        "s0_evaluator_manifest": str(s0_evaluator_path),
        "s0_evaluator_manifest_sha256": config["s0_evaluator_manifest_sha256"],
        "generation": GENERATION,
        "private_artifact_access": "only_after_local_generation_for_both_behavioral_panels",
        "implementation_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/diagnose_world_model_checkpoints.py", "runtime/world_model_training.py",
            "runtime/world_model_loss_metrics.py", "scripts/run_world_model_s1_exact_generation.py",
            "scripts/run_world_model_v2_s0_exact_generation.py", "scripts/evaluate_world_model_s1.py",
            "scripts/evaluate_world_model_v2_s0.py", "runtime/world_model/contract.yaml",
        )},
    }
    plan["plan_sha256"] = stable_sha256(plan)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(output / "plan.json", plan, private=True)
    build_generation_bundle(evaluator_manifest_path=s0_evaluator_path,
        evaluator_manifest_sha256=config["s0_evaluator_manifest_sha256"],
        output_dir=output / "s0_bundle", panel_name="behavioral_validation")
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    """Recheck prepared bindings immediately before each worker or merge."""
    unsigned = dict(plan)
    if unsigned.pop("plan_sha256", None) != stable_sha256(unsigned):
        raise ValueError("Diagnostic plan checksum changed")
    checked(Path(plan["config_path"]), plan["config_sha256"])
    for name, digest in plan["implementation_sha256"].items():
        checked(ROOT / name, digest)
    for item in plan["checkpoints"]:
        actual = checkpoint_identity(Path(item["path"]), plan["training"], plan["training_config_sha256"], item["step"], 1728)
        if actual != item:
            raise ValueError("Prepared checkpoint identity changed")


def reference_loss(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure detached per-row segment and field NLL under frozen gold context."""
    import torch
    from runtime.world_model_loss_metrics import segment_token_totals, summarize_segment_totals
    results = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for index, row in enumerate(rows):
            encoded = tokenize_v3_trace_for_sft(row, tokenizer, max_length=4096, segment_telemetry=True)
            ids = torch.tensor([encoded["input_ids"]], device=device)
            labels = ids.clone()
            labels[torch.tensor([encoded["completion_mask"]], device=device) == 0] = -100
            labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
            segments = torch.tensor([encoded["loss_segments"]], device=device)
            segments = torch.nn.functional.pad(segments, (0, 1), value=0)[..., 1:]
            masks = torch.tensor([encoded["loss_field_masks"]], device=device)
            masks = torch.nn.functional.pad(masks, (0, 1), value=0)[..., 1:]
            outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = outputs.logits.float()
            labels, segments, masks = (value.to(logits.device) for value in (labels, segments, masks))
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none").reshape_as(labels)
            totals = segment_token_totals(token_loss=nll, shift_labels=labels, shift_segments=segments, logits=logits)
            correct = logits.argmax(dim=-1) == labels
            fields = {}
            for bit, name in FIELDS.items():
                selected = (labels != -100) & ((masks & bit) != 0)
                fields[name] = {"nll_sum": float(nll[selected].double().sum()), "token_count": int(selected.sum()), "correct_tokens": int((correct & selected).sum())}
            results.append({"record_id": row["row_id"], "source_trace": row["metadata"]["source_trace"],
                            "family": row["metadata"]["question_family"], "segments": summarize_segment_totals(totals), "fields": fields})
            del outputs, logits, nll
            if (index + 1) % 25 == 0:
                print(json.dumps({"phase": "reference_loss", "completed": index + 1, "total": len(rows)}), flush=True)
    return results


def validation_scoring_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Use the frozen validation name vocabulary for diagnostic name breakdowns."""
    result = deepcopy(contract)
    family = result["curriculum"]["families"]["S1.8"]
    for field in ("name_variants", "shared_aliases"):
        family[field]["test"] = family[field]["validation"]
    return result


def checked_record_order(rows: list[dict[str, Any]], expected_ids: list[str]) -> None:
    """Reject missing, duplicated, or reordered records before a scorer stamps IDs."""
    if [row.get("record_id") for row in rows] != expected_ids:
        raise ValueError("Diagnostic record coverage or order changed")


def worker(plan: dict[str, Any], output: Path, rank: int, base_model: Path) -> None:
    """Reuse one base model across adapters, generating before opening gold rows."""
    import torch
    if not 0 <= rank < plan["shard_count"]:
        raise ValueError("Diagnostic worker rank is outside the shard set")
    validate_plan(plan)
    training = plan["training"]
    contract = load_contract()
    scoring_contract = validation_scoring_contract(contract)
    validate_base_model(base_model, training["model"]["base_model_artifact_sha256"], training["model"]["weight_shard_count"], output / f"base-model-{rank:03d}.json")
    install_network_guard()
    model = tokenizer = None
    active_adapter = "default"
    s1_all = [validate_question(row, contract, expected_split="validation") for row in read_checked_rows(plan["panels"]["s1"], "questions")]
    indexes = {row["record_id"]: index for index, row in enumerate(s1_all)}
    s1_rows = s1_all[rank::plan["shard_count"]]
    s0_rows = read_checked_rows(plan["panels"]["s0"], "questions")[rank::plan["shard_count"]]
    for checkpoint in plan["checkpoints"]:
        directory = output / f"step-{checkpoint['step']:06d}" / "shards" / f"shard-{rank:03d}"
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        torch.manual_seed(plan["seed"])
        if model is None:
            model, tokenizer = s0_generation.load_backend(base_model_path=base_model, checkpoint_path=Path(checkpoint["path"]), tokenizer_path=ROOT / training["methods"][0]["tokenizer_path"], local_files_only=True)
        else:
            next_adapter = f"step_{checkpoint['step']}"
            model.load_adapter(checkpoint["path"], adapter_name=next_adapter, is_trainable=False)
            model.set_adapter(next_adapter)
            model.delete_adapter(active_adapter)
            active_adapter = next_adapter
        model.eval()
        tokenizer.padding_side = "left"
        runtime = DeterministicEvidenceRuntime.from_source_index(Path(plan["source_index_path"]), contract=contract, record_indexes=indexes)
        predictions = []
        for start in range(0, len(s1_rows), plan["batch_size"]):
            predictions.extend(generate_trajectory_batch(s1_rows[start:start + plan["batch_size"]], model=model, tokenizer=tokenizer,
                device=next(model.parameters()).device, torch_module=torch, runtime=runtime, contract=contract,
                shard_index=rank, network_guarded=True, history_policy="retain_current_turn", expected_split="validation", **plan["generation"]))
            write_prediction_jsonl(directory / "s1_predictions.jsonl", predictions)
        original_loader = s0_generation.load_backend
        s0_generation.load_backend = lambda **kwargs: (model, tokenizer)
        try:
            s0_generation.generate_predictions(bundle_root=output / "s0_bundle", base_model_path=base_model,
                checkpoint_path=Path(checkpoint["path"]), tokenizer_path=ROOT / training["methods"][0]["tokenizer_path"],
                tokenizer_manifest_path=(ROOT / training["methods"][0]["tokenizer_path"]).parent / "tokenizer_manifest.json",
                output_dir=directory / "s0_generation", method_id=training["methods"][0]["method_id"],
                corpus_manifest_sha256=training["corpus"]["manifest_sha256"],
                tokenizer_manifest_sha256=training["methods"][0]["tokenizer_manifest_sha256"],
                model_identity_sha256=training["model"]["model_identity_sha256"],
                enable_thinking=True, seed=plan["seed"], batch_size=plan["batch_size"], local_files_only=True,
                shard_index=rank, shard_count=plan["shard_count"], shard_output=True,
                snapshot_validation=checkpoint["step"] != 1728, **plan["generation"])
        finally:
            s0_generation.load_backend = original_loader
        # Both behavioral panels have finished generation before any private read.
        checked_record_order(predictions, [q["record_id"] for q in s1_rows])
        answers = {row["record_id"]: row for row in read_checked_rows(plan["panels"]["s1"], "answer_key")}
        s1_scores = [score_s1_record(q, validate_answer_row(answers[q["record_id"]], q, contract, expected_split="validation"), p, contract=scoring_contract) for q, p in zip(s1_rows, predictions, strict=True)]
        s0_answers = {row["record_id"]: row for row in read_checked_rows(plan["panels"]["s0"], "answer_key")}
        s0_predictions = read_jsonl(directory / "s0_generation/predictions.jsonl")
        checked_record_order(s0_predictions, [q["record_id"] for q in s0_rows])
        s0_runtime = WorldModelToolRuntime.from_source_index(Path(plan["source_index_path"]), contract=contract)
        s0_scores = [score_tool_trajectory_record(q, s0_answers[q["record_id"]], p, runtime=s0_runtime) for q, p in zip(s0_rows, s0_predictions, strict=True)]
        for question, score in zip(s0_rows, s0_scores, strict=True):
            score["source_trace"] = question["metadata"]["source_trace"]
        s1_references = {row["row_id"]: row for row in read_checked_rows(plan["panels"]["s1"], "reference")}
        s0_references = read_checked_rows(plan["panels"]["s0_teacher_forced"], "reference")
        if [row["row_id"] for row in s0_references] != plan["panels"]["s0_teacher_forced"]["record_ids"]:
            raise ValueError("S0 teacher-forced reference panel changed")
        report = {"schema_version": SCHEMA, "plan_sha256": plan["plan_sha256"],
                  "checkpoint": checkpoint, "shard_index": rank,
                  "s1_scores": s1_scores, "s0_scores": s0_scores,
                  "s1_loss": reference_loss(model, tokenizer, [s1_references[q["record_id"]] for q in s1_rows]),
                  "s0_loss": reference_loss(model, tokenizer, s0_references[rank::plan["shard_count"]])}
        write_json(directory / "report.json", report, private=True)
        write_json(directory / "complete.json", {"report_sha256": sha256_file(directory / "report.json"), "plan_sha256": plan["plan_sha256"], "checkpoint": checkpoint, "shard_index": rank}, private=True)


def summarized_loss(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum sufficient statistics before computing token-weighted means."""
    result = {}
    for group in ("segments", "fields"):
        result[group] = {}
        for name in (SEGMENTS if group == "segments" else FIELDS.values()):
            sums = {key: sum(row[group][name][key] for row in rows) for key in ("nll_sum", "token_count", "correct_tokens")}
            count = sums["token_count"]
            if not math.isfinite(sums["nll_sum"]) or sums["nll_sum"] < 0 or count < 0 or count != int(count) or not 0 <= sums["correct_tokens"] <= count or sums["correct_tokens"] != int(sums["correct_tokens"]):
                raise ValueError("Invalid additive checkpoint loss statistics")
            result[group][name] = {**sums, "nll": sums["nll_sum"] / count if count else None,
                                  "teacher_forced_token_accuracy": sums["correct_tokens"] / count if count else None}
    return result


def score_values(row: dict[str, Any], panel: str) -> dict[str, bool]:
    """Map the native evaluators onto explicitly named generated accuracies."""
    if panel == "s1":
        return {"structured": row["structured_answer_accuracy"], "result": row["result_accuracy"],
                "continuation": row["continuation_decision_accuracy"],
                "tool": row["valid_trajectory"] and row["exact_tool_selection"] and row["exact_tool_arguments"],
                "json": row["structured_final_json"]}
    return {"structured": s0_structured_accuracy(row),
            "result": row["result_exact"], "continuation": row["continuation_decision_exact"],
            "tool": row["valid_single_tool_call"] and row["exact_tool_argument"], "json": row["structured_json_valid"]}


def s0_structured_accuracy(row: dict[str, Any]) -> bool:
    """Score every S0 structured value and evidence binding, excluding prose wording."""
    if not row["structured_json_valid"] or not row["valid_final_answer"] or not row["payload_exact"]:
        return False
    try:
        generated = parse_v3_answer(row["assistant_final"]).payload
        expected = parse_v3_answer(row["expected_final_answer"]).payload
    except WorldModelContractError:
        return False
    payload = row.get("generated_tool_payload")
    if not isinstance(payload, dict) or generated["provenance"] != payload.get("provenance"):
        return False
    if (generated["blocked_reason"] is not None) != (expected["blocked_reason"] is not None):
        return False
    return _normalize_evidence_ids({**generated, "summary": None, "blocked_reason": None}) == _normalize_evidence_ids({**expected, "summary": None, "blocked_reason": None})


def merge(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    """Require exact shard coverage, join immutable identities, and publish trends."""
    validate_plan(plan)
    report = {"schema_version": SCHEMA, "plan_sha256": plan["plan_sha256"], "diagnostic_only": True,
              "promotion_eligible": False, "teacher_forced_context": "frozen_reference_reasoning_and_actions",
              "s0_loss_panel_differs_from_behavioral_panel": True, "checkpoints": []}
    joined = []
    for checkpoint in plan["checkpoints"]:
        combined = {name: [] for name in ("s1_scores", "s0_scores", "s1_loss", "s0_loss")}
        for rank in range(plan["shard_count"]):
            directory = output / f"step-{checkpoint['step']:06d}/shards/shard-{rank:03d}"
            completion = read_json(directory / "complete.json")
            checked(directory / "report.json", completion["report_sha256"])
            shard = read_json(directory / "report.json")
            for key, value in (("plan_sha256", plan["plan_sha256"]), ("checkpoint", checkpoint), ("shard_index", rank)):
                if completion[key] != value or shard[key] != value:
                    raise ValueError("Diagnostic shard identity changed")
            for name in combined:
                panel = name.split("_", 1)[0]
                coverage_panel = "s0_teacher_forced" if name == "s0_loss" else panel
                checked_record_order(shard[name], plan["panels"][coverage_panel]["record_ids"][rank::plan["shard_count"]])
                combined[name].extend(shard[name])
        summary = {"checkpoint": checkpoint, "panels": {}}
        for panel in ("s1", "s0"):
            scores, losses = combined[panel + "_scores"], combined[panel + "_loss"]
            ids = [row["record_id"] for row in scores]
            expected = plan["panels"][panel]["record_ids"]
            loss_ids = [row["record_id"] for row in losses]
            if len(ids) != len(set(ids)) or set(ids) != set(expected) or len(loss_ids) != len(set(loss_ids)):
                raise ValueError("Diagnostic score/loss coverage changed")
            if panel == "s1" and set(loss_ids) != set(expected):
                raise ValueError("S1 loss and generated records do not join exactly")
            if panel == "s0" and set(loss_ids) != set(plan["panels"]["s0_teacher_forced"]["record_ids"]):
                raise ValueError("S0 teacher-forced loss coverage changed")
            values = {row["record_id"]: score_values(row, panel) for row in scores}
            metrics = {key: sum(bool(value[key]) for value in values.values()) / len(values) for key in ACCURACIES}
            groups = {}
            for group in ("family", "source_trace"):
                grouped = defaultdict(list)
                for row in losses:
                    grouped[row[group]].append(row)
                score_groups = defaultdict(list)
                for row in scores:
                    key = row.get("question_family") if group == "family" and panel == "s1" else row.get(group)
                    score_groups[key].append(values[row["record_id"]])
                groups[group] = {
                    name: {"loss": summarized_loss(rows),
                           "reference_row_count": len(rows),
                           "generated_row_count": len(score_groups[name]),
                           "accuracy": {metric: sum(bool(row[metric]) for row in score_groups[name]) / len(score_groups[name]) for metric in ACCURACIES}}
                    for name, rows in sorted(grouped.items()) if score_groups[name]
                }
            summary["panels"][panel] = {"generated_row_count": len(scores), "reference_row_count": len(losses),
                "questions_sha256": plan["panels"][panel]["questions_sha256"], "accuracy": metrics,
                "loss": summarized_loss(losses), "loss_by": groups,
                "reference_sha256": plan["panels"]["s1" if panel == "s1" else "s0_teacher_forced"]["reference_sha256"]}
            if panel == "s1":
                for row in losses:
                    joined.append({**row, "checkpoint": checkpoint, "panel_sha256": plan["panels"][panel]["questions_sha256"], "generated_accuracy": values[row["record_id"]]})
                outcome_groups = {}
                for metric in ACCURACIES:
                    outcome_groups[metric] = {}
                    for success in (False, True):
                        selected = [row for row in losses if bool(values[row["record_id"]][metric]) == success]
                        outcome_groups[metric]["correct" if success else "incorrect"] = {"row_count": len(selected), "loss": summarized_loss(selected)}
                summary["panels"][panel]["loss_by_generated_outcome"] = outcome_groups
        report["checkpoints"].append(summary)
    write_prediction_jsonl(output / "joined_s1_rows.jsonl", joined)
    report["joined_rows_sha256"] = sha256_file(output / "joined_s1_rows.jsonl")
    report["correlations"] = correlations(report["checkpoints"])
    report["paired_error_transitions"] = paired_transitions(joined)
    write_json(output / "report.json", report, private=True)
    plots_and_wandb(report, output)
    return report


def paired_transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count matched generated error transitions between adjacent checkpoints."""
    by_step = defaultdict(dict)
    for row in rows:
        by_step[row["checkpoint"]["step"]][row["record_id"]] = row
    results = []
    steps = sorted(by_step)
    for before_step, after_step in zip(steps, steps[1:]):
        before, after = by_step[before_step], by_step[after_step]
        if set(before) != set(after):
            raise ValueError("Checkpoint comparisons require identical validation records")
        for metric in ACCURACIES:
            counts = defaultdict(int)
            for record_id, row in before.items():
                counts[f"{int(row['generated_accuracy'][metric])}->{int(after[record_id]['generated_accuracy'][metric])}"] += 1
            results.append({"before_step": before_step, "after_step": after_step, "metric": metric, "counts": dict(sorted(counts.items()))})
    return results


def correlations(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report descriptive checkpoint correlations with their small sample counts."""
    from scipy.stats import pearsonr, spearmanr
    results = []
    for panel in ("s1", "s0"):
        scopes = [("overall", None)]
        for breakdown in ("family", "source_trace"):
            names = sorted(set.intersection(*(set(row["panels"][panel]["loss_by"][breakdown]) for row in checkpoints)))
            scopes.extend((breakdown, name) for name in names)
        for scope, scope_name in scopes:
            measurements = [row["panels"][panel] if scope == "overall" else row["panels"][panel]["loss_by"][scope][scope_name] for row in checkpoints]
            for group, names in (("segments", SEGMENTS), ("fields", FIELDS.values())):
                for name in names:
                    for metric in ACCURACIES:
                        pairs = [(row["loss"][group][name]["nll"], row["accuracy"][metric]) for row in measurements]
                        pairs = [(x, y) for x, y in pairs if x is not None]
                        defined = len(pairs) >= 3 and len({x for x, y in pairs}) > 1 and len({y for x, y in pairs}) > 1
                        results.append({"panel": panel, "scope": scope, "scope_name": scope_name,
                            "loss_group": group, "loss_name": name, "accuracy_name": metric,
                            "sample_count": len(pairs), "pearson": float(pearsonr(*zip(*pairs)).statistic) if defined else None,
                            "spearman": float(spearmanr(*zip(*pairs)).statistic) if defined else None,
                            "interpretation": "descriptive_within_run_not_causal"})
    return results


def plots_and_wandb(report: dict[str, Any], output: Path) -> None:
    """Write shareable plots and a separate diagnostic W&B run after all workers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for panel in ("s1", "s0"):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        steps = [row["checkpoint"]["step"] for row in report["checkpoints"]]
        for name in SEGMENTS:
            axes[0].plot(steps, [row["panels"][panel]["loss"]["segments"][name]["nll"] for row in report["checkpoints"]], marker="o", label=name)
        for name in ACCURACIES:
            axes[1].plot(steps, [row["panels"][panel]["accuracy"][name] for row in report["checkpoints"]], marker="o", label=name)
        for axis in axes:
            axis.set_xlabel("Optimizer update")
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
        axes[0].set_ylabel("Teacher-forced mean NLL")
        axes[1].set_ylabel("Generated validation accuracy")
        figure.suptitle(panel.upper() + " checkpoint diagnostic")
        figure.tight_layout()
        figure.savefig(output / f"{panel}_loss_accuracy.png", dpi=160)
        plt.close(figure)
    import wandb
    plan = read_json(output / "plan.json")
    run = wandb.init(entity=plan["tracking"]["wandb_entity"], project=plan["tracking"]["wandb_project"],
                     group=report["checkpoints"][-1]["checkpoint"]["run_id"],
                     name=f"s1-v9-checkpoint-diagnostics-{report['plan_sha256'][:10]}",
                     job_type="checkpoint_validation_diagnostic", config={"plan_sha256": report["plan_sha256"], "promotion_eligible": False},
                     dir=str(output), reinit=True)
    try:
        run.define_metric("checkpoint_step", hidden=True)
        run.define_metric("s1/*", step_metric="checkpoint_step")
        run.define_metric("s0/*", step_metric="checkpoint_step")
        for row in report["checkpoints"]:
            metrics = {"checkpoint_step": row["checkpoint"]["step"]}
            for panel, values in row["panels"].items():
                accuracy = "structured" if panel == "s1" else "result"
                metrics[f"{panel}/accuracy/{accuracy}"] = values["accuracy"][accuracy]
                # Keep field breakdowns and plot images in the local report directory.
                for name in ("reasoning", "tool_call", "structured_final"):
                    nll = values["loss"]["segments"][name]["nll"]
                    if nll is not None:
                        metrics[f"{panel}/nll/{name}"] = nll
            run.log(metrics)
        run.summary["diagnostic_report_sha256"] = sha256_file(output / "report.json")
        write_json(output / "wandb.json", {"run_id": run.id, "url": run.url}, private=True)
    finally:
        run.finish()


def main() -> None:
    """Prepare once after training, run disjoint workers, then merge once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "merge"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--base-model-cache-path", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.mode == "prepare":
        prepare(args.config.resolve(), args.checkpoint_path.resolve(), output)
    else:
        plan = read_json(output / "plan.json")
        if str(args.config.resolve()) != plan["config_path"] or str(args.checkpoint_path.resolve()) != plan["checkpoints"][-1]["path"]:
            raise ValueError("Worker arguments differ from the prepared run")
        if args.mode == "worker":
            if args.shard_index is None or args.base_model_cache_path is None:
                parser.error("worker requires --shard-index and --base-model-cache-path")
            worker(plan, output, args.shard_index, args.base_model_cache_path.resolve())
        else:
            merge(plan, output)


if __name__ == "__main__":
    main()
