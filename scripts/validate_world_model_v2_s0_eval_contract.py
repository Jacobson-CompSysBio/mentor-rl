#!/usr/bin/env python3
"""Validate one S0 test contract and print its shell values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any


TEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-test-v4"
EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s0-evaluator-manifest-v4"
EXPOSURE_SCHEMA_VERSION = "mentor-rl-s0-training-exposure-v2"
EVALUATION_CONTRACT = "seen_fact_closed_book_recall_v1"
DATASET_ID = "world_model_v2_s0_human_identifiers_v4"
CHECKPOINT_FORMAT = "mentor-rl-s0-tp-lora-v1"
TOKENIZER_METHODS = {
    "plain_base_tokenizer",
    "ordinary_domain_bpe",
    "atomic_plus_domain_bpe",
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 value for canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 value for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    """Read one required JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain one JSON object")
    return payload


def require_string(value: Any, label: str) -> str:
    """Return one required nonempty string."""

    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} must be a nonempty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    """Return one required SHA-256 value."""

    text = require_string(value, label)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise SystemExit(f"{label} must be one lowercase SHA-256 value")
    return text


def require_int(value: Any, label: str) -> int:
    """Return one positive integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise SystemExit(f"{label} must be a positive integer")
    return value


def require_number(value: Any, label: str) -> float:
    """Return one finite numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise SystemExit(f"{label} must be between zero and one")
    return result


def resolve_repo_path(root: Path, value: Any, label: str) -> Path:
    """Resolve one path below the repository."""

    path = Path(require_string(value, label))
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"{label} must stay in the repository") from error
    return path


def resolve_model_path(root: Path, value: Any) -> Path:
    """Resolve one model path in the shared LLM project root."""

    path = Path(require_string(value, "model path"))
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        path.relative_to(root.parent)
    except ValueError as error:
        raise SystemExit(
            "The model path must stay in the shared LLM project root"
        ) from error
    return path


def verify_file(path: Path, expected_sha256: str, label: str) -> None:
    """Verify one required file identity."""

    if not path.is_file():
        raise SystemExit(f"{label} is absent: {path}")
    if sha256_file(path) != expected_sha256:
        raise SystemExit(f"{label} identity changed")


def validate_internal_manifest(
    payload: dict[str, Any], label: str
) -> str:
    """Verify one content-addressed JSON manifest."""

    claimed = require_sha256(payload.get("manifest_sha256"), label)
    identity = {
        str(key): value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if stable_sha256(identity) != claimed:
        raise SystemExit(f"{label} failed its internal identity")
    return claimed


def main() -> int:
    """Validate and export one S0 test contract."""

    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: validate_world_model_v2_s0_eval_contract.py "
            "REPO_ROOT RUN_CONFIG METHOD_ID CHECKPOINT_PATH"
        )
    root = Path(sys.argv[1]).resolve()
    config_path = resolve_repo_path(root, sys.argv[2], "RUN_CONFIG")
    method_id = require_string(sys.argv[3], "METHOD_ID")
    checkpoint_path = Path(sys.argv[4]).resolve()
    if not checkpoint_path.is_dir():
        raise SystemExit("The completed checkpoint directory is absent")

    config = read_json(config_path, "the S0 test config")
    if (
        config.get("schema_version") != TEST_SCHEMA_VERSION
        or config.get("evaluation_contract") != EVALUATION_CONTRACT
    ):
        raise SystemExit("The S0 test config contract changed")
    methods = config.get("methods")
    if not isinstance(methods, list):
        raise SystemExit("The S0 test config has no methods")
    selected = [item for item in methods if item.get("method_id") == method_id]
    if len(selected) != 1:
        raise SystemExit("METHOD_ID must select one S0 test method")
    method = selected[0]
    tokenizer_method = require_string(
        method.get("tokenizer_method"), "tokenizer_method"
    )
    if tokenizer_method not in TOKENIZER_METHODS:
        raise SystemExit("The tokenizer method is invalid")
    if (
        method.get("checkpoint_kind") != "lora"
        or method.get("fine_tune_configuration") != "lora_r32"
    ):
        raise SystemExit("The 20B test requires one LoRA-r32 checkpoint")

    model = config.get("model")
    corpus = config.get("corpus")
    panel = config.get("test_panel")
    generation = config.get("generation")
    gate = config.get("gate")
    tracking = config.get("tracking")
    if not all(
        isinstance(value, dict)
        for value in (model, corpus, panel, generation, gate, tracking)
    ):
        raise SystemExit("The S0 test config has an invalid section")

    model_path = resolve_model_path(root, model.get("path"))
    model_identity = require_sha256(
        model.get("model_identity_sha256"), "model identity"
    )
    if model.get("base_model_artifact_schema") != "mentor-rl-base-model-artifact-v2":
        raise SystemExit("The base-model artifact schema changed")
    base_artifact = require_sha256(
        model.get("base_model_artifact_sha256"), "base-model artifact"
    )
    weight_shards = require_int(
        model.get("weight_shard_count"), "model weight shard count"
    )
    metadata_names = ["config.json", "model.safetensors.index.json"]
    if (model_path / "generation_config.json").is_file():
        metadata_names.append("generation_config.json")
    model_metadata_identity = hashlib.sha256(
        "".join(
            f"{sha256_file(model_path / name)}  {name}\n"
            for name in metadata_names
        ).encode("utf-8")
    ).hexdigest()
    if model_metadata_identity != model_identity:
        raise SystemExit("The base-model metadata identity changed")

    corpus_root = resolve_repo_path(root, corpus.get("root"), "corpus root")
    corpus_manifest_hash = require_sha256(
        corpus.get("manifest_sha256"), "corpus manifest"
    )
    corpus_manifest_path = corpus_root / "manifest.json"
    verify_file(corpus_manifest_path, corpus_manifest_hash, "corpus manifest")
    corpus_manifest = read_json(corpus_manifest_path, "the corpus manifest")
    if (
        corpus_manifest.get("dataset_id") != DATASET_ID
        or corpus_manifest.get("evaluation_contract") != EVALUATION_CONTRACT
        or corpus_manifest.get("row_counts", {}).get("test")
        != require_int(panel.get("row_count"), "test row count")
        or corpus_manifest.get("train_population", {}).get(
            "eligible_train_rows"
        )
        != require_int(corpus.get("eligible_train_rows"), "eligible train rows")
    ):
        raise SystemExit("The corpus test contract changed")
    train_sha256 = require_sha256(corpus.get("train_sha256"), "train file")
    if corpus_manifest.get("file_hashes", {}).get("train.jsonl") != train_sha256:
        raise SystemExit("The train identity differs from the corpus manifest")

    evaluator_root = resolve_repo_path(
        root, panel.get("evaluator_root"), "evaluator root"
    )
    evaluator_manifest_path = evaluator_root / require_string(
        panel.get("evaluator_manifest"), "evaluator manifest name"
    )
    evaluator_manifest_hash = require_sha256(
        panel.get("evaluator_manifest_sha256"), "evaluator manifest"
    )
    verify_file(
        evaluator_manifest_path,
        evaluator_manifest_hash,
        "evaluator manifest",
    )
    evaluator = read_json(evaluator_manifest_path, "the evaluator manifest")
    test = evaluator.get("test")
    if (
        evaluator.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or evaluator.get("dataset_id") != DATASET_ID
        or evaluator.get("evaluation_contract") != EVALUATION_CONTRACT
        or not isinstance(test, dict)
    ):
        raise SystemExit("The evaluator manifest contract changed")
    questions_path = evaluator_root / require_string(
        panel.get("questions_file"), "questions file"
    )
    answer_key_path = evaluator_root / require_string(
        panel.get("answer_key_file"), "answer-key file"
    )
    questions_sha256 = require_sha256(
        panel.get("questions_sha256"), "test questions"
    )
    answer_key_sha256 = require_sha256(
        panel.get("answer_key_sha256"), "test answer key"
    )
    verify_file(questions_path, questions_sha256, "test questions")
    verify_file(answer_key_path, answer_key_sha256, "test answer key")
    test_panel_id = require_sha256(panel.get("test_panel_id"), "test panel ID")
    if (
        test.get("questions_sha256") != questions_sha256
        or test.get("answer_key_sha256") != answer_key_sha256
        or test.get("test_panel_id") != test_panel_id
        or test.get("row_count") != panel.get("row_count")
    ):
        raise SystemExit("The fixed test panel identity changed")

    tokenizer_path = resolve_repo_path(
        root, method.get("tokenizer_path"), "tokenizer path"
    )
    tokenizer_manifest_path = resolve_repo_path(
        root, method.get("tokenizer_manifest"), "tokenizer manifest"
    )
    tokenizer_manifest = read_json(
        tokenizer_manifest_path, "the tokenizer manifest"
    )
    tokenizer_manifest_sha256 = require_sha256(
        method.get("tokenizer_manifest_sha256"), "tokenizer manifest identity"
    )
    if (
        validate_internal_manifest(tokenizer_manifest, "tokenizer manifest")
        != tokenizer_manifest_sha256
        or tokenizer_manifest.get("method") != tokenizer_method
        or not (tokenizer_path / "tokenizer.json").is_file()
    ):
        raise SystemExit("The tokenizer contract changed")

    adapter_manifest = read_json(
        checkpoint_path / "tp_adapter_manifest.json", "the adapter manifest"
    )
    adapter_identity = adapter_manifest.get("identity")
    if (
        adapter_manifest.get("format") != CHECKPOINT_FORMAT
        or not isinstance(adapter_identity, dict)
        or adapter_identity.get("method_id") != method_id
        or adapter_identity.get("model_identity_sha256") != model_identity
        or adapter_identity.get("corpus_manifest_sha256")
        != corpus_manifest_hash
        or adapter_identity.get("train_sha256") != train_sha256
        or adapter_identity.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
    ):
        raise SystemExit("The selected checkpoint contract changed")
    train_run_id = require_string(adapter_identity.get("run_id"), "train run ID")
    exposure_path = checkpoint_path / "run_contract" / "training_exposure.json"
    exposure = read_json(exposure_path, "the training exposure receipt")
    validate_internal_manifest(exposure, "training exposure receipt")
    logical = exposure.get("logical_exposure")
    exposure_contract = exposure.get("exposure_contract")
    if (
        exposure.get("schema_version") != EXPOSURE_SCHEMA_VERSION
        or exposure.get("status") != "complete"
        or exposure.get("method_id") != method_id
        or not isinstance(logical, dict)
        or logical.get("all_eligible_train_rows_exposed") is not True
        or not isinstance(exposure_contract, dict)
        or exposure_contract.get("scope") != "all_eligible_train_rows"
        or exposure_contract.get("satisfied") is not True
    ):
        raise SystemExit(
            "The test requires one complete full-exposure checkpoint"
        )
    required_checkpoint_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tp_adapter_manifest.json",
    )
    if any(not (checkpoint_path / name).is_file() for name in required_checkpoint_files):
        raise SystemExit("The consolidated LoRA checkpoint is incomplete")
    custom = tokenizer_method != "plain_base_tokenizer"
    token_files = (
        checkpoint_path / "biological_token_adapter.safetensors",
        checkpoint_path / "tokenizer_manifest.json",
    )
    if custom != all(path.is_file() for path in token_files):
        raise SystemExit("The checkpoint token adapter contract changed")

    if (
        require_int(generation.get("num_nodes"), "evaluation node count") != 1
        or require_int(generation.get("tasks_per_node"), "tasks per node") != 1
        or require_int(generation.get("gpus_per_node"), "GPUs per node") != 8
        or generation.get("do_sample") is not False
        or generation.get("enable_thinking") is not False
        or generation.get("local_files_only") is not True
    ):
        raise SystemExit("The deterministic generation contract changed")
    max_new_tokens = require_int(
        generation.get("max_new_tokens"), "maximum new tokens"
    )
    max_total_tokens = require_int(
        generation.get("max_total_tokens"), "maximum total tokens"
    )
    if max_total_tokens <= max_new_tokens:
        raise SystemExit("The total token limit must exceed the new token limit")
    minimum_accuracy = require_number(
        gate.get("minimum_family_mapping_accuracy"), "family accuracy floor"
    )
    target_accuracy = require_number(
        gate.get("target_family_mapping_accuracy"), "family accuracy target"
    )
    if target_accuracy < minimum_accuracy:
        raise SystemExit("The family accuracy target is below its floor")
    if (
        tracking.get("entity") != "jail-ai"
        or tracking.get("project") != "mentor-sft"
        or tracking.get("resume_training_run") is not True
    ):
        raise SystemExit("The W&B test contract changed")

    values = {
        "S0_EVAL_CONFIG": str(config_path),
        "S0_EVAL_CONFIG_SHA256": sha256_file(config_path),
        "S0_EVALUATION_ID": require_string(
            config.get("evaluation_id"), "evaluation ID"
        ),
        "S0_METHOD_ID": method_id,
        "S0_TOKENIZER_METHOD": tokenizer_method,
        "TOKENIZER_PATH": str(tokenizer_path),
        "TOKENIZER_MANIFEST": str(tokenizer_manifest_path),
        "S0_TOKENIZER_MANIFEST_SHA256": tokenizer_manifest_sha256,
        "MODEL_ID": require_string(model.get("model_id"), "model ID"),
        "MODEL_PATH": str(model_path),
        "MODEL_IDENTITY_SHA256": model_identity,
        "BASE_MODEL_ARTIFACT_SHA256": base_artifact,
        "BASE_MODEL_WEIGHT_SHARD_COUNT": str(weight_shards),
        "S0_CORPUS_MANIFEST_SHA256": corpus_manifest_hash,
        "S0_EVALUATOR_MANIFEST": str(evaluator_manifest_path),
        "S0_EVALUATOR_MANIFEST_SHA256": evaluator_manifest_hash,
        "S0_TEST_QUESTIONS": str(questions_path),
        "S0_TEST_QUESTIONS_SHA256": questions_sha256,
        "S0_TEST_ANSWER_KEY": str(answer_key_path),
        "S0_TEST_ANSWER_KEY_SHA256": answer_key_sha256,
        "S0_TEST_PANEL_ID": test_panel_id,
        "S0_TEST_ROWS": str(panel["row_count"]),
        "CHECKPOINT_PATH": str(checkpoint_path),
        "S0_TRAIN_RUN_ID": train_run_id,
        "S0_EVAL_OUTPUT_ROOT": str(
            resolve_repo_path(root, method.get("output_root"), "output root")
        ),
        "S0_EVAL_NUM_NODES": str(generation["num_nodes"]),
        "S0_EVAL_TASKS_PER_NODE": str(generation["tasks_per_node"]),
        "S0_EVAL_GPUS_PER_NODE": str(generation["gpus_per_node"]),
        "S0_MAX_NEW_TOKENS": str(max_new_tokens),
        "S0_MAX_TOTAL_TOKENS": str(max_total_tokens),
        "S0_REASONING_EFFORT": require_string(
            generation.get("reasoning_effort"), "reasoning effort"
        ),
        "S0_EVAL_SEED": str(require_int(generation.get("seed"), "seed")),
        "S0_MINIMUM_FAMILY_ACCURACY": str(minimum_accuracy),
        "S0_TARGET_FAMILY_ACCURACY": str(target_accuracy),
        "S0_WANDB_ENTITY": "jail-ai",
        "S0_WANDB_PROJECT": "mentor-sft",
    }
    for name, value in values.items():
        print(f"{name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
