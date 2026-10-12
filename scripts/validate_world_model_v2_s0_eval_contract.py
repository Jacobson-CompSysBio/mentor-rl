#!/usr/bin/env python3
"""Validate one S0 final-checkpoint test contract and export its values.

The launcher runs this check before any test work. It verifies the config, the
base model, the corpus, the frozen full-registry panel, the tokenizer, and the
final checkpoint of one training run. It then prints shell assignments for the
launcher to source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract  # noqa: E402
from runtime.world_model.source_index import (  # noqa: E402
    check_source_index,
    content_sha256,
)


TEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-full-registry-final-test-v4"
EXPOSURE_SCHEMA_VERSION = "mentor-rl-s0-training-exposure-v2"
CHECKPOINT_FORMAT = "mentor-rl-s0-tp-lora-v1"
CORPUS_DATASET_ID = "world_model_v4_s0_human_identifiers"
LOSS_CONTRACTS = frozenset({"multiplex_sft_v3", "multiplex_sft_v4_priority"})
LORA_CONFIGURATIONS = frozenset({"lora_r32", "lora_r128"})
# The v4 test reuses the frozen v3 full-registry panel. The panel labels use
# the v4 answer rules; see the "Test" section of the S0 curriculum document.
FULL_REGISTRY_V3_EVALUATOR_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-full-registry-evaluator-v3"
)
FULL_REGISTRY_V3_EVALUATION_CONTRACT = "full_registry_multiplex_sft_v3"
FULL_REGISTRY_V3_DATASET_ID = (
    "world_model_v3_s0_human_identifiers_full_registry"
)
FULL_REGISTRY_V3_SOURCE_DATASET_ID = "world_model_v3_s0_human_identifiers"
FROZEN_FULL_REGISTRY_V3_SHA256 = {
    "evaluator_manifest": (
        "dbca23382f0881aba83a6100ea54004031337bef53415236e8f1042970c53beb"
    ),
    "questions": (
        "c9714921c9bd348051ad05452dade6bd32166102a3ae3c9fb12e118c04e41320"
    ),
    "answer_key": (
        "bc8d4c9b83781d69f8e4f0457ec9d84ddf4eefe9b3be76b940c45a5d1598b65d"
    ),
}
FULL_REGISTRY_V3_MAXIMUM_AMBIGUOUS_CANDIDATES = 49
FULL_REGISTRY_V3_LARGEST_INCLUDED_AMBIGUITY = 37
# These three ambiguous symbols each have more than 49 candidate gene IDs.
FULL_REGISTRY_V3_EXCLUDED_RECORDS = [
    {
        "candidate_gene_id_count": 170,
        "fact_id": "fact_1544b1236c875b2ad88138d0",
        "family": "human_ambiguous_symbol",
        "gene_symbol": "Metazoa_SRP",
        "record_id": "wm3_s0_417b9df7d77e7cc389a35da8",
    },
    {
        "candidate_gene_id_count": 50,
        "fact_id": "fact_39c1052bc4d239a961544728",
        "family": "human_ambiguous_symbol",
        "gene_symbol": "U3",
        "record_id": "wm3_s0_67c5581aede6c0a14a336c64",
    },
    {
        "candidate_gene_id_count": 758,
        "fact_id": "fact_fd3fc526542e105e08c060b9",
        "family": "human_ambiguous_symbol",
        "gene_symbol": "Y_RNA",
        "record_id": "wm3_s0_fc487fbf26ebafc414e677d0",
    },
]

GATE_FLOORS = (
    "minimum_valid_tool_trajectory",
    "minimum_reasoning_present",
    "minimum_exact_reasoning",
    "minimum_valid_single_tool_call",
    "minimum_exact_tool_argument",
    "minimum_family_payload_accuracy",
    "minimum_family_final_answer_accuracy",
    "minimum_ambiguous_defer_accuracy",
    "maximum_direct_answer_rate",
    "minimum_structured_json_valid",
    "minimum_status_exact",
    "minimum_result_exact",
    "minimum_continuation_decision_exact",
    "minimum_not_found_defer_accuracy",
    "minimum_unnamed_id_resolved_accuracy",
)


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
    """Validate and export one S0 final-checkpoint test contract."""

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
        or config.get("evaluation_contract")
        != FULL_REGISTRY_V3_EVALUATION_CONTRACT
    ):
        raise SystemExit("The S0 test config contract changed")
    methods = config.get("methods")
    if not isinstance(methods, list):
        raise SystemExit("The S0 test config has no methods")
    selected = [
        item
        for item in methods
        if isinstance(item, dict) and item.get("method_id") == method_id
    ]
    if len(selected) != 1:
        raise SystemExit("METHOD_ID must select one S0 test method")
    method = selected[0]
    tokenizer_method = require_string(
        method.get("tokenizer_method"), "tokenizer_method"
    )
    if tokenizer_method != "plain_base_tokenizer":
        raise SystemExit(
            "The trajectory test requires the base tokenizer"
        )
    fine_tune_configuration = method.get("fine_tune_configuration")
    if (
        method.get("checkpoint_kind") != "lora"
        or fine_tune_configuration not in LORA_CONFIGURATIONS
    ):
        raise SystemExit("The trajectory test LoRA configuration changed")
    expected_lora_rank = int(str(fine_tune_configuration).removeprefix("lora_r"))

    sections = (
        config.get("model"),
        config.get("corpus"),
        config.get("test_panel"),
        config.get("generation"),
        config.get("gate"),
        config.get("tracking"),
    )
    if not all(isinstance(value, dict) for value in sections):
        raise SystemExit("The S0 test config has an invalid section")
    model, corpus, panel, generation, gate, tracking = sections

    # Base model: the identity covers the metadata files that define it.
    model_path = resolve_model_path(root, model.get("path"))
    model_identity = require_sha256(
        model.get("model_identity_sha256"), "model identity"
    )
    if (
        model.get("base_model_artifact_schema")
        != "mentor-rl-base-model-artifact-v2"
    ):
        raise SystemExit("The base-model artifact schema changed")
    base_artifact = require_sha256(
        model.get("base_model_artifact_sha256"),
        "base-model artifact",
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

    # Training corpus: the checkpoint must come from this exact v4 corpus.
    corpus_root = resolve_repo_path(
        root, corpus.get("root"), "corpus root"
    )
    corpus_manifest_hash = require_sha256(
        corpus.get("manifest_sha256"), "corpus manifest"
    )
    corpus_manifest_path = corpus_root / "manifest.json"
    verify_file(
        corpus_manifest_path, corpus_manifest_hash, "corpus manifest"
    )
    corpus_manifest = read_json(
        corpus_manifest_path, "the corpus manifest"
    )
    eligible_train_rows = require_int(
        corpus.get("eligible_train_rows"), "eligible train rows"
    )
    train_sha256 = require_sha256(
        corpus.get("train_sha256"), "train file"
    )
    if (
        corpus_manifest.get("dataset_id") != CORPUS_DATASET_ID
        or corpus_manifest.get("schema_version")
        != "mentor-rl-world-model-s0-manifest-v2"
        or corpus_manifest.get("corpus_schema_version")
        != "mentor-rl-world-model-s0-corpus-v2"
        or corpus_manifest.get("row_counts", {}).get("train")
        != eligible_train_rows
        or corpus_manifest.get("files", {})
        .get("train.jsonl", {})
        .get("sha256")
        != train_sha256
    ):
        raise SystemExit("The corpus test contract changed")

    # Test panel: the byte-frozen full-registry panel.
    panel_row_count = require_int(
        panel.get("row_count"), "test row count"
    )
    evaluator_root = resolve_repo_path(
        root, panel.get("evaluator_root"), "evaluator root"
    )
    evaluator_manifest_path = evaluator_root / require_string(
        panel.get("evaluator_manifest"), "evaluator manifest name"
    )
    evaluator_manifest_hash = require_sha256(
        panel.get("evaluator_manifest_sha256"),
        "evaluator manifest",
    )
    verify_file(
        evaluator_manifest_path,
        evaluator_manifest_hash,
        "evaluator manifest",
    )
    evaluator = read_json(
        evaluator_manifest_path, "the evaluator manifest"
    )
    test = evaluator.get("test")
    if (
        evaluator.get("schema_version")
        != FULL_REGISTRY_V3_EVALUATOR_SCHEMA_VERSION
        or evaluator.get("dataset_id") != FULL_REGISTRY_V3_DATASET_ID
        or evaluator.get("source_dataset_id")
        != FULL_REGISTRY_V3_SOURCE_DATASET_ID
        or evaluator.get("evaluation_contract")
        != FULL_REGISTRY_V3_EVALUATION_CONTRACT
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
    if {
        "evaluator_manifest": evaluator_manifest_hash,
        "questions": questions_sha256,
        "answer_key": answer_key_sha256,
    } != FROZEN_FULL_REGISTRY_V3_SHA256:
        raise SystemExit("The byte-frozen full-registry gate identity changed")
    verify_file(questions_path, questions_sha256, "test questions")
    verify_file(answer_key_path, answer_key_sha256, "test answer key")
    test_panel_id = require_sha256(
        panel.get("test_panel_id"), "test panel ID"
    )
    if (
        test.get("questions_sha256") != questions_sha256
        or test.get("answer_key_sha256") != answer_key_sha256
        or test.get("test_panel_id") != test_panel_id
        or test.get("row_count") != panel_row_count
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
        method.get("tokenizer_manifest_sha256"),
        "tokenizer manifest identity",
    )
    if (
        validate_internal_manifest(
            tokenizer_manifest, "tokenizer manifest"
        )
        != tokenizer_manifest_sha256
        or tokenizer_manifest.get("method") != tokenizer_method
        or not (tokenizer_path / "tokenizer.json").is_file()
    ):
        raise SystemExit("The tokenizer contract changed")

    # Checkpoint: the final adapter of one complete v4 training run.
    adapter_manifest = read_json(
        checkpoint_path / "tp_adapter_manifest.json",
        "the adapter manifest",
    )
    adapter_identity = adapter_manifest.get("identity")
    if (
        adapter_manifest.get("format") != CHECKPOINT_FORMAT
        or not isinstance(adapter_identity, dict)
        or adapter_identity.get("method_id") != method_id
        or adapter_identity.get("model_identity_sha256")
        != model_identity
        or adapter_identity.get("corpus_manifest_sha256")
        != corpus_manifest_hash
        or adapter_identity.get("train_sha256") != train_sha256
        or adapter_identity.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
    ):
        raise SystemExit("The selected checkpoint contract changed")
    if (
        adapter_manifest.get("objective", {}).get("loss_contract")
        not in LOSS_CONTRACTS
        or adapter_manifest.get("lora", {}).get("r") != expected_lora_rank
    ):
        raise SystemExit("The checkpoint loss contract changed")
    train_run_id = require_string(
        adapter_identity.get("run_id"), "train run ID"
    )
    exposure = read_json(
        checkpoint_path / "run_contract" / "training_exposure.json",
        "the training exposure receipt",
    )
    validate_internal_manifest(exposure, "training exposure receipt")
    logical = exposure.get("logical_exposure")
    exposure_contract = exposure.get("exposure_contract")
    if (
        exposure.get("schema_version") != EXPOSURE_SCHEMA_VERSION
        or exposure.get("status") != "complete"
        or exposure.get("method_id") != method_id
        or not isinstance(logical, dict)
        or not isinstance(exposure_contract, dict)
        or exposure_contract.get("satisfied") is not True
    ):
        raise SystemExit("The test requires one complete checkpoint")
    required_checkpoint_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tp_adapter_manifest.json",
    )
    if any(
        not (checkpoint_path / name).is_file()
        for name in required_checkpoint_files
    ):
        raise SystemExit("The consolidated LoRA checkpoint is incomplete")

    num_nodes = require_int(
        generation.get("num_nodes"), "evaluation node count"
    )
    tasks_per_node = require_int(
        generation.get("tasks_per_node"), "tasks per node"
    )
    gpus_per_node = require_int(
        generation.get("gpus_per_node"), "GPUs per node"
    )
    if (
        tasks_per_node != 1
        or gpus_per_node != 8
        or generation.get("do_sample") is not False
        or generation.get("enable_thinking") is not True
        or generation.get("local_files_only") is not True
    ):
        raise SystemExit(
            "The deterministic generation contract changed"
        )
    batch_size = require_int(
        generation.get("batch_size"), "generation batch size"
    )
    max_new_tokens = require_int(
        generation.get("max_new_tokens"), "maximum new tokens"
    )
    max_total_tokens = require_int(
        generation.get("max_total_tokens"), "maximum total tokens"
    )
    max_tool_call_tokens = require_int(
        generation.get("max_tool_call_tokens"),
        "maximum tool-call tokens",
    )
    max_final_prompt_tokens = require_int(
        generation.get("max_final_prompt_tokens"),
        "maximum final prompt tokens",
    )
    if (
        max_total_tokens <= max_new_tokens
        or max_new_tokens < 256
        or max_tool_call_tokens > max_new_tokens
        or max_final_prompt_tokens >= max_total_tokens
    ):
        raise SystemExit(
            "The trajectory token limits are invalid"
        )
    reasoning_effort = require_string(
        generation.get("reasoning_effort"), "reasoning effort"
    )
    if reasoning_effort != "low":
        raise SystemExit(
            "The trajectory test requires low reasoning effort"
        )

    # Gate: the config must set every floor; see the S0 document.
    gate_floors = {
        name: require_number(gate.get(name), name)
        for name in GATE_FLOORS
    }
    if gate.get("require_network_used_false") is not True:
        raise SystemExit(
            "The trajectory test must disable network access"
        )

    registry = config.get("identifier_registry")
    evaluator_registry = evaluator.get("identifier_registry")
    if not isinstance(registry, dict) or not isinstance(
        evaluator_registry, dict
    ):
        raise SystemExit(
            "The trajectory test lacks the registry contract"
        )
    registry_path = resolve_repo_path(
        root, registry.get("path"), "identifier registry"
    )
    registry_sha256 = require_sha256(
        registry.get("sha256"), "identifier registry"
    )
    source_index = check_source_index(registry_path, load_contract())
    if content_sha256(source_index) != registry_sha256:
        raise SystemExit("The v3 source index identity changed")
    if (
        evaluator_registry.get("path") != registry.get("path")
        or evaluator_registry.get("sha256") != registry_sha256
    ):
        raise SystemExit("The evaluator registry identity changed")

    # The panel covers every registry ID and symbol, except the long
    # ambiguities, so its counts follow from the source index.
    counts = source_index.get("counts")
    population = evaluator.get("population")
    if not isinstance(counts, dict) or not isinstance(population, dict):
        raise SystemExit("The full-registry v3 counts are absent")
    gene_count = require_int(
        counts.get("registry_gene_count"), "registry gene count"
    )
    symbol_count = require_int(
        counts.get("registry_symbol_count"), "registry symbol count"
    )
    ambiguous_count = require_int(
        counts.get("registry_ambiguous_symbol_count"),
        "registry ambiguous-symbol count",
    )
    unnamed_count = require_int(
        counts.get("registry_unnamed_gene_count"),
        "registry unnamed-gene count",
    )
    largest_candidates = require_int(
        counts.get("registry_largest_candidate_count"),
        "largest registry candidate count",
    )
    excluded_records = FULL_REGISTRY_V3_EXCLUDED_RECORDS
    excluded_count = len(excluded_records)
    maximum_candidates = FULL_REGISTRY_V3_MAXIMUM_AMBIGUOUS_CANDIDATES
    expected_rows = gene_count + symbol_count - excluded_count
    expected_family_counts = {
        "human_ambiguous_symbol": ambiguous_count - excluded_count,
        "human_ensembl_to_symbol": gene_count,
        "human_symbol_to_ensembl": symbol_count - ambiguous_count,
    }
    expected_status_counts = {
        "ambiguous": ambiguous_count - excluded_count,
        "resolved": expected_rows - ambiguous_count + excluded_count,
    }
    expected_population_filter = {
        "maximum_ambiguous_candidate_gene_ids": maximum_candidates,
    }
    expected_panel_filter = {
        **expected_population_filter,
        "excluded_records": excluded_records,
    }
    if (
        panel_row_count != expected_rows
        or test.get("population_filter") != expected_panel_filter
        or population.get("scope")
        != (
            "all_ensembl_registry_gene_ids_and_symbols_except_declared_"
            "long_ambiguities"
        )
        or population.get("row_count") != expected_rows
        or population.get("family_counts") != expected_family_counts
        or population.get("status_counts") != expected_status_counts
        or population.get("registry_gene_id_count") != gene_count
        or population.get("registry_symbol_count") != symbol_count
        or population.get("registry_ambiguous_symbol_count")
        != ambiguous_count
        or population.get("registry_unnamed_gene_id_count")
        != unnamed_count
        or population.get("registry_largest_ambiguous_candidate_count")
        != largest_candidates
        or population.get("largest_included_ambiguous_candidate_count")
        != FULL_REGISTRY_V3_LARGEST_INCLUDED_AMBIGUITY
        or population.get("population_filter")
        != expected_population_filter
        or population.get("excluded_row_count") != excluded_count
        or population.get("excluded_records") != excluded_records
    ):
        raise SystemExit("The full-registry v3 panel scope changed")

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
        "S0_EVALUATION_CONTRACT": FULL_REGISTRY_V3_EVALUATION_CONTRACT,
        "S0_METHOD_ID": method_id,
        "S0_TOKENIZER_METHOD": tokenizer_method,
        "TOKENIZER_PATH": str(tokenizer_path),
        "TOKENIZER_MANIFEST": str(tokenizer_manifest_path),
        "S0_TOKENIZER_MANIFEST_SHA256": (
            tokenizer_manifest_sha256
        ),
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
        "S0_TEST_ROWS": str(panel_row_count),
        "CHECKPOINT_PATH": str(checkpoint_path),
        "S0_TRAIN_RUN_ID": train_run_id,
        "S0_EVAL_OUTPUT_ROOT": str(
            resolve_repo_path(
                root, method.get("output_root"), "output root"
            )
        ),
        "S0_EVAL_NUM_NODES": str(num_nodes),
        "S0_EVAL_TASKS_PER_NODE": str(tasks_per_node),
        "S0_EVAL_WORLD_SIZE": str(num_nodes * tasks_per_node),
        "S0_EVAL_GPUS_PER_NODE": str(gpus_per_node),
        "S0_MAX_NEW_TOKENS": str(max_new_tokens),
        "S0_MAX_TOTAL_TOKENS": str(max_total_tokens),
        "S0_MAX_TOOL_CALL_TOKENS": str(max_tool_call_tokens),
        "S0_MAX_FINAL_PROMPT_TOKENS": str(max_final_prompt_tokens),
        "S0_REASONING_EFFORT": reasoning_effort,
        "S0_ENABLE_THINKING": "1",
        "S0_EVAL_SEED": str(
            require_int(generation.get("seed"), "seed")
        ),
        "S0_EVAL_BATCH_SIZE": str(batch_size),
        **{
            f"S0_{name.upper()}": str(value)
            for name, value in gate_floors.items()
        },
        "S0_IDENTIFIER_REGISTRY": str(registry_path),
        "S0_IDENTIFIER_REGISTRY_SHA256": registry_sha256,
        "S0_WANDB_ENTITY": "jail-ai",
        "S0_WANDB_PROJECT": "mentor-sft",
    }
    for name, value in values.items():
        print(f"{name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
