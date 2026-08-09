#!/usr/bin/env python3
"""Generate deterministic S0 test predictions without answer-key access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_training import (  # noqa: E402
    build_world_model_prompt_messages,
    load_model_text_codec_for_token_manifest,
    tokenizer_artifact_hashes,
    validated_tokenizer_manifest,
)
from scripts.build_world_model_v2_s0_generation_bundle import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    QUESTION_KEYS,
)
from scripts.evaluate_world_model_v2_s0 import (  # noqa: E402
    PREDICTION_SCHEMA_VERSION,
)


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
ALLOWED_TERMINAL_SPECIAL_TOKENS = frozenset({"<|return|>", "<|end|>"})
GPT_OSS_FINAL_PROMPT_SUFFIX = (
    "<|start|>assistant<|channel|>final<|message|>"
)
EXPOSURE_SCHEMA_VERSION = "mentor-rl-s0-training-exposure-v2"
CHECKPOINT_FORMAT = "mentor-rl-s0-tp-lora-v1"


class S0GenerationError(RuntimeError):
    """Report one invalid S0 generation input."""


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


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S0GenerationError(
            f"Could not read one JSON object from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise S0GenerationError(f"Expected one JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise S0GenerationError(
                        f"Expected one JSON object at {path}:{line_number}"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise S0GenerationError(
            f"Could not read JSONL from {path}: {error}"
        ) from error
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o640)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write stable JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o640)


def load_bundle(bundle_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one valid answer-free test bundle."""

    manifest = read_json(bundle_root / "manifest.json")
    claimed = manifest.get("bundle_sha256")
    identity = {
        str(key): value
        for key, value in manifest.items()
        if key != "bundle_sha256"
    }
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or manifest.get("reads_private_answer_keys") is not False
    ):
        raise S0GenerationError("The generation bundle identity changed")
    questions_path = bundle_root / "questions.jsonl"
    if sha256_file(questions_path) != manifest.get("questions_sha256"):
        raise S0GenerationError("The bundled questions changed")
    rows = read_jsonl(questions_path)
    record_ids = []
    for row in rows:
        if set(row) != QUESTION_KEYS or "answer" in row:
            raise S0GenerationError("A bundled question has unsafe fields")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise S0GenerationError("A bundled record ID is invalid")
        record_ids.append(record_id)
    if (
        len(rows) != manifest.get("record_count")
        or len(set(record_ids)) != len(record_ids)
        or stable_sha256(record_ids) != manifest.get("record_ids_sha256")
    ):
        raise S0GenerationError("The generation record set changed")
    return manifest, rows


def checkpoint_artifact_identity(checkpoint_path: Path) -> dict[str, Any]:
    """Return the exact inference checkpoint identity."""

    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tp_adapter_manifest.json",
        "run_contract/training_exposure.json",
    ]
    optional_pair = [
        "biological_token_adapter.safetensors",
        "tokenizer_manifest.json",
    ]
    present = [(checkpoint_path / name).is_file() for name in optional_pair]
    if any(present) and not all(present):
        raise S0GenerationError("The token adapter checkpoint is incomplete")
    if all(present):
        required.extend(optional_pair)
    files = {}
    for name in required:
        path = checkpoint_path / name
        if not path.is_file():
            raise S0GenerationError(f"A checkpoint file is absent: {name}")
        files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {"files": files}
    payload["checkpoint_identity_sha256"] = stable_sha256(payload)
    return payload


def validate_checkpoint(
    checkpoint_path: Path,
    *,
    method_id: str,
    corpus_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    model_identity_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify one complete full-exposure LoRA checkpoint."""

    adapter = read_json(checkpoint_path / "tp_adapter_manifest.json")
    identity = adapter.get("identity")
    if (
        adapter.get("format") != CHECKPOINT_FORMAT
        or not isinstance(identity, Mapping)
        or identity.get("method_id") != method_id
        or identity.get("corpus_manifest_sha256")
        != corpus_manifest_sha256
        or identity.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
        or identity.get("model_identity_sha256") != model_identity_sha256
    ):
        raise S0GenerationError("The checkpoint contract changed")
    exposure = read_json(
        checkpoint_path / "run_contract" / "training_exposure.json"
    )
    claimed = exposure.get("manifest_sha256")
    exposure_identity = {
        str(key): value
        for key, value in exposure.items()
        if key != "manifest_sha256"
    }
    logical = exposure.get("logical_exposure")
    contract = exposure.get("exposure_contract")
    if (
        exposure.get("schema_version") != EXPOSURE_SCHEMA_VERSION
        or exposure.get("status") != "complete"
        or exposure.get("method_id") != method_id
        or not isinstance(claimed, str)
        or stable_sha256(exposure_identity) != claimed
        or not isinstance(logical, Mapping)
        or logical.get("all_eligible_train_rows_exposed") is not True
        or not isinstance(contract, Mapping)
        or contract.get("scope") != "all_eligible_train_rows"
        or contract.get("satisfied") is not True
    ):
        raise S0GenerationError(
            "The test requires one complete full-exposure checkpoint"
        )
    artifact_identity = checkpoint_artifact_identity(checkpoint_path)
    return artifact_identity, str(identity.get("run_id"))


def clean_decoded_generation(
    raw: str,
    *,
    eos_token: str | None,
    pad_token: str | None,
) -> tuple[str, list[str]]:
    """Remove only valid terminal special tokens."""

    terminal_tokens = set(ALLOWED_TERMINAL_SPECIAL_TOKENS)
    if eos_token:
        terminal_tokens.add(str(eos_token))
    if pad_token:
        terminal_tokens.add(str(pad_token))
    special_tokens = SPECIAL_TOKEN_RE.findall(raw)
    disallowed = sorted(
        {token for token in special_tokens if token not in terminal_tokens}
    )
    prediction = raw.strip()
    ordered = sorted(
        {token for token in terminal_tokens if token},
        key=len,
        reverse=True,
    )
    while True:
        for token in ordered:
            if prediction.endswith(token):
                prediction = prediction[: -len(token)].rstrip()
                break
        else:
            break
    return prediction, disallowed


def load_backend(
    *,
    base_model_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    local_files_only: bool,
):
    """Load one consolidated LoRA checkpoint for inference."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from runtime.world_model_token_adapter import (
        load_token_adapter_for_inference,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=local_files_only,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        str(checkpoint_path),
        local_files_only=local_files_only,
        is_trainable=False,
    )
    load_token_adapter_for_inference(model, checkpoint_path)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_predictions(
    *,
    bundle_root: Path,
    base_model_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    tokenizer_manifest_path: Path,
    output_dir: Path,
    method_id: str,
    corpus_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    model_identity_sha256: str,
    max_new_tokens: int,
    max_total_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str,
    seed: int,
    local_files_only: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate and record one complete S0 test panel."""

    if max_new_tokens < 1 or max_total_tokens <= max_new_tokens:
        raise S0GenerationError("The generation token limits are invalid")
    bundle, rows = load_bundle(bundle_root.resolve())
    tokenizer_manifest = validated_tokenizer_manifest(
        tokenizer_manifest_path.resolve()
    )
    if tokenizer_manifest.get("manifest_sha256") != tokenizer_manifest_sha256:
        raise S0GenerationError("The tokenizer manifest identity changed")
    tokenizer_files = tokenizer_artifact_hashes(tokenizer_path.resolve())
    checkpoint, train_run_id = validate_checkpoint(
        checkpoint_path.resolve(),
        method_id=method_id,
        corpus_manifest_sha256=corpus_manifest_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        model_identity_sha256=model_identity_sha256,
    )
    generation_config = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "max_total_tokens": max_total_tokens,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "seed": seed,
    }
    plan = {
        "schema_version": "mentor-rl-world-model-s0-generation-plan-v4",
        "test_panel_id": bundle["test_panel_id"],
        "record_count": len(rows),
        "method_id": method_id,
        "train_run_id": train_run_id,
        "base_model_identity_sha256": model_identity_sha256,
        "checkpoint_identity_sha256": checkpoint[
            "checkpoint_identity_sha256"
        ],
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_artifact_hashes": tokenizer_files,
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "reads_private_answer_keys": False,
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(output_dir / "generation_plan.json", plan)
    if dry_run:
        return {"status": "dry_run", "plan": plan}

    import torch

    torch.manual_seed(seed)
    model, tokenizer = load_backend(
        base_model_path=base_model_path.resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        tokenizer_path=tokenizer_path.resolve(),
        local_files_only=local_files_only,
    )
    codec = load_model_text_codec_for_token_manifest(
        tokenizer_manifest_path.resolve()
    )
    device = next(model.parameters()).device
    predictions: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        messages = build_world_model_prompt_messages(
            system=str(row["system"]),
            question=str(row["question"]),
            metadata=row["metadata"],
            context=row.get("context"),
        )
        if codec is not None:
            messages = codec.encode_messages(
                messages,
                question_family=str(
                    row["metadata"]["question_family"]
                ),
            )
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        if not enable_thinking and not prompt.endswith(
            GPT_OSS_FINAL_PROMPT_SUFFIX
        ):
            raise S0GenerationError(
                "The test prompt lacks the final assistant boundary"
            )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        remaining = max_total_tokens - prompt_tokens
        if remaining < 1:
            raise S0GenerationError("A test prompt exceeds max_total_tokens")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, remaining),
                do_sample=False,
                eos_token_id=(
                    model.generation_config.eos_token_id
                    if model.generation_config.eos_token_id is not None
                    else tokenizer.eos_token_id
                ),
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[0][prompt_tokens:]
        raw = tokenizer.decode(generated, skip_special_tokens=False)
        prediction, disallowed = clean_decoded_generation(
            raw,
            eos_token=tokenizer.eos_token,
            pad_token=tokenizer.pad_token,
        )
        predictions.append(
            {
                "record_id": row["record_id"],
                "encoded_prediction": prediction,
                "raw_generation": raw,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(generated.shape[0]),
                "disallowed_special_tokens": disallowed,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "event": "test_generation_progress",
                        "completed": index,
                        "total": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    manifest = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "test_panel_id": bundle["test_panel_id"],
        "record_count": len(predictions),
        "method_id": method_id,
        "train_run_id": train_run_id,
        "base_model_identity_sha256": model_identity_sha256,
        "checkpoint_identity_sha256": checkpoint[
            "checkpoint_identity_sha256"
        ],
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": time.monotonic() - started,
        "reads_private_answer_keys": False,
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "generation_manifest.json", manifest)
    return {"status": "complete", "manifest": manifest}


def parse_args() -> argparse.Namespace:
    """Parse command-line values."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-bundle", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--corpus-manifest-sha256", required=True)
    parser.add_argument("--tokenizer-manifest-sha256", required=True)
    parser.add_argument("--model-identity-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Generate one exact S0 test result."""

    args = parse_args()
    result = generate_predictions(
        bundle_root=args.generation_bundle,
        base_model_path=args.base_model_path,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        tokenizer_manifest_path=args.tokenizer_manifest,
        output_dir=args.output_dir,
        method_id=args.method_id,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
        tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
        model_identity_sha256=args.model_identity_sha256,
        max_new_tokens=args.max_new_tokens,
        max_total_tokens=args.max_total_tokens,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        local_files_only=not args.allow_remote_files,
        dry_run=args.dry_run,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
