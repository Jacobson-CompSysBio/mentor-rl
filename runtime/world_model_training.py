"""Prepare the canonical S0 tool trajectory data and exposure receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

from runtime.world_model_prompts import s0_tool_trajectory_prompt_contract
from runtime.world_model_schemas import S0_FAMILIES, IdentifierToolSFTRecord


TOKENIZER_ARTIFACT_NAMES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "vocab.json",
    }
)
TOKENIZER_ARTIFACT_PREFIXES = ("chat_template", "tokenizer")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

S0_TOOL_TRAJECTORY_LOSS_CONTRACT = "s0_tool_trajectory_v1"
S0_TOOL_TRAJECTORY_DATASET_ID = (
    "world_model_v2_s0_human_identifier_trajectories_v6"
)
S0_EXPOSURE_SCOPE_BY_RUN_SCOPE = {
    "qualification": "unrestricted",
    "production": "all_eligible_train_rows",
}
S0_EXPOSURE_CORPUS_BY_LOSS_CONTRACT = {
    S0_TOOL_TRAJECTORY_LOSS_CONTRACT: {
        "dataset_id": S0_TOOL_TRAJECTORY_DATASET_ID,
        "system_prompt_sha256": (
            s0_tool_trajectory_prompt_contract().system_prompt_sha256
        ),
    }
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for one value."""

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
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject another top-level type."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    """Return one valid lowercase SHA-256 value."""

    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 value")
    return value


def tokenizer_artifact_hashes(path: Path) -> dict[str, str]:
    """Return hashes for all tokenizer files in one local directory."""

    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"The tokenizer path is not a directory: {root}")
    artifacts = {
        item.name: sha256_file(item)
        for item in sorted(root.iterdir())
        if item.is_file()
        and (
            item.name in TOKENIZER_ARTIFACT_NAMES
            or item.name.startswith(TOKENIZER_ARTIFACT_PREFIXES)
        )
    }
    if not artifacts:
        raise ValueError(f"The tokenizer path has no tokenizer files: {root}")
    return artifacts


def validated_tokenizer_manifest(path: Path) -> dict[str, Any]:
    """Check one tokenizer manifest, and then return it."""

    payload = read_json_object(path)
    claimed = _require_sha256(
        payload.get("manifest_sha256"),
        "tokenizer manifest_sha256",
    )
    identity = {
        str(key): value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if stable_sha256(identity) != claimed:
        raise ValueError("The tokenizer manifest failed its identity check")
    return payload


def flatten_tool_sft_record_for_arrow(
    record: Mapping[str, Any],
    *,
    expected_split: str = "train",
) -> dict[str, str]:
    """Convert one S0 trajectory record to a stable scalar schema."""

    parsed = IdentifierToolSFTRecord.from_dict(record)
    if parsed.split != expected_split:
        raise ValueError(
            f"The S0 trajectory adapter requires split {expected_split!r}"
        )
    provenance = parsed.provenance
    fact_id = provenance.get("fact_id")
    if not isinstance(fact_id, str) or not fact_id:
        raise ValueError("The S0 trajectory record requires one fact_id")
    prompt_form_id = {
        "train": "train",
        "val": "validation",
    }.get(expected_split)
    if prompt_form_id is None:
        raise ValueError(
            f"The S0 trajectory split is not supported: {expected_split!r}"
        )
    if provenance.get("prompt_form_id") != prompt_form_id:
        raise ValueError("The S0 prompt form differs from its split")
    expected_role = "train" if expected_split == "train" else "unseen"
    if provenance.get("fact_role") != expected_role:
        raise ValueError("The S0 fact role differs from its split")
    return {
        "system": parsed.system,
        "question": parsed.question,
        "input_json": canonical_json(dict(parsed.input)),
        "tools_json": canonical_json(list(parsed.tools)),
        "assistant_tool_call_json": canonical_json(
            dict(parsed.assistant_tool_call)
        ),
        "assistant_thinking": parsed.assistant_thinking or "",
        "tool_result_json": canonical_json(dict(parsed.tool_result or {})),
        "assistant_final": parsed.assistant_final or "",
        "metadata_json": canonical_json(parsed.metadata.to_dict()),
        "question_family": parsed.metadata.question_family,
        "record_id": parsed.record_id,
        "fact_id": fact_id,
        "prompt_form_id": prompt_form_id,
        "split": expected_split,
    }


def normalize_token_ids(value: Any) -> list[int]:
    """Return one flat token sequence from one tokenizer result."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("The tokenizer result has no input_ids field")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("Tokenizer input_ids must be one sequence")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected one token sequence, but received a batch")
        value = value[0]
    if any(isinstance(item, (list, tuple, Mapping)) for item in value):
        raise ValueError("Tokenizer input_ids must be one flat sequence")
    token_ids = [int(item) for item in value]
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("Tokenizer input_ids cannot contain negative values")
    return token_ids


def tokenize_tool_trajectory_for_sft(
    record: IdentifierToolSFTRecord,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]]:
    """Render one trajectory and mask non-assistant message tokens."""

    messages = record.to_messages()
    if len(messages) != 5:
        raise ValueError("A tool trajectory must contain five messages")
    template_args = {
        "tools": list(record.tools),
        "enable_thinking": True,
        "reasoning_effort": "low",
    }

    def render(selected: list[dict[str, Any]]) -> list[int]:
        return normalize_token_ids(
            tokenizer.apply_chat_template(
                selected,
                tokenize=True,
                add_generation_prompt=False,
                **template_args,
            )
        )

    prompt_ids = render(messages[:2])
    assistant_call_ids = render(messages[:3])
    through_tool_ids = render(messages[:4])
    if assistant_call_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "The trajectory prompt is not a prefix of the tool call"
        )
    if through_tool_ids[: len(assistant_call_ids)] != assistant_call_ids:
        raise RuntimeError(
            "The assistant tool call is not a prefix of the tool result"
        )

    messages_without_thinking = [dict(message) for message in messages]
    messages_without_thinking[2].pop("thinking", None)
    plain_tool_ids = render(messages_without_thinking[:4])
    plain_full_ids = render(messages_without_thinking)
    if plain_full_ids[: len(plain_tool_ids)] != plain_tool_ids:
        raise RuntimeError(
            "The tool result is not a prefix of the final answer"
        )
    final_ids = plain_full_ids[len(plain_tool_ids) :]
    if not final_ids:
        raise RuntimeError("The trajectory has no final answer tokens")

    full_ids = through_tool_ids + final_ids
    if len(full_ids) > max_length:
        raise RuntimeError(
            f"Tokenized trajectory has {len(full_ids)} tokens, exceeding "
            f"--max_length={max_length}"
        )
    completion_mask = (
        [0] * len(prompt_ids)
        + [1] * (len(assistant_call_ids) - len(prompt_ids))
        + [0] * (len(through_tool_ids) - len(assistant_call_ids))
        + [1] * len(final_ids)
    )
    if len(completion_mask) != len(full_ids):
        raise RuntimeError("The trajectory loss mask has an invalid length")
    return {
        "input_ids": full_ids,
        "completion_mask": completion_mask,
    }


def epoch_training_indices(
    dataset_size: int,
    seed: int,
    preserve_order: bool,
) -> list[int]:
    """Return one explicit epoch order."""

    if (
        not isinstance(dataset_size, int)
        or isinstance(dataset_size, bool)
        or dataset_size < 1
    ):
        raise ValueError("dataset_size must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(preserve_order, bool):
        raise ValueError("preserve_order must be a Boolean")
    indices = list(range(dataset_size))
    if not preserve_order:
        random.Random(seed).shuffle(indices)
    return indices


def epoch_training_indices_with_replica_padding(
    dataset_size: int,
    seed: int,
    preserve_order: bool,
    replica_count: int,
) -> tuple[list[int], list[int]]:
    """Add one local repeat for each short strided replica shard."""

    if (
        not isinstance(replica_count, int)
        or isinstance(replica_count, bool)
        or replica_count < 1
    ):
        raise ValueError("replica_count must be positive")
    order = epoch_training_indices(dataset_size, seed, preserve_order)
    remainder = dataset_size % replica_count
    if remainder == 0:
        return order, []
    if dataset_size < replica_count:
        raise ValueError(
            "dataset_size must be at least replica_count when padding is required"
        )
    padding = order[remainder:replica_count]
    expected_padding = replica_count - remainder
    if len(padding) != expected_padding:
        raise RuntimeError(
            f"Replica padding produced {len(padding)} rows, "
            f"but expected {expected_padding}"
        )
    return order + padding, padding


def derive_optimizer_step_schedule(
    dataset_size: int,
    *,
    num_train_epochs: int,
    replica_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, int]:
    """Calculate the optimizer step schedule for complete epochs."""

    for name, value in (
        ("dataset_size", dataset_size),
        ("num_train_epochs", num_train_epochs),
        ("replica_count", replica_count),
        ("per_device_train_batch_size", per_device_train_batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive")
    if dataset_size < replica_count:
        raise ValueError("dataset_size must be at least replica_count")

    per_replica_rows = math.ceil(dataset_size / replica_count)
    batches_per_epoch = math.ceil(
        per_replica_rows / per_device_train_batch_size
    )
    updates_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    return {
        "dataset_size": dataset_size,
        "num_train_epochs": num_train_epochs,
        "replica_count": replica_count,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "global_batch_size": (
            replica_count
            * per_device_train_batch_size
            * gradient_accumulation_steps
        ),
        "per_replica_rows": per_replica_rows,
        "batches_per_epoch": batches_per_epoch,
        "updates_per_epoch": updates_per_epoch,
        "total_steps": updates_per_epoch * num_train_epochs,
    }


def consumed_training_index_plan(
    dataset_size: int,
    *,
    total_steps: int,
    num_train_epochs: int,
    replica_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    seed: int,
    preserve_order: bool,
) -> dict[str, Any]:
    """Separate logical exposure from distributed padding."""

    schedule = derive_optimizer_step_schedule(
        dataset_size,
        num_train_epochs=num_train_epochs,
        replica_count=replica_count,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if (
        not isinstance(total_steps, int)
        or isinstance(total_steps, bool)
        or total_steps < 1
    ):
        raise ValueError("total_steps must be positive")
    if total_steps > schedule["total_steps"]:
        raise ValueError(
            f"total_steps={total_steps} exceeds the "
            f"{schedule['total_steps']} updates available within "
            f"num_train_epochs={num_train_epochs}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(preserve_order, bool):
        raise ValueError("preserve_order must be a Boolean")

    logical_indices: list[int] = []
    padding_indices: list[int] = []
    completed_steps = 0
    steps_per_epoch = 0
    padded_rows_per_epoch = 0
    for epoch in range(num_train_epochs):
        padded_order, epoch_padding = epoch_training_indices_with_replica_padding(
            dataset_size,
            seed + epoch,
            preserve_order,
            replica_count,
        )
        rows_per_update = (
            per_device_train_batch_size * gradient_accumulation_steps
        )
        steps_per_epoch = schedule["updates_per_epoch"]
        padded_rows_per_epoch = len(padded_order)
        steps_in_epoch = min(
            steps_per_epoch,
            total_steps - completed_steps,
        )
        physical_limit = min(
            padded_rows_per_epoch,
            steps_in_epoch * replica_count * rows_per_update,
        )
        logical_limit = min(dataset_size, physical_limit)
        logical_indices.extend(padded_order[:logical_limit])
        padding_limit = max(0, physical_limit - dataset_size)
        padding_indices.extend(epoch_padding[:padding_limit])
        completed_steps += steps_in_epoch
        if completed_steps == total_steps:
            return {
                "logical_indices": logical_indices,
                "distributed_padding_indices": padding_indices,
                "completed_steps": completed_steps,
                "steps_per_epoch": steps_per_epoch,
                "logical_rows_per_epoch": dataset_size,
                "padded_rows_per_epoch": padded_rows_per_epoch,
                "padding_rows_per_epoch": padded_rows_per_epoch - dataset_size,
                "physical_record_occurrences": (
                    len(logical_indices) + len(padding_indices)
                ),
                "padding_policy": (
                    "repeat_first_local_item_for_short_strided_replica_shards"
                ),
            }
    raise ValueError(
        f"total_steps={total_steps} exceeds the {completed_steps} updates "
        f"available within num_train_epochs={num_train_epochs}"
    )


def _sequence_sha256(values: list[str]) -> str:
    """Return the identity of one ordered string sequence."""

    return stable_sha256(values)


def s0_exposure_scope(run_scope: str) -> str:
    """Return the exposure scope for one valid S0 run scope."""

    try:
        return S0_EXPOSURE_SCOPE_BY_RUN_SCOPE[run_scope]
    except KeyError as error:
        raise ValueError(
            f"The S0 run scope is invalid: {run_scope!r}"
        ) from error


def s0_exposure_corpus_identity(loss_contract: str) -> dict[str, str]:
    """Return the corpus identity for the S0 trajectory loss."""

    try:
        return dict(S0_EXPOSURE_CORPUS_BY_LOSS_CONTRACT[loss_contract])
    except KeyError as error:
        raise ValueError(
            f"The S0 loss contract is invalid: {loss_contract!r}"
        ) from error


def build_training_exposure_manifest(
    *,
    run_id: str,
    method_id: str,
    run_config_sha256: str,
    corpus_manifest_sha256: str,
    train_sha256: str,
    tokenizer_arm_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    record_ids: list[str],
    fact_ids: list[str],
    question_families: list[str],
    prompt_form_ids: list[str],
    consumed_indices: list[int],
    distributed_padding_indices: list[int],
    seed: int,
    total_steps: int,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    data_parallel_size: int,
    preserve_order: bool,
    padding_policy: str,
    exposure_scope: str,
    status: str,
    completed_global_step: int | None = None,
    dataset_id: str = S0_TOOL_TRAJECTORY_DATASET_ID,
    system_prompt_sha256: str = (
        s0_tool_trajectory_prompt_contract().system_prompt_sha256
    ),
) -> dict[str, Any]:
    """Build one compact content-addressed S0 exposure receipt."""

    for label, value in (("run_id", run_id), ("method_id", method_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty string")
    identities = {
        "run_config_sha256": run_config_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "train_sha256": train_sha256,
        "tokenizer_arm_manifest_sha256": tokenizer_arm_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
    }
    row_count = len(record_ids)
    if row_count < 1:
        raise ValueError("The S0 exposure receipt requires train rows")
    parallel_values = fact_ids, question_families, prompt_form_ids
    if any(len(values) != row_count for values in parallel_values):
        raise ValueError("The S0 train identity lists have different lengths")
    if len(set(record_ids)) != row_count:
        raise ValueError("S0 train record_ids must be unique")
    if any(not isinstance(value, str) or not value for value in record_ids):
        raise ValueError("Each S0 record_id must be a nonempty string")
    if any(not isinstance(value, str) or not value for value in fact_ids):
        raise ValueError("Each S0 fact_id must be a nonempty string")
    if any(family not in S0_FAMILIES for family in question_families):
        raise ValueError("The S0 exposure receipt has an unknown family")
    if set(prompt_form_ids) != {"train"}:
        raise ValueError("The S0 exposure receipt requires train prompt forms")

    for name, value in (
        ("total_steps", total_steps),
        ("num_train_epochs", num_train_epochs),
        ("per_device_train_batch_size", per_device_train_batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("data_parallel_size", data_parallel_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(preserve_order, bool):
        raise ValueError("preserve_order must be a Boolean")
    if status not in {"planned", "complete"}:
        raise ValueError("status must be 'planned' or 'complete'")
    if status == "planned" and completed_global_step is not None:
        raise ValueError("A planned receipt cannot have a completed step")
    if status == "complete" and completed_global_step != total_steps:
        raise ValueError("A complete receipt must match total_steps")
    if not isinstance(padding_policy, str) or not padding_policy:
        raise ValueError("padding_policy must be a nonempty string")
    if exposure_scope not in set(S0_EXPOSURE_SCOPE_BY_RUN_SCOPE.values()):
        raise ValueError("The S0 exposure scope is invalid")
    if dataset_id != S0_TOOL_TRAJECTORY_DATASET_ID:
        raise ValueError("The S0 dataset ID must identify the v6 corpus")
    prompt_sha256 = _require_sha256(
        system_prompt_sha256,
        "S0 system prompt SHA-256",
    )
    if prompt_sha256 != (
        s0_tool_trajectory_prompt_contract().system_prompt_sha256
    ):
        raise ValueError("The S0 trajectory prompt identity changed")

    for label, indices in (
        ("consumed_indices", consumed_indices),
        ("distributed_padding_indices", distributed_padding_indices),
    ):
        if any(
            not isinstance(index, int) or index < 0 or index >= row_count
            for index in indices
        ):
            raise ValueError(f"{label} must reference S0 train rows")
    if not consumed_indices:
        raise ValueError("consumed_indices must reference S0 train rows")
    all_eligible_rows_exposed = set(consumed_indices) == set(range(row_count))
    full_exposure_required = exposure_scope == "all_eligible_train_rows"
    if full_exposure_required and not all_eligible_rows_exposed:
        raise ValueError(
            "The S0 production row plan must expose every eligible train row"
        )

    consumed_record_ids = [record_ids[index] for index in consumed_indices]
    consumed_fact_ids = [fact_ids[index] for index in consumed_indices]
    consumed_families = [
        question_families[index] for index in consumed_indices
    ]
    padding_record_ids = [
        record_ids[index] for index in distributed_padding_indices
    ]
    padding_fact_ids = [
        fact_ids[index] for index in distributed_padding_indices
    ]
    padding_families = [
        question_families[index] for index in distributed_padding_indices
    ]
    physical_record_ids = consumed_record_ids + padding_record_ids
    physical_fact_ids = consumed_fact_ids + padding_fact_ids
    physical_families = consumed_families + padding_families
    payload: dict[str, Any] = {
        "schema_version": "mentor-rl-s0-training-exposure-v2",
        "status": status,
        "run_id": run_id,
        "method_id": method_id,
        "identity": identities,
        "corpus": {
            "dataset_id": dataset_id,
            "eligible_train_rows": row_count,
            "record_sequence_sha256": _sequence_sha256(record_ids),
            "fact_sequence_sha256": _sequence_sha256(fact_ids),
            "question_family_counts": dict(
                sorted(Counter(question_families).items())
            ),
            "prompt_form_id": "train",
            "system_prompt_sha256": prompt_sha256,
        },
        "schedule": {
            "seed": seed,
            "total_steps": total_steps,
            "completed_global_step": completed_global_step,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "data_parallel_size": data_parallel_size,
            "global_batch_size": (
                per_device_train_batch_size
                * gradient_accumulation_steps
                * data_parallel_size
            ),
            "preserve_order": preserve_order,
            "order_strategy": (
                "source_order_each_epoch"
                if preserve_order
                else "python_seeded_epoch_shuffle"
            ),
        },
        "exposure_contract": {
            "scope": exposure_scope,
            "all_eligible_train_rows_required": full_exposure_required,
            "satisfied": True,
        },
        "logical_exposure": {
            "record_occurrences": len(consumed_record_ids),
            "unique_record_count": len(set(consumed_record_ids)),
            "unique_fact_count": len(set(consumed_fact_ids)),
            "all_eligible_train_rows_exposed": all_eligible_rows_exposed,
            "record_sequence_sha256": _sequence_sha256(consumed_record_ids),
            "fact_sequence_sha256": _sequence_sha256(consumed_fact_ids),
            "question_family_counts": dict(
                sorted(Counter(consumed_families).items())
            ),
        },
        "physical_exposure": {
            "record_occurrences": len(physical_record_ids),
            "record_sequence_sha256": _sequence_sha256(physical_record_ids),
            "fact_sequence_sha256": _sequence_sha256(physical_fact_ids),
            "question_family_counts": dict(
                sorted(Counter(physical_families).items())
            ),
            "distributed_padding": {
                "policy": padding_policy,
                "included_in_logical_exposure": False,
                "record_occurrences": len(padding_record_ids),
                "record_ids": padding_record_ids,
                "fact_ids": padding_fact_ids,
                "record_sequence_sha256": _sequence_sha256(
                    padding_record_ids
                ),
                "fact_sequence_sha256": _sequence_sha256(padding_fact_ids),
                "question_family_counts": dict(
                    sorted(Counter(padding_families).items())
                ),
            },
        },
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    return payload
