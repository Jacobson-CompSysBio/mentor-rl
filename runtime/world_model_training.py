"""Prepare S0 SFT data and exposure receipts.

The trainer reads canonical rows of the shared world-model contract
(`runtime/world_model/contract.yaml`, row schema `multiplex_sft_v3`). This
module validates those rows, tokenizes them with a loss mask over the
assistant turns, and builds the content-addressed exposure receipt that ties a
checkpoint to the exact train rows it consumed.
"""

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

from runtime.world_model.contract import load_contract, system_prompt_sha256


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
_V3_CONTRACT = load_contract()
_V3_SYSTEM_PROMPT_SHA256 = system_prompt_sha256(_V3_CONTRACT)

S0_FAMILIES = frozenset(
    {
        "human_symbol_to_ensembl",
        "human_ensembl_to_symbol",
        "human_ambiguous_symbol",
    }
)
# "multiplex_sft_v3" weights every assistant token equally (the uniform arms).
# "multiplex_sft_v4_priority" upweights the tool call, the final JSON, and its
# decision fields (the priority arms); see tokenize_v3_trace_for_sft.
S0_V3_TRACE_LOSS_CONTRACT = "multiplex_sft_v3"
S0_V4_PRIORITY_LOSS_CONTRACT = "multiplex_sft_v4_priority"
S0_LOSS_CONTRACTS = frozenset(
    {S0_V3_TRACE_LOSS_CONTRACT, S0_V4_PRIORITY_LOSS_CONTRACT}
)
S0_V4_DATASET_ID = "world_model_v4_s0_human_identifiers"
S0_EXPOSURE_SCOPE_BY_RUN_SCOPE = {
    "qualification": "unrestricted",
    "production": "all_eligible_train_rows",
}
V3_TOOL_DESCRIPTIONS = {
    "lookup_human_gene_symbol": (
        "Look up one human gene symbol in Ensembl release 116."
    ),
    "lookup_human_ensembl_gene_id": (
        "Look up one human Ensembl gene ID in release 116."
    ),
    "check_multiplex_membership": (
        "Check human Ensembl gene IDs in one registered multiplex."
    ),
}
V3_HARNESS_HEADER = "Context (from the harness):"
V3_USER_HEADER = "User request:"


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for one value."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def ordered_json(value: Any) -> str:
    """Return compact JSON without changing contract-significant key order."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 value for canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def question_families_for_dataset(dataset_id: str) -> frozenset[str]:
    """Return the exact question-family contract for one training corpus."""

    if dataset_id != S0_V4_DATASET_ID:
        raise ValueError(f"The S0 dataset ID is not supported: {dataset_id!r}")
    return S0_FAMILIES


def renderable_v3_tools(tools: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add tokenizer-only descriptions without changing the v3 wire."""

    result: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, Mapping):
            raise ValueError("A v3 tool definition lacks function metadata")
        name = function.get("name")
        description = V3_TOOL_DESCRIPTIONS.get(str(name))
        if description is None:
            raise ValueError(f"Unknown v3 tool definition: {name!r}")
        rendered_function = dict(function)
        rendered_function.setdefault("description", description)
        result.append({"type": "function", "function": rendered_function})
    return result


def _validate_v3_message_sequence(messages: Any) -> None:
    """Validate the canonical message grammar for one v3 row."""

    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("A v3 row must contain at least three messages")
    if any(not isinstance(message, Mapping) for message in messages):
        raise ValueError("A v3 message is not an object")
    if messages[0].get("role") != "system":
        raise ValueError("A v3 row must start with one system message")
    if not isinstance(messages[0].get("content"), str):
        raise ValueError("The v3 system message requires text content")

    message_index = 1
    if messages[message_index].get("role") == "harness":
        if not isinstance(messages[message_index].get("content"), str):
            raise ValueError("The v3 harness message requires text content")
        message_index += 1

    turn_count = 0
    while message_index < len(messages):
        user_message = messages[message_index]
        if user_message.get("role") != "user":
            raise ValueError("Each v3 turn must start with one user message")
        if not isinstance(user_message.get("content"), str):
            raise ValueError("A v3 user message requires text content")
        message_index += 1

        while (
            message_index < len(messages)
            and messages[message_index].get("role") == "assistant"
            and "tool_call" in messages[message_index]
        ):
            call_message = messages[message_index]
            call = call_message.get("tool_call")
            if "content" in call_message or not isinstance(call, Mapping):
                raise ValueError("A v3 assistant call has an invalid shape")
            if not isinstance(call.get("name"), str) or not isinstance(
                call.get("arguments"), Mapping
            ):
                raise ValueError("A v3 assistant call lacks tool metadata")
            if not isinstance(call_message.get("thinking", ""), str):
                raise ValueError("A v3 assistant call has invalid thinking")
            message_index += 1
            if (
                message_index >= len(messages)
                or messages[message_index].get("role") != "tool"
            ):
                raise ValueError("A v3 assistant call requires one tool result")
            tool_message = messages[message_index]
            if not isinstance(tool_message.get("name"), str):
                raise ValueError("A v3 tool result requires one tool name")
            if not isinstance(tool_message.get("content"), Mapping):
                raise ValueError("A v3 tool result requires object content")
            message_index += 1

        if message_index >= len(messages):
            raise ValueError("A v3 turn requires one final assistant answer")
        final_message = messages[message_index]
        if (
            final_message.get("role") != "assistant"
            or "tool_call" in final_message
            or not isinstance(final_message.get("content"), Mapping)
        ):
            raise ValueError("A v3 turn has an invalid final assistant answer")
        if list(final_message["content"]) != _V3_CONTRACT["answer"]["keys"]:
            raise ValueError("The v3 final answer field order changed")
        if not isinstance(final_message.get("thinking", ""), str):
            raise ValueError("A v3 final answer has invalid thinking")
        message_index += 1
        turn_count += 1

    if turn_count == 0:
        raise ValueError("A v3 row requires at least one user turn")


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


def flatten_v3_trace_for_arrow(
    record: Mapping[str, Any],
    *,
    expected_split: str = "train",
    dataset_id: str | None = None,
) -> dict[str, str]:
    """Convert one canonical v3 row to the trainer's scalar schema."""

    if expected_split not in {"train", "validation"}:
        raise ValueError(f"Unsupported v3 split: {expected_split!r}")
    if list(record) != ["row_id", "messages", "tools", "metadata"]:
        raise ValueError("The v3 row field order changed")
    row_id = record.get("row_id")
    messages = record.get("messages")
    tools = record.get("tools")
    metadata = record.get("metadata")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("The v3 row requires one row_id")
    _validate_v3_message_sequence(messages)
    if not isinstance(tools, list) or not tools:
        raise ValueError("The v3 row requires tool definitions")
    if not isinstance(metadata, Mapping):
        raise ValueError("The v3 row requires metadata")
    contract = _V3_CONTRACT
    if (
        metadata.get("schema_version")
        != contract["runtime"]["row_schema_version"]
        or metadata.get("tool_contract_version")
        != contract["runtime"]["tool_contract_version"]
        or metadata.get("system_prompt_sha256")
        != _V3_SYSTEM_PROMPT_SHA256
        or messages[0].get("content") != contract["runtime"]["system_prompt"]
    ):
        raise ValueError("The v3 row does not match the canonical contract")
    family = metadata.get("question_family")
    split_unit = metadata.get("split_unit")
    if dataset_id is not None:
        question_families_for_dataset(dataset_id)
    if family not in S0_FAMILIES:
        raise ValueError("The v3 row has an unsupported question family")
    if not isinstance(split_unit, str) or not split_unit:
        raise ValueError("The v3 row requires one split_unit")
    return {
        # The v3 wire contract assigns meaning to object key order. This
        # scalar Arrow projection is later decoded and rendered for training,
        # so canonical sorting here would silently change supervised labels.
        "trace_json": ordered_json(dict(record)),
        "metadata_json": canonical_json(dict(metadata)),
        "question_family": str(family),
        "record_id": row_id,
        "fact_id": split_unit,
        "prompt_form_id": expected_split,
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





def tokenize_v3_trace_for_sft(
    record: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    *,
    priority_weights: bool = False,
) -> dict[str, list[int] | list[float]]:
    """Tokenize one canonical S0/S1 row and mask non-assistant turns."""

    source_messages = record.get("messages")
    tools = record.get("tools")
    if not isinstance(tools, list):
        raise ValueError("The v3 trace lacks messages or tools")
    _validate_v3_message_sequence(source_messages)
    messages: list[dict[str, Any]] = []
    harness_content: str | None = None
    for message in source_messages:
        role = message.get("role")
        if role == "harness":
            harness_content = str(message["content"])
        elif role == "system":
            messages.append({"role": role, "content": message["content"]})
        elif role == "user":
            content = str(message["content"])
            if harness_content is not None:
                content = (
                    f"{V3_HARNESS_HEADER}\n{harness_content}\n\n"
                    f"{V3_USER_HEADER}\n{content}"
                )
                harness_content = None
            messages.append({"role": role, "content": content})
        elif role == "assistant" and "tool_call" in message:
            call = message["tool_call"]
            messages.append(
                {
                    "role": "assistant",
                    "thinking": message.get("thinking", ""),
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                    ],
                }
            )
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "name": message["name"],
                    "content": message["content"],
                }
            )
        elif role == "assistant" and "content" in message:
            messages.append(
                {
                    "role": "assistant",
                    "thinking": message.get("thinking", ""),
                    "content": json.dumps(
                        message["content"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        else:
            raise ValueError("A v3 message has an unsupported shape")
    if harness_content is not None:
        raise ValueError("A v3 harness message requires a following user")

    template_args = {
        "tools": renderable_v3_tools(tools),
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

    current_user_index = max(
        index
        for index, message in enumerate(messages)
        if message["role"] == "user"
    )
    base_messages = [
        dict(message) for message in messages[: current_user_index + 1]
    ]
    for message in base_messages:
        if message["role"] == "assistant":
            message.pop("thinking", None)
    base_ids = render(base_messages)
    full_ids = list(base_ids)
    completion_mask = [0] * len(base_ids)
    raw_loss_weights: list[float] = [0.0] * len(base_ids)

    current_messages = messages[current_user_index + 1 :]
    for message_index in range(0, len(current_messages) - 1, 2):
        assistant_message = current_messages[message_index]
        tool_message = current_messages[message_index + 1]
        if (
            assistant_message.get("role") != "assistant"
            or "tool_calls" not in assistant_message
            or tool_message.get("role") != "tool"
        ):
            raise ValueError(
                "A v3 tool action requires one assistant call and result"
            )

        assistant_ids = render(base_messages + [assistant_message])
        if assistant_ids[: len(base_ids)] != base_ids:
            raise RuntimeError("The v3 prefix is not a prefix of the tool call")
        assistant_delta = assistant_ids[len(base_ids) :]
        if not assistant_delta:
            raise RuntimeError("The v3 trace has no assistant call tokens")
        full_ids.extend(assistant_delta)
        completion_mask.extend([1] * len(assistant_delta))
        assistant_weights = [1.0] * len(assistant_delta)
        if priority_weights:
            plain_call = dict(assistant_message)
            plain_call.pop("thinking", None)
            plain_call_ids = render(base_messages + [plain_call])
            if plain_call_ids[: len(base_ids)] != base_ids:
                raise RuntimeError(
                    "The v3 prefix is not a prefix of the plain tool call"
                )
            call_delta = plain_call_ids[len(base_ids) :]
            if (
                not call_delta
                or assistant_delta[-len(call_delta) :] != call_delta
            ):
                raise RuntimeError(
                    "The v3 tool call is not a suffix after its thinking"
                )
            assistant_weights[-len(call_delta) :] = [4.0] * len(
                call_delta
            )
        raw_loss_weights.extend(assistant_weights)

        through_tool_ids = render(
            base_messages + [assistant_message, tool_message]
        )
        if through_tool_ids[: len(assistant_ids)] != assistant_ids:
            raise RuntimeError("The v3 tool call is not a prefix of its result")
        tool_delta = through_tool_ids[len(assistant_ids) :]
        if not tool_delta:
            raise RuntimeError("The v3 trace has no tool result tokens")
        full_ids.extend(tool_delta)
        completion_mask.extend([0] * len(tool_delta))
        raw_loss_weights.extend([0.0] * len(tool_delta))

        plain_assistant = dict(assistant_message)
        plain_assistant.pop("thinking", None)
        base_messages = base_messages + [plain_assistant, tool_message]
        base_ids = render(base_messages)

    final_message = current_messages[-1]
    if (
        final_message.get("role") != "assistant"
        or "tool_calls" in final_message
    ):
        raise ValueError("The v3 trace must end with one assistant answer")
    final_render_ids = render(base_messages + [final_message])
    if final_render_ids[: len(base_ids)] != base_ids:
        raise RuntimeError("The v3 prefix is not a prefix of the final answer")
    final_ids = final_render_ids[len(base_ids) :]
    if not final_ids:
        raise RuntimeError("The v3 trace has no final answer tokens")
    full_ids.extend(final_ids)
    completion_mask.extend([1] * len(final_ids))
    final_weights = [1.0] * len(final_ids)
    if priority_weights:
        plain_final = dict(final_message)
        plain_final.pop("thinking", None)
        plain_final_render_ids = render(base_messages + [plain_final])
        if plain_final_render_ids[: len(base_ids)] != base_ids:
            raise RuntimeError(
                "The v3 prefix is not a prefix of the plain final answer"
            )
        plain_final_delta = plain_final_render_ids[len(base_ids) :]
        if (
            not plain_final_delta
            or final_ids[-len(plain_final_delta) :] != plain_final_delta
        ):
            raise RuntimeError(
                "The v3 final answer is not a suffix after its thinking"
            )
        final_payload = json.loads(str(final_message["content"]))
        final_json = ordered_json(final_payload)
        encoded_json = tokenizer(
            final_json,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        json_ids = normalize_token_ids(encoded_json)
        offsets = encoded_json.get("offset_mapping")
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if (
            not isinstance(offsets, (list, tuple))
            or len(offsets) != len(json_ids)
            or any(
                not isinstance(offset, (list, tuple))
                or len(offset) != 2
                for offset in offsets
            )
        ):
            raise RuntimeError(
                "The priority loss requires tokenizer offset mappings"
            )
        matches = [
            index
            for index in range(len(plain_final_delta) - len(json_ids) + 1)
            if plain_final_delta[index : index + len(json_ids)] == json_ids
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "The final JSON tokens do not have one exact rendered span"
            )
        json_start = len(final_ids) - len(plain_final_delta) + matches[0]
        for token_index in range(len(json_ids)):
            final_weights[json_start + token_index] = 4.0

        # Build exact top-level field spans from the same compact serializer.
        # Marking by offset safely handles BPE tokens that straddle punctuation.
        critical = {
            "status",
            "result",
            "continuation_decision",
            "blocked_reason",
        }
        cursor = 1
        critical_spans: list[tuple[int, int]] = []
        items = list(final_payload.items())
        for item_index, (key, value) in enumerate(items):
            encoded_item = (
                ordered_json(str(key)) + ":" + ordered_json(value)
            )
            item_start = cursor
            item_end = item_start + len(encoded_item)
            if key in critical:
                critical_spans.append((item_start, item_end))
            cursor = item_end + (1 if item_index + 1 < len(items) else 0)
        if cursor != len(final_json) - 1:
            raise RuntimeError("The final JSON field spans are inconsistent")
        for token_index, offset in enumerate(offsets):
            token_start, token_end = (int(offset[0]), int(offset[1]))
            if any(
                token_end > field_start and token_start < field_end
                for field_start, field_end in critical_spans
            ):
                final_weights[json_start + token_index] = 8.0
    raw_loss_weights.extend(final_weights)

    if len(full_ids) > max_length:
        raise RuntimeError(
            f"Tokenized trace has {len(full_ids)} tokens, exceeding "
            f"--max_length={max_length}"
        )
    if len(completion_mask) != len(full_ids):
        raise RuntimeError("The v3 loss mask has an invalid length")
    if len(raw_loss_weights) != len(full_ids):
        raise RuntimeError("The v3 priority loss mask has an invalid length")
    result: dict[str, list[int] | list[float]] = {
        "input_ids": full_ids,
        "completion_mask": completion_mask,
    }
    if priority_weights:
        positive_weights = [
            weight for weight in raw_loss_weights if weight > 0.0
        ]
        if len(positive_weights) != sum(completion_mask):
            raise RuntimeError(
                "Each v3 completion token requires one positive loss weight"
            )
        mean_weight = sum(positive_weights) / len(positive_weights)
        result["loss_weights"] = [
            weight / mean_weight if weight > 0.0 else 0.0
            for weight in raw_loss_weights
        ]
    return result


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


def s0_exposure_corpus_identity(
    loss_contract: str,
    dataset_id: str,
) -> dict[str, str]:
    """Return the corpus identity for one dataset and loss contract."""

    question_families_for_dataset(dataset_id)
    if loss_contract not in S0_LOSS_CONTRACTS:
        raise ValueError(
            f"Dataset {dataset_id!r} requires one of the loss contracts "
            f"{sorted(S0_LOSS_CONTRACTS)!r}, got {loss_contract!r}"
        )
    return {
        "dataset_id": dataset_id,
        "system_prompt_sha256": _V3_SYSTEM_PROMPT_SHA256,
    }


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
    dataset_id: str,
    system_prompt_sha256: str,
    completed_global_step: int | None = None,
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
    allowed_question_families = question_families_for_dataset(dataset_id)
    if any(
        family not in allowed_question_families
        for family in question_families
    ):
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
    prompt_sha256 = _require_sha256(
        system_prompt_sha256,
        "S0 system prompt SHA-256",
    )
    if prompt_sha256 != _V3_SYSTEM_PROMPT_SHA256:
        raise ValueError("The S0 system prompt identity changed")

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
    exposure_satisfied = not full_exposure_required or all_eligible_rows_exposed
    if not exposure_satisfied:
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
            "satisfied": exposure_satisfied,
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
