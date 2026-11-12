"""Measure untruncated S2 verification traces with the pinned local tokenizer.

This mirrors the current-turn token concatenation used by the S0/S1 serializer,
while accepting the separately pinned S2 tool definitions. It is a CPU-only
verification utility, not a trainer adapter or permission to export training
data. Oversized examples remain intact and are reported explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import Any

from .contract import REPOSITORY_ROOT
from ..world_model_training import (
    V3_HARNESS_HEADER, V3_USER_HEADER, _validate_v3_message_sequence,
    normalize_token_ids, tokenizer_artifact_hashes,
)


ARTIFACT_NAMES = frozenset({
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
})
RENDER_DATE = "2026-10-28"


def verification_clock(format_string: str) -> str:
    """Freeze the template's clock so identical artifacts produce identical tokens."""
    return datetime.strptime(RENDER_DATE, "%Y-%m-%d").strftime(format_string)


def load_tokenizer(pin: Mapping[str, Any]) -> Any:
    """Verify all four local artifacts before loading a fast tokenizer offline.

    Framework imports are disabled for this verification process before the
    lazy Transformers import, avoiding GPU-library initialization on login
    nodes. No model weights or remote Python implementation are loaded.
    """
    if not isinstance(pin, Mapping) or set(pin) != {"path", "artifacts"}:
        raise ValueError("a tokenizer pin requires path and exact artifact hashes")
    if not isinstance(pin["path"], str) or not pin["path"]:
        raise ValueError("the tokenizer path must be nonempty")
    if not isinstance(pin["artifacts"], Mapping) or set(pin["artifacts"]) != ARTIFACT_NAMES:
        raise ValueError("the tokenizer requires exactly four pinned artifacts")
    path = Path(pin["path"])
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    actual = tokenizer_artifact_hashes(path)
    if actual != dict(pin["artifacts"]):
        raise ValueError("local tokenizer artifacts differ from their pins")
    for name in ("USE_TORCH", "USE_TF", "USE_FLAX"):
        os.environ.setdefault(name, "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
    )
    if not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str):
        raise ValueError("the verification tokenizer needs its local fast backend and chat template")
    return tokenizer


def _model_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use the existing harness, thinking, tool-call, and compact final-JSON wire."""
    _validate_v3_message_sequence(row["messages"])
    messages, harness = [], None
    for source in row["messages"]:
        role = source["role"]
        if role == "harness":
            harness = source["content"]
        elif role == "system":
            messages.append({"role": role, "content": source["content"]})
        elif role == "user":
            content = source["content"]
            if harness is not None:
                content = f"{V3_HARNESS_HEADER}\n{harness}\n\n{V3_USER_HEADER}\n{content}"
                harness = None
            messages.append({"role": role, "content": content})
        elif role == "assistant" and "tool_call" in source:
            call = source["tool_call"]
            messages.append({"role": role, "thinking": source.get("thinking", ""),
                             "tool_calls": [{"type": "function", "function": {
                                 "name": call["name"], "arguments": call["arguments"],
                             }}]})
        elif role == "tool":
            messages.append({"role": role, "name": source["name"], "content": source["content"]})
        else:
            messages.append({"role": role, "thinking": source.get("thinking", ""),
                             "content": json.dumps(source["content"], ensure_ascii=False,
                                                   separators=(",", ":"), allow_nan=False)})
    if harness is not None:
        raise ValueError("harness context requires a following user request")
    return messages


def tokenize_trace(row: Mapping[str, Any], tokenizer: Any) -> dict[str, list[int]]:
    """Keep every current-turn reasoning token and mask all supplied/tool evidence.

    Rendering a full conversation at once can remove previous assistant
    analysis. Concatenating verified call/result/final deltas retains each
    current-turn analysis segment, matching the existing S0/S1 algorithm.
    No max-length truncation or padding is performed.
    """
    tools = row.get("tools")
    if not isinstance(tools, list) or any(
        not isinstance(tool, Mapping) or tool.get("type") != "function"
        or not isinstance(tool.get("function"), Mapping)
        or not isinstance(tool["function"].get("description"), str)
        for tool in tools
    ):
        raise ValueError("S2 verification requires explicit described tool definitions")
    messages = _model_messages(row)

    def render(selected: list[dict[str, Any]]) -> list[int]:
        return normalize_token_ids(tokenizer.apply_chat_template(
            selected, tools=tools, tokenize=True, add_generation_prompt=False,
            enable_thinking=True, reasoning_effort="low", strftime_now=verification_clock,
        ))

    current_user = max(i for i, message in enumerate(messages) if message["role"] == "user")
    base = [dict(message) for message in messages[:current_user + 1]]
    for message in base:
        if message["role"] == "assistant":
            message.pop("thinking", None)
    base_ids = render(base)
    full_ids, mask = list(base_ids), [0] * len(base_ids)
    current = messages[current_user + 1:]

    def append_delta(ids: list[int], prefix: list[int], supervised: int) -> None:
        if ids[:len(prefix)] != prefix or len(ids) == len(prefix):
            raise ValueError("the tokenizer changed a trace prefix or emitted an empty segment")
        delta = ids[len(prefix):]
        full_ids.extend(delta)
        mask.extend([supervised] * len(delta))

    for i in range(0, len(current) - 1, 2):
        assistant, response = current[i:i + 2]
        if assistant["role"] != "assistant" or "tool_calls" not in assistant or response["role"] != "tool":
            raise ValueError("a current-turn call requires its immediate tool response")
        if response["name"] != assistant["tool_calls"][0]["function"]["name"]:
            raise ValueError("tool response name differs from its call")
        call_ids = render(base + [assistant])
        append_delta(call_ids, base_ids, 1)
        append_delta(render(base + [assistant, response]), call_ids, 0)
        plain = dict(assistant)
        plain.pop("thinking", None)
        base = base + [plain, response]
        base_ids = render(base)
    final = current[-1]
    if final["role"] != "assistant" or "tool_calls" in final:
        raise ValueError("the trace must finish with an assistant answer")
    append_delta(render(base + [final]), base_ids, 1)
    if len(mask) != len(full_ids) or not any(mask) or any(value not in (0, 1) for value in mask):
        raise ValueError("invalid current-turn assistant mask")
    return {"input_ids": full_ids, "completion_mask": mask}


def audit_rows(rows: Iterable[Mapping[str, Any]], tokenizer: Any,
               max_length: int = 4096) -> dict[str, Any]:
    """Count every complete verification wire and retain oversized row identities."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("the context budget must be an explicit positive integer")
    counts, seen, oversized = [], set(), []
    for row in rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise ValueError("token auditing requires distinct nonempty row IDs")
        seen.add(row_id)
        encoded = tokenize_trace(row, tokenizer)
        tokens = len(encoded["input_ids"])
        assistant_tokens = sum(encoded["completion_mask"])
        final_json = json.dumps(row["messages"][-1]["content"], ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False)
        final_tokens = len(normalize_token_ids(tokenizer(final_json, add_special_tokens=False)))
        over_budget = tokens > max_length
        if over_budget:
            oversized.append(row_id)
        counts.append({"row_id": row_id, "tokens": tokens, "assistant_tokens": assistant_tokens,
                       "masked_tokens": tokens - assistant_tokens, "final_json_tokens": final_tokens,
                       "over_budget": over_budget})
    return {
        "schema_version": "mentor-rl-world-model-s2-token-verification-v1",
        "serializer": "s2_current_turn_delta_verification_v1",
        "render_date": RENDER_DATE,
        "package_versions": {name: version(name) for name in ("transformers", "tokenizers", "jinja2")},
        "max_length": max_length, "rows": len(counts),
        "max_tokens": max((row["tokens"] for row in counts), default=0),
        "max_final_json_tokens": max((row["final_json_tokens"] for row in counts), default=0),
        "over_budget_count": len(oversized), "over_budget_row_ids": oversized,
        "context_fits": not oversized, "current_turn_masks_checked": True,
        "truncation_applied": False, "adapter_verified": False,
        "production_export_allowed": False, "row_counts": counts,
        "row_token_counts_sha256": hashlib.sha256(json.dumps(counts, sort_keys=True,
                                                              separators=(",", ":")).encode()).hexdigest(),
    }
