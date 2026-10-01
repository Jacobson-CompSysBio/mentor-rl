"""Define shared answer and supervised trace contracts."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import re
from typing import Any, Mapping


WORLD_MODEL_ANSWER_SCHEMA_VERSION = "mentor-rl-world-model-answer-v1"
SUPERVISED_TRACE_SCHEMA_VERSION = (
    "mentor-rl-world-model-supervised-trace-v1"
)
SUPERVISED_EXPLANATION_SCHEMA_VERSION = (
    "mentor-rl-supervised-explanation-v1"
)
SUPERVISED_EXPLANATION_LOSS_CONTRACT = "supervised_explanation_trace_v1"
ANSWER_FIELD_ORDER = (
    "status",
    "result",
    "continuation_decision",
    "summary",
)
ANSWER_STATUSES = frozenset(
    {"resolved", "ambiguous", "not_found", "invalid_input"}
)
CONTINUATION_DECISIONS = frozenset(
    {"continue", "revise", "defer", "stop"}
)
BOOK_MODES = frozenset({"closed_book", "open_book", "tool_call"})
TRACE_SPLITS = frozenset({"train", "val"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TOOL_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)

EXPLANATION_RECEIPT_FIELDS = frozenset(
    {
        "explanation_contract_version",
        "explanation_template_id",
        "explanation_template_sha256",
        "explanation_rule_ids",
        "explanation_source_refs",
        "explanation_input_sha256",
        "explanation_sha256",
    }
)
TRACE_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "metadata",
        "tools",
        "messages",
        "split",
        "provenance",
    }
)


class WorldModelContractError(ValueError):
    """Report one stable contract error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _JSONObjectPairs(list[tuple[str, Any]]):
    """Keep JSON object pairs and their source order."""


def canonical_json(value: Any) -> str:
    """Return stable JSON for hashes and contract checks."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise WorldModelContractError(
            "json_value_invalid",
            f"The value is not valid JSON: {error}",
        ) from error


def stable_json_sha256(value: Any) -> str:
    """Return the SHA-256 value for stable JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    """Return the SHA-256 value for one text value."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_nonempty_string(value: Any, label: str) -> str:
    """Return one nonempty string."""

    if not isinstance(value, str) or not value.strip():
        raise WorldModelContractError(
            f"{label}_invalid",
            f"{label} must be a nonempty string",
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    """Return one lowercase SHA-256 value."""

    text = _require_nonempty_string(value, label)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise WorldModelContractError(
            f"{label}_invalid",
            f"{label} must be a lowercase SHA-256 value",
        )
    return text


def _raise_json_constant(value: str) -> None:
    """Reject a nonstandard JSON numeric constant."""

    raise WorldModelContractError(
        "answer_json_constant",
        f"The answer contains the nonstandard JSON value {value}",
    )


def _decode_json_tree(value: Any, path: str) -> Any:
    """Convert one pair tree and reject duplicate object keys."""

    if isinstance(value, _JSONObjectPairs):
        result: dict[str, Any] = {}
        for key, item in value:
            if key in result:
                raise WorldModelContractError(
                    "answer_duplicate_key",
                    f"The answer has duplicate key {key!r} at {path}",
                )
            result[key] = _decode_json_tree(item, f"{path}.{key}")
        return result
    if isinstance(value, list):
        return [
            _decode_json_tree(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


@dataclass(frozen=True)
class WorldModelAnswer:
    """Store one four-field world-model answer."""

    status: str
    result: Mapping[str, Any]
    continuation_decision: str
    summary: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.status, str)
            or self.status not in ANSWER_STATUSES
        ):
            raise WorldModelContractError(
                "answer_status_invalid",
                f"status must be one of {sorted(ANSWER_STATUSES)}",
            )
        if not isinstance(self.result, Mapping):
            raise WorldModelContractError(
                "answer_result_invalid",
                "result must be one JSON object",
            )
        canonical_json(dict(self.result))
        if (
            not isinstance(self.continuation_decision, str)
            or self.continuation_decision not in CONTINUATION_DECISIONS
        ):
            raise WorldModelContractError(
                "answer_continuation_decision_invalid",
                "continuation_decision has an invalid value",
            )
        _require_nonempty_string(self.summary, "answer_summary")
        if self.status == "invalid_input":
            if set(self.result) != {"input", "expected"}:
                raise WorldModelContractError(
                    "answer_invalid_input_result",
                    "An invalid_input result requires input and expected",
                )
            _require_nonempty_string(
                self.result.get("expected"),
                "answer_invalid_input_expected",
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WorldModelAnswer":
        """Create one answer from an internal JSON object."""

        if not isinstance(payload, Mapping):
            raise WorldModelContractError(
                "answer_not_object",
                "The answer must be one JSON object",
            )
        if set(payload) != set(ANSWER_FIELD_ORDER):
            missing = sorted(set(ANSWER_FIELD_ORDER) - set(payload))
            extra = sorted(set(payload) - set(ANSWER_FIELD_ORDER))
            raise WorldModelContractError(
                "answer_fields_invalid",
                f"The answer fields differ: missing={missing}, extra={extra}",
            )
        return cls(
            status=payload["status"],
            result=copy.deepcopy(payload["result"]),
            continuation_decision=payload["continuation_decision"],
            summary=payload["summary"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return one answer object with the fixed field order."""

        return {
            "status": self.status,
            "result": copy.deepcopy(dict(self.result)),
            "continuation_decision": self.continuation_decision,
            "summary": self.summary,
        }


def parse_world_model_answer(text: Any) -> WorldModelAnswer:
    """Parse one complete answer and enforce its source field order."""

    if not isinstance(text, str):
        raise WorldModelContractError(
            "answer_not_text",
            "The answer must be text",
        )
    try:
        value = json.loads(
            text,
            object_pairs_hook=_JSONObjectPairs,
            parse_constant=_raise_json_constant,
        )
    except WorldModelContractError:
        raise
    except json.JSONDecodeError as error:
        raise WorldModelContractError(
            "answer_json_invalid",
            f"The answer is not complete JSON: {error.msg}",
        ) from error
    if not isinstance(value, _JSONObjectPairs):
        raise WorldModelContractError(
            "answer_not_object",
            "The answer must be one JSON object",
        )
    observed_order = tuple(key for key, _ in value)
    if len(set(observed_order)) != len(observed_order):
        _decode_json_tree(value, "answer")
    if set(observed_order) != set(ANSWER_FIELD_ORDER):
        missing = sorted(set(ANSWER_FIELD_ORDER) - set(observed_order))
        extra = sorted(set(observed_order) - set(ANSWER_FIELD_ORDER))
        raise WorldModelContractError(
            "answer_fields_invalid",
            f"The answer fields differ: missing={missing}, extra={extra}",
        )
    if observed_order != ANSWER_FIELD_ORDER:
        raise WorldModelContractError(
            "answer_field_order_invalid",
            "The answer fields have an invalid order",
        )
    decoded = _decode_json_tree(value, "answer")
    return WorldModelAnswer.from_mapping(decoded)


def serialize_world_model_answer(
    value: WorldModelAnswer | Mapping[str, Any],
) -> str:
    """Serialize one answer with the fixed top-level field order."""

    answer = (
        value
        if isinstance(value, WorldModelAnswer)
        else WorldModelAnswer.from_mapping(value)
    )
    try:
        result = json.dumps(
            answer.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise WorldModelContractError(
            "answer_json_invalid",
            f"The answer is not valid JSON: {error}",
        ) from error
    parse_world_model_answer(result)
    return result


@dataclass(frozen=True)
class ExplanationReceipt:
    """Store the deterministic source for one assistant explanation."""

    explanation_contract_version: str
    explanation_template_id: str
    explanation_template_sha256: str
    explanation_rule_ids: tuple[str, ...]
    explanation_source_refs: tuple[str, ...]
    explanation_input_sha256: str
    explanation_sha256: str

    def __post_init__(self) -> None:
        if (
            self.explanation_contract_version
            != SUPERVISED_EXPLANATION_SCHEMA_VERSION
        ):
            raise WorldModelContractError(
                "explanation_contract_version_invalid",
                "The supervised explanation contract version changed",
            )
        _require_nonempty_string(
            self.explanation_template_id,
            "explanation_template_id",
        )
        _require_sha256(
            self.explanation_template_sha256,
            "explanation_template_sha256",
        )
        for label, values in (
            ("explanation_rule_ids", self.explanation_rule_ids),
            ("explanation_source_refs", self.explanation_source_refs),
        ):
            if (
                not isinstance(values, tuple)
                or not values
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in values
                )
                or len(set(values)) != len(values)
            ):
                raise WorldModelContractError(
                    f"{label}_invalid",
                    f"{label} must contain unique nonempty strings",
                )
        _require_sha256(
            self.explanation_input_sha256,
            "explanation_input_sha256",
        )
        _require_sha256(self.explanation_sha256, "explanation_sha256")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        explanation: str,
    ) -> "ExplanationReceipt":
        """Create one receipt and bind it to its explanation."""

        if not isinstance(payload, Mapping):
            raise WorldModelContractError(
                "explanation_receipt_invalid",
                "explanation_receipt must be one JSON object",
            )
        if set(payload) != EXPLANATION_RECEIPT_FIELDS:
            missing = sorted(EXPLANATION_RECEIPT_FIELDS - set(payload))
            extra = sorted(set(payload) - EXPLANATION_RECEIPT_FIELDS)
            raise WorldModelContractError(
                "explanation_receipt_fields_invalid",
                "The explanation receipt fields differ: "
                f"missing={missing}, extra={extra}",
            )
        rule_ids = payload["explanation_rule_ids"]
        source_references = payload["explanation_source_refs"]
        if not isinstance(rule_ids, list):
            raise WorldModelContractError(
                "explanation_rule_ids_invalid",
                "explanation_rule_ids must be one JSON list",
            )
        if not isinstance(source_references, list):
            raise WorldModelContractError(
                "explanation_source_refs_invalid",
                "explanation_source_refs must be one JSON list",
            )
        receipt = cls(
            explanation_contract_version=payload[
                "explanation_contract_version"
            ],
            explanation_template_id=payload[
                "explanation_template_id"
            ],
            explanation_template_sha256=payload[
                "explanation_template_sha256"
            ],
            explanation_rule_ids=tuple(rule_ids),
            explanation_source_refs=tuple(source_references),
            explanation_input_sha256=payload[
                "explanation_input_sha256"
            ],
            explanation_sha256=payload["explanation_sha256"],
        )
        if text_sha256(explanation) != receipt.explanation_sha256:
            raise WorldModelContractError(
                "explanation_hash_mismatch",
                "The explanation does not match explanation_sha256",
            )
        return receipt

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON receipt."""

        return {
            "explanation_contract_version": (
                self.explanation_contract_version
            ),
            "explanation_template_id": self.explanation_template_id,
            "explanation_template_sha256": (
                self.explanation_template_sha256
            ),
            "explanation_rule_ids": list(self.explanation_rule_ids),
            "explanation_source_refs": list(self.explanation_source_refs),
            "explanation_input_sha256": self.explanation_input_sha256,
            "explanation_sha256": self.explanation_sha256,
        }


def _validate_tool_parameter_schema(
    value: Any,
    *,
    require_object_properties: bool = False,
) -> None:
    """Validate one schema that the GPT-OSS tool template supports."""

    if not isinstance(value, Mapping):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "Each tool parameter schema must be one JSON object",
        )
    if any(not isinstance(key, str) for key in value):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "Each tool parameter schema key must be text",
        )
    schema_type = value.get("type")
    one_of = value.get("oneOf")
    if isinstance(schema_type, list):
        if (
            not schema_type
            or any(
                not isinstance(item, str) or item not in TOOL_SCHEMA_TYPES
                for item in schema_type
            )
            or len(set(schema_type)) != len(schema_type)
        ):
            raise WorldModelContractError(
                "tool_parameters_invalid",
                "A tool parameter type list has invalid values",
            )
    elif schema_type is not None and not isinstance(schema_type, str):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "A tool parameter type must be text or a text list",
        )
    elif schema_type not in TOOL_SCHEMA_TYPES and one_of is None:
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "Each tool parameter needs a supported type or oneOf",
        )

    if "description" in value and not isinstance(value["description"], str):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "A tool parameter description must be text",
        )
    if "nullable" in value and not isinstance(value["nullable"], bool):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "A tool parameter nullable value must be a bool",
        )
    if "enum" in value:
        enum = value["enum"]
        if not isinstance(enum, list) or not enum:
            raise WorldModelContractError(
                "tool_parameters_invalid",
                "A tool parameter enum must be a nonempty list",
            )
        if schema_type == "string" and any(
            not isinstance(item, str) for item in enum
        ):
            raise WorldModelContractError(
                "tool_parameters_invalid",
                "A string parameter enum must contain only text",
            )
        if "default" in value and not isinstance(value["default"], str):
            raise WorldModelContractError(
                "tool_parameters_invalid",
                "A tool parameter enum default must be text",
            )
    if (
        one_of is not None
        and "default" in value
        and not isinstance(value["default"], str)
    ):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "A tool parameter oneOf default must be text",
        )

    if one_of is not None:
        if not isinstance(one_of, list) or not one_of:
            raise WorldModelContractError(
                "tool_parameters_invalid",
                "A tool parameter oneOf must be a nonempty list",
            )
        for variant in one_of:
            _validate_tool_parameter_schema(variant)

    properties = value.get("properties")
    if require_object_properties and (
        schema_type != "object" or not isinstance(properties, Mapping)
    ):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "Tool parameters must define an object with properties",
        )
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise WorldModelContractError(
                "tool_parameters_invalid",
                "Tool parameter properties must be one JSON object",
            )
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str) or not property_name:
                raise WorldModelContractError(
                    "tool_parameters_invalid",
                    "Each tool property must have a nonempty name",
                )
            _validate_tool_parameter_schema(property_schema)

    required = value.get("required", [])
    if (
        not isinstance(required, list)
        or any(
            not isinstance(item, str)
            or not isinstance(properties, Mapping)
            or item not in properties
            for item in required
        )
        or len(set(required)) != len(required)
    ):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "Tool required fields must be unique property names",
        )
    if (
        "additionalProperties" in value
        and not isinstance(value["additionalProperties"], bool)
    ):
        raise WorldModelContractError(
            "tool_parameters_invalid",
            "Tool additionalProperties must be a bool",
        )
    if "items" in value:
        _validate_tool_parameter_schema(value["items"])
    canonical_json(dict(value))


def _validate_tool_definitions(
    tools: tuple[Mapping[str, Any], ...],
) -> frozenset[str]:
    """Return all unique function names from the tool list."""

    if not isinstance(tools, tuple) or not tools:
        raise WorldModelContractError(
            "tools_invalid",
            "tools must be a nonempty tuple",
        )
    names: set[str] = set()
    for tool in tools:
        if (
            not isinstance(tool, Mapping)
            or set(tool) != {"type", "function"}
            or tool.get("type") != "function"
        ):
            raise WorldModelContractError(
                "tool_definition_invalid",
                "Each tool must be one function definition",
            )
        function = tool.get("function")
        if not isinstance(function, Mapping) or not {
            "name",
            "description",
            "parameters",
        }.issubset(function):
            raise WorldModelContractError(
                "tool_definition_invalid",
                "Each tool must define its name, description, and parameters",
            )
        name = _require_nonempty_string(function.get("name"), "tool_name")
        _require_nonempty_string(
            function.get("description"),
            "tool_description",
        )
        parameters = function.get("parameters")
        _validate_tool_parameter_schema(
            parameters,
            require_object_properties=True,
        )
        if name in names:
            raise WorldModelContractError(
                "tool_name_duplicate",
                "Tool names must be unique",
            )
        names.add(name)
        canonical_json(dict(tool))
    return frozenset(names)


def _validate_tool_call(
    value: Any,
    declared_tools: frozenset[str],
) -> tuple[str, str]:
    """Validate one assistant function call."""

    if not isinstance(value, Mapping):
        raise WorldModelContractError(
            "tool_call_invalid",
            "tool_call must be one JSON object",
        )
    if set(value) != {"id", "type", "function"} or value.get("type") != "function":
        raise WorldModelContractError(
            "tool_call_invalid",
            "tool_call must contain one function call",
        )
    call_id = _require_nonempty_string(value.get("id"), "tool_call_id")
    function = value.get("function")
    if not isinstance(function, Mapping) or set(function) != {
        "name",
        "arguments",
    }:
        raise WorldModelContractError(
            "tool_call_function_invalid",
            "The tool function fields are invalid",
        )
    name = _require_nonempty_string(function.get("name"), "tool_call_name")
    if name not in declared_tools:
        raise WorldModelContractError(
            "tool_call_undeclared",
            "The assistant called an undeclared tool",
        )
    if not isinstance(function.get("arguments"), Mapping):
        raise WorldModelContractError(
            "tool_call_arguments_invalid",
            "Tool arguments must be one JSON object",
        )
    canonical_json(dict(function["arguments"]))
    return call_id, name


@dataclass(frozen=True)
class SupervisedTraceRecord:
    """Store one generic supervised world-model trace."""

    record_id: str
    metadata: Mapping[str, Any]
    tools: tuple[Mapping[str, Any], ...]
    messages: tuple[Mapping[str, Any], ...]
    split: str
    provenance: Mapping[str, Any]
    schema_version: str = SUPERVISED_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SUPERVISED_TRACE_SCHEMA_VERSION:
            raise WorldModelContractError(
                "trace_schema_version_invalid",
                "The supervised trace schema version changed",
            )
        _require_nonempty_string(self.record_id, "record_id")
        if not isinstance(self.metadata, Mapping):
            raise WorldModelContractError(
                "metadata_invalid",
                "metadata must be one JSON object",
            )
        _require_nonempty_string(
            self.metadata.get("question_family"),
            "question_family",
        )
        book_mode = self.metadata.get("book_mode")
        if not isinstance(book_mode, str) or book_mode not in BOOK_MODES:
            raise WorldModelContractError(
                "book_mode_invalid",
                f"book_mode must be one of {sorted(BOOK_MODES)}",
            )
        if (
            self.metadata.get("answer_schema_version")
            != WORLD_MODEL_ANSWER_SCHEMA_VERSION
        ):
            raise WorldModelContractError(
                "answer_schema_version_invalid",
                "The answer schema version changed",
            )
        if (
            self.metadata.get("loss_contract")
            != SUPERVISED_EXPLANATION_LOSS_CONTRACT
        ):
            raise WorldModelContractError(
                "loss_contract_invalid",
                "The supervised explanation loss contract changed",
            )
        prompt_sha256 = _require_sha256(
            self.metadata.get("system_prompt_sha256"),
            "system_prompt_sha256",
        )
        canonical_json(dict(self.metadata))
        declared_tools = _validate_tool_definitions(self.tools)
        if not isinstance(self.split, str) or self.split not in TRACE_SPLITS:
            raise WorldModelContractError(
                "trace_split_invalid",
                f"split must be one of {sorted(TRACE_SPLITS)}",
            )
        if not isinstance(self.provenance, Mapping):
            raise WorldModelContractError(
                "provenance_invalid",
                "provenance must be one JSON object",
            )
        canonical_json(dict(self.provenance))
        if not isinstance(self.messages, tuple) or len(self.messages) < 3:
            raise WorldModelContractError(
                "messages_invalid",
                "messages must contain system, user, and assistant turns",
            )
        self._validate_messages(
            declared_tools=declared_tools,
            prompt_sha256=prompt_sha256,
            book_mode=str(book_mode),
        )

    def _validate_messages(
        self,
        *,
        declared_tools: frozenset[str],
        prompt_sha256: str,
        book_mode: str,
    ) -> None:
        """Validate the role order and each message shape."""

        system = self.messages[0]
        user = self.messages[1]
        if not isinstance(system, Mapping) or set(system) != {
            "role",
            "content",
        } or system.get("role") != "system":
            raise WorldModelContractError(
                "system_message_invalid",
                "The first message must be one system message",
            )
        system_text = _require_nonempty_string(
            system.get("content"),
            "system_content",
        )
        if text_sha256(system_text) != prompt_sha256:
            raise WorldModelContractError(
                "system_prompt_hash_mismatch",
                "The system message does not match its SHA-256 value",
            )
        if not isinstance(user, Mapping) or set(user) != {
            "role",
            "content",
        } or user.get("role") != "user":
            raise WorldModelContractError(
                "user_message_invalid",
                "The second message must be one user message",
            )
        _require_nonempty_string(user.get("content"), "user_content")

        remaining = self.messages[2:]
        if len(remaining) % 2 == 0:
            raise WorldModelContractError(
                "message_sequence_invalid",
                "The trace must end with one assistant answer",
            )
        call_ids: set[str] = set()
        call_count = (len(remaining) - 1) // 2
        for index in range(call_count):
            assistant = remaining[index * 2]
            tool = remaining[index * 2 + 1]
            call_id, tool_name = self._validate_assistant_call(
                assistant,
                declared_tools=declared_tools,
            )
            if call_id in call_ids:
                raise WorldModelContractError(
                    "tool_call_id_duplicate",
                    "Tool call IDs must be unique in one trace",
                )
            call_ids.add(call_id)
            if not isinstance(tool, Mapping) or set(tool) != {
                "role",
                "tool_call_id",
                "name",
                "content",
            } or tool.get("role") != "tool":
                raise WorldModelContractError(
                    "tool_message_invalid",
                    "Each assistant call must have one tool result",
                )
            if tool.get("tool_call_id") != call_id:
                raise WorldModelContractError(
                    "tool_call_id_mismatch",
                    "The tool result must reference the prior call ID",
                )
            if tool.get("name") != tool_name:
                raise WorldModelContractError(
                    "tool_name_mismatch",
                    "The tool result must name the called tool",
                )
            if not isinstance(tool.get("content"), Mapping):
                raise WorldModelContractError(
                    "tool_result_invalid",
                    "Tool content must be one JSON object",
                )
            canonical_json(dict(tool["content"]))
        self._validate_assistant_final(remaining[-1])
        if book_mode == "closed_book" and call_count != 0:
            raise WorldModelContractError(
                "closed_book_tool_call",
                "A closed-book trace cannot call a tool",
            )

    @staticmethod
    def _validate_explanation(
        message: Mapping[str, Any],
    ) -> None:
        """Validate one explanation and its receipt."""

        explanation = _require_nonempty_string(
            message.get("assistant_explanation"),
            "assistant_explanation",
        )
        ExplanationReceipt.from_dict(
            message.get("explanation_receipt"),
            explanation=explanation,
        )

    @classmethod
    def _validate_assistant_call(
        cls,
        message: Any,
        *,
        declared_tools: frozenset[str],
    ) -> tuple[str, str]:
        """Validate one assistant call message."""

        expected = {
            "role",
            "assistant_explanation",
            "explanation_receipt",
            "tool_call",
        }
        if not isinstance(message, Mapping) or set(message) != expected:
            raise WorldModelContractError(
                "assistant_call_message_invalid",
                "An assistant call message has invalid fields",
            )
        if message.get("role") != "assistant":
            raise WorldModelContractError(
                "assistant_call_message_invalid",
                "A call message must have the assistant role",
            )
        cls._validate_explanation(message)
        return _validate_tool_call(
            message.get("tool_call"),
            declared_tools,
        )

    @classmethod
    def _validate_assistant_final(
        cls,
        message: Any,
    ) -> None:
        """Validate the final assistant message."""

        expected = {
            "role",
            "assistant_explanation",
            "explanation_receipt",
            "content",
        }
        if not isinstance(message, Mapping) or set(message) != expected:
            raise WorldModelContractError(
                "assistant_final_message_invalid",
                "The final assistant message has invalid fields",
            )
        if message.get("role") != "assistant":
            raise WorldModelContractError(
                "assistant_final_message_invalid",
                "The final message must have the assistant role",
            )
        cls._validate_explanation(message)
        parse_world_model_answer(message.get("content"))

    @property
    def tool_call_count(self) -> int:
        """Return the count of assistant tool calls."""

        return (len(self.messages) - 3) // 2

    @property
    def final_answer(self) -> WorldModelAnswer:
        """Return the validated final answer."""

        return parse_world_model_answer(self.messages[-1]["content"])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisedTraceRecord":
        """Create one supervised trace from JSON data."""

        if not isinstance(payload, Mapping):
            raise WorldModelContractError(
                "trace_record_invalid",
                "A supervised trace must be one JSON object",
            )
        if set(payload) != TRACE_RECORD_FIELDS:
            missing = sorted(TRACE_RECORD_FIELDS - set(payload))
            extra = sorted(set(payload) - TRACE_RECORD_FIELDS)
            raise WorldModelContractError(
                "trace_record_fields_invalid",
                f"The trace fields differ: missing={missing}, extra={extra}",
            )
        tools = payload["tools"]
        messages = payload["messages"]
        if not isinstance(tools, list):
            raise WorldModelContractError(
                "tools_invalid",
                "tools must be one JSON list",
            )
        if not isinstance(messages, list):
            raise WorldModelContractError(
                "messages_invalid",
                "messages must be one JSON list",
            )
        return cls(
            schema_version=payload["schema_version"],
            record_id=payload["record_id"],
            metadata=copy.deepcopy(payload["metadata"]),
            tools=tuple(copy.deepcopy(tools)),
            messages=tuple(copy.deepcopy(messages)),
            split=payload["split"],
            provenance=copy.deepcopy(payload["provenance"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return one supervised trace JSON object."""

        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "metadata": copy.deepcopy(dict(self.metadata)),
            "tools": [copy.deepcopy(dict(tool)) for tool in self.tools],
            "messages": [
                copy.deepcopy(dict(message)) for message in self.messages
            ],
            "split": self.split,
            "provenance": copy.deepcopy(dict(self.provenance)),
        }

    def to_messages(self) -> list[dict[str, Any]]:
        """Map contract messages to the GPT-OSS chat fields."""

        result: list[dict[str, Any]] = []
        for message in self.messages:
            role = message["role"]
            if role in {"system", "user"}:
                result.append(
                    {"role": role, "content": message["content"]}
                )
            elif role == "tool":
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "name": message["name"],
                        "content": copy.deepcopy(dict(message["content"])),
                    }
                )
            elif "tool_call" in message:
                result.append(
                    {
                        "role": "assistant",
                        "thinking": message["assistant_explanation"],
                        "tool_calls": [
                            copy.deepcopy(dict(message["tool_call"]))
                        ],
                    }
                )
            else:
                result.append(
                    {
                        "role": "assistant",
                        "thinking": message["assistant_explanation"],
                        "content": message["content"],
                    }
                )
        return result

    def to_chat_messages(self) -> list[dict[str, Any]]:
        """Return the chat messages through the public transport method."""

        return self.to_messages()
