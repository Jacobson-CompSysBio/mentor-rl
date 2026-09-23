"""Typed public record contracts for the multiplex world-model curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any, Mapping


### IMMUTABLE VARS FOR WORLD MODEL CURRICULUM/TRACKING
WORLD_MODEL_SCHEMA_VERSION = "multiplex_sft_v2"
IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION = "identifier_tool_sft_v2"
IDENTIFIER_TOOL_CONTRACT_VERSION = (
    "mentor-rl-world-model-s0-identifier-tools-v1"
)
TOOL_TRAJECTORY_FORMAT = "tool_trajectory"
BOOK_MODES = frozenset({"closed_book", "open_book", "tool_call"})
STEPS = tuple(f"S{index}" for index in range(1, 10))
ANSWER_FORMAT = "json"
ENTITY_NAMESPACE = "ensembl_gene_id_primary"
HUMAN_TAXON_ID = "NCBITaxon:9606"
ENSEMBL_RELEASE = "Ensembl_116"
S0_FAMILIES = frozenset(
    {
        "human_symbol_to_ensembl",
        "human_ensembl_to_symbol",
        "human_ambiguous_symbol",
    }
)
S0_TOOL_BY_FAMILY = {
    "human_symbol_to_ensembl": "lookup_human_gene_symbol",
    "human_ambiguous_symbol": "lookup_human_gene_symbol",
    "human_ensembl_to_symbol": "lookup_human_ensembl_gene_id",
}
S0_TOOL_ARGUMENT_BY_FAMILY = {
    "human_symbol_to_ensembl": "gene_symbol",
    "human_ambiguous_symbol": "gene_symbol",
    "human_ensembl_to_symbol": "gene_id",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SHA256_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class WorldModelMetadata:
    """Dataclass that stores metadata for multiplex world-model records.
    used for later curriculum stages that require graph context"""

    book_mode: str
    step: str
    question_family: str
    multiplex_id: str
    layer_scope: str
    layer_ids: tuple[str, ...]
    graph_version: str
    evidence_ids: tuple[str, ...] = ()
    context_id: str | None = None
    registry_hash: str | None = None
    ontology_bundle_hash: str | None = None
    context_identity_sha256: str | None = None
    gene_universe_hash: str | None = None
    construction_recipe_hash: str | None = None
    flist_hash: str | None = None
    schema_version: str = WORLD_MODEL_SCHEMA_VERSION
    entity_namespace: str = ENTITY_NAMESPACE
    answer_format: str = ANSWER_FORMAT

    def __post_init__(self) -> None:
        # check that model is consistent with the schema version
        if self.schema_version != WORLD_MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {WORLD_MODEL_SCHEMA_VERSION!r}"
            )
        # check valid book mode
        if self.book_mode not in BOOK_MODES:
            raise ValueError(
                f"book_mode must be one of {sorted(BOOK_MODES)}"
            )
        # curriculum stage
        if self.step not in STEPS:
            raise ValueError(f"step must be one of {list(STEPS)}")
        if not self.question_family:
            raise ValueError("question_family cannot be empty")
        if not self.multiplex_id or not self.graph_version:
            raise ValueError("multiplex_id and graph_version are required")
        if self.context_id is not None and not self.context_id.strip():
            raise ValueError(
                "context_id must be a non-empty ontology address when supplied"
            )
        for field_name in (
            "registry_hash",
            "ontology_bundle_hash",
            "context_identity_sha256",
            "gene_universe_hash",
            "construction_recipe_hash",
            "flist_hash",
        ):
            value = getattr(self, field_name)
            if value is not None and SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(
                    f"{field_name} must be a lowercase SHA-256 digest"
                )
        if self.layer_scope not in {
            "single_layer",
            "layer_subset",
            "all_layers",
        }:
            raise ValueError(
                "layer_scope must be single_layer, layer_subset, or all_layers"
            )
        if self.layer_scope != "all_layers" and not self.layer_ids:
            raise ValueError(
                "single_layer and layer_subset metadata require layer_ids"
            )
        if self.entity_namespace != ENTITY_NAMESPACE:
            raise ValueError(
                f"entity_namespace must be {ENTITY_NAMESPACE!r}"
            )
        if self.answer_format != ANSWER_FORMAT:
            raise ValueError("answer_format must be 'json'")
        if self.book_mode == "closed_book" and self.step != "S1":
            raise ValueError(
                "Closed-book graph records are restricted to S1"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorldModelMetadata":
        """Create graph metadata from one JSON object."""

        return cls(
            schema_version=str(
                payload.get("schema_version", WORLD_MODEL_SCHEMA_VERSION)
            ),
            book_mode=str(payload["book_mode"]),
            step=str(payload["step"]),
            question_family=str(payload["question_family"]),
            multiplex_id=str(payload["multiplex_id"]),
            layer_scope=str(payload["layer_scope"]),
            layer_ids=tuple(str(value) for value in payload["layer_ids"]),
            graph_version=str(payload["graph_version"]),
            evidence_ids=tuple(
                str(value) for value in payload.get("evidence_ids", [])
            ),
            context_id=(
                str(payload["context_id"])
                if payload.get("context_id") is not None
                else None
            ),
            registry_hash=(
                str(payload["registry_hash"])
                if payload.get("registry_hash") is not None
                else None
            ),
            ontology_bundle_hash=(
                str(payload["ontology_bundle_hash"])
                if payload.get("ontology_bundle_hash") is not None
                else None
            ),
            context_identity_sha256=(
                str(payload["context_identity_sha256"])
                if payload.get("context_identity_sha256") is not None
                else None
            ),
            gene_universe_hash=(
                str(payload["gene_universe_hash"])
                if payload.get("gene_universe_hash") is not None
                else None
            ),
            construction_recipe_hash=(
                str(payload["construction_recipe_hash"])
                if payload.get("construction_recipe_hash") is not None
                else None
            ),
            flist_hash=(
                str(payload["flist_hash"])
                if payload.get("flist_hash") is not None
                else None
            ),
            entity_namespace=str(
                payload.get("entity_namespace", ENTITY_NAMESPACE)
            ),
            answer_format=str(payload.get("answer_format", ANSWER_FORMAT)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert graph metadata to one JSON object."""

        payload = asdict(self)
        payload["layer_ids"] = list(self.layer_ids)
        payload["evidence_ids"] = list(self.evidence_ids)
        for field_name in (
            "context_id",
            "registry_hash",
            "ontology_bundle_hash",
            "context_identity_sha256",
            "gene_universe_hash",
            "construction_recipe_hash",
            "flist_hash",
        ):
            if payload[field_name] is None:
                payload.pop(field_name)
        return payload


@dataclass(frozen=True)
class IdentifierToolSFTMetadata:
    """Store metadata for the S0 tool trajectory corpus."""

    book_mode: str
    step: str
    question_family: str
    species_taxon_id: str
    ensembl_release: str
    identifier_registry_id: str
    system_prompt_sha256: str
    tool_contract_version: str
    schema_version: str = IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION
    target_format: str = TOOL_TRAJECTORY_FORMAT

    def __post_init__(self) -> None:
        if self.schema_version != IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must identify the S0 tool trajectory schema"
            )

        if self.book_mode != "tool_call":
            raise ValueError("S0 trajectory book_mode must be tool_call")

        if self.step != "S0":
            raise ValueError(
                "S0 tool metadata requires step 'S0'"
            )

        if self.question_family not in S0_FAMILIES:
            raise ValueError(
                f"question_family must be one of {sorted(S0_FAMILIES)}"
            )

        if self.species_taxon_id != HUMAN_TAXON_ID:
            raise ValueError(
                f"species_taxon_id must be {HUMAN_TAXON_ID!r}"
            )

        if self.ensembl_release != ENSEMBL_RELEASE:
            raise ValueError(
                f"ensembl_release must be {ENSEMBL_RELEASE!r}"
            )

        if SHA256_ID_PATTERN.fullmatch(
            self.identifier_registry_id
        ) is None:
            raise ValueError(
                "identifier_registry_id must be a sha256 URI"
            )

        if SHA256_PATTERN.fullmatch(
            self.system_prompt_sha256
        ) is None:
            raise ValueError(
                "system_prompt_sha256 must be a lowercase SHA-256 digest"
            )

        if (
            self.tool_contract_version
            != IDENTIFIER_TOOL_CONTRACT_VERSION
        ):
            raise ValueError(
                "The S0 identifier tool contract version changed"
            )

        if self.target_format != TOOL_TRAJECTORY_FORMAT:
            raise ValueError(
                "target_format must be "
                f"{TOOL_TRAJECTORY_FORMAT!r}"
            )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "IdentifierToolSFTMetadata":
        """Create S0 tool metadata from one JSON object."""

        return cls(
            schema_version=str(
                payload.get(
                    "schema_version",
                    IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
                )
            ),
            book_mode=str(payload["book_mode"]),
            step=str(payload["step"]),
            question_family=str(payload["question_family"]),
            species_taxon_id=str(payload["species_taxon_id"]),
            ensembl_release=str(payload["ensembl_release"]),
            identifier_registry_id=str(
                payload["identifier_registry_id"]
            ),
            system_prompt_sha256=str(
                payload["system_prompt_sha256"]
            ),
            tool_contract_version=str(
                payload["tool_contract_version"]
            ),
            target_format=str(
                payload.get(
                    "target_format",
                    TOOL_TRAJECTORY_FORMAT,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert S0 tool metadata to one JSON object."""

        return asdict(self)


WorldModelMetadataType = (
    WorldModelMetadata
    | IdentifierToolSFTMetadata
)


def metadata_from_dict(
    payload: Mapping[str, Any],
) -> WorldModelMetadataType:
    """Parse metadata from its explicit schema tag."""

    schema_version = str(
        payload.get("schema_version", WORLD_MODEL_SCHEMA_VERSION)
    )
    if schema_version == IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION:
        return IdentifierToolSFTMetadata.from_dict(payload)
    if schema_version == WORLD_MODEL_SCHEMA_VERSION:
        return WorldModelMetadata.from_dict(payload)
    raise ValueError(
        f"Unsupported world-model schema_version: {schema_version!r}"
    )


@dataclass(frozen=True)
class IdentifierToolSFTRecord:
    """Store one S0 tool trajectory record."""

    record_id: str
    metadata: IdentifierToolSFTMetadata
    system: str
    question: str
    input: Mapping[str, Any]
    tools: tuple[Mapping[str, Any], ...]
    assistant_tool_call: Mapping[str, Any]
    split: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    assistant_thinking: str | None = None
    tool_result: Mapping[str, Any] | None = None
    assistant_final: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("record_id must be a non-empty string")

        if not isinstance(self.metadata, IdentifierToolSFTMetadata):
            raise TypeError(
                "metadata must be IdentifierToolSFTMetadata"
            )

        if not isinstance(self.system, str) or not self.system:
            raise ValueError("system must be a non-empty string")

        observed_prompt_sha256 = hashlib.sha256(
            self.system.encode("utf-8")
        ).hexdigest()

        if (
            observed_prompt_sha256
            != self.metadata.system_prompt_sha256
        ):
            raise ValueError(
                "The system prompt does not match its SHA-256 value"
            )

        if not isinstance(self.question, str) or not self.question:
            raise ValueError("question must be a non-empty string")

        if not isinstance(self.input, Mapping) or len(self.input) != 1:
            raise ValueError(
                "input must contain exactly one tool argument"
            )
        expected_argument = S0_TOOL_ARGUMENT_BY_FAMILY[
            self.metadata.question_family
        ]
        if set(self.input) != {expected_argument}:
            raise ValueError(
                "The input field differs from the question family"
            )

        if self.split not in {"train", "val"}:
            raise ValueError("split must be train or val")

        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be one object")

        if not isinstance(self.tools, tuple) or not self.tools:
            raise ValueError("tools must be a non-empty tuple")

        tool_names: set[str] = set()

        for tool in self.tools:
            if not isinstance(tool, Mapping):
                raise TypeError("Each tool must be one object")

            if tool.get("type") != "function":
                raise ValueError("Each tool must have type 'function'")

            function = tool.get("function")
            if not isinstance(function, Mapping):
                raise ValueError(
                    "Each tool must contain one function object"
                )

            tool_name = function.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError(
                    "Each tool function must have a name"
                )

            if tool_name in tool_names:
                raise ValueError("Tool names must be unique")

            tool_names.add(tool_name)

        call = self.assistant_tool_call
        if not isinstance(call, Mapping):
            raise TypeError(
                "assistant_tool_call must be one object"
            )

        if set(call) != {"type", "function"}:
            raise ValueError(
                "assistant_tool_call has incorrect fields"
            )

        if call.get("type") != "function":
            raise ValueError(
                "assistant_tool_call must have type 'function'"
            )

        function = call.get("function")
        if not isinstance(function, Mapping):
            raise ValueError(
                "assistant_tool_call must contain one function object"
            )

        if set(function) != {"name", "arguments"}:
            raise ValueError(
                "The assistant function has incorrect fields"
            )

        tool_name = function.get("name")
        if tool_name not in tool_names:
            raise ValueError(
                "The assistant function is not a declared tool"
            )
        expected_tool = S0_TOOL_BY_FAMILY[
            self.metadata.question_family
        ]
        if tool_name != expected_tool:
            raise ValueError(
                "The assistant function differs from the question family"
            )

        arguments = function.get("arguments")
        if not isinstance(arguments, Mapping):
            raise TypeError(
                "The assistant arguments must be one object"
            )

        if dict(arguments) != dict(self.input):
            raise ValueError(
                "The assistant arguments must match the record input"
            )

        if (
            not isinstance(self.assistant_thinking, str)
            or not self.assistant_thinking.strip()
        ):
            raise ValueError(
                "assistant_thinking must be a non-empty string"
            )
        if (
            not isinstance(self.tool_result, Mapping)
            or not self.tool_result
        ):
            raise ValueError(
                "tool_result must be a non-empty object."
            )
        if (
            not isinstance(self.assistant_final, str)
            or not self.assistant_final.strip()
        ):
            raise ValueError(
                "assistant_final must be a non-empty string."
            )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "IdentifierToolSFTRecord":
        """Create one S0 tool trajectory record from JSON data."""

        if not isinstance(payload, Mapping):
            raise TypeError(
                "An S0 tool trajectory record must be one object"
            )

        required_fields = {
            "record_id",
            "metadata",
            "system",
            "question",
            "input",
            "tools",
            "assistant_tool_call",
            "assistant_thinking",
            "tool_result",
            "assistant_final",
            "split",
            "provenance",
        }

        observed_fields = set(payload)
        if observed_fields != required_fields:
            missing_fields = sorted(
                required_fields - observed_fields
            )
            extra_fields = sorted(
                observed_fields - required_fields
            )
            raise ValueError(
                "The S0 tool trajectory record has incorrect fields: "
                f"missing={missing_fields}, extra={extra_fields}"
            )

        tools = payload["tools"]
        if not isinstance(tools, list):
            raise TypeError("tools must be a JSON list")

        metadata = metadata_from_dict(payload["metadata"])

        return cls(
            record_id=payload["record_id"],
            metadata=metadata,
            system=payload["system"],
            question=payload["question"],
            input=payload["input"],
            tools=tuple(tools),
            assistant_tool_call=payload["assistant_tool_call"],
            split=payload["split"],
            provenance=payload["provenance"],
            assistant_thinking=payload.get("assistant_thinking"),
            tool_result=payload.get("tool_result"),
            assistant_final=payload.get("assistant_final"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert one S0 tool trajectory record to JSON data."""

        payload = {
            "record_id": self.record_id,
            "metadata": self.metadata.to_dict(),
            "system": self.system,
            "question": self.question,
            "input": dict(self.input),
            "tools": [
                dict(tool)
                for tool in self.tools
            ],
            "assistant_tool_call": dict(
                self.assistant_tool_call
            ),
            "split": self.split,
            "provenance": dict(self.provenance),
        }

        payload.update(
            {
                "assistant_thinking": self.assistant_thinking,
                "tool_result": dict(self.tool_result or {}),
                "assistant_final": self.assistant_final,
            }
        )

        return payload

    def to_messages(self) -> list[dict[str, Any]]:
        """Return the messages for the model chat template."""

        function = self.assistant_tool_call["function"]

        assistant_call = {
            "type": "function",
            "function": {
                "name": function["name"],
                "arguments": dict(function["arguments"]),
            },
        }

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system,
            },
            {
                "role": "user",
                "content": self.question,
            },
        ]

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [assistant_call],
        }

        assistant_message["thinking"] = self.assistant_thinking
        messages.extend(
            [
                assistant_message,
                {
                    "role": "tool",
                    "content": dict(self.tool_result or {}),
                },
                {
                    "role": "assistant",
                    "content": self.assistant_final,
                },
            ]
        )
        return messages


@dataclass(frozen=True)
class FieldValidatorSpec:
    """Store one answer field validator contract."""

    primitive: str
    required: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FieldValidatorSpec":
        """Create one validator contract from one JSON object."""

        return cls(
            primitive=str(payload["primitive"]),
            required=bool(payload.get("required", True)),
            options=dict(payload.get("options", {})),
        )


@dataclass(frozen=True)
class WorldModelRecord:
    """Store one train or scored evaluation record."""

    record_id: str
    metadata: WorldModelMetadataType
    question: str
    answer: Mapping[str, Any]
    validators: Mapping[str, FieldValidatorSpec]
    split: str | None = None
    system: str | None = None
    input: Mapping[str, Any] | None = None
    context: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id or not self.question:
            raise ValueError("record_id and question are required")
        if not isinstance(self.answer, Mapping):
            raise TypeError("answer must be one JSON object")
        if self.split is not None and self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        # Graph records require context unless the contract is closed book.
        if self.metadata.book_mode == "closed_book":
            if self.context is not None:
                raise ValueError(
                    "Closed-book records cannot contain graph context"
                )
        elif self.context is None:
            raise ValueError(
                "Open-book and tool-call records require context"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorldModelRecord":
        """Create one train or scored record from one JSON object."""

        validators = {
            str(field_name): FieldValidatorSpec.from_dict(spec)
            for field_name, spec in payload["validators"].items()
        }
        return cls(
            record_id=str(payload["record_id"]),
            metadata=metadata_from_dict(payload["metadata"]),
            question=str(payload["question"]),
            system=(
                str(payload["system"])
                if payload.get("system") is not None
                else None
            ),
            input=payload.get("input"),
            context=payload.get("context"),
            answer=payload["answer"],
            validators=validators,
            split=(
                str(payload["split"])
                if payload.get("split") is not None
                else None
            ),
            provenance=payload.get("provenance", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert one train or scored record to one JSON object."""

        payload = {
            "record_id": self.record_id,
            "metadata": self.metadata.to_dict(),
            "question": self.question,
            "context": self.context,
            "answer": dict(self.answer),
            "validators": {
                name: {
                    "primitive": spec.primitive,
                    "required": spec.required,
                    "options": dict(spec.options),
                }
                for name, spec in self.validators.items()
            },
            "provenance": dict(self.provenance),
        }
        if self.input is not None:
            payload["input"] = dict(self.input)
        if self.system is not None:
            payload["system"] = self.system
        if self.split is not None:
            payload["split"] = self.split
        return payload
