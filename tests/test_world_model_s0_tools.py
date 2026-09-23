"""Test the deterministic S0 identifier tools."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime.tools import ToolExecutionError
from runtime.world_model_training import (
    flatten_tool_sft_record_for_arrow,
    tokenize_tool_trajectory_for_sft,
)
from runtime.world_model_prompts import (
    S0_TOOL_TRAJECTORY_SYSTEM_PROMPT,
    s0_tool_trajectory_prompt_contract,
)
from runtime.world_model_schemas import (
    ENSEMBL_RELEASE,
    HUMAN_TAXON_ID,
    IDENTIFIER_TOOL_CONTRACT_VERSION,
    IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
    TOOL_TRAJECTORY_FORMAT,
    IdentifierToolSFTMetadata,
    IdentifierToolSFTRecord,
    metadata_from_dict,
)
from runtime.world_model_s0_tools import (
    ENSEMBL_TOOL_NAME,
    REGISTRY_SCHEMA_VERSION,
    SYMBOL_TOOL_NAME,
    S0IdentifierToolRuntime,
    identifier_tool_definitions,
)


def _registry_payload() -> dict[str, object]:
    """Return one valid test registry."""

    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "source": {"release": 116},
        "gene_symbols_by_id": {
            "ENSG00000000003": ["SHARED", "TSPAN6"],
            "ENSG00000000005": ["SHARED", "TNMD"],
            "ENSG00000000419": ["DPM1"],
        },
    }


def _write_registry(
    directory: Path,
    payload: object | None = None,
) -> tuple[Path, str]:
    """Write one test registry and return its file identity."""

    if payload is None:
        payload = _registry_payload()

    registry_path = directory / "registry.json"
    registry_text = json.dumps(payload, sort_keys=True)
    registry_path.write_text(registry_text, encoding="utf-8")

    registry_sha256 = hashlib.sha256(
        registry_text.encode("utf-8")
    ).hexdigest()

    return registry_path, registry_sha256


def _tool_metadata_payload() -> dict[str, str]:
    """Return one valid S0 tool trajectory metadata object."""

    return {
        "schema_version": IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
        "book_mode": "tool_call",
        "step": "S0",
        "question_family": "human_symbol_to_ensembl",
        "species_taxon_id": HUMAN_TAXON_ID,
        "ensembl_release": ENSEMBL_RELEASE,
        "identifier_registry_id": "sha256:" + "0" * 64,
        "system_prompt_sha256": "1" * 64,
        "tool_contract_version": IDENTIFIER_TOOL_CONTRACT_VERSION,
        "target_format": TOOL_TRAJECTORY_FORMAT,
    }


def _tool_record_payload() -> dict[str, object]:
    """Return one valid S0 tool trajectory record."""

    prompt_contract = s0_tool_trajectory_prompt_contract()
    metadata = _tool_metadata_payload()
    metadata["system_prompt_sha256"] = (
        prompt_contract.system_prompt_sha256
    )

    return {
        "record_id": "wm2_s0_tool_0001",
        "metadata": metadata,
        "system": prompt_contract.system_prompt,
        "question": "Resolve this human gene symbol: DPM1.",
        "input": {"gene_symbol": "DPM1"},
        "tools": identifier_tool_definitions(),
        "assistant_tool_call": {
            "type": "function",
            "function": {
                "name": SYMBOL_TOOL_NAME,
                "arguments": {"gene_symbol": "DPM1"},
            },
        },
        "split": "train",
        "provenance": {
            "fact_id": "fact_0001",
            "fact_role": "train",
            "prompt_form_id": "train",
        },
        "assistant_thinking": (
            "The next step requires a human Ensembl gene ID. "
            "I need to resolve DPM1 against the Ensembl release "
            "116 registry."
        ),
        "tool_result": {
            "status": "resolved",
            "gene_id": "ENSG00000000419",
            "gene_symbol": "DPM1",
        },
        "assistant_final": (
            "The Ensembl gene ID for DPM1 is ENSG00000000419."
        ),
    }


class _TrajectoryTokenizer:
    """Render stable token segments for one trajectory test."""

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **_: object,
    ) -> list[int]:
        """Return one token for each message segment."""

        token_ids = [1]
        for index, message in enumerate(messages):
            role = message["role"]
            if role == "system":
                token_ids.append(10)
            elif role == "user":
                token_ids.append(20)
            elif role == "tool":
                token_ids.append(50)
            elif "tool_calls" in message:
                has_future_final = any(
                    future["role"] == "assistant"
                    and "tool_calls" not in future
                    for future in messages[index + 1:]
                )
                if message.get("thinking") and not has_future_final:
                    token_ids.extend([30, 31])
                token_ids.append(40)
            else:
                token_ids.append(60)
        return token_ids


class S0IdentifierToolTests(unittest.TestCase):
    """Test the S0 identifier tool runtime."""

    def setUp(self) -> None:
        """Create one test registry."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        directory = Path(self.temporary_directory.name)
        self.registry_path, self.registry_sha256 = _write_registry(
            directory
        )

    def test_tool_definitions_have_exact_names(self) -> None:
        """Check the public tool names."""

        definitions = identifier_tool_definitions()

        self.assertEqual(
            [item["function"]["name"] for item in definitions],
            [SYMBOL_TOOL_NAME, ENSEMBL_TOOL_NAME],
        )

    def test_runtime_loads_both_lookup_indexes(self) -> None:
        """Check both registry indexes."""

        runtime = S0IdentifierToolRuntime.from_registry(
            self.registry_path,
            expected_sha256=self.registry_sha256,
        )

        self.assertEqual(
            runtime.gene_symbols_by_id["ENSG00000000419"],
            ("DPM1",),
        )
        self.assertEqual(
            runtime.gene_ids_by_symbol["SHARED"],
            ("ENSG00000000003", "ENSG00000000005"),
        )

        with self.assertRaises(TypeError):
            runtime.gene_symbols_by_id["ENSG00000000419"] = (
                "CHANGED",
            )

    def _load_runtime(self) -> S0IdentifierToolRuntime:
        """Load the test registry."""

        return S0IdentifierToolRuntime.from_registry(
            self.registry_path,
            expected_sha256=self.registry_sha256,
        )

    def test_symbol_lookup_returns_a_resolved_result(self) -> None:
        """Check one symbol with one gene ID."""

        result = self._load_runtime().execute(
            SYMBOL_TOOL_NAME,
            {"gene_symbol": "DPM1"},
        )

        self.assertEqual(
            result.payload,
            {
                "status": "resolved",
                "gene_id": "ENSG00000000419",
                "gene_symbol": "DPM1",
            },
        )
        self.assertFalse(result.is_empty)

    def test_symbol_lookup_returns_an_ambiguous_result(self) -> None:
        """Check one symbol with multiple gene IDs."""

        result = self._load_runtime().execute(
            SYMBOL_TOOL_NAME,
            {"gene_symbol": "SHARED"},
        )

        self.assertEqual(
            result.payload,
            {
                "status": "ambiguous",
                "gene_symbol": "SHARED",
                "candidate_gene_ids": [
                    "ENSG00000000003",
                    "ENSG00000000005",
                ],
                "action": "defer",
            },
        )
        self.assertFalse(result.is_empty)

    def test_gene_id_lookup_returns_all_symbols(self) -> None:
        """Check one gene ID with multiple symbols."""

        result = self._load_runtime().execute(
            ENSEMBL_TOOL_NAME,
            {"gene_id": "ENSG00000000003"},
        )

        self.assertEqual(
            result.payload,
            {
                "status": "resolved",
                "gene_id": "ENSG00000000003",
                "gene_symbols": ["SHARED", "TSPAN6"],
            },
        )
        self.assertFalse(result.is_empty)

    def test_absent_values_return_empty_results(self) -> None:
        """Check valid values that the registry does not contain."""

        runtime = self._load_runtime()
        symbol_result = runtime.execute(
            SYMBOL_TOOL_NAME,
            {"gene_symbol": "ABSENT"},
        )
        gene_result = runtime.execute(
            ENSEMBL_TOOL_NAME,
            {"gene_id": "ENSG99999999999"},
        )

        self.assertEqual(
            symbol_result.payload,
            {
                "status": "not_found",
                "gene_symbol": "ABSENT",
                "candidate_gene_ids": [],
            },
        )
        self.assertTrue(symbol_result.is_empty)
        self.assertEqual(
            gene_result.payload,
            {
                "status": "not_found",
                "gene_id": "ENSG99999999999",
                "gene_symbols": [],
            },
        )
        self.assertTrue(gene_result.is_empty)

    def test_results_include_pinned_provenance(self) -> None:
        """Check the registry identity and network policy."""

        result = self._load_runtime().execute(
            SYMBOL_TOOL_NAME,
            {"gene_symbol": "DPM1"},
        )

        self.assertEqual(
            result.provenance,
            {
                "tool_name": SYMBOL_TOOL_NAME,
                "tool_contract_version": (
                    "mentor-rl-world-model-s0-identifier-tools-v1"
                ),
                "source": "pinned_registry",
                "registry_schema_version": REGISTRY_SCHEMA_VERSION,
                "registry_sha256": self.registry_sha256,
                "ensembl_release": "Ensembl_116",
                "network_used": False,
            },
        )

    def test_runtime_rejects_invalid_calls(self) -> None:
        """Reject calls that break the tool contract."""

        runtime = self._load_runtime()
        cases = (
            (SYMBOL_TOOL_NAME, [], "one JSON object"),
            (SYMBOL_TOOL_NAME, {}, "only 'gene_symbol'"),
            (
                SYMBOL_TOOL_NAME,
                {"gene_symbol": "DPM1", "extra": True},
                "only 'gene_symbol'",
            ),
            (
                SYMBOL_TOOL_NAME,
                {"gene_symbol": " DPM1"},
                "outer whitespace",
            ),
            (
                SYMBOL_TOOL_NAME,
                {"gene_symbol": "ＤＰＭ１"},
                "canonical Unicode",
            ),
            (
                ENSEMBL_TOOL_NAME,
                {"gene_id": "DPM1"},
                "must match ENSG",
            ),
            ("unknown", {}, "Unknown S0 identifier tool"),
        )

        for tool_name, arguments, error_text in cases:
            with self.subTest(
                tool_name=tool_name,
                arguments=arguments,
            ):
                with self.assertRaisesRegex(
                    ToolExecutionError,
                    error_text,
                ):
                    runtime.execute(tool_name, arguments)

    def test_runtime_rejects_a_changed_file_identity(self) -> None:
        """Reject a registry with a different file identity."""

        with self.assertRaisesRegex(
            ToolExecutionError,
            "SHA-256 value changed",
        ):
            S0IdentifierToolRuntime.from_registry(
                self.registry_path,
                expected_sha256="0" * 64,
            )

    def test_runtime_rejects_invalid_registry_contracts(self) -> None:
        """Reject invalid registry structures and values."""

        wrong_schema = _registry_payload()
        wrong_schema["schema_version"] = "wrong"

        wrong_release = _registry_payload()
        wrong_release["source"] = {"release": 115}

        invalid_gene_id = _registry_payload()
        invalid_gene_id["gene_symbols_by_id"] = {
            "DPM1": ["DPM1"],
        }

        empty_symbols = _registry_payload()
        empty_symbols["gene_symbols_by_id"] = {
            "ENSG00000000419": [],
        }

        unsorted_symbols = _registry_payload()
        unsorted_symbols["gene_symbols_by_id"] = {
            "ENSG00000000003": ["TSPAN6", "SHARED"],
        }

        cases = (
            ([], "one JSON object"),
            (wrong_schema, "schema version changed"),
            (
                {
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "source": [],
                    "gene_symbols_by_id": {},
                },
                "source must be one object",
            ),
            (wrong_release, "must use Ensembl release 116"),
            (
                {
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "source": {"release": 116},
                    "gene_symbols_by_id": [],
                },
                "contains no gene mappings",
            ),
            (
                {
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "source": {"release": 116},
                    "gene_symbols_by_id": {},
                },
                "contains no gene mappings",
            ),
            (invalid_gene_id, "must match ENSG"),
            (empty_symbols, "has no symbols"),
            (unsorted_symbols, "are not canonical"),
        )

        for index, (payload, error_text) in enumerate(cases):
            with self.subTest(index=index, error_text=error_text):
                case_directory = (
                    Path(self.temporary_directory.name) / str(index)
                )
                case_directory.mkdir()
                registry_path, registry_sha256 = _write_registry(
                    case_directory,
                    payload,
                )
                with self.assertRaisesRegex(
                    ToolExecutionError,
                    error_text,
                ):
                    S0IdentifierToolRuntime.from_registry(
                        registry_path,
                        expected_sha256=registry_sha256,
                    )

    def test_runtime_rejects_invalid_json(self) -> None:
        """Reject a registry that does not contain JSON."""

        registry_path = Path(self.temporary_directory.name) / "bad.json"
        registry_text = "not JSON"
        registry_path.write_text(registry_text, encoding="utf-8")
        registry_sha256 = hashlib.sha256(
            registry_text.encode("utf-8")
        ).hexdigest()

        with self.assertRaisesRegex(
            ToolExecutionError,
            "cannot be read",
        ):
            S0IdentifierToolRuntime.from_registry(
                registry_path,
                expected_sha256=registry_sha256,
            )


class IdentifierToolMetadataTests(unittest.TestCase):
    """Test the S0 tool metadata contract."""

    def test_trajectory_metadata_round_trip_uses_its_schema(self) -> None:
        """Check the tool trajectory metadata contract."""

        payload = _tool_metadata_payload()
        metadata = metadata_from_dict(payload)

        self.assertIsInstance(metadata, IdentifierToolSFTMetadata)
        self.assertEqual(metadata.to_dict(), payload)

    def test_trajectory_metadata_uses_its_default_format(self) -> None:
        """Select the trajectory format from the schema tag."""

        payload = _tool_metadata_payload()
        payload.pop("target_format")

        metadata = IdentifierToolSFTMetadata.from_dict(payload)

        self.assertEqual(metadata.target_format, TOOL_TRAJECTORY_FORMAT)

    def test_metadata_rejects_a_schema_format_mismatch(self) -> None:
        """Reject a non-trajectory target format."""

        payload = _tool_metadata_payload()
        payload["target_format"] = "tool_call"

        with self.assertRaisesRegex(ValueError, "target_format"):
            IdentifierToolSFTMetadata.from_dict(payload)

    def test_metadata_rejects_invalid_contract_fields(self) -> None:
        """Reject each invalid tool metadata field."""

        cases = (
            ("schema_version", "wrong", "schema_version"),
            ("book_mode", "closed_book", "book_mode"),
            ("step", "S1", "requires step"),
            ("question_family", "unknown", "question_family"),
            ("species_taxon_id", "NCBITaxon:10090", "species_taxon_id"),
            ("ensembl_release", "Ensembl_115", "ensembl_release"),
            ("identifier_registry_id", "not-a-hash", "sha256 URI"),
            ("system_prompt_sha256", "not-a-hash", "SHA-256 digest"),
            ("tool_contract_version", "wrong", "contract version"),
            ("target_format", "json", "target_format"),
        )

        for field_name, value, error_text in cases:
            with self.subTest(field_name=field_name):
                payload = _tool_metadata_payload()
                payload[field_name] = value
                with self.assertRaisesRegex(ValueError, error_text):
                    IdentifierToolSFTMetadata.from_dict(payload)


class IdentifierToolPromptTests(unittest.TestCase):
    """Test the S0 tool prompt contract."""

    def test_trajectory_prompt_requires_the_complete_exchange(
        self,
    ) -> None:
        """Require thought text, one tool call, and one answer."""

        contract = s0_tool_trajectory_prompt_contract()
        expected_sha256 = hashlib.sha256(
            S0_TOOL_TRAJECTORY_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()

        self.assertEqual(
            contract.system_prompt_sha256,
            expected_sha256,
        )
        self.assertIn(
            "Before the tool call",
            contract.system_prompt,
        )
        self.assertIn(
            "After the tool result",
            contract.system_prompt,
        )
        self.assertEqual(contract.allowed_book_modes, ("tool_call",))


class IdentifierToolRecordTests(unittest.TestCase):
    """Test the answer-free S0 tool trajectory record."""

    def test_trajectory_record_has_an_exact_json_round_trip(
        self,
    ) -> None:
        """Check the tool trajectory parser and serializer."""

        payload = _tool_record_payload()
        record = IdentifierToolSFTRecord.from_dict(payload)

        self.assertEqual(record.to_dict(), payload)
        json.dumps(record.to_dict())

    def test_trajectory_record_returns_five_messages(self) -> None:
        """Build the five semantic trajectory messages."""

        payload = _tool_record_payload()
        record = IdentifierToolSFTRecord.from_dict(payload)
        messages = record.to_messages()

        self.assertEqual(
            [message["role"] for message in messages],
            [
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
            ],
        )
        self.assertEqual(
            messages[2]["thinking"],
            payload["assistant_thinking"],
        )
        self.assertEqual(
            messages[2]["tool_calls"],
            [payload["assistant_tool_call"]],
        )
        self.assertEqual(
            messages[3]["content"],
            payload["tool_result"],
        )
        self.assertEqual(
            messages[4]["content"],
            payload["assistant_final"],
        )

    def test_trajectory_record_requires_each_new_field(
        self,
    ) -> None:
        """Require every trajectory field for the v6 schema."""

        for field_name in (
            "assistant_thinking",
            "tool_result",
            "assistant_final",
        ):
            with self.subTest(field_name=field_name):
                payload = _tool_record_payload()
                payload.pop(field_name)

                with self.assertRaisesRegex(
                    ValueError,
                    "incorrect fields",
                ):
                    IdentifierToolSFTRecord.from_dict(payload)

    def test_trajectory_record_rejects_empty_new_fields(
        self,
    ) -> None:
        """Reject an empty trajectory value."""

        cases = (
            ("assistant_thinking", ""),
            ("tool_result", {}),
            ("assistant_final", ""),
        )

        for field_name, value in cases:
            with self.subTest(field_name=field_name):
                payload = _tool_record_payload()
                payload[field_name] = value

                with self.assertRaisesRegex(
                    ValueError,
                    field_name,
                ):
                    IdentifierToolSFTRecord.from_dict(payload)

    def test_trajectory_record_flattens_all_scalar_fields(
        self,
    ) -> None:
        """Keep all trajectory fields in the Arrow row."""

        payload = _tool_record_payload()
        payload["provenance"]["prompt_form_id"] = "train"

        flattened = flatten_tool_sft_record_for_arrow(
            payload,
        )

        self.assertEqual(
            flattened["assistant_thinking"],
            payload["assistant_thinking"],
        )
        self.assertEqual(
            json.loads(flattened["tool_result_json"]),
            payload["tool_result"],
        )
        self.assertEqual(
            flattened["assistant_final"],
            payload["assistant_final"],
        )

    def test_trajectory_loss_masks_only_non_assistant_turns(
        self,
    ) -> None:
        """Train both assistant turns and mask the tool result."""

        record = IdentifierToolSFTRecord.from_dict(
            _tool_record_payload()
        )
        tokenized = tokenize_tool_trajectory_for_sft(
            record,
            _TrajectoryTokenizer(),
            max_length=16,
        )

        self.assertEqual(
            tokenized["input_ids"],
            [1, 10, 20, 30, 31, 40, 50, 60],
        )
        self.assertEqual(
            tokenized["completion_mask"],
            [0, 0, 0, 1, 1, 1, 0, 1],
        )

    def test_record_rejects_incorrect_top_level_fields(self) -> None:
        """Reject an absent field or an extra answer field."""

        missing_field = _tool_record_payload()
        missing_field.pop("assistant_tool_call")

        extra_answer = _tool_record_payload()
        extra_answer["answer"] = {
            "gene_id": "ENSG00000000419",
        }

        for payload in (missing_field, extra_answer):
            with self.subTest(fields=sorted(payload)):
                with self.assertRaisesRegex(
                    ValueError,
                    "incorrect fields",
                ):
                    IdentifierToolSFTRecord.from_dict(payload)

    def test_record_rejects_a_nonlist_tool_block(self) -> None:
        """Require a JSON list for the tool definitions."""

        payload = _tool_record_payload()
        payload["tools"] = tuple(payload["tools"])

        with self.assertRaisesRegex(TypeError, "JSON list"):
            IdentifierToolSFTRecord.from_dict(payload)

    def test_record_rejects_invalid_basic_fields(self) -> None:
        """Reject invalid record text, input, split, and provenance."""

        cases = (
            ("record_id", "", "record_id"),
            ("system", "", "system"),
            ("question", "", "question"),
            ("input", {}, "exactly one"),
            ("input", {"a": 1, "b": 2}, "exactly one"),
            ("split", "test", "train or val"),
            ("provenance", [], "one object"),
        )

        for field_name, value, error_text in cases:
            with self.subTest(field_name=field_name, value=value):
                payload = _tool_record_payload()
                payload[field_name] = value
                with self.assertRaisesRegex(
                    (TypeError, ValueError),
                    error_text,
                ):
                    IdentifierToolSFTRecord.from_dict(payload)

    def test_record_rejects_a_changed_system_prompt(self) -> None:
        """Require the prompt identity from the metadata."""

        payload = _tool_record_payload()
        payload["system"] = "A changed system prompt."

        with self.assertRaisesRegex(ValueError, "SHA-256 value"):
            IdentifierToolSFTRecord.from_dict(payload)

    def test_record_rejects_invalid_tool_definitions(self) -> None:
        """Reject invalid or duplicate tool definitions."""

        invalid_type = _tool_record_payload()
        invalid_type["tools"][0]["type"] = "command"

        absent_function = _tool_record_payload()
        absent_function["tools"][0].pop("function")

        absent_name = _tool_record_payload()
        absent_name["tools"][0]["function"]["name"] = ""

        duplicate_name = _tool_record_payload()
        duplicate_name["tools"][1]["function"]["name"] = (
            SYMBOL_TOOL_NAME
        )

        cases = (
            (invalid_type, "type 'function'"),
            (absent_function, "function object"),
            (absent_name, "must have a name"),
            (duplicate_name, "must be unique"),
        )

        for payload, error_text in cases:
            with self.subTest(error_text=error_text):
                with self.assertRaisesRegex(ValueError, error_text):
                    IdentifierToolSFTRecord.from_dict(payload)

    def test_record_rejects_invalid_assistant_calls(self) -> None:
        """Reject invalid tool names, fields, and arguments."""

        extra_call_field = _tool_record_payload()
        extra_call_field["assistant_tool_call"]["id"] = "call_1"

        wrong_call_type = _tool_record_payload()
        wrong_call_type["assistant_tool_call"]["type"] = "command"

        extra_function_field = _tool_record_payload()
        extra_function_field["assistant_tool_call"]["function"][
            "extra"
        ] = True

        undeclared_tool = _tool_record_payload()
        undeclared_tool["assistant_tool_call"]["function"]["name"] = (
            "unknown"
        )

        nonobject_arguments = _tool_record_payload()
        nonobject_arguments["assistant_tool_call"]["function"][
            "arguments"
        ] = "DPM1"

        changed_arguments = _tool_record_payload()
        changed_arguments["assistant_tool_call"]["function"][
            "arguments"
        ] = {"gene_symbol": "TP53"}

        wrong_family_tool = _tool_record_payload()
        wrong_family_tool["assistant_tool_call"]["function"]["name"] = (
            ENSEMBL_TOOL_NAME
        )

        cases = (
            (extra_call_field, "incorrect fields"),
            (wrong_call_type, "type 'function'"),
            (extra_function_field, "incorrect fields"),
            (undeclared_tool, "not a declared tool"),
            (nonobject_arguments, "must be one object"),
            (changed_arguments, "must match the record input"),
            (wrong_family_tool, "differs from the question family"),
        )

        for payload, error_text in cases:
            with self.subTest(error_text=error_text):
                with self.assertRaisesRegex(
                    (TypeError, ValueError),
                    error_text,
                ):
                    IdentifierToolSFTRecord.from_dict(
                        copy.deepcopy(payload)
                    )


if __name__ == "__main__":
    unittest.main()
