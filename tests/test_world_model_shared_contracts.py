"""Test the shared world-model answer and trace contracts."""

from __future__ import annotations

import copy
import json
import unittest

from runtime.world_model_training import (
    flatten_supervised_trace_for_arrow,
    tokenize_supervised_trace_for_sft,
)
from runtime.world_model_contracts import (
    SUPERVISED_TRACE_SCHEMA_VERSION,
    SUPERVISED_EXPLANATION_SCHEMA_VERSION,
    SUPERVISED_EXPLANATION_LOSS_CONTRACT,
    WORLD_MODEL_ANSWER_SCHEMA_VERSION,
    ExplanationReceipt,
    SupervisedTraceRecord,
    WorldModelContractError,
    parse_world_model_answer,
    serialize_world_model_answer,
    stable_json_sha256,
    text_sha256,
)


SYSTEM = "Use the listed tools when the task requires a lookup."
ANSWER = {
    "status": "resolved",
    "result": {
        "gene_symbol": "DPM1",
        "gene_id": "ENSG00000000419",
    },
    "continuation_decision": "stop",
    "summary": "<GENE:ENSG00000000419|DPM1> is the resolved gene.",
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_human_gene_symbol",
            "description": "Resolve one human gene symbol.",
            "parameters": {
                "type": "object",
                "properties": {"gene_symbol": {"type": "string"}},
                "required": ["gene_symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_human_ensembl_gene_id",
            "description": "Resolve one human Ensembl gene ID.",
            "parameters": {
                "type": "object",
                "properties": {"gene_id": {"type": "string"}},
                "required": ["gene_id"],
                "additionalProperties": False,
            },
        },
    },
]


class _TraceTokenizer:
    """Provide stable token segments for one shared trace."""

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **_: object,
    ) -> list[int]:
        """Return one stable segment for each message field."""

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
                has_future_assistant = any(
                    future["role"] == "assistant"
                    for future in messages[index + 1 :]
                )
                if message.get("thinking") and not has_future_assistant:
                    token_ids.extend([30, 31])
                token_ids.append(40)
            else:
                if message.get("thinking"):
                    token_ids.extend([60, 61])
                token_ids.append(70)
        return token_ids


class SharedContractVersionTests(unittest.TestCase):
    """Test the exact public contract names."""

    def test_contract_names_are_stable(self) -> None:
        """Keep each public version and loss name fixed."""

        self.assertEqual(
            WORLD_MODEL_ANSWER_SCHEMA_VERSION,
            "mentor-rl-world-model-answer-v1",
        )
        self.assertEqual(
            SUPERVISED_TRACE_SCHEMA_VERSION,
            "mentor-rl-world-model-supervised-trace-v1",
        )
        self.assertEqual(
            SUPERVISED_EXPLANATION_SCHEMA_VERSION,
            "mentor-rl-supervised-explanation-v1",
        )
        self.assertEqual(
            SUPERVISED_EXPLANATION_LOSS_CONTRACT,
            "supervised_explanation_trace_v1",
        )


def _receipt(
    action_id: str,
    explanation: str,
    explanation_input: dict[str, object],
) -> dict[str, object]:
    """Return one valid deterministic explanation receipt."""

    return {
        "explanation_contract_version": (
            SUPERVISED_EXPLANATION_SCHEMA_VERSION
        ),
        "explanation_template_id": "template_direct_lookup_v1",
        "explanation_template_sha256": text_sha256(
            "template_direct_lookup_v1:text"
        ),
        "explanation_rule_ids": ["rule_resolve_before_compare"],
        "explanation_source_refs": [
            "agents/world_model_v2_curriculum/S1-coordinate-system.md"
        ],
        "explanation_input_sha256": stable_json_sha256(
            {"action_id": action_id, "input": explanation_input}
        ),
        "explanation_sha256": text_sha256(explanation),
    }


def _trace_payload(
    call_count: int,
    *,
    book_mode: str | None = None,
) -> dict[str, object]:
    """Return one valid trace with the selected tool-call count."""

    symbols = ["DPM1", "GLP1R", "TP53", "EGFR"]
    gene_ids = [
        "ENSG00000000419",
        "ENSG00000112164",
        "ENSG00000141510",
        "ENSG00000146648",
    ]
    if call_count == 0:
        user_content = (
            "The registry maps DPM1 to ENSG00000000419. "
            "Return the resolved gene."
        )
    elif call_count == 4:
        user_content = (
            "Artifact A lists DPM1, GLP1R, TP53, and EGFR. "
            "Artifact B lists ENSG00000000419, ENSG00000112164, "
            "ENSG00000141510, and ENSG00000146648. "
            "Return the aligned Ensembl IDs."
        )
    else:
        user_content = "Resolve DPM1."
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
    ]
    for index in range(call_count):
        gene_symbol = symbols[index]
        gene_id = gene_ids[index]
        explanation = f"Resolve the visible symbol {gene_symbol} now."
        messages.extend(
            [
                {
                    "role": "assistant",
                    "assistant_explanation": explanation,
                    "explanation_receipt": _receipt(
                        f"receipt_call_{index}",
                        explanation,
                        {
                            "action_index": index,
                            "gene_symbol": gene_symbol,
                        },
                    ),
                    "tool_call": {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": "lookup_human_gene_symbol",
                            "arguments": {"gene_symbol": gene_symbol},
                        },
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "name": "lookup_human_gene_symbol",
                    "content": {
                        "status": "resolved",
                        "gene_id": gene_id,
                        "gene_symbol": gene_symbol,
                    },
                },
            ]
        )
    final_explanation = (
        "Use the visible registry mapping in the final answer."
        if call_count == 0
        else (
            "Use all four verified mappings in the final answer."
            if call_count == 4
            else "Use the verified tool result in the final answer."
        )
    )
    answer = copy.deepcopy(ANSWER)
    if call_count == 4:
        answer = {
            "status": "resolved",
            "result": {
                "id_space": "ensembl",
                "aligned": gene_ids,
                "only_in_a": [],
                "only_in_b": [],
                "unresolved": [],
            },
            "continuation_decision": "stop",
            "summary": (
                "<GENE:ENSG00000000419|DPM1>, "
                "<GENE:ENSG00000112164|GLP1R>, "
                "<GENE:ENSG00000141510|TP53>, and "
                "<GENE:ENSG00000146648|EGFR> occur in both artifacts."
            ),
        }
    messages.append(
        {
            "role": "assistant",
            "assistant_explanation": final_explanation,
            "explanation_receipt": _receipt(
                "receipt_final",
                final_explanation,
                {
                    "call_count": call_count,
                    "gene_ids": gene_ids[: max(call_count, 1)],
                },
            ),
            "content": serialize_world_model_answer(answer),
        }
    )
    return {
        "schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        "record_id": f"trace_{call_count}",
        "metadata": {
            "book_mode": book_mode or "tool_call",
            "question_family": (
                "cross_context_alignment"
                if call_count == 4
                else "human_symbol_to_ensembl"
            ),
            "answer_schema_version": WORLD_MODEL_ANSWER_SCHEMA_VERSION,
            "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
            "system_prompt_sha256": text_sha256(SYSTEM),
        },
        "tools": copy.deepcopy(TOOLS),
        "messages": messages,
        "split": "train",
        "provenance": {
            "fact_id": (
                "fact_alignment_4" if call_count == 4 else "fact_dpm1"
            ),
            "prompt_form_id": (
                "symbols_against_ids" if call_count == 4 else "direct"
            ),
        },
    }


class WorldModelAnswerTests(unittest.TestCase):
    """Test the fixed four-field answer contract."""

    def test_serializer_uses_the_exact_top_level_order(self) -> None:
        """Keep the decision fields before the readable summary."""

        text = serialize_world_model_answer(ANSWER)

        self.assertEqual(
            text,
            '{"status":"resolved","result":'
            '{"gene_symbol":"DPM1","gene_id":"ENSG00000000419"},'
            '"continuation_decision":"stop","summary":'
            '"<GENE:ENSG00000000419|DPM1> is the resolved gene."}',
        )
        self.assertEqual(parse_world_model_answer(text).to_dict(), ANSWER)

    def test_parser_rejects_wrong_field_order(self) -> None:
        """Reject valid fields when their source order differs."""

        text = (
            '{"result":{},"status":"resolved",'
            '"continuation_decision":"stop","summary":"Done."}'
        )

        with self.assertRaisesRegex(
            WorldModelContractError,
            "invalid order",
        ) as caught:
            parse_world_model_answer(text)

        self.assertEqual(caught.exception.code, "answer_field_order_invalid")

    def test_parser_rejects_a_non_string_value(self) -> None:
        """Reject a value that is not answer text."""

        with self.assertRaises(WorldModelContractError) as caught:
            parse_world_model_answer(ANSWER)

        self.assertEqual(caught.exception.code, "answer_not_text")

    def test_parser_rejects_missing_extra_and_duplicate_fields(self) -> None:
        """Reject each unsafe top-level field set."""

        cases = (
            (
                '{"status":"resolved","result":{},'
                '"continuation_decision":"stop"}',
                "answer_fields_invalid",
            ),
            (
                '{"status":"resolved","result":{},'
                '"continuation_decision":"stop","summary":"Done.",'
                '"extra":true}',
                "answer_fields_invalid",
            ),
            (
                '{"status":"resolved","status":"not_found",'
                '"result":{},"continuation_decision":"stop",'
                '"summary":"Done."}',
                "answer_duplicate_key",
            ),
            (
                '{"status":"resolved","result":{"x":1,"x":2},'
                '"continuation_decision":"stop","summary":"Done."}',
                "answer_duplicate_key",
            ),
        )

        for text, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(WorldModelContractError) as caught:
                    parse_world_model_answer(text)
                self.assertEqual(caught.exception.code, code)

    def test_parser_rejects_invalid_values(self) -> None:
        """Reject invalid enums, result types, and empty summaries."""

        cases = (
            ({**ANSWER, "status": "unknown"}, "answer_status_invalid"),
            ({**ANSWER, "result": []}, "answer_result_invalid"),
            (
                {**ANSWER, "continuation_decision": "wait"},
                "answer_continuation_decision_invalid",
            ),
            ({**ANSWER, "summary": ""}, "answer_summary_invalid"),
            (
                {
                    **ANSWER,
                    "status": "invalid_input",
                    "result": {"input": "ENST1"},
                },
                "answer_invalid_input_result",
            ),
        )

        for payload, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(WorldModelContractError) as caught:
                    serialize_world_model_answer(payload)
                self.assertEqual(caught.exception.code, code)

    def test_answer_supports_an_ensembl_id_without_a_symbol(self) -> None:
        """Keep a valid gene ID when its symbol is null."""

        answer = {
            "status": "resolved",
            "result": {
                "gene_id": "ENSG00000299999",
                "gene_symbol": None,
            },
            "continuation_decision": "continue",
            "summary": "<GENE:ENSG00000299999> has no known symbol.",
        }

        text = serialize_world_model_answer(answer)

        self.assertIn('"gene_symbol":null', text)
        self.assertEqual(
            parse_world_model_answer(text).to_dict(),
            answer,
        )


class ExplanationReceiptTests(unittest.TestCase):
    """Test deterministic explanation receipts."""

    def test_receipt_binds_inputs_and_explanation(self) -> None:
        """Accept one receipt with exact input and text hashes."""

        explanation = "Use the visible registry result."
        payload = _receipt(
            "receipt_1",
            explanation,
            {"gene_symbol": "DPM1"},
        )

        receipt = ExplanationReceipt.from_dict(
            payload,
            explanation=explanation,
        )

        self.assertEqual(receipt.to_dict(), payload)
        self.assertEqual(
            SUPERVISED_EXPLANATION_LOSS_CONTRACT,
            "supervised_explanation_trace_v1",
        )

    def test_receipt_rejects_extra_fields_and_hash_changes(self) -> None:
        """Reject an extra field and a changed explanation."""

        explanation = "Use the visible registry result."
        extra_payload = _receipt(
            "receipt_1",
            explanation,
            {"gene_symbol": "DPM1"},
        )
        extra_payload["extra"] = False

        with self.assertRaises(WorldModelContractError) as field_error:
            ExplanationReceipt.from_dict(
                extra_payload,
                explanation=explanation,
            )
        self.assertEqual(
            field_error.exception.code,
            "explanation_receipt_fields_invalid",
        )

        hash_payload = _receipt(
            "receipt_2",
            explanation,
            {"gene_symbol": "DPM1"},
        )
        with self.assertRaises(WorldModelContractError) as hash_error:
            ExplanationReceipt.from_dict(
                hash_payload,
                explanation="Changed text.",
            )
        self.assertEqual(hash_error.exception.code, "explanation_hash_mismatch")


class SupervisedTraceRecordTests(unittest.TestCase):
    """Test zero-call and tool-call supervised traces."""

    def test_zero_call_trace_maps_explanation_to_thinking(self) -> None:
        """Map a no-call tool trace to one final assistant turn."""

        record = SupervisedTraceRecord.from_dict(_trace_payload(0))
        chat = record.to_messages()

        self.assertEqual(record.tool_call_count, 0)
        self.assertEqual(
            [message["role"] for message in chat],
            ["system", "user", "assistant"],
        )
        self.assertEqual(
            chat[-1]["thinking"],
            "Use the visible registry mapping in the final answer.",
        )
        self.assertEqual(
            parse_world_model_answer(chat[-1]["content"]).to_dict(),
            ANSWER,
        )

    def test_one_call_trace_maps_each_transport_field(self) -> None:
        """Map one explanation, call, result, and final answer."""

        payload = _trace_payload(1)
        record = SupervisedTraceRecord.from_dict(payload)
        chat = record.to_messages()

        self.assertEqual(record.to_dict(), payload)
        self.assertEqual(record.tool_call_count, 1)
        self.assertEqual(
            [message["role"] for message in chat],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(
            chat[2]["thinking"],
            payload["messages"][2]["assistant_explanation"],
        )
        self.assertEqual(
            chat[2]["tool_calls"],
            [payload["messages"][2]["tool_call"]],
        )
        self.assertEqual(chat[3]["tool_call_id"], "call_0")
        self.assertEqual(
            chat[3]["content"],
            payload["messages"][3]["content"],
        )
        self.assertEqual(
            chat[3]["name"],
            "lookup_human_gene_symbol",
        )
        self.assertNotIn("explanation_receipt", chat[2])

    def test_four_call_trace_keeps_all_role_pairs(self) -> None:
        """Accept one trace that has four tool calls."""

        record = SupervisedTraceRecord.from_dict(_trace_payload(4))
        chat = record.to_messages()

        self.assertEqual(record.tool_call_count, 4)
        self.assertEqual(len(chat), 11)
        self.assertEqual(
            [message["role"] for message in chat],
            [
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
            ],
        )

    def test_trace_rejects_role_and_tool_contract_errors(self) -> None:
        """Reject a broken pair, an undeclared call, and a closed-book call."""

        broken_pair = _trace_payload(1)
        broken_pair["messages"][3]["role"] = "assistant"

        undeclared = _trace_payload(1)
        undeclared["messages"][2]["tool_call"]["function"]["name"] = (
            "unknown_tool"
        )

        closed_book = _trace_payload(1, book_mode="closed_book")

        changed_call_id = _trace_payload(1)
        changed_call_id["messages"][3]["tool_call_id"] = "call_other"

        duplicate_call_id = _trace_payload(4)
        duplicate_call_id["messages"][4]["tool_call"]["id"] = "call_0"
        duplicate_call_id["messages"][5]["tool_call_id"] = "call_0"

        changed_tool_name = _trace_payload(1)
        changed_tool_name["messages"][3]["name"] = (
            "lookup_human_ensembl_gene_id"
        )

        changed_loss = _trace_payload(1)
        changed_loss["metadata"]["loss_contract"] = "other"

        extra_record_field = _trace_payload(1)
        extra_record_field["answer_schema_version"] = (
            WORLD_MODEL_ANSWER_SCHEMA_VERSION
        )

        incomplete_tool = _trace_payload(1)
        incomplete_tool["tools"][0]["function"].pop("description")

        invalid_parameter_schemas = []
        for parameters in (
            {},
            {"type": "array"},
            {"type": "object", "properties": "not-an-object"},
            {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": "not-an-object",
                    }
                },
            },
            {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": 7,
                    }
                },
            },
            {
                "type": "object",
                "properties": {"gene_symbol": {"type": {}}},
            },
        ):
            invalid_tool = _trace_payload(1)
            invalid_tool["tools"][0]["function"]["parameters"] = parameters
            invalid_parameter_schemas.append(invalid_tool)

        cases = [
            (broken_pair, "tool_message_invalid"),
            (undeclared, "tool_call_undeclared"),
            (closed_book, "closed_book_tool_call"),
            (changed_call_id, "tool_call_id_mismatch"),
            (duplicate_call_id, "tool_call_id_duplicate"),
            (changed_tool_name, "tool_name_mismatch"),
            (changed_loss, "loss_contract_invalid"),
            (extra_record_field, "trace_record_fields_invalid"),
            (incomplete_tool, "tool_definition_invalid"),
            *[
                (invalid_tool, "tool_parameters_invalid")
                for invalid_tool in invalid_parameter_schemas
            ],
        ]
        for payload, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(WorldModelContractError) as caught:
                    SupervisedTraceRecord.from_dict(payload)
                self.assertEqual(caught.exception.code, code)

    def test_trace_rejects_changed_prompt_and_explanation(self) -> None:
        """Reject a changed prompt and explanation."""

        changed_prompt = _trace_payload(1)
        changed_prompt["messages"][0]["content"] = "Changed prompt."

        changed_explanation = _trace_payload(1)
        changed_explanation["messages"][-1]["assistant_explanation"] = (
            "Changed explanation."
        )

        cases = (
            (changed_prompt, "system_prompt_hash_mismatch"),
            (changed_explanation, "explanation_hash_mismatch"),
        )
        for payload, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(WorldModelContractError) as caught:
                    SupervisedTraceRecord.from_dict(payload)
                self.assertEqual(caught.exception.code, code)

    def test_json_round_trip_keeps_the_contract(self) -> None:
        """Keep one trace valid after a stable JSON round trip."""

        payload = _trace_payload(4)
        encoded = json.dumps(payload, sort_keys=True)
        decoded = json.loads(encoded)

        record = SupervisedTraceRecord.from_dict(decoded)

        self.assertEqual(record.to_dict(), decoded)
        self.assertTrue(
            record.to_messages()[-1]["content"].startswith(
                '{"status":"resolved","result":'
            )
        )


class SupervisedTraceTrainingTests(unittest.TestCase):
    """Test the shared trace loss mask."""

    def test_arrow_adapter_keeps_one_complete_shared_trace(self) -> None:
        """Keep the trace and its train identity in scalar fields."""

        payload = _trace_payload(1)
        payload["provenance"].update(
            {
                "fact_role": "train",
                "prompt_form_id": "train",
            }
        )

        flattened = flatten_supervised_trace_for_arrow(payload)

        self.assertEqual(flattened["record_id"], payload["record_id"])
        self.assertEqual(flattened["fact_id"], "fact_dpm1")
        self.assertEqual(flattened["prompt_form_id"], "train")
        self.assertEqual(
            json.loads(flattened["trace_json"]),
            payload,
        )

    def test_loss_mask_supports_zero_one_and_four_calls(self) -> None:
        """Train assistant tokens and mask prompt and tool tokens."""

        for call_count in (0, 1, 4):
            with self.subTest(call_count=call_count):
                record = SupervisedTraceRecord.from_dict(
                    _trace_payload(call_count)
                )
                result = tokenize_supervised_trace_for_sft(
                    record,
                    _TraceTokenizer(),
                    max_length=64,
                )

                expected_ids = [1, 10, 20]
                expected_mask = [0, 0, 0]
                for _ in range(call_count):
                    expected_ids.extend([30, 31, 40, 50])
                    expected_mask.extend([1, 1, 1, 0])
                expected_ids.extend([60, 61, 70])
                expected_mask.extend([1, 1, 1])

                self.assertEqual(result["input_ids"], expected_ids)
                self.assertEqual(
                    result["completion_mask"],
                    expected_mask,
                )

    def test_loss_mask_rejects_a_long_trace(self) -> None:
        """Reject a token sequence above the selected limit."""

        record = SupervisedTraceRecord.from_dict(_trace_payload(1))

        with self.assertRaisesRegex(RuntimeError, "exceeding"):
            tokenize_supervised_trace_for_sft(
                record,
                _TraceTokenizer(),
                max_length=9,
            )


if __name__ == "__main__":
    unittest.main()
