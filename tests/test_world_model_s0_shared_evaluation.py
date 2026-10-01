"""Test structured S0 generation and evaluation."""

from __future__ import annotations

import unittest

from scripts.evaluate_world_model_v2_s0 import (
    parse_shared_final_action,
    score_tool_trajectory_record,
)


CALL_EXPLANATION = (
    "The task needs an exact ID for TP53, so I will copy the visible symbol "
    "into the lookup tool."
)
FINAL_EXPLANATION = (
    "The tool linked TP53 to ENSG00000141510, so I will return both verified "
    "identifiers and continue."
)
FINAL_ANSWER = (
    '{"status":"resolved","result":{"gene_id":"ENSG00000141510",'
    '"gene_symbol":"TP53"},"continuation_decision":"continue",'
    '"summary":"<GENE:ENSG00000141510|TP53> is the resolved human gene."}'
)
TOOL_CALL = {
    "name": "lookup_human_gene_symbol",
    "arguments": {"gene_symbol": "TP53"},
}
TOOL_PAYLOAD = {
    "status": "resolved",
    "gene_id": "ENSG00000141510",
    "gene_symbol": "TP53",
}
ENCODED_CALL = (
    "<|start|>assistant<|channel|>analysis<|message|>"
    f"{CALL_EXPLANATION}<|end|><|start|>assistant "
    "to=functions.lookup_human_gene_symbol"
    "<|channel|>commentary json<|message|>"
    '{"gene_symbol":"TP53"}<|call|>'
)
ENCODED_FINAL = (
    "<|start|>assistant<|channel|>analysis<|message|>"
    f"{FINAL_EXPLANATION}<|end|><|start|>assistant"
    "<|channel|>final<|message|>"
    f"{FINAL_ANSWER}<|return|>"
)


class _Result:
    """Store one local version 2 tool result."""

    payload = TOOL_PAYLOAD
    provenance = {"network_used": False}


class _Runtime:
    """Return one exact local tool result."""

    def execute(self, name: str, arguments: object) -> _Result:
        """Check and execute one expected call."""

        if name != TOOL_CALL["name"] or arguments != TOOL_CALL["arguments"]:
            raise AssertionError("The test received an unexpected tool call")
        return _Result()


def _answer_key() -> dict[str, object]:
    """Return one private shared trace answer."""

    return {
        "record_id": "record-shared",
        "fact_id": "fact-tp53",
        "family": "human_symbol_to_ensembl",
        "expected_status": "resolved",
        "expected_continuation_decision": "continue",
        "expected_tool_call": TOOL_CALL,
        "expected_tool_payload": TOOL_PAYLOAD,
        "expected_assistant_thinking": CALL_EXPLANATION,
        "expected_final_explanation": FINAL_EXPLANATION,
        "expected_final_answer": FINAL_ANSWER,
    }


def _prediction() -> dict[str, object]:
    """Return one generated shared trace prediction."""

    return {
        "record_id": "record-shared",
        "encoded_prediction": ENCODED_CALL,
        "assistant_thinking": CALL_EXPLANATION,
        "tool_payload": TOOL_PAYLOAD,
        "tool_error": None,
        "network_used_false": True,
        "disallowed_special_tokens": [],
        "encoded_final_answer": ENCODED_FINAL,
        "assistant_final_explanation": FINAL_EXPLANATION,
        "assistant_final": FINAL_ANSWER,
        "final_disallowed_special_tokens": [],
    }


class S0SharedEvaluationTests(unittest.TestCase):
    """Test final explanation and JSON answer checks."""

    def test_parser_returns_the_explanation_and_json(self) -> None:
        """Parse both parts of one final assistant action."""

        explanation, answer, error = parse_shared_final_action(ENCODED_FINAL)

        self.assertEqual(explanation, FINAL_EXPLANATION)
        self.assertEqual(answer, FINAL_ANSWER)
        self.assertIsNone(error)

    def test_exact_shared_trace_passes_each_structured_check(self) -> None:
        """Accept an exact tool result, explanation, and JSON answer."""

        result = score_tool_trajectory_record(
            {"record_id": "record-shared"},
            _answer_key(),
            _prediction(),
            runtime=_Runtime(),
        )

        self.assertTrue(result["valid_tool_trajectory"])
        self.assertTrue(result["structured_json_valid"])
        self.assertTrue(result["status_exact"])
        self.assertTrue(result["continuation_decision_exact"])
        self.assertTrue(result["final_explanation_present"])
        self.assertTrue(result["final_explanation_exact"])

    def test_out_of_order_answer_fields_fail_the_json_contract(self) -> None:
        """Reject JSON with the correct values in the wrong field order."""

        prediction = _prediction()
        wrong_answer = (
            '{"result":{"gene_id":"ENSG00000141510",'
            '"gene_symbol":"TP53"},"status":"resolved",'
            '"continuation_decision":"continue","summary":"Resolved."}'
        )
        prediction["encoded_final_answer"] = ENCODED_FINAL.replace(
            FINAL_ANSWER,
            wrong_answer,
        )
        prediction["assistant_final"] = wrong_answer

        result = score_tool_trajectory_record(
            {"record_id": "record-shared"},
            _answer_key(),
            prediction,
            runtime=_Runtime(),
        )

        self.assertFalse(result["structured_json_valid"])
        self.assertEqual(
            result["structured_answer_error"],
            "answer_field_order_invalid",
        )
        self.assertFalse(result["valid_tool_trajectory"])


if __name__ == "__main__":
    unittest.main()
