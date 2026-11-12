"""Actual local-tokenizer checks for untruncated S2 verification serialization."""

from copy import deepcopy
from pathlib import Path
import unittest

from runtime.world_model.s2_3_tokens import audit_rows, load_tokenizer, tokenize_trace, verification_clock
from runtime.world_model_training import (
    renderable_v3_tools, tokenize_v3_trace_for_sft, tokenizer_artifact_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/plain_base_tokenizer/tokenizer"


class FixedDateTokenizer:
    """Compare old and new serializers with the same explicit template clock."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        kwargs["strftime_now"] = verification_clock
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def _answer(value: str) -> dict:
    """Keep the real six-field answer wire with explicitly artificial test content."""
    return {"status": "resolved", "provenance": {"registry": [], "graph": []},
            "result": {"value": value}, "continuation_decision": "stop",
            "blocked_reason": None, "summary": "Unit serialization fixture."}


def _row() -> dict:
    """Two calls expose accidental removal or supervision of prior analysis/evidence."""
    tool = {"type": "function", "function": {"name": "lookup_human_gene_symbol",
            "parameters": {"type": "object", "properties": {"gene_symbol": {"type": "string"}},
                           "required": ["gene_symbol"], "additionalProperties": False}}}
    messages = [
        {"role": "system", "content": "Use the explicit unit-test evidence."},
        {"role": "harness", "content": '{"given":"HARNESS_MARKER"}'},
        {"role": "user", "content": "USER_MARKER"},
    ]
    for index in range(2):
        messages.extend([
            {"role": "assistant", "thinking": f"CALL_REASON_MARKER_{index}",
             "tool_call": {"name": "lookup_human_gene_symbol", "arguments": {"gene_symbol": f"UNIT_{index}"}}},
            {"role": "tool", "name": "lookup_human_gene_symbol", "content": {"value": f"TOOL_EVIDENCE_MARKER_{index}"}},
        ])
    messages.append({"role": "assistant", "thinking": "FINAL_REASON_MARKER", "content": _answer("FINAL_VALUE_MARKER")})
    return {"row_id": "unit_token_trace", "messages": messages, "tools": renderable_v3_tools([tool])}


class WorldModelS23TokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Use the pinned local fast tokenizer; no model or network is required."""
        cls.pin = {"path": str(TOKENIZER), "artifacts": tokenizer_artifact_hashes(TOKENIZER)}
        cls.tokenizer = load_tokenizer(cls.pin)

    def test_current_turn_wire_and_masks_match_existing_s1_serializer(self) -> None:
        """Shared message shapes produce byte-identical token IDs and supervision masks."""
        row = _row()
        expected = tokenize_v3_trace_for_sft(row, FixedDateTokenizer(self.tokenizer), 100000)
        actual = tokenize_trace(row, self.tokenizer)
        self.assertEqual(actual["input_ids"], expected["input_ids"])
        self.assertEqual(actual["completion_mask"], expected["completion_mask"])

    def test_current_reasoning_is_retained_and_supplied_evidence_is_masked(self) -> None:
        """Every current analysis segment is supervised; harness/user/tool material is not."""
        encoded = tokenize_trace(_row(), self.tokenizer)
        selected = self.tokenizer.decode([token for token, keep in zip(encoded["input_ids"], encoded["completion_mask"]) if keep])
        masked = self.tokenizer.decode([token for token, keep in zip(encoded["input_ids"], encoded["completion_mask"]) if not keep])
        for marker in ("CALL_REASON_MARKER_0", "CALL_REASON_MARKER_1", "FINAL_REASON_MARKER", "FINAL_VALUE_MARKER"):
            self.assertIn(marker, selected)
            self.assertNotIn(marker, masked)
        for marker in ("HARNESS_MARKER", "USER_MARKER", "TOOL_EVIDENCE_MARKER_0", "TOOL_EVIDENCE_MARKER_1"):
            self.assertIn(marker, masked)
            self.assertNotIn(marker, selected)

    def test_earlier_turn_reasoning_is_removed_and_earlier_answer_is_masked(self) -> None:
        """Only current-turn assistant actions are targets in a multi-turn wire."""
        row = _row()
        row["messages"][2:2] = [
            {"role": "user", "content": "Earlier request."},
            {"role": "assistant", "thinking": "EARLIER_REASON_SECRET", "content": _answer("EARLIER_VALUE")},
        ]
        encoded = tokenize_trace(row, self.tokenizer)
        full = self.tokenizer.decode(encoded["input_ids"])
        selected = self.tokenizer.decode([token for token, keep in zip(encoded["input_ids"], encoded["completion_mask"]) if keep])
        self.assertNotIn("EARLIER_REASON_SECRET", full)
        self.assertIn("EARLIER_VALUE", full)
        self.assertNotIn("EARLIER_VALUE", selected)
        self.assertEqual(encoded, tokenize_v3_trace_for_sft(row, FixedDateTokenizer(self.tokenizer), 100000))

    def test_new_s2_tool_is_counted_without_changing_the_production_whitelist(self) -> None:
        """The isolated serializer accepts pinned S2 tools while S0/S1 remains frozen."""
        row = _row()
        row["tools"][0]["function"]["name"] = "get_neighbors"
        for message in row["messages"]:
            if "tool_call" in message:
                message["tool_call"]["name"] = "get_neighbors"
            elif message["role"] == "tool":
                message["name"] = "get_neighbors"
        self.assertGreater(len(tokenize_trace(row, self.tokenizer)["input_ids"]), 0)
        with self.assertRaises(ValueError):
            renderable_v3_tools(row["tools"])

    def test_oversized_rows_are_retained_measured_and_blocked(self) -> None:
        """A verification audit reports the full answer instead of clipping at its limit."""
        row = _row()
        row["messages"][-1]["content"]["result"] = {"neighbors": [
            {"gene_id": f"ENSG{index:011d}", "weight": index / 10000}
            for index in range(600)
        ]}
        original = deepcopy(row)
        measured = audit_rows([row], self.tokenizer, max_length=4096)
        self.assertEqual(row, original)
        self.assertGreater(measured["max_tokens"], 4096)
        self.assertGreater(measured["max_final_json_tokens"], 4096)
        self.assertEqual(measured["over_budget_row_ids"], [row["row_id"]])
        self.assertFalse(measured["context_fits"])
        self.assertFalse(measured["truncation_applied"])
        self.assertFalse(measured["production_export_allowed"])
        counts = measured["row_counts"][0]
        self.assertEqual(counts["tokens"], counts["assistant_tokens"] + counts["masked_tokens"])

    def test_exact_boundary_and_unique_row_ids_are_checked(self) -> None:
        """The context limit is inclusive, and duplicate records cannot hide audit omissions."""
        row = _row()
        n = len(tokenize_trace(row, self.tokenizer)["input_ids"])
        self.assertTrue(audit_rows([row], self.tokenizer, n)["context_fits"])
        self.assertFalse(audit_rows([row], self.tokenizer, n - 1)["context_fits"])
        with self.assertRaises(ValueError):
            audit_rows([row, row], self.tokenizer)
        with self.assertRaises(ValueError):
            audit_rows([row], self.tokenizer, True)

    def test_artifact_pin_mismatch_is_rejected_before_loading(self) -> None:
        """Every local tokenizer/template byte is bound to the verification contract."""
        changed = deepcopy(self.pin)
        changed["artifacts"]["chat_template.jinja"] = "0" * 64
        with self.assertRaises(ValueError):
            load_tokenizer(changed)
        changed = deepcopy(self.pin)
        del changed["artifacts"]["tokenizer_config.json"]
        with self.assertRaises(ValueError):
            load_tokenizer(changed)


if __name__ == "__main__":
    unittest.main()
