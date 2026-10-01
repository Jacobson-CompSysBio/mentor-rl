"""Test S0 rows that use the shared supervised trace contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from runtime.world_model_contracts import (
    SUPERVISED_EXPLANATION_LOSS_CONTRACT,
    WORLD_MODEL_ANSWER_SCHEMA_VERSION,
    SupervisedTraceRecord,
)
from runtime.world_model_prompts import s0_shared_trace_prompt_contract
from runtime.world_model_s0_shared import (
    S0_EXPLANATION_TEMPLATES,
    S0_SHARED_ACTION_PATHS,
    build_s0_shared_trace,
)
from runtime.world_model_s0_tools import (
    ENSEMBL_TOOL_NAME,
    REGISTRY_SCHEMA_VERSION_V2,
    SYMBOL_TOOL_NAME,
    TOOL_CONTRACT_VERSION_V2,
    S0IdentifierToolRuntime,
)
from scripts.build_world_model_v2_s0_tool_sft import make_shared_record


CASES = (
    {
        "name": "symbol_resolved",
        "family": "human_symbol_to_ensembl",
        "tool_name": SYMBOL_TOOL_NAME,
        "arguments": {"gene_symbol": "TP53"},
        "tool_result": {
            "status": "resolved",
            "gene_id": "ENSG00000141510",
            "gene_symbol": "TP53",
        },
        "status": "resolved",
        "decision": "continue",
        "result": {
            "gene_id": "ENSG00000141510",
            "gene_symbol": "TP53",
        },
    },
    {
        "name": "symbol_ambiguous",
        "family": "human_ambiguous_symbol",
        "tool_name": SYMBOL_TOOL_NAME,
        "arguments": {"gene_symbol": "GENE2"},
        "tool_result": {
            "status": "ambiguous",
            "gene_symbol": "GENE2",
            "candidate_gene_ids": [
                "ENSG00000000002",
                "ENSG00000000102",
            ],
        },
        "status": "ambiguous",
        "decision": "defer",
        "result": {
            "gene_symbol": "GENE2",
            "candidate_gene_ids": [
                "ENSG00000000002",
                "ENSG00000000102",
            ],
        },
    },
    {
        "name": "symbol_not_found",
        "family": "human_symbol_to_ensembl",
        "tool_name": SYMBOL_TOOL_NAME,
        "arguments": {"gene_symbol": "NOT_A_GENE"},
        "tool_result": {
            "status": "not_found",
            "gene_symbol": "NOT_A_GENE",
            "candidate_gene_ids": [],
        },
        "status": "not_found",
        "decision": "defer",
        "result": {
            "gene_symbol": "NOT_A_GENE",
            "candidate_gene_ids": [],
        },
    },
    {
        "name": "gene_id_named",
        "family": "human_ensembl_to_symbol",
        "tool_name": ENSEMBL_TOOL_NAME,
        "arguments": {"gene_id": "ENSG00000141510"},
        "tool_result": {
            "status": "resolved",
            "gene_id": "ENSG00000141510",
            "gene_symbol": "TP53",
        },
        "status": "resolved",
        "decision": "continue",
        "result": {
            "gene_id": "ENSG00000141510",
            "gene_symbol": "TP53",
        },
    },
    {
        "name": "gene_id_unnamed",
        "family": "human_ensembl_to_symbol",
        "tool_name": ENSEMBL_TOOL_NAME,
        "arguments": {"gene_id": "ENSG00000000002"},
        "tool_result": {
            "status": "resolved",
            "gene_id": "ENSG00000000002",
            "gene_symbol": None,
        },
        "status": "resolved",
        "decision": "continue",
        "result": {
            "gene_id": "ENSG00000000002",
            "gene_symbol": None,
        },
    },
    {
        "name": "gene_id_not_found",
        "family": "human_ensembl_to_symbol",
        "tool_name": ENSEMBL_TOOL_NAME,
        "arguments": {"gene_id": "ENSG99999999999"},
        "tool_result": {
            "status": "not_found",
            "gene_id": "ENSG99999999999",
            "gene_symbol": None,
        },
        "status": "not_found",
        "decision": "defer",
        "result": {
            "gene_id": "ENSG99999999999",
            "gene_symbol": None,
        },
    },
)


def _trace(case: dict[str, object]) -> SupervisedTraceRecord:
    """Build one shared S0 trace test case."""

    return build_s0_shared_trace(
        record_id=f"trace_{case['name']}",
        question=f"Resolve the test input for {case['name']}.",
        question_family=str(case["family"]),
        tool_name=str(case["tool_name"]),
        arguments=case["arguments"],
        tool_result=case["tool_result"],
        split="train",
        provenance={"fact_id": f"fact_{case['name']}"},
        identifier_registry_id="sha256:" + "0" * 64,
    )


class S0SharedTraceTests(unittest.TestCase):
    """Test each S0 result branch and explanation contract."""

    def test_all_action_paths_have_three_reviewed_templates(self) -> None:
        """Keep three bounded templates for each S0 action path."""

        self.assertEqual(
            set(S0_EXPLANATION_TEMPLATES),
            set(S0_SHARED_ACTION_PATHS),
        )
        template_ids = set()
        for action_path, templates in S0_EXPLANATION_TEMPLATES.items():
            with self.subTest(action_path=action_path):
                self.assertEqual(len(templates), 3)
                for template in templates:
                    text = template.text.format(
                        gene_symbol="TP53",
                        gene_id="ENSG00000141510",
                    )
                    sentences = [
                        item
                        for item in re.split(r"(?<=[.!?])\s+", text)
                        if item
                    ]
                    words = [
                        len(re.findall(r"\b[\w'-]+\b", item))
                        for item in sentences
                    ]
                    self.assertIn(len(sentences), {1, 2})
                    self.assertLessEqual(sum(words), 40)
                    self.assertTrue(all(count <= 25 for count in words))
                    self.assertTrue(template.rule_ids)
                    self.assertNotIn(template.template_id, template_ids)
                    template_ids.add(template.template_id)

    def test_each_tool_result_builds_one_valid_trace(self) -> None:
        """Build the six S0 answer branches."""

        for case in CASES:
            with self.subTest(case=case["name"]):
                trace = _trace(case)
                decoded = SupervisedTraceRecord.from_dict(trace.to_dict())
                answer = decoded.final_answer

                self.assertEqual(decoded.tool_call_count, 1)
                self.assertEqual(answer.status, case["status"])
                self.assertEqual(
                    answer.continuation_decision,
                    case["decision"],
                )
                self.assertEqual(dict(answer.result), case["result"])
                self.assertEqual(
                    decoded.metadata["requires_user_input"],
                    case["decision"] == "defer",
                )
                self.assertEqual(
                    decoded.metadata["answer_schema_version"],
                    WORLD_MODEL_ANSWER_SCHEMA_VERSION,
                )
                self.assertEqual(
                    decoded.metadata["loss_contract"],
                    SUPERVISED_EXPLANATION_LOSS_CONTRACT,
                )
                self.assertEqual(
                    decoded.metadata["tool_contract_version"],
                    TOOL_CONTRACT_VERSION_V2,
                )
                final_receipt = decoded.messages[-1][
                    "explanation_receipt"
                ]
                self.assertTrue(
                    final_receipt["explanation_template_id"].startswith(
                        f"s0_{case['name']}_v1_"
                    )
                )

    def test_trace_build_is_deterministic(self) -> None:
        """Keep all text, IDs, and receipts stable."""

        first = _trace(CASES[0]).to_dict()
        second = _trace(CASES[0]).to_dict()

        self.assertEqual(first, second)
        self.assertNotEqual(
            first["messages"][2]["explanation_receipt"][
                "explanation_input_sha256"
            ],
            first["messages"][4]["explanation_receipt"][
                "explanation_input_sha256"
            ],
        )

    def test_named_and_unnamed_ids_have_one_result_shape(self) -> None:
        """Use a null symbol for a known unnamed gene ID."""

        named = _trace(CASES[3]).final_answer
        unnamed = _trace(CASES[4]).final_answer
        absent = _trace(CASES[5]).final_answer

        self.assertEqual(set(named.result), {"gene_id", "gene_symbol"})
        self.assertEqual(set(unnamed.result), set(named.result))
        self.assertEqual(set(absent.result), set(named.result))
        self.assertIsNone(unnamed.result["gene_symbol"])
        self.assertEqual(unnamed.status, "resolved")
        self.assertEqual(absent.status, "not_found")

    def test_shared_prompt_differs_from_the_legacy_prompt(self) -> None:
        """Pin the S0 JSON answer prompt as a separate contract."""

        prompt = s0_shared_trace_prompt_contract()

        self.assertIn("continuation_decision", prompt.system_prompt)
        self.assertIn("one JSON object", prompt.system_prompt)
        self.assertEqual(len(prompt.system_prompt_sha256), 64)


class S0SharedBuilderTests(unittest.TestCase):
    """Test the builder adapter with a version 2 registry."""

    def test_builder_keeps_a_known_id_with_no_symbol(self) -> None:
        """Create a resolved trace for one unnamed gene ID."""

        registry = {
            "schema_version": REGISTRY_SCHEMA_VERSION_V2,
            "source": {"release": 116},
            "gene_symbols_by_id": {
                "ENSG00000000001": ["GENE1"],
                "ENSG00000000002": [],
            },
        }
        registry_text = json.dumps(registry, sort_keys=True) + "\n"
        registry_sha256 = hashlib.sha256(
            registry_text.encode("utf-8")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            registry_path.write_text(registry_text, encoding="utf-8")
            runtime = S0IdentifierToolRuntime.from_registry(
                registry_path,
                expected_sha256=registry_sha256,
            )
            record = make_shared_record(
                {
                    "fact_id": "fact_unnamed",
                    "fact_group_id": "group_unnamed",
                    "family": "human_ensembl_to_symbol",
                    "input": {"gene_id": "ENSG00000000002"},
                    "answer": {},
                },
                config={
                    "prompt_forms": {
                        "train": {
                            "human_ensembl_to_symbol": (
                                "Resolve {gene_id}."
                            )
                        }
                    }
                },
                registry_id="sha256:" + registry_sha256,
                prompt_form="train",
                runtime=runtime,
            )

        trace = SupervisedTraceRecord.from_dict(record)
        self.assertEqual(trace.final_answer.status, "resolved")
        self.assertEqual(
            dict(trace.final_answer.result),
            {
                "gene_id": "ENSG00000000002",
                "gene_symbol": None,
            },
        )
        self.assertEqual(trace.provenance["fact_role"], "train")
        self.assertEqual(
            trace.provenance["tool_result_provenance"][
                "registry_schema_version"
            ],
            REGISTRY_SCHEMA_VERSION_V2,
        )


if __name__ == "__main__":
    unittest.main()
