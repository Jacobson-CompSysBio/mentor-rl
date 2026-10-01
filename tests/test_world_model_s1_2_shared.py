"""Test S1.2 Ensembl-to-symbol data construction."""

from __future__ import annotations

import unittest

from runtime.world_model_s1_shared import (
    S1_2_EXPLANATION_TEMPLATES,
    build_s1_2_trace,
)
from scripts.build_world_model_v2_s0_tool_sft import build_components
from scripts.build_world_model_v2_s1_2_sft import (
    SelectedFact,
    answer_key_row,
    assign_registry_facts,
    public_question_row,
    select_form_facts,
)


REGISTRY_ID = "sha256:" + "0" * 64
GENE_ID = "ENSG00000141510"
GENE_SYMBOL = "TP53"


def _provenance(
    form: str = "direct",
    task_context: str = "direct",
) -> dict[str, object]:
    """Return one valid S1.2 provenance object."""

    return {
        "fact_id": "fact_tp53",
        "fact_group_id": "group_tp53",
        "fact_role": "train",
        "prompt_form_id": "train",
        "question_form": form,
        "task_context": task_context,
        "rendering_index": 0,
        "source_kind": "ensembl_registry",
        "s0_split_contract": "identifier_component_v7",
    }


def _resolved_trace(form: str):
    """Return one named S1.2 trace."""

    task_context = "inside_task" if form == "inside_task" else "direct"
    input_gene_id = GENE_ID
    question = f"Resolve {GENE_ID}."
    tool_result = {
        "status": "resolved",
        "gene_id": GENE_ID,
        "gene_symbol": GENE_SYMBOL,
    }
    if form == "already_resolved":
        question = (
            "Pinned Ensembl release 116 registry block: "
            f'{{"status":"resolved","gene_id":"{GENE_ID}",'
            f'"gene_symbol":"{GENE_SYMBOL}"}}. Resolve {GENE_ID}.'
        )
        tool_result = None
    elif form == "version_suffix":
        input_gene_id = f"{GENE_ID}.15"
        question = f"Resolve {input_gene_id}."
    return build_s1_2_trace(
        record_id=f"trace_{form}",
        question=question,
        form=form,
        input_gene_id=input_gene_id,
        canonical_gene_id=GENE_ID,
        gene_symbol=GENE_SYMBOL,
        tool_result=tool_result,
        task_context=task_context,
        invalid_kind=None,
        split="train",
        provenance=_provenance(form, task_context),
        identifier_registry_id=REGISTRY_ID,
    )


class S1SharedTraceTests(unittest.TestCase):
    """Test S1.2 trace branches and decisions."""

    def test_direct_row_calls_once_and_stops(self) -> None:
        """Resolve a direct ID question with one call and stop."""

        trace = _resolved_trace("direct")

        self.assertEqual(trace.tool_call_count, 1)
        self.assertEqual(trace.final_answer.status, "resolved")
        self.assertEqual(trace.final_answer.continuation_decision, "stop")
        self.assertEqual(
            trace.final_answer.result,
            {"gene_id": GENE_ID, "gene_symbol": GENE_SYMBOL},
        )

    def test_inside_task_row_calls_once_and_continues(self) -> None:
        """Resolve an ID inside a task and continue."""

        trace = _resolved_trace("inside_task")

        self.assertEqual(trace.tool_call_count, 1)
        self.assertEqual(
            trace.final_answer.continuation_decision,
            "continue",
        )

    def test_registry_block_row_lists_tools_but_does_not_call(self) -> None:
        """Use a visible registry block without a repeated call."""

        trace = _resolved_trace("already_resolved")

        self.assertEqual(trace.tool_call_count, 0)
        self.assertEqual(len(trace.tools), 2)
        self.assertEqual(
            [message["role"] for message in trace.messages],
            ["system", "user", "assistant"],
        )

    def test_version_suffix_row_calls_with_the_canonical_id(self) -> None:
        """Remove the version suffix before the tool call."""

        trace = _resolved_trace("version_suffix")
        function = trace.messages[2]["tool_call"]["function"]

        self.assertEqual(function["name"], "lookup_human_ensembl_gene_id")
        self.assertEqual(function["arguments"], {"gene_id": GENE_ID})
        self.assertEqual(trace.final_answer.result["gene_id"], GENE_ID)

    def test_no_symbol_direct_row_resolves_and_stops(self) -> None:
        """Keep a valid ID when the direct lookup returns null."""

        trace = self._no_symbol_trace("direct")

        self.assertEqual(trace.final_answer.status, "resolved")
        self.assertEqual(trace.final_answer.result["gene_symbol"], None)
        self.assertEqual(trace.final_answer.continuation_decision, "stop")

    def test_no_symbol_task_row_resolves_and_continues(self) -> None:
        """Continue a task with a valid unnamed gene ID."""

        trace = self._no_symbol_trace("inside_task")

        self.assertEqual(trace.final_answer.status, "resolved")
        self.assertEqual(trace.final_answer.result["gene_symbol"], None)
        self.assertEqual(
            trace.final_answer.continuation_decision,
            "continue",
        )

    def _no_symbol_trace(self, task_context: str):
        """Return one no-symbol S1.2 trace."""

        return build_s1_2_trace(
            record_id=f"trace_no_symbol_{task_context}",
            question=f"Resolve {GENE_ID} before the next step.",
            form="no_symbol",
            input_gene_id=GENE_ID,
            canonical_gene_id=GENE_ID,
            gene_symbol=None,
            tool_result={
                "status": "resolved",
                "gene_id": GENE_ID,
                "gene_symbol": None,
            },
            task_context=task_context,
            invalid_kind=None,
            split="train",
            provenance=_provenance("no_symbol", task_context),
            identifier_registry_id=REGISTRY_ID,
        )

    def test_invalid_inputs_defer_without_a_tool_call(self) -> None:
        """Reject transcript, mouse, and malformed IDs before a call."""

        inputs = {
            "transcript": "ENST00000141510",
            "mouse_gene": "ENSMUSG00000141510",
            "malformed": "ENSG0000014151",
        }
        for invalid_kind, input_gene_id in inputs.items():
            with self.subTest(invalid_kind=invalid_kind):
                trace = build_s1_2_trace(
                    record_id=f"trace_invalid_{invalid_kind}",
                    question=f"Resolve {input_gene_id}.",
                    form="invalid_input",
                    input_gene_id=input_gene_id,
                    canonical_gene_id=None,
                    gene_symbol=None,
                    tool_result=None,
                    task_context="direct",
                    invalid_kind=invalid_kind,
                    split="train",
                    provenance=_provenance("invalid_input"),
                    identifier_registry_id=REGISTRY_ID,
                )
                self.assertEqual(trace.tool_call_count, 0)
                self.assertEqual(trace.final_answer.status, "invalid_input")
                self.assertEqual(
                    trace.final_answer.continuation_decision,
                    "defer",
                )
                self.assertEqual(
                    trace.final_answer.result,
                    {
                        "input": input_gene_id,
                        "expected": "human_gene_id",
                    },
                )

    def test_each_action_path_has_three_review_templates(self) -> None:
        """Keep three reviewed templates for each S1.2 action path."""

        self.assertEqual(len(S1_2_EXPLANATION_TEMPLATES), 12)
        self.assertTrue(
            all(
                len(templates) == 3
                for templates in S1_2_EXPLANATION_TEMPLATES.values()
            )
        )


class S1CorpusHelperTests(unittest.TestCase):
    """Test S1.2 split and evaluator helpers."""

    def test_registry_facts_keep_the_s0_component_split(self) -> None:
        """Keep each Ensembl ID in its S0 component split."""

        components = build_components(
            {
                "ENSG00000000001": ["ONE"],
                "ENSG00000000002": ["TWO"],
                "ENSG00000000003": [],
            }
        )
        by_id = {
            component["gene_ids"][0]: component for component in components
        }
        assigned = assign_registry_facts(
            components,
            {
                "validation_fact_group_ids": [
                    by_id["ENSG00000000002"]["fact_group_id"]
                ],
                "test_fact_group_ids": [
                    by_id["ENSG00000000003"]["fact_group_id"]
                ],
            },
        )

        self.assertEqual(
            [_fact["input"]["gene_id"] for _fact in assigned["train"]],
            ["ENSG00000000001"],
        )
        self.assertEqual(
            [
                _fact["input"]["gene_id"]
                for _fact in assigned["validation"]
            ],
            ["ENSG00000000002"],
        )
        self.assertEqual(
            [_fact["input"]["gene_id"] for _fact in assigned["test"]],
            ["ENSG00000000003"],
        )

    def test_form_selection_uses_named_and_unnamed_pools(self) -> None:
        """Select each form from the required source pool."""

        components = build_components(
            {
                "ENSG00000000001": ["ONE"],
                "ENSG00000000002": ["TWO"],
                "ENSG00000000003": ["THREE"],
                "ENSG00000000004": [],
                "ENSG00000000005": [],
            }
        )
        facts = [
            fact
            for component in components
            for fact in component["facts"]
            if fact["family"] == "human_ensembl_to_symbol"
        ]
        selected = select_form_facts(
            facts,
            counts={
                "direct": 1,
                "inside_task": 1,
                "already_resolved": 1,
                "version_suffix": 1,
                "no_symbol": 2,
                "invalid_input": 3,
            },
            seed=17,
            split="train",
        )

        no_symbol_rows = [
            item for item in selected if item.form == "no_symbol"
        ]
        invalid_rows = [
            item for item in selected if item.form == "invalid_input"
        ]
        self.assertEqual(
            {item.task_context for item in no_symbol_rows},
            {"direct", "inside_task"},
        )
        self.assertTrue(
            all(not item.fact["answer"]["gene_symbols"] for item in no_symbol_rows)
        )
        self.assertEqual(
            {item.invalid_kind for item in invalid_rows},
            {"transcript", "mouse_gene", "malformed"},
        )

    def test_public_question_is_answer_free_and_key_is_plural(self) -> None:
        """Keep targets private and use the multi-action key shape."""

        trace = _resolved_trace("direct")
        fact = {
            "fact_id": "fact_tp53",
            "fact_group_id": "group_tp53",
            "family": "human_ensembl_to_symbol",
            "input": {"gene_id": GENE_ID},
            "answer": {
                "status": "resolved",
                "gene_id": GENE_ID,
                "gene_symbols": [GENE_SYMBOL],
            },
        }
        selection = SelectedFact(
            form="direct",
            fact=fact,
            input_gene_id=GENE_ID,
            task_context="direct",
        )

        question = public_question_row(trace, selection, split="test")
        answer = answer_key_row(trace, selection)

        self.assertNotIn("messages", question)
        self.assertNotIn("requires_user_input", question["metadata"])
        self.assertNotIn("expected_final_answer", question)
        self.assertEqual(answer["expected_tool_call_count"], 1)
        self.assertEqual(len(answer["expected_tool_actions"]), 1)


if __name__ == "__main__":
    unittest.main()
