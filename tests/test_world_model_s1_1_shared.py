"""Test S1.1 symbol-to-Ensembl data construction."""

from __future__ import annotations

import unittest

from runtime.world_model_s1_shared import (
    S1_1_EXPLANATION_TEMPLATES,
    build_s1_1_trace,
)
from scripts.build_world_model_v2_s0_shared_sft import (
    make_not_found_symbol_facts,
)
from scripts.build_world_model_v2_s0_tool_sft import build_components
from scripts.build_world_model_v2_s1_1_sft import (
    answer_key_row,
    assign_registry_facts,
    public_question_row,
    select_form_facts,
)


REGISTRY_ID = "sha256:" + "0" * 64
GENE_ID = "ENSG00000141510"
GENE_SYMBOL = "TP53"


def _provenance(form: str = "direct") -> dict[str, object]:
    """Return one valid S1.1 provenance object."""

    return {
        "fact_id": "fact_tp53",
        "fact_group_id": "group_tp53",
        "fact_role": "train",
        "prompt_form_id": "train",
        "question_form": form,
        "rendering_index": 0,
        "source_kind": "ensembl_registry",
        "s0_split_contract": "identifier_component_v7",
    }


def _resolved_trace(form: str):
    """Return one resolved S1.1 trace."""

    question = f"Resolve {GENE_SYMBOL}."
    if form == "already_resolved":
        question = (
            "Pinned Ensembl release 116 registry block: "
            f'{{"status":"resolved","gene_id":"{GENE_ID}",'
            f'"gene_symbol":"{GENE_SYMBOL}"}}. Resolve {GENE_SYMBOL}.'
        )
        tool_result = None
    else:
        tool_result = {
            "status": "resolved",
            "gene_id": GENE_ID,
            "gene_symbol": GENE_SYMBOL,
        }
    return build_s1_1_trace(
        record_id=f"trace_{form}",
        question=question,
        form=form,
        gene_symbol=GENE_SYMBOL,
        gene_id=GENE_ID,
        tool_result=tool_result,
        split="train",
        provenance=_provenance(form),
        identifier_registry_id=REGISTRY_ID,
    )


class S1SharedTraceTests(unittest.TestCase):
    """Test S1.1 trace branches and decisions."""

    def test_direct_row_calls_once_and_stops(self) -> None:
        """Resolve a direct question with one call and stop."""

        trace = _resolved_trace("direct")

        self.assertEqual(trace.tool_call_count, 1)
        self.assertEqual(trace.final_answer.status, "resolved")
        self.assertEqual(trace.final_answer.continuation_decision, "stop")
        self.assertEqual(
            trace.final_answer.result,
            {"gene_id": GENE_ID, "gene_symbol": GENE_SYMBOL},
        )

    def test_inside_task_row_calls_once_and_continues(self) -> None:
        """Resolve an inside-task question and continue."""

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

    def test_no_call_row_requires_a_labeled_registry_block(self) -> None:
        """Reject a free-text mapping as authority for a no-call row."""

        with self.assertRaisesRegex(ValueError, "labeled registry block"):
            build_s1_1_trace(
                record_id="trace_unlabeled",
                question=f"{GENE_SYMBOL} is {GENE_ID}.",
                form="already_resolved",
                gene_symbol=GENE_SYMBOL,
                gene_id=GENE_ID,
                tool_result=None,
                split="train",
                provenance=_provenance("already_resolved"),
                identifier_registry_id=REGISTRY_ID,
            )

    def test_absent_symbol_defers_without_an_id_claim(self) -> None:
        """Defer after the registry returns no symbol match."""

        trace = build_s1_1_trace(
            record_id="trace_absent",
            question="Resolve OLDTP53.",
            form="absent_symbol",
            gene_symbol="OLDTP53",
            gene_id=None,
            tool_result={
                "status": "not_found",
                "gene_symbol": "OLDTP53",
                "candidate_gene_ids": [],
            },
            split="train",
            provenance=_provenance("absent_symbol"),
            identifier_registry_id=REGISTRY_ID,
        )

        self.assertEqual(trace.tool_call_count, 1)
        self.assertEqual(trace.final_answer.status, "not_found")
        self.assertEqual(trace.final_answer.result, {"gene_symbol": "OLDTP53"})
        self.assertEqual(trace.final_answer.continuation_decision, "defer")
        self.assertTrue(trace.metadata["requires_user_input"])

    def test_s1_1_rejects_an_ambiguous_result(self) -> None:
        """Keep ambiguous rows for the separate S1.3 block."""

        with self.assertRaisesRegex(ValueError, "resolved S1.1"):
            build_s1_1_trace(
                record_id="trace_ambiguous",
                question="Resolve SHARED.",
                form="direct",
                gene_symbol="SHARED",
                gene_id=None,
                tool_result={
                    "status": "ambiguous",
                    "gene_symbol": "SHARED",
                    "candidate_gene_ids": [
                        "ENSG00000000001",
                        "ENSG00000000002",
                    ],
                },
                split="train",
                provenance=_provenance(),
                identifier_registry_id=REGISTRY_ID,
            )

    def test_each_action_path_has_three_review_templates(self) -> None:
        """Keep three reviewed templates for each S1.1 action path."""

        self.assertEqual(len(S1_1_EXPLANATION_TEMPLATES), 7)
        self.assertTrue(
            all(
                len(templates) == 3
                for templates in S1_1_EXPLANATION_TEMPLATES.values()
            )
        )


class S1CorpusHelperTests(unittest.TestCase):
    """Test S1.1 split and evaluator helpers."""

    def test_registry_facts_keep_the_s0_component_split(self) -> None:
        """Keep each resolved symbol in its S0 component split."""

        components = build_components(
            {
                "ENSG00000000001": ["ONE"],
                "ENSG00000000002": ["TWO"],
                "ENSG00000000003": ["THREE"],
            }
        )
        by_symbol = {
            component["gene_symbols"][0]: component
            for component in components
        }
        assigned = assign_registry_facts(
            components,
            {
                "validation_fact_group_ids": [
                    by_symbol["TWO"]["fact_group_id"]
                ],
                "test_fact_group_ids": [
                    by_symbol["THREE"]["fact_group_id"]
                ],
            },
        )

        self.assertEqual(
            [fact["input"]["gene_symbol"] for fact in assigned["train"]],
            ["ONE"],
        )
        self.assertEqual(
            [
                fact["input"]["gene_symbol"]
                for fact in assigned["validation"]
            ],
            ["TWO"],
        )
        self.assertEqual(
            [fact["input"]["gene_symbol"] for fact in assigned["test"]],
            ["THREE"],
        )

    def test_form_selection_uses_absent_rows_only_for_absent_form(self) -> None:
        """Select the source pool that matches each form."""

        registry = [
            component["facts"][0]
            for component in build_components(
                {
                    "ENSG00000000001": ["ONE"],
                    "ENSG00000000002": ["TWO"],
                    "ENSG00000000003": ["THREE"],
                }
            )
        ]
        absent = make_not_found_symbol_facts(["OLD1"])

        selected = select_form_facts(
            registry,
            absent,
            counts={
                "direct": 1,
                "inside_task": 1,
                "already_resolved": 1,
                "absent_symbol": 1,
            },
            seed=17,
            split="train",
        )

        for form, fact in selected:
            if form == "absent_symbol":
                self.assertEqual(fact["source_kind"], "hgnc_previous_symbol")
            else:
                self.assertEqual(fact["answer"]["status"], "resolved")

    def test_public_question_is_answer_free_and_key_is_plural(self) -> None:
        """Keep targets private and use the multi-action key shape."""

        trace = _resolved_trace("direct")
        fact = {
            "fact_id": "fact_tp53",
            "fact_group_id": "group_tp53",
            "family": "human_symbol_to_ensembl",
            "input": {"gene_symbol": GENE_SYMBOL},
        }

        question = public_question_row(trace, fact, split="test")
        answer = answer_key_row(trace, fact)

        self.assertNotIn("messages", question)
        self.assertNotIn("requires_user_input", question["metadata"])
        self.assertNotIn("expected_final_answer", question)
        self.assertEqual(answer["expected_tool_call_count"], 1)
        self.assertEqual(len(answer["expected_tool_actions"]), 1)


if __name__ == "__main__":
    unittest.main()
