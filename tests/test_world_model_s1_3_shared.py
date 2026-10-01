"""Test S1.3 ambiguous-symbol data construction."""

from __future__ import annotations

import unittest

from runtime.world_model_s1_shared import (
    S1_3_EXPLANATION_TEMPLATES,
    build_s1_3_trace,
)
from scripts.build_world_model_v2_s0_tool_sft import build_components
from scripts.build_world_model_v2_s1_3_sft import (
    SelectedFact,
    answer_key_row,
    assign_ambiguous_facts,
    public_question_row,
    select_form_facts,
)


REGISTRY_ID = "sha256:" + "0" * 64
GENE_SYMBOL = "SHARED"
CANDIDATES = ["ENSG00000000001", "ENSG00000000002"]


def _provenance(form: str = "direct") -> dict[str, object]:
    """Return one valid S1.3 provenance object."""

    return {
        "fact_id": "fact_shared",
        "fact_group_id": "group_shared",
        "fact_role": "train",
        "prompt_form_id": "train",
        "question_form": form,
        "rendering_index": 0,
        "sample_index": 0,
        "source_kind": "ensembl_registry",
        "s0_split_contract": "identifier_component_v7",
        "multiplex_candidate": True,
        "candidate_gene_id_count": 2,
    }


def _ambiguous_trace(form: str):
    """Return one S1.3 ambiguous trace."""

    question = f"Resolve {GENE_SYMBOL}."
    tool_result = {
        "status": "ambiguous",
        "gene_symbol": GENE_SYMBOL,
        "candidate_gene_ids": CANDIDATES,
    }
    if form == "already_marked":
        question = (
            "Pinned Ensembl release 116 registry block: "
            f'{{"status":"ambiguous","gene_symbol":"{GENE_SYMBOL}",'
            '"candidate_gene_ids":'
            f'["{CANDIDATES[0]}","{CANDIDATES[1]}"]}}. '
            f"Resolve {GENE_SYMBOL}."
        )
        tool_result = None
    return build_s1_3_trace(
        record_id=f"trace_{form}",
        question=question,
        form=form,
        gene_symbol=GENE_SYMBOL,
        candidate_gene_ids=CANDIDATES,
        tool_result=tool_result,
        split="train",
        provenance=_provenance(form),
        identifier_registry_id=REGISTRY_ID,
    )


def _fact(
    symbol: str,
    candidates: list[str],
    index: int,
) -> dict[str, object]:
    """Return one ambiguous source fact."""

    return {
        "fact_id": f"fact_{index}",
        "fact_group_id": f"group_{index}",
        "family": "human_ambiguous_symbol",
        "input": {"gene_symbol": symbol},
        "answer": {
            "status": "ambiguous",
            "gene_symbol": symbol,
            "candidate_gene_ids": candidates,
            "action": "defer",
        },
    }


class S1SharedTraceTests(unittest.TestCase):
    """Test S1.3 trace branches and decisions."""

    def test_direct_row_calls_once_and_defers(self) -> None:
        """Return every candidate for a direct ambiguous lookup."""

        trace = _ambiguous_trace("direct")

        self.assertEqual(trace.tool_call_count, 1)
        self.assertEqual(trace.final_answer.status, "ambiguous")
        self.assertEqual(trace.final_answer.continuation_decision, "defer")
        self.assertEqual(
            trace.final_answer.result,
            {
                "gene_symbol": GENE_SYMBOL,
                "candidate_gene_ids": CANDIDATES,
            },
        )
        self.assertNotIn("action", trace.messages[3]["content"])

    def test_inside_task_row_calls_once_and_defers(self) -> None:
        """Defer the larger task after an ambiguous lookup."""

        trace = _ambiguous_trace("inside_task")

        self.assertEqual(trace.tool_call_count, 1)
        self.assertEqual(trace.final_answer.continuation_decision, "defer")
        self.assertTrue(trace.metadata["requires_user_input"])

    def test_registry_block_row_lists_tools_but_does_not_call(self) -> None:
        """Use a visible ambiguous block without a repeated call."""

        trace = _ambiguous_trace("already_marked")

        self.assertEqual(trace.tool_call_count, 0)
        self.assertEqual(len(trace.tools), 2)
        self.assertEqual(
            [message["role"] for message in trace.messages],
            ["system", "user", "assistant"],
        )

    def test_summary_anchors_every_candidate(self) -> None:
        """Anchor all candidates without one preferred ID."""

        summary = _ambiguous_trace("direct").final_answer.summary

        for gene_id in CANDIDATES:
            self.assertIn(f"<GENE:{gene_id}|{GENE_SYMBOL}>", summary)
        self.assertIn("Choose one ID", summary)

    def test_trace_rejects_an_action_in_the_tool_payload(self) -> None:
        """Keep the continuation action out of the tool result."""

        with self.assertRaisesRegex(ValueError, "tool result"):
            build_s1_3_trace(
                record_id="trace_action",
                question=f"Resolve {GENE_SYMBOL}.",
                form="direct",
                gene_symbol=GENE_SYMBOL,
                candidate_gene_ids=CANDIDATES,
                tool_result={
                    "status": "ambiguous",
                    "gene_symbol": GENE_SYMBOL,
                    "candidate_gene_ids": CANDIDATES,
                    "action": "defer",
                },
                split="train",
                provenance=_provenance(),
                identifier_registry_id=REGISTRY_ID,
            )

    def test_trace_rejects_more_than_49_candidates(self) -> None:
        """Reject a candidate list above the declared cap."""

        candidates = [f"ENSG{index:011d}" for index in range(1, 51)]
        with self.assertRaisesRegex(ValueError, "between 2 and 49"):
            build_s1_3_trace(
                record_id="trace_long",
                question=f"Resolve {GENE_SYMBOL}.",
                form="direct",
                gene_symbol=GENE_SYMBOL,
                candidate_gene_ids=candidates,
                tool_result={
                    "status": "ambiguous",
                    "gene_symbol": GENE_SYMBOL,
                    "candidate_gene_ids": candidates,
                },
                split="train",
                provenance=_provenance(),
                identifier_registry_id=REGISTRY_ID,
            )

    def test_each_action_path_has_three_review_templates(self) -> None:
        """Keep three reviewed templates for each S1.3 action path."""

        self.assertEqual(len(S1_3_EXPLANATION_TEMPLATES), 5)
        self.assertTrue(
            all(
                len(templates) == 3
                for templates in S1_3_EXPLANATION_TEMPLATES.values()
            )
        )


class S1CorpusHelperTests(unittest.TestCase):
    """Test S1.3 split, sample, and evaluator helpers."""

    def test_ambiguous_facts_keep_the_s0_component_split_and_cap(self) -> None:
        """Keep S0 splits and exclude the fact above the cap."""

        mappings = {
            "ENSG00000000001": ["ONE"],
            "ENSG00000000002": ["ONE"],
            "ENSG00000000003": ["TWO"],
            "ENSG00000000004": ["TWO"],
            "ENSG00000000005": ["THREE"],
            "ENSG00000000006": ["THREE"],
            "ENSG00000000007": ["LONG"],
            "ENSG00000000008": ["LONG"],
            "ENSG00000000009": ["LONG"],
        }
        components = build_components(mappings)
        by_symbol = {
            component["gene_symbols"][0]: component
            for component in components
        }
        long_fact = next(
            fact
            for fact in by_symbol["LONG"]["facts"]
            if fact["family"] == "human_ambiguous_symbol"
        )
        assigned, excluded = assign_ambiguous_facts(
            components,
            {
                "validation_fact_group_ids": [
                    by_symbol["TWO"]["fact_group_id"]
                ],
                "test_fact_group_ids": [
                    by_symbol["THREE"]["fact_group_id"]
                ],
            },
            maximum_candidates=2,
            expected_excluded_fact_ids=frozenset(
                {str(long_fact["fact_id"])}
            ),
        )

        self.assertEqual(
            [_fact["input"]["gene_symbol"] for _fact in assigned["train"]],
            ["ONE"],
        )
        self.assertEqual(
            [
                _fact["input"]["gene_symbol"]
                for _fact in assigned["validation"]
            ],
            ["TWO"],
        )
        self.assertEqual(
            [_fact["input"]["gene_symbol"] for _fact in assigned["test"]],
            ["THREE"],
        )
        self.assertEqual([fact["fact_id"] for fact in excluded], [long_fact["fact_id"]])

    def test_train_selection_prefers_multiplex_candidates(self) -> None:
        """Use the declared multiplex fraction in each train form."""

        facts = [
            _fact(
                f"S{index}",
                [
                    f"ENSG{index * 2 + 1:011d}",
                    f"ENSG{index * 2 + 2:011d}",
                ],
                index,
            )
            for index in range(6)
        ]
        multiplex = frozenset(
            {
                facts[0]["answer"]["candidate_gene_ids"][0],
                facts[1]["answer"]["candidate_gene_ids"][0],
                facts[2]["answer"]["candidate_gene_ids"][0],
            }
        )
        selected = select_form_facts(
            facts,
            multiplex,
            counts={"direct": 10, "inside_task": 10, "already_marked": 5},
            seed=17,
            split="train",
            target_multiplex_fraction=0.8,
        )

        self.assertEqual(len(selected), 25)
        self.assertEqual(
            sum(item.multiplex_candidate for item in selected),
            20,
        )

    def test_public_question_is_answer_free_and_key_is_plural(self) -> None:
        """Keep targets private and use the multi-action key shape."""

        trace = _ambiguous_trace("direct")
        fact = _fact(GENE_SYMBOL, CANDIDATES, 1)
        selection = SelectedFact(
            form="direct",
            fact=fact,
            sample_index=0,
            multiplex_candidate=True,
        )

        question = public_question_row(trace, selection, split="test")
        answer = answer_key_row(trace, selection)

        self.assertNotIn("messages", question)
        self.assertNotIn("requires_user_input", question["metadata"])
        self.assertNotIn("expected_final_answer", question)
        self.assertNotIn("candidate_gene_id_count", question["provenance"])
        self.assertEqual(answer["expected_tool_call_count"], 1)
        self.assertEqual(len(answer["expected_tool_actions"]), 1)


if __name__ == "__main__":
    unittest.main()
