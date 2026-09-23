"""Test the deterministic S0 tool-call corpus builder."""

from __future__ import annotations

import unittest

from runtime.world_model_prompts import (
    s0_tool_trajectory_prompt_contract,
)
from runtime.world_model_schemas import (
    IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
    TOOL_TRAJECTORY_FORMAT,
)
from runtime.world_model_s0_tools import (
    ENSEMBL_TOOL_NAME,
    SYMBOL_TOOL_NAME,
)
from scripts.build_world_model_v2_s0_tool_sft import (
    CONFIG_SCHEMA_VERSION,
    make_record,
    public_question_row,
    select_train_facts,
    trajectory_fields,
)


def _fact(
    index: int,
    family: str,
    group: str,
) -> dict[str, object]:
    """Return one small source fact."""

    if family == "human_ensembl_to_symbol":
        inputs = {"gene_id": f"ENSG{index:011d}"}
        answer = {
            "status": "resolved",
            "gene_id": inputs["gene_id"],
            "gene_symbols": [f"GENE{index}"],
        }
    elif family == "human_ambiguous_symbol":
        inputs = {"gene_symbol": f"GENE{index}"}
        answer = {
            "status": "ambiguous",
            "gene_symbol": inputs["gene_symbol"],
            "candidate_gene_ids": [
                f"ENSG{index:011d}",
                f"ENSG{index + 100:011d}",
            ],
            "action": "defer",
        }
    else:
        inputs = {"gene_symbol": f"GENE{index}"}
        answer = {
            "status": "resolved",
            "gene_id": f"ENSG{index:011d}",
            "gene_symbol": inputs["gene_symbol"],
        }
    return {
        "fact_id": f"fact_{index}",
        "fact_group_id": group,
        "family": family,
        "input": inputs,
        "answer": answer,
    }


class ToolCorpusSelectionTests(unittest.TestCase):
    """Test train selection and record output."""

    def test_selection_is_balanced_and_excludes_eval_groups(self) -> None:
        """Select each required train family from unassigned groups."""

        components = [
            {
                "fact_group_id": "eval",
                "facts": [
                    _fact(1, "human_ambiguous_symbol", "eval"),
                    _fact(2, "human_ensembl_to_symbol", "eval"),
                ],
            },
            {
                "fact_group_id": "train",
                "facts": [
                    _fact(3, "human_ambiguous_symbol", "train"),
                    _fact(4, "human_symbol_to_ensembl", "train"),
                    _fact(5, "human_symbol_to_ensembl", "train"),
                    _fact(6, "human_ensembl_to_symbol", "train"),
                    _fact(7, "human_ensembl_to_symbol", "train"),
                ],
            },
        ]
        selected = select_train_facts(
            components,
            {"eval": "validation"},
            {
                "assignment_seed": 10,
                "symbol_rows": 2,
                "gene_id_rows": 2,
            },
        )

        self.assertEqual(len(selected), 4)
        self.assertNotIn("eval", {fact["fact_group_id"] for fact in selected})
        self.assertIn(
            "human_ambiguous_symbol",
            {fact["family"] for fact in selected},
        )
        self.assertEqual(
            sum(
                fact["family"] == "human_ensembl_to_symbol"
                for fact in selected
            ),
            2,
        )

    def test_record_contains_the_complete_trajectory(self) -> None:
        """Keep each supervised trajectory field in one train row."""

        config = {
            "prompt_forms": {
                "train": {
                    "human_symbol_to_ensembl": "Input: {input_json}.",
                }
            }
        }
        record = make_record(
            _fact(8, "human_symbol_to_ensembl", "train"),
            config=config,
            registry_id="sha256:" + "0" * 64,
            prompt_form="train",
        )

        self.assertNotIn("answer", record)
        self.assertNotIn("context", record)
        self.assertEqual(
            record["metadata"]["schema_version"],
            IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertEqual(
            record["assistant_tool_call"]["function"]["name"],
            SYMBOL_TOOL_NAME,
        )
        self.assertEqual(
            record["assistant_tool_call"]["function"]["arguments"],
            record["input"],
        )
        self.assertIn("assistant_thinking", record)
        self.assertIn("tool_result", record)
        self.assertIn("assistant_final", record)

    def test_selection_excludes_one_configured_train_fact(self) -> None:
        """Exclude one fact and fill its symbol allocation."""

        components = [
            {
                "fact_group_id": "train",
                "facts": [
                    _fact(20, "human_ambiguous_symbol", "train"),
                    _fact(21, "human_symbol_to_ensembl", "train"),
                    _fact(22, "human_symbol_to_ensembl", "train"),
                    _fact(23, "human_ensembl_to_symbol", "train"),
                ],
            },
        ]
        selected = select_train_facts(
            components,
            {},
            {
                "assignment_seed": 10,
                "symbol_rows": 2,
                "gene_id_rows": 1,
                "excluded_fact_ids": ["fact_20"],
            },
        )

        self.assertEqual(len(selected), 3)
        self.assertNotIn(
            "fact_20",
            {fact["fact_id"] for fact in selected},
        )
        self.assertEqual(
            sum(
                fact["family"] == "human_symbol_to_ensembl"
                for fact in selected
            ),
            2,
        )

    def test_trajectory_record_contains_the_complete_exchange(
        self,
    ) -> None:
        """Create one complete v6 trajectory record."""

        fact = _fact(
            11,
            "human_symbol_to_ensembl",
            "train",
        )
        fact["answer"] = {
            "status": "resolved",
            "gene_id": "ENSG00000000011",
            "gene_symbol": "GENE11",
        }
        config = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "prompt_forms": {
                "train": {
                    "human_symbol_to_ensembl": [
                        (
                            "Prepare the human Ensembl gene identifier. "
                            "The current gene symbol is {gene_symbol}."
                        ),
                    ],
                }
            },
        }

        record = make_record(
            fact,
            config=config,
            registry_id="sha256:" + "0" * 64,
            prompt_form="train",
        )

        self.assertEqual(
            record["metadata"]["schema_version"],
            IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertEqual(
            record["metadata"]["target_format"],
            TOOL_TRAJECTORY_FORMAT,
        )
        self.assertEqual(
            record["system"],
            s0_tool_trajectory_prompt_contract().system_prompt,
        )
        self.assertEqual(record["tool_result"], fact["answer"])
        self.assertIn("GENE11", record["assistant_thinking"])
        self.assertEqual(
            record["assistant_final"],
            (
                "The Ensembl gene ID for GENE11 is "
                "ENSG00000000011."
            ),
        )

        question = public_question_row(record)
        for field_name in (
            "assistant_tool_call",
            "assistant_thinking",
            "tool_result",
            "assistant_final",
        ):
            self.assertNotIn(field_name, question)

    def test_question_template_selection_is_deterministic(
        self,
    ) -> None:
        """Select one stable human-readable task template."""

        templates = [
            (
                "Prepare the human Ensembl gene identifier. "
                "The current gene symbol is {gene_symbol}."
            ),
            (
                "The next analysis step requires an Ensembl gene "
                "identifier. The current gene symbol is {gene_symbol}."
            ),
        ]
        config = {
            "prompt_assignment_seed": 17,
            "prompt_forms": {
                "train": {
                    "human_symbol_to_ensembl": templates,
                }
            },
        }
        fact = _fact(
            10,
            "human_symbol_to_ensembl",
            "train",
        )

        first = make_record(
            fact,
            config=config,
            registry_id="sha256:" + "0" * 64,
            prompt_form="train",
        )
        second = make_record(
            fact,
            config=config,
            registry_id="sha256:" + "0" * 64,
            prompt_form="train",
        )

        expected_questions = {
            template.format(gene_symbol="GENE10")
            for template in templates
        }

        self.assertEqual(first["question"], second["question"])
        self.assertIn(first["question"], expected_questions)
        self.assertEqual(
            first["provenance"]["rendering_index"],
            second["provenance"]["rendering_index"],
        )
        self.assertIn(
            first["provenance"]["rendering_index"],
            {0, 1},
        )

    def test_trajectory_fields_cover_each_family(self) -> None:
        """Create deterministic fields for each mapping family."""

        resolved_symbol = _fact(
            20,
            "human_symbol_to_ensembl",
            "resolved",
        )
        resolved_symbol["answer"] = {
            "status": "resolved",
            "gene_id": "ENSG00000000020",
            "gene_symbol": "GENE20",
        }

        ambiguous_symbol = _fact(
            21,
            "human_ambiguous_symbol",
            "ambiguous",
        )
        ambiguous_symbol["answer"] = {
            "status": "ambiguous",
            "gene_symbol": "GENE21",
            "candidate_gene_ids": [
                "ENSG00000000021",
                "ENSG00000000121",
            ],
            "action": "defer",
        }

        resolved_gene_id = _fact(
            22,
            "human_ensembl_to_symbol",
            "resolved",
        )
        resolved_gene_id["answer"] = {
            "status": "resolved",
            "gene_id": "ENSG00000000022",
            "gene_symbols": ["ALIAS22", "GENE22"],
        }

        cases = (
            (
                resolved_symbol,
                (
                    "The Ensembl gene ID for GENE20 is "
                    "ENSG00000000020."
                ),
            ),
            (
                ambiguous_symbol,
                (
                    "The gene symbol GENE21 maps to multiple Ensembl "
                    "gene IDs, ENSG00000000021, ENSG00000000121, so "
                    "this analysis must defer the mapping."
                ),
            ),
            (
                resolved_gene_id,
                (
                    "The human gene symbols for Ensembl gene ID "
                    "ENSG00000000022 are ALIAS22, GENE22."
                ),
            ),
        )

        for fact, expected_final in cases:
            with self.subTest(family=fact["family"]):
                fields = trajectory_fields(fact)

                self.assertEqual(
                    fields["tool_result"],
                    fact["answer"],
                )
                self.assertIn(
                    "Ensembl release 116 registry",
                    fields["assistant_thinking"],
                )
                self.assertEqual(
                    fields["assistant_final"],
                    expected_final,
                )

    def test_gene_id_record_selects_the_gene_id_tool(self) -> None:
        """Select the reverse lookup tool for a gene ID."""

        config = {
            "prompt_forms": {
                "validation": {
                    "human_ensembl_to_symbol": "Input: {input_json}.",
                }
            }
        }
        record = make_record(
            _fact(9, "human_ensembl_to_symbol", "validation"),
            config=config,
            registry_id="sha256:" + "0" * 64,
            prompt_form="validation",
        )

        self.assertEqual(record["split"], "val")
        self.assertEqual(record["provenance"]["fact_role"], "unseen")
        self.assertEqual(
            record["assistant_tool_call"]["function"]["name"],
            ENSEMBL_TOOL_NAME,
        )


if __name__ == "__main__":
    unittest.main()
