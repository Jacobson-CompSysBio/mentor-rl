import json
import tempfile
import unittest
from pathlib import Path

from scripts import evaluate_pretrajectory_sft_predictions as eval_sft


def _v3_row(
    *,
    record_id: str,
    object_id: str,
    question_family: str,
    book_mode: str,
    answer: dict,
) -> dict:
    return {
        "question": f"Return the {question_family} result as JSON.",
        "answer": json.dumps(answer, separators=(",", ":"), sort_keys=True),
        "metadata": {
            "answer_format": "json",
            "book_mode": book_mode,
            "canonical_object_id": object_id,
            "mixture_bucket": "structured_context_and_tools"
            if book_mode == "tool_call"
            else "module_set_algebra",
            "oracle_fact_id": object_id,
            "question_family": question_family,
            "record_id": record_id,
            "schema_version": "pretrajectory-sft-v3",
            "validator": {"type": "exact_json"},
        },
    }


class PretrajectorySftV3ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            _v3_row(
                record_id="closed_1",
                object_id="fact_closed",
                question_family="module_provenance",
                book_mode="closed_book",
                answer={
                    "construction_rule": "seed-centered RWR-LOE vector",
                    "module_id": "rwr_loe_module_023786",
                    "module_source": "rwr_loe",
                },
            ),
            _v3_row(
                record_id="open_1",
                object_id="fact_open",
                question_family="gene_absent_from_layer",
                book_mode="open_book",
                answer={
                    "gene_id": "ENSG00000183814",
                    "layer_id": "scPEN:brain:myelencephalon:vascular_smooth_muscle_cell",
                    "node_present": False,
                },
            ),
            _v3_row(
                record_id="tool_1",
                object_id="fact_tool",
                question_family="choose_layer_ablation_tool",
                book_mode="tool_call",
                answer={
                    "tool_action": {
                        "arguments": {
                            "distance_metric": "spearman",
                            "seed_genes": ["ENSG00000188580", "ENSG00000272167"],
                            "top_k": 20,
                        },
                        "tool_name": "get_layer_ablation",
                    },
                    "result_status": "schema_validated_request_not_materialized",
                },
            ),
        ]
        self.canonical = {
            row["metadata"]["canonical_object_id"]: {
                "object_id": row["metadata"]["canonical_object_id"],
                "object_type": row["metadata"]["question_family"],
                "payload": json.loads(row["answer"]),
            }
            for row in self.rows
        }

    def test_gold_copies_score_one_across_closed_open_and_tool_families(self) -> None:
        predicted_rows = [
            {**row, "prediction": row["answer"]}
            for row in self.rows
        ]

        report = eval_sft.evaluate_prediction_rows(
            predicted_rows,
            canonical_objects_by_id=self.canonical,
        )
        gold = report["gold_self_evaluation"]

        self.assertTrue(gold["passed"])
        self.assertEqual(gold["exact_row_count"], 3)
        self.assertEqual(gold["summary"]["exact_graph_fact_pass_rate"], 1.0)
        self.assertEqual(report["summary"]["exact_only"]["count"], 3)
        self.assertEqual(report["summary"]["exact_graph_fact_pass_rate"], 1.0)
        self.assertEqual(report["failure_counts"], {})
        self.assertEqual(
            set(report["by_view_type"]),
            {
                "choose_layer_ablation_tool",
                "gene_absent_from_layer",
                "module_provenance",
            },
        )
        self.assertEqual(report["by_context_mode"]["no_context"]["count"], 1)
        self.assertEqual(report["by_context_mode"]["open_book_context"]["count"], 1)
        self.assertEqual(report["by_context_mode"]["tool_observation"]["count"], 1)
        self.assertTrue(all(item["dataset_contract_valid"] for item in report["examples"]))

    def test_oracle_fact_id_is_a_supported_canonical_reference_fallback(self) -> None:
        row = json.loads(json.dumps(self.rows[0]))
        del row["metadata"]["canonical_object_id"]
        row["prediction"] = row["answer"]

        item = eval_sft.evaluate_prediction_rows(
            [row],
            canonical_objects_by_id=self.canonical,
        )["examples"][0]

        self.assertTrue(item["canonical_reference_valid"])
        self.assertTrue(item["exact_graph_fact_pass"])

    def test_v3_exact_json_rejects_additional_fields(self) -> None:
        row = json.loads(json.dumps(self.rows[0]))
        prediction = json.loads(row["answer"])
        prediction["unsupported_extra_claim"] = "novel"
        row["prediction"] = json.dumps(prediction, sort_keys=True)

        item = eval_sft.evaluate_prediction_rows(
            [row],
            canonical_objects_by_id=self.canonical,
        )["examples"][0]

        self.assertTrue(item["json_subset_match"])
        self.assertFalse(item["json_exact_match"])
        self.assertFalse(item["exact_graph_fact_pass"])

    def test_missing_canonical_object_fails_v3_contract_but_is_diagnostic(self) -> None:
        contract = eval_sft.evaluate_gold_self_contract(
            [self.rows[0]],
            canonical_objects_by_id={},
        )

        self.assertFalse(contract["passed"])
        self.assertFalse(contract["checks"]["dataset_contract_passed"])
        self.assertEqual(contract["failing_record_ids"], ["closed_1"])

    def test_gold_self_only_file_mode_scores_dataset_rows_without_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "val.jsonl"
            canonical_path = root / "canonical_objects.jsonl"
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.rows),
                encoding="utf-8",
            )
            canonical_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.canonical.values()),
                encoding="utf-8",
            )

            report = eval_sft.evaluate_predictions_file(
                rows_path,
                canonical_objects_path=canonical_path,
                gold_self_only=True,
            )

        self.assertTrue(report["gold_self_evaluation"]["passed"])
        self.assertEqual(report["summary"]["exact_graph_fact_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
