import unittest

from scripts import evaluate_pretrajectory_sft_predictions as eval_sft


def _row(*, idx: int, view_type: str, object_id: str, answer: str, prediction: str) -> dict:
    return {
        "idx": idx,
        "question": "toy",
        "answer": answer,
        "prediction": prediction,
        "metadata": {
            "canonical_object_id": object_id,
            "mixture_bucket": "edge_existence" if "edge" in view_type else "neighbors_layer_membership",
            "record_id": f"record_{idx}",
            "view_type": view_type,
        },
    }


class PretrajectorySftPredictionEvalTests(unittest.TestCase):
    def test_positive_and_negative_edge_exactness(self) -> None:
        canonical = {
            "edge_yes": {
                "payload": {
                    "layer": "HumanNetV3:string_ppi",
                    "source_gene_id": "ENSG00000000003",
                    "target_gene_id": "ENSG00000000005",
                    "has_edge": True,
                }
            },
            "edge_no": {
                "payload": {
                    "layer": "HumanNetV3:string_ppi",
                    "source_gene_id": "ENSG00000000003",
                    "target_gene_id": "ENSG00000000007",
                    "has_edge": False,
                }
            },
        }
        rows = [
            _row(
                idx=0,
                view_type="monoplex_edge_existence",
                object_id="edge_yes",
                answer="Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected to ENSG00000000005.",
                prediction="final Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected to ENSG00000000005.",
            ),
            _row(
                idx=1,
                view_type="monoplex_edge_existence",
                object_id="edge_no",
                answer="No. In `HumanNetV3:string_ppi`, no edge is recorded between ENSG00000000003 and ENSG00000000007.",
                prediction="No. In `HumanNetV3:string_ppi`, no edge is recorded between ENSG00000000003 and ENSG00000000007.",
            ),
        ]

        report = eval_sft.evaluate_prediction_rows(rows, canonical_objects_by_id=canonical)

        self.assertEqual(report["summary"]["exact_graph_fact_pass_rate"], 1.0)
        self.assertEqual(report["failure_counts"], {})

    def test_neighbor_repetition_fails_exactness(self) -> None:
        canonical = {
            "neighbors": {
                "payload": {
                    "layer": "HumanNetV3:coexpression",
                    "gene_id": "ENSG00000000003",
                    "neighbors": ["ENSG00000000005", "ENSG00000000007"],
                }
            }
        }
        rows = [
            _row(
                idx=0,
                view_type="direct_neighbors_by_layer",
                object_id="neighbors",
                answer=(
                    "In `HumanNetV3:coexpression`, ENSG00000000003 has 2 direct neighbors: "
                    "ENSG00000000005, ENSG00000000007."
                ),
                prediction=(
                    "final The direct neighbors of ENSG00000000003 in `HumanNetV3:coexpression` are: "
                    "ENSG00000000003, ENSG00000000003."
                ),
            )
        ]

        report = eval_sft.evaluate_prediction_rows(rows, canonical_objects_by_id=canonical)

        self.assertEqual(report["summary"]["exact_graph_fact_pass_rate"], 0.0)
        self.assertEqual(report["failure_counts"]["missing_ids"], 1)

    def test_module_algebra_and_tool_json_exactness(self) -> None:
        canonical = {
            "module_overlap": {
                "payload": {
                    "left_module_id": "gw_dendrogram_module_1",
                    "right_module_id": "rwr_loe_module_2",
                    "intersection_genes": ["ENSG00000000003"],
                    "intersection_size": 1,
                    "union_size": 3,
                    "overlap_jaccard": 0.333333,
                }
            },
            "tool_choice": {
                "payload": {
                    "tool_name": "get_neighbors",
                    "arguments": {
                        "gene_id": "ENSG00000000003",
                        "layer": "HumanNetV3:coexpression",
                        "graph_version": "toy-graph-v1",
                    },
                    "reason": "direct_neighbors_by_layer",
                }
            },
        }
        rows = [
            _row(
                idx=0,
                view_type="module_overlap_set_algebra",
                object_id="module_overlap",
                answer=(
                    "`gw_dendrogram_module_1` and `rwr_loe_module_2` share 1 genes. "
                    "Union size is 3; Jaccard overlap is 0.333333. "
                    "Intersection genes: ENSG00000000003."
                ),
                prediction=(
                    "`gw_dendrogram_module_1` and `rwr_loe_module_2` share 1 genes. "
                    "Union size is 3; Jaccard overlap is 0.333333. "
                    "Intersection genes: ENSG00000000003."
                ),
            ),
            _row(
                idx=1,
                view_type="tool_call_choice",
                object_id="tool_choice",
                answer=(
                    '{"arguments": {"gene_id": "ENSG00000000003", "graph_version": "toy-graph-v1", '
                    '"layer": "HumanNetV3:coexpression"}, "reason": "direct_neighbors_by_layer", '
                    '"tool_name": "get_neighbors"}'
                ),
                prediction=(
                    '{"arguments": {"gene_id": "ENSG00000000003", "graph_version": "toy-graph-v1", '
                    '"layer": "HumanNetV3:coexpression"}, "reason": "direct_neighbors_by_layer", '
                    '"tool_name": "get_neighbors"}'
                ),
            ),
        ]

        report = eval_sft.evaluate_prediction_rows(rows, canonical_objects_by_id=canonical)

        self.assertEqual(report["summary"]["exact_graph_fact_pass_rate"], 1.0)
        self.assertEqual(report["failure_counts"], {})


if __name__ == "__main__":
    unittest.main()
