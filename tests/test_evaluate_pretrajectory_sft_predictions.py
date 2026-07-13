import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    def test_terminal_punctuation_is_not_part_of_numbers_or_layers(self) -> None:
        self.assertEqual(
            eval_sft.numbers_from_text("weight 0.495161. score 2e-07."),
            {"0.495161", "2e-07"},
        )
        self.assertEqual(
            eval_sft.layers_from_text("Use HumanNetV3:string_ppi."),
            {"HumanNetV3:string_ppi"},
        )
        self.assertEqual(
            eval_sft.layers_from_text("Use HumanNetV3:string_ppi.release_1."),
            {"HumanNetV3:string_ppi.release_1"},
        )
        self.assertEqual(eval_sft.numbers_from_text("malformed 2.0.7"), set())

    def test_arbitrary_depth_layers_in_bare_backticked_and_json_text(self) -> None:
        deep_layer = "scPEN:brain:amygdaloid-complex:astrocyte"
        text = (
            f"Bare {deep_layer}, backticked `{deep_layer}`, and JSON "
            f'{{"layer": "{deep_layer}"}}.'
        )

        self.assertEqual(eval_sft.layers_from_text(text), {deep_layer})

    def test_signed_scientific_numbers_compare_by_value_and_rendered_precision(self) -> None:
        self.assertEqual(
            eval_sft.numbers_from_text("scores -1.25e-07, +2.0E+3, and .5"),
            {"-1.25e-07", "+2.0E+3", ".5"},
        )
        self.assertTrue(eval_sft.numbers_match("2.03268025316041e-07", "2.0326802531604104e-07"))
        self.assertTrue(eval_sft.numbers_match("2e-07", "2.04e-07"))
        self.assertFalse(eval_sft.numbers_match("2e-07", "2.6e-07"))
        self.assertTrue(eval_sft.numbers_match("0.333333", "0.333333333333"))
        self.assertFalse(eval_sft.numbers_match("3", "4"))

    def test_unsupported_language_is_negation_aware(self) -> None:
        self.assertFalse(eval_sft.has_unsupported_language("This is not a confirmed causal gene."))
        self.assertFalse(
            eval_sft.has_unsupported_language(
                "This is network support, not a claim that the module is causally validated."
            )
        )
        self.assertFalse(
            eval_sft.has_unsupported_language(
                "This does not prove there is no biological relationship in general."
            )
        )
        self.assertFalse(eval_sft.has_unsupported_language("It isn't causally validated."))
        self.assertTrue(eval_sft.has_unsupported_language("This is a confirmed causal gene."))
        self.assertTrue(
            eval_sft.has_unsupported_language(
                "It is not merely associated; it is definitely causal."
            )
        )

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

    def test_rendered_numbers_are_required_for_every_exact_view(self) -> None:
        row = _row(
            idx=0,
            view_type="monoplex_edge_existence",
            object_id="weighted_edge",
            answer=(
                "Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected "
                "to ENSG00000000005 with weight 0.495161."
            ),
            prediction=(
                "Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected "
                "to ENSG00000000005 with weight 0.9."
            ),
        )

        report = eval_sft.evaluate_prediction_rows([row])
        item = report["examples"][0]

        self.assertFalse(item["exact_graph_fact_pass"])
        self.assertEqual(item["missing_numbers"], ["0.495161"])
        self.assertEqual(item["typed_field_failures"], ["weight"])

    def test_unmatched_numbers_are_exact_fatal_and_equivalent_renderings_deduplicate(self) -> None:
        row = _row(
            idx=0,
            view_type="monoplex_edge_existence",
            object_id="weighted_edge",
            answer=(
                "Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected "
                "to ENSG00000000005 with weight 0.5."
            ),
            prediction=(
                "Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected "
                "to ENSG00000000005 with weight 0.500. Equivalently, +5e-1."
            ),
        )

        equivalent = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertEqual(equivalent["extra_numbers"], [])
        self.assertEqual(equivalent["number_precision"], 1.0)
        self.assertTrue(equivalent["exact_graph_fact_pass"])

        row["prediction"] += " Confidence is 0.9."
        extra = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertEqual(extra["extra_numbers"], ["0.9"])
        self.assertEqual(extra["number_precision"], 0.5)
        self.assertFalse(extra["exact_graph_fact_pass"])


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

    def test_hallucinated_extra_layer_fails_exactness(self) -> None:
        expected_layer = "HumanNetV3:coexpression"
        extra_layer = "bulkPEN:brain:basal_ganglia:caudate"
        answer = (
            f"Yes. In `{expected_layer}`, ENSG00000000003 is directly connected to "
            "ENSG00000000005."
        )
        rows = [
            _row(
                idx=0,
                view_type="monoplex_edge_existence",
                object_id="edge",
                answer=answer,
                prediction=f"{answer} The same edge is also in `{extra_layer}`.",
            )
        ]

        report = eval_sft.evaluate_prediction_rows(rows)
        item = report["examples"][0]

        self.assertEqual(item["extra_layers"], [extra_layer])
        self.assertEqual(item["layer_precision"], 0.5)
        self.assertFalse(item["exact_graph_fact_pass"])
        self.assertEqual(report["failure_counts"]["extra_layers"], 1)

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

    def test_typed_module_algebra_rejects_unbound_duplicate_number(self) -> None:
        row = _row(
            idx=503,
            view_type="module_overlap_set_algebra",
            object_id="module_overlap",
            answer=(
                "`gw_dendrogram_module_1` and `rwr_loe_module_2` share 0 genes. "
                "Union size is 3; Jaccard overlap is 0. "
                "Intersection genes: none."
            ),
            prediction=(
                "`gw_dendrogram_module_1` and `rwr_loe_module_2` share 0 genes. "
                "Union size is 3; an unrelated count is 0."
            ),
        )

        report = eval_sft.evaluate_prediction_rows([row])
        item = report["examples"][0]

        self.assertEqual(item["missing_numbers"], [])
        self.assertEqual(item["typed_field_failures"], ["overlap_jaccard"])
        self.assertFalse(item["exact_graph_fact_pass"])

    def test_path_layer_counts_are_bound_to_their_layers(self) -> None:
        row = _row(
            idx=0,
            view_type="path_layer_decomposition",
            object_id="path_layers",
            answer=(
                "Path layer counts: `HumanNetV3:string_ppi` has 1 path edge and "
                "`HumanNetV3:coexpression` has 2 path edges."
            ),
            prediction=(
                "Path layer counts: `HumanNetV3:string_ppi` has 2 path edges and "
                "`HumanNetV3:coexpression` has 1 path edge."
            ),
        )

        report = eval_sft.evaluate_prediction_rows([row])
        item = report["examples"][0]

        self.assertEqual(item["missing_numbers"], [])
        self.assertEqual(item["typed_field_failures"], ["layer_counts"])
        self.assertFalse(item["exact_graph_fact_pass"])

    def test_nested_neighbor_layer_map_and_generic_json_are_bound(self) -> None:
        answer = json.dumps(
            {
                "unique_neighbor_count": 2,
                "neighbor_layer_map": {
                    "ENSG00000000005": ["HumanNetV3:string_ppi"],
                    "ENSG00000000007": ["HumanNetV3:coexpression"],
                },
            },
            sort_keys=True,
        )
        row = _row(
            idx=0,
            view_type="unique_multiplex_neighbors",
            object_id="neighbors",
            answer=answer,
            prediction=answer,
        )

        passed = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertTrue(passed["json_subset_match"])
        self.assertTrue(passed["exact_graph_fact_pass"])

        row["prediction"] = json.dumps(
            {
                "unique_neighbor_count": 2,
                "neighbor_layer_map": {
                    "ENSG00000000005": ["HumanNetV3:coexpression"],
                    "ENSG00000000007": ["HumanNetV3:string_ppi"],
                },
            },
            sort_keys=True,
        )
        failed = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertFalse(failed["json_subset_match"])
        self.assertEqual(failed["typed_field_failures"], ["neighbor_layer_map"])
        self.assertFalse(failed["exact_graph_fact_pass"])

    def test_embedded_edge_object_may_be_paraphrased_as_graph_facts(self) -> None:
        row = _row(
            idx=0,
            view_type="monoplex_shortest_path",
            object_id="path",
            answer=(
                "The shortest path in `HumanNetV3:string_ppi` is ENSG00000000003 -> "
                "ENSG00000000005 with hop count 1. Edges: "
                '[{"layer": "HumanNetV3:string_ppi", "source_gene_id": '
                '"ENSG00000000003", "target_gene_id": "ENSG00000000005"}].'
            ),
            prediction=(
                "The shortest path in `HumanNetV3:string_ppi` is ENSG00000000003 -> "
                "ENSG00000000005 (hop 1). Its edge connects ENSG00000000003 to "
                "ENSG00000000005 in `HumanNetV3:string_ppi`."
            ),
        )

        item = eval_sft.evaluate_prediction_rows([row])["examples"][0]

        self.assertIsNone(item["json_subset_match"])
        self.assertEqual(item["typed_field_failures"], [])
        self.assertTrue(item["exact_graph_fact_pass"])

    def test_connected_component_query_restatement_is_task_aware(self) -> None:
        row = _row(
            idx=0,
            view_type="connected_components",
            object_id="components",
            answer=(
                "There are 2 connected components: ENSG00000000003 and ENSG00000000005 "
                "form one; ENSG00000000007 forms the other."
            ),
            prediction=(
                "For queried genes ENSG00000000003, ENSG00000000005, ENSG00000000007, "
                "and ENSG00000000009 in HumanNetV3:string_ppi, there are 2 connected "
                "components: ENSG00000000003 and ENSG00000000005 form one; "
                "ENSG00000000007 forms the other."
            ),
        )
        row["question"] = (
            "In HumanNetV3:string_ppi, find components among ENSG00000000003, "
            "ENSG00000000005, ENSG00000000007, and ENSG00000000009."
        )

        passed = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertTrue(passed["exact_graph_fact_pass"])
        self.assertEqual(passed["allowed_context_ids"], ["ENSG00000000009"])
        self.assertEqual(passed["allowed_context_layers"], ["HumanNetV3:string_ppi"])

        row["prediction"] += " Novel ENSG00000000011 is in bulkPEN:brain:cortex."
        failed = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertEqual(failed["extra_ids"], ["ENSG00000000011"])
        self.assertEqual(failed["extra_layers"], ["bulkPEN:brain:cortex"])
        self.assertFalse(failed["exact_graph_fact_pass"])

    def test_nodes_present_context_id_is_allowed_only_when_explicitly_absent(self) -> None:
        row = _row(
            idx=0,
            view_type="nodes_present_by_layer",
            object_id="nodes",
            answer=(
                "`HumanNetV3:string_ppi` contains ENSG00000000003 and ENSG00000000005."
            ),
            prediction=(
                "`HumanNetV3:string_ppi` contains ENSG00000000003 and ENSG00000000005; "
                "ENSG00000000007 is not present."
            ),
        )
        row["question"] = (
            "Which of ENSG00000000003, ENSG00000000005, and ENSG00000000007 are "
            "present in HumanNetV3:string_ppi?"
        )

        absent = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertTrue(absent["exact_graph_fact_pass"])
        self.assertEqual(absent["allowed_context_ids"], ["ENSG00000000007"])

        row["prediction"] = (
            "`HumanNetV3:string_ppi` contains ENSG00000000003, ENSG00000000005, and "
            "ENSG00000000007."
        )
        falsely_present = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertEqual(falsely_present["extra_ids"], ["ENSG00000000007"])
        self.assertFalse(falsely_present["exact_graph_fact_pass"])

    def test_rwr_context_allows_seed_but_not_unrendered_candidates(self) -> None:
        canonical = {"rwr": {"payload": {"seed_gene_id": "ENSG00000000003"}}}
        row = _row(
            idx=0,
            view_type="rwr_neighborhood_interpretation",
            object_id="rwr",
            answer=(
                "The retrieved neighborhood contains ENSG00000000005, ENSG00000000007, "
                "and ENSG00000000009."
            ),
            prediction=(
                "For seed ENSG00000000003, the retrieved neighborhood contains "
                "ENSG00000000005, ENSG00000000007, ENSG00000000009, ENSG00000000011, "
                "and ENSG00000000013."
            ),
        )
        row["question"] = (
            "For seed ENSG00000000003, interpret ranked candidates ENSG00000000005, "
            "ENSG00000000007, ENSG00000000009, ENSG00000000011, and ENSG00000000013."
        )

        item = eval_sft.evaluate_prediction_rows(
            [row], canonical_objects_by_id=canonical
        )["examples"][0]

        self.assertEqual(item["allowed_context_ids"], ["ENSG00000000003"])
        self.assertEqual(item["extra_ids"], ["ENSG00000000011", "ENSG00000000013"])
        self.assertFalse(item["exact_graph_fact_pass"])

    def test_path_query_ids_may_be_restated_but_novel_ids_remain_fatal(self) -> None:
        row = _row(
            idx=0,
            view_type="path_layer_decomposition",
            object_id="path",
            answer="`HumanNetV3:string_ppi` has 2 path edges.",
            prediction=(
                "For path ENSG00000000003 to ENSG00000000005 via ENSG00000000007, "
                "`HumanNetV3:string_ppi` has 2 path edges."
            ),
        )
        row["question"] = (
            "Decompose the path ENSG00000000003 to ENSG00000000005 via "
            "ENSG00000000007 by layer."
        )

        passed = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertTrue(passed["exact_graph_fact_pass"])

        row["prediction"] += " It also visits ENSG00000000009."
        failed = eval_sft.evaluate_prediction_rows([row])["examples"][0]
        self.assertEqual(failed["extra_ids"], ["ENSG00000000009"])
        self.assertFalse(failed["exact_graph_fact_pass"])

    def test_rendered_answer_not_unrendered_payload_defines_expected_facts(self) -> None:
        canonical = {
            "rank": {
                "payload": {
                    "seed_gene_id": "ENSG00000275484",
                    "candidate_gene_id": "ENSG00000271653",
                    "winner_gene_id": "ENSG00000999999",
                    "rank": 1,
                    "score": 2.0326802531604104e-07,
                    "rank_cache_context": {"unrendered_rank": 999},
                }
            }
        }
        answer = (
            "For seed ENSG00000275484, candidate ENSG00000271653 has RWR-LOE rank 1 and "
            "score 2.03268025316041e-07. It is not a confirmed causal gene."
        )
        prediction = answer.replace("2.03268025316041e-07", "2.0326802531604104e-07")
        rows = [
            _row(
                idx=0,
                view_type="rwr_loe_rank_lookup",
                object_id="rank",
                answer=answer,
                prediction=prediction,
            )
        ]

        report = eval_sft.evaluate_prediction_rows(rows, canonical_objects_by_id=canonical)
        item = report["examples"][0]

        self.assertTrue(item["exact_graph_fact_pass"])
        self.assertEqual(item["id_recall"], 1.0)
        self.assertEqual(item["number_recall"], 1.0)
        self.assertEqual(item["missing_ids"], [])
        self.assertEqual(item["missing_numbers"], [])
        self.assertFalse(item["unsupported_language"])

    def test_structured_json_uses_rendered_target_as_recursive_subset(self) -> None:
        canonical = {
            "tool": {
                "payload": {
                    "tool_name": "get_neighbors",
                    "arguments": {
                        "gene_id": "ENSG00000000003",
                        "layer": "bulkPEN:brain:basal_ganglia:caudate",
                    },
                    "rank_cache_context": {"hidden": True},
                }
            }
        }
        answer = (
            '{"tool_name": "get_neighbors", "arguments": {'
            '"gene_id": "ENSG00000000003", '
            '"layer": "bulkPEN:brain:basal_ganglia:caudate"}}'
        )
        prediction = (
            '{"tool_name": "get_neighbors", "arguments": {'
            '"gene_id": "ENSG00000000003", '
            '"layer": "bulkPEN:brain:basal_ganglia:caudate"}, "trace_id": "extra"}'
        )
        rows = [
            _row(
                idx=0,
                view_type="tool_call_choice",
                object_id="tool",
                answer=answer,
                prediction=prediction,
            )
        ]

        report = eval_sft.evaluate_prediction_rows(rows, canonical_objects_by_id=canonical)

        self.assertTrue(report["examples"][0]["json_subset_match"])
        self.assertEqual(report["summary"]["exact_graph_fact_pass_rate"], 1.0)

        rows[0]["prediction"] = '{"tool_name": "get_neighbors", "arguments": {}}'
        failed = eval_sft.evaluate_prediction_rows(rows, canonical_objects_by_id=canonical)
        self.assertFalse(failed["examples"][0]["json_subset_match"])
        self.assertEqual(failed["summary"]["exact_graph_fact_pass_rate"], 0.0)

    def test_gold_self_evaluation_contract_reports_perfect_applicable_metrics(self) -> None:
        deep_layer = "scPEN:brain:amygdaloid-complex:committed_oligodendrocyte_precursor"
        rows = [
            _row(
                idx=0,
                view_type="monoplex_edge_existence",
                object_id="edge",
                answer=(
                    f"Yes. In `{deep_layer}`, ENSG00000001629 is directly connected to "
                    "ENSG00000155367 with weight -1.25e-07."
                ),
                prediction="wrong",
            ),
            _row(
                idx=1,
                view_type="rwr_loe_topk_membership",
                object_id="rank",
                answer=(
                    "No. For seed ENSG00000275484, ENSG00000271653 is not in the top 3. "
                    "It is not a confirmed causal gene."
                ),
                prediction="wrong",
            ),
        ]

        contract = eval_sft.evaluate_gold_self_contract(rows)
        report = eval_sft.evaluate_prediction_rows(rows)

        self.assertEqual(contract["status"], "passed")
        self.assertTrue(contract["passed"])
        self.assertEqual(contract["failure_count"], 0)
        self.assertEqual(contract["summary"]["exact_graph_fact_pass_rate"], 1.0)
        self.assertEqual(contract["summary"]["exact_applicable_count"], 2)
        self.assertEqual(contract["summary"]["exact_pass_count"], 2)
        self.assertEqual(contract["summary"]["id_applicable_count"], 2)
        self.assertEqual(contract["summary"]["layer_applicable_count"], 1)
        self.assertEqual(contract["summary"]["number_applicable_count"], 2)
        self.assertEqual(contract["summary"]["yes_no_applicable_count"], 2)
        self.assertEqual(contract["summary"]["mean_id_recall"], 1.0)
        self.assertEqual(contract["summary"]["mean_layer_recall"], 1.0)
        self.assertEqual(contract["summary"]["mean_number_recall"], 1.0)
        self.assertEqual(contract["summary"]["yes_no_accuracy"], 1.0)
        self.assertEqual(contract["summary"]["unsupported_language_rate"], 0.0)
        self.assertEqual(report["evaluator_contract"]["expected_fact_source"], "rendered_answer")
        self.assertEqual(report["gold_self_evaluation"], contract)

    def test_gold_self_evaluation_contract_marks_invalid_targets(self) -> None:
        rows = [
            _row(
                idx=0,
                view_type="rwr_neighborhood_interpretation",
                object_id="unsafe_gold",
                answer="ENSG00000275484 is definitely causal.",
                prediction="unused",
            )
        ]

        contract = eval_sft.evaluate_gold_self_contract(rows)

        self.assertEqual(contract["status"], "failed")
        self.assertFalse(contract["passed"])
        self.assertEqual(contract["failure_count"], 1)
        self.assertFalse(contract["checks"]["exact_graph_fact_pass_rate_is_one"])
        self.assertFalse(contract["checks"]["unsupported_language_rate_is_zero"])

    def test_extraction_coverage_is_independent_of_gold_self_comparison(self) -> None:
        row = _row(
            idx=0,
            view_type="monoplex_edge_existence",
            object_id="edge",
            answer=(
                "Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected "
                "to ENSG00000000005 with weight 0.495161."
            ),
            prediction="unused",
        )
        broken_number_re = re.compile(
            r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
            r"(?![A-Za-z0-9_.])"
        )

        with mock.patch.object(eval_sft, "NUMBER_RE", broken_number_re):
            coverage = eval_sft.evaluate_extraction_coverage([row])
            contract = eval_sft.evaluate_gold_self_contract([row])

        self.assertEqual(coverage["status"], "failed")
        self.assertIn(
            "number_not_extracted:weight:0.495161",
            coverage["failures"][0]["issues"],
        )
        self.assertFalse(contract["checks"]["extraction_coverage_passed"])
        self.assertEqual(contract["status"], "failed")

    def test_summary_separates_exact_rows_and_exact_failure_counts(self) -> None:
        exact = _row(
            idx=0,
            view_type="monoplex_edge_existence",
            object_id="edge",
            answer=(
                "Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected "
                "to ENSG00000000005."
            ),
            prediction=(
                "Yes. In `HumanNetV3:string_ppi`, ENSG00000000003 is directly connected "
                "to ENSG00000000005."
            ),
        )
        diagnostic_only = _row(
            idx=1,
            view_type="diagnostic_free_form",
            object_id="diagnostic",
            answer="Discuss ENSG00000000007.",
            prediction="Discuss ENSG00000000009 and the number 42.",
        )

        report = eval_sft.evaluate_prediction_rows([exact, diagnostic_only])

        self.assertEqual(report["summary"]["count"], 2)
        self.assertEqual(report["summary"]["mean_id_recall"], 0.5)
        self.assertEqual(report["summary"]["exact_only"]["count"], 1)
        self.assertEqual(report["summary"]["exact_only"]["mean_id_recall"], 1.0)
        self.assertEqual(report["exact_failure_counts"], {})
        self.assertEqual(report["failure_counts"]["missing_ids"], 1)
        self.assertEqual(report["failure_counts"]["extra_ids"], 1)
        self.assertEqual(report["failure_counts"]["extra_numbers"], 1)

    def test_report_uses_v3_contract(self) -> None:
        report = eval_sft.evaluate_prediction_rows([])

        self.assertEqual(eval_sft.EVALUATOR_CONTRACT_VERSION, "pretrajectory-sft-exact-v3")
        self.assertEqual(
            report["evaluator_contract"]["version"],
            "pretrajectory-sft-exact-v3",
        )

    def test_cli_writes_invalid_contract_then_exits_nonzero(self) -> None:
        row = _row(
            idx=0,
            view_type="rwr_neighborhood_interpretation",
            object_id="unsafe_gold",
            answer="ENSG00000275484 is definitely causal.",
            prediction="ENSG00000275484 is definitely causal.",
        )
        with tempfile.TemporaryDirectory() as directory:
            predictions_path = Path(directory) / "predictions.jsonl"
            report_path = Path(directory) / "report.json"
            predictions_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            argv = [
                "evaluate_pretrajectory_sft_predictions.py",
                "--predictions",
                str(predictions_path),
                "--json-out",
                str(report_path),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                eval_sft.main()

            self.assertEqual(raised.exception.code, 2)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["gold_self_evaluation"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
