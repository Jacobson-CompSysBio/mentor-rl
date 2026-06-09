import unittest
from unittest.mock import patch

import networkx as nx

from runtime.environment import RuntimeEnvironment
from runtime.schemas import ToolAction, ToolObservationStatus
from runtime.state import initialize_state_from_corum_task
from runtime.tools import ToolExecutionResult
from utils.multiplex import Multiplex


def _build_task_row() -> dict:
    return {
        "task_id": "corum_complex_00001.refinement.easy.graph",
        "query_text": "Refine the candidate complex around HDAC4 and BCL6.",
        "evidence_mode": "graph",
        "visible_inputs": {
            "seed_gene_ids": ["ENSG1", "ENSG2"],
            "seed_gene_symbols": ["HDAC4", "BCL6"],
            "context_text": None,
            "graph_query_spec": {"operator": "induce_subgraph"},
            "structured_annotations": None,
        },
    }


def _build_environment() -> RuntimeEnvironment:
    multiplex = Multiplex()

    ppi = nx.Graph()
    ppi.add_edge("ENSG1", "ENSG2", weight=1.0)
    ppi.add_edge("ENSG2", "ENSG3", weight=0.9)
    multiplex.add_layer(ppi, "ppi")

    tf = nx.Graph()
    tf.add_edge("ENSG3", "ENSG4", weight=0.5)
    multiplex.add_layer(tf, "tf")

    return RuntimeEnvironment(
        multiplex=multiplex,
        mygene_cache={
            "BCL6": [
                {
                    "_id": "604",
                    "query": "BCL6",
                    "symbol": "BCL6",
                    "name": "BCL6 transcription repressor",
                }
            ]
        },
        enrichment_cache={
            "fake_key": {
                "query_gene_ids": ["ENSG1", "ENSG2"],
                "query_gene_count": 2,
                "background_gene_count": 4,
                "background_hash": "fake_hash",
                "organism": "hsapiens",
                "sources": ["GO:BP"],
                "user_threshold": 0.05,
                "top_k": 10,
                "results": [
                    {
                        "source": "GO:BP",
                        "native": "GO:0000001",
                        "name": "toy process",
                        "p_value": 0.01,
                        "significant": True,
                        "intersection_size": 2,
                        "query_size": 2,
                        "precision": 1.0,
                    }
                ],
                "raw_result_count": 1,
                "meta": {},
            }
        },
        enrichment_background_gene_ids=["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
    )


class FakeStructuredRwrHpcBackend:
    def __init__(self) -> None:
        self.calls = []

    def run_rwr(self, request) -> ToolExecutionResult:
        self.calls.append(("rwr", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "rwr",
                "seed_genes": list(request.seed_genes),
                "ranked_genes": [{"gene": "ENSG3", "gene_id": "ENSG3", "rank": 1.0}],
                "results": [{"gene": "ENSG3", "gene_id": "ENSG3", "rank": 1.0}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_rwr_loe(self, request) -> ToolExecutionResult:
        self.calls.append(("rwr_loe", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "rwr_loe",
                "seed_genes": list(request.seed_genes),
                "query_genes": list(request.query_genes),
                "ranked_genes": [{"gene": "ENSG3", "gene_id": "ENSG3", "rank": 1}],
                "results": [{"gene": "ENSG3", "gene_id": "ENSG3", "rank": 1}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_shortest_paths(self, request) -> ToolExecutionResult:
        self.calls.append(("shortest_paths", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "shortest_paths",
                "source_genes": list(request.source_genes),
                "target_genes": list(request.target_genes),
                "paths": [{"path_genes": ["ENSG1", "ENSG2"], "path_length": 1}],
                "results": [{"path_genes": ["ENSG1", "ENSG2"], "path_length": 1}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_rank(self, request) -> ToolExecutionResult:
        self.calls.append(("get_rank", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "get_rank",
                "source_gene": request.source_gene,
                "target_gene": request.target_gene,
                "target_rank": 2.0,
                "rank_result": {"gene": request.target_gene, "rank": 2.0},
                "results": [{"gene": request.target_gene, "rank": 2.0}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_distance(self, request) -> ToolExecutionResult:
        self.calls.append(("get_distance", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "get_distance",
                "gene_a": request.gene_a,
                "gene_b": request.gene_b,
                "distance_metric": request.distance_metric,
                "distance": 0.25,
                "results": [{"gene_a": request.gene_a, "gene_b": request.gene_b, "distance": 0.25}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_spearman(self, request) -> ToolExecutionResult:
        self.calls.append(("get_spearman", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "get_spearman",
                "gene_a": request.gene_a,
                "gene_b": request.gene_b,
                "spearman_correlation": 0.75,
                "results": [{"gene_a": request.gene_a, "gene_b": request.gene_b, "spearman_correlation": 0.75}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_pearson(self, request) -> ToolExecutionResult:
        self.calls.append(("get_pearson", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "get_pearson",
                "gene_a": request.gene_a,
                "gene_b": request.gene_b,
                "pearson_correlation": 0.6,
                "results": [{"gene_a": request.gene_a, "gene_b": request.gene_b, "pearson_correlation": 0.6}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_dot_similarity(self, request) -> ToolExecutionResult:
        self.calls.append(("get_dot_similarity", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "get_dot_similarity",
                "gene_a": request.gene_a,
                "gene_b": request.gene_b,
                "dot_similarity": 0.8,
                "results": [{"gene_a": request.gene_a, "gene_b": request.gene_b, "dot_similarity": 0.8}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_rank_vector_summary(self, request) -> ToolExecutionResult:
        self.calls.append(("get_rank_vector_summary", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "get_rank_vector_summary",
                "seed_genes": list(request.seed_genes),
                "rank_summary": [{"gene": "ENSG3", "rank": 1.0}],
                "results": [{"gene": "ENSG3", "rank": 1.0}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_encoding_summary(self, request) -> ToolExecutionResult:
        self.calls.append(("get_encoding_summary", request))
        return ToolExecutionResult(
            payload={
                "tool_name": "get_encoding_summary",
                "seed_genes": list(request.seed_genes),
                "encoding_summary": [{"gene": "ENSG3", "score": 0.9}],
                "results": [{"gene": "ENSG3", "score": 0.9}],
            },
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_gene_layers(self, request) -> ToolExecutionResult:
        self.calls.append(("get_gene_layers", request))
        return ToolExecutionResult(
            payload={"tool_name": "get_gene_layers", "gene": request.gene, "layers": ["ppi"], "results": [{"gene": request.gene, "layers": ["ppi"]}]},
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_nodes_by_layer(self, request) -> ToolExecutionResult:
        self.calls.append(("get_nodes_by_layer", request))
        return ToolExecutionResult(
            payload={"tool_name": "get_nodes_by_layer", "gene": request.gene, "layers": ["ppi"], "results": [{"gene": request.gene, "layers": ["ppi"]}]},
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_layer_stats(self, request) -> ToolExecutionResult:
        self.calls.append(("get_layer_stats", request))
        return ToolExecutionResult(
            payload={"tool_name": "get_layer_stats", "layer_stats": [{"layer": "ppi", "edge_count": 2}], "results": [{"layer": "ppi", "edge_count": 2}]},
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_path_layer_counts(self, request) -> ToolExecutionResult:
        self.calls.append(("get_path_layer_counts", request))
        return ToolExecutionResult(
            payload={"tool_name": "get_path_layer_counts", "layer_counts": [{"layer": "ppi", "edge_count": 2}], "results": [{"layer": "ppi", "edge_count": 2}]},
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_component_summary(self, request) -> ToolExecutionResult:
        self.calls.append(("get_component_summary", request))
        return ToolExecutionResult(
            payload={"tool_name": "get_component_summary", "total_components": 1, "components": [{"component_id": "c1", "size": 2}], "results": [{"component_id": "c1", "size": 2}]},
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_seed_essentiality(self, request) -> ToolExecutionResult:
        self.calls.append(("get_seed_essentiality", request))
        return ToolExecutionResult(
            payload={"tool_name": "get_seed_essentiality", "seed_genes": list(request.seed_genes), "essentiality": [{"gene": "ENSG1"}], "results": [{"gene": "ENSG1"}]},
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_layer_ablation(self, request) -> ToolExecutionResult:
        self.calls.append(("get_layer_ablation", request))
        return ToolExecutionResult(
            payload={"tool_name": "get_layer_ablation", "seed_genes": list(request.seed_genes), "layer_effects": [{"layer": "ppi"}], "results": [{"layer": "ppi"}]},
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )

    def run_get_node_perturbation(self, request) -> ToolExecutionResult:
        self.calls.append(("get_node_perturbation", request))
        return ToolExecutionResult(
            payload={"tool_name": "get_node_perturbation", "seed_genes": list(request.seed_genes), "perturb_genes": list(request.perturb_genes), "perturbation_effects": [{"perturb_gene": "ENSG2"}], "results": [{"perturb_gene": "ENSG2"}]},
            provenance={"backend": "fake_structured_rwr_hpc"},
            is_empty=False,
        )


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_describe_reports_basic_runtime_summary(self) -> None:
        environment = _build_environment()

        summary = environment.describe()

        self.assertEqual(summary["layer_count"], 2)
        self.assertEqual(summary["gene_count"], 4)
        self.assertEqual(summary["mygene_cache_size"], 1)
        self.assertEqual(summary["enrichment_cache_size"], 1)
        self.assertEqual(summary["enrichment_background_gene_count"], 4)
        self.assertFalse(summary["rwr_hpc_structured_tools_required"])
        self.assertIsNone(summary["rwr_hpc_flist"])

    def test_execute_invalid_action_returns_invalid_observation(self) -> None:
        environment = _build_environment()
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG1", "layers": ["missing_layer"]},
            call_id="call_invalid",
        )

        observation = environment.execute(action)

        self.assertEqual(observation.status, ToolObservationStatus.INVALID)
        self.assertIn("unknown layers", observation.error.lower())
        self.assertEqual(observation.provenance["tool_name"], "get_neighbors")

    def test_execute_duplicate_action_returns_invalid_observation(self) -> None:
        environment = _build_environment()
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG1", "layers": ["ppi"]},
            call_id="call_repeat",
        )

        observation = environment.execute(action, prior_actions=[action])

        self.assertEqual(observation.status, ToolObservationStatus.INVALID)
        self.assertIn("duplicate", observation.error.lower())

    def test_execute_successful_graph_call_returns_success(self) -> None:
        environment = _build_environment()
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG1", "layers": ["ppi"]},
            call_id="call_neighbors",
        )

        observation = environment.execute(action)

        self.assertEqual(observation.status, ToolObservationStatus.SUCCESS)
        self.assertEqual(observation.payload["unique_neighbors"], ["ENSG2"])
        self.assertEqual(observation.provenance["runtime_version"], "mentor-rl-runtime-v1")

    def test_execute_normalizes_all_layer_alias(self) -> None:
        environment = _build_environment()
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG1", "layers": ["all"]},
            call_id="call_neighbors",
        )

        observation = environment.execute(action)

        self.assertEqual(observation.status, ToolObservationStatus.SUCCESS)
        self.assertEqual(observation.provenance["queried_layers"], ["ppi", "tf"])

    def test_execute_empty_path_returns_empty_observation(self) -> None:
        environment = _build_environment()
        action = ToolAction(
            tool_name="shortest_path",
            arguments={"source": "ENSG1", "target": "ENSG4", "layer": "ppi"},
            call_id="call_empty_path",
        )

        observation = environment.execute(action)

        self.assertEqual(observation.status, ToolObservationStatus.EMPTY)
        self.assertEqual(observation.payload["path_gene_ids"], [])

    def test_execute_query_mygene_reads_cached_hit(self) -> None:
        environment = _build_environment()
        action = ToolAction(
            tool_name="query_mygene",
            arguments={"query": "BCL6", "fields": ["symbol", "name"]},
            call_id="call_mygene",
        )

        observation = environment.execute(action)

        self.assertEqual(observation.status, ToolObservationStatus.SUCCESS)
        self.assertEqual(observation.provenance["source"], "cache")
        self.assertEqual(observation.payload["results"][0]["symbol"], "BCL6")

    def test_execute_enrich_gene_set_reads_cached_result(self) -> None:
        environment = _build_environment()
        action = ToolAction(
            tool_name="enrich_gene_set",
            arguments={"genes": ["ENSG1", "ENSG2"], "sources": ["GO:BP"]},
            call_id="call_enrich",
        )

        with patch(
            "runtime.environment.enrich_gene_set",
            return_value=ToolExecutionResult(
                payload={
                    "query_gene_ids": ["ENSG1", "ENSG2"],
                    "results": [{"name": "toy process"}],
                },
                provenance={"tool_name": "enrich_gene_set", "source": "cache"},
            ),
        ) as enrich:
            observation = environment.execute(action)

        self.assertEqual(observation.status, ToolObservationStatus.SUCCESS)
        self.assertEqual(observation.provenance["source"], "cache")
        self.assertEqual(observation.payload["results"][0]["name"], "toy process")
        self.assertEqual(enrich.call_args.kwargs["background_gene_ids"], ["ENSG1", "ENSG2", "ENSG3", "ENSG4"])
        self.assertIs(enrich.call_args.kwargs["cache"], environment.enrichment_cache)

    def test_execute_dispatches_rwr_plus_plus_structured_tools(self) -> None:
        environment = _build_environment()
        fake_backend = FakeStructuredRwrHpcBackend()
        environment.rwr_hpc_structured_backend = fake_backend

        actions = [
            ToolAction(
                tool_name="rwr",
                arguments={"seed_genes": ["ENSG1"], "layers": ["ppi"], "top_k": 5},
                call_id="call_rwr",
            ),
            ToolAction(
                tool_name="rwr_loe",
                arguments={"seed_genes": ["ENSG1"], "query_genes": ["ENSG3"], "top_k": 5},
                call_id="call_rwr_loe",
            ),
            ToolAction(
                tool_name="shortest_paths",
                arguments={"source_genes": ["ENSG1"], "target_genes": ["ENSG2"]},
                call_id="call_shortest_paths",
            ),
            ToolAction(
                tool_name="get_rank",
                arguments={"source_gene": "ENSG1", "target_gene": "ENSG2"},
                call_id="call_get_rank",
            ),
            ToolAction(
                tool_name="get_distance",
                arguments={"gene_a": "ENSG1", "gene_b": "ENSG2"},
                call_id="call_get_distance",
            ),
            ToolAction(
                tool_name="get_spearman",
                arguments={"gene_a": "ENSG1", "gene_b": "ENSG2"},
                call_id="call_get_spearman",
            ),
            ToolAction(
                tool_name="get_pearson",
                arguments={"gene_a": "ENSG1", "gene_b": "ENSG2"},
                call_id="call_get_pearson",
            ),
            ToolAction(
                tool_name="get_dot_similarity",
                arguments={"gene_a": "ENSG1", "gene_b": "ENSG2"},
                call_id="call_get_dot_similarity",
            ),
            ToolAction(
                tool_name="get_rank_vector_summary",
                arguments={"seed_genes": ["ENSG1"], "top_k": 2},
                call_id="call_get_rank_vector_summary",
            ),
            ToolAction(
                tool_name="get_encoding_summary",
                arguments={"seed_genes": ["ENSG1"], "top_k": 2},
                call_id="call_get_encoding_summary",
            ),
            ToolAction(
                tool_name="get_gene_layers",
                arguments={"gene": "ENSG1"},
                call_id="call_get_gene_layers",
            ),
            ToolAction(
                tool_name="get_nodes_by_layer",
                arguments={"gene": "ENSG1"},
                call_id="call_get_nodes_by_layer",
            ),
            ToolAction(
                tool_name="get_layer_stats",
                arguments={"top_k": 2},
                call_id="call_get_layer_stats",
            ),
            ToolAction(
                tool_name="get_path_layer_counts",
                arguments={"source_genes": ["ENSG1"], "target_genes": ["ENSG2"], "top_k": 2},
                call_id="call_get_path_layer_counts",
            ),
            ToolAction(
                tool_name="get_component_summary",
                arguments={"genes": ["ENSG1"], "max_components": 2},
                call_id="call_get_component_summary",
            ),
            ToolAction(
                tool_name="get_seed_essentiality",
                arguments={"seed_genes": ["ENSG1", "ENSG2"], "top_k": 2},
                call_id="call_get_seed_essentiality",
            ),
            ToolAction(
                tool_name="get_layer_ablation",
                arguments={"seed_genes": ["ENSG1"], "top_k": 2},
                call_id="call_get_layer_ablation",
            ),
            ToolAction(
                tool_name="get_node_perturbation",
                arguments={"seed_genes": ["ENSG1"], "perturb_genes": ["ENSG2"], "top_k": 2},
                call_id="call_get_node_perturbation",
            ),
        ]

        observations = [environment.execute(action) for action in actions]

        self.assertTrue(all(observation.status == ToolObservationStatus.SUCCESS for observation in observations))
        self.assertEqual(
            [name for name, _ in fake_backend.calls],
            [
                "rwr",
                "rwr_loe",
                "shortest_paths",
                "get_rank",
                "get_distance",
                "get_spearman",
                "get_pearson",
                "get_dot_similarity",
                "get_rank_vector_summary",
                "get_encoding_summary",
                "get_gene_layers",
                "get_nodes_by_layer",
                "get_layer_stats",
                "get_path_layer_counts",
                "get_component_summary",
                "get_seed_essentiality",
                "get_layer_ablation",
                "get_node_perturbation",
            ],
        )
        self.assertEqual(fake_backend.calls[0][1].seed_genes, ("ENSG1",))
        self.assertEqual(fake_backend.calls[0][1].layers, ("ppi",))
        self.assertEqual(fake_backend.calls[1][1].query_genes, ("ENSG3",))
        self.assertEqual(fake_backend.calls[2][1].source_genes, ("ENSG1",))
        self.assertEqual(fake_backend.calls[3][1].target_gene, "ENSG2")
        self.assertEqual(fake_backend.calls[4][1].gene_b, "ENSG2")
        self.assertEqual(fake_backend.calls[5][1].gene_a, "ENSG1")
        self.assertEqual(fake_backend.calls[8][1].seed_genes, ("ENSG1",))
        self.assertEqual(fake_backend.calls[10][1].gene, "ENSG1")
        self.assertEqual(fake_backend.calls[17][1].perturb_genes, ("ENSG2",))

    def test_legacy_rwr_tool_names_route_to_structured_rwr_when_available(self) -> None:
        environment = _build_environment()
        fake_backend = FakeStructuredRwrHpcBackend()
        environment.rwr_hpc_structured_backend = fake_backend

        actions = [
            ToolAction(
                tool_name="rwr_multiplex",
                arguments={"seeds": ["ENSG1"], "top_k": 5},
                call_id="call_rwr_multiplex",
            ),
            ToolAction(
                tool_name="rwr_monoplex",
                arguments={"seeds": ["ENSG1"], "layer": "ppi", "top_k": 5},
                call_id="call_rwr_monoplex",
            ),
        ]

        observations = [environment.execute(action) for action in actions]

        self.assertTrue(all(observation.status == ToolObservationStatus.SUCCESS for observation in observations))
        self.assertEqual([name for name, _ in fake_backend.calls], ["rwr", "rwr"])
        self.assertEqual(fake_backend.calls[0][1].seed_genes, ("ENSG1",))
        self.assertEqual(fake_backend.calls[0][1].layers, ())
        self.assertEqual(fake_backend.calls[1][1].layers, ("ppi",))

    def test_require_rwr_hpc_structured_tools_fails_without_backend(self) -> None:
        multiplex = Multiplex()
        ppi = nx.Graph()
        ppi.add_edge("ENSG1", "ENSG2", weight=1.0)
        multiplex.add_layer(ppi, "ppi")

        with self.assertRaisesRegex(ValueError, "requires rwr_hpc_flist"):
            RuntimeEnvironment(
                multiplex=multiplex,
                require_rwr_hpc_structured_tools=True,
            )

    def test_required_rwr_hpc_flag_blocks_legacy_rwr_fallback(self) -> None:
        environment = _build_environment()
        environment.require_rwr_hpc_structured_tools = True
        action = ToolAction(
            tool_name="rwr",
            arguments={"seed_genes": ["ENSG1"], "top_k": 5},
            call_id="call_rwr_required",
        )

        observation = environment.execute(action)

        self.assertEqual(observation.status, ToolObservationStatus.ERROR)
        self.assertIn("required", observation.error.lower())

    def test_execute_respects_runtime_state_budget_validation(self) -> None:
        environment = _build_environment()
        _, state = initialize_state_from_corum_task(_build_task_row(), max_budget=0)
        action = ToolAction(
            tool_name="rwr_monoplex",
            arguments={"seeds": ["ENSG1"], "layer": "ppi", "top_k": 5},
            call_id="call_budget",
        )

        observation = environment.execute(action, state=state)

        self.assertEqual(observation.status, ToolObservationStatus.INVALID)
        self.assertIn("remaining budget is 0", observation.error.lower())


if __name__ == "__main__":
    unittest.main()
