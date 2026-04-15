import unittest

import networkx as nx

from runtime.environment import RuntimeEnvironment
from runtime.schemas import ToolAction, ToolObservationStatus
from runtime.state import initialize_state_from_corum_task
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
    )


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_describe_reports_basic_runtime_summary(self) -> None:
        environment = _build_environment()

        summary = environment.describe()

        self.assertEqual(summary["layer_count"], 2)
        self.assertEqual(summary["gene_count"], 4)
        self.assertEqual(summary["mygene_cache_size"], 1)

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
