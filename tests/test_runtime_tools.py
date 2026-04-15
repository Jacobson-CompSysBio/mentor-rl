import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from runtime.tools import (
    build_multiplex_index,
    get_neighbors,
    induce_subgraph,
    load_mygene_cache,
    query_mygene,
    rwr_monoplex,
    rwr_multiplex,
    save_mygene_cache,
    shortest_path,
)
from utils.multiplex import Multiplex


def _build_test_multiplex() -> Multiplex:
    multiplex = Multiplex()

    ppi = nx.Graph()
    ppi.add_edge("ENSG1", "ENSG2", weight=1.0)
    ppi.add_edge("ENSG2", "ENSG3", weight=0.9)
    multiplex.add_layer(ppi, "ppi")

    coexp = nx.Graph()
    coexp.add_edge("ENSG1", "ENSG3", weight=0.8)
    coexp.add_edge("ENSG3", "ENSG4", weight=0.7)
    multiplex.add_layer(coexp, "coexp")

    return multiplex


class RuntimeToolsTests(unittest.TestCase):
    def test_get_neighbors_returns_layer_breakdown(self) -> None:
        index = build_multiplex_index(_build_test_multiplex())

        result = get_neighbors(index, "ENSG1", layers=["ppi", "coexp"])

        self.assertFalse(result.is_empty)
        self.assertEqual(result.payload["unique_neighbors"], ["ENSG2", "ENSG3"])
        self.assertEqual(result.payload["layers"][0]["layer_name"], "ppi")
        self.assertEqual(result.payload["layers"][1]["layer_name"], "coexp")

    def test_induce_subgraph_only_keeps_requested_nodes_and_edges(self) -> None:
        index = build_multiplex_index(_build_test_multiplex())

        result = induce_subgraph(index, ["ENSG1", "ENSG2", "ENSG4"], layers=["ppi", "coexp"])

        self.assertFalse(result.is_empty)
        self.assertEqual(result.payload["present_gene_ids"], ["ENSG1", "ENSG2", "ENSG4"])
        self.assertEqual(result.payload["combined_edge_count"], 1)
        self.assertEqual(result.payload["layers"][0]["edge_count"], 1)
        self.assertEqual(result.payload["layers"][1]["edge_count"], 0)

    def test_shortest_path_can_use_aggregate_graph(self) -> None:
        index = build_multiplex_index(_build_test_multiplex())

        result = shortest_path(index, "ENSG2", "ENSG4")

        self.assertFalse(result.is_empty)
        self.assertEqual(result.provenance["search_mode"], "aggregate_multiplex")
        self.assertEqual(result.payload["path_gene_ids"], ["ENSG2", "ENSG3", "ENSG4"])
        self.assertEqual(result.payload["hop_count"], 2)

    def test_rwr_monoplex_returns_ranked_results(self) -> None:
        index = build_multiplex_index(_build_test_multiplex())

        result = rwr_monoplex(index, ["ENSG1"], layer="ppi", top_k=3)

        self.assertFalse(result.is_empty)
        self.assertEqual(result.payload["layer_name"], "ppi")
        self.assertEqual(result.payload["results"][0]["gene_id"], "ENSG1")
        self.assertEqual(len(result.payload["results"]), 3)

    def test_rwr_multiplex_averages_across_active_layers(self) -> None:
        index = build_multiplex_index(_build_test_multiplex())

        result = rwr_multiplex(index, ["ENSG1"], top_k=4)

        self.assertFalse(result.is_empty)
        self.assertEqual(result.payload["active_layers"], ["ppi", "coexp"])
        self.assertEqual(result.payload["results"][0]["gene_id"], "ENSG1")
        ranked_gene_ids = [item["gene_id"] for item in result.payload["results"]]
        self.assertIn("ENSG3", ranked_gene_ids)

    def test_query_mygene_uses_cache_without_network(self) -> None:
        cache = {
            "BCL6": [
                {
                    "_id": "604",
                    "query": "BCL6",
                    "symbol": "BCL6",
                    "name": "BCL6 transcription repressor",
                    "entrezgene": 604,
                }
            ]
        }

        result = query_mygene("BCL6", fields=["symbol", "entrezgene"], cache=cache)

        self.assertFalse(result.is_empty)
        self.assertEqual(result.provenance["source"], "cache")
        self.assertEqual(result.payload["result_count"], 1)
        self.assertEqual(result.payload["results"][0]["symbol"], "BCL6")
        self.assertEqual(result.payload["results"][0]["entrezgene"], 604)

    def test_mygene_cache_helpers_round_trip_json(self) -> None:
        cache = {"HDAC4": [{"_id": "9759", "query": "HDAC4", "symbol": "HDAC4"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "mygene_cache.json"
            save_mygene_cache(cache, str(cache_path))
            loaded = load_mygene_cache(str(cache_path))

        self.assertEqual(json.dumps(loaded, sort_keys=True), json.dumps(cache, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
