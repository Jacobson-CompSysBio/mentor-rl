import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from runtime.tools import (
    build_multiplex_index,
    enrich_gene_set,
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

    def test_get_neighbors_treats_all_layer_alias_as_all_layers(self) -> None:
        index = build_multiplex_index(_build_test_multiplex())

        result = get_neighbors(index, "ENSG1", layers=["all"])

        self.assertFalse(result.is_empty)
        self.assertEqual(result.payload["unique_neighbors"], ["ENSG2", "ENSG3"])
        self.assertEqual(result.provenance["queried_layers"], ["ppi", "coexp"])

    def test_induce_subgraph_only_keeps_requested_nodes_and_edges(self) -> None:
        index = build_multiplex_index(_build_test_multiplex())

        result = induce_subgraph(index, ["ENSG1", "ENSG2", "ENSG4"], layers=["ppi", "coexp"])

        self.assertFalse(result.is_empty)
        self.assertEqual(result.payload["present_gene_ids"], ["ENSG1", "ENSG2", "ENSG4"])
        self.assertEqual(result.payload["combined_edge_count"], 1)
        self.assertEqual(result.payload["layers"][0]["edge_count"], 1)
        self.assertEqual(result.payload["layers"][1]["edge_count"], 0)

    def test_induce_subgraph_treats_empty_layers_as_all_layers(self) -> None:
        index = build_multiplex_index(_build_test_multiplex())

        result = induce_subgraph(index, ["ENSG1", "ENSG2", "ENSG3"], layers=[])

        self.assertFalse(result.is_empty)
        self.assertEqual(result.payload["combined_edge_count"], 3)
        self.assertEqual(result.provenance["queried_layers"], ["ppi", "coexp"])

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

    def test_query_mygene_network_expands_default_fields_and_writes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "mygene_cache.json"

            with patch(
                "runtime.tools._fetch_mygene_hits",
                return_value=[
                    {
                        "_id": "1",
                        "query": "ENSG1",
                        "symbol": "GENE1",
                        "summary": "A useful annotation.",
                        "type_of_gene": "protein-coding",
                        "go": {"BP": [{"id": "GO:1", "term": "toy process"}]},
                    }
                ],
            ) as fetch:
                result = query_mygene(
                    "ENSG1",
                    cache={},
                    cache_path=str(cache_path),
                    allow_network=True,
                )

            self.assertTrue(cache_path.exists())

        self.assertFalse(result.is_empty)
        self.assertEqual(result.provenance["source"], "network")
        self.assertIn("summary", result.payload["requested_fields"])
        self.assertIn("go", result.payload["requested_fields"])
        self.assertEqual(result.payload["results"][0]["type_of_gene"], "protein-coding")
        fetch.assert_called_once()

    def test_enrich_gene_set_uses_network_cache_and_custom_background(self) -> None:
        cache: dict[str, dict] = {}
        with patch(
            "runtime.tools._fetch_gprofiler_enrichment",
            return_value={
                "results": [
                    {
                        "source": "GO:BP",
                        "native": "GO:0000001",
                        "name": "toy process",
                        "p_value": 0.001,
                        "significant": True,
                        "intersection_size": 2,
                        "query_size": 3,
                        "precision": 0.67,
                    }
                ],
                "raw_result_count": 1,
                "meta": {"version": "fake"},
            },
        ) as fetch:
            first = enrich_gene_set(
                ["ENSG1", "ENSG2", "ENSG3"],
                background_gene_ids=["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
                sources=["GO:BP"],
                cache=cache,
                allow_network=True,
            )
            second = enrich_gene_set(
                ["ENSG1", "ENSG2", "ENSG3"],
                background_gene_ids=["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
                sources=["GO:BP"],
                cache=cache,
                allow_network=False,
            )

        self.assertFalse(first.is_empty)
        self.assertEqual(first.provenance["source"], "network")
        self.assertEqual(second.provenance["source"], "cache")
        self.assertEqual(first.payload["background_gene_count"], 4)
        self.assertEqual(first.payload["results"][0]["native"], "GO:0000001")
        fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
