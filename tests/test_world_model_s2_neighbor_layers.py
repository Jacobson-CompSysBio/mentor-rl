"""Check canonical neighbor selection, native support, and exact map transport."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import unittest

from runtime.world_model.s2_multiplex import SelectionError
from runtime.world_model.s2_neighbor_layers import NeighborLayersRuntime
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


class S2NeighborLayerTests(unittest.TestCase):
    """Use independent graph fixtures with reciprocal arcs, loops, and mixed support."""

    def setUp(self) -> None:
        self.a, self.b, self.c, self.d, self.e, self.missing = [f"ENSG{n:011d}" for n in range(1, 7)]
        self.ppi, self.coex, self.brain, self.cd4 = (
            "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells",
        )
        specs = {
            self.ppi: (False, {(self.a, self.b): 0, (self.a, self.d): 0.99}),
            self.coex: (False, {(self.a, self.b): 0.123456789012345, (self.c, self.d): 0.2}),
            self.brain: (True, {(self.a, self.a): 0.8, (self.b, self.a): 0.3,
                               (self.a, self.b): 0.4, (self.c, self.a): 0.9,
                               (self.a, self.d): 0.5}),
            self.cd4: (True, {(self.e, self.a): 1}),
        }
        self.view = GraphView("unit_neighbor_layers", "unit_map_version", {
            name: LayerGraph(name, directed, edges, frozenset(gene for pair in edges for gene in pair))
            for name, (directed, edges) in specs.items()
        })
        self.index = NeighborIndex(self.view)
        self.runtime = NeighborLayersRuntime(self.view, index=self.index)
        self.all = {"kind": "all", "layer_ids": [], "layer_family": None}

    def scope(self, *layers: str) -> dict:
        """Declare an explicit scope in the supplied user order."""
        return {"kind": "layers", "layer_ids": list(layers), "layer_family": None}

    def arguments(self, **changes) -> dict:
        """Construct the exact explicit acquisition arguments for this fixture."""
        return {**self.runtime.arguments(self.a, self.all, "incident"), **changes}

    def query(self, runtime=None, **changes) -> dict:
        """Acquire one native map with a unique test-local evidence identity."""
        return (runtime or self.runtime).get_neighbor_layers(**self.arguments(**changes), evidence_id="ev_map")

    def traverse(self, runtime, arguments) -> list:
        """Follow returned opaque cursors without inference from their encoded bytes."""
        pages = []
        next_args = deepcopy(arguments)
        for ordinal in range(20):
            response = runtime.get_neighbor_layers(**next_args, evidence_id=f"ev_page_{ordinal}")
            self.assertEqual(response["status"], "resolved")
            pages.append(response)
            cursor = response["result"]["collection"]["next_cursor"]
            if cursor is None:
                return pages
            next_args["cursor"] = cursor
        self.fail("finite fixture did not terminate")

    def test_canonical_prefix_retains_all_support_and_native_weights(self) -> None:
        result = self.query(mode="top_k", top_k=2)["result"]
        self.assertEqual([row["gene_id"] for row in result["neighbors"]], [self.a, self.b])
        loop, supported = result["neighbors"]
        self.assertEqual(loop, {"gene_id": self.a, "supporting_layers": [
            {"layer_id": self.brain, "directed": True, "edges": [{"gene_ids": [self.a, self.a], "weight": 0.8}]},
        ], "supporting_layer_count": 1})
        self.assertEqual(supported, {"gene_id": self.b, "supporting_layers": [
            {"layer_id": self.ppi, "directed": False, "edges": [{"gene_ids": [self.a, self.b], "weight": 0}]},
            {"layer_id": self.coex, "directed": False,
             "edges": [{"gene_ids": [self.a, self.b], "weight": 0.123456789012345}]},
            {"layer_id": self.brain, "directed": True, "edges": [
                {"gene_ids": [self.b, self.a], "weight": 0.3}, {"gene_ids": [self.a, self.b], "weight": 0.4},
            ]},
        ], "supporting_layer_count": 3})
        self.assertEqual(result["unique_neighbor_count"], 5)
        self.assertEqual(result["collection"], {"mode": "top_k", "returned_count": 2, "total_count": 5,
                                               "complete": False, "next_cursor": None})
        self.assertEqual(result["coverage"]["checked_layer_ids"], list(self.view.layer_order))
        self.assertEqual(result["coverage"]["unchecked_layer_ids"], [])

    def test_complete_union_counts_each_neighbor_once(self) -> None:
        result = self.query()["result"]
        self.assertEqual(list(result), ["query_gene_id", "multiplex_id", "layer_scope", "direction",
                                       "layer_results", "neighbors", "unique_neighbor_count", "missing_gene_ids",
                                       "layers_without_gene", "coverage", "collection"])
        self.assertEqual([row["gene_id"] for row in result["neighbors"]], [self.a, self.b, self.c, self.d, self.e])
        self.assertEqual(result["unique_neighbor_count"], 5)
        self.assertTrue(result["collection"]["complete"])
        self.assertEqual([row["supporting_layer_count"] for row in result["neighbors"]], [1, 3, 1, 2, 1])
        for limit in (5, 6, 1000):
            bounded = self.query(mode="top_k", top_k=limit)["result"]
            self.assertEqual(bounded["neighbors"], result["neighbors"])
            self.assertTrue(bounded["collection"]["complete"])

    def test_layer_order_and_scope_deduplication_preserve_source_roles(self) -> None:
        result = self.query(layer_scope=self.scope(self.brain, self.ppi, self.brain))["result"]
        self.assertEqual(result["layer_scope"], self.scope(self.ppi, self.brain))
        self.assertEqual([row["layer_id"] for row in result["layer_results"]], [self.ppi, self.brain])
        neighbor = next(row for row in result["neighbors"] if row["gene_id"] == self.b)
        self.assertEqual([row["layer_id"] for row in neighbor["supporting_layers"]], [self.ppi, self.brain])
        reverse = self.query(gene_id=self.b, layer_scope=self.scope(self.brain))["result"]["neighbors"][0]
        self.assertEqual(reverse["gene_id"], self.a)
        self.assertEqual(reverse["supporting_layers"][0]["edges"], [
            {"gene_ids": [self.a, self.b], "weight": 0.4}, {"gene_ids": [self.b, self.a], "weight": 0.3},
        ])

    def test_native_inputs_follow_manifest_and_incoming_then_outgoing(self) -> None:
        expected = [{"layer_id": self.ppi, "direction": "undirected"},
                    {"layer_id": self.coex, "direction": "undirected"},
                    {"layer_id": self.brain, "direction": "in"},
                    {"layer_id": self.brain, "direction": "out"},
                    {"layer_id": self.cd4, "direction": "in"},
                    {"layer_id": self.cd4, "direction": "out"}]
        self.assertEqual(self.runtime.native_inputs(self.arguments()), tuple(expected))
        self.assertEqual(self.runtime.native_inputs(self.arguments(layer_scope=self.scope(self.cd4), direction="out")),
                         ({"layer_id": self.cd4, "direction": "out"},))

    def test_directed_roles_and_self_loop_remain_exact(self) -> None:
        for direction, expected in (("in", [self.a, self.b, self.c]), ("out", [self.a, self.b, self.d])):
            result = self.query(layer_scope=self.scope(self.brain), direction=direction)["result"]
            self.assertEqual([row["gene_id"] for row in result["neighbors"]], expected)
            self.assertEqual(result["neighbors"][0]["supporting_layer_count"], 1)
            self.assertEqual(len(result["neighbors"][0]["supporting_layers"][0]["edges"]), 1)
            for row in result["neighbors"]:
                edge = row["supporting_layers"][0]["edges"][0]
                self.assertEqual(edge["gene_ids"][1 if direction == "in" else 0], self.a)

    def test_present_zero_layer_absence_and_global_absence_are_distinct(self) -> None:
        zero = self.query(layer_scope=self.scope(self.cd4), direction="out")["result"]
        self.assertEqual(zero["neighbors"], [])
        self.assertEqual(zero["layers_without_gene"], [])
        self.assertEqual(zero["layer_results"][0]["not_in_layer_gene_ids"], [])
        absent = self.query(gene_id=self.e, layer_scope=self.scope(self.ppi))["result"]
        self.assertEqual(absent["neighbors"], [])
        self.assertEqual(absent["layers_without_gene"], [self.ppi])
        self.assertEqual(absent["layer_results"][0]["not_in_layer_gene_ids"], [self.e])
        for result in (zero, absent):
            self.assertEqual(result["unique_neighbor_count"], 0)
            self.assertEqual(result["collection"], {"mode": "complete", "returned_count": 0, "total_count": 0,
                                                   "complete": True, "next_cursor": None})
        missing = self.query(gene_id=self.missing)
        self.assertEqual(missing["status"], "not_found")
        result = missing["result"]
        self.assertEqual(result["missing_gene_ids"], [self.missing])
        for field in ("layer_results", "neighbors", "unique_neighbor_count", "layers_without_gene", "collection"):
            self.assertIsNone(result[field])
        self.assertEqual(result["coverage"]["checked_layer_ids"], [])
        self.assertEqual(result["coverage"]["unchecked_layer_ids"], list(self.view.layer_order))
        self.assertTrue(missing["provenance"]["graph"])

    def test_families_expand_exhaustively_and_empty_family_checks_membership(self) -> None:
        tf = {**self.all, "layer_family": "tf_target"}
        result = self.query(layer_scope=tf, direction="in")["result"]
        self.assertEqual([row["layer_id"] for row in result["layer_results"]], [self.brain, self.cd4])
        self.assertEqual(result["layer_scope"], tf)
        empty = {**self.all, "layer_family": "scpen"}
        result = self.query(layer_scope=empty, direction="in")["result"]
        self.assertEqual(result["neighbors"], [])
        self.assertEqual(result["layer_results"], [])
        self.assertEqual(result["unique_neighbor_count"], 0)
        self.assertEqual(result["coverage"], {"operation": "layer_neighbor_maps", "requested_layer_ids": [],
                                              "checked_layer_ids": [], "unchecked_layer_ids": []})
        missing = self.query(gene_id=self.missing, layer_scope=empty)
        self.assertEqual(missing["status"], "not_found")
        self.assertIsNone(missing["result"]["collection"])

    def test_each_transport_page_has_complete_scoped_support(self) -> None:
        full = self.query()["result"]
        for disclose in (True, False):
            runtime = NeighborLayersRuntime(self.view, page_size=1, disclose_total=disclose, index=self.index)
            pages = self.traverse(runtime, self.arguments())
            self.assertEqual(len(pages), 5)
            self.assertEqual([page["result"]["neighbors"][0] for page in pages], full["neighbors"])
            for page in pages:
                result = page["result"]
                self.assertEqual(result["coverage"], full["coverage"])
                self.assertEqual(result["layer_results"], full["layer_results"])
                self.assertFalse(result["collection"]["complete"])
                self.assertEqual(result["unique_neighbor_count"], 5 if disclose else None)
                self.assertEqual(result["collection"]["total_count"], 5 if disclose else None)

    def test_bounded_transport_does_not_extend_selection_or_expose_hidden_total(self) -> None:
        runtime = NeighborLayersRuntime(self.view, page_size=1, disclose_total=False, index=self.index)
        pages = self.traverse(runtime, self.arguments(mode="top_k", top_k=2))
        self.assertEqual([page["result"]["neighbors"][0]["gene_id"] for page in pages], [self.a, self.b])
        self.assertTrue(all(page["result"]["unique_neighbor_count"] is None for page in pages))
        self.assertTrue(all(page["result"]["collection"]["total_count"] is None for page in pages))
        self.assertTrue(all(not page["result"]["collection"]["complete"] for page in pages))

    def test_overlap_and_empty_pages_preserve_rows_and_support(self) -> None:
        runtime = NeighborLayersRuntime(self.view, page_size=2, overlap=1, empty_page_at=1, index=self.index)
        pages = self.traverse(runtime, self.arguments())
        self.assertEqual([len(page["result"]["neighbors"]) for page in pages], [2, 0, 2, 2, 2])
        expected = {row["gene_id"]: row for row in self.query()["result"]["neighbors"]}
        for page in pages:
            for row in page["result"]["neighbors"]:
                self.assertEqual(row, expected[row["gene_id"]])
        first_empty = NeighborLayersRuntime(self.view, empty_first_page=True, index=self.index)
        pages = self.traverse(first_empty, self.arguments())
        self.assertEqual([len(page["result"]["neighbors"]) for page in pages], [0, 5])
        self.assertTrue(pages[-1]["result"]["collection"]["complete"])

    def test_cursor_binds_every_argument_graph_version_and_transport(self) -> None:
        runtime = NeighborLayersRuntime(self.view, page_size=1, index=self.index)
        first = self.query(runtime)
        cursor = first["result"]["collection"]["next_cursor"]
        same = NeighborLayersRuntime(self.view, page_size=1, index=self.index)
        self.assertEqual(self.query(same)["result"], first["result"])
        for changes in ({"gene_id": self.b}, {"layer_scope": self.scope(self.ppi)},
                        {"mode": "top_k", "top_k": 2}, {"cursor": "invented"}):
            arguments = self.arguments(cursor=cursor)
            arguments.update(changes)
            response = runtime.get_neighbor_layers(**arguments, evidence_id="unused")
            self.assertEqual(response["result"]["issue_code"], "INVALID_CURSOR")
            self.assertEqual(response["provenance"], {"registry": [], "graph": []})
        variants = (NeighborLayersRuntime(self.view, page_size=2, index=self.index),
                    NeighborLayersRuntime(self.view, page_size=1, disclose_total=False, index=self.index),
                    NeighborLayersRuntime(GraphView(self.view.multiplex_id, "changed", self.view.layers), page_size=1))
        for changed in variants:
            self.assertEqual(self.query(changed, cursor=cursor)["result"]["issue_code"], "INVALID_CURSOR")

    def test_diagnostics_are_atomic_and_keep_the_exact_rejected_arguments(self) -> None:
        for changes, status, issue in (
            ({"multiplex_id": None}, "invalid_input", "MISSING_SCOPE"),
            ({"multiplex_id": "other"}, "not_found", "UNKNOWN_MULTIPLEX"),
            ({"gene_id": "MFRP"}, "invalid_input", "INVALID_GENE_INPUT"),
            ({"layer_scope": None}, "invalid_input", "MISSING_SCOPE"),
            ({"layer_scope": self.scope()}, "invalid_input", "INVALID_SELECTION"),
            ({"layer_scope": self.scope("TFs:unknown")}, "not_found", "UNKNOWN_LAYER"),
            ({"layer_scope": {**self.all, "layer_family": "unknown"}}, "invalid_input", "INVALID_SELECTION"),
            ({"direction": "in"}, "invalid_input", "DIRECTION_UNAVAILABLE"),
            ({"direction": "out"}, "invalid_input", "DIRECTION_UNAVAILABLE"),
            ({"direction": "undirected"}, "invalid_input", "DIRECTION_UNAVAILABLE"),
            ({"direction": "both"}, "invalid_input", "DIRECTION_UNAVAILABLE"),
            ({"mode": "top_k"}, "invalid_input", "MISSING_LIMIT"),
            ({"mode": "top_k", "top_k": True}, "invalid_input", "INVALID_SELECTION"),
            ({"mode": "top_k", "top_k": 0}, "invalid_input", "INVALID_SELECTION"),
            ({"top_k": 1}, "invalid_input", "INVALID_SELECTION"),
            ({"cursor": ""}, "invalid_input", "INVALID_CURSOR"),
            ({"cursor": []}, "invalid_input", "INVALID_CURSOR"),
        ):
            with self.subTest(changes=changes):
                arguments = self.arguments(**changes)
                response = self.runtime.get_neighbor_layers(**arguments, evidence_id="unused")
                self.assertEqual(response, {"status": status, "provenance": {"registry": [], "graph": []},
                                            "result": {"issue_code": issue, "arguments": arguments}})

    def test_all_scopes_and_directions_match_independent_source_edge_union(self) -> None:
        for size in range(1, len(self.view.layer_order) + 1):
            for layers in combinations(self.view.layer_order, size):
                for direction in ("incident", "in", "out", "undirected"):
                    if direction in ("in", "out") and any(not self.view.layers[layer].directed for layer in layers):
                        continue
                    if direction == "undirected" and any(self.view.layers[layer].directed for layer in layers):
                        continue
                    for gene in sorted(self.view.multiplex_nodes):
                        expected = {}
                        for layer_id in layers:
                            layer = self.view.layers[layer_id]
                            for pair, weight in layer.edges.items():
                                if gene not in pair:
                                    continue
                                if direction == "in" and pair[1] != gene:
                                    continue
                                if direction == "out" and pair[0] != gene:
                                    continue
                                neighbor = pair[1] if pair[0] == gene else pair[0]
                                expected.setdefault(neighbor, {})[layer_id, pair] = weight
                        with self.subTest(layers=layers, direction=direction, gene=gene):
                            result = self.query(gene_id=gene, layer_scope=self.scope(*layers), direction=direction)["result"]
                            self.assertEqual([row["gene_id"] for row in result["neighbors"]], sorted(expected))
                            self.assertEqual(result["unique_neighbor_count"], len(expected))
                            for row in result["neighbors"]:
                                actual = {(support["layer_id"], tuple(edge["gene_ids"])): edge["weight"]
                                          for support in row["supporting_layers"] for edge in support["edges"]}
                                self.assertEqual(actual, expected[row["gene_id"]])

    def test_transport_configuration_and_successful_evidence_are_explicit(self) -> None:
        for options in ({"page_size": True}, {"page_size": 0}, {"overlap": 1},
                        {"page_size": 1, "overlap": 1}, {"disclose_total": 1},
                        {"empty_page_at": -1}, {"empty_first_page": True, "empty_page_at": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                NeighborLayersRuntime(self.view, **options)
        other_view = GraphView(self.view.multiplex_id, self.view.graph_version, self.view.layers)
        with self.assertRaisesRegex(ValueError, "exact immutable view"):
            NeighborLayersRuntime(other_view, index=self.index)
        with self.assertRaisesRegex(ValueError, "evidence ID"):
            self.runtime.get_neighbor_layers(**self.arguments(), evidence_id=" ")
        with self.assertRaises(SelectionError):
            self.runtime.native_inputs(self.arguments(direction="in"))


if __name__ == "__main__":
    unittest.main()
