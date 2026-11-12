"""Check inventory membership, scoped ordering, and exact membership transport."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import unittest

from runtime.world_model.s2_gene_layers import ARGUMENT_KEYS, GeneLayersRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class S2GeneLayerTests(unittest.TestCase):
    """Use explicit node inventories with a target-only node and a layer isolate."""

    def setUp(self) -> None:
        self.a, self.b, self.c, self.d, self.e, self.f, self.missing, self.global_only = (
            f"ENSG{number:011d}" for number in range(1, 9)
        )
        self.ppi, self.coex, self.brain, self.cd4 = (
            "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells",
        )
        self.view = GraphView("unit_gene_layers", "unit_membership_version", {
            self.ppi: LayerGraph(self.ppi, False, {(self.a, self.b): .8}, frozenset((self.a, self.b, self.c))),
            self.coex: LayerGraph(self.coex, False, {(self.a, self.d): .4}, frozenset((self.a, self.d))),
            self.brain: LayerGraph(self.brain, True, {(self.e, self.a): .2}, frozenset((self.a, self.e))),
            self.cd4: LayerGraph(self.cd4, True, {(self.f, self.d): 1}, frozenset((self.d, self.f))),
        })
        self.runtime = GeneLayersRuntime(self.view)
        self.all = {"kind": "all", "layer_ids": [], "layer_family": None}

    def scope(self, *layers: str) -> dict:
        """Declare an explicit scope in its original mention order."""
        return {"kind": "layers", "layer_ids": list(layers), "layer_family": None}

    def arguments(self, **changes) -> dict:
        """Construct all six explicit acquisition fields for this fixture."""
        return {**self.runtime.arguments(self.a, self.all), **changes}

    def query(self, runtime=None, **changes) -> dict:
        """Acquire one membership record with an explicit test evidence identity."""
        return (runtime or self.runtime).get_gene_layers(**self.arguments(**changes), evidence_id="ev_membership")

    def traverse(self, runtime, arguments) -> list:
        """Follow returned opaque cursors without decoding their internal representation."""
        pages, pending = [], deepcopy(arguments)
        for number in range(20):
            response = runtime.get_gene_layers(**pending, evidence_id=f"ev_page_{number}")
            self.assertEqual(response["status"], "resolved")
            pages.append(response)
            cursor = response["result"]["collection"]["next_cursor"]
            if cursor is None:
                return pages
            pending["cursor"] = cursor
        self.fail("the finite membership fixture did not terminate")

    def test_fixed_result_shape_and_manifest_order(self) -> None:
        response = self.query()
        self.assertEqual(list(response), ["status", "provenance", "result"])
        result = response["result"]
        self.assertEqual(list(result), ["gene_id", "multiplex_id", "layer_scope", "present", "layer_ids",
                                       "layer_count", "coverage", "collection"])
        self.assertEqual(result["layer_ids"], [self.ppi, self.coex, self.brain])
        self.assertEqual(result["layer_count"], 3)
        self.assertIs(result["present"], True)
        self.assertEqual(result["coverage"], {"operation": "layer_membership",
            "requested_layer_ids": list(self.view.layer_order), "checked_layer_ids": list(self.view.layer_order),
            "unchecked_layer_ids": []})
        self.assertEqual(result["collection"], {"mode": "complete", "returned_count": 3,
            "total_count": 3, "complete": True, "next_cursor": None})
        self.assertNotIn("missing_gene_ids", result)
        self.assertNotIn("direction", result)

    def test_membership_uses_node_inventories_including_an_isolate_and_target_only_node(self) -> None:
        isolated = self.query(gene_id=self.c, layer_scope=self.scope(self.ppi))["result"]
        self.assertIs(isolated["present"], True)
        self.assertEqual(isolated["layer_ids"], [self.ppi])
        target_only = self.query(layer_scope=self.scope(self.brain))["result"]
        self.assertIs(target_only["present"], True)
        self.assertEqual(target_only["layer_ids"], [self.brain])
        self.assertFalse(any(source == self.a for source, _ in self.view.layers[self.brain].edges))

    def test_scope_absence_and_global_absence_are_both_resolved_but_distinct(self) -> None:
        for gene, present in ((self.c, True), (self.missing, False)):
            response = self.query(gene_id=gene, layer_scope={"kind": "all", "layer_ids": [], "layer_family": "tf_target"})
            self.assertEqual(response["status"], "resolved")
            result = response["result"]
            self.assertIs(result["present"], present)
            self.assertEqual(result["layer_ids"], [])
            self.assertEqual(result["layer_count"], 0)
            self.assertEqual(result["coverage"]["checked_layer_ids"], [self.brain, self.cd4])
            self.assertTrue(result["collection"]["complete"])

    def test_global_inventory_can_contain_a_node_without_layer_memberships(self) -> None:
        object.__setattr__(self.view, "multiplex_nodes", self.view.multiplex_nodes | {self.global_only})
        result = self.query(gene_id=self.global_only)["result"]
        self.assertIs(result["present"], True)
        self.assertEqual(result["layer_ids"], [])
        self.assertEqual(result["layer_count"], 0)

    def test_scope_normalization_preserves_original_arguments_without_mutation(self) -> None:
        arguments = self.arguments(layer_scope=self.scope(self.brain, self.ppi, self.brain))
        original = deepcopy(arguments)
        result = self.runtime.get_gene_layers(**arguments, evidence_id="ev_order")["result"]
        self.assertEqual(arguments, original)
        self.assertEqual(result["layer_scope"], self.scope(self.ppi, self.brain))
        self.assertEqual(result["layer_ids"], [self.ppi, self.brain])
        self.assertEqual(result["coverage"]["requested_layer_ids"], [self.ppi, self.brain])

    def test_first_k_is_a_manifest_prefix_and_count_remains_the_full_scoped_total(self) -> None:
        for limit, expected, complete in ((1, [self.ppi], False), (2, [self.ppi, self.coex], False),
                                          (3, [self.ppi, self.coex, self.brain], True),
                                          (7, [self.ppi, self.coex, self.brain], True)):
            result = self.query(mode="top_k", top_k=limit)["result"]
            self.assertEqual(result["layer_ids"], expected)
            self.assertEqual(result["layer_count"], 3)
            self.assertEqual(result["collection"]["total_count"], 3)
            self.assertEqual(result["collection"]["returned_count"], len(expected))
            self.assertIs(result["collection"]["complete"], complete)
            self.assertEqual(result["coverage"]["checked_layer_ids"], list(self.view.layer_order))

    def test_all_valid_families_expand_exhaustively_including_empty_selections(self) -> None:
        for family in self.runtime.multiplex.families:
            scope = {"kind": "all", "layer_ids": [], "layer_family": family}
            selected = [layer for layer in self.view.layer_order if self.runtime.multiplex.family_of(layer) == family]
            for gene in (self.a, self.missing):
                result = self.query(gene_id=gene, layer_scope=scope)["result"]
                expected = [layer for layer in selected if gene in self.view.layers[layer].nodes]
                self.assertEqual(result["layer_ids"], expected)
                self.assertEqual(result["layer_count"], len(expected))
                self.assertIs(result["present"], gene in self.view.multiplex_nodes)
                self.assertEqual(result["coverage"]["requested_layer_ids"], selected)

    def test_unknown_totals_stay_unknown_for_nonempty_responses(self) -> None:
        runtime = GeneLayersRuntime(self.view, disclose_total=False)
        result = self.query(runtime, mode="top_k", top_k=1)["result"]
        self.assertEqual(result["layer_ids"], [self.ppi])
        self.assertIsNone(result["layer_count"])
        self.assertIsNone(result["collection"]["total_count"])
        self.assertFalse(result["collection"]["complete"])
        for gene in (self.c, self.missing):
            empty = self.query(runtime, gene_id=gene, layer_scope=self.scope(self.cd4), mode="top_k", top_k=5)["result"]
            self.assertEqual(empty["layer_count"], 0)
            self.assertTrue(empty["collection"]["complete"])

    def test_page_boundary_is_separate_from_selection_coverage(self) -> None:
        runtime = GeneLayersRuntime(self.view, page_size=1)
        pages = self.traverse(runtime, self.arguments())
        self.assertEqual([page["result"]["layer_ids"] for page in pages], [[self.ppi], [self.coex], [self.brain]])
        self.assertTrue(all(not page["result"]["collection"]["complete"] for page in pages))
        self.assertIsNone(pages[-1]["result"]["collection"]["next_cursor"])
        for page in pages:
            self.assertEqual(page["result"]["coverage"]["checked_layer_ids"], list(self.view.layer_order))
            self.assertEqual(page["result"]["layer_count"], 3)

    def test_bounded_pages_return_the_bound_without_exposing_an_unknown_total(self) -> None:
        runtime = GeneLayersRuntime(self.view, page_size=1, disclose_total=False)
        pages = self.traverse(runtime, self.arguments(mode="top_k", top_k=2))
        self.assertEqual([layer for page in pages for layer in page["result"]["layer_ids"]], [self.ppi, self.coex])
        self.assertEqual(len(pages), 2)
        self.assertTrue(all(page["result"]["layer_count"] is None for page in pages))

    def test_overlapping_and_empty_pages_keep_the_same_selected_list(self) -> None:
        for transport in ({"page_size": 2, "overlap": 1}, {"page_size": 1, "empty_page_at": 1},
                          {"page_size": 1, "empty_page_at": 0}):
            runtime = GeneLayersRuntime(self.view, **transport)
            pages = self.traverse(runtime, self.arguments())
            layers = list(dict.fromkeys(layer for page in pages for layer in page["result"]["layer_ids"]))
            self.assertEqual(layers, [self.ppi, self.coex, self.brain])
            if "empty_page_at" in transport:
                self.assertTrue(any(not page["result"]["layer_ids"] and page["result"]["collection"]["next_cursor"] for page in pages))

    def test_cursor_binds_each_selection_field_graph_version_and_transport(self) -> None:
        runtime = GeneLayersRuntime(self.view, page_size=1)
        first = self.query(runtime)
        token = first["result"]["collection"]["next_cursor"]
        self.assertIsInstance(token, str)
        original = self.arguments(cursor=token)
        self.assertEqual(runtime.get_gene_layers(**original, evidence_id="ev_second")["result"]["layer_ids"], [self.coex])
        changes = ({"multiplex_id": "another_view"}, {"gene_id": self.d},
                   {"layer_scope": self.scope(self.ppi, self.coex, self.brain)},
                   {"mode": "top_k", "top_k": 2}, {"cursor": "invented_cursor"})
        for change in changes:
            response = runtime.get_gene_layers(**{**original, **change}, evidence_id="ev_rejected")
            self.assertNotEqual(response["status"], "resolved")
            self.assertEqual(response["provenance"], {"registry": [], "graph": []})
        other_transport = GeneLayersRuntime(self.view, page_size=2)
        self.assertEqual(other_transport.get_gene_layers(**original, evidence_id="ev_other")["result"]["issue_code"], "INVALID_CURSOR")
        other_view = GraphView(self.view.multiplex_id, "different_version", self.view.layers)
        other_version = GeneLayersRuntime(other_view, page_size=1)
        self.assertEqual(other_version.get_gene_layers(**original, evidence_id="ev_version")["result"]["issue_code"], "INVALID_CURSOR")

    def test_local_diagnostics_are_atomic_and_preserve_the_exact_rejected_request(self) -> None:
        cases = [({"layer_scope": None}, "invalid_input", "MISSING_SCOPE"),
                 ({"multiplex_id": "other"}, "not_found", "UNKNOWN_MULTIPLEX"),
                 ({"gene_id": "A"}, "invalid_input", "INVALID_GENE_INPUT"),
                 ({"layer_scope": self.scope("HumanNetV3:unregistered")}, "not_found", "UNKNOWN_LAYER"),
                 ({"layer_scope": self.scope()}, "invalid_input", "INVALID_SELECTION"),
                 ({"layer_scope": {"kind": "all", "layer_ids": [], "layer_family": "UNKNOWN"}}, "invalid_input", "INVALID_LAYER_FAMILY"),
                 ({"mode": "top_k"}, "invalid_input", "MISSING_LIMIT"),
                 ({"mode": "top_k", "top_k": 0}, "invalid_input", "INVALID_SELECTION"),
                 ({"mode": "top_k", "top_k": True}, "invalid_input", "INVALID_SELECTION"),
                 ({"mode": "complete", "top_k": 1}, "invalid_input", "INVALID_SELECTION"),
                 ({"cursor": ""}, "invalid_input", "INVALID_CURSOR")]
        for changes, status, issue in cases:
            arguments = self.arguments(**changes)
            response = self.runtime.get_gene_layers(**arguments, evidence_id="")
            self.assertEqual(response, {"status": status, "provenance": {"registry": [], "graph": []},
                "result": {"issue_code": issue, "arguments": arguments}})

    def test_all_scope_combinations_match_independent_inventory_membership(self) -> None:
        scopes = [self.all]
        for size in range(1, len(self.view.layers) + 1):
            scopes.extend(self.scope(*layers) for layers in combinations(self.view.layer_order, size))
        for scope in scopes:
            selected = list(self.view.layer_order) if scope["kind"] == "all" else scope["layer_ids"]
            for gene in (*sorted(self.view.multiplex_nodes), self.missing):
                expected = [layer for layer in selected if gene in self.view.layers[layer].nodes]
                for mode, limit in (("complete", None), ("top_k", 1), ("top_k", len(selected) + 1)):
                    result = self.query(gene_id=gene, layer_scope=scope, mode=mode, top_k=limit)["result"]
                    self.assertEqual(result["layer_ids"], expected if limit is None else expected[:limit])
                    self.assertEqual(result["layer_count"], len(expected))
                    self.assertIs(result["present"], gene in self.view.multiplex_nodes)

    def test_transport_and_successful_evidence_require_explicit_valid_values(self) -> None:
        for changes in ({"page_size": 0}, {"page_size": True}, {"disclose_total": 1}, {"overlap": -1},
                        {"overlap": 1}, {"page_size": 1, "overlap": 1}, {"empty_page_at": -1}, {"empty_page_at": True}):
            with self.assertRaises(ValueError):
                GeneLayersRuntime(self.view, **changes)
        for identity in (None, "", " "):
            with self.assertRaises(ValueError):
                self.runtime.get_gene_layers(**self.arguments(), evidence_id=identity)
        self.assertEqual(tuple(self.arguments()), ARGUMENT_KEYS)


if __name__ == "__main__":
    unittest.main()
