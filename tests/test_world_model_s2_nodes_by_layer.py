"""Check complete finite membership batches against authoritative node sets."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import unittest

from runtime.world_model.s2_multiplex import SelectionError
from runtime.world_model.s2_nodes_by_layer import ARGUMENT_KEYS, NodesByLayerRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class S2NodesByLayerTests(unittest.TestCase):
    """Use explicit inventories with layer isolates and target-only nodes."""

    def setUp(self) -> None:
        self.a, self.b, self.c, self.d, self.e, self.f, self.missing, self.global_only = (
            f"ENSG{number:011d}" for number in range(1, 9)
        )
        self.ppi, self.coex, self.brain, self.cd4 = (
            "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells",
        )
        self.view = GraphView("unit_nodes_by_layer", "unit_membership_version", {
            self.ppi: LayerGraph(self.ppi, False, {(self.a, self.b): .8}, frozenset((self.a, self.b, self.c))),
            self.coex: LayerGraph(self.coex, False, {(self.a, self.d): .4}, frozenset((self.a, self.d))),
            self.brain: LayerGraph(self.brain, True, {(self.e, self.a): .2}, frozenset((self.a, self.e))),
            self.cd4: LayerGraph(self.cd4, True, {(self.f, self.d): 1}, frozenset((self.d, self.f))),
        })
        self.runtime = NodesByLayerRuntime(self.view)

    def arguments(self, **changes) -> dict:
        """Construct the exact three-field request for this fixture."""
        return {**self.runtime.arguments(self.brain, [self.c, self.a, self.missing, self.e]), **changes}

    def query(self, **changes) -> dict:
        """Acquire one complete classification with an explicit evidence ID."""
        return self.runtime.get_nodes_by_layer(**self.arguments(**changes), evidence_id="ev_membership")

    def test_fixed_result_shape_and_exact_three_arguments(self) -> None:
        response = self.query()
        self.assertEqual(tuple(self.arguments()), ARGUMENT_KEYS)
        self.assertEqual(list(response), ["status", "provenance", "result"])
        result = response["result"]
        self.assertEqual(list(result), ["multiplex_id", "layer_id", "layer_gene_ids", "layer_gene_count",
                                       "not_in_layer_gene_ids", "missing_gene_ids", "coverage", "collection"])
        self.assertEqual(response["status"], "resolved")
        self.assertEqual(result["layer_gene_ids"], [self.a, self.e])
        self.assertEqual(result["not_in_layer_gene_ids"], [self.c])
        self.assertEqual(result["missing_gene_ids"], [self.missing])
        self.assertEqual(result["layer_gene_count"], 2)
        self.assertEqual(result["collection"], {"mode": "complete", "returned_count": 2,
            "total_count": 2, "complete": True, "next_cursor": None})
        self.assertEqual(response["provenance"], {"registry": [], "graph": [{
            "multiplex_id": self.view.multiplex_id, "graph_version": self.view.graph_version,
            "layer_scope": {"kind": "layers", "layer_ids": [self.brain], "layer_family": None},
            "evidence_id": "ev_membership"}]})

    def test_membership_uses_inventory_for_an_isolate_and_target_only_node(self) -> None:
        isolated = self.query(layer_id=self.ppi, gene_ids=[self.c])["result"]
        self.assertEqual(isolated["layer_gene_ids"], [self.c])
        self.assertFalse(any(self.c in pair for pair in self.view.layers[self.ppi].edges))
        target_only = self.query(gene_ids=[self.a])["result"]
        self.assertEqual(target_only["layer_gene_ids"], [self.a])
        self.assertFalse(any(source == self.a for source, _ in self.view.layers[self.brain].edges))

    def test_global_inventory_can_retain_a_multiplex_only_isolate(self) -> None:
        object.__setattr__(self.view, "multiplex_nodes", self.view.multiplex_nodes | {self.global_only})
        result = self.query(gene_ids=[self.global_only])["result"]
        self.assertEqual(result["not_in_layer_gene_ids"], [self.global_only])
        self.assertEqual(result["missing_gene_ids"], [])

    def test_global_absence_is_a_checked_negative_and_never_a_layer_negative(self) -> None:
        result = self.query(gene_ids=[self.missing])["result"]
        self.assertEqual(result["layer_gene_ids"], [])
        self.assertEqual(result["not_in_layer_gene_ids"], [])
        self.assertEqual(result["missing_gene_ids"], [self.missing])
        self.assertEqual(result["coverage"], {"operation": "layer_membership",
            "requested_gene_ids": [self.missing], "checked_gene_ids": [self.missing], "unchecked_gene_ids": []})

    def test_duplicate_input_order_is_separate_from_sorted_category_order(self) -> None:
        genes = [self.e, self.c, self.missing, self.a, self.e, self.c, self.a, self.missing]
        result = self.query(gene_ids=genes)["result"]
        self.assertEqual(result["coverage"]["requested_gene_ids"], [self.e, self.c, self.missing, self.a])
        self.assertEqual(result["coverage"]["checked_gene_ids"], [self.e, self.c, self.missing, self.a])
        self.assertEqual(result["layer_gene_ids"], [self.a, self.e])
        self.assertEqual(result["not_in_layer_gene_ids"], [self.c])
        self.assertEqual(result["missing_gene_ids"], [self.missing])
        self.assertEqual(genes, [self.e, self.c, self.missing, self.a, self.e, self.c, self.a, self.missing])

    def test_positive_counts_cover_only_the_supplied_genes(self) -> None:
        result = self.query(layer_id=self.ppi, gene_ids=[self.b])["result"]
        self.assertEqual(result["layer_gene_count"], 1)
        self.assertNotEqual(result["layer_gene_count"], len(self.view.layers[self.ppi].nodes))
        self.assertEqual(result["collection"]["returned_count"], 1)
        self.assertEqual(result["collection"]["total_count"], 1)

    def test_all_negative_partition_is_complete_and_resolved(self) -> None:
        result = self.query(gene_ids=[self.c, self.missing])["result"]
        self.assertEqual(result["layer_gene_count"], 0)
        self.assertEqual(result["not_in_layer_gene_ids"], [self.c])
        self.assertEqual(result["missing_gene_ids"], [self.missing])
        self.assertTrue(result["collection"]["complete"])
        self.assertEqual(result["coverage"]["checked_gene_ids"], [self.c, self.missing])

    def test_empty_list_is_valid_after_scope_validation_and_has_no_evidence(self) -> None:
        response = self.runtime.get_nodes_by_layer(**self.arguments(gene_ids=[]), evidence_id="")
        self.assertEqual(response["status"], "resolved")
        self.assertEqual(response["provenance"], {"registry": [], "graph": []})
        result = response["result"]
        for key in ("layer_gene_ids", "not_in_layer_gene_ids", "missing_gene_ids"):
            self.assertEqual(result[key], [])
        self.assertEqual(result["layer_gene_count"], 0)
        self.assertEqual(result["coverage"], {"operation": "layer_membership",
            "requested_gene_ids": [], "checked_gene_ids": [], "unchecked_gene_ids": []})
        self.assertEqual(self.query(layer_id="TFs:unregistered", gene_ids=[])["status"], "not_found")

    def test_argument_helper_copies_the_supplied_json_without_normalization(self) -> None:
        genes = [self.a, self.a, self.c]
        arguments = self.runtime.arguments(self.brain, genes)
        genes.append(self.b)
        self.assertEqual(arguments["gene_ids"], [self.a, self.a, self.c])
        self.assertEqual(self.runtime._validate(arguments), (self.a, self.c))

    def test_diagnostic_rejection_is_atomic_and_keeps_the_exact_supplied_json(self) -> None:
        cases = [({"multiplex_id": None}, "invalid_input", "MISSING_SCOPE"),
                 ({"layer_id": None}, "invalid_input", "MISSING_SCOPE"),
                 ({"multiplex_id": "other"}, "not_found", "UNKNOWN_MULTIPLEX"),
                 ({"layer_id": "TFs:unregistered"}, "not_found", "UNKNOWN_LAYER"),
                 ({"layer_id": {"kind": "all", "layer_ids": [], "layer_family": None}}, "not_found", "UNKNOWN_LAYER"),
                 ({"gene_ids": None}, "invalid_input", "MISSING_GENES"),
                 ({"gene_ids": self.a}, "invalid_input", "INVALID_GENE_LIST"),
                 ({"gene_ids": (self.a,)}, "invalid_input", "INVALID_GENE_LIST"),
                 ({"gene_ids": [self.e, self.e, "NFYA", self.a]}, "invalid_input", "INVALID_GENE_LIST"),
                 ({"gene_ids": [self.a, None]}, "invalid_input", "INVALID_GENE_LIST"),
                 ({"gene_ids": [self.a, [self.b]]}, "invalid_input", "INVALID_GENE_LIST")]
        for changes, status, issue in cases:
            arguments = self.arguments(**changes)
            original = deepcopy(arguments)
            response = self.runtime.get_nodes_by_layer(**arguments, evidence_id="")
            self.assertEqual(response, {"status": status, "provenance": {"registry": [], "graph": []},
                "result": {"issue_code": issue, "arguments": original}})
            self.assertEqual(arguments, original)
            self.assertNotIn("coverage", response["result"])

    def test_each_finite_subset_matches_an_independent_disjoint_partition(self) -> None:
        genes = [self.a, self.b, self.c, self.d, self.e, self.f, self.missing]
        for size in range(len(genes) + 1):
            for subset in combinations(genes, size):
                request = list(reversed(subset))
                for layer in self.view.layer_order:
                    result = self.query(layer_id=layer, gene_ids=request)["result"]
                    positive = set(request) & self.view.layers[layer].nodes
                    missing = set(request) - self.view.multiplex_nodes
                    outside = set(request) - positive - missing
                    self.assertEqual(result["layer_gene_ids"], sorted(positive))
                    self.assertEqual(result["not_in_layer_gene_ids"], sorted(outside))
                    self.assertEqual(result["missing_gene_ids"], sorted(missing))
                    self.assertEqual(result["coverage"]["checked_gene_ids"], request)
                    self.assertEqual(result["coverage"]["unchecked_gene_ids"], [])

    def test_missing_argument_keys_keep_their_prerequisite_diagnostics(self) -> None:
        for key, issue in (("multiplex_id", "MISSING_SCOPE"), ("layer_id", "MISSING_SCOPE"),
                           ("gene_ids", "MISSING_GENES")):
            arguments = self.arguments()
            del arguments[key]
            with self.assertRaises(SelectionError) as caught:
                self.runtime._validate(arguments)
            self.assertEqual(caught.exception.status, "invalid_input")
            self.assertEqual(caught.exception.issue_code, issue)
        with self.assertRaises(SelectionError) as caught:
            self.runtime._validate({**self.arguments(), "cursor": None})
        self.assertEqual(caught.exception.issue_code, "INVALID_GENE_LIST")

    def test_large_finite_list_has_no_implicit_limit_or_cursor(self) -> None:
        genes = [f"ENSG{number:011d}" for number in range(1, 5001)]
        result = self.query(gene_ids=genes)["result"]
        self.assertEqual(result["coverage"]["checked_gene_ids"], genes)
        self.assertEqual(len(result["layer_gene_ids"]) + len(result["not_in_layer_gene_ids"])
                         + len(result["missing_gene_ids"]), 5000)
        self.assertIsNone(result["collection"]["next_cursor"])

    def test_nonempty_acquisition_requires_an_evidence_id(self) -> None:
        for evidence_id in (None, "", " "):
            with self.assertRaises(ValueError):
                self.runtime.get_nodes_by_layer(**self.arguments(), evidence_id=evidence_id)


if __name__ == "__main__":
    unittest.main()
