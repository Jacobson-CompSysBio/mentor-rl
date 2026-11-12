"""Exercise S2.2 native batch queries, saved proofs, and exact source inventory."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations, permutations
import unittest
from unittest.mock import patch

from runtime.world_model import s2_multiplex
from runtime.world_model.s2_multiplex import CandidateInventory, EvidenceError, MultiplexRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class S2MultiplexTests(unittest.TestCase):
    """Use an explicit graph with asymmetric arcs and disjoint layer membership."""

    def setUp(self) -> None:
        self.a, self.b, self.c, self.d, self.e, self.missing = [
            f"ENSG{n:011d}" for n in range(1, 7)
        ]
        self.ppi, self.coex, self.brain, self.cd4 = (
            "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells",
        )
        specs = [
            (self.ppi, False, {(self.a, self.b): 0, (self.c, self.d): 0.8}),
            (self.coex, False, {(self.a, self.c): 0.125}),
            (self.brain, True, {(self.a, self.b): 0.2, (self.b, self.a): 0.3,
                                (self.a, self.c): 0.4, (self.c, self.d): 0.7}),
            (self.cd4, True, {(self.a, self.b): 0.123456789012345,
                              (self.d, self.c): 0.9, (self.e, self.e): 1}),
        ]
        layers = {
            tag: LayerGraph(tag, directed, edges, frozenset(gene for pair in edges for gene in pair))
            for tag, directed, edges in specs
        }
        self.view = GraphView("unit_multiplex", "unit_graph_version", layers)
        self.runtime = MultiplexRuntime(self.view)
        self.all = {"kind": "all", "layer_ids": [], "layer_family": None}
        self.tf = {"kind": "all", "layer_ids": [], "layer_family": "tf_target"}

    def scope(self, *layers: str) -> dict:
        """Build an explicit scope so test intent is visible at each call."""
        return {"kind": "layers", "layer_ids": list(layers), "layer_family": None}

    def query(self, genes=None, scope=None, query="any_direction", mode="complete", k=None):
        """Acquire one fresh batch with a local evidence identifier."""
        return self.runtime.get_edge_layers(
            self.view.multiplex_id, genes or [self.a, self.b], scope or self.all,
            query, mode, k, evidence_id="ev_unit_batch",
        )

    def test_complete_retains_negative_layers_and_native_weights(self) -> None:
        response = self.query()
        result = response["result"]
        self.assertEqual(response["status"], "resolved")
        self.assertEqual(list(result), [
            "multiplex_id", "gene_ids", "layer_scope", "edge_query", "mode",
            "edge_exists_any_layer", "layer_results", "supporting_layer_ids",
            "supporting_layer_count", "missing_gene_ids", "coverage", "collection",
        ])
        self.assertEqual(result["supporting_layer_ids"], [self.ppi, self.brain, self.cd4])
        self.assertEqual(result["supporting_layer_count"], 3)
        self.assertEqual(result["coverage"]["checked_layer_ids"], list(self.view.layer_order))
        self.assertEqual(result["layer_results"][0]["edges"][0]["weight"], 0)
        self.assertEqual(result["layer_results"][1]["not_in_layer_gene_ids"], [self.b])
        self.assertEqual(result["layer_results"][3]["edges"][0]["weight"], 0.123456789012345)
        self.assertEqual(result["collection"], {
            "mode": "complete", "returned_count": 3, "total_count": 3,
            "complete": True, "next_cursor": None,
        })
        self.assertEqual(response["provenance"]["graph"][0]["layer_scope"], self.all)

    def test_family_grammar_cache_tracks_actual_contract_hash(self) -> None:
        contract = s2_multiplex.load_contract()
        s2_multiplex._family_contract.cache_clear()
        self.addCleanup(s2_multiplex._family_contract.cache_clear)
        with patch.object(s2_multiplex, "file_sha256", side_effect=["first", "first", "second"]), \
                patch.object(s2_multiplex, "load_contract", return_value=contract) as load:
            first = MultiplexRuntime(self.view)
            again = MultiplexRuntime(self.view)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(first.layer_families, again.layer_families)
            changed = MultiplexRuntime(self.view)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(changed.family_of(self.brain), "tf_target")

    def test_full_inner_coverage_even_after_a_qualifying_arc(self) -> None:
        result = self.query(scope=self.tf, mode="exists")["result"]
        self.assertEqual(result["supporting_layer_ids"], [self.brain])
        self.assertIsNone(result["supporting_layer_count"])
        self.assertIsNone(result["collection"])
        row = result["layer_results"][0]
        self.assertEqual(row["coverage"]["checked_pairs"], [[self.a, self.b], [self.b, self.a]])
        self.assertEqual(row["coverage"]["unchecked_pairs"], [])
        self.assertEqual([edge["weight"] for edge in row["edges"]], [0.2, 0.3])

    def test_both_directions_retains_single_positive_arc_when_predicate_false(self) -> None:
        result = self.query(scope=self.tf, query="both_directions")["result"]
        self.assertEqual(result["supporting_layer_ids"], [self.brain])
        row = result["layer_results"][1]
        self.assertFalse(row["edge_exists"])
        self.assertEqual(row["edges"], [{"gene_ids": [self.a, self.b], "weight": 0.123456789012345}])
        self.assertEqual(row["coverage"]["checked_pairs"], [[self.a, self.b], [self.b, self.a]])
        opposite_layers = self.query([self.c, self.d], self.tf, "both_directions")["result"]
        self.assertFalse(opposite_layers["edge_exists_any_layer"])
        self.assertEqual(opposite_layers["supporting_layer_count"], 0)
        self.assertEqual(self.query([self.c, self.d], self.tf)["result"]["supporting_layer_count"], 2)

    def test_reference_order_and_ordered_predicate(self) -> None:
        reverse = self.query([self.b, self.a])["result"]
        self.assertEqual(reverse["gene_ids"], [self.b, self.a])
        self.assertEqual(reverse["layer_results"][0]["edges"][0]["gene_ids"], [self.a, self.b])
        self.assertEqual([edge["gene_ids"] for edge in reverse["layer_results"][2]["edges"]],
                         [[self.b, self.a], [self.a, self.b]])
        source = self.query([self.b, self.a], self.tf, "source_to_target")["result"]
        self.assertEqual(source["supporting_layer_ids"], [self.brain])
        self.assertEqual(source["layer_results"][0]["coverage"]["requested_pairs"], [[self.b, self.a]])

    def test_bounded_scan_uses_manifest_prefix_and_checked_provenance(self) -> None:
        response = self.query(mode="top_k", k=2)
        result = response["result"]
        self.assertEqual(result["coverage"]["checked_layer_ids"], [self.ppi, self.coex, self.brain])
        self.assertEqual(result["coverage"]["unchecked_layer_ids"], [self.cd4])
        self.assertEqual(result["supporting_layer_ids"], [self.ppi, self.brain])
        self.assertIsNone(result["supporting_layer_count"])
        self.assertEqual(result["collection"], {
            "mode": "top_k", "returned_count": 2, "total_count": None,
            "complete": False, "next_cursor": None,
        })
        self.assertEqual(response["provenance"]["graph"][0]["layer_scope"],
                         self.scope(self.ppi, self.coex, self.brain))
        exhausted = self.query(mode="top_k", k=99)["result"]
        self.assertEqual(exhausted["supporting_layer_count"], 3)
        self.assertTrue(exhausted["collection"]["complete"])

    def test_explicit_scope_normalizes_duplicates_and_order(self) -> None:
        result = self.query(scope=self.scope(self.cd4, self.ppi, self.cd4))["result"]
        self.assertEqual(result["layer_scope"], self.scope(self.ppi, self.cd4))
        self.assertEqual(result["coverage"]["requested_layer_ids"], [self.ppi, self.cd4])
        self.assertEqual(result["supporting_layer_count"], 2)
        self.assertEqual(self.runtime.family_of(self.brain), "tf_target")

    def test_empty_family_requires_global_membership_before_zero(self) -> None:
        empty = {"kind": "all", "layer_ids": [], "layer_family": "scpen"}
        result = self.query(scope=empty, query="both_directions")["result"]
        self.assertFalse(result["edge_exists_any_layer"])
        self.assertEqual(result["layer_results"], [])
        self.assertEqual(result["supporting_layer_count"], 0)
        self.assertTrue(result["collection"]["complete"])
        missing = self.query([self.missing, self.a], empty, "source_to_target")
        self.assertEqual(missing["status"], "not_found")
        self.assertIsNone(missing["result"]["edge_exists_any_layer"])
        self.assertIsNone(missing["result"]["supporting_layer_count"])
        self.assertIsNone(missing["result"]["collection"])

    def test_missing_global_has_no_layer_predicates_or_counts(self) -> None:
        response = self.query([self.missing, self.e])
        result = response["result"]
        self.assertEqual(response["status"], "not_found")
        self.assertEqual(result["missing_gene_ids"], [self.missing])
        for key in ("edge_exists_any_layer", "layer_results", "supporting_layer_ids",
                    "supporting_layer_count", "collection"):
            self.assertIsNone(result[key])
        self.assertEqual(result["coverage"]["checked_layer_ids"], [])
        self.assertEqual(result["coverage"]["unchecked_layer_ids"], list(self.view.layer_order))
        self.assertEqual(response["provenance"]["graph"][0]["layer_scope"], self.all)
        outside = self.query([self.e, self.a], self.scope(self.ppi))["result"]
        self.assertFalse(outside["edge_exists_any_layer"])
        self.assertEqual(outside["layer_results"][0]["not_in_layer_gene_ids"], [self.e])

    def test_atomic_diagnostics_preserve_all_raw_arguments(self) -> None:
        base = {"multiplex_id": self.view.multiplex_id, "gene_ids": [self.a, self.b],
                "layer_scope": self.all, "edge_query": "any_direction", "mode": "complete", "top_k": None}
        cases = [
            ({"multiplex_id": None}, "invalid_input", "MISSING_SCOPE"),
            ({"multiplex_id": "other"}, "not_found", "UNKNOWN_MULTIPLEX"),
            ({"layer_scope": None}, "invalid_input", "MISSING_SCOPE"),
            ({"layer_scope": self.scope("TFs:missing")}, "not_found", "UNKNOWN_LAYER"),
            ({"layer_scope": self.scope()}, "invalid_input", "INVALID_SELECTION"),
            ({"layer_scope": {**self.all, "layer_family": "unknown"}}, "invalid_input", "INVALID_SELECTION"),
            ({"edge_query": "source_to_target"}, "invalid_input", "DIRECTION_UNAVAILABLE"),
            ({"edge_query": "reverse"}, "invalid_input", "INVALID_SELECTION"),
            ({"mode": "top_k", "top_k": True}, "invalid_input", "INVALID_SELECTION"),
            ({"mode": "top_k", "top_k": 0}, "invalid_input", "INVALID_SELECTION"),
            ({"mode": "exists", "top_k": 1}, "invalid_input", "INVALID_SELECTION"),
            ({"gene_ids": [self.a, self.a]}, "invalid_input", "INVALID_PAIR"),
            ({"gene_ids": [self.a, "TP53"]}, "invalid_input", "INVALID_GENE_INPUT"),
        ]
        for changes, status, code in cases:
            with self.subTest(changes=changes):
                arguments = {**base, **changes}
                response = self.runtime.get_edge_layers(**arguments, evidence_id="unused")
                self.assertEqual(response, {
                    "status": status, "provenance": {"registry": [], "graph": []},
                    "result": {"issue_code": code, "arguments": arguments},
                })

    def test_saved_positive_any_and_negative_both_preserve_unchecked_reverse(self) -> None:
        positive = self.view.get_edge(self.view.multiplex_id, self.cd4, [self.a, self.b], evidence_id="ev_forward")["result"]
        row = self.runtime.layer_from_edge(positive, "any_direction", [self.a, self.b])
        self.assertTrue(row["edge_exists"])
        self.assertEqual(row["coverage"]["checked_pairs"], [[self.a, self.b]])
        self.assertEqual(row["coverage"]["unchecked_pairs"], [[self.b, self.a]])
        result = self.runtime.assemble_result([self.a, self.b], self.all, "any_direction", "exists", None, [row])
        self.assertEqual(result["supporting_layer_ids"], [self.cd4])
        self.assertEqual(result["coverage"]["unchecked_layer_ids"], [self.ppi, self.coex, self.brain])
        self.assertIsNone(result["supporting_layer_count"])
        negative = self.view.get_edge(self.view.multiplex_id, self.cd4, [self.b, self.a], evidence_id="ev_reverse")["result"]
        row = self.runtime.layer_from_edge(negative, "both_directions", [self.a, self.b])
        self.assertFalse(row["edge_exists"])
        self.assertEqual(row["coverage"]["checked_pairs"], [[self.b, self.a]])
        self.assertEqual(row["coverage"]["unchecked_pairs"], [[self.a, self.b]])
        with self.assertRaises(EvidenceError):
            self.runtime.layer_from_edge(positive, "both_directions", [self.a, self.b])
        with self.assertRaises(EvidenceError):
            self.runtime.layer_from_edge(negative, "any_direction", [self.a, self.b])

    def test_saved_layer_absence_proves_both_orientations(self) -> None:
        saved = self.view.get_edge(self.view.multiplex_id, self.brain, [self.e, self.a], evidence_id="ev_absence")["result"]
        for predicate in ("any_direction", "source_to_target", "both_directions"):
            row = self.runtime.layer_from_edge(saved, predicate, [self.a, self.e])
            self.assertFalse(row["edge_exists"])
            self.assertEqual(row["coverage"]["unchecked_pairs"], [])
            self.assertEqual(row["not_in_layer_gene_ids"], [self.e])

    def test_assemble_unions_compatible_rows_and_rejects_conflicting_proofs(self) -> None:
        rows = self.query()["result"]["layer_results"]
        result = self.runtime.assemble_result([self.a, self.b], self.all, "any_direction", "complete", None,
                                              rows[::-1] + [rows[0]])
        self.assertEqual(result["supporting_layer_count"], 3)
        self.assertEqual(result["coverage"]["checked_layer_ids"], list(self.view.layer_order))
        conflict = deepcopy(rows[0])
        conflict["edges"][0]["weight"] = 0.5
        with self.assertRaisesRegex(EvidenceError, "contradictory"):
            self.runtime.assemble_result([self.a, self.b], self.all, "any_direction", "complete", None,
                                         rows + [conflict])
        with self.assertRaisesRegex(EvidenceError, "preceding"):
            self.runtime.assemble_result([self.a, self.b], self.all, "any_direction", "top_k", 1, [rows[-1]])
        with self.assertRaisesRegex(EvidenceError, "every selected"):
            self.runtime.assemble_result([self.a, self.b], self.all, "any_direction", "complete", None, [rows[-1]])
        rich_exists = self.runtime.assemble_result([self.a, self.b], self.all, "any_direction", "exists", None, rows)
        self.assertEqual(rich_exists["supporting_layer_count"], 3)
        self.assertEqual(len(rich_exists["layer_results"]), 4)
        self.assertIsNone(rich_exists["collection"])

    def test_bounded_projection_retains_richer_checked_evidence(self) -> None:
        rows = self.query()["result"]["layer_results"]
        full = self.runtime.assemble_result([self.a, self.b], self.all, "any_direction", "top_k", 1, rows)
        self.assertEqual(full["supporting_layer_ids"], [self.ppi])
        self.assertEqual(full["supporting_layer_count"], 3)
        self.assertEqual(full["layer_results"], rows)
        self.assertEqual(full["coverage"]["unchecked_layer_ids"], [])
        self.assertEqual(full["collection"], {
            "mode": "top_k", "returned_count": 1, "total_count": 3,
            "complete": False, "next_cursor": None,
        })
        partial = self.runtime.assemble_result([self.a, self.b], self.all, "any_direction", "top_k", 1,
                                               [rows[-1], rows[0]])
        self.assertEqual(partial["supporting_layer_ids"], [self.ppi])
        self.assertEqual(partial["coverage"]["checked_layer_ids"], [self.ppi, self.cd4])
        self.assertEqual(partial["coverage"]["unchecked_layer_ids"], [self.coex, self.brain])
        self.assertIsNone(partial["supporting_layer_count"])
        self.assertFalse(partial["collection"]["complete"])

    def test_saved_symmetric_reference_reversal_preserves_native_roles(self) -> None:
        for query, scope in (("any_direction", self.all), ("both_directions", self.tf)):
            saved = self.query(scope=scope, query=query)["result"]["layer_results"]
            result = self.runtime.assemble_result([self.b, self.a], scope, query, "complete", None, saved)
            self.assertEqual(result, self.query([self.b, self.a], scope, query)["result"])
            brain = next(row for row in result["layer_results"] if row["layer_id"] == self.brain)
            self.assertEqual(brain["edges"], [
                {"gene_ids": [self.b, self.a], "weight": 0.3},
                {"gene_ids": [self.a, self.b], "weight": 0.2},
            ])
        ordered = self.query(scope=self.tf, query="source_to_target")["result"]["layer_results"]
        with self.assertRaisesRegex(EvidenceError, "coverage differs"):
            self.runtime.assemble_result([self.b, self.a], self.tf, "source_to_target", "complete", None, ordered)

    def test_requery_complete_any_preserves_native_arcs_for_other_predicates(self) -> None:
        rows = self.query(scope=self.tf)["result"]["layer_results"]
        for query in ("source_to_target", "both_directions"):
            for pair in ([self.a, self.b], [self.b, self.a]):
                derived = [self.runtime.requery_layer(row, query, pair) for row in rows]
                expected = self.query(pair, self.tf, query)["result"]["layer_results"]
                self.assertEqual(derived, expected)

    def test_requery_single_orientation_requires_sufficient_native_evidence(self) -> None:
        scope = self.scope(self.cd4)
        positive = self.query(scope=scope, query="source_to_target")["result"]["layer_results"][0]
        negative = self.query([self.b, self.a], scope, "source_to_target")["result"]["layer_results"][0]
        any_row = self.runtime.requery_layer(positive, "any_direction", [self.a, self.b])
        self.assertTrue(any_row["edge_exists"])
        self.assertEqual(any_row["coverage"]["checked_pairs"], [[self.a, self.b]])
        self.assertEqual(any_row["coverage"]["unchecked_pairs"], [[self.b, self.a]])
        both_row = self.runtime.requery_layer(negative, "both_directions", [self.a, self.b])
        self.assertFalse(both_row["edge_exists"])
        self.assertEqual(both_row["coverage"]["checked_pairs"], [[self.b, self.a]])
        self.assertEqual(both_row["coverage"]["unchecked_pairs"], [[self.a, self.b]])
        for row, query, pair in (
            (positive, "both_directions", [self.a, self.b]),
            (negative, "any_direction", [self.a, self.b]),
            (positive, "source_to_target", [self.b, self.a]),
        ):
            with self.assertRaisesRegex(EvidenceError, "do not settle"):
                self.runtime.requery_layer(row, query, pair)

    def test_requery_layer_absence_and_structure_guards(self) -> None:
        absent = self.query([self.e, self.a], self.scope(self.brain), "source_to_target")["result"]["layer_results"][0]
        derived = self.runtime.requery_layer(absent, "any_direction", [self.a, self.e])
        self.assertFalse(derived["edge_exists"])
        self.assertEqual(derived["coverage"]["unchecked_pairs"], [])
        self.assertEqual(derived["not_in_layer_gene_ids"], [self.e])
        row = self.query(scope=self.scope(self.cd4))["result"]["layer_results"][0]
        for changes in (
            {"directed": False},
            {"not_in_layer_gene_ids": [self.a]},
            {"coverage": {**row["coverage"], "checked_pairs": [[self.a, self.b]], "unchecked_pairs": []}},
            {"edges": [{"gene_ids": [self.c, self.d], "weight": 0.1}]},
        ):
            with self.subTest(changes=changes), self.assertRaises(EvidenceError):
                self.runtime.requery_layer({**row, **changes}, "any_direction", [self.a, self.b])

    def test_inventory_all_subset_histograms_match_brute_force_native_queries(self) -> None:
        inventory = CandidateInventory(self.view)
        self.assertEqual(inventory.nodes, (self.a, self.b, self.c, self.d, self.e))
        self.assertEqual(inventory.pair_universe_size, 10)
        self.assertNotIn((self.e, self.e), inventory.pairs)
        self.assertEqual(sum(inventory.mask_frequencies.values()), len(inventory.pairs))
        for size in range(5):
            for layers in combinations(self.view.layer_order, size):
                predicates = ("any_direction",) if any(not self.view.layers[layer].directed for layer in layers) else (
                    "any_direction", "source_to_target", "both_directions",
                )
                for predicate in predicates:
                    with self.subTest(layers=layers, predicate=predicate):
                        pairs = (list(permutations(inventory.nodes, 2)) if predicate == "source_to_target"
                                 else list(combinations(inventory.nodes, 2)))
                        expected = Counter()
                        pools = {}
                        for pair in pairs:
                            count = 0
                            for layer_id in layers:
                                layer = self.view.layers[layer_id]
                                forward = pair in layer.edges if layer.directed else tuple(sorted(pair)) in layer.edges
                                reverse = pair[::-1] in layer.edges if layer.directed else forward
                                count += (forward or reverse) if predicate == "any_direction" else (
                                    forward and reverse if predicate == "both_directions" else forward
                                )
                            expected[count] += 1
                            pools.setdefault(count, set()).add(pair)
                            self.assertEqual(inventory.query_support(pair, layers, predicate), count)
                        self.assertEqual(inventory.histogram(layers, predicate), dict(sorted(expected.items())))
                        for count in range(1, len(layers) + 1):
                            actual = list(inventory.iter_candidates(layers, predicate, count))
                            self.assertEqual(set(actual), pools.get(count, set()))
                            self.assertEqual(len(actual), len(set(actual)))
        with self.assertRaisesRegex(ValueError, "missing multiplex"):
            inventory.query_support([self.a, self.missing], self.view.layer_order)
        with self.assertRaisesRegex(ValueError, "positive support"):
            list(inventory.iter_candidates(self.view.layer_order, "any_direction", 0))


if __name__ == "__main__":
    unittest.main()
