"""Check native neighbor counts, explicit transport, and evidence-only reuse."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from runtime.world_model.s2_multiplex import EvidenceError
from runtime.world_model.s2_neighbors import NeighborIndex, NeighborRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class S2NeighborTests(unittest.TestCase):
    """Use real-shaped explicit fixtures with ties, zeros, loops, and asymmetric arcs."""

    def setUp(self) -> None:
        self.a, self.b, self.c, self.d, self.e, self.missing = [f"ENSG{n:011d}" for n in range(1, 7)]
        self.ppi, self.tf = "HumanNetV3:string_ppi", "TFs:brain"
        specs = {
            self.ppi: (False, {(self.a, self.a): 0.8, (self.a, self.b): 0.5,
                               (self.a, self.c): 0.5, (self.a, self.d): 0}),
            self.tf: (True, {(self.a, self.a): 1, (self.a, self.b): 0.3,
                             (self.b, self.a): 0.4, (self.a, self.c): 0.2,
                             (self.c, self.a): 0.9, (self.b, self.d): 0.7,
                             (self.e, self.b): 0.6}),
        }
        layers = {name: LayerGraph(name, directed, edges, frozenset(gene for pair in edges for gene in pair))
                  for name, (directed, edges) in specs.items()}
        self.view = GraphView("unit_neighbors", "unit_neighbor_version", layers)
        self.index = NeighborIndex(self.view)
        self.runtime = NeighborRuntime(self.view, index=self.index)

    def arguments(self, **changes) -> dict:
        """Supply the exact seven arguments, including initial cursor:null."""
        return {"multiplex_id": self.view.multiplex_id, "layer_id": self.ppi,
                "gene_id": self.a, "direction": "undirected", "mode": "complete",
                "top_k": None, "cursor": None, **changes}

    def observe(self, runtime, arguments, evidence_id="ev_test") -> dict:
        """Acquire a page and retain its unchanged operation attribution."""
        return {"operation": "get_neighbors", "arguments": deepcopy(arguments),
                **runtime.get_neighbors(**arguments, evidence_id=evidence_id)}

    def traverse(self, runtime, arguments) -> tuple[list, object]:
        """Follow only the next arguments supplied by the visible-page helper."""
        observations = []
        for index in range(20):
            proof = runtime.collect(arguments, observations)
            if proof.result is not None:
                return observations, proof
            observations.append(self.observe(runtime, proof.next_arguments, f"ev_page_{index}"))
        self.fail("finite native fixture did not finish traversal")

    def test_native_order_weights_direction_and_loop_cardinality(self) -> None:
        response = self.runtime.get_neighbors(**self.arguments(), evidence_id="ev_native")
        result = response["result"]
        self.assertEqual(list(result), ["gene_id", "multiplex_id", "layer_id", "direction", "neighbors",
                                        "missing_gene_ids", "not_in_layer_gene_ids", "collection"])
        self.assertEqual(result["neighbors"], [
            {"gene_id": self.a, "weight": 0.8}, {"gene_id": self.b, "weight": 0.5},
            {"gene_id": self.c, "weight": 0.5}, {"gene_id": self.d, "weight": 0},
        ])
        self.assertEqual(self.index.histogram(self.ppi, "undirected"), {1: 3, 4: 1})
        self.assertEqual(sum(n * count for n, count in self.index.histogram(self.ppi, "undirected").items()), 7)
        for direction in ("in", "out"):
            self.assertEqual(sum(n * count for n, count in self.index.histogram(self.tf, direction).items()), 7)
        incoming = self.index.neighbors(self.tf, self.a, "in")
        outgoing = self.index.neighbors(self.tf, self.a, "out")
        self.assertEqual(incoming, ((self.a, 1), (self.c, 0.9), (self.b, 0.4)))
        self.assertEqual(outgoing, ((self.a, 1), (self.b, 0.3), (self.c, 0.2)))

    def test_membership_absence_differs_from_member_zero_and_global_missing(self) -> None:
        zero = self.runtime.get_neighbors(**self.arguments(layer_id=self.tf, gene_id=self.d, direction="out"), evidence_id="ev_zero")
        self.assertEqual(zero["result"]["neighbors"], [])
        self.assertEqual(zero["result"]["not_in_layer_gene_ids"], [])
        absent = self.runtime.get_neighbors(**self.arguments(gene_id=self.e), evidence_id="ev_absent")
        self.assertEqual(absent["status"], "resolved")
        self.assertEqual(absent["result"]["neighbors"], [])
        self.assertEqual(absent["result"]["not_in_layer_gene_ids"], [self.e])
        self.assertEqual(absent["result"]["collection"]["total_count"], 0)
        missing = self.observe(self.runtime, self.arguments(gene_id=self.missing))
        self.assertEqual(missing["status"], "not_found")
        self.assertIsNone(missing["result"]["neighbors"])
        self.assertIsNone(missing["result"]["collection"])
        proof = self.runtime.collect(self.arguments(gene_id=self.missing), [missing])
        self.assertEqual(proof.result, missing["result"])
        self.assertIsNone(proof.next_arguments)

    def test_exact_tie_cutoff_and_no_default_neighbor_cap(self) -> None:
        bound = self.runtime.get_neighbors(**self.arguments(mode="top_k", top_k=2), evidence_id="ev_tie")["result"]
        self.assertEqual([row["gene_id"] for row in bound["neighbors"]], [self.a, self.b])
        self.assertEqual(bound["collection"], {"mode": "top_k", "returned_count": 2, "total_count": 4,
                                               "complete": False, "next_cursor": None})
        more = self.runtime.get_neighbors(**self.arguments(mode="top_k", top_k=100), evidence_id="ev_large")["result"]
        self.assertEqual(len(more["neighbors"]), 4)
        self.assertTrue(more["collection"]["complete"])
        self.assertEqual(len(self.runtime.get_neighbors(**self.arguments(), evidence_id="ev_all")["result"]["neighbors"]), 4)

    def test_known_and_unknown_total_complete_pagination(self) -> None:
        for disclose in (True, False):
            runtime = NeighborRuntime(self.view, page_size=1, disclose_total=disclose, index=self.index)
            observations, proof = self.traverse(runtime, self.arguments())
            self.assertEqual(len(observations), 4)
            self.assertTrue(all(not row["result"]["collection"]["complete"] for row in observations))
            self.assertIsNone(observations[-1]["result"]["collection"]["next_cursor"])
            self.assertEqual(proof.result["collection"], {
                "mode": "complete", "returned_count": 4, "total_count": 4, "complete": True, "next_cursor": None,
            })
            self.assertEqual(len(proof.provenance["graph"]), 4)
            if not disclose:
                self.assertTrue(all(row["result"]["collection"]["total_count"] is None for row in observations))

    def test_unknown_total_bounded_exhaustion_preserves_what_is_unproved(self) -> None:
        runtime = NeighborRuntime(self.view, page_size=2, disclose_total=False, index=self.index)
        for limit, expected_total, complete in ((2, None, False), (4, None, False), (5, 4, True)):
            _, proof = self.traverse(runtime, self.arguments(mode="top_k", top_k=limit))
            self.assertEqual(proof.result["collection"]["total_count"], expected_total)
            self.assertEqual(proof.result["collection"]["complete"], complete)
            self.assertEqual(len(proof.result["neighbors"]), min(limit, 4))

    def test_boundary_overlap_and_empty_intermediate_page(self) -> None:
        runtime = NeighborRuntime(self.view, page_size=2, overlap=1, empty_page_at=1,
                                  disclose_total=False, index=self.index)
        observations, proof = self.traverse(runtime, self.arguments())
        self.assertEqual([len(page["result"]["neighbors"]) for page in observations], [2, 0, 2, 2])
        self.assertEqual(len(proof.result["neighbors"]), 4)
        self.assertEqual(len({page["arguments"]["cursor"] for page in observations}), 4)
        first_empty = NeighborRuntime(self.view, empty_first_page=True, index=self.index)
        pages, proof = self.traverse(first_empty, self.arguments())
        self.assertEqual([len(page["result"]["neighbors"]) for page in pages], [0, 4])
        self.assertTrue(proof.result["collection"]["complete"])

    def test_cursor_is_deterministic_and_bound_to_every_selection_and_transport(self) -> None:
        runtime = NeighborRuntime(self.view, page_size=1, index=self.index)
        first = runtime.get_neighbors(**self.arguments(), evidence_id="ev_first")
        cursor = first["result"]["collection"]["next_cursor"]
        same = NeighborRuntime(self.view, page_size=1, index=self.index)
        self.assertEqual(same.get_neighbors(**self.arguments(), evidence_id="ev_other")["result"], first["result"])
        for changes in ({"gene_id": self.b}, {"mode": "top_k", "top_k": 2}, {"cursor": "invented"}):
            args = self.arguments(cursor=cursor, **{key: value for key, value in changes.items() if key != "cursor"})
            if "cursor" in changes:
                args["cursor"] = changes["cursor"]
            response = runtime.get_neighbors(**args, evidence_id="unused")
            self.assertEqual(response["result"]["issue_code"], "INVALID_CURSOR")
        for changed in (NeighborRuntime(self.view, page_size=2, index=self.index),
                        NeighborRuntime(self.view, page_size=1, disclose_total=False, index=self.index),
                        NeighborRuntime(GraphView(self.view.multiplex_id, "new_version", self.view.layers), page_size=1)):
            response = changed.get_neighbors(**self.arguments(cursor=cursor), evidence_id="unused")
            self.assertEqual(response["result"]["issue_code"], "INVALID_CURSOR")

    def test_saved_complete_and_sufficient_prefix_project_without_new_calls(self) -> None:
        full = self.observe(self.runtime, self.arguments(), "ev_full")
        proof = self.runtime.collect(self.arguments(mode="top_k", top_k=2), [full])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.result["collection"]["total_count"], 4)
        self.assertFalse(proof.result["collection"]["complete"])
        self.assertEqual(proof.provenance, full["provenance"])
        prefix = self.observe(self.runtime, self.arguments(mode="top_k", top_k=3), "ev_prefix")
        for limit in (2, 3):
            proof = self.runtime.collect(self.arguments(mode="top_k", top_k=limit), [prefix])
            self.assertEqual(len(proof.result["neighbors"]), limit)
            self.assertIsNone(proof.next_arguments)

    def test_changed_selection_restarts_null_but_retains_supported_prefix(self) -> None:
        paged = NeighborRuntime(self.view, page_size=1, index=self.index)
        saved = self.observe(paged, self.arguments(mode="top_k", top_k=2))
        proof = paged.collect(self.arguments(), [saved])
        self.assertEqual(proof.next_arguments, self.arguments())
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.provenance, saved["provenance"])
        same = paged.collect(self.arguments(mode="top_k", top_k=2), [saved])
        self.assertEqual(same.next_arguments["cursor"], saved["result"]["collection"]["next_cursor"])
        complete_prefix = self.observe(paged, self.arguments(), "ev_complete_prefix")
        continuation = paged.collect(self.arguments(), [complete_prefix])
        self.assertEqual(continuation.next_arguments["cursor"], complete_prefix["result"]["collection"]["next_cursor"])

    def test_collection_uses_visible_pages_without_reading_the_native_index(self) -> None:
        runtime = NeighborRuntime(self.view, page_size=2, disclose_total=False, index=self.index)
        initial = self.observe(runtime, self.arguments(), "ev_prefix")
        with patch.object(self.index, "neighbors", side_effect=AssertionError("hidden adjacency read")), \
                patch.object(self.index, "histogram", side_effect=AssertionError("hidden count read")):
            bounded = runtime.collect(self.arguments(mode="top_k", top_k=2), [initial])
            self.assertEqual(len(bounded.result["neighbors"]), 2)
            self.assertIsNone(bounded.result["collection"]["total_count"])
            pending = runtime.collect(self.arguments(), [initial])
            self.assertIsNone(pending.result)
            self.assertEqual(pending.next_arguments["cursor"], initial["result"]["collection"]["next_cursor"])

    def test_conflicting_acquisition_ids_and_cross_selection_totals_are_rejected(self) -> None:
        full = self.observe(self.runtime, self.arguments(), "ev_same")
        bound = self.observe(self.runtime, self.arguments(mode="top_k", top_k=2), "ev_same")
        with self.assertRaisesRegex(EvidenceError, "evidence ID"):
            self.runtime.collect(self.arguments(), [full, bound])
        bound["provenance"]["graph"][0]["evidence_id"] = "ev_bound"
        bound["result"]["collection"]["total_count"] = 5
        with self.assertRaisesRegex(EvidenceError, "totals"):
            self.runtime.collect(self.arguments(), [full, bound])

    def test_wrong_direction_layer_or_version_cannot_supply_adjacency(self) -> None:
        target = self.arguments(layer_id=self.tf, direction="out")
        incoming = self.observe(self.runtime, {**target, "direction": "in"})
        other_layer = self.observe(self.runtime, self.arguments())
        stale = deepcopy(self.observe(self.runtime, target))
        stale["provenance"]["graph"][0]["graph_version"] = "stale"
        for saved in (incoming, other_layer, stale):
            proof = self.runtime.collect(target, [saved])
            self.assertIsNone(proof.result)
            self.assertEqual(proof.next_arguments, target)
            self.assertEqual(proof.observations, ())

    def test_faults_reject_bad_chains_totals_ordering_and_conflicting_weights(self) -> None:
        runtime = NeighborRuntime(self.view, page_size=2, overlap=1, index=self.index)
        pages, _ = self.traverse(runtime, self.arguments())
        with self.assertRaisesRegex(EvidenceError, "initial"):
            runtime.collect(self.arguments(), pages[1:])
        bad = deepcopy(pages)
        bad[1]["result"]["collection"]["total_count"] = 99
        with self.assertRaisesRegex(EvidenceError, "totals"):
            runtime.collect(self.arguments(), bad)
        bad = deepcopy(pages)
        bad[1]["result"]["neighbors"][0]["weight"] = 0.6
        with self.assertRaisesRegex(EvidenceError, "conflicting weights"):
            runtime.collect(self.arguments(), bad)
        bad = deepcopy(pages)
        bad[1]["result"]["neighbors"].reverse()
        with self.assertRaisesRegex(EvidenceError, "ordering"):
            runtime.collect(self.arguments(), bad)
        bad = deepcopy(pages)
        bad[-1]["result"]["collection"]["next_cursor"] = bad[1]["arguments"]["cursor"]
        with self.assertRaisesRegex(EvidenceError, "loops"):
            runtime.collect(self.arguments(), bad)
        bad = deepcopy(pages)
        bad[1]["arguments"]["cursor"] = "orphan"
        with self.assertRaisesRegex(EvidenceError, "disconnected"):
            runtime.collect(self.arguments(), bad)

    def test_diagnostics_are_atomic_and_preserve_seven_raw_arguments(self) -> None:
        cases = [({"multiplex_id": None}, "invalid_input", "MISSING_SCOPE"),
                 ({"multiplex_id": "unknown"}, "not_found", "UNKNOWN_MULTIPLEX"),
                 ({"layer_id": "TFs:unknown"}, "not_found", "UNKNOWN_LAYER"),
                 ({"gene_id": "TP53"}, "invalid_input", "INVALID_GENE_INPUT"),
                 ({"direction": "in"}, "invalid_input", "DIRECTION_UNAVAILABLE"),
                 ({"mode": "top_k"}, "invalid_input", "MISSING_LIMIT"),
                 ({"mode": "top_k", "top_k": True}, "invalid_input", "INVALID_SELECTION"),
                 ({"cursor": "unknown"}, "invalid_input", "INVALID_CURSOR")]
        for changes, status, issue in cases:
            with self.subTest(changes=changes):
                args = self.arguments(**changes)
                self.assertEqual(self.runtime.get_neighbors(**args, evidence_id="unused"), {
                    "status": status, "provenance": {"registry": [], "graph": []},
                    "result": {"issue_code": issue, "arguments": args},
                })


if __name__ == "__main__":
    unittest.main()
