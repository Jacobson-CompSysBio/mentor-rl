"""Check native S2 source loading and edge-query prerequisite distinctions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime.world_model.s2_sources import (
    SOURCE_MANIFEST_PATH,
    S2SourceError,
    load_source_view,
)


class S2NativeSourceTests(unittest.TestCase):
    """Use explicit tiny source files to exercise loader and tool semantics."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = [f"ENSG{n:011d}" for n in range(1, 7)]
        self.manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
        descriptor = self.manifest["descriptor"]
        descriptor["multiplex_id"] = "unit_native_graph"
        descriptor["layer_order"] = ["HumanNetV3:string_ppi", "TFs:brain"]
        self.undirected = self.root / "undirected.tsv"
        self.directed = self.root / "directed.tsv"
        self.undirected.write_text(
            "from\tto\tweight\n"
            f"{self.ids[1]}\t{self.ids[0]}\t0\n"
            f"{self.ids[2]}\t{self.ids[3]}\t0.123456789012345\n"
        )
        self.directed.write_text(
            "TF\ttarget\tscore\n"
            f"{self.ids[2]}\t{self.ids[0]}\t0.987654321012345\n"
            f"{self.ids[0]}\t{self.ids[2]}\t0.765432109876543\n"
            f"{self.ids[2]}\t{self.ids[2]}\t1\n"
            f"{self.ids[2]}\t{self.ids[4]}\t0.5\n"
        )
        descriptor["layers"] = [
            {"layer_id": descriptor["layer_order"][0], "source_path": self.undirected.name,
             "source_sha256": "", "directed": False},
            {"layer_id": descriptor["layer_order"][1], "source_path": self.directed.name,
             "source_sha256": "", "directed": True},
        ]
        metadata = self.root / "direction.txt"
        metadata.write_text("The first directed column is source; the second is target.\n")
        descriptor["direction_metadata"] = [{
            "path": metadata.name, "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }]
        self.path = self.root / "sources.json"
        self.repin()

    def repin(self) -> None:
        """Pin an intentionally changed unit fixture before testing its semantics."""
        descriptor = self.manifest["descriptor"]
        for spec in descriptor["layers"]:
            spec["source_sha256"] = hashlib.sha256(
                (self.root / spec["source_path"]).read_bytes()
            ).hexdigest()
        self.manifest["graph_version"] = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.path.write_text(json.dumps(self.manifest))

    def query(self, view, genes, layer="HumanNetV3:string_ppi"):
        """Call the edge primitive with one explicit evidence acquisition."""
        return view.get_edge(view.multiplex_id, layer, genes, evidence_id="ev_unit_edge")

    def test_native_direction_loops_target_membership_and_weights(self) -> None:
        view = load_source_view(self.path)
        layer = view.layers["TFs:brain"]
        self.assertEqual(view.layer_order, ("HumanNetV3:string_ppi", "TFs:brain"))
        self.assertEqual(view.multiplex_nodes, frozenset(self.ids[:5]))
        self.assertIn(self.ids[4], layer.nodes)  # Target-only membership.
        self.assertIn((self.ids[2], self.ids[2]), layer.edges)
        self.assertEqual(len(layer.edges), 4)
        forward = self.query(view, [self.ids[2], self.ids[0]], "TFs:brain")
        reverse = self.query(view, [self.ids[0], self.ids[2]], "TFs:brain")
        self.assertEqual(forward["result"]["gene_ids"], [self.ids[2], self.ids[0]])
        self.assertEqual(forward["result"]["weight"], 0.987654321012345)
        self.assertEqual(reverse["result"]["weight"], 0.765432109876543)
        self.assertTrue(forward["result"]["directed"])
        no_reverse = self.query(view, [self.ids[4], self.ids[2]], "TFs:brain")
        self.assertFalse(no_reverse["result"]["edge_exists"])
        self.assertIsNone(no_reverse["result"]["weight"])

    def test_zero_weight_is_positive_and_undirected_pair_is_canonical(self) -> None:
        view = load_source_view(self.path)
        response = self.query(view, [self.ids[1], self.ids[0]])
        self.assertEqual(response["status"], "resolved")
        result = response["result"]
        self.assertEqual(list(result), [
            "gene_ids", "multiplex_id", "layer_id", "directed", "edge_exists",
            "weight", "missing_gene_ids", "not_in_layer_gene_ids",
        ])
        self.assertEqual(result["gene_ids"], self.ids[:2])
        self.assertTrue(result["edge_exists"])
        self.assertEqual(result["weight"], 0)
        self.assertEqual(response["provenance"]["registry"], [])
        evidence = response["provenance"]["graph"][0]
        self.assertEqual(evidence["graph_version"], self.manifest["graph_version"])
        self.assertEqual(evidence["layer_scope"], {
            "kind": "layers", "layer_ids": ["HumanNetV3:string_ppi"], "layer_family": None,
        })

    def test_negative_layer_absence_and_global_absence_are_distinct(self) -> None:
        view = load_source_view(self.path)
        negative = self.query(view, [self.ids[0], self.ids[3]])
        self.assertEqual(negative["status"], "resolved")
        self.assertFalse(negative["result"]["edge_exists"])
        self.assertEqual(negative["result"]["not_in_layer_gene_ids"], [])
        outside = self.query(view, [self.ids[0], self.ids[4]])
        self.assertEqual(outside["status"], "resolved")
        self.assertFalse(outside["result"]["edge_exists"])
        self.assertEqual(outside["result"]["not_in_layer_gene_ids"], [self.ids[4]])
        missing = self.query(view, [self.ids[5], self.ids[4]])
        self.assertEqual(missing["status"], "not_found")
        self.assertIsNone(missing["result"]["edge_exists"])
        self.assertIsNone(missing["result"]["weight"])
        self.assertEqual(missing["result"]["missing_gene_ids"], [self.ids[5]])
        self.assertEqual(missing["result"]["not_in_layer_gene_ids"], [self.ids[4]])
        self.assertEqual(len(missing["provenance"]["graph"]), 1)

    def test_rejections_preserve_raw_arguments_without_graph_evidence(self) -> None:
        view = load_source_view(self.path)
        cases = [
            (view.multiplex_id, "TFs:brain", [self.ids[0]] * 2, "invalid_input", "INVALID_PAIR"),
            (view.multiplex_id, "TFs:brain", ["TP53", self.ids[0]], "invalid_input", "INVALID_GENE_INPUT"),
            (view.multiplex_id, "TFs:brain", [], "invalid_input", "INVALID_PAIR"),
            ("unknown", "TFs:brain", self.ids[:2], "not_found", "UNKNOWN_MULTIPLEX"),
            (view.multiplex_id, "TFs:unknown", self.ids[:2], "not_found", "UNKNOWN_LAYER"),
            (view.multiplex_id, None, self.ids[:2], "invalid_input", "MISSING_SCOPE"),
        ]
        for multiplex, layer, genes, status, code in cases:
            with self.subTest(code=code):
                response = view.get_edge(multiplex, layer, genes, evidence_id="unused")
                self.assertEqual(response, {
                    "status": status, "provenance": {"registry": [], "graph": []},
                    "result": {"issue_code": code, "arguments": {
                        "multiplex_id": multiplex, "layer_id": layer, "gene_ids": genes,
                    }},
                })

    def test_duplicate_native_keys_are_rejected(self) -> None:
        with self.undirected.open("a") as handle:
            handle.write(f"{self.ids[0]}\t{self.ids[1]}\t0\n")
        self.repin()
        with self.assertRaisesRegex(S2SourceError, "duplicate native edge"):
            load_source_view(self.path)

    def test_hashes_are_rechecked_even_after_a_cached_load(self) -> None:
        first = load_source_view(self.path)
        second = load_source_view(self.path)
        self.assertIs(first.layers["TFs:brain"], second.layers["TFs:brain"])
        with self.directed.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(S2SourceError, "SHA-256 mismatch"):
            load_source_view(self.path)

    def test_direction_metadata_and_descriptor_are_pinned(self) -> None:
        (self.root / "direction.txt").write_text("changed orientation\n")
        with self.assertRaisesRegex(S2SourceError, "SHA-256 mismatch"):
            load_source_view(self.path)
        self.manifest["descriptor"]["layers"][1]["directed"] = False
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(S2SourceError, "descriptor SHA-256 mismatch"):
            load_source_view(self.path)

    def test_bad_header_and_nonfinite_weights_are_rejected(self) -> None:
        for text, expected in [
            ("source\tdestination\tstrength\n", "unrecognized edge header"),
            (f"from\tto\tweight\n{self.ids[0]}\t{self.ids[1]}\tNaN\n", "finite JSON number"),
        ]:
            self.undirected.write_text(text)
            self.repin()
            with self.assertRaisesRegex(S2SourceError, expected):
                load_source_view(self.path)

    def test_default_descriptor_retains_the_reviewed_view_identity(self) -> None:
        manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
        self.assertEqual(manifest["graph_version"], "5c78cd6e3430b8ab384753e2d9ead4d5ec701e70c18bface1acb7060e3d5a29d")
        self.assertEqual(len(manifest["descriptor"]["layers"]), 4)
        self.assertEqual(manifest["descriptor"]["multiplex_id"], "s2_dataset_review_20261026_v1")


if __name__ == "__main__":
    unittest.main()
