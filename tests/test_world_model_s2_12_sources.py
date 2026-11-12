"""Check the separate PEN source view and its exact archived-byte relation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime.world_model.s2_12_sources import (
    GRAPH_VERSION, MULTIPLEX_ID, SOURCE_MANIFEST_PATH, load_source_view,
    verify_source_transformation,
)
from runtime.world_model.s2_multiplex import MultiplexRuntime
from runtime.world_model.s2_sources import S2SourceError, load_source_view as load_old_view


def digest(raw):
    """Hash explicit fixture bytes without changing their representation."""
    return hashlib.sha256(raw).hexdigest()


class ArchivedBulkTests(unittest.TestCase):
    """Use small byte fixtures to reject changes beyond endpoint suffixes."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
        descriptor = self.manifest["descriptor"]
        self.layer = deepcopy(descriptor["layers"][4])
        self.layer["source_path"] = "current.tsv"
        descriptor["multiplex_id"] = "unit_bulk_view"
        descriptor["layers"] = [self.layer]
        descriptor["layer_order"] = [self.layer["layer_id"]]
        self.relation = descriptor["source_transformations"][0]
        self.relation.update(archived_source_path="archive.tsv", source_path="current.tsv",
                             interpretation_metadata_path="archive.json")
        descriptor["direction_metadata"] = [{"path": "archive.json", "sha256": ""}]
        self.archived = (b"from\tto\tweight\r\n"
                         b"ENSG00000000002.12\tENSG00000000001.3\t0.123456789012345\r\n"
                         b"ENSG00000000002.12\tENSG00000000003\t0\r\n")
        self.current = self.archived.replace(b".12\t", b"\t").replace(b".3\t", b"\t")
        self.path = self.root / "sources.json"
        self.repin()

    def repin(self):
        """Pin a deliberate fixture change before checking its semantic validity."""
        (self.root / "archive.tsv").write_bytes(self.archived)
        (self.root / "current.tsv").write_bytes(self.current)
        metadata = {"configuration": {"network_mode": "undirected"},
                    "output": {"table_sha256": digest(self.archived)}}
        raw = json.dumps(metadata).encode()
        (self.root / "archive.json").write_bytes(raw)
        self.relation.update(archived_source_sha256=digest(self.archived),
                             source_sha256=digest(self.current),
                             interpretation_metadata_sha256=digest(raw))
        self.layer["source_sha256"] = digest(self.current)
        self.manifest["descriptor"]["direction_metadata"][0]["sha256"] = digest(raw)
        self.write_manifest()

    def write_manifest(self):
        """Update only the descriptor pin after a fixture field changes."""
        self.manifest["graph_version"] = digest(json.dumps(
            self.manifest["descriptor"], sort_keys=True, separators=(",", ":")).encode())
        self.path.write_text(json.dumps(self.manifest))

    def test_suffixes_only_preserve_exact_bytes_weights_and_endpoint_union(self):
        report = verify_source_transformation(self.relation, self.root)
        self.assertEqual(report["endpoint_occurrences"], 4)
        self.assertEqual(report["removed_version_suffixes"], 3)
        self.assertTrue(report["byte_identical"])
        view = load_source_view(self.path)
        layer = view.layers[self.layer["layer_id"]]
        self.assertEqual(len(view.multiplex_nodes), 3)
        self.assertEqual(layer.edges[("ENSG00000000001", "ENSG00000000002")], 0.123456789012345)
        self.assertEqual(layer.edges[("ENSG00000000002", "ENSG00000000003")], 0)
        self.assertEqual(len(view.source_pins), 3)

    def test_nonendpoint_bytes_and_row_order_must_match(self):
        original = self.current
        changes = [original.replace(b"0.123456789012345", b"0.123456789012346"),
                   original.replace(b"\r\n", b"\n"),
                   original.replace(b"from\tto", b"to\tfrom"),
                   b"".join(original.splitlines(keepends=True)[i] for i in (0, 2, 1)),
                   original + original.splitlines(keepends=True)[1]]
        for current in changes:
            with self.subTest(current=current):
                self.current = current
                self.repin()
                with self.assertRaises(S2SourceError):
                    load_source_view(self.path)

    def test_only_terminal_numeric_versions_are_accepted(self):
        self.archived = self.archived.replace(b".12\t", b".12a\t")
        self.repin()
        with self.assertRaisesRegex(S2SourceError, "invalid archived bulk endpoint"):
            load_source_view(self.path)

    def test_pins_rechecked_after_cached_parse(self):
        load_source_view(self.path)
        (self.root / "archive.tsv").write_bytes(self.archived + b"\n")
        with self.assertRaisesRegex(S2SourceError, "SHA-256 mismatch"):
            load_source_view(self.path)

    def test_relation_must_bind_the_registered_undirected_layer(self):
        self.relation["layer_id"] = "bulkPEN:brain:other"
        self.write_manifest()
        with self.assertRaisesRegex(S2SourceError, "bind its native layer"):
            load_source_view(self.path)

    def test_native_duplicate_pairs_still_rejected_after_valid_transform(self):
        self.archived += self.archived.splitlines(keepends=True)[1]
        self.current += self.current.splitlines(keepends=True)[1]
        self.repin()
        with self.assertRaisesRegex(S2SourceError, "duplicate native edge"):
            load_source_view(self.path)

    def test_descriptor_and_interpretation_are_pinned(self):
        self.manifest["descriptor"]["weights"] = "float32"
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(S2SourceError, "descriptor SHA-256 mismatch"):
            load_source_view(self.path)
        self.write_manifest()
        with self.assertRaisesRegex(S2SourceError, "unsupported native graph interpretation"):
            load_source_view(self.path)

    def test_archived_metadata_must_declare_the_pinned_undirected_source(self):
        raw = json.dumps({"configuration": {"network_mode": "directed"},
                          "output": {"table_sha256": digest(self.archived)}}).encode()
        (self.root / "archive.json").write_bytes(raw)
        self.relation["interpretation_metadata_sha256"] = digest(raw)
        self.manifest["descriptor"]["direction_metadata"][0]["sha256"] = digest(raw)
        self.write_manifest()
        with self.assertRaisesRegex(S2SourceError, "does not pin the undirected source"):
            load_source_view(self.path)


class ApprovedSixFileViewTests(unittest.TestCase):
    """Recompute the approved complete view and its real old-version contrast."""

    def test_complete_view_retains_old_layers_and_expands_membership(self):
        view, old = load_source_view(), load_old_view()
        self.assertEqual((view.multiplex_id, view.graph_version), (MULTIPLEX_ID, GRAPH_VERSION))
        self.assertEqual((len(old.multiplex_nodes), len(view.multiplex_nodes)), (19033, 26240))
        self.assertEqual(len(view.multiplex_nodes - old.multiplex_nodes), 7207)
        self.assertEqual(view.layer_order[:4], old.layer_order)
        self.assertEqual([(len(layer.edges), len(layer.nodes)) for layer in view.layers.values()],
                         [(996123, 18001), (78991, 11859), (201218, 11361),
                          (7968, 5818), (77708, 21064), (13084, 10896)])
        for layer in old.layer_order:
            self.assertIs(view.layers[layer], old.layers[layer])
        runtime = MultiplexRuntime(view)
        self.assertEqual([runtime.family_of(layer) for layer in view.layer_order[4:]],
                         ["bulk_pen", "scpen"])
        gene = "ENSG00000063127"
        self.assertNotIn(gene, old.multiplex_nodes)
        self.assertIn(gene, view.layers[view.layer_order[4]].nodes)
        self.assertIn(gene, view.layers[view.layer_order[5]].nodes)


if __name__ == "__main__":
    unittest.main()
