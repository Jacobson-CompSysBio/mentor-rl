from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.multiplex_store_oracle import EdgeFact, MultiplexStoreOracle


def _write_string_table(root: Path, values: list[str]) -> None:
    encoded = [value.encode("utf-8") for value in values]
    (root / "genes_data.bin").write_bytes(b"".join(encoded))
    offsets = np.array(
        [0] + list(np.cumsum([len(value) for value in encoded])), dtype=np.uint64
    )
    offsets.tofile(root / "genes_offsets.bin")


def _write_csr(root: Path, prefix: str, edges: list[tuple[int, int, float]], n: int) -> dict:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for left, right, weight in edges:
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    indptr = [0]
    indices: list[int] = []
    weights: list[float] = []
    for row in adjacency:
        for target, weight in sorted(row):
            indices.append(target)
            weights.append(weight)
        indptr.append(len(indices))
    np.asarray(indptr, dtype=np.uint64).tofile(root / f"{prefix}_indptr.bin")
    np.asarray(indices, dtype=np.uint32).tofile(root / f"{prefix}_indices.bin")
    np.asarray(weights, dtype=np.float32).tofile(root / f"{prefix}_weights.bin")
    return {
        "indptr_file": f"{prefix}_indptr.bin",
        "indices_file": f"{prefix}_indices.bin",
        "weights_file": f"{prefix}_weights.bin",
        "stored_nnz": len(indices),
        "undirected_edge_count": len(edges),
    }


def _build_store(root: Path) -> None:
    genes = ["ENSG0001", "ENSG0002", "ENSG0003", "ENSG0004", "ENSG0005"]
    _write_string_table(root, genes)
    layer_a = _write_csr(root, "layer_0000", [(0, 1, 0.5), (1, 2, 0.7)], len(genes))
    layer_b = _write_csr(root, "layer_0001", [(2, 3, 0.9)], len(genes))
    aggregate = _write_csr(
        root,
        "aggregate",
        [(0, 1, 0.5), (1, 2, 0.7), (2, 3, 0.9)],
        len(genes),
    )
    layers = [
        {**layer_a, "layer_index": 0, "layer_name": "bulkPEN:brain:test", "node_count": 3},
        {**layer_b, "layer_index": 1, "layer_name": "scPEN:brain:test", "node_count": 2},
    ]
    manifest = {
        "format_version": "mentor-rl-multiplex-store-v2",
        "dtypes": {"indices": "uint32", "indptr": "uint64", "weights": "float32"},
        "num_genes": len(genes),
        "num_layers": len(layers),
        "layers": layers,
        "files": {
            "aggregate": aggregate,
            "binary_metadata": {
                "genes": {
                    "data_file": "genes_data.bin",
                    "offsets_file": "genes_offsets.bin",
                }
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class MultiplexStoreOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        _build_store(root)
        self.oracle = MultiplexStoreOracle(root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_reads_full_store_and_layer_facts(self) -> None:
        self.assertEqual(self.oracle.gene_count, 5)
        self.assertEqual(self.oracle.layer_count, 2)
        self.assertEqual(self.oracle.aggregate_edge_count, 3)
        self.assertTrue(self.oracle.has_edge("ENSG0001", "ENSG0002"))
        self.assertFalse(
            self.oracle.has_edge("ENSG0001", "ENSG0002", layer="scPEN:brain:test")
        )
        self.assertEqual(
            self.oracle.gene_layers("ENSG0003"),
            ["bulkPEN:brain:test", "scPEN:brain:test"],
        )
        self.assertTrue(
            self.oracle.validate_edge_fact(EdgeFact("ENSG0001", "ENSG0002", 0.5))
        )

    def test_exact_paths_components_and_induced_edges(self) -> None:
        self.assertEqual(
            self.oracle.shortest_path("ENSG0001", "ENSG0004"),
            ["ENSG0001", "ENSG0002", "ENSG0003", "ENSG0004"],
        )
        self.assertIsNone(
            self.oracle.shortest_path("ENSG0001", "ENSG0004", max_hops=2)
        )
        self.assertEqual(
            self.oracle.induced_components(
                ["ENSG0001", "ENSG0002", "ENSG0004", "ENSG0005"]
            ),
            [["ENSG0001", "ENSG0002"], ["ENSG0004"], ["ENSG0005"]],
        )
        edges = self.oracle.induced_edges(["ENSG0001", "ENSG0002", "ENSG0003"])
        self.assertEqual(
            [(edge.source_gene_id, edge.target_gene_id) for edge in edges],
            [("ENSG0001", "ENSG0002"), ("ENSG0002", "ENSG0003")],
        )

    def test_sampling_never_relabels_edges_as_nonedges(self) -> None:
        edges = self.oracle.sample_edges(layer=None, count=3, seed=9)
        self.assertTrue(edges)
        self.assertTrue(all(self.oracle.validate_edge_fact(edge) for edge in edges))
        nonedges = self.oracle.sample_nonedges(
            layer=None, count=3, seed=17, require_present=False
        )
        self.assertTrue(nonedges)
        self.assertTrue(all(not self.oracle.has_edge(left, right) for left, right in nonedges))


if __name__ == "__main__":
    unittest.main()
