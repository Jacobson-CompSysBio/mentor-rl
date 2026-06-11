import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_multiplex_store import build_store, read_binary_string_table


def _write_test_inputs(base_dir: Path) -> Path:
    layer_one = base_dir / "layer_one.tsv"
    layer_one.write_text(
        "\n".join(
            [
                "ENSG1 ENSG2 1.0",
                "ENSG2 ENSG3 0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    layer_two = base_dir / "layer_two.tsv"
    layer_two.write_text(
        "\n".join(
            [
                "ENSG1 ENSG3 0.8",
                "ENSG3 ENSG4 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    flist_path = base_dir / "test.flist"
    flist_path.write_text(
        "\n".join(
            [
                f"{layer_one}\tppi\t1",
                f"{layer_two}\tcoexp\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return flist_path


class MultiplexStoreBuilderTests(unittest.TestCase):
    def test_build_store_writes_expected_metadata_and_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            flist_path = _write_test_inputs(base_dir)
            out_dir = base_dir / "store"
            progress_path = out_dir / "progress.json"

            manifest = build_store(
                multiplex_flist=str(flist_path),
                out_dir=str(out_dir),
                progress_path=str(progress_path),
            )

            self.assertEqual(manifest["format_version"], "mentor-rl-multiplex-store-v2")
            self.assertEqual(manifest["num_genes"], 4)
            self.assertEqual(manifest["num_layers"], 2)
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "genes.tsv").exists())
            self.assertTrue((out_dir / "layers.tsv").exists())
            self.assertTrue((out_dir / "genes_data.bin").exists())
            self.assertTrue((out_dir / "genes_offsets.bin").exists())
            self.assertTrue((out_dir / "layer_names_data.bin").exists())
            self.assertTrue((out_dir / "layer_names_offsets.bin").exists())
            self.assertTrue((out_dir / "layer_node_counts.bin").exists())
            self.assertTrue((out_dir / "aggregate_indptr.bin").exists())
            self.assertTrue(progress_path.exists())

            saved_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["num_layers"], 2)
            self.assertEqual(len(saved_manifest["layers"]), 2)
            self.assertEqual(saved_manifest["files"]["binary_metadata"]["genes"]["data_file"], "genes_data.bin")
            self.assertEqual(saved_manifest["files"]["binary_metadata"]["layers"]["node_counts_file"], "layer_node_counts.bin")

            saved_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_progress["status"], "completed")
            self.assertEqual(saved_progress["overall_progress"], 1.0)
            self.assertEqual(saved_progress["metrics"]["num_layers"], 2)
            self.assertEqual(saved_progress["metrics"]["num_genes"], 4)

            genes_table = (out_dir / "genes.tsv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(genes_table[0], "0\tENSG1")
            self.assertEqual(genes_table[-1], "3\tENSG4")
            binary_genes = read_binary_string_table(out_dir / "genes_data.bin", out_dir / "genes_offsets.bin")
            self.assertEqual(binary_genes, ["ENSG1", "ENSG2", "ENSG3", "ENSG4"])
            binary_layers = read_binary_string_table(
                out_dir / "layer_names_data.bin",
                out_dir / "layer_names_offsets.bin",
            )
            self.assertEqual(binary_layers, ["ppi", "coexp"])

            aggregate_indptr = np.fromfile(out_dir / "aggregate_indptr.bin", dtype=np.uint64)
            aggregate_indices = np.fromfile(out_dir / "aggregate_indices.bin", dtype=np.uint32)
            aggregate_weights = np.fromfile(out_dir / "aggregate_weights.bin", dtype=np.float32)

            self.assertEqual(aggregate_indptr.tolist(), [0, 2, 4, 7, 8])
            self.assertEqual(aggregate_indices.tolist(), [1, 2, 0, 2, 0, 1, 3, 2])
            self.assertEqual(len(aggregate_weights), len(aggregate_indices))

    def test_build_store_can_emit_binary_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            flist_path = _write_test_inputs(base_dir)
            out_dir = base_dir / "store"

            manifest = build_store(
                multiplex_flist=str(flist_path),
                out_dir=str(out_dir),
                write_legacy_text_metadata=False,
            )

            self.assertFalse((out_dir / "genes.tsv").exists())
            self.assertFalse((out_dir / "layers.tsv").exists())
            self.assertTrue((out_dir / "genes_data.bin").exists())
            self.assertTrue((out_dir / "layer_names_data.bin").exists())
            self.assertIsNone(manifest["files"]["gene_table"])
            self.assertIsNone(manifest["files"]["layer_table"])


if __name__ == "__main__":
    unittest.main()
