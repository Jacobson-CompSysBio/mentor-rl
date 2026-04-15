import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_multiplex_store import build_store


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

            manifest = build_store(multiplex_flist=str(flist_path), out_dir=str(out_dir))

            self.assertEqual(manifest["format_version"], "mentor-rl-multiplex-store-v1")
            self.assertEqual(manifest["num_genes"], 4)
            self.assertEqual(manifest["num_layers"], 2)
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "genes.tsv").exists())
            self.assertTrue((out_dir / "layers.tsv").exists())
            self.assertTrue((out_dir / "aggregate_indptr.bin").exists())

            saved_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["num_layers"], 2)
            self.assertEqual(len(saved_manifest["layers"]), 2)

            genes_table = (out_dir / "genes.tsv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(genes_table[0], "0\tENSG1")
            self.assertEqual(genes_table[-1], "3\tENSG4")

            aggregate_indptr = np.fromfile(out_dir / "aggregate_indptr.bin", dtype=np.uint64)
            aggregate_indices = np.fromfile(out_dir / "aggregate_indices.bin", dtype=np.uint32)
            aggregate_weights = np.fromfile(out_dir / "aggregate_weights.bin", dtype=np.float32)

            self.assertEqual(aggregate_indptr.tolist(), [0, 2, 4, 7, 8])
            self.assertEqual(aggregate_indices.tolist(), [1, 2, 0, 2, 0, 1, 3, 2])
            self.assertEqual(len(aggregate_weights), len(aggregate_indices))


if __name__ == "__main__":
    unittest.main()
