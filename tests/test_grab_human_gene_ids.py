"""Test the Ensembl registry extractor."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from runtime.world_model_s0_tools import (
    REGISTRY_SCHEMA_VERSION_V1,
    REGISTRY_SCHEMA_VERSION_V2,
)
from scripts.grab_human_gene_ids import (
    _read_gene_symbols,
    write_gene_registry,
)


class HumanGeneRegistryExtractorTests(unittest.TestCase):
    """Test the shared-contract registry output."""

    def test_extractor_keeps_gene_ids_without_names(self) -> None:
        """Represent an unnamed gene ID with an empty symbol list."""

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            gtf_path = directory / "Homo_sapiens.GRCh38.116.gtf.gz"
            rows = [
                (
                    "1\tEnsembl\tgene\t1\t10\t.\t+\t.\t"
                    'gene_id "ENSG00000000003"; gene_name "TSPAN6";'
                ),
                (
                    "1\tEnsembl\tgene\t20\t30\t.\t+\t.\t"
                    'gene_id "ENSG00000000005";'
                ),
                (
                    "1\tEnsembl\tgene\t40\t50\t.\t+\t.\t"
                    'gene_id "invalid"; gene_name "INVALID";'
                ),
                (
                    "1\tEnsembl\ttranscript\t1\t10\t.\t+\t.\t"
                    'gene_id "ENSG00000000419"; gene_name "DPM1";'
                ),
            ]
            with gzip.open(gtf_path, "wt", encoding="utf-8") as handle:
                handle.write("\n".join(rows) + "\n")

            mappings, counts = _read_gene_symbols(gtf_path)

            self.assertEqual(
                mappings,
                {
                    "ENSG00000000003": ["TSPAN6"],
                    "ENSG00000000005": [],
                },
            )
            self.assertEqual(counts["gene_feature_rows"], 3)
            self.assertEqual(counts["total_gene_ids"], 2)
            self.assertEqual(counts["named_gene_ids"], 1)
            self.assertEqual(counts["gene_ids_with_multiple_symbols"], 0)
            self.assertEqual(counts["gene_ids_without_name"], 1)
            self.assertEqual(counts["retained_gene_ids_without_name"], 1)
            self.assertEqual(counts["excluded_gene_ids_without_name"], 0)
            self.assertEqual(counts["invalid_gene_id_rows"], 1)

            registry_path = directory / "registry.json"
            write_gene_registry(
                mappings,
                counts,
                gtf_path=gtf_path,
                out_path=registry_path,
            )
            registry = json.loads(registry_path.read_text(encoding="utf-8"))

        self.assertEqual(
            registry["schema_version"],
            REGISTRY_SCHEMA_VERSION_V2,
        )
        self.assertEqual(
            registry["gene_symbols_by_id"]["ENSG00000000005"],
            [],
        )

    def test_writer_can_reproduce_the_named_only_v1_shape(self) -> None:
        """Keep the legacy registry schema available for v6."""

        mappings = {
            "ENSG00000000003": ["TSPAN6"],
            "ENSG00000000005": [],
        }
        counts = {
            "gene_feature_rows": 2,
            "total_gene_ids": 2,
            "named_gene_ids": 1,
            "gene_ids_with_multiple_symbols": 0,
            "gene_ids_without_name": 1,
            "retained_gene_ids_without_name": 1,
            "excluded_gene_ids_without_name": 0,
            "unique_gene_names": 1,
            "ambiguous_gene_names": 0,
            "largest_candidate_set": 1,
            "invalid_gene_id_rows": 0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            gtf_path = directory / "source.gtf.gz"
            with gzip.open(gtf_path, "wt", encoding="utf-8") as handle:
                handle.write("# test\n")
            registry_path = directory / "registry.json"

            write_gene_registry(
                mappings,
                counts,
                gtf_path=gtf_path,
                out_path=registry_path,
                schema_version=REGISTRY_SCHEMA_VERSION_V1,
            )
            registry = json.loads(registry_path.read_text(encoding="utf-8"))

        self.assertEqual(
            registry["schema_version"],
            REGISTRY_SCHEMA_VERSION_V1,
        )
        self.assertEqual(
            registry["gene_symbols_by_id"],
            {"ENSG00000000003": ["TSPAN6"]},
        )
        self.assertEqual(
            registry["counts"]["excluded_gene_ids_without_name"],
            1,
        )
        self.assertNotIn(
            "retained_gene_ids_without_name",
            registry["counts"],
        )

    def test_extractor_rejects_multiple_symbols_for_one_id(self) -> None:
        """Reject a GTF that cannot use the singular result field."""

        with tempfile.TemporaryDirectory() as tmpdir:
            gtf_path = Path(tmpdir) / "multiple.gtf.gz"
            rows = [
                (
                    "1\tEnsembl\tgene\t1\t10\t.\t+\t.\t"
                    'gene_id "ENSG00000000003"; gene_name "FIRST";'
                ),
                (
                    "1\tEnsembl\tgene\t20\t30\t.\t+\t.\t"
                    'gene_id "ENSG00000000003"; gene_name "SECOND";'
                ),
            ]
            with gzip.open(gtf_path, "wt", encoding="utf-8") as handle:
                handle.write("\n".join(rows) + "\n")

            with self.assertRaisesRegex(ValueError, "multiple normalized"):
                _read_gene_symbols(gtf_path)


if __name__ == "__main__":
    unittest.main()
