"""Focused source tests for the canonical S1.2 corpus builder."""

from __future__ import annotations

import gzip
from pathlib import Path
import tempfile
import unittest

from runtime.world_model.sources import (
    EnsemblGtfIdentifiers,
    SourceError,
    TranscriptIdentifier,
    VersionedGeneId,
    read_ensembl_gtf,
    read_ensembl_gtf_identifiers,
)


def _gtf_row(feature: str, attributes: str) -> str:
    """Return one syntactically complete GTF row."""

    return (
        f"1\tEnsembl\t{feature}\t1\t10\t.\t+\t.\t{attributes}\n"
    )


class WorldModelS12SourceTests(unittest.TestCase):
    """Validate release-native gene versions and transcript accessions."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "registry.gtf.gz"

    def _write(self, *rows: str) -> Path:
        with gzip.open(self.path, "wt", encoding="utf-8") as handle:
            handle.write("##gtf-version 3\n")
            handle.writelines(rows)
        return self.path

    def test_reads_real_versions_and_transcript_parent_links(self) -> None:
        path = self._write(
            _gtf_row(
                "gene",
                'gene_id "ENSG00000000002"; gene_version "12"; '
                'gene_name "TWO"; gene_biotype "protein_coding";',
            ),
            _gtf_row(
                "transcript",
                'gene_id "ENSG00000000002"; gene_version "12"; '
                'transcript_id "ENST00000000002"; '
                'transcript_version "7";',
            ),
            _gtf_row(
                "exon",
                'gene_id "ignored"; transcript_id "ignored";',
            ),
            _gtf_row(
                "gene",
                'gene_id "ENSG00000000001"; gene_version "3"; '
                'gene_name "ONE"; gene_biotype "lncRNA";',
            ),
            _gtf_row(
                "transcript",
                'gene_id "ENSG00000000001"; gene_version "3"; '
                'transcript_id "ENST00000000001"; '
                'transcript_version "1";',
            ),
        )

        identifiers = read_ensembl_gtf_identifiers(path)

        self.assertIsInstance(identifiers, EnsemblGtfIdentifiers)
        self.assertEqual(
            identifiers.genes,
            (
                VersionedGeneId("ENSG00000000001", 3),
                VersionedGeneId("ENSG00000000002", 12),
            ),
        )
        self.assertEqual(
            identifiers.transcripts,
            (
                TranscriptIdentifier(
                    "ENST00000000001", "ENSG00000000001"
                ),
                TranscriptIdentifier(
                    "ENST00000000002", "ENSG00000000002"
                ),
            ),
        )
        self.assertEqual(
            identifiers.genes[0].versioned_gene_id,
            "ENSG00000000001.3",
        )

    def test_requires_native_positive_versions(self) -> None:
        cases = {
            "missing gene version": (
                _gtf_row(
                    "gene",
                    'gene_id "ENSG00000000001"; '
                    'gene_biotype "protein_coding";',
                ),
                _gtf_row(
                    "transcript",
                    'gene_id "ENSG00000000001"; '
                    'transcript_id "ENST00000000001"; '
                    'transcript_version "1";',
                ),
            ),
            "zero transcript version": (
                _gtf_row(
                    "gene",
                    'gene_id "ENSG00000000001"; gene_version "1"; '
                    'gene_biotype "protein_coding";',
                ),
                _gtf_row(
                    "transcript",
                    'gene_id "ENSG00000000001"; '
                    'transcript_id "ENST00000000001"; '
                    'transcript_version "0";',
                ),
            ),
        }
        for label, rows in cases.items():
            with self.subTest(label=label):
                path = self._write(*rows)
                with self.assertRaises(SourceError):
                    read_ensembl_gtf_identifiers(path)

    def test_rejects_invalid_or_duplicate_transcript_ids(self) -> None:
        gene = _gtf_row(
            "gene",
            'gene_id "ENSG00000000001"; gene_version "1"; '
            'gene_biotype "protein_coding";',
        )
        cases = {
            "wrong accession type": (
                _gtf_row(
                    "transcript",
                    'gene_id "ENSG00000000001"; '
                    'transcript_id "ENSP00000000001"; '
                    'transcript_version "1";',
                ),
            ),
            "duplicate": (
                _gtf_row(
                    "transcript",
                    'gene_id "ENSG00000000001"; '
                    'transcript_id "ENST00000000001"; '
                    'transcript_version "1";',
                ),
                _gtf_row(
                    "transcript",
                    'gene_id "ENSG00000000001"; '
                    'transcript_id "ENST00000000001"; '
                    'transcript_version "2";',
                ),
            ),
        }
        for label, transcripts in cases.items():
            with self.subTest(label=label):
                path = self._write(gene, *transcripts)
                with self.assertRaises(SourceError):
                    read_ensembl_gtf_identifiers(path)

    def test_rejects_transcript_with_an_absent_parent_gene(self) -> None:
        path = self._write(
            _gtf_row(
                "gene",
                'gene_id "ENSG00000000001"; gene_version "1"; '
                'gene_biotype "protein_coding";',
            ),
            _gtf_row(
                "transcript",
                'gene_id "ENSG00000000002"; '
                'transcript_id "ENST00000000001"; '
                'transcript_version "1";',
            ),
        )

        with self.assertRaisesRegex(SourceError, "parent genes are absent"):
            read_ensembl_gtf_identifiers(path)

    def test_existing_gene_registry_reader_is_unchanged(self) -> None:
        path = self._write(
            _gtf_row(
                "gene",
                'gene_id "ENSG00000000001"; '
                'gene_name "ENSG00000000001"; '
                'gene_biotype "protein_coding";',
            ),
            _gtf_row(
                "transcript",
                'gene_id "ENSG00000000001"; '
                'transcript_id "ENST00000000001";',
            ),
        )

        genes = read_ensembl_gtf(path)

        self.assertEqual(len(genes), 1)
        self.assertEqual(genes[0].gene_id, "ENSG00000000001")
        self.assertIsNone(genes[0].gene_symbol)
        self.assertEqual(genes[0].chromosome, "1")
        self.assertEqual(genes[0].biotype, "protein_coding")


if __name__ == "__main__":
    unittest.main()
