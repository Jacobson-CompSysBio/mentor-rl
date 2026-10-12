"""Focused tests for the compact S0/S1 source index."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime.world_model.source_index import (
    SOURCE_INDEX_SCHEMA_VERSION,
    SourceIndexError,
    build_payload,
    build_source_index,
    check_source_index,
    content_sha256,
    load_source_index,
    write_source_index,
)
from runtime.world_model.sources import SourceError, read_dendrogram


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WorldModelSourceIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.gene_ids = tuple(f"ENSG{number:011d}" for number in range(1, 6))

        self.gtf_path = self.root / "tiny.gtf.gz"
        gene_rows = [
            ("1", self.gene_ids[0], "Alpha", "protein_coding"),
            ("2", self.gene_ids[1], "Dup", "lncRNA"),
            ("X", self.gene_ids[2], "dup", "pseudogene"),
            ("MT", self.gene_ids[3], None, "protein_coding"),
        ]
        with gzip.open(self.gtf_path, "wt", encoding="utf-8", newline="") as handle:
            handle.write("##gtf-version 3\n")
            for chromosome, gene_id, symbol, biotype in gene_rows:
                attributes = f'gene_id "{gene_id}";'
                if symbol is not None:
                    attributes += f' gene_name "{symbol}";'
                attributes += f' gene_biotype "{biotype}";'
                handle.write(
                    f"{chromosome}\tfixture\tgene\t1\t10\t.\t+\t.\t"
                    f"{attributes}\n"
                )

        self.stable_id_path = self.root / "stable_id_event.txt.gz"
        retired_linked = "ENSG00000000010"
        retired_unlinked = "ENSG00000000011"
        with gzip.open(
            self.stable_id_path, "wt", encoding="latin-1", newline=""
        ) as handle:
            handle.write(
                f"{retired_linked}\t1\t{self.gene_ids[0]}\t2\t1\tgene\t1\n"
                f"{retired_unlinked}\t1\t\\N\t\\N\t1\tgene\t0\n"
                f"{self.gene_ids[0]}\t2\t{self.gene_ids[0]}\t3\t2\tgene\t1\n"
                "ENST00000000001\t1\tENST00000000002\t1\t1\ttranscript\t1\n"
            )

        self.hgnc_path = self.root / "hgnc.tsv"
        self.hgnc_path.write_text(
            "alias_symbol\tprev_symbol\tensembl_gene_id\n"
            f"OldAlpha|Shared\tFormerAlpha\t{self.gene_ids[0]}\n"
            f"dup|Shared\t\t{self.gene_ids[1]}\n"
            "Unlinked\t\t\n"
        )

        self.flist_path = self.root / "tiny_flist.tsv"
        self.flist_path.write_text(
            "/graphs/a.tsv\tHumanNetV3:string_ppi\n"
            "/graphs/b.tsv\tscPEN:brain:cortex:neuron\n"
        )

        self.dendrogram_path = self.root / "tiny_dendrogram.tsv"
        self.dendrogram_path.write_text(
            "node_id\tleft_id\tright_id\theight\tlabel\n"
            "501\t41\t77\t0.2\tNA\n"
            f"41\t-1\t-1\t0\t{self.gene_ids[0]}\n"
            "900\t501\t303\t1.0\tNA\n"
            f"77\t-1\t-1\t0\t{self.gene_ids[1]}\n"
            "303\t12\t88\t0.3\tNA\n"
            f"88\t-1\t-1\t0\t{self.gene_ids[4]}\n"
            f"12\t-1\t-1\t0\t{self.gene_ids[2]}\n"
        )

        self.contract = {
            "coordinate_system": {
                "ensembl_release": "116",
                "identifier_registry_id": "tiny_registry",
                "identifier_registry": {
                    "source_path": str(self.gtf_path),
                    "source_sha256": _sha256(self.gtf_path),
                    "gene_count": 4,
                    "named_gene_count": 3,
                    "unnamed_gene_count": 1,
                    "symbol_count": 2,
                    "ambiguous_symbol_count": 1,
                    "largest_candidate_count": 2,
                    "chromosome_count": 4,
                    "biotype_count": 3,
                    "casefold_collision_count": 1,
                    "stable_id_history": {
                        "source_path": str(self.stable_id_path),
                        "source_sha256": _sha256(self.stable_id_path),
                        "retired_gene_count": 2,
                        "linked_retired_gene_count": 1,
                    },
                    "hgnc_history": {
                        "source_path": str(self.hgnc_path),
                        "source_sha256": _sha256(self.hgnc_path),
                        "symbol_count": 4,
                        "symbol_key_count": 4,
                        "linked_symbol_count": 3,
                    },
                },
                "multiplex_id": "tiny_multiplex",
                "source_flist_path": str(self.flist_path),
                "source_flist_sha256": _sha256(self.flist_path),
                "layer_count": 2,
                "graph_gene_count": 4,
                "registry_graph_intersection_count": 3,
                "mentor_ev": {
                    "source_path": str(self.dendrogram_path),
                    "sha256": _sha256(self.dendrogram_path),
                    "node_count": 7,
                    "leaf_count": 4,
                    "eligible_clade_count": 2,
                    "eligible_leaf_count": {"min": 2, "max": 2},
                    "dendrogram_id": "tiny_tree",
                },
            },
            "source_index": {
                "schema_version": SOURCE_INDEX_SCHEMA_VERSION,
                "output_directory": str(self.root / "index"),
            },
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_payload_has_expected_counts_data_and_casefold_ambiguity(self) -> None:
        payload = build_payload(self.contract)

        self.assertEqual(
            payload["counts"],
            {
                "registry_gene_count": 4,
                "registry_named_gene_count": 3,
                "registry_unnamed_gene_count": 1,
                "registry_symbol_count": 2,
                "registry_ambiguous_symbol_count": 1,
                "registry_largest_candidate_count": 2,
                "registry_chromosome_count": 4,
                "registry_biotype_count": 3,
                "registry_casefold_collision_count": 1,
                "historical_symbol_count": 4,
                "historical_symbol_key_count": 4,
                "historical_symbol_linked_count": 3,
                "retired_gene_count": 2,
                "retired_gene_linked_count": 1,
                "layer_count": 2,
                "multiplex_gene_count": 4,
                "registry_multiplex_intersection_count": 3,
                "mentor_node_count": 7,
                "mentor_leaf_count": 4,
                "mentor_eligible_clade_count": 2,
            },
        )
        self.assertEqual(
            payload["layer_ids"],
            ["HumanNetV3:string_ppi", "scPEN:brain:cortex:neuron"],
        )

        fields = payload["gene_fields"]
        ambiguous = {
            gene_id: dict(zip(fields, values))
            for gene_id, values in payload["genes_by_id"].items()
            if values[0] is not None and values[0].casefold() == "dup"
        }
        self.assertEqual(
            ambiguous,
            {
                self.gene_ids[1]: {
                    "gene_symbol": "Dup",
                    "chromosome": "2",
                    "biotype": "lncRNA",
                },
                self.gene_ids[2]: {
                    "gene_symbol": "dup",
                    "chromosome": "X",
                    "biotype": "pseudogene",
                },
            },
        )
        self.assertEqual(
            payload["genes_by_id"][self.gene_ids[3]],
            [None, "MT", "protein_coding"],
        )
        self.assertEqual(
            payload["historical_symbols"],
            [
                ["FormerAlpha", ["previous"], [self.gene_ids[0]]],
                ["OldAlpha", ["alias"], [self.gene_ids[0]]],
                ["Shared", ["alias"], [self.gene_ids[0], self.gene_ids[1]]],
                ["Unlinked", ["alias"], []],
            ],
        )
        self.assertEqual(
            payload["retired_genes"],
            [
                ["ENSG00000000010", [self.gene_ids[0]]],
                ["ENSG00000000011", []],
            ],
        )

    def test_graph_membership_comes_from_leaves_and_clades_use_bounds(self) -> None:
        payload = build_payload(self.contract)

        self.assertEqual(
            payload["multiplex_gene_ids"],
            sorted(
                [
                    self.gene_ids[0],
                    self.gene_ids[1],
                    self.gene_ids[2],
                    self.gene_ids[4],
                ]
            ),
        )
        self.assertNotIn(self.gene_ids[3], payload["multiplex_gene_ids"])
        self.assertNotIn(self.gene_ids[4], payload["genes_by_id"])
        self.assertEqual(payload["mentor_clades"], [[303, 2], [501, 2]])

    def test_content_address_and_gzip_bytes_are_deterministic(self) -> None:
        first_payload = build_payload(self.contract)
        second_payload = build_payload(self.contract)
        self.assertEqual(content_sha256(first_payload), content_sha256(second_payload))

        first_path = write_source_index(first_payload, self.root / "first")
        second_path = write_source_index(second_payload, self.root / "second")
        self.assertEqual(first_path.name, second_path.name)
        self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
        self.assertIn(content_sha256(first_payload), first_path.name)

    def test_build_load_check_and_pinned_cache_reuse(self) -> None:
        payload = build_payload(self.contract)
        pinned_contract = deepcopy(self.contract)
        pinned_contract["source_index"]["content_sha256"] = content_sha256(payload)
        output_directory = self.root / "cache"

        path = build_source_index(pinned_contract, output_directory)
        self.assertEqual(load_source_index(path), payload)
        self.assertEqual(check_source_index(path, pinned_contract), payload)

        with patch(
            "runtime.world_model.source_index.build_payload",
            side_effect=AssertionError("a cache hit must not rebuild source data"),
        ):
            cached_path = build_source_index(pinned_contract, output_directory)
        self.assertEqual(cached_path, path)

        renamed = output_directory / "wrong-name.json.gz"
        renamed.write_bytes(path.read_bytes())
        with self.assertRaisesRegex(SourceIndexError, "filename does not match"):
            check_source_index(renamed)

    def test_malformed_dendrogram_is_rejected(self) -> None:
        malformed = self.root / "missing_child.tsv"
        malformed.write_text(
            "node_id\tleft_id\tright_id\theight\tlabel\n"
            f"5\t-1\t-1\t0\t{self.gene_ids[0]}\n"
            "100\t5\t999\t1\tNA\n"
        )
        with self.assertRaisesRegex(SourceError, "references missing child 999"):
            read_dendrogram(malformed)


if __name__ == "__main__":
    unittest.main()
