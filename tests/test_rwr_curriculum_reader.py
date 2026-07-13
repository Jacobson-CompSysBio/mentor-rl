from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime.rwr_curriculum_reader import (
    CrossShardDistanceUnavailable,
    RwrArtifactError,
    RwrCurriculumReader,
    rank_cache_file_stem,
)


def _context_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    context: dict[str, object] = {
        "schema_version": "rwr-loe-rank-cache-v1",
        "network_flist": "/private/secret/full_brain_flist.tsv",
        "network_flist_sha256": "a" * 64,
        "rwr_hpc_build_id": "/private/secret/rwr/build_frontier",
        "restart": 0.7,
        "delta": 0.5,
        "reduction_method": "geometric",
        "threshold": 1e-10,
        "edgelist_has_headers": True,
        "parser_version": 1,
        "ranking_semantics": "rwr_encoding_desc_min_rank_seed_excluded",
    }
    context_dir = root / f"context_{_context_hash(context)[:16]}"
    _write_json(context_dir / "cache_context.json", context)

    shard_seeds = {
        0: ("ENSG_A", "ENSG_C", "ENSG_E"),
        1: ("ENSG_B", "ENSG_D"),
    }
    rank_rows = {
        "ENSG_A": (("ENSG_X", 9.0, 1), ("ENSG_Y", 8.0, 2), ("ENSG_B", 7.0, 3), ("ENSG_Z", 2.0, 4), ("ENSG_C", 1.0, 5), ("ENSG_W", 0.5, 6)),
        "ENSG_B": (("ENSG_Y", 9.5, 1), ("ENSG_X", 8.5, 2), ("ENSG_A", 7.5, 3), ("ENSG_Q", 2.5, 4), ("ENSG_D", 1.5, 5), ("ENSG_W", 0.4, 6)),
        "ENSG_C": (("ENSG_X", 9.0, 1), ("ENSG_A", 8.0, 2), ("ENSG_Y", 7.0, 3), ("ENSG_Z", 2.0, 4), ("ENSG_E", 1.0, 5), ("ENSG_W", 0.5, 6)),
        "ENSG_D": (("ENSG_Q", 9.0, 1), ("ENSG_B", 8.0, 2), ("ENSG_Y", 7.0, 3), ("ENSG_Z", 2.0, 4), ("ENSG_X", 1.0, 5), ("ENSG_W", 0.5, 6)),
        "ENSG_E": (("ENSG_A", 9.0, 1), ("ENSG_X", 8.0, 2), ("ENSG_Y", 7.0, 3), ("ENSG_Z", 2.0, 4), ("ENSG_C", 1.0, 5), ("ENSG_W", 0.5, 6)),
    }

    for shard_index, seeds in shard_seeds.items():
        shard_id = f"shard_{shard_index:05d}"
        shard_dir = context_dir / "shards" / shard_id
        _write_json(
            shard_dir / "rank_cache_manifest.json",
            {
                "schema_version": "rwr-loe-rank-cache-v1",
                "shard_index": shard_index,
                "shard_count": len(shard_seeds),
                "completed_seed_count": len(seeds),
                "seed_gene_ids": list(seeds),
                "encoding_matrix_path": f"/private/secret/{shard_id}/encodings.tsv",
                "cache_context": context,
            },
        )
        output_dir = shard_dir / "rwr_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        distance_path = output_dir / f"fixture_{shard_id}_spearman_dist_matrix.tsv"
        with distance_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("INDEX", *seeds))
            if shard_index == 0:
                writer.writerow(("ENSG_A", "NA", "NA", "NA"))
                writer.writerow(("ENSG_C", "0.25", "NA", "NA"))
                writer.writerow(("ENSG_E", "0.10", "0.50", "NA"))
            else:
                writer.writerow(("ENSG_B", "NA", "NA"))
                writer.writerow(("ENSG_D", "0.75", "NA"))

        for seed in seeds:
            rows = rank_rows[seed]
            stem = rank_cache_file_stem(seed)
            metadata = {
                "schema_version": "rwr-loe-rank-cache-v1",
                "seed_gene_id": seed,
                "shard_index": shard_index,
                "shard_count": len(shard_seeds),
                "ranked_gene_count": len(rows),
                "encoding_matrix_path": f"/private/secret/{shard_id}/encodings.tsv",
                "cache_context": context,
                "created_at": "2026-01-01T00:00:00Z",
            }
            _write_json(context_dir / "ranks" / f"{stem}.metadata.json", metadata)
            rank_path = context_dir / "ranks" / f"{stem}.ranks.tsv.gz"
            rank_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(rank_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("NodeNames", "Scores", "rank"),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                for gene, score, rank in rows:
                    writer.writerow({"NodeNames": gene, "Scores": score, "rank": rank})

    return context_dir, context


class RwrCurriculumReaderTests(unittest.TestCase):
    def test_loads_the_complete_rank_vector_and_exact_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir, _ = _build_fixture(Path(tmp))
            reader = RwrCurriculumReader(context_dir)

            vector = reader.rank_vector("ensg_a")
            self.assertEqual(len(vector.rows), 6)
            self.assertEqual(vector.metadata.ranked_gene_count, 6)
            self.assertEqual(reader.rank("ENSG_A", "ENSG_W").rank, 6)
            comparison = reader.compare_ranks("ENSG_A", "ENSG_Y", "ENSG_Z")
            self.assertEqual(comparison.closer_gene_id, "ENSG_Y")
            self.assertFalse(comparison.is_tie)
            self.assertEqual(
                [row.gene_id for row in reader.top_k("ENSG_A", 3, exclude_genes=("ENSG_B",))],
                ["ENSG_X", "ENSG_Y", "ENSG_Z"],
            )

            filtered = reader.filter_queries(
                "ENSG_A", ("ENSG_W", "ENSG_MISSING", "ENSG_Y", "ENSG_W")
            )
            self.assertEqual(filtered.query_gene_ids, ("ENSG_W", "ENSG_MISSING", "ENSG_Y"))
            self.assertEqual([row.gene_id for row in filtered.ranked_query_genes], ["ENSG_Y", "ENSG_W"])
            self.assertEqual(filtered.missing_gene_ids, ("ENSG_MISSING",))

    def test_elbow_and_top_k_intersection_use_full_ordered_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir, _ = _build_fixture(Path(tmp))
            reader = RwrCurriculumReader(context_dir)

            elbow = reader.elbow("ENSG_A")
            self.assertEqual(elbow.status, "ok")
            self.assertIsNotNone(elbow.elbow_rank_cutoff)
            expected = tuple(
                row.gene_id
                for row in reader.rank_vector("ENSG_A").rows
                if row.rank < int(elbow.elbow_rank_cutoff or 0)
            )
            self.assertEqual(elbow.retained_gene_ids, expected)
            self.assertTrue(all(reader.rank("ENSG_A", gene).rank < elbow.elbow_rank_cutoff for gene in expected))

            overlap = reader.top_k_intersection("ENSG_A", "ENSG_B", 3)
            self.assertEqual(overlap.neighborhood_a, ("ENSG_X", "ENSG_Y", "ENSG_Z"))
            self.assertEqual(overlap.neighborhood_b, ("ENSG_Y", "ENSG_X", "ENSG_Q"))
            self.assertEqual(overlap.intersection_gene_ids, ("ENSG_X", "ENSG_Y"))
            self.assertEqual(overlap.intersection_size, 2)
            self.assertEqual(overlap.union_size, 4)
            self.assertEqual(overlap.jaccard, 0.5)

    def test_lower_triangle_is_symmetric_and_supports_rows_and_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir, _ = _build_fixture(Path(tmp))
            reader = RwrCurriculumReader(context_dir)

            shard = reader.distance_shard("shard_00000")
            self.assertEqual(shard.distance("ENSG_A", "ENSG_C"), 0.25)
            self.assertEqual(shard.distance("ENSG_C", "ENSG_A"), 0.25)
            self.assertEqual(shard.distance("ENSG_A", "ENSG_A"), 0.0)
            self.assertEqual(
                [(row.gene_id, row.distance) for row in shard.closest("ENSG_A", 2)],
                [("ENSG_E", 0.1), ("ENSG_C", 0.25)],
            )
            comparison = reader.compare_distances("ENSG_A", "ENSG_C", "ENSG_E")
            self.assertEqual(comparison.closer_gene_id, "ENSG_E")
            self.assertEqual(reader.distance("ENSG_E", "ENSG_C"), 0.5)

    def test_shard_routing_reports_materialized_and_cross_shard_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir, _ = _build_fixture(Path(tmp))
            reader = RwrCurriculumReader(context_dir)

            seed_route = reader.route_seed("ENSG_C")
            self.assertEqual(seed_route.shard_id, "shard_00000")
            self.assertEqual(seed_route.position_in_shard, 1)
            same = reader.route_distance_pair("ENSG_A", "ENSG_E")
            self.assertTrue(same.distance_available)
            self.assertEqual(same.status, "same_shard")
            self.assertEqual(same.distance_shard_id, "shard_00000")

            cross = reader.route_distance_pair("ENSG_A", "ENSG_B")
            self.assertFalse(cross.distance_available)
            self.assertEqual(cross.status, "cross_shard_not_materialized")
            self.assertEqual(cross.gene_a_shard_id, "shard_00000")
            self.assertEqual(cross.gene_b_shard_id, "shard_00001")
            with self.assertRaises(CrossShardDistanceUnavailable) as caught:
                reader.distance("ENSG_A", "ENSG_B")
            self.assertEqual(caught.exception.route, cross)

    def test_public_provenance_preserves_hash_identity_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir, context = _build_fixture(Path(tmp))
            reader = RwrCurriculumReader(context_dir)
            payloads = [
                reader.public_provenance(),
                dict(reader.seed_metadata("ENSG_A").provenance),
                dict(reader.route_seed("ENSG_A").provenance),
                dict(reader.route_distance_pair("ENSG_A", "ENSG_C").provenance),
                dict(reader.distance_shard("shard_00000").provenance),
            ]

            rendered = json.dumps(payloads, sort_keys=True)
            self.assertNotIn("/private", rendered)
            self.assertNotIn("full_brain_flist.tsv", rendered)
            self.assertNotIn("build_frontier", rendered)
            self.assertIn(str(context["network_flist_sha256"]), rendered)
            self.assertIn(_context_hash(context), rendered)
            self.assertEqual(reader.identity.cache_id, context_dir.name)

    def test_rejects_incomplete_lower_triangle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir, _ = _build_fixture(Path(tmp))
            matrix = next(
                (context_dir / "shards" / "shard_00000" / "rwr_output").glob(
                    "*_spearman_dist_matrix.tsv"
                )
            )
            text = matrix.read_text(encoding="utf-8").replace(
                "ENSG_E\t0.10\t0.50\tNA", "ENSG_E\t0.10\tNA\tNA"
            )
            matrix.write_text(text, encoding="utf-8")
            reader = RwrCurriculumReader(context_dir)

            with self.assertRaisesRegex(RwrArtifactError, "missing 1 lower-triangle"):
                reader.distance_shard("shard_00000")


if __name__ == "__main__":
    unittest.main()
