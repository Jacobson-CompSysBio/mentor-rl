import json
import tempfile
import unittest
from pathlib import Path

from scripts import audit_pretrajectory_sft_dataset as audit
from scripts import build_pretrajectory_sft_dataset as sft
from tests.test_build_pretrajectory_sft_dataset import _read_jsonl, _write_json, _write_rank_cache, _write_toy_corpus, _write_toy_graph


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class PretrajectorySftAuditTests(unittest.TestCase):
    def _build_toy_dataset(self, tmp_path: Path) -> Path:
        graph_flist = _write_toy_graph(tmp_path)
        corpus_dir = _write_toy_corpus(tmp_path)
        rank_context = _write_rank_cache(tmp_path)
        store_manifest = tmp_path / "store_manifest.json"
        _write_json(store_manifest, {"format_version": "toy-graph-v1"})
        out_dir = tmp_path / "sft_out"
        sft.build_pretrajectory_sft_dataset(
            out_dir=out_dir,
            mixed_corpus_dir=corpus_dir,
            store_manifest_path=store_manifest,
            graph_flist=graph_flist,
            graph_layer_limit=None,
            graph_max_edges_per_layer=None,
            rank_cache_context_dir=rank_context,
            seed=17,
            max_rwr_seeds=5,
            target_counts=None,
        )
        return out_dir

    def test_audit_pretrajectory_sft_dataset_passes_clean_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = self._build_toy_dataset(Path(tmp))
            report = audit.audit_pretrajectory_sft_dataset(
                out_dir,
                output_path=out_dir / "audit_report.json",
                coverage_min_records=10_000,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["fatal_error_count"], 0)
            self.assertTrue((out_dir / "audit_report.json").exists())
            self.assertGreater(report["record_count_by_view_type"]["monoplex_edge_existence"], 0)
            self.assertGreater(report["record_count_by_view_type"]["module_overlap_set_algebra"], 0)
            self.assertGreater(report["record_count_by_curriculum_stage"]["stage1_entity_schema"], 0)
            self.assertGreater(report["record_count_by_curriculum_stage"]["stage5_structured_tools"], 0)
            self.assertGreater(report["record_count_by_source"]["MENTOR_GW_DENDROGRAM"], 0)
            self.assertGreater(report["record_count_by_source"]["RWR_LOE_FULL_BRAIN"], 0)

    def test_audit_catches_split_leakage_and_unsupported_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = self._build_toy_dataset(Path(tmp))
            train_rows = _read_jsonl(out_dir / "train.jsonl")
            val_rows = _read_jsonl(out_dir / "val.jsonl")
            train_record = train_rows[0]
            val_rows[0]["metadata"]["canonical_object_id"] = train_record["metadata"]["canonical_object_id"]
            val_rows[0]["answer"] = "This evidence definitely causally proves the relationship."
            _write_jsonl(out_dir / "val.jsonl", val_rows)

            report = audit.audit_pretrajectory_sft_dataset(out_dir, coverage_min_records=10_000)
            codes = {issue["code"] for issue in report["issues"]}

            self.assertFalse(report["passed"])
            self.assertIn("canonical_object_split_leakage", codes)
            self.assertIn("unsupported_causal_language_in_answer", codes)


if __name__ == "__main__":
    unittest.main()
