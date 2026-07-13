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


def _write_v3_curriculum_fixture(tmp_path: Path) -> Path:
    out_dir = tmp_path / "v3_curriculum"
    out_dir.mkdir()
    plan = json.loads(
        (audit.REPO_ROOT / "config" / "pretrajectory_sft_curriculum_v1.json").read_text(
            encoding="utf-8"
        )
    )
    plan_hash = audit.curriculum_plan_hash(plan)
    _write_json(out_dir / "curriculum_plan.json", plan)
    stages = {item["name"]: item for item in plan["stages"]}
    family_by_bucket: dict[str, dict] = {}
    for family in plan["question_families"]:
        family_by_bucket.setdefault(family["mixture_bucket"], family)
    bucket_counts = {
        name: int(round(weight * 100))
        for name, weight in plan["mixture"]["content_buckets"].items()
    }
    self_check_total = sum(bucket_counts.values())
    if self_check_total != 100:  # pragma: no cover - fixture contract guard
        raise AssertionError(self_check_total)

    multiplex_id = plan["graph_contract"]["multiplex_id"]
    store_id = "sha256:test-store"
    flist_id = "sha256:test-flist"
    rows: list[dict] = []
    canonical_objects: list[dict] = []
    index = 0
    for bucket, count in bucket_counts.items():
        family = family_by_bucket[bucket]
        stage = stages[family["primary_stage"]]
        book_mode = next(
            mode for mode in family["allowed_book_modes"] if mode in stage["allowed_book_modes"]
        )
        profile_name = stage["allowed_budget_profiles"][0]
        profile = plan["context_budget_profiles"][profile_name]
        for _ in range(count):
            split = "train" if index < 90 else "val" if index < 95 else "test"
            object_id = f"fact_{index:04d}"
            metadata = {
                "schema_version": "pretrajectory-sft-v3",
                "record_id": f"sft_{index:04d}",
                "oracle_fact_id": object_id,
                "oracle_fact_group_id": f"group_{index:04d}",
                "canonical_object_id": object_id,
                "book_mode": book_mode,
                "question_family": family["name"],
                "curriculum_stage": family["primary_stage"],
                "mixture_bucket": bucket,
                "split": split,
                "multiplex_id": multiplex_id,
                "store_id": store_id,
                "flist_id": flist_id,
                "layer_scope": "none",
                "layer_ids": [],
                "layer_families": [],
                "entity_namespace": "ensembl_gene_id_primary",
                "module_source": "none",
                "answer_format": "json",
                "difficulty_source": family["difficulty_source"],
                "context_budget_profile": profile_name,
                "evidence_handles": [f"fixture:{index}"],
                "provenance": {"plan_hash": plan_hash, "source": "unit_fixture"},
            }
            if book_mode == "tool_call" or family["primary_stage"] == "stage5_structured_tools":
                metadata["tool_schema_validated"] = True
            record = {
                "system": "Use the bounded curriculum fixture and return exact JSON.",
                "question": f"Return the fixture result for object {index}.",
                "answer": json.dumps({"index": index}, separators=(",", ":")),
                "metadata": metadata,
            }
            measurement = audit._curriculum_budget_measurement(record, profile)
            metadata["budget_measurement"] = measurement
            metadata["answer_budget"] = {
                "profile": profile_name,
                **audit._curriculum_generator_budget_measurement(record),
                "violations": [],
            }
            rows.append(record)
            canonical_objects.append(
                {
                    "object_id": object_id,
                    "object_type": family["name"],
                    "multiplex_id": multiplex_id,
                    "store_id": store_id,
                    "flist_id": flist_id,
                    "payload": {"index": index},
                }
            )
            index += 1

    rows_by_split = {
        split: [row for row in rows if row["metadata"]["split"] == split]
        for split in ("train", "val", "test")
    }
    for split, split_rows in rows_by_split.items():
        _write_jsonl(out_dir / f"{split}.jsonl", split_rows)
    _write_jsonl(out_dir / "canonical_objects.jsonl", canonical_objects)
    split_counts = {split: len(split_rows) for split, split_rows in rows_by_split.items()}
    family_counts: dict[str, int] = {}
    for row in rows:
        name = row["metadata"]["question_family"]
        family_counts[name] = family_counts.get(name, 0) + 1
    content_hash = audit._curriculum_stable_hash(
        sorted(rows, key=lambda row: row["metadata"]["record_id"])
    )
    manifest = {
        "schema_version": "pretrajectory-sft-curriculum-artifacts-v1",
        "dataset_schema_version": "pretrajectory-sft-v3",
        "plan_id": plan["plan_id"],
        "plan_hash": plan_hash,
        "build_profile": "unit_test",
        "selected_record_count": len(rows),
        "canonical_object_count": len(canonical_objects),
        "record_count_by_split": split_counts,
        "record_count_by_mixture_bucket": bucket_counts,
        "record_count_by_question_family": family_counts,
        "content_hash": content_hash,
        "source_identities": {
            "multiplex_id": multiplex_id,
            "store_id": store_id,
            "flist_id": flist_id,
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(
        out_dir / "audit_report.json",
        {
            "schema_version": manifest["schema_version"],
            "passed": True,
            "fatal_error_count": 0,
            "budget_violation_count_in_selected": 0,
            "raw_path_violation_count_in_selected": 0,
            "metadata_violation_count_in_selected": 0,
            "tool_schema_violation_count_in_selected": 0,
            "leakage_passed": True,
            "selected_record_count": len(rows),
            "record_count_by_split": split_counts,
            "record_count_by_mixture_bucket": bucket_counts,
            "question_family_counts": family_counts,
            "plan_id": plan["plan_id"],
            "plan_hash": plan_hash,
        },
    )
    _write_json(
        out_dir / "leakage_report.json",
        {
            "schema_version": manifest["schema_version"],
            "passed": True,
            "oracle_fact_cross_split_count": 0,
            "optional_group_cross_split_count": 0,
            "exact_duplicate_cross_split_count": 0,
            "near_duplicate_cross_split_count": 0,
            "selected_record_count": len(rows),
            "plan_id": plan["plan_id"],
            "plan_hash": plan_hash,
        },
    )
    _write_json(
        out_dir / "coverage_report.json",
        {
            "schema_version": manifest["schema_version"],
            "passed": True,
            "underfilled_material_cross_cell_count": 0,
            "plan_id": plan["plan_id"],
            "plan_hash": plan_hash,
        },
    )
    return out_dir


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
            self.assertEqual(report["schema_version"], "pretrajectory-sft-audit-v2")
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

    def test_audit_catches_over_budget_answer_and_stale_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = self._build_toy_dataset(Path(tmp))
            train_rows = _read_jsonl(out_dir / "train.jsonl")
            train_rows[0]["answer"] = "oversized_answer_token " * 1000
            _write_jsonl(out_dir / "train.jsonl", train_rows)

            report = audit.audit_pretrajectory_sft_dataset(out_dir, coverage_min_records=10_000)
            codes = {issue["code"] for issue in report["issues"]}

            self.assertFalse(report["passed"])
            self.assertEqual(report["answer_budget_report"]["over_budget_record_count"], 1)
            self.assertIn("answer_budget_exceeded", codes)
            self.assertIn("answer_budget_measurement_mismatch", codes)

    def test_audit_rejects_v2_manifest_without_budget_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = self._build_toy_dataset(Path(tmp))
            manifest_path = out_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("answer_budget_contract")
            _write_json(manifest_path, manifest)

            report = audit.audit_pretrajectory_sft_dataset(out_dir, coverage_min_records=10_000)
            codes = {issue["code"] for issue in report["issues"]}

            self.assertFalse(report["passed"])
            self.assertIn("missing_answer_budget_contract", codes)

    def test_audit_can_make_material_mixture_underfill_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = self._build_toy_dataset(Path(tmp))
            report = audit.audit_pretrajectory_sft_dataset(
                out_dir,
                coverage_min_records=10_000,
                mixture_contract_min_records=1,
                mixture_absolute_underfill_tolerance=0.0,
                mixture_relative_underfill_tolerance=0.0,
                mixture_underfill_policy="fatal",
            )
            codes = {issue["code"] for issue in report["issues"]}

            self.assertFalse(report["passed"])
            self.assertIn("mixture_bucket_materially_underfilled", codes)
            self.assertTrue(report["mixture_report"]["material_underfilled_buckets"])

    def test_audit_accepts_plan_driven_v3_without_legacy_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_v3_curriculum_fixture(Path(tmp))
            report = audit.audit_pretrajectory_sft_dataset(
                out_dir,
                output_path=out_dir / "audit_report_contract_v3.json",
                coverage_min_records=1_000,
                mixture_underfill_policy="fatal",
                training_max_sequence_tokens=4_096,
                eval_max_answer_tokens=384,
                max_answer_characters=16_384,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["schema_version"], "pretrajectory-sft-audit-v3")
            self.assertEqual(report["dataset_schema_version"], "pretrajectory-sft-v3")
            self.assertEqual(report["fatal_error_count"], 0)
            self.assertEqual(report["record_count"], 100)
            self.assertEqual(report["canonical_object_count"], 100)
            self.assertNotIn("validation_report.json", json.dumps(report))

    def test_v3_audit_recomputes_budget_and_selected_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_v3_curriculum_fixture(Path(tmp))
            train_rows = _read_jsonl(out_dir / "train.jsonl")
            train_rows[0]["answer"] = json.dumps({"index": 999_999, "tampered": True})
            _write_jsonl(out_dir / "train.jsonl", train_rows)

            report = audit.audit_pretrajectory_sft_dataset(
                out_dir,
                coverage_min_records=1_000,
                mixture_underfill_policy="fatal",
            )
            codes = {issue["code"] for issue in report["issues"]}

            self.assertFalse(report["passed"])
            self.assertIn("budget_measurement_mismatch", codes)
            self.assertIn("selected_content_hash_mismatch", codes)

    def test_v3_audit_requires_native_leakage_report_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_v3_curriculum_fixture(Path(tmp))
            leakage_path = out_dir / "leakage_report.json"
            leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
            leakage["passed"] = False
            leakage["oracle_fact_cross_split_count"] = 1
            _write_json(leakage_path, leakage)

            report = audit.audit_pretrajectory_sft_dataset(
                out_dir,
                coverage_min_records=1_000,
                mixture_underfill_policy="fatal",
            )
            codes = {issue["code"] for issue in report["issues"]}

            self.assertFalse(report["passed"])
            self.assertIn("native_report_failed", codes)
            self.assertIn("native_leakage_count_nonzero", codes)

    def test_v3_audit_independently_rejects_material_cross_cell_underfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_v3_curriculum_fixture(Path(tmp))
            manifest_path = out_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["build_profile"] = "patchcheck"
            _write_json(manifest_path, manifest)
            coverage_path = out_dir / "coverage_report.json"
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            coverage["passed"] = True
            coverage["underfilled_material_cross_cell_count"] = 0
            coverage["cross_cells"] = [
                {
                    "question_family": "entity_symbol_to_ensembl",
                    "compacted": 4,
                    "filtered": 0,
                    "selected": 3,
                }
            ]
            _write_json(coverage_path, coverage)

            report = audit.audit_pretrajectory_sft_dataset(
                out_dir,
                coverage_min_records=1_000,
                mixture_underfill_policy="fatal",
            )
            issues_by_code = {issue["code"]: issue for issue in report["issues"]}

            self.assertFalse(report["passed"])
            self.assertEqual(
                issues_by_code["material_cross_cell_underfill"]["context"][
                    "underfilled_count"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
