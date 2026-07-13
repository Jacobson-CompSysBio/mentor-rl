import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import pretrajectory_curriculum_artifacts as artifacts
from scripts.validate_pretrajectory_sft_curriculum_plan import load_curriculum_plan


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "config" / "pretrajectory_sft_curriculum_v1.json"


def _synthetic_plan() -> dict:
    plan = load_curriculum_plan(PLAN_PATH)
    plan["build_profiles"]["synthetic"] = {
        "split_counts": {"train": 180, "val": 180, "test": 180},
        "minimum_selected_per_required_family": 1,
        "minimum_selected_per_material_cross_cell": 1,
    }
    return plan


def _repair_plan(*, split_count: int = 240, cell_minimum: int = 4) -> dict:
    plan = _synthetic_plan()
    plan["build_profiles"]["synthetic"]["split_counts"] = {
        "train": split_count,
        "val": split_count,
        "test": split_count,
    }
    plan["build_profiles"]["synthetic"][
        "minimum_selected_per_material_cross_cell"
    ] = cell_minimum
    return plan


def _book_mode(family: dict, stage: dict) -> str:
    return sorted(set(family["allowed_book_modes"]) & set(stage["allowed_book_modes"]))[0]


def _budget_profile(stage: dict) -> str:
    return stage["allowed_budget_profiles"][0]


def _candidate(plan: dict, family: dict, split: str, index: int) -> dict:
    stages = {stage["name"]: stage for stage in plan["stages"]}
    stage = stages[family["primary_stage"]]
    record_id = f"{split}-{family['id']:02d}-{index:02d}"
    object_id = f"object-{record_id}"
    book_mode = _book_mode(family, stage)
    metadata = {
        "schema_version": plan["dataset_schema_version"],
        "record_id": record_id,
        "oracle_fact_id": f"fact-{record_id}",
        "book_mode": book_mode,
        "question_family": family["name"],
        "multiplex_id": plan["graph_contract"]["multiplex_id"],
        "store_id": "synthetic-full-store-v1",
        "flist_id": "synthetic-flist-v1",
        "layer_scope": "all_layers",
        "layer_ids": [],
        "layer_families": [],
        "entity_namespace": "ensembl_gene_id_primary",
        "module_source": "none",
        "answer_format": "json",
        "difficulty_source": family["difficulty_source"],
        "context_budget_profile": _budget_profile(stage),
        "evidence_handles": [],
        "provenance": {"source": "synthetic_oracle"},
        "split": split,
        "curriculum_stage": family["primary_stage"],
        "mixture_bucket": family["mixture_bucket"],
        "canonical_object_id": object_id,
        "tool_schema_valid": book_mode != "tool_call" or True,
        "coverage_objects": {
            "canonical_genes": [f"ENSG{family['id']:011d}"],
        },
    }
    for metric in plan["coverage_contract"]["required_unique_object_metrics"]:
        if metric not in {"canonical_genes", "live_tool_schemas"}:
            metadata["coverage_objects"][metric] = [f"{metric}:{record_id}"]
    if book_mode == "tool_call":
        metadata["tool_name"] = f"synthetic_tool_{family['id'] % 3}"
    return {
        "record": {
            "system": "Answer only with the exact synthetic oracle fact.",
            "question": f"Resolve synthetic record {record_id}.",
            "answer": {"record_id": record_id, "value": index},
            "metadata": metadata,
        },
        "canonical_object": {
            "object_id": object_id,
            "object_type": family["name"],
            "payload": {
                "record_id": record_id,
                "gene_id": f"ENSG{family['id']:011d}",
                "value": index,
            },
        },
    }


def _synthetic_candidates(plan: dict, per_family_split: int = 8) -> list[dict]:
    return [
        _candidate(plan, family, split, index)
        for split in ("train", "val", "test")
        for family in plan["question_families"]
        for index in range(per_family_split)
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class PretrajectoryCurriculumArtifactsTests(unittest.TestCase):
    def test_largest_remainder_is_exact_and_deterministic(self) -> None:
        quotas = artifacts.largest_remainder_quotas(17, {"a": 0.5, "b": 0.3, "c": 0.2})
        self.assertEqual(quotas, {"a": 9, "b": 5, "c": 3})
        self.assertEqual(sum(quotas.values()), 17)

        tied = artifacts.largest_remainder_quotas(2, {"z": 1.0, "a": 1.0, "m": 1.0})
        self.assertEqual(tied, {"z": 0, "a": 1, "m": 1})

    def test_compiler_writes_exact_plan_driven_artifact_set(self) -> None:
        plan = _synthetic_plan()
        candidates = _synthetic_candidates(plan)
        # Exercise record-id and rendered-text deduplication without changing quotas.
        candidates.append(copy.deepcopy(candidates[0]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            out_dir = Path(temporary_directory) / "dataset"
            result = artifacts.compile_pretrajectory_curriculum_artifacts(
                candidates=candidates,
                out_dir=out_dir,
                plan=plan,
                build_profile="synthetic",
                seed=17,
                tool_validator=lambda record: record["metadata"].get("tool_schema_valid") is True,
            )

            required = {
                "curriculum_plan.json",
                "manifest.json",
                "coverage_report.json",
                "leakage_report.json",
                "audit_report.json",
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "canonical_objects.jsonl",
            }
            self.assertTrue(all((out_dir / name).is_file() for name in required))
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            audit = json.loads((out_dir / "audit_report.json").read_text(encoding="utf-8"))
            leakage = json.loads((out_dir / "leakage_report.json").read_text(encoding="utf-8"))
            coverage = json.loads((out_dir / "coverage_report.json").read_text(encoding="utf-8"))

            self.assertTrue(audit["passed"])
            self.assertTrue(leakage["passed"])
            self.assertEqual(audit["plan_hash"], manifest["plan_hash"])
            self.assertEqual(coverage["plan_hash"], manifest["plan_hash"])
            self.assertEqual(leakage["plan_hash"], manifest["plan_hash"])
            self.assertEqual(manifest["record_count_by_split"], {"train": 180, "val": 180, "test": 180})
            self.assertEqual(manifest["deduplicated_candidate_count"], 81 * 8 * 3)
            self.assertEqual(len(_read_jsonl(out_dir / "train.jsonl")), 180)
            expected_bucket_counts = artifacts.largest_remainder_quotas(
                180, plan["mixture"]["content_buckets"]
            )
            train_bucket_counts = {
                name: payload["selected"]
                for name, payload in coverage["selection"]["splits"]["train"]["buckets"].items()
            }
            self.assertEqual(train_bucket_counts, expected_bucket_counts)
            self.assertTrue(all(not value["underfilled"] for value in coverage["question_families"].values()))
            self.assertGreater(coverage["unique_object_coverage"]["canonical_genes"], 0)
            self.assertEqual(
                set(coverage["unique_object_coverage"]),
                set(plan["coverage_contract"]["required_unique_object_metrics"]),
            )
            self.assertTrue(all(coverage["unique_object_coverage"].values()))
            self.assertEqual(
                coverage["unique_object_coverage"]["live_tool_schemas"], 3
            )
            self.assertEqual(
                coverage["unique_object_coverage_details"]["live_tool_schemas"][
                    "declared_count"
                ],
                0,
            )
            self.assertEqual(
                coverage["unique_object_coverage_details"]["live_tool_schemas"][
                    "derived_count"
                ],
                3,
            )

            stage2_rows = _read_jsonl(
                out_dir / "curriculum" / "stage2_topology_priors" / "train.jsonl"
            )
            self.assertTrue(any(row["metadata"]["curriculum_role"] == "replay" for row in stage2_rows))
            self.assertTrue(any(row["metadata"]["curriculum_role"] == "primary" for row in stage2_rows))
            blend_rows = _read_jsonl(out_dir / "curriculum" / "stage6_blend" / "train.jsonl")
            self.assertEqual(len(blend_rows), 180)
            blend_counts = result["manifest"]["curriculum_stage_files"]["stages"]["stage6_blend"]["train"]
            self.assertEqual(blend_counts["source_stage_counts"], blend_counts["source_stage_quotas"])
            self.assertTrue(
                all(row["metadata"]["curriculum_role"] == "consolidation" for row in blend_rows)
            )

            canonical_ids = {
                row["object_id"] for row in _read_jsonl(out_dir / "canonical_objects.jsonl")
            }
            output_rows = _read_jsonl(out_dir / "train.jsonl") + blend_rows
            self.assertTrue(
                all(row["metadata"]["canonical_object_id"] in canonical_ids for row in output_rows)
            )

    def test_post_selection_repair_is_deterministic_and_preserves_quotas(self) -> None:
        plan = _repair_plan()
        candidates = _synthetic_candidates(plan)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = artifacts.compile_pretrajectory_curriculum_artifacts(
                candidates=candidates,
                out_dir=root / "first",
                plan=plan,
                build_profile="synthetic",
                seed=29,
                tool_validator=lambda record: record["metadata"].get("tool_schema_valid") is True,
            )
            second = artifacts.compile_pretrajectory_curriculum_artifacts(
                candidates=candidates,
                out_dir=root / "second",
                plan=plan,
                build_profile="synthetic",
                seed=29,
                tool_validator=lambda record: record["metadata"].get("tool_schema_valid") is True,
            )

            first_repair = first["coverage_report"]["selection"]["post_selection_repair"]
            second_repair = second["coverage_report"]["selection"]["post_selection_repair"]
            self.assertGreater(first_repair["underfilled_material_cell_count_before"], 0)
            self.assertEqual(first_repair["underfilled_material_cell_count_after"], 0)
            self.assertGreater(first_repair["swap_count"], 0)
            self.assertEqual(first_repair, second_repair)
            self.assertEqual(
                first["manifest"]["content_hash"], second["manifest"]["content_hash"]
            )
            self.assertEqual(
                first["manifest"]["record_count_by_split"],
                {"train": 240, "val": 240, "test": 240},
            )
            expected_quotas = artifacts.largest_remainder_quotas(
                240, plan["mixture"]["content_buckets"]
            )
            for split in ("train", "val", "test"):
                observed = {
                    bucket: payload["selected"]
                    for bucket, payload in first["coverage_report"]["selection"]["splits"][
                        split
                    ]["buckets"].items()
                }
                self.assertEqual(observed, expected_quotas)
            self.assertEqual(
                first["coverage_report"]["underfilled_material_cross_cell_count"], 0
            )
            self.assertTrue(
                all(
                    not cell["underfilled"]
                    for cell in first["coverage_report"]["cross_cells"]
                    if cell["material"]
                )
            )
            family_minimum = plan["build_profiles"]["synthetic"][
                "minimum_selected_per_required_family"
            ]
            self.assertTrue(
                all(
                    count >= family_minimum
                    for count in first["manifest"]["record_count_by_question_family"].values()
                )
            )

    def test_infeasible_material_cross_cell_minimum_is_fatal(self) -> None:
        # At 180 rows/split the 7% tool bucket has 39 rows total, fewer than
        # 12 tool families * 4 required selections, so no quota-preserving
        # repair can satisfy the declared material-cell lower bounds.
        plan = _repair_plan(split_count=180, cell_minimum=4)
        candidates = _synthetic_candidates(plan)

        with tempfile.TemporaryDirectory() as temporary_directory:
            out_dir = Path(temporary_directory) / "dataset"
            with self.assertRaises(artifacts.CurriculumArtifactError) as caught:
                artifacts.compile_pretrajectory_curriculum_artifacts(
                    candidates=candidates,
                    out_dir=out_dir,
                    plan=plan,
                    build_profile="synthetic",
                    tool_validator=lambda record: record["metadata"].get(
                        "tool_schema_valid"
                    )
                    is True,
                )

            self.assertIn(
                "material_cross_cell_repair_infeasible",
                {error["code"] for error in caught.exception.errors},
            )
            self.assertFalse(out_dir.exists())

    def test_invalid_candidates_are_filtered_and_audited_before_selection(self) -> None:
        plan = _synthetic_plan()
        candidates = _synthetic_candidates(plan)
        candidates[0]["record"]["question"] = "Read /lustre/project/private/oracle.json first."
        candidates[1]["record"]["answer"] = {"value": "x" * 20000}
        tool_index = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate["record"]["metadata"]["book_mode"] == "tool_call"
        )
        candidates[tool_index]["record"]["metadata"]["tool_schema_valid"] = False

        with tempfile.TemporaryDirectory() as temporary_directory:
            out_dir = Path(temporary_directory) / "dataset"
            result = artifacts.compile_pretrajectory_curriculum_artifacts(
                candidates=candidates,
                out_dir=out_dir,
                plan=plan,
                build_profile="synthetic",
                tool_validator=lambda record: record["metadata"].get("tool_schema_valid") is True,
            )

            codes = {issue["code"] for issue in result["audit_report"]["filtered_issues"]}
            self.assertIn("raw_path_in_model_text", codes)
            self.assertIn("budget_exceeded", codes)
            self.assertIn("invalid_tool_schema", codes)
            self.assertEqual(result["audit_report"]["fatal_error_count"], 0)
            self.assertGreaterEqual(result["audit_report"]["filtered_candidate_count"], 3)
            selected_text = "\n".join(
                row["question"] for row in _read_jsonl(out_dir / "train.jsonl")
            )
            self.assertNotIn("/lustre/", selected_text)

    def test_cross_split_oracle_fact_is_fatal_and_publishes_nothing(self) -> None:
        plan = _synthetic_plan()
        candidates = _synthetic_candidates(plan)
        leak = copy.deepcopy(candidates[0])
        leak["record"]["metadata"]["split"] = "val"
        leak["record"]["metadata"]["record_id"] = "leaking-variant"
        leak["record"]["question"] = "A distinct rendering of the leaking fact."
        leak["canonical_object"]["object_id"] = "object-leaking-variant"
        leak["record"]["metadata"]["canonical_object_id"] = "object-leaking-variant"
        candidates.append(leak)

        with tempfile.TemporaryDirectory() as temporary_directory:
            out_dir = Path(temporary_directory) / "dataset"
            with self.assertRaises(artifacts.CurriculumArtifactError) as caught:
                artifacts.compile_pretrajectory_curriculum_artifacts(
                    candidates=candidates,
                    out_dir=out_dir,
                    plan=plan,
                    build_profile="synthetic",
                    tool_validator=lambda record: record["metadata"].get("tool_schema_valid") is True,
                )

            self.assertFalse(out_dir.exists())
            self.assertIn("oracle_fact_cross_split", {error["code"] for error in caught.exception.errors})

    def test_missing_required_family_is_fatal_even_when_bucket_quota_can_fill(self) -> None:
        plan = _synthetic_plan()
        candidates = [
            candidate
            for candidate in _synthetic_candidates(plan)
            if candidate["record"]["metadata"]["question_family"] != "structured_state_update"
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            out_dir = Path(temporary_directory) / "dataset"
            with self.assertRaises(artifacts.CurriculumArtifactError) as caught:
                artifacts.compile_pretrajectory_curriculum_artifacts(
                    candidates=candidates,
                    out_dir=out_dir,
                    plan=plan,
                    build_profile="synthetic",
                    tool_validator=lambda record: record["metadata"].get("tool_schema_valid") is True,
                )

            errors = caught.exception.errors
            self.assertTrue(
                any(
                    error["code"] == "question_family_underfill"
                    and error.get("question_family") == "structured_state_update"
                    for error in errors
                )
            )
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
