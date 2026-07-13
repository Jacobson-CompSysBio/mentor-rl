import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import validate_pretrajectory_sft_curriculum_plan as contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "config" / "pretrajectory_sft_curriculum_v1.json"


class PretrajectorySftCurriculumContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = contract.load_curriculum_plan(PLAN_PATH)

    def test_source_controlled_plan_is_valid_and_complete(self) -> None:
        self.assertEqual(self.plan["contract_version"], contract.CONTRACT_VERSION)
        self.assertEqual(len(self.plan["question_families"]), 81)
        self.assertEqual(
            [stage["name"] for stage in self.plan["stages"]],
            contract.EXPECTED_STAGE_NAMES,
        )
        self.assertEqual(
            {family["id"] for family in self.plan["question_families"]},
            set(range(1, 82)),
        )
        self.assertIn(
            "oracle_fact_id",
            self.plan["record_contract"]["required_metadata_fields"],
        )
        self.assertFalse(self.plan["numeric_policy"]["closed_book_arbitrary_floats_allowed"])
        primary_families = {
            family["name"] for family in self.plan["question_families"]
        }
        staged_families = {
            name
            for stage in self.plan["stages"][:5]
            for name in stage["required_question_families"]
        }
        self.assertEqual(primary_families, staged_families)

    def test_stable_hash_is_independent_of_json_key_order(self) -> None:
        reordered = dict(reversed(list(self.plan.items())))
        self.assertEqual(
            contract.curriculum_plan_hash(self.plan),
            contract.curriculum_plan_hash(reordered),
        )
        self.assertEqual(len(contract.curriculum_plan_hash(self.plan)), 64)

    def test_missing_question_family_reports_id_and_stage_coverage_errors(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["question_families"] = broken["question_families"][:-1]

        errors = contract.validate_curriculum_plan(broken)

        self.assertTrue(any("ids must cover 1..81 exactly" in error for error in errors))
        self.assertTrue(
            any("stage4_module_world_model" in error and "extra=" in error for error in errors)
        )

    def test_mixture_and_stage_mode_violations_are_aggregated(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["mixture"]["content_buckets"]["entity_normalization_schema"] = 0.09
        broken["stages"][4]["allowed_book_modes"] = ["closed_book"]

        errors = contract.validate_curriculum_plan(broken)

        self.assertTrue(any("fractions must sum to 1.0" in error for error in errors))
        self.assertTrue(
            any("choose_rwr_loe_tool" in error and "no allowed book mode" in error for error in errors)
        )
        self.assertGreater(len(errors), 2)

    def test_loader_reports_invalid_json_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plan.json"
            path.write_text('{"contract_version": ', encoding="utf-8")

            with self.assertRaises(contract.CurriculumPlanValidationError) as caught:
                contract.load_curriculum_plan(path)

        message = str(caught.exception)
        self.assertIn("invalid JSON at line 1", message)
        self.assertIn(str(path), message)

    def test_plan_round_trip_remains_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "curriculum_plan.json"
            path.write_text(json.dumps(self.plan, indent=2), encoding="utf-8")
            loaded = contract.load_curriculum_plan(path)

        self.assertEqual(
            contract.curriculum_plan_hash(loaded),
            contract.curriculum_plan_hash(self.plan),
        )


if __name__ == "__main__":
    unittest.main()
