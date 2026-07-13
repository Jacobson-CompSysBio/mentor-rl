import argparse
import unittest

from scripts import check_pretrajectory_sft_readiness as gate


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        fail_on_dataset_warnings=False,
        min_overall_exact=0.80,
        min_open_book_exact=0.90,
        min_tool_observation_exact=0.90,
        min_no_context_exact=0.55,
        min_schema_exact=0.95,
        min_topology_exact=0.80,
        min_rwr_exact=0.85,
        min_module_exact=0.85,
        min_tool_schema_exact=0.95,
        min_id_recall=0.90,
        min_layer_recall=0.85,
        min_number_recall=0.85,
        max_unsupported_language_rate=0.005,
        min_recovery_exact_success_rate=0.25,
        min_refinement_exact_success_rate=0.25,
        min_exact_membership_pair_rate=0.01,
    )


def _dataset_audit() -> dict:
    return {
        "schema_version": "pretrajectory-sft-audit-v2",
        "fatal_error_count": 0,
        "warning_count": 0,
        "answer_budget_report": {
            "manifest_contract_present": True,
            "over_budget_record_count": 0,
        },
    }


def _exact_report(**summary_overrides: float) -> dict:
    summary = {
        "exact_graph_fact_pass_rate": 0.82,
        "unsupported_language_rate": 0.0,
        "mean_id_recall": 0.91,
        "mean_layer_recall": 0.86,
        "mean_number_recall": 0.86,
    }
    summary.update(summary_overrides)
    return {
        "evaluator_contract": {"version": "pretrajectory-sft-exact-v3"},
        "gold_self_evaluation": {"passed": True},
        "summary": {**summary, "exact_only": dict(summary)},
        "by_context_mode": {},
        "by_mixture_bucket": {},
    }


def _v3_dataset_audit() -> dict:
    plan_hash = "plan-hash-v3"
    return {
        "schema_version": "pretrajectory-sft-audit-v3",
        "dataset_schema_version": "pretrajectory-sft-v3",
        "passed": True,
        "fatal_error_count": 0,
        "warning_count": 0,
        "plan_hash": plan_hash,
        "content_hash": "content-hash-v3",
        "native_reports": {
            name: {"passed": True, "plan_hash": plan_hash}
            for name in gate.CURRICULUM_NATIVE_REPORT_NAMES
        },
        "answer_budget_report": {
            "manifest_contract_present": True,
            "over_budget_record_count": 0,
            "missing_record_budget_metadata_count": 0,
        },
    }


def _v3_exact_report(**summary_overrides: float) -> dict:
    report = _exact_report(**summary_overrides)
    report["dataset_identity"] = {
        "dataset_schema_version": "pretrajectory-sft-v3",
        "plan_hash": "plan-hash-v3",
        "content_hash": "content-hash-v3",
    }
    return report


class PretrajectorySftReadinessGateTests(unittest.TestCase):
    def test_v3_bridge_and_matching_exact_identity_pass_contract_gates(self) -> None:
        report = gate.check_readiness(
            dataset_audit=_v3_dataset_audit(),
            exact_report=_v3_exact_report(),
            trajectory_audit=None,
            args=_args(),
        )

        self.assertTrue(report["valid"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["failed_contract_gates"], [])

    def test_v3_stale_exact_identity_and_failed_leakage_report_are_invalid(self) -> None:
        dataset_audit = _v3_dataset_audit()
        dataset_audit["native_reports"]["leakage_report.json"]["passed"] = False
        exact_report = _v3_exact_report()
        exact_report["dataset_identity"]["content_hash"] = "stale-content"

        report = gate.check_readiness(
            dataset_audit=dataset_audit,
            exact_report=exact_report,
            trajectory_audit=None,
            args=_args(),
        )

        self.assertFalse(report["valid"])
        self.assertEqual(
            {item["name"] for item in report["failed_contract_gates"]},
            {
                "dataset_native_reports_current",
                "exact_report_dataset_identity_current",
            },
        )

    def test_required_gates_pass_when_exact_metrics_clear_thresholds(self) -> None:
        report = gate.check_readiness(
            dataset_audit={**_dataset_audit(), "warning_count": 2},
            exact_report=_exact_report(),
            trajectory_audit=None,
            args=_args(),
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["valid"])
        self.assertEqual(report["decision"], "move_to_dpo_trajectory_generation")

    def test_required_gate_fails_on_zero_exact_rate(self) -> None:
        report = gate.check_readiness(
            dataset_audit=_dataset_audit(),
            exact_report=_exact_report(exact_graph_fact_pass_rate=0.0),
            trajectory_audit=None,
            args=_args(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(report["valid"])
        self.assertEqual(report["failed_required_gates"][0]["observed"], 0.0)

    def test_stale_dataset_and_invalid_gold_make_report_invalid(self) -> None:
        dataset_audit = _dataset_audit()
        dataset_audit["schema_version"] = "pretrajectory-sft-audit-v1"
        exact_report = _exact_report()
        exact_report["gold_self_evaluation"]["passed"] = False

        report = gate.check_readiness(
            dataset_audit=dataset_audit,
            exact_report=exact_report,
            trajectory_audit=None,
            args=_args(),
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["valid"])
        self.assertEqual(report["decision"], "repair_evaluation_or_dataset_contract")
        self.assertEqual(
            {item["name"] for item in report["failed_contract_gates"]},
            {"dataset_audit_schema_current", "gold_self_evaluation_passed"},
        )


if __name__ == "__main__":
    unittest.main()
