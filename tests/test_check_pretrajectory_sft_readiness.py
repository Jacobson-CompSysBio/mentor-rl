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


class PretrajectorySftReadinessGateTests(unittest.TestCase):
    def test_required_gates_pass_when_exact_metrics_clear_thresholds(self) -> None:
        report = gate.check_readiness(
            dataset_audit={"fatal_error_count": 0, "warning_count": 2},
            exact_report={
                "summary": {
                    "exact_graph_fact_pass_rate": 0.82,
                    "unsupported_language_rate": 0.0,
                    "mean_id_recall": 0.91,
                    "mean_layer_recall": 0.86,
                    "mean_number_recall": 0.86,
                },
                "by_context_mode": {},
                "by_mixture_bucket": {},
            },
            trajectory_audit=None,
            args=_args(),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "move_to_dpo_trajectory_generation")

    def test_required_gate_fails_on_zero_exact_rate(self) -> None:
        report = gate.check_readiness(
            dataset_audit={"fatal_error_count": 0, "warning_count": 0},
            exact_report={
                "summary": {
                    "exact_graph_fact_pass_rate": 0.0,
                    "unsupported_language_rate": 0.0,
                    "mean_id_recall": 0.91,
                    "mean_layer_recall": 0.86,
                    "mean_number_recall": 0.86,
                },
                "by_context_mode": {},
                "by_mixture_bucket": {},
            },
            trajectory_audit=None,
            args=_args(),
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_required_gates"][0]["observed"], 0.0)


if __name__ == "__main__":
    unittest.main()
