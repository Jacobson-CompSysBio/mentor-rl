import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from runtime.environment import RuntimeEnvironment
from scripts.audit_trajectory_run import (
    AuditConfig,
    _defer_alignment,
    _final_summary_alignment,
    audit_run,
)
from scripts.dpo_pair_loader_smoke import smoke_load_pairs
from scripts.generate_trajectories import TrajectoryGenerationConfig, generate_trajectories
from scripts.select_verification_tasks import (
    select_pilot_rows,
    select_smoke_task_ids,
    size_bin_for_task_row,
)
from utils.multiplex import Multiplex


def _build_environment() -> RuntimeEnvironment:
    multiplex = Multiplex()
    graph = nx.Graph()
    graph.add_nodes_from(["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"])
    graph.add_edge("ENSG1", "ENSG2", weight=1.0)
    graph.add_edge("ENSG2", "ENSG3", weight=0.9)
    multiplex.add_layer(graph, "ppi")
    return RuntimeEnvironment(multiplex=multiplex)


def _task_rows() -> list[dict]:
    return [
        {
            "task_id": "corum_complex_test.recovery.easy.contextual",
            "task_type": "recovery",
            "difficulty": "easy",
            "query_text": "Recover the shared group around ENSG1 and ENSG2.",
            "evidence_mode": "contextual",
            "visible_inputs": {
                "seed_gene_ids": ["ENSG1", "ENSG2"],
                "seed_gene_symbols": ["GENE1", "GENE2"],
                "context_text": "The seed genes appear in the same candidate module.",
                "graph_query_spec": None,
                "structured_annotations": None,
            },
            "hidden_target": {
                "relationship_status": "validated_group",
                "target_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                "target_gene_symbols": ["GENE1", "GENE2", "GENE3"],
            },
            "mechanism_labels": {
                "go_ids": ["GO:0000001"],
                "go_names": ["toy process"],
                "fcgs_ids": [],
                "fcgs_names": [],
                "primary_label": "toy process",
            },
        },
        {
            "task_id": "none_test.contextual",
            "task_type": "none",
            "difficulty": "complete",
            "query_text": "Do ENSG4 and ENSG5 support one shared mechanism?",
            "evidence_mode": "contextual",
            "visible_inputs": {
                "seed_gene_ids": ["ENSG4", "ENSG5"],
                "seed_gene_symbols": ["GENE4", "GENE5"],
                "context_text": "No curated shared-context note was attached to this pair.",
                "graph_query_spec": None,
                "structured_annotations": None,
            },
            "hidden_target": {
                "relationship_status": "insufficient_support",
                "target_gene_ids": None,
                "target_gene_symbols": None,
            },
            "mechanism_labels": None,
        },
    ]


def _generate_small_run(
    out_dir: Path,
    *,
    task_rows: list[dict] | None = None,
) -> Path:
    rows = task_rows or _task_rows()
    tasks_path = out_dir.parent / f"{out_dir.name}_tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    generate_trajectories(
        task_rows=rows,
        out_dir=out_dir,
        environment=_build_environment(),
        config=TrajectoryGenerationConfig(max_steps=3, n_act=4, n_ver=2, seed=7),
        task_selection={"tasks_path": str(tasks_path)},
    )
    return tasks_path


class VerificationGateTests(unittest.TestCase):
    def test_audit_accepts_structurally_valid_small_run_with_relaxed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

            self.assertTrue(report.ok, [finding.to_dict() for finding in report.findings])
            self.assertEqual(report.metrics["final_summary_count"], 2)
            self.assertIn("task_success_level_counts_by_task", report.metrics)
            self.assertIn("terminal_jaccard_mean", report.metrics)
            self.assertEqual(report.metrics["defer_count"], 0)
            self.assertEqual(report.metrics["expected_defer_count"], 0)
            self.assertEqual(report.metrics["correct_defer_count"], 0)
            self.assertEqual(report.metrics["false_defer_count"], 0)
            self.assertEqual(report.metrics["missed_defer_count"], 0)
            self.assertEqual(report.metrics["false_defer_rate"], 0.0)
            self.assertIsNone(report.metrics["defer_recall"])

    def test_audit_reports_false_defer_rate_and_defer_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            task_rows = _task_rows()
            task_rows[0]["hidden_target"]["requires_user_input"] = True
            _generate_small_run(out_dir, task_rows=task_rows)
            summaries_path = out_dir / "final_summaries.jsonl"
            rows = [
                json.loads(line)
                for line in summaries_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["requires_user_input"] = True
            rows[0]["defer_correct"] = True
            rows[0]["false_defer"] = False
            rows[0]["missed_defer"] = False
            rows[0]["final_state"]["continuation_state"] = "defer"
            rows[0]["final_state"]["termination_reason"] = "user_input_required"
            rows[1]["requires_user_input"] = False
            rows[1]["defer_correct"] = False
            rows[1]["false_defer"] = True
            rows[1]["missed_defer"] = False
            rows[1]["final_state"]["continuation_state"] = "defer"
            rows[1]["final_state"]["termination_reason"] = "user_input_required"
            summaries_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

        self.assertEqual(report.metrics["defer_count"], 2)
        self.assertEqual(report.metrics["expected_defer_count"], 1)
        self.assertEqual(report.metrics["correct_defer_count"], 1)
        self.assertEqual(report.metrics["false_defer_count"], 1)
        self.assertEqual(report.metrics["missed_defer_count"], 0)
        self.assertEqual(report.metrics["false_defer_rate"], 1.0)
        self.assertEqual(report.metrics["defer_recall"], 1.0)

    def test_audit_uses_source_task_defer_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)
            summaries_path = out_dir / "final_summaries.jsonl"
            rows = [
                json.loads(line)
                for line in summaries_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["requires_user_input"] = True
            rows[0]["defer_correct"] = True
            rows[0]["false_defer"] = False
            rows[0]["missed_defer"] = False
            rows[0]["final_state"]["continuation_state"] = "defer"
            rows[0]["final_state"]["termination_reason"] = "user_input_required"
            summaries_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

        self.assertEqual(report.metrics["expected_defer_count"], 0)
        self.assertEqual(report.metrics["correct_defer_count"], 0)
        self.assertEqual(report.metrics["false_defer_count"], 1)
        self.assertTrue(
            any(
                finding.code == "defer_summary_mismatch"
                for finding in report.findings
            )
        )

    def test_audit_rejects_an_unknown_source_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)
            summaries_path = out_dir / "final_summaries.jsonl"
            rows = [
                json.loads(line)
                for line in summaries_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["source_task_id"] = "unknown_task"
            rows[0]["requires_user_input"] = True
            summaries_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

        self.assertEqual(report.metrics["expected_defer_count"], 0)
        self.assertTrue(
            any(
                finding.code == "summary_source_task_missing"
                for finding in report.findings
            )
        )

    def test_audit_rejects_an_invalid_configured_task_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            tasks_path = _generate_small_run(out_dir)
            tasks_path.write_text("{invalid\n", encoding="utf-8")
            summaries_path = out_dir / "final_summaries.jsonl"
            rows = [
                json.loads(line)
                for line in summaries_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["requires_user_input"] = True
            summaries_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

        self.assertEqual(report.metrics["expected_defer_count"], 0)
        finding_codes = {finding.code for finding in report.findings}
        self.assertIn("invalid_jsonl", finding_codes)
        self.assertIn("task_source_empty", finding_codes)
        self.assertIn("summary_source_task_missing", finding_codes)

    def test_audit_limits_task_success_checks_to_defer_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)
            summaries_path = out_dir / "final_summaries.jsonl"
            rows = [
                json.loads(line)
                for line in summaries_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["final_state"]["relationship_status"] = "validated_group"
            rows[0]["final_state"]["predicted_groups"] = [
                {"gene_ids": ["ENSG1", "ENSG2", "ENSG3"]}
            ]
            rows[0]["final_state"]["evidence_log"] = []
            rows[0]["task_success"] = False
            rows[0]["task_success_level"] = "partial"
            summaries_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

        self.assertFalse(
            any(
                finding.code == "task_success_summary_mismatch"
                for finding in report.findings
            )
        )

    def test_audit_rejects_copied_task_success_for_a_correct_defer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            task_rows = _task_rows()
            task_rows[0]["hidden_target"]["requires_user_input"] = True
            _generate_small_run(out_dir, task_rows=task_rows)
            summaries_path = out_dir / "final_summaries.jsonl"
            rows = [
                json.loads(line)
                for line in summaries_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["final_state"]["continuation_state"] = "defer"
            rows[0]["final_state"]["termination_reason"] = "user_input_required"
            rows[0]["requires_user_input"] = True
            rows[0]["defer_correct"] = True
            rows[0]["false_defer"] = False
            rows[0]["missed_defer"] = False
            rows[0]["task_success"] = False
            rows[0]["task_success_level"] = "negative"
            summaries_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

        self.assertEqual(report.metrics["correct_defer_count"], 1)
        self.assertGreaterEqual(
            report.metrics["task_success_level_counts"].get("positive", 0),
            1,
        )
        self.assertTrue(
            any(
                finding.code == "task_success_summary_mismatch"
                for finding in report.findings
            )
        )

    def test_audit_rejects_defer_with_wrong_termination_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            task_rows = _task_rows()
            task_rows[0]["hidden_target"]["requires_user_input"] = True
            _generate_small_run(out_dir, task_rows=task_rows)
            summaries_path = out_dir / "final_summaries.jsonl"
            rows = [
                json.loads(line)
                for line in summaries_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["requires_user_input"] = True
            rows[0]["defer_correct"] = False
            rows[0]["false_defer"] = False
            rows[0]["missed_defer"] = True
            rows[0]["final_state"]["continuation_state"] = "defer"
            rows[0]["final_state"]["termination_reason"] = "model_stop"
            summaries_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

        self.assertEqual(report.metrics["correct_defer_count"], 0)
        self.assertEqual(report.metrics["missed_defer_count"], 1)
        self.assertTrue(
            any(
                finding.code == "defer_termination_invalid"
                for finding in report.findings
            )
        )

    def test_audit_alignment_applies_correct_false_and_missed_defer(self) -> None:
        task_row = _task_rows()[0]
        task_row["hidden_target"]["requires_user_input"] = True
        final_state = {
            "relationship_status": "insufficient_support",
            "predicted_groups": [],
            "continuation_state": "defer",
            "termination_reason": "user_input_required",
        }
        summary = {"task_type": "recovery", "final_state": final_state}

        correct = _final_summary_alignment(summary, task_row)

        self.assertIsNotNone(correct)
        self.assertTrue(correct["task_success"])
        self.assertEqual(correct["task_success_level"], "positive")
        self.assertTrue(correct["correct_defer"])

        task_row["hidden_target"]["requires_user_input"] = False
        false_defer = _final_summary_alignment(summary, task_row)

        self.assertIsNotNone(false_defer)
        self.assertFalse(false_defer["task_success"])
        self.assertEqual(false_defer["task_success_level"], "negative")
        self.assertTrue(false_defer["false_defer"])

        task_row["hidden_target"]["requires_user_input"] = True
        final_state["continuation_state"] = "stop"
        final_state["termination_reason"] = "model_stop"
        missed = _final_summary_alignment(summary, task_row)

        self.assertIsNotNone(missed)
        self.assertFalse(missed["task_success"])
        self.assertEqual(missed["task_success_level"], "negative")
        self.assertTrue(missed["missed_defer"])

    def test_defer_alignment_requires_the_terminal_pair(self) -> None:
        task_row = _task_rows()[0]
        task_row["hidden_target"]["requires_user_input"] = True
        summary = {
            "final_state": {
                "continuation_state": "defer",
                "termination_reason": "model_stop",
            }
        }

        alignment = _defer_alignment(summary, task_row)

        self.assertFalse(alignment["valid_defer"])
        self.assertFalse(alignment["correct_defer"])
        self.assertTrue(alignment["missed_defer"])

    def test_audit_can_downgrade_tie_rate_for_dpo_pair_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=-1.0,
                    max_top_tie_rate=1.0,
                    tie_rate_severity="warning",
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

            self.assertTrue(report.ok, [finding.to_dict() for finding in report.findings])
            self.assertTrue(any(finding.code == "all_tie_rate_high" for finding in report.findings))
            self.assertTrue(
                all(
                    finding.severity == "warning"
                    for finding in report.findings
                    if finding.code == "all_tie_rate_high"
                )
            )

    def test_audit_reports_large_scale_pair_quality_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                    max_selected_no_tool_rate=-1.0,
                    max_mechanism_label_only_pair_rate=-1.0,
                ),
            )

            self.assertFalse(report.ok)
            self.assertIn("selected_no_tool_rate", report.metrics)
            self.assertIn("preference_pair_category_counts", report.metrics)
            self.assertTrue(any(finding.code == "selected_no_tool_rate_high" for finding in report.findings))

    def test_audit_reports_rwr_hpc_and_weak_evidence_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                    min_selected_rwr_hpc_tool_rate=1.1,
                    min_rwr_hpc_candidate_rate=1.1,
                    min_rwr_hpc_supported_pair_rate=1.1,
                    max_rwr_hpc_observation_error_rate=0.0,
                    max_validated_weak_evidence_rate=0.0,
                ),
            )

            self.assertFalse(report.ok)
            self.assertIn("selected_rwr_hpc_tool_rate", report.metrics)
            self.assertIn("rwr_hpc_candidate_rate", report.metrics)
            self.assertIn("rwr_hpc_observation_error_rate", report.metrics)
            self.assertIn("rwr_hpc_cache_hit_rate", report.metrics)
            self.assertIn("rwr_hpc_supported_pair_rate", report.metrics)
            self.assertIn("validated_weak_evidence_rate", report.metrics)
            self.assertTrue(any(finding.code == "selected_rwr_hpc_tool_rate_low" for finding in report.findings))

    def test_audit_rejects_hidden_supervision_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)
            branch_pool_path = out_dir / "branch_pools.jsonl"
            rows = [json.loads(line) for line in branch_pool_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["hidden_target"] = {"target_gene_ids": ["ENSG1"]}
            branch_pool_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_run(
                out_dir,
                AuditConfig(
                    max_all_tie_rate=1.0,
                    max_top_tie_rate=1.0,
                    min_balanced_pair_bins=0,
                    require_pairs=False,
                    required_task_types=("recovery", "none"),
                    required_evidence_modes=("contextual",),
                ),
            )

            self.assertFalse(report.ok)
            self.assertTrue(any(finding.code == "blocked_artifact_key" for finding in report.findings))

    def test_dpo_pair_loader_smoke_parses_generated_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)

            report = smoke_load_pairs(out_dir / "preference_pairs.jsonl", max_pairs=4, min_pairs=0)

            self.assertTrue(report.ok, report.errors)
            self.assertIn("pairs_loaded", report.metrics)

    def test_select_verification_tasks_covers_buckets(self) -> None:
        rows = []
        for complex_index in range(2):
            for task_type in ("explanation", "recovery"):
                for evidence_mode in ("contextual", "graph"):
                    rows.append(
                        {
                            "task_id": f"corum_complex_{complex_index:05d}.{task_type}.easy.{evidence_mode}",
                            "task_type": task_type,
                            "evidence_mode": evidence_mode,
                        }
                    )

        smoke_ids = select_smoke_task_ids(rows, per_bucket=1)
        pilot_rows = select_pilot_rows(rows, pilot_size=4)

        self.assertEqual(len(smoke_ids), 4)
        self.assertEqual(len(pilot_rows), 4)
        self.assertEqual(
            {(row["task_type"], row["evidence_mode"]) for row in pilot_rows},
            {
                ("explanation", "contextual"),
                ("explanation", "graph"),
                ("recovery", "contextual"),
                ("recovery", "graph"),
            },
        )

    def test_select_verification_tasks_stratifies_pilot_by_difficulty(self) -> None:
        rows = []
        for complex_index in range(6):
            for difficulty in ("easy", "medium", "hard"):
                rows.append(
                    {
                        "task_id": f"corum_complex_{complex_index:05d}.recovery.{difficulty}.graph",
                        "task_type": "recovery",
                        "evidence_mode": "graph",
                        "difficulty": difficulty,
                    }
                )

        pilot_rows = select_pilot_rows(rows, pilot_size=6, seed=7)

        difficulty_counts = {}
        complex_ids = set()
        for row in pilot_rows:
            difficulty_counts[row["difficulty"]] = difficulty_counts.get(row["difficulty"], 0) + 1
            complex_ids.add(row["task_id"].split(".", 1)[0])

        self.assertEqual(difficulty_counts, {"easy": 2, "hard": 2, "medium": 2})
        self.assertGreater(len(complex_ids), 1)

    def test_select_verification_tasks_stratifies_by_dendrogram_size_bin(self) -> None:
        rows = []
        for size, gene_count in (("small", 6), ("medium", 12), ("large", 20)):
            for task_type in ("explanation", "recovery"):
                rows.append(
                    {
                        "task_id": f"gw_dendrogram_module_{size}.{task_type}.easy.graph",
                        "task_type": task_type,
                        "evidence_mode": "graph",
                        "difficulty": "easy",
                        "hidden_target": {
                            "relationship_status": "validated_group",
                            "target_gene_ids": [f"ENSG{size}{index}" for index in range(gene_count)],
                        },
                    }
                )

        pilot_rows = select_pilot_rows(rows, pilot_size=6, seed=11)

        self.assertEqual({size_bin_for_task_row(row) for row in pilot_rows}, {"small", "medium", "large"})
        self.assertEqual(len(pilot_rows), 6)


if __name__ == "__main__":
    unittest.main()
