import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from runtime.environment import RuntimeEnvironment
from scripts.audit_trajectory_run import AuditConfig, _final_summary_alignment, audit_run
from scripts.dpo_pair_loader_smoke import smoke_load_pairs
from scripts.generate_trajectories import TrajectoryGenerationConfig, generate_trajectories
from scripts.select_verification_tasks import (
    _complex_key,
    select_pilot_rows,
    select_smoke_task_ids,
    size_bin_for_task_row,
    source_for_task_row,
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


def _generate_small_run(out_dir: Path) -> None:
    generate_trajectories(
        task_rows=_task_rows(),
        out_dir=out_dir,
        environment=_build_environment(),
        config=TrajectoryGenerationConfig(max_steps=3, n_act=4, n_ver=2, seed=7),
    )


class VerificationGateTests(unittest.TestCase):
    def test_audit_alignment_treats_near_miss_as_partial_not_success(self) -> None:
        task_row = _task_rows()[0]
        summary = {
            "task_type": "recovery",
            "final_state": {
                "relationship_status": "validated_group",
                "predicted_groups": [
                    {
                        "group_id": "group_0",
                        "gene_ids": ["ENSG1", "ENSG2"],
                    }
                ],
            },
        }

        alignment = _final_summary_alignment(summary, task_row)

        self.assertIsNotNone(alignment)
        assert alignment is not None
        self.assertEqual(alignment["task_success_level"], "partial")
        self.assertFalse(alignment["task_success"])
        self.assertEqual(alignment["overlap_count"], 2)

    def test_audit_alignment_requires_validated_group_for_success(self) -> None:
        task_row = _task_rows()[0]
        summary = {
            "task_type": "recovery",
            "final_state": {
                "relationship_status": "partially_observed_group",
                "predicted_groups": [
                    {
                        "group_id": "group_0",
                        "gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                    }
                ],
            },
        }

        alignment = _final_summary_alignment(summary, task_row)

        self.assertIsNotNone(alignment)
        assert alignment is not None
        self.assertEqual(alignment["task_success_level"], "partial")
        self.assertFalse(alignment["task_success"])
        self.assertEqual(alignment["jaccard"], 1.0)

    def test_audit_alignment_honors_weak_evidence_downgrade(self) -> None:
        task_row = _task_rows()[0]
        summary = {
            "task_type": "recovery",
            "task_quality_failure_reasons": ["validated_group_weak_enrichment_support"],
            "final_state": {
                "relationship_status": "validated_group",
                "predicted_groups": [
                    {
                        "group_id": "group_0",
                        "gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                    }
                ],
            },
        }

        alignment = _final_summary_alignment(summary, task_row)

        self.assertIsNotNone(alignment)
        assert alignment is not None
        self.assertEqual(alignment["task_success_level"], "partial")
        self.assertFalse(alignment["task_success"])
        self.assertEqual(alignment["jaccard"], 1.0)

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

    def test_audit_reports_exact_success_and_exact_pair_gates(self) -> None:
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
                    min_recovery_exact_success_rate=1.1,
                    min_exact_membership_pair_rate=1.1,
                ),
            )

            self.assertFalse(report.ok)
            self.assertIn("recovery_exact_success_rate", report.metrics)
            self.assertIn("refinement_exact_success_rate", report.metrics)
            self.assertIn("exact_membership_pair_rate", report.metrics)
            self.assertIn("near_miss_missing_gene_count_mean", report.metrics)
            self.assertIn("near_miss_extra_gene_count_mean", report.metrics)
            self.assertIn("terminal_membership_metrics_exact_mismatch_count", report.metrics)
            self.assertIn("branch_membership_metrics_exact_mismatch_count", report.metrics)
            self.assertTrue(
                any(finding.code == "recovery_exact_success_rate_low" for finding in report.findings)
            )
            self.assertTrue(
                any(finding.code == "exact_membership_pair_rate_low" for finding in report.findings)
            )

    def test_audit_reports_exact_metric_status_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            _generate_small_run(out_dir)
            tasks_path = out_dir / "tasks.jsonl"
            tasks_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in _task_rows()),
                encoding="utf-8",
            )
            manifest_path = out_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["task_selection"] = {"tasks_path": str(tasks_path)}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            final_summaries_path = out_dir / "final_summaries.jsonl"
            rows = [
                json.loads(line)
                for line in final_summaries_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for row in rows:
                if row.get("task_type") != "recovery":
                    continue
                row["final_state"]["relationship_status"] = "partially_observed_group"
                row["final_state"]["predicted_groups"] = [
                    {
                        "group_id": "group_0",
                        "gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                        "gene_symbols": ["GENE1", "GENE2", "GENE3"],
                        "rationale": "Exact genes but verifier did not validate.",
                    }
                ]
                break
            final_summaries_path.write_text(
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

            self.assertTrue(report.ok, [finding.to_dict() for finding in report.findings])
            self.assertGreaterEqual(report.metrics["terminal_membership_metrics_exact_count"], 1)
            self.assertGreaterEqual(
                report.metrics["terminal_membership_metrics_exact_mismatch_count"],
                1,
            )
            self.assertIn(
                "recovery/partial/partially_observed_group",
                report.metrics["terminal_membership_metrics_exact_mismatch_by_task"],
            )

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

    def test_select_verification_tasks_recognizes_rwr_loe_module_family(self) -> None:
        row = {
            "task_id": "rwr_loe_module_000123.recovery.hard.graph",
            "provenance": {"source": "RWR_LOE_FULL_BRAIN"},
        }

        self.assertEqual(_complex_key(row["task_id"]), "rwr_loe_module_000123")
        self.assertEqual(source_for_task_row(row), "RWR_LOE_FULL_BRAIN")

    def test_select_verification_tasks_can_stratify_mixed_corpus_by_source(self) -> None:
        rows = []
        for source, prefix in (
            ("MENTOR_GW_DENDROGRAM", "gw_dendrogram_module_000001"),
            ("RWR_LOE_FULL_BRAIN", "rwr_loe_module_000001"),
        ):
            for task_type in ("explanation", "recovery"):
                rows.append(
                    {
                        "task_id": f"{prefix}.{task_type}.easy.graph",
                        "task_type": task_type,
                        "evidence_mode": "graph",
                        "difficulty": "easy",
                        "size_bin": "small",
                        "provenance": {"source": source},
                    }
                )

        pilot_rows = select_pilot_rows(
            rows,
            pilot_size=4,
            seed=13,
            stratify_by_difficulty=False,
            stratify_by_size_bin=False,
            stratify_by_source=True,
        )

        self.assertEqual(len(pilot_rows), 4)
        self.assertEqual(
            {
                (source_for_task_row(row), row["task_type"])
                for row in pilot_rows
            },
            {
                ("MENTOR_GW_DENDROGRAM", "explanation"),
                ("MENTOR_GW_DENDROGRAM", "recovery"),
                ("RWR_LOE_FULL_BRAIN", "explanation"),
                ("RWR_LOE_FULL_BRAIN", "recovery"),
            },
        )

        prefix_rows = select_pilot_rows(
            rows,
            pilot_size=2,
            seed=13,
            stratify_by_difficulty=False,
            stratify_by_size_bin=False,
            stratify_by_source=True,
        )

        self.assertEqual(
            {source_for_task_row(row) for row in prefix_rows},
            {"MENTOR_GW_DENDROGRAM", "RWR_LOE_FULL_BRAIN"},
        )

    def test_select_verification_tasks_can_filter_task_types(self) -> None:
        rows = []
        for task_type in ("explanation", "recovery", "refinement", "none"):
            for evidence_mode in ("graph", "minimal"):
                rows.append(
                    {
                        "task_id": f"gw_dendrogram_module_000001.{task_type}.easy.{evidence_mode}",
                        "task_type": task_type,
                        "evidence_mode": evidence_mode,
                        "difficulty": "easy",
                        "size_bin": "small",
                    }
                )

        smoke_ids = select_smoke_task_ids(
            rows,
            per_bucket=1,
            stratify_by_size_bin=False,
            task_types=("recovery", "refinement"),
        )
        pilot_rows = select_pilot_rows(
            rows,
            pilot_size=4,
            seed=17,
            stratify_by_difficulty=False,
            stratify_by_size_bin=False,
            task_types=("recovery", "refinement"),
        )

        self.assertEqual(
            {task_id.split(".", 2)[1] for task_id in smoke_ids},
            {"recovery", "refinement"},
        )
        self.assertEqual({row["task_type"] for row in pilot_rows}, {"recovery", "refinement"})
        self.assertEqual(len(pilot_rows), 4)


if __name__ == "__main__":
    unittest.main()
