import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from runtime.environment import RuntimeEnvironment
from scripts.audit_trajectory_run import AuditConfig, audit_run
from scripts.dpo_pair_loader_smoke import smoke_load_pairs
from scripts.generate_trajectories import TrajectoryGenerationConfig, generate_trajectories
from scripts.select_verification_tasks import select_pilot_rows, select_smoke_task_ids
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


if __name__ == "__main__":
    unittest.main()
