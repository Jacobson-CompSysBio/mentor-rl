import json
import tempfile
import unittest
from pathlib import Path

from scripts.frontier_diagnostics import diagnose_frontier_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _branch(
    branch_id: str,
    *,
    predicted_gene_ids: list[str],
    ranked_gene_ids: list[str] | None = None,
    edit: dict | None = None,
    success_level: str = "partial",
) -> dict:
    metrics = (
        {"jaccard": 1.0, "precision": 1.0, "recall": 1.0}
        if success_level == "positive"
        else {"jaccard": 2.0 / 3.0, "precision": 1.0, "recall": 2.0 / 3.0}
    )
    metadata = {"task_type": "recovery"}
    if edit is not None:
        metadata["generator_backend"] = "deterministic_membership_edit"
        metadata["deterministic_membership_edit"] = edit
        metadata["candidate_frontier"] = edit.get("candidate_frontier", [])
    payload = {
        "ranked_genes": [
            {"gene_id": gene_id, "rank": index}
            for index, gene_id in enumerate(ranked_gene_ids or [], start=1)
        ]
    }
    return {
        "branch_id": branch_id,
        "actor_step": {
            "reasoning_text": "Run RWR.",
            "tool_action": {
                "tool_name": "rwr",
                "arguments": {"seed_genes": ["ENSG1", "ENSG2"], "top_k": 10},
                "call_id": branch_id + ".tool",
            },
        },
        "observation": {
            "status": "success",
            "call_id": branch_id + ".tool",
            "payload": payload,
            "provenance": {
                "tool_name": "rwr",
                "candidate_frontier": metadata.get("candidate_frontier", []),
            },
        },
        "verifier_step": {
            "updated_interpretation": {
                "mechanistic_claim": "toy",
                "main_evidence": "toy",
                "uncertainty": "",
                "next_subgoal": "",
            },
            "updated_state": {
                "user_anchors": {
                    "query_text": "recover toy",
                    "evidence": {"seed_gene_ids": ["ENSG1", "ENSG2"]},
                    "evidence_mode": "minimal",
                    "source_task_id": "toy.recovery.easy",
                },
                "relationship_status": "validated_group" if success_level == "positive" else "partially_observed_group",
                "predicted_groups": [
                    {"group_id": "group_0", "gene_ids": predicted_gene_ids, "gene_symbols": predicted_gene_ids}
                ],
                "evidence_log": [],
                "mechanistic_labels": [],
                "remaining_budget": 1,
                "continuation_state": "stop",
                "total_tool_call_count": 1,
                "invalid_tool_call_count": 0,
            },
            "continuation_decision": "stop",
            "verifier_notes": "",
        },
        "local_score": {
            "schema_score": 1.0,
            "complex_membership_delta": 0.0,
            "mechanistic_label_delta": 0.0,
            "mechanism_evidence_delta": 0.0,
            "mechanism_evidence_score": 0.0,
            "efficiency_penalty": 0.0,
            "total_score": 1.0 if success_level == "positive" else 0.5,
            "normalized_score": 1.0 if success_level == "positive" else 0.0,
            "score_metadata": {
                "complex": {"best_group_post": {"metrics": metrics}},
                "task_success": {
                    "task_success_level": success_level,
                    "metrics": metrics,
                },
            },
        },
        "metadata": metadata,
    }


class FrontierDiagnosticsTests(unittest.TestCase):
    def test_recovery_diagnostics_track_recall_preview_edit_and_exact_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            tasks_path = run_dir / "tasks.jsonl"
            _write_jsonl(
                tasks_path,
                [
                    {
                        "task_id": "toy.recovery.easy",
                        "task_type": "recovery",
                        "difficulty": "easy",
                        "evidence_mode": "minimal",
                        "hidden_target": {
                            "relationship_status": "validated_group",
                            "target_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                        },
                    }
                ],
            )
            (run_dir / "run_freeze.json").write_text(
                json.dumps({"task_selection": {"tasks_path": str(tasks_path)}}),
                encoding="utf-8",
            )
            edit = {
                "task_type": "recovery",
                "edit_kind": "recovery_add_single",
                "source_branch_id": "toy.step0.a0.v0",
                "source_tool_name": "rwr",
                "added_gene_ids": ["ENSG3"],
                "removed_gene_ids": [],
                "candidate_gene_ids": ["ENSG3"],
                "candidate_frontier": [{"gene_id": "ENSG3", "rank": 5}],
            }
            branch_pools_path = run_dir / "branch_pools.jsonl"
            _write_jsonl(
                branch_pools_path,
                [
                    {
                        "trajectory_id": "toy.recovery.easy.seed0",
                        "source_task_id": "toy.recovery.easy",
                        "task_type": "recovery",
                        "difficulty": "easy",
                        "evidence_mode": "minimal",
                        "step_index": 0,
                        "context": {
                            "state": {
                                "predicted_groups": [
                                    {"group_id": "group_0", "gene_ids": ["ENSG1", "ENSG2"]}
                                ]
                            }
                        },
                        "selected_branch_id": "toy.step0.edit",
                        "branches": [
                            _branch(
                                "toy.step0.a0.v0",
                                predicted_gene_ids=["ENSG1", "ENSG2"],
                                ranked_gene_ids=["ENSG1", "ENSG2", "ENSG4", "ENSG5", "ENSG3"],
                            ),
                            _branch(
                                "toy.step0.edit",
                                predicted_gene_ids=["ENSG1", "ENSG2", "ENSG3"],
                                ranked_gene_ids=["ENSG1", "ENSG2", "ENSG4", "ENSG5", "ENSG3"],
                                edit=edit,
                                success_level="positive",
                            ),
                        ],
                    }
                ],
            )
            with branch_pools_path.open("a", encoding="utf-8") as handle:
                handle.write("{bad json tail\n")
            _write_jsonl(
                run_dir / "preference_pairs_raw.jsonl",
                [
                    {
                        "trajectory_id": "toy.recovery.easy.seed0",
                        "provenance": {
                            "pair_category": "exact_recovery",
                            "chosen_exact_membership": True,
                            "rejected_exact_membership": False,
                        },
                    }
                ],
            )

            report = diagnose_frontier_run(run_dir=run_dir, prompt_preview_limit=2)

        aggregate = report["aggregate"]
        self.assertEqual(aggregate["branch_pool_parse_error_count"], 1)
        self.assertEqual(aggregate["recovery_missing_target_gene_count"], 1)
        self.assertEqual(aggregate["recovery_frontier_recalled_gene_count"], 1)
        self.assertEqual(aggregate["recovery_frontier_surfaced_at_preview_gene_count"], 0)
        self.assertEqual(aggregate["recovery_edit_frontier_gene_count"], 1)
        self.assertEqual(aggregate["recovery_added_to_selected_branch_gene_count"], 1)
        self.assertEqual(aggregate["exact_branch_count"], 1)
        self.assertEqual(aggregate["exact_pair_count_raw"], 1)
        task_report = report["by_task"][0]
        self.assertEqual(task_report["missing_target_gene_details"][0]["gene_id"], "ENSG3")
        self.assertEqual(task_report["missing_target_gene_details"][0]["rwr_best_rank"], 5.0)
        self.assertFalse(task_report["missing_target_gene_details"][0]["in_prompt_preview"])
        self.assertTrue(task_report["missing_target_gene_details"][0]["in_edit_frontier"])


if __name__ == "__main__":
    unittest.main()
