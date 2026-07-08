import json
import tempfile
import unittest
from pathlib import Path

from runtime import (
    ActorStep,
    CandidateBranch,
    ContinuationState,
    GeneGroup,
    Interpretation,
    LocalScoreBreakdown,
    PreferenceDifficulty,
    PreferencePair,
    RelationshipStatus,
    SharedPrefixContext,
    TaskType,
    VerifierStep,
    initialize_state_from_corum_task,
    replace_predicted_groups,
)
from scripts.export_training_records import export_training_records


def _task_row() -> dict:
    return {
        "task_id": "corum_complex_test.recovery.easy.contextual",
        "task_type": "recovery",
        "difficulty": "easy",
        "query_text": "Recover the shared group around ENSG1 and ENSG2.",
        "evidence_mode": "contextual",
        "visible_inputs": {
            "seed_gene_ids": ["ENSG1", "ENSG2"],
            "seed_gene_symbols": ["GENE1", "GENE2"],
            "context_text": "The seed genes appear in the same candidate module.",
        },
        "hidden_target": {
            "relationship_status": "validated_group",
            "target_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
            "target_gene_symbols": ["GENE1", "GENE2", "GENE3"],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _branch(branch_id: str, state, gene_ids: list[str], *, normalized_score: float) -> CandidateBranch:
    updated_state = replace_predicted_groups(
        state,
        [
            GeneGroup(
                group_id=branch_id,
                gene_ids=gene_ids,
                gene_symbols=[],
                rationale="Exporter test branch.",
            )
        ],
        relationship_status=RelationshipStatus.VALIDATED_GROUP,
    )
    updated_state.continuation_state = ContinuationState.STOP
    return CandidateBranch(
        branch_id=branch_id,
        actor_step=ActorStep(reasoning_text="Exporter test branch.", tool_action=None),
        verifier_step=VerifierStep(
            updated_interpretation=Interpretation(
                mechanistic_claim="Exporter test claim.",
                main_evidence="Exporter test evidence.",
                uncertainty="",
                next_subgoal="",
            ),
            updated_state=updated_state,
            continuation_decision=ContinuationState.STOP,
            verifier_notes="Exporter test verifier.",
        ),
        local_score=LocalScoreBreakdown(
            schema_score=1.0,
            complex_membership_delta=1.0,
            mechanistic_label_delta=0.5,
            efficiency_penalty=0.0,
            total_score=normalized_score * 10.0,
            normalized_score=normalized_score,
        ),
    )


def _preference_pair() -> PreferencePair:
    task_row = _task_row()
    interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
    chosen = _branch("chosen_exact", state, ["ENSG1", "ENSG2", "ENSG3"], normalized_score=0.25)
    rejected = _branch("rejected_partial", state, ["ENSG1", "ENSG2"], normalized_score=0.99)
    context = SharedPrefixContext(
        query_text=task_row["query_text"],
        user_evidence=task_row["visible_inputs"],
        interpretation=interpretation,
        state=state,
        source_task_id=task_row["task_id"],
    )
    return PreferencePair(
        pair_id="pair_exact_over_partial",
        context=context,
        chosen=chosen,
        rejected=rejected,
        task_type=TaskType.RECOVERY,
        difficulty_bin=PreferenceDifficulty.EASY,
        decision_step=0,
        raw_score_chosen=2.5,
        raw_score_rejected=9.9,
        normalized_score_chosen=0.25,
        normalized_score_rejected=0.99,
        score_margin=0.0,
        source_task_id=task_row["task_id"],
        trajectory_id="traj_export",
        trajectory_seed=0,
        evidence_mode="contextual",
        provenance={
            "pair_category": "exact_over_partial",
            "raw_score_delta": -7.4,
            "normalized_score_delta": -0.74,
            "chosen_candidate_frontier": [],
            "rejected_candidate_frontier": [],
        },
    )


class ExportTrainingRecordsTests(unittest.TestCase):
    def test_export_training_records_writes_dpo_and_exact_sft_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            out_dir = Path(tmpdir) / "export"
            run_dir.mkdir()
            pair = _preference_pair()

            (run_dir / "manifest.json").write_text(
                json.dumps({"artifacts": {"preference_pair_count": 1}}),
                encoding="utf-8",
            )
            (run_dir / "progress.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            _write_jsonl(run_dir / "preference_pairs.jsonl", [pair.to_dict()])
            _write_jsonl(run_dir / "preference_pairs_raw.jsonl", [pair.to_dict()])
            _write_jsonl(run_dir / "branch_pools.jsonl", [{"trajectory_id": "traj_export", "branches": []}])
            _write_jsonl(
                run_dir / "trajectory_turns.jsonl",
                [
                    {
                        "trajectory_id": "traj_recovery_positive",
                        "branch": {"branch_id": "normal_terminal", "metadata": {}},
                    },
                    {
                        "trajectory_id": "traj_recovery_scaffolded",
                        "branch": {
                            "branch_id": "scaffold_terminal",
                            "metadata": {
                                "deterministic_membership_edit": {
                                    "requires_model_validation": True,
                                    "validation_status": "pending_model_or_tool_validation",
                                }
                            },
                        },
                    },
                ],
            )
            _write_jsonl(
                run_dir / "final_summaries.jsonl",
                [
                    {
                        "trajectory_id": "traj_recovery_positive",
                        "source_task_id": "recovery_positive",
                        "task_type": "recovery",
                        "task_success_level": "positive",
                        "selected_branch_ids": ["normal_terminal"],
                        "rendered_summary": "Recovered the exact module.",
                    },
                    {
                        "trajectory_id": "traj_recovery_scaffolded",
                        "source_task_id": "recovery_scaffolded",
                        "task_type": "recovery",
                        "task_success_level": "positive",
                        "selected_branch_ids": ["scaffold_terminal"],
                        "rendered_summary": "Controller scaffold found the exact module.",
                    },
                    {
                        "trajectory_id": "traj_refinement_partial",
                        "source_task_id": "refinement_partial",
                        "task_type": "refinement",
                        "task_success_level": "partial",
                        "selected_branch_ids": ["partial_terminal"],
                        "rendered_summary": "Partially refined the module.",
                    },
                    {
                        "trajectory_id": "traj_explanation_positive",
                        "source_task_id": "explanation_positive",
                        "task_type": "explanation",
                        "task_success_level": "positive",
                        "rendered_summary": "Explained the module.",
                    },
                ],
            )

            manifest = export_training_records(run_dir, out_dir)

            self.assertEqual(manifest["dpo_record_count"], 1)
            self.assertEqual(manifest["sft_record_count"], 1)
            self.assertEqual(manifest["sft_scaffolded_membership_edit_excluded_count"], 1)
            dpo_records = _read_jsonl(out_dir / "dpo_records.jsonl")
            self.assertEqual(dpo_records[0]["metadata"]["pair_category"], "exact_over_partial")
            self.assertEqual(dpo_records[0]["metadata"]["score_margin"], 0.0)
            self.assertLess(dpo_records[0]["metadata"]["normalized_score_delta"], 0.0)
            sft_records = _read_jsonl(out_dir / "sft_exact_trajectories.jsonl")
            self.assertEqual(sft_records[0]["answer"], "Recovered the exact module.")
            self.assertNotIn("hidden_target", json.dumps(dpo_records + sft_records))

            include_partial_dir = Path(tmpdir) / "export_include_partial"
            partial_manifest = export_training_records(
                run_dir,
                include_partial_dir,
                include_partial=True,
            )
            self.assertEqual(partial_manifest["sft_record_count"], 2)
            self.assertEqual(partial_manifest["sft_scaffolded_membership_edit_excluded_count"], 1)


if __name__ == "__main__":
    unittest.main()
