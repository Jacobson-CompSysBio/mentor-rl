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
    MechanisticLabel,
    RelationshipStatus,
    StructuredState,
    ToolAction,
    ToolObservation,
    ToolObservationStatus,
    TrajectoryTurn,
    UserAnchors,
    VerifierStep,
)
from scripts.render_trajectory_review import (
    load_branch_pools,
    load_summaries,
    load_turns,
    render_html,
    render_markdown,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _toy_turn() -> TrajectoryTurn:
    anchors = UserAnchors(
        query_text="Explain the shared mechanism for ENSG1 and ENSG2.",
        evidence={"seed_gene_ids": ["ENSG1", "ENSG2"], "seed_gene_symbols": ["G1", "G2"]},
        evidence_mode="minimal",
        source_task_id="toy.explanation.minimal",
    )
    prior_state = StructuredState(
        user_anchors=anchors,
        relationship_status=RelationshipStatus.UNKNOWN,
        predicted_groups=[GeneGroup(group_id="group_0", gene_ids=["ENSG1", "ENSG2"], gene_symbols=["G1", "G2"])],
        evidence_log=[],
        mechanistic_labels=[],
        remaining_budget=2,
        continuation_state=ContinuationState.CONTINUE,
    )
    updated_interpretation = Interpretation(
        mechanistic_claim="The genes support toy process.",
        main_evidence="Enrichment found GO:0000001 toy process.",
        uncertainty="Low coverage in this toy fixture.",
        next_subgoal="Stop.",
    )
    updated_state = StructuredState(
        user_anchors=anchors,
        relationship_status=RelationshipStatus.VALIDATED_GROUP,
        predicted_groups=[GeneGroup(group_id="group_0", gene_ids=["ENSG1", "ENSG2"], gene_symbols=["G1", "G2"])],
        evidence_log=[],
        mechanistic_labels=[
            MechanisticLabel(
                label_source="go",
                label_name="toy process",
                label_id="GO:0000001",
                evidence_ids=["toy.step0.actor0"],
            )
        ],
        remaining_budget=1,
        continuation_state=ContinuationState.STOP,
        total_tool_call_count=1,
    )
    branch = CandidateBranch(
        branch_id="toy.explanation.minimal.seed7.step0.a0.v0",
        actor_step=ActorStep(
            reasoning_text="Run enrichment before claiming a mechanism.",
            tool_action=ToolAction(
                tool_name="enrich_gene_set",
                arguments={"genes": ["ENSG1", "ENSG2"]},
                call_id="toy.step0.actor0",
            ),
        ),
        observation=ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            provenance={"tool_name": "enrich_gene_set", "source": "cache"},
            call_id="toy.step0.actor0",
            payload={
                "query_gene_ids": ["ENSG1", "ENSG2"],
                "results": [{"native": "GO:0000001", "name": "toy process", "p_value": 0.01}],
            },
        ),
        verifier_step=VerifierStep(
            updated_interpretation=updated_interpretation,
            updated_state=updated_state,
            continuation_decision=ContinuationState.STOP,
            verifier_notes="Grounded in enrichment evidence.",
        ),
        local_score=LocalScoreBreakdown(
            schema_score=1.0,
            complex_membership_delta=0.5,
            mechanistic_label_delta=0.4,
            efficiency_penalty=0.0,
            total_score=1.9,
            normalized_score=1.0,
            mechanism_evidence_delta=0.7,
            mechanism_evidence_score=0.7,
        ),
    )
    return TrajectoryTurn(
        trajectory_id="toy.explanation.minimal.seed7",
        step_index=0,
        prior_interpretation=Interpretation("", "Seed genes are visible.", "", "Inspect evidence."),
        prior_state=prior_state,
        branch=branch,
        selected=True,
        finding_text="[validated_group] enrichment supports toy process.",
    )


class RenderTrajectoryReviewTests(unittest.TestCase):
    def test_render_markdown_includes_step_details_and_mermaid_graph(self) -> None:
        turn = _toy_turn()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            turns_path = run_dir / "trajectory_turns.jsonl"
            summaries_path = run_dir / "final_summaries.jsonl"
            branch_pools_path = run_dir / "branch_pools.jsonl"
            alt_branch = turn.branch.to_dict()
            alt_branch["branch_id"] = "toy.explanation.minimal.seed7.step0.a1.v0"
            alt_branch["actor_step"]["tool_action"] = None
            alt_branch["local_score"]["total_score"] = 0.5
            _write_jsonl(turns_path, [turn.to_dict()])
            _write_jsonl(
                branch_pools_path,
                [
                    {
                        "trajectory_id": turn.trajectory_id,
                        "step_index": 0,
                        "selected_branch_id": turn.branch.branch_id,
                        "branches": [turn.branch.to_dict(), alt_branch],
                    }
                ],
            )
            _write_jsonl(
                summaries_path,
                [
                    {
                        "trajectory_id": turn.trajectory_id,
                        "source_task_id": "toy.explanation.minimal",
                        "task_type": "explanation",
                        "evidence_mode": "minimal",
                        "difficulty": "complete",
                        "step_count": 1,
                        "finding_count": 1,
                        "terminal_reward": 2.0,
                        "terminal_absolute_complex_score": 1.0,
                        "terminal_mechanism_evidence_score": 0.7,
                        "final_interpretation": turn.branch.verifier_step.updated_interpretation.to_dict(),
                        "final_state": turn.branch.verifier_step.updated_state.to_dict(),
                    }
                ],
            )

            turns_by_trajectory = load_turns(turns_path)
            summaries = load_summaries(summaries_path)
            branch_pools = load_branch_pools(branch_pools_path)
            markdown = render_markdown(
                turns_by_trajectory=turns_by_trajectory,
                summaries=summaries,
                trajectory_ids=[turn.trajectory_id],
                source_label=str(run_dir),
                max_text_chars=500,
                branch_pools=branch_pools,
                max_unselected_per_step=1,
            )
            html = render_html(
                turns_by_trajectory=turns_by_trajectory,
                summaries=summaries,
                trajectory_ids=[turn.trajectory_id],
                source_label=str(run_dir),
                branch_pools=branch_pools,
                task_rows={
                    "toy.explanation.minimal": {
                        "task_id": "toy.explanation.minimal",
                        "task_type": "explanation",
                        "query_text": "Explain the shared mechanism for ENSG1 and ENSG2.",
                        "hidden_target": {
                            "relationship_status": "validated_group",
                            "target_gene_ids": ["ENSG1", "ENSG2"],
                        },
                        "mechanism_labels": None,
                    }
                },
                max_unselected_per_step=1,
                max_text_chars=500,
            )

        self.assertIn("# Trajectory Review", markdown)
        self.assertIn("```mermaid", markdown)
        self.assertIn("Step 0 | Selected", markdown)
        self.assertIn("Step 0 | Alt 1", markdown)
        self.assertIn("Top Unselected Alternatives", markdown)
        self.assertIn("enrich_gene_set", markdown)
        self.assertIn("GO:0000001", markdown)
        self.assertIn("toy process", markdown)
        self.assertIn("mechanism-evidence", markdown)
        self.assertIn("[validated_group] enrichment supports toy process.", markdown)
        self.assertIn("<!doctype html>", html)
        self.assertIn("Interactive Trajectory Graph Review", html)
        self.assertIn("selected branch", html)
        self.assertIn("unselected candidate", html)
        self.assertIn("Input Visible To Model", html)
        self.assertIn("Hidden Training Target", html)
        self.assertIn("Final Output", html)
        self.assertIn("Alignment Summary", html)
        self.assertIn("Reasoning And Verification Details", html)
        self.assertIn("mermaid", html)


if __name__ == "__main__":
    unittest.main()
