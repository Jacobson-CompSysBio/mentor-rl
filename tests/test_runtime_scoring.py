import unittest

from runtime.scoring import score_candidate_branch, score_terminal_trajectory
from runtime.schemas import (
    ActorStep,
    CandidateBranch,
    ContinuationState,
    EvidenceRecord,
    EvidenceSourceType,
    GeneGroup,
    Interpretation,
    LabelSource,
    LocalScoreBreakdown,
    MechanisticLabel,
    RelationshipStatus,
    TerminationReason,
    ToolAction,
    ToolObservation,
    ToolObservationStatus,
    VerifierStep,
)
from runtime.state import (
    append_evidence_record,
    initialize_state_from_corum_task,
    replace_mechanistic_labels,
    replace_predicted_groups,
)


def _clone_state(state):
    return state.from_dict(state.to_dict())


def _placeholder_score() -> LocalScoreBreakdown:
    return LocalScoreBreakdown(
        schema_score=0.0,
        complex_membership_delta=0.0,
        mechanistic_label_delta=0.0,
        efficiency_penalty=0.0,
        total_score=0.0,
    )


def _make_branch(
    branch_id: str,
    interpretation: Interpretation,
    updated_state,
    *,
    tool_action: ToolAction | None = None,
    observation: ToolObservation | None = None,
) -> CandidateBranch:
    return CandidateBranch(
        branch_id=branch_id,
        actor_step=ActorStep(
            reasoning_text="Take the next deterministic step.",
            tool_action=tool_action,
        ),
        observation=observation,
        verifier_step=VerifierStep(
            updated_interpretation=Interpretation.from_dict(interpretation.to_dict()),
            updated_state=updated_state,
            continuation_decision=updated_state.continuation_state,
            verifier_notes="Updated after reviewing the branch result.",
        ),
        local_score=_placeholder_score(),
    )


def _recovery_task() -> dict:
    return {
        "task_id": "corum_complex_00004.recovery.easy.contextual",
        "task_type": "recovery",
        "query_text": "Recover the full mechanistic group for CREBBP, EP300, and NCOA3.",
        "evidence_mode": "contextual",
        "visible_inputs": {
            "seed_gene_ids": ["ENSG_CREBBP", "ENSG_EP300", "ENSG_NCOA3"],
            "seed_gene_symbols": ["CREBBP", "EP300", "NCOA3"],
            "context_text": "ACTR binds nuclear receptors and stimulates transcription.",
            "graph_query_spec": None,
            "structured_annotations": None,
        },
        "hidden_target": {
            "relationship_status": "validated_group",
            "target_gene_ids": ["ENSG_CREBBP", "ENSG_EP300", "ENSG_KAT2B", "ENSG_NCOA3"],
            "target_gene_symbols": ["CREBBP", "EP300", "KAT2B", "NCOA3"],
        },
        "mechanism_labels": {
            "go_ids": ["GO:0004402", "GO:0004879"],
            "go_names": ["histone acetyltransferase activity", "nuclear receptor activity"],
            "fcgs_ids": [],
            "fcgs_names": [],
            "primary_label": "histone acetyltransferase activity",
        },
    }


def _explanation_task() -> dict:
    return {
        "task_id": "corum_complex_00001.explanation.complete.contextual",
        "task_type": "explanation",
        "query_text": "What mechanism connects HDAC4 and BCL6?",
        "evidence_mode": "contextual",
        "visible_inputs": {
            "seed_gene_ids": ["ENSG_HDAC4", "ENSG_BCL6"],
            "seed_gene_symbols": ["HDAC4", "BCL6"],
            "context_text": "BCL6 recruits histone deacetylases during transcriptional repression.",
            "graph_query_spec": None,
            "structured_annotations": None,
        },
        "hidden_target": {
            "relationship_status": "validated_group",
            "target_gene_ids": ["ENSG_HDAC4", "ENSG_BCL6"],
            "target_gene_symbols": ["HDAC4", "BCL6"],
        },
        "mechanism_labels": {
            "go_ids": ["GO:0004407"],
            "go_names": ["histone deacetylase activity"],
            "fcgs_ids": [],
            "fcgs_names": [],
            "primary_label": "histone deacetylase activity",
        },
    }


def _none_task() -> dict:
    return {
        "task_id": "none.matched_to.example",
        "task_type": "none",
        "query_text": "Do PIK3CB and STAT6 support a shared curated mechanism?",
        "evidence_mode": "contextual",
        "visible_inputs": {
            "seed_gene_ids": ["ENSG_PIK3CB", "ENSG_STAT6"],
            "seed_gene_symbols": ["PIK3CB", "STAT6"],
            "context_text": "This seed set was flagged for follow-up, but no shared mechanism was curated.",
            "graph_query_spec": None,
            "structured_annotations": None,
        },
        "hidden_target": {
            "relationship_status": "insufficient_support",
            "target_gene_ids": None,
            "target_gene_symbols": None,
        },
        "mechanism_labels": None,
    }


def _no_label_explanation_task() -> dict:
    task = _explanation_task()
    task["mechanism_labels"] = None
    return task


class RuntimeScoringTests(unittest.TestCase):
    def test_recovery_branch_gets_positive_complex_delta(self) -> None:
        interpretation, prior_state = initialize_state_from_corum_task(_recovery_task(), max_budget=4)
        updated_state = replace_predicted_groups(
            prior_state,
            predicted_groups=[
                GeneGroup(
                    group_id="group_0",
                    gene_ids=["ENSG_CREBBP", "ENSG_EP300", "ENSG_KAT2B", "ENSG_NCOA3"],
                    gene_symbols=["CREBBP", "EP300", "KAT2B", "NCOA3"],
                    rationale="Add the missing acetyltransferase member.",
                )
            ],
            relationship_status=RelationshipStatus.VALIDATED_GROUP,
        )
        branch = _make_branch("recovery_good", interpretation, updated_state)

        score = score_candidate_branch(
            _recovery_task(),
            prior_state,
            branch,
            step_index=1,
            max_steps=6,
        )

        self.assertEqual(score.schema_score, 1.0)
        self.assertGreater(score.complex_membership_delta, 0.0)
        self.assertEqual(score.mechanistic_label_delta, 0.0)
        self.assertEqual(score.score_metadata["complex"]["best_group_post"]["metrics"]["recall"], 1.0)
        self.assertEqual(score.score_metadata["task_profile"]["recall_weight"], 0.6)

    def test_explanation_branch_rewards_correct_mechanistic_label(self) -> None:
        interpretation, prior_state = initialize_state_from_corum_task(_explanation_task(), max_budget=4)
        updated_state = replace_mechanistic_labels(
            prior_state,
            [
                MechanisticLabel(
                    label_source=LabelSource.GO,
                    label_id="GO:0004407",
                    label_name="histone deacetylase activity",
                    evidence_ids=[],
                )
            ],
        )
        branch = _make_branch("explanation_label", interpretation, updated_state)

        score = score_candidate_branch(
            _explanation_task(),
            prior_state,
            branch,
            step_index=0,
            max_steps=6,
        )

        self.assertEqual(score.schema_score, 1.0)
        self.assertAlmostEqual(score.complex_membership_delta, 0.0)
        self.assertEqual(score.mechanistic_label_delta, 1.0)
        self.assertEqual(score.score_metadata["mechanistic"]["matched_count_post"], 1)

    def test_none_branch_rewards_abstention(self) -> None:
        interpretation, prior_state = initialize_state_from_corum_task(_none_task(), max_budget=4)
        updated_state = replace_predicted_groups(
            prior_state,
            predicted_groups=[],
            relationship_status=RelationshipStatus.INSUFFICIENT_SUPPORT,
        )
        branch = _make_branch("none_abstain", interpretation, updated_state)

        score = score_candidate_branch(
            _none_task(),
            prior_state,
            branch,
            step_index=1,
            max_steps=6,
        )

        self.assertEqual(score.schema_score, 1.0)
        self.assertGreater(score.complex_membership_delta, 0.0)

    def test_terminal_score_rewards_positive_recovery_and_records_termination(self) -> None:
        _, initial_state = initialize_state_from_corum_task(_recovery_task(), max_budget=4)
        final_state = replace_predicted_groups(
            initial_state,
            predicted_groups=[
                GeneGroup(
                    group_id="group_0",
                    gene_ids=["ENSG_CREBBP", "ENSG_EP300", "ENSG_KAT2B", "ENSG_NCOA3"],
                    gene_symbols=["CREBBP", "EP300", "KAT2B", "NCOA3"],
                    rationale="Recovered the full group before stopping.",
                )
            ],
            relationship_status=RelationshipStatus.VALIDATED_GROUP,
        )
        final_state = replace_mechanistic_labels(
            final_state,
            [
                MechanisticLabel(
                    label_source=LabelSource.GO,
                    label_id="GO:0004402",
                    label_name="histone acetyltransferase activity",
                    evidence_ids=[],
                )
            ],
        )
        final_state.continuation_state = ContinuationState.STOP
        final_state.termination_reason = TerminationReason.MODEL_STOP

        score = score_terminal_trajectory(
            _recovery_task(),
            initial_state,
            final_state,
            step_count=2,
            max_steps=4,
        )

        self.assertEqual(score["schema_score"], 1.0)
        self.assertGreater(score["absolute_complex_score"], 0.0)
        self.assertGreater(score["complex_delta"], 0.0)
        self.assertGreater(score["absolute_mechanistic_score"], 0.0)
        self.assertGreater(score["terminal_reward"], 0.0)

    def test_terminal_score_handles_none_task_abstention(self) -> None:
        _, initial_state = initialize_state_from_corum_task(_none_task(), max_budget=4)
        final_state = replace_predicted_groups(
            initial_state,
            predicted_groups=[],
            relationship_status=RelationshipStatus.INSUFFICIENT_SUPPORT,
        )
        final_state.continuation_state = ContinuationState.STOP
        final_state.termination_reason = TerminationReason.MODEL_STOP

        score = score_terminal_trajectory(
            _none_task(),
            initial_state,
            final_state,
            step_count=1,
            max_steps=4,
        )

        self.assertEqual(score["schema_score"], 1.0)
        self.assertGreater(score["absolute_complex_score"], 0.0)
        self.assertGreaterEqual(score["complex_delta"], 0.0)
        self.assertEqual(score["metadata"]["complex"]["expected_relationship"], "insufficient_support")
        self.assertEqual(score["metadata"]["complex"]["final"]["predicted_gene_count"], 0)
        self.assertEqual(
            score["metadata"]["schema"]["termination_reason"],
            TerminationReason.MODEL_STOP.value,
        )

    def test_duplicate_empty_tool_call_increases_efficiency_penalty(self) -> None:
        interpretation, prior_state = initialize_state_from_corum_task(_recovery_task(), max_budget=4)
        updated_state = _clone_state(prior_state)
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG_CREBBP", "layers": ["ppi"]},
            call_id="call_1",
        )
        prior_action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG_CREBBP", "layers": ["ppi"]},
            call_id="older_call",
        )
        observation = ToolObservation(
            status=ToolObservationStatus.EMPTY,
            provenance={"tool_name": "get_neighbors", "layer_name": "ppi"},
            call_id="call_1",
        )
        branch = _make_branch(
            "duplicate_empty",
            interpretation,
            updated_state,
            tool_action=action,
            observation=observation,
        )

        score = score_candidate_branch(
            _recovery_task(),
            prior_state,
            branch,
            step_index=2,
            max_steps=6,
            prior_actions=[prior_action],
            available_gene_ids={"ENSG_CREBBP", "ENSG_EP300", "ENSG_KAT2B", "ENSG_NCOA3"},
            available_layers={"ppi"},
        )

        self.assertGreater(score.efficiency_penalty, 0.9)
        self.assertTrue(score.score_metadata["efficiency"]["duplicate_tool_call"])
        self.assertTrue(score.score_metadata["efficiency"]["invalid_observation"])
        self.assertEqual(score.score_metadata["efficiency"]["invalid_tool_call_increment"], 1)

    def test_schema_invalid_branch_gets_zero_schema_score(self) -> None:
        interpretation, prior_state = initialize_state_from_corum_task(_recovery_task(), max_budget=4)
        updated_state = _clone_state(prior_state)
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG_CREBBP", "layers": ["ppi"]},
            call_id="call_1",
        )
        observation = ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            payload={"neighbors": ["ENSG_EP300"]},
            provenance={"tool_name": "get_neighbors", "layer_name": "ppi"},
            call_id="different_call_id",
        )
        branch = _make_branch(
            "schema_bad",
            interpretation,
            updated_state,
            tool_action=action,
            observation=observation,
        )

        score = score_candidate_branch(
            _recovery_task(),
            prior_state,
            branch,
            step_index=1,
            max_steps=6,
            available_gene_ids={"ENSG_CREBBP", "ENSG_EP300", "ENSG_KAT2B", "ENSG_NCOA3"},
            available_layers={"ppi"},
        )

        self.assertEqual(score.schema_score, 0.0)
        self.assertTrue(any("call_id" in error for error in score.score_metadata["schema_errors"]))

    def test_generator_errors_also_zero_out_schema_score(self) -> None:
        interpretation, prior_state = initialize_state_from_corum_task(_recovery_task(), max_budget=4)
        updated_state = _clone_state(prior_state)
        branch = _make_branch("generator_bad", interpretation, updated_state)
        branch.metadata["generator_errors"] = ["actor_json_parse_error: malformed output"]

        score = score_candidate_branch(
            _recovery_task(),
            prior_state,
            branch,
            step_index=1,
            max_steps=6,
        )

        self.assertEqual(score.schema_score, 0.0)
        self.assertIn("actor_json_parse_error", " ".join(score.score_metadata["schema_errors"]))

    def test_no_label_task_rewards_grounded_enrichment_mechanism(self) -> None:
        task = _no_label_explanation_task()
        interpretation, prior_state = initialize_state_from_corum_task(task, max_budget=4)
        updated_state = append_evidence_record(
            prior_state,
            EvidenceRecord(
                evidence_id="evidence_enrich",
                source_type=EvidenceSourceType.TOOL_OBSERVATION,
                summary="Found enriched terms; top term is histone deacetylase activity.",
                provenance={
                    "tool_name": "enrich_gene_set",
                    "payload": {
                        "results": [
                            {
                                "source": "GO:MF",
                                "native": "GO:0004407",
                                "name": "histone deacetylase activity",
                                "p_value": 0.001,
                                "significant": True,
                                "intersection_size": 2,
                                "query_size": 2,
                                "precision": 1.0,
                            }
                        ]
                    },
                },
                supporting_gene_ids=["ENSG_HDAC4", "ENSG_BCL6"],
            ),
        )
        updated_state = replace_mechanistic_labels(
            updated_state,
            [
                MechanisticLabel(
                    label_source=LabelSource.GO,
                    label_id="GO:0004407",
                    label_name="histone deacetylase activity",
                    evidence_ids=["evidence_enrich"],
                )
            ],
        )
        branch = _make_branch(
            "grounded_mechanism",
            Interpretation(
                mechanistic_claim="Histone deacetylase activity is supported by enrichment evidence.",
                main_evidence="The enrichment tool returned GO:0004407.",
                uncertainty="",
                next_subgoal="",
            ),
            updated_state,
        )

        score = score_candidate_branch(task, prior_state, branch, step_index=1, max_steps=6)

        self.assertGreater(score.mechanistic_label_delta, 0.0)
        self.assertGreater(score.mechanism_evidence_score, 0.0)
        self.assertEqual(
            score.score_metadata["mechanistic"]["score_source"],
            "evidence_grounded_unsupervised",
        )

    def test_no_label_task_penalizes_unsupported_generic_mechanism(self) -> None:
        task = _no_label_explanation_task()
        interpretation, prior_state = initialize_state_from_corum_task(task, max_budget=4)
        updated_state = replace_mechanistic_labels(
            prior_state,
            [
                MechanisticLabel(
                    label_source=LabelSource.FREE_TEXT,
                    label_name="network module",
                    evidence_ids=[],
                )
            ],
        )
        branch = _make_branch(
            "unsupported_mechanism",
            Interpretation(
                mechanistic_claim="The genes form a network module.",
                main_evidence="No annotation evidence was observed.",
                uncertainty="",
                next_subgoal="",
            ),
            updated_state,
        )

        score = score_candidate_branch(task, prior_state, branch, step_index=1, max_steps=6)

        self.assertLess(score.mechanism_evidence_score, 0.1)

    def test_positive_no_label_task_does_not_reward_ungrounded_abstention(self) -> None:
        task = _no_label_explanation_task()
        interpretation, prior_state = initialize_state_from_corum_task(task, max_budget=4)
        updated_state = replace_predicted_groups(
            prior_state,
            predicted_groups=[],
            relationship_status=RelationshipStatus.INSUFFICIENT_SUPPORT,
        )
        branch = _make_branch(
            "ungrounded_abstention",
            Interpretation(
                mechanistic_claim="No specific mechanism is supported.",
                main_evidence="Only seed identifiers were visible.",
                uncertainty="No annotation evidence was observed.",
                next_subgoal="",
            ),
            updated_state,
        )

        score = score_candidate_branch(task, prior_state, branch, step_index=1, max_steps=6)

        self.assertEqual(score.mechanism_evidence_score, 0.0)

    def test_no_label_none_task_rewards_calibrated_abstention_on_empty_enrichment(self) -> None:
        task = _none_task()
        interpretation, prior_state = initialize_state_from_corum_task(task, max_budget=4)
        updated_state = append_evidence_record(
            prior_state,
            EvidenceRecord(
                evidence_id="evidence_empty_enrich",
                source_type=EvidenceSourceType.TOOL_OBSERVATION,
                summary="No significant gene-set enrichment terms were found.",
                provenance={
                    "tool_name": "enrich_gene_set",
                    "payload": {"results": [], "query_gene_ids": ["ENSG_PIK3CB", "ENSG_STAT6"]},
                },
                supporting_gene_ids=["ENSG_PIK3CB", "ENSG_STAT6"],
            ),
        )
        updated_state = replace_predicted_groups(
            updated_state,
            predicted_groups=[],
            relationship_status=RelationshipStatus.INSUFFICIENT_SUPPORT,
        )
        branch = _make_branch(
            "calibrated_abstention",
            Interpretation(
                mechanistic_claim="The mechanism remains unresolved with insufficient support.",
                main_evidence="Enrichment returned no significant shared terms.",
                uncertainty="No shared label is supported.",
                next_subgoal="",
            ),
            updated_state,
        )

        score = score_candidate_branch(task, prior_state, branch, step_index=1, max_steps=6)

        self.assertGreater(score.mechanism_evidence_score, 0.4)


if __name__ == "__main__":
    unittest.main()
