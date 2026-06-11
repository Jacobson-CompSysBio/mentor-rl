import json
import unittest

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
    PreferenceDifficulty,
    PreferencePair,
    RelationshipStatus,
    SchemaValidationError,
    SharedPrefixContext,
    StructuredState,
    TaskType,
    ToolAction,
    ToolObservation,
    ToolObservationStatus,
    UserAnchors,
    VerifierStep,
)


def _build_base_state() -> StructuredState:
    anchors = UserAnchors(
        query_text="What mechanism connects HDAC4 and BCL6?",
        evidence={
            "seed_gene_ids": ["ENSG00000068024", "ENSG00000113916"],
            "seed_gene_symbols": ["HDAC4", "BCL6"],
            "context_text": "Transcriptional repression by BCL6 recruits histone deacetylases.",
        },
        evidence_mode="contextual",
        source_task_id="corum_complex_00001.explanation.complete.contextual",
    )
    return StructuredState(
        user_anchors=anchors,
        relationship_status=RelationshipStatus.UNKNOWN,
        predicted_groups=[
            GeneGroup(
                group_id="group_0",
                gene_ids=["ENSG00000068024", "ENSG00000113916"],
                gene_symbols=["HDAC4", "BCL6"],
                rationale="Initial seed group.",
            )
        ],
        evidence_log=[],
        mechanistic_labels=[],
        remaining_budget=8,
        continuation_state=ContinuationState.CONTINUE,
    )


def _build_candidate_branch(*, branch_id: str, total_score: float, normalized_score: float) -> CandidateBranch:
    next_state = StructuredState(
        user_anchors=_build_base_state().user_anchors,
        relationship_status=RelationshipStatus.VALIDATED_GROUP,
        predicted_groups=[
            GeneGroup(
                group_id="group_0",
                gene_ids=["ENSG00000068024", "ENSG00000113916"],
                gene_symbols=["HDAC4", "BCL6"],
                rationale="Recovered coherent complex.",
            )
        ],
        evidence_log=[
            EvidenceRecord(
                evidence_id="evidence_0",
                source_type=EvidenceSourceType.TOOL_OBSERVATION,
                summary="Neighborhood lookup supports co-complex membership.",
                provenance={"tool_name": "get_neighbors", "layer_name": "ppi"},
                supporting_gene_ids=["ENSG00000068024", "ENSG00000113916"],
                supporting_gene_symbols=["HDAC4", "BCL6"],
                tool_call_id="call_0",
            )
        ],
        mechanistic_labels=[
            MechanisticLabel(
                label_source=LabelSource.GO,
                label_id="GO:0004407",
                label_name="histone deacetylase activity",
                evidence_ids=["evidence_0"],
            )
        ],
        remaining_budget=7,
        continuation_state=ContinuationState.CONTINUE,
        total_tool_call_count=1,
    )
    return CandidateBranch(
        branch_id=branch_id,
        actor_step=ActorStep(
            reasoning_text="I should inspect the local graph neighborhood before revising the mechanism.",
            tool_action=ToolAction(
                tool_name="get_neighbors",
                arguments={"gene": "ENSG00000068024", "layers": ["ppi"]},
                call_id="call_0",
            ),
        ),
        observation=ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            payload={"neighbors": ["ENSG00000113916"]},
            provenance={"tool_name": "get_neighbors", "layer_name": "ppi"},
            call_id="call_0",
        ),
        verifier_step=VerifierStep(
            updated_interpretation=Interpretation(
                mechanistic_claim="HDAC4 and BCL6 participate in a shared repression mechanism.",
                main_evidence="Graph neighbors and user context both support a co-complex interpretation.",
                uncertainty="Additional evidence could test whether the interaction is direct.",
                next_subgoal="Check whether the induced subgraph remains coherent.",
            ),
            updated_state=next_state,
            continuation_decision=ContinuationState.CONTINUE,
            verifier_notes="State is internally consistent.",
        ),
        local_score=LocalScoreBreakdown(
            schema_score=1.0,
            complex_membership_delta=0.5,
            mechanistic_label_delta=0.25,
            efficiency_penalty=0.1,
            total_score=total_score,
            normalized_score=normalized_score,
            score_metadata={"task_type": "explanation"},
        ),
        metadata={"generator": "unit-test"},
    )


class RuntimeSchemaTests(unittest.TestCase):
    def test_structured_state_round_trip_matches_init_state_contract(self) -> None:
        state = _build_base_state()
        serialized = state.to_dict()
        restored = StructuredState.from_dict(serialized)

        self.assertEqual(restored.to_dict(), serialized)
        self.assertEqual(restored.relationship_status, RelationshipStatus.UNKNOWN)
        self.assertEqual(
            restored.predicted_groups[0].gene_ids,
            ["ENSG00000068024", "ENSG00000113916"],
        )
        self.assertEqual(restored.remaining_budget, 8)
        self.assertEqual(restored.continuation_state, ContinuationState.CONTINUE)

    def test_mechanistic_label_accepts_reactome_source(self) -> None:
        label = MechanisticLabel(
            label_source="reactome",
            label_name="Initial triggering of complement",
            label_id="REAC:R-HSA-166663",
            evidence_ids=["evidence_0"],
        )

        self.assertEqual(label.label_source, LabelSource.REACTOME)
        self.assertEqual(
            MechanisticLabel.from_dict(label.to_dict()).label_source,
            LabelSource.REACTOME,
        )

    def test_verifier_step_requires_state_and_decision_alignment(self) -> None:
        state = _build_base_state()
        state.continuation_state = ContinuationState.STOP

        with self.assertRaisesRegex(
            SchemaValidationError, "must match continuation_decision"
        ):
            VerifierStep(
                updated_interpretation=Interpretation("", "", "", ""),
                updated_state=state,
                continuation_decision=ContinuationState.CONTINUE,
            )

    def test_preference_pair_round_trip_preserves_nested_branch_content(self) -> None:
        pair = PreferencePair(
            pair_id="pair_0",
            context=SharedPrefixContext(
                query_text="What mechanism connects HDAC4 and BCL6?",
                user_evidence=_build_base_state().user_anchors.evidence,
                interpretation=Interpretation("", "", "", "Inspect the graph."),
                state=_build_base_state(),
                source_task_id="corum_complex_00001.explanation.complete.contextual",
            ),
            chosen=_build_candidate_branch(
                branch_id="branch_good", total_score=1.3, normalized_score=1.0
            ),
            rejected=_build_candidate_branch(
                branch_id="branch_bad", total_score=0.6, normalized_score=0.35
            ),
            task_type=TaskType.EXPLANATION,
            difficulty_bin=PreferenceDifficulty.MEDIUM,
            decision_step=0,
            raw_score_chosen=1.3,
            raw_score_rejected=0.6,
            normalized_score_chosen=1.0,
            normalized_score_rejected=0.35,
            score_margin=0.65,
            source_task_id="corum_complex_00001.explanation.complete.contextual",
            trajectory_id="trajectory_0",
            trajectory_seed=42,
            evidence_mode="contextual",
            provenance={"split": "train"},
        )

        restored = PreferencePair.from_dict(json.loads(pair.to_json()))

        self.assertEqual(restored.to_dict(), pair.to_dict())
        self.assertGreater(
            restored.chosen.local_score.total_score,
            restored.rejected.local_score.total_score,
        )
        self.assertGreater(
            restored.normalized_score_chosen,
            restored.normalized_score_rejected,
        )

    def test_gene_group_rejects_duplicate_gene_ids(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "must not contain duplicates"):
            GeneGroup(
                group_id="bad_group",
                gene_ids=["ENSG00000068024", "ENSG00000068024"],
                gene_symbols=["HDAC4", "HDAC4_dup"],
            )

    def test_tool_observation_requires_payload_for_success(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "must include a payload"):
            ToolObservation(
                status=ToolObservationStatus.SUCCESS,
                payload=None,
                provenance={"tool_name": "get_neighbors"},
                call_id="call_missing_payload",
            )

    def test_structured_state_allows_empty_groups_for_insufficient_support(self) -> None:
        state = StructuredState(
            user_anchors=_build_base_state().user_anchors,
            relationship_status=RelationshipStatus.INSUFFICIENT_SUPPORT,
            predicted_groups=[],
            evidence_log=[],
            mechanistic_labels=[],
            remaining_budget=0,
            continuation_state=ContinuationState.STOP,
        )

        self.assertEqual(state.to_dict()["predicted_groups"], [])


if __name__ == "__main__":
    unittest.main()
