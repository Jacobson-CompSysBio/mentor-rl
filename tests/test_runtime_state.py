import unittest

from runtime.schemas import (
    ContinuationState,
    EvidenceRecord,
    EvidenceSourceType,
    MechanisticLabel,
    RelationshipStatus,
    TerminationReason,
)
from runtime.state import (
    append_evidence_record,
    decrement_budget,
    finalize_budget_exhausted,
    finalize_model_stop,
    initialize_state_from_corum_task,
    record_tool_call,
    replace_mechanistic_labels,
    replace_predicted_groups,
)


def _build_task_row() -> dict:
    return {
        "task_id": "corum_complex_00001.explanation.complete.contextual",
        "query_text": "What mechanism connects HDAC4 and BCL6?",
        "evidence_mode": "contextual",
        "visible_inputs": {
            "seed_gene_ids": ["ENSG00000068024", "ENSG00000113916"],
            "seed_gene_symbols": ["HDAC4", "BCL6"],
            "context_text": "BCL6 recruits histone deacetylases during transcriptional repression.",
            "graph_query_spec": None,
            "structured_annotations": None,
        },
    }


class RuntimeStateTests(unittest.TestCase):
    def test_initialize_state_from_corum_task_matches_proposal_defaults(self) -> None:
        interpretation, state = initialize_state_from_corum_task(_build_task_row(), max_budget=6)

        self.assertEqual(interpretation.mechanistic_claim, "")
        self.assertEqual(interpretation.uncertainty, "")
        self.assertEqual(interpretation.next_subgoal, _build_task_row()["query_text"])
        self.assertEqual(state.relationship_status, RelationshipStatus.UNKNOWN)
        self.assertEqual(state.remaining_budget, 6)
        self.assertEqual(state.continuation_state, ContinuationState.CONTINUE)
        self.assertEqual(
            state.predicted_groups[0].gene_ids,
            ["ENSG00000068024", "ENSG00000113916"],
        )
        self.assertEqual(state.evidence_log, [])
        self.assertEqual(state.mechanistic_labels, [])

    def test_append_evidence_record_returns_new_state(self) -> None:
        _, state = initialize_state_from_corum_task(_build_task_row(), max_budget=6)
        record = EvidenceRecord(
            evidence_id="evidence_1",
            source_type=EvidenceSourceType.TOOL_OBSERVATION,
            summary="Neighborhood evidence supports a shared complex.",
            provenance={"tool_name": "get_neighbors"},
            supporting_gene_ids=["ENSG00000068024"],
            supporting_gene_symbols=["HDAC4"],
            tool_call_id="call_1",
        )

        updated = append_evidence_record(state, record)

        self.assertEqual(len(state.evidence_log), 0)
        self.assertEqual(len(updated.evidence_log), 1)
        self.assertEqual(updated.evidence_log[0].evidence_id, "evidence_1")

    def test_replace_helpers_do_not_mutate_original_state(self) -> None:
        _, state = initialize_state_from_corum_task(_build_task_row(), max_budget=6)
        label = MechanisticLabel(
            label_source="go",
            label_id="GO:0004407",
            label_name="histone deacetylase activity",
            evidence_ids=[],
        )
        updated_groups = replace_predicted_groups(
            state,
            predicted_groups=[],
            relationship_status=RelationshipStatus.INSUFFICIENT_SUPPORT,
        )
        updated_labels = replace_mechanistic_labels(state, [label])

        self.assertEqual(len(state.predicted_groups), 1)
        self.assertEqual(state.mechanistic_labels, [])
        self.assertEqual(updated_groups.predicted_groups, [])
        self.assertEqual(
            updated_groups.relationship_status, RelationshipStatus.INSUFFICIENT_SUPPORT
        )
        self.assertEqual(updated_labels.mechanistic_labels[0].label_id, "GO:0004407")

    def test_budget_and_tool_counter_helpers(self) -> None:
        _, state = initialize_state_from_corum_task(_build_task_row(), max_budget=2)
        updated = record_tool_call(state, invalid=True)
        updated = decrement_budget(updated)

        self.assertEqual(updated.total_tool_call_count, 1)
        self.assertEqual(updated.invalid_tool_call_count, 1)
        self.assertEqual(updated.remaining_budget, 1)
        self.assertEqual(state.total_tool_call_count, 0)
        self.assertEqual(state.remaining_budget, 2)

    def test_finalize_helpers_set_stop_state(self) -> None:
        interpretation, state = initialize_state_from_corum_task(_build_task_row(), max_budget=2)

        stopped_interpretation, stopped_state = finalize_model_stop(interpretation, state)
        exhausted_interpretation, exhausted_state = finalize_budget_exhausted(interpretation, state)

        self.assertEqual(stopped_interpretation.next_subgoal, "")
        self.assertEqual(stopped_state.continuation_state, ContinuationState.STOP)
        self.assertEqual(stopped_state.termination_reason, TerminationReason.MODEL_STOP)
        self.assertEqual(exhausted_interpretation.next_subgoal, interpretation.next_subgoal)
        self.assertEqual(exhausted_state.remaining_budget, 0)
        self.assertEqual(
            exhausted_state.termination_reason,
            TerminationReason.BUDGET_EXHAUSTED,
        )


if __name__ == "__main__":
    unittest.main()
