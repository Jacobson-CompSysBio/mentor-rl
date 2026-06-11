import unittest

from runtime.schemas import (
    ActorStep,
    CandidateBranch,
    ContinuationState,
    Interpretation,
    LocalScoreBreakdown,
    PreferenceDifficulty,
    PreferencePair,
    SharedPrefixContext,
    TaskType,
    ToolAction,
    ToolObservation,
    ToolObservationStatus,
    VerifierStep,
)
from runtime.state import initialize_state_from_corum_task
from runtime.validators import (
    is_duplicate_tool_action,
    normalize_tool_arguments,
    tool_action_fingerprint,
    validate_candidate_branch,
    validate_preference_pair,
    validate_tool_action,
    validate_tool_action_schema,
    validate_tool_action_semantics,
)


def _build_task_row() -> dict:
    return {
        "task_id": "corum_complex_00001.explanation.complete.graph",
        "query_text": "What mechanism connects HDAC4 and BCL6?",
        "evidence_mode": "graph",
        "visible_inputs": {
            "seed_gene_ids": ["ENSG00000068024", "ENSG00000113916"],
            "seed_gene_symbols": ["HDAC4", "BCL6"],
            "context_text": None,
            "graph_query_spec": {"operator": "induce_subgraph"},
            "structured_annotations": None,
        },
    }


def _build_branch(branch_id: str) -> CandidateBranch:
    interpretation, state = initialize_state_from_corum_task(_build_task_row(), max_budget=4)
    next_state = state
    next_state.continuation_state = ContinuationState.CONTINUE

    return CandidateBranch(
        branch_id=branch_id,
        actor_step=ActorStep(
            reasoning_text="Look at the direct neighborhood first.",
            tool_action=ToolAction(
                tool_name="get_neighbors",
                arguments={"gene": "ENSG00000068024", "layers": ["ppi"]},
                call_id=f"call_{branch_id}",
            ),
        ),
        observation=ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            payload={"neighbors": ["ENSG00000113916"]},
            provenance={"tool_name": "get_neighbors", "layer_name": "ppi"},
            call_id=f"call_{branch_id}",
        ),
        verifier_step=VerifierStep(
            updated_interpretation=Interpretation(
                mechanistic_claim=interpretation.mechanistic_claim,
                main_evidence="Neighborhood evidence supports a shared module.",
                uncertainty="",
                next_subgoal="Check the induced subgraph.",
            ),
            updated_state=next_state,
            continuation_decision=ContinuationState.CONTINUE,
        ),
        local_score=LocalScoreBreakdown(
            schema_score=1.0,
            complex_membership_delta=0.4,
            mechanistic_label_delta=0.1,
            efficiency_penalty=0.05,
            total_score=1.0 if branch_id == "good" else 0.3,
            normalized_score=1.0 if branch_id == "good" else 0.2,
        ),
    )


class RuntimeValidatorTests(unittest.TestCase):
    def test_validate_tool_action_schema_accepts_valid_graph_call(self) -> None:
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG00000068024", "layers": ["ppi"]},
            call_id="call_1",
        )

        result = validate_tool_action_schema(action)

        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])

    def test_all_layer_aliases_are_validated_as_omitted_layers(self) -> None:
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG00000068024", "layers": ["all"]},
            call_id="call_1",
        )

        result = validate_tool_action_semantics(
            action,
            available_gene_ids={"ENSG00000068024"},
            available_layers={"ppi"},
        )

        self.assertTrue(result.valid)
        self.assertEqual(
            normalize_tool_arguments(action.tool_name, action.arguments),
            {"gene": "ENSG00000068024"},
        )

    def test_empty_and_null_layers_are_validated_as_omitted_layers(self) -> None:
        for layers in ([], None):
            with self.subTest(layers=layers):
                action = ToolAction(
                    tool_name="induce_subgraph",
                    arguments={
                        "genes": ["ENSG00000068024", "ENSG00000113916"],
                        "layers": layers,
                    },
                    call_id="call_1",
                )

                result = validate_tool_action_schema(action)

                self.assertTrue(result.valid)

    def test_empty_and_null_shortest_path_layers_are_validated_as_omitted_layer(self) -> None:
        for layer in ([], None, ["all"], "all"):
            with self.subTest(layer=layer):
                action = ToolAction(
                    tool_name="shortest_path",
                    arguments={
                        "source": "ENSG00000068024",
                        "target": "ENSG00000113916",
                        "layer": layer,
                    },
                    call_id="call_1",
                )

                result = validate_tool_action_schema(action)

                self.assertTrue(result.valid)
                self.assertEqual(
                    normalize_tool_arguments(action.tool_name, action.arguments),
                    {"source": "ENSG00000068024", "target": "ENSG00000113916"},
                )

    def test_validate_tool_action_schema_rejects_bad_arguments(self) -> None:
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "", "layer": "ppi"},
            call_id="call_1",
        )

        result = validate_tool_action_schema(action)

        self.assertFalse(result.valid)
        self.assertGreaterEqual(len(result.errors), 2)

    def test_validate_tool_action_schema_accepts_rwr_plus_plus_names(self) -> None:
        actions = [
            ToolAction(
                tool_name="rwr",
                arguments={"seed_genes": ["ENSG00000068024"], "layers": ["ppi"], "top_k": 5},
                call_id="call_rwr",
            ),
            ToolAction(
                tool_name="rwr",
                arguments={"seed_genes": ["ENSG00000068024"], "layer": "ppi", "top_k": 5},
                call_id="call_rwr_layer_alias",
            ),
            ToolAction(
                tool_name="rwr_loe",
                arguments={
                    "seed_genes": ["ENSG00000068024"],
                    "query_genes": ["ENSG00000113916"],
                    "top_k": 5,
                },
                call_id="call_rwr_loe",
            ),
            ToolAction(
                tool_name="shortest_paths",
                arguments={
                    "source_genes": ["ENSG00000068024"],
                    "target_genes": ["ENSG00000113916"],
                    "merge_method": "max",
                },
                call_id="call_shortest_paths",
            ),
            ToolAction(
                tool_name="get_rank",
                arguments={
                    "source_gene": "ENSG00000068024",
                    "target_gene": "ENSG00000113916",
                },
                call_id="call_get_rank",
            ),
            ToolAction(
                tool_name="get_distance",
                arguments={
                    "gene_a": "ENSG00000068024",
                    "gene_b": "ENSG00000113916",
                    "distance_metric": "spearman",
                },
                call_id="call_get_distance",
            ),
            ToolAction(
                tool_name="get_spearman",
                arguments={
                    "gene_a": "ENSG00000068024",
                    "gene_b": "ENSG00000113916",
                },
                call_id="call_get_spearman",
            ),
            ToolAction(
                tool_name="get_pearson",
                arguments={
                    "gene_a": "ENSG00000068024",
                    "gene_b": "ENSG00000113916",
                },
                call_id="call_get_pearson",
            ),
            ToolAction(
                tool_name="get_dot_similarity",
                arguments={
                    "gene_a": "ENSG00000068024",
                    "gene_b": "ENSG00000113916",
                },
                call_id="call_get_dot_similarity",
            ),
            ToolAction(
                tool_name="get_rank_vector_summary",
                arguments={"seed_genes": ["ENSG00000068024"], "top_k": 5},
                call_id="call_get_rank_vector_summary",
            ),
            ToolAction(
                tool_name="get_encoding_summary",
                arguments={"seed_genes": ["ENSG00000068024"], "top_k": 5},
                call_id="call_get_encoding_summary",
            ),
            ToolAction(
                tool_name="get_gene_layers",
                arguments={"gene": "ENSG00000068024"},
                call_id="call_get_gene_layers",
            ),
            ToolAction(
                tool_name="get_nodes_by_layer",
                arguments={"gene": "ENSG00000068024"},
                call_id="call_get_nodes_by_layer",
            ),
            ToolAction(
                tool_name="get_layer_stats",
                arguments={"top_k": 5},
                call_id="call_get_layer_stats",
            ),
            ToolAction(
                tool_name="get_path_layer_counts",
                arguments={
                    "source_genes": ["ENSG00000068024"],
                    "target_genes": ["ENSG00000113916"],
                    "top_k": 5,
                },
                call_id="call_get_path_layer_counts",
            ),
            ToolAction(
                tool_name="get_component_summary",
                arguments={"genes": ["ENSG00000068024"], "max_components": 5},
                call_id="call_get_component_summary",
            ),
            ToolAction(
                tool_name="get_seed_essentiality",
                arguments={"seed_genes": ["ENSG00000068024", "ENSG00000113916"], "top_k": 2},
                call_id="call_get_seed_essentiality",
            ),
            ToolAction(
                tool_name="get_layer_ablation",
                arguments={"seed_genes": ["ENSG00000068024"], "top_k": 5},
                call_id="call_get_layer_ablation",
            ),
            ToolAction(
                tool_name="get_node_perturbation",
                arguments={
                    "seed_genes": ["ENSG00000068024"],
                    "perturb_genes": ["ENSG00000113916"],
                    "top_k": 5,
                },
                call_id="call_get_node_perturbation",
            ),
        ]

        for action in actions:
            with self.subTest(tool_name=action.tool_name):
                result = validate_tool_action_schema(action)

                self.assertTrue(result.valid, result.errors)

    def test_validate_tool_action_schema_rejects_low_level_rwr_plus_plus_args(self) -> None:
        action = ToolAction(
            tool_name="rwr",
            arguments={"seed_genes": ["ENSG00000068024"], "seed_file": "/tmp/seeds.txt"},
            call_id="call_rwr",
        )

        result = validate_tool_action_schema(action)

        self.assertFalse(result.valid)
        self.assertTrue(any("Unexpected argument" in error for error in result.errors))

    def test_validate_tool_action_semantics_checks_rwr_plus_plus_genes_and_layers(self) -> None:
        action = ToolAction(
            tool_name="rwr",
            arguments={"seed_genes": ["ENSG00000068024"], "layers": ["unknown_layer"]},
            call_id="call_rwr",
        )

        result = validate_tool_action_semantics(
            action,
            available_gene_ids={"ENSG00000068024"},
            available_layers={"ppi"},
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("unknown layers" in error for error in result.errors))

    def test_validate_tool_action_semantics_checks_rwr_single_layer_alias(self) -> None:
        action = ToolAction(
            tool_name="rwr",
            arguments={"seed_genes": ["ENSG00000068024"], "layer": "ppi"},
            call_id="call_rwr",
        )

        result = validate_tool_action_semantics(
            action,
            available_gene_ids={"ENSG00000068024"},
            available_layers={"ppi"},
        )

        self.assertTrue(result.valid, result.errors)

    def test_validate_tool_action_semantics_checks_gene_and_layer_membership(self) -> None:
        interpretation, state = initialize_state_from_corum_task(_build_task_row(), max_budget=4)
        action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG_MISSING", "layers": ["unknown_layer"]},
            call_id="call_1",
        )

        result = validate_tool_action_semantics(
            action,
            state=state,
            available_gene_ids={"ENSG00000068024", "ENSG00000113916"},
            available_layers={"ppi"},
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("not present in the runtime graph" in error for error in result.errors))
        self.assertTrue(any("unknown layers" in error for error in result.errors))
        self.assertEqual(interpretation.next_subgoal, _build_task_row()["query_text"])

    def test_duplicate_tool_detection_uses_stable_fingerprint(self) -> None:
        action_a = ToolAction(
            tool_name="induce_subgraph",
            arguments={"genes": ["ENSG1", "ENSG2"], "layers": ["ppi"]},
            call_id="call_1",
        )
        action_b = ToolAction(
            tool_name="induce_subgraph",
            arguments={"genes": ["ENSG1", "ENSG2"], "layers": ["ppi"]},
            call_id="call_2",
        )

        self.assertEqual(tool_action_fingerprint(action_a), tool_action_fingerprint(action_b))
        self.assertTrue(is_duplicate_tool_action(action_b, [action_a]))

    def test_duplicate_tool_detection_normalizes_all_layer_aliases(self) -> None:
        action_a = ToolAction(
            tool_name="induce_subgraph",
            arguments={"genes": ["ENSG1", "ENSG2"]},
            call_id="call_1",
        )
        action_b = ToolAction(
            tool_name="induce_subgraph",
            arguments={"genes": ["ENSG1", "ENSG2"], "layers": ["all"]},
            call_id="call_2",
        )

        self.assertEqual(tool_action_fingerprint(action_a), tool_action_fingerprint(action_b))
        self.assertTrue(is_duplicate_tool_action(action_b, [action_a]))

    def test_validate_candidate_branch_catches_call_id_mismatch(self) -> None:
        branch = _build_branch("bad")
        branch.observation.call_id = "different_call_id"

        result = validate_candidate_branch(branch)

        self.assertFalse(result.valid)
        self.assertTrue(any("call_id" in error for error in result.errors))

    def test_validate_preference_pair_checks_nested_objects(self) -> None:
        interpretation, state = initialize_state_from_corum_task(_build_task_row(), max_budget=4)
        pair = PreferencePair(
            pair_id="pair_1",
            context=SharedPrefixContext(
                query_text=_build_task_row()["query_text"],
                user_evidence=_build_task_row()["visible_inputs"],
                interpretation=interpretation,
                state=state,
                source_task_id=_build_task_row()["task_id"],
            ),
            chosen=_build_branch("good"),
            rejected=_build_branch("bad"),
            task_type=TaskType.EXPLANATION,
            difficulty_bin=PreferenceDifficulty.EASY,
            decision_step=0,
            raw_score_chosen=1.0,
            raw_score_rejected=0.3,
            normalized_score_chosen=1.0,
            normalized_score_rejected=0.2,
            score_margin=0.8,
            source_task_id=_build_task_row()["task_id"],
            trajectory_id="trajectory_1",
            trajectory_seed=42,
        )

        result = validate_preference_pair(pair)

        self.assertTrue(result.valid)

    def test_validate_tool_action_combines_schema_and_semantics(self) -> None:
        _, state = initialize_state_from_corum_task(_build_task_row(), max_budget=0)
        action = ToolAction(
            tool_name="rwr_monoplex",
            arguments={"seeds": ["ENSG00000068024"], "layer": "ppi", "top_k": 5},
            call_id="call_1",
        )

        result = validate_tool_action(
            action,
            state=state,
            available_gene_ids={"ENSG00000068024"},
            available_layers={"ppi"},
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("remaining budget is 0" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
