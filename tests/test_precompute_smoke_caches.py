import unittest

from scripts.precompute_smoke_caches import (
    _report_has_errors,
    _visible_seed_gene_ids,
    build_native_store_precompute_actions,
    build_rwr_precompute_actions,
    skipped_rwr_precompute_tools,
)


def _task_row(seed_gene_ids=None):
    return {
        "task_id": "module.recovery",
        "task_type": "recovery",
        "visible_inputs": {
            "seed_gene_ids": seed_gene_ids if seed_gene_ids is not None else ["ENSG1", "ENSG2", "ENSG3"],
            "seed_gene_symbols": ["GENE1", "GENE2", "GENE3"],
        },
        "hidden_target": {
            "target_gene_ids": ["HIDDEN1", "HIDDEN2"],
        },
    }


class PrecomputeSmokeCachesTests(unittest.TestCase):
    def test_visible_seed_gene_ids_ignores_hidden_target(self) -> None:
        self.assertEqual(_visible_seed_gene_ids(_task_row(seed_gene_ids=[])), [])

    def test_build_rwr_precompute_actions_for_visible_seed_set(self) -> None:
        actions = build_rwr_precompute_actions(
            _task_row(),
            profile="core",
            rwr_top_k=500,
            rwr_loe_top_k=20,
            shortest_paths_max_pairs=2,
            shortest_paths_max_paths=10,
        )

        self.assertEqual(
            [action.tool_name for action in actions],
            [
                "rwr",
                "shortest_paths",
                "shortest_paths",
                "get_rank",
                "get_distance",
                "get_spearman",
                "get_pearson",
                "get_dot_similarity",
                "get_path_layer_counts",
                "get_rank_vector_summary",
                "get_encoding_summary",
                "get_layer_stats",
                "get_component_summary",
                "get_gene_layers",
                "get_nodes_by_layer",
                "get_node_perturbation",
            ],
        )
        self.assertEqual(actions[0].arguments, {"seed_genes": ["ENSG1", "ENSG2", "ENSG3"], "top_k": 500})
        self.assertEqual(
            actions[1].arguments,
            {"source_genes": ["ENSG1"], "target_genes": ["ENSG2"], "max_paths": 10},
        )
        self.assertEqual(
            actions[2].arguments,
            {"source_genes": ["ENSG1"], "target_genes": ["ENSG3"], "max_paths": 10},
        )
        self.assertEqual(actions[3].arguments, {"source_gene": "ENSG1", "target_gene": "ENSG2"})
        self.assertEqual(actions[4].arguments, {"gene_a": "ENSG1", "gene_b": "ENSG2"})
        self.assertEqual(actions[5].arguments, {"gene_a": "ENSG1", "gene_b": "ENSG2"})
        self.assertEqual(actions[8].tool_name, "get_path_layer_counts")
        self.assertEqual(actions[9].arguments["include_seed_genes"], False)
        self.assertEqual(actions[-1].arguments, {"seed_genes": ["ENSG1"], "perturb_genes": ["ENSG2"], "top_k": 10})
        self.assertEqual(
            [item["tool_name"] for item in skipped_rwr_precompute_tools("core")],
            ["rwr_loe", "get_seed_essentiality", "get_layer_ablation"],
        )

    def test_build_rwr_precompute_actions_extended_profile_includes_heavy_tools(self) -> None:
        actions = build_rwr_precompute_actions(
            _task_row(),
            profile="extended",
            rwr_top_k=500,
            rwr_loe_top_k=20,
            shortest_paths_max_pairs=1,
            shortest_paths_max_paths=10,
        )

        tool_names = [action.tool_name for action in actions]
        self.assertIn("rwr_loe", tool_names)
        self.assertIn("get_seed_essentiality", tool_names)
        self.assertIn("get_layer_ablation", tool_names)
        self.assertEqual(skipped_rwr_precompute_tools("extended"), [])

    def test_build_rwr_precompute_actions_skips_without_visible_seeds(self) -> None:
        self.assertEqual(
            build_rwr_precompute_actions(
                _task_row(seed_gene_ids=[]),
                profile="core",
                rwr_top_k=500,
                rwr_loe_top_k=20,
                shortest_paths_max_pairs=1,
                shortest_paths_max_paths=20,
            ),
            [],
        )

    def test_build_native_store_precompute_actions_touches_neighbors_and_subgraph(self) -> None:
        actions = build_native_store_precompute_actions(_task_row(), max_genes=2)

        self.assertEqual([action.tool_name for action in actions], ["get_neighbors", "get_neighbors", "induce_subgraph"])
        self.assertEqual(actions[0].arguments, {"gene": "ENSG1"})
        self.assertEqual(actions[1].arguments, {"gene": "ENSG2"})
        self.assertEqual(actions[2].arguments, {"genes": ["ENSG1", "ENSG2"]})

    def test_report_has_errors_checks_enabled_sections(self) -> None:
        self.assertFalse(_report_has_errors({"rwr_precompute": {"error_count": 0}}))
        self.assertTrue(_report_has_errors({"native_store_precompute": {"error_count": 1}}))


if __name__ == "__main__":
    unittest.main()
