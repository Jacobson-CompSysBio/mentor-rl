import unittest

from runtime.schemas import KNOWN_TOOL_NAMES, ToolObservation, ToolObservationStatus
from runtime.tool_curriculum_contract import (
    CURRICULUM_TOOL_NAMES,
    REDACTED_LOCAL_VALUE,
    TOOL_INTENT_POLICIES,
    ToolCurriculumContractError,
    assert_no_provenance_leakage,
    build_tool_action,
    build_tool_exchange,
    build_tool_observation,
    find_provenance_leaks,
    is_curriculum_tool,
    sanitize_tool_payload,
    select_tool_for_intent,
    tool_policy_metadata,
    validate_tool_observation,
)
from runtime.validators import validate_tool_action_schema


class ToolCurriculumContractTests(unittest.TestCase):
    def test_curriculum_vocabulary_is_derived_from_live_runtime_tools(self) -> None:
        self.assertTrue(set(CURRICULUM_TOOL_NAMES).issubset(KNOWN_TOOL_NAMES))
        self.assertNotIn("rwr_hpc_app", CURRICULUM_TOOL_NAMES)
        for fabricated_name in (
            "choose_next_tool",
            "get_graph_schema",
            "query_module_oracle",
            "resolve_gene_alias",
        ):
            self.assertFalse(is_curriculum_tool(fabricated_name))
            with self.assertRaises(ToolCurriculumContractError):
                build_tool_action(fabricated_name, {})

    def test_every_policy_uses_a_live_curriculum_tool(self) -> None:
        for intent, policy in TOOL_INTENT_POLICIES.items():
            with self.subTest(intent=intent):
                self.assertEqual(select_tool_for_intent(intent), policy.preferred_tool)
                self.assertIn(policy.preferred_tool, CURRICULUM_TOOL_NAMES)
                self.assertGreaterEqual(policy.cost_rank, 1)
                metadata = tool_policy_metadata(intent)
                self.assertEqual(metadata["policy"], "cheapest_specific_tool")
                self.assertTrue(metadata["selected_is_preferred"])

    def test_tool_action_normalizes_and_validates_runtime_arguments(self) -> None:
        action = build_tool_action(
            "get_neighbors",
            {"gene": "ENSG000001", "layers": ["all"]},
            available_gene_ids={"ENSG000001"},
            available_layers={"brain_ppi"},
        )

        self.assertEqual(action.arguments, {"gene": "ENSG000001"})
        self.assertTrue(action.call_id.startswith("curriculum_"))
        self.assertTrue(validate_tool_action_schema(action).valid)

    def test_tool_action_rejects_old_generator_argument_shape(self) -> None:
        with self.assertRaisesRegex(ToolCurriculumContractError, "Unexpected argument"):
            build_tool_action(
                "get_neighbors",
                {
                    "gene_id": "ENSG000001",
                    "layer": "brain_ppi",
                    "graph_version": "v1",
                },
            )

    def test_tool_action_rejects_semantically_unknown_gene_or_layer(self) -> None:
        with self.assertRaisesRegex(ToolCurriculumContractError, "not present"):
            build_tool_action(
                "get_neighbors",
                {"gene": "ENSG_UNKNOWN", "layers": ["brain_ppi"]},
                available_gene_ids={"ENSG000001"},
                available_layers={"brain_ppi"},
            )
        with self.assertRaisesRegex(ToolCurriculumContractError, "unknown layers"):
            build_tool_action(
                "get_neighbors",
                {"gene": "ENSG000001", "layers": ["bad_layer"]},
                available_gene_ids={"ENSG000001"},
                available_layers={"brain_ppi"},
            )

    def test_cheapest_specific_policy_rejects_broader_tool(self) -> None:
        with self.assertRaisesRegex(ToolCurriculumContractError, "requires cheapest specific"):
            tool_policy_metadata("pair_rank", selected_tool="rwr")

        diagnostic = tool_policy_metadata(
            "pair_rank",
            selected_tool="rwr",
            require_preferred=False,
        )
        self.assertFalse(diagnostic["selected_is_preferred"])
        self.assertEqual(diagnostic["preferred_tool"], "get_rank")

    def test_payload_sanitizer_removes_machine_details_but_preserves_graph_paths(self) -> None:
        raw = {
            "path_gene_ids": ["ENSG000001", "ENSG000002"],
            "source_path": "/lustre/orion/private/edges.tsv",
            "raw_stdout": "secret output",
            "note": "loaded data/runtime/full_brain_multiplex_store/manifest.json",
            "url": "https://example.org/annotation",
        }

        clean = sanitize_tool_payload(raw)

        self.assertEqual(clean["path_gene_ids"], raw["path_gene_ids"])
        self.assertNotIn("source_path", clean)
        self.assertNotIn("raw_stdout", clean)
        self.assertIn(REDACTED_LOCAL_VALUE, clean["note"])
        self.assertEqual(clean["url"], raw["url"])
        self.assertEqual(find_provenance_leaks(clean), [])

    def test_exchange_is_schema_valid_and_sanitizes_payload_and_provenance(self) -> None:
        exchange = build_tool_exchange(
            "direct_neighbors",
            {"gene": "ENSG000001", "layers": ["brain_ppi"]},
            payload={
                "tool_name": "get_neighbors",
                "query_gene_id": "ENSG000001",
                "unique_neighbors": ["ENSG000002"],
                "cache_path": "/autofs/home/private/cache.json",
                "debug": "read from /lustre/orion/private/edges.tsv",
            },
            provenance={
                "tool_name": "get_neighbors",
                "backend": "compiled_store",
                "network_flist_sha256": "abc123",
                "cwd": "/lustre/orion/private/repo",
                "hostname": "compute-001",
                "raw_stdout": "private command output",
            },
            available_gene_ids={"ENSG000001", "ENSG000002"},
            available_layers={"brain_ppi"},
        )

        action = build_tool_action(
            exchange["tool_action"]["tool_name"],
            exchange["tool_action"]["arguments"],
            call_id=exchange["tool_action"]["call_id"],
        )
        observation = validate_tool_observation(
            exchange["tool_observation"],
            action=action,
        )

        self.assertEqual(observation.status, ToolObservationStatus.SUCCESS)
        self.assertEqual(observation.provenance["backend"], "compiled_store")
        self.assertEqual(observation.provenance["network_flist_sha256"], "abc123")
        self.assertNotIn("cwd", observation.provenance)
        self.assertNotIn("hostname", observation.provenance)
        self.assertNotIn("cache_path", observation.payload)
        self.assertIn(REDACTED_LOCAL_VALUE, observation.payload["debug"])
        self.assertTrue(exchange["tool_policy"]["selected_is_preferred"])
        assert_no_provenance_leakage(exchange)

    def test_error_observation_sanitizes_local_path(self) -> None:
        action = build_tool_action("query_mygene", {"query": "TP53"}, call_id="call_1")
        observation = build_tool_observation(
            action,
            status=ToolObservationStatus.ERROR,
            error="cache failed at /home/user/private/cache.json",
            provenance={"tool_name": "query_mygene", "cache_path": "/home/user/private"},
        )

        self.assertIn(REDACTED_LOCAL_VALUE, observation.error)
        self.assertNotIn("cache_path", observation.provenance)
        validate_tool_observation(observation, action=action)

    def test_validate_supplied_observation_rejects_leak_and_call_mismatch(self) -> None:
        action = build_tool_action("query_mygene", {"query": "TP53"}, call_id="call_1")
        leaking = ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            provenance={"tool_name": "query_mygene", "cwd": "/tmp/private"},
            call_id="call_1",
            payload={"results": []},
        )
        with self.assertRaisesRegex(ToolCurriculumContractError, "local provenance"):
            validate_tool_observation(leaking, action=action)

        clean_wrong_call = ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            provenance={"tool_name": "query_mygene"},
            call_id="call_2",
            payload={"results": []},
        )
        with self.assertRaisesRegex(ToolCurriculumContractError, "does not match"):
            validate_tool_observation(clean_wrong_call, action=action)


if __name__ == "__main__":
    unittest.main()
