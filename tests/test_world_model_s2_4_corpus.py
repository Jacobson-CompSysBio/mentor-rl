"""Check source-bound multiplex-map traces and their visible evidence limits."""

from collections import defaultdict
import json
from pathlib import Path
import unittest

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_4_corpus import (
    _fact, build_diagnostic, build_row, load_build_contract, select_facts, validate_row,
)
from runtime.world_model.s2_neighbor_layers import NeighborLayersRuntime
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
PPI, COEX, BRAIN, CD4 = "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells"
A, B, C, D, E, F, G, H, MISSING = tuple(f"ENSG{i:011d}" for i in range(1, 10))
SEED = 20261028


def _scope(*layers: str) -> dict:
    """Declare an explicit typed layer selection."""
    return {"kind": "layers", "layer_ids": list(layers), "layer_family": None}


def _calls(row: dict) -> list[dict]:
    """Read actual current-turn calls separately from supplied observations."""
    return [message["tool_call"] for message in row["messages"] if "tool_call" in message]


class WorldModelS24CorpusTests(unittest.TestCase):
    """Use a complete small graph with independent source facts and registry symbols."""

    @classmethod
    def setUpClass(cls) -> None:
        """Bind frozen wording and native indexes to an explicit test view."""
        specs = {
            PPI: (False, {(A, B): .9, (A, C): .9, (A, D): .5, (A, E): 0, (B, F): .3}),
            COEX: (False, {(A, B): .21, (B, C): .7, (G, H): .2}),
            BRAIN: (True, {(A, A): 1, (A, B): .8, (A, C): .8, (B, A): .4,
                           (D, A): .2, (C, D): .1, (G, H): .6}),
            CD4: (True, {(B, C): .6, (F, G): .3, (D, D): .1}),
        }
        cls.view = GraphView("unit_neighbor_map_view", "c" * 64, {
            name: LayerGraph(name, directed, edges, frozenset(gene for pair in edges for gene in pair))
            for name, (directed, edges) in specs.items()
        })
        cls.index = NeighborIndex(cls.view)
        cls.runtime = NeighborLayersRuntime(cls.view, index=cls.index)
        cls.contract = load_build_contract()
        cls.bank = json.loads((ROOT / "runtime/world_model/s2_v1_prompt_banks.json").read_text())
        cls.registry_ids = tuple(f"ENSG{i:011d}" for i in range(1, 40))
        cls.registry = RegistryContext("110", "ensembl_homo_sapiens_110_unit", {
            gene: f"UNIT_{i}" for i, gene in enumerate(cls.registry_ids, 1)
        })

    def _row(self, gene: str = A, scope: dict | None = None, direction: str = "incident", *,
             mode: str = "complete", top_k: int | None = None, framing: str = "direct",
             variant: str = "lookup") -> dict:
        """Render one source-selected fact with an explicit request and evidence contrast."""
        fact = _fact(self.runtime, gene, scope or _scope(PPI, COEX), direction, SEED)
        return build_row(self.view, fact, mode, top_k, framing, variant, 0, self.bank, self.contract,
                         index=self.index, registry=self.registry, seed=SEED)

    def _validate(self, row: dict) -> None:
        """Check source replay and visible proof with the same independent fixture registry."""
        validate_row(row, self.view, index=self.index, bank=self.bank, contract=self.contract, registry=self.registry)

    def test_union_and_task_completion_preserve_every_selected_layer_support(self) -> None:
        for framing, decision in (("direct", "stop"), ("step", "continue")):
            row = self._row(framing=framing)
            self._validate(row)
            answer = row["messages"][-1]["content"]
            result = answer["result"]
            self.assertEqual(answer["continuation_decision"], decision)
            self.assertEqual([entry["gene_id"] for entry in result["neighbors"]], [B, C, D, E])
            self.assertEqual(result["unique_neighbor_count"], 4)
            self.assertEqual(result["neighbors"][0], {"gene_id": B, "supporting_layers": [
                {"layer_id": PPI, "directed": False, "edges": [{"gene_ids": [A, B], "weight": .9}]},
                {"layer_id": COEX, "directed": False, "edges": [{"gene_ids": [A, B], "weight": .21}]},
            ], "supporting_layer_count": 2})
            self.assertEqual(result["neighbors"][-1]["supporting_layers"][0]["edges"][0]["weight"], 0)

    def test_mixed_support_reciprocal_weights_and_self_loop_are_native(self) -> None:
        row = self._row(scope=_scope(PPI, COEX, BRAIN, CD4))
        self._validate(row)
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual([entry["gene_id"] for entry in result["neighbors"]], [A, B, C, D, E])
        loop, reciprocal = result["neighbors"][:2]
        self.assertEqual(loop["supporting_layer_count"], 1)
        self.assertEqual(loop["supporting_layers"][0]["edges"], [{"gene_ids": [A, A], "weight": 1}])
        self.assertEqual(reciprocal["supporting_layer_count"], 3)
        self.assertEqual(reciprocal["supporting_layers"][-1]["edges"], [
            {"gene_ids": [B, A], "weight": .4}, {"gene_ids": [A, B], "weight": .8},
        ])
        self.assertEqual(result["layers_without_gene"], [CD4])

    def test_present_zero_absent_layers_and_global_absence_remain_distinct(self) -> None:
        for gene, scope, direction, absent in ((G, _scope(BRAIN), "in", []), (A, _scope(CD4), "incident", [CD4])):
            row = self._row(gene, scope, direction)
            self._validate(row)
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["status"], "resolved")
            self.assertEqual(answer["result"]["neighbors"], [])
            self.assertEqual(answer["result"]["unique_neighbor_count"], 0)
            self.assertEqual(answer["result"]["layers_without_gene"], absent)
        row = self._row(MISSING)
        self._validate(row)
        answer = row["messages"][-1]["content"]
        self.assertEqual((answer["status"], answer["continuation_decision"]), ("not_found", "defer"))
        for field in ("neighbors", "layer_results", "unique_neighbor_count", "layers_without_gene", "collection"):
            self.assertIsNone(answer["result"][field])
        self.assertEqual(answer["result"]["coverage"]["checked_layer_ids"], [])

    def test_context_supplies_same_result_complete_prefix_and_shorter_prefix_without_calls(self) -> None:
        for variant, mode, k in (("context", "complete", None), ("complete_context", "top_k", 2),
                                 ("shorter_prefix_context", "top_k", 2), ("duplicate_context", "complete", None)):
            with self.subTest(variant=variant):
                row = self._row(variant=variant, mode=mode, top_k=k)
                self._validate(row)
                self.assertEqual(_calls(row), [])
                result = row["messages"][-1]["content"]["result"]
                self.assertEqual([entry["gene_id"] for entry in result["neighbors"]], [B, C, D, E] if k is None else [B, C])
                self.assertEqual(result["unique_neighbor_count"], 4)
                self.assertEqual(len(row["messages"][-1]["content"]["provenance"]["graph"]), 1)

    def test_complete_context_projects_scope_and_bounded_context_requires_fresh_acquisition(self) -> None:
        row = self._row(scope=_scope(COEX), variant="projected_context")
        self._validate(row)
        self.assertEqual(_calls(row), [])
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual([entry["gene_id"] for entry in result["neighbors"]], [B])
        self.assertEqual(result["neighbors"][0]["supporting_layer_count"], 1)
        self.assertEqual(result["unique_neighbor_count"], 1)
        source = row["messages"][-1]["content"]["provenance"]["graph"][0]
        self.assertEqual(source["layer_scope"], _scope(PPI, COEX, BRAIN, CD4))
        for variant in ("bounded_context", "broader_bounded_context"):
            row = self._row(scope=_scope(COEX), variant=variant) if variant == "broader_bounded_context" else self._row(variant=variant)
            self._validate(row)
            self.assertEqual(_calls(row)[0]["name"], "get_neighbor_layers")
            self.assertIsNone(_calls(row)[0]["arguments"]["cursor"])
            self.assertEqual(_calls(row)[0]["arguments"]["mode"], "complete")

    def test_native_context_fetches_only_missing_collections_and_regroups_the_union(self) -> None:
        row = self._row(scope=_scope(PPI, BRAIN), variant="native_context")
        self._validate(row)
        self.assertEqual([call["name"] for call in _calls(row)], ["get_neighbor_layers"])
        self.assertEqual(_calls(row)[0]["arguments"]["layer_scope"], _scope(BRAIN))
        self.assertEqual(_calls(row)[0]["arguments"]["direction"], "incident")
        self.assertEqual(row["messages"][-1]["content"]["result"]["unique_neighbor_count"], 5)
        row = self._row(scope=_scope(BRAIN), variant="native_in_context")
        self._validate(row)
        self.assertEqual([call["name"] for call in _calls(row)], ["get_neighbors"])
        self.assertEqual([call["arguments"]["direction"] for call in _calls(row)], ["out"])

    def test_saved_native_layer_and_missing_map_preserve_original_scopes_and_both_supports(self) -> None:
        row = self._row(scope=_scope(PPI, COEX), variant="native_context")
        self._validate(row)
        self.assertEqual(len(_calls(row)), 1)
        self.assertEqual(_calls(row)[0], {"name": "get_neighbor_layers", "arguments": {
            "multiplex_id": self.view.multiplex_id, "gene_id": A, "layer_scope": _scope(COEX),
            "direction": "incident", "mode": "complete", "top_k": None, "cursor": None,
        }})
        saved = json.loads(row["messages"][1]["content"])["observations"][0]
        answer = row["messages"][-1]["content"]
        self.assertEqual(answer["provenance"]["graph"][0], saved["provenance"]["graph"][0])
        self.assertEqual([record["layer_scope"] for record in answer["provenance"]["graph"]], [_scope(PPI), _scope(COEX)])
        self.assertEqual(answer["result"]["unique_neighbor_count"], 4)
        self.assertEqual(answer["result"]["neighbors"][0], {"gene_id": B, "supporting_layers": [
            {"layer_id": PPI, "directed": False, "edges": [{"gene_ids": [A, B], "weight": .9}]},
            {"layer_id": COEX, "directed": False, "edges": [{"gene_ids": [A, B], "weight": .21}]},
        ], "supporting_layer_count": 2})

    def test_missing_whole_layer_failure_preserves_native_evidence_and_layer_map_coverage(self) -> None:
        for variant in ("native_terminal", "native_recovered", "native_exhausted"):
            row = self._row(scope=_scope(PPI, COEX), variant=variant)
            self._validate(row)
            calls = _calls(row)
            self.assertTrue(all(call == calls[0] for call in calls))
            self.assertEqual(calls[0]["name"], "get_neighbor_layers")
            self.assertEqual(calls[0]["arguments"]["layer_scope"], _scope(COEX))
            answer = row["messages"][-1]["content"]
            if variant == "native_recovered":
                self.assertEqual(answer["status"], "resolved")
                self.assertEqual(answer["result"]["neighbors"][0]["supporting_layer_count"], 2)
                continue
            self.assertEqual((answer["status"], answer["continuation_decision"]), ("error", "defer"))
            self.assertEqual(answer["result"]["failed_tool"], "get_neighbor_layers")
            self.assertEqual(answer["result"]["coverage"], {
                "operation": "layer_neighbor_maps", "requested_layer_ids": [PPI, COEX],
                "checked_layer_ids": [PPI], "unchecked_layer_ids": [COEX],
            })
            partial = answer["result"]["partial_results"]
            self.assertEqual(len(partial), 1)
            self.assertEqual(partial[0]["operation"], "get_neighbors")
            self.assertEqual(partial[0]["arguments"]["layer_id"], PPI)
            self.assertEqual(answer["provenance"], partial[0]["provenance"])

    def test_scope_order_preserves_literal_mentions_and_normalizes_the_final_map(self) -> None:
        for framing, decision in (("direct", "stop"), ("step", "continue")):
            row = self._row(scope=_scope(PPI, COEX), variant="scope_order", framing=framing)
            self._validate(row)
            request = row["messages"][2]["content"]
            self.assertEqual(request.count(PPI), 1)
            self.assertEqual(request.count(COEX), 2)
            self.assertLess(request.index(COEX), request.index(PPI))
            self.assertEqual(row["metadata"]["layer_scope"], _scope(COEX, PPI, COEX))
            self.assertEqual(_calls(row)[0]["arguments"]["layer_scope"], _scope(PPI, COEX))
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["continuation_decision"], decision)
            self.assertEqual(answer["result"]["layer_scope"], _scope(PPI, COEX))
            self.assertEqual(answer["result"]["coverage"]["requested_layer_ids"], [PPI, COEX])
            self.assertEqual([support["layer_id"] for support in answer["result"]["neighbors"][0]["supporting_layers"]],
                             [PPI, COEX])

    def test_pagination_retains_every_page_and_exact_cursor(self) -> None:
        row = self._row(variant="paged")
        self._validate(row)
        calls = _calls(row)
        responses = [message["content"] for message in row["messages"] if message["role"] == "tool"]
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["arguments"]["cursor"])
        self.assertEqual(calls[1]["arguments"]["cursor"], responses[0]["result"]["collection"]["next_cursor"])
        self.assertFalse(responses[-1]["result"]["collection"]["complete"])
        final = row["messages"][-1]["content"]
        self.assertEqual([entry["gene_id"] for entry in final["result"]["neighbors"]], [B, C, D, E])
        self.assertTrue(final["result"]["collection"]["complete"])
        self.assertEqual(len(final["provenance"]["graph"]), 2)
        for variant in ("overlap", "empty_page", "prefix_context"):
            row = self._row(variant=variant)
            self._validate(row)
            self.assertEqual(row["messages"][-1]["content"]["result"]["neighbors"], final["result"]["neighbors"])

    def test_unknown_total_requires_exhaustive_evidence(self) -> None:
        for mode, k, total, complete in (("complete", None, 4, True), ("top_k", 2, None, False), ("top_k", 9, 4, True)):
            row = self._row(mode=mode, top_k=k, variant="unknown_total")
            self._validate(row)
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(result["unique_neighbor_count"], total)
            self.assertEqual(result["collection"]["total_count"], total)
            self.assertIs(result["collection"]["complete"], complete)

    def test_page_failures_preserve_partial_map_support_and_identical_retry_cursor(self) -> None:
        for variant in ("terminal", "recovered", "exhausted"):
            row = self._row(variant=variant)
            self._validate(row)
            calls = _calls(row)
            self.assertEqual(len(calls), 1 if variant == "terminal" else 2)
            self.assertTrue(all(call == calls[0] for call in calls))
            self.assertIsNotNone(calls[0]["arguments"]["cursor"])
            answer = row["messages"][-1]["content"]
            if variant == "recovered":
                self.assertEqual(answer["status"], "resolved")
            else:
                self.assertEqual((answer["status"], answer["continuation_decision"]), ("error", "defer"))
                self.assertEqual(answer["result"]["coverage"]["operation"], "layer_neighbor_maps")
                self.assertEqual(answer["result"]["coverage"]["unchecked_layer_ids"], [])
                self.assertEqual(len(answer["result"]["partial_results"]), 1)

    def test_native_failure_retains_fine_direction_coverage(self) -> None:
        row = self._row(scope=_scope(BRAIN), variant="native_terminal")
        self._validate(row)
        answer = row["messages"][-1]["content"]
        self.assertEqual(answer["status"], "error")
        self.assertEqual(answer["result"]["arguments"]["direction"], "out")
        self.assertEqual(answer["result"]["coverage"], {
            "operation": "neighbor_collections", "requested_inputs": [
                {"layer_id": BRAIN, "direction": "in"}, {"layer_id": BRAIN, "direction": "out"}],
            "checked_inputs": [{"layer_id": BRAIN, "direction": "in"}],
            "unchecked_inputs": [{"layer_id": BRAIN, "direction": "out"}],
        })

    def test_symbol_and_stale_context_obey_identity_and_version_requirements(self) -> None:
        row = self._row(variant="symbol_context")
        self._validate(row)
        self.assertIn("UNIT_1", row["messages"][2]["content"])
        self.assertEqual(_calls(row)[0]["arguments"]["gene_id"], A)
        self.assertEqual(len(row["messages"][-1]["content"]["provenance"]["registry"]), 1)
        row = self._row(variant="stale")
        self._validate(row)
        self.assertEqual(len(_calls(row)), 1)
        self.assertTrue(all(record["graph_version"] == self.view.graph_version
                            for record in row["messages"][-1]["content"]["provenance"]["graph"]))

    def test_local_diagnostics_keep_request_and_need_no_tool_call(self) -> None:
        fact = _fact(self.runtime, A, _scope(PPI, BRAIN), "incident", SEED)
        for ranking, status, code in ((False, "invalid_input", "DIRECTION_UNAVAILABLE"),
                                      (True, "ambiguous", "NEIGHBOR_RANKING_AMBIGUOUS")):
            row = build_diagnostic(self.view, fact, "step", 0, self.bank, self.contract, ranking=ranking)
            self._validate(row)
            self.assertEqual(_calls(row), [])
            answer = row["messages"][-1]["content"]
            self.assertEqual((answer["status"], answer["continuation_decision"]), (status, "defer"))
            self.assertEqual(answer["result"]["issue_code"], code)
            self.assertIsNone(answer["result"]["rejected_arguments"])
            self.assertEqual(answer["provenance"], {"registry": [], "graph": []})

    def test_validator_rejects_hidden_total_wrong_source_support_and_changed_cursor(self) -> None:
        mutations = []
        row = self._row(mode="top_k", top_k=2, variant="unknown_total")
        result = row["messages"][-1]["content"]["result"]
        result["unique_neighbor_count"] = result["collection"]["total_count"] = 4
        mutations.append(row)
        row = self._row(variant="context")
        context = json.loads(row["messages"][1]["content"])
        context["observations"][0]["result"]["neighbors"][0]["supporting_layers"][0]["edges"][0]["weight"] = .123
        row["messages"][1]["content"] = json.dumps(context)
        mutations.append(row)
        row = self._row(variant="recovered")
        _calls(row)[-1]["arguments"]["cursor"] = None
        mutations.append(row)
        row = self._row()
        row["messages"][-1]["content"]["result"]["neighbors"][0]["supporting_layer_count"] = 1
        mutations.append(row)
        row = self._row(variant="terminal")
        row["messages"][-1]["content"]["result"]["partial_results"] = []
        mutations.append(row)
        for number, row in enumerate(mutations):
            with self.subTest(mutation=number), self.assertRaises(ValueError):
                self._validate(row)

    def test_private_labels_do_not_enter_context_and_summary_remains_free_text(self) -> None:
        row = self._row(mode="top_k", top_k=2, variant="unknown_total")
        context = json.loads(row["messages"][1]["content"])
        self.assertEqual(set(context), {"active_graph", "observations"})
        self.assertNotIn("category", row["messages"][1]["content"])
        self.assertNotIn("selection_seed", row["messages"][1]["content"])
        self.assertEqual(row["metadata"]["unique_neighbor_count"], 4)
        self.assertIsNone(row["messages"][-1]["content"]["result"]["unique_neighbor_count"])
        row["messages"][-1]["content"]["summary"] = "The requested prefix is complete, and the total remains unavailable."
        self._validate(row)
        context["private_expected_count"] = 4
        row["messages"][1]["content"] = json.dumps(context)
        with self.assertRaises(ValueError):
            self._validate(row)

    def test_selection_is_deterministic_and_query_subjects_keep_the_same_split(self) -> None:
        counts = {split: 1 for split in ("train", "validation", "test")}
        facts, audit = select_facts(self.view, self.index, self.registry_ids, counts, SEED)
        self.assertEqual((facts, audit), select_facts(self.view, self.index, self.registry_ids, counts, SEED))
        subjects = defaultdict(set)
        actual_domains = defaultdict(set)
        for fact in facts:
            self.assertEqual((fact.split_unit, fact.split), gene_group(fact.gene_id, SEED))
            subjects[fact.gene_id].add(fact.split)
            if fact.unique_neighbor_count is not None:
                actual_domains[json.dumps(fact.layer_scope, sort_keys=True), fact.direction].add(fact.unique_neighbor_count)
        self.assertTrue(all(len(splits) == 1 for splits in subjects.values()))
        self.assertFalse(audit["edge_fact_disjoint"])
        for domain in audit["domains"]:
            self.assertEqual(actual_domains[json.dumps(domain["layer_scope"], sort_keys=True), domain["direction"]],
                             set(domain["unique_neighbor_count_histogram"]))


if __name__ == "__main__":
    unittest.main()
