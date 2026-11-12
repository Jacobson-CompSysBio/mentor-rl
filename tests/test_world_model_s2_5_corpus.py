"""Check membership traces against explicit inventories and visible evidence."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_5_corpus import (
    DIAGNOSTICS, _fact, build_diagnostic, build_row, load_build_contract, select_facts, validate_row,
)
from runtime.world_model.s2_gene_layers import ARGUMENT_KEYS, GeneLayersRuntime
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
PPI, COEX, BRAIN, CD4 = "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells"
A, B, C, D, E, F, G, ISOLATE, MISSING = tuple(f"ENSG{i:011d}" for i in range(1, 10))
UNREGISTERED = "ENSG00000000040"
SEED = 20261028
ALL = {"kind": "all", "layer_ids": [], "layer_family": None}


def _scope(*layers: str) -> dict:
    """Declare an explicit scope in the user's original mention order."""
    return {"kind": "layers", "layer_ids": list(layers), "layer_family": None}


def _calls(row: dict) -> list[dict]:
    """Read current-turn calls separately from supplied observations."""
    return [message["tool_call"] for message in row["messages"] if "tool_call" in message]


def _context(row: dict) -> dict:
    """Read the model-visible harness data without private sampling labels."""
    return json.loads(row["messages"][1]["content"])


class WorldModelS25CorpusTests(unittest.TestCase):
    """Bind frozen questions to independent node inventories and registry symbols."""

    @classmethod
    def setUpClass(cls) -> None:
        """Provide distinct overlap masks, target-only membership, and an explicit isolate."""
        specs = {
            PPI: (False, {(A, B): .9, (C, D): .5}, frozenset((A, B, C, D, ISOLATE))),
            COEX: (False, {(A, C): .7, (E, F): .3}, frozenset((A, C, E, F))),
            BRAIN: (True, {(B, A): .2, (A, C): .8, (D, E): .6}, frozenset((A, B, C, D, E))),
            CD4: (True, {(F, A): 1, (G, UNREGISTERED): .4}, frozenset((A, F, G, UNREGISTERED))),
        }
        cls.view = GraphView("unit_membership_corpus_view", "d" * 64, {
            name: LayerGraph(name, directed, edges, nodes) for name, (directed, edges, nodes) in specs.items()
        })
        cls.runtime = GeneLayersRuntime(cls.view)
        cls.index = NeighborIndex(cls.view)
        cls.contract = load_build_contract()
        cls.bank = json.loads((ROOT / "runtime/world_model/s2_v1_prompt_banks.json").read_text())
        cls.registry_ids = tuple(f"ENSG{i:011d}" for i in range(1, 40))
        cls.registry = RegistryContext("110", "ensembl_homo_sapiens_110_unit", {
            gene: f"UNIT_{number}" for number, gene in enumerate(cls.registry_ids, 1)
        })

    def _row(self, gene: str = A, scope: dict | None = None, *, mode: str = "complete",
             top_k: int | None = None, framing: str = "direct", variant: str = "lookup") -> dict:
        """Build one independently selected classification and evidence contrast."""
        fact = _fact(self.runtime, gene, ALL if scope is None else scope, SEED)
        return build_row(self.view, fact, mode, top_k, framing, variant, 0, self.bank, self.contract,
                         index=self.index, registry=self.registry, seed=SEED)

    def _validate(self, row: dict) -> None:
        """Replay source facts with the fixture registry and native index."""
        validate_row(row, self.view, bank=self.bank, contract=self.contract, registry=self.registry, index=self.index)

    def test_global_negative_is_resolved_and_completes_both_task_frames(self) -> None:
        for framing, decision in (("direct", "stop"), ("step", "continue")):
            row = self._row(MISSING, framing=framing)
            self._validate(row)
            answer = row["messages"][-1]["content"]
            result = answer["result"]
            self.assertEqual((answer["status"], answer["continuation_decision"]), ("resolved", decision))
            self.assertIsNone(answer["blocked_reason"])
            self.assertIs(result["present"], False)
            self.assertEqual(result["layer_ids"], [])
            self.assertEqual(result["layer_count"], 0)
            self.assertEqual(result["coverage"]["checked_layer_ids"], [PPI, COEX, BRAIN, CD4])
            self.assertEqual(result["collection"], {"mode": "complete", "returned_count": 0,
                "total_count": 0, "complete": True, "next_cursor": None})

    def test_selected_absence_retains_global_presence(self) -> None:
        row = self._row(ISOLATE, _scope(BRAIN, CD4))
        self._validate(row)
        result = row["messages"][-1]["content"]["result"]
        self.assertIs(result["present"], True)
        self.assertEqual(result["layer_ids"], [])
        self.assertEqual(result["layer_count"], 0)
        self.assertEqual(row["metadata"]["category"], "selected_absence")

    def test_empty_registered_family_keeps_independent_global_classification(self) -> None:
        scope = {"kind": "all", "layer_ids": [], "layer_family": "bulk_pen"}
        for gene, present in ((A, True), (MISSING, False)):
            row = self._row(gene, scope, mode="top_k", top_k=2, variant="unknown_total")
            self._validate(row)
            result = row["messages"][-1]["content"]["result"]
            self.assertIs(result["present"], present)
            self.assertEqual(result["layer_ids"], [])
            self.assertEqual(result["layer_count"], 0)
            self.assertEqual(result["coverage"], {"operation": "layer_membership", "requested_layer_ids": [],
                                                "checked_layer_ids": [], "unchecked_layer_ids": []})
            self.assertTrue(result["collection"]["complete"])

    def test_inventory_isolate_and_target_only_membership_are_preserved(self) -> None:
        isolated = self._row(ISOLATE, _scope(PPI))
        self._validate(isolated)
        self.assertEqual(isolated["messages"][-1]["content"]["result"]["layer_ids"], [PPI])
        row = self._row(A, _scope(CD4), variant="neighbor_zero")
        self._validate(row)
        saved = _context(row)["observations"][0]
        self.assertEqual(saved["operation"], "get_neighbors")
        self.assertEqual(saved["arguments"]["direction"], "out")
        self.assertEqual(saved["result"]["neighbors"], [])
        self.assertEqual(saved["result"]["not_in_layer_gene_ids"], [])
        self.assertEqual(_calls(row), [])
        self.assertEqual(row["messages"][-1]["content"]["result"]["layer_ids"], [CD4])

    def test_first_k_uses_manifest_order_and_preserves_the_full_scoped_count(self) -> None:
        for k, ids, complete in ((1, [PPI], False), (2, [PPI, COEX], False),
                                 (4, [PPI, COEX, BRAIN, CD4], True), (8, [PPI, COEX, BRAIN, CD4], True)):
            row = self._row(mode="top_k", top_k=k)
            self._validate(row)
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(result["layer_ids"], ids)
            self.assertEqual(result["layer_count"], 4)
            self.assertEqual(result["collection"]["returned_count"], len(ids))
            self.assertEqual(result["collection"]["total_count"], 4)
            self.assertIs(result["collection"]["complete"], complete)

    def test_unknown_counts_require_exhaustive_visible_retrieval(self) -> None:
        for mode, k, total, complete in (("complete", None, 4, True), ("top_k", 2, None, False),
                                         ("top_k", 4, None, False), ("top_k", 9, 4, True)):
            row = self._row(mode=mode, top_k=k, variant="unknown_total")
            self._validate(row)
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(row["metadata"]["layer_count"], 4)
            self.assertEqual(result["layer_count"], total)
            self.assertEqual(result["collection"]["total_count"], total)
            self.assertIs(result["collection"]["complete"], complete)

    def test_complete_and_shorter_prefix_context_need_no_new_calls(self) -> None:
        for variant, mode, k in (("context", "complete", None), ("complete_context", "top_k", 2),
                                 ("shorter_prefix_context", "top_k", 2)):
            row = self._row(mode=mode, top_k=k, variant=variant)
            self._validate(row)
            self.assertEqual(_calls(row), [])
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(result["layer_ids"], [PPI, COEX, BRAIN, CD4] if k is None else [PPI, COEX])
            self.assertEqual(result["layer_count"], 4)

    def test_complete_context_projects_a_family_and_preserves_original_provenance(self) -> None:
        scope = {"kind": "all", "layer_ids": [], "layer_family": "tf_target"}
        row = self._row(scope=scope, variant="projected_context")
        self._validate(row)
        self.assertEqual(_calls(row), [])
        answer = row["messages"][-1]["content"]
        self.assertEqual(answer["result"]["layer_ids"], [BRAIN, CD4])
        self.assertEqual(answer["result"]["layer_count"], 2)
        self.assertEqual(answer["provenance"]["graph"][0]["layer_scope"], ALL)

    def test_partial_and_family_context_acquire_only_unresolved_layers(self) -> None:
        for variant, expected_missing in (("partial_context", [BRAIN, CD4]), ("family_context", [PPI, COEX])):
            row = self._row(variant=variant, framing="step")
            self._validate(row)
            calls = _calls(row)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["arguments"]["layer_scope"], _scope(*expected_missing))
            self.assertEqual(calls[0]["arguments"]["mode"], "complete")
            self.assertIsNone(calls[0]["arguments"]["top_k"])
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["result"]["layer_ids"], [PPI, COEX, BRAIN, CD4])
            self.assertEqual(answer["result"]["layer_count"], 4)
            self.assertEqual(answer["continuation_decision"], "continue")
            self.assertEqual(len(answer["provenance"]["graph"]), 2)

    def test_overlapping_scopes_and_duplicate_observations_do_not_add_counts(self) -> None:
        for variant, evidence_count in (("overlapping_context", 2), ("duplicate_context", 1)):
            row = self._row(variant=variant)
            self._validate(row)
            self.assertEqual(_calls(row), [])
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["result"]["layer_ids"], [PPI, COEX, BRAIN, CD4])
            self.assertEqual(answer["result"]["layer_count"], 4)
            self.assertEqual(len(answer["provenance"]["graph"]), evidence_count)

    def test_insufficient_bounded_context_starts_a_new_selection_at_null(self) -> None:
        for variant, scope in (("bounded_context", ALL), ("broader_bounded_context", _scope(CD4))):
            row = self._row(scope=scope, variant=variant)
            self._validate(row)
            first = _calls(row)[0]["arguments"]
            self.assertIsNone(first["cursor"])
            self.assertEqual(first["mode"], "complete")
            self.assertEqual(first["layer_scope"], scope)
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(result["layer_ids"], [PPI, COEX, BRAIN, CD4] if scope == ALL else [CD4])

    def test_pagination_retains_all_pages_and_exact_cursors(self) -> None:
        for variant in ("paged", "overlap", "empty_page", "prefix_context"):
            row = self._row(variant=variant)
            self._validate(row)
            calls = _calls(row)
            responses = [message["content"] for message in row["messages"] if message["role"] == "tool"]
            if variant == "prefix_context":
                saved = _context(row)["observations"][0]
                self.assertEqual(calls[0]["arguments"]["cursor"], saved["result"]["collection"]["next_cursor"])
            else:
                self.assertIsNone(calls[0]["arguments"]["cursor"])
            for previous, call in zip(responses, calls[1:]):
                self.assertEqual(call["arguments"]["cursor"], previous["result"]["collection"]["next_cursor"])
            final = row["messages"][-1]["content"]["result"]
            self.assertEqual(final["layer_ids"], [PPI, COEX, BRAIN, CD4])
            self.assertEqual(final["layer_count"], 4)
            self.assertTrue(final["collection"]["complete"])

    def test_known_total_equal_to_paged_bound_proves_full_coverage_after_exhaustion(self) -> None:
        row = self._row(mode="top_k", top_k=4, variant="paged")
        self._validate(row)
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["layer_count"], 4)
        self.assertTrue(result["collection"]["complete"])

    def test_first_call_failures_preserve_no_fabricated_evidence_and_exact_retries(self) -> None:
        for variant in ("terminal", "recovered", "exhausted"):
            row = self._row(variant=variant)
            self._validate(row)
            calls = _calls(row)
            self.assertEqual(len(calls), 1 if variant == "terminal" else 2)
            self.assertTrue(all(call == calls[0] for call in calls))
            answer = row["messages"][-1]["content"]
            if variant == "recovered":
                self.assertEqual(answer["status"], "resolved")
                self.assertEqual(len(answer["provenance"]["graph"]), 1)
            else:
                self.assertEqual((answer["status"], answer["continuation_decision"]), ("error", "defer"))
                self.assertEqual(answer["provenance"], {"registry": [], "graph": []})
                self.assertEqual(answer["result"]["partial_results"], [])
                self.assertEqual(answer["result"]["coverage"]["checked_layer_ids"], [])

    def test_page_and_partial_scope_failure_keep_their_different_coverage(self) -> None:
        for variant, checked, unfinished in (("page_terminal", [PPI, COEX, BRAIN, CD4], []),
                                            ("partial_terminal", [PPI, COEX], [BRAIN, CD4])):
            row = self._row(variant=variant)
            self._validate(row)
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["status"], "error")
            self.assertEqual(answer["result"]["coverage"], {"operation": "layer_membership",
                "requested_layer_ids": [PPI, COEX, BRAIN, CD4], "checked_layer_ids": checked,
                "unchecked_layer_ids": unfinished})
            self.assertEqual(len(answer["result"]["partial_results"]), 1)
            self.assertEqual(len(answer["provenance"]["graph"]), 1)
            if variant == "page_terminal":
                self.assertIsNotNone(answer["result"]["arguments"]["cursor"])
            else:
                self.assertEqual(answer["result"]["arguments"]["layer_scope"], _scope(BRAIN, CD4))

    def test_page_and_partial_scope_retry_schedules_recover_or_exhaust(self) -> None:
        for variant in ("page_recovered", "page_exhausted", "partial_recovered", "partial_exhausted"):
            row = self._row(variant=variant)
            self._validate(row)
            calls = _calls(row)
            self.assertEqual(calls[0], calls[1])
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["status"], "resolved" if variant.endswith("recovered") else "error")

    def test_neighbor_and_global_context_supply_explicit_membership_without_degree_shortcuts(self) -> None:
        row = self._row(variant="neighbor_context")
        self._validate(row)
        self.assertEqual(_calls(row)[0]["arguments"]["layer_scope"], _scope(COEX, BRAIN, CD4))
        self.assertEqual(_context(row)["observations"][0]["operation"], "get_neighbors")
        row = self._row(MISSING, variant="global_context", framing="step")
        self._validate(row)
        self.assertEqual(_calls(row), [])
        answer = row["messages"][-1]["content"]
        self.assertIs(answer["result"]["present"], False)
        self.assertEqual((answer["status"], answer["continuation_decision"]), ("resolved", "continue"))
        positive = self._row(variant="global_context")
        self._validate(positive)
        self.assertTrue(_calls(positive))

    def test_symbol_context_preserves_registry_evidence_and_canonical_graph_arguments(self) -> None:
        row = self._row(variant="symbol_context")
        self._validate(row)
        self.assertIn("UNIT_1", row["messages"][2]["content"])
        self.assertEqual(_calls(row)[0]["arguments"]["gene_id"], A)
        self.assertEqual(len(row["messages"][-1]["content"]["provenance"]["registry"]), 1)
        self.assertEqual([call["name"] for call in _calls(row)], ["get_gene_layers"])

    def test_stale_and_wrong_subject_context_cannot_answer_the_current_query(self) -> None:
        for variant in ("stale", "wrong_gene"):
            row = self._row(variant=variant)
            self._validate(row)
            saved = _context(row)["observations"][0]
            self.assertEqual(len(_calls(row)), 1)
            self.assertNotIn(saved["provenance"]["graph"][0], row["messages"][-1]["content"]["provenance"]["graph"])
            if variant == "wrong_gene":
                self.assertEqual(gene_group(saved["arguments"]["gene_id"])[1], row["metadata"]["split"])

    def test_reversed_duplicate_mentions_normalize_calls_and_results(self) -> None:
        for framing in ("direct", "step"):
            row = self._row(scope=_scope(PPI, BRAIN, CD4), variant="scope_order", framing=framing)
            self._validate(row)
            self.assertEqual(row["metadata"]["layer_scope"], _scope(CD4, BRAIN, PPI, CD4))
            self.assertIn(f"{CD4}, {BRAIN}, {PPI}, {CD4}", row["messages"][2]["content"])
            self.assertEqual(_calls(row)[0]["arguments"]["layer_scope"], _scope(PPI, BRAIN, CD4))
            self.assertEqual(tuple(_calls(row)[0]["arguments"]), ARGUMENT_KEYS)
            self.assertEqual(row["messages"][-1]["content"]["result"]["layer_scope"], _scope(PPI, BRAIN, CD4))

    def test_all_six_local_diagnostics_preserve_literal_requests_without_graph_evidence(self) -> None:
        codes = {"unknown_layer": ("not_found", "UNKNOWN_LAYER"), "unknown_family": ("invalid_input", "INVALID_LAYER_FAMILY"),
                 "missing_scope": ("invalid_input", "MISSING_SCOPE"), "empty_scope": ("invalid_input", "INVALID_SELECTION"),
                 "missing_limit": ("invalid_input", "MISSING_LIMIT"), "invalid_limit": ("invalid_input", "INVALID_SELECTION")}
        fact = _fact(self.runtime, A, ALL, SEED)
        self.assertEqual(set(codes), set(DIAGNOSTICS))
        for kind, (status, code) in codes.items():
            row = build_diagnostic(self.view, fact, "step", 0, self.bank, self.contract, kind=kind)
            self._validate(row)
            self.assertEqual(_calls(row), [])
            answer = row["messages"][-1]["content"]
            self.assertEqual((answer["status"], answer["continuation_decision"]), (status, "defer"))
            self.assertEqual(answer["provenance"], {"registry": [], "graph": []})
            self.assertEqual(answer["result"]["issue_code"], code)
            self.assertIsNone(answer["result"]["rejected_arguments"])
            self.assertEqual(answer["result"]["request"]["gene"], A)

    def test_validator_rejects_hidden_counts_invented_membership_and_changed_cursor(self) -> None:
        mutations = []
        row = self._row(mode="top_k", top_k=2, variant="unknown_total")
        result = row["messages"][-1]["content"]["result"]
        result["layer_count"] = result["collection"]["total_count"] = 4
        mutations.append(row)
        row = self._row(variant="context")
        context = _context(row)
        context["observations"][0]["result"]["present"] = False
        row["messages"][1]["content"] = json.dumps(context)
        mutations.append(row)
        row = self._row(variant="page_recovered")
        _calls(row)[1]["arguments"]["cursor"] = None
        mutations.append(row)
        row = self._row()
        row["messages"][-1]["content"]["result"]["layer_ids"] = [PPI, COEX]
        mutations.append(row)
        row = self._row(variant="partial_terminal")
        row["messages"][-1]["content"]["result"]["partial_results"] = []
        mutations.append(row)
        row = self._row()
        row["metadata"]["present"] = False
        mutations.append(row)
        for number, row in enumerate(mutations):
            with self.subTest(mutation=number), self.assertRaises(ValueError):
                self._validate(row)

    def test_private_sampling_labels_stay_out_of_context_and_final_prose_is_free(self) -> None:
        row = self._row(mode="top_k", top_k=2, variant="unknown_total")
        self.assertEqual(set(_context(row)), {"active_graph", "observations"})
        self.assertNotIn("selection_seed", row["messages"][1]["content"])
        self.assertNotIn("category", row["messages"][1]["content"])
        row["messages"][-1]["content"]["summary"] = "The requested membership prefix is supplied; its full count remains unknown."
        self._validate(row)
        failed = self._row(variant="terminal")
        failed["messages"][-1]["content"]["blocked_reason"] = "The required backend must recover."
        self._validate(failed)
        context = _context(row)
        context["private_expected_count"] = 4
        row["messages"][1]["content"] = json.dumps(context)
        with self.assertRaises(ValueError):
            self._validate(row)

    def test_selection_keeps_actual_domains_overlap_patterns_and_unregistered_graph_nodes(self) -> None:
        counts = {split: 1 for split in ("train", "validation", "test")}
        facts, audit = select_facts(self.view, self.registry_ids, counts, SEED)
        self.assertEqual((facts, audit), select_facts(self.view, self.registry_ids, counts, SEED))
        subjects, domains = defaultdict(set), defaultdict(set)
        for fact in facts:
            self.assertEqual((fact.split_unit, fact.split), gene_group(fact.gene_id, SEED))
            subjects[fact.gene_id].add(fact.split)
            self.assertEqual(fact.present, fact.gene_id in self.view.multiplex_nodes)
            if fact.present:
                domains[json.dumps(fact.layer_scope, sort_keys=True)].add(fact.layer_count)
        self.assertTrue(all(len(splits) == 1 for splits in subjects.values()))
        self.assertFalse(audit["edge_fact_disjoint"])
        self.assertTrue(any(f.gene_id == UNREGISTERED for f in facts))
        for domain in audit["domains"]:
            self.assertEqual(domains[json.dumps(domain["layer_scope"], sort_keys=True)], set(domain["whole_graph_count_histogram"]))
            self.assertEqual(domain["whole_graph_members"], len(self.view.multiplex_nodes))
            self.assertEqual(domain["unregistered_graph_members"], 1)
            self.assertEqual(sum(domain["registered_graph_count_histogram"].values()), len(self.view.multiplex_nodes) - 1)
        self.assertTrue(audit["membership_patterns"])


if __name__ == "__main__":
    unittest.main()
