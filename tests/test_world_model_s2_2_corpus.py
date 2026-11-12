"""Independent corpus checks for S2.2 scope, evidence, and sampling behavior.

The six-node graph is an explicit unit-test fixture. Its four-layer positive
tests generic behavior; it is not asserted to exist in the pinned review data.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from runtime.world_model.s2_2_corpus import (
    MultiplexFact,
    VariantUnavailable,
    build_corpus,
    build_direction_diagnostic,
    build_row,
    load_build_contract,
    select_facts,
    validate_row,
)
from runtime.world_model.s2_corpus import _pair_group
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_multiplex import CandidateInventory, MultiplexRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
PPI, COEXPR, BRAIN, CD4 = (
    "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells",
)
ALL = {"kind": "all", "layer_ids": [], "layer_family": None}
TF_FAMILY = {"kind": "all", "layer_ids": [], "layer_family": "tf_target"}
EMPTY_FAMILY = {"kind": "all", "layer_ids": [], "layer_family": "bulk_pen"}
SEED = 20261028


def _gene(number: int) -> str:
    """Return an explicit unit-fixture identifier, never a production source fact."""
    return f"ENSG{number:011d}"


A, B, C, D, E, F, MISSING = tuple(_gene(i) for i in range(1, 8))


def _view() -> GraphView:
    """Cover h=0..4, opposite cross-layer arcs, and per-layer reciprocal arcs."""
    maps = {
        PPI: (False, {(A, B): .11, (A, C): .12, (A, D): .13, (E, F): .15}),
        COEXPR: (False, {(A, B): .21, (A, C): .22, (E, F): .25}),
        BRAIN: (True, {(A, B): .31, (A, D): .34, (B, C): .32, (C, B): .33, (E, F): .35}),
        CD4: (True, {(D, A): .41, (E, F): .45}),
    }
    layers = {}
    for name, (directed, edges) in maps.items():
        nodes = frozenset(gene for pair in edges for gene in pair)
        layers[name] = LayerGraph(name, directed, edges, nodes)
    return GraphView("unit_s22_view", "b" * 64, layers)


def _calls(row: dict) -> list[dict]:
    """Separate current-turn tool calls from supplied historical observations."""
    return [message["tool_call"] for message in row["messages"] if "tool_call" in message]


class WorldModelS22CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Load frozen wording resources with an isolated source graph."""
        cls.view = _view()
        cls.runtime = MultiplexRuntime(cls.view)
        cls.inventory = CandidateInventory(cls.view)
        cls.contract = load_build_contract()
        cls.bank = json.loads((ROOT / "runtime/world_model/s2_v1_prompt_banks.json").read_text())
        cls.registry_ids = tuple(_gene(i) for i in range(1, 11))
        cls.registry = RegistryContext("110", "ensembl_homo_sapiens_110_unit", {
            gene: f"UNIT_{index}" for index, gene in enumerate(cls.registry_ids, 1)
        })

    def _fact(self, pair: tuple[str, str] = (A, B), *, scope: dict | None = None,
              query: str = "any_direction") -> MultiplexFact:
        """Keep the exact S2.1 pair assignment for later cross-family bundling."""
        scope = deepcopy(ALL if scope is None else scope)
        unit, split = _pair_group(pair, SEED)
        layers = self.runtime.resolve_scope(scope, query)
        support = (self.inventory.query_support(pair, layers, query)
                   if set(pair) <= self.view.multiplex_nodes else None)
        return MultiplexFact(split, pair, scope, query, support, unit)

    def _row(self, pair: tuple[str, str] = (A, B), *, scope: dict | None = None,
             query: str = "any_direction", mode: str = "complete", top_k: int | None = None,
             framing: str = "direct", variant: str = "lookup") -> dict:
        """Render one independently checked request, without changing the source facts."""
        fact = self._fact(pair, scope=scope, query=query)
        return build_row(self.view, fact, mode, top_k, framing, variant, 0, self.bank, self.contract,
                         registry=self.registry)

    def test_complete_counts_layers_and_preserves_native_reciprocal_weights(self) -> None:
        """Two arcs in one TF layer count once; cross-layer reverse arcs do not pair up."""
        for framing in ("direct", "step"):
            row = self._row(framing=framing)
            validate_row(row, self.view)
            answer = row["messages"][-1]["content"]
            result = answer["result"]
            self.assertEqual(result["supporting_layer_count"], 3)
            self.assertEqual(result["supporting_layer_ids"], [PPI, COEXPR, BRAIN])
            self.assertEqual(answer["continuation_decision"], "stop" if framing == "direct" else "continue")
            self.assertEqual(result["coverage"]["checked_layer_ids"], list(self.view.layer_order))
        reciprocal = self._row((B, C), scope=TF_FAMILY, query="both_directions")
        validate_row(reciprocal, self.view)
        result = reciprocal["messages"][-1]["content"]["result"]
        self.assertEqual(result["supporting_layer_count"], 1)
        self.assertEqual(result["layer_results"][0]["edges"], [
            {"gene_ids": [B, C], "weight": .32}, {"gene_ids": [C, B], "weight": .33},
        ])
        across = self._row((A, D), scope=TF_FAMILY, query="both_directions")
        validate_row(across, self.view)
        result = across["messages"][-1]["content"]["result"]
        self.assertEqual(result["supporting_layer_count"], 0)
        self.assertTrue(all(not layer["edge_exists"] and len(layer["edges"]) == 1
                            for layer in result["layer_results"]))

    def test_directed_reference_order_changes_the_predicate_without_sorting_edges(self) -> None:
        """The same endpoints can have different forward-supporting TF layers."""
        first = self._row((A, D), scope=TF_FAMILY, query="source_to_target")
        reverse = self._row((D, A), scope=TF_FAMILY, query="source_to_target")
        for row in (first, reverse):
            validate_row(row, self.view)
        self.assertEqual(first["messages"][-1]["content"]["result"]["supporting_layer_ids"], [BRAIN])
        self.assertEqual(reverse["messages"][-1]["content"]["result"]["supporting_layer_ids"], [CD4])
        self.assertEqual(reverse["messages"][-1]["content"]["result"]["gene_ids"], [D, A])
        self.assertEqual(first["metadata"]["split_unit"], reverse["metadata"]["split_unit"])

    def test_inventory_total_cannot_leak_into_early_exists_or_bounded_answers(self) -> None:
        """Knowing h during sampling does not make it visible in an early-stopping trace."""
        for mode, k in (("exists", None), ("top_k", 1)):
            row = self._row(mode=mode, top_k=k)
            validate_row(row, self.view)
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(result["supporting_layer_ids"], [PPI])
            self.assertEqual(result["coverage"]["checked_layer_ids"], [PPI])
            self.assertIsNone(result["supporting_layer_count"])
            if mode == "exists":
                self.assertIsNone(result["collection"])
            else:
                self.assertIsNone(result["collection"]["total_count"])
                self.assertFalse(result["collection"]["complete"])
        # k=h still does not prove exhaustion if the final match precedes an unchecked layer.
        row = self._row((B, C), mode="top_k", top_k=1)
        validate_row(row, self.view)
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["supporting_layer_ids"], [BRAIN])
        self.assertEqual(result["coverage"]["unchecked_layer_ids"], [CD4])
        self.assertIsNone(result["supporting_layer_count"])

    def test_full_saved_context_can_supply_a_richer_exists_or_bounded_answer(self) -> None:
        """Reuse keeps all known layer observations even when the requested display is shorter."""
        for mode, k in (("exists", None), ("top_k", 1)):
            row = self._row(mode=mode, top_k=k, variant="full_context")
            validate_row(row, self.view)
            self.assertEqual(_calls(row), [])
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(len(result["layer_results"]), 4)
            self.assertEqual(result["supporting_layer_count"], 3)
            self.assertEqual(result["coverage"]["unchecked_layer_ids"], [])
            if mode == "top_k":
                self.assertEqual(result["supporting_layer_ids"], [PPI])
                self.assertEqual(result["collection"]["total_count"], 3)
                self.assertFalse(result["collection"]["complete"])
            else:
                self.assertEqual(result["supporting_layer_ids"], [PPI, COEXPR, BRAIN])

    def test_a_later_witness_answers_exists_without_claiming_first_position_or_reverse(self) -> None:
        """A saved directed positive suffices for existence with partial inner coverage."""
        row = self._row((B, C), mode="exists", variant="later_witness")
        validate_row(row, self.view)
        self.assertEqual(_calls(row), [])
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["supporting_layer_ids"], [BRAIN])
        self.assertEqual(result["coverage"]["checked_layer_ids"], [BRAIN])
        self.assertIsNone(result["supporting_layer_count"])
        self.assertEqual(len(result["layer_results"][0]["coverage"]["checked_pairs"]), 1)
        self.assertEqual(len(result["layer_results"][0]["coverage"]["unchecked_pairs"]), 1)
        with self.assertRaises(VariantUnavailable):
            self._row((B, C), mode="top_k", top_k=1, variant="later_witness")

    def test_partial_completion_queries_only_remaining_scope_and_retains_evidence(self) -> None:
        """The final complete result is a calculation over the saved and new subsets."""
        row = self._row(variant="partial", framing="step")
        validate_row(row, self.view)
        calls = _calls(row)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["layer_scope"], {
            "kind": "layers", "layer_ids": [COEXPR, BRAIN, CD4], "layer_family": None,
        })
        answer = row["messages"][-1]["content"]
        self.assertEqual(answer["result"]["supporting_layer_count"], 3)
        self.assertEqual(answer["result"]["coverage"]["checked_layer_ids"], list(self.view.layer_order))
        self.assertEqual(len(answer["provenance"]["graph"]), 2)
        self.assertEqual(answer["continuation_decision"], "continue")

    def test_bounded_context_is_completed_and_top_k_remaining_bound_is_reduced(self) -> None:
        """A saved prefix changes both the remaining layer list and the remaining K."""
        row = self._row(variant="bounded_context")
        validate_row(row, self.view)
        self.assertEqual(_calls(row)[0]["arguments"]["layer_scope"]["layer_ids"], [COEXPR, BRAIN, CD4])
        self.assertEqual(row["messages"][-1]["content"]["result"]["supporting_layer_count"], 3)
        row = self._row(mode="top_k", top_k=3, variant="partial")
        validate_row(row, self.view)
        self.assertEqual(_calls(row)[0]["arguments"]["top_k"], 2)
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["supporting_layer_ids"], [PPI, COEXPR, BRAIN])
        self.assertEqual(result["coverage"]["unchecked_layer_ids"], [CD4])
        self.assertIsNone(result["supporting_layer_count"])

    def test_a_bound_larger_than_scope_and_empty_family_bounds_are_valid(self) -> None:
        """Positive K is valid even when the selected scope contains fewer than K layers."""
        for scope, count in ((ALL, 3), (EMPTY_FAMILY, 0)):
            row = self._row(scope=scope, mode="top_k", top_k=5)
            validate_row(row, self.view)
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(result["supporting_layer_count"], count)
            self.assertEqual(result["collection"]["returned_count"], count)
            self.assertEqual(result["collection"]["total_count"], count)
            self.assertTrue(result["collection"]["complete"])

    def test_stale_context_requires_fresh_acquisition_and_cannot_support_final_provenance(self) -> None:
        """Current source equality does not make an explicitly stale acquisition authoritative."""
        row = self._row(variant="stale")
        validate_row(row, self.view)
        self.assertEqual(len(_calls(row)), 1)
        context = json.loads(row["messages"][1]["content"])
        old_record = context["observations"][0]["provenance"]["graph"][0]
        answer = row["messages"][-1]["content"]
        self.assertEqual(len(answer["provenance"]["graph"]), 1)
        self.assertNotIn(old_record, answer["provenance"]["graph"])
        answer["provenance"]["graph"].append(old_record)
        with self.assertRaises(ValueError):
            validate_row(row, self.view)

    def test_reversed_saved_reference_preserves_native_roles_and_predicate_meaning(self) -> None:
        """Symmetric queries rebase saved arrays; ordered queries still need their own arc."""
        row = self._row((A, D), variant="reversed_context")
        validate_row(row, self.view)
        self.assertEqual(_calls(row), [])
        context = json.loads(row["messages"][1]["content"])
        self.assertEqual(context["observations"][0]["arguments"]["gene_ids"], [D, A])
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["gene_ids"], [A, D])
        rows = {layer["layer_id"]: layer for layer in result["layer_results"]}
        self.assertEqual(rows[BRAIN]["edges"], [{"gene_ids": [A, D], "weight": .34}])
        self.assertEqual(rows[CD4]["edges"], [{"gene_ids": [D, A], "weight": .41}])
        row = self._row((B, C), scope=TF_FAMILY, query="both_directions", variant="reversed_context")
        validate_row(row, self.view)
        self.assertEqual(_calls(row), [])
        self.assertEqual(row["messages"][-1]["content"]["result"]["layer_results"][0]["edges"], [
            {"gene_ids": [B, C], "weight": .32}, {"gene_ids": [C, B], "weight": .33},
        ])
        row = self._row((A, D), scope=TF_FAMILY, query="source_to_target", variant="reversed_context")
        validate_row(row, self.view)
        self.assertEqual(len(_calls(row)), 1)
        self.assertEqual(row["messages"][-1]["content"]["result"]["supporting_layer_ids"], [BRAIN])

    def test_symbol_context_resolves_arguments_and_retains_registry_and_graph_evidence(self) -> None:
        """Unique supplied identities license canonical calls, but prove no graph membership."""
        row = self._row(variant="symbol_context", framing="step")
        validate_row(row, self.view, registry=self.registry)
        self.assertIn("UNIT_1", row["messages"][2]["content"])
        self.assertIn("UNIT_2", row["messages"][2]["content"])
        self.assertNotIn(A, row["messages"][2]["content"])
        calls = _calls(row)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_edge_layers")
        self.assertEqual(calls[0]["arguments"]["gene_ids"], [A, B])
        answer = row["messages"][-1]["content"]
        self.assertEqual(len(answer["provenance"]["registry"]), 2)
        self.assertEqual(len(answer["provenance"]["graph"]), 1)
        answer["provenance"]["registry"].pop()
        with self.assertRaises(ValueError):
            validate_row(row, self.view, registry=self.registry)
        row = self._row(variant="symbol_context")
        context = json.loads(row["messages"][1]["content"])
        context["observations"][0]["result"]["gene_id"] = C
        row["messages"][1]["content"] = json.dumps(context)
        with self.assertRaises(ValueError):
            validate_row(row, self.view, registry=self.registry)

    def test_ambiguous_symbols_are_not_silently_treated_as_unique_context(self) -> None:
        """This variant must skip a nonunique name instead of manufacturing a selection."""
        ambiguous = RegistryContext("110", "ensembl_homo_sapiens_110_unit", {
            A: "DUPLICATE", B: "UNIT_B", C: "DUPLICATE",
        })
        with self.assertRaises(VariantUnavailable):
            build_row(self.view, self._fact(), "complete", None, "direct", "symbol_context", 0,
                      self.bank, self.contract, registry=ambiguous)

    def test_wrong_pair_context_does_not_cross_pair_split_boundaries(self) -> None:
        """Distractor graph facts obey the same train/evaluation partition as requested facts."""
        checked = 0
        for pair in combinations(sorted(self.view.multiplex_nodes), 2):
            try:
                row = self._row(pair, variant="wrong_pair")
            except VariantUnavailable:
                continue
            validate_row(row, self.view)
            context = json.loads(row["messages"][1]["content"])
            saved = context["observations"][0]
            self.assertNotEqual(set(saved["arguments"]["gene_ids"]), set(pair))
            self.assertEqual(_pair_group(tuple(saved["arguments"]["gene_ids"]), SEED)[1],
                             row["metadata"]["split"])
            self.assertEqual(len(_calls(row)), 1)
            self.assertNotIn(saved["provenance"]["graph"][0],
                             row["messages"][-1]["content"]["provenance"]["graph"])
            checked += 1
        self.assertGreater(checked, 0)

    def test_complete_orientation_evidence_survives_a_predicate_change(self) -> None:
        """A different saved Boolean label does not erase fully observed native arcs."""
        for target, saved_query in (("any_direction", "both_directions"),
                                    ("both_directions", "any_direction"),
                                    ("source_to_target", "any_direction")):
            row = self._row((A, D), scope=TF_FAMILY, query=target, variant="full_context")
            context = json.loads(row["messages"][1]["content"])
            saved = context["observations"][0]
            saved["arguments"]["edge_query"] = saved_query
            response = self.runtime.get_edge_layers(
                **saved["arguments"], evidence_id=saved["provenance"]["graph"][0]["evidence_id"])
            saved.update(response)
            row["messages"][1]["content"] = json.dumps(context)
            row["messages"][-1]["thinking"] = (
                "The saved record checks both native orientations in each selected layer. "
                "Compute the requested predicate from those observations without acquiring new evidence."
            )
            with self.subTest(target=target):
                validate_row(row, self.view)
                self.assertEqual(_calls(row), [])
                expected_count = {"any_direction": 2, "both_directions": 0, "source_to_target": 1}[target]
                self.assertEqual(row["messages"][-1]["content"]["result"]["supporting_layer_count"], expected_count)

    def test_partial_cross_predicate_context_keeps_settled_later_layer(self) -> None:
        """A saved forward negative settles both=false even before other layers are settled."""
        for mode, k in (("complete", None), ("top_k", 1)):
            row = self._row((A, D), scope=TF_FAMILY, query="both_directions",
                            mode=mode, top_k=k, variant="wrong_predicate")
            validate_row(row, self.view)
            calls = _calls(row)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["arguments"]["layer_scope"]["layer_ids"], [BRAIN])
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(result["supporting_layer_count"], 0)
            self.assertEqual(result["coverage"]["checked_layer_ids"], [BRAIN, CD4])
            cd4 = result["layer_results"][1]
            self.assertFalse(cd4["edge_exists"])
            self.assertEqual(cd4["edges"], [])
            self.assertEqual(cd4["coverage"]["checked_pairs"], [[A, D]])
            self.assertEqual(cd4["coverage"]["unchecked_pairs"], [[D, A]])
            self.assertEqual(len(row["messages"][-1]["content"]["provenance"]["graph"]), 2)

    def test_broader_saved_scope_answers_subset_without_relabelling_its_acquisition(self) -> None:
        """Project the requested layer results while retaining the original broad source record."""
        scope = {"kind": "layers", "layer_ids": [PPI, BRAIN], "layer_family": None}
        row = self._row(scope=scope, variant="broader_context")
        validate_row(row, self.view)
        self.assertEqual(_calls(row), [])
        context = json.loads(row["messages"][1]["content"])
        saved = context["observations"][0]
        answer = row["messages"][-1]["content"]
        self.assertEqual(answer["result"]["layer_scope"], scope)
        self.assertEqual(answer["result"]["supporting_layer_count"], 2)
        self.assertEqual(answer["result"]["coverage"]["checked_layer_ids"], [PPI, BRAIN])
        self.assertEqual([layer["layer_id"] for layer in answer["result"]["layer_results"]], [PPI, BRAIN])
        self.assertEqual(answer["provenance"], saved["provenance"])
        self.assertNotEqual(saved["provenance"]["graph"][0]["layer_scope"], scope)

    def test_validator_keeps_boolean_and_numeric_json_types_distinct(self) -> None:
        """Python equality between True and 1 must not make an invalid answer pass."""
        row = self._row((B, C))
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["supporting_layer_count"], 1)
        result["supporting_layer_count"] = True
        result["collection"]["total_count"] = True
        with self.assertRaises(ValueError):
            validate_row(row, self.view)
        row = self._row()
        row["messages"][-1]["content"]["result"]["edge_exists_any_layer"] = 1
        with self.assertRaises(ValueError):
            validate_row(row, self.view)

    def test_remaining_scope_failure_is_atomic_and_retries_do_not_recheck_saved_layers(self) -> None:
        """A failed batch contributes no progress; only the successful old subset is checked."""
        for variant in ("terminal", "recovered", "exhausted"):
            row = self._row(variant=variant, framing="step")
            validate_row(row, self.view)
            calls = _calls(row)
            self.assertEqual(len(calls), 1 if variant == "terminal" else 2)
            self.assertTrue(all(call == calls[0] for call in calls))
            self.assertEqual(calls[0]["arguments"]["layer_scope"]["layer_ids"], [COEXPR, BRAIN, CD4])
            answer = row["messages"][-1]["content"]
            if variant == "recovered":
                self.assertEqual((answer["status"], answer["continuation_decision"]), ("resolved", "continue"))
                self.assertEqual(answer["result"]["supporting_layer_count"], 3)
            else:
                self.assertEqual((answer["status"], answer["continuation_decision"]), ("error", "defer"))
                self.assertEqual(len(answer["result"]["partial_results"]), 1)
                self.assertEqual(len(answer["provenance"]["graph"]), 1)
                self.assertEqual(answer["result"]["coverage"]["requested_layer_ids"], list(self.view.layer_order))
                self.assertEqual(answer["result"]["coverage"]["checked_layer_ids"], [PPI])
                self.assertEqual(answer["result"]["coverage"]["unchecked_layer_ids"], [COEXPR, BRAIN, CD4])

    def test_empty_family_and_global_absence_are_distinct_even_without_layers(self) -> None:
        """An empty selected scope does not bypass the two global-membership prerequisites."""
        row = self._row(scope=EMPTY_FAMILY)
        validate_row(row, self.view)
        answer = row["messages"][-1]["content"]
        self.assertEqual(answer["status"], "resolved")
        self.assertIs(answer["result"]["edge_exists_any_layer"], False)
        self.assertEqual(answer["result"]["supporting_layer_count"], 0)
        self.assertEqual(answer["result"]["layer_results"], [])
        for scope in (ALL, EMPTY_FAMILY):
            missing = self._row((A, MISSING), scope=scope)
            validate_row(missing, self.view)
            answer = missing["messages"][-1]["content"]
            self.assertEqual((answer["status"], answer["continuation_decision"]), ("not_found", "defer"))
            result = answer["result"]
            self.assertEqual(result["missing_gene_ids"], [MISSING])
            for field in ("edge_exists_any_layer", "layer_results", "supporting_layer_ids", "supporting_layer_count", "collection"):
                self.assertIsNone(result[field])
            self.assertEqual(result["coverage"]["checked_layer_ids"], [])

    def test_selected_facts_cover_attainable_ranges_and_reuse_s21_split_units(self) -> None:
        """Enumerate feasible scope/count domains before assigning wording or row budgets."""
        counts = {"train": 1, "validation": 1, "test": 1}
        facts, audit = select_facts(self.view, self.registry_ids, counts, SEED)
        again, again_audit = select_facts(self.view, self.registry_ids, counts, SEED)
        self.assertEqual(facts, again)
        self.assertEqual(audit, again_audit)
        sizes, full_counts = set(), set()
        pair_splits = defaultdict(set)
        for fact in facts:
            unit, split = _pair_group(fact.gene_ids, SEED)
            self.assertEqual((fact.split_unit, fact.split), (unit, split))
            pair_splits[tuple(sorted(fact.gene_ids))].add(fact.split)
            layers = self.runtime.resolve_scope(fact.layer_scope, fact.edge_query)
            if fact.support_count is not None:
                self.assertEqual(fact.support_count, self.inventory.query_support(fact.gene_ids, layers, fact.edge_query))
            if fact.edge_query == "any_direction" and fact.layer_scope["kind"] == "layers":
                sizes.add(len(layers))
            if fact.edge_query == "any_direction" and fact.layer_scope == ALL and fact.support_count is not None:
                full_counts.add(fact.support_count)
        self.assertEqual(sizes, {1, 2, 3, 4})
        self.assertEqual(full_counts, {0, 1, 2, 3, 4})
        self.assertTrue(all(len(splits) == 1 for splits in pair_splits.values()))

    def test_validator_rejects_unobserved_totals_false_completion_and_lost_partials(self) -> None:
        """Plausible true source facts are still invalid when the trace did not acquire them."""
        mutations = []
        row = self._row(mode="top_k", top_k=1)
        result = row["messages"][-1]["content"]["result"]
        result["supporting_layer_count"] = 3
        result["collection"]["total_count"] = 3
        mutations.append(row)
        row = self._row(variant="terminal")
        row["messages"][-1]["content"].update(continuation_decision="stop", blocked_reason=None)
        mutations.append(row)
        row = self._row(variant="terminal")
        row["messages"][-1]["content"]["result"]["partial_results"] = []
        mutations.append(row)
        row = self._row(variant="recovered")
        messages = [message for message in row["messages"] if "tool_call" in message]
        messages[-1]["tool_call"]["arguments"] = {**messages[-1]["tool_call"]["arguments"], "layer_scope": ALL}
        mutations.append(row)
        for index, row in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                validate_row(row, self.view)

    def test_validator_rejects_true_but_unobserved_reverse_arc_and_relabelled_evidence(self) -> None:
        """Source truth cannot replace the actual acquisition's orientation and scope limits."""
        row = self._row((B, C), mode="exists", variant="later_witness")
        layer = row["messages"][-1]["content"]["result"]["layer_results"][0]
        layer["edges"].append({"gene_ids": [C, B], "weight": .33})
        layer["coverage"]["checked_pairs"] = [[B, C], [C, B]]
        layer["coverage"]["unchecked_pairs"] = []
        with self.assertRaises(ValueError):
            validate_row(row, self.view)
        row = self._row(variant="partial")
        context = json.loads(row["messages"][1]["content"])
        context["observations"][0]["provenance"]["graph"][0]["layer_scope"] = ALL
        row["messages"][1]["content"] = json.dumps(context)
        with self.assertRaises(ValueError):
            validate_row(row, self.view)

    def test_direction_diagnostic_uses_manifest_without_graph_acquisition(self) -> None:
        """A directed predicate over a visible undirected layer is a local invalid request."""
        unit, split = _pair_group((A, B), SEED)
        for query in ("source_to_target", "both_directions"):
            fact = MultiplexFact(split, (A, B), ALL, query, None, unit)
            row = build_direction_diagnostic(self.view, fact, "step", 0, self.bank, self.contract)
            validate_row(row, self.view)
            self.assertEqual(_calls(row), [])
            answer = row["messages"][-1]["content"]
            self.assertEqual((answer["status"], answer["continuation_decision"]), ("invalid_input", "defer"))
            self.assertEqual(answer["result"]["issue_code"], "DIRECTION_UNAVAILABLE")
            self.assertEqual(answer["result"]["request"]["layer_scope"], ALL)
            self.assertEqual(answer["provenance"], {"registry": [], "graph": []})

    def test_production_budget_is_never_implicitly_selected(self) -> None:
        """The default call must stop before reading sources or writing any corpus files."""
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "corpus"
            with self.assertRaises(ValueError):
                build_corpus(target)
            with self.assertRaises(ValueError):
                build_corpus(target, smoke=True, counts={"train": 1, "validation": 1, "test": 1})
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
