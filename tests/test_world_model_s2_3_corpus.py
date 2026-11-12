"""Independent S2.3 checks over explicit small native neighbor fixtures."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import (
    NeighborFact, VariantUnavailable, build_row, gene_group,
    load_build_contract, select_facts, validate_row,
)
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
PPI, COEXPR, BRAIN, CD4 = "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells"
SEED = 20261028
A, B, C, D, E, F, G, H, MISSING = tuple(f"ENSG{i:011d}" for i in range(1, 10))


def _view() -> GraphView:
    """Provide ties, a zero-weight edge, reciprocal roles, and one native self-loop."""
    sources = {
        PPI: (False, {(A, B): .9, (A, C): .9, (A, D): .5, (A, E): 0, (B, F): .3}),
        COEXPR: (False, {(B, C): .7, (G, H): .2}),
        BRAIN: (True, {(A, A): 1, (A, B): .8, (A, C): .8, (B, A): .4,
                       (D, A): .2, (C, D): .1, (G, H): .6}),
        CD4: (True, {(B, C): .6, (F, G): .3, (D, D): .1}),
    }
    layers = {name: LayerGraph(name, directed, edges, frozenset(gene for pair in edges for gene in pair))
              for name, (directed, edges) in sources.items()}
    return GraphView("unit_neighbor_view", "c" * 64, layers)


def _calls(row: dict) -> list[dict]:
    """Return actual current-turn calls without confusing them with saved observations."""
    return [message["tool_call"] for message in row["messages"] if "tool_call" in message]


class WorldModelS23CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Bind frozen wording to a small explicit unit graph and unique test symbols."""
        cls.view = _view()
        cls.index = NeighborIndex(cls.view)
        cls.contract = load_build_contract()
        cls.bank = json.loads((ROOT / "runtime/world_model/s2_v1_prompt_banks.json").read_text())
        cls.registry_ids = tuple(f"ENSG{i:011d}" for i in range(1, 21))
        cls.registry = RegistryContext("110", "ensembl_homo_sapiens_110_unit", {
            gene: f"UNIT_{i}" for i, gene in enumerate(cls.registry_ids, 1)
        })

    def _fact(self, gene: str = A, layer: str = PPI, direction: str = "undirected") -> NeighborFact:
        """Group the query subject across every layer and directional contrast."""
        unit, split = gene_group(gene, SEED)
        category = ("outside_multiplex" if gene not in self.view.multiplex_nodes else
                    "outside_layer" if gene not in self.view.layers[layer].nodes else "present")
        sides = ("in", "out") if direction == "both" else (direction,)
        counts = {side: None if category == "outside_multiplex" else len(self.index.neighbors(layer, gene, side))
                  for side in sides}
        return NeighborFact(split, gene, layer, direction, category, counts, unit)

    def _row(self, gene: str = A, layer: str = PPI, direction: str = "undirected", *,
             mode: str = "complete", top_k: int | None = None, framing: str = "direct",
             variant: str = "lookup") -> dict:
        """Render one source-bound contrast using the public builder API."""
        return build_row(self.view, self._fact(gene, layer, direction), mode, top_k,
                         framing, variant, 0, self.bank, self.contract,
                         index=self.index, registry=self.registry)

    def _validate(self, row: dict) -> None:
        """Use the same explicit fixture registry for supplied-symbol checks."""
        validate_row(row, self.view, index=self.index, bank=self.bank,
                     contract=self.contract, registry=self.registry)

    def test_ranked_complete_and_exact_tie_cutoff_preserve_zero_weight(self) -> None:
        """Weights sort numerically; ties use ID order and never enlarge K."""
        for framing in ("direct", "step"):
            row = self._row(framing=framing)
            self._validate(row)
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["result"]["neighbors"], [
                {"gene_id": B, "weight": .9}, {"gene_id": C, "weight": .9},
                {"gene_id": D, "weight": .5}, {"gene_id": E, "weight": 0},
            ])
            self.assertEqual(answer["continuation_decision"], "stop" if framing == "direct" else "continue")
        row = self._row(mode="top_k", top_k=1)
        self._validate(row)
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["neighbors"], [{"gene_id": B, "weight": .9}])
        self.assertEqual(result["collection"], {"mode": "top_k", "returned_count": 1,
                         "total_count": 4, "complete": False, "next_cursor": None})

    def test_source_self_loop_and_two_directional_weights_remain_separate(self) -> None:
        """A self-loop occurs once per side; a reciprocal neighbor keeps both native weights."""
        row = self._row(layer=BRAIN, direction="both")
        self._validate(row)
        self.assertEqual([call["arguments"]["direction"] for call in _calls(row)], ["in", "out"])
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["incoming"]["neighbors"], [
            {"gene_id": A, "weight": 1}, {"gene_id": B, "weight": .4}, {"gene_id": D, "weight": .2}])
        self.assertEqual(result["outgoing"]["neighbors"], [
            {"gene_id": A, "weight": 1}, {"gene_id": B, "weight": .8}, {"gene_id": C, "weight": .8}])
        self.assertEqual(result["coverage"]["checked_directions"], ["in", "out"])
        self.assertNotIn("neighbor_count", result)

    def test_compound_top_k_applies_the_bound_to_each_side(self) -> None:
        """Both requested prefixes can be complete tasks while both full collections are incomplete."""
        row = self._row(layer=BRAIN, direction="both", mode="top_k", top_k=2)
        self._validate(row)
        result = row["messages"][-1]["content"]["result"]
        for side in ("incoming", "outgoing"):
            self.assertEqual([entry["gene_id"] for entry in result[side]["neighbors"]], [A, B])
            self.assertEqual(result[side]["collection"]["returned_count"], 2)
            self.assertEqual(result[side]["collection"]["total_count"], 3)
            self.assertFalse(result[side]["collection"]["complete"])
        self.assertEqual(result["incoming"]["neighbors"][1]["weight"], .4)
        self.assertEqual(result["outgoing"]["neighbors"][1]["weight"], .8)
        self.assertEqual(result["coverage"]["checked_directions"], ["in", "out"])

    def test_present_directional_zero_layer_absence_and_global_absence_are_distinct(self) -> None:
        """Only missing global membership makes the requested operation unavailable."""
        for gene, expected_outside in ((G, []), (F, [F])):
            row = self._row(gene, BRAIN, "in")
            self._validate(row)
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["status"], "resolved")
            self.assertEqual(answer["result"]["neighbors"], [])
            self.assertEqual(answer["result"]["not_in_layer_gene_ids"], expected_outside)
            self.assertEqual(answer["result"]["collection"]["total_count"], 0)
        row = self._row(MISSING)
        self._validate(row)
        answer = row["messages"][-1]["content"]
        self.assertEqual((answer["status"], answer["continuation_decision"]), ("not_found", "defer"))
        self.assertIsNone(answer["result"]["neighbors"])
        self.assertIsNone(answer["result"]["collection"])

    def test_compound_membership_findings_avoid_an_unnecessary_second_lookup(self) -> None:
        """Layer absence proves both empties; global absence leaves the unqueried side null."""
        for gene in (F, MISSING):
            row = self._row(gene, BRAIN, "both")
            self._validate(row)
            self.assertEqual(len(_calls(row)), 1)
            result = row["messages"][-1]["content"]["result"]
            if gene == F:
                self.assertEqual(result["incoming"]["neighbors"], [])
                self.assertEqual(result["outgoing"]["neighbors"], [])
                self.assertEqual(result["coverage"]["checked_directions"], ["in", "out"])
            else:
                self.assertIsNone(result["incoming"]["neighbors"])
                self.assertIsNone(result["outgoing"])
                self.assertEqual(result["coverage"]["checked_directions"], [])

    def test_complete_pagination_accumulates_every_page_and_copies_opaque_cursors(self) -> None:
        """The terminal page is not a complete list by itself."""
        row = self._row(variant="paged")
        self._validate(row)
        calls = _calls(row)
        responses = [m["content"] for m in row["messages"] if m["role"] == "tool"]
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["arguments"]["cursor"])
        self.assertEqual(calls[1]["arguments"]["cursor"], responses[0]["result"]["collection"]["next_cursor"])
        self.assertFalse(responses[-1]["result"]["collection"]["complete"])
        self.assertIsNone(responses[-1]["result"]["collection"]["next_cursor"])
        final = row["messages"][-1]["content"]
        self.assertEqual(len(final["result"]["neighbors"]), 4)
        self.assertTrue(final["result"]["collection"]["complete"])
        self.assertEqual(len(final["provenance"]["graph"]), 2)

    def test_paged_top_k_finishes_exact_prefix_without_claiming_a_complete_full_list(self) -> None:
        """The bound applies to accumulated rows, not independently to each page."""
        row = self._row(mode="top_k", top_k=3, variant="paged")
        self._validate(row)
        calls = _calls(row)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["arguments"]["top_k"] == 3 for call in calls))
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual([entry["gene_id"] for entry in result["neighbors"]], [B, C, D])
        self.assertEqual(result["collection"]["returned_count"], 3)
        self.assertEqual(result["collection"]["total_count"], 4)
        self.assertFalse(result["collection"]["complete"])

    def test_overlap_and_empty_intermediate_pages_do_not_change_the_selection(self) -> None:
        """Deduplicate exact overlap and follow an empty page's cursor until exhaustion."""
        for variant in ("overlap", "empty_page"):
            row = self._row(variant=variant)
            self._validate(row)
            self.assertGreaterEqual(len(_calls(row)), 3)
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual([n["gene_id"] for n in result["neighbors"]], [B, C, D, E])
            self.assertEqual(result["collection"]["returned_count"], 4)
            if variant == "empty_page":
                pages = [m["content"]["result"] for m in row["messages"] if m["role"] == "tool"]
                self.assertTrue(any(not page["neighbors"] and page["collection"]["next_cursor"] for page in pages))

    def test_unknown_total_is_inferred_only_after_full_set_exhaustion(self) -> None:
        """A bounded prefix does not reveal its private true total."""
        for mode, k, total, complete in (("complete", None, 4, True),
                                        ("top_k", 2, None, False), ("top_k", 8, 4, True)):
            row = self._row(mode=mode, top_k=k, variant="unknown_total")
            self._validate(row)
            result = row["messages"][-1]["content"]["result"]
            self.assertEqual(result["collection"]["total_count"], total)
            self.assertIs(result["collection"]["complete"], complete)

    def test_saved_full_list_can_answer_top_k_without_losing_known_total(self) -> None:
        """Ranking a checked full list needs no new acquisition."""
        row = self._row(mode="top_k", top_k=1, variant="complete_context")
        self._validate(row)
        self.assertEqual(_calls(row), [])
        result = row["messages"][-1]["content"]["result"]
        self.assertEqual(result["neighbors"], [{"gene_id": B, "weight": .9}])
        self.assertEqual(result["collection"]["total_count"], 4)
        self.assertFalse(result["collection"]["complete"])

    def test_saved_complete_prefix_continues_but_changed_selection_restarts(self) -> None:
        """A cursor belongs to its exact selection and cannot transfer from top-K to complete."""
        row = self._row(variant="prefix_context")
        self._validate(row)
        saved = json.loads(row["messages"][1]["content"])["observations"][0]
        self.assertEqual(_calls(row)[0]["arguments"]["cursor"], saved["result"]["collection"]["next_cursor"])
        row = self._row(variant="bounded_context")
        self._validate(row)
        self.assertIsNone(_calls(row)[0]["arguments"]["cursor"])
        self.assertEqual(_calls(row)[0]["arguments"]["mode"], "complete")
        self.assertEqual(len(row["messages"][-1]["content"]["provenance"]["graph"]), 2)

    def test_failed_page_keeps_partial_records_and_retry_repeats_exact_cursor(self) -> None:
        """A failure contributes no page evidence and never turns a prefix into a full list."""
        for variant in ("terminal", "recovered", "exhausted"):
            row = self._row(variant=variant)
            self._validate(row)
            calls = _calls(row)
            self.assertEqual(len(calls), 1 if variant == "terminal" else 2)
            self.assertIsNotNone(calls[0]["arguments"]["cursor"])
            self.assertTrue(all(call == calls[0] for call in calls))
            answer = row["messages"][-1]["content"]
            if variant == "recovered":
                self.assertEqual(answer["status"], "resolved")
                self.assertEqual(len(answer["result"]["neighbors"]), 4)
            else:
                self.assertEqual((answer["status"], answer["continuation_decision"]), ("error", "defer"))
                self.assertIsNone(answer["result"]["coverage"])
                self.assertEqual(len(answer["result"]["partial_results"]), 1)
                self.assertEqual(len(answer["provenance"]["graph"]), 1)

    def test_second_direction_failure_preserves_completed_incoming_coverage(self) -> None:
        """A blocked outgoing side does not discard the completed incoming side."""
        row = self._row(layer=BRAIN, direction="both", variant="second_terminal")
        self._validate(row)
        answer = row["messages"][-1]["content"]
        self.assertEqual(answer["status"], "error")
        self.assertEqual(answer["result"]["coverage"]["checked_directions"], ["in"])
        self.assertEqual(answer["result"]["coverage"]["unchecked_directions"], ["out"])
        self.assertEqual(len(answer["result"]["partial_results"]), 1)
        self.assertEqual(answer["result"]["arguments"]["direction"], "out")

    def test_partial_first_direction_failure_keeps_both_sides_unchecked(self) -> None:
        """A successful first page does not complete even its own directional task."""
        row = self._row(layer=BRAIN, direction="both", variant="terminal")
        self._validate(row)
        answer = row["messages"][-1]["content"]
        self.assertEqual(answer["status"], "error")
        self.assertEqual(answer["result"]["coverage"]["checked_directions"], [])
        self.assertEqual(answer["result"]["coverage"]["unchecked_directions"], ["in", "out"])
        self.assertEqual(len(answer["result"]["partial_results"]), 1)
        self.assertEqual(answer["result"]["arguments"]["direction"], "in")

    def test_wrong_context_is_excluded_and_distractor_subjects_share_the_split(self) -> None:
        """A matching name cannot repair the wrong version, layer, direction, or subject."""
        for variant, layer, direction in (("stale", PPI, "undirected"),
                                         ("wrong_layer", PPI, "undirected"),
                                         ("wrong_direction", BRAIN, "out"),
                                         ("distractor", PPI, "undirected")):
            row = self._row(layer=layer, direction=direction, variant=variant)
            self._validate(row)
            saved = json.loads(row["messages"][1]["content"])["observations"][0]
            self.assertEqual(len(_calls(row)), 1)
            self.assertNotIn(saved["provenance"]["graph"][0], row["messages"][-1]["content"]["provenance"]["graph"])
            if variant == "distractor":
                self.assertEqual(gene_group(saved["arguments"]["gene_id"])[1], row["metadata"]["split"])

    def test_cross_layer_global_absence_is_never_an_unnecessary_lookup_contrast(self) -> None:
        """Global absence already observed in the same view is independent of layer selection."""
        try:
            row = self._row(MISSING, variant="wrong_layer")
        except VariantUnavailable:
            return
        self._validate(row)
        self.assertEqual(_calls(row), [])
        self.assertEqual(row["messages"][-1]["content"]["status"], "not_found")

    def test_symbol_context_preserves_identity_and_obtains_graph_evidence(self) -> None:
        """A registry mapping supplies the ID but never a membership or adjacency conclusion."""
        row = self._row(variant="symbol_context")
        self._validate(row)
        self.assertIn("UNIT_1", row["messages"][2]["content"])
        self.assertEqual(_calls(row)[0]["arguments"]["gene_id"], A)
        provenance = row["messages"][-1]["content"]["provenance"]
        self.assertEqual(len(provenance["registry"]), 1)
        self.assertEqual(len(provenance["graph"]), 1)

    def test_sampling_uses_actual_count_values_and_query_subject_splits(self) -> None:
        """No missing integer counts are invented and no edge-disjointness is claimed."""
        counts = {split: 1 for split in ("train", "validation", "test")}
        facts, audit = select_facts(self.view, self.index, self.registry_ids, counts, SEED)
        self.assertEqual((facts, audit), select_facts(self.view, self.index, self.registry_ids, counts, SEED))
        subjects = defaultdict(set)
        actual_counts = defaultdict(set)
        for fact in facts:
            self.assertEqual(gene_group(fact.gene_id), (fact.split_unit, fact.split))
            subjects[fact.gene_id].add(fact.split)
            for side, number in fact.neighbor_counts.items():
                if fact.category != "outside_multiplex":
                    self.assertEqual(number, len(self.index.neighbors(fact.layer_id, fact.gene_id, side)))
                if fact.category == "present":
                    actual_counts[(fact.layer_id, side)].add(number)
        for layer in self.view.layer_order:
            for side in self.index.directions(layer):
                self.assertEqual(actual_counts[(layer, side)], set(self.index.histogram(layer, side)))
        self.assertTrue(all(len(splits) == 1 for splits in subjects.values()))
        self.assertFalse(audit["edge_fact_disjoint"])

    def test_validator_rejects_hidden_totals_last_page_only_and_changed_retry_cursor(self) -> None:
        """True source facts still require visible evidence and a valid acquisition chain."""
        mutations = []
        row = self._row(mode="top_k", top_k=2, variant="unknown_total")
        row["messages"][-1]["content"]["result"]["collection"]["total_count"] = 4
        mutations.append(row)
        row = self._row(variant="paged")
        last_page = [m["content"]["result"] for m in row["messages"] if m["role"] == "tool"][-1]
        row["messages"][-1]["content"]["result"]["neighbors"] = last_page["neighbors"]
        mutations.append(row)
        row = self._row(variant="recovered")
        _calls(row)[-1]["arguments"]["cursor"] = None
        mutations.append(row)
        row = self._row(variant="terminal")
        row["messages"][-1]["content"]["result"]["partial_results"] = []
        mutations.append(row)
        row = self._row(E)
        row["messages"][-1]["content"]["result"]["collection"]["returned_count"] = True
        mutations.append(row)
        row = self._row(variant="context")
        context = json.loads(row["messages"][1]["content"])
        context["observations"][0]["result"]["neighbors"][0]["weight"] = .123
        row["messages"][1]["content"] = json.dumps(context)
        mutations.append(row)
        for index, row in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                self._validate(row)

    def test_validator_rejects_extra_hints_malformed_wire_and_corrupted_stale_facts(self) -> None:
        """A version mismatch licenses ignoring evidence, not inventing its source contents."""
        mutations = []
        row = self._row()
        context = json.loads(row["messages"][1]["content"])
        context["private_expected_count"] = 4
        row["messages"][1]["content"] = json.dumps(context)
        mutations.append(row)
        row = self._row()
        response = next(message["content"] for message in row["messages"] if message["role"] == "tool")
        response["private_expected_decision"] = "stop"
        mutations.append(row)
        row = self._row()
        row["messages"][-1]["role"] = "user"
        mutations.append(row)
        row = self._row(variant="stale")
        self._validate(row)
        context = json.loads(row["messages"][1]["content"])
        context["observations"][0]["result"]["neighbors"][0]["weight"] = .123
        row["messages"][1]["content"] = json.dumps(context)
        mutations.append(row)
        for index, row in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                self._validate(row)


if __name__ == "__main__":
    unittest.main()
