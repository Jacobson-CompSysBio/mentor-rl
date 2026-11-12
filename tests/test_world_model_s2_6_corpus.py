"""Check finite-set corpus traces against source truth and visible evidence."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model import s2_6_corpus as corpus
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_nodes_by_layer import ARGUMENT_KEYS
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
PPI, COEX, BRAIN, CD4 = "HumanNetV3:string_ppi", "HumanNetV3:coexpression", "TFs:brain", "TFs:cd4+t_cells"
SEED = 20261028
SPLITS = ("train", "validation", "test")


def _calls(row: dict) -> list[dict]:
    """Read current-turn calls without supplied context observations."""
    return [message["tool_call"] for message in row["messages"] if "tool_call" in message]


def _context(row: dict) -> dict:
    """Read the exact model-visible harness data."""
    return json.loads(row["messages"][1]["content"])


def _answer(row: dict) -> dict:
    """Read the final structured answer without prose or tool actions."""
    return row["messages"][-1]["content"]


class WorldModelS26CorpusTests(unittest.TestCase):
    """Keep query subjects fixed by split, including isolates and graph-only IDs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create real node inventories with enough identities in every fixed split."""
        pools = defaultdict(list)
        for number in range(1, 700):
            gene = f"ENSG{number:011d}"
            pools[gene_group(gene, SEED)[1]].append(gene)
        cls.pools = {split: tuple(pools[split][:64]) for split in SPLITS}
        specifications = {layer: [directed, {}, set()] for layer, directed in
                          ((PPI, False), (COEX, False), (BRAIN, True), (CD4, True))}
        for pool in cls.pools.values():
            a, b, c, d, e, f, isolated, unregistered = pool[:8]
            masks = {PPI: ({(a, b): .9, (c, d): .5}, (a, b, c, d, isolated)),
                     COEX: ({(a, c): .7, (e, f): .3}, (a, c, e, f)),
                     BRAIN: ({(b, a): .2, (a, c): .8, (d, e): .6}, (a, b, c, d, e)),
                     CD4: ({(f, a): 1, (f, unregistered): .4}, (a, f, unregistered))}
            for layer, (edges, nodes) in masks.items():
                specifications[layer][1].update(edges)
                specifications[layer][2].update(nodes)
        cls.view = GraphView("unit_nodes_corpus_view", "e" * 64, {
            layer: LayerGraph(layer, directed, edges, frozenset(nodes))
            for layer, (directed, edges, nodes) in specifications.items()
        })
        cls.a, cls.b, cls.c, cls.d, cls.e, cls.f, cls.isolated, cls.unregistered, cls.missing = cls.pools["train"][:9]
        cls.genes = (cls.e, cls.a, cls.isolated, cls.missing)
        cls.registry_ids = tuple(gene for pool in cls.pools.values() for i, gene in enumerate(pool) if i != 7)
        cls.registry = RegistryContext("110", "ensembl_homo_sapiens_110_unit", {
            gene: f"UNIT_{number}" for number, gene in enumerate(cls.registry_ids, 1)
        })
        cls.index = NeighborIndex(cls.view)
        cls.contract = corpus.load_build_contract()
        cls.bank = json.loads((ROOT / "runtime/world_model/s2_v1_prompt_banks.json").read_text())

    def _fact(self, genes=None, layer=BRAIN, split=None):
        """Measure one source-derived set without a cross-split query."""
        return corpus._fact(self.view, layer, self.genes if genes is None else genes, SEED, split=split)

    def _row(self, genes=None, layer=BRAIN, *, framing="direct", variant="lookup", split=None):
        """Build one finite classification with a selected evidence contrast."""
        return corpus.build_row(self.view, self._fact(genes, layer, split), framing, variant, 0,
                                self.bank, self.contract, registry=self.registry, index=self.index, seed=SEED)

    def _validate(self, row):
        """Replay source facts and require only visible evidence in the final answer."""
        corpus.validate_row(row, self.view, bank=self.bank, contract=self.contract,
                            registry=self.registry, index=self.index)

    def test_three_categories_partition_every_requested_id_with_distinct_order_rules(self):
        row = self._row()
        self._validate(row)
        result = _answer(row)["result"]
        self.assertEqual(_calls(row)[0]["arguments"]["gene_ids"], list(self.genes))
        self.assertEqual(tuple(_calls(row)[0]["arguments"]), ARGUMENT_KEYS)
        self.assertEqual(result["layer_gene_ids"], sorted((self.e, self.a)))
        self.assertEqual(result["not_in_layer_gene_ids"], [self.isolated])
        self.assertEqual(result["missing_gene_ids"], [self.missing])
        self.assertEqual(result["coverage"], {"operation": "layer_membership", "requested_gene_ids": list(self.genes),
            "checked_gene_ids": list(self.genes), "unchecked_gene_ids": []})
        self.assertEqual(result["layer_gene_count"], 2)
        self.assertEqual(result["collection"], {"mode": "complete", "returned_count": 2,
            "total_count": 2, "complete": True, "next_cursor": None})

    def test_complete_negatives_stop_direct_tasks_and_continue_workflow_steps(self):
        for genes in ([self.isolated], [self.missing], [self.isolated, self.missing]):
            for framing, decision in (("direct", "stop"), ("step", "continue")):
                row = self._row(genes, framing=framing)
                self._validate(row)
                answer = _answer(row)
                self.assertEqual((answer["status"], answer["continuation_decision"]), ("resolved", decision))
                self.assertIsNone(answer["blocked_reason"])
                self.assertEqual(answer["result"]["layer_gene_ids"], [])
                self.assertEqual(answer["result"]["coverage"]["checked_gene_ids"], genes)

    def test_empty_list_is_a_split_specific_no_call_classification(self):
        units = set()
        for split in SPLITS:
            row = self._row([], split=split)
            self._validate(row)
            self.assertEqual(_calls(row), [])
            self.assertEqual(_answer(row)["provenance"], {"registry": [], "graph": []})
            self.assertEqual(_answer(row)["result"]["coverage"]["requested_gene_ids"], [])
            self.assertEqual(_answer(row)["result"]["layer_gene_count"], 0)
            self.assertEqual(row["metadata"]["split"], split)
            units.add(row["metadata"]["split_unit"])
        self.assertEqual(len(units), 3)
        with self.assertRaises(VariantUnavailable):
            self._row([], variant="context")

    def test_duplicates_and_reordered_inputs_preserve_the_current_first_mentions(self):
        baseline = self._row()
        duplicate = self._row(variant="duplicates")
        reverse = self._row(variant="reordered")
        for row in (baseline, duplicate, reverse):
            self._validate(row)
        self.assertGreater(len(duplicate["metadata"]["input_genes"]), len(self.genes))
        self.assertEqual(_calls(duplicate)[0]["arguments"]["gene_ids"], list(self.genes))
        self.assertEqual(_calls(reverse)[0]["arguments"]["gene_ids"], list(reversed(self.genes)))
        for category in corpus.CATEGORIES:
            self.assertEqual(_answer(baseline)["result"][category], _answer(duplicate)["result"][category])
            self.assertEqual(_answer(baseline)["result"][category], _answer(reverse)["result"][category])

    def test_complete_larger_overlapping_and_duplicate_context_need_no_call(self):
        for variant, evidence_count in (("context", 1), ("larger_context", 1),
                                        ("overlapping_context", 2), ("duplicate_context", 1)):
            row = self._row(variant=variant)
            self._validate(row)
            self.assertEqual(_calls(row), [])
            result = _answer(row)["result"]
            self.assertEqual(result["coverage"]["requested_gene_ids"], list(self.genes))
            self.assertEqual(result["layer_gene_count"], 2)
            self.assertEqual(len(_answer(row)["provenance"]["graph"]), evidence_count)
            if variant == "larger_context":
                saved = _context(row)["observations"][0]
                self.assertGreater(len(saved["arguments"]["gene_ids"]), len(self.genes))
                self.assertEqual(_answer(row)["provenance"], saved["provenance"])

    def test_partial_context_queries_only_the_still_unknown_ids(self):
        row = self._row(variant="partial_context")
        self._validate(row)
        self.assertEqual(_calls(row)[0]["arguments"]["gene_ids"], list(self.genes[2:]))
        self.assertEqual(len(_answer(row)["provenance"]["graph"]), 2)
        self.assertEqual(_answer(row)["result"]["layer_gene_count"], 2)

    def test_global_negative_context_is_sufficient_but_global_positive_is_not(self):
        row = self._row(variant="global_context")
        self._validate(row)
        self.assertEqual(_calls(row)[0]["arguments"]["gene_ids"], list(self.genes[:-1]))
        self.assertEqual(_answer(row)["result"]["missing_gene_ids"], [self.missing])
        negative = self._row([self.missing], variant="global_context")
        self._validate(negative)
        self.assertEqual(_calls(negative), [])
        self.assertEqual(_answer(negative)["result"]["missing_gene_ids"], [self.missing])

    def test_complete_s25_context_supplies_negatives_bounded_omission_needs_a_call(self):
        complete = self._row(variant="membership_context")
        self._validate(complete)
        self.assertEqual(_calls(complete), [])
        bounded = self._row(variant="bounded_membership_context")
        self._validate(bounded)
        self.assertEqual(_calls(bounded)[0]["arguments"]["gene_ids"], list(self.genes[:2]))
        saved = _context(bounded)["observations"]
        self.assertFalse(saved[0]["result"]["collection"]["complete"])
        self.assertTrue(saved[2]["result"]["collection"]["complete"])
        self.assertEqual(_answer(complete)["result"], _answer(bounded)["result"])

    def test_neighbor_zero_proves_present_membership_without_a_directional_edge(self):
        row = self._row([self.e], variant="neighbor_zero")
        self._validate(row)
        self.assertEqual(_calls(row), [])
        saved = _context(row)["observations"][0]
        self.assertEqual(saved["arguments"]["direction"], "out")
        self.assertEqual(saved["result"]["neighbors"], [])
        self.assertEqual(_answer(row)["result"]["layer_gene_ids"], [self.e])

    def test_neighbor_context_classifies_all_categories_without_a_new_call(self):
        row = self._row(variant="neighbor_context")
        self._validate(row)
        self.assertEqual(_calls(row), [])
        self.assertEqual(_answer(row)["result"]["missing_gene_ids"], [self.missing])
        self.assertEqual(_answer(row)["result"]["not_in_layer_gene_ids"], [self.isolated])

    def test_symbol_context_retains_raw_symbols_and_registry_evidence(self):
        row = self._row(variant="symbol_context")
        self._validate(row)
        raw = row["metadata"]["input_genes"]
        self.assertTrue(all(symbol.startswith("UNIT_") for symbol in raw))
        self.assertTrue(all(symbol in row["messages"][2]["content"] for symbol in raw))
        self.assertEqual(_calls(row)[0]["arguments"]["gene_ids"], list(self.genes))
        self.assertEqual(len(_answer(row)["provenance"]["registry"]), len(self.genes))
        with self.assertRaises(VariantUnavailable):
            self._row([self.unregistered], variant="symbol_context")

    def test_stale_or_wrong_layer_context_requires_fresh_classification(self):
        for variant in ("stale", "wrong_layer"):
            row = self._row(self.genes[:-1] if variant == "wrong_layer" else None, variant=variant)
            self._validate(row)
            self.assertEqual(len(_calls(row)), 1)
            saved = _context(row)["observations"][0]["provenance"]["graph"][0]
            self.assertNotIn(saved, _answer(row)["provenance"]["graph"])

    def test_initial_terminal_and_exhausted_failures_leave_every_input_unchecked(self):
        for variant, count in (("terminal", 1), ("exhausted", 2)):
            row = self._row(variant=variant)
            self._validate(row)
            answer = _answer(row)
            self.assertEqual(len(_calls(row)), count)
            self.assertEqual((answer["status"], answer["continuation_decision"]), ("error", "defer"))
            self.assertTrue(answer["blocked_reason"])
            self.assertEqual(answer["provenance"], {"registry": [], "graph": []})
            self.assertEqual(answer["result"]["partial_results"], [])
            self.assertEqual(answer["result"]["coverage"]["unchecked_gene_ids"], list(self.genes))
            self.assertEqual(answer["result"]["coverage"]["checked_gene_ids"], [])
            self.assertNotIn("missing_gene_ids", answer["result"])
            self.assertNotIn("layer_gene_count", answer["result"])
            if variant == "exhausted":
                self.assertEqual(_calls(row)[0], _calls(row)[1])

    def test_authorized_retry_recovers_with_identical_arguments(self):
        row = self._row(variant="recovered", framing="step")
        self._validate(row)
        self.assertEqual(len(_calls(row)), 2)
        self.assertEqual(_calls(row)[0], _calls(row)[1])
        self.assertEqual((_answer(row)["status"], _answer(row)["continuation_decision"]), ("resolved", "continue"))
        self.assertEqual(len(_answer(row)["provenance"]["graph"]), 1)

    def test_partial_failure_keeps_supported_work_without_classifying_failed_inputs(self):
        for variant in ("partial_terminal", "partial_exhausted"):
            row = self._row(variant=variant)
            self._validate(row)
            result = _answer(row)["result"]
            self.assertEqual(result["coverage"]["checked_gene_ids"], list(self.genes[:2]))
            self.assertEqual(result["coverage"]["unchecked_gene_ids"], list(self.genes[2:]))
            self.assertEqual(result["partial_results"], _context(row)["observations"])
            self.assertEqual(result["arguments"], _calls(row)[-1]["arguments"])

    def test_multiple_batches_require_a_visible_limit_and_cover_the_full_unique_union(self):
        row = self._row(variant="batched")
        self._validate(row)
        self.assertEqual(_context(row)["batch_limit"], 2)
        self.assertEqual([call["arguments"]["gene_ids"] for call in _calls(row)],
                         [list(self.genes[:2]), list(self.genes[2:])])
        self.assertEqual(_answer(row)["result"]["coverage"]["checked_gene_ids"], list(self.genes))
        self.assertEqual(len(_answer(row)["provenance"]["graph"]), 2)
        self.assertEqual(_answer(row)["result"]["layer_gene_count"], 2)

    def test_later_failed_batch_stays_unchecked_and_retains_the_earlier_batch(self):
        for variant, call_count in (("batched_terminal", 2), ("batched_exhausted", 3)):
            row = self._row(variant=variant)
            self._validate(row)
            result = _answer(row)["result"]
            self.assertEqual(len(_calls(row)), call_count)
            self.assertEqual(result["coverage"]["checked_gene_ids"], list(self.genes[:2]))
            self.assertEqual(result["coverage"]["unchecked_gene_ids"], list(self.genes[2:]))
            self.assertEqual(len(result["partial_results"]), 1)
            self.assertEqual(result["partial_results"][0]["arguments"]["gene_ids"], list(self.genes[:2]))
            self.assertNotIn("missing_gene_ids", result)
            self.assertNotIn("not_in_layer_gene_ids", result)

    def test_later_recovered_batch_retries_only_that_batch(self):
        row = self._row(variant="batched_recovered")
        self._validate(row)
        self.assertEqual(len(_calls(row)), 3)
        self.assertEqual(_calls(row)[1], _calls(row)[2])
        self.assertNotEqual(_calls(row)[0], _calls(row)[1])
        self.assertEqual(_answer(row)["result"]["coverage"]["unchecked_gene_ids"], [])
        self.assertEqual(len(_answer(row)["provenance"]["graph"]), 2)

    def test_all_six_diagnostics_keep_literal_gene_values_without_graph_evidence(self):
        expected = {"unknown_layer": ("not_found", "UNKNOWN_LAYER"),
                    "unknown_multiplex": ("not_found", "UNKNOWN_MULTIPLEX"),
                    "missing_scope": ("invalid_input", "MISSING_SCOPE"),
                    "missing_genes": ("invalid_input", "MISSING_GENES"),
                    "invalid_scalar": ("invalid_input", "INVALID_GENE_LIST"),
                    "invalid_list": ("invalid_input", "INVALID_GENE_LIST")}
        self.assertEqual(set(corpus.DIAGNOSTICS), set(expected))
        for kind, (status, issue) in expected.items():
            row = corpus.build_diagnostic(self.view, self._fact(), "step", 0, self.bank, self.contract, kind=kind)
            self._validate(row)
            answer = _answer(row)
            self.assertEqual(_calls(row), [])
            self.assertEqual((answer["status"], answer["continuation_decision"]), (status, "defer"))
            self.assertEqual(answer["result"]["issue_code"], issue)
            self.assertEqual(answer["provenance"], {"registry": [], "graph": []})
            self.assertIsNone(answer["result"]["rejected_arguments"])
            self.assertTrue(answer["blocked_reason"])
            raw = answer["result"]["request"]["genes"]
            self.assertIn(json.dumps(raw, ensure_ascii=False), row["messages"][2]["content"])
            if kind == "missing_genes":
                self.assertIsNone(raw)
            elif kind == "invalid_list":
                self.assertIsInstance(raw, list)
                self.assertEqual(raw, [self.genes[0], {"invalid": "gene"}])
            elif kind != "invalid_scalar":
                self.assertEqual(raw, list(self.genes))
            self.assertEqual(raw, row["metadata"]["input_genes"])

    def test_larger_context_source_facts_do_not_depend_on_wording_bank_size(self):
        bank = deepcopy(self.bank)
        index = len(bank["train"]["direct"])
        before = corpus.build_row(self.view, self._fact(), "direct", "larger_context", index,
                                  self.bank, self.contract, registry=self.registry, index=self.index, seed=SEED)
        bank["train"]["direct"].append("{request} Please finish this request here.")
        after = corpus.build_row(self.view, self._fact(), "direct", "larger_context", index,
                                 bank, self.contract, registry=self.registry, index=self.index, seed=SEED)
        self.assertNotEqual(before["messages"][2]["content"], after["messages"][2]["content"])
        first = _context(before)["observations"][0]
        second = _context(after)["observations"][0]
        self.assertEqual(first["arguments"], second["arguments"])
        self.assertEqual(first["result"], second["result"])

    def test_validator_rejects_partition_evidence_call_and_metadata_changes(self):
        mutations = []
        row = self._row()
        _answer(row)["result"]["missing_gene_ids"] = []
        mutations.append(row)
        row = self._row()
        _answer(row)["result"]["layer_gene_count"] = 3
        mutations.append(row)
        row = self._row(variant="context")
        context = _context(row)
        context["observations"][0]["result"]["missing_gene_ids"] = []
        row["messages"][1]["content"] = json.dumps(context)
        mutations.append(row)
        row = self._row(variant="recovered")
        _calls(row)[1]["arguments"]["gene_ids"].reverse()
        mutations.append(row)
        row = self._row(variant="batched_terminal")
        _answer(row)["result"]["partial_results"] = []
        mutations.append(row)
        row = self._row()
        row["metadata"]["category_counts"] = [1, 1, 2]
        mutations.append(row)
        row = self._row()
        row["metadata"]["input_genes"] = list(reversed(self.genes))
        mutations.append(row)
        for number, row in enumerate(mutations):
            with self.subTest(mutation=number), self.assertRaises(ValueError):
                self._validate(row)

    def test_context_excludes_private_labels_and_final_prose_remains_free(self):
        row = self._row()
        self.assertEqual(set(_context(row)), {"active_graph", "observations"})
        self.assertNotIn("category_counts", row["messages"][1]["content"])
        _answer(row)["summary"] = "The requested finite classification is complete."
        self._validate(row)
        failed = self._row(variant="terminal")
        _answer(failed)["blocked_reason"] = "The membership backend must recover."
        _answer(failed)["summary"] = "The required classification remains unavailable."
        self._validate(failed)
        context = _context(row)
        context["private_category_counts"] = [2, 1, 1]
        row["messages"][1]["content"] = json.dumps(context)
        with self.assertRaises(ValueError):
            self._validate(row)

    def test_sampling_covers_feasible_triples_and_reports_unselected_sizes(self):
        counts = {split: 1 for split in SPLITS}
        settings = {"input_sizes": [0, 1, 2, 3, 8, 32, 256], "smoke": True}
        facts, audit = corpus.select_facts(self.view, self.registry_ids, counts, SEED, **settings)
        self.assertEqual((facts, audit), corpus.select_facts(self.view, self.registry_ids, counts, SEED, **settings))
        subjects, cells = defaultdict(set), defaultdict(set)
        for fact in facts:
            for gene in fact.gene_ids:
                subjects[gene].add(fact.split)
                self.assertEqual(gene_group(gene, SEED)[1], fact.split)
            self.assertEqual(sum(fact.category_counts), len(fact.gene_ids))
            cells[(fact.layer_id, fact.split, len(fact.gene_ids))].add(fact.category_counts)
        self.assertTrue(all(len(splits) == 1 for splits in subjects.values()))
        self.assertTrue(any(self.unregistered in fact.gene_ids for fact in facts))
        for cell in audit["cells"]:
            key = (cell["layer_id"], cell["split"], cell["input_size"])
            selected = {tuple(triple) for triple in cell["selected_triples"]}
            self.assertEqual(cells[key], selected)
            self.assertEqual(cell["feasible_triple_count"] - len(selected), cell["unselected_triple_count"])
        for domain in audit["domains"]:
            self.assertIn([4, 7], domain["unselected_input_size_ranges"])
            self.assertEqual(domain["attainable_input_size_range"][0], 0)
            self.assertNotIn(256, domain["selected_input_sizes"])
            self.assertLess(sum(domain["registered_category_populations"]),
                            sum(domain["whole_view_category_populations"]))
        self.assertTrue(any(item["input_size"] == 256 for item in audit["omissions"]))
        self.assertFalse(audit["edge_fact_disjoint"])

    def test_full_triple_schedule_is_complete_and_independent_of_wording_banks(self):
        facts, audit = corpus.select_facts(self.view, self.registry_ids, {split: 1 for split in SPLITS}, SEED,
                                           input_sizes=[0, 1, 2, 3], smoke=False)
        self.assertTrue(facts)
        self.assertTrue(all(cell["unselected_triple_count"] == 0 for cell in audit["cells"]))
        self.assertEqual(corpus.feasible_triple_count(3, [3, 3, 3]), 10)
        self.assertEqual(corpus.feasible_triple_count(3, [1, 1, 1]), 1)
        self.assertEqual(corpus.feasible_triple_count(4, [1, 1, 1]), 0)
        self.assertIn("independent of wording", audit["conditioning"])

    def test_cross_split_sets_and_invalid_selection_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            self._fact([self.a, self.pools["validation"][0]])
        with self.assertRaises(ValueError):
            self._fact([self.a], split="test")
        for sizes in ([], [-1], [True]):
            with self.assertRaises(ValueError):
                corpus.select_facts(self.view, self.registry_ids, {split: 1 for split in SPLITS}, SEED,
                                    input_sizes=sizes)
        with self.assertRaises(ValueError):
            corpus.select_facts(self.view, self.registry_ids, {split: 0 for split in SPLITS}, SEED, input_sizes=[1])


if __name__ == "__main__":
    unittest.main()
