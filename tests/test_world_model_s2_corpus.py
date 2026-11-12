"""Exercise S2 corpus selection, evidence use, and immutable build outputs.

The small graphs below are explicit unit-test fixtures, never training sources.
Production source pins and loading have their own source tests.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime.world_model.s2_corpus import (
    BUCKETS,
    FRAMINGS,
    SCHEMA_VERSION,
    SPLITS,
    VARIANTS,
    EdgeFact,
    build_corpus,
    build_row,
    select_facts,
    validate_row,
)
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
PPI = "HumanNetV3:string_ppi"
TF = "TFs:brain"


def _gene(number: int) -> str:
    """Return a canonical-looking identifier solely for the unit fixture."""
    return f"ENSG{number:011d}"


def _view() -> GraphView:
    """Provide native cycles with overlap, layer absences, and registry absences."""
    first = tuple(_gene(i) for i in range(120))
    second = tuple(_gene(i) for i in range(60, 180))
    undirected = {
        tuple(sorted((gene, first[(i + 1) % len(first)]))): (i + 1) / 1000
        for i, gene in enumerate(first)
    }
    directed = {
        (gene, second[(i - 1) % len(second)]): 0.123456789
        for i, gene in enumerate(second)
    }
    directed[(_gene(62), _gene(61))] = 0
    return GraphView(
        "unit_test_view", "a" * 64,
        {
            PPI: LayerGraph(PPI, False, undirected, frozenset(first)),
            TF: LayerGraph(TF, True, directed, frozenset(second)),
        },
    )


def _calls(row: dict) -> list[dict]:
    """Extract the current-turn calls without treating saved context as a call."""
    return [message["tool_call"] for message in row["messages"] if "tool_call" in message]


class WorldModelS2CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Use the real wording bank with an isolated test-only wire contract."""
        cls.view = _view()
        cls.registry_ids = tuple(_gene(i) for i in range(210))
        cls.bank = json.loads((ROOT / "runtime/world_model/s2_v1_prompt_banks.json").read_text())
        cls.contract = {
            "row_schema_version": SCHEMA_VERSION,
            "contract_version": "unit-test-s2-v1",
            "implemented_families": ["S2.1.single_edge"],
            "training_ready": False,
            "system_prompt": "Answer from visible evidence with the declared S2 envelope.",
            "tools": [{"name": "get_edge"}],
            "remaining_coverage": ["This is an isolated unit-test build."],
        }
        requests = ROOT / "runtime/world_model/s2_v1_edge_requests.json"
        cls.contract["request_bank"] = {
            "path": str(requests), "sha256": hashlib.sha256(requests.read_bytes()).hexdigest(),
        }
        cls.contract["_edge_requests"] = json.loads(requests.read_text())

    def _fact(self, bucket: str = "recorded", *, directed: bool = False) -> EdgeFact:
        """Choose explicit witnesses whose semantics are independently readable."""
        pairs = {
            "recorded": (_gene(2), _gene(1)),
            "no_edge": (_gene(1), _gene(3)),
            "outside_layer": (_gene(1), _gene(150)),
            "outside_multiplex": (_gene(1), _gene(190)),
        }
        pair = (_gene(61), _gene(60)) if directed else pairs[bucket]
        return EdgeFact("train", TF if directed else PPI, pair, bucket, "unit:" + ":".join(sorted(pair)))

    def _row(self, bucket: str = "recorded", framing: str = "direct", variant: str = "lookup") -> dict:
        """Render a new row so mutation tests cannot alter another test's row."""
        return build_row(self.view, self._fact(bucket), framing, variant, 3, self.bank, self.contract)

    def test_fact_selection_is_reproducible_stratified_and_pair_grouped(self) -> None:
        """Outcome, layer, framing contrasts cannot leak one pair across splits."""
        counts = {"train": 2, "validation": 1, "test": 1}
        first = select_facts(self.view, self.registry_ids, counts, 20261028)
        self.assertEqual(first, select_facts(self.view, self.registry_ids, counts, 20261028))
        expected = Counter({(split, layer, bucket): n for split, n in counts.items()
                            for layer in self.view.layer_order for bucket in BUCKETS})
        self.assertEqual(Counter((f.split, f.layer_id, f.bucket) for f in first), expected)
        pair_splits, unit_splits = {}, {}
        for fact in first:
            pair = tuple(sorted(fact.gene_ids))
            self.assertEqual(pair_splits.setdefault(pair, fact.split), fact.split)
            self.assertEqual(unit_splits.setdefault(fact.split_unit, fact.split), fact.split)
            result = self.view.get_edge(self.view.multiplex_id, fact.layer_id,
                                        list(fact.gene_ids), evidence_id="test-selection")["result"]
            if fact.bucket == "recorded":
                self.assertIs(result["edge_exists"], True)
            elif fact.bucket == "no_edge":
                self.assertIs(result["edge_exists"], False)
                self.assertEqual(result["missing_gene_ids"] + result["not_in_layer_gene_ids"], [])
            elif fact.bucket == "outside_layer":
                self.assertTrue(result["not_in_layer_gene_ids"])
                self.assertEqual(result["missing_gene_ids"], [])
            else:
                self.assertTrue(result["missing_gene_ids"])
                self.assertIsNone(result["edge_exists"])

    def test_invalid_or_unattainable_selection_cannot_silently_drop_strata(self) -> None:
        """Invalid quotas and a missing source population must fail explicitly."""
        for counts in ({"train": 1}, {"train": 0, "validation": 1, "test": 1},
                       {"train": True, "validation": 1, "test": 1}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                select_facts(self.view, self.registry_ids, counts, 1)
        with self.assertRaises(ValueError):
            select_facts(self.view, tuple(self.view.multiplex_nodes), dict.fromkeys(SPLITS, 1), 1)

    def test_outcomes_and_workflow_decisions_preserve_false_versus_null(self) -> None:
        """A scoped negative completes either framing; global absence still defers."""
        for bucket in BUCKETS:
            for framing in FRAMINGS:
                with self.subTest(bucket=bucket, framing=framing):
                    row = self._row(bucket, framing)
                    validate_row(row, self.view)
                    answer = row["messages"][-1]["content"]
                    self.assertEqual(list(answer), ["status", "provenance", "result",
                                     "continuation_decision", "blocked_reason", "summary"])
                    if bucket == "outside_multiplex":
                        self.assertEqual((answer["status"], answer["continuation_decision"]), ("not_found", "defer"))
                        self.assertIsNone(answer["result"]["edge_exists"])
                        self.assertTrue(answer["blocked_reason"])
                    else:
                        self.assertEqual(answer["status"], "resolved")
                        self.assertEqual(answer["continuation_decision"], "stop" if framing == "direct" else "continue")
                        self.assertIsNone(answer["blocked_reason"])
                        self.assertIs(answer["result"]["edge_exists"], bucket == "recorded")
                    if bucket != "recorded":
                        self.assertIsNone(answer["result"]["weight"])

    def test_direction_input_order_and_zero_weight_survive_rendering(self) -> None:
        """An undirected result may sort its pair; a directed result must not."""
        undirected = self._row()
        self.assertEqual(_calls(undirected)[0]["arguments"]["gene_ids"], [_gene(2), _gene(1)])
        self.assertEqual(undirected["messages"][-1]["content"]["result"]["gene_ids"], [_gene(1), _gene(2)])
        fact = self._fact(directed=True)
        directed = build_row(self.view, fact, "direct", "lookup", 0, self.bank, self.contract)
        self.assertEqual(directed["messages"][-1]["content"]["result"]["gene_ids"], list(fact.gene_ids))
        request = directed["messages"][2]["content"]
        self.assertIn(fact.gene_ids[0], request)
        self.assertIn(fact.gene_ids[1], request)
        self.assertLess(request.index(fact.gene_ids[0]), request.index(fact.gene_ids[1]))
        zero = EdgeFact("train", TF, (_gene(62), _gene(61)), "recorded", "unit-zero")
        row = build_row(self.view, zero, "direct", "lookup", 0, self.bank, self.contract)
        validate_row(row, self.view)
        self.assertIs(row["messages"][-1]["content"]["result"]["edge_exists"], True)
        self.assertEqual(row["messages"][-1]["content"]["result"]["weight"], 0)

    def test_saved_evidence_answers_without_a_new_acquisition(self) -> None:
        """Positive, checked-negative, and missing-prerequisite context are reusable."""
        for bucket in BUCKETS:
            row = self._row(bucket, variant="context")
            validate_row(row, self.view)
            self.assertEqual(_calls(row), [])
            self.assertFalse(any(m["role"] == "tool" for m in row["messages"]))
            saved = json.loads(row["messages"][1]["content"])["observations"][0]
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer["result"], saved["result"])
            self.assertEqual(answer["provenance"], saved["provenance"])
            self.assertEqual(row["tools"], self.contract["tools"])

    def test_failure_and_retry_contrasts_keep_exact_requests_and_no_failed_evidence(self) -> None:
        """Recovery ends retrying; exhaustion never turns an unchecked edge into false."""
        for bucket in BUCKETS:
            for variant in ("terminal", "recovered", "exhausted"):
                with self.subTest(bucket=bucket, variant=variant):
                    row = self._row(bucket, "step", variant)
                    validate_row(row, self.view)
                    calls = _calls(row)
                    self.assertEqual(len(calls), 1 if variant == "terminal" else 2)
                    self.assertTrue(all(call == calls[0] for call in calls))
                    answer = row["messages"][-1]["content"]
                    if variant == "recovered":
                        self.assertEqual(answer["continuation_decision"], "defer" if bucket == "outside_multiplex" else "continue")
                        self.assertEqual(len(answer["provenance"]["graph"]), 1)
                    else:
                        self.assertEqual((answer["status"], answer["continuation_decision"]), ("error", "defer"))
                        self.assertEqual(answer["provenance"], {"registry": [], "graph": []})
                        self.assertEqual(answer["result"], {
                            "failed_tool": "get_edge", "error_code": "BACKEND_UNAVAILABLE",
                            "arguments": calls[-1]["arguments"], "partial_results": [], "coverage": None,
                        })

    def test_reasoning_is_bound_and_wording_changes_do_not_change_facts(self) -> None:
        """Changing the bank affects wording, not selected inputs or graph answers."""
        alternate = deepcopy(self.bank)
        for framing in FRAMINGS:
            alternate["train"][framing] = list(reversed(alternate["train"][framing]))[:16]
        for variant in VARIANTS:
            row = self._row(variant=variant)
            changed = build_row(self.view, self._fact(), "direct", variant, 3, alternate, self.contract)
            self.assertNotEqual(row["messages"][2]["content"], changed["messages"][2]["content"])
            self.assertEqual(_calls(row), _calls(changed))
            self.assertEqual(row["messages"][-1]["content"]["result"], changed["messages"][-1]["content"]["result"])
            self.assertEqual(row["metadata"]["split_unit"], changed["metadata"]["split_unit"])
            reasoning = " ".join(m.get("thinking", "") for m in row["messages"])
            for value in (*self._fact().gene_ids, PPI):
                self.assertIn(value, reasoning)
            self.assertEqual(row["metadata"]["supervision"], "current_turn_assistant_only")
            self.assertIs(row["metadata"]["training_ready"], False)

    def test_validator_rejects_false_completion_and_mismatched_visible_evidence(self) -> None:
        """Corrupted branches must fail even when their remaining JSON is well formed."""
        mutations = []
        row = self._row(variant="terminal")
        row["messages"][-1]["content"].update(continuation_decision="stop", blocked_reason=None)
        mutations.append(row)
        for key, value in (("failed_tool", "get_neighbors"), ("error_code", "INVENTED"), ("coverage", {})):
            row = self._row(variant="terminal")
            row["messages"][-1]["content"]["result"][key] = value
            mutations.append(row)
        row = self._row(variant="context")
        harness = json.loads(row["messages"][1]["content"])
        harness["active_graph"]["graph_version"] = "b" * 64
        row["messages"][1]["content"] = json.dumps(harness)
        mutations.append(row)
        row = self._row(variant="context")
        harness = json.loads(row["messages"][1]["content"])
        harness["observations"][0]["arguments"]["gene_ids"] = [_gene(1), _gene(3)]
        row["messages"][1]["content"] = json.dumps(harness)
        mutations.append(row)
        row = self._row(variant="recovered")
        calls = [m for m in row["messages"] if "tool_call" in m]
        calls[1]["tool_call"]["arguments"] = {**calls[1]["tool_call"]["arguments"], "gene_ids": [_gene(1), _gene(3)]}
        mutations.append(row)
        for index, row in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                validate_row(row, self.view)

    def test_build_is_reproducible_immutable_and_keeps_answers_out_of_questions(self) -> None:
        """A complete small build cannot replace different files or leak answer labels."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}\n")
            contract = deepcopy(self.contract)
            contract.update(source_manifest={"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                            prompt_bank={"path": "unit-test", "sha256": "0" * 64},
                            registry_index={"path": "unit-test", "sha256": "0" * 64})
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract))
            output = root / "corpus"
            with patch("runtime.world_model.s2_corpus.load_source_view", return_value=self.view), \
                 patch("runtime.world_model.s2_corpus._registry_ids", return_value=self.registry_ids), \
                 patch("runtime.world_model.s2_corpus.load_s2_prompt_banks", return_value=self.bank):
                arguments = dict(counts={"train": 16, "validation": 1, "test": 1},
                                 source_manifest=source, contract_path=contract_path)
                first = build_corpus(output, **arguments)
                self.assertEqual(first, build_corpus(output, **arguments))
                self.assertEqual(first["row_counts"], {"train": 1280, "validation": 80, "test": 80})
                training = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
                wrapper_cells = Counter((row["metadata"]["framing"], row["metadata"]["wrapper_index"],
                                         row["metadata"]["bucket"], row["metadata"]["variant"])
                                        for row in training)
                self.assertEqual(len(wrapper_cells), 2 * 32 * 4 * 5)
                self.assertEqual(set(wrapper_cells.values()), {1})
                for split in ("validation", "test"):
                    questions = [json.loads(line) for line in (output / f"eval/{split}.questions.jsonl").read_text().splitlines()]
                    self.assertEqual(len(questions), 80)
                    for question in questions:
                        self.assertFalse({"bucket", "variant", "status", "continuation_decision"} & set(question.get("metadata", {})))
                        self.assertEqual([m["role"] for m in question["messages"]], ["system", "harness", "user"])
                        self.assertNotIn("final_answer", question)
                target = output / "train.jsonl"
                target.write_text("different existing artifact\n")
                with self.assertRaises(ValueError):
                    build_corpus(output, **arguments)
                self.assertEqual(target.read_text(), "different existing artifact\n")


if __name__ == "__main__":
    unittest.main()
