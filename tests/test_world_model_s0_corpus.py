"""Tests for the minimal canonical S0 corpus builder."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import unittest

from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s0_corpus import (
    AMBIGUOUS_SYMBOL,
    CORPUS_SCHEMA_VERSION,
    EvidenceSequence,
    HISTORICAL_SYMBOL,
    MANIFEST_SCHEMA_VERSION,
    NAMED_GENE_ID,
    PROMPT_TEMPLATES,
    RESOLVED_SYMBOL,
    RETIRED_GENE_ID,
    S0_V4_METADATA_ADDITIONS,
    S0_V4_OVERLAY_PATH,
    S0_V4_OVERLAY_SCHEMA_VERSION,
    UNNAMED_GENE_ID,
    _build_row,
    assign_evaluation_components,
    build_evaluation_renderings,
    build_identity_components,
    build_train_renderings,
    load_s0_v4_settings,
    select_registry_facts,
    select_supplemental_facts,
)
from runtime.world_model.source_index import check_source_index
from runtime.world_model.tools import WorldModelToolRuntime


class WorldModelS0CorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = tuple(f"ENSG{number:011d}" for number in range(1, 7))
        self.genes = {
            self.ids[0]: ["Alpha", "1", "protein_coding"],
            self.ids[1]: ["Dup", "2", "lncRNA"],
            self.ids[2]: ["dup", "X", "pseudogene"],
            self.ids[3]: [None, "MT", "protein_coding"],
            self.ids[4]: ["Solo", "3", "protein_coding"],
            self.ids[5]: ["Last", "4", "protein_coding"],
        }

    def test_components_use_casefolded_symbols_and_stable_split_units(self) -> None:
        first, first_by_gene = build_identity_components(self.genes)
        second, second_by_gene = build_identity_components(self.genes)
        self.assertEqual(first, second)
        self.assertEqual(first_by_gene, second_by_gene)
        self.assertEqual(len(first), 5)
        self.assertEqual(first_by_gene[self.ids[1]], first_by_gene[self.ids[2]])
        ambiguous = [component for component in first if component.is_ambiguous]
        self.assertEqual(len(ambiguous), 1)
        self.assertEqual(
            [fact.kind for fact in ambiguous[0].facts].count(AMBIGUOUS_SYMBOL),
            2,
        )

    def test_component_assignment_is_deterministic_and_disjoint(self) -> None:
        components, _ = build_identity_components(self.genes)
        settings = {
            "validation": {
                "unambiguous_component_count": 1,
                "ambiguous_component_count": 0,
            },
            "test": {
                "unambiguous_component_count": 1,
                "ambiguous_component_count": 1,
            },
        }
        first = assign_evaluation_components(components, settings, seed=1729)
        second = assign_evaluation_components(components, settings, seed=1729)
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("validation"), 1)
        self.assertEqual(list(first.values()).count("test"), 2)
        self.assertEqual(len(first), len(set(first)))

    def test_supplemental_pools_follow_linked_component_splits(self) -> None:
        _, component_by_gene = build_identity_components(self.genes)
        assignments = {
            component_by_gene[self.ids[0]]: "validation",
            component_by_gene[self.ids[4]]: "test",
        }
        payload = {
            "historical_symbols": [
                ["ValOld", ["previous"], [self.ids[0]]],
                ["TestOld", ["alias"], [self.ids[4]]],
                ["TrainOld", ["previous"], [self.ids[5]]],
            ],
            "retired_genes": [
                ["ENSG00000000010", [self.ids[0]]],
                ["ENSG00000000011", [self.ids[4]]],
                ["ENSG00000000012", [self.ids[5]]],
            ],
        }
        settings = {
            "train": {
                "historical_symbol_rows": 1,
                "retired_gene_id_rows": 1,
            },
            "evaluation": {
                split: {
                    "historical_symbol_rows": 1,
                    "retired_gene_id_rows": 1,
                }
                for split in ("validation", "test")
            },
        }
        selected = select_supplemental_facts(
            payload,
            component_by_gene,
            assignments,
            settings,
            seed=1729,
        )
        self.assertEqual(
            {fact.kind for fact in selected["train"]},
            {HISTORICAL_SYMBOL, RETIRED_GENE_ID},
        )
        self.assertEqual(
            {fact.value for fact in selected["validation"]},
            {"ValOld", "ENSG00000000010"},
        )
        self.assertEqual(
            {fact.value for fact in selected["test"]},
            {"TestOld", "ENSG00000000011"},
        )

    def test_evidence_sequence_is_compact_unique_and_channel_typed(self) -> None:
        sequence = EvidenceSequence()
        values = [
            sequence("registry"),
            sequence("graph"),
            sequence("registry"),
        ]
        self.assertEqual(len(values), len(set(values)))
        self.assertRegex(values[0], re.compile(r"^ev_reg_[a-z0-9]{6}$"))
        self.assertRegex(values[1], re.compile(r"^ev_mem_[a-z0-9]{6}$"))

    def test_prompt_banks_are_paired_and_split_isolated(self) -> None:
        expected_counts = {"train": 8, "validation": 4, "test": 1}
        prompts_by_split: dict[str, set[str]] = {}
        for split, expected_count in expected_counts.items():
            prompts_by_split[split] = set()
            for input_kind in ("symbol", "gene_id"):
                direct = PROMPT_TEMPLATES[split][input_kind]["direct"]
                step = PROMPT_TEMPLATES[split][input_kind]["step"]
                self.assertEqual(len(direct), expected_count)
                self.assertEqual(len(step), expected_count)
                self.assertEqual(len(set(direct)), expected_count)
                self.assertEqual(len(set(step)), expected_count)
                prompts_by_split[split].update(direct)
                prompts_by_split[split].update(step)
        self.assertFalse(prompts_by_split["train"] & prompts_by_split["validation"])
        self.assertFalse(prompts_by_split["train"] & prompts_by_split["test"])
        self.assertFalse(prompts_by_split["validation"] & prompts_by_split["test"])
        self.assertIn(
            "The next analysis operation needs a human Ensembl gene ID. "
            "Resolve the current symbol {value}.",
            prompts_by_split["test"],
        )
        self.assertIn(
            "The next analysis operation needs a human gene symbol. "
            "Resolve the current Ensembl gene ID {value}.",
            prompts_by_split["test"],
        )

    def test_v4_overlay_versions_the_corpus_without_changing_the_wire(self) -> None:
        contract = load_contract()
        base_settings = contract["curriculum"]["families"]["S0"]
        settings, overlay = load_s0_v4_settings(contract)
        self.assertEqual(
            base_settings["dataset_id"], "world_model_v3_s0_human_identifiers"
        )
        self.assertEqual(base_settings["rows"]["train"], 4096)
        self.assertEqual(
            contract["rows"]["metadata"]["optional"],
            ["found_input_type", "input_species"],
        )
        self.assertEqual(
            settings["dataset_id"], "world_model_v4_s0_human_identifiers"
        )
        self.assertEqual(settings["output_directory"], "data/world_model_v4/corpora/s0")
        self.assertEqual(settings["max_length"], 4096)
        self.assertEqual(overlay["schema_version"], S0_V4_OVERLAY_SCHEMA_VERSION)
        self.assertEqual(
            overlay["metadata_optional_additions"],
            list(S0_V4_METADATA_ADDITIONS),
        )
        self.assertEqual(
            overlay["base_contract"]["sha256"],
            "f629c28de87a0440b40130d78614d4c5f65d55cd32e68b7bd144d09df81d9f0f",
        )
        self.assertTrue(S0_V4_OVERLAY_PATH.is_file())
        self.assertEqual(contract["runtime"]["row_schema_version"], "multiplex_sft_v3")
        self.assertEqual(CORPUS_SCHEMA_VERSION, "mentor-rl-world-model-s0-corpus-v2")
        self.assertEqual(MANIFEST_SCHEMA_VERSION, "mentor-rl-world-model-s0-manifest-v2")


class WorldModelS0RealIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract()
        cls.settings, cls.overlay = load_s0_v4_settings(cls.contract)
        digest = cls.contract["source_index"]["content_sha256"]
        cls.path = (
            Path("data/world_model_v3/source_indexes")
            / f"world_model_source_index_{digest}.json.gz"
        )
        if not cls.path.is_file():
            raise unittest.SkipTest(f"generated source index is absent: {cls.path}")
        cls.payload = check_source_index(cls.path, cls.contract)
        cls.seed = int(cls.settings["assignment_seed"])
        cls.components, cls.component_by_gene = build_identity_components(
            cls.payload["genes_by_id"]
        )
        cls.assignments = assign_evaluation_components(
            cls.components, cls.settings["evaluation"], seed=cls.seed
        )
        cls.registry = select_registry_facts(
            cls.components,
            cls.assignments,
            cls.settings,
            seed=cls.seed,
            maximum_ambiguous_candidates=int(
                cls.settings["maximum_train_ambiguous_candidates"]
            ),
        )
        cls.supplemental = select_supplemental_facts(
            cls.payload,
            cls.component_by_gene,
            cls.assignments,
            cls.settings,
            seed=cls.seed,
        )
        cls.train_facts = cls.registry["train"] + cls.supplemental["train"]
        cls.train_renderings = build_train_renderings(
            cls.train_facts, cls.settings, seed=cls.seed
        )

    def test_pinned_selection_reconciles_all_split_allocations(self) -> None:
        self.assertEqual(len(self.components), 77311)
        self.assertEqual(sum(item.is_ambiguous for item in self.components), 484)
        self.assertEqual(
            len(self.train_facts),
            sum(self.settings["train"]["unique_fact_counts"].values()),
        )
        self.assertEqual(len(self.train_renderings), self.settings["rows"]["train"])
        for split in ("validation", "test"):
            self.assertEqual(
                len(self.registry[split]) + len(self.supplemental[split]),
                self.settings["rows"][split],
            )

    def test_train_allocations_are_exact_paired_and_case_balanced(self) -> None:
        row_counts = Counter(item.fact.kind for item in self.train_renderings)
        self.assertEqual(
            row_counts,
            Counter(
                {
                    RESOLVED_SYMBOL: 5120,
                    NAMED_GENE_ID: 4608,
                    UNNAMED_GENE_ID: 5056,
                    AMBIGUOUS_SYMBOL: 832,
                    HISTORICAL_SYMBOL: 384,
                    RETIRED_GENE_ID: 384,
                }
            ),
        )
        unique_counts = Counter(fact.kind for fact in self.train_facts)
        self.assertEqual(unique_counts, Counter(self.settings["train"]["unique_fact_counts"]))

        frames_by_rendering: dict[tuple[str, int, str], set[str]] = defaultdict(set)
        templates_by_kind: dict[str, set[int]] = defaultdict(set)
        for item in self.train_renderings:
            key = (item.fact.fact_id, item.template_index, item.input_variant)
            frames_by_rendering[key].add(item.framing)
            templates_by_kind[item.fact.kind].add(item.template_index)
        self.assertTrue(frames_by_rendering)
        self.assertTrue(
            all(framings == {"direct", "step"} for framings in frames_by_rendering.values())
        )
        self.assertTrue(
            all(indices == set(range(8)) for indices in templates_by_kind.values())
        )

        casefold_counts = Counter(
            item.fact.kind
            for item in self.train_renderings
            if item.input_variant == "casefold"
        )
        self.assertEqual(
            casefold_counts,
            Counter({kind: count // 8 for kind, count in row_counts.items()}),
        )

    def test_all_long_ambiguities_cover_every_template_and_framing(self) -> None:
        long_facts = {
            fact.fact_id: (fact.value, fact.candidate_count)
            for fact in self.train_facts
            if fact.kind == AMBIGUOUS_SYMBOL and fact.candidate_count > 20
        }
        self.assertEqual(
            set(long_facts.values()),
            {("U2", 21), ("U8", 22), ("SNORA70", 27), ("U6", 37)},
        )
        coverage: dict[str, set[tuple[int, str]]] = defaultdict(set)
        for item in self.train_renderings:
            if item.fact.fact_id in long_facts:
                coverage[item.fact.fact_id].add(
                    (item.template_index, item.framing)
                )
        expected = {
            (template_index, framing)
            for template_index in range(8)
            for framing in ("direct", "step")
        }
        self.assertEqual(set(coverage), set(long_facts))
        self.assertTrue(all(values == expected for values in coverage.values()))

    def test_rows_expose_rendering_metadata_and_decision_aware_thinking(self) -> None:
        fact = next(
            item for item in self.train_facts if item.kind == UNNAMED_GENE_ID
        )
        runtime = WorldModelToolRuntime.from_source_index(
            self.path,
            contract=self.contract,
            evidence_id_factory=EvidenceSequence(),
        )
        common = {
            "split": "train",
            "seed": self.seed,
            "runtime": runtime,
            "contract": self.contract,
            "tools": tool_definitions(self.contract),
            "template_index": 6,
            "input_variant": "canonical",
        }
        direct, _ = _build_row(
            fact, framing="direct", rendering_index=6, **common
        )
        step, _ = _build_row(fact, framing="step", rendering_index=6, **common)
        rerendered, _ = _build_row(
            fact, framing="direct", rendering_index=106, **common
        )

        metadata = direct["metadata"]
        self.assertEqual(metadata["fact_kind"], UNNAMED_GENE_ID)
        self.assertEqual(metadata["framing"], "direct")
        self.assertEqual(metadata["prompt_template_id"], "s0.train.gene_id.t06")
        self.assertEqual(metadata["rendering_index"], 6)
        self.assertEqual(metadata["input_variant"], "canonical")
        self.assertEqual(metadata["split_unit"], step["metadata"]["split_unit"])
        self.assertEqual(metadata["split_unit"], rerendered["metadata"]["split_unit"])
        self.assertEqual(len({direct["row_id"], step["row_id"], rerendered["row_id"]}), 3)

        call_thinking = direct["messages"][2]["thinking"]
        self.assertIn(fact.tool_name, call_thinking)
        self.assertIn("gene_id=", call_thinking)
        direct_final = direct["messages"][-1]
        step_final = step["messages"][-1]
        self.assertIsNone(direct_final["content"]["result"]["gene_symbol"])
        self.assertEqual(direct_final["content"]["continuation_decision"], "stop")
        self.assertEqual(step_final["content"]["continuation_decision"], "continue")
        self.assertIn("preserve that null", direct_final["thinking"])
        self.assertIn("should stop", direct_final["thinking"])
        self.assertIn("should continue", step_final["thinking"])

    def test_evaluation_prompts_use_only_their_split_template_banks(self) -> None:
        for split in ("validation", "test"):
            facts = self.registry[split] + self.supplemental[split]
            renderings = build_evaluation_renderings(
                facts, split, seed=self.seed
            )
            expected = 4 if split == "validation" else 1
            self.assertEqual(len(renderings), self.settings["rows"][split])
            self.assertTrue(
                all(0 <= item.template_index < expected for item in renderings)
            )


if __name__ == "__main__":
    unittest.main()
