from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from types import SimpleNamespace
import unittest

from scripts.pretrajectory_curriculum_global_families import (
    GLOBAL_FAMILY_NAMES,
    generate_global_families,
)


class FakeExample:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass(frozen=True)
class FakeEdge:
    source_gene_id: str
    target_gene_id: str
    weight: float = 1.0


class FakeDistanceShard:
    def __init__(self, shard_id: str, genes: list[str]) -> None:
        self.shard_id = shard_id
        self.genes = tuple(genes)
        self._index = {gene: index for index, gene in enumerate(self.genes)}

    def distance(self, gene_a: str, gene_b: str) -> float:
        if gene_a == gene_b:
            return 0.0
        return 0.05 + abs(self._index[gene_a] - self._index[gene_b]) / 100.0

    def row(self, gene_id: str) -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(gene_id=other, distance=self.distance(gene_id, other))
            for other in self.genes
            if other != gene_id
        )


class FakeRwr:
    def __init__(self) -> None:
        self._shards = {
            "shard_00000": FakeDistanceShard(
                "shard_00000", [f"ENSG_G{index:03d}" for index in range(40)]
            ),
            "shard_00001": FakeDistanceShard(
                "shard_00001", [f"ENSG_H{index:03d}" for index in range(40)]
            ),
        }
        self.shard_ids = tuple(self._shards)
        self.identity = SimpleNamespace(
            ranking_semantics="rwr_encoding_desc_min_rank_seed_excluded"
        )

    def route_seed(self, gene_id: str) -> SimpleNamespace:
        if gene_id.startswith("ENSG_G"):
            return SimpleNamespace(shard_id="shard_00000")
        if gene_id.startswith("ENSG_H"):
            return SimpleNamespace(shard_id="shard_00001")
        raise KeyError(gene_id)

    def distance_shard(self, shard_id: str) -> FakeDistanceShard:
        return self._shards[shard_id]

    def seed_metadata(self, gene_id: str) -> SimpleNamespace:
        self.route_seed(gene_id)
        return SimpleNamespace(ranked_gene_count=79)

    def public_provenance(self) -> dict[str, str]:
        return {
            "rank_cache_id": "context_fixture",
            "network_flist_sha256": "a" * 64,
        }


class FakeOracle:
    def __init__(self, rwr_modules: list[dict]) -> None:
        self.layer_names = (
            "scPEN:brain:region_a:astrocyte",
            "scPEN:brain:region_a:neuron",
            "scPEN:brain:region_b:astrocyte",
            "scPEN:brain:region_b:neuron",
        )
        self.layer_count = len(self.layer_names)
        self._edges: dict[str, set[tuple[str, str]]] = {
            layer: set() for layer in self.layer_names
        }
        for index, module in enumerate(rwr_modules):
            seed = module["seed_gene_id"]
            genes = module["gene_ids"]
            self._edges[self.layer_names[0]].add(tuple(sorted((seed, genes[1]))))
            self._edges[self.layer_names[1]].add(tuple(sorted((seed, genes[2]))))
            if index % 2 == 0:
                self._edges[self.layer_names[2]].add(tuple(sorted((genes[1], genes[2]))))

    def sample_edges(self, *, layer: str, count: int, seed: int) -> list[FakeEdge]:
        del seed
        return [FakeEdge(left, right) for left, right in sorted(self._edges[layer])[:count]]

    def induced_edges(self, gene_ids, *, layer: str | None = None) -> list[FakeEdge]:
        genes = set(gene_ids)
        if layer is None:
            pairs = set().union(*self._edges.values())
        else:
            pairs = self._edges[layer]
        return [FakeEdge(left, right) for left, right in sorted(pairs) if left in genes and right in genes]

    def degree(self, gene_id: str, *, layer: str | None = None) -> int:
        if layer is None:
            return 10 + (sum(ord(char) for char in gene_id) % 5)
        internal_degree = sum(gene_id in pair for pair in self._edges[layer])
        return internal_degree + 1

    def gene_layers(self, gene_id: str) -> list[str]:
        present = [
            layer
            for layer in self.layer_names
            if any(gene_id in pair for pair in self._edges[layer])
        ]
        return present or [self.layer_names[3]]


class FakeBuilder:
    curriculum_example_type = FakeExample

    def __init__(self) -> None:
        self.seed = 17
        self.multiplex_id = "fixture_multiplex_v1"
        rwr_modules = []
        g_genes = [f"ENSG_G{index:03d}" for index in range(40)]
        for index in range(30):
            module_genes = [g_genes[(index + offset) % len(g_genes)] for offset in range(8)]
            rwr_modules.append(
                {
                    "module_id": f"rwr_module_{index:03d}",
                    "source": "RWR_LOE_FULL_BRAIN",
                    "seed_gene_id": module_genes[0],
                    "gene_ids": module_genes,
                }
            )
        mentor_modules = [
            {
                "module_id": "mentor_module_000",
                "source": "MENTOR_GW_DENDROGRAM",
                "gene_ids": [f"ENSG_H{index:03d}" for index in range(8)],
            }
        ]
        self.modules_by_source = {
            "RWR_LOE_FULL_BRAIN": rwr_modules,
            "MENTOR_GW_DENDROGRAM": mentor_modules,
        }
        self.modules = [*rwr_modules, *mentor_modules]
        self.rwr = FakeRwr()
        self.oracle = FakeOracle(rwr_modules)
        self.examples: list[FakeExample] = []

    def family(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(name=name)

    def candidate_goal(self, family_name: str) -> int:
        assert family_name in GLOBAL_FAMILY_NAMES
        return 2

    def add(self, example: FakeExample) -> None:
        self.examples.append(example)

    def _base_provenance(self, source: str) -> dict[str, str]:
        return {
            "source": source,
            "store_id": "sha256:" + "b" * 64,
            "flist_id": "sha256:" + "a" * 64,
            "multiplex_id": self.multiplex_id,
        }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


class GlobalCurriculumFamiliesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = FakeBuilder()
        self.result = generate_global_families(self.builder)
        self.by_family: dict[str, list[FakeExample]] = defaultdict(list)
        for example in self.builder.examples:
            self.by_family[example.family.name].append(example)

    def test_emits_every_requested_family_to_goal_with_bounded_evidence(self) -> None:
        self.assertEqual(self.result, {name: 2 for name in GLOBAL_FAMILY_NAMES})
        self.assertEqual(Counter(example.family.name for example in self.builder.examples), Counter(self.result))
        self.assertEqual(len(self.builder.examples), 18)
        for example in self.builder.examples:
            self.assertEqual(example.book_mode, "open_book")
            self.assertIsNotNone(example.evidence)
            self.assertEqual(example.context_budget_profile, "evidence_2k")
            self.assertIn("scope", example.page)
            self.assertGreaterEqual(example.page["count"], 1)
            rendered = json.dumps(
                {
                    "task": example.task,
                    "answer": example.answer,
                    "evidence": example.evidence,
                    "provenance": example.provenance,
                },
                sort_keys=True,
            ).lower()
            self.assertNotIn("/lustre/", rendered)
            self.assertNotIn("/private/", rendered)
            self.assertNotIn("jaccard", rendered)

        for example in self.by_family["within_clade_distance"]:
            self.assertLessEqual(len(example.evidence["pair_distances"]), 10)
        for example in self.by_family["within_clade_vs_random"]:
            self.assertLessEqual(len(example.evidence["null_set_summaries"]), 8)
        for example in self.by_family["nearest_modules"]:
            self.assertLessEqual(len(example.evidence["candidate_table"]), 5)
        for example in self.by_family["whole_multiplex_context_profile"]:
            page = example.evidence["full_store_summary"]["layer_ids_page"]
            self.assertLessEqual(len(page), 16)

    def test_numeric_answers_recompute_from_supplied_evidence(self) -> None:
        within = self.by_family["within_clade_distance"][0]
        observed = _mean(row["distance"] for row in within.evidence["pair_distances"])
        self.assertAlmostEqual(within.answer["mean_within_distance"], observed)

        null = self.by_family["within_clade_vs_random"][0]
        observed = null.evidence["observed_mean_within_distance"]
        null_means = [row["mean_distance"] for row in null.evidence["null_set_summaries"]]
        expected_p = (1 + sum(value <= observed for value in null_means)) / (1 + len(null_means))
        self.assertAlmostEqual(null.answer["empirical_p_value"], expected_p)

        ratio = self.by_family["clustering_ratio"][0]
        within_mean = _mean(row["distance"] for row in ratio.evidence["pair_distances"])
        outside_mean = _mean(row["distance"] for row in ratio.evidence["cross_boundary_distances"])
        self.assertAlmostEqual(ratio.answer["mean_within_distance"], within_mean)
        self.assertAlmostEqual(ratio.answer["mean_outside_distance"], outside_mean)
        self.assertAlmostEqual(ratio.answer["clustering_ratio"], outside_mean / within_mean)

        density = self.by_family["subgraph_density"][0]
        summary = density.evidence["layer_summary"]
        self.assertAlmostEqual(
            density.answer["edge_density"],
            summary["internal_edge_count"] / summary["possible_edge_count"],
        )

        boundary = self.by_family["conductance_boundary_ratio"][0]
        degree_sum = sum(row["layer_degree"] for row in boundary.evidence["per_gene_layer_degrees"])
        internal = len(boundary.evidence["internal_edges"])
        boundary_count = degree_sum - 2 * internal
        self.assertEqual(boundary.answer["boundary_edge_count"], boundary_count)
        self.assertAlmostEqual(
            boundary.answer["boundary_ratio"],
            boundary_count / (internal + boundary_count),
        )

    def test_context_comparison_ranking_and_profile_are_scope_correct(self) -> None:
        cell = self.by_family["cell_type_specific_cohesion"][0]
        summaries = cell.evidence["layer_summaries"]
        self.assertGreater(summaries[0]["edge_density"], summaries[1]["edge_density"])
        self.assertIn("bounded", cell.answer["aggregation_scope"])

        ablation = self.by_family["layer_sensitive_cohesion"][0]
        layers = ablation.evidence["layer_subset"]
        higher = {
            tuple(sorted((row["gene_a"], row["gene_b"])))
            for row in layers[0]["internal_edges"]
        }
        lower = {
            tuple(sorted((row["gene_a"], row["gene_b"])))
            for row in layers[1]["internal_edges"]
        }
        self.assertEqual(ablation.answer["baseline_unique_internal_edge_count"], len(higher | lower))
        self.assertEqual(ablation.answer["post_ablation_unique_internal_edge_count"], len(lower))

        nearest = self.by_family["nearest_modules"][0]
        expected = sorted(
            nearest.evidence["candidate_table"],
            key=lambda row: (row["distance"], row["module_id"]),
        )
        self.assertEqual(
            nearest.answer["nearest_modules"],
            [{"module_id": row["module_id"], "distance": row["distance"]} for row in expected],
        )
        self.assertEqual(nearest.answer["retrieval_scope"], "supplied_same_shard_candidate_table")

        profile = self.by_family["whole_multiplex_context_profile"][0]
        store = profile.evidence["full_store_summary"]
        self.assertAlmostEqual(
            profile.answer["layer_coverage_fraction"],
            store["layer_coverage_count"] / store["multiplex_layer_count"],
        )
        self.assertIn("same-shard", profile.answer["scope_note"])
        self.assertNotIn("hub", profile.answer["global_profile"])


if __name__ == "__main__":
    unittest.main()
