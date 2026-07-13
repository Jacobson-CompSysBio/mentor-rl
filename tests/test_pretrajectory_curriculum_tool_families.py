import json
import unittest
from collections import Counter
from types import SimpleNamespace

from runtime.schemas import ToolAction
from runtime.tool_curriculum_contract import (
    CURRICULUM_TOOL_NAMES,
    assert_no_provenance_leakage,
    validate_tool_observation,
)
from runtime.validators import validate_tool_action
from scripts.pretrajectory_curriculum_tool_families import (
    TOOL_FAMILY_NAMES,
    generate_tool_families,
)


GENES = [f"ENSG{i:06d}" for i in range(1, 31)]
LAYERS = ["HumanNetV3:coexpression", "TFs:target"]


class _Example:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Edge:
    def __init__(self, source, target, weight):
        self.source_gene_id = source
        self.target_gene_id = target
        self.weight = weight

    def as_dict(self):
        return {
            "source_gene_id": self.source_gene_id,
            "target_gene_id": self.target_gene_id,
            "weight": self.weight,
        }


class _Oracle:
    gene_ids = tuple(GENES)
    layer_names = tuple(LAYERS)
    layer_count = len(LAYERS)

    def sample_node_indices(self, *, layer, count, seed, minimum_degree):
        return list(range(count))

    def gene_layers(self, gene):
        index = self.gene_ids.index(gene)
        return list(self.layer_names if index % 2 == 0 else self.layer_names[:1])

    def sample_edges(self, *, layer, count, seed):
        offset = 0 if layer is None else self.layer_names.index(layer) * 7
        return [
            _Edge(
                self.gene_ids[(offset + index * 3) % len(self.gene_ids)],
                self.gene_ids[(offset + index * 3 + 1) % len(self.gene_ids)],
                0.5 + index / 1000.0,
            )
            for index in range(count)
        ]

    def neighbors(self, gene, *, layer, limit):
        index = self.gene_ids.index(gene)
        return [
            {"gene_id": self.gene_ids[(index + offset) % len(self.gene_ids)], "weight": 0.4}
            for offset in range(1, min(limit, 5) + 1)
        ]

    def induced_components(self, genes, *, layer):
        return [sorted(set(genes))]

    def induced_edges(self, genes, *, layer):
        values = sorted(set(genes))
        return [
            _Edge(values[index], values[index + 1], 0.7 + index / 100.0)
            for index in range(len(values) - 1)
        ]


class _RankRow:
    def __init__(self, gene_id, rank, score):
        self.gene_id = gene_id
        self.rank = rank
        self.score = score


class _RankVector:
    def __init__(self, seed, genes):
        ranked = [gene for gene in genes if gene != seed]
        self.rows = tuple(
            _RankRow(gene, index + 1, 1.0 / (index + 2))
            for index, gene in enumerate(ranked)
        )
        self.metadata = SimpleNamespace(ranked_gene_count=len(self.rows))

    def top_k(self, count, *, exclude_genes=()):
        excluded = set(exclude_genes)
        return tuple(row for row in self.rows if row.gene_id not in excluded)[:count]


class _DistanceShard:
    shard_id = "shard_00000"
    genes = tuple(GENES[:12])
    provenance = {"distance_metric": "spearman_distance"}

    def distance(self, gene_a, gene_b):
        return abs(self.genes.index(gene_a) - self.genes.index(gene_b)) / 10.0


class _Rwr:
    seed_gene_ids = tuple(GENES[:12])
    shard_ids = ("shard_00000",)
    identity = SimpleNamespace(network_flist_sha256="a" * 64)

    def __init__(self):
        self._rank_vectors = {}
        self._distance_shards = {}

    def rank_vector(self, seed):
        self._rank_vectors.setdefault(seed, _RankVector(seed, GENES))
        return self._rank_vectors[seed]

    def distance_shard(self, shard_id):
        self._distance_shards.setdefault(shard_id, _DistanceShard())
        return self._distance_shards[shard_id]


class _Builder:
    seed = 17
    oracle = _Oracle()
    rwr = _Rwr()
    multiplex_id = "full_brain_multiplex_v1"
    store_id = "sha256:" + "b" * 64
    flist_id = "sha256:" + "a" * 64
    curriculum_example_type = _Example

    def __init__(self):
        self.examples = []
        self.modules_by_source = {
            "MENTOR_GW_DENDROGRAM": [
                {"module_id": "mentor:M1", "gene_ids": GENES[0:3]},
                {"module_id": "mentor:M2", "gene_ids": GENES[3:6]},
                {"module_id": "mentor:M3", "gene_ids": GENES[6:9]},
            ],
            "RWR_LOE_FULL_BRAIN": [
                {"module_id": "rwr:R1", "gene_ids": GENES[0:5]},
                {"module_id": "rwr:R2", "gene_ids": GENES[3:7]},
                {"module_id": "rwr:R3", "gene_ids": GENES[9:14]},
                {"module_id": "rwr:R4", "gene_ids": GENES[0:2]},
            ],
        }

    def family(self, name):
        return SimpleNamespace(name=name, id=TOOL_FAMILY_NAMES.index(name) + 69)

    def candidate_goal(self, family_name):
        return 2

    def add(self, example):
        self.examples.append(example)


def _walk_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


class ToolCurriculumFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = _Builder()
        generate_tool_families(cls.builder)

    def test_generates_every_required_family_at_goal_in_tool_call_mode(self):
        counts = Counter(example.family.name for example in self.builder.examples)
        self.assertEqual(counts, {family: 2 for family in TOOL_FAMILY_NAMES})
        self.assertTrue(all(example.book_mode == "tool_call" for example in self.builder.examples))
        self.assertTrue(all(example.tool_exchange is not None for example in self.builder.examples))

    def test_every_action_and_observation_uses_the_live_schema(self):
        genes = set(self.builder.oracle.gene_ids)
        layers = set(self.builder.oracle.layer_names)
        for example in self.builder.examples:
            with self.subTest(family=example.family.name):
                exchange = example.tool_exchange
                action = ToolAction.from_dict(exchange["tool_action"])
                self.assertIn(action.tool_name, CURRICULUM_TOOL_NAMES)
                validation = validate_tool_action(
                    action,
                    available_gene_ids=genes,
                    available_layers=layers,
                )
                self.assertTrue(validation.valid, validation.errors)
                validate_tool_observation(exchange["tool_observation"], action=action)
                assert_no_provenance_leakage(exchange)
                self.assertEqual(
                    exchange["tool_observation"]["payload"]["tool_name"],
                    action.tool_name,
                )

    def test_no_local_paths_or_fabricated_tool_names_are_visible(self):
        forbidden_names = {
            "query_module_oracle",
            "choose_next_tool",
            "get_graph_schema",
            "resolve_gene_alias",
        }
        forbidden_fragments = ("/lustre/", "/autofs/", "/tmp/", "file://", "data/")
        for example in self.builder.examples:
            visible = {
                "task": example.task,
                "answer": example.answer,
                "evidence": example.evidence,
                "exchange": example.tool_exchange,
            }
            text = json.dumps(visible, sort_keys=True)
            self.assertFalse(any(fragment in text for fragment in forbidden_fragments))
            self.assertTrue(forbidden_names.isdisjoint(_walk_strings(visible)))

    def test_selection_and_parse_answers_are_grounded_in_visible_observations(self):
        for example in self.builder.examples:
            family = example.family.name
            exchange = example.tool_exchange
            payload = exchange["tool_observation"]["payload"]
            answer = example.answer
            with self.subTest(family=family):
                if family == "choose_rwr_loe_tool":
                    self.assertEqual(answer["tool_action"], exchange["tool_action"])
                    self.assertEqual(answer["comparison_gene_ids"], example.evidence["comparison_gene_ids"])
                elif family == "choose_pairwise_distance_tool":
                    self.assertEqual(answer["observed_distance"], payload["distance"])
                elif family == "choose_layer_membership_tool":
                    self.assertEqual(answer["tool_action"], exchange["tool_action"])
                    self.assertEqual(payload["result_status"], "schema_validated_request_not_materialized")
                    reference = example.evidence["layer_membership_reference"]
                    self.assertEqual(reference["layer_count"], len(self.builder.oracle.gene_layers(reference["gene"])))
                elif family == "choose_component_summary_tool":
                    observation = example.evidence["component_connectivity_observation"]
                    self.assertEqual(answer["component_count"], observation["component_count"])
                    self.assertEqual(payload["result_status"], "schema_validated_request_not_materialized")
                elif family == "choose_induced_subgraph_tool":
                    self.assertEqual(answer["edges"], payload["layers"][0]["edges"])
                elif family == "choose_layer_ablation_tool":
                    self.assertEqual(answer["result_status"], payload["result_status"])
                    self.assertEqual(payload["layer_effects"], [])
                elif family == "parse_rwr_loe_result":
                    self.assertEqual(answer["closest_non_seed_genes"], payload["ranked_genes"][:3])
                elif family == "parse_distance_shard":
                    observation = example.evidence["bounded_distance_observation"]
                    self.assertEqual(answer["distances"], observation["cells"])
                elif family == "parse_module_overlap":
                    rows = example.evidence["module_overlap_observation"]["candidate_rows"]
                    expected_ids = [row["module_id"] for row in rows if row["contains_query_module"]]
                    self.assertEqual(
                        [row["module_id"] for row in answer["superset_modules"]],
                        expected_ids,
                    )
                elif family == "provenance_answer":
                    self.assertEqual(answer["evidence_id"], payload["evidence_id"])
                    self.assertEqual(answer["multiplex_id"], payload["multiplex_id"])
                elif family == "refuse_raw_cli_path":
                    self.assertEqual(answer, example.evidence["request_contract_observation"])
                    ToolAction.from_dict(answer["corrected_tool_action"])
                elif family == "structured_state_update":
                    ranked = payload["ranked_genes"][:2]
                    expected = payload["seed_genes"] + [row["gene_id"] for row in ranked]
                    self.assertEqual(answer["predicted_groups"][0]["gene_ids"], expected)
                    self.assertEqual(answer["mechanistic_labels"] if "mechanistic_labels" in answer else [], [])


if __name__ == "__main__":
    unittest.main()
