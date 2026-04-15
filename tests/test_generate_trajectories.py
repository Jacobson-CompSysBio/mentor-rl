import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from runtime.environment import RuntimeEnvironment
from scripts.generate_trajectories import (
    ModelGeneratorConfig,
    TrajectoryGenerationConfig,
    generate_trajectories,
)
from utils.multiplex import Multiplex


def _build_environment() -> RuntimeEnvironment:
    multiplex = Multiplex()

    ppi = nx.Graph()
    ppi.add_nodes_from(["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"])
    ppi.add_edge("ENSG1", "ENSG2", weight=1.0)
    ppi.add_edge("ENSG2", "ENSG3", weight=0.9)
    multiplex.add_layer(ppi, "ppi")

    tf = nx.Graph()
    tf.add_nodes_from(["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"])
    tf.add_edge("ENSG1", "ENSG3", weight=0.7)
    multiplex.add_layer(tf, "tf")

    return RuntimeEnvironment(multiplex=multiplex)


def _task_rows() -> list[dict]:
    return [
        {
            "task_id": "corum_complex_test.recovery.easy.contextual",
            "task_type": "recovery",
            "difficulty": "easy",
            "query_text": "Recover the shared group around ENSG1 and ENSG2.",
            "evidence_mode": "contextual",
            "visible_inputs": {
                "seed_gene_ids": ["ENSG1", "ENSG2"],
                "seed_gene_symbols": ["GENE1", "GENE2"],
                "context_text": "The seed genes appear in the same candidate module.",
                "graph_query_spec": None,
                "structured_annotations": None,
            },
            "hidden_target": {
                "relationship_status": "validated_group",
                "target_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                "target_gene_symbols": ["GENE1", "GENE2", "GENE3"],
            },
            "mechanism_labels": {
                "go_ids": ["GO:0000001"],
                "go_names": ["toy process"],
                "fcgs_ids": [],
                "fcgs_names": [],
                "primary_label": "toy process",
            },
        },
        {
            "task_id": "none_test.contextual",
            "task_type": "none",
            "difficulty": "complete",
            "query_text": "Do ENSG4 and ENSG5 support one shared mechanism?",
            "evidence_mode": "contextual",
            "visible_inputs": {
                "seed_gene_ids": ["ENSG4", "ENSG5"],
                "seed_gene_symbols": ["GENE4", "GENE5"],
                "context_text": "No curated shared-context note was attached to this pair.",
                "graph_query_spec": None,
                "structured_annotations": None,
            },
            "hidden_target": {
                "relationship_status": "insufficient_support",
                "target_gene_ids": None,
                "target_gene_symbols": None,
            },
            "mechanism_labels": None,
        },
    ]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _FakeModelGenerator:
    model_name = "gpt-oss-120b-bf16"

    def generate_actor_candidates(self, context, *, task_row, step_index, n_act, seed):
        del context, n_act, seed
        if task_row["task_type"] == "recovery":
            return [
                {
                    "reasoning_text": "Expand the current group with a restart walk.",
                    "tool_action": {"tool_name": "rwr_multiplex", "arguments": {"seeds": ["ENSG1", "ENSG2"], "top_k": 5}},
                    "raw_text": '{"reasoning_text":"Expand","tool_action":{"tool_name":"rwr_multiplex","arguments":{"seeds":["ENSG1","ENSG2"],"top_k":5}}}',
                    "generator_errors": [],
                }
            ]
        return [
            {
                "reasoning_text": "Probe the induced subgraph before abstaining.",
                "tool_action": {"tool_name": "induce_subgraph", "arguments": {"genes": ["ENSG4", "ENSG5"]}},
                "raw_text": '{"reasoning_text":"Probe","tool_action":{"tool_name":"induce_subgraph","arguments":{"genes":["ENSG4","ENSG5"]}}}',
                "generator_errors": [],
            }
        ]

    def generate_verifier_candidates(self, context, *, task_row, actor_candidate, actor_step, observation, step_index, n_ver, seed):
        del context, actor_candidate, actor_step, observation, step_index, n_ver, seed
        if task_row["task_type"] == "recovery":
            return [
                {
                    "payload": {
                        "updated_interpretation": {
                            "mechanistic_claim": "The evidence supports a shared module.",
                            "main_evidence": "The restart walk ranks ENSG3 with the seeds.",
                            "uncertainty": "",
                            "next_subgoal": "",
                        },
                        "updated_state": {
                            "relationship_status": "validated_group",
                            "predicted_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                            "mechanistic_labels": [
                                {"label_source": "go", "label_name": "toy process", "label_id": "GO:0000001"}
                            ],
                            "continuation_decision": "stop",
                            "verifier_notes": "Model-backed verifier accepted the expanded group.",
                        },
                    },
                    "raw_text": '{"updated_interpretation":{"mechanistic_claim":"The evidence supports a shared module.","main_evidence":"The restart walk ranks ENSG3 with the seeds.","uncertainty":"","next_subgoal":""},"updated_state":{"relationship_status":"validated_group","predicted_gene_ids":["ENSG1","ENSG2","ENSG3"],"mechanistic_labels":[{"label_source":"go","label_name":"toy process","label_id":"GO:0000001"}],"continuation_decision":"stop","verifier_notes":"Model-backed verifier accepted the expanded group."}}',
                    "generator_errors": [],
                }
            ]
        return [
            {
                "payload": {
                    "updated_interpretation": {
                        "mechanistic_claim": "The current evidence does not support one shared mechanism.",
                        "main_evidence": "No supporting subgraph edges were found.",
                        "uncertainty": "",
                        "next_subgoal": "",
                    },
                    "updated_state": {
                        "relationship_status": "insufficient_support",
                        "predicted_gene_ids": [],
                        "mechanistic_labels": [],
                        "continuation_decision": "stop",
                        "verifier_notes": "Abstain on the none task.",
                    },
                },
                "raw_text": '{"updated_interpretation":{"mechanistic_claim":"The current evidence does not support one shared mechanism.","main_evidence":"No supporting subgraph edges were found.","uncertainty":"","next_subgoal":""},"updated_state":{"relationship_status":"insufficient_support","predicted_gene_ids":[],"mechanistic_labels":[],"continuation_decision":"stop","verifier_notes":"Abstain on the none task."}}',
                "generator_errors": [],
            }
        ]


class GenerateTrajectoriesTests(unittest.TestCase):
    def test_generate_trajectories_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "trajectories"
            manifest = generate_trajectories(
                task_rows=_task_rows(),
                out_dir=out_dir,
                environment=_build_environment(),
                config=TrajectoryGenerationConfig(
                    max_steps=3,
                    n_act=4,
                    n_ver=2,
                    seed=7,
                ),
            )

            self.assertEqual(manifest["num_trajectories"], 2)
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "progress.json").exists())
            self.assertTrue((out_dir / "branch_pools.jsonl").exists())
            self.assertTrue((out_dir / "trajectory_turns.jsonl").exists())
            self.assertTrue((out_dir / "final_summaries.jsonl").exists())

            progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(progress["overall_progress"], 1.0)

            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            trajectory_turns = _read_jsonl(out_dir / "trajectory_turns.jsonl")
            final_summaries = _read_jsonl(out_dir / "final_summaries.jsonl")

            self.assertGreaterEqual(len(branch_pools), 2)
            self.assertGreaterEqual(len(trajectory_turns), 2)
            self.assertEqual(len(final_summaries), 2)

            first_pool = branch_pools[0]
            self.assertIn("context", first_pool)
            self.assertIn("branches", first_pool)
            self.assertIn("selected_branch_id", first_pool)
            self.assertTrue(
                any(branch["branch_id"] == first_pool["selected_branch_id"] for branch in first_pool["branches"])
            )
            self.assertTrue(
                all(branch["local_score"]["normalized_score"] is not None for branch in first_pool["branches"])
            )

    def test_generate_trajectories_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_a = tmp_path / "run_a"
            run_b = tmp_path / "run_b"
            config = TrajectoryGenerationConfig(max_steps=3, n_act=4, n_ver=2, seed=11)

            generate_trajectories(
                task_rows=_task_rows(),
                out_dir=run_a,
                environment=_build_environment(),
                config=config,
            )
            generate_trajectories(
                task_rows=_task_rows(),
                out_dir=run_b,
                environment=_build_environment(),
                config=config,
            )

            self.assertEqual(
                _read_jsonl(run_a / "branch_pools.jsonl"),
                _read_jsonl(run_b / "branch_pools.jsonl"),
            )
            self.assertEqual(
                _read_jsonl(run_a / "trajectory_turns.jsonl"),
                _read_jsonl(run_b / "trajectory_turns.jsonl"),
            )
            self.assertEqual(
                _read_jsonl(run_a / "final_summaries.jsonl"),
                _read_jsonl(run_b / "final_summaries.jsonl"),
            )

    def test_generate_trajectories_supports_model_backed_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "model_run"
            manifest = generate_trajectories(
                task_rows=_task_rows(),
                out_dir=out_dir,
                environment=_build_environment(),
                config=TrajectoryGenerationConfig(
                    max_steps=3,
                    n_act=1,
                    n_ver=1,
                    seed=5,
                    candidate_source="model_vllm",
                ),
                model_generator_config=ModelGeneratorConfig(api_base="http://unused"),
                candidate_generator=_FakeModelGenerator(),
            )

            self.assertEqual(manifest["generator"]["candidate_source"], "model_vllm")
            self.assertEqual(manifest["generator"]["model_name"], "gpt-oss-120b-bf16")
            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            first_branch = branch_pools[0]["branches"][0]
            self.assertEqual(first_branch["metadata"]["generator_backend"], "model_vllm")
            self.assertEqual(first_branch["verifier_step"]["updated_state"]["relationship_status"], "validated_group")
            self.assertEqual(first_branch["local_score"]["schema_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
