import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from runtime import ActorStep, SharedPrefixContext, ToolAction, initialize_state_from_corum_task
from runtime.environment import RuntimeEnvironment
from scripts.generate_trajectories import (
    ModelGeneratorConfig,
    OpenAICompatibleCandidateGenerator,
    TrajectoryGenerationConfig,
    _normalize_runtime_tool_action,
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


class _UnusableModelGenerator:
    model_name = "gpt-oss-120b-bf16"

    def generate_actor_candidates(self, context, *, task_row, step_index, n_act, seed):
        del context, task_row, step_index, n_act, seed
        return [
            {
                "reasoning_text": "",
                "tool_action": None,
                "raw_text": '{"finish_reason":"length","message":{}}',
                "generator_errors": [
                    "actor_response_truncated_before_visible_output",
                    "actor_tool_action_missing",
                    "actor_reasoning_and_tool_action_blank",
                ],
            }
        ]

    def generate_verifier_candidates(self, context, *, task_row, actor_candidate, actor_step, observation, step_index, n_ver, seed):
        del context, task_row, actor_candidate, actor_step, observation, step_index, n_ver, seed
        raise AssertionError("Verifier generation should not run when the actor candidate is unusable.")


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _RecordingSession:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses[:]
        self.requests: list[dict] = []

    def get(self, *args, **kwargs):
        raise AssertionError("Model discovery should not run when the model name is provided explicitly.")

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("No fake response was configured for this request.")
        return _FakeHTTPResponse(self.responses.pop(0))


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        rendered = json.dumps(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            },
            sort_keys=True,
        )
        if not add_generation_prompt:
            return rendered
        if kwargs.get("enable_thinking") is False:
            return rendered + "<|start|>assistant<|channel|>final<|message|>"
        return rendered + "<|start|>assistant"


class _BrokenFinalChannelTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        rendered = json.dumps(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            },
            sort_keys=True,
        )
        if not add_generation_prompt:
            return rendered
        return rendered + "<|start|>assistant"


class GenerateTrajectoriesTests(unittest.TestCase):
    def test_model_tool_action_normalization_removes_all_layer_alias(self) -> None:
        tool_action, errors = _normalize_runtime_tool_action(
            {
                "tool_name": "induce_subgraph",
                "arguments": {"genes": ["ENSG1", "ENSG2"], "layers": ["all"]},
            }
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            tool_action,
            {"tool_name": "induce_subgraph", "arguments": {"genes": ["ENSG1", "ENSG2"]}},
        )

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
            self.assertTrue((out_dir / "finding_records.jsonl").exists())
            self.assertTrue((out_dir / "preference_pairs_raw.jsonl").exists())
            self.assertTrue((out_dir / "preference_pairs.jsonl").exists())
            self.assertTrue((out_dir / "final_summaries.jsonl").exists())

            progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(progress["overall_progress"], 1.0)

            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            trajectory_turns = _read_jsonl(out_dir / "trajectory_turns.jsonl")
            finding_records = _read_jsonl(out_dir / "finding_records.jsonl")
            preference_pairs_raw = _read_jsonl(out_dir / "preference_pairs_raw.jsonl")
            preference_pairs = _read_jsonl(out_dir / "preference_pairs.jsonl")
            final_summaries = _read_jsonl(out_dir / "final_summaries.jsonl")

            self.assertGreaterEqual(len(branch_pools), 2)
            self.assertGreaterEqual(len(trajectory_turns), 2)
            self.assertEqual(len(finding_records), len(trajectory_turns))
            self.assertGreaterEqual(len(preference_pairs_raw), len(preference_pairs))
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
            self.assertIn("terminal_reward", final_summaries[0])
            self.assertIn("finding_count", final_summaries[0])
            self.assertIn("terminal_schema_score", final_summaries[0])
            self.assertEqual(manifest["artifacts"]["finding_record_count"], len(finding_records))

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
            self.assertEqual(
                _read_jsonl(run_a / "finding_records.jsonl"),
                _read_jsonl(run_b / "finding_records.jsonl"),
            )
            self.assertEqual(
                _read_jsonl(run_a / "preference_pairs_raw.jsonl"),
                _read_jsonl(run_b / "preference_pairs_raw.jsonl"),
            )
            self.assertEqual(
                _read_jsonl(run_a / "preference_pairs.jsonl"),
                _read_jsonl(run_b / "preference_pairs.jsonl"),
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
                    task_concurrency=2,
                    seed=5,
                    candidate_source="model_vllm",
                ),
                model_generator_config=ModelGeneratorConfig(
                    api_base="http://unused",
                    request_timeout_seconds=7200,
                ),
                candidate_generator=_FakeModelGenerator(),
            )

            self.assertEqual(manifest["generator"]["candidate_source"], "model_vllm")
            self.assertEqual(manifest["generator"]["model_name"], "gpt-oss-120b-bf16")
            self.assertEqual(manifest["generator"]["configured_api_mode"], "auto")
            self.assertEqual(manifest["config"]["task_concurrency"], 2)
            self.assertEqual(manifest["generator"]["request_timeout_seconds"], 7200)
            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            first_branch = branch_pools[0]["branches"][0]
            self.assertEqual(first_branch["metadata"]["generator_backend"], "model_vllm")
            self.assertEqual(first_branch["verifier_step"]["updated_state"]["relationship_status"], "validated_group")
            self.assertEqual(first_branch["local_score"]["schema_score"], 1.0)

    def test_task_concurrency_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_concurrency must be positive"):
            TrajectoryGenerationConfig(task_concurrency=0)

    def test_openai_candidate_generator_uses_native_runtime_tools_on_responses_actor(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="responses",
                model_name="gpt-oss-120b-bf16",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "id": "resp_actor",
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "content": [
                                {
                                    "type": "reasoning_text",
                                    "text": "Inspect the current group with a multiplex walk.",
                                }
                            ],
                        },
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Inspect the current group with a multiplex walk.",
                                }
                            ],
                        },
                        {
                            "type": "function_call",
                            "name": "rwr_multiplex",
                            "arguments": json.dumps(
                                {
                                    "seeds": ["ENSG1", "ENSG2"],
                                    "top_k": 5,
                                }
                            ),
                            "call_id": "call_rwr",
                        },
                    ],
                },
                {
                    "id": "resp_verifier",
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "content": [
                                {
                                    "type": "reasoning_text",
                                    "text": "The restart walk elevated ENSG3 with the seeds.",
                                }
                            ],
                        },
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "updated_interpretation": {
                                                "mechanistic_claim": "The evidence supports one shared module.",
                                                "main_evidence": "The restart walk elevated ENSG3 with the seeds.",
                                                "uncertainty": "",
                                                "next_subgoal": "",
                                            },
                                            "updated_state": {
                                                "relationship_status": "validated_group",
                                                "predicted_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                                                "mechanistic_labels": [
                                                    {
                                                        "label_source": "go",
                                                        "label_name": "toy process",
                                                        "label_id": "GO:0000001",
                                                    }
                                                ],
                                                "continuation_decision": "stop",
                                                "verifier_notes": "Accepted the ranked expansion.",
                                            },
                                        }
                                    ),
                                }
                            ],
                        },
                    ],
                },
            ]
        )

        actor_candidates = generator.generate_actor_candidates(
            context,
            task_row=task_row,
            step_index=0,
            n_act=1,
            seed=5,
        )
        self.assertEqual(generator.api_mode, "responses")
        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr_multiplex")
        self.assertEqual(actor_candidates[0]["tool_action"]["arguments"]["top_k"], 5)
        first_request = generator.session.requests[0]["json"]
        self.assertTrue(generator.session.requests[0]["url"].endswith("/responses"))
        self.assertIn("tools", first_request)
        self.assertEqual(first_request["tools"][0]["name"], "query_mygene")
        self.assertNotIn("tool_choice", first_request)
        self.assertNotIn("reasoning", first_request)
        self.assertNotIn("text", first_request)
        self.assertEqual(first_request["input"][0]["role"], "system")
        self.assertEqual(first_request["input"][1]["role"], "user")

        tool_action = ToolAction(
            tool_name=actor_candidates[0]["tool_action"]["tool_name"],
            arguments=actor_candidates[0]["tool_action"]["arguments"],
            call_id="call_0",
        )
        actor_step = ActorStep(
            reasoning_text=actor_candidates[0]["reasoning_text"],
            tool_action=tool_action,
        )
        verifier_candidates = generator.generate_verifier_candidates(
            context,
            task_row=task_row,
            actor_candidate=actor_candidates[0],
            actor_step=actor_step,
            observation=None,
            step_index=0,
            n_ver=1,
            seed=9,
        )
        self.assertEqual(
            verifier_candidates[0]["payload"]["updated_state"]["relationship_status"],
            "validated_group",
        )
        second_request = generator.session.requests[1]["json"]
        self.assertTrue(generator.session.requests[1]["url"].endswith("/responses"))
        self.assertNotIn("tools", second_request)
        self.assertNotIn("tool_choice", second_request)
        self.assertNotIn("reasoning", second_request)
        self.assertEqual(second_request["text"]["format"]["type"], "json_schema")
        self.assertEqual(second_request["text"]["format"]["name"], "emit_verifier_update")

    def test_openai_candidate_generator_auto_mode_prefers_completions_for_gpt_oss(self) -> None:
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="auto",
                model_name="gpt-oss-120b-bf16",
            )
        )
        self.assertEqual(generator.api_mode, "completions")

        non_gpt_oss_generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="auto",
                model_name="llama-3.1-70b-instruct",
            )
        )
        self.assertEqual(non_gpt_oss_generator.api_mode, "chat_completions")

    def test_openai_candidate_generator_can_force_completions_for_gpt_oss(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="completions",
                model_name="gpt-oss-120b-bf16",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "text": (
                                "Use a restart walk to expand the current group.\n"
                                'TOOL_ACTION: {"tool_name": "rwr_multiplex", '
                                '"arguments": {"seeds": ["ENSG1", "ENSG2"], "top_k": 5}}'
                            ),
                            "token_ids": [1, 2, 3],
                        }
                    ]
                }
            ]
        )

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeTokenizer())
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            actor_candidates = generator.generate_actor_candidates(
                context,
                task_row=task_row,
                step_index=0,
                n_act=1,
                seed=5,
            )

        self.assertEqual(generator.api_mode, "completions")
        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr_multiplex")
        self.assertIn("restart walk", actor_candidates[0]["reasoning_text"])
        first_request = generator.session.requests[0]["json"]
        self.assertTrue(generator.session.requests[0]["url"].endswith("/completions"))
        self.assertNotIn("guided_json", first_request)
        self.assertFalse(first_request["add_special_tokens"])
        self.assertTrue(first_request["return_token_ids"])
        self.assertIn('"enable_thinking": false', first_request["prompt"])

    def test_openai_candidate_generator_can_force_completions_for_gpt_oss_verifier(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        actor_candidate = {
            "reasoning_text": "A restart walk is the cheapest grounded expansion move.",
            "tool_action": {
                "tool_name": "rwr_multiplex",
                "arguments": {"seeds": ["ENSG1", "ENSG2"], "top_k": 5},
            },
            "generator_errors": [],
        }
        actor_step = ActorStep(
            reasoning_text=actor_candidate["reasoning_text"],
            tool_action=ToolAction(
                tool_name="rwr_multiplex",
                arguments={"seeds": ["ENSG1", "ENSG2"], "top_k": 5},
                call_id="call_1",
            ),
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="completions",
                model_name="gpt-oss-120b-bf16",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "text": json.dumps(
                                {
                                    "updated_interpretation": {
                                        "mechanistic_claim": "The visible evidence supports one coherent module.",
                                        "main_evidence": "The restart walk pulled ENSG3 close to the seed genes.",
                                        "uncertainty": "",
                                        "next_subgoal": "",
                                    },
                                    "updated_state": {
                                        "relationship_status": "validated_group",
                                        "predicted_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                                        "mechanistic_labels": [],
                                        "continuation_decision": "stop",
                                        "verifier_notes": "Accepted the grounded expansion.",
                                    },
                                }
                            ),
                            "token_ids": [4, 5, 6],
                        }
                    ]
                }
            ]
        )

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeTokenizer())
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            verifier_candidates = generator.generate_verifier_candidates(
                context,
                task_row=task_row,
                actor_candidate=actor_candidate,
                actor_step=actor_step,
                observation=None,
                step_index=0,
                n_ver=1,
                seed=11,
            )

        self.assertEqual(verifier_candidates[0]["payload"]["updated_state"]["relationship_status"], "validated_group")
        first_request = generator.session.requests[0]["json"]
        self.assertTrue(generator.session.requests[0]["url"].endswith("/completions"))
        self.assertEqual(first_request["guided_json"]["required"], ["updated_interpretation", "updated_state"])
        self.assertFalse(first_request["add_special_tokens"])
        self.assertTrue(first_request["return_token_ids"])
        self.assertIn('"enable_thinking": false', first_request["prompt"])
        self.assertNotIn('"task_type"', first_request["prompt"])
        self.assertNotIn('"difficulty"', first_request["prompt"])
        self.assertNotIn('"evidence_mode"', first_request["prompt"])
        self.assertIn('\\"deterministic_observation\\": null', first_request["prompt"])

    def test_openai_candidate_generator_fails_fast_when_gpt_oss_template_ignores_enable_thinking(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="completions",
                model_name="gpt-oss-120b-bf16",
            )
        )
        generator.session = _RecordingSession([])

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: _BrokenFinalChannelTokenizer()
            )
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with self.assertRaisesRegex(
                RuntimeError,
                r"gpt-oss-120b-bf16.*does not appear to honor enable_thinking=False",
            ):
                generator.generate_actor_candidates(
                    context,
                    task_row=task_row,
                    step_index=0,
                    n_act=1,
                    seed=5,
                )

        self.assertEqual(generator.session.requests, [])

    def test_gpt_oss_120b_real_chat_template_supports_enable_thinking_for_completions(self) -> None:
        model_path = Path(__file__).resolve().parents[2] / "models" / "gpt-oss-120b-bf16"
        if not model_path.exists():
            self.skipTest(f"Model path not found: {model_path}")

        try:
            import transformers  # type: ignore
        except ImportError:
            self.skipTest("transformers is not installed in this environment")

        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="completions",
                model_name=str(model_path),
            )
        )
        prompt = generator._render_completion_prompt(
            [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Emit one object."},
            ],
            disable_hidden_thinking=True,
        )

        self.assertTrue(
            prompt.rstrip().endswith("<|start|>assistant<|channel|>final<|message|>"),
            msg=prompt[-200:],
        )

    def test_openai_candidate_generator_can_force_chat_completions(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="chat_completions",
                model_name="llama-3.1-70b-instruct",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "content": "Inspect the current group with a multiplex walk.",
                                "tool_calls": [
                                    {
                                        "id": "call_rwr",
                                        "type": "function",
                                        "function": {
                                            "name": "rwr_multiplex",
                                            "arguments": json.dumps(
                                                {
                                                    "seeds": ["ENSG1", "ENSG2"],
                                                    "top_k": 5,
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        )

        actor_candidates = generator.generate_actor_candidates(
            context,
            task_row=task_row,
            step_index=0,
            n_act=1,
            seed=5,
        )

        self.assertEqual(generator.api_mode, "chat_completions")
        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr_multiplex")
        first_request = generator.session.requests[0]["json"]
        self.assertTrue(generator.session.requests[0]["url"].endswith("/chat/completions"))
        self.assertIn("tools", first_request)
        self.assertEqual(first_request["tool_choice"], "auto")
        self.assertNotIn("reasoning_effort", first_request)
        self.assertNotIn("guided_json", first_request)
        self.assertNotIn('"task_type"', first_request["messages"][1]["content"])
        self.assertNotIn('"difficulty"', first_request["messages"][1]["content"])
        self.assertNotIn('"evidence_mode"', first_request["messages"][1]["content"])

    def test_openai_candidate_generator_accepts_freeform_actor_text_without_tool(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="chat_completions",
                model_name="llama-3.1-70b-instruct",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "content": "The visible context is enough to ask the verifier to summarize.",
                            },
                        }
                    ]
                }
            ]
        )

        actor_candidates = generator.generate_actor_candidates(
            context,
            task_row=task_row,
            step_index=0,
            n_act=1,
            seed=5,
        )

        self.assertIn("visible context", actor_candidates[0]["reasoning_text"])
        self.assertIsNone(actor_candidates[0]["tool_action"])
        self.assertEqual(actor_candidates[0]["generator_errors"], [])

    def test_openai_candidate_generator_uses_runtime_tool_for_gpt_oss_chat(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="chat_completions",
                model_name="gpt-oss-20b-bf16",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": "Inspect the current group with a multiplex walk.",
                                "tool_calls": [
                                    {
                                        "id": "call_rwr",
                                        "type": "function",
                                        "function": {
                                            "name": "rwr_multiplex",
                                            "arguments": json.dumps(
                                                {
                                                    "seeds": ["ENSG1", "ENSG2"],
                                                    "top_k": 5,
                                                }
                                            ),
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            ]
        )

        actor_candidates = generator.generate_actor_candidates(
            context,
            task_row=task_row,
            step_index=0,
            n_act=1,
            seed=5,
        )

        self.assertEqual(generator.api_mode, "chat_completions")
        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr_multiplex")
        self.assertIn("multiplex walk", actor_candidates[0]["reasoning_text"])
        self.assertEqual(len(generator.session.requests), 1)
        first_request = generator.session.requests[0]["json"]
        self.assertTrue(generator.session.requests[0]["url"].endswith("/chat/completions"))
        self.assertEqual(first_request["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("tools", first_request)
        self.assertEqual(first_request["tools"][3]["function"]["name"], "rwr_multiplex")
        self.assertEqual(first_request["tool_choice"], "auto")
        self.assertNotIn("guided_json", first_request)
        self.assertNotIn("reasoning_effort", first_request)
        self.assertNotIn("draft_reasoning_text", first_request["messages"][1]["content"])
        self.assertNotIn('"task_type"', first_request["messages"][1]["content"])
        self.assertNotIn('"difficulty"', first_request["messages"][1]["content"])
        self.assertNotIn('"evidence_mode"', first_request["messages"][1]["content"])

    def test_gpt_oss_chat_falls_back_to_completions_when_response_is_blank(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="chat_completions",
                model_name="gpt-oss-20b-bf16",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "message": {
                                "content": None,
                                "reasoning_content": None,
                                "tool_calls": [],
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "text": (
                                "Inspect the current group with a multiplex walk.\n"
                                'TOOL_ACTION: {"tool_name": "rwr_multiplex", '
                                '"arguments": {"seeds": ["ENSG1", "ENSG2"], "top_k": 5}}'
                            ),
                            "token_ids": [7, 8, 9],
                        }
                    ]
                },
            ]
        )

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeTokenizer())
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            actor_candidates = generator.generate_actor_candidates(
                context,
                task_row=task_row,
                step_index=0,
                n_act=1,
                seed=5,
            )

        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr_multiplex")
        self.assertEqual(actor_candidates[0]["generator_errors"], [])
        self.assertEqual(len(generator.session.requests), 2)
        self.assertTrue(generator.session.requests[0]["url"].endswith("/chat/completions"))
        self.assertTrue(generator.session.requests[1]["url"].endswith("/completions"))
        fallback_request = generator.session.requests[1]["json"]
        self.assertNotIn("guided_json", fallback_request)
        self.assertIn('"enable_thinking": false', fallback_request["prompt"])
        self.assertIn("chat_completions_blank_visible_output", actor_candidates[0]["raw_text"])

    def test_openai_candidate_generator_disables_hidden_thinking_for_gpt_oss_verifier_chat(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        actor_candidate = {
            "reasoning_text": "A restart walk is the cheapest grounded expansion move.",
            "tool_action": {
                "tool_name": "rwr_multiplex",
                "arguments": {"seeds": ["ENSG1", "ENSG2"], "top_k": 5},
            },
            "generator_errors": [],
        }
        actor_step = ActorStep(
            reasoning_text=actor_candidate["reasoning_text"],
            tool_action=ToolAction(
                tool_name="rwr_multiplex",
                arguments={"seeds": ["ENSG1", "ENSG2"], "top_k": 5},
                call_id="call_1",
            ),
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="chat_completions",
                model_name="gpt-oss-20b-bf16",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_emit_verifier_update",
                                        "type": "function",
                                        "function": {
                                            "name": "emit_verifier_update",
                                            "arguments": json.dumps(
                                                {
                                                    "updated_interpretation": {
                                                        "mechanistic_claim": "The visible evidence supports one coherent module.",
                                                        "main_evidence": "The restart walk pulled ENSG3 close to the seed genes.",
                                                        "uncertainty": "",
                                                        "next_subgoal": "",
                                                    },
                                                    "updated_state": {
                                                        "relationship_status": "validated_group",
                                                        "predicted_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                                                        "mechanistic_labels": [],
                                                        "continuation_decision": "stop",
                                                        "verifier_notes": "Accepted the grounded expansion.",
                                                    },
                                                }
                                            ),
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            ]
        )

        verifier_candidates = generator.generate_verifier_candidates(
            context,
            task_row=task_row,
            actor_candidate=actor_candidate,
            actor_step=actor_step,
            observation=None,
            step_index=0,
            n_ver=1,
            seed=11,
        )

        self.assertEqual(verifier_candidates[0]["payload"]["updated_state"]["relationship_status"], "validated_group")
        first_request = generator.session.requests[0]["json"]
        self.assertEqual(first_request["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(first_request["tool_choice"]["function"]["name"], "emit_verifier_update")
        self.assertNotIn("reasoning_effort", first_request)
        self.assertIn('"deterministic_observation": null', first_request["messages"][1]["content"])
        self.assertNotIn('"task_type"', first_request["messages"][1]["content"])
        self.assertNotIn('"difficulty"', first_request["messages"][1]["content"])
        self.assertNotIn('"evidence_mode"', first_request["messages"][1]["content"])

    def test_openai_candidate_generator_falls_back_to_chat_when_responses_returns_blank(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="responses",
                model_name="gpt-oss-120b-bf16",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "id": "resp_actor_blank",
                    "status": "completed",
                    "output": [],
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "content": "Inspect the current group with a multiplex walk.",
                                "tool_calls": [
                                    {
                                        "id": "call_rwr",
                                        "type": "function",
                                        "function": {
                                            "name": "rwr_multiplex",
                                            "arguments": json.dumps(
                                                {
                                                    "seeds": ["ENSG1", "ENSG2"],
                                                    "top_k": 5,
                                                }
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                },
            ]
        )

        actor_candidates = generator.generate_actor_candidates(
            context,
            task_row=task_row,
            step_index=0,
            n_act=1,
            seed=5,
        )

        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr_multiplex")
        self.assertTrue(generator.session.requests[0]["url"].endswith("/responses"))
        self.assertTrue(generator.session.requests[1]["url"].endswith("/chat/completions"))
        self.assertIn("tools", generator.session.requests[1]["json"])

    def test_model_backed_generation_fails_when_no_usable_model_candidates_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "strict_model_run"
            with self.assertRaisesRegex(RuntimeError, "no usable candidates"):
                generate_trajectories(
                    task_rows=_task_rows()[:1],
                    out_dir=out_dir,
                    environment=_build_environment(),
                    config=TrajectoryGenerationConfig(
                        max_steps=3,
                        n_act=1,
                        n_ver=1,
                        seed=3,
                        candidate_source="model_vllm",
                    ),
                    model_generator_config=ModelGeneratorConfig(api_base="http://unused"),
                    candidate_generator=_UnusableModelGenerator(),
                )

    def test_model_backed_generation_can_opt_in_to_heuristic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "fallback_model_run"
            manifest = generate_trajectories(
                task_rows=_task_rows()[:1],
                out_dir=out_dir,
                environment=_build_environment(),
                config=TrajectoryGenerationConfig(
                    max_steps=3,
                    n_act=1,
                    n_ver=1,
                    seed=3,
                    candidate_source="model_vllm",
                    allow_model_fallback=True,
                ),
                model_generator_config=ModelGeneratorConfig(api_base="http://unused"),
                candidate_generator=_UnusableModelGenerator(),
            )

            self.assertTrue(manifest["config"]["allow_model_fallback"])
            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            self.assertEqual(branch_pools[0]["branches"][0]["metadata"]["generator_backend"], "heuristic_fallback")


if __name__ == "__main__":
    unittest.main()
