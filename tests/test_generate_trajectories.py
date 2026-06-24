import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from runtime import (
    ActorStep,
    CandidateBranch,
    ContinuationState,
    GeneGroup,
    Interpretation,
    LocalScoreBreakdown,
    RelationshipStatus,
    SharedPrefixContext,
    TerminationReason,
    ToolAction,
    ToolObservation,
    ToolObservationStatus,
    VerifierStep,
    append_evidence_record,
    initialize_state_from_corum_task,
    replace_predicted_groups,
    set_continuation_state,
)
from runtime.environment import RuntimeEnvironment
from scripts.generate_trajectories import (
    DEFAULT_FULL_BRAIN_RWR_HPC_FLIST,
    DEFAULT_MEMBERSHIP_EDIT_BRANCHES,
    DEFAULT_MEMBERSHIP_EDIT_MAX_CUMULATIVE_ADDITIONS,
    DEFAULT_MEMBERSHIP_EDIT_MAX_DROP_PAIRS,
    DEFAULT_MEMBERSHIP_EDIT_TOP_K,
    DEFAULT_REQUIRE_RWR_HPC,
    DEFAULT_RWR_HPC_EDGELIST_HAS_HEADERS,
    DEFAULT_STORE_DIR,
    DEFAULT_USE_FULL_BRAIN_RWR_HPC,
    ModelGeneratorConfig,
    OpenAICompatibleCandidateGenerator,
    TrajectoryGenerationConfig,
    _actor_prompt_payload,
    _actor_sampling_directive_payload,
    _build_actor_step_from_model_candidate,
    _build_evidence_record,
    _build_labels_from_model_payload,
    _deterministic_membership_edit_branches,
    _expand_tool_action_gene_set_handles,
    _load_gene_id_background,
    _load_task_rows,
    _mine_preference_pairs,
    _normalize_runtime_tool_action,
    _observation_for_verifier_prompt,
    _pair_is_task_safe,
    _preference_difficulty_for_rank,
    _prefetch_mechanism_evidence_cache,
    _resolve_rwr_hpc_build_dir,
    _resolve_rwr_hpc_flist,
    _resolve_store_dir,
    _runtime_tool_parameters,
    _score_branch,
    _select_best_branch,
    _task_shard_bucket,
    _validate_verifier_payload,
    _verifier_output_schema,
    _verifier_prompt_payload,
    generate_trajectories,
    parse_args,
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


def _collect_json_keys(value) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_collect_json_keys(item))
        return keys
    if isinstance(value, list):
        keys = set()
        for item in value:
            keys.update(_collect_json_keys(item))
        return keys
    return set()


def _branch_for_pair_filter(branch_id: str, state, gene_ids: list[str], metrics: dict[str, float], complex_delta: float) -> CandidateBranch:
    updated_state = replace_predicted_groups(
        state,
        predicted_groups=[
            GeneGroup(
                group_id=branch_id,
                gene_ids=gene_ids,
                gene_symbols=[],
                rationale="Pair-filter test branch.",
            )
        ],
        relationship_status=RelationshipStatus.VALIDATED_GROUP,
    )
    updated_state.continuation_state = ContinuationState.STOP
    return CandidateBranch(
        branch_id=branch_id,
        actor_step=ActorStep(reasoning_text="Test branch.", tool_action=None),
        verifier_step=VerifierStep(
            updated_interpretation=Interpretation(
                mechanistic_claim="Test claim.",
                main_evidence="Test evidence.",
                uncertainty="",
                next_subgoal="",
            ),
            updated_state=updated_state,
            continuation_decision=ContinuationState.STOP,
            verifier_notes="Pair-filter test verifier.",
        ),
        local_score=LocalScoreBreakdown(
            schema_score=1.0,
            complex_membership_delta=complex_delta,
            mechanistic_label_delta=0.8,
            efficiency_penalty=0.0,
            total_score=2.0,
            normalized_score=1.0,
            mechanism_evidence_delta=0.8,
            mechanism_evidence_score=0.8,
            score_metadata={
                "complex": {
                    "best_group_pre": {"metrics": {"jaccard": 0.5, "precision": 1.0, "recall": 0.5}},
                    "best_group_post": {"metrics": metrics},
                }
            },
        ),
    )


def _branch_for_exactness_test(
    branch_id: str,
    state,
    gene_ids: list[str],
    *,
    metrics: dict[str, float],
    task_success_level: str,
    normalized_score: float,
    total_score: float,
    relationship_status: RelationshipStatus = RelationshipStatus.VALIDATED_GROUP,
) -> CandidateBranch:
    updated_state = replace_predicted_groups(
        state,
        predicted_groups=[
            GeneGroup(
                group_id=branch_id,
                gene_ids=gene_ids,
                gene_symbols=[],
                rationale="Exactness test branch.",
            )
        ],
        relationship_status=relationship_status,
    )
    updated_state.continuation_state = ContinuationState.STOP
    return CandidateBranch(
        branch_id=branch_id,
        actor_step=ActorStep(reasoning_text="Exactness test branch.", tool_action=None),
        verifier_step=VerifierStep(
            updated_interpretation=Interpretation(
                mechanistic_claim="Exactness test claim.",
                main_evidence="Exactness test evidence.",
                uncertainty="",
                next_subgoal="",
            ),
            updated_state=updated_state,
            continuation_decision=ContinuationState.STOP,
            verifier_notes="Exactness test verifier.",
        ),
        local_score=LocalScoreBreakdown(
            schema_score=1.0,
            complex_membership_delta=float(metrics["jaccard"]),
            mechanistic_label_delta=0.5,
            efficiency_penalty=0.0,
            total_score=total_score,
            normalized_score=normalized_score,
            mechanism_evidence_delta=0.5,
            mechanism_evidence_score=0.5,
            score_metadata={
                "schema_valid": True,
                "complex": {
                    "best_group_pre": {
                        "metrics": {"jaccard": 0.5, "precision": 1.0, "recall": 0.5}
                    },
                    "best_group_post": {"metrics": metrics},
                },
                "task_success": {"task_success_level": task_success_level},
            },
        ),
    )


def _tool_evidence_branch(
    branch_id: str,
    state,
    gene_ids: list[str],
    *,
    tool_name: str,
    arguments: dict,
    payload: dict,
    metrics: dict[str, float],
    task_success_level: str = "partial",
) -> CandidateBranch:
    branch = _branch_for_exactness_test(
        branch_id,
        state,
        gene_ids,
        metrics=metrics,
        task_success_level=task_success_level,
        normalized_score=0.5,
        total_score=5.0,
        relationship_status=RelationshipStatus.PARTIALLY_OBSERVED_GROUP,
    )
    call_id = f"call_{branch_id}"
    branch.actor_step = ActorStep(
        reasoning_text="Collect visible tool evidence.",
        tool_action=ToolAction(tool_name=tool_name, arguments=arguments, call_id=call_id),
    )
    branch.observation = ToolObservation(
        status=ToolObservationStatus.SUCCESS,
        provenance={"tool_name": tool_name, "runtime": "unit_test"},
        call_id=call_id,
        payload=payload,
    )
    return branch


class _FakeModelGenerator:
    model_name = "gpt-oss-120b-bf16"

    def generate_actor_candidates(self, context, *, task_row, step_index, n_act, seed, environment=None, **kwargs):
        del context, n_act, seed, environment, kwargs
        if task_row["task_type"] == "recovery":
            return [
                {
                    "reasoning_text": "Expand the current group with a restart walk.",
                    "tool_action": {"tool_name": "rwr", "arguments": {"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5}},
                    "raw_text": '{"reasoning_text":"Expand","tool_action":{"tool_name":"rwr","arguments":{"seed_genes":["ENSG1","ENSG2"],"top_k":5}}}',
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

    def generate_actor_candidates(self, context, *, task_row, step_index, n_act, seed, environment=None, **kwargs):
        del context, task_row, step_index, n_act, seed, environment, kwargs
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


class _InvalidToolModelGenerator:
    model_name = "gpt-oss-120b-bf16"

    def generate_actor_candidates(self, context, *, task_row, step_index, n_act, seed, environment=None, **kwargs):
        del context, task_row, step_index, n_act, seed, environment, kwargs
        return [
            {
                "reasoning_text": "Probe a gene that is not present in the runtime graph.",
                "tool_action": {
                    "tool_name": "get_neighbors",
                    "arguments": {"gene": "ENSG_MISSING"},
                },
                "raw_text": '{"reasoning_text":"Probe missing gene","tool_action":{"tool_name":"get_neighbors","arguments":{"gene":"ENSG_MISSING"}}}',
                "generator_errors": [],
            }
        ]

    def generate_verifier_candidates(self, context, *, task_row, actor_candidate, actor_step, observation, step_index, n_ver, seed):
        del context, task_row, actor_candidate, actor_step, observation, step_index, n_ver, seed
        raise AssertionError("Verifier generation should not run for a semantically invalid actor tool.")


class _InvalidVerifierThenValidGenerator:
    model_name = "gpt-oss-120b-bf16"

    def generate_actor_candidates(self, context, *, task_row, step_index, n_act, seed, environment=None, **kwargs):
        del context, task_row, step_index, n_act, seed, environment, kwargs
        return [
            {
                "reasoning_text": "Expand the current group with a restart walk.",
                "tool_action": {
                    "tool_name": "rwr",
                    "arguments": {"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5},
                },
                "raw_text": '{"reasoning_text":"Expand","tool_action":{"tool_name":"rwr","arguments":{"seed_genes":["ENSG1","ENSG2"],"top_k":5}}}',
                "generator_errors": [],
            }
        ]

    def generate_verifier_candidates(self, context, *, task_row, actor_candidate, actor_step, observation, step_index, n_ver, seed):
        del context, task_row, actor_candidate, actor_step, observation, step_index, n_ver, seed
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
                        "mechanistic_labels": [{"label_source": "go"}],
                        "continuation_decision": "stop",
                        "verifier_notes": "This verifier payload has an invalid label.",
                    },
                },
                "raw_text": '{"updated_state":{"mechanistic_labels":[{"label_source":"go"}]}}',
                "generator_errors": [],
            },
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
                        "verifier_notes": "Accepted the grounded expansion.",
                    },
                },
                "raw_text": '{"updated_state":{"mechanistic_labels":[{"label_source":"go","label_name":"toy process","label_id":"GO:0000001"}]}}',
                "generator_errors": [],
            },
        ]


class _ToolCoverageRetryGenerator:
    model_name = "gpt-oss-120b-bf16"

    def __init__(self) -> None:
        self.force_tool_coverage_flags: list[bool] = []

    def generate_actor_candidates(
        self,
        context,
        *,
        task_row,
        step_index,
        n_act,
        seed,
        environment=None,
        force_tool_coverage=False,
        **kwargs,
    ):
        del context, task_row, step_index, n_act, seed, environment, kwargs
        self.force_tool_coverage_flags.append(force_tool_coverage)
        if force_tool_coverage:
            return [
                {
                    "reasoning_text": "Retry with a restart walk to recover missing members.",
                    "tool_action": {
                        "tool_name": "rwr",
                        "arguments": {"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5},
                    },
                    "raw_text": "{}",
                    "generator_errors": [],
                }
            ]
        return [
            {
                "reasoning_text": "Stop with the visible seed pair.",
                "tool_action": None,
                "raw_text": "{}",
                "generator_errors": [],
            }
        ]

    def generate_verifier_candidates(self, context, *, task_row, actor_candidate, actor_step, observation, step_index, n_ver, seed):
        del context, task_row, actor_candidate, observation, step_index, n_ver, seed
        predicted_gene_ids = ["ENSG1", "ENSG2", "ENSG3"] if actor_step.tool_action is not None else ["ENSG1", "ENSG2"]
        return [
            {
                "payload": {
                    "updated_interpretation": {
                        "mechanistic_claim": "The evidence supports one coherent module.",
                        "main_evidence": "The branch updates the candidate group.",
                        "uncertainty": "",
                        "next_subgoal": "",
                    },
                    "updated_state": {
                        "relationship_status": "validated_group",
                        "predicted_gene_ids": predicted_gene_ids,
                        "mechanistic_labels": [],
                        "continuation_decision": "stop",
                        "verifier_notes": "test branch",
                    },
                },
                "raw_text": "{}",
                "generator_errors": [],
            }
        ]


class _NativeToolThenRwrCoverageGenerator(_ToolCoverageRetryGenerator):
    def generate_actor_candidates(
        self,
        context,
        *,
        task_row,
        step_index,
        n_act,
        seed,
        environment=None,
        force_tool_coverage=False,
        **kwargs,
    ):
        if force_tool_coverage:
            return super().generate_actor_candidates(
                context,
                task_row=task_row,
                step_index=step_index,
                n_act=n_act,
                seed=seed,
                environment=environment,
                force_tool_coverage=force_tool_coverage,
                **kwargs,
            )
        del context, task_row, step_index, n_act, seed, environment, kwargs
        self.force_tool_coverage_flags.append(False)
        return [
            {
                "reasoning_text": "Inspect a native neighborhood before expanding.",
                "tool_action": {
                    "tool_name": "get_neighbors",
                    "arguments": {"gene": "ENSG1"},
                },
                "raw_text": "{}",
                "generator_errors": [],
            }
        ]


class _DuplicateThenRepairGenerator:
    model_name = "gpt-oss-120b-bf16"

    def __init__(self) -> None:
        self.config = ModelGeneratorConfig(
            api_base="http://unused",
            actor_tool_repair_retry_count=1,
        )
        self.repair_calls: list[dict] = []

    def generate_actor_candidates(
        self,
        context,
        *,
        task_row,
        step_index,
        n_act,
        seed,
        environment=None,
        **kwargs,
    ):
        del context, task_row, n_act, seed, environment, kwargs
        return [
            {
                "reasoning_text": "Inspect the same seed neighborhood.",
                "tool_action": {
                    "tool_name": "get_neighbors",
                    "arguments": {"gene": "ENSG1"},
                },
                "raw_text": "{}",
                "generator_errors": [],
            }
        ]

    def repair_actor_candidate(
        self,
        context,
        *,
        task_row,
        step_index,
        actor_index,
        actor_candidate,
        actor_step,
        observation,
        errors,
        seed,
        environment=None,
        prior_actions=None,
        attempt_index=0,
    ):
        del context, task_row, step_index, actor_index, actor_candidate, actor_step
        del seed, environment, attempt_index
        self.repair_calls.append(
            {
                "errors": list(errors),
                "observation_status": observation.status.value if observation else None,
                "prior_action_count": len(list(prior_actions or [])),
            }
        )
        return {
            "reasoning_text": "Repair by inspecting a different seed neighborhood.",
            "tool_action": {
                "tool_name": "get_neighbors",
                "arguments": {"gene": "ENSG2"},
            },
            "raw_text": "{}",
            "generator_errors": [],
            "actor_repair": {
                "attempted": True,
                "attempt_count": 1,
                "success": False,
                "previous_errors": list(errors),
            },
        }

    def generate_verifier_candidates(
        self,
        context,
        *,
        task_row,
        actor_candidate,
        actor_step,
        observation,
        step_index,
        n_ver,
        seed,
    ):
        del context, task_row, actor_candidate, actor_step, observation, n_ver, seed
        return [
            {
                "payload": {
                    "updated_interpretation": {
                        "mechanistic_claim": "The evidence supports one coherent module.",
                        "main_evidence": "Neighborhood probes were completed without repeating a tool call.",
                        "uncertainty": "",
                        "next_subgoal": "",
                    },
                    "updated_state": {
                        "relationship_status": "validated_group",
                        "predicted_gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
                        "mechanistic_labels": [],
                        "continuation_decision": "continue" if step_index == 0 else "stop",
                        "verifier_notes": "test branch",
                    },
                },
                "raw_text": "{}",
                "generator_errors": [],
            }
        ]


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
    def test_verifier_schema_and_parser_accept_reactome_labels(self) -> None:
        label_source_enum = _verifier_output_schema()["properties"]["updated_state"]["properties"][
            "mechanistic_labels"
        ]["items"]["properties"]["label_source"]["enum"]
        labels, errors = _build_labels_from_model_payload(
            {
                "mechanistic_labels": [
                    {
                        "label_source": "REAC",
                        "label_name": "Initial triggering of complement",
                        "label_id": "REAC:R-HSA-166663",
                        "evidence_ids": ["call_0"],
                    },
                    {
                        "label_source": "GO:BP",
                        "label_name": "immune response",
                        "label_id": "GO:0006955",
                        "evidence_ids": ["call_1"],
                    },
                    {
                        "label_source": "KEGG",
                        "label_name": "complement and coagulation cascades",
                        "label_id": "KEGG:hsa04610",
                        "evidence_ids": ["call_2"],
                    }
                ]
            }
        )

        self.assertIn("reactome", label_source_enum)
        self.assertEqual(errors, [])
        self.assertEqual(labels[0].label_source.value, "reactome")
        self.assertEqual(labels[1].label_source.value, "go")
        self.assertEqual(labels[2].label_source.value, "other")
        corum_labels, corum_errors = _build_labels_from_model_payload(
            {
                "mechanistic_labels": [
                    {
                        "label_source": "CORUM",
                        "label_name": "legacy complex label",
                        "label_id": "CORUM:1",
                        "evidence_ids": ["call_3"],
                    }
                ]
            }
        )
        self.assertEqual(corum_labels, [])
        self.assertEqual(len(corum_errors), 1)
        self.assertIn("label_source must be one of", corum_errors[0])
        self.assertEqual(
            _validate_verifier_payload(
                {
                    "updated_interpretation": {
                        "mechanistic_claim": "A visible Reactome pathway is enriched.",
                        "main_evidence": "The enrichment observation reported REAC:R-HSA-166663.",
                        "uncertainty": "",
                        "next_subgoal": "",
                    },
                    "updated_state": {
                        "relationship_status": "validated_group",
                        "predicted_gene_ids": ["ENSG1", "ENSG2"],
                        "mechanistic_labels": [
                            {
                                "label_source": "REAC",
                                "label_name": "Initial triggering of complement",
                                "label_id": "REAC:R-HSA-166663",
                                "evidence_ids": ["call_0"],
                            }
                        ],
                        "continuation_decision": "stop",
                        "verifier_notes": "Accepted Reactome alias source.",
                    },
                }
            ),
            [],
        )

    def test_parse_args_defaults_to_full_brain_required_rwr_hpc(self) -> None:
        args = parse_args([])
        rwr_hpc_flist = _resolve_rwr_hpc_flist(args)

        self.assertTrue(DEFAULT_USE_FULL_BRAIN_RWR_HPC)
        self.assertTrue(DEFAULT_REQUIRE_RWR_HPC)
        self.assertTrue(DEFAULT_RWR_HPC_EDGELIST_HAS_HEADERS)
        self.assertTrue(args.use_full_brain_rwr_hpc)
        self.assertTrue(args.require_rwr_hpc)
        self.assertTrue(args.rwr_hpc_edgelist_has_headers)
        self.assertEqual(args.generator_verifier_repair_retry_count, 1)
        self.assertEqual(args.generator_actor_tool_repair_retry_count, 1)
        self.assertEqual(args.generator_prompt_token_limit, 0)
        self.assertEqual(args.membership_edit_branches, DEFAULT_MEMBERSHIP_EDIT_BRANCHES)
        self.assertEqual(args.membership_edit_top_k, DEFAULT_MEMBERSHIP_EDIT_TOP_K)
        self.assertEqual(
            args.membership_edit_max_cumulative_additions,
            DEFAULT_MEMBERSHIP_EDIT_MAX_CUMULATIVE_ADDITIONS,
        )
        self.assertEqual(args.membership_edit_max_drop_pairs, DEFAULT_MEMBERSHIP_EDIT_MAX_DROP_PAIRS)
        self.assertEqual(rwr_hpc_flist, DEFAULT_FULL_BRAIN_RWR_HPC_FLIST)

    def test_parse_args_uses_default_store_only_after_rwr_hpc_opt_out(self) -> None:
        args = parse_args(["--no-use-full-brain-rwr-hpc", "--no-require-rwr-hpc"])
        rwr_hpc_flist = _resolve_rwr_hpc_flist(args)

        self.assertIsNone(rwr_hpc_flist)
        self.assertFalse(args.require_rwr_hpc)
        self.assertEqual(_resolve_store_dir(args, rwr_hpc_flist), DEFAULT_STORE_DIR)

    def test_parse_args_full_brain_rwr_hpc_suppresses_default_store(self) -> None:
        args = parse_args(["--use-full-brain-rwr-hpc"])
        rwr_hpc_flist = _resolve_rwr_hpc_flist(args)

        self.assertEqual(rwr_hpc_flist, DEFAULT_FULL_BRAIN_RWR_HPC_FLIST)
        self.assertIsNone(
            _resolve_store_dir(
                args,
                rwr_hpc_flist,
                default_full_brain_store_dir=Path("missing_full_brain_store"),
            )
        )

    def test_parse_args_full_brain_rwr_hpc_uses_existing_binary_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "full_brain_store"
            store_dir.mkdir()
            args = parse_args(["--use-full-brain-rwr-hpc"])
            rwr_hpc_flist = _resolve_rwr_hpc_flist(args)

            self.assertEqual(
                _resolve_store_dir(args, rwr_hpc_flist, default_full_brain_store_dir=store_dir),
                store_dir,
            )

    def test_parse_args_can_override_rwr_hpc_build_dir(self) -> None:
        args = parse_args(["--multiplex-flist", "toy.tsv", "--rwr-hpc-build-dir", "build_rwr"])

        self.assertEqual(_resolve_rwr_hpc_build_dir(args, structured_backend_requested=True), Path("build_rwr"))

    def test_pair_filter_rejects_mechanism_improvement_with_worse_gene_correctness(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        chosen = _branch_for_pair_filter(
            "chosen_mechanism_only",
            state,
            ["ENSG1"],
            {"jaccard": 1.0 / 3.0, "precision": 1.0, "recall": 1.0 / 3.0},
            complex_delta=-0.1,
        )
        rejected = _branch_for_pair_filter(
            "rejected_better_genes",
            state,
            ["ENSG1", "ENSG2", "ENSG3"],
            {"jaccard": 1.0, "precision": 1.0, "recall": 1.0},
            complex_delta=0.5,
        )

        self.assertFalse(
            _pair_is_task_safe(
                task_type="recovery",
                context=context,
                chosen_branch=chosen,
                rejected_branch=rejected,
            )
        )

    def test_task_quality_selection_prefers_exact_recovery_over_near_top_partial(self) -> None:
        task_row = _task_rows()[0]
        _interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        exact = _branch_for_exactness_test(
            "exact_recovery",
            state,
            ["ENSG1", "ENSG2", "ENSG3"],
            metrics={"jaccard": 1.0, "precision": 1.0, "recall": 1.0},
            task_success_level="positive",
            normalized_score=0.90,
            total_score=9.0,
        )
        partial = _branch_for_exactness_test(
            "partial_recovery",
            state,
            ["ENSG1", "ENSG2"],
            metrics={"jaccard": 2.0 / 3.0, "precision": 1.0, "recall": 2.0 / 3.0},
            task_success_level="partial",
            normalized_score=0.99,
            total_score=9.9,
        )

        selected = _select_best_branch(
            [partial, exact],
            task_row=task_row,
            prior_state=state,
            selection_policy="task_quality",
            selection_score_epsilon=0.10,
        )

        self.assertEqual(selected.branch_id, "exact_recovery")
        self.assertTrue(selected.metadata["selection_quality"]["exact_membership"])

    def test_task_quality_selection_prefers_membership_metrics_outside_score_window(self) -> None:
        task_row = _task_rows()[0]
        _interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        membership_best = _branch_for_exactness_test(
            "membership_best",
            state,
            ["ENSG1", "ENSG2", "ENSG3"],
            metrics={"jaccard": 1.0, "precision": 1.0, "recall": 1.0},
            task_success_level="partial",
            normalized_score=0.25,
            total_score=2.5,
            relationship_status=RelationshipStatus.PARTIALLY_OBSERVED_GROUP,
        )
        mechanism_top = _branch_for_exactness_test(
            "mechanism_top",
            state,
            ["ENSG1", "ENSG2"],
            metrics={"jaccard": 2.0 / 3.0, "precision": 1.0, "recall": 2.0 / 3.0},
            task_success_level="partial",
            normalized_score=1.0,
            total_score=10.0,
            relationship_status=RelationshipStatus.PARTIALLY_OBSERVED_GROUP,
        )

        selected = _select_best_branch(
            [mechanism_top, membership_best],
            task_row=task_row,
            prior_state=state,
            selection_policy="task_quality",
            selection_score_epsilon=0.10,
        )

        self.assertEqual(selected.branch_id, "membership_best")
        self.assertEqual(
            selected.metadata["selection_quality_scope"],
            "all_candidates_membership_first",
        )
        self.assertGreater(selected.metadata["selection_score_gap"], 0.10)
        self.assertTrue(selected.metadata["selection_quality"]["membership_metrics_exact"])
        self.assertFalse(selected.metadata["selection_quality"]["exact_membership"])

    def test_exact_recovery_pair_mining_keeps_exact_over_partial_below_margin(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        exact = _branch_for_exactness_test(
            "exact_recovery",
            state,
            ["ENSG1", "ENSG2", "ENSG3"],
            metrics={"jaccard": 1.0, "precision": 1.0, "recall": 1.0},
            task_success_level="positive",
            normalized_score=0.90,
            total_score=9.0,
        )
        partial = _branch_for_exactness_test(
            "partial_recovery",
            state,
            ["ENSG1", "ENSG2"],
            metrics={"jaccard": 2.0 / 3.0, "precision": 1.0, "recall": 2.0 / 3.0},
            task_success_level="partial",
            normalized_score=0.85,
            total_score=8.5,
        )

        pairs = _mine_preference_pairs(
            task_row=task_row,
            trajectory_id="traj_exact",
            trajectory_seed=7,
            step_index=0,
            context=context,
            branches=[exact, partial],
            chosen_branch=exact,
            score_margin=0.10,
            pair_mining_strategy="quality_balanced",
        )

        self.assertEqual(len(pairs), 1)
        self.assertLess(pairs[0].score_margin, 0.10)
        self.assertEqual(pairs[0].provenance["pair_category"], "exact_over_partial")
        self.assertTrue(pairs[0].provenance["chosen_exact_membership"])
        self.assertEqual(pairs[0].provenance["chosen_final_jaccard"], 1.0)
        self.assertEqual(pairs[0].provenance["rejected_task_success_level"], "partial")

    def test_exact_over_partial_pair_allows_lower_scalar_score_with_nonnegative_margin(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        exact = _branch_for_exactness_test(
            "exact_lower_scalar",
            state,
            ["ENSG1", "ENSG2", "ENSG3"],
            metrics={"jaccard": 1.0, "precision": 1.0, "recall": 1.0},
            task_success_level="positive",
            normalized_score=0.25,
            total_score=2.5,
        )
        partial = _branch_for_exactness_test(
            "partial_higher_scalar",
            state,
            ["ENSG1", "ENSG2"],
            metrics={"jaccard": 2.0 / 3.0, "precision": 1.0, "recall": 2.0 / 3.0},
            task_success_level="partial",
            normalized_score=0.99,
            total_score=9.9,
        )

        pairs = _mine_preference_pairs(
            task_row=task_row,
            trajectory_id="traj_exact_lower_scalar",
            trajectory_seed=7,
            step_index=0,
            context=context,
            branches=[exact, partial],
            chosen_branch=exact,
            score_margin=0.10,
            pair_mining_strategy="quality_balanced",
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].score_margin, 0.0)
        self.assertLess(pairs[0].provenance["normalized_score_delta"], 0.0)
        self.assertLess(pairs[0].provenance["raw_score_delta"], 0.0)
        self.assertEqual(pairs[0].provenance["pair_category"], "exact_over_partial")

    def test_deterministic_recovery_edits_generate_top_and_single_add_candidates(self) -> None:
        task_row = _task_rows()[0]
        _interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        source = _tool_evidence_branch(
            "rwr_source",
            state,
            ["ENSG1", "ENSG2"],
            tool_name="rwr",
            arguments={"seed_genes": ["ENSG1", "ENSG2"], "top_k": 8},
            payload={
                "seed_gene_ids": ["ENSG1", "ENSG2"],
                "results": [
                    {"gene_id": "ENSG1", "rank": 1, "score": 1.0},
                    {"gene_id": "ENSG3", "rank": 2, "score": 0.9},
                    {"gene_id": "ENSG4", "rank": 3, "score": 0.8},
                    {"gene_id": "ENSG5", "rank": 4, "score": 0.7},
                    {"gene_id": "ENSG2", "rank": 5, "score": 0.6},
                ],
            },
            metrics={"jaccard": 2.0 / 3.0, "precision": 1.0, "recall": 2.0 / 3.0},
        )

        branches = _deterministic_membership_edit_branches(
            task_row=task_row,
            prior_state=state,
            branches=[source],
            trajectory_id="traj_recovery_edit",
            step_index=0,
            max_steps=3,
            symbol_lookup={"ENSG1": "GENE1", "ENSG2": "GENE2", "ENSG3": "GENE3"},
            environment=_build_environment(),
            prior_actions=[],
            top_k=3,
            max_cumulative_additions=3,
            max_drop_pairs=0,
        )

        gene_sets = {tuple(group["gene_ids"]) for branch in branches for group in branch.to_dict()["verifier_step"]["updated_state"]["predicted_groups"]}
        self.assertIn(("ENSG1", "ENSG2", "ENSG3"), gene_sets)
        self.assertIn(("ENSG1", "ENSG2", "ENSG3", "ENSG4"), gene_sets)
        self.assertIn(("ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"), gene_sets)
        self.assertTrue(any(
            branch.metadata["deterministic_membership_edit"]["edit_kind"] == "recovery_add_single"
            for branch in branches
        ))
        for branch in branches:
            self.assertIn("deterministic_membership_edit", branch.metadata)
            self.assertFalse(_collect_json_keys(branch.metadata) & {"hidden_target", "target_gene_ids", "target_gene_symbols"})
            self.assertFalse(_collect_json_keys(branch.observation.to_dict()["provenance"]) & {"hidden_target", "target_gene_ids", "target_gene_symbols"})

    def test_deterministic_refinement_edits_generate_leave_one_out_and_bounded_pair_drops(self) -> None:
        task_row = json.loads(json.dumps(_task_rows()[0]))
        task_row["task_id"] = "corum_complex_test.refinement.easy.contextual"
        task_row["task_type"] = "refinement"
        task_row["visible_inputs"]["seed_gene_ids"] = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
        task_row["visible_inputs"]["seed_gene_symbols"] = ["GENE1", "GENE2", "GENE3", "GENE4"]
        _interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        source = _tool_evidence_branch(
            "refine_rwr_source",
            state,
            ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            tool_name="rwr",
            arguments={"seed_genes": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"], "top_k": 8},
            payload={
                "seed_gene_ids": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
                "results": [
                    {"gene_id": "ENSG1", "rank": 1, "score": 1.0},
                    {"gene_id": "ENSG2", "rank": 2, "score": 0.9},
                    {"gene_id": "ENSG3", "rank": 3, "score": 0.8},
                    {"gene_id": "ENSG4", "rank": 8, "score": 0.1},
                ],
            },
            metrics={"jaccard": 0.75, "precision": 0.75, "recall": 1.0},
        )

        branches = _deterministic_membership_edit_branches(
            task_row=task_row,
            prior_state=state,
            branches=[source],
            trajectory_id="traj_refinement_edit",
            step_index=0,
            max_steps=3,
            symbol_lookup={"ENSG1": "GENE1", "ENSG2": "GENE2", "ENSG3": "GENE3", "ENSG4": "GENE4"},
            environment=_build_environment(),
            prior_actions=[],
            top_k=4,
            max_cumulative_additions=3,
            max_drop_pairs=2,
        )

        edit_kinds = [branch.metadata["deterministic_membership_edit"]["edit_kind"] for branch in branches]
        self.assertGreaterEqual(edit_kinds.count("refinement_drop_single"), 4)
        self.assertLessEqual(edit_kinds.count("refinement_drop_pair"), 2)
        gene_sets = {tuple(group["gene_ids"]) for branch in branches for group in branch.to_dict()["verifier_step"]["updated_state"]["predicted_groups"]}
        self.assertIn(("ENSG1", "ENSG2", "ENSG3"), gene_sets)

    def test_exact_refinement_pair_category_beats_negative_extra_gene_branch(self) -> None:
        task_row = json.loads(json.dumps(_task_rows()[0]))
        task_row["task_id"] = "corum_complex_test.refinement.easy.contextual"
        task_row["task_type"] = "refinement"
        task_row["visible_inputs"]["seed_gene_ids"] = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
        task_row["visible_inputs"]["seed_gene_symbols"] = ["GENE1", "GENE2", "GENE3", "GENE4"]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        exact = _branch_for_exactness_test(
            "exact_refinement",
            state,
            ["ENSG1", "ENSG2", "ENSG3"],
            metrics={"jaccard": 1.0, "precision": 1.0, "recall": 1.0},
            task_success_level="positive",
            normalized_score=0.92,
            total_score=9.2,
        )
        negative = _branch_for_exactness_test(
            "negative_extra_gene",
            state,
            ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            metrics={"jaccard": 0.75, "precision": 0.75, "recall": 1.0},
            task_success_level="negative",
            normalized_score=0.80,
            total_score=8.0,
        )

        pairs = _mine_preference_pairs(
            task_row=task_row,
            trajectory_id="traj_refine_exact",
            trajectory_seed=11,
            step_index=0,
            context=context,
            branches=[exact, negative],
            chosen_branch=exact,
            score_margin=0.10,
            pair_mining_strategy="quality_balanced",
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].provenance["pair_category"], "exact_refinement")
        self.assertEqual(pairs[0].provenance["chosen_final_precision"], 1.0)
        self.assertEqual(pairs[0].provenance["rejected_final_precision"], 0.75)

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

    def test_model_prompt_payloads_do_not_include_corum_ground_truth_metadata(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        actor_payload = _actor_prompt_payload(context, step_index=0)
        verifier_payload = _verifier_prompt_payload(
            context,
            actor_step=ActorStep(reasoning_text="Probe the graph.", tool_action=None),
            observation=None,
            step_index=0,
        )

        blocked_keys = {
            "difficulty",
            "evidence_mode",
            "hidden_target",
            "mechanism_labels",
            "source_task_id",
            "target_gene_ids",
            "target_gene_symbols",
            "task_id",
            "task_type",
        }
        self.assertTrue(blocked_keys.isdisjoint(_collect_json_keys(actor_payload)))
        self.assertTrue(blocked_keys.isdisjoint(_collect_json_keys(verifier_payload)))
        self.assertIn("prompt_state", actor_payload)
        self.assertIn("prior_prompt_state", verifier_payload)
        self.assertNotIn("state", actor_payload)
        self.assertNotIn("prior_state", verifier_payload)
        self.assertNotIn("user_anchors", actor_payload["prompt_state"])
        self.assertNotIn("user_anchors", verifier_payload["prior_prompt_state"])

    def test_actor_prompt_payload_includes_tool_argument_reference(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )

        actor_payload = _actor_prompt_payload(
            context,
            step_index=0,
            environment=_build_environment(),
        )

        reference = actor_payload["tool_argument_reference"]
        self.assertIn("ENSG1", reference["candidate_gene_ids"])
        self.assertIn("ENSG2", reference["candidate_gene_ids"])
        self.assertEqual(reference["unavailable_candidate_gene_ids"], [])
        self.assertEqual(reference["gene_set_handles"]["__visible_seed_genes__"]["gene_count"], 2)
        self.assertEqual(reference["gene_set_handles"]["__current_candidate_group__"]["gene_count"], 2)
        self.assertIn("ppi", reference["available_layer_names"])
        self.assertIn("shortest_paths source_genes and target_genes", " ".join(reference["rules"]))
        self.assertIn("enrich_gene_set", reference["argument_shapes"])
        self.assertIn("query_mygene", reference["argument_shapes"])

    def test_actor_prompt_payload_includes_prior_tool_actions(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        prior_action = ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": "ENSG1"},
            call_id="prior_0",
        )

        actor_payload = _actor_prompt_payload(
            context,
            step_index=1,
            environment=_build_environment(),
            prior_actions=[prior_action],
        )

        reference = actor_payload["tool_argument_reference"]
        self.assertEqual(
            reference["prior_tool_actions"],
            [{"tool_name": "get_neighbors", "arguments": {"gene": "ENSG1"}, "index": 0}],
        )
        self.assertIn("Do not repeat", " ".join(reference["rules"]))

    def test_tool_action_gene_set_handles_expand_before_execution(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        state = replace_predicted_groups(
            state,
            [
                GeneGroup(
                    group_id="group_0",
                    gene_ids=["ENSG1", "ENSG2", "ENSG3"],
                    gene_symbols=["GENE1", "GENE2", "GENE3"],
                    rationale="Current candidate group.",
                )
            ],
        )
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )
        action = ToolAction(
            tool_name="enrich_gene_set",
            arguments={"genes": ["__current_candidate_group__"]},
            call_id="call_handle",
        )

        expanded = _expand_tool_action_gene_set_handles(action, context=context)

        assert expanded is not None
        self.assertEqual(expanded.arguments["genes"], ["ENSG1", "ENSG2", "ENSG3"])
        self.assertEqual(action.arguments["genes"], ["__current_candidate_group__"])

    def test_runtime_tool_schemas_document_strict_argument_shapes(self) -> None:
        shortest_path_schema = _runtime_tool_parameters("shortest_paths")
        self.assertEqual(shortest_path_schema["properties"]["source_genes"]["minItems"], 1)
        self.assertEqual(shortest_path_schema["properties"]["target_genes"]["minItems"], 1)
        self.assertIn("one-element array", shortest_path_schema["properties"]["source_genes"]["description"])

        induce_subgraph_schema = _runtime_tool_parameters("induce_subgraph")
        self.assertEqual(induce_subgraph_schema["properties"]["genes"]["minItems"], 1)
        self.assertEqual(induce_subgraph_schema["properties"]["layers"]["minItems"], 1)

        enrich_schema = _runtime_tool_parameters("enrich_gene_set")
        self.assertEqual(enrich_schema["properties"]["genes"]["minItems"], 1)
        self.assertIn("GO:BP", enrich_schema["properties"]["sources"]["description"])
        self.assertNotIn("CORUM", enrich_schema["properties"]["sources"]["description"])

        rank_schema = _runtime_tool_parameters("get_rank")
        self.assertEqual(rank_schema["required"], ["source_gene", "target_gene"])

        distance_schema = _runtime_tool_parameters("get_distance")
        self.assertEqual(distance_schema["properties"]["distance_metric"]["enum"], ["spearman", "pearson", "dot"])

        spearman_schema = _runtime_tool_parameters("get_spearman")
        self.assertEqual(spearman_schema["required"], ["gene_a", "gene_b"])

        layer_stats_schema = _runtime_tool_parameters("get_layer_stats")
        self.assertEqual(layer_stats_schema["properties"]["sort_by"]["enum"], ["edge_count", "node_count", "layer"])

        perturbation_schema = _runtime_tool_parameters("get_node_perturbation")
        self.assertEqual(perturbation_schema["required"], ["seed_genes", "perturb_genes"])

    def test_enrichment_observation_is_visible_and_recorded_as_evidence(self) -> None:
        observation = ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            provenance={"tool_name": "enrich_gene_set", "source": "cache"},
            call_id="call_enrich",
            payload={
                "query_gene_ids": ["ENSG1", "ENSG2"],
                "query_gene_count": 2,
                "background_gene_count": 5,
                "organism": "hsapiens",
                "sources": ["GO:BP"],
                "raw_result_count": 1,
                "results": [
                    {
                        "source": "GO:BP",
                        "native": "GO:0000001",
                        "name": "toy process",
                        "p_value": 0.001,
                        "significant": True,
                        "intersection_size": 2,
                        "precision": 1.0,
                    }
                ],
            },
        )

        prompt_payload = _observation_for_verifier_prompt(observation)
        evidence = _build_evidence_record(
            observation,
            step_index=0,
            branch_id="branch",
            symbol_lookup={},
        )

        self.assertEqual(prompt_payload["tool_name"], "enrich_gene_set")
        self.assertEqual(prompt_payload["payload"]["results"][0]["native"], "GO:0000001")
        self.assertEqual(evidence.provenance["payload"]["results"][0]["name"], "toy process")
        self.assertEqual(evidence.supporting_gene_ids, ["ENSG1", "ENSG2"])

    def test_graph_evidence_record_compacts_large_layer_provenance(self) -> None:
        observation = ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            provenance={
                "tool_name": "get_neighbors",
                "queried_layers": [f"layer_{index}" for index in range(100)],
            },
            call_id="call_neighbors",
            payload={
                "query_gene_id": "ENSG1",
                "unique_neighbor_count": 2,
                "unique_neighbors": ["ENSG2", "ENSG3"],
            },
        )

        evidence = _build_evidence_record(
            observation,
            step_index=0,
            branch_id="branch",
            symbol_lookup={},
        )

        self.assertNotIn("queried_layers", evidence.provenance)
        self.assertEqual(evidence.provenance["queried_layers_count"], 100)
        self.assertLessEqual(len(evidence.provenance["queried_layers_sample"]), 20)

    def test_prefetch_mechanism_evidence_cache_runs_annotation_tools(self) -> None:
        report = _prefetch_mechanism_evidence_cache(
            [_task_rows()[0]],
            _build_environment(),
            mygene_per_task=2,
            enrichment_top_k=5,
        )

        self.assertEqual(report["task_count"], 1)
        self.assertEqual(report["unique_enrichment_queries"], 1)
        self.assertEqual(report["unique_mygene_queries"], 2)
        self.assertEqual(report["tool_status_counts"]["enrich_gene_set.empty"], 1)
        self.assertEqual(report["tool_status_counts"]["query_mygene.empty"], 2)

    def test_module_key_sharding_keeps_task_family_blocks_together(self) -> None:
        rows = []
        task_types = ["explanation", "none", "recovery", "refinement"]
        evidence_modes = ["graph", "minimal"]
        for module_index in range(24):
            for task_type in task_types:
                for evidence_mode in evidence_modes:
                    rows.append(
                        {
                            "task_id": f"gw_dendrogram_module_{module_index:06d}.{task_type}.easy.{evidence_mode}",
                            "task_type": task_type,
                        }
                    )

        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            with task_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            for shard_index in range(4):
                shard_rows = _load_task_rows(
                    task_path,
                    task_shard_index=shard_index,
                    task_shard_count=4,
                )
                modules = {row["task_id"].split(".", 1)[0] for row in shard_rows}
                for module in modules:
                    module_task_types = {
                        row["task_type"]
                        for row in shard_rows
                        if row["task_id"].startswith(module + ".")
                    }
                    self.assertEqual(module_task_types, set(task_types))
                    self.assertEqual(
                        _task_shard_bucket({"task_id": module + ".explanation.easy.graph"}, 4),
                        shard_index,
                    )

    def test_load_gene_id_background_reads_modules_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "modules.jsonl"
            path.write_text(
                json.dumps({"gene_ids": ["ENSG1", "ENSG2"]}) + "\n"
                + json.dumps({"gene_ids": ["ENSG2", "ENSG3"]}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(_load_gene_id_background(path), ["ENSG1", "ENSG2", "ENSG3"])

    def test_actor_verbalized_sampling_uses_distinct_prompt_directives(self) -> None:
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
                actor_sampling_strategy="verbalized",
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"content": "Use the current visible evidence directly."},
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"content": "Probe the induced subgraph for coherence."},
                        }
                    ]
                },
            ]
        )

        actor_candidates = generator.generate_actor_candidates(
            context,
            task_row=task_row,
            step_index=0,
            n_act=2,
            seed=5,
            environment=_build_environment(),
        )

        self.assertEqual(len(actor_candidates), 2)
        self.assertEqual(len(generator.session.requests), 2)
        prompts = [request["json"]["messages"][1]["content"] for request in generator.session.requests]
        self.assertIn('"actor_sampling_directive"', prompts[0])
        self.assertIn('"recovery_rwr_expansion"', prompts[0])
        self.assertIn('"recovery_neighbor_expansion"', prompts[1])
        self.assertEqual(
            actor_candidates[0]["actor_sampling_directive"]["directive_name"],
            "recovery_rwr_expansion",
        )
        self.assertEqual(
            actor_candidates[1]["actor_sampling_directive"]["directive_name"],
            "recovery_neighbor_expansion",
        )

    def test_actor_sampling_strategy_must_be_known(self) -> None:
        with self.assertRaisesRegex(ValueError, "actor_sampling_strategy"):
            ModelGeneratorConfig(actor_sampling_strategy="unknown")

    def test_generator_prompt_token_budget_fails_locally_with_section_diagnostics(self) -> None:
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="chat_completions",
                model_name="llama-3.1-70b-instruct",
                prompt_token_limit=10,
            )
        )
        generator.session = _RecordingSession([])

        with self.assertRaisesRegex(RuntimeError, "prompt_token_budget_exceeded.*largest_sections"):
            generator._chat(
                [
                    {"role": "system", "content": "System prompt."},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "prompt_state": {"summary": "x" * 200},
                                "deterministic_observation": {"summary": "y" * 200},
                            }
                        ),
                    },
                ],
                n=1,
                seed=0,
            )

    def test_verifier_prompt_compacts_large_tool_observations(self) -> None:
        layers = []
        unique_neighbors = []
        for layer_index in range(60):
            neighbors = [
                f"ENSG_LAYER_{layer_index}_{neighbor_index}"
                for neighbor_index in range(30)
            ]
            unique_neighbors.extend(neighbors)
            layers.append(
                {
                    "layer_name": f"layer_{layer_index}",
                    "neighbors": neighbors,
                    "neighbor_count": len(neighbors),
                }
            )

        observation = ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            provenance={
                "tool_name": "get_neighbors",
                "queried_layers": [f"layer_{index}" for index in range(60)],
            },
            call_id="call_big_neighbors",
            payload={
                "query_gene_id": "ENSG1",
                "layers": layers,
                "unique_neighbors": unique_neighbors,
                "unique_neighbor_count": len(unique_neighbors),
            },
        )

        compact = _observation_for_verifier_prompt(observation)

        self.assertIsNotNone(compact)
        assert compact is not None
        self.assertEqual(compact["provenance"]["queried_layers_count"], 60)
        self.assertEqual(len(compact["provenance"]["queried_layers_sample"]), 8)
        self.assertEqual(compact["payload"]["unique_neighbor_count"], len(unique_neighbors))
        self.assertEqual(len(compact["payload"]["unique_neighbors_sample"]), 12)
        self.assertEqual(len(compact["payload"]["layers_with_neighbors_sample"]), 8)
        self.assertNotIn("layers", compact["payload"])
        self.assertNotIn("unique_neighbors", compact["payload"])
        self.assertLess(len(json.dumps(compact, sort_keys=True)), 7000)

    def test_verifier_prompt_highlights_rwr_non_seed_candidates(self) -> None:
        observation = ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            provenance={"tool_name": "rwr_multiplex", "active_layers": ["ppi"]},
            call_id="call_rwr",
            payload={
                "seed_gene_ids": ["ENSG1", "ENSG2"],
                "active_seed_gene_ids": ["ENSG1", "ENSG2"],
                "active_layers": ["ppi"],
                "top_k": 500,
                "results": [
                    {"gene_id": "ENSG1", "score": 0.30},
                    {"gene_id": "ENSG3", "score": 0.12},
                    {"gene_id": "ENSG2", "score": 0.11},
                    {"gene_id": "ENSG4", "score": 0.05},
                ],
            },
        )

        compact = _observation_for_verifier_prompt(observation)

        self.assertIsNotNone(compact)
        assert compact is not None
        payload = compact["payload"]
        self.assertEqual(payload["top_k"], 500)
        self.assertEqual(payload["non_seed_result_count"], 2)
        self.assertEqual(
            [result["gene_id"] for result in payload["top_non_seed_results"]],
            ["ENSG3", "ENSG4"],
        )
        self.assertEqual(payload["ranked_non_seed_gene_ids"], ["ENSG3", "ENSG4"])
        self.assertIn("ranked_non_seed_gene_ids", payload["recovery_interpretation_hint"])

    def test_prior_prompt_state_uses_markov_digest_for_accumulated_rwr_evidence(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=6)
        large_results = [
            {"gene_id": "ENSG1", "score": 1.0},
            {"gene_id": "ENSG2", "score": 0.9},
        ] + [
            {"gene_id": f"ENSG_NON_SEED_{index:03d}", "score": 0.5 / (index + 1)}
            for index in range(500)
        ]
        observation = ToolObservation(
            status=ToolObservationStatus.SUCCESS,
            provenance={"tool_name": "rwr", "active_layers": [f"layer_{index}" for index in range(100)]},
            call_id="call_rwr_large",
            payload={
                "seed_gene_ids": ["ENSG1", "ENSG2"],
                "active_seed_gene_ids": ["ENSG1", "ENSG2"],
                "active_layers": [f"layer_{index}" for index in range(100)],
                "top_k": 500,
                "results": large_results,
            },
        )
        for index in range(6):
            evidence = _build_evidence_record(
                observation,
                step_index=index,
                branch_id=f"branch_{index}",
                symbol_lookup={},
            )
            assert evidence is not None
            state = append_evidence_record(state, evidence)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )

        payload = _verifier_prompt_payload(
            context,
            actor_step=ActorStep(reasoning_text="Continue recovery.", tool_action=None),
            observation=None,
            step_index=3,
            task_type="recovery",
        )
        prior_state = payload["prior_prompt_state"]
        prompt_text = json.dumps(prior_state, sort_keys=True)

        self.assertEqual(prior_state["evidence_digest"]["evidence_count"], 6)
        self.assertEqual(prior_state["evidence_digest"]["tool_counts"]["rwr"], 6)
        self.assertEqual(len(prior_state["evidence_digest"]["recent_evidence_summaries"]), 6)
        self.assertNotIn("evidence_log", prior_state)
        self.assertNotIn("results_sample", prompt_text)
        self.assertNotIn("ranked_non_seed_gene_ids_sample", prompt_text)
        self.assertNotIn("ENSG_NON_SEED_499", prompt_text)
        self.assertLess(len(prompt_text), 5000)

    def test_recovery_verifier_prompt_includes_expansion_guidance(self) -> None:
        task_row = _task_rows()[0]
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )

        payload = _verifier_prompt_payload(
            context,
            actor_step=ActorStep(reasoning_text="Run RWR.", tool_action=None),
            observation=None,
            step_index=0,
            task_type="recovery",
        )

        self.assertEqual(
            payload["task_guidance"]["objective"],
            "Recover exact coherent complex membership beyond the current seed/candidate group.",
        )
        self.assertIn("non-seed", " ".join(payload["task_guidance"]["candidate_policy"]))
        self.assertIn("update predicted_gene_ids", " ".join(payload["task_guidance"]["candidate_policy"]))
        self.assertIn("Exact recovery", payload["task_guidance"]["exact_success_policy"])
        self.assertIn("continue", " ".join(payload["task_guidance"]["candidate_policy"]))

        directive = _actor_sampling_directive_payload(task_type="recovery", sample_index=4)
        self.assertEqual(directive["directive_name"], "recovery_commit_expansion")
        self.assertIn("commit that expanded membership", directive["instruction"])

    def test_refinement_verifier_prompt_includes_exact_nonterminal_guidance(self) -> None:
        task_row = _task_rows()[0]
        task_row = dict(task_row)
        task_row["task_type"] = "refinement"
        interpretation, state = initialize_state_from_corum_task(task_row, max_budget=3)
        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=interpretation,
            state=state,
            source_task_id=task_row["task_id"],
        )

        payload = _verifier_prompt_payload(
            context,
            actor_step=ActorStep(reasoning_text="Probe pruning.", tool_action=None),
            observation=None,
            step_index=0,
            task_type="refinement",
        )

        self.assertIn("Exact refinement", payload["task_guidance"]["exact_success_policy"])
        self.assertIn("questionable extra", " ".join(payload["task_guidance"]["candidate_policy"]))
        self.assertIn("pruned coherent subset", " ".join(payload["task_guidance"]["candidate_policy"]))

        directive = _actor_sampling_directive_payload(task_type="refinement", sample_index=3)
        self.assertEqual(directive["directive_name"], "refinement_commit_pruned_subset")
        self.assertIn("commit the pruned membership", directive["instruction"])

    def test_partial_recovery_stop_is_forced_to_continue_while_budget_remains(self) -> None:
        task_row = _task_rows()[0]
        interpretation, prior_state = initialize_state_from_corum_task(task_row, max_budget=3)
        partial_state = replace_predicted_groups(
            prior_state,
            [
                GeneGroup(
                    group_id="group_0",
                    gene_ids=["ENSG1", "ENSG2"],
                    gene_symbols=["GENE1", "GENE2"],
                    rationale="Seed-only partial recovery.",
                )
            ],
            relationship_status=RelationshipStatus.PARTIALLY_OBSERVED_GROUP,
        )
        partial_state.remaining_budget = 2
        partial_state = set_continuation_state(
            partial_state,
            ContinuationState.STOP,
            termination_reason=TerminationReason.MODEL_STOP,
        )
        branch = CandidateBranch(
            branch_id="partial_stop",
            actor_step=ActorStep(reasoning_text="Stop with the seed subset.", tool_action=None),
            observation=None,
            verifier_step=VerifierStep(
                updated_interpretation=interpretation,
                updated_state=partial_state,
                continuation_decision=ContinuationState.STOP,
                verifier_notes="Partial group looks plausible.",
            ),
            local_score=LocalScoreBreakdown(
                schema_score=0.0,
                complex_membership_delta=0.0,
                mechanistic_label_delta=0.0,
                efficiency_penalty=0.0,
                total_score=0.0,
            ),
        )

        scored = _score_branch(
            task_row,
            prior_state,
            branch,
            step_index=0,
            max_steps=3,
            prior_actions=[],
            environment=_build_environment(),
        )

        self.assertEqual(
            scored.local_score.score_metadata["task_success"]["task_success_level"],
            "partial",
        )
        self.assertEqual(scored.verifier_step.continuation_decision, ContinuationState.CONTINUE)
        self.assertEqual(scored.verifier_step.updated_state.continuation_state, ContinuationState.CONTINUE)
        self.assertIsNone(scored.verifier_step.updated_state.termination_reason)
        override = scored.metadata["exact_membership_nonterminal_override"]
        self.assertTrue(override["applied"])
        self.assertIn("task_success_level=partial", override["reason"])

    def test_actor_rationale_prompt_does_not_include_corum_ground_truth_metadata(self) -> None:
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
        captured: dict[str, str] = {}

        def fake_chat(**kwargs):
            captured["user_prompt"] = kwargs["messages"][1]["content"]
            return [
                {
                    "message": {"content": '{"reasoning_text":"Probe the visible graph."}'},
                    "finish_reason": "stop",
                }
            ]

        generator._chat = fake_chat
        reasoning_text, errors = generator._generate_actor_reasoning(
            context,
            task_row=task_row,
            step_index=0,
            seed=5,
        )

        blocked_keys = {
            "difficulty",
            "evidence_mode",
            "hidden_target",
            "mechanism_labels",
            "source_task_id",
            "target_gene_ids",
            "target_gene_symbols",
            "task_id",
            "task_row",
            "task_type",
        }
        self.assertEqual(reasoning_text, "Probe the visible graph.")
        self.assertEqual(errors, [])
        prompt_payload = json.loads(captured["user_prompt"])
        self.assertTrue(blocked_keys.isdisjoint(_collect_json_keys(prompt_payload)))
        self.assertNotIn("ENSG3", captured["user_prompt"])
        self.assertNotIn("toy process", captured["user_prompt"])

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
            self.assertIn("task_success", final_summaries[0])
            self.assertIn("task_success_level", final_summaries[0])
            self.assertIn("task_success_metadata", final_summaries[0])
            blocked_artifact_keys = {
                "terminal_score_metadata",
                "raw_actor_response",
                "raw_verifier_response",
                "token_ids",
                "prompt_token_ids",
            }
            for artifact_rows in (
                branch_pools,
                trajectory_turns,
                finding_records,
                preference_pairs_raw,
                preference_pairs,
                final_summaries,
            ):
                for row in artifact_rows:
                    self.assertTrue(blocked_artifact_keys.isdisjoint(_collect_json_keys(row)))
            self.assertEqual(manifest["artifacts"]["finding_record_count"], len(finding_records))

    def test_model_trajectory_artifacts_exclude_raw_model_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "trajectories"
            generate_trajectories(
                task_rows=_task_rows(),
                out_dir=out_dir,
                environment=_build_environment(),
                config=TrajectoryGenerationConfig(
                    candidate_source="model_vllm",
                    max_steps=3,
                    n_act=1,
                    n_ver=1,
                    seed=7,
                ),
                candidate_generator=_FakeModelGenerator(),
            )

            blocked_artifact_keys = {
                "terminal_score_metadata",
                "raw_actor_response",
                "raw_verifier_response",
                "token_ids",
                "prompt_token_ids",
            }
            for filename in (
                "branch_pools.jsonl",
                "trajectory_turns.jsonl",
                "finding_records.jsonl",
                "preference_pairs_raw.jsonl",
                "preference_pairs.jsonl",
                "final_summaries.jsonl",
            ):
                for row in _read_jsonl(out_dir / filename):
                    self.assertTrue(blocked_artifact_keys.isdisjoint(_collect_json_keys(row)))

    def test_model_actor_step_normalizes_all_layer_alias_before_storage(self) -> None:
        actor_step, errors = _build_actor_step_from_model_candidate(
            {
                "reasoning_text": "Inspect all available graph layers.",
                "tool_action": {
                    "tool_name": "induce_subgraph",
                    "arguments": {
                        "genes": ["ENSG1", "ENSG2"],
                        "layers": ["all"],
                    },
                },
            },
            trajectory_id="trajectory",
            step_index=0,
            actor_index=0,
        )

        self.assertEqual(errors, [])
        self.assertIsNotNone(actor_step.tool_action)
        self.assertEqual(actor_step.tool_action.arguments, {"genes": ["ENSG1", "ENSG2"]})

    def test_model_actor_step_normalizes_null_shortest_path_layer_before_storage(self) -> None:
        actor_step, errors = _build_actor_step_from_model_candidate(
            {
                "reasoning_text": "Check connectivity across all available graph layers.",
                "tool_action": {
                    "tool_name": "shortest_path",
                    "arguments": {
                        "source": "ENSG1",
                        "target": "ENSG2",
                        "layer": None,
                    },
                },
            },
            trajectory_id="trajectory",
            step_index=0,
            actor_index=0,
        )

        self.assertEqual(errors, [])
        self.assertIsNotNone(actor_step.tool_action)
        self.assertEqual(actor_step.tool_action.arguments, {"source": "ENSG1", "target": "ENSG2"})

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

    def test_model_actor_tool_repair_recovers_duplicate_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "model_repair"
            generator = _DuplicateThenRepairGenerator()
            generate_trajectories(
                task_rows=[_task_rows()[0]],
                out_dir=out_dir,
                environment=_build_environment(),
                config=TrajectoryGenerationConfig(
                    max_steps=2,
                    n_act=1,
                    n_ver=1,
                    seed=5,
                    candidate_source="model_vllm",
                ),
                candidate_generator=generator,
            )

            self.assertEqual(len(generator.repair_calls), 1)
            self.assertEqual(generator.repair_calls[0]["observation_status"], "invalid")
            self.assertIn("duplicate_tool_call", " ".join(generator.repair_calls[0]["errors"]))
            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            repaired_branch = branch_pools[1]["branches"][0]
            self.assertEqual(
                repaired_branch["actor_step"]["tool_action"]["arguments"]["gene"],
                "ENSG2",
            )
            self.assertTrue(repaired_branch["metadata"]["actor_repair"]["success"])

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
                            "name": "rwr",
                            "arguments": json.dumps(
                                {
                                    "seed_genes": ["ENSG1", "ENSG2"],
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
                                                        "evidence_ids": [],
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
        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr")
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
                                'TOOL_ACTION: {"tool_name": "rwr", '
                                '"arguments": {"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5}}'
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
        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr")
        self.assertIn("restart walk", actor_candidates[0]["reasoning_text"])
        first_request = generator.session.requests[0]["json"]
        self.assertTrue(generator.session.requests[0]["url"].endswith("/completions"))
        self.assertNotIn("guided_json", first_request)
        self.assertFalse(first_request["add_special_tokens"])
        self.assertNotIn("return_token_ids", first_request)
        self.assertNotIn("token_ids", actor_candidates[0]["raw_text"])
        self.assertNotIn("prompt_token_ids", actor_candidates[0]["raw_text"])
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
                "tool_name": "rwr",
                "arguments": {"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5},
            },
            "generator_errors": [],
        }
        actor_step = ActorStep(
            reasoning_text=actor_candidate["reasoning_text"],
            tool_action=ToolAction(
                tool_name="rwr",
                arguments={"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5},
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
        self.assertNotIn("return_token_ids", first_request)
        self.assertNotIn("token_ids", verifier_candidates[0]["raw_text"])
        self.assertNotIn("prompt_token_ids", verifier_candidates[0]["raw_text"])
        self.assertIn('"enable_thinking": false', first_request["prompt"])
        self.assertNotIn('"task_type"', first_request["prompt"])
        self.assertNotIn('"difficulty"', first_request["prompt"])
        self.assertNotIn('"evidence_mode"', first_request["prompt"])
        self.assertIn('\\"deterministic_observation\\": null', first_request["prompt"])

    def test_openai_candidate_generator_repairs_incomplete_verifier_json(self) -> None:
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
            "reasoning_text": "The current evidence can be summarized directly.",
            "tool_action": None,
            "generator_errors": [],
        }
        actor_step = ActorStep(reasoning_text=actor_candidate["reasoning_text"])
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="completions",
                model_name="gpt-oss-120b-bf16",
                verifier_repair_retry_count=1,
            )
        )
        generator.session = _RecordingSession(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "text": '{"updated_interpretation": {"mechanistic_claim": "partial"',
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "text": json.dumps(
                                {
                                    "updated_interpretation": {
                                        "mechanistic_claim": "The visible evidence supports one coherent module.",
                                        "main_evidence": "The current seed genes remain the best supported group.",
                                        "uncertainty": "",
                                        "next_subgoal": "",
                                    },
                                    "updated_state": {
                                        "relationship_status": "validated_group",
                                        "predicted_gene_ids": ["ENSG1", "ENSG2"],
                                        "mechanistic_labels": [],
                                        "continuation_decision": "stop",
                                        "verifier_notes": "Repaired verifier JSON.",
                                    },
                                }
                            ),
                        }
                    ]
                },
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

        self.assertEqual(len(generator.session.requests), 2)
        self.assertEqual(verifier_candidates[0]["generator_errors"], [])
        self.assertEqual(
            verifier_candidates[0]["payload"]["updated_state"]["relationship_status"],
            "validated_group",
        )
        repair = verifier_candidates[0]["verifier_repair"]
        self.assertTrue(repair["success"])
        self.assertIn("verifier_json_parse_error:", repair["original_errors"][0])
        repair_request = generator.session.requests[1]["json"]
        self.assertEqual(repair_request["temperature"], 0.0)
        self.assertEqual(repair_request["top_p"], 1.0)
        self.assertIn("repair_task", repair_request["prompt"])

    def test_openai_candidate_generator_repairs_schema_incomplete_verifier_label(self) -> None:
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
            "reasoning_text": "The current evidence can be summarized directly.",
            "tool_action": None,
            "generator_errors": [],
        }
        actor_step = ActorStep(reasoning_text=actor_candidate["reasoning_text"])
        generator = OpenAICompatibleCandidateGenerator(
            ModelGeneratorConfig(
                api_base="http://unused",
                api_mode="completions",
                model_name="gpt-oss-120b-bf16",
                verifier_repair_retry_count=1,
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
                                        "main_evidence": "The current seed genes remain the best supported group.",
                                        "uncertainty": "",
                                        "next_subgoal": "",
                                    },
                                    "updated_state": {
                                        "relationship_status": "validated_group",
                                        "predicted_gene_ids": ["ENSG1", "ENSG2"],
                                        "mechanistic_labels": [{"label_source": "go"}],
                                        "continuation_decision": "stop",
                                        "verifier_notes": "Missing label fields.",
                                    },
                                }
                            ),
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "text": json.dumps(
                                {
                                    "updated_interpretation": {
                                        "mechanistic_claim": "The visible evidence supports one coherent module.",
                                        "main_evidence": "The current seed genes remain the best supported group.",
                                        "uncertainty": "",
                                        "next_subgoal": "",
                                    },
                                    "updated_state": {
                                        "relationship_status": "validated_group",
                                        "predicted_gene_ids": ["ENSG1", "ENSG2"],
                                        "mechanistic_labels": [],
                                        "continuation_decision": "stop",
                                        "verifier_notes": "Dropped the unsupported malformed label.",
                                    },
                                }
                            ),
                        }
                    ]
                },
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

        self.assertEqual(verifier_candidates[0]["generator_errors"], [])
        self.assertEqual(verifier_candidates[0]["payload"]["updated_state"]["mechanistic_labels"], [])
        repair = verifier_candidates[0]["verifier_repair"]
        self.assertTrue(repair["success"])
        self.assertIn("verifier_label_0_missing_name", repair["original_errors"])

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
                                            "name": "rwr",
                                            "arguments": json.dumps(
                                                {
                                                    "seed_genes": ["ENSG1", "ENSG2"],
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
        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr")
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
                                            "name": "rwr",
                                            "arguments": json.dumps(
                                                {
                                                    "seed_genes": ["ENSG1", "ENSG2"],
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
        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr")
        self.assertIn("multiplex walk", actor_candidates[0]["reasoning_text"])
        self.assertEqual(len(generator.session.requests), 1)
        first_request = generator.session.requests[0]["json"]
        self.assertTrue(generator.session.requests[0]["url"].endswith("/chat/completions"))
        self.assertEqual(first_request["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("tools", first_request)
        self.assertIn(
            "rwr",
            [tool["function"]["name"] for tool in first_request["tools"]],
        )
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
                                'TOOL_ACTION: {"tool_name": "rwr", '
                                '"arguments": {"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5}}'
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

        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr")
        self.assertEqual(actor_candidates[0]["generator_errors"], [])
        self.assertEqual(len(generator.session.requests), 2)
        self.assertTrue(generator.session.requests[0]["url"].endswith("/chat/completions"))
        self.assertTrue(generator.session.requests[1]["url"].endswith("/completions"))
        fallback_request = generator.session.requests[1]["json"]
        self.assertNotIn("guided_json", fallback_request)
        self.assertNotIn("return_token_ids", fallback_request)
        self.assertNotIn("token_ids", actor_candidates[0]["raw_text"])
        self.assertNotIn("prompt_token_ids", actor_candidates[0]["raw_text"])
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
                "tool_name": "rwr",
                "arguments": {"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5},
            },
            "generator_errors": [],
        }
        actor_step = ActorStep(
            reasoning_text=actor_candidate["reasoning_text"],
            tool_action=ToolAction(
                tool_name="rwr",
                arguments={"seed_genes": ["ENSG1", "ENSG2"], "top_k": 5},
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
                                            "name": "rwr",
                                            "arguments": json.dumps(
                                                {
                                                    "seed_genes": ["ENSG1", "ENSG2"],
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

        self.assertEqual(actor_candidates[0]["tool_action"]["tool_name"], "rwr")
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

    def test_model_backed_generation_rejects_semantically_invalid_actor_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "invalid_tool_model_run"
            with self.assertRaisesRegex(RuntimeError, "actor_tool_semantics_invalid"):
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
                    candidate_generator=_InvalidToolModelGenerator(),
                )

    def test_model_backed_generation_skips_invalid_verifier_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "invalid_verifier_model_run"
            generate_trajectories(
                task_rows=_task_rows()[:1],
                out_dir=out_dir,
                environment=_build_environment(),
                config=TrajectoryGenerationConfig(
                    max_steps=3,
                    n_act=1,
                    n_ver=2,
                    seed=3,
                    candidate_source="model_vllm",
                    membership_edit_branches="off",
                ),
                model_generator_config=ModelGeneratorConfig(api_base="http://unused"),
                candidate_generator=_InvalidVerifierThenValidGenerator(),
            )

            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            self.assertEqual(len(branch_pools[0]["branches"]), 1)
            retained_branch = branch_pools[0]["branches"][0]
            self.assertEqual(retained_branch["metadata"]["generator_errors"], [])
            self.assertTrue(retained_branch["local_score"]["score_metadata"]["schema_valid"])

    def test_tool_coverage_retry_and_quality_pair_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "tool_coverage_quality_run"
            generator = _ToolCoverageRetryGenerator()
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
                    selection_policy="task_quality",
                    pair_mining_strategy="quality_balanced",
                    tool_coverage_retry_count=1,
                    membership_edit_branches="off",
                ),
                model_generator_config=ModelGeneratorConfig(api_base="http://unused"),
                candidate_generator=generator,
            )

            self.assertEqual(generator.force_tool_coverage_flags, [False, True])
            self.assertEqual(manifest["config"]["selection_policy"], "task_quality")
            self.assertEqual(manifest["config"]["pair_mining_strategy"], "quality_balanced")
            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            selected = next(
                branch
                for branch in branch_pools[0]["branches"]
                if branch["branch_id"] == branch_pools[0]["selected_branch_id"]
            )
            self.assertEqual(selected["actor_step"]["tool_action"]["tool_name"], "rwr")
            self.assertEqual(selected["actor_step"]["tool_action"]["arguments"]["top_k"], 1500)
            self.assertEqual(selected["metadata"]["selection_policy"], "task_quality")
            self.assertEqual(
                selected["metadata"]["tool_argument_defaults"]["reason"],
                "recovery_expansion_requires_broad_non_seed_candidate_search",
            )

            preference_pairs_raw = _read_jsonl(out_dir / "preference_pairs_raw.jsonl")
            self.assertTrue(preference_pairs_raw)
            provenance = preference_pairs_raw[0]["provenance"]
            self.assertEqual(provenance["pair_mining_strategy"], "quality_balanced")
            self.assertEqual(provenance["pair_category"], "exact_over_partial")
            self.assertEqual(provenance["chosen_tool_name"], "rwr")
            self.assertEqual(provenance["rejected_tool_name"], "no_tool")
            self.assertTrue(provenance["chosen_exact_membership"])
            self.assertEqual(provenance["rejected_task_success_level"], "partial")
            self.assertGreater(provenance["chosen_gene_count"], provenance["rejected_gene_count"])

    def test_rwr_coverage_retry_runs_when_only_native_graph_tool_was_sampled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "rwr_coverage_quality_run"
            generator = _NativeToolThenRwrCoverageGenerator()
            generate_trajectories(
                task_rows=_task_rows()[:1],
                out_dir=out_dir,
                environment=_build_environment(),
                config=TrajectoryGenerationConfig(
                    max_steps=1,
                    n_act=1,
                    n_ver=1,
                    seed=3,
                    candidate_source="model_vllm",
                    selection_policy="task_quality",
                    pair_mining_strategy="quality_balanced",
                    tool_coverage_retry_count=1,
                ),
                model_generator_config=ModelGeneratorConfig(api_base="http://unused"),
                candidate_generator=generator,
            )

            self.assertEqual(generator.force_tool_coverage_flags, [False, True])
            branch_pools = _read_jsonl(out_dir / "branch_pools.jsonl")
            tools = {
                branch["actor_step"]["tool_action"]["tool_name"]
                for branch in branch_pools[0]["branches"]
                if branch["actor_step"]["tool_action"]
            }
            self.assertEqual(tools, {"get_neighbors", "rwr"})
            selected = next(
                branch
                for branch in branch_pools[0]["branches"]
                if branch["branch_id"] == branch_pools[0]["selected_branch_id"]
            )
            self.assertEqual(selected["actor_step"]["tool_action"]["tool_name"], "rwr")

    def test_quality_pair_single_rejected_uses_task_difficulty_bin(self) -> None:
        self.assertEqual(
            _preference_difficulty_for_rank(0, 1, task_difficulty="hard").value,
            "hard",
        )
        self.assertEqual(
            _preference_difficulty_for_rank(0, 1, task_difficulty="complete").value,
            "medium",
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
