#!/usr/bin/env python
"""Run a one-step trajectory-generation smoke test against the live model server."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import RuntimeEnvironment, SharedPrefixContext, initialize_state_from_corum_task
from scripts.generate_trajectories import (
    DEFAULT_ACTOR_RATIONALE_MAX_TOKENS,
    DEFAULT_GENERATOR_API_BASE,
    DEFAULT_GENERATOR_API_KEY_ENV,
    DEFAULT_STORE_DIR,
    STRUCTURED_OUTPUT_MAX_TOKENS,
    DEFAULT_TASKS_PATH,
    ModelGeneratorConfig,
    OpenAICompatibleCandidateGenerator,
    _actor_candidate_is_usable,
    _build_actor_step_from_model_candidate,
    _build_branch_from_model_output,
    _gene_symbol_lookup,
    _normalize_branch_pool,
    _score_branch,
    _select_best_branch,
    _unique,
    _verifier_candidate_is_usable,
)


_TRAJECTORY_SEED_SUFFIX_RE = re.compile(r"^(?P<task_id>.+)\.seed\d+$")
_REPEATED_TOKEN_RE = re.compile(r"\b([A-Za-z#]{1,16})(?:\s+\1){7,}\b")
_REPEATED_CHAR_RE = re.compile(r"([^\s])\1{23,}")


def _normalize_task_id(task_id: str) -> str:
    match = _TRAJECTORY_SEED_SUFFIX_RE.match(task_id)
    if match:
        return match.group("task_id")
    return task_id


def _load_task(task_path: Path, *, task_id: str | None, task_index: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if task_id is not None:
        normalized_task_id = _normalize_task_id(task_id)
        for row in rows:
            if row.get("task_id") == normalized_task_id:
                return row
        suggestions: list[str] = []
        for row in rows:
            candidate_task_id = row.get("task_id")
            if (
                isinstance(candidate_task_id, str)
                and candidate_task_id.startswith(normalized_task_id.split(".", 1)[0])
            ):
                suggestions.append(candidate_task_id)
            if len(suggestions) == 4:
                break
        detail = ""
        if normalized_task_id != task_id:
            detail = f" Normalized trajectory id {task_id!r} to task id {normalized_task_id!r}."
        if suggestions:
            detail += f" Example matching task ids: {', '.join(suggestions)}."
        raise ValueError(f"Task id not found: {task_id}.{detail}")
    if task_index < 0 or task_index >= len(rows):
        raise ValueError(f"task_index must be in [0, {len(rows) - 1}]")
    return rows[task_index]


def _build_environment(args: argparse.Namespace) -> RuntimeEnvironment:
    if args.store_dir is not None:
        return RuntimeEnvironment(
            store_dir=str(args.store_dir),
            compiled_library_path=str(args.compiled_library_path) if args.compiled_library_path else None,
        )
    if args.multiplex_flist is not None:
        return RuntimeEnvironment(multiplex_flist=str(args.multiplex_flist))
    raise ValueError("Provide either --store-dir or --multiplex-flist.")


def _extract_visible_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        return ""
    stripped = raw_text.strip()
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        if isinstance(payload.get("text"), str):
            return payload["text"]
    return stripped


def _classify_corrupted_output(raw_text: str) -> str | None:
    visible_text = _extract_visible_text(raw_text)
    if not visible_text.strip():
        return "whitespace_only_output"
    if _REPEATED_CHAR_RE.search(visible_text):
        return "repeated_character_collapse"
    if _REPEATED_TOKEN_RE.search(visible_text):
        return "repeated_token_collapse"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test one full actor/verifier step.")
    parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--compiled-library-path", type=Path, default=None)
    parser.add_argument("--multiplex-flist", type=Path, default=None)
    parser.add_argument("--generator-api-base", type=str, default=DEFAULT_GENERATOR_API_BASE)
    parser.add_argument(
        "--generator-api-mode",
        choices=("auto", "chat_completions", "responses", "completions"),
        default="auto",
    )
    parser.add_argument("--generator-model", type=str, default=None)
    parser.add_argument("--generator-api-key", type=str, default=None)
    parser.add_argument("--generator-api-key-env", type=str, default=DEFAULT_GENERATOR_API_KEY_ENV)
    parser.add_argument("--generator-temperature", type=float, default=0.2)
    parser.add_argument("--generator-top-p", type=float, default=0.95)
    parser.add_argument(
        "--generator-max-completion-tokens",
        type=int,
        default=STRUCTURED_OUTPUT_MAX_TOKENS,
    )
    parser.add_argument(
        "--generator-actor-rationale-max-completion-tokens",
        type=int,
        default=DEFAULT_ACTOR_RATIONALE_MAX_TOKENS,
    )
    parser.add_argument(
        "--generator-reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--generator-timeout-seconds", type=int, default=600)
    parser.add_argument("--n-act", type=int, default=1)
    parser.add_argument("--n-ver", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        task_row = _load_task(args.tasks_path, task_id=args.task_id, task_index=args.task_index)
        environment = _build_environment(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    interpretation, state = initialize_state_from_corum_task(task_row, max_budget=1)
    context = SharedPrefixContext(
        query_text=task_row["query_text"],
        user_evidence=task_row["visible_inputs"],
        interpretation=interpretation,
        state=state,
        source_task_id=task_row["task_id"],
    )
    generator = OpenAICompatibleCandidateGenerator(
        ModelGeneratorConfig(
            api_base=args.generator_api_base,
            api_mode=args.generator_api_mode,
            model_name=args.generator_model,
            api_key=args.generator_api_key,
            api_key_env=args.generator_api_key_env,
            request_timeout_seconds=args.generator_timeout_seconds,
            temperature=args.generator_temperature,
            top_p=args.generator_top_p,
            max_completion_tokens=args.generator_max_completion_tokens,
            actor_rationale_max_completion_tokens=args.generator_actor_rationale_max_completion_tokens,
            reasoning_effort=args.generator_reasoning_effort,
        )
    )
    prior_actions = []
    symbol_lookup = _gene_symbol_lookup(task_row)

    try:
        actor_candidates = generator.generate_actor_candidates(
            context,
            task_row=task_row,
            step_index=args.step_index,
            n_act=args.n_act,
            seed=args.seed,
        )
    except requests.exceptions.RequestException as error:
        raise SystemExit(
            "Failed to reach the generator API at "
            f"{args.generator_api_base}. Check that the vLLM job is still running "
            f"and that the served model endpoint is reachable. Original error: {error}"
        ) from error

    print(
        json.dumps(
            {
                "task_id": task_row["task_id"],
                "resolved_api_mode": generator.api_mode,
                "model_name": generator.model_name,
                "actor_candidate_count": len(actor_candidates),
                "runtime": environment.describe(),
            },
            indent=2,
            sort_keys=True,
        )
    )

    actor_records: list[dict[str, Any]] = []
    verifier_records: list[dict[str, Any]] = []
    branches = []
    for actor_index, actor_candidate in enumerate(actor_candidates):
        actor_step, actor_errors = _build_actor_step_from_model_candidate(
            actor_candidate,
            trajectory_id=f"{task_row['task_id']}.seed{args.seed}",
            step_index=args.step_index,
            actor_index=actor_index,
        )
        actor_generation_errors = _unique(actor_errors + list(actor_candidate.get("generator_errors", [])))
        actor_corruption = _classify_corrupted_output(actor_candidate.get("raw_text", ""))
        if actor_corruption is not None:
            actor_generation_errors = _unique(actor_generation_errors + [actor_corruption])
        actor_usable = _actor_candidate_is_usable(actor_step, actor_generation_errors)
        actor_record = {
            "phase": "actor",
            "candidate_index": actor_index,
            "usable": actor_usable,
            "reasoning_text": actor_step.reasoning_text,
            "tool_action": actor_step.tool_action.to_dict() if actor_step.tool_action is not None else None,
            "generator_errors": actor_generation_errors,
            "raw_text": actor_candidate.get("raw_text", ""),
        }
        actor_records.append(actor_record)
        print(json.dumps(actor_record, indent=2, sort_keys=True))

        if not actor_usable:
            continue

        observation = None
        if actor_step.tool_action is not None:
            observation = environment.execute(
                actor_step.tool_action,
                state=state,
                prior_actions=prior_actions,
            )

        verifier_candidates = generator.generate_verifier_candidates(
            context,
            task_row=task_row,
            actor_candidate=actor_candidate,
            actor_step=actor_step,
            observation=observation,
            step_index=args.step_index,
            n_ver=args.n_ver,
            seed=args.seed + (actor_index * 100),
        )
        for verifier_index, verifier_candidate in enumerate(verifier_candidates):
            branch_generation_errors = _unique(
                actor_generation_errors + list(verifier_candidate.get("generator_errors", []))
            )
            verifier_corruption = _classify_corrupted_output(verifier_candidate.get("raw_text", ""))
            if verifier_corruption is not None:
                branch_generation_errors = _unique(branch_generation_errors + [verifier_corruption])
            verifier_usable = _verifier_candidate_is_usable(verifier_candidate, branch_generation_errors)
            verifier_record = {
                "phase": "verifier",
                "actor_index": actor_index,
                "verifier_index": verifier_index,
                "usable": verifier_usable,
                "observation": observation.to_dict() if observation is not None else None,
                "generator_errors": branch_generation_errors,
                "payload": verifier_candidate.get("payload", {}),
                "raw_text": verifier_candidate.get("raw_text", ""),
            }
            verifier_records.append(verifier_record)
            print(json.dumps(verifier_record, indent=2, sort_keys=True))

            if not verifier_usable:
                continue

            branch = _build_branch_from_model_output(
                task_row,
                state,
                actor_step,
                observation,
                verifier_candidate,
                branch_id=(
                    f"{task_row['task_id']}.seed{args.seed}.step{args.step_index}."
                    f"a{actor_index}.v{verifier_index}"
                ),
                step_index=args.step_index,
                symbol_lookup=symbol_lookup,
                generator_errors=actor_generation_errors,
            )
            branch.metadata["raw_actor_response"] = actor_candidate.get("raw_text", "")
            branch = _score_branch(
                task_row,
                state,
                branch,
                step_index=args.step_index,
                max_steps=1,
                prior_actions=prior_actions,
                environment=environment,
            )
            branches.append(branch)

    if not branches:
        failure_summary = {
            "error": "smoke_test_no_usable_branches",
            "task_id": task_row["task_id"],
            "resolved_api_mode": generator.api_mode,
            "model_name": generator.model_name,
            "actor_candidates": actor_records,
            "verifier_candidates": verifier_records,
        }
        print(json.dumps(failure_summary, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)

    _normalize_branch_pool(branches)
    selected_branch = _select_best_branch(branches)
    print(
        json.dumps(
            {
                "selected_branch_id": selected_branch.branch_id,
                "branch_count": len(branches),
                "selected_branch": selected_branch.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
