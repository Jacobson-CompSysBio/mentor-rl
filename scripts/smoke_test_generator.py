#!/usr/bin/env python
"""Run a fast actor-generation smoke test against the live model server."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import ActorStep, SharedPrefixContext, ToolAction, initialize_state_from_corum_task
from scripts.generate_trajectories import (
    DEFAULT_ACTOR_RATIONALE_MAX_TOKENS,
    DEFAULT_GENERATOR_API_BASE,
    DEFAULT_GENERATOR_API_KEY_ENV,
    DEFAULT_TASKS_PATH,
    ModelGeneratorConfig,
    OpenAICompatibleCandidateGenerator,
    _actor_candidate_is_usable,
)


_TRAJECTORY_SEED_SUFFIX_RE = re.compile(r"^(?P<task_id>.+)\.seed\d+$")


def _normalize_task_id(task_id: str) -> str:
    match = _TRAJECTORY_SEED_SUFFIX_RE.match(task_id)
    if match:
        return match.group("task_id")
    return task_id


def _load_task(task_path: Path, *, task_id: str | None, task_index: int) -> dict:
    rows: list[dict] = []
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test one actor-generation request.")
    parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--generator-api-base", type=str, default=DEFAULT_GENERATOR_API_BASE)
    parser.add_argument(
        "--generator-api-mode",
        choices=("auto", "chat_completions", "responses"),
        default="auto",
    )
    parser.add_argument("--generator-model", type=str, default=None)
    parser.add_argument("--generator-api-key", type=str, default=None)
    parser.add_argument("--generator-api-key-env", type=str, default=DEFAULT_GENERATOR_API_KEY_ENV)
    parser.add_argument("--generator-temperature", type=float, default=0.2)
    parser.add_argument("--generator-top-p", type=float, default=0.95)
    parser.add_argument("--generator-max-completion-tokens", type=int, default=768)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        task_row = _load_task(args.tasks_path, task_id=args.task_id, task_index=args.task_index)
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
    try:
        candidates = generator.generate_actor_candidates(
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
                "task_type": task_row["task_type"],
                "resolved_api_mode": generator.api_mode,
                "model_name": generator.model_name,
                "candidate_count": len(candidates),
            },
            indent=2,
            sort_keys=True,
        )
    )
    usable_count = 0
    rendered_candidates: list[dict] = []
    for index, candidate in enumerate(candidates):
        tool_action = candidate.get("tool_action")
        actor_step = ActorStep(
            reasoning_text=candidate.get("reasoning_text", ""),
            tool_action=(
                ToolAction(
                    tool_name=tool_action["tool_name"],
                    arguments=tool_action["arguments"],
                    call_id=f"smoke_{index}",
                )
                if isinstance(tool_action, dict)
                else None
            ),
        )
        usable = _actor_candidate_is_usable(actor_step, candidate.get("generator_errors", []))
        usable_count += int(usable)
        rendered_candidate = {
            "candidate_index": index,
            "usable": usable,
            "reasoning_text": candidate.get("reasoning_text"),
            "tool_action": candidate.get("tool_action"),
            "generator_errors": candidate.get("generator_errors", []),
            "raw_text": candidate.get("raw_text", ""),
        }
        rendered_candidates.append(rendered_candidate)
        print(
            json.dumps(
                rendered_candidate,
                indent=2,
                sort_keys=True,
            )
        )
    if usable_count == 0:
        failure_summary = {
            "error": "smoke_test_no_usable_candidates",
            "task_id": task_row["task_id"],
            "resolved_api_mode": generator.api_mode,
            "model_name": generator.model_name,
            "candidate_count": len(candidates),
            "generator_errors": [
                candidate.get("generator_errors", [])
                for candidate in rendered_candidates
            ],
            "raw_text_preview": [
                candidate.get("raw_text", "")[:500]
                for candidate in rendered_candidates
            ],
        }
        print(json.dumps(failure_summary, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
