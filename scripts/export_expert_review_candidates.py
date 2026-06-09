#!/usr/bin/env python3
"""Export trajectory-candidate review packets for expert alignment checks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import CandidateBranch


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {path}:{row_index}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {path}:{row_index}.")
            yield row_index, payload


def _one_line(value: Any, *, max_chars: int = 700) -> str:
    text = " ".join(("" if value is None else str(value)).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "... [truncated]"


def _tool_action(branch: CandidateBranch) -> dict[str, Any] | None:
    action = branch.actor_step.tool_action
    return action.to_dict() if action is not None else None


def _observation_summary(branch: CandidateBranch) -> dict[str, Any] | None:
    observation = branch.observation
    if observation is None:
        return None
    provenance = observation.provenance
    payload = observation.payload if isinstance(observation.payload, dict) else {}
    return {
        "status": observation.status.value,
        "tool_name": provenance.get("tool_name"),
        "cache_hit": provenance.get("cache_hit"),
        "error": observation.error,
        "payload_keys": sorted(payload),
        "payload_preview": _one_line(json.dumps(payload, sort_keys=True, ensure_ascii=True), max_chars=1200),
    }


def _visible_branch(branch: CandidateBranch, *, selected: bool) -> dict[str, Any]:
    state = branch.verifier_step.updated_state
    interpretation = branch.verifier_step.updated_interpretation
    return {
        "branch_id": branch.branch_id,
        "selected_by_pipeline": selected,
        "actor_reasoning": _one_line(branch.actor_step.reasoning_text, max_chars=1200),
        "tool_action": _tool_action(branch),
        "tool_observation": _observation_summary(branch),
        "verifier_interpretation": {
            "main_evidence": interpretation.main_evidence,
            "mechanistic_claim": interpretation.mechanistic_claim,
            "uncertainty": interpretation.uncertainty,
            "next_subgoal": interpretation.next_subgoal,
        },
        "final_verifier_state": state.to_dict(),
    }


def _hidden_scoring(branch: CandidateBranch) -> dict[str, Any]:
    score = branch.local_score
    return {
        "branch_id": branch.branch_id,
        "total_score": score.total_score,
        "normalized_score": score.normalized_score,
        "schema_score": score.schema_score,
        "complex_membership_delta": score.complex_membership_delta,
        "mechanistic_label_delta": score.mechanistic_label_delta,
        "mechanism_evidence_delta": score.mechanism_evidence_delta,
        "mechanism_evidence_score": score.mechanism_evidence_score,
        "efficiency_penalty": score.efficiency_penalty,
        "score_metadata": score.score_metadata,
    }


def _score_value(branch: CandidateBranch) -> float:
    normalized = branch.local_score.normalized_score
    if isinstance(normalized, (int, float)):
        return float(normalized)
    return float(branch.local_score.total_score)


def _select_branches(
    branches: list[CandidateBranch],
    *,
    selected_branch_id: str,
    max_rejected: int,
) -> list[CandidateBranch]:
    by_id = {branch.branch_id: branch for branch in branches}
    chosen: list[CandidateBranch] = []
    selected = by_id.get(selected_branch_id)
    if selected is not None:
        chosen.append(selected)
    rejected = [
        branch
        for branch in sorted(branches, key=_score_value, reverse=True)
        if branch.branch_id != selected_branch_id
    ]
    chosen.extend(rejected[:max(0, max_rejected)])
    return chosen


def export_review_packets(
    run_dir: Path,
    *,
    sample_size: int = 30,
    seed: int = 42,
    max_rejected: int = 3,
) -> list[dict[str, Any]]:
    branch_pool_path = run_dir / "branch_pools.jsonl"
    if not branch_pool_path.exists():
        raise FileNotFoundError(f"Missing branch pools: {branch_pool_path}")

    packets: list[dict[str, Any]] = []
    for row_index, row in _read_jsonl(branch_pool_path):
        branches: list[CandidateBranch] = []
        for branch_index, branch_payload in enumerate(row.get("branches", [])):
            if not isinstance(branch_payload, dict):
                raise ValueError(f"Invalid branch at {branch_pool_path}:{row_index} branch {branch_index}.")
            try:
                branches.append(CandidateBranch.from_dict(branch_payload))
            except Exception as exc:
                raise ValueError(
                    f"Could not parse branch at {branch_pool_path}:{row_index} branch {branch_index}: {exc}"
                ) from exc
        if not branches:
            continue

        trajectory_id = str(row.get("trajectory_id"))
        step_index = row.get("step_index")
        selected_branch_id = str(row.get("selected_branch_id") or "")
        selected_branches = _select_branches(
            branches,
            selected_branch_id=selected_branch_id,
            max_rejected=max_rejected,
        )
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        review_item_id = f"{trajectory_id}.step{step_index}"
        packets.append(
            {
                "review_item_id": review_item_id,
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "source_task_id": row.get("source_task_id"),
                "task_type": row.get("task_type"),
                "difficulty": row.get("difficulty"),
                "evidence_mode": row.get("evidence_mode"),
                "selected_branch_id": selected_branch_id,
                "expert_packet": {
                    "review_item_id": review_item_id,
                    "task": {
                        "source_task_id": row.get("source_task_id"),
                        "task_type": row.get("task_type"),
                        "difficulty": row.get("difficulty"),
                        "evidence_mode": row.get("evidence_mode"),
                        "query_text": context.get("query_text"),
                        "initial_interpretation": context.get("interpretation"),
                        "initial_state": context.get("state"),
                    },
                    "candidate_branches": [
                        _visible_branch(branch, selected=branch.branch_id == selected_branch_id)
                        for branch in selected_branches
                    ],
                },
                "hidden_scoring": {
                    branch.branch_id: _hidden_scoring(branch)
                    for branch in branches
                },
            }
        )

    if sample_size > 0 and len(packets) > sample_size:
        rng = random.Random(seed)
        packets = rng.sample(packets, sample_size)
        packets.sort(key=lambda item: (str(item.get("trajectory_id")), int(item.get("step_index") or 0)))
    return packets


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export expert trajectory-candidate review packets.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of review items to sample; use 0 for all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rejected", type=int, default=3)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    packets = export_review_packets(
        args.run_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        max_rejected=args.max_rejected,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, sort_keys=True, ensure_ascii=True) + "\n")
    print(f"Wrote {len(packets)} review packets to {args.out}")


if __name__ == "__main__":
    main()
