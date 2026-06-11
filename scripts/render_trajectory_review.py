#!/usr/bin/env python3
"""Render selected trajectory paths into human-reviewable Markdown.

The trajectory generator writes detailed JSONL artifacts that are useful for
training but hard to inspect manually. This script reads the selected
trajectory turns plus final summaries and produces a compact Markdown report
with a Mermaid graph for each trajectory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import CandidateBranch, GeneGroup, MechanisticLabel, ToolObservation, TrajectoryTurn


DEFAULT_TEXT_CHARS = 900
DEFAULT_SAMPLE_SIZE = 0
DEFAULT_MAX_UNSELECTED_PER_STEP = 3


def _truncate(value: Any, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "... [truncated]"


def _one_line(value: Any, *, max_chars: int = 240) -> str:
    return " ".join(_truncate(value, max_chars=max_chars).split())


def _json_inline(value: Any, *, max_chars: int = 600) -> str:
    return _truncate(json.dumps(value, sort_keys=True, ensure_ascii=True), max_chars=max_chars)


def _html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_float(value: Any, *, digits: int = 3, default: str = "n/a") -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return default


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{row_index}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {path}:{row_index}, got {type(payload).__name__}.")
            yield row_index, payload


def load_turns(path: Path) -> dict[str, list[TrajectoryTurn]]:
    """Load selected trajectory turns, grouped by trajectory id."""

    turns_by_trajectory: dict[str, list[TrajectoryTurn]] = defaultdict(list)
    for row_index, payload in _read_jsonl(path):
        try:
            turn = TrajectoryTurn.from_dict(payload)
        except Exception as exc:
            raise ValueError(f"Failed to parse trajectory turn at {path}:{row_index}: {exc}") from exc
        if turn.selected:
            turns_by_trajectory[turn.trajectory_id].append(turn)
    for turns in turns_by_trajectory.values():
        turns.sort(key=lambda item: item.step_index)
    return dict(turns_by_trajectory)


def load_summaries(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load final summaries keyed by trajectory id."""

    if path is None or not path.exists():
        return {}
    summaries: dict[str, dict[str, Any]] = {}
    for row_index, payload in _read_jsonl(path):
        trajectory_id = payload.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"Missing trajectory_id in {path}:{row_index}.")
        summaries[trajectory_id] = payload
    return summaries


def load_branch_pools(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    """Load branch pools keyed by trajectory id and step index."""

    if path is None or not path.exists():
        return {}
    pools: dict[tuple[str, int], dict[str, Any]] = {}
    for row_index, payload in _read_jsonl(path):
        trajectory_id = payload.get("trajectory_id")
        step_index = payload.get("step_index")
        if not isinstance(trajectory_id, str) or not isinstance(step_index, int):
            raise ValueError(f"Missing trajectory_id/step_index in {path}:{row_index}.")
        branches = []
        for branch_index, branch_payload in enumerate(payload.get("branches", [])):
            try:
                branches.append(CandidateBranch.from_dict(branch_payload))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse branch at {path}:{row_index} branch {branch_index}: {exc}"
                ) from exc
        pools[(trajectory_id, step_index)] = {
            "selected_branch_id": payload.get("selected_branch_id"),
            "branches": branches,
        }
    return pools


def load_task_rows(path: Path | None, source_task_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load only task rows needed for the rendered trajectories."""

    if path is None or not path.exists() or not source_task_ids:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{row_index}: {exc}") from exc
            task_id = payload.get("task_id")
            if task_id in source_task_ids:
                rows[task_id] = payload
                if len(rows) == len(source_task_ids):
                    break
    return rows


def infer_tasks_path(run_dir: Path | None) -> Path | None:
    """Infer the input task JSONL from run_freeze.json when available."""

    if run_dir is None:
        return None
    freeze_path = run_dir / "run_freeze.json"
    if not freeze_path.exists():
        return None
    try:
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    task_selection = payload.get("task_selection", {})
    if not isinstance(task_selection, dict):
        return None
    tasks_path = task_selection.get("tasks_path")
    if isinstance(tasks_path, str) and tasks_path:
        return Path(tasks_path)
    return None


def _safe_mermaid_label(value: Any, *, max_chars: int = 110) -> str:
    text = _one_line(value, max_chars=max_chars)
    text = text.replace("\\", "\\\\").replace('"', "'")
    text = text.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    return text


def _safe_anchor(value: str) -> str:
    anchor = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return anchor or "trajectory"


def _tool_name(branch: CandidateBranch) -> str:
    action = branch.actor_step.tool_action
    if action is None:
        return "no_tool"
    return action.tool_name


def _tool_action_summary(branch: CandidateBranch) -> str:
    action = branch.actor_step.tool_action
    if action is None:
        return "`no_tool`"
    return f"`{action.tool_name}` `{_json_inline(action.arguments, max_chars=700)}`"


def _branch_claim(branch: CandidateBranch, *, max_chars: int = 120) -> str:
    interpretation = branch.verifier_step.updated_interpretation
    return _one_line(interpretation.mechanistic_claim or interpretation.main_evidence, max_chars=max_chars)


def _branch_observation_status(branch: CandidateBranch) -> str:
    if branch.observation is None:
        return "none"
    return branch.observation.status.value


def _score_summary(branch: CandidateBranch) -> str:
    score = branch.local_score
    return (
        f"total `{score.total_score:.3f}`, normalized `{float(score.normalized_score or 0.0):.3f}`, "
        f"schema `{score.schema_score:.3f}`, gene-delta `{score.complex_membership_delta:.3f}`, "
        f"mechanism-delta `{score.mechanistic_label_delta:.3f}`, "
        f"mechanism-evidence `{score.mechanism_evidence_score:.3f}`"
    )


def _group_summary(groups: list[GeneGroup], *, max_chars: int) -> str:
    if not groups:
        return "`[]`"
    parts = []
    for group in groups:
        genes = ", ".join(group.gene_ids)
        parts.append(f"{group.group_id} ({len(group.gene_ids)}): {genes}")
    return _truncate("; ".join(parts), max_chars=max_chars)


def _label_summary(labels: list[MechanisticLabel], *, max_chars: int) -> str:
    if not labels:
        return "`[]`"
    parts = []
    for label in labels:
        label_id = f" ({label.label_id})" if label.label_id else ""
        evidence = f" evidence={','.join(label.evidence_ids)}" if label.evidence_ids else ""
        parts.append(f"{label.label_source.value}:{label.label_name}{label_id}{evidence}")
    return _truncate("; ".join(parts), max_chars=max_chars)


def _top_enrichment_terms(results: list[Any], *, max_terms: int = 5) -> list[str]:
    terms: list[str] = []
    for result in results[:max_terms]:
        if not isinstance(result, dict):
            continue
        term_id = result.get("native") or result.get("term_id") or result.get("id")
        name = result.get("name") or result.get("description")
        p_value = result.get("p_value") or result.get("pval") or result.get("fdr")
        label = str(name or term_id or "term")
        if term_id and term_id not in label:
            label = f"{term_id} {label}"
        if p_value is not None:
            label = f"{label} p={p_value}"
        terms.append(_one_line(label, max_chars=160))
    return terms


def _observation_summary(observation: ToolObservation | None, *, max_chars: int) -> str:
    if observation is None:
        return "No tool observation."
    payload = observation.payload or {}
    provenance = observation.provenance or {}
    tool_name = provenance.get("tool_name") or "unknown"
    pieces = [f"status `{observation.status.value}`", f"tool `{tool_name}`"]
    source = provenance.get("source") or ("cache" if provenance.get("cache_hit") else None)
    if source:
        pieces.append(f"source `{source}`")
    if observation.error:
        pieces.append(f"error `{_one_line(observation.error, max_chars=240)}`")

    if "combined_edge_count" in payload:
        pieces.append(f"edges `{payload.get('combined_edge_count')}`")
    if "unique_neighbor_count" in payload:
        pieces.append(f"neighbors `{payload.get('unique_neighbor_count')}`")
    if "hop_count" in payload:
        pieces.append(f"hops `{payload.get('hop_count')}`")
    if "path_gene_ids" in payload:
        pieces.append(f"path `{_json_inline(payload.get('path_gene_ids'), max_chars=360)}`")
    if "top_genes" in payload:
        pieces.append(f"top genes `{_json_inline(payload.get('top_genes')[:8], max_chars=500)}`")
    if "results" in payload and isinstance(payload["results"], list):
        terms = _top_enrichment_terms(payload["results"])
        if terms:
            pieces.append(f"terms `{_json_inline(terms, max_chars=600)}`")
        else:
            pieces.append(f"results `{_json_inline(payload['results'][:3], max_chars=600)}`")
    if "hits" in payload and isinstance(payload["hits"], list):
        pieces.append(f"hits `{_json_inline(payload['hits'][:3], max_chars=600)}`")
    if "query_gene_ids" in payload:
        genes = payload.get("query_gene_ids")
        if isinstance(genes, list):
            pieces.append(f"queried `{len(genes)}` genes")
    return _truncate("; ".join(pieces), max_chars=max_chars)


def _summary_value(summary: dict[str, Any], key: str, default: str = "unknown") -> str:
    value = summary.get(key)
    if value is None:
        return default
    return str(value)


def _top_unselected_branches(
    branch_pool: dict[str, Any] | None,
    *,
    selected_branch_id: str,
    max_unselected: int,
) -> list[CandidateBranch]:
    if not branch_pool or max_unselected <= 0:
        return []
    branches = [
        branch
        for branch in branch_pool.get("branches", [])
        if isinstance(branch, CandidateBranch) and branch.branch_id != selected_branch_id
    ]
    branches.sort(
        key=lambda branch: (
            float(branch.local_score.normalized_score or 0.0),
            branch.local_score.total_score,
            branch.local_score.mechanism_evidence_score,
            branch.branch_id,
        ),
        reverse=True,
    )
    return branches[:max_unselected]


def _trajectory_title(trajectory_id: str, summary: dict[str, Any] | None) -> str:
    if not summary:
        return trajectory_id
    return (
        f"{trajectory_id} "
        f"({summary.get('task_type', 'unknown')}/{summary.get('evidence_mode', 'unknown')}/"
        f"{summary.get('difficulty', 'unknown')})"
    )


def _is_non_none_task(summary: dict[str, Any] | None) -> bool:
    return bool(summary and summary.get("task_type") in {"explanation", "recovery", "refinement"})


def _branch_visual_class(
    branch: CandidateBranch,
    *,
    summary: dict[str, Any] | None,
    selected: bool,
) -> str:
    """Return the Mermaid/CSS class that should dominate this branch node."""

    relationship = branch.verifier_step.updated_state.relationship_status.value
    if _is_non_none_task(summary) and relationship == "insufficient_support":
        return "lowQuality"
    if (
        summary
        and summary.get("task_type") == "explanation"
        and branch.local_score.complex_membership_delta < -0.25
    ):
        return "taskMismatch"
    return "selected" if selected else "unselected"


def _final_visual_class(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "final"
    final_state = summary.get("final_state", {})
    relationship = final_state.get("relationship_status") if isinstance(final_state, dict) else None
    gene_score = summary.get("terminal_absolute_complex_score")
    if _is_non_none_task(summary) and relationship == "insufficient_support":
        return "lowQuality"
    if isinstance(gene_score, (int, float)):
        if gene_score < 0.25 and _is_non_none_task(summary):
            return "lowQuality"
        if summary.get("task_type") == "explanation" and gene_score < 0.8:
            return "taskMismatch"
    return "final"


def _final_warning(summary: dict[str, Any] | None) -> str | None:
    if not summary:
        return None
    final_state = summary.get("final_state", {})
    relationship = final_state.get("relationship_status") if isinstance(final_state, dict) else "unknown"
    gene_score = summary.get("terminal_absolute_complex_score")
    if _is_non_none_task(summary) and relationship == "insufficient_support":
        return "Low-quality final state: a non-none task ended as insufficient support."
    if summary.get("task_type") == "explanation" and isinstance(gene_score, (int, float)) and gene_score < 0.8:
        return (
            "Task mismatch risk: explanation task should preserve the input module, "
            f"but final gene-set score is {gene_score:.3f}."
        )
    if _is_non_none_task(summary) and isinstance(gene_score, (int, float)) and gene_score < 0.25:
        return f"Low-quality final state: final gene-set score is {gene_score:.3f}."
    return None


def _task_filters_match(trajectory_id: str, summary: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if args.trajectory_id and trajectory_id not in set(args.trajectory_id):
        return False
    if args.source_task_id:
        source = summary.get("source_task_id", "") if summary else trajectory_id
        if not any(fragment in source for fragment in args.source_task_id):
            return False
    if summary:
        if args.task_type and summary.get("task_type") not in set(args.task_type):
            return False
        if args.evidence_mode and summary.get("evidence_mode") not in set(args.evidence_mode):
            return False
        if args.difficulty and summary.get("difficulty") not in set(args.difficulty):
            return False
    return True


def _select_trajectory_ids(
    turns_by_trajectory: dict[str, list[TrajectoryTurn]],
    summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    trajectory_ids = [
        trajectory_id
        for trajectory_id in sorted(turns_by_trajectory)
        if _task_filters_match(trajectory_id, summaries.get(trajectory_id), args)
    ]
    if args.sample_size <= 0 or args.sample_size >= len(trajectory_ids):
        return trajectory_ids
    rng = random.Random(args.seed)
    selected = list(trajectory_ids)
    rng.shuffle(selected)
    return sorted(selected[: args.sample_size])


def _mermaid_graph_source(
    trajectory_id: str,
    turns: list[TrajectoryTurn],
    summary: dict[str, Any] | None,
    *,
    branch_pools: dict[tuple[str, int], dict[str, Any]] | None = None,
    max_unselected_per_step: int = 0,
) -> str:
    prefix = "n" + re.sub(r"[^A-Za-z0-9_]", "_", trajectory_id)
    lines = [
        "flowchart TD",
        "  classDef selected fill:#dbeafe,stroke:#2563eb,stroke-width:2px;",
        "  classDef unselected fill:#f8fafc,stroke:#94a3b8,stroke-width:1px,stroke-dasharray:4 3;",
        "  classDef final fill:#dcfce7,stroke:#16a34a,stroke-width:2px;",
        "  classDef taskMismatch fill:#ffedd5,stroke:#f97316,stroke-width:2px;",
        "  classDef lowQuality fill:#fee2e2,stroke:#dc2626,stroke-width:2px;",
    ]
    start_label = _safe_mermaid_label("start")
    lines.append(f'  {prefix}_start["{start_label}"]')
    lines.append(f"  class {prefix}_start selected;")
    previous_node = f"{prefix}_start"
    for turn in turns:
        branch = turn.branch
        state = branch.verifier_step.updated_state
        score = branch.local_score
        label = (
            f"Step {turn.step_index} | Selected<br/>"
            f"Tool: {_tool_name(branch)} ({_branch_observation_status(branch)})<br/>"
            f"State: {state.relationship_status.value}<br/>"
            f"Score: {score.total_score:.2f} | Mechanism: {score.mechanism_evidence_score:.2f}<br/>"
            f"{_safe_mermaid_label(_branch_claim(branch, max_chars=80), max_chars=80)}"
        )
        node_id = f"{prefix}_s{turn.step_index}"
        lines.append(f'  {node_id}["{label}"]')
        lines.append(f"  {previous_node} --> {node_id}")
        lines.append(f"  class {node_id} {_branch_visual_class(branch, summary=summary, selected=True)};")

        branch_pool = branch_pools.get((trajectory_id, turn.step_index)) if branch_pools else None
        for alt_index, alt_branch in enumerate(
            _top_unselected_branches(
                branch_pool,
                selected_branch_id=branch.branch_id,
                max_unselected=max_unselected_per_step,
            ),
            start=1,
        ):
            alt_state = alt_branch.verifier_step.updated_state
            alt_score = alt_branch.local_score
            alt_label = (
                f"Step {turn.step_index} | Alt {alt_index}<br/>"
                f"Tool: {_tool_name(alt_branch)} ({_branch_observation_status(alt_branch)})<br/>"
                f"State: {alt_state.relationship_status.value}<br/>"
                f"Score: {alt_score.total_score:.2f} | Mechanism: {alt_score.mechanism_evidence_score:.2f}<br/>"
                f"{_safe_mermaid_label(_branch_claim(alt_branch, max_chars=75), max_chars=75)}"
            )
            alt_node_id = f"{prefix}_s{turn.step_index}_alt{alt_index}"
            lines.append(f'  {alt_node_id}["{alt_label}"]')
            lines.append(f"  {previous_node} -.-> {alt_node_id}")
            lines.append(
                f"  class {alt_node_id} "
                f"{_branch_visual_class(alt_branch, summary=summary, selected=False)};"
            )
        previous_node = node_id
    if summary:
        final_status = summary.get("final_state", {}).get("relationship_status", "unknown")
        final_reward = summary.get("terminal_reward")
        final_label = f"Final<br/>State: {final_status}"
        if isinstance(final_reward, (int, float)):
            final_label += f"<br/>Reward: {final_reward:.2f}"
        final_node = f"{prefix}_final"
        lines.append(f'  {final_node}["{_safe_mermaid_label(final_label, max_chars=140)}"]')
        lines.append(f"  {previous_node} --> {final_node}")
        lines.append(f"  class {final_node} {_final_visual_class(summary)};")
    return "\n".join(lines)


def _mermaid_graph(
    trajectory_id: str,
    turns: list[TrajectoryTurn],
    summary: dict[str, Any] | None,
    *,
    branch_pools: dict[tuple[str, int], dict[str, Any]] | None = None,
    max_unselected_per_step: int = 0,
) -> str:
    lines = [
        "```mermaid",
        _mermaid_graph_source(
            trajectory_id,
            turns,
            summary,
            branch_pools=branch_pools,
            max_unselected_per_step=max_unselected_per_step,
        ),
        "```",
    ]
    return "\n".join(lines)


def _trajectory_overview(
    trajectory_id: str,
    turns: list[TrajectoryTurn],
    summary: dict[str, Any] | None,
    *,
    max_text_chars: int,
) -> list[str]:
    anchor = _safe_anchor(trajectory_id).lower()
    lines = [f'<a id="{anchor}"></a>', "", f"## `{trajectory_id}`", ""]
    if summary:
        final_state = summary.get("final_state", {}) if isinstance(summary.get("final_state"), dict) else {}
        final_interpretation = (
            summary.get("final_interpretation", {})
            if isinstance(summary.get("final_interpretation"), dict)
            else {}
        )
        lines.extend(
            [
                f"- Task/evidence/difficulty: `{_summary_value(summary, 'task_type')}` / "
                f"`{_summary_value(summary, 'evidence_mode')}` / `{_summary_value(summary, 'difficulty')}`",
                f"- Source task: `{_summary_value(summary, 'source_task_id')}`",
                f"- Steps/findings: `{summary.get('step_count', len(turns))}` / `{summary.get('finding_count', len(turns))}`",
                f"- Final status: `{final_state.get('relationship_status', 'unknown')}`; "
                f"termination: `{final_state.get('termination_reason', 'unknown')}`",
                f"- Terminal reward: `{float(summary.get('terminal_reward', 0.0)):.3f}`; "
                f"gene score `{float(summary.get('terminal_absolute_complex_score', 0.0)):.3f}`; "
                f"mechanism-evidence `{float(summary.get('terminal_mechanism_evidence_score', 0.0)):.3f}`",
                f"- Final predicted groups: {_group_summary([GeneGroup.from_dict(item) for item in final_state.get('predicted_groups', [])], max_chars=max_text_chars)}",
                f"- Final mechanistic labels: {_label_summary([MechanisticLabel.from_dict(item) for item in final_state.get('mechanistic_labels', [])], max_chars=max_text_chars)}",
                "",
                "**Final Interpretation**",
                "",
                f"- Claim: {_truncate(final_interpretation.get('mechanistic_claim', ''), max_chars=max_text_chars) or '_empty_'}",
                f"- Evidence: {_truncate(final_interpretation.get('main_evidence', ''), max_chars=max_text_chars) or '_empty_'}",
                f"- Uncertainty: {_truncate(final_interpretation.get('uncertainty', ''), max_chars=max_text_chars) or '_empty_'}",
                f"- Next subgoal: {_truncate(final_interpretation.get('next_subgoal', ''), max_chars=max_text_chars) or '_empty_'}",
                "",
            ]
        )
    else:
        lines.extend(["- Final summary: `missing`", ""])
    return lines


def _branch_compact_summary(branch: CandidateBranch, *, max_text_chars: int) -> str:
    state = branch.verifier_step.updated_state
    return (
        f"`{branch.branch_id}` | `{_tool_name(branch)}`/{_branch_observation_status(branch)} | "
        f"score `{branch.local_score.total_score:.3f}` | mech `{branch.local_score.mechanism_evidence_score:.3f}` | "
        f"`{state.relationship_status.value}` | {_truncate(_branch_claim(branch, max_chars=max_text_chars), max_chars=max_text_chars)}"
    )


def _predicted_gene_count(branch: CandidateBranch) -> int:
    gene_ids: set[str] = set()
    for group in branch.verifier_step.updated_state.predicted_groups:
        gene_ids.update(group.gene_ids)
    return len(gene_ids)


def _labels_plain(labels: list[MechanisticLabel], *, max_chars: int) -> str:
    summary = _label_summary(labels, max_chars=max_chars)
    return "[]" if summary == "`[]`" else summary


def _gene_list_preview(genes: Any, *, max_items: int = 10) -> str:
    if not isinstance(genes, list):
        return "none"
    values = [str(gene) for gene in genes]
    if not values:
        return "none"
    preview = ", ".join(values[:max_items])
    if len(values) > max_items:
        preview += f", ... (+{len(values) - max_items})"
    return preview


def _final_predicted_gene_ids(summary: dict[str, Any] | None) -> list[str]:
    final_state = _html_final_state(summary)
    gene_ids: set[str] = set()
    for group in final_state.get("predicted_groups", []):
        if isinstance(group, dict):
            gene_ids.update(str(gene_id) for gene_id in group.get("gene_ids", []) if isinstance(gene_id, str))
    return sorted(gene_ids)


def _alignment_metrics(
    *,
    summary: dict[str, Any] | None,
    task_row: dict[str, Any] | None,
) -> dict[str, Any]:
    predicted = set(_final_predicted_gene_ids(summary))
    hidden = task_row.get("hidden_target", {}) if task_row else {}
    target_genes = hidden.get("target_gene_ids") if isinstance(hidden, dict) else None
    target_status = hidden.get("relationship_status") if isinstance(hidden, dict) else None
    if isinstance(target_genes, list):
        target = {str(gene_id) for gene_id in target_genes}
        overlap = predicted & target
        precision = len(overlap) / len(predicted) if predicted else 0.0
        recall = len(overlap) / len(target) if target else 0.0
        union = predicted | target
        jaccard = len(overlap) / len(union) if union else 1.0
        return {
            "target_status": target_status,
            "target_count": len(target),
            "predicted_count": len(predicted),
            "overlap_count": len(overlap),
            "precision": precision,
            "recall": recall,
            "jaccard": jaccard,
            "target_preview": _gene_list_preview(target_genes),
            "predicted_preview": _gene_list_preview(sorted(predicted)),
        }
    expected_none = target_genes is None
    abstained = not predicted
    return {
        "target_status": target_status,
        "target_count": None,
        "predicted_count": len(predicted),
        "overlap_count": None,
        "precision": 1.0 if expected_none and abstained else 0.0,
        "recall": 1.0 if expected_none and abstained else 0.0,
        "jaccard": 1.0 if expected_none and abstained else 0.0,
        "target_preview": "no functional group expected",
        "predicted_preview": _gene_list_preview(sorted(predicted)),
    }


def _step_block(
    turn: TrajectoryTurn,
    *,
    max_text_chars: int,
    branch_pool: dict[str, Any] | None = None,
    max_unselected_per_step: int = 0,
) -> list[str]:
    branch = turn.branch
    verifier = branch.verifier_step
    interpretation = verifier.updated_interpretation
    state = verifier.updated_state
    directive = branch.metadata.get("actor_sampling_directive")
    directive_name = directive.get("directive_name") if isinstance(directive, dict) else None
    lines = [
        f"### Step `{turn.step_index}`",
        "",
        f"- Branch: `{branch.branch_id}`",
        f"- Tool action: {_tool_action_summary(branch)}",
        f"- Observation: {_observation_summary(branch.observation, max_chars=max_text_chars)}",
        f"- Score: {_score_summary(branch)}",
        f"- Relationship: `{state.relationship_status.value}`; continuation: `{state.continuation_state.value}`",
        f"- Sampling directive: `{directive_name or 'none'}`",
        f"- Predicted groups: {_group_summary(state.predicted_groups, max_chars=max_text_chars)}",
        f"- Mechanistic labels: {_label_summary(state.mechanistic_labels, max_chars=max_text_chars)}",
        f"- Finding: {_truncate(turn.finding_text, max_chars=max_text_chars) or '_empty_'}",
        "",
        "**Actor Reasoning**",
        "",
        _truncate(branch.actor_step.reasoning_text, max_chars=max_text_chars) or "_empty_",
        "",
        "**Verifier Update**",
        "",
        f"- Claim: {_truncate(interpretation.mechanistic_claim, max_chars=max_text_chars) or '_empty_'}",
        f"- Evidence: {_truncate(interpretation.main_evidence, max_chars=max_text_chars) or '_empty_'}",
        f"- Uncertainty: {_truncate(interpretation.uncertainty, max_chars=max_text_chars) or '_empty_'}",
        f"- Next subgoal: {_truncate(interpretation.next_subgoal, max_chars=max_text_chars) or '_empty_'}",
        f"- Verifier notes: {_truncate(verifier.verifier_notes, max_chars=max_text_chars) or '_empty_'}",
        "",
    ]
    unselected = _top_unselected_branches(
        branch_pool,
        selected_branch_id=branch.branch_id,
        max_unselected=max_unselected_per_step,
    )
    if unselected:
        lines.extend(["**Top Unselected Alternatives**", ""])
        for index, alt_branch in enumerate(unselected, start=1):
            lines.append(f"{index}. {_branch_compact_summary(alt_branch, max_text_chars=max_text_chars)}")
        lines.append("")
    return lines


def render_markdown(
    *,
    turns_by_trajectory: dict[str, list[TrajectoryTurn]],
    summaries: dict[str, dict[str, Any]],
    trajectory_ids: list[str],
    source_label: str,
    max_text_chars: int,
    include_mermaid: bool = True,
    branch_pools: dict[tuple[str, int], dict[str, Any]] | None = None,
    max_unselected_per_step: int = 0,
) -> str:
    """Render selected trajectories as Markdown."""

    task_counts = Counter(
        (
            summaries.get(trajectory_id, {}).get("task_type", "unknown"),
            summaries.get(trajectory_id, {}).get("evidence_mode", "unknown"),
            summaries.get(trajectory_id, {}).get("difficulty", "unknown"),
        )
        for trajectory_id in trajectory_ids
    )
    lines = [
        "# Trajectory Review",
        "",
        f"- Source: `{source_label}`",
        f"- Rendered trajectories: `{len(trajectory_ids)}`",
        f"- Buckets: `{dict(sorted((('/'.join(key), value) for key, value in task_counts.items())))}`",
        "",
        (
            "Selected branches form the main path. Unselected branches, when shown, are one-step "
            "alternatives from the same prefix; they were scored but not rolled out further."
        ),
        "",
        "## Index",
        "",
    ]
    for index, trajectory_id in enumerate(trajectory_ids, start=1):
        summary = summaries.get(trajectory_id)
        lines.append(f"{index}. [{_trajectory_title(trajectory_id, summary)}](#{_safe_anchor(trajectory_id).lower()})")
    lines.append("")

    for trajectory_id in trajectory_ids:
        turns = turns_by_trajectory[trajectory_id]
        summary = summaries.get(trajectory_id)
        lines.extend(_trajectory_overview(trajectory_id, turns, summary, max_text_chars=max_text_chars))
        if include_mermaid:
            lines.extend(
                [
                    "**Graph View**",
                    "",
                    _mermaid_graph(
                        trajectory_id,
                        turns,
                        summary,
                        branch_pools=branch_pools,
                        max_unselected_per_step=max_unselected_per_step,
                    ),
                    "",
                ]
            )
        for turn in turns:
            lines.extend(
                _step_block(
                    turn,
                    max_text_chars=max_text_chars,
                    branch_pool=branch_pools.get((trajectory_id, turn.step_index)) if branch_pools else None,
                    max_unselected_per_step=max_unselected_per_step,
                )
            )
        lines.extend(["---", ""])
    return "\n".join(lines)


def render_mermaid_graphs(
    *,
    turns_by_trajectory: dict[str, list[TrajectoryTurn]],
    summaries: dict[str, dict[str, Any]],
    trajectory_ids: list[str],
    branch_pools: dict[tuple[str, int], dict[str, Any]] | None = None,
    max_unselected_per_step: int = 0,
) -> str:
    """Render one standalone Mermaid document containing all trajectory graphs."""

    lines = [
        "flowchart TD",
        "  classDef selected fill:#dbeafe,stroke:#2563eb,stroke-width:2px;",
        "  classDef unselected fill:#f8fafc,stroke:#94a3b8,stroke-width:1px,stroke-dasharray:4 3;",
        "  classDef final fill:#dcfce7,stroke:#16a34a,stroke-width:2px;",
        "  classDef taskMismatch fill:#ffedd5,stroke:#f97316,stroke-width:2px;",
        "  classDef lowQuality fill:#fee2e2,stroke:#dc2626,stroke-width:2px;",
    ]
    for trajectory_id in trajectory_ids:
        prefix = "n" + re.sub(r"[^A-Za-z0-9_]", "_", trajectory_id)
        title = _safe_mermaid_label(_trajectory_title(trajectory_id, summaries.get(trajectory_id)), max_chars=90)
        lines.append(f"  subgraph {prefix}_subgraph[\"{title}\"]")
        lines.append(f'    {prefix}_start["start"]')
        previous_node = f"{prefix}_start"
        for turn in turns_by_trajectory[trajectory_id]:
            branch = turn.branch
            state = branch.verifier_step.updated_state
            label = (
                f"Step {turn.step_index} | Selected<br/>"
                f"Tool: {_tool_name(branch)} ({_branch_observation_status(branch)})<br/>"
                f"State: {state.relationship_status.value}<br/>"
                f"Score: {branch.local_score.total_score:.2f}"
            )
            node_id = f"{prefix}_s{turn.step_index}"
            lines.append(f'    {node_id}["{_safe_mermaid_label(label, max_chars=120)}"]')
            lines.append(f"    {previous_node} --> {node_id}")
            lines.append(
                f"    class {node_id} "
                f"{_branch_visual_class(branch, summary=summaries.get(trajectory_id), selected=True)};"
            )
            branch_pool = branch_pools.get((trajectory_id, turn.step_index)) if branch_pools else None
            for alt_index, alt_branch in enumerate(
                _top_unselected_branches(
                    branch_pool,
                    selected_branch_id=branch.branch_id,
                    max_unselected=max_unselected_per_step,
                ),
                start=1,
            ):
                alt_state = alt_branch.verifier_step.updated_state
                alt_label = (
                    f"Step {turn.step_index} | Alt {alt_index}<br/>"
                    f"Tool: {_tool_name(alt_branch)} ({_branch_observation_status(alt_branch)})<br/>"
                    f"State: {alt_state.relationship_status.value}<br/>"
                    f"Score: {alt_branch.local_score.total_score:.2f}"
                )
                alt_node_id = f"{prefix}_s{turn.step_index}_alt{alt_index}"
                lines.append(f'    {alt_node_id}["{_safe_mermaid_label(alt_label, max_chars=120)}"]')
                lines.append(f"    {previous_node} -.-> {alt_node_id}")
                lines.append(
                    f"    class {alt_node_id} "
                    f"{_branch_visual_class(alt_branch, summary=summaries.get(trajectory_id), selected=False)};"
                )
            previous_node = node_id
        summary = summaries.get(trajectory_id)
        if summary:
            final_state = summary.get("final_state", {})
            final_status = (
                final_state.get("relationship_status", "unknown")
                if isinstance(final_state, dict)
                else "unknown"
            )
            final_reward = summary.get("terminal_reward")
            final_label = f"Final<br/>State: {final_status}"
            if isinstance(final_reward, (int, float)):
                final_label += f"<br/>Reward: {final_reward:.2f}"
            final_node = f"{prefix}_final"
            lines.append(f'    {final_node}["{_safe_mermaid_label(final_label, max_chars=120)}"]')
            lines.append(f"    {previous_node} --> {final_node}")
            lines.append(f"    class {final_node} {_final_visual_class(summary)};")
        lines.append("  end")
    return "\n".join(lines)


def _html_badge(text: Any, class_name: str = "") -> str:
    class_attr = f" {class_name}" if class_name else ""
    return f'<span class="badge{class_attr}">{_html_escape(text)}</span>'


def _html_metric(label: str, value: Any, class_name: str = "") -> str:
    class_attr = f" {class_name}" if class_name else ""
    return (
        f'<div class="metric{class_attr}">'
        f'<span class="metric-label">{_html_escape(label)}</span>'
        f'<strong>{_html_escape(value)}</strong>'
        "</div>"
    )


def _html_final_state(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    final_state = summary.get("final_state", {})
    return final_state if isinstance(final_state, dict) else {}


def _html_final_interpretation(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    final_interpretation = summary.get("final_interpretation", {})
    return final_interpretation if isinstance(final_interpretation, dict) else {}


def _html_summary_panel(
    trajectory_id: str,
    turns: list[TrajectoryTurn],
    summary: dict[str, Any] | None,
    *,
    task_row: dict[str, Any] | None = None,
    max_text_chars: int,
) -> str:
    final_state = _html_final_state(summary)
    final_interpretation = _html_final_interpretation(summary)
    warning = _final_warning(summary)
    first_turn = turns[0] if turns else None
    user_anchors = first_turn.prior_state.user_anchors if first_turn else None
    visible = user_anchors.evidence if user_anchors is not None else {}
    query_text = (
        user_anchors.query_text
        if user_anchors is not None
        else (task_row.get("query_text", "") if task_row else "")
    )
    seed_gene_ids = visible.get("seed_gene_ids", []) if isinstance(visible, dict) else []
    graph_query = visible.get("graph_query_spec") if isinstance(visible, dict) else None
    graph_text = "none"
    if isinstance(graph_query, dict):
        graph_text = (
            f"{graph_query.get('operator', 'graph query')} over "
            f"{len(graph_query.get('seed_gene_ids', []) or [])} seed genes; "
            f"layer scope {graph_query.get('layer_scope', 'unknown')}"
        )
    context_text = visible.get("context_text") if isinstance(visible, dict) else None
    annotations = visible.get("structured_annotations") if isinstance(visible, dict) else None
    alignment = _alignment_metrics(summary=summary, task_row=task_row)
    task_type = summary.get("task_type") if summary else (task_row.get("task_type") if task_row else None)
    expected_behavior = {
        "explanation": "Preserve the visible module and explain the evidence supporting it.",
        "recovery": "Recover the full hidden module from a subset of its genes.",
        "refinement": "Remove added noise genes while preserving true module genes.",
        "none": "Abstain because no coherent functional group is expected.",
    }.get(str(task_type), "unknown")
    if summary:
        badges = " ".join(
            [
                _html_badge(summary.get("task_type", "unknown"), "task"),
                _html_badge(summary.get("evidence_mode", "unknown"), "mode"),
                _html_badge(summary.get("difficulty", "unknown"), "difficulty"),
                _html_badge(final_state.get("relationship_status", "unknown"), _final_visual_class(summary)),
            ]
        )
        metrics = "".join(
            [
                _html_metric("terminal reward", _fmt_float(summary.get("terminal_reward"))),
                _html_metric("gene score", _fmt_float(summary.get("terminal_absolute_complex_score"))),
                _html_metric("mechanism evidence", _fmt_float(summary.get("terminal_mechanism_evidence_score"))),
                _html_metric("steps", summary.get("step_count", len(turns))),
            ]
        )
        labels = _labels_plain(
            [MechanisticLabel.from_dict(item) for item in final_state.get("mechanistic_labels", [])],
            max_chars=max_text_chars,
        )
        source_task = summary.get("source_task_id", trajectory_id)
    else:
        badges = _html_badge("summary missing", "lowQuality")
        metrics = _html_metric("steps", len(turns))
        labels = "n/a"
        source_task = trajectory_id

    warning_html = f'<div class="warning">{_html_escape(warning)}</div>' if warning else ""
    return f"""
      <div class="summary-panel">
        <div class="summary-heading">
          <div>
            <div class="source-task">Source task: <code>{_html_escape(source_task)}</code></div>
            <div class="badges">{badges}</div>
          </div>
          <div class="metrics">{metrics}</div>
        </div>
        {warning_html}
        <div class="story-grid">
          <div>
            <h4>Input Visible To Model</h4>
            <p><strong>Query:</strong> {_html_escape(_truncate(query_text, max_chars=max_text_chars))}</p>
            <p><strong>Seed genes:</strong> {_html_escape(len(seed_gene_ids) if isinstance(seed_gene_ids, list) else 0)} ({_html_escape(_gene_list_preview(seed_gene_ids))})</p>
            <p><strong>Evidence:</strong> {_html_escape(graph_text)}</p>
            <p><strong>Context:</strong> {_html_escape(_truncate(context_text, max_chars=260) if context_text else "none")}</p>
            <p><strong>Structured annotations:</strong> {_html_escape("present" if annotations else "none")}</p>
          </div>
          <div>
            <h4>Hidden Training Target</h4>
            <p><strong>Scoring intent:</strong> {_html_escape(expected_behavior)}</p>
            <p><strong>Target status:</strong> <code>{_html_escape(alignment.get("target_status", "unknown"))}</code></p>
            <p><strong>Target genes:</strong> {_html_escape("null" if alignment.get("target_count") is None else alignment.get("target_count"))} ({_html_escape(alignment.get("target_preview", "unknown"))})</p>
            <p><strong>Hidden mechanism labels:</strong> {_html_escape("none" if not task_row or not task_row.get("mechanism_labels") else _json_inline(task_row.get("mechanism_labels"), max_chars=300))}</p>
          </div>
          <div>
            <h4>Final Output</h4>
            <p><strong>Relationship:</strong> <code>{_html_escape(final_state.get("relationship_status", "unknown"))}</code></p>
            <p><strong>Predicted genes:</strong> {_html_escape(alignment.get("predicted_count", 0))} ({_html_escape(alignment.get("predicted_preview", "none"))})</p>
            <p><strong>Mechanistic labels:</strong> {_html_escape(labels)}</p>
            <p><strong>Claim:</strong> {_html_escape(_truncate(final_interpretation.get("mechanistic_claim", ""), max_chars=max_text_chars) or "empty")}</p>
          </div>
          <div>
            <h4>Alignment Summary</h4>
            <p><strong>Overlap:</strong> {_html_escape("n/a" if alignment.get("overlap_count") is None else alignment.get("overlap_count"))}</p>
            <p><strong>Precision:</strong> {_fmt_float(alignment.get("precision"))}</p>
            <p><strong>Recall:</strong> {_fmt_float(alignment.get("recall"))}</p>
            <p><strong>Jaccard:</strong> {_fmt_float(alignment.get("jaccard"))}</p>
            <p><strong>Final evidence:</strong> {_html_escape(_truncate(final_interpretation.get("main_evidence", ""), max_chars=max_text_chars) or "empty")}</p>
          </div>
        </div>
      </div>
    """


def _branch_row_html(
    *,
    step_index: int,
    role: str,
    branch: CandidateBranch,
    row_class: str,
    max_text_chars: int,
) -> str:
    state = branch.verifier_step.updated_state
    interpretation = branch.verifier_step.updated_interpretation
    labels = _labels_plain(state.mechanistic_labels, max_chars=350)
    observation = _observation_summary(branch.observation, max_chars=500).replace("`", "")
    claim = _truncate(interpretation.mechanistic_claim, max_chars=max_text_chars) or "empty"
    evidence = _truncate(interpretation.main_evidence, max_chars=max_text_chars) or "empty"
    actor = _truncate(branch.actor_step.reasoning_text, max_chars=max_text_chars) or "empty"
    return f"""
      <tr class="{_html_escape(row_class)}">
        <td><strong>{step_index}</strong></td>
        <td>{_html_escape(role)}</td>
        <td><code>{_html_escape(_tool_name(branch))}</code><br><span class="muted">{_html_escape(_branch_observation_status(branch))}</span></td>
        <td>
          <strong>{branch.local_score.total_score:.3f}</strong><br>
          <span class="muted">mech {_fmt_float(branch.local_score.mechanism_evidence_score)}</span>
        </td>
        <td><code>{_html_escape(state.relationship_status.value)}</code></td>
        <td>{_predicted_gene_count(branch)}</td>
        <td>{_html_escape(labels)}</td>
        <td title="{_html_escape(observation)}">{_html_escape(_truncate(observation, max_chars=220))}</td>
        <td>
          <details>
            <summary>{_html_escape(_truncate(claim, max_chars=180))}</summary>
            <h5>Claim</h5>
            <p>{_html_escape(claim)}</p>
            <h5>Evidence</h5>
            <p>{_html_escape(evidence)}</p>
            <h5>Actor Reasoning</h5>
            <p>{_html_escape(actor)}</p>
          </details>
        </td>
      </tr>
    """


def _html_branch_rows(
    trajectory_id: str,
    turns: list[TrajectoryTurn],
    summary: dict[str, Any] | None,
    *,
    branch_pools: dict[tuple[str, int], dict[str, Any]],
    max_unselected_per_step: int,
    max_text_chars: int,
) -> str:
    rows: list[str] = []
    for turn in turns:
        selected = turn.branch
        rows.append(
            _branch_row_html(
                step_index=turn.step_index,
                role="Selected",
                branch=selected,
                row_class=_branch_visual_class(selected, summary=summary, selected=True),
                max_text_chars=max_text_chars,
            )
        )
        branch_pool = branch_pools.get((trajectory_id, turn.step_index))
        for alt_index, alt_branch in enumerate(
            _top_unselected_branches(
                branch_pool,
                selected_branch_id=selected.branch_id,
                max_unselected=max_unselected_per_step,
            ),
            start=1,
        ):
            rows.append(
                _branch_row_html(
                    step_index=turn.step_index,
                    role=f"Alt {alt_index}",
                    branch=alt_branch,
                    row_class=_branch_visual_class(alt_branch, summary=summary, selected=False),
                    max_text_chars=max_text_chars,
                )
            )
    return "\n".join(rows)


def _html_trajectory_section(
    *,
    index: int,
    trajectory_id: str,
    turns: list[TrajectoryTurn],
    summary: dict[str, Any] | None,
    task_row: dict[str, Any] | None,
    branch_pools: dict[tuple[str, int], dict[str, Any]],
    max_unselected_per_step: int,
    max_text_chars: int,
) -> str:
    title = _trajectory_title(trajectory_id, summary)
    warning = _final_warning(summary)
    open_attr = " open" if index == 1 else ""
    warning_marker = " task-warning" if warning else ""
    mermaid_source = _mermaid_graph_source(
        trajectory_id,
        turns,
        summary,
        branch_pools=branch_pools,
        max_unselected_per_step=max_unselected_per_step,
    )
    rows = _html_branch_rows(
        trajectory_id,
        turns,
        summary,
        branch_pools=branch_pools,
        max_unselected_per_step=max_unselected_per_step,
        max_text_chars=max_text_chars,
    )
    return f"""
    <details class="trajectory{warning_marker}" id="{_html_escape(_safe_anchor(trajectory_id))}"{open_attr}>
      <summary>
        <span>{index}. {_html_escape(title)}</span>
        <span class="summary-right">{_html_badge("review", _final_visual_class(summary))}</span>
      </summary>
      {_html_summary_panel(trajectory_id, turns, summary, task_row=task_row, max_text_chars=max_text_chars)}
      <div class="graph-card">
        <div class="graph-toolbar">
          <strong>Branch graph</strong>
          <span>Selected path plus top {max_unselected_per_step} one-step alternatives per step.</span>
        </div>
        <pre class="mermaid">{_html_escape(mermaid_source)}</pre>
      </div>
      <div class="table-card">
        <h3>Reasoning And Verification Details</h3>
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>Role</th>
              <th>Tool</th>
              <th>Score</th>
              <th>State</th>
              <th>Genes</th>
              <th>Labels</th>
              <th>Observation</th>
              <th>Claim / Evidence</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </details>
    """


def render_html(
    *,
    turns_by_trajectory: dict[str, list[TrajectoryTurn]],
    summaries: dict[str, dict[str, Any]],
    trajectory_ids: list[str],
    source_label: str,
    branch_pools: dict[tuple[str, int], dict[str, Any]],
    task_rows: dict[str, dict[str, Any]] | None = None,
    max_unselected_per_step: int,
    max_text_chars: int,
) -> str:
    """Render selected trajectories as a standalone Mermaid HTML review page."""

    task_counts = Counter(
        (
            summaries.get(trajectory_id, {}).get("task_type", "unknown"),
            summaries.get(trajectory_id, {}).get("evidence_mode", "unknown"),
            summaries.get(trajectory_id, {}).get("difficulty", "unknown"),
        )
        for trajectory_id in trajectory_ids
    )
    warning_count = sum(1 for trajectory_id in trajectory_ids if _final_warning(summaries.get(trajectory_id)))
    nav_items = "\n".join(
        (
            f'<a href="#{_html_escape(_safe_anchor(trajectory_id))}">'
            f'{index}. {_html_escape(_trajectory_title(trajectory_id, summaries.get(trajectory_id)))}</a>'
        )
        for index, trajectory_id in enumerate(trajectory_ids, start=1)
    )
    sections = "\n".join(
        _html_trajectory_section(
            index=index,
            trajectory_id=trajectory_id,
            turns=turns_by_trajectory[trajectory_id],
            summary=summaries.get(trajectory_id),
            task_row=(task_rows or {}).get(summaries.get(trajectory_id, {}).get("source_task_id", "")),
            branch_pools=branch_pools,
            max_unselected_per_step=max_unselected_per_step,
            max_text_chars=max_text_chars,
        )
        for index, trajectory_id in enumerate(trajectory_ids, start=1)
    )
    bucket_text = ", ".join(f"{'/'.join(key)}={value}" for key, value in sorted(task_counts.items()))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trajectory Graph Review</title>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose', flowchart: {{ htmlLabels: true, curve: 'basis' }} }});
  </script>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #dbe3ef;
      --selected: #2563eb;
      --selected-bg: #dbeafe;
      --unselected: #94a3b8;
      --final: #16a34a;
      --final-bg: #dcfce7;
      --warn: #f97316;
      --warn-bg: #ffedd5;
      --bad: #dc2626;
      --bad-bg: #fee2e2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    .layout {{ display: grid; grid-template-columns: 300px minmax(0, 1fr); min-height: 100vh; }}
    aside {{
      position: sticky;
      top: 0;
      align-self: start;
      height: 100vh;
      overflow: auto;
      padding: 24px 18px;
      background: #0f172a;
      color: #e2e8f0;
    }}
    aside h1 {{ margin: 0 0 8px; font-size: 20px; }}
    aside p {{ color: #b6c2d1; font-size: 13px; margin: 0 0 16px; }}
    aside a {{
      display: block;
      color: #dbeafe;
      text-decoration: none;
      padding: 8px 0;
      border-top: 1px solid rgba(226, 232, 240, 0.14);
      font-size: 13px;
    }}
    main {{ padding: 28px; min-width: 0; }}
    .run-header {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: 0 6px 24px rgba(15, 23, 42, 0.05);
    }}
    .run-header h2 {{ margin: 0 0 8px; font-size: 26px; }}
    .run-header code, .source-task code {{ background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--muted); }}
    .swatch {{ width: 18px; height: 12px; border-radius: 3px; border: 2px solid currentColor; background: #fff; }}
    .swatch.selected {{ color: var(--selected); background: var(--selected-bg); }}
    .swatch.unselected {{ color: var(--unselected); background: #f8fafc; border-style: dashed; }}
    .swatch.final {{ color: var(--final); background: var(--final-bg); }}
    .swatch.mismatch {{ color: var(--warn); background: var(--warn-bg); }}
    .swatch.low {{ color: var(--bad); background: var(--bad-bg); }}
    details.trajectory {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      margin: 18px 0;
      box-shadow: 0 6px 24px rgba(15, 23, 42, 0.05);
      overflow: hidden;
    }}
    details.trajectory.task-warning {{ border-color: #fdba74; }}
    details.trajectory > summary {{
      cursor: pointer;
      padding: 16px 18px;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
    }}
    .summary-panel, .graph-card, .table-card {{ padding: 18px; }}
    .summary-heading {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; }}
    .badges, .metrics {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 700;
      background: #eef2ff;
      color: #3730a3;
    }}
    .badge.final {{ background: var(--final-bg); color: #166534; }}
    .badge.taskMismatch {{ background: var(--warn-bg); color: #9a3412; }}
    .badge.lowQuality {{ background: var(--bad-bg); color: #991b1b; }}
    .metric {{
      min-width: 115px;
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 8px;
      padding: 8px 10px;
    }}
    .metric-label {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
    .metric strong {{ font-size: 17px; }}
    .warning {{
      margin-top: 14px;
      border-left: 4px solid var(--warn);
      background: var(--warn-bg);
      padding: 10px 12px;
      border-radius: 6px;
      color: #7c2d12;
      font-weight: 650;
    }}
    .story-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 16px; }}
    .story-grid div {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fcfdff; }}
    .story-grid h4 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .story-grid p {{ margin: 0 0 7px; font-size: 14px; }}
    .graph-card {{ border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: #f8fafc; }}
    .graph-toolbar {{ display: flex; justify-content: space-between; gap: 14px; color: var(--muted); margin-bottom: 10px; }}
    pre.mermaid {{
      overflow: auto;
      min-height: 180px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}
    .table-card {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-top: 1px solid var(--line); padding: 9px; vertical-align: top; text-align: left; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; background: #f8fafc; }}
    tr.selected td {{ background: rgba(219, 234, 254, 0.32); }}
    tr.unselected td {{ background: rgba(248, 250, 252, 0.72); color: #334155; }}
    tr.taskMismatch td {{ background: rgba(255, 237, 213, 0.68); }}
    tr.lowQuality td {{ background: rgba(254, 226, 226, 0.68); }}
    td details summary {{ cursor: pointer; color: #1d4ed8; }}
    td details p {{ max-width: 760px; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{ position: static; height: auto; }}
      main {{ padding: 16px; }}
      .summary-heading, .story-grid {{ display: block; }}
      .story-grid div {{ margin-top: 10px; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Trajectory Review</h1>
      <p>{_html_escape(source_label)}</p>
      <p>{len(trajectory_ids)} trajectories; {warning_count} warning(s).</p>
      {nav_items}
    </aside>
    <main>
      <section class="run-header">
        <h2>Interactive Trajectory Graph Review</h2>
        <p>Selected branches form the main path. Dashed alternatives are one-step candidates from the same prefix; they were scored but not rolled out further.</p>
        <p><strong>Buckets:</strong> {_html_escape(bucket_text)}</p>
        <div class="legend">
          <span><i class="swatch selected"></i> selected branch</span>
          <span><i class="swatch unselected"></i> unselected candidate</span>
          <span><i class="swatch final"></i> final state</span>
          <span><i class="swatch mismatch"></i> task mismatch risk</span>
          <span><i class="swatch low"></i> low-quality state</span>
        </div>
      </section>
      {sections}
    </main>
  </div>
</body>
</html>
"""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render selected trajectory artifacts into Markdown.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Trajectory run directory containing JSONL artifacts.")
    parser.add_argument("--turns-path", type=Path, default=None, help="Path to trajectory_turns.jsonl.")
    parser.add_argument("--summaries-path", type=Path, default=None, help="Path to final_summaries.jsonl.")
    parser.add_argument("--branch-pools-path", type=Path, default=None, help="Path to branch_pools.jsonl for unselected alternatives.")
    parser.add_argument("--tasks-path", type=Path, default=None, help="Optional task JSONL path for hidden targets.")
    parser.add_argument("--out", type=Path, default=None, help="Output Markdown path. Defaults to RUN_DIR/trajectory_review.md.")
    parser.add_argument("--graph-out", type=Path, default=None, help="Optional standalone Mermaid graph output path.")
    parser.add_argument("--html-out", type=Path, default=None, help="Optional standalone HTML trajectory review output path.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of trajectories to render. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--max-text-chars", type=int, default=DEFAULT_TEXT_CHARS, help="Maximum chars for each free-text field. Use 0 for full text.")
    parser.add_argument("--trajectory-id", action="append", default=[], help="Render one exact trajectory id. Can be repeated.")
    parser.add_argument("--source-task-id", action="append", default=[], help="Filter by source-task-id substring. Can be repeated.")
    parser.add_argument("--task-type", action="append", default=[], help="Filter by task type. Can be repeated.")
    parser.add_argument("--evidence-mode", action="append", default=[], help="Filter by evidence mode. Can be repeated.")
    parser.add_argument("--difficulty", action="append", default=[], help="Filter by difficulty. Can be repeated.")
    parser.add_argument(
        "--max-unselected-per-step",
        type=int,
        default=DEFAULT_MAX_UNSELECTED_PER_STEP,
        help="Number of top unselected branches to render beside each selected step. Use 0 to hide alternatives.",
    )
    parser.add_argument("--no-mermaid", action="store_true", help="Omit Mermaid graph blocks from the Markdown.")
    return parser


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path | None, Path | None, Path | None, Path, Path | None, Path | None, str]:
    run_dir = args.run_dir
    turns_path = args.turns_path
    summaries_path = args.summaries_path
    branch_pools_path = args.branch_pools_path
    tasks_path = args.tasks_path
    if run_dir is None and turns_path is None:
        raise SystemExit("Provide --run-dir or --turns-path.")
    if run_dir is not None:
        turns_path = turns_path or (run_dir / "trajectory_turns.jsonl")
        summaries_path = summaries_path or (run_dir / "final_summaries.jsonl")
        branch_pools_path = branch_pools_path or (run_dir / "branch_pools.jsonl")
        tasks_path = tasks_path or infer_tasks_path(run_dir)
    assert turns_path is not None
    if not turns_path.exists():
        raise SystemExit(f"Missing trajectory turns file: {turns_path}")
    if summaries_path is not None and not summaries_path.exists():
        summaries_path = None
    if branch_pools_path is not None and not branch_pools_path.exists():
        branch_pools_path = None
    if tasks_path is not None and not tasks_path.exists():
        tasks_path = None

    out_path = args.out
    if out_path is None:
        if run_dir is not None:
            out_path = run_dir / "trajectory_review.md"
        else:
            out_path = Path("trajectory_review.md")
    graph_out = args.graph_out
    html_out = args.html_out
    source_label = str(run_dir if run_dir is not None else turns_path)
    return turns_path, summaries_path, branch_pools_path, tasks_path, out_path, graph_out, html_out, source_label


def main() -> None:
    args = _build_arg_parser().parse_args()
    turns_path, summaries_path, branch_pools_path, tasks_path, out_path, graph_out, html_out, source_label = _resolve_paths(args)
    turns_by_trajectory = load_turns(turns_path)
    summaries = load_summaries(summaries_path)
    branch_pools = load_branch_pools(branch_pools_path)
    trajectory_ids = _select_trajectory_ids(turns_by_trajectory, summaries, args)
    source_task_ids = {
        summaries.get(trajectory_id, {}).get("source_task_id", "")
        for trajectory_id in trajectory_ids
    }
    task_rows = load_task_rows(tasks_path, {task_id for task_id in source_task_ids if task_id})
    markdown = render_markdown(
        turns_by_trajectory=turns_by_trajectory,
        summaries=summaries,
        trajectory_ids=trajectory_ids,
        source_label=source_label,
        max_text_chars=args.max_text_chars,
        include_mermaid=not args.no_mermaid,
        branch_pools=branch_pools,
        max_unselected_per_step=max(args.max_unselected_per_step, 0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown + "\n", encoding="utf-8")
    message = f"Wrote {len(trajectory_ids)} rendered trajectories to {out_path}"
    if graph_out is not None:
        graph_out.parent.mkdir(parents=True, exist_ok=True)
        graph_out.write_text(
            render_mermaid_graphs(
                turns_by_trajectory=turns_by_trajectory,
                summaries=summaries,
                trajectory_ids=trajectory_ids,
                branch_pools=branch_pools,
                max_unselected_per_step=max(args.max_unselected_per_step, 0),
            )
            + "\n",
            encoding="utf-8",
        )
        message += f" and Mermaid graph to {graph_out}"
    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(
            render_html(
                turns_by_trajectory=turns_by_trajectory,
                summaries=summaries,
                trajectory_ids=trajectory_ids,
                source_label=source_label,
                branch_pools=branch_pools,
                task_rows=task_rows,
                max_unselected_per_step=max(args.max_unselected_per_step, 0),
                max_text_chars=args.max_text_chars,
            ),
            encoding="utf-8",
        )
        message += f" and HTML review to {html_out}"
    print(message)


if __name__ == "__main__":
    main()
