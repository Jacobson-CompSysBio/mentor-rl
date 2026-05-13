#!/usr/bin/env python3
"""Render generated DPO preference pairs into a human-reviewable Markdown file."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import CandidateBranch, PreferencePair
from runtime.validators import validate_preference_pair


DEFAULT_SAMPLE_SIZE = 24
DEFAULT_TEXT_CHARS = 1200


def _truncate(value: Any, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "... [truncated]"


def _one_line(value: Any, *, max_chars: int = 240) -> str:
    return " ".join(_truncate(value, max_chars=max_chars).split())


def _json_inline(value: Any, *, max_chars: int = 800) -> str:
    return _truncate(json.dumps(value, sort_keys=True, ensure_ascii=True), max_chars=max_chars)


def _load_pairs(path: Path) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                pair = PreferencePair.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Failed to parse {path.name}:{row_index}: {exc}") from exc
            validation = validate_preference_pair(pair)
            if not validation.valid:
                raise ValueError(
                    f"Invalid pair at {path.name}:{row_index}: {'; '.join(validation.errors)}"
                )
            pairs.append(pair)
    return pairs


def _matches_filters(pair: PreferencePair, args: argparse.Namespace) -> bool:
    if args.pair_id and pair.pair_id not in set(args.pair_id):
        return False
    if args.task_type and pair.task_type.value not in set(args.task_type):
        return False
    if args.evidence_mode and str(pair.evidence_mode) not in set(args.evidence_mode):
        return False
    if args.difficulty_bin and pair.difficulty_bin.value not in set(args.difficulty_bin):
        return False
    if args.pair_category and str(pair.provenance.get("pair_category", "score_margin")) not in set(args.pair_category):
        return False
    if args.decision_step and pair.decision_step not in set(args.decision_step):
        return False
    if args.tool_backed_only and not bool(pair.provenance.get("chosen_has_successful_tool")):
        return False
    if args.min_chosen_group_size is not None:
        chosen_gene_count = pair.provenance.get("chosen_gene_count")
        if not isinstance(chosen_gene_count, int):
            chosen_gene_count = _predicted_gene_count(pair.chosen)
        if chosen_gene_count < args.min_chosen_group_size:
            return False
    if args.source_task_id:
        return any(fragment in pair.source_task_id for fragment in args.source_task_id)
    return True


def _stratified_sample(pairs: list[PreferencePair], *, sample_size: int, seed: int) -> list[PreferencePair]:
    if sample_size <= 0 or sample_size >= len(pairs):
        return list(pairs)

    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[PreferencePair]] = defaultdict(list)
    for pair in pairs:
        buckets[(pair.task_type.value, str(pair.evidence_mode), pair.difficulty_bin.value)].append(pair)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[PreferencePair] = []
    bucket_keys = sorted(buckets)
    rng.shuffle(bucket_keys)
    while len(selected) < sample_size:
        made_progress = False
        for key in bucket_keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            made_progress = True
            if len(selected) >= sample_size:
                break
        if not made_progress:
            break
    return sorted(
        selected,
        key=lambda pair: (
            pair.task_type.value,
            str(pair.evidence_mode),
            pair.difficulty_bin.value,
            pair.source_task_id,
            pair.decision_step,
            pair.pair_id,
        ),
    )


def _tool_action_summary(branch: CandidateBranch) -> str:
    action = branch.actor_step.tool_action
    if action is None:
        return "`no_tool`"
    return f"`{action.tool_name}` `{_json_inline(action.arguments, max_chars=600)}`"


def _predicted_gene_summary(branch: CandidateBranch) -> str:
    groups = branch.verifier_step.updated_state.predicted_groups
    if not groups:
        return "`[]`"
    parts = []
    for group in groups:
        parts.append(f"{group.group_id}: {', '.join(group.gene_ids)}")
    return _truncate("; ".join(parts), max_chars=600)


def _predicted_gene_count(branch: CandidateBranch) -> int:
    gene_ids: list[str] = []
    for group in branch.verifier_step.updated_state.predicted_groups:
        gene_ids.extend(group.gene_ids)
    return len(set(gene_ids))


def _label_summary(branch: CandidateBranch) -> str:
    labels = branch.verifier_step.updated_state.mechanistic_labels
    if not labels:
        return "`[]`"
    return _truncate(
        "; ".join(
            f"{label.label_source.value}:{label.label_name}"
            + (f" ({label.label_id})" if label.label_id else "")
            for label in labels
        ),
        max_chars=600,
    )


def _observation_summary(branch: CandidateBranch, *, max_chars: int) -> str:
    observation = branch.observation
    if observation is None:
        return "None"
    payload = observation.payload or {}
    provenance = observation.provenance or {}
    pieces = [
        f"status={observation.status.value}",
        f"tool={provenance.get('tool_name')}",
    ]
    if observation.error:
        pieces.append(f"error={_one_line(observation.error, max_chars=240)}")
    for key in (
        "query_gene_id",
        "query_gene_ids",
        "source_gene_id",
        "target_gene_id",
        "unique_neighbor_count",
        "combined_edge_count",
        "hop_count",
        "top_k",
    ):
        if key in payload:
            pieces.append(f"{key}={_json_inline(payload[key], max_chars=240)}")
    if "results" in payload and isinstance(payload["results"], list):
        pieces.append(f"results_sample={_json_inline(payload['results'][:5], max_chars=600)}")
    return _truncate("; ".join(pieces), max_chars=max_chars)


def _score_summary(branch: CandidateBranch) -> str:
    score = branch.local_score
    return (
        f"schema={score.schema_score:.3f}, complex_delta={score.complex_membership_delta:.3f}, "
        f"mechanism_delta={score.mechanistic_label_delta:.3f}, "
        f"efficiency_penalty={score.efficiency_penalty:.3f}, total={score.total_score:.3f}, "
        f"normalized={float(score.normalized_score or 0.0):.3f}"
    )


def _branch_block(title: str, branch: CandidateBranch, *, max_text_chars: int) -> str:
    verifier = branch.verifier_step
    state = verifier.updated_state
    interpretation = verifier.updated_interpretation
    directive = branch.metadata.get("actor_sampling_directive")
    directive_name = directive.get("directive_name") if isinstance(directive, dict) else None
    lines = [
        f"### {title}",
        "",
        f"- Branch: `{branch.branch_id}`",
        f"- Tool action: {_tool_action_summary(branch)}",
        f"- Sampling directive: `{directive_name or 'none'}`",
        f"- Score: {_score_summary(branch)}",
        f"- Observation: {_observation_summary(branch, max_chars=max_text_chars)}",
        f"- Relationship: `{state.relationship_status.value}`; continuation: `{state.continuation_state.value}`",
        f"- Predicted genes: {_predicted_gene_summary(branch)}",
        f"- Mechanistic labels: {_label_summary(branch)}",
        "",
        "**Actor reasoning**",
        "",
        _truncate(branch.actor_step.reasoning_text, max_chars=max_text_chars) or "_empty_",
        "",
        "**Verifier update**",
        "",
        f"- Claim: {_truncate(interpretation.mechanistic_claim, max_chars=max_text_chars) or '_empty_'}",
        f"- Evidence: {_truncate(interpretation.main_evidence, max_chars=max_text_chars) or '_empty_'}",
        f"- Uncertainty: {_truncate(interpretation.uncertainty, max_chars=max_text_chars) or '_empty_'}",
        f"- Next subgoal: {_truncate(interpretation.next_subgoal, max_chars=max_text_chars) or '_empty_'}",
        "",
    ]
    return "\n".join(lines)


def _context_block(pair: PreferencePair, *, max_text_chars: int) -> str:
    context = pair.context
    visible = context.user_evidence if isinstance(context.user_evidence, dict) else {}
    state = context.state
    seed_symbols = visible.get("seed_gene_symbols")
    seed_ids = visible.get("seed_gene_ids")
    lines = [
        "### Context",
        "",
        f"- Query: {_truncate(context.query_text, max_chars=max_text_chars)}",
        f"- Source task: `{pair.source_task_id}`",
        f"- Task/evidence/difficulty: `{pair.task_type.value}` / `{pair.evidence_mode}` / `{pair.difficulty_bin.value}`",
        f"- Decision step: `{pair.decision_step}`; trajectory seed: `{pair.trajectory_seed}`",
        f"- Seeds: `{_json_inline(seed_symbols, max_chars=500)}` / `{_json_inline(seed_ids, max_chars=500)}`",
        f"- Prior relationship: `{state.relationship_status.value}`; remaining budget: `{state.remaining_budget}`",
        f"- Prior interpretation: {_truncate(context.interpretation.mechanistic_claim, max_chars=max_text_chars) or '_empty_'}",
    ]
    context_text = visible.get("context_text")
    if context_text:
        lines.append(f"- Context text: {_truncate(context_text, max_chars=max_text_chars)}")
    graph_query = visible.get("graph_query_spec")
    if graph_query:
        lines.append(f"- Graph query spec: `{_json_inline(graph_query, max_chars=700)}`")
    annotations = visible.get("structured_annotations")
    if annotations:
        lines.append(f"- Structured annotations: `{_json_inline(annotations, max_chars=700)}`")
    lines.append("")
    return "\n".join(lines)


def render_markdown(pairs: list[PreferencePair], *, pairs_path: Path, max_text_chars: int) -> str:
    counts = Counter((pair.task_type.value, str(pair.evidence_mode), pair.difficulty_bin.value) for pair in pairs)
    category_counts = Counter(str(pair.provenance.get("pair_category", "score_margin")) for pair in pairs)
    lines = [
        "# Preference Pair Review",
        "",
        f"- Source: `{pairs_path}`",
        f"- Rendered pairs: `{len(pairs)}`",
        f"- Buckets: `{dict(sorted((('/'.join(key), value) for key, value in counts.items())))}`",
        f"- Pair categories: `{dict(sorted(category_counts.items()))}`",
        "",
        "Use this for manual review: chosen should be visibly better grounded, more valid, or more useful than rejected.",
        "",
    ]
    for index, pair in enumerate(pairs, start=1):
        lines.extend(
            [
                f"## {index}. `{pair.pair_id}`",
                "",
                f"- Score margin: `{pair.score_margin:.6f}`",
                f"- Chosen score: raw `{pair.raw_score_chosen:.6f}`, normalized `{pair.normalized_score_chosen:.6f}`",
                f"- Rejected score: raw `{pair.raw_score_rejected:.6f}`, normalized `{pair.normalized_score_rejected:.6f}`",
                f"- Pair category: `{pair.provenance.get('pair_category', 'score_margin')}`",
                (
                    f"- Group sizes: chosen `{pair.provenance.get('chosen_gene_count', _predicted_gene_count(pair.chosen))}` "
                    f"(delta `{pair.provenance.get('chosen_group_size_delta', 'unknown')}`), "
                    f"rejected `{pair.provenance.get('rejected_gene_count', _predicted_gene_count(pair.rejected))}` "
                    f"(delta `{pair.provenance.get('rejected_group_size_delta', 'unknown')}`)"
                ),
                (
                    f"- Tool transition: `{pair.provenance.get('chosen_tool_name', 'unknown')}` -> "
                    f"`{pair.provenance.get('rejected_tool_name', 'unknown')}`"
                ),
                "",
                _context_block(pair, max_text_chars=max_text_chars),
                _branch_block("Chosen", pair.chosen, max_text_chars=max_text_chars),
                _branch_block("Rejected", pair.rejected, max_text_chars=max_text_chars),
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render preference pairs into Markdown for manual review.")
    parser.add_argument("--pairs-path", type=Path, required=True, help="Path to preference_pairs.jsonl or preference_pairs_raw.jsonl.")
    parser.add_argument("--out", type=Path, default=None, help="Output Markdown path. Defaults to stdout.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of pairs to render. Use 0 for all filtered pairs.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--max-text-chars", type=int, default=DEFAULT_TEXT_CHARS, help="Maximum chars for each free-text field. Use 0 for full text.")
    parser.add_argument("--pair-id", action="append", default=[], help="Render one exact pair id. Can be repeated.")
    parser.add_argument("--task-type", action="append", default=[], help="Filter by task type. Can be repeated.")
    parser.add_argument("--evidence-mode", action="append", default=[], help="Filter by evidence mode. Can be repeated.")
    parser.add_argument("--difficulty-bin", action="append", default=[], help="Filter by difficulty bin. Can be repeated.")
    parser.add_argument("--pair-category", action="append", default=[], help="Filter by pair category from provenance. Can be repeated.")
    parser.add_argument("--decision-step", action="append", type=int, default=[], help="Filter by exact decision step. Can be repeated.")
    parser.add_argument("--tool-backed-only", action="store_true", help="Only render pairs whose chosen branch has a successful tool observation.")
    parser.add_argument("--min-chosen-group-size", type=int, default=None, help="Only render pairs with at least this many chosen predicted genes.")
    parser.add_argument("--source-task-id", action="append", default=[], help="Filter by source-task-id substring. Can be repeated.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    pairs = [pair for pair in _load_pairs(args.pairs_path) if _matches_filters(pair, args)]
    selected = _stratified_sample(pairs, sample_size=args.sample_size, seed=args.seed)
    markdown = render_markdown(
        selected,
        pairs_path=args.pairs_path,
        max_text_chars=args.max_text_chars,
    )
    if args.out is None:
        print(markdown)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {len(selected)} rendered preference pairs to {args.out}")


if __name__ == "__main__":
    main()
