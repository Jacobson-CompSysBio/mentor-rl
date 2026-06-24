#!/usr/bin/env python3
"""Smoke-test loading generated PreferencePair artifacts for DPO use.

This is not a training script and does not define the final DPO dataset schema.
It only verifies that `preference_pairs.jsonl` can be parsed, validated, and
rendered into non-empty prompt/chosen/rejected text records.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import CandidateBranch, PreferencePair
from runtime.validators import validate_preference_pair


TEXT_PREVIEW_CHARS = 1200
LIST_PREVIEW_ITEMS = 20
LAYER_PREVIEW_ITEMS = 12


@dataclass
class LoaderSmokeReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate_text(value: Any, *, max_chars: int = TEXT_PREVIEW_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "... [truncated]"


def _preview_list(value: Any, *, limit: int = LIST_PREVIEW_ITEMS) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def _compact_list_sample(value: list[Any], *, limit: int = LIST_PREVIEW_ITEMS) -> list[Any]:
    sample: list[Any] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            sample.append(_compact_visible_inputs(item))
        elif isinstance(item, list):
            sample.append(_compact_list_sample(item, limit=min(limit, 5)))
        elif isinstance(item, str):
            sample.append(_truncate_text(item, max_chars=240))
        else:
            sample.append(item)
    return sample


def _compact_payload_mapping(value: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, list):
            compact[f"{key}_count"] = len(item)
            compact[f"{key}_sample"] = _compact_list_sample(item, limit=3)
        elif isinstance(item, dict):
            compact[key] = _compact_payload_mapping(item)
        elif isinstance(item, str):
            compact[key] = _truncate_text(item, max_chars=240)
        else:
            compact[key] = item
    return compact


def _compact_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"active_layers", "queried_layers"} and isinstance(item, list):
            compact[f"{key}_count"] = len(item)
            compact[f"{key}_sample"] = _compact_list_sample(item, limit=LAYER_PREVIEW_ITEMS)
        elif isinstance(item, list):
            compact[f"{key}_count"] = len(item)
            compact[f"{key}_sample"] = _compact_list_sample(item, limit=LIST_PREVIEW_ITEMS)
        elif isinstance(item, dict):
            if key == "payload":
                compact[key] = _compact_payload_mapping(item)
            else:
                compact[key] = _compact_visible_inputs(item)
        elif isinstance(item, str):
            compact[key] = _truncate_text(item, max_chars=240)
        else:
            compact[key] = item
    return compact


def _compact_evidence_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "evidence_id": value.get("evidence_id"),
        "source_type": value.get("source_type"),
        "summary": _truncate_text(value.get("summary"), max_chars=400),
        "supporting_gene_ids_sample": _preview_list(value.get("supporting_gene_ids")),
        "supporting_gene_symbols_sample": _preview_list(value.get("supporting_gene_symbols")),
        "tool_call_id": value.get("tool_call_id"),
        "provenance": _compact_provenance(value.get("provenance")),
    }


def _compact_visible_inputs(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, list):
            compact[f"{key}_count"] = len(item)
            compact[f"{key}_sample"] = _compact_list_sample(item, limit=LIST_PREVIEW_ITEMS)
        elif isinstance(item, dict):
            nested: dict[str, Any] = {}
            for nested_key, nested_item in item.items():
                if isinstance(nested_item, list):
                    nested[f"{nested_key}_count"] = len(nested_item)
                    nested[f"{nested_key}_sample"] = _compact_list_sample(
                        nested_item,
                        limit=LIST_PREVIEW_ITEMS,
                    )
                elif isinstance(nested_item, dict):
                    nested[nested_key] = _compact_visible_inputs(nested_item)
                elif isinstance(nested_item, str):
                    nested[nested_key] = _truncate_text(nested_item, max_chars=400)
                else:
                    nested[nested_key] = nested_item
            compact[key] = nested
        elif isinstance(item, str):
            compact[key] = _truncate_text(item, max_chars=800)
        else:
            compact[key] = item
    return compact


def _compact_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(state_payload)
    user_anchors = compact.get("user_anchors")
    if isinstance(user_anchors, dict):
        compact["user_anchors"] = dict(user_anchors)
        compact["user_anchors"]["evidence"] = _compact_visible_inputs(user_anchors.get("evidence"))
    compact["evidence_log"] = [
        _compact_evidence_record(record)
        for record in compact.get("evidence_log", [])
        if isinstance(record, dict)
    ]
    for group in compact.get("predicted_groups", []):
        if isinstance(group, dict):
            group["rationale"] = _truncate_text(group.get("rationale"), max_chars=600)
    return compact


def _compact_observation_payload(branch: CandidateBranch) -> dict[str, Any] | None:
    if branch.observation is None:
        return None
    payload = branch.observation.to_dict()
    observation_payload = payload.get("payload")
    compact_payload: dict[str, Any] = {}
    if isinstance(observation_payload, dict):
        for key, value in observation_payload.items():
            if isinstance(value, list):
                compact_payload[f"{key}_count"] = len(value)
                compact_payload[f"{key}_sample"] = value[:LIST_PREVIEW_ITEMS]
            elif isinstance(value, str):
                compact_payload[key] = _truncate_text(value, max_chars=400)
            else:
                compact_payload[key] = value
    return {
        "status": payload.get("status"),
        "call_id": payload.get("call_id"),
        "provenance": _compact_provenance(payload.get("provenance")),
        "payload": compact_payload,
    }


def _read_pairs(path: Path, *, max_pairs: int | None, report: LoaderSmokeReport) -> list[PreferencePair]:
    if not path.exists():
        report.error(f"Pairs file does not exist: {path}")
        return []

    pairs: list[PreferencePair] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                pair = PreferencePair.from_dict(payload)
            except Exception as exc:
                report.error(f"Failed to parse {path.name}:{row_index}: {exc}")
                continue
            validation = validate_preference_pair(pair)
            if not validation.valid:
                report.error(f"Invalid pair at {path.name}:{row_index}: {'; '.join(validation.errors)}")
                continue
            pairs.append(pair)
    return pairs


def _branch_for_dpo_text(branch: CandidateBranch) -> str:
    verifier_payload = branch.verifier_step.to_dict()
    updated_state = verifier_payload.get("updated_state")
    if isinstance(updated_state, dict):
        verifier_payload["updated_state"] = _compact_state_payload(updated_state)
    payload = {
        "actor": branch.actor_step.to_dict(),
        "observation": _compact_observation_payload(branch),
        "verifier": verifier_payload,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def render_dpo_record(pair: PreferencePair) -> dict[str, Any]:
    """Render a pair into a minimal DPO-style prompt/chosen/rejected record."""

    prompt_payload = {
        "query_text": pair.context.query_text,
        "visible_inputs": _compact_visible_inputs(pair.context.user_evidence),
        "interpretation": pair.context.interpretation.to_dict(),
        "state": _compact_state_payload(pair.context.state.to_dict()),
        "source_task_id": pair.source_task_id,
        "decision_step": pair.decision_step,
    }
    return {
        "prompt": json.dumps(prompt_payload, sort_keys=True, separators=(",", ":")),
        "chosen": _branch_for_dpo_text(pair.chosen),
        "rejected": _branch_for_dpo_text(pair.rejected),
        "metadata": {
            "pair_id": pair.pair_id,
            "source_task_id": pair.source_task_id,
            "trajectory_id": pair.trajectory_id,
            "decision_step": pair.decision_step,
            "task_type": pair.task_type.value,
            "evidence_mode": pair.evidence_mode,
            "difficulty_bin": pair.difficulty_bin.value,
            "score_margin": pair.score_margin,
            "pair_category": pair.provenance.get("pair_category"),
            "raw_score_delta": pair.provenance.get("raw_score_delta"),
            "normalized_score_delta": pair.provenance.get("normalized_score_delta"),
            "chosen_deterministic_membership_edit": pair.provenance.get(
                "chosen_deterministic_membership_edit"
            ),
            "rejected_deterministic_membership_edit": pair.provenance.get(
                "rejected_deterministic_membership_edit"
            ),
            "chosen_candidate_frontier": pair.provenance.get("chosen_candidate_frontier", []),
            "rejected_candidate_frontier": pair.provenance.get("rejected_candidate_frontier", []),
        },
    }


def smoke_load_pairs(
    path: Path,
    *,
    max_pairs: int | None = 32,
    min_pairs: int = 1,
    max_prompt_chars: int = 20000,
    max_response_chars: int = 50000,
) -> LoaderSmokeReport:
    report = LoaderSmokeReport()
    pairs = _read_pairs(path, max_pairs=max_pairs, report=report)
    if len(pairs) < min_pairs:
        report.error(f"Loaded {len(pairs)} valid pairs, below required minimum {min_pairs}.")

    task_type_counts: Counter[str] = Counter()
    evidence_mode_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    duplicate_text_count = 0

    for pair in pairs:
        record = render_dpo_record(pair)
        prompt = record["prompt"]
        chosen = record["chosen"]
        rejected = record["rejected"]
        if not prompt.strip() or not chosen.strip() or not rejected.strip():
            report.error(f"Rendered DPO record has empty text for pair {pair.pair_id}.")
        if chosen == rejected:
            duplicate_text_count += 1
            report.error(f"Rendered chosen and rejected text are identical for pair {pair.pair_id}.")
        if len(prompt) > max_prompt_chars:
            report.error(
                f"Rendered prompt for pair {pair.pair_id} has {len(prompt)} chars, "
                f"above limit {max_prompt_chars}."
            )
        if len(chosen) > max_response_chars:
            report.error(
                f"Rendered chosen response for pair {pair.pair_id} has {len(chosen)} chars, "
                f"above limit {max_response_chars}."
            )
        if len(rejected) > max_response_chars:
            report.error(
                f"Rendered rejected response for pair {pair.pair_id} has {len(rejected)} chars, "
                f"above limit {max_response_chars}."
            )
        task_type_counts[pair.task_type.value] += 1
        evidence_mode_counts[str(pair.evidence_mode)] += 1
        difficulty_counts[pair.difficulty_bin.value] += 1
        prompt_lengths.append(len(prompt))
        chosen_lengths.append(len(chosen))
        rejected_lengths.append(len(rejected))

    report.metrics = {
        "pairs_loaded": len(pairs),
        "duplicate_text_count": duplicate_text_count,
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "evidence_mode_counts": dict(sorted(evidence_mode_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "prompt_chars_max": max(prompt_lengths) if prompt_lengths else 0,
        "chosen_chars_max": max(chosen_lengths) if chosen_lengths else 0,
        "rejected_chars_max": max(rejected_lengths) if rejected_lengths else 0,
    }
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test loading DPO preference pairs.")
    parser.add_argument("--pairs-path", type=Path, required=True, help="Path to preference_pairs.jsonl.")
    parser.add_argument("--max-pairs", type=int, default=32, help="Maximum number of valid pairs to load. Use 0 for all pairs.")
    parser.add_argument("--min-pairs", type=int, default=1, help="Minimum valid pairs required for success.")
    parser.add_argument("--max-prompt-chars", type=int, default=20000, help="Maximum rendered prompt size allowed.")
    parser.add_argument("--max-response-chars", type=int, default=50000, help="Maximum rendered chosen/rejected response size allowed.")
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    return parser


def _print_human(report: LoaderSmokeReport, pairs_path: Path) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"DPO pair loader smoke: {status}")
    print(f"Pairs path: {pairs_path}")
    print("Metrics:")
    for key, value in report.metrics.items():
        print(f"  {key}: {value}")
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"  {error}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    max_pairs = None if args.max_pairs == 0 else args.max_pairs
    report = smoke_load_pairs(
        args.pairs_path,
        max_pairs=max_pairs,
        min_pairs=args.min_pairs,
        max_prompt_chars=args.max_prompt_chars,
        max_response_chars=args.max_response_chars,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_human(report, args.pairs_path)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
