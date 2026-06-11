#!/usr/bin/env python3
"""Evaluate expert labels against trajectory-candidate review packets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ExpertLabel:
    review_item_id: str
    expert_id: str
    chosen_branch_id: str
    confidence: float | str | None = None
    acceptable_branch_ids: list[str] = field(default_factory=list)
    notes: str | None = None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
            yield payload


def _split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for separator in (";", "|"):
        text = text.replace(separator, ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_confidence(value: Any) -> float | str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text.lower()


def _read_labels(path: Path) -> list[ExpertLabel]:
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = list(_read_jsonl(path))

    labels: list[ExpertLabel] = []
    for row in rows:
        labels.append(
            ExpertLabel(
                review_item_id=str(row.get("review_item_id") or "").strip(),
                expert_id=str(row.get("expert_id") or "").strip(),
                chosen_branch_id=str(row.get("chosen_branch_id") or "").strip(),
                confidence=_parse_confidence(row.get("confidence")),
                acceptable_branch_ids=_split_ids(row.get("acceptable_branch_ids")),
                notes=str(row.get("notes") or "").strip() or None,
            )
        )
    return labels


def _score_value(scoring: dict[str, Any]) -> float:
    normalized = scoring.get("normalized_score")
    if isinstance(normalized, (int, float)):
        return float(normalized)
    total = scoring.get("total_score")
    if isinstance(total, (int, float)):
        return float(total)
    return float("-inf")


def _candidate_branch_ids(packet: dict[str, Any]) -> list[str]:
    expert_packet = packet.get("expert_packet")
    if not isinstance(expert_packet, dict):
        return []
    candidates = expert_packet.get("candidate_branches")
    if not isinstance(candidates, list):
        return []
    return [
        str(candidate.get("branch_id"))
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("branch_id")
    ]


def _model_ranked_branch_ids(packet: dict[str, Any]) -> list[str]:
    hidden_scoring = packet.get("hidden_scoring")
    if not isinstance(hidden_scoring, dict):
        return _candidate_branch_ids(packet)
    return [
        str(branch_id)
        for branch_id, _ in sorted(
            hidden_scoring.items(),
            key=lambda item: _score_value(item[1]) if isinstance(item[1], dict) else float("-inf"),
            reverse=True,
        )
    ]


def _is_high_confidence(confidence: float | str | None) -> bool:
    if isinstance(confidence, float):
        return confidence >= 0.75
    return isinstance(confidence, str) and confidence in {"high", "strong", "certain"}


def evaluate_alignment(packets_path: Path, labels_path: Path) -> dict[str, Any]:
    packets = {str(packet.get("review_item_id")): packet for packet in _read_jsonl(packets_path)}
    labels = _read_labels(labels_path)
    usable_labels = [
        label
        for label in labels
        if label.review_item_id in packets and label.chosen_branch_id
    ]

    selected_top1_hits = 0
    model_top1_hits = 0
    model_top2_hits = 0
    pairwise_correct = 0
    pairwise_total = 0
    disagreement_examples: list[dict[str, Any]] = []

    for label in usable_labels:
        packet = packets[label.review_item_id]
        selected_branch_id = str(packet.get("selected_branch_id") or "")
        ranked_branch_ids = _model_ranked_branch_ids(packet)
        model_top1 = ranked_branch_ids[0] if ranked_branch_ids else ""
        model_top2 = set(ranked_branch_ids[:2])
        acceptable = set(label.acceptable_branch_ids)
        acceptable.add(label.chosen_branch_id)

        if selected_branch_id == label.chosen_branch_id:
            selected_top1_hits += 1
        if model_top1 == label.chosen_branch_id:
            model_top1_hits += 1
        if model_top2 & acceptable:
            model_top2_hits += 1

        hidden_scoring = packet.get("hidden_scoring")
        if isinstance(hidden_scoring, dict) and label.chosen_branch_id in hidden_scoring:
            chosen_score = _score_value(hidden_scoring[label.chosen_branch_id])
            for branch_id, scoring in hidden_scoring.items():
                if branch_id == label.chosen_branch_id or not isinstance(scoring, dict):
                    continue
                pairwise_total += 1
                if chosen_score >= _score_value(scoring):
                    pairwise_correct += 1

        if (
            _is_high_confidence(label.confidence)
            and (
                selected_branch_id != label.chosen_branch_id
                or model_top1 != label.chosen_branch_id
            )
        ):
            disagreement_examples.append(
                {
                    "review_item_id": label.review_item_id,
                    "expert_id": label.expert_id,
                    "chosen_branch_id": label.chosen_branch_id,
                    "confidence": label.confidence,
                    "selected_branch_id": selected_branch_id,
                    "model_top1_branch_id": model_top1,
                    "acceptable_branch_ids": sorted(acceptable),
                    "notes": label.notes,
                }
            )

    label_count = len(usable_labels)
    return {
        "packet_count": len(packets),
        "label_count": len(labels),
        "usable_label_count": label_count,
        "selected_vs_expert_top1_agreement": selected_top1_hits / label_count if label_count else None,
        "model_score_vs_expert_top1_agreement": model_top1_hits / label_count if label_count else None,
        "model_score_top2_agreement": model_top2_hits / label_count if label_count else None,
        "pairwise_preference_accuracy": pairwise_correct / pairwise_total if pairwise_total else None,
        "pairwise_preference_comparisons": pairwise_total,
        "missing_label_review_item_ids": sorted(
            label.review_item_id
            for label in labels
            if label.review_item_id not in packets
        ),
        "high_confidence_disagreement_count": len(disagreement_examples),
        "high_confidence_disagreements": disagreement_examples[:25],
        "labels": [asdict(label) for label in usable_labels],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate expert labels for trajectory-candidate packets.")
    parser.add_argument("--review-packets", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Emit full JSON report.")
    return parser


def _print_human(report: dict[str, Any]) -> None:
    print("Expert alignment:")
    for key in (
        "packet_count",
        "label_count",
        "usable_label_count",
        "selected_vs_expert_top1_agreement",
        "model_score_vs_expert_top1_agreement",
        "model_score_top2_agreement",
        "pairwise_preference_accuracy",
        "pairwise_preference_comparisons",
        "high_confidence_disagreement_count",
    ):
        print(f"  {key}: {report.get(key)}")
    disagreements = report.get("high_confidence_disagreements")
    if disagreements:
        print("High-confidence disagreements:")
        for item in disagreements:
            print(
                "  "
                f"{item['review_item_id']}: expert={item['chosen_branch_id']} "
                f"selected={item['selected_branch_id']} model_top1={item['model_top1_branch_id']}"
            )


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = evaluate_alignment(args.review_packets, args.labels)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)


if __name__ == "__main__":
    main()
