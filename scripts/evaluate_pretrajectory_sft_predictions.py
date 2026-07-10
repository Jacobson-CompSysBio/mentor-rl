#!/usr/bin/env python3
"""Evaluate pre-trajectory SFT predictions with exact graph-fact checks."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


GENE_RE = re.compile(r"\bENSG\d{11}\b")
MODULE_RE = re.compile(r"\b(?:rwr_loe_module|gw_dendrogram_module|mentor_ev_module)_\d+\b")
LAYER_RE = re.compile(r"(?:`([^`]*:[^`]*)`|\b([A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+)\b)")
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])")
FINAL_MARKER_RE = re.compile(r"(?im)(?:^|\n)\s*final\b[:\s]*")
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")

TOPOLOGY_VIEW_TYPES = {
    "monoplex_edge_existence",
    "multiplex_edge_existence",
    "direct_neighbors_by_layer",
    "unique_multiplex_neighbors",
    "gene_layer_membership",
    "nodes_present_by_layer",
    "monoplex_shortest_path",
    "aggregate_multiplex_shortest_path",
    "path_layer_decomposition",
    "monoplex_vs_multiplex_path_comparison",
    "induced_subgraph",
    "connected_components",
    "shared_common_neighbors",
    "degree_hub_bias",
}
ENTITY_SCHEMA_VIEW_TYPES = {
    "entity_id_normalization",
    "gene_alias_disambiguation",
    "graph_schema_provenance",
    "layer_tag_metadata",
    "layer_family_membership",
}
MODULE_SET_VIEW_TYPES = {
    "mentor_ev_module_membership",
    "module_overlap_set_algebra",
    "module_containment_set_algebra",
    "module_source_distinction",
    "module_cohesion_summary",
}
RWR_VIEW_TYPES = {
    "rwr_loe_rank_lookup",
    "rwr_loe_rank_comparison",
    "rwr_loe_topk_membership",
    "rwr_neighborhood_interpretation",
}
STRUCTURED_TOOL_VIEW_TYPES = {
    "tool_call_choice",
    "structured_state_update",
    "provenance_refusal_raw_cli",
}
EXACT_VIEW_TYPES = TOPOLOGY_VIEW_TYPES | ENTITY_SCHEMA_VIEW_TYPES | MODULE_SET_VIEW_TYPES | RWR_VIEW_TYPES | STRUCTURED_TOOL_VIEW_TYPES

UNSUPPORTED_PATTERNS = (
    "definitely causal",
    "confirmed causal",
    "proves caus",
    "causally validated",
    "direct physical interaction",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_canonical_objects(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    objects: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        object_id = row.get("object_id")
        if isinstance(object_id, str):
            objects[object_id] = row
    return objects


def clean_prediction_text(text: str) -> str:
    cleaned = SPECIAL_TOKEN_RE.sub("", text or "").strip()
    matches = list(FINAL_MARKER_RE.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()
    return cleaned


def ids_from_text(text: str) -> set[str]:
    return set(GENE_RE.findall(text or "")) | set(MODULE_RE.findall(text or ""))


def layers_from_text(text: str) -> set[str]:
    layers: set[str] = set()
    for backtick_layer, bare_layer in LAYER_RE.findall(text or ""):
        layers.add(backtick_layer or bare_layer)
    return layers


def numbers_from_text(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text or ""))


def prediction_contains_term(prediction: str, term: str) -> bool:
    if not term:
        return True
    return term in prediction


def json_object_from_text(text: str) -> dict[str, Any] | None:
    cleaned = clean_prediction_text(text)
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        for end in range(len(cleaned), start, -1):
            snippet = cleaned[start:end]
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            return payload if isinstance(payload, dict) else None
    return None


def yes_no_label(text: str) -> str | None:
    stripped = clean_prediction_text(text).strip().lower()
    if re.match(r"^yes\b", stripped):
        return "yes"
    if re.match(r"^no\b", stripped):
        return "no"
    first_clause = stripped[:240]
    if any(phrase in first_clause for phrase in ("not directly connected", "no edge", "no recorded edge", "no path")):
        return "no"
    if "directly connected" in first_clause or "has an edge" in first_clause:
        return "yes"
    return None


def has_unsupported_language(text: str) -> bool:
    lowered = clean_prediction_text(text).lower()
    if any(pattern in lowered for pattern in UNSUPPORTED_PATTERNS):
        return True
    if "no biological relationship" in lowered:
        calibrated = (
            "does not prove there is no biological relationship" in lowered
            or "does not imply there is no biological relationship" in lowered
            or "not prove there is no biological relationship" in lowered
            or "not imply there is no biological relationship" in lowered
        )
        return not calibrated
    return False


def _add_gene_value(target: set[str], value: Any) -> None:
    if isinstance(value, str) and GENE_RE.fullmatch(value):
        target.add(value)


def _add_identifier_value(target: set[str], value: Any) -> None:
    if isinstance(value, str) and (GENE_RE.fullmatch(value) or MODULE_RE.fullmatch(value)):
        target.add(value)


def _add_layer_value(target: set[str], value: Any) -> None:
    if isinstance(value, str) and ":" in value:
        target.add(value)


def _add_term_value(target: set[str], value: Any) -> None:
    if isinstance(value, str) and len(value) >= 3:
        target.add(value)


def _payload_facts(view_type: str, payload: dict[str, Any], answer: str) -> dict[str, Any]:
    expected_ids: set[str] = set()
    allowed_extra_ids: set[str] = set()
    expected_layers: set[str] = set()
    expected_numbers: set[str] = set()
    expected_terms: set[str] = set()
    expected_label: str | None = None

    for key in ("source_gene_id", "target_gene_id", "gene_id", "canonical_gene_id", "candidate_gene_id", "seed_gene_id", "left_candidate_gene_id", "right_candidate_gene_id", "winner_gene_id"):
        _add_gene_value(expected_ids, payload.get(key))
        _add_term_value(expected_terms, payload.get(key))
    for key in ("module_id", "left_module_id", "right_module_id"):
        _add_identifier_value(expected_ids, payload.get(key))
        _add_term_value(expected_terms, payload.get(key))
    for key in ("alias", "source_name", "source", "tool_name", "relationship_status", "allowed_interface", "forbidden_interface", "layer_family", "layer_namespace"):
        _add_term_value(expected_terms, payload.get(key))
    for key in ("layer", "ablated_layer"):
        _add_layer_value(expected_layers, payload.get(key))
        _add_term_value(expected_terms, payload.get(key))
    for key in ("supporting_layers", "layers"):
        values = payload.get(key)
        if isinstance(values, list):
            for value in values:
                _add_layer_value(expected_layers, value)
                _add_term_value(expected_terms, value)
    for key in ("path_gene_ids", "present_gene_ids", "common_neighbors", "intersection_genes", "violating_genes", "predicted_gene_ids"):
        values = payload.get(key)
        if isinstance(values, list):
            for value in values:
                _add_gene_value(expected_ids, value)
                _add_term_value(expected_terms, value)
    if isinstance(payload.get("query_gene_ids"), list):
        for value in payload["query_gene_ids"]:
            _add_gene_value(expected_ids, value)
            _add_term_value(expected_terms, value)
    if isinstance(payload.get("neighbors"), list):
        for value in payload["neighbors"]:
            _add_gene_value(expected_ids, value)
            _add_term_value(expected_terms, value)
        _add_gene_value(allowed_extra_ids, payload.get("gene_id"))
    neighbor_map = payload.get("neighbor_layer_map")
    if isinstance(neighbor_map, dict):
        for gene, layer_values in neighbor_map.items():
            _add_gene_value(expected_ids, gene)
            _add_term_value(expected_terms, gene)
            if isinstance(layer_values, list):
                for value in layer_values:
                    _add_layer_value(expected_layers, value)
                    _add_term_value(expected_terms, value)
        _add_gene_value(allowed_extra_ids, payload.get("gene_id"))
    if isinstance(payload.get("edges"), list):
        for edge in payload["edges"]:
            if isinstance(edge, dict):
                _add_gene_value(expected_ids, edge.get("source_gene_id"))
                _add_gene_value(expected_ids, edge.get("target_gene_id"))
                _add_term_value(expected_terms, edge.get("source_gene_id"))
                _add_term_value(expected_terms, edge.get("target_gene_id"))
                _add_layer_value(expected_layers, edge.get("layer"))
                _add_term_value(expected_terms, edge.get("layer"))
                supporting_layers = edge.get("supporting_layers")
                if isinstance(supporting_layers, list):
                    for value in supporting_layers:
                        _add_layer_value(expected_layers, value)
                        _add_term_value(expected_terms, value)
    layer_counts = payload.get("layer_counts")
    if isinstance(layer_counts, dict):
        for layer, count in layer_counts.items():
            _add_layer_value(expected_layers, layer)
            _add_term_value(expected_terms, layer)
            expected_numbers.add(str(count))
    for key in (
        "degree",
        "rwr_rank",
        "rank",
        "left_rank",
        "right_rank",
        "module_size",
        "size",
        "node_count",
        "edge_count",
        "intersection_size",
        "union_size",
        "top_k",
    ):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            expected_numbers.add(f"{value:g}")
    for key in ("score", "density", "overlap_jaccard", "degree_percentile"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            expected_numbers.add(f"{value:g}")
            expected_numbers.add(f"{value:.6f}")

    if view_type in {"monoplex_edge_existence", "multiplex_edge_existence"}:
        expected_label = "no" if payload.get("has_edge") is False else "yes"
    elif view_type in {"mentor_ev_module_membership", "rwr_loe_topk_membership", "module_containment_set_algebra", "layer_family_membership"}:
        bool_key = {
            "mentor_ev_module_membership": "has_membership",
            "rwr_loe_topk_membership": "is_in_top_k",
            "module_containment_set_algebra": "exact_subset",
            "layer_family_membership": "has_gene",
        }[view_type]
        expected_label = "yes" if payload.get(bool_key) is True else "no"
    elif str(answer).strip().lower().startswith("yes"):
        expected_label = "yes"
    elif str(answer).strip().lower().startswith("no"):
        expected_label = "no"

    if not payload:
        expected_ids = ids_from_text(answer)
        expected_layers = layers_from_text(answer)
        expected_numbers = numbers_from_text(answer)

    return {
        "expected_ids": sorted(expected_ids),
        "allowed_extra_ids": sorted(allowed_extra_ids),
        "expected_layers": sorted(expected_layers),
        "expected_numbers": sorted(expected_numbers),
        "expected_terms": sorted(expected_terms),
        "expected_label": expected_label,
    }


def _rate(numerator: int, denominator: int | None) -> float | None:
    if denominator is None:
        return None
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def evaluate_row(row: dict[str, Any], canonical_objects_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    view_type = str(metadata.get("view_type", ""))
    bucket = str(metadata.get("mixture_bucket", ""))
    answer = str(row.get("answer", ""))
    prediction_raw = str(row.get("prediction", ""))
    prediction = clean_prediction_text(prediction_raw)
    canonical = canonical_objects_by_id.get(str(metadata.get("canonical_object_id", "")), {})
    payload = canonical.get("payload") if isinstance(canonical.get("payload"), dict) else {}
    facts = _payload_facts(view_type, payload, answer)

    expected_ids = set(facts["expected_ids"]) or ids_from_text(answer)
    allowed_extra_ids = set(facts["allowed_extra_ids"])
    expected_layers = set(facts["expected_layers"]) or layers_from_text(answer)
    expected_numbers = set(facts["expected_numbers"])
    expected_terms = set(facts["expected_terms"])
    if not expected_numbers:
        expected_numbers = numbers_from_text(answer)

    pred_ids = ids_from_text(prediction)
    pred_layers = layers_from_text(prediction)
    pred_numbers = numbers_from_text(prediction)
    expected_label = facts["expected_label"]
    pred_label = yes_no_label(prediction)

    missing_ids = sorted(expected_ids - pred_ids)
    extra_ids = sorted(pred_ids - expected_ids - allowed_extra_ids)
    missing_layers = sorted(expected_layers - pred_layers)
    extra_layers = sorted(pred_layers - expected_layers)
    missing_numbers = sorted(expected_numbers - pred_numbers)
    missing_terms = sorted(term for term in expected_terms if not prediction_contains_term(prediction, term))

    id_recall = _rate(len(expected_ids & pred_ids), len(expected_ids)) if expected_ids else None
    id_precision = _rate(len((pred_ids - allowed_extra_ids) & expected_ids), len(pred_ids - allowed_extra_ids)) if expected_ids or pred_ids else None
    layer_recall = _rate(len(expected_layers & pred_layers), len(expected_layers)) if expected_layers else None
    layer_precision = _rate(len(pred_layers & expected_layers), len(pred_layers)) if expected_layers or pred_layers else None
    number_recall = _rate(len(expected_numbers & pred_numbers), len(expected_numbers)) if expected_numbers else None
    label_correct = expected_label is None or pred_label == expected_label
    unsupported = has_unsupported_language(prediction)
    json_subset_match: bool | None = None
    if view_type in {"tool_call_choice", "structured_state_update"}:
        prediction_payload = json_object_from_text(prediction)
        json_subset_match = False
        if prediction_payload is not None:
            for key, value in payload.items():
                if key in {"rank_cache_context"}:
                    continue
                if prediction_payload.get(key) != value:
                    break
            else:
                json_subset_match = True

    exact_graph_fact_pass = None
    if view_type in EXACT_VIEW_TYPES:
        exact_graph_fact_pass = (
            label_correct
            and not missing_ids
            and not extra_ids
            and not missing_layers
            and not missing_terms
            and (view_type not in {"path_layer_decomposition", "degree_hub_bias", "rwr_loe_rank_lookup", "rwr_loe_rank_comparison", "rwr_loe_topk_membership", "module_overlap_set_algebra", "module_cohesion_summary", "layer_tag_metadata"} or not missing_numbers)
            and (json_subset_match is not False)
            and not unsupported
        )

    return {
        "idx": row.get("idx"),
        "record_id": metadata.get("record_id"),
        "view_type": view_type,
        "mixture_bucket": bucket,
        "context_mode": metadata.get("context_mode"),
        "overlap_score": row.get("overlap_score"),
        "prediction_clean": prediction,
        "expected_label": expected_label,
        "predicted_label": pred_label,
        "label_correct": label_correct,
        "id_recall": id_recall,
        "id_precision": id_precision,
        "layer_recall": layer_recall,
        "layer_precision": layer_precision,
        "number_recall": number_recall,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "missing_layers": missing_layers,
        "extra_layers": extra_layers,
        "missing_numbers": missing_numbers,
        "missing_terms": missing_terms,
        "json_subset_match": json_subset_match,
        "unsupported_language": unsupported,
        "exact_graph_fact_pass": exact_graph_fact_pass,
    }


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(mean(present)) if present else None


def _summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    exact_items = [item for item in items if item["exact_graph_fact_pass"] is not None]
    return {
        "count": len(items),
        "mean_overlap_score": _mean([item.get("overlap_score") for item in items]),
        "mean_id_recall": _mean([item["id_recall"] for item in items]),
        "mean_id_precision": _mean([item["id_precision"] for item in items]),
        "mean_layer_recall": _mean([item["layer_recall"] for item in items]),
        "mean_layer_precision": _mean([item["layer_precision"] for item in items]),
        "mean_number_recall": _mean([item["number_recall"] for item in items]),
        "yes_no_accuracy": _mean([1.0 if item["label_correct"] else 0.0 for item in items if item["expected_label"] is not None]),
        "exact_graph_fact_pass_rate": _mean([1.0 if item["exact_graph_fact_pass"] else 0.0 for item in exact_items]),
        "unsupported_language_rate": _mean([1.0 if item["unsupported_language"] else 0.0 for item in items]),
    }


def evaluate_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    canonical_objects_by_id: dict[str, dict[str, Any]] | None = None,
    max_examples: int = 80,
) -> dict[str, Any]:
    canonical_objects_by_id = canonical_objects_by_id or {}
    row_metrics = [evaluate_row(row, canonical_objects_by_id) for row in rows]
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_context_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    for item in row_metrics:
        by_view[item["view_type"]].append(item)
        by_bucket[item["mixture_bucket"]].append(item)
        by_context_mode[str(item.get("context_mode") or "unspecified")].append(item)
        if item["missing_ids"]:
            failure_counts["missing_ids"] += 1
        if item["extra_ids"]:
            failure_counts["extra_ids"] += 1
        if item["missing_layers"]:
            failure_counts["missing_layers"] += 1
        if item["extra_layers"]:
            failure_counts["extra_layers"] += 1
        if item["missing_numbers"]:
            failure_counts["missing_numbers"] += 1
        if item["missing_terms"]:
            failure_counts["missing_terms"] += 1
        if item["json_subset_match"] is False:
            failure_counts["json_subset_mismatch"] += 1
        if item["expected_label"] is not None and not item["label_correct"]:
            failure_counts["wrong_yes_no"] += 1
        if item["unsupported_language"]:
            failure_counts["unsupported_language"] += 1
        if item["exact_graph_fact_pass"] is False:
            failure_counts["exact_graph_fact_fail"] += 1

    examples = sorted(
        row_metrics,
        key=lambda item: (
            item["exact_graph_fact_pass"] is not False,
            item.get("overlap_score") if isinstance(item.get("overlap_score"), (int, float)) else 1.0,
        ),
    )[:max_examples]

    return {
        "sample_count": len(rows),
        "summary": _summarize_group(row_metrics),
        "by_view_type": {key: _summarize_group(value) for key, value in sorted(by_view.items())},
        "by_mixture_bucket": {key: _summarize_group(value) for key, value in sorted(by_bucket.items())},
        "by_context_mode": {key: _summarize_group(value) for key, value in sorted(by_context_mode.items())},
        "failure_counts": dict(sorted(failure_counts.items())),
        "examples": examples,
    }


def render_html_report(report: dict[str, Any], rows_by_idx: dict[Any, dict[str, Any]], *, title: str = "Pre-Trajectory SFT Exact Evaluation") -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return html.escape(str(value))

    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;line-height:1.35}"
        "table{border-collapse:collapse;width:100%;margin:14px 0}td,th{border:1px solid #ddd;padding:6px;vertical-align:top}"
        "th{background:#f5f5f5}.fail{background:#fff0f0}.pass{background:#f0fff4}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:8px;border-radius:4px}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<h2>Summary</h2>",
        "<pre>" + html.escape(json.dumps(report.get("summary", {}), indent=2, sort_keys=True)) + "</pre>",
        "<h2>By View Type</h2><table><tr><th>View</th><th>n</th><th>Exact</th><th>ID R</th><th>ID P</th><th>Layer R</th><th>Layer P</th><th>Yes/No</th></tr>",
    ]
    for view, metrics in report.get("by_view_type", {}).items():
        sections.append(
            "<tr>"
            f"<td>{html.escape(view)}</td><td>{fmt(metrics.get('count'))}</td>"
            f"<td>{fmt(metrics.get('exact_graph_fact_pass_rate'))}</td>"
            f"<td>{fmt(metrics.get('mean_id_recall'))}</td>"
            f"<td>{fmt(metrics.get('mean_id_precision'))}</td>"
            f"<td>{fmt(metrics.get('mean_layer_recall'))}</td>"
            f"<td>{fmt(metrics.get('mean_layer_precision'))}</td>"
            f"<td>{fmt(metrics.get('yes_no_accuracy'))}</td>"
            "</tr>"
        )
    sections.append("</table><h2>Failure Examples</h2>")
    for item in report.get("examples", []):
        row = rows_by_idx.get(item.get("idx"), {})
        css = "pass" if item.get("exact_graph_fact_pass") else "fail"
        sections.append(
            f"<section class='{css}'><h3>idx {fmt(item.get('idx'))} | {fmt(item.get('view_type'))}</h3>"
            "<pre><b>Question</b>\n" + html.escape(str(row.get("question", ""))) + "</pre>"
            "<pre><b>Expected</b>\n" + html.escape(str(row.get("answer", ""))) + "</pre>"
            "<pre><b>Prediction</b>\n" + html.escape(str(row.get("prediction", ""))) + "</pre>"
            "<pre><b>Metrics</b>\n" + html.escape(json.dumps(item, indent=2, sort_keys=True)) + "</pre>"
            "</section>"
        )
    sections.append("</body></html>")
    return "\n".join(sections)


def evaluate_predictions_file(
    predictions_path: Path,
    *,
    canonical_objects_path: Path | None = None,
    json_out: Path | None = None,
    html_out: Path | None = None,
    max_examples: int = 80,
) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    canonical = load_canonical_objects(canonical_objects_path)
    report = evaluate_prediction_rows(rows, canonical_objects_by_id=canonical, max_examples=max_examples)
    if json_out is not None:
        write_json(json_out, report)
    if html_out is not None:
        rows_by_idx = {row.get("idx"): row for row in rows}
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render_html_report(report, rows_by_idx), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pre-trajectory SFT holdout predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset directory containing canonical_objects.jsonl.")
    parser.add_argument("--canonical-objects", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--html-out", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_path = args.canonical_objects
    if canonical_path is None and args.dataset_dir is not None:
        canonical_path = args.dataset_dir / "canonical_objects.jsonl"
    report = evaluate_predictions_file(
        args.predictions,
        canonical_objects_path=canonical_path,
        json_out=args.json_out,
        html_out=args.html_out,
        max_examples=args.max_examples,
    )
    print(json.dumps({"sample_count": report["sample_count"], "summary": report["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
