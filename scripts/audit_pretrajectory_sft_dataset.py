#!/usr/bin/env python3
"""Audit a pre-trajectory MENTOR-RL SFT dataset directory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_pretrajectory_sft_dataset import (  # noqa: E402
    BUCKET_BY_VIEW,
    CURRICULUM_STAGES,
    DEFAULT_BUCKET_WEIGHTS,
    GRAPH_TOPOLOGY_SOURCE,
    MENTOR_EV_SOURCE,
    MIXED_SOURCE,
    RWR_LOE_SOURCE,
    SCHEMA_VERSION,
    SPLITS,
)


EXPECTED_MENTOR_EV_SOURCE_DIR = "data/gw_dendrogram_corpus_full_brain"
EXPECTED_RWR_LOE_SOURCE_DIR = "data/rwr_loe_corpus_full_brain"

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
    "layer_specific_claim_calibration",
}
ENTITY_SCHEMA_VIEW_TYPES = {"entity_id_normalization", "gene_alias_disambiguation", "graph_schema_provenance", "layer_tag_metadata", "layer_family_membership"}
MODULE_SET_VIEW_TYPES = {
    "mentor_ev_module_membership",
    "module_overlap_set_algebra",
    "module_containment_set_algebra",
    "module_source_distinction",
    "module_cohesion_summary",
}
RWR_VIEW_TYPES = {"rwr_loe_rank_lookup", "rwr_loe_rank_comparison", "rwr_loe_topk_membership", "rwr_neighborhood_interpretation"}
STRUCTURED_TOOL_VIEW_TYPES = {"tool_call_choice", "structured_state_update", "provenance_refusal_raw_cli"}
CALIBRATION_VIEW_TYPES = {"no_edge_no_path_calibration", "critique_preference_sft"}
RECOMMENDED_VIEW_TYPES = set(BUCKET_BY_VIEW)
OPTIONAL_CONDITIONAL_VIEW_TYPES = {"gene_alias_disambiguation", "shared_common_neighbors"}
KNOWN_CONTEXT_MODES = {"no_context", "open_book_context", "tool_observation"}


@dataclass
class AuditIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    line: int | None = None
    record_id: str | None = None
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.line is not None:
            payload["line"] = self.line
        if self.record_id is not None:
            payload["record_id"] = self.record_id
        if self.context:
            payload["context"] = self.context
        return payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json_file(path: Path, add_issue: Callable[..., None]) -> dict[str, Any]:
    if not path.exists():
        add_issue("fatal", "missing_file", f"Missing required JSON file: {path}", path=path)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        add_issue(
            "fatal",
            "invalid_json",
            f"Invalid JSON in {path}: {exc}",
            path=path,
            line=exc.lineno,
        )
        return {}
    if not isinstance(payload, dict):
        add_issue("fatal", "json_not_object", f"Expected JSON object in {path}", path=path)
        return {}
    return payload


def read_jsonl_file(path: Path, add_issue: Callable[..., None]) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    if not path.exists():
        add_issue("fatal", "missing_file", f"Missing required JSONL file: {path}", path=path)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                add_issue(
                    "fatal",
                    "invalid_jsonl",
                    f"Invalid JSONL row in {path}: {exc}",
                    path=path,
                    line=line_number,
                )
                continue
            if not isinstance(payload, dict):
                add_issue(
                    "fatal",
                    "jsonl_row_not_object",
                    f"Expected JSON object row in {path}",
                    path=path,
                    line=line_number,
                )
                continue
            rows.append((line_number, payload))
    return rows


def has_negation_near(text: str, start: int) -> bool:
    prefix = text[max(0, start - 32) : start]
    return bool(re.search(r"\b(no|not|never|without|avoid|unsupported)\b", prefix))


def unsupported_answer_claim(answer: str) -> str | None:
    text = answer.lower()
    if "definitely causally" in text:
        return "answer contains 'definitely causally'"
    if re.search(r"\bproves? causality\b", text):
        return "answer says the evidence proves causality"
    for match in re.finditer(r"\bconfirmed causal (gene|relationship|candidate)s?\b", text):
        if not has_negation_near(text, match.start()):
            return "answer makes an uncaveated confirmed-causal claim"
    phrase = "there is no biological relationship"
    pos = text.find(phrase)
    if pos >= 0 and not has_negation_near(text, pos) and "does not prove" not in text:
        return "answer treats graph absence as biological absence"
    return None


def require_payload_keys(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    *,
    add_issue: Callable[..., None],
    record_id: str | None,
    path: Path,
    line: int,
    view_type: str,
) -> None:
    for key in keys:
        if key not in payload:
            add_issue(
                "fatal",
                "missing_payload_key",
                f"{view_type} payload is missing `{key}`.",
                path=path,
                line=line,
                record_id=record_id,
                context={"view_type": view_type, "key": key},
            )


def audit_topology_payload(
    *,
    record: dict[str, Any],
    canonical_object: dict[str, Any],
    path: Path,
    line: int,
    add_issue: Callable[..., None],
) -> None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_id = metadata.get("record_id") if isinstance(metadata.get("record_id"), str) else None
    view_type = str(metadata.get("view_type", ""))
    payload = canonical_object.get("payload")
    if not isinstance(payload, dict):
        add_issue("fatal", "missing_payload", f"{view_type} canonical object has no payload.", path=path, line=line, record_id=record_id)
        return

    paired_layer_views = {
        "monoplex_edge_existence",
        "monoplex_vs_multiplex_path_comparison",
        "layer_specific_claim_calibration",
        "shared_common_neighbors",
    }
    if view_type in paired_layer_views:
        require_payload_keys(
            payload,
            ("layer", "source_gene_id", "target_gene_id"),
            add_issue=add_issue,
            record_id=record_id,
            path=path,
            line=line,
            view_type=view_type,
        )
    if view_type == "multiplex_edge_existence":
        require_payload_keys(
            payload,
            ("source_gene_id", "target_gene_id", "supporting_layers"),
            add_issue=add_issue,
            record_id=record_id,
            path=path,
            line=line,
            view_type=view_type,
        )
    if view_type in {"direct_neighbors_by_layer", "degree_hub_bias"}:
        require_payload_keys(
            payload,
            ("layer", "gene_id", "neighbors") if view_type == "direct_neighbors_by_layer" else ("layer", "gene_id", "degree", "degree_percentile"),
            add_issue=add_issue,
            record_id=record_id,
            path=path,
            line=line,
            view_type=view_type,
        )
    if view_type == "multiplex_edge_existence":
        layers = payload.get("supporting_layers")
        has_edge = payload.get("has_edge", True)
        if not isinstance(layers, list):
            add_issue("fatal", "supporting_layers_not_list", "Multiplex edge record supporting_layers must be a list.", path=path, line=line, record_id=record_id)
        elif has_edge is False and layers:
            add_issue("fatal", "negative_edge_has_supporting_layers", "Negative multiplex edge record must not list supporting layers.", path=path, line=line, record_id=record_id)
        elif has_edge is not False and not layers:
            add_issue("fatal", "empty_supporting_layers", "Positive multiplex edge record has no supporting layers.", path=path, line=line, record_id=record_id)
    if view_type == "direct_neighbors_by_layer":
        neighbors = payload.get("neighbors")
        if not isinstance(neighbors, list):
            add_issue("fatal", "neighbors_not_list", "Direct-neighbor payload must contain a list.", path=path, line=line, record_id=record_id)
        elif str(len(neighbors)) not in str(record.get("answer", "")):
            add_issue("warning", "neighbor_count_not_rendered", "Answer does not render the neighbor count from payload.", path=path, line=line, record_id=record_id)
    if view_type == "unique_multiplex_neighbors":
        neighbor_map = payload.get("neighbor_layer_map")
        if not isinstance(neighbor_map, dict):
            add_issue("fatal", "neighbor_layer_map_not_object", "Unique-neighbor payload must contain a neighbor-to-layer map.", path=path, line=line, record_id=record_id)
    if view_type in {"monoplex_shortest_path", "aggregate_multiplex_shortest_path", "path_layer_decomposition"}:
        path_nodes = payload.get("path_gene_ids")
        if not isinstance(path_nodes, list) or len(path_nodes) < 2:
            add_issue("fatal", "invalid_path_nodes", f"{view_type} payload must contain at least two path nodes.", path=path, line=line, record_id=record_id)
        if view_type in {"monoplex_shortest_path", "aggregate_multiplex_shortest_path"}:
            expected_hops = len(path_nodes) - 1 if isinstance(path_nodes, list) else None
            if expected_hops is not None and str(expected_hops) not in str(record.get("answer", "")):
                add_issue("warning", "hop_count_not_rendered", "Answer does not render the expected hop count.", path=path, line=line, record_id=record_id)
    if view_type == "path_layer_decomposition":
        layer_counts = payload.get("layer_counts")
        if not isinstance(layer_counts, dict) or not layer_counts:
            add_issue("fatal", "empty_layer_counts", "Path layer decomposition requires nonempty layer counts.", path=path, line=line, record_id=record_id)
    if view_type == "induced_subgraph":
        edges = payload.get("edges")
        if not isinstance(edges, list):
            add_issue("fatal", "induced_edges_not_list", "Induced-subgraph payload must contain an edge list.", path=path, line=line, record_id=record_id)
    if view_type == "connected_components":
        components = payload.get("components")
        if not isinstance(components, dict) or not components:
            add_issue("fatal", "empty_components", "Connected-components payload must contain component membership.", path=path, line=line, record_id=record_id)
    if view_type == "degree_hub_bias":
        percentile = payload.get("degree_percentile")
        answer = str(record.get("answer", "")).lower()
        if isinstance(percentile, (int, float)) and percentile > 0.95 and "hub-bias" not in answer:
            add_issue("fatal", "missing_hub_bias_caveat", "Hub-like degree payload lacks hub-bias caveat in answer.", path=path, line=line, record_id=record_id)
    if view_type in {"no_edge_no_path_calibration", "monoplex_vs_multiplex_path_comparison"}:
        answer = str(record.get("answer", "")).lower()
        layer_specific_calibration = "no direct support is recorded" in answer and "layer-specific" in answer
        if "does not" not in answer and "not prove" not in answer and not layer_specific_calibration:
            add_issue("warning", "absence_not_calibrated", "Absence/calibration answer may not clearly avoid biological-absence overclaiming.", path=path, line=line, record_id=record_id)


def audit_rwr_payload(
    *,
    record: dict[str, Any],
    canonical_object: dict[str, Any],
    path: Path,
    line: int,
    add_issue: Callable[..., None],
) -> None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_id = metadata.get("record_id") if isinstance(metadata.get("record_id"), str) else None
    view_type = str(metadata.get("view_type", ""))
    payload = canonical_object.get("payload")
    if not isinstance(payload, dict):
        add_issue("fatal", "missing_payload", f"{view_type} canonical object has no payload.", path=path, line=line, record_id=record_id)
        return
    if view_type == "rwr_loe_rank_lookup":
        require_payload_keys(
            payload,
            ("seed_gene_id", "candidate_gene_id", "rank", "score"),
            add_issue=add_issue,
            record_id=record_id,
            path=path,
            line=line,
            view_type=view_type,
        )
        rank = payload.get("rank")
        score = payload.get("score")
        if not isinstance(rank, int) or rank <= 0:
            add_issue("fatal", "invalid_rwr_rank", "RWR-LOE rank must be a positive integer.", path=path, line=line, record_id=record_id)
        if not isinstance(score, (int, float)):
            add_issue("fatal", "invalid_rwr_score", "RWR-LOE score must be numeric.", path=path, line=line, record_id=record_id)
    if view_type == "rwr_neighborhood_interpretation":
        top_candidates = payload.get("top_candidates")
        if not isinstance(top_candidates, list) or not top_candidates:
            add_issue("fatal", "empty_rwr_top_candidates", "RWR neighborhood record requires top candidates.", path=path, line=line, record_id=record_id)
        elif not all(isinstance(row, dict) and "gene" in row and "rank" in row for row in top_candidates):
            add_issue("fatal", "malformed_rwr_top_candidates", "RWR top candidates must include gene and rank fields.", path=path, line=line, record_id=record_id)
    if view_type == "rwr_loe_rank_comparison":
        require_payload_keys(
            payload,
            ("seed_gene_id", "left_candidate_gene_id", "right_candidate_gene_id", "left_rank", "right_rank", "winner_gene_id"),
            add_issue=add_issue,
            record_id=record_id,
            path=path,
            line=line,
            view_type=view_type,
        )
        left_rank = payload.get("left_rank")
        right_rank = payload.get("right_rank")
        if not isinstance(left_rank, int) or not isinstance(right_rank, int):
            add_issue("fatal", "invalid_rwr_rank_comparison", "RWR rank comparison ranks must be integers.", path=path, line=line, record_id=record_id)
    if view_type == "rwr_loe_topk_membership":
        require_payload_keys(
            payload,
            ("seed_gene_id", "candidate_gene_id", "top_k", "rank", "is_in_top_k"),
            add_issue=add_issue,
            record_id=record_id,
            path=path,
            line=line,
            view_type=view_type,
        )
        rank = payload.get("rank")
        top_k = payload.get("top_k")
        is_in_top_k = payload.get("is_in_top_k")
        if isinstance(rank, int) and isinstance(top_k, int) and isinstance(is_in_top_k, bool):
            if (rank <= top_k) != is_in_top_k:
                add_issue("fatal", "topk_membership_label_mismatch", "RWR top-k membership label does not match rank/top_k.", path=path, line=line, record_id=record_id)
    answer = str(record.get("answer", "")).lower()
    if "causality" not in answer and "causal" not in answer:
        add_issue("warning", "rwr_missing_causal_caveat", "RWR answer should explicitly avoid causal overclaiming.", path=path, line=line, record_id=record_id)


def audit_entity_schema_payload(
    *,
    record: dict[str, Any],
    canonical_object: dict[str, Any],
    path: Path,
    line: int,
    add_issue: Callable[..., None],
) -> None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_id = metadata.get("record_id") if isinstance(metadata.get("record_id"), str) else None
    view_type = str(metadata.get("view_type", ""))
    payload = canonical_object.get("payload")
    if not isinstance(payload, dict):
        add_issue("fatal", "missing_payload", f"{view_type} canonical object has no payload.", path=path, line=line, record_id=record_id)
        return
    if view_type == "entity_id_normalization":
        require_payload_keys(payload, ("alias", "canonical_gene_id", "is_ambiguous"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
        if payload.get("is_ambiguous") is not False:
            add_issue("fatal", "normalization_marked_ambiguous", "Entity normalization rows must be unambiguous.", path=path, line=line, record_id=record_id)
    if view_type == "gene_alias_disambiguation":
        require_payload_keys(payload, ("alias", "candidate_gene_ids", "is_ambiguous"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
        candidates = payload.get("candidate_gene_ids")
        if payload.get("is_ambiguous") is not True or not isinstance(candidates, list) or len(candidates) < 2:
            add_issue("fatal", "alias_disambiguation_not_ambiguous", "Alias disambiguation rows must list multiple candidates.", path=path, line=line, record_id=record_id)
    if view_type in {"layer_tag_metadata", "layer_family_membership"}:
        require_payload_keys(payload, ("layer", "layer_family", "layer_namespace"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
    if view_type == "graph_schema_provenance":
        require_payload_keys(payload, ("source_name", "source_dir", "role"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)


def audit_module_set_payload(
    *,
    record: dict[str, Any],
    canonical_object: dict[str, Any],
    path: Path,
    line: int,
    add_issue: Callable[..., None],
) -> None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_id = metadata.get("record_id") if isinstance(metadata.get("record_id"), str) else None
    view_type = str(metadata.get("view_type", ""))
    payload = canonical_object.get("payload")
    if not isinstance(payload, dict):
        add_issue("fatal", "missing_payload", f"{view_type} canonical object has no payload.", path=path, line=line, record_id=record_id)
        return
    if view_type == "mentor_ev_module_membership":
        require_payload_keys(payload, ("module_id", "gene_id", "has_membership"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
    if view_type == "module_source_distinction":
        require_payload_keys(payload, ("module_id", "source"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
    if view_type == "module_overlap_set_algebra":
        require_payload_keys(payload, ("left_module_id", "right_module_id", "intersection_genes", "intersection_size", "union_size", "overlap_jaccard"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
        intersection = payload.get("intersection_genes")
        if isinstance(intersection, list) and payload.get("intersection_size") != len(intersection):
            add_issue("fatal", "intersection_size_mismatch", "Module overlap intersection_size does not match intersection_genes.", path=path, line=line, record_id=record_id)
    if view_type == "module_containment_set_algebra":
        require_payload_keys(payload, ("left_module_id", "right_module_id", "exact_subset", "violating_genes"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
        violating = payload.get("violating_genes")
        if isinstance(violating, list) and bool(violating) == bool(payload.get("exact_subset")):
            add_issue("fatal", "subset_label_mismatch", "Module containment exact_subset does not match violating_genes.", path=path, line=line, record_id=record_id)
    if view_type == "module_cohesion_summary":
        require_payload_keys(payload, ("module_id", "node_count", "edge_count", "density"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
        density = payload.get("density")
        if not isinstance(density, (int, float)) or density < 0 or density > 1:
            add_issue("fatal", "invalid_module_density", "Module cohesion density must be between 0 and 1.", path=path, line=line, record_id=record_id)


def audit_structured_tool_payload(
    *,
    record: dict[str, Any],
    canonical_object: dict[str, Any],
    path: Path,
    line: int,
    add_issue: Callable[..., None],
) -> None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_id = metadata.get("record_id") if isinstance(metadata.get("record_id"), str) else None
    view_type = str(metadata.get("view_type", ""))
    payload = canonical_object.get("payload")
    if not isinstance(payload, dict):
        add_issue("fatal", "missing_payload", f"{view_type} canonical object has no payload.", path=path, line=line, record_id=record_id)
        return
    if view_type == "tool_call_choice":
        require_payload_keys(payload, ("tool_name", "arguments"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
    if view_type == "structured_state_update":
        require_payload_keys(payload, ("predicted_gene_ids", "relationship_status", "continue"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)
        try:
            answer_payload = json.loads(str(record.get("answer", "")))
        except json.JSONDecodeError:
            add_issue("fatal", "structured_state_answer_not_json", "Structured state update answer must be JSON.", path=path, line=line, record_id=record_id)
        else:
            if not isinstance(answer_payload, dict):
                add_issue("fatal", "structured_state_answer_not_object", "Structured state update answer must be a JSON object.", path=path, line=line, record_id=record_id)
    if view_type == "provenance_refusal_raw_cli":
        require_payload_keys(payload, ("forbidden_interface", "allowed_interface", "graph_version"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)


def audit_pretrajectory_sft_dataset(
    dataset_dir: Path,
    *,
    output_path: Path | None = None,
    mixture_tolerance: float = 0.10,
    coverage_min_records: int = 1000,
    strict_coverage: bool = False,
    fail_on_warnings: bool = False,
    min_records: int = 0,
    max_issues: int = 200,
) -> dict[str, Any]:
    issues: list[AuditIssue] = []

    def add_issue(
        severity: str,
        code: str,
        message: str,
        *,
        path: Path | None = None,
        line: int | None = None,
        record_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        issues.append(
            AuditIssue(
                severity=severity,
                code=code,
                message=message,
                path=str(path) if path is not None else None,
                line=line,
                record_id=record_id,
                context=context,
            )
        )

    manifest_path = dataset_dir / "manifest.json"
    validation_path = dataset_dir / "validation_report.json"
    canonical_path = dataset_dir / "canonical_objects.jsonl"
    manifest = read_json_file(manifest_path, add_issue)
    validation = read_json_file(validation_path, add_issue)
    canonical_rows = read_jsonl_file(canonical_path, add_issue)

    records: list[dict[str, Any]] = []
    record_locations: dict[str, tuple[Path, int]] = {}
    for split in SPLITS:
        split_path = dataset_dir / f"{split}.jsonl"
        for line, record in read_jsonl_file(split_path, add_issue):
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            record_id = metadata.get("record_id") if isinstance(metadata.get("record_id"), str) else None
            if record_id:
                record_locations[record_id] = (split_path, line)
            records.append({"_file_split": split, "_path": split_path, "_line": line, "record": record})

    canonical_by_id: dict[str, dict[str, Any]] = {}
    canonical_seen: set[str] = set()
    for line, obj in canonical_rows:
        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            add_issue("fatal", "missing_canonical_object_id", "Canonical object is missing object_id.", path=canonical_path, line=line)
            continue
        if object_id in canonical_seen:
            add_issue("fatal", "duplicate_canonical_object_id", "Duplicate canonical object id.", path=canonical_path, line=line, context={"object_id": object_id})
        canonical_seen.add(object_id)
        canonical_by_id[object_id] = obj

    accepted_schema_versions = {SCHEMA_VERSION, "pretrajectory-sft-v1"}
    if manifest.get("schema_version") not in accepted_schema_versions:
        add_issue(
            "fatal",
            "schema_version_mismatch",
            "Manifest schema_version does not match the pre-trajectory SFT schema.",
            path=manifest_path,
            context={"expected_one_of": sorted(accepted_schema_versions), "actual": manifest.get("schema_version")},
        )
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), dict) else {}
    if sources.get("mentor_ev_module_source") != EXPECTED_MENTOR_EV_SOURCE_DIR:
        add_issue(
            "fatal",
            "mentor_ev_source_dir_mismatch",
            "Manifest must identify gw_dendrogram as the MENTOR-EV module source.",
            path=manifest_path,
            context={"expected": EXPECTED_MENTOR_EV_SOURCE_DIR, "actual": sources.get("mentor_ev_module_source")},
        )
    if sources.get("rwr_loe_source") != EXPECTED_RWR_LOE_SOURCE_DIR:
        add_issue(
            "fatal",
            "rwr_loe_source_dir_mismatch",
            "Manifest must identify rwr_loe as the RWR-LOE source.",
            path=manifest_path,
            context={"expected": EXPECTED_RWR_LOE_SOURCE_DIR, "actual": sources.get("rwr_loe_source")},
        )

    if min_records and len(records) < min_records:
        add_issue(
            "fatal",
            "record_count_below_minimum",
            "Dataset contains fewer records than requested.",
            context={"actual": len(records), "minimum": min_records},
        )
    if isinstance(manifest.get("selected_record_count"), int) and manifest["selected_record_count"] != len(records):
        add_issue(
            "fatal",
            "manifest_selected_record_count_mismatch",
            "Manifest selected_record_count does not match split JSONL row count.",
            path=manifest_path,
            context={"manifest": manifest["selected_record_count"], "actual": len(records)},
        )
    if isinstance(validation.get("fatal_error_count"), int) and validation["fatal_error_count"] != 0:
        add_issue(
            "fatal",
            "upstream_validation_failed",
            "Builder validation_report has nonzero fatal_error_count.",
            path=validation_path,
            context={"fatal_error_count": validation["fatal_error_count"]},
        )
    if isinstance(validation.get("record_count"), int) and validation["record_count"] != len(records):
        add_issue(
            "fatal",
            "validation_record_count_mismatch",
            "Validation report record_count does not match split JSONL row count.",
            path=validation_path,
            context={"validation": validation["record_count"], "actual": len(records)},
        )

    record_ids: set[str] = set()
    object_splits: dict[str, set[str]] = defaultdict(set)
    counts_by_split: Counter[str] = Counter()
    counts_by_view: Counter[str] = Counter()
    counts_by_bucket: Counter[str] = Counter()
    counts_by_stage: Counter[str] = Counter()
    counts_by_source: Counter[str] = Counter()
    counts_by_context_mode: Counter[str] = Counter()
    allowed_sources = {GRAPH_TOPOLOGY_SOURCE, MENTOR_EV_SOURCE, RWR_LOE_SOURCE, MIXED_SOURCE}
    manifest_graph_version = manifest.get("graph_version")

    for item in records:
        record = item["record"]
        path = item["_path"]
        line = item["_line"]
        file_split = item["_file_split"]
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            add_issue("fatal", "missing_metadata", "Record is missing metadata.", path=path, line=line)
            continue
        record_id = metadata.get("record_id") if isinstance(metadata.get("record_id"), str) else None
        if not record_id:
            add_issue("fatal", "missing_record_id", "Record metadata is missing record_id.", path=path, line=line)
        elif record_id in record_ids:
            add_issue("fatal", "duplicate_record_id", "Duplicate record_id.", path=path, line=line, record_id=record_id)
        else:
            record_ids.add(record_id)
        for key in ("system", "question", "answer"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                add_issue("fatal", f"missing_{key}", f"Record is missing nonempty `{key}`.", path=path, line=line, record_id=record_id)

        split = metadata.get("split")
        if split != file_split:
            add_issue(
                "fatal",
                "metadata_split_file_mismatch",
                "Record metadata split does not match the split file.",
                path=path,
                line=line,
                record_id=record_id,
                context={"metadata_split": split, "file_split": file_split},
            )
        if split not in SPLITS:
            add_issue("fatal", "invalid_split", "Record metadata split is invalid.", path=path, line=line, record_id=record_id, context={"split": split})

        view_type = metadata.get("view_type")
        bucket = metadata.get("mixture_bucket")
        source = metadata.get("source")
        graph_version = metadata.get("graph_version")
        object_id = metadata.get("canonical_object_id")
        context_mode = metadata.get("context_mode", "unspecified")
        curriculum_stage = metadata.get("curriculum_stage", "unspecified")
        if context_mode != "unspecified" and context_mode not in KNOWN_CONTEXT_MODES:
            add_issue(
                "fatal",
                "unknown_context_mode",
                "Record has unknown context_mode.",
                path=path,
                line=line,
                record_id=record_id,
                context={"context_mode": context_mode},
            )
        if view_type not in BUCKET_BY_VIEW:
            add_issue("fatal", "unknown_view_type", "Record has unknown view_type.", path=path, line=line, record_id=record_id, context={"view_type": view_type})
        elif bucket != BUCKET_BY_VIEW[view_type]:
            add_issue(
                "fatal",
                "mixture_bucket_mismatch",
                "Record mixture_bucket does not match view_type mapping.",
                path=path,
                line=line,
                record_id=record_id,
                context={"view_type": view_type, "expected": BUCKET_BY_VIEW[view_type], "actual": bucket},
            )
        if curriculum_stage != "unspecified" and curriculum_stage not in CURRICULUM_STAGES:
            add_issue(
                "fatal",
                "unknown_curriculum_stage",
                "Record has unknown curriculum_stage.",
                path=path,
                line=line,
                record_id=record_id,
                context={"curriculum_stage": curriculum_stage},
            )
        if source not in allowed_sources:
            add_issue("fatal", "unknown_source", "Record source is not one of the expected SFT sources.", path=path, line=line, record_id=record_id, context={"source": source})
        if manifest_graph_version and graph_version != manifest_graph_version:
            add_issue(
                "fatal",
                "graph_version_mismatch",
                "Record graph_version does not match manifest graph_version.",
                path=path,
                line=line,
                record_id=record_id,
                context={"manifest_graph_version": manifest_graph_version, "record_graph_version": graph_version},
            )
        if isinstance(object_id, str):
            object_splits[object_id].add(str(split))
        else:
            add_issue("fatal", "missing_canonical_object_id", "Record metadata is missing canonical_object_id.", path=path, line=line, record_id=record_id)

        canonical_object = canonical_by_id.get(object_id) if isinstance(object_id, str) else None
        if canonical_object is None:
            add_issue("fatal", "missing_canonical_object", "Record points to a canonical object not present in canonical_objects.jsonl.", path=path, line=line, record_id=record_id, context={"canonical_object_id": object_id})
        else:
            if canonical_object.get("source") != source:
                add_issue("fatal", "canonical_source_mismatch", "Record source does not match canonical object source.", path=path, line=line, record_id=record_id)
            if canonical_object.get("split") != split:
                add_issue("fatal", "canonical_split_mismatch", "Record split does not match canonical object split.", path=path, line=line, record_id=record_id)
            if manifest_graph_version and canonical_object.get("graph_version") != manifest_graph_version:
                add_issue("fatal", "canonical_graph_version_mismatch", "Canonical object graph_version does not match manifest.", path=path, line=line, record_id=record_id)
            payload = canonical_object.get("payload")
            if isinstance(payload, dict) and payload.get("source") and payload.get("source") != source:
                add_issue("fatal", "payload_source_mismatch", "Canonical payload source does not match record source.", path=path, line=line, record_id=record_id)
            if view_type in TOPOLOGY_VIEW_TYPES:
                audit_topology_payload(record=record, canonical_object=canonical_object, path=path, line=line, add_issue=add_issue)
            if view_type in ENTITY_SCHEMA_VIEW_TYPES:
                audit_entity_schema_payload(record=record, canonical_object=canonical_object, path=path, line=line, add_issue=add_issue)
            if view_type in MODULE_SET_VIEW_TYPES:
                audit_module_set_payload(record=record, canonical_object=canonical_object, path=path, line=line, add_issue=add_issue)
            if view_type in RWR_VIEW_TYPES:
                audit_rwr_payload(record=record, canonical_object=canonical_object, path=path, line=line, add_issue=add_issue)
            if view_type in STRUCTURED_TOOL_VIEW_TYPES:
                audit_structured_tool_payload(record=record, canonical_object=canonical_object, path=path, line=line, add_issue=add_issue)

        if view_type in TOPOLOGY_VIEW_TYPES and source != GRAPH_TOPOLOGY_SOURCE:
            add_issue("fatal", "topology_source_mismatch", "Topology view must use the multiplex graph source.", path=path, line=line, record_id=record_id, context={"actual": source})
        if view_type in ENTITY_SCHEMA_VIEW_TYPES and source != GRAPH_TOPOLOGY_SOURCE:
            add_issue("fatal", "entity_schema_source_mismatch", "Entity/schema view must use the graph schema source.", path=path, line=line, record_id=record_id, context={"actual": source})
        if view_type in RWR_VIEW_TYPES and source != RWR_LOE_SOURCE:
            add_issue("fatal", "rwr_source_mismatch", "RWR view must use the RWR-LOE source.", path=path, line=line, record_id=record_id, context={"actual": source})
        text = f"{record.get('question', '')}\n{record.get('answer', '')}".lower()
        if source == MENTOR_EV_SOURCE and view_type not in MODULE_SET_VIEW_TYPES and "rwr-loe" in text:
            add_issue("warning", "mentor_ev_mentions_rwr_loe", "MENTOR-EV sourced example mentions RWR-LOE; verify source separation.", path=path, line=line, record_id=record_id)
        if source == RWR_LOE_SOURCE and view_type not in MODULE_SET_VIEW_TYPES and "mentor-ev" in text:
            add_issue("warning", "rwr_loe_mentions_mentor_ev", "RWR-LOE sourced example mentions MENTOR-EV; verify source separation.", path=path, line=line, record_id=record_id)
        claim_issue = unsupported_answer_claim(str(record.get("answer", "")))
        if claim_issue:
            add_issue("fatal", "unsupported_causal_language_in_answer", claim_issue, path=path, line=line, record_id=record_id)

        if isinstance(split, str):
            counts_by_split[split] += 1
        if isinstance(view_type, str):
            counts_by_view[view_type] += 1
        if isinstance(bucket, str):
            counts_by_bucket[bucket] += 1
        if isinstance(curriculum_stage, str):
            counts_by_stage[curriculum_stage] += 1
        if isinstance(source, str):
            counts_by_source[source] += 1
        if isinstance(context_mode, str):
            counts_by_context_mode[context_mode] += 1

    for object_id, splits in sorted(object_splits.items()):
        if len(splits) > 1:
            sample_record = next(
                (item for item in records if item["record"].get("metadata", {}).get("canonical_object_id") == object_id),
                None,
            )
            path, line = (sample_record["_path"], sample_record["_line"]) if sample_record else (None, None)
            add_issue(
                "fatal",
                "canonical_object_split_leakage",
                "Canonical object appears in multiple splits.",
                path=path,
                line=line,
                context={"canonical_object_id": object_id, "splits": sorted(splits)},
            )

    actual_manifest_counts = {
        "record_count_by_split": dict(sorted(counts_by_split.items())),
        "record_count_by_view_type": dict(sorted(counts_by_view.items())),
        "record_count_by_mixture_bucket": dict(sorted(counts_by_bucket.items())),
    }
    if isinstance(manifest.get("record_count_by_curriculum_stage"), dict):
        actual_stage_counts = dict(sorted(counts_by_stage.items()))
        if manifest["record_count_by_curriculum_stage"] != actual_stage_counts:
            add_issue(
                "fatal",
                "manifest_record_count_by_curriculum_stage_mismatch",
                "Manifest record_count_by_curriculum_stage does not match actual records.",
                path=manifest_path,
                context={"manifest": manifest["record_count_by_curriculum_stage"], "actual": actual_stage_counts},
            )
    if isinstance(manifest.get("record_count_by_context_mode"), dict):
        actual_context_counts = dict(sorted(counts_by_context_mode.items()))
        if manifest["record_count_by_context_mode"] != actual_context_counts:
            add_issue(
                "fatal",
                "manifest_record_count_by_context_mode_mismatch",
                "Manifest record_count_by_context_mode does not match actual records.",
                path=manifest_path,
                context={"manifest": manifest["record_count_by_context_mode"], "actual": actual_context_counts},
            )
    for key, actual in actual_manifest_counts.items():
        if isinstance(manifest.get(key), dict) and manifest[key] != actual:
            add_issue(
                "fatal",
                f"manifest_{key}_mismatch",
                f"Manifest {key} does not match actual records.",
                path=manifest_path,
                context={"manifest": manifest[key], "actual": actual},
            )

    total = len(records)
    mixture_report: dict[str, dict[str, float | int]] = {}
    for bucket, expected_share in DEFAULT_BUCKET_WEIGHTS.items():
        count = counts_by_bucket.get(bucket, 0)
        actual_share = count / total if total else 0.0
        delta = actual_share - expected_share
        mixture_report[bucket] = {
            "count": count,
            "expected_share": expected_share,
            "actual_share": actual_share,
            "delta": delta,
        }
        if total >= coverage_min_records and abs(delta) > mixture_tolerance:
            add_issue(
                "warning",
                "mixture_bucket_outside_tolerance",
                "Mixture bucket share is outside tolerance.",
                context={"bucket": bucket, "expected_share": expected_share, "actual_share": actual_share, "delta": delta, "tolerance": mixture_tolerance},
            )

    missing_view_types = sorted(RECOMMENDED_VIEW_TYPES - OPTIONAL_CONDITIONAL_VIEW_TYPES - set(counts_by_view))
    if total >= coverage_min_records and missing_view_types:
        severity = "fatal" if strict_coverage else "warning"
        add_issue(
            severity,
            "missing_recommended_view_types",
            "Dataset is missing recommended SFT view types.",
            context={"missing_view_types": missing_view_types},
        )
    missing_buckets = sorted(set(DEFAULT_BUCKET_WEIGHTS) - set(counts_by_bucket))
    if total >= coverage_min_records and missing_buckets:
        severity = "fatal" if strict_coverage else "warning"
        add_issue(
            severity,
            "missing_mixture_buckets",
            "Dataset is missing one or more topology-heavy mixture buckets.",
            context={"missing_buckets": missing_buckets},
        )

    issue_dicts = [issue.to_dict() for issue in issues]
    fatal_count = sum(1 for issue in issues if issue.severity == "fatal")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    report = {
        "schema_version": "pretrajectory-sft-audit-v1",
        "generated_at": utc_now_iso(),
        "dataset_dir": str(dataset_dir),
        "passed": fatal_count == 0 and (warning_count == 0 or not fail_on_warnings),
        "fatal_error_count": fatal_count,
        "warning_count": warning_count,
        "record_count": total,
        "canonical_object_count": len(canonical_by_id),
        "record_count_by_split": dict(sorted(counts_by_split.items())),
        "record_count_by_view_type": dict(sorted(counts_by_view.items())),
        "record_count_by_mixture_bucket": dict(sorted(counts_by_bucket.items())),
        "record_count_by_curriculum_stage": dict(sorted(counts_by_stage.items())),
        "record_count_by_context_mode": dict(sorted(counts_by_context_mode.items())),
        "record_count_by_source": dict(sorted(counts_by_source.items())),
        "mixture_report": mixture_report,
        "missing_recommended_view_types": missing_view_types,
        "issues": issue_dicts[:max_issues],
        "truncated_issue_count": max(0, len(issue_dicts) - max_issues),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a pre-trajectory MENTOR-RL SFT dataset directory.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Audit report path. Defaults to DATASET_DIR/audit_report.json.")
    parser.add_argument("--mixture-tolerance", type=float, default=0.10)
    parser.add_argument("--coverage-min-records", type=int, default=1000)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument("--min-records", type=int, default=0)
    parser.add_argument("--max-issues", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="Print the full report instead of a compact summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.out if args.out is not None else args.dataset_dir / "audit_report.json"
    report = audit_pretrajectory_sft_dataset(
        args.dataset_dir,
        output_path=output_path,
        mixture_tolerance=args.mixture_tolerance,
        coverage_min_records=args.coverage_min_records,
        strict_coverage=args.strict_coverage,
        fail_on_warnings=args.fail_on_warnings,
        min_records=args.min_records,
        max_issues=args.max_issues,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = {
            "audit_report": str(output_path),
            "passed": report["passed"],
            "fatal_error_count": report["fatal_error_count"],
            "warning_count": report["warning_count"],
            "record_count": report["record_count"],
            "record_count_by_split": report["record_count_by_split"],
            "record_count_by_mixture_bucket": report["record_count_by_mixture_bucket"],
            "record_count_by_curriculum_stage": report["record_count_by_curriculum_stage"],
            "record_count_by_context_mode": report["record_count_by_context_mode"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    if report["fatal_error_count"] or (args.fail_on_warnings and report["warning_count"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
