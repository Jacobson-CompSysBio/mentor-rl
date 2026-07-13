#!/usr/bin/env python3
"""Audit a pre-trajectory MENTOR-RL SFT dataset directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    ANSWER_BUDGET_CONTRACT_VERSION,
    BUCKET_BY_VIEW,
    CURRICULUM_STAGES,
    DEFAULT_MIXTURE_ABSOLUTE_UNDERFILL_TOLERANCE,
    DEFAULT_BUCKET_WEIGHTS,
    DEFAULT_MIXTURE_CONTRACT_MIN_RECORDS,
    DEFAULT_MIXTURE_RELATIVE_UNDERFILL_TOLERANCE,
    DEFAULT_MIXTURE_UNDERFILL_POLICY,
    GRAPH_TOPOLOGY_SOURCE,
    MENTOR_EV_SOURCE,
    MIXTURE_UNDERFILL_POLICIES,
    MIXED_SOURCE,
    PORTABLE_TOKEN_ESTIMATOR,
    RWR_LOE_SOURCE,
    SCHEMA_VERSION,
    SPLITS,
    AnswerBudgetContract,
    answer_budget_contract_from_dict,
    answer_budget_measurements,
    build_mixture_contract_report,
)
from scripts.validate_pretrajectory_sft_curriculum_plan import (  # noqa: E402
    curriculum_plan_hash,
    validate_curriculum_plan,
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
CURRICULUM_ARTIFACT_SCHEMA_VERSION = "pretrajectory-sft-curriculum-artifacts-v1"
CURRICULUM_DATASET_SCHEMA_VERSION = "pretrajectory-sft-v3"
CURRICULUM_AUDIT_SCHEMA_VERSION = "pretrajectory-sft-audit-v3"
CURRICULUM_REQUIRED_REPORTS = ("audit_report.json", "leakage_report.json", "coverage_report.json")
CURRICULUM_MODEL_TEXT_FIELDS = ("system", "question", "context", "answer")
CURRICULUM_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?"
)
CURRICULUM_RELATIVE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:data|config|scripts|runtime|checkpoints|logs)/[A-Za-z0-9_./-]+"
)
CURRICULUM_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\s\\]+\\)+[^\s\\]+")


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
        if payload.get("neighbor_map_scope") == "lexicographic_prefix":
            require_payload_keys(
                payload,
                ("unique_neighbor_count", "omitted_neighbor_count", "omitted_layer_count_by_neighbor"),
                add_issue=add_issue,
                record_id=record_id,
                path=path,
                line=line,
                view_type=view_type,
            )
            total_neighbors = payload.get("unique_neighbor_count")
            omitted_neighbors = payload.get("omitted_neighbor_count")
            if (
                isinstance(neighbor_map, dict)
                and isinstance(total_neighbors, int)
                and isinstance(omitted_neighbors, int)
                and len(neighbor_map) + omitted_neighbors != total_neighbors
            ):
                add_issue(
                    "fatal",
                    "compact_neighbor_count_mismatch",
                    "Compacted unique-neighbor counts do not reconcile.",
                    path=path,
                    line=line,
                    record_id=record_id,
                )
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
        if payload.get("prediction_scope") == "lexicographic_prefix":
            require_payload_keys(
                payload,
                ("predicted_gene_count", "omitted_predicted_gene_count"),
                add_issue=add_issue,
                record_id=record_id,
                path=path,
                line=line,
                view_type=view_type,
            )
            predicted = payload.get("predicted_gene_ids")
            predicted_count = payload.get("predicted_gene_count")
            omitted_count = payload.get("omitted_predicted_gene_count")
            if (
                isinstance(predicted, list)
                and isinstance(predicted_count, int)
                and isinstance(omitted_count, int)
                and len(predicted) + omitted_count != predicted_count
            ):
                add_issue(
                    "fatal",
                    "compact_structured_state_count_mismatch",
                    "Compacted structured-state gene counts do not reconcile.",
                    path=path,
                    line=line,
                    record_id=record_id,
                )
        try:
            answer_payload = json.loads(str(record.get("answer", "")))
        except json.JSONDecodeError:
            add_issue("fatal", "structured_state_answer_not_json", "Structured state update answer must be JSON.", path=path, line=line, record_id=record_id)
        else:
            if not isinstance(answer_payload, dict):
                add_issue("fatal", "structured_state_answer_not_object", "Structured state update answer must be a JSON object.", path=path, line=line, record_id=record_id)
    if view_type == "provenance_refusal_raw_cli":
        require_payload_keys(payload, ("forbidden_interface", "allowed_interface", "graph_version"), add_issue=add_issue, record_id=record_id, path=path, line=line, view_type=view_type)


def _curriculum_estimate_tokens(text: str) -> int:
    """Match the deterministic tokenizer-free estimator used by the v3 compiler."""

    encoded_length = len(text.encode("utf-8"))
    byte_chunks = math.ceil(encoded_length / 4) if encoded_length else 0
    lexical = 0
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        token = match.group(0)
        if re.fullmatch(r"\w+", token, flags=re.UNICODE):
            lexical += max(1, math.ceil(len(token.encode("utf-8")) / 4))
        else:
            lexical += 1
    return max(byte_chunks, lexical)


def _curriculum_budget_measurement(record: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    prompt_text = "\n".join(str(record.get(field, "")) for field in ("system", "question", "context"))
    answer = str(record.get("answer", ""))
    prompt_tokens = _curriculum_estimate_tokens(prompt_text)
    answer_tokens = _curriculum_estimate_tokens(answer)
    total_tokens = prompt_tokens + answer_tokens
    answer_characters = len(answer)
    violations: list[str] = []
    if prompt_tokens > int(profile["max_prompt_tokens"]):
        violations.append("max_prompt_tokens")
    if answer_tokens > int(profile["max_answer_tokens"]):
        violations.append("max_answer_tokens")
    if total_tokens > int(profile["max_total_tokens"]):
        violations.append("max_total_tokens")
    if answer_characters > int(profile["max_answer_characters"]):
        violations.append("max_answer_characters")
    return {
        "prompt_token_estimate": prompt_tokens,
        "answer_token_estimate": answer_tokens,
        "total_token_estimate": total_tokens,
        "answer_character_count": answer_characters,
        "violations": violations,
        "passed": not violations,
    }


def _curriculum_generator_budget_measurement(record: dict[str, Any]) -> dict[str, Any]:
    answer = str(record.get("answer", ""))
    prompt_tokens = (
        _curriculum_estimate_tokens(str(record.get("system", "")))
        + _curriculum_estimate_tokens(str(record.get("question", "")))
        + 16
    )
    answer_tokens = _curriculum_estimate_tokens(answer)
    return {
        "prompt_token_estimate": prompt_tokens,
        "answer_token_estimate": answer_tokens,
        "total_token_estimate": prompt_tokens + answer_tokens,
        "answer_character_count": len(answer),
    }


def _curriculum_stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _audit_curriculum_v3_dataset(
    dataset_dir: Path,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    issues: list[AuditIssue],
    add_issue: Callable[..., None],
    output_path: Path | None,
    mixture_tolerance: float | None,
    coverage_min_records: int,
    strict_coverage: bool,
    fail_on_warnings: bool,
    min_records: int,
    max_issues: int,
    training_max_sequence_tokens: int | None,
    eval_max_answer_tokens: int | None,
    max_answer_characters: int | None,
    mixture_contract_min_records: int | None,
    mixture_absolute_underfill_tolerance: float | None,
    mixture_relative_underfill_tolerance: float | None,
    mixture_underfill_policy: str | None,
) -> dict[str, Any]:
    """Audit the plan-driven v3 artifact without applying legacy view/source rules."""

    plan_path = dataset_dir / "curriculum_plan.json"
    canonical_path = dataset_dir / "canonical_objects.jsonl"
    plan = read_json_file(plan_path, add_issue)
    native_reports = {
        name: read_json_file(dataset_dir / name, add_issue) for name in CURRICULUM_REQUIRED_REPORTS
    }
    canonical_rows = read_jsonl_file(canonical_path, add_issue)

    if manifest.get("schema_version") != CURRICULUM_ARTIFACT_SCHEMA_VERSION:
        add_issue(
            "fatal",
            "artifact_schema_version_mismatch",
            "The v3 dataset manifest has an unsupported artifact-wrapper schema.",
            path=manifest_path,
            context={
                "expected": CURRICULUM_ARTIFACT_SCHEMA_VERSION,
                "actual": manifest.get("schema_version"),
            },
        )
    if plan.get("dataset_schema_version") != CURRICULUM_DATASET_SCHEMA_VERSION:
        add_issue(
            "fatal",
            "plan_dataset_schema_version_mismatch",
            "The embedded curriculum plan does not declare pretrajectory-sft-v3.",
            path=plan_path,
            context={"actual": plan.get("dataset_schema_version")},
        )
    plan_errors = validate_curriculum_plan(plan)
    for error in plan_errors:
        add_issue(
            "fatal",
            "invalid_curriculum_plan",
            error,
            path=plan_path,
        )
    observed_plan_hash = curriculum_plan_hash(plan) if not plan_errors else None
    declared_plan_hash = manifest.get("plan_hash")
    if observed_plan_hash is not None and declared_plan_hash != observed_plan_hash:
        add_issue(
            "fatal",
            "curriculum_plan_hash_mismatch",
            "Manifest plan_hash does not match curriculum_plan.json.",
            path=manifest_path,
            context={"manifest": declared_plan_hash, "actual": observed_plan_hash},
        )
    if plan.get("plan_id") != manifest.get("plan_id"):
        add_issue(
            "fatal",
            "curriculum_plan_id_mismatch",
            "Manifest plan_id does not match curriculum_plan.json.",
            path=manifest_path,
            context={"manifest": manifest.get("plan_id"), "plan": plan.get("plan_id")},
        )

    for report_name, native_report in native_reports.items():
        report_path = dataset_dir / report_name
        if native_report.get("schema_version") != CURRICULUM_ARTIFACT_SCHEMA_VERSION:
            add_issue(
                "fatal",
                "native_report_schema_mismatch",
                f"{report_name} has an unsupported artifact schema.",
                path=report_path,
                context={"actual": native_report.get("schema_version")},
            )
        if native_report.get("plan_id") != manifest.get("plan_id"):
            add_issue(
                "fatal",
                "native_report_plan_id_mismatch",
                f"{report_name} plan_id does not match the manifest.",
                path=report_path,
            )
        if native_report.get("plan_hash") != declared_plan_hash:
            add_issue(
                "fatal",
                "native_report_plan_hash_mismatch",
                f"{report_name} plan_hash does not match the manifest.",
                path=report_path,
            )
        if native_report.get("passed") is not True:
            add_issue(
                "fatal",
                "native_report_failed",
                f"{report_name} is not marked passed.",
                path=report_path,
            )

    native_audit = native_reports["audit_report.json"]
    for field in (
        "fatal_error_count",
        "budget_violation_count_in_selected",
        "raw_path_violation_count_in_selected",
        "metadata_violation_count_in_selected",
        "tool_schema_violation_count_in_selected",
    ):
        if native_audit.get(field) != 0:
            add_issue(
                "fatal",
                "native_selected_row_audit_failed",
                f"Native audit field {field} must be zero.",
                path=dataset_dir / "audit_report.json",
                context={"field": field, "actual": native_audit.get(field)},
            )
    if native_audit.get("leakage_passed") is not True:
        add_issue(
            "fatal",
            "native_audit_leakage_failed",
            "Native audit does not attest that leakage checks passed.",
            path=dataset_dir / "audit_report.json",
        )

    native_leakage = native_reports["leakage_report.json"]
    for field in (
        "oracle_fact_cross_split_count",
        "optional_group_cross_split_count",
        "exact_duplicate_cross_split_count",
        "near_duplicate_cross_split_count",
    ):
        if native_leakage.get(field) != 0:
            add_issue(
                "fatal",
                "native_leakage_count_nonzero",
                f"Native leakage field {field} must be zero.",
                path=dataset_dir / "leakage_report.json",
                context={"field": field, "actual": native_leakage.get(field)},
            )
    native_coverage = native_reports["coverage_report.json"]
    if int(native_coverage.get("underfilled_material_cross_cell_count", 0) or 0) != 0:
        add_issue(
            "fatal",
            "native_material_cross_cell_underfill",
            "Native coverage report contains underfilled material cross-cells.",
            path=dataset_dir / "coverage_report.json",
            context={
                "count": native_coverage.get("underfilled_material_cross_cell_count"),
            },
        )
    build_profiles = plan.get("build_profiles") if isinstance(plan.get("build_profiles"), dict) else {}
    build_profile = build_profiles.get(manifest.get("build_profile"))
    if isinstance(build_profile, dict):
        material_minimum = int(build_profile["minimum_selected_per_material_cross_cell"])
        cross_cells = native_coverage.get("cross_cells")
        if not isinstance(cross_cells, list):
            add_issue(
                "fatal",
                "native_coverage_missing_cross_cells",
                "Native coverage report does not expose required orthogonal cross-cells.",
                path=dataset_dir / "coverage_report.json",
            )
        else:
            independently_underfilled: list[dict[str, Any]] = []
            for cell in cross_cells:
                if not isinstance(cell, dict):
                    continue
                valid = cell.get("valid")
                if not isinstance(valid, int):
                    compacted = cell.get("compacted")
                    filtered = cell.get("filtered")
                    valid = (
                        compacted - filtered
                        if isinstance(compacted, int) and isinstance(filtered, int)
                        else 0
                    )
                selected = cell.get("selected")
                if valid >= material_minimum and (
                    not isinstance(selected, int) or selected < material_minimum
                ):
                    independently_underfilled.append(cell)
            if independently_underfilled:
                add_issue(
                    "fatal",
                    "material_cross_cell_underfill",
                    "Selected rows underfill material orthogonal cross-cells.",
                    path=dataset_dir / "coverage_report.json",
                    context={
                        "minimum_selected": material_minimum,
                        "underfilled_count": len(independently_underfilled),
                        "sample": independently_underfilled[:10],
                    },
                )

    records: list[dict[str, Any]] = []
    for split in SPLITS:
        split_path = dataset_dir / f"{split}.jsonl"
        for line, record in read_jsonl_file(split_path, add_issue):
            records.append({"_file_split": split, "_path": split_path, "_line": line, "record": record})

    canonical_by_id: dict[str, dict[str, Any]] = {}
    for line, obj in canonical_rows:
        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            add_issue(
                "fatal",
                "missing_canonical_object_id",
                "Canonical object is missing object_id.",
                path=canonical_path,
                line=line,
            )
            continue
        if object_id in canonical_by_id:
            add_issue(
                "fatal",
                "duplicate_canonical_object_id",
                "Duplicate canonical object id.",
                path=canonical_path,
                line=line,
                context={"object_id": object_id},
            )
        canonical_by_id[object_id] = obj

    if min_records and len(records) < min_records:
        add_issue(
            "fatal",
            "record_count_below_minimum",
            "Dataset contains fewer records than requested.",
            context={"actual": len(records), "minimum": min_records},
        )
    if manifest.get("selected_record_count") != len(records):
        add_issue(
            "fatal",
            "manifest_selected_record_count_mismatch",
            "Manifest selected_record_count does not match split JSONL rows.",
            path=manifest_path,
            context={"manifest": manifest.get("selected_record_count"), "actual": len(records)},
        )
    if manifest.get("canonical_object_count") != len(canonical_by_id):
        add_issue(
            "fatal",
            "manifest_canonical_object_count_mismatch",
            "Manifest canonical_object_count does not match canonical_objects.jsonl.",
            path=manifest_path,
            context={"manifest": manifest.get("canonical_object_count"), "actual": len(canonical_by_id)},
        )

    families = {
        str(item.get("name")): item
        for item in plan.get("question_families", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    stages = {
        str(item.get("name")): item
        for item in plan.get("stages", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    profiles = plan.get("context_budget_profiles") if isinstance(plan.get("context_budget_profiles"), dict) else {}
    required_metadata = set(
        plan.get("record_contract", {}).get("required_metadata_fields", [])
        if isinstance(plan.get("record_contract"), dict)
        else []
    )
    source_identities = manifest.get("source_identities") if isinstance(manifest.get("source_identities"), dict) else {}
    expected_multiplex_id = plan.get("graph_contract", {}).get("multiplex_id") if isinstance(plan.get("graph_contract"), dict) else None
    expected_store_id = source_identities.get("store_id")
    expected_flist_id = source_identities.get("flist_id")

    record_ids: set[str] = set()
    object_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    text_splits: dict[str, set[str]] = defaultdict(set)
    near_text_splits: dict[str, set[str]] = defaultdict(set)
    counts_by_split: Counter[str] = Counter()
    counts_by_family: Counter[str] = Counter()
    counts_by_bucket: Counter[str] = Counter()
    counts_by_stage: Counter[str] = Counter()
    counts_by_book_mode: Counter[str] = Counter()
    counts_by_source: Counter[str] = Counter()
    counts_by_budget_profile: Counter[str] = Counter()
    counts_by_answer_budget_action: Counter[str] = Counter()
    over_budget_record_count = 0
    missing_budget_count = 0
    max_answer_token_estimate = 0
    max_total_token_estimate = 0
    max_answer_character_count = 0
    optional_group_keys = list(plan.get("split_contract", {}).get("co_grouping_keys_when_present", [])) if isinstance(plan.get("split_contract"), dict) else []

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
            if not isinstance(record.get(key), str) or not str(record.get(key)).strip():
                add_issue("fatal", f"missing_{key}", f"Record is missing nonempty `{key}`.", path=path, line=line, record_id=record_id)
        missing_fields = sorted(field for field in required_metadata if field not in metadata)
        if missing_fields:
            add_issue(
                "fatal",
                "missing_required_v3_metadata",
                "Record is missing plan-required v3 metadata.",
                path=path,
                line=line,
                record_id=record_id,
                context={"missing_fields": missing_fields},
            )
        if metadata.get("schema_version") != CURRICULUM_DATASET_SCHEMA_VERSION:
            add_issue("fatal", "record_schema_version_mismatch", "Record metadata schema_version is not pretrajectory-sft-v3.", path=path, line=line, record_id=record_id)
        split = metadata.get("split")
        if split != file_split:
            add_issue(
                "fatal",
                "metadata_split_file_mismatch",
                "Record metadata split does not match its split file.",
                path=path,
                line=line,
                record_id=record_id,
                context={"metadata_split": split, "file_split": file_split},
            )
        family_name = metadata.get("question_family")
        family = families.get(str(family_name))
        stage_name = metadata.get("curriculum_stage")
        stage = stages.get(str(stage_name))
        bucket = metadata.get("mixture_bucket")
        book_mode = metadata.get("book_mode")
        profile_name = metadata.get("context_budget_profile")
        if family is None:
            add_issue("fatal", "unknown_question_family", "Record question_family is not in the curriculum plan.", path=path, line=line, record_id=record_id, context={"question_family": family_name})
        else:
            if family.get("primary_stage") != stage_name:
                add_issue("fatal", "family_stage_mismatch", "Record stage does not match its question-family contract.", path=path, line=line, record_id=record_id)
            if family.get("mixture_bucket") != bucket:
                add_issue("fatal", "family_bucket_mismatch", "Record mixture bucket does not match its question-family contract.", path=path, line=line, record_id=record_id)
            if book_mode not in family.get("allowed_book_modes", []):
                add_issue("fatal", "family_book_mode_mismatch", "Record book mode is not allowed for its question family.", path=path, line=line, record_id=record_id)
            if metadata.get("difficulty_source") != family.get("difficulty_source"):
                add_issue("fatal", "family_difficulty_source_mismatch", "Record difficulty source does not match its question-family contract.", path=path, line=line, record_id=record_id)
        if stage is None or stage_name == "stage6_blend":
            add_issue("fatal", "invalid_primary_curriculum_stage", "Record does not name a primary stage from 1 through 5.", path=path, line=line, record_id=record_id, context={"stage": stage_name})
        else:
            if book_mode not in stage.get("allowed_book_modes", []):
                add_issue("fatal", "stage_book_mode_mismatch", "Record book mode is not allowed in its primary stage.", path=path, line=line, record_id=record_id)
            if profile_name not in stage.get("allowed_budget_profiles", []):
                add_issue("fatal", "stage_budget_profile_mismatch", "Record budget profile is not allowed in its primary stage.", path=path, line=line, record_id=record_id)

        profile = profiles.get(profile_name) if isinstance(profile_name, str) else None
        if not isinstance(profile, dict):
            missing_budget_count += 1
            add_issue("fatal", "unknown_budget_profile", "Record context_budget_profile is not defined by the plan.", path=path, line=line, record_id=record_id, context={"profile": profile_name})
        else:
            measurement = _curriculum_budget_measurement(record, profile)
            max_answer_token_estimate = max(max_answer_token_estimate, int(measurement["answer_token_estimate"]))
            max_total_token_estimate = max(max_total_token_estimate, int(measurement["total_token_estimate"]))
            max_answer_character_count = max(max_answer_character_count, int(measurement["answer_character_count"]))
            stored_measurement = metadata.get("budget_measurement")
            if stored_measurement != measurement:
                add_issue(
                    "fatal",
                    "budget_measurement_mismatch",
                    "Stored profile budget_measurement does not match model-facing text.",
                    path=path,
                    line=line,
                    record_id=record_id,
                    context={"stored": stored_measurement, "actual": measurement},
                )
            generator_measurement = _curriculum_generator_budget_measurement(record)
            answer_budget = metadata.get("answer_budget")
            if not isinstance(answer_budget, dict):
                missing_budget_count += 1
                counts_by_answer_budget_action["missing"] += 1
                add_issue("fatal", "missing_record_answer_budget_metadata", "Record is missing generator answer_budget metadata.", path=path, line=line, record_id=record_id)
            else:
                counts_by_answer_budget_action["profile_valid"] += 1
                for field, expected in generator_measurement.items():
                    if answer_budget.get(field) != expected:
                        add_issue(
                            "fatal",
                            "answer_budget_measurement_mismatch",
                            "Stored generator answer_budget does not match model-facing text.",
                            path=path,
                            line=line,
                            record_id=record_id,
                            context={"field": field, "stored": answer_budget.get(field), "actual": expected},
                        )
                if answer_budget.get("profile") != profile_name or answer_budget.get("violations") != []:
                    add_issue("fatal", "invalid_answer_budget_metadata", "Generator answer_budget does not attest the selected profile without violations.", path=path, line=line, record_id=record_id)
            external_violations = list(measurement["violations"])
            if training_max_sequence_tokens is not None and measurement["total_token_estimate"] > training_max_sequence_tokens:
                external_violations.append("training_max_sequence_tokens")
            if eval_max_answer_tokens is not None and measurement["answer_token_estimate"] > eval_max_answer_tokens:
                external_violations.append("eval_max_answer_tokens")
            if max_answer_characters is not None and measurement["answer_character_count"] > max_answer_characters:
                external_violations.append("max_answer_characters")
            if external_violations:
                over_budget_record_count += 1
                add_issue(
                    "fatal",
                    "answer_budget_exceeded",
                    "Record exceeds its profile or supplied training/evaluation ceiling.",
                    path=path,
                    line=line,
                    record_id=record_id,
                    context={"profile": profile_name, "violations": sorted(set(external_violations))},
                )

        if metadata.get("answer_format") == "json":
            try:
                json.loads(str(record.get("answer", "")))
            except json.JSONDecodeError:
                add_issue("fatal", "invalid_json_answer", "answer_format=json requires a valid JSON answer.", path=path, line=line, record_id=record_id)
        model_text = "\n".join(str(record.get(field, "")) for field in CURRICULUM_MODEL_TEXT_FIELDS)
        if (
            "file://" in model_text
            or CURRICULUM_ABSOLUTE_PATH_RE.search(model_text)
            or CURRICULUM_RELATIVE_PATH_RE.search(model_text)
            or CURRICULUM_WINDOWS_PATH_RE.search(model_text)
        ):
            add_issue("fatal", "raw_path_in_model_text", "Model-facing text contains a raw filesystem path.", path=path, line=line, record_id=record_id)
        claim_issue = unsupported_answer_claim(str(record.get("answer", "")))
        if claim_issue:
            add_issue("fatal", "unsupported_causal_language_in_answer", claim_issue, path=path, line=line, record_id=record_id)

        object_id = metadata.get("canonical_object_id")
        oracle_fact_id = metadata.get("oracle_fact_id")
        canonical_object = canonical_by_id.get(object_id) if isinstance(object_id, str) else None
        if canonical_object is None:
            add_issue("fatal", "missing_canonical_object", "Record points to no object in canonical_objects.jsonl.", path=path, line=line, record_id=record_id, context={"canonical_object_id": object_id})
        else:
            if oracle_fact_id != object_id:
                add_issue("fatal", "oracle_fact_canonical_object_mismatch", "oracle_fact_id and canonical_object_id must identify the same pre-render fact.", path=path, line=line, record_id=record_id)
            if canonical_object.get("object_type") != family_name:
                add_issue("fatal", "canonical_object_type_mismatch", "Canonical object type does not match record question_family.", path=path, line=line, record_id=record_id)
            for field, expected in (
                ("multiplex_id", expected_multiplex_id),
                ("store_id", expected_store_id),
                ("flist_id", expected_flist_id),
            ):
                if expected is not None and canonical_object.get(field) != expected:
                    add_issue("fatal", f"canonical_{field}_mismatch", f"Canonical object {field} does not match the declared source identity.", path=path, line=line, record_id=record_id)
        for field, expected in (
            ("multiplex_id", expected_multiplex_id),
            ("store_id", expected_store_id),
            ("flist_id", expected_flist_id),
        ):
            if expected is not None and metadata.get(field) != expected:
                add_issue("fatal", f"record_{field}_mismatch", f"Record {field} does not match the declared source identity.", path=path, line=line, record_id=record_id)
        if (book_mode == "tool_call" or stage_name == "stage5_structured_tools") and metadata.get("tool_schema_validated") is not True:
            add_issue("fatal", "tool_schema_not_validated", "Tool-curriculum record lacks live-schema validation attestation.", path=path, line=line, record_id=record_id)

        split_string = str(split)
        if isinstance(object_id, str):
            object_splits[object_id].add(split_string)
        fact_group = metadata.get("oracle_fact_group_id")
        if isinstance(fact_group, str) and fact_group:
            group_splits[("oracle_fact_group_id", fact_group)].add(split_string)
        else:
            add_issue("fatal", "missing_oracle_fact_group_id", "Record is missing oracle_fact_group_id used for split isolation.", path=path, line=line, record_id=record_id)
        for group_key in optional_group_keys:
            group_value = metadata.get(group_key)
            if isinstance(group_value, str) and group_value:
                group_splits[(str(group_key), group_value)].add(split_string)
        text_fingerprint = _curriculum_stable_hash([str(record.get(field, "")) for field in CURRICULUM_MODEL_TEXT_FIELDS])
        normalized = re.sub(r"[^a-z0-9]+", " ", model_text.lower()).strip()
        text_splits[text_fingerprint].add(split_string)
        near_text_splits[_curriculum_stable_hash(normalized)].add(split_string)

        counts_by_split[split_string] += 1
        if isinstance(family_name, str):
            counts_by_family[family_name] += 1
        if isinstance(bucket, str):
            counts_by_bucket[bucket] += 1
        if isinstance(stage_name, str):
            counts_by_stage[stage_name] += 1
        if isinstance(book_mode, str):
            counts_by_book_mode[book_mode] += 1
        if isinstance(profile_name, str):
            counts_by_budget_profile[profile_name] += 1
        provenance = metadata.get("provenance")
        if isinstance(provenance, dict) and isinstance(provenance.get("source"), str):
            counts_by_source[provenance["source"]] += 1

    for object_id, splits in object_splits.items():
        if len(splits) > 1:
            add_issue("fatal", "canonical_object_split_leakage", "Canonical oracle fact appears in multiple splits.", context={"canonical_object_id": object_id, "splits": sorted(splits)})
    for (group_key, group_value), splits in group_splits.items():
        if len(splits) > 1:
            add_issue("fatal", "oracle_group_split_leakage", "Oracle grouping key appears in multiple splits.", context={"group_key": group_key, "group_value": group_value, "splits": sorted(splits)})
    for fingerprint, splits in text_splits.items():
        if len(splits) > 1:
            add_issue("fatal", "exact_duplicate_cross_split", "Exact model-facing text appears in multiple splits.", context={"fingerprint": fingerprint, "splits": sorted(splits)})
    for fingerprint, splits in near_text_splits.items():
        if len(splits) > 1:
            add_issue("fatal", "near_duplicate_cross_split", "Normalized model-facing text appears in multiple splits.", context={"fingerprint": fingerprint, "splits": sorted(splits)})

    actual_counts = {
        "record_count_by_split": dict(sorted(counts_by_split.items())),
        "record_count_by_mixture_bucket": dict(sorted(counts_by_bucket.items())),
        "record_count_by_question_family": dict(sorted(counts_by_family.items())),
    }
    if isinstance(build_profile, dict):
        expected_split_counts = build_profile.get("split_counts")
        if expected_split_counts != actual_counts["record_count_by_split"]:
            add_issue(
                "fatal",
                "build_profile_split_count_mismatch",
                "Selected split counts do not match the manifest build profile.",
                path=manifest_path,
                context={"expected": expected_split_counts, "actual": actual_counts["record_count_by_split"]},
            )
        family_minimum = int(build_profile["minimum_selected_per_required_family"])
        underfilled_families = {
            name: int(counts_by_family.get(name, 0))
            for name in families
            if int(counts_by_family.get(name, 0)) < family_minimum
        }
        if underfilled_families:
            add_issue(
                "fatal",
                "required_question_family_underfill",
                "One or more plan-required question families are below the build-profile minimum.",
                context={"minimum": family_minimum, "families": underfilled_families},
            )
    for field, actual in actual_counts.items():
        if manifest.get(field) != actual:
            add_issue("fatal", f"manifest_{field}_mismatch", f"Manifest {field} does not match selected rows.", path=manifest_path, context={"manifest": manifest.get(field), "actual": actual})
    for report_name in ("audit_report.json",):
        native_report = native_reports[report_name]
        for field in ("selected_record_count", "record_count_by_split", "record_count_by_mixture_bucket", "question_family_counts"):
            expected = len(records) if field == "selected_record_count" else actual_counts.get("record_count_by_question_family" if field == "question_family_counts" else field)
            if native_report.get(field) != expected:
                add_issue("fatal", "native_audit_count_mismatch", f"Native audit {field} does not match selected rows.", path=dataset_dir / report_name, context={"field": field, "report": native_report.get(field), "actual": expected})
    if native_leakage.get("selected_record_count") != len(records):
        add_issue("fatal", "native_leakage_record_count_mismatch", "Native leakage report selected_record_count does not match selected rows.", path=dataset_dir / "leakage_report.json")

    sorted_records = [
        item["record"]
        for item in sorted(
            records,
            key=lambda entry: str(entry["record"].get("metadata", {}).get("record_id", "")),
        )
    ]
    observed_content_hash = _curriculum_stable_hash(sorted_records)
    if manifest.get("content_hash") != observed_content_hash:
        add_issue(
            "fatal",
            "selected_content_hash_mismatch",
            "Selected split records no longer match the manifest content_hash.",
            path=manifest_path,
            context={"manifest": manifest.get("content_hash"), "actual": observed_content_hash},
        )

    total = len(records)
    target_weights = {
        str(key): float(value)
        for key, value in (
            plan.get("mixture", {}).get("content_buckets", {})
            if isinstance(plan.get("mixture"), dict)
            else {}
        ).items()
    }
    effective_policy = mixture_underfill_policy or "fatal"
    effective_min_records = mixture_contract_min_records if mixture_contract_min_records is not None else 0
    effective_absolute_tolerance = mixture_absolute_underfill_tolerance if mixture_absolute_underfill_tolerance is not None else 0.0
    effective_relative_tolerance = mixture_relative_underfill_tolerance if mixture_relative_underfill_tolerance is not None else 0.0
    mixture_report = build_mixture_contract_report(
        counts_by_bucket,
        total_records=total,
        target_weights=target_weights,
        minimum_records=effective_min_records,
        absolute_underfill_tolerance=effective_absolute_tolerance,
        relative_underfill_tolerance=effective_relative_tolerance,
        underfill_policy=effective_policy,
    )
    mixture_report["manifest_contract_present"] = True
    for bucket in mixture_report["material_underfilled_buckets"]:
        if effective_policy == "ignore":
            continue
        add_issue(
            "fatal" if effective_policy == "fatal" else "warning",
            "mixture_bucket_materially_underfilled",
            "V3 mixture bucket is materially below its plan target.",
            context={"bucket": bucket, **mixture_report["buckets"][bucket]},
        )
    if mixture_tolerance is not None and total >= coverage_min_records:
        for bucket, bucket_report in mixture_report["buckets"].items():
            if abs(float(bucket_report["delta_share"])) > mixture_tolerance:
                add_issue("warning", "mixture_bucket_outside_tolerance", "V3 mixture bucket share is outside the requested symmetric tolerance.", context={"bucket": bucket, "delta_share": bucket_report["delta_share"], "tolerance": mixture_tolerance})

    missing_families = sorted(set(families) - set(counts_by_family))
    missing_buckets = sorted(set(target_weights) - set(counts_by_bucket))
    if total >= coverage_min_records and missing_families:
        add_issue("fatal" if strict_coverage else "warning", "missing_recommended_view_types", "Dataset is missing plan-required question families.", context={"missing_question_families": missing_families})
    if total >= coverage_min_records and missing_buckets:
        add_issue("fatal" if strict_coverage else "warning", "missing_mixture_buckets", "Dataset is missing plan-defined mixture buckets.", context={"missing_buckets": missing_buckets})

    issue_dicts = [issue.to_dict() for issue in issues]
    fatal_count = sum(issue.severity == "fatal" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    report = {
        "schema_version": CURRICULUM_AUDIT_SCHEMA_VERSION,
        "dataset_schema_version": CURRICULUM_DATASET_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "dataset_dir": str(dataset_dir),
        "passed": fatal_count == 0 and (warning_count == 0 or not fail_on_warnings),
        "fatal_error_count": fatal_count,
        "warning_count": warning_count,
        "record_count": total,
        "canonical_object_count": len(canonical_by_id),
        "plan_id": manifest.get("plan_id"),
        "plan_hash": declared_plan_hash,
        "content_hash": observed_content_hash,
        "native_reports": {
            name: {
                "passed": payload.get("passed"),
                "plan_hash": payload.get("plan_hash"),
            }
            for name, payload in native_reports.items()
        },
        "record_count_by_split": dict(sorted(counts_by_split.items())),
        "record_count_by_view_type": dict(sorted(counts_by_family.items())),
        "record_count_by_question_family": dict(sorted(counts_by_family.items())),
        "record_count_by_mixture_bucket": dict(sorted(counts_by_bucket.items())),
        "record_count_by_curriculum_stage": dict(sorted(counts_by_stage.items())),
        "record_count_by_context_mode": dict(sorted(counts_by_book_mode.items())),
        "record_count_by_book_mode": dict(sorted(counts_by_book_mode.items())),
        "record_count_by_budget_profile": dict(sorted(counts_by_budget_profile.items())),
        "record_count_by_answer_budget_action": dict(sorted(counts_by_answer_budget_action.items())),
        "record_count_by_source": dict(sorted(counts_by_source.items())),
        "answer_budget_contract": {
            "context_budget_profiles": profiles,
            "training_max_sequence_tokens": training_max_sequence_tokens,
            "eval_max_answer_tokens": eval_max_answer_tokens,
            "max_answer_characters": max_answer_characters,
        },
        "answer_budget_report": {
            "manifest_contract_present": True,
            "record_count_checked": total,
            "over_budget_record_count": over_budget_record_count,
            "missing_record_budget_metadata_count": missing_budget_count,
            "max_answer_token_estimate": max_answer_token_estimate,
            "max_training_sequence_token_estimate": max_total_token_estimate,
            "max_answer_character_count": max_answer_character_count,
            "record_count_by_action": dict(sorted(counts_by_answer_budget_action.items())),
        },
        "mixture_contract": mixture_report["contract"],
        "mixture_report": mixture_report,
        "missing_recommended_view_types": missing_families,
        "issues": issue_dicts[:max_issues],
        "truncated_issue_count": max(0, len(issue_dicts) - max_issues),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def audit_pretrajectory_sft_dataset(
    dataset_dir: Path,
    *,
    output_path: Path | None = None,
    mixture_tolerance: float | None = None,
    coverage_min_records: int = 1000,
    strict_coverage: bool = False,
    fail_on_warnings: bool = False,
    min_records: int = 0,
    max_issues: int = 200,
    training_max_sequence_tokens: int | None = None,
    eval_max_answer_tokens: int | None = None,
    max_answer_characters: int | None = None,
    mixture_contract_min_records: int | None = None,
    mixture_absolute_underfill_tolerance: float | None = None,
    mixture_relative_underfill_tolerance: float | None = None,
    mixture_underfill_policy: str | None = None,
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
    if manifest.get("dataset_schema_version") == CURRICULUM_DATASET_SCHEMA_VERSION:
        return _audit_curriculum_v3_dataset(
            dataset_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            issues=issues,
            add_issue=add_issue,
            output_path=output_path,
            mixture_tolerance=mixture_tolerance,
            coverage_min_records=coverage_min_records,
            strict_coverage=strict_coverage,
            fail_on_warnings=fail_on_warnings,
            min_records=min_records,
            max_issues=max_issues,
            training_max_sequence_tokens=training_max_sequence_tokens,
            eval_max_answer_tokens=eval_max_answer_tokens,
            max_answer_characters=max_answer_characters,
            mixture_contract_min_records=mixture_contract_min_records,
            mixture_absolute_underfill_tolerance=mixture_absolute_underfill_tolerance,
            mixture_relative_underfill_tolerance=mixture_relative_underfill_tolerance,
            mixture_underfill_policy=mixture_underfill_policy,
        )
    validation = read_json_file(validation_path, add_issue)
    canonical_rows = read_jsonl_file(canonical_path, add_issue)

    manifest_budget_payload = manifest.get("answer_budget_contract")
    has_manifest_budget_contract = isinstance(manifest_budget_payload, dict)
    if has_manifest_budget_contract:
        declared_budget_contract = answer_budget_contract_from_dict(manifest_budget_payload)
        if declared_budget_contract.contract_version != ANSWER_BUDGET_CONTRACT_VERSION:
            add_issue(
                "fatal",
                "answer_budget_contract_version_mismatch",
                "Manifest answer-budget contract version is unsupported.",
                path=manifest_path,
                context={
                    "expected": ANSWER_BUDGET_CONTRACT_VERSION,
                    "actual": declared_budget_contract.contract_version,
                },
            )
        if declared_budget_contract.token_estimator != PORTABLE_TOKEN_ESTIMATOR:
            add_issue(
                "fatal",
                "answer_budget_token_estimator_mismatch",
                "Manifest answer-budget token estimator is unsupported by this auditor.",
                path=manifest_path,
                context={"expected": PORTABLE_TOKEN_ESTIMATOR, "actual": declared_budget_contract.token_estimator},
            )
    else:
        declared_budget_contract = AnswerBudgetContract()
        legacy_severity = "fatal" if manifest.get("schema_version") == SCHEMA_VERSION else "warning"
        add_issue(
            legacy_severity,
            "missing_answer_budget_contract",
            "Dataset manifest has no answer-budget contract; rows are audited against current defaults.",
            path=manifest_path,
            context={"defaults": declared_budget_contract.to_dict()},
        )

    effective_budget_contract = AnswerBudgetContract(
        training_max_sequence_tokens=(
            training_max_sequence_tokens
            if training_max_sequence_tokens is not None
            else declared_budget_contract.training_max_sequence_tokens
        ),
        eval_max_answer_tokens=(
            eval_max_answer_tokens
            if eval_max_answer_tokens is not None
            else declared_budget_contract.eval_max_answer_tokens
        ),
        max_answer_characters=(
            max_answer_characters
            if max_answer_characters is not None
            else declared_budget_contract.max_answer_characters
        ),
        chat_template_overhead_tokens=declared_budget_contract.chat_template_overhead_tokens,
    )

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
    counts_by_answer_budget_action: Counter[str] = Counter()
    over_budget_record_count = 0
    max_answer_token_estimate = 0
    max_training_sequence_token_estimate = 0
    max_answer_character_count = 0
    missing_answer_budget_metadata_count = 0
    missing_answer_budget_metadata_sample_ids: list[str] = []
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

        budget_measurements = answer_budget_measurements(record, effective_budget_contract)
        max_answer_token_estimate = max(max_answer_token_estimate, int(budget_measurements["answer_token_estimate"]))
        max_training_sequence_token_estimate = max(
            max_training_sequence_token_estimate,
            int(budget_measurements["training_sequence_token_estimate"]),
        )
        max_answer_character_count = max(max_answer_character_count, int(budget_measurements["answer_character_count"]))
        if budget_measurements["violations"]:
            over_budget_record_count += 1
            add_issue(
                "fatal",
                "answer_budget_exceeded",
                "Record exceeds the declared training/eval answer-budget contract.",
                path=path,
                line=line,
                record_id=record_id,
                context={
                    **budget_measurements,
                    "effective_contract": effective_budget_contract.to_dict(),
                },
            )
        answer_budget = metadata.get("answer_budget")
        if not isinstance(answer_budget, dict):
            missing_answer_budget_metadata_count += 1
            if record_id and len(missing_answer_budget_metadata_sample_ids) < 10:
                missing_answer_budget_metadata_sample_ids.append(record_id)
            budget_action = "missing"
        else:
            budget_action = str(answer_budget.get("action", "unspecified"))
            if budget_action not in {"unchanged", "compacted"}:
                add_issue(
                    "fatal",
                    "invalid_answer_budget_action",
                    "Record answer-budget metadata has an invalid action.",
                    path=path,
                    line=line,
                    record_id=record_id,
                    context={"action": budget_action},
                )
            for key in ("answer_token_estimate", "training_sequence_token_estimate", "answer_character_count"):
                if answer_budget.get(key) != budget_measurements[key]:
                    add_issue(
                        "fatal",
                        "answer_budget_measurement_mismatch",
                        "Stored answer-budget measurement does not match the record.",
                        path=path,
                        line=line,
                        record_id=record_id,
                        context={"field": key, "stored": answer_budget.get(key), "actual": budget_measurements[key]},
                    )
            if answer_budget.get("contract_version") != declared_budget_contract.contract_version:
                add_issue(
                    "fatal",
                    "record_answer_budget_contract_mismatch",
                    "Record answer-budget contract version does not match the manifest.",
                    path=path,
                    line=line,
                    record_id=record_id,
                )
            if answer_budget.get("token_estimator") != declared_budget_contract.token_estimator:
                add_issue(
                    "fatal",
                    "record_answer_budget_estimator_mismatch",
                    "Record answer-budget token estimator does not match the manifest.",
                    path=path,
                    line=line,
                    record_id=record_id,
                )
            expected_limits = {
                "training_max_sequence_tokens": declared_budget_contract.training_max_sequence_tokens,
                "eval_max_answer_tokens": declared_budget_contract.eval_max_answer_tokens,
                "max_answer_characters": declared_budget_contract.max_answer_characters,
                "chat_template_overhead_tokens": declared_budget_contract.chat_template_overhead_tokens,
            }
            if answer_budget.get("limits") != expected_limits:
                add_issue(
                    "fatal",
                    "record_answer_budget_limits_mismatch",
                    "Record answer-budget limits do not match the manifest.",
                    path=path,
                    line=line,
                    record_id=record_id,
                    context={"expected": expected_limits, "actual": answer_budget.get("limits")},
                )
        counts_by_answer_budget_action[budget_action] += 1

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

    if missing_answer_budget_metadata_count:
        missing_metadata_severity = "fatal" if has_manifest_budget_contract else "warning"
        add_issue(
            missing_metadata_severity,
            "missing_record_answer_budget_metadata",
            "Records are missing per-record answer-budget measurements.",
            context={
                "count": missing_answer_budget_metadata_count,
                "sample_record_ids": missing_answer_budget_metadata_sample_ids,
            },
        )

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
        "record_count_by_answer_budget_action": dict(sorted(counts_by_answer_budget_action.items())),
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
    manifest_mixture_contract = manifest.get("mixture_contract")
    has_manifest_mixture_contract = isinstance(manifest_mixture_contract, dict)
    if not has_manifest_mixture_contract:
        missing_mixture_severity = "fatal" if manifest.get("schema_version") == SCHEMA_VERSION else "warning"
        add_issue(
            missing_mixture_severity,
            "missing_mixture_contract",
            "Dataset manifest has no explicit target-mixture underfill contract; current defaults are used.",
            path=manifest_path,
        )
    manifest_mixture_contract = manifest_mixture_contract if has_manifest_mixture_contract else {}
    target_weights = manifest_mixture_contract.get("target_weights")
    if not isinstance(target_weights, dict):
        target_weights = DEFAULT_BUCKET_WEIGHTS
    elif target_weights != DEFAULT_BUCKET_WEIGHTS:
        add_issue(
            "fatal",
            "mixture_target_weights_mismatch",
            "Manifest mixture target weights do not match the current curriculum contract.",
            path=manifest_path,
            context={"expected": DEFAULT_BUCKET_WEIGHTS, "actual": target_weights},
        )
    effective_mixture_policy = str(
        mixture_underfill_policy
        if mixture_underfill_policy is not None
        else manifest_mixture_contract.get("underfill_policy", DEFAULT_MIXTURE_UNDERFILL_POLICY)
    )
    if effective_mixture_policy not in MIXTURE_UNDERFILL_POLICIES:
        add_issue(
            "fatal",
            "invalid_mixture_underfill_policy",
            "Mixture underfill policy is invalid.",
            path=manifest_path,
            context={"policy": effective_mixture_policy},
        )
        effective_mixture_policy = DEFAULT_MIXTURE_UNDERFILL_POLICY
    effective_mixture_min_records = int(
        mixture_contract_min_records
        if mixture_contract_min_records is not None
        else manifest_mixture_contract.get("minimum_records", DEFAULT_MIXTURE_CONTRACT_MIN_RECORDS)
    )
    effective_absolute_underfill_tolerance = float(
        mixture_absolute_underfill_tolerance
        if mixture_absolute_underfill_tolerance is not None
        else manifest_mixture_contract.get(
            "absolute_underfill_tolerance",
            DEFAULT_MIXTURE_ABSOLUTE_UNDERFILL_TOLERANCE,
        )
    )
    effective_relative_underfill_tolerance = float(
        mixture_relative_underfill_tolerance
        if mixture_relative_underfill_tolerance is not None
        else manifest_mixture_contract.get(
            "relative_underfill_tolerance",
            DEFAULT_MIXTURE_RELATIVE_UNDERFILL_TOLERANCE,
        )
    )
    mixture_contract_report = build_mixture_contract_report(
        counts_by_bucket,
        total_records=total,
        target_weights={str(key): float(value) for key, value in target_weights.items()},
        minimum_records=effective_mixture_min_records,
        absolute_underfill_tolerance=effective_absolute_underfill_tolerance,
        relative_underfill_tolerance=effective_relative_underfill_tolerance,
        underfill_policy=effective_mixture_policy,
    )
    mixture_contract_report["manifest_contract_present"] = has_manifest_mixture_contract
    for bucket in mixture_contract_report["material_underfilled_buckets"]:
        if effective_mixture_policy == "ignore":
            continue
        severity = "fatal" if effective_mixture_policy == "fatal" else "warning"
        add_issue(
            severity,
            "mixture_bucket_materially_underfilled",
            "Mixture bucket is materially below its target share.",
            context={
                "bucket": bucket,
                **mixture_contract_report["buckets"][bucket],
                "contract": mixture_contract_report["contract"],
            },
        )
    if mixture_tolerance is not None:
        for bucket, bucket_report in mixture_contract_report["buckets"].items():
            delta = float(bucket_report["delta_share"])
            if total >= coverage_min_records and abs(delta) > mixture_tolerance:
                add_issue(
                    "warning",
                    "mixture_bucket_outside_tolerance",
                    "Mixture bucket share is outside the legacy symmetric tolerance.",
                    context={
                        "bucket": bucket,
                        "target_share": bucket_report["target_share"],
                        "actual_share": bucket_report["actual_share"],
                        "delta_share": delta,
                        "tolerance": mixture_tolerance,
                    },
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
        "schema_version": "pretrajectory-sft-audit-v2",
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
        "record_count_by_answer_budget_action": dict(sorted(counts_by_answer_budget_action.items())),
        "record_count_by_source": dict(sorted(counts_by_source.items())),
        "answer_budget_contract": effective_budget_contract.to_dict(),
        "answer_budget_report": {
            "manifest_contract_present": has_manifest_budget_contract,
            "record_count_checked": total,
            "over_budget_record_count": over_budget_record_count,
            "missing_record_budget_metadata_count": missing_answer_budget_metadata_count,
            "max_answer_token_estimate": max_answer_token_estimate,
            "max_training_sequence_token_estimate": max_training_sequence_token_estimate,
            "max_answer_character_count": max_answer_character_count,
            "record_count_by_action": dict(sorted(counts_by_answer_budget_action.items())),
        },
        "mixture_contract": mixture_contract_report["contract"],
        "mixture_report": mixture_contract_report,
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
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Audit report path. Defaults to DATASET_DIR/audit_report.json for legacy datasets "
            "and DATASET_DIR/audit_report_contract_v3.json for plan-driven v3 artifacts."
        ),
    )
    parser.add_argument(
        "--mixture-tolerance",
        type=float,
        default=None,
        help="Optional legacy symmetric share tolerance; the underfill contract is enforced separately.",
    )
    parser.add_argument("--mixture-contract-min-records", type=int, default=None)
    parser.add_argument("--mixture-absolute-underfill-tolerance", type=float, default=None)
    parser.add_argument("--mixture-relative-underfill-tolerance", type=float, default=None)
    parser.add_argument("--mixture-underfill-policy", choices=MIXTURE_UNDERFILL_POLICIES, default=None)
    parser.add_argument("--training-max-sequence-tokens", type=int, default=None)
    parser.add_argument("--eval-max-answer-tokens", type=int, default=None)
    parser.add_argument("--max-answer-characters", type=int, default=None)
    parser.add_argument("--coverage-min-records", type=int, default=1000)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument("--min-records", type=int, default=0)
    parser.add_argument("--max-issues", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="Print the full report instead of a compact summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.out
    if output_path is None:
        output_name = "audit_report.json"
        try:
            manifest_preview = json.loads(
                (args.dataset_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            manifest_preview = {}
        if (
            isinstance(manifest_preview, dict)
            and manifest_preview.get("dataset_schema_version") == CURRICULUM_DATASET_SCHEMA_VERSION
        ):
            # audit_report.json is a required native compiler attestation in v3.
            output_name = "audit_report_contract_v3.json"
        output_path = args.dataset_dir / output_name
    report = audit_pretrajectory_sft_dataset(
        args.dataset_dir,
        output_path=output_path,
        mixture_tolerance=args.mixture_tolerance,
        coverage_min_records=args.coverage_min_records,
        strict_coverage=args.strict_coverage,
        fail_on_warnings=args.fail_on_warnings,
        min_records=args.min_records,
        max_issues=args.max_issues,
        training_max_sequence_tokens=args.training_max_sequence_tokens,
        eval_max_answer_tokens=args.eval_max_answer_tokens,
        max_answer_characters=args.max_answer_characters,
        mixture_contract_min_records=args.mixture_contract_min_records,
        mixture_absolute_underfill_tolerance=args.mixture_absolute_underfill_tolerance,
        mixture_relative_underfill_tolerance=args.mixture_relative_underfill_tolerance,
        mixture_underfill_policy=args.mixture_underfill_policy,
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
            "record_count_by_answer_budget_action": report["record_count_by_answer_budget_action"],
            "over_budget_record_count": report["answer_budget_report"]["over_budget_record_count"],
            "material_underfilled_mixture_buckets": report["mixture_report"]["material_underfilled_buckets"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    if report["fatal_error_count"] or (args.fail_on_warnings and report["warning_count"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
