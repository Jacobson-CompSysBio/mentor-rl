#!/usr/bin/env python3
"""Build pre-trajectory SFT records for MENTOR-RL.

The generated dataset is intentionally upstream of trajectory SFT/DPO. It
teaches graph topology, MENTOR-EV module context, RWR-LOE rank context, and
calibrated interpretation from deterministic artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.tools import MultiplexIndex
from scripts.build_rwr_loe_corpus import load_seed_rank_cache, load_rank_cache_context_from_dir


SCHEMA_VERSION = "pretrajectory-sft-v2"
GRAPH_TOPOLOGY_SOURCE = "MENTOR_FULL_BRAIN_MULTIPLEX"
MENTOR_EV_SOURCE = "MENTOR_GW_DENDROGRAM"
RWR_LOE_SOURCE = "RWR_LOE_FULL_BRAIN"
MIXED_SOURCE = "MIXED_MODULE_CORPUS_FULL_BRAIN"
DEFAULT_GRAPH_VERSION = "mentor-rl-multiplex-store-v2"
DEFAULT_GRAPH_FLIST_PATH = REPO_ROOT / "data" / "full_brain_flist.tsv"
DEFAULT_MIXED_CORPUS_DIR = REPO_ROOT / "data" / "module_corpus_full_brain_mixed"
DEFAULT_STORE_MANIFEST_PATH = REPO_ROOT / "data" / "runtime" / "full_brain_multiplex_store" / "manifest.json"
DEFAULT_RANK_CACHE_ROOT = REPO_ROOT / "data" / "runtime" / "rwr_loe_full_brain_rank_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "pretrajectory_sft" / "v4_spec"
SPLITS = ("train", "val", "test")
TARGET_SPLIT_COUNTS = {"train": 75_000, "val": 5_000, "test": 5_000}
PATCHCHECK_SPLIT_COUNTS = {"train": 10_000, "val": 1_000, "test": 1_000}
FULL_1M_SPLIT_COUNTS = {"train": 1_000_000, "val": 50_000, "test": 50_000}
CONTEXT_MODES = ("no_context", "open_book_context", "tool_observation")
DEFAULT_CONTEXT_MODES = ("no_context",)
DEFAULT_BUCKET_WEIGHTS = {
    "entity_schema_grounding": 0.08,
    "multiplex_layer_metadata": 0.10,
    "local_topology": 0.12,
    "shortest_paths": 0.08,
    "subgraphs_components_degree": 0.10,
    "rwr_vector_lookup": 0.15,
    "module_set_algebra": 0.20,
    "calibration_global_context": 0.10,
    "tool_observation_state_updates": 0.07,
}
CURRICULUM_STAGE_BY_BUCKET = {
    "entity_schema_grounding": "stage1_entity_schema",
    "multiplex_layer_metadata": "stage1_entity_schema",
    "local_topology": "stage2_topology_priors",
    "shortest_paths": "stage2_topology_priors",
    "subgraphs_components_degree": "stage2_topology_priors",
    "rwr_vector_lookup": "stage3_open_book_vectors",
    "module_set_algebra": "stage4_module_world_model",
    "calibration_global_context": "stage4_module_world_model",
    "tool_observation_state_updates": "stage5_structured_tools",
}
CURRICULUM_STAGES = (
    "stage1_entity_schema",
    "stage2_topology_priors",
    "stage3_open_book_vectors",
    "stage4_module_world_model",
    "stage5_structured_tools",
    "stage6_blend",
)
SYSTEM_PROMPTS = {
    "entity_schema": (
        "You are Mentor-RL. Answer exact questions about graph provenance, "
        "canonical gene IDs, aliases, layer tags, and source schemas. Ensembl "
        "IDs are canonical; symbols are aliases only when explicitly mapped. "
        "Reject ambiguous aliases instead of guessing."
    ),
    "module_algebra": (
        "You are Mentor-RL. Answer module-membership and set-algebra questions "
        "with exact genes, counts, and provenance from the provided MENTOR-EV "
        "or RWR-LOE source. Do not turn module membership into causal proof."
    ),
    "structured_tool": (
        "You are Mentor-RL. Choose model-facing graph/RWR/module tools and "
        "return schema-valid, evidence-backed state updates. Do not expose raw "
        "filesystem paths or hidden targets in downstream trajectory prompts."
    ),
    "topology": (
        "You are Mentor-RL. Answer only from the provided graph version and "
        "return exact graph facts without inventing edges, paths, or layers. "
        "Use internal reasoning to check IDs and layers, then give a concise final answer. "
        "Use literal gene IDs and layer names; never use numeric layer placeholders."
    ),
    "edge_topology": (
        "You are Mentor-RL. For edge-existence questions, answer Yes or No first, "
        "then give only the recorded edge, weight, and supporting layer facts from "
        "the specified graph version. If no edge is recorded, say that directly; "
        "do not infer biological absence. Use internal reasoning to verify the pair, "
        "then keep the final answer concise."
    ),
    "neighbor_topology": (
        "You are Mentor-RL. For neighbor and layer-membership questions, return "
        "only the requested genes and real layer names from the graph. Do not "
        "repeat the query gene as its own neighbor unless the graph explicitly records it. "
        "Use internal reasoning to count neighbors before the final answer."
    ),
    "path_topology": (
        "You are Mentor-RL. For shortest-path and path-decomposition questions, "
        "return exact path nodes, edge endpoints, hop counts, and real supporting "
        "layer names. Use internal reasoning to verify the path before answering. "
        "Do not invent numbered layer placeholders."
    ),
    "rwr": (
        "You are Mentor-RL. Interpret RWR-LOE rank evidence faithfully and avoid "
        "unsupported causal or direct-interaction claims. Preserve exact gene IDs, "
        "rank numbers, and caveats from the provided evidence. Use internal reasoning "
        "to check rank/caveat consistency, then answer concisely."
    ),
    "interpretation": (
        "You are Mentor-RL. Give concise, evidence-grounded mechanistic "
        "interpretations with scientific restraint. Keep module claims at the "
        "network/module-support level unless direct evidence is explicitly provided. "
        "Use internal reasoning to separate evidence from hypothesis."
    ),
    "critique": (
        "You are Mentor-RL. Choose or write the scientifically calibrated answer "
        "and briefly explain why. Use internal reasoning to check calibration, but "
        "do not turn graph absence into biological absence."
    ),
}
BUCKET_BY_VIEW = {
    "entity_id_normalization": "entity_schema_grounding",
    "gene_alias_disambiguation": "entity_schema_grounding",
    "graph_schema_provenance": "entity_schema_grounding",
    "layer_tag_metadata": "multiplex_layer_metadata",
    "layer_family_membership": "multiplex_layer_metadata",
    "monoplex_edge_existence": "local_topology",
    "multiplex_edge_existence": "local_topology",
    "direct_neighbors_by_layer": "local_topology",
    "unique_multiplex_neighbors": "local_topology",
    "gene_layer_membership": "multiplex_layer_metadata",
    "nodes_present_by_layer": "multiplex_layer_metadata",
    "monoplex_shortest_path": "shortest_paths",
    "aggregate_multiplex_shortest_path": "shortest_paths",
    "path_layer_decomposition": "shortest_paths",
    "monoplex_vs_multiplex_path_comparison": "shortest_paths",
    "induced_subgraph": "subgraphs_components_degree",
    "connected_components": "subgraphs_components_degree",
    "shared_common_neighbors": "subgraphs_components_degree",
    "degree_hub_bias": "subgraphs_components_degree",
    "rwr_loe_rank_lookup": "rwr_vector_lookup",
    "rwr_loe_rank_comparison": "rwr_vector_lookup",
    "rwr_loe_topk_membership": "rwr_vector_lookup",
    "rwr_neighborhood_interpretation": "rwr_vector_lookup",
    "mentor_ev_module_membership": "module_set_algebra",
    "module_overlap_set_algebra": "module_set_algebra",
    "module_containment_set_algebra": "module_set_algebra",
    "module_source_distinction": "module_set_algebra",
    "module_cohesion_summary": "calibration_global_context",
    "no_edge_no_path_calibration": "calibration_global_context",
    "layer_specific_claim_calibration": "calibration_global_context",
    "calibration_negative_null_module": "calibration_global_context",
    "closed_book_module_qa": "module_set_algebra",
    "open_book_module_interpretation": "calibration_global_context",
    "shadow_tool_recovery": "tool_observation_state_updates",
    "counterfactual_ablation": "calibration_global_context",
    "critique_preference_sft": "calibration_global_context",
    "tool_call_choice": "tool_observation_state_updates",
    "structured_state_update": "tool_observation_state_updates",
    "provenance_refusal_raw_cli": "tool_observation_state_updates",
}


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    canonical_object: dict[str, Any]

    @property
    def split(self) -> str:
        return str(self.record["metadata"]["split"])

    @property
    def view_type(self) -> str:
        return str(self.record["metadata"]["view_type"])

    @property
    def bucket(self) -> str:
        return str(self.record["metadata"]["mixture_bucket"])


def deduplicate_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    by_record_id: dict[str, Candidate] = {}
    for candidate in candidates:
        record_id = str(candidate.record["metadata"]["record_id"])
        if record_id not in by_record_id:
            by_record_id[record_id] = candidate
    return list(by_record_id.values())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, payload: Any) -> str:
    return f"{prefix}_{stable_hash(payload)[:16]}"


def stable_order_key(value: Any, *, seed: int) -> str:
    return stable_hash({"seed": seed, "value": value})


def split_for_key(key: Any, *, seed: int) -> str:
    digest = stable_hash({"seed": seed, "split_key": key})
    bucket = int(digest[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def source_for_row(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        source = provenance.get("source")
        if isinstance(source, str) and source:
            return source
    source = row.get("source")
    return source if isinstance(source, str) and source else "unknown"


def module_id_for_row(row: dict[str, Any]) -> str | None:
    module_id = row.get("module_id")
    if isinstance(module_id, str) and module_id:
        return module_id
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        for key in ("source_module_id", "anchor_module_id"):
            value = provenance.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def gene_ids_for_row(row: dict[str, Any]) -> list[str]:
    genes = row.get("gene_ids")
    if isinstance(genes, list):
        return [str(gene) for gene in genes if isinstance(gene, str)]
    visible_inputs = row.get("visible_inputs")
    if isinstance(visible_inputs, dict):
        genes = visible_inputs.get("seed_gene_ids")
        if isinstance(genes, list):
            return [str(gene) for gene in genes if isinstance(gene, str)]
    return []


def target_gene_ids_for_task(row: dict[str, Any]) -> list[str]:
    hidden = row.get("hidden_target")
    if not isinstance(hidden, dict):
        return []
    genes = hidden.get("target_gene_ids")
    return [str(gene) for gene in genes if isinstance(gene, str)] if isinstance(genes, list) else []


def short_list(values: Iterable[str], *, max_items: int = 8) -> str:
    items = [str(value) for value in values]
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + f", and {len(items) - max_items} more"


def source_label(source: str) -> str:
    if source == MENTOR_EV_SOURCE:
        return "MENTOR-EV dendrogram module"
    if source == RWR_LOE_SOURCE:
        return "RWR-LOE module"
    if source == GRAPH_TOPOLOGY_SOURCE:
        return "full-brain multiplex graph"
    if source == MIXED_SOURCE:
        return "mixed MENTOR-EV/RWR-LOE module comparison"
    return source


def graph_version_from_manifest(path: Path | None) -> str:
    if path is None:
        return DEFAULT_GRAPH_VERSION
    manifest = read_json(path)
    value = manifest.get("format_version") or manifest.get("schema_version")
    return str(value) if isinstance(value, str) and value else DEFAULT_GRAPH_VERSION


def split_layer_name(layer: str) -> tuple[str, str]:
    if ":" in layer:
        namespace, family = layer.split(":", 1)
        return namespace, family
    return "unknown", layer


def build_alias_tables(
    *,
    graph_gene_ids: Iterable[str],
    modules: Iterable[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    aliases_by_gene: dict[str, set[str]] = {str(gene): {str(gene)} for gene in graph_gene_ids}
    genes_by_alias: dict[str, set[str]] = defaultdict(set)
    for gene in graph_gene_ids:
        gene_text = str(gene)
        genes_by_alias[gene_text].add(gene_text)

    for module in modules:
        gene_ids = module.get("gene_ids")
        gene_symbols = module.get("gene_symbols")
        if not isinstance(gene_ids, list) or not isinstance(gene_symbols, list):
            continue
        for gene_id, symbol in zip(gene_ids, gene_symbols):
            if not isinstance(gene_id, str) or not isinstance(symbol, str):
                continue
            aliases_by_gene.setdefault(gene_id, {gene_id}).add(symbol)
            genes_by_alias[symbol].add(gene_id)
            genes_by_alias[gene_id].add(gene_id)
    return aliases_by_gene, genes_by_alias


def _record(
    *,
    view_type: str,
    split: str,
    source: str,
    graph_version: str,
    system: str,
    question: str,
    answer: str,
    object_type: str,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Candidate:
    object_id = stable_id(object_type, {"view_type": view_type, "payload": payload})
    record_id = stable_id("sft", {"view_type": view_type, "object_id": object_id, "question": question})
    mixture_bucket = BUCKET_BY_VIEW[view_type]
    curriculum_stage = CURRICULUM_STAGE_BY_BUCKET[mixture_bucket]
    canonical_object = {
        "object_id": object_id,
        "object_type": object_type,
        "source": source,
        "split": split,
        "graph_version": graph_version,
        "payload": payload,
    }
    record_metadata = {
        "record_id": record_id,
        "schema_version": SCHEMA_VERSION,
        "view_type": view_type,
        "mixture_bucket": mixture_bucket,
        "curriculum_stage": curriculum_stage,
        "source": source,
        "split": split,
        "canonical_object_id": object_id,
        "graph_version": graph_version,
    }
    if metadata:
        record_metadata.update(metadata)
    return Candidate(
        record={
            "system": system,
            "question": question,
            "answer": answer,
            "metadata": record_metadata,
        },
        canonical_object=canonical_object,
    )


def _compact_value(value: Any, *, max_list_items: int = 24, max_dict_items: int = 24) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, key in enumerate(sorted(value)):
            if index >= max_dict_items:
                compact["__truncated_key_count__"] = len(value) - max_dict_items
                break
            compact[str(key)] = _compact_value(value[key], max_list_items=max_list_items, max_dict_items=max_dict_items)
        return compact
    if isinstance(value, list):
        compact_list = [_compact_value(item, max_list_items=max_list_items, max_dict_items=max_dict_items) for item in value[:max_list_items]]
        if len(value) > max_list_items:
            compact_list.append({"__truncated_item_count__": len(value) - max_list_items})
        return compact_list
    return value


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for key, value in payload.items():
        if key == "rank_cache_context":
            compact[key] = _compact_value(value, max_list_items=8, max_dict_items=8)
        else:
            compact[key] = _compact_value(value)
    return compact


def _payload_summary_lines(payload: dict[str, Any]) -> list[str]:
    lines = []
    preferred = [
        "module_id",
        "left_module_id",
        "right_module_id",
        "task_id",
        "task_type",
        "layer",
        "layer_family",
        "layer_namespace",
        "source_gene_id",
        "target_gene_id",
        "gene_id",
        "canonical_gene_id",
        "alias",
        "seed_gene_id",
        "candidate_gene_id",
        "rank",
        "left_rank",
        "right_rank",
        "score",
        "size",
        "size_bin",
        "intersection_size",
        "union_size",
        "overlap_jaccard",
        "density",
        "degree",
        "degree_percentile",
    ]
    for key in preferred:
        if key in payload:
            lines.append(f"- {key}: {payload[key]}")
    for key in ("supporting_layers", "layers", "path_gene_ids", "query_gene_ids", "present_gene_ids", "intersection_genes", "violating_genes"):
        value = payload.get(key)
        if isinstance(value, list):
            lines.append(f"- {key}: {short_list([str(item) for item in value], max_items=12)}")
    for key in ("genes", "visible_genes", "target_genes", "missing_target_genes"):
        value = payload.get(key)
        if isinstance(value, list):
            lines.append(f"- {key}: {short_list([str(item) for item in value], max_items=12)}")
    for key in ("neighbors", "top_candidates"):
        value = payload.get(key)
        if isinstance(value, list):
            lines.append(f"- {key}: {json.dumps(_compact_value(value, max_list_items=12), sort_keys=True)}")
    for key in ("neighbor_layer_map", "layer_counts", "components", "edges"):
        value = payload.get(key)
        if isinstance(value, (dict, list)):
            lines.append(f"- {key}: {json.dumps(_compact_value(value, max_list_items=12, max_dict_items=12), sort_keys=True)}")
    if not lines:
        lines.append(f"- payload: {json.dumps(_compact_payload(payload), sort_keys=True)}")
    return lines


def _base_question(candidate: Candidate) -> str:
    record = candidate.record
    payload = candidate.canonical_object.get("payload", {})
    if not isinstance(payload, dict):
        return str(record["question"])
    view_type = str(record["metadata"]["view_type"])
    graph_version = str(record["metadata"]["graph_version"])
    if view_type == "open_book_module_interpretation":
        if payload.get("task_id"):
            return f"Write a cautious evidence-grounded interpretation for task {payload['task_id']}."
        if payload.get("module_id"):
            return f"Write a cautious interpretation of module {payload['module_id']} in graph version {graph_version}."
    if view_type == "shadow_tool_recovery":
        return f"What is the best training interpretation for recovery task {payload.get('task_id', 'this task')}?"
    if view_type == "critique_preference_sft":
        return "Which answer is better, and why?"
    if view_type == "counterfactual_ablation":
        return "How should the interpretation change after the layer ablation?"
    return str(record["question"])


def _tool_name_for_view(view_type: str, source: str) -> str:
    if view_type in {"rwr_loe_rank_lookup", "rwr_loe_rank_comparison", "rwr_loe_topk_membership", "rwr_neighborhood_interpretation"}:
        return "rwr_loe"
    if view_type in {"entity_id_normalization", "gene_alias_disambiguation"}:
        return "resolve_gene_alias"
    if view_type in {"graph_schema_provenance", "layer_tag_metadata", "layer_family_membership"}:
        return "get_graph_schema"
    if view_type in {"mentor_ev_module_membership", "module_overlap_set_algebra", "module_containment_set_algebra", "module_source_distinction", "module_cohesion_summary"}:
        return "query_module_oracle"
    if view_type in {"tool_call_choice", "structured_state_update", "provenance_refusal_raw_cli"}:
        return "choose_next_tool"
    if view_type in {"monoplex_shortest_path", "aggregate_multiplex_shortest_path", "monoplex_vs_multiplex_path_comparison"}:
        return "shortest_paths"
    if view_type == "path_layer_decomposition":
        return "get_path_layer_counts"
    if view_type in {"direct_neighbors_by_layer", "unique_multiplex_neighbors", "monoplex_edge_existence", "multiplex_edge_existence", "shared_common_neighbors"}:
        return "get_neighbors"
    if view_type == "gene_layer_membership":
        return "get_gene_layers"
    if view_type == "nodes_present_by_layer":
        return "get_nodes_by_layer"
    if view_type == "induced_subgraph":
        return "induce_subgraph"
    if view_type == "connected_components":
        return "get_component_summary"
    if view_type == "counterfactual_ablation":
        return "get_layer_ablation"
    if view_type == "degree_hub_bias":
        return "get_layer_stats"
    if source == RWR_LOE_SOURCE:
        return "rwr_loe"
    return "induce_subgraph"


def _tool_arguments_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for key in (
        "module_id",
        "left_module_id",
        "right_module_id",
        "task_id",
        "layer",
        "layer_family",
        "layer_namespace",
        "gene_id",
        "canonical_gene_id",
        "alias",
        "source_gene_id",
        "target_gene_id",
        "seed_gene_id",
        "candidate_gene_id",
        "left_candidate_gene_id",
        "right_candidate_gene_id",
        "path_gene_ids",
        "query_gene_ids",
        "visible_genes",
        "genes",
        "intersection_genes",
        "violating_genes",
        "ablated_layer",
    ):
        if key in payload:
            args[key] = _compact_value(payload[key], max_list_items=16, max_dict_items=16)
    return args


def _question_for_context_mode(candidate: Candidate, context_mode: str) -> str:
    record = candidate.record
    payload = candidate.canonical_object.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    source = str(record["metadata"]["source"])
    graph_version = str(record["metadata"]["graph_version"])
    view_type = str(record["metadata"]["view_type"])
    base_question = _base_question(candidate)
    if context_mode == "no_context":
        return base_question
    if context_mode == "open_book_context":
        lines = [
            "Context:",
            f"- graph_version: {graph_version}",
            f"- source: {source}",
            f"- view_type: {view_type}",
            *_payload_summary_lines(payload),
            "",
            f"Question: {base_question}",
        ]
        return "\n".join(lines)
    if context_mode == "tool_observation":
        tool_name = _tool_name_for_view(view_type, source)
        observation = {
            "tool_name": tool_name,
            "arguments": _tool_arguments_for_payload(payload),
            "result": _compact_payload(payload),
        }
        return (
            "Tool observation:\n"
            f"{json.dumps(observation, sort_keys=True)}\n\n"
            f"Question: {base_question}"
        )
    raise ValueError(f"Unsupported context mode: {context_mode}")


def expand_context_mode_candidates(candidates: list[Candidate], context_modes: tuple[str, ...]) -> list[Candidate]:
    if context_modes == ("no_context",):
        expanded = []
        for candidate in candidates:
            record = dict(candidate.record)
            metadata = dict(record["metadata"])
            metadata["context_mode"] = "no_context"
            record["metadata"] = metadata
            expanded.append(Candidate(record=record, canonical_object=candidate.canonical_object))
        return expanded

    expanded: list[Candidate] = []
    for candidate in candidates:
        metadata = candidate.record["metadata"]
        object_id = str(metadata["canonical_object_id"])
        view_type = str(metadata["view_type"])
        for context_mode in context_modes:
            question = _question_for_context_mode(candidate, context_mode)
            record_metadata = dict(metadata)
            record_metadata["context_mode"] = context_mode
            record_metadata["record_id"] = stable_id(
                "sft",
                {
                    "view_type": view_type,
                    "object_id": object_id,
                    "context_mode": context_mode,
                    "question": question,
                },
            )
            expanded.append(
                Candidate(
                    record={
                        **candidate.record,
                        "question": question,
                        "metadata": record_metadata,
                    },
                    canonical_object=candidate.canonical_object,
                )
            )
    return expanded


def load_sampled_multiplex_index_from_flist(
    flist_path: Path,
    *,
    max_layers: int | None,
    max_edges_per_layer: int | None,
    edgelist_has_headers: bool,
) -> MultiplexIndex:
    layer_graphs: dict[str, nx.Graph] = {}
    aggregate = nx.Graph()
    genes: set[str] = set()
    layer_names: list[str] = []
    with flist_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for layer_index, row in enumerate(reader):
            if max_layers is not None and len(layer_names) >= max_layers:
                break
            if len(row) < 2:
                continue
            edge_path = Path(row[0])
            layer_name = str(row[1])
            graph = nx.Graph()
            edge_count = 0
            with edge_path.open("r", encoding="utf-8") as edge_handle:
                for line_number, line in enumerate(edge_handle, start=1):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    if line_number == 1 and edgelist_has_headers:
                        head = (parts[0].lower(), parts[1].lower())
                        if head in {("source", "target"), ("from", "to"), ("node1", "node2")}:
                            continue
                    source, target = str(parts[0]), str(parts[1])
                    try:
                        weight = float(parts[2]) if len(parts) >= 3 else 1.0
                    except ValueError:
                        weight = 1.0
                    graph.add_edge(source, target, weight=weight)
                    genes.update((source, target))
                    if aggregate.has_edge(source, target):
                        aggregate[source][target]["weight"] += weight
                        aggregate[source][target]["layers"].append(layer_name)
                    else:
                        aggregate.add_edge(source, target, weight=weight, layers=[layer_name])
                    edge_count += 1
                    if max_edges_per_layer is not None and edge_count >= max_edges_per_layer:
                        break
            layer_graphs[layer_name] = graph
            layer_names.append(layer_name)
    return MultiplexIndex(
        layer_graphs=layer_graphs,
        aggregate_graph=aggregate,
        gene_ids=genes,
        layer_names=tuple(layer_names),
    )


def generate_entity_schema_candidates(
    *,
    modules: list[dict[str, Any]],
    graph_gene_ids: Iterable[str],
    graph_version: str,
    seed: int,
    max_alias_examples: int,
) -> list[Candidate]:
    aliases_by_gene, genes_by_alias = build_alias_tables(graph_gene_ids=graph_gene_ids, modules=modules)
    candidates: list[Candidate] = []

    source_payloads = [
        {
            "source_name": GRAPH_TOPOLOGY_SOURCE,
            "source_dir": "data/runtime/full_brain_multiplex_store",
            "role": "versioned full-brain multiplex topology oracle",
        },
        {
            "source_name": MENTOR_EV_SOURCE,
            "source_dir": "data/gw_dendrogram_corpus_full_brain",
            "role": "MENTOR-EV dendrogram module oracle",
        },
        {
            "source_name": RWR_LOE_SOURCE,
            "source_dir": "data/rwr_loe_corpus_full_brain",
            "role": "RWR-LOE rank and module oracle",
        },
    ]
    for payload in source_payloads:
        split = split_for_key(("schema", payload["source_name"]), seed=seed)
        candidates.append(
            _record(
                view_type="graph_schema_provenance",
                split=split,
                source=GRAPH_TOPOLOGY_SOURCE,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["entity_schema"],
                question=f"What is the role of source `{payload['source_name']}` in pre-trajectory SFT?",
                answer=(
                    f"`{payload['source_name']}` is the {payload['role']}. "
                    f"Its source path is `{payload['source_dir']}` and claims must stay versioned to {graph_version}."
                ),
                object_type="graph_schema_provenance",
                payload=payload,
            )
        )

    alias_rows: list[tuple[str, str, bool]] = []
    for gene_id in sorted(aliases_by_gene):
        aliases = sorted(aliases_by_gene[gene_id])
        for alias in aliases:
            is_ambiguous = len(genes_by_alias.get(alias, set())) > 1
            alias_rows.append((gene_id, alias, is_ambiguous))
    for gene_id, alias, is_ambiguous in alias_rows[:max_alias_examples]:
        split = split_for_key(("alias", gene_id, alias), seed=seed)
        if is_ambiguous:
            answer = (
                f"`{alias}` is ambiguous in the available alias table and maps to "
                f"{len(genes_by_alias[alias])} canonical IDs. Do not guess; request disambiguation."
            )
            view_type = "gene_alias_disambiguation"
            payload = {
                "alias": alias,
                "candidate_gene_ids": sorted(genes_by_alias[alias]),
                "is_ambiguous": True,
            }
        else:
            answer = f"`{alias}` resolves to canonical gene ID `{gene_id}`. Use `{gene_id}` as the canonical graph entity."
            view_type = "entity_id_normalization"
            payload = {
                "alias": alias,
                "canonical_gene_id": gene_id,
                "candidate_gene_ids": [gene_id],
                "is_ambiguous": False,
            }
        candidates.append(
            _record(
                view_type=view_type,
                split=split,
                source=GRAPH_TOPOLOGY_SOURCE,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["entity_schema"],
                question=f"Resolve gene alias `{alias}` for graph version {graph_version}.",
                answer=answer,
                object_type="gene_alias_resolution",
                payload=payload,
            )
        )
    return candidates


def generate_layer_metadata_candidates(
    index: MultiplexIndex,
    *,
    graph_version: str,
    seed: int,
    max_layers: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for layer in list(index.layer_names)[:max_layers]:
        namespace, family = split_layer_name(layer)
        graph = index.layer_graphs[layer]
        split = split_for_key(("layer_metadata", layer), seed=seed)
        payload = {
            "layer": layer,
            "layer_namespace": namespace,
            "layer_family": family,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "directionality": "undirected",
        }
        candidates.append(
            _record(
                view_type="layer_tag_metadata",
                split=split,
                source=GRAPH_TOPOLOGY_SOURCE,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["entity_schema"],
                question=f"Parse layer tag `{layer}` and return its namespace, family, directionality, node count, and edge count.",
                answer=(
                    f"`{layer}` has namespace `{namespace}`, family `{family}`, directionality `undirected`, "
                    f"{graph.number_of_nodes()} nodes, and {graph.number_of_edges()} recorded edges."
                ),
                object_type="layer_tag_metadata",
                payload=payload,
            )
        )

        layer_nodes = sorted(str(node) for node in graph.nodes())
        if layer_nodes:
            gene_id = layer_nodes[0]
            candidates.append(
                _record(
                    view_type="layer_family_membership",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["entity_schema"],
                    question=f"Is gene `{gene_id}` present in layer `{layer}`, and what layer family is that?",
                    answer=f"Yes. `{gene_id}` is present in `{layer}`, whose family tag is `{family}`.",
                    object_type="layer_family_membership",
                    payload={
                        "gene_id": gene_id,
                        "layer": layer,
                        "layer_family": family,
                        "layer_namespace": namespace,
                        "has_gene": True,
                    },
                    metadata={"answer_label": "yes"},
                )
            )
    return candidates


def _sorted_edge_rows(graph: nx.Graph) -> list[tuple[str, str, float]]:
    rows = []
    for source, target, data in graph.edges(data=True):
        left, right = sorted((str(source), str(target)))
        rows.append((left, right, float(data.get("weight", 1.0))))
    return sorted(rows)


def _edge_layers(index: MultiplexIndex, source: str, target: str) -> list[str]:
    layers = []
    for layer in index.layer_names:
        graph = index.layer_graphs[layer]
        if graph.has_edge(source, target):
            layers.append(layer)
    return layers


def _first_non_edge(nodes: list[str], graph: nx.Graph) -> tuple[str, str] | None:
    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            if source != target and not graph.has_edge(source, target):
                return source, target
    return None


def _first_path_pair(graph: nx.Graph) -> tuple[str, str, list[str]] | None:
    for component in nx.connected_components(graph):
        nodes = sorted(str(node) for node in component)
        if len(nodes) < 2:
            continue
        source, target = nodes[0], nodes[-1]
        path = [str(node) for node in nx.shortest_path(graph, source, target)]
        return source, target, path
    return None


def _path_edges_with_layers(index: MultiplexIndex, path: list[str]) -> list[dict[str, Any]]:
    edges = []
    for source, target in zip(path, path[1:]):
        edges.append(
            {
                "source_gene_id": source,
                "target_gene_id": target,
                "supporting_layers": _edge_layers(index, source, target),
            }
        )
    return edges


def _layer_counts_for_path(index: MultiplexIndex, path: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for edge in _path_edges_with_layers(index, path):
        for layer in edge["supporting_layers"]:
            counts[layer] += 1
    return dict(sorted(counts.items()))


def _common_neighbors(graph: nx.Graph, source: str, target: str) -> list[str]:
    if source not in graph or target not in graph:
        return []
    return sorted(str(gene) for gene in nx.common_neighbors(graph, source, target))


def _nodes_by_degree(graph: nx.Graph, *, limit: int, reverse: bool = False) -> list[str]:
    rows = [(int(degree), str(node)) for node, degree in graph.degree() if int(degree) > 0]
    if reverse:
        rows.sort(key=lambda item: (-item[0], item[1]))
    else:
        rows.sort(key=lambda item: (item[0], item[1]))
    return [node for _, node in rows[:limit]]


def _non_edge_pairs(nodes: list[str], graph: nx.Graph, *, limit: int) -> list[tuple[str, str]]:
    if len(nodes) < 2 or limit <= 0:
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    jumps = [len(nodes) - 1, max(1, len(nodes) // 2), max(1, len(nodes) // 3), 7, 17, 31]
    for i, source in enumerate(nodes[: max(limit * 6, min(len(nodes), 2000))]):
        for jump in jumps:
            target = nodes[(i + jump) % len(nodes)]
            if source == target or graph.has_edge(source, target):
                continue
            left, right = sorted((source, target))
            pair = (left, right)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= limit:
                return pairs
    fallback_nodes = nodes[: min(len(nodes), 400)]
    for i, source in enumerate(fallback_nodes):
        for target in fallback_nodes[i + 1 :]:
            if graph.has_edge(source, target):
                continue
            pair = (source, target)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= limit:
                return pairs
    return pairs


def _aggregate_edge_rows(index: MultiplexIndex, *, limit: int) -> list[tuple[str, str, list[str]]]:
    rows: list[tuple[str, str, list[str]]] = []
    for source, target, data in index.aggregate_graph.edges(data=True):
        left, right = sorted((str(source), str(target)))
        layers = data.get("layers")
        if not isinstance(layers, list):
            layers = _edge_layers(index, left, right)
        rows.append((left, right, sorted(str(layer) for layer in layers)))
    return sorted(rows)[:limit]


def generate_topology_candidates(
    index: MultiplexIndex,
    *,
    graph_version: str,
    seed: int,
    max_examples_per_layer: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    all_nodes = sorted(index.gene_ids)
    if not all_nodes:
        return candidates
    seen_multiplex_neighbor_genes: set[str] = set()
    seen_gene_layer_genes: set[str] = set()
    max_examples_per_layer = max(1, max_examples_per_layer)

    for layer in index.layer_names:
        graph = index.layer_graphs[layer]
        edges = _sorted_edge_rows(graph)[:max_examples_per_layer]
        layer_nodes = sorted(str(node) for node in graph.nodes())
        absent_nodes = sorted(set(all_nodes) - set(layer_nodes))
        low_degree_nodes = _nodes_by_degree(graph, limit=max_examples_per_layer, reverse=False)
        high_degree_nodes = _nodes_by_degree(graph, limit=max_examples_per_layer, reverse=True)

        for edge_index, (source, target, weight) in enumerate(edges):
            split = split_for_key(("edge", layer, source, target), seed=seed)
            payload = {"layer": layer, "source_gene_id": source, "target_gene_id": target, "weight": weight, "has_edge": True}
            candidates.append(
                _record(
                    view_type="monoplex_edge_existence",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["edge_topology"],
                    question=(
                        f"In graph version {graph_version}, is there an edge between {source} "
                        f"and {target} in layer `{layer}`? If so, return the edge."
                    ),
                    answer=(
                        f"Yes. In `{layer}`, {source} is directly connected to {target} "
                        f"by edge ({source}, {target}) with weight {weight:g}."
                    ),
                    object_type="graph_edge",
                    payload=payload,
                    metadata={"answer_label": "yes"},
                )
            )

            support_layers = _edge_layers(index, source, target)
            candidates.append(
                _record(
                    view_type="multiplex_edge_existence",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["edge_topology"],
                    question=(
                        f"Across the aggregate multiplex in graph version {graph_version}, "
                        f"are {source} and {target} directly connected? List every supporting layer."
                    ),
                    answer=(
                        f"Yes. {source} and {target} are directly connected in these layers: "
                        f"{', '.join(support_layers)}."
                    ),
                    object_type="multiplex_edge",
                    payload={**payload, "supporting_layers": support_layers},
                    metadata={"answer_label": "yes"},
                )
            )

            if edge_index < max_examples_per_layer:
                if "coexpression" in layer.lower():
                    claim_answer = (
                        f"Support in `{layer}` allows a coexpression/network-support claim only. "
                        "It does not justify a direct physical-interaction claim."
                    )
                else:
                    claim_answer = (
                        f"Support in `{layer}` should be described as layer-specific graph support. "
                        "Do not generalize it to unsupported edge types."
                    )
                candidates.append(
                    _record(
                        view_type="layer_specific_claim_calibration",
                        split=split,
                        source=GRAPH_TOPOLOGY_SOURCE,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["critique"],
                        question=(
                            f"{source} and {target} are connected in `{layer}`. What type of claim "
                            "is allowed from this layer-specific support?"
                        ),
                        answer=claim_answer,
                        object_type="layer_claim_calibration",
                        payload={"layer": layer, "source_gene_id": source, "target_gene_id": target},
                    )
                )

        for source in low_degree_nodes:
            split = split_for_key(("neighbors", layer, source), seed=seed)
            neighbors = sorted(str(node) for node in graph.neighbors(source))
            candidates.append(
                _record(
                    view_type="direct_neighbors_by_layer",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["neighbor_topology"],
                    question=f"What are {source}'s direct neighbors in layer `{layer}`?",
                    answer=(
                        f"In `{layer}`, {source} has {len(neighbors)} direct neighbors: "
                        f"{', '.join(neighbors)}."
                    ),
                    object_type="layer_neighbors",
                    payload={"layer": layer, "gene_id": source, "neighbors": neighbors},
                )
            )

            if source not in seen_multiplex_neighbor_genes:
                seen_multiplex_neighbor_genes.add(source)
                neighbor_layers: dict[str, list[str]] = defaultdict(list)
                for support_layer in index.layer_names:
                    support_graph = index.layer_graphs[support_layer]
                    if source in support_graph:
                        for neighbor in support_graph.neighbors(source):
                            neighbor_layers[str(neighbor)].append(support_layer)
                neighbor_map = {gene: layers for gene, layers in sorted(neighbor_layers.items())}
                neighbor_split = split_for_key(("multiplex_neighbors", source), seed=seed)
                candidates.append(
                    _record(
                        view_type="unique_multiplex_neighbors",
                        split=neighbor_split,
                        source=GRAPH_TOPOLOGY_SOURCE,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["neighbor_topology"],
                        question=(
                            f"What are {source}'s unique direct neighbors across the aggregate "
                            "multiplex, and which layers support each neighbor?"
                        ),
                        answer=(
                            f"{source} has {len(neighbor_map)} unique multiplex neighbors. "
                            f"Neighbor-to-layer support: {json.dumps(neighbor_map, sort_keys=True)}."
                        ),
                        object_type="multiplex_neighbors",
                        payload={"gene_id": source, "neighbor_layer_map": neighbor_map},
                    )
                )

            if source not in seen_gene_layer_genes:
                seen_gene_layer_genes.add(source)
                gene_layers = [support_layer for support_layer in index.layer_names if source in index.layer_graphs[support_layer]]
                layer_split = split_for_key(("gene_layer_membership", source), seed=seed)
                candidates.append(
                    _record(
                        view_type="gene_layer_membership",
                        split=layer_split,
                        source=GRAPH_TOPOLOGY_SOURCE,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["neighbor_topology"],
                        question=f"Which layers contain gene {source} in graph version {graph_version}?",
                        answer=f"{source} is present in these layers: {', '.join(gene_layers)}.",
                        object_type="gene_layer_membership",
                        payload={"gene_id": source, "layers": gene_layers},
                    )
                )

            present_query = [source] + neighbors[:2]
            if absent_nodes:
                present_query.append(absent_nodes[len(candidates) % len(absent_nodes)])
            present_query = list(dict.fromkeys(present_query))[:4]
            present_genes = [gene for gene in present_query if gene in graph]
            candidates.append(
                _record(
                    view_type="nodes_present_by_layer",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["neighbor_topology"],
                    question=(
                        f"Which of these genes are present in layer `{layer}`: "
                        f"{', '.join(present_query)}?"
                    ),
                    answer=f"In `{layer}`, the present genes from the query are: {', '.join(present_genes)}.",
                    object_type="nodes_present_by_layer",
                    payload={"layer": layer, "query_gene_ids": present_query, "present_gene_ids": present_genes},
                )
            )

        for source in high_degree_nodes:
            split = split_for_key(("degree", layer, source), seed=seed)
            degree = graph.degree[source]
            degree_values = sorted(dict(graph.degree()).values())
            rank = sum(1 for value in degree_values if value <= degree)
            percentile = rank / len(degree_values) if degree_values else 0.0
            caveat = " Add a hub-bias caveat." if percentile > 0.95 else " No hub-bias caveat is required by a >95th percentile rule."
            candidates.append(
                _record(
                    view_type="degree_hub_bias",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["neighbor_topology"],
                    question=f"What is {source}'s degree in layer `{layer}`, and is it hub-like by a >95th percentile rule?",
                    answer=f"In `{layer}`, {source} has degree {degree} and degree percentile {percentile:.3f}.{caveat}",
                    object_type="degree_hub_bias",
                    payload={"layer": layer, "gene_id": source, "degree": degree, "degree_percentile": percentile},
                )
            )

        for source, target in _non_edge_pairs(layer_nodes, graph, limit=max_examples_per_layer):
            split = split_for_key(("non_edge", layer, source, target), seed=seed)
            candidates.append(
                _record(
                    view_type="monoplex_edge_existence",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["edge_topology"],
                    question=(
                        f"In graph version {graph_version}, is there an edge between {source} "
                        f"and {target} in layer `{layer}`? If so, return the edge."
                    ),
                    answer=(
                        f"No. In `{layer}`, no edge is recorded between {source} and {target} "
                        f"in graph version {graph_version}."
                    ),
                    object_type="graph_edge_absence",
                    payload={
                        "layer": layer,
                        "source_gene_id": source,
                        "target_gene_id": target,
                        "weight": None,
                        "has_edge": False,
                    },
                    metadata={"answer_label": "no"},
                )
            )
            candidates.append(
                _record(
                    view_type="no_edge_no_path_calibration",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["critique"],
                    question=(
                        f"No direct edge is recorded between {source} and {target} in layer `{layer}`. "
                        "What can and cannot be concluded?"
                    ),
                    answer=(
                        f"The safe conclusion is that no {source}-{target} edge is recorded in `{layer}` "
                        f"in graph version {graph_version}. This does not prove there is no biological "
                        "relationship in general."
                    ),
                    object_type="no_edge_calibration",
                    payload={"layer": layer, "source_gene_id": source, "target_gene_id": target},
                )
            )

        for source, target, _weight in edges:
            path = [source, target]
            split = split_for_key(("path", layer, source, target), seed=seed)
            edge_payloads = [
                {"source_gene_id": left, "target_gene_id": right, "layer": layer}
                for left, right in zip(path, path[1:])
            ]
            candidates.append(
                _record(
                    view_type="monoplex_shortest_path",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["path_topology"],
                    question=(
                        f"What is the shortest path between {source} and {target} in layer `{layer}`? "
                        "Return nodes and edges."
                    ),
                    answer=(
                        f"The shortest path in `{layer}` is {' -> '.join(path)} with hop count {len(path) - 1}. "
                        f"Edges: {json.dumps(edge_payloads, sort_keys=True)}."
                    ),
                    object_type="monoplex_shortest_path",
                    payload={"layer": layer, "source_gene_id": source, "target_gene_id": target, "path_gene_ids": path},
                )
            )

            layer_counts = {layer: len(path) - 1}
            candidates.append(
                _record(
                    view_type="path_layer_decomposition",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["path_topology"],
                    question=f"How many path edges in {' -> '.join(path)} are supported by each layer?",
                    answer=f"Path layer counts: {json.dumps(layer_counts, sort_keys=True)}.",
                    object_type="path_layer_decomposition",
                    payload={"path_gene_ids": path, "layer_counts": layer_counts},
                )
            )

        for source in low_degree_nodes:
            neighbors = sorted(str(node) for node in graph.neighbors(source))
            component_nodes = list(dict.fromkeys([source] + neighbors[:3]))
            if component_nodes:
                edge_rows = []
                subgraph = graph.subgraph(component_nodes)
                for left, right, data in _sorted_edge_rows(subgraph):
                    edge_rows.append({"source_gene_id": left, "target_gene_id": right, "layer": layer, "weight": data})
                split = split_for_key(("subgraph", layer, component_nodes), seed=seed)
                candidates.append(
                    _record(
                        view_type="induced_subgraph",
                        split=split,
                        source=GRAPH_TOPOLOGY_SOURCE,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["neighbor_topology"],
                        question=f"Return all recorded edges among {', '.join(component_nodes)} in layer `{layer}`.",
                        answer=f"The induced subgraph in `{layer}` has {len(edge_rows)} edges: {json.dumps(edge_rows, sort_keys=True)}.",
                        object_type="induced_subgraph",
                        payload={"layer": layer, "query_gene_ids": component_nodes, "edges": edge_rows},
                    )
                )

                component_payload = {
                    str(index): sorted(str(node) for node in component)
                    for index, component in enumerate(nx.connected_components(subgraph), start=1)
                }
                candidates.append(
                    _record(
                        view_type="connected_components",
                        split=split,
                        source=GRAPH_TOPOLOGY_SOURCE,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["neighbor_topology"],
                        question=f"Do {', '.join(component_nodes)} form one connected component in `{layer}`?",
                        answer=(
                            "Yes, they form one connected component."
                            if len(component_payload) == 1
                            else f"No. Component membership: {json.dumps(component_payload, sort_keys=True)}."
                        ),
                        object_type="connected_components",
                        payload={"layer": layer, "query_gene_ids": component_nodes, "components": component_payload},
                    )
                )

            if len(neighbors) >= 2:
                left, right = sorted((neighbors[0], neighbors[1]))
                common = _common_neighbors(graph, left, right)
                if common:
                    split = split_for_key(("common", layer, left, right), seed=seed)
                    candidates.append(
                        _record(
                            view_type="shared_common_neighbors",
                            split=split,
                            source=GRAPH_TOPOLOGY_SOURCE,
                            graph_version=graph_version,
                            system=SYSTEM_PROMPTS["neighbor_topology"],
                            question=f"What direct neighbors are shared by {left} and {right} in layer `{layer}`?",
                            answer=f"In `{layer}`, {left} and {right} share these direct neighbors: {', '.join(common)}.",
                            object_type="shared_common_neighbors",
                            payload={"layer": layer, "source_gene_id": left, "target_gene_id": right, "common_neighbors": common},
                        )
                    )

        path_pair = _first_path_pair(graph)
        if path_pair is not None:
            source, target, path = path_pair
            if len(path) > 2:
                split = split_for_key(("path", layer, source, target), seed=seed)
                edge_payloads = [
                    {"source_gene_id": left, "target_gene_id": right, "layer": layer}
                    for left, right in zip(path, path[1:])
                ]
                candidates.append(
                    _record(
                        view_type="monoplex_shortest_path",
                        split=split,
                        source=GRAPH_TOPOLOGY_SOURCE,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["path_topology"],
                        question=(
                            f"What is the shortest path between {source} and {target} in layer `{layer}`? "
                            "Return nodes and edges."
                        ),
                        answer=(
                            f"The shortest path in `{layer}` is {' -> '.join(path)} with hop count {len(path) - 1}. "
                            f"Edges: {json.dumps(edge_payloads, sort_keys=True)}."
                        ),
                        object_type="monoplex_shortest_path",
                        payload={"layer": layer, "source_gene_id": source, "target_gene_id": target, "path_gene_ids": path},
                    )
                )
                layer_counts = {layer: len(path) - 1}
                candidates.append(
                    _record(
                        view_type="path_layer_decomposition",
                        split=split,
                        source=GRAPH_TOPOLOGY_SOURCE,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["path_topology"],
                        question=f"How many path edges in {' -> '.join(path)} are supported by each layer?",
                        answer=f"Path layer counts: {json.dumps(layer_counts, sort_keys=True)}.",
                        object_type="path_layer_decomposition",
                        payload={"path_gene_ids": path, "layer_counts": layer_counts},
                    )
                )

    aggregate_negative_limit = max_examples_per_layer * max(1, len(index.layer_names))
    for source, target in _non_edge_pairs(all_nodes, index.aggregate_graph, limit=aggregate_negative_limit):
        split = split_for_key(("aggregate_non_edge", source, target), seed=seed)
        candidates.append(
            _record(
                view_type="multiplex_edge_existence",
                split=split,
                source=GRAPH_TOPOLOGY_SOURCE,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["edge_topology"],
                question=(
                    f"Across the aggregate multiplex in graph version {graph_version}, "
                    f"are {source} and {target} directly connected? List every supporting layer."
                ),
                answer=(
                    f"No. {source} and {target} are not directly connected in any recorded layer "
                    f"of graph version {graph_version}."
                ),
                object_type="multiplex_edge_absence",
                payload={
                    "source_gene_id": source,
                    "target_gene_id": target,
                    "supporting_layers": [],
                    "has_edge": False,
                },
                metadata={"answer_label": "no"},
            )
        )

    for source, target, support_layers in _aggregate_edge_rows(index, limit=max_examples_per_layer):
        path = [source, target]
        split = split_for_key(("aggregate_path", source, target), seed=seed)
        path_edges = _path_edges_with_layers(index, path)
        candidates.append(
            _record(
                view_type="aggregate_multiplex_shortest_path",
                split=split,
                source=GRAPH_TOPOLOGY_SOURCE,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["path_topology"],
                question=(
                    f"What is the shortest path between {source} and {target} across the aggregate multiplex? "
                    "Return nodes, edges, supporting layers, and hop count."
                ),
                answer=(
                    f"The aggregate multiplex shortest path is {' -> '.join(path)} with hop count {len(path) - 1}. "
                    f"Edges with supporting layers: {json.dumps(path_edges, sort_keys=True)}."
                ),
                object_type="aggregate_multiplex_shortest_path",
                payload={"source_gene_id": source, "target_gene_id": target, "path_gene_ids": path, "edges": path_edges},
            )
        )

        layer_counts = _layer_counts_for_path(index, path)
        candidates.append(
            _record(
                view_type="path_layer_decomposition",
                split=split,
                source=GRAPH_TOPOLOGY_SOURCE,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["path_topology"],
                question=f"For the aggregate path {' -> '.join(path)}, how many path edges are supported by each layer?",
                answer=f"Path layer counts: {json.dumps(layer_counts, sort_keys=True)}.",
                object_type="aggregate_path_layer_decomposition",
                payload={"path_gene_ids": path, "layer_counts": layer_counts},
            )
        )

        removed_layer = support_layers[0] if support_layers else "unknown"
        remaining_layers = [layer for layer in support_layers if layer != removed_layer]
        if remaining_layers:
            ablation_answer = (
                f"Ablating `{removed_layer}` removes one support layer for {source}-{target}, but direct "
                f"multiplex support remains through {', '.join(remaining_layers)}. The interpretation should "
                "be weaker and layer-aware, not discarded."
            )
        else:
            ablation_answer = (
                f"Ablating `{removed_layer}` removes the recorded direct multiplex support for {source}-{target}. "
                "The safe conclusion is layer-sensitive graph absence in the ablated graph version, not absence "
                "of a biological relationship."
            )
        candidates.append(
            _record(
                view_type="counterfactual_ablation",
                split=split,
                source=GRAPH_TOPOLOGY_SOURCE,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["interpretation"],
                question=(
                    f"Counterfactual context:\n- Pair: {source}, {target}\n"
                    f"- Supporting layers before ablation: {', '.join(support_layers)}\n"
                    f"- Ablated layer: {removed_layer}\n\nHow should the interpretation change?"
                ),
                answer=ablation_answer,
                object_type="counterfactual_ablation",
                payload={
                    "source_gene_id": source,
                    "target_gene_id": target,
                    "supporting_layers": support_layers,
                    "ablated_layer": removed_layer,
                    "remaining_layers": remaining_layers,
                },
            )
        )

    aggregate_pair = _first_path_pair(index.aggregate_graph)
    if aggregate_pair is not None:
        source, target, path = aggregate_pair
        if len(path) > 2:
            split = split_for_key(("aggregate_path", source, target), seed=seed)
            path_edges = _path_edges_with_layers(index, path)
            candidates.append(
                _record(
                    view_type="aggregate_multiplex_shortest_path",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["path_topology"],
                    question=(
                        f"What is the shortest path between {source} and {target} across the aggregate multiplex? "
                        "Return nodes, edges, supporting layers, and hop count."
                    ),
                    answer=(
                        f"The aggregate multiplex shortest path is {' -> '.join(path)} with hop count {len(path) - 1}. "
                        f"Edges with supporting layers: {json.dumps(path_edges, sort_keys=True)}."
                    ),
                    object_type="aggregate_multiplex_shortest_path",
                    payload={"source_gene_id": source, "target_gene_id": target, "path_gene_ids": path, "edges": path_edges},
                )
            )

            layer_counts = _layer_counts_for_path(index, path)
            candidates.append(
                _record(
                    view_type="path_layer_decomposition",
                    split=split,
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["path_topology"],
                    question=f"For the aggregate path {' -> '.join(path)}, how many path edges are supported by each layer?",
                    answer=f"Path layer counts: {json.dumps(layer_counts, sort_keys=True)}.",
                    object_type="aggregate_path_layer_decomposition",
                    payload={"path_gene_ids": path, "layer_counts": layer_counts},
                )
            )

    mono_vs_count = 0
    for source, target, _layers in _aggregate_edge_rows(index, limit=max_examples_per_layer * max(1, len(index.layer_names))):
        for layer in index.layer_names:
            if not index.layer_graphs[layer].has_edge(source, target):
                split = split_for_key(("mono_vs_multi", layer, source, target), seed=seed)
                candidates.append(
                    _record(
                        view_type="monoplex_vs_multiplex_path_comparison",
                        split=split,
                        source=GRAPH_TOPOLOGY_SOURCE,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["critique"],
                        question=(
                            f"{source} and {target} are connected in the aggregate multiplex but not in `{layer}`. "
                            "What does that mean?"
                        ),
                        answer=(
                            f"The pair has graph support in other multiplex layer(s), but no direct support is recorded "
                            f"in `{layer}`. Interpret the relationship as layer-specific rather than universal."
                        ),
                        object_type="monoplex_vs_multiplex",
                        payload={"layer": layer, "source_gene_id": source, "target_gene_id": target},
                    )
                )
                mono_vs_count += 1
                if mono_vs_count >= max_examples_per_layer:
                    return candidates
    return candidates


def load_corpus_rows(corpus_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = read_json(corpus_dir / "manifest.json")
    modules = read_jsonl(corpus_dir / "modules.jsonl")
    tasks_by_split = {split: read_jsonl(corpus_dir / f"tasks.{split}.jsonl") for split in SPLITS}
    return manifest, modules, tasks_by_split


def generate_module_candidates(
    *,
    corpus_dir: Path,
    graph_version: str,
    seed: int,
    max_modules: int | None,
    max_tasks_per_split: int | None,
) -> list[Candidate]:
    manifest, modules, tasks_by_split = load_corpus_rows(corpus_dir)
    candidates: list[Candidate] = []
    modules_by_id = {module_id_for_row(row): row for row in modules if module_id_for_row(row)}
    all_module_genes = sorted({gene for row in modules for gene in gene_ids_for_row(row)})
    selected_modules = sorted(modules, key=lambda row: str(row.get("module_id")))[:max_modules]
    for module in selected_modules:
        module_id = str(module.get("module_id"))
        source = source_for_row(module)
        split = str(module.get("split") or split_for_key(("module", module_id), seed=seed))
        genes = gene_ids_for_row(module)
        size = int(module.get("size", len(genes)) or len(genes))
        size_bin = str(module.get("size_bin", "unknown"))
        candidates.append(
            _record(
                view_type="closed_book_module_qa",
                split=split if split in SPLITS else split_for_key(("module", module_id), seed=seed),
                source=source,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["interpretation"],
                question=f"In the pre-trajectory SFT source corpus, what source and size are recorded for module {module_id}?",
                answer=(
                    f"{module_id} is recorded as a {source_label(source)} from {source}. "
                    f"It has {size} genes and size bin `{size_bin}`."
                ),
                object_type="module_fact",
                payload={"module_id": module_id, "source": source, "size": size, "size_bin": size_bin, "genes": genes},
                metadata={"source_corpus_dir": str(corpus_dir), "corpus_source": manifest.get("source")},
            )
        )
        candidates.append(
            _record(
                view_type="module_source_distinction",
                split=split if split in SPLITS else split_for_key(("module_source", module_id), seed=seed),
                source=source,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["module_algebra"],
                question=f"Which source oracle records module `{module_id}`, and what kind of claim does that support?",
                answer=(
                    f"`{module_id}` is recorded by `{source}` ({source_label(source)}). "
                    "That supports a source-versioned module/set fact, not an unsupported causal claim."
                ),
                object_type="module_source_distinction",
                payload={"module_id": module_id, "source": source, "size": size, "size_bin": size_bin},
                metadata={"source_corpus_dir": str(corpus_dir), "corpus_source": manifest.get("source")},
            )
        )
        for gene in genes[:3]:
            candidates.append(
                _record(
                    view_type="mentor_ev_module_membership",
                    split=split if split in SPLITS else split_for_key(("module_member", module_id, gene), seed=seed),
                    source=source,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["module_algebra"],
                    question=f"Is `{gene}` a recorded member of module `{module_id}`?",
                    answer=f"Yes. `{gene}` is recorded as a member of `{module_id}` in `{source}`.",
                    object_type="module_membership",
                    payload={"module_id": module_id, "gene_id": gene, "has_membership": True, "source": source},
                    metadata={"answer_label": "yes"},
                )
            )
        negative_gene = next((gene for gene in all_module_genes if gene not in set(genes)), None)
        if negative_gene:
            candidates.append(
                _record(
                    view_type="mentor_ev_module_membership",
                    split=split if split in SPLITS else split_for_key(("module_nonmember", module_id, negative_gene), seed=seed),
                    source=source,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["module_algebra"],
                    question=f"Is `{negative_gene}` a recorded member of module `{module_id}`?",
                    answer=(
                        f"No. `{negative_gene}` is not recorded as a member of `{module_id}` in `{source}`. "
                        "This is module-corpus absence, not biological absence."
                    ),
                    object_type="module_membership",
                    payload={"module_id": module_id, "gene_id": negative_gene, "has_membership": False, "source": source},
                    metadata={"answer_label": "no"},
                )
            )
        if source == MENTOR_EV_SOURCE:
            candidates.append(
                _record(
                    view_type="open_book_module_interpretation",
                    split=split if split in SPLITS else split_for_key(("module_interp", module_id), seed=seed),
                    source=source,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["interpretation"],
                    question=(
                        f"Context:\n- Source: MENTOR-EV dendrogram module corpus\n- Module: {module_id}\n"
                        f"- Size bin: {size_bin}\n- Module size: {size}\n"
                        f"- Example genes: {short_list(genes)}\n\nWrite a cautious interpretation of this module shard."
                    ),
                    answer=(
                        f"{module_id} should be interpreted as a MENTOR-EV dendrogram-derived module in "
                        f"graph version {graph_version}. The listed genes support a network/module-level "
                        "hypothesis in this graph version, not a claim that every pair directly interacts "
                        "or that the module is causally validated."
                    ),
                    object_type="module_interpretation",
                    payload={"module_id": module_id, "source": source, "size": size, "size_bin": size_bin, "genes": genes},
                )
            )

    for split, rows in tasks_by_split.items():
        selected_rows = sorted(rows, key=lambda row: str(row.get("task_id")))[:max_tasks_per_split]
        for task in selected_rows:
            task_id = str(task.get("task_id"))
            source = source_for_row(task)
            task_type = str(task.get("task_type"))
            module_id = module_id_for_row(task)
            target_genes = target_gene_ids_for_task(task)
            visible_genes = gene_ids_for_row(task)
            module = modules_by_id.get(module_id or "")
            if task_type in {"explanation", "recovery", "refinement"} and module_id:
                candidates.append(
                    _record(
                        view_type="open_book_module_interpretation",
                        split=split,
                        source=source,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["interpretation"],
                        question=(
                            f"Context:\n- Source: {source_label(source)}\n- Task: {task_id}\n"
                            f"- Visible genes: {short_list(visible_genes)}\n"
                            f"- Recorded target/module genes: {short_list(target_genes)}\n\n"
                            "Write a cautious evidence-grounded interpretation."
                        ),
                        answer=(
                            f"This is a `{task_type}` example anchored to {module_id}. The visible genes should be "
                            f"interpreted against the recorded {source_label(source)} context. The appropriate claim is "
                            "network/module support in this graph version, with no unsupported causal or all-pairs "
                            "direct-interaction claim."
                        ),
                        object_type="task_interpretation",
                        payload={"task_id": task_id, "task_type": task_type, "module_id": module_id, "visible_genes": visible_genes, "target_genes": target_genes},
                    )
                )
            if task_type == "recovery" and target_genes:
                missing = sorted(set(target_genes) - set(visible_genes))
                candidates.append(
                    _record(
                        view_type="shadow_tool_recovery",
                        split=split,
                        source=source,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["interpretation"],
                        question=(
                            "Observation sequence:\n"
                            f"- The task starts from a partial seed set for {module_id}.\n"
                            f"- Visible genes: {short_list(visible_genes)}.\n"
                            f"- Missing recorded module genes include: {short_list(missing)}.\n"
                            "- Hidden targets are for offline supervision and should not be exposed in trajectory prompts.\n\n"
                            "What is the best training interpretation?"
                        ),
                        answer=(
                            "This is a recovery-style pre-trajectory example. The model should learn that a partial "
                            "module seed set is not a stopping point; it should seek evidence-backed additions from "
                            "graph/RWR context while keeping hidden targets out of downstream trajectory prompts."
                        ),
                        object_type="shadow_tool_recovery",
                        payload={"task_id": task_id, "module_id": module_id, "visible_genes": visible_genes, "missing_target_genes": missing},
                    )
                )
            if task_type == "none":
                candidates.append(
                    _record(
                        view_type="no_edge_no_path_calibration",
                        split=split,
                        source=source,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["critique"],
                        question=(
                            f"Task {task_id} is labeled insufficient support in the source corpus. "
                            "What should the model conclude?"
                        ),
                        answer=(
                        "The model should conclude that this source corpus records insufficient support for one "
                        "shared module for the provided query. It should not claim that the genes have no "
                        "biological relationships outside this graph version."
                    ),
                    object_type="insufficient_support_calibration",
                    payload={"task_id": task_id, "visible_genes": visible_genes},
                )
            )
                candidates.append(
                    _record(
                        view_type="calibration_negative_null_module",
                        split=split,
                        source=source,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["critique"],
                        question=f"How should null/insufficient-support task `{task_id}` be used in SFT?",
                        answer=(
                            f"`{task_id}` is a calibration negative. It teaches insufficient source-corpus support "
                            "and should not be rewritten as proof that no biological relationship exists."
                        ),
                        object_type="calibration_negative_null_module",
                        payload={"task_id": task_id, "visible_genes": visible_genes, "source": source},
                    )
                )
            if module_id and module is not None:
                candidates.append(
                    _record(
                        view_type="critique_preference_sft",
                        split=split,
                        source=source,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["critique"],
                        question=(
                            f"Context:\n- Source: {source_label(source)}\n- Module/task: {module_id} / {task_id}\n\n"
                            "Answer A: These genes are definitely causally related and all directly interact with each other.\n\n"
                            "Answer B: These genes have source-corpus module or network support in this graph version, "
                            "but this does not by itself prove all-pairs direct interaction or causality.\n\n"
                            "Which answer is better?"
                        ),
                        answer="Answer B is better because it preserves graph-versioned support while avoiding unsupported causal and direct-interaction claims.",
                        object_type="critique_preference",
                        payload={"task_id": task_id, "module_id": module_id, "preferred_answer": "B"},
                    )
                )
                if task_type == "recovery":
                    update_payload = {
                        "task_id": task_id,
                        "module_id": module_id,
                        "visible_genes": visible_genes,
                        "predicted_gene_ids": visible_genes,
                        "relationship_status": "partial_module_support",
                        "continue": True,
                    }
                    candidates.append(
                        _record(
                            view_type="structured_state_update",
                            split=split,
                            source=source,
                            graph_version=graph_version,
                            system=SYSTEM_PROMPTS["structured_tool"],
                            question=(
                                f"Given visible genes {short_list(visible_genes)} for recovery task `{task_id}`, "
                                "write the evidence-backed state update without using hidden targets."
                            ),
                            answer=json.dumps(update_payload, sort_keys=True),
                            object_type="structured_state_update",
                            payload=update_payload,
                        )
                    )
    for left, right in zip(selected_modules, selected_modules[1:]):
        left_id = module_id_for_row(left)
        right_id = module_id_for_row(right)
        if not left_id or not right_id:
            continue
        left_genes = set(gene_ids_for_row(left))
        right_genes = set(gene_ids_for_row(right))
        if not left_genes or not right_genes:
            continue
        left_source = source_for_row(left)
        right_source = source_for_row(right)
        pair_source = left_source if left_source == right_source else MIXED_SOURCE
        split = split_for_key(("module_pair", left_id, right_id), seed=seed)
        intersection = sorted(left_genes & right_genes)
        union_size = len(left_genes | right_genes)
        jaccard = (len(intersection) / union_size) if union_size else 0.0
        candidates.append(
            _record(
                view_type="module_overlap_set_algebra",
                split=split,
                source=pair_source,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["module_algebra"],
                question=f"What is the exact overlap between modules `{left_id}` and `{right_id}`?",
                answer=(
                    f"`{left_id}` and `{right_id}` share {len(intersection)} genes. "
                    f"Union size is {union_size}; Jaccard overlap is {jaccard:.6f}. "
                    f"Intersection genes: {short_list(intersection, max_items=20)}."
                ),
                object_type="module_overlap_set_algebra",
                payload={
                    "left_module_id": left_id,
                    "right_module_id": right_id,
                    "left_source": left_source,
                    "right_source": right_source,
                    "intersection_genes": intersection,
                    "intersection_size": len(intersection),
                    "union_size": union_size,
                    "overlap_jaccard": round(jaccard, 6),
                },
            )
        )
        left_minus_right = sorted(left_genes - right_genes)
        is_subset = not left_minus_right
        candidates.append(
            _record(
                view_type="module_containment_set_algebra",
                split=split,
                source=pair_source,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["module_algebra"],
                question=f"Is module `{left_id}` an exact subset of module `{right_id}`? If not, list violating genes.",
                answer=(
                    f"Yes. `{left_id}` is an exact subset of `{right_id}`."
                    if is_subset
                    else f"No. `{left_id}` is not an exact subset of `{right_id}`; violating genes: {short_list(left_minus_right, max_items=20)}."
                ),
                object_type="module_containment_set_algebra",
                payload={
                    "left_module_id": left_id,
                    "right_module_id": right_id,
                    "left_source": left_source,
                    "right_source": right_source,
                    "exact_subset": is_subset,
                    "violating_genes": left_minus_right,
                },
                metadata={"answer_label": "yes" if is_subset else "no"},
            )
        )
    return candidates


def generate_module_cohesion_candidates(
    *,
    corpus_dir: Path,
    index: MultiplexIndex,
    graph_version: str,
    seed: int,
    max_modules: int | None,
    max_genes_per_module: int = 12,
) -> list[Candidate]:
    _manifest, modules, _tasks_by_split = load_corpus_rows(corpus_dir)
    candidates: list[Candidate] = []
    selected_modules = sorted(modules, key=lambda row: str(row.get("module_id")))[:max_modules]
    for module in selected_modules:
        module_id = module_id_for_row(module)
        if not module_id:
            continue
        source = source_for_row(module)
        split = str(module.get("split") or split_for_key(("module_cohesion", module_id), seed=seed))
        split = split if split in SPLITS else split_for_key(("module_cohesion", module_id), seed=seed)
        genes = [gene for gene in gene_ids_for_row(module) if gene in index.aggregate_graph]
        query_genes = sorted(genes)[:max_genes_per_module]
        if len(query_genes) < 2:
            continue
        subgraph = index.aggregate_graph.subgraph(query_genes)
        possible_edges = len(query_genes) * (len(query_genes) - 1) / 2
        density = (subgraph.number_of_edges() / possible_edges) if possible_edges else 0.0
        payload = {
            "module_id": module_id,
            "source": source,
            "query_gene_ids": query_genes,
            "node_count": len(query_genes),
            "edge_count": subgraph.number_of_edges(),
            "density": round(density, 6),
        }
        candidates.append(
            _record(
                view_type="module_cohesion_summary",
                split=split,
                source=source,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["module_algebra"],
                question=(
                    f"Using the aggregate multiplex, what is the induced cohesion summary for "
                    f"the sampled genes from module `{module_id}`?"
                ),
                answer=(
                    f"For sampled genes from `{module_id}`, the aggregate induced subgraph has "
                    f"{len(query_genes)} nodes, {subgraph.number_of_edges()} edges, and density {density:.6f}. "
                    "This is a graph cohesion statistic, not causal validation."
                ),
                object_type="module_cohesion_summary",
                payload=payload,
            )
        )
    return candidates


def generate_structured_tool_candidates(
    *,
    index: MultiplexIndex | None,
    graph_version: str,
    seed: int,
    max_examples: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    if index is not None:
        for layer in list(index.layer_names)[:max_examples]:
            graph = index.layer_graphs[layer]
            nodes = sorted(str(node) for node in graph.nodes())
            if not nodes:
                continue
            gene_id = nodes[0]
            payload = {
                "tool_name": "get_neighbors",
                "arguments": {"gene_id": gene_id, "layer": layer, "graph_version": graph_version},
                "reason": "direct_neighbors_by_layer",
            }
            candidates.append(
                _record(
                    view_type="tool_call_choice",
                    split=split_for_key(("tool_choice", layer, gene_id), seed=seed),
                    source=GRAPH_TOPOLOGY_SOURCE,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["structured_tool"],
                    question=f"Which model-facing tool should be used to get `{gene_id}`'s direct neighbors in `{layer}`?",
                    answer=json.dumps(payload, sort_keys=True),
                    object_type="tool_call_choice",
                    payload=payload,
                )
            )
    candidates.append(
        _record(
            view_type="provenance_refusal_raw_cli",
            split=split_for_key(("raw_cli_refusal", graph_version), seed=seed),
            source=GRAPH_TOPOLOGY_SOURCE,
            graph_version=graph_version,
            system=SYSTEM_PROMPTS["structured_tool"],
            question=(
                "A prompt asks the model to read raw graph files from `data/runtime/full_brain_multiplex_store` "
                "during a trajectory. What should the model do instead?"
            ),
            answer=(
                "Do not expose or request `raw_filesystem_or_cli` access in a trajectory prompt. Use "
                "`model_facing_graph_rwr_module_tools` with biological arguments and preserve graph-version provenance."
            ),
            object_type="provenance_refusal_raw_cli",
            payload={
                "forbidden_interface": "raw_filesystem_or_cli",
                "allowed_interface": "model_facing_graph_rwr_module_tools",
                "graph_version": graph_version,
            },
        )
    )
    return candidates


def discover_rank_cache_context_dir(rank_cache_root: Path) -> Path | None:
    if not rank_cache_root.exists():
        return None
    contexts = sorted(path for path in rank_cache_root.iterdir() if path.is_dir() and path.name.startswith("context_"))
    return contexts[-1] if contexts else None


def seed_genes_from_candidates(candidates: list[Candidate], *, max_seeds: int) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for candidate in sorted(candidates, key=lambda item: item.record["metadata"]["record_id"]):
        payload = candidate.canonical_object.get("payload", {})
        genes: list[str] = []
        for key in ("genes", "visible_genes", "target_genes"):
            values = payload.get(key)
            if isinstance(values, list):
                genes.extend(str(gene) for gene in values if isinstance(gene, str))
        for gene in genes:
            if gene not in seen:
                seen.add(gene)
                result.append((gene, candidate.split))
                if len(result) >= max_seeds:
                    return result
    return result


def generate_rwr_candidates(
    *,
    rank_cache_context_dir: Path | None,
    module_candidates: list[Candidate],
    graph_version: str,
    max_seeds: int,
) -> tuple[list[Candidate], dict[str, Any]]:
    if rank_cache_context_dir is None:
        return [], {"status": "skipped", "reason": "rank_cache_context_dir_not_provided"}
    context = load_rank_cache_context_from_dir(rank_cache_context_dir)
    candidates: list[Candidate] = []
    missing = 0
    for seed_gene, split in seed_genes_from_candidates(module_candidates, max_seeds=max_seeds):
        try:
            ranked = load_seed_rank_cache(rank_cache_context_dir, seed_gene, max_rows=5)
        except FileNotFoundError:
            missing += 1
            continue
        if not ranked:
            continue
        top = ranked[0]
        source = RWR_LOE_SOURCE
        candidates.append(
            _record(
                view_type="rwr_loe_rank_lookup",
                split=split,
                source=source,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["rwr"],
                question=(
                    f"Using RWR-LOE rank-cache schema {context.get('schema_version', 'unknown')}, "
                    f"what is candidate {top['gene']}'s rank and score for seed gene {seed_gene}?"
                ),
                answer=(
                    f"For seed {seed_gene}, candidate {top['gene']} has RWR-LOE rank {top['rank']} "
                    f"and score {top['score']:.16g}. This is network-proximity evidence, not proof of "
                    "direct interaction or causality."
                ),
                object_type="rwr_loe_rank_lookup",
                payload={"seed_gene_id": seed_gene, "candidate_gene_id": top["gene"], "rank": top["rank"], "score": top["score"], "rank_cache_context": context},
            )
        )
        top_genes = [str(row["gene"]) for row in ranked[:5]]
        candidates.append(
            _record(
                view_type="rwr_neighborhood_interpretation",
                split=split,
                source=source,
                graph_version=graph_version,
                system=SYSTEM_PROMPTS["rwr"],
                question=(
                    f"Given seed gene {seed_gene}, the top RWR-LOE candidates are {', '.join(top_genes)}. "
                    "Which candidates are strongest network-proximal additions, and how should the result be framed?"
                ),
                answer=(
                    f"The strongest network-proximal additions are the highest-ranked candidates: {', '.join(top_genes[:3])}. "
                    "They should be framed as RWR-LOE network-proximity candidates in this graph version, not as confirmed causal genes."
                ),
                object_type="rwr_neighborhood_interpretation",
                payload={"seed_gene_id": seed_gene, "top_candidates": ranked[:5], "rank_cache_context": context},
            )
        )
        if len(ranked) >= 2:
            left = ranked[0]
            right = ranked[1]
            candidates.append(
                _record(
                    view_type="rwr_loe_rank_comparison",
                    split=split,
                    source=source,
                    graph_version=graph_version,
                    system=SYSTEM_PROMPTS["rwr"],
                    question=(
                        f"For seed gene {seed_gene}, which RWR-LOE candidate is ranked higher: "
                        f"{left['gene']} or {right['gene']}?"
                    ),
                    answer=(
                        f"{left['gene']} is ranked higher for seed {seed_gene}: rank {left['rank']} "
                        f"versus {right['gene']} at rank {right['rank']}. This is network-proximity evidence, "
                        "not causal proof."
                    ),
                    object_type="rwr_loe_rank_comparison",
                    payload={
                        "seed_gene_id": seed_gene,
                        "left_candidate_gene_id": left["gene"],
                        "right_candidate_gene_id": right["gene"],
                        "left_rank": left["rank"],
                        "right_rank": right["rank"],
                        "winner_gene_id": left["gene"],
                        "rank_cache_context": context,
                    },
                    metadata={"answer_label": "left"},
                )
            )
        for top_k in (3, 5):
            if len(ranked) >= top_k:
                candidate_gene = str(ranked[top_k - 1]["gene"])
                candidates.append(
                    _record(
                        view_type="rwr_loe_topk_membership",
                        split=split,
                        source=source,
                        graph_version=graph_version,
                        system=SYSTEM_PROMPTS["rwr"],
                        question=f"For seed gene {seed_gene}, is {candidate_gene} in the top {top_k} RWR-LOE candidates?",
                        answer=(
                            f"Yes. For seed {seed_gene}, {candidate_gene} is in the top {top_k} with rank "
                            f"{ranked[top_k - 1]['rank']}. This is network-proximity evidence, not causal proof."
                        ),
                        object_type="rwr_loe_topk_membership",
                        payload={
                            "seed_gene_id": seed_gene,
                            "candidate_gene_id": candidate_gene,
                            "top_k": top_k,
                            "rank": ranked[top_k - 1]["rank"],
                            "is_in_top_k": True,
                            "rank_cache_context": context,
                        },
                        metadata={"answer_label": "yes"},
                    )
                )
    return candidates, {
        "status": "loaded",
        "context_dir": str(rank_cache_context_dir),
        "schema_version": context.get("schema_version"),
        "missing_seed_cache_count": missing,
        "record_count": len(candidates),
    }


def sample_records(
    candidates: list[Candidate],
    *,
    target_counts: dict[str, int] | None,
    seed: int,
) -> list[Candidate]:
    candidates = deduplicate_candidates(candidates)
    if target_counts is None:
        return sorted(candidates, key=lambda item: item.record["metadata"]["record_id"])
    selected: list[Candidate] = []
    for split in SPLITS:
        target = int(target_counts.get(split, 0))
        split_candidates = [candidate for candidate in candidates if candidate.split == split]
        if target <= 0 or len(split_candidates) <= target:
            selected.extend(sorted(split_candidates, key=lambda item: item.record["metadata"]["record_id"]))
            continue
        by_bucket: dict[str, list[Candidate]] = defaultdict(list)
        by_view: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in split_candidates:
            by_bucket[candidate.bucket].append(candidate)
            by_view[candidate.view_type].append(candidate)
        chosen_ids: set[str] = set()
        for view_type in sorted(BUCKET_BY_VIEW):
            if view_type not in by_view or len(chosen_ids) >= target:
                continue
            view_candidates = sorted(
                by_view[view_type],
                key=lambda item: stable_order_key(item.record["metadata"]["record_id"], seed=seed),
            )
            if view_candidates:
                chosen_ids.add(view_candidates[0].record["metadata"]["record_id"])
        for bucket, weight in DEFAULT_BUCKET_WEIGHTS.items():
            bucket_target = int(round(target * weight))
            bucket_candidates = sorted(
                by_bucket.get(bucket, []),
                key=lambda item: stable_order_key(item.record["metadata"]["record_id"], seed=seed),
            )
            already_in_bucket = sum(
                1
                for candidate in bucket_candidates
                if candidate.record["metadata"]["record_id"] in chosen_ids
            )
            remaining_bucket_target = max(0, bucket_target - already_in_bucket)
            for candidate in bucket_candidates:
                if remaining_bucket_target <= 0 or len(chosen_ids) >= target:
                    break
                record_id = candidate.record["metadata"]["record_id"]
                if record_id in chosen_ids:
                    continue
                chosen_ids.add(record_id)
                remaining_bucket_target -= 1
        if len(chosen_ids) < target:
            remaining = sorted(
                [candidate for candidate in split_candidates if candidate.record["metadata"]["record_id"] not in chosen_ids],
                key=lambda item: stable_order_key(item.record["metadata"]["record_id"], seed=seed),
            )
            for candidate in remaining[: target - len(chosen_ids)]:
                chosen_ids.add(candidate.record["metadata"]["record_id"])
        selected.extend(
            sorted(
                [candidate for candidate in split_candidates if candidate.record["metadata"]["record_id"] in chosen_ids],
                key=lambda item: item.record["metadata"]["record_id"],
            )
        )
    return selected


def validate_records(candidates: list[Candidate]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    object_splits: dict[str, set[str]] = defaultdict(set)
    record_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        record = candidate.record
        metadata = record.get("metadata")
        record_id = None
        if isinstance(metadata, dict):
            record_id = metadata.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            errors.append({"index": index, "error": "missing_record_id"})
        elif record_id in record_ids:
            errors.append({"record_id": record_id, "error": "duplicate_record_id"})
        else:
            record_ids.add(record_id)
        for key in ("system", "question", "answer"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                errors.append({"record_id": record_id, "error": f"missing_{key}"})
        if not isinstance(metadata, dict):
            errors.append({"record_id": record_id, "error": "missing_metadata"})
            continue
        for key in ("view_type", "mixture_bucket", "curriculum_stage", "source", "split", "canonical_object_id", "graph_version"):
            if not metadata.get(key):
                errors.append({"record_id": record_id, "error": f"missing_metadata_{key}"})
        if metadata.get("mixture_bucket") not in DEFAULT_BUCKET_WEIGHTS:
            errors.append({"record_id": record_id, "error": "unknown_mixture_bucket"})
        if metadata.get("curriculum_stage") not in CURRICULUM_STAGES:
            errors.append({"record_id": record_id, "error": "unknown_curriculum_stage"})
        object_id = metadata.get("canonical_object_id")
        split = metadata.get("split")
        if isinstance(object_id, str) and isinstance(split, str):
            object_splits[object_id].add(split)
        if "definitely causally" in str(record.get("answer", "")).lower():
            errors.append({"record_id": record_id, "error": "unsupported_causal_language_in_answer"})
    leaking = {object_id: sorted(splits) for object_id, splits in object_splits.items() if len(splits) > 1}
    for object_id, splits in leaking.items():
        errors.append({"object_id": object_id, "splits": splits, "error": "canonical_object_split_leakage"})
    counts_by_split = Counter(candidate.split for candidate in candidates)
    counts_by_view = Counter(candidate.view_type for candidate in candidates)
    counts_by_bucket = Counter(candidate.bucket for candidate in candidates)
    counts_by_stage = Counter(str(candidate.record.get("metadata", {}).get("curriculum_stage", "unspecified")) for candidate in candidates)
    counts_by_context_mode = Counter(str(candidate.record.get("metadata", {}).get("context_mode", "unspecified")) for candidate in candidates)
    return {
        "schema_version": SCHEMA_VERSION,
        "fatal_error_count": len(errors),
        "errors": errors[:100],
        "truncated_error_count": max(0, len(errors) - 100),
        "record_count": len(candidates),
        "record_count_by_split": dict(sorted(counts_by_split.items())),
        "record_count_by_view_type": dict(sorted(counts_by_view.items())),
        "record_count_by_mixture_bucket": dict(sorted(counts_by_bucket.items())),
        "record_count_by_curriculum_stage": dict(sorted(counts_by_stage.items())),
        "record_count_by_context_mode": dict(sorted(counts_by_context_mode.items())),
    }


def write_curriculum_stage_files(out_dir: Path, records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    curriculum_dir = out_dir / "curriculum"
    for stage in CURRICULUM_STAGES:
        outputs[stage] = {}
        for split in SPLITS:
            if stage == "stage6_blend":
                rows = records_by_split[split]
            else:
                rows = [
                    record
                    for record in records_by_split[split]
                    if record.get("metadata", {}).get("curriculum_stage") == stage
                ]
            path = curriculum_dir / stage / f"{split}.jsonl"
            write_jsonl(path, rows)
            outputs[stage][split] = str(path)
    return outputs


def build_pretrajectory_sft_dataset(
    *,
    out_dir: Path,
    mixed_corpus_dir: Path = DEFAULT_MIXED_CORPUS_DIR,
    store_manifest_path: Path | None = DEFAULT_STORE_MANIFEST_PATH,
    graph_flist: Path | None = DEFAULT_GRAPH_FLIST_PATH,
    graph_layer_limit: int | None = None,
    graph_max_edges_per_layer: int | None = 2000,
    graph_edgelist_has_headers: bool = False,
    rank_cache_context_dir: Path | None = None,
    seed: int = 42,
    max_modules: int | None = None,
    max_tasks_per_split: int | None = None,
    max_graph_examples_per_layer: int = 1000,
    max_rwr_seeds: int = 500,
    context_modes: tuple[str, ...] = DEFAULT_CONTEXT_MODES,
    target_counts: dict[str, int] | None = TARGET_SPLIT_COUNTS,
    overwrite: bool = False,
) -> dict[str, Any]:
    invalid_context_modes = [mode for mode in context_modes if mode not in CONTEXT_MODES]
    if invalid_context_modes:
        raise ValueError(f"Unsupported context mode(s): {', '.join(invalid_context_modes)}")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_version = graph_version_from_manifest(store_manifest_path)
    _corpus_manifest, corpus_modules, _tasks_by_split = load_corpus_rows(mixed_corpus_dir)
    index: MultiplexIndex | None = None
    module_candidates = generate_module_candidates(
        corpus_dir=mixed_corpus_dir,
        graph_version=graph_version,
        seed=seed,
        max_modules=max_modules,
        max_tasks_per_split=max_tasks_per_split,
    )
    entity_schema_candidates: list[Candidate] = []
    layer_metadata_candidates: list[Candidate] = []
    module_cohesion_candidates: list[Candidate] = []
    structured_tool_candidates: list[Candidate] = []
    topology_candidates: list[Candidate] = []
    graph_summary: dict[str, Any] = {"status": "skipped", "reason": "graph_flist_not_provided"}
    if graph_flist is not None:
        index = load_sampled_multiplex_index_from_flist(
            graph_flist,
            max_layers=graph_layer_limit,
            max_edges_per_layer=graph_max_edges_per_layer,
            edgelist_has_headers=graph_edgelist_has_headers,
        )
        entity_schema_candidates = generate_entity_schema_candidates(
            modules=corpus_modules,
            graph_gene_ids=index.gene_ids,
            graph_version=graph_version,
            seed=seed,
            max_alias_examples=max_graph_examples_per_layer,
        )
        layer_metadata_candidates = generate_layer_metadata_candidates(
            index,
            graph_version=graph_version,
            seed=seed,
            max_layers=max_graph_examples_per_layer,
        )
        topology_candidates = generate_topology_candidates(
            index,
            graph_version=graph_version,
            seed=seed,
            max_examples_per_layer=max_graph_examples_per_layer,
        )
        module_cohesion_candidates = generate_module_cohesion_candidates(
            corpus_dir=mixed_corpus_dir,
            index=index,
            graph_version=graph_version,
            seed=seed,
            max_modules=max_modules,
        )
        structured_tool_candidates = generate_structured_tool_candidates(
            index=index,
            graph_version=graph_version,
            seed=seed,
            max_examples=max_graph_examples_per_layer,
        )
        graph_summary = {
            "status": "loaded",
            "graph_flist": str(graph_flist),
            "layer_count": len(index.layer_names),
            "gene_count": len(index.gene_ids),
            "aggregate_edge_count": index.aggregate_graph.number_of_edges(),
            "record_count": len(topology_candidates),
            "graph_layer_limit": graph_layer_limit,
            "graph_max_edges_per_layer": graph_max_edges_per_layer,
        }
    else:
        module_gene_ids = sorted({gene for module in corpus_modules for gene in gene_ids_for_row(module)})
        entity_schema_candidates = generate_entity_schema_candidates(
            modules=corpus_modules,
            graph_gene_ids=module_gene_ids,
            graph_version=graph_version,
            seed=seed,
            max_alias_examples=max_graph_examples_per_layer,
        )
        structured_tool_candidates = generate_structured_tool_candidates(
            index=None,
            graph_version=graph_version,
            seed=seed,
            max_examples=max_graph_examples_per_layer,
        )
    if rank_cache_context_dir is None:
        rank_cache_context_dir = discover_rank_cache_context_dir(DEFAULT_RANK_CACHE_ROOT)
    rwr_candidates, rwr_summary = generate_rwr_candidates(
        rank_cache_context_dir=rank_cache_context_dir,
        module_candidates=module_candidates,
        graph_version=graph_version,
        max_seeds=max_rwr_seeds,
    )
    base_candidates = deduplicate_candidates(
        entity_schema_candidates
        + layer_metadata_candidates
        + module_candidates
        + module_cohesion_candidates
        + topology_candidates
        + rwr_candidates
        + structured_tool_candidates
    )
    if target_counts is not None:
        preexpanded_candidates = sample_records(base_candidates, target_counts=target_counts, seed=seed)
    else:
        preexpanded_candidates = base_candidates
    all_candidates = deduplicate_candidates(expand_context_mode_candidates(preexpanded_candidates, context_modes))
    selected = sample_records(all_candidates, target_counts=target_counts, seed=seed)
    validation = validate_records(selected)
    if validation["fatal_error_count"]:
        raise ValueError(f"Generated dataset failed validation with {validation['fatal_error_count']} errors.")

    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for candidate in selected:
        records_by_split[candidate.split].append(candidate.record)
    for split in SPLITS:
        write_jsonl(out_dir / f"{split}.jsonl", records_by_split[split])
    curriculum_outputs = write_curriculum_stage_files(out_dir, records_by_split)
    canonical_by_id = {
        candidate.canonical_object["object_id"]: candidate.canonical_object
        for candidate in selected
    }
    canonical_objects = [canonical_by_id[key] for key in sorted(canonical_by_id)]
    write_jsonl(out_dir / "canonical_objects.jsonl", canonical_objects)
    write_json(out_dir / "validation_report.json", validation)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "out_dir": str(out_dir),
        "seed": seed,
        "graph_version": graph_version,
        "mixed_corpus_dir": str(mixed_corpus_dir),
        "store_manifest_path": str(store_manifest_path) if store_manifest_path else None,
        "target_counts": target_counts,
        "mixture_weights": DEFAULT_BUCKET_WEIGHTS,
        "context_modes": list(context_modes),
        "base_candidate_count": len(base_candidates),
        "preexpanded_candidate_count": len(preexpanded_candidates),
        "base_candidate_count_by_family": {
            "entity_schema": len(entity_schema_candidates),
            "layer_metadata": len(layer_metadata_candidates),
            "module": len(module_candidates),
            "module_cohesion": len(module_cohesion_candidates),
            "topology": len(topology_candidates),
            "rwr": len(rwr_candidates),
            "structured_tool": len(structured_tool_candidates),
        },
        "candidate_count": len(all_candidates),
        "selected_record_count": len(selected),
        "canonical_object_count": len(canonical_objects),
        "outputs": {
            "train": str(out_dir / "train.jsonl"),
            "val": str(out_dir / "val.jsonl"),
            "test": str(out_dir / "test.jsonl"),
            "canonical_objects": str(out_dir / "canonical_objects.jsonl"),
            "validation_report": str(out_dir / "validation_report.json"),
            "curriculum": curriculum_outputs,
        },
        "sources": {
            "mentor_ev_module_source": "data/gw_dendrogram_corpus_full_brain",
            "rwr_loe_source": "data/rwr_loe_corpus_full_brain",
            "mixed_source": str(mixed_corpus_dir),
            "graph": graph_summary,
            "rwr": rwr_summary,
        },
        "record_count_by_split": validation["record_count_by_split"],
        "record_count_by_view_type": validation["record_count_by_view_type"],
        "record_count_by_mixture_bucket": validation["record_count_by_mixture_bucket"],
        "record_count_by_curriculum_stage": validation["record_count_by_curriculum_stage"],
        "record_count_by_context_mode": validation["record_count_by_context_mode"],
    }
    write_json(out_dir / "manifest.json", manifest)
    return {"manifest": manifest, "validation_report": validation, "records": selected}


def _target_counts_from_args(args: argparse.Namespace) -> dict[str, int] | None:
    if args.all_candidates:
        return None
    if args.preset == "patchcheck":
        return dict(PATCHCHECK_SPLIT_COUNTS)
    if args.preset == "full_1m":
        return dict(FULL_1M_SPLIT_COUNTS)
    return {
        "train": args.target_train_records,
        "val": args.target_val_records,
        "test": args.target_test_records,
    }


def _context_modes_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    modes = tuple(mode.strip() for mode in args.context_modes.split(",") if mode.strip())
    return modes or DEFAULT_CONTEXT_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pre-trajectory MENTOR-RL SFT records.")
    parser.add_argument(
        "--preset",
        choices=("custom", "patchcheck", "full_1m"),
        default="custom",
        help="Named target-count preset. `custom` uses the explicit target-* arguments.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mixed-corpus-dir", type=Path, default=DEFAULT_MIXED_CORPUS_DIR)
    parser.add_argument("--store-manifest-path", type=Path, default=DEFAULT_STORE_MANIFEST_PATH)
    parser.add_argument("--graph-flist", type=Path, default=DEFAULT_GRAPH_FLIST_PATH, help="Sampled/header-compatible graph flist for topology QA.")
    parser.add_argument("--skip-graph-topology", action="store_true", help="Skip graph-topology QA generation.")
    parser.add_argument("--graph-layer-limit", type=int, default=None)
    parser.add_argument("--graph-max-edges-per-layer", type=int, default=2000)
    parser.add_argument("--graph-edgelist-has-headers", action="store_true")
    parser.add_argument("--rank-cache-context-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--max-tasks-per-split", type=int, default=None)
    parser.add_argument("--max-graph-examples-per-layer", type=int, default=1000)
    parser.add_argument("--max-rwr-seeds", type=int, default=500)
    parser.add_argument(
        "--context-modes",
        type=str,
        default=",".join(DEFAULT_CONTEXT_MODES),
        help=f"Comma-separated prompt context modes. Choices: {', '.join(CONTEXT_MODES)}.",
    )
    parser.add_argument("--target-train-records", type=int, default=TARGET_SPLIT_COUNTS["train"])
    parser.add_argument("--target-val-records", type=int, default=TARGET_SPLIT_COUNTS["val"])
    parser.add_argument("--target-test-records", type=int, default=TARGET_SPLIT_COUNTS["test"])
    parser.add_argument("--all-candidates", action="store_true", help="Do not sample to target counts; write every generated candidate.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_pretrajectory_sft_dataset(
        out_dir=args.out_dir,
        mixed_corpus_dir=args.mixed_corpus_dir,
        store_manifest_path=args.store_manifest_path,
        graph_flist=None if args.skip_graph_topology else args.graph_flist,
        graph_layer_limit=args.graph_layer_limit,
        graph_max_edges_per_layer=args.graph_max_edges_per_layer,
        graph_edgelist_has_headers=args.graph_edgelist_has_headers,
        rank_cache_context_dir=args.rank_cache_context_dir,
        seed=args.seed,
        max_modules=args.max_modules,
        max_tasks_per_split=args.max_tasks_per_split,
        max_graph_examples_per_layer=args.max_graph_examples_per_layer,
        max_rwr_seeds=args.max_rwr_seeds,
        context_modes=_context_modes_from_args(args),
        target_counts=_target_counts_from_args(args),
        overwrite=args.overwrite,
    )
    summary = {
        "manifest_path": str(args.out_dir / "manifest.json"),
        "selected_record_count": result["manifest"]["selected_record_count"],
        "record_count_by_split": result["manifest"]["record_count_by_split"],
        "record_count_by_mixture_bucket": result["manifest"]["record_count_by_mixture_bucket"],
        "record_count_by_context_mode": result["manifest"]["record_count_by_context_mode"],
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
