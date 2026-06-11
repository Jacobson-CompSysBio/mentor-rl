#!/usr/bin/env python3
"""Build a MENTOR genome-wide dendrogram corpus.

This mirrors the CORUM corpus row shape while replacing curated complexes with
tree-derived MENTOR modules from ``data/gw_dendrogram.txt``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DENDROGRAM_PATH = REPO_ROOT / "data" / "gw_dendrogram.txt"
DEFAULT_STORE_DIR = REPO_ROOT / "data" / "humannet_multiplex_store"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "gw_dendrogram_corpus"

SPLITS = ("train", "val", "test")
TASK_TYPES = ("explanation", "recovery", "refinement", "none")
EVIDENCE_MODES = ("minimal", "graph")
DIFFICULTIES = ("easy", "medium", "hard")
EXPLANATION_DIFFICULTY = "complete"
POSITIVE_RELATIONSHIP_STATUS = "validated_group"
NONE_RELATIONSHIP_STATUS = "insufficient_support"
SCHEMA_VERSION = "gw-dendrogram-corpus-v1"
SOURCE_NAME = "MENTOR_GW_DENDROGRAM"

DENDROGRAM_DISTANCE_PERCENTILE_BANDS = {
    "easy": (0.0, 0.25),
    "medium": (0.25, 0.50),
    "hard": (0.50, 0.75),
}

DENDROGRAM_DISTANCE_FALLBACK_PERCENTILE_BANDS = {
    "easy": (0.0, 0.25),
    "medium": (0.25, 0.50),
    "hard": (0.50, 0.75),
}

BUILD_STAGES = (
    ("load_store_genes", "Load HumanNet store gene universe"),
    ("parse_dendrogram", "Parse genome-wide dendrogram"),
    ("extract_modules", "Extract filtered dendrogram modules"),
    ("assign_splits", "Assign train/val/test splits"),
    ("assign_difficulties", "Assign task difficulties"),
    ("build_prototypes", "Build task prototypes"),
    ("balance_prototypes", "Balance task types"),
    ("materialize_tasks", "Materialize canonical tasks"),
    ("write_outputs", "Write corpus files"),
)


@dataclass(frozen=True)
class DendrogramNode:
    node_id: int
    left_id: int
    right_id: int
    height: float
    label: str

    @property
    def is_leaf(self) -> bool:
        return self.left_id == -1 and self.right_id == -1


@dataclass(frozen=True)
class DendrogramDistanceIndex:
    nodes: dict[int, DendrogramNode]
    parent_by_node: dict[int, int]
    subtree_genes: dict[int, list[str]]
    candidate_gene_universe: set[str]


@dataclass
class BuildStats:
    skipped_refinement: Counter = field(default_factory=Counter)
    skipped_none: Counter = field(default_factory=Counter)
    balance: dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Persist a small progress JSON file for long corpus builds."""

    def __init__(self, path: Path) -> None:
        self.path = path
        timestamp = utc_now_iso()
        self.stage_lookup = {stage: index for index, (stage, _) in enumerate(BUILD_STAGES, start=1)}
        self.state: dict[str, Any] = {
            "status": "running",
            "current_stage": None,
            "current_stage_label": None,
            "stage_index": 0,
            "stage_count": len(BUILD_STAGES),
            "stage_progress": {"completed": 0, "total": None, "unit": None},
            "overall_progress": 0.0,
            "message": "Initialized genome-wide dendrogram corpus build.",
            "metrics": {},
            "run_context": {},
            "started_at": timestamp,
            "updated_at": timestamp,
            "stages": [
                {"stage": stage, "label": label, "status": "pending"}
                for stage, label in BUILD_STAGES
            ],
        }
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _touch(self) -> None:
        self.state["updated_at"] = utc_now_iso()

    def _stage_entry(self, stage_name: str) -> dict[str, Any]:
        for stage in self.state["stages"]:
            if stage["stage"] == stage_name:
                return stage
        raise KeyError(stage_name)

    def _mark_current_complete(self) -> None:
        current = self.state.get("current_stage")
        if current:
            entry = self._stage_entry(current)
            if entry["status"] == "running":
                entry["status"] = "completed"

    def _recompute_overall(self) -> None:
        if self.state["status"] == "completed":
            self.state["overall_progress"] = 1.0
            return
        stage_index = int(self.state.get("stage_index", 0))
        if stage_index <= 0:
            self.state["overall_progress"] = 0.0
            return
        progress = self.state.get("stage_progress", {})
        completed = progress.get("completed")
        total = progress.get("total")
        fraction = 0.0
        if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
            fraction = min(max(float(completed) / float(total), 0.0), 1.0)
        self.state["overall_progress"] = round(((stage_index - 1) + fraction) / len(BUILD_STAGES), 6)

    def set_context(self, context: dict[str, Any]) -> None:
        self.state["run_context"] = context
        self._touch()
        self._write()

    def start(self, stage_name: str, *, total: int | None = None, unit: str | None = None) -> None:
        self._mark_current_complete()
        entry = self._stage_entry(stage_name)
        entry["status"] = "running"
        self.state["status"] = "running"
        self.state["current_stage"] = stage_name
        self.state["current_stage_label"] = entry["label"]
        self.state["stage_index"] = self.stage_lookup[stage_name]
        self.state["stage_progress"] = {"completed": 0, "total": total, "unit": unit}
        self.state["message"] = entry["label"]
        self.state["metrics"] = {}
        self._touch()
        self._recompute_overall()
        self._write()
        print(f"[gw_dendrogram_corpus] {self.state['stage_index']}/{len(BUILD_STAGES)} {entry['label']}", flush=True)

    def update(
        self,
        *,
        completed: int | float | None = None,
        total: int | float | None = None,
        unit: str | None = None,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if completed is not None:
            self.state["stage_progress"]["completed"] = completed
        if total is not None:
            self.state["stage_progress"]["total"] = total
        if unit is not None:
            self.state["stage_progress"]["unit"] = unit
        if message is not None:
            self.state["message"] = message
        if metrics:
            self.state["metrics"].update(metrics)
        self._touch()
        self._recompute_overall()
        self._write()

    def complete(self, *, metrics: dict[str, Any]) -> None:
        self._mark_current_complete()
        self.state["status"] = "completed"
        self.state["message"] = "Completed genome-wide dendrogram corpus build."
        self.state["metrics"].update(metrics)
        self._touch()
        self._recompute_overall()
        self._write()
        print("[gw_dendrogram_corpus] build complete", flush=True)

    def fail(self, error: Exception) -> None:
        current = self.state.get("current_stage")
        if current:
            self._stage_entry(current)["status"] = "failed"
        self.state["status"] = "failed"
        self.state["message"] = f"Build failed: {error}"
        self.state["error"] = {"type": error.__class__.__name__, "message": str(error)}
        self._touch()
        self._recompute_overall()
        self._write()
        print(f"[gw_dendrogram_corpus] build failed: {error}", flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def deterministic_select_subset(values: Iterable[str], subset_size: int, seed: int, salt: str) -> list[str]:
    if subset_size <= 0:
        return []
    ordered = sorted(values)
    rng = random.Random(f"{seed}|subset|{salt}")
    rng.shuffle(ordered)
    return sorted(ordered[:subset_size])


def stratified_split_counts(size: int) -> tuple[int, int, int]:
    if size <= 0:
        return 0, 0, 0
    val_count = int(round(size * 0.1))
    test_count = int(round(size * 0.1))
    train_count = size - val_count - test_count
    if train_count < 1:
        deficit = 1 - train_count
        while deficit > 0 and test_count > 0:
            test_count -= 1
            deficit -= 1
        while deficit > 0 and val_count > 0:
            val_count -= 1
            deficit -= 1
        train_count = size - val_count - test_count
    return train_count, val_count, test_count


def load_store_gene_universe(store_dir: Path) -> list[str]:
    binary_data_path = store_dir / "genes_data.bin"
    binary_offsets_path = store_dir / "genes_offsets.bin"
    if binary_data_path.exists() and binary_offsets_path.exists():
        data = binary_data_path.read_bytes()
        offsets_bytes = binary_offsets_path.read_bytes()
        if len(offsets_bytes) % 8 != 0:
            raise ValueError(f"Malformed binary gene offsets file: {binary_offsets_path}")
        offsets = struct.unpack(f"<{len(offsets_bytes) // 8}Q", offsets_bytes)
        if offsets and offsets[-1] != len(data):
            raise ValueError(f"Binary gene offsets do not match data length: {binary_offsets_path}")
        gene_ids = [
            data[int(start):int(end)].decode("utf-8")
            for start, end in zip(offsets[:-1], offsets[1:])
        ]
        return sorted(set(gene_ids))

    genes_path = store_dir / "genes.tsv"
    if not genes_path.exists():
        raise FileNotFoundError(
            "HumanNet store gene table not found. Expected binary metadata "
            f"({binary_data_path}, {binary_offsets_path}) or text table {genes_path}."
        )

    gene_ids = []
    with genes_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            columns = stripped.split("\t")
            if len(columns) == 1:
                gene_id = columns[0]
            elif len(columns) >= 2:
                gene_id = columns[1]
            else:
                raise ValueError(f"Malformed gene row {line_number} in {genes_path}.")
            gene_ids.append(str(gene_id))
    return sorted(set(gene_ids))


def parse_dendrogram(path: Path) -> dict[int, DendrogramNode]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"node_id", "left_id", "right_id", "height", "label"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Dendrogram header must be exactly: {', '.join(sorted(expected))}.")

        nodes: dict[int, DendrogramNode] = {}
        for row in reader:
            node = DendrogramNode(
                node_id=int(row["node_id"]),
                left_id=int(row["left_id"]),
                right_id=int(row["right_id"]),
                height=float(row["height"]),
                label=str(row["label"]),
            )
            if node.node_id in nodes:
                raise ValueError(f"Duplicate dendrogram node_id: {node.node_id}.")
            if (node.left_id == -1) != (node.right_id == -1):
                raise ValueError(f"Node {node.node_id} must have either two children or no children.")
            nodes[node.node_id] = node

    child_ids = {child for node in nodes.values() for child in (node.left_id, node.right_id) if child != -1}
    missing = sorted(child_id for child_id in child_ids if child_id not in nodes)
    if missing:
        raise ValueError(f"Dendrogram references missing child nodes: {missing[:10]}.")
    roots = sorted(set(nodes) - child_ids)
    if len(roots) != 1:
        raise ValueError(f"Dendrogram must contain exactly one root; found {len(roots)}.")
    return nodes


def size_bin(size: int) -> str | None:
    if 5 <= size <= 10:
        return "small"
    if 11 <= size <= 15:
        return "medium"
    if 16 <= size <= 30:
        return "large"
    return None


def compute_subtree_genes(
    nodes: dict[int, DendrogramNode],
    allowed_gene_ids: set[str],
) -> tuple[dict[int, list[str]], dict[int, int]]:
    filtered: dict[int, list[str]] = {}
    raw_counts: dict[int, int] = {}
    pending = set(nodes)
    while pending:
        progressed = False
        for node_id in sorted(list(pending), reverse=True):
            node = nodes[node_id]
            if node.is_leaf:
                raw_counts[node_id] = 1
                filtered[node_id] = [node.label] if node.label in allowed_gene_ids else []
                pending.remove(node_id)
                progressed = True
                continue
            if node.left_id in filtered and node.right_id in filtered:
                raw_counts[node_id] = raw_counts[node.left_id] + raw_counts[node.right_id]
                filtered[node_id] = sorted(set(filtered[node.left_id]) | set(filtered[node.right_id]))
                pending.remove(node_id)
                progressed = True
        if not progressed:
            raise ValueError("Could not resolve dendrogram subtree genes; check for cycles.")
    return filtered, raw_counts


def build_parent_index(nodes: dict[int, DendrogramNode]) -> dict[int, int]:
    parent_by_node: dict[int, int] = {}
    for node in nodes.values():
        if node.is_leaf:
            continue
        for child_id in (node.left_id, node.right_id):
            if child_id in parent_by_node:
                raise ValueError(f"Dendrogram child {child_id} has multiple parents.")
            parent_by_node[child_id] = node.node_id
    return parent_by_node


def build_dendrogram_distance_index(
    nodes: dict[int, DendrogramNode],
    allowed_gene_ids: set[str],
) -> DendrogramDistanceIndex:
    subtree_genes, _ = compute_subtree_genes(nodes, allowed_gene_ids)
    candidate_gene_universe = {
        node.label for node in nodes.values() if node.is_leaf and node.label in allowed_gene_ids
    }
    return DendrogramDistanceIndex(
        nodes=nodes,
        parent_by_node=build_parent_index(nodes),
        subtree_genes=subtree_genes,
        candidate_gene_universe=candidate_gene_universe,
    )


def extract_modules(
    nodes: dict[int, DendrogramNode],
    allowed_gene_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subtree_genes, raw_counts = compute_subtree_genes(nodes, allowed_gene_ids)
    candidates = []
    for node_id, node in nodes.items():
        if node.is_leaf:
            continue
        genes = subtree_genes[node_id]
        bin_name = size_bin(len(genes))
        if bin_name is None:
            continue
        candidates.append(
            {
                "source_node_id": node_id,
                "left_id": node.left_id,
                "right_id": node.right_id,
                "height": node.height,
                "gene_ids": genes,
                "gene_symbols": list(genes),
                "size": len(genes),
                "size_bin": bin_name,
                "raw_leaf_count": raw_counts[node_id],
                "filtered_out_leaf_count": raw_counts[node_id] - len(genes),
            }
        )

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[tuple(candidate["gene_ids"])].append(candidate)

    modules = []
    duplicate_count = 0
    for duplicate_rows in grouped.values():
        duplicate_rows.sort(key=lambda row: (row["height"], row["source_node_id"]))
        primary = copy.deepcopy(duplicate_rows[0])
        duplicate_node_ids = [row["source_node_id"] for row in duplicate_rows[1:]]
        primary["duplicate_source_node_ids"] = duplicate_node_ids
        duplicate_count += len(duplicate_node_ids)
        modules.append(primary)

    modules.sort(key=lambda row: (row["size_bin"], row["size"], row["source_node_id"]))
    for module in modules:
        module["module_id"] = f"gw_dendrogram_module_{module['source_node_id']:06d}"

    summary = {
        "raw_node_count": len(nodes),
        "raw_leaf_count": sum(1 for node in nodes.values() if node.is_leaf),
        "candidate_module_count": len(candidates),
        "deduplicated_module_count": len(modules),
        "duplicate_module_count": duplicate_count,
        "module_count_by_size_bin": dict(Counter(module["size_bin"] for module in modules)),
    }
    return modules, summary


def assign_splits(modules: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    assignments = []
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for module in modules:
        strata[module["size_bin"]].append(module)

    for bin_name in sorted(strata):
        rows = sorted(strata[bin_name], key=lambda row: row["module_id"])
        rng = random.Random(f"{seed}|split|{bin_name}")
        rng.shuffle(rows)
        train_count, val_count, test_count = stratified_split_counts(len(rows))
        for index, row in enumerate(rows):
            row_copy = copy.deepcopy(row)
            if index < train_count:
                row_copy["split"] = "train"
            elif index < train_count + val_count:
                row_copy["split"] = "val"
            else:
                row_copy["split"] = "test"
            assignments.append(row_copy)

    return sorted(assignments, key=lambda row: row["module_id"])


def assign_difficulties(modules: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    modules_by_id = {module["module_id"]: copy.deepcopy(module) for module in modules}
    for module in modules_by_id.values():
        module["difficulty_by_task"] = {}

    for task_type in ("recovery", "refinement", "none"):
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for module in modules_by_id.values():
            strata[(module["split"], module["size_bin"])].append(module)
        for key in sorted(strata):
            rows = sorted(strata[key], key=lambda row: row["module_id"])
            rng = random.Random(f"{seed}|difficulty|{task_type}|{key[0]}|{key[1]}")
            rng.shuffle(rows)
            for index, row in enumerate(rows):
                row["difficulty_by_task"][task_type] = DIFFICULTIES[index % len(DIFFICULTIES)]

    return sorted(modules_by_id.values(), key=lambda row: row["module_id"])


def recovery_drop_count(size: int, difficulty: str) -> int:
    if difficulty == "easy":
        raw = 1
    elif difficulty == "medium":
        raw = max(1, round(0.20 * size))
    elif difficulty == "hard":
        raw = max(2, round(0.33 * size))
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}.")
    return min(raw, max(1, size - 2))


def noise_gene_count(size: int, difficulty: str) -> int:
    if difficulty == "easy":
        return 1
    if difficulty == "medium":
        return max(1, round(0.20 * size))
    if difficulty == "hard":
        return max(2, round(0.33 * size))
    raise ValueError(f"Unknown difficulty: {difficulty}.")


def dendrogram_distance_groups(
    *,
    module: dict[str, Any],
    distance_index: DendrogramDistanceIndex,
    candidate_gene_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    module_node_id = int(module["source_node_id"])
    module_height = float(distance_index.nodes[module_node_id].height)
    target_set = set(module["gene_ids"])
    candidate_set = (
        set(distance_index.candidate_gene_universe)
        if candidate_gene_ids is None
        else set(candidate_gene_ids)
    ) - target_set
    groups = []
    current_node_id = module_node_id
    while current_node_id in distance_index.parent_by_node:
        parent_id = distance_index.parent_by_node[current_node_id]
        parent = distance_index.nodes[parent_id]
        if parent.left_id == current_node_id:
            sibling_id = parent.right_id
        elif parent.right_id == current_node_id:
            sibling_id = parent.left_id
        else:
            raise ValueError(f"Parent index is inconsistent for node {current_node_id}.")
        genes = [gene_id for gene_id in distance_index.subtree_genes[sibling_id] if gene_id in candidate_set]
        if genes:
            groups.append(
                {
                    "distance": max(0.0, float(parent.height) - module_height),
                    "lca_node_id": parent_id,
                    "gene_ids": genes,
                }
            )
        current_node_id = parent_id
    return sorted(groups, key=lambda row: (row["distance"], row["lca_node_id"]))


def _distance_entries_for_percentile_band(
    groups: list[dict[str, Any]],
    percentile_range: tuple[float, float],
) -> list[dict[str, Any]]:
    total = sum(len(group["gene_ids"]) for group in groups)
    if total <= 0:
        return []

    lower, upper = percentile_range
    lower = min(max(lower, 0.0), 1.0)
    upper = min(max(upper, 0.0), 1.0)
    if lower > upper:
        return []

    if total == 1:
        start_rank = 0
        end_rank = 0 if lower <= 1.0 <= upper else -1
    else:
        start_rank = int(math.ceil(lower * (total - 1) - 1e-12))
        end_rank = int(math.floor(upper * (total - 1) + 1e-12))
    if end_rank < start_rank:
        return []

    entries = []
    rank_offset = 0
    for group in groups:
        genes = list(group["gene_ids"])
        group_start = rank_offset
        group_end = rank_offset + len(genes) - 1
        overlap_start = max(start_rank, group_start)
        overlap_end = min(end_rank, group_end)
        if overlap_start <= overlap_end:
            for rank in range(overlap_start, overlap_end + 1):
                gene_id = genes[rank - group_start]
                percentile = 1.0 if total == 1 else rank / (total - 1)
                entries.append(
                    {
                        "gene_id": gene_id,
                        "distance": group["distance"],
                        "lca_node_id": group["lca_node_id"],
                        "percentile": percentile,
                    }
                )
        rank_offset += len(genes)
    return entries


def select_dendrogram_negative_genes(
    *,
    module: dict[str, Any],
    distance_index: DendrogramDistanceIndex,
    candidate_gene_ids: Iterable[str] | None,
    sample_size: int,
    difficulty: str,
    seed: int,
    salt: str,
    gene_to_modules: dict[str, set[str]] | None = None,
    conflict_free: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    if sample_size <= 0:
        return [], {"selection_mode": "empty", "candidate_count": 0}

    preferred_band = DENDROGRAM_DISTANCE_PERCENTILE_BANDS[difficulty]
    fallback_band = DENDROGRAM_DISTANCE_FALLBACK_PERCENTILE_BANDS[difficulty]
    groups = dendrogram_distance_groups(
        module=module,
        distance_index=distance_index,
        candidate_gene_ids=candidate_gene_ids,
    )
    candidate_count = sum(len(group["gene_ids"]) for group in groups)
    preferred_entries = _distance_entries_for_percentile_band(groups, preferred_band)
    fallback_entries = _distance_entries_for_percentile_band(groups, fallback_band)
    preferred_by_gene = {entry["gene_id"]: entry for entry in preferred_entries}
    fallback_by_gene = {entry["gene_id"]: entry for entry in fallback_entries}
    selected = _sample_from_candidates(
        preferred_by_gene,
        sample_size=sample_size,
        seed=seed,
        salt=f"{salt}|preferred",
        gene_to_modules=gene_to_modules,
        conflict_free=conflict_free,
        initial_used_modules=None,
    )
    selection_mode = "preferred_band"
    if len(selected) < sample_size:
        selected = _sample_from_candidates(
            fallback_by_gene,
            sample_size=sample_size,
            seed=seed,
            salt=f"{salt}|fallback",
            gene_to_modules=gene_to_modules,
            conflict_free=conflict_free,
            initial_used_modules=None,
        )
        selection_mode = "fallback_percentile_band"

    if len(selected) < sample_size:
        raise ValueError(
            f"Not enough {difficulty} dendrogram-distance candidates for {salt}: "
            f"needed {sample_size}, preferred={len(preferred_entries)}, "
            f"fallback={len(fallback_entries)}, total={candidate_count}."
        )

    selected_entry_lookup = {**fallback_by_gene, **preferred_by_gene}
    return selected, {
        "selection_mode": selection_mode,
        "difficulty": difficulty,
        "distance_metric": "dendrogram_lca_height_delta",
        "distance_definition": "height(LCA(module_root, candidate_leaf)) - height(module_root)",
        "preferred_percentile_range": list(preferred_band),
        "fallback_percentile_range": list(fallback_band),
        "candidate_count": candidate_count,
        "preferred_candidate_count": len(preferred_entries),
        "fallback_candidate_count": len(fallback_entries),
        "selected_distances": {
            gene_id: selected_entry_lookup[gene_id]["distance"] for gene_id in selected
        },
        "selected_lca_node_ids": {
            gene_id: selected_entry_lookup[gene_id]["lca_node_id"] for gene_id in selected
        },
        "selected_percentiles": {
            gene_id: selected_entry_lookup[gene_id]["percentile"] for gene_id in selected
        },
    }


def _sample_from_candidates(
    candidates: Iterable[str],
    *,
    sample_size: int,
    seed: int,
    salt: str,
    gene_to_modules: dict[str, set[str]] | None,
    conflict_free: bool,
    initial_used_modules: set[str] | None,
) -> list[str]:
    ordered = sorted(set(candidates))
    for attempt in range(32):
        rng = random.Random(f"{seed}|sample|{salt}|{attempt}")
        shuffled = ordered[:]
        rng.shuffle(shuffled)
        selected: list[str] = []
        used_modules: set[str] = set(initial_used_modules or set())
        for gene_id in shuffled:
            if conflict_free:
                memberships = gene_to_modules.get(gene_id, set()) if gene_to_modules else set()
                if memberships & used_modules:
                    continue
                used_modules.update(memberships)
            selected.append(gene_id)
            if len(selected) == sample_size:
                return sorted(selected)
    return []


def _memberships_for_genes(
    gene_ids: Iterable[str],
    gene_to_modules: dict[str, set[str]] | None,
) -> set[str]:
    if gene_to_modules is None:
        return set()
    memberships: set[str] = set()
    for gene_id in gene_ids:
        memberships.update(gene_to_modules.get(gene_id, set()))
    return memberships


def build_gene_to_modules(modules: list[dict[str, Any]]) -> dict[str, set[str]]:
    gene_to_modules: dict[str, set[str]] = defaultdict(set)
    for module in modules:
        for gene_id in module["gene_ids"]:
            gene_to_modules[gene_id].add(module["module_id"])
    return gene_to_modules


def build_task_prototypes(
    *,
    modules: list[dict[str, Any]],
    candidate_gene_universe: set[str],
    distance_index: DendrogramDistanceIndex,
    seed: int,
    stats: BuildStats | None = None,
) -> list[dict[str, Any]]:
    stats = stats or BuildStats()
    gene_to_modules = build_gene_to_modules(modules)
    prototypes: list[dict[str, Any]] = []

    for index, module in enumerate(modules, start=1):
        module_id = module["module_id"]
        target_gene_ids = list(module["gene_ids"])

        prototypes.append(
            {
                "prototype_id": f"{module_id}.explanation.{EXPLANATION_DIFFICULTY}",
                "task_type": "explanation",
                "difficulty": EXPLANATION_DIFFICULTY,
                "split": module["split"],
                "size_bin": module["size_bin"],
                "source_module_id": module_id,
                "input_gene_ids": list(target_gene_ids),
                "relationship_status": POSITIVE_RELATIONSHIP_STATUS,
                "sampling_metadata": {},
            }
        )

        recovery_difficulty = module["difficulty_by_task"]["recovery"]
        drop_count = recovery_drop_count(len(target_gene_ids), recovery_difficulty)
        dropped_genes = deterministic_select_subset(
            target_gene_ids,
            subset_size=drop_count,
            seed=seed,
            salt=f"{module_id}|recovery|{recovery_difficulty}",
        )
        prototypes.append(
            {
                "prototype_id": f"{module_id}.recovery.{recovery_difficulty}",
                "task_type": "recovery",
                "difficulty": recovery_difficulty,
                "split": module["split"],
                "size_bin": module["size_bin"],
                "source_module_id": module_id,
                "input_gene_ids": sorted(set(target_gene_ids) - set(dropped_genes)),
                "relationship_status": POSITIVE_RELATIONSHIP_STATUS,
                "sampling_metadata": {"dropped_gene_ids": dropped_genes},
            }
        )

        refinement_difficulty = module["difficulty_by_task"]["refinement"]
        add_count = noise_gene_count(len(target_gene_ids), refinement_difficulty)
        try:
            noise_gene_ids, metadata = select_dendrogram_negative_genes(
                module=module,
                distance_index=distance_index,
                candidate_gene_ids=candidate_gene_universe,
                sample_size=add_count,
                difficulty=refinement_difficulty,
                seed=seed,
                salt=f"{module_id}|refinement|{refinement_difficulty}",
            )
            prototypes.append(
                {
                    "prototype_id": f"{module_id}.refinement.{refinement_difficulty}",
                    "task_type": "refinement",
                    "difficulty": refinement_difficulty,
                    "split": module["split"],
                    "size_bin": module["size_bin"],
                    "source_module_id": module_id,
                    "input_gene_ids": sorted(set(target_gene_ids) | set(noise_gene_ids)),
                    "relationship_status": POSITIVE_RELATIONSHIP_STATUS,
                    "sampling_metadata": {
                        "noise_gene_ids": noise_gene_ids,
                        "dendrogram_negative_sampling": metadata,
                    },
                }
            )
        except ValueError:
            stats.skipped_refinement[(module["split"], module["size_bin"], refinement_difficulty)] += 1

        none_difficulty = module["difficulty_by_task"]["none"]
        try:
            none_gene_ids, metadata = select_dendrogram_negative_genes(
                module=module,
                distance_index=distance_index,
                candidate_gene_ids=candidate_gene_universe,
                sample_size=len(target_gene_ids),
                difficulty=none_difficulty,
                seed=seed,
                salt=f"{module_id}|none|{none_difficulty}",
                gene_to_modules=gene_to_modules,
                conflict_free=True,
            )
            prototypes.append(
                {
                    "prototype_id": f"{module_id}.none.{none_difficulty}",
                    "task_type": "none",
                    "difficulty": none_difficulty,
                    "split": module["split"],
                    "size_bin": module["size_bin"],
                    "source_module_id": None,
                    "anchor_module_id": module_id,
                    "input_gene_ids": none_gene_ids,
                    "relationship_status": NONE_RELATIONSHIP_STATUS,
                    "sampling_metadata": {"dendrogram_negative_sampling": metadata},
                }
            )
        except ValueError:
            stats.skipped_none[(module["split"], module["size_bin"], none_difficulty)] += 1

        if index % 250 == 0:
            print(f"[gw_dendrogram_corpus] built prototypes for {index} modules", flush=True)

    return sorted(prototypes, key=lambda row: row["prototype_id"])


def balance_prototypes(
    prototypes: list[dict[str, Any]],
    *,
    seed: int,
    stats: BuildStats | None = None,
) -> list[dict[str, Any]]:
    stats = stats or BuildStats()
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for prototype in prototypes:
        groups[(prototype["split"], prototype["size_bin"])][prototype["task_type"]].append(prototype)

    balanced: list[dict[str, Any]] = []
    balance_summary = {}
    for key in sorted(groups):
        task_rows = groups[key]
        counts = {task_type: len(task_rows.get(task_type, [])) for task_type in TASK_TYPES}
        target = min(counts.values())
        balance_summary[f"{key[0]}|{key[1]}"] = {"pre_balance_counts": counts, "selected_per_task_type": target}
        if target <= 0:
            continue
        for task_type in TASK_TYPES:
            rows = sorted(task_rows[task_type], key=lambda row: row["prototype_id"])
            if len(rows) > target:
                selected_ids = set(
                    deterministic_select_subset(
                        [row["prototype_id"] for row in rows],
                        subset_size=target,
                        seed=seed,
                        salt=f"balance|{key[0]}|{key[1]}|{task_type}",
                    )
                )
                rows = [row for row in rows if row["prototype_id"] in selected_ids]
            balanced.extend(rows)

    stats.balance = balance_summary
    return sorted(balanced, key=lambda row: row["prototype_id"])


def build_graph_query_spec(store_dir: Path, seed_gene_ids: list[str], seed_gene_symbols: list[str]) -> dict[str, Any]:
    return {
        "store_dir": str(store_dir),
        "operator": "induce_subgraph",
        "layer_scope": "all",
        "materialized": False,
        "seed_gene_ids": seed_gene_ids,
        "seed_gene_symbols": seed_gene_symbols,
    }


def build_query_text(task_type: str, evidence_mode: str, seed_gene_symbols: list[str]) -> tuple[str, str]:
    genes = ", ".join(seed_gene_symbols)
    if task_type == "explanation":
        templates = {
            "minimal": "Explain the strongest shared mechanism supported by the following gene set: {genes}.",
            "graph": (
                "Using the provided seed genes and graph query specification, explain the strongest "
                "shared mechanism supported by: {genes}."
            ),
        }
    elif task_type == "recovery":
        templates = {
            "minimal": (
                "Starting from this partial seed set, recover the most coherent related module and explain "
                "its shared mechanism: {genes}."
            ),
            "graph": (
                "Use the seed genes and graph query specification to recover the most coherent related "
                "module and explain its shared mechanism: {genes}."
            ),
        }
    elif task_type == "refinement":
        templates = {
            "minimal": (
                "Refine this seed set by removing unrelated genes, then explain the shared mechanism of "
                "the remaining module: {genes}."
            ),
            "graph": (
                "Using the seed genes and graph query specification, refine this seed set by removing "
                "unrelated genes, then explain the shared mechanism of the remaining module: {genes}."
            ),
        }
    else:
        templates = {
            "minimal": (
                "Determine whether this gene set provides insufficient evidence for one shared functional "
                "module. If so, return insufficient support: {genes}."
            ),
            "graph": (
                "Using the seed genes and graph query specification, determine whether this gene set "
                "provides insufficient evidence for one shared functional module. If so, return "
                "insufficient support: {genes}."
            ),
        }
    return templates[evidence_mode].format(genes=genes), f"{task_type}.{evidence_mode}.v1"


def materialize_tasks(
    *,
    modules: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    dendrogram_path: Path,
    store_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    modules_by_id = {module["module_id"]: module for module in modules}
    tasks = []
    for prototype in prototypes:
        target_module = modules_by_id.get(prototype.get("source_module_id"))
        seed_gene_ids = list(prototype["input_gene_ids"])
        seed_gene_symbols = list(seed_gene_ids)
        for evidence_mode in EVIDENCE_MODES:
            query_text, query_template_id = build_query_text(
                prototype["task_type"],
                evidence_mode,
                seed_gene_symbols,
            )
            visible_inputs = {
                "seed_gene_ids": seed_gene_ids,
                "seed_gene_symbols": seed_gene_symbols,
                "context_text": None,
                "graph_query_spec": (
                    build_graph_query_spec(store_dir, seed_gene_ids, seed_gene_symbols)
                    if evidence_mode == "graph"
                    else None
                ),
                "structured_annotations": None,
            }
            if target_module is None:
                hidden_target = {
                    "target_gene_ids": None,
                    "target_gene_symbols": None,
                    "relationship_status": NONE_RELATIONSHIP_STATUS,
                }
            else:
                hidden_target = {
                    "target_gene_ids": target_module["gene_ids"],
                    "target_gene_symbols": target_module["gene_symbols"],
                    "relationship_status": POSITIVE_RELATIONSHIP_STATUS,
                }

            task_id = f"{prototype['prototype_id']}.{evidence_mode}"
            provenance = {
                "source": SOURCE_NAME,
                "source_module_id": prototype.get("source_module_id"),
                "anchor_module_id": prototype.get("anchor_module_id"),
                "dendrogram_path": str(dendrogram_path),
                "store_dir": str(store_dir),
                "generation_seed": seed,
                "sampling_metadata": prototype.get("sampling_metadata", {}),
            }
            if target_module is not None:
                provenance["source_node_id"] = target_module["source_node_id"]

            tasks.append(
                {
                    "task_id": task_id,
                    "split": prototype["split"],
                    "task_type": prototype["task_type"],
                    "difficulty": prototype["difficulty"],
                    "query_text": query_text,
                    "query_template_id": query_template_id,
                    "evidence_mode": evidence_mode,
                    "visible_inputs": visible_inputs,
                    "hidden_target": hidden_target,
                    "mechanism_labels": None,
                    "normalization": {
                        "gene_id_namespace": "ensembl.gene",
                        "visible_gene_mappings": [
                            {
                                "ensembl_gene_id": gene_id,
                                "display_symbol": gene_id,
                                "resolved_via": "dendrogram_leaf",
                            }
                            for gene_id in seed_gene_ids
                        ],
                    },
                    "provenance": provenance,
                }
            )
    return sorted(tasks, key=lambda row: row["task_id"])


def build_split_report(
    modules: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    extraction_summary: dict[str, Any],
    stats: BuildStats,
) -> dict[str, Any]:
    module_counts = Counter(module["split"] for module in modules)
    module_size_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    for module in modules:
        module_size_counts[module["split"]][module["size_bin"]] += 1
    task_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    task_size_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    module_size_lookup = {module["module_id"]: module["size_bin"] for module in modules}
    for task in tasks:
        task_counts[task["split"]][task["task_type"]] += 1
        source_module_id = task["provenance"].get("source_module_id") or task["provenance"].get("anchor_module_id")
        size_bin_name = module_size_lookup.get(source_module_id)
        if size_bin_name:
            task_size_counts[task["split"]][f"{size_bin_name}|{task['task_type']}"] += 1

    return {
        "extraction_summary": extraction_summary,
        "module_count_by_split": dict(module_counts),
        "module_count_by_split_and_size_bin": {
            split: dict(module_size_counts[split]) for split in SPLITS
        },
        "prototype_count_by_type": dict(Counter(row["task_type"] for row in prototypes)),
        "task_count_by_split_and_type": {split: dict(task_counts[split]) for split in SPLITS},
        "task_count_by_split_size_bin_and_type": {
            split: dict(task_size_counts[split]) for split in SPLITS
        },
        "skipped_refinement": {str(key): value for key, value in stats.skipped_refinement.items()},
        "skipped_none": {str(key): value for key, value in stats.skipped_none.items()},
        "balance": stats.balance,
    }


def build_manifest(
    *,
    dendrogram_path: Path,
    store_dir: Path,
    out_dir: Path,
    seed: int,
    modules: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    extraction_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "dendrogram_path": str(dendrogram_path),
        "store_dir": str(store_dir),
        "out_dir": str(out_dir),
        "seed": seed,
        "module_size_bins": {"small": [5, 10], "medium": [11, 15], "large": [16, 30]},
        "negative_sampling": {
            "method": "dendrogram_lca_height_delta",
            "distance_definition": "height(LCA(module_root, candidate_leaf)) - height(module_root)",
            "preferred_percentile_bands": DENDROGRAM_DISTANCE_PERCENTILE_BANDS,
            "fallback_percentile_bands": DENDROGRAM_DISTANCE_FALLBACK_PERCENTILE_BANDS,
            "difficulty_semantics": {
                "easy": "0-25th percentile outside-module leaves after nearest-to-farthest sorting",
                "medium": "25-50th percentile outside-module leaves after nearest-to-farthest sorting",
                "hard": "50-75th percentile outside-module leaves after nearest-to-farthest sorting",
            },
        },
        "evidence_modes": list(EVIDENCE_MODES),
        "module_count": len(modules),
        "prototype_count": len(prototypes),
        "task_count": len(tasks),
        "task_count_by_split": dict(Counter(task["split"] for task in tasks)),
        "task_count_by_type": dict(Counter(task["task_type"] for task in tasks)),
        "task_count_by_family": {
            ("insufficient_evidence" if task_type == "none" else task_type): count
            for task_type, count in Counter(task["task_type"] for task in tasks).items()
        },
        "extraction_summary": extraction_summary,
    }


def build_module_jsonl_rows(modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for module in modules:
        rows.append(
            {
                "module_id": module["module_id"],
                "split": module["split"],
                "source": SOURCE_NAME,
                "source_node_id": module["source_node_id"],
                "left_id": module["left_id"],
                "right_id": module["right_id"],
                "height": module["height"],
                "size": module["size"],
                "size_bin": module["size_bin"],
                "gene_ids": module["gene_ids"],
                "gene_symbols": module["gene_symbols"],
                "raw_leaf_count": module["raw_leaf_count"],
                "filtered_out_leaf_count": module["filtered_out_leaf_count"],
                "duplicate_source_node_ids": module["duplicate_source_node_ids"],
                "difficulty_by_task": module.get("difficulty_by_task", {}),
            }
        )
    return sorted(rows, key=lambda row: row["module_id"])


def build_gw_dendrogram_corpus(
    *,
    dendrogram_path: Path,
    store_dir: Path,
    out_dir: Path,
    seed: int = 42,
    progress_path: Path | None = None,
    allowed_gene_ids: set[str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_path or out_dir / "progress.json"
    tracker = ProgressTracker(progress_path)
    tracker.set_context(
        {
            "dendrogram_path": str(dendrogram_path),
            "store_dir": str(store_dir),
            "out_dir": str(out_dir),
            "seed": seed,
        }
    )
    stats = BuildStats()

    try:
        tracker.start("load_store_genes", unit="genes")
        if allowed_gene_ids is None:
            store_gene_ids = load_store_gene_universe(store_dir)
            allowed_gene_ids = set(store_gene_ids)
        else:
            store_gene_ids = sorted(allowed_gene_ids)
        tracker.update(
            completed=len(store_gene_ids),
            total=len(store_gene_ids),
            unit="genes",
            metrics={"store_gene_count": len(store_gene_ids)},
        )

        tracker.start("parse_dendrogram", unit="nodes")
        nodes = parse_dendrogram(dendrogram_path)
        tracker.update(
            completed=len(nodes),
            total=len(nodes),
            unit="nodes",
            metrics={"raw_node_count": len(nodes)},
        )

        tracker.start("extract_modules", unit="modules")
        modules, extraction_summary = extract_modules(nodes, allowed_gene_ids)
        distance_index = build_dendrogram_distance_index(nodes, allowed_gene_ids)
        candidate_gene_universe = distance_index.candidate_gene_universe
        tracker.update(
            completed=len(modules),
            total=len(modules),
            unit="modules",
            metrics=extraction_summary,
        )

        tracker.start("assign_splits", total=len(modules), unit="modules")
        modules = assign_splits(modules, seed)
        tracker.update(completed=len(modules), total=len(modules), unit="modules")

        tracker.start("assign_difficulties", total=len(modules), unit="modules")
        modules = assign_difficulties(modules, seed)
        tracker.update(completed=len(modules), total=len(modules), unit="modules")

        tracker.start("build_prototypes", total=len(modules), unit="modules")
        prototypes = build_task_prototypes(
            modules=modules,
            candidate_gene_universe=candidate_gene_universe,
            distance_index=distance_index,
            seed=seed,
            stats=stats,
        )
        tracker.update(
            completed=len(modules),
            total=len(modules),
            unit="modules",
            metrics={
                "prototype_count_before_balance": len(prototypes),
                "skipped_refinement_count": sum(stats.skipped_refinement.values()),
                "skipped_none_count": sum(stats.skipped_none.values()),
            },
        )

        tracker.start("balance_prototypes", unit="prototypes")
        prototypes = balance_prototypes(prototypes, seed=seed, stats=stats)
        tracker.update(
            completed=len(prototypes),
            total=len(prototypes),
            unit="prototypes",
            metrics={"prototype_count_after_balance": len(prototypes)},
        )

        tracker.start("materialize_tasks", total=len(prototypes) * len(EVIDENCE_MODES), unit="tasks")
        tasks = materialize_tasks(
            modules=modules,
            prototypes=prototypes,
            dendrogram_path=dendrogram_path,
            store_dir=store_dir,
            seed=seed,
        )
        tracker.update(completed=len(tasks), total=len(tasks), unit="tasks", metrics={"task_count": len(tasks)})

        module_rows = build_module_jsonl_rows(modules)
        split_report = build_split_report(modules, prototypes, tasks, extraction_summary, stats)
        manifest = build_manifest(
            dendrogram_path=dendrogram_path,
            store_dir=store_dir,
            out_dir=out_dir,
            seed=seed,
            modules=modules,
            prototypes=prototypes,
            tasks=tasks,
            extraction_summary=extraction_summary,
        )

        tracker.start("write_outputs", total=4 + len(SPLITS), unit="files")
        write_json(out_dir / "manifest.json", manifest)
        tracker.update(completed=1, total=4 + len(SPLITS), unit="files", message="Wrote manifest.json.")
        write_json(out_dir / "split_report.json", split_report)
        tracker.update(completed=2, total=4 + len(SPLITS), unit="files", message="Wrote split_report.json.")
        write_jsonl(out_dir / "modules.jsonl", module_rows)
        tracker.update(completed=3, total=4 + len(SPLITS), unit="files", message="Wrote modules.jsonl.")
        write_jsonl(out_dir / "prototypes.jsonl", prototypes)
        tracker.update(completed=4, total=4 + len(SPLITS), unit="files", message="Wrote prototypes.jsonl.")
        for index, split in enumerate(SPLITS, start=1):
            split_tasks = [task for task in tasks if task["split"] == split]
            write_jsonl(out_dir / f"tasks.{split}.jsonl", split_tasks)
            tracker.update(
                completed=4 + index,
                total=4 + len(SPLITS),
                unit="files",
                message=f"Wrote tasks.{split}.jsonl.",
            )

        tracker.complete(metrics={"module_count": len(modules), "task_count": len(tasks)})
        return {
            "manifest": manifest,
            "split_report": split_report,
            "modules": module_rows,
            "prototypes": prototypes,
            "tasks": tasks,
            "progress_path": progress_path,
        }
    except Exception as error:
        tracker.fail(error)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the genome-wide dendrogram canonical corpus.")
    parser.add_argument("--dendrogram-path", type=Path, default=DEFAULT_DENDROGRAM_PATH)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_gw_dendrogram_corpus(
        dendrogram_path=args.dendrogram_path,
        store_dir=args.store_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        progress_path=args.progress_path,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.out_dir / "manifest.json"),
                "module_count": result["manifest"]["module_count"],
                "task_count": result["manifest"]["task_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
