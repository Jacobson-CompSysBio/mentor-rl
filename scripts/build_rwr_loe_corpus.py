#!/usr/bin/env python3
"""Build a full-brain RWR-LOE module corpus.

The expensive full-brain step is a cache prewarm: run the MPI-capable RWR++
``rwr`` app over shard seed files, then postprocess the recorded encoding
matrix into per-seed LOE-style rank caches. Corpus materialization reads those
rank caches and emits the same task-row contract as the MENTOR dendrogram
corpus.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.rwr_hpc_app_backend import RwrHpcAppBackend
from runtime.rwr_hpc_cache import file_sha256
from runtime.rwr_hpc_requests import RwrLoeRequest
from runtime.rwr_hpc_structured_backend import RwrHpcStructuredBackend
from scripts.build_gw_dendrogram_corpus import (
    EXPLANATION_DIFFICULTY,
    NONE_RELATIONSHIP_STATUS,
    POSITIVE_RELATIONSHIP_STATUS,
    TASK_TYPES,
    EVIDENCE_MODES,
    deterministic_select_subset,
    load_store_gene_universe,
    noise_gene_count,
    recovery_drop_count,
    stratified_split_counts,
    build_graph_query_spec,
    build_query_text,
    write_json,
    write_jsonl,
)


DEFAULT_STORE_DIR = REPO_ROOT / "data" / "runtime" / "full_brain_multiplex_store"
DEFAULT_RWR_HPC_FLIST = Path(
    "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/data/full_brain_flist.tsv"
)
DEFAULT_RWR_HPC_BUILD_DIR = REPO_ROOT / "external" / "rwr_hpc" / "build_frontier"
DEFAULT_RWR_HPC_CACHE_DIR = REPO_ROOT / "data" / "runtime" / "rwr_hpc_cache"
DEFAULT_LOE_RANK_CACHE_DIR = REPO_ROOT / "data" / "runtime" / "rwr_loe_full_brain_rank_cache"
DEFAULT_MENTOR_CORPUS_DIR = REPO_ROOT / "data" / "gw_dendrogram_corpus_full_brain"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "rwr_loe_corpus_full_brain"

SPLITS = ("train", "val", "test")
DIFFICULTIES = ("easy", "medium", "hard")
SCHEMA_VERSION = "rwr-loe-corpus-v1"
SOURCE_NAME = "RWR_LOE_FULL_BRAIN"
RANK_CACHE_SCHEMA_VERSION = "rwr-loe-rank-cache-v1"
RANKING_SEMANTICS = "rwr_encoding_desc_min_rank_seed_excluded"
PARSER_VERSION = 1
EDGE_HEADER_TOKENS = {"source", "target"}
MATERIALIZE_CHECKPOINT_SCHEMA_VERSION = "rwr-loe-materialize-checkpoint-v1"
CHECKPOINT_FLUSH_INTERVAL = 25
DEFAULT_MATERIALIZE_WORKERS = 1

DEFAULT_MODULE_SIZES = {"small": 8, "medium": 13, "large": 23}
DEFAULT_MODULE_SELECTION_METHOD = "elbow"
MIN_ELBOW_MODULE_SIZE = 3
RANK_PERCENTILE_BANDS = {
    "easy": (0.75, 1.0),
    "medium": (0.25, 0.50),
    "hard": (0.0, 0.25),
}
RANK_FALLBACK_PERCENTILE_BANDS = {
    "easy": (0.50, 1.0),
    "medium": (0.25, 0.75),
    "hard": (0.0, 0.50),
}

BUILD_STAGES = (
    ("load_genes", "Load full-brain store gene universe"),
    ("assign_modules", "Assign LOE module sizes and splits"),
    ("load_rank_cache", "Load per-seed LOE rank cache"),
    ("build_prototypes", "Build LOE task prototypes"),
    ("balance_prototypes", "Balance task types"),
    ("materialize_tasks", "Materialize canonical LOE tasks"),
    ("write_outputs", "Write LOE corpus files"),
)


@dataclass
class BuildStats:
    skipped_modules: Counter = field(default_factory=Counter)
    skipped_refinement: Counter = field(default_factory=Counter)
    skipped_none: Counter = field(default_factory=Counter)
    balance: dict[str, Any] = field(default_factory=dict)
    rank_cache: dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Persist a small progress JSON file for long LOE builds."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stage_lookup = {stage: index for index, (stage, _) in enumerate(BUILD_STAGES, start=1)}
        timestamp = utc_now_iso()
        self.state: dict[str, Any] = {
            "status": "running",
            "current_stage": None,
            "current_stage_label": None,
            "stage_index": 0,
            "stage_count": len(BUILD_STAGES),
            "stage_progress": {"completed": 0, "total": None, "unit": None},
            "overall_progress": 0.0,
            "message": "Initialized RWR-LOE corpus build.",
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
        print(f"[rwr_loe_corpus] {self.state['stage_index']}/{len(BUILD_STAGES)} {entry['label']}", flush=True)

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
        self.state["message"] = "Completed RWR-LOE corpus build."
        self.state["metrics"].update(metrics)
        self._touch()
        self._recompute_overall()
        self._write()
        print("[rwr_loe_corpus] build complete", flush=True)

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
        print(f"[rwr_loe_corpus] build failed: {error}", flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_json_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_file_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:80]}_{digest}" if safe else digest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    tmp_path.replace(path)


def _jsonable_counter(counter: Counter) -> list[dict[str, Any]]:
    entries = []
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        entries.append(
            {
                "key": list(key) if isinstance(key, tuple) else key,
                "count": count,
            }
        )
    return entries


def _counter_from_jsonable(entries: Iterable[dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for entry in entries:
        key = entry["key"]
        if isinstance(key, list):
            key = tuple(key)
        counter[key] += int(entry.get("count", 0))
    return counter


def materialize_checkpoint_context(
    *,
    store_dir: Path,
    rwr_hpc_flist: Path,
    rank_cache_context: dict[str, Any],
    rank_cache_context_dir_path: Path,
    mentor_corpus_dir: Path,
    seed: int,
    max_genes: int | None,
    module_sizes: dict[str, int],
    module_selection_method: str,
) -> dict[str, Any]:
    return {
        "schema_version": MATERIALIZE_CHECKPOINT_SCHEMA_VERSION,
        "corpus_schema_version": SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "store_dir": str(store_dir),
        "rwr_hpc_flist": str(rwr_hpc_flist),
        "rank_cache_context_hash": stable_json_hash(rank_cache_context),
        "rank_cache_context_dir": str(rank_cache_context_dir_path),
        "mentor_corpus_dir": str(mentor_corpus_dir),
        "seed": seed,
        "max_genes": max_genes,
        "module_sizes": module_sizes,
        "module_selection_method": module_selection_method,
        "evidence_modes": list(EVIDENCE_MODES),
    }


def materialize_checkpoint_dir(out_dir: Path, context: dict[str, Any]) -> Path:
    return out_dir / "_materialize_checkpoints" / f"context_{stable_json_hash(context)[:16]}"


def _append_checkpoint_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _read_checkpoint_records(path: Path, *, key_field: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[rwr_loe_corpus] ignoring truncated checkpoint line {line_number} in {path}",
                    flush=True,
                )
                break
            key = record.get(key_field)
            if not key or key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def build_rank_cache_context(
    *,
    rwr_hpc_flist: Path,
    rwr_hpc_build_id: str,
    restart: float,
    delta: float,
    reduction_method: str,
    threshold: float,
    edgelist_has_headers: bool,
) -> dict[str, Any]:
    flist_hash = file_sha256(rwr_hpc_flist) if rwr_hpc_flist.exists() else f"missing:{rwr_hpc_flist}"
    return {
        "schema_version": RANK_CACHE_SCHEMA_VERSION,
        "network_flist": str(rwr_hpc_flist),
        "network_flist_sha256": flist_hash,
        "rwr_hpc_build_id": rwr_hpc_build_id,
        "restart": restart,
        "delta": delta,
        "reduction_method": reduction_method,
        "threshold": threshold,
        "edgelist_has_headers": edgelist_has_headers,
        "parser_version": PARSER_VERSION,
        "ranking_semantics": RANKING_SEMANTICS,
    }


def rank_cache_context_dir(rank_cache_dir: Path, context: dict[str, Any]) -> Path:
    return rank_cache_dir / f"context_{stable_json_hash(context)[:16]}"


def load_rank_cache_context_from_dir(context_dir: Path) -> dict[str, Any]:
    context_path = context_dir / "cache_context.json"
    if not context_path.exists():
        raise FileNotFoundError(f"Missing RWR-LOE rank-cache context file: {context_path}")
    return json.loads(context_path.read_text(encoding="utf-8"))


def rank_cache_path_for_seed(context_dir: Path, seed_gene_id: str) -> Path:
    return context_dir / "ranks" / f"{_safe_file_stem(seed_gene_id)}.ranks.tsv.gz"


def rank_cache_metadata_path_for_seed(context_dir: Path, seed_gene_id: str) -> Path:
    return context_dir / "ranks" / f"{_safe_file_stem(seed_gene_id)}.metadata.json"


def load_rwr_loe_gene_universe(store_dir: Path) -> list[str]:
    """Load store genes while dropping edge-list header tokens found in legacy stores."""

    gene_ids = [gene.upper() for gene in load_store_gene_universe(store_dir)]
    return sorted({gene for gene in gene_ids if gene.lower() not in EDGE_HEADER_TOKENS})


def loe_size_bin(size: int) -> str:
    if 5 <= size <= 10:
        return "small"
    if 11 <= size <= 15:
        return "medium"
    if size >= 16:
        return "large"
    return "other"


def geometric_elbow_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Mirror RWR++ elbow_point::elbow_point for rank/score curves."""

    if len(points) < 2:
        raise ValueError("Elbow point requires at least two points.")
    min_x_index = min(range(len(points)), key=lambda index: points[index][0])
    max_x_index = max(range(len(points)), key=lambda index: points[index][0])
    x_start, y_start = points[min_x_index]
    x_end, y_end = points[max_x_index]
    dx = x_end - x_start
    dy = y_end - y_start
    distances = [
        abs(dx * (y_start - y_value) - dy * (x_start - x_value))
        for x_value, y_value in points
    ]
    elbow_index = max(range(len(distances)), key=lambda index: distances[index])
    return points[elbow_index]


def select_elbow_module_ranked_genes(ranked_genes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select genes above the RWR++ geometric elbow cutoff.

    RWR++ keeps entries whose per-vector rank is strictly below the elbow
    x-coordinate. The cached LOE ranks already exclude the seed gene.
    """

    points = [(float(row["rank"]), float(row["score"])) for row in ranked_genes]
    if len(points) < 2:
        return [], {
            "method": "elbow",
            "status": "insufficient_ranked_genes",
            "elbow_rank_cutoff": None,
            "elbow_score": None,
            "retained_ranked_gene_count": 0,
        }
    elbow_rank, elbow_score = geometric_elbow_point(points)
    retained = [row for row in ranked_genes if float(row["rank"]) < elbow_rank]
    return retained, {
        "method": "elbow",
        "status": "ok",
        "elbow_rank_cutoff": elbow_rank,
        "elbow_score": elbow_score,
        "retention_rule": "rank < elbow_rank_cutoff",
        "retained_ranked_gene_count": len(retained),
        "ranked_gene_count": len(ranked_genes),
    }


def rank_scores_min_desc(
    node_scores: Iterable[tuple[str, float]],
    *,
    seed_gene_id: str,
) -> list[dict[str, Any]]:
    seed_key = seed_gene_id.upper()
    rows = [
        (str(gene_id).upper(), float(score))
        for gene_id, score in node_scores
        if str(gene_id).strip() and str(gene_id).upper() != seed_key and math.isfinite(float(score))
    ]
    rows.sort(key=lambda item: (-item[1], item[0]))

    ranked: list[dict[str, Any]] = []
    index = 0
    current_rank = 1
    while index < len(rows):
        score = rows[index][1]
        end = index + 1
        while end < len(rows) and rows[end][1] == score:
            end += 1
        for gene_id, tied_score in rows[index:end]:
            ranked.append({"gene": gene_id, "score": tied_score, "rank": current_rank})
        current_rank += end - index
        index = end
    return ranked


def write_seed_rank_cache(
    *,
    context_dir: Path,
    seed_gene_id: str,
    ranked_genes: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    rank_path = rank_cache_path_for_seed(context_dir, seed_gene_id)
    metadata_path = rank_cache_metadata_path_for_seed(context_dir, seed_gene_id)
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(rank_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("NodeNames", "Scores", "rank"), delimiter="\t")
        writer.writeheader()
        for row in ranked_genes:
            writer.writerow(
                {
                    "NodeNames": row["gene"],
                    "Scores": f"{float(row['score']):.16e}",
                    "rank": int(row["rank"]),
                }
            )
    write_json(metadata_path, metadata)
    return rank_path


def load_seed_rank_cache_path(
    rank_path: Path,
    *,
    max_rows: int | None = None,
    sort_rows: bool = True,
) -> list[dict[str, Any]]:
    if not rank_path.exists():
        raise FileNotFoundError(f"Missing RWR-LOE rank cache: {rank_path}")
    opener = gzip.open if rank_path.suffix == ".gz" else open
    with opener(rank_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for raw in reader:
            gene = str(raw.get("NodeNames", "")).strip().upper()
            if not gene:
                continue
            rows.append(
                {
                    "gene": gene,
                    "score": float(raw.get("Scores", 0.0)),
                    "rank": int(float(raw.get("rank", len(rows) + 1))),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    if sort_rows:
        rows.sort(key=lambda item: (item["rank"], item["gene"]))
    return rows


def load_seed_rank_cache(
    context_dir: Path,
    seed_gene_id: str,
    *,
    max_rows: int | None = None,
    sort_rows: bool = True,
) -> list[dict[str, Any]]:
    rank_path = rank_cache_path_for_seed(context_dir, seed_gene_id)
    try:
        return load_seed_rank_cache_path(rank_path, max_rows=max_rows, sort_rows=sort_rows)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing RWR-LOE rank cache for {seed_gene_id}: {rank_path}") from error


def write_rank_cache_from_encoding_matrix(
    *,
    encoding_matrix_path: Path,
    context_dir: Path,
    cache_context: dict[str, Any],
    shard_index: int,
    shard_count: int,
    requested_seed_gene_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    requested = {gene.upper() for gene in requested_seed_gene_ids or []}
    completed = 0
    skipped = 0
    seed_ids: list[str] = []

    with encoding_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if not header or header[0] != "INDEX" or len(header) < 2:
            raise ValueError(f"RWR encoding matrix lacks INDEX header: {encoding_matrix_path}")
        node_labels = [label.strip().upper() for label in header[1:]]
        for row in reader:
            if not row:
                continue
            seed_gene_id = row[0].strip().upper()
            if requested and seed_gene_id not in requested:
                skipped += 1
                continue
            if len(row) != len(node_labels) + 1:
                raise ValueError(
                    f"Malformed encoding row for {seed_gene_id}: expected {len(node_labels) + 1} columns, "
                    f"found {len(row)}."
                )
            scores = [float(value) for value in row[1:]]
            ranked = rank_scores_min_desc(zip(node_labels, scores), seed_gene_id=seed_gene_id)
            metadata = {
                "schema_version": RANK_CACHE_SCHEMA_VERSION,
                "seed_gene_id": seed_gene_id,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "ranked_gene_count": len(ranked),
                "encoding_matrix_path": str(encoding_matrix_path),
                "cache_context": cache_context,
                "created_at": utc_now_iso(),
            }
            write_seed_rank_cache(
                context_dir=context_dir,
                seed_gene_id=seed_gene_id,
                ranked_genes=ranked,
                metadata=metadata,
            )
            completed += 1
            seed_ids.append(seed_gene_id)

    shard_manifest = {
        "schema_version": RANK_CACHE_SCHEMA_VERSION,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "completed_seed_count": completed,
        "skipped_seed_count": skipped,
        "seed_gene_ids": seed_ids,
        "encoding_matrix_path": str(encoding_matrix_path),
        "cache_context": cache_context,
        "updated_at": utc_now_iso(),
    }
    shard_dir = context_dir / "shards" / f"shard_{shard_index:05d}"
    write_json(shard_dir / "rank_cache_manifest.json", shard_manifest)
    return shard_manifest


def _shard_genes(gene_ids: list[str], *, shard_index: int, shard_count: int) -> list[str]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1.")
    return [gene for index, gene in enumerate(sorted(gene_ids)) if index % shard_count == shard_index]


def _write_seed_file(path: Path, gene_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for gene_id in gene_ids:
            handle.write(gene_id)
            handle.write("\n")


def prewarm_rwr_loe_rank_cache(
    *,
    store_dir: Path,
    rwr_hpc_flist: Path,
    rwr_hpc_build_dir: Path,
    rank_cache_dir: Path,
    scratch_dir: Path,
    shard_index: int,
    shard_count: int,
    seed: int,
    restart: float,
    delta: float,
    reduction_method: str,
    threshold: float,
    edgelist_has_headers: bool,
    rwr_hpc_build_id: str | None = None,
    launcher_prefix: str = "",
    force: bool = False,
    max_genes: int | None = None,
) -> dict[str, Any]:
    gene_ids = load_rwr_loe_gene_universe(store_dir)
    if max_genes is not None:
        gene_ids = gene_ids[:max_genes]
    shard_gene_ids = _shard_genes(gene_ids, shard_index=shard_index, shard_count=shard_count)
    if not shard_gene_ids:
        return {"status": "empty", "shard_index": shard_index, "shard_count": shard_count, "seed_count": 0}

    build_id = rwr_hpc_build_id or str(rwr_hpc_build_dir.resolve())
    cache_context = build_rank_cache_context(
        rwr_hpc_flist=rwr_hpc_flist,
        rwr_hpc_build_id=build_id,
        restart=restart,
        delta=delta,
        reduction_method=reduction_method,
        threshold=threshold,
        edgelist_has_headers=edgelist_has_headers,
    )
    context_dir = rank_cache_context_dir(rank_cache_dir, cache_context)
    shard_dir = context_dir / "shards" / f"shard_{shard_index:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_json(context_dir / "cache_context.json", cache_context)

    missing = [gene for gene in shard_gene_ids if not rank_cache_path_for_seed(context_dir, gene).exists()]
    if not missing and not force:
        return {
            "status": "cached",
            "context_dir": str(context_dir),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "seed_count": len(shard_gene_ids),
        }

    seed_file = shard_dir / "seed_genes.tsv"
    _write_seed_file(seed_file, shard_gene_ids)
    output_dir = shard_dir / "rwr_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"rwr_loe_shard_{shard_index:05d}"
    encoding_matrix_path = output_dir / f"{run_tag}_encodings.tsv"

    if force or not encoding_matrix_path.exists():
        app_backend = RwrHpcAppBackend(build_dir=rwr_hpc_build_dir)
        executable = app_backend.require_app("rwr")
        app_args = [
            "--flist",
            str(rwr_hpc_flist),
            "--seed_file",
            str(seed_file),
            "--no_set_ids",
            "--runtag",
            run_tag,
            "--output_dir",
            str(output_dir),
            "--restart",
            str(restart),
            "--delta",
            str(delta),
            "--reduction_method",
            reduction_method,
            "--threshold",
            str(threshold),
            "--record_encodings",
        ]
        if not edgelist_has_headers:
            app_args.append("--no_edgelist_headers")
        command = [*shlex.split(launcher_prefix), str(executable), *app_args]
        scratch_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env={**os.environ, "TMPDIR": str(scratch_dir)},
        )
        write_json(
            shard_dir / "rwr_app_result.json",
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "updated_at": utc_now_iso(),
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"RWR app failed for LOE shard {shard_index} with return code {completed.returncode}. "
                f"See {shard_dir / 'rwr_app_result.json'}."
            )

    if not encoding_matrix_path.exists():
        raise FileNotFoundError(f"Expected RWR encoding matrix was not produced: {encoding_matrix_path}")

    shard_manifest = write_rank_cache_from_encoding_matrix(
        encoding_matrix_path=encoding_matrix_path,
        context_dir=context_dir,
        cache_context=cache_context,
        shard_index=shard_index,
        shard_count=shard_count,
        requested_seed_gene_ids=shard_gene_ids,
    )
    return {
        "status": "prewarmed",
        "context_dir": str(context_dir),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "seed_count": len(shard_gene_ids),
        "rank_cache_manifest": shard_manifest,
    }


def load_mentor_size_distribution(mentor_corpus_dir: Path) -> dict[str, int]:
    modules_path = mentor_corpus_dir / "modules.jsonl"
    counts: Counter[str] = Counter()
    if modules_path.exists():
        with modules_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                size_bin = row.get("size_bin")
                if size_bin in DEFAULT_MODULE_SIZES:
                    counts[size_bin] += 1
    if not counts:
        counts.update({"small": 1, "medium": 1, "large": 1})
    return {size_bin: counts.get(size_bin, 0) for size_bin in DEFAULT_MODULE_SIZES}


def _proportional_counts(total: int, weights: dict[str, int]) -> dict[str, int]:
    positive_total = sum(max(value, 0) for value in weights.values())
    if total <= 0:
        return {key: 0 for key in weights}
    if positive_total <= 0:
        base = total // len(weights)
        counts = {key: base for key in weights}
        for key in list(weights)[: total - sum(counts.values())]:
            counts[key] += 1
        return counts
    raw = {
        key: total * (max(value, 0) / positive_total)
        for key, value in weights.items()
    }
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def assign_size_bins(
    gene_ids: list[str],
    *,
    mentor_size_distribution: dict[str, int],
    seed: int,
) -> dict[str, str]:
    counts = _proportional_counts(len(gene_ids), mentor_size_distribution)
    ordered = sorted(gene_ids)
    rng = random.Random(f"{seed}|loe_size_bins")
    rng.shuffle(ordered)
    assignment: dict[str, str] = {}
    offset = 0
    for size_bin in ("small", "medium", "large"):
        for gene_id in ordered[offset : offset + counts.get(size_bin, 0)]:
            assignment[gene_id] = size_bin
        offset += counts.get(size_bin, 0)
    for gene_id in ordered:
        assignment.setdefault(gene_id, "small")
    return assignment


def assign_splits_and_difficulties(
    modules: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for module in modules:
        strata[module["size_bin"]].append(module)
    for size_bin in sorted(strata):
        rows = sorted(strata[size_bin], key=lambda row: row["module_id"])
        rng = random.Random(f"{seed}|loe_split|{size_bin}")
        rng.shuffle(rows)
        train_count, val_count, test_count = stratified_split_counts(len(rows))
        for index, row in enumerate(rows):
            row = dict(row)
            if index < train_count:
                row["split"] = "train"
            elif index < train_count + val_count:
                row["split"] = "val"
            else:
                row["split"] = "test"
            assigned.append(row)

    modules_by_id = {module["module_id"]: module for module in assigned}
    for module in modules_by_id.values():
        module["difficulty_by_task"] = {}
    for task_type in ("recovery", "refinement", "none"):
        difficulty_strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for module in modules_by_id.values():
            difficulty_strata[(module["split"], module["size_bin"])].append(module)
        for key in sorted(difficulty_strata):
            rows = sorted(difficulty_strata[key], key=lambda row: row["module_id"])
            rng = random.Random(f"{seed}|loe_difficulty|{task_type}|{key[0]}|{key[1]}")
            rng.shuffle(rows)
            for index, row in enumerate(rows):
                row["difficulty_by_task"][task_type] = DIFFICULTIES[index % len(DIFFICULTIES)]
    return sorted(modules_by_id.values(), key=lambda row: row["module_id"])


def _sample_from_candidate_entries(
    entries_by_gene: dict[str, dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    salt: str,
    gene_to_modules: dict[str, set[str]] | None = None,
    conflict_free: bool = False,
    initial_used_modules: set[str] | None = None,
) -> list[str]:
    ordered = sorted(entries_by_gene)
    if sample_size <= 0:
        return []
    if len(ordered) < sample_size:
        return []
    if not conflict_free:
        rng = random.Random(f"{seed}|loe_sample|{salt}|0")
        return sorted(rng.sample(ordered, sample_size))

    for attempt in range(32):
        rng = random.Random(f"{seed}|loe_sample|{salt}|{attempt}")
        if len(ordered) <= 2048 or attempt == 31:
            candidates = ordered[:]
            rng.shuffle(candidates)
        else:
            window_size = min(
                len(ordered),
                max(sample_size + 128, sample_size * (4 + attempt * 2), 512),
            )
            candidates = rng.sample(ordered, window_size)
        selected: list[str] = []
        used_modules = set(initial_used_modules or set())
        for gene_id in candidates:
            if conflict_free:
                memberships = gene_to_modules.get(gene_id, set()) if gene_to_modules else set()
                if memberships & used_modules:
                    continue
                used_modules.update(memberships)
            selected.append(gene_id)
            if len(selected) == sample_size:
                return sorted(selected)
    return []


def _rank_entries_for_percentile_band(
    ranked_genes: list[dict[str, Any]],
    *,
    target_gene_ids: set[str],
    candidate_gene_ids: set[str],
    percentile_range: tuple[float, float],
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in ranked_genes
        if row["gene"] in candidate_gene_ids and row["gene"] not in target_gene_ids
    ]
    total = len(candidates)
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
    for ordinal, row in enumerate(candidates[start_rank : end_rank + 1], start=start_rank):
        entries.append(
            {
                "gene_id": row["gene"],
                "rank": row["rank"],
                "score": row["score"],
                "percentile": 1.0 if total == 1 else ordinal / (total - 1),
            }
        )
    return entries


def select_rank_band_negative_genes(
    *,
    ranked_genes: list[dict[str, Any]],
    target_gene_ids: Iterable[str],
    candidate_gene_ids: Iterable[str],
    sample_size: int,
    difficulty: str,
    seed: int,
    salt: str,
    gene_to_modules: dict[str, set[str]] | None = None,
    conflict_free: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    if sample_size <= 0:
        return [], {"selection_mode": "empty", "candidate_count": 0}
    target_set = {gene.upper() for gene in target_gene_ids}
    candidate_set = {gene.upper() for gene in candidate_gene_ids}
    preferred_band = RANK_PERCENTILE_BANDS[difficulty]
    fallback_band = RANK_FALLBACK_PERCENTILE_BANDS[difficulty]
    all_candidates = [
        row for row in ranked_genes if row["gene"] in candidate_set and row["gene"] not in target_set
    ]
    preferred_entries = _rank_entries_for_percentile_band(
        ranked_genes,
        target_gene_ids=target_set,
        candidate_gene_ids=candidate_set,
        percentile_range=preferred_band,
    )
    fallback_entries = _rank_entries_for_percentile_band(
        ranked_genes,
        target_gene_ids=target_set,
        candidate_gene_ids=candidate_set,
        percentile_range=fallback_band,
    )
    preferred_by_gene = {entry["gene_id"]: entry for entry in preferred_entries}
    fallback_by_gene = {entry["gene_id"]: entry for entry in fallback_entries}
    selected = _sample_from_candidate_entries(
        preferred_by_gene,
        sample_size=sample_size,
        seed=seed,
        salt=f"{salt}|preferred",
        gene_to_modules=gene_to_modules,
        conflict_free=conflict_free,
    )
    selection_mode = "preferred_rank_band"
    if len(selected) < sample_size:
        selected = _sample_from_candidate_entries(
            fallback_by_gene,
            sample_size=sample_size,
            seed=seed,
            salt=f"{salt}|fallback",
            gene_to_modules=gene_to_modules,
            conflict_free=conflict_free,
        )
        selection_mode = "fallback_rank_band"
    if len(selected) < sample_size:
        raise ValueError(
            f"Not enough {difficulty} RWR-LOE rank-band candidates for {salt}: "
            f"needed {sample_size}, preferred={len(preferred_entries)}, "
            f"fallback={len(fallback_entries)}, total={len(all_candidates)}."
        )
    lookup = {**fallback_by_gene, **preferred_by_gene}
    return selected, {
        "selection_mode": selection_mode,
        "difficulty": difficulty,
        "rank_metric": "rwr_loe_rank_percentile",
        "rank_definition": "per-seed RWR encoding rank after excluding target module genes",
        "preferred_percentile_range": list(preferred_band),
        "fallback_percentile_range": list(fallback_band),
        "candidate_count": len(all_candidates),
        "preferred_candidate_count": len(preferred_entries),
        "fallback_candidate_count": len(fallback_entries),
        "selected_ranks": {gene_id: lookup[gene_id]["rank"] for gene_id in selected},
        "selected_scores": {gene_id: lookup[gene_id]["score"] for gene_id in selected},
        "selected_percentiles": {gene_id: lookup[gene_id]["percentile"] for gene_id in selected},
    }


def build_gene_to_modules(modules: list[dict[str, Any]]) -> dict[str, set[str]]:
    gene_to_modules: dict[str, set[str]] = defaultdict(set)
    for module in modules:
        for gene_id in module["gene_ids"]:
            gene_to_modules[gene_id].add(module["module_id"])
    return gene_to_modules


def _build_loe_module_for_seed(
    *,
    index: int,
    seed_gene_id: str,
    context_dir: Path,
    module_selection_method: str = DEFAULT_MODULE_SELECTION_METHOD,
    size_bin: str | None = None,
    module_sizes: dict[str, int] | None = None,
    min_elbow_module_size: int = MIN_ELBOW_MODULE_SIZE,
) -> tuple[dict[str, Any] | None, str | None]:
    module_sizes = dict(module_sizes or DEFAULT_MODULE_SIZES)
    if module_selection_method == "topk":
        if size_bin is None:
            raise ValueError("size_bin is required for top-k LOE module selection.")
        target_size = int(module_sizes[size_bin])
        try:
            target_ranked = load_seed_rank_cache(
                context_dir,
                seed_gene_id,
                max_rows=max(0, target_size - 1),
                sort_rows=False,
            )
        except FileNotFoundError:
            return None, "missing_rank_cache"
        if len(target_ranked) < target_size - 1:
            return None, "insufficient_ranked_genes"
        module_selection = {
            "method": "topk",
            "target_module_size": target_size,
            "retained_ranked_gene_count": len(target_ranked),
        }
    elif module_selection_method == "elbow":
        try:
            ranked_genes = load_seed_rank_cache(context_dir, seed_gene_id, sort_rows=False)
        except FileNotFoundError:
            return None, "missing_rank_cache"
        target_ranked, module_selection = select_elbow_module_ranked_genes(ranked_genes)
        if 1 + len(target_ranked) < min_elbow_module_size:
            return None, "below_min_elbow_module_size"
        size_bin = loe_size_bin(1 + len(target_ranked))
        target_size = 1 + len(target_ranked)
    else:
        raise ValueError(f"Unknown LOE module selection method: {module_selection_method}")

    gene_members = [seed_gene_id.upper(), *[row["gene"] for row in target_ranked]]
    module = {
        "module_id": f"rwr_loe_module_{index:06d}",
        "source": SOURCE_NAME,
        "seed_gene_id": seed_gene_id.upper(),
        "gene_ids": sorted(set(gene_members)),
        "gene_symbols": sorted(set(gene_members)),
        "size": len(set(gene_members)),
        "size_bin": size_bin,
        "target_module_size": target_size,
        "module_selection": module_selection,
        "rank_cache_path": str(rank_cache_path_for_seed(context_dir, seed_gene_id)),
    }
    return module, None


def _build_loe_module_checkpoint_record(payload: dict[str, Any]) -> dict[str, Any]:
    module, skip_reason = _build_loe_module_for_seed(
        index=int(payload["index"]),
        seed_gene_id=str(payload["seed_gene_id"]),
        context_dir=Path(payload["context_dir"]),
        module_selection_method=str(payload["module_selection_method"]),
        size_bin=payload.get("size_bin"),
        module_sizes=payload.get("module_sizes"),
        min_elbow_module_size=int(payload.get("min_elbow_module_size", MIN_ELBOW_MODULE_SIZE)),
    )
    record = {
        "schema_version": MATERIALIZE_CHECKPOINT_SCHEMA_VERSION,
        "seed_gene_id": str(payload["seed_gene_id"]).upper(),
        "module_id": f"rwr_loe_module_{int(payload['index']):06d}",
        "status": "skipped" if module is None else "module",
        "updated_at": utc_now_iso(),
    }
    if module is None:
        record["skip_reason"] = skip_reason or "unknown"
    else:
        record["module"] = module
    return record


def _module_assignment_work_items(
    *,
    gene_ids: list[str],
    context_dir: Path,
    module_selection_method: str,
    size_bin_by_gene: dict[str, str] | None,
    module_sizes: dict[str, int],
    min_elbow_module_size: int,
    completed_gene_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    completed_gene_ids = completed_gene_ids or set()
    items = []
    for index, seed_gene_id in enumerate(sorted(gene_ids), start=1):
        seed_key = seed_gene_id.upper()
        if seed_key in completed_gene_ids:
            continue
        items.append(
            {
                "index": index,
                "seed_gene_id": seed_key,
                "context_dir": str(context_dir),
                "module_selection_method": module_selection_method,
                "size_bin": size_bin_by_gene.get(seed_gene_id) if size_bin_by_gene else None,
                "module_sizes": module_sizes,
                "min_elbow_module_size": min_elbow_module_size,
            }
        )
    return items


def _apply_module_assignment_record(
    record: dict[str, Any],
    *,
    modules: list[dict[str, Any]],
    stats: BuildStats,
) -> None:
    if record.get("status") == "module":
        module = record.get("module")
        if isinstance(module, dict):
            modules.append(module)
    elif record.get("status") == "skipped":
        stats.skipped_modules[str(record.get("skip_reason", "unknown"))] += 1


def build_loe_modules(
    *,
    gene_ids: list[str],
    context_dir: Path,
    stats: BuildStats,
    module_selection_method: str = DEFAULT_MODULE_SELECTION_METHOD,
    size_bin_by_gene: dict[str, str] | None = None,
    module_sizes: dict[str, int] | None = None,
    min_elbow_module_size: int = MIN_ELBOW_MODULE_SIZE,
) -> list[dict[str, Any]]:
    module_sizes = dict(module_sizes or DEFAULT_MODULE_SIZES)
    modules = []
    for item in _module_assignment_work_items(
        gene_ids=gene_ids,
        context_dir=context_dir,
        module_selection_method=module_selection_method,
        size_bin_by_gene=size_bin_by_gene,
        module_sizes=module_sizes,
        min_elbow_module_size=min_elbow_module_size,
    ):
        record = _build_loe_module_checkpoint_record(item)
        _apply_module_assignment_record(record, modules=modules, stats=stats)
    return modules


def build_loe_modules_checkpointed(
    *,
    gene_ids: list[str],
    context_dir: Path,
    stats: BuildStats,
    module_selection_method: str = DEFAULT_MODULE_SELECTION_METHOD,
    size_bin_by_gene: dict[str, str] | None = None,
    module_sizes: dict[str, int] | None = None,
    min_elbow_module_size: int = MIN_ELBOW_MODULE_SIZE,
    checkpoint_path: Path,
    tracker: ProgressTracker,
    workers: int = DEFAULT_MATERIALIZE_WORKERS,
) -> list[dict[str, Any]]:
    module_sizes = dict(module_sizes or DEFAULT_MODULE_SIZES)
    records = _read_checkpoint_records(checkpoint_path, key_field="seed_gene_id")
    completed_gene_ids = {str(record["seed_gene_id"]).upper() for record in records}
    modules: list[dict[str, Any]] = []
    for record in records:
        _apply_module_assignment_record(record, modules=modules, stats=stats)

    if records:
        tracker.update(
            completed=len(completed_gene_ids),
            total=len(gene_ids),
            unit="genes",
            message=f"Loaded {len(completed_gene_ids)} checkpointed LOE module assignments.",
            metrics={
                "checkpoint_path": str(checkpoint_path),
                "module_count_so_far": len(modules),
                "skipped_modules": dict(stats.skipped_modules),
            },
        )

    work_items = _module_assignment_work_items(
        gene_ids=gene_ids,
        context_dir=context_dir,
        module_selection_method=module_selection_method,
        size_bin_by_gene=size_bin_by_gene,
        module_sizes=module_sizes,
        min_elbow_module_size=min_elbow_module_size,
        completed_gene_ids=completed_gene_ids,
    )
    worker_count = max(1, int(workers))
    if worker_count > 1 and len(work_items) > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            records_iter = executor.map(_build_loe_module_checkpoint_record, work_items, chunksize=1)
            for offset, record in enumerate(records_iter, start=1):
                _append_checkpoint_record(checkpoint_path, record)
                completed_gene_ids.add(str(record["seed_gene_id"]).upper())
                _apply_module_assignment_record(record, modules=modules, stats=stats)
                if offset % CHECKPOINT_FLUSH_INTERVAL == 0 or offset == len(work_items):
                    tracker.update(
                        completed=len(completed_gene_ids),
                        total=len(gene_ids),
                        unit="genes",
                        message=(
                            f"Assigned {len(completed_gene_ids)}/{len(gene_ids)} LOE module seeds "
                            f"({len(modules)} modules, workers={worker_count})."
                        ),
                        metrics={
                            "checkpoint_path": str(checkpoint_path),
                            "module_count_so_far": len(modules),
                            "skipped_modules": dict(stats.skipped_modules),
                            "materialize_workers": worker_count,
                        },
                    )
    else:
        for offset, item in enumerate(work_items, start=1):
            record = _build_loe_module_checkpoint_record(item)
            _append_checkpoint_record(checkpoint_path, record)
            completed_gene_ids.add(str(record["seed_gene_id"]).upper())
            _apply_module_assignment_record(record, modules=modules, stats=stats)
            if offset % CHECKPOINT_FLUSH_INTERVAL == 0 or offset == len(work_items):
                tracker.update(
                    completed=len(completed_gene_ids),
                    total=len(gene_ids),
                    unit="genes",
                    message=(
                        f"Assigned {len(completed_gene_ids)}/{len(gene_ids)} LOE module seeds "
                        f"({len(modules)} modules)."
                    ),
                    metrics={
                        "checkpoint_path": str(checkpoint_path),
                        "module_count_so_far": len(modules),
                        "skipped_modules": dict(stats.skipped_modules),
                        "materialize_workers": worker_count,
                    },
                )
    return sorted(modules, key=lambda row: row["module_id"])


def build_task_prototypes(
    *,
    modules: list[dict[str, Any]],
    candidate_gene_universe: set[str],
    seed: int,
    stats: BuildStats,
) -> list[dict[str, Any]]:
    gene_to_modules = build_gene_to_modules(modules)
    prototypes: list[dict[str, Any]] = []
    for module in modules:
        module_prototypes, skipped_refinement, skipped_none = build_task_prototypes_for_module(
            module=module,
            candidate_gene_universe=candidate_gene_universe,
            gene_to_modules=gene_to_modules,
            seed=seed,
        )
        prototypes.extend(module_prototypes)
        stats.skipped_refinement.update(skipped_refinement)
        stats.skipped_none.update(skipped_none)
    return sorted(prototypes, key=lambda row: row["prototype_id"])


def build_task_prototypes_for_module(
    *,
    module: dict[str, Any],
    candidate_gene_universe: set[str],
    gene_to_modules: dict[str, set[str]],
    seed: int,
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    module_id = module["module_id"]
    target_gene_ids = list(module["gene_ids"])
    ranked_genes = load_seed_rank_cache_path(Path(module["rank_cache_path"]), sort_rows=False)
    prototypes: list[dict[str, Any]] = []
    skipped_refinement: Counter = Counter()
    skipped_none: Counter = Counter()

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
        noise_gene_ids, metadata = select_rank_band_negative_genes(
            ranked_genes=ranked_genes,
            target_gene_ids=target_gene_ids,
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
                    "rwr_loe_negative_sampling": metadata,
                },
            }
        )
    except ValueError:
        skipped_refinement[(module["split"], module["size_bin"], refinement_difficulty)] += 1

    none_difficulty = module["difficulty_by_task"]["none"]
    try:
        none_gene_ids, metadata = select_rank_band_negative_genes(
            ranked_genes=ranked_genes,
            target_gene_ids=target_gene_ids,
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
                "sampling_metadata": {"rwr_loe_negative_sampling": metadata},
            }
        )
    except ValueError:
        skipped_none[(module["split"], module["size_bin"], none_difficulty)] += 1

    return prototypes, skipped_refinement, skipped_none


_PROTOTYPE_WORKER_CONTEXT: dict[str, Any] = {}


def _init_task_prototype_worker(
    candidate_gene_universe: set[str],
    gene_to_modules: dict[str, set[str]],
    seed: int,
) -> None:
    _PROTOTYPE_WORKER_CONTEXT.clear()
    _PROTOTYPE_WORKER_CONTEXT.update(
        {
            "candidate_gene_universe": candidate_gene_universe,
            "gene_to_modules": gene_to_modules,
            "seed": seed,
        }
    )


def _build_task_prototype_checkpoint_record_worker(module: dict[str, Any]) -> dict[str, Any]:
    module_prototypes, skipped_refinement, skipped_none = build_task_prototypes_for_module(
        module=module,
        candidate_gene_universe=_PROTOTYPE_WORKER_CONTEXT["candidate_gene_universe"],
        gene_to_modules=_PROTOTYPE_WORKER_CONTEXT["gene_to_modules"],
        seed=int(_PROTOTYPE_WORKER_CONTEXT["seed"]),
    )
    return {
        "schema_version": MATERIALIZE_CHECKPOINT_SCHEMA_VERSION,
        "module_id": module["module_id"],
        "prototype_count": len(module_prototypes),
        "prototypes": module_prototypes,
        "skipped_refinement": _jsonable_counter(skipped_refinement),
        "skipped_none": _jsonable_counter(skipped_none),
        "updated_at": utc_now_iso(),
    }


def build_task_prototypes_checkpointed(
    *,
    modules: list[dict[str, Any]],
    candidate_gene_universe: set[str],
    seed: int,
    stats: BuildStats,
    checkpoint_path: Path,
    tracker: ProgressTracker,
    workers: int = DEFAULT_MATERIALIZE_WORKERS,
) -> list[dict[str, Any]]:
    gene_to_modules = build_gene_to_modules(modules)
    records = _read_checkpoint_records(checkpoint_path, key_field="module_id")
    completed_module_ids = {record["module_id"] for record in records}
    prototypes: list[dict[str, Any]] = []
    for record in records:
        prototypes.extend(record.get("prototypes", []))
        stats.skipped_refinement.update(_counter_from_jsonable(record.get("skipped_refinement", [])))
        stats.skipped_none.update(_counter_from_jsonable(record.get("skipped_none", [])))

    if records:
        tracker.update(
            completed=len(completed_module_ids),
            total=len(modules),
            unit="modules",
            message=f"Loaded {len(completed_module_ids)} checkpointed prototype modules.",
            metrics={"checkpoint_path": str(checkpoint_path)},
        )

    remaining = [module for module in modules if module["module_id"] not in completed_module_ids]
    worker_count = max(1, int(workers))

    def consume_record(offset: int, record: dict[str, Any]) -> None:
        _append_checkpoint_record(checkpoint_path, record)
        completed_module_ids.add(str(record["module_id"]))
        prototypes.extend(record.get("prototypes", []))
        stats.skipped_refinement.update(_counter_from_jsonable(record.get("skipped_refinement", [])))
        stats.skipped_none.update(_counter_from_jsonable(record.get("skipped_none", [])))
        if offset % CHECKPOINT_FLUSH_INTERVAL == 0 or offset == len(remaining):
            tracker.update(
                completed=len(completed_module_ids),
                total=len(modules),
                unit="modules",
                message=(
                    f"Built prototypes for {len(completed_module_ids)}/{len(modules)} modules "
                    f"({len(prototypes)} prototypes, workers={worker_count})."
                ),
                metrics={
                    "checkpoint_path": str(checkpoint_path),
                    "prototype_count_so_far": len(prototypes),
                    "skipped_refinement_count": sum(stats.skipped_refinement.values()),
                    "skipped_none_count": sum(stats.skipped_none.values()),
                    "materialize_workers": worker_count,
                },
            )

    if worker_count > 1 and len(remaining) > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_task_prototype_worker,
            initargs=(candidate_gene_universe, gene_to_modules, seed),
        ) as executor:
            records_iter = executor.map(
                _build_task_prototype_checkpoint_record_worker,
                remaining,
                chunksize=1,
            )
            for offset, record in enumerate(records_iter, start=1):
                consume_record(offset, record)
    else:
        _init_task_prototype_worker(candidate_gene_universe, gene_to_modules, seed)
        for offset, module in enumerate(remaining, start=1):
            record = _build_task_prototype_checkpoint_record_worker(module)
            consume_record(offset, record)
    return sorted(prototypes, key=lambda row: row["prototype_id"])


def balance_prototypes(
    prototypes: list[dict[str, Any]],
    *,
    seed: int,
    stats: BuildStats,
) -> list[dict[str, Any]]:
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
                        salt=f"loe_balance|{key[0]}|{key[1]}|{task_type}",
                    )
                )
                rows = [row for row in rows if row["prototype_id"] in selected_ids]
            balanced.extend(rows)
    stats.balance = balance_summary
    return sorted(balanced, key=lambda row: row["prototype_id"])


def materialize_tasks(
    *,
    modules: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    store_dir: Path,
    rwr_hpc_flist: Path,
    rank_cache_context_dir_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    modules_by_id = {module["module_id"]: module for module in modules}
    tasks = []
    for prototype in prototypes:
        tasks.extend(
            materialize_tasks_for_prototype(
                prototype=prototype,
                modules_by_id=modules_by_id,
                store_dir=store_dir,
                rwr_hpc_flist=rwr_hpc_flist,
                rank_cache_context_dir_path=rank_cache_context_dir_path,
                seed=seed,
            )
        )
    return sorted(tasks, key=lambda row: row["task_id"])


def materialize_tasks_for_prototype(
    *,
    prototype: dict[str, Any],
    modules_by_id: dict[str, dict[str, Any]],
    store_dir: Path,
    rwr_hpc_flist: Path,
    rank_cache_context_dir_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    target_module = modules_by_id.get(prototype.get("source_module_id"))
    seed_gene_ids = list(prototype["input_gene_ids"])
    seed_gene_symbols = list(seed_gene_ids)
    tasks = []
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
            "store_dir": str(store_dir),
            "rwr_hpc_flist": str(rwr_hpc_flist),
            "rank_cache_context_dir": str(rank_cache_context_dir_path),
            "generation_seed": seed,
            "sampling_metadata": prototype.get("sampling_metadata", {}),
        }
        if target_module is not None:
            provenance["seed_gene_id"] = target_module["seed_gene_id"]
            provenance["rank_cache_path"] = target_module["rank_cache_path"]
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
                            "resolved_via": "rwr_loe_rank_cache",
                        }
                        for gene_id in seed_gene_ids
                    ],
                },
                "provenance": provenance,
            }
        )
    return tasks


def materialize_tasks_checkpointed(
    *,
    modules: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    store_dir: Path,
    rwr_hpc_flist: Path,
    rank_cache_context_dir_path: Path,
    seed: int,
    checkpoint_path: Path,
    tracker: ProgressTracker,
) -> list[dict[str, Any]]:
    modules_by_id = {module["module_id"]: module for module in modules}
    records = _read_checkpoint_records(checkpoint_path, key_field="prototype_id")
    completed_prototype_ids = {record["prototype_id"] for record in records}
    tasks: list[dict[str, Any]] = []
    for record in records:
        tasks.extend(record.get("tasks", []))
    if records:
        tracker.update(
            completed=len(completed_prototype_ids) * len(EVIDENCE_MODES),
            total=len(prototypes) * len(EVIDENCE_MODES),
            unit="tasks",
            message=f"Loaded {len(completed_prototype_ids)} checkpointed task prototypes.",
            metrics={"checkpoint_path": str(checkpoint_path)},
        )

    remaining = [prototype for prototype in prototypes if prototype["prototype_id"] not in completed_prototype_ids]
    for offset, prototype in enumerate(remaining, start=1):
        prototype_tasks = materialize_tasks_for_prototype(
            prototype=prototype,
            modules_by_id=modules_by_id,
            store_dir=store_dir,
            rwr_hpc_flist=rwr_hpc_flist,
            rank_cache_context_dir_path=rank_cache_context_dir_path,
            seed=seed,
        )
        record = {
            "schema_version": MATERIALIZE_CHECKPOINT_SCHEMA_VERSION,
            "prototype_id": prototype["prototype_id"],
            "task_count": len(prototype_tasks),
            "tasks": prototype_tasks,
            "updated_at": utc_now_iso(),
        }
        _append_checkpoint_record(checkpoint_path, record)
        completed_prototype_ids.add(prototype["prototype_id"])
        tasks.extend(prototype_tasks)
        if offset % CHECKPOINT_FLUSH_INTERVAL == 0 or offset == len(remaining):
            tracker.update(
                completed=len(completed_prototype_ids) * len(EVIDENCE_MODES),
                total=len(prototypes) * len(EVIDENCE_MODES),
                unit="tasks",
                message=(
                    f"Materialized {len(completed_prototype_ids)}/{len(prototypes)} "
                    f"prototype task groups."
                ),
                metrics={"checkpoint_path": str(checkpoint_path), "task_count_so_far": len(tasks)},
            )
    return sorted(tasks, key=lambda row: row["task_id"])


def build_module_jsonl_rows(modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for module in modules:
        rows.append(
            {
                "module_id": module["module_id"],
                "split": module["split"],
                "source": SOURCE_NAME,
                "seed_gene_id": module["seed_gene_id"],
                "size": module["size"],
                "size_bin": module["size_bin"],
                "target_module_size": module["target_module_size"],
                "module_selection": module.get("module_selection", {}),
                "gene_ids": module["gene_ids"],
                "gene_symbols": module["gene_symbols"],
                "rank_cache_path": module["rank_cache_path"],
                "difficulty_by_task": module.get("difficulty_by_task", {}),
            }
        )
    return sorted(rows, key=lambda row: row["module_id"])


def build_split_report(
    *,
    modules: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
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
        "module_count_by_split": dict(module_counts),
        "module_count_by_split_and_size_bin": {split: dict(module_size_counts[split]) for split in SPLITS},
        "prototype_count_by_type": dict(Counter(row["task_type"] for row in prototypes)),
        "task_count_by_split_and_type": {split: dict(task_counts[split]) for split in SPLITS},
        "task_count_by_split_size_bin_and_type": {split: dict(task_size_counts[split]) for split in SPLITS},
        "skipped_modules": dict(stats.skipped_modules),
        "skipped_refinement": {str(key): value for key, value in stats.skipped_refinement.items()},
        "skipped_none": {str(key): value for key, value in stats.skipped_none.items()},
        "rank_cache": stats.rank_cache,
        "balance": stats.balance,
    }


def build_manifest(
    *,
    store_dir: Path,
    rwr_hpc_flist: Path,
    rank_cache_dir: Path,
    rank_cache_context_path: Path,
    out_dir: Path,
    mentor_size_distribution: dict[str, int],
    module_sizes: dict[str, int],
    module_selection_method: str,
    seed: int,
    modules: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "store_dir": str(store_dir),
        "rwr_hpc_flist": str(rwr_hpc_flist),
        "rank_cache_dir": str(rank_cache_dir),
        "rank_cache_context_dir": str(rank_cache_context_path),
        "out_dir": str(out_dir),
        "seed": seed,
        "mentor_size_distribution": mentor_size_distribution,
        "module_selection": {
            "method": module_selection_method,
            "topk_module_sizes": module_sizes if module_selection_method == "topk" else None,
            "elbow": (
                {
                    "algorithm": "RWR++ geometric elbow point over per-seed rank/score curve",
                    "retention_rule": "rank < elbow_rank_cutoff",
                    "min_module_size": MIN_ELBOW_MODULE_SIZE,
                }
                if module_selection_method == "elbow"
                else None
            ),
        },
        "negative_sampling": {
            "method": "rwr_loe_rank_percentile",
            "rank_definition": "per-seed RWR-LOE rank after excluding target module genes",
            "preferred_percentile_bands": RANK_PERCENTILE_BANDS,
            "fallback_percentile_bands": RANK_FALLBACK_PERCENTILE_BANDS,
            "difficulty_semantics": {
                "easy": "75-100th percentile candidates after nearest-to-farthest RWR-LOE sorting",
                "medium": "25-50th percentile candidates after nearest-to-farthest RWR-LOE sorting",
                "hard": "0-25th percentile candidates after nearest-to-farthest RWR-LOE sorting",
            },
        },
        "evidence_modes": list(EVIDENCE_MODES),
        "module_count": len(modules),
        "prototype_count": len(prototypes),
        "task_count": len(tasks),
        "task_count_by_split": dict(Counter(task["split"] for task in tasks)),
        "task_count_by_type": dict(Counter(task["task_type"] for task in tasks)),
    }


def build_rwr_loe_corpus(
    *,
    store_dir: Path,
    rwr_hpc_flist: Path,
    rank_cache_dir: Path,
    out_dir: Path,
    mentor_corpus_dir: Path = DEFAULT_MENTOR_CORPUS_DIR,
    seed: int = 42,
    module_sizes: dict[str, int] | None = None,
    module_selection_method: str = DEFAULT_MODULE_SELECTION_METHOD,
    progress_path: Path | None = None,
    max_genes: int | None = None,
    rank_cache_context: dict[str, Any] | None = None,
    rank_cache_context_dir_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    materialize_workers: int = DEFAULT_MATERIALIZE_WORKERS,
) -> dict[str, Any]:
    module_sizes = dict(module_sizes or DEFAULT_MODULE_SIZES)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_path or out_dir / "progress.json"
    tracker = ProgressTracker(progress_path)
    stats = BuildStats()

    if rank_cache_context_dir_path is not None and rank_cache_context is None:
        rank_cache_context = load_rank_cache_context_from_dir(rank_cache_context_dir_path)
    if rank_cache_context is None:
        rank_cache_context = build_rank_cache_context(
            rwr_hpc_flist=rwr_hpc_flist,
            rwr_hpc_build_id=str(DEFAULT_RWR_HPC_BUILD_DIR.resolve()),
            restart=0.7,
            delta=0.5,
            reduction_method="geometric",
            threshold=1e-10,
            edgelist_has_headers=True,
        )
    context_dir = rank_cache_context_dir_path or rank_cache_context_dir(rank_cache_dir, rank_cache_context)
    rank_dir = context_dir / "ranks"
    if not rank_dir.exists() or next(rank_dir.glob("*.ranks.tsv.gz"), None) is None:
        raise FileNotFoundError(
            f"No RWR-LOE rank cache files found under {rank_dir}. "
            "Run LOE_MODE=prewarm first, or pass --rank-cache-context-dir/LOE_RANK_CACHE_CONTEXT_DIR "
            "for the existing prewarmed context."
        )
    checkpoint_context = materialize_checkpoint_context(
        store_dir=store_dir,
        rwr_hpc_flist=rwr_hpc_flist,
        rank_cache_context=rank_cache_context,
        rank_cache_context_dir_path=context_dir,
        mentor_corpus_dir=mentor_corpus_dir,
        seed=seed,
        max_genes=max_genes,
        module_sizes=module_sizes,
        module_selection_method=module_selection_method,
    )
    checkpoint_dir = checkpoint_dir or materialize_checkpoint_dir(out_dir, checkpoint_context)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_context_path = checkpoint_dir / "context.json"
    if checkpoint_context_path.exists():
        existing_checkpoint_context = json.loads(checkpoint_context_path.read_text(encoding="utf-8"))
        if existing_checkpoint_context != checkpoint_context:
            raise ValueError(
                f"Checkpoint directory context mismatch for {checkpoint_dir}. "
                "Use a different --checkpoint-dir or remove the stale checkpoint directory."
            )
    else:
        write_json(checkpoint_context_path, checkpoint_context)
    module_checkpoint_path = checkpoint_dir / "modules.assigned.jsonl"
    module_assignment_checkpoint_path = checkpoint_dir / "modules.assigned.by_gene.jsonl"
    module_checkpoint_meta_path = checkpoint_dir / "modules.assigned.meta.json"
    raw_prototype_checkpoint_path = checkpoint_dir / "prototypes.raw.by_module.jsonl"
    raw_prototype_checkpoint_meta_path = checkpoint_dir / "prototypes.raw.meta.json"
    balanced_prototype_checkpoint_path = checkpoint_dir / "prototypes.balanced.jsonl"
    balanced_prototype_checkpoint_meta_path = checkpoint_dir / "prototypes.balanced.meta.json"
    task_checkpoint_path = checkpoint_dir / "tasks.by_prototype.jsonl"
    task_checkpoint_meta_path = checkpoint_dir / "tasks.meta.json"

    stats.rank_cache = {
        "context_dir": str(context_dir),
        "context_hash": stable_json_hash(rank_cache_context),
    }
    tracker.set_context(
        {
            "store_dir": str(store_dir),
            "rwr_hpc_flist": str(rwr_hpc_flist),
            "rank_cache_dir": str(rank_cache_dir),
            "rank_cache_context_dir": str(context_dir),
            "out_dir": str(out_dir),
            "seed": seed,
            "max_genes": max_genes,
            "module_selection_method": module_selection_method,
            "checkpoint_dir": str(checkpoint_dir),
            "materialize_workers": max(1, int(materialize_workers)),
        }
    )

    try:
        tracker.start("load_genes", unit="genes")
        gene_ids = load_rwr_loe_gene_universe(store_dir)
        if max_genes is not None:
            gene_ids = gene_ids[:max_genes]
        tracker.update(completed=len(gene_ids), total=len(gene_ids), unit="genes")

        tracker.start("assign_modules", total=len(gene_ids), unit="genes")
        mentor_size_distribution = load_mentor_size_distribution(mentor_corpus_dir)
        module_checkpoint_meta = {}
        if module_checkpoint_meta_path.exists():
            module_checkpoint_meta = json.loads(module_checkpoint_meta_path.read_text(encoding="utf-8"))
        if (
            module_checkpoint_path.exists()
            and module_checkpoint_meta.get("status") == "completed"
            and module_checkpoint_meta.get("gene_count") == len(gene_ids)
        ):
            modules = read_jsonl(module_checkpoint_path)
            stats.skipped_modules.update(_counter_from_jsonable(module_checkpoint_meta.get("skipped_modules", [])))
            tracker.update(
                completed=len(modules),
                total=len(gene_ids),
                unit="modules",
                message=f"Loaded {len(modules)} checkpointed LOE modules.",
                metrics={
                    "module_count": len(modules),
                    "skipped_modules": dict(stats.skipped_modules),
                    "mentor_size_distribution": mentor_size_distribution,
                    "module_selection_method": module_selection_method,
                    "checkpoint_path": str(module_checkpoint_path),
                    "assignment_checkpoint_path": str(module_assignment_checkpoint_path),
                    "materialize_workers": max(1, int(materialize_workers)),
                },
            )
        else:
            size_bin_by_gene = (
                assign_size_bins(gene_ids, mentor_size_distribution=mentor_size_distribution, seed=seed)
                if module_selection_method == "topk"
                else None
            )
            modules = build_loe_modules_checkpointed(
                gene_ids=gene_ids,
                context_dir=context_dir,
                stats=stats,
                module_selection_method=module_selection_method,
                size_bin_by_gene=size_bin_by_gene,
                module_sizes=module_sizes,
                checkpoint_path=module_assignment_checkpoint_path,
                tracker=tracker,
                workers=materialize_workers,
            )
            modules = assign_splits_and_difficulties(modules, seed=seed)
            if gene_ids and not modules:
                raise RuntimeError(
                    "RWR-LOE materialization produced zero modules. "
                    f"Skipped modules: {dict(stats.skipped_modules)}. "
                    f"Rank cache context: {context_dir}"
                )
            write_jsonl_atomic(module_checkpoint_path, modules)
            write_json(
                module_checkpoint_meta_path,
                {
                    "schema_version": MATERIALIZE_CHECKPOINT_SCHEMA_VERSION,
                    "status": "completed",
                    "gene_count": len(gene_ids),
                    "module_count": len(modules),
                    "skipped_modules": _jsonable_counter(stats.skipped_modules),
                    "updated_at": utc_now_iso(),
                },
            )
            tracker.update(
                completed=len(modules),
                total=len(gene_ids),
                unit="modules",
                metrics={
                    "module_count": len(modules),
                    "skipped_modules": dict(stats.skipped_modules),
                    "mentor_size_distribution": mentor_size_distribution,
                    "module_selection_method": module_selection_method,
                    "checkpoint_path": str(module_checkpoint_path),
                    "assignment_checkpoint_path": str(module_assignment_checkpoint_path),
                    "materialize_workers": max(1, int(materialize_workers)),
                },
            )

        tracker.start("load_rank_cache", total=len(modules), unit="modules")
        candidate_gene_universe = {gene for module in modules for gene in module["gene_ids"]}
        candidate_gene_universe.update(gene_ids)
        tracker.update(
            completed=len(modules),
            total=len(modules),
            unit="modules",
            metrics={"candidate_gene_count": len(candidate_gene_universe)},
        )

        tracker.start("build_prototypes", total=len(modules), unit="modules")
        prototypes = build_task_prototypes_checkpointed(
            modules=modules,
            candidate_gene_universe=candidate_gene_universe,
            seed=seed,
            stats=stats,
            checkpoint_path=raw_prototype_checkpoint_path,
            tracker=tracker,
            workers=materialize_workers,
        )
        write_json(
            raw_prototype_checkpoint_meta_path,
            {
                "schema_version": MATERIALIZE_CHECKPOINT_SCHEMA_VERSION,
                "status": "completed",
                "module_count": len(modules),
                "prototype_count": len(prototypes),
                "skipped_refinement": _jsonable_counter(stats.skipped_refinement),
                "skipped_none": _jsonable_counter(stats.skipped_none),
                "updated_at": utc_now_iso(),
            },
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
        balanced_checkpoint_meta = {}
        if balanced_prototype_checkpoint_meta_path.exists():
            balanced_checkpoint_meta = json.loads(balanced_prototype_checkpoint_meta_path.read_text(encoding="utf-8"))
        if (
            balanced_prototype_checkpoint_path.exists()
            and balanced_checkpoint_meta.get("status") == "completed"
            and balanced_checkpoint_meta.get("raw_prototype_count") == len(prototypes)
        ):
            prototypes = read_jsonl(balanced_prototype_checkpoint_path)
            stats.balance = balanced_checkpoint_meta.get("balance", {})
            tracker.update(
                completed=len(prototypes),
                total=len(prototypes),
                unit="prototypes",
                message=f"Loaded {len(prototypes)} checkpointed balanced prototypes.",
                metrics={"prototype_count_after_balance": len(prototypes), "checkpoint_path": str(balanced_prototype_checkpoint_path)},
            )
        else:
            raw_prototype_count = len(prototypes)
            prototypes = balance_prototypes(prototypes, seed=seed, stats=stats)
            write_jsonl_atomic(balanced_prototype_checkpoint_path, prototypes)
            write_json(
                balanced_prototype_checkpoint_meta_path,
                {
                    "schema_version": MATERIALIZE_CHECKPOINT_SCHEMA_VERSION,
                    "status": "completed",
                    "raw_prototype_count": raw_prototype_count,
                    "prototype_count": len(prototypes),
                    "balance": stats.balance,
                    "updated_at": utc_now_iso(),
                },
            )
            tracker.update(
                completed=len(prototypes),
                total=len(prototypes),
                unit="prototypes",
                metrics={
                    "prototype_count_after_balance": len(prototypes),
                    "checkpoint_path": str(balanced_prototype_checkpoint_path),
                },
            )

        tracker.start("materialize_tasks", total=len(prototypes) * len(EVIDENCE_MODES), unit="tasks")
        tasks = materialize_tasks_checkpointed(
            modules=modules,
            prototypes=prototypes,
            store_dir=store_dir,
            rwr_hpc_flist=rwr_hpc_flist,
            rank_cache_context_dir_path=context_dir,
            seed=seed,
            checkpoint_path=task_checkpoint_path,
            tracker=tracker,
        )
        write_json(
            task_checkpoint_meta_path,
            {
                "schema_version": MATERIALIZE_CHECKPOINT_SCHEMA_VERSION,
                "status": "completed",
                "prototype_count": len(prototypes),
                "task_count": len(tasks),
                "updated_at": utc_now_iso(),
            },
        )
        tracker.update(completed=len(tasks), total=len(tasks), unit="tasks", metrics={"task_count": len(tasks)})

        module_rows = build_module_jsonl_rows(modules)
        split_report = build_split_report(modules=modules, prototypes=prototypes, tasks=tasks, stats=stats)
        manifest = build_manifest(
            store_dir=store_dir,
            rwr_hpc_flist=rwr_hpc_flist,
            rank_cache_dir=rank_cache_dir,
            rank_cache_context_path=context_dir,
            out_dir=out_dir,
            mentor_size_distribution=mentor_size_distribution,
            module_sizes=module_sizes,
            module_selection_method=module_selection_method,
            seed=seed,
            modules=modules,
            prototypes=prototypes,
            tasks=tasks,
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


def validate_rank_cache_against_rwr_loe(
    *,
    seed_gene_ids: list[str],
    context_dir: Path,
    rwr_hpc_flist: Path,
    rwr_hpc_build_dir: Path,
    scratch_dir: Path,
    rwr_hpc_cache_dir: Path,
    top_k: int = 20,
    edgelist_has_headers: bool = True,
) -> dict[str, Any]:
    app_backend = RwrHpcAppBackend(build_dir=rwr_hpc_build_dir)
    backend = RwrHpcStructuredBackend(
        flist=rwr_hpc_flist,
        app_backend=app_backend,
        scratch_root=scratch_dir,
        cache=None,
        rwr_hpc_build_id=str(rwr_hpc_build_dir.resolve()),
        no_edgelist_headers=not edgelist_has_headers,
    )
    mismatches = []
    for seed_gene_id in seed_gene_ids:
        cached = load_seed_rank_cache(context_dir, seed_gene_id)[:top_k]
        request = RwrLoeRequest.from_tool_arguments(
            {"seed_genes": [seed_gene_id], "top_k": top_k, "exclude_seed_genes": True}
        )
        direct = backend.run_rwr_loe(request).payload.get("ranked_genes", [])
        cached_genes = [row["gene"] for row in cached]
        direct_genes = [row["gene"] for row in direct]
        if cached_genes != direct_genes:
            mismatches.append(
                {
                    "seed_gene_id": seed_gene_id,
                    "cached_top_genes": cached_genes,
                    "direct_top_genes": direct_genes,
                }
            )
    return {
        "checked_seed_count": len(seed_gene_ids),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "rwr_hpc_cache_dir": str(rwr_hpc_cache_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or prewarm the full-brain RWR-LOE corpus.")
    parser.add_argument("--mode", choices=("prewarm", "materialize", "all", "validate-cache"), default="materialize")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--rwr-hpc-flist", type=Path, default=DEFAULT_RWR_HPC_FLIST)
    parser.add_argument("--rwr-hpc-build-dir", type=Path, default=DEFAULT_RWR_HPC_BUILD_DIR)
    parser.add_argument("--rwr-hpc-build-id", type=str, default=None)
    parser.add_argument("--rwr-hpc-cache-dir", type=Path, default=DEFAULT_RWR_HPC_CACHE_DIR)
    parser.add_argument("--rank-cache-dir", type=Path, default=DEFAULT_LOE_RANK_CACHE_DIR)
    parser.add_argument("--rank-cache-context-dir", type=Path, default=None)
    parser.add_argument("--scratch-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "mentor_rl_loe")
    parser.add_argument("--mentor-corpus-dir", type=Path, default=DEFAULT_MENTOR_CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--progress-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--shard-count", type=int, default=int(os.environ.get("LOE_SHARD_COUNT", "1")))
    parser.add_argument("--restart", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--reduction-method", type=str, default="geometric")
    parser.add_argument("--threshold", type=float, default=1e-10)
    parser.add_argument("--rwr-hpc-edgelist-has-headers", dest="edgelist_has_headers", action="store_true", default=True)
    parser.add_argument("--rwr-hpc-edgelist-no-headers", dest="edgelist_has_headers", action="store_false")
    parser.add_argument("--rwr-launcher-prefix", type=str, default=os.environ.get("LOE_RWR_LAUNCHER_PREFIX", ""))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument(
        "--materialize-workers",
        type=int,
        default=int(os.environ.get("LOE_MATERIALIZE_WORKERS", str(DEFAULT_MATERIALIZE_WORKERS))),
        help="Worker processes for checkpointed materialization stages. Default: LOE_MATERIALIZE_WORKERS or 1.",
    )
    parser.add_argument("--allow-local-full-run", action="store_true")
    parser.add_argument(
        "--module-selection-method",
        choices=("elbow", "topk"),
        default=os.environ.get("LOE_MODULE_SELECTION_METHOD", DEFAULT_MODULE_SELECTION_METHOD),
        help="How to convert each per-seed LOE ranking into a module. Default: elbow.",
    )
    parser.add_argument("--small-module-size", type=int, default=DEFAULT_MODULE_SIZES["small"])
    parser.add_argument("--medium-module-size", type=int, default=DEFAULT_MODULE_SIZES["medium"])
    parser.add_argument("--large-module-size", type=int, default=DEFAULT_MODULE_SIZES["large"])
    parser.add_argument("--validate-sample-size", type=int, default=3)
    parser.add_argument("--validate-top-k", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _guard_local_full_run(args: argparse.Namespace) -> None:
    if args.mode in {"prewarm", "all"} and args.max_genes is None and "SLURM_JOB_ID" not in os.environ:
        if not args.allow_local_full_run:
            raise SystemExit(
                "Refusing full RWR-LOE prewarm outside Slurm. Pass --max-genes for a smoke run "
                "or --allow-local-full-run if this is intentional."
            )


def main() -> None:
    args = parse_args()
    _guard_local_full_run(args)
    module_sizes = {
        "small": args.small_module_size,
        "medium": args.medium_module_size,
        "large": args.large_module_size,
    }
    build_id = args.rwr_hpc_build_id or str(args.rwr_hpc_build_dir.resolve())
    if args.rank_cache_context_dir is not None:
        context_dir = args.rank_cache_context_dir
        cache_context = load_rank_cache_context_from_dir(context_dir)
    else:
        cache_context = build_rank_cache_context(
            rwr_hpc_flist=args.rwr_hpc_flist,
            rwr_hpc_build_id=build_id,
            restart=args.restart,
            delta=args.delta,
            reduction_method=args.reduction_method,
            threshold=args.threshold,
            edgelist_has_headers=args.edgelist_has_headers,
        )
        context_dir = rank_cache_context_dir(args.rank_cache_dir, cache_context)

    result: dict[str, Any] = {"mode": args.mode, "rank_cache_context_dir": str(context_dir)}
    if args.mode in {"prewarm", "all"}:
        result["prewarm"] = prewarm_rwr_loe_rank_cache(
            store_dir=args.store_dir,
            rwr_hpc_flist=args.rwr_hpc_flist,
            rwr_hpc_build_dir=args.rwr_hpc_build_dir,
            rank_cache_dir=args.rank_cache_dir,
            scratch_dir=args.scratch_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            seed=args.seed,
            restart=args.restart,
            delta=args.delta,
            reduction_method=args.reduction_method,
            threshold=args.threshold,
            edgelist_has_headers=args.edgelist_has_headers,
            rwr_hpc_build_id=build_id,
            launcher_prefix=args.rwr_launcher_prefix,
            force=args.force,
            max_genes=args.max_genes,
        )
    if args.mode in {"materialize", "all"}:
        materialized = build_rwr_loe_corpus(
            store_dir=args.store_dir,
            rwr_hpc_flist=args.rwr_hpc_flist,
            rank_cache_dir=args.rank_cache_dir,
            out_dir=args.out_dir,
            mentor_corpus_dir=args.mentor_corpus_dir,
            seed=args.seed,
            module_sizes=module_sizes,
            module_selection_method=args.module_selection_method,
            progress_path=args.progress_path,
            max_genes=args.max_genes,
            rank_cache_context=cache_context,
            rank_cache_context_dir_path=context_dir,
            checkpoint_dir=args.checkpoint_dir,
            materialize_workers=args.materialize_workers,
        )
        result["materialize"] = {
            "manifest_path": str(args.out_dir / "manifest.json"),
            "module_count": materialized["manifest"]["module_count"],
            "task_count": materialized["manifest"]["task_count"],
        }
    if args.mode == "validate-cache":
        gene_ids = load_rwr_loe_gene_universe(args.store_dir)
        if args.max_genes is not None:
            gene_ids = gene_ids[:args.max_genes]
        sample_genes = deterministic_select_subset(
            gene_ids,
            subset_size=min(args.validate_sample_size, len(gene_ids)),
            seed=args.seed,
            salt="validate_rwr_loe_cache",
        )
        result["validate_cache"] = validate_rank_cache_against_rwr_loe(
            seed_gene_ids=sample_genes,
            context_dir=context_dir,
            rwr_hpc_flist=args.rwr_hpc_flist,
            rwr_hpc_build_dir=args.rwr_hpc_build_dir,
            scratch_dir=args.scratch_dir,
            rwr_hpc_cache_dir=args.rwr_hpc_cache_dir,
            top_k=args.validate_top_k,
            edgelist_has_headers=args.edgelist_has_headers,
        )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
