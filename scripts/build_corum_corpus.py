#!/usr/bin/env python3
"""Build the CORUM-grounded canonical corpus.

This script turns the raw CORUM human complexes export into the structured
corpus used by the rest of the MENTOR-RL pipeline. At a high level it:

1. parses CORUM rows
2. resolves member genes to Ensembl IDs
3. drops unusable complexes
4. deduplicates exact gene sets
5. assigns train/val/test splits
6. creates canonical task prototypes
7. materializes final task JSONL files

The code is intentionally split into small stages so long runs are easier to
follow and debug.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORUM_PATH = REPO_ROOT / "data" / "corum_humanComplexes.txt"
DEFAULT_MULTIPLEX_FLIST = Path(
    "/lustre/orion/syb111/proj-shared/Personal/sullivanka/Data/Human_multiplex_networks/"
    "hnv3_ppi_tftarget_bulkscbrainpen_385layer_multiplex_flist.txt"
)
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "corum_corpus"
DEFAULT_CACHE_DIR = DEFAULT_OUT_DIR / "cache"

MYGENE_API_URL = "https://mygene.info/v3/query"
MYGENE_FIELDS = (
    "symbol,alias,ensembl.gene,uniprot.Swiss-Prot,uniprot.TrEMBL,name,taxid"
)
SPLITS = ("train", "val", "test")
EVIDENCE_MODES = ("minimal", "graph", "contextual", "full")
RECOVERY_REFINEMENT_DIFFICULTIES = ("easy", "medium", "hard")
EXPLANATION_DIFFICULTY = "complete"
NONE_RELATIONSHIP_STATUS = "insufficient_support"
POSITIVE_RELATIONSHIP_STATUS = "validated_group"
CORPUS_BUILD_STAGES = (
    ("parse_corum", "Parse CORUM complexes"),
    ("load_multiplex_gene_universe", "Load multiplex gene universe"),
    ("prefetch_mygene_terms", "Prefetch MyGene mappings"),
    ("normalize_complexes", "Normalize CORUM complexes"),
    ("deduplicate_complexes", "Deduplicate normalized complexes"),
    ("assign_splits", "Assign train/val/test splits"),
    ("build_task_prototypes", "Build canonical task prototypes"),
    ("build_gene_annotation_index", "Build gene annotation index"),
    ("materialize_tasks", "Materialize canonical tasks"),
    ("build_split_report", "Build split report"),
    ("build_manifest", "Build manifest"),
    ("write_outputs", "Write corpus files"),
)


def utc_now_iso() -> str:
    """Return the current UTC time in a JSON-friendly ISO format."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ProgressTracker:
    """Write a simple progress file for long corpus-generation jobs."""

    def __init__(
        self,
        path: Path,
        stage_defs: tuple[tuple[str, str], ...],
    ) -> None:
        self.path = path
        self.stage_defs = stage_defs
        self.stage_index_lookup = {
            stage_name: index for index, (stage_name, _) in enumerate(stage_defs, start=1)
        }
        timestamp = utc_now_iso()
        self.state: dict[str, Any] = {
            "status": "running",
            "current_stage": None,
            "current_stage_label": None,
            "stage_index": 0,
            "stage_count": len(stage_defs),
            "stage_progress": {
                "completed": 0,
                "total": None,
                "unit": None,
            },
            "overall_progress": 0.0,
            "message": "Initialized CORUM corpus build.",
            "metrics": {},
            "run_context": {},
            "started_at": timestamp,
            "updated_at": timestamp,
            "stages": [
                {
                    "stage": stage_name,
                    "label": stage_label,
                    "status": "pending",
                }
                for stage_name, stage_label in stage_defs
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

    def _find_stage_entry(self, stage_name: str) -> dict[str, Any]:
        for entry in self.state["stages"]:
            if entry["stage"] == stage_name:
                return entry
        raise KeyError(f"Unknown progress stage: {stage_name}")

    def _mark_running_stage_complete(self) -> None:
        current_stage = self.state.get("current_stage")
        if not current_stage:
            return
        current_entry = self._find_stage_entry(current_stage)
        if current_entry["status"] == "running":
            current_entry["status"] = "completed"

    def _recompute_overall_progress(self) -> None:
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
        stage_fraction = 0.0
        if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
            stage_fraction = min(max(float(completed) / float(total), 0.0), 1.0)

        overall = ((stage_index - 1) + stage_fraction) / max(1, self.state["stage_count"])
        self.state["overall_progress"] = round(overall, 6)

    def set_context(self, context: dict[str, Any]) -> None:
        self.state["run_context"] = context
        self._touch()
        self._write()

    def start_stage(
        self,
        stage_name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        stage_index = self.stage_index_lookup[stage_name]
        stage_entry = self._find_stage_entry(stage_name)
        self._mark_running_stage_complete()
        stage_entry["status"] = "running"
        self.state["status"] = "running"
        self.state["current_stage"] = stage_name
        self.state["current_stage_label"] = stage_entry["label"]
        self.state["stage_index"] = stage_index
        self.state["stage_progress"] = {
            "completed": 0,
            "total": total,
            "unit": unit,
        }
        self.state["metrics"] = metrics or {}
        self.state["message"] = message or stage_entry["label"]
        self._touch()
        self._recompute_overall_progress()
        self._write()
        print(
            f"[corum_corpus] {stage_index}/{self.state['stage_count']} {stage_entry['label']}",
            flush=True,
        )

    def update(
        self,
        *,
        completed: int | float | None = None,
        total: int | float | None = None,
        unit: str | None = None,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.state.get("current_stage") is None:
            return

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
        self._recompute_overall_progress()
        self._write()

    def complete(self, *, message: str | None = None, metrics: dict[str, Any] | None = None) -> None:
        self._mark_running_stage_complete()
        self.state["status"] = "completed"
        if metrics:
            self.state["metrics"].update(metrics)
        self.state["message"] = message or "Completed CORUM corpus build."
        self._touch()
        self._recompute_overall_progress()
        self._write()
        print("[corum_corpus] build complete", flush=True)

    def fail(self, error: Exception) -> None:
        current_stage = self.state.get("current_stage")
        if current_stage:
            self._find_stage_entry(current_stage)["status"] = "failed"
        self.state["status"] = "failed"
        self.state["message"] = f"Build failed: {error}"
        self.state["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
        }
        self._touch()
        self._recompute_overall_progress()
        self._write()
        print(f"[corum_corpus] build failed: {error}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for corpus generation."""

    parser = argparse.ArgumentParser(description="Build the CORUM-grounded canonical corpus.")
    parser.add_argument(
        "--corum-path",
        type=Path,
        default=DEFAULT_CORUM_PATH,
        help="Path to the CORUM human complexes TSV export.",
    )
    parser.add_argument(
        "--multiplex-flist",
        type=Path,
        default=DEFAULT_MULTIPLEX_FLIST,
        help="Path to the multiplex flist used for the HumanNet gene universe.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the canonical corpus files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits and task generation.",
    )
    parser.add_argument(
        "--min-complex-size",
        type=int,
        default=2,
        help="Minimum number of unique mapped genes required to retain a complex.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Cache directory for MyGene responses and the multiplex gene universe.",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help="Optional path for the build progress tracker JSON. Defaults to <out-dir>/progress.json.",
    )
    return parser.parse_args()


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    """Yield fixed-size chunks from a list."""

    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_semicolon_field(raw_value: str) -> list[str]:
    """Split a CORUM semicolon-separated field into clean values."""

    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";")]


def parse_member_alias_field(raw_value: str) -> list[list[str]]:
    """Parse the nested alias column used for member gene synonyms."""

    alias_entries = parse_semicolon_field(raw_value)
    parsed: list[list[str]] = []
    for entry in alias_entries:
        parsed.append([alias.strip() for alias in entry.split(",") if alias.strip()])
    return parsed


def align_list(values: list[Any], size: int, fill_value: Any) -> list[Any]:
    """Pad or trim a list so it matches an expected width."""

    aligned = list(values[:size])
    while len(aligned) < size:
        aligned.append(copy.deepcopy(fill_value))
    return aligned


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    """Drop duplicates while keeping the first-seen order."""

    seen = set()
    unique_values = []
    for value in values:
        marker = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        if marker in seen:
            continue
        seen.add(marker)
        unique_values.append(value)
    return unique_values


def size_bin(size: int) -> str:
    """Map a complex size to the split-stratification bin used in the corpus."""

    if size <= 2:
        return "2"
    if size == 3:
        return "3"
    if 4 <= size <= 5:
        return "4-5"
    return "6+"


def parse_go_pairs(row: dict[str, str]) -> list[dict[str, str]]:
    """Align GO IDs and GO names from one raw CORUM row."""

    go_ids = parse_semicolon_field(row.get("functions_go_id", ""))
    go_names = parse_semicolon_field(row.get("functions_go_name", ""))
    width = max(len(go_ids), len(go_names))
    go_ids = align_list(go_ids, width, "")
    go_names = align_list(go_names, width, "")
    pairs = []
    for go_id, go_name in zip(go_ids, go_names):
        if not go_id and not go_name:
            continue
        pairs.append({"go_id": go_id, "go_name": go_name})
    return unique_preserve_order(pairs)


def parse_corum_row(row: dict[str, str]) -> dict[str, Any]:
    """Convert one raw CORUM TSV row into a structured Python dictionary."""

    gene_symbols = parse_semicolon_field(row.get("subunits_gene_name", ""))
    uniprot_ids = align_list(
        parse_semicolon_field(row.get("subunits_uniprot_id", "")),
        len(gene_symbols),
        "",
    )
    gene_aliases = align_list(
        parse_member_alias_field(row.get("subunits_gene_name_synonyms", "")),
        len(gene_symbols),
        [],
    )

    members = []
    for index, gene_symbol in enumerate(gene_symbols):
        members.append(
            {
                "member_index": index,
                "source_symbol": gene_symbol,
                "source_uniprot_id": uniprot_ids[index],
                "source_aliases": gene_aliases[index],
            }
        )

    synonyms = [syn for syn in parse_semicolon_field(row.get("synonyms", "")) if syn]
    pmids = [pmid for pmid in parse_semicolon_field(row.get("pmid", "")) if pmid]
    fcgs_names = [value for value in parse_semicolon_field(row.get("fcgs_name", "")) if value]
    fcgs_categories = [
        value for value in parse_semicolon_field(row.get("fcgs_category_name", "")) if value
    ]
    fcgs_ids = [value for value in parse_semicolon_field(row.get("fcgs_id", "")) if value]

    return {
        "source": "CORUM",
        "source_complex_id": int(row["complex_id"]),
        "source_complex_name": row.get("complex_name", "").strip(),
        "synonyms": synonyms,
        "organism": row.get("organism", "").strip(),
        "cell_line": row.get("cell_line", "").strip(),
        "pmids": pmids,
        "comments": {
            "complex": row.get("comment_complex", "").strip(),
            "members": row.get("comment_members", "").strip(),
            "disease": row.get("comment_disease", "").strip(),
        },
        "members_raw": members,
        "go_terms": parse_go_pairs(row),
        "fcgs": {
            "ids": unique_preserve_order(fcgs_ids),
            "names": unique_preserve_order(fcgs_names),
            "categories": unique_preserve_order(fcgs_categories),
        },
        "raw_fields": {
            "functions_evi": row.get("functions_evi", "").strip(),
            "functions_pmid": row.get("functions_pmid", "").strip(),
            "comment_drug": row.get("comment_drug", "").strip(),
            "comment_drug_formal": row.get("comment_drug_formal", "").strip(),
        },
    }


def load_corum_complexes(corum_path: Path) -> list[dict[str, Any]]:
    """Read and parse the full CORUM human complexes export."""

    with corum_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [parse_corum_row(row) for row in reader]


def load_json(path: Path, default: Any) -> Any:
    """Load JSON if it exists, otherwise return the provided default."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def extract_ensembl_gene_ids(hit: dict[str, Any]) -> list[str]:
    """Pull Ensembl gene IDs out of a MyGene hit."""

    ensembl = hit.get("ensembl")
    if not ensembl:
        return []
    if isinstance(ensembl, dict):
        gene = ensembl.get("gene")
        return [gene] if gene else []
    if isinstance(ensembl, list):
        genes = []
        for entry in ensembl:
            if not isinstance(entry, dict):
                continue
            gene = entry.get("gene")
            if gene:
                genes.append(gene)
        return sorted(set(genes))
    return []


def extract_uniprot_ids(hit: dict[str, Any]) -> set[str]:
    """Pull UniProt accessions out of a MyGene hit."""

    uniprot = hit.get("uniprot")
    if not isinstance(uniprot, dict):
        return set()
    values = set()
    for key in ("Swiss-Prot", "TrEMBL"):
        raw = uniprot.get(key)
        if isinstance(raw, str):
            values.add(raw.upper())
        elif isinstance(raw, list):
            values.update(value.upper() for value in raw if isinstance(value, str))
    return values


def extract_aliases(hit: dict[str, Any]) -> set[str]:
    """Pull alias names out of a MyGene hit."""

    aliases = hit.get("alias", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    return {alias.upper() for alias in aliases if isinstance(alias, str)}


class MyGeneResolver:
    """Resolve CORUM members to Ensembl IDs using cached MyGene lookups."""

    def __init__(
        self,
        cache_dir: Path,
        multiplex_genes: set[str] | None = None,
        batch_size: int = 500,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "mygene_query_cache.json"
        self.cache: dict[str, list[dict[str, Any]]] = load_json(self.cache_path, {})
        self.multiplex_genes = multiplex_genes or set()
        self.batch_size = batch_size
        self._dirty = False

    def cache_key(self, scope: str, query: str) -> str:
        """Build the normalized cache key for one lookup term."""

        return f"{scope}::{query.strip().upper()}"

    def save(self) -> None:
        """Write the query cache if it changed during this run."""

        if not self._dirty:
            return
        write_json(self.cache_path, self.cache)
        self._dirty = False

    def prefetch(
        self,
        scope: str,
        queries: Iterable[str],
        *,
        tracker: ProgressTracker | None = None,
        base_completed: int = 0,
        total_queries: int | None = None,
    ) -> None:
        """Preload MyGene hits for a batch of terms and update progress if requested."""

        ordered_queries = []
        seen = set()
        for query in queries:
            if not query or not str(query).strip():
                continue
            normalized = self.cache_key(scope, str(query))
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_queries.append(str(query).strip())

        missing = [
            query for query in ordered_queries if self.cache_key(scope, query) not in self.cache
        ]
        progress_total = (
            total_queries if total_queries is not None else base_completed + len(ordered_queries)
        )
        cached_count = len(ordered_queries) - len(missing)
        if not missing:
            if tracker:
                tracker.update(
                    completed=base_completed + len(ordered_queries),
                    total=progress_total,
                    unit="queries",
                    message=f"MyGene {scope} queries satisfied from cache.",
                    metrics={
                        "prefetch_scope": scope,
                        "queries_total": len(ordered_queries),
                        "queries_missing": 0,
                    },
                )
            return

        if tracker:
            tracker.update(
                completed=base_completed + cached_count,
                total=progress_total,
                unit="queries",
                message=f"Prefetching MyGene {scope} queries.",
                metrics={
                    "prefetch_scope": scope,
                    "queries_total": len(ordered_queries),
                    "queries_missing": len(missing),
                },
            )

        fetched_count = 0
        for chunk in chunks(missing, self.batch_size):
            response = requests.post(
                MYGENE_API_URL,
                data={
                    "q": ",".join(chunk),
                    "scopes": scope,
                    "species": "human",
                    "fields": MYGENE_FIELDS,
                    "size": 20,
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            if isinstance(payload, list):
                for item in payload:
                    query = item.get("query")
                    if not query:
                        continue
                    grouped[self.cache_key(scope, str(query))].append(item)
            for query in chunk:
                self.cache[self.cache_key(scope, query)] = grouped.get(
                    self.cache_key(scope, query),
                    [],
                )
            self._dirty = True
            fetched_count += len(chunk)
            if tracker:
                tracker.update(
                    completed=base_completed + cached_count + fetched_count,
                    total=progress_total,
                    unit="queries",
                    message=f"Prefetching MyGene {scope} queries.",
                    metrics={
                        "prefetch_scope": scope,
                        "queries_total": len(ordered_queries),
                        "queries_missing": len(missing),
                        "queries_fetched": fetched_count,
                    },
                )

        self.save()

    def get_hits(self, scope: str, query: str) -> list[dict[str, Any]]:
        """Return cached MyGene hits for one query term."""

        return self.cache.get(self.cache_key(scope, query), [])

    def _matched_gene_candidates(self, scope: str, query: str) -> dict[str, list[dict[str, Any]]]:
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        query_upper = query.strip().upper()
        for hit in self.get_hits(scope, query):
            if hit.get("taxid") not in (None, 9606):
                continue

            gene_ids = extract_ensembl_gene_ids(hit)
            if not gene_ids:
                continue

            matched = False
            if scope == "symbol":
                matched = str(hit.get("symbol", "")).upper() == query_upper
            elif scope == "uniprot":
                matched = query_upper in extract_uniprot_ids(hit)
            elif scope == "alias":
                matched = query_upper in extract_aliases(hit)
            elif scope == "ensembl.gene":
                matched = query_upper in {gene_id.upper() for gene_id in gene_ids}

            if not matched:
                continue

            if self.multiplex_genes:
                gene_ids = [gene_id for gene_id in gene_ids if gene_id in self.multiplex_genes]
                if not gene_ids:
                    continue

            for gene_id in gene_ids:
                candidates[gene_id].append(
                    {
                        "hit_id": hit.get("_id"),
                        "symbol": hit.get("symbol"),
                        "name": hit.get("name"),
                        "matched_scope": scope,
                        "matched_query": query,
                    }
                )
        return candidates

    def resolve_member(
        self,
        gene_symbol: str,
        uniprot_id: str,
        aliases: list[str],
    ) -> dict[str, Any]:
        """Resolve one CORUM member using symbol, then UniProt, then alias."""

        attempts = []
        if gene_symbol:
            attempts.append(("symbol", gene_symbol))
        if uniprot_id:
            attempts.append(("uniprot", uniprot_id))
        for alias in aliases:
            if alias and alias.upper() != gene_symbol.upper():
                attempts.append(("alias", alias))

        for scope, query in attempts:
            candidates = self._matched_gene_candidates(scope, query)
            if len(candidates) != 1:
                continue
            ensembl_gene_id, hit_details = next(iter(candidates.items()))
            return {
                "status": "resolved",
                "ensembl_gene_id": ensembl_gene_id,
                "display_symbol": hit_details[0].get("symbol") or gene_symbol or ensembl_gene_id,
                "resolved_via": scope,
                "matched_query": query,
                "hit_ids": sorted(
                    {
                        detail["hit_id"]
                        for detail in hit_details
                        if detail.get("hit_id") is not None
                    }
                ),
                "candidate_count": len(candidates),
            }

        return {
            "status": "unresolved",
            "ensembl_gene_id": None,
            "display_symbol": gene_symbol or None,
            "resolved_via": None,
            "matched_query": None,
            "hit_ids": [],
            "candidate_count": 0,
        }

    def get_gene_annotation(self, ensembl_gene_id: str) -> dict[str, Any]:
        """Fetch a display symbol and lookup metadata for a resolved gene ID."""

        self.prefetch("ensembl.gene", [ensembl_gene_id])
        candidates = self._matched_gene_candidates("ensembl.gene", ensembl_gene_id)
        annotation_candidates = candidates.get(ensembl_gene_id, [])
        if not annotation_candidates:
            return {
                "ensembl_gene_id": ensembl_gene_id,
                "symbol": ensembl_gene_id,
                "resolved_via": "fallback_gene_id",
                "hit_ids": [],
            }
        symbols = sorted(
            {
                detail["symbol"]
                for detail in annotation_candidates
                if detail.get("symbol")
            }
        )
        symbol = symbols[0] if len(symbols) == 1 else (symbols[0] if symbols else ensembl_gene_id)
        return {
            "ensembl_gene_id": ensembl_gene_id,
            "symbol": symbol,
            "resolved_via": "ensembl.gene",
            "hit_ids": sorted(
                {
                    detail["hit_id"]
                    for detail in annotation_candidates
                    if detail.get("hit_id") is not None
                }
            ),
        }


def load_multiplex_gene_universe(
    flist_path: Path,
    cache_dir: Path,
    tracker: ProgressTracker | None = None,
) -> set[str]:
    """Build or load the set of genes that appear anywhere in the multiplex."""

    cache_path = cache_dir / "multiplex_gene_universe.json"
    cached = load_json(cache_path, None)
    if isinstance(cached, dict) and cached.get("multiplex_flist") == str(flist_path):
        if tracker:
            tracker.update(
                completed=1,
                total=1,
                unit="cache",
                message="Loaded multiplex gene universe from cache.",
                metrics={
                    "from_cache": True,
                    "gene_universe_size": len(cached.get("gene_ids", [])),
                    "network_files_processed": 0,
                },
            )
        return set(cached.get("gene_ids", []))

    network_paths: list[Path] = []
    with flist_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            network_path = Path(parts[0])
            if not network_path.is_absolute():
                network_path = flist_path.parent / network_path
            network_paths.append(network_path)

    gene_ids = set()
    edges_processed = 0
    total_networks = len(network_paths)
    if tracker:
        tracker.update(
            completed=0,
            total=total_networks,
            unit="network files",
            message="Scanning multiplex network files.",
            metrics={
                "from_cache": False,
                "network_files_processed": 0,
                "gene_universe_size": 0,
                "edges_processed": 0,
            },
        )

    for index, network_path in enumerate(network_paths, start=1):
        with network_path.open("r", encoding="utf-8") as network_handle:
            for edge_line in network_handle:
                edge_parts = edge_line.rstrip("\n").split("\t")
                if len(edge_parts) < 2:
                    continue
                gene_ids.add(edge_parts[0])
                gene_ids.add(edge_parts[1])
                edges_processed += 1
        if tracker:
            tracker.update(
                completed=index,
                total=total_networks,
                unit="network files",
                message=f"Scanned multiplex network file {index} of {total_networks}.",
                metrics={
                    "from_cache": False,
                    "network_files_processed": index,
                    "gene_universe_size": len(gene_ids),
                    "edges_processed": edges_processed,
                },
            )

    write_json(
        cache_path,
        {
            "multiplex_flist": str(flist_path),
            "gene_ids": sorted(gene_ids),
        },
    )
    return gene_ids


def collect_prefetch_terms(parsed_complexes: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Collect symbol, UniProt, and alias terms needed for MyGene prefetch."""

    symbol_terms = set()
    uniprot_terms = set()
    alias_terms = set()
    for complex_row in parsed_complexes:
        for member in complex_row["members_raw"]:
            symbol = member["source_symbol"].strip()
            if symbol:
                symbol_terms.add(symbol)
            uniprot_id = member["source_uniprot_id"].strip()
            if uniprot_id:
                uniprot_terms.add(uniprot_id)
            for alias in member["source_aliases"]:
                alias = alias.strip()
                if alias:
                    alias_terms.add(alias)
    return {
        "symbol": symbol_terms,
        "uniprot": uniprot_terms,
        "alias": alias_terms,
    }


def normalize_complexes(
    parsed_complexes: list[dict[str, Any]],
    resolver: MyGeneResolver,
    min_complex_size: int,
    tracker: ProgressTracker | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve members, drop unusable complexes, and create normalized rows."""

    normalization_summary = {
        "parsed_complexes": len(parsed_complexes),
        "excluded_complexes_unresolved_members": 0,
        "excluded_complexes_below_min_size": 0,
        "excluded_member_resolution_events": 0,
        "duplicate_member_mappings_collapsed": 0,
    }
    unresolved_examples = []
    normalized = []

    total_complexes = len(parsed_complexes)
    for index, complex_row in enumerate(parsed_complexes, start=1):
        member_mappings = []
        unresolved_members = []
        for member in complex_row["members_raw"]:
            resolution = resolver.resolve_member(
                gene_symbol=member["source_symbol"],
                uniprot_id=member["source_uniprot_id"],
                aliases=member["source_aliases"],
            )
            enriched_member = {
                **member,
                **resolution,
            }
            if resolution["status"] != "resolved":
                unresolved_members.append(enriched_member)
            member_mappings.append(enriched_member)

        if unresolved_members:
            normalization_summary["excluded_complexes_unresolved_members"] += 1
            normalization_summary["excluded_member_resolution_events"] += len(unresolved_members)
            unresolved_examples.append(
                {
                    "source_complex_id": complex_row["source_complex_id"],
                    "source_complex_name": complex_row["source_complex_name"],
                    "unresolved_members": unresolved_members,
                }
            )
            continue

        member_records = []
        seen_gene_ids = set()
        duplicate_member_mappings = 0
        for member in member_mappings:
            gene_id = member["ensembl_gene_id"]
            if gene_id in seen_gene_ids:
                duplicate_member_mappings += 1
                continue
            seen_gene_ids.add(gene_id)
            member_records.append(member)

        gene_ids = sorted(seen_gene_ids)
        if len(gene_ids) < min_complex_size:
            normalization_summary["excluded_complexes_below_min_size"] += 1
            continue

        normalization_summary["duplicate_member_mappings_collapsed"] += duplicate_member_mappings
        gene_symbol_map = {member["ensembl_gene_id"]: member["display_symbol"] for member in member_records}
        normalized.append(
            {
                **complex_row,
                "source_complex_ids": [complex_row["source_complex_id"]],
                "source_complex_names": [complex_row["source_complex_name"]],
                "duplicate_source_complex_ids": [],
                "member_mappings": member_records,
                "gene_ids": gene_ids,
                "gene_symbols": [gene_symbol_map[gene_id] for gene_id in gene_ids],
                "gene_symbol_map": gene_symbol_map,
                "size": len(gene_ids),
                "size_bin": size_bin(len(gene_ids)),
                "has_fcgs": bool(complex_row["fcgs"]["names"]),
            }
        )

        if tracker and (index % 100 == 0 or index == total_complexes):
            tracker.update(
                completed=index,
                total=total_complexes,
                unit="complexes",
                message=f"Normalized {index} of {total_complexes} CORUM complexes.",
                metrics={
                    "retained_after_normalization": len(normalized),
                    "excluded_complexes_unresolved_members": normalization_summary[
                        "excluded_complexes_unresolved_members"
                    ],
                    "excluded_complexes_below_min_size": normalization_summary[
                        "excluded_complexes_below_min_size"
                    ],
                    "excluded_member_resolution_events": normalization_summary[
                        "excluded_member_resolution_events"
                    ],
                },
            )

    normalization_summary["retained_after_normalization"] = len(normalized)
    normalization_summary["unresolved_examples"] = unresolved_examples[:20]
    return normalized, normalization_summary


def aggregate_complex_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge duplicate normalized rows that collapse to the same gene set."""

    ordered = sorted(group, key=lambda row: row["source_complex_id"])
    primary = copy.deepcopy(ordered[0])
    duplicate_rows = ordered[1:]

    primary["source_complex_ids"] = [row["source_complex_id"] for row in ordered]
    primary["source_complex_names"] = unique_preserve_order(
        [row["source_complex_name"] for row in ordered if row["source_complex_name"]]
    )
    primary["duplicate_source_complex_ids"] = [row["source_complex_id"] for row in duplicate_rows]
    primary["synonyms"] = unique_preserve_order(
        item for row in ordered for item in row.get("synonyms", [])
    )
    primary["pmids"] = unique_preserve_order(item for row in ordered for item in row.get("pmids", []))
    primary["go_terms"] = unique_preserve_order(
        item for row in ordered for item in row.get("go_terms", [])
    )
    primary["fcgs"] = {
        "ids": unique_preserve_order(
            item for row in ordered for item in row.get("fcgs", {}).get("ids", [])
        ),
        "names": unique_preserve_order(
            item for row in ordered for item in row.get("fcgs", {}).get("names", [])
        ),
        "categories": unique_preserve_order(
            item for row in ordered for item in row.get("fcgs", {}).get("categories", [])
        ),
    }
    primary["comments"] = {
        key: unique_preserve_order(
            row.get("comments", {}).get(key, "")
            for row in ordered
            if row.get("comments", {}).get(key, "")
        )
        for key in ("complex", "members", "disease")
    }
    primary["duplicate_source_complex_names"] = unique_preserve_order(
        row["source_complex_name"] for row in duplicate_rows if row["source_complex_name"]
    )
    return primary


def build_mechanism_labels(complex_row: dict[str, Any]) -> dict[str, Any]:
    """Create the structured mechanism label block stored with each complex."""

    go_ids = [entry["go_id"] for entry in complex_row["go_terms"] if entry.get("go_id")]
    go_names = [entry["go_name"] for entry in complex_row["go_terms"] if entry.get("go_name")]
    fcgs_names = complex_row["fcgs"]["names"]
    fcgs_categories = complex_row["fcgs"]["categories"]
    primary_label = (
        (fcgs_names[0] if fcgs_names else None)
        or (go_names[0] if go_names else None)
        or complex_row["source_complex_name"]
    )
    return {
        "go_terms": complex_row["go_terms"],
        "go_ids": go_ids,
        "go_names": go_names,
        "fcgs_ids": complex_row["fcgs"]["ids"],
        "fcgs_names": fcgs_names,
        "fcgs_categories": fcgs_categories,
        "primary_label": primary_label,
    }


def deduplicate_normalized_complexes(
    normalized_complexes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse exact post-normalization duplicates into one retained record."""

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for complex_row in normalized_complexes:
        grouped[tuple(complex_row["gene_ids"])].append(complex_row)

    deduplicated = []
    removed_duplicates = 0
    for group in grouped.values():
        aggregated = aggregate_complex_group(group)
        aggregated["mechanism_labels"] = build_mechanism_labels(aggregated)
        deduplicated.append(aggregated)
        removed_duplicates += max(0, len(group) - 1)

    deduplicated.sort(key=lambda row: row["source_complex_id"])
    for index, complex_row in enumerate(deduplicated, start=1):
        complex_row["complex_record_id"] = f"corum_complex_{complex_row['source_complex_id']:05d}"

    dedup_summary = {
        "deduplicated_complexes": len(deduplicated),
        "removed_duplicate_complex_rows": removed_duplicates,
    }
    return deduplicated, dedup_summary


def stratified_split_counts(size: int) -> tuple[int, int, int]:
    """Compute train/val/test counts for one stratum."""

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


def assign_splits(
    complex_rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Assign deterministic complex-level splits inside each stratum."""

    split_assignments = []
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for complex_row in complex_rows:
        strata[(complex_row["size_bin"], complex_row["has_fcgs"])].append(complex_row)

    for key in sorted(strata.keys()):
        rows = sorted(strata[key], key=lambda row: row["complex_record_id"])
        rng = random.Random(f"{seed}|split|{key[0]}|{key[1]}")
        rng.shuffle(rows)
        train_count, val_count, test_count = stratified_split_counts(len(rows))
        boundaries = {
            "train": train_count,
            "val": train_count + val_count,
            "test": train_count + val_count + test_count,
        }
        for index, row in enumerate(rows):
            row_copy = copy.deepcopy(row)
            if index < boundaries["train"]:
                row_copy["split"] = "train"
            elif index < boundaries["val"]:
                row_copy["split"] = "val"
            else:
                row_copy["split"] = "test"
            split_assignments.append(row_copy)

    split_assignments.sort(key=lambda row: row["complex_record_id"])
    return split_assignments


def build_gene_conflict_index(
    complex_rows: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Track which complexes each gene appears in."""

    gene_to_complexes: dict[str, set[str]] = defaultdict(set)
    gene_to_symbol: dict[str, str] = {}
    for complex_row in complex_rows:
        for gene_id, gene_symbol in zip(complex_row["gene_ids"], complex_row["gene_symbols"]):
            gene_to_complexes[gene_id].add(complex_row["complex_record_id"])
            gene_to_symbol.setdefault(gene_id, gene_symbol)
    return gene_to_complexes, gene_to_symbol


def deterministic_select_subset(
    values: list[str],
    subset_size: int,
    seed: int,
    salt: str,
) -> list[str]:
    """Choose a deterministic subset using a seed plus a per-call salt."""

    if subset_size <= 0:
        return []
    ordered = sorted(values)
    rng = random.Random(f"{seed}|subset|{salt}")
    rng.shuffle(ordered)
    return sorted(ordered[:subset_size])


def sample_conflict_free_genes(
    *,
    universe: Iterable[str],
    sample_size: int,
    gene_to_complexes: dict[str, set[str]],
    seed: int,
    salt: str,
    forbidden_genes: set[str] | None = None,
    forbidden_complexes: set[str] | None = None,
) -> list[str]:
    """Sample genes that do not clash through shared-complex membership."""

    if sample_size <= 0:
        return []

    forbidden_genes = forbidden_genes or set()
    forbidden_complexes = forbidden_complexes or set()
    candidate_universe = sorted(set(universe) - forbidden_genes)
    if len(candidate_universe) < sample_size:
        raise ValueError(f"Cannot sample {sample_size} genes from a universe of {len(candidate_universe)}.")

    for attempt in range(32):
        rng = random.Random(f"{seed}|sample|{salt}|{attempt}")
        shuffled = candidate_universe[:]
        rng.shuffle(shuffled)

        selected = []
        used_complexes = set(forbidden_complexes)
        for gene_id in shuffled:
            gene_complexes = gene_to_complexes.get(gene_id, set())
            if gene_complexes & used_complexes:
                continue
            selected.append(gene_id)
            used_complexes.update(gene_complexes)
            if len(selected) == sample_size:
                return sorted(selected)

    raise ValueError(f"Failed to sample {sample_size} conflict-free genes for {salt}.")


def recovery_drop_count(size: int, difficulty: str) -> int:
    """Return how many genes to hide for a recovery task."""

    if difficulty == "easy":
        raw = 1
    elif difficulty == "medium":
        raw = max(1, round(0.2 * size))
    else:
        raw = max(2, round(0.33 * size))
    return min(raw, max(1, size - 2))


def refinement_add_count(size: int, difficulty: str) -> int:
    """Return how many noise genes to add for a refinement task."""

    if difficulty == "easy":
        return 1
    if difficulty == "medium":
        return max(1, round(0.2 * size))
    return max(2, round(0.33 * size))


def positive_prototype_count(complex_size: int) -> int:
    """Count how many positive prototypes one complex will generate."""

    count = 1
    if complex_size >= 3:
        count += len(RECOVERY_REFINEMENT_DIFFICULTIES)
    if complex_size >= 2:
        count += len(RECOVERY_REFINEMENT_DIFFICULTIES)
    return count


def build_context_text(task_type: str, complex_row: dict[str, Any] | None) -> str:
    """Choose the text context shown to the model for contextual evidence modes."""

    if complex_row is None:
        return (
            "This seed set was flagged for mechanistic follow-up, but no curated shared-context note "
            "was attached to the set."
        )
    for key in ("complex", "members", "disease"):
        entries = complex_row.get("comments", {}).get(key, [])
        if entries:
            return entries[0]
    if task_type == "recovery":
        return "The seed genes were drawn from a partially observed mechanistic module."
    if task_type == "refinement":
        return "The seed genes may include unrelated noise and should be refined before interpretation."
    return "The seed genes were curated as a potentially coherent module for mechanistic interpretation."


def build_graph_query_spec(
    multiplex_flist: Path,
    seed_gene_ids: list[str],
    seed_gene_symbols: list[str],
) -> dict[str, Any]:
    """Create the lightweight graph request stored in graph/full evidence modes."""

    return {
        "multiplex_flist": str(multiplex_flist),
        "operator": "induce_subgraph",
        "layer_scope": "all",
        "materialized": False,
        "seed_gene_ids": seed_gene_ids,
        "seed_gene_symbols": seed_gene_symbols,
    }


def build_structured_annotations(complex_row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Create the structured annotation block shown in the full evidence mode."""

    if complex_row is None:
        return {
            "go_terms": [],
            "fcgs_ids": [],
            "fcgs_names": [],
            "fcgs_categories": [],
            "shared_annotations_available": False,
        }
    mechanism_labels = complex_row["mechanism_labels"]
    return {
        "go_terms": mechanism_labels["go_terms"],
        "fcgs_ids": mechanism_labels["fcgs_ids"],
        "fcgs_names": mechanism_labels["fcgs_names"],
        "fcgs_categories": mechanism_labels["fcgs_categories"],
        "shared_annotations_available": True,
    }


def build_query_text(
    task_type: str,
    evidence_mode: str,
    seed_gene_symbols: list[str],
) -> tuple[str, str]:
    """Render the user-facing query text for one task and evidence mode."""

    gene_list = ", ".join(seed_gene_symbols)
    if task_type == "explanation":
        templates = {
            "minimal": "Explain the strongest shared mechanism supported by the following gene set: {genes}.",
            "graph": (
                "Using the provided seed genes and graph query specification, explain the strongest "
                "shared mechanism supported by: {genes}."
            ),
            "contextual": (
                "Using the seed genes and supporting context, explain the strongest shared mechanism "
                "supported by: {genes}."
            ),
            "full": (
                "Using the seed genes, graph query specification, context, and structured annotations, "
                "explain the strongest shared mechanism supported by: {genes}."
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
            "contextual": (
                "Use the seed genes and supporting context to recover the most coherent related module "
                "and explain its shared mechanism: {genes}."
            ),
            "full": (
                "Use the seed genes, graph query specification, context, and structured annotations to "
                "recover the most coherent related module and explain its shared mechanism: {genes}."
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
            "contextual": (
                "Using the seed genes and supporting context, refine this seed set by removing unrelated "
                "genes, then explain the shared mechanism of the remaining module: {genes}."
            ),
            "full": (
                "Using the seed genes, graph query specification, context, and structured annotations, "
                "refine this seed set by removing unrelated genes, then explain the shared mechanism of "
                "the remaining module: {genes}."
            ),
        }
    else:
        templates = {
            "minimal": (
                "Determine whether this gene set supports one shared mechanism. If not, return insufficient "
                "support or multiple groups: {genes}."
            ),
            "graph": (
                "Using the seed genes and graph query specification, determine whether this gene set "
                "supports one shared mechanism. If not, return insufficient support or multiple groups: {genes}."
            ),
            "contextual": (
                "Using the seed genes and supporting context, determine whether this gene set supports one "
                "shared mechanism. If not, return insufficient support or multiple groups: {genes}."
            ),
            "full": (
                "Using the seed genes, graph query specification, context, and structured annotations, "
                "determine whether this gene set supports one shared mechanism. If not, return insufficient "
                "support or multiple groups: {genes}."
            ),
        }
    template_id = f"{task_type}.{evidence_mode}.v1"
    return templates[evidence_mode].format(genes=gene_list), template_id


def build_visible_inputs(
    *,
    evidence_mode: str,
    task_type: str,
    seed_gene_ids: list[str],
    seed_gene_symbols: list[str],
    multiplex_flist: Path,
    complex_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the part of the task row that the agent is allowed to see."""

    visible_inputs = {
        "seed_gene_ids": seed_gene_ids,
        "seed_gene_symbols": seed_gene_symbols,
        "context_text": None,
        "graph_query_spec": None,
        "structured_annotations": None,
    }
    if evidence_mode in ("contextual", "full"):
        visible_inputs["context_text"] = build_context_text(task_type, complex_row)
    if evidence_mode in ("graph", "full"):
        visible_inputs["graph_query_spec"] = build_graph_query_spec(
            multiplex_flist=multiplex_flist,
            seed_gene_ids=seed_gene_ids,
            seed_gene_symbols=seed_gene_symbols,
        )
    if evidence_mode == "full":
        visible_inputs["structured_annotations"] = build_structured_annotations(complex_row)
    return visible_inputs


def build_task_prototypes(
    complex_rows: list[dict[str, Any]],
    multiplex_gene_universe: set[str],
    seed: int,
    tracker: ProgressTracker | None = None,
) -> list[dict[str, Any]]:
    """Create task skeletons before expanding them across evidence modes."""

    gene_to_complexes, gene_to_symbol = build_gene_conflict_index(complex_rows)
    retained_corum_gene_universe = sorted(gene_to_symbol.keys())
    multiplex_only_noise_genes = sorted(set(multiplex_gene_universe) - set(retained_corum_gene_universe))
    unique_genes_by_complex = {
        complex_row["complex_record_id"]: [
            gene_id
            for gene_id in complex_row["gene_ids"]
            if len(gene_to_complexes.get(gene_id, set())) == 1
        ]
        for complex_row in complex_rows
    }
    complexes_with_unique_genes = sorted(
        complex_id for complex_id, gene_ids in unique_genes_by_complex.items() if gene_ids
    )
    prototypes = []
    positive_prototype_target = sum(
        positive_prototype_count(complex_row["size"]) for complex_row in complex_rows
    )
    total_work_units = len(complex_rows) + positive_prototype_target
    total_complexes = len(complex_rows)
    for index, complex_row in enumerate(complex_rows, start=1):
        complex_id = complex_row["complex_record_id"]
        target_gene_ids = list(complex_row["gene_ids"])

        explanation_prototype = {
            "prototype_id": f"{complex_id}.explanation.{EXPLANATION_DIFFICULTY}",
            "task_type": "explanation",
            "difficulty": EXPLANATION_DIFFICULTY,
            "split": complex_row["split"],
            "target_complex_record_id": complex_id,
            "input_gene_ids": list(target_gene_ids),
            "relationship_status": POSITIVE_RELATIONSHIP_STATUS,
            "matched_positive_prototype_id": None,
        }
        prototypes.append(explanation_prototype)

        if len(target_gene_ids) >= 3:
            for difficulty in RECOVERY_REFINEMENT_DIFFICULTIES:
                drop_count = recovery_drop_count(len(target_gene_ids), difficulty)
                dropped_genes = deterministic_select_subset(
                    target_gene_ids,
                    subset_size=drop_count,
                    seed=seed,
                    salt=f"{complex_id}|recovery|{difficulty}",
                )
                input_gene_ids = sorted(set(target_gene_ids) - set(dropped_genes))
                prototypes.append(
                    {
                        "prototype_id": f"{complex_id}.recovery.{difficulty}",
                        "task_type": "recovery",
                        "difficulty": difficulty,
                        "split": complex_row["split"],
                        "target_complex_record_id": complex_id,
                        "input_gene_ids": input_gene_ids,
                        "relationship_status": POSITIVE_RELATIONSHIP_STATUS,
                        "matched_positive_prototype_id": None,
                        "dropped_gene_ids": dropped_genes,
                    }
                )

        if len(target_gene_ids) >= 2:
            target_related_complexes = set()
            for gene_id in target_gene_ids:
                target_related_complexes.update(gene_to_complexes.get(gene_id, set()))
            for difficulty in RECOVERY_REFINEMENT_DIFFICULTIES:
                add_count = refinement_add_count(len(target_gene_ids), difficulty)
                multiplex_only_candidates = sorted(set(multiplex_only_noise_genes) - set(target_gene_ids))
                if len(multiplex_only_candidates) >= add_count:
                    noise_gene_ids = deterministic_select_subset(
                        multiplex_only_candidates,
                        subset_size=add_count,
                        seed=seed,
                        salt=f"{complex_id}|refinement|{difficulty}|multiplex_only",
                    )
                else:
                    noise_gene_ids = sample_conflict_free_genes(
                        universe=multiplex_gene_universe,
                        sample_size=add_count,
                        gene_to_complexes=gene_to_complexes,
                        seed=seed,
                        salt=f"{complex_id}|refinement|{difficulty}",
                        forbidden_genes=set(target_gene_ids),
                        forbidden_complexes=target_related_complexes,
                    )
                input_gene_ids = sorted(set(target_gene_ids) | set(noise_gene_ids))
                prototypes.append(
                    {
                        "prototype_id": f"{complex_id}.refinement.{difficulty}",
                        "task_type": "refinement",
                        "difficulty": difficulty,
                        "split": complex_row["split"],
                        "target_complex_record_id": complex_id,
                        "input_gene_ids": input_gene_ids,
                        "relationship_status": POSITIVE_RELATIONSHIP_STATUS,
                        "matched_positive_prototype_id": None,
                        "noise_gene_ids": noise_gene_ids,
                    }
                )

        if tracker and (index % 100 == 0 or index == total_complexes):
            tracker.update(
                completed=index,
                total=total_work_units,
                unit="work units",
                message=f"Built positive task prototypes for {index} of {total_complexes} complexes.",
                metrics={
                    "positive_prototypes_built": len(prototypes),
                    "positive_prototype_target": positive_prototype_target,
                    "none_prototypes_built": 0,
                },
            )

    positive_prototypes = list(prototypes)
    total_positive_prototypes = len(positive_prototypes)
    for index, positive_prototype in enumerate(positive_prototypes, start=1):
        matched_input_size = len(positive_prototype["input_gene_ids"])
        if len(complexes_with_unique_genes) >= matched_input_size:
            chosen_complex_ids = deterministic_select_subset(
                complexes_with_unique_genes,
                subset_size=matched_input_size,
                seed=seed,
                salt=f"{positive_prototype['prototype_id']}|none|complexes",
            )
            none_input_gene_ids = sorted(
                deterministic_select_subset(
                    unique_genes_by_complex[complex_id],
                    subset_size=1,
                    seed=seed,
                    salt=f"{positive_prototype['prototype_id']}|none|{complex_id}",
                )[0]
                for complex_id in chosen_complex_ids
            )
        else:
            none_input_gene_ids = sample_conflict_free_genes(
                universe=retained_corum_gene_universe,
                sample_size=matched_input_size,
                gene_to_complexes=gene_to_complexes,
                seed=seed,
                salt=f"{positive_prototype['prototype_id']}|none",
            )
        prototypes.append(
            {
                "prototype_id": f"none.matched_to.{positive_prototype['prototype_id']}",
                "task_type": "none",
                "difficulty": positive_prototype["difficulty"],
                "split": positive_prototype["split"],
                "target_complex_record_id": None,
                "input_gene_ids": none_input_gene_ids,
                "relationship_status": NONE_RELATIONSHIP_STATUS,
                "matched_positive_prototype_id": positive_prototype["prototype_id"],
            }
        )

        if tracker and (index % 250 == 0 or index == total_positive_prototypes):
            tracker.update(
                completed=len(complex_rows) + index,
                total=total_work_units,
                unit="work units",
                message=f"Built matched none prototypes for {index} of {total_positive_prototypes} positive prototypes.",
                metrics={
                    "positive_prototypes_built": total_positive_prototypes,
                    "positive_prototype_target": positive_prototype_target,
                    "none_prototypes_built": index,
                    "prototype_count": len(prototypes),
                },
            )

    return sorted(prototypes, key=lambda prototype: prototype["prototype_id"])


def build_gene_annotation_index(
    complex_rows: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    resolver: MyGeneResolver,
    tracker: ProgressTracker | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect display metadata for every gene that can appear in a task."""

    annotation_index: dict[str, dict[str, Any]] = {}
    for complex_row in complex_rows:
        for member in complex_row["member_mappings"]:
            annotation_index[member["ensembl_gene_id"]] = {
                "ensembl_gene_id": member["ensembl_gene_id"],
                "symbol": member["display_symbol"],
                "resolved_via": member["resolved_via"],
                "matched_query": member["matched_query"],
                "hit_ids": member["hit_ids"],
            }

    missing_gene_ids = sorted(
        {
            gene_id
            for prototype in prototypes
            for gene_id in prototype["input_gene_ids"]
            if gene_id not in annotation_index
        }
    )
    stage_total = max(1, len(missing_gene_ids))
    if tracker:
        tracker.update(
            completed=0 if missing_gene_ids else 1,
            total=stage_total,
            unit="genes",
            message=(
                "Resolving missing gene annotations."
                if missing_gene_ids
                else "No missing gene annotations required."
            ),
            metrics={
                "annotation_index_size": len(annotation_index),
                "missing_gene_annotation_count": len(missing_gene_ids),
            },
        )
    if missing_gene_ids:
        resolver.prefetch(
            "ensembl.gene",
            missing_gene_ids,
            tracker=tracker,
            base_completed=0,
            total_queries=stage_total,
        )
        for index, gene_id in enumerate(missing_gene_ids, start=1):
            annotation_index[gene_id] = resolver.get_gene_annotation(gene_id)
            if tracker and (index % 250 == 0 or index == len(missing_gene_ids)):
                tracker.update(
                    completed=index,
                    total=stage_total,
                    unit="genes",
                    message=f"Resolved {index} of {len(missing_gene_ids)} missing gene annotations.",
                    metrics={
                        "annotation_index_size": len(annotation_index),
                        "missing_gene_annotation_count": len(missing_gene_ids),
                    },
                )
    return annotation_index


def materialize_tasks(
    complex_rows: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    annotation_index: dict[str, dict[str, Any]],
    multiplex_flist: Path,
    seed: int,
    tracker: ProgressTracker | None = None,
) -> list[dict[str, Any]]:
    """Expand prototypes into the final canonical task rows."""

    complexes_by_id = {row["complex_record_id"]: row for row in complex_rows}
    tasks = []
    total_task_count = len(prototypes) * len(EVIDENCE_MODES)
    for prototype_index, prototype in enumerate(prototypes, start=1):
        complex_row = (
            complexes_by_id.get(prototype["target_complex_record_id"])
            if prototype["target_complex_record_id"]
            else None
        )
        seed_gene_ids = list(prototype["input_gene_ids"])
        seed_gene_symbols = [
            annotation_index.get(gene_id, {"symbol": gene_id})["symbol"] for gene_id in seed_gene_ids
        ]
        for evidence_mode in EVIDENCE_MODES:
            query_text, query_template_id = build_query_text(
                task_type=prototype["task_type"],
                evidence_mode=evidence_mode,
                seed_gene_symbols=seed_gene_symbols,
            )
            visible_inputs = build_visible_inputs(
                evidence_mode=evidence_mode,
                task_type=prototype["task_type"],
                seed_gene_ids=seed_gene_ids,
                seed_gene_symbols=seed_gene_symbols,
                multiplex_flist=multiplex_flist,
                complex_row=complex_row,
            )
            if complex_row is None:
                hidden_target = {
                    "target_gene_ids": None,
                    "target_gene_symbols": None,
                    "relationship_status": NONE_RELATIONSHIP_STATUS,
                }
                mechanism_labels = None
                source_complex_ids = []
                source_pmid_ids = []
            else:
                hidden_target = {
                    "target_gene_ids": complex_row["gene_ids"],
                    "target_gene_symbols": complex_row["gene_symbols"],
                    "relationship_status": POSITIVE_RELATIONSHIP_STATUS,
                }
                mechanism_labels = complex_row["mechanism_labels"]
                source_complex_ids = complex_row["source_complex_ids"]
                source_pmid_ids = complex_row["pmids"]

            task_id = f"{prototype['prototype_id']}.{evidence_mode}"
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
                    "mechanism_labels": mechanism_labels,
                    "normalization": {
                        "gene_id_namespace": "ensembl.gene",
                        "visible_gene_mappings": [
                            {
                                "ensembl_gene_id": gene_id,
                                "display_symbol": annotation_index.get(gene_id, {"symbol": gene_id})["symbol"],
                                "resolved_via": annotation_index.get(gene_id, {"resolved_via": "unknown"})[
                                    "resolved_via"
                                ],
                            }
                            for gene_id in seed_gene_ids
                        ],
                    },
                    "provenance": {
                        "source": "CORUM",
                        "source_complex_record_id": prototype["target_complex_record_id"],
                        "source_complex_ids": source_complex_ids,
                        "source_pmid_ids": source_pmid_ids,
                        "matched_positive_prototype_id": prototype["matched_positive_prototype_id"],
                        "multiplex_flist": str(multiplex_flist),
                        "generation_seed": seed,
                    },
                }
            )

            if tracker and (len(tasks) % 250 == 0 or len(tasks) == total_task_count):
                tracker.update(
                    completed=len(tasks),
                    total=total_task_count,
                    unit="tasks",
                    message=(
                        f"Materialized {len(tasks)} of {total_task_count} canonical tasks "
                        f"across {prototype_index} of {len(prototypes)} prototypes."
                    ),
                    metrics={
                        "prototype_count": len(prototypes),
                        "tasks_materialized": len(tasks),
                    },
                )

    return sorted(tasks, key=lambda task: task["task_id"])


def build_split_report(
    complex_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    normalization_summary: dict[str, Any],
    dedup_summary: dict[str, Any],
) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in complex_rows)
    size_histograms: dict[str, dict[str, int]] = {split: Counter() for split in SPLITS}
    task_counts: dict[str, dict[str, int]] = {split: Counter() for split in SPLITS}
    gene_to_symbol = {}
    split_to_genes: dict[str, set[str]] = {split: set() for split in SPLITS}
    gene_split_counts: dict[str, Counter] = defaultdict(Counter)

    for complex_row in complex_rows:
        size_histograms[complex_row["split"]][complex_row["size_bin"]] += 1
        for gene_id, gene_symbol in zip(complex_row["gene_ids"], complex_row["gene_symbols"]):
            gene_to_symbol.setdefault(gene_id, gene_symbol)
            split_to_genes[complex_row["split"]].add(gene_id)
            gene_split_counts[gene_id][complex_row["split"]] += 1

    for task in tasks:
        task_counts[task["split"]][task["task_type"]] += 1

    cross_split_shared_genes = {}
    for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
        shared_gene_ids = sorted(split_to_genes[left_split] & split_to_genes[right_split])
        cross_split_shared_genes[f"{left_split}__{right_split}"] = {
            "shared_gene_count": len(shared_gene_ids),
            "top_shared_genes": [
                {
                    "ensembl_gene_id": gene_id,
                    "symbol": gene_to_symbol.get(gene_id, gene_id),
                    "complex_count_by_split": dict(gene_split_counts[gene_id]),
                }
                for gene_id in shared_gene_ids[:25]
            ],
        }

    ranked_shared_genes = sorted(
        (
            {
                "ensembl_gene_id": gene_id,
                "symbol": gene_to_symbol.get(gene_id, gene_id),
                "complex_count_by_split": dict(split_counts),
                "split_memberships": sorted(split_counts.keys()),
                "total_complex_count": sum(split_counts.values()),
            }
            for gene_id, split_counts in gene_split_counts.items()
            if len(split_counts) > 1
        ),
        key=lambda entry: (-entry["total_complex_count"], entry["ensembl_gene_id"]),
    )

    return {
        "normalization_summary": normalization_summary,
        "dedup_summary": dedup_summary,
        "split_counts": dict(split_counts),
        "split_size_histograms": {
            split: dict(size_histograms[split]) for split in SPLITS
        },
        "task_counts_by_split_and_type": {
            split: dict(task_counts[split]) for split in SPLITS
        },
        "cross_split_shared_genes": cross_split_shared_genes,
        "top_shared_genes_across_splits": ranked_shared_genes[:50],
    }


def build_manifest(
    *,
    corum_path: Path,
    multiplex_flist: Path,
    out_dir: Path,
    cache_dir: Path,
    seed: int,
    min_complex_size: int,
    complex_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    normalization_summary: dict[str, Any],
    dedup_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "corum-corpus-v1",
        "corum_path": str(corum_path),
        "multiplex_flist": str(multiplex_flist),
        "out_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "seed": seed,
        "min_complex_size": min_complex_size,
        "complex_count": len(complex_rows),
        "task_count": len(tasks),
        "task_count_by_split": dict(Counter(task["split"] for task in tasks)),
        "task_count_by_type": dict(Counter(task["task_type"] for task in tasks)),
        "normalization_summary": {
            key: value
            for key, value in normalization_summary.items()
            if key != "unresolved_examples"
        },
        "dedup_summary": dedup_summary,
    }


def build_complex_jsonl_rows(complex_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for complex_row in complex_rows:
        rows.append(
            {
                "complex_record_id": complex_row["complex_record_id"],
                "split": complex_row["split"],
                "source": "CORUM",
                "source_complex_ids": complex_row["source_complex_ids"],
                "source_complex_names": complex_row["source_complex_names"],
                "duplicate_source_complex_ids": complex_row["duplicate_source_complex_ids"],
                "size": complex_row["size"],
                "size_bin": complex_row["size_bin"],
                "gene_ids": complex_row["gene_ids"],
                "gene_symbols": complex_row["gene_symbols"],
                "mechanism_labels": complex_row["mechanism_labels"],
                "synonyms": complex_row["synonyms"],
                "pmids": complex_row["pmids"],
                "comments": complex_row["comments"],
                "normalization": {
                    "gene_id_namespace": "ensembl.gene",
                    "member_mappings": [
                        {
                            "member_index": member["member_index"],
                            "source_symbol": member["source_symbol"],
                            "source_uniprot_id": member["source_uniprot_id"],
                            "source_aliases": member["source_aliases"],
                            "ensembl_gene_id": member["ensembl_gene_id"],
                            "display_symbol": member["display_symbol"],
                            "resolved_via": member["resolved_via"],
                            "matched_query": member["matched_query"],
                            "hit_ids": member["hit_ids"],
                        }
                        for member in complex_row["member_mappings"]
                    ],
                },
                "provenance": {
                    "source_file": "corum_humanComplexes.txt",
                    "organism": complex_row["organism"],
                    "cell_line": complex_row["cell_line"],
                },
            }
        )
    return rows


def build_corum_corpus(
    *,
    corum_path: Path,
    multiplex_flist: Path,
    out_dir: Path,
    seed: int,
    min_complex_size: int,
    cache_dir: Path,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_path or (out_dir / "progress.json")
    tracker = ProgressTracker(progress_path, CORPUS_BUILD_STAGES)
    tracker.set_context(
        {
            "corum_path": str(corum_path),
            "multiplex_flist": str(multiplex_flist),
            "out_dir": str(out_dir),
            "cache_dir": str(cache_dir),
            "seed": seed,
            "min_complex_size": min_complex_size,
        }
    )

    try:
        tracker.start_stage("parse_corum", unit="complexes")
        parsed_complexes = load_corum_complexes(corum_path)
        tracker.update(
            completed=len(parsed_complexes),
            total=len(parsed_complexes),
            unit="complexes",
            message=f"Parsed {len(parsed_complexes)} CORUM complexes.",
            metrics={"parsed_complexes": len(parsed_complexes)},
        )

        tracker.start_stage("load_multiplex_gene_universe")
        multiplex_gene_universe = load_multiplex_gene_universe(
            multiplex_flist,
            cache_dir,
            tracker=tracker,
        )

        resolver = MyGeneResolver(cache_dir=cache_dir, multiplex_genes=multiplex_gene_universe)
        prefetch_terms = collect_prefetch_terms(parsed_complexes)
        prefetch_total = sum(len(prefetch_terms[scope]) for scope in ("symbol", "uniprot", "alias"))

        tracker.start_stage(
            "prefetch_mygene_terms",
            total=prefetch_total,
            unit="queries",
            metrics={"prefetch_scope": None},
        )
        prefetch_completed = 0
        for scope in ("symbol", "uniprot", "alias"):
            scope_queries = sorted(prefetch_terms[scope])
            resolver.prefetch(
                scope,
                scope_queries,
                tracker=tracker,
                base_completed=prefetch_completed,
                total_queries=prefetch_total,
            )
            prefetch_completed += len(scope_queries)
        tracker.update(
            completed=prefetch_total,
            total=prefetch_total,
            unit="queries",
            message=f"Prefetched {prefetch_total} MyGene lookup terms.",
            metrics={
                "prefetch_scope": "complete",
                "prefetch_total": prefetch_total,
            },
        )

        tracker.start_stage(
            "normalize_complexes",
            total=len(parsed_complexes),
            unit="complexes",
        )
        normalized_complexes, normalization_summary = normalize_complexes(
            parsed_complexes=parsed_complexes,
            resolver=resolver,
            min_complex_size=min_complex_size,
            tracker=tracker,
        )

        tracker.start_stage("deduplicate_complexes", unit="complexes")
        deduplicated_complexes, dedup_summary = deduplicate_normalized_complexes(normalized_complexes)
        tracker.update(
            completed=len(deduplicated_complexes),
            total=len(deduplicated_complexes),
            unit="complexes",
            message=(
                f"Deduplicated {len(normalized_complexes)} normalized complexes down to "
                f"{len(deduplicated_complexes)} retained records."
            ),
            metrics=dedup_summary,
        )

        tracker.start_stage("assign_splits", total=len(deduplicated_complexes), unit="complexes")
        split_complexes = assign_splits(deduplicated_complexes, seed=seed)
        tracker.update(
            completed=len(split_complexes),
            total=len(split_complexes),
            unit="complexes",
            message=f"Assigned splits for {len(split_complexes)} retained complexes.",
            metrics={"split_complex_count": len(split_complexes)},
        )

        tracker.start_stage("build_task_prototypes")
        prototypes = build_task_prototypes(
            complex_rows=split_complexes,
            multiplex_gene_universe=multiplex_gene_universe,
            seed=seed,
            tracker=tracker,
        )

        tracker.start_stage("build_gene_annotation_index")
        annotation_index = build_gene_annotation_index(
            complex_rows=split_complexes,
            prototypes=prototypes,
            resolver=resolver,
            tracker=tracker,
        )

        tracker.start_stage(
            "materialize_tasks",
            total=len(prototypes) * len(EVIDENCE_MODES),
            unit="tasks",
        )
        tasks = materialize_tasks(
            complex_rows=split_complexes,
            prototypes=prototypes,
            annotation_index=annotation_index,
            multiplex_flist=multiplex_flist,
            seed=seed,
            tracker=tracker,
        )

        tracker.start_stage("build_split_report", unit="reports")
        split_report = build_split_report(
            complex_rows=split_complexes,
            tasks=tasks,
            normalization_summary=normalization_summary,
            dedup_summary=dedup_summary,
        )
        tracker.update(
            completed=1,
            total=1,
            unit="reports",
            message="Built split report.",
        )

        tracker.start_stage("build_manifest", unit="manifests")
        manifest = build_manifest(
            corum_path=corum_path,
            multiplex_flist=multiplex_flist,
            out_dir=out_dir,
            cache_dir=cache_dir,
            seed=seed,
            min_complex_size=min_complex_size,
            complex_rows=split_complexes,
            tasks=tasks,
            normalization_summary=normalization_summary,
            dedup_summary=dedup_summary,
        )
        tracker.update(
            completed=1,
            total=1,
            unit="manifests",
            message="Built manifest.",
            metrics={
                "complex_count": manifest["complex_count"],
                "task_count": manifest["task_count"],
            },
        )

        tracker.start_stage("write_outputs", total=3 + len(SPLITS), unit="files")
        complex_rows_for_output = build_complex_jsonl_rows(split_complexes)
        write_json(out_dir / "manifest.json", manifest)
        tracker.update(completed=1, total=3 + len(SPLITS), unit="files", message="Wrote manifest.json.")
        write_json(out_dir / "split_report.json", split_report)
        tracker.update(completed=2, total=3 + len(SPLITS), unit="files", message="Wrote split_report.json.")
        write_jsonl(out_dir / "complexes.jsonl", complex_rows_for_output)
        tracker.update(completed=3, total=3 + len(SPLITS), unit="files", message="Wrote complexes.jsonl.")
        for index, split in enumerate(SPLITS, start=1):
            split_tasks = [task for task in tasks if task["split"] == split]
            write_jsonl(out_dir / f"tasks.{split}.jsonl", split_tasks)
            tracker.update(
                completed=3 + index,
                total=3 + len(SPLITS),
                unit="files",
                message=f"Wrote tasks.{split}.jsonl.",
                metrics={"last_written_split": split},
            )

        tracker.complete(
            metrics={
                "complex_count": manifest["complex_count"],
                "task_count": manifest["task_count"],
                "progress_path": str(progress_path),
            }
        )
        return {
            "manifest": manifest,
            "split_report": split_report,
            "complex_rows": complex_rows_for_output,
            "tasks": tasks,
            "progress_path": progress_path,
        }
    except Exception as error:
        tracker.fail(error)
        raise


def main() -> None:
    args = parse_args()
    result = build_corum_corpus(
        corum_path=args.corum_path,
        multiplex_flist=args.multiplex_flist,
        out_dir=args.out_dir,
        seed=args.seed,
        min_complex_size=args.min_complex_size,
        cache_dir=args.cache_dir,
        progress_path=args.progress_path,
    )
    print(
        json.dumps(
            {
                "complex_count": result["manifest"]["complex_count"],
                "progress_path": str(result["progress_path"]),
                "task_count": result["manifest"]["task_count"],
                "task_count_by_split": result["manifest"]["task_count_by_split"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
