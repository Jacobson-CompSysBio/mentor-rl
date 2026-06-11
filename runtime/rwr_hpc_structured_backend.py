"""Structured, model-facing backend for RWR-HPC tools.

This module keeps RWR-HPC logically in-memory from the model's perspective:
the model provides biological arguments, while this backend handles temporary
files, app invocation, parsing, caching, and provenance.
"""

from __future__ import annotations

import csv
import math
import time
import tempfile
from pathlib import Path
from typing import Any

from .rwr_hpc_cache import (
    RwrHpcCache,
    file_sha256,
    make_rwr_hpc_cache_key,
    make_rwr_loe_cache_key,
)
from .rwr_hpc_requests import (
    ComponentSummaryRequest,
    GeneLayersRequest,
    LayerAblationRequest,
    LayerStatsRequest,
    NodePerturbationRequest,
    PathLayerCountsRequest,
    RwrDistanceRequest,
    RwrDotSimilarityRequest,
    RwrEncodingSummaryRequest,
    RwrLoeRequest,
    RwrPearsonRequest,
    RwrRankRequest,
    RwrRankVectorSummaryRequest,
    RwrRequest,
    SeedEssentialityRequest,
    RwrSpearmanRequest,
    ShortestPathsRequest,
)
from .tools import ToolExecutionError, ToolExecutionResult


_GENE_COLUMNS = (
    "gene",
    "Gene",
    "GENE",
    "node",
    "Node",
    "NODE",
    "NodeNames",
    "node_name",
    "gene_id",
    "GeneID",
)

_SCORE_COLUMNS = (
    "score",
    "Score",
    "SCORE",
    "scores",
    "Scores",
    "rwr_score",
    "RWR_score",
    "RWRScore",
    "RWR",
    "probability",
    "Probability",
)

_RANK_COLUMNS = (
    "rank",
    "Rank",
    "RANK",
    "rerank",
    "meanrank",
    "mean_rank",
)


def _first_existing_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def parse_rwr_loe_ranks(
    ranks_path: str | Path,
    *,
    seed_genes: tuple[str, ...] = (),
    query_genes: tuple[str, ...] = (),
    top_k: int | None = 20,
    exclude_seed_genes: bool = True,
) -> list[dict[str, Any]]:
    """Parse an RWR_LOE ranks TSV into a stable schema.

    The exact RWR-HPC output column names may vary across app versions, so this
    parser accepts several common gene/score/rank column spellings.
    """

    path = Path(ranks_path)
    if not path.exists():
        raise ToolExecutionError(f"RWR_LOE ranks file does not exist: {path}")

    seed_set = {gene.upper() for gene in seed_genes}
    query_set = {gene.upper() for gene in query_genes}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])

        if not fieldnames:
            raise ToolExecutionError(f"RWR_LOE ranks file has no header: {path}")

        gene_col = _first_existing_column(fieldnames, _GENE_COLUMNS)
        score_col = _first_existing_column(fieldnames, _SCORE_COLUMNS)
        rank_col = _first_existing_column(fieldnames, _RANK_COLUMNS)

        if gene_col is None:
            raise ToolExecutionError(
                "Could not find a gene column in RWR_LOE ranks file. "
                f"Available columns: {fieldnames}"
            )

        parsed: list[dict[str, Any]] = []

        for row_index, row in enumerate(reader, start=1):
            raw_gene = row.get(gene_col)
            if raw_gene is None:
                continue

            gene = str(raw_gene).strip()
            if not gene:
                continue

            gene_key = gene.upper()
            is_seed = gene_key in seed_set
            is_query = gene_key in query_set

            if exclude_seed_genes and is_seed:
                continue

            rank = _parse_int(row.get(rank_col)) if rank_col is not None else None
            if rank is None:
                rank = row_index

            score = _parse_float(row.get(score_col)) if score_col is not None else None

            parsed.append(
                {
                    "gene": gene_key,
                    "rank": rank,
                    "score": score,
                    "is_seed": is_seed,
                    "is_query": is_query,
                    "raw": dict(row),
                }
            )

    parsed.sort(key=lambda item: item["rank"])

    if top_k is not None:
        parsed = parsed[:top_k]

    return parsed


def parse_rwr_ranks(
    ranks_path: str | Path,
    *,
    seed_genes: tuple[str, ...] = (),
    top_k: int | None = 20,
) -> list[dict[str, Any]]:
    """Parse the RWR app ranks matrix into a ranked gene list.

    The RWR app records ranks as a transposed matrix: the first row contains
    node labels, and each following row corresponds to one seed set. For the
    model-facing single seed-set request, one following row is expected. If
    multiple rows exist, the best rank per gene is used.
    """

    path = Path(ranks_path)
    if not path.exists():
        raise ToolExecutionError(f"RWR ranks file does not exist: {path}")

    lines = [line.rstrip("\n").split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ToolExecutionError(f"RWR ranks file is empty: {path}")

    header = lines[0]
    if header and header[0] == "INDEX":
        gene_labels = header[1:]
        value_rows = lines[1:]
    else:
        gene_labels = header
        value_rows = lines[1:]

    if not gene_labels or not value_rows:
        return []

    seed_set = {gene.upper() for gene in seed_genes}
    best_by_gene: dict[str, dict[str, Any]] = {}

    for row in value_rows:
        row_label = row[0] if len(row) == len(gene_labels) + 1 else None
        values = row[1:] if row_label is not None else row
        for gene, value in zip(gene_labels, values):
            gene_key = str(gene).strip().upper()
            rank = _parse_float(value)
            if not gene_key or rank is None:
                continue
            item = best_by_gene.get(gene_key)
            if item is None or rank < item["rank"]:
                best_by_gene[gene_key] = {
                    "gene": gene_key,
                    "gene_id": gene_key,
                    "rank": rank,
                    "is_seed": gene_key in seed_set,
                    "seed_set": row_label,
                }

    ranked = sorted(best_by_gene.values(), key=lambda item: item["rank"])
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def parse_rwr_distance_matrix(
    distance_matrix_path: str | Path,
    *,
    gene_a: str,
    gene_b: str,
) -> float:
    """Parse the RWR app lower-triangle distance matrix for one gene pair."""

    if gene_a.upper() == gene_b.upper():
        return 0.0

    path = Path(distance_matrix_path)
    if not path.exists():
        raise ToolExecutionError(f"RWR distance matrix file does not exist: {path}")

    lines = [line.rstrip("\n").split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ToolExecutionError(f"RWR distance matrix file is empty: {path}")

    header = lines[0]
    if not header or header[0] != "INDEX":
        raise ToolExecutionError(f"RWR distance matrix file has no INDEX header: {path}")

    col_labels = [label.strip().upper() for label in header[1:]]
    gene_a_key = gene_a.upper()
    gene_b_key = gene_b.upper()

    for row in lines[1:]:
        if not row:
            continue
        row_label = row[0].strip().upper()
        for col_label, raw_value in zip(col_labels, row[1:]):
            if {row_label, col_label} != {gene_a_key, gene_b_key}:
                continue
            text = str(raw_value).strip()
            if not text or text.upper() == "NA":
                continue
            value = _parse_float(text)
            if value is None or not math.isfinite(value):
                raise ToolExecutionError(
                    f"RWR distance matrix value for {gene_a_key}/{gene_b_key} is not finite: {text}"
                )
            return value

    raise ToolExecutionError(
        f"Could not find a finite RWR distance value for {gene_a_key}/{gene_b_key} in {path}"
    )


def parse_shortest_paths(
    paths_path: str | Path,
    *,
    max_paths: int | None = 20,
) -> list[dict[str, Any]]:
    """Parse shortest_paths edge rows into path-level records."""

    rows = _parse_tsv_rows(paths_path)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        path_name = row.get("pathname") or f"{row.get('from', '')}_{row.get('to', '')}"
        path_elements = [
            part for part in str(row.get("pathelements", "")).split("->") if part
        ]
        path_length = _parse_int(row.get("pathlength"))
        item = grouped.setdefault(
            path_name,
            {
                "path_name": path_name,
                "source_gene": path_elements[0] if path_elements else row.get("from"),
                "target_gene": path_elements[-1] if path_elements else row.get("to"),
                "path_genes": path_elements,
                "path_length": path_length,
                "edges": [],
                "layers": set(),
            },
        )
        layer = row.get("type")
        if layer:
            item["layers"].add(layer)
        item["edges"].append(
            {
                "from": row.get("from"),
                "to": row.get("to"),
                "weight": _parse_float(row.get("weight")),
                "layer": layer,
            }
        )

    paths = []
    for item in grouped.values():
        item["layers"] = sorted(item["layers"])
        paths.append(item)

    paths.sort(
        key=lambda item: (
            item["path_length"] if item["path_length"] is not None else 10**9,
            item["path_name"],
        )
    )
    if max_paths is not None:
        paths = paths[:max_paths]
    return paths


def parse_path_layer_counts(
    layer_counts_path: str | Path,
    *,
    top_k: int | None = 20,
) -> list[dict[str, Any]]:
    """Parse shortest_paths layer-count output."""

    path = Path(layer_counts_path)
    if not path.exists():
        raise ToolExecutionError(f"shortest_paths layer-count file does not exist: {path}")

    counts: list[dict[str, Any]] = []
    for row in path.read_text(encoding="utf-8").splitlines():
        parts = row.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        count = _parse_int(parts[1])
        if count is None:
            continue
        counts.append({"layer": parts[0], "edge_count": count})

    counts.sort(key=lambda item: (-item["edge_count"], item["layer"]))
    if top_k is not None:
        counts = counts[:top_k]
    return counts


def _parse_labeled_matrix(path: str | Path) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Parse an RWR++ row/column-labeled matrix TSV."""

    matrix_path = Path(path)
    if not matrix_path.exists():
        raise ToolExecutionError(f"RWR++ matrix file does not exist: {matrix_path}")

    lines = [
        line.rstrip("\n").split("\t")
        for line in matrix_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ToolExecutionError(f"RWR++ matrix file is empty: {matrix_path}")

    header = lines[0]
    if header and header[0] == "INDEX":
        col_labels = header[1:]
        data_lines = lines[1:]
        row_labels = [row[0] for row in data_lines if row]
        values = [
            [_parse_float(value) for value in row[1 : 1 + len(col_labels)]]
            for row in data_lines
            if row
        ]
        return row_labels, col_labels, values

    raise ToolExecutionError(f"RWR++ matrix file lacks an INDEX header: {matrix_path}")


def parse_gene_layers(
    nodes_by_layer_path: str | Path,
    *,
    gene: str,
) -> dict[str, Any] | None:
    """Parse a nodes-by-layer matrix and return one gene's layer membership."""

    row_labels, col_labels, values = _parse_labeled_matrix(nodes_by_layer_path)
    gene_key = gene.upper()
    true_tokens = {"1", "1.0", "true", "TRUE", "True"}
    for row_label, row_values in zip(row_labels, values):
        if row_label.upper() != gene_key:
            continue
        raw_line = [
            str(value) if value is not None else ""
            for value in row_values
        ]
        present_layers = [
            layer
            for layer, value, raw_value in zip(col_labels, row_values, raw_line)
            if value not in {None, 0.0} or raw_value in true_tokens
        ]
        return {
            "gene": gene_key,
            "layers": present_layers,
            "layer_count": len(present_layers),
        }
    return None


def parse_layer_stats(
    network_stats_path: str | Path,
    *,
    top_k: int | None = 50,
    sort_by: str = "edge_count",
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Parse gene_layer_map network statistics."""

    path = Path(network_stats_path)
    if not path.exists():
        raise ToolExecutionError(f"gene_layer_map network stats file does not exist: {path}")

    stats: list[dict[str, Any]] = []
    for row in path.read_text(encoding="utf-8").splitlines():
        parts = row.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        node_count = _parse_int(parts[1])
        edge_count = _parse_int(parts[2])
        if node_count is None or edge_count is None:
            continue
        stats.append(
            {
                "layer": parts[0],
                "node_count": node_count,
                "edge_count": edge_count,
            }
        )

    stats.sort(key=lambda item: item[sort_by], reverse=descending)
    if top_k is not None:
        stats = stats[:top_k]
    return stats


def parse_component_summary(
    output_dir: str | Path,
    *,
    genes: tuple[str, ...] = (),
    max_components: int | None = 20,
) -> dict[str, Any]:
    """Parse disconnected_components component seed files."""

    root = Path(output_dir)
    component_files = sorted(root.rglob("*_comp*_seeds.tsv"))
    components: list[dict[str, Any]] = []
    gene_membership: dict[str, str] = {}
    requested_genes = {gene.upper() for gene in genes}

    for component_file in component_files:
        members: list[str] = []
        component_id = component_file.stem.replace("_seeds", "")
        for row in component_file.read_text(encoding="utf-8").splitlines():
            parts = row.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            component_id = parts[0] or component_id
            gene = parts[1].strip().upper()
            if not gene:
                continue
            members.append(gene)
            if not requested_genes or gene in requested_genes:
                gene_membership[gene] = component_id
        components.append(
            {
                "component_id": component_id,
                "size": len(members),
                "member_preview": members[:25],
            }
        )

    components.sort(key=lambda item: (-item["size"], item["component_id"]))
    total_components = len(components)
    if max_components is not None:
        components = components[:max_components]
    return {
        "total_components": total_components,
        "components": components,
        "gene_membership": gene_membership,
    }


def parse_seed_essentiality(
    gene_ranks_path: str | Path,
    null_ranks_path: str | Path,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Parse GRIN leave-one-out ranks and null-rank medians."""

    gene_rows, gene_cols, gene_values = _parse_labeled_matrix(gene_ranks_path)
    null_rows, _null_cols, null_values = _parse_labeled_matrix(null_ranks_path)

    rank_col_index = gene_cols.index("rank") if "rank" in gene_cols else 0
    null_rank_by_position: dict[int, float] = {}
    for row_label, row_values in zip(null_rows, null_values):
        pos_text = row_label.replace("null_rank_pos", "")
        try:
            pos = int(pos_text)
        except ValueError:
            continue
        if row_values and row_values[0] is not None:
            null_rank_by_position[pos] = float(row_values[0])

    results: list[dict[str, Any]] = []
    for index, (gene, row_values) in enumerate(zip(gene_rows, gene_values)):
        observed_rank = row_values[rank_col_index] if rank_col_index < len(row_values) else None
        if observed_rank is None:
            continue
        null_median_rank = null_rank_by_position.get(index)
        essentiality_delta = (
            null_median_rank - observed_rank
            if null_median_rank is not None
            else None
        )
        results.append(
            {
                "gene": gene.upper(),
                "leave_one_out_rank": observed_rank,
                "null_median_rank": null_median_rank,
                "essentiality_delta": essentiality_delta,
            }
        )

    results.sort(
        key=lambda item: (
            -(item["essentiality_delta"] if item["essentiality_delta"] is not None else float("-inf")),
            item["leave_one_out_rank"],
            item["gene"],
        )
    )
    if top_k is not None:
        results = results[:top_k]
    return results


def parse_rectangular_effect_matrix(
    matrix_path: str | Path,
    *,
    target_key: str,
    top_k: int | None = 20,
) -> list[dict[str, Any]]:
    """Summarize a row-labeled, column-target RWR++ effect matrix."""

    row_labels, col_labels, values = _parse_labeled_matrix(matrix_path)
    summaries: list[dict[str, Any]] = []

    for col_index, col_label in enumerate(col_labels):
        per_seed: list[dict[str, Any]] = []
        numeric_values: list[float] = []
        for row_label, row_values in zip(row_labels, values):
            value = row_values[col_index] if col_index < len(row_values) else None
            if value is None or not math.isfinite(value):
                continue
            numeric_values.append(float(value))
            per_seed.append({"seed_gene": row_label.upper(), "distance": float(value)})
        if not numeric_values:
            continue
        summaries.append(
            {
                target_key: col_label,
                "mean_distance": sum(numeric_values) / len(numeric_values),
                "max_distance": max(numeric_values),
                "min_distance": min(numeric_values),
                "seed_count": len(numeric_values),
                "per_seed": per_seed[:25],
            }
        )

    summaries.sort(key=lambda item: (-item["mean_distance"], item[target_key]))
    if top_k is not None:
        summaries = summaries[:top_k]
    return summaries


def parse_rwr_matrix_summary(
    matrix_path: str | Path,
    *,
    seed_genes: tuple[str, ...] = (),
    top_k: int | None = 20,
    include_seed_genes: bool = True,
    value_key: str = "score",
    lower_is_better: bool = False,
) -> list[dict[str, Any]]:
    """Parse RWR recorded rank/encoding matrices into compact top entries."""

    row_labels, col_labels, values = _parse_labeled_matrix(matrix_path)
    seed_set = {gene.upper() for gene in seed_genes}
    items: list[dict[str, Any]] = []

    for seed_label, row_values in zip(row_labels, values):
        seed_key = seed_label.upper()
        row_items: list[dict[str, Any]] = []
        for gene, value in zip(col_labels, row_values):
            if value is None or not math.isfinite(value):
                continue
            gene_key = gene.upper()
            is_seed = gene_key in seed_set
            if is_seed and not include_seed_genes:
                continue
            row_items.append(
                {
                    "seed_gene": seed_key,
                    "gene": gene_key,
                    value_key: float(value),
                    "is_seed": is_seed,
                }
            )
        row_items.sort(
            key=lambda item: (
                item[value_key] if lower_is_better else -item[value_key],
                item["gene"],
            )
        )
        if top_k is not None:
            row_items = row_items[:top_k]
        items.extend(row_items)

    return items


def _write_gene_set_file(path: Path, genes: tuple[str, ...]) -> None:
    path.write_text("\t".join(genes) + "\n", encoding="utf-8")


def _write_gene_set_rows_file(path: Path, gene_sets: tuple[tuple[str, ...], ...]) -> None:
    lines = ["\t".join(genes) for genes in gene_sets]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gene_list_file(path: Path, genes: tuple[str, ...]) -> None:
    path.write_text("\n".join(genes) + "\n", encoding="utf-8")


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique_candidates = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)
    return unique_candidates


def _rwr_loe_rank_candidates(output_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(sorted(output_dir.rglob("*.ranks.tsv")))
    candidates.extend(sorted(output_dir.rglob("*ranks*.tsv")))
    candidates.extend(sorted(output_dir.rglob("*.tsv")))
    return _unique_paths(candidates)


def _find_rwr_loe_ranks_file(output_dir: Path) -> Path:
    """Find the rank file emitted by RWR_LOE.

    Lustre metadata visibility can lag immediately after a subprocess exits, so
    retry briefly before declaring the app output missing.
    """

    for attempt in range(20):
        unique_candidates = _rwr_loe_rank_candidates(output_dir)
        if unique_candidates:
            return unique_candidates[0]
        if attempt < 19:
            time.sleep(0.25)
    raise ToolExecutionError(f"No RWR_LOE ranks TSV file found under {output_dir}")


def _find_required_file(output_dir: Path, patterns: tuple[str, ...], *, label: str) -> Path:
    for attempt in range(20):
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(sorted(output_dir.rglob(pattern)))
        unique_candidates = _unique_paths(candidates)
        if unique_candidates:
            return unique_candidates[0]
        if attempt < 19:
            time.sleep(0.25)
    raise ToolExecutionError(f"No {label} file found under {output_dir}")


def _find_rwr_distance_matrix_file(output_dir: Path, *, distance_metric: str) -> Path:
    return _find_required_file(
        output_dir,
        (
            f"*_{distance_metric}_dist_matrix.tsv",
            "*dist_matrix.tsv",
        ),
        label=f"RWR {distance_metric} distance matrix TSV",
    )


def _find_rwr_recorded_matrix_file(
    output_dir: Path,
    *,
    kind: str,
) -> Path:
    return _find_required_file(
        output_dir,
        (
            f"*_{kind}.tsv",
            f"*{kind}*.tsv",
        ),
        label=f"RWR {kind} TSV",
    )


def _filtered_flist_for_layers(source_flist: Path, output_flist: Path, layers: tuple[str, ...]) -> Path:
    """Write an flist containing only selected layers and return its path."""

    if not layers:
        return source_flist

    requested = set(layers)
    found: set[str] = set()
    kept_lines: list[str] = []

    for raw_line in source_flist.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        layer_name = parts[1].strip()
        if layer_name in requested:
            found.add(layer_name)
            kept_lines.append(line)

    missing = sorted(requested - found)
    if missing:
        raise ToolExecutionError(
            "Requested RWR layers were not present in the RWR-HPC flist: "
            + ", ".join(missing)
            + "."
        )

    output_flist.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return output_flist


class RwrHpcStructuredBackend:
    """Structured backend for model-facing RWR-HPC tools."""

    def __init__(
        self,
        *,
        flist: str | Path,
        app_backend: Any | None,
        cache: RwrHpcCache | None = None,
        scratch_root: str | Path | None = None,
        rwr_hpc_build_id: str = "unknown",
        timeout_seconds: int = 1800,
        no_edgelist_headers: bool = True,
    ) -> None:
        self.flist = Path(flist).resolve()
        if not self.flist.exists():
            raise FileNotFoundError(f"RWR-HPC flist does not exist: {self.flist}")

        self.app_backend = app_backend
        self.cache = cache
        self.scratch_root = Path(scratch_root).resolve() if scratch_root else None
        self.rwr_hpc_build_id = rwr_hpc_build_id
        self.timeout_seconds = timeout_seconds
        self.no_edgelist_headers = no_edgelist_headers

        if self.scratch_root is not None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)

    def _cache_context(
        self,
        tool_name: str,
        request_payload: dict[str, Any],
        *,
        version: int = 1,
    ) -> tuple[dict[str, Any], str, str]:
        payload = dict(request_payload)
        payload["no_edgelist_headers"] = self.no_edgelist_headers
        payload["structured_tool_version"] = version
        network_flist_sha256 = file_sha256(self.flist)
        cache_key = make_rwr_hpc_cache_key(
            tool_name=tool_name,
            request_payload=payload,
            network_flist_sha256=network_flist_sha256,
            rwr_hpc_build_id=self.rwr_hpc_build_id,
        )
        return payload, network_flist_sha256, cache_key

    def _cached_tool_result(
        self,
        tool_name: str,
        cache_key: str,
        *,
        empty_payload_key: str,
    ) -> ToolExecutionResult | None:
        if self.cache is None:
            return None
        cached = self.cache.get(tool_name, cache_key)
        if cached is None:
            return None
        provenance = dict(cached["provenance"])
        provenance["cache_hit"] = True
        provenance["cache_key"] = cache_key
        payload = cached["payload"]
        return ToolExecutionResult(
            payload=payload,
            provenance=provenance,
            is_empty=not bool(payload.get(empty_payload_key)),
        )

    def _require_app_backend(self) -> None:
        if self.app_backend is None:
            raise ToolExecutionError(
                "RWR-HPC structured backend has no app backend and no cached result."
            )

    def _run_app_or_raise(
        self,
        tool_name: str,
        app_args: list[str],
    ) -> Any:
        self._require_app_backend()
        app_result = self.app_backend.run_app(
            tool_name,
            app_args,
            timeout_seconds=self.timeout_seconds,
        )
        if app_result.returncode != 0:
            text = app_result.stderr or app_result.stdout or ""
            preview = text[:1000]
            raise ToolExecutionError(
                f"{tool_name} app failed with return code {app_result.returncode}. "
                f"Output: {preview}"
            )
        return app_result

    def run_rwr_loe(self, request: RwrLoeRequest) -> ToolExecutionResult:
        """Run structured RWR_LOE using cache first, then app fallback."""

        request_payload = request.cache_key_payload()
        request_payload["no_edgelist_headers"] = self.no_edgelist_headers
        network_flist_sha256 = file_sha256(self.flist)

        cache_key = make_rwr_loe_cache_key(
            request_payload=request_payload,
            network_flist_sha256=network_flist_sha256,
            rwr_hpc_build_id=self.rwr_hpc_build_id,
        )

        if self.cache is not None:
            cached = self.cache.get("rwr_loe", cache_key)
            if cached is not None:
                provenance = dict(cached["provenance"])
                provenance["cache_hit"] = True
                provenance["cache_key"] = cache_key

                payload = cached["payload"]
                return ToolExecutionResult(
                    payload=payload,
                    provenance=provenance,
                    is_empty=not bool(payload.get("ranked_genes")),
                )

        if self.app_backend is None:
            raise ToolExecutionError(
                "RWR-HPC structured backend has no app backend and no cached result."
            )

        result = self._run_rwr_loe_app_fallback(
            request=request,
            request_payload=request_payload,
            cache_key=cache_key,
            network_flist_sha256=network_flist_sha256,
        )

        return result

    def _run_rwr_loe_app_fallback(
        self,
        *,
        request: RwrLoeRequest,
        request_payload: dict[str, Any],
        cache_key: str,
        network_flist_sha256: str,
    ) -> ToolExecutionResult:
        """Run RWR_LOE via the standalone app using hidden scratch files."""

        temp_dir_kwargs: dict[str, Any] = {
            "prefix": "rwr_loe_",
        }
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)

        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            seed_file = scratch_dir / "seed_genes.txt"
            query_file = scratch_dir / "query_genes.txt"
            output_dir = scratch_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            _write_gene_set_file(seed_file, request.seed_genes)

            query_file_arg: str | None = None
            if request.query_genes:
                _write_gene_set_file(query_file, request.query_genes)
                query_file_arg = str(query_file)

            app_args = self._build_rwr_loe_app_args(
                seed_file=str(seed_file),
                query_file=query_file_arg,
                output_dir=str(output_dir),
                request=request,
            )

            app_result = self.app_backend.run_app(
                "rwr_loe",
                app_args,
                timeout_seconds=self.timeout_seconds,
            )

            if app_result.returncode != 0:
                text = app_result.stderr or app_result.stdout or ""
                preview = text[:1000]
                raise ToolExecutionError(
                    f"RWR_LOE app failed with return code {app_result.returncode}. "
                    f"Output: {preview}"
                )

            ranks_file = _find_rwr_loe_ranks_file(output_dir)

            ranked_genes = parse_rwr_loe_ranks(
                ranks_file,
                seed_genes=request.seed_genes,
                query_genes=request.query_genes,
                top_k=request.top_k,
                exclude_seed_genes=request.exclude_seed_genes,
            )

            payload = {
                "tool_name": "rwr_loe",
                "seed_genes": list(request.seed_genes),
                "query_genes": list(request.query_genes),
                "top_k": request.top_k,
                "restart": request.restart,
                "delta": request.delta,
                "reduction_method": request.reduction_method,
                "threshold": request.threshold,
                "exclude_seed_genes": request.exclude_seed_genes,
                "ranked_genes": ranked_genes,
                "results": ranked_genes,
            }

            provenance = {
                "backend": "rwr_hpc_app",
                "implementation": "structured_app_fallback",
                "cache_hit": False,
                "cache_key": cache_key,
                "network_flist_sha256": network_flist_sha256,
                "rwr_hpc_build_id": self.rwr_hpc_build_id,
                "app_returncode": app_result.returncode,
            }

            if self.cache is not None:
                self.cache.put(
                    "rwr_loe",
                    cache_key,
                    request=request_payload,
                    payload=payload,
                    provenance=provenance,
                    raw_stdout=app_result.stdout,
                    raw_stderr=app_result.stderr,
                )

            return ToolExecutionResult(
                payload=payload,
                provenance=provenance,
                is_empty=not bool(ranked_genes),
            )

    def _build_rwr_loe_app_args(
        self,
        *,
        seed_file: str,
        query_file: str | None,
        output_dir: str,
        request: RwrLoeRequest,
    ) -> list[str]:
        """Build CLI args for the app fallback.

        Before the real Frontier smoke test, compare these flags against
        docs/rwr_hpc_help/rwr_loe.txt and adjust names if needed.
        """

        args = [
            "--flist",
            str(self.flist),
            "--seed_file",
            seed_file,
            "--no_set_ids",
            "--output_dir",
            output_dir,
            "--restart",
            str(request.restart),
            "--delta",
            str(request.delta),
            "--reduction_method",
            request.reduction_method,
            "--threshold",
            str(request.threshold),
            "--run_tag",
            "mentor_rwr_loe",
        ]

        if self.no_edgelist_headers:
            args.append("--no_edgelist_headers")

        if query_file is not None:
            args.extend(["--query_file", query_file])

        return args

    def run_rwr(self, request: RwrRequest) -> ToolExecutionResult:
        """Run structured RWR using cache first, then app fallback."""

        request_payload = request.cache_key_payload()
        request_payload["no_edgelist_headers"] = self.no_edgelist_headers
        network_flist_sha256 = file_sha256(self.flist)

        cache_key = make_rwr_hpc_cache_key(
            tool_name="rwr",
            request_payload=request_payload,
            network_flist_sha256=network_flist_sha256,
            rwr_hpc_build_id=self.rwr_hpc_build_id,
        )

        if self.cache is not None:
            cached = self.cache.get("rwr", cache_key)
            if cached is not None:
                provenance = dict(cached["provenance"])
                provenance["cache_hit"] = True
                provenance["cache_key"] = cache_key

                payload = cached["payload"]
                return ToolExecutionResult(
                    payload=payload,
                    provenance=provenance,
                    is_empty=not bool(payload.get("ranked_genes")),
                )

        if self.app_backend is None:
            raise ToolExecutionError(
                "RWR-HPC structured backend has no app backend and no cached result."
            )

        return self._run_rwr_app_fallback(
            request=request,
            request_payload=request_payload,
            cache_key=cache_key,
            network_flist_sha256=network_flist_sha256,
        )

    def _run_rwr_app_fallback(
        self,
        *,
        request: RwrRequest,
        request_payload: dict[str, Any],
        cache_key: str,
        network_flist_sha256: str,
    ) -> ToolExecutionResult:
        """Run RWR via the standalone app using hidden scratch files."""

        temp_dir_kwargs: dict[str, Any] = {"prefix": "rwr_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)

        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            seed_file = scratch_dir / "seed_genes.txt"
            filtered_flist = scratch_dir / "network.flist"
            output_dir = scratch_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            _write_gene_set_file(seed_file, request.seed_genes)
            flist = _filtered_flist_for_layers(self.flist, filtered_flist, request.layers)

            app_args = self._build_rwr_app_args(
                flist=str(flist),
                seed_file=str(seed_file),
                output_dir=str(output_dir),
                request=request,
            )

            app_result = self.app_backend.run_app(
                "rwr",
                app_args,
                timeout_seconds=self.timeout_seconds,
            )

            if app_result.returncode != 0:
                text = app_result.stderr or app_result.stdout or ""
                preview = text[:1000]
                raise ToolExecutionError(
                    f"RWR app failed with return code {app_result.returncode}. "
                    f"Output: {preview}"
                )

            ranks_file = _find_required_file(
                output_dir,
                ("*_ranks.tsv", "*ranks*.tsv"),
                label="RWR ranks TSV",
            )

            ranked_genes = parse_rwr_ranks(
                ranks_file,
                seed_genes=request.seed_genes,
                top_k=request.top_k,
            )

            payload = {
                "tool_name": "rwr",
                "seed_genes": list(request.seed_genes),
                "seed_gene_ids": list(request.seed_genes),
                "layers": list(request.layers),
                "top_k": request.top_k,
                "restart": request.restart,
                "delta": request.delta,
                "reduction_method": request.reduction_method,
                "threshold": request.threshold,
                "ranked_genes": ranked_genes,
                "results": ranked_genes,
            }

            provenance = {
                "backend": "rwr_hpc_app",
                "implementation": "structured_app_fallback",
                "cache_hit": False,
                "cache_key": cache_key,
                "network_flist_sha256": network_flist_sha256,
                "rwr_hpc_build_id": self.rwr_hpc_build_id,
                "requested_layers": list(request.layers),
                "app_returncode": app_result.returncode,
            }

            if self.cache is not None:
                self.cache.put(
                    "rwr",
                    cache_key,
                    request=request_payload,
                    payload=payload,
                    provenance=provenance,
                    raw_stdout=app_result.stdout,
                    raw_stderr=app_result.stderr,
                )

            return ToolExecutionResult(
                payload=payload,
                provenance=provenance,
                is_empty=not bool(ranked_genes),
            )

    def _build_rwr_app_args(
        self,
        *,
        flist: str,
        seed_file: str,
        output_dir: str,
        request: RwrRequest,
    ) -> list[str]:
        args = [
            "--flist",
            flist,
            "--seed_file",
            seed_file,
            "--no_set_ids",
            "--runtag",
            "mentor_rwr",
            "--output_dir",
            output_dir,
            "--restart",
            str(request.restart),
            "--delta",
            str(request.delta),
            "--reduction_method",
            request.reduction_method,
            "--threshold",
            str(request.threshold),
            "--record_ranks",
        ]
        if self.no_edgelist_headers:
            args.append("--no_edgelist_headers")
        return args

    def run_get_rank(self, request: RwrRankRequest) -> ToolExecutionResult:
        """Return the target gene rank in a single-source RWR vector."""

        request_payload = request.cache_key_payload()
        request_payload["no_edgelist_headers"] = self.no_edgelist_headers
        request_payload["derived_tool_version"] = 1
        network_flist_sha256 = file_sha256(self.flist)

        cache_key = make_rwr_hpc_cache_key(
            tool_name="get_rank",
            request_payload=request_payload,
            network_flist_sha256=network_flist_sha256,
            rwr_hpc_build_id=self.rwr_hpc_build_id,
        )

        if self.cache is not None:
            cached = self.cache.get("get_rank", cache_key)
            if cached is not None:
                provenance = dict(cached["provenance"])
                provenance["cache_hit"] = True
                provenance["cache_key"] = cache_key
                payload = cached["payload"]
                return ToolExecutionResult(
                    payload=payload,
                    provenance=provenance,
                    is_empty=not bool(payload.get("rank_result")),
                )

        rwr_result = self.run_rwr(
            RwrRequest(
                seed_genes=(request.source_gene,),
                layers=request.layers,
                top_k=None,
                restart=request.restart,
                delta=request.delta,
                reduction_method=request.reduction_method,
                threshold=request.threshold,
            )
        )
        ranked_genes = rwr_result.payload.get("ranked_genes", [])
        ranked_list = ranked_genes if isinstance(ranked_genes, list) else []
        target_key = request.target_gene.upper()
        rank_result = next(
            (
                dict(item)
                for item in ranked_list
                if isinstance(item, dict)
                and str(item.get("gene") or item.get("gene_id") or "").upper() == target_key
            ),
            None,
        )

        payload = {
            "tool_name": "get_rank",
            "source_gene": request.source_gene,
            "target_gene": request.target_gene,
            "layers": list(request.layers),
            "restart": request.restart,
            "delta": request.delta,
            "reduction_method": request.reduction_method,
            "threshold": request.threshold,
            "rank_semantics": (
                "single-source RWR rank of target_gene in source_gene's restart vector; "
                "this is not LoE"
            ),
            "rank_result": rank_result,
            "target_rank": rank_result.get("rank") if rank_result else None,
            "ranked_gene_count": len(ranked_list),
            "results": [rank_result] if rank_result else [],
        }

        provenance = {
            "backend": rwr_result.provenance.get("backend", "rwr_hpc_app"),
            "implementation": "derived_from_rwr_rank_matrix",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "source_rwr_cache_hit": rwr_result.provenance.get("cache_hit"),
            "source_rwr_cache_key": rwr_result.provenance.get("cache_key"),
        }

        if self.cache is not None:
            self.cache.put(
                "get_rank",
                cache_key,
                request=request_payload,
                payload=payload,
                provenance=provenance,
            )

        return ToolExecutionResult(
            payload=payload,
            provenance=provenance,
            is_empty=rank_result is None,
        )

    def run_get_distance(self, request: RwrDistanceRequest) -> ToolExecutionResult:
        """Return the RWR++ distance/dissimilarity between two one-gene seed vectors."""

        request_payload = request.cache_key_payload()
        request_payload["no_edgelist_headers"] = self.no_edgelist_headers
        request_payload["derived_tool_version"] = 1
        network_flist_sha256 = file_sha256(self.flist)

        cache_key = make_rwr_hpc_cache_key(
            tool_name="get_distance",
            request_payload=request_payload,
            network_flist_sha256=network_flist_sha256,
            rwr_hpc_build_id=self.rwr_hpc_build_id,
        )

        if self.cache is not None:
            cached = self.cache.get("get_distance", cache_key)
            if cached is not None:
                provenance = dict(cached["provenance"])
                provenance["cache_hit"] = True
                provenance["cache_key"] = cache_key
                payload = cached["payload"]
                return ToolExecutionResult(
                    payload=payload,
                    provenance=provenance,
                    is_empty=payload.get("distance") is None,
                )

        if request.gene_a == request.gene_b:
            payload = {
                "tool_name": "get_distance",
                "gene_a": request.gene_a,
                "gene_b": request.gene_b,
                "layers": list(request.layers),
                "distance_metric": request.distance_metric,
                "distance": 0.0,
                "dissimilarity": 0.0,
                "restart": request.restart,
                "delta": request.delta,
                "reduction_method": request.reduction_method,
                "threshold": request.threshold,
                "results": [
                    {
                        "gene_a": request.gene_a,
                        "gene_b": request.gene_b,
                        "distance_metric": request.distance_metric,
                        "distance": 0.0,
                    }
                ],
            }
            provenance = {
                "backend": "rwr_hpc_app",
                "implementation": "identity_distance_short_circuit",
                "cache_hit": False,
                "cache_key": cache_key,
                "network_flist_sha256": network_flist_sha256,
                "rwr_hpc_build_id": self.rwr_hpc_build_id,
            }
            if self.cache is not None:
                self.cache.put(
                    "get_distance",
                    cache_key,
                    request=request_payload,
                    payload=payload,
                    provenance=provenance,
                )
            return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=False)

        if self.app_backend is None:
            raise ToolExecutionError(
                "RWR-HPC structured backend has no app backend and no cached result."
            )

        return self._run_rwr_pair_distance_app_fallback(
            request=request,
            request_payload=request_payload,
            cache_key=cache_key,
            network_flist_sha256=network_flist_sha256,
        )

    def _run_rwr_pair_distance_app_fallback(
        self,
        *,
        request: RwrDistanceRequest,
        request_payload: dict[str, Any],
        cache_key: str,
        network_flist_sha256: str,
    ) -> ToolExecutionResult:
        """Run RWR++ with two one-gene seed vectors and parse the distance matrix."""

        temp_dir_kwargs: dict[str, Any] = {"prefix": "rwr_pair_distance_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)

        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            seed_file = scratch_dir / "seed_genes.txt"
            filtered_flist = scratch_dir / "network.flist"
            output_dir = scratch_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            _write_gene_set_rows_file(seed_file, ((request.gene_a,), (request.gene_b,)))
            flist = _filtered_flist_for_layers(self.flist, filtered_flist, request.layers)

            app_args = self._build_rwr_pair_distance_app_args(
                flist=str(flist),
                seed_file=str(seed_file),
                output_dir=str(output_dir),
                request=request,
            )
            app_result = self.app_backend.run_app(
                "rwr",
                app_args,
                timeout_seconds=self.timeout_seconds,
            )

            if app_result.returncode != 0:
                text = app_result.stderr or app_result.stdout or ""
                preview = text[:1000]
                raise ToolExecutionError(
                    f"RWR app failed with return code {app_result.returncode}. "
                    f"Output: {preview}"
                )

            distance_matrix_file = _find_rwr_distance_matrix_file(
                output_dir,
                distance_metric=request.distance_metric,
            )
            distance = parse_rwr_distance_matrix(
                distance_matrix_file,
                gene_a=request.gene_a,
                gene_b=request.gene_b,
            )

            payload = {
                "tool_name": "get_distance",
                "gene_a": request.gene_a,
                "gene_b": request.gene_b,
                "layers": list(request.layers),
                "distance_metric": request.distance_metric,
                "distance": distance,
                "dissimilarity": distance,
                "restart": request.restart,
                "delta": request.delta,
                "reduction_method": request.reduction_method,
                "threshold": request.threshold,
                "results": [
                    {
                        "gene_a": request.gene_a,
                        "gene_b": request.gene_b,
                        "distance_metric": request.distance_metric,
                        "distance": distance,
                    }
                ],
            }
            provenance = {
                "backend": "rwr_hpc_app",
                "implementation": "rwr_pair_distance_matrix",
                "cache_hit": False,
                "cache_key": cache_key,
                "network_flist_sha256": network_flist_sha256,
                "rwr_hpc_build_id": self.rwr_hpc_build_id,
                "requested_layers": list(request.layers),
                "distance_metric": request.distance_metric,
                "app_returncode": app_result.returncode,
            }

            if self.cache is not None:
                self.cache.put(
                    "get_distance",
                    cache_key,
                    request=request_payload,
                    payload=payload,
                    provenance=provenance,
                    raw_stdout=app_result.stdout,
                    raw_stderr=app_result.stderr,
                )

            return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=False)

    def _build_rwr_pair_distance_app_args(
        self,
        *,
        flist: str,
        seed_file: str,
        output_dir: str,
        request: RwrDistanceRequest,
    ) -> list[str]:
        args = [
            "--flist",
            flist,
            "--seed_file",
            seed_file,
            "--no_set_ids",
            "--runtag",
            "mentor_rwr_pair",
            "--output_dir",
            output_dir,
            "--restart",
            str(request.restart),
            "--delta",
            str(request.delta),
            "--reduction_method",
            request.reduction_method,
            "--threshold",
            str(request.threshold),
            "--distance_metric",
            request.distance_metric,
        ]
        if self.no_edgelist_headers:
            args.append("--no_edgelist_headers")
        return args

    def run_get_spearman(self, request: RwrSpearmanRequest) -> ToolExecutionResult:
        """Return Spearman correlation between two RWR rank/encoding vectors."""

        request_payload = request.cache_key_payload()
        request_payload["no_edgelist_headers"] = self.no_edgelist_headers
        request_payload["derived_tool_version"] = 1
        network_flist_sha256 = file_sha256(self.flist)

        cache_key = make_rwr_hpc_cache_key(
            tool_name="get_spearman",
            request_payload=request_payload,
            network_flist_sha256=network_flist_sha256,
            rwr_hpc_build_id=self.rwr_hpc_build_id,
        )

        if self.cache is not None:
            cached = self.cache.get("get_spearman", cache_key)
            if cached is not None:
                provenance = dict(cached["provenance"])
                provenance["cache_hit"] = True
                provenance["cache_key"] = cache_key
                payload = cached["payload"]
                return ToolExecutionResult(
                    payload=payload,
                    provenance=provenance,
                    is_empty=payload.get("spearman_correlation") is None,
                )

        distance_result = self.run_get_distance(request.to_distance_request())
        distance = distance_result.payload.get("distance")
        correlation = 1.0 - float(distance) if distance is not None else None

        payload = {
            "tool_name": "get_spearman",
            "gene_a": request.gene_a,
            "gene_b": request.gene_b,
            "layers": list(request.layers),
            "spearman_correlation": correlation,
            "spearman_distance": distance,
            "restart": request.restart,
            "delta": request.delta,
            "reduction_method": request.reduction_method,
            "threshold": request.threshold,
            "results": [
                {
                    "gene_a": request.gene_a,
                    "gene_b": request.gene_b,
                    "spearman_correlation": correlation,
                    "spearman_distance": distance,
                }
            ] if correlation is not None else [],
        }
        provenance = {
            "backend": distance_result.provenance.get("backend", "rwr_hpc_app"),
            "implementation": "derived_from_rwr_spearman_distance",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "distance_cache_hit": distance_result.provenance.get("cache_hit"),
            "distance_cache_key": distance_result.provenance.get("cache_key"),
        }

        if self.cache is not None:
            self.cache.put(
                "get_spearman",
                cache_key,
                request=request_payload,
                payload=payload,
                provenance=provenance,
            )

        return ToolExecutionResult(
            payload=payload,
            provenance=provenance,
            is_empty=correlation is None,
        )

    def run_get_pearson(self, request: RwrPearsonRequest) -> ToolExecutionResult:
        """Return Pearson correlation between two RWR encoding vectors."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_pearson",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_pearson", cache_key, empty_payload_key="results")
        if cached is not None:
            return cached

        distance_result = self.run_get_distance(request.to_distance_request())
        distance = distance_result.payload.get("distance")
        correlation = 1.0 - float(distance) if distance is not None else None

        payload = {
            "tool_name": "get_pearson",
            "gene_a": request.gene_a,
            "gene_b": request.gene_b,
            "layers": list(request.layers),
            "pearson_correlation": correlation,
            "pearson_distance": distance,
            "restart": request.restart,
            "delta": request.delta,
            "reduction_method": request.reduction_method,
            "threshold": request.threshold,
            "results": [
                {
                    "gene_a": request.gene_a,
                    "gene_b": request.gene_b,
                    "pearson_correlation": correlation,
                    "pearson_distance": distance,
                }
            ] if correlation is not None else [],
        }
        provenance = {
            "backend": distance_result.provenance.get("backend", "rwr_hpc_app"),
            "implementation": "derived_from_rwr_pearson_distance",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "distance_cache_hit": distance_result.provenance.get("cache_hit"),
            "distance_cache_key": distance_result.provenance.get("cache_key"),
        }
        if self.cache is not None:
            self.cache.put("get_pearson", cache_key, request=request_payload, payload=payload, provenance=provenance)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=correlation is None)

    def run_get_dot_similarity(self, request: RwrDotSimilarityRequest) -> ToolExecutionResult:
        """Return RWR++ dot similarity between two one-gene seed vectors."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_dot_similarity",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_dot_similarity", cache_key, empty_payload_key="results")
        if cached is not None:
            return cached

        if request.gene_a == request.gene_b:
            similarity: float | None = 1.0
            source_cache_hit = None
            source_cache_key = None
        else:
            distance_result = self.run_get_distance(request.to_distance_request())
            similarity = distance_result.payload.get("distance")
            source_cache_hit = distance_result.provenance.get("cache_hit")
            source_cache_key = distance_result.provenance.get("cache_key")

        payload = {
            "tool_name": "get_dot_similarity",
            "gene_a": request.gene_a,
            "gene_b": request.gene_b,
            "layers": list(request.layers),
            "dot_similarity": similarity,
            "restart": request.restart,
            "delta": request.delta,
            "reduction_method": request.reduction_method,
            "threshold": request.threshold,
            "results": [
                {
                    "gene_a": request.gene_a,
                    "gene_b": request.gene_b,
                    "dot_similarity": similarity,
                }
            ] if similarity is not None else [],
        }
        provenance = {
            "backend": "rwr_hpc_app",
            "implementation": "derived_from_rwr_dot_matrix",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "distance_cache_hit": source_cache_hit,
            "distance_cache_key": source_cache_key,
        }
        if self.cache is not None:
            self.cache.put("get_dot_similarity", cache_key, request=request_payload, payload=payload, provenance=provenance)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=similarity is None)

    def run_get_rank_vector_summary(self, request: RwrRankVectorSummaryRequest) -> ToolExecutionResult:
        """Return a compact summary of RWR rank vectors."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_rank_vector_summary",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_rank_vector_summary", cache_key, empty_payload_key="rank_summary")
        if cached is not None:
            return cached

        rwr_result = self.run_rwr(
            RwrRequest(
                seed_genes=request.seed_genes,
                layers=request.layers,
                top_k=None,
                restart=request.restart,
                delta=request.delta,
                reduction_method=request.reduction_method,
                threshold=request.threshold,
            )
        )
        seed_set = set(request.seed_genes)
        ranked_list = [
            dict(item)
            for item in rwr_result.payload.get("ranked_genes", [])
            if isinstance(item, dict)
            and (request.include_seed_genes or str(item.get("gene") or item.get("gene_id") or "").upper() not in seed_set)
        ]
        ranked_list.sort(key=lambda item: (item.get("rank", float("inf")), item.get("gene", "")))
        if request.top_k is not None:
            ranked_list = ranked_list[: request.top_k]
        payload = {
            "tool_name": "get_rank_vector_summary",
            "seed_genes": list(request.seed_genes),
            "layers": list(request.layers),
            "top_k": request.top_k,
            "include_seed_genes": request.include_seed_genes,
            "rank_summary": ranked_list,
            "results": ranked_list,
        }
        provenance = {
            "backend": rwr_result.provenance.get("backend", "rwr_hpc_app"),
            "implementation": "derived_from_rwr_rank_matrix",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "source_rwr_cache_hit": rwr_result.provenance.get("cache_hit"),
            "source_rwr_cache_key": rwr_result.provenance.get("cache_key"),
        }
        if self.cache is not None:
            self.cache.put("get_rank_vector_summary", cache_key, request=request_payload, payload=payload, provenance=provenance)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=not bool(ranked_list))

    def run_get_encoding_summary(self, request: RwrEncodingSummaryRequest) -> ToolExecutionResult:
        """Run RWR++ with recorded encodings and return a compact top-score summary."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_encoding_summary",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_encoding_summary", cache_key, empty_payload_key="encoding_summary")
        if cached is not None:
            return cached

        temp_dir_kwargs: dict[str, Any] = {"prefix": "rwr_encoding_summary_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)

        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            seed_file = scratch_dir / "seed_genes.txt"
            filtered_flist = scratch_dir / "network.flist"
            output_dir = scratch_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_gene_set_file(seed_file, request.seed_genes)
            flist = _filtered_flist_for_layers(self.flist, filtered_flist, request.layers)

            app_args = self._build_rwr_encoding_summary_app_args(
                flist=str(flist),
                seed_file=str(seed_file),
                output_dir=str(output_dir),
                request=request,
            )
            app_result = self._run_app_or_raise("rwr", app_args)
            encodings_file = _find_rwr_recorded_matrix_file(output_dir, kind="encodings")
            encoding_summary = parse_rwr_matrix_summary(
                encodings_file,
                seed_genes=request.seed_genes,
                top_k=request.top_k,
                include_seed_genes=request.include_seed_genes,
                value_key="score",
                lower_is_better=False,
            )
        payload = {
            "tool_name": "get_encoding_summary",
            "seed_genes": list(request.seed_genes),
            "layers": list(request.layers),
            "top_k": request.top_k,
            "include_seed_genes": request.include_seed_genes,
            "encoding_summary": encoding_summary,
            "results": encoding_summary,
        }
        provenance = {
            "backend": "rwr_hpc_app",
            "implementation": "rwr_record_encodings_summary",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "app_returncode": app_result.returncode,
        }
        if self.cache is not None:
            self.cache.put(
                "get_encoding_summary",
                cache_key,
                request=request_payload,
                payload=payload,
                provenance=provenance,
                raw_stdout=app_result.stdout,
                raw_stderr=app_result.stderr,
            )
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=not bool(encoding_summary))

    def _build_rwr_encoding_summary_app_args(
        self,
        *,
        flist: str,
        seed_file: str,
        output_dir: str,
        request: RwrEncodingSummaryRequest,
    ) -> list[str]:
        args = [
            "--flist",
            flist,
            "--seed_file",
            seed_file,
            "--no_set_ids",
            "--runtag",
            "mentor_rwr_encoding",
            "--output_dir",
            output_dir,
            "--restart",
            str(request.restart),
            "--delta",
            str(request.delta),
            "--reduction_method",
            request.reduction_method,
            "--threshold",
            str(request.threshold),
            "--record_encodings",
        ]
        if self.no_edgelist_headers:
            args.append("--no_edgelist_headers")
        return args

    def run_get_gene_layers(self, request: GeneLayersRequest, *, tool_name: str = "get_gene_layers") -> ToolExecutionResult:
        """Return the multiplex layers that contain one gene."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            tool_name,
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result(tool_name, cache_key, empty_payload_key="layers")
        if cached is not None:
            return cached

        result, app_result = self._run_gene_layer_map_app(request=request)
        payload = {
            "tool_name": tool_name,
            "gene": request.gene,
            "layers": result["layers"] if result else [],
            "layer_count": result["layer_count"] if result else 0,
            "results": [result] if result else [],
        }
        provenance = {
            "backend": "rwr_hpc_app",
            "implementation": "gene_layer_map",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "app_returncode": app_result.returncode,
        }
        if self.cache is not None:
            self.cache.put(tool_name, cache_key, request=request_payload, payload=payload, provenance=provenance, raw_stdout=app_result.stdout, raw_stderr=app_result.stderr)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=result is None)

    def run_get_nodes_by_layer(self, request: GeneLayersRequest) -> ToolExecutionResult:
        """Alias for get_gene_layers using RWR++ nodes-by-layer output."""

        return self.run_get_gene_layers(request, tool_name="get_nodes_by_layer")

    def run_get_layer_stats(self, request: LayerStatsRequest) -> ToolExecutionResult:
        """Return compact layer node/edge statistics from gene_layer_map."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_layer_stats",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_layer_stats", cache_key, empty_payload_key="layer_stats")
        if cached is not None:
            return cached

        stats, app_result = self._run_layer_stats_app(request=request)
        payload = {
            "tool_name": "get_layer_stats",
            "top_k": request.top_k,
            "sort_by": request.sort_by,
            "descending": request.descending,
            "layer_stats": stats,
            "results": stats,
        }
        provenance = {
            "backend": "rwr_hpc_app",
            "implementation": "gene_layer_map_network_stats",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "app_returncode": app_result.returncode,
        }
        if self.cache is not None:
            self.cache.put("get_layer_stats", cache_key, request=request_payload, payload=payload, provenance=provenance, raw_stdout=app_result.stdout, raw_stderr=app_result.stderr)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=not bool(stats))

    def _run_gene_layer_map_app(self, *, request: GeneLayersRequest) -> tuple[dict[str, Any] | None, Any]:
        temp_dir_kwargs: dict[str, Any] = {"prefix": "gene_layer_map_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)
        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            output_dir = Path(tmp) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            app_args = self._build_gene_layer_map_args(output_dir=str(output_dir))
            app_result = self._run_app_or_raise("gene_layer_map", app_args)
            nodes_file = _find_required_file(
                output_dir,
                ("*nodes_by_layer.tsv",),
                label="gene_layer_map nodes-by-layer TSV",
            )
            return parse_gene_layers(nodes_file, gene=request.gene), app_result

    def _run_layer_stats_app(self, *, request: LayerStatsRequest) -> tuple[list[dict[str, Any]], Any]:
        temp_dir_kwargs: dict[str, Any] = {"prefix": "gene_layer_stats_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)
        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            output_dir = Path(tmp) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            app_args = self._build_gene_layer_map_args(output_dir=str(output_dir))
            app_result = self._run_app_or_raise("gene_layer_map", app_args)
            stats_file = _find_required_file(
                output_dir,
                ("*network_stats.tsv",),
                label="gene_layer_map network stats TSV",
            )
            return (
                parse_layer_stats(
                    stats_file,
                    top_k=request.top_k,
                    sort_by=request.sort_by,
                    descending=request.descending,
                ),
                app_result,
            )

    def _build_gene_layer_map_args(self, *, output_dir: str) -> list[str]:
        args = [
            "--flist",
            str(self.flist),
            "--output_dir",
            output_dir,
            "--runtag",
            "mentor_layer_map",
        ]
        if self.no_edgelist_headers:
            args.append("--no_edgelist_headers")
        return args

    def run_get_path_layer_counts(self, request: PathLayerCountsRequest) -> ToolExecutionResult:
        """Return layer support counts for RWR++ shortest paths."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_path_layer_counts",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_path_layer_counts", cache_key, empty_payload_key="layer_counts")
        if cached is not None:
            return cached

        shortest_result = self.run_shortest_paths(request.to_shortest_paths_request())
        layer_counts = shortest_result.payload.get("layer_counts")
        if not isinstance(layer_counts, list):
            edge_counts: dict[str, int] = {}
            for path in shortest_result.payload.get("paths", []):
                if not isinstance(path, dict):
                    continue
                for edge in path.get("edges", []):
                    if isinstance(edge, dict) and edge.get("layer"):
                        edge_counts[str(edge["layer"])] = edge_counts.get(str(edge["layer"]), 0) + 1
            layer_counts = [
                {"layer": layer, "edge_count": count}
                for layer, count in sorted(edge_counts.items(), key=lambda item: (-item[1], item[0]))
            ]
        if request.top_k is not None:
            layer_counts = layer_counts[: request.top_k]
        payload = {
            "tool_name": "get_path_layer_counts",
            "source_genes": list(request.source_genes),
            "target_genes": list(request.target_genes),
            "merge_method": request.merge_method,
            "ignore_weights": request.ignore_weights,
            "max_paths": request.max_paths,
            "top_k": request.top_k,
            "layer_counts": layer_counts,
            "results": layer_counts,
        }
        provenance = {
            "backend": shortest_result.provenance.get("backend", "rwr_hpc_app"),
            "implementation": "derived_from_shortest_paths_layer_counts",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "source_shortest_paths_cache_hit": shortest_result.provenance.get("cache_hit"),
            "source_shortest_paths_cache_key": shortest_result.provenance.get("cache_key"),
        }
        if self.cache is not None:
            self.cache.put("get_path_layer_counts", cache_key, request=request_payload, payload=payload, provenance=provenance)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=not bool(layer_counts))

    def run_get_component_summary(self, request: ComponentSummaryRequest) -> ToolExecutionResult:
        """Return connected-component summaries from disconnected_components."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_component_summary",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_component_summary", cache_key, empty_payload_key="components")
        if cached is not None:
            return cached

        temp_dir_kwargs: dict[str, Any] = {"prefix": "component_summary_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)
        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            output_dir = Path(tmp) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            app_args = [
                "--flist",
                str(self.flist),
                "--output_dir",
                str(output_dir),
                "--runtag",
                "mentor_components",
            ]
            if self.no_edgelist_headers:
                app_args.append("--no_edgelist_headers")
            app_result = self._run_app_or_raise("disconnected_components", app_args)
            summary = parse_component_summary(
                output_dir,
                genes=request.genes,
                max_components=request.max_components,
            )
        payload = {
            "tool_name": "get_component_summary",
            "genes": list(request.genes),
            "max_components": request.max_components,
            **summary,
            "results": summary["components"],
        }
        provenance = {
            "backend": "rwr_hpc_app",
            "implementation": "disconnected_components_summary",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "app_returncode": app_result.returncode,
        }
        if self.cache is not None:
            self.cache.put("get_component_summary", cache_key, request=request_payload, payload=payload, provenance=provenance, raw_stdout=app_result.stdout, raw_stderr=app_result.stderr)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=not bool(summary["components"]))

    def run_get_seed_essentiality(self, request: SeedEssentialityRequest) -> ToolExecutionResult:
        """Return GRIN leave-one-out seed essentiality summaries."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_seed_essentiality",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_seed_essentiality", cache_key, empty_payload_key="essentiality")
        if cached is not None:
            return cached

        temp_dir_kwargs: dict[str, Any] = {"prefix": "seed_essentiality_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)
        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            output_dir = scratch_dir / "output"
            temp_dir = scratch_dir / "temp"
            seed_file = scratch_dir / "seed_genes.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
            _write_gene_set_file(seed_file, request.seed_genes)
            app_args = [
                "--flist",
                str(self.flist),
                "--seed_file",
                str(seed_file),
                "--no_set_ids",
                "--runtag",
                "mentor_grin",
                "--output_dir",
                str(output_dir),
                "--temp_dir",
                str(temp_dir),
                "--restart",
                str(request.restart),
                "--delta",
                str(request.delta),
                "--reduction_method",
                request.reduction_method,
                "--threshold",
                str(request.threshold),
                "--n_samples_null_dist",
                str(request.n_samples_null_dist),
                "--seed",
                str(request.seed),
            ]
            if self.no_edgelist_headers:
                app_args.append("--no_edgelist_headers")
            app_result = self._run_app_or_raise("grin", app_args)
            gene_ranks_file = _find_required_file(output_dir, ("*_gene_ranks.tsv",), label="GRIN gene ranks TSV")
            null_ranks_file = _find_required_file(output_dir, ("*_null_ranks.tsv",), label="GRIN null ranks TSV")
            essentiality = parse_seed_essentiality(
                gene_ranks_file,
                null_ranks_file,
                top_k=request.top_k,
            )
        payload = {
            "tool_name": "get_seed_essentiality",
            "seed_genes": list(request.seed_genes),
            "n_samples_null_dist": request.n_samples_null_dist,
            "seed": request.seed,
            "top_k": request.top_k,
            "essentiality": essentiality,
            "results": essentiality,
        }
        provenance = {
            "backend": "rwr_hpc_app",
            "implementation": "grin_leave_one_out",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "app_returncode": app_result.returncode,
        }
        if self.cache is not None:
            self.cache.put("get_seed_essentiality", cache_key, request=request_payload, payload=payload, provenance=provenance, raw_stdout=app_result.stdout, raw_stderr=app_result.stderr)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=not bool(essentiality))

    def run_get_layer_ablation(self, request: LayerAblationRequest) -> ToolExecutionResult:
        """Return layer-ablation distance effects."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_layer_ablation",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_layer_ablation", cache_key, empty_payload_key="layer_effects")
        if cached is not None:
            return cached

        temp_dir_kwargs: dict[str, Any] = {"prefix": "layer_ablation_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)
        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            output_dir = scratch_dir / "output"
            seed_file = scratch_dir / "seed_genes.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_gene_set_file(seed_file, request.seed_genes)
            app_args = [
                "--flist",
                str(self.flist),
                "--seed_file",
                str(seed_file),
                "--no_set_ids",
                "--output_dir",
                str(output_dir) + "/",
                "--restart",
                str(request.restart),
                "--delta",
                str(request.delta),
                "--reduction_method",
                request.reduction_method,
                "--threshold",
                str(request.threshold),
                "--distance_metric",
                request.distance_metric,
            ]
            if self.no_edgelist_headers:
                app_args.append("--no_edgelist_headers")
            app_result = self._run_app_or_raise("rwr_ablation", app_args)
            matrix_file = _find_required_file(
                output_dir,
                (f"{request.distance_metric}_ablation_distance_matrix.tsv", "*ablation_distance_matrix.tsv"),
                label="RWR ablation distance matrix TSV",
            )
            layer_effects = parse_rectangular_effect_matrix(
                matrix_file,
                target_key="layer",
                top_k=request.top_k,
            )
        payload = {
            "tool_name": "get_layer_ablation",
            "seed_genes": list(request.seed_genes),
            "distance_metric": request.distance_metric,
            "top_k": request.top_k,
            "layer_effects": layer_effects,
            "results": layer_effects,
        }
        provenance = {
            "backend": "rwr_hpc_app",
            "implementation": "rwr_ablation_summary",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "app_returncode": app_result.returncode,
        }
        if self.cache is not None:
            self.cache.put("get_layer_ablation", cache_key, request=request_payload, payload=payload, provenance=provenance, raw_stdout=app_result.stdout, raw_stderr=app_result.stderr)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=not bool(layer_effects))

    def run_get_node_perturbation(self, request: NodePerturbationRequest) -> ToolExecutionResult:
        """Return node-perturbation distance effects."""

        request_payload, network_flist_sha256, cache_key = self._cache_context(
            "get_node_perturbation",
            request.cache_key_payload(),
        )
        cached = self._cached_tool_result("get_node_perturbation", cache_key, empty_payload_key="perturbation_effects")
        if cached is not None:
            return cached

        temp_dir_kwargs: dict[str, Any] = {"prefix": "node_perturbation_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)
        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            output_dir = scratch_dir / "output"
            seed_file = scratch_dir / "seed_genes.txt"
            perturb_file = scratch_dir / "perturb_genes.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_gene_set_file(seed_file, request.seed_genes)
            _write_gene_set_rows_file(perturb_file, tuple((gene,) for gene in request.perturb_genes))
            app_args = [
                "--flist",
                str(self.flist),
                "--seed_file",
                str(seed_file),
                "--pertubation_file",
                str(perturb_file),
                "--no_set_ids",
                "--output_dir",
                str(output_dir) + "/",
                "--restart",
                str(request.restart),
                "--delta",
                str(request.delta),
                "--reduction_method",
                request.reduction_method,
                "--threshold",
                str(request.threshold),
                "--distance_metric",
                request.distance_metric,
            ]
            if self.no_edgelist_headers:
                app_args.append("--no_edgelist_headers")
            app_result = self._run_app_or_raise("rwr_perturbation", app_args)
            matrix_file = _find_required_file(
                output_dir,
                (f"{request.distance_metric}_perturbation_distance_matrix.tsv", "*perturbation_distance_matrix.tsv"),
                label="RWR perturbation distance matrix TSV",
            )
            effects = parse_rectangular_effect_matrix(
                matrix_file,
                target_key="perturb_gene",
                top_k=request.top_k,
            )
        payload = {
            "tool_name": "get_node_perturbation",
            "seed_genes": list(request.seed_genes),
            "perturb_genes": list(request.perturb_genes),
            "distance_metric": request.distance_metric,
            "top_k": request.top_k,
            "perturbation_effects": effects,
            "results": effects,
        }
        provenance = {
            "backend": "rwr_hpc_app",
            "implementation": "rwr_perturbation_summary",
            "cache_hit": False,
            "cache_key": cache_key,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": self.rwr_hpc_build_id,
            "app_returncode": app_result.returncode,
        }
        if self.cache is not None:
            self.cache.put("get_node_perturbation", cache_key, request=request_payload, payload=payload, provenance=provenance, raw_stdout=app_result.stdout, raw_stderr=app_result.stderr)
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=not bool(effects))

    def run_shortest_paths(self, request: ShortestPathsRequest) -> ToolExecutionResult:
        """Run structured shortest_paths using cache first, then app fallback."""

        request_payload = request.cache_key_payload()
        request_payload["no_edgelist_headers"] = self.no_edgelist_headers
        network_flist_sha256 = file_sha256(self.flist)

        cache_key = make_rwr_hpc_cache_key(
            tool_name="shortest_paths",
            request_payload=request_payload,
            network_flist_sha256=network_flist_sha256,
            rwr_hpc_build_id=self.rwr_hpc_build_id,
        )

        if self.cache is not None:
            cached = self.cache.get("shortest_paths", cache_key)
            if cached is not None:
                provenance = dict(cached["provenance"])
                provenance["cache_hit"] = True
                provenance["cache_key"] = cache_key

                payload = cached["payload"]
                return ToolExecutionResult(
                    payload=payload,
                    provenance=provenance,
                    is_empty=not bool(payload.get("paths")),
                )

        if self.app_backend is None:
            raise ToolExecutionError(
                "RWR-HPC structured backend has no app backend and no cached result."
            )

        return self._run_shortest_paths_app_fallback(
            request=request,
            request_payload=request_payload,
            cache_key=cache_key,
            network_flist_sha256=network_flist_sha256,
        )

    def _run_shortest_paths_app_fallback(
        self,
        *,
        request: ShortestPathsRequest,
        request_payload: dict[str, Any],
        cache_key: str,
        network_flist_sha256: str,
    ) -> ToolExecutionResult:
        """Run shortest_paths via the standalone app using hidden scratch files."""

        temp_dir_kwargs: dict[str, Any] = {"prefix": "shortest_paths_"}
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)

        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            sources_file = scratch_dir / "source_genes.txt"
            targets_file = scratch_dir / "target_genes.txt"
            output_dir = scratch_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            _write_gene_list_file(sources_file, request.source_genes)

            targets_file_arg: str | None = None
            if request.target_genes:
                _write_gene_list_file(targets_file, request.target_genes)
                targets_file_arg = str(targets_file)

            app_args = self._build_shortest_paths_app_args(
                sources_file=str(sources_file),
                targets_file=targets_file_arg,
                output_dir=str(output_dir),
                request=request,
            )

            app_result = self.app_backend.run_app(
                "shortest_paths",
                app_args,
                timeout_seconds=self.timeout_seconds,
            )

            if app_result.returncode != 0:
                text = app_result.stderr or app_result.stdout or ""
                preview = text[:1000]
                raise ToolExecutionError(
                    f"shortest_paths app failed with return code {app_result.returncode}. "
                    f"Output: {preview}"
                )

            paths_file = _find_required_file(
                output_dir,
                ("*_shortest_paths.tsv", "*shortest_paths*.tsv"),
                label="shortest_paths TSV",
            )

            paths = parse_shortest_paths(paths_file, max_paths=request.max_paths)
            layer_counts_file = _find_required_file(
                output_dir,
                ("*_layer_counts.tsv", "*layer_counts*.tsv"),
                label="shortest_paths layer-count TSV",
            )
            layer_counts = parse_path_layer_counts(layer_counts_file, top_k=None)

            payload = {
                "tool_name": "shortest_paths",
                "source_genes": list(request.source_genes),
                "target_genes": list(request.target_genes),
                "merge_method": request.merge_method,
                "ignore_weights": request.ignore_weights,
                "max_paths": request.max_paths,
                "paths": paths,
                "layer_counts": layer_counts,
                "results": paths,
            }

            provenance = {
                "backend": "rwr_hpc_app",
                "implementation": "structured_app_fallback",
                "cache_hit": False,
                "cache_key": cache_key,
                "network_flist_sha256": network_flist_sha256,
                "rwr_hpc_build_id": self.rwr_hpc_build_id,
                "app_returncode": app_result.returncode,
            }

            if self.cache is not None:
                self.cache.put(
                    "shortest_paths",
                    cache_key,
                    request=request_payload,
                    payload=payload,
                    provenance=provenance,
                    raw_stdout=app_result.stdout,
                    raw_stderr=app_result.stderr,
                )

            return ToolExecutionResult(
                payload=payload,
                provenance=provenance,
                is_empty=not bool(paths),
            )

    def _build_shortest_paths_app_args(
        self,
        *,
        sources_file: str,
        targets_file: str | None,
        output_dir: str,
        request: ShortestPathsRequest,
    ) -> list[str]:
        args = [
            "--flist",
            str(self.flist),
            "--sources_file",
            sources_file,
            "--no_set_ids",
            "--merge_method",
            request.merge_method,
            "--output_dir",
            output_dir,
            "--runtag",
            "mentor_shortest_paths",
        ]

        if self.no_edgelist_headers:
            args.append("--no_edgelist_headers")

        if targets_file is not None:
            args.extend(["--targets_file", targets_file])
        if request.ignore_weights:
            args.append("--ignore_weights")

        return args
