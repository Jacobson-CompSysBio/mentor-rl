"""Build a binary CSR multiplex store from a text edgelist flist.

This script converts the raw HumanNet-style multiplex into a binary directory
store that is cheap for a compiled runtime to load:

- one global gene table
- one layer table
- raw CSR arrays for each layer
- one aggregate graph

The store is intentionally simple. It uses raw binary arrays plus a JSON
manifest so the C++ runtime can load it without pulling in a heavy Python
stack or large graph libraries.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy import sparse


STORE_FORMAT_VERSION = "mentor-rl-multiplex-store-v1"
WEIGHT_DTYPE = np.float32
INDEX_DTYPE = np.uint32
INDPTR_DTYPE = np.uint64
PROGRESS_STAGE_ORDER = (
    "parse_flist",
    "collect_gene_universe",
    "write_gene_table",
    "build_layers",
    "write_aggregate",
    "write_manifest",
)


@dataclass(frozen=True)
class FlistEntry:
    """One layer entry from the multiplex flist."""

    path: str
    layer_name: str
    enabled: bool = True


@dataclass
class BuildProgressTracker:
    """Persist simple progress updates for long multiplex-store builds."""

    path: Path
    stage_order: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    started_unix_time: float = field(default_factory=time.time)

    def update(
        self,
        stage: str,
        message: str,
        *,
        completed_in_stage: int | None = None,
        total_in_stage: int | None = None,
        metrics: dict[str, Any] | None = None,
        status: str = "running",
    ) -> None:
        """Write the current build stage and counters to disk."""

        stage_index = self.stage_order.index(stage)
        stage_count = len(self.stage_order)
        stage_progress = 0.0
        if total_in_stage not in (None, 0):
            stage_progress = max(
                0.0,
                min(1.0, float(completed_in_stage or 0) / float(total_in_stage)),
            )
        overall_progress = (stage_index + stage_progress) / float(stage_count)

        payload = {
            "status": status,
            "current_stage": stage,
            "stage_index": stage_index + 1,
            "stage_count": stage_count,
            "stage_progress": round(stage_progress, 6),
            "overall_progress": round(overall_progress, 6),
            "message": message,
            "metrics": metrics or {},
            "metadata": dict(self.metadata),
            "started_unix_time": int(self.started_unix_time),
            "updated_unix_time": int(time.time()),
        }
        if completed_in_stage is not None:
            payload["completed_in_stage"] = int(completed_in_stage)
        if total_in_stage is not None:
            payload["total_in_stage"] = int(total_in_stage)

        self._write(payload)

    def mark_completed(self, *, metrics: dict[str, Any] | None = None) -> None:
        """Mark the full build as complete."""

        self._write(
            {
                "status": "completed",
                "current_stage": "completed",
                "stage_index": len(self.stage_order),
                "stage_count": len(self.stage_order),
                "stage_progress": 1.0,
                "overall_progress": 1.0,
                "message": "Multiplex store build completed.",
                "metrics": metrics or {},
                "metadata": dict(self.metadata),
                "started_unix_time": int(self.started_unix_time),
                "updated_unix_time": int(time.time()),
            }
        )

    def mark_failed(self, error_message: str, *, stage: str, metrics: dict[str, Any] | None = None) -> None:
        """Mark the build as failed and persist the last known context."""

        stage_index = self.stage_order.index(stage)
        self._write(
            {
                "status": "failed",
                "current_stage": stage,
                "stage_index": stage_index + 1,
                "stage_count": len(self.stage_order),
                "stage_progress": 0.0,
                "overall_progress": round(stage_index / float(len(self.stage_order)), 6),
                "message": error_message,
                "metrics": metrics or {},
                "metadata": dict(self.metadata),
                "started_unix_time": int(self.started_unix_time),
                "updated_unix_time": int(time.time()),
            }
        )

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(self.path)


def parse_flist(flist_path: str, *, limit_layers: int | None = None) -> list[FlistEntry]:
    """Read the multiplex flist and return enabled layer entries."""

    entries: list[FlistEntry] = []
    with Path(flist_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError("Each flist row must have at least two tab-separated columns.")

            path = parts[0]
            layer_name = parts[1]
            enabled = True
            if len(parts) >= 3:
                enabled = parts[2].strip() not in {"0", "false", "False"}

            if enabled:
                entries.append(FlistEntry(path=path, layer_name=layer_name, enabled=enabled))

            if limit_layers is not None and len(entries) >= limit_layers:
                break

    if not entries:
        raise ValueError("No enabled layers were found in the flist.")
    return entries


def iter_weighted_edges(edgelist_path: str) -> Iterable[tuple[str, str, float]]:
    """Yield weighted undirected edges from one whitespace-delimited edgelist."""

    with Path(edgelist_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            source = parts[0]
            target = parts[1]
            if source == target:
                continue

            weight = 1.0
            if len(parts) >= 3:
                try:
                    weight = float(parts[2])
                except ValueError:
                    weight = 1.0
            yield source, target, weight


def collect_gene_universe(
    entries: list[FlistEntry],
    *,
    progress_callback: Callable[[int, int, FlistEntry, int], None] | None = None,
) -> list[str]:
    """Read all layers once and collect the global sorted gene universe."""

    genes: set[str] = set()
    total_entries = len(entries)
    for entry_index, entry in enumerate(entries, start=1):
        for source, target, _ in iter_weighted_edges(entry.path):
            genes.add(source)
            genes.add(target)
        if progress_callback is not None:
            progress_callback(entry_index, total_entries, entry, len(genes))

    if not genes:
        raise ValueError("No genes were found across the multiplex edgelists.")
    return sorted(genes)


def build_layer_csr(
    entry: FlistEntry,
    *,
    gene_to_index: dict[str, int],
    num_genes: int,
) -> sparse.csr_matrix:
    """Build one undirected CSR matrix for a single layer."""

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for source, target, weight in iter_weighted_edges(entry.path):
        source_index = gene_to_index[source]
        target_index = gene_to_index[target]
        rows.extend((source_index, target_index))
        cols.extend((target_index, source_index))
        data.extend((weight, weight))

    matrix = sparse.coo_matrix(
        (
            np.asarray(data, dtype=WEIGHT_DTYPE),
            (
                np.asarray(rows, dtype=INDEX_DTYPE),
                np.asarray(cols, dtype=INDEX_DTYPE),
            ),
        ),
        shape=(num_genes, num_genes),
        dtype=WEIGHT_DTYPE,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def write_binary_array(path: Path, array: np.ndarray) -> None:
    """Write one typed NumPy array as raw binary bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array)
    contiguous.tofile(path)


def write_gene_table(genes: list[str], out_dir: Path) -> Path:
    """Write the global gene vocabulary as a simple TSV lookup table."""

    path = out_dir / "genes.tsv"
    with path.open("w", encoding="utf-8") as handle:
        for index, gene_id in enumerate(genes):
            handle.write(f"{index}\t{gene_id}\n")
    return path


def write_layer_table(rows: list[dict[str, object]], out_dir: Path) -> Path:
    """Write the layer metadata table used by the compiled runtime loader."""

    path = out_dir / "layers.tsv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    "layer_index",
                    "layer_name",
                    "indptr_file",
                    "indices_file",
                    "weights_file",
                    "node_count",
                    "undirected_edge_count",
                    "stored_nnz",
                ]
            )
            + "\n"
        )
        for row in rows:
            handle.write(
                "\t".join(
                    str(row[key])
                    for key in [
                        "layer_index",
                        "layer_name",
                        "indptr_file",
                        "indices_file",
                        "weights_file",
                        "node_count",
                        "undirected_edge_count",
                        "stored_nnz",
                    ]
                )
                + "\n"
            )
    return path


def build_store(
    *,
    multiplex_flist: str,
    out_dir: str,
    limit_layers: int | None = None,
    progress_path: str | None = None,
) -> dict[str, object]:
    """Build the binary multiplex store and return the manifest payload."""

    start_time = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if progress_path is None:
        progress_path = str(out_path / "progress.json")

    progress = BuildProgressTracker(
        path=Path(progress_path),
        stage_order=PROGRESS_STAGE_ORDER,
        metadata={
            "source_flist": multiplex_flist,
            "out_dir": str(out_path),
            "limit_layers": limit_layers,
        },
    )
    current_stage = "parse_flist"

    try:
        progress.update(current_stage, "Reading multiplex flist.")
        entries = parse_flist(multiplex_flist, limit_layers=limit_layers)

        current_stage = "collect_gene_universe"
        progress.update(
            current_stage,
            "Scanning multiplex layers to collect the global gene universe.",
            completed_in_stage=0,
            total_in_stage=len(entries),
            metrics={"layers_total": len(entries), "unique_genes": 0},
        )

        genes = collect_gene_universe(
            entries,
            progress_callback=lambda completed, total, entry, unique_genes: progress.update(
                current_stage,
                f"Collected genes from layer {completed}/{total}: {entry.layer_name}",
                completed_in_stage=completed,
                total_in_stage=total,
                metrics={
                    "layers_total": total,
                    "layers_completed": completed,
                    "current_layer_name": entry.layer_name,
                    "current_layer_path": entry.path,
                    "unique_genes": unique_genes,
                },
            ),
        )
        gene_to_index = {gene_id: index for index, gene_id in enumerate(genes)}
        num_genes = len(genes)

        current_stage = "write_gene_table"
        progress.update(
            current_stage,
            "Writing global gene table.",
            completed_in_stage=0,
            total_in_stage=1,
            metrics={"num_genes": num_genes},
        )
        write_gene_table(genes, out_path)
        progress.update(
            current_stage,
            "Global gene table written.",
            completed_in_stage=1,
            total_in_stage=1,
            metrics={"num_genes": num_genes, "gene_table": "genes.tsv"},
        )

        current_stage = "build_layers"
        aggregate_csr: sparse.csr_matrix | None = None
        layer_rows: list[dict[str, object]] = []
        manifest_layers: list[dict[str, object]] = []
        total_layers = len(entries)
        progress.update(
            current_stage,
            "Building per-layer CSR arrays.",
            completed_in_stage=0,
            total_in_stage=total_layers,
            metrics={"layers_total": total_layers, "num_genes": num_genes},
        )

        for layer_index, entry in enumerate(entries):
            progress.update(
                current_stage,
                f"Building CSR for layer {layer_index + 1}/{total_layers}: {entry.layer_name}",
                completed_in_stage=layer_index,
                total_in_stage=total_layers,
                metrics={
                    "layers_total": total_layers,
                    "layers_completed": layer_index,
                    "current_layer_index": layer_index,
                    "current_layer_name": entry.layer_name,
                    "current_layer_path": entry.path,
                    "num_genes": num_genes,
                },
            )

            layer_csr = build_layer_csr(entry, gene_to_index=gene_to_index, num_genes=num_genes)
            if aggregate_csr is None:
                aggregate_csr = layer_csr.copy()
            else:
                aggregate_csr = (aggregate_csr + layer_csr).tocsr()
                aggregate_csr.sum_duplicates()
                aggregate_csr.sort_indices()

            indptr_file = f"layer_{layer_index:04d}_indptr.bin"
            indices_file = f"layer_{layer_index:04d}_indices.bin"
            weights_file = f"layer_{layer_index:04d}_weights.bin"

            write_binary_array(out_path / indptr_file, layer_csr.indptr.astype(INDPTR_DTYPE, copy=False))
            write_binary_array(out_path / indices_file, layer_csr.indices.astype(INDEX_DTYPE, copy=False))
            write_binary_array(out_path / weights_file, layer_csr.data.astype(WEIGHT_DTYPE, copy=False))

            node_count = int(np.count_nonzero(np.diff(layer_csr.indptr)))
            undirected_edge_count = int(layer_csr.nnz // 2)
            layer_row = {
                "layer_index": layer_index,
                "layer_name": entry.layer_name,
                "indptr_file": indptr_file,
                "indices_file": indices_file,
                "weights_file": weights_file,
                "node_count": node_count,
                "undirected_edge_count": undirected_edge_count,
                "stored_nnz": int(layer_csr.nnz),
            }
            layer_rows.append(layer_row)
            manifest_layers.append(layer_row.copy())
            progress.update(
                current_stage,
                f"Finished layer {layer_index + 1}/{total_layers}: {entry.layer_name}",
                completed_in_stage=layer_index + 1,
                total_in_stage=total_layers,
                metrics={
                    "layers_total": total_layers,
                    "layers_completed": layer_index + 1,
                    "current_layer_index": layer_index,
                    "current_layer_name": entry.layer_name,
                    "current_layer_path": entry.path,
                    "current_layer_edge_count": undirected_edge_count,
                    "current_layer_nnz": int(layer_csr.nnz),
                    "num_genes": num_genes,
                },
            )

        if aggregate_csr is None:
            raise ValueError("The aggregate graph could not be constructed.")

        current_stage = "write_aggregate"
        progress.update(
            current_stage,
            "Writing aggregate graph arrays.",
            completed_in_stage=0,
            total_in_stage=1,
            metrics={"num_genes": num_genes, "num_layers": len(entries)},
        )
        aggregate_csr.eliminate_zeros()
        aggregate_csr.sort_indices()
        write_binary_array(
            out_path / "aggregate_indptr.bin",
            aggregate_csr.indptr.astype(INDPTR_DTYPE, copy=False),
        )
        write_binary_array(
            out_path / "aggregate_indices.bin",
            aggregate_csr.indices.astype(INDEX_DTYPE, copy=False),
        )
        write_binary_array(
            out_path / "aggregate_weights.bin",
            aggregate_csr.data.astype(WEIGHT_DTYPE, copy=False),
        )

        write_layer_table(layer_rows, out_path)
        progress.update(
            current_stage,
            "Aggregate graph arrays and layer table written.",
            completed_in_stage=1,
            total_in_stage=1,
            metrics={
                "num_genes": num_genes,
                "num_layers": len(entries),
                "aggregate_stored_nnz": int(aggregate_csr.nnz),
                "aggregate_undirected_edge_count": int(aggregate_csr.nnz // 2),
            },
        )

        current_stage = "write_manifest"
        progress.update(
            current_stage,
            "Writing final manifest.",
            completed_in_stage=0,
            total_in_stage=1,
            metrics={"num_genes": num_genes, "num_layers": len(entries)},
        )
        manifest = {
            "format_version": STORE_FORMAT_VERSION,
            "store_type": "raw_csr_directory",
            "source_flist": multiplex_flist,
            "created_unix_time": int(time.time()),
            "num_genes": num_genes,
            "num_layers": len(entries),
            "dtypes": {
                "weights": str(np.dtype(WEIGHT_DTYPE)),
                "indices": str(np.dtype(INDEX_DTYPE)),
                "indptr": str(np.dtype(INDPTR_DTYPE)),
            },
            "files": {
                "gene_table": "genes.tsv",
                "layer_table": "layers.tsv",
                "aggregate": {
                    "indptr_file": "aggregate_indptr.bin",
                    "indices_file": "aggregate_indices.bin",
                    "weights_file": "aggregate_weights.bin",
                    "stored_nnz": int(aggregate_csr.nnz),
                    "undirected_edge_count": int(aggregate_csr.nnz // 2),
                },
            },
            "layers": manifest_layers,
            "build_time_seconds": round(time.time() - start_time, 3),
        }
        with (out_path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        progress.update(
            current_stage,
            "Manifest written.",
            completed_in_stage=1,
            total_in_stage=1,
            metrics={
                "num_genes": num_genes,
                "num_layers": len(entries),
                "build_time_seconds": manifest["build_time_seconds"],
            },
        )
        progress.mark_completed(
            metrics={
                "num_genes": num_genes,
                "num_layers": len(entries),
                "build_time_seconds": manifest["build_time_seconds"],
                "manifest_path": "manifest.json",
            }
        )
        return manifest
    except Exception as exc:
        progress.mark_failed(str(exc), stage=current_stage)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for store building."""

    parser = argparse.ArgumentParser(description="Build a binary multiplex CSR store.")
    parser.add_argument("--multiplex-flist", required=True, help="Path to the multiplex flist.")
    parser.add_argument("--out-dir", required=True, help="Directory where the store will be written.")
    parser.add_argument(
        "--limit-layers",
        type=int,
        default=None,
        help="Optional number of enabled layers to build. Useful for smoke tests.",
    )
    parser.add_argument(
        "--progress-path",
        default=None,
        help="Optional JSON file used to track build progress. Defaults to OUT_DIR/progress.json.",
    )
    return parser


def main() -> None:
    """Command-line entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    manifest = build_store(
        multiplex_flist=args.multiplex_flist,
        out_dir=args.out_dir,
        limit_layers=args.limit_layers,
        progress_path=args.progress_path,
    )
    print(
        json.dumps(
            {
                "format_version": manifest["format_version"],
                "num_genes": manifest["num_genes"],
                "num_layers": manifest["num_layers"],
                "out_dir": args.out_dir,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
