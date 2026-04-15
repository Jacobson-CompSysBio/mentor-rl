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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


STORE_FORMAT_VERSION = "mentor-rl-multiplex-store-v1"
WEIGHT_DTYPE = np.float32
INDEX_DTYPE = np.uint32
INDPTR_DTYPE = np.uint64


@dataclass(frozen=True)
class FlistEntry:
    """One layer entry from the multiplex flist."""

    path: str
    layer_name: str
    enabled: bool = True


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


def collect_gene_universe(entries: list[FlistEntry]) -> list[str]:
    """Read all layers once and collect the global sorted gene universe."""

    genes: set[str] = set()
    for entry in entries:
        for source, target, _ in iter_weighted_edges(entry.path):
            genes.add(source)
            genes.add(target)

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
) -> dict[str, object]:
    """Build the binary multiplex store and return the manifest payload."""

    start_time = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    entries = parse_flist(multiplex_flist, limit_layers=limit_layers)
    genes = collect_gene_universe(entries)
    gene_to_index = {gene_id: index for index, gene_id in enumerate(genes)}
    num_genes = len(genes)

    write_gene_table(genes, out_path)

    aggregate_csr: sparse.csr_matrix | None = None
    layer_rows: list[dict[str, object]] = []
    manifest_layers: list[dict[str, object]] = []

    for layer_index, entry in enumerate(entries):
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

    if aggregate_csr is None:
        raise ValueError("The aggregate graph could not be constructed.")

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

    return manifest


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
    return parser


def main() -> None:
    """Command-line entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    manifest = build_store(
        multiplex_flist=args.multiplex_flist,
        out_dir=args.out_dir,
        limit_layers=args.limit_layers,
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
