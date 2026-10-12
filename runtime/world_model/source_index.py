"""Build the compact, content-addressed S0/S1 source index.

The generated file contains only facts needed to construct corpora, form
leakage-safe identity splits, or serve the three S1 tools.  It intentionally
excludes graph edges, per-layer memberships, parsed layer fields, and
MENTOR-EV subtree members because those are either out of scope for S1 or
cheaply derivable from pinned sources.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .contract import CONTRACT_PATH, REPOSITORY_ROOT, load_contract
from .sources import (
    GENE_ID_PATTERN,
    SourceError,
    file_sha256,
    read_dendrogram,
    read_ensembl_gtf,
    read_ensembl_stable_id_history,
    read_flist,
    read_hgnc_historical_symbols,
)


SOURCE_INDEX_SCHEMA_VERSION = "mentor-rl-world-model-source-index-v1"
SOURCE_INDEX_PREFIX = "world_model_source_index_"
GENE_FIELDS = ["gene_symbol", "chromosome", "biotype"]
HISTORICAL_SYMBOL_FIELDS = ["gene_symbol", "sources", "linked_gene_ids"]
RETIRED_GENE_FIELDS = ["gene_id", "linked_gene_ids"]
MENTOR_CLADE_FIELDS = ["node_id", "leaf_count"]


class SourceIndexError(ValueError):
    """Report a source-index build or validation failure."""


def _source_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise SourceIndexError("source paths must be nonempty strings")
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _require_digest(actual: str, expected: Any, label: str) -> None:
    if actual != expected:
        raise SourceIndexError(f"{label} SHA-256 is {actual}, expected {expected}")


def _require_count(actual: int, expected: Any, label: str) -> None:
    if actual != expected:
        raise SourceIndexError(f"{label} is {actual:,}, expected {expected!r}")


def _registry_statistics(
    genes_by_id: Mapping[str, list[str | None]],
) -> dict[str, int]:
    symbols_by_key: dict[str, list[str]] = defaultdict(list)
    spellings_by_key: dict[str, set[str]] = defaultdict(set)
    chromosomes: set[str] = set()
    biotypes: set[str] = set()
    named = 0
    for gene_id, values in genes_by_id.items():
        symbol, chromosome, biotype = values
        chromosomes.add(str(chromosome))
        biotypes.add(str(biotype))
        if symbol is not None:
            named += 1
            key = symbol.casefold()
            symbols_by_key[key].append(gene_id)
            spellings_by_key[key].add(symbol)
    candidate_sizes = [len(values) for values in symbols_by_key.values()]
    return {
        "registry_gene_count": len(genes_by_id),
        "registry_named_gene_count": named,
        "registry_unnamed_gene_count": len(genes_by_id) - named,
        "registry_symbol_count": len(symbols_by_key),
        "registry_ambiguous_symbol_count": sum(
            size > 1 for size in candidate_sizes
        ),
        "registry_largest_candidate_count": max(candidate_sizes, default=0),
        "registry_chromosome_count": len(chromosomes),
        "registry_biotype_count": len(biotypes),
        "registry_casefold_collision_count": sum(
            len(spellings) > 1 for spellings in spellings_by_key.values()
        ),
    }


def _retired_gene_rows(
    old_gene_ids: Sequence[str],
    gene_links: Sequence[tuple[str, str]],
    current_gene_ids: set[str],
) -> list[list[Any]]:
    """Return retired IDs with every current gene in their history component."""

    parent: dict[str, str] = {}

    def find(gene_id: str) -> str:
        root = parent.setdefault(gene_id, gene_id)
        while parent[root] != root:
            root = parent[root]
        while parent[gene_id] != gene_id:
            next_id = parent[gene_id]
            parent[gene_id] = root
            gene_id = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    for gene_id in old_gene_ids:
        find(gene_id)
    for old_id, new_id in gene_links:
        union(old_id, new_id)

    current_by_root: dict[str, list[str]] = defaultdict(list)
    for gene_id in sorted(current_gene_ids.intersection(parent)):
        current_by_root[find(gene_id)].append(gene_id)
    return [
        [gene_id, current_by_root.get(find(gene_id), [])]
        for gene_id in old_gene_ids
        if gene_id not in current_gene_ids
    ]


def build_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate the logical index payload from pinned sources."""

    coordinate = contract["coordinate_system"]
    registry_spec = coordinate["identifier_registry"]
    stable_id_spec = registry_spec["stable_id_history"]
    hgnc_spec = registry_spec["hgnc_history"]
    mentor_spec = coordinate["mentor_ev"]
    index_spec = contract["source_index"]
    if index_spec["schema_version"] != SOURCE_INDEX_SCHEMA_VERSION:
        raise SourceIndexError("the source-index schema version is unsupported")

    gtf_path = _source_path(registry_spec["source_path"])
    stable_id_path = _source_path(stable_id_spec["source_path"])
    hgnc_path = _source_path(hgnc_spec["source_path"])
    flist_path = _source_path(coordinate["source_flist_path"])
    dendrogram_path = _source_path(mentor_spec["source_path"])
    for path in (
        gtf_path,
        stable_id_path,
        hgnc_path,
        flist_path,
        dendrogram_path,
    ):
        if not path.is_file():
            raise SourceIndexError(f"pinned source is not mounted: {path}")

    source_hashes = {
        "ensembl_gtf": file_sha256(gtf_path),
        "ensembl_stable_id_history": file_sha256(stable_id_path),
        "hgnc_history": file_sha256(hgnc_path),
        "multiplex_flist": file_sha256(flist_path),
        "mentor_ev_dendrogram": file_sha256(dendrogram_path),
    }
    _require_digest(
        source_hashes["ensembl_gtf"],
        registry_spec["source_sha256"],
        "Ensembl GTF",
    )
    _require_digest(
        source_hashes["ensembl_stable_id_history"],
        stable_id_spec["source_sha256"],
        "Ensembl stable-ID history",
    )
    _require_digest(
        source_hashes["hgnc_history"],
        hgnc_spec["source_sha256"],
        "HGNC history",
    )
    _require_digest(
        source_hashes["multiplex_flist"],
        coordinate["source_flist_sha256"],
        "multiplex flist",
    )
    _require_digest(
        source_hashes["mentor_ev_dendrogram"],
        mentor_spec["sha256"],
        "MENTOR-EV dendrogram",
    )

    try:
        registry_genes = read_ensembl_gtf(gtf_path)
        stable_id_history = read_ensembl_stable_id_history(stable_id_path)
        layers = read_flist(flist_path)
        dendrogram = read_dendrogram(dendrogram_path)
    except SourceError as exc:
        raise SourceIndexError(str(exc)) from exc

    genes_by_id = {
        gene.gene_id: [gene.gene_symbol, gene.chromosome, gene.biotype]
        for gene in registry_genes
    }
    current_gene_ids = set(genes_by_id)
    current_symbols = {
        str(values[0]).casefold()
        for values in genes_by_id.values()
        if values[0] is not None
    }
    try:
        historical_symbols = [
            [
                symbol.gene_symbol,
                list(symbol.sources),
                list(symbol.linked_gene_ids),
            ]
            for symbol in read_hgnc_historical_symbols(
                hgnc_path,
                current_gene_ids=current_gene_ids,
                current_symbols=current_symbols,
            )
        ]
    except SourceError as exc:
        raise SourceIndexError(str(exc)) from exc
    retired_genes = _retired_gene_rows(
        stable_id_history.old_gene_ids,
        stable_id_history.gene_links,
        current_gene_ids,
    )
    multiplex_gene_ids = list(dendrogram.leaf_gene_ids())
    leaf_counts = dendrogram.leaf_counts()
    bounds = mentor_spec["eligible_leaf_count"]
    mentor_clades = [
        [node_id, leaf_counts[node_id]]
        for node_id in sorted(dendrogram.nodes)
        if not dendrogram.nodes[node_id].is_leaf
        and bounds["min"] <= leaf_counts[node_id] <= bounds["max"]
    ]

    counts = _registry_statistics(genes_by_id)
    counts.update(
        {
            "historical_symbol_count": len(historical_symbols),
            "historical_symbol_key_count": len(
                {row[0].casefold() for row in historical_symbols}
            ),
            "historical_symbol_linked_count": sum(
                bool(row[2]) for row in historical_symbols
            ),
            "retired_gene_count": len(retired_genes),
            "retired_gene_linked_count": sum(
                bool(row[1]) for row in retired_genes
            ),
            "layer_count": len(layers),
            "multiplex_gene_count": len(multiplex_gene_ids),
            "registry_multiplex_intersection_count": len(
                set(genes_by_id).intersection(multiplex_gene_ids)
            ),
            "mentor_node_count": len(dendrogram.nodes),
            "mentor_leaf_count": len(multiplex_gene_ids),
            "mentor_eligible_clade_count": len(mentor_clades),
        }
    )

    registry_count_keys = {
        "gene_count": "registry_gene_count",
        "named_gene_count": "registry_named_gene_count",
        "unnamed_gene_count": "registry_unnamed_gene_count",
        "symbol_count": "registry_symbol_count",
        "ambiguous_symbol_count": "registry_ambiguous_symbol_count",
        "largest_candidate_count": "registry_largest_candidate_count",
        "chromosome_count": "registry_chromosome_count",
        "biotype_count": "registry_biotype_count",
        "casefold_collision_count": "registry_casefold_collision_count",
    }
    for contract_key, count_key in registry_count_keys.items():
        _require_count(counts[count_key], registry_spec[contract_key], count_key)
    for contract_key, count_key in {
        "symbol_count": "historical_symbol_count",
        "symbol_key_count": "historical_symbol_key_count",
        "linked_symbol_count": "historical_symbol_linked_count",
    }.items():
        _require_count(counts[count_key], hgnc_spec[contract_key], count_key)
    for contract_key, count_key in {
        "retired_gene_count": "retired_gene_count",
        "linked_retired_gene_count": "retired_gene_linked_count",
    }.items():
        _require_count(counts[count_key], stable_id_spec[contract_key], count_key)
    _require_count(counts["layer_count"], coordinate["layer_count"], "layer_count")
    _require_count(
        counts["multiplex_gene_count"],
        coordinate["graph_gene_count"],
        "multiplex_gene_count",
    )
    _require_count(
        counts["registry_multiplex_intersection_count"],
        coordinate["registry_graph_intersection_count"],
        "registry_multiplex_intersection_count",
    )
    _require_count(
        counts["mentor_node_count"], mentor_spec["node_count"], "mentor_node_count"
    )
    _require_count(
        counts["mentor_leaf_count"], mentor_spec["leaf_count"], "mentor_leaf_count"
    )
    _require_count(
        counts["mentor_eligible_clade_count"],
        mentor_spec["eligible_clade_count"],
        "mentor_eligible_clade_count",
    )

    payload = {
        "schema_version": SOURCE_INDEX_SCHEMA_VERSION,
        "sources": {
            "ensembl_gtf": {
                "file_name": gtf_path.name,
                "sha256": source_hashes["ensembl_gtf"],
            },
            "ensembl_stable_id_history": {
                "file_name": stable_id_path.name,
                "sha256": source_hashes["ensembl_stable_id_history"],
            },
            "hgnc_history": {
                "file_name": hgnc_path.name,
                "sha256": source_hashes["hgnc_history"],
            },
            "multiplex_flist": {
                "file_name": flist_path.name,
                "sha256": source_hashes["multiplex_flist"],
            },
            "mentor_ev_dendrogram": {
                "file_name": dendrogram_path.name,
                "sha256": source_hashes["mentor_ev_dendrogram"],
            },
        },
        "coordinate_system": {
            "ensembl_release": coordinate["ensembl_release"],
            "identifier_registry_id": coordinate["identifier_registry_id"],
            "multiplex_id": coordinate["multiplex_id"],
            "dendrogram_id": mentor_spec["dendrogram_id"],
            "eligible_leaf_count": dict(bounds),
            "multiplex_gene_rule": "mentor_ev_all_gene_dendrogram_leaves",
        },
        "counts": counts,
        "gene_fields": GENE_FIELDS,
        "genes_by_id": genes_by_id,
        "historical_symbol_fields": HISTORICAL_SYMBOL_FIELDS,
        "historical_symbols": historical_symbols,
        "retired_gene_fields": RETIRED_GENE_FIELDS,
        "retired_genes": retired_genes,
        "multiplex_gene_ids": multiplex_gene_ids,
        "layer_ids": [layer.layer_id for layer in layers],
        "mentor_clade_fields": MENTOR_CLADE_FIELDS,
        "mentor_clades": mentor_clades,
    }
    _validate_payload(payload)
    return payload


def _validate_payload(payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "sources",
        "coordinate_system",
        "counts",
        "gene_fields",
        "genes_by_id",
        "historical_symbol_fields",
        "historical_symbols",
        "retired_gene_fields",
        "retired_genes",
        "multiplex_gene_ids",
        "layer_ids",
        "mentor_clade_fields",
        "mentor_clades",
    }
    if set(payload) != expected_keys:
        raise SourceIndexError("source-index top-level fields changed")
    if payload["schema_version"] != SOURCE_INDEX_SCHEMA_VERSION:
        raise SourceIndexError("source-index schema version changed")
    if payload["gene_fields"] != GENE_FIELDS:
        raise SourceIndexError("source-index gene fields changed")
    if payload["historical_symbol_fields"] != HISTORICAL_SYMBOL_FIELDS:
        raise SourceIndexError("source-index historical-symbol fields changed")
    if payload["retired_gene_fields"] != RETIRED_GENE_FIELDS:
        raise SourceIndexError("source-index retired-gene fields changed")
    if payload["mentor_clade_fields"] != MENTOR_CLADE_FIELDS:
        raise SourceIndexError("source-index clade fields changed")

    sources = payload["sources"]
    expected_sources = {
        "ensembl_gtf",
        "ensembl_stable_id_history",
        "hgnc_history",
        "multiplex_flist",
        "mentor_ev_dendrogram",
    }
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise SourceIndexError("source-index source manifest changed")
    for name, source in sources.items():
        if not isinstance(source, dict) or set(source) != {"file_name", "sha256"}:
            raise SourceIndexError(f"invalid source manifest for {name}")
        if not isinstance(source["file_name"], str) or not source["file_name"]:
            raise SourceIndexError(f"invalid source file name for {name}")
        if re.fullmatch(r"[0-9a-f]{64}", str(source["sha256"])) is None:
            raise SourceIndexError(f"invalid source SHA-256 for {name}")

    genes = payload["genes_by_id"]
    if not isinstance(genes, dict) or not genes:
        raise SourceIndexError("source index contains no registry genes")
    gene_ids = set(genes)
    for gene_id, values in genes.items():
        if GENE_ID_PATTERN.fullmatch(gene_id) is None:
            raise SourceIndexError(f"invalid indexed gene ID: {gene_id!r}")
        if not isinstance(values, list) or len(values) != len(GENE_FIELDS):
            raise SourceIndexError(f"invalid registry row for {gene_id}")
        symbol, chromosome, biotype = values
        if symbol is not None and (not isinstance(symbol, str) or not symbol):
            raise SourceIndexError(f"invalid registry symbol for {gene_id}")
        if not isinstance(chromosome, str) or not chromosome:
            raise SourceIndexError(f"invalid registry chromosome for {gene_id}")
        if not isinstance(biotype, str) or not biotype:
            raise SourceIndexError(f"invalid registry biotype for {gene_id}")

    historical_symbols = payload["historical_symbols"]
    historical_order: list[tuple[str, str]] = []
    for row in historical_symbols:
        if not isinstance(row, list) or len(row) != 3:
            raise SourceIndexError(f"invalid historical-symbol row: {row!r}")
        symbol, sources, linked_gene_ids = row
        if not isinstance(symbol, str) or not symbol:
            raise SourceIndexError("historical symbols must be nonempty strings")
        if (
            not isinstance(sources, list)
            or sources != sorted(set(sources))
            or not sources
            or not set(sources) <= {"alias", "previous"}
        ):
            raise SourceIndexError(f"invalid historical sources for {symbol!r}")
        if (
            not isinstance(linked_gene_ids, list)
            or linked_gene_ids != sorted(set(linked_gene_ids))
            or not set(linked_gene_ids) <= gene_ids
        ):
            raise SourceIndexError(f"invalid historical links for {symbol!r}")
        historical_order.append((symbol.casefold(), symbol))
    if historical_order != sorted(set(historical_order)):
        raise SourceIndexError("historical symbols are not sorted and unique")

    retired_genes = payload["retired_genes"]
    retired_ids: list[str] = []
    for row in retired_genes:
        if not isinstance(row, list) or len(row) != 2:
            raise SourceIndexError(f"invalid retired-gene row: {row!r}")
        gene_id, linked_gene_ids = row
        if GENE_ID_PATTERN.fullmatch(gene_id) is None or gene_id in gene_ids:
            raise SourceIndexError(f"invalid retired gene ID: {gene_id!r}")
        if (
            not isinstance(linked_gene_ids, list)
            or linked_gene_ids != sorted(set(linked_gene_ids))
            or not set(linked_gene_ids) <= gene_ids
        ):
            raise SourceIndexError(f"invalid retired links for {gene_id!r}")
        retired_ids.append(gene_id)
    if retired_ids != sorted(set(retired_ids)):
        raise SourceIndexError("retired gene IDs are not sorted and unique")

    multiplex_genes = payload["multiplex_gene_ids"]
    if (
        not isinstance(multiplex_genes, list)
        or multiplex_genes != sorted(set(multiplex_genes))
        or any(GENE_ID_PATTERN.fullmatch(value) is None for value in multiplex_genes)
    ):
        raise SourceIndexError("multiplex genes are not sorted unique ENSG IDs")

    layer_ids = payload["layer_ids"]
    if (
        not isinstance(layer_ids, list)
        or not layer_ids
        or len(layer_ids) != len(set(layer_ids))
        or any(not isinstance(value, str) or not value for value in layer_ids)
    ):
        raise SourceIndexError("layer IDs are empty, duplicated, or invalid")

    bounds = payload["coordinate_system"]["eligible_leaf_count"]
    clades = payload["mentor_clades"]
    node_ids: list[int] = []
    for row in clades:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(isinstance(value, int) for value in row)
            or not bounds["min"] <= row[1] <= bounds["max"]
        ):
            raise SourceIndexError(f"invalid eligible clade row: {row!r}")
        node_ids.append(row[0])
    if node_ids != sorted(set(node_ids)):
        raise SourceIndexError("eligible clade node IDs are not sorted and unique")

    counts = payload["counts"]
    actual = _registry_statistics(genes)
    actual.update(
        {
            "historical_symbol_count": len(historical_symbols),
            "historical_symbol_key_count": len(
                {row[0].casefold() for row in historical_symbols}
            ),
            "historical_symbol_linked_count": sum(
                bool(row[2]) for row in historical_symbols
            ),
            "retired_gene_count": len(retired_genes),
            "retired_gene_linked_count": sum(
                bool(row[1]) for row in retired_genes
            ),
            "layer_count": len(layer_ids),
            "multiplex_gene_count": len(multiplex_genes),
            "registry_multiplex_intersection_count": len(
                set(genes).intersection(multiplex_genes)
            ),
            "mentor_leaf_count": len(multiplex_genes),
            "mentor_eligible_clade_count": len(clades),
        }
    )
    for key, value in actual.items():
        if counts.get(key) != value:
            raise SourceIndexError(
                f"source-index count {key} is {counts.get(key)!r}, expected {value}"
            )


def content_sha256(payload: Mapping[str, Any]) -> str:
    """Return the digest used to address a logical source-index payload."""

    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _artifact_name(digest: str) -> str:
    return f"{SOURCE_INDEX_PREFIX}{digest}.json.gz"


def write_source_index(
    payload: Mapping[str, Any], output_directory: Path
) -> Path:
    """Atomically write one canonical gzip artifact and return its path."""

    _validate_payload(payload)
    canonical = _canonical_json(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / _artifact_name(digest)
    if target.exists():
        check_source_index(target)
        return target

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".source_index_",
        suffix=".tmp",
        dir=output_directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as compressed:
                compressed.write(canonical)
        temporary.replace(target)
        target.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_source_index(path: Path) -> dict[str, Any]:
    """Load a canonical source-index artifact."""

    try:
        with gzip.open(path, "rb") as handle:
            raw = handle.read()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceIndexError(f"cannot read source index {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SourceIndexError("source index root must be one object")
    if raw != _canonical_json(payload):
        raise SourceIndexError("source index is not canonical JSON")
    _validate_payload(payload)
    return payload


def check_source_index(
    path: Path, contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate one artifact, its content address, and optional contract pin."""

    payload = load_source_index(path)
    digest = content_sha256(payload)
    if path.name != _artifact_name(digest):
        raise SourceIndexError(
            f"source-index filename does not match content SHA-256 {digest}"
        )
    if contract is not None:
        index_spec = contract["source_index"]
        expected_digest = index_spec.get("content_sha256")
        if expected_digest is not None and digest != expected_digest:
            raise SourceIndexError(
                f"source-index content SHA-256 is {digest}, expected {expected_digest}"
            )
        coordinate = contract["coordinate_system"]
        registry = coordinate["identifier_registry"]
        expected_sources = {
            "ensembl_gtf": registry["source_sha256"],
            "ensembl_stable_id_history": registry["stable_id_history"][
                "source_sha256"
            ],
            "hgnc_history": registry["hgnc_history"]["source_sha256"],
            "multiplex_flist": coordinate["source_flist_sha256"],
            "mentor_ev_dendrogram": coordinate["mentor_ev"]["sha256"],
        }
        for name, expected in expected_sources.items():
            _require_digest(payload["sources"][name]["sha256"], expected, name)
        expected_coordinate = {
            "ensembl_release": coordinate["ensembl_release"],
            "identifier_registry_id": coordinate["identifier_registry_id"],
            "multiplex_id": coordinate["multiplex_id"],
            "dendrogram_id": coordinate["mentor_ev"]["dendrogram_id"],
            "eligible_leaf_count": coordinate["mentor_ev"]["eligible_leaf_count"],
            "multiplex_gene_rule": "mentor_ev_all_gene_dendrogram_leaves",
        }
        if payload["coordinate_system"] != expected_coordinate:
            raise SourceIndexError("source-index coordinate system changed")
    return payload


def build_source_index(
    contract: Mapping[str, Any], output_directory: Path
) -> Path:
    """Build, write, and revalidate one source-index artifact."""

    expected = contract["source_index"].get("content_sha256")
    if expected is not None:
        existing = output_directory / _artifact_name(expected)
        if existing.is_file():
            check_source_index(existing, contract)
            return existing

    payload = build_payload(contract)
    digest = content_sha256(payload)
    if expected is not None and digest != expected:
        raise SourceIndexError(
            f"built source-index SHA-256 is {digest}, expected {expected}"
        )
    path = write_source_index(payload, output_directory)
    check_source_index(path, contract)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=CONTRACT_PATH,
        help="Canonical S0/S1 contract (default: runtime/world_model/contract.yaml).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the pinned source index")
    build.add_argument(
        "--output-directory",
        type=Path,
        help="Override the generated-data directory from the contract.",
    )
    check = subparsers.add_parser("check", help="validate one source index")
    check.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = load_contract(args.contract)
    if args.command == "build":
        configured = _source_path(contract["source_index"]["output_directory"])
        output_directory = args.output_directory or configured
        path = build_source_index(contract, output_directory.resolve())
        payload = load_source_index(path)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "content_sha256": content_sha256(payload),
                    "counts": payload["counts"],
                },
                sort_keys=True,
            )
        )
    else:
        payload = check_source_index(args.path.resolve(), contract)
        print(
            json.dumps(
                {
                    "path": str(args.path.resolve()),
                    "content_sha256": content_sha256(payload),
                    "counts": payload["counts"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
