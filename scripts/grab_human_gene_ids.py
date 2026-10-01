import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_s0_tools import (  # noqa: E402
    REGISTRY_SCHEMA_VERSION_V1,
    REGISTRY_SCHEMA_VERSION_V2,
)

# Match human Ensembl gene IDs and quoted GTF attributes.
GENE_ID_RE = re.compile(r"ENSG\d{11}")
GTF_ATTRIBUTE_RE = re.compile(r'([A-Za-z0-9_]+) "([^"]*)";')


def _parse_gtf_attributes(raw_attributes: str) -> dict[str, str]:
    """Convert one GTF attribute field to a dictionary."""
    return dict(GTF_ATTRIBUTE_RE.findall(raw_attributes))


def _read_gene_symbols(
    gtf_path: Path,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Read all human gene IDs from an Ensembl GTF file.

    Return the ID-to-symbol map and the summary counts.
    """
    all_gene_ids: set[str] = set()
    symbols_by_id: dict[str, set[str]] = {}
    gene_feature_rows = 0
    invalid_gene_id_rows = 0

    with gzip.open(gtf_path, "rt", encoding="utf-8") as gtf_file:
        for line_number, line in enumerate(gtf_file, start=1):
            if not line.strip() or line.startswith("#"):
                continue

            # Split each data row into the nine GTF fields.
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Invalid GTF row at line {line_number}")

            # Keep only gene feature rows.
            if fields[2] != "gene":
                continue

            gene_feature_rows += 1

            # Parse the attributes from the ninth GTF field.
            attributes = _parse_gtf_attributes(fields[8])
            gene_id = attributes.get("gene_id", "").strip()

            if GENE_ID_RE.fullmatch(gene_id) is None:
                invalid_gene_id_rows += 1
                continue

            all_gene_ids.add(gene_id)

            # Normalize each gene name before storage.
            gene_name = unicodedata.normalize(
                "NFKC",
                attributes.get("gene_name", ""),
            ).strip()

            if not gene_name:
                continue

            symbols_by_id.setdefault(gene_id, set()).add(gene_name)

    if not all_gene_ids:
        raise ValueError("The GTF contains no valid human gene IDs")

    ids_by_symbol: dict[str, set[str]] = {}
    for gene_id, symbols in symbols_by_id.items():
        for symbol in symbols:
            ids_by_symbol.setdefault(symbol, set()).add(gene_id)

    multiple_symbol_gene_ids = sorted(
        gene_id
        for gene_id, symbols in symbols_by_id.items()
        if len(symbols) > 1
    )
    if multiple_symbol_gene_ids:
        raise ValueError(
            "The GTF maps one gene ID to multiple normalized symbols: "
            f"{multiple_symbol_gene_ids[:10]}"
        )

    mappings = {
        gene_id: sorted(symbols_by_id.get(gene_id, set()))
        for gene_id in sorted(all_gene_ids)
    }
    gene_ids_without_name = len(all_gene_ids - set(symbols_by_id))

    counts = {
        "gene_feature_rows": gene_feature_rows,
        "total_gene_ids": len(all_gene_ids),
        "named_gene_ids": len(symbols_by_id),
        "gene_ids_with_multiple_symbols": 0,
        "gene_ids_without_name": gene_ids_without_name,
        "retained_gene_ids_without_name": gene_ids_without_name,
        "excluded_gene_ids_without_name": 0,
        "unique_gene_names": len(ids_by_symbol),
        "ambiguous_gene_names": sum(
            len(gene_ids) > 1 for gene_ids in ids_by_symbol.values()
        ),
        "largest_candidate_set": max(
            (len(gene_ids) for gene_ids in ids_by_symbol.values()),
            default=0,
        ),
        "invalid_gene_id_rows": invalid_gene_id_rows,
    }

    return mappings, counts


def _sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def write_gene_registry(
    mappings: dict[str, list[str]],
    counts: dict[str, int],
    *,
    gtf_path: Path,
    out_path: Path,
    schema_version: str = REGISTRY_SCHEMA_VERSION_V2,
) -> dict[str, object]:
    """Write the extracted Ensembl gene registry."""

    if schema_version == REGISTRY_SCHEMA_VERSION_V1:
        output_mappings = {
            gene_id: symbols
            for gene_id, symbols in mappings.items()
            if symbols
        }
        output_counts = {
            key: counts[key]
            for key in (
                "gene_feature_rows",
                "total_gene_ids",
                "named_gene_ids",
                "unique_gene_names",
                "ambiguous_gene_names",
                "largest_candidate_set",
                "invalid_gene_id_rows",
            )
        }
        output_counts["excluded_gene_ids_without_name"] = (
            counts["total_gene_ids"] - counts["named_gene_ids"]
        )
        gene_symbols_rule = "deduplicate and sort"
    elif schema_version == REGISTRY_SCHEMA_VERSION_V2:
        output_mappings = mappings
        output_counts = counts
        gene_symbols_rule = (
            "deduplicate and sort; use an empty list when no name exists"
        )
    else:
        raise ValueError("The registry schema version is not supported")

    payload = {
        "schema_version": schema_version,
        "source": {
            "assembly": "GRCh38",
            "file_name": gtf_path.name,
            "release": 116,
            "sha256": _sha256_file(gtf_path),
            "species": "Homo sapiens",
            "url": (
                "https://ftp.ensembl.org/pub/release-116/gtf/"
                "homo_sapiens/Homo_sapiens.GRCh38.116.gtf.gz"
            ),
        },
        "normalization": {
            "gene_id": "strip and require ENSG plus 11 digits",
            "gene_name": "apply NFKC and strip",
            "gene_symbols": gene_symbols_rule,
        },
        "counts": output_counts,
        "gene_symbols_by_id": output_mappings,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(output_mappings):,} gene IDs to {out_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract all human gene IDs from the Ensembl release 116 GTF."
        )
    )
    parser.add_argument(
        "--gtf-path",
        type=Path,
        required=True,
        help="Path to Homo_sapiens.GRCh38.116.gtf.gz.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Path for the extracted JSON registry.",
    )
    parser.add_argument(
        "--registry-schema-version",
        choices=(
            REGISTRY_SCHEMA_VERSION_V1,
            REGISTRY_SCHEMA_VERSION_V2,
        ),
        default=REGISTRY_SCHEMA_VERSION_V2,
        help="Registry schema to write. The default keeps unnamed gene IDs.",
    )
    return parser.parse_args()


def main() -> int:
    """Extract and write the Ensembl gene registry."""
    args = _parse_args()
    gtf_path = args.gtf_path.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()

    if not gtf_path.is_file():
        raise FileNotFoundError(gtf_path)

    if out_path == gtf_path:
        raise ValueError("The output path cannot replace the GTF file")

    mappings, counts = _read_gene_symbols(gtf_path)
    registry = write_gene_registry(
        mappings,
        counts,
        gtf_path=gtf_path,
        out_path=out_path,
        schema_version=args.registry_schema_version,
    )

    print(json.dumps(registry["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
