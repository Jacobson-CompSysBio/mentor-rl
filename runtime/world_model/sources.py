"""Read the pinned source formats used by the world-model corpora.

The readers are deliberately independent of corpus construction.  They turn
the authoritative files into small, validated Python records so the contract
checker and source-index builder share one interpretation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import hashlib
import math
from pathlib import Path
import re
import unicodedata


GENE_ID_PATTERN = re.compile(r"^ENSG[0-9]{11}$")
GTF_ATTRIBUTE_PATTERN = re.compile(r'([A-Za-z0-9_]+) "([^"]*)";')


class SourceError(ValueError):
    """Report a malformed or inconsistent pinned source."""


@dataclass(frozen=True, slots=True)
class RegistryGene:
    """One Ensembl gene feature needed by S0 and S1."""

    gene_id: str
    gene_symbol: str | None
    chromosome: str
    biotype: str


@dataclass(frozen=True, slots=True)
class LayerSource:
    """One path and native layer ID from a multiplex flist."""

    path: str
    layer_id: str


@dataclass(frozen=True, slots=True)
class StableIdHistory:
    """The human gene portion of an Ensembl stable-ID event table."""

    old_gene_ids: tuple[str, ...]
    gene_links: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class HistoricalSymbol:
    """One absent HGNC name and the current genes it remains tied to."""

    gene_symbol: str
    sources: tuple[str, ...]
    linked_gene_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DendrogramNode:
    """One binary MENTOR-EV dendrogram node."""

    node_id: int
    left_id: int
    right_id: int
    height: float
    label: str

    @property
    def is_leaf(self) -> bool:
        return self.left_id == -1


@dataclass(frozen=True, slots=True)
class Dendrogram:
    """A validated binary tree in child-before-parent traversal order."""

    nodes: dict[int, DendrogramNode]
    root_id: int
    postorder: tuple[int, ...]

    def leaf_counts(self) -> dict[int, int]:
        """Return the number of descendant leaves for every node."""

        counts: dict[int, int] = {}
        for node_id in self.postorder:
            node = self.nodes[node_id]
            counts[node_id] = (
                1
                if node.is_leaf
                else counts[node.left_id] + counts[node.right_id]
            )
        return counts

    def leaf_gene_ids(self) -> tuple[str, ...]:
        """Return the sorted, unique gene universe represented by the tree."""

        return tuple(
            sorted(node.label for node in self.nodes.values() if node.is_leaf)
        )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ensembl_gtf(path: Path) -> tuple[RegistryGene, ...]:
    """Read every human gene feature from one gzipped Ensembl GTF."""

    genes: dict[str, RegistryGene] = {}
    try:
        handle = gzip.open(path, "rt", encoding="utf-8")
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9:
                    raise SourceError(
                        f"{path}:{line_number}: expected 9 GTF fields"
                    )
                if fields[2] != "gene":
                    continue

                attributes = dict(GTF_ATTRIBUTE_PATTERN.findall(fields[8]))
                gene_id = attributes.get("gene_id", "").strip()
                if GENE_ID_PATTERN.fullmatch(gene_id) is None:
                    raise SourceError(
                        f"{path}:{line_number}: invalid human gene ID "
                        f"{gene_id!r}"
                    )
                chromosome = fields[0].strip()
                biotype = attributes.get("gene_biotype", "").strip()
                if not chromosome or not biotype:
                    raise SourceError(
                        f"{path}:{line_number}: gene lacks chromosome or biotype"
                    )

                raw_symbol = unicodedata.normalize(
                    "NFKC", attributes.get("gene_name", "")
                ).strip()
                gene_symbol = raw_symbol or None
                if gene_symbol == gene_id:
                    gene_symbol = None
                gene = RegistryGene(
                    gene_id=gene_id,
                    gene_symbol=gene_symbol,
                    chromosome=chromosome,
                    biotype=biotype,
                )
                if gene_id in genes:
                    raise SourceError(
                        f"{path}:{line_number}: duplicate gene ID {gene_id}"
                    )
                genes[gene_id] = gene
    except (OSError, UnicodeError) as exc:
        raise SourceError(f"cannot read Ensembl GTF {path}: {exc}") from exc

    if not genes:
        raise SourceError(f"{path}: no human gene features")
    return tuple(genes[gene_id] for gene_id in sorted(genes))


def read_ensembl_stable_id_history(path: Path) -> StableIdHistory:
    """Read canonical human gene events from an Ensembl MySQL table dump.

    The dump also contains transcripts, translations, RNA products, and old
    pre-Ensembl identifiers.  Only canonical ENSG-to-ENSG links are needed to
    form leakage-safe corpus split units.
    """

    null = r"\N"
    old_gene_ids: set[str] = set()
    gene_links: set[tuple[str, str]] = set()
    try:
        with gzip.open(path, "rt", encoding="latin-1", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for line_number, fields in enumerate(reader, 1):
                if len(fields) != 7:
                    raise SourceError(
                        f"{path}:{line_number}: expected 7 stable-ID fields"
                    )
                (
                    old_id,
                    old_version,
                    new_id,
                    new_version,
                    mapping_session_id,
                    feature_type,
                    score,
                ) = fields
                try:
                    if old_version != null:
                        int(old_version)
                    if new_version != null:
                        int(new_version)
                    int(mapping_session_id)
                    float(score)
                except ValueError as exc:
                    raise SourceError(
                        f"{path}:{line_number}: invalid stable-ID event value"
                    ) from exc
                if feature_type not in {
                    "gene",
                    "transcript",
                    "translation",
                    "rnaproduct",
                }:
                    raise SourceError(
                        f"{path}:{line_number}: unknown feature type "
                        f"{feature_type!r}"
                    )
                if feature_type != "gene":
                    continue
                old_is_gene = GENE_ID_PATTERN.fullmatch(old_id) is not None
                new_is_gene = GENE_ID_PATTERN.fullmatch(new_id) is not None
                if old_is_gene:
                    old_gene_ids.add(old_id)
                    if new_is_gene:
                        gene_links.add((old_id, new_id))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SourceError(
            f"cannot read Ensembl stable-ID history {path}: {exc}"
        ) from exc

    if not old_gene_ids:
        raise SourceError(f"{path}: no historical human gene IDs")
    return StableIdHistory(
        old_gene_ids=tuple(sorted(old_gene_ids)),
        gene_links=tuple(sorted(gene_links)),
    )


def read_hgnc_historical_symbols(
    path: Path,
    *,
    current_gene_ids: set[str],
    current_symbols: set[str],
) -> tuple[HistoricalSymbol, ...]:
    """Read HGNC previous names and aliases absent from the current registry.

    ``current_symbols`` must already be case-folded.  Links retain only genes
    in the pinned Ensembl release; they are split metadata, not lookup aliases.
    """

    required = {"alias_symbol", "prev_symbol", "ensembl_gene_id"}
    symbols: dict[str, tuple[set[str], set[str]]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise SourceError(
                    f"{path}: HGNC header lacks "
                    + ", ".join(sorted(required - set(reader.fieldnames or ())))
                )
            for line_number, row in enumerate(reader, 2):
                raw_ids = row.get("ensembl_gene_id")
                if not isinstance(raw_ids, str):
                    raise SourceError(
                        f"{path}:{line_number}: invalid Ensembl gene IDs"
                    )
                linked_ids: set[str] = set()
                for raw_id in raw_ids.strip('"').split("|"):
                    gene_id = raw_id.strip()
                    if not gene_id:
                        continue
                    if GENE_ID_PATTERN.fullmatch(gene_id) is None:
                        raise SourceError(
                            f"{path}:{line_number}: invalid Ensembl gene ID "
                            f"{gene_id!r}"
                        )
                    if gene_id in current_gene_ids:
                        linked_ids.add(gene_id)

                for column, source in (
                    ("alias_symbol", "alias"),
                    ("prev_symbol", "previous"),
                ):
                    raw_symbols = row.get(column)
                    if not isinstance(raw_symbols, str):
                        raise SourceError(
                            f"{path}:{line_number}: invalid {column} value"
                        )
                    for raw_symbol in raw_symbols.strip('"').split("|"):
                        symbol = unicodedata.normalize("NFKC", raw_symbol).strip()
                        if not symbol or symbol.casefold() in current_symbols:
                            continue
                        sources, gene_ids = symbols.setdefault(
                            symbol, (set(), set())
                        )
                        sources.add(source)
                        gene_ids.update(linked_ids)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SourceError(f"cannot read HGNC history {path}: {exc}") from exc

    if not symbols:
        raise SourceError(f"{path}: no absent HGNC symbols")
    return tuple(
        HistoricalSymbol(
            gene_symbol=symbol,
            sources=tuple(sorted(sources)),
            linked_gene_ids=tuple(sorted(gene_ids)),
        )
        for symbol, (sources, gene_ids) in sorted(
            symbols.items(), key=lambda item: (item[0].casefold(), item[0])
        )
    )


def read_flist(path: Path) -> tuple[LayerSource, ...]:
    """Read an exact two-column multiplex flist in source order."""

    layers: list[LayerSource] = []
    layer_ids: set[str] = set()
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 2 or not all(fields):
                raise SourceError(
                    f"{path}:{line_number}: expected path and layer ID"
                )
            layer = LayerSource(path=fields[0], layer_id=fields[1])
            if layer.layer_id in layer_ids:
                raise SourceError(
                    f"{path}:{line_number}: duplicate layer ID {layer.layer_id!r}"
                )
            layers.append(layer)
            layer_ids.add(layer.layer_id)
    except (OSError, UnicodeError) as exc:
        raise SourceError(f"cannot read multiplex flist {path}: {exc}") from exc

    if not layers:
        raise SourceError(f"{path}: no multiplex layers")
    return tuple(layers)


def read_dendrogram(path: Path) -> Dendrogram:
    """Read and fully validate one binary MENTOR-EV dendrogram."""

    nodes: dict[int, DendrogramNode] = {}
    leaf_labels: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            expected_header = [
                "node_id",
                "left_id",
                "right_id",
                "height",
                "label",
            ]
            if reader.fieldnames != expected_header:
                raise SourceError(
                    f"{path}: expected dendrogram header "
                    + "\t".join(expected_header)
                )
            for line_number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise SourceError(
                        f"{path}:{line_number}: expected 5 dendrogram fields"
                    )
                try:
                    node = DendrogramNode(
                        node_id=int(row["node_id"]),
                        left_id=int(row["left_id"]),
                        right_id=int(row["right_id"]),
                        height=float(row["height"]),
                        label=row["label"],
                    )
                except (TypeError, ValueError) as exc:
                    raise SourceError(
                        f"{path}:{line_number}: invalid dendrogram value"
                    ) from exc
                if node.node_id < 0 or node.node_id in nodes:
                    raise SourceError(
                        f"{path}:{line_number}: duplicate or negative node ID"
                    )
                if not math.isfinite(node.height) or node.height < 0:
                    raise SourceError(
                        f"{path}:{line_number}: height must be finite and nonnegative"
                    )

                sentinels = (node.left_id == -1, node.right_id == -1)
                if sentinels == (True, True):
                    if GENE_ID_PATTERN.fullmatch(node.label) is None:
                        raise SourceError(
                            f"{path}:{line_number}: invalid leaf label {node.label!r}"
                        )
                    if node.label in leaf_labels:
                        raise SourceError(
                            f"{path}:{line_number}: duplicate leaf {node.label}"
                        )
                    leaf_labels.add(node.label)
                elif sentinels != (False, False):
                    raise SourceError(
                        f"{path}:{line_number}: node must have zero or two children"
                    )
                elif (
                    node.left_id < 0
                    or node.right_id < 0
                    or node.left_id == node.right_id
                    or node.node_id in (node.left_id, node.right_id)
                    or node.label != "NA"
                ):
                    raise SourceError(
                        f"{path}:{line_number}: invalid internal node"
                    )
                nodes[node.node_id] = node
    except (OSError, UnicodeError) as exc:
        raise SourceError(f"cannot read dendrogram {path}: {exc}") from exc

    if not nodes:
        raise SourceError(f"{path}: no dendrogram nodes")

    parent_by_node: dict[int, int] = {}
    for node in nodes.values():
        if node.is_leaf:
            continue
        for child_id in (node.left_id, node.right_id):
            if child_id not in nodes:
                raise SourceError(
                    f"{path}: node {node.node_id} references missing child {child_id}"
                )
            if child_id in parent_by_node:
                raise SourceError(f"{path}: node {child_id} has multiple parents")
            parent_by_node[child_id] = node.node_id

    roots = sorted(set(nodes) - set(parent_by_node))
    if len(roots) != 1:
        raise SourceError(f"{path}: expected one root, found {len(roots)}")
    root_id = roots[0]

    postorder: list[int] = []
    active: set[int] = set()
    complete: set[int] = set()
    stack: list[tuple[int, bool]] = [(root_id, False)]
    while stack:
        node_id, exiting = stack.pop()
        if exiting:
            active.remove(node_id)
            complete.add(node_id)
            postorder.append(node_id)
            continue
        if node_id in active:
            raise SourceError(f"{path}: dendrogram contains a cycle")
        if node_id in complete:
            continue
        active.add(node_id)
        stack.append((node_id, True))
        node = nodes[node_id]
        if not node.is_leaf:
            stack.append((node.right_id, False))
            stack.append((node.left_id, False))

    if len(complete) != len(nodes):
        raise SourceError(f"{path}: dendrogram is disconnected")
    return Dendrogram(
        nodes=nodes,
        root_id=root_id,
        postorder=tuple(postorder),
    )
