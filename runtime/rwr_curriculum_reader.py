"""Deterministic, path-safe access to the precomputed RWR-LOE artifacts.

The full-brain cache has two related products:

* one complete, seed-excluded rank vector per seed gene; and
* lower-triangular Spearman-distance matrices over the seed genes assigned to
  each precompute shard.

This module deliberately does not know anything about SFT rendering.  It turns
those products into exact oracle operations and emits only opaque cache/build
identifiers plus the network-flist digest as provenance.  Source filesystem
paths in the historical cache metadata never cross the public API boundary.
"""

from __future__ import annotations

from array import array
import csv
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


class RwrArtifactError(ValueError):
    """Base class for malformed or incompatible RWR curriculum artifacts."""


class UnknownSeedError(KeyError):
    """Raised when the requested seed has no rank-cache artifact."""


class UnknownRankedGeneError(KeyError):
    """Raised when a gene is absent from a seed's rank vector."""


class CrossShardDistanceUnavailable(LookupError):
    """Raised when no materialized distance shard contains both genes."""

    def __init__(self, route: "PairShardRoute") -> None:
        self.route = route
        super().__init__(
            f"No materialized distance shard contains {route.gene_a} and "
            f"{route.gene_b}; routes are {route.gene_a_shard_id!r} and "
            f"{route.gene_b_shard_id!r}."
        )


def _canonical_gene(value: str) -> str:
    gene = str(value).strip().upper()
    if not gene:
        raise ValueError("Gene identifiers must be non-empty.")
    return gene


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _opaque_sha256(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def rank_cache_file_stem(seed_gene_id: str) -> str:
    """Return the stable filename stem used by ``build_rwr_loe_corpus.py``."""

    seed = _canonical_gene(seed_gene_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", seed).strip("._")
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:80]}_{digest}" if safe else digest


@dataclass(frozen=True)
class CacheIdentity:
    """Path-free identity of one rank-cache context."""

    cache_id: str
    context_sha256: str
    schema_version: str
    network_flist_sha256: str
    parser_version: str
    ranking_semantics: str
    rwr_hpc_build_sha256: str
    restart: float | None
    delta: float | None
    reduction_method: str | None
    threshold: float | None

    def provenance(self, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rank_cache_id": self.cache_id,
            "rank_cache_context_sha256": self.context_sha256,
            "rank_cache_schema_version": self.schema_version,
            "network_flist_sha256": self.network_flist_sha256,
            "rank_cache_parser_version": self.parser_version,
            "ranking_semantics": self.ranking_semantics,
            "rwr_hpc_build_sha256": self.rwr_hpc_build_sha256,
        }
        payload.update(extra)
        return payload


@dataclass(frozen=True, order=True)
class RankRow:
    rank: int
    gene_id: str
    score: float = field(compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {"gene_id": self.gene_id, "rank": self.rank, "score": self.score}


@dataclass(frozen=True)
class SeedRankMetadata:
    seed_gene_id: str
    ranked_gene_count: int
    shard_id: str
    shard_index: int
    shard_count: int
    created_at: str | None
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ElbowSummary:
    seed_gene_id: str
    status: str
    elbow_rank_cutoff: int | None
    elbow_score: float | None
    retained_gene_ids: tuple[str, ...]
    ranked_gene_count: int
    membership_rule: str = "rank < elbow_rank_cutoff"

    def contains(self, gene_id: str) -> bool:
        return _canonical_gene(gene_id) in set(self.retained_gene_ids)


@dataclass(frozen=True)
class RankComparison:
    seed_gene_id: str
    candidate_a: RankRow
    candidate_b: RankRow
    closer_gene_id: str | None
    is_tie: bool


@dataclass(frozen=True)
class QueryRankResult:
    seed_gene_id: str
    query_gene_ids: tuple[str, ...]
    ranked_query_genes: tuple[RankRow, ...]
    missing_gene_ids: tuple[str, ...]


@dataclass(frozen=True)
class TopKIntersection:
    seed_gene_id_a: str
    seed_gene_id_b: str
    top_k: int
    neighborhood_a: tuple[str, ...]
    neighborhood_b: tuple[str, ...]
    intersection_gene_ids: tuple[str, ...]
    intersection_size: int
    union_size: int
    jaccard: float


@dataclass(frozen=True)
class RankVector:
    seed_gene_id: str
    rows: tuple[RankRow, ...]
    metadata: SeedRankMetadata
    _by_gene: Mapping[str, RankRow] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_gene = {row.gene_id: row for row in self.rows}
        if len(by_gene) != len(self.rows):
            raise RwrArtifactError(f"Duplicate genes in rank vector for {self.seed_gene_id}.")
        object.__setattr__(self, "_by_gene", MappingProxyType(by_gene))

    def lookup(self, gene_id: str) -> RankRow | None:
        return self._by_gene.get(_canonical_gene(gene_id))

    def require(self, gene_id: str) -> RankRow:
        gene = _canonical_gene(gene_id)
        row = self._by_gene.get(gene)
        if row is None:
            raise UnknownRankedGeneError(
                f"{gene} is absent from the full rank vector for {self.seed_gene_id}."
            )
        return row

    def compare(self, candidate_a: str, candidate_b: str) -> RankComparison:
        row_a = self.require(candidate_a)
        row_b = self.require(candidate_b)
        is_tie = row_a.rank == row_b.rank
        closer = None if is_tie else min((row_a, row_b), key=lambda row: (row.rank, row.gene_id)).gene_id
        return RankComparison(self.seed_gene_id, row_a, row_b, closer, is_tie)

    def top_k(self, k: int, *, exclude_genes: Iterable[str] = ()) -> tuple[RankRow, ...]:
        if k < 0:
            raise ValueError("k must be non-negative.")
        excluded = {_canonical_gene(gene) for gene in exclude_genes}
        return tuple(row for row in self.rows if row.gene_id not in excluded)[:k]

    def filter_queries(self, query_genes: Iterable[str]) -> QueryRankResult:
        query_ids = tuple(dict.fromkeys(_canonical_gene(gene) for gene in query_genes))
        present = tuple(sorted((self._by_gene[gene] for gene in query_ids if gene in self._by_gene)))
        missing = tuple(gene for gene in query_ids if gene not in self._by_gene)
        return QueryRankResult(self.seed_gene_id, query_ids, present, missing)

    def elbow(self) -> ElbowSummary:
        if len(self.rows) < 2:
            return ElbowSummary(
                seed_gene_id=self.seed_gene_id,
                status="insufficient_ranked_genes",
                elbow_rank_cutoff=None,
                elbow_score=None,
                retained_gene_ids=(),
                ranked_gene_count=len(self.rows),
            )

        points = [(float(row.rank), row.score) for row in self.rows]
        min_point = min(points, key=lambda point: point[0])
        max_point = max(points, key=lambda point: point[0])
        dx = max_point[0] - min_point[0]
        dy = max_point[1] - min_point[1]
        distances = [
            abs(dx * (min_point[1] - score) - dy * (min_point[0] - rank))
            for rank, score in points
        ]
        elbow_index = max(range(len(distances)), key=distances.__getitem__)
        elbow_row = self.rows[elbow_index]
        retained = tuple(row.gene_id for row in self.rows if row.rank < elbow_row.rank)
        return ElbowSummary(
            seed_gene_id=self.seed_gene_id,
            status="ok",
            elbow_rank_cutoff=elbow_row.rank,
            elbow_score=elbow_row.score,
            retained_gene_ids=retained,
            ranked_gene_count=len(self.rows),
        )


@dataclass(frozen=True)
class SeedShardRoute:
    seed_gene_id: str
    shard_id: str
    shard_index: int
    position_in_shard: int
    shard_seed_count: int
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PairShardRoute:
    gene_a: str
    gene_b: str
    status: str
    distance_available: bool
    distance_shard_id: str | None
    gene_a_shard_id: str | None
    gene_b_shard_id: str | None
    reason: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True, order=True)
class DistanceRow:
    distance: float
    gene_id: str

    def as_dict(self) -> dict[str, Any]:
        return {"gene_id": self.gene_id, "distance": self.distance}


@dataclass(frozen=True)
class DistanceComparison:
    anchor_gene_id: str
    candidate_a: DistanceRow
    candidate_b: DistanceRow
    closer_gene_id: str | None
    is_tie: bool
    rule: str = "lower_distance_is_closer"


class SpearmanDistanceShard:
    """In-memory view of one complete lower-triangular distance shard."""

    def __init__(
        self,
        *,
        shard_id: str,
        genes: Sequence[str],
        lower_triangle: Sequence[float],
        provenance: Mapping[str, Any],
    ) -> None:
        self.shard_id = str(shard_id)
        self.genes = tuple(_canonical_gene(gene) for gene in genes)
        if len(set(self.genes)) != len(self.genes):
            raise RwrArtifactError(f"Distance shard {self.shard_id} has duplicate gene labels.")
        expected = len(self.genes) * (len(self.genes) - 1) // 2
        if len(lower_triangle) != expected:
            raise RwrArtifactError(
                f"Distance shard {self.shard_id} requires {expected} lower-triangle values; "
                f"found {len(lower_triangle)}."
            )
        self._lower_triangle = array("d", (float(value) for value in lower_triangle))
        self._index = MappingProxyType({gene: index for index, gene in enumerate(self.genes)})
        self.provenance = MappingProxyType(dict(provenance))

    @staticmethod
    def _offset(row_index: int, column_index: int) -> int:
        if row_index < column_index:
            row_index, column_index = column_index, row_index
        return row_index * (row_index - 1) // 2 + column_index

    @classmethod
    def from_tsv(
        cls,
        path: Path,
        *,
        shard_id: str,
        provenance: Mapping[str, Any],
        expected_genes: Sequence[str] | None = None,
    ) -> "SpearmanDistanceShard":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if not header or header[0].strip().upper() != "INDEX":
                raise RwrArtifactError(f"Distance shard {shard_id} has no INDEX header.")
            genes = tuple(_canonical_gene(gene) for gene in header[1:])
            if expected_genes is not None:
                expected = tuple(_canonical_gene(gene) for gene in expected_genes)
                if genes != expected:
                    raise RwrArtifactError(
                        f"Distance shard {shard_id} header does not match its shard manifest."
                    )
            if len(set(genes)) != len(genes):
                raise RwrArtifactError(f"Distance shard {shard_id} has duplicate column labels.")

            gene_index = {gene: index for index, gene in enumerate(genes)}
            values: list[float | None] = [None] * (len(genes) * (len(genes) - 1) // 2)
            seen_rows: set[str] = set()
            for raw_row in reader:
                if not raw_row or not any(cell.strip() for cell in raw_row):
                    continue
                row_gene = _canonical_gene(raw_row[0])
                if row_gene not in gene_index:
                    raise RwrArtifactError(
                        f"Distance shard {shard_id} row {row_gene} is absent from the header."
                    )
                if row_gene in seen_rows:
                    raise RwrArtifactError(f"Distance shard {shard_id} repeats row {row_gene}.")
                seen_rows.add(row_gene)
                row_index = gene_index[row_gene]
                for column_index, raw_value in enumerate(raw_row[1 : len(genes) + 1]):
                    text = raw_value.strip()
                    if not text or text.upper() == "NA":
                        continue
                    try:
                        value = float(text)
                    except ValueError as error:
                        raise RwrArtifactError(
                            f"Distance shard {shard_id} has a non-numeric distance."
                        ) from error
                    if not math.isfinite(value):
                        raise RwrArtifactError(
                            f"Distance shard {shard_id} has a non-finite distance."
                        )
                    if row_index == column_index:
                        if value != 0.0:
                            raise RwrArtifactError(
                                f"Distance shard {shard_id} has a nonzero diagonal."
                            )
                        continue
                    offset = cls._offset(row_index, column_index)
                    previous = values[offset]
                    if previous is not None and not math.isclose(previous, value, rel_tol=1e-12, abs_tol=1e-15):
                        raise RwrArtifactError(
                            f"Distance shard {shard_id} has conflicting symmetric values."
                        )
                    values[offset] = value

        if seen_rows != set(genes):
            missing_rows = sorted(set(genes) - seen_rows)
            raise RwrArtifactError(
                f"Distance shard {shard_id} is missing {len(missing_rows)} matrix rows."
            )
        missing_values = sum(value is None for value in values)
        if missing_values:
            raise RwrArtifactError(
                f"Distance shard {shard_id} is missing {missing_values} lower-triangle values."
            )
        return cls(
            shard_id=shard_id,
            genes=genes,
            lower_triangle=[float(value) for value in values if value is not None],
            provenance=provenance,
        )

    def contains(self, gene_id: str) -> bool:
        return _canonical_gene(gene_id) in self._index

    def distance(self, gene_a: str, gene_b: str) -> float:
        gene_a_key = _canonical_gene(gene_a)
        gene_b_key = _canonical_gene(gene_b)
        try:
            index_a = self._index[gene_a_key]
            index_b = self._index[gene_b_key]
        except KeyError as error:
            raise KeyError(
                f"Distance shard {self.shard_id} does not contain {error.args[0]}."
            ) from error
        if index_a == index_b:
            return 0.0
        return float(self._lower_triangle[self._offset(index_a, index_b)])

    def row(self, anchor_gene_id: str) -> tuple[DistanceRow, ...]:
        anchor = _canonical_gene(anchor_gene_id)
        if anchor not in self._index:
            raise KeyError(f"Distance shard {self.shard_id} does not contain {anchor}.")
        return tuple(
            DistanceRow(self.distance(anchor, gene), gene)
            for gene in self.genes
            if gene != anchor
        )

    def closest(self, anchor_gene_id: str, k: int) -> tuple[DistanceRow, ...]:
        if k < 0:
            raise ValueError("k must be non-negative.")
        return tuple(sorted(self.row(anchor_gene_id)))[:k]

    def compare(self, anchor_gene_id: str, candidate_a: str, candidate_b: str) -> DistanceComparison:
        anchor = _canonical_gene(anchor_gene_id)
        row_a = DistanceRow(self.distance(anchor, candidate_a), _canonical_gene(candidate_a))
        row_b = DistanceRow(self.distance(anchor, candidate_b), _canonical_gene(candidate_b))
        is_tie = math.isclose(row_a.distance, row_b.distance, rel_tol=1e-12, abs_tol=1e-15)
        closer = None if is_tie else min((row_a, row_b)).gene_id
        return DistanceComparison(anchor, row_a, row_b, closer, is_tie)


@dataclass(frozen=True)
class _ShardDescriptor:
    shard_id: str
    shard_index: int
    shard_count: int
    seed_gene_ids: tuple[str, ...]
    directory: Path = field(repr=False, compare=False)


class RwrCurriculumReader:
    """Exact reader for a single versioned RWR-LOE cache context."""

    def __init__(self, context_dir: str | Path) -> None:
        self._context_dir = Path(context_dir)
        context_path = self._context_dir / "cache_context.json"
        if not context_path.is_file():
            raise FileNotFoundError(f"Missing rank-cache context metadata: {context_path}")
        raw_context = json.loads(context_path.read_text(encoding="utf-8"))
        if not isinstance(raw_context, dict):
            raise RwrArtifactError("Rank-cache context metadata must be a JSON object.")
        context_sha256 = _stable_json_sha256(raw_context)
        if self._context_dir.name.startswith("context_"):
            declared = self._context_dir.name.removeprefix("context_")
            if declared and not context_sha256.startswith(declared):
                raise RwrArtifactError("Rank-cache directory identifier does not match cache_context.json.")

        network_flist_sha256 = str(raw_context.get("network_flist_sha256", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", network_flist_sha256):
            raise RwrArtifactError("cache_context.json lacks a valid network_flist_sha256.")
        cache_id = f"context_{context_sha256[:16]}"
        self.identity = CacheIdentity(
            cache_id=cache_id,
            context_sha256=context_sha256,
            schema_version=str(raw_context.get("schema_version", "unknown")),
            network_flist_sha256=network_flist_sha256,
            parser_version=str(raw_context.get("parser_version", "unknown")),
            ranking_semantics=str(raw_context.get("ranking_semantics", "unknown")),
            rwr_hpc_build_sha256=_opaque_sha256(raw_context.get("rwr_hpc_build_id", "unknown")),
            restart=_optional_float(raw_context.get("restart")),
            delta=_optional_float(raw_context.get("delta")),
            reduction_method=_optional_string(raw_context.get("reduction_method")),
            threshold=_optional_float(raw_context.get("threshold")),
        )
        self._raw_context_sha256 = context_sha256
        self._shards, self._seed_routes = self._load_shard_index()
        self._rank_vectors: dict[str, RankVector] = {}
        self._distance_shards: dict[str, SpearmanDistanceShard] = {}

    @property
    def seed_gene_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._seed_routes))

    @property
    def shard_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._shards, key=lambda key: self._shards[key].shard_index))

    def public_provenance(self) -> dict[str, Any]:
        return self.identity.provenance()

    def _load_shard_index(self) -> tuple[dict[str, _ShardDescriptor], dict[str, SeedShardRoute]]:
        shards_root = self._context_dir / "shards"
        if not shards_root.is_dir():
            raise FileNotFoundError(f"Missing rank-cache shards directory: {shards_root}")
        descriptors: dict[str, _ShardDescriptor] = {}
        routes: dict[str, SeedShardRoute] = {}
        declared_shard_count: int | None = None
        for shard_dir in sorted(path for path in shards_root.iterdir() if path.is_dir()):
            manifest_path = shard_dir / "rank_cache_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise RwrArtifactError(f"Shard manifest {shard_dir.name} must be a JSON object.")
            nested_context = manifest.get("cache_context")
            if not isinstance(nested_context, dict) or _stable_json_sha256(nested_context) != self._raw_context_sha256:
                raise RwrArtifactError(f"Shard {shard_dir.name} belongs to a different cache context.")
            shard_index = int(manifest["shard_index"])
            shard_count = int(manifest["shard_count"])
            if shard_index < 0 or shard_index >= shard_count:
                raise RwrArtifactError(f"Shard {shard_dir.name} has an invalid shard index.")
            if shard_dir.name != f"shard_{shard_index:05d}":
                raise RwrArtifactError(f"Shard directory {shard_dir.name} disagrees with its manifest.")
            if declared_shard_count is None:
                declared_shard_count = shard_count
            elif declared_shard_count != shard_count:
                raise RwrArtifactError("Shard manifests disagree about shard_count.")
            seed_ids = tuple(_canonical_gene(gene) for gene in manifest.get("seed_gene_ids", []))
            if len(set(seed_ids)) != len(seed_ids):
                raise RwrArtifactError(f"Shard {shard_dir.name} repeats a seed gene.")
            descriptor = _ShardDescriptor(
                shard_id=shard_dir.name,
                shard_index=shard_index,
                shard_count=shard_count,
                seed_gene_ids=seed_ids,
                directory=shard_dir,
            )
            descriptors[descriptor.shard_id] = descriptor
            for position, seed in enumerate(seed_ids):
                if seed in routes:
                    raise RwrArtifactError(f"Seed {seed} is assigned to multiple distance shards.")
                routes[seed] = SeedShardRoute(
                    seed_gene_id=seed,
                    shard_id=descriptor.shard_id,
                    shard_index=shard_index,
                    position_in_shard=position,
                    shard_seed_count=len(seed_ids),
                    provenance=MappingProxyType(
                        self.identity.provenance(
                            artifact_type="rwr_loe_distance_shard_route",
                            shard_id=descriptor.shard_id,
                        )
                    ),
                )
        if not descriptors:
            raise RwrArtifactError("No rank-cache shard manifests were found.")
        if declared_shard_count != len(descriptors):
            raise RwrArtifactError(
                f"Expected {declared_shard_count} shard manifests; found {len(descriptors)}."
            )
        return descriptors, routes

    def route_seed(self, seed_gene_id: str) -> SeedShardRoute:
        seed = _canonical_gene(seed_gene_id)
        route = self._seed_routes.get(seed)
        if route is None:
            raise UnknownSeedError(f"No RWR-LOE cache artifact exists for seed {seed}.")
        return route

    def route_distance_pair(self, gene_a: str, gene_b: str) -> PairShardRoute:
        gene_a_key = _canonical_gene(gene_a)
        gene_b_key = _canonical_gene(gene_b)
        route_a = self._seed_routes.get(gene_a_key)
        route_b = self._seed_routes.get(gene_b_key)
        if gene_a_key == gene_b_key and route_a is not None:
            status = "same_gene"
            available = True
            shard_id = route_a.shard_id
            reason = "The diagonal distance is exactly zero."
        elif route_a is None or route_b is None:
            status = "unknown_seed"
            available = False
            shard_id = None
            reason = "At least one gene is absent from the materialized seed-shard index."
        elif route_a.shard_id == route_b.shard_id:
            status = "same_shard"
            available = True
            shard_id = route_a.shard_id
            reason = "Both seed encodings are present in the same lower-triangular shard."
        else:
            status = "cross_shard_not_materialized"
            available = False
            shard_id = None
            reason = (
                "The current precompute contains within-shard distances only; no existing "
                "distance shard contains this cross-shard pair."
            )
        return PairShardRoute(
            gene_a=gene_a_key,
            gene_b=gene_b_key,
            status=status,
            distance_available=available,
            distance_shard_id=shard_id,
            gene_a_shard_id=route_a.shard_id if route_a else None,
            gene_b_shard_id=route_b.shard_id if route_b else None,
            reason=reason,
            provenance=MappingProxyType(
                self.identity.provenance(artifact_type="rwr_loe_distance_shard_route")
            ),
        )

    def seed_metadata(self, seed_gene_id: str) -> SeedRankMetadata:
        seed = _canonical_gene(seed_gene_id)
        route = self.route_seed(seed)
        metadata_path = self._context_dir / "ranks" / f"{rank_cache_file_stem(seed)}.metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing rank metadata for seed {seed}: {metadata_path}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _canonical_gene(raw.get("seed_gene_id", "")) != seed:
            raise RwrArtifactError(f"Rank metadata for {seed} declares a different seed.")
        nested_context = raw.get("cache_context")
        if not isinstance(nested_context, dict) or _stable_json_sha256(nested_context) != self._raw_context_sha256:
            raise RwrArtifactError(f"Rank metadata for {seed} belongs to a different cache context.")
        shard_index = int(raw["shard_index"])
        shard_count = int(raw["shard_count"])
        if shard_index != route.shard_index or f"shard_{shard_index:05d}" != route.shard_id:
            raise RwrArtifactError(f"Rank metadata and shard manifest disagree for seed {seed}.")
        if shard_count != len(self._shards):
            raise RwrArtifactError(f"Rank metadata has the wrong shard count for seed {seed}.")
        return SeedRankMetadata(
            seed_gene_id=seed,
            ranked_gene_count=int(raw["ranked_gene_count"]),
            shard_id=route.shard_id,
            shard_index=shard_index,
            shard_count=shard_count,
            created_at=_optional_string(raw.get("created_at")),
            provenance=MappingProxyType(
                self.identity.provenance(
                    artifact_type="rwr_loe_full_rank_vector",
                    seed_gene_id=seed,
                    shard_id=route.shard_id,
                )
            ),
        )

    def rank_vector(self, seed_gene_id: str) -> RankVector:
        seed = _canonical_gene(seed_gene_id)
        cached = self._rank_vectors.get(seed)
        if cached is not None:
            return cached
        metadata = self.seed_metadata(seed)
        rank_path = self._context_dir / "ranks" / f"{rank_cache_file_stem(seed)}.ranks.tsv.gz"
        if not rank_path.is_file():
            raise FileNotFoundError(f"Missing full rank vector for seed {seed}: {rank_path}")
        opener = gzip.open if rank_path.suffix == ".gz" else open
        rows: list[RankRow] = []
        with opener(rank_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"NodeNames", "Scores", "rank"}
            if not required.issubset(reader.fieldnames or ()):
                raise RwrArtifactError(f"Rank vector for {seed} lacks required TSV columns.")
            for raw in reader:
                gene = _canonical_gene(raw["NodeNames"])
                try:
                    score = float(raw["Scores"])
                    numeric_rank = float(raw["rank"])
                except (TypeError, ValueError) as error:
                    raise RwrArtifactError(f"Rank vector for {seed} contains non-numeric data.") from error
                if not math.isfinite(score) or not numeric_rank.is_integer() or numeric_rank < 1:
                    raise RwrArtifactError(f"Rank vector for {seed} contains an invalid rank or score.")
                rows.append(RankRow(rank=int(numeric_rank), gene_id=gene, score=score))
        rows.sort()
        if seed in {row.gene_id for row in rows}:
            raise RwrArtifactError(f"Rank vector for {seed} does not exclude its seed.")
        if len(rows) != metadata.ranked_gene_count:
            raise RwrArtifactError(
                f"Rank vector for {seed} declares {metadata.ranked_gene_count} rows but contains {len(rows)}."
            )
        vector = RankVector(seed_gene_id=seed, rows=tuple(rows), metadata=metadata)
        self._rank_vectors[seed] = vector
        return vector

    def rank(self, seed_gene_id: str, gene_id: str) -> RankRow:
        return self.rank_vector(seed_gene_id).require(gene_id)

    def compare_ranks(self, seed_gene_id: str, candidate_a: str, candidate_b: str) -> RankComparison:
        return self.rank_vector(seed_gene_id).compare(candidate_a, candidate_b)

    def top_k(
        self,
        seed_gene_id: str,
        k: int,
        *,
        exclude_genes: Iterable[str] = (),
    ) -> tuple[RankRow, ...]:
        return self.rank_vector(seed_gene_id).top_k(k, exclude_genes=exclude_genes)

    def filter_queries(self, seed_gene_id: str, query_genes: Iterable[str]) -> QueryRankResult:
        return self.rank_vector(seed_gene_id).filter_queries(query_genes)

    def elbow(self, seed_gene_id: str) -> ElbowSummary:
        return self.rank_vector(seed_gene_id).elbow()

    def top_k_intersection(
        self,
        seed_gene_id_a: str,
        seed_gene_id_b: str,
        k: int,
        *,
        exclude_both_seeds: bool = True,
    ) -> TopKIntersection:
        seed_a = _canonical_gene(seed_gene_id_a)
        seed_b = _canonical_gene(seed_gene_id_b)
        excluded: set[str] = {seed_a, seed_b} if exclude_both_seeds else set()
        top_a = self.top_k(seed_a, k, exclude_genes=excluded)
        top_b = self.top_k(seed_b, k, exclude_genes=excluded)
        genes_a = tuple(row.gene_id for row in top_a)
        genes_b = tuple(row.gene_id for row in top_b)
        set_a = set(genes_a)
        set_b = set(genes_b)
        intersection = tuple(sorted(set_a & set_b))
        union_size = len(set_a | set_b)
        return TopKIntersection(
            seed_gene_id_a=seed_a,
            seed_gene_id_b=seed_b,
            top_k=k,
            neighborhood_a=genes_a,
            neighborhood_b=genes_b,
            intersection_gene_ids=intersection,
            intersection_size=len(intersection),
            union_size=union_size,
            jaccard=(len(intersection) / union_size if union_size else 0.0),
        )

    def distance_shard(self, shard_id: str) -> SpearmanDistanceShard:
        key = str(shard_id)
        cached = self._distance_shards.get(key)
        if cached is not None:
            return cached
        descriptor = self._shards.get(key)
        if descriptor is None:
            raise KeyError(f"Unknown RWR-LOE distance shard {key}.")
        matches = sorted(
            (descriptor.directory / "rwr_output").glob("*_spearman_dist_matrix.tsv")
        )
        if len(matches) != 1:
            raise RwrArtifactError(
                f"Distance shard {key} requires exactly one Spearman matrix; found {len(matches)}."
            )
        shard = SpearmanDistanceShard.from_tsv(
            matches[0],
            shard_id=key,
            expected_genes=descriptor.seed_gene_ids,
            provenance=self.identity.provenance(
                artifact_type="rwr_loe_spearman_distance_matrix",
                shard_id=key,
                distance_metric="spearman_distance",
                lower_is_closer=True,
                matrix_layout="lower_triangular",
            ),
        )
        self._distance_shards[key] = shard
        return shard

    def distance(self, gene_a: str, gene_b: str) -> float:
        route = self.route_distance_pair(gene_a, gene_b)
        if not route.distance_available or route.distance_shard_id is None:
            raise CrossShardDistanceUnavailable(route)
        if route.gene_a == route.gene_b:
            return 0.0
        return self.distance_shard(route.distance_shard_id).distance(route.gene_a, route.gene_b)

    def compare_distances(
        self,
        anchor_gene_id: str,
        candidate_a: str,
        candidate_b: str,
    ) -> DistanceComparison:
        anchor = _canonical_gene(anchor_gene_id)
        route_a = self.route_distance_pair(anchor, candidate_a)
        route_b = self.route_distance_pair(anchor, candidate_b)
        if (
            not route_a.distance_available
            or not route_b.distance_available
            or route_a.distance_shard_id != route_b.distance_shard_id
            or route_a.distance_shard_id is None
        ):
            unavailable = route_a if not route_a.distance_available else route_b
            raise CrossShardDistanceUnavailable(unavailable)
        return self.distance_shard(route_a.distance_shard_id).compare(anchor, candidate_a, candidate_b)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RwrArtifactError("Cache context contains a non-finite numeric parameter.")
    return parsed


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "CacheIdentity",
    "CrossShardDistanceUnavailable",
    "DistanceComparison",
    "DistanceRow",
    "ElbowSummary",
    "PairShardRoute",
    "QueryRankResult",
    "RankComparison",
    "RankRow",
    "RankVector",
    "RwrArtifactError",
    "RwrCurriculumReader",
    "SeedRankMetadata",
    "SeedShardRoute",
    "SpearmanDistanceShard",
    "TopKIntersection",
    "UnknownRankedGeneError",
    "UnknownSeedError",
    "rank_cache_file_stem",
]
