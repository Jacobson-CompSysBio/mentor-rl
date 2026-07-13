#!/usr/bin/env python3
"""Read-only, memory-mapped oracle for a MENTOR multiplex CSR store.

The pre-trajectory curriculum must never infer full-graph facts from a prefix
sample of text edge lists.  This module reads the same binary CSR artifact as
the compiled runtime and exposes deterministic primitives for dataset
generation and validation without materializing the 49M-edge aggregate graph
in NetworkX.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


SUPPORTED_FORMAT_VERSION = "mentor-rl-multiplex-store-v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_binary_string_table(data_path: Path, offsets_path: Path) -> tuple[str, ...]:
    offsets = np.memmap(offsets_path, dtype=np.uint64, mode="r")
    if len(offsets) == 0:
        return ()
    data = data_path.read_bytes()
    if int(offsets[-1]) != len(data):
        raise ValueError(f"String-table offsets do not match {data_path}.")
    return tuple(
        data[int(start) : int(end)].decode("utf-8")
        for start, end in zip(offsets[:-1], offsets[1:])
    )


@dataclass(frozen=True)
class EdgeFact:
    source_gene_id: str
    target_gene_id: str
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_gene_id": self.source_gene_id,
            "target_gene_id": self.target_gene_id,
            "weight": self.weight,
        }


class CsrOracleView:
    """One lazily memory-mapped CSR adjacency matrix."""

    def __init__(
        self,
        *,
        name: str,
        indptr_path: Path,
        indices_path: Path,
        weights_path: Path,
        gene_count: int,
    ) -> None:
        self.name = name
        self.indptr_path = indptr_path
        self.indices_path = indices_path
        self.weights_path = weights_path
        self.indptr = np.memmap(indptr_path, dtype=np.uint64, mode="r")
        self.indices = np.memmap(indices_path, dtype=np.uint32, mode="r")
        self.weights = np.memmap(weights_path, dtype=np.float32, mode="r")
        if len(self.indptr) != gene_count + 1:
            raise ValueError(
                f"CSR indptr for {name!r} has {len(self.indptr)} entries; "
                f"expected {gene_count + 1}."
            )
        if len(self.indices) != len(self.weights) or int(self.indptr[-1]) != len(self.indices):
            raise ValueError(f"CSR arrays for {name!r} have inconsistent lengths.")

    def bounds(self, node_index: int) -> tuple[int, int]:
        return int(self.indptr[node_index]), int(self.indptr[node_index + 1])

    def degree(self, node_index: int) -> int:
        start, end = self.bounds(node_index)
        return end - start

    def neighbor_indices(self, node_index: int) -> np.ndarray:
        start, end = self.bounds(node_index)
        return self.indices[start:end]

    def neighbor_weights(self, node_index: int) -> np.ndarray:
        start, end = self.bounds(node_index)
        return self.weights[start:end]

    def edge_weight(self, source_index: int, target_index: int) -> float | None:
        neighbors = self.neighbor_indices(source_index)
        offset = int(np.searchsorted(neighbors, target_index))
        if offset >= len(neighbors) or int(neighbors[offset]) != target_index:
            return None
        start, _ = self.bounds(source_index)
        return float(self.weights[start + offset])

    def nonisolated_node_indices(self) -> np.ndarray:
        return np.flatnonzero(np.diff(self.indptr) > 0)


class MultiplexStoreOracle:
    """Authoritative graph-fact reader backed by the full binary store."""

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.manifest_path = self.store_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing multiplex manifest: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != SUPPORTED_FORMAT_VERSION:
            raise ValueError(
                "Unsupported multiplex-store format: "
                f"{self.manifest.get('format_version')!r}."
            )
        dtypes = self.manifest.get("dtypes", {})
        expected_dtypes = {"indices": "uint32", "indptr": "uint64", "weights": "float32"}
        if dtypes != expected_dtypes:
            raise ValueError(f"Unexpected multiplex CSR dtypes: {dtypes!r}.")

        files = self.manifest.get("files", {})
        genes = files.get("binary_metadata", {}).get("genes", {})
        self.gene_ids = _read_binary_string_table(
            self.store_dir / str(genes.get("data_file", "genes_data.bin")),
            self.store_dir / str(genes.get("offsets_file", "genes_offsets.bin")),
        )
        if len(self.gene_ids) != int(self.manifest.get("num_genes", -1)):
            raise ValueError("Gene-table length does not match the multiplex manifest.")
        self.gene_to_index = {gene_id: index for index, gene_id in enumerate(self.gene_ids)}

        layer_rows = self.manifest.get("layers")
        if not isinstance(layer_rows, list) or len(layer_rows) != int(self.manifest.get("num_layers", -1)):
            raise ValueError("Layer metadata does not match the multiplex manifest.")
        self.layer_rows = tuple(dict(row) for row in layer_rows)
        self.layer_names = tuple(str(row["layer_name"]) for row in self.layer_rows)
        if len(set(self.layer_names)) != len(self.layer_names):
            raise ValueError("Multiplex layer names must be unique.")
        self.layer_to_row = {str(row["layer_name"]): row for row in self.layer_rows}
        self._views: dict[str, CsrOracleView] = {}
        self.store_id = f"sha256:{_sha256_file(self.manifest_path)}"

    @property
    def gene_count(self) -> int:
        return len(self.gene_ids)

    @property
    def layer_count(self) -> int:
        return len(self.layer_names)

    @property
    def aggregate_edge_count(self) -> int:
        return int(self.manifest["files"]["aggregate"]["undirected_edge_count"])

    def _gene_index(self, gene_id: str) -> int:
        try:
            return self.gene_to_index[gene_id]
        except KeyError as exc:
            raise KeyError(f"Unknown multiplex gene ID: {gene_id}") from exc

    def view(self, layer_name: str | None = None) -> CsrOracleView:
        cache_key = layer_name or "__aggregate__"
        cached = self._views.get(cache_key)
        if cached is not None:
            return cached
        if layer_name is None:
            row = self.manifest["files"]["aggregate"]
            name = "aggregate_multiplex"
        else:
            if layer_name not in self.layer_to_row:
                raise KeyError(f"Unknown multiplex layer: {layer_name}")
            row = self.layer_to_row[layer_name]
            name = layer_name
        result = CsrOracleView(
            name=name,
            indptr_path=self.store_dir / str(row["indptr_file"]),
            indices_path=self.store_dir / str(row["indices_file"]),
            weights_path=self.store_dir / str(row["weights_file"]),
            gene_count=self.gene_count,
        )
        self._views[cache_key] = result
        return result

    def has_edge(self, source_gene_id: str, target_gene_id: str, *, layer: str | None = None) -> bool:
        return self.edge_weight(source_gene_id, target_gene_id, layer=layer) is not None

    def edge_weight(
        self,
        source_gene_id: str,
        target_gene_id: str,
        *,
        layer: str | None = None,
    ) -> float | None:
        return self.view(layer).edge_weight(
            self._gene_index(source_gene_id), self._gene_index(target_gene_id)
        )

    def degree(self, gene_id: str, *, layer: str | None = None) -> int:
        return self.view(layer).degree(self._gene_index(gene_id))

    def neighbors(
        self,
        gene_id: str,
        *,
        layer: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        node_index = self._gene_index(gene_id)
        view = self.view(layer)
        neighbor_indices = view.neighbor_indices(node_index)
        weights = view.neighbor_weights(node_index)
        if limit is not None:
            neighbor_indices = neighbor_indices[:limit]
            weights = weights[:limit]
        return [
            {"gene_id": self.gene_ids[int(index)], "weight": float(weight)}
            for index, weight in zip(neighbor_indices, weights)
        ]

    def gene_layers(self, gene_id: str) -> list[str]:
        node_index = self._gene_index(gene_id)
        return [name for name in self.layer_names if self.view(name).degree(node_index) > 0]

    def edge_layers(self, source_gene_id: str, target_gene_id: str) -> list[dict[str, Any]]:
        source_index = self._gene_index(source_gene_id)
        target_index = self._gene_index(target_gene_id)
        rows: list[dict[str, Any]] = []
        for layer_name in self.layer_names:
            weight = self.view(layer_name).edge_weight(source_index, target_index)
            if weight is not None:
                rows.append({"layer_id": layer_name, "weight": weight})
        return rows

    def layer_metadata(self, layer_name: str) -> dict[str, Any]:
        row = self.layer_to_row[layer_name]
        parts = layer_name.split(":")
        return {
            "layer_id": layer_name,
            "layer_family": parts[0],
            "layer_tags": parts[1:],
            "node_count": int(row["node_count"]),
            "edge_count": int(row["undirected_edge_count"]),
        }

    def sample_node_indices(
        self,
        *,
        layer: str | None,
        count: int,
        seed: int,
        minimum_degree: int = 1,
    ) -> list[int]:
        view = self.view(layer)
        degrees = np.diff(view.indptr)
        eligible = np.flatnonzero(degrees >= minimum_degree)
        if not len(eligible) or count <= 0:
            return []
        rng = np.random.default_rng(seed)
        selected = rng.choice(eligible, size=min(count, len(eligible)), replace=False)
        return [int(index) for index in selected]

    def sample_edges(
        self,
        *,
        layer: str | None,
        count: int,
        seed: int,
    ) -> list[EdgeFact]:
        view = self.view(layer)
        rng = np.random.default_rng(seed)
        candidates = self.sample_node_indices(
            layer=layer,
            count=max(count * 4, count),
            seed=seed,
            minimum_degree=1,
        )
        result: list[EdgeFact] = []
        seen: set[tuple[int, int]] = set()
        for source_index in candidates:
            neighbors = view.neighbor_indices(source_index)
            weights = view.neighbor_weights(source_index)
            order = rng.permutation(len(neighbors))
            for offset in order:
                target_index = int(neighbors[int(offset)])
                pair = tuple(sorted((source_index, target_index)))
                if pair in seen:
                    continue
                seen.add(pair)
                result.append(
                    EdgeFact(
                        source_gene_id=self.gene_ids[pair[0]],
                        target_gene_id=self.gene_ids[pair[1]],
                        weight=float(weights[int(offset)]),
                    )
                )
                break
            if len(result) >= count:
                break
        return result

    def sample_nonedges(
        self,
        *,
        layer: str | None,
        count: int,
        seed: int,
        require_present: bool = True,
    ) -> list[tuple[str, str]]:
        view = self.view(layer)
        eligible = (
            view.nonisolated_node_indices()
            if require_present
            else np.arange(self.gene_count, dtype=np.int64)
        )
        rng = np.random.default_rng(seed)
        result: list[tuple[str, str]] = []
        seen: set[tuple[int, int]] = set()
        max_attempts = max(100, count * 100)
        for _ in range(max_attempts):
            source_index, target_index = (
                int(value) for value in rng.choice(eligible, size=2, replace=False)
            )
            pair = tuple(sorted((source_index, target_index)))
            if pair in seen or view.edge_weight(*pair) is not None:
                continue
            seen.add(pair)
            result.append((self.gene_ids[pair[0]], self.gene_ids[pair[1]]))
            if len(result) >= count:
                break
        return result

    def common_neighbors(
        self,
        gene_a: str,
        gene_b: str,
        *,
        layer: str | None = None,
    ) -> list[str]:
        view = self.view(layer)
        left = view.neighbor_indices(self._gene_index(gene_a))
        right = view.neighbor_indices(self._gene_index(gene_b))
        return [self.gene_ids[int(index)] for index in np.intersect1d(left, right, assume_unique=True)]

    def induced_edges(
        self,
        gene_ids: Sequence[str],
        *,
        layer: str | None = None,
    ) -> list[EdgeFact]:
        unique = sorted(set(gene_ids))
        result: list[EdgeFact] = []
        for offset, source in enumerate(unique):
            for target in unique[offset + 1 :]:
                weight = self.edge_weight(source, target, layer=layer)
                if weight is not None:
                    result.append(EdgeFact(source, target, weight))
        return result

    def induced_components(
        self,
        gene_ids: Sequence[str],
        *,
        layer: str | None = None,
    ) -> list[list[str]]:
        nodes = sorted(set(gene_ids))
        adjacency = {node: set() for node in nodes}
        for edge in self.induced_edges(nodes, layer=layer):
            adjacency[edge.source_gene_id].add(edge.target_gene_id)
            adjacency[edge.target_gene_id].add(edge.source_gene_id)
        components: list[list[str]] = []
        unseen = set(nodes)
        while unseen:
            root = min(unseen)
            queue = [root]
            component: list[str] = []
            unseen.remove(root)
            while queue:
                node = queue.pop()
                component.append(node)
                for neighbor in sorted(adjacency[node] & unseen):
                    unseen.remove(neighbor)
                    queue.append(neighbor)
            components.append(sorted(component))
        return sorted(components, key=lambda values: (values[0], len(values)))

    def shortest_path(
        self,
        source_gene_id: str,
        target_gene_id: str,
        *,
        layer: str | None = None,
        max_hops: int | None = None,
        max_visited: int = 100_000,
    ) -> list[str] | None:
        """Return a deterministic unweighted shortest path using bidirectional BFS."""

        source_index = self._gene_index(source_gene_id)
        target_index = self._gene_index(target_gene_id)
        if source_index == target_index:
            return [source_gene_id]
        view = self.view(layer)
        if view.degree(source_index) == 0 or view.degree(target_index) == 0:
            return None

        left_parent: dict[int, int | None] = {source_index: None}
        right_parent: dict[int, int | None] = {target_index: None}
        left_frontier = {source_index}
        right_frontier = {target_index}
        depth = 0

        def expand(
            frontier: set[int],
            parents: dict[int, int | None],
            other: dict[int, int | None],
        ) -> tuple[set[int], int | None]:
            next_frontier: set[int] = set()
            for node in sorted(frontier):
                for raw_neighbor in view.neighbor_indices(node):
                    neighbor = int(raw_neighbor)
                    if neighbor in parents:
                        continue
                    parents[neighbor] = node
                    if neighbor in other:
                        return next_frontier, neighbor
                    next_frontier.add(neighbor)
                    if len(parents) + len(other) > max_visited:
                        raise RuntimeError("Shortest-path query exceeded max_visited.")
            return next_frontier, None

        meeting: int | None = None
        while left_frontier and right_frontier:
            if max_hops is not None and depth >= max_hops:
                return None
            if len(left_frontier) <= len(right_frontier):
                left_frontier, meeting = expand(left_frontier, left_parent, right_parent)
            else:
                right_frontier, meeting = expand(right_frontier, right_parent, left_parent)
            depth += 1
            if meeting is not None:
                break
        if meeting is None:
            return None

        left_path: list[int] = []
        node: int | None = meeting
        while node is not None:
            left_path.append(node)
            node = left_parent.get(node)
        left_path.reverse()
        right_path: list[int] = []
        node = right_parent.get(meeting)
        while node is not None:
            right_path.append(node)
            node = right_parent.get(node)
        return [self.gene_ids[index] for index in left_path + right_path]

    def sample_nontrivial_paths(
        self,
        *,
        layer: str | None,
        count: int,
        seed: int,
        minimum_hops: int = 2,
        maximum_hops: int = 4,
    ) -> list[list[str]]:
        """Sample exact paths while explicitly rejecting direct-edge degeneracy."""

        view = self.view(layer)
        rng = np.random.default_rng(seed)
        starts = self.sample_node_indices(
            layer=layer,
            count=max(count * 12, count),
            seed=seed,
            minimum_degree=2,
        )
        result: list[list[str]] = []
        seen: set[tuple[str, str]] = set()
        for source_index in starts:
            first_neighbors = view.neighbor_indices(source_index)
            if not len(first_neighbors):
                continue
            middle_index = int(first_neighbors[int(rng.integers(len(first_neighbors)))])
            second_neighbors = view.neighbor_indices(middle_index)
            if not len(second_neighbors):
                continue
            for raw_target in second_neighbors[rng.permutation(len(second_neighbors))[:32]]:
                target_index = int(raw_target)
                if target_index == source_index or view.edge_weight(source_index, target_index) is not None:
                    continue
                source = self.gene_ids[source_index]
                target = self.gene_ids[target_index]
                pair = tuple(sorted((source, target)))
                if pair in seen:
                    continue
                path = self.shortest_path(
                    source,
                    target,
                    layer=layer,
                    max_hops=maximum_hops,
                )
                if path is None or not (minimum_hops <= len(path) - 1 <= maximum_hops):
                    continue
                seen.add(pair)
                result.append(path)
                break
            if len(result) >= count:
                break
        return result

    def validate_edge_fact(self, fact: EdgeFact, *, layer: str | None = None, atol: float = 1e-6) -> bool:
        observed = self.edge_weight(fact.source_gene_id, fact.target_gene_id, layer=layer)
        return observed is not None and abs(observed - fact.weight) <= atol


def portable_source_path(path: str | Path, *, repo_root: Path) -> str:
    """Return a repository-relative provenance path or a logical basename."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return f"external://{candidate.name}"


def iter_gene_pairs(values: Sequence[str]) -> Iterator[tuple[str, str]]:
    unique = sorted(set(values))
    for offset, source in enumerate(unique):
        for target in unique[offset + 1 :]:
            yield source, target
