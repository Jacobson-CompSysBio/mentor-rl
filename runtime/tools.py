"""Simple deterministic tool implementations for the MENTOR-RL runtime.

This file keeps the runtime tools easy to read:

- graph tools operate on a small multiplex index
- MyGene lookup can use a local cache and only uses the network if allowed
- every tool returns a payload plus provenance metadata

These functions do not know about `ToolObservation`. They only compute results.
The environment wrapper turns these results into runtime observations.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from utils.multiplex import Multiplex


DEFAULT_RESTART_PROBABILITY = 0.35
DEFAULT_TOP_K = 20
DEFAULT_MYGENE_FIELDS = ("symbol", "name", "entrezgene", "uniprot", "ensembl")


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot complete its work cleanly."""


@dataclass(frozen=True)
class ToolExecutionResult:
    """Internal container for one successful tool execution."""

    payload: dict[str, Any]
    provenance: dict[str, Any]
    is_empty: bool = False


@dataclass(frozen=True)
class MultiplexIndex:
    """Small read-only view over the multiplex graphs used by the runtime."""

    layer_graphs: dict[str, nx.Graph]
    aggregate_graph: nx.Graph
    gene_ids: set[str]
    layer_names: tuple[str, ...]

    @classmethod
    def from_multiplex(cls, multiplex: Multiplex) -> "MultiplexIndex":
        """Build a searchable index from the existing multiplex helper."""

        layer_graphs: dict[str, nx.Graph] = {}
        aggregate_graph = nx.Graph()
        gene_ids: set[str] = set()
        layer_names: list[str] = []

        for layer in multiplex.layers:
            layer_name = str(layer["layer_name"])
            graph = layer["graph"]
            layer_graphs[layer_name] = graph
            layer_names.append(layer_name)
            gene_ids.update(str(node) for node in graph.nodes())

            for node in graph.nodes():
                aggregate_graph.add_node(str(node))
            for source, target, data in graph.edges(data=True):
                edge_weight = float(data.get("weight", 1.0))
                if aggregate_graph.has_edge(source, target):
                    aggregate_graph[source][target]["weight"] += edge_weight
                    aggregate_graph[source][target]["layers"].append(layer_name)
                else:
                    aggregate_graph.add_edge(
                        str(source),
                        str(target),
                        weight=edge_weight,
                        layers=[layer_name],
                    )

        return cls(
            layer_graphs=layer_graphs,
            aggregate_graph=aggregate_graph,
            gene_ids=gene_ids,
            layer_names=tuple(layer_names),
        )

    @classmethod
    def from_flist(cls, flist_path: str) -> "MultiplexIndex":
        """Load a multiplex from an flist and index it."""

        return cls.from_multiplex(Multiplex(flist=flist_path))


def build_multiplex_index(multiplex: Multiplex) -> MultiplexIndex:
    """Build the runtime graph index used by the tool layer."""

    return MultiplexIndex.from_multiplex(multiplex)


def load_mygene_cache(cache_path: str | None) -> dict[str, list[dict[str, Any]]]:
    """Load a simple JSON cache for MyGene responses."""

    if not cache_path:
        return {}

    path = Path(cache_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ToolExecutionError("MyGene cache file must contain a JSON object.")

    cache: dict[str, list[dict[str, Any]]] = {}
    for query, hits in payload.items():
        if not isinstance(query, str) or not isinstance(hits, list):
            continue
        clean_hits = [hit for hit in hits if isinstance(hit, dict)]
        cache[query] = clean_hits
    return cache


def save_mygene_cache(cache: dict[str, list[dict[str, Any]]], cache_path: str | None) -> None:
    """Write the MyGene cache back to disk."""

    if not cache_path:
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, sort_keys=True, indent=2)


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _resolve_layer_names(index: MultiplexIndex, layers: list[str] | None = None) -> list[str]:
    if layers is None:
        return list(index.layer_names)

    resolved = _unique_preserving_order([str(layer) for layer in layers])
    unknown = [layer for layer in resolved if layer not in index.layer_graphs]
    if unknown:
        raise ToolExecutionError("Unknown layers requested: " + ", ".join(sorted(unknown)) + ".")
    return resolved


def _serialize_edges(graph: nx.Graph, allowed_nodes: set[str]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for source, target, data in graph.edges(data=True):
        if source not in allowed_nodes or target not in allowed_nodes:
            continue
        edges.append(
            {
                "source_gene_id": str(source),
                "target_gene_id": str(target),
                "weight": float(data.get("weight", 1.0)),
            }
        )
    edges.sort(key=lambda item: (item["source_gene_id"], item["target_gene_id"]))
    return edges


def _personalized_pagerank(
    graph: nx.Graph,
    seeds: list[str],
    *,
    top_k: int,
    restart_probability: float,
) -> list[dict[str, Any]]:
    active_seeds = [seed for seed in seeds if seed in graph]
    if not active_seeds:
        return []

    personalization = {node: 0.0 for node in graph.nodes()}
    seed_mass = 1.0 / len(active_seeds)
    for seed in active_seeds:
        personalization[seed] = seed_mass

    alpha = 1.0 - restart_probability
    scores = nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        weight="weight",
    )

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    limited = ranked[:top_k]
    return [{"gene_id": gene_id, "score": float(score)} for gene_id, score in limited]


def _filter_mygene_hit(hit: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
    if not fields:
        fields = list(DEFAULT_MYGENE_FIELDS)

    filtered = {"query": hit.get("query"), "_id": hit.get("_id")}
    for field_name in fields:
        if field_name in hit:
            filtered[field_name] = hit[field_name]
    return filtered


def _fetch_mygene_hits(query: str, fields: list[str] | None) -> list[dict[str, Any]]:
    query_params = {
        "q": query,
        "species": "human",
        "size": "10",
        "fields": ",".join(fields or DEFAULT_MYGENE_FIELDS),
    }
    url = "https://mygene.info/v3/query?" + urllib.parse.urlencode(query_params)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hits = payload.get("hits", [])
    if not isinstance(hits, list):
        raise ToolExecutionError("MyGene response did not contain a valid 'hits' list.")
    return [hit for hit in hits if isinstance(hit, dict)]


def query_mygene(
    query: str,
    *,
    fields: list[str] | None = None,
    cache: dict[str, list[dict[str, Any]]] | None = None,
    cache_path: str | None = None,
    allow_network: bool = False,
) -> ToolExecutionResult:
    """Resolve one MyGene query using a local cache first."""

    if not isinstance(query, str) or not query:
        raise ToolExecutionError("query_mygene requires a non-empty query string.")

    if cache is None:
        cache = {}

    provenance_source = "cache"
    if query in cache:
        hits = cache[query]
    elif allow_network:
        hits = _fetch_mygene_hits(query, fields)
        cache[query] = hits
        save_mygene_cache(cache, cache_path)
        provenance_source = "network"
    else:
        hits = []
        provenance_source = "cache_miss"

    filtered_hits = [_filter_mygene_hit(hit, fields) for hit in hits]
    payload = {
        "query": query,
        "requested_fields": list(fields or DEFAULT_MYGENE_FIELDS),
        "results": filtered_hits,
        "result_count": len(filtered_hits),
    }
    provenance = {
        "tool_name": "query_mygene",
        "source": provenance_source,
        "network_used": provenance_source == "network",
        "cache_hit": provenance_source == "cache",
    }
    return ToolExecutionResult(
        payload=payload,
        provenance=provenance,
        is_empty=len(filtered_hits) == 0,
    )


def get_neighbors(
    index: MultiplexIndex,
    gene: str,
    *,
    layers: list[str] | None = None,
) -> ToolExecutionResult:
    """Return first-hop neighbors for one gene across one or more layers."""

    selected_layers = _resolve_layer_names(index, layers)
    unique_neighbors: set[str] = set()
    layer_results: list[dict[str, Any]] = []

    for layer_name in selected_layers:
        graph = index.layer_graphs[layer_name]
        neighbors = sorted(str(neighbor) for neighbor in graph.neighbors(gene)) if gene in graph else []
        unique_neighbors.update(neighbors)
        layer_results.append(
            {
                "layer_name": layer_name,
                "neighbors": neighbors,
                "neighbor_count": len(neighbors),
            }
        )

    payload = {
        "query_gene_id": gene,
        "layers": layer_results,
        "unique_neighbors": sorted(unique_neighbors),
        "unique_neighbor_count": len(unique_neighbors),
    }
    provenance = {
        "tool_name": "get_neighbors",
        "queried_layers": selected_layers,
    }
    return ToolExecutionResult(
        payload=payload,
        provenance=provenance,
        is_empty=len(unique_neighbors) == 0,
    )


def induce_subgraph(
    index: MultiplexIndex,
    genes: list[str],
    *,
    layers: list[str] | None = None,
) -> ToolExecutionResult:
    """Return the induced subgraph on a given gene set."""

    selected_layers = _resolve_layer_names(index, layers)
    query_gene_ids = _unique_preserving_order([str(gene) for gene in genes])
    allowed_nodes = set(query_gene_ids)
    present_gene_ids = [gene for gene in query_gene_ids if gene in index.gene_ids]
    missing_gene_ids = [gene for gene in query_gene_ids if gene not in index.gene_ids]

    layer_payloads: list[dict[str, Any]] = []
    combined_edge_count = 0
    for layer_name in selected_layers:
        graph = index.layer_graphs[layer_name].subgraph(allowed_nodes)
        nodes = sorted(str(node) for node in graph.nodes())
        edges = _serialize_edges(graph, allowed_nodes)
        combined_edge_count += len(edges)
        layer_payloads.append(
            {
                "layer_name": layer_name,
                "present_gene_ids": nodes,
                "edges": edges,
                "edge_count": len(edges),
            }
        )

    payload = {
        "query_gene_ids": query_gene_ids,
        "present_gene_ids": present_gene_ids,
        "missing_gene_ids": missing_gene_ids,
        "layers": layer_payloads,
        "combined_edge_count": combined_edge_count,
    }
    provenance = {
        "tool_name": "induce_subgraph",
        "queried_layers": selected_layers,
    }
    return ToolExecutionResult(
        payload=payload,
        provenance=provenance,
        is_empty=combined_edge_count == 0,
    )


def shortest_path(
    index: MultiplexIndex,
    source: str,
    target: str,
    *,
    layer: str | None = None,
) -> ToolExecutionResult:
    """Return the shortest unweighted path between two genes."""

    if layer is None:
        graph = index.aggregate_graph
        search_mode = "aggregate_multiplex"
        queried_layers = list(index.layer_names)
    else:
        selected_layers = _resolve_layer_names(index, [layer])
        graph = index.layer_graphs[selected_layers[0]]
        search_mode = "single_layer"
        queried_layers = selected_layers

    if source not in graph or target not in graph or not nx.has_path(graph, source, target):
        payload = {
            "source_gene_id": source,
            "target_gene_id": target,
            "path_gene_ids": [],
            "hop_count": None,
            "layer_name": layer,
        }
        provenance = {
            "tool_name": "shortest_path",
            "search_mode": search_mode,
            "queried_layers": queried_layers,
            "distance_type": "unweighted_hops",
        }
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=True)

    path_gene_ids = [str(node) for node in nx.shortest_path(graph, source, target)]
    payload = {
        "source_gene_id": source,
        "target_gene_id": target,
        "path_gene_ids": path_gene_ids,
        "hop_count": len(path_gene_ids) - 1,
        "layer_name": layer,
    }
    provenance = {
        "tool_name": "shortest_path",
        "search_mode": search_mode,
        "queried_layers": queried_layers,
        "distance_type": "unweighted_hops",
    }
    return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=False)


def rwr_monoplex(
    index: MultiplexIndex,
    seeds: list[str],
    *,
    layer: str,
    top_k: int = DEFAULT_TOP_K,
    restart_probability: float = DEFAULT_RESTART_PROBABILITY,
) -> ToolExecutionResult:
    """Run a simple personalized PageRank on one graph layer."""

    selected_layers = _resolve_layer_names(index, [layer])
    graph = index.layer_graphs[selected_layers[0]]
    unique_seeds = _unique_preserving_order([str(seed) for seed in seeds])
    active_seeds = [seed for seed in unique_seeds if seed in graph]
    results = _personalized_pagerank(
        graph,
        unique_seeds,
        top_k=top_k,
        restart_probability=restart_probability,
    )

    payload = {
        "seed_gene_ids": unique_seeds,
        "active_seed_gene_ids": active_seeds,
        "layer_name": selected_layers[0],
        "top_k": top_k,
        "results": results,
    }
    provenance = {
        "tool_name": "rwr_monoplex",
        "layer_name": selected_layers[0],
        "algorithm": "personalized_pagerank",
        "restart_probability": restart_probability,
    }
    return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=len(results) == 0)


def rwr_multiplex(
    index: MultiplexIndex,
    seeds: list[str],
    *,
    top_k: int = DEFAULT_TOP_K,
    restart_probability: float = DEFAULT_RESTART_PROBABILITY,
) -> ToolExecutionResult:
    """Run a simple multiplex ranking by averaging layer-level PageRank scores."""

    unique_seeds = _unique_preserving_order([str(seed) for seed in seeds])
    layer_scores: dict[str, list[float]] = {}
    active_layers: list[str] = []
    active_seed_ids: set[str] = set()

    for layer_name in index.layer_names:
        graph = index.layer_graphs[layer_name]
        present_seeds = [seed for seed in unique_seeds if seed in graph]
        if not present_seeds:
            continue

        active_layers.append(layer_name)
        active_seed_ids.update(present_seeds)
        for result in _personalized_pagerank(
            graph,
            unique_seeds,
            top_k=max(top_k, len(graph.nodes())),
            restart_probability=restart_probability,
        ):
            layer_scores.setdefault(result["gene_id"], []).append(float(result["score"]))

    if not active_layers:
        payload = {
            "seed_gene_ids": unique_seeds,
            "active_seed_gene_ids": [],
            "active_layers": [],
            "top_k": top_k,
            "results": [],
        }
        provenance = {
            "tool_name": "rwr_multiplex",
            "algorithm": "mean_personalized_pagerank",
            "restart_probability": restart_probability,
            "active_layers": [],
        }
        return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=True)

    averaged = [
        {
            "gene_id": gene_id,
            "score": float(sum(scores) / len(scores)),
        }
        for gene_id, scores in layer_scores.items()
    ]
    averaged.sort(key=lambda item: (-item["score"], item["gene_id"]))
    results = averaged[:top_k]

    payload = {
        "seed_gene_ids": unique_seeds,
        "active_seed_gene_ids": sorted(active_seed_ids),
        "active_layers": active_layers,
        "top_k": top_k,
        "results": results,
    }
    provenance = {
        "tool_name": "rwr_multiplex",
        "algorithm": "mean_personalized_pagerank",
        "restart_probability": restart_probability,
        "active_layers": active_layers,
        "layer_count": len(active_layers),
    }
    return ToolExecutionResult(payload=payload, provenance=provenance, is_empty=len(results) == 0)


__all__ = [
    "DEFAULT_MYGENE_FIELDS",
    "DEFAULT_RESTART_PROBABILITY",
    "DEFAULT_TOP_K",
    "MultiplexIndex",
    "ToolExecutionError",
    "ToolExecutionResult",
    "build_multiplex_index",
    "get_neighbors",
    "induce_subgraph",
    "load_mygene_cache",
    "query_mygene",
    "rwr_monoplex",
    "rwr_multiplex",
    "save_mygene_cache",
    "shortest_path",
]
