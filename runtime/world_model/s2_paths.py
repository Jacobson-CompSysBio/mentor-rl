"""Return exact native shortest paths with canonical witnesses and full membership.

Paths use native edges without weights or a hop limit. Endpoint order remains
meaningful even when the selected layer is undirected.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from typing import Any

from .s2_multiplex import SelectionError
from .s2_neighbors import NeighborIndex
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


ARGUMENT_KEYS = ('multiplex_id', 'layer_id', 'gene_ids', 'direction')
DIRECTIONS = ('undirected', 'source_to_target')


class PathIndex:
    """Index native adjacency by ID so equal-length routes have one exact witness."""

    def __init__(self, view: GraphView, index: NeighborIndex | None = None) -> None:
        if index is not None and index.view is not view:
            raise ValueError('a shared neighbor index must belong to this exact immutable view')
        self.view = view
        self.native_index = index if index is not None else NeighborIndex(view)
        self.global_nodes = view.multiplex_nodes
        self._neighbors = {
            layer_id: {gene: tuple(sorted(neighbor for neighbor, _ in self.native_index.neighbors(
                layer_id, gene, 'out' if layer.directed else 'undirected')))
                for gene in layer.nodes}
            for layer_id, layer in view.layers.items()}

    @lru_cache(maxsize=32)
    def _tree(self, layer_id: str, source: str) -> tuple[dict[str, str | None], dict[str, int]]:
        """Keep a small cache of full BFS trees without a limit on search depth."""
        if source not in self._neighbors[layer_id]:
            return {}, {}
        parents, distances, pending = {source: None}, {source: 0}, deque([source])
        while pending:
            node = pending.popleft()
            for target in self._neighbors[layer_id][node]:
                if target not in parents:
                    parents[target] = node
                    distances[target] = distances[node] + 1
                    pending.append(target)
        return parents, distances

    def distances(self, layer_id: str, source: str) -> dict[str, int]:
        """Measure the complete attainable hop domain from one native source."""
        return dict(self._tree(layer_id, source)[1])

    def path(self, layer_id: str, source: str, target: str) -> list[str]:
        """Return the lexicographically first shortest native route, or no route."""
        native = self._neighbors[layer_id].get(source)
        if native is None or target not in self._neighbors[layer_id]:
            return []
        if source == target:
            return [source]
        if target in native:
            return [source, target]
        parents, _ = self._tree(layer_id, source)
        if target not in parents:
            return []
        route = []
        while target is not None:
            route.append(target)
            target = parents[target]
        return list(reversed(route))


def path_result(view: GraphView, arguments: Mapping[str, Any],
                missing_gene_ids: Sequence[str], not_in_layer_gene_ids: Sequence[str],
                path_gene_ids: Sequence[str] | None) -> dict[str, Any]:
    """Format a proved path using metadata and supplied facts without source reads."""
    endpoints = PathRuntime(view).validate_arguments(arguments)
    missing, outside = sorted(set(missing_gene_ids)), sorted(set(not_in_layer_gene_ids))
    if set(missing) & set(outside) or not set(missing + outside) <= set(endpoints):
        raise ValueError('path exclusions must be disjoint requested endpoint sets')
    route = None if path_gene_ids is None else list(path_gene_ids)
    if missing:
        if route is not None:
            raise ValueError('a missing multiplex endpoint leaves traversal unevaluated')
    elif route is None:
        raise ValueError('a checked path result requires a route list')
    elif outside and route:
        raise ValueError('a layer-excluded endpoint cannot have a layer path')
    elif not outside and endpoints[0] == endpoints[1] and route != [endpoints[0]]:
        raise ValueError('a repeated layer member has an exact zero-hop path')
    elif route and (route[0] != endpoints[0] or route[-1] != endpoints[1]
                    or len(set(route)) != len(route)
                    or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene) for gene in route)):
        raise ValueError('a shortest witness must retain ordered endpoints and distinct canonical IDs')
    return {'multiplex_id': arguments['multiplex_id'], 'layer_id': arguments['layer_id'],
        'directed': view.layers[arguments['layer_id']].directed, 'direction': arguments['direction'],
        'gene_ids': list(endpoints), 'path_found': None if route is None else bool(route),
        'path_gene_ids': route, 'hop_count': len(route) - 1 if route else None,
        'missing_gene_ids': missing, 'not_in_layer_gene_ids': outside}


class PathRuntime:
    """Serve native unweighted paths and distinguish checked absence from blocked search."""

    def __init__(self, view: GraphView, *, index: PathIndex | None = None) -> None:
        if index is not None and index.view is not view:
            raise ValueError('a shared path index must belong to this exact immutable view')
        self.view, self.index = view, index

    def arguments(self, gene_ids: Any, layer_id: Any, direction: Any) -> dict[str, Any]:
        """Copy all four arguments without a change to either endpoint position."""
        return {'multiplex_id': self.view.multiplex_id, 'layer_id': deepcopy(layer_id),
                'gene_ids': deepcopy(gene_ids), 'direction': deepcopy(direction)}

    def validate_arguments(self, arguments: Mapping[str, Any]) -> tuple[str, str]:
        """Check the ordered pair and native direction from manifest metadata only."""
        if set(arguments) != set(ARGUMENT_KEYS):
            raise SelectionError('invalid_input', 'INVALID_SELECTION')
        if not arguments['multiplex_id'] or not arguments['layer_id']:
            raise SelectionError('invalid_input', 'MISSING_SCOPE')
        if arguments['multiplex_id'] != self.view.multiplex_id:
            raise SelectionError('not_found', 'UNKNOWN_MULTIPLEX')
        layer_id = arguments['layer_id']
        if not isinstance(layer_id, str) or layer_id not in self.view.layers:
            raise SelectionError('not_found', 'UNKNOWN_LAYER')
        genes = arguments['gene_ids']
        if (not isinstance(genes, list) or len(genes) != 2
                or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene) for gene in genes)):
            raise SelectionError('invalid_input', 'INVALID_PAIR')
        direction, directed = arguments['direction'], self.view.layers[layer_id].directed
        if direction is None and directed:
            raise SelectionError('ambiguous', 'AMBIGUOUS_PATH_DIRECTION')
        if not isinstance(direction, str) or direction not in DIRECTIONS:
            raise SelectionError('invalid_input', 'INVALID_DIRECTION')
        if directed != (direction == 'source_to_target'):
            raise SelectionError('invalid_input', 'DIRECTION_UNAVAILABLE')
        return tuple(genes)

    def shortest_path(self, multiplex_id: Any, layer_id: Any, gene_ids: Any,
                      direction: Any, *, evidence_id: str) -> dict[str, Any]:
        """Check both memberships, then search every reachable native vertex as needed."""
        arguments = dict(zip(ARGUMENT_KEYS, (multiplex_id, layer_id, gene_ids, direction)))
        try:
            requested = self.validate_arguments(arguments)
        except SelectionError as error:
            return {'status': error.status, 'provenance': {'registry': [], 'graph': []},
                    'result': {'issue_code': error.issue_code, 'arguments': deepcopy(arguments)}}
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError('a successful acquisition requires a nonempty evidence ID')
        missing = sorted(set(requested) - self.view.multiplex_nodes)
        outside = sorted((set(requested) & self.view.multiplex_nodes) - self.view.layers[layer_id].nodes)
        route = None if missing else []
        if not missing and not outside:
            if requested[0] == requested[1]:
                route = [requested[0]]
            else:
                if self.index is None:
                    self.index = PathIndex(self.view)
                route = self.index.path(layer_id, *requested)
        return {'status': 'not_found' if missing else 'resolved',
            'provenance': {'registry': [], 'graph': [{'multiplex_id': self.view.multiplex_id,
                'graph_version': self.view.graph_version,
                'layer_scope': {'kind': 'layers', 'layer_ids': [layer_id], 'layer_family': None},
                'evidence_id': evidence_id}]},
            'result': path_result(self.view, arguments, missing, outside, route)}
