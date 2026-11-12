"""Project finite gene inputs onto native layer membership and edge records.

Pages divide canonical edge rows. Every successful page retains the complete
input partition, including selected members without an induced incident edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any

from .s2_multiplex import SelectionError
from .s2_neighbors import NeighborIndex, _selection
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView


ARGUMENT_KEYS = ('multiplex_id', 'layer_id', 'gene_ids', 'mode', 'top_k', 'cursor')


class SubgraphIndex:
    """Use full native inventories and adjacency without a quadratic pair array."""

    def __init__(self, view: GraphView, index: NeighborIndex | None = None) -> None:
        if index is not None and index.view is not view:
            raise ValueError('a shared neighbor index must belong to this exact immutable view')
        self.view = view
        self.native_index = index if index is not None else NeighborIndex(view)
        self.global_nodes = view.multiplex_nodes

    def partition(self, layer_id: str, gene_ids: Sequence[str]) -> dict[str, list[str]]:
        """Classify unique inputs with authoritative endpoint inventories."""
        genes, nodes = set(gene_ids), self.view.layers[layer_id].nodes
        return {'layer_gene_ids': sorted(genes & nodes),
                'not_in_layer_gene_ids': sorted((genes & self.global_nodes) - nodes),
                'missing_gene_ids': sorted(genes - self.global_nodes)}

    def edges(self, layer_id: str, gene_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Keep each exact native edge whose two endpoints belong to the input."""
        layer = self.view.layers[layer_id]
        selected = set(gene_ids) & layer.nodes
        direction = 'out' if layer.directed else 'undirected'
        rows = {}
        for source in selected:
            for target, weight in self.native_index.neighbors(layer_id, source, direction):
                if target in selected:
                    pair = (source, target) if layer.directed else tuple(sorted((source, target)))
                    rows[pair] = weight
        return [{'gene_ids': list(pair), 'weight': rows[pair]} for pair in sorted(rows)]


class SubgraphRuntime:
    """Acquire canonical edge pages with complete input membership coverage."""

    def __init__(self, view: GraphView, *, index: SubgraphIndex | None = None,
                 page_size: int | None = None, disclose_total: bool = True,
                 overlap: int = 0, empty_page_at: int | None = None) -> None:
        if index is not None and index.view is not view:
            raise ValueError('a shared subgraph index must belong to this exact immutable view')
        if page_size is not None and (type(page_size) is not int or page_size < 1):
            raise ValueError('page_size must be a positive integer or null')
        if type(disclose_total) is not bool:
            raise ValueError('disclose_total must be a boolean')
        if type(overlap) is not int or overlap < 0 or overlap and (page_size is None or overlap >= page_size):
            raise ValueError('overlap must be nonnegative and smaller than page_size')
        if empty_page_at is not None and (type(empty_page_at) is not int or empty_page_at < 0):
            raise ValueError('empty_page_at must be a nonnegative page ordinal or null')
        self.view, self.index = view, index
        self.transport = {'page_size': page_size, 'disclose_total': disclose_total,
                          'overlap': overlap, 'empty_page_at': empty_page_at}
        self._schedules = {}

    def arguments(self, gene_ids: Any, layer_id: Any, mode: str = 'complete',
                  top_k: int | None = None, cursor: str | None = None) -> dict[str, Any]:
        """Copy the explicit six-field selection without an implicit bound."""
        return {'multiplex_id': self.view.multiplex_id, 'layer_id': deepcopy(layer_id),
                'gene_ids': deepcopy(gene_ids), 'mode': mode, 'top_k': top_k, 'cursor': cursor}

    def validate_arguments(self, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        """Validate syntax from manifest metadata without source inventory access."""
        source = {key: arguments.get(key) for key in ('multiplex_id', 'layer_id', 'gene_ids')}
        requested = NodesByLayerRuntime(self.view)._validate(source)
        if set(arguments) != set(ARGUMENT_KEYS):
            raise SelectionError('invalid_input', 'INVALID_SELECTION')
        try:
            _selection(arguments['mode'], arguments['top_k'])
        except SelectionError as error:
            raise SelectionError('invalid_input', 'INVALID_SELECTION') from error
        cursor = arguments['cursor']
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise SelectionError('invalid_input', 'INVALID_CURSOR')
        return requested

    def _schedule(self, arguments: Mapping, requested: Sequence[str], size: int):
        """Bind opaque cursors to the ordered input, selection, version, and pages."""
        selection = {key: deepcopy(arguments[key]) for key in ARGUMENT_KEYS if key != 'cursor'}
        selection['gene_ids'] = list(requested)
        identity = json.dumps([self.view.graph_version, selection, self.transport], sort_keys=True, separators=(',', ':'))
        if identity not in self._schedules:
            pages, offset = [], 0
            page_size = self.transport['page_size'] or max(size, 1)
            if size == 0:
                pages.append((0, 0))
            while offset < size:
                if len(pages) == self.transport['empty_page_at']:
                    pages.append((offset, offset))
                end = min(offset + page_size, size)
                pages.append((offset, end))
                if end == size:
                    break
                offset = end - self.transport['overlap']
            cursors = {None: 0}
            for ordinal in range(1, len(pages)):
                token = 's29_' + hashlib.sha256((identity + ':' + str(ordinal)).encode()).hexdigest()
                cursors[token] = ordinal
            self._schedules[identity] = pages, cursors
        return self._schedules[identity]

    def induce_subgraph(self, multiplex_id: Any, layer_id: Any, gene_ids: Any,
                        mode: Any, top_k: Any, cursor: Any, *, evidence_id: str) -> dict[str, Any]:
        """Return one native edge page and all three requested membership groups."""
        arguments = dict(zip(ARGUMENT_KEYS, (multiplex_id, layer_id, gene_ids, mode, top_k, cursor)))
        try:
            requested = self.validate_arguments(arguments)
            if requested:
                if self.index is None:
                    self.index = SubgraphIndex(self.view)
                partition = self.index.partition(layer_id, requested)
                edges = self.index.edges(layer_id, partition['layer_gene_ids'])
            else:
                partition = {'layer_gene_ids': [], 'not_in_layer_gene_ids': [], 'missing_gene_ids': []}
                edges = []
            selected_size = len(edges) if mode == 'complete' else min(top_k, len(edges))
            pages, cursors = self._schedule(arguments, requested, selected_size)
            if cursor not in cursors:
                raise SelectionError('invalid_input', 'INVALID_CURSOR')
        except SelectionError as error:
            return {'status': error.status, 'provenance': {'registry': [], 'graph': []},
                    'result': {'issue_code': error.issue_code, 'arguments': deepcopy(arguments)}}
        if requested and (not isinstance(evidence_id, str) or not evidence_id.strip()):
            raise ValueError('a successful acquisition requires a nonempty evidence ID')
        ordinal = cursors[cursor]
        start, end = pages[ordinal]
        page = edges[start:end]
        next_cursor = next((token for token, number in cursors.items() if number == ordinal + 1), None)
        provenance = {'registry': [], 'graph': []}
        if requested:
            provenance['graph'].append({'multiplex_id': self.view.multiplex_id,
                'graph_version': self.view.graph_version,
                'layer_scope': {'kind': 'layers', 'layer_ids': [layer_id], 'layer_family': None},
                'evidence_id': evidence_id})
        return {'status': 'resolved', 'provenance': provenance,
            'result': {'multiplex_id': multiplex_id, 'layer_id': layer_id, 'directed': self.view.layers[layer_id].directed,
                'layer_gene_ids': partition['layer_gene_ids'], 'layer_gene_count': len(partition['layer_gene_ids']),
                'not_in_layer_gene_ids': partition['not_in_layer_gene_ids'], 'missing_gene_ids': partition['missing_gene_ids'],
                'edges': deepcopy(page), 'edge_count': len(page),
                'coverage': {'operation': 'layer_membership', 'requested_gene_ids': list(requested),
                    'checked_gene_ids': list(requested), 'unchecked_gene_ids': []},
                'collection': {'mode': mode, 'returned_count': len(page),
                    'total_count': len(edges) if self.transport['disclose_total'] or not edges else None,
                    'complete': start == 0 and end == len(edges), 'next_cursor': next_cursor}}}
